# Streamed editable voxel terrain — system design

**Status**: design (2026-05-18). No code yet. V1 plan ready to execute when prioritised.

This document is the authoritative design for the arvx tiling terrain
system. It was written across one extended design conversation that
worked from the loosest framing ("we need a tiling terrain system") down
to the specific data structures, policies, and phasing. Where forks
existed, the chosen path is recorded with the rationale.

The design's load-bearing constraint: **a terrain tile is just an arvx
asset under streaming control.** The renderer, sculpt stack, paint
stack, glass, CSM, user-shaders, and collider worker are all reused
unchanged. The terrain system's only job is: *where do tiles come from,
when do they live and die, and how do their edges stay watertight.*

---

## Design inputs (locked)

The system is targeting the maximally-ambitious quadrant on every axis:

| Axis | Choice |
|---|---|
| World scale | Open world, streamed (runtime capability); bounded grid by default (authoring UX) |
| Default extent | 16 × 16 × 4 tiles = 1024 m × 1024 m × 256 m (resizable, unbounded opt-in) |
| Voxel size | Drawn from the **unified** `arvx_core::constants::RESOLUTION_TIERS` (power-of-2 fractions of 1 m, 2× ratios). Terrain default: Tier 2 = 0.25 m. LOD pyramid walks one tier COARSER per LOD level (lower index). |
| Octree alignment | `voxelize_octree` requires **pow2-cubic-aligned AABB**. Terrain tile AABBs satisfy this by construction (64 m / pow2-fraction-of-1 m = pow2). No hidden internal padding anywhere. |
| Source data | Three layers: Base `TerrainFn` (procedural) + Stamps (local SDF features) + Sculpt (per-tile baked edits) |
| Editability | Fully editable, runtime + editor |
| Seam strategy | Shared boundary voxels (watertight) |
| Tile shape | 3D cubic tiles (caves / overhangs / floating geometry all natural) |
| Tile footprint | 64 m on a side |
| LOD strategy | Tile quadtree with merged coarse tiles (V2) — V1 ships uniform fine LOD with API stub |

The "ambitious quadrant" choice means the system is genuinely game-grade
on day one in scope, but it ships behind a careful phasing that earns
each capability by demonstrating the simpler version first.

---

## Architecture

### One sentence

A new `arvx-terrain` crate manages a sparse 3D tile-octree of arvx
assets that materialise from a user-supplied `TerrainFn`, share a
1-voxel halo with their neighbours for watertight surface-nets seams,
accept the existing sculpt/paint kernel without modification, and emit
Rapier TriMesh colliders via the existing `collider_worker` under
consumer-supplied policy traits.

### Crate boundary

```
arvx-terrain  (new)
  depends on:
    arvx-core         — sparse octree, brick pool, LeafAttr
    arvx-import       — voxelizer (TerrainFn → octree), mesh extract, cluster DAG bake
    rkf-physics       — Rapier integration, collider_worker
  used by:
    arvx-runtime      — frame scheduling, streamer tick, ECS glue
    arvx-engine       — command handling for terrain edits
```

Critically: **no dependency on `arvx-render`, and no new code in
`arvx-render`.** Tiles flow through the renderer as ordinary arvx asset
instances.

### Core types

```rust
/// Node in the sparse 3D tile-octree. `level = 0` is the fine LOD.
/// V1 only allocates level=0 tiles; the field exists so V2 can
/// add coarse tiles without an API break.
pub struct TileKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// Procedural terrain source. Implementations are user-supplied.
/// Sampled in tile-local coords so noise lookups can seed on the
/// integer key — no FP drift at large world coords.
pub trait TerrainFn: Send + Sync {
    fn sample(
        &self,
        tile: TileKey,
        local: Vec3,         // in tile-local meters, 0..tile_size
        voxel_size_m: f32,
    ) -> TerrainSample;
}

pub struct TerrainSample {
    pub sd: f32,              // signed distance to surface (positive = outside)
    pub primary_mat: MaterialId,
    pub secondary_mat: MaterialId,
    pub blend: f32,           // 0..1
}

/// A materialised terrain tile = an arvx asset + halo + lifecycle state.
pub struct TerrainTile {
    pub key: TileKey,
    pub voxel_size_m: f32,            // derived from level + tile_size
    pub octree: SparseOctree,         // arvx-core types
    pub brick_pool_slot: BrickSlot,
    pub mesh: BakedMesh,              // surface-nets output
    pub clusters: ClusterArray,
    pub dag: ClusterDag,
    pub halo: [HaloFace; 6],          // -X, +X, -Y, +Y, -Z, +Z
    pub state: TileState,
    pub persisted: bool,              // true => `.arvxtile` exists on disk
}

pub enum TileState {
    Unmaterialised,           // key exists in streamer, geometry not built
    Materialising,            // worker thread is building it
    Live,                     // ready for render + physics
    Dirty(AABB),              // edit applied since last mesh; AABB is the dirty region
    Persisting,               // mid-flight write of `.arvxtile`
}

pub struct TileStreamer { /* sparse 3D tile-octree of slots */ }
pub struct HaloManager   { /* routes boundary edits to up-to-3 neighbours */ }
```

---

## Authoring at a glance — a 5-minute session

Concrete picture of what using this feels like. The bounded-grid default
makes the scope of the world visible from the first click.

**Step 1 — Drop a Terrain node.** Scene tree gains a `Terrain` node. The
viewport shows a flat 1024 × 1024 m grass plane bordered by sky on all
sides past the grid boundary; a faint wireframe shows the 16 × 16 tile
cells.

**Step 2 — Pick a procedural source.** Inspector → Source → Function:
"FBM heightmap." Seed, octaves, scale, sea-level fields appear. As you
tweak, the 256 tiles within bounds re-bake on the worker — hills
appear in waves over ~1-2 seconds on a strong machine.

**Step 3 — Place a Mountain stamp.** Pick the Mountain stamp from the
viewport toolbar (or the Stamps section of the Inspector), click in
the viewport. A `Mountain_01` node appears under `Terrain ▸ Stamps`.
Tiles whose AABB intersects the stamp footprint re-bake; a peak rises.
Drag the gizmo to move it; the mountain follows, tiles re-bake live.

**Step 4 — Place a Lake and a Flatten.** Same workflow. Lake uses
Smooth-Min so it carves a depression with smooth shores. Flatten uses
Replace so it overrides the procedural shape entirely within its
radius — useful for building footprints, roads, plazas.

**Step 5 — Sculpt detail.** Pick the Sculpt brush from the viewport
toolbar. Drag on the mountain to carve a ledge, raise a small
overhang. Each stroke re-meshes touched tiles. Touched tiles persist
as `.arvxtile` files on save. The base TerrainFn + stamps are
unaffected — sculpt is a layer on top.

**Step 6 — Save.** `File → Save scene`. The scene file stores: the
TerrainFn type and params, the world bounds, and the stamps list
(small data, ~hundreds of bytes per stamp). The `<scene>/tiles/`
directory gets `.arvxtile` files only for the tiles you actually
sculpted. Reopening the scene regenerates the rest from the
deterministic base + stamps.

---

## The three-layer source model

The shape of the world at any point is the composition of three layers,
each authored differently, each invalidating different sets of tiles
when edited:

```
┌───────────────────────────────────────────────────────┐
│ Layer 3: Sculpt edits      (per tile, persisted)      │
│          Authored with brushes → `.arvxtile`          │
│          Layered ON TOP of the voxelised baseline     │
├───────────────────────────────────────────────────────┤
│ Layer 2: Stamps            (scene data, spatial idx)  │
│          Local SDF features (mountains, lakes, etc.)  │
│          Authored as scene-tree objects               │
│          Deterministic; move/delete → tiles re-bake   │
├───────────────────────────────────────────────────────┤
│ Layer 1: Base TerrainFn    (code or graph)            │
│          Global noise / ridge / warp                  │
│          + biome regions (Sphere/Box/OBB, membership- │
│            weighted blend; see docs/REGIONS.md)       │
│          Erosion (V2) bakes its output INTO this layer│
└───────────────────────────────────────────────────────┘
```

**Sampling a leaf at world position P** during voxelisation:

1. Evaluate Layer 1 → base `TerrainSample`.
2. Query Layer 2 spatial index for stamps overlapping P → combine via
   the stamp's chosen op (smooth-min, smooth-max, replace, add,
   subtract).
3. The result is what gets written to the leaf's SDF cell. Layer 3
   (sculpt) mutates the voxel result *after* meshing, never feeding
   back into Layers 1-2.

The three layers behave totally differently and that's load-bearing:

| Layer | Authored via | Storage | Edit frequency | Re-bake scope on change |
|---|---|---|---|---|
| 1 Base | Code (V1) / node graph (V2) | Code or graph blob | Rare (authoring setup) | Every loaded tile |
| 2 Stamps | Place + tweak in viewport | Per-stamp record in scene file | Often during authoring | Tiles intersecting stamp's AABB |
| 3 Sculpt | Brushes in viewport | Per-tile `.arvxtile` | Constant, drag-stroke speed | Just the touched tile |

This is why "re-bake everything" (Layer 1 change) is a confirmation
operation, "move a stamp" (Layer 2) is interactive, and "sculpt a
stroke" (Layer 3) is real-time.

---

## World bounds and the authoring model

A Terrain is **bounded by default**: a fixed grid of tiles with a
defined extent that the author can see and reason about. This matches
the mental model of Unity Terrain, Unreal Landscape, Godot Terrain,
and World Machine-style tools. The streaming machinery underneath is
unchanged — the streamer simply doesn't materialise tiles outside the
bounds.

```rust
pub struct Terrain {
    pub bounds: TerrainBounds,
    pub source: Box<dyn TerrainFn>,
    pub stamps: SpatialIndex<StampHandle>,
    pub edit_set: HashSet<TileKey>,   // tiles with .arvxtile on disk
}

pub enum TerrainBounds {
    /// Fixed grid of tiles. The default.
    /// Default extent: 16 × 16 × 4 tiles = 1024 × 1024 × 256 m.
    Bounded {
        origin: TileKey,           // bottom-corner tile (inclusive)
        extent: (u32, u32, u32),   // size in tiles along x, y, z
    },
    /// Infinite world; streamer materialises tiles around the camera
    /// indefinitely. Opt-in only — typically only for true open-world
    /// procedural games. Most projects want Bounded.
    Unbounded,
}
```

**Inspector bounds controls:**

```
Bounds
  Mode:    ▼ Bounded
  Origin:  (0, 0, 0)         ← in tile coordinates
  Extent:  16 × 16 × 4
          = 1024 m × 1024 m × 256 m
  [ Resize… ]   [ Switch to Unbounded ]
```

**Viewport feedback:** a wireframe box overlay shows the bounds. Tiles
outside the bounds render as sky / void. Walking the camera past the
edge shows the world ending cleanly at a visible boundary.

**Resizing:** extending bounds materialises new tiles from the
TerrainFn + stamps (no edit data loss). Shrinking bounds prompts
before deleting `.arvxtile` files that would fall outside the new
extent.

**Unbounded mode** is the strict superset — same TerrainFn, same
stamps, no bounds clamp. It's an opt-in toggle for projects that
actually want infinite procedural worlds. The runtime cost is the
same; the difference is purely whether the streamer has an outer
clamp.

---

## Stamps

Stamps are local SDF features placed in the scene as authored objects.
Each stamp is a small data record (type, transform, parameters,
combine op) that gets queried at voxelisation time by tiles whose
AABB intersects its footprint.

```rust
pub struct Stamp {
    pub kind: StampKind,
    pub transform: Transform,         // position, rotation, scale
    pub footprint_radius: f32,        // for spatial indexing
    pub op: StampOp,                  // chosen per instance
    pub material_override: Option<MaterialRule>,
    pub params: StampParams,          // type-specific
}

pub enum StampKind {
    Mountain { peak_height: f32, base_radius: f32, ridged_octaves: u8 },
    Lake     { depth: f32, shore_smoothness: f32 },
    Plateau  { top_height: f32, top_radius: f32, cliff_steepness: f32 },
    Flatten  { plane_height: f32, falloff: f32 },
    Hill     { peak_height: f32, base_radius: f32 },
    // V2: River (polyline channel), CaveEntry (subtractive volume).
}

pub enum StampOp {
    Add,         // signed-distance add (raises terrain)
    Subtract,    // signed-distance subtract (carves)
    SmoothMin { k: f32 },   // blended union, k controls smoothness
    SmoothMax { k: f32 },   // blended intersection
    Replace,    // overrides base entirely within footprint
}
```

**Spatial indexing:** the `SpatialIndex<StampHandle>` is a sparse
grid (or R-tree if profiling demands) keyed on tile-sized cells. A
voxelising tile queries the index for stamps within
`tile_aabb + max_stamp_footprint`. Typical tile sees 0-3 stamps; the
combine loop is tiny.

**Move/delete invalidation:** a stamp records its previous-frame
AABB. When it moves, the union of (old AABB ∪ new AABB) → set of
intersecting tiles → marked dirty for re-bake. Deleting a stamp
marks its last AABB dirty. Adding marks its new AABB dirty.

**Material handling:** each stamp can optionally override the
biome material rule within its footprint (e.g. Mountain stamp →
force rock above slope 30° even if the base Material rule said
grass). If `material_override` is `None`, the base TerrainFn's
biome rules apply normally.

**Stamps live in the scene tree** under a `Stamps` group node:

```
Scene
└── Terrain
    ├── Stamps
    │   ├── Mountain_01
    │   ├── Lake_01
    │   ├── BuildingPlot_flatten
    │   └── …
    └── (terrain-level settings)
```

This gives stamps normal scene-tree affordances: select, multi-select,
rename, group, copy/paste, undo/redo, transform gizmo. The Inspector
shows stamp params when a stamp is selected.

**V1 stamp library:** Mountain, Lake, Plateau, Flatten, Hill. Covers
the common terrain-shaping idioms. River and CaveEntry deferred to
V2 (polyline-defined and explicitly volumetric, respectively — both
have more involved SDF definitions).

---

## The two LOD pyramids

There are **two stacked LOD systems** and the clean mental model is to
keep them strictly orthogonal:

1. **Inside a tile** — the existing per-asset cluster DAG, shipping since
   2026-05-06 (see `docs/PERF_DEBT.md`, [[project-mesh-phase5-shipped]]).
   Gives smooth, sub-pixel transitions on a single tile's geometry.
   Already in production. Terrain inherits it for free.

2. **Across tiles** — the tile-octree itself. Camera-near regions are
   covered by level-0 64³ m tiles at the fine voxel size. Camera-mid by
   level-1 128³ m tiles at 2× cell size (one coarse tile covers eight
   fine tiles). Camera-far by level-2 256³ m at 4× cell size. And so on.
   Each level is its own voxelisation of the same `TerrainFn`.

These compose: a coarse tile renders its own internal DAG, which gives
sub-tile transitions; the streamer decides which level's tile is *live*
in each spatial region.

**V1 only allocates level 0.** The `TileKey.level` field is in the API
from day one so V2 can extend without breaking on-disk format or any
caller code.

---

## The genuinely hard problems

These dominate the schedule and are called out explicitly so they don't
get papered over:

### Cross-LOD seams (V2)

A level-0 tile abutting a level-1 tile has 2× cells on its side of the
boundary. Surface-nets vertices don't line up → visible cracks. Three
industrial solutions:

- **Transvoxel-style transition cells** — special triangulation tables
  for cells that straddle a LOD boundary. Correct, expensive to
  implement, the "real" answer.
- **Fine-ring policy** — force a ring of fine tiles around every fine
  tile, so transitions only happen between same-LOD tiles. Wastes a
  ring's worth of memory; dead simple; the V2 starting point.
- **Skirts** — each tile renders a downward skirt at its edges that
  covers the gap. Cheap, can look bad on horizon silhouettes.

V2 ships fine-ring; Transvoxel is V3 if cracks become visible at
realistic view distances.

### Edit propagation up the LOD pyramid (V2)

A sculpt stroke on a level-0 tile invalidates the level-1 tile that
covers it (and level-2, etc.). The coarse tile must re-voxelise *applying
a downsampled form of the edit*. With our chosen edit representation
(full baked tile), the propagation works by:

1. Voxelising the coarse tile from `TerrainFn` (its normal procedural
   baseline).
2. Computing the fine tile's diff on demand: re-sample `TerrainFn` at
   fine resolution and compare against the baked fine tile's octree.
3. Downsampling that diff into the coarse tile's cells.

Cost is per-LOD-rebuild on edited tiles, not per-frame. Doable.

### Halo correctness under streaming

Tile A is loaded, neighbour B isn't yet. A's halo is stale until B
materialises; when B arrives, A's boundary cells re-mesh. Explicit
"halo dirty" tracking must survive async neighbour materialisation. The
`HaloManager` owns this state — when B finishes materialising, it
notifies up to 6 neighbours that their corresponding halo face needs
refresh.

### Procedural fn at multiple voxel sizes (V2)

Naïve sampling at 1 m doesn't equal the average of 64 samples at
0.25 m — coarse tiles will look subtly different from fine tiles even
at the same spot. Either accept that (transitions hidden by view
distance) or pre-filter the noise per level (Mip-style; cheap for
heightmaps, expensive for live noise). V2 decision; can ship either.

---

## Edit persistence

**Match the existing sculpt system exactly: mutate the octree in place,
serialise to `.arvxtile` when touched, regenerate from `TerrainFn` when
not.**

The arvx sculpt system has converged on mutate-in-place after an early
attempt at a sparse `SculptOverlay` delta (see
[[project-sculpt-phase-a-overlay-plan]]) was retired as an architectural
dead end. The terrain system reuses that pattern unchanged:

- A brush stroke produces `LeafEditOp { Add / Remove / Paint }` events
  that go straight into the tile's brick pool.
- Surface-nets re-extracts the affected clusters.
- The cluster DAG re-bakes the touched chains.
- Save = serialise the current octree state to `.arvxtile`.

There is no overlay, no replay log, no diff machinery. The octree IS
the source of truth.

A `.arvxtile` is just an `.arvx` with a tile-tag header. It is written
to `<scene>/tiles/<level>_<x>_<y>_<z>.arvxtile` on first persist and
re-read on subsequent loads. Untouched tiles never hit disk; they
regenerate deterministically from `TerrainFn(tile_key, …)`.

---

## Watertight seams via halo

Each tile carries a 1-voxel halo replicated from its 6 neighbours:

```
   [ ── face B's interior boundary ──]    ← neighbour B
       │
       │  halo edge (1 voxel deep)
       ▼
   [ ── face A's halo face from B ──]    ← tile A's halo
   [ ── face A's interior boundary ──]    ← tile A
```

Mesh extract runs on `tile A + halo`. The cells in A's halo region have
the same voxel values as B's interior boundary cells (because the halo
is replicated, not derived). Surface-nets produces vertices at the same
world positions on both sides → watertight.

**Edit propagation:** when a sculpt op touches a leaf within 1 voxel of
a tile boundary, the affected neighbour's halo for that face is marked
dirty and the neighbour re-meshes. An edit at a corner can dirty up to
3 neighbours (the 3 faces meeting at that corner).

**Boundary arithmetic in integer space:** the shared boundary plane
between tile (i, j, k) and (i+1, j, k) is computed in integer tile-key
space and only converted to local f32 *inside* each tile. Doing it in
f32 world coords at large coords produces divergent values on either
side and breaks watertightness. See "Floating-point handling" below.

---

## Floating-point handling

**Principle: integers all the way down to a tile boundary. f32 only
inside a tile, never crossing one.**

| Surface | Drift-safe treatment |
|---|---|
| Tile addressing | `TileKey { x, y, z: i32 }` × `tile_size_m` → ±137 billion km range. Never f32. |
| Per-tile geometry | Vertices, leaves, halo all in tile-local f32, range `0..tile_size_m`. Excellent precision regardless of world position. |
| Rendering | Camera-relative model matrix: compute `(tile_origin - camera_origin)` at i64/f64, demote to f32 once, build matrix. Translation values are always small → no vertex shimmer. |
| Physics | Rapier is f32 internally. Periodic origin rebase wrapper recenters all bodies + camera when camera moves beyond a threshold from physics origin. Terrain re-registers tile colliders on rebase event. |
| `TerrainFn` sampling | `fn sample(tile: TileKey, local: Vec3, voxel_size_m: f32)`. Integer tile key seeds noise; local f32 stays bounded. No call site ever passes a world-space f32 into a noise lookup. |
| Halo seams | Boundary plane computed in `TileKey` integer space; converted to local f32 inside each tile only. Watertight at any world coord. |
| Sculpt brush | Brush position is `WorldPosition`. Per-tile decomposition: `brush_in_tile_local = (brush_world - tile_origin)` at i64 precision, demoted to f32 *inside* the tile's frame. |
| `.arvxtile` persistence | Stores `TileKey` (integers) and tile-local geometry. Reloading at a different floating origin places the tile exactly. Never stores world-space f32. |

**Open question for Phase 1:** what's `WorldPosition`'s actual
representation in `rkf-core`? (sector + local? fixed-point i64? f64?)
This determines the exact `TileKey ↔ WorldPosition` conversion shape
and where the integer/f32 boundary sits. 30-second code read when
Phase 1 starts. Do not guess.

---

## Physics integration

Terrain owns the *what* (per-tile trimesh colliders, rebuild
scheduling, seam continuity). Consumers own the *when* (which regions
need colliders, when to rebuild during a drag, what motion to predict
ahead of). This split is deliberate: the user has said "we can't
predict who's standing on this terrain" — so the terrain system must
not bake in assumptions about consumers.

### What terrain provides (mechanism)

- **Per-tile Rapier `TriMesh` collider** sourced from the tile's LOD-0
  cluster triangles via the existing `collider_worker` (shipped
  2026-05-14, `9cd5f3cc`). Static rigid body; shape swap via `Arc`
  replacement on the main thread.
- **Seam continuity for free:** halo guarantees adjacent tiles produce
  identical surface-nets vertices at boundaries, so adjacent tile
  trimeshes meet exactly. No skirts, no overlap geometry, no
  character-snagging-on-seams.
- **Wake-on-rebuild:** after every collider swap, query Rapier for
  sleeping bodies whose AABB intersects the rebuilt region and wake
  them. This is correct for any consumer (KCC, dynamic body, vehicle,
  whatever) so it lives in terrain, not in policy.
- **Tile lifecycle events:** `TileColliderBuilt { key, aabb }`,
  `TileColliderDestroyed { key }`, `TileRegionRebuilt { key, dirty_aabb }`
  for consumer subscription.

### What consumers provide (policy traits)

```rust
pub trait ColliderResidencyPolicy {
    /// Given streamer state + consumer-supplied interest regions
    /// (positions, AABBs, velocity rays, whatever the consumer cares
    /// about), returns the set of tiles that should have live colliders.
    fn residency(&self, ctx: &ResidencyContext) -> TileSet;
}

pub trait EditRebuildPolicy {
    /// Given a tile's accumulated dirty AABB and time/edits since last
    /// rebuild, decides whether to rebuild now, wait, or defer to
    /// stroke release.
    fn decide(&self, ctx: &RebuildContext) -> RebuildDecision;
}

pub trait PredictiveMaterializationPolicy {
    /// Given consumer-supplied trajectory hints (e.g., velocity rays
    /// from active bodies), returns unmaterialised tiles to prioritise
    /// on the worker.
    fn prioritise(&self, ctx: &TrajectoryContext) -> TileSet;
}

pub enum RebuildDecision { Rebuild, Wait, DeferredOnRelease }
```

**Editor V1 defaults:** simple radius around editor camera + any
spawned bodies (residency), `OnStrokeRelease` (rebuild), no-op
(predictive). A future in-game tunnelling system swaps in
`Debounced { interval_ms: 50 }` and a velocity-ray residency policy
without touching terrain code.

### Awkward interactions called out

- **Edit-under-a-body.** Carve a hole under a sleeping body → wake the
  body on collider swap. Handled by the wake-on-rebuild mechanism;
  consumer doesn't see this.
- **Carving while standing on it.** *Carve a hole under your feet:*
  consumer's character system handles via ground-check (if KCC) or
  automatic wake (if dynamic). *Raise terrain under your feet:* needs
  consumer's update to run *after* terrain's collider swap that frame.
  Ordering is a consumer concern; terrain emits events that consumers
  can sequence against.
- **Drag-stroke collider thrash.** A drag fires N edits/sec. The
  `EditRebuildPolicy` chooses cadence: editor uses `OnStrokeRelease`,
  an in-game tunnelling system might use `Debounced { 50ms }`. Visual
  mesh can rebuild more aggressively (renderer is already async); the
  collider can lag a frame or three with no visible cost.

### V2 LOD pyramid implication

**Only level-0 tiles carry colliders.** Coarse tiles are pure render.
The physics residency radius is by definition smaller than the radius
at which we'd ever LOD down a tile. Falls out cleanly; no extra work.

---

## Editor integration

The terrain system surfaces in arvx's UI through three coordinated
elements: a single scene-tree node, an inspector panel, and a
context-sensitive viewport toolbar. **Individual tiles are deliberately
invisible to the author** — they're a streaming implementation detail,
not an authored primitive. Authors think in *places and regions*, never
in tile-coordinate triples.

### Scene tree

Exactly one `Terrain` node per scene (singleton, enforced by the
`Terrain` ECS component). It has no children in the tree. Tiles do
not appear in the scene tree at any time, regardless of their
materialisation or edit state.

```
Scene
├── Sun
├── Camera
├── Terrain                    ← one node, the feature
│   └── Stamps                 ← group for stamp instances
│       ├── Mountain_01
│       ├── Lake_01
│       └── BuildingPlot_flatten
├── Buildings
│   ├── House_01
│   └── House_02
└── …
```

### Inspector panel — shown when Terrain node is selected

```
TERRAIN
─────────────────────────
  Bounds
    Mode:    ▼ Bounded
    Origin:  (0, 0, 0)
    Extent:  16 × 16 × 4
            = 1024 m × 1024 m × 256 m
    [ Resize… ]  [ Switch to Unbounded ]

  Source
    Function:  ▼ FBM (default)
    Seed:     [    42   ]
    Octaves:  [     6   ]
    Scale:    [   120 m ]
    Sea level:[     0 m ]
    [ Edit function… ]   [ Re-bake all tiles ]

  Materials
    Slope rules:
      < 30°   → Grass
      30-60°  → Rock
      > 60°   → Cliff
    Height rules:
      < 0 m   → Sand
      > 200 m → Snow
    [ Edit rules… ]

  Stamps  (5)
    Mountain_01     · Smooth-Max · r=180 m
    Lake_01         · Smooth-Min · r=90 m
    Plateau_01      · Smooth-Max · r=140 m
    BuildingPlot    · Replace    · r=20 m
    Hill_01         · Add        · r=50 m
    [ + Add stamp ▾ ]   [ Manage… ]

  Streaming
    Render radius:    [  800 m ]
    Physics radius:   [  200 m ]
    Tile size:        64 m (locked V1)
    Loaded:           256 tiles
    In flight:        0

  Edits
    237 tiles edited · 14.2 MB on disk
    ☐ Show edit heatmap (viewport)
    [ Revert in radius… ]
    [ Bake snapshot of region… ]
    [ Clear ALL edits ]   ⚠

  Debug overlays
    ☐ Bounds wireframe
    ☐ Tile boundaries
    ☐ Materialisation state
    ☐ Halo dirty state
    ☐ Streaming radii
```

The Stamps section in the Inspector is a *summary* of the
scene-tree group `Terrain ▸ Stamps`; clicking a row selects that
stamp's scene-tree node (and the Inspector switches to showing
that stamp's params). It's an affordance for quick access without
opening the scene tree, not a separate authoritative store.

Inspector trigger is select-driven (clicking the Terrain node),
consistent with every other scene node. There is no separate dockable
terrain panel and no always-on terrain UI. To keep the panel visible
while authoring elsewhere, pin the Terrain node selected — same
mechanism as any other node.

### Viewport toolbar — shown when Terrain node is selected

```
┌────────────────────────────────┐
│ 🖌 Sculpt  🎨 Paint  🔥 Heatmap │
│ ⬛ Region…  ↩ Revert  💾 Bake   │
└────────────────────────────────┘
```

Sculpt and Paint are the existing global brushes; their presence here
is affordance, not a new system — they're routed through brush dispatch
the same way as anywhere else. *Heatmap* toggles the viewport overlay
that tints regions diverging from procedural baseline. *Region* opens a
viewport drag-box for the next operation. *Revert* and *Bake* operate
on the active Region (or camera radius if no region is active).

The toolbar follows the same show/hide rule as the inspector (Terrain
node selected). Pinning the Terrain node keeps both visible while
inspecting another node.

### Edits UX — region-based, never tile-based

The author never sees a list of tile keys. Edits are surfaced as:

- An aggregate counter (`237 tiles · 14.2 MB`) in the Inspector.
- A viewport heatmap that tints edited regions in space.
- Region-scoped operations (Revert, Bake) driven by viewport drag-box
  or camera radius.

Tiles are bookkeeping for the streamer; regions are bookkeeping for the
author. A scene with 1000 edited tiles produces no UI scaling problem
because the UI never enumerates them.

### Brush dispatch — invisible to the author

The existing brush tools (Sculpt, Paint, Smooth, etc.) dispatch to
whatever asset is under the cursor. The only change for terrain: when a
brush hits a terrain tile, the world-space brush AABB is decomposed
across all intersecting tiles and the brush kernel runs per tile in
tile-local coordinates (boundary-edit halo propagation as described in
"Watertight seams via halo"). This is invisible to the brush UI; from
the author's perspective, brushes work on terrain like they work on any
other asset.

### Save / load

`File → Save scene` automatically flushes all touched tiles to
`<scene>/tiles/<level>_<x>_<y>_<z>.arvxtile`. Opening a saved scene
re-attaches those files lazily as tiles materialise. There is no
separate "save terrain" action.

---

## V1 implementation phases

Each phase ships a measurable milestone. No phase ships V2 features.

### Phase 1 — Skeleton + one tile end-to-end

Create the `arvx-terrain` crate. Define `TileKey`, `TerrainFn`,
`TerrainSample`, `TerrainTile`. Ship a trivial `TerrainFn` impl (FBM
heightmap + density). Voxelise one tile via the existing
`arvx-import` voxeliser → octree → surface-nets mesh → cluster DAG.
Hand it to the renderer as a static asset instance. Visible terrain
in 1 tile.

Deliverable: a single 64³ m tile of procedural terrain renders in the
editor.

### Phase 2 — Streamer + sparse 3D tile-octree

Build `TileStreamer` with a sparse 3D tile-octree backing store.
Camera-radius load/unload at level 0. Materialisation runs on the
existing async worker thread infrastructure
([[project-perf-debt-plan]] E2 pattern).

Deliverable: walk around in the editor, tiles materialise ahead and
unload behind. No seams handled yet (cracks expected).

### Phase 3 — Halo + watertight seams

Build `HaloManager`. 1-voxel halo replicated across all 6 faces.
Mesh extract takes `tile + halo`. Neighbour-load gating: a tile
defers full-quality mesh until all 6 neighbours have at least their
boundary face materialised (or remains in a degraded "edges only"
mode and re-meshes on neighbour arrival).

Deliverable: **the first "looks right" milestone.** Continuous
terrain with no visible cracks at tile boundaries.

### Phase 4 — Brush + edits + persistence

World-space brush AABB → enumerate intersecting tiles → run existing
sculpt kernel per tile in local coords. Per-tile dirty tracking →
re-mesh on the worker. Boundary-edit detection → mark up to 3
neighbours' halos dirty. `.arvxtile` write on first persist;
`.arvxtile` read on subsequent loads.

Deliverable: sculpt and paint across tile boundaries; save the
scene; reopen; sculpt persists.

### Phase 5 — Stamps (Layer 2)

`Stamp` ECS component + `StampKind` library (Mountain, Lake,
Plateau, Flatten, Hill) + per-stamp combine op
(Add/Subtract/SmoothMin/SmoothMax/Replace) + spatial index on the
Terrain + voxelisation queries that index and combines into the
base TerrainSample + move/delete invalidation (union of old + new
AABB → dirty tiles) + transform gizmo + Inspector for each stamp
type. Stamps appear as scene-tree children under `Terrain ▸ Stamps`.

Stamps sit between Phase 4 (brush/edit) and Phase 6 (materials)
because (a) they need the voxelisation pipeline working and (b) they
live below sculpt in the layer stack, so their behaviour should be
solid before sculpt-on-top is exercised at scale.

Deliverable: drop a Mountain stamp, see a peak rise; drag the
gizmo, peak follows; place Lake / Flatten / Hill, all combine
correctly; sculpt on top of a stamp, both layers visible.

### Phase 6 — Region primitive (foundational)

Lands `arvx-regions` as a foundational cross-cutting crate, paired
with its first consumer (terrain biomes in Phase 7). Covers Sphere
/ Box / OBB analytical shapes + Falloff (Hard / Linear /
Smoothstep) + `membership(point) -> f32 in [0..1]` + `RegionIndex`
BVH + `BiomeRegion` data component (struct only; terrain
integration in Phase 7) + scene-tree placement (anywhere, optional
`Regions` group convention) + Inspector + viewport gizmos.

Voxelized regions deferred to V2 (additive — same API, new
`RegionShape` variant + paint kernel).

**Full design**: `docs/REGIONS.md`.

Deliverable: drop a Sphere/Box/OBB region in the scene; gizmo to
shape and falloff; attach a `BiomeRegion` component with a
TerrainFn override (not yet consumed by terrain — that's Phase 7).
Scene with two overlapping spheres shows correct membership
weights in a debug overlay.

### Phase 7 — Per-leaf material rule (region-aware)

`TerrainSample` already returns primary/secondary/blend. Wire a
default `TerrainFn` impl that uses slope (from sd gradient) and
height (from world Y) to assign materials: grass on flats, rock on
slopes, snow above N metres, etc.

**Region-aware from day one:** the per-leaf sample loop queries
`region_index.query::<BiomeRegion>(P)` and blends TerrainFn outputs
+ material rules across overlapping biomes by membership weight,
with `Region.priority` resolving single-valued conflicts.

Brushes already override via the existing leaf-write path. Stamps
optionally override the rule within their footprint. Biome region
move / change invalidates intersecting tiles via union(old, new)
AABB (same pattern as stamps).

Deliverable: procedural terrain with believable material blending
out of the box, stamps overriding where they should, biome regions
shifting materials and TerrainFn within their footprint, paintable
on top.

### Phase 8 — Physics

Per-tile Rapier `TriMesh` via `collider_worker`. The three policy
traits with editor defaults. Wake-on-rebuild. Tile lifecycle events.
Rapier origin-rebase event handler (re-register colliders).

Deliverable: drop a Rapier dynamic body on the terrain in the
editor; sculpt; the body wakes and reacts correctly.

### Phase 9 — Editor integration

`Terrain` ECS component + scene-tree node (singleton-enforced) +
default bounded extent on creation (16 × 16 × 4). Inspector panel
with Bounds / Source / Materials / Stamps / Streaming / Edits /
Debug sections per the layout above. Viewport toolbar (Sculpt /
Paint / Heatmap / Region / Revert / Bake + stamp-add menu) bound
to Terrain-selected context. Brush dispatch wires terrain-tile
hits into the existing sculpt/paint kernel transparently
(world-space brush AABB decomposed across intersecting tiles).
Viewport heatmap overlay shaded by per-leaf divergence from
procedural-plus-stamp baseline. Bounds wireframe overlay.
`File → Save scene` flushes touched `.arvxtile`s and serialises
stamps + bounds + TerrainFn config into the scene file.

Deliverable: the full session walkthrough at the top of this doc
works end to end. Drop a Terrain node → bounded 16 × 16 × 4 grid
appears → pick TerrainFn → place stamps → sculpt → save → reopen →
everything persists.

---

## V2 follow-ups (separate sessions, not in V1)

- **LOD pyramid:** light up `TileKey.level > 0` allocation. Coarse-tile
  voxelisation; fine-ring residency policy for cross-LOD seams.
- **Transvoxel transition cells:** if fine-ring's wasted ring becomes a
  memory issue, replace with proper transition cells.
- **Edit-diff propagation up the pyramid:** re-voxelise coarse tile +
  apply downsampled diff between (baked fine tile) and (procedural fine
  sample).
- **Per-level `TerrainFn` pre-filtering:** coarse sample ≈ avg of fine
  samples. Heightmap-style impls get it for free; noise-style impls
  need explicit mip-style filtering.
- **Node-graph `TerrainFn` editor.** Visual node graph (noise / ridge /
  warp / mask / threshold / blend / biome nodes) compiling to a
  `TerrainFn` impl. Same caching, same sampling API — purely an
  authoring affordance. The single largest UX investment in V2.
- **Erosion baking.** "Bake erosion" action: evaluate current
  TerrainFn + stamps on a 2D heightmap grid, run hydraulic + thermal
  erosion N iterations, swap TerrainFn to a heightmap-sampling impl
  using the eroded result. Slow (seconds-to-minutes) but only run on
  request. Output is part of the deterministic baseline.
- **Region erosion brush.** Run a single iteration of erosion within a
  brush footprint, write the result as sculpt edits to the affected
  tiles. Fits as a new brush kernel under the existing sculpt
  pipeline.
- **Additional stamp types:** River (polyline-defined channel), Cave
  entry (subtractive volume opening into the surface), Road (polyline
  flatten with shoulder profile).
- **Vehicle-scale predictive materialisation:** velocity-ray
  `PredictiveMaterializationPolicy` impl.
- **Compound / simplified collision shapes:** if trimesh-per-tile
  profiles hot at high tile counts, profile-driven replacement.
- **Voxelized regions** (`docs/REGIONS.md`) — paintable region
  membership for irregular biome boundaries that don't fit analytical
  shapes. Same `Region` API, additive `RegionShape` variant + new
  "Region paint" brush kernel.
- **Other region consumers** — `AmbientAudio`, `SpawnZone`, `FogVolume`,
  `GameplayTrigger` as the respective consumer systems land.

---

## What this design explicitly does NOT do

These are not "missing"; they are *intentionally absent* to keep V1
focused and the API surface honest:

- No new render path. No new mesh format. No new shaders.
- No baked character controller (KCC / dynamic / FPS / etc.). Consumers
  ship their own.
- No baked rebuild interval / physics radius / velocity heuristic. All
  policy.
- No heightfield colliders (would foreclose caves, overhangs, floating
  geometry).
- No per-leaf cube colliders (orders of magnitude too many bodies).
- No edit replay log or sparse delta overlay (matches existing arvx
  sculpt — overlay was retired in Phase A as an architectural dead
  end).
- No assumed "player" concept anywhere in terrain code.
- No node-graph `TerrainFn` editor in V1 — code-defined `TerrainFn`
  impls only (FBM, heightmap-import shipped as examples). Authors who
  want different macro shape write a Rust impl. Node graph is V2.
- No erosion in V1 (bake-time or runtime). Authors get the result of
  whatever their TerrainFn produces, augmented by stamps + sculpt.
  Erosion is V2.
- No River / CaveEntry / Road stamps in V1 — they need more involved
  SDF definitions (polyline-driven, explicitly volumetric). V2.

---

## Decision log

| Decision | Choice | Rejected alternatives |
|---|---|---|
| World scale | Open world, streamed | Bounded level / planet / local-only |
| Source data | Procedural + in-editor sculpt | Heightmap import / pre-baked tiles |
| Editability | Fully editable runtime + editor | Editor-only / read-only |
| Seam strategy | Shared boundary voxels (watertight) via halo | Skirts / continuous global field |
| Tile shape | 3D cubic 64³ m | 2.5D columns (bounded or unbounded) |
| Tile footprint | 64 m | 32 m / 128 m |
| LOD strategy | Tile-octree with merged coarse tiles (V2), uniform fine LOD (V1) | Per-tile DAG only / coarser voxelisation at far rings |
| V1 scope on LOD | Uniform fine LOD + stub the pyramid API | Pyramid in V1 / no level field at all |
| Edit overlay | Full baked tile when edited (matches existing sculpt) | Replay log / sparse leaf delta / RAM-only |
| `TerrainFn` signature | Tile-local + integer key | `WorldPosition` arg / both |
| Physics shape | Per-tile Rapier TriMesh from LOD-0 cluster triangles | Heightfield / per-leaf cubes / convex decomposition / compound-per-cluster |
| Physics policy | Three policy traits (residency, rebuild, predictive) | Baked radii + cadences |
| Character controller | Consumer-supplied | KCC / dynamic body / both / none — terrain has no opinion |
| FP drift handling | Integer tile-keys cross boundaries; f32 inside one tile only | Single f32 world frame / f64 throughout |
| Terrain count per scene | Exactly one Terrain node (singleton) | Multiple terrains / design-for-multiple-later |
| Scene-tree representation | One `Terrain` node; tiles never appear in the tree | Each tile as a scene-tree child |
| Edits UX | Aggregate counter + viewport heatmap + region ops | Per-tile list / search / grouping |
| Inspector trigger | Inspector-on-select (consistent with other nodes); pin to keep visible | Pinnable / always-on dockable panel |
| Terrain-specific viewport UI | Toolbar shown when Terrain selected (Sculpt/Paint/Heatmap/Region/Revert/Bake + stamp-add menu) | None / always-on |
| Source model | Three layers: Base `TerrainFn` + Stamps + Sculpt | Monolithic procedural fn / sculpt-only / heightmap-only |
| Authoring scope | Bounded grid by default (16×16×4 = 1024×1024×256 m); Unbounded opt-in | Always unbounded / always bounded / no defaults |
| Stamps scoping | Scene-tree children under `Terrain ▸ Stamps`; Inspector summary mirrors the group | Inline list only / global stamps panel / spatial-only no tree |
| Stamp combine op | Per-instance op (Add / Subtract / SmoothMin / SmoothMax / Replace) | Fixed-per-type / single global op |
| V1 stamp library | Mountain, Lake, Plateau, Flatten, Hill | Smaller / larger / River+Cave in V1 |
| `TerrainFn` authoring (V1) | Code-defined Rust impl only (FBM + heightmap-import as examples) | Node-graph editor in V1 |
| Erosion | V2+ bake-time pass producing heightmap input to TerrainFn; V2+ region-erosion brush | Built into TerrainFn / runtime sim / V1 |
| Biome variation | Cross-cutting `arvx-regions` primitive consumed via `BiomeRegion` data component (see `docs/REGIONS.md`) | Terrain-specific biome system / purely positional rules only / single global biome |
| Resolution table | One unified pow2 table in `arvx_core::constants::RESOLUTION_TIERS` (1.0 m → 0.0078125 m, 2× ratios, 8 tiers). Shared across terrain, mesh imports, procedurals. | Per-system tables / 4× ratios / non-power-of-2 sizes |
| AABB padding | `voxelize_octree` requires caller to pre-align AABB to pow2-cubic; no hidden internal padding. Callers with arbitrary AABBs use `arvx_core::pad_to_pow2_cubic`. | Silent internal rounding-up + centring (pre-unification behaviour) |

---

## Open questions before Phase 1 starts

1. **`WorldPosition` representation in `rkf-core`** — sector+local?
   fixed-point i64? f64? Determines exact `TileKey ↔ WorldPosition`
   conversion shape. 30-second code read; do not guess.
2. **Rapier origin-rebase wrapper** — does one already exist anywhere
   in `rkf-physics`, or does it need building? Affects scope of
   Phase 8.

Everything else in this document is decided.

---

## See also

- `docs/REGIONS.md` — region primitive design (consumed in Phase 7
  via `BiomeRegion`).
- `docs/PERF_DEBT.md` — `collider_worker` pattern (consumed in
  Phase 8) and per-asset async worker model the streamer follows.
