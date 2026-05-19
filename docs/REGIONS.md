# Region system — design

**Status**: design (2026-05-18). No code yet. Will land as a foundational
primitive used first by terrain biomes, then by other systems as
needs arise.

A region is a **named volume in the world that systems can query for
membership**, with arbitrary per-system data attached. The region
system is intentionally cross-cutting — biomes, audio zones, AI
behaviour areas, fog volumes, spawn zones, gameplay triggers are all
"things that apply within this volume." Building region as a primitive
instead of a terrain feature means each consumer attaches its own data
component and queries the same spatial index.

Terrain is the first (and V1-only) consumer. The shape of the API is
chosen so audio, gameplay, fog, etc. can plug in later without
touching the primitive.

---

## Core types

```rust
/// The spatial part — what makes an entity a region.
pub struct Region {
    pub shape: RegionShape,
    pub falloff: Falloff,
    pub priority: i32,         // resolves overlaps when a property must be single-valued
}

pub enum RegionShape {
    Sphere     { radius: f32 },
    Box        { half_extents: Vec3 },
    Obb        { half_extents: Vec3, rotation: Quat },
    Voxelized  { grid: SparseVoxelBits, voxel_size_m: f32 },   // V2 (see "Voxelized regions")
    // Further V2/V3: ConvexHull, Polygon2DExtruded, Sdf
}

pub enum Falloff {
    Hard,                                 // membership ∈ {0, 1}
    Linear     { transition_m: f32 },     // linear blend zone outside shape
    Smoothstep { transition_m: f32 },     // smoother blend
}

/// Every consumer uses this single membership query.
/// Returns 0..1: deep inside = 1.0, well outside = 0.0,
/// falloff zone in between as defined by `Falloff`.
pub fn membership(
    region: &Region,
    transform: &Transform,
    point: WorldPosition,
) -> f32;
```

The `Region` component owns *where*. System-specific data components
own *what for*:

```rust
// Consumed by terrain (Phase 6 of the V1 terrain plan)
pub struct BiomeRegion {
    pub terrain_fn_override: Option<Box<dyn TerrainFn>>,
    pub material_override:   Option<MaterialRule>,
}

// Future consumers — not in V1, just illustrative
pub struct SpawnZone   { pub spawn_table: SpawnTableId, pub density: f32 }
pub struct AmbientAudio { pub clip: AudioClipId, pub gain: f32 }
pub struct FogVolume    { pub colour: Vec3, pub density: f32, pub blend: BlendMode }
pub struct GameplayTrigger { /* enter/exit event emitters */ }
```

A single region entity carries `Region` plus zero-or-more data
components. The "Dark Forest" region might be
`Region + BiomeRegion + AmbientAudio + FogVolume + SpawnZone` all on
one entity. Each consumer queries the index for *its* component type
and ignores the rest.

---

## Membership query semantics

The membership function is the *only* public spatial primitive.
Returns `f32` in `[0, 1]`:

- `1.0` — fully inside the shape
- `0.0` — fully outside the shape AND outside any falloff transition
- in between — within the falloff zone, weighted per `Falloff` variant

**Why a soft membership and not a boolean?** Biomes blend in nature;
audio cross-fades; fog gradients exist. Forcing every consumer to do
its own falloff math fragments the model. One function, every consumer
uses it, biomes use `Smoothstep`, gameplay triggers use `Hard`.

**Priority** resolves overlap *only* for single-valued properties.
Two BiomeRegions overlapping with weights 0.6 and 0.4 blend their
heightmap contributions continuously (no priority needed). But if one
forces material = Snow and the other forces material = Sand, the
higher-priority region wins in the overlap; the loser's material
contribution is dropped. The `priority` field exists for exactly this
case.

---

## Spatial indexing

```rust
pub struct RegionIndex {
    /* sparse BVH over region bounding-spheres / AABBs */
}

impl RegionIndex {
    pub fn query<D: RegionData>(&self, point: WorldPosition) -> Vec<(EntityId, f32)>;
    //          ↑ data component type           ↑ membership weight at point
}
```

Built once per frame after region transforms settle. Each consumer
queries the index by data-component type at a point and gets back
`(entity, membership_weight)` pairs. The index doesn't know or care
which data components a region carries — it just indexes spatially
and lets the consumer's type filter the result.

For analytical shapes (Sphere / Box / OBB) the index is over bounding
spheres or AABBs and membership is computed analytically per query.
For voxelized regions (V2) the index returns the region's analytical
bounds and the consumer samples the bitmap inside the bounds.

Typical scenes have tens to low-hundreds of regions; query cost is
dominated by per-shape membership maths, not BVH traversal. Cheap to
query thousands of times per frame (e.g., per-leaf during terrain
voxelization).

---

## Scene-tree integration

Regions are general-purpose, so they don't live under Terrain. Pattern:

```
Scene
├── Terrain
│   └── Stamps
│       └── …
├── Regions                       ← optional convenience group
│   ├── DarkForest_biome          ← Region + BiomeRegion + AmbientAudio
│   ├── BossArena_trigger         ← Region + GameplayTrigger
│   └── UnderwaterVolume          ← Region + FogVolume + AmbientAudio
├── Lights
│   └── …
└── …
```

A region is just an entity with a `Region` component, placeable
anywhere in the tree. The `Regions` group is a convenience for
authors who want them all together; not enforced. Authors who prefer
to keep a fog volume next to its environment node, or an AI region
next to its NPCs, are free to.

**Inspector when a region is selected** shows:

```
REGION
  Shape:    ▼ Sphere
  Radius:   [  120 m ]
  Falloff:  ▼ Smoothstep
  Transition: [  20 m ]
  Priority:   [    0 ]

[ + Add data: ▾ Biome | Audio | Fog | Trigger | … ]

— BIOME REGION —                     ⋮
  TerrainFn override: ▼ FBM (forest preset)
  Material override:  ▼ Grass + moss

— AMBIENT AUDIO —                    ⋮
  Clip: ▼ forest_ambient.ogg
  Gain: [ 0.8 ]
```

Adding a data component is `[ + Add data ]` → menu of registered
component types. Each attached component is a collapsible card in the
inspector. Remove with the `⋮` menu.

**Gizmo** in the viewport: sphere/box/OBB handle for shape, smaller
handle for falloff transition radius (visualised as a semi-transparent
shell outside the main shape).

---

## Voxelized regions (V2)

The analytical shapes cover ~85% of biome authoring needs. The
remaining cases — "this irregular blob of forest that hugs the river
on one side and breaks against the mountain on the other" — want
**paintable** region membership.

```rust
RegionShape::Voxelized {
    grid: SparseVoxelBits,     // sparse octree of membership bits
    voxel_size_m: f32,         // typically 1-2 m; coarser than terrain voxels
}
```

Design points:

- Region voxel grid is **coarser than terrain** (default ~1 m, not
  terrain's 0.25 m). Biomes don't need sub-meter precision.
- Storage = sparse octree of bits (existing arvx octree infrastructure
  serves; just a different leaf type).
- **Authored via a new brush kernel** — "Region paint" — that reuses
  the existing brush dispatch and footprint logic but writes
  membership bits instead of material IDs. Same brush UI affordance,
  new write target.
- Membership query trilinearly samples the bitmap, then applies the
  region's `Falloff` (so a voxelized region can still have a smooth
  outer transition).
- Larger regions use the sparse octree; small ones could use a dense
  bitmap if profiling shows octree overhead.

**Why V2:** the analytical shapes ship the primitive and the data-
component pattern. Voxelized regions are an additive shape variant
that doesn't change the API. Splitting them out lets V1 ship the
region system on a small footprint and lets V2 add the brush-paint
authoring flow as a focused piece.

If voxelized regions become a V1 blocker (e.g. terrain biomes feel
too geometric without them), they can be pulled forward without
restructuring anything.

---

## How terrain consumes regions

In the terrain V1 plan (`docs/TERRAIN.md`), Phase 6 (per-leaf material
rule) becomes region-aware:

```rust
// Pseudocode for a per-leaf TerrainSample computation
fn sample_leaf(p: WorldPosition) -> TerrainSample {
    // Layer 1a: global base
    let global = global_terrain_fn.sample(...);

    // Layer 1b: biome regions
    let biome_regions = region_index.query::<BiomeRegion>(p);
    let after_biomes = blend_terrain(global, &biome_regions, p);
    //                                       ↑ each (region, weight)

    // Layer 2: stamps
    let stamps = stamp_index.query(p);
    let after_stamps = combine_stamps(after_biomes, &stamps, p);

    // Material rule (also region-aware)
    let material = resolve_material(after_stamps, &biome_regions, p);

    TerrainSample { sd: after_stamps.sd, primary_mat: material, … }
}
```

Continuously-blendable properties (height, slope influence) interpolate
across overlapping biome regions weighted by membership. Single-valued
properties (primary material assignment) use priority to resolve, with
membership weight controlling the secondary material's blend.

When a biome region moves or changes, the terrain system invalidates
tiles intersecting the union of (old region AABB ∪ new region AABB) —
same pattern as stamps.

---

## V1 scope

### Phase 1.5 — Region primitive (analytical shapes)

Lands as Phase 1.5 in the terrain V1 plan (after terrain skeleton,
before streamer) so subsequent terrain phases can use it.

Deliverables:
- `arvx-regions` crate with `Region`, `RegionShape::{Sphere, Box, Obb}`,
  `Falloff`, `membership(...)`.
- `RegionIndex` with BVH-backed spatial query by data-component type.
- `BiomeRegion` data component (struct only; consumer integration is
  Phase 6).
- Scene-tree integration: regions placeable anywhere; optional
  `Regions` group node convention.
- Inspector with shape / falloff / priority controls + data-component
  picker.
- Viewport gizmos for shape + falloff visualisation.

Out of scope for Phase 1.5 (V2 or later):
- `RegionShape::Voxelized` — its own sub-phase, additive.
- Non-terrain consumers (`SpawnZone`, `AmbientAudio`, `FogVolume`,
  `GameplayTrigger`) — added when concrete needs land.

### Phase 6 (terrain) becomes region-aware

The terrain plan's Phase 6 (per-leaf material rule in `TerrainFn`)
extends from "simple slope/height rules" to "simple rules + biome
region overlays." Adds:

- Biome blending in the per-leaf sample loop.
- Material resolution honouring region priority.
- Tile-invalidation hook when a `BiomeRegion` moves/changes (union
  AABB → dirty tiles).

---

## V2 follow-ups

- **`RegionShape::Voxelized`** — paintable region membership via sparse
  octree of bits; new "Region paint" brush kernel.
- **Other data components** — `SpawnZone`, `AmbientAudio`, `FogVolume`,
  `GameplayTrigger` as the consumer systems land.
- **Polygon2DExtruded shape** — 2D polygon extruded vertically, the
  natural shape for biome maps authored as outlines.
- **ConvexHull shape** — author-defined convex volumes for irregular
  but analytical regions.
- **`Sdf` shape** — region defined by an SDF function (for animated
  regions, smooth procedural boundaries).
- **Region instancing** — share the same region definition across
  multiple transforms (forests of the same biome type).

---

## Decision log

| Decision | Choice | Rejected |
|---|---|---|
| Cross-cutting vs terrain-only | Cross-cutting primitive in `arvx-regions` | Terrain-specific feature in `arvx-terrain` |
| Membership return type | `f32` (0..1) | `bool` only / bool+separate-falloff API |
| Data attachment | ECS data components (one entity, many components) | Region carries typed payload / parallel maps from region-id to data |
| V1 shape library | Sphere + Box + OBB analytical | Sphere+Box only / + Polygon2D / + ConvexHull / + Voxelized in V1 |
| Voxelized regions | V2 (additive, no API break) | V1 / never |
| V1 consumers | Terrain biomes only (`BiomeRegion`) | + Audio / + Fog / + Triggers all in V1 |
| Scene-tree placement | Anywhere; optional `Regions` convenience group | Forced under a `Regions` root / forced under per-system roots |
| Crate placement | New `arvx-regions` crate | In `arvx-core` / in `arvx-terrain` / preemptively shared as `rkf-regions` |
| Priority resolution | Per-property: blend continuously when blendable, priority-resolve single-valued | Always priority / always blend |

---

## Open questions

1. **Voxelized region resolution default** — 1 m, 2 m, configurable
   per-region? Resolved when Phase voxelized lands.
2. **Brush-paint integration for voxelized regions** — reuse the
   existing brush dispatch (same kernel, different write target) or a
   separate code path? Resolved when voxelized regions are scoped.
3. **`RegionIndex` rebuild cadence** — once per frame, or only when
   regions actually change? Probably the latter (regions are rare to
   move at runtime); profile to confirm.

---

## See also

- `docs/TERRAIN.md` — the V1 terrain plan that consumes regions via
  `BiomeRegion` (Phase 6).
