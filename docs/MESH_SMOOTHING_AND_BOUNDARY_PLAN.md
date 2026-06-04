# Mesh Smoothing + Voxel/Mesh Boundary — Implementation Plan

**Status:** ready to execute. **Owner:** —. **Created:** 2026-06-04.

This plan is written so a *lower-effort* model (or a human) can execute it one task at a
time without re-deriving the architecture. Background and the full reasoning live in the
session memory files:

- `project_smooth_mesh_resolution_gibson.md` — why meshes are blocky and the Gibson fix
- `project_scattered_authority_diagnosis.md` — the 6 missing authorities
- `project_voxel_mesh_boundary_architecture.md` — the target VoxelModel/MeshView boundary

## The one idea

Everything below installs **one boundary**: a `VoxelModel` (source of truth) and a
`MeshView` (disposable derived view) connected by a single `RemeshRegion` change-feed, with
the mesher a near-pure function of voxel data. **Stage A is the keystone and the decision
gate** — until the mesher stops needing the brush (projection deleted), nothing else can
proceed. Stages B–D are gated on Stage A's visual result and should be re-planned after it.

## How to run this (execution protocol)

1. Work on a branch: `git checkout -b mesh-smoothing-boundary` (don't commit to `master`).
2. Do **one task** at a time, top to bottom. Each task has a **Verify** block — run it and
   confirm green before moving on.
3. **STOP at every `🚧 HUMAN GATE`** — those need a person to run the editor and look at the
   result; an agent cannot judge them.
4. Build/test commands: `cargo build -p <crate>`, `cargo test -p <crate>`,
   `cargo clippy -p <crate>`. Visual: `cargo run -p arvx-editor`.
5. Per project rule (CLAUDE.md): **tests first**, correctness over speed, no shortcuts.

---

## ⚠️ The single most important correctness note (read before Stage A)

The naive smooth-placement trap: **do NOT place vertices with a QEF driven by the stored
`LeafAttr.normal_oct` on the CPU sculpt path.** That is exactly what commit `9b4930c8`
removed, because sculpt deliberately homogenizes per-leaf normals (anti-shading-stripe), so
the tangent-plane set is **rank-1** and the QEF degenerates / swells the surface. The GPU
`proc_surface_nets.wesl` QEF works *only* because procedurals feed it full-rank gradient
normals from a live field — that path keeps its QEF; the CPU path must not copy it.

The correct occupancy-only placer for the CPU path is **Gibson Constrained Elastic Surface
Nets**: naive seed → Laplacian/Taubin smoothing → **clamp each vertex to its own cell box
(±h/2 of its original position)** → recompute normals from the relaxed faces. This needs no
field and is robust to rank-1 normals. The stored normal is used only as an optional *soft
anchor* to stop convex-feature erosion, never as the QEF's sole constraint.

---

## Stage 0 — Quick win: `is_glass()` authority (independent, do anytime)

Cheap, near-zero risk, kills a real hole bug. Not on the critical path; parallelizable.

### Task 0.1 — Single glass predicate + threshold
- **Goal:** one owner for the `opacity < 0.99` rule, replacing 6 hand-spelled sites.
- **Files:** `crates/arvx-render/src/shaders/lib/types.wesl` (add `is_glass()` +
  `GLASS_OPACITY_THRESHOLD` + `clamp_material_id()`); callers `mesh.wesl`,
  `mesh_shadow.wesl`, `mesh_glass_shadow.wesl`, `arvx_glass.wesl`, `mesh_glass.wesl`; CPU
  side reads the same constant (today `DEFAULT_OPACITY_THRESHOLD`, ~`mesh_glass_pass.rs`).
- **Steps:** add the fn + const to the shared `types.wesl`; replace each literal `0.99` and
  the OOB `select(...,0u,...)` clamp with a call; promote one CPU constant the GPU reads.
- **Verify:** `cargo build -p arvx-render` green; `grep -rn "0.99" crates/arvx-render/src/shaders`
  returns no glass-threshold literals; editor renders glass assets unchanged.
- **Risk:** low. Pure consolidation.

---

## Stage A — KEYSTONE: occupancy smoothing + delete projection (the decision gate)

This ships the smoothness fix **and** is the precondition for the whole boundary. Small,
high-leverage, reversible. The result of `🚧 A5` reshapes Stages B–D.

### Task A1 — Falsification tests (test-first; pins the honest limit)
- **Goal:** prove in-repo what occupancy + known `h` can and cannot recover, so nobody later
  "fixes" rounded corners by over-tuning the smoother.
- **File:** `crates/arvx-core/src/mesh_extract.rs` (`#[cfg(test)] mod tests`), or a new
  `crates/arvx-core/tests/occupancy_limits.rs`.
- **Steps:** write two tests that synthesize occupancy on a known grid and assert
  *identical* occupancy for geometrically different surfaces:
  1. `flat_plane_and_dome_have_identical_occupancy`: a flat plane `y = 6h` vs a parabolic
     dome of curvature radius `R = 4h` over a small footprint → assert the per-cell
     inside/outside bits match. (Proves the ±h/2 normal-position floor.)
  2. `sharp_corner_and_fillet_have_identical_occupancy`: a 90° corner at a grid corner vs an
     `r = h/2` fillet → assert identical bits. (Proves sub-2h sharp features are
     unrecoverable from occupancy alone.)
- **Verify:** `cargo test -p arvx-core occupancy` green (both asserts pass).
- **Risk:** none — pure tests. They are guardrails, not production code.

### Task A2 — Box-constrained relaxation placer (Gibson) — the core change
- **Goal:** turn the dead, unconstrained, shrink-only `relax_surface_net_vertices` into a
  constrained elastic surface net that de-staircases with bounded (≤ h/2) error.
- **File:** `crates/arvx-core/src/mesh_extract.rs:1281` (`relax_surface_net_vertices`).
- **Steps (modify the existing fn):**
  1. Before the iteration loop, capture origins: `let orig: Vec<Vec3> = vertices.iter()
     .map(|v| Vec3::from(v.local_pos)).collect();` and `let half = voxel_size * 0.5;`.
  2. Make it real Taubin (optional but recommended): alternate a shrink step (λ≈0.33 toward
     1-ring mean) and an inflate step (μ≈−0.34) instead of shrink-only — prevents global
     shrinkage. If you keep shrink-only, the box clamp below still bounds it; Taubin is
     cleaner.
  3. **After** computing each vertex's new position, **clamp componentwise to the cell box:**
     `new_pos[i] = new_pos[i].clamp(orig[i] - half, orig[i] + half);`. This is the Gibson
     constraint and the only place `h`/`voxel_size` is load-bearing.
  4. (Soft anchor, recommended) blend the relaxed position a little toward the plane through
     `orig[i]` with the vertex's current normal, to resist convex-feature erosion — but keep
     it inside the box. Keep weight small (e.g. 0.25); this is a guard, not the placer.
  5. Fix the faked neighbor dedup (currently relies on adjacent-duplicate skip after sort) —
     do a real dedup so accumulation weights can't silently double.
- **Add tests** in the same file:
  - `box_clamp_bounds_displacement`: after relax, every vertex moved ≤ `h/2` from its origin.
  - `relax_removes_staircase_on_synthetic_slope`: a tilted-plane occupancy patch → staircase
     RMS drops below ~0.1h after relax.
  - `relax_preserves_volume_within_few_percent` on a synthetic sphere (vs unbounded Taubin
     which shrinks).
- **Verify:** `cargo test -p arvx-core relax` green; `cargo clippy -p arvx-core`.
- **Risk:** medium. This is the algorithmic heart. Keep the QEF trap note above in mind.

### Task A3 — Recompute normals from relaxed faces
- **Goal:** make the de-staircasing visible under lighting (geometry + shading agree) and
  sidestep the rank-1 stale-normal trap.
- **File:** same fn (`relax_surface_net_vertices`), after the position loop.
- **Steps:** accumulate area-weighted (or angle-weighted) triangle face normals from the
  *relaxed* positions over `indices.chunks_exact(3)`, normalize per vertex, re-pack into
  `MeshVertex.normal_oct` via `pack_oct`. The adjacency/index structure is already built.
- **Verify:** add `relaxed_normals_vary_continuously_on_sphere` (no 26-direction lattice
  banding); `cargo test -p arvx-core`.
- **Risk:** low–medium. Watch winding/orientation so normals point outward.

### Task A4 — Wire it in; delete the projection layer
- **Goal:** call constrained relax in the sculpt re-extract path and remove every brush-aware
  placement hack, so the mesher stops needing the brush.
- **Files:** `crates/arvx-render/src/arvx_scene_manager/sculpt.rs`,
  `crates/arvx-core/src/mesh_extract.rs`.
- **Steps:**
  1. In `rebuild_dirty_clusters` (`sculpt.rs:952`), after the extract and **instead of** the
     `match op.mode { project_onto_brush_capsule / project_clay_strip }` block
     (`sculpt.rs:1251-1273`), call
     `arvx_core::mesh_extract::relax_surface_net_vertices(&mut verts, &indices, base_vs, 6, None)`.
     Use `pin_boundary = None` for sculpt patches (the h/2 clamp keeps the patch seam tight).
  2. Do the same in `rebuild_stroke_clusters` (`sculpt.rs:1653`): delete the `stroke_sdf`
     closure (`sculpt.rs:1795`), stop passing `Some(&stroke_sdf)` (`sculpt.rs:1817`) →
     pass `None`, and replace the `match op.mode` projection block (`sculpt.rs:1822-1842`)
     with the same relax call.
  3. Delete the now-unused projection fns from `mesh_extract.rs`:
     `project_onto_brush_capsule` (:1403), `project_clay_strip` (:1449),
     `project_onto_stroke_capsules` (:1563), `project_clay_strip_stroke` (:1591), and
     `nearest_on_polyline` (:1535) if it has no other callers (check first).
  4. Remove the `sdf_fn` parameter from `build_cube_vertex` (:1140) and the
     extract-region functions (`:822`) and the `Some`/`None` plumbing — the smooth path is
     now the relax pass, not the edge-crossing interpolation. (Leave `extract_surface_mesh`
     / `_haloed` public signatures stable if other callers depend on them; only drop the
     internal `sdf_fn` thread.)
  5. Remove the `use` imports of the deleted fns in `sculpt.rs:33-35`.
  6. Note: the GPU `proc_surface_nets.wesl` QEF is **untouched** — it keeps working from its
     live field. This task only changes the CPU occupancy path.
- **Verify:** `cargo build --workspace`; `cargo test -p arvx-core -p arvx-render`;
  `cargo clippy --workspace` (no dead-code warnings for the deleted fns).
- **Risk:** medium. The patch/kept-geometry weld seam may show a rim crack (see A5 / Stage B);
  that's expected and is handled by the boundary work, not here.

### 🚧 A5 — HUMAN GATE: visual parity (this decides Stages B–D)
- **Run:** `cargo run -p arvx-editor`. Sculpt a **ClayStrip** drag along a terrain **tile
  boundary**, with a non-grid-aligned radius and strip-top height. Also test Raise/Carve.
- **Judge two things:**
  1. **Smoothness:** the stroke renders smooth (no staircase), shading is continuous.
  2. **Crease fidelity:** does the clay-strip **flat-top crease** survive, or does it round
     off? (Gibson/Laplacian-family placers round convex creases by construction.)
- **Decision:**
  - ✅ **Flat-top survives + smooth** → occupancy + relax is sufficient. Proceed to **Stage B**.
  - ⚠️ **Crease rounds** → do **Task A6** (sharp-feature channel) before deleting projection
    is considered final, then re-gate. This is the evidence-triggered point where stored
    Hermite data earns its place — *not before*.

### Task A6 — (conditional) Sharp-feature flag in the normal channel
- **Trigger:** only if A5 shows convex creases rounding and that's unacceptable.
- **Goal:** let the placer pin a feature vertex where a crease exists, without a full DC.
- **Sketch:** detect creases geometrically at extract (divergent face normals across a cube /
  high dihedral), tag those vertices, and for tagged vertices skip the relax clamp toward the
  mean and instead place via a 2-plane QEF from the *divergent face normals* (full-rank by
  definition at a crease). This is the minimal slice of "stored Hermite" — gated on real
  need. Re-run 🚧 A5 after.
- **Verify:** a beveled/clay-strip flat-top keeps its edge while curved regions stay smooth.

---

## Stage B — Install the `RemeshRegion` authority (gated on A5 ✅)

Collapse the three private re-extract span definitions and the `skip_remesh` bool into one
owner. This is "missing authority #1" and the first brick of the boundary.

- **B1.** Define `RemeshRegion { lo: IVec3, hi: IVec3, reason: RemeshReason, epoch: u64 }`
  (finest-grid cells, half-open) and `ClusterDelta { replaced: Vec<u32>, appended:
  Vec<MeshletCluster>, vertex_ranges, index_ranges }` in `arvx-render` (or `arvx-core`).
- **B2.** Introduce one `remesh(view, &[RemeshRegion]) -> ClusterDelta` that internally
  computes extract-span = union and filter-span = union + halo (the **one** place that
  knowledge lives), replacing the private spans in `rebuild_dirty_clusters` /
  `rebuild_stroke_clusters` (`sculpt.rs:952/1653`) and `rebuild_face_band_clusters`
  (`terrain_halo_refresh.rs`).
- **B3.** Delete `skip_remesh` (`terrain_halo_refresh.rs:69`) by emitting an edit **and its
  triggered halo refreshes as one atomic epoch batch** (this is the part the bool was
  hiding — see the irreducible-coupling note in the boundary memory).
- **B4.** Move `mark_lod_dirty_chains`, cluster-table construction, slab-allocator surgery,
  and the `mesh_dirty`/`clusters_dirty`/epoch bumps off the sculpt caller onto the mesher.
- **Also close** the last delta-upload violator: cluster table currently does a full
  `queue.write_buffer` every stamp — give it `clusters_dirty: DirtyRanges` so `ClusterDelta`
  drives a delta upload.
- **Verify:** sculpt + cross-tile drag still watertight; `cargo test --workspace`; per-stamp
  cluster upload is a delta, not a full rewrite.

## Stage C — Split `VoxelModel` / `MeshView` (gated on B)

Today `AssetEntry` (`types.rs`) interleaves voxel truth, the derived mesh, and the GPU
allocator. Split it.

- **C1.** `VoxelModel`: `cpu_octree`, brick/leaf pool slices, `halo_cells`, a per-cell
  provenance `priority: u8` (replacing the `sculpt_owned_slots` FxHashSet threaded into the
  mesher), the dirty queue. Only edits mutate it.
- **C2.** `MeshView`: vertices, indices + slab allocator/free-list, `meshlet_clusters`, DAG,
  all `DirtyRanges`, stroke/scratch state. Rebuilt, never edited by voxel code.
- **C3.** `Mesher` trait: `remesh(&VoxelModel, &[RemeshRegion], &mut MeshView) ->
  ClusterDelta`. **Note the `&mut MeshView` read input is required** — re-meshing a patch must
  weld to kept surrounding triangles (the patch/kept seam is irreducible; the mesher is
  `fn(VoxelModel, MeshView)`, not `fn(VoxelModel)` alone).
- **C4.** Kill the `CELL_INTERIOR` (u32::MAX) vs `CELL_GRID_EMPTY` collision by giving the
  occupancy cell two fields (solid-bit + slot) instead of one sentinel-overloaded u32.
- **Verify:** voxel-edit code no longer references any mesh type; `cargo test --workspace`.

## Stage D — GPU mesher + atomic swap (endgame, gated on C)

- **D1.** GPU-resident occupancy mirror per `VoxelModel` (packed `cell_solid` + `cell_slot`),
  written by the **same pool delta-upload** (the pools already are the occupancy source).
- **D2.** Port `proc_surface_nets.wesl` to a second `Mesher` impl: reuse `surface_normal`,
  `vertex_emit` (its QEF is fine here — GPU occupancy gradients are full-rank),
  `index_emit`; only the `classify` body changes from `eval_tree_distance` to an occupancy
  slab read. Same trait, drop-in behind C3.
- **D3.** Double-buffered cluster pool + **epoch-versioned atomic swap** (this is net-new and
  a torn-table risk until built — budget it as real work). Apply `ClusterDelta`
  transactionally; old mesh renders until the new one lands; spread swaps across frames under
  an ~8 ms budget (Godot `ApplyMeshUpdateTask` model).
- **D4.** Colliders: a **separate** amortized consumer that reads back LOD-0 CPU triangles
  (Rapier TriMesh) on its own clock — cooking is 3–5× meshing, main-thread. This cost is
  real and cannot be made free; it lives on the consumer side, not in the mesher.
- **Verify:** sculpt is GPU-meshed with no render-path readback; latency budget held;
  watertight under async (epoch token prevents torn views).

---

## Irreducible couplings (accept these — every shipped voxel engine pays them)

- Seams need the mesher to read a **halo** beyond the dirty region (data dependency, not
  leaked authority). Only LOD-transition boundaries need skirts; same-LOD is watertight.
- The mesher reads the current **MeshView** to weld patch-to-kept geometry.
- **LOD-DAG conservative mip:** an edited child cell makes coarse ancestors stale.
- **Colliders** are a second derived view needing a GPU→CPU readback.
- **Async latency:** render/collider views lag voxel truth by an epoch; carry a version token
  so a stale view is never a torn one.

## Done-ness

- Stage A done = sculpted geometry renders smooth, projection layer deleted, A5 passed.
- Full done = voxel-edit code references no mesh type; one `RemeshRegion` feed; everything
  ships as deltas; GPU mesher behind the same trait with an atomic swap.
