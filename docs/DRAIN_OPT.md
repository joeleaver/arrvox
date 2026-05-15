# Sculpt drain optimization plan

**Status:** D0 + D1 + D2 + D6.0 + D6.1 + D6.2 + D6.3.a + D6.3.b + D7
shipped. Post-D6.2 data showed `mesh` sub-phase still spiked to
9.7 ms on high-density stamps (40-50 k cells × 12 FxHash probes ≈
9-19 ms). D6.3 replaces the inner-loop HashMap probes with two dense
`CellGrid`s (one for `leaf_attr_id`, one for the cube → vertex cache)
sized to the brush footprint. D5 + D6.3.c still deferred — pending
measurement to confirm whether allocator pressure is real.

This plan picks up where `docs/PERF_DEBT.md` Phase E left off. After
Phase A–E, the sculpt per-stamp sim is **drain-bound at ~15 ms** —
the next lever lives inside `apply_sculpt_brush`
(`crates/rkp-render/src/rkp_scene_manager/sculpt.rs:205`), not in the
gpu-derive loop the perf-debt plan addressed.

---

## What `apply_sculpt_brush` does

Seven internal phases per stamp:

| # | Phase | Notes |
|---|---|---|
| 1 | Resolve grid coords | Sub-millisecond |
| 2 | `compute_brush_edits` | Walks brush region of octree |
| 3 | Resolve removes → leaf_attr_ids | Per-edit octree lookup |
| 4 | `apply_delta` | Mutates octree + brick pool |
| 5 | OctreeGpu sync | `try_extend_in_slack` / `apply_mutation_log` |
| 6 | Write/free LeafAttrs | Per-slot pool ops |
| 7 | `rebuild_dirty_clusters` | The big one — see below |

`rebuild_dirty_clusters` (Phase 7) has five sub-phases:

| # | Phase | Pre-D1 behaviour |
|---|---|---|
| 1 | Filter | Per-tri sphere test against ~240 k tris on splat5 elephant |
| 2 | Extract brush region | Surface Nets on brush volume |
| 3 | Append patch cluster | Tail-append to mesh_vertices / mesh_indices |
| 4 | CC walk | DAG-group walk marks dirty chains |
| 5 | Refresh dirty flags | Sum mesh_lod0_index_count, set dirty bits |

---

## Shipped

### D0 — per-phase timing (`0813cc17`)

Extends the existing `[sculpt] stamp …` and `[sculpt] V2 patch …`
log lines with per-phase ms breakdowns. No new log lines, no env-var
gating; the existing logs already fire every stamp.

Look for `[phases: …]` at the end of each log line. The dominant
sub-phase identifies the next drain target.

### D1 — cluster-AABB → brush-sphere rejection (`ad04b406`)

`rebuild_dirty_clusters` Phase 1 used to run the per-tri sphere test
for every triangle in every cluster `clusters_in_brush_grid_aabb`
returned — clusters are admitted by **box-vs-box** AABB overlap, but
the sphere may not actually touch all of them.

D1 adds a per-cluster sphere-AABB rejection: closest point on the
cluster's float AABB to the brush center; if outside `brush_radius`,
every triangle is kept via a single `indices[start..start+count].
to_vec()` (the per-tri test would have unanimously kept them anyway).

Correctness-preserving — output identical to pre-D1. New telemetry:
`(sphere_outside=N)` count in the `[sculpt] V2 patch` log.

### D2 — parallel filter (`2cbe2ed0`)

The per-cluster filter is embarrassingly parallel. Two-step:

1. `dirty.par_iter().map(|cid| ...).collect()` — rayon fans the
   per-tri test across the pool. D1's AABB rejection runs here.
2. Sequential merge — walks the `(cid, kept)` pairs in order,
   `extend_from_slice` into `mesh_indices` and writes new
   `index_offset` / `index_count` per cluster.

Order preserved (rayon `par_iter().collect()` is order-stable).
`d1_clusters_sphere_outside` counter migrated to `AtomicUsize` so
each rayon worker can bump it via `fetch_add(Relaxed)`.

### D6.0 — split extract timing into collect + mesh (`e8771047`)

Real measurements after D2 confirmed extract was the bottleneck
(10-18 ms per stamp) — but the phase has two distinct callees
(`collect_cell_map_in_region` and `extract_mesh_region_from_cells`)
and the lumped log line couldn't tell them apart. D6.0 splits the
timing so the next optimization can target the right one.

The `[sculpt] V2 patch …` log gains:

```
extract=X.XX (collect=A.AA cells=N mesh=B.BB)
```

`cells` is `HashMap::len()` of the collected solid set — useful
for sanity-checking the iteration cost (loop size scales with
`cells_count` after D6.1).

### D6.2 — `FxHashMap` for cells + cube_vertex (`009fe74a`)

D6.1's drag-stamp data showed `mesh` (the SN-vertex inner loop)
still ran 4-10 ms per stamp. The post-D6.1 inner loop does ~12
HashMap probes per solid cell (6 face neighbors + ~6 cube_vertex
lookups, plus `build_cube_vertex`'s 8 corner lookups for new
cubes). At ~27 k cells × 12 probes × ~50 ns std-SipHash ≈ 16 ms —
matches the observed budget. The HashMap itself was the bottleneck.

D6.2 replaces `HashMap<IVec3, u32>` with `rustc_hash::FxHashMap`
on both `cells` (per-stamp solid-cell occupancy) and `cube_vertex`
(SN-cube → vertex id cache). FxHash's single-multiply-mix is
~3-5× faster than std SipHash on 12-byte `IVec3` keys; no
DoS-resistance concern for internal data. New `pub type CellMap =
FxHashMap<IVec3, u32>` in `mesh_extract` captures the contract.

### D7 — spatial index for `clusters_in_brush_grid_aabb` (`bb51d050`)

The per-stamp `dirty_q` query (1.1-1.8 ms in the data) was a
linear scan over all 104 k LOD-0 clusters. D7 adds a per-asset
`ClusterSpatialIndex` — a bucket grid keyed by `IVec3` over the
finest-grid cell coords, divided by 50 cells (= 1 m at
base_vs = 0.02). Each LOD-0 cluster is inserted into every bucket
its grid AABB overlaps. Query walks the buckets touching the
brush AABB → unions cluster lists → exact AABB filter on the
small candidate set.

Maintenance: built at asset load, rebuilt on full re-extract,
incrementally updated on patch-cluster append. 5 unit tests
cover empty/LOD-filter/multi-bucket/incremental/empty-brush
behaviour. All 920+ workspace tests pass. Adds `rustc-hash` to
rkp-render deps (already a rkp-core dep from D6.2).

### D6.1 — iterate cells map directly in extract loop (`3662ad84`)

`extract_mesh_region_from_cells` walked the brush's padded bounding
box and ran `cells.contains_key` per cell to skip empties. For a
~50-cell brush radius, the box was ~58³ ≈ 195 k cells but only
~10 k were solid (matches ~10 k brush_patch verts in the data).
**95 % of iterations were wasted HashMap probes.**

D6.1 iterates `cells.iter()` directly — visits exactly the solid
set. Bounds check filters out the +2 outer ring kept in the map
purely for `build_cube_vertex` neighbor lookups at the iteration
boundary. HashMap iteration order is non-deterministic; resulting
vertex/index ordering inside the patch cluster differs but
triangles are independent (order has no visual or correctness
consequence).

Expected impact: extract phase 10-18 ms → 1-3 ms; total
`apply_sculpt_brush` 18 ms → 8-10 ms (hits the <8 ms success
criterion most of the time).

All 920+ workspace tests pass.

### D6.3.a + D6.3.b — dense `CellGrid` in extract inner loop (`5e29f8f5`, `772cb1a3`)

Post-D6.2 the `mesh` sub-phase still spiked to ~10 ms on
high-density stamps (40-50 k cells × 12 FxHash probes per cell ≈
9-19 ms). D6.3 replaces both internal FxHashMaps in
`extract_mesh_region_from_cells` with a pair of dense
`Vec<u32>`-backed grids sized to the brush footprint:

* `cells_grid` — `leaf_attr_id` lookup, replaces the
  `cells.contains_key(&neighbor)` probe (face-neighbor solidity
  test) and the `cells.get(&corner)` probes inside
  `build_cube_vertex`.
* `cube_vertex_grid` — SN-cube → vertex_id cache, replacing the
  per-stamp `FxHashMap<IVec3, u32>`.

Grid extent = `[pad_min - 1, pad_max + 1)` covers every coord the
inner loop probes (worked out from `FACE_DIRS`, `CUBE_OFFSETS_PER_FACE`,
and `corner_offset` ranges). For a 50-cell brush radius that's
~104³ ≈ 1.12 M entries × 4 bytes per grid = ~9 MB scratch. A grid
read is one bounds check + one indexed load (~3-5 ns) vs ~30 ns for
even FxHash on 12-byte `IVec3` keys.

`build_cube_vertex` is now generic over the cell-lookup primitive
(`F: Fn(IVec3) -> Option<u32>`) so the full-asset extract path can
keep using its `CellMap` (whose surface bbox makes densification
untenable) while the region path hands in a `CellGrid::get` closure.

D6.3.a adds the `CellGrid` type + 6 unit tests; D6.3.b wires it
into the region extract. All 932 workspace tests pass, including
`two_step_form_matches_convenience_wrapper` which verifies the
region path produces the same triangle set as the full extract.

Expected (per the D6.3 plan):
- 40 k-cell stamp `mesh` sub-phase: 9.71 ms → ~1.5-2 ms
- 40 k-cell stamp total: 21.68 ms → ~7-8 ms
- 16 k-cell stamp total: 7.27 ms → ~5-6 ms

Pending: drag-paint measurement on splat5 to confirm. If the
~9 MB scratch shows allocator pressure (1-2 ms per stamp on
alloc + memset), D6.3.c moves the grids onto `RkpSceneManager`
with grow-on-demand pool reuse.

---

## Deferred (still pending data)

### D3 — strip debug prints from hot path

The `[sculpt] stamp …`, `[sculpt] V2 patch …`, and engine-side
`[sculpt] stamp entity=…` `eprintln`s fire every stamp; stderr writes
serialize across threads. **Skipped for now** — D0's per-phase log
relies on them, and we want them on during the upcoming measurement
run. Revisit only if D0 reveals eprintln cost is non-trivial.

### D4 — reuse scratch allocations

Each filter iteration allocates a fresh `Vec<u32>` for the kept
indices (~80 allocs/stamp). With D2's `par_iter`, the natural
pattern is thread-local scratch via `rayon::ThreadPoolBuilder` —
but allocation cost was never measured. Defer until D0 says it
matters.

### D5 — `apply_delta` + OctreeGpu sync

Phases 4-5 of `apply_sculpt_brush`. Real numbers after D2:
`apply_delta` is 0.04-0.34 ms most stamps but spikes to 2.4-4.0 ms
on Raise stamps that allocate many new leaf cells. `octree_sync` is
0.01-0.03 ms (negligible). The 4 ms spikes are bursty and rare —
not the steady-state bottleneck. Defer until either steady-state
extract is fully optimized OR a sustained high-allocation workload
needs it.

### `dirty_q` (clusters_in_brush_grid_aabb spatial index)

✅ Shipped as D7 (`bb51d050`).

### D6.3.c — pool-reuse the CellGrid scratch

Per-stamp `CellGrid::new(.., size)` allocates ~9 MB and `memset`s it
to `u32::MAX`. Math: alloc ~50-200 µs + memset ~0.9 ms ≈ 1 ms per
stamp. The memset cost is unavoidable (grids must start empty) so
plain pool-reuse saves only the alloc overhead (~100-200 µs).

Defer until measurement shows the alloc is a hotspot. If it bites,
candidate designs:

1. **Plain reuse + memset.** Store the largest-needed `CellGrid`
   pair on `RkpSceneManager`; grow-on-demand, `memset(u32::MAX)`
   on reuse. Saves alloc cost only.
2. **Dirty-set reuse.** Track flat indices written during the stamp
   in a parallel `Vec<u32>`; on reuse, walk that list and reset only
   those slots. Cheaper reset (~12 µs for 30 k entries) but adds a
   push per `set` call (~500 µs on 50 k cells — net negative in the
   common case).
3. **Memcpy from a pre-cleared template.** Allocate a "blank" grid
   once at startup; on each stamp, memcpy from it. Same cost as the
   memset path — no win.

(1) is the only plausible candidate; whether it's worth the
SceneManager-state addition depends on the post-D6.3.b measurement.

### D6 follow-ups

If D6.3 doesn't fully resolve extract:
- **Drag-cache for `collect_cell_map_in_region`.** Brush moves
  slowly during a drag; consecutive stamps overlap heavily. Cache
  the cell map between stamps, invalidate only the brush footprint.
- **Parallelize `extract_mesh_region_from_cells`** via rayon over
  sub-volumes. Cube-vertex grid is shared state — would need
  either per-thread shards merged at the end or atomics.
- **Drop the +3 cell padding when the brush isn't near a
  boundary.** Currently the padding is unconditional.

---

## Out of scope

- **mesh_indices compaction over a drag**. Tail-appending kept tris
  every stamp grows `mesh_indices` unboundedly until full re-extract.
  This is a render-side upload concern (D-phase delta), not a sim
  drain concern.
- **Pick path round-trip** (~67 ms in project memory). GPU-side, not
  drain.
- **LeafAttrPool / brick pool growth** — already addressed by the
  shipped D-phase delta uploads.

---

## How to verify the shipped wins

Run a drag-paint session on splat5 elephant in release mode. Each
stamp emits two log lines with `[phases: …]` tails:

```
[sculpt] stamp handle=… mode=Raise edits=… removed=… applied(adds=… freed=… interior=…)
  (depth=…, base_vs=…) total=X.XXms
  [phases: resolve=… edits=… resolve_rm=… apply_delta=… octree_sync=… leaf_attr=… rebuild_clusters=…]

[sculpt] V2 patch: handle=… dirty=N (sphere_outside=M) kept_tris=… dropped_tris=…
  brush_patch verts=… tris=… total flat verts=… indices=…
  lod_dirty=…/… (…%) (X.XXms)
  [phases: setup=… dirty_q=… filter=… extract=… (collect=… cells=N mesh=…) append=… cc_walk=…]
```

Key numbers to track:

- **Outer `total`** — should drop from ~18 ms (pre-D6.1) to ~8-10 ms.
- **`rebuild_clusters`** — should drop from ~14 ms to ~4-6 ms.
- **Inner `extract`** — should drop from 10-18 ms to 1-3 ms.
- **`mesh` (within extract)** — the dominant sub-phase D6.1 attacks.
  Should drop substantially relative to its pre-D6.0 share of extract.
- **`cells`** — useful for sanity. With D6.1, loop cost scales with
  `cells` (was scaling with the full bounding box).
- **`sphere_outside / dirty`** — D1 rejection ratio. Highly variable
  (0%-80%) depending on brush position. Doesn't need to be high to
  be worthwhile.

---

## Success criteria

- Sculpt per-stamp sim: **<8 ms** on splat5 elephant drag (from 15 ms).
- D1 rejection ratio: **≥50%** of dirty clusters take the fast path.
- D2 filter wall-clock: **<1 ms** per stamp (down from suspected 5-10 ms).
- No regression in single-click stamps or asset-load full re-extract.

---

## Risks

- All shipped numbers assume the project-memory estimate of
  "11-19 ms internal" for `rebuild_dirty_clusters`. D0 may reveal the
  breakdown is different — e.g., if `apply_delta` is the dominant
  cost, D1/D2 (filter-focused) won't move the headline number and
  D5 becomes priority. Hence D5/D6 are explicitly deferred.

- D1 changes behaviour only when the AABB check is conservative
  (clusters near but outside the sphere). The math is symmetric with
  the per-tri test (vertices ⊂ AABB ⊂ outside-sphere ⇒ tris kept),
  but if a future change adds non-sphere brush shapes the AABB check
  must be revisited.

- D2's `par_iter` adds rayon worker spin-up overhead per stamp
  (~10s of µs). For small dirty sets this could be net-negative;
  D0 will show. If it bites, add a `dirty.len() < THRESHOLD` guard
  to fall back to serial.
