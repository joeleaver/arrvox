# Sculpt drain optimization plan

**Status:** D0 + D1 + D2 shipped (2026-05-14). D5 / D6 deferred until
the user runs a real drag-stamp session with the new `[sculpt-detail]`
logs and reports the per-phase breakdown.

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

---

## Deferred (need D0 measurements first)

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

Phases 4-5 of `apply_sculpt_brush` mutate the octree + brick pool
and sync the GPU packed buffer. The `try_extend_in_slack` / fallback
`deallocate + allocate_with_slack` paths plus `apply_mutation_log`
have non-trivial CPU work but no current breakdown. Targets if D0
shows them >2 ms each:
- Incremental log application instead of full sync.
- Batch the mutation log entries before applying.
- Track per-node mutations and skip identity rewrites.

### D6 — brush-region extract

`collect_cell_map_in_region` + `extract_mesh_region_from_cells`
(Phase 2 of `rebuild_dirty_clusters`). Surface Nets bounded by brush
volume — should be small but worth measuring. Optimization
opportunities if D0 shows it dominates:
- Cache cell map across consecutive overlapping stamps in a drag.
- Parallelize SN over sub-volumes.
- Drop the +3 cell padding when not at a brush boundary.

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
stamp emits two log lines with the new `[phases: …]` tails:

```
[sculpt] stamp handle=… mode=Carve edits=… removed=… applied(adds=… freed=… interior=…)
  (depth=…, base_vs=…) total=X.XXms
  [phases: resolve=… edits=… resolve_rm=… apply_delta=… octree_sync=… leaf_attr=… rebuild_clusters=…]

[sculpt] V2 patch: handle=… dirty=N (sphere_outside=M) kept_tris=… dropped_tris=…
  brush_patch verts=… tris=… total flat verts=… indices=…
  lod_dirty=…/… (…%) (X.XXms)
  [phases: setup=… dirty_q=… filter=… extract=… append=… cc_walk=…]
```

Read `rebuild_clusters` from the outer log + `filter` from the inner
to see the cluster-patch filter cost. `sphere_outside / dirty` is the
D1 rejection ratio (target: ≥50% on a typical drag stamp).

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
