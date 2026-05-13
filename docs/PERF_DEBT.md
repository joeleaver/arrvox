# Engine performance-debt eradication plan

**Status**: in progress (Phase A complete; B1 + B2 + C2 + C3 shipped; B3 + C1 + Phase D/E next). Feature work paused.

This document is the authoritative plan for eliminating systemic "rebuild
everything every tick" patterns from rkp-engine and rkp-render. It was
written on 2026-05-13 after a sculpt-latency session surfaced that the
real bottleneck was not in any single phase but in the architecture's
shape: the sim treats every mutation as "the world might have changed"
and rebuilds all derived state from scratch.

The framing is intentional: **infinite time, infinite budget — do it
right, not fast**. Each phase ends with a measurable improvement
backed by telemetry. No phase ships feature work.

---

## What we're fixing

Per-stamp sim wall time (steady state, post-stopgaps from 2026-05-13):

| Phase | Cost | Cause |
|---|---|---|
| `drain_render_results` | ~15 ms | pick-result processing + sculpt mutation |
| `update_scene_gpu` | 60-75 ms | rebuilds `gpu_instances` + overlays + sculpts + splat_draws + proxy_draws for **all** entities |
| `walk_snapshot()` clone | ~80 ms | clones entire octree + brick pool + leaf_attr pool (~345 MB memcpy) |
| `scan_painted_aabbs` | ~80 ms | walks the affected entity's full octree to find shader-bearing painted leaves |
| other | <1 ms | snapshot construction housekeeping |
| **total** | **~240 ms** | for a mutation that touched ~80 LOD-0 clusters on 1 entity out of 22 |

Per-frame render uploads (steady state):

| Buffer | Size | Fraction fresh |
|---|---|---|
| Material palette | ~1 KB | ~0% |
| Lights list | ~2 KB | ~0% |
| Shader params slots | 32 B × materials × VPs | ~0% |
| Bone matrix palette | up to ~58 MB | variable, often 0 |
| `gpu_instance_overlays` | up to 10s MB | small fraction |
| `gpu_instance_sculpts` | up to 10 MB | small fraction |
| `gpu_assets` / `gpu_instances` | ~30 KB | 10-50% in typical scenes |

These all use `ensure_and_write` (full buffer rewrite). They predate the
delta-upload work that landed for the octree/brick/leaf_attr pools.

---

## The 5 architectural principles

### 1. Data is owned in one place, shared by reference

Large mutable state lives in `RkpSceneManager`. Other threads see it
through `Arc<T>` where `T` is internally structured for safe concurrent
reads with generation/epoch versioning.

- No `.to_vec()` "for snapshot."
- No `.clone()` of large `Vec`s for cross-thread handoff.
- Generation counters disambiguate reader/writer views.

### 2. Every mutation describes its scope

No bare `dirty: bool`. Every mutation produces a typed event carrying
the affected scope (entity, region, buffer slot).

- `gpu_objects_dirty: bool` becomes `HashSet<Entity>` (or richer events).
- `geometry_dirty`, `scene_dirty`, `collider_caches_dirty` same.
- Setters pass the affected entity at the call site.
- Consumers iterate the scope, not the world.

### 3. Derived state is maintained, not rebuilt

Every derived structure has two paths: a constructor (used at asset
load / project open) and an incremental updater (used per mutation
event). The hot loop only takes the second path.

- `gpu_instances`: per-row update via `update_scene_gpu_entity(entity)`.
- `painted_per_entity`: paint stamps add to `mat_tiles`; sculpt-Carve
  removes evicted leaves; **no octree walk in steady state**. Walk
  retained only for asset-load initial scan.
- `collider_caches`: per-entity rebuild on geometry-mutation event.
- `SceneObjectInfo`: maintained list, updated incrementally on
  add/remove/rename/reparent.

After this principle is enforced, no "full rebuild" path exists in
steady state.

### 4. GPU uploads are always deltas

Every GPU buffer has CPU-side `DirtyRanges`. Uploads convert ranges
to `queue.write_buffer` calls. Full upload is reserved for the
"initial alloc" branch when a `wgpu::Buffer` is first created.

Already done for octree/brick/leaf_attr (D5-D10) and the cluster
table (Option B). Apply universally to bone matrix, overlays, sculpts,
material palette, lights, shader params.

### 5. Sim and render are decoupled in time

The sim never blocks on derived-state freshness. Submit a
`RenderFrame` whenever a geometry change is ready; derived state
catches up on subsequent frames.

- Painted-walk, collider rebuild, snapshot construction run on worker
  threads.
- Snapshots are `arc-swap`ed atomically; sim and render never serialize
  on each other.
- Load-bearing invariant: GPU buffers must always be consistent with the
  `geometry_epoch` the render is rendering. All other derived state
  may lag by 1-2 frames invisibly.

---

## Migration phases

Each phase ends with: passing tests + telemetry confirmation the target
metric moved.

### Phase A — Foundations

| Step | Change | Result | Files |
|---|---|---|---|
| **A1** | `MutationEvent` enum + log/subscriber scaffolding in EngineState. All mutation sites push events. No consumers yet. Existing dirty flags stay. | Universal scope-carrying mutation API. | `crates/rkp-engine/src/engine/mutation_log.rs` (new), all `*_ops.rs` |
| **A2 ✅** | Pools (`OctreeAllocator`, `BrickPool`, `LeafAttrPool`) now hold `data: Arc<Vec<…>>` with copy-on-write via `Arc::make_mut`. `walk_snapshot()` is O(1) Arc::clone of three handles, plus the geometry epoch as a generation counter. Cache fields (`walk_snapshot_cache`, `walk_snapshot_epoch`) deleted — no longer needed. **In steady state (no outstanding snapshot) writes are in-place**; an outstanding walk forces a one-time clone-on-next-write. | -345 MB memcpy per epoch bump | `crates/rkp-core/src/{brick_pool,leaf_attr_pool,octree_allocator}.rs`, `crates/rkp-render/src/{octree_gpu,rkp_scene_manager/manager}.rs`, `crates/rkp-engine/src/engine/lifecycle.rs` |
| **A3 ✅** | `RenderFrame` large fields become `Arc<...>`: `bone_matrix_lbs/dqs` (BoneMatrixAllocator now stores `Arc<Vec<u8>>`, mutations via `Arc::make_mut`); `gpu_assets`/`gpu_instances`/`gpu_instance_overlays`/`gpu_instance_sculpts`/`splat_draws`/`proxy_draws` (EngineState fields now `Arc<Vec<…>>`, mutations via `Arc::make_mut` in `update_scene_gpu` + `clear_scene`); `user_shader_entries` (UserShaderRegistry now stores entries as `Arc<Vec<…>>`, parser builds via local Vec then wraps). Snapshot construction is all `Arc::clone`. Render's interp path borrows when α=1 instead of cloning. | -58 MB/frame (bone matrix) + ~280 KB/frame (gpu_*, user_shader_entries) clone in steady state | `crates/rkp-engine/src/{render_frame,scene_sync}.rs`, `crates/rkp-engine/src/engine/{lifecycle,scene_gpu,entity_ops,state/{mod,constructor}}.rs`, `crates/rkp-engine/src/render_worker/loop_thread.rs`, `crates/rkp-render/src/shader_composer/{types,parser,compose}.rs` |

### Phase B — Per-entity dirty sets

| Step | Change | Result |
|---|---|---|
| **B1 ✅ (plumbing)** | `gpu_objects_dirty: bool` → `GpuObjectsDirty` (HashSet + sticky-all). 31 setter sites migrated: hot stamp paths (sculpt/paint/gizmo/picking/proc-bake) narrowed to `mark_entity(e)`; world-level events (project load, scene clear, gameplay reset, etc.) keep `mark_all()`. **Today's consumer still does a full rebuild whenever `is_dirty()`** — the per-row fast path is the C2 work. Module: `crates/rkp-engine/src/engine/gpu_objects_dirty.rs`. | Plumbing only; perf delivers in C2. |
| **B2 ✅** | `geometry_dirty` and `collider_caches_dirty` both replaced with `GeometryDirty` (HashSet + sticky-all). New module `engine/geometry_dirty.rs`. Lifecycle drains `geometry_dirty` → `collider_caches_dirty` per-entity. Procedural-bake completion narrowed to `mark_entity(e)`; world-level events keep `mark_all()`. | Per-entity collider rebuild path (delivered together with C3 below). |
| **B3** | `scene_dirty: bool` → typed event stream consumed by UI/inspector snapshot builder. | UI updates incrementally |

### Phase C — Incremental derived state

| Step | Change | Result |
|---|---|---|
| **C1** | `painted_per_entity` maintained via paint/sculpt event handlers; no octree walk in steady state. Walk retained for asset load. | -150 ms/stamp painted_walk |
| **C2 ✅ (transform fast path)** | `update_scene_gpu_transform_only` patches just `RkpGpuInstance.world` (and matching `SplatDraw.world` / `ProxyDraw.world`) for each Transform-dirty entity. Lifecycle gates on `gpu_objects_dirty.is_transform_only()`: gizmo/drag stamps now run the fast path; sculpt/paint/proc-bake/scene-load still go through the full rebuild. New `DirtyKind` enum (`Transform`/`Structural`) on `GpuObjectsDirty` with `mark_entity_transform` / `mark_entity` / `mark_entity_kind`. | -60+ ms per gizmo-drag stamp on splat5 elephant (full rebuild → in-place row patch). Structural dirty events still pay the full rebuild — generalising that is a future follow-up. |
| **C3 ✅** | `rebuild_collider_cache_for(entity)` extracted from the world-walking `rebuild_collider_caches`. Lifecycle iterates `collider_caches_dirty.dirty_entities()` per-entity when scope is narrow; falls back to the world walk only when `is_all()` (project load, asset import, generator regen). Note: today no per-stamp setter triggers `geometry_dirty` (sculpt defers collider rebuild to play-mode entry — intentional), so the per-stamp savings here are latent until a future setter narrows. | O(1) per changed entity in narrow path; world walk reserved for `all`. |

### Phase D — Universal delta GPU uploads

| Step | Change | Result |
|---|---|---|
| **D1** | Bone matrix allocator gains DirtyRanges per bone subrange. Skin systems mark which bones moved. | -58 MB upload/frame |
| **D2** | `gpu_instance_overlays` per-instance DirtyRanges. Paint/sculpt stamps mark the affected instance's range. | Overlay uploads scale with stamps, not scene size |
| **D3** | `gpu_instance_sculpts` same. | Same for sculpt overlay |
| **D4** | Material palette / lights / shader_params hash-gated upload. | -3 small uploads/frame |

### Phase E — Cross-thread decoupling (optional but ideal)

| Step | Change | Result |
|---|---|---|
| **E1** | Painted-walk on worker thread. Sim submits RenderFrame without waiting; walk result lands in next snapshot. | Sim never blocks on walk |
| **E2** | Collider rebuild on worker thread. | Sim never blocks on collider |
| **E3** | `arc-swap` for atomic snapshot-swap of heavy state. | Render reads lock-free |

---

## Success criteria

- Per-frame heap allocation **~0 bytes** in steady state.
- Per-tick wall time **~1-2 ms** when world is at rest, regardless of world size.
- Sculpt stamp click→visible **<100 ms** steady state.
- Single entity transform: **O(1) sim work**, <1 ms.
- **No code path** does "rebuild N because flag is true."
- Per-stamp GPU upload total: **<1 MB** for any single-entity mutation.

---

## What we keep (no migration needed)

- Mesh + cluster DAG (today's commits `6eab1573`, `2357bc09`, `256b3352`).
- Pool delta upload infrastructure (D5-D10).
- Option B in-place cluster table upload.
- `[sculpt-pipeline]` + `[sculpt-pipeline-sim]` telemetry — keep until
  each migration phase verifies its improvement, then prune.

## What replaces what (stopgaps → real fixes)

The 2026-05-13 sculpt session shipped several stopgaps that will be
replaced by the migration:

| Stopgap (today) | Replaced by | Phase |
|---|---|---|
| `entities_known_empty` cache for painted_walk | Incremental `painted_per_entity` maintenance | C1 |
| Narrow `painted_dirty_entities` to sculpted entity | Same — keep, generalize | C1 |
| `[sculpt-pipeline-sim]` phase log | Keep as telemetry; remove when all phases verified |  |

---

## Audit findings (raw, with file:line citations)

### Full-world rebuild sites (sim-side)

- **`update_scene_gpu`** — `crates/rkp-engine/src/engine/scene_gpu.rs` and called from `crates/rkp-engine/src/engine/lifecycle.rs:156-159`. Iterates all entities with `Renderable`, rebuilds `gpu_assets`, `gpu_instances`, `gpu_instance_overlays`, `gpu_instance_sculpts`, `splat_draws`, `proxy_draws`, `gpu_to_entity`, `entity_to_gpu`. Triggered by 13+ setter sites for `gpu_objects_dirty`.
- **`bone_matrix_allocator.rebuild()`** — `crates/rkp-engine/src/engine/scene_gpu.rs:18`. Runs unconditionally every `submit_render_frame` tick.
- **`bone_matrix.bytes().to_vec()` clones** — `crates/rkp-engine/src/engine/lifecycle.rs:712-713`. ~58 MB memcpy per frame. **(Resolved in A3.)** Now `bytes_arc()` returns an `Arc<Vec<u8>>` cloned from the allocator's internal storage; the allocator's `rebuild()` uses `Arc::make_mut` so the COW only fires when render still holds last frame's snapshot, and the immediate `.clear()` after means we never copy stale payload.
- **GPU lights walk** — `crates/rkp-engine/src/engine/lifecycle.rs:613-647`. Queries every `PointLight` / `SpotLight` every tick.
- **`rebuild_collider_caches`** — `crates/rkp-engine/src/engine/gizmo_ops.rs:379-477`. Iterates all entities with `RigidBody`, runs `compute_tight_local_aabb` + `build_coarse_collider` per entity. Triggered by `geometry_dirty` (8 setter sites) → `collider_caches_dirty`.
- **`scene_dirty` → SceneObjectInfo rebuild** — `crates/rkp-engine/src/engine/state_update.rs:298`. Sorts + rebuilds the full UI scene list. 7 setter sites.

### Full upload sites (render-side)

- **Material palette** — `crates/rkp-render/src/rkp_renderer.rs:2455-2470`. Full upload every frame.
- **Lights** — `crates/rkp-render/src/rkp_renderer.rs:2434-2453`. Full upload every frame.
- **Shader params slots** — `crates/rkp-render/src/rkp_shade.rs:539-568`. Full upload every frame per viewport.
- **Bone matrix palette** — `crates/rkp-render/src/rkp_scene.rs:755-760`. `ensure_and_write` (full buffer rewrite).
- **`gpu_instance_overlays`** — `crates/rkp-render/src/rkp_scene.rs:712-724`. `ensure_and_write`.
- **`gpu_instance_sculpts`** — `crates/rkp-render/src/rkp_scene.rs:731-743`. `ensure_and_write`.
- **`gpu_assets` / `gpu_instances`** — `crates/rkp-render/src/rkp_scene.rs:748-749`. `ensure_and_write`.

### Coarse dirty flags

- **`gpu_objects_dirty: bool`** — `crates/rkp-engine/src/engine/state/mod.rs:658`. 13 setter sites across sculpt/paint/gameplay/gizmo/scene_tree/picking/procedural/scene_io/cmd_edit ops. Consumer: full `update_scene_gpu`. **(B1 plumbing resolved — replaced with `GpuObjectsDirty` carrying per-entity scope. Actual setter count was 31. Hot stamp paths now `mark_entity(e)`. Consumer still does full rebuild until C2.)**
- **`geometry_dirty: bool`** — `crates/rkp-engine/src/engine/state/mod.rs:654`. 8 setter sites. Triggers `collider_caches_dirty`. **(B2 plumbing resolved — replaced with `GeometryDirty` carrying per-entity scope. Actual setter count was 14.)**
- **`scene_dirty: bool`** — `crates/rkp-engine/src/engine/state/mod.rs:656`. 7 setter sites. Triggers full SceneObjectInfo rebuild.
- **`collider_caches_dirty: bool`** — `crates/rkp-engine/src/engine/state/mod.rs:594`. Derivative of `geometry_dirty`. **(B2+C3 resolved — same `GeometryDirty` type; per-entity rebuild via `rebuild_collider_cache_for(entity)`.)**
- **`faces_dirty: bool`** — `crates/rkp-render/src/rkp_scene_manager/manager.rs:93`. **5 setter sites, ZERO consumer sites** — orphaned flag. Either wire a consumer or remove.

### Heavy clone patterns

- **`walk_snapshot()`** — `crates/rkp-render/src/rkp_scene_manager/manager.rs:437-442`. ~345 MB memcpy (octree + brick pool + leaf_attr). **(Resolved in A2.)** Pools were refactored to hold their data behind `Arc<Vec<…>>`; walk_snapshot is now three constant-time `Arc::clone`s, with copy-on-write (`Arc::make_mut`) on the next mutation if a snapshot is still outstanding.
- **`gpu_*.clone()` in submit_render_frame** — `crates/rkp-engine/src/engine/lifecycle.rs:1197-1202`. ~230 KB/frame. **(Resolved in A3.)** Each of the six fields is now `Arc<Vec<…>>`; snapshot construction is `Arc::clone`. Mutations route through `Arc::make_mut`, paying a one-time copy-on-write only when render still holds last frame's snapshot.
- **`bone_matrix_lbs/dqs.to_vec()`** — `crates/rkp-engine/src/engine/lifecycle.rs:712-713`. ~58 MB/frame.
- **`SkinBatchScratch.clone()`** — `crates/rkp-engine/src/engine/lifecycle.rs:737`. ~36 KB/frame.
- **`user_shader_entries.to_vec()`** — `crates/rkp-engine/src/engine/lifecycle.rs:114`. ~50 KB/frame. **(Resolved in A3.)** UserShaderRegistry stores entries as `Arc<Vec<UserShaderEntry>>`; sim calls `entries_arc()` and ships the Arc::clone.

---

## Discipline

- Each commit is scoped to one phase step.
- No feature work merges until the migration is complete.
- Telemetry (`[sculpt-pipeline]`, `[sculpt-pipeline-sim]`, `[geo-epoch]`, `[delta upload]`) stays in tree throughout the migration; it is how we verify each phase delivered.
- Stopgaps shipped during the surgery (today's commits `1f8852b9` through `e8377f49`) stay until their replacement migration step lands; then we remove the stopgap in the same commit as the replacement.
