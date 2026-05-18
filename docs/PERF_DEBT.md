# Engine performance-debt eradication plan

**Status**: complete (Phase A + B + C + D + E all shipped). Feature work may resume.

This document is the authoritative plan for eliminating systemic "rebuild
everything every tick" patterns from arvx-engine and arvx-render. It was
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

Large mutable state lives in `ArvxSceneManager`. Other threads see it
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
| **A1** | `MutationEvent` enum + log/subscriber scaffolding in EngineState. All mutation sites push events. No consumers yet. Existing dirty flags stay. | Universal scope-carrying mutation API. | `crates/arvx-engine/src/engine/mutation_log.rs` (new), all `*_ops.rs` |
| **A2 ✅** | Pools (`OctreeAllocator`, `BrickPool`, `LeafAttrPool`) now hold `data: Arc<Vec<…>>` with copy-on-write via `Arc::make_mut`. `walk_snapshot()` is O(1) Arc::clone of three handles, plus the geometry epoch as a generation counter. Cache fields (`walk_snapshot_cache`, `walk_snapshot_epoch`) deleted — no longer needed. **In steady state (no outstanding snapshot) writes are in-place**; an outstanding walk forces a one-time clone-on-next-write. | -345 MB memcpy per epoch bump | `crates/arvx-core/src/{brick_pool,leaf_attr_pool,octree_allocator}.rs`, `crates/arvx-render/src/{octree_gpu,arvx_scene_manager/manager}.rs`, `crates/arvx-engine/src/engine/lifecycle.rs` |
| **A3 ✅** | `RenderFrame` large fields become `Arc<...>`: `bone_matrix_lbs/dqs` (BoneMatrixAllocator now stores `Arc<Vec<u8>>`, mutations via `Arc::make_mut`); `gpu_assets`/`gpu_instances`/`gpu_instance_overlays`/`gpu_instance_sculpts`/`splat_draws`/`proxy_draws` (EngineState fields now `Arc<Vec<…>>`, mutations via `Arc::make_mut` in `update_scene_gpu` + `clear_scene`); `user_shader_entries` (UserShaderRegistry now stores entries as `Arc<Vec<…>>`, parser builds via local Vec then wraps). Snapshot construction is all `Arc::clone`. Render's interp path borrows when α=1 instead of cloning. | -58 MB/frame (bone matrix) + ~280 KB/frame (gpu_*, user_shader_entries) clone in steady state | `crates/arvx-engine/src/{render_frame,scene_sync}.rs`, `crates/arvx-engine/src/engine/{lifecycle,scene_gpu,entity_ops,state/{mod,constructor}}.rs`, `crates/arvx-engine/src/render_worker/loop_thread.rs`, `crates/arvx-render/src/shader_composer/{types,parser,compose}.rs` |

### Phase B — Per-entity dirty sets

| Step | Change | Result |
|---|---|---|
| **B1 ✅ (plumbing)** | `gpu_objects_dirty: bool` → `GpuObjectsDirty` (HashSet + sticky-all). 31 setter sites migrated: hot stamp paths (sculpt/paint/gizmo/picking/proc-bake) narrowed to `mark_entity(e)`; world-level events (project load, scene clear, gameplay reset, etc.) keep `mark_all()`. **Today's consumer still does a full rebuild whenever `is_dirty()`** — the per-row fast path is the C2 work. Module: `crates/arvx-engine/src/engine/gpu_objects_dirty.rs`. | Plumbing only; perf delivers in C2. |
| **B2 ✅** | `geometry_dirty` and `collider_caches_dirty` both replaced with `GeometryDirty` (HashSet + sticky-all). New module `engine/geometry_dirty.rs`. Lifecycle drains `geometry_dirty` → `collider_caches_dirty` per-entity. Procedural-bake completion narrowed to `mark_entity(e)`; world-level events keep `mark_all()`. | Per-entity collider rebuild path (delivered together with C3 below). |
| **B3 ✅ (plumbing)** | `scene_dirty: bool` → `SceneDirty` (HashSet + sticky-all), matching the B1/B2 shape. 23 setter sites migrated: spawn / delete / duplicate / reparent / component add+remove → `mark_entity(e)`; scene load / project load / clear / gameplay register / enter-mode → `mark_all()`. Today's consumer in `build_state_update` still does the full sorted rebuild on `is_dirty()`; the per-entity scope is foundation for a future delta-protocol path that sends only Added/Removed/Renamed/Reparented rows across the sim→editor boundary. Module: `crates/arvx-engine/src/engine/scene_dirty.rs`. | Plumbing only — perf delivers when a future narrow consumer lands. |

### Phase C — Incremental derived state

| Step | Change | Result |
|---|---|---|
| **C1 ✅** | Region-bounded `painted_walk`. New `painted_dirty_regions: HashMap<Entity, Vec<Aabb>>` captures each stamp's world-space brush AABB at `apply_paint_stamp` (Material mode) and `apply_sculpt_stamp` (Raise + Carve). The lifecycle walk transforms each region into object-local space, clears overlapping tile entries per material (tile-coord range = `floor(local_dirty * inv_tile)..=floor(local_dirty.max * inv_tile)`), and re-fills via `scan_painted_aabbs_clipped` — an octree descent that bails at any node whose AABB doesn't intersect `local_dirty + max(@tile_size)` and that filters per-tile inserts against `local_dirty` so smaller-tile-size materials outside their cleared range aren't double-counted. Asset-load / geometry-epoch invalidation populates `painted_dirty_entities` without regions → falls back to the full walk. Any shader material with `tile_size=None` also forces full-walk (the NO_TILE_COORD AABB spans the entire entity and can't survive a clipped rebuild). | -160 ms/stamp painted_walk on splat5 elephant: ~80 ms → 0.04 ms steady state (2000×). Stopgap `entities_known_empty` removed in the same commit. |
| **C2 ✅ (transform fast path)** | `update_scene_gpu_transform_only` patches just `ArvxGpuInstance.world` (and matching `SplatDraw.world` / `ProxyDraw.world`) for each Transform-dirty entity. Lifecycle gates on `gpu_objects_dirty.is_transform_only()`: gizmo/drag stamps now run the fast path; sculpt/paint/proc-bake/scene-load still go through the full rebuild. New `DirtyKind` enum (`Transform`/`Structural`) on `GpuObjectsDirty` with `mark_entity_transform` / `mark_entity` / `mark_entity_kind`. | -60+ ms per gizmo-drag stamp on splat5 elephant (full rebuild → in-place row patch). |
| **C2-narrow ✅ (sculpt fast path)** | Two-pronged. (a) New `update_scene_gpu_structural_narrow` — per-entity splice of `paint_overlays` and `sculpt_overlays` into the flat GPU vecs + suffix-shift of subsequent rows' offsets. Skips bone_matrix repack, skin replan, asset table dedupe, world-wide query. Dispatch tries narrow first; falls back to full rebuild only on `is_all()` or when a dirty entity isn't yet in `entity_to_gpu`. (b) `asset_has_glass_cache` invalidation moved from blanket-on-geom-epoch (was firing every sculpt → ~50 ms rescan of 2.5M leaves) to call-site-targeted: `remap_entity_material` and sculpt-Raise-with-glass-brush each drop their single asset's cache entry. Sculpt-Carve never adds glass, so a stale-true verdict is just a wasted glass pass (no correctness issue). (c) `animation::tick` skips paused players entirely so their identity-evaluated pose doesn't fire `mark_all` every tick. | **-60 ms per stamp on splat5 elephant: update_scene_gpu 60-172 ms → 0.10-0.15 ms** (~500-1000×). Whichever dispatch path runs is now cheap. Total per-stamp sim ~15 ms (was ~80-240 ms), dominated by `drain_render_results` (the sculpt mutation itself). |
| **C3 ✅** | `rebuild_collider_cache_for(entity)` extracted from the world-walking `rebuild_collider_caches`. Lifecycle iterates `collider_caches_dirty.dirty_entities()` per-entity when scope is narrow; falls back to the world walk only when `is_all()` (project load, asset import, generator regen). Note: today no per-stamp setter triggers `geometry_dirty` (sculpt defers collider rebuild to play-mode entry — intentional), so the per-stamp savings here are latent until a future setter narrows. | O(1) per changed entity in narrow path; world walk reserved for `all`. |

### Phase D — Universal delta GPU uploads

| Step | Change | Result |
|---|---|---|
| **D1 ✅** | `BoneMatrixAllocator` gains per-entity dirty ranges. Entities sorted by `Entity::to_bits()` for stable layout across rebuilds; layout-equality check decides per-entity-delta vs `mark_full` fallback. Each entity's slot compared bone-by-bone against the previous frame's pose (`Vec<Mat4>` equality, exact-match); identical → slot dropped from dirty ranges → render side skips its upload. `ArvxScene::upload_frame` routes via new `write_with_dirty` helper: empty ranges → skip; `is_full_pool` → `ensure_and_write`; else per-range `queue.write_buffer`. Telemetry: `ARVX_BONE_UPLOAD_PROFILE=1` logs `[bone-upload] mat=… ({n} ranges) dq=… total_buf=…` per frame. | Validated on splat5 (Walking ×2 + CesiumMan, ~23 KiB buffer total): every-frame upload still ~full when all 3 animate continuously, but slots correctly drop out when any pose is bit-identical to last frame (saw 16.25 KiB on the frames CesiumMan's pose held). Bigger wins on scenes with paused/static skeletons or many low-frequency animations. The C2-narrow path (no bone rebuild) now also produces zero-byte uploads instead of full-buffer rewrites. |
| **D2 ✅** | `gpu_instance_overlays_dirty: bool` on `EngineState`. Set by paint stamps (any mode — Color/Erase also mutate the overlay), entity removes that had a non-empty overlay, and `clear_scene`. Snapshot converts to `arvx_core::DirtyRanges` with `mark_full(buf_len * 16 B)`; render-side routes through `ArvxScene::write_with_dirty` (D1's helper). Empty bool → empty ranges → `write_with_dirty` skips. `FrameUpload` gains `instance_overlays_dirty: &DirtyRanges`. | Idle frames between paint stamps now skip the overlay upload entirely (was `ensure_and_write` of the full buffer every tick). Stamp frames still do one full upload — a future enhancement could narrow that to the spliced range, but the C2-narrow splice already shifts the tail so the per-stamp delta is at best ~half the buffer; the bigger win was the "every idle frame" rewrite. |
| **D3 ✅** | Same shape as D2 for `gpu_instance_sculpts`. `gpu_instance_sculpts_dirty: bool` bumped by sculpt stamps + sculpt-bearing entity removes + clear_scene. | Idle-frame skip for the sculpt buffer (was ~13 K × 4 B = 52 KiB ensure_and_write per frame on the splat5 elephant after a drag, regardless of motion). |
| **D4 ✅** | Hash-gated uploads on three small buffers: `update_lights` and `update_materials` (ArvxRenderer-owned, ~1-2 KiB each) cache a `last_*_hash: u64` and short-circuit the `queue.write_buffer` when the incoming bytes match. `upload_shader_params` (per-VR ArvxShade-owned, ~32 B × material count) does the same. Realloc / buffer-grow paths always write (different buffer); the hash is reset there so the next identical-content tick still short-circuits. Helper: `d4_hash_bytes` (`DefaultHasher` — small enough that the ~µs hash cost dominates over `queue.write_buffer` driver overhead). | Skips 3 small writes per frame in steady state. Quiet win (small absolute bytes, but every frame counts in idle). Verified live in the editor: scene load + animation playback complete without runtime issues; hash gates are structurally identical to D1's tested path. |

### Phase E — Cross-thread decoupling (shipped)

| Step | Change | Result |
|---|---|---|
| **E1 ✅** | Painted-walk on dedicated worker thread (`engine/paint_walk.rs`). Sim drains `painted_dirty_entities` + `painted_dirty_regions` into a `PaintWalkBatch`, submits via bounded(1) crossbeam inbox, merges the `PaintWalkResult` on a subsequent tick. Three correctness twists vs naive offload: (a) don't pre-clear `painted_per_entity` on geom-bump blanket-invalidate — old entries stay through the worker's in-flight window so the flat rebuild keeps producing anchors at last-known positions; (b) clone (don't move) the existing per-entity cache when building each job; (c) track `painted_walk_submitted_geom_epoch` to suppress redundant blanket-invalidations while a batch is in flight. Result-merge also filters out entries for entities that were despawned mid-batch. | Sim never blocks on the painted-material walk. `submit_render_frame` no longer contains an O(dirty-tree) walk at all. |
| **E2 ✅** | Collider rebuild on dedicated worker thread (`engine/collider_worker.rs`). Mirrors E1's shape: bounded(1) crossbeam inbox + `in_flight` atomic. Sim drains `collider_caches_dirty` (or world-walks every RigidBody on `is_all()`), captures per-entity inputs into `ColliderJob`s, submits with a `WalkSnapshot`, inserts the resulting `ColliderCache` ECS components after `try_recv`. Refactored `compute_tight_local_aabb` + `build_coarse_collider` to take `brick_pool: &[u32]` (the same data the `BrickPool` holds inside its `Arc<Vec<u32>>`) instead of `&BrickPool` so the compute is worker-callable. Synchronous `rebuild_collider_caches` remains for the `PlayStart` handler in `cmd_runtime` — play-mode entry needs collider caches present immediately. | Sim never blocks on collider rebuild. (Latent today — no per-stamp setter touches `geometry_dirty`; defensive scaffolding for future setters.) |
| **E3 ✅** | `RenderInbox` slot moves from `Mutex<Option<RenderFrame>>` to `arc_swap::ArcSwapOption<RenderFrame>`. Steady-state `try_take` (called every render iteration after bootstrap) is now lock-free — one atomic swap, no mutex. Bootstrap `take_blocking` + `submit` still use a tiny `Mutex<()> + Condvar` for the wakeup signal (held only across `notify_one` / pre-wait recheck, never across frame transit). SPSC invariant guarantees `Arc::try_unwrap` succeeds in steady state; `unwrap_inbox_arc` defensively spin+yields a few times before panicking. | Render reads frame slot lock-free. Closes the principle "sim and render are decoupled in time" by removing the last shared lock from the steady-state handoff path. |

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
| ~~`entities_known_empty` cache for painted_walk~~ | Region-bounded `painted_walk` makes the "no shader materials" path naturally fast (~0.04 ms) without the cache | C1 ✅ (removed in the same commit) |
| Narrow `painted_dirty_entities` to sculpted entity | Kept + generalized — `painted_dirty_regions` now also carries the brush footprint | C1 ✅ |
| `[sculpt-pipeline-sim]` phase log | Keep as telemetry; remove when all phases verified |  |

---

## Audit findings (raw, with file:line citations)

### Full-world rebuild sites (sim-side)

- **`update_scene_gpu`** — `crates/arvx-engine/src/engine/scene_gpu.rs` and called from `crates/arvx-engine/src/engine/lifecycle.rs:156-159`. Iterates all entities with `Renderable`, rebuilds `gpu_assets`, `gpu_instances`, `gpu_instance_overlays`, `gpu_instance_sculpts`, `splat_draws`, `proxy_draws`, `gpu_to_entity`, `entity_to_gpu`. Triggered by 13+ setter sites for `gpu_objects_dirty`.
- **`bone_matrix_allocator.rebuild()`** — `crates/arvx-engine/src/engine/scene_gpu.rs:18`. Runs unconditionally every `submit_render_frame` tick.
- **`bone_matrix.bytes().to_vec()` clones** — `crates/arvx-engine/src/engine/lifecycle.rs:712-713`. ~58 MB memcpy per frame. **(Resolved in A3.)** Now `bytes_arc()` returns an `Arc<Vec<u8>>` cloned from the allocator's internal storage; the allocator's `rebuild()` uses `Arc::make_mut` so the COW only fires when render still holds last frame's snapshot, and the immediate `.clear()` after means we never copy stale payload.
- **GPU lights walk** — `crates/arvx-engine/src/engine/lifecycle.rs:613-647`. Queries every `PointLight` / `SpotLight` every tick.
- **`rebuild_collider_caches`** — `crates/arvx-engine/src/engine/gizmo_ops.rs:379-477`. Iterates all entities with `RigidBody`, runs `compute_tight_local_aabb` + `build_coarse_collider` per entity. Triggered by `geometry_dirty` (8 setter sites) → `collider_caches_dirty`.
- **`scene_dirty` → SceneObjectInfo rebuild** — `crates/arvx-engine/src/engine/state_update.rs:298`. Sorts + rebuilds the full UI scene list. 7 setter sites.

### Full upload sites (render-side)

- **Material palette** — `crates/arvx-render/src/arvx_renderer.rs:2455-2470`. Full upload every frame.
- **Lights** — `crates/arvx-render/src/arvx_renderer.rs:2434-2453`. Full upload every frame.
- **Shader params slots** — `crates/arvx-render/src/arvx_shade.rs:539-568`. Full upload every frame per viewport.
- **Bone matrix palette** — `crates/arvx-render/src/arvx_scene.rs:755-760`. `ensure_and_write` (full buffer rewrite).
- **`gpu_instance_overlays`** — `crates/arvx-render/src/arvx_scene.rs:712-724`. `ensure_and_write`.
- **`gpu_instance_sculpts`** — `crates/arvx-render/src/arvx_scene.rs:731-743`. `ensure_and_write`.
- **`gpu_assets` / `gpu_instances`** — `crates/arvx-render/src/arvx_scene.rs:748-749`. `ensure_and_write`.

### Coarse dirty flags

- **`gpu_objects_dirty: bool`** — `crates/arvx-engine/src/engine/state/mod.rs:658`. 13 setter sites across sculpt/paint/gameplay/gizmo/scene_tree/picking/procedural/scene_io/cmd_edit ops. Consumer: full `update_scene_gpu`. **(B1 plumbing resolved — replaced with `GpuObjectsDirty` carrying per-entity scope. Actual setter count was 31. Hot stamp paths now `mark_entity(e)`. Consumer still does full rebuild until C2.)**
- **`geometry_dirty: bool`** — `crates/arvx-engine/src/engine/state/mod.rs:654`. 8 setter sites. Triggers `collider_caches_dirty`. **(B2 plumbing resolved — replaced with `GeometryDirty` carrying per-entity scope. Actual setter count was 14.)**
- **`scene_dirty: bool`** — `crates/arvx-engine/src/engine/state/mod.rs:656`. 7 setter sites. Triggers full SceneObjectInfo rebuild.
- **`collider_caches_dirty: bool`** — `crates/arvx-engine/src/engine/state/mod.rs:594`. Derivative of `geometry_dirty`. **(B2+C3 resolved — same `GeometryDirty` type; per-entity rebuild via `rebuild_collider_cache_for(entity)`.)**
- **`faces_dirty: bool`** — `crates/arvx-render/src/arvx_scene_manager/manager.rs:93`. **5 setter sites, ZERO consumer sites** — orphaned flag. Either wire a consumer or remove.

### Heavy clone patterns

- **`walk_snapshot()`** — `crates/arvx-render/src/arvx_scene_manager/manager.rs:437-442`. ~345 MB memcpy (octree + brick pool + leaf_attr). **(Resolved in A2.)** Pools were refactored to hold their data behind `Arc<Vec<…>>`; walk_snapshot is now three constant-time `Arc::clone`s, with copy-on-write (`Arc::make_mut`) on the next mutation if a snapshot is still outstanding.
- **`gpu_*.clone()` in submit_render_frame** — `crates/arvx-engine/src/engine/lifecycle.rs:1197-1202`. ~230 KB/frame. **(Resolved in A3.)** Each of the six fields is now `Arc<Vec<…>>`; snapshot construction is `Arc::clone`. Mutations route through `Arc::make_mut`, paying a one-time copy-on-write only when render still holds last frame's snapshot.
- **`bone_matrix_lbs/dqs.to_vec()`** — `crates/arvx-engine/src/engine/lifecycle.rs:712-713`. ~58 MB/frame.
- **`SkinBatchScratch.clone()`** — `crates/arvx-engine/src/engine/lifecycle.rs:737`. ~36 KB/frame.
- **`user_shader_entries.to_vec()`** — `crates/arvx-engine/src/engine/lifecycle.rs:114`. ~50 KB/frame. **(Resolved in A3.)** UserShaderRegistry stores entries as `Arc<Vec<UserShaderEntry>>`; sim calls `entries_arc()` and ships the Arc::clone.

---

## Discipline

- Each commit is scoped to one phase step.
- No feature work merges until the migration is complete.
- Telemetry (`[sculpt-pipeline]`, `[sculpt-pipeline-sim]`, `[geo-epoch]`, `[delta upload]`) stays in tree throughout the migration; it is how we verify each phase delivered.
- Stopgaps shipped during the surgery (today's commits `1f8852b9` through `e8377f49`) stay until their replacement migration step lands; then we remove the stopgap in the same commit as the replacement.
