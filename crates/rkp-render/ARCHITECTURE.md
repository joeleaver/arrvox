# `rkp-render` architecture

A working map of the renderer's hot paths. Module-level only — the file-level rustdoc covers individual types.

> **Living doc.** Update each entry when its module changes. If a section is stale, fix it before reading the rest — wrong maps are worse than no maps.

---

## User-shader pipeline (paint-grass-on-host, etc.)

### Data flow

```
[CPU sim] ECS scan: for each painted host with a user-shader material →
   • build ShaderRegionRequest { aabb, shader_name, host_octree_*,
     painted_world_min/max, host_surface_y, is_band_region, ... }
   • RenderFrame.user_shader_regions: Vec<ShaderRegionRequest>

         │  RenderFrame
         ▼
[GPU per-frame] tick_instance_pipeline (render_worker.rs):
   • bake user-shader prototypes via instance_proto + user_shader_proto.wgsl
     → asset records → host scene main pool

         │  RenderFrame, prototypes ready
         ▼
[GPU per-frame] run_user_shader_geom (render_worker.rs → user_shader_pass):
   • UserShaderObjectCache (cache.rs) keys (host_id, material_id, tile)
     → BucketPoolAllocator hands out (octree, brick, leaf-attr, fill-task)
       extents in the global pools; topology_hash + fill_hash decide
       whether classify / fill can be skipped
   • build_region_uniform (region.rs) packs per-region inputs into
     RegionUniform (240 B std430)
   • UserShaderPass.dispatch_regions (dispatch.rs):
       1. seed active_queue[L=0] with one root cell per topology-dirty region
       2. classify_main per BFS level — atomicAdd into global pools,
          push child cells to L+1
       3. brick_fill_main — runs the user's `dispatch_user_generate`,
          OR (for `is_band_region == true`) writes one GpuBandCell per
          max-depth band cell tagged with OCTREE_LEAF_BIT | OCTREE_BAND_BIT
   • OverflowReadback (overflow.rs) async-reads the per-pool
     overflow counters, logs hits

         │  octree_nodes / brick_pool / leaf_attr_pool now contain
         │  per-region BFS output, addressable via the region's
         │  octree_root in cache entries
         ▼
[GPU per-pixel] octree_march.wgsl (host march):
   • for each tile-list object: descend the host octree
   • on band-cell hit (OCTREE_BAND_BIT) → read GpuBandCell → call
     dispatch_user_instance_descend → descend_proto_octree into the
     baked prototype → return hit, normal, material
   • write G-buffer

         │
         ▼
[GPU per-pixel] shadow_trace + shadow_scatter + shadow_map (Phase 7-8)
[GPU per-pixel] rkp_shade — PBR using the merged G-buffer
[GPU per-pixel] fog / GI / TAA / present
```

### Modules

| Module | Owns | Key types |
|---|---|---|
| `user_shader_pass::cache` | Persistent per-region cache + variable-size pool allocators + sim→render request type | `BucketPoolAllocator`, `ShaderRegionRequest`, `UserShaderObjectCache`, `CachedSlot`, `PoolEstimate`, `estimate_region_pool` |
| `user_shader_pass::region` | GPU-side per-region uniform + band-cell wire format | `RegionUniform` (240 B), `GpuBandCell` (16 B), `build_region_uniform` |
| `user_shader_pass::dispatch` | BFS pipelines, transient buffers, per-frame dispatch encoder | `UserShaderPass`, `LevelUniform`, `compose_geom_source`, `resolve_shader_id` |
| `user_shader_pass::overflow` | Async readback ring for GPU overflow counters | `OverflowReadback` (private — internal use only) |
| `user_shader_proto_pass` | Prototype bake (`@instance_proto` shaders → octree in host main pool) | `PrototypeUniform`, prototype cache |
| `instance_proto` | Authoring-side prototype representation (CPU) | `InstanceProto` |
| `octree_march.wgsl` | Per-pixel host march; on `OCTREE_BAND_BIT` hit, descends into prototypes | (WGSL) |
| `user_shader_geom.wgsl` | BFS classify + fill compute kernels | (WGSL) |
| `user_shader_proto.wgsl` | Prototype bake compute kernel | (WGSL) |
| `shader_composer::types` | Public data types for the registry | `ParamDef`, `ShaderMetadata`, `UserShaderEntry`, `UserShaderRegistry`, `UserShaderInfo`, `ShaderComposerError`, `ComposedChunks` |
| `shader_composer::parser` | WGSL source → registry: `scan_dir`, `parse_file`, header `@`-directives, low-level scanner | `scan_dir`, `parse_file` |
| `shader_composer::compose` | Registry → per-pipeline WGSL chunks; per-shader `instance_descend` body emission; template splice | `compose`, `splice_inst_chunks` |
| `shader_composer::hash` | Deterministic FNV-1a 64 of registry contents (cache-key stable across restarts) | `fnv1a_64` |
| `rkp_engine::render_worker::state` | RenderWorker handle, RenderInbox mailbox, internal RenderState | `RenderWorker`, `RenderInbox`, `RenderState` |
| `rkp_engine::render_worker::loop_thread` | Render-thread main loop + per-snapshot interpolation | `run_render_thread`, `interpolate_instances`, `lerp_world_matrix` |
| `rkp_engine::render_worker::frame` | Per-frame orchestration (`render_one_frame` ~800 lines) | `render_one_frame`, `RenderOutcome` |
| `rkp_engine::render_worker::frame_helpers` | Tile-list splice, AABB transforms, shadow-map setup | `splice_transient_into_tile_lists`, `merge_tile_lists`, `compute_tlas_scene_aabb`, `transform_aabb_world`, `prepare_shadow_maps` |
| `rkp_engine::render_worker::user_shader_tick` | Per-frame user-shader bake + region BFS dispatch + cache hashing | `tick_instance_pipeline`, `run_user_shader_geom`, `topology_hash_for`, `fill_hash_for` |
| `tlas_build_pass::types` | Wire-format types + uniform structs for the TLAS build chain | `TlasPrim`, `InstanceTileCullEntry`, `AssembleHost/Morton/Radix/Karras` uniforms, RADIX_* constants |
| `tlas_build_pass::pass` | `TlasBuildPass` GPU pipelines + buffers + per-frame dispatch chain | `TlasBuildPass`, `GpuTlasBuildInputs` |
| `tlas_build_pass::cpu_reference` | CPU oracle for every stage (used by integration tests) | `cpu_reference_assemble_host/_user_shader/_morton/_radix_sort/_full_tree/_karras_node`, `karras_delta`, `scene_aabb_from_prims` |
| `rkp_scene_manager::types` | Public data types + private AssetCache machinery + emit_faces helper | `FaceInstance`, `AssetHandle`, `AssetInfo`, `SkinBrick`, `SkinningAssetData`, `ReloadResult`, `VoxelizeResult` |
| `rkp_scene_manager::manager` | RkpSceneManager struct + core methods (construction, faces, geometry epoch, slices, deallocation) | `RkpSceneManager` |
| `rkp_scene_manager::asset_load` | impl block for `acquire_asset` / `reload_asset` / `release_asset` / `load_asset_from_disk` / `skinning_data` | (impl methods) |
| `rkp_scene_manager::paint` | impl block for paint_epoch + brush_overlay + apply_paint_sphere + slice accessors | (impl methods) |
| `rkp_scene_manager::voxelize` | impl block for voxelize_primitive / voxelize_sdf_fn / integrate_artifact / deallocate_geometry | (impl methods) |
| `paint::select` | Spatial selection: sphere brush, single-cell pick, geodesic flood. Pure octree + brick reads. | `leaves_in_sphere`, `leaf_at_local_pos`, `surface_flood_fill`, `PaintedLeaf`, `LeafHit`, `FloodedLeaf` |
| `paint::write` | Paint write ops + brush math + color packing | `PaintStamp`, `paint_leaf_material/color`, `erase_leaf_color`, `compute_painted_attr/color`, `compute_erased_color`, `brush_weight`, `pack_color`, `unpack_color` |
| `shadow_map_pass::types` | Constants + LightCameraUniform + SetupParams wire types | `LightCameraUniform`, `SetupParams`, `SHADOW_MAP_*`, `SCATTER_INSTANCE_STRIDE` |
| `shadow_map_pass::light_camera` | CPU-side light-camera derivation (scene fit + frustum fit) | `compute_light_camera`, `compute_light_camera_frustum_fit` |
| `shadow_map_pass::pass` | ShadowMapPass GPU runtime — 5 pipelines + buffers + per-frame dispatch chain | `ShadowMapPass` |
| `user_shader_proto_pass::types` | Pool sizing constants + PrototypeEntry + depth helpers + OCTREE_EMPTY/INTERNAL_ATTR_NONE sentinels | `PrototypeEntry`, `MAX_PROTO_MAX_DEPTH`, `PROTO_*_POOL_CAPACITY` |
| `user_shader_proto_pass::cache` | PrototypeCache + per-shader octree-extent allocator + build_internal_levels pre-builder | `PrototypeCache`, `build_internal_levels` |
| `user_shader_proto_pass::pass` | PrototypeBakePass GPU runtime + PrototypeUniform + compose_proto_source | `PrototypeBakePass`, `PrototypeUniform`, `compose_proto_source` |

---

## What's NOT here (don't be fooled by old memory files)

These names appear in the conversation memory snapshots from 2026-04-29 / 04-30 but **do not exist in the current code**. They were shipped, then reverted in the grass debug session.

| Name | Status | Where to look in git history |
|---|---|---|
| Option B per-pixel pipeline | DELETED | Phase 5.2-5.5 — `9c36590` |
| `instance_emit_pass.rs` / `user_shader_emit.wgsl` | DELETED | Phase 5.2-5.5 — `9c36590` |
| `instance_march_pass.rs`, `instance_composite_pass.rs` | DELETED | Phase 5.2-5.5 — `9c36590` |
| Phase 6 GPU tile cull (`user_shader_tile_count/cull/prefix/scatter`) | DELETED | Reverted in Phase 5.2-5.5 — `9c36590` |
| `tile_cull_scratch`, `us_tile_entries`, `dispatch_user_inst_aabb`, `PREFIX_MAX_TILES` | NEVER LIVE on master HEAD | Existed only at `d9ca54d`, deleted before any commit consumed them |
| `instance_pool_buffer`, `dispatch_user_inst_to_local/aabb` chain | DELETED | Phase 5.6 A — `13e542a` |
| `emit_text`, `is_instance_pipeline`, `user_grass_emit` | DELETED | Phase 5.6 B+C — `c1cd310` |

If a memory file references any of these as if they were live, treat the memory as stale and confirm against the code before acting on it.

---

## V1.1 / band-cell debug session (uncommitted as of 2026-05-02)

The current uncommitted work layers band-cell shadow + V1.1 anchor projection on top of the band-cell architecture. This is in flight, not yet shipped, and adds known-band-aid fields:

- `GpuMaterial.instance_shader_id` (separate from `shader_id`) — band-aid, see `project_grass_debug_session`
- `RegionUniform.host_surface_y`, `painted_world_min/max` — V1.1 anchor projection inputs
- `user_shader_geom.wgsl` BFS gates for x/z (currently not constraining blade placement — open bug)
- `rkp_shadow_trace.wgsl` band-cell shadow path disabled (would otherwise produce dense self-shadow → black grass)

When the debug session lands or gets reset, update this section.

---

## File-size budget

CLAUDE.md targets ~700 lines per file. As of the user-shader + shader_composer splits:

| File | Lines | Status |
|---|---|---|
| `user_shader_pass.rs` (mod root) | 89 | ✅ |
| `user_shader_pass/cache.rs` | 924 | ⚠️ slightly over; coherent (allocator + cache + estimator) — split `BucketPoolAllocator` out if it grows further |
| `user_shader_pass/dispatch.rs` | 707 | ✅ at budget |
| `user_shader_pass/region.rs` | 165 | ✅ |
| `user_shader_pass/overflow.rs` | 169 | ✅ |
| `shader_composer.rs` (mod root) | 71 | ✅ |
| `shader_composer/types.rs` | 325 | ✅ |
| `shader_composer/parser.rs` | 710 | ✅ at budget |
| `shader_composer/compose.rs` | 478 | ✅ |
| `shader_composer/hash.rs` | 88 | ✅ |
| `shader_composer/tests.rs` | 827 | (tests file, exempt from budget) |
| `render_worker.rs` (mod root) | 83 | ✅ |
| `render_worker/state.rs` | 484 | ✅ |
| `render_worker/loop_thread.rs` | 340 | ✅ |
| `render_worker/frame.rs` | 820 | ⚠️ over; structurally one big function (`render_one_frame`) — splitting it further is a refactor not a move |
| `render_worker/frame_helpers.rs` | 178 | ✅ |
| `render_worker/user_shader_tick.rs` | 578 | ✅ |
| `tlas_build_pass.rs` (mod root) | 57 | ✅ |
| `tlas_build_pass/types.rs` | 131 | ✅ |
| `tlas_build_pass/pass.rs` | 1052 | ⚠️ over; per-stage split (assemble / morton-radix / karras / propagate) is a refactor not a move — deferred |
| `tlas_build_pass/cpu_reference.rs` | 387 | ✅ |
| `tlas_build_pass/tests.rs` | 341 | (tests file, exempt) |
| `rkp_scene_manager.rs` (mod root) | 35 | ✅ |
| `rkp_scene_manager/types.rs` | 269 | ✅ |
| `rkp_scene_manager/manager.rs` | 272 | ✅ |
| `rkp_scene_manager/asset_load.rs` | 463 | ✅ |
| `rkp_scene_manager/paint.rs` | 346 | ✅ |
| `rkp_scene_manager/voxelize.rs` | 343 | ✅ |
| `paint.rs` (mod root) | 39 | ✅ |
| `paint/select.rs` | 453 | ✅ |
| `paint/write.rs` | 220 | ✅ |
| `paint/tests.rs` | 495 | (tests file, exempt) |
| `shadow_map_pass.rs` (mod root) | 67 | ✅ |
| `shadow_map_pass/types.rs` | 86 | ✅ |
| `shadow_map_pass/light_camera.rs` | 250 | ✅ |
| `shadow_map_pass/pass.rs` | 669 | ✅ at budget |
| `shadow_map_pass/tests.rs` | 67 | (tests file, exempt) |
| `user_shader_proto_pass.rs` (mod root) | 52 | ✅ |
| `user_shader_proto_pass/types.rs` | 159 | ✅ |
| `user_shader_proto_pass/cache.rs` | 282 | ✅ |
| `user_shader_proto_pass/pass.rs` | 273 | ✅ |
| `user_shader_proto_pass/tests.rs` | 259 | (tests file, exempt) |

Other crate files still over budget (in size order):

In rkp-render:
- `rkp_volumetric.rs` 859 (NOT V1.1-touched)
- `rkp_shade.rs` 866 (V1.1-modified — Material struct mirror)
- `tlas_build_pass/pass.rs` 1052 (already split-once; would need per-stage refactor not module move)
- `user_shader_pass/cache.rs` 924 (already split-once)
- `shader_composer/parser.rs` 710 (already split-once, slightly over)
- `user_shader_pass/dispatch.rs` 707 (already split-once, at budget)

In rkp-engine:
- `lifecycle.rs` 1579 (V1.1-modified — wait for in-flight work)
- `component_registry.rs` 896 (NOT V1.1)
- `engine/state.rs` 886 (NOT V1.1)
- `render_worker/frame.rs` 820 (already split-once)
- `play_mode.rs` 710 (NOT V1.1)
- `command.rs` 705 (NOT V1.1)
- `material_library.rs` 703 (V1.1-modified)

In other crates (NOT touched this session):
- `rkp-core/src/sparse_octree.rs` 1820
- `rkp-core/src/voxelize_octree.rs` 1179
- `rkp-runtime/src/input/system.rs` 1140
- `rkp-procedural/src/arena.rs` 1029
- `rkp-core/src/scene_node.rs` 892
- `rkp-physics/src/sdf_collision.rs` 879
- `rkp-editor/src/ui/panels/object_properties.rs` 858
- `rkp-core/src/asset_file.rs` 821
- `rkp-procedural/src/flatten.rs` 801
- `rkp-editor/src/ui/panels/asset_properties.rs` 790
- `rkp-editor/src/ui/panels/prop_controls.rs` 743
- `rkp-core/src/companion.rs` 737

WGSL files over budget (no hard 700-line rule but worth flagging):

- `octree_march.wgsl` 2391 — host march + band-cell descent inlined throughout
- `rkp_shadow_trace.wgsl` 1331 — shadow trace + disabled band-cell shadow branch
- `rkp_shade.wgsl` 1093
- `user_shader_geom.wgsl` 1080 — BFS classify + V13-inline duplicate + V1.1 gates
- `shadow_scatter.wgsl` 972
