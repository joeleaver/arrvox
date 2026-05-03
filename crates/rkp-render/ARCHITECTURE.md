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

Other crate files still over budget (next cleanup targets, in size order):

- `tlas_build_pass.rs` 1899
- `rkp_scene_manager.rs` 1649
- `paint.rs` 1164
- `shadow_map_pass.rs` 1088
- `rkp_shade.rs` 866

(Engine-side: `render_worker.rs` 2461, `lifecycle.rs` 1586 are next-priority targets in their respective crate.)

WGSL files over budget (no hard 700-line rule but worth flagging):

- `octree_march.wgsl` 2391 — host march + band-cell descent inlined throughout
- `rkp_shadow_trace.wgsl` 1331 — shadow trace + disabled band-cell shadow branch
- `rkp_shade.wgsl` 1093
- `user_shader_geom.wgsl` 1080 — BFS classify + V13-inline duplicate + V1.1 gates
- `shadow_scatter.wgsl` 972
