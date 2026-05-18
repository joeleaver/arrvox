# `arvx-render` architecture

A working map of the renderer's hot paths. Module-level only — the file-level rustdoc covers individual types.

> **Living doc.** Update each entry when its module changes. If a section is stale, fix it before reading the rest — wrong maps are worse than no maps. Last refreshed 2026-05-18b (mesh-only cleanup session).

---

## Pipeline overview

Every viewport runs the same forward-raster + deferred-shade chain. Per-frame in dispatch order:

```
mesh_lod_select       compute   Karis-Nanite per-cluster admit → indirect-args
mesh raster           render    visibility-buffer (position, pick, leaf_slot, rest_pos)
mesh_resolve          compute   per-pixel octree descent → fills normal/material/glass
mesh_proxy raster     render    procedural triangle meshes composite (depth-test load)
user_shader_mesh      compute   spawn_count → prefix_sum → fill (per painted material)
user_shader_mesh      render    indirect-draw per material — composites onto G-buffer
mesh_glass front+back render    two depth/colour captures of glass material instances
mesh_glass_combine    compute   packs into gbuf_glass for the arvx_glass post
mesh_shadow render    render    per-cascade depth-only raster (CSM, 4 cascades)
mesh_shadow_blit      compute   bitcast depth into shadow_buffer the shade pass samples
mesh_glass_shadow     render    per-cascade glass entry/exit depth for Beer attenuation
ssao                  compute   half-res AO
brush_state           compute   single-thread cursor-pixel probe → screen-space paint cursor
arvx_shade             compute   deferred PBR; reads G-buffer + shadow + SSAO + atmosphere
arvx_volumetric        compute   fog / dust / clouds (cloud march + history blend)
arvx_glass             compute   Fresnel + Beer + screen-space refraction composite
arvx_god_rays          compute   radial blur from sun
bloom + composite     compute   threshold → downsample → upsample → composite
tone_map              compute   HDR → LDR
wireframe / grid      render    overlays (gizmo, isolation grid)
```

Build viewport in `Raymarch` preview mode replaces the mesh raster with `proc_raymarch` (compute, one thread/pixel, sphere-traces the flattened RPN tree). Everything downstream is unchanged.

---

## Modules

Grouped by role. File names map directly to `src/` paths unless noted.

### Scene + GPU buffers

| Module | Owns | Key types |
|---|---|---|
| `arvx_scene` | Single scene bind group (13 bindings: pools + camera + bone palettes + overlays) + per-frame upload | `ArvxScene`, `GeometryUpload`, `FrameUpload` |
| `arvx_scene_manager` | Voxel pools + octree + asset cache + paint + sculpt + voxelization | `ArvxSceneManager`, `AssetHandle`, `AssetInfo` |
| `arvx_scene_manager::types` | Wire-format types + private AssetCache machinery | `AssetEntry`, `AssetInfo`, `SkinningAssetData`, `FaceInstance` (legacy), `VoxelizeResult` |
| `arvx_scene_manager::manager` | Construction, geometry epoch, slice accessors, deallocation | `ArvxSceneManager` |
| `arvx_scene_manager::asset_load` | `.arvx` load/reload/release + `skinning_data()` | (impl methods) |
| `arvx_scene_manager::paint` | Brush stamps + paint_epoch + per-instance overlay | (impl methods) |
| `arvx_scene_manager::voxelize` | Primitive + SDF voxelization + integrate_artifact | (impl methods) |
| `arvx_scene_manager::sculpt` | Per-stamp filter+patch mesh re-extract + slab IBO allocator + cluster-table compaction | (impl methods) |
| `arvx_scene_manager::cluster_spatial_index` | Per-asset spatial index over LOD-0 clusters for brush AABB queries | (private) |
| `arvx_gpu_object` | Per-instance + per-asset GPU records (112 + 80 B) | `ArvxGpuInstance`, `ArvxGpuAsset` |
| `octree_gpu` | GPU octree buffer management | `OctreeGpu` |
| `sentinels` | Shared `0xFFFFFFFFu` sentinel constants | `OCTREE_EMPTY`, etc. |

### Mesh raster path (the only primary visibility path)

| Module | Owns | Key types |
|---|---|---|
| `mesh_instance` | Shared g0/g1 bind-group layouts + per-instance uniform (`world`, `object_id`, `grid_origin`, bone offsets, `skinning_mode`). Used by every raster path. | `MeshInstanceLayouts`, `MeshInstanceUniform`, `MeshDraw`, `SKINNING_MODE_NONE` |
| `mesh_pass` | Forward triangle raster — writes visibility-buffer triplet + rest_pos. CCW cull, per-vertex LBS/DQS skinning in VS. | `MeshPass`, `MeshVertex` (re-export from arvx-core) |
| `mesh_resolve_pass` | Per-pixel resolve compute — reads (leaf_slot, pick, rest_pos), descends asset octree, fills normal/material/glass | `MeshResolvePass` |
| `mesh_lod_select_pass` | Karis-Nanite per-cluster admit compute → DrawIndexedIndirectArgs table + draw_count | `MeshLodSelectPass`, `MeshLodSelectParams`, `DrawIndexedIndirectArgs` |
| `mesh_shadow_map_pass` | Directional CSM render + blit (4 cascades, depth-only VS+FS, bitcast→shadow_buffer) | `MeshShadowMapPass`, `MeshShadowParams`, `MeshShadowBlitParams` |
| `mesh_glass_pass` | Front + back glass raster + combine compute → `gbuf_glass` (Rg32Uint) | `MeshGlassPass`, `GlassFsParams` |
| `mesh_glass_shadow_pass` | Per-cascade front + back glass depth captures for Beer-attenuated CSM | `MeshGlassShadowPass` |
| `mesh_proxy_pass` | Procedural proxy-mesh raster (writes full G-buffer directly, no resolve indirection) | `MeshProxyPass`, `ProxyDraw`, `ProxyVertex`, `ProxyInstance` |
| `shadow_map_pass` | CSM uniform + shared `shadow_buffer` (atomic-u32) | `ShadowMapPass`, `LightCameraCsm`, `LightCameraShadeCsm`, CsmInputs |

### Shade + post

| Module | Owns | Key types |
|---|---|---|
| `arvx_shade` | Deferred PBR compute — atmosphere, CSM sample, dual-material blend, paint cursor | `ArvxShadePass`, `ShadeParams`, `GpuLight`, `GpuMaterial` |
| `arvx_atmosphere` | Atmosphere LUTs (transmittance / multi-scatter / sky-view / aerial perspective) | `ArvxAtmospherePass` |
| `arvx_volumetric` | Fog march + cloud march + history blend + composite | `ArvxVolumetricPass` |
| `arvx_glass` | Fresnel + Beer + screen-space refraction composite | `ArvxGlassPass` |
| `arvx_god_rays` | Radial blur from sun | `ArvxGodRayPass` |
| `arvx_ssao` | Half-res ambient occlusion | `ArvxSsaoPass` |
| `bloom`, `bloom_composite` | Multi-mip threshold + downsample/upsample + final composite | `BloomPass`, `BloomCompositePass` |
| `tone_map` | HDR → LDR | `ToneMapPass` |
| `gbuffer` | G-buffer texture set + formats | `GBuffer`, `GBUFFER_*_FORMAT` |
| `brush_state_pass` | Single-thread cursor-pixel probe feeding screen-space paint cursor | `BrushStatePass` |

### User-shader mesh path (V1)

| Module | Owns | Key types |
|---|---|---|
| `user_shader_mesh_pass` | spawn_count → prefix_sum → fill compute + indirect raster + shadow raster | `UserShaderMeshPass`, `UserShaderMeshDraw`, `AnchorContext`, `InstanceRecord` |
| `shader_composer` | Public registry of user shaders + WGSL compose | `UserShaderRegistry`, `compose`, `splice_inst_chunks` |
| `shader_composer::types` | Public types | `ParamDef`, `ShaderMetadata`, `UserShaderEntry`, `ComposedChunks` |
| `shader_composer::parser` | Filesystem scan + `@`-directive parse | `scan_dir`, `parse_file` |
| `shader_composer::compose` | Per-pipeline WGSL chunk emission + template splice | `compose`, `splice_inst_chunks` |
| `shader_composer::hash` | FNV-1a 64 over registry contents | `fnv1a_64` |

### Procedural (build-viewport live preview + proxy bake)

| Module | Owns | Key types |
|---|---|---|
| `proc_raymarch` | Sphere-trace compute for build-viewport live preview | `ProcRaymarchPass`, `RaymarchParams` |
| `proc_sample` | "Sample N positions" GPU evaluator (shared with bake path) | `ProcSamplePass`, `GpuEvaluator` |
| `proc_surface_nets` | GPU surface-nets-from-SDF — produces a `ProxyMesh` for procedurals | `ProcSurfaceNetsPass` |
| `proc_outline` | Selected-primitive outline overlay | `ProcOutlinePass` |
| `proc_ghost` | Subtract/Intersect ghost-cutter overlay | `ProcGhostPass` |
| `arvx_grid` | Infinite world-space grid (isolation mode) | `ArvxGridPass` |
| `wireframe` | Gizmo / debug line raster | `WireframePass`, `LineVertex` |

### Paint (CPU-side)

| Module | Owns | Key types |
|---|---|---|
| `paint::select` | Sphere/single-cell/flood-fill leaf selection | `leaves_in_sphere`, `leaf_at_local_pos`, `surface_flood_fill` |
| `paint::write` | Material + colour writes against `LeafAttrPool` | `PaintStamp`, `paint_leaf_material/color`, `erase_leaf_color` |

### TLAS (currently dead — see "What's NOT here")

| Module | Owns | Key types |
|---|---|---|
| `tlas_pass` | Wire-format types + nodes/leaves storage | `TlasPass`, `TlasNode`, `TlasInstanceLeaf` |
| `tlas_build_pass` | GPU build chain (Morton → radix sort → Karras → AABB propagate) | `TlasBuildPass`, `GpuTlasBuildInputs` |
| `tlas_build_pass::cpu_reference` | CPU oracle for integration tests | (test helpers) |

### Orchestration

| Module | Owns | Key types |
|---|---|---|
| `arvx_renderer` | Shared GPU state + `render_to` orchestration | `ArvxRenderer` |
| `viewport_renderer` | Per-viewport render targets + per-VR bind groups | `ViewportRenderer` |
| `context` | Device/queue wrapper | `RenderContext` |

---

## What's NOT here

After the 2026-05-18 mesh-only collapse + cleanup, these have been retired:

| Retired | When | Notes |
|---|---|---|
| `octree_march.wgsl` + per-pixel ray-march primary path | `955d235e` | Mesh raster replaced it |
| Splat-rasterizer prototype + `splat_*` shaders | `0b244c78` | Was the brief Phase B-2 prototype |
| `splat_pass/` module (renamed to `mesh_instance/`) | `4ea7ddd6` | Names dated from the prototype |
| `splat_resolve_pass.rs` / `splat_resolve.wesl` | `4ea7ddd6` | Renamed to `mesh_resolve_pass.rs` / `mesh_resolve.wesl` |
| `arvx_shadow_trace.wgsl` ray-traced shadow source | with march | Replaced by `mesh_shadow_map_pass` CSM |
| `shadow_fallback_texture` 1×1 placeholder | `246c6598` | Bilateral upsample + spot/point shadow path → constant 1.0 (tracked: `project_spot_point_shadows_todo`) |
| `skin_deform.rs` + GPU scatter pass + `bone_field_buffer` | `7641a9c0` | Mesh VS skins per-vertex; scatter+field weren't read |
| `ArvxGpuInstance.bone_field_*` (8 fields, 32 B) | `7641a9c0` | Struct shrunk 144→112 B |
| `PrimaryMode` enum + `ARVX_PRIMARY` env-var | `955d235e` | Mesh is the only path |
| `MarchParams` WESL struct | `99d89c9d` | No shader imported it |
| `merge_tile_lists` + CPU tile-cull merge | `99d89c9d` | Fed args that were march-only |
| 14 `_*` args on `render_to` (`_object_count`, `_tile_*`, `_tlas_*`, `_scene_extent`, etc.) | `99d89c9d` | All march-only |
| `eprint_march_stats` + `eprint_primary_march` | `99d89c9d` | March-only stat formatters, zero callers |

If a memory file references any of these as if they were live, treat the memory as stale and confirm against the code before acting on it.

### Known live-but-unused

- **TLAS pipeline.** `tlas_pass` + `tlas_build_pass` modules + 5 WESL shaders (`tlas_morton`, `tlas_radix_sort`, `tlas_karras`, `tlas_assemble_host`, `tlas_compute_dispatch_args`) still build a BVH each frame in `pre.rs`, but **nothing reads the output** (`tlas_pass.nodes_buffer` / `leaves_buffer`) after the march retirement. The BVH was the march's shadow-ray accelerator. Captured for follow-up in `project_tlas_dead_code_todo`. CPU `MARCH_TILE_SIZE` + `screen_aabbs_to_tiles` in `arvx-engine::scene_sync` are likely in the same boat — sweep alongside.
- **Spot / point light shadows.** Currently constant 1.0. Captured in `project_spot_point_shadows_todo` — needs spot=single-perspective-map / point=cubemap renders to mirror the directional CSM design.

---

## File-size budget

CLAUDE.md targets ~700 lines per file. Current state, files at or over budget:

| File | Lines | Status |
|---|---|---|
| `arvx_renderer.rs` | 2311 | ⚠️ over; mostly per-pass orchestration. Could split per-stage (atmosphere / mesh / shade / post) if it keeps growing. |
| `viewport_renderer.rs` | 2243 | ⚠️ over; per-VR state owner. Bind-group rebuilds are the bulk; candidates for extraction once stable. |
| `arvx_scene_manager/sculpt.rs` | 1816 | ⚠️ over; sculpt mutation core (slab allocator + filter+patch). |
| `arvx_scene_manager/types.rs` | 1311 | ⚠️ over; private AssetCache machinery lives here. Candidate to extract. |
| `arvx_scene_manager/asset_load.rs` | 948 | ⚠️ over |
| `arvx_scene.rs` | 944 | ⚠️ over; scene-buffer upload contract. |
| `arvx_shade.rs` | 883 | ⚠️ over; PBR shade pass — bind groups + ShadeParams + setters. |
| `user_shader_mesh_pass.rs` | 868 | ⚠️ over |
| `shader_composer/tests.rs` | 865 | tests file — exempt |
| `shader_composer/parser.rs` | 819 | ⚠️ at-budget+ |
| `proc_surface_nets.rs` | 691 | ✅ at budget |
| `mesh_glass_pass.rs` | 527 | ✅ |
| `bloom.rs` | 511 | ✅ |
| `shadow_map_pass/light_camera.rs` | 501 | ✅ |
| `proc_sample.rs` | 495 | ✅ |
| `arvx_scene_manager/manager.rs` | 486 | ✅ |
| `tlas_build_pass/pass/build.rs` | 471 | ✅ |
| `gbuffer.rs` | 471 | ✅ |
| `mesh_shadow_map_pass.rs` | 463 | ✅ |
| `proc_raymarch.rs` | 443 | ✅ |
| `arvx_volumetric/mod.rs` | 440 | ✅ |
| `arvx_atmosphere.rs` | 436 | ✅ |

Every other module is comfortably under 400 lines. Files marked ⚠️ have a clear extraction strategy when they grow further; the at-2k-line `arvx_renderer.rs` / `viewport_renderer.rs` are structurally one orchestration unit each, so further splitting would be a refactor (not a move).

WESL files over budget (no hard rule, flagged):

- `arvx_shade.wesl` ~1050 — PBR + atmosphere lookups + paint cursor in one entry point
- `user_shader_mesh.wesl`, `user_shader_mesh_compute.wesl` — V1 mesh-path user-shader compute trio
- `proc_eval.wesl` — RPN interpreter (shared)
