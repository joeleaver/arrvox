# Proxy Mesh as First-Class Geometry — Phase 1 Plan

## Goal

Procedural objects render directly as triangle meshes via the surface-nets-from-SDF path, with full SDF-derived per-vertex shading data. The synthesized per-proxy `LeafAttr` slot is dropped; `splat_resolve` is not involved for proxy-mesh pixels. Procedural editing in the scene gets the same shading fidelity as voxel-baked objects, without paying voxel-bake cost.

## Background and scope

Procedurals are currently rendered by:
1. Baking them to a voxel octree (`BakeMode::Voxels`), then
2. Drawing through the standard octree path (mesh raster or march), which reads `LeafAttr` per cell via `splat_resolve`'s octree descent.

The proxy-mesh path (commit `4613056` + ancestors) introduced GPU surface-nets-from-SDF as an alternative bake mode but routes the result through the asset-mesh pipeline, synthesizing one `LeafAttr` slot per proxy and patching every vertex to reference it. This gives flat shading with a single material — strictly worse than voxel bake.

After discussion, we're committing to: **the proxy-mesh path is the production rendering target for procedurals**. Voxel-bake demotes to an opt-in conversion ("Convert to Voxel Object") for users who want paint or other per-cell features. The march path is on a longer-term track for retirement.

**In scope for Phase 1:**
- Per-vertex normal from SDF gradient (already computed; route to FS).
- Per-vertex primary + secondary material + blend, from `TreeSample`.
- Per-vertex procedural color, from `TreeSample`.
- Shadows / GI / SSAO via standard G-buffer consumers (free).
- Own G-buffer write path — proxy raster writes `gbuf_normal/material/glass` directly. No `splat_resolve` participation.
- Drop synthesized `LeafAttr` slot. Drop `ProxyMeshData.leaf_attr_slot`.

**Out of scope (explicitly):**
- Paint on procedurals. Use convert-to-voxels.
- Skinning. Procedurals don't animate.
- Glass per surface region. Single material per proxy ⇒ single opacity; if procedural's primary material is glass, the proxy renders glass-uniform or opaque-uniform. Mixed glass/opaque procedurals require voxel-bake.
- Texture-space paint, atlas auto-unwrap, LOD beyond single cluster, animation.
- March-path retirement (separate effort).

## Architecture

### Pipeline ordering per frame

```
1. Octree-mesh raster   → vis buffer + depth                (existing)
2. Splat raster         → vis buffer + depth                (existing)
3. splat_resolve compute → gbuf_normal/material/glass        (existing — writes for octree+splat pixels)
4. Proxy-mesh raster    → gbuf_normal/material/glass +      (NEW)
                          gbuf_position/pick + depth
                          (depth-tests against shared depth attachment)
```

Step 4 reuses the existing G-buffer storage textures as color attachments. `gbuffer.rs:141` already declares `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`, so no usage-flag change is required. Depth-test against the shared depth attachment ensures proxy meshes composite correctly with voxel-baked geometry behind them.

### Vertex extraction (modified `proc_surface_nets.wesl::vertex_emit`)

Today the shader calls `eval_tree_distance` and emits a vertex with `normal_oct` set, `leaf_attr_id = 0` (patched on CPU side after readback).

Change:
- Call `eval_tree(world_pos, count)` instead — returns `TreeSample { distance, material_id, secondary_material_id, blend_weight, color, node_id }`.
- Pack into the vertex (see "Vertex layout" below):
  - `normal_oct` — SDF gradient as today.
  - `material_packed = primary | (secondary << 16) | (blend4 << 28)` (same layout as `LeafAttr.material_packed`).
  - `color_packed = RGBA8` from `TreeSample.color` (alpha = 0xFF; reserve high bits if useful later).

### Vertex layout

Two reasonable options. Both keep the on-GPU 32 B size of `MeshVertex`.

**Option A — keep unified `MeshVertex`, repurpose fields per render path.**

```
struct MeshVertex {              // 32 B (unchanged size)
    local_pos:    [f32; 3],      // 12 B
    normal_oct:   u32,           //  4 B
    a:            u32,           //  4 B  octree-mesh: leaf_attr_id    | proxy: material_packed
    b:            u32,           //  4 B  octree-mesh: bone_indices    | proxy: color_packed
    c:            u32,           //  4 B  octree-mesh: bone_weights    | proxy: reserved
    d:            u32,           //  4 B  octree-mesh: pad             | proxy: reserved
}
```

Pros: one vertex struct in `rkp_core`. Both pipelines bind the same vertex buffer layout.

Cons: same struct means different things to different shaders; field names lie. Could mitigate by using neutral names + doc.

**Option B — introduce `rkp_core::mesh_extract::ProxyVertex` (32 B), distinct from `MeshVertex`.**

```
struct ProxyVertex {             // 32 B
    local_pos:        [f32; 3],
    normal_oct:       u32,
    material_packed:  u32,
    color_packed:     u32,
    _reserved:        [u32; 2],
}
```

Pros: field names mean what they say. Octree-mesh path is unaffected.

Cons: another vertex type; surface-nets emits its own format, breaks the "same MeshVertex flows everywhere" property the spike was designed around.

**Recommendation:** Option B. The aspirational "MeshVertex flows everywhere" property was load-bearing while we hoped proxies could ride the asset-mesh pipeline; we're explicitly giving that up. Field-naming clarity wins.

### Shaders

#### `mesh_proxy.wesl` (new)

VS bindings: camera + per-instance world matrix. No bone matrices (proxy meshes don't skin).
- Inputs: `local_pos`, `normal_oct`, `material_packed`, `color_packed`.
- Decodes `normal_oct` → vec3 in object-local space.
- Rotates normal into world space using `world` upper-3x3.
- Outputs to FS:
  - `clip_pos` (builtin)
  - `world_pos: vec3<f32>`
  - `world_normal: vec3<f32>` (interpolated; renormalize in FS)
  - `material_packed: u32` (flat — material IDs don't interpolate)
  - `color: vec3<f32>` (interpolated)
  - `blend: f32` (interpolated — extracted from material_packed.bits[28..32])

FS — writes G-buffer attachments directly:
- Renormalize interpolated `world_normal`.
- Pack material the same way `splat_resolve.wesl` does (so downstream PBR shader is agnostic to source):
  - `packed_r = primary | (secondary << 16)`
  - `packed_g = blend8 | (paint_intensity_zero << 8) | (color_rgb565 << 16)`
- Write attachments: `position` (world.xyz + hit_distance), `pick` (instance.object_id), `normal` (world_normal, alpha=1), `material` (packed_r, packed_g), `glass` (zeros).

#### `proc_surface_nets.wesl` — `vertex_emit` modifications

- Switch `eval_tree_distance` → `eval_tree` for the surface point evaluation.
- Pack `TreeSample` into the new vertex fields.
- Distance-only eval is still used for gradient finite differences (`gradient_normal` reads `eval_tree_distance` 6 times) — keep that as-is for cost.

### Attachment count check

Proxy-mesh raster needs to write: position, pick, normal, material, glass = 5 color attachments + 1 depth. WebGPU minimum spec is 4 color attachments. Need to check rinch's `required_limits` for `max_color_attachments`.

If 5 is available: write all five attachments in one pass.
If only 4: skip glass (clear-to-zero already happens; proxy meshes don't write glass anyway). Or skip pick (route picking through a separate, simpler pass).

**Action:** verify `max_color_attachments` early — straightforward fix either way.

## Rust changes

| File | Change |
|------|--------|
| `crates/rkp-core/src/mesh_extract.rs` (or new file) | Add `ProxyVertex` struct (32 B, `Pod + Zeroable`). |
| `crates/rkp-render/src/proc_surface_nets.rs` | Change `SurfaceMesh.vertices` to `Vec<ProxyVertex>`. Update extract path to read back ProxyVertex. |
| `crates/rkp-render/src/shaders/proc_surface_nets.wesl` | `vertex_emit` uses `eval_tree`, packs full shading data. |
| `crates/rkp-render/src/shaders/mesh_proxy.wesl` (new) | Proxy-mesh raster VS+FS. |
| `crates/rkp-render/src/mesh_proxy_pass.rs` (new) | Raster pipeline, render-pass orchestration, per-instance uniform upload. Borrows shape from `mesh_pass.rs`. |
| `crates/rkp-render/src/rkp_renderer.rs` | Schedule proxy-mesh raster after `splat_resolve`. |
| `crates/rkp-engine/src/components.rs` | Drop `ProxyMeshData.leaf_attr_slot`. |
| `crates/rkp-engine/src/engine/procedural_ops.rs::apply_proxy_mesh_result` | Drop the `LeafAttrPool::allocate()` call, the LeafAttr fill, and the per-vertex `leaf_attr_id` patch loop. Just upload geometry + spatial stamp. |
| `crates/rkp-engine/src/engine/procedural_ops.rs::release_proxy_handle_if_any` | Drop the `deallocate_range` call (no slot to free). |
| `crates/rkp-engine/src/engine/scene_gpu.rs` | Proxy meshes go onto a new `proxy_draws` list, not `splat_draws`. The synthesized `GpuAsset` and the entry in `gpu_instances` go away (or are slimmed to what shadow + GI need; TBD). |
| `crates/rkp-render/src/render_frame.rs` (or wherever `RenderCommand::UploadProxyMesh` is handled) | No change in signature; the renderer's proxy bookkeeping just doesn't touch LeafAttr pool. |

## Validation

Order of validation, each a separate before/after check:

1. **Single primitive (sphere, box, capsule).** Smooth shading instead of flat. Normals visibly track curvature.
2. **Union of two primitives with different materials, `MaterialCombine::Winner`.** Hard material boundary at the surface intersection.
3. **Union with `MaterialCombine::Layered`.** Dual-material seam over the SDF blend region.
4. **NoiseDisplace effect.** Domain-warped surface, normals follow warped shape (gradient computed against warped SDF, as today).
5. **MaterialByHeight / ColorByHeight / MaterialByNoise / ColorByNoise.** Band/pattern coloring visible per-vertex.
6. **Tower preset.** Multi-primitive composition renders correctly.
7. **Shadow casting.** Sun shadow lands on proxy meshes; proxy meshes cast shadows onto voxel-baked geometry and onto themselves.
8. **GI / SSAO.** Bounce + occlusion read from proxy G-buffer just like voxel geometry.
9. **Re-bake on parameter change.** Drag a slider, mesh updates within a frame or two with no flicker / no stale shading.
10. **Generators emitting multiple proxy meshes (5-50 instances).** Each renders correctly; bake cost stays acceptable.

## Risks / open questions

- **Attachment count (5 vs 4).** Verify rinch's `max_color_attachments`. Workaround if 4: drop glass attachment in proxy pass (zero clear pre-handled), or split into two passes.
- **`GpuInstance` / `GpuAsset` for proxy meshes.** Shadow + GI passes read instances; we need to keep proxy meshes in the instance list with valid `object_id` for picking. But the synthesized `GpuAsset` only existed to satisfy mesh LOD-select, which proxy meshes will bypass. Some pruning is possible — TBD during implementation.
- **Picking via `gbuf_pick`.** Proxy raster writes `instance.object_id` into pick. ECS pickup-from-pick-buffer should keep working transparently. Confirm during validation #1.
- **Mesh-frame vs grid-frame.** Proxy meshes were uploaded with `grid_origin = (0,0,0)` (proxy writes world-space positions in `local_pos`). Need to confirm the new pipeline doesn't drag the grid-frame conjugation logic from `mesh.wesl` along by accident.
- **Cluster-LOD path.** Proxy meshes are single-cluster (`PARENT_GROUP_ERROR_ROOT`). The new proxy raster shouldn't go through `mesh_lod_select` at all — direct indexed draws per instance, simpler than the octree-mesh path. Worth measuring perf to confirm we haven't accidentally regressed against the current proxy path through the LOD pipeline.

## After Phase 1

If Phase 1 lands clean, follow-ups in priority order:
- **Make proxy mesh the default bake mode.** Editor's "bake" verb produces proxy mesh; `BakeMode::Voxels` becomes the explicit "Convert to Voxel Object" path.
- **Glass uniform per proxy.** If primary material is glass, route the whole proxy through `mesh_glass` instead of the opaque proxy pass. Mixed glass/opaque remains voxel-bake-only.
- **Bigger surface-nets grids on demand.** Per-procedural `N` override for high-fidelity assets.
- **March-path retirement plan** (separate doc).
