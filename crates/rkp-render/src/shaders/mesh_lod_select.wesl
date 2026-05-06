// Per-cluster LOD selection — Phase 6.2 of the splat-to-mesh pivot.
//
// One thread per (asset_instance, cluster). Reads the asset's
// `MeshletCluster` table; applies the Karis-Nanite admit rule
//
//     parent_group_error_proj >= pixel_threshold  AND
//     cluster_error_proj      <  pixel_threshold
//
// using the cluster's object-local error metrics projected to
// screen-pixels through the per-instance world matrix and the
// camera's perspective focal length. Each thread writes one entry
// into the per-draw `args` buffer; non-admitted clusters write a
// zeroed `DrawIndexedIndirectArgs` (index_count = 0) so the matching
// `multi_draw_indexed_indirect` no-ops them at draw time without a
// CPU-side count read.
//
// Bind-group layout:
//   @group(0) @binding(0)  CameraUniforms (read)            — per-VR
//   @group(0) @binding(1)  MeshLodSelectParams (read)       — per-VR
//   @group(1) @binding(0)  MeshInstance (read)              — per-draw
//   @group(2) @binding(0)  array<MeshletCluster> (read)     — per-asset
//   @group(2) @binding(1)  array<DrawIndexedIndirectArgs>   — per-draw, RW

import package::lib::types::CameraUniforms;

struct MeshLodSelectParams {
    /// Pixel threshold for the Karis admit rule. Primary path uses
    /// 1.0 (one-pixel error budget); shadow path doubles this so the
    /// chain effectively picks `lod + 1`.
    pixel_threshold: f32,
    /// Number of clusters in the asset's cluster table for this
    /// dispatch. Threads beyond this are no-ops.
    cluster_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct MeshInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshletCluster {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    index_offset: u32,
    index_count: u32,
    lod_level: u32,
    _pad2: u32,
    cluster_error: f32,
    parent_group_error: f32,
    _pad3: vec2<u32>,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: MeshLodSelectParams;
@group(1) @binding(0) var<uniform> instance: MeshInstance;
@group(2) @binding(0) var<storage, read> clusters: array<MeshletCluster>;
@group(2) @binding(1) var<storage, read_write> args: array<DrawIndexedIndirectArgs>;

/// Approximate the maximum axis-scale of an affine 4x4 transform.
/// Cluster errors are object-local lengths; we multiply by this to
/// get world-space lengths for the perspective projection.
fn world_scale_factor(m: mat4x4<f32>) -> f32 {
    // Length of each basis row of the upper 3x3 — the per-axis scale
    // for the world transform applied to (1,0,0) / (0,1,0) / (0,0,1).
    let sx = length(m[0].xyz);
    let sy = length(m[1].xyz);
    let sz = length(m[2].xyz);
    return max(max(sx, sy), sz);
}

@compute @workgroup_size(64, 1, 1)
fn lod_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= params.cluster_count) {
        return;
    }

    let c = clusters[cidx];

    // Default: zeroed args entry — `index_count = 0` makes
    // `multi_draw_indexed_indirect` a no-op for non-admitted slots.
    var out: DrawIndexedIndirectArgs;
    out.index_count = 0u;
    out.instance_count = 0u;
    out.first_index = 0u;
    out.base_vertex = 0;
    out.first_instance = 0u;

    // World-space center via the instance transform; per-axis scale
    // converts cluster_error / parent_group_error from object-local
    // to world-space length.
    let center_local = (c.aabb_min + c.aabb_max) * 0.5;
    let world_center = (instance.world * vec4<f32>(center_local, 1.0)).xyz;
    let scale = world_scale_factor(instance.world);

    let view_distance = max(length(world_center - camera.position.xyz), 0.001);
    // `view_proj[1][1]` ~= cot(fov_y/2) for a standard perspective
    // projection (glam `Mat4::perspective_*`). Multiplied by half
    // viewport height gives pixels per world-unit at unit distance;
    // dividing by view_distance gives projected pixels per world-unit
    // at this depth. Approximation is exact for axis-aligned camera
    // basis and within ~10% for typical orbit views — well within
    // LOD selection's tolerance.
    let focal_pixels = abs(camera.view_proj[1][1]) * 0.5 * camera.resolution.y;
    let world_to_pixels = focal_pixels / view_distance;

    let cluster_err_px = c.cluster_error * scale * world_to_pixels;
    let parent_err_px = c.parent_group_error * scale * world_to_pixels;

    // Karis '21 admit rule. `parent_group_error = +∞` for DAG
    // roots ensures `parent_err_px > threshold` is always true, so
    // those clusters are admitted whenever their own error fits.
    let admit_parent = parent_err_px >= params.pixel_threshold;
    let admit_self = cluster_err_px < params.pixel_threshold;

    if (admit_parent && admit_self) {
        out.index_count = c.index_count;
        out.instance_count = 1u;
        out.first_index = c.index_offset;
        out.base_vertex = 0;
        // `first_instance` is unused by the existing mesh.wesl
        // (instance is read from a uniform binding, not from
        // `@builtin(instance_index)`), but encoding `object_id`
        // here keeps the field meaningful for any downstream
        // shader that does want to read it via vertex stage.
        out.first_instance = instance.object_id;
    }

    args[cidx] = out;
}
