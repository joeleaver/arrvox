// Phase 7c Session 1 — TLAS primitive assembly (host-instance path).
//
// Walks the per-frame `gpu_instances` array, skips entries whose
// asset is a user-shader prototype (those flow through the
// scratch path in `tlas_assemble_user_shader.wgsl`), transforms
// the asset's local-space AABB through the instance's world matrix
// (Arvo's algorithm — exact for any affine transform), and
// atomic-appends a `TlasPrim` carrying the host instance index in
// `instance_index`.
//
// Writes into the same `tlas_prims` + `tlas_prim_count` pair the
// user-shader pass uses, so the engine fires both dispatches
// before Session 2's Morton sort consumes the unified buffer.

import package::lib::types::TlasPrim;

// Mirrors `ArvxGpuInstance` (144 B). Only the fields this pass needs
// are referenced; the rest are present so the WGSL stride matches the
// Rust upload stride exactly (otherwise `arrayLength(&instances)`
// miscounts and the assemble loop reads garbage).
struct GpuInstance {
    world: mat4x4<f32>,
    asset_id: u32,
    material_id: u32,
    object_id: u32,
    layer_mask: u32,
    is_skinned: u32,
    bone_buffer_offset: u32,
    overlay_offset: u32,
    overlay_count: u32,
    sculpt_offset: u32,
    sculpt_count: u32,
    _pad0: u32,
    _pad1: u32,
}

// Mirrors `ArvxGpuAsset` (80 B).
struct GpuAsset {
    aabb_min: vec3<f32>,
    octree_root: u32,
    aabb_max: vec3<f32>,
    octree_depth: u32,
    octree_extent_bits: u32,
    voxel_size: f32,
    geom_type: u32,
    bone_count: u32,
    grid_origin: vec3<f32>,
    rest_octree_root: u32,
    rest_octree_depth: u32,
    rest_octree_extent_bits: u32,
    shader_id: u32,
    _pad: u32,
}

struct AssembleHostUniform {
    instance_count: u32,
    asset_count: u32,
    prims_capacity: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> host_instances: array<GpuInstance>;
@group(0) @binding(1) var<storage, read> host_assets: array<GpuAsset>;
@group(0) @binding(2) var<storage, read_write> tlas_prims: array<TlasPrim>;
@group(0) @binding(3) var<storage, read_write> tlas_prim_count: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> u: AssembleHostUniform;

// Arvo's transform-AABB. For each output axis i, sum over input
// axes j the min/max of `M[j][i] * local_extent[j]`. Exact for any
// affine transform, O(9) ops. Mirrors `tlas_pass.rs::transform_aabb`.
fn transform_aabb_world(local_min: vec3<f32>, local_max: vec3<f32>, world: mat4x4<f32>) -> array<vec3<f32>, 2> {
    let translation = world[3].xyz;
    var new_min = translation;
    var new_max = translation;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        for (var j: u32 = 0u; j < 3u; j = j + 1u) {
            let m_ji = world[j][i];
            let a = m_ji * local_min[j];
            let b = m_ji * local_max[j];
            new_min[i] = new_min[i] + min(a, b);
            new_max[i] = new_max[i] + max(a, b);
        }
    }
    return array<vec3<f32>, 2>(new_min, new_max);
}

@compute @workgroup_size(64, 1, 1)
fn assemble_host_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.instance_count) {
        return;
    }
    let inst = host_instances[i];
    if (inst.asset_id >= u.asset_count) {
        return;
    }
    let asset = host_assets[inst.asset_id];
    // User-shader prototypes flow through the scratch path; skip
    // here. Same gate as `tlas_pass.rs::build_tlas`.
    if (asset.shader_id != 0u) {
        return;
    }
    let world_aabb = transform_aabb_world(asset.aabb_min, asset.aabb_max, inst.world);
    let slot = atomicAdd(&tlas_prim_count[0], 1u);
    if (slot >= u.prims_capacity) {
        return;
    }
    var p: TlasPrim;
    p.aabb_min = world_aabb[0];
    p.asset_id = inst.asset_id;
    p.aabb_max = world_aabb[1];
    // For host primitives, `instance_state_offset` is unused at
    // shadow-trace time (the affine `inv_world` path runs instead).
    // CPU `build_tlas` set it to 0 — keep that contract.
    p.instance_state_offset = 0u;
    p.material_id = inst.material_id;
    p.instance_index = i;
    p._pad0 = 0u;
    p._pad1 = 0u;
    tlas_prims[slot] = p;
}
