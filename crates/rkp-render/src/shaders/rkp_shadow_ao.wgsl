// RKIPatch shadow + AO compute shader.
//
// Half-resolution: each invocation processes one half-res pixel.
// Reads G-buffer position + normal, traces shadow rays and AO probes
// through the per-voxel octree, writes shadow+AO to a storage texture.
//
// No BVH, no coarse field, no SDF sphere tracing. Just octree traversal.

// --- Constants ---
const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const PI: f32 = 3.14159265;

// --- Structs ---

struct VoxelSample {
    word0: u32,
    word1: u32,
}

struct RkpGpuObject {
    world: mat4x4<f32>,
    aabb_min: vec3<f32>, octree_root: u32,
    aabb_max: vec3<f32>, octree_depth: u32,
    octree_extent_bits: u32, voxel_size: f32,
    material_id: u32, object_id: u32,
    geom_type: u32, is_skinned: u32,
    bone_count: u32, bone_buffer_offset: u32,
    rest_octree_root: u32, rest_octree_depth: u32,
    rest_octree_extent_bits: u32, deformed_pool_offset: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
    _pad4: u32, _pad5: u32, _pad6: u32, _pad7: u32,
    _pad8: u32, _pad9: u32, _pad10: u32, _pad11: u32,
    inverse_world: mat4x4<f32>,
}

struct CameraUniforms {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
    view_proj: mat4x4<f32>,
}

struct ShadowAoParams {
    light_dir: vec3<f32>,
    num_objects: u32,
    light_intensity: f32,
    ao_radius: f32,
    ao_steps: u32,
    shadow_steps: u32,
}

// --- Bindings ---

// Group 0: scene data (RkpScene layout)
@group(0) @binding(0) var<storage, read> voxel_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<RkpGpuObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool: array<u32>;

// Group 1: G-buffer (read)
@group(1) @binding(0) var gbuf_position: texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal: texture_2d<f32>;

// Group 2: output shadow+AO texture (write, half-res)
@group(2) @binding(0) var shadow_ao_out: texture_storage_2d<rgba8unorm, write>;

// Group 3: params
@group(3) @binding(0) var<uniform> params: ShadowAoParams;

// --- Helpers ---

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

/// Cofactor-based 4x4 matrix inverse. Used to compute world→local from
/// the forward world matrix. Only called at half-res so cost is acceptable.
fn mat4_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0]; let a01 = m[1][0]; let a02 = m[2][0]; let a03 = m[3][0];
    let a10 = m[0][1]; let a11 = m[1][1]; let a12 = m[2][1]; let a13 = m[3][1];
    let a20 = m[0][2]; let a21 = m[1][2]; let a22 = m[2][2]; let a23 = m[3][2];
    let a30 = m[0][3]; let a31 = m[1][3]; let a32 = m[2][3]; let a33 = m[3][3];

    let b00 = a00*a11 - a01*a10;  let b01 = a00*a12 - a02*a10;
    let b02 = a00*a13 - a03*a10;  let b03 = a01*a12 - a02*a11;
    let b04 = a01*a13 - a03*a11;  let b05 = a02*a13 - a03*a12;
    let b06 = a20*a31 - a21*a30;  let b07 = a20*a32 - a22*a30;
    let b08 = a20*a33 - a23*a30;  let b09 = a21*a32 - a22*a31;
    let b10 = a21*a33 - a23*a31;  let b11 = a22*a33 - a23*a32;

    let det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
    let inv_det = 1.0 / det;

    return mat4x4<f32>(
        vec4<f32>( (a11*b11 - a12*b10 + a13*b09) * inv_det, (-a10*b11 + a12*b08 - a13*b07) * inv_det, (a10*b10 - a11*b08 + a13*b06) * inv_det, (-a10*b09 + a11*b07 - a12*b06) * inv_det),
        vec4<f32>((-a01*b11 + a02*b10 - a03*b09) * inv_det, (a00*b11 - a02*b08 + a03*b07) * inv_det, (-a00*b10 + a01*b08 - a03*b06) * inv_det, (a00*b09 - a01*b07 + a02*b06) * inv_det),
        vec4<f32>( (a31*b05 - a32*b04 + a33*b03) * inv_det, (-a30*b05 + a32*b02 - a33*b01) * inv_det, (a30*b04 - a31*b02 + a33*b00) * inv_det, (-a30*b03 + a31*b01 - a32*b00) * inv_det),
        vec4<f32>((-a21*b05 + a22*b04 - a23*b03) * inv_det, (a20*b05 - a22*b02 + a23*b01) * inv_det, (-a20*b04 + a21*b02 - a23*b00) * inv_det, (a20*b03 - a21*b01 + a22*b00) * inv_det),
    );
}

// --- Per-voxel octree sampling ---

/// Sample opacity at a local-space position by traversing the per-voxel octree.
fn octree_sample_opacity(local_pos: vec3<f32>, root: u32, depth: u32, extent: f32) -> f32 {
    // Convert local to octree space [0, extent].
    let octree_pos = local_pos + vec3<f32>(extent * 0.5);
    if any(octree_pos < vec3<f32>(0.0)) || any(octree_pos >= vec3<f32>(extent)) {
        return 0.0;
    }

    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);

    for (var level = 0u; level < depth; level++) {
        let node = octree_nodes[offset];
        if node == OCTREE_EMPTY { return 0.0; }
        if node == OCTREE_INTERIOR { return 1.0; }
        if (node & OCTREE_LEAF_BIT) != 0u {
            let slot = node & ~OCTREE_LEAF_BIT;
            return extract_opacity(voxel_pool[slot].word0);
        }
        let gt = vec3<u32>(octree_pos >= center);
        let child = gt.x + gt.y * 2u + gt.z * 4u;
        offset = node + child;
        half *= 0.5;
        center += vec3<f32>(
            select(-half, half, octree_pos.x >= center.x),
            select(-half, half, octree_pos.y >= center.y),
            select(-half, half, octree_pos.z >= center.z),
        );
    }

    let node = octree_nodes[offset];
    if node == OCTREE_EMPTY { return 0.0; }
    if node == OCTREE_INTERIOR { return 1.0; }
    if (node & OCTREE_LEAF_BIT) != 0u {
        let slot = node & ~OCTREE_LEAF_BIT;
        return extract_opacity(voxel_pool[slot].word0);
    }
    return 0.0;
}

// --- Shadow ray ---

/// March a ray through the opacity field accumulating transmittance.
/// Returns transmittance (1.0 = fully lit, 0.0 = fully shadowed).
fn trace_shadow(
    world_origin: vec3<f32>,
    world_dir: vec3<f32>,
    inv_world: mat4x4<f32>,
    obj: RkpGpuObject,
) -> f32 {
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = obj.voxel_size;
    let step_size = vs * 2.0;
    let max_dist = extent * 1.5;
    let max_steps = params.shadow_steps;

    // Skip initial region near surface to avoid self-shadowing.
    let start_offset = vs * 4.0;

    var transmittance = 1.0;
    for (var i = 0u; i < max_steps; i++) {
        let t = start_offset + f32(i) * step_size;
        if t > max_dist { break; }

        let world_pos = world_origin + world_dir * t;
        let local_pos = (inv_world * vec4<f32>(world_pos, 1.0)).xyz;
        let opacity = octree_sample_opacity(local_pos, obj.octree_root, obj.octree_depth, extent);

        transmittance *= (1.0 - opacity);
        if transmittance < 0.01 {
            return 0.0;
        }
    }

    return transmittance;
}

// --- AO probes ---

/// Sample ambient occlusion via N opacity probes along the surface normal.
fn compute_ao(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    inv_world: mat4x4<f32>,
    obj: RkpGpuObject,
) -> f32 {
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = obj.voxel_size;
    let probe_step = vs * 3.0;
    let num_steps = params.ao_steps;

    var occlusion = 0.0;
    var weight_sum = 0.0;

    for (var i = 1u; i <= num_steps; i++) {
        let dist = f32(i) * probe_step;
        let probe_world = world_pos + world_normal * dist;
        let probe_local = (inv_world * vec4<f32>(probe_world, 1.0)).xyz;
        let opacity = octree_sample_opacity(probe_local, obj.octree_root, obj.octree_depth, extent);

        let w = 1.0 / f32(i); // closer probes matter more
        occlusion += opacity * w;
        weight_sum += w;
    }

    if weight_sum < 0.001 {
        return 1.0;
    }

    let ao = 1.0 - clamp(occlusion / weight_sum, 0.0, 1.0);
    return ao;
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(shadow_ao_out);
    if gid.x >= out_dims.x || gid.y >= out_dims.y {
        return;
    }

    // Half-res → full-res pixel (sample center of 2x2 block).
    let full_coord = vec2<i32>(gid.xy) * 2 + vec2<i32>(1, 1);
    let pos_data = textureLoad(gbuf_position, full_coord, 0);
    let world_pos = pos_data.xyz;
    let hit_t = pos_data.w;

    // No geometry hit → fully lit, no AO.
    if hit_t >= 9999.0 || hit_t <= 0.0 {
        textureStore(shadow_ao_out, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 0.0, 1.0));
        return;
    }

    let world_normal = normalize(textureLoad(gbuf_normal, full_coord, 0).xyz);
    let light_dir = normalize(params.light_dir);

    // Accumulate shadow + AO across all objects.
    var shadow = 1.0;
    var ao = 1.0;

    for (var oi = 0u; oi < params.num_objects; oi++) {
        let obj = objects[oi];
        if obj.geom_type == 0u { continue; }

        let inv_world = obj.inverse_world;

        // Shadow: trace toward light.
        let obj_shadow = trace_shadow(world_pos, light_dir, inv_world, obj);
        shadow = min(shadow, obj_shadow);

        // AO: probe along normal.
        let obj_ao = compute_ao(world_pos, world_normal, inv_world, obj);
        ao = min(ao, obj_ao);
    }

    textureStore(shadow_ao_out, vec2<i32>(gid.xy), vec4<f32>(shadow, ao, 0.0, 1.0));
}
