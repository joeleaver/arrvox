// Rasterization vertex + fragment shader for surface voxel face quads.
//
// The vertex shader expands face instances (from the emit pass) into world-space
// quads. The fragment shader does trilinear refinement + gradient normal +
// material reads and writes the final G-buffer via MRT.
//
// This replaces the compute ray march for primary visibility.

// --- Constants ---
const OPACITY_THRESHOLD: f32 = 0.5;
const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// --- Structs ---

struct FaceInstance {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    voxel_size: f32,
    brick_slot: u32,
    packed: u32,
}

struct VoxelSample {
    word0: u32,
    word1: u32,
}

struct GpuObject {
    inverse_world: mat4x4<f32>,     // offset 0
    aabb_min: vec4<f32>,            // offset 64
    aabb_max: vec4<f32>,            // offset 80
    octree_root: u32,               // offset 96
    octree_depth: u32,              // offset 100
    octree_extent_bits: u32,        // offset 104
    _reserved_dims_z: u32,          // offset 108
    voxel_size: f32,                // offset 112
    material_id: u32,               // offset 116
    geom_type: u32,                 // offset 120
    blend_mode: u32,                // offset 124
    blend_radius: f32,              // offset 128
    sdf_param_0: f32,               // offset 132
    sdf_param_1: f32,               // offset 136
    sdf_param_2: f32,               // offset 140
    sdf_param_3: f32,               // offset 144
    accumulated_scale_x: f32,       // offset 148
    accumulated_scale_y: f32,       // offset 152
    accumulated_scale_z: f32,       // offset 156
    lod_level: u32,                 // offset 160
    object_id: u32,                 // offset 164
    primitive_type: u32,            // offset 168
    geometry_aabb_min_x: f32,       // offset 172
    geometry_aabb_min_y: f32,       // offset 176
    geometry_aabb_min_z: f32,       // offset 180
    geometry_aabb_max_x: f32,       // offset 184
    geometry_aabb_max_y: f32,       // offset 188
    geometry_aabb_max_z: f32,       // offset 192
    is_skinned: u32,                // offset 196
    bone_count: u32,                // offset 200
    bone_buffer_offset: u32,        // offset 204
    rest_octree_root: u32,          // offset 208
    rest_octree_depth: u32,         // offset 212
    rest_octree_extent_bits: u32,   // offset 216
    _rest_reserved: u32,            // offset 220
    shell_height: f32,              // offset 224
    sdf_shader_id: u32,             // offset 228
    sdf_shader_material: u32,       // offset 232
    deformed_pool_offset: u32,      // offset 236
    _pad10: u32, _pad11: u32, _pad12: u32, _pad13: u32,  // → 256 bytes
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

// --- Bindings ---

// Group 0: scene data (same as existing — octree_nodes occupies brick_maps slot)
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
// @group(0) @binding(4..5): scene uniforms, bvh (not needed by raster)
@group(0) @binding(6)  var<storage, read> bone_matrices: array<mat4x4<f32>>;
// @group(0) @binding(7): bone_positions (not needed by raster)
@group(0) @binding(8)  var<storage, read> bone_weights: array<u32>;
@group(0) @binding(9)  var<storage, read> deformed_pool: array<VoxelSample>;
@group(0) @binding(10) var<storage, read> color_pool_data: array<u32>;
@group(0) @binding(11) var<storage, read> color_companion_map: array<u32>;

// Group 1: face instances (from emit pass)
@group(1) @binding(0) var<storage, read> face_instances: array<FaceInstance>;

// Group 2: surface shell occupancy
@group(2) @binding(0) var<storage, read> surface_shell: array<u32>;

// --- Helpers ---

fn unpack_voxel_index(packed: u32) -> u32 { return packed & 0x1FFu; }
fn unpack_face_id(packed: u32) -> u32 { return (packed >> 9u) & 0x7u; }
fn unpack_obj_idx(packed: u32) -> u32 { return (packed >> 12u) & 0xFFFFu; }

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

fn extract_material_id(word1: u32) -> u32 { return word1 & 0xFFFFu; }
fn extract_secondary_material_id(word1: u32) -> u32 { return (word1 >> 16u) & 0xFFFFu; }
fn extract_blend_weight(word0: u32) -> u32 { return (word0 >> 16u) & 0xFFu; }

// Face normal directions: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
fn face_normal(face_id: u32) -> vec3<f32> {
    switch face_id {
        case 0u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 1u: { return vec3<f32>( 1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, -1.0, 0.0); }
        case 3u: { return vec3<f32>(0.0,  1.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 5u: { return vec3<f32>(0.0, 0.0,  1.0); }
        default: { return vec3<f32>(0.0, 1.0, 0.0); }
    }
}

// Get two tangent axes for a face (perpendicular to the face normal).
fn face_tangents(face_id: u32) -> mat2x3<f32> {
    switch face_id {
        // -X/+X: tangents are Y and Z
        case 0u, 1u: { return mat2x3<f32>(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)); }
        // -Y/+Y: tangents are X and Z
        case 2u, 3u: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)); }
        // -Z/+Z: tangents are X and Y
        case 4u, 5u: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }
        default: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }
    }
}

// Compute inverse of mat4 (needed for local→world transform).
// For rigid body transforms, inverse = transpose of rotation * negate translation.
// But GpuObject stores inverse_world directly, so we need its inverse.
// We'll use the object's AABB to approximate, or compute it analytically.
// For now, use the classic cofactor method for correctness.
fn transform_local_to_world(local_pos: vec3<f32>, inv_world: mat4x4<f32>) -> vec3<f32> {
    // For a rigid transform, inv_world = R^T * T(-pos).
    // The world transform is T(pos) * R.
    // world_pos = R * local_pos + pos
    // R = transpose(upper3x3(inv_world)), pos comes from -R * col3
    let r0 = vec3<f32>(inv_world[0].x, inv_world[1].x, inv_world[2].x);
    let r1 = vec3<f32>(inv_world[0].y, inv_world[1].y, inv_world[2].y);
    let r2 = vec3<f32>(inv_world[0].z, inv_world[1].z, inv_world[2].z);

    let rotated = vec3<f32>(
        dot(r0, local_pos),
        dot(r1, local_pos),
        dot(r2, local_pos),
    );

    // Translation: -R^T * t where t = inv_world column 3 xyz
    let t = vec3<f32>(inv_world[3].x, inv_world[3].y, inv_world[3].z);
    let world_origin = -vec3<f32>(dot(r0, t), dot(r1, t), dot(r2, t));

    return rotated + world_origin;
}

fn transform_dir_to_world(local_dir: vec3<f32>, inv_world: mat4x4<f32>) -> vec3<f32> {
    let r0 = vec3<f32>(inv_world[0].x, inv_world[1].x, inv_world[2].x);
    let r1 = vec3<f32>(inv_world[0].y, inv_world[1].y, inv_world[2].y);
    let r2 = vec3<f32>(inv_world[0].z, inv_world[1].z, inv_world[2].z);
    return vec3<f32>(dot(r0, local_dir), dot(r1, local_dir), dot(r2, local_dir));
}

// --- Skinning helpers ---

/// Read bone data from the deformed pool at a voxel coordinate.
fn raster_read_bone_field(vc: vec3<i32>, dims: vec3<u32>, total_voxels: vec3<i32>, pool_offset: u32) -> vec2<u32> {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c) / vec3<u32>(8u);
    let local = vec3<u32>(c) % vec3<u32>(8u);
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let vi = local.x + local.y * 8u + local.z * 64u;
    let idx = pool_offset + flat_brick * 512u + vi;
    let s = deformed_pool[idx];
    return vec2<u32>(s.word0, s.word1);
}

/// Inverse-skin a deformed position to rest-pose.
fn raster_inverse_skin_pos(pos: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        let inv_mat = bone_matrices[obj.bone_buffer_offset + obj.bone_count + bone_idx];
        let rp = (inv_mat * vec4<f32>(pos, 1.0)).xyz;
        result += rp * w;
        total_w += w;
    }
    if total_w > 0.0 { return result / total_w; }
    return pos;
}

/// Forward-skin a position from rest-pose to posed space.
fn raster_forward_skin_pos(pos: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        let fwd_mat = bone_matrices[obj.bone_buffer_offset + bone_idx];
        let fp = (fwd_mat * vec4<f32>(pos, 1.0)).xyz;
        result += fp * w;
        total_w += w;
    }
    if total_w > 0.0 { return result / total_w; }
    return pos;
}

/// Forward-skin a direction vector from rest-pose to posed space.
fn raster_forward_skin_dir(dir: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        let fwd_mat = bone_matrices[obj.bone_buffer_offset + bone_idx];
        let td = (fwd_mat * vec4<f32>(dir, 0.0)).xyz;
        result += td * w;
        total_w += w;
    }
    if total_w > 0.0 { return result / total_w; }
    return dir;
}

/// Read bone data at a local-space position, with 6-neighbor fallback.
fn raster_lookup_bone_data(local_pos: vec3<f32>, obj: GpuObject) -> vec2<u32> {
    // For skinned objects, the emit pass emits from the rest-pose octree.
    // The deformed pool contains bone weights in posed-space grid layout.
    // We use octree_depth/extent as the rest-pose dims (reinterpreted).
    let vs = obj.voxel_size;
    let depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let grid_size = extent;
    let grid_pos = local_pos; // local_pos is already in object's local space

    let dims_bricks = 1u << depth;
    let dims = vec3<u32>(dims_bricks);
    let total_v = vec3<i32>(dims) * 8;

    let voxel_coord = grid_pos / vs;
    let vc = vec3<i32>(floor(voxel_coord));

    var bone_data = raster_read_bone_field(vc, dims, total_v, obj.deformed_pool_offset);
    if bone_data.x == 0u && bone_data.y == 0u {
        let offsets = array<vec3<i32>, 6>(
            vec3<i32>(-1,0,0), vec3<i32>(1,0,0),
            vec3<i32>(0,-1,0), vec3<i32>(0,1,0),
            vec3<i32>(0,0,-1), vec3<i32>(0,0,1),
        );
        for (var ni = 0u; ni < 6u; ni++) {
            let nb = raster_read_bone_field(vc + offsets[ni], dims, total_v, obj.deformed_pool_offset);
            if nb.x != 0u || nb.y != 0u {
                bone_data = nb;
                break;
            }
        }
    }
    return bone_data;
}

/// Sample rest-pose opacity using the rest octree.
fn raster_sample_rest_opacity(rest_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let rest_root = obj.rest_octree_root;
    let rest_depth = obj.rest_octree_depth;
    let rest_extent = bitcast<f32>(obj.rest_octree_extent_bits);
    return octree_trilinear(rest_pos, rest_root, rest_depth, rest_extent, obj.voxel_size);
}

// --- Octree point query (inline, uses octree_nodes binding) ---

fn octree_sample_opacity(local_pos: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32) -> f32 {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);

    for (var level = 0u; level < depth; level++) {
        let node = octree_nodes[offset];
        if node == OCTREE_EMPTY { return 0.0; }
        if node == OCTREE_INTERIOR { return 1.0; }
        if (node & OCTREE_LEAF_BIT) != 0u {
            // Leaf at coarser level.
            let slot = node & ~OCTREE_LEAF_BIT;
            let dd = depth - level;
            let leaf_vs = vs * f32(1u << dd);
            let leaf_ext = leaf_vs * 8.0;
            let leaf_origin = floor(local_pos / leaf_ext) * leaf_ext;
            let in_brick = (local_pos - leaf_origin) / leaf_vs;
            let vx = clamp(u32(in_brick.x), 0u, 7u);
            let vy = clamp(u32(in_brick.y), 0u, 7u);
            let vz = clamp(u32(in_brick.z), 0u, 7u);
            return extract_opacity(brick_pool[slot * 512u + vx + vy * 8u + vz * 64u].word0);
        }
        let gt = vec3<u32>(local_pos >= center);
        let child = gt.x + gt.y * 2u + gt.z * 4u;
        offset = node + child;
        half *= 0.5;
        center += vec3<f32>(
            select(-half, half, local_pos.x >= center.x),
            select(-half, half, local_pos.y >= center.y),
            select(-half, half, local_pos.z >= center.z),
        );
    }

    let node = octree_nodes[offset];
    if node == OCTREE_EMPTY { return 0.0; }
    if node == OCTREE_INTERIOR { return 1.0; }
    if (node & OCTREE_LEAF_BIT) != 0u {
        let slot = node & ~OCTREE_LEAF_BIT;
        let leaf_ext = vs * 8.0;
        let leaf_origin = floor(local_pos / leaf_ext) * leaf_ext;
        let in_brick = (local_pos - leaf_origin) / vs;
        let vx = clamp(u32(in_brick.x), 0u, 7u);
        let vy = clamp(u32(in_brick.y), 0u, 7u);
        let vz = clamp(u32(in_brick.z), 0u, 7u);
        return extract_opacity(brick_pool[slot * 512u + vx + vy * 8u + vz * 64u].word0);
    }
    return 0.0;
}

fn octree_trilinear(local_pos: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32) -> f32 {
    let h = vs * 0.5;
    let s000 = octree_sample_opacity(local_pos + vec3(-h, -h, -h), root, depth, extent, vs);
    let s100 = octree_sample_opacity(local_pos + vec3( h, -h, -h), root, depth, extent, vs);
    let s010 = octree_sample_opacity(local_pos + vec3(-h,  h, -h), root, depth, extent, vs);
    let s110 = octree_sample_opacity(local_pos + vec3( h,  h, -h), root, depth, extent, vs);
    let s001 = octree_sample_opacity(local_pos + vec3(-h, -h,  h), root, depth, extent, vs);
    let s101 = octree_sample_opacity(local_pos + vec3( h, -h,  h), root, depth, extent, vs);
    let s011 = octree_sample_opacity(local_pos + vec3(-h,  h,  h), root, depth, extent, vs);
    let s111 = octree_sample_opacity(local_pos + vec3( h,  h,  h), root, depth, extent, vs);

    let f = fract(local_pos / vs + 0.5);
    let x0 = mix(s000, s100, f.x);
    let x1 = mix(s010, s110, f.x);
    let x2 = mix(s001, s101, f.x);
    let x3 = mix(s011, s111, f.x);
    let y0 = mix(x0, x1, f.y);
    let y1 = mix(x2, x3, f.y);
    return mix(y0, y1, f.z);
}

// --- Vertex Shader ---

struct VsOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) local_pos: vec3<f32>,
    @location(2) @interpolate(flat) brick_slot: u32,
    @location(3) @interpolate(flat) packed: u32,
    @location(4) @interpolate(flat) voxel_size: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VsOutput {
    let face = face_instances[instance_id];
    let face_id = unpack_face_id(face.packed);
    let obj_idx = unpack_obj_idx(face.packed);
    let obj = objects[obj_idx];

    let local_center = vec3<f32>(face.pos_x, face.pos_y, face.pos_z);
    let vs = face.voxel_size;
    let half = vs * 0.5;

    // Expand vertex_id (0-5) into quad corner.
    // Triangle list: 0,1,2, 2,1,3 → forms a quad.
    let quad_idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner = quad_idx[vertex_id];
    let cu = f32(corner & 1u) * 2.0 - 1.0; // -1 or +1
    let cv = f32((corner >> 1u) & 1u) * 2.0 - 1.0;

    let tangents = face_tangents(face_id);
    let fn_dir = face_normal(face_id);

    // Quad corners: offset from voxel center along face normal and tangents.
    let local_pos = local_center + fn_dir * half + tangents[0] * cu * half + tangents[1] * cv * half;

    // For skinned objects: forward-skin the local (rest-pose) position through
    // bone matrices to get posed-space position, then transform to world space.
    var world_pos: vec3<f32>;
    if obj.is_skinned != 0u && obj.bone_count > 0u {
        // Look up bone weights at the voxel center (not corner — bones are per-voxel).
        let bone_data = raster_lookup_bone_data(local_center, obj);
        if bone_data.x != 0u || bone_data.y != 0u {
            let posed_pos = raster_forward_skin_pos(local_pos, bone_data.x, bone_data.y, obj);
            world_pos = transform_local_to_world(posed_pos, obj.inverse_world);
        } else {
            world_pos = transform_local_to_world(local_pos, obj.inverse_world);
        }
    } else {
        world_pos = transform_local_to_world(local_pos, obj.inverse_world);
    }

    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    return VsOutput(
        clip_pos,
        world_pos,
        local_pos,
        face.brick_slot,
        face.packed,
        vs,
    );
}

// --- Fragment Shader ---

struct GBufferOutput {
    @location(0) position: vec4<f32>,   // Rgba32Float: xyz + hit_distance
    @location(1) normal: vec4<f32>,     // Rgba16Float: xyz + 0
    @location(2) material: vec4<u32>,   // Rg32Uint: packed_r, packed_g, 0, 0
    // Motion vectors omitted from MRT (32 byte/sample limit).
}

@fragment
fn fs_main(in: VsOutput) -> GBufferOutput {
    let obj_idx = unpack_obj_idx(in.packed);
    let obj = objects[obj_idx];
    let root = obj.octree_root;
    let depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = in.voxel_size;

    // Trilinear refinement: check if there's actually a surface at this position.
    // The rasterized face is a proxy — the true isosurface may be slightly offset.
    let local_pos = in.local_pos;
    let opacity = octree_trilinear(local_pos, root, depth, extent, vs);

    if opacity < OPACITY_THRESHOLD {
        discard;
    }

    // Gradient normal — different path for skinned vs static objects.
    var normal = vec3<f32>(0.0, 1.0, 0.0);

    if obj.is_skinned != 0u && obj.bone_count > 0u {
        // Skinned: inverse-skin to rest-pose, compute gradient there, forward-skin back.
        let bone_data = raster_lookup_bone_data(local_pos, obj);
        if bone_data.x != 0u || bone_data.y != 0u {
            let rest_pos = raster_inverse_skin_pos(local_pos, bone_data.x, bone_data.y, obj);
            let eps = vs * 2.0;
            let gx = raster_sample_rest_opacity(rest_pos + vec3(eps, 0.0, 0.0), obj)
                   - raster_sample_rest_opacity(rest_pos - vec3(eps, 0.0, 0.0), obj);
            let gy = raster_sample_rest_opacity(rest_pos + vec3(0.0, eps, 0.0), obj)
                   - raster_sample_rest_opacity(rest_pos - vec3(0.0, eps, 0.0), obj);
            let gz = raster_sample_rest_opacity(rest_pos + vec3(0.0, 0.0, eps), obj)
                   - raster_sample_rest_opacity(rest_pos - vec3(0.0, 0.0, eps), obj);
            let rest_grad = -vec3<f32>(gx, gy, gz);
            let posed_grad = raster_forward_skin_dir(rest_grad, bone_data.x, bone_data.y, obj);
            let world_grad = transform_dir_to_world(posed_grad, obj.inverse_world);
            let grad_len = length(world_grad);
            if grad_len > 1e-8 {
                normal = world_grad / grad_len;
            }
        }
    } else {
        // Static: gradient in local space, transform to world.
        let eps = vs * 2.0;
        let gx = octree_trilinear(local_pos + vec3(eps, 0.0, 0.0), root, depth, extent, vs)
               - octree_trilinear(local_pos - vec3(eps, 0.0, 0.0), root, depth, extent, vs);
        let gy = octree_trilinear(local_pos + vec3(0.0, eps, 0.0), root, depth, extent, vs)
               - octree_trilinear(local_pos - vec3(0.0, eps, 0.0), root, depth, extent, vs);
        let gz = octree_trilinear(local_pos + vec3(0.0, 0.0, eps), root, depth, extent, vs)
               - octree_trilinear(local_pos - vec3(0.0, 0.0, eps), root, depth, extent, vs);
        let local_grad = -vec3<f32>(gx, gy, gz);
        let world_grad = transform_dir_to_world(local_grad, obj.inverse_world);
        let grad_len = length(world_grad);
        if grad_len > 1e-8 {
            normal = world_grad / grad_len;
        }
    }

    // Material read from the brick pool.
    let slot = in.brick_slot;
    let voxel_idx = unpack_voxel_index(in.packed);
    let voxel = brick_pool[slot * 512u + voxel_idx];
    let mat_id = extract_material_id(voxel.word1);
    let sec_mat = extract_secondary_material_id(voxel.word1);
    let blend = extract_blend_weight(voxel.word0);

    let packed_r = (mat_id & 0xFFFFu) | ((sec_mat & 0xFFFFu) << 16u);
    let packed_g = (blend & 0xFFu) | ((obj.object_id & 0xFFu) << 8u);

    // Hit distance from camera.
    let hit_t = length(in.world_pos - camera.position.xyz);

    return GBufferOutput(
        vec4<f32>(in.world_pos, hit_t),
        vec4<f32>(normal, 0.0),
        vec4<u32>(packed_r, packed_g, 0u, 0u),
    );
}
