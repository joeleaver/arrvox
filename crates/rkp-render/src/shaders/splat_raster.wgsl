// Splat rasterization — camera-facing billboards with gradient normals.
//
// Single pass: closest billboard wins (depth test). Gradient normal from
// trilinear opacity field gives smooth shading. Circular discard gives
// round silhouettes.

// --- Constants ---
const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;

// --- Structs ---

struct FaceInstance {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    voxel_size: f32,
    voxel_slot: u32,
    packed: u32,
}

struct VoxelSample {
    word0: u32,
    word1: u32,
}

struct RkpObject {
    world: mat4x4<f32>,
    aabb_min: vec3<f32>,
    octree_root: u32,
    aabb_max: vec3<f32>,
    octree_depth: u32,
    octree_extent_bits: u32,
    voxel_size: f32,
    material_id: u32,
    object_id: u32,
    geom_type: u32,
    is_skinned: u32,
    bone_count: u32,
    bone_buffer_offset: u32,
    rest_octree_root: u32,
    rest_octree_depth: u32,
    rest_octree_extent_bits: u32,
    deformed_pool_offset: u32,
    layer_mask: u32,
    _pre_grid0: u32, _pre_grid1: u32, _pre_grid2: u32,
    grid_origin: vec3<f32>,
    _post_grid: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
}

struct CameraUniforms {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    layer_mask: u32,
    focus_object_id: u32,
    _cam_pad0: u32,
    _cam_pad1: u32,
    prev_vp: mat4x4<f32>,
    view_proj: mat4x4<f32>,
}

// --- Bindings ---

@group(0) @binding(0) var<storage, read> voxel_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<RkpObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;

@group(1) @binding(0) var<storage, read> face_instances: array<FaceInstance>;

// --- Helpers ---

fn unpack_obj_idx(packed: u32) -> u32 { return (packed >> 3u) & 0xFFFFFu; }

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

fn face_tangents(face_id: u32) -> mat2x3<f32> {
    switch face_id {
        case 0u: { return mat2x3<f32>(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0)); }
        case 1u: { return mat2x3<f32>(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)); }
        case 2u: { return mat2x3<f32>(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0)); }
        case 3u: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)); }
        case 4u: { return mat2x3<f32>(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0)); }
        case 5u: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }
        default: { return mat2x3<f32>(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)); }
    }
}

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

fn extract_material_id(word1: u32) -> u32 { return word1 & 0xFFFFu; }
fn extract_secondary_material_id(word1: u32) -> u32 { return (word1 >> 16u) & 0xFFFFu; }
fn extract_blend_weight(word0: u32) -> u32 { return (word0 >> 16u) & 0xFFu; }

fn transform_local_to_world(local_pos: vec3<f32>, world: mat4x4<f32>) -> vec3<f32> {
    return (world * vec4<f32>(local_pos, 1.0)).xyz;
}

fn transform_dir_to_world(local_dir: vec3<f32>, world: mat4x4<f32>) -> vec3<f32> {
    return (world * vec4<f32>(local_dir, 0.0)).xyz;
}

fn octree_sample_opacity(local_pos: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32) -> f32 {
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
        return extract_opacity(voxel_pool[slot].word0);
    }
    return 0.0;
}

fn trilinear_grad_uvw(c: array<f32, 8>, u: f32, v: f32, w: f32) -> vec3<f32> {
    return vec3<f32>(
        c[1] + c[4]*v + c[5]*w + c[7]*v*w,
        c[2] + c[4]*u + c[6]*w + c[7]*u*w,
        c[3] + c[5]*u + c[6]*v + c[7]*u*v,
    );
}

fn compute_gradient_normal(
    p: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32, world: mat4x4<f32>, fallback: vec3<f32>
) -> vec3<f32> {
    let h = vs * 0.5;
    let s000 = octree_sample_opacity(p + vec3(-h, -h, -h), root, depth, extent, vs);
    let s100 = octree_sample_opacity(p + vec3( h, -h, -h), root, depth, extent, vs);
    let s010 = octree_sample_opacity(p + vec3(-h,  h, -h), root, depth, extent, vs);
    let s110 = octree_sample_opacity(p + vec3( h,  h, -h), root, depth, extent, vs);
    let s001 = octree_sample_opacity(p + vec3(-h, -h,  h), root, depth, extent, vs);
    let s101 = octree_sample_opacity(p + vec3( h, -h,  h), root, depth, extent, vs);
    let s011 = octree_sample_opacity(p + vec3(-h,  h,  h), root, depth, extent, vs);
    let s111 = octree_sample_opacity(p + vec3( h,  h,  h), root, depth, extent, vs);

    var c: array<f32, 8>;
    c[0] = s000;
    c[1] = s100 - s000;
    c[2] = s010 - s000;
    c[3] = s001 - s000;
    c[4] = s110 - s100 - s010 + s000;
    c[5] = s101 - s100 - s001 + s000;
    c[6] = s011 - s010 - s001 + s000;
    c[7] = s111 - s110 - s101 - s011 + s100 + s010 + s001 - s000;

    let uvw = fract(p / vs + 0.5);
    let grad_uvw = trilinear_grad_uvw(c, uvw.x, uvw.y, uvw.z);
    let local_grad = -grad_uvw;
    let grad_len = length(local_grad);
    if grad_len > 0.01 {
        let world_grad = transform_dir_to_world(local_grad, world);
        return normalize(world_grad);
    }
    return fallback;
}

// --- Vertex Shader ---

struct VsOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) octree_pos: vec3<f32>,
    @location(2) billboard_uv: vec2<f32>,
    @location(3) @interpolate(flat) voxel_slot: u32,
    @location(4) @interpolate(flat) packed: u32,
    @location(5) @interpolate(flat) voxel_size: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VsOutput {
    let splat = face_instances[instance_id];
    let obj_idx = unpack_obj_idx(splat.packed);
    let obj = objects[obj_idx];

    let octree_center = vec3<f32>(splat.pos_x, splat.pos_y, splat.pos_z);
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let grid_offset = vec3<f32>(-extent * 0.5);
    let vs = splat.voxel_size;
    let face_id = splat.packed & 0x7u;
    let half = vs * 0.5;

    // Face-oriented quad (fixed in world space — no camera rotation flicker).
    let quad_idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let corner = quad_idx[vertex_id];
    let cu = f32(corner & 1u) * 2.0 - 1.0;
    let cv = f32((corner >> 1u) & 1u) * 2.0 - 1.0;

    let fn_dir = face_normal(face_id);
    let tangents = face_tangents(face_id);

    let octree_pos = octree_center + fn_dir * half + tangents[0] * cu * half + tangents[1] * cv * half;
    let local_pos = octree_pos + grid_offset;
    let world_pos = transform_local_to_world(local_pos, obj.world);
    let clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    return VsOutput(
        clip_pos, world_pos, octree_center,
        vec2<f32>(cu, cv),
        splat.voxel_slot, splat.packed, vs,
    );
}

// --- Fragment Shader ---

struct GBufferOutput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<u32>,
}

@fragment
fn fs_main(in: VsOutput) -> GBufferOutput {
    let obj_idx = unpack_obj_idx(in.packed);
    let obj = objects[obj_idx];
    let vs = in.voxel_size;
    let root = obj.octree_root;
    let depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);

    // Gradient normal from trilinear opacity field.
    let face_id = in.packed & 0x7u;
    let flat_fn = face_normal(face_id);
    let world_face_normal = normalize(transform_dir_to_world(flat_fn, obj.world));
    let normal = compute_gradient_normal(
        in.octree_pos, root, depth, extent, vs, obj.world, world_face_normal,
    );

    // Material + color.
    let voxel = voxel_pool[in.voxel_slot];
    let mat_id = extract_material_id(voxel.word1);
    let sec_mat = extract_secondary_material_id(voxel.word1);
    let blend = extract_blend_weight(voxel.word0);

    let color_packed = color_pool_data[in.voxel_slot];
    var color_rgb565 = 0u;
    if color_packed != 0u {
        let cr = (color_packed & 0xFFu) >> 3u;
        let cg = ((color_packed >> 8u) & 0xFFu) >> 2u;
        let cb = ((color_packed >> 16u) & 0xFFu) >> 3u;
        color_rgb565 = cr | (cg << 5u) | (cb << 11u);
    }

    let packed_r = (mat_id & 0xFFFFu) | ((sec_mat & 0xFFFFu) << 16u);
    let packed_g = (blend & 0xFFu) | (((obj.object_id + 1u) & 0xFFu) << 8u) | (color_rgb565 << 16u);
    let hit_t = length(in.world_pos - camera.position.xyz);

    return GBufferOutput(
        vec4<f32>(in.world_pos, hit_t),
        vec4<f32>(normal, 0.0),
        vec4<u32>(packed_r, packed_g, 0u, 0u),
    );
}
