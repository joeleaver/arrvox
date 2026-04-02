// Splat March — surface-finding through trilinear opacity field, G-buffer output.
//
// Replaces rkf-render's ray_march.wgsl. Same bind groups 0-2, same G-buffer
// output format. The difference: fixed-step march through opacity (not sphere
// tracing through SDF distance).

// ── Constants ──────────────────────────────────────────────────────────────

const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const OPACITY_THRESHOLD: f32 = 0.5; // Surface is at 50% opacity
const MAX_MARCH_STEPS: u32 = 512u;
const OBJECT_TILE_SIZE: u32 = 16u;
const TILE_MAX_OBJECTS: u32 = 32u;

// ── Structs ────────────────────────────────────────────────────────────────

struct VoxelSample {
    word0: u32,
    word1: u32,
}

// GpuObject: 256 bytes, must match Rust #[repr(C)] layout exactly.
// All multi-component fields use scalar u32/f32 to avoid WGSL vec3/vec4
// alignment (16-byte) vs Rust [f32; 3]/[u32; 3] alignment (4-byte) mismatches.
struct GpuObject {
    inverse_world: mat4x4<f32>,  // 64 bytes @ offset 0
    aabb_min: vec4<f32>,         // 16 bytes @ offset 64
    aabb_max: vec4<f32>,         // 16 bytes @ offset 80
    brick_map_offset: u32,       // @ 96
    brick_map_dims_x: u32,      // @ 100
    brick_map_dims_y: u32,      // @ 104
    brick_map_dims_z: u32,      // @ 108
    voxel_size: f32,             // @ 112
    material_id: u32,            // @ 116
    geom_type: u32,               // @ 120
    blend_mode: u32,             // @ 124
    blend_radius: f32,           // @ 128
    sdf_param_0: f32,            // @ 132
    sdf_param_1: f32,            // @ 136
    sdf_param_2: f32,            // @ 140
    sdf_param_3: f32,            // @ 144
    accumulated_scale_x: f32,    // @ 148
    accumulated_scale_y: f32,    // @ 152
    accumulated_scale_z: f32,    // @ 156
    lod_level: u32,              // @ 160
    object_id: u32,              // @ 164
    primitive_type: u32,         // @ 168
    geometry_aabb_min_x: f32,    // @ 172
    geometry_aabb_min_y: f32,    // @ 176
    geometry_aabb_min_z: f32,    // @ 180
    geometry_aabb_max_x: f32,    // @ 184
    geometry_aabb_max_y: f32,    // @ 188
    geometry_aabb_max_z: f32,    // @ 192
    is_skinned: u32,             // @ 196
    bone_count: u32,             // @ 200
    bone_buffer_offset: u32,     // @ 204
    rest_brick_map_offset: u32,  // @ 208
    rest_brick_map_dims_x: u32,  // @ 212
    rest_brick_map_dims_y: u32,  // @ 216
    rest_brick_map_dims_z: u32,  // @ 220
    shell_height: f32,           // @ 224
    sdf_shader_id: u32,          // @ 228
    sdf_shader_material: u32,    // @ 232
    deformed_pool_offset: u32,   // @ 236
    _pad10: u32, _pad11: u32, _pad12: u32,
    _pad13: u32,                 // → 256
}

struct CameraUniforms {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
}

struct SceneUniforms {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

struct BvhNode {
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_min_z: f32,
    left: u32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    aabb_max_z: f32,
    right_or_object: u32,
}

struct MarchResult {
    hit: bool,
    t: f32,
    material_id: u32,
    secondary_material_id: u32,
    blend_weight: u32,
    object_id: u32,
    obj_idx: u32,
}

// ── Bind Groups ────────────────────────────────────────────────────────────

// Group 0: scene (same as rkf-render ray_march)
@group(0) @binding(0)  var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1)  var<storage, read> brick_maps: array<u32>;
@group(0) @binding(2)  var<storage, read> objects: array<GpuObject>;
@group(0) @binding(3)  var<uniform>       camera: CameraUniforms;
@group(0) @binding(4)  var<uniform>       scene: SceneUniforms;
@group(0) @binding(5)  var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(6)  var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(7)  var<storage, read> bone_positions: array<vec4<f32>>;
@group(0) @binding(8)  var<storage, read> bone_weights: array<u32>;
@group(0) @binding(9)  var<storage, read> deformed_pool: array<VoxelSample>;
@group(0) @binding(10) var<storage, read> color_pool_data: array<u32>;
@group(0) @binding(11) var<storage, read> color_companion_map: array<u32>;

// Group 1: G-buffer outputs (same format as rkf-render)
@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg32uint, write>;
@group(1) @binding(3) var gbuf_motion:   texture_storage_2d<rgba32float, write>;

// Group 2: tile object culling results (must match TileObjectCullPass read_bind_group)
@group(2) @binding(0) var<storage, read> tile_object_indices: array<u32>;
@group(2) @binding(1) var<storage, read> tile_object_counts: array<u32>;

// Group 3: materials + shader params (for opacity shader evaluation)
@group(3) @binding(0) var<storage, read> materials: array<Material>;
@group(3) @binding(1) var<storage, read> shader_params: array<ShaderParams>;

struct Material {
    albedo_r: f32, albedo_g: f32, albedo_b: f32, roughness: f32,
    metallic: f32, emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32, subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32, ior: f32,
    noise_scale: f32, noise_strength: f32, noise_channels: u32,
    shader_id: u32, _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
}

struct ShaderParams {
    param0: f32, param1: f32, param2: f32, param3: f32,
    param4: f32, param5: f32, param6: f32, param7: f32,
}

// ── Voxel Extraction ───────────────────────────────────────────────────────

/// Extract f16 opacity from word0 bits 0–15, returned as f32 clamped to [0,1].
fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

/// Extract primary material ID from word1 bits 0–15.
fn extract_material_id(word1: u32) -> u32 {
    return word1 & 0xFFFFu;
}

/// Extract secondary material ID from word1 bits 16–31.
fn extract_secondary_material_id(word1: u32) -> u32 {
    return (word1 >> 16u) & 0xFFFFu;
}

/// Extract blend weight from word0 bits 16–23.
fn extract_blend_weight(word0: u32) -> u32 {
    return (word0 >> 16u) & 0xFFu;
}

// ── Brick Pool Sampling ────────────────────────────────────────────────────

/// Sample a single voxel's opacity from the brick pool.
/// Returns 0.0 for EMPTY_SLOT (exterior), 1.0 for INTERIOR_SLOT (deep inside).
fn sample_opacity_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                     total_voxels: vec3<i32>) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return 0.0;
    }
    if slot == INTERIOR_SLOT {
        return 1.0;
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_opacity(brick_pool[idx].word0);
}

/// Sample a single voxel's full data (opacity + material) from the brick pool.
fn sample_voxel_data_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                        total_voxels: vec3<i32>) -> VoxelSample {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return VoxelSample(0u, 0u);
    }
    if slot == INTERIOR_SLOT {
        // Fully opaque, default material
        return VoxelSample(0x3C00u, 0u); // f16(1.0) = 0x3C00
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return brick_pool[idx];
}

/// Trilinear interpolation of the opacity field at a local-space position.
fn sample_opacity_trilinear(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);

    // Outside the grid — empty
    if outside_dist > brick_extent {
        return 0.0;
    }

    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    let c000 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 0), dims, total_voxels);
    let c100 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 0), dims, total_voxels);
    let c010 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 0), dims, total_voxels);
    let c110 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 0), dims, total_voxels);
    let c001 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 1), dims, total_voxels);
    let c101 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 1), dims, total_voxels);
    let c011 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 1), dims, total_voxels);
    let c111 = sample_opacity_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 1), dims, total_voxels);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}

// ── Inverse Skinning ───────────────────────────────────────────────────────

/// Look up bone weights at a local-space position (nearest-neighbor).
/// Returns packed (indices: u32, weights: u32) from the BoneBrick companion data.
fn lookup_bone_data(local_pos: vec3<f32>, obj: GpuObject) -> vec2<u32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.rest_brick_map_dims_x, obj.rest_brick_map_dims_y, obj.rest_brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return vec2<u32>(0u, 0u);
    }

    let voxel_coord = grid_pos / vs;
    let vc = clamp(vec3<i32>(floor(voxel_coord)), vec3<i32>(0), vec3<i32>(dims) * 8 - vec3<i32>(1));
    let brick = vec3<u32>(vc / vec3<i32>(8));
    let local = vec3<u32>(vc % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj.rest_brick_map_offset + flat_brick];

    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return vec2<u32>(0u, 0u);
    }

    let vi = local.x + local.y * 8u + local.z * 64u;
    let bw_base = slot * 1024u + vi * 2u;
    return vec2<u32>(bone_weights[bw_base], bone_weights[bw_base + 1u]);
}

/// Inverse-skin a position from posed space to rest-pose space.
/// Uses the inverse bone matrices (stored after forward matrices in bone_matrices buffer).
fn inverse_skin_pos(pos: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let bone_idx = (packed_indices >> (i * 8u)) & 0xFFu;
        let w = f32((packed_weights >> (i * 8u)) & 0xFFu);
        if w < 1.0 { continue; }
        // Inverse matrices are stored after forward matrices: offset + bone_count + bone_idx
        let inv_mat = bone_matrices[obj.bone_buffer_offset + obj.bone_count + bone_idx];
        let tp = (inv_mat * vec4<f32>(pos, 1.0)).xyz;
        result += tp * w;
        total_w += w;
    }
    if total_w > 0.0 { return result / total_w; }
    return pos;
}

/// Forward-skin a direction vector from rest-pose to posed space.
/// Uses the forward bone matrices for normal transformation.
fn forward_skin_dir(dir: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
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

// ── AABB Ray Intersection ──────────────────────────────────────────────────

/// Returns (t_enter, t_exit) for ray-AABB intersection. t_enter > t_exit means miss.
fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>,
                  aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec2<f32> {
    let t0 = (aabb_min - origin) * inv_dir;
    let t1 = (aabb_max - origin) * inv_dir;
    let t_near = min(t0, t1);
    let t_far = max(t0, t1);
    let t_enter = max(max(t_near.x, t_near.y), t_near.z);
    let t_exit = min(min(t_far.x, t_far.y), t_far.z);
    return vec2<f32>(max(t_enter, 0.0), t_exit);
}

// ── Material Sampling ──────────────────────────────────────────────────────

/// Sample nearest-neighbor material data at the hit point.
/// Returns (material_id, secondary_material_id, blend_weight).
fn sample_material_at_hit(local_pos: vec3<f32>, obj: GpuObject) -> vec3<u32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let voxel_coord = grid_pos / vs;
    let vc = vec3<i32>(floor(voxel_coord));
    let total_voxels = vec3<i32>(dims) * 8;

    let voxel = sample_voxel_data_at(obj.brick_map_offset, vc, dims, total_voxels);
    let mat = extract_material_id(voxel.word1);
    let sec_mat = extract_secondary_material_id(voxel.word1);
    let blend = extract_blend_weight(voxel.word0);
    return vec3<u32>(mat, sec_mat, blend);
}

// ── Opacity Shader Functions ───────────────────────────────────────────────
//
// User-provided opacity shader functions are injected here by ShaderComposer.
// Each function has signature:
//   fn opacity_<name>(local_pos: vec3<f32>, h_above: f32, obj: GpuObject, mat_id: u32) -> f32
// Returns opacity: 0.0 = empty, 1.0 = solid.
// The dispatch_opacity_shader() switch is also generated here.
//
// OPACITY_SHADER_FUNCTIONS

// ── Shell March ───────────────────────────────────────────────────────────

const SHELL_MAX_STEPS: u32 = 64u;

/// Compute the local-space surface normal at a position from the opacity gradient.
fn compute_local_normal(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let eps = obj.voxel_size * 2.0;
    let gx = sample_opacity_trilinear(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_opacity_trilinear(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_opacity_trilinear(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    let g = -vec3<f32>(gx, gy, gz);
    let len = length(g);
    if len < 1e-10 { return vec3<f32>(0.0, 1.0, 0.0); }
    return g / len;
}

/// March through the shell above a surface to find procedural geometry.
///
/// Evaluates user opacity shaders in the shell region above the voxel surface.
/// Returns the local-space t of the closest shell hit, or -1.0 on miss.
/// Also returns the world-space normal and the opacity shader's material ID
/// via output parameters.
struct ShellHit {
    t: f32,
    local_hit: vec3<f32>,
    normal_world: vec3<f32>,
    mat_id: u32,
}

fn march_shell(
    local_origin: vec3<f32>,
    local_dir: vec3<f32>,
    surface_t: f32,
    surface_local: vec3<f32>,
    surface_normal_local: vec3<f32>,
    shader_id: u32,
    mat_id: u32,
    shell_height: f32,
    obj: GpuObject,
) -> ShellHit {
    var result: ShellHit;
    result.t = -1.0;

    // Compute how much ray-space distance corresponds to the shell height.
    // At oblique angles the ray travels further through the shell.
    let cos_angle = max(abs(dot(local_dir, surface_normal_local)), 0.05);
    let march_range = shell_height / cos_angle;
    let march_start = max(surface_t - march_range, 0.001);
    let march_end = surface_t;
    let step_size = march_range / f32(SHELL_MAX_STEPS);

    var prev_opacity = 0.0;
    var prev_t = march_start;
    var t = march_start;

    for (var step = 0u; step < SHELL_MAX_STEPS; step++) {
        if t >= march_end { break; }

        let local_pos = local_origin + local_dir * t;
        let h_above = dot(local_pos - surface_local, surface_normal_local);

        // Skip if below the surface or above the shell
        if h_above < 0.0 {
            prev_opacity = 0.0;
            prev_t = t;
            t += step_size;
            continue;
        }
        if h_above > shell_height {
            prev_opacity = 0.0;
            prev_t = t;
            t += step_size;
            continue;
        }

        let opacity = dispatch_opacity_shader(shader_id, local_pos, h_above, obj, mat_id);

        // Check for surface crossing (opacity rises above threshold)
        if opacity >= OPACITY_THRESHOLD && prev_opacity < OPACITY_THRESHOLD {
            let frac = (OPACITY_THRESHOLD - prev_opacity) / (opacity - prev_opacity + 1e-10);
            let hit_t = mix(prev_t, t, frac);
            let hit_pos = local_origin + local_dir * hit_t;
            let hit_h_above = dot(hit_pos - surface_local, surface_normal_local);

            // Compute normal from gradient of the procedural opacity function
            let e = obj.voxel_size * 0.5;
            let gx = dispatch_opacity_shader(shader_id, hit_pos + vec3<f32>(e, 0.0, 0.0), hit_h_above, obj, mat_id)
                   - dispatch_opacity_shader(shader_id, hit_pos - vec3<f32>(e, 0.0, 0.0), hit_h_above, obj, mat_id);
            let gy = dispatch_opacity_shader(shader_id, hit_pos + vec3<f32>(0.0, e, 0.0), hit_h_above + e, obj, mat_id)
                   - dispatch_opacity_shader(shader_id, hit_pos - vec3<f32>(0.0, e, 0.0), hit_h_above - e, obj, mat_id);
            let gz = dispatch_opacity_shader(shader_id, hit_pos + vec3<f32>(0.0, 0.0, e), hit_h_above, obj, mat_id)
                   - dispatch_opacity_shader(shader_id, hit_pos - vec3<f32>(0.0, 0.0, e), hit_h_above, obj, mat_id);
            let local_grad = -vec3<f32>(gx, gy, gz);
            let world_grad = (transpose(obj.inverse_world) * vec4<f32>(local_grad, 0.0)).xyz;
            let glen = length(world_grad);

            result.t = hit_t;
            result.local_hit = hit_pos;
            result.mat_id = mat_id;
            if glen > 1e-10 {
                result.normal_world = world_grad / glen;
            } else {
                // Fallback: use the surface normal
                let world_n = (transpose(obj.inverse_world) * vec4<f32>(surface_normal_local, 0.0)).xyz;
                result.normal_world = normalize(world_n);
            }
            return result;
        }

        prev_opacity = opacity;
        prev_t = t;
        t += step_size;
    }

    return result;
}

// ── Per-Object March ───────────────────────────────────────────────────────

/// March a ray through a single object's opacity field.
/// Returns the t value of the surface hit, or -1.0 on miss.
/// Dispatches to skinned variant for animated objects.
fn march_object(origin: vec3<f32>, dir: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];

    if obj.is_skinned != 0u && obj.bone_count > 0u {
        return march_object_skinned(origin, dir, obj_idx);
    }

    return march_object_static(origin, dir, obj_idx);
}

/// March through a static (non-skinned) object.
fn march_object_static(origin: vec3<f32>, dir: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    let inv_world = obj.inverse_world;

    let local_origin = (inv_world * vec4<f32>(origin, 1.0)).xyz;
    let local_dir = normalize((inv_world * vec4<f32>(dir, 0.0)).xyz);
    let safe_dir = select(local_dir, vec3<f32>(1e-10), abs(local_dir) < vec3<f32>(1e-10));
    let inv_local_dir = 1.0 / safe_dir;

    let brick_extent = obj.voxel_size * 8.0;
    let dims = vec3<f32>(
        f32(obj.brick_map_dims_x),
        f32(obj.brick_map_dims_y),
        f32(obj.brick_map_dims_z),
    );
    let half_grid = dims * brick_extent * 0.5;
    let t_range = intersect_aabb(local_origin, inv_local_dir, -half_grid, half_grid);
    if t_range.x > t_range.y {
        return -1.0;
    }

    let fine_step = obj.voxel_size * 0.5;
    let coarse_step = brick_extent; // skip entire brick when in empty space
    let udims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(udims) * brick_extent;

    var t = t_range.x;
    var prev_opacity = 0.0;
    var prev_t = t;

    for (var step = 0u; step < MAX_MARCH_STEPS; step++) {
        if t > t_range.y { break; }

        let local_pos = local_origin + safe_dir * t;

        // Check if we're in an empty brick — skip the whole brick if so
        let grid_pos = local_pos + grid_size * 0.5;
        let brick_coord = vec3<i32>(floor(grid_pos / brick_extent));

        var in_empty_brick = false;
        if all(brick_coord >= vec3<i32>(0)) && all(vec3<u32>(brick_coord) < udims) {
            let bc = vec3<u32>(brick_coord);
            let flat = bc.x + bc.y * udims.x + bc.z * udims.x * udims.y;
            let slot = brick_maps[obj.brick_map_offset + flat];
            if slot == EMPTY_SLOT {
                in_empty_brick = true;
            }
        }

        if in_empty_brick {
            // Jump to the exit of this brick
            let brick_min = vec3<f32>(brick_coord) * brick_extent - grid_size * 0.5;
            let brick_max = brick_min + vec3<f32>(brick_extent);
            let t_exit = intersect_aabb(local_origin, inv_local_dir, brick_min, brick_max);
            prev_opacity = 0.0;
            prev_t = t;
            t = t_exit.y + obj.voxel_size * 0.1; // step just past the brick boundary
            continue;
        }

        let opacity = sample_opacity_trilinear(local_pos, obj);

        if opacity >= OPACITY_THRESHOLD && prev_opacity < OPACITY_THRESHOLD {
            let frac = (OPACITY_THRESHOLD - prev_opacity) / (opacity - prev_opacity + 1e-10);
            return mix(prev_t, t, frac);
        }

        prev_opacity = opacity;
        prev_t = t;
        t += fine_step;
    }

    return -1.0;
}

/// Read bone data from the deformed pool at a posed-space voxel position.
/// Returns (packed_indices, packed_weights). Both 0 = no bone data.
fn read_bone_field(vc: vec3<i32>, dims: vec3<u32>, total_voxels: vec3<i32>, pool_offset: u32) -> vec2<u32> {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c) / vec3<u32>(8u);
    let local = vec3<u32>(c) % vec3<u32>(8u);
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let vi = local.x + local.y * 8u + local.z * 64u;
    let idx = pool_offset + flat_brick * 512u + vi;
    let s = deformed_pool[idx];
    return vec2<u32>(s.word0, s.word1);
}

/// Inverse-skin a deformed position to rest-pose using packed bone weights.
fn inverse_skin_position(pos: vec3<f32>, packed_indices: u32, packed_weights: u32, obj: GpuObject) -> vec3<f32> {
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

/// Sample rest-pose opacity at a continuous position via the rest brick map.
fn sample_rest_opacity(rest_pos: vec3<f32>, obj: GpuObject) -> f32 {
    var rest_obj = obj;
    rest_obj.brick_map_offset = obj.rest_brick_map_offset;
    rest_obj.brick_map_dims_x = obj.rest_brick_map_dims_x;
    rest_obj.brick_map_dims_y = obj.rest_brick_map_dims_y;
    rest_obj.brick_map_dims_z = obj.rest_brick_map_dims_z;
    return sample_opacity_trilinear(rest_pos, rest_obj);
}

/// March through a skinned object.
///
/// Reads bone weights from the deformed pool (scattered by SkinDeformPass),
/// inverse-skins to rest-pose, and samples opacity from the rest-pose brick pool.
fn march_object_skinned(origin: vec3<f32>, dir: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    let inv_world = obj.inverse_world;

    let local_origin = (inv_world * vec4<f32>(origin, 1.0)).xyz;
    let local_dir = normalize((inv_world * vec4<f32>(dir, 0.0)).xyz);
    let safe_dir = select(local_dir, vec3<f32>(1e-10), abs(local_dir) < vec3<f32>(1e-10));
    let inv_local_dir = 1.0 / safe_dir;

    // Use the world-space AABB (covers the deformed pose)
    let local_aabb_min = (inv_world * vec4<f32>(obj.aabb_min.xyz, 1.0)).xyz;
    let local_aabb_max = (inv_world * vec4<f32>(obj.aabb_max.xyz, 1.0)).xyz;
    let t_range = intersect_aabb(local_origin, inv_local_dir,
        min(local_aabb_min, local_aabb_max),
        max(local_aabb_min, local_aabb_max));
    if t_range.x > t_range.y {
        return -1.0;
    }

    // Deformed pool grid dimensions
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let grid_size = vec3<f32>(dims) * brick_extent;
    let total_v = vec3<i32>(dims) * 8;

    let fine_step = vs * 0.5;

    var t = t_range.x;
    var prev_opacity = 0.0;
    var prev_t = t;

    for (var step = 0u; step < MAX_MARCH_STEPS; step++) {
        if t > t_range.y { break; }

        let local_pos = local_origin + safe_dir * t;
        let grid_pos = local_pos + grid_size * 0.5;

        // Check if we're inside the deformed grid at all
        let brick_coord = vec3<i32>(floor(grid_pos / brick_extent));
        if any(brick_coord < vec3<i32>(0)) || any(vec3<u32>(brick_coord) >= dims) {
            // Outside deformed grid — skip ahead
            prev_opacity = 0.0;
            prev_t = t;
            t += brick_extent;
            continue;
        }

        // Read bone weights from deformed pool at this voxel
        let vc = vec3<i32>(floor(grid_pos / vs));
        let total_v = vec3<i32>(dims) * 8;

        var bone_data = read_bone_field(vc, dims, total_v, obj.deformed_pool_offset);
        // If no bone data, try 6-connected neighbors
        if bone_data.x == 0u && bone_data.y == 0u {
            let offsets = array<vec3<i32>, 6>(
                vec3<i32>(-1,0,0), vec3<i32>(1,0,0),
                vec3<i32>(0,-1,0), vec3<i32>(0,1,0),
                vec3<i32>(0,0,-1), vec3<i32>(0,0,1),
            );
            for (var ni = 0u; ni < 6u; ni++) {
                let nb = read_bone_field(vc + offsets[ni], dims, total_v, obj.deformed_pool_offset);
                if nb.x != 0u || nb.y != 0u {
                    bone_data = nb;
                    break;
                }
            }
        }
        if bone_data.x == 0u && bone_data.y == 0u {
            // No bone data at this voxel or neighbors — safe to skip ahead.
            // Use 4-voxel jump (half a brick) as a conservative skip.
            prev_opacity = 0.0;
            prev_t = t;
            t += vs * 4.0;
            continue;
        }

        let rest_pos = inverse_skin_position(local_pos, bone_data.x, bone_data.y, obj);
        let opacity = sample_rest_opacity(rest_pos, obj);

        if opacity >= OPACITY_THRESHOLD && prev_opacity < OPACITY_THRESHOLD {
            let frac = (OPACITY_THRESHOLD - prev_opacity) / (opacity - prev_opacity + 1e-10);
            return mix(prev_t, t, frac);
        }

        prev_opacity = opacity;
        prev_t = t;
        t += fine_step;
    }

    return -1.0;
}

// ── Tiled March ────────────────────────────────────────────────────────────

/// March all objects in the current pixel's tile, find closest surface hit.
fn march_tiled(origin: vec3<f32>, dir: vec3<f32>, pixel: vec2<u32>) -> MarchResult {
    var result: MarchResult;
    result.hit = false;
    result.t = MAX_FLOAT;
    result.material_id = 0u;
    result.secondary_material_id = 0u;
    result.blend_weight = 0u;
    result.object_id = 0u;
    result.obj_idx = 0u;

    let tile_x = pixel.x / OBJECT_TILE_SIZE;
    let tile_y = pixel.y / OBJECT_TILE_SIZE;
    let dims = vec2<u32>(textureDimensions(gbuf_position));
    let num_tiles_x = (dims.x + OBJECT_TILE_SIZE - 1u) / OBJECT_TILE_SIZE;
    let tile_id = tile_y * num_tiles_x + tile_x;

    let count = tile_object_counts[tile_id];
    if count == 0u {
        return result;
    }

    let base = tile_id * TILE_MAX_OBJECTS;

    for (var i = 0u; i < count; i++) {
        let obj_idx = tile_object_indices[base + i];
        let obj = objects[obj_idx];

        // Quick world-space AABB check
        let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
        let world_t = intersect_aabb(origin, 1.0 / safe_dir, obj.aabb_min.xyz, obj.aabb_max.xyz);
        if world_t.x > world_t.y || world_t.x > result.t {
            continue; // Miss or already found something closer
        }

        let local_t = march_object(origin, dir, obj_idx);
        if local_t < 0.0 {
            continue;
        }

        // Convert local-space t to world-space t
        let inv_world = obj.inverse_world;
        let local_origin = (inv_world * vec4<f32>(origin, 1.0)).xyz;
        let local_dir = normalize((inv_world * vec4<f32>(dir, 0.0)).xyz);
        let local_hit = local_origin + local_dir * local_t;

        // Transform hit back to world to get world-space t
        // Invert the inverse_world (which is world_to_local) to get local_to_world
        // Since we have the ray origin in world space, compute t from the world hit
        let world_hit_h = vec4<f32>(local_hit, 1.0);
        // Convert local-space t to world-space t.
        // Since inverse_world is a linear transform, the ray parameter scales by
        // the length of the transformed direction: world_t = local_t / |inv_world * dir|.
        let local_dir_unnorm = (inv_world * vec4<f32>(dir, 0.0)).xyz;
        let world_t_approx = local_t / max(length(local_dir_unnorm), 1e-10);

        if world_t_approx < result.t {
            result.hit = true;
            result.t = world_t_approx;
            result.object_id = obj.object_id;
            result.obj_idx = obj_idx;

            // Sample material at hit point
            let mat_data = sample_material_at_hit(local_hit, obj);
            result.material_id = mat_data.x;
            result.secondary_material_id = mat_data.y;
            result.blend_weight = mat_data.z;
        }
    }

    return result;
}

// ── Normal Computation ─────────────────────────────────────────────────────

/// Compute world-space normal from opacity field gradient.
fn compute_normal(hit_pos: vec3<f32>, obj_idx: u32) -> vec3<f32> {
    let obj = objects[obj_idx];
    let local_pos = (obj.inverse_world * vec4<f32>(hit_pos, 1.0)).xyz;

    if obj.is_skinned != 0u && obj.bone_count > 0u {
        return compute_normal_skinned(local_pos, obj);
    }

    return compute_normal_static(local_pos, obj);
}

/// Normal for static objects — gradient directly in local space.
fn compute_normal_static(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let eps = obj.voxel_size * 2.0;
    let gx = sample_opacity_trilinear(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_opacity_trilinear(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_opacity_trilinear(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_opacity_trilinear(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    let local_grad = -vec3<f32>(gx, gy, gz);

    let world_grad = (transpose(obj.inverse_world) * vec4<f32>(local_grad, 0.0)).xyz;
    let len = length(world_grad);
    if len < 1e-10 { return vec3<f32>(0.0, 1.0, 0.0); }
    return world_grad / len;
}

/// Normal for skinned objects — read bone data from deformed pool, inverse-skin,
/// compute gradient in rest-pose, forward-skin back.
fn compute_normal_skinned(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    // Read bone weights from deformed pool
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let vs = obj.voxel_size;
    let grid_size = vec3<f32>(dims) * vs * 8.0;
    let total_v = vec3<i32>(dims) * 8;

    let grid_pos = local_pos + grid_size * 0.5;
    let vc = vec3<i32>(floor(grid_pos / vs));

    var bone_data = read_bone_field(vc, dims, total_v, obj.deformed_pool_offset);
    if bone_data.x == 0u && bone_data.y == 0u {
        let offsets = array<vec3<i32>, 6>(
            vec3<i32>(-1,0,0), vec3<i32>(1,0,0),
            vec3<i32>(0,-1,0), vec3<i32>(0,1,0),
            vec3<i32>(0,0,-1), vec3<i32>(0,0,1),
        );
        for (var ni = 0u; ni < 6u; ni++) {
            let nb = read_bone_field(vc + offsets[ni], dims, total_v, obj.deformed_pool_offset);
            if nb.x != 0u || nb.y != 0u {
                bone_data = nb;
                break;
            }
        }
    }

    if bone_data.x == 0u && bone_data.y == 0u {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    // Inverse-skin to rest-pose
    let rest_pos = inverse_skin_position(local_pos, bone_data.x, bone_data.y, obj);

    // Gradient in rest-pose space
    let eps = vs * 2.0;
    let gx = sample_rest_opacity(rest_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_rest_opacity(rest_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_rest_opacity(rest_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_rest_opacity(rest_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_rest_opacity(rest_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_rest_opacity(rest_pos - vec3<f32>(0.0, 0.0, eps), obj);
    let rest_grad = -vec3<f32>(gx, gy, gz);

    // Forward-skin the gradient from rest → posed
    let posed_grad = forward_skin_dir(rest_grad, bone_data.x, bone_data.y, obj);

    // Transform from local → world space
    let world_grad = (transpose(obj.inverse_world) * vec4<f32>(posed_grad, 0.0)).xyz;
    let len = length(world_grad);
    if len < 1e-10 { return vec3<f32>(0.0, 1.0, 0.0); }
    return world_grad / len;
}

/// Sample per-voxel color from the color companion pool at a rest-pose position.
/// Returns packed RGB24 as u32 (0 = no color data).
fn sample_rest_color(rest_pos: vec3<f32>, obj: GpuObject) -> u32 {
    let vs = obj.voxel_size;
    let rest_dims = vec3<u32>(obj.rest_brick_map_dims_x, obj.rest_brick_map_dims_y, obj.rest_brick_map_dims_z);
    let rest_grid_size = vec3<f32>(rest_dims) * vs * 8.0;
    let gp = rest_pos + rest_grid_size * 0.5;

    if any(gp < vec3<f32>(0.0)) || any(gp >= rest_grid_size) {
        return 0u;
    }

    let vc = clamp(vec3<i32>(floor(gp / vs)), vec3<i32>(0), vec3<i32>(rest_dims) * 8 - vec3<i32>(1));
    let brick = vec3<u32>(vc) / vec3<u32>(8u);
    let local = vec3<u32>(vc) % vec3<u32>(8u);
    let flat_brick = brick.x + brick.y * rest_dims.x + brick.z * rest_dims.x * rest_dims.y;
    let slot = brick_maps[obj.rest_brick_map_offset + flat_brick];

    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return 0u;
    }

    let color_slot = color_companion_map[slot];
    if color_slot == EMPTY_SLOT {
        return 0u;
    }

    let vi = local.x + local.y * 8u + local.z * 64u;
    return color_pool_data[color_slot * 512u + vi];
}

// ── Motion Vector ──────────────────────────────────────────────────────────

fn compute_motion_vector(hit_pos: vec3<f32>, pixel: vec2<u32>) -> vec2<f32> {
    // Without a current view_proj matrix, we approximate motion as zero.
    // The prev_vp is used for reprojection; for MVP we return zero motion.
    let prev_clip = camera.prev_vp * vec4<f32>(hit_pos, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;
    // Current NDC from pixel coords
    let dims = vec2<f32>(textureDimensions(gbuf_position));
    let uv = (vec2<f32>(pixel.xy) + 0.5) / dims;
    let cur_ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return (cur_ndc - prev_ndc) * 0.5;
}

// ── Entry Point ────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(gbuf_position));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);

    // Generate camera ray with jitter
    let uv = (vec2<f32>(pixel.xy) + 0.5 + camera.jitter) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz
    );

    // March all objects in this pixel's tile
    let result = march_tiled(ray_origin, ray_dir, pixel.xy);

    if result.hit {
        var hit_pos = ray_origin + ray_dir * result.t;
        var normal = compute_normal(hit_pos, result.obj_idx);
        var final_mat_id = result.material_id;
        var final_sec_mat_id = result.secondary_material_id;
        var final_blend_weight = result.blend_weight;
        var final_t = result.t;

        // --- Opacity shell march: check if the hit material has a registered opacity shader ---
        // dispatch_opacity_shader returns -1.0 (sentinel) for shader IDs without an
        // opacity function. Probe with h_above=-1.0 to distinguish: real opacity shaders
        // return 0.0 (early-out for h_above<0), the sentinel returns -1.0.
        let hit_obj = objects[result.obj_idx];
        let shader_id = materials[final_mat_id].shader_id;
        if dispatch_opacity_shader(shader_id, vec3<f32>(0.0), -1.0, hit_obj, final_mat_id) > -0.5 {
            // Material has a registered opacity shader — march the shell above the surface.
            let inv_world = hit_obj.inverse_world;
            let local_origin = (inv_world * vec4<f32>(ray_origin, 1.0)).xyz;
            let local_dir = normalize((inv_world * vec4<f32>(ray_dir, 0.0)).xyz);
            let local_hit = (inv_world * vec4<f32>(hit_pos, 1.0)).xyz;

            // Compute surface normal in local space for h_above calculation
            let surface_normal_local = compute_local_normal(local_hit, hit_obj);

            // Shell height from shader params (param1 = height)
            let raw_shell_h = shader_params[final_mat_id].param1;
            let shell_h = select(0.5, raw_shell_h, raw_shell_h > 0.0);

            // Compute local-space t of the surface hit
            let local_dir_unnorm = (inv_world * vec4<f32>(ray_dir, 0.0)).xyz;
            let scale = length(local_dir_unnorm);
            let surface_t_local = result.t * scale;

            let shell_result = march_shell(
                local_origin, local_dir, surface_t_local,
                local_hit, surface_normal_local,
                shader_id, final_mat_id, shell_h, hit_obj,
            );

            if shell_result.t >= 0.0 {
                // Shell hit — replace the surface with the procedural geometry
                final_t = shell_result.t / scale; // convert back to world-space t
                hit_pos = ray_origin + ray_dir * final_t;
                normal = shell_result.normal_world;
                final_mat_id = shell_result.mat_id;
                // Shell geometry uses the same material, no secondary blend
                final_sec_mat_id = shell_result.mat_id;
                final_blend_weight = 0u;
            }
        }

        let motion = compute_motion_vector(hit_pos, pixel.xy);

        // Pack material data (same format as rkf-render)
        let packed_r = (final_mat_id & 0xFFFFu)
                     | ((final_sec_mat_id & 0xFFFFu) << 16u);
        let packed_g = (final_blend_weight & 0xFFu)
                     | ((result.object_id & 0xFFu) << 8u);

        // For skinned objects, sample per-voxel color at rest-pose position
        // and pack into gbuf_motion.z (same convention as rkf-render).
        var skinned_color_u32 = 0u;
        if hit_obj.is_skinned != 0u && hit_obj.bone_count > 0u {
            let local_hit = (hit_obj.inverse_world * vec4<f32>(hit_pos, 1.0)).xyz;
            let dims = vec3<u32>(hit_obj.brick_map_dims_x, hit_obj.brick_map_dims_y, hit_obj.brick_map_dims_z);
            let vs = hit_obj.voxel_size;
            let grid_size = vec3<f32>(dims) * vs * 8.0;
            let total_v = vec3<i32>(dims) * 8;
            let grid_pos = local_hit + grid_size * 0.5;
            let vc = vec3<i32>(floor(grid_pos / vs));

            var bone_data = read_bone_field(vc, dims, total_v, hit_obj.deformed_pool_offset);
            if bone_data.x == 0u && bone_data.y == 0u {
                let offsets = array<vec3<i32>, 6>(
                    vec3<i32>(-1,0,0), vec3<i32>(1,0,0),
                    vec3<i32>(0,-1,0), vec3<i32>(0,1,0),
                    vec3<i32>(0,0,-1), vec3<i32>(0,0,1),
                );
                for (var ni = 0u; ni < 6u; ni++) {
                    let nb = read_bone_field(vc + offsets[ni], dims, total_v, hit_obj.deformed_pool_offset);
                    if nb.x != 0u || nb.y != 0u { bone_data = nb; break; }
                }
            }
            if bone_data.x != 0u || bone_data.y != 0u {
                let rest_pos = inverse_skin_position(local_hit, bone_data.x, bone_data.y, hit_obj);
                skinned_color_u32 = sample_rest_color(rest_pos, hit_obj);
            }
        }

        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, final_t));
        textureStore(gbuf_normal,   coord, vec4<f32>(normal, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(motion, bitcast<f32>(skinned_color_u32), 0.0));
    } else {
        // Miss — clear G-buffer
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
