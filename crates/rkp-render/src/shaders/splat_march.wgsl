// Splat March — surface-finding through trilinear opacity field, G-buffer output.
//
// Replaces rkf-render's ray_march.wgsl. Same bind groups 0-2, same G-buffer
// output format. The difference: fixed-step march through opacity (not sphere
// tracing through SDF distance).

// ── Constants ──────────────────────────────────────────────────────────────

const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const SURFACE_THRESHOLD: f32 = 0.0; // SDF surface is at distance = 0
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
    sdf_type: u32,               // @ 120
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

// ── Voxel Extraction ───────────────────────────────────────────────────────

/// Extract f16 distance from word0 bits 0–15, returned as f32.
fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0 & 0xFFFFu).x;
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

/// Sample a single voxel's SDF distance from the brick pool.
/// Returns large positive for EMPTY_SLOT (exterior), large negative for INTERIOR_SLOT.
fn sample_distance_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                      total_voxels: vec3<i32>, vs: f32) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return vs * 8.0;
    }
    if slot == INTERIOR_SLOT {
        return -(vs * 2.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_distance(brick_pool[idx].word0);
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

/// Trilinear interpolation of the SDF distance field at a local-space position.
fn sample_distance_trilinear(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);

    // Outside the grid — return large positive (exterior)
    if outside_dist > brick_extent * 2.0 {
        return outside_dist;
    }

    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    let c000 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 0), dims, total_voxels, vs);
    let c100 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 0), dims, total_voxels, vs);
    let c010 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 0), dims, total_voxels, vs);
    let c110 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 0), dims, total_voxels, vs);
    let c001 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 1), dims, total_voxels, vs);
    let c101 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 1), dims, total_voxels, vs);
    let c011 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 1), dims, total_voxels, vs);
    let c111 = sample_distance_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 1), dims, total_voxels, vs);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z) + outside_dist;
}

/// Sample a single voxel for gradient computation.
/// INTERIOR_SLOT returns -vs*8 (matching EMPTY_SLOT's +vs*8 in magnitude)
/// to produce correct gradient direction across sentinel boundaries.
fn sample_distance_at_grad(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                           total_voxels: vec3<i32>, vs: f32) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return vs * 8.0;
    }
    if slot == INTERIOR_SLOT {
        return -(vs * 8.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_distance(brick_pool[idx].word0);
}

/// Trilinear SDF sampling for gradient computation (gradient-safe sentinel values).
fn sample_distance_trilinear_grad(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);

    if outside_dist > brick_extent * 2.0 {
        return outside_dist;
    }

    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    let c000 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,0,0), dims, total_voxels, vs);
    let c100 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,0,0), dims, total_voxels, vs);
    let c010 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,1,0), dims, total_voxels, vs);
    let c110 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,1,0), dims, total_voxels, vs);
    let c001 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,0,1), dims, total_voxels, vs);
    let c101 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,0,1), dims, total_voxels, vs);
    let c011 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,1,1), dims, total_voxels, vs);
    let c111 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,1,1), dims, total_voxels, vs);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z) + outside_dist;
}

/// Sample binary density field: solid=1, empty=0, trilinearly interpolated.
/// Uses the SIGN of SDF distance (not magnitude) to avoid discontinuities
/// from SDF magnitude artifacts at brick boundaries.
fn sample_density(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));

    if any(grid_pos < vec3<f32>(-brick_extent)) || any(grid_pos > grid_size + vec3<f32>(brick_extent)) {
        return 0.0;
    }

    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    // Sample 8 corners, apply smoothstep on distance for a gradual 0→1 transition.
    // This avoids the staircase banding of binary density and the pockmarks of raw SDF.
    let w = vs * 1.5; // transition half-width
    let d000 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,0,0), dims, total_voxels, vs);
    let d100 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,0,0), dims, total_voxels, vs);
    let d010 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,1,0), dims, total_voxels, vs);
    let d110 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,1,0), dims, total_voxels, vs);
    let d001 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,0,1), dims, total_voxels, vs);
    let d101 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,0,1), dims, total_voxels, vs);
    let d011 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(0,1,1), dims, total_voxels, vs);
    let d111 = sample_distance_at_grad(obj.brick_map_offset, v0 + vec3<i32>(1,1,1), dims, total_voxels, vs);

    let b000 = 1.0 - smoothstep(-w, w, d000);
    let b100 = 1.0 - smoothstep(-w, w, d100);
    let b010 = 1.0 - smoothstep(-w, w, d010);
    let b110 = 1.0 - smoothstep(-w, w, d110);
    let b001 = 1.0 - smoothstep(-w, w, d001);
    let b101 = 1.0 - smoothstep(-w, w, d101);
    let b011 = 1.0 - smoothstep(-w, w, d011);
    let b111 = 1.0 - smoothstep(-w, w, d111);

    let b00 = mix(b000, b100, t.x);
    let b10 = mix(b010, b110, t.x);
    let b01 = mix(b001, b101, t.x);
    let b11 = mix(b011, b111, t.x);
    let b0 = mix(b00, b10, t.y);
    let b1 = mix(b01, b11, t.y);
    return mix(b0, b1, t.z);
}

/// Compute surface normal from SDF gradient (gradient-safe sentinel values).
/// Matches rkf-render's sample_voxelized_gradient exactly.
fn sample_sdf_gradient(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let eps = obj.voxel_size * 2.0;
    let gx = sample_distance_trilinear_grad(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_distance_trilinear_grad(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_distance_trilinear_grad(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_distance_trilinear_grad(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_distance_trilinear_grad(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_distance_trilinear_grad(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    return vec3<f32>(gx, gy, gz);
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

// ── Per-Object March ───────────────────────────────────────────────────────

/// March a ray through a single object's opacity field.
/// Returns the t value of the surface hit, or -1.0 on miss.
fn march_object(origin: vec3<f32>, dir: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    let inv_world = obj.inverse_world;

    // Transform ray to local space
    let local_origin = (inv_world * vec4<f32>(origin, 1.0)).xyz;
    let local_dir = normalize((inv_world * vec4<f32>(dir, 0.0)).xyz);
    let safe_dir = select(local_dir, vec3<f32>(1e-10), abs(local_dir) < vec3<f32>(1e-10));
    let inv_local_dir = 1.0 / safe_dir;

    // Compute local-space AABB from brick grid dimensions.
    // The grid is centered at the local origin: [-grid_size/2, +grid_size/2].
    let brick_extent = obj.voxel_size * 8.0;
    let dims = vec3<f32>(
        f32(obj.brick_map_dims_x),
        f32(obj.brick_map_dims_y),
        f32(obj.brick_map_dims_z),
    );
    let half_grid = dims * brick_extent * 0.5;
    let local_aabb_min = -half_grid;
    let local_aabb_max = half_grid;
    let t_range = intersect_aabb(local_origin, inv_local_dir, local_aabb_min, local_aabb_max);
    if t_range.x > t_range.y {
        return -1.0; // Ray misses AABB
    }

    // Sphere tracing with overshoot detection and bisection.
    // Mirrors rkf-render's proven approach for voxelized SDFs.
    let hit_threshold = obj.voxel_size * 0.01;
    let min_step = obj.voxel_size * 0.1;

    var t = t_range.x;
    var prev_dist = MAX_FLOAT;
    var prev_t = t;

    for (var step = 0u; step < MAX_MARCH_STEPS; step++) {
        if t > t_range.y {
            break;
        }

        let local_pos = local_origin + safe_dir * t;
        let dist = sample_distance_trilinear(local_pos, obj);

        // Overshoot detection: distance went negative (stepped through surface).
        // Bisect to find the zero-crossing.
        if dist < 0.0 && prev_dist > hit_threshold {
            var lo = prev_t;
            var hi = t;
            for (var b = 0u; b < 8u; b++) {
                let mid = (lo + hi) * 0.5;
                let mid_pos = local_origin + safe_dir * mid;
                let mid_dist = sample_distance_trilinear(mid_pos, obj);
                if mid_dist < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let refined_t = (lo + hi) * 0.5;
            return refined_t;
        }

        // Direct hit: distance is small and positive
        if dist < hit_threshold {
            return t;
        }

        prev_dist = dist;
        prev_t = t;
        // Step by distance, but use a conservative multiplier to avoid overshooting
        // voxelized SDF fields where trilinear distances can be inaccurate.
        t += max(dist * 0.8, min_step);
    }

    return -1.0; // No surface found
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
        // We need local_to_world. For now, compute world t from the ray equation.
        // world_hit = origin + dir * world_t => world_t = dot(world_hit - origin, dir) / dot(dir, dir)
        // But we don't have local_to_world directly. Use the AABB relationship.
        // Actually, we can compute: world_t ≈ local_t * |local_dir| / |world_dir|
        // but local_dir is normalize(inv_world * dir), so the scale is embedded.
        let local_dir_unnorm = (inv_world * vec4<f32>(dir, 0.0)).xyz;
        let scale = length(local_dir_unnorm);
        let world_t_approx = local_t / scale;

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

    // Smooth density gradient: smoothstep on SDF distance, then central differences.
    // Produces normals free of both pockmarks and staircase banding.
    let eps = obj.voxel_size * 2.0;
    let gx = sample_density(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_density(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_density(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_density(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_density(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_density(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    let local_grad = -vec3<f32>(gx, gy, gz);

    // Transform gradient from local → world space
    let world_grad = (transpose(obj.inverse_world) * vec4<f32>(local_grad, 0.0)).xyz;

    let len = length(world_grad);
    if len < 1e-10 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return world_grad / len;
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
        let hit_pos = ray_origin + ray_dir * result.t;
        let normal = compute_normal(hit_pos, result.obj_idx);
        let motion = compute_motion_vector(hit_pos, pixel.xy);

        // Pack material data (same format as rkf-render)
        let packed_r = (result.material_id & 0xFFFFu)
                     | ((result.secondary_material_id & 0xFFFFu) << 16u);
        let packed_g = (result.blend_weight & 0xFFu)
                     | ((result.object_id & 0xFFu) << 8u);

        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, result.t));
        textureStore(gbuf_normal,   coord, vec4<f32>(normal, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(motion, 0.0, 0.0));
    } else {
        // Miss — clear G-buffer
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
