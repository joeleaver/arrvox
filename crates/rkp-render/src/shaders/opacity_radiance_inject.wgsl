// Opacity-field radiance injection compute shader — replaces SDF radiance_inject.wgsl.
//
// Dispatched over the Level 0 radiance volume (64³). For each texel:
// 1. Compute world position from volume uniforms
// 2. Sample opacity via coarse field + BVH — classify as surface / interior / exterior
// 3. Surface voxels: compute normal via opacity gradient, evaluate direct lighting
// 4. Write radiance (RGB) + opacity (A) to the 3D storage texture
//
// Workgroup size: 4×4×4 → dispatch 16×16×16 for 64³ volume.

// ---------- Structs (must match Rust/shade.wgsl layouts exactly) ----------

struct VoxelSample {
    word0: u32, // bits 0-15 = f16 opacity, bits 16-23 = blend_weight, bits 24-31 = reserved
    word1: u32, // bits 0-15 = primary material_id, bits 16-31 = secondary material_id
}

struct GpuObject {
    inverse_world: mat4x4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    brick_map_offset: u32,
    brick_map_dims_x: u32,
    brick_map_dims_y: u32,
    brick_map_dims_z: u32,
    voxel_size: f32,
    material_id: u32,
    geom_type: u32,
    blend_mode: u32,
    blend_radius: f32,
    sdf_param_0: f32,
    sdf_param_1: f32,
    sdf_param_2: f32,
    sdf_param_3: f32,
    accumulated_scale_x: f32,
    accumulated_scale_y: f32,
    accumulated_scale_z: f32,
    lod_level: u32,
    object_id: u32,
    primitive_type: u32,
    geometry_aabb_min_x: f32, geometry_aabb_min_y: f32, geometry_aabb_min_z: f32,
    geometry_aabb_max_x: f32, geometry_aabb_max_y: f32, geometry_aabb_max_z: f32,
    is_skinned: u32, bone_count: u32, bone_buffer_offset: u32, rest_brick_map_offset: u32,
    rest_brick_map_dims_x: u32, rest_brick_map_dims_y: u32, rest_brick_map_dims_z: u32,
    shell_height: f32, sdf_shader_id: u32, sdf_shader_material: u32,
    deformed_pool_offset: u32, _pad10: u32, _pad11: u32, _pad12: u32, _pad13: u32,
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

struct SceneUniformsV2 {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

struct Material {
    albedo_r: f32, albedo_g: f32, albedo_b: f32, roughness: f32,
    metallic: f32, emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32, subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32, ior: f32,
    noise_scale: f32, noise_strength: f32, noise_channels: u32,
    shader_id: u32, _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
}

struct Light {
    light_type: u32,
    pos_x: f32, pos_y: f32, pos_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    color_r: f32, color_g: f32, color_b: f32,
    intensity: f32, range: f32,
    inner_angle: f32, outer_angle: f32,
    cookie_index: i32, shadow_caster: u32,
}

struct InjectUniforms {
    num_lights: u32,
    max_shadow_lights: u32,
    _pad0: u32,
    _pad1: u32,
}

struct RadianceVolumeUniforms {
    center:      vec4<f32>,
    voxel_sizes: vec4<f32>,
    inv_extents: vec4<f32>,
    params:      vec4<u32>,   // x = dim, y = num_levels
}

struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,
    dims: vec4<u32>,
    voxel_size: f32,
    inv_voxel_size: f32,
    _cf_pad0: f32,
    _cf_pad1: f32,
}

// ---------- Bindings ----------

// Group 0: v2 Scene data (same layout as ray march / shade group)
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;
// binding 3 = camera uniforms (unused here)
@group(0) @binding(4) var<uniform> v2_scene: SceneUniformsV2;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 1: Material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: Lights + inject uniforms
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var<uniform> inject: InjectUniforms;

// Group 3: Radiance volume Level 0 write + volume uniforms
@group(3) @binding(0) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(1) var<uniform> vol: RadianceVolumeUniforms;

// Group 4: Coarse acceleration field
@group(4) @binding(0) var coarse_field: texture_3d<f32>;
@group(4) @binding(1) var coarse_sampler: sampler;
@group(4) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// Group 5: per-material shader extension parameters
struct ShaderParams {
    param0: f32, param1: f32, param2: f32, param3: f32,
    param4: f32, param5: f32, param6: f32, param7: f32,
}
@group(5) @binding(0) var<storage, read> shader_params: array<ShaderParams>;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const BVH_STACK_SIZE: u32 = 32u;

const GEOM_TYPE_NONE: u32       = 0u;
const GEOM_TYPE_ANALYTICAL: u32 = 1u;
const GEOM_TYPE_VOXELIZED: u32  = 2u;
const GEOM_TYPE_PROCEDURAL: u32 = 3u;

const INJECT_SHADOW_MAX_DIST: f32 = 20.0;
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

// ---------- Opacity Helpers ----------

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

fn extract_material_id(word1: u32) -> u32 {
    return word1 & 0xFFFFu;
}

// ---------- Opacity Field Sampling ----------

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

fn sample_opacity_trilinear(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);
    if outside_dist > brick_extent * 2.0 {
        return 0.0;
    }
    // Geometry AABB early-out: skip empty expanded brick-map region.
    let geom_min = vec3<f32>(obj.geometry_aabb_min_x, obj.geometry_aabb_min_y, obj.geometry_aabb_min_z);
    let geom_max = vec3<f32>(obj.geometry_aabb_max_x, obj.geometry_aabb_max_y, obj.geometry_aabb_max_z);
    if geom_max.x > geom_min.x {
        let geom_closest = clamp(local_pos, geom_min, geom_max);
        let geom_dist = length(local_pos - geom_closest);
        if geom_dist > brick_extent {
            return 0.0;
        }
    }
    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    // Pre-filter: read nearest voxel. If clearly empty or solid, skip trilinear.
    let nn = clamp(v0, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let nn_brick = vec3<u32>(nn / vec3<i32>(8));
    let nn_local = vec3<u32>(nn % vec3<i32>(8));
    let nn_flat = nn_brick.x + nn_brick.y * dims.x + nn_brick.z * dims.x * dims.y;
    let nn_slot = brick_maps[obj.brick_map_offset + nn_flat];
    if nn_slot == EMPTY_SLOT { return 0.0; }
    if nn_slot == INTERIOR_SLOT { return 1.0; }
    let nn_idx = nn_slot * 512u + nn_local.x + nn_local.y * 8u + nn_local.z * 64u;
    let nn_opacity = extract_opacity(brick_pool[nn_idx].word0);
    if nn_opacity < 0.01 { return 0.0; }
    if nn_opacity > 0.99 { return 1.0; }

    // Same-brick fast path: all 8 corners in one brick → 1 brick_maps lookup.
    let v1 = v0 + vec3<i32>(1);
    if all(v0 >= vec3<i32>(0)) && all(v1 < total_voxels) {
        let b0 = v0 / vec3<i32>(8);
        let b1 = v1 / vec3<i32>(8);
        if all(b0 == b1) {
            let l = vec3<u32>(v0 - b0 * vec3<i32>(8));
            let base = nn_slot * 512u;
            let c000 = extract_opacity(brick_pool[base + l.x + l.y * 8u + l.z * 64u].word0);
            let c100 = extract_opacity(brick_pool[base + l.x + 1u + l.y * 8u + l.z * 64u].word0);
            let c010 = extract_opacity(brick_pool[base + l.x + (l.y + 1u) * 8u + l.z * 64u].word0);
            let c110 = extract_opacity(brick_pool[base + l.x + 1u + (l.y + 1u) * 8u + l.z * 64u].word0);
            let c001 = extract_opacity(brick_pool[base + l.x + l.y * 8u + (l.z + 1u) * 64u].word0);
            let c101 = extract_opacity(brick_pool[base + l.x + 1u + l.y * 8u + (l.z + 1u) * 64u].word0);
            let c011 = extract_opacity(brick_pool[base + l.x + (l.y + 1u) * 8u + (l.z + 1u) * 64u].word0);
            let c111 = extract_opacity(brick_pool[base + l.x + 1u + (l.y + 1u) * 8u + (l.z + 1u) * 64u].word0);
            let c00 = mix(c000, c100, t.x);
            let c10 = mix(c010, c110, t.x);
            let c01 = mix(c001, c101, t.x);
            let c11 = mix(c011, c111, t.x);
            let c0 = mix(c00, c10, t.y);
            let c1 = mix(c01, c11, t.y);
            return mix(c0, c1, t.z);
        }
    }

    // Cross-brick fallback.
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

// ---------- Object Evaluation ----------

/// Evaluate object opacity at a world-space position. Returns (opacity, material_id).
fn evaluate_object_opacity(world_pos: vec3<f32>, obj_idx: u32) -> vec2<f32> {
    let obj = objects[obj_idx];
    if obj.geom_type == GEOM_TYPE_NONE || obj.geom_type == GEOM_TYPE_PROCEDURAL {
        return vec2<f32>(0.0, 0.0);
    }
    // inverse_world is world-space — transform directly.
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    var opacity: f32;
    var mat_id = obj.material_id;
    // All object types use trilinear opacity sampling (no analytical SDF primitives).
    opacity = sample_opacity_trilinear(local_pos, obj);
    return vec2<f32>(opacity, f32(mat_id));
}

// ---------- Coarse Field Sampling ----------

fn sample_coarse_field(cam_rel_pos: vec3<f32>) -> f32 {
    let field_pos = cam_rel_pos - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }
    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
}

// ---------- Opacity BVH Point Query (coarse field + BVH) ----------

// Coordinate space note:
// - BVH AABBs are in WORLD space (built from Scene object AABBs).
// - GpuObject.inverse_world is WORLD-SPACE (transforms world pos to local).
// - Coarse field expects CAMERA-RELATIVE input.
// - vol.center.xyz = camera world position, so cam_rel = world_pos - vol.center.xyz.

/// Convert world-space position to camera-relative.
fn world_to_cam_rel(world_pos: vec3<f32>) -> vec3<f32> {
    return world_pos - vol.center.xyz;
}

/// Sample opacity at a world-space position via BVH traversal. Returns max opacity.
fn sample_opacity_bvh(world_pos: vec3<f32>) -> f32 {
    let cam_rel = world_to_cam_rel(world_pos);

    // Phase 1: coarse field check (camera-relative).
    let coarse_dist = sample_coarse_field(cam_rel);
    if coarse_dist > COARSE_NEAR_THRESHOLD {
        return 0.0;
    }

    // Phase 2: BVH traversal (world-space AABBs).
    if v2_scene.num_objects == 0u {
        return 0.0;
    }

    var max_opacity = 0.0;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
        let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

        // BVH AABBs are world-space; test point containment with margin.
        let closest = clamp(world_pos, node_min, node_max);
        let box_dist = length(closest - world_pos);
        // For opacity, we cannot early-out by distance comparison like SDF.
        // Use a generous threshold to skip clearly distant nodes.
        if box_dist > vol.voxel_sizes.x * 8.0 {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let result = evaluate_object_opacity(world_pos, leaf_obj_idx);
                max_opacity = max(max_opacity, result.x);
            }
        } else {
            if stack_ptr < BVH_STACK_SIZE - 1u {
                stack[stack_ptr] = node.left;
                stack_ptr += 1u;
                stack[stack_ptr] = node.right_or_object;
                stack_ptr += 1u;
            }
        }
    }

    return max_opacity;
}

/// Sample opacity and return (opacity, material_id, object_index) at a world-space position.
/// Returns the material and object from the object with highest opacity.
fn sample_opacity_with_material(world_pos: vec3<f32>) -> vec3<f32> {
    let cam_rel = world_to_cam_rel(world_pos);

    // Coarse field early-out (camera-relative).
    let coarse_dist = sample_coarse_field(cam_rel);
    if coarse_dist > COARSE_NEAR_THRESHOLD {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    if v2_scene.num_objects == 0u {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    var max_opacity = 0.0;
    var mat_id = 0.0;
    var obj_idx_f = 0.0;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
        let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

        let closest = clamp(world_pos, node_min, node_max);
        let box_dist = length(closest - world_pos);
        if box_dist > vol.voxel_sizes.x * 8.0 {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let result = evaluate_object_opacity(world_pos, leaf_obj_idx);
                if result.x > max_opacity {
                    max_opacity = result.x;
                    mat_id = result.y;
                    obj_idx_f = f32(leaf_obj_idx);
                }
            }
        } else {
            if stack_ptr < BVH_STACK_SIZE - 1u {
                stack[stack_ptr] = node.left;
                stack_ptr += 1u;
                stack[stack_ptr] = node.right_or_object;
                stack_ptr += 1u;
            }
        }
    }

    return vec3<f32>(max_opacity, mat_id, obj_idx_f);
}

// ---------- Opacity Gradient Normal ----------

/// Compute gradient normal using direct trilinear samples on the dominant object.
/// Replaces 6 full BVH traversals with 6 direct trilinear reads.
fn opacity_gradient_normal_direct(world_pos: vec3<f32>, obj_idx: u32) -> vec3<f32> {
    let obj = objects[obj_idx];
    let eps = vol.voxel_sizes.x * 0.5;
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    // Transform world-space epsilon offsets to local space
    let dx = (obj.inverse_world * vec4<f32>(eps, 0.0, 0.0, 0.0)).xyz;
    let dy = (obj.inverse_world * vec4<f32>(0.0, eps, 0.0, 0.0)).xyz;
    let dz = (obj.inverse_world * vec4<f32>(0.0, 0.0, eps, 0.0)).xyz;

    let nx = sample_opacity_trilinear(local_pos + dx, obj) - sample_opacity_trilinear(local_pos - dx, obj);
    let ny = sample_opacity_trilinear(local_pos + dy, obj) - sample_opacity_trilinear(local_pos - dy, obj);
    let nz = sample_opacity_trilinear(local_pos + dz, obj) - sample_opacity_trilinear(local_pos - dz, obj);
    let grad = vec3<f32>(nx, ny, nz);
    let len = length(grad);
    if len < 1e-10 { return vec3<f32>(0.0, 1.0, 0.0); }
    return -grad / len;
}

// ---------- Transmittance Shadow ----------

fn inject_shadow(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> f32 {
    var transmittance = 1.0;
    var t = 0.02;
    let step_size = 0.08;
    for (var i = 0u; i < 32u; i++) {
        if t > max_dist || transmittance < 0.01 { break; }
        let pos = origin + dir * t;
        let cam_rel = pos - vol.center.xyz;
        let coarse_dist = sample_coarse_field(cam_rel);
        if coarse_dist > 0.5 {
            t += coarse_dist * 0.8;
            continue;
        }
        let opacity = sample_opacity_bvh(pos);
        transmittance *= (1.0 - opacity);
        t += mix(step_size * 2.0, step_size * 0.5, opacity);
    }
    return transmittance;
}

// ---------- Light Attenuation ----------

fn distance_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = dist * dist;
    let r2 = range * range;
    let factor = d2 / r2;
    let w = clamp(1.0 - factor, 0.0, 1.0);
    return (w * w) / max(d2, 0.0001);
}

// ---------- Entry Point ----------

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = vol.params.x;
    if gid.x >= dim || gid.y >= dim || gid.z >= dim {
        return;
    }

    // Compute world position of this texel centre.
    // The radiance volume center is in world space.
    let voxel_size = vol.voxel_sizes.x; // Level 0
    let half_extent = voxel_size * f32(dim) * 0.5;
    let pos = vol.center.xyz
        + (vec3<f32>(gid) + 0.5) * voxel_size
        - vec3<f32>(half_extent);

    // Sample opacity + material + object index in a single BVH traversal.
    let opacity_mat = sample_opacity_with_material(pos);
    let opacity = opacity_mat.x;

    // Deep interior: opacity > 0.7 → opaque black (blocks light)
    if opacity > 0.7 {
        textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    // Exterior: opacity < 0.01 → transparent
    if opacity < 0.01 {
        textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // --- Near surface (opacity in [0.01, 0.7]): compute direct lighting ---

    let hit_obj_idx = u32(opacity_mat.z);
    let hit_obj = objects[hit_obj_idx];

    // Gradient normal via direct trilinear on the dominant object (no BVH).
    let normal = opacity_gradient_normal_direct(pos, hit_obj_idx);

    // Concavity probe: direct trilinear on the same object (no BVH).
    let probe_local = (hit_obj.inverse_world * vec4<f32>(pos + normal * voxel_size * 3.0, 1.0)).xyz;
    let probe_opacity = sample_opacity_trilinear(probe_local, hit_obj);
    let concavity = 1.0 - smoothstep(0.0, 0.5, probe_opacity);

    // Material from the already-computed BVH result
    let mat_id = u32(opacity_mat.y);
    let mat = materials[mat_id];
    let albedo = vec3<f32>(mat.albedo_r, mat.albedo_g, mat.albedo_b);

    // Emissive self-injection
    let emission = vec3<f32>(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;

    var radiance = vec3<f32>(0.0);
    var shadow_count = 0u;
    let max_shadow = inject.max_shadow_lights;
    let num_lights_val = inject.num_lights;

    // Evaluate direct lighting (Lambertian diffuse — no specular for GI bounce).
    // Light positions are camera-relative in the light buffer, but injection
    // works in world space. For directional lights this doesn't matter (only
    // direction). For point/spot lights, convert camera-relative → world-space
    // by adding vol.center.xyz (which equals the camera world position).
    for (var i = 0u; i < num_lights_val; i++) {
        let light = lights[i];
        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);

        var light_dir: vec3<f32>;
        var atten = 1.0;

        if light.light_type == 0u {
            // Directional
            light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
        } else if light.light_type == 1u {
            // Point — light pos is camera-relative; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + vol.center.xyz;
            let to_light = light_pos - pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            atten = distance_attenuation(dist, light.range);
        } else {
            // Spot — light pos is camera-relative; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + vol.center.xyz;
            let spot_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let to_light = light_pos - pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            let cos_angle = dot(-light_dir, spot_dir);
            let cos_outer = cos(light.outer_angle);
            let cos_inner = cos(light.inner_angle);
            let spot = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
            atten = spot * distance_attenuation(dist, light.range);
        }

        if atten < 0.001 {
            continue;
        }

        let n_dot_l = max(dot(normal, light_dir), 0.0);
        if n_dot_l <= 0.0 {
            continue;
        }

        // Transmittance shadow (only for first max_shadow shadow casters)
        var shadow = 1.0;
        if light.shadow_caster == 1u && shadow_count < max_shadow {
            let shadow_origin = pos + normal * 0.02;
            shadow = inject_shadow(shadow_origin, light_dir, INJECT_SHADOW_MAX_DIST);
            shadow_count += 1u;
        }

        // Lambertian diffuse (no specular for GI bounce)
        radiance += albedo / PI * light_color * light.intensity * atten * n_dot_l * shadow;
    }

    // Add emission
    radiance += emission;

    // Apply concavity attenuation to prevent bright halos at junctions.
    // Use opacity directly as output alpha — near-surface voxels are partially opaque.
    textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(radiance * concavity, opacity));
}
