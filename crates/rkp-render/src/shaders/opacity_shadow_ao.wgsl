// Half-resolution shadow + AO compute pass for opacity-field geometry.
//
// Replaces rkf-render's shadow_ao.wgsl — uses transmittance marching through
// the opacity field instead of SDF sphere-tracing. Same bind groups (0-5),
// same entry point signature, same output format (Rgba8Unorm: R=shadow,
// G=AO, B=cloud_transmittance, A=1.0).
//
// Dispatched at half the internal resolution. For each half-res pixel:
// 1. Read G-buffer position and normal at the corresponding full-res coord
// 2. Compute shadow via fixed-step transmittance marching through opacity field
// 3. Compute ambient occlusion via opacity probes along normal
// 4. Write (shadow, ao, cloud_transmittance, 1.0) to Rgba8Unorm storage texture

// ---------- Structs (must match Rust/shade layouts) ----------

struct VoxelSample {
    word0: u32,
    word1: u32,
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
    sdf_type: u32,
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
    _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: u32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32,
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

struct Light {
    light_type: u32,
    pos_x: f32, pos_y: f32, pos_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    color_r: f32, color_g: f32, color_b: f32,
    intensity: f32, range: f32,
    inner_angle: f32, outer_angle: f32,
    cookie_index: i32, shadow_caster: u32,
}

struct ShadeUniforms {
    debug_mode: u32,
    num_lights: u32,
    _su_pad0: u32,
    shadow_budget_k: u32,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_params: vec4<f32>,
    cam_forward: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    shadow_params: vec4<f32>,  // x=shadow_softness, y=shadow_density, z=ambient_intensity, w=shadow_fill
    cloud_shadow_params: vec4<f32>, // x=cloud_base, y=cloud_coverage, z=cloud_shadow_enabled, w=unused
    ambient_sky: vec4<f32>,   // xyz=precomputed hemisphere-average sky irradiance, w=unused
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

// Group 0: G-buffer read (same layout as shade group 0)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;
@group(0) @binding(3) var gbuf_motion:   texture_2d<f32>;

// Group 1: GpuScene (same layout as shade group 4 / ray_march group 0)
@group(1) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(1) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(1) @binding(2) var<storage, read> objects: array<GpuObject>;
// binding 3 = camera uniforms (not used here)
@group(1) @binding(4) var<uniform> v2_scene: SceneUniformsV2;
@group(1) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 2: Shade uniforms + lights
@group(2) @binding(0) var<uniform> shade_uniforms: ShadeUniforms;
@group(2) @binding(1) var<storage, read> lights: array<Light>;

// Group 3: Coarse field
@group(3) @binding(0) var coarse_field: texture_3d<f32>;
@group(3) @binding(1) var coarse_sampler: sampler;
@group(3) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// Group 4: Output
@group(4) @binding(0) var shadow_ao_out: texture_storage_2d<rgba8unorm, write>;

// Group 5: Cloud shadow map
@group(5) @binding(0) var cloud_shadow_tex: texture_2d<f32>;
@group(5) @binding(1) var cloud_shadow_smp: sampler;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const BVH_STACK_SIZE: u32 = 32u;

const SDF_TYPE_NONE: u32       = 0u;
const SDF_TYPE_ANALYTICAL: u32 = 1u;
const SDF_TYPE_VOXELIZED: u32  = 2u;
const SDF_TYPE_PROCEDURAL: u32 = 3u;

const PRIM_SPHERE: u32   = 0u;
const PRIM_BOX: u32      = 1u;
const PRIM_CAPSULE: u32  = 2u;
const PRIM_TORUS: u32    = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_PLANE: u32    = 5u;

// Shadow parameters
const SHADOW_MAX_DIST: f32 = 12.0;
const SHADOW_BIAS: f32 = 0.08;
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

// Opacity AO parameters
const AO_STEP_SIZE: f32 = 0.08;
const AO_STRENGTH: f32 = 1.5;

// ---------- VoxelSample Helpers ----------

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

// ---------- SDF Primitives (kept for bind group layout compatibility) ----------

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let q = vec3<f32>(p.x, max(abs(p.y) - half_height, 0.0), p.z);
    return length(q) - radius;
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normal) + dist;
}

fn evaluate_analytical(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    switch obj.primitive_type {
        case PRIM_SPHERE: { return sdf_sphere(local_pos, obj.sdf_param_0); }
        case PRIM_BOX: { return sdf_box(local_pos, vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)); }
        case PRIM_CAPSULE: { return sdf_capsule(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_TORUS: { return sdf_torus(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_CYLINDER: { return sdf_cylinder(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_PLANE: { return sdf_plane(local_pos, normalize(vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)), 0.0); }
        default: { return MAX_FLOAT; }
    }
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

    // Far outside the grid — transparent
    if outside_dist > brick_extent * 2.0 {
        return 0.0;
    }

    // Geometry AABB early-out
    let geom_min = vec3<f32>(obj.geometry_aabb_min_x, obj.geometry_aabb_min_y, obj.geometry_aabb_min_z);
    let geom_max = vec3<f32>(obj.geometry_aabb_max_x, obj.geometry_aabb_max_y, obj.geometry_aabb_max_z);
    if geom_max.x > geom_min.x {
        let geom_closest = clamp(local_pos, geom_min, geom_max);
        let geom_dist = length(local_pos - geom_closest);
        if geom_dist > brick_extent {
            return 0.0;
        }
    }

    // Trilinear interpolation of opacity values
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

// ---------- Object Opacity Evaluation ----------

fn evaluate_object_opacity(world_pos: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    if obj.sdf_type == SDF_TYPE_NONE {
        return 0.0;
    }
    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        // Splat objects are never analytical — return transparent
        return 0.0;
    }
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    // SDF_TYPE_VOXELIZED and SDF_TYPE_PROCEDURAL both sample from the brick map.
    return sample_opacity_trilinear(local_pos, obj);
}

// ---------- Coarse Field ----------

fn sample_coarse_field(cam_rel_pos: vec3<f32>) -> f32 {
    let field_pos = cam_rel_pos - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }
    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
}

// ---------- BVH Opacity Query ----------

fn sample_opacity_bvh(pos: vec3<f32>) -> f32 {
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
        let closest = clamp(pos, node_min, node_max);
        let box_dist = length(closest - pos);
        // For opacity, we can skip nodes that are far away — no opacity contribution
        // Use a generous threshold since opacity falls off at object boundaries
        if box_dist > 1.0 {
            continue;
        }
        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let opacity = evaluate_object_opacity(pos, leaf_obj_idx);
                max_opacity = max(max_opacity, opacity);
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

// ---------- Shadow via Transmittance Marching ----------

fn compute_shadow(origin: vec3<f32>, light_dir: vec3<f32>, max_dist: f32) -> f32 {
    var transmittance = 1.0;
    var t = 0.02;  // small bias to avoid self-shadowing
    let step_size = 0.06;
    for (var i = 0u; i < 48u; i++) {
        if t > max_dist || transmittance < 0.01 {
            break;
        }
        let pos = origin + light_dir * t;
        // Coarse field early-out for empty space
        let cam_rel = pos - shade_uniforms.camera_pos.xyz;
        let coarse_dist = sample_coarse_field(cam_rel);
        if coarse_dist > 0.5 {
            t += coarse_dist * 0.8;
            continue;
        }
        let opacity = sample_opacity_bvh(pos);
        transmittance *= (1.0 - opacity);
        // Adaptive stepping: small steps in high-opacity regions
        t += mix(step_size * 2.0, step_size * 0.5, opacity);
    }
    return transmittance;
}

// ---------- Opacity-Probe Ambient Occlusion ----------
//
// 5-step opacity probe along normal — view-independent, captures true geometry
// occlusion. Each probe samples the opacity field via BVH traversal, with
// coarse field early-out when the sample point is in open space.

fn compute_opacity_ao(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var weight = 1.0;
    for (var i = 1u; i <= 5u; i++) {
        let probe_dist = AO_STEP_SIZE * f32(i);
        let probe_pos = pos + normal * probe_dist;
        let opacity = sample_opacity_bvh(probe_pos);
        occlusion += weight * opacity;
        weight *= 0.5;
    }
    return clamp(1.0 - AO_STRENGTH * occlusion, 0.0, 1.0);
}

// ---------- Entry Point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let half_dims = vec2<u32>(textureDimensions(shadow_ao_out));
    if pixel.x >= half_dims.x || pixel.y >= half_dims.y {
        return;
    }

    let half_pixel = vec2<i32>(pixel.xy);
    // Read G-buffer at the corresponding full-res pixel
    let full_coord = half_pixel;
    let pos_data = textureLoad(gbuf_position, full_coord, 0);
    let hit_dist = pos_data.w;

    // Sky pixel — no shadow or AO needed
    if hit_dist >= MAX_FLOAT * 0.5 {
        textureStore(shadow_ao_out, half_pixel, vec4<f32>(1.0, 1.0, 0.0, 1.0));
        return;
    }

    let world_pos = pos_data.xyz;
    let normal = normalize(textureLoad(gbuf_normal, full_coord, 0).xyz);

    // --- Single pass: shadow + cloud shadow for first shadow-casting light ---
    var shadow = 1.0;
    var cloud_transmittance = 1.0;
    let cloud_coverage = shade_uniforms.cloud_shadow_params.y;
    let cloud_base = shade_uniforms.cloud_shadow_params.x;
    let cloud_shadow_enabled = shade_uniforms.cloud_shadow_params.z > 0.5;
    var found_shadow_caster = false;

    let num_lights_val = shade_uniforms.num_lights;
    for (var li = 0u; li < num_lights_val; li++) {
        let light = lights[li];
        if light.shadow_caster != 1u {
            continue;
        }

        var light_dir: vec3<f32>;
        var shadow_max = SHADOW_MAX_DIST;

        if light.light_type == 0u {
            // Directional
            light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));

            // Cloud shadow — project surface position along sun direction to cloud layer
            if cloud_shadow_enabled {
                var cloud_query_xz = world_pos.xz;
                let actual_y = shade_uniforms.camera_pos.y + world_pos.y;
                let dy = cloud_base - actual_y;
                if dy > 0.0 {
                    let sun_horiz = length(light_dir.xz);
                    if sun_horiz > 0.001 {
                        let sun_vert = max(light_dir.y, 0.001);
                        let offset_dist = dy * sun_horiz / sun_vert;
                        cloud_query_xz = world_pos.xz + light_dir.xz / sun_horiz * offset_dist;
                    }
                }
                let cloud_uv = cloud_query_xz / cloud_coverage + 0.5;
                if cloud_uv.x >= 0.0 && cloud_uv.x <= 1.0 && cloud_uv.y >= 0.0 && cloud_uv.y <= 1.0 {
                    cloud_transmittance = textureSampleLevel(cloud_shadow_tex, cloud_shadow_smp, cloud_uv, 0.0).r;
                }
            }
        } else {
            // Point or spot
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + shade_uniforms.camera_pos.xyz;
            let to_light = light_pos - world_pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            shadow_max = min(dist, SHADOW_MAX_DIST);
        }

        let n_dot_l = dot(normal, light_dir);
        if n_dot_l <= 0.0 {
            shadow = 1.0; // behind surface
        } else {
            let shadow_origin = world_pos + normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
            shadow = compute_shadow(shadow_origin, light_dir, shadow_max);
        }

        shadow *= cloud_transmittance;
        found_shadow_caster = true;
        break; // only first shadow caster
    }

    // --- Opacity-Probe Ambient Occlusion ---
    let ao_origin = world_pos + normal * SHADOW_BIAS;
    let ao = compute_opacity_ao(ao_origin, normal);

    // B channel = cloud transmittance (for debug visualization)
    textureStore(shadow_ao_out, half_pixel, vec4<f32>(shadow, ao, cloud_transmittance, 1.0));
}
