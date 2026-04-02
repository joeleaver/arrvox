// Shared shading infrastructure — structs, bindings, constants, and utility functions.
//
// Opacity-field version for RKIPatch. This file is concatenated with per-model shader
// files (shade_pbr.wgsl, etc.) and shade_main.wgsl by the CPU-side ShaderComposer to
// produce the final uber-shader. All functions defined here are available to shading
// model functions.

// ---------- Material struct (must match Rust Material, 96 bytes) ----------

struct Material {
    // PBR baseline (0–15)
    albedo_r: f32,
    albedo_g: f32,
    albedo_b: f32,
    roughness: f32,
    // 16–31
    metallic: f32,
    emission_r: f32,
    emission_g: f32,
    emission_b: f32,
    // 32–35
    emission_strength: f32,
    // SSS (36–55)
    subsurface: f32,
    subsurface_r: f32,
    subsurface_g: f32,
    subsurface_b: f32,
    opacity: f32,
    ior: f32,
    // Noise (60–71)
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    // Shader selection (72–75)
    shader_id: u32,
    // Padding (76–95)
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

// ---------- v2 Scene data types (must match ray_march.wgsl) ----------

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
    is_skinned: u32,
    bone_count: u32,
    bone_buffer_offset: u32,
    rest_brick_map_offset: u32,
    rest_brick_map_dims_x: u32,
    rest_brick_map_dims_y: u32,
    rest_brick_map_dims_z: u32,
    shell_height: f32, sdf_shader_id: u32, sdf_shader_material: u32,
    deformed_pool_offset: u32, _pad10: u32, _pad11: u32, _pad12: u32,
    _pad13: u32,
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

// ---------- Light type (must match Rust Light, 64 bytes) ----------

struct Light {
    light_type: u32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    cookie_index: i32,
    shadow_caster: u32,
}

// ---------- Coarse field info ----------

struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,
    dims: vec4<u32>,
    voxel_size: f32,
    inv_voxel_size: f32,
    _cf_pad0: f32,
    _cf_pad1: f32,
}

struct RadianceVolumeUniforms {
    center:      vec4<f32>,
    voxel_sizes: vec4<f32>,
    inv_extents: vec4<f32>,
    params:      vec4<u32>,   // x = dim, y = num_levels
}

// ---------- ShadingContext — the user shader API ----------

struct ShadingContext {
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    n_dot_v: f32,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    emission_strength: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    opacity: f32,
    ior: f32,
    f0: vec3<f32>,
    reflect_dir: vec3<f32>,
    cam_dist: f32,
    jitter: f32,
    contact: f32,
    atmo_shadow_fill: f32,
    pixel: vec2<u32>,
    material_id: u32,
    sss_color: vec3<f32>,
}

// ---------- Bindings ----------

// Group 0: G-buffer read (sampled textures)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;
@group(0) @binding(3) var gbuf_motion:   texture_2d<f32>;

// Group 1: material table + shader params
@group(1) @binding(0) var<storage, read> materials: array<Material>;

struct ShaderParams {
    param0: f32, param1: f32, param2: f32, param3: f32,
    param4: f32, param5: f32, param6: f32, param7: f32,
}
@group(1) @binding(1) var<storage, read> shader_params: array<ShaderParams>;

// Group 2: HDR output
@group(2) @binding(0) var hdr_output: texture_storage_2d<rgba16float, write>;

// Group 3: Shade uniforms
struct ShadeUniforms {
    debug_mode: u32,
    num_lights: u32,
    _su_pad0: u32,
    shadow_budget_k: u32,
    camera_pos: vec4<f32>,
    // Atmosphere
    sun_dir: vec4<f32>,        // xyz = direction toward sun, w = sun_intensity
    sun_color: vec4<f32>,      // xyz = sun color (linear RGB), w = gi_intensity
    sky_params: vec4<f32>,     // x = rayleigh_scale, y = mie_scale, z = atmosphere_enabled, w = unused
    // Camera basis for sky ray reconstruction
    cam_forward: vec4<f32>,    // xyz = camera forward (unit), w = unused
    cam_right: vec4<f32>,      // xyz = camera right * tan(fov/2) * aspect, w = unused
    cam_up: vec4<f32>,         // xyz = camera up * tan(fov/2), w = unused
    shadow_params: vec4<f32>,  // x=shadow_softness, y=shadow_density, z=ambient_intensity, w=shadow_fill
    cloud_shadow_params: vec4<f32>, // x=cloud_base, y=cloud_coverage, z=cloud_shadow_enabled, w=unused
    ambient_sky: vec4<f32>,   // xyz=precomputed hemisphere-average sky irradiance, w=unused
}
@group(3) @binding(0) var<uniform> shade_uniforms: ShadeUniforms;

// Group 4: v2 Scene data (same layout as ray march group 0)
@group(4) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(4) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(4) @binding(2) var<storage, read> objects: array<GpuObject>;
// binding 3 = camera uniforms (not used here — shade_uniforms has camera_pos)
// binding 4 = scene uniforms
@group(4) @binding(4) var<uniform> v2_scene: SceneUniformsV2;
@group(4) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 5: Light buffer
@group(5) @binding(0) var<storage, read> lights: array<Light>;

// Group 6: Coarse acceleration field
@group(6) @binding(0) var coarse_field: texture_3d<f32>;
@group(6) @binding(1) var coarse_sampler: sampler;
@group(6) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// Group 7: Radiance volume (4 clipmap levels + sampler + uniforms)
@group(7) @binding(0) var radiance_L0: texture_3d<f32>;
@group(7) @binding(1) var radiance_L1: texture_3d<f32>;
@group(7) @binding(2) var radiance_L2: texture_3d<f32>;
@group(7) @binding(3) var radiance_L3: texture_3d<f32>;
@group(7) @binding(4) var radiance_sampler: sampler;
@group(7) @binding(5) var<uniform> radiance_vol: RadianceVolumeUniforms;

// Group 3 continued: Brush overlay — geodesic distance for cursor visualization
@group(3) @binding(1) var<storage, read> brush_overlay_data: array<f32>;
@group(3) @binding(2) var<storage, read> brush_overlay_map: array<u32>;

struct BrushOverlayUniforms {
    brush_radius: f32,
    brush_falloff: f32,
    brush_object_id: u32,
    brush_active: u32,
    brush_color: vec4<f32>,
    brush_center_local: vec4<f32>,
}
@group(3) @binding(3) var<uniform> brush_overlay: BrushOverlayUniforms;

// Group 3 continued: Color companion pool for per-voxel paint
@group(3) @binding(4) var<storage, read> color_pool_data: array<u32>;
@group(3) @binding(5) var<storage, read> color_companion_map: array<u32>;

// Group 3 continued: Half-res precomputed shadow + SSAO
@group(3) @binding(6) var shadow_ao_tex: texture_2d<f32>;

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

// Light types
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// Ambient/sky
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.03, 0.035, 0.05);
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.12, 0.18, 0.45);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.95, 0.6, 0.3);
const SKY_REFLECT_STRENGTH: f32 = 0.15;

// Shadow parameters
const MAX_SHADOW_STEPS: u32 = 16u;
const SHADOW_EPSILON: f32 = 0.005;
// SHADOW_K is now shade_uniforms.shadow_params.x
const SHADOW_MAX_DIST: f32 = 12.0;
const SHADOW_BIAS: f32 = 0.08;
// SHADOW_ATMO_DENSITY is now shade_uniforms.shadow_params.w
const SHADOW_ATMO_MAX_FILL: f32 = 0.65;

// AO parameters
const AO_STEP_SIZE: f32 = 0.12;
const AO_STRENGTH: f32 = 0.7;

// SSS parameters
const SSS_MAX_THICKNESS: f32 = 0.3;
const SSS_SIGMA: f32 = 8.0;
const SSS_WRAP: f32 = 0.3;

// Coarse field threshold for switching to per-object evaluation
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

// GI cone tracing parameters
const GI_CONE_STEPS: u32 = 16u;
const GI_MAX_STEP: f32 = 0.16;
// GI intensity: shade_uniforms.sun_color.w
const GI_DIFFUSE_MAX_DIST: f32 = 5.0;
const GI_SPECULAR_MAX_DIST: f32 = 8.0;

// Noise channel constants
const NOISE_CHANNEL_ALBEDO: u32 = 1u;
const NOISE_CHANNEL_ROUGHNESS: u32 = 2u;
const NOISE_CHANNEL_NORMAL: u32 = 4u;

// Opacity field constants
const OPACITY_SURFACE_THRESHOLD: f32 = 0.5;
const OPACITY_TRANSMITTANCE_STEP: f32 = 0.03;

// ---------- VoxelSample Helpers ----------

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}

// ---------- SDF Primitives (stubs — not used for opacity objects, kept for compilation) ----------

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

// ---------- Opacity Field Evaluation ----------

/// No analytical splat objects — return 0.0 (empty).
fn evaluate_analytical(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    return 0.0;
}

/// Sample a single voxel's opacity from the brick pool.
/// EMPTY_SLOT → 0.0 (empty space), INTERIOR_SLOT → 1.0 (fully opaque interior).
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

/// 8-tap trilinear interpolation of the opacity field within an object.
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

/// Compatibility alias: shade_main.wgsl glass path calls `sample_voxelized()` to detect
/// when a ray exits an object. Returns a pseudo-SDF: negative inside (opacity > 0.5),
/// positive outside. The magnitude is scaled by voxel_size so the glass exit check
/// `probe_sdf > vs` still works.
fn sample_voxelized(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let opacity = sample_opacity_trilinear(local_pos, obj);
    return (0.5 - opacity) * obj.voxel_size * 2.0;
}

/// Sample per-voxel paint color from the companion color pool.
/// Returns vec4(r, g, b, intensity) where intensity is the paint alpha (0 = no paint).
/// If no color brick exists for this location, returns vec4(0.0) (no paint).
fn sample_voxelized_color(local_pos: vec3<f32>, obj: GpuObject) -> vec4<f32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return vec4<f32>(0.0);
    }

    let voxel_coord = grid_pos / vs;
    let vc = clamp(vec3<i32>(floor(voxel_coord)), vec3<i32>(0), vec3<i32>(dims) * 8 - vec3<i32>(1));
    let brick = vec3<u32>(vc / vec3<i32>(8));
    let local = vec3<u32>(vc % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return vec4<f32>(0.0);
    }

    // Look up companion map: brick_slot → color_slot
    let color_slot = color_companion_map[slot];
    if color_slot == EMPTY_SLOT {
        return vec4<f32>(0.0);  // no color brick for this slot
    }

    let voxel_idx = local.x + local.y * 8u + local.z * 64u;
    let packed = color_pool_data[color_slot * 512u + voxel_idx];
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let intensity = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, intensity);
}

/// Sample per-voxel blend data from a voxelized object at a local-space position.
/// Returns vec2(secondary_material_id, blend_weight_0to1).
/// secondary_material_id is from word1 bits 16-31, blend_weight from word0 bits 16-23.
fn sample_voxelized_blend(local_pos: vec3<f32>, obj: GpuObject) -> vec2<f32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return vec2<f32>(0.0, 0.0);
    }

    let voxel_coord = grid_pos / vs;
    let vc = clamp(vec3<i32>(floor(voxel_coord)), vec3<i32>(0), vec3<i32>(dims) * 8 - vec3<i32>(1));
    let brick = vec3<u32>(vc / vec3<i32>(8));
    let local = vec3<u32>(vc % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return vec2<f32>(0.0, 0.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    let w0 = brick_pool[idx].word0;
    let w1 = brick_pool[idx].word1;
    let secondary_mat = f32((w1 >> 16u) & 0xFFFFu);
    let blend_weight = f32((w0 >> 16u) & 0xFFu) / 255.0;
    return vec2<f32>(secondary_mat, blend_weight);
}

// SDF shader functions injected by ShaderComposer (dispatch_sdf_shader, etc.)
// SDF_SHADER_FUNCTIONS

/// Sample per-voxel material IDs and blend weight (primary, secondary, blend_weight).
/// Returns vec3(primary_mat_id, secondary_mat_id, blend_weight).
fn sample_voxelized_material_full(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return vec3<f32>(f32(obj.material_id), 0.0, 0.0);
    }

    let voxel_coord = grid_pos / vs;
    let vc = clamp(vec3<i32>(floor(voxel_coord)), vec3<i32>(0), vec3<i32>(dims) * 8 - vec3<i32>(1));
    let brick = vec3<u32>(vc / vec3<i32>(8));
    let local = vec3<u32>(vc % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return vec3<f32>(f32(obj.material_id), 0.0, 0.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    let w0 = brick_pool[idx].word0;
    let w1 = brick_pool[idx].word1;
    let primary_mat = f32(w1 & 0xFFFFu);
    let secondary_mat = f32((w1 >> 16u) & 0xFFFFu);
    let blend_weight = f32((w0 >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(primary_mat, secondary_mat, blend_weight);
}

// ---------- Opacity Field Object Evaluation ----------

/// Evaluate opacity for a single object at a world-space position.
/// Returns opacity in [0, 1]. For analytical objects: returns 0.0.
/// No scale multiplication — opacity is dimensionless.
fn evaluate_object_opacity(world_pos: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    if obj.sdf_type == SDF_TYPE_NONE {
        return 0.0;
    }
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        return 0.0;
    }
    return sample_opacity_trilinear(local_pos, obj);
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

/// Sample the maximum opacity at a world-space position using coarse field + BVH.
/// Returns the maximum opacity across all objects at this position.
///
/// Coarse field uses camera-relative coordinates (centered on camera).
/// Object evaluation uses world-space (inverse_world is world-space).
/// BVH traversal uses world-space positions against world-space AABBs.
fn sample_opacity_bvh(pos: vec3<f32>) -> f32 {
    // Coarse field is camera-relative (centered on camera).
    let cam_rel = pos - shade_uniforms.camera_pos.xyz;
    let coarse_dist = sample_coarse_field(cam_rel);
    if coarse_dist > COARSE_NEAR_THRESHOLD {
        return 0.0;
    }

    // BVH traversal for precise per-object opacity (world-space).
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
        // Skip nodes whose AABB is far away — no opacity contribution possible.
        if box_dist > 0.0 {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let o = evaluate_object_opacity(pos, leaf_obj_idx);
                max_opacity = max(max_opacity, o);
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

// ---------- Compatibility Shims ----------
//
// shade_main.wgsl and shade_common_shading.wgsl call sample_sdf / sample_sdf_shadow /
// sample_sdf_shadow_conf. These shims convert opacity to a pseudo-SDF so downstream
// code works unchanged. The pseudo-SDF is negative inside (opacity > 0.5) and positive
// outside (opacity < 0.5), matching SDF sign conventions.

/// Pseudo-SDF from opacity: negative inside, positive outside.
/// sample_sdf(pos) → -(opacity - 0.5)
fn sample_sdf(pos: vec3<f32>) -> f32 {
    let opacity = sample_opacity_bvh(pos);
    return -(opacity - OPACITY_SURFACE_THRESHOLD);
}

/// Shadow pseudo-SDF: same as sample_sdf for opacity fields.
fn sample_sdf_shadow(pos: vec3<f32>) -> f32 {
    let opacity = sample_opacity_bvh(pos);
    return -(opacity - OPACITY_SURFACE_THRESHOLD);
}

/// Shadow pseudo-SDF with confidence: returns vec2(pseudo_sdf, confidence).
/// Confidence is 1.0 where opacity > 0.01 (some geometry present), 0.0 in empty space.
fn sample_sdf_shadow_conf(pos: vec3<f32>) -> vec2<f32> {
    let opacity = sample_opacity_bvh(pos);
    let pseudo_sdf = -(opacity - OPACITY_SURFACE_THRESHOLD);
    let confidence = select(0.0, 1.0, opacity > 0.01);
    return vec2<f32>(pseudo_sdf, confidence);
}

/// Evaluate a single object at a world-space position. Returns world-space distance (pseudo-SDF).
/// Provided for compatibility with glass refraction and other code that calls evaluate_object_dist.
fn evaluate_object_dist(world_pos: vec3<f32>, obj_idx: u32) -> f32 {
    let opacity = evaluate_object_opacity(world_pos, obj_idx);
    let obj = objects[obj_idx];
    let min_scale = min(obj.accumulated_scale_x, min(obj.accumulated_scale_y, obj.accumulated_scale_z));
    // Convert opacity to a pseudo-distance scaled by voxel size for correct step sizing.
    return -(opacity - OPACITY_SURFACE_THRESHOLD) * obj.voxel_size * min_scale;
}

/// Shadow variant of evaluate_object_dist returning vec2(distance, confidence).
fn evaluate_object_dist_shadow(world_pos: vec3<f32>, obj_idx: u32) -> vec2<f32> {
    let opacity = evaluate_object_opacity(world_pos, obj_idx);
    let obj = objects[obj_idx];
    let min_scale = min(obj.accumulated_scale_x, min(obj.accumulated_scale_y, obj.accumulated_scale_z));
    let pseudo_dist = -(opacity - OPACITY_SURFACE_THRESHOLD) * obj.voxel_size * min_scale;
    let confidence = select(0.0, 1.0, opacity > 0.01);
    return vec2<f32>(pseudo_dist, confidence);
}
