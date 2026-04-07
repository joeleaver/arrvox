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
@group(4) @binding(1) var<storage, read> octree_nodes: array<u32>;
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

const GEOM_TYPE_NONE: u32       = 0u;
const GEOM_TYPE_ANALYTICAL: u32 = 1u;
const GEOM_TYPE_VOXELIZED: u32  = 2u;
const GEOM_TYPE_PROCEDURAL: u32 = 3u;

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

// ---------- Octree Traversal ----------

const OCTREE_LEAF_BIT: u32 = 0x80000000u;

/// Traverse the octree to find the brick slot at a given octree-space position.
/// Returns the brick pool slot, or EMPTY_SLOT/INTERIOR_SLOT.
fn octree_find_slot(root: u32, max_depth: u32, extent: f32, octree_pos: vec3<f32>) -> u32 {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);

    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset];
        if node == EMPTY_SLOT { return EMPTY_SLOT; }
        if node == INTERIOR_SLOT { return INTERIOR_SLOT; }
        if (node & OCTREE_LEAF_BIT) != 0u { return node & 0x7FFFFFFFu; }

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
    if node == EMPTY_SLOT { return EMPTY_SLOT; }
    if node == INTERIOR_SLOT { return INTERIOR_SLOT; }
    if (node & OCTREE_LEAF_BIT) != 0u { return node & 0x7FFFFFFFu; }
    return EMPTY_SLOT;
}

/// Convert local-space position to octree-space position.
/// The octree is centered on the object's local origin.
fn to_octree_pos(local_pos: vec3<f32>, extent: f32) -> vec3<f32> {
    return local_pos + vec3<f32>(extent * 0.5);
}

/// Read octree parameters from GpuObject fields.
/// brick_map_offset = octree root, brick_map_dims_x = depth, brick_map_dims_y = extent bits.
fn octree_params(obj: GpuObject) -> vec3<f32> {
    // Returns (root as f32, depth as f32, extent). Cast root/depth to u32 at call site.
    return vec3<f32>(f32(obj.brick_map_offset), f32(obj.brick_map_dims_x), bitcast<f32>(obj.brick_map_dims_y));
}

/// Find brick slot and voxel index within brick for a local-space position.
/// Returns (slot, voxel_idx). slot=EMPTY_SLOT means no brick.
fn octree_voxel_at(local_pos: vec3<f32>, obj: GpuObject) -> vec2<u32> {
    let extent = bitcast<f32>(obj.brick_map_dims_y);
    let octree_pos = to_octree_pos(local_pos, extent);

    if any(octree_pos < vec3<f32>(0.0)) || any(octree_pos >= vec3<f32>(extent)) {
        return vec2<u32>(EMPTY_SLOT, 0u);
    }

    let slot = octree_find_slot(obj.brick_map_offset, obj.brick_map_dims_x, extent, octree_pos);
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return vec2<u32>(slot, 0u);
    }

    // Per-voxel octree: slot IS the voxel. No within-brick index needed.
    return vec2<u32>(slot, 0u);
}

// ---------- Opacity Field Evaluation ----------

/// Sample opacity at a local-space position using octree traversal.
fn sample_opacity_point(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let sv = octree_voxel_at(local_pos, obj);
    if sv.x == EMPTY_SLOT { return 0.0; }
    if sv.x == INTERIOR_SLOT { return 1.0; }
    return extract_opacity(brick_pool[sv.x].word0);
}

/// 8-tap trilinear interpolation of the opacity field using octree traversal.
fn sample_opacity_trilinear(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let h = vs * 0.5;

    let s000 = sample_opacity_point(local_pos + vec3(-h, -h, -h), obj);
    let s100 = sample_opacity_point(local_pos + vec3( h, -h, -h), obj);
    let s010 = sample_opacity_point(local_pos + vec3(-h,  h, -h), obj);
    let s110 = sample_opacity_point(local_pos + vec3( h,  h, -h), obj);
    let s001 = sample_opacity_point(local_pos + vec3(-h, -h,  h), obj);
    let s101 = sample_opacity_point(local_pos + vec3( h, -h,  h), obj);
    let s011 = sample_opacity_point(local_pos + vec3(-h,  h,  h), obj);
    let s111 = sample_opacity_point(local_pos + vec3( h,  h,  h), obj);

    let extent = bitcast<f32>(obj.brick_map_dims_y);
    let f = fract((local_pos + vec3(extent * 0.5)) / vs + 0.5);

    let x0 = mix(s000, s100, f.x);
    let x1 = mix(s010, s110, f.x);
    let x2 = mix(s001, s101, f.x);
    let x3 = mix(s011, s111, f.x);
    let y0 = mix(x0, x1, f.y);
    let y1 = mix(x2, x3, f.y);
    return mix(y0, y1, f.z);
}

/// Sample per-voxel paint color from the companion color pool via octree lookup.
fn sample_voxelized_color(local_pos: vec3<f32>, obj: GpuObject) -> vec4<f32> {
    let sv = octree_voxel_at(local_pos, obj);
    if sv.x == EMPTY_SLOT || sv.x == INTERIOR_SLOT {
        return vec4<f32>(0.0);
    }

    // Per-voxel octree: color is a parallel array indexed by voxel_slot directly.
    // No companion map — same index as the voxel pool.
    let packed = color_pool_data[sv.x];
    if packed == 0u {
        return vec4<f32>(0.0);
    }
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let intensity = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, intensity);
}

/// Sample per-voxel blend data via octree lookup.
fn sample_voxelized_blend(local_pos: vec3<f32>, obj: GpuObject) -> vec2<f32> {
    let sv = octree_voxel_at(local_pos, obj);
    if sv.x == EMPTY_SLOT || sv.x == INTERIOR_SLOT {
        return vec2<f32>(0.0, 0.0);
    }
    let w0 = brick_pool[sv.x].word0;
    let w1 = brick_pool[sv.x].word1;
    let secondary_mat = f32((w1 >> 16u) & 0xFFFFu);
    let blend_weight = f32((w0 >> 16u) & 0xFFu) / 255.0;
    return vec2<f32>(secondary_mat, blend_weight);
}

// SDF shader functions injected by ShaderComposer (dispatch_sdf_shader, etc.)
// SDF_SHADER_FUNCTIONS

/// Sample per-voxel material IDs and blend weight via octree lookup.
fn sample_voxelized_material_full(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let sv = octree_voxel_at(local_pos, obj);
    if sv.x == EMPTY_SLOT || sv.x == INTERIOR_SLOT {
        return vec3<f32>(f32(obj.material_id), 0.0, 0.0);
    }
    let w0 = brick_pool[sv.x].word0;
    let w1 = brick_pool[sv.x].word1;
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
    if obj.geom_type == GEOM_TYPE_NONE {
        return 0.0;
    }
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
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

// ---------- AABB Ray Intersection ----------

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

