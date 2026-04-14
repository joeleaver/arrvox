// Triangle G-buffer pass — writes deferred shading inputs from rasterized
// marching-cubes meshes. Replaces the per-pixel octree march for mesh-backed
// objects; the compute march still runs alongside for octree-only objects in
// Phase 1.
//
// G-buffer targets (matches rkf-render's GBuffer layout, sans motion —
// 48 bytes/sample across all 4 attachments exceeds the default wgpu limit
// of 32. Motion target is zero-write anyway for Phase 1, so we skip it and
// let whatever the march wrote (or cleared) pass through):
//   0: position.xyz + hit_distance   Rgba32Float  (16 B)
//   1: normal.xyz + blend_weight     Rgba16Float  ( 8 B)
//   2: material_ids + blend+object_id Rg32Uint    ( 8 B)  — total 32 B ✓
//
// Scene bind group (group 0) matches rkp_scene (Phase 4 minimal layout):
//   0: objects (storage, read) — RkpGpuObject array
//   1: camera  (uniform)       — CameraUniforms

struct CameraUniforms {
    position:   vec4<f32>,
    forward:    vec4<f32>,
    right:      vec4<f32>,
    up:         vec4<f32>,
    resolution: vec2<f32>,
    jitter:     vec2<f32>,
    prev_vp:    mat4x4<f32>,
    view_proj:  mat4x4<f32>,
}

struct RkpGpuObject {
    world:                     mat4x4<f32>,  // 0  (64 B)
    aabb_min:                  vec3<f32>,    // 64
    octree_root:               u32,          // 76
    aabb_max:                  vec3<f32>,    // 80
    octree_depth:              u32,          // 92
    octree_extent_bits:        u32,          // 96
    voxel_size:                f32,          // 100
    material_id:               u32,          // 104
    object_id:                 u32,          // 108
    geom_type:                 u32,          // 112
    is_skinned:                u32,          // 116
    bone_count:                u32,          // 120
    bone_buffer_offset:        u32,          // 124
    rest_octree_root:          u32,          // 128
    rest_octree_depth:         u32,          // 132
    rest_octree_extent_bits:   u32,          // 136
    deformed_pool_offset:      u32,          // 140
    _padding:                  array<u32,12>,// 144 (48 B)
    inverse_world:             mat4x4<f32>,  // 192 (64 B)
}

@group(0) @binding(0) var<storage, read> objects: array<RkpGpuObject>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

struct VertexIn {
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) color:         u32,
    @location(3) material_pack: u32,  // primary(lo16) | secondary(hi16)
    @location(4) blend_weight:  u32,  // 0..=255 in low byte
}

// world_pos is no longer interpolated — the fragment reconstructs it from
// depth when needed (it doesn't here; shade/SSAO/volumetric do). Saves one
// interpolator slot.
struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal:  vec3<f32>,
    @location(1) color:         vec3<f32>,
    @location(2) @interpolate(flat) material_pack:   u32,  // primary | secondary << 16
    @location(3) @interpolate(flat) blend_weight:    u32,
    @location(4) @interpolate(flat) object_id:       u32,
    @location(5) @interpolate(flat) obj_material_id: u32,
}

// ----- Vertex -----
// `instance_index` carries the gpu-objects array index the mesh belongs to.
// Each draw call issues instance_count=1, first_instance=gpu_object_idx.
@vertex
fn vs_main(v: VertexIn, @builtin(instance_index) inst: u32) -> VertexOut {
    let obj = objects[inst];
    let world_pos4 = obj.world * vec4<f32>(v.position, 1.0);
    // Normals transform by the transposed inverse of the world matrix —
    // correct even under non-uniform scale. `obj.inverse_world` is already
    // precomputed on CPU for the raymarch, we reuse it here.
    let world_n = normalize((transpose(obj.inverse_world) * vec4<f32>(v.normal, 0.0)).xyz);

    var out: VertexOut;
    out.clip_pos     = camera.view_proj * world_pos4;
    out.world_normal = world_n;
    out.color = vec3<f32>(
        f32((v.color >> 0u) & 0xFFu) / 255.0,
        f32((v.color >> 8u) & 0xFFu) / 255.0,
        f32((v.color >> 16u) & 0xFFu) / 255.0,
    );
    out.material_pack   = v.material_pack;
    out.blend_weight    = v.blend_weight & 0xFFu;
    out.object_id       = obj.object_id;
    out.obj_material_id = obj.material_id & 0xFFFFu;
    return out;
}

// ----- Fragment -----
// Writes 4 G-buffer targets. Matches the layout produced by `octree_march.wgsl`
// so downstream passes (SSAO, shade) don't care whether a pixel came from
// march or raster.

// Two render targets post-Phase-5 — position is dropped, shade/ssao/volumetric
// reconstruct world_pos from depth + camera.inverse_view_proj.
struct FragOut {
    @location(0) normal:   vec4<f32>,
    @location(1) material: vec2<u32>,
}

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    // Pack per-vertex color into RGB565 to match octree_march.wgsl's layout.
    let cr = u32(clamp(in.color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(in.color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(in.color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    // Material channel packing — identical bit layout to the old march:
    //   r: primary_id(lo16) | secondary_id(hi16)
    //   g: blend(lo8) | (object_id+1)(bits 8-15) | color_rgb565(hi16)
    //
    // If the voxel's baked primary material is 0, fall back to the object's
    // override material_id so editor AssignMaterial works without re-voxelizing.
    // `+1` on object_id: 0 means "no hit" to the picker and shade pass.
    let primary_in = in.material_pack & 0xFFFFu;
    let secondary_in = (in.material_pack >> 16u) & 0xFFFFu;
    let effective_primary = select(in.obj_material_id, primary_in, primary_in != 0u);
    let packed_r = effective_primary | (secondary_in << 16u);
    let packed_g = (in.blend_weight & 0xFFu)
                 | (((in.object_id + 1u) & 0xFFu) << 8u)
                 | (color_rgb565 << 16u);

    var out: FragOut;
    out.normal   = vec4<f32>(normalize(in.world_normal), 1.0);
    out.material = vec2<u32>(packed_r, packed_g);
    return out;
}
