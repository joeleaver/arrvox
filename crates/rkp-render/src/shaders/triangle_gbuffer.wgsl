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
// Scene bind group (group 0) matches rkp_scene:
//   2: objects (storage, read) — RkpGpuObject array
//   3: camera  (uniform)       — CameraUniforms

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

@group(0) @binding(2) var<storage, read> objects: array<RkpGpuObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;

struct VertexIn {
    @location(0) position:    vec3<f32>,
    @location(1) normal:      vec3<f32>,
    @location(2) color:       u32,
    @location(3) material_id: u32,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:     vec3<f32>,
    @location(1) world_normal:  vec3<f32>,
    @location(2) color:         vec3<f32>,
    @location(3) @interpolate(flat) material_id:     u32,
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
    // Upper-left 3x3 for normal transform — uniform scale is assumed for
    // Phase 1 so this is correct. Non-uniform scale will need the transposed
    // inverse (we already store `inverse_world` on the object; Phase 2 tweak).
    let world_n = normalize((obj.world * vec4<f32>(v.normal, 0.0)).xyz);

    var out: VertexOut;
    out.clip_pos     = camera.view_proj * world_pos4;
    out.world_pos    = world_pos4.xyz;
    out.world_normal = world_n;
    out.color = vec3<f32>(
        f32((v.color >> 0u) & 0xFFu) / 255.0,
        f32((v.color >> 8u) & 0xFFu) / 255.0,
        f32((v.color >> 16u) & 0xFFu) / 255.0,
    );
    out.material_id     = v.material_id & 0xFFFFu;
    out.object_id       = obj.object_id;
    out.obj_material_id = obj.material_id & 0xFFFFu;
    return out;
}

// ----- Fragment -----
// Writes 4 G-buffer targets. Matches the layout produced by `octree_march.wgsl`
// so downstream passes (SSAO, shade) don't care whether a pixel came from
// march or raster.

struct FragOut {
    @location(0) position: vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) material: vec2<u32>,
}

@fragment
fn fs_main(in: VertexOut) -> FragOut {
    let cam_to_p = in.world_pos - camera.position.xyz;
    let hit_dist = length(cam_to_p);

    // Pack per-vertex color into RGB565 to match octree_march.wgsl's layout.
    let cr = u32(clamp(in.color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(in.color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(in.color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    // Material channel packing — identical bit layout to octree_march.wgsl:
    //   r: primary_id(lo16) | secondary_id(hi16)
    //   g: blend(lo8) | (object_id+1)(bits 8-15) | color_rgb565(hi16)
    //
    // If the voxel's baked material is 0, fall back to the object's override
    // material_id — this matches the compute march's behavior and makes the
    // editor's AssignMaterial command work for mesh-backed objects without
    // re-voxelizing.
    //
    // Note the `+ 1` on object_id — 0 means "no hit" in the picker / shade
    // pass, so objects are offset by one.
    let effective_mat_id = select(in.obj_material_id, in.material_id, in.material_id != 0u);
    let packed_r = effective_mat_id & 0xFFFFu; // no secondary in Phase 3 simple
    let packed_g = 0u                                         // blend = 0
                 | (((in.object_id + 1u) & 0xFFu) << 8u)      // object_id+1
                 | (color_rgb565 << 16u);                     // RGB565 color

    var out: FragOut;
    out.position = vec4<f32>(in.world_pos, hit_dist);
    // .w carries accum_alpha in the march's format — use 1.0 for fully
    // opaque mesh pixels.
    out.normal   = vec4<f32>(normalize(in.world_normal), 1.0);
    out.material = vec2<u32>(packed_r, packed_g);
    return out;
}
