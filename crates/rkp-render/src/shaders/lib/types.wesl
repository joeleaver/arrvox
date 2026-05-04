// Shared GPU type definitions.
//
// IMPORTANT — splice-chunk types stay LOCAL.
// `HostSample`, `UserCtx`, `VoxelEmit`, `ShadeCtx`, `ShadeResult`, `Aabb`,
// `RkpAsset`, `Material` are all referenced by `shader_composer/compose.rs`
// in spliced text chunks. WESL mangles imported symbols at emit time, but
// the splice chunk arrives as raw text using unmangled names; resolving
// the mismatch requires either Phase 4 (WESL-native composition) or
// emitting mangled names from compose.rs. Until then, those types must
// be declared in each splice-consumer file.
//
// Grep `crates/rkp-render/src/shader_composer/compose.rs` before adding
// any type to this module. If the type name appears there, keep it local.

// Result of an octree descent: the terminating slot, the depth at which
// it terminated, and the spatial bounds of the terminating cell in
// object-local oc-space. For BRICK results the cell bounds are the
// brick's bounds; the brick DDA loop uses them to compute local cell
// coords without re-descending.
struct OctreeResult {
    slot: u32,
    depth: u32,
    cell_center: vec3<f32>,
    cell_half: f32,
}

// Per-leaf attributes (8 bytes), one entry in the per-asset pool. The
// march samples one of these on every leaf hit to drive shading.
//
// `normal_oct`         2× snorm16 octahedral surface normal.
// `material_packed`    bits  0..16: material_primary (u16)
//                      bits 16..28: material_secondary (u12, shifted 16)
//                      bits 28..32: blend_weight     (u4,  shifted 28)
struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

fn leaf_attr_material_primary(a: LeafAttr) -> u32 { return a.material_packed & 0xFFFFu; }

// Per-instance band-cell record (16 bytes), one entry per painted-band
// cell on a host instance. Reinterprets the leaf-attr slot when the
// band sentinel bit is set on the slot — `anchor_world_pos` is the
// world-space anchor for the painted material, `material_id` selects
// the host material override.
struct BandCell {
    anchor_world_pos: vec3<f32>,
    material_id: u32,
}

// One per scene instance. CPU-side mirror is `rkp_scene::GpuInstance`;
// keep field order aligned with that struct (the layout is locked by
// `bytemuck` casts on the Rust side).
//
// `overlay_offset`/`overlay_count` slice into the `instance_overlay`
// binding for per-instance paint overlays — `overlay_count == 0` ⇒
// unpainted, `fetch_leaf_attr` falls through to the asset's
// `leaf_attr_pool[slot]`.
struct RkpInstance {
    world: mat4x4<f32>,
    asset_id: u32,
    material_id: u32,
    object_id: u32,
    layer_mask: u32,
    is_skinned: u32,
    bone_buffer_offset: u32,
    bone_field_offset: u32,
    bone_field_occ_offset: u32,
    bone_field_dim_x: u32, bone_field_dim_y: u32,
    bone_field_dim_z: u32,
    bone_field_origin_x: f32,
    bone_field_origin_y: f32,
    bone_field_origin_z: f32,
    overlay_offset: u32,
    overlay_count: u32,
}

// GPU material record (96 bytes). CPU-side mirror is
// `rkp_scene::GpuMaterial`; field order locked by `bytemuck` cast.
//
// NOTE: rkp_shade.wesl and rkp_glass.wesl declare a *separate* `Material`
// struct (similar shape, different padding tail, currently diverged
// from each other on `instance_shader_id`). Unifying `Material` ↔
// `GpuMaterial` is its own refactor — out of scope for this module.
struct GpuMaterial {
    albedo_r: f32, albedo_g: f32, albedo_b: f32,
    roughness: f32,
    metallic: f32,
    emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32,
    subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    shader_id: u32,
    instance_shader_id: u32,
    _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32,
}

fn mat_albedo(m: GpuMaterial) -> vec3<f32> {
    return vec3<f32>(m.albedo_r, m.albedo_g, m.albedo_b);
}

// Per-frame camera uniforms (192 bytes). CPU-side mirror is
// `rkp_scene::CameraUniforms`; field order locked by `bytemuck` cast.
//
// `prev_vp` is the previous frame's view-projection matrix used by
// motion-vector and TAA passes; `jitter` is the current-frame TAA
// sub-pixel offset in NDC.
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

// TLAS primitive (32 bytes) — one per scene instance, fed into Karras
// LBVH construction. CPU-side mirror is `rkp_render::tlas`.
struct TlasPrim {
    aabb_min: vec3<f32>,
    asset_id: u32,
    aabb_max: vec3<f32>,
    instance_state_offset: u32,
    material_id: u32,
    instance_index: u32,
    _pad0: u32,
    _pad1: u32,
}

// TLAS BVH node (32 bytes). Output of Karras construction; consumed
// by rkp_shadow_trace's BVH descent.
//
// `left_or_leaf` high bit (0x80000000) is the leaf marker — when set,
// the low bits index into the TlasInstanceLeaf array; when clear, they
// index the next interior node.
struct TlasNode {
    aabb_min: vec3<f32>,
    left_or_leaf: u32,
    aabb_max: vec3<f32>,
    right_or_count: u32,
}

// One per TLAS leaf — what the BVH descent yields when it hits a leaf.
// `instance_state_offset` is kept for TLAS wire-format compatibility;
// `instance_index` is the index into `instances[]` for hosts, or the
// `TLAS_LEAF_USER_SHADER` sentinel for user-shader proto leaves.
struct TlasInstanceLeaf {
    asset_id: u32,
    instance_state_offset: u32,
    material_id: u32,
    instance_index: u32,
}

// One per-leaf paint-overlay record (16 bytes). Packed into the
// per-instance overlay slice (`RkpInstance.overlay_offset` /
// `overlay_count`); referenced by `fetch_leaf_attr` and the geom
// pass when re-emitting overlay-modified leaves.
struct OverlayEntry {
    leaf_slot: u32,
    normal_oct: u32,
    material_packed: u32,
    color_packed: u32,
}

// Per-instance shadow-scatter descent metadata. Setup pass writes
// these; emit pass reads them to fill the global work-list, scatter
// pass reads them again to descend the octree at each tile.
//
// `(tx0, ty0)` is the upper-left texel of the instance's projected
// rect in shadow-map space; `tile_w/tile_h` are 8×8 tile counts.
// `work_offset` is the start of this instance's tile entries in
// the global `work_list[]`.
struct ScatterInstance {
    tx0: u32, ty0: u32,
    tile_w: u32, tile_h: u32,
    asset_id: u32,
    instance_state_offset: u32,
    instance_index: u32,
    work_offset: u32,
}

// User-shader Option B prototype-march hit record. Returned by the
// per-pixel instance march and consumed by the composite pass.
//
// `local_t`         distance in oc-space along the safe ray direction
//                   from `oc_origin` (the instance-local ray origin).
// `oc_pos`          hit position in oc-space.
// `local_normal`    octahedral-decoded normal from the prototype's
//                   leaf_attr; transformed to world by the composite.
// `leaf_slot`       prototype's leaf-attr slot, used for material
//                   inheritance and overlay lookup.
struct ProtoHit {
    valid: bool,
    local_t: f32,
    oc_pos: vec3<f32>,
    local_normal: vec3<f32>,
    leaf_slot: u32,
}

// Per-frame uniform shared by octree_march and rkp_shadow_trace —
// the two passes bind the same buffer, so the layout has to match.
// Some fields are read by only one of them (notes inline); naga
// DCE drops the unused fields from the SPIR-V on each side.
struct MarchParams {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    lod_enabled: u32,
    surfacenet_enabled: u32,
    tile_count_x: u32,
    tlas_node_count: u32,
    shadow_map_enabled: u32,
    time: f32,
    asset_count: u32,
    _pad0: u32,
}

// Light-space camera + shadow-map metadata for shadow scatter.
// CPU-side mirror in `rkp_render::shadow_map`; layout locked.
struct LightCameraShadow {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    light_dir: vec3<f32>,
    depth_bias: f32,
    inv_shadow_map_size: vec2<f32>,
    shadow_map_size: vec2<u32>,
}

// Procedural-eval volumetric clouds parameters. Six tightly-packed
// vec4s bundling the noise / wind / quality / flags state, sized
// to fit a small uniform binding. cloud_march reads everything;
// cloud_sun_atten only reads `altitude`/`noise`/`wind`/`flags`/
// `coverage` but binds the same uniform.
struct CloudParams {
    altitude: vec4<f32>,   // x=cloud_min, y=cloud_max, z=threshold, w=density_scale
    noise:    vec4<f32>,   // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    wind:     vec4<f32>,   // x=wind_dir.x, y=wind_dir.y, z=wind_speed, w=time
    flags:    vec4<f32>,   // x=enable, y=coverage
    quality:  vec4<f32>,   // x=slab_steps, y=shadow_steps, z=detail_octaves, w=ms_octaves
    quality2: vec4<f32>,   // x=taa_alpha
}

// One procedural-eval bytecode instruction (96 bytes). CPU writer is
// `rkp_procedural::compile_tree`; consumers are proc_eval (the
// interpreter) and proc_ghost (the gizmo / picking helper).
//
// `distance_scale` is the local-to-world scale baked at compile
// time so primitives can return scale-invariant local distances —
// see compile_tree docs for the math.
// Light record (64 bytes). CPU-side mirror is `rkp_shade::GpuLight`.
//
// `position.w`     light type — 0 = directional, 1 = point, 2 = spot.
//                  For directional lights, `position.xyz` is the
//                  light direction; for point/spot it's world position.
// `color.rgb`      linear color; `color.w` is intensity (lumens for
//                  point, lux for directional).
// `direction.xyz`  spot direction; `direction.w` is spot outer half-angle.
// `params`         x = range, y = inner_half_angle, z = shadow_softness,
//                  w = cast_shadow flag (0 / 1).
// Volumetric march parameters (224 bytes). CPU mirror is
// `rkp_volumetric::VolumetricParams`. Bound by both the cloud march
// (which reads `prev_view_proj` for reprojection) and the fog march
// (which doesn't); naga DCE drops the unused tail from fog's SPIR-V.
struct VolumetricParams {
    cam_pos: vec4<f32>,
    cam_forward: vec4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    width: u32,
    height: u32,
    full_width: u32,
    full_height: u32,
    max_steps: u32,
    step_size: f32,
    near: f32,
    far: f32,
    fog_color: vec4<f32>,
    fog_height: vec4<f32>,
    frame_index: u32,
    vol_ambient_r: f32,
    vol_ambient_g: f32,
    vol_ambient_b: f32,
    prev_view_proj: mat4x4<f32>,
}

// Shading-pass parameters (144 bytes). CPU mirror is
// `rkp_shade::ShadeParams`. Bound by both the shade pass (which
// uses the brush, time, and shadow_map_enabled fields for paint
// preview, user-shader timing, and shadow-map gating) and the
// glass pass (which only reads the lighting prefix). naga DCE
// drops the unused tail from glass's SPIR-V.
struct ShadeParams {
    num_lights: u32,
    ambient_intensity: f32,
    camera_altitude: f32,
    sun_intensity: f32,
    sky_color_top: vec3<f32>,
    _pad0: f32,
    sky_color_horizon: vec3<f32>,
    _pad1: f32,
    sun_dir: vec3<f32>,
    _pad2: f32,
    ambient_color: vec3<f32>,
    isolation: u32,
    brush_active: u32,
    brush_radius: f32,
    brush_falloff: f32,
    time: f32,
    brush_center: vec4<f32>,
    brush_color: vec4<f32>,
    shadow_map_enabled: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

struct GpuLight {
    position: vec4<f32>,
    color: vec4<f32>,
    direction: vec4<f32>,
    params: vec4<f32>,
}

struct ProcInstruction {
    op: u32,
    arity: u32,
    material_combine: u32,
    material_id: u32,
    node_id: u32,
    distance_scale: f32,
    _pad1: u32,
    _pad2: u32,
    params_lo: vec4<f32>,
    params_hi: vec4<f32>,
    color: vec4<f32>,
    inverse_world: mat4x4<f32>,
}
