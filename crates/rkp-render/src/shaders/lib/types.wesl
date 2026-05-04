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
