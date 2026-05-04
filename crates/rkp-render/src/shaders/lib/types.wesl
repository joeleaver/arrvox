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
