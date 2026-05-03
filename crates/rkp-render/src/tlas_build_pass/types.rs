//! Wire-format types + uniform structs + size/count constants for the
//! GPU TLAS build pipeline.
//!
//! No logic — just `#[repr(C)]` data shapes that match the WGSL side.
//! Each struct is paired with a `const _: () = assert!(size_of...)` so
//! a layout drift breaks the build.

/// 48-byte scratch entry used by Phase 6's deleted tile-cull AABB
/// pass. Phase 5 cleanup retires the per-pixel emit/cull/scatter
/// pipeline that produced these; the type is kept here as the wire
/// shape for [`TlasPrim`] (and for the test-only CPU oracle path).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceTileCullEntry {
    pub aabb_min: [f32; 3],
    pub asset_id: u32,
    pub aabb_max: [f32; 3],
    pub instance_state_offset: u32,
    pub material_id: u32,
    pub live: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceTileCullEntry>() == 48);

/// One primitive in the unified TLAS-build input list. Plus an
/// `instance_index` field that distinguishes host (real
/// `RkpGpuInstance` index) from user-shader
/// ([`TLAS_LEAF_USER_SHADER`]) leaves.
///
/// `aabb_min` / `aabb_max` are tight world-space bounds:
/// * **Host** — `world × asset.local_aabb` via Arvo's transform.
/// * **User-shader** — the user shader's `inst_aabb` hook,
///   evaluated by the Phase 6 tile-cull AABB pass.
///
/// 48 bytes; vec3<f32> alignment in WGSL packs the trailing u32
/// fields into the same 16-byte slot as each vec3, so no extra
/// padding is needed beyond the explicit `_pad0` / `_pad1` that
/// keep the struct on a 16-byte boundary for the storage-buffer
/// stride invariant.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasPrim {
    pub aabb_min: [f32; 3],
    pub asset_id: u32,
    pub aabb_max: [f32; 3],
    pub instance_state_offset: u32,
    pub material_id: u32,
    pub instance_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<TlasPrim>() == 48);

/// Sentinel `instance_index` value for user-shader primitives.
/// Mirror of [`crate::tlas_pass::TLAS_LEAF_USER_SHADER`]. Kept here
/// duplicated so `tlas_build_pass.rs` doesn't need to depend on
/// `tlas_pass.rs` in V1; once Session 5 retires the CPU path we
/// can hoist the constant to a shared module.
pub const TLAS_LEAF_USER_SHADER: u32 = 0xFFFF_FFFEu32;

/// Per-dispatch uniform for the host-instance assembly pass. 16 B —
/// matches `AssembleHostUniform` in `tlas_assemble_host.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AssembleHostUniform {
    pub instance_count: u32,
    pub asset_count: u32,
    pub prims_capacity: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<AssembleHostUniform>() == 16);

/// Per-dispatch uniform for the Morton-code compute pass. 32 B —
/// matches `MortonUniform` in `tlas_morton.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MortonUniform {
    pub scene_min: [f32; 3],
    pub _pad0: u32,
    pub scene_max: [f32; 3],
    pub prim_count: u32,
}

const _: () = assert!(std::mem::size_of::<MortonUniform>() == 32);

/// Per-dispatch uniform for one radix-sort sub-pass. 16 B — matches
/// `RadixUniform` in `tlas_radix_sort.wgsl`. The engine bumps
/// `digit_shift` by 8 between the four passes (0, 8, 16, 24).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixUniform {
    pub prim_count: u32,
    pub digit_shift: u32,
    pub num_workgroups: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<RadixUniform>() == 16);

/// Workgroup size of the radix count + scatter passes. 64 threads
/// per workgroup; `num_workgroups = prim_count.div_ceil(64)`.
pub const RADIX_WG_SIZE: u32 = 64;

/// Number of radix buckets (= 1 << bits-per-digit). 8-bit digits → 256.
pub const RADIX_BUCKETS: u32 = 256;

/// Number of radix passes. 32-bit Morton ÷ 8-bit digit = 4 passes.
pub const RADIX_PASSES: u32 = 4;

/// Per-dispatch uniform for the Karras tree builder. 16 B —
/// matches `KarrasUniform` in `tlas_karras.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KarrasUniform {
    pub prim_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<KarrasUniform>() == 16);

/// Initial `tlas_prims` buffer capacity in entries. Grows on demand
/// via [`super::pass::TlasBuildPass::ensure_prims_capacity`]. Sized
/// for one entry so the buffer exists for bind-group validation
/// before the first dispatch.
pub const TLAS_PRIMS_INITIAL_ENTRIES: u32 = 1;
