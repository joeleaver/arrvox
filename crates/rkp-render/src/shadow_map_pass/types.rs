//! Constants + GPU-side uniform types for the shadow map pass.
//!
//! Everything here is wire-format or capacity tuning — no logic.

use bytemuck;

/// Default shadow-map resolution. 1 K square at 4 bytes / texel
/// = 4 MB. With frustum-fit + scene clip, a 1 K map at typical
/// view bounds gives ~3 cm/texel — sharper than scene-fit 2 K
/// would. CSM is the long-term lever for far-field detail.
pub const SHADOW_MAP_DEFAULT_SIZE: u32 = 1024;

/// Cap on the distance (world units, from camera) that the
/// frustum-fit shadow camera covers. The camera's actual far
/// plane can be 10 km+; capping keeps per-meter texel density
/// high in the visible region. The proper fix (visible-caster
/// AABB fit) makes this less load-bearing — until then, 30 m
/// keeps shadow-map texels in the ~1.5 cm range for typical
/// scenes.
pub const SHADOW_FAR_DISTANCE: f32 = 30.0;

/// "Sky" depth marker. Per-pixel shadow query treats
/// `sample == FAR_DEPTH` as "no occluder" → returns full
/// transmittance.
pub const SHADOW_MAP_FAR_DEPTH: f32 = 1.0;

/// `bitcast::<u32>(1.0)` — what the clear pass writes into every
/// entry of `shadow_buffer`. atomic-min on the u32 representation
/// works because f32 in [0, 1] is monotonic in IEEE-754.
pub const SHADOW_MAP_FAR_DEPTH_BITS: u32 = 0x3F800000;

/// Initial capacity for the per-frame TLAS-prim → ScatterInstance
/// scratch arrays. Grows on demand if the prim count exceeds it.
pub const SHADOW_MAP_MAX_CASTERS_INITIAL: u32 = 2048;

/// Initial work-list capacity (one entry per 8×8 tile). 256 K
/// entries ≈ 1 MB at 4 bytes / entry. Covers ~4 instances each
/// fully covering a 2K shadow map (65 536 tiles each), or ~10 k
/// grass blades (~25 tiles each). Grows on demand.
pub const SHADOW_MAP_WORK_LIST_INITIAL: u32 = 262144;

/// Scatter pass dispatch X dimension — must match the constant in
/// `shadow_scatter.wgsl` and `shadow_scatter_finalize.wgsl`. The
/// finalize pass writes `(DISPATCH_X, ceil(total / DISPATCH_X), 1)`
/// into the indirect dispatch args.
pub const SHADOW_SCATTER_DISPATCH_X: u32 = 256;

/// Stride of a `ScatterInstance` slot — see WGSL definition. 8 ×
/// u32 = 32 bytes.
pub const SCATTER_INSTANCE_STRIDE: u64 = 32;

/// Per-frame uniform shared between the shadow-map setup +
/// scatter passes (writes the depth) and the shade-side query
/// (reads it). 160 B.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_proj_inv: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub depth_bias: f32,
    pub inv_shadow_map_size: [f32; 2],
    pub shadow_map_size: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<LightCameraUniform>() == 160);

/// Setup-pass per-frame uniform. Layout matches WGSL struct in
/// `shadow_scatter_setup.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SetupParams {
    pub prim_count: u32,
    /// Maximum distance a shadow can travel through the scene.
    /// Setup uses this to extrude per-prim AABBs along
    /// `light_dir` for the shadow-frustum cull.
    pub scene_extent: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    /// Camera view-proj matrix (world → camera NDC). The cull
    /// projects each prim's swept AABB through this and tests
    /// the resulting NDC bounds against `[-1,1]² × [0,1]`.
    pub camera_view_proj: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<SetupParams>() == 80);
