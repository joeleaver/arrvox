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

/// Number of CSM cascades. 4 is the industry default and fits within
/// wgpu's 8-storage-per-stage limit at the shade pass with the
/// consolidated `shadow_buffer` (1 storage binding regardless of
/// cascade count).
///
/// In mesh-mode we render all `CSM_CASCADE_COUNT` slices. In
/// march-mode the scatter chain still writes only into slice 0
/// and the shade pass sees `LightCameraCsm.cascade_count = 1`,
/// preserving today's single-cascade behavior bit-for-bit.
pub const CSM_CASCADE_COUNT: u32 = 4;

/// Per-cascade slice. Same wire layout as the original
/// `LightCameraUniform` (the type is kept under the old name so
/// existing shaders + setup-pass code that read a single slice
/// don't have to be churned). The CSM uniform is an array of
/// these plus a few global fields.
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

/// Wire format for the CSM uniform shared between
/// `mesh_shadow_map_pass` (writes per-cascade depth), the scatter
/// chain (writes slice 0 in march-mode), and the shade pass (per-
/// pixel cascade selection + shadow sample).
///
/// Layout (672 B, std140-safe — `cascades` is `array<LightCameraSlice, 4>`
/// where each slice is 160 B and starts at a 16 B-aligned offset;
/// the trailing `vec4` + `u32 + 12 B pad` block satisfies std140 too):
///
///   offset 0    cascades[0] (160 B)
///   offset 160  cascades[1] (160 B)
///   offset 320  cascades[2] (160 B)
///   offset 480  cascades[3] (160 B)
///   offset 640  cascade_far_view_z[4]  (16 B)
///   offset 656  cascade_count           (4 B)
///   offset 660  _pad[3]                 (12 B)
///
/// `cascade_far_view_z[i]` is the *view-space* far Z of cascade `i`
/// (positive into the scene). The shade pass picks the smallest `i`
/// such that the fragment's view_z <= `cascade_far_view_z[i]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCameraCsm {
    pub cascades: [LightCameraUniform; CSM_CASCADE_COUNT as usize],
    pub cascade_far_view_z: [f32; CSM_CASCADE_COUNT as usize],
    pub cascade_count: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<LightCameraCsm>() == 672);
const _: () = assert!(std::mem::offset_of!(LightCameraCsm, cascades) == 0);
const _: () = assert!(std::mem::offset_of!(LightCameraCsm, cascade_far_view_z) == 640);
const _: () = assert!(std::mem::offset_of!(LightCameraCsm, cascade_count) == 656);

impl LightCameraCsm {
    /// Build a CSM uniform that degenerates to a single-cascade
    /// shadow map. Used by the march-mode scatter path so the shade
    /// pass's per-pixel cascade selection picks slice 0 and behaves
    /// identically to today's single-shadow-map flow.
    ///
    /// `cascade_far_view_z[0] = +INF` so any view-space depth
    /// selects cascade 0; trailing slots are zeroed.
    pub fn single_cascade(cam: LightCameraUniform) -> Self {
        let zero_slice = LightCameraUniform {
            view_proj: [[0.0; 4]; 4],
            view_proj_inv: [[0.0; 4]; 4],
            light_dir: [0.0; 3],
            depth_bias: 0.0,
            inv_shadow_map_size: [0.0; 2],
            shadow_map_size: [0; 2],
        };
        let mut cascades = [zero_slice; CSM_CASCADE_COUNT as usize];
        cascades[0] = cam;
        let mut cascade_far_view_z = [0.0_f32; CSM_CASCADE_COUNT as usize];
        cascade_far_view_z[0] = f32::INFINITY;
        Self {
            cascades,
            cascade_far_view_z,
            cascade_count: 1,
            _pad: [0; 3],
        }
    }
}

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
