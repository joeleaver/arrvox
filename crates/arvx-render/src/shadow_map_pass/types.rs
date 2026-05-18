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

/// Number of CSM cascades. 4 is the industry default and fits within
/// wgpu's 8-storage-per-stage limit at the shade pass with the
/// consolidated `shadow_buffer` (1 storage binding regardless of
/// cascade count).
///
/// All `CSM_CASCADE_COUNT` slices are rendered each frame by the
/// mesh shadow path.
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
/// `mesh_shadow_map_pass` (writes per-cascade depth) and the shade
/// pass (per-pixel cascade selection + shadow sample).
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

