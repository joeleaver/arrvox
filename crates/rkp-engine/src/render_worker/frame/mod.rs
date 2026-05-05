//! Per-frame orchestration. `render_one_frame` runs the full
//! encode/submit/readback cycle for one snapshot, split into three phases:
//!
//! - [`pre::run_pre_frame`] — phases 0–2: per-frame uploads (materials,
//!   lights, env, user-shader registry), geometry/instance/TLAS uploads,
//!   shadow-map prep, skin scatter.
//! - [`encode::encode_viewports`] — phase 3: per-VR encoder + dispatch
//!   chain + pick capture + composite readback.
//! - [`post::finalize_frame`] — phases 4–7: cloud-sun-atten readback,
//!   wgpu async poll, pick wiring, composite-readback drain + pixel ship.
//!
//! [`PreFrameOutput`] threads cross-phase state from `pre` → `encode`;
//! [`EncodeOutput`] threads `encode` → `post`. [`RenderOutcome`] is the
//! per-frame result the loop returns to sim.

use crate::render_frame::{PendingPick, RenderFrame};

use super::state::{FrameCallback, RenderState};

mod encode;
mod post;
mod pre;

/// Per-frame result handed back to the render loop. The loop ships
/// the contained values into the next `RenderResult` for sim.
pub(super) struct RenderOutcome {
    /// Latest cloud-sun attenuation read from MAIN's volumetric
    /// pass (NaN if MAIN isn't visible).
    pub(super) cloud_sun_atten_raw: f32,
    /// Wall-clock ms since the previous iteration that successfully
    /// shipped pixels to the editor. `None` when this iteration did
    /// not ship (skipped via `ship_pixels` gate) — sim uses `None`
    /// to hold the previous delivered-FPS EMA sample unchanged.
    pub(super) delivered_dt_ms: Option<f32>,
}

/// Cross-phase state from `run_pre_frame` consumed by `encode_viewports`.
pub(super) struct PreFrameOutput {
    /// User-shader transient region instance indices in the combined
    /// objects buffer (= persistent_count..persistent_count+transient_count).
    /// Phase 3 splices these into every viewport's per-tile object lists.
    pub(super) transient_indices: Vec<u32>,
    /// Total instance count after splicing transients onto the persistent
    /// (sim-supplied) instance list.
    pub(super) object_count: u32,
    /// Total asset count after splicing instance prototypes + transient
    /// region assets onto the persistent (sim-supplied) asset list.
    pub(super) asset_count: u32,
    /// CPU-derived scene AABB used by Phase 3 for the shadow-frustum
    /// cull extent and re-derived by `prepare_shadow_maps`.
    pub(super) scene_aabb: ([f32; 3], [f32; 3]),
    /// Whether this frame's directional shadow map will dispatch.
    /// Phase 3's per-VR `shade_params.shadow_map_enabled` mirrors this
    /// (gated additionally on InSitu mode).
    pub(super) shadow_map_enabled: bool,
    /// Upper-bound count for user-shader emitted instances:
    /// `painted_leaves × MAX_EMITS_PER_LEAF`. Used to size the tile-bin
    /// dispatch; the GPU shader threshold-checks against the actual
    /// count written to `user_shader_instance_count_buffer` by the
    /// emit pass.
    pub(super) user_shader_instance_count: u32,
}

/// Cross-phase state from `encode_viewports` consumed by `finalize_frame`.
pub(super) struct EncodeOutput {
    /// `true` when at least one viewport encoded a pick read this frame.
    /// `finalize_frame` issues the `map_async` only on `true`.
    pub(super) pick_issued: bool,
    /// The pending pick this frame's encode targeted (filtered to drop
    /// freshly-arrived requests when a previous pick is still in flight).
    pub(super) active_pending_pick: Option<PendingPick>,
}

/// Render a single snapshot. See module docs for the phase split.
///
/// `frame` is the canonical sim snapshot (lights, environment,
/// cameras, proc raymarch state, etc.). `gpu_instances` is the
/// possibly-interpolated object list to upload — at α=1 or when
/// there's no prev snapshot, it's `frame.gpu_objects.clone()`;
/// otherwise it's the TRS-blended version from
/// `interpolate_gpu_objects`.
///
/// `new_snapshot_consumed` is true on the iteration that just took a
/// fresh snapshot from the inbox — gates the editor pixel callback.
/// When false (we're re-rendering the same snapshot for interpolation),
/// GPU work still runs but pixels are not shipped to the editor surface
/// (the content didn't change, so shipping would just thrash rinch's
/// `Mutex<RenderSurfaceBuffer>` with no visible benefit).
pub(super) fn render_one_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    new_snapshot_consumed: bool,
    frame_callback: &FrameCallback,
) -> RenderOutcome {
    let pre = pre::run_pre_frame(state, frame, gpu_instances);
    let encode = encode::encode_viewports(state, frame, &pre);
    post::finalize_frame(state, frame, new_snapshot_consumed, frame_callback, encode)
}
