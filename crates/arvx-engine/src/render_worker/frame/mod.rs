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

use super::state::RenderState;

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
    /// CPU-derived scene AABB, used by encode for the shadow-frustum
    /// cull extent in the per-viewport shade params.
    pub(super) scene_aabb: ([f32; 3], [f32; 3]),
    /// Whether this frame's directional shadow map will dispatch.
    /// The per-VR `shade_params.shadow_map_enabled` mirrors this
    /// (gated additionally on InSitu mode).
    pub(super) shadow_map_enabled: bool,
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
/// P2: pixel shipping is no longer the render thread's job — the readback-poll
/// thread ships the newest composite per viewport — so this no longer takes a
/// frame callback or a `new_snapshot_consumed` gate. GPU work still runs every
/// iteration; the poll thread rate-limits and dedupes what actually reaches the
/// editor surface.
pub(super) fn render_one_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[arvx_render::arvx_gpu_object::ArvxGpuInstance],
) -> RenderOutcome {
    // Sub-phase timing inside the render thread, gated on
    // `ARVX_RENDER_PROFILE=1`. Splits the `render` bucket of
    // `[render]` into the three internal calls so we can attribute
    // any unaccounted-for CPU cost.
    let render_profile = std::env::var("ARVX_RENDER_PROFILE").is_ok();
    let phase_start = std::time::Instant::now();
    let pre = pre::run_pre_frame(state, frame, gpu_instances);
    let t_pre = phase_start.elapsed();
    let encode = encode::encode_viewports(state, frame, &pre);
    let t_encode = phase_start.elapsed();
    let outcome = post::finalize_frame(state, encode);
    let t_post = phase_start.elapsed();
    if render_profile {
        let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[render.frame] pre={:.2} encode={:.2} post={:.2} | total={:.2}",
            to_ms(t_pre),
            to_ms(t_encode) - to_ms(t_pre),
            to_ms(t_post) - to_ms(t_encode),
            to_ms(t_post),
        );
    }
    outcome
}
