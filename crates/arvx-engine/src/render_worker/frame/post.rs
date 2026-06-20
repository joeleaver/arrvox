//! Post-frame: kick off MAIN's cloud-sun-atten readback and wire up any
//! pending pick `map_async`.
//!
//! P2: the composite readback drain + pixel ship moved OFF the render thread
//! entirely — the dedicated readback-poll thread now owns map_async → read →
//! ship → unmap → recycle (see `super::super::readback_poll`). The render
//! thread no longer polls the device (the poll thread is the sole poller),
//! and `delivered_dt_ms` is read back from a poll-thread-written atomic.
//!
//! Phase 4-7 of [`super::render_one_frame`]. Reads pick state from
//! [`super::EncodeOutput`] and assembles the final [`super::RenderOutcome`].

use std::sync::atomic::Ordering;

use crate::viewport::ViewportId;

use super::super::state::RenderState;

use super::{EncodeOutput, RenderOutcome};

pub(super) fn finalize_frame(state: &mut RenderState, encode: EncodeOutput) -> RenderOutcome {
    // 4. Kick off MAIN's cloud-sun-atten readback (used by sim's smoothed
    //    sun-color attenuation next frame). The map_async callback fires on
    //    the poll thread's `device.poll`; it stores into atomics this reads.
    let cloud_sun_atten_raw =
        if let Some(main_vr) = state.viewport_renderers.get(&ViewportId::MAIN) {
            main_vr.volumetric.issue_sun_atten_map();
            main_vr.volumetric.sun_atten_value()
        } else {
            f32::NAN
        };

    // 5. (P2) No `device.poll` here. The dedicated readback-poll thread is the
    //    sole poller; it drives every map_async (composite + pick + sun-atten +
    //    LOD stats) and every `on_submitted_work_done` callback on this device.

    // 6. If we issued a pick this frame, wire it up so next frame's
    //    `drain_pick` can return the result to sim. The `active_pending_pick`
    //    filter in encode guarantees `pick_in_flight` is `None` here whenever
    //    `pick_issued` is true, so the `map_async` can't double-map. The
    //    completion callback fires on the poll thread; the render thread reads
    //    the mapped pixels synchronously in `drain_pick` (pick stays render-
    //    side per the P2 design).
    if encode.pick_issued {
        if let Some(pp) = encode.active_pending_pick {
            let (tx, rx) = std::sync::mpsc::channel();
            state
                .pick_readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            state.pick_in_flight = Some((pp, rx));
        }
    }

    // 7. Delivered-frame dt for sim's "delivered FPS" panel. Shipping now
    //    happens on the poll thread, so it publishes the inter-ship interval
    //    into `delivered_dt_bits`; NaN means nothing has shipped yet, which we
    //    surface as `None` so sim holds its previous EMA sample unchanged.
    let delivered_dt_ms = {
        let v = f32::from_bits(state.delivered_dt_bits.load(Ordering::Relaxed));
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    };

    RenderOutcome {
        cloud_sun_atten_raw,
        delivered_dt_ms,
    }
}
