//! Post-frame: kick off MAIN's cloud-sun-atten readback, drive the
//! wgpu async runtime, wire up any pending pick map_async, drain the
//! per-VR composite readbacks and ship pixels (gated on
//! `new_snapshot_consumed` + `MIN_FRAME_CALLBACK_INTERVAL`).
//!
//! Phase 4-7 of [`super::render_one_frame`]. Reads pick state from
//! [`super::EncodeOutput`] and assembles the final
//! [`super::RenderOutcome`].

use crate::render_frame::RenderFrame;
use crate::viewport::ViewportId;

use super::super::state::{FrameCallback, RenderState, MIN_FRAME_CALLBACK_INTERVAL};

use super::{EncodeOutput, RenderOutcome};

pub(super) fn finalize_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    new_snapshot_consumed: bool,
    frame_callback: &FrameCallback,
    encode: EncodeOutput,
) -> RenderOutcome {
    // 4. Kick off MAIN's cloud-sun-atten readback (used by sim's
    //    smoothed sun-color attenuation next frame).
    let cloud_sun_atten_raw = if let Some(main_vr) =
        state.viewport_renderers.get(&ViewportId::MAIN)
    {
        main_vr.volumetric.issue_sun_atten_map();
        main_vr.volumetric.sun_atten_value()
    } else {
        f32::NAN
    };

    // 5. Drive async runtime so map_async callbacks can fire.
    let _ = state.device.poll(wgpu::PollType::Poll);

    // 6. If we issued a pick this frame, wire it up so next frame's
    //    `drain_pick` can return the result to sim. The
    //    `active_pending_pick` filter above guarantees `pick_in_flight`
    //    is `None` here whenever `pick_issued` is true, so the
    //    `map_async` can't double-map.
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

    // 7. Drain composite readbacks for each visible viewport. The
    //    readback drain itself runs every iteration so the rings
    //    don't back up. Whether to fire the editor pixel callback
    //    is gated on TWO things:
    //
    //    a) (HISTORICAL) `new_snapshot_consumed` — was used to gate
    //       on fresh sim state, but that loses the benefit of render-
    //       side interpolation: between two sim snapshots the render
    //       thread blends per iteration (`alpha` walks 0 → 1), so each
    //       interpolated frame is visually distinct even on the same
    //       snapshot pair. Gating on fresh-snapshot capped the editor
    //       at sim rate (60 fps); dropping it lets the editor see the
    //       interpolated rate up to the rate-limit below.
    //
    //    b) `MIN_FRAME_CALLBACK_INTERVAL` — hard cap (~120 Hz) that
    //       protects the rinch surface buffer Mutex from starving the
    //       editor main thread when render iterates faster than the
    //       editor can composite. Always applied.
    //
    //    Net: pixel ship rate = min(render_rate, 120 Hz). Sim rate
    //    stops gating the editor-visible fps.
    let _ = new_snapshot_consumed; // kept in signature for callers/tests
    let now = std::time::Instant::now();
    let time_ok = now.duration_since(state.last_frame_callback)
        >= MIN_FRAME_CALLBACK_INTERVAL;
    let ship_pixels = time_ok;
    // Interval since the previous successful pixel ship. Sampled
    // BEFORE we update `last_frame_callback` below so we get the
    // gap between ship N-1 and ship N. Only populated when at least
    // one viewport actually handed fresh pixels to the callback —
    // `ship_pixels` gates the try, but `cached_pixels()` may still
    // return None (readback not ready). Delivered FPS should only
    // count real pixel deliveries; a skipped ship leaves the sim
    // EMA unchanged rather than double-counting.
    let mut delivered_dt_ms: Option<f32> = None;
    let mut shipped_any = false;
    let post_t0 = std::time::Instant::now();
    let mut total_drain = std::time::Duration::ZERO;
    let mut total_ship = std::time::Duration::ZERO;
    for vp in &frame.viewports {
        let vr = state
            .viewport_renderers
            .get_mut(&vp.id)
            .expect("viewport renderer must exist");
        let w = vr.width;
        let h = vr.height;
        let padded_row = vr.readback_padded_row();
        let drain_t0 = std::time::Instant::now();
        vr.readback.drain_completed(w, h, padded_row);
        total_drain += drain_t0.elapsed();
        if ship_pixels {
            if let Some((pixels, cw, ch)) = vr.readback.cached_pixels() {
                let ship_t0 = std::time::Instant::now();
                frame_callback(vp.id, pixels, cw, ch);
                total_ship += ship_t0.elapsed();
                shipped_any = true;
            }
        }
    }
    if shipped_any {
        let dt = now.duration_since(state.last_frame_callback);
        delivered_dt_ms = Some(dt.as_secs_f32() * 1000.0);
        state.last_frame_callback = now;
        // Periodic log of the editor-pipeline timing so the user can
        // see where the click-to-visible latency lives. Mirrors the
        // [render-frame] cadence (every 60 ships).
        use std::sync::atomic::{AtomicU32, Ordering};
        static SHIP_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = SHIP_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
        if n % 60 == 0 {
            let post_total = post_t0.elapsed();
            eprintln!(
                "[ship] drain={:.2}ms callback={:.2}ms post_total={:.2}ms \
                 deliver_dt={:.2}ms (~{:.1} fps shipped)",
                total_drain.as_secs_f64() * 1000.0,
                total_ship.as_secs_f64() * 1000.0,
                post_total.as_secs_f64() * 1000.0,
                dt.as_secs_f64() * 1000.0,
                1.0 / dt.as_secs_f64().max(0.0001),
            );
        }
    }

    RenderOutcome { cloud_sun_atten_raw, delivered_dt_ms }
}
