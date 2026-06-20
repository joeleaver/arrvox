//! Render-thread main loop + per-snapshot interpolation helpers.
//!
//! [`run_render_thread`] is the function spawned by [`super::state::RenderWorker::spawn`].
//! It bootstraps `RenderState`, then loops:
//!   1. drain commands
//!   2. take the newest snapshot non-blockingly (newest-wins inbox)
//!   3. compute interpolation alpha
//!   4. blend prev/curr instances via [`interpolate_instances`]
//!   5. dispatch one frame via [`super::frame::render_one_frame`]
//!   6. ship `RenderResult` back to sim
//!   7. pace
//!
//! Interpolation:
//! - [`interpolate_instances`] matches by `object_id`, TRS-blends world
//!   matrices for objects that exist in both prev and curr.
//! - [`lerp_world_matrix`] does the per-object TRS decompose / lerp / slerp /
//!   recompose.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender};

use crate::render_frame::{RenderCommand, RenderFrame, RenderInit, RenderResult};

use super::frame::render_one_frame;
use super::readback_poll::RenderReadbackHandles;
use super::state::{RenderInbox, RenderState};

/// Top-level render-thread entry point.
///
/// Sequence:
/// 1. Bootstrap: block on the inbox for the first sim snapshot.
///    Nothing to render before then.
/// 2. Loop:
///    a. Drain commands (resize, shutdown, etc.).
///    b. Try-take the next snapshot non-blockingly. On hit, shift
///       curr → prev, install new as curr, update the `sim_dt_estimate`
///       EMA from the observed arrival interval.
///    c. Compute the interpolation alpha from wall-clock elapsed
///       since `curr_snap_time` divided by the sim_dt estimate.
///    d. Build an interpolated `gpu_objects` vec (prev → curr by
///       object_id, TRS-blend world matrices). When α hits 1.0 or
///       no prev exists, this is a cheap `.clone()` of curr.
///    e. Run the full per-VR encode/submit/readback cycle against
///       the interpolated objects + curr's other fields.
///    f. Send the RenderResult back to sim.
///    g. Sleep to the configured `render_pacing` target.
pub(super) fn run_render_thread(
    init: RenderInit,
    inbox: Arc<RenderInbox>,
    cmd_rx: Receiver<RenderCommand>,
    out_tx: Sender<RenderResult>,
    handles: RenderReadbackHandles,
) {
    let render_pacing = init.render_pacing;
    let mut state = RenderState::new(init, handles);

    // P2 queue-depth pacing cap: max submissions allowed in flight on the GPU
    // before the render thread skips an iteration. Bounds queue depth (the old
    // gate's job) without coupling to readback. Default 4 ≈ a few frames; raise
    // for more GPU overlap, lower for lower latency.
    let max_inflight: u32 = std::env::var("ARVX_MAX_INFLIGHT_SUBMITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    // Bootstrap: wait for the first snapshot. `None` = shutdown
    // signal arrived before any snapshot ever did.
    let first = match inbox.take_blocking() {
        Some(f) => f,
        None => return,
    };
    state.curr_snap_time = std::time::Instant::now();
    state.curr_snap = Some(Arc::new(first));

    // Wall-clock instant of the last *actual* render iteration. We skip
    // queue-depth-paced iterations (see `2a. Queue-depth pacing`) when
    // computing this — counting them would inflate the panel's "Render FPS"
    // to reflect the 500 µs pacing sleep instead of the true GPU-bound
    // production rate. Reset to `None` here so the first real iteration
    // carries no dt (sim falls back to its prior EMA value for that frame).
    let mut prev_render_start: Option<std::time::Instant> = None;

    // Queue-depth-pacing wedge probe. The render thread refuses to submit a new
    // frame while `inflight_submits >= cap`; that count is drained only by the
    // poll thread's `device.poll` firing `on_submitted_work_done`. A BRIEF stay
    // at the cap is healthy pacing under load; a LONG one means submissions
    // aren't completing (or the poll thread isn't draining) — a wedge. We log a
    // canary so that failure mode is visible in the log instead of a silent
    // freeze. `pace_blocked_since` is `Some` while we're parked at the cap.
    let mut pace_blocked_since: Option<std::time::Instant> = None;
    let mut pace_last_log: Option<std::time::Instant> = None;

    // Per-scope rolling sample buffer for the [render.gpu.percentiles]
    // diagnostic dump. Bounded to PERCENTILE_WINDOW; oldest sample
    // drops when full. Costs ~PERCENTILE_WINDOW × 4 B × scope_count
    // (≤ 30 scopes typical → ~3.5 KB) and only allocates when
    // `ARVX_RENDER_PROFILE=1` is set.
    const PERCENTILE_WINDOW: usize = 64;
    let mut percentile_history: std::collections::HashMap<String, std::collections::VecDeque<f32>> =
        std::collections::HashMap::new();

    loop {
        let iter_start = std::time::Instant::now();

        // 1. Commands. Shutdown exits the loop immediately; other
        //    commands (resize, visibility) apply to renderer state
        //    for the upcoming render.
        while let Ok(cmd) = cmd_rx.try_recv() {
            if matches!(cmd, RenderCommand::Shutdown) {
                return;
            }
            state.apply_command(cmd);
        }
        if inbox.is_shutdown() {
            return;
        }

        // 2. Recycle composite ring slots the readback-poll thread has
        //    finished reading + unmapping, so the next encode can claim a
        //    writable slot. Non-blocking; the generation tag makes a stale free
        //    (e.g. arriving after a viewport resize recreated the buffers) a
        //    harmless no-op (see `ReadbackRing::free_slot`).
        while let Ok(free) = state.slot_free_rx.try_recv() {
            if let Some(vr) = state.viewport_renderers.get_mut(&free.vp_id) {
                vr.readback.free_slot(free.slot, free.generation);
            }
        }

        // 2a. Queue-depth pacing — the replacement for the DELETED readback
        //     backpressure gate.
        //
        //     The old gate coupled frame submission to readback-slot
        //     availability, which is coupled to GPU-queue depth: a load burst
        //     filled the ring and the render thread either spun (stale surface)
        //     or hard-blocked on `wait_indefinitely` (frozen renderer). P2
        //     deletes that coupling entirely — the dedicated poll thread owns
        //     readback + present and keeps shipping the newest frame regardless.
        //
        //     The one thing the gate got RIGHT was bounding queue depth so a
        //     fast CPU loop can't pile 70+ frames of GPU work behind everything.
        //     We keep THAT, decoupled from readback: cap the number of
        //     submissions still executing on the GPU (tracked via
        //     `on_submitted_work_done`, fired by the poll thread's
        //     `device.poll`). At the cap, skip this iteration with a short
        //     bounded sleep — never an indefinite block, never readback-coupled.
        let inflight_now = state.inflight_submits.load(Ordering::Relaxed);
        if inflight_now >= max_inflight {
            let now = std::time::Instant::now();
            let since = *pace_blocked_since.get_or_insert(now);
            let blocked_ms = now.duration_since(since).as_secs_f32() * 1000.0;
            // Healthy pacing clears in a few ms. A sustained park means
            // submissions aren't completing / the poll thread isn't draining
            // `inflight_submits` — that's the freeze signature, made loud.
            if blocked_ms > 250.0 {
                let due = pace_last_log
                    .map(|t| now.duration_since(t).as_millis() >= 500)
                    .unwrap_or(true);
                if due {
                    pace_last_log = Some(now);
                    eprintln!(
                        "[render-pace] WEDGED inflight={inflight_now}/{max_inflight} for {blocked_ms:.0}ms \
                         — submissions not draining (poll thread stalled?)"
                    );
                }
            }
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        // Cleared the cap — pair the wedge canary if we'd been parked long.
        if let Some(since) = pace_blocked_since.take() {
            let blocked_ms = std::time::Instant::now()
                .duration_since(since)
                .as_secs_f32()
                * 1000.0;
            if blocked_ms > 250.0 {
                eprintln!("[render-pace] recovered after {blocked_ms:.0}ms");
            }
            pace_last_log = None;
        }

        // We are about to render a real frame. Compute the dt back
        // to the last real render — this is what becomes the panel's
        // "Render FPS". Excluding queue-depth-paced iterations from the
        // dt means the rate reflects honest GPU-bound throughput, not
        // the 500 µs pacing sleep.
        let render_dt_ms = prev_render_start
            .map(|p| iter_start.duration_since(p).as_secs_f32() * 1000.0);
        prev_render_start = Some(iter_start);

        // Periodically log the frame cadence so the user can tell
        // whether the GPU is keeping up with sim. One line every 60
        // real frames — at 60 fps that's once a second; at 1 fps
        // it's once a minute. The dt itself is what matters for
        // diagnosis (>=33 ms → <30 fps → user-perceptible
        // upload-to-visible lag).
        if let Some(dt) = render_dt_ms {
            use std::sync::atomic::{AtomicU32, Ordering};
            static FRAME_LOG_COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = FRAME_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
            // Always surface a STALL frame (>100ms = <10fps). The
            // 60-frame cadence below would otherwise miss the single long
            // frame during a cold terrain generation that lets the window
            // surface go Outdated — the trigger for rinch #42's permanent
            // "surface lost". This is the frame to correlate with the
            // sim-side `[terrain-tick]` / `[geo-epoch]` lines.
            if dt > 100.0 {
                eprintln!(
                    "[render-frame] STALL dt={dt:.1}ms (~{:.1} fps) — surface may go Outdated",
                    1000.0 / dt.max(0.1)
                );
            } else if n % 60 == 0 {
                eprintln!("[render-frame] dt={dt:.1}ms (~{:.1} fps)", 1000.0 / dt.max(0.1));
            }
        }

        // Render-thread sub-phase timing, gated on
        // `ARVX_RENDER_PROFILE=1`. Mirrors the sim-thread `[snap]`
        // line — emits one ms-split per real frame so we can
        // attribute the gap between GPU-pass timestamps and
        // wallclock frame time.
        let render_profile = std::env::var("ARVX_RENDER_PROFILE").is_ok();
        let render_phase_start = std::time::Instant::now();

        // 2b. Drain a completed pick, if any. Non-blocking.
        //
        // Must run AFTER the queue-depth pacing skip above. `drain_pick` calls
        // `pick_in_flight.take()` so the pick is consumed; if we drained before
        // the pacing skip and then hit the `continue`, the PickResult would be
        // silently dropped (no `out_tx.send` runs on a skipped iteration) and
        // the click would never reach sim. Picks tolerate a few ms of extra
        // latency; outright losing them does not. (The pick's map_async
        // completion is driven by the poll thread's `device.poll`.)
        let pick_result = state.drain_pick();
        let r_t_pick = render_phase_start.elapsed();

        // 3. Check for a fresh snapshot — non-blocking. If present,
        //    update the two-snapshot window and refresh the sim_dt
        //    EMA from the observed interval.
        if let Some(new) = inbox.try_take() {
            let observed = iter_start.duration_since(state.curr_snap_time);
            // EMA the sim_dt estimate toward the observed interval.
            // Clamp observed to avoid div-by-zero if two snapshots
            // arrive in the same microsecond.
            let observed = observed.max(std::time::Duration::from_micros(100));
            const SIM_DT_EMA_ALPHA: f32 = 0.25;
            let prev_s = state.sim_dt_estimate.as_secs_f32();
            let obs_s = observed.as_secs_f32();
            let smoothed = prev_s * (1.0 - SIM_DT_EMA_ALPHA) + obs_s * SIM_DT_EMA_ALPHA;
            state.sim_dt_estimate = std::time::Duration::from_secs_f32(smoothed);

            state.prev_snap = state.curr_snap.take();
            state.curr_snap = Some(Arc::new(new));
            state.curr_snap_time = iter_start;
        }

        // Arc-clone so we hold a borrow-checker-friendly ref to the
        // snapshot that's disjoint from `&mut state` below.
        let curr: Arc<RenderFrame> = state
            .curr_snap
            .as_ref()
            .expect("bootstrap guarantees curr_snap is Some")
            .clone();
        let prev: Option<Arc<RenderFrame>> = state.prev_snap.clone();
        let frame_index = curr.frame_index;

        let r_t_snap = render_phase_start.elapsed();

        // 4. Interpolation alpha. At α=0 we'd show prev; at α=1 we
        //    show curr. Clamp to [0,1] so render never extrapolates
        //    past the latest sim state (extrapolation is a correctness
        //    minefield when sim is faster than render's expectation).
        let alpha = (iter_start
            .duration_since(state.curr_snap_time)
            .as_secs_f32()
            / state.sim_dt_estimate.as_secs_f32().max(1e-4))
            .clamp(0.0, 1.0);

        // 5. Build the instance list we'll actually upload. If there's
        //    a prev snapshot and α < 1, blend; otherwise use curr
        //    directly (free — borrows the snapshot's `Arc<Vec<…>>`,
        //    no per-object work at sim rate).
        //    Assets don't interpolate — they're pose-static for a frame.
        //
        //    `interp_owned` is `Some(_)` only when we actually had to
        //    interpolate (allocates a fresh Vec). The non-interp branch
        //    leaves it `None` and we deref into the snapshot's Arc
        //    directly — saves the per-tick `Vec::clone` that the prior
        //    `curr.gpu_instances.clone()` paid (PERF_DEBT A3).
        let interp_owned: Option<Vec<arvx_render::arvx_gpu_object::ArvxGpuInstance>> =
            match (prev.as_ref(), alpha < 0.999) {
                (Some(p), true) => Some(interpolate_instances(
                    &p.gpu_instances,
                    &curr.gpu_instances,
                    alpha,
                )),
                _ => None,
            };
        let interp_instances: &[arvx_render::arvx_gpu_object::ArvxGpuInstance] =
            interp_owned
                .as_deref()
                .unwrap_or(curr.gpu_instances.as_slice());

        let r_t_interp = render_phase_start.elapsed();

        // 6. Render — `render_one_frame` takes the interpolated objects as an
        //    explicit parameter separate from the snapshot (the snapshot's own
        //    `gpu_objects` field is the canonical curr data). It encodes,
        //    submits, and hands the composite slot to the readback-poll thread;
        //    it does NOT ship pixels itself (P2).
        let outcome = render_one_frame(&mut state, &curr, interp_instances);
        let r_t_render = render_phase_start.elapsed();

        // 7. GPU profiler — drain resolved timings for sim's history.
        let gpu_passes = state.renderer.end_profiler_frame(frame_index);
        let r_t_prof = render_phase_start.elapsed();

        // 7b. Optional GPU-side breakdown of the TLAS chain. The `tlas`
        //     line in `[render.pre]` is CPU wall-time; this one shows
        //     where the GPU is actually spending its time inside
        //     `build_gpu_tlas`. Gated on `render_profile` like the
        //     `[render]` summary.
        if render_profile {
            // Update rolling per-scope sample history every frame
            // when profiling is on. Cap at PERCENTILE_WINDOW so the
            // memory stays bounded; oldest sample drops first.
            for (label, ms) in &gpu_passes {
                let buf = percentile_history
                    .entry(label.clone())
                    .or_insert_with(|| {
                        std::collections::VecDeque::with_capacity(PERCENTILE_WINDOW)
                    });
                if buf.len() == PERCENTILE_WINDOW {
                    buf.pop_front();
                }
                buf.push_back(*ms);
            }
            // Emit every 30 frames at steady state, AND every frame
            // when the render frame is slow (>33 ms — anything below
            // 30 fps). The slow-frame gate keeps perf hunts loud
            // during the spike that triggered them; the periodic
            // gate keeps idle from spamming.
            let slow_frame = render_dt_ms.map(|d| d > 33.0).unwrap_or(false);
            if frame_index % 30 == 0 || slow_frame {
                // Latest single-sample line — same format as before.
                let labels: Vec<String> = gpu_passes
                    .iter()
                    .map(|(l, ms)| format!("{l}={ms:.2}"))
                    .collect();
                eprintln!(
                    "[render.gpu] frame={} count={} | {}",
                    frame_index,
                    gpu_passes.len(),
                    labels.join(" ")
                );
                // Rolling-window percentiles. Same scope ordering as
                // the latest line so a reader can scan side-by-side.
                let pct: Vec<String> = gpu_passes
                    .iter()
                    .map(|(l, _)| {
                        let buf = percentile_history.get(l);
                        let (p50, p95, p99) = match buf {
                            Some(b) if !b.is_empty() => percentiles_p50_p95_p99(b),
                            _ => (0.0, 0.0, 0.0),
                        };
                        let n = buf.map(|b| b.len()).unwrap_or(0);
                        format!("{l}=p50:{p50:.2}/p95:{p95:.2}/p99:{p99:.2}@{n}")
                    })
                    .collect();
                eprintln!(
                    "[render.gpu.percentiles] frame={} window={} | {}",
                    frame_index,
                    PERCENTILE_WINDOW,
                    pct.join(" ")
                );
            }
        }

        // 8. Send result back to sim. Exit on disconnect.
        if out_tx
            .send(RenderResult {
                frame_index,
                pick_result,
                cloud_sun_atten_raw: outcome.cloud_sun_atten_raw,
                gpu_passes,
                render_dt_ms,
                delivered_dt_ms: outcome.delivered_dt_ms,
            })
            .is_err()
        {
            return;
        }
        let r_t_send = render_phase_start.elapsed();
        if render_profile {
            let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
            eprintln!(
                "[render] pick={:.2} snap={:.2} interp={:.2} render={:.2} prof={:.2} send={:.2} | total={:.2}",
                to_ms(r_t_pick),
                to_ms(r_t_snap) - to_ms(r_t_pick),
                to_ms(r_t_interp) - to_ms(r_t_snap),
                to_ms(r_t_render) - to_ms(r_t_interp),
                to_ms(r_t_prof) - to_ms(r_t_render),
                to_ms(r_t_send) - to_ms(r_t_prof),
                to_ms(r_t_send),
            );
        }

        // 9. Pace. `Uncapped` skips the sleep entirely; `TargetHz(N)`
        //    sleeps the remainder of this iteration's target interval.
        if let Some(target) = render_pacing.target_interval() {
            let elapsed = iter_start.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
    }
}

/// Interpolate per-object world transforms between two snapshots.
///
/// Matches objects by `object_id` (stable across frames). For each
/// object in `curr`, if a same-id counterpart exists in `prev` the
/// world matrix is TRS-blended (translation / scale lerp, rotation
/// slerp). Objects without a prev counterpart (newly spawned this sim
/// tick) use `curr` verbatim.
///
/// All non-transform fields (asset_id, material id, bone-field offsets,
/// etc.) come from `curr` — those change on sim edits, not between sim
/// ticks. Skinned entities still carry their bone pose via the separate
/// bone-field buffer; their `world` is usually identity and the lerp is
/// a no-op. Inverse-world isn't stored anymore — shaders compute
/// `mat4_affine_inverse(inst.world)` on demand.
fn interpolate_instances(
    prev: &[arvx_render::arvx_gpu_object::ArvxGpuInstance],
    curr: &[arvx_render::arvx_gpu_object::ArvxGpuInstance],
    alpha: f32,
) -> Vec<arvx_render::arvx_gpu_object::ArvxGpuInstance> {
    // object_id → index-in-prev, built once per render tick. Small
    // HashMap is fine; scenes rarely have > a few hundred objects.
    let mut prev_by_id: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::with_capacity(prev.len());
    for (i, p) in prev.iter().enumerate() {
        prev_by_id.insert(p.object_id, i);
    }

    curr
        .iter()
        .map(|c| {
            let Some(&pi) = prev_by_id.get(&c.object_id) else {
                return *c;
            };
            let p = &prev[pi];
            // Fast path: world matrices byte-identical → no motion,
            // skip the decompose/recompose dance entirely.
            if p.world == c.world {
                return *c;
            }
            let mut out = *c;
            out.world = lerp_world_matrix(&p.world, &c.world, alpha);
            out
        })
        .collect()
}

/// TRS-decompose both matrices, blend components separately, recompose.
/// Rotation uses `Quat::slerp` for shortest-arc correctness; scale and
/// translation use linear lerp.
///
/// `to_scale_rotation_translation` can misbehave on degenerate matrices
/// (zero determinant, reflections, etc.); for well-formed affine world
/// matrices — the common case — it's correct.
fn lerp_world_matrix(
    a: &[[f32; 4]; 4],
    b: &[[f32; 4]; 4],
    alpha: f32,
) -> [[f32; 4]; 4] {
    let ma = glam::Mat4::from_cols_array_2d(a);
    let mb = glam::Mat4::from_cols_array_2d(b);
    let (sa, ra, ta) = ma.to_scale_rotation_translation();
    let (sb, rb, tb) = mb.to_scale_rotation_translation();
    let s = sa.lerp(sb, alpha);
    let r = ra.slerp(rb, alpha);
    let t = ta.lerp(tb, alpha);
    glam::Mat4::from_scale_rotation_translation(s, r, t).to_cols_array_2d()
}

/// p50 / p95 / p99 over a window of `f32` samples. Sorts a clone
/// (input is left untouched). `total_cmp` for NaN-safety so a
/// stray NaN in the profiler stream doesn't panic the diagnostic
/// path. Returns (0,0,0) for empty input.
fn percentiles_p50_p95_p99(samples: &std::collections::VecDeque<f32>) -> (f32, f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut v: Vec<f32> = samples.iter().copied().collect();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let p50 = v[n / 2];
    let p95 = v[(n * 95 / 100).min(n - 1)];
    let p99 = v[(n * 99 / 100).min(n - 1)];
    (p50, p95, p99)
}
