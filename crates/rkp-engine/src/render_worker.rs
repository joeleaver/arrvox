//! Render thread — owns wgpu, consumes [`RenderFrame`] snapshots from
//! sim, returns [`RenderResult`] back.
//!
//! Architecture overview
//! ---------------------
//!
//! Sim and render run on independent threads at independent rates:
//!
//! ```text
//!   sim thread                         render thread
//!   ──────────                         ─────────────
//!   tick()                             loop {
//!     run gameplay/physics/anim          drain commands (resize, etc.)
//!     build RenderFrame snapshot         drain pick_in_flight
//!     inbox.submit(snap) ─────────►      inbox.try_take() ─► update prev/curr
//!     drain RenderResult                 compute interpolation α
//!     pace by sim_pacing                 render interpolated state
//!   }                                    outbox.send(result) ──────►  back to sim
//!                                        pace by render_pacing
//!                                      }
//! ```
//!
//! - **inbox** is single-slot, newest-wins. Sim never blocks; if render
//!   is behind, the older snapshot is silently discarded.
//! - **outbox** is unbounded; sim drains every tick.
//! - **commands** are aperiodic events (resize, viewport visibility,
//!   shutdown) sent over a normal crossbeam channel.
//!
//! Interpolation
//! -------------
//!
//! When `render_pacing` runs faster than `sim_pacing` (e.g. render at
//! 240 Hz, sim at 60 Hz), the render thread ticks 4× per sim snapshot.
//! Without interpolation each of those 4 renders would show the same
//! frozen sim state — fast-moving objects would *judder* at the sim
//! rate instead of smoothly at the render rate.
//!
//! To fix that, the render thread keeps the two most recent snapshots
//! (`prev` + `curr`) and a running estimate of `sim_dt` (time between
//! snapshot arrivals). Each render tick computes
//! `α = (now - curr_arrival) / sim_dt`, clamped to [0,1], and blends
//! object world transforms between prev and curr before upload.
//! Rotation uses slerp, translation / scale use linear lerp; other
//! snapshot fields (lights, environment, cameras, etc.) use `curr`
//! directly — the MVP covers physics-driven object motion, which is
//! where judder is most visible.
//!
//! When `render_pacing` runs slower than or equal to `sim_pacing`,
//! α hits 1.0 almost immediately and the interpolation math is a
//! no-op — we just use `curr.gpu_objects` directly.
//!
//! Ownership
//! ---------
//!
//! The render thread owns: `device`, `queue`, [`RkpRenderer`], the
//! per-viewport renderer hashmap, the pick-readback buffer, and the
//! GPU profiler (inside `RkpRenderer`).
//!
//! `scene_mgr` is shared via `Arc<Mutex<>>` with sim *and* the bake
//! worker. Render only locks it during geometry uploads — a few hundred
//! microseconds — so contention is negligible. Sim writes to scene_mgr
//! between snapshots (when loading assets, applying bake results); it
//! sets `RenderFrame::geometry_dirty = true` whenever something changed
//! so render knows to re-upload.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::JoinHandle;

use crossbeam::channel::{Receiver, Sender};

use rkp_render::{
    rkp_renderer::RkpRenderer, rkp_scene::FrameUpload, rkp_scene_manager::RkpSceneManager,
    ViewportRenderer,
};

use crate::render_frame::{
    PendingPick, PickResult, RenderCommand, RenderFrame, RenderInit, RenderResult,
};
use crate::viewport::ViewportId;

/// Handle returned by [`RenderWorker::spawn`]. The sim side keeps this;
/// dropping it triggers a graceful shutdown of the render thread.
pub struct RenderWorker {
    /// Submit a [`RenderFrame`] for rendering. Newest-wins: if the
    /// previous frame hasn't been consumed yet, it's dropped.
    pub inbox: Arc<RenderInbox>,
    /// Receive one [`RenderResult`] per rendered frame. Sim drains
    /// every tick — non-blocking.
    pub outbox: Receiver<RenderResult>,
    /// Send aperiodic commands (resize, shutdown, …).
    pub commands: Sender<RenderCommand>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for RenderWorker {
    fn drop(&mut self) {
        // Best-effort shutdown — if the channel is already closed
        // (render thread crashed), the send fails and we still wake the
        // condvar so the inbox waiter exits.
        let _ = self.commands.send(RenderCommand::Shutdown);
        self.inbox.shutdown();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Single-slot mailbox with newest-wins replacement and condvar wakeup.
///
/// Sim thread calls [`submit`] every tick; the previous frame (if any)
/// is dropped. Render thread calls [`take_blocking`] to wait for the
/// next frame; returns `None` once `shutdown` has been signalled.
///
/// [`submit`]: RenderInbox::submit
/// [`take_blocking`]: RenderInbox::take_blocking
pub struct RenderInbox {
    slot: Mutex<Option<RenderFrame>>,
    notify: Condvar,
    shutdown: AtomicBool,
}

impl RenderInbox {
    fn new() -> Self {
        Self {
            slot: Mutex::new(None),
            notify: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Place `frame` in the inbox, dropping any previously-unconsumed
    /// frame. O(1); never blocks.
    pub fn submit(&self, frame: RenderFrame) {
        let mut slot = self.slot.lock().expect("RenderInbox slot poisoned");
        *slot = Some(frame);
        // Drop the lock before notifying — wakers grab it next.
        drop(slot);
        self.notify.notify_one();
    }

    /// Block until either a frame arrives (returned as `Some`) or the
    /// shutdown flag is set (returned as `None`). Used once at render-
    /// thread bootstrap — there's nothing to render before the first
    /// sim tick.
    fn take_blocking(&self) -> Option<RenderFrame> {
        let mut slot = self.slot.lock().expect("RenderInbox slot poisoned");
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            if let Some(f) = slot.take() {
                return Some(f);
            }
            slot = self.notify.wait(slot).expect("RenderInbox cond wait poisoned");
        }
    }

    /// Non-blocking take. Returns `None` if no new frame has arrived
    /// since the last call. Used in the steady-state render loop
    /// where render has its own clock and re-renders the current
    /// snapshot (interpolated) when no newer one is waiting.
    fn try_take(&self) -> Option<RenderFrame> {
        self.slot
            .lock()
            .expect("RenderInbox slot poisoned")
            .take()
    }

    /// `true` once [`Self::shutdown`] has been signalled.
    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Signal the render thread to exit at its next inbox check.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_all();
    }
}

/// Frame-pixel callback fired once per visible viewport per frame.
/// Same shape as the legacy `FrameCallback` on the sim path —
/// the editor surface writers don't care which thread it runs on.
pub type FrameCallback =
    Box<dyn Fn(ViewportId, &[u8], u32, u32) + Send + 'static>;

impl RenderWorker {
    /// Spawn the render thread and return a handle.
    ///
    /// The render thread takes ownership of `init.device`, `init.queue`,
    /// builds the [`RkpRenderer`] + per-viewport renderers there, then
    /// enters its render loop. `frame_callback` is invoked once per
    /// visible viewport per produced frame, on the render thread.
    pub fn spawn(init: RenderInit, frame_callback: FrameCallback) -> Self {
        let inbox = Arc::new(RenderInbox::new());
        let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded::<RenderCommand>();
        let (out_tx, out_rx) = crossbeam::channel::unbounded::<RenderResult>();

        let inbox_for_thread = inbox.clone();
        let handle = std::thread::Builder::new()
            .name("rkp-render".to_string())
            .spawn(move || {
                run_render_thread(init, inbox_for_thread, cmd_rx, out_tx, frame_callback);
            })
            .expect("spawn rkp-render thread");

        Self {
            inbox,
            outbox: out_rx,
            commands: cmd_tx,
            handle: Some(handle),
        }
    }
}

/// Internal per-render-thread state. Owns wgpu resources, the
/// in-flight pick channel, and the two-snapshot interpolation window.
struct RenderState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: RkpRenderer,
    viewport_renderers: std::collections::HashMap<ViewportId, ViewportRenderer>,
    scene_mgr: Arc<Mutex<RkpSceneManager>>,

    /// Phase C — GPU runtime geometry pass + transient cache. The
    /// pass owns the geom-build pipeline; the cache tracks
    /// per-(host_object, material) slices in the scene's transient
    /// pool tail. Both live on the render thread because they're
    /// driven entirely by GPU work that downstream march/shade
    /// dispatches consume in the same encoder.
    user_shader_pass: rkp_render::user_shader_pass::UserShaderPass,
    user_shader_cache: rkp_render::user_shader_pass::UserShaderObjectCache,

    /// Pick readback target. 1×1 region of the gbuf_material at offset
    /// 0, 1×1 region of the gbuf_pick at offset 256 — both 256-byte
    /// aligned per wgpu's copy alignment rules.
    pick_readback_buffer: wgpu::Buffer,

    /// In-flight pick — set when a pick was encoded last frame and we
    /// kicked off `map_async` post-submit. Drained at the top of each
    /// frame; if ready, render returns the raw payload back to sim
    /// (which owns the gpu_to_entity mapping for the final resolve).
    pick_in_flight: Option<(PendingPick, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,

    /// Most recent snapshot; the source of truth for non-interpolated
    /// fields (lights, environment, cameras, proc_raymarch, etc.).
    /// `Arc` so we can hand a cheap reference to `render_one_frame`
    /// without aliasing the rest of `RenderState` from the borrow
    /// checker's point of view.
    curr_snap: Option<Arc<RenderFrame>>,
    /// Wall-clock instant at which `curr_snap` was received. Used to
    /// compute the interpolation alpha for the active render tick.
    curr_snap_time: std::time::Instant,
    /// Snapshot immediately before `curr_snap`, kept for world-matrix
    /// interpolation. `None` while we've only seen the first snapshot
    /// (render proceeds without interpolation in that case).
    prev_snap: Option<Arc<RenderFrame>>,
    /// EMA of time between snapshot arrivals. Used as the denominator
    /// when converting wall-clock time since `curr_snap_time` to
    /// interpolation alpha. Starts at 16.67 ms (60 Hz) so the first
    /// few frames have a sane estimate before the EMA has converged.
    sim_dt_estimate: std::time::Duration,
    /// Last `scene_mgr.geometry_epoch()` value we successfully
    /// uploaded to the GPU. When a snapshot arrives with a higher
    /// epoch, render takes the scene_mgr lock and re-uploads.
    /// Robust to snapshot drops by design (sim ships epoch every
    /// frame, not a one-shot dirty bit).
    last_uploaded_geometry_epoch: u64,

    /// Brush-overlay epoch of the last successful upload to shade's
    /// per-leaf distance buffer. Compared each frame to the incoming
    /// snapshot's `brush_overlay_epoch` to decide whether the cursor
    /// data needs re-uploading.
    last_uploaded_brush_overlay_epoch: u64,

    /// Paint-data epoch of the last successful slice-upload to
    /// `leaf_attr_pool_buffer` + `color_pool_buffer`. Bumped by
    /// paint strokes; compared each frame to decide whether to
    /// slice-write the dirty range.
    last_uploaded_paint_epoch: u64,

    /// `view_proj` of the most recent rendered frame, per viewport.
    /// Overrides the `prev_vp` baked into incoming snapshots before
    /// camera + volumetric uploads — without this, TAA reprojection
    /// (cloud march, octree march, shade) reads from a `prev_vp` that
    /// describes the camera one *sim* tick ago, which no longer
    /// matches what we actually drew last because the GPU-backpressure
    /// gate may skip multiple snapshots between renders. The result is
    /// streaks/blur in any temporal accumulator. We stash the
    /// un-interpolated `curr.viewports[i].camera.view_proj` here at
    /// the end of each render iteration and read it back at the top
    /// of the next.
    last_rendered_vp: std::collections::HashMap<ViewportId, [[f32; 4]; 4]>,

    /// Wall-clock instant of the last `frame_callback` invocation.
    /// We rate-limit pixel callbacks (see [`MIN_FRAME_CALLBACK_INTERVAL`])
    /// because rinch's `surface_writer.submit_frame` holds an
    /// `8 MB` Mutex<Buffer> lock for the duration of a memcpy. At
    /// `Uncapped` render rates that lock is held >100% of wall time,
    /// starving the editor's main thread (which holds the same lock
    /// for an 8 MB clone during composite). The visible symptom is
    /// "render reports 200 fps but the editor surface updates at
    /// ~1 fps" because the main thread can't get the lock.
    ///
    /// Render still iterates uncapped — interpolation and physics
    /// keep the GPU pipeline full, the readback rings keep filling.
    /// We just don't ship every produced frame to the editor; the
    /// editor only displays at vsync anyway.
    last_frame_callback: std::time::Instant,
}

/// Minimum wall-clock between two `frame_callback` invocations. ~120 Hz —
/// generous enough for high-refresh editor surfaces while still keeping
/// the surface buffer Mutex out of the lock-saturated regime.
const MIN_FRAME_CALLBACK_INTERVAL: std::time::Duration =
    std::time::Duration::from_micros(8_300);

impl RenderState {
    fn new(init: RenderInit) -> Self {
        let device = init.device;
        let queue = init.queue;
        let mut renderer = RkpRenderer::new(&device, &queue, init.initial_width, init.initial_height);

        // Pre-build the MAIN + BUILD viewport renderers at their default
        // sizes. BUILD is preallocated (~20 MiB) so flipping its
        // visibility on later doesn't pay creation latency mid-session.
        let main_vr = ViewportRenderer::new(
            &device, &queue, &mut renderer, init.initial_width, init.initial_height,
        );
        let build_vr = ViewportRenderer::new(&device, &queue, &mut renderer, 800, 600);

        let mut viewport_renderers = std::collections::HashMap::new();
        viewport_renderers.insert(ViewportId::MAIN, main_vr);
        viewport_renderers.insert(ViewportId::BUILD, build_vr);

        // Pick readback: 768 B = 256 (material slot) + 256 (pick slot)
        // + 256 (position slot). Each slot is 256-B aligned to satisfy
        // wgpu's `copy_texture_to_buffer` row-alignment requirement.
        let pick_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp pick readback"),
            size: 768,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let user_shader_pass = rkp_render::user_shader_pass::UserShaderPass::new(&device);
        let user_shader_cache = rkp_render::user_shader_pass::UserShaderObjectCache::new();

        Self {
            device,
            queue,
            renderer,
            viewport_renderers,
            scene_mgr: init.scene_mgr,
            user_shader_pass,
            user_shader_cache,
            pick_readback_buffer,
            pick_in_flight: None,
            curr_snap: None,
            curr_snap_time: std::time::Instant::now(),
            prev_snap: None,
            // Seed with a plausible sim rate so alpha starts sane
            // before the EMA has any data. 60 Hz is the default.
            sim_dt_estimate: std::time::Duration::from_nanos(16_666_667),
            // 0 = "never uploaded any geometry yet" — the first
            // snapshot with epoch > 0 triggers an upload.
            last_uploaded_geometry_epoch: 0,
            last_uploaded_brush_overlay_epoch: 0,
            last_uploaded_paint_epoch: 0,
            // Empty until the first render — the first frame's
            // override falls back to the snapshot's own view_proj
            // (i.e. prev_vp == view_proj, no motion).
            last_rendered_vp: std::collections::HashMap::new(),
            // Sub-zero so the first frame's callback always fires.
            last_frame_callback: std::time::Instant::now()
                - std::time::Duration::from_secs(1),
        }
    }

    /// Apply an aperiodic command. Resize / visibility commands take
    /// effect immediately; subsequent snapshots will fill the new
    /// dimensions.
    fn apply_command(&mut self, cmd: RenderCommand) {
        match cmd {
            RenderCommand::ResizeViewport { id, width, height } => {
                if let Some(vr) = self.viewport_renderers.get_mut(&id) {
                    vr.resize(&self.device, &mut self.renderer, width, height);
                }
            }
            RenderCommand::SetViewportVisible { .. } => {
                // Visibility is reflected in the snapshot's `viewports`
                // list — render walks only what sim ships. Nothing to
                // do here today; left as a hook for future allocator
                // work (free a hidden VR's gbuffer to reclaim VRAM).
            }
            RenderCommand::SetViewportMode { .. }
            | RenderCommand::SetBuildPreviewMode(_) => {
                // Same: per-frame snapshot carries `mode` + `preview_mode`
                // for every viewport. Render reads from there.
            }
            RenderCommand::Shutdown => {
                // Handled by the outer loop's poll of `is_shutdown`.
            }
        }
    }

    /// Drain the in-flight pick if its async map has completed.
    /// Returns the decoded payload to ship back to sim, or `None` if
    /// nothing is ready (or no pick is in flight).
    fn drain_pick(&mut self) -> Option<PickResult> {
        let ready = self
            .pick_in_flight
            .as_ref()
            .map(|(_, rx)| rx.try_recv().is_ok())
            .unwrap_or(false);
        if !ready {
            return None;
        }
        let (pp, _rx) = self.pick_in_flight.take().expect("ready check passed");
        let slice = self.pick_readback_buffer.slice(..);
        let (raw_payload, position) = {
            let data = slice.get_mapped_range();
            // Buffer layout (all three textures copied per pick):
            //   0..8    gbuf_material (Rg32Uint):
            //             R = primary_id_lo16 | secondary_id_lo16
            //             G = blend(8) | reserved(8) | color_rgb565(16)
            //   256..260 gbuf_pick (R32Uint):
            //             MAIN voxel march: `gpu_idx` of the hit entity,
            //               or 0xFFFFFFFF on sky miss.
            //             BUILD proc raymarch: primitive NodeId (low 16),
            //               or 0xFFFF on miss.
            //   512..528 gbuf_position (Rgba32Float):
            //             xyz = world position, w = hit distance
            //             (1e10 on miss).
            let mut payload = [0u32; 2];
            if data.len() >= 528 {
                // Both pick kinds read the same 32-bit slot — the shader
                // picked the right meaning. Byte 4..8 (material G) is
                // still copied for potential future material-info-on-
                // pick needs, but no longer carries the object id.
                let pick = u32::from_le_bytes([data[256], data[257], data[258], data[259]]);
                let material_g = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                payload = [pick, material_g];
            }
            let position = if data.len() >= 528 {
                let px = f32::from_le_bytes([data[512], data[513], data[514], data[515]]);
                let py = f32::from_le_bytes([data[516], data[517], data[518], data[519]]);
                let pz = f32::from_le_bytes([data[520], data[521], data[522], data[523]]);
                let hit_dist = f32::from_le_bytes([data[524], data[525], data[526], data[527]]);
                // Shader writes 1e10 for sky-miss; anything larger than
                // a plausible scene extent means "no geometry here."
                if hit_dist < 1.0e9 {
                    Some(glam::Vec3::new(px, py, pz))
                } else {
                    None
                }
            } else {
                None
            };
            (payload, position)
        };
        self.pick_readback_buffer.unmap();
        Some(PickResult {
            viewport: pp.viewport,
            kind: pp.kind,
            raw_payload,
            position,
        })
    }
}

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
fn run_render_thread(
    init: RenderInit,
    inbox: Arc<RenderInbox>,
    cmd_rx: Receiver<RenderCommand>,
    out_tx: Sender<RenderResult>,
    frame_callback: FrameCallback,
) {
    let render_pacing = init.render_pacing;
    let mut state = RenderState::new(init);

    // Bootstrap: wait for the first snapshot. `None` = shutdown
    // signal arrived before any snapshot ever did.
    let first = match inbox.take_blocking() {
        Some(f) => f,
        None => return,
    };
    state.curr_snap_time = std::time::Instant::now();
    state.curr_snap = Some(Arc::new(first));

    // Wall-clock instant of the last *actual* render iteration. We
    // skip backoff iterations (see `2a. GPU backpressure gate`) when
    // computing this — counting them would inflate the panel's
    // "Render FPS" to reflect the 500 µs sleep loop instead of the
    // true GPU-bound production rate. Reset to `None` here so the
    // first real iteration carries no dt (sim falls back to its prior
    // EMA value for that one frame).
    let mut prev_render_start: Option<std::time::Instant> = None;

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

        // 2a. GPU backpressure gate.
        //
        // The composite readback ring is only 3 buffers deep per
        // viewport. When every slot is still waiting for its
        // previously-issued `map_async` callback to fire, encoding
        // another frame would:
        //
        //   1. Drop the readback copy (acquire_write_idx → None)
        //      so this iteration's pixels never reach the editor.
        //   2. Submit a full pass chain anyway, deepening the wgpu
        //      queue behind every still-pending readback.
        //
        // (2) is the real killer — a 450 Hz CPU encode loop can
        // easily pile 70+ frames of GPU work into the queue, which
        // pushes each in-flight readback's `map_async` completion
        // seconds into the future. The visible symptom is what
        // prompted this fix: the engine reported ~170 fps "shipping"
        // but the editor closure saw 80% of callbacks carrying
        // byte-identical pixel content — because `cached_pixels`
        // kept returning the same drained buffer while new readbacks
        // were stuck behind the backlog.
        //
        // The fix is to self-pace CPU encoding to the rate readbacks
        // actually complete at — which is our proxy for true GPU
        // throughput. If MAIN has no idle slot, poll the device,
        // drain any newly-complete maps, back off briefly, and retry.
        // When GPU keeps up (idle slot available) we run full tilt.
        //
        // This preserves the "uncapped render" intent: there's no
        // fixed Hz cap, render runs as fast as the GPU sustains. It
        // just stops submitting work the GPU can't actually execute.
        let _ = state.device.poll(wgpu::PollType::Poll);
        for vp_id in state
            .viewport_renderers
            .keys()
            .copied()
            .collect::<Vec<_>>()
        {
            if let Some(vr) = state.viewport_renderers.get_mut(&vp_id) {
                let w = vr.width;
                let h = vr.height;
                let padded_row = vr.readback_padded_row();
                vr.readback.drain_completed(w, h, padded_row);
            }
        }
        let main_has_slot = state
            .viewport_renderers
            .get(&crate::viewport::ViewportId::MAIN)
            .map(|vr| vr.readback.has_idle_slot())
            .unwrap_or(true);
        if !main_has_slot {
            // Don't spin — sleep long enough to let in-flight
            // map_asyncs complete on whatever cadence the GPU offers,
            // but short enough not to add perceptible latency once
            // the GPU frees a slot. 500 µs is well under a 60 Hz
            // frame budget and well over the cost of a context
            // switch.
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }

        // We are about to render a real frame. Compute the dt back
        // to the last real render — this is what becomes the panel's
        // "Render FPS". Excluding backoff iterations from the dt
        // means the rate reflects honest GPU-bound throughput, not
        // the 2 kHz spin of the backoff sleep.
        let render_dt_ms = prev_render_start
            .map(|p| iter_start.duration_since(p).as_secs_f32() * 1000.0);
        prev_render_start = Some(iter_start);

        // 2b. Drain a completed pick, if any. Non-blocking.
        //
        // Must run AFTER the backoff gate above. `drain_pick` calls
        // `pick_in_flight.take()` so the pick is consumed; if we
        // drained pre-backoff and then hit the `continue`, the
        // PickResult would be silently dropped (no `out_tx.send` runs
        // on backoff iterations) and the click would never reach
        // sim. Picks tolerate a few ms of extra latency; outright
        // losing them does not.
        let pick_result = state.drain_pick();

        // 3. Check for a fresh snapshot — non-blocking. If present,
        //    update the two-snapshot window and refresh the sim_dt
        //    EMA from the observed interval.
        let mut new_snapshot_consumed = false;
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
            new_snapshot_consumed = true;
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

        // 4. Interpolation alpha. At α=0 we'd show prev; at α=1 we
        //    show curr. Clamp to [0,1] so render never extrapolates
        //    past the latest sim state (extrapolation is a correctness
        //    minefield when sim is faster than render's expectation).
        let alpha = (iter_start
            .duration_since(state.curr_snap_time)
            .as_secs_f32()
            / state.sim_dt_estimate.as_secs_f32().max(1e-4))
            .clamp(0.0, 1.0);

        // 5. Build the object list we'll actually upload. If there's
        //    a prev snapshot and α < 1, blend; otherwise use curr
        //    directly (free — avoids per-object work at sim rate).
        let interp_objects: Vec<rkp_render::rkp_gpu_object::RkpGpuObject> =
            match (prev.as_ref(), alpha < 0.999) {
                (Some(p), true) => interpolate_gpu_objects(
                    &p.gpu_objects,
                    &curr.gpu_objects,
                    alpha,
                ),
                _ => curr.gpu_objects.clone(),
            };

        // 6. Render — same pipeline as before; `render_one_frame`
        //    now takes the interpolated objects as an explicit
        //    parameter separate from the snapshot (the snapshot's
        //    own `gpu_objects` field is the canonical curr data).
        //    Pass `new_snapshot_consumed` so the readback path can
        //    skip the editor pixel callback on iterations that just
        //    re-render the same snapshot data — those frames have
        //    no new content to display and just thrash rinch's
        //    surface buffer Mutex.
        let outcome = render_one_frame(
            &mut state,
            &curr,
            &interp_objects,
            new_snapshot_consumed,
            &frame_callback,
        );

        // 7. GPU profiler — drain resolved timings for sim's history.
        let gpu_passes = state.renderer.end_profiler_frame(frame_index);

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
/// slerp) and `inverse_world` is recomputed. Objects without a prev
/// counterpart (newly spawned this sim tick) use `curr` verbatim.
///
/// All non-transform fields (AABBs, octree roots, bone-field offsets,
/// material ids, etc.) come from `curr` — those change on sim edits,
/// not between sim ticks. Skinned entities still carry their bone
/// pose via the separate bone-field buffer; their `world` is usually
/// identity and the lerp is a no-op.
fn interpolate_gpu_objects(
    prev: &[rkp_render::rkp_gpu_object::RkpGpuObject],
    curr: &[rkp_render::rkp_gpu_object::RkpGpuObject],
    alpha: f32,
) -> Vec<rkp_render::rkp_gpu_object::RkpGpuObject> {
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
            // Inverse is a proper 4x4 invert; the only "cheap" path
            // would assume no scale, which we can't. Single invert
            // per moving object per render tick is fine.
            out.inverse_world = glam::Mat4::from_cols_array_2d(&out.world)
                .inverse()
                .to_cols_array_2d();
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

/// Render a single snapshot.
///
/// `frame` is the canonical sim snapshot (lights, environment,
/// cameras, proc raymarch state, etc.). `gpu_objects` is the
/// possibly-interpolated object list to upload — at α=1 or when
/// there's no prev snapshot, it's `frame.gpu_objects.clone()`;
/// otherwise it's the TRS-blended version from
/// [`interpolate_gpu_objects`].
///
/// `new_snapshot_consumed` is true on the iteration that just
/// took a fresh snapshot from the inbox — gates the editor pixel
/// callback. When false (we're re-rendering the same snapshot for
/// interpolation), GPU work still runs but pixels are not shipped
/// to the editor surface (the content didn't change, so shipping
/// would just thrash rinch's `Mutex<RenderSurfaceBuffer>` with no
/// visible benefit).
struct RenderOutcome {
    /// Latest cloud-sun attenuation read from MAIN's volumetric
    /// pass (NaN if MAIN isn't visible).
    cloud_sun_atten_raw: f32,
    /// Wall-clock ms since the previous iteration that successfully
    /// shipped pixels to the editor. `None` when this iteration did
    /// not ship (skipped via `ship_pixels` gate) — sim uses `None`
    /// to hold the previous delivered-FPS EMA sample unchanged.
    delivered_dt_ms: Option<f32>,
}

fn render_one_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_objects: &[rkp_render::rkp_gpu_object::RkpGpuObject],
    new_snapshot_consumed: bool,
    frame_callback: &FrameCallback,
) -> RenderOutcome {
    // 0. Drive the wgpu async runtime so any in-flight async maps can
    //    complete (volumetric sun-atten readbacks, frame readbacks,
    //    pick readbacks).
    let _ = state.device.poll(wgpu::PollType::Poll);

    // 0a. Material palette upload — every frame (cheap; ~1 KB).
    state
        .renderer
        .update_materials(&state.queue, &frame.materials);

    // 0b. Lights upload — sim hands us the full list each tick
    //     (entry 0 = sun, 1..N = scene point/spot lights).
    state.renderer.update_lights(&state.queue, &frame.lights);

    // 0c. Environment-driven bloom + tonemap settings — every frame.
    //     Walk every viewport renderer (each VR owns its own bloom +
    //     tonemap pass; no per-VR override today). Each set_* is one
    //     small queue.write_buffer.
    let env = frame.env_update;
    let vr_ids: Vec<_> = state.viewport_renderers.keys().copied().collect();
    for vr_id in &vr_ids {
        let vr = state
            .viewport_renderers
            .get_mut(vr_id)
            .expect("viewport renderer must exist");
        vr.tone_map.set_exposure(&state.queue, env.exposure);
        vr.bloom
            .set_threshold(&state.queue, env.bloom_threshold, env.bloom_knee);
        vr.bloom_composite
            .set_intensity(&state.queue, env.bloom_intensity);
    }

    // 0d. User-shader integration. Each viewport's shade pass owns
    //     its own pipeline + per-material params buffer. Recompile
    //     when the registry's source hash changes (idempotent). Upload
    //     params alongside the materials buffer; if the buffer grew,
    //     the bind group gets cleared and we rebuild it via
    //     set_shade_data. Uploading on every viewport repeats the
    //     queue.write_buffer (cost: 32 B × num_materials) but keeps
    //     bind-group lifetimes simple.
    for vr_id in &vr_ids {
        let vr = state
            .viewport_renderers
            .get_mut(vr_id)
            .expect("viewport renderer must exist");
        vr.shade.reload_user_shaders(
            &state.device,
            &frame.user_shader_shade_chunk,
            frame.user_shader_source_hash,
        );
        vr.shade.upload_shader_params(
            &state.device,
            &state.queue,
            &frame.shader_params_slots,
        );
        // Re-bind unconditionally — set_shade_data is one bind-group
        // create; cheaper than threading state for "did the buffer
        // grow" through callers, and the materials/lights buffers may
        // have just been swapped above too.
        vr.shade.set_shade_data(
            &state.device,
            &state.renderer.shade_params_buffer,
            &state.renderer.lights_buffer,
            &state.renderer.materials_buffer,
        );
    }

    // 1. Geometry upload — epoch-driven. Robust to snapshot drops:
    //    sim ships scene_mgr's current epoch every frame, so we'll
    //    catch up on the next snapshot if an intermediate one was
    //    dropped by the newest-wins inbox.
    if frame.geometry_epoch > state.last_uploaded_geometry_epoch {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let geo = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &geo);
        // Read-back the epoch *under the same lock* so concurrent
        // mutations (bake worker integrating an artifact mid-frame)
        // don't trick us into thinking we're caught up when we're
        // not. Worst case: we re-upload next frame, which is fine.
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
    }

    // 1.5. Paint-data upload — slice-write the dirty slot range to
    //      leaf_attr_pool + color_pool. Bypasses the full
    //      upload_geometry path (which would re-upload octree +
    //      bricks + face links too — ~45 MB on a 1M-leaf scene for a
    //      stroke that touches ~64 KB of actual data).
    if frame.paint_epoch > state.last_uploaded_paint_epoch {
        let mut sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        if let Some((min_slot, max_slot)) = sm.take_paint_dirty_range() {
            let slot_count = max_slot - min_slot + 1;
            let leaf_attr_bytes_per: u64 =
                std::mem::size_of::<rkp_core::LeafAttr>() as u64;
            let leaf_attr_offset = min_slot as u64 * leaf_attr_bytes_per;
            let color_offset = min_slot as u64 * 4;
            let la_slice = sm.leaf_attr_slice_bytes(min_slot, slot_count);
            if !la_slice.is_empty() {
                state.queue.write_buffer(
                    &state.renderer.scene.leaf_attr_pool_buffer,
                    leaf_attr_offset,
                    la_slice,
                );
            }
            let color_slice = sm.color_slice_bytes(min_slot, slot_count);
            if !color_slice.is_empty() {
                state.queue.write_buffer(
                    &state.renderer.scene.color_pool_buffer,
                    color_offset,
                    color_slice,
                );
            }
        }
        state.last_uploaded_paint_epoch = sm.paint_epoch();
        drop(sm);
    }

    // 1.6. Brush-overlay upload — paint cursor geodesic distances.
    //      MAIN-only (BUILD viewport doesn't show the paint cursor).
    //      queue.write_buffer is cheap (staging-buffer enqueue), so
    //      we do it inside the scene_mgr lock — cheaper than cloning
    //      the full ~4 MB overlay buffer out of the critical section.
    if frame.brush_overlay_epoch > state.last_uploaded_brush_overlay_epoch {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let bytes = sm.brush_overlay_bytes();
        if let Some(main_vr) = state.viewport_renderers.get_mut(&ViewportId::MAIN) {
            main_vr.shade.upload_brush_overlay(&state.device, &state.queue, bytes);
        }
        state.last_uploaded_brush_overlay_epoch = sm.brush_overlay_epoch();
        drop(sm);
    }

    // 1.7. User-shader geometry pass (Phase C). Reserve transient
    //      pool tail, reload pipeline if the shader source changed,
    //      walk regions, dispatch the geom-build pipeline for any
    //      that need a re-bake, and concatenate the resulting
    //      transient `RkpGpuObject`s into `gpu_objects` so the
    //      per-frame upload below ships them alongside persistent
    //      objects. The march/shade passes treat them identically to
    //      bake-built objects — same octree node encoding, same leaf
    //      attr layout — so no shader-side branching is needed.
    let transient_objects = run_user_shader_geom(state, frame);
    let mut combined_objects: Vec<rkp_render::rkp_gpu_object::RkpGpuObject>;
    let gpu_objects_for_upload: &[rkp_render::rkp_gpu_object::RkpGpuObject] =
        if transient_objects.is_empty() {
            gpu_objects
        } else {
            combined_objects = Vec::with_capacity(gpu_objects.len() + transient_objects.len());
            combined_objects.extend_from_slice(gpu_objects);
            combined_objects.extend_from_slice(&transient_objects);
            &combined_objects
        };

    // 1b. Per-frame `RkpGpuObject` upload. `gpu_objects` here may be
    //     interpolated between the last two sim snapshots (see
    //     `interpolate_gpu_objects`), so at high render rates
    //     physics-driven motion is smooth instead of stuttering at
    //     the sim rate.
    state.renderer.upload_frame(
        &state.queue,
        &FrameUpload {
            objects: gpu_objects_for_upload,
            bone_matrices: &frame.bone_matrix_lbs,
            bone_dual_quats: &frame.bone_matrix_dqs,
        },
    );

    // 2. Skin scatter (one batched compute dispatch). Sim folded every
    //    skinned entity into `frame.skin.batch`; we just fire it.
    if let Some(skin) = &frame.skin {
        let mut skin_encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rkp_skin_deform_encoder"),
            });
        let q = state
            .renderer
            .profiler
            .begin_query("skin_deform", &mut skin_encoder);
        state.renderer.prepare_bone_field(
            &state.queue,
            &mut skin_encoder,
            skin.bone_field_bytes,
            skin.bone_field_occ_bytes,
        );
        state
            .renderer
            .scatter_skin_batch(&state.queue, &mut skin_encoder, &skin.batch);
        state.renderer.profiler.end_query(&mut skin_encoder, q);
        state.queue.submit(std::iter::once(skin_encoder.finish()));
    }

    // 3. Per-viewport encode + submit + readback. One submit per VR so
    //    `queue.write_buffer` writes for that VR's per-frame params
    //    (vol/cloud/atmo/god-ray/shade) are correctly paired with the
    //    encoded dispatches reading them.
    let mut pick_issued = false;

    // Drop a freshly-arrived pick request if a previous pick is
    // still in flight on the readback buffer. Encoding a second
    // copy_texture_to_buffer + map_async into a still-mapped buffer
    // causes a validation error at submit and a panic in map_async.
    //
    // This race was rare at 60 Hz (picks resolve in 1-2 sim frames)
    // but very common with `render_pacing: Uncapped`: at 200 Hz a
    // pick takes ~10 render iterations to complete, plenty of time
    // for the user to click again. Dropping the new request is the
    // simplest correct behavior — the user can re-click; a second
    // click 50 ms later is invisibly close to the first as far as
    // pick UX goes.
    let active_pending_pick = if state.pick_in_flight.is_some() {
        None
    } else {
        frame.pending_pick
    };

    // Object count comes from the interpolated list plus any
    // transient user-shader objects appended by `run_user_shader_geom`.
    // Transient objects sit at the tail of `objects_buffer`, indices
    // `gpu_objects.len()..gpu_objects.len()+transient_objects.len()`.
    let transient_count = transient_objects.len() as u32;
    let persistent_count = gpu_objects.len() as u32;
    let object_count = persistent_count + transient_count;
    // Transient object ids the per-VR tile list rebuild splices into
    // every tile. Sim's tile_object_ids only enumerated persistent
    // objects, so without this the march would never visit transient
    // bricks no matter how many were uploaded.
    let transient_indices: Vec<u32> =
        (persistent_count..persistent_count + transient_count).collect();

    for vp in &frame.viewports {
        // Override `prev_vp` (and the parallel `prev_view_proj` field
        // on the volumetric params) with the view_proj we actually
        // rendered last for THIS viewport. Sim bakes its previous
        // tick's view_proj into the snapshot, but with the GPU-
        // backpressure backoff we may have skipped several sim ticks
        // between renders — TAA reprojection (cloud march, octree
        // march, shade) would then sample history with a `prev_vp`
        // that doesn't describe what's actually in the history
        // texture, producing the streak/blur seen on the sky.
        //
        // Both the camera uniform and the volumetric params carry
        // their own copy of the matrix; patch them in lock-step so
        // the cloud-TAA reprojection and the rest of the pipeline
        // agree on the same previous frame.
        let prev_vp_override = state
            .last_rendered_vp
            .get(&vp.id)
            .copied()
            .unwrap_or(vp.camera.view_proj);
        let mut camera = vp.camera;
        camera.prev_vp = prev_vp_override;
        let mut vol_params = vp.vol_params;
        vol_params.prev_view_proj = prev_vp_override;

        let vr = state
            .viewport_renderers
            .get_mut(&vp.id)
            .expect("snapshot referenced an unknown viewport");

        // 3a. Per-VR camera + scene/lights bind group refresh.
        vr.upload_camera(&state.queue, &camera);
        vr.refresh_bindings(&state.device, &state.renderer);

        // 3b. Per-VR per-frame param uploads (vol/cloud/god-ray).
        vr.volumetric.update_params(&state.queue, &vol_params);
        vr.volumetric.update_cloud_params(&state.queue, &vp.cloud_params);
        vr.god_rays.update_params(&state.queue, &vp.god_ray_params);

        // 3c. Per-VR shade params (isolation-aware).
        state
            .renderer
            .update_shade_params(&state.queue, &vp.shade_params);

        // 3d. Bloom-composite intensity (zero in isolation mode).
        vr.bloom_composite
            .set_intensity(&state.queue, vp.bloom_composite_intensity);

        // 3e. BUILD viewport: optionally pin the studio floor under the
        //     previewed entity instead of world origin.
        if let Some(grid) = vp.grid_override {
            vr.grid.update_params(&state.queue, &grid);
        }

        // 3f. Per-viewport encoder.
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rkp viewport"),
            });

        // 3g. Procedural raymarch upload (instructions + outline + ghosts)
        //     when this VR is in raymarch preview mode.
        if let Some(proc) = &vp.proc_raymarch {
            vr.proc_raymarch.upload_instructions(
                &state.device,
                &state.queue,
                &proc.instructions,
            );
            vr.proc_raymarch.set_params(
                &state.queue,
                proc.instructions.len() as u32,
                proc.object_id + 1,
                proc.entity_world,
                proc.aabb_min,
                proc.aabb_max,
            );
            let outline_params = match proc.selected_node {
                Some(n) => rkp_render::proc_outline::OutlineParams::new(
                    n,
                    [1.0, 0.55, 0.15, 1.0],
                ),
                None => rkp_render::proc_outline::OutlineParams::NONE,
            };
            vr.proc_outline.update_params(&state.queue, &outline_params);
            vr.proc_ghost.upload_instructions(
                &state.device,
                &state.queue,
                &proc.ghost_instructions,
            );
            vr.proc_ghost.update_params(
                &state.queue,
                &rkp_render::proc_ghost::GhostParams::new(
                    proc.ghost_instructions.len() as u32,
                    [0.25, 0.7, 1.0, 0.35],
                ),
            );
        }

        // 3h. The big one — full per-VR dispatch chain (atmo, march or
        //     proc_raymarch, shadow, ssao, shade, vol, god_rays, bloom,
        //     bloom_composite, tone_map, composite, grid).
        //
        // When transient user-shader objects exist, splice their
        // indices into every tile's list so the march visits them
        // alongside persistent objects. Sim's tile_object_ids only
        // enumerated persistent objects (sim doesn't see the
        // render-thread-built transient list); cheap to fix here as
        // a per-frame O(tiles × transient_count) rebuild.
        let (effective_tile_offsets, effective_tile_object_ids);
        let (tile_offsets_ref, tile_object_ids_ref): (&[u8], &[u8]) = if transient_count == 0 {
            (&vp.tile_offsets_bytes, &vp.tile_object_ids_bytes)
        } else {
            let (offsets, ids) = splice_transient_into_tile_lists(
                &vp.tile_offsets_bytes,
                &vp.tile_object_ids_bytes,
                &transient_indices,
            );
            effective_tile_offsets = offsets;
            effective_tile_object_ids = ids;
            (
                bytemuck::cast_slice(&effective_tile_offsets),
                bytemuck::cast_slice(&effective_tile_object_ids),
            )
        };
        state.renderer.render_to(
            &mut encoder,
            &state.queue,
            vr,
            object_count,
            frame.shadow_steps,
            vp.shade_params.num_lights,
            frame.lod_enabled,
            frame.surfacenet_enabled,
            tile_offsets_ref,
            tile_object_ids_ref,
            vp.tile_count_x,
            &vp.atmo_frame,
            vp.mode,
            vp.preview_mode,
        );

        // 3i. Pick encode — if there's a pending pick targeted at this
        //     viewport AND no previous pick is still in flight (see
        //     `active_pending_pick`), copy the relevant 1×1 G-buffer
        //     pixels into the readback buffer slots.
        if let Some(pp) = &active_pending_pick {
            if pp.viewport == vp.id && pp.x < vr.width && pp.y < vr.height {
                pick_issued = true;
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.gbuffer.material_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.pick_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 256,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                // Position slot (Rgba32Float, 16 B per texel). The sim
                // reads xyz + hit_distance; drag-drop uses the xyz as
                // the surface snap point and the hit_distance (>1e9 →
                // sky miss) as the "did it hit anything" bit.
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.gbuffer.position_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 512,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        // 3j. Wireframe overlays — gizmo on MAIN (when editor overlays
        //     are enabled) and procedural-node gizmo on BUILD. Sim
        //     pre-built the verts; render just submits.
        if vp.show_editor_overlays && !vp.wireframe_verts.is_empty() {
            let composite_view = &vr.composite_view;
            let vw = vr.width as f32;
            let vh = vr.height as f32;
            vr.wireframe_pass.draw(
                &state.device,
                &state.queue,
                &mut encoder,
                composite_view,
                vp.vp_matrix,
                (0.0, 0.0, vw, vh),
                &vp.wireframe_verts,
            );
        }

        // 3k. Composite readback (frame pixels back to the editor).
        let readback_idx = vr.encode_composite_readback(&mut encoder);
        state.renderer.resolve_profiler_queries(&mut encoder);
        state.queue.submit(std::iter::once(encoder.finish()));

        if let Some(idx) = readback_idx {
            vr.readback.issue_map_async(idx);
        }
    }

    // Stash this frame's un-interpolated view_proj per viewport for
    // next render's `prev_vp` override. See `last_rendered_vp` doc
    // comment on `RenderState` for why this lives render-side now
    // instead of being trusted from the snapshot.
    for vp in &frame.viewports {
        state.last_rendered_vp.insert(vp.id, vp.camera.view_proj);
    }

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
    if pick_issued {
        if let Some(pp) = active_pending_pick {
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
    //    a) `new_snapshot_consumed` — there's no point shipping
    //       pixels for an iteration that just re-rendered the same
    //       sim state. The visual content is identical to whatever
    //       we shipped last time. With Uncapped render at 200 Hz
    //       and 60 Hz sim, this alone drops pixel ships from 200
    //       /sec to 60 /sec — matching display refresh.
    //
    //    b) `MIN_FRAME_CALLBACK_INTERVAL` — soft cap that handles
    //       the edge case where sim itself runs faster than
    //       display refresh (Uncapped sim, very fast scenes).
    //       Without this an Uncapped sim at 600 Hz would still try
    //       to ship 600 frames/sec to the editor and saturate
    //       rinch's surface buffer Mutex.
    //
    //    Together: pixel ship rate = min(sim_rate, display_rate),
    //    which is exactly what the editor surface can usefully
    //    consume.
    let now = std::time::Instant::now();
    let time_ok = now.duration_since(state.last_frame_callback)
        >= MIN_FRAME_CALLBACK_INTERVAL;
    let ship_pixels = new_snapshot_consumed && time_ok;
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
    for vp in &frame.viewports {
        let vr = state
            .viewport_renderers
            .get_mut(&vp.id)
            .expect("viewport renderer must exist");
        let w = vr.width;
        let h = vr.height;
        let padded_row = vr.readback_padded_row();
        vr.readback.drain_completed(w, h, padded_row);
        if ship_pixels {
            if let Some((pixels, cw, ch)) = vr.readback.cached_pixels() {
                frame_callback(vp.id, pixels, cw, ch);
                shipped_any = true;
            }
        }
    }
    if shipped_any {
        delivered_dt_ms = Some(
            now.duration_since(state.last_frame_callback).as_secs_f32() * 1000.0,
        );
        state.last_frame_callback = now;
    }

    RenderOutcome { cloud_sun_atten_raw, delivered_dt_ms }
}

/// Phase C V1.5 — append `transient_indices` to every tile's object
/// list, returning rebuilt `(tile_offsets, tile_object_ids)` arrays.
///
/// Sim's tile_object_ids only enumerate persistent objects; transient
/// ones (built render-thread-side after the snapshot arrives) need to
/// be visible from every tile so the march visits them. With T tiles
/// and N transient objects the cost is O(T × N) per frame; for V1's
/// few-region demos that's negligible (~MB/frame at most).
///
/// Layout: `tile_offsets` is a prefix-sum (length `T + 1`), so each
/// tile `t` has range `[offsets[t]..offsets[t+1])` in `tile_object_ids`.
/// We splice `transient_indices` after each tile's existing range,
/// shifting downstream offsets accordingly.
fn splice_transient_into_tile_lists(
    tile_offsets_bytes: &[u8],
    tile_object_ids_bytes: &[u8],
    transient_indices: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n_tile = if tile_offsets_bytes.is_empty() {
        0
    } else {
        (tile_offsets_bytes.len() / 4).saturating_sub(1)
    };
    if n_tile == 0 || transient_indices.is_empty() {
        // Empty input — return whatever was passed in as u32 vecs.
        let offsets = bytemuck::cast_slice::<u8, u32>(tile_offsets_bytes).to_vec();
        let ids = bytemuck::cast_slice::<u8, u32>(tile_object_ids_bytes).to_vec();
        return (offsets, ids);
    }
    let orig_offsets: &[u32] = bytemuck::cast_slice(tile_offsets_bytes);
    let orig_ids: &[u32] = bytemuck::cast_slice(tile_object_ids_bytes);
    let n_transient = transient_indices.len();
    let mut new_offsets: Vec<u32> = Vec::with_capacity(n_tile + 1);
    let mut new_ids: Vec<u32> = Vec::with_capacity(orig_ids.len() + n_tile * n_transient);

    new_offsets.push(0);
    for t in 0..n_tile {
        let a = orig_offsets[t] as usize;
        let b = orig_offsets[t + 1] as usize;
        new_ids.extend_from_slice(&orig_ids[a..b]);
        new_ids.extend_from_slice(transient_indices);
        new_offsets.push(new_ids.len() as u32);
    }
    (new_offsets, new_ids)
}

/// Phase C — user-shader runtime geometry. Reserves transient pool
/// capacity, rebuilds the geom-build pipeline on shader changes, walks
/// the snapshot's regions, and dispatches one workgroup per region
/// whose cache slot is dirty. Returns the transient `RkpGpuObject`
/// list to concatenate with `gpu_objects` for this frame's
/// `upload_frame` (so the march/shade passes find the new geometry
/// alongside persistent objects).
///
/// Pool reservation per region (V1 single-brick shape):
///   octree_nodes: 1 × `vec2<u32>` = 8 B
///   brick_pool:   `BRICK_CELLS` × u32 = 256 B
///   leaf_attr_pool: `BRICK_CELLS` × `LeafAttr` = 512 B
///
/// We size the reservation for `MAX_REGIONS` and refuse new entries
/// past that (the cache logs and drops the request — sim can lower
/// fidelity if it sees its regions disappearing).
fn run_user_shader_geom(
    state: &mut RenderState,
    frame: &RenderFrame,
) -> Vec<rkp_render::rkp_gpu_object::RkpGpuObject> {
    use rkp_render::user_shader_pass::{
        bricks_per_region, build_internal_nodes, build_region_uniform,
        effective_hash, octree_node_count, resolve_shader_id, RegionUniform,
        BRICK_CELLS,
    };

    // V4 — per-region depth from `@octree_depth` directive. Pool
    // capacity sums per-region bytes so shallow + deep regions can
    // coexist without one's worst case forcing the other's
    // reservation. Memory still scales with `8^depth` per region;
    // the brick proximity gate (this pass) skips over far-from-host
    // bricks at compute-time but doesn't shrink the reservation.
    const MAX_REGIONS: u32 = 256;
    const FACE_EMPTY: u32 = 0xFFFFFFFFu32;
    fn region_octree_bytes(depth: u32) -> u64 { octree_node_count(depth) as u64 * 8 }
    // V5 — sparse brick reservation matching `lookup_or_allocate`.
    // Reserve a quarter of the worst-case brick count per region;
    // workgroups whose brick survives the proximity gate atomic-claim
    // a slot. Sparse-mode shaders (with `@region_thickness > 0`) fit
    // comfortably; ungated shaders may overflow → some bricks render
    // as OCTREE_EMPTY.
    fn bricks_reserved(depth: u32) -> u32 {
        // Match user_shader_pass — full dense reserve to avoid
        // dispatch-order asymmetric grass dropouts.
        bricks_per_region(depth)
    }
    fn region_brick_bytes(depth: u32) -> u64 {
        bricks_reserved(depth) as u64 * BRICK_CELLS as u64 * 4
    }
    fn region_leaf_bytes(depth: u32) -> u64 {
        (bricks_reserved(depth) as u64 * BRICK_CELLS as u64 / 2)
            .max(BRICK_CELLS as u64) * 8
    }
    fn region_face_link_bytes(depth: u32) -> u64 {
        bricks_reserved(depth) as u64 * 6 * 4
    }

    // 1. Pipeline reload — track the shade-side hash; the geom and
    //    shade chunks share the same `source_hash`.
    state.user_shader_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_generate_chunk,
        frame.user_shader_source_hash,
    );

    if frame.user_shader_regions.is_empty() {
        // Nothing to bake — skip even the capacity reservation. Cache
        // entries from previous frames stick around (cheap; capacity
        // for them is already reserved if any were created last
        // frame), so a region appearing again will hit the cache
        // immediately if its inputs match.
        return state.user_shader_cache.build_transient_objects();
    }

    // 2. Reserve transient pool capacity. Sum per-region byte sizes
    //    based on each shader's `@octree_depth`. Cap on region count
    //    (not bytes); a deep shader on many objects can blow memory —
    //    user is responsible for keeping depth + region count sane.
    let need_regions = (frame.user_shader_regions.len() as u32).min(MAX_REGIONS);
    let regions_for_alloc: Vec<&_> = frame
        .user_shader_regions
        .iter()
        .take(need_regions as usize)
        .collect();
    let extra_octree: u64 = regions_for_alloc
        .iter()
        .map(|r| region_octree_bytes(r.octree_depth))
        .sum();
    let extra_brick: u64 = regions_for_alloc
        .iter()
        .map(|r| region_brick_bytes(r.octree_depth))
        .sum();
    let extra_leaf: u64 = regions_for_alloc
        .iter()
        .map(|r| region_leaf_bytes(r.octree_depth))
        .sum();

    // CPU pool sizes — read under the scene_mgr lock so we know where
    // the CPU-managed head ends and the transient tail starts.
    let (cpu_octree_bytes, cpu_brick_bytes, cpu_leaf_attr_bytes, cpu_face_links_bytes) = {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        (
            g.octree_nodes.len() as u64 * 8, // interleaved vec2<u32>
            g.brick_pool.len() as u64,
            g.leaf_attr_pool.len() as u64,
            g.brick_face_links.len() as u64,
        )
    };
    let extra_face_links: u64 = regions_for_alloc
        .iter()
        .map(|r| region_face_link_bytes(r.octree_depth))
        .sum();
    let realloc = state.renderer.scene.ensure_user_shader_capacity(
        &state.device,
        cpu_octree_bytes, extra_octree,
        cpu_brick_bytes, extra_brick,
        cpu_leaf_attr_bytes, extra_leaf,
        cpu_face_links_bytes, extra_face_links,
    );
    if realloc {
        // Reallocation invalidated BOTH the transient writes (tail) AND
        // the CPU-uploaded head — `create_storage` returns a fresh
        // buffer with undefined contents. Re-upload CPU data so the
        // head is valid; flush the cache so the dispatch below
        // re-bakes the tail this frame too.
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        // The re-upload's `ensure_and_write` may have grown buffers
        // again above what we asked for if the CPU side moved
        // forward in the interim — but we only hold the lock briefly
        // and the next frame's epoch check will catch any gap. Track
        // that we did upload so the epoch-driven path doesn't double
        // up next iteration.
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        state.user_shader_cache.flush();
    }

    // Initialize the transient face-links tail to FACE_EMPTY across
    // every region's brick range. Required for march correctness —
    // without this the DDA chain reads zero and jumps into CPU
    // brick_id=0. With per-region depth, total face-link u32 count
    // sums per-region.
    if need_regions > 0 {
        let total_face_link_u32s: u64 = regions_for_alloc
            .iter()
            .map(|r| bricks_reserved(r.octree_depth) as u64 * 6)
            .sum();
        let face_link_data: Vec<u32> = vec![FACE_EMPTY; total_face_link_u32s as usize];
        let face_link_offset_bytes = cpu_face_links_bytes;
        state.queue.write_buffer(
            &state.renderer.scene.brick_face_links_buffer,
            face_link_offset_bytes,
            bytemuck::cast_slice(&face_link_data),
        );
    }

    // 3. Cache flushes — geometry epoch bump invalidates any
    //    host-derived data.
    state.user_shader_cache.reconcile_epoch(frame.geometry_epoch);

    // Configure pool bases — base = CPU bytes / element size, capacity
    // sized to the reservation. Re-set every frame so the cache picks
    // up any growth.
    // octree_nodes: bound as `array<vec2<u32>>` (8 B/elem) — the
    // interleaved upload doubles each CPU u32 to a vec2<u32>, so the
    // transient tail starts at `cpu_octree_bytes / 8` elements in.
    // brick_pool: bound as `array<u32>` (4 B/elem). leaf_attr_pool:
    // bound as `array<LeafAttr>` (8 B/elem). Geom shader brick_offset
    // values are u32-element indices, so divide bytes by 4.
    let octree_base_elems = (cpu_octree_bytes / 8) as u32;
    let brick_base_elems = (cpu_brick_bytes / 4) as u32;
    let leaf_base_elems = (cpu_leaf_attr_bytes / 8) as u32;
    // Pool capacities are total ELEMENT counts at the variable
    // per-region depth — sum each region's slot size.
    let total_octree_elems: u32 = regions_for_alloc
        .iter()
        .map(|r| octree_node_count(r.octree_depth))
        .sum();
    let total_brick_elems: u32 = regions_for_alloc
        .iter()
        .map(|r| bricks_reserved(r.octree_depth) * BRICK_CELLS)
        .sum();
    let total_leaf_elems: u32 = regions_for_alloc
        .iter()
        .map(|r| (region_leaf_bytes(r.octree_depth) / 8) as u32)
        .sum();
    state.user_shader_cache.set_pool_bases(
        octree_base_elems, total_octree_elems,
        brick_base_elems, total_brick_elems,
        leaf_base_elems, total_leaf_elems,
    );

    // 4. Resolve regions, look up cache slots, build per-region
    //    uniforms for the dirty ones. For fresh slots, also write
    //    the perfect-tree internal nodes (levels 0..depth-1) into
    //    the octree buffer — these are deterministic from
    //    `octree_offset` + `depth`, so we compute and queue them
    //    CPU-side once. The GPU dispatch only writes brick-leaf
    //    nodes (level depth) and brick cells.
    let mut uniforms: Vec<RegionUniform> = Vec::with_capacity(need_regions as usize);
    let time_seconds = frame.shade_params_base.time;
    for req in frame.user_shader_regions.iter().take(need_regions as usize) {
        let shader_id = resolve_shader_id(&frame.user_shader_infos, &req.shader_name);
        if shader_id == 0 {
            continue;
        }
        let h = effective_hash(req, frame.user_shader_source_hash, frame.geometry_epoch);
        let slot = match state
            .user_shader_cache
            .lookup_or_allocate(req, h, req.octree_depth)
        {
            Some(s) => s,
            None => continue,
        };
        if slot.fresh && slot.depth > 0 {
            let internals = build_internal_nodes(slot.octree_offset, slot.depth);
            if !internals.is_empty() {
                let byte_offset = slot.octree_offset as u64 * 8;
                state.queue.write_buffer(
                    &state.renderer.scene.octree_nodes_buffer,
                    byte_offset,
                    bytemuck::cast_slice(&internals),
                );
            }
        }
        if !slot.was_dirty {
            continue;
        }
        uniforms.push(build_region_uniform(req, &slot, shader_id, time_seconds));
    }

    // 5. Refresh group-0 binding (scene buffers may have been swapped
    //    out from under us by the upload above).
    state.user_shader_pass.ensure_group0(
        &state.device,
        &state.renderer.scene.octree_nodes_buffer,
        &state.renderer.scene.brick_pool_buffer,
        &state.renderer.scene.leaf_attr_pool_buffer,
        state.renderer.scene.buffers_epoch(),
    );

    if !uniforms.is_empty() {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("user_shader_geom_encoder"),
            });
        let max_region_index = state.user_shader_cache.max_region_index();
        state.user_shader_pass.dispatch_regions(
            &state.device,
            &state.queue,
            &mut encoder,
            &uniforms,
            max_region_index,
        );
        state.queue.submit(Some(encoder.finish()));
    }

    // 6. Build transient `RkpGpuObject`s for every live cache entry —
    //    even ones that hit the cache this frame, since the per-frame
    //    upload rewrites the entire objects buffer (objects from
    //    previous frames are gone unless re-shipped).
    state.user_shader_cache.build_transient_objects()
}

#[cfg(test)]
mod tests {
    use super::splice_transient_into_tile_lists;

    fn u32s(v: &[u32]) -> Vec<u8> { bytemuck::cast_slice(v).to_vec() }

    #[test]
    fn splice_no_transient_passes_through() {
        let offsets = u32s(&[0, 2, 5]);
        let ids = u32s(&[1, 2, 3, 4, 5]);
        let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[]);
        assert_eq!(no, vec![0, 2, 5]);
        assert_eq!(ni, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn splice_appends_transient_to_each_tile() {
        // 2 tiles. Tile 0 has [1, 2], tile 1 has [3, 4, 5]. After
        // splicing transient [99, 100] into both, tile 0 → [1, 2, 99, 100],
        // tile 1 → [3, 4, 5, 99, 100].
        let offsets = u32s(&[0, 2, 5]);
        let ids = u32s(&[1, 2, 3, 4, 5]);
        let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[99, 100]);
        // New offsets: [0, 4, 9].
        assert_eq!(no, vec![0, 4, 9]);
        // Concatenated ids in tile order.
        assert_eq!(ni, vec![1, 2, 99, 100, 3, 4, 5, 99, 100]);
    }

    #[test]
    fn splice_empty_tile_still_gets_transient() {
        // Tile 0 has no objects, but transient should still appear.
        let offsets = u32s(&[0, 0, 1]);
        let ids = u32s(&[42]);
        let (no, ni) = splice_transient_into_tile_lists(&offsets, &ids, &[7]);
        assert_eq!(no, vec![0, 1, 3]);
        assert_eq!(ni, vec![7, 42, 7]);
    }
}

