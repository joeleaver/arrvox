//! Render-thread state types — handles, channels, and the per-thread
//! state struct that the loop / frame / user-shader-tick modules read
//! and mutate.
//!
//! Owns:
//! - [`RenderWorker`] — the sim-side handle returned by [`spawn`].
//! - [`RenderInbox`] — single-slot newest-wins mailbox for `RenderFrame`.
//! - [`FrameCallback`] — typedef for the per-VR pixel callback.
//! - [`RenderState`] — internal state owned exclusively by the render
//!   thread; passed `&mut` into the loop / frame / user-shader-tick
//!   functions in sibling modules.
//!
//! `RenderState`'s fields are `pub(super)` so the render-thread loop
//! and per-frame functions in sibling modules can drive it directly
//! without an inflated method surface. The struct is internal to the
//! render_worker module — the engine sees only the [`RenderWorker`]
//! handle.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::JoinHandle;

use arc_swap::ArcSwapOption;

use crossbeam::channel::{Receiver, Sender};

use arvx_render::{
    arvx_renderer::ArvxRenderer, arvx_scene_manager::ArvxSceneManager, ViewportRenderer,
};

use crate::render_frame::{
    PendingPick, PickResult, RenderCommand, RenderFrame, RenderInit, RenderResult,
};
use crate::viewport::ViewportId;

use super::loop_thread::run_render_thread;

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
/// Phase E3 of `docs/PERF_DEBT.md`: the frame slot itself lives in an
/// `ArcSwapOption<RenderFrame>` so [`Self::try_take`] (the steady-state
/// poll the render thread runs every iteration) is lock-free. The
/// blocking wait used at render-thread bootstrap still needs a wakeup
/// primitive, which costs one tiny `Mutex<()> + Condvar` notify per
/// submit — held only across the `notify_one` call, never across the
/// frame transit itself.
///
/// [`submit`]: RenderInbox::submit
/// [`take_blocking`]: RenderInbox::take_blocking
pub struct RenderInbox {
    /// Newest-wins frame slot. Lock-free swap on submit / try_take.
    slot: ArcSwapOption<RenderFrame>,
    /// Wakeup lock — held only across the brief moment of
    /// `notify_one` (submit) or the bootstrap-`wait` loop
    /// (take_blocking). Never held across the frame swap.
    sleep_mu: Mutex<()>,
    notify: Condvar,
    shutdown: AtomicBool,
}

/// Recover the inner `RenderFrame` from the inbox's sole-owner Arc.
///
/// SPSC invariant: sim is the only submitter, render is the only
/// taker, the inbox holds the only `Arc<RenderFrame>` reference. When
/// render's swap-None returns an `Arc`, the refcount should be 1.
///
/// arc-swap's lock-free read path can in pathological cases keep a
/// transient extra refcount in flight; we busy-spin a few iterations
/// to let it clear before falling back to a clone-from-shared path.
/// In practice the spin loop should exit on the first iteration —
/// this is belt-and-suspenders for the race against `slot.load()`
/// inside `take_blocking`.
#[inline]
fn unwrap_inbox_arc(mut arc: Arc<RenderFrame>) -> RenderFrame {
    for _ in 0..16 {
        match Arc::try_unwrap(arc) {
            Ok(frame) => return frame,
            Err(shared) => {
                arc = shared;
                std::thread::yield_now();
            }
        }
    }
    panic!(
        "RenderInbox: failed to recover sole ownership after spin — \
         someone else is holding an Arc<RenderFrame> reference (this \
         should never happen in SPSC; if you see this, the inbox has \
         been mis-shared)"
    );
}

impl RenderInbox {
    fn new() -> Self {
        Self {
            slot: ArcSwapOption::const_empty(),
            sleep_mu: Mutex::new(()),
            notify: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Place `frame` in the inbox, dropping any previously-unconsumed
    /// frame. O(1); lock-free swap + a one-line condvar notify.
    pub fn submit(&self, frame: RenderFrame) {
        // Atomic swap — old Arc (if any) drops immediately. The render
        // thread is the only other reader/consumer; it either took the
        // previous Arc already or this submit observed the slot empty.
        // Either way refcount returns to 1 at swap time.
        let _previous = self.slot.swap(Some(Arc::new(frame)));
        drop(_previous);
        // Bounce the wakeup mutex so the bootstrap waiter can't miss
        // the notification — without this, a Condvar `wait` that has
        // released its mutex but hasn't parked yet would miss our
        // `notify_one`. Acquiring `sleep_mu` here serializes that
        // boundary: either the waiter is already parked (notify wakes
        // it) or it hasn't reached `wait()` yet (its next loop re-
        // checks the slot and finds the new frame).
        let _g = self.sleep_mu.lock().expect("RenderInbox sleep_mu poisoned");
        self.notify.notify_one();
    }

    /// Block until either a frame arrives (returned as `Some`) or the
    /// shutdown flag is set (returned as `None`). Used once at render-
    /// thread bootstrap — there's nothing to render before the first
    /// sim tick.
    pub(super) fn take_blocking(&self) -> Option<RenderFrame> {
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            if let Some(arc) = self.slot.swap(None) {
                return Some(unwrap_inbox_arc(arc));
            }
            let g = self.sleep_mu.lock().expect("RenderInbox sleep_mu poisoned");
            // Re-check the slot under the mutex — a `submit` that
            // raced our load above will already have notified the
            // condvar before we grabbed `sleep_mu`. Without this re-
            // check the wait would miss that notify and park forever.
            if self.slot.load().is_some() || self.shutdown.load(Ordering::Acquire) {
                continue;
            }
            let _woken = self.notify.wait(g).expect("RenderInbox cond wait poisoned");
            // _woken drops at end of loop iteration; next swap runs
            // lock-free, so the brief mutex hold here covers only the
            // park/wakeup boundary.
            drop(_woken);
        }
    }

    /// Non-blocking take. Returns `None` if no new frame has arrived
    /// since the last call. Used in the steady-state render loop
    /// where render has its own clock and re-renders the current
    /// snapshot (interpolated) when no newer one is waiting.
    ///
    /// Phase E3: lock-free — one `ArcSwap::swap`, no Mutex.
    pub(super) fn try_take(&self) -> Option<RenderFrame> {
        let arc = self.slot.swap(None)?;
        Some(unwrap_inbox_arc(arc))
    }

    /// `true` once [`Self::shutdown`] has been signalled.
    pub(super) fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Signal the render thread to exit at its next inbox check.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let _g = self.sleep_mu.lock().expect("RenderInbox sleep_mu poisoned");
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
    /// builds the [`ArvxRenderer`] + per-viewport renderers there, then
    /// enters its render loop. `frame_callback` is invoked once per
    /// visible viewport per produced frame, on the render thread.
    pub fn spawn(init: RenderInit, frame_callback: FrameCallback) -> Self {
        let inbox = Arc::new(RenderInbox::new());
        let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded::<RenderCommand>();
        let (out_tx, out_rx) = crossbeam::channel::unbounded::<RenderResult>();

        let inbox_for_thread = inbox.clone();
        let handle = std::thread::Builder::new()
            .name("arvx-render".to_string())
            .spawn(move || {
                run_render_thread(init, inbox_for_thread, cmd_rx, out_tx, frame_callback);
            })
            .expect("spawn arvx-render thread");

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
pub(super) struct RenderState {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) renderer: ArvxRenderer,
    pub(super) viewport_renderers: std::collections::HashMap<ViewportId, ViewportRenderer>,
    pub(super) scene_mgr: Arc<Mutex<ArvxSceneManager>>,

    /// Per-material cache of pipelines + buffers + bind groups for
    /// the V1 mesh-path. Created on first frame a mesh-path
    /// material is painted; rebuilt on shader source-hash change.
    pub(super) mesh_user_shader_cache: std::collections::HashMap<
        u16,
        super::user_shader_mesh_tick::MeshUserShaderMaterialState,
    >,
    /// Per-frame draw set for the V1 mesh-path. Repopulated each
    /// frame by `tick_user_shader_mesh`; the renderer consumes
    /// during the per-VR encode phase.
    pub(super) user_shader_mesh_draws:
        Vec<arvx_render::user_shader_mesh_pass::UserShaderMeshDraw>,
    /// `Arc` handle to the painted-anchor map we last uploaded to
    /// the mesh-path GPU buffers. Sim swaps the inner `Arc` only on
    /// paint/geometry/param-epoch rebuild, so an `Arc::ptr_eq` check
    /// against this handle lets us skip the per-material upload +
    /// compute-trio dispatch on idle frames (steady state). Without
    /// this gate, every frame paid ~5 dispatches × N materials of
    /// CPU encoding for work whose output was already on the GPU.
    pub(super) last_uploaded_painted_anchors: Option<
        std::sync::Arc<
            std::collections::HashMap<
                u16,
                Vec<arvx_render::user_shader_mesh_pass::AnchorRecord>,
            >,
        >,
    >,

    /// Phase 7 — TLAS over instance AABBs. Session 2 ships the host
    /// CPU builder + GPU upload (called per frame from
    /// `compose_render_one_frame`); Session 3 adds user-shader
    /// instances; Session 4 plumbs into shadow trace. Until Session 4
    /// the GPU buffers are written but unread.
    pub(super) tlas_pass: arvx_render::tlas_pass::TlasPass,
    /// Phase 7c — GPU-built TLAS pipeline (assembly + Morton + radix
    /// + Karras + AABB propagation). Replaces the CPU median-split
    /// builder in `tlas_pass.rs`. Writes its final tlas_nodes /
    /// tlas_leaves into `tlas_pass`'s buffers (which the shadow
    /// trace already binds).
    pub(super) tlas_build_pass: arvx_render::tlas_build_pass::TlasBuildPass,

    /// Pick readback target. 1×1 region of the gbuf_material at offset
    /// 0, 1×1 region of the gbuf_pick at offset 256 — both 256-byte
    /// aligned per wgpu's copy alignment rules.
    pub(super) pick_readback_buffer: wgpu::Buffer,

    /// In-flight pick — set when a pick was encoded last frame and we
    /// kicked off `map_async` post-submit. Drained at the top of each
    /// frame; if ready, render returns the raw payload back to sim
    /// (which owns the gpu_to_entity mapping for the final resolve).
    pub(super) pick_in_flight: Option<(PendingPick, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,

    /// Most recent snapshot; the source of truth for non-interpolated
    /// fields (lights, environment, cameras, proc_raymarch, etc.).
    /// `Arc` so we can hand a cheap reference to `render_one_frame`
    /// without aliasing the rest of `RenderState` from the borrow
    /// checker's point of view.
    pub(super) curr_snap: Option<Arc<RenderFrame>>,
    /// Wall-clock instant at which `curr_snap` was received. Used to
    /// compute the interpolation alpha for the active render tick.
    pub(super) curr_snap_time: std::time::Instant,
    /// Snapshot immediately before `curr_snap`, kept for world-matrix
    /// interpolation. `None` while we've only seen the first snapshot
    /// (render proceeds without interpolation in that case).
    pub(super) prev_snap: Option<Arc<RenderFrame>>,
    /// EMA of time between snapshot arrivals. Used as the denominator
    /// when converting wall-clock time since `curr_snap_time` to
    /// interpolation alpha. Starts at 16.67 ms (60 Hz) so the first
    /// few frames have a sane estimate before the EMA has converged.
    pub(super) sim_dt_estimate: std::time::Duration,
    /// Last `scene_mgr.geometry_epoch()` value we successfully
    /// uploaded to the GPU. When a snapshot arrives with a higher
    /// epoch, render takes the scene_mgr lock and re-uploads.
    /// Robust to snapshot drops by design (sim ships epoch every
    /// frame, not a one-shot dirty bit).
    pub(super) last_uploaded_geometry_epoch: u64,

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
    pub(super) last_rendered_vp: std::collections::HashMap<ViewportId, [[f32; 4]; 4]>,

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
    pub(super) last_frame_callback: std::time::Instant,
}

/// Minimum wall-clock between two `frame_callback` invocations. ~120 Hz —
/// generous enough for high-refresh editor surfaces while still keeping
/// the surface buffer Mutex out of the lock-saturated regime.
pub(super) const MIN_FRAME_CALLBACK_INTERVAL: std::time::Duration =
    std::time::Duration::from_micros(8_300);

impl RenderState {
    pub(super) fn new(init: RenderInit) -> Self {
        let device = init.device;
        let queue = init.queue;
        let mut renderer = ArvxRenderer::new(&device, &queue, init.initial_width, init.initial_height);

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

        // Phase 7 Session 1 — TLAS foundation.
        let tlas_pass = arvx_render::tlas_pass::TlasPass::new(&device);
        // Phase 7c — GPU TLAS build pipeline.
        let tlas_build_pass = arvx_render::tlas_build_pass::TlasBuildPass::new(&device);

        Self {
            device,
            queue,
            renderer,
            viewport_renderers,
            scene_mgr: init.scene_mgr,
            mesh_user_shader_cache: std::collections::HashMap::new(),
            user_shader_mesh_draws: Vec::new(),
            last_uploaded_painted_anchors: None,
            tlas_pass,
            tlas_build_pass,
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
    pub(super) fn apply_command(&mut self, cmd: RenderCommand) {
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
            RenderCommand::UploadProxyMesh {
                handle_raw,
                vertices,
                indices,
                cluster: _,
            } => {
                // Proxy meshes go into the dedicated proxy slab —
                // `ProxyVertex` layout + single direct indexed draw,
                // no LOD select / cluster table needed.
                self.renderer
                    .upload_proxy_mesh_for_asset(handle_raw, &vertices, &indices);
            }
            RenderCommand::ReleaseProxyMesh { handle_raw } => {
                self.renderer.release_proxy_mesh_for_asset(handle_raw);
            }
            RenderCommand::Shutdown => {
                // Handled by the outer loop's poll of `is_shutdown`.
            }
        }
    }

    /// Drain the in-flight pick if its async map has completed.
    /// Returns the decoded payload to ship back to sim, or `None` if
    /// nothing is ready (or no pick is in flight).
    pub(super) fn drain_pick(&mut self) -> Option<PickResult> {
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
