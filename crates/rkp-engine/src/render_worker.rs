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

    /// Option B (voxel sprite instancing) — scene-wide pieces. The
    /// per-viewport march + composite live in [`ViewportRenderer`]
    /// (Stage 6c-2). All four pieces here are constructed at startup
    /// and ticked per-frame even when no instance shader is registered;
    /// each cache's `begin_frame` / `evict_untouched` is a cheap no-op
    /// when no requests come in.
    ///
    /// `dead_code` allowance covers the bake/scatter pass + pool
    /// buffer that Stage 6c-3.5 will start dispatching against; the
    /// caches themselves ARE read (begin_frame / evict_untouched in
    /// `tick_instance_pipeline`).
    #[allow(dead_code)]
    instance_proto_pass: rkp_render::user_shader_proto_pass::PrototypeBakePass,
    instance_proto_cache: rkp_render::user_shader_proto_pass::PrototypeCache,
    #[allow(dead_code)]
    instance_emit_pass: rkp_render::user_shader_emit_pass::EmitPass,
    instance_region_cache: rkp_render::user_shader_emit_pass::InstanceRegionCache,

    /// Phase 6 Session 3 — tile-cull pipeline. Four passes that turn
    /// the per-instance state in `instance_pool` into per-tile
    /// `UserShaderTileEntry` lists the host march can iterate. Pure
    /// side-effect during Session 3d (March doesn't yet read the
    /// produced lists — Session 4 wires that consumer).
    #[allow(dead_code)]
    instance_tile_cull_pass: rkp_render::user_shader_tile_cull_pass::TileCullPass,
    #[allow(dead_code)]
    instance_tile_count_pass: rkp_render::user_shader_tile_count_pass::TileCountPass,
    #[allow(dead_code)]
    instance_tile_prefix_pass: rkp_render::user_shader_tile_prefix_pass::TilePrefixPass,
    #[allow(dead_code)]
    instance_tile_scatter_pass: rkp_render::user_shader_tile_scatter_pass::TileScatterPass,
    /// Per-frame scratch buffer holding one
    /// `InstanceTileCullEntry` (48 B) per reserved instance slot across
    /// all regions. The AABB pass writes; count + scatter read.
    /// Doesn't carry state across frames — overwritten in place every
    /// frame, sized by the engine to current scratch totals.
    instance_tile_cull_scratch_buffer: wgpu::Buffer,
    /// Capacity of `instance_tile_cull_scratch_buffer` in entries.
    instance_tile_cull_scratch_capacity_entries: u32,
    /// Phase 6 Session 3d — shared uniform buffer for the per-VR
    /// `TileCullViewportUniform` (96 B). Written before each VR's
    /// count/scatter dispatch via `queue.write_buffer`; wgpu serializes
    /// against the dispatches behind it on the same queue.
    instance_tile_view_uniform_buffer: wgpu::Buffer,
    /// Phase 6 Session 3d — shared uniform buffer for the per-VR
    /// `PrefixUniform` (16 B). Written before each VR's prefix-sum
    /// dispatch; same lifecycle as `instance_tile_view_uniform_buffer`.
    instance_tile_prefix_uniform_buffer: wgpu::Buffer,

    /// Phase 7 — TLAS over instance AABBs. Session 2 ships the host
    /// CPU builder + GPU upload (called per frame from
    /// `compose_render_one_frame`); Session 3 adds user-shader
    /// instances; Session 4 plumbs into shadow trace. Until Session 4
    /// the GPU buffers are written but unread.
    tlas_pass: rkp_render::tlas_pass::TlasPass,

    /// Stage 6c-3 — global `array<u32>` storage buffer holding all
    /// scattered instance bytes. Each region's slice is bucket-allocated
    /// inside [`InstanceRegionCache`] (which thinks in u32 units rooted
    /// Cached u32 capacity of the scene's `instance_pool_buffer`.
    /// Mirrors what was passed to `InstanceRegionCache::with_capacity`.
    /// The buffer itself lives on `RkpScene::instance_pool_buffer`
    /// (Phase 4c — bound at scene group binding(14) so the host march
    /// can read it for shader-asset paths).
    #[allow(dead_code)]
    instance_pool_capacity_u32: u32,

    /// Per-frame flat `array<PaintedLeaf>` storage buffer. Holds every
    /// region's painted leaves concatenated end-to-end; the emit
    /// shader's region uniform carries `leaf_offset`/`leaf_count` to
    /// index into this. Grown to the high-water of total leaves.
    instance_leaves_buffer: wgpu::Buffer,
    /// Capacity of `instance_leaves_buffer` in `PaintedLeaf` entries
    /// (32 B each).
    instance_leaves_capacity_entries: u32,

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

// Option B proto-cache pool capacities — Phase 4 sizes. Must match the
// `PROTO_TAIL_*_BYTES` reservation `tick_instance_pipeline` requests in
// the host scene buffers. The cache sub-allocates octree slots within
// `pool_octree_capacity`, and the bake's overflow check uses these to
// gate out-of-bounds writes (via the global brick / leaf-attr cursors).
//
// Smaller than the pre-Phase-4 dedicated proto-buffer caps (32 M /
// 256 K / 4 M) on purpose: the proto tail lives inside the host pool
// alongside Phase C's much larger transient tail, so the proto budget
// has to stay tight to keep the total under `max_buffer_size`. Authors
// needing more proto headroom should reduce paint area or cell_size,
// not these caps.
const INSTANCE_PROTO_OCTREE_CAPACITY_U32: u32 =
    (rkp_render::user_shader_proto_pass::PROTO_TAIL_OCTREE_BYTES / 8) as u32; // 2 M nodes
const INSTANCE_PROTO_BRICK_CAPACITY_BRICKS: u32 =
    (rkp_render::user_shader_proto_pass::PROTO_TAIL_BRICK_BYTES / 256) as u32; // 64 K bricks
const INSTANCE_PROTO_LEAF_ATTR_CAPACITY_U32: u32 =
    (rkp_render::user_shader_proto_pass::PROTO_TAIL_LEAF_ATTR_BYTES / 8) as u32; // 1 M slots

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

        // Option B instance-pipeline scene-wide pieces. Stages 6c-1/2/3
        // wire them into RenderState; Stages 6c-3.5b/c add the per-frame
        // dispatch + per-viewport march/composite encoding.
        let instance_proto_pass =
            rkp_render::user_shader_proto_pass::PrototypeBakePass::new(&device);
        // V1 proto-cache pool capacities — much smaller than the library
        // defaults (PROTO_OCTREE_POOL_CAPACITY = 1.3M / etc., which would
        // demand ~530 MB of additional buffer space on top of the
        // user-shader tail). At these caps a project can register up to
        // ~16 small instance shaders at depth 2-3 — plenty for early
        // development. Stage 6e (perf) raises if a real workload bumps
        // the high water. See module-level consts above.
        let instance_proto_cache =
            rkp_render::user_shader_proto_pass::PrototypeCache::with_capacities(
                INSTANCE_PROTO_OCTREE_CAPACITY_U32,
                INSTANCE_PROTO_BRICK_CAPACITY_BRICKS,
                INSTANCE_PROTO_LEAF_ATTR_CAPACITY_U32,
            );
        let instance_emit_pass =
            rkp_render::user_shader_emit_pass::EmitPass::new(&device);
        // Phase 6 Session 3d — tile-cull GPU pipeline. Four compute
        // passes; bake/emit feed `instance_pool`, this pipeline turns
        // it into per-tile lists for the host march. Constructed
        // unconditionally so the pipelines exist when the first
        // user-shader region appears; they sit idle until then.
        let instance_tile_cull_pass =
            rkp_render::user_shader_tile_cull_pass::TileCullPass::new(&device);
        let instance_tile_count_pass =
            rkp_render::user_shader_tile_count_pass::TileCountPass::new(&device);
        let instance_tile_prefix_pass =
            rkp_render::user_shader_tile_prefix_pass::TilePrefixPass::new(&device);
        let instance_tile_scatter_pass =
            rkp_render::user_shader_tile_scatter_pass::TileScatterPass::new(&device);
        // Initial scratch buffer — single-entry placeholder. Grown per
        // frame by `tick_instance_pipeline` when scatter totals exceed
        // capacity.
        const INSTANCE_TILE_CULL_INITIAL_ENTRIES: u32 = 1;
        let instance_tile_cull_scratch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inst tile_cull_scratch"),
            size: (INSTANCE_TILE_CULL_INITIAL_ENTRIES as u64) * 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_tile_view_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inst tile_view_uniform"),
            size: std::mem::size_of::<rkp_render::user_shader_tile_count_pass::TileCullViewportUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_tile_prefix_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inst tile_prefix_uniform"),
            size: std::mem::size_of::<rkp_render::user_shader_tile_prefix_pass::PrefixUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Phase 7 Session 1 — TLAS foundation. Empty buffers + no
        // builder yet; Sessions 2-4 plumb in the actual BVH.
        let tlas_pass = rkp_render::tlas_pass::TlasPass::new(&device);

        // The instance_pool_buffer lives on RkpScene now (bound at
        // scene group binding(14) so the host march can read it for
        // shader-asset paths). Capacity is set in rkp_scene.rs via
        // INSTANCE_POOL_CAPACITY_U32.
        let mut instance_region_cache =
            rkp_render::user_shader_emit_pass::InstanceRegionCache::with_capacity(
                rkp_render::rkp_scene::INSTANCE_POOL_CAPACITY_U32,
            );
        // Pool base = 0 — the cache's `instance_block_offset` values
        // are absolute u32 indices into `instance_pool_buffer`.
        instance_region_cache.set_pool_base(0);

        // Per-frame flat `array<PaintedLeaf>` for the emit shader.
        // Sized to a single 32 B entry at startup; grown by
        // `tick_instance_pipeline` when paint accumulates.
        const INSTANCE_LEAVES_INITIAL_ENTRIES: u32 = 1;
        let instance_leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inst leaves"),
            size: (INSTANCE_LEAVES_INITIAL_ENTRIES as u64) * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            renderer,
            viewport_renderers,
            scene_mgr: init.scene_mgr,
            user_shader_pass,
            user_shader_cache,
            instance_proto_pass,
            instance_proto_cache,
            instance_emit_pass,
            instance_region_cache,
            instance_tile_cull_pass,
            instance_tile_count_pass,
            instance_tile_prefix_pass,
            instance_tile_scatter_pass,
            instance_tile_cull_scratch_buffer,
            instance_tile_cull_scratch_capacity_entries: INSTANCE_TILE_CULL_INITIAL_ENTRIES,
            instance_tile_view_uniform_buffer,
            instance_tile_prefix_uniform_buffer,
            tlas_pass,
            instance_pool_capacity_u32: rkp_render::rkp_scene::INSTANCE_POOL_CAPACITY_U32,
            instance_leaves_buffer,
            instance_leaves_capacity_entries: INSTANCE_LEAVES_INITIAL_ENTRIES,
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

        // 5. Build the instance list we'll actually upload. If there's
        //    a prev snapshot and α < 1, blend; otherwise use curr
        //    directly (free — avoids per-object work at sim rate).
        //    Assets don't interpolate — they're pose-static for a frame.
        let interp_instances: Vec<rkp_render::rkp_gpu_object::RkpGpuInstance> =
            match (prev.as_ref(), alpha < 0.999) {
                (Some(p), true) => interpolate_instances(
                    &p.gpu_instances,
                    &curr.gpu_instances,
                    alpha,
                ),
                _ => curr.gpu_instances.clone(),
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
            &interp_instances,
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
    prev: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    curr: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    alpha: f32,
) -> Vec<rkp_render::rkp_gpu_object::RkpGpuInstance> {
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
    gpu_instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
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
        // Phase 4c — host march + shadow trace splice the user-shader
        // inst_to_local + inst_aabb chunks the same way Option B's
        // instance_march does. Hash gate inside each `reload` makes
        // the no-change frame a no-op.
        vr.march.reload_user_shaders(
            &state.device,
            &frame.user_shader_inst_to_local_chunk,
            &frame.user_shader_inst_aabb_chunk,
            frame.user_shader_source_hash,
        );
        vr.shadow_trace.reload_user_shaders(
            &state.device,
            &frame.user_shader_inst_to_local_chunk,
            &frame.user_shader_inst_aabb_chunk,
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

    // 1.5. Phase 3 — paint mutations land in per-instance overlays
    //      (`paint_overlays` on EngineState), shipped each tick as
    //      `frame.gpu_instance_overlays`. The upload happens
    //      unconditionally inside `upload_frame` below; no slot-range
    //      slice-upload of `leaf_attr_pool`/`color_pool` is needed.
    //      `frame.paint_epoch` is informational only.

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

    // 1.7. Option B (instance pipeline) per-frame tick. Stage 6c-3.5c
    //      reserves the proto sub-pool past the user-shader-cache tail,
    //      walks `instance_region_requests`, dispatches bake/scatter,
    //      and uploads TileIndex + ProtoLookup. **Runs BEFORE
    //      run_user_shader_geom** so the proto tail's buffer reservation
    //      stacks correctly: tick reserves the full
    //      `cpu + user_shader_max + proto_max` envelope when there's
    //      any instance work, then run_user_shader_geom's own reservation
    //      is a no-op (buffer already big enough).
    let inst_result = tick_instance_pipeline(state, frame);

    // 1.7b. User-shader geometry pass (Phase C). Reserve transient
    //       pool tail, reload pipeline if the shader source changed,
    //       walk regions, dispatch the geom-build pipeline for any
    //       that need a re-bake, and concatenate the resulting
    //       transient (asset, instance) pairs onto the persistent
    //       lists so the per-frame upload below ships them alongside.
    //       Each transient region is its own asset slot — assigned
    //       via `asset_id_base` so it points into the correct slot in
    //       the combined assets vec. The march/shade passes treat
    //       transients identically to bake-built objects — same
    //       octree node encoding, same leaf attr layout.
    //
    //       Phase 4 — user-shader instance assets/instances from
    //       `tick_instance_pipeline` go BETWEEN the host's persistent
    //       set and Phase C's transient set. Phase C's `asset_id_base`
    //       shifts up to account for them so its per-instance
    //       `asset_id` references stay correct after splicing.
    let asset_id_base =
        frame.gpu_assets.len() as u32 + inst_result.user_shader_assets.len() as u32;
    let (transient_assets, transient_instances) =
        run_user_shader_geom(state, frame, asset_id_base);

    let mut combined_assets: Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>;
    let mut combined_instances: Vec<rkp_render::rkp_gpu_object::RkpGpuInstance>;
    // Phase 6 Session 4d — `inst_result.user_shader_instances` is gone.
    // User-shader instance work flows through the GPU tile-cull pipeline
    // (host march iterates `us_tile_entries[]` per tile). Only the
    // user-shader ASSETS still need to be in the combined assets vec
    // — each tile entry's `asset_id` indexes them.
    let need_combine = !inst_result.user_shader_assets.is_empty()
        || !transient_instances.is_empty();
    let (assets_for_upload, instances_for_upload): (
        &[rkp_render::rkp_gpu_object::RkpGpuAsset],
        &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    ) = if !need_combine {
        (frame.gpu_assets.as_slice(), gpu_instances)
    } else {
        combined_assets = Vec::with_capacity(
            frame.gpu_assets.len()
                + inst_result.user_shader_assets.len()
                + transient_assets.len(),
        );
        combined_assets.extend_from_slice(&frame.gpu_assets);
        combined_assets.extend_from_slice(&inst_result.user_shader_assets);
        combined_assets.extend_from_slice(&transient_assets);
        combined_instances = Vec::with_capacity(
            gpu_instances.len() + transient_instances.len(),
        );
        combined_instances.extend_from_slice(gpu_instances);
        combined_instances.extend_from_slice(&transient_instances);
        (combined_assets.as_slice(), combined_instances.as_slice())
    };

    // 1b. Per-frame upload. `gpu_instances` here may be interpolated
    //     between the last two sim snapshots (see `interpolate_instances`),
    //     so at high render rates physics-driven motion is smooth
    //     instead of stuttering at the sim rate. Assets are pose-static
    //     within a frame.
    // 1b'. Phase 7 Session 2 — TLAS build over host instances. User-
    //      shader instances aren't in `instances_for_upload` (they
    //      flow through Phase 6's tile-cull pipeline), so this frame's
    //      TLAS only contains the host BVH. Session 3 will add the
    //      user-shader path; Session 4 plumbs the buffers into shadow
    //      trace. Today nothing reads them — pure side-effect.
    state.tlas_pass.build_host_tlas(
        &state.device,
        &state.queue,
        assets_for_upload,
        instances_for_upload,
    );
    let overlay_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_overlays);
    state.renderer.upload_frame(
        &state.queue,
        &FrameUpload {
            assets: assets_for_upload,
            instances: instances_for_upload,
            bone_matrices: &frame.bone_matrix_lbs,
            bone_dual_quats: &frame.bone_matrix_dqs,
            instance_overlays: overlay_bytes,
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

    // Phase 6 Session 4d — user-shader instances no longer appear in
    // the host instances buffer. Layout is now:
    //   [persistent | transient]
    // and the host march iterates `us_tile_entries[]` for user-shader
    // work (Sessions 1–3 + 4b). The per-VR `compute_screen_aabbs` /
    // `build_tile_lists` for user-shader instances + the
    // `user_shader_tile_lists_per_vp` Vec are gone.
    let transient_count = transient_instances.len() as u32;
    let persistent_count = gpu_instances.len() as u32;
    let object_count = persistent_count + transient_count;
    let transient_indices: Vec<u32> =
        (persistent_count..persistent_count + transient_count).collect();

    for (vp_idx, vp) in frame.viewports.iter().enumerate() {
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

        // 3f.5. Phase 6 Session 3d — per-VR user-shader tile-cull
        //       dispatch chain (count → prefix → scatter). Side-effect
        //       only during Session 3d: writes per-tile entry lists
        //       into `vr.march.us_tile_offsets/entries`, but the host
        //       march doesn't yet iterate them (Session 4). Skipped
        //       when no user-shader instances ran this frame.
        if inst_result.tile_cull_scratch_count > 0 {
            // Disjoint borrow of state's tile-cull fields so they
            // don't conflict with the outer `vr` (which holds a
            // &mut borrow on `state.viewport_renderers`).
            let args = UsTileCullArgs {
                device: &state.device,
                queue: &state.queue,
                renderer: &mut state.renderer,
                scratch_buffer: &state.instance_tile_cull_scratch_buffer,
                view_uniform_buffer: &state.instance_tile_view_uniform_buffer,
                prefix_uniform_buffer: &state.instance_tile_prefix_uniform_buffer,
                tile_count_pass: &state.instance_tile_count_pass,
                tile_prefix_pass: &state.instance_tile_prefix_pass,
                tile_scatter_pass: &state.instance_tile_scatter_pass,
            };
            dispatch_us_tile_cull_inner(args, vr, &mut encoder, &camera,
                vp.width, vp.height, inst_result.tile_cull_scratch_count);
        }

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
        // Merge per-tile object lists across two sources:
        //   - sim's persistent objects (`vp.tile_*_bytes`, already culled)
        //   - Phase C transient indices (broadcast to every tile;
        //     small N, mostly used for whole-entity user-shader regions)
        // No-op pass-through when transients are empty. User-shader
        // instances flow through the GPU tile-cull pipeline now and
        // don't go through this CPU merge (Phase 6 Session 4d).
        let (effective_tile_offsets, effective_tile_object_ids);
        let need_merge = !transient_indices.is_empty();
        let (tile_offsets_ref, tile_object_ids_ref): (&[u8], &[u8]) = if !need_merge {
            (&vp.tile_offsets_bytes, &vp.tile_object_ids_bytes)
        } else {
            let (offsets, ids) = merge_tile_lists(
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

/// Phase 4 — merge sim's per-tile object lists with two render-side
/// sources: a properly-culled per-tile user-shader-instance list (one
/// entry per (tile, instance) where the instance's screen-AABB
/// overlaps the tile), and a Phase C broadcast list (every tile gets
/// every transient).
///
/// For each tile, output is `[sim_persistent | user_shader_in_tile |
/// transient_broadcast]`. All three index spaces are disjoint
/// (persistent < persistent_count ≤ user-shader < persistent+
/// user_shader ≤ transient < object_count), so the march can dispatch
/// any of them without aliasing.
fn merge_tile_lists(
    sim_offsets_bytes: &[u8],
    sim_ids_bytes: &[u8],
    transient_broadcast: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n_tile = if sim_offsets_bytes.is_empty() {
        0
    } else {
        (sim_offsets_bytes.len() / 4).saturating_sub(1)
    };
    if n_tile == 0 {
        return (
            bytemuck::cast_slice::<u8, u32>(sim_offsets_bytes).to_vec(),
            bytemuck::cast_slice::<u8, u32>(sim_ids_bytes).to_vec(),
        );
    }
    let sim_offsets: &[u32] = bytemuck::cast_slice(sim_offsets_bytes);
    let sim_ids: &[u32] = bytemuck::cast_slice(sim_ids_bytes);

    let mut new_offsets: Vec<u32> = Vec::with_capacity(n_tile + 1);
    let mut new_ids: Vec<u32> =
        Vec::with_capacity(sim_ids.len() + n_tile * transient_broadcast.len());
    new_offsets.push(0);
    for t in 0..n_tile {
        let sa = sim_offsets[t] as usize;
        let sb = sim_offsets[t + 1] as usize;
        new_ids.extend_from_slice(&sim_ids[sa..sb]);
        new_ids.extend_from_slice(transient_broadcast);
        new_offsets.push(new_ids.len() as u32);
    }
    (new_offsets, new_ids)
}

/// Stage 6c-3.5c — per-frame tick for the Option B instance pipeline.
///
/// Sequence (executed every frame, runs BEFORE `run_user_shader_geom`
/// so the proto-pool buffer reservation stacks cleanly with the
/// user-shader-cache reservation):
///
///   1. Reload bake + emit pipelines (idempotent on source-hash match).
///   2. `begin_frame` both caches.
///   3. Early-out if no instance shaders are registered AND the request
///      list is empty.
///   4. Snapshot `cpu_*_bytes` from scene_mgr.
///   5. Reserve `cpu + user_shader_max + proto_max` on the scene
///      buffers (one ensure call per pool). Re-upload geometry +
///      face-links init on realloc — same shape as `run_user_shader_geom`.
///   6. Configure `instance_proto_cache.set_pool_bases(...)` with
///      offsets pointing AFTER the user-shader tail.
///   7. Walk requests:
///      - `instance_proto_cache.lookup_or_allocate(...)` — queue bake
///        when dirty.
///      - `instance_region_cache.lookup_or_allocate(...)` — V1 always
///        queues scatter (the march reads `instance_alloc[region_index]`
///        per frame; skipping non-dirty regions while their region_index
///        rotates would corrupt the count).
///   8. Encode bake + scatter into ONE local encoder + queue.submit.
///      Submit-ordering ensures these complete before the per-VR
///      encoders' march reads from the same buffers.
///   9. `evict_untouched` both caches.
///
/// Phase 5 retired Option B's per-frame TileIndex + ProtoLookup
/// upload — the host march doesn't need either since it routes
/// through `asset.shader_id` + `inst.instance_state_offset` directly.
fn tick_instance_pipeline(state: &mut RenderState, frame: &RenderFrame) -> InstancePipelineResult {
    use rkp_render::user_shader_emit_pass::{
        build_emit_region_uniform, resolve_instance_shader,
        workgroups_for_leaf_count, EmitDispatchUniform, EMIT_DISPATCH_UNIFORM_STRIDE,
        PaintedLeaf,
    };
    use rkp_render::user_shader_proto_pass::{
        build_internal_levels, PrototypeUniform, MAX_PROTO_MAX_DEPTH,
        PROTO_TAIL_OCTREE_BYTES, PROTO_TAIL_BRICK_BYTES, PROTO_TAIL_LEAF_ATTR_BYTES,
    };
    use rkp_render::user_shader_pass::{
        BRICK_CELLS, MAX_GLOBAL_BRICKS, MAX_GLOBAL_LEAF_ATTRS, MAX_GLOBAL_OCTREE_NODES,
    };
    use rkp_core::brick_pool::BRICK_DIM;

    // 1. Pipeline reload — cheap when source hash unchanged.
    state.instance_proto_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_proto_chunk,
        frame.user_shader_source_hash,
    );
    state.instance_emit_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_emit_chunk,
        frame.user_shader_source_hash,
    );
    // Phase 5 — Option B's per-pixel `instance_march_pass` is gone;
    // host march reloads instead via `vr.march.reload_user_shaders`
    // earlier in the per-VR loop (see lifecycle.rs).

    // 2. Mark cache entries untouched.
    state.instance_proto_cache.begin_frame();
    state.instance_region_cache.begin_frame();

    // 3. Early-out when there are no requests this frame. The
    //    user_shader-tail + proto-pool reservations below grow
    //    `brick_pool_buffer` / `leaf_attr_pool_buffer` by ~768 MB
    //    (`MAX_GLOBAL_BRICKS × 64 × 4`) plus a small proto sub-pool;
    //    on scenes whose CPU brick footprint already runs into the
    //    hundreds of MB this pushes past the 1 GB
    //    `max_storage_buffer_binding_size` we request, leaving the
    //    reallocated buffers in an invalid state. Reserve lazily —
    //    same shape as `run_user_shader_geom`'s `regions.is_empty()`
    //    early-out — so we only pay the cost when there's actual
    //    instance work to dispatch.
    if frame.instance_region_requests.is_empty() {
        state.instance_proto_cache.evict_untouched();
        state.instance_region_cache.evict_untouched();
        return InstancePipelineResult {
            user_shader_assets: Vec::new(),
            tile_cull_scratch_count: 0,
        };
    }

    // 4. Snapshot `cpu_*_bytes` from scene_mgr — same shape as
    //    `run_user_shader_geom` step 3.
    let (cpu_octree_bytes, cpu_brick_bytes, cpu_leaf_attr_bytes, cpu_face_links_bytes) = {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        (
            g.octree_nodes.len() as u64 * 8,
            g.brick_pool.len() as u64,
            g.leaf_attr_pool.len() as u64,
            g.brick_face_links.len() as u64,
        )
    };
    // 5. Phase 4 — reserve the proto tail in the host scene buffers
    //    (between CPU asset data and Phase C's per-frame transient
    //    tail). Putting proto data in the host pools means the host
    //    march descends a baked user-shader instance asset via its
    //    existing `octree_nodes_buffer` / `brick_pool_buffer` /
    //    `leaf_attr_pool_buffer` bindings — no "which pool?"
    //    indirection in the shader.
    //
    //    Sized small (16 + 16 + 8 MB) on purpose: the previous attempt
    //    to layer proto onto cpu + Phase C's huge transient (~768 MB
    //    on the brick buffer) breached the 1 GB binding cap and was
    //    rolled back. Tight proto-tail caps stay well within budget.
    // Reserve the three-tier envelope here. The Phase C transient size
    // only matters if Phase C will actually dispatch this frame —
    // run_user_shader_geom early-exits when `user_shader_regions` is
    // empty, in which case its `ensure_pool_layout` call never runs, so
    // we don't need to leave room for its extras. Mirror that gate
    // here: include extras IFF Phase C has work, else just `cpu + proto`.
    //
    // Without this gate, sizing for Phase C's extras (~1 GB on the
    // brick buffer at MAX_GLOBAL_BRICKS = 3M) breaches `max_buffer_size`
    // on devices that don't support a full 2 GB binding. Pre-Phase-4
    // tick_instance_pipeline didn't grow the host buffer at all, so the
    // breach only surfaces now that we route proto data into it.
    let phase_c_active = !frame.user_shader_regions.is_empty();
    let extra_octree: u64 = if phase_c_active {
        MAX_GLOBAL_OCTREE_NODES as u64 * 8
    } else { 0 };
    let extra_brick: u64 = if phase_c_active {
        MAX_GLOBAL_BRICKS as u64 * BRICK_CELLS as u64 * 4
    } else { 0 };
    let extra_leaf: u64 = if phase_c_active {
        MAX_GLOBAL_LEAF_ATTRS as u64 * 8
    } else { 0 };
    let extra_face_links: u64 = if phase_c_active {
        MAX_GLOBAL_BRICKS as u64 * 6 * 4
    } else { 0 };
    let proto_brick_count =
        (PROTO_TAIL_BRICK_BYTES / 4 / BRICK_CELLS as u64) as u32;
    let proto_face_links_bytes = (proto_brick_count as u64) * 6 * 4;
    let realloc = state.renderer.scene.ensure_pool_layout(
        &state.device,
        cpu_octree_bytes, PROTO_TAIL_OCTREE_BYTES, extra_octree,
        cpu_brick_bytes, PROTO_TAIL_BRICK_BYTES, extra_brick,
        cpu_leaf_attr_bytes, PROTO_TAIL_LEAF_ATTR_BYTES, extra_leaf,
        cpu_face_links_bytes, proto_face_links_bytes, extra_face_links,
    );
    if realloc {
        // A reallocation invalidates any persistent baked data and
        // forces a full geometry re-upload — the buffer's identity
        // changed.
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        // Initialize the proto + Phase C face_links regions to FACE_EMPTY
        // (mirrors run_user_shader_geom's existing init for the Phase C
        // tail). Both regions sit past `cpu_face_links_bytes`.
        const FACE_EMPTY: u32 = 0xFFFFFFFFu32;
        const FACE_INIT_CHUNK: usize = 4 * 1024 * 1024;
        let chunk_data: Vec<u32> = vec![FACE_EMPTY; FACE_INIT_CHUNK];
        let init_total = proto_face_links_bytes + extra_face_links;
        let mut written: u64 = 0;
        while written < init_total {
            let remaining = (init_total - written) as usize;
            let this_chunk_bytes = (FACE_INIT_CHUNK * 4).min(remaining);
            state.queue.write_buffer(
                &state.renderer.scene.brick_face_links_buffer,
                cpu_face_links_bytes + written,
                bytemuck::cast_slice(&chunk_data[..this_chunk_bytes / 4]),
            );
            written += this_chunk_bytes as u64;
        }
    }

    // 6. Configure proto pool bases to point at the host pool tails.
    //    Element units (octree slot, brick id, leaf-attr slot) — match
    //    what `RkpGpuAsset.octree_root` / brick_id pointers use.
    let proto_octree_base_elems = (cpu_octree_bytes / 8) as u32;
    let proto_brick_base_bricks =
        (cpu_brick_bytes / 4 / BRICK_CELLS as u64) as u32;
    let proto_leaf_attr_base_elems = (cpu_leaf_attr_bytes / 8) as u32;
    let bases_changed = state.instance_proto_cache.set_pool_bases(
        proto_octree_base_elems,
        proto_brick_base_bricks,
        proto_leaf_attr_base_elems,
    );
    // On bases-changed (or first frame, or buffer realloc), reset the
    // GPU cursors to the new bases so the next bake's atomic-bumps
    // produce slot ids inside the proto reservation.
    if bases_changed || realloc {
        state.instance_proto_pass.reset_cursors(
            &state.queue,
            proto_brick_base_bricks,
            proto_leaf_attr_base_elems,
        );
    }

    // 7. Walk requests, look up cache, gather dirty bakes + scatters.
    struct DirtyBake {
        uniform: PrototypeUniform,
        max_depth: u32,
        octree_extent_offset: u32,
    }
    struct Scatter {
        region_uniform: rkp_render::user_shader_emit_pass::EmitRegionUniform,
        region_index: u32,
        leaf_count: u32,
        instance_alloc_offset_bytes: u64,
    }
    let mut dirty_bakes: Vec<DirtyBake> = Vec::new();
    let mut scatters: Vec<Scatter> = Vec::new();
    // Phase 6 Session 3d — tile-cull region uniforms collected
    // alongside `scatters`. One per region, with `scratch_offset`
    // tracking the cumulative entry index into
    // `instance_tile_cull_scratch_buffer`.
    let mut tile_cull_regions: Vec<rkp_render::user_shader_tile_cull_pass::TileCullRegionUniform> =
        Vec::new();
    let mut tile_cull_scratch_running_offset: u32 = 0;
    // All regions' leaves concatenated into one upload. Each region's
    // EmitRegionUniform.leaf_offset is the index of its first leaf
    // here.
    let mut leaves_flat: Vec<PaintedLeaf> = Vec::new();

    // Phase 4 — user-shader instance asset registration + per-leaf
    // host-instance emit. One asset per `@instance_proto` shader
    // (deduped by shader_id); one host instance per painted leaf. The
    // host march reads these through the existing pool bindings and
    // treats user-shader instances identically to ordinary host objects
    // — so shadows, picking, fog, GI all "just work" the moment the
    // bake completes. Phase 4c branches on `asset.shader_id != 0` to
    // call the user shader's `inst_to_local` / `inst_aabb` hooks.
    use rkp_render::rkp_gpu_object::{geom_type, RkpGpuAsset};
    let mut asset_for_shader: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut user_shader_assets: Vec<RkpGpuAsset> = Vec::new();
    let asset_id_base = frame.gpu_assets.len() as u32;

    let time_seconds = frame.shade_params_base.time;
    for req in &frame.instance_region_requests {
        // Resolve the shader. Skip when not in registry or when it's
        // not an instance shader (shouldn't happen — the painter only
        // emits requests for instance shaders — but defend the path).
        let info = match resolve_instance_shader(&frame.user_shader_infos, &req.shader_name) {
            Some(i) => i,
            None => continue,
        };
        // Resolve to an id by name + index in user_shader_entries.
        // (UserShaderInfo doesn't carry id, so look it up in entries.)
        let shader_id = match frame
            .user_shader_entries
            .iter()
            .find(|e| e.name == info.name)
            .map(|e| e.id)
        {
            Some(id) => id,
            None => continue,
        };
        let max_depth = info.max_depth.unwrap_or(2).min(MAX_PROTO_MAX_DEPTH);

        // 7a. Proto cache lookup.
        let (proto_entry, proto_dirty) = match state.instance_proto_cache.lookup_or_allocate(
            shader_id,
            frame.user_shader_source_hash,
            max_depth,
        ) {
            Some(p) => p,
            None => {
                eprintln!(
                    "[inst] proto cache exhausted for shader_id {shader_id} \
                     — bake skipped this frame"
                );
                continue;
            }
        };
        if proto_dirty {
            dirty_bakes.push(DirtyBake {
                uniform: PrototypeUniform::from_entry(&proto_entry, &state.instance_proto_cache),
                max_depth: proto_entry.max_depth,
                octree_extent_offset: proto_entry.octree_extent.0,
            });
        }

        // 7b. Region cache lookup. V1: scatter every touched region
        //     regardless of dirty flag (see Stage 6c-3.5c memo for the
        //     alloc-count rationale).
        let topology_hash = topology_hash_for_inst(req, frame.geometry_epoch);
        let fill_hash = fill_hash_for_inst(
            req, topology_hash, frame.user_shader_source_hash, time_seconds,
        );
        let cached = match state
            .instance_region_cache
            .lookup_or_allocate(req, topology_hash, fill_hash)
        {
            Some(s) => s,
            None => {
                eprintln!(
                    "[inst] region cache exhausted for shader_id {shader_id} \
                     — scatter skipped this frame"
                );
                continue;
            }
        };

        // Assign region_index by walk order. Append this region's
        // painted leaves onto the flat upload buffer; remember the
        // start offset so the EmitRegionUniform points at it.
        let region_index = scatters.len() as u32;
        let leaf_offset = leaves_flat.len() as u32;
        let leaf_count = req.leaves.len() as u32;
        leaves_flat.extend_from_slice(&req.leaves);
        let mut slot = cached;
        slot.region_index = region_index;
        let region_uniform = build_emit_region_uniform(
            req, &slot, shader_id, time_seconds, leaf_offset,
        );
        let _ = proto_dirty;
        scatters.push(Scatter {
            region_uniform,
            region_index,
            leaf_count,
            instance_alloc_offset_bytes: (region_index as u64) * 4,
        });

        // Phase 4 — register the user-shader instance asset (once per
        // shader) and emit one host instance per painted leaf.
        //
        // The per-instance state (yaw, scale, lean, wind sway, etc — whatever
        // the shader's `@instance_proto` struct declared) lives in
        // `instance_pool`. The host instance carries:
        //   * `asset_id` → the prototype's RkpGpuAsset (which has
        //     `shader_id != 0`, telling the march to use user-shader hooks)
        //   * `instance_state_offset` → u32 offset into `instance_pool`
        //     where this instance's per-instance state record lives
        //   * `world` (translation-only, centered on the leaf, scaled by
        //     `cell_size`) — used only for the screen-AABB tile cull;
        //     the user's `inst_to_local` / `inst_aabb` hooks override
        //     world placement at march time using the per-instance state.
        //
        // The GPU emit pass writes per-instance records into
        // `instance_pool` via atomicAdd, so the slot ordering across
        // leaves is non-deterministic. That's fine: each rendered
        // instance appears at its actual world position dictated by the
        // user's hooks reading from slot N (whichever leaf wrote there).
        // The CPU's per-leaf tile-cull AABB is conservative-but-correct
        // because the per-instance AABB extent (~scale) is much larger
        // than inter-leaf spacing (~cell_size), so a slot-permuted
        // instance still falls inside its predicted screen tiles.
        let asset_index = match asset_for_shader.get(&shader_id) {
            Some(&idx) => idx,
            None => {
                let idx = user_shader_assets.len() as u32;
                let extent = 1.0_f32; // prototype's local-space cube
                let voxel_size_local =
                    extent / ((1u32 << proto_entry.max_depth) as f32 * BRICK_DIM as f32);
                user_shader_assets.push(RkpGpuAsset {
                    aabb_min: [0.0, 0.0, 0.0],
                    octree_root: proto_entry
                        .octree_root(state.instance_proto_cache.pool_octree_base()),
                    aabb_max: [extent, extent, extent],
                    octree_depth: proto_entry.max_depth,
                    octree_extent_bits: extent.to_bits(),
                    voxel_size: voxel_size_local,
                    geom_type: geom_type::VOXELIZED,
                    bone_count: 0,
                    grid_origin: [0.0, 0.0, 0.0],
                    rest_octree_root: 0,
                    rest_octree_depth: 0,
                    rest_octree_extent_bits: 0,
                    // Phase 4c — flags this asset as a user-shader proto.
                    // The host march reads `asset.shader_id` and routes
                    // descent through the user's `inst_to_local` /
                    // `inst_aabb` hooks instead of the affine
                    // `inv_world` path.
                    shader_id,
                    _pad: 0,
                });
                asset_for_shader.insert(shader_id, idx);
                idx
            }
        };
        let asset_id = asset_id_base + asset_index;

        // Phase 6 Session 3d — tile-cull region uniform. One per
        // region. `scratch_offset` accumulates: each region's slice
        // covers `instance_block_size` consecutive entries in the
        // scratch buffer. The AABB pass writes there; count + scatter
        // (per-VR) read.
        tile_cull_regions.push(rkp_render::user_shader_tile_cull_pass::TileCullRegionUniform {
            region_index,
            asset_id,
            material_id: req.material_id,
            shader_id,
            instance_block_offset: region_uniform.instance_block_offset,
            instance_block_size: region_uniform.instance_block_size,
            instance_stride_u32: region_uniform.instance_stride_u32,
            scratch_offset: tile_cull_scratch_running_offset,
        });
        tile_cull_scratch_running_offset = tile_cull_scratch_running_offset
            .saturating_add(region_uniform.instance_block_size);

        // Phase 6 Session 4d — leaf-driven CPU instance emit ripped.
        // The per-leaf `RkpGpuInstance` push that lived here is gone:
        // user-shader instances flow exclusively through the GPU
        // tile-cull pipeline now (Sessions 2 + 3 + 4b). The `asset_id`
        // computed above stays in `tile_cull_regions[r].asset_id` so
        // each entry the host march iterates can look up the right
        // asset slot in the combined assets vec.
        //
        // V1 trade-off: user-shader instances no longer cast shadows
        // (shadow trace iterates `instances[]` per pixel; it can't
        // see the GPU-discovered entries). Documented in
        // `project_phase_6_session_4`; future work adds light-aligned
        // tile cull for shadows.
        let _ = asset_id;
    }
    // 8. Encode bake + scatter dispatches.
    if !dirty_bakes.is_empty() || !scatters.is_empty() {
        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("inst bake+scatter"),
        });

        // 8a. Bake — one dispatch per dirty proto. Each dispatch writes
        //     the prototype's leaf-level octree slots + brick + leaf
        //     attrs starting from the proto cache's reserved offset.
        //     We pre-fill the internal levels CPU-side per Stage 2's
        //     pattern. Wrap the whole bake loop in one profiler query
        //     so a frame with N dirty bakes shows as one `inst_bake`
        //     bucket; rare per-shader rebakes mean N is usually 0 or 1.
        let bake_q = if !dirty_bakes.is_empty() {
            Some(state.renderer.profiler.begin_query("inst_bake", &mut encoder))
        } else {
            None
        };
        for bake in &dirty_bakes {
            // Pre-fill internal octree levels at the proto's reserved
            // offset within the host octree pool. `build_internal_levels`
            // returns the entire octree subtree's nodes (internal +
            // EMPTY leaf); the bake fills the leaf level only.
            //
            // `pool_octree_base = proto_octree_base_elems` so the
            // internal branches reference children at absolute host-pool
            // slot indices.
            let internal = build_internal_levels(
                proto_octree_base_elems,
                bake.octree_extent_offset,
                bake.max_depth,
            );
            // Each node = 2 u32s = 8 bytes.
            let mut bytes: Vec<u8> = Vec::with_capacity(internal.len() * 8);
            for [v0, v1] in internal {
                bytes.extend_from_slice(&v0.to_le_bytes());
                bytes.extend_from_slice(&v1.to_le_bytes());
            }
            // Absolute byte offset in the host octree_nodes_buffer.
            let octree_byte_offset =
                (proto_octree_base_elems as u64 + bake.octree_extent_offset as u64) * 8;
            state.queue.write_buffer(
                &state.renderer.scene.octree_nodes_buffer,
                octree_byte_offset,
                &bytes,
            );

            // Reset overflow only — brick + leaf-attr cursors are
            // PERSISTENT across bakes (different prototypes' slots
            // interleave in the global pools). Clearing them here
            // would leak unreachable slots in the previously-baked
            // prototypes' octree branches.
            state.queue.write_buffer(&state.instance_proto_pass.overflow_buffer, 0, &[0u8; 12 * 4]);

            // Upload the prototype uniform.
            state.queue.write_buffer(
                &state.instance_proto_pass.proto_uniform_buffer,
                0,
                bytemuck::bytes_of(&bake.uniform),
            );

            // Build bind groups — proto bake now writes into the host
            // scene's main pool buffers (Phase 4). The atomic cursors
            // were initialized to `(proto_brick_base, proto_leaf_attr_base)`
            // in step 6 so the first emitted slot lands at the proto
            // tail, not at offset 0 (which would clobber CPU asset data).
            let bake_g0 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst bake g0"),
                layout: &state.instance_proto_pass.group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.renderer.scene.octree_nodes_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.renderer.scene.brick_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.renderer.scene.leaf_attr_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: state.instance_proto_pass.cursors_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: state.instance_proto_pass.overflow_buffer.as_entire_binding() },
                ],
            });
            let bake_g1 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst bake g1"),
                layout: &state.instance_proto_pass.group1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.instance_proto_pass.proto_uniform_buffer.as_entire_binding(),
                }],
            });

            let bricks_per_axis = 1u32 << bake.max_depth;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inst bake"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&state.instance_proto_pass.bake_pipeline);
            cpass.set_bind_group(0, &bake_g0, &[]);
            cpass.set_bind_group(1, &bake_g1, &[]);
            cpass.dispatch_workgroups(bricks_per_axis, bricks_per_axis, bricks_per_axis);
        }
        if let Some(q) = bake_q {
            state.renderer.profiler.end_query(&mut encoder, q);
        }

        // 8b. Scatter — one dispatch per touched region.
        if !scatters.is_empty() {
            // Upload all region uniforms + dispatch uniforms once.
            // `regions_buffer` is sized for MAX_INSTANCE_REGIONS slots
            // (192 B each). Reset alloc counters for every region we
            // dispatch.
            let mut region_bytes: Vec<u8> =
                Vec::with_capacity(scatters.len() * std::mem::size_of::<rkp_render::user_shader_emit_pass::EmitRegionUniform>());
            for s in &scatters {
                region_bytes.extend_from_slice(bytemuck::bytes_of(&s.region_uniform));
            }
            state.queue.write_buffer(
                &state.instance_emit_pass.regions_buffer,
                0,
                &region_bytes,
            );

            // Upload the concatenated leaves buffer. Dispatch fires one
            // thread per leaf; each region's uniform points at its
            // leaf range via `leaf_offset` / `leaf_count`.
            let leaves_needed = leaves_flat.len() as u32;
            ensure_instance_leaves_capacity(state, leaves_needed.max(1));
            if !leaves_flat.is_empty() {
                state.queue.write_buffer(
                    &state.instance_leaves_buffer,
                    0,
                    bytemuck::cast_slice(&leaves_flat),
                );
            }

            // Dispatch uniforms — one per scatter, EMIT_DISPATCH_UNIFORM_STRIDE
            // apart (256 B per wgpu's dynamic-offset alignment rule).
            let stride = EMIT_DISPATCH_UNIFORM_STRIDE as usize;
            let mut dispatch_bytes: Vec<u8> = vec![0u8; scatters.len() * stride];
            for (i, s) in scatters.iter().enumerate() {
                let dispatch_u = EmitDispatchUniform {
                    region_index: s.region_index,
                    leaf_count: s.leaf_count,
                    _pad0: 0,
                    _pad1: 0,
                };
                let off = i * stride;
                dispatch_bytes[off..off + std::mem::size_of::<EmitDispatchUniform>()]
                    .copy_from_slice(bytemuck::bytes_of(&dispatch_u));
            }
            state.queue.write_buffer(
                &state.instance_emit_pass.dispatch_uniforms_buffer,
                0,
                &dispatch_bytes,
            );

            // Reset alloc counters for every region we're scattering.
            // Simplest correct approach: zero the whole alloc buffer
            // once. Cost: 4 KB at MAX_INSTANCE_REGIONS=1024. The
            // alternative (per-region writes) is slower for >32
            // regions and risks leaving stale values in slots that
            // were touched in a prior frame but aren't this frame.
            // Note: since V1 always re-scatters every touched region,
            // every counter that the march reads gets repopulated
            // before submit — zeroing first is safe.
            let zeros = vec![
                0u8;
                (rkp_render::user_shader_emit_pass::MAX_INSTANCE_REGIONS as usize) * 4
            ];
            state.queue.write_buffer(
                &state.instance_emit_pass.instance_alloc_buffer,
                0,
                &zeros,
            );
            // Reset overflow counters.
            state.queue.write_buffer(
                &state.instance_emit_pass.overflow_buffer,
                0,
                &[0u8; 16],
            );

            // Build bind groups (groups 0 + 1 are stable across
            // dispatches; group 2 takes a dynamic offset per dispatch).
            // Group 0 binding 2 is the painted-leaves buffer — the
            // emit shader fetches `leaves[region.leaf_offset + gid]`
            // directly, no host-octree descent.
            let scatter_g0 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst scatter g0"),
                layout: &state.instance_emit_pass.group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.renderer.scene.instance_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.instance_emit_pass.instance_alloc_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.instance_leaves_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: state.instance_emit_pass.overflow_buffer.as_entire_binding() },
                ],
            });
            let scatter_g1 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst scatter g1"),
                layout: &state.instance_emit_pass.group1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.instance_emit_pass.regions_buffer.as_entire_binding(),
                }],
            });
            let scatter_g2 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst scatter g2 (dynamic)"),
                layout: &state.instance_emit_pass.group2_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &state.instance_emit_pass.dispatch_uniforms_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(
                            std::mem::size_of::<EmitDispatchUniform>() as u64,
                        ),
                    }),
                }],
            });

            // One profiler bucket for the full scatter loop. With
            // @animated shaders this fires every frame for every
            // touched region, so this is the budget number to watch.
            let scatter_q = state.renderer.profiler.begin_query("inst_scatter", &mut encoder);
            for (i, s) in scatters.iter().enumerate() {
                if s.leaf_count == 0 {
                    continue;
                }
                let dynamic_offset = (i * stride) as u32;
                let wgs = workgroups_for_leaf_count(s.leaf_count);
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inst scatter"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&state.instance_emit_pass.emit_pipeline);
                cpass.set_bind_group(0, &scatter_g0, &[]);
                cpass.set_bind_group(1, &scatter_g1, &[]);
                cpass.set_bind_group(2, &scatter_g2, &[dynamic_offset]);
                // 1-D dispatch: `leaf_count.div_ceil(64)` workgroups, 64
                // threads each → one thread per painted leaf.
                cpass.dispatch_workgroups(wgs, 1, 1);
                let _ = s.instance_alloc_offset_bytes;
            }
            state.renderer.profiler.end_query(&mut encoder, scatter_q);
        }

        // 8c. Phase 6 Session 3d — tile-cull AABB pass. Per region,
        //     one workgroup-grid sized to its `instance_block_size`;
        //     each thread reads the per-instance state from
        //     `instance_pool` and writes one `InstanceTileCullEntry`
        //     into `instance_tile_cull_scratch_buffer`. Per-VR
        //     count/prefix/scatter dispatches downstream consume
        //     this scratch.
        if !tile_cull_regions.is_empty() {
            // Ensure scratch buffer is large enough for the running
            // total computed during region walk. 48 B per entry.
            let scratch_total_entries = tile_cull_scratch_running_offset.max(1);
            ensure_tile_cull_scratch_capacity(state, scratch_total_entries);

            // Reload the pipeline against the current user-shader
            // chunks. Cheap when source hash unchanged.
            state.instance_tile_cull_pass.reload_user_shaders(
                &state.device,
                &frame.user_shader_inst_to_local_chunk,
                &frame.user_shader_inst_aabb_chunk,
                frame.user_shader_source_hash,
            );

            // Upload region uniforms — one per region, 256 B stride
            // per wgpu's dynamic-offset alignment. Pad with zero bytes
            // between records; we only read the first 32 B per slot.
            const STRIDE: usize = rkp_render::user_shader_tile_cull_pass::TILE_CULL_REGION_UNIFORM_STRIDE as usize;
            let mut region_bytes: Vec<u8> = vec![0u8; tile_cull_regions.len() * STRIDE];
            for (i, r) in tile_cull_regions.iter().enumerate() {
                let off = i * STRIDE;
                let end = off + std::mem::size_of::<rkp_render::user_shader_tile_cull_pass::TileCullRegionUniform>();
                region_bytes[off..end].copy_from_slice(bytemuck::bytes_of(r));
            }
            state.queue.write_buffer(
                &state.instance_tile_cull_pass.regions_buffer, 0, &region_bytes,
            );

            // Bind groups — group(0) shared across all per-region
            // dispatches; group(1) takes a dynamic offset per region.
            let cull_g0 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst tile_cull g0"),
                layout: &state.instance_tile_cull_pass.group0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state.renderer.scene.instance_pool_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: state.instance_emit_pass.instance_alloc_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: state.instance_tile_cull_scratch_buffer.as_entire_binding() },
                ],
            });
            let cull_g1 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inst tile_cull g1 (dynamic)"),
                layout: &state.instance_tile_cull_pass.group1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &state.instance_tile_cull_pass.regions_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(
                            std::mem::size_of::<rkp_render::user_shader_tile_cull_pass::TileCullRegionUniform>() as u64,
                        ),
                    }),
                }],
            });

            let cull_q = state.renderer.profiler.begin_query("inst_tile_cull", &mut encoder);
            for (i, r) in tile_cull_regions.iter().enumerate() {
                let dynamic_offset = (i * STRIDE) as u32;
                let wgs = rkp_render::user_shader_tile_cull_pass::workgroups_for_region(r.instance_block_size);
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("inst tile_cull"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&state.instance_tile_cull_pass.pipeline);
                cpass.set_bind_group(0, &cull_g0, &[]);
                cpass.set_bind_group(1, &cull_g1, &[dynamic_offset]);
                cpass.dispatch_workgroups(wgs, 1, 1);
            }
            state.renderer.profiler.end_query(&mut encoder, cull_q);
        }

        state.queue.submit(Some(encoder.finish()));
    }

    // 9. Drop cache entries not referenced this frame.
    state.instance_proto_cache.evict_untouched();
    state.instance_region_cache.evict_untouched();

    InstancePipelineResult {
        user_shader_assets,
        tile_cull_scratch_count: tile_cull_scratch_running_offset,
    }
}

/// Hash inputs that affect the proto bake's topology output for one
/// instance region. Folds host-octree state + AABB + cell_size +
/// max_depth + tile + region_thickness. Any change invalidates the
/// region's cached extent.
fn topology_hash_for_inst(
    req: &rkp_render::user_shader_emit_pass::InstanceRegionRequest,
    geometry_epoch: u64,
) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &geometry_epoch.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_root.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_depth.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_extent.to_le_bytes() { mix(&mut h, b); }
    for v in req.host_grid_origin.iter() {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for row in req.host_inverse_world.iter() {
        for v in row.iter() {
            for &b in &v.to_le_bytes() { mix(&mut h, b); }
        }
    }
    for &b in &req.region_thickness.to_le_bytes() { mix(&mut h, b); }
    for v in req.aabb_min.iter().chain(req.aabb_max.iter()) {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for &b in &req.cell_size.to_le_bytes() { mix(&mut h, b); }
    for v in req.tile_index.iter() {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    h
}

/// Hash inputs that affect the scatter output (per-instance struct
/// values). Builds on top of `topology_hash` so changing topology
/// implies fill-dirty.
fn fill_hash_for_inst(
    req: &rkp_render::user_shader_emit_pass::InstanceRegionRequest,
    topology_hash: u64,
    shader_source_hash: u64,
    time_seconds: f32,
) -> u64 {
    let mut h = topology_hash;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &shader_source_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.input_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.material_id.to_le_bytes() { mix(&mut h, b); }
    for p in &req.params {
        for &b in &p.to_le_bytes() { mix(&mut h, b); }
    }
    if req.animated {
        for &b in &time_seconds.to_le_bytes() { mix(&mut h, b); }
    }
    h
}

/// What `tick_instance_pipeline` returns: Phase 4c's host-side
/// user-shader instance asset + instance pair that splices onto
/// `frame.gpu_assets` / `gpu_instances` so the host march can render
/// user-shader instances natively.
struct InstancePipelineResult {
    /// One asset per `@instance_proto` shader, sourced from the proto
    /// cache's reserved octree slot in the host pool tail. Spliced onto
    /// the frame's persistent assets vec just like
    /// `run_user_shader_geom`'s transient assets. Each entry has
    /// `shader_id != 0` so the host march dispatches user-shader
    /// hooks instead of the affine `inv_world` path. The host march
    /// reads each `UserShaderTileEntry`'s `asset_id` and indexes into
    /// `assets[asset_id]` directly.
    user_shader_assets: Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>,
    /// Phase 6 Session 3d — total entry count in
    /// `instance_tile_cull_scratch_buffer` after the AABB pass writes.
    /// Equal to `Σ instance_block_size` across all regions; the per-VR
    /// count/scatter dispatches use this as their thread count.
    /// Zero when no user-shader regions ran this frame.
    tile_cull_scratch_count: u32,
}

/// Phase 6 Session 3d — per-VR tile-cull dispatch chain.
///
/// Runs count → prefix → scatter against `state.instance_tile_cull_scratch_buffer`
/// (filled by `tick_instance_pipeline`'s AABB pass), writing per-tile
/// entry lists into `vr.march.us_tile_offsets` / `us_tile_entries`. The
/// host march doesn't yet read these (Session 4 wires that consumer);
/// Session 3d is purely "make the dispatch chain run end-to-end".
///
/// Skipped entirely when `scratch_count == 0` (no user-shader
/// instances this frame).
/// Pre-borrowed handles to RenderState's tile-cull fields. The caller
/// constructs this at the call site so the inner helper doesn't need
/// `&mut RenderState` (which would conflict with the outer
/// `viewport_renderers.get_mut(...)`).
struct UsTileCullArgs<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    /// Renderer borrow only used to access `profiler`. Held as
    /// `&mut RkpRenderer` rather than the inner profiler so engine
    /// callers don't need a `wgpu_profiler` import.
    renderer: &'a mut rkp_render::rkp_renderer::RkpRenderer,
    scratch_buffer: &'a wgpu::Buffer,
    view_uniform_buffer: &'a wgpu::Buffer,
    prefix_uniform_buffer: &'a wgpu::Buffer,
    tile_count_pass: &'a rkp_render::user_shader_tile_count_pass::TileCountPass,
    tile_prefix_pass: &'a rkp_render::user_shader_tile_prefix_pass::TilePrefixPass,
    tile_scatter_pass: &'a rkp_render::user_shader_tile_scatter_pass::TileScatterPass,
}

fn dispatch_us_tile_cull_inner(
    args: UsTileCullArgs,
    vr: &mut rkp_render::ViewportRenderer,
    encoder: &mut wgpu::CommandEncoder,
    camera: &rkp_render::rkp_scene::CameraUniforms,
    width: u32,
    height: u32,
    scratch_count: u32,
) {
    use rkp_render::user_shader_tile_count_pass::{
        tile_count_for_viewport, workgroups_for_scratch, TileCullViewportUniform,
    };
    use rkp_render::user_shader_tile_prefix_pass::{PrefixUniform, PREFIX_MAX_TILES};

    let (tile_count_x, tile_count_y, tile_count) = tile_count_for_viewport(width, height);
    if tile_count > PREFIX_MAX_TILES {
        // V1 cap: skip the entire VR rather than producing partial
        // results. Above 65536 tiles (~2300×800 at 8 px tiles) the
        // single-WG prefix-sum can't scan in one dispatch; multi-WG
        // scan is a follow-up.
        eprintln!(
            "[tile_cull] tile_count {tile_count} > PREFIX_MAX_TILES {PREFIX_MAX_TILES} \
             ({width}×{height}); skipping VR"
        );
        return;
    }

    // Conservative entries estimate. Two regimes mixed:
    // (a) typical: each AABB covers a small number of tiles at distance
    //     (~16-64 tiles for grass-blade-sized AABBs).
    // (b) close to camera: tight near-plane clipping in the projection
    //     keeps the screen AABB bounded, but a single blade right at
    //     the camera can still cover hundreds of tiles. With a few
    //     close blades, the average shoots up.
    // V1 heuristic is `scratch_count × 256 + tile_count × 4` — covers
    // the average case with headroom and handles up to ~4 blades
    // touching every tile without overflow. Grows on demand.
    let entries_estimate = scratch_count
        .saturating_mul(256)
        .saturating_add(tile_count.saturating_mul(4))
        .max(1024);
    let _grew_entries = vr.march.ensure_us_tile_entries_capacity(args.device, entries_estimate);
    let _grew_grid = vr.march.ensure_us_tile_grid_capacity(args.device, tile_count);

    // Upload the shared per-VR uniform buffers.
    let view_u = TileCullViewportUniform {
        view_proj: camera.view_proj,
        resolution_x: width as f32,
        resolution_y: height as f32,
        tile_count_x,
        tile_count_y,
        tile_count,
        scratch_count,
        _pad0: 0,
        _pad1: 0,
    };
    args.queue.write_buffer(
        args.view_uniform_buffer, 0, bytemuck::bytes_of(&view_u),
    );
    let prefix_u = PrefixUniform {
        tile_count, _pad0: 0, _pad1: 0, _pad2: 0,
    };
    args.queue.write_buffer(
        args.prefix_uniform_buffer, 0, bytemuck::bytes_of(&prefix_u),
    );

    // Reset us_tile_counts to zeros for this VR. clear_buffer requires
    // the encoder; size is `tile_count * 4` rounded up to a multiple
    // of `wgpu::COPY_BUFFER_ALIGNMENT` (= 4) — already aligned.
    encoder.clear_buffer(
        &vr.march.us_tile_counts_buffer,
        0,
        Some((tile_count as u64) * 4),
    );

    // ── Bind groups ──────────────────────────────────────────────────
    // Count pass: scratch (ro) + counts (rw atomic) + view uniform.
    let count_g0 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_count g0"),
        layout: &args.tile_count_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: args.scratch_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vr.march.us_tile_counts_buffer.as_entire_binding() },
        ],
    });
    let count_g1 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_count g1"),
        layout: &args.tile_count_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: args.view_uniform_buffer.as_entire_binding(),
        }],
    });

    // Prefix pass: counts (ro) + offsets (rw) + prefix uniform.
    let prefix_g0 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_prefix g0"),
        layout: &args.tile_prefix_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vr.march.us_tile_counts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vr.march.us_tile_offsets_buffer.as_entire_binding() },
        ],
    });
    let prefix_g1 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_prefix g1"),
        layout: &args.tile_prefix_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: args.prefix_uniform_buffer.as_entire_binding(),
        }],
    });

    // Scatter pass: scratch (ro) + scatter_cursor (rw atomic) +
    // entries (rw) + view uniform.
    let scatter_g0 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_scatter g0"),
        layout: &args.tile_scatter_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: args.scratch_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vr.march.us_tile_scatter_cursor_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vr.march.us_tile_entries_buffer.as_entire_binding() },
        ],
    });
    let scatter_g1 = args.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("inst tile_scatter g1"),
        layout: &args.tile_scatter_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: args.view_uniform_buffer.as_entire_binding(),
        }],
    });

    // ── Dispatches ──────────────────────────────────────────────────
    let q = args.renderer.profiler.begin_query("inst_tile_dispatch", encoder);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("inst tile_count"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&args.tile_count_pass.pipeline);
        cpass.set_bind_group(0, &count_g0, &[]);
        cpass.set_bind_group(1, &count_g1, &[]);
        cpass.dispatch_workgroups(workgroups_for_scratch(scratch_count), 1, 1);
    }

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("inst tile_prefix"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&args.tile_prefix_pass.pipeline);
        cpass.set_bind_group(0, &prefix_g0, &[]);
        cpass.set_bind_group(1, &prefix_g1, &[]);
        // Single workgroup — see PREFIX_MAX_TILES contract.
        cpass.dispatch_workgroups(1, 1, 1);
    }

    // Initialize scatter_cursor[t] = us_tile_offsets[t] for t in
    // [0, tile_count). The scatter pass atomicAdd's into cursor, so
    // each entry's slot lands in [offsets[t], offsets[t+1]) — leaving
    // us_tile_offsets unchanged for the host march to read.
    vr.march.init_scatter_cursor(encoder, tile_count);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("inst tile_scatter"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&args.tile_scatter_pass.pipeline);
        cpass.set_bind_group(0, &scatter_g0, &[]);
        cpass.set_bind_group(1, &scatter_g1, &[]);
        cpass.dispatch_workgroups(workgroups_for_scratch(scratch_count), 1, 1);
    }

    args.renderer.profiler.end_query(encoder, q);
}

/// Phase 6 Session 3d — grow `instance_tile_cull_scratch_buffer` to
/// fit `needed_entries` × 48 B `InstanceTileCullEntry` records.
fn ensure_tile_cull_scratch_capacity(state: &mut RenderState, needed_entries: u32) {
    if needed_entries <= state.instance_tile_cull_scratch_capacity_entries {
        return;
    }
    let mut new_cap = state.instance_tile_cull_scratch_capacity_entries.max(1);
    while new_cap < needed_entries {
        new_cap = new_cap.saturating_mul(2);
    }
    state.instance_tile_cull_scratch_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inst tile_cull_scratch"),
        size: (new_cap as u64) * 48,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    state.instance_tile_cull_scratch_capacity_entries = new_cap;
}

/// Grow `instance_leaves_buffer` to fit `needed_entries`. No-op when
/// capacity already suffices. Doubles capacity on growth so a
/// gradually-growing workload doesn't reallocate every frame. The
/// buffer holds 32 B per `PaintedLeaf`.
fn ensure_instance_leaves_capacity(state: &mut RenderState, needed_entries: u32) {
    if needed_entries <= state.instance_leaves_capacity_entries {
        return;
    }
    let mut new_cap = state.instance_leaves_capacity_entries.max(1);
    while new_cap < needed_entries {
        new_cap = new_cap.saturating_mul(2);
    }
    state.instance_leaves_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inst leaves"),
        size: (new_cap as u64) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    state.instance_leaves_capacity_entries = new_cap;
}

/// User-shader runtime geometry, global-pool + persistent-cache variant.
///
/// Reserves a global transient pool tail in each scene buffer,
/// rebuilds the geom-build pipeline on shader source changes, walks
/// the frame's regions and consults the persistent cache to decide
/// per-region: skip / fill-only / full-bake. Returns the transient
/// `(asset, instance)` pair lists to concatenate with the persistent
/// vecs for this frame's `upload_frame`.
///
/// `asset_id_base` is the index into the combined assets vec where the
/// first transient asset will land — the cache assigns sequential ids
/// from this base.
///
/// Each region computes two hashes: `topology_hash` (host_geom +
/// region_thickness + max_depth + aabb + cell_size; classify can be
/// skipped when unchanged) and `fill_hash` (shader source + params +
/// paint epoch + time when `@animated`; fill can be skipped when
/// unchanged AND topology unchanged).
fn run_user_shader_geom(
    state: &mut RenderState,
    frame: &RenderFrame,
    asset_id_base: u32,
) -> (
    Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>,
    Vec<rkp_render::rkp_gpu_object::RkpGpuInstance>,
) {
    use rkp_render::user_shader_pass::{
        build_region_uniform, estimate_region_pool, resolve_shader_id, CachedSlot,
        RegionUniform, BRICK_CELLS, MAX_GLOBAL_BRICKS, MAX_GLOBAL_LEAF_ATTRS,
        MAX_GLOBAL_OCTREE_NODES, MAX_REGIONS,
    };
    use rkp_render::user_shader_proto_pass::{
        PROTO_TAIL_OCTREE_BYTES, PROTO_TAIL_BRICK_BYTES, PROTO_TAIL_LEAF_ATTR_BYTES,
    };

    const FACE_EMPTY: u32 = 0xFFFFFFFFu32;

    // 1. Pipeline reload — track the shade-side hash; the geom and
    //    shade chunks share the same `source_hash`.
    state.user_shader_pass.reload_user_shaders(
        &state.device,
        &frame.user_shader_generate_chunk,
        frame.user_shader_source_hash,
    );

    // 2. Mark cache entries untouched; we'll touch the ones we hit.
    state.user_shader_cache.begin_frame();

    if frame.user_shader_regions.is_empty() {
        // Nothing to dispatch — drop any entries left over from prior
        // frames so they release their pool extents.
        state.user_shader_cache.evict_untouched();
        return state.user_shader_cache.build_transient_assets_and_instances(asset_id_base);
    }

    // 3. Buffer reservation. Stable across frames once geometry is
    //    loaded. The user-shader transient tail is sized at the
    //    global caps; the cache sub-allocates within.
    let extra_octree: u64 = MAX_GLOBAL_OCTREE_NODES as u64 * 8;
    let extra_brick: u64 = MAX_GLOBAL_BRICKS as u64 * BRICK_CELLS as u64 * 4;
    let extra_leaf: u64 = MAX_GLOBAL_LEAF_ATTRS as u64 * 8;
    let extra_face_links: u64 = MAX_GLOBAL_BRICKS as u64 * 6 * 4;

    let need_regions = (frame.user_shader_regions.len() as u32).min(MAX_REGIONS);

    let (cpu_octree_bytes, cpu_brick_bytes, cpu_leaf_attr_bytes, cpu_face_links_bytes) = {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        (
            g.octree_nodes.len() as u64 * 8,
            g.brick_pool.len() as u64,
            g.leaf_attr_pool.len() as u64,
            g.brick_face_links.len() as u64,
        )
    };
    // Phase 4 — proto tail sits between CPU and Phase C transient.
    // The proto reservation matches `tick_instance_pipeline`'s; both
    // calls grow the buffer to the union, so order doesn't matter.
    // Phase C's brick range needs face_links covering its absolute
    // brick_ids; proto's bricks need face_links too, init'd to
    // FACE_EMPTY so the march cleanly exits proto bricks at boundaries.
    let proto_brick_count =
        (PROTO_TAIL_BRICK_BYTES / 4 / BRICK_CELLS as u64) as u32;
    let proto_face_links_bytes = (proto_brick_count as u64) * 6 * 4;
    let realloc = state.renderer.scene.ensure_pool_layout(
        &state.device,
        cpu_octree_bytes, PROTO_TAIL_OCTREE_BYTES, extra_octree,
        cpu_brick_bytes, PROTO_TAIL_BRICK_BYTES, extra_brick,
        cpu_leaf_attr_bytes, PROTO_TAIL_LEAF_ATTR_BYTES, extra_leaf,
        cpu_face_links_bytes, proto_face_links_bytes, extra_face_links,
    );
    if realloc {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let g = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &g);
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        // One-time face-links init: the user-shader BFS never writes
        // into this buffer but the march reads it for any
        // user-shader-allocated brick. Uninitialised values would jump
        // the DDA chain into stale brick_id=0. Also covers the proto
        // range — proto bake doesn't write face_links either, so leaving
        // them at FACE_EMPTY makes the host march cleanly exit
        // user-shader instance bricks at boundaries (cross-brick
        // navigation within a single instance is unsupported until a
        // follow-up fix populates them).
        let init_total_bytes = proto_face_links_bytes + extra_face_links;
        const FACE_INIT_CHUNK: usize = 4 * 1024 * 1024;
        let chunk_data: Vec<u32> = vec![FACE_EMPTY; FACE_INIT_CHUNK];
        let mut written: u64 = 0;
        while written < init_total_bytes {
            let remaining = (init_total_bytes - written) as usize;
            let this_chunk_bytes = (FACE_INIT_CHUNK * 4).min(remaining);
            state.queue.write_buffer(
                &state.renderer.scene.brick_face_links_buffer,
                cpu_face_links_bytes + written,
                bytemuck::cast_slice(&chunk_data[..this_chunk_bytes / 4]),
            );
            written += this_chunk_bytes as u64;
        }
    }

    // 4. Configure pool bases — flushes the cache if they changed.
    //    Also reconcile against the host's geometry epoch (any host
    //    geometry change invalidates every region's topology_hash).
    //    Phase C's transient range starts past the proto reservation
    //    so user-shader proto bricks and Phase C bricks have disjoint
    //    brick_ids (and disjoint face_links slots).
    let proto_octree_elems = (PROTO_TAIL_OCTREE_BYTES / 8) as u32;
    let proto_leaf_attr_elems = (PROTO_TAIL_LEAF_ATTR_BYTES / 8) as u32;
    let octree_base_elems = (cpu_octree_bytes / 8) as u32 + proto_octree_elems;
    let brick_base_bricks =
        (cpu_brick_bytes / 4 / BRICK_CELLS as u64) as u32 + proto_brick_count;
    let leaf_base_elems = (cpu_leaf_attr_bytes / 8) as u32 + proto_leaf_attr_elems;
    state.user_shader_cache.set_pool_bases(
        octree_base_elems, brick_base_bricks, leaf_base_elems,
    );
    state.user_shader_cache.reconcile_epoch(frame.geometry_epoch);

    // 5. Walk regions, look up cache, gather dirty ones into two
    //    contiguous groups: topology-dirty first, then fill-only.
    let mut topology_dirty_uniforms: Vec<RegionUniform> = Vec::new();
    let mut fill_only_uniforms: Vec<RegionUniform> = Vec::new();
    let mut topology_dirty_slots: Vec<CachedSlot> = Vec::new();
    let mut fill_only_slots: Vec<CachedSlot> = Vec::new();
    let mut max_max_depth: u32 = 0;
    let time_seconds = frame.shade_params_base.time;
    for req in frame.user_shader_regions.iter().take(need_regions as usize) {
        let shader_id = resolve_shader_id(&frame.user_shader_infos, &req.shader_name);
        if shader_id == 0 {
            continue;
        }
        let topology_hash = topology_hash_for(req, frame.geometry_epoch);
        let fill_hash = fill_hash_for(
            req,
            topology_hash,
            frame.user_shader_source_hash,
            time_seconds,
        );
        let estimate = estimate_region_pool(req);
        let mut slot = match state.user_shader_cache.lookup_or_allocate(
            req, topology_hash, fill_hash, &estimate,
        ) {
            Some(s) => s,
            None => continue,
        };
        if !slot.topology_dirty && !slot.fill_dirty {
            continue; // Skip entirely; cached GPU contents still valid.
        }
        max_max_depth = max_max_depth.max(slot.max_depth);
        if slot.topology_dirty {
            slot.region_index = topology_dirty_slots.len() as u32;
            // index will get bumped by fill_only's count below; we
            // patch the uniform's region_index after gathering.
            topology_dirty_slots.push(slot);
            topology_dirty_uniforms.push(
                build_region_uniform(req, &slot, shader_id, time_seconds),
            );
        } else {
            fill_only_slots.push(slot);
            fill_only_uniforms.push(
                build_region_uniform(req, &slot, shader_id, time_seconds),
            );
        }
    }

    let topology_dirty_count = topology_dirty_uniforms.len() as u32;
    // Fill-only regions live at indices [topology_dirty_count, total).
    // Their region_index in the dispatch uniform must reflect that.
    for (i, _slot) in fill_only_slots.iter().enumerate() {
        // No-op: region_index is implicit in array order; the WGSL
        // reads `regions[wid.y]`, where wid.y = topology_dirty_count + i.
        let _ = i;
    }

    let mut uniforms = topology_dirty_uniforms;
    uniforms.extend(fill_only_uniforms);

    if !uniforms.is_empty() {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("user_shader_geom_encoder"),
            });
        state.user_shader_pass.dispatch_regions(
            &state.device,
            &state.queue,
            &mut encoder,
            &uniforms,
            topology_dirty_count,
            max_max_depth,
            &state.renderer.scene.octree_nodes_buffer,
            &state.renderer.scene.brick_pool_buffer,
            &state.renderer.scene.leaf_attr_pool_buffer,
            state.renderer.scene.buffers_epoch(),
        );
        state.queue.submit(Some(encoder.finish()));
        state.user_shader_pass.submit_overflow_readback();
    }

    // 6. Drop entries not touched this frame; their extents go back
    //    to the bucket allocators' free lists.
    state.user_shader_cache.evict_untouched();

    state.user_shader_cache.build_transient_assets_and_instances(asset_id_base)
}

/// Hash inputs that affect classify (BFS topology). Unchanged
/// topology hash → skip classify dispatch for this region.
fn topology_hash_for(
    req: &rkp_render::user_shader_pass::ShaderRegionRequest,
    geometry_epoch: u64,
) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &geometry_epoch.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_root.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_depth.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.host_octree_extent.to_le_bytes() { mix(&mut h, b); }
    for v in req.host_grid_origin.iter() {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for row in req.host_inverse_world.iter() {
        for v in row.iter() {
            for &b in &v.to_le_bytes() { mix(&mut h, b); }
        }
    }
    for &b in &req.region_thickness.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.max_depth.to_le_bytes() { mix(&mut h, b); }
    for v in req.aabb_min.iter().chain(req.aabb_max.iter()) {
        for &b in &v.to_le_bytes() { mix(&mut h, b); }
    }
    for &b in &req.cell_size.to_le_bytes() { mix(&mut h, b); }
    h
}

/// Hash inputs that affect fill (per-cell shader output). Unchanged
/// fill hash AND unchanged topology hash → skip fill dispatch.
fn fill_hash_for(
    req: &rkp_render::user_shader_pass::ShaderRegionRequest,
    topology_hash: u64,
    shader_source_hash: u64,
    time_seconds: f32,
) -> u64 {
    let mut h = topology_hash;
    let prime = 0x100000001b3u64;
    let mix = |h: &mut u64, b: u8| {
        *h ^= b as u64;
        *h = h.wrapping_mul(prime);
    };
    for &b in &shader_source_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.input_hash.to_le_bytes() { mix(&mut h, b); }
    for &b in &req.material_id.to_le_bytes() { mix(&mut h, b); }
    for &p in &req.params {
        for &b in &p.to_le_bytes() { mix(&mut h, b); }
    }
    if req.animated {
        for &b in &time_seconds.to_le_bytes() { mix(&mut h, b); }
    }
    h
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

