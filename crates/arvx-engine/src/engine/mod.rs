//! ArvxEngine — the self-contained game engine.
//!
//! Owns the tick loop, scene state, renderer, and all GPU resources.
//! Runs on its own thread. Communicates with clients via command channel
//! and shared snapshot.

use std::thread::JoinHandle;
use std::time::Duration;

use crate::command::EngineCommand;
use crate::snapshot::StateUpdate;

pub(crate) mod state;

mod asset_load_worker;
mod collider_worker;
mod cmd_edit;
mod cmd_runtime;
mod cmd_scene;
mod command_handler;
mod entity_ops;
mod gameplay_ops;
mod generator_ops;
mod geometry_dirty;
mod gpu_objects_dirty;
mod scene_dirty;
mod gizmo_ops;
mod import_ops;
mod lifecycle;
mod model_scan;
mod paint_ops;
mod paint_walk;
mod picking_ops;
mod mutation_log;
mod sculpt_ops;
mod procedural_gizmo;
mod procedural_ops;
mod procedural_params;
mod region_ops;
mod scene_gpu;
pub(crate) mod scene_io_ops;
mod scene_tree_ops;
mod stamp_ops;
mod state_update;
mod terrain_ops;
mod tick;
mod viewport_ops;

use tick::tick_loop;

/// Frame delivery callback — called once per visible viewport each tick.
/// `id` identifies which viewport this frame belongs to (the editor maps
/// each `ViewportId` to its own `RenderSurface`). RGBA8 pixels, length
/// `width * height * 4`.
pub type FrameCallback = Box<dyn Fn(crate::viewport::ViewportId, &[u8], u32, u32) + Send>;

/// State update callback — called each tick with engine state.
pub type StateCallback = Box<dyn Fn(&StateUpdate) + Send>;

/// How aggressively a thread loop should pace itself.
///
/// Used by both the sim tick loop ([`EngineConfig::sim_pacing`]) and
/// the render thread loop ([`EngineConfig::render_pacing`]).
///
/// - `Uncapped` runs as fast as the CPU/GPU can sustain. Right for
///   game builds shipped to players, or whenever you want maximum
///   throughput at the cost of CPU.
/// - `TargetHz(N)` sleeps each loop iteration's remainder to hold at
///   most `N` iterations per second. Right for the editor (60 Hz keeps
///   battery / fan reasonable), or to cap render at a display refresh
///   rate.
///
/// Sim correctness is independent of these knobs: physics and behavior
/// `FixedUpdate` run via accumulators on real wall-clock dt and tick
/// at the same simulation rate regardless of any pacing. Per-frame
/// systems (animation, camera/input, behavior `Update` / `LateUpdate`)
/// advance by real_dt and stay frame-rate-correct.
///
/// Display-rate vsync is *not* a value in this enum: the engine
/// renders headless to an offscreen texture and ships pixels to the
/// editor, where the actual presentation (and any vsync) is owned by
/// the rinch surface chain. To approximate vsync at the engine level,
/// set `render_pacing: TargetHz(display_refresh_hz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// Run as fast as possible.
    Uncapped,
    /// Sleep at the end of each loop iteration so the loop holds at
    /// most this many iterations per second.
    TargetHz(u32),
}

impl PacingMode {
    /// Sleep target as a Duration, or None for uncapped.
    pub fn target_interval(&self) -> Option<Duration> {
        match *self {
            PacingMode::Uncapped => None,
            PacingMode::TargetHz(0) => None,
            PacingMode::TargetHz(hz) => {
                Some(Duration::from_nanos(1_000_000_000u64 / hz as u64))
            }
        }
    }
}

/// Backwards-compatibility alias. New code should use [`PacingMode`].
#[deprecated(note = "use `PacingMode` — `RenderPacing` was a misleading name when only the sim loop used it")]
pub type RenderPacing = PacingMode;

/// Configuration for spawning the engine.
pub struct EngineConfig {
    /// Initial render width.
    pub width: u32,
    /// Initial render height.
    pub height: u32,
    /// Sim tick-loop pacing. Drives ECS, physics, behavior, animation,
    /// snapshot construction. `TargetHz(60)` is the editor default;
    /// games typically run sim at a fixed step (60 or 120 Hz).
    pub sim_pacing: PacingMode,
    /// Render thread pacing. Independent of sim. When `render_pacing`'s
    /// rate exceeds `sim_pacing`'s, render interpolates between the
    /// last two snapshots so visuals stay smooth at the higher rate
    /// instead of strobing the same sim state. `TargetHz(60)` is the
    /// editor default; games can set `Uncapped` or `TargetHz(144)` /
    /// `TargetHz(240)` to match a high-refresh display.
    pub render_pacing: PacingMode,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            // Sim caps at 60 Hz: physics + behavior FixedUpdate both
            // accumulate against fixed 1/60 steps, so an uncapped sim
            // would spin doing zero work most ticks. 60 Hz matches
            // the fixed-step rate exactly — every tick produces one
            // physics step on average.
            sim_pacing: PacingMode::TargetHz(60),
            // Render is uncapped by default. This is a game engine —
            // players on 240 Hz monitors should get 240 fps. The
            // render thread interpolates between sim snapshots so
            // visuals stay smooth even though sim is locked at 60 Hz.
            // Editor / dev tooling can override to TargetHz(N) if
            // they want a softer cap (battery, fans, etc.).
            render_pacing: PacingMode::Uncapped,
        }
    }
}

/// The Arrvox game engine.
///
/// Created via [`ArvxEngine::spawn`], which starts the engine on a background thread.
/// The caller communicates via the command channel and receives state via callbacks.
pub struct ArvxEngine {
    /// Handle to the engine thread.
    thread: Option<JoinHandle<()>>,
    /// Send commands to the engine.
    pub cmd_tx: crossbeam::channel::Sender<EngineCommand>,
}

impl ArvxEngine {
    /// Spawn the engine on a background thread.
    ///
    /// - `frame_callback`: called each tick with RGBA8 pixels (`width * height * 4` bytes)
    /// - `state_callback`: called each tick with current engine state
    pub fn spawn(
        config: EngineConfig,
        frame_callback: FrameCallback,
        state_callback: StateCallback,
    ) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded();

        let thread = std::thread::Builder::new()
            .name("arvx-engine".into())
            .spawn(move || {
                tick_loop(cmd_rx, frame_callback, state_callback, config);
            })
            .expect("failed to spawn engine thread");

        Self {
            thread: Some(thread),
            cmd_tx,
        }
    }

    /// Send a command to the engine (non-blocking).
    pub fn send(&self, cmd: EngineCommand) {
        let _ = self.cmd_tx.send(cmd);
    }
}

impl Drop for ArvxEngine {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(EngineCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Camera state tracked by the engine.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    pub position: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: glam::Vec3::new(0.0, 2.0, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0,
            near: 0.01,
            far: 1000.0,
        }
    }
}
