//! Headless editor-path harness — drive the REAL sim+render+integrate loop
//! windowless and assert on it, so render/load/sculpt fixes are verifiable
//! WITHOUT a human watching the GUI editor.
//!
//! This is the foundation the terrain/sculpt work has been missing: every
//! prior fix was only confirmable by the user running the editor and looking,
//! so regressions slipped back in silently. The engine already renders
//! offscreen on a headless wgpu device (`RenderContext::new_headless`) and
//! ships the exact composite the editor displays through `FrameCallback` — so
//! a `cargo test` can spawn the whole engine, drive it via `EngineCommand`,
//! and assert on frame cadence (freeze), pixels, and (later) mesh geometry.
//!
//! Drive surface (no GUI, no MCP):
//!   - IN:  typed `EngineCommand` over `engine.cmd_tx`
//!   - OUT: `FrameCallback` (RGBA8 composite) + `StateCallback` (StateUpdate)
//!
//! Discipline: the engine free-runs on its own thread with no completion ack,
//! so every assertion is wait-on-condition with a deadline. A TIMEOUT IS THE
//! FREEZE SIGNAL. Never assert absolute fps or sleep-then-check; assert "N more
//! frames observed within budget".
//!
//! Run with: `ARVX_TERRAIN_PROFILE=1 cargo test -p arvx-engine --release`
//! (release per project policy — debug cadence is unrepresentative for freeze timing).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arvx_engine::command::EngineCommand;
use arvx_engine::engine::{ArvxEngine, EngineConfig};

/// Poll `cond` every 10 ms until it returns true or `deadline` elapses.
/// Returns true if the condition was met (false = timed out = a freeze/stall).
fn wait_until(deadline: Duration, cond: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    cond()
}

/// Observables filled by the engine callbacks from its own thread.
struct Observed {
    /// Monotonic count of composite frames shipped to the (would-be) display.
    /// Stops rising iff the render thread stops shipping — i.e. a freeze.
    frames: Arc<AtomicU64>,
    /// Latest `StateUpdate.gpu_object_count` — non-zero once terrain/assets land.
    gpu_objects: Arc<AtomicU32>,
}

impl Observed {
    fn spawn_engine(width: u32, height: u32) -> (ArvxEngine, Observed) {
        let frames = Arc::new(AtomicU64::new(0));
        let gpu_objects = Arc::new(AtomicU32::new(0));

        let f = frames.clone();
        let g = gpu_objects.clone();

        let engine = ArvxEngine::spawn(
            EngineConfig {
                width,
                height,
                ..Default::default()
            },
            Box::new(move |_vp, _pixels, _w, _h| {
                f.fetch_add(1, Ordering::Relaxed);
            }),
            Box::new(move |state| {
                g.store(state.gpu_object_count, Ordering::Relaxed);
            }),
        );

        (engine, Observed { frames, gpu_objects })
    }

    fn frames(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }
    fn gpu_objects(&self) -> u32 {
        self.gpu_objects.load(Ordering::Relaxed)
    }
}

/// Spawning the engine headless must produce rendered frames — proves the
/// whole sim+render pipeline stands up on a headless wgpu device under `cargo test`.
#[test]
fn headless_engine_renders_frames() {
    let (_engine, obs) = Observed::spawn_engine(640, 360);
    assert!(
        wait_until(Duration::from_secs(60), || obs.frames() > 0),
        "engine never produced a first frame within 60s — headless render init failed"
    );
}

/// Generating terrain must not freeze the renderer. The render thread must keep
/// shipping frames through and after the integrate burst. A freeze (gate wedge /
/// unpaced integrate holding the loop) shows up as the frame counter going flat.
///
/// NOTE: this is the COLD-gen path (no `.arvxtile` cache). The confirmed
/// worst-case freeze is the WARM-cache reload burst; that repro (gen -> save ->
/// reload) is the next test to add once this proves the harness end-to-end.
#[test]
fn terrain_gen_keeps_rendering() {
    // Surface the machine-parseable freeze canaries on stderr. Safe: set
    // before the engine thread is spawned, and tests run single-threaded.
    unsafe { std::env::set_var("ARVX_TERRAIN_PROFILE", "1") };

    let (engine, obs) = Observed::spawn_engine(640, 360);

    assert!(
        wait_until(Duration::from_secs(60), || obs.frames() > 0),
        "engine never produced a first frame within 60s"
    );

    engine
        .cmd_tx
        .send(EngineCommand::SpawnTerrain)
        .expect("send SpawnTerrain");

    assert!(
        wait_until(Duration::from_secs(60), || obs.gpu_objects() >= 1),
        "terrain never integrated a tile within 60s (bake/integrate stall) — gpu_objects={}",
        obs.gpu_objects()
    );

    // The render thread must keep shipping frames through/after the burst.
    let base = obs.frames();
    let advanced = wait_until(Duration::from_secs(15), || obs.frames() >= base + 30);
    assert!(
        advanced,
        "render FROZE during/after terrain integrate: only {} new frames in 15s \
         (gpu_objects={})",
        obs.frames() - base,
        obs.gpu_objects()
    );
}
