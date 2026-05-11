//! Render thread — owns wgpu, consumes [`RenderFrame`] snapshots from
//! sim, returns [`RenderResult`] back.
//!
//! ## Module layout (post-split)
//!
//! - [`state`] — `RenderWorker`, `RenderInbox`, `RenderState`, `FrameCallback`.
//! - [`loop_thread`] — `run_render_thread` main loop + interpolation helpers.
//! - [`frame`] — `render_one_frame` orchestration (~800 lines).
//! - [`frame_helpers`] — small frame helpers (tile-list splice, AABB
//!   transforms, shadow-map setup).
//!
//! Public API surface (`crate::render_worker::*`) is stable: only
//! `RenderWorker` + `RenderInbox` are exposed externally.
//!
//! ## Architecture overview
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
//! ## Interpolation
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
//! ## Ownership
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

mod frame;
mod frame_helpers;
mod loop_thread;
mod state;
mod user_shader_mesh_tick;

pub use state::{FrameCallback, RenderInbox, RenderWorker};
