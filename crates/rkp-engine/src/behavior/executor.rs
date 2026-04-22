//! Behavior executor — runs gameplay systems per frame.
//!
//! The executor owns the schedule and drives the per-frame loop:
//! 1. `tick()`: Update phase → flush → FixedUpdate phase → flush
//! 2. (engine steps physics externally)
//! 3. `tick_late()`: LateUpdate phase → flush → drain events
//!
//! Each system is wrapped in `catch_unwind` for panic isolation.

use std::collections::HashSet;
use std::time::Instant;

use super::command_queue::CommandQueue;
use super::engine_access::{TransformUpdate, WorldEngineAccess};
use super::game_store::GameStore;
use super::scheduler::{Schedule, ScheduleError, build_schedule};
use super::system_context::SystemContext;
use super::system_entry::SystemEntry;

/// Runs gameplay systems in scheduled order with panic recovery.
pub struct BehaviorExecutor {
    schedule: Schedule,
    faulted_systems: HashSet<usize>,
    system_timings: Vec<Option<u64>>,
}

impl BehaviorExecutor {
    /// Build an executor from a list of system entries.
    pub fn new(systems: &[&SystemEntry]) -> Result<Self, ScheduleError> {
        let schedule = build_schedule(systems)?;
        let count = systems.len();
        Ok(Self {
            schedule,
            faulted_systems: HashSet::new(),
            system_timings: vec![None; count],
        })
    }

    /// Rebuild the schedule (e.g., after hot-reload changes systems).
    pub fn rebuild(&mut self, systems: &[&SystemEntry]) -> Result<(), ScheduleError> {
        self.schedule = build_schedule(systems)?;
        self.faulted_systems.clear();
        self.system_timings = vec![None; systems.len()];
        Ok(())
    }

    /// Run the variable-rate `Update` phase. Call once per render
    /// frame with the real wall-clock delta. Reset per-system timings
    /// at the start so the panel only shows the most recent frame's
    /// numbers (FixedUpdate accumulates into the same slot below; the
    /// reset here ensures we don't carry timings from a frame that
    /// happened to skip Update).
    pub fn tick_update(
        &mut self,
        systems: &[&SystemEntry],
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        for t in self.system_timings.iter_mut() {
            *t = None;
        }
        self.run_phase(&self.schedule.update.clone(), systems, world, commands, store, delta_time, total_time, frame);
        commands.flush(world);
    }

    /// Run a single `FixedUpdate` step. Call N times per render frame
    /// from an accumulator-driven loop in `tick_loop`. `delta_time`
    /// must be the fixed step (typically 1/60) — mismatched values
    /// will desync any system that integrates state.
    pub fn tick_fixed_update(
        &mut self,
        systems: &[&SystemEntry],
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        self.run_phase(&self.schedule.fixed_update.clone(), systems, world, commands, store, delta_time, total_time, frame);
        commands.flush(world);
    }

    /// Run LateUpdate phase + drain events. Call after physics.
    pub fn tick_late(
        &mut self,
        systems: &[&SystemEntry],
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        self.run_phase(&self.schedule.late_update.clone(), systems, world, commands, store, delta_time, total_time, frame);
        commands.flush(world);
        store.drain_events();
    }

    /// Clear fault tracking, re-enabling all systems.
    pub fn clear_faults(&mut self) {
        self.faulted_systems.clear();
    }

    /// Per-system execution time in microseconds (None if skipped/faulted).
    pub fn system_timing(&self, index: usize) -> Option<u64> {
        self.system_timings.get(index).copied().flatten()
    }

    /// Whether a system is faulted (panicked and disabled).
    pub fn is_faulted(&self, index: usize) -> bool {
        self.faulted_systems.contains(&index)
    }

    fn run_phase(
        &mut self,
        phase_indices: &[usize],
        systems: &[&SystemEntry],
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        for &idx in phase_indices {
            if self.faulted_systems.contains(&idx) {
                continue;
            }

            let meta = &systems[idx];

            // SAFETY: fn_ptr was produced by casting `fn(&mut SystemContext)` to
            // `*const ()` during system registration. The #[rkp_system] proc macro
            // guarantees this invariant.
            let system_fn: fn(&mut SystemContext) = unsafe {
                std::mem::transmute(meta.fn_ptr)
            };

            // SAFETY: Raw pointer for read-only EngineAccess. Derived before the
            // &mut borrow for SystemContext. EngineAccess methods only read from
            // the world (no mutation through this pointer).
            let world_ptr: *const hecs::World = &raw const *world;
            let engine_access = unsafe { WorldEngineAccess::new(world_ptr) };

            let mut ctx = SystemContext::new(
                world, commands, store, &engine_access,
                delta_time, total_time, frame,
            );

            let start = Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                system_fn(&mut ctx);
            }));
            let elapsed_us = start.elapsed().as_micros() as u64;

            let pending_updates = ctx.take_transform_updates();
            drop(ctx);

            if result.is_err() {
                eprintln!("[BehaviorExecutor] system '{}' panicked — disabling", meta.name);
                self.faulted_systems.insert(idx);
            } else {
                apply_transform_updates(world, pending_updates);
                if idx < self.system_timings.len() {
                    self.system_timings[idx] = Some(elapsed_us);
                }
            }
        }
    }
}

/// Apply buffered transform updates to the world.
fn apply_transform_updates(world: &mut hecs::World, updates: Vec<TransformUpdate>) {
    for update in updates {
        if let Ok(mut t) = world.get::<&mut crate::components::Transform>(update.entity) {
            if let Some(pos) = update.position {
                t.position = pos;
            }
            if let Some(rot) = update.rotation {
                t.rotation = rot;
            }
            if let Some(scale) = update.scale {
                t.scale = scale;
            }
        }
    }
}
