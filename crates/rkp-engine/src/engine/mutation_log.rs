//! Typed mutation log — every CPU-side mutation the sim makes describes
//! its scope as a [`MutationEvent`]. Consumers replay events to update
//! their derived state incrementally instead of doing full-world
//! rebuilds.
//!
//! # Why
//!
//! Before this scaffolding, every mutation set a coarse `dirty: bool`
//! flag (gpu_objects_dirty, geometry_dirty, scene_dirty, …) and its
//! consumer interpreted the flag as "rebuild everything." On a 22-entity
//! scene a single sculpt stamp rebuilt all GPU instances + walked every
//! entity's painted-material cache + recomputed every rigid body's
//! collider AABB — ~240 ms of sim work per stamp.
//!
//! `MutationEvent` carries the scope (which entity, which region) at
//! the call site. Phase B/C migrations will replace the boolean flags
//! with per-entity dirty sets driven by the log; consumers will iterate
//! the changed entities, not the world.
//!
//! # Phase A1 status
//!
//! This module is the scaffolding only. Mutation sites push events;
//! no consumers are wired yet. Existing `*_dirty` flags stay until
//! their replacements land in later phases. `MutationLog::drain` runs
//! at the top of `submit_render_frame` to discard events the consumers
//! haven't subscribed to yet — this prevents the log from growing
//! unboundedly while we incrementally adopt it.
//!
//! See `docs/PERF_DEBT.md` for the migration plan.

use hecs::Entity;

use crate::command::{PaintMode, SculptMode};

/// One scope-carrying mutation event. Replaces the meaning of a
/// coarse boolean dirty flag.
///
/// **Scope discipline**: every variant identifies the smallest scope
/// the mutation affects — usually an entity, sometimes an asset
/// handle, occasionally a "world reset" sentinel for project-level
/// changes. Consumers translate this into the updates they need to
/// apply to their own derived state.
#[derive(Debug, Clone)]
pub enum MutationEvent {
    /// Entity's octree geometry was mutated by a sculpt stamp. The
    /// affected leaf-attr slots aren't carried yet — Phase C1
    /// (incremental `painted_per_entity` maintenance) will extend
    /// the variant with `added_leaf_slots` / `removed_leaf_slots`
    /// once the sculpt path can report them cheaply.
    SculptStamp {
        entity: Entity,
        mode: SculptMode,
        /// Brush material — relevant for Raise mode to detect whether
        /// the stamp may have introduced a shader-bearing material.
        material_id: u16,
    },
    /// Entity's paint overlay was updated by a paint stamp. The
    /// material may differ between modes:
    ///   - `Material` — `material_id` is the painted material.
    ///   - `Color` / `Erase` — `material_id` is informational; the
    ///     leaf's resolved material doesn't change.
    PaintStamp {
        entity: Entity,
        mode: PaintMode,
        material_id: u16,
    },
    /// New entity added to the ECS world. Consumers add the entity
    /// to whatever per-entity structures they maintain.
    EntityAdded { entity: Entity },
    /// Entity removed from the ECS world. Consumers drop their
    /// per-entity cache entries.
    EntityRemoved { entity: Entity },
    /// Project-level reset (scene load, clear, close). Consumers
    /// drop ALL their per-entity state and re-bootstrap.
    WorldReset,
}

/// Append-only log of `MutationEvent`s accumulated during a sim tick.
///
/// Sim writes; consumers drain at the start of the next
/// `submit_render_frame`. Phase A1 has no consumers — `drain_and_log`
/// just empties the log so it doesn't grow unbounded. As Phase B/C
/// land, real consumers replace the drop with an event-driven
/// rebuild of their derived state.
#[derive(Debug, Default)]
pub struct MutationLog {
    events: Vec<MutationEvent>,
}

impl MutationLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a mutation event. O(1) amortized.
    pub fn push(&mut self, event: MutationEvent) {
        self.events.push(event);
    }

    /// Return `true` when there are no pending events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Pending event count. Used by the Phase-A1 telemetry log so we
    /// can confirm events are being pushed at the expected rate.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Drop all pending events. Phase A1 calls this at the start of
    /// each `submit_render_frame` — once consumers are wired (Phase
    /// B+), they call [`Self::drain_events`] instead and process the
    /// events before this clears them.
    pub fn drain_and_log(&mut self) {
        if !self.events.is_empty() {
            eprintln!(
                "[mutation-log] draining {} event(s) (Phase A1 — no consumers wired yet)",
                self.events.len(),
            );
            self.events.clear();
        }
    }

    /// Drain pending events for a consumer to process. After this
    /// returns, the log is empty. Phase B+ consumers use this; Phase
    /// A1 has no consumers and uses [`Self::drain_and_log`] instead.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, MutationEvent> {
        self.events.drain(..)
    }
}
