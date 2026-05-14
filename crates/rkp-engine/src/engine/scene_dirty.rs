//! Per-entity dirty tracking for the UI / scene-tree snapshot.
//!
//! Replaces the prior `scene_dirty: bool` flag with the same
//! HashSet+sticky-all shape used by
//! [`super::gpu_objects_dirty::GpuObjectsDirty`] and
//! [`super::geometry_dirty::GeometryDirty`]. Tracks which entities'
//! `SceneObjectInfo` (id / name / parent_id / tree_order / kind bits)
//! needs republishing to the editor on the next `build_state_update`.
//!
//! ## Phase B3 status
//!
//! Plumbing only — today's consumer in `state_update.rs` still does a
//! full sorted rebuild of every `EditorMetadata` entity whenever
//! `is_dirty()` returns true. The per-entity scope carried here is
//! foundation for a future "narrow" pass that drains specific
//! Added/Removed/Renamed/Reparented events into an arc-shared
//! `SceneObjectInfo` list, sending only the deltas across the
//! sim→editor protocol.
//!
//! See `docs/PERF_DEBT.md` B3.

use std::collections::HashSet;

use hecs::Entity;

/// Records which entities' editor snapshot rows need rebuilding on
/// the next call to `build_state_update`.
///
/// * `entities` — narrow events. Spawn, delete, duplicate, rename,
///   reparent, component (light / camera / procedural) add or remove
///   on a single entity.
/// * `all` — sticky world-level event. Project / scene load, scene
///   clear, gameplay mode toggle, generator regen — anything that
///   touches the entity set wholesale. Once set, [`Self::is_all`]
///   stays true until [`Self::clear`].
#[derive(Debug, Default, Clone)]
pub(crate) struct SceneDirty {
    entities: HashSet<Entity>,
    all: bool,
}

impl SceneDirty {
    /// Empty / clean — nothing to rebuild.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mark a single entity dirty. No-op once `all` is set — the
    /// future full rebuild covers this entity anyway, and keeping
    /// it would just bloat the set.
    pub(crate) fn mark_entity(&mut self, entity: Entity) {
        if self.all {
            return;
        }
        self.entities.insert(entity);
    }

    /// Mark every entity as needing republish. Drops the per-entity
    /// set since it's now redundant. Use for world-level events:
    /// project / scene load, scene clear, gameplay reset.
    pub(crate) fn mark_all(&mut self) {
        self.all = true;
        self.entities.clear();
    }

    /// True iff anything is dirty — either at least one entity is
    /// flagged or [`Self::mark_all`] has been called since the last
    /// [`Self::clear`].
    pub(crate) fn is_dirty(&self) -> bool {
        self.all || !self.entities.is_empty()
    }

    /// True iff [`Self::mark_all`] has been called.
    #[allow(dead_code)] // Plumbing — consumer will use this once narrow path lands.
    pub(crate) fn is_all(&self) -> bool {
        self.all
    }

    /// The set of individually-flagged entities. Empty when nothing
    /// narrow is pending OR when `all` is set.
    #[allow(dead_code)] // Plumbing — consumer will use this once narrow path lands.
    pub(crate) fn dirty_entities(&self) -> &HashSet<Entity> {
        &self.entities
    }

    /// Reset to clean. Called by the consumer after publishing the
    /// fresh snapshot so subsequent ticks observe a fresh-empty state.
    pub(crate) fn clear(&mut self) {
        self.all = false;
        self.entities.clear();
    }
}
