//! Per-entity dirty tracking for geometry changes that drive
//! collider-cache rebuilds.
//!
//! Replaces the prior pair of `geometry_dirty: bool` /
//! `collider_caches_dirty: bool` flags with a single per-entity set.
//! `rebuild_collider_caches` previously walked every `RigidBody`
//! entity in the world whenever any single voxel got carved or any
//! single transform changed; this type lets the consumer iterate
//! only the entities that actually changed.
//!
//! ## API mirror to [`super::gpu_objects_dirty::GpuObjectsDirty`]
//!
//! Same shape: a per-entity HashSet plus a sticky `all` bit for
//! world-level events (project load, scene clear). The collider
//! pipeline doesn't need the `DirtyKind` taxonomy
//! (`Transform`/`Structural`) that GPU-objects does — every
//! geometry change is structural for collider purposes (the tight
//! AABB and coarse-collider voxel set both depend on octree
//! contents). PERF_DEBT.md B2+C3.

use std::collections::HashSet;

use hecs::Entity;

/// Records which entities' colliders need rebuilding on the next
/// call to `rebuild_collider_caches`.
///
/// * `entities` — narrow events (sculpt/paint mutating one asset,
///   RigidBody component added/modified on a single entity).
/// * `all` — sticky world-level event (project load, scene clear,
///   procedural bake completion when many entities may be affected).
///   Once set, `is_all` stays true until `clear`.
#[derive(Debug, Default, Clone)]
pub(crate) struct GeometryDirty {
    entities: HashSet<Entity>,
    all: bool,
}

impl GeometryDirty {
    /// Empty / clean — nothing to rebuild.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mark a single entity dirty. No-op once `all` is set.
    pub(crate) fn mark_entity(&mut self, entity: Entity) {
        if self.all {
            return;
        }
        self.entities.insert(entity);
    }

    /// Mark every entity dirty. Drops the per-entity set.
    pub(crate) fn mark_all(&mut self) {
        self.all = true;
        self.entities.clear();
    }

    /// True iff anything is dirty.
    pub(crate) fn is_dirty(&self) -> bool {
        self.all || !self.entities.is_empty()
    }

    /// True iff [`Self::mark_all`] has been called — caller iterates
    /// every `RigidBody` entity in the world rather than the
    /// per-entity set.
    pub(crate) fn is_all(&self) -> bool {
        self.all
    }

    /// The set of individually-flagged entities. Empty when nothing
    /// narrow is pending OR when `all` is set (caller should iterate
    /// the world directly in the `all` case).
    pub(crate) fn dirty_entities(&self) -> &HashSet<Entity> {
        &self.entities
    }

    /// Reset to clean. Called by the consumer after a successful
    /// rebuild.
    pub(crate) fn clear(&mut self) {
        self.all = false;
        self.entities.clear();
    }
}
