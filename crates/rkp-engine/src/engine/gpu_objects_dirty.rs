//! Per-entity dirty tracking for the GPU-objects derivation
//! ([`update_scene_gpu`](super::scene_gpu)).
//!
//! The previous design used a single `bool` flag, which gave the
//! consumer no way to do anything other than a full O(N) rebuild of
//! every derived structure (gpu_assets, gpu_instances, overlays,
//! sculpts, splat_draws, proxy_draws, gpu_to_entity, entity_to_gpu)
//! whenever any single entity changed. On a 22-entity scene a single
//! sculpt stamp would therefore re-process all 22 entities — wasting
//! 60-75 ms per stamp.
//!
//! This type carries the precise scope of each mutation: a set of
//! entities whose derived rows need rebuilding plus a sticky `all`
//! bit for world-level events that genuinely invalidate everything
//! (project load, scene clear, gameplay reset). PERF_DEBT.md A1+B1.
//!
//! ## Today's consumer
//!
//! The current `update_scene_gpu` still does a full rebuild whenever
//! [`Self::is_dirty`] returns true — the per-entity hash set carries
//! information that no consumer reads yet. C2 wires the per-row
//! rebuild path that actually iterates [`Self::dirty_entities`] for
//! the perf win; this type lands the data plumbing first so all 31
//! setter sites converge on a single API before the consumer changes.

use std::collections::HashSet;

use hecs::Entity;

/// Records which entities' GPU-object rows need rebuilding on the
/// next call to [`update_scene_gpu`](super::scene_gpu).
///
/// Two independent dirty signals coexist:
///
/// * `entities` — narrow, entity-scoped events. Sculpt/paint/gizmo
///   stamps push individual entities here.
/// * `all` — sticky world-level event. Project load, scene clear,
///   asset (un)load: anything that genuinely invalidates the asset
///   table or the entity set itself. Once set, [`Self::is_all`]
///   stays true until [`Self::clear`].
///
/// Any caller can check [`Self::is_dirty`] (true iff `all` OR
/// non-empty `entities`) for the "needs rebuild?" question, and
/// [`Self::is_all`] for the "needs full rebuild?" question. The
/// per-row consumer (C2) iterates [`Self::dirty_entities`].
#[derive(Debug, Default, Clone)]
pub(crate) struct GpuObjectsDirty {
    entities: HashSet<Entity>,
    all: bool,
}

impl GpuObjectsDirty {
    /// Empty / clean — nothing to rebuild.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mark a single entity as needing its derived row rebuilt. No-op
    /// once `all` is set — the future full rebuild will cover this
    /// entity anyway, and storing it would just inflate the set.
    pub(crate) fn mark_entity(&mut self, entity: Entity) {
        if self.all {
            return;
        }
        self.entities.insert(entity);
    }

    /// Mark every entity as needing a full rebuild. Drops the
    /// per-entity set since it's now redundant. Use for world-level
    /// events: project load, scene clear, asset (un)load, gameplay
    /// reset, anything that touches the asset table or the set of
    /// entities itself.
    pub(crate) fn mark_all(&mut self) {
        self.all = true;
        self.entities.clear();
    }

    /// True iff anything is dirty — either at least one entity is
    /// flagged or [`Self::mark_all`] has been called since the last
    /// [`Self::clear`]. The `update_scene_gpu` consumer gates on this.
    pub(crate) fn is_dirty(&self) -> bool {
        self.all || !self.entities.is_empty()
    }

    /// True iff [`Self::mark_all`] has been called — caller must do
    /// a full rebuild rather than per-entity iteration. Future C2
    /// per-row code will branch on this to decide between fast path
    /// and full rebuild.
    pub(crate) fn is_all(&self) -> bool {
        self.all
    }

    /// The set of individually-flagged entities. Empty when nothing
    /// narrow is pending OR when `all` is set (in which case the
    /// caller should iterate the world directly).
    pub(crate) fn dirty_entities(&self) -> &HashSet<Entity> {
        &self.entities
    }

    /// Reset to clean. Called by the consumer after a successful
    /// rebuild so subsequent ticks observe a fresh-empty state.
    pub(crate) fn clear(&mut self) {
        self.all = false;
        self.entities.clear();
    }
}
