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

use std::collections::HashMap;

use hecs::Entity;

/// What kind of change happened to an entity. Lets the consumer
/// pick between an in-place fast path (Transform — patch the
/// matching `gpu_instances` row's `world`) and a full rebuild
/// (Structural — sculpt overlay grew, paint stamp wrote, proxy
/// handle swapped, etc.). PERF_DEBT.md C2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirtyKind {
    /// Transform-only edit — entity's `Transform` component changed
    /// but its asset, materials, overlays, sculpts, and skinning
    /// state are unchanged. The fast path patches just the
    /// `RkpGpuInstance.world` matrix in place; everything else in
    /// `update_scene_gpu`'s output is bit-identical to the prior
    /// frame.
    Transform,
    /// Structural edit — overlay/sculpt added or changed, asset
    /// handle swapped, material changed, anything beyond pure
    /// transform. Today the consumer falls back to a full rebuild
    /// for any structural-dirty entity in the set; future phases
    /// (D2 per-entity overlay arenas, etc.) will let this be O(1)
    /// per change too.
    Structural,
}

impl DirtyKind {
    /// Coarsens a (transform, structural) pair into the strictest
    /// kind — `Structural > Transform`. Used when the same entity
    /// is marked twice in one frame: a Transform mark followed by
    /// a Structural mark must remain Structural.
    fn merge(self, other: DirtyKind) -> DirtyKind {
        match (self, other) {
            (DirtyKind::Structural, _) | (_, DirtyKind::Structural) => DirtyKind::Structural,
            _ => DirtyKind::Transform,
        }
    }
}

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
    /// Per-entity dirty tag — `DirtyKind` describes what kind of
    /// edit happened so the consumer can pick a fast path. Same
    /// entity marked twice in one frame coarsens to the strictest
    /// kind via `DirtyKind::merge`.
    entities: HashMap<Entity, DirtyKind>,
    all: bool,
}

impl GpuObjectsDirty {
    /// Empty / clean — nothing to rebuild.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mark a single entity dirty with the given kind. No-op once
    /// `all` is set — the future full rebuild will cover this entity
    /// anyway, and storing it would just inflate the map. If the
    /// entity is already marked, the kind coarsens to the stricter
    /// of the two (Structural > Transform).
    pub(crate) fn mark_entity_kind(&mut self, entity: Entity, kind: DirtyKind) {
        if self.all {
            return;
        }
        self.entities
            .entry(entity)
            .and_modify(|existing| *existing = existing.merge(kind))
            .or_insert(kind);
    }

    /// Convenience for the common Structural case (sculpt, paint,
    /// proc-bake, asset swap, etc.). Equivalent to
    /// `mark_entity_kind(e, DirtyKind::Structural)`.
    pub(crate) fn mark_entity(&mut self, entity: Entity) {
        self.mark_entity_kind(entity, DirtyKind::Structural);
    }

    /// Convenience for transform-only edits (gizmo drag,
    /// drag-preview snap). Equivalent to
    /// `mark_entity_kind(e, DirtyKind::Transform)`.
    pub(crate) fn mark_entity_transform(&mut self, entity: Entity) {
        self.mark_entity_kind(entity, DirtyKind::Transform);
    }

    /// Mark every entity as needing a full rebuild. Drops the
    /// per-entity map since it's now redundant. Use for world-level
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
    /// a full rebuild rather than per-entity iteration.
    pub(crate) fn is_all(&self) -> bool {
        self.all
    }

    /// The map of individually-flagged entities to their dirty kind.
    /// Empty when nothing narrow is pending OR when `all` is set (in
    /// which case the caller should iterate the world directly).
    pub(crate) fn dirty_entities(&self) -> &HashMap<Entity, DirtyKind> {
        &self.entities
    }

    /// True iff dirty AND every flagged entity is `Transform` — the
    /// per-row fast path can patch matrices in place without touching
    /// the asset table or the flat overlay/sculpt vecs. Returns false
    /// when `all` is set (full rebuild required) or any entity is
    /// `Structural`.
    pub(crate) fn is_transform_only(&self) -> bool {
        if self.all {
            return false;
        }
        if self.entities.is_empty() {
            return false;
        }
        self.entities
            .values()
            .all(|k| matches!(k, DirtyKind::Transform))
    }

    /// Reset to clean. Called by the consumer after a successful
    /// rebuild so subsequent ticks observe a fresh-empty state.
    pub(crate) fn clear(&mut self) {
        self.all = false;
        self.entities.clear();
    }
}
