//! Sculpt-command handling.
//!
//! Phase 0 (input + UX): the editor emits [`EngineCommand::SculptAtPixel`]
//! for every brush stamp along a stroke. The pick readback routes the
//! result here through [`EngineState::apply_sculpt_stamp`], which gates
//! on selection / procedural / generator-owned / skinned and (for now)
//! just logs the resolved brush op.
//!
//! Phase 1 swaps the stub for real octree mutation: the brush AABB will
//! walk the octree, transition Empty↔Mixed cells under a smoothstep
//! sphere, and emit a `SculptDelta { dirty_clusters, dirty_leaves, … }`
//! that downstream phases consume for the DAG re-bake.

use glam::Vec3;

use crate::command::SculptMode;
use crate::components::{ProceduralGeometry, Renderable, Skeleton};
use crate::generator::GeneratorOwned;

use super::state::EngineState;

impl EngineState {
    /// Apply a single sculpt brush stamp to a known entity. Phase 0 is
    /// a stub — it runs every gate the real path will (selection /
    /// procedural / generator-owned / skinned / asset-backed) and logs
    /// the resolved op. Returns 0 on a gated stamp, 1 when the op
    /// would have been applied. Phase 1 swaps the body for actual
    /// octree mutation and returns the number of transitioned leaves.
    pub(crate) fn apply_sculpt_stamp(
        &mut self,
        entity: hecs::Entity,
        world_pos: Vec3,
        radius: f32,
        falloff: f32,
        mode: SculptMode,
        material_id: u16,
    ) -> usize {
        // ── Selection gate ──
        // Selection-locked like paint — see `apply_paint_stamp` for
        // rationale. A picked surface on something other than the
        // selected entity is a no-op, not a deselect.
        if self.selected_entity != Some(entity) {
            return 0;
        }

        // ── Procedural / generator-owned gates ──
        // Sculpting a procedural would contradict the procedural
        // definition — the next bake would overwrite the carved
        // geometry. Generator children are re-emitted on every run, so
        // the same caveat applies.
        if self.world.get::<&ProceduralGeometry>(entity).is_ok() {
            self.console.warn(
                "Sculpt on procedural entity skipped — geometry is regenerated \
                 on rebake.".to_string(),
            );
            return 0;
        }
        if self.world.get::<&GeneratorOwned>(entity).is_ok() {
            self.console.warn(
                "Sculpt on generator-emitted entity skipped — generators \
                 re-emit their children on every run.".to_string(),
            );
            return 0;
        }

        // ── Skinned gate ──
        // V1 doesn't support sculpting skinned characters — would
        // require rest-pose octree edits + skin re-apply. Flagged as
        // future work in the sculpt POC plan.
        if self.world.get::<&Skeleton>(entity).is_ok() {
            self.console.warn(
                "Sculpt on skinned entity skipped — sculpting characters \
                 isn't supported in V1.".to_string(),
            );
            return 0;
        }

        // ── Entity must have an octree (asset-backed Renderable) ──
        let has_octree = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| r.spatial.as_ref().and_then(|g| g.as_octree()).map(|_| ()))
            .is_some();
        if !has_octree {
            return 0;
        }

        // Phase 0: log the op so manual tests can confirm the wiring
        // end-to-end. Phase 1 replaces this body with actual mutation.
        eprintln!(
            "[sculpt] stub stamp entity={:?} pos=({:.3}, {:.3}, {:.3}) \
             radius={:.3} falloff={:.3} mode={:?} material={}",
            entity, world_pos.x, world_pos.y, world_pos.z,
            radius, falloff, mode, material_id,
        );
        1
    }
}

/// Route the legacy [`EngineCommand::Sculpt`] (world-position variant)
/// to [`EngineState::apply_sculpt_stamp`]. Used by tests + any caller
/// that has already resolved the hit point; the editor's UI flow takes
/// the `SculptAtPixel` → pick-readback path instead.
pub(crate) fn dispatch_sculpt(
    state: &mut EngineState,
    position: Vec3,
    _normal: Vec3,
    radius: f32,
    _strength: f32,
    mode: SculptMode,
) {
    let Some(entity) = state.selected_entity else { return };
    let material_id = state.selected_material.unwrap_or(0);
    let _ = state.apply_sculpt_stamp(entity, position, radius, 0.5, mode, material_id);
}
