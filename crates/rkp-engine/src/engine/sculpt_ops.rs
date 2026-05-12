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

        // ── Entity must be asset-backed (octree + asset_handle). ─
        let (asset_handle, entity_world) = {
            let renderable = match self.world.get::<&Renderable>(entity) {
                Ok(r) => r,
                Err(_) => return 0,
            };
            let Some(handle) = renderable.asset_handle else {
                // Procedurally-baked voxels carry a SpatialData but no
                // AssetHandle. Sculpt is asset-only for V1 (procedural
                // mutation belongs in the procedural tree, not the
                // post-bake octree).
                return 0;
            };
            if renderable.spatial.as_ref().and_then(|g| g.as_octree()).is_none() {
                return 0;
            }
            let transform = match self.world.get::<&crate::components::Transform>(entity) {
                Ok(t) => t,
                Err(_) => return 0,
            };
            let entity_world = glam::Affine3A::from_scale_rotation_translation(
                transform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    transform.rotation.x.to_radians(),
                    transform.rotation.y.to_radians(),
                    transform.rotation.z.to_radians(),
                ),
                transform.position,
            );
            (handle, entity_world)
        };

        // ── Engine enum → core enum. Smooth / Flatten are V2 — bail
        // with a warning until those land.
        let brush_mode = match mode {
            SculptMode::Raise => rkp_core::sculpt::BrushMode::Raise,
            SculptMode::Carve => rkp_core::sculpt::BrushMode::Carve,
            SculptMode::Smooth | SculptMode::Flatten => {
                self.console.warn(format!(
                    "Sculpt mode {mode:?} not implemented yet — Raise / Carve only in V1.",
                ));
                return 0;
            }
        };

        // ── Mutate the asset's octree + GPU buffers under the lock. ─
        let result = {
            let mut scene = self.scene_mgr.lock().expect("scene_mgr poisoned");
            scene.apply_sculpt_brush(
                asset_handle,
                world_pos,
                entity_world,
                radius,
                falloff,
                brush_mode,
                material_id,
            )
        };

        let Some(result) = result else {
            // No edits — brush outside, or every cell under the brush
            // was Interior / unchanged (Phase 1 semantics).
            return 0;
        };

        // ── Refresh the entity's cached Renderable.spatial with the
        // new GPU octree handle + leaf-attr range. Without this the
        // shader keeps reading the deallocated old octree slot range,
        // which is a vivid "rendering goes black / wrong" bug.
        {
            use rkp_core::scene_node::SpatialHandle;
            use crate::components::{RenderGeometry, SpatialData};
            if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
                let info = &result.new_info;
                if let SpatialHandle::Octree { root_offset, len, depth, base_voxel_size } = info.spatial {
                    let extent = (1u32 << depth) as f32 * base_voxel_size;
                    let aabb_center = (info.aabb.min + info.aabb.max) * 0.5;
                    let grid_origin = aabb_center - glam::Vec3::splat(extent * 0.5);
                    renderable.spatial = Some(RenderGeometry::Octree(SpatialData {
                        root_offset,
                        len,
                        depth,
                        base_voxel_size,
                        aabb: info.aabb,
                        voxel_size: info.voxel_size,
                        grid_origin,
                        voxel_slot_start: info.leaf_attr_slot_start,
                        voxel_slot_count: info.leaf_attr_slot_count,
                        brick_ids: Vec::new(),
                    }));
                    renderable.voxel_count = info.voxel_count;
                }
            }
        }

        // Force the next tick to rebuild gpu_instances so the new
        // spatial handle propagates into RkpGpuInstance.
        self.gpu_objects_dirty = true;
        // Mark this entity's painted-material cache stale — slot ids
        // moved, the walk needs to re-scan.
        self.painted_dirty_entities.insert(entity);

        eprintln!(
            "[sculpt] stamp entity={:?} added={} removed={}",
            entity, result.leaves_added, result.leaves_removed,
        );

        result.leaves_added + result.leaves_removed
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
