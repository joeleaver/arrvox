//! Paint-command handling.
//!
//! The editor emits an [`EngineCommand::Paint`] for every brush stamp
//! along a stroke (begin / continue / end sample points). Each command
//! carries the world-space hit position, brush settings, and paint mode;
//! the engine resolves the entity under that position, guards against
//! procedural / generator-owned geometry, and delegates the actual
//! LeafAttrPool write to [`ArvxSceneManager::apply_paint_sphere`].
//!
//! Phase 1 — no UI. Phase 5 adds the floating paint palette; Phase 3
//! upgrades the brush footprint from euclidean sphere to geodesic surface
//! flood fill.

use glam::Vec3;

use crate::command::PaintMode;
use crate::components::{ProceduralGeometry, Renderable};
use crate::generator::GeneratorOwned;

use super::state::EngineState;

/// Window after the last paint stamp during which `ARVX_PAINT_PROFILE`
/// traces are emitted. Long enough to cover the sim ticks between
/// stamps in a drag (pick round-trip is typically a few frames),
/// short enough that idle silences within a beat of releasing.
const PAINT_PROFILE_WINDOW: std::time::Duration =
    std::time::Duration::from_millis(500);

impl EngineState {
    /// `true` when paint profiling traces should fire. Gated on the
    /// `ARVX_PAINT_PROFILE` env var AND a recent stamp activity, so
    /// hover / idle paint mode stay quiet.
    pub(crate) fn paint_profile_active(&self) -> bool {
        if std::env::var("ARVX_PAINT_PROFILE").is_err() {
            return false;
        }
        match self.last_paint_stamp_at {
            Some(t) => t.elapsed() < PAINT_PROFILE_WINDOW,
            None => false,
        }
    }
    /// Dispatch an [`EngineCommand::Paint`] stamp. Returns the number of
    /// leaves written (0 if the command was dropped — see [`Self::apply_paint_stamp`]).
    pub(crate) fn handle_paint_command(
        &mut self,
        position: Vec3,
        _normal: Vec3,
        radius: f32,
        color: [f32; 3],
        strength: f32,
        mode: PaintMode,
    ) -> usize {
        // Find the entity under the brush. Phase 1 uses AABB containment
        // — the editor's real flow will route a pick readback's
        // `(gpu_idx, world_pos)` into this path, but for headless
        // testing + the naive first pass, "nearest entity whose world
        // AABB contains the point" is good enough.
        let entity = match self.find_entity_at_world_pos(position) {
            Some(e) => e,
            None => return 0,
        };
        let material_id = self.selected_material.unwrap_or(0);
        self.apply_paint_stamp(
            entity, position, radius, strength, 0.5, color, mode, material_id,
        )
    }

    /// Apply a single brush stamp to a known entity. Separated from the
    /// command handler so unit tests and the editor's pick-readback
    /// flow (which already knows the entity from `gpu_to_entity`) can
    /// bypass the world-position → entity resolution.
    ///
    /// Returns the number of leaves written. Returns `0` (silently)
    /// when the entity isn't selected, or with a console warning
    /// when the entity is procedural / generator-owned — painting
    /// unselected objects would let a brushstroke "leak" through
    /// geometry the user isn't aiming at.
    pub(crate) fn apply_paint_stamp(
        &mut self,
        entity: hecs::Entity,
        world_pos: Vec3,
        radius: f32,
        strength: f32,
        falloff: f32,
        color: [f32; 3],
        mode: PaintMode,
        material_id: u16,
    ) -> usize {
        // ── Selection gate ──
        // Paint only acts on the currently selected entity. Picking
        // through a non-selected object returns 0 silently so casual
        // clicks don't deselect or paint unrelated geometry. The
        // cursor visualization is gated the same way (per-pixel
        // selection lock in `arvx_shade.wesl`).
        if self.selected_entity != Some(entity) {
            return 0;
        }
        // ── Procedural / generator gate ──
        if self.world.get::<&ProceduralGeometry>(entity).is_ok() {
            self.console.warn(
                "Paint on procedural entity skipped — voxels are regenerated \
                 on rebake, so paint wouldn't persist.".to_string(),
            );
            return 0;
        }
        if self.world.get::<&GeneratorOwned>(entity).is_ok() {
            self.console.warn(
                "Paint on generator-emitted entity skipped — generators \
                 re-emit their children on every run.".to_string(),
            );
            return 0;
        }

        // ── Resolve entity → AssetInfo + world transform ──
        let (asset_info, entity_world) = match self.build_paint_context(entity) {
            Some(ctx) => ctx,
            None => return 0,
        };

        let stamp = match mode {
            PaintMode::Material => {
                arvx_render::paint::PaintStamp::Material { material_id }
            }
            PaintMode::Color => arvx_render::paint::PaintStamp::Color { rgb: color },
            PaintMode::Erase => arvx_render::paint::PaintStamp::Erase,
        };

        let profile = std::env::var("ARVX_PAINT_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        // Stretch the profile-active window forward — covers the
        // sim ticks BETWEEN stamps too, so update_scene_gpu's
        // gap-since-last trace fires throughout the drag instead of
        // dropping out 500 ms after the most recent commit.
        if profile {
            self.last_paint_stamp_at = Some(t0);
        }
        let (written, overlay_len_after) = {
            // Take the per-entity overlay first so its `&mut`
            // borrow into `self.paint_overlays` doesn't fight the
            // scene_mgr lock. `entry().or_default()` lazily allocates
            // the overlay on the first stamp into this entity.
            let overlay = self.paint_overlays.entry(entity).or_default();
            let mut scene = self.scene_mgr.lock().expect("scene_mgr poisoned");
            let w = scene.apply_paint_sphere(
                &asset_info,
                entity_world,
                world_pos,
                radius,
                strength,
                falloff,
                stamp,
                overlay,
            );
            (w, overlay.len())
        };
        if profile && written > 0 {
            eprintln!(
                "[paint] stamp written={} overlay_len={} stamp_dt={:?}",
                written, overlay_len_after, t0.elapsed(),
            );
        }
        if written > 0 {
            // Bump the active-window anchor on actual stamps too, so
            // a long pause between stamps doesn't leak into idle
            // before the user resumes the drag.
            self.last_paint_stamp_at = Some(std::time::Instant::now());
            // PERF_DEBT.md D2: this stamp grew (or replaced) the
            // entity's paint overlay slice, so the concatenated
            // `gpu_instance_overlays` content the render side reads
            // will differ from last frame after `update_scene_gpu`
            // re-flattens. Marks the upload as dirty so the snapshot
            // ships a non-empty `DirtyRanges`; an idle tick without a
            // stamp leaves this false → render skips the overlay
            // upload entirely.
            self.gpu_instance_overlays_dirty = true;
            // Per-instance `overlay_offset` / `overlay_count` shift
            // every time the overlay vec grows (or, in theory,
            // shrinks via erase). Force a `gpu_instances` rebuild on
            // the next tick so the GPU side picks up the new slice
            // into `instance_overlay_buffer`.
            //
            // PERF_DEBT B1: only the painted entity's overlay
            // changed. Today's flat overlay layout means the
            // consumer still has to patch every subsequent entity's
            // offsets — C2 will decide between per-row patch +
            // suffix-shift or full rebuild based on this scope.
            self.gpu_objects_dirty.mark_entity(entity);
            // Mark this entity for an incremental painted-material
            // re-scan in the next lifecycle tick — but ONLY for
            // material-mode stamps. Color and Erase stamps keep the
            // leaf's `material_id` (and therefore its shader-material
            // membership) unchanged, so the walk's result can't move.
            // Skipping the dirty-mark on color drags keeps the
            // 60+ Hz drag path off the O(octree) walk entirely.
            if matches!(mode, PaintMode::Material) {
                self.painted_dirty_entities.insert(entity);
                // Phase C1: record the brush footprint (world space) so
                // the walk can scope its octree scan to this region
                // instead of walking the full entity octree. The walk
                // converts to object-local at consume time via the
                // entity's current Transform — decoupling stamp-time
                // and walk-time transforms.
                self.painted_dirty_regions
                    .entry(entity)
                    .or_default()
                    .push(arvx_core::Aabb::from_center_half_extents(
                        world_pos,
                        Vec3::splat(radius),
                    ));
            }

            // Push a scope-carrying mutation event. Phase A1 scaffolding;
            // see docs/PERF_DEBT.md.
            self.mutation_log.push(super::mutation_log::MutationEvent::PaintStamp {
                entity,
                mode,
                material_id,
            });
        }
        written
    }

    /// Resolve an entity into the data paint needs: an `AssetInfo`
    /// describing its octree + leaf_attr slot range, and the affine
    /// world transform to convert the brush world position into
    /// object-local space.
    ///
    /// Returns `None` for entities without a populated `Renderable.spatial`
    /// (e.g. unbaked procedurals or ones that already early-returned
    /// above).
    pub(crate) fn build_paint_context(
        &self,
        entity: hecs::Entity,
    ) -> Option<(arvx_render::AssetInfo, glam::Affine3A)> {
        use crate::components::Transform;
        let renderable = self.world.get::<&Renderable>(entity).ok()?;
        let spatial = renderable.spatial.as_ref().and_then(|g| g.as_octree())?;
        let transform = self.world.get::<&Transform>(entity).ok()?;

        let asset_info = arvx_render::AssetInfo {
            spatial: arvx_core::scene_node::SpatialHandle::Octree {
                root_offset: spatial.root_offset,
                len: spatial.len,
                depth: spatial.depth,
                base_voxel_size: spatial.base_voxel_size,
            },
            voxel_size: spatial.voxel_size,
            aabb: spatial.aabb,
            grid_origin: spatial.grid_origin,
            voxel_count: spatial.voxel_slot_count,
            // `Renderable.spatial` predates the leaf_attr rename — its
            // `voxel_slot_*` fields carry the leaf_attr slot range
            // (same value, historical name).
            leaf_attr_slot_start: spatial.voxel_slot_start,
            leaf_attr_slot_count: spatial.voxel_slot_count,
            has_skinning: false,
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

        Some((asset_info, entity_world))
    }

    /// World-space AABB containment test against every asset-backed
    /// entity. Returns the first match — tie-breaking is by iteration
    /// order, which is arbitrary. The real flow will use pick
    /// readback's `gpu_idx → entity` (`self.gpu_to_entity`) so this is
    /// only a fallback for command-driven / headless paint.
    fn find_entity_at_world_pos(&self, world_pos: Vec3) -> Option<hecs::Entity> {
        use crate::components::Transform;
        let mut best: Option<(hecs::Entity, f32)> = None;
        for (entity, (renderable, transform)) in
            self.world.query::<(&Renderable, &Transform)>().iter()
        {
            let Some(spatial) = renderable.spatial.as_ref().and_then(|g| g.as_octree()) else {
                continue;
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
            let local = entity_world.inverse().transform_point3(world_pos);
            if local.x >= spatial.aabb.min.x && local.x <= spatial.aabb.max.x
                && local.y >= spatial.aabb.min.y && local.y <= spatial.aabb.max.y
                && local.z >= spatial.aabb.min.z && local.z <= spatial.aabb.max.z
            {
                let center = (spatial.aabb.min + spatial.aabb.max) * 0.5;
                let dist = (local - center).length_squared();
                match best {
                    None => best = Some((entity, dist)),
                    Some((_, d)) if dist < d => best = Some((entity, dist)),
                    _ => {}
                }
            }
        }
        best.map(|(e, _)| e)
    }
}

/// Route the `EngineCommand::Paint` arm. Called from `process_cmd_edit`
/// when the dispatcher's edit chunk pattern-matches. Lives out-of-line
/// from `cmd_edit.rs` to keep this file the sole owner of paint wiring.
pub(crate) fn dispatch_paint(
    state: &mut EngineState,
    position: Vec3,
    normal: Vec3,
    radius: f32,
    color: [f32; 3],
    strength: f32,
    mode: PaintMode,
) {
    let _ = state.handle_paint_command(position, normal, radius, color, strength, mode);
}

#[cfg(test)]
mod tests {
    //! Headless paint tests. These use `apply_paint_stamp` directly
    //! with a known entity — the command path's `find_entity_at_world_pos`
    //! is tested separately in `test_find_entity_at_world_pos`.

    // TODO: these tests require a full EngineState harness. The
    // ArvxSceneManager apply_paint_sphere path is already covered by
    // arvx-render's paint::tests. A minimal engine harness for the
    // procedural / generator-owned gates is a follow-up; for now
    // those paths are covered by code review + the eventual UI
    // smoke test.
}
