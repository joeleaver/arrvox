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
        falloff_curve: arvx_core::sculpt::FalloffCurve,
        strength: f32,
        stroke_seq: u64,
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
        let (asset_handle, asset_root_offset, entity_world) = {
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
            let Some(spatial) = renderable.spatial.as_ref().and_then(|g| g.as_octree()) else {
                return 0;
            };
            let root_offset = spatial.root_offset;
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
            (handle, root_offset, entity_world)
        };

        // ── Engine enum → core enum. Flatten is still deferred to
        // a later plan phase; everything else is wired through.
        let brush_mode = match mode {
            SculptMode::Raise => arvx_core::sculpt::BrushMode::Raise,
            SculptMode::Carve => arvx_core::sculpt::BrushMode::Carve,
            SculptMode::Inflate => arvx_core::sculpt::BrushMode::Inflate,
            SculptMode::Deflate => arvx_core::sculpt::BrushMode::Deflate,
            SculptMode::Smooth => arvx_core::sculpt::BrushMode::Smooth,
            SculptMode::Flatten => {
                self.console.warn(format!(
                    "Sculpt mode {mode:?} not implemented yet — \
                     Raise / Carve / Inflate / Deflate / Smooth are wired through P2.",
                ));
                return 0;
            }
        };

        // ── Resolve the stamp against the asset's octree (read-only). ─
        // Phase A: the scene manager does *not* mutate; it returns the
        // list of `leaf_attr_id`s to carve away. We merge that into
        // this entity's `SculptOverlay` below and ship it on the next
        // frame's `instance_sculpts` upload.
        let result = {
            let mut scene = self.scene_mgr.lock().expect("scene_mgr poisoned");
            scene.apply_sculpt_brush(
                asset_handle,
                world_pos,
                entity_world,
                radius,
                falloff_curve,
                strength,
                stroke_seq,
                brush_mode,
                material_id,
            )
        };

        let Some(result) = result else {
            return 0;
        };

        // Phase B R2/R4-minimal: Raise + Carve both apply real
        // mutation. `leaves_add_skipped` counts the kernel's Add
        // edits and is no longer informational — apply_delta on the
        // scene-manager side processes them. Kept on the result
        // struct for backward compat; ignore here.
        let _ = result.leaves_add_skipped;

        if result.removed_leaf_attr_ids.is_empty() && result.leaves_removed == 0 {
            // Stamp produced no overlay-eligible removes — it might
            // still have added geometry (Raise) or carved interior
            // bulk. Don't early-return; the geometry mutation already
            // happened in the scene manager and the visible result
            // comes from the mesh re-extract on the next frame.
        }

        // ── Merge into the per-entity sculpt overlay. ────────────────
        // `insert_batch` is O(N + K log K) so a drag stamp stays fast
        // even after the overlay has accumulated thousands of entries.
        let overlay = self.sculpt_overlays.entry(entity).or_default();
        overlay.insert_batch(result.removed_leaf_attr_ids);
        // Drop any slot IDs the stamp REUSED for new surface cells. The
        // LeafAttrPool's free list hands back recently-freed slot IDs
        // first, so a Raise after a Carve typically reuses the slots
        // the Carve just freed — and those slots are sitting in the
        // overlay's "carved" set. Leaving them there makes the mesh
        // FS `is_leaf_removed` check discard every fragment that
        // resolves to the reused slot, which manifests as a half-dome
        // after the first Carve. Removing them here keeps the overlay
        // honest: only slots whose surface cell is genuinely missing
        // remain.
        for slot in &result.allocated_leaf_attr_ids {
            overlay.remove(*slot);
        }

        // PERF_DEBT.md D3: this stamp added removed-leaf-attr ids to
        // the entity's sculpt overlay, so the concatenated
        // `gpu_instance_sculpts` content the render side reads will
        // differ from last frame after `update_scene_gpu` re-flattens.
        // Drives the same "skip on idle ticks" path as D2.
        self.gpu_instance_sculpts_dirty = true;

        // Force the next tick to rebuild gpu_instances + flatten the
        // overlay vec — the per-instance `sculpt_offset` / `sculpt_count`
        // get re-assigned each frame inside `update_scene_gpu`.
        // PERF_DEBT B1: only the sculpted entity's sculpt overlay
        // changed. C2 will use this to drive a per-row update.
        self.gpu_objects_dirty.mark_entity(entity);

        // Tell the painted-materials walk that THIS entity's geometry
        // changed. Without this, the walk's `geom_changed` branch
        // blanket-invalidates `painted_per_entity` and rewalks every
        // entity in the world — measured at ~586 ms on a 22-entity
        // splat5 scene (dominant component of the `[sculpt-pipeline]
        // bump→submit` gap). With the entity in `painted_dirty_entities`,
        // the walk re-scans only this one octree (~ms).
        self.painted_dirty_entities.insert(entity);

        // Phase C1: record the brush footprint (world space) so the
        // painted-materials walk can scope its octree scan to this
        // region instead of walking the full entity octree. Both Raise
        // and Carve get a region entry — Carve might evict shader-
        // bearing leaves whose tiles need rebuilding; Raise might add
        // new shader-bearing leaves under the brush. See
        // `docs/PERF_DEBT.md` C1.
        self.painted_dirty_regions
            .entry(entity)
            .or_default()
            .push(arvx_core::Aabb::from_center_half_extents(
                world_pos,
                Vec3::splat(radius),
            ));

        // PERF_DEBT.md C2-extension: sculpt-Raise with a glass brush
        // can flip the asset's has_glass verdict from false→true.
        // Drop the cache entry for this asset's root_offset so the
        // next has_glass check rescans. Carve cannot *add* glass
        // (only remove), so a stale-true verdict for the asset is
        // just an empty glass pass — perf cost only, no visual bug.
        if matches!(mode, SculptMode::Raise) {
            let is_glass_brush = (material_id as usize) < self.material_is_glass.len()
                && self.material_is_glass[material_id as usize];
            if is_glass_brush {
                self.asset_has_glass_cache.remove(&asset_root_offset);
            }
        }

        // Push a scope-carrying mutation event so Phase B/C consumers
        // can update their derived state incrementally. Phase A1 is
        // scaffolding only — the log drains unobserved every tick.
        self.mutation_log.push(super::mutation_log::MutationEvent::SculptStamp {
            entity,
            mode,
            material_id,
        });

        eprintln!(
            "[sculpt] stamp entity={:?} mode={:?} overlay_size={} (+{} this stamp)",
            entity, mode, overlay.len(), result.leaves_removed,
        );

        result.leaves_removed
    }
}

impl EngineState {
    /// Phase 4 brush dispatch: apply a single sculpt brush stamp to
    /// every live terrain tile whose AABB intersects the world-space
    /// brush AABB. Bypasses the selection / procedural / skeleton
    /// gates that `apply_sculpt_stamp` runs — terrain tiles don't
    /// participate in scene-tree selection (no `EditorMetadata`) and
    /// can't be procedural / generator-owned / skinned by construction.
    ///
    /// Returns the total number of leaves removed across all touched
    /// tiles. Returns 0 when no terrain runtime is active or no tiles
    /// intersect the brush.
    pub(crate) fn apply_sculpt_stamp_terrain(
        &mut self,
        world_pos: Vec3,
        radius: f32,
        falloff_curve: arvx_core::sculpt::FalloffCurve,
        strength: f32,
        stroke_seq: u64,
        mode: SculptMode,
        material_id: u16,
    ) -> usize {
        if radius <= 0.0 {
            return 0;
        }
        // Engine enum → core enum. Flatten is still deferred to a later
        // plan phase; everything else routes through.
        let brush_mode = match mode {
            SculptMode::Raise => arvx_core::sculpt::BrushMode::Raise,
            SculptMode::Carve => arvx_core::sculpt::BrushMode::Carve,
            SculptMode::Inflate => arvx_core::sculpt::BrushMode::Inflate,
            SculptMode::Deflate => arvx_core::sculpt::BrushMode::Deflate,
            SculptMode::Smooth => arvx_core::sculpt::BrushMode::Smooth,
            SculptMode::Flatten => {
                self.console.warn(format!(
                    "Sculpt mode {mode:?} not implemented yet — \
                     Raise / Carve / Inflate / Deflate / Smooth are wired through P2.",
                ));
                return 0;
            }
        };

        // Snapshot the live `(TileKey, Entity, AssetHandle)` triples we
        // intend to stamp before doing any &mut self work. Holding a
        // borrow into `self.terrain.tile_keys` across the per-tile
        // mutation loop would conflict with `scene_mgr` / overlays /
        // dirty marks below.
        let candidate_keys = arvx_terrain::tile_keys_intersecting_aabb(
            world_pos - Vec3::splat(radius),
            world_pos + Vec3::splat(radius),
        );
        let mut targets: Vec<(arvx_terrain::TileKey, hecs::Entity, arvx_render::AssetHandle)> =
            Vec::new();
        if let Some(runtime) = self.terrain.as_ref() {
            for key in &candidate_keys {
                if let Some(&(entity, handle)) = runtime.tile_keys.get(key) {
                    targets.push((*key, entity, handle));
                }
            }
        }
        if targets.is_empty() {
            return 0;
        }

        let mut total_removed: usize = 0;
        let identity = glam::Affine3A::IDENTITY;
        let brush_aabb = arvx_core::Aabb::from_center_half_extents(
            world_pos,
            Vec3::splat(radius),
        );

        // Track which tiles produced edits this stamp so the dirty
        // set captures only tiles whose disk state will diverge from
        // a fresh TerrainFn bake. Tiles in the brush AABB but with
        // no actual cell changes don't enter the set.
        let mut touched_keys: Vec<arvx_terrain::TileKey> = Vec::new();

        for (key, entity, asset_handle) in targets {
            // Per-tile brush stamp. Tile entities sit at world-frame
            // identity; their `SpatialData.grid_origin` carries the
            // tile-origin offset, so `apply_sculpt_brush` resolves
            // grid coords correctly with an identity entity_world.
            let result = {
                let mut scene = self.scene_mgr.lock().expect("scene_mgr poisoned");
                scene.apply_sculpt_brush(
                    asset_handle,
                    world_pos,
                    identity,
                    radius,
                    falloff_curve,
                    strength,
                    stroke_seq,
                    brush_mode,
                    material_id,
                )
            };
            let Some(result) = result else {
                continue;
            };
            total_removed += result.leaves_removed;
            touched_keys.push(key);

            // Mirror the asset path's per-entity bookkeeping for the
            // tile's entity. See `apply_sculpt_stamp` for rationale on
            // each line.
            let overlay = self.sculpt_overlays.entry(entity).or_default();
            overlay.insert_batch(result.removed_leaf_attr_ids);
            for slot in &result.allocated_leaf_attr_ids {
                overlay.remove(*slot);
            }
            self.gpu_instance_sculpts_dirty = true;
            self.gpu_objects_dirty.mark_entity(entity);
            self.painted_dirty_entities.insert(entity);
            self.painted_dirty_regions
                .entry(entity)
                .or_default()
                .push(brush_aabb);

            if matches!(mode, SculptMode::Raise) {
                let is_glass_brush = (material_id as usize) < self.material_is_glass.len()
                    && self.material_is_glass[material_id as usize];
                if is_glass_brush {
                    // Tiles share `release_asset` cache keys via their
                    // octree root_offset; flush so the next has_glass
                    // verdict rescans.
                    if let Ok(r) = self.world.get::<&crate::components::Renderable>(entity) {
                        if let Some(spatial) = r.spatial.as_ref().and_then(|g| g.as_octree()) {
                            self.asset_has_glass_cache.remove(&spatial.root_offset);
                        }
                    }
                }
            }

            self.mutation_log.push(super::mutation_log::MutationEvent::SculptStamp {
                entity,
                mode,
                material_id,
            });
        }

        // Mark every tile this stamp actually edited as dirty for
        // Phase 4.3 save. Idempotent — HashSet folds repeated stamps
        // on the same stroke into one entry.
        if let Some(runtime) = self.terrain.as_mut() {
            for k in &touched_keys {
                runtime.dirty_tiles.insert(*k);
            }
        }

        if total_removed > 0 {
            eprintln!(
                "[sculpt-terrain] stamp candidate_tiles={} touched={} mode={:?} total_removed={}",
                candidate_keys.len(),
                touched_keys.len(),
                mode,
                total_removed,
            );
        }

        total_removed
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
    strength: f32,
    mode: SculptMode,
) {
    let Some(entity) = state.selected_entity else { return };
    let material_id = state.selected_material.unwrap_or(0);
    let _ = state.apply_sculpt_stamp(
        entity,
        position,
        radius,
        arvx_core::sculpt::FalloffCurve::default(),
        strength,
        0,
        mode,
        material_id,
    );
}
