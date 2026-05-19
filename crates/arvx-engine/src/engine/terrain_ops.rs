//! Terrain streamer tick — the per-frame hook called from `tick_loop`.
//!
//! The streamer lives in `arvx-terrain` and is engine-agnostic (no
//! `arvx-render` dependency). This module is the engine-side
//! bridge: three phased calls into the streamer with the engine's
//! integrate / evict operations interleaved.
//!
//! 1. `drain_completed` — get ready bakes from the worker.
//!    For each, lock `scene_mgr`, call `integrate_baked_tile`, spawn
//!    a tile ECS entity. Report success/failure back via
//!    `record_integrated` / `record_failed`.
//! 2. `update_residency` — recompute the camera-radius desired set.
//!    The returned `(key, token)` pairs are the tiles to evict;
//!    despawn each entity and release its asset.
//! 3. `submit_pending` — nearest-first kickoff for new bakes.
//!
//! The phased API exists to keep `&mut self.world` and `&mut
//! self.scene_mgr` borrows scoped — a single closure-based tick
//! couldn't satisfy the borrow checker without `RefCell` (the
//! integrate closure mutates `world`; the evict closure mutates
//! `world` too).

use std::path::PathBuf;

use arvx_terrain::{Terrain, TerrainTile};

use crate::components::{
    EditorMetadata, RenderGeometry, Renderable, SpatialData, Transform,
};
use crate::terrain_state::TerrainRuntime;

impl super::state::EngineState {
    /// Drive one tick of the active Terrain streamer (no-op when no
    /// Terrain is spawned).
    pub(crate) fn tick_terrain_streamer(&mut self) {
        // Take the runtime out so closures don't conflict with
        // `self.world` / `self.scene_mgr` borrows. We reinstall it
        // before returning (unless the Terrain entity is gone, in
        // which case we drop the runtime).
        let Some(mut runtime) = self.terrain.take() else {
            return;
        };

        // Read the live Terrain config from the ECS. Clone and drop
        // the Ref so subsequent self.world / scene_mgr mutations are
        // borrow-safe.
        let terrain: Terrain = {
            let r = self.world.get::<&Terrain>(runtime.terrain_entity);
            match r {
                Ok(t) => (*t).clone(),
                Err(_) => {
                    // Force-clear and drop runtime.
                    drop(r);
                    let live = runtime.streamer.drain_all_live();
                    self.evict_terrain_tiles(&mut runtime, &live);
                    self.gpu_objects_dirty.mark_all();
                    self.console.info(
                        "Terrain entity removed — streamer shut down".to_string(),
                    );
                    // Drop runtime by not reinstalling.
                    let _ = runtime;
                    return;
                }
            }
        };

        let camera_world: arvx_core::WorldPosition = self.camera.position.into();

        // Phase 1 — drain completed bakes and integrate each.
        let completed = runtime.streamer.drain_completed();
        let integrated_any = !completed.is_empty();
        for (key, baked) in completed {
            let token = self.integrate_terrain_tile(&mut runtime, key, baked);
            match token {
                Some(tok) => runtime.streamer.record_integrated(key, tok),
                None => runtime.streamer.record_failed(key),
            }
        }

        // Phase 2 — residency + eviction.
        let evictions = runtime.streamer.update_residency(&terrain, camera_world);
        let evicted_any = !evictions.is_empty();
        self.evict_terrain_tiles(&mut runtime, &evictions);

        // Phase 3 — nearest-first submission.
        runtime.streamer.submit_pending(&terrain);

        if integrated_any || evicted_any {
            // Defeat the transform-only fast path so the next
            // render-frame build picks up the new / removed tile
            // entities.
            self.gpu_objects_dirty.mark_all();
        }

        self.terrain = Some(runtime);
    }

    /// Integrate one finished bake. Returns the per-tile engine
    /// token on success, `None` on failure.
    fn integrate_terrain_tile(
        &mut self,
        runtime: &mut TerrainRuntime,
        key: arvx_terrain::TileKey,
        baked: arvx_terrain::BakedTile,
    ) -> Option<u64> {
        let arvx_terrain::BakedTile {
            artifact,
            mesh,
            voxel_size_m,
            ..
        } = baked;
        let synthetic_path = PathBuf::from(format!(
            "terrain://{}_{}_{}_{}",
            key.level, key.x, key.y, key.z
        ));
        let tile_origin = key.origin_world().to_vec3();
        let aabb = arvx_core::Aabb {
            min: tile_origin,
            max: tile_origin + glam::Vec3::splat(key.extent_m()),
        };

        let (asset_handle, info) = {
            let mut sm = match self.scene_mgr.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            sm.integrate_baked_tile(
                artifact,
                mesh,
                aabb,
                voxel_size_m,
                synthetic_path,
            )?
        };

        let (root_offset, len, depth, base_voxel_size) = match info.spatial {
            arvx_core::scene_node::SpatialHandle::Octree {
                root_offset,
                len,
                depth,
                base_voxel_size,
            } => (root_offset, len, depth, base_voxel_size),
            arvx_core::scene_node::SpatialHandle::BrickMap(_) => {
                eprintln!(
                    "[terrain] integrate_baked_tile returned BrickMap handle — skipping tile"
                );
                let mut sm = self.scene_mgr.lock().unwrap();
                sm.release_asset(asset_handle);
                return None;
            }
        };

        let spatial = SpatialData {
            root_offset,
            len,
            depth,
            base_voxel_size,
            aabb: info.aabb,
            voxel_size: info.voxel_size,
            grid_origin: info.grid_origin,
            voxel_slot_start: info.leaf_attr_slot_start,
            voxel_slot_count: info.leaf_attr_slot_count,
            brick_ids: Vec::new(),
        };

        // Tile entity:
        // - Transform: identity in world frame; the SpatialData
        //   already carries the grid_origin (= tile origin) so the
        //   renderer places the geometry correctly.
        // - No EditorMetadata: the save-loop iterator queries
        //   `(Transform, EditorMetadata)`; omitting EditorMetadata
        //   keeps tiles invisible to the save path even without the
        //   TerrainTile filter (defense in depth — both gates are in
        //   place).
        let entity = self.world.spawn((
            Transform::default(),
            Renderable {
                asset_path: None,
                primitive: None,
                material_id: 0,
                voxel_count: info.voxel_count,
                spatial: Some(RenderGeometry::Octree(spatial)),
                asset_handle: Some(asset_handle),
                material_overrides: Vec::new(),
            },
            TerrainTile { key },
        ));

        let token = runtime.next_token;
        runtime.next_token = runtime.next_token.wrapping_add(1);
        runtime.live_tiles.insert(token, (entity, asset_handle));
        runtime.tile_keys.insert(key, (entity, asset_handle));
        // Silence "unused import" if EditorMetadata isn't used here.
        let _ = std::any::type_name::<EditorMetadata>();
        Some(token)
    }

    /// Phase 4.3: write every dirty terrain tile as `.arvxtile`
    /// alongside the scene file. Clears the dirty set on success.
    ///
    /// Cost: O(dirty_tile_count × tile_cell_count) for the artifact
    /// extraction + write. Each tile bake-time-ish (~1-2 s per Tier-2
    /// tile on cold cache); usually the dirty set is small.
    ///
    /// Per-tile failures are logged but don't abort the loop — partial
    /// progress is better than no progress.
    pub(crate) fn flush_dirty_terrain_tiles(&mut self, scene_dir: &std::path::Path) {
        let Some(runtime) = self.terrain.as_mut() else { return };
        if runtime.dirty_tiles.is_empty() {
            return;
        }
        let dirty: Vec<arvx_terrain::TileKey> =
            runtime.dirty_tiles.iter().copied().collect();
        let mut succeeded: Vec<arvx_terrain::TileKey> = Vec::new();
        let mut failed: usize = 0;
        for key in &dirty {
            // Resolve the asset handle. The tile may have been
            // evicted between the last edit and the save — log and
            // skip (V1 limitation; the .arvxtile would need to come
            // from a not-yet-implemented "edit log" or in-memory
            // shadow to survive eviction).
            let Some(&(_entity, handle)) = runtime.tile_keys.get(key) else {
                self.console.warn(format!(
                    "Skipping save for evicted dirty tile (lvl {}, {}, {}, {}). \
                     Phase 4.3 V1 limitation: edits to evicted tiles aren't \
                     persisted.",
                    key.level, key.x, key.y, key.z,
                ));
                failed += 1;
                continue;
            };

            // Compute voxel size for this tile's level. Pull from the
            // Terrain config so the saved file's voxel size matches
            // what the next bake would produce.
            let voxel_size = self
                .world
                .get::<&arvx_terrain::Terrain>(runtime.terrain_entity)
                .ok()
                .map(|t| t.voxel_size_for_level(key.level))
                .unwrap_or(
                    arvx_core::constants::RESOLUTION_TIERS
                        [arvx_core::constants::DEFAULT_TERRAIN_TIER]
                        .voxel_size,
                );

            let (artifact, _mesh) = {
                let scene = match self.scene_mgr.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let artifact = match scene.extract_artifact_from_handle(handle) {
                    Some(a) => a,
                    None => {
                        failed += 1;
                        continue;
                    }
                };
                // Mesh blob deliberately read but unused for v0 of
                // Phase 4.3 — we let the load path re-extract the
                // mesh from the saved octree to avoid the file-local
                // vertex-ID relocation complexity in the
                // not-yet-implemented Phase 4.4 reader. Once 4.4
                // lands the mesh blob, swap to passing it through
                // `write_rkp` directly. Computed-and-dropped here so
                // any borrow / cost issues show up early.
                let mesh = scene.extract_mesh_blob_from_handle(handle);
                (artifact, mesh)
            };
            match arvx_terrain::save_tile(scene_dir, *key, &artifact, voxel_size) {
                Ok(path) => {
                    if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
                        eprintln!(
                            "[terrain-save] tile ({}, {}, {}, lvl {}) -> {}",
                            key.x, key.y, key.z, key.level, path.display(),
                        );
                    }
                    succeeded.push(*key);
                }
                Err(e) => {
                    self.console.warn(format!(
                        "Save tile ({}, {}, {}, lvl {}) failed: {e}",
                        key.x, key.y, key.z, key.level,
                    ));
                    failed += 1;
                }
            }
        }
        for k in &succeeded {
            runtime.dirty_tiles.remove(k);
        }
        self.console.info(format!(
            "Terrain: persisted {} tile(s) ({} failed)",
            succeeded.len(),
            failed,
        ));
    }

    /// Despawn + release_asset for every `(key, token)` pair.
    fn evict_terrain_tiles(
        &mut self,
        runtime: &mut TerrainRuntime,
        evictions: &[(arvx_terrain::TileKey, u64)],
    ) {
        for (key, token) in evictions {
            // Drop the reverse-map entry up front so brush dispatch
            // can't pick up a stale handle mid-eviction.
            runtime.tile_keys.remove(key);
            if let Some((entity, asset_handle)) = runtime.live_tiles.remove(token) {
                let _ = self.world.despawn(entity);
                // Drop any per-entity sculpt overlay for this tile —
                // it points into the leaf_attr pool slots we're about
                // to release. Leaving it behind would let a stale slot
                // re-appear in fragment discard if a future tile
                // reuses the same entity id.
                self.sculpt_overlays.remove(&entity);
                let mut sm = match self.scene_mgr.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                sm.release_asset(asset_handle);
            }
        }
    }
}
