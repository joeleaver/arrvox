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
        // Silence "unused import" if EditorMetadata isn't used here.
        let _ = std::any::type_name::<EditorMetadata>();
        Some(token)
    }

    /// Despawn + release_asset for every `(key, token)` pair.
    fn evict_terrain_tiles(
        &mut self,
        runtime: &mut TerrainRuntime,
        evictions: &[(arvx_terrain::TileKey, u64)],
    ) {
        for (_key, token) in evictions {
            if let Some((entity, asset_handle)) = runtime.live_tiles.remove(token) {
                let _ = self.world.despawn(entity);
                let mut sm = match self.scene_mgr.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                sm.release_asset(asset_handle);
            }
        }
    }
}
