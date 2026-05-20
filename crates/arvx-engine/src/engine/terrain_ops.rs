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

        // Phase 8: snapshot the LOD-0 triangle data so the play-mode
        // physics path can build a Rapier TriMesh later. Reading
        // before `mesh` moves into `sm.integrate_baked_tile` because
        // the scene manager consumes the blob.
        let collider_mesh = arvx_terrain::TileColliderMesh::from_mesh_blob(&mesh);
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
            collider_mesh,
        ));

        // Phase 8: if play mode is live, tell it about the new tile so
        // it builds (or rebuilds, on re-bake) the Rapier TriMesh
        // collider AND wakes any sleeping bodies inside the rebuilt
        // tile's AABB. Edit-mode tiles still carry the
        // `TileColliderMesh` component; entering play later picks
        // them up via `build_initial_from_world`.
        if let Some(ref mut play) = self.play_state {
            play.on_terrain_tile_added(&self.world, entity, key, aabb);
        }

        let token = runtime.next_token;
        runtime.next_token = runtime.next_token.wrapping_add(1);
        runtime.live_tiles.insert(token, (entity, asset_handle));
        // Phase 9b: divergence detection. If a `.arvxtile` exists on
        // disk for this key, the bake worker just loaded it instead
        // of running the procedural baseline — that means it has
        // sculpt edits saved from a prior session. Carry-forward into
        // the heatmap. (The session's own sculpts mark divergence via
        // `mark_dirty` in sculpt_ops / paint_ops.)
        if let Some(scene_dir) = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
        {
            if arvx_terrain::tile_path(scene_dir, key).exists() {
                runtime.divergent_tiles.insert(key);
            }
        }
        // DIAGNOSTIC — Phase 5 stamp bug hunt: if tile_keys already
        // has an entry for this key, the old entity got orphaned
        // (still in the ECS world, no longer tracked → never
        // despawned). Log so we can confirm.
        if let Some((prev_entity, prev_handle)) =
            runtime.tile_keys.insert(key, (entity, asset_handle))
        {
            if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
                eprintln!(
                    "[integrate] WARN tile ({},{},{},lvl{}) was already \
                     in tile_keys — prev entity={:?} prev_handle={:?} \
                     (orphaned; old mesh stays in scene)",
                    key.x, key.y, key.z, key.level, prev_entity, prev_handle,
                );
            }
        }
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

    /// Phase 9: invalidate every live terrain tile + the streamer's
    /// view. Used after Terrain Inspector edits change the world's
    /// procedural source — every loaded tile is stale and needs to
    /// be re-baked under the new parameters. The streamer
    /// republishes them on the next residency tick.
    pub(crate) fn invalidate_all_terrain_tiles(&mut self) {
        let Some(runtime) = self.terrain.as_mut() else {
            return;
        };
        let evictions = runtime.streamer.invalidate_all();
        if evictions.is_empty() {
            return;
        }
        // Process evictions through the shared collider-aware path.
        // The streamer already bumped its slot states; we just need
        // to drop the live entity / asset / collider for each one.
        let mut to_drop: Vec<(arvx_terrain::TileKey, hecs::Entity, arvx_render::AssetHandle)> =
            Vec::new();
        for (key, token) in &evictions {
            runtime.tile_keys.remove(key);
            if let Some((entity, asset_handle)) = runtime.live_tiles.remove(token) {
                to_drop.push((*key, entity, asset_handle));
            }
        }
        for (key, entity, asset_handle) in to_drop {
            let _ = self.world.despawn(entity);
            self.sculpt_overlays.remove(&entity);
            let mut sm = match self.scene_mgr.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            sm.release_asset(asset_handle);
            drop(sm);
            if let Some(ref mut play) = self.play_state {
                play.on_terrain_tile_evicted(key);
            }
        }
        self.gpu_objects_dirty.mark_all();
    }

    /// Phase 9b: revert sculpt edits inside `aabb`. Two steps:
    ///   1. Delete every `.arvxtile` file whose tile-cube intersects
    ///      the AABB. Without this the next bake would load the
    ///      persisted (edited) artifact instead of running the
    ///      procedural baseline.
    ///   2. `streamer.invalidate_aabb(aabb)` — evict every live tile
    ///      intersecting the AABB. The streamer republishes them on
    ///      the next residency tick, this time procedurally.
    ///
    /// No-op + console warn when no Terrain is in the scene.
    pub(crate) fn revert_terrain_in_aabb(&mut self, aabb: arvx_core::Aabb) {
        if self.terrain.is_none() {
            self.console
                .warn("Revert: no Terrain in this scene.".to_string());
            return;
        }
        // Step 1 — delete any `.arvxtile` files in range. Scope mirrors
        // the live-tile invalidation in step 2 so disk + in-memory
        // stay in lock-step.
        let mut deleted: usize = 0;
        if let Some(scene_dir) = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
        {
            let keys = arvx_terrain::tile_keys_intersecting_aabb(aabb.min, aabb.max);
            for key in keys {
                let path = arvx_terrain::tile_path(&scene_dir, key);
                if path.exists() {
                    match std::fs::remove_file(&path) {
                        Ok(()) => deleted += 1,
                        Err(e) => self.console.warn(format!(
                            "Revert: delete {} failed: {e}",
                            path.display(),
                        )),
                    }
                }
            }
        }

        // Step 2 — invalidate every live tile inside the AABB. Reuses
        // the collider-aware eviction path from `invalidate_all_terrain_tiles`.
        let evictions = {
            let runtime = self.terrain.as_mut().unwrap();
            runtime.streamer.invalidate_aabb(aabb)
        };
        if !evictions.is_empty() {
            let mut to_drop: Vec<(arvx_terrain::TileKey, hecs::Entity, arvx_render::AssetHandle)> =
                Vec::new();
            {
                let runtime = self.terrain.as_mut().unwrap();
                for (key, token) in &evictions {
                    runtime.tile_keys.remove(key);
                    if let Some((entity, asset_handle)) = runtime.live_tiles.remove(token) {
                        to_drop.push((*key, entity, asset_handle));
                    }
                    runtime.dirty_tiles.remove(key);
                    // Phase 9b: revert wipes the edit history for
                    // this tile, so it drops out of the heatmap.
                    // (The next bake will run the procedural baseline
                    // since we deleted the `.arvxtile` above.)
                    runtime.divergent_tiles.remove(key);
                }
            }
            for (key, entity, asset_handle) in to_drop {
                let _ = self.world.despawn(entity);
                self.sculpt_overlays.remove(&entity);
                let mut sm = match self.scene_mgr.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                sm.release_asset(asset_handle);
                drop(sm);
                if let Some(ref mut play) = self.play_state {
                    play.on_terrain_tile_evicted(key);
                }
            }
            self.gpu_objects_dirty.mark_all();
        }
        self.console.info(format!(
            "Terrain: reverted {} live tile(s), deleted {} .arvxtile file(s)",
            evictions.len(),
            deleted,
        ));
    }

    /// Phase 9b: persist every live terrain tile whose AABB intersects
    /// `aabb` as a `.arvxtile`. Analogous to `flush_dirty_terrain_tiles`
    /// but scoped to the AABB instead of the dirty set — Bake-snapshot
    /// is an explicit "freeze the current edits in this area to disk"
    /// op, dirty-or-not.
    pub(crate) fn bake_terrain_snapshot_in_aabb(&mut self, aabb: arvx_core::Aabb) {
        let Some(_runtime) = self.terrain.as_ref() else {
            self.console
                .warn("Bake snapshot: no Terrain in this scene.".to_string());
            return;
        };
        let Some(scene_dir) = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
        else {
            self.console.warn(
                "Bake snapshot: scene has no save path yet. Save the scene first.".to_string(),
            );
            return;
        };

        // Snapshot the keys + handles to bake. Borrow scoped so the
        // scene-mgr lock below doesn't conflict with `&mut self.terrain`.
        let targets: Vec<(arvx_terrain::TileKey, arvx_render::AssetHandle)> = {
            let runtime = self.terrain.as_ref().unwrap();
            let candidate_keys = arvx_terrain::tile_keys_intersecting_aabb(aabb.min, aabb.max);
            candidate_keys
                .into_iter()
                .filter_map(|key| runtime.tile_keys.get(&key).map(|&(_e, h)| (key, h)))
                .collect()
        };
        if targets.is_empty() {
            self.console.info(
                "Bake snapshot: no live tiles inside the target region.".to_string(),
            );
            return;
        }

        let mut succeeded: Vec<arvx_terrain::TileKey> = Vec::new();
        let mut failed: usize = 0;
        for (key, handle) in &targets {
            let voxel_size = self
                .world
                .get::<&arvx_terrain::Terrain>(self.terrain.as_ref().unwrap().terrain_entity)
                .ok()
                .map(|t| t.voxel_size_for_level(key.level))
                .unwrap_or(
                    arvx_core::constants::RESOLUTION_TIERS
                        [arvx_core::constants::DEFAULT_TERRAIN_TIER]
                        .voxel_size,
                );

            let artifact = {
                let scene = match self.scene_mgr.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                match scene.extract_artifact_from_handle(*handle) {
                    Some(a) => a,
                    None => {
                        failed += 1;
                        continue;
                    }
                }
            };
            match arvx_terrain::save_tile(&scene_dir, *key, &artifact, voxel_size) {
                Ok(_path) => succeeded.push(*key),
                Err(e) => {
                    self.console.warn(format!(
                        "Bake tile ({}, {}, {}, lvl {}) failed: {e}",
                        key.x, key.y, key.z, key.level,
                    ));
                    failed += 1;
                }
            }
        }
        // Successful saves clear the dirty bit (the on-disk state now
        // matches in-memory).
        {
            let runtime = self.terrain.as_mut().unwrap();
            for k in &succeeded {
                runtime.dirty_tiles.remove(k);
            }
        }
        self.console.info(format!(
            "Terrain: baked snapshot of {} tile(s) ({} failed)",
            succeeded.len(),
            failed,
        ));
    }

    /// Phase 9b: set the active terrain region from a screen-space
    /// drag-box rect. Projects each screen corner onto the y=0 plane
    /// using the viewport's camera + projection; the resulting world
    /// AABB takes its vertical span from the active Terrain's bounds
    /// (Unbounded → fixed ±256 m). Stored in `active_terrain_region`
    /// + flagged dirty so the next `StateUpdate` echoes it back.
    pub(crate) fn set_terrain_region_from_screen_rect(
        &mut self,
        viewport_id: crate::viewport::ViewportId,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) {
        // Resolve the viewport (handles missing id gracefully — no-op).
        let vp = match self.viewports.get(viewport_id) {
            Some(v) => v,
            None => return,
        };
        let cam = &self.camera;
        let width = vp.width.max(1) as f32;
        let height = vp.height.max(1) as f32;

        // Build view-projection inverse. Mirrors the projection used
        // by `mesh_lod_select` / the primary raster — perspective,
        // glam's `perspective_rh`. We re-derive here rather than
        // reading back the matrix because the editor-side camera is
        // the source of truth in edit mode (play mode would use the
        // live override).
        let aspect = width / height;
        let proj = glam::Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.near, cam.far);
        // Forward derived from yaw / pitch, same formula as
        // `camera::fly_direction`. We can't call that fn (it's
        // module-private) but the convention has to match — otherwise
        // the picks land left/right of the cursor.
        let forward = glam::Vec3::new(
            -cam.yaw.sin() * cam.pitch.cos(),
            cam.pitch.sin(),
            -cam.yaw.cos() * cam.pitch.cos(),
        )
        .normalize();
        let view = glam::Mat4::look_to_rh(cam.position, forward, glam::Vec3::Y);
        let vp_inv = (proj * view).inverse();

        let project_to_ground = |px: f32, py: f32| -> Option<glam::Vec3> {
            // Convert pixel → NDC. y flips (top-left → top of NDC is +1).
            let ndx = (px / width) * 2.0 - 1.0;
            let ndy = 1.0 - (py / height) * 2.0;
            // Near + far points on the ray.
            let near = vp_inv * glam::Vec4::new(ndx, ndy, 0.0, 1.0);
            let far = vp_inv * glam::Vec4::new(ndx, ndy, 1.0, 1.0);
            if near.w.abs() < 1.0e-6 || far.w.abs() < 1.0e-6 {
                return None;
            }
            let near = near.truncate() / near.w;
            let far = far.truncate() / far.w;
            let dir = far - near;
            // Ground plane y=0. Bail if the ray is near-parallel.
            if dir.y.abs() < 1.0e-4 {
                return None;
            }
            let t = -near.y / dir.y;
            // Behind the camera (or extremely distant) — clamp the
            // drag to the visible portion of the world. A negative t
            // means the ray would hit y=0 behind the camera; in that
            // case fall back to extending the ray forward 1 km on the
            // XZ plane at y=0.
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return None;
            }
            Some(near + dir * t)
        };

        let corners = [
            project_to_ground(x0, y0),
            project_to_ground(x1, y0),
            project_to_ground(x0, y1),
            project_to_ground(x1, y1),
        ];
        let valid: Vec<glam::Vec3> = corners.into_iter().flatten().collect();
        if valid.len() < 2 {
            self.console.warn(
                "Region drag-box: could not project enough corners onto the ground plane."
                    .to_string(),
            );
            return;
        }

        let mut min = valid[0];
        let mut max = valid[0];
        for v in &valid[1..] {
            min = min.min(*v);
            max = max.max(*v);
        }
        // Degenerate rect (zero area) — bail rather than store a
        // useless region.
        if (max.x - min.x) < 0.1 || (max.z - min.z) < 0.1 {
            self.console.warn(
                "Region drag-box: rect is too small — ignored.".to_string(),
            );
            return;
        }

        // Vertical span: bounded terrain uses its extent_y; unbounded
        // uses a fixed ±256 m span centred on y=0. Either way, the
        // AABB's vertical reach covers the full terrain stack so
        // Revert / Bake hit caves + overhangs.
        let (y_lo, y_hi) = self
            .terrain
            .as_ref()
            .and_then(|rt| {
                self.world
                    .get::<&arvx_terrain::Terrain>(rt.terrain_entity)
                    .ok()
                    .map(|t| match t.bounds {
                        arvx_terrain::TerrainBounds::Bounded { origin, extent } => {
                            let y_lo = origin.y as f32 * arvx_terrain::TILE_SIZE_M;
                            let y_hi =
                                (origin.y + extent.y as i32) as f32
                                    * arvx_terrain::TILE_SIZE_M;
                            (y_lo, y_hi)
                        }
                        arvx_terrain::TerrainBounds::Unbounded => (-256.0_f32, 256.0_f32),
                    })
            })
            .unwrap_or((-256.0_f32, 256.0_f32));
        let aabb = arvx_core::Aabb {
            min: glam::Vec3::new(min.x, y_lo, min.z),
            max: glam::Vec3::new(max.x, y_hi, max.z),
        };
        self.active_terrain_region = Some(aabb);
        self.active_terrain_region_dirty = true;
        self.console.info(format!(
            "Region set: ({:.1}, {:.1}, {:.1}) → ({:.1}, {:.1}, {:.1})",
            aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z,
        ));
    }

    /// Phase 9b: clear the active terrain region.
    pub(crate) fn clear_terrain_region(&mut self) {
        if self.active_terrain_region.is_some() {
            self.active_terrain_region = None;
            self.active_terrain_region_dirty = true;
        }
    }

    /// Phase 9b: compute a world-space AABB centred on the editor
    /// camera, used as the Revert / Bake target when no active
    /// region is set. Vertical span follows the same Bounded /
    /// Unbounded rule as the drag-box path.
    pub(crate) fn camera_radius_aabb(&self, radius: f32) -> arvx_core::Aabb {
        let cp = self.camera.position;
        let r = radius.max(0.5);
        let (y_lo, y_hi) = self
            .terrain
            .as_ref()
            .and_then(|rt| {
                self.world
                    .get::<&arvx_terrain::Terrain>(rt.terrain_entity)
                    .ok()
                    .map(|t| match t.bounds {
                        arvx_terrain::TerrainBounds::Bounded { origin, extent } => {
                            let y_lo = origin.y as f32 * arvx_terrain::TILE_SIZE_M;
                            let y_hi =
                                (origin.y + extent.y as i32) as f32
                                    * arvx_terrain::TILE_SIZE_M;
                            (y_lo, y_hi)
                        }
                        arvx_terrain::TerrainBounds::Unbounded => (-256.0_f32, 256.0_f32),
                    })
            })
            .unwrap_or((-256.0_f32, 256.0_f32));
        arvx_core::Aabb {
            min: glam::Vec3::new(cp.x - r, y_lo, cp.z - r),
            max: glam::Vec3::new(cp.x + r, y_hi, cp.z + r),
        }
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
            // Phase 8: drop the tile's Rapier collider if play mode is
            // active. The `on_terrain_tile_added` hook will rebuild it
            // when the next bake completes (stamp / region / sculpt
            // re-bake spawns a fresh tile entity for the same key).
            if let Some(ref mut play) = self.play_state {
                play.on_terrain_tile_evicted(*key);
            }
        }
    }
}
