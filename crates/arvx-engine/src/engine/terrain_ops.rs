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

        // Phase 1 — drain completed bakes and integrate each. The
        // streamer's `record_integrated` returns the *previous* live
        // token when this integration is a hot-swap (stamp / region /
        // Inspector edit re-queued an already-live tile). The
        // deferred eviction releases the old entity / asset *after*
        // the new one is in place so the renderer never sees a gap.
        // Diagnostic: `ARVX_TERRAIN_PROFILE=1` logs the per-tick
        // breakdown (how many tiles integrated this tick + the integrate
        // cost). This is the unpaced main-thread cost suspected of
        // stalling presentation long enough to trip the surface-Outdated
        // path (rinch #42) on a cold terrain generation — correlate the
        // `[terrain-tick]` line with `[render-frame] STALL` / `[geo-epoch]`.
        let prof = std::env::var("ARVX_TERRAIN_PROFILE").is_ok();
        let t_tick = std::time::Instant::now();
        // Drain freshly-completed bakes into the backlog queue, but do NOT
        // integrate the whole batch this tick. On a warm `.arvxtile` cache a
        // tile bakes in ~0 ms, so an entire footprint can land in one drain;
        // integrating all of them in a single sim tick (each under the
        // scene_mgr lock + a full GPU upload) stalls the sim for seconds
        // before it publishes a new snapshot — the load freeze. The sibling
        // asset-load path is already budgeted (`drain_pending_asset_loads`);
        // this gives terrain the same treatment so a burst materialises
        // progressively instead of freezing. (P3-A.)
        for kb in runtime.streamer.drain_completed() {
            runtime.pending_integrations.push_back(kb);
        }
        // Per-tick integrate budget (tiles). Override via
        // `ARVX_TERRAIN_INTEGRATE_BUDGET`; 0/invalid → default. Every queued
        // tile is still integrated (just spread across ticks), so the
        // streamer's `record_integrated` / hot-swap eviction bookkeeping sees
        // each one exactly as before.
        let integrate_budget: usize = std::env::var("ARVX_TERRAIN_INTEGRATE_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&b| b > 0)
            .unwrap_or(2);
        let n_this_tick = runtime.pending_integrations.len().min(integrate_budget);
        let integrated_any = n_this_tick > 0;
        let debug_hotswap = std::env::var("ARVX_TERRAIN_DEBUG").is_ok();
        let mut integ_sum = std::time::Duration::ZERO;
        let mut integ_max = std::time::Duration::ZERO;
        for _ in 0..n_this_tick {
            let Some((key, baked)) = runtime.pending_integrations.pop_front() else {
                break;
            };
            let t_i = std::time::Instant::now();
            let token = self.integrate_terrain_tile(&mut runtime, key, baked);
            let dt_i = t_i.elapsed();
            integ_sum += dt_i;
            integ_max = integ_max.max(dt_i);
            match token {
                Some(tok) => {
                    let prev = runtime.streamer.record_integrated(key, tok);
                    if debug_hotswap {
                        eprintln!(
                            "[hot-swap] tile ({},{},{},lvl{}) new_token={} prev_token={:?}",
                            key.x, key.y, key.z, key.level, tok, prev,
                        );
                    }
                    if let Some(prev_tok) = prev {
                        self.evict_hot_swap_predecessor(&mut runtime, key, prev_tok);
                    }
                }
                None => {
                    let prev = runtime.streamer.record_failed(key);
                    if debug_hotswap {
                        eprintln!(
                            "[hot-swap] INTEGRATE FAILED tile ({},{},{},lvl{}) prev_token={:?}",
                            key.x, key.y, key.z, key.level, prev,
                        );
                    }
                    if let Some(prev_tok) = prev {
                        // Integrate failed but a predecessor was being
                        // hot-swapped — evict the old pair so it
                        // doesn't orphan. Use the full
                        // `evict_terrain_tiles` path (includes
                        // `on_terrain_tile_evicted`) since no fresh
                        // collider replaced it.
                        self.evict_terrain_tiles(&mut runtime, &[(key, prev_tok)]);
                    }
                }
            }
        }

        let t_p2 = std::time::Instant::now();
        // Phase 2 — residency + eviction. The dirty-tile set keeps
        // sculpted level-0 tiles pinned at fine LOD regardless of
        // distance, suppressing coarse tiles that would otherwise
        // draw over the sculpted region with procedural geometry.
        let evictions = runtime.streamer.update_residency_with_pinned(
            &terrain,
            camera_world,
            &runtime.dirty_tiles,
        );
        let evicted_any = !evictions.is_empty();
        self.evict_terrain_tiles(&mut runtime, &evictions);
        let p2 = t_p2.elapsed();

        // Phase 3 — nearest-first submission.
        let t_p3 = std::time::Instant::now();
        runtime.streamer.submit_pending(&terrain);
        let p3 = t_p3.elapsed();

        if integrated_any || evicted_any {
            // Defeat the transform-only fast path so the next
            // render-frame build picks up the new / removed tile
            // entities.
            self.gpu_objects_dirty.mark_all();
        }

        let backlog = runtime.pending_integrations.len();
        if prof && (integrated_any || evicted_any || backlog > 0) {
            eprintln!(
                "[terrain-tick] integrated={n_this_tick} backlog={backlog} \
                 (integrate {:.1}ms, max_tile {:.1}ms) \
                 evicted={} residency+evict={:.1}ms submit={:.1}ms tick_total={:.1}ms",
                integ_sum.as_secs_f64() * 1000.0,
                integ_max.as_secs_f64() * 1000.0,
                evictions.len(),
                p2.as_secs_f64() * 1000.0,
                p3.as_secs_f64() * 1000.0,
                t_tick.elapsed().as_secs_f64() * 1000.0,
            );
        }

        self.terrain = Some(runtime);
    }

    /// Construct a fresh `TerrainRuntime` for `terrain_entity` and
    /// install it on `self.terrain`. Seeds the streamer's
    /// `scene_dir` from `self.scene_path` (so subsequent bakes
    /// prefer on-disk `.arvxtile` files over re-running `TerrainFn`)
    /// and hydrates `runtime.diffs` from any `.arvxsculpt` sidecars
    /// already on disk for this scene.
    ///
    /// Two callers:
    /// 1. The `SpawnTerrain` command, after creating the entity.
    /// 2. Scene load completion, when the Terrain singleton was
    ///    restored via the component registry and needs a runtime
    ///    so the streamer actually ticks. Without this, a saved
    ///    scene's terrain would silently fail to materialise on
    ///    reload (the next tick `tick_terrain_streamer` early-
    ///    returns on `self.terrain.is_none()`).
    ///
    /// Idempotent in spirit but NOT in effect — replacing an
    /// existing runtime would lose in-flight bakes and live tile
    /// state. Callers are expected to have enforced the singleton
    /// already (`SpawnTerrain`'s explicit check; scene-load's
    /// "only on first match" loop).
    pub(crate) fn init_terrain_runtime(&mut self, terrain_entity: hecs::Entity) {
        // Bring the Terrain entity's runtime `terrain_fn` in line with
        // the live `MaterialLibrary`. Scene-load and SpawnTerrain both
        // construct the Terrain through the component registry, which
        // can't see the material library — its `terrain_fn` resolves
        // every `MaterialRef::Path` to slot 0 until refreshed here.
        //
        // Catch the "scene loaded before material scan" footgun: the
        // refresh below silently collapses every `MaterialRef::Path`
        // to slot 0 against an empty library, so the streamer would
        // bake every procedural tile against the default material.
        // The orchestration in `OpenProject` is supposed to scan
        // materials first; this warn fires if a future refactor
        // reorders things.
        if self.material_lib.slot_count() == 0 {
            self.console.warn(
                "init_terrain_runtime called before MaterialLibrary scan — \
                 terrain materials will collapse to slot 0. Check OpenProject \
                 ordering (material_lib.scan must run before load_scene_from_file)."
                    .to_string(),
            );
        }
        self.rebuild_terrain_fn_from_material_lib();
        let mut runtime = Box::new(
            crate::terrain_state::TerrainRuntime::new(terrain_entity),
        );
        if let Some(scene_dir) = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
        {
            runtime.streamer.set_scene_dir(Some(scene_dir.clone()));
            // V2 LOD pyramid: hydrate the in-RAM diff map from any
            // `.arvxsculpt` sidecars saved alongside this scene.
            // The post-integrate replay path
            // (`gather_replay_edits`) then re-applies each diff
            // onto the first matching bake — restoring sculpts
            // authored in a previous session even when the
            // corresponding `.arvxtile` is missing
            // (eviction-before-save case). No-op when the
            // `sculpt/` subdirectory doesn't exist.
            runtime.diffs = arvx_terrain::load_all_sculpt_diffs(&scene_dir);
            if !runtime.diffs.is_empty() {
                self.console.info(format!(
                    "Terrain: loaded {} sculpt diff(s) from disk",
                    runtime.diffs.len(),
                ));
            }
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
            surface_index_count,
            ..
        } = baked;

        // Phase 8: snapshot the LOD-0 triangle data so the play-mode
        // physics path can build a Rapier TriMesh later. Reading
        // before `mesh` moves into `sm.integrate_baked_tile` because
        // the scene manager consumes the blob. Use the SURFACE-only index
        // count (skirts are folded into lod0 but back-culled / never drawn,
        // so a full-lod0 collider would snag bodies on invisible seam walls).
        let collider_mesh =
            arvx_terrain::TileColliderMesh::from_mesh_blob_prefix(&mesh, surface_index_count);
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
            // [lock] probe (ARVX_LOCK_PROFILE): WAIT (contention) vs HOLD (work)
            // for the terrain-tile integrate — settles whether the 400-1048ms
            // integrate spikes are blocking on a concurrent splice/upload or
            // are genuine integrate work held under the lock.
            let t_acq = std::time::Instant::now();
            let mut sm = match self.scene_mgr.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let wait_ms = t_acq.elapsed().as_secs_f32() * 1000.0;
            let t_hold = std::time::Instant::now();
            let out = sm.integrate_baked_tile(
                artifact,
                mesh,
                aabb,
                voxel_size_m,
                synthetic_path,
            );
            if std::env::var("ARVX_LOCK_PROFILE").is_ok() {
                eprintln!(
                    "[lock] terrain-integrate scene_mgr wait={:.1}ms hold={:.1}ms (tile {},{},{} lvl{})",
                    wait_ms,
                    t_hold.elapsed().as_secs_f32() * 1000.0,
                    key.x, key.y, key.z, key.level,
                );
            }
            out?
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
        // Hot-swap: `tile_keys` may already hold a (prev_entity,
        // prev_handle) pair from the pre-rebake integration. The
        // insert overwrites it with the fresh pair; the deferred
        // eviction in `evict_hot_swap_predecessor` will release the
        // old pair (sourced from `live_tiles` via `record_integrated`'s
        // returned previous token). The replaced value here is the
        // same pair but resolved key-side — discarded as we already
        // own the eviction via the token path.
        let prev_pair = runtime.tile_keys.insert(key, (entity, asset_handle));
        if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
            eprintln!(
                "[integrate] tile ({},{},{},lvl{}) new=(entity={:?}, handle={}) prev={:?} \
                 root_offset={} voxels={}",
                key.x, key.y, key.z, key.level,
                entity, asset_handle.raw(),
                prev_pair.map(|(e, h)| (e, h.raw())),
                root_offset, info.voxel_count,
            );
        }
        // Silence "unused import" if EditorMetadata isn't used here.
        let _ = std::any::type_name::<EditorMetadata>();

        // V2 LOD pyramid follow-up: replay any persistent sculpt diff
        // applicable to this tile. Three cases produce a non-empty
        // replay:
        //
        //   * Level-0 tile after eviction → re-load: `runtime.diffs`
        //     still holds the per-tile diff (eviction touches only
        //     the streamer slot, never the diff map). Replay restores
        //     the sculpt onto the fresh procedural bake.
        //   * Level-N≥1 coarse-LOD ancestor: enumerate the 8^N
        //     level-0 descendants present in `runtime.diffs`,
        //     downsample each into the coarse grid, and replay the
        //     composed diff so the sculpt appears at coarse LOD too.
        //   * Scene reload after `.arvxsculpt` load (P6 lands this).
        //
        // `tile_keys` already holds the freshly-integrated handle
        // (the `insert` above). We build the edit batch from the
        // runtime borrow we hold, then briefly grab the scene_mgr
        // lock to apply.
        let replay_edits = self.gather_replay_edits(runtime, key);
        if !replay_edits.is_empty() {
            let mut sm = match self.scene_mgr.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            sm.apply_diff_to_handle(asset_handle, &replay_edits);
            if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
                eprintln!(
                    "[terrain] replay sculpt diff: key=({},{},{},lvl{}) edits={}",
                    key.x, key.y, key.z, key.level, replay_edits.len(),
                );
            }
        }

        Some(token)
    }

    /// Build the LeafEdit batch that should be replayed onto `key`'s
    /// freshly-baked asset. Level-0 returns the per-tile diff edits
    /// verbatim; level-N≥1 enumerates every level-0 descendant key
    /// present in `runtime.diffs` and downsamples each into the
    /// coarse grid via [`arvx_terrain::SculptDiff::downsampled_to`].
    ///
    /// `&self` only — reads `runtime.diffs` and the Terrain entity's
    /// `voxel_size_for_level` mapping. Empty vec when no diffs apply.
    fn gather_replay_edits(
        &self,
        runtime: &TerrainRuntime,
        key: arvx_terrain::TileKey,
    ) -> Vec<arvx_core::sculpt::LeafEdit> {
        if runtime.diffs.is_empty() {
            return Vec::new();
        }
        if key.level == 0 {
            return runtime
                .diffs
                .get(&key)
                .map(|d| d.edits.clone())
                .unwrap_or_default();
        }
        let (fine_vs, coarse_vs) = match self
            .world
            .get::<&arvx_terrain::Terrain>(runtime.terrain_entity)
        {
            Ok(t) => (
                t.voxel_size_for_level(0),
                t.voxel_size_for_level(key.level),
            ),
            Err(_) => return Vec::new(),
        };

        // Coarse tile (level=N, x, y, z) covers level-0 tiles in
        // `(x*span..(x+1)*span)` along each axis. The lod_levels
        // clamp caps `key.level` at 7, so worst-case span = 128 (≈2M
        // descendants); typical N=1..=3 gives 8..512 candidates per
        // coarse bake, each a `runtime.diffs.get` hash lookup.
        let span = 1i32 << key.level;
        let base_x = key.x * span;
        let base_y = key.y * span;
        let base_z = key.z * span;

        let mut combined: Vec<arvx_core::sculpt::LeafEdit> = Vec::new();
        for dz in 0..span {
            for dy in 0..span {
                for dx in 0..span {
                    let fine = arvx_terrain::TileKey::level0(
                        base_x + dx,
                        base_y + dy,
                        base_z + dz,
                    );
                    let Some(diff) = runtime.diffs.get(&fine) else {
                        continue;
                    };
                    if diff.is_empty() {
                        continue;
                    }
                    let ds = diff.downsampled_to(fine, fine_vs, key, coarse_vs);
                    combined.extend(ds.edits);
                }
            }
        }
        combined
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
        // Persist EVERY resident tile (not just sculpted/dirty ones) so a
        // reload restores them from disk instead of re-baking the whole
        // procedural terrain from `TerrainFn` (the load-time freeze).
        // Validity is guarded by the bake signature written at the end:
        // a later terrain edit changes the signature, so the streamer
        // ignores these cached tiles (re-bakes) until the next save.
        let resident: Vec<arvx_terrain::TileKey> =
            runtime.tile_keys.keys().copied().collect();
        let mut succeeded: Vec<arvx_terrain::TileKey> = Vec::new();
        let mut failed: usize = 0;
        for key in &resident {
            // Resolve the asset handle. The tile may have been
            // evicted between the last edit and the save — log and
            // skip (V1 limitation; the .arvxtile would need to come
            // from a not-yet-implemented "edit log" or in-memory
            // shadow to survive eviction).
            let Some(&(_entity, handle)) = runtime.tile_keys.get(key) else {
                // `resident` came straight from `tile_keys`, so this is
                // unreachable in practice — defensive skip.
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
        runtime.dirty_tiles.clear();
        // Stamp the cache with the current terrain signature so the
        // loader can tell whether these tiles still match the live
        // terrain (else it re-bakes rather than loading stale geometry).
        if let Ok(t) = self.world.get::<&arvx_terrain::Terrain>(runtime.terrain_entity) {
            let sig = t.bake_signature();
            drop(t);
            if let Err(e) = arvx_terrain::write_signature(scene_dir, sig) {
                self.console
                    .warn(format!("Terrain: write bake signature failed: {e}"));
            }
        }
        self.console.info(format!(
            "Terrain: persisted {} tile(s) ({} failed)",
            succeeded.len(),
            failed,
        ));

        // V2 LOD pyramid: persist sculpt diffs as `.arvxsculpt`
        // sidecars. Independent of the `.arvxtile` loop above —
        // diffs live in `runtime.diffs` and survive tile eviction,
        // so this loop persists every authored sculpt even for
        // tiles that already evicted (the case the warning loop
        // above logs as a known V1 limitation). On scene reload
        // these diffs feed `runtime.diffs` again via
        // `load_all_sculpt_diffs`, and the post-integrate replay
        // hook puts the sculpt back onto fresh procedural bakes.
        let mut sculpt_saved = 0usize;
        let mut sculpt_failed = 0usize;
        let mut sculpt_skipped_empty = 0usize;
        for (key, diff) in &runtime.diffs {
            if diff.is_empty() {
                sculpt_skipped_empty += 1;
                continue;
            }
            match arvx_terrain::save_sculpt_diff(scene_dir, *key, diff) {
                Ok(_) => sculpt_saved += 1,
                Err(e) => {
                    self.console.warn(format!(
                        "Save sculpt diff ({}, {}, {}, lvl {}) failed: {e}",
                        key.x, key.y, key.z, key.level,
                    ));
                    sculpt_failed += 1;
                }
            }
        }
        if sculpt_saved + sculpt_failed > 0 {
            self.console.info(format!(
                "Terrain: persisted {sculpt_saved} sculpt diff(s) \
                 ({sculpt_failed} failed, {sculpt_skipped_empty} empty)",
            ));
        }
    }

    /// Phase 9: invalidate every live terrain tile + the streamer's
    /// view. Used after Terrain Inspector edits change the world's
    /// procedural source — every loaded tile is stale and needs to
    /// be re-baked under the new parameters.
    ///
    /// Hot-swap: the existing tile geometry stays resident in the
    /// scene until each new bake completes; the deferred eviction
    /// inside `tick_terrain_streamer`'s integrate path releases the
    /// old (entity, asset) pair once the fresh tile is in place. No
    /// "tiles flicker out" window.
    ///
    /// Sculpt preservation: dirty tiles are excluded — Inspector edits
    /// to TerrainFn don't drop authored sculpts. User must Revert a
    /// tile to re-apply the procedural baseline. See
    /// `stamp_ops::sync_terrain_stamps_and_invalidate` for the full
    /// rationale.
    pub(crate) fn invalidate_all_terrain_tiles(&mut self) {
        let Some(runtime) = self.terrain.as_mut() else {
            return;
        };
        runtime
            .streamer
            .invalidate_all_excluding(&runtime.dirty_tiles);
    }

    /// Rebuild every Terrain's runtime `terrain_fn` from its stored
    /// spec using the current [`MaterialLibrary`]. Does NOT invalidate
    /// tiles — call after construction (`init_terrain_runtime`,
    /// scene-load) where there's nothing live to invalidate yet, or
    /// from inside `refresh_terrain_fn_from_material_lib` which
    /// handles invalidation separately.
    ///
    /// Editor-singleton invariant: at most one entity carries
    /// `Terrain`; we iterate defensively in case that changes.
    pub(crate) fn rebuild_terrain_fn_from_material_lib(&mut self) {
        // Collect entity ids first so we can borrow `material_lib`
        // immutably while taking a `&mut Terrain` from the world.
        let entities: Vec<hecs::Entity> = self
            .world
            .query::<&arvx_terrain::Terrain>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for entity in entities {
            if let Ok(mut t) = self.world.get::<&mut arvx_terrain::Terrain>(entity) {
                t.refresh_terrain_fn(&self.material_lib);
            }
        }
    }

    /// Same as [`Self::rebuild_terrain_fn_from_material_lib`] but
    /// also invalidates every live tile so they re-bake under the
    /// new material resolution.
    ///
    /// Called when the material library's slot mapping changes — a new
    /// material is created, an existing material is deleted, or a
    /// fresh scan loads a different set of files. Without this, a
    /// `MaterialRef::Path` that resolved to slot 0 (default opaque)
    /// the first time around would silently stay at slot 0 even after
    /// the user adds the missing `.arvxmat` file.
    pub(crate) fn refresh_terrain_fn_from_material_lib(&mut self) {
        self.rebuild_terrain_fn_from_material_lib();
        // Every live tile may now resolve to different slot ids,
        // so re-bake the lot. Dirty (sculpt-edited) tiles are
        // preserved by `invalidate_all_terrain_tiles`.
        self.invalidate_all_terrain_tiles();
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

        // Step 2 — invalidate every live tile inside the AABB. Hot-swap:
        // existing geometry stays resident until each fresh bake lands;
        // the deferred eviction in `tick_terrain_streamer` releases the
        // previous (entity, asset) pair after the new tile is in place.
        // We still clear `dirty_tiles` + `divergent_tiles` immediately so
        // the heatmap drops the reverted region in the same frame.
        let invalidated: usize = {
            let runtime = self.terrain.as_mut().unwrap();
            let intersecting: Vec<arvx_terrain::TileKey> = runtime
                .tile_keys
                .keys()
                .copied()
                .filter(|k| {
                    let origin = k.origin_world().to_vec3();
                    let extent = k.extent_m();
                    let max = origin + glam::Vec3::splat(extent);
                    aabb.max.x >= origin.x
                        && aabb.min.x <= max.x
                        && aabb.max.y >= origin.y
                        && aabb.min.y <= max.y
                        && aabb.max.z >= origin.z
                        && aabb.min.z <= max.z
                })
                .collect();
            for key in &intersecting {
                runtime.dirty_tiles.remove(key);
                runtime.divergent_tiles.remove(key);
            }
            runtime.streamer.invalidate_aabb(aabb);
            intersecting.len()
        };
        self.console.info(format!(
            "Terrain: reverted {invalidated} live tile(s), deleted {deleted} .arvxtile file(s)",
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

    /// Hot-swap deferred eviction: drop the previous (entity, asset)
    /// pair for `key` now that the fresh integration is in place.
    /// Used only on the hot-swap path — the `on_terrain_tile_added`
    /// step during the fresh integrate already replaced the
    /// `TileKey`-keyed collider, so we MUST NOT call
    /// `on_terrain_tile_evicted` here (that would drop the
    /// just-installed new collider).
    ///
    /// `runtime.tile_keys[key]` was already overwritten with the new
    /// pair during integrate, so we drive the lookup off `prev_token`
    /// via `runtime.live_tiles` instead.
    fn evict_hot_swap_predecessor(
        &mut self,
        runtime: &mut TerrainRuntime,
        key: arvx_terrain::TileKey,
        prev_token: u64,
    ) {
        let Some((prev_entity, prev_handle)) = runtime.live_tiles.remove(&prev_token)
        else {
            return;
        };
        let _ = self.world.despawn(prev_entity);
        self.sculpt_overlays.remove(&prev_entity);
        let mut sm = match self.scene_mgr.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        sm.release_asset(prev_handle);
        drop(sm);
        if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
            eprintln!(
                "[hot-swap] tile ({},{},{},lvl{}) replaced prev entity={:?} \
                 handle={:?} (new entity already wired)",
                key.x, key.y, key.z, key.level, prev_entity, prev_handle,
            );
        }
        // Deliberate: no `on_terrain_tile_evicted(key)` here — the
        // fresh integrate's `on_terrain_tile_added(key)` already
        // replaced the collider for this `TileKey`.
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
