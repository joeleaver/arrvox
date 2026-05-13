//! Frame lifecycle — render-frame submission, result drain, and the
//! sim-thread tick loop.
//!
//! `tick_loop` owns the outer pacing + command-drain cycle that
//! `RkpEngine::spawn` launches on the engine thread.
//! `submit_render_frame` builds the per-frame `RenderFrame` snapshot
//! and ships it to the render worker; `drain_render_results` consumes
//! the corresponding `RenderResult`s on the return channel.




use super::picking_ops::collect_ghost_primitives;
use super::state::EngineState;

impl EngineState {
    /// Build a [`RenderFrame`] snapshot from current ECS / environment
    /// state and submit it to the render thread.
    ///
    /// Sim does no GPU work directly anymore — every per-frame thing the
    /// renderer used to read off `EngineState` is now packaged into a
    /// snapshot and shipped over `render_worker.inbox`. The render
    /// thread consumes, encodes, submits, and returns a
    /// [`RenderResult`] back via `render_worker.outbox` (which we drain
    /// in [`Self::drain_render_results`] called from the tick loop).
    ///
    /// Returns the CPU phases for this submission (setup vs. snapshot
    /// build vs. submit-handoff). The post-submit bucket reflects the
    /// time spent waiting for render-thread results, which is also a
    /// proxy for GPU backpressure.
    ///
    /// Originally a 700-line method that owned both the build *and* the
    /// GPU work. The latter migrated to [`crate::render_worker`]; what
    /// remains here is purely sim-side data assembly.
    pub(crate) fn submit_render_frame(&mut self) {
        use crate::viewport::ViewportId;
        let frame_start = std::time::Instant::now();

        // Phase A1 (docs/PERF_DEBT.md) — drain the typed mutation log
        // accumulated during the previous tick. No consumers wired
        // yet; this just keeps the log from growing unbounded while
        // we incrementally adopt the event-driven pattern. Phase B/C
        // replace this with actual event-driven derived-state updates.
        self.mutation_log.drain_and_log();

        // ── [sculpt-pipeline-sim] phase timings ────────────────────────
        // To debug the bump→submit gap surfaced by `[sculpt-pipeline]`,
        // capture each major sim-side phase's wall time and emit one
        // breakdown log when the tick processes (or carries forward)
        // a pending geometry bump. Cheap (just Instant::elapsed) and
        // only logs when relevant.
        let phase_pre_drain = std::time::Instant::now();
        let (pre_bump, pre_submit) = {
            let sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
            (sm.last_geometry_bump_ns(), sm.last_geometry_submit_ns())
        };

        // 0. Drain RenderResults that landed since last submit. The
        //    render thread runs on its own pace; the latest result it
        //    finished publishing carries the freshest pick decoding,
        //    cloud-sun atten, and GPU pass timings for us to fold back
        //    into sim state before we build the next snapshot.
        self.drain_render_results();
        let phase_drain_ms = phase_pre_drain.elapsed().as_secs_f64() * 1000.0;

        // 0a. Material palette — built every tick and shipped in the
        //     snapshot. Render uploads every frame. Cheap (small Vec)
        //     and robust to snapshot drops; the old "ship only when
        //     dirty" pattern could lose the upload if its carrying
        //     snapshot was dropped by the newest-wins inbox before
        //     render saw it.
        let (materials, shader_params_slots) = {
            let registry = &self.user_shader_registry;
            // Two separate dispatch ids on each material:
            //   * `shader_id`          → shade-pass dispatch. Resolved
            //     ONLY for shaders with a `shade` hook; otherwise stays
            //     0 so the shade pass takes the PBR path. (Resolving
            //     for non-shade shaders would route the dispatcher
            //     through its identity arm and emit raw albedo, which
            //     tone-maps to black against direct sun.)
            //   * `instance_shader_id` → band-cell descent dispatch
            //     (Phase B-redux). Resolved ONLY for shaders with an
            //     `instance_at` hook; the march reads it on a band-cell
            //     hit to find the prototype asset.
            // A shader can populate one, both, or neither.
            let palette = self.material_lib.build_palette(
                &|name| {
                    registry
                        .entries()
                        .iter()
                        .find(|e| e.name == name && e.shade_text.is_some())
                        .map(|e| e.id)
                },
                &|name| {
                    // V1 mesh-path geometry-emitting shaders. Populates
                    // `GpuMaterial.instance_shader_id`; the orchestration
                    // layer routes mesh-path materials to
                    // `tick_user_shader_mesh`.
                    registry
                        .entries()
                        .iter()
                        .find(|e| e.name == name && e.is_mesh_path())
                        .map(|e| e.id)
                },
            );
            let params = self.material_lib.build_shader_params(registry);
            (palette, params)
        };
        // Compose the shade-pass chunk once per tick. Cheap (small
        // string) and the render thread compares the hash to skip
        // pipeline rebuilds when nothing changed.
        let composed = rkp_render::shader_composer::compose(&self.user_shader_registry);
        let user_shader_shade_chunk = composed.shade;
        let _ = composed.generate; // band-cell BFS strip — no consumer
        let user_shader_source_hash = self.user_shader_registry.source_hash();
        let user_shader_infos = self.user_shader_registry.shader_infos();
        // Full registry entries — render thread reads these to drive the
        // proto bake (one bake per shader_id with an `instance_at` hook)
        // and (TODO Phase 9) the new emit pass. Cheap `Arc::clone` —
        // the registry holds entries inside `Arc<Vec<…>>` so the per-
        // tick handoff is a refcount bump (PERF_DEBT A3).
        let user_shader_entries = self.user_shader_registry.entries_arc();
        // Painted-material scan: walks each entity's leaf_attr range to
        // find which shader-bearing materials are present (entity-level
        // fallback or painted leaves). Cached on (paint_epoch,
        // geometry_epoch). The new emit pass (Phase 9) consumes this
        // cache to dispatch per painted leaf × density. The
        // ShaderRegionRequest construction was deleted along with the
        // band-cell BFS pipeline.
        let infos = self.user_shader_registry.shader_infos();

        // Build the set of "shader-bearing material ids" — materials
        // whose shader has either a `generate` hook (Phase C per-cell
        // pipeline) or an `instance_at` hook (Phase B-redux band-cell
        // derivation). Same painted-AABB scan feeds both kinds; the
        // per-tile emit loop below partitions on shader kind.
        let mut shader_materials: std::collections::HashMap<
            u16,
            rkp_render::shader_composer::UserShaderInfo,
        > = std::collections::HashMap::new();
        let any_shader_pipeline = infos
            .iter()
            .any(|i| i.has_generate || i.has_vs);
        if any_shader_pipeline {
            for slot_id in 0..self.material_lib.slot_count() as u16 {
                let Some(def) = self.material_lib.get_def(slot_id) else { continue; };
                let Some(shader_name) = def.shader.as_deref() else { continue; };
                let Some(info) = infos.iter().find(|i| i.name == shader_name) else { continue; };
                if info.has_generate || info.has_vs {
                    shader_materials.insert(slot_id, info.clone());
                }
            }
        }

        // Rebuild GPU objects from ECS world BEFORE constructing
        // ShaderRegionRequests below — those requests carry the host
        // instance's `overlay_offset`/`overlay_count`, which the
        // user-shader-pass uses to find painted material at each
        // anchor. Reading stale values here causes the BFS probe to
        // see last frame's overlay slice while the GPU buffer holds
        // this frame's content, missing the latest paint.
        let gpu_objects_dirty_this_frame = self.gpu_objects_dirty;
        let phase_update_scene_gpu_t0 = std::time::Instant::now();
        if self.gpu_objects_dirty {
            let profile = self.paint_profile_active();
            let t0 = std::time::Instant::now();
            self.update_scene_gpu();
            if profile {
                use std::sync::atomic::{AtomicU64, Ordering};
                static LAST_NS: AtomicU64 = AtomicU64::new(0);
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                let prev = LAST_NS.swap(now_ns, Ordering::Relaxed);
                let gap_ms = if prev == 0 { 0.0 } else { (now_ns.saturating_sub(prev)) as f64 / 1.0e6 };
                eprintln!(
                    "[paint] update_scene_gpu dt={:?} instances={} overlays={} gap_since_last={:.1}ms",
                    t0.elapsed(),
                    self.gpu_instances.len(),
                    self.gpu_instance_overlays.len(),
                    gap_ms,
                );
            }
            self.gpu_objects_dirty = false;
        }
        let phase_update_scene_gpu_ms =
            phase_update_scene_gpu_t0.elapsed().as_secs_f64() * 1000.0;

        let phase_painted_walk_t0 = std::time::Instant::now();
        if !shader_materials.is_empty() {
            // Reconcile the per-entity painted-material cache against
            // current paint + geometry epochs.
            //
            // - paint_epoch advances only via `apply_paint_stamp`, which
            //   also adds the painted entity to `painted_dirty_entities`.
            //   So a paint-only frame walks just that one entity.
            // - geometry_epoch advances on any voxel-pool / octree write
            //   (asset load, voxelize, bake, sculpt). We mirror the old
            //   "wipe all" behavior here: every renderable goes into the
            //   dirty set so the next walk pass rebuilds them.
            // Lock-free walk path. We acquire `scene_mgr` only long
            // enough to (a) read both epoch counters and (b) take an
            // `Arc`-shared `WalkSnapshot` of the three pool buffers
            // the scan needs. The `O(tree)` walks themselves run
            // outside the lock — render and other sim paths can
            // proceed in parallel.
            //
            // Snapshot construction is O(1) (three `Arc::clone`s) —
            // pool data lives behind `Arc<Vec<…>>` so there is no
            // memcpy on the geometry-bump frame either. The walk's
            // outstanding snapshot will trigger a one-time
            // copy-on-write inside the affected pool's next mutation
            // (via `Arc::make_mut`); dropping the snapshot promptly
            // after the walk lets subsequent writes stay in place.
            // See PERF_DEBT.md A2.
            let (cur_paint, cur_geom, snapshot_opt) = {
                let mut sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
                let cur_paint = sm.paint_epoch();
                let cur_geom = sm.geometry_epoch();
                // Skip the snapshot rebuild on frames where there's
                // no work to do (no dirty entities AND no geometry
                // change). This keeps the brief lock-and-clone path
                // off frames that don't reach the walk loop below.
                let want_snapshot = !self.painted_dirty_entities.is_empty()
                    || cur_geom != self.painted_materials_geometry_epoch;
                let snapshot = if want_snapshot {
                    Some(sm.walk_snapshot())
                } else {
                    None
                };
                (cur_paint, cur_geom, snapshot)
            };
            let geom_changed =
                cur_geom != self.painted_materials_geometry_epoch;
            if geom_changed {
                use crate::components::Renderable;
                // Blanket-invalidate ONLY when nobody told us which
                // entities changed (asset load, bake worker, world
                // reshuffle). When `painted_dirty_entities` already
                // has specific entries — populated by `apply_sculpt_
                // stamp`, paint, etc. — trust them: the walk below
                // re-scans those entities' painted aabbs and updates
                // their `painted_per_entity` cache. Other entities'
                // cached scans stay valid (their octree topology
                // didn't change).
                //
                // This is the difference between O(world)
                // re-scan (~586 ms on splat5) and O(stamp footprint)
                // re-scan (~ms). The dominant component of the
                // `[sculpt-pipeline] bump→submit` gap.
                if self.painted_dirty_entities.is_empty() {
                    self.painted_per_entity.clear();
                    for (entity, _) in self.world.query::<&Renderable>().iter() {
                        self.painted_dirty_entities.insert(entity);
                    }
                }
            }

            let painted_walk_profile = self.paint_profile_active();
            let painted_walk_t0 = std::time::Instant::now();
            let dirty_count = self.painted_dirty_entities.len();
            if dirty_count > 0 {
                use crate::components::Renderable;
                // The snapshot is guaranteed `Some` here: `want_snapshot`
                // was true iff dirty was non-empty OR geom_changed (which
                // also adds to dirty above), so reaching this branch
                // implies we asked for a snapshot.
                let snapshot = snapshot_opt.expect(
                    "walk_snapshot must be present when dirty set is non-empty",
                );
                // Drain rather than iterate-then-clear so we can mutate
                // `painted_per_entity` from inside the loop without
                // double-borrowing `self`. `mem::take` keeps the dirty
                // set's allocation around for the next stamp.
                let dirty: std::collections::HashSet<hecs::Entity> =
                    std::mem::take(&mut self.painted_dirty_entities);
                for entity in dirty {
                    // Despawned-while-dirty: drop any stale cache so
                    // the flat concat below doesn't carry phantoms.
                    if !self.world.contains(entity) {
                        self.painted_per_entity.remove(&entity);
                        continue;
                    }
                    let (root_offset, depth, grid_origin, base_voxel_size) = {
                        let Ok(r) = self.world.get::<&Renderable>(entity) else {
                            self.painted_per_entity.remove(&entity);
                            continue;
                        };
                        let Some(spatial) = r.spatial.as_ref().and_then(|g| g.as_octree()) else {
                            self.painted_per_entity.remove(&entity);
                            continue;
                        };
                        (
                            spatial.root_offset,
                            spatial.depth,
                            spatial.grid_origin,
                            spatial.base_voxel_size,
                        )
                    };
                    let Some(&gpu_idx) = self.entity_to_gpu.get(&entity) else {
                        // No GPU instance yet — keep the entity dirty
                        // for a later tick (re-add) and leave any old
                        // cache alone. Without re-adding the walk would
                        // never come back to it after the gpu mapping
                        // appears (the dirty set has been drained).
                        self.painted_dirty_entities.insert(entity);
                        continue;
                    };
                    let _ = gpu_idx;

                    // Skip the O(octree) scan when we already know this
                    // entity has no shader-bearing materials. The cache
                    // is invalidated by paint Material stamps and
                    // sculpt Raise (the only paths that can introduce
                    // new shader-bearing materials). Carve sculpts on
                    // an unpainted asset hit this fast path — saves
                    // ~150 ms per stamp on the splat5 elephant.
                    if self.entities_known_empty.contains(&entity) {
                        continue;
                    }

                    let mut entry = super::state::EntityPaintedCache::default();
                    scan_painted_aabbs(
                        &snapshot.octree_data,
                        &snapshot.brick_pool_data,
                        &snapshot.leaf_attr_data,
                        self.paint_overlays.get(&entity),
                        root_offset as usize,
                        depth,
                        grid_origin,
                        base_voxel_size,
                        &shader_materials,
                        &mut entry.mat_tiles,
                    );
                    if entry.mat_tiles.is_empty() {
                        self.painted_per_entity.remove(&entity);
                        self.entities_known_empty.insert(entity);
                    } else {
                        self.painted_per_entity.insert(entity, entry);
                        self.entities_known_empty.remove(&entity);
                    }
                }
            }

            // Rebuild the flat views whenever per-entity contents
            // changed (dirty set was non-empty) OR object_id mappings
            // shifted (gpu_objects rebuild this frame). The latter
            // matters because `painted_materials` is keyed by
            // object_id — without rebuilding, a frame that moves
            // entity A from object_id=3 to object_id=4 leaves a stale
            // entry under the old key.
            // The flat rebuild only needs to run when painted content
            // or the entity-to-gpu mapping actually changed. Dropping
            // `gpu_objects_dirty_this_frame` from the trigger so that
            // animation-only frames (which set `gpu_objects_dirty` via
            // `animation::tick` every frame a skeleton is playing)
            // don't churn through cloning every entity's `mat_tiles`
            // and re-running the new mesh-path compute trio. Animation
            // doesn't move entity world transforms (just bones within
            // them), so painted_anchors / painted_materials are
            // content-stable across animation ticks.
            //
            // Caveat: if a Renderable entity is added or removed without
            // any paint or geometry epoch change, the `entity_to_gpu`
            // mapping can shift but we won't refresh `painted_materials`
            // here. In practice the entity add/remove paths already mark
            // affected entities dirty (or bump geom_epoch), so this
            // shouldn't trigger in normal use.
            let need_flat_rebuild = dirty_count > 0 || geom_changed;
            if need_flat_rebuild {
                self.painted_materials.clear();
                let mut new_painted_anchors: std::collections::HashMap<
                    u16,
                    Vec<rkp_render::user_shader_mesh_pass::AnchorRecord>,
                > = std::collections::HashMap::new();
                for (entity, entry) in &self.painted_per_entity {
                    let Some(&gpu_idx) = self.entity_to_gpu.get(entity) else {
                        continue;
                    };
                    if !entry.mat_tiles.is_empty() {
                        self.painted_materials
                            .insert(gpu_idx as u32, entry.mat_tiles.clone());
                    }

                    // V1 mesh-path AnchorRecords from the per-tile
                    // PaintedTileEntry table. Two bounds in play:
                    //
                    //   · **Tile cube** — derived from `tile_coord ×
                    //     tile_size` (object-local), transformed to
                    //     world. Stable across frames as paint
                    //     extends inside the tile, so blade
                    //     positions don't shimmer.
                    //   · **Painted-leaf AABB** (`te.aabb`) — only
                    //     used to pick `surface_y` (the y the blade
                    //     base sits on). Stable for flat-ground
                    //     paint; deferred concern for slopes.
                    //
                    // When `tile_size` is `None` (shader didn't
                    // declare `@tile_size`), the tile_coord is
                    // `NO_TILE_COORD` and tile cube bounds fall back
                    // to the painted-leaf AABB — degraded but
                    // deterministic.
                    let object_id = gpu_idx as u32;
                    let entity_world: Option<glam::Mat4> = self
                        .gpu_instances
                        .iter()
                        .find(|i| i.object_id == object_id)
                        .map(|i| glam::Mat4::from_cols_array_2d(&i.world));
                    for (&mat, tiles) in &entry.mat_tiles {
                        let tile_size = shader_materials
                            .get(&mat)
                            .and_then(|i| i.tile_size);
                        let bucket = new_painted_anchors.entry(mat).or_default();
                        bucket.reserve(tiles.len());
                        for (&tile_coord, te) in tiles {
                            // Tile cube bounds (object-local). When no
                            // `@tile_size`, fall back to painted-leaf
                            // bounds for a defined-but-coarse anchor.
                            let (tile_local_min, tile_local_max) = match tile_size {
                                Some(s) if s > 0.0
                                    && tile_coord != [i32::MIN, i32::MIN, i32::MIN] =>
                                {
                                    let lo = glam::Vec3::new(
                                        tile_coord[0] as f32 * s,
                                        tile_coord[1] as f32 * s,
                                        tile_coord[2] as f32 * s,
                                    );
                                    (lo, lo + glam::Vec3::splat(s))
                                }
                                _ => (te.aabb.min, te.aabb.max),
                            };
                            let (tile_world_min, tile_world_max) = transform_aabb_to_world(
                                tile_local_min,
                                tile_local_max,
                                entity_world,
                            );
                            // Painted-leaf BB world bounds — actual
                            // paint coverage in this tile (te.aabb is
                            // the union of every painted leaf cell's
                            // object-local AABB). The shader spawns
                            // blades inside paint_min/max, so blades
                            // land on the painted area instead of the
                            // unpainted parts of the tile cube. This
                            // BB grows when paint extends within the
                            // tile (jitter on active paint; stable for
                            // a left-alone painted region).
                            let (paint_world_min, paint_world_max) = transform_aabb_to_world(
                                te.aabb.min,
                                te.aabb.max,
                                entity_world,
                            );
                            // Surface y = top of painted leaves in
                            // world. Blade base sits on this y so
                            // blades grow up FROM the painted ground
                            // (not from the tile-cube floor, which
                            // could be tile_size below).
                            let surface_y = paint_world_max.y;
                            // Per-tile surface normal: normalize the
                            // object-local sum of painted-leaf normals,
                            // rotate by the entity's world transform,
                            // re-normalize. Fall back to +Y on either
                            // degenerate stage (cancelling normals or
                            // an entity transform that maps Y to 0).
                            let local_normal = te.normal_sum
                                .try_normalize()
                                .unwrap_or(glam::Vec3::Y);
                            let world_normal = entity_world
                                .map(|m| m.transform_vector3(local_normal))
                                .unwrap_or(local_normal)
                                .try_normalize()
                                .unwrap_or(glam::Vec3::Y);
                            // Stable seed: tile coord + material.
                            // Dropped `object_id` because `gpu_idx`
                            // can shuffle on `gpu_objects` rebuild
                            // (which the paint hot path triggers via
                            // `gpu_objects_dirty`), which re-randomized
                            // every blade on every paint stamp. Same
                            // local tile_coord across entities will
                            // correlate yaws/jitters; acceptable for
                            // V1.
                            let seed = rkp_render::user_shader_mesh_pass::anchor_seed([
                                tile_coord[0] as f32,
                                tile_coord[1] as f32,
                                tile_coord[2] as f32,
                            ]) ^ (mat as u32).wrapping_mul(0x9E37_79B9);
                            bucket.push(
                                rkp_render::user_shader_mesh_pass::AnchorRecord {
                                    tile_min: tile_world_min.to_array(),
                                    material_id: mat as u32,
                                    tile_max: tile_world_max.to_array(),
                                    leaf_count: te.leaf_count,
                                    paint_min: paint_world_min.to_array(),
                                    object_id,
                                    paint_max: paint_world_max.to_array(),
                                    surface_y,
                                    surface_normal: world_normal.to_array(),
                                    seed,
                                },
                            );
                        }
                    }
                }
                // Debug: detect per-tile seed instability. Compare each
                // tile's (object_id, tile_min) → seed against last
                // rebuild's value; only log when the seed for an
                // already-known tile actually changed. Quiet when
                // things are stable; loud only when there's a real bug.
                // `RKP_GRASS_DEBUG=1` enables.
                if std::env::var("RKP_GRASS_DEBUG").is_ok() {
                    let mut changed = 0u32;
                    let mut new_tiles = 0u32;
                    let mut dropped = 0u32;
                    let mut cur_map: std::collections::HashMap<
                        (u32, u32, u32, u32, u16),
                        u32,
                    > = std::collections::HashMap::new();
                    for (&mat, bucket) in &new_painted_anchors {
                        for a in bucket {
                            // Quantize tile_min to mm to dodge fp noise.
                            let key = (
                                a.object_id,
                                (a.tile_min[0] * 1000.0).round() as i32 as u32,
                                (a.tile_min[1] * 1000.0).round() as i32 as u32,
                                (a.tile_min[2] * 1000.0).round() as i32 as u32,
                                mat,
                            );
                            cur_map.insert(key, a.seed);
                        }
                    }
                    if let Some(last) = self.debug_last_anchor_seeds.as_ref() {
                        for (k, &cur_seed) in &cur_map {
                            match last.get(k) {
                                Some(&prev_seed) if prev_seed != cur_seed => {
                                    changed += 1;
                                    if changed <= 5 {
                                        eprintln!(
                                            "[grass-debug] SEED CHANGED obj={} tile_min=({:.4},{:.4},{:.4}) mat={} prev=0x{:08x} cur=0x{:08x}",
                                            k.0,
                                            k.1 as i32 as f32 / 1000.0,
                                            k.2 as i32 as f32 / 1000.0,
                                            k.3 as i32 as f32 / 1000.0,
                                            k.4,
                                            prev_seed,
                                            cur_seed,
                                        );
                                    }
                                }
                                None => new_tiles += 1,
                                _ => {}
                            }
                        }
                        for k in last.keys() {
                            if !cur_map.contains_key(k) {
                                dropped += 1;
                            }
                        }
                    } else {
                        new_tiles = cur_map.len() as u32;
                    }
                    eprintln!(
                        "[grass-debug] rebuild paint={} geom={} dirty={} mats={} anchors={} changed={} new={} dropped={}",
                        cur_paint, cur_geom, dirty_count,
                        new_painted_anchors.len(),
                        cur_map.len(),
                        changed, new_tiles, dropped,
                    );
                    self.debug_last_anchor_seeds = Some(cur_map);
                }

                self.painted_anchors = std::sync::Arc::new(new_painted_anchors);
            }

            self.painted_materials_paint_epoch = cur_paint;
            self.painted_materials_geometry_epoch = cur_geom;

            if painted_walk_profile && (dirty_count > 0 || geom_changed) {
                eprintln!(
                    "[paint] painted_materials_walk dt={:?} entities_walked={} cached_entities={} shader_materials={} geom_changed={}",
                    painted_walk_t0.elapsed(),
                    dirty_count,
                    self.painted_per_entity.len(),
                    shader_materials.len(),
                    geom_changed,
                );
            }
        }
        let phase_painted_walk_ms =
            phase_painted_walk_t0.elapsed().as_secs_f64() * 1000.0;

        // Clear the dirty flag so any other consumers (UI, etc.)
        // know the palette they observed has been published. We
        // ship every tick regardless, so the flag is purely for
        // outside-of-render bookkeeping now.
        self.material_lib.clear_dirty();

        // MAIN camera first: atmosphere LUTs + sun-light tinting both
        // depend on its altitude (scene-wide values shared across VRs).
        let main_cam = self.build_camera_uniforms(ViewportId::MAIN);
        let cam_y = main_cam.position[1];

        // Cloud-sun atten: smooth toward the latest render-thread
        // readback (fed in via `last_cloud_sun_atten_raw` by
        // `drain_render_results`). NaN sentinel = render hasn't
        // published one yet (first frame, MAIN hidden), so we hold the
        // last EMA target.
        let target_atten = if self.environment.attenuate_sun_by_clouds
            && self.environment.clouds_enabled
        {
            if self.last_cloud_sun_atten_raw.is_nan() {
                self.cloud_sun_atten
            } else {
                self.last_cloud_sun_atten_raw
            }
        } else {
            1.0
        };
        self.cloud_sun_atten += (target_atten - self.cloud_sun_atten) * 0.04;

        // Sun + entity-driven point/spot lights, all in the order the
        // shade shader expects (entry 0 = sun).
        let mut sun_light = self.environment.to_gpu_light(cam_y);
        sun_light.color[0] *= self.cloud_sun_atten;
        sun_light.color[1] *= self.cloud_sun_atten;
        sun_light.color[2] *= self.cloud_sun_atten;
        let mut gpu_lights = vec![sun_light];
        for (_entity, (transform, pl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::PointLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 1.0],
                color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                direction: [0.0, 0.0, 0.0, 0.0],
                params: [pl.range, 0.0, 0.0, if pl.cast_shadow { 1.0 } else { 0.0 }],
            });
        }
        for (_entity, (transform, sl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::SpotLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 2.0],
                color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                direction: [
                    sl.direction.x,
                    sl.direction.y,
                    sl.direction.z,
                    sl.outer_angle.to_radians(),
                ],
                params: [
                    sl.range,
                    sl.inner_angle.to_radians(),
                    0.0,
                    if sl.cast_shadow { 1.0 } else { 0.0 },
                ],
            });
        }

        let mut shade_params = self.environment.to_shade_params(cam_y);
        shade_params.num_lights = gpu_lights.len() as u32;
        // Engine clock for user shaders that need a time input (hologram
        // scroll, fresnel pulse). Frame-index based at 60 Hz — same
        // convention used elsewhere (cloud_params). Wraps at ~414 days.
        shade_params.time = self.frame_index as f32 / 60.0;
        self.shade_params_base = shade_params;
        self.num_lights_cache = shade_params.num_lights;

        // Env update — shipped every tick (cheap; render writes a few
        // u32-sized queue.write_buffers). Same drop-safety rationale
        // as `materials`.
        let env_update = crate::render_frame::EnvUpdate {
            exposure: self.environment.exposure,
            bloom_threshold: self.environment.bloom_threshold,
            bloom_knee: self.environment.bloom_knee,
            bloom_intensity: self.environment.bloom_intensity,
        };
        // Clear the legacy flag for other consumers; render no longer
        // gates on it.
        self.environment_dirty = false;

        // GPU objects rebuild was relocated upstream so it runs
        // BEFORE the user_shader_regions request loop reads
        // `inst.overlay_offset`/`overlay_count`. See comment there.

        let t_cpu_setup = frame_start.elapsed();
        // Sub-phase timing inside the snapshot build, gated on
        // `RKP_SNAP_PROFILE=1`. Drops the per-phase ms split into
        // stderr each frame so we can attribute snapshot wallclock
        // when the cumulative `snapshot_ms` blows up.
        let snap_profile = std::env::var("RKP_SNAP_PROFILE").is_ok();
        let snap_phase_start = std::time::Instant::now();

        // 1. Geometry epoch — read lock-free via the shared atomic
        //    handle. Render compares against its own last-uploaded
        //    epoch and re-uploads when behind. Robust to dropped
        //    snapshots: the next snapshot still carries the latest
        //    epoch, so render always catches up.
        //
        //    The lock-free read is what keeps sim at 60 Hz while
        //    bake_worker is busy — taking `scene_mgr.lock()` here
        //    would block sim for the full duration of any bake
        //    integrate (50 ms+).
        //
        //    The legacy `self.geometry_dirty` flag is kept for collider
        //    rebuild scheduling (independent of GPU upload). It's set
        //    by every code path that mutates scene geometry.
        let geometry_epoch = self
            .geometry_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);
        if self.geometry_dirty {
            self.collider_caches_dirty = true;
            self.geometry_dirty = false;
        }
        if self.collider_caches_dirty {
            self.rebuild_collider_caches();
            self.collider_caches_dirty = false;
        }

        let snap_t_geom = snap_phase_start.elapsed();

        // 2. Bone matrix bytes for shading (LBS + DQ paths). Cheap
        //    `Arc::clone` — the allocator now holds the bytes inside
        //    `Arc<Vec<u8>>` so the per-tick handoff costs a refcount
        //    bump instead of the ~58 MB memcpy this used to do via
        //    `.to_vec()`. See PERF_DEBT.md A3.
        let bone_matrix_lbs = self.bone_matrix_allocator.bytes_arc();
        let bone_matrix_dqs = self.bone_matrix_allocator.bytes_dq_arc();

        // 2b. Skin scatter — fold per-entity dispatches into one
        //     batched compute dispatch sim-side; render fires the
        //     batch on its thread. `skin_reuse` short-circuits when
        //     every skinned pose was byte-identical to the previous
        //     frame (paused animation), in which case the bone_field
        //     buffer from last frame is still valid and the scatter
        //     can skip entirely.
        let skin = if self.skinning_enabled
            && !self.skin_dispatches.is_empty()
            && !self.skin_reuse
        {
            self.skin_batch.clear();
            for plan in &self.skin_dispatches {
                let d = rkp_render::SkinDispatch {
                    uniforms: plan.uniforms,
                    bricks: &plan.bricks,
                };
                self.skin_batch.push(&d);
            }
            Some(crate::render_frame::RenderSkin {
                bone_field_bytes: self.skin_bone_field_bytes,
                bone_field_occ_bytes: self.skin_bone_field_occ_bytes,
                batch: self.skin_batch.clone(),
            })
        } else {
            if self.skinning_enabled && self.frame_index % 60 == 0 {
                // Once a second, log why scatter isn't running when
                // the user has the toggle on — most common reason is
                // a stale `.rkp` without the new skin-meta section.
                let skinned_entities = self
                    .world
                    .query::<&crate::components::Skeleton>()
                    .iter()
                    .count();
                if skinned_entities > 0 {
                    eprintln!(
                        "[RkpEngine] skinning enabled, {} skinned entities, but 0 scatter dispatches this frame. \
                         Likely cause: stale .rkp without skin-meta section — re-import the asset.",
                        skinned_entities,
                    );
                }
            }
            None
        };

        let snap_t_bone = snap_phase_start.elapsed();

        // 3. Per-viewport snapshot build — derive every per-VR
        //    parameter the render thread needs from current sim state
        //    and stash it in `viewports` for the snapshot. No GPU
        //    calls; the render thread does all the actual encoding
        //    and submission against this data.
        let visible_ids: Vec<ViewportId> = self
            .viewports
            .iter()
            .filter(|(_, v)| v.visible)
            .map(|(id, _)| *id)
            .collect();

        // Gizmo overlay is drawn on MAIN only — selection state is global.
        let gizmo_verts_main = self.build_gizmo_wireframe();
        let mut vp_list: Vec<crate::render_frame::RenderViewport> =
            Vec::with_capacity(visible_ids.len());

        for &viewport_id in &visible_ids {
            let cam_uniforms = self.build_camera_uniforms(viewport_id);
            let (vp_w, vp_h) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.width, v.height))
                .expect("viewport must exist");

            // Per-viewport screen-AABBs (camera-dependent) for tile cull.
            let vp_matrix = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            let screen_aabbs = crate::scene_sync::compute_screen_aabbs(
                &self.gpu_instances,
                &self.gpu_assets,
                &vp_matrix,
                vp_w as f32,
                vp_h as f32,
            );
            let screen_aabbs_bytes: Vec<u8> = bytemuck::cast_slice(&screen_aabbs).to_vec();
            // Per-tile object lists — replaces the 32-object bitmask so
            // the march shader handles arbitrary scene object counts.
            let tile_lists = crate::scene_sync::build_tile_lists(
                &screen_aabbs, vp_w, vp_h,
            );
            let tile_offsets_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.offsets).to_vec();
            let tile_object_ids_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.object_ids).to_vec();
            let tile_count_x = tile_lists.tile_count_x;

            // Per-VR vol/cloud/atmo/god-ray params — derived from
            // environment + this VR's camera. Render writes them into
            // the corresponding per-VR uniform buffers right before
            // submit (one submit per VR keeps the writes correctly
            // paired with their dispatches).
            let vol_params = self.environment.to_volumetric_params(
                &cam_uniforms,
                vp_w,
                vp_h,
                self.frame_index as u32,
            );
            let cloud_params =
                self.environment.to_cloud_params(self.frame_index as f32 / 60.0);

            let sun_d = self.environment.sun_direction();
            let cam_y_vp = cam_uniforms.position[1];
            let atmo_frame = rkp_render::rkp_atmosphere::AtmosphereFrameParams {
                sun_dir: [-sun_d[0], -sun_d[1], -sun_d[2]],
                sun_intensity: self.environment.sun_intensity,
                camera_altitude: self.environment.effective_altitude(cam_y_vp),
                ground_albedo: self.environment.ground_albedo,
                cam_pos: [
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                ],
                _pad1b: 0.0,
                cam_forward: [
                    cam_uniforms.forward[0],
                    cam_uniforms.forward[1],
                    cam_uniforms.forward[2],
                ],
                _pad2: 0.0,
                cam_right: [
                    cam_uniforms.right[0],
                    cam_uniforms.right[1],
                    cam_uniforms.right[2],
                ],
                _pad3: 0.0,
                cam_up: [
                    cam_uniforms.up[0],
                    cam_uniforms.up[1],
                    cam_uniforms.up[2],
                ],
                _pad4: 0.0,
            };

            let god_ray_params = {
                let sun_toward = [-sun_d[0], -sun_d[1], -sun_d[2]];
                let sun_world = glam::Vec3::new(
                    cam_uniforms.position[0] + sun_toward[0] * 1000.0,
                    cam_uniforms.position[1] + sun_toward[1] * 1000.0,
                    cam_uniforms.position[2] + sun_toward[2] * 1000.0,
                );
                let clip = vp_matrix * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);
                let sun_on_screen = if clip.w > 0.0 { 1.0 } else { 0.0 };
                let ndc = if clip.w > 0.0 {
                    glam::Vec2::new(clip.x / clip.w, clip.y / clip.w)
                } else {
                    glam::Vec2::ZERO
                };
                let sun_uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];
                rkp_render::rkp_god_rays::GodRayParams {
                    sun_screen_pos: sun_uv,
                    sun_on_screen,
                    density: self.environment.god_ray_density,
                    weight: self.environment.god_ray_weight,
                    decay: self.environment.god_ray_decay,
                    exposure: self.environment.god_ray_exposure,
                    num_samples: 64,
                    sun_color: self.environment.sun_tint(cam_y_vp),
                    _pad: 0.0,
                }
            };

            let (vp_mode, vp_preview_mode) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.mode, v.preview_mode))
                .unwrap_or((
                    rkp_render::RenderMode::InSitu,
                    rkp_render::BuildPreviewMode::Voxel,
                ));

            // The procedural being previewed in raymarch mode is
            // always the currently-selected entity — keeps the
            // preview following selection automatically.
            let vp_preview_entity = self.selected_entity.and_then(|entity| {
                if self
                    .world
                    .get::<&crate::components::ProceduralGeometry>(entity)
                    .is_ok()
                {
                    self.entity_uuids.get(&entity).copied()
                } else {
                    None
                }
            });

            // Per-VR shade params: same scene-wide values plus the
            // per-VR `isolation` flag and a clamp on the light count
            // when isolated (so the BUILD preview doesn't pick up
            // the main scene's point lights).
            let isolation = matches!(vp_mode, rkp_render::RenderMode::Isolation);
            let mut shade_params_vr = self.shade_params_base;
            shade_params_vr.isolation = isolation as u32;
            if isolation {
                shade_params_vr.num_lights = shade_params_vr.num_lights.min(1);
            }
            // Paint cursor overlay — MAIN only. The brush sphere is a
            // viewport-camera-centric tool; a BUILD preview of the
            // same scene shouldn't show it. When the engine isn't in
            // paint mode `brush_active` stays 0 and the shader skips
            // the overlay entirely.
            //
            // The cursor's world center comes from the brush-state
            // probe pass (which reads gbuf_position at the cursor
            // pixel each frame), so engine just sets the static
            // bits — radius, color, the selected entity's
            // `gpu_idx` for the per-pixel selection-lock gate.
            let mut brush_pixel: Option<(u32, u32)> = None;
            // Paint and sculpt share the cursor visualization. Mutually
            // exclusive — the editor's toggle handlers guarantee at most
            // one is active. Sculpt uses a teal rim to distinguish it
            // from paint's warm yellow.
            let (active, radius, color): (bool, f32, [f32; 4]) =
                if viewport_id == ViewportId::MAIN && self.paint_mode_active {
                    (true, self.paint_mode_radius, [1.0, 0.85, 0.2, 1.0])
                } else if viewport_id == ViewportId::MAIN && self.sculpt_mode_active {
                    (true, self.sculpt_mode_radius, [0.2, 0.85, 0.85, 1.0])
                } else {
                    (false, 0.0, [0.0; 4])
                };
            if active {
                shade_params_vr.brush_active = 1;
                shade_params_vr.brush_radius = radius;
                shade_params_vr.brush_color = color;
                // Selection-lock: only paint/sculpt on the selected
                // entity. `u32::MAX` keeps the cursor hidden while
                // nothing is selected.
                shade_params_vr.brush_object_id = self
                    .selected_entity
                    .and_then(|e| self.entity_to_gpu.get(&e).copied())
                    .map(|i| i as u32)
                    .unwrap_or(u32::MAX);
                // Probe at the live mouse pixel iff the cursor is
                // inside the framebuffer.
                let mx = self.mouse_pos.x;
                let my = self.mouse_pos.y;
                if mx >= 0.0 && my >= 0.0 && (mx as u32) < vp_w && (my as u32) < vp_h {
                    brush_pixel = Some((mx as u32, my as u32));
                }
            }
            let bloom_composite_intensity = if isolation {
                0.0
            } else {
                self.environment.bloom_intensity
            };

            // Procedural raymarch state — only when this VR is in
            // raymarch preview mode AND a procedural entity is
            // selected. Sim flattens the tree, builds the AABB, and
            // pre-filters ghost primitives; render uploads + binds.
            let proc_raymarch =
                if matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
                    let entity = vp_preview_entity.and_then(|uuid| {
                        self.entity_uuids
                            .iter()
                            .find_map(|(e, u)| (*u == uuid).then_some(*e))
                    });

                    let (instructions, aabb_min, aabb_max) = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| {
                                    let ins = rkp_procedural::flatten_tree(&pg.tree);
                                    let bounds = rkp_procedural::compute_bounds(&pg.tree);
                                    (ins, bounds.min, bounds.max)
                                })
                        })
                        // Empty-AABB sentinel: -1..+1 degenerate box
                        // any sane ray-AABB slab test fails. Covers
                        // "raymarch enabled but no procedural entity
                        // selected" so we don't get a bogus hit.
                        .unwrap_or_else(|| {
                            (Vec::new(), glam::Vec3::splat(1.0), glam::Vec3::splat(-1.0))
                        });

                    // Any stable per-entity u32 works — the shader
                    // packs it into the material G channel for the
                    // (now-unused) old 8-bit pick byte; retained here
                    // only as a non-breaking placeholder until
                    // `ProcRaymarchParams.object_id` gets cleaned up.
                    let object_id = entity.map(|e| e.to_bits().get() as u32).unwrap_or(0);

                    let entity_world = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::Transform>(e)
                                .ok()
                                .map(|xf| {
                                    glam::Affine3A::from_scale_rotation_translation(
                                        xf.scale,
                                        glam::Quat::from_euler(
                                            glam::EulerRot::XYZ,
                                            xf.rotation.x.to_radians(),
                                            xf.rotation.y.to_radians(),
                                            xf.rotation.z.to_radians(),
                                        ),
                                        xf.position,
                                    )
                                })
                        })
                        .unwrap_or(glam::Affine3A::IDENTITY);

                    // Ghost overlay: every cutter-role primitive,
                    // regardless of selection. Filter the flattened
                    // instruction stream so ghost renders use the
                    // same composed transforms the main raymarch does.
                    let ghost_ids = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| collect_ghost_primitives(&pg.tree))
                        })
                        .unwrap_or_default();
                    let ghost_set: std::collections::HashSet<u32> =
                        ghost_ids.into_iter().collect();
                    let ghost_instructions: Vec<rkp_procedural::ProcInstruction> = instructions
                        .iter()
                        .filter(|ins| ghost_set.contains(&ins.node_id))
                        .copied()
                        .collect();

                    Some(crate::render_frame::RenderProcRaymarch {
                        instructions,
                        ghost_instructions,
                        object_id,
                        entity_world,
                        aabb_min,
                        aabb_max,
                        selected_node: self.selected_procedural_node,
                    })
                } else {
                    None
                };

            // Wireframe verts: gizmo on MAIN, procedural-node gizmo
            // on BUILD when in raymarch preview. The procedural-node
            // gizmo is only meaningful in raymarch mode — in voxel
            // mode the user sees the baked result and any drag would
            // silently edit the tree without visual feedback.
            let wireframe_verts = if viewport_id == ViewportId::MAIN {
                gizmo_verts_main.clone()
            } else if viewport_id == ViewportId::BUILD
                && matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch)
            {
                let cam_pos = glam::Vec3::new(
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                );
                self.build_procedural_gizmo_wireframe(cam_pos)
            } else {
                Vec::new()
            };

            // Editor-overlay gate. MAIN gates on the EDITOR_ONLY
            // layer bit (off in play mode); BUILD always shows its
            // proc-gizmo when one's present.
            let show_editor_overlays = if viewport_id == ViewportId::MAIN {
                self.viewports
                    .get(ViewportId::MAIN)
                    .map(|v| v.filter.base_layers & crate::viewport::layer::EDITOR_ONLY != 0)
                    .unwrap_or(false)
            } else {
                true
            };

            // BUILD: pin the studio-floor grid under the previewed
            // entity instead of world origin. Without this, moving
            // the entity in world-Y leaves the grid at y=0 while the
            // camera orbits around the entity, so the object floats
            // relative to the grid.
            let grid_override = if viewport_id == ViewportId::BUILD {
                let p = proc_raymarch
                    .as_ref()
                    .map(|p| p.entity_world.translation)
                    .unwrap_or(glam::Vec3A::ZERO);
                Some(rkp_render::rkp_grid::GridParams {
                    plane_origin: [p.x, p.y, p.z, 0.0],
                    ..Default::default()
                })
            } else {
                None
            };

            vp_list.push(crate::render_frame::RenderViewport {
                id: viewport_id,
                width: vp_w,
                height: vp_h,
                mode: vp_mode,
                preview_mode: vp_preview_mode,
                camera: cam_uniforms,
                screen_aabbs_bytes,
                tile_offsets_bytes,
                tile_object_ids_bytes,
                tile_count_x,
                vp_matrix,
                vol_params,
                cloud_params,
                atmo_frame,
                god_ray_params,
                shade_params: shade_params_vr,
                bloom_composite_intensity,
                grid_override,
                wireframe_verts,
                show_editor_overlays,
                proc_raymarch,
                brush_pixel,
            });

            // Update sim-side `prev_view_proj` so next frame's
            // CameraUniforms carry the right reprojection matrix for
            // cloud TAA / temporal upscale.
            if let Some(v) = self.viewports.get_mut(viewport_id) {
                v.prev_view_proj = cam_uniforms.view_proj;
            }
        }

        let snap_t_vrs = snap_phase_start.elapsed();

        // 4. Pending pick — convert sim's `PendingPick` (which carries
        //    a CPU-resolved ghost hint) to the render-side struct.
        //    Ghost hint stays sim-side; we'll re-apply it when the
        //    matching `PickResult` comes back.
        //
        //    Re-ship every snapshot until [`process_pick_result`]
        //    clears `self.pending_pick`. Picks used to be cleared
        //    eagerly with `take()`, but the GPU-backpressure backoff
        //    in `render_worker` now causes the inbox (newest-wins) to
        //    drop a sizeable fraction of snapshots before render sees
        //    them — eager-clearing meant the click was lost forever
        //    whenever its carrier snapshot got dropped. Re-shipping
        //    is safe because render's `pick_in_flight` gate dedupes
        //    duplicates: at most one map_async is ever in flight per
        //    pick request.
        let pending_pick = if let Some(pp) = self.pending_pick.as_ref() {
            // Map viewport+preview-mode → kind. BUILD raymarch decodes
            // the gbuf_pick texture for procedural NodeIds; everything
            // else (MAIN voxel, BUILD voxel) decodes gbuf_material for
            // the entity scene_id.
            let kind = if pp.viewport == ViewportId::BUILD
                && self
                    .viewports
                    .get(ViewportId::BUILD)
                    .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                    .unwrap_or(false)
            {
                crate::render_frame::PickKind::ProceduralNode
            } else {
                crate::render_frame::PickKind::Material
            };
            self.in_flight_pick_ghost = pp.ghost_pick_node_id;
            Some(crate::render_frame::PendingPick {
                viewport: pp.viewport,
                x: pp.x,
                y: pp.y,
                kind,
            })
        } else {
            None
        };

        let snap_t_pick = snap_phase_start.elapsed();

        // 5. Build + submit the snapshot. `submit` is non-blocking;
        //    if render hadn't consumed the previous snapshot yet,
        //    that one is dropped (newest-wins). Sim never stalls on
        //    render's GPU rate.
        let paint_epoch = self
            .paint_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);

        let frame = crate::render_frame::RenderFrame {
            frame_index: self.frame_index,
            // Cheap `Arc::clone`s — the underlying Vecs live behind
            // `Arc<Vec<…>>` in EngineState. Per-tick handoff costs a
            // refcount bump rather than ~30 KB × 6 of memcpy. See
            // PERF_DEBT.md A3.
            gpu_assets: std::sync::Arc::clone(&self.gpu_assets),
            gpu_instances: std::sync::Arc::clone(&self.gpu_instances),
            gpu_instance_overlays: std::sync::Arc::clone(&self.gpu_instance_overlays),
            gpu_instance_sculpts: std::sync::Arc::clone(&self.gpu_instance_sculpts),
            splat_draws: std::sync::Arc::clone(&self.splat_draws),
            proxy_draws: std::sync::Arc::clone(&self.proxy_draws),
            gpu_objects_dirty: gpu_objects_dirty_this_frame,
            geometry_epoch,
            paint_epoch,
            materials,
            shader_params_slots,
            user_shader_shade_chunk,
            user_shader_source_hash,
            user_shader_infos,
            user_shader_entries,
            painted_anchors: std::sync::Arc::clone(&self.painted_anchors),
            lights: gpu_lights,
            shade_params_base: self.shade_params_base,
            env_update,
            viewports: vp_list,
            skin,
            bone_matrix_lbs,
            bone_matrix_dqs,
            pending_pick,
            cloud_sun_atten: self.cloud_sun_atten,
            lod_enabled: self.lod_enabled,
            surfacenet_enabled: self.surfacenet_enabled,
            shadow_steps: self.environment.shadow_steps,
            shadow_csm_near: self.environment.shadow_csm_near,
            shadow_csm_max_distance: self.environment.shadow_csm_max_distance,
            shadow_csm_lambda: self.environment.shadow_csm_lambda,
            shadow_csm_depth_bias: self.environment.shadow_csm_depth_bias,
            shadow_csm_threshold_falloff: self.environment.shadow_csm_threshold_falloff,
            shadow_csm_sharp_distance: self.environment.shadow_csm_sharp_distance,
            shadow_csm_map_size: self.environment.shadow_csm_map_size,
            shadow_csm_pcf_taps: self.environment.shadow_csm_pcf_taps,
        };

        let t_encode = frame_start.elapsed();
        let snap_t_frame = snap_phase_start.elapsed();
        if snap_profile {
            // Per-phase ms — each value is the cumulative time from
            // t_cpu_setup to that phase's end, so deltas read left
            // to right.
            let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
            eprintln!(
                "[snap] geom={:.2} bone={:.2} vrs={:.2} pick={:.2} frame={:.2} | total={:.2}",
                to_ms(snap_t_geom),
                to_ms(snap_t_bone) - to_ms(snap_t_geom),
                to_ms(snap_t_vrs) - to_ms(snap_t_bone),
                to_ms(snap_t_pick) - to_ms(snap_t_vrs),
                to_ms(snap_t_frame) - to_ms(snap_t_pick),
                to_ms(snap_t_frame),
            );
        }
        // Record the sim-side submit timestamp for the [sculpt-pipeline]
        // latency decomposition (bump→submit vs submit→pickup). Cheap
        // (one atomic write); no-op when there's no fresh bump to
        // attribute the submit to.
        let post_bump = {
            let sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
            sm.record_geometry_submit_now();
            sm.last_geometry_bump_ns()
        };
        // Emit a phase breakdown when this tick had a bump to
        // attribute — either a fresh one triggered during
        // drain_render_results (sculpt) or one pending from before
        // this tick that we're still working to submit.
        if post_bump > pre_submit {
            let total_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            let other_ms = total_ms
                - phase_drain_ms
                - phase_update_scene_gpu_ms
                - phase_painted_walk_ms;
            eprintln!(
                "[sculpt-pipeline-sim] drain={:.2}ms update_scene_gpu={:.2}ms painted_walk={:.2}ms other={:.2}ms total={:.2}ms",
                phase_drain_ms,
                phase_update_scene_gpu_ms,
                phase_painted_walk_ms,
                other_ms,
                total_ms,
            );
        }
        self.render_worker.inbox.submit(frame);
        let t_frame_end = frame_start.elapsed();

        // 6. Push CPU-side timings into profiling history. GPU pass
        //    timings get stitched into the most-recent sample by
        //    `drain_render_results` once the render thread publishes
        //    them (typically 1-2 frames behind sim).
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        let cpu = crate::profiling::CpuPhaseTimings {
            setup_ms: ms(t_cpu_setup),
            snapshot_ms: ms(t_encode - t_cpu_setup),
            submit_ms: ms(t_frame_end - t_encode),
            total_ms: ms(t_frame_end),
        };
        self.profiling.push(crate::profiling::FrameSample {
            frame_idx: self.frame_index,
            cpu,
            // Both filled in by `drain_render_results` once the render
            // thread publishes the matching frame's `RenderResult`.
            // Lag is typically 1-2 frames, fine for display.
            gpu_passes: Vec::new(),
            render_dt_ms: 0.0,
            gpu_object_count: self.gpu_instances.len() as u32,
        });

        self.frame_index += 1;
    }

}

/// Walk an entity's octree (rooted at `root_offset` inside the
/// global packed `octree_data` buffer) and accumulate, per
/// shader-bearing material, the object-local AABB of leaves with
/// that material. Used by the per-leaf-material auto-scan to size
/// the geom-pipeline region tightly.
///
/// `octree_data` is the absolute-rebased packed buffer; branches
/// store offsets directly into this slice. Bricks are flattened
/// further — for each brick we walk its 64 cells in `brick_pool` and
/// look up cell leaf-attrs. Leaves at higher levels (shallow trees
/// without bricks) cover a 2^(depth-leaf_level) cube of voxel cells.
/// Resolve the effective `LeafAttr` for `slot` on a specific instance —
/// overlay if present (Phase 3), else the asset's shared pool. Mirrors
/// `fetch_leaf_attr_for` in WGSL.
#[inline]
fn resolve_leaf_attr(
    overlay: Option<&rkp_core::LeafAttrOverlay>,
    leaf_attrs: &[rkp_core::LeafAttr],
    slot: u32,
) -> rkp_core::LeafAttr {
    if let Some(o) = overlay {
        if let Some(e) = o.get(slot) {
            return e.attr();
        }
    }
    leaf_attrs[slot as usize]
}

fn scan_painted_aabbs(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &[rkp_core::LeafAttr],
    overlay: Option<&rkp_core::LeafAttrOverlay>,
    root_offset: usize,
    depth: u8,
    grid_origin: glam::Vec3,
    base_voxel_size: f32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
    out: &mut std::collections::HashMap<
        u16,
        std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
    >,
) {
    use rkp_core::sparse_octree::{
        is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
    };
    use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
    const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

    #[allow(clippy::too_many_arguments)]
    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &[rkp_core::LeafAttr],
        overlay: Option<&rkp_core::LeafAttrOverlay>,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: glam::Vec3,
        base_vs: f32,
        shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
        out: &mut std::collections::HashMap<
            u16,
            std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
        >,
    ) {
        use rkp_core::sparse_octree::{
            is_brick, is_leaf, leaf_slot, brick_id, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
        const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

        if offset >= octree_data.len() { return; }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE { return; }
        if is_brick(node) {
            let brick_id = brick_id(node);
            let base_idx = (brick_id * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() { continue; }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR { continue; }
                        let attr = resolve_leaf_attr(overlay, leaf_attrs, cell);
                        let primary = attr.material_primary;
                        let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
                        let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
                        let painted_mat = if shader_materials.contains_key(&primary) {
                            Some(primary)
                        } else if blend > 0 && shader_materials.contains_key(&secondary) {
                            Some(secondary)
                        } else {
                            None
                        };
                        if let Some(mat) = painted_mat {
                            let cell_voxel = glam::UVec3::new(
                                coord_voxels.x + cx,
                                coord_voxels.y + cy,
                                coord_voxels.z + cz,
                            );
                            let cell_local = grid_origin
                                + glam::Vec3::new(
                                    cell_voxel.x as f32,
                                    cell_voxel.y as f32,
                                    cell_voxel.z as f32,
                                ) * base_vs;
                            let cell_max = cell_local + glam::Vec3::splat(base_vs);
                            let info = shader_materials.get(&mat);
                            let tile_size = info.and_then(|i| i.tile_size);
                            super::lifecycle::expand_aabb(
                                out, mat, cell_local, cell_max,
                                attr.normal(), tile_size,
                            );
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            let slot = leaf_slot(node);
            let attr = resolve_leaf_attr(overlay, leaf_attrs, slot);
            let primary = attr.material_primary;
            let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
            let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
            let painted_mat = if shader_materials.contains_key(&primary) {
                Some(primary)
            } else if blend > 0 && shader_materials.contains_key(&secondary) {
                Some(secondary)
            } else {
                None
            };
            if let Some(mat) = painted_mat {
                let voxels_per_side = 1u32 << (max_depth - level);
                let leaf_size = voxels_per_side as f32 * base_vs;
                let leaf_min = grid_origin
                    + glam::Vec3::new(
                        coord_voxels.x as f32,
                        coord_voxels.y as f32,
                        coord_voxels.z as f32,
                    ) * base_vs;
                let leaf_max = leaf_min + glam::Vec3::splat(leaf_size);
                let info = shader_materials.get(&mat);
                let tile_size = info.and_then(|i| i.tile_size);
                super::lifecycle::expand_aabb(
                    out, mat, leaf_min, leaf_max,
                    attr.normal(), tile_size,
                );
            }
            return;
        }
        // Branch — descend into 8 children. `node` is the absolute
        // offset of the first child (rebased at allocation time).
        let _ = leaf_slot(0);
        let _ = INTERIOR_NODE;
        if level >= max_depth { return; }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            let child_offset = node as usize + octant as usize;
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                overlay,
                child_offset,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                shader_materials,
                out,
            );
        }
    }

    walk(
        octree_data,
        brick_pool,
        leaf_attrs,
        overlay,
        root_offset,
        0,
        depth,
        glam::UVec3::ZERO,
        grid_origin,
        base_voxel_size,
        shader_materials,
        out,
    );
    let _ = (is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE, BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR, BRICK_CELL_EMPTY);
}

/// Diagnostic — count painted leaves on each side of `mid_x` (in
/// object-local space). Mirrors the structure of `scan_painted_aabbs`
/// but only tallies; used once per scan to disambiguate "scan misses
/// half" from "paint really only on one half".
fn count_painted_halves(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &rkp_core::LeafAttrPool,
    root_offset: usize,
    depth: u8,
    grid_origin: glam::Vec3,
    base_voxel_size: f32,
    mid_x: f32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
    left: &mut u32,
    right: &mut u32,
) {
    use rkp_core::sparse_octree::{
        is_brick, is_leaf, leaf_slot, brick_id, EMPTY_NODE, INTERIOR_NODE,
    };
    use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
    const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

    fn check(material_packed: u32,
             shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>) -> bool {
        let primary = (material_packed & 0xFFFF) as u16;
        let sec_blend = ((material_packed >> 16) & 0xFFFF) as u16;
        let secondary = sec_blend & 0x0FFF;
        let blend = (sec_blend >> 12) & 0xF;
        if shader_materials.contains_key(&primary) {
            return true;
        }
        if blend > 0 && shader_materials.contains_key(&secondary) {
            return true;
        }
        false
    }

    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &rkp_core::LeafAttrPool,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: glam::Vec3,
        base_vs: f32,
        mid_x: f32,
        shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
        left: &mut u32,
        right: &mut u32,
    ) {
        use rkp_core::sparse_octree::{
            is_brick, is_leaf, brick_id, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
        const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;
        if offset >= octree_data.len() { return; }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE { return; }
        if is_brick(node) {
            let bid = brick_id(node);
            let base_idx = (bid * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() { continue; }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR { continue; }
                        let attr = leaf_attrs.get(cell);
                        let packed = (attr.material_primary as u32)
                            | ((attr.material_secondary_blend as u32) << 16);
                        if super::lifecycle::count_painted_halves_check(packed, shader_materials) {
                            let cell_x = grid_origin.x + (coord_voxels.x + cx) as f32 * base_vs;
                            if cell_x < mid_x { *left += 1; } else { *right += 1; }
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            return;
        }
        if level >= max_depth { return; }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                node as usize + octant as usize,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                mid_x,
                shader_materials,
                left,
                right,
            );
        }
    }

    walk(
        octree_data, brick_pool, leaf_attrs, root_offset, 0, depth,
        glam::UVec3::ZERO, grid_origin, base_voxel_size, mid_x,
        shader_materials, left, right,
    );
    let _ = (is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE, BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR, BRICK_CELL_EMPTY);
}

pub(super) fn count_painted_halves_check(
    material_packed: u32,
    shader_materials: &std::collections::HashMap<u16, rkp_render::shader_composer::UserShaderInfo>,
) -> bool {
    let primary = (material_packed & 0xFFFF) as u16;
    let sec_blend = ((material_packed >> 16) & 0xFFFF) as u16;
    let secondary = sec_blend & 0x0FFF;
    let blend = (sec_blend >> 12) & 0xF;
    if shader_materials.contains_key(&primary) {
        return true;
    }
    if blend > 0 && shader_materials.contains_key(&secondary) {
        return true;
    }
    false
}

/// Sentinel tile coord for non-tiled shaders (matching
/// `rkp_render::user_shader_pass::NO_TILE`). Single inner-map entry
/// for the whole painted area — V9 single-region behaviour.
const NO_TILE_COORD: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Register a painted leaf occupying `[mn, mx]` (host-local) for
/// material `mat`. For shaders with a `@tile_size`, the leaf is
/// bucketed into every tile coord it overlaps; for non-tiled
/// shaders, the leaf is registered under `NO_TILE_COORD` and its
/// AABB merged into the running painted-leaf bounds.
///
/// Tiling assumes the grid is anchored at host-local origin
/// (tile_coord = floor(pos / tile_size)). For typical paint where a
/// leaf's size ≪ tile_size, each leaf lives in a single tile — the
/// boundary case (leaf straddling tiles) registers it in all
/// overlapping tiles, which slightly inflates per-tile counts but
/// is correct for the band gate.
/// Transform an axis-aligned bounding box from one frame to another
/// via an arbitrary 4×4 matrix. Iterates the 8 corners and bounds the
/// result — produces a tight AABB for axis-aligned matrices and a
/// conservative AABB for rotated ones. When `matrix` is `None`, the
/// input is returned as-is (caller is treating local-space as world).
fn transform_aabb_to_world(
    local_min: glam::Vec3,
    local_max: glam::Vec3,
    matrix: Option<glam::Mat4>,
) -> (glam::Vec3, glam::Vec3) {
    let Some(m) = matrix else {
        return (local_min, local_max);
    };
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for i in 0..8u32 {
        let cx = if (i & 1) != 0 { local_max.x } else { local_min.x };
        let cy = if (i & 2) != 0 { local_max.y } else { local_min.y };
        let cz = if (i & 4) != 0 { local_max.z } else { local_min.z };
        let p = m.transform_point3(glam::Vec3::new(cx, cy, cz));
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

fn expand_aabb(
    out: &mut std::collections::HashMap<
        u16,
        std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
    >,
    mat: u16,
    mn: glam::Vec3,
    mx: glam::Vec3,
    normal: glam::Vec3,
    tile_size: Option<f32>,
) {
    let mat_map = out.entry(mat).or_default();

    fn merge(
        mat_map: &mut std::collections::HashMap<[i32; 3], super::state::PaintedTileEntry>,
        key: [i32; 3],
        mn: glam::Vec3,
        mx: glam::Vec3,
        normal: glam::Vec3,
    ) {
        let entry = mat_map.entry(key).or_insert_with(super::state::PaintedTileEntry::empty);
        entry.aabb.min = entry.aabb.min.min(mn);
        entry.aabb.max = entry.aabb.max.max(mx);
        entry.leaf_count = entry.leaf_count.saturating_add(1);
        // Sum (not mean) — for leaves that span tile boundaries, the
        // outer match below calls merge() once per overlapped tile, so
        // each tile gets that leaf's normal added once. Normalize at
        // anchor build time.
        entry.normal_sum += normal;
    }

    match tile_size {
        None => merge(mat_map, NO_TILE_COORD, mn, mx, normal),
        Some(s) if s > 0.0 => {
            // Compute tile coord range the leaf overlaps. Use a tiny
            // epsilon on the upper bound so a leaf whose max sits
            // exactly on a tile boundary doesn't count for the next
            // tile too.
            let inv = 1.0 / s;
            let lo = (mn * inv).floor();
            let hi = ((mx - glam::Vec3::splat(1e-6)) * inv).floor();
            for ix in (lo.x as i32)..=(hi.x as i32) {
                for iy in (lo.y as i32)..=(hi.y as i32) {
                    for iz in (lo.z as i32)..=(hi.z as i32) {
                        merge(mat_map, [ix, iy, iz], mn, mx, normal);
                    }
                }
            }
        }
        // tile_size 0 or negative → treat as non-tiled.
        Some(_) => merge(mat_map, NO_TILE_COORD, mn, mx, normal),
    }
}

