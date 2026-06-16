//! Frame lifecycle — render-frame submission, result drain, and the
//! sim-thread tick loop.
//!
//! `tick_loop` owns the outer pacing + command-drain cycle that
//! `ArvxEngine::spawn` launches on the engine thread.
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
        let (_pre_bump, pre_submit) = {
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
        let composed = arvx_render::shader_composer::compose(&self.user_shader_registry);
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
            arvx_render::shader_composer::UserShaderInfo,
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
        let gpu_objects_dirty_this_frame = self.gpu_objects_dirty.is_dirty();
        let phase_update_scene_gpu_t0 = std::time::Instant::now();
        // C2 transform-only fast path. When every dirty entity is
        // marked `Transform` (gizmo drag, drag-preview snap, future
        // physics-driven rigid-body moves), patch matching matrices
        // in place — saves ~60-75 ms vs. the full re-walk on
        // elephant-scale scenes. Anything Structural or `all` falls
        // through to the full rebuild below. PERF_DEBT.md C2.
        if self.gpu_objects_dirty.is_transform_only() {
            self.update_scene_gpu_transform_only();
            self.gpu_objects_dirty.clear();
        } else if self.gpu_objects_dirty.is_dirty() {
            // `ARVX_GPU_OBJECTS_PROFILE=1` logs every GPU-objects rebuild
            // independently of paint profiling, attributing the cause —
            // `all` (a `mark_all()`, e.g. terrain tile integrate/evict
            // forcing a full rebuild; see terrain_ops L1 perf debt),
            // `narrow-bail` (a structural-narrow attempt that hit a
            // not-yet-mapped entity), or `narrow` (the fast path). Pair
            // with `ARVX_TERRAIN_PROFILE=1` to correlate full rebuilds
            // with the per-tick integrate/evict counts that triggered
            // them. Baseline for the terrain-tick L1 fix.
            let profile = self.paint_profile_active()
                || std::env::var("ARVX_GPU_OBJECTS_PROFILE").is_ok();
            let was_all = self.gpu_objects_dirty.is_all();
            let t0 = std::time::Instant::now();
            // PERF_DEBT.md C2-extension: structural-narrow fast path.
            // When the dirty set is non-empty and `is_all()` is false,
            // try the per-entity splice+suffix-shift path before
            // falling back to the world-walking full rebuild. Returns
            // false if it can't handle the dirty set (e.g. an entity
            // spawned since the last full rebuild has no
            // entity_to_gpu mapping); the full path catches that.
            let used_narrow = if !self.gpu_objects_dirty.is_all() {
                self.update_scene_gpu_structural_narrow()
            } else {
                false
            };
            if !used_narrow {
                self.update_scene_gpu();
            }
            if profile {
                use std::sync::atomic::{AtomicU64, Ordering};
                static LAST_NS: AtomicU64 = AtomicU64::new(0);
                let now_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                let prev = LAST_NS.swap(now_ns, Ordering::Relaxed);
                let gap_ms = if prev == 0 { 0.0 } else { (now_ns.saturating_sub(prev)) as f64 / 1.0e6 };
                let cause = if used_narrow {
                    "narrow"
                } else if was_all {
                    "all" // mark_all() — full rebuild forced (terrain tick, scene load, ...)
                } else {
                    "narrow-bail" // narrow tried but an entity had no gpu mapping
                };
                eprintln!(
                    "[gpu-objects] update_scene_gpu dt={:?} path={} cause={} instances={} assets={} mesh_draws={} proxy_draws={} overlays={} gap_since_last={:.1}ms",
                    t0.elapsed(),
                    if used_narrow { "narrow" } else { "full" },
                    cause,
                    self.gpu_instances.len(),
                    self.gpu_assets.len(),
                    self.mesh_draws.len(),
                    self.proxy_draws.len(),
                    self.gpu_instance_overlays.len(),
                    gap_ms,
                );
            }
            self.gpu_objects_dirty.clear();
        }
        let phase_update_scene_gpu_ms =
            phase_update_scene_gpu_t0.elapsed().as_secs_f64() * 1000.0;

        let phase_painted_walk_t0 = std::time::Instant::now();
        let painted_walk_profile = self.paint_profile_active();
        // Phase E1 of `docs/PERF_DEBT.md`: drain any completed worker
        // result into `painted_per_entity` first. The result may carry
        // entries with empty `mat_tiles` — those entities had no shader-
        // bearing materials this scan; remove them from the cache so the
        // flat rebuild below doesn't iterate phantoms.
        let mut painted_state_changed = false;
        let mut painted_worker_walk_dt = std::time::Duration::ZERO;
        let mut painted_worker_jobs = 0usize;
        if let Some(result) = self.paint_walk_worker.try_recv() {
            painted_worker_jobs = result.entries.len();
            for (entity, cache) in result.entries {
                // Filter out entries for entities that were despawned
                // or scene-cleared while the worker was running. Without
                // this check, a stale result could re-populate
                // `painted_per_entity` with phantom entries that nothing
                // else removes — `delete_entity` and `clear_scene` both
                // wipe the cache, but the in-flight batch's output
                // arrives after them.
                if !self.world.contains(entity) {
                    self.painted_per_entity.remove(&entity);
                    continue;
                }
                if cache.mat_tiles.is_empty() {
                    self.painted_per_entity.remove(&entity);
                } else {
                    self.painted_per_entity.insert(entity, cache);
                }
            }
            self.painted_materials_paint_epoch = result.completed_paint_epoch;
            self.painted_materials_geometry_epoch = result.completed_geom_epoch;
            painted_worker_walk_dt = result.worker_walk_duration;
            painted_state_changed = true;
        }

        let mut geom_changed = false;
        let mut cur_paint_epoch_telemetry = 0u64;
        let mut cur_geom_epoch_telemetry = 0u64;
        if !shader_materials.is_empty() {
            // Reconcile against current paint + geometry epochs.
            //
            // - paint_epoch advances only via `apply_paint_stamp`, which
            //   also adds the painted entity to `painted_dirty_entities`.
            //   So a paint-only frame walks just that one entity.
            // - geometry_epoch advances on any voxel-pool / octree write
            //   (asset load, voxelize, bake, sculpt). When no specific
            //   dirty entities are recorded, blanket-invalidate so every
            //   renderable gets re-scanned by the next worker batch.
            let (cur_paint, cur_geom) = {
                let sm = self.scene_mgr.lock().expect("scene_mgr poisoned");
                (sm.paint_epoch(), sm.geometry_epoch())
            };
            cur_paint_epoch_telemetry = cur_paint;
            cur_geom_epoch_telemetry = cur_geom;
            geom_changed = cur_geom != self.painted_materials_geometry_epoch;
            // Blanket-invalidate ONLY when (a) nobody told us which
            // entities changed and (b) we haven't already submitted a
            // batch for this geom epoch. The submitted-epoch check
            // suppresses redundant invalidations during the 1-2 ticks
            // between submit and result-merge — without it, every tick
            // in that window would re-stage all renderables and queue
            // duplicate work. Specific-entity callers (sculpt/paint/
            // proc-bake) populate `painted_dirty_entities` themselves
            // — trust them so other entities' caches stay valid.
            //
            // Note: we do NOT clear `painted_per_entity` here. The OLD
            // entries stay in place so the flat rebuild during the
            // worker's in-flight window keeps producing anchors at the
            // last-known positions (1-2 frame lag, invisible). The
            // worker's result will replace each entity's cache as it
            // returns; empty caches signal removal.
            if geom_changed
                && self.painted_dirty_entities.is_empty()
                && self.painted_walk_submitted_geom_epoch != cur_geom
            {
                use crate::components::Renderable;
                for (entity, _) in self.world.query::<&Renderable>().iter() {
                    self.painted_dirty_entities.insert(entity);
                }
            }

            // Submit a fresh batch when the worker is idle and we have
            // dirty entities. While the worker is busy we leave
            // `painted_dirty_entities` untouched so this tick's
            // additions get picked up by the next batch — derived state
            // lags geometry by at most one walk + one tick, which is
            // invisible for grass-anchor positions.
            if self.paint_walk_worker.is_idle()
                && !self.painted_dirty_entities.is_empty()
            {
                use crate::components::Renderable;
                let snapshot = self
                    .scene_mgr
                    .lock()
                    .expect("scene_mgr poisoned")
                    .walk_snapshot();
                // Conservative walk-clip expansion: the largest
                // `@tile_size` across all shader materials. A leaf at
                // position P that contributes to a tile T in M's grid
                // can sit anywhere inside T, so the worker's clipped
                // descent must reach `dirty + tile_size_M` to recover
                // every leaf of every overlapping tile.
                let max_tile_size: Option<f32> = shader_materials
                    .values()
                    .filter_map(|i| i.tile_size)
                    .filter(|s| *s > 0.0)
                    .fold(None::<f32>, |acc, s| {
                        Some(acc.map_or(s, |a| a.max(s)))
                    });
                let any_unsized = shader_materials
                    .values()
                    .any(|i| i.tile_size.is_none());

                let dirty: std::collections::HashSet<hecs::Entity> =
                    std::mem::take(&mut self.painted_dirty_entities);
                // Drain the region log so each entity's job carries the
                // brush footprint that caused its dirty entry. An
                // entity in `dirty` without a regions entry (asset-load
                // geom-epoch fallback) falls through to the worker's
                // full-walk path.
                let mut regions: std::collections::HashMap<
                    hecs::Entity,
                    Vec<arvx_core::Aabb>,
                > = std::mem::take(&mut self.painted_dirty_regions);

                let mut jobs: Vec<super::paint_walk::PaintWalkJob> =
                    Vec::with_capacity(dirty.len());
                for entity in dirty {
                    // Despawned-while-dirty: drop any stale cache so
                    // the flat rebuild doesn't carry phantoms.
                    if !self.world.contains(entity) {
                        self.painted_per_entity.remove(&entity);
                        regions.remove(&entity);
                        continue;
                    }
                    let (root_offset, depth, grid_origin, base_voxel_size) = {
                        let Ok(r) = self.world.get::<&Renderable>(entity) else {
                            self.painted_per_entity.remove(&entity);
                            regions.remove(&entity);
                            continue;
                        };
                        let Some(spatial) =
                            r.spatial.as_ref().and_then(|g| g.as_octree())
                        else {
                            self.painted_per_entity.remove(&entity);
                            regions.remove(&entity);
                            continue;
                        };
                        (
                            spatial.root_offset as usize,
                            spatial.depth,
                            spatial.grid_origin,
                            spatial.base_voxel_size,
                        )
                    };
                    if self.entity_to_gpu.get(&entity).is_none() {
                        // No GPU mapping yet — keep the entity dirty
                        // for a later tick (re-add). Preserve regions
                        // so the next walk stays region-bounded.
                        self.painted_dirty_entities.insert(entity);
                        if let Some(v) = regions.remove(&entity) {
                            self.painted_dirty_regions.insert(entity, v);
                        }
                        continue;
                    }
                    let entity_inverse = self
                        .world
                        .get::<&crate::components::Transform>(entity)
                        .ok()
                        .map(|t| {
                            glam::Affine3A::from_scale_rotation_translation(
                                t.scale,
                                glam::Quat::from_euler(
                                    glam::EulerRot::XYZ,
                                    t.rotation.x.to_radians(),
                                    t.rotation.y.to_radians(),
                                    t.rotation.z.to_radians(),
                                ),
                                t.position,
                            )
                        })
                        .map(|a| a.inverse())
                        .unwrap_or(glam::Affine3A::IDENTITY);
                    let overlay = self.paint_overlays.get(&entity).cloned();
                    let regions_for_entity = regions.remove(&entity);
                    // CLONE the existing entry (don't move out) so the
                    // flat rebuild during the worker's in-flight window
                    // keeps producing anchors at the last-known
                    // positions. The worker mutates its copy and ships
                    // the new cache back; sim replaces the entry on
                    // result-merge. The clone cost is modest — a few µs
                    // for a typical sculpt entity, microseconds even on
                    // heavily-painted assets — and pays for the 1-2
                    // frame visual continuity during the lag.
                    //
                    // The full-walk path inside `walk_one` ignores
                    // `existing_mat_tiles` (it builds a fresh entry), so
                    // for non-region-bounded jobs the clone is wasted
                    // bytes. Acceptable; the region-bounded path is the
                    // hot one.
                    let existing_mat_tiles = self
                        .painted_per_entity
                        .get(&entity)
                        .map(|e| e.mat_tiles.clone())
                        .unwrap_or_default();
                    jobs.push(super::paint_walk::PaintWalkJob {
                        entity,
                        root_offset,
                        depth,
                        grid_origin,
                        base_voxel_size,
                        overlay,
                        entity_inverse,
                        regions: regions_for_entity,
                        existing_mat_tiles,
                    });
                }
                // Any regions left in `regions` belong to entries that
                // the loop bypassed (despawned / missing Renderable /
                // no spatial). They've already been removed from
                // `painted_per_entity`; drop the regions implicitly.
                let _ = regions;

                if !jobs.is_empty() {
                    let batch = super::paint_walk::PaintWalkBatch {
                        snapshot,
                        jobs,
                        shader_materials: std::sync::Arc::new(
                            shader_materials.clone(),
                        ),
                        max_tile_size,
                        any_unsized,
                        submitted_paint_epoch: cur_paint,
                        submitted_geom_epoch: cur_geom,
                        submitted_at: std::time::Instant::now(),
                    };
                    self.paint_walk_worker.submit(batch);
                    // Record the geom epoch this batch will cover so a
                    // tick between submit and result-merge doesn't
                    // re-fire the blanket-invalidate branch above.
                    self.painted_walk_submitted_geom_epoch = cur_geom;
                }
            }
        }

        // Rebuild flat views whenever per-entity contents changed (a
        // worker result merged) OR a geom-epoch jump invalidated the
        // cache. `painted_materials` is keyed by gpu_idx, so any
        // entity_to_gpu shift would otherwise leave stale entries — in
        // practice the add/remove paths already mark affected entities
        // dirty (or bump geom_epoch), so this trigger covers them.
        let need_flat_rebuild =
            !shader_materials.is_empty() && (painted_state_changed || geom_changed);
        if need_flat_rebuild {
            self.painted_materials.clear();
            let mut new_painted_anchors: std::collections::HashMap<
                u16,
                Vec<arvx_render::user_shader_mesh_pass::AnchorRecord>,
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
                //     tile_size` (object-local), transformed to world.
                //     Stable across frames as paint extends inside the
                //     tile, so blade positions don't shimmer.
                //   · **Painted-leaf AABB** (`te.aabb`) — only used to
                //     pick `surface_y` (the y the blade base sits on).
                //     Stable for flat-ground paint; deferred concern
                //     for slopes.
                //
                // When `tile_size` is `None` (shader didn't declare
                // `@tile_size`), the tile_coord is `NO_TILE_COORD` and
                // tile cube bounds fall back to the painted-leaf AABB
                // — degraded but deterministic.
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
                        let (paint_world_min, paint_world_max) = transform_aabb_to_world(
                            te.aabb.min,
                            te.aabb.max,
                            entity_world,
                        );
                        // Surface anchor (Y + normal) policy:
                        //
                        //   · With a per-tile sample (best_xz_dist_sq
                        //     finite ⇒ tile_size was known + a leaf
                        //     landed near the tile-center XZ): take
                        //     that leaf's top-Y and its prefiltered
                        //     LeafAttr.normal. Same world-locked tile
                        //     across LODs picks the same world location
                        //     for the sample, so the resulting
                        //     surface_y / surface_normal stay stable
                        //     when a tile entity swaps voxel sizes.
                        //   · Without a per-tile sample (NO_TILE_COORD
                        //     fall-back, or no leaf near tile center):
                        //     use the legacy aggregate — paint_max.y +
                        //     `normal_sum` averaging. Blades shimmer on
                        //     LOD swap in this branch but it's the
                        //     historical behavior; only the @tile_size
                        //     path gets the world-locked guarantee.
                        let (surface_y, local_normal) = if te.best_xz_dist_sq.is_finite() {
                            // Per-tile sample (object-local). XYZ
                            // because non-identity entity transforms
                            // need the leaf's XZ to land at the right
                            // world-Y after rotation.
                            let world_pos = entity_world
                                .map(|m| m.transform_point3(te.surface_sample_pos))
                                .unwrap_or(te.surface_sample_pos);
                            let n = te
                                .surface_sample_normal
                                .try_normalize()
                                .unwrap_or(glam::Vec3::Y);
                            (world_pos.y, n)
                        } else {
                            // NO_TILE_COORD fall-back (paint_max.y +
                            // averaged normal). Pre-tile-sample
                            // behavior, drift-prone on LOD swap.
                            let n = te
                                .normal_sum
                                .try_normalize()
                                .unwrap_or(glam::Vec3::Y);
                            (paint_world_max.y, n)
                        };
                        let world_normal = entity_world
                            .map(|m| m.transform_vector3(local_normal))
                            .unwrap_or(local_normal)
                            .try_normalize()
                            .unwrap_or(glam::Vec3::Y);
                        let seed = arvx_render::user_shader_mesh_pass::anchor_seed([
                            tile_coord[0] as f32,
                            tile_coord[1] as f32,
                            tile_coord[2] as f32,
                        ]) ^ (mat as u32).wrapping_mul(0x9E37_79B9);
                        bucket.push(
                            arvx_render::user_shader_mesh_pass::AnchorRecord {
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
                                paint_mask: te.paint_mask,
                                _pad: [0; 3],
                            },
                        );
                    }
                }
            }
            // ARVX_GRASS_DEBUG: detect per-tile seed instability. Loud
            // only when seeds for known tiles actually change.
            if std::env::var("ARVX_GRASS_DEBUG").is_ok() {
                let mut changed = 0u32;
                let mut new_tiles = 0u32;
                let mut dropped = 0u32;
                let mut cur_map: std::collections::HashMap<
                    (u32, u32, u32, u32, u16),
                    u32,
                > = std::collections::HashMap::new();
                for (&mat, bucket) in &new_painted_anchors {
                    for a in bucket {
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
                    "[grass-debug] rebuild paint={} geom={} mats={} anchors={} changed={} new={} dropped={}",
                    cur_paint_epoch_telemetry, cur_geom_epoch_telemetry,
                    new_painted_anchors.len(),
                    cur_map.len(),
                    changed, new_tiles, dropped,
                );
                self.debug_last_anchor_seeds = Some(cur_map);
            }

            self.painted_anchors = std::sync::Arc::new(new_painted_anchors);
        }

        if painted_walk_profile && (painted_state_changed || geom_changed) {
            eprintln!(
                "[paint] painted_materials_walk worker_dt={:?} merged_jobs={} cached_entities={} shader_materials={} geom_changed={}",
                painted_worker_walk_dt,
                painted_worker_jobs,
                self.painted_per_entity.len(),
                shader_materials.len(),
                geom_changed,
            );
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
            gpu_lights.push(arvx_render::arvx_shade::GpuLight {
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
            gpu_lights.push(arvx_render::arvx_shade::GpuLight {
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
        // `ARVX_SNAP_PROFILE=1`. Drops the per-phase ms split into
        // stderr each frame so we can attribute snapshot wallclock
        // when the cumulative `snapshot_ms` blows up.
        let snap_profile = std::env::var("ARVX_SNAP_PROFILE").is_ok();
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
        // Drain geometry_dirty into collider_caches_dirty: every
        // entity whose geometry changed needs its collider cache
        // rebuilt. PERF_DEBT B2 — the per-entity scope flows
        // through; consumer below picks per-entity vs world-walk
        // based on `is_all()`.
        if self.geometry_dirty.is_dirty() {
            if self.geometry_dirty.is_all() {
                self.collider_caches_dirty.mark_all();
            } else {
                for &entity in self.geometry_dirty.dirty_entities() {
                    self.collider_caches_dirty.mark_entity(entity);
                }
            }
            self.geometry_dirty.clear();
        }
        // Phase E2 of `docs/PERF_DEBT.md`: drain any completed
        // collider-worker result first, then submit a fresh batch when
        // the worker is idle. Sim never blocks on collider rebuild —
        // ColliderCache lands on the entity 1-2 ticks after the
        // geometry change. Play mode (the only current reader of
        // ColliderCache) re-reads on entry, so the lag is invisible.
        if let Some(result) = self.collider_worker.try_recv() {
            for (entity, cache) in result.entries {
                if !self.world.contains(entity) {
                    continue;
                }
                if self.world.get::<&crate::components::ColliderCache>(entity).is_ok() {
                    let _ = self
                        .world
                        .remove_one::<crate::components::ColliderCache>(entity);
                }
                let _ = self.world.insert_one(entity, cache);
            }
            let _ = result.worker_duration;
        }
        if self.collider_worker.is_idle() && self.collider_caches_dirty.is_dirty() {
            use crate::components::{EditorMetadata, Renderable, Transform};
            // Resolve which entities need a rebuild. `is_all()` → world
            // walk every RigidBody (project load, asset import); else
            // iterate the narrow per-entity set populated by
            // `geometry_dirty` callers (PERF_DEBT B2+C3).
            let candidates: Vec<hecs::Entity> = if self.collider_caches_dirty.is_all() {
                self.world
                    .query::<&crate::components::RigidBody>()
                    .iter()
                    .map(|(e, _)| e)
                    .collect()
            } else {
                self.collider_caches_dirty
                    .dirty_entities()
                    .iter()
                    .copied()
                    .collect()
            };

            // Per-entity input capture happens on sim (the worker
            // doesn't see the ECS). Mirrors `rebuild_collider_cache_for`
            // pre-E2.
            let mut jobs: Vec<super::collider_worker::ColliderJob> =
                Vec::with_capacity(candidates.len());
            for entity in candidates {
                if !self.world.contains(entity) {
                    continue;
                }
                let rb = match self
                    .world
                    .get::<&crate::components::RigidBody>(entity)
                {
                    Ok(rb) => (*rb).clone(),
                    Err(_) => continue,
                };
                let spatial = self
                    .world
                    .get::<&Renderable>(entity)
                    .ok()
                    .and_then(|r| r.spatial.clone())
                    .and_then(|g| g.into_octree())
                    .map(|sp| super::collider_worker::JobSpatial {
                        root_offset: sp.root_offset,
                        depth: sp.depth,
                        len: sp.len,
                        base_voxel_size: sp.base_voxel_size,
                        grid_origin: sp.grid_origin,
                        aabb_min: sp.aabb.min,
                        aabb_max: sp.aabb.max,
                    });
                let scale = self
                    .world
                    .get::<&Transform>(entity)
                    .map(|t| t.scale)
                    .unwrap_or(glam::Vec3::ONE);
                let pos = self
                    .world
                    .get::<&Transform>(entity)
                    .map(|t| t.position)
                    .unwrap_or_default();
                let name = self
                    .world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_default();
                jobs.push(super::collider_worker::ColliderJob {
                    entity,
                    rb,
                    spatial,
                    scale,
                    name,
                    pos,
                });
            }

            if !jobs.is_empty() {
                let snapshot = self
                    .scene_mgr
                    .lock()
                    .expect("scene_mgr poisoned")
                    .walk_snapshot();
                let batch = super::collider_worker::ColliderBatch {
                    snapshot,
                    jobs,
                };
                self.collider_worker.submit(batch);
            }
            self.collider_caches_dirty.clear();
        }

        let snap_t_geom = snap_phase_start.elapsed();

        // 2. Bone matrix bytes for shading (LBS + DQ paths). Cheap
        //    `Arc::clone` — the allocator now holds the bytes inside
        //    `Arc<Vec<u8>>` so the per-tick handoff costs a refcount
        //    bump instead of the ~58 MB memcpy this used to do via
        //    `.to_vec()`. See PERF_DEBT.md A3.
        let bone_matrix_lbs = self.bone_matrix_allocator.bytes_arc();
        let bone_matrix_dqs = self.bone_matrix_allocator.bytes_dq_arc();
        // PERF_DEBT.md D1: drain the dirty ranges that the last
        // `rebuild()` produced. Empty when this tick took the
        // C2-narrow path (no rebuild ran) or when every animated
        // entity's pose was byte-identical to last frame; the render
        // side reads `is_empty()` and skips the upload. The take
        // resets the allocator's local state so a subsequent
        // snapshot without an intervening rebuild reports empty too.
        let bone_matrix_lbs_dirty = self.bone_matrix_allocator.take_mat_dirty();
        let bone_matrix_dqs_dirty = self.bone_matrix_allocator.take_dq_dirty();
        // PERF_DEBT.md D2/D3: convert the boolean dirty flags into
        // DirtyRanges with the current buffer size. Set on stamps,
        // entity removes, scene clears — empty on idle ticks. Empty
        // → render side skips the overlay/sculpt upload entirely.
        let gpu_instance_overlays_dirty = if self.gpu_instance_overlays_dirty {
            self.gpu_instance_overlays_dirty = false;
            let mut d = arvx_core::DirtyRanges::new();
            // Buffer length in BYTES — gpu_instance_overlays stores
            // OverlayEntry (16 B) values, so the byte length is
            // `len * size_of::<OverlayEntry>()`.
            let byte_len = self.gpu_instance_overlays.len()
                .saturating_mul(std::mem::size_of::<arvx_core::OverlayEntry>())
                as u32;
            d.mark_full(byte_len);
            d
        } else {
            arvx_core::DirtyRanges::new()
        };
        let gpu_instance_sculpts_dirty = if self.gpu_instance_sculpts_dirty {
            self.gpu_instance_sculpts_dirty = false;
            let mut d = arvx_core::DirtyRanges::new();
            // gpu_instance_sculpts stores u32 (4 B) values.
            let byte_len = self.gpu_instance_sculpts.len()
                .saturating_mul(std::mem::size_of::<u32>())
                as u32;
            d.mark_full(byte_len);
            d
        } else {
            arvx_core::DirtyRanges::new()
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
            let atmo_frame = arvx_render::arvx_atmosphere::AtmosphereFrameParams {
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
                arvx_render::arvx_god_rays::GodRayParams {
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

            let vp_mode = self
                .viewports
                .get(viewport_id)
                .map(|v| v.mode)
                .unwrap_or(arvx_render::RenderMode::InSitu);
            // Raymarch is BUILD-only — every other viewport uses the
            // mesh raster. Single-viewport check, no per-frame mode
            // field needed.
            let vp_raymarch = viewport_id == crate::viewport::ViewportId::BUILD;

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
            let isolation = matches!(vp_mode, arvx_render::RenderMode::Isolation);
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

            // Procedural raymarch state — only on the BUILD viewport
            // (the only one that runs the raymarch pass) and only when
            // a procedural entity is selected. Sim flattens the tree,
            // builds the AABB, and pre-filters ghost primitives;
            // render uploads + binds.
            let proc_raymarch =
                if vp_raymarch {
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
                                    let ins = arvx_procedural::flatten_tree(&pg.tree);
                                    let bounds = arvx_procedural::compute_bounds(&pg.tree);
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
                    let ghost_instructions: Vec<arvx_procedural::ProcInstruction> = instructions
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
            // The procedural-node gizmo lives on the BUILD viewport
            // (which always raymarches). MAIN gets the standard
            // entity gizmo; everything else gets nothing.
            let wireframe_verts = if viewport_id == ViewportId::MAIN {
                gizmo_verts_main.clone()
            } else if viewport_id == ViewportId::BUILD {
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
                Some(arvx_render::arvx_grid::GridParams {
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
            let kind = if pp.viewport == ViewportId::BUILD {
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
            mesh_draws: std::sync::Arc::clone(&self.mesh_draws),
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
            bone_matrix_lbs,
            bone_matrix_dqs,
            bone_matrix_lbs_dirty,
            bone_matrix_dqs_dirty,
            gpu_instance_overlays_dirty,
            gpu_instance_sculpts_dirty,
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


/// Diagnostic — count painted leaves on each side of `mid_x` (in
/// object-local space). Mirrors the structure of `scan_painted_aabbs`
/// but only tallies; used once per scan to disambiguate "scan misses
/// half" from "paint really only on one half".
fn count_painted_halves(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &arvx_core::LeafAttrPool,
    root_offset: usize,
    depth: u8,
    grid_origin: glam::Vec3,
    base_voxel_size: f32,
    mid_x: f32,
    shader_materials: &std::collections::HashMap<u16, arvx_render::shader_composer::UserShaderInfo>,
    left: &mut u32,
    right: &mut u32,
) {
    use arvx_core::sparse_octree::{
        is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
    };
    use arvx_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
    const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

    fn check(material_packed: u32,
             shader_materials: &std::collections::HashMap<u16, arvx_render::shader_composer::UserShaderInfo>) -> bool {
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
        leaf_attrs: &arvx_core::LeafAttrPool,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: glam::Vec3,
        base_vs: f32,
        mid_x: f32,
        shader_materials: &std::collections::HashMap<u16, arvx_render::shader_composer::UserShaderInfo>,
        left: &mut u32,
        right: &mut u32,
    ) {
        use arvx_core::sparse_octree::{
            is_brick, is_leaf, brick_id, EMPTY_NODE, INTERIOR_NODE,
        };
        use arvx_core::brick_pool::{BRICK_DIM, BRICK_CELLS, BRICK_INTERIOR};
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
    shader_materials: &std::collections::HashMap<u16, arvx_render::shader_composer::UserShaderInfo>,
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
