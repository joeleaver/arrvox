//! Per-frame derive of `gpu_assets` + `gpu_instances` + skin dispatch
//! plans from the ECS.
//!
//! Flattens `world` → `gpu_instances` (per-entity) and dedupes by
//! `octree_root` into `gpu_assets` (per-unique-asset) on every tick where
//! either geometry or transforms are dirty. Updates the UUID ↔ gpu-index
//! maps used for pick resolution and walks skinned entities to build
//! the skin scatter dispatch list the render thread consumes.

use super::state::EngineState;

impl EngineState {
    pub(crate) fn update_scene_gpu(&mut self) {
        use crate::components::*;

        // Re-pack every skinned entity's current pose into the scene
        // bone buffer. Empty when no animated entities are loaded.
        self.bone_matrix_allocator.rebuild(&self.world);

        // Route through `Arc::make_mut`: in steady state refcount=1
        // (render dropped last frame's snapshot) so this is a free
        // `&mut Vec`. When render still holds the Arc, `make_mut`
        // reallocates a fresh empty Vec — same cost as today's
        // implicit `.clone()` would have been, but we get the
        // wide-cap snapshot Arc-clone-only benefit on clean ticks.
        // See PERF_DEBT.md A3.
        std::sync::Arc::make_mut(&mut self.gpu_assets).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instances).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instance_overlays).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instance_sculpts).clear();
        std::sync::Arc::make_mut(&mut self.mesh_draws).clear();
        std::sync::Arc::make_mut(&mut self.proxy_draws).clear();
        self.gpu_to_entity.clear();
        self.entity_to_gpu.clear();

        // Refresh `material_is_glass` from the material library —
        // O(slot_count), typically dozens. If the resulting Vec
        // differs from last frame, clear `asset_has_glass_cache` so
        // every asset re-scans its leaves on its next draw.
        //
        // PERF_DEBT.md C2-extension: the geom_epoch-driven cache
        // invalidation was removed. It was firing on every sculpt
        // stamp (which bumps geom_epoch) and triggering a 2.5M-leaf
        // rescan inside `update_scene_gpu` — ~50 ms per stamp on the
        // splat5 elephant. Sculpt-Carve can never *add* glass (only
        // remove), so a stale-true verdict is just a perf cost (an
        // empty glass pass), not a correctness issue. The cases that
        // legitimately mutate per-asset glass state — `remap_entity_
        // material` and sculpt-Raise with a glass brush — now
        // invalidate `asset_has_glass_cache[root_offset]` directly at
        // the call site. `material_glass_lib_epoch` is kept as a
        // sentinel for the material-library-change branch above so
        // its semantics remain "the cache was last refreshed at this
        // epoch."
        {
            let slot_count = self.material_lib.slot_count();
            let new_material_is_glass: Vec<bool> = (0..slot_count)
                .map(|id| {
                    self.material_lib
                        .get_def(id as u16)
                        .map(|d| d.opacity < 0.99)
                        .unwrap_or(false)
                })
                .collect();
            if new_material_is_glass != self.material_is_glass {
                self.material_is_glass = new_material_is_glass;
                self.asset_has_glass_cache.clear();
            }
            // Advance the epoch marker (no cache work) so any future
            // consumer that compares against it has a fresh value.
            self.material_glass_lib_epoch = self
                .geometry_epoch_handle
                .load(std::sync::atomic::Ordering::Acquire);
        }
        // Per-frame asset table — `octree_root` → index into
        // `gpu_assets`. Two entities sharing one .arvx asset share one
        // slot; the dedupe authoritatively builds the asset record from
        // the asset cache's skinning_data (when present), not from the
        // first-encountered instance's per-frame skinning binding.
        let mut asset_table: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();

        // Refresh the sim-side skinning_data cache only when a
        // geometry mutation has happened since we last built it.
        // Steady-state (no bakes / no asset (un)loads) does ZERO
        // scene_mgr lock acquisitions per tick — so even a busy
        // bake_worker holding the Mutex for hundreds of ms can't
        // stall sim. When the epoch does advance (a bake completed,
        // an asset was loaded), we lock once to refresh; that one
        // wait might be long if bake_worker has *already started*
        // the next bake, but we pay it once per epoch bump rather
        // than once per skinned entity per tick.
        if self.skinning_enabled {
            let current_epoch = self
                .geometry_epoch_handle
                .load(std::sync::atomic::Ordering::Acquire);
            if current_epoch > self.skinning_data_cache_epoch {
                let asset_handles: std::collections::HashSet<arvx_render::AssetHandle> = self
                    .world
                    .query::<&Renderable>()
                    .iter()
                    .filter_map(|(_, r)| r.asset_handle)
                    .collect();
                self.skinning_data_cache.clear();
                if !asset_handles.is_empty() {
                    let sm = self.scene_mgr.lock().unwrap();
                    for h in asset_handles {
                        if let Some(data) = sm.skinning_data(h).cloned() {
                            self.skinning_data_cache.insert(h, data);
                        }
                    }
                }
                self.skinning_data_cache_epoch = current_epoch;
            }
        }

        // Collect renderable entities and sort by `Entity::to_bits()`
        // — hecs assigns monotonically-increasing bits to every spawn
        // (generation << 32 | index), so this is stable per entity
        // while alive AND gives newest-at-bottom ordering naturally.
        // hecs query iteration order follows archetype layout, which
        // shifts when a new archetype appears — without this sort,
        // gpu vec positions of existing entities reshuffle on every
        // such event. Since render-side `interpolate_gpu_objects`
        // matches prev↔curr by `object_id == gpu_idx`, any shift
        // would blend each entity against some unrelated entity's
        // previous world matrix (visible smear, then pop-back).
        let mut ordered: Vec<hecs::Entity> = self.world
            .query::<(&Transform, &Renderable)>()
            .iter()
            .filter_map(|(entity, (_, r))| {
                if r.spatial.is_some() { Some(entity) } else { None }
            })
            .collect();
        ordered.sort_by_key(|e| e.to_bits());

        for entity in ordered {
            let Ok(transform) = self.world.get::<&Transform>(entity) else { continue };
            let Ok(renderable) = self.world.get::<&Renderable>(entity) else { continue };
            let transform = (*transform).clone();
            if let Some(spatial) = renderable.spatial.as_ref().and_then(|g| g.as_octree()) {
                let world_matrix = glam::Mat4::from_scale_rotation_translation(
                    transform.scale,
                    glam::Quat::from_euler(
                        glam::EulerRot::XYZ,
                        transform.rotation.x.to_radians(),
                        transform.rotation.y.to_radians(),
                        transform.rotation.z.to_radians(),
                    ),
                    transform.position,
                );
                let gpu_idx = self.gpu_instances.len() as u32;
                let spatial_handle = arvx_core::scene_node::SpatialHandle::Octree {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                // `skinning_enabled = false` forces every entity to
                // emit the rest pose (mesh VS sees `skinning_mode =
                // SKINNING_MODE_NONE`).
                let skinning = if self.skinning_enabled {
                    self.bone_matrix_allocator.binding(entity)
                } else {
                    None
                };
                // Asset side — dedupe by `octree_root`. Source the
                // skinning template (`bone_count`, `rest_octree_*`) from
                // the asset cache's skinning_data so it stays correct
                // even when this particular instance's skin plan bails.
                let bone_count_for_asset = renderable
                    .asset_handle
                    .and_then(|h| self.skinning_data_cache.get(&h))
                    .map(|sd| sd.rest_bone_aabbs.len() as u32)
                    .unwrap_or(0);
                let asset_id = match asset_table.get(&spatial.root_offset) {
                    Some(&id) => id,
                    None => {
                        let id = self.gpu_assets.len() as u32;
                        std::sync::Arc::make_mut(&mut self.gpu_assets).push(
                            crate::scene_sync::build_gpu_asset(
                                &spatial.aabb,
                                spatial.grid_origin,
                                &spatial_handle,
                                spatial.voxel_size,
                                bone_count_for_asset,
                            ),
                        );
                        asset_table.insert(spatial.root_offset, id);
                        id
                    }
                };
                let mut inst = crate::scene_sync::build_gpu_instance(
                    &world_matrix,
                    asset_id,
                    renderable.material_id,
                    gpu_idx,
                    skinning,
                );
                // Render-layer mask — entity opt-in via RenderLayer
                // component, otherwise the system DEFAULT bit.
                inst.layer_mask = self
                    .world
                    .get::<&crate::viewport::RenderLayer>(entity)
                    .map(|l| l.mask)
                    .unwrap_or(crate::viewport::layer::DEFAULT);
                // Per-instance paint overlay slice. Empty (overlay_count=0)
                // when this entity has never been painted; the WGSL fetch
                // helper falls through to `leaf_attr_pool[slot]` in that
                // case. We append to the global overlay vec in entity-walk
                // order so each instance points at its own contiguous slice.
                if let Some(overlay) = self.paint_overlays.get(&entity) {
                    if !overlay.is_empty() {
                        let off = self.gpu_instance_overlays.len() as u32;
                        let count = overlay.len() as u32;
                        std::sync::Arc::make_mut(&mut self.gpu_instance_overlays)
                            .extend_from_slice(overlay.entries());
                        inst.overlay_offset = off;
                        inst.overlay_count = count;
                    }
                }
                // Per-instance sculpt overlay slice (Phase A — Carve).
                // Empty (sculpt_count=0) when this entity has no carve
                // edits; the WGSL `is_leaf_removed` short-circuits in
                // that case. Appended in entity-walk order, same shape
                // as paint above.
                if let Some(sculpt) = self.sculpt_overlays.get(&entity) {
                    if !sculpt.is_empty() {
                        let off = self.gpu_instance_sculpts.len() as u32;
                        let count = sculpt.len() as u32;
                        std::sync::Arc::make_mut(&mut self.gpu_instance_sculpts)
                            .extend_from_slice(sculpt.entries());
                        inst.sculpt_offset = off;
                        inst.sculpt_count = count;
                    }
                }
                self.entity_to_gpu.insert(entity, self.gpu_instances.len());
                self.gpu_to_entity.push(entity);
                std::sync::Arc::make_mut(&mut self.gpu_instances).push(inst);

                // Mesh-raster per-instance draw record. Only
                // `Renderable` entities with an `asset_handle` (i.e.
                // loaded `.arvx` assets, not procedurals) make it into
                // the mesh path — procedurals ride `proxy_draws`
                // extracted today. Built every frame because world
                // transforms can change per-tick; the engine ships
                // this list to the render thread alongside the
                // existing `gpu_instances`.
                if let Some(handle) = renderable.asset_handle {
                    // Per-instance skinning state — when `skinning`
                    // is `Some(_)` the entity has a live `Skeleton`
                    // and `bone_matrix_allocator` packed its current
                    // pose into the per-frame bone palette. The mesh
                    // VS picks them up via `bone_offset_*`.
                    // `dqs_enabled` selects LBS vs DQS.
                    let (skinning_mode, bone_offset_lbs, bone_offset_dqs) = match skinning {
                        Some(b) => (
                            if self.dqs_enabled { 1 } else { 0 },
                            b.bone_buffer_offset,
                            b.bone_dq_offset,
                        ),
                        None => (
                            arvx_render::mesh_instance::SKINNING_MODE_NONE,
                            0,
                            0,
                        ),
                    };
                    // The asset's `grid_origin` (in object-local /
                    // mesh frame) is what the surface-mesh extractor
                    // baked into every vertex's `local_pos`. Bone
                    // matrices in the per-frame palette operate on
                    // **grid-frame** positions (origin at octree
                    // corner), so the mesh VS subtracts grid_origin
                    // before applying bones and adds it back after.
                    // `Skeleton.grid_offset = -grid_origin`. For
                    // unskinned entities (no Skeleton component) we
                    // fall back to `[0, 0, 0]`; the VS skips the
                    // bridge entirely when `skinning_mode == NONE`,
                    // so the value doesn't matter on that path.
                    let grid_origin: [f32; 3] = self
                        .world
                        .get::<&crate::components::Skeleton>(entity)
                        .ok()
                        .map(|s| (-s.grid_offset).to_array())
                        .unwrap_or([0.0, 0.0, 0.0]);
                    // Compute `has_glass` for this draw — the mesh
                    // primary path skips the front/back glass raster
                    // passes for `!has_glass` instances, recovering
                    // most of the perf cost on opaque-only assets.
                    //
                    // Two sources of glass on an instance:
                    //   1. Asset leaves carry a glass material in
                    //      their `material_primary` slot (bake-time
                    //      assignment, post-paint mutations of the
                    //      shared pool). Cached per asset by
                    //      `root_offset` since the scan walks every
                    //      leaf and is expensive on big meshes.
                    //   2. Per-entity `material_overrides` map an
                    //      asset's original material to a glass one
                    //      (deferred remap, e.g. for save/load
                    //      replay). Checked per draw — overrides are
                    //      a short Vec.
                    let root_offset = spatial.root_offset;
                    let asset_has_glass = match self
                        .asset_has_glass_cache
                        .get(&root_offset)
                    {
                        Some(&v) => v,
                        None => {
                            // Authority for an UNPAINTED asset is its ≤16-entry
                            // material palette (the `.arvx` header lists exactly
                            // the materials its leaves use) — an O(16) check, no
                            // leaf walk. This is the path scene-load streaming
                            // takes; it replaces a ~2.3M-leaf scan per big asset.
                            //
                            // Once the shared pool has been painted with a
                            // possibly-non-palette glass material the palette is
                            // no longer authoritative, so those roots fall back
                            // to the per-leaf walk (the same scan as before).
                            let found = if self.assets_painted_glass.contains(&root_offset) {
                                let sm = self.scene_mgr.lock().unwrap();
                                let mut leaf_slots: Vec<u32> = Vec::new();
                                let all_nodes = sm.octree.data();
                                let internal_attrs = sm.octree.internal_attrs_data();
                                crate::engine::model_scan::collect_leaf_slots(
                                    all_nodes,
                                    &sm.brick_pool,
                                    root_offset as usize,
                                    &mut leaf_slots,
                                );
                                crate::engine::model_scan::collect_internal_attr_slots(
                                    all_nodes,
                                    internal_attrs,
                                    root_offset as usize,
                                    &mut leaf_slots,
                                );
                                let pool_size = sm.leaf_attr_pool.allocated_count();
                                leaf_slots.iter().any(|&slot| {
                                    slot < pool_size && {
                                        let mat_id =
                                            sm.leaf_attr_pool.get(slot).material_primary as usize;
                                        mat_id < self.material_is_glass.len()
                                            && self.material_is_glass[mat_id]
                                    }
                                })
                            } else if let Some(h) = renderable.asset_handle {
                                self.scene_mgr
                                    .lock()
                                    .unwrap()
                                    .asset_palette_has_glass(h, &self.material_is_glass)
                            } else {
                                false
                            };
                            self.asset_has_glass_cache.insert(root_offset, found);
                            found
                        }
                    };
                    let overlay_glass =
                        renderable.material_overrides.iter().any(|(_, to)| {
                            let to = *to as usize;
                            to < self.material_is_glass.len()
                                && self.material_is_glass[to]
                        });
                    let has_glass = asset_has_glass || overlay_glass;

                    std::sync::Arc::make_mut(&mut self.mesh_draws).push(
                        arvx_render::mesh_instance::MeshDraw {
                            asset_handle_raw: handle.raw(),
                            world: world_matrix.to_cols_array_2d(),
                            object_id: gpu_idx,
                            grid_origin,
                            bone_offset_lbs,
                            bone_offset_dqs,
                            skinning_mode,
                            has_glass,
                        },
                    );
                }
            } else if let Some(proxy) = renderable.spatial.as_ref().and_then(|g| g.as_proxy_mesh()) {
                // Procedural rendered as a first-class triangle proxy
                // mesh. Own raster pipeline (`mesh_proxy_pass`); no
                // octree, no LeafAttr indirection, no LOD select, no
                // skinning, no shadow yet. The GpuInstance entry is
                // kept so pick-buffer reads map back to a hecs entity;
                // no synthesized GpuAsset record is needed since the
                // proxy raster doesn't consult the asset buffer.
                let world_matrix = glam::Mat4::from_scale_rotation_translation(
                    transform.scale,
                    glam::Quat::from_euler(
                        glam::EulerRot::XYZ,
                        transform.rotation.x.to_radians(),
                        transform.rotation.y.to_radians(),
                        transform.rotation.z.to_radians(),
                    ),
                    transform.position,
                );
                let gpu_idx = self.gpu_instances.len() as u32;

                // Asset id is a don't-care for proxies — point it at
                // slot 0 so the GpuInstance is well-formed even
                // though the proxy raster ignores it.
                let mut inst = crate::scene_sync::build_gpu_instance(
                    &world_matrix,
                    /* asset_id */ 0,
                    renderable.material_id,
                    gpu_idx,
                    None,
                );
                inst.layer_mask = self
                    .world
                    .get::<&crate::viewport::RenderLayer>(entity)
                    .map(|l| l.mask)
                    .unwrap_or(crate::viewport::layer::DEFAULT);
                self.entity_to_gpu.insert(entity, self.gpu_instances.len());
                self.gpu_to_entity.push(entity);
                std::sync::Arc::make_mut(&mut self.gpu_instances).push(inst);
                // Suppress unused-field warning on the proxy aabb —
                // kept on `ProxyMeshData` for pick/overlap CPU queries.
                let _proxy_aabb = proxy.aabb;

                std::sync::Arc::make_mut(&mut self.proxy_draws).push(
                    arvx_render::mesh_proxy_pass::ProxyDraw {
                        handle_raw: proxy.handle.raw(),
                        world: world_matrix.to_cols_array_2d(),
                        object_id: gpu_idx,
                    },
                );
            }
        }

    }

    /// Per-entity transform-only fast path. Patches just the
    /// `ArvxGpuInstance.world` matrix for each dirty entity in
    /// `self.gpu_objects_dirty.dirty_entities()`. Skips the full
    /// rebuild's per-tick work (bone_matrix repack, skin scatter
    /// planning, asset table dedup, overlay/sculpt flatten) — those
    /// are all bit-identical to the prior frame when only transforms
    /// changed.
    ///
    /// Caller (lifecycle's `submit_render_frame`) gates this on
    /// `gpu_objects_dirty.is_transform_only()` — i.e. all dirty
    /// entities carry `DirtyKind::Transform` and `is_all()` is
    /// false. Anything stricter (Structural, or `all`) routes
    /// through the full [`Self::update_scene_gpu`] rebuild.
    ///
    /// PERF_DEBT.md C2. Saves ~60-75 ms per gizmo-drag stamp on
    /// the splat5 elephant scene by avoiding the world-wide
    /// re-walk + asset re-dedup + flat-vec re-flatten.
    pub(crate) fn update_scene_gpu_transform_only(&mut self) {
        use crate::components::Transform;

        // `dirty_entities()` is a `&HashMap<Entity, DirtyKind>`;
        // collect (entity, gpu_idx, new_world) eagerly so we can drop
        // the world/state borrows before taking `Arc::make_mut` on
        // gpu_instances/mesh_draws/proxy_draws below.
        let dirty: Vec<hecs::Entity> = self
            .gpu_objects_dirty
            .dirty_entities()
            .keys()
            .copied()
            .collect();
        let mut updates: Vec<(u32, [[f32; 4]; 4])> = Vec::with_capacity(dirty.len());
        for entity in dirty {
            let Some(&gpu_idx) = self.entity_to_gpu.get(&entity) else {
                // No GPU row yet — entity was added since the last
                // full rebuild. The C2 fast path doesn't handle
                // additions; the next structural mark will trigger
                // a full rebuild and pick this up. Skip.
                continue;
            };
            let Ok(transform) = self.world.get::<&Transform>(entity) else {
                continue;
            };
            let world_matrix = glam::Mat4::from_scale_rotation_translation(
                transform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    transform.rotation.x.to_radians(),
                    transform.rotation.y.to_radians(),
                    transform.rotation.z.to_radians(),
                ),
                transform.position,
            );
            updates.push((gpu_idx as u32, world_matrix.to_cols_array_2d()));
        }

        // Patch the matching gpu_instances row.
        let gpu_instances = std::sync::Arc::make_mut(&mut self.gpu_instances);
        for &(gpu_idx, world) in &updates {
            if let Some(inst) = gpu_instances.get_mut(gpu_idx as usize) {
                inst.world = world;
            }
        }

        // Mesh / proxy draws carry their own per-instance world too —
        // keyed by `object_id` (= gpu_idx). Both lists are short
        // (one entry per asset-backed / proxy entity), so a linear
        // scan per update is cheap; no separate index map needed.
        // `Arc::make_mut` no-ops when the list is empty (refcount=1
        // on a fresh empty Vec).
        if !self.mesh_draws.is_empty() {
            let mesh_draws = std::sync::Arc::make_mut(&mut self.mesh_draws);
            for &(gpu_idx, world) in &updates {
                for d in mesh_draws.iter_mut() {
                    if d.object_id == gpu_idx {
                        d.world = world;
                    }
                }
            }
        }
        if !self.proxy_draws.is_empty() {
            let proxy_draws = std::sync::Arc::make_mut(&mut self.proxy_draws);
            for &(gpu_idx, world) in &updates {
                for d in proxy_draws.iter_mut() {
                    if d.object_id == gpu_idx {
                        d.world = world;
                    }
                }
            }
        }
    }

    /// Per-entity structural-narrow fast path. Handles dirty entities
    /// whose Renderable / overlays / sculpts / material / Transform
    /// changed but where the *set* of entities is unchanged (no
    /// add/remove since the last full rebuild) — i.e. the common
    /// sculpt/paint stamp case.
    ///
    /// Returns `false` if it can't handle the dirty set (e.g. an
    /// entity newer than the last full rebuild has no `entity_to_gpu`
    /// entry yet); the caller falls back to [`Self::update_scene_gpu`].
    ///
    /// What it does:
    ///   - Splices each dirty entity's `paint_overlays` slice into
    ///     `gpu_instance_overlays` in place; splices its
    ///     `sculpt_overlays` slice into `gpu_instance_sculpts`.
    ///   - Updates that entity's `gpu_instances` row (`world`,
    ///     `overlay_offset/count`, `sculpt_offset/count`, `material_id`,
    ///     `layer_mask`).
    ///   - Shifts `overlay_offset` and `sculpt_offset` on every
    ///     subsequent row by the delta produced by this entity's
    ///     splice.
    ///   - Patches `mesh_draws[i].world` / `proxy_draws[i].world`
    ///     for any Transform-dirty entity (mixed-kind ticks).
    ///
    /// What it skips (vs. the full rebuild):
    ///   - `bone_matrix_allocator.rebuild` — skeleton state stable
    ///     across sculpt/paint stamps.
    ///   - `gpu_assets` rebuild — depends only on spatial.aabb /
    ///     grid_origin / voxel_size / bone_count, none of which a
    ///     sculpt/paint stamp mutates on a loaded asset.
    ///   - The world-wide query + sort for `ordered`.
    ///   - `mesh_draws[i].has_glass` re-evaluation. Same staleness
    ///     model as the C2 transform-only path: the value is left at
    ///     its prior verdict, refreshed only when sculpt-Raise lands
    ///     a glass brush (sculpt_ops drops the per-asset cache entry)
    ///     or `remap_entity_material` runs. Carve cannot add glass,
    ///     so a stale-true verdict for an emptied asset is just a
    ///     wasted glass pass.
    ///
    /// On the splat5 elephant scene this drops `update_scene_gpu`
    /// from ~60-172 ms per stamp (full rebuild + skin replan + glass
    /// scan for 22 entities) to under 5 ms for the single dirty
    /// entity's splice + suffix shift. On scenes where animation is
    /// playing, `mark_all` from `animation::tick` still forces the
    /// full rebuild — but with the glass-scan cache invalidation
    /// moved to the call sites that legitimately flip has_glass, the
    /// full rebuild itself is now ~0.1 ms per stamp.
    pub(crate) fn update_scene_gpu_structural_narrow(&mut self) -> bool {
        use crate::components::{Renderable, Transform};

        // Collect (gpu_idx, entity) pairs; bail to full rebuild if
        // any dirty entity is newer than the last gpu mapping.
        let dirty_set = self.gpu_objects_dirty.dirty_entities();
        let mut work: Vec<(usize, hecs::Entity)> = Vec::with_capacity(dirty_set.len());
        for &entity in dirty_set.keys() {
            let Some(&gpu_idx) = self.entity_to_gpu.get(&entity) else {
                return false;
            };
            work.push((gpu_idx, entity));
        }
        // Sort by gpu_idx so each entity's splice + suffix shift
        // updates downstream offsets in order — the next entity's
        // already-shifted offset is what we read from gpu_instances.
        work.sort_by_key(|&(idx, _)| idx);

        for (gpu_idx, entity) in work {
            // Re-read per-entity inputs. Each scope drops its
            // hecs::Ref before we mutate gpu_instances below.
            let Ok(transform) = self.world.get::<&Transform>(entity) else { continue };
            let Ok(renderable) = self.world.get::<&Renderable>(entity) else { continue };
            let transform_clone = (*transform).clone();
            let new_material_id = renderable.material_id as u32;
            let asset_handle_raw = renderable.asset_handle.map(|h| h.raw());
            // Spatial variant decides whether the entity contributes a
            // mesh_draws row (octree assets) or a proxy_draws row
            // (procedural proxies). `has_glass` re-eval is deliberately
            // skipped here — see fn doc comment.
            let is_octree = matches!(
                renderable.spatial.as_ref(),
                Some(crate::components::RenderGeometry::Octree(_))
            );
            drop(renderable);
            drop(transform);

            let new_layer_mask = self
                .world
                .get::<&crate::viewport::RenderLayer>(entity)
                .map(|l| l.mask)
                .unwrap_or(crate::viewport::layer::DEFAULT);

            let world_matrix = glam::Mat4::from_scale_rotation_translation(
                transform_clone.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    transform_clone.rotation.x.to_radians(),
                    transform_clone.rotation.y.to_radians(),
                    transform_clone.rotation.z.to_radians(),
                ),
                transform_clone.position,
            );
            let world_arr = world_matrix.to_cols_array_2d();

            // Snapshot the current overlay/sculpt slice for this entity
            // from gpu_instances, then compute the new slice content
            // and deltas.
            let (cur_overlay_off, cur_overlay_count, cur_sculpt_off, cur_sculpt_count) = {
                let inst = &self.gpu_instances[gpu_idx];
                (
                    inst.overlay_offset,
                    inst.overlay_count,
                    inst.sculpt_offset,
                    inst.sculpt_count,
                )
            };

            // Empty-slice entries today carry overlay_offset = 0 (left
            // at the build_gpu_instance default). When the entity
            // gains its first non-empty slice we need a real insertion
            // point — right after the previous entity's slice end.
            // Same logic for sculpt below.
            let new_overlay: Vec<_> = self
                .paint_overlays
                .get(&entity)
                .map(|o| o.entries().to_vec())
                .unwrap_or_default();
            let new_sculpt: Vec<_> = self
                .sculpt_overlays
                .get(&entity)
                .map(|s| s.entries().to_vec())
                .unwrap_or_default();
            let new_overlay_count = new_overlay.len() as u32;
            let new_sculpt_count = new_sculpt.len() as u32;

            let overlay_insert_off: u32 = if cur_overlay_count > 0 {
                cur_overlay_off
            } else if gpu_idx == 0 {
                0
            } else {
                let prev = &self.gpu_instances[gpu_idx - 1];
                prev.overlay_offset + prev.overlay_count
            };
            let sculpt_insert_off: u32 = if cur_sculpt_count > 0 {
                cur_sculpt_off
            } else if gpu_idx == 0 {
                0
            } else {
                let prev = &self.gpu_instances[gpu_idx - 1];
                prev.sculpt_offset + prev.sculpt_count
            };

            let overlay_delta =
                new_overlay_count as i64 - cur_overlay_count as i64;
            let sculpt_delta =
                new_sculpt_count as i64 - cur_sculpt_count as i64;

            // Splice in place. `splice(start..end, iter)` replaces the
            // range and shifts the tail by the size difference. We
            // unconditionally splice so paint stamps that update
            // existing entries (same length, different content) also
            // propagate.
            if cur_overlay_count > 0 || new_overlay_count > 0 {
                let overlays = std::sync::Arc::make_mut(&mut self.gpu_instance_overlays);
                let start = overlay_insert_off as usize;
                let end = start + cur_overlay_count as usize;
                overlays.splice(start..end, new_overlay);
            }
            if cur_sculpt_count > 0 || new_sculpt_count > 0 {
                let sculpts = std::sync::Arc::make_mut(&mut self.gpu_instance_sculpts);
                let start = sculpt_insert_off as usize;
                let end = start + cur_sculpt_count as usize;
                sculpts.splice(start..end, new_sculpt);
            }

            // Patch this entity's gpu_instances row + shift downstream
            // overlay/sculpt offsets.
            let gpu_instances = std::sync::Arc::make_mut(&mut self.gpu_instances);
            {
                let inst = &mut gpu_instances[gpu_idx];
                inst.world = world_arr;
                inst.overlay_offset = overlay_insert_off;
                inst.overlay_count = new_overlay_count;
                inst.sculpt_offset = sculpt_insert_off;
                inst.sculpt_count = new_sculpt_count;
                inst.material_id = new_material_id;
                inst.layer_mask = new_layer_mask;
            }
            if overlay_delta != 0 || sculpt_delta != 0 {
                for inst in &mut gpu_instances[gpu_idx + 1..] {
                    if overlay_delta > 0 {
                        inst.overlay_offset =
                            inst.overlay_offset.saturating_add(overlay_delta as u32);
                    } else if overlay_delta < 0 {
                        inst.overlay_offset =
                            inst.overlay_offset.saturating_sub((-overlay_delta) as u32);
                    }
                    if sculpt_delta > 0 {
                        inst.sculpt_offset =
                            inst.sculpt_offset.saturating_add(sculpt_delta as u32);
                    } else if sculpt_delta < 0 {
                        inst.sculpt_offset =
                            inst.sculpt_offset.saturating_sub((-sculpt_delta) as u32);
                    }
                }
            }

            // mesh_draws / proxy_draws entry for this entity — keyed
            // by object_id == gpu_idx. We patch only `world` (Transform-
            // dirty case); `has_glass` stays at its prior value since
            // re-scanning the asset's leaves on every stamp is the
            // dominant remaining cost (see fn doc comment).
            if is_octree {
                if asset_handle_raw.is_some() {
                    let mesh_draws = std::sync::Arc::make_mut(&mut self.mesh_draws);
                    for d in mesh_draws.iter_mut() {
                        if d.object_id == gpu_idx as u32 {
                            d.world = world_arr;
                            break;
                        }
                    }
                }
            } else if !self.proxy_draws.is_empty() {
                let proxy_draws = std::sync::Arc::make_mut(&mut self.proxy_draws);
                for d in proxy_draws.iter_mut() {
                    if d.object_id == gpu_idx as u32 {
                        d.world = world_arr;
                        break;
                    }
                }
            }
        }

        true
    }
}
