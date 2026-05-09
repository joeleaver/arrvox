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
        // Wipe last frame's scatter plan — rebuilt below per skinned
        // entity. Bone-field cells are u64s of offset; cells × 8 B =
        // total bone-field bytes.
        self.skin_dispatches.clear();
        let mut running_bone_field_cells: u32 = 0;
        let mut running_bone_field_occ_u32s: u32 = 0;

        // Pose cache for the scatter-skip check. Built in lock-step
        // with the planning loop and swapped with `last_skin_poses`
        // at the end of this function. Equal maps → skip scatter.
        let mut this_frame_poses: std::collections::HashMap<hecs::Entity, Vec<glam::Mat4>>
            = std::collections::HashMap::new();

        self.gpu_assets.clear();
        self.gpu_instances.clear();
        self.gpu_instance_overlays.clear();
        self.splat_draws.clear();
        self.gpu_to_entity.clear();
        self.entity_to_gpu.clear();

        // Refresh `material_is_glass` from the material library —
        // O(slot_count), typically dozens. If the resulting Vec
        // differs from last frame, clear `asset_has_glass_cache` so
        // every asset re-scans its leaves on its next draw. Also
        // invalidate the cache when geometry changes
        // (`remap_entity_material` mutates leaf materials in-place
        // and bumps geometry_epoch).
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
            let geom_epoch = self
                .geometry_epoch_handle
                .load(std::sync::atomic::Ordering::Acquire);
            if geom_epoch > self.material_glass_lib_epoch {
                self.asset_has_glass_cache.clear();
                self.material_glass_lib_epoch = geom_epoch;
            }
        }
        // Per-frame asset table — `octree_root` → index into
        // `gpu_assets`. Two entities sharing one .rkp asset share one
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
                let asset_handles: std::collections::HashSet<rkp_render::AssetHandle> = self
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
                let spatial_handle = rkp_core::scene_node::SpatialHandle::Octree {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let mut skinning = self.bone_matrix_allocator.binding(entity);
                // Plan the skin-deform scatter for this entity if it's
                // animated AND the asset has baked skinning metadata.
                if self.skinning_enabled {
                    if let (Some(bind), Some(handle)) = (skinning, renderable.asset_handle) {
                        // Cache lookup — no scene_mgr lock here. The
                        // cache is refreshed at the top of this fn
                        // only when geometry epoch advances.
                        if let (Some(skel), Some(skin_data)) = (
                            self.world.get::<&crate::components::Skeleton>(entity).ok(),
                            self.skinning_data_cache.get(&handle),
                        ) {
                            if let Some(plan) = crate::scene_sync::plan_skin_dispatch(
                                bind.bone_buffer_offset,
                                bind.bone_count,
                                &skel.current_pose,
                                skin_data,
                                spatial.voxel_size,
                                &mut running_bone_field_cells,
                                &mut running_bone_field_occ_u32s,
                                if self.dqs_enabled { 1 } else { 0 },
                                bind.bone_dq_offset,
                            ) {
                                // Copy the plan's bone-field geometry
                                // into the SkinnedBinding so the GPU
                                // object carries the same coords the
                                // scatter wrote to. Without this the
                                // march would descend a bone field
                                // sized in one frame and origin'd from
                                // another.
                                skinning = Some(crate::scene_sync::SkinnedBinding {
                                    bone_count: bind.bone_count,
                                    bone_buffer_offset: bind.bone_buffer_offset,
                                    bone_field_offset: plan.uniforms.bone_field_offset,
                                    bone_field_dims: [
                                        plan.uniforms.bone_field_dim_x,
                                        plan.uniforms.bone_field_dim_y,
                                        plan.uniforms.bone_field_dim_z,
                                    ],
                                    bone_field_origin: [
                                        plan.uniforms.grid_origin_x,
                                        plan.uniforms.grid_origin_y,
                                        plan.uniforms.grid_origin_z,
                                    ],
                                    bone_field_occ_offset: plan.uniforms.bone_field_occ_offset,
                                    bone_dq_offset: bind.bone_dq_offset,
                                });
                                self.skin_dispatches.push(plan);
                                // Cache this entity's pose for the
                                // scatter-skip check at the end of the
                                // function. Only records entities that
                                // made it to a plan — a plan bail
                                // below treats this entity as "not
                                // animated this frame", same as last
                                // frame's cache if it was also missing.
                                this_frame_poses.insert(entity, skel.current_pose.clone());
                            } else {
                                // Plan bailed (no extent, or dims > cap).
                                // Leave skinning = None so march falls
                                // back to the rigid path for this
                                // entity.
                                skinning = None;
                            }
                        }
                    }
                }
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
                        self.gpu_assets.push(crate::scene_sync::build_gpu_asset(
                            &spatial.aabb,
                            spatial.grid_origin,
                            &spatial_handle,
                            spatial.voxel_size,
                            bone_count_for_asset,
                        ));
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
                        self.gpu_instance_overlays.extend_from_slice(overlay.entries());
                        inst.overlay_offset = off;
                        inst.overlay_count = count;
                    }
                }
                self.entity_to_gpu.insert(entity, self.gpu_instances.len());
                self.gpu_to_entity.push(entity);
                self.gpu_instances.push(inst);

                // Splat-rasterizer per-instance draw record. Only
                // `Renderable` entities with an `asset_handle` (i.e.
                // loaded `.rkp` assets, not procedurals) make it into
                // the splat path — procedurals don't have splat data
                // extracted today. Built every frame because world
                // transforms can change per-tick; the engine ships
                // this list to the render thread alongside the
                // existing `gpu_instances`.
                if let Some(handle) = renderable.asset_handle {
                    // Per-instance skinning state (Phase 6.6) — when
                    // `skinning` is `Some(_)` the entity has both a
                    // live `Skeleton` and a baked skin-meta payload,
                    // and the skin_deform plan above already pushed
                    // bone matrices into the per-frame palette. The
                    // mesh VS picks them up via `bone_offset_*`.
                    // `dqs_enabled` is the renderer-wide toggle that
                    // also drives `skin_deform`'s mode — keeping mesh
                    // raster and the legacy march path aligned.
                    let (skinning_mode, bone_offset_lbs, bone_offset_dqs) = match skinning {
                        Some(b) => (
                            if self.dqs_enabled { 1 } else { 0 },
                            b.bone_buffer_offset,
                            b.bone_dq_offset,
                        ),
                        None => (
                            rkp_render::splat_pass::SKINNING_MODE_NONE,
                            0,
                            0,
                        ),
                    };
                    // The asset's `grid_origin` (in object-local /
                    // mesh frame) is what the surface-mesh extractor
                    // baked into every vertex's `local_pos`. Bone
                    // matrices in `skin_deform`'s palette operate on
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
                            let mut found = false;
                            for slot in &leaf_slots {
                                if *slot >= pool_size {
                                    continue;
                                }
                                let attr = sm.leaf_attr_pool.get(*slot);
                                let mat_id = attr.material_primary as usize;
                                if mat_id < self.material_is_glass.len()
                                    && self.material_is_glass[mat_id]
                                {
                                    found = true;
                                    break;
                                }
                            }
                            drop(sm);
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

                    self.splat_draws.push(rkp_render::splat_pass::SplatDraw {
                        asset_handle_raw: handle.raw(),
                        world: world_matrix.to_cols_array_2d(),
                        object_id: gpu_idx,
                        grid_origin,
                        bone_offset_lbs,
                        bone_offset_dqs,
                        skinning_mode,
                        has_glass,
                    });
                }
            } else if let Some(proxy) = renderable.spatial.as_ref().and_then(|g| g.as_proxy_mesh()) {
                // Procedural rendered as a triangle proxy mesh — no
                // octree, no leaf_attr pool entry, no skinning. The
                // mesh raster path consumes a SplatDraw with the
                // proxy's reserved asset handle plus the entity's
                // world transform; everything downstream (LOD-select,
                // shadow, shading) reuses the existing pipeline since
                // the GPU buffers were uploaded under that handle by
                // `RenderCommand::UploadProxyMesh`.
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

                // Build a minimal GpuInstance — proxy meshes don't
                // share a baked asset record, so synthesize one
                // per-instance asset entry covering this mesh's AABB.
                // `asset_table` keys on octree root_offset, which proxy
                // meshes don't have; the asset record only matters for
                // the AABB the LOD-select pass reads. Use the proxy's
                // own AABB.
                let proxy_aabb = proxy.aabb;
                let asset_id = self.gpu_assets.len() as u32;
                self.gpu_assets.push(crate::scene_sync::build_gpu_asset(
                    &proxy_aabb,
                    glam::Vec3::ZERO, // no grid origin — proxy mesh writes
                                      // world-space positions today.
                    &rkp_core::scene_node::SpatialHandle::Octree {
                        root_offset: 0,
                        len: 0,
                        depth: 0,
                        base_voxel_size: 0.0,
                    },
                    0.0,
                    0,
                ));
                let mut inst = crate::scene_sync::build_gpu_instance(
                    &world_matrix,
                    asset_id,
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
                self.gpu_instances.push(inst);

                self.splat_draws.push(rkp_render::splat_pass::SplatDraw {
                    asset_handle_raw: proxy.handle.raw(),
                    world: world_matrix.to_cols_array_2d(),
                    object_id: gpu_idx,
                    grid_origin: [0.0, 0.0, 0.0],
                    bone_offset_lbs: 0,
                    bone_offset_dqs: 0,
                    skinning_mode: rkp_render::splat_pass::SKINNING_MODE_NONE,
                    has_glass: false,
                });
            }
        }

        // Each bone-field cell is a `vec2<u32>` (packed bone indices +
        // weights) = 8 bytes. Used by the render loop to size the
        // scene's bone_field_buffer before the scatter dispatch.
        self.skin_bone_field_bytes = (running_bone_field_cells as u64).saturating_mul(8);
        self.skin_bone_field_occ_bytes = (running_bone_field_occ_u32s as u64).saturating_mul(4);


        // Pause-aware scatter skip: if the set of skinned entities and
        // their per-bone matrices are byte-identical to last frame,
        // the `bone_field` buffer still holds valid data — render loop
        // skips both the clear and the scatter dispatch. Big win when
        // the user pauses the animation to inspect a frame.
        //
        // Empty-to-empty doesn't count as a reuse opportunity: there
        // was nothing to clear last frame either, so the render loop
        // already skips via `skin_dispatches.is_empty()`.
        self.skin_reuse = !this_frame_poses.is_empty()
            && this_frame_poses == self.last_skin_poses;
        self.last_skin_poses = this_frame_poses;

    }
}
