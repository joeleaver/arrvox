//! Per-frame derive of `gpu_objects` + skin dispatch plans from the ECS.
//!
//! Flattens `world` → `gpu_objects` on every tick where either geometry
//! or transforms are dirty, updates the UUID ↔ gpu-index maps used for
//! pick resolution, and walks skinned entities to build the skin
//! scatter dispatch list the render thread consumes.

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

        self.gpu_objects.clear();
        self.gpu_to_entity.clear();
        self.entity_to_gpu.clear();

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
            if let Some(ref spatial) = renderable.spatial {
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
                let gpu_idx = self.gpu_objects.len() as u32;
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
                let mut gpu_obj = crate::scene_sync::build_gpu_object(
                    &world_matrix,
                    &spatial.aabb,
                    spatial.grid_origin,
                    &spatial_handle,
                    spatial.voxel_size,
                    renderable.material_id,
                    gpu_idx,
                    skinning,
                );
                // Render-layer mask — entity opt-in via RenderLayer
                // component, otherwise the system DEFAULT bit.
                gpu_obj.layer_mask = self
                    .world
                    .get::<&crate::viewport::RenderLayer>(entity)
                    .map(|l| l.mask)
                    .unwrap_or(crate::viewport::layer::DEFAULT);
                self.entity_to_gpu.insert(entity, self.gpu_objects.len());
                self.gpu_to_entity.push(entity);
                self.gpu_objects.push(gpu_obj);
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
