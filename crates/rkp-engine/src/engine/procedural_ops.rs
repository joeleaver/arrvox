//! Procedural-object operations.
//!
//! Bake queue, dirty-tracking, per-node transform/param mutations, and
//! the procedural preview gizmo. Bake execution itself runs in the
//! background `BakeWorker` thread; these methods submit jobs and apply
//! the resulting artifacts back to the scene.

use super::state::EngineState;

impl EngineState {
    /// Read-modify-write a procedural node's local `Affine3A`.
    ///
    /// The stored transform is a full SRT compose, not just a
    /// translation — so each of the three `SetProceduralNode*` commands
    /// must preserve the two components it doesn't own. `f` takes
    /// `(scale, rotation, translation)` as decomposed from the current
    /// transform and returns the new triple.
    pub(crate) fn update_procedural_node_transform(
        &mut self,
        node_id: u32,
        f: impl FnOnce(glam::Vec3, glam::Quat, glam::Vec3) -> (glam::Vec3, glam::Quat, glam::Vec3),
    ) {
        let entity = match self.selected_entity {
            Some(e) => e,
            None => return,
        };
        let Ok(mut proc_geo) = self
            .world
            .get::<&mut crate::components::ProceduralGeometry>(entity)
        else {
            return;
        };
        let id = rkp_procedural::NodeId(node_id);
        let current = match proc_geo.tree.get(id) {
            Some(n) => n.transform,
            None => return,
        };

        // Decompose current transform.
        let t = current.translation.into();
        let m = current.matrix3;
        let sx = glam::Vec3::from(m.x_axis).length();
        let sy = glam::Vec3::from(m.y_axis).length();
        let sz = glam::Vec3::from(m.z_axis).length();
        let scale = glam::Vec3::new(sx.max(1e-8), sy.max(1e-8), sz.max(1e-8));
        let rot_mat = glam::Mat3::from_cols(
            (glam::Vec3::from(m.x_axis) / scale.x).into(),
            (glam::Vec3::from(m.y_axis) / scale.y).into(),
            (glam::Vec3::from(m.z_axis) / scale.z).into(),
        );
        let rotation = glam::Quat::from_mat3(&rot_mat);

        let (new_scale, new_rot, new_t) = f(scale, rotation, t);
        let new_affine =
            glam::Affine3A::from_scale_rotation_translation(new_scale, new_rot, new_t);
        proc_geo.tree.set_transform(id, new_affine);
        proc_geo.dirty = true;
    }

    /// Rebuild GPU objects from the hecs world.
    /// Per-tick procedural maintenance. Bakes any entity whose
    /// `pending_bake` has settled past the debounce window. We
    /// deliberately do NOT auto-bake on `dirty` alone: scene load
    /// restores procedurals with a cached spatial and a clean flag,
    /// but if a rogue edit path left `dirty = true` we'd silently
    /// re-run a potentially huge bake at startup — historically a
    /// source of UI freezes and crashes. Manual bakes (the build
    /// panel's Bake button, `BakeProceduralEntity`, `BakeAllDirty`)
    /// explicitly set `pending_bake` so they still ride this path.
    pub(crate) fn update_dirty_procedurals(&mut self) {
        use crate::components::*;

        let mut to_update: Vec<hecs::Entity> = Vec::new();

        // Debounce window for `pending_bake` — long enough to suppress
        // bakes mid-scrub on a slider, short enough to feel immediate
        // when the user releases.
        const BAKE_DEBOUNCE: std::time::Duration =
            std::time::Duration::from_millis(150);
        let now = std::time::Instant::now();
        // Bakes are sync and can take ~1s on big objects. Firing one
        // mid-drag freezes the engine tick and the gizmo can't track
        // the cursor for the duration — looks like the bake "ate" the
        // drag motion when the queued events finally drain. Defer
        // until the gizmo is released; the existing debounce timestamp
        // was bumped by the last drag tick, so this only delays the
        // bake by however long the user keeps dragging.
        let drag_active = self.gizmo.dragging || self.proc_gizmo.dragging;

        for (entity, (_renderable, proc_geo)) in self
            .world
            .query::<(&Renderable, &ProceduralGeometry)>()
            .iter()
        {
            // Only one bake per entity in flight at a time — the
            // worker channel could otherwise bloat with dozens of
            // requests during a long bake, all destined to be stale.
            // New edits while a bake runs will queue a fresh request
            // on the tick after the current one returns, via the
            // preserved `dirty` / `pending_bake` flags.
            if proc_geo.bake_in_flight {
                continue;
            }
            let pending_settled = !drag_active
                && proc_geo.pending_bake
                && proc_geo
                    .bake_dirty_at
                    .map(|t| now.duration_since(t) >= BAKE_DEBOUNCE)
                    .unwrap_or(true);
            if pending_settled {
                to_update.push(entity);
            }
        }

        for entity in to_update {
            self.enqueue_bake(entity);
        }
    }

    /// Move the just-set `Transform.scale` onto the procedural Root
    /// node (preserving Root's existing rotation / translation), set
    /// `Transform.scale` to the preview multiplier
    /// `new_root / last_evaluated_root` so the still-old baked voxels
    /// stretch to the user's target size during the debounce window,
    /// and queue an auto-bake. No-op for non-procedural entities.
    ///
    /// **Invariant**: after every call, for procedural entities,
    /// `Transform.scale == Root.scale / last_evaluated_root_scale`. The
    /// caller is expected to have written the user's intended absolute
    /// scale into `Transform.scale` first; this method captures it,
    /// stores it on the tree, and overwrites `Transform.scale` with the
    /// preview multiplier. Skipping this overwrite — even on a "no
    /// change" tick — causes a visual jump because the rendered size
    /// is `Transform.scale × baked_voxels`, and a stale absolute value
    /// in `Transform.scale` will multiply the already-baked-up voxels
    /// a second time.
    pub(crate) fn redirect_transform_scale_to_root(&mut self, entity: hecs::Entity) {
        use crate::components::*;
        // Hard cap on Root.scale per axis. The voxel budget scales as
        // the squared surface area (roughly) and the bake wall time
        // follows suit, so uncapped scaling blows through GPU memory
        // and wall-clock quickly. 20× the default primitive's extent
        // puts a 0.35 m sphere at 7 m radius, well inside the octree's
        // depth-11 cap and the 4.2 M-per-dispatch GPU chunking. Tune
        // in one place; the field meta's slider range below is kept
        // in lockstep.
        const SCALE_MIN: f32 = 0.01;
        const SCALE_MAX: f32 = 20.0;

        let user_scale = match self.world.get::<&Transform>(entity) {
            Ok(t) => t.scale,
            Err(_) => return,
        };
        // Procedurals-only: clamp here so the property slider + gizmo
        // both hit the same ceiling. Non-procedurals keep whatever
        // scale they were given.
        let is_procedural = self.world.get::<&ProceduralGeometry>(entity).is_ok();
        let user_scale = if is_procedural {
            glam::Vec3::new(
                user_scale.x.clamp(SCALE_MIN, SCALE_MAX),
                user_scale.y.clamp(SCALE_MIN, SCALE_MAX),
                user_scale.z.clamp(SCALE_MIN, SCALE_MAX),
            )
        } else {
            user_scale
        };
        let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) else {
            // Non-procedural entity — leave Transform.scale alone.
            return;
        };
        let root_id = proc_geo.tree.root();
        let root_xf = proc_geo
            .tree
            .get(root_id)
            .map(|n| n.transform)
            .unwrap_or(glam::Affine3A::IDENTITY);
        let (current_root_scale, rot, trans) = root_xf.to_scale_rotation_translation();
        // Push to Root + queue an auto-bake only when the value
        // actually changed; spammy slider events that re-write the
        // same scale shouldn't bump the debounce timestamp.
        if (user_scale - current_root_scale).length() >= 1e-6 {
            let new_root =
                glam::Affine3A::from_scale_rotation_translation(user_scale, rot, trans);
            proc_geo.tree.set_transform(root_id, new_root);
            proc_geo.pending_bake = true;
            proc_geo.bake_dirty_at = Some(std::time::Instant::now());
        }
        // Always restore the preview multiplier — see invariant above.
        let baked = proc_geo.last_evaluated_root_scale;
        let safe = |a: f32, b: f32| if b.abs() > 1e-6 { a / b } else { 1.0 };
        let preview = glam::Vec3::new(
            safe(user_scale.x, baked.x),
            safe(user_scale.y, baked.y),
            safe(user_scale.z, baked.z),
        );
        drop(proc_geo);
        if let Ok(mut t) = self.world.get::<&mut Transform>(entity) {
            t.scale = preview;
        }
    }

    /// Rescan `<project_root>/assets/shaders/*.wgsl`, replace the
    /// engine's `user_shader_registry`. Phase A: that's it — the
    /// registry is consulted on the next material upload to set
    /// `GpuMaterial.shader_id`. Phase B+ extends this to recompile
    /// pipelines and trigger frame-level re-renders.
    ///
    /// Idempotent: same source hash → no-op. Returns `None` on scan
    /// failure (parse error / IO); the previous registry stays in
    /// place so the renderer keeps working.
    pub(crate) fn reload_user_shaders(&mut self) -> Option<()> {
        use rkp_render::shader_composer;

        let Some(shaders_dir) = self.shaders_dir() else {
            return Some(());
        };

        let new_reg = match shader_composer::scan_dir(&shaders_dir) {
            Ok(r) => r,
            Err(e) => {
                self.console.warn(format!("user shader scan failed: {e}"));
                return None;
            }
        };

        if new_reg.source_hash() == self.user_shader_registry.source_hash() {
            return Some(());
        }

        self.user_shader_registry = new_reg;
        // The materials palette is rebuilt on every snapshot tick from
        // the registry, so newly-resolved shader_ids reach the GPU on
        // the very next frame without an explicit re-upload here. See
        // `engine/lifecycle.rs::build_palette` call site.
        Some(())
    }

    /// `<project_root>/assets/shaders/`, or `None` if no project loaded.
    pub(crate) fn shaders_dir(&self) -> Option<std::path::PathBuf> {
        let project_dir = self.project_dir.as_ref()?;
        Some(project_dir.join("assets").join("shaders"))
    }

    /// Compute the bake-cache sidecar path for a procedural entity:
    /// `{scene_dir}/{scene_stem}.bakes/{uuid}.rkp`. Returns `None` when
    /// the scene has no on-disk path yet (unsaved scratch session) or
    /// the entity has no UUID. The relative form
    /// (`{scene_stem}.bakes/{uuid}.rkp`) is what `SceneObject.procedural_cache`
    /// stores — use [`procedural_cache_relative`] for that.
    pub(crate) fn procedural_cache_path(&self, entity: hecs::Entity) -> Option<std::path::PathBuf> {
        let uuid = self.entity_uuids.get(&entity).copied()?;
        let scene_path = self.scene_path.as_ref()?;
        let parent = scene_path.parent()?;
        let stem = scene_path.file_stem()?;
        let mut dir = parent.to_path_buf();
        dir.push(format!("{}.bakes", stem.to_string_lossy()));
        dir.push(format!("{}.rkp", uuid));
        Some(dir)
    }

    /// Enqueue an async bake for a procedural entity. Bumps the
    /// entity's `bake_generation` and sends a [`BakeRequest`] to the
    /// worker thread. The result (an integrate-able [`BakeArtifact`])
    /// is picked up later by `drain_bake_results`. If the user keeps
    /// editing before the bake finishes, subsequent calls bump the
    /// generation and the old result gets dropped on arrival.
    ///
    /// Returns the generation number assigned, or `None` if the entity
    /// isn't a procedural.
    pub(crate) fn enqueue_bake(&mut self, entity: hecs::Entity) -> Option<u64> {
        use crate::components::*;

        let (tree_clone, base_voxel_size, generation) = {
            let mut proc_geo = self.world.get::<&mut ProceduralGeometry>(entity).ok()?;
            proc_geo.bake_generation = proc_geo.bake_generation.wrapping_add(1);
            // Clear the edit flags that triggered this bake — the
            // request captures the current state, so if no new edit
            // follows, we shouldn't re-fire next tick. A subsequent
            // edit will set `dirty` / `pending_bake` again, and the
            // NEXT tick after the bake returns will pick those up.
            proc_geo.dirty = false;
            proc_geo.pending_bake = false;
            proc_geo.bake_dirty_at = None;
            proc_geo.bake_in_flight = true;
            (proc_geo.tree.clone(), proc_geo.voxel_size, proc_geo.bake_generation)
        };
        // Worker needs the previous allocation to free it under the
        // integrate lock — pull it now so the worker doesn't round-
        // trip through the ECS.
        let prev_spatial = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| r.spatial.clone())
            .and_then(|g| g.into_octree());
        let (aabb, voxel_size) = procedural_voxel_params(&tree_clone, base_voxel_size);
        let instructions = rkp_procedural::flatten_tree(&tree_clone);
        let root_scale = tree_clone
            .get(tree_clone.root())
            .map(|n| n.transform.to_scale_rotation_translation().0)
            .unwrap_or(glam::Vec3::ONE);

        // Build the sidecar .rkp path: `{scene_dir}/{scene_stem}.bakes/{uuid}.rkp`.
        // Both the scene path and entity UUID must be known; unsaved
        // scratch scenes skip caching and just rely on next-spawn re-bake.
        let cache_output_path = self.procedural_cache_path(entity);

        let req = crate::bake_worker::BakeRequest {
            entity,
            generation,
            input: crate::bake_worker::BakeInput::Procedural(instructions),
            aabb,
            voxel_size,
            root_scale,
            prev_spatial,
            cache_output_path,
            generator_child: None,
        };
        if self.bake_worker.tx_request.send(req).is_err() {
            self.console.warn("bake worker channel closed".to_string());
            // Revert the in-flight flag — otherwise a permanently
            // dead channel would lock the entity out of future bakes.
            if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                proc_geo.bake_in_flight = false;
            }
            return None;
        }
        Some(generation)
    }

    /// Drain any finished bake results from the worker and integrate
    /// each one whose generation still matches the entity's latest
    /// request (stale results from superseded edits get silently
    /// dropped). Called once per tick, before rendering.
    pub(crate) fn drain_bake_results(&mut self) {
        use crate::components::*;
        use crate::bake_worker::BakeOutcome;

        // Drain everything the worker has produced since the last
        // tick. `try_recv` is non-blocking — we never wait here.
        while let Ok(result) = self.bake_worker.rx_result.try_recv() {
            // Generator-emitted child: spawn a new entity (anonymous)
            // or update an existing entity (persistent slot_key). The
            // `entity` field on the request is the generator (parent);
            // the spec carries everything needed downstream.
            if let Some(spec) = result.generator_child {
                match result.outcome {
                    BakeOutcome::Ok { spatial, voxel_count } => {
                        self.spawn_or_update_generated_child(
                            spec.parent_entity,
                            spec.local_transform,
                            spec.generation,
                            spec.slot_key,
                            spatial,
                            voxel_count,
                            spec.name_hint,
                        );
                    }
                    BakeOutcome::ProxyMeshOk { .. } => {
                        // Generator children always voxelize today
                        // (see context.rs's BakeRequest construction);
                        // a ProxyMeshOk for a generator child means
                        // the gate was bypassed somewhere upstream.
                        self.console.warn(format!(
                            "Generator child returned ProxyMesh — only Voxelize is supported \
                             for generator-emitted children (parent={:?}).",
                            spec.parent_entity,
                        ));
                    }
                    BakeOutcome::Failed => {
                        self.console.warn(format!(
                            "Generator child voxelization failed (parent={:?}, vs={:.4}).",
                            spec.parent_entity, result.voxel_size,
                        ));
                    }
                }
                continue;
            }

            // Regular procedural-entity bake below.
            let entity = result.entity;
            if !self.world.contains(entity) {
                continue;
            }

            // Every result clears the in-flight gate. If the user
            // edited after the request was sent, `dirty` /
            // `pending_bake` will already be set again and the next
            // tick's `update_dirty_procedurals` will enqueue a fresh
            // bake. We deliberately do NOT clear those flags here —
            // that would swallow the new edit.
            if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                proc_geo.bake_in_flight = false;
            }

            // Generation-mismatch = stale, drop.
            let current_gen = self
                .world
                .get::<&ProceduralGeometry>(entity)
                .map(|pg| pg.bake_generation)
                .unwrap_or(0);
            if result.generation != current_gen {
                continue;
            }

            match result.outcome {
                BakeOutcome::Ok { spatial, voxel_count } => {
                    // Switch from ProxyMesh → Voxelize: the bake
                    // worker can't release the proxy handle (it's
                    // owned by the renderer/scene_mgr on the engine
                    // side), so do it here before installing the
                    // new octree spatial.
                    self.release_proxy_handle_if_any(entity);
                    self.apply_bake_result(
                        entity,
                        result.root_scale,
                        spatial,
                        voxel_count,
                    );
                }
                BakeOutcome::ProxyMeshOk { surface_mesh, cluster } => {
                    self.apply_proxy_mesh_result(
                        entity,
                        result.root_scale,
                        surface_mesh,
                        cluster,
                    );
                }
                BakeOutcome::Failed => {
                    // Keep `dirty` / `pending_bake` intact so the user
                    // can retry (via a new edit or the Bake button) —
                    // clearing them would pretend the bake succeeded.
                    self.console.warn(format!(
                        "Procedural bake failed (voxel_size={:.4}, extent={:.1}).",
                        result.voxel_size,
                        (result.aabb.max - result.aabb.min).length(),
                    ));
                }
            }
        }
    }

    /// Apply a completed proxy-mesh bake. Releases any previous
    /// renderer handle on the entity, reserves a fresh one,
    /// commands the render thread to upload the GPU buffers, and
    /// stamps the entity's `Renderable.spatial` with the
    /// `RenderGeometry::ProxyMesh` variant.
    pub(crate) fn apply_proxy_mesh_result(
        &mut self,
        entity: hecs::Entity,
        baked_root_scale: glam::Vec3,
        surface_mesh: rkp_render::proc_surface_nets::SurfaceMesh,
        cluster: rkp_core::mesh_cluster::MeshletCluster,
    ) {
        use crate::components::*;

        // Free the previous geometry — either an octree allocation
        // (first switch from Voxelize to ProxyMesh) or a previous
        // ProxyMesh handle (re-bake of a proxy-mesh procedural).
        self.release_renderable_geometry(entity);

        // Proxy meshes carry their full shading payload (material,
        // color, dual-material blend) per-vertex via `ProxyVertex`.
        // No LeafAttr pool slot, no octree, no per-cell indirection.
        // The proxy raster reads vertex attrs directly and writes
        // the G-buffer.
        let handle = {
            let mut sm = self.scene_mgr.lock().unwrap();
            sm.reserve_procedural_handle()
        };
        let aabb = rkp_core::Aabb {
            min: surface_mesh.aabb_min,
            max: surface_mesh.aabb_max,
        };
        let _ = self
            .render_worker
            .commands
            .send(crate::render_frame::RenderCommand::UploadProxyMesh {
                handle_raw: handle.raw(),
                vertices: surface_mesh.vertices,
                indices: surface_mesh.indices,
                cluster,
            });

        if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
            renderable.spatial = Some(RenderGeometry::ProxyMesh(ProxyMeshData {
                handle,
                aabb,
            }));
            renderable.asset_handle = Some(handle);
            renderable.voxel_count = 0;
        } else {
            // Entity disappeared between request and result —
            // release the freshly-reserved handle so we don't leak.
            self.scene_mgr
                .lock()
                .unwrap()
                .release_procedural_handle(handle);
            let _ = self
                .render_worker
                .commands
                .send(crate::render_frame::RenderCommand::ReleaseProxyMesh {
                    handle_raw: handle.raw(),
                });
            return;
        }

        // Update last_evaluated_root_scale so the preview-multiplier
        // path zeroes back out (procedural now matches the latest
        // scale).
        if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
            proc_geo.dirty = false;
            proc_geo.pending_bake = false;
            proc_geo.last_evaluated_root_scale = baked_root_scale;
        }

        // PERF_DEBT B2+C3: this entity's geometry handle changed
        // — collider rebuild for THIS entity only.
        self.geometry_dirty.mark_entity(entity);
        // PERF_DEBT B1+C2: this entity's RkpGpuInstance row needs
        // refresh (asset slot may have moved, proxy handle
        // swapped). Structural since asset_id can change.
        self.gpu_objects_dirty.mark_entity(entity);
    }

    /// If `entity` currently has a `RenderGeometry::ProxyMesh`
    /// spatial, free its renderer handle (both scene-manager-side
    /// and render-thread GPU buffers) and clear the field. Returns
    /// silently if the entity has no proxy handle.
    pub(crate) fn release_proxy_handle_if_any(&mut self, entity: hecs::Entity) {
        use crate::components::*;
        let Some(handle) = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| {
                r.spatial
                    .as_ref()
                    .and_then(|g| g.as_proxy_mesh())
                    .map(|p| p.handle)
            })
        else {
            return;
        };
        self.scene_mgr
            .lock()
            .unwrap()
            .release_procedural_handle(handle);
        let _ = self
            .render_worker
            .commands
            .send(crate::render_frame::RenderCommand::ReleaseProxyMesh {
                handle_raw: handle.raw(),
            });
        if let Ok(mut r) = self.world.get::<&mut Renderable>(entity) {
            r.spatial = None;
            r.asset_handle = None;
            r.voxel_count = 0;
        }
    }

    /// Apply a completed (already-integrated) bake's result to the
    /// ECS. The heavy work — dealloc, artifact remap, pool writes —
    /// already happened on the bake worker under the `scene_mgr`
    /// lock; this runs on the engine tick and only touches the ECS,
    /// so it's microseconds.
    pub(crate) fn apply_bake_result(
        &mut self,
        entity: hecs::Entity,
        baked_root_scale: glam::Vec3,
        spatial: crate::components::SpatialData,
        voxel_count: u32,
    ) {
        use crate::components::*;

        if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
            renderable.voxel_count = voxel_count;
            renderable.spatial = Some(RenderGeometry::Octree(spatial));
        }

        // Recompute `Transform.scale` as `current_root / baked_root`
        // so the visual size stays equal to the user's latest intent
        // across the integrate:
        //     visual = Transform.scale × baked_voxels_world_scale
        //            = (current_root / baked_root) × baked_root
        //            = current_root
        // If no mid-bake edit happened, current_root == baked_root
        // and `Transform.scale` collapses to 1.
        let current_root_scale = self
            .world
            .get::<&ProceduralGeometry>(entity)
            .ok()
            .and_then(|pg| {
                pg.tree
                    .get(pg.tree.root())
                    .map(|n| n.transform.to_scale_rotation_translation().0)
            })
            .unwrap_or(baked_root_scale);
        let safe = |a: f32, b: f32| if b.abs() > 1e-6 { a / b } else { 1.0 };
        let new_transform_scale = glam::Vec3::new(
            safe(current_root_scale.x, baked_root_scale.x),
            safe(current_root_scale.y, baked_root_scale.y),
            safe(current_root_scale.z, baked_root_scale.z),
        );

        if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
            // `dirty` / `pending_bake` / `bake_dirty_at` were cleared
            // at enqueue time. If they're set *now*, the user edited
            // after the request was sent — preserve the new intent so
            // the next tick re-enqueues.
            proc_geo.last_evaluated_root_scale = baked_root_scale;
        }
        if let Ok(mut t) = self.world.get::<&mut Transform>(entity) {
            t.scale = new_transform_scale;
        }

        // PERF_DEBT B2+C3: scaled this entity only.
        self.geometry_dirty.mark_entity(entity);
        // PERF_DEBT B1+C2: scaled the procedural's Root + transform
        // on a single entity.
        self.gpu_objects_dirty.mark_entity(entity);
    }

}

/// Extract the rotation component from an `Affine3A` by normalizing
/// the 3×3 matrix's columns to remove per-axis scale. Matches the
/// decomposition used in `procedural_snapshot::decompose_affine`.
pub(crate) fn decompose_affine_rotation(t: &glam::Affine3A) -> glam::Quat {
    let m = t.matrix3;
    let sx = glam::Vec3::from(m.x_axis).length().max(1e-8);
    let sy = glam::Vec3::from(m.y_axis).length().max(1e-8);
    let sz = glam::Vec3::from(m.z_axis).length().max(1e-8);
    let rot_mat = glam::Mat3::from_cols(
        (glam::Vec3::from(m.x_axis) / sx).into(),
        (glam::Vec3::from(m.y_axis) / sy).into(),
        (glam::Vec3::from(m.z_axis) / sz).into(),
    );
    glam::Quat::from_mat3(&rot_mat)
}

pub(crate) fn procedural_voxel_params(tree: &rkp_procedural::ProceduralObject, base_voxel_size: f32) -> (rkp_core::Aabb, f32) {
    let tight = rkp_procedural::compute_bounds(tree);

    // Add margin for boundary sampling (same approach as voxelize_primitive).
    // Grid placement is handled by threading `grid_origin` through to the
    // shader (`local_origin - grid_origin` replaces the old
    // `local_origin + extent/2`), so we can return a tight AABB here
    // without wasting voxel budget on symmetric padding around the origin.
    let margin = base_voxel_size * 8.0 * 1.8 + base_voxel_size;
    let aabb = rkp_core::Aabb {
        min: tight.min - glam::Vec3::splat(margin),
        max: tight.max + glam::Vec3::splat(margin),
    };

    // Ensure depth won't exceed MAX_DEPTH (11). Max voxels per axis = 2^11 = 2048.
    let extent = aabb.max - aabb.min;
    let max_dim = extent.x.max(extent.y).max(extent.z);
    let max_voxels = 2048.0_f32; // 2^11
    let min_voxel_size = max_dim / max_voxels;
    let voxel_size = base_voxel_size.max(min_voxel_size);

    (aabb, voxel_size)
}


