//! Scene-construction command handlers: spawn, drop, drag preview,
//! entity transforms, project load/save.
//!
//! Second chunk of the `process_command` match. Called after
//! `process_cmd_edit` passes unmatched commands through.

use crate::command::EngineCommand;

use super::procedural_ops::procedural_voxel_params;
use super::state::{
    DragPreviewKind, DragPreviewState, EngineState, PendingDrop, PendingDropAction, PendingPick,
};

impl EngineState {
    pub(crate) fn process_cmd_scene(
        &mut self,
        cmd: EngineCommand,
    ) -> Result<(), EngineCommand> {
        match cmd {
            EngineCommand::CopyProceduralToNewVoxel { entity_id } => {
                use crate::components::*;
                let Some(src_entity) = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e))
                else {
                    self.console.warn("Copy: entity not found".to_string());
                    return Ok(());
                };
                // Same gate: won't copy a snapshot the user didn't
                // ask for.
                let can_copy = self
                    .world
                    .get::<&ProceduralGeometry>(src_entity)
                    .map(|pg| !pg.bake_in_flight && !pg.pending_bake && !pg.dirty)
                    .unwrap_or(false);
                if !can_copy {
                    self.console.warn(
                        "Copy: bake pending or in flight — let it settle first".to_string(),
                    );
                    return Ok(());
                }
                // Read what we need from the source entity. The
                // baked voxels live in shared scene pools, so we
                // re-voxelize the tree for the new entity rather
                // than refcounting the source's allocation — two
                // entities refcounting the same octree isn't what
                // we've got today (asset_cache is path-keyed, not
                // generalized). A second bake of the same tree
                // reuses the GPU evaluator's warmed pipelines and is
                // fast; we also go through the async path so the
                // engine tick stays smooth.
                let (src_name, src_transform, src_scale_for_bake, src_tree, src_voxel_size) = {
                    let name = self
                        .world
                        .get::<&EditorMetadata>(src_entity)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|_| "Procedural".to_string());
                    let transform = self
                        .world
                        .get::<&Transform>(src_entity)
                        .map(|t| (*t).clone())
                        .unwrap_or_else(|_| Transform::default());
                    let proc_geo = match self.world.get::<&ProceduralGeometry>(src_entity) {
                        Ok(pg) => pg,
                        Err(_) => {
                            self.console.warn("Copy: source has no ProceduralGeometry".to_string());
                            return Ok(());
                        }
                    };
                    let root_scale = proc_geo
                        .tree
                        .get(proc_geo.tree.root())
                        .map(|n| n.transform.to_scale_rotation_translation().0)
                        .unwrap_or(glam::Vec3::ONE);
                    (
                        name,
                        transform,
                        root_scale,
                        proc_geo.tree.clone(),
                        proc_geo.voxel_size,
                    )
                };

                // Spawn the destination entity. No ProceduralGeometry —
                // this is the static voxel copy. Starts with
                // spatial=None; the bake we enqueue below fills it.
                let new_name = self.unique_name(&format!("{src_name} (copy)"));
                let new_entity = self.world.spawn((
                    src_transform,
                    EditorMetadata { name: new_name.clone() },
                    Renderable {
                        primitive: None,
                        voxel_count: 0,
                        spatial: None,
                        ..Default::default()
                    },
                ));
                self.assign_entity_uuid(new_entity);
                self.scene_dirty = true;

                // Enqueue a bake for the copy. We're reusing the
                // async bake pipeline but the target entity has no
                // ProceduralGeometry — so `enqueue_bake` won't
                // accept it. Build the request by hand.
                let (aabb, voxel_size) = procedural_voxel_params(&src_tree, src_voxel_size);
                let instructions = rkp_procedural::flatten_tree(&src_tree);
                // `generation: 0` so the staleness check in
                // `drain_bake_results` (which reads the target
                // entity's current generation and defaults to 0 when
                // the entity has no ProceduralGeometry) matches and
                // we actually apply the result. Copy targets never
                // re-bake, so a single-shot value is fine.
                let req = crate::bake_worker::BakeRequest {
                    entity: new_entity,
                    generation: 0,
                    input: crate::bake_worker::BakeInput::Procedural(instructions),
                    aabb,
                    voxel_size,
                    root_scale: src_scale_for_bake,
                    prev_spatial: None,
                    // Copy targets have no ProceduralGeometry, so no
                    // scene reload would look for a sidecar here.
                    cache_output_path: None,
                    generator_child: None,
                    bake_mode: crate::components::BakeMode::Voxelize,
                };
                if self.bake_worker.tx_request.send(req).is_err() {
                    self.console.warn("Copy: bake worker channel closed".to_string());
                    return Ok(());
                }
                self.console.info(format!(
                    "Copied '{src_name}' → '{new_name}' (baking…)",
                ));
            }

            EngineCommand::SetBuildPreviewMode { mode } => {
                if let Some(vp) = self.viewports.get_mut(crate::viewport::ViewportId::BUILD) {
                    vp.preview_mode = mode;
                    eprintln!("[preview] build viewport preview_mode -> {mode:?}");
                } else {
                    eprintln!("[preview] SetBuildPreviewMode but no BUILD viewport registered");
                }
            }

            EngineCommand::SpawnPointLight => {
                use crate::components::*;
                let name = self.unique_name("Point Light");
                let mut transform = Transform::default();
                transform.position = self.camera.position + glam::Vec3::new(0.0, 2.0, 0.0);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    PointLight::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnSpotLight => {
                use crate::components::*;
                let name = self.unique_name("Spot Light");
                let mut transform = Transform::default();
                transform.position = self.camera.position + glam::Vec3::new(0.0, 3.0, 0.0);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    SpotLight::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnCamera => {
                use crate::components::*;
                let name = self.unique_name("Camera");
                let mut transform = Transform::default();
                transform.position = self.camera.position;
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    Camera::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnGenerator { generator_name } => {
                self.spawn_generator(&generator_name, None);
            }

            EngineCommand::SpawnGeneratorPreset { path } => {
                self.spawn_generator_preset(&path, None);
            }

            EngineCommand::DropGenerator { id, generator_name, x, y } => {
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::Generator { name: generator_name },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
            }

            EngineCommand::DropGeneratorPreset { id, path, x, y } => {
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::GeneratorPreset { path },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
            }

            EngineCommand::LoadAsset { path, position } => {
                self.spawn_asset(&path, position);
            }

            EngineCommand::DropAsset { id, path, x, y } => {
                // Drag-drop placement: issue a position-readback pick at
                // the drop pixel, queue a pending drop, and spawn when
                // the pick result arrives (process_pick_result handles
                // it — see `PendingDrop`).
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::Asset { path },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y,
                    ghost_pick_node_id: None,
                });
            }

            EngineCommand::DragPreviewEnter { id, source, x, y } => {
                // Clean up any orphaned preview from a previous drag
                // (two DragEnters with no Cancel / Commit between).
                if let Some(prev) = self.drag_preview.take() {
                    if let DragPreviewKind::Model { entity, .. } = prev.kind {
                        self.delete_entity(entity);
                    }
                }
                // Initial position: ground-plane raycast at the cursor
                // so the preview doesn't flash at the origin before the
                // first pick readback lands. Falls back to 3m in front
                // of the camera for rays that miss the plane.
                let provisional = {
                    let (ro, rd) = self.screen_to_ray_for_viewport(id, x as f32, y as f32);
                    if rd.y.abs() > 1e-6 {
                        let t = -ro.y / rd.y;
                        if t > 0.0 { ro + rd * t }
                        else { self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0) }
                    } else {
                        self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
                    }
                };
                let kind = match source {
                    crate::command::DragPreviewSource::Asset { path } => {
                        // Models: spawn the real asset now. The first
                        // pick readback snaps it to the cursor.
                        match self.spawn_asset_ex(&path, provisional, false) {
                            Some((entity, aabb_min_y)) => {
                                Some(DragPreviewKind::Model { entity, aabb_min_y })
                            }
                            None => None,
                        }
                    }
                    src @ (crate::command::DragPreviewSource::Generator { .. }
                        | crate::command::DragPreviewSource::GeneratorPreset { .. }) => {
                        // Generators: no spawn yet — the real entity
                        // only materialises on commit. Meanwhile draw a
                        // 1 m half-extent wireframe box at the cursor.
                        // We don't know the baked bounds until after a
                        // run, so a single conservative default beats
                        // introspecting parameters per-generator.
                        Some(DragPreviewKind::Generator {
                            source: src,
                            gizmo_half: glam::Vec3::splat(0.5),
                        })
                    }
                };
                if let Some(kind) = kind {
                    self.drag_preview = Some(DragPreviewState {
                        viewport: id,
                        kind,
                        last_surface_pos: Some(provisional),
                        last_cursor: (x, y),
                    });
                    self.pending_pick = Some(PendingPick {
                        viewport: id, x, y, ghost_pick_node_id: None,
                    });
                }
            }

            EngineCommand::DragPreviewOver { id, x, y } => {
                if let Some(preview) = self.drag_preview.as_mut() {
                    if preview.viewport == id {
                        preview.last_cursor = (x, y);
                        // Overwrite any in-flight request with the
                        // freshest pixel. Render-side `pick_in_flight`
                        // gate throttles to one readback per frame, so
                        // newer coords win naturally.
                        self.pending_pick = Some(PendingPick {
                            viewport: id, x, y, ghost_pick_node_id: None,
                        });
                    }
                }
            }

            EngineCommand::DragPreviewCommit => {
                if let Some(preview) = self.drag_preview.take() {
                    match preview.kind {
                        // Models: entity is already live at the final
                        // position. Just retire the preview state —
                        // subsequent pick results won't touch it.
                        DragPreviewKind::Model { .. } => {}
                        // Generators: now spawn the real source at the
                        // last-known surface position. Falls back to a
                        // ground-plane cast at the final cursor pixel
                        // if no valid surface hit ever landed.
                        DragPreviewKind::Generator { source, .. } => {
                            let pos = preview.last_surface_pos.unwrap_or_else(|| {
                                let (cx, cy) = preview.last_cursor;
                                let (ro, rd) = self.screen_to_ray_for_viewport(
                                    preview.viewport, cx as f32, cy as f32,
                                );
                                if rd.y.abs() > 1e-6 {
                                    let t = -ro.y / rd.y;
                                    if t > 0.0 { ro + rd * t }
                                    else { self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0) }
                                } else {
                                    self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
                                }
                            });
                            match source {
                                crate::command::DragPreviewSource::Generator { name } => {
                                    self.spawn_generator(&name, Some(pos));
                                }
                                crate::command::DragPreviewSource::GeneratorPreset { path } => {
                                    self.spawn_generator_preset(&path, Some(pos));
                                }
                                crate::command::DragPreviewSource::Asset { .. } => {
                                    // Unreachable — Asset paths produce
                                    // `DragPreviewKind::Model`, handled
                                    // above.
                                }
                            }
                        }
                    }
                }
            }

            EngineCommand::DragPreviewCancel => {
                if let Some(preview) = self.drag_preview.take() {
                    // Only the model path has a live entity to delete;
                    // generators never spawned anything during drag.
                    if let DragPreviewKind::Model { entity, .. } = preview.kind {
                        self.delete_entity(entity);
                    }
                }
            }

            EngineCommand::Pick { id, x, y } => {
                // BUILD + Voxel: picking doesn't make sense. The G-buffer
                // slot the raymarch uses for NodeId is occupied by
                // secondary_material_id in voxel mode, so decoding
                // would return arbitrary node ids. Skip entirely —
                // the user selects tree nodes via the build panel
                // in voxel mode.
                if id == crate::viewport::ViewportId::BUILD {
                    let is_raymarch = self
                        .viewports
                        .get(crate::viewport::ViewportId::BUILD)
                        .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                        .unwrap_or(false);
                    if !is_raymarch {
                        return Ok(());
                    }
                }

                // Route the pick by viewport — MAIN picks scene entities
                // (old path), BUILD picks procedural primitives when in
                // raymarch preview. Either way, a click landing on a
                // gizmo axis should NOT fall through to pick — that
                // deselects the currently-manipulated object and
                // prevents the drag from starting. Each viewport has
                // its own gizmo state; pick the right one.
                let gizmo_blocking = match id {
                    crate::viewport::ViewportId::MAIN => {
                        self.gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
                            || self.gizmo.dragging
                    }
                    crate::viewport::ViewportId::BUILD => {
                        self.proc_gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
                            || self.proc_gizmo.dragging
                    }
                    _ => false,
                };
                if !gizmo_blocking {
                    // Ghost-priority pick: on BUILD in raymarch mode,
                    // CPU-raycast the tree's ghost-role primitives at
                    // the click ray. If any hits, remember which one —
                    // it takes priority over the G-buffer decode
                    // (matches the visual rule that a ghost painted
                    // on the pixel owns the click).
                    let ghost_pick_node_id = self
                        .compute_ghost_pick(id, x, y);
                    self.pending_pick = Some(PendingPick {
                        viewport: id, x, y, ghost_pick_node_id,
                    });
                }
            }

            EngineCommand::ImportAsset { source_path } => {
                let source = std::path::PathBuf::from(&source_path);
                let output = crate::import_worker::rkp_output_path(&source);
                self.import_worker.submit(crate::import_worker::ImportRequest {
                    source_path: source,
                    output_path: output,
                    config: crate::import_worker::default_import_config(),
                });
            }

            EngineCommand::SetObjectPosition { entity_id, position } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.position = position;
                    }
                }
            }

            EngineCommand::SetObjectRotation { entity_id, rotation } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.rotation = rotation;
                    }
                }
            }

            EngineCommand::SetObjectScale { entity_id, scale } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.scale = scale;
                    }
                }
            }

            EngineCommand::SelectEntity { entity_id } => {
                self.selected_entity = self.resolve_entity(&entity_id);
            }

            EngineCommand::DeleteObject { entity_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    self.delete_entity(entity);
                }
            }

            EngineCommand::ReorderEntity { entity, new_parent, new_order } => {
                self.handle_reorder(entity, new_parent, new_order);
            }

            EngineCommand::DeleteSelected => {
                if let Some(entity) = self.selected_entity {
                    self.delete_entity(entity);
                }
            }

            EngineCommand::DuplicateObject { entity_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    self.duplicate_entity(entity);
                }
            }

            EngineCommand::DuplicateSelected => {
                if let Some(entity) = self.selected_entity {
                    self.duplicate_entity(entity);
                }
            }

            EngineCommand::NewProject { path } => {
                let path = std::path::PathBuf::from(&path);
                match crate::project::create_project(&path) {
                    Ok(project_dir) => {
                        self.clear_scene();
                        let project_name = project_dir.file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let project_file = project_dir.join(format!("{project_name}.rkproject"));
                        self.project_dir = Some(project_dir.clone());
                        self.project_path = Some(project_file);
                        self.scene_path = Some(project_dir.join("scenes/default.rkscene"));
                        self.project_name = project_name;
                        self.project_loaded = true;
                        self.project_dirty = true;
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                        self.scan_models();
                        if let Some(ref dir) = self.project_dir {
                            // Write starter materials before scanning.
                            crate::material_library::write_starter_materials(
                                &dir.join("assets/materials"),
                            );
                            self.material_lib.scan(&dir.join("assets/materials"));
                        }
                        self.init_file_watcher();
                        // Pick up any pre-existing user shaders shipped
                        // with the project. No-op if `assets/shaders/`
                        // is empty.
                        let _ = self.reload_user_shaders();
                        self.scaffold_and_build_gameplay();
                        self.auto_import_meshes();
                        if let Some(ref pp) = self.project_path {
                            crate::recent_projects::add_recent(&self.project_name, &pp.to_string_lossy());
                        }
                    }
                    Err(e) => eprintln!("[RkpEngine] new project failed: {e}"),
                }
            }

            EngineCommand::OpenProject { path } => {
                let path = std::path::PathBuf::from(&path);
                match crate::project::load_project(&path) {
                    Ok((project, project_dir)) => {
                        self.clear_scene();
                        self.project_dir = Some(project_dir.clone());
                        self.project_path = Some(path);
                        self.project_name = project.name;
                        self.project_loaded = true;
                        self.project_dirty = true;
                        // Cache + flag the editor layout so the editor
                        // hydrates its docking state on the next tick.
                        // `None` is meaningful — it means "reset to
                        // default" for projects saved pre-persistence.
                        self.editor_layout_json = project.editor_layout;
                        self.editor_layout_pending = true;

                        // Scaffold + build gameplay BEFORE loading the scene,
                        // so gameplay components (Spin, Health, etc.) are registered
                        // and can be deserialized from the scene file.
                        self.scaffold_and_build_gameplay();

                        // `scene_path` must be set BEFORE loading so
                        // `load_scene_from_file` can resolve
                        // procedural bake-cache sidecars relative to
                        // the scene file's directory.
                        let scene_path = project_dir.join(format!("scenes/{}.rkscene", project.default_scene));
                        self.scene_path = Some(scene_path.clone());
                        if scene_path.exists() {
                            self.load_scene_from_file(&scene_path);
                        }

                        self.scan_models();
                        if let Some(ref dir) = self.project_dir {
                            // Reseed any starter materials the project
                            // is missing (user deleted them, or schema
                            // churn left stale files that were cleaned
                            // out). No-op for starters that exist.
                            crate::material_library::write_starter_materials(
                                &dir.join("assets/materials"),
                            );
                            self.material_lib.scan(&dir.join("assets/materials"));
                        }
                        self.init_file_watcher();
                        let _ = self.reload_user_shaders();
                        self.auto_import_meshes();
                        if let Some(ref pp) = self.project_path {
                            crate::recent_projects::add_recent(&self.project_name, &pp.to_string_lossy());
                        }
                    }
                    Err(e) => eprintln!("[RkpEngine] open project failed: {e}"),
                }
            }

            EngineCommand::SaveScene { path } => {
                let save_path = path.map(std::path::PathBuf::from)
                    .or_else(|| self.scene_path.clone());
                if let Some(save_path) = save_path {
                    let scene = self.build_scene_file();
                    if let Err(e) = crate::scene_io::save_scene(&scene, &save_path) {
                        eprintln!("[RkpEngine] save scene failed: {e}");
                    }
                    self.scene_path = Some(save_path);
                }
                // Persist the project descriptor alongside the scene so
                // the cached editor layout (and anything else on
                // ProjectFile) actually hits disk on Ctrl+S. Without
                // this, layout state would only be written by explicit
                // SaveProject, which the UI doesn't wire up.
                self.save_project_file();
            }

            EngineCommand::SaveProject => {
                self.save_project_file();
            }

            EngineCommand::SetEditorLayout { json } => {
                // Cache only — actual write happens on save. Don't echo
                // back to the editor; it's the source of truth for this.
                self.editor_layout_json = Some(json);
            }

            // ── Raw input → feed to InputSystem ──────────────────────
            other => return Err(other),
        }
        Ok(())
    }
}
