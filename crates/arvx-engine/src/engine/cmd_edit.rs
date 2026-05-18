//! Editor-side command handlers: viewport camera + procedural node ops.
//!
//! This file owns the first chunk of the `process_command` match. The
//! dispatcher in `command_handler` calls `process_cmd_edit` first;
//! arms it doesn't match fall through as `Err(cmd)` to the next chunk.

use crate::command::EngineCommand;

use super::procedural_params::{apply_procedural_param, parse_node_kind};
use super::state::EngineState;

impl EngineState {
    pub(crate) fn process_cmd_edit(
        &mut self,
        cmd: EngineCommand,
    ) -> Result<(), EngineCommand> {
        match cmd {

            EngineCommand::SetCamera { id, position, yaw, pitch, fov } => {
                // Phase 3: only MAIN is wired to the legacy `self.camera`.
                // Non-MAIN viewports update their own editor_camera once
                // multi-viewport rendering lands (Phase 4+).
                if id == crate::viewport::ViewportId::MAIN {
                    self.camera.position = position;
                    self.camera.yaw = yaw;
                    self.camera.pitch = pitch;
                    self.camera.fov = fov;
                    self.sync_main_viewport_from_legacy_camera();
                } else if let Some(vp) = self.viewports.get_mut(id) {
                    use crate::viewport::{EditorCamera, FlyCameraState};
                    vp.editor_camera = EditorCamera::Fly(FlyCameraState {
                        position, yaw, pitch, fov,
                        near: 0.01, far: 1000.0,
                    });
                }
            }

            EngineCommand::Resize { id, width, height } => {
                // Each VR has its own per-resolution pass chain now, so
                // Resize is per-viewport. Resizing BUILD doesn't affect
                // MAIN (and vice versa). The editor sends Resize on
                // every event (mouse move etc.) and relies on this
                // handler to no-op when the size hasn't actually changed
                // — without that guard `vr.resize` rebuilds bloom /
                // tonemap each frame and `environment_dirty` ticks every
                // tick.
                let unchanged = self
                    .viewports
                    .get(id)
                    .map(|vp| vp.width == width && vp.height == height)
                    .unwrap_or(false);
                if unchanged {
                    return Ok(());
                }
                let _ = self.render_worker.commands.send(
                    crate::render_frame::RenderCommand::ResizeViewport { id, width, height },
                );
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.width = width;
                    vp.height = height;
                }
                // `vr.resize` reconstructs the bloom + tonemap passes with
                // their hard-coded defaults, so the scene's exposure and
                // bloom knobs have to be re-uploaded afterwards on EVERY
                // resize — not just MAIN. Previously this was gated to
                // MAIN and BUILD's first Resize (sent by the editor when
                // the build panel sizes up) left it running with default
                // exposure → blown-out preview until something else
                // flipped environment_dirty back on.
                self.environment_dirty = true;
                if id == crate::viewport::ViewportId::MAIN {
                    // MAIN drives the legacy width/height on EngineState
                    // for hot paths that haven't migrated (sculpt/paint
                    // ray math).
                    self.width = width;
                    self.height = height;
                    self.environment_dirty = true;
                    self.environment_ui_dirty = true;
                    eprintln!("[ArvxEngine] MAIN resized to {}x{}", width, height);
                }
            }

            EngineCommand::SetViewportVisible { id, visible } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.visible = visible;
                }
            }

            EngineCommand::SetViewportFilter { id, base_layers, focus_entity_id } => {
                let focus_entity = focus_entity_id
                    .and_then(|uuid| self.uuid_to_entity.get(&uuid).copied());
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.filter = crate::viewport::SceneFilter {
                        base_layers,
                        focus_entity,
                    };
                }
            }

            EngineCommand::SetViewportCamera { id, entity_id } => {
                if let Some(entity) = self.uuid_to_entity.get(&entity_id).copied() {
                    if let Some(vp) = self.viewports.get_mut(id) {
                        vp.runtime_override =
                            Some(crate::viewport::CameraSource::Entity(entity));
                    }
                }
            }

            EngineCommand::ClearViewportCamera { id } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.runtime_override = None;
                }
            }

            EngineCommand::SetViewportMode { id, mode } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.mode = mode;
                }
            }

            EngineCommand::SpawnProceduralObject { name, leaf_kind } => {
                use crate::components::*;
                let name = self.unique_name(&name);
                let mut proc_geo = match leaf_kind {
                    Some(kind) => ProceduralGeometry::with_leaf(parse_node_kind(&kind)),
                    None => ProceduralGeometry::default_sphere(),
                };
                // Freshly-spawned procedurals should bake immediately so
                // the user sees a visible object. We set `pending_bake`
                // (not just `dirty`) so the debounced auto-bake path in
                // `update_dirty_procedurals` picks it up — scene *load*
                // deliberately never auto-bakes, so riding on `dirty`
                // alone would leave the spawn invisible.
                proc_geo.dirty = false;
                proc_geo.pending_bake = true;
                proc_geo.bake_dirty_at = Some(std::time::Instant::now());
                let entity = self.world.spawn((
                    Transform::default(),
                    EditorMetadata { name: name.clone() },
                    Renderable {
                        primitive: Some("procedural".to_string()),
                        voxel_count: 0,
                        spatial: None,
                        ..Default::default()
                    },
                    proc_geo,
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty.mark_entity(entity);
                self.console.info(format!("Spawned procedural '{name}' (baking…)"));
            }

            EngineCommand::SelectProceduralNode { node_id } => {
                self.selected_procedural_node = node_id;
            }

            EngineCommand::SetProceduralVoxelSize { tier } => {
                const VOXEL_TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if let Ok(vs) = tier.parse::<f32>() {
                            let snapped = VOXEL_TIERS.iter()
                                .min_by(|a, b| ((**a) - vs).abs().partial_cmp(&((**b) - vs).abs()).unwrap())
                                .copied()
                                .unwrap_or(0.02);
                            if (snapped - proc_geo.voxel_size).abs() > 1e-6 {
                                proc_geo.voxel_size = snapped;
                                // Auto-bake — voxel-size changes are
                                // single-click tier flips; the debounce
                                // window absorbs rapid double-clicks but
                                // otherwise the user expects an immediate
                                // rebake.
                                proc_geo.pending_bake = true;
                                proc_geo.bake_dirty_at =
                                    Some(std::time::Instant::now());
                            }
                        }
                    }
                }
            }

            EngineCommand::AddProceduralNode { parent_node_id, kind } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let parent = arvx_procedural::NodeId(parent_node_id);
                        let node_kind = parse_node_kind(&kind);
                        // Root accepts children directly — no
                        // auto-promote, no special cases. Drops onto
                        // a leaf are rejected by the UI (is_leaf →
                        // no "+" affordance).
                        let new_id = proc_geo.tree.add_child(parent, node_kind);
                        proc_geo.dirty = true;
                        self.selected_procedural_node = Some(new_id.0);
                    }
                }
            }

            EngineCommand::RemoveProceduralNode { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = arvx_procedural::NodeId(node_id);
                        if proc_geo.tree.remove(id) {
                            proc_geo.dirty = true;
                            if self.selected_procedural_node == Some(node_id) {
                                self.selected_procedural_node = None;
                            }
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNodeUp { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_up(arvx_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNodeDown { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_down(arvx_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::ReparentProceduralNode { node_id, new_parent_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.reparent(
                            arvx_procedural::NodeId(node_id),
                            arvx_procedural::NodeId(new_parent_id),
                        ) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNode { node_id, new_parent_id, index } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_to(
                            arvx_procedural::NodeId(node_id),
                            arvx_procedural::NodeId(new_parent_id),
                            index as usize,
                        ) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::DuplicateProceduralNode { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if let Some(new_id) = proc_geo.tree.duplicate(arvx_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                            self.selected_procedural_node = Some(new_id.0);
                        }
                    }
                }
            }

            EngineCommand::SetProceduralNodeCombinator { node_id, kind } => {
                // Local helper — returns true when a kind change was
                // actually applied. Early-returns via `?` / plain
                // `return None` keep the body flat and side-step the
                // `continue` footgun (there's no outer loop here,
                // this is a one-off match arm).
                fn swap_kind(
                    proc_geo: &mut crate::components::ProceduralGeometry,
                    id: arvx_procedural::NodeId,
                    kind: &str,
                ) -> bool {
                    let node = match proc_geo.tree.get_mut(id) {
                        Some(n) => n,
                        None => return false,
                    };
                    // Only swap between combinators; silently ignore on
                    // leaves (UI should hide the menu there anyway, but
                    // defend at the boundary).
                    let current_mc = match &node.kind {
                        arvx_procedural::NodeKind::Union { material_combine }
                        | arvx_procedural::NodeKind::Intersect { material_combine } => {
                            Some(*material_combine)
                        }
                        arvx_procedural::NodeKind::Subtract => None,
                        _ => return false, // leaf
                    };
                    let new_kind = match kind {
                        "Union" => arvx_procedural::NodeKind::Union {
                            material_combine: current_mc
                                .unwrap_or(arvx_procedural::MaterialCombine::Winner),
                        },
                        "Intersect" => arvx_procedural::NodeKind::Intersect {
                            material_combine: current_mc
                                .unwrap_or(arvx_procedural::MaterialCombine::Winner),
                        },
                        "Subtract" => arvx_procedural::NodeKind::Subtract,
                        _ => return false,
                    };
                    // No-op when the user re-picks the current kind —
                    // without this the version bump would force a rebake.
                    let same_kind = matches!(
                        (&node.kind, &new_kind),
                        (arvx_procedural::NodeKind::Union { .. }, arvx_procedural::NodeKind::Union { .. })
                            | (arvx_procedural::NodeKind::Intersect { .. }, arvx_procedural::NodeKind::Intersect { .. })
                            | (arvx_procedural::NodeKind::Subtract, arvx_procedural::NodeKind::Subtract)
                    );
                    if same_kind {
                        return false;
                    }
                    node.kind = new_kind;
                    true
                }

                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = arvx_procedural::NodeId(node_id);
                        if swap_kind(&mut proc_geo, id, &kind) {
                            proc_geo.tree.bump_version(id);
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::SetProceduralNodePosition { node_id, position } => {
                self.update_procedural_node_transform(node_id, |s, r, _| (s, r, position));
            }

            EngineCommand::SetProceduralNodeRotation { node_id, rotation_deg } => {
                let rot = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    rotation_deg.x.to_radians(),
                    rotation_deg.y.to_radians(),
                    rotation_deg.z.to_radians(),
                );
                self.update_procedural_node_transform(node_id, |s, _, t| (s, rot, t));
            }

            EngineCommand::SetProceduralNodeScale { node_id, scale } => {
                self.update_procedural_node_transform(node_id, |_, r, t| (scale, r, t));
            }

            EngineCommand::SetProceduralNodeParam { node_id, param_name, value } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = arvx_procedural::NodeId(node_id);
                        if apply_procedural_param(&mut proc_geo.tree, id, &param_name, &value) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::BakeProceduralEntity { entity_id } => {
                let entity = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e));
                if let Some(entity) = entity {
                    self.enqueue_bake(entity);
                }
            }

            EngineCommand::BakeAllDirtyProcedurals => {
                use crate::components::*;
                let dirty: Vec<hecs::Entity> = self
                    .world
                    .query::<&ProceduralGeometry>()
                    .iter()
                    .filter(|(_, p)| p.dirty)
                    .map(|(e, _)| e)
                    .collect();
                for entity in dirty {
                    self.enqueue_bake(entity);
                }
            }

            EngineCommand::ConvertProceduralToVoxel { entity_id } => {
                use crate::components::*;
                let Some(entity) = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e))
                else {
                    self.console.warn("Convert: entity not found".to_string());
                    return Ok(());
                };
                // Gate on a clean bake state — a pending/in-flight
                // bake means the voxels aren't what the user just
                // asked for.
                let can_convert = self
                    .world
                    .get::<&ProceduralGeometry>(entity)
                    .map(|pg| !pg.bake_in_flight && !pg.pending_bake && !pg.dirty)
                    .unwrap_or(false);
                if !can_convert {
                    self.console.warn(
                        "Convert: bake pending or in flight — let it settle first".to_string(),
                    );
                    return Ok(());
                }
                // Hard requirements for promoting the procedural to a
                // first-class asset: an open project (so we have an
                // assets/ directory to write to) and a saved scene
                // (so the bake worker has been writing the cache to a
                // known location). Without either, we can't produce
                // a persistent on-disk asset for the converted voxels.
                let Some(project_dir) = self.project_dir.clone() else {
                    self.console.warn(
                        "Convert: open or save a project first so the converted asset has somewhere to live.".to_string(),
                    );
                    return Ok(());
                };
                let name = self
                    .world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{entity:?}"));

                // Sanitize the entity name into a filename-safe slug:
                // lowercase, [a-z0-9_-] only, collapse runs of '_'.
                let mut slug: String = name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect();
                // Trim leading/trailing underscores; collapse runs.
                while slug.contains("__") {
                    slug = slug.replace("__", "_");
                }
                let slug = slug.trim_matches('_').to_string();
                let slug = if slug.is_empty() { "converted".to_string() } else { slug };

                // Drop converted assets under `assets/converted/` so
                // they're discoverable from the Models panel (which
                // recursively scans `assets/`) but visually grouped
                // separately from imported meshes and authored .arvx
                // files. The directory is created lazily.
                let target_dir = project_dir.join("assets").join("converted");
                if let Err(e) = std::fs::create_dir_all(&target_dir) {
                    self.console.error(format!(
                        "Convert: failed to create '{}': {e}",
                        target_dir.display(),
                    ));
                    return Ok(());
                }
                let mut target = target_dir.join(format!("{slug}.arvx"));
                let mut suffix = 1u32;
                while target.exists() {
                    target = target_dir.join(format!("{slug}_{suffix}.arvx"));
                    suffix += 1;
                }

                // Enqueue an async voxelize bake that writes the
                // resulting artifact to `target` on the worker thread
                // (via `cache_output_path`). The drain handler reads
                // `pending_conversions[entity]` on completion, acquires
                // the new .arvx as a regular asset, swaps the entity's
                // Renderable from the scene-pool spatial to the asset-
                // cache spatial, and removes `ProceduralGeometry`.
                let (tree_clone, base_voxel_size, root_scale) = {
                    let proc_geo = match self.world.get::<&ProceduralGeometry>(entity) {
                        Ok(pg) => pg,
                        Err(_) => return Ok(()),
                    };
                    let root_scale = proc_geo
                        .tree
                        .get(proc_geo.tree.root())
                        .map(|n| n.transform.to_scale_rotation_translation().0)
                        .unwrap_or(glam::Vec3::ONE);
                    (proc_geo.tree.clone(), proc_geo.voxel_size, root_scale)
                };
                let generation = if let Ok(mut pg) =
                    self.world.get::<&mut ProceduralGeometry>(entity)
                {
                    pg.bake_generation = pg.bake_generation.wrapping_add(1);
                    pg.dirty = false;
                    pg.pending_bake = false;
                    pg.bake_dirty_at = None;
                    pg.bake_in_flight = true;
                    pg.bake_generation
                } else {
                    return Ok(());
                };
                let prev_spatial = self
                    .world
                    .get::<&Renderable>(entity)
                    .ok()
                    .and_then(|r| r.spatial.clone())
                    .and_then(|g| g.into_octree());
                let (aabb, voxel_size) = super::procedural_ops::procedural_voxel_params(
                    &tree_clone,
                    base_voxel_size,
                );
                let instructions = arvx_procedural::flatten_tree(&tree_clone);
                let req = crate::bake_worker::BakeRequest {
                    entity,
                    generation,
                    input: crate::bake_worker::BakeInput::ProceduralVoxelize(instructions),
                    aabb,
                    voxel_size,
                    root_scale,
                    prev_spatial,
                    cache_output_path: Some(target.clone()),
                    generator_child: None,
                };
                if self.bake_worker.tx_request.send(req).is_err() {
                    self.console.warn("Convert: bake worker channel closed".to_string());
                    if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        pg.bake_in_flight = false;
                    }
                    return Ok(());
                }
                self.pending_conversions.insert(entity, target.clone());
                self.console.info(format!(
                    "Converting '{name}' to voxel asset → assets/converted/{} (voxelizing…)",
                    target.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                ));
            }
            EngineCommand::Paint { position, normal, radius, color, strength, mode } => {
                super::paint_ops::dispatch_paint(
                    self, position, normal, radius, color, strength, mode,
                );
            }

            EngineCommand::SetPaintActive { active, radius } => {
                self.paint_mode_active = active;
                self.paint_mode_radius = radius;
                // Cursor visualization is GPU-driven now — the
                // brush-state probe pass reads gbuf at the live
                // mouse pixel each frame and the shade pass gates the
                // ring on `paint_mode_active`. No CPU state to clear.
            }

            EngineCommand::PaintAtPixel {
                id, x, y, radius, color, strength, falloff, mode, material_id,
            } => {
                use super::state::{PaintPickSettings, PendingPick};
                // Stage a pick at (x, y); `paint_pick_settings` flags
                // this as a paint readback so the result bypasses
                // selection / drag-preview handling when it returns.
                // Matches the drag-preview pattern (cmd_scene.rs:291)
                // which also rides on the pending_pick pipeline.
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
                self.paint_pick_settings = Some(PaintPickSettings {
                    radius, color, strength, falloff, mode, material_id,
                });
            }

            EngineCommand::Sculpt { position, normal, radius, strength, mode } => {
                super::sculpt_ops::dispatch_sculpt(
                    self, position, normal, radius, strength, mode,
                );
            }

            EngineCommand::SetSculptActive { active, radius } => {
                self.sculpt_mode_active = active;
                self.sculpt_mode_radius = radius;
            }

            EngineCommand::SculptAtPixel {
                id, x, y, radius, falloff_curve, strength, stroke_seq, mode, material_id,
            } => {
                use super::state::{SculptPickSettings, PendingPick};
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
                self.sculpt_pick_settings = Some(SculptPickSettings {
                    radius, falloff_curve, strength, stroke_seq, mode, material_id,
                });
                // Stamp the wallclock so we can measure click-to-mutation
                // latency end-to-end (SculptAtPixel arrival → apply_sculpt_brush
                // call). Spans the pick round-trip + every sim/render
                // hop in between.
                self.sculpt_pending_at = Some(std::time::Instant::now());
            }

            other => return Err(other),
        }
        Ok(())
    }
}
