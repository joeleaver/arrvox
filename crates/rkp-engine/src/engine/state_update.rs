//! Per-tick `StateUpdate` assembly for the editor.
//!
//! Reflects the live ECS + project state into the `StateUpdate` that
//! `tick_loop` ships to the editor via the state-callback. Also hosts
//! the inspector + procedural snapshot builders that pack selected
//! entity details for the properties panel, plus the material-usage
//! walk that feeds `RemapMaterial`.

use super::model_scan::{collect_internal_attr_slots, collect_leaf_slots};
use super::state::EngineState;
use crate::snapshot::StateUpdate;
use std::time::Duration;

impl EngineState {
    pub(crate) fn build_inspector_snapshot(&mut self) -> Option<crate::inspector::InspectorSnapshot> {
        let selected = self.selected_entity?;
        if !self.world.contains(selected) {
            return None;
        }

        let name = self.world.get::<&crate::components::EditorMetadata>(selected)
            .map(|m| m.name.clone())
            .unwrap_or_default();

        use crate::inspector::*;

        // For procedural entities, the Transform.scale slider is a
        // proxy for Root.transform.scale (see
        // `redirect_transform_scale_to_root`). Pull the displayed value
        // from the tree so the slider reflects what's actually baked.
        let proc_root_scale: Option<[f32; 3]> = self
            .world
            .get::<&crate::components::ProceduralGeometry>(selected)
            .ok()
            .and_then(|pg| {
                let root = pg.tree.root();
                pg.tree.get(root).map(|node| {
                    node.transform
                        .to_scale_rotation_translation()
                        .0
                        .to_array()
                })
            });

        // Build component snapshots from the registry.
        let mut components = Vec::new();
        for entry in self.registry.components_on(&self.world, selected) {
            let fields: Vec<FieldSnapshot> = entry.meta.iter().map(|meta| {
                let mut value = (entry.get_field)(&self.world, selected, meta.name)
                    .unwrap_or(FieldValue::String("<error>".into()));
                if entry.name == "Transform"
                    && meta.name == "scale"
                    && let Some(s) = proc_root_scale
                {
                    value = FieldValue::Vec3(s);
                }
                FieldSnapshot {
                    name: meta.name.to_string(),
                    field_type: meta.field_type,
                    value,
                    range: meta.range,
                    transient: meta.transient,
                    enum_options: meta.enum_options
                        .map(|opts| opts.iter().map(|(v, l)| (v.to_string(), l.to_string())).collect())
                        .unwrap_or_default(),
                    scrub: meta.scrub,
                    ..Default::default()
                }
            }).collect();
            components.push(ComponentSnapshot {
                name: entry.name.to_string(),
                fields,
                removable: !entry.mandatory,
            });
        }

        // Extract position/rotation/scale from Transform if present.
        let transform = self.world.get::<&crate::components::Transform>(selected).ok();
        let pos = transform.as_ref().map(|t| t.position.to_array()).unwrap_or([0.0; 3]);
        let rot = transform.as_ref().map(|t| t.rotation.to_array()).unwrap_or([0.0; 3]);
        let scl = transform.as_ref().map(|t| t.scale.to_array()).unwrap_or([1.0; 3]);

        // Count per-material voxel usage if entity has spatial data.
        // The walk is O(voxels) so results are cached and reused
        // across ticks; only the selection changing or a geometry
        // epoch bump (bake / remap / load) invalidates it.
        let current_epoch = self
            .geometry_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);
        let material_usage = match &self.cached_material_usage {
            Some((e, epoch, usage)) if *e == selected && *epoch == current_epoch => {
                usage.clone()
            }
            _ => {
                let fresh = self.count_material_usage(selected);
                self.cached_material_usage = Some((selected, current_epoch, fresh.clone()));
                fresh
            }
        };

        // Skeleton sidecar — clip + bone metadata for the dedicated
        // animation panel. Skipped when the entity has no skeleton.
        let skeleton = self.world.get::<&crate::components::Skeleton>(selected).ok()
            .map(|skel| {
                let bone_names: Vec<String> = skel.asset.skeleton.bones.iter().map(|b| b.name.clone()).collect();
                let bone_parents: Vec<i32> = skel.asset.skeleton.hierarchy.clone();
                let clips: Vec<ClipInfo> = skel.asset.clips.iter().map(|c| ClipInfo {
                    name: c.name.clone(),
                    duration: c.duration,
                }).collect();
                SkeletonInspector {
                    path: skel.path.to_string_lossy().into_owned(),
                    bone_names,
                    bone_parents,
                    clips,
                }
            });

        Some(InspectorSnapshot {
            entity_name: name,
            entity_id: format!("{}", self.get_entity_uuid(selected).as_simple()),
            position: pos,
            rotation: rot,
            scale: scl,
            components,
            material_usage,
            skeleton,
        })
    }

    /// Count per-material voxel usage for an entity's octree.
    ///
    /// When the entity has a `Renderable` but no voxel data yet (e.g. an
    /// unbaked procedural primitive), a synthetic fallback row is
    /// returned carrying `Renderable.material_id` so the UI always has
    /// at least one drop slot to assign against.
    pub(crate) fn count_material_usage(&self, entity: hecs::Entity) -> Vec<crate::inspector::MaterialUsage> {
        let renderable = match self.world.get::<&crate::components::Renderable>(entity) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let fallback_row = || crate::inspector::MaterialUsage {
            material_id: renderable.material_id,
            voxel_count: 0,
            is_fallback: true,
        };

        let spatial = match &renderable.spatial {
            Some(s) => s,
            None => return vec![fallback_row()],
        };

        // Collect leaf voxel slots from the packed octree buffer.
        // Branch offsets in the packed buffer are ABSOLUTE, so we traverse
        // the full buffer starting at root_offset (not a sub-slice).
        // Brick-terminated subtrees expand via the brick_pool.
        let sm = self.scene_mgr.lock().unwrap();
        let all_nodes = sm.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, &sm.brick_pool, spatial.root_offset as usize, &mut leaf_slots);

        // Count material IDs across all leaf slots. Every leaf is a surface
        // voxel now — no opacity gate.
        let pool_size = sm.leaf_attr_pool.allocated_count();
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        for slot in leaf_slots {
            if slot >= pool_size {
                continue; // stale or invalid slot — skip
            }
            let attr = sm.leaf_attr_pool.get(slot);
            *counts.entry(attr.material_primary).or_insert(0) += 1;
        }

        if counts.is_empty() {
            return vec![fallback_row()];
        }

        // Sort by voxel count descending.
        let mut usage: Vec<crate::inspector::MaterialUsage> = counts
            .into_iter()
            .map(|(material_id, voxel_count)| crate::inspector::MaterialUsage {
                material_id,
                voxel_count,
                is_fallback: false,
            })
            .collect();
        usage.sort_by(|a, b| b.voxel_count.cmp(&a.voxel_count));
        usage
    }

    /// Compose a new `(from, to)` remap into the entity's persistent
    /// override list, in-place. Invariant: each original material
    /// appears at most once in the list, and identity entries
    /// (orig == current) are dropped so a "remap back to original"
    /// doesn't leave a no-op pair lying around.
    ///
    /// Existing pairs whose current value matches `from` are updated
    /// (their voxels just moved again). If no existing pair tracks
    /// `from` as an original yet, a fresh `(from, to)` pair is
    /// appended — this captures the case where untouched original-
    /// material voxels were hit by the new remap.
    pub(crate) fn compose_material_override(
        overrides: &mut Vec<(u16, u16)>,
        from: u16,
        to: u16,
    ) {
        for (_orig, cur) in overrides.iter_mut() {
            if *cur == from {
                *cur = to;
            }
        }
        let already_tracked = overrides.iter().any(|(orig, _)| *orig == from);
        if !already_tracked && from != to {
            overrides.push((from, to));
        }
        overrides.retain(|(o, c)| o != c);
    }

    /// Remap all voxels on an entity from one material to another.
    /// Returns the number of voxels changed.
    pub(crate) fn remap_entity_material(
        &mut self,
        entity: hecs::Entity,
        from_material: u16,
        to_material: u16,
    ) -> u32 {
        let renderable = match self.world.get::<&crate::components::Renderable>(entity) {
            Ok(r) => r.clone(),
            Err(_) => return 0,
        };
        let spatial = match &renderable.spatial {
            Some(s) => s.clone(),
            None => return 0,
        };

        // Collect every leaf_attr slot reachable from this entity:
        //   - per-voxel slots inside LEAFs + BRICKs (real geometry),
        //   - prefilter slots at BRANCH nodes (LOD aggregates).
        // Both kinds hold a material_primary that must stay in sync;
        // missing the LOD set here is what made close-up material
        // changes "disappear" once distance kicked in the LOD cutoff.
        let mut sm = self.scene_mgr.lock().unwrap();
        let mut leaf_slots = Vec::new();
        {
            let all_nodes = sm.octree.data();
            let internal_attrs = sm.octree.internal_attrs_data();
            collect_leaf_slots(all_nodes, &sm.brick_pool, spatial.root_offset as usize, &mut leaf_slots);
            collect_internal_attr_slots(all_nodes, internal_attrs, spatial.root_offset as usize, &mut leaf_slots);
        }

        let pool_size = sm.leaf_attr_pool.allocated_count();
        let mut count = 0u32;
        for slot in leaf_slots {
            if slot >= pool_size { continue; }
            let attr = sm.leaf_attr_pool.get(slot);
            let primary = attr.material_primary;
            let secondary = attr.material_secondary();
            let mut changed = false;

            if primary == from_material {
                let m = sm.leaf_attr_pool.get_mut(slot);
                m.material_primary = to_material;
                changed = true;
            }
            if secondary == from_material {
                // Re-pack secondary + blend, since both share material_secondary_blend.
                let attr = *sm.leaf_attr_pool.get(slot);
                let blend = attr.blend_weight();
                let m = sm.leaf_attr_pool.get_mut(slot);
                let secondary_bits = (to_material & 0x0FFF) as u16;
                let blend_bits = ((blend as u16) & 0x0F) << 12;
                m.material_secondary_blend = secondary_bits | blend_bits;
                changed = true;
            }
            if changed {
                count += 1;
            }
        }
        if count > 0 {
            // Bump so the render thread re-uploads leaf_attr_pool on its
            // next iteration; the plain `geometry_dirty` flag only
            // schedules collider rebuilds, not GPU upload.
            sm.bump_geometry_epoch();
        }
        count
    }

    pub(crate) fn build_state_update(&mut self, _sim_frame_time: Duration) -> StateUpdate {
        // FPS = render thread's actual iteration rate, EMA-smoothed.
        // The previous formula was `1 / sim_cpu_work_time`, which
        // measured sim CPU headroom rather than what's on screen.
        // After the sim/render thread split they're independent
        // numbers — sim might be at 600 Hz "could do" while render
        // is paced to 60 Hz. The user-visible FPS is the render rate.
        let fps = self.render_hz_ema;

        let objects = if self.scene_dirty {
            self.scene_dirty = false;
            // Sort by `entity_tree_order` — user-arrangeable (via a
            // future drag-reorder command) and persisted in the scene
            // file, so the arrangement survives save/reload. Entities
            // missing from the map (transient edge cases) fall back
            // to `Entity::to_bits()` as a tiebreaker, which preserves
            // spawn order for them.
            //
            // The editor's scene-tree panel groups by `parent_id`
            // after the fact, so children of the same parent end up
            // displayed in their shared TreeOrder order naturally.
            let mut ordered: Vec<hecs::Entity> = self.world
                .query::<&crate::components::EditorMetadata>()
                .iter()
                .map(|(entity, _)| entity)
                .collect();
            ordered.sort_by(|a, b| {
                let ka = self.entity_tree_order.get(a).copied();
                let kb = self.entity_tree_order.get(b).copied();
                match (ka, kb) {
                    (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.to_bits().cmp(&b.to_bits()),
                }
            });

            let mut objs = Vec::with_capacity(ordered.len());
            for entity in ordered {
                let Ok(meta) = self.world.get::<&crate::components::EditorMetadata>(entity) else {
                    continue;
                };
                let is_light = self.world.get::<&crate::components::PointLight>(entity).is_ok()
                    || self.world.get::<&crate::components::SpotLight>(entity).is_ok();
                let is_camera = self.world.get::<&crate::components::Camera>(entity).is_ok();
                let is_procedural = self
                    .world
                    .get::<&crate::components::ProceduralGeometry>(entity)
                    .is_ok();
                let parent_id = self.world.get::<&crate::components::Parent>(entity)
                    .ok()
                    .map(|p| p.parent_id);
                objs.push(crate::snapshot::SceneObjectInfo {
                    id: self.get_entity_uuid(entity),
                    name: meta.name.clone(),
                    parent_id,
                    tree_order: self.entity_tree_order.get(&entity).copied().unwrap_or(0.0),
                    is_camera,
                    is_light,
                    is_procedural,
                });
            }
            Some(objs)
        } else {
            None
        };

        let project = if self.project_dirty {
            self.project_dirty = false;
            Some(self.project_loaded)
        } else {
            None
        };

        let project_name = if project.is_some() {
            Some(self.project_name.clone())
        } else {
            None
        };

        // Ride the same `project_dirty` flag — project_dir changes
        // exactly when project_loaded / project_name do.
        let project_dir = if project.is_some() {
            Some(
                self.project_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
            )
        } else {
            None
        };

        let models = if self.models_dirty {
            self.models_dirty = false;
            Some(self.available_models.clone())
        } else {
            None
        };
        let generators = if self.generators_dirty {
            self.generators_dirty = false;
            let mut names: Vec<String> = self
                .generator_system
                .registry()
                .names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            names.sort();
            Some(names)
        } else {
            None
        };
        let generator_presets = if self.generator_presets_dirty {
            self.generator_presets_dirty = false;
            Some(
                self.available_generator_presets
                    .iter()
                    .map(|p| crate::snapshot::GeneratorPresetEntry {
                        path: p.path.to_string_lossy().into_owned(),
                        display_name: p.display_name.clone(),
                        generator_name: p.generator_name.clone(),
                    })
                    .collect(),
            )
        } else {
            None
        };
        let importing = if self.importing_dirty {
            self.importing_dirty = false;
            Some(self.importing_sources.iter().cloned().collect())
        } else {
            None
        };
        // Send live progress every tick while any import is in flight.
        // Outside an active import this is `None` so the UI skips
        // re-rendering the panel.
        let import_progress = if self.importing_progress.is_empty() {
            None
        } else {
            Some(self.importing_progress.values().cloned().collect())
        };
        let editor_layout = if self.editor_layout_pending {
            self.editor_layout_pending = false;
            Some(self.editor_layout_json.clone())
        } else {
            None
        };

        // Inspector + procedural: send only on change. Both rebuild every
        // tick (cheap) but the panel re-render they trigger on the editor
        // thread is not — sending an identical snapshot 60Hz used to chunk
        // the UI when physics drove a selected RigidBody's Transform.
        let new_inspector = self.build_inspector_snapshot();
        let inspector_update = if new_inspector != self.prev_inspector {
            self.prev_inspector = new_inspector.clone();
            Some(new_inspector)
        } else {
            None
        };
        let new_procedural = self.build_procedural_snapshot();
        let procedural_update = if new_procedural != self.prev_procedural {
            self.prev_procedural = new_procedural.clone();
            Some(new_procedural)
        } else {
            None
        };

        StateUpdate {
            fps,
            delivered_fps: self.delivered_hz_ema,
            tick_hz: self.tick_hz_ema,
            physics_hz: self.physics_hz_ema,
            gpu_object_count: self.gpu_instances.len() as u32,
            camera_position: self.camera.position,
            play_mode: self.play_state.is_some(),
            selected_entity: self.selected_entity.map(|e| self.get_entity_uuid(e)),
            objects,
            project_loaded: project,
            project_name,
            project_dir,
            available_models: models,
            available_generators: generators,
            available_generator_presets: generator_presets,
            importing_models: importing,
            import_progress,
            editor_layout,
            inspector: inspector_update,
            recent_projects: if self.frame_index == 1 {
                Some(crate::recent_projects::load_recent())
            } else {
                None
            },
            available_components: self.selected_entity.map(|entity| {
                self.registry.available_for(&self.world, entity)
                    .iter()
                    .map(|e| e.name.to_string())
                    .collect()
            }),
            materials: if self.material_lib.is_ui_dirty() {
                self.material_lib.clear_ui_dirty();
                Some(self.material_lib.build_info())
            } else {
                None
            },
            user_shaders: {
                let cur_hash = self.user_shader_registry.source_hash();
                if self.user_shader_first_send || cur_hash != self.prev_user_shader_hash {
                    self.user_shader_first_send = false;
                    self.prev_user_shader_hash = cur_hash;
                    Some(self.user_shader_registry.shader_infos())
                } else {
                    None
                }
            },
            selected_material: self.selected_material,
            selected_model: self.selected_model.clone(),
            environment: {
                // Always build, diff-suppress. The old `environment_ui_dirty`
                // gate explicitly avoided echoing slider edits back to the
                // editor (it would have remounted the form mid-drag);
                // diff-suppression makes that hack unnecessary because user
                // edits round-trip back as exact-match no-ops here. With the
                // env panel reading per-field Memos against `store.environment`,
                // these pushes also don't remount anything — only the changed
                // field's DOM updates.
                let _ = self.environment_ui_dirty; // legacy flag, no longer gates
                self.environment_ui_dirty = false;
                if Some(&self.environment) != self.prev_environment.as_ref() {
                    self.prev_environment = Some(self.environment.clone());
                    Some(self.environment.clone())
                } else {
                    None
                }
            },
            procedural: procedural_update,
            console_entries: self.console.drain_new(),
            // Pull the most recent sample whose render-thread data
            // has been stitched in. Render publishes 1-2 frames
            // behind sim — `latest()` would return a still-empty
            // sample sim just pushed and the panel would show no GPU
            // / frame-time data. Falls back to `latest()` during the
            // first few frames before any render results land.
            profiling: self
                .profiling
                .latest_with_render_data()
                .or_else(|| self.profiling.latest())
                .cloned(),
        }
    }

    pub(crate) fn build_procedural_snapshot(&self) -> Option<crate::procedural_snapshot::ProceduralSnapshot> {
        let entity = self.selected_entity?;
        let proc_geo = self.world.get::<&crate::components::ProceduralGeometry>(entity).ok()?;
        let uuid = self.get_entity_uuid(entity);
        let vs = proc_geo.voxel_size;
        // Renderable carries the post-bake voxel count. Procedurals
        // always have one paired with their ProceduralGeometry, but
        // defend with 0 rather than panic if something gets out of
        // sync mid-edit.
        let voxel_count = self
            .world
            .get::<&crate::components::Renderable>(entity)
            .map(|r| r.voxel_count)
            .unwrap_or(0);
        Some(crate::procedural_snapshot::build_procedural_snapshot(
            uuid,
            &proc_geo,
            self.selected_procedural_node,
            vs,
            voxel_count,
        ))
    }
}
