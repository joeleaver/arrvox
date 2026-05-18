//! Generator-object operations.
//!
//! Scans entities with `GeneratorState`, submits runs to the
//! `BakeWorker`'s generator channel, updates the ECS as results land,
//! and spawns generator presets / dropped generator entities. Works
//! alongside `procedural_ops` — generator output IS a procedural tree
//! (parent entity) plus any emitted child entities.

use super::procedural_params::json_to_field_value;
use super::state::EngineState;

impl EngineState {
    /// Pump the generator system once per frame.
    ///
    /// Surfaces notable lifecycle events (submit / complete / fail /
    /// cancel) to the console so the user sees them there even when
    /// the generator panel is hidden. Per-entity status updates land
    /// on the ECS via `tick()` itself.
    pub(crate) fn tick_generators(&mut self) {
        // Compute the per-session bake-cache directory for emitted
        // children: `{scene_dir}/{scene_stem}.bakes/`. Same directory
        // procedurals use for their per-entity caches — generator
        // children get filenames keyed by `(parent_uuid, slot_key)` so
        // the two never collide. `None` here means the scene is
        // unsaved; persistent emits run but won't write a cache, so a
        // save+reload of an unsaved-then-saved scene will trigger a
        // one-time regen until the first bake completes after save.
        let child_cache_dir = self
            .scene_path
            .as_ref()
            .and_then(|p| {
                let parent = p.parent()?;
                let stem = p.file_stem()?;
                Some(parent.join(format!("{}.bakes", stem.to_string_lossy())))
            });
        if let Some(ref dir) = child_cache_dir {
            // Best-effort create — same lazy create pattern used by
            // procedural caches.
            let _ = std::fs::create_dir_all(dir);
        }
        let events = self.generator_system.tick(
            &mut self.world,
            &self.registry,
            &self.entity_uuids,
            child_cache_dir.as_deref(),
        );
        for ev in events {
            use crate::generator::GeneratorEvent;
            match ev {
                GeneratorEvent::WillResubmit { entity, name: _ } => {
                    // Reset the per-generation slot-key tracker before
                    // the new run's emits land. The spawn-or-update
                    // path repopulates it as each child arrives, and
                    // the Completed handler diffs the resulting set
                    // against the existing children to despawn slots
                    // the generator no longer emits.
                    self.pending_generator_slot_keys.remove(&entity);
                }
                GeneratorEvent::Submitted { entity, name } => {
                    eprintln!("[gen] submit entity={entity:?} name={name}");
                }
                GeneratorEvent::Completed { entity, name } => {
                    eprintln!("[gen] completed entity={entity:?} name={name}");
                    // Orphan cleanup: delete persistent children whose
                    // slot_key wasn't emitted in this generation.
                    let seen = self
                        .pending_generator_slot_keys
                        .remove(&entity)
                        .unwrap_or_default();
                    let parent_uuid = self.entity_uuids.get(&entity).copied();
                    let orphans: Vec<hecs::Entity> = if let Some(pu) = parent_uuid {
                        self.world
                            .query::<&crate::generator::GeneratorOwned>()
                            .iter()
                            .filter(|(_, owned)| {
                                owned.parent_uuid == pu && !seen.contains(&owned.slot_key)
                            })
                            .map(|(e, _)| e)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for child in orphans {
                        self.delete_entity(child);
                    }
                    // Multi-entity delete loop; delete_entity already
                    // marks each removed entity narrowly.
                    self.scene_dirty.mark_all();
                }
                GeneratorEvent::Failed { entity, name, error } => {
                    self.console.error(format!(
                        "Generator '{name}' on {entity:?} failed: {error}"
                    ));
                }
                GeneratorEvent::Cancelled { entity, name } => {
                    eprintln!("[gen] cancelled entity={entity:?} name={name}");
                }
            }
        }
    }

    /// Spawn or update a child entity for a generator's emitted bake.
    ///
    /// * Anonymous (`slot_key.is_none()`): always spawn a brand-new
    ///   entity. The previous generation's anonymous children were
    ///   blown away in `tick_generators`'s `WillResubmit` handler.
    /// * Persistent (`slot_key.is_some()`): look for an existing
    ///   `(parent, slot_key)` match. If found, replace its Transform +
    ///   Renderable.spatial in place — preserves any user-attached
    ///   components (lights, scripts, colliders). If not, spawn a new
    ///   entity tagged with the slot key.
    ///
    /// Either way, the world transform is composed against the
    /// parent's *current* transform (not a stale snapshot from the
    /// worker), so dragging the generator between emit and spawn
    /// still places the child correctly.
    pub(crate) fn spawn_or_update_generated_child(
        &mut self,
        generator_entity: hecs::Entity,
        local_transform: crate::components::Transform,
        generation: u64,
        slot_key: String,
        spatial: crate::components::SpatialData,
        voxel_count: u32,
        name_hint: Option<String>,
    ) {
        use crate::components::*;
        if !self.world.contains(generator_entity) {
            return;
        }

        let parent_transform = self
            .world
            .get::<&Transform>(generator_entity)
            .map(|t| (*t).clone())
            .unwrap_or_default();
        let world_transform = compose_generator_transforms(
            &parent_transform,
            &local_transform,
        );

        // Track that this slot was emitted this generation, so the
        // Completed handler knows which children survived (vs. which
        // to despawn as orphans because the generator stopped emitting
        // them).
        self.pending_generator_slot_keys
            .entry(generator_entity)
            .or_default()
            .insert(slot_key.clone());

        if let Some(existing) = self.find_persistent_child(generator_entity, &slot_key) {
            // Reuse: free the old geometry, swap the Renderable's
            // spatial in place, refresh transform + generation. Other
            // components stay → user-attached lights / scripts survive
            // across regens.
            self.release_renderable_geometry(existing);
            if let Ok(mut t) = self.world.get::<&mut Transform>(existing) {
                *t = world_transform;
            }
            if let Ok(mut r) = self.world.get::<&mut Renderable>(existing) {
                r.spatial = Some(crate::components::RenderGeometry::Octree(spatial));
                r.voxel_count = voxel_count;
                // Reload-from-cache populates `asset_handle` (children
                // round-trip through the asset cache on load). The
                // fresh bake hands us a raw scene-pool allocation,
                // not an asset — clear the stale handle so the NEXT
                // regen's release_renderable_geometry takes the
                // deallocate-spatial path instead of releasing an
                // asset that was already released up above.
                r.asset_handle = None;
            }
            if let Ok(mut owned) =
                self.world.get::<&mut crate::generator::GeneratorOwned>(existing)
            {
                owned.generation = generation;
            }
            self.scene_dirty.mark_entity(existing);
            self.geometry_dirty.mark_all();
            self.gpu_objects_dirty.mark_all();
            eprintln!(
                "[gen] reused child entity={existing:?} parent={generator_entity:?} \
                 slot='{slot_key}' voxels={voxel_count} gen={generation}"
            );
            return;
        }

        let base_name = name_hint.unwrap_or_else(|| {
            self.world
                .get::<&EditorMetadata>(generator_entity)
                .map(|m| format!("{}.child", m.name))
                .unwrap_or_else(|_| "child".into())
        });
        let name = self.unique_name(&base_name);
        let renderable = Renderable {
            asset_path: None,
            primitive: None,
            material_id: 0,
            voxel_count,
            spatial: Some(crate::components::RenderGeometry::Octree(spatial)),
            ..Default::default()
        };
        let parent_uuid = self.entity_uuids.get(&generator_entity).copied();
        // GeneratorOwned needs the parent's UUID; without it the marker
        // can't survive a save/load (queries match by UUID, not Entity).
        // If the generator entity has somehow lost its UUID, fail loud
        // instead of silently spawning an orphan.
        let owned_parent_uuid = match parent_uuid {
            Some(u) => u,
            None => {
                eprintln!(
                    "[gen] spawn_or_update_generated_child: generator entity {generator_entity:?} has no UUID; dropping child",
                );
                return;
            }
        };
        let child = self.world.spawn((
            world_transform,
            EditorMetadata { name: name.clone() },
            renderable,
            crate::generator::GeneratorOwned {
                parent_uuid: owned_parent_uuid,
                generation,
                slot_key: slot_key.clone(),
            },
        ));
        // Attach Parent for the scene tree. The transform stored on
        // `child` is already absolute world — we don't compose on GPU
        // build — but Parent makes the scene_tree panel show the child
        // indented under its generator.
        if let Some(uuid) = parent_uuid {
            let _ = self.world.insert_one(
                child,
                crate::components::Parent { parent_id: uuid },
            );
        }
        self.assign_entity_uuid(child);
        self.scene_dirty.mark_entity(child);
        self.geometry_dirty.mark_all();
        self.gpu_objects_dirty.mark_all();
        eprintln!(
            "[gen] spawned child entity={child:?} parent={generator_entity:?} \
             name='{name}' slot='{slot_key}' voxels={voxel_count} gen={generation}"
        );
    }

    /// Find the existing persistent child of `parent` matching
    /// `slot_key`, if any.
    pub(crate) fn find_persistent_child(
        &self,
        parent: hecs::Entity,
        slot_key: &str,
    ) -> Option<hecs::Entity> {
        let parent_uuid = self.entity_uuids.get(&parent).copied()?;
        self.world
            .query::<&crate::generator::GeneratorOwned>()
            .iter()
            .find(|(_, owned)| {
                owned.parent_uuid == parent_uuid && owned.slot_key == slot_key
            })
            .map(|(e, _)| e)
    }

    /// Release the GPU pool slots held by `entity`'s `Renderable`,
    /// without despawning the entity. Used by the persistent-child
    /// reuse path: the entity stays, only the geometry is swapped.
    /// Mirrors the asset/spatial branches of `delete_entity`.
    pub(crate) fn release_renderable_geometry(&mut self, entity: hecs::Entity) {
        use crate::components::*;
        if let Ok(renderable) = self.world.get::<&Renderable>(entity) {
            if let Some(handle) = renderable.asset_handle {
                drop(renderable);
                self.scene_mgr.lock().unwrap().release_asset(handle);
            } else if let Some(spatial) = renderable.spatial.as_ref().and_then(|g| g.as_octree()) {
                let handle = arvx_core::OctreeHandle {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let slot_start = spatial.voxel_slot_start;
                let slot_count = spatial.voxel_slot_count;
                let brick_ids = spatial.brick_ids.clone();
                drop(renderable);
                self.scene_mgr.lock().unwrap().deallocate_geometry(
                    &handle, slot_start, slot_count, &brick_ids,
                );
            }
        }
        // Wipe the spatial so the renderer doesn't see stale data
        // between this point and the caller's swap-in.
        if let Ok(mut r) = self.world.get::<&mut Renderable>(entity) {
            r.spatial = None;
        }
    }

    /// Load a `.arvxgen` preset and spawn the generator entity it
    /// describes. Param overrides flow through the component
    /// registry's typed `set_field` so partial preset files work —
    /// any field absent from `params` keeps its `Default` value.
    /// Spawn a bare generator (no preset overrides). `pos = None` uses
    /// the click-path default of 3m in front of the camera; `Some(p)`
    /// places the generator's origin at `p` (drop-on-geometry path).
    pub(crate) fn spawn_generator(&mut self, generator_name: &str, pos: Option<glam::Vec3>) {
        let _ = self.spawn_generator_ex(generator_name, pos, true);
    }

    /// Spawn helper that returns the entity so drag-preview can track
    /// it. `verbose` gates the console log — drag-preview spawns are
    /// transient and shouldn't spam the log on every cancel/recreate.
    pub(crate) fn spawn_generator_ex(
        &mut self,
        generator_name: &str,
        pos: Option<glam::Vec3>,
        verbose: bool,
    ) -> Option<hecs::Entity> {
        use crate::components::*;
        let Some(entry) = self.generator_system.registry().get(generator_name) else {
            self.console.error(format!(
                "Unknown generator '{generator_name}' — not registered in gameplay dylib"
            ));
            return None;
        };
        let name = self.unique_name(generator_name);
        let mut transform = Transform::default();
        transform.position = pos.unwrap_or_else(|| {
            self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
        });
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: name.clone() },
            crate::generator::GeneratorState::new(generator_name),
        ));
        (entry.insert_default_params)(&mut self.world, entity);
        self.assign_entity_uuid(entity);
        self.scene_dirty.mark_entity(entity);
        if verbose {
            self.console.info(format!("Spawned generator '{name}'"));
        }
        Some(entity)
    }

    pub(crate) fn spawn_generator_preset(&mut self, path: &str, pos: Option<glam::Vec3>) {
        let _ = self.spawn_generator_preset_ex(path, pos, true);
    }

    pub(crate) fn spawn_generator_preset_ex(
        &mut self,
        path: &str,
        pos: Option<glam::Vec3>,
        verbose: bool,
    ) -> Option<hecs::Entity> {
        use crate::components::*;
        let preset_path = std::path::PathBuf::from(path);
        let cfg = match crate::generator::GeneratorAssetConfig::load(&preset_path) {
            Ok(c) => c,
            Err(e) => {
                self.console.error(format!("Load preset failed: {e}"));
                return None;
            }
        };
        let Some(entry) = self.generator_system.registry().get(&cfg.generator) else {
            self.console.error(format!(
                "Preset '{}' targets unknown generator '{}'",
                cfg.name, cfg.generator,
            ));
            return None;
        };
        let display_name = self.unique_name(&cfg.name);
        let mut transform = Transform::default();
        transform.position = pos.unwrap_or_else(|| {
            self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
        });
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: display_name.clone() },
            crate::generator::GeneratorState::new(&cfg.generator),
        ));
        (entry.insert_default_params)(&mut self.world, entity);
        if !cfg.params.is_empty() {
            if let Some(comp_entry) = self.registry.get(entry.param_component_name) {
                for (field_name, value) in &cfg.params {
                    match json_to_field_value(value, field_name, comp_entry) {
                        Ok(fv) => {
                            if let Err(e) = (comp_entry.set_field)(
                                &mut self.world, entity, field_name, fv,
                            ) {
                                self.console.warn(format!(
                                    "Preset '{}': set {field_name} failed: {e}",
                                    display_name,
                                ));
                            }
                        }
                        Err(e) => {
                            self.console.warn(format!(
                                "Preset '{}': skip {field_name}: {e}",
                                display_name,
                            ));
                        }
                    }
                }
            } else {
                self.console.warn(format!(
                    "Preset '{}': param component '{}' not registered",
                    display_name, entry.param_component_name,
                ));
            }
        }
        self.assign_entity_uuid(entity);
        self.scene_dirty.mark_entity(entity);
        if verbose {
            self.console.info(format!(
                "Spawned preset '{}' ({}) with {} override(s)",
                display_name, cfg.generator, cfg.params.len(),
            ));
        }
        Some(entity)
    }

    pub(crate) fn scan_generator_presets(&mut self) {
        self.available_generator_presets.clear();
        let Some(ref project_dir) = self.project_dir else { return };
        let presets_dir = project_dir.join("assets/generators");
        if !presets_dir.exists() {
            self.generator_presets_dirty = true;
            return;
        }
        let Ok(entries) = std::fs::read_dir(&presets_dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "arvxgen").unwrap_or(false) {
                match crate::generator::GeneratorAssetConfig::load(&path) {
                    Ok(cfg) => {
                        self.available_generator_presets.push(
                            crate::generator::GeneratorPresetInfo {
                                path: path.clone(),
                                display_name: cfg.name,
                                generator_name: cfg.generator,
                            },
                        );
                    }
                    Err(e) => {
                        self.console.warn(format!(
                            "Skipping malformed preset {}: {e}",
                            path.display(),
                        ));
                    }
                }
            }
        }
        self.available_generator_presets
            .sort_by(|a, b| a.display_name.cmp(&b.display_name));
        self.generator_presets_dirty = true;
        eprintln!(
            "[ArvxEngine] scanned {} generator presets",
            self.available_generator_presets.len(),
        );
    }
}

/// Compose a generator's parent transform with a child's local transform
/// to produce the child's absolute world transform. `Transform.rotation`
/// is Euler XYZ degrees (engine convention).
pub(crate) fn compose_generator_transforms(
    parent: &crate::components::Transform,
    child: &crate::components::Transform,
) -> crate::components::Transform {
    let parent_rot = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        parent.rotation.x.to_radians(),
        parent.rotation.y.to_radians(),
        parent.rotation.z.to_radians(),
    );
    let child_rot = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        child.rotation.x.to_radians(),
        child.rotation.y.to_radians(),
        child.rotation.z.to_radians(),
    );
    let world_rot = parent_rot * child_rot;
    let (ex, ey, ez) = world_rot.to_euler(glam::EulerRot::XYZ);
    let scaled_child_pos = parent_rot * (parent.scale * child.position);
    crate::components::Transform {
        position: parent.position + scaled_child_pos,
        rotation: glam::Vec3::new(ex.to_degrees(), ey.to_degrees(), ez.to_degrees()),
        scale: parent.scale * child.scale,
    }
}
