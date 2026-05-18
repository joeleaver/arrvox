//! Scene / project file load + save.
//!
//! Reads `.arvxscene` from disk into the ECS world, writes the current
//! world back to `.arvxscene`, and writes the `.arvxproject` sidecar. The
//! actual serde types live in `crate::scene_io`; these methods are the
//! EngineState-side orchestration that walks the world.

use super::state::EngineState;
use super::model_scan::spatial_from_handle;

impl EngineState {
    pub(crate) fn load_scene_from_file(&mut self, path: &std::path::Path) {
        // Resolve the scene directory from the passed-in path rather
        // than `self.scene_path` — we used to rely on the latter being
        // set before load, but order-of-operations bugs there silently
        // broke procedural bake-cache restoration. The path we're
        // loading is authoritative.
        let scene_dir = path.parent().map(|p| p.to_path_buf());
        match crate::scene_io::load_scene(path) {
            Ok(scene) => {
                // Restore camera.
                self.camera.position = glam::Vec3::from_array(scene.camera.position);
                self.camera.yaw = scene.camera.yaw;
                self.camera.pitch = scene.camera.pitch;
                self.camera.fov = scene.camera.fov;
                self.sync_main_viewport_from_legacy_camera();

                // Restore environment.
                if let Some(ref env) = scene.environment {
                    self.environment = env.clone();
                    self.environment_dirty = true;
                    self.environment_ui_dirty = true;
                }

                // Load objects as hecs entities.
                // First pass: create entities + map scene UUID → hecs entity.
                use crate::components::*;
                let mut uuid_to_hecs: std::collections::HashMap<uuid::Uuid, hecs::Entity> =
                    std::collections::HashMap::new();

                for obj in &scene.objects {
                    let transform = Transform {
                        position: glam::Vec3::from_array(obj.position),
                        rotation: glam::Vec3::from_array(obj.rotation),
                        scale: glam::Vec3::from_array(obj.scale),
                    };
                    let meta = EditorMetadata { name: obj.name.clone() };

                    let entity = if let Some(ref asset_path) = obj.asset_path {
                        let full_path = self.project_dir.as_ref()
                            .map(|d| d.join("assets").join(asset_path))
                            .unwrap_or_else(|| std::path::PathBuf::from(asset_path));
                        match self.scene_mgr.lock().unwrap().acquire_asset(&full_path.to_string_lossy()) {
                            Ok((handle, info)) => {
                                let spatial = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.grid_origin, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
                                let e = self.world.spawn((transform, meta, Renderable {
                                    asset_path: Some(asset_path.clone()),
                                    material_id: obj.material_id,
                                    voxel_count: info.voxel_count,
                                    spatial: Some(crate::components::RenderGeometry::Octree(spatial)),
                                    asset_handle: Some(handle),
                                    material_overrides: obj.material_overrides.clone(),
                                    ..Default::default()
                                }));
                                self.geometry_dirty.mark_all();
                                Some(e)
                            }
                            Err(_) => None,
                        }
                    } else if obj.procedural_cache.is_some() {
                        // Two cases land here:
                        //   - Procedurals (`primitive == Some("procedural")`):
                        //     the tree component arrives via the generic
                        //     components pass (ProceduralGeometry). The
                        //     cache provides the pre-baked voxels so
                        //     reload is instant.
                        //   - Persistent generator children (`primitive
                        //     == None`, GeneratorOwned arrives via the
                        //     generic components pass): the cache is
                        //     their only geometry source. Without
                        //     loading it here the child is invisible
                        //     until the generator regens.
                        //
                        // Either way, attach the baked spatial. Missing
                        // or unreadable cache leaves the entity empty —
                        // for procedurals that's recoverable via Bake;
                        // for generator children the parent generator's
                        // next tick will detect the missing slot and
                        // re-emit it.
                        let (spatial, asset_handle, voxel_count) = match (&obj.procedural_cache, &scene_dir) {
                            (Some(rel), Some(dir)) => {
                                let full = dir.join(rel);
                                if full.exists() {
                                    match self.scene_mgr.lock().unwrap().acquire_asset(&full.to_string_lossy()) {
                                        Ok((handle, info)) => {
                                            let sp = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.grid_origin, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
                                            (Some(crate::components::RenderGeometry::Octree(sp)), Some(handle), info.voxel_count)
                                        }
                                        Err(e) => {
                                            self.console.warn(format!(
                                                "Failed to load procedural cache '{rel}' for '{}': {e}",
                                                obj.name,
                                            ));
                                            (None, None, 0)
                                        }
                                    }
                                } else {
                                    self.console.warn(format!(
                                        "Procedural cache '{rel}' referenced by '{}' not found — entity will load unbaked",
                                        obj.name,
                                    ));
                                    (None, None, 0)
                                }
                            }
                            _ => (None, None, 0),
                        };
                        let e = self.world.spawn((transform, meta, Renderable {
                            // Preserve the saved primitive tag —
                            // `Some("procedural")` for un-converted
                            // procedurals (so the inspector still
                            // recognises them and the components pass
                            // attaches the tree); `None` for generator
                            // children (no tree, no procedural
                            // affordances, just baked voxels).
                            primitive: obj.primitive.clone(),
                            material_id: obj.material_id,
                            voxel_count,
                            spatial,
                            asset_handle,
                            material_overrides: obj.material_overrides.clone(),
                            ..Default::default()
                        }));
                        self.geometry_dirty.mark_all();
                        Some(e)
                    } else if let Some(ref prim_name) = obj.primitive {
                        let primitive = match prim_name.as_str() {
                            "box" => arvx_core::scene_node::SdfPrimitive::Box {
                                half_extents: glam::Vec3::from_array(obj.scale) * 0.5,
                            },
                            "sphere" => arvx_core::scene_node::SdfPrimitive::Sphere {
                                radius: obj.scale[0] * 0.5,
                            },
                            _ => continue,
                        };
                        // `object_id` is only forwarded to the retired
                        // `pending_faces` emit path; pass 0 to indicate
                        // "no pickable identity" until we either revive
                        // face emission or drop the parameter.
                        self.scene_mgr.lock().unwrap().voxelize_primitive(
                            &primitive, obj.material_id, 0.05, glam::Vec3::ONE, 0,
                        ).map(|result| {
                            let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.grid_origin, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
                            let e = self.world.spawn((transform, meta, Renderable {
                                primitive: Some(prim_name.clone()),
                                material_id: obj.material_id,
                                voxel_count: result.voxel_count,
                                spatial: Some(crate::components::RenderGeometry::Octree(spatial)),
                                material_overrides: obj.material_overrides.clone(),
                                ..Default::default()
                            }));
                            self.geometry_dirty.mark_all();
                            e
                        })
                    } else {
                        // Entity with no renderable (e.g. empty transform node).
                        Some(self.world.spawn((transform, meta)))
                    };

                    if let Some(e) = entity {
                        // Keep the UUID from the scene file — freshly
                        // generating a new one would orphan anything
                        // keyed off the ID (bake-cache sidecars, MCP
                        // references, per-entity persisted data).
                        self.set_entity_uuid(e, obj.id);
                        uuid_to_hecs.insert(obj.id, e);

                        // Replay any persisted material overrides
                        // against the freshly-loaded voxels. The
                        // `Renderable` already carries the override
                        // list for future saves; this brings the
                        // live voxel state in line with it.
                        // `remap_entity_material` is a no-op when
                        // `from == to` or when no voxels match, so
                        // stale entries self-heal rather than error.
                        for &(from, to) in &obj.material_overrides {
                            self.remap_entity_material(e, from, to);
                        }
                        // Tree order: prefer the persisted value.
                        // Legacy saves without `tree_order` get a
                        // fresh monotonic key *in file order* — the
                        // file lists objects in tree order, which is
                        // what the user last saw. The alternative
                        // (backfilling later via hecs query iteration)
                        // would reorder in archetype order, which
                        // feels arbitrary to the user.
                        match obj.tree_order {
                            Some(k) => {
                                self.entity_tree_order.insert(e, k);
                            }
                            None => {
                                let k = self.next_tree_order;
                                self.next_tree_order += 1.0;
                                self.entity_tree_order.insert(e, k);
                            }
                        }

                        // Restore PointLight component.
                        if let Some(ref pl) = obj.point_light {
                            let _ = self.world.insert_one(e, PointLight {
                                color: pl.color,
                                intensity: pl.intensity,
                                range: pl.range,
                                cast_shadow: pl.cast_shadow,
                            });
                        }

                        // Restore Camera component.
                        if let Some(ref cam) = obj.camera {
                            let _ = self.world.insert_one(e, Camera {
                                fov: cam.fov,
                                near: cam.near,
                                far: cam.far,
                                active: cam.active,
                            });
                        }
                    }
                }

                // Second pass: restore parent-child relationships.
                for obj in &scene.objects {
                    if let Some(parent_uuid) = obj.parent_id {
                        if let Some(&entity) = uuid_to_hecs.get(&obj.id) {
                            let _ = self.world.insert_one(entity, Parent { parent_id: parent_uuid });
                        }
                    }
                }

                // Third pass: restore generic components via registry.
                // Skeleton is deferred to a fourth pass because it
                // depends on sibling `.arvxskel` discovery off the
                // Renderable's asset path, and on `AnimationPlayer`
                // already being in place so `try_attach_skeleton`
                // doesn't overwrite the restored playback state.
                for obj in &scene.objects {
                    if obj.components.is_empty() {
                        continue;
                    }
                    let Some(&entity) = uuid_to_hecs.get(&obj.id) else { continue };
                    for (comp_name, json) in &obj.components {
                        if comp_name == "Skeleton" {
                            continue; // handled in the fourth pass below
                        }
                        if let Some(entry) = self.registry.get(comp_name) {
                            if let Err(e) = (entry.deserialize_insert)(&mut self.world, entity, json) {
                                self.console.warn(format!(
                                    "Failed to restore component '{comp_name}' on '{}': {e}",
                                    obj.name,
                                ));
                            }
                        } else {
                            self.console.warn(format!(
                                "Unknown component '{comp_name}' on '{}' — skipped (gameplay dylib not loaded?)",
                                obj.name,
                            ));
                        }
                    }
                }

                // Fourth pass: re-attach Skeleton (+ bundled
                // AnimationPlayer, preserving the restored-from-disk
                // player state). Uses the same engine-side helper the
                // AddComponent command routes through, so the asset
                // cache + grid-offset derivation stay in one place.
                for obj in &scene.objects {
                    if !obj.components.iter().any(|(n, _)| n == "Skeleton") {
                        continue;
                    }
                    let Some(&entity) = uuid_to_hecs.get(&obj.id) else { continue };
                    let Some(ref asset_path) = obj.asset_path else {
                        self.console.warn(format!(
                            "Restore Skeleton on '{}': no Renderable asset — skipped",
                            obj.name,
                        ));
                        continue;
                    };
                    let full_path = self.project_dir.as_ref()
                        .map(|d| d.join("assets").join(asset_path))
                        .unwrap_or_else(|| std::path::PathBuf::from(asset_path));
                    self.try_attach_skeleton(entity, &full_path);
                }

                // Fifth pass: reconcile ProceduralGeometry.dirty with
                // whether a bake cache actually loaded. Deserialization
                // defaults `dirty = true` to cover legacy scenes with
                // no cache concept; after the cache load we flip that
                // to `false` on entities whose Renderable has a spatial
                // — otherwise the properties panel would mislead the
                // user into thinking a freshly-restored procedural
                // needed rebaking. Entities without a loaded spatial
                // stay dirty so the UI's unbaked chip is accurate.
                let proc_entities: Vec<hecs::Entity> = self
                    .world
                    .query::<(&ProceduralGeometry, &Renderable)>()
                    .iter()
                    .filter(|(_, (_, r))| r.spatial.is_some())
                    .map(|(e, _)| e)
                    .collect();
                for entity in proc_entities {
                    if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        pg.dirty = false;
                        // Seed `last_evaluated_root_scale` from the
                        // tree's Root so `redirect_transform_scale_to_root`
                        // computes a sane preview multiplier on the
                        // first interaction.
                        let root_id = pg.tree.root();
                        if let Some(root) = pg.tree.get(root_id) {
                            let (s, _, _) = root.transform.to_scale_rotation_translation();
                            pg.last_evaluated_root_scale = s;
                        }
                    }
                }

                // Reseed `next_tree_order` past the max value loaded
                // from the scene file so post-load spawns continue to
                // append at the bottom. Entities missing a persisted
                // `tree_order` already got fresh monotonic keys in
                // file order in the spawn loop above — no second pass
                // here would help, and a hecs-query iteration would
                // actively hurt (archetype order ≠ file order).
                let max_loaded = self
                    .entity_tree_order
                    .values()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_loaded.is_finite() {
                    self.next_tree_order = max_loaded + 1.0;
                }

                self.scene_dirty.mark_all();
                self.gpu_objects_dirty.mark_all();
            }
            Err(e) => self.console.error(format!("Load scene failed: {e}")),
        }
    }

    /// Write the current project descriptor to disk, folding in the
    /// latest editor layout blob. No-op when no project is loaded
    /// (prevents the unnamed-scratch-session case from spraying files).
    pub(crate) fn save_project_file(&self) {
        let (Some(project_path), Some(_)) = (&self.project_path, &self.project_dir) else {
            return;
        };
        let project = crate::project::ProjectFile {
            name: self.project_name.clone(),
            default_scene: "default".to_string(),
            recent_scenes: Vec::new(),
            editor_layout: self.editor_layout_json.clone(),
        };
        if let Err(e) = crate::project::save_project(&project, project_path) {
            eprintln!("[ArvxEngine] save project failed: {e}");
        }
    }

    pub(crate) fn build_scene_file(&self) -> crate::scene_io::SceneFile {
        use crate::components::*;
        let mut objects = Vec::new();
        let scene_dir = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        for (entity, (transform, meta)) in self.world.query::<(&Transform, &EditorMetadata)>().iter() {
            let renderable = self.world.get::<&Renderable>(entity).ok();
            let parent = self.world.get::<&Parent>(entity).ok();
            let point_light = self.world.get::<&PointLight>(entity).ok();
            let camera = self.world.get::<&Camera>(entity).ok();

            // Serialize extra components (gameplay + any non-hardcoded) via registry.
            let hardcoded = ["Transform", "EditorMetadata", "Renderable", "PointLight", "Camera", "Parent"];
            let mut components = std::collections::HashMap::new();
            for entry in self.registry.components_on(&self.world, entity) {
                if hardcoded.contains(&entry.name) {
                    continue;
                }
                if let Some(json) = (entry.serialize)(&self.world, entity) {
                    components.insert(entry.name.to_string(), json);
                }
            }

            // Procedural bake cache reference — points at the .arvx
            // sidecar that holds this entity's pre-baked voxels so
            // load can restore them without re-running anything. Two
            // sources flow through this same field:
            //
            //   1. Procedurals: `procedural_cache_path()` →
            //      `{scene}.bakes/{uuid}.arvx` written by the bake
            //      worker on every procedural bake.
            //   2. Persistent generator children: derived from
            //      `(parent_uuid, slot_key)` →
            //      `{scene}.bakes/gen_{parent}_{slot}.arvx` written by
            //      the bake worker via the `cache_output_path` set on
            //      the BakeRequest by `enqueue_child_bake`.
            //
            // Either way, only emit when the file actually exists. An
            // unsaved scratch scene (no `scene_path`) or a never-baked
            // entity won't have one. Converted procedurals took a
            // different route (`assets/converted/*.arvx` via
            // `asset_path`) so they don't appear here.
            let procedural_cache = {
                let abs = if components.contains_key("ProceduralGeometry") {
                    self.procedural_cache_path(entity)
                } else if let Ok(owned) = self.world.get::<&crate::generator::GeneratorOwned>(entity) {
                    let stem_opt = self
                        .scene_path
                        .as_ref()
                        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()));
                    match (&scene_dir, stem_opt) {
                        (Some(dir), Some(stem)) => {
                            let bakes = dir.join(format!("{stem}.bakes"));
                            Some(crate::generator::child_cache_path(
                                &bakes,
                                owned.parent_uuid,
                                &owned.slot_key,
                            ))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match (abs, &scene_dir) {
                    (Some(abs), Some(dir)) if abs.exists() => abs
                        .strip_prefix(dir)
                        .ok()
                        .map(|rel| rel.to_string_lossy().to_string()),
                    _ => None,
                }
            };

            objects.push(crate::scene_io::SceneObject {
                id: self.get_entity_uuid(entity),
                name: meta.name.clone(),
                position: transform.position.to_array(),
                rotation: transform.rotation.to_array(),
                scale: transform.scale.to_array(),
                tree_order: self.entity_tree_order.get(&entity).copied(),
                parent_id: parent.map(|p| p.parent_id),
                asset_path: renderable.as_ref().and_then(|r| r.asset_path.clone()),
                primitive: renderable.as_ref().and_then(|r| r.primitive.clone()),
                procedural_cache,
                material_id: renderable.as_ref().map(|r| r.material_id).unwrap_or(0),
                material_overrides: renderable
                    .as_ref()
                    .map(|r| r.material_overrides.clone())
                    .unwrap_or_default(),
                point_light: point_light.map(|l| crate::scene_io::ScenePointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    cast_shadow: l.cast_shadow,
                }),
                camera: camera.map(|c| crate::scene_io::SceneCamera {
                    fov: c.fov,
                    near: c.near,
                    far: c.far,
                    active: c.active,
                }),
                components,
            });
        }

        crate::scene_io::SceneFile {
            objects,
            camera: crate::scene_io::CameraState {
                position: self.camera.position.to_array(),
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
                fov: self.camera.fov,
            },
            lights: Vec::new(),
            environment: Some(self.environment.clone()),
        }
    }
}
