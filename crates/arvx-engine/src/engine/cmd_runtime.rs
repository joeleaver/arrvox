//! Runtime-side command handlers: input events, component mutations,
//! material edits, import-profile / reimport, environment, play mode.
//!
//! Third chunk of the `process_command` match. Anything this file
//! doesn't match returns `Err(cmd)` to the dispatcher, which logs it
//! as unhandled.

use crate::command::EngineCommand;

use super::state::EngineState;

impl EngineState {
    pub(crate) fn process_cmd_runtime(
        &mut self,
        cmd: EngineCommand,
    ) -> Result<(), EngineCommand> {
        match cmd {
            // input system. Build viewport / PiP wiring lands in Phase 6.
            EngineCommand::MouseMove { id, x, y, dx, dy } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.mouse_pos = glam::Vec2::new(x, y);
                    self.input_system.feed_mouse_delta(glam::Vec2::new(dx, dy));
                } else if id == crate::viewport::ViewportId::BUILD {
                    self.build_mouse_pos = glam::Vec2::new(x, y);
                    let _ = (dx, dy);
                }
            }
            EngineCommand::MouseButton { id, button, pressed } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.input_system.feed_mouse_button(button, pressed);
                } else if id == crate::viewport::ViewportId::BUILD {
                    if button == arvx_runtime::input::InputMouseButton::Left {
                        self.build_mouse_left = pressed;
                    }
                }
            }
            EngineCommand::Scroll { id, delta } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.input_system.feed_scroll(delta);
                }
            }
            EngineCommand::KeyDown { key } => {
                self.input_system.feed_key_down(key);
            }
            EngineCommand::KeyUp { key } => {
                self.input_system.feed_key_up(key);
            }

            EngineCommand::SetComponentField { entity_id, component_name, field_name, value } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Some(entry) = self.registry.get(&component_name) {
                        if let Ok(fv) = serde_json::from_str::<crate::inspector::FieldValue>(&value) {
                            if let Err(e) = (entry.set_field)(&mut self.world, entity, &field_name, fv) {
                                eprintln!("[ArvxEngine] set_field failed: {e}");
                            } else {
                                if component_name == "Transform" {
                                    // Procedural entities treat Transform.scale as
                                    // an alias for the Root node's scale: bake the
                                    // value into the tree, reset the entity scale,
                                    // and queue an auto-bake. Keeps procedural
                                    // entities at world scale 1 so colliders /
                                    // gizmos / physics aren't double-scaled, and
                                    // makes the bake actually produce voxels at
                                    // the right density (the entity-level scale
                                    // path was a no-op visually — same voxels,
                                    // just stretched at render time).
                                    if field_name == "scale" {
                                        self.redirect_transform_scale_to_root(entity);
                                    }
                                    self.gpu_objects_dirty.mark_all();
                                }
                                if component_name == "RigidBody" {
                                    self.collider_caches_dirty.mark_all();
                                }
                                // Phase 5.6: Inspector edits to a stamp's
                                // amplitude / radius / priority change its
                                // contribution to the heightmap. Re-sync the
                                // index + invalidate the stamp's AABB so the
                                // streamer re-bakes affected tiles. AABB might
                                // have grown (bigger radius / amplitude), so
                                // compute it AFTER the set_field write.
                                if component_name == "Stamp" {
                                    if let Some(aabb) = self
                                        .world
                                        .get::<&arvx_terrain::Stamp>(entity)
                                        .ok()
                                        .map(|s| s.aabb())
                                    {
                                        self.sync_terrain_stamps_and_invalidate(Some(aabb));
                                    }
                                }
                                // Phase 7: Inspector edits to a Region's
                                // shape / falloff / priority OR its
                                // BiomeRegion's material override change
                                // what the bake sees. Re-sync + invalidate
                                // the post-edit AABB (or both old and new
                                // for Region itself — `maybe_sync_region`
                                // would need a captured pre-edit AABB; for
                                // V1 we conservatively invalidate the new
                                // AABB only; the streamer's
                                // `invalidate_aabb` is liberal enough that
                                // a grown region still picks up its
                                // newly-covered tiles).
                                if component_name == "Region" {
                                    let new_aabb = {
                                        let region = self
                                            .world
                                            .get::<&arvx_regions::Region>(entity)
                                            .ok()
                                            .map(|r| *r);
                                        let position = self
                                            .world
                                            .get::<&crate::components::Transform>(entity)
                                            .ok()
                                            .map(|t| t.position);
                                        region.zip(position).map(|(r, p)| r.world_aabb(p))
                                    };
                                    if let Some(aabb) = new_aabb {
                                        self.sync_terrain_regions_and_invalidate(Some(aabb));
                                    }
                                }
                                if component_name == "BiomeRegion" {
                                    // BiomeRegion has no AABB of its own —
                                    // its reach is the host Region's. Use
                                    // that.
                                    let new_aabb = {
                                        let region = self
                                            .world
                                            .get::<&arvx_regions::Region>(entity)
                                            .ok()
                                            .map(|r| *r);
                                        let position = self
                                            .world
                                            .get::<&crate::components::Transform>(entity)
                                            .ok()
                                            .map(|t| t.position);
                                        region.zip(position).map(|(r, p)| r.world_aabb(p))
                                    };
                                    if let Some(aabb) = new_aabb {
                                        self.sync_terrain_regions_and_invalidate(Some(aabb));
                                    }
                                }
                                // Phase 9: Terrain Inspector edits
                                // change the world's procedural source
                                // (TerrainFn parameters), bounds, or
                                // residency radius. Any of these
                                // invalidates every live tile —
                                // simplest correct response is to
                                // evict all and let the streamer
                                // repopulate on the next residency
                                // tick.
                                if component_name == "Terrain" {
                                    self.invalidate_all_terrain_tiles();
                                }
                            }
                        }
                    }
                }
            }

            EngineCommand::AddComponent { entity_id, component_name } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    // Skeleton needs more context than the registry's
                    // plain (World, Entity) `add_default` — it has to
                    // find the sibling `.arvxskel` next to the entity's
                    // Renderable asset and load it. Route here first;
                    // the attach helper also inserts an
                    // AnimationPlayer alongside (components are
                    // bundled — you never want one without the other).
                    if component_name == "Skeleton" {
                        let arvx_path = self.world
                            .get::<&crate::components::Renderable>(entity)
                            .ok()
                            .and_then(|r| r.asset_path.clone());
                        match arvx_path {
                            Some(p) => self.try_attach_skeleton(entity, std::path::Path::new(&p)),
                            None => self.console.warn(
                                "Add Skeleton: entity has no Renderable asset — attach a model first".to_string(),
                            ),
                        }
                        self.scene_dirty.mark_entity(entity);
                        self.gpu_objects_dirty.mark_all();
                    } else if let Some(entry) = self.registry.get(&component_name) {
                        if let Err(e) = (entry.add_default)(&mut self.world, entity) {
                            eprintln!("[ArvxEngine] add component failed: {e}");
                        }
                        self.scene_dirty.mark_entity(entity);
                        self.gpu_objects_dirty.mark_all();
                        if component_name == "RigidBody" {
                            self.collider_caches_dirty.mark_all();
                        }
                        // Phase 7: adding a BiomeRegion changes what
                        // the bake sees for the host region. Re-sync
                        // + invalidate against the region's AABB.
                        if component_name == "BiomeRegion" {
                            self.sync_terrain_regions_for_entity(entity);
                        }
                    }
                }
            }

            EngineCommand::RemoveComponent { entity_id, component_name } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    // Capture the region AABB BEFORE the remove if this
                    // is going to drop a BiomeRegion — we still need to
                    // invalidate the tiles it was contributing to.
                    let biome_aabb_pre_remove = if component_name == "BiomeRegion" {
                        self.capture_region_aabb_before_delete(entity)
                    } else {
                        None
                    };
                    if let Some(entry) = self.registry.get(&component_name) {
                        if let Err(e) = (entry.remove)(&mut self.world, entity) {
                            eprintln!("[ArvxEngine] remove component failed: {e}");
                        }
                        // Skeleton + AnimationPlayer are bundled —
                        // pulling the skeleton also pulls the player
                        // (ui treats AnimationPlayer as part of the
                        // Skeleton section, so an orphaned player
                        // would be invisible and confusing).
                        if component_name == "Skeleton" {
                            let _ = self.world.remove_one::<crate::components::AnimationPlayer>(entity);
                        }
                        self.scene_dirty.mark_entity(entity);
                        self.gpu_objects_dirty.mark_all();
                        if component_name == "RigidBody" {
                            self.collider_caches_dirty.mark_all();
                        }
                        // Phase 7: BiomeRegion removed → tiles it
                        // contributed to need to re-bake without it.
                        if let Some(aabb) = biome_aabb_pre_remove {
                            self.sync_terrain_regions_and_invalidate(Some(aabb));
                        }
                    }
                }
            }

            EngineCommand::CreateMaterial { name } => {
                match self.material_lib.create(&name) {
                    Ok(id) => {
                        eprintln!("[ArvxEngine] created material '{name}' as id {id}");
                        self.selected_material = Some(id);
                    }
                    Err(e) => eprintln!("[ArvxEngine] create material failed: {e}"),
                }
            }

            EngineCommand::UpdateMaterialField { material_id, field, value } => {
                if let Some(def) = self.material_lib.get_def_mut(material_id) {
                    match field.as_str() {
                        "name" => { def.name = value; }
                        "albedo" => {
                            if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) {
                                def.albedo = v;
                            }
                        }
                        "emission_color" => {
                            if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) {
                                def.emission_color = v;
                            }
                        }
                        "subsurface_color" => {
                            if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) {
                                def.subsurface_color = v;
                            }
                        }
                        "roughness" => {
                            if let Ok(v) = value.parse::<f32>() { def.roughness = v; }
                        }
                        "metallic" => {
                            if let Ok(v) = value.parse::<f32>() { def.metallic = v; }
                        }
                        "emission_strength" => {
                            if let Ok(v) = value.parse::<f32>() { def.emission_strength = v; }
                        }
                        "subsurface" => {
                            if let Ok(v) = value.parse::<f32>() { def.subsurface = v; }
                        }
                        "opacity" => {
                            if let Ok(v) = value.parse::<f32>() { def.opacity = v; }
                        }
                        "ior" => {
                            if let Ok(v) = value.parse::<f32>() { def.ior = v; }
                        }
                        "noise_scale" => {
                            if let Ok(v) = value.parse::<f32>() { def.noise_scale = v; }
                        }
                        "noise_strength" => {
                            if let Ok(v) = value.parse::<f32>() { def.noise_strength = v; }
                        }
                        "noise_channels" => {
                            if let Ok(v) = value.parse::<u32>() { def.noise_channels = v; }
                        }
                        _ => { eprintln!("[ArvxEngine] unknown material field: {field}"); }
                    }
                    self.material_lib.mark_dirty();
                    let _ = self.material_lib.save(material_id);
                }
            }

            EngineCommand::SetMaterialShader { material_id, shader_name } => {
                if let Some(def) = self.material_lib.get_def_mut(material_id) {
                    def.shader = match &shader_name {
                        Some(n) if !n.is_empty() => Some(n.clone()),
                        _ => None,
                    };
                    self.material_lib.mark_dirty();
                    let _ = self.material_lib.save(material_id);
                }
            }

            EngineCommand::SetMaterialShaderParam { material_id, name, value } => {
                if let Some(def) = self.material_lib.get_def_mut(material_id) {
                    def.shader_params.insert(name, serde_json::json!(value));
                    self.material_lib.mark_dirty();
                    let _ = self.material_lib.save(material_id);
                }
            }

            EngineCommand::DeleteMaterial { material_id } => {
                if let Some(path) = self.material_lib.path_for_id(material_id).map(|p| p.to_owned()) {
                    let _ = std::fs::remove_file(&path);
                    self.material_lib.remove(&path);
                    if self.selected_material == Some(material_id) {
                        self.selected_material = None;
                    }
                }
            }

            EngineCommand::AssignMaterial { entity_id, material_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                        r.material_id = material_id;
                        self.gpu_objects_dirty.mark_all();
                    }
                }
            }

            EngineCommand::SelectMaterial { material_id } => {
                // The Asset Properties panel inspects one thing at a time —
                // picking a material drops any prior model selection so the
                // panel swaps over instead of staying stuck on the model (or
                // vice versa).
                self.selected_material = material_id;
                if material_id.is_some() {
                    self.selected_model = None;
                }
            }

            EngineCommand::RemapMaterial { object_id, from_material, to_material } => {
                if let Some(entity) = self.resolve_entity(&object_id) {
                    let count = self.remap_entity_material(entity, from_material, to_material);
                    if count > 0 {
                        eprintln!("[ArvxEngine] remapped {count} voxels from material {from_material} to {to_material}");
                        self.geometry_dirty.mark_all();
                        if let Ok(mut r) =
                            self.world.get::<&mut crate::components::Renderable>(entity)
                        {
                            // Record the remap so it survives save/load.
                            Self::compose_material_override(
                                &mut r.material_overrides,
                                from_material,
                                to_material,
                            );
                            // Also track the latest-remap as the
                            // entity's default material. INTERIOR
                            // subtrees (fully-solid bulk regions
                            // produced by the voxelizer) have no
                            // per-voxel material and fall back to
                            // this at march time; without updating
                            // it, a glass remap leaves the cube's
                            // interior rendering as whatever material
                            // was here first (typically the opaque
                            // Default), so only the surface shell
                            // reads as glass.
                            r.material_id = to_material;
                        }
                    }
                }
            }

            EngineCommand::SetPrimitiveMaterial { object_id, material_id } => {
                if let Some(entity) = self.resolve_entity(&object_id) {
                    if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                        r.material_id = material_id;
                        self.gpu_objects_dirty.mark_all();
                    }
                }
            }

            EngineCommand::SelectModel { path } => {
                self.selected_model = path;
                if self.selected_model.is_some() {
                    self.selected_material = None;
                }
            }

            EngineCommand::UpdateImportField { source_path, field, value } => {
                // Find the model info, update its import profile, save sidecar.
                let source = std::path::PathBuf::from(&source_path);
                let mut profile = crate::import_profile::ImportProfile::load_or_default(&source);
                match field.as_str() {
                    "display_name" => {
                        profile.display_name = if value.is_empty() { None } else { Some(value) };
                    }
                    "voxel_size" => {
                        profile.voxel_size = value.parse::<f32>().ok().filter(|&v| v > 0.0);
                    }
                    "target_size" => {
                        if let Ok(v) = value.parse::<f32>() { profile.target_size = v; }
                    }
                    "no_normalize" => {
                        profile.no_normalize = value == "true";
                    }
                    "import_colors" => {
                        profile.import_colors = value == "true";
                    }
                    "rotation_x" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[0] = v; }
                    }
                    "rotation_y" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[1] = v; }
                    }
                    "rotation_z" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[2] = v; }
                    }
                    _ => {
                        eprintln!("[ArvxEngine] unknown import field: {field}");
                    }
                }
                if let Err(e) = profile.save_for(&source) {
                    eprintln!("[ArvxEngine] save import profile failed: {e}");
                }
                // Update the in-memory model info.
                if let Some(mi) = self.available_models.iter_mut().find(|m| m.source_path == source_path) {
                    if let Some(ref name) = profile.display_name {
                        mi.name = name.clone();
                    }
                    mi.import_profile = Some(profile);
                }
                self.models_dirty = true;
            }

            EngineCommand::ReimportModel { source_path } => {
                let source = std::path::PathBuf::from(&source_path);
                let source_key = source.to_string_lossy().into_owned();
                // Drop the request if this source already has an import
                // in flight. Without the guard a double-click would queue
                // two identical jobs, and the spinner would clear halfway
                // through while the second still ran in the background.
                if self.importing_sources.contains(&source_key) {
                    eprintln!(
                        "[ArvxEngine] re-import already in flight for {} — ignoring",
                        source.display(),
                    );
                    return Ok(());
                }
                let profile = crate::import_profile::ImportProfile::load_or_default(&source);
                let config = profile.to_import_config();
                let output = crate::import_worker::arvx_output_path(&source);
                eprintln!(
                    "[ArvxEngine] re-importing {} → {} \
                     (target_size={}, voxel_size={:?}, rotation={:?}, import_colors={})",
                    source.display(), output.display(),
                    config.target_size, config.voxel_size,
                    config.rotation_offset, config.import_colors,
                );
                let name = source.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.console.info(format!("Re-importing '{name}'…"));
                self.importing_sources.insert(source_key);
                self.importing_dirty = true;
                self.import_worker.submit(crate::import_worker::ImportRequest {
                    source_path: source,
                    output_path: output,
                    config,
                });
            }

            EngineCommand::SetViewOption { option, enabled } => {
                match option.as_str() {
                    "show_colliders" => self.show_colliders = enabled,
                    "skinning" => self.skinning_enabled = enabled,
                    "dqs" => self.dqs_enabled = enabled,
                    _ => eprintln!("[ArvxEngine] unknown view option: {option}"),
                }
            }

            EngineCommand::ClearConsole => {
                self.console.clear();
            }

            EngineCommand::UpdateEnvironment { field, value } => {
                let env = &mut self.environment;
                match field.as_str() {
                    "sky_color_top_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_top_override = Some(v); }
                    }
                    "sky_color_top_override_enabled" => {
                        if value == "false" { env.sky_color_top_override = None; }
                    }
                    "sky_color_horizon_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_horizon_override = Some(v); }
                    }
                    "sky_color_horizon_override_enabled" => {
                        if value == "false" { env.sky_color_horizon_override = None; }
                    }
                    "sun_color_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sun_color_override = Some(v); }
                    }
                    "sun_color_override_enabled" => {
                        if value == "false" { env.sun_color_override = None; }
                    }
                    "ambient_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.ambient_intensity = v; }
                    }
                    "sun_azimuth" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_azimuth = v; }
                    }
                    "sun_elevation" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_elevation = v; }
                    }
                    "sun_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_intensity = v; }
                    }
                    "shadow_steps" => {
                        if let Ok(v) = value.parse::<u32>() { env.shadow_steps = v; }
                    }
                    "shadow_csm_near" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_near = v.clamp(0.05, 10.0);
                        }
                    }
                    "shadow_csm_max_distance" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_max_distance = v.clamp(10.0, 1000.0);
                        }
                    }
                    "shadow_csm_lambda" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_lambda = v.clamp(0.0, 1.0);
                        }
                    }
                    "shadow_csm_depth_bias" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_depth_bias = v.clamp(0.0, 0.05);
                        }
                    }
                    "shadow_csm_threshold_falloff" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_threshold_falloff = v.clamp(1.0, 6.0);
                        }
                    }
                    "shadow_csm_sharp_distance" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.shadow_csm_sharp_distance = v.clamp(0.5, 50.0);
                        }
                    }
                    "shadow_csm_map_size" => {
                        if let Ok(v) = value.parse::<u32>() {
                            // Clamp to powers of two we actually
                            // expose: 256..=4096. Other values
                            // technically work but the preset row
                            // doesn't recognize them.
                            env.shadow_csm_map_size = v.clamp(256, 4096);
                        }
                    }
                    "shadow_csm_pcf_taps" => {
                        if let Ok(v) = value.parse::<u32>() {
                            env.shadow_csm_pcf_taps = v.clamp(1, 16);
                        }
                    }
                    "ao_radius" => {
                        if let Ok(v) = value.parse::<f32>() { env.ao_radius = v; }
                    }
                    "ao_steps" => {
                        if let Ok(v) = value.parse::<u32>() { env.ao_steps = v; }
                    }
                    "exposure" => {
                        if let Ok(v) = value.parse::<f32>() { env.exposure = v; }
                    }
                    "bloom_threshold" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_threshold = v; }
                    }
                    "bloom_knee" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_knee = v; }
                    }
                    "bloom_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_intensity = v; }
                    }
                    "god_ray_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_density = v; }
                    }
                    "god_ray_weight" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_weight = v; }
                    }
                    "god_ray_decay" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_decay = v; }
                    }
                    "god_ray_exposure" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_exposure = v; }
                    }
                    "scene_elevation" => {
                        if let Ok(v) = value.parse::<f32>() { env.scene_elevation = v; }
                    }
                    "ground_albedo" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.ground_albedo = v; }
                    }
                    // Fog
                    "fog_color" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.fog_color = v; }
                    }
                    "height_fog_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.height_fog_density = v; }
                    }
                    "fog_base_height" => {
                        if let Ok(v) = value.parse::<f32>() { env.fog_base_height = v; }
                    }
                    "fog_height_falloff" => {
                        if let Ok(v) = value.parse::<f32>() { env.fog_height_falloff = v; }
                    }
                    "vol_far" => {
                        if let Ok(v) = value.parse::<f32>() { env.vol_far = v; }
                    }
                    // Clouds
                    "clouds_enabled" => {
                        env.clouds_enabled = value == "true" || value == "1";
                    }
                    "attenuate_sun_by_clouds" => {
                        env.attenuate_sun_by_clouds = value == "true" || value == "1";
                    }
                    "cloud_slab_steps" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_slab_steps = (v as u32).clamp(8, 128);
                        }
                    }
                    "cloud_shadow_steps" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_shadow_steps = (v as u32).clamp(1, 8);
                        }
                    }
                    "cloud_detail_octaves" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_detail_octaves = (v as u32).clamp(1, 6);
                        }
                    }
                    "cloud_ms_octaves" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_ms_octaves = (v as u32).clamp(1, 5);
                        }
                    }
                    "cloud_taa_alpha" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_taa_alpha = v.clamp(0.05, 0.7);
                        }
                    }
                    "cloud_altitude_min" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_altitude_min = v; }
                    }
                    "cloud_altitude_max" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_altitude_max = v; }
                    }
                    "cloud_coverage" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_coverage = v; }
                    }
                    "cloud_density_scale" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_density_scale = v; }
                    }
                    "cloud_wind_speed" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_wind_speed = v; }
                    }
                    "cloud_wind_dir" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_wind_dir = v; }
                    }
                    _ => { eprintln!("[ArvxEngine] unknown environment field: {field}"); }
                }
                self.environment_dirty = true;
                // Deliberately do NOT set environment_ui_dirty: the UI already holds
                // the authoritative value (it just sent it). Echoing back would cause
                // the form to remount mid-drag on every slider tick.
            }

            EngineCommand::SetGizmoMode { mode } => {
                self.gizmo.mode = mode;
            }

            EngineCommand::PlayStart => {
                if self.play_state.is_none() {
                    // Ensure collider caches are up to date before entering play mode.
                    // Play-mode entry is rare; full rebuild is fine here.
                    if self.collider_caches_dirty.is_dirty() {
                        self.rebuild_collider_caches();
                        self.collider_caches_dirty.clear();
                    }
                    let play = crate::play_mode::PlayModeState::start(&mut self.world);
                    self.play_state = Some(play);
                    // Build behavior executor from gameplay systems.
                    match crate::behavior::BehaviorExecutor::new(&self.gameplay_systems) {
                        Ok(executor) => {
                            self.behavior_executor = Some(executor);
                            self.console.info(format!(
                                "Play mode started ({} systems)",
                                self.gameplay_systems.len(),
                            ));
                        }
                        Err(e) => {
                            self.behavior_executor = None;
                            self.console.error(format!("Failed to build system schedule: {e}"));
                            self.console.info("Play mode started (no systems)");
                        }
                    }
                    self.play_total_time = 0.0;
                    self.play_frame_count = 0;
                    // Reset FixedUpdate accumulator so play mode
                    // starts from a clean zero rather than firing a
                    // burst of catch-up steps.
                    self.behavior_fixed_accumulator = 0.0;
                    self.enter_play_mode_viewports();
                }
            }

            EngineCommand::PlayStop => {
                if let Some(play) = self.play_state.take() {
                    play.stop(&mut self.world);
                    self.behavior_executor = None;
                    self.gpu_objects_dirty.mark_all();
                    self.console.info("Play mode stopped — transforms restored");
                    self.exit_play_mode_viewports();
                }
            }

            other => return Err(other),
        }
        Ok(())
    }
}
