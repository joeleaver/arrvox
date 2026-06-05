//! Scene / project file load + save.
//!
//! Reads `.arvxscene` from disk into the ECS world, writes the current
//! world back to `.arvxscene`, and writes the `.arvxproject` sidecar. The
//! actual serde types live in `crate::scene_io`; these methods are the
//! EngineState-side orchestration that walks the world.

use super::state::EngineState;
use super::model_scan::spatial_from_handle;

/// One queued voxel-asset load, drained by
/// [`EngineState::drain_pending_asset_loads`]. See the field doc on
/// `EngineState::pending_asset_loads` for why loads are deferred.
pub(crate) struct PendingAssetLoad {
    pub(crate) entity: hecs::Entity,
    pub(crate) full_path: std::path::PathBuf,
}

/// Decide whether a procedural-cache file holds a proxy mesh (`AVXP`)
/// or a voxel asset (`AVX\x01`) by sniffing the 4-byte magic, NOT the
/// file extension. The auto-bake path historically wrote proxy-mesh
/// content to a `.arvx`-suffixed path (`procedural_cache_path` always
/// returns `.arvx`), so routing on extension fed proxy caches to the
/// voxel loader → "Invalid magic" → fall through to a full re-bake on
/// every load. Content sniffing fixes those existing caches in place
/// with no migration. Unreadable / too-short files return `false` so
/// the voxel loader runs and surfaces its own diagnostic.
fn cache_file_is_proxy(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok()
        && buf == arvx_core::asset_file::ARVXPROXY_MAGIC
}

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
                        // Deferred: spawn the entity geometry-less now and
                        // queue the heavy voxel integrate for
                        // `drain_pending_asset_loads` (a budget per tick),
                        // so loading a scene full of million-voxel assets
                        // doesn't freeze the viewport for seconds. Parenting,
                        // components, ordering and the inspector all resolve
                        // against the real entity immediately; the geometry
                        // pops in when its load completes.
                        let e = self.world.spawn((transform, meta, Renderable {
                            asset_path: Some(asset_path.clone()),
                            material_id: obj.material_id,
                            voxel_count: 0,
                            spatial: None,
                            asset_handle: None,
                            material_overrides: obj.material_overrides.clone(),
                            ..Default::default()
                        }));
                        self.pending_asset_loads.push_back(PendingAssetLoad {
                            entity: e,
                            full_path,
                        });
                        Some(e)
                    } else if obj.procedural_cache.is_some() {
                        // Three cases land here:
                        //   - Procedurals (`primitive == Some("procedural")`):
                        //     the tree component arrives via the generic
                        //     components pass (ProceduralGeometry). The
                        //     cache provides the pre-baked geometry so
                        //     reload is instant.
                        //   - Persistent generator children with proxy-
                        //     mesh geometry (`.arvxproxy` cache): the
                        //     mesh-first default for `emit_child`.
                        //     `GeneratorOwned` arrives via the generic
                        //     components pass.
                        //   - Persistent generator children with voxel
                        //     geometry (`.arvx` cache): the rarer
                        //     `emit_child_artifact` path.
                        //
                        // Cache extension picks the loader:
                        // `.arvxproxy` → ProxyMesh spatial via direct
                        // read + render-thread upload; anything else →
                        // shared asset cache (`.arvx` voxel asset).
                        // Missing / unreadable caches leave the entity
                        // empty — recoverable on the next Bake (for
                        // single procedurals) or generator regen (for
                        // children).
                        // Set by the voxel sub-case below to defer the
                        // heavy `.arvx` integrate to `drain_pending_asset_loads`.
                        let mut deferred_voxel_path: Option<std::path::PathBuf> = None;
                        let (spatial, asset_handle, voxel_count) = match (&obj.procedural_cache, &scene_dir) {
                            (Some(rel), Some(dir)) => {
                                let full = dir.join(rel);
                                if !full.exists() {
                                    self.console.warn(format!(
                                        "Procedural cache '{rel}' referenced by '{}' not found — entity will load unbaked",
                                        obj.name,
                                    ));
                                    (None, None, 0)
                                } else if cache_file_is_proxy(&full) {
                                    match arvx_core::asset_file::read_arvxproxy(&full) {
                                        Ok(cache) => {
                                            let aabb = arvx_core::Aabb {
                                                min: glam::Vec3::from_array(cache.aabb_min),
                                                max: glam::Vec3::from_array(cache.aabb_max),
                                            };
                                            let surface_mesh = arvx_render::proc_surface_nets::SurfaceMesh {
                                                vertices: cache.vertices,
                                                indices: cache.indices,
                                                aabb_min: aabb.min,
                                                aabb_max: aabb.max,
                                            };
                                            let cluster = surface_mesh.single_cluster();
                                            let handle = self
                                                .scene_mgr
                                                .lock()
                                                .unwrap()
                                                .reserve_procedural_handle();
                                            let _ = self.render_worker.commands.send(
                                                crate::render_frame::RenderCommand::UploadProxyMesh {
                                                    handle_raw: handle.raw(),
                                                    vertices: surface_mesh.vertices,
                                                    indices: surface_mesh.indices,
                                                    cluster,
                                                },
                                            );
                                            (
                                                Some(crate::components::RenderGeometry::ProxyMesh(
                                                    crate::components::ProxyMeshData { handle, aabb },
                                                )),
                                                Some(handle),
                                                0,
                                            )
                                        }
                                        Err(e) => {
                                            self.console.warn(format!(
                                                "Failed to load proxy cache '{rel}' for '{}': {e}",
                                                obj.name,
                                            ));
                                            (None, None, 0)
                                        }
                                    }
                                } else {
                                    // Voxel cache (Convert'd object or
                                    // voxelized generator child — never a
                                    // ProceduralGeometry entity, which always
                                    // caches as proxy). Defer the integrate to
                                    // drain_pending_asset_loads so the big
                                    // multi-million-voxel caches don't freeze
                                    // the load; the entity spawns geometry-less
                                    // and reveals when the load completes.
                                    deferred_voxel_path = Some(full.clone());
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
                            // children.
                            primitive: obj.primitive.clone(),
                            material_id: obj.material_id,
                            voxel_count,
                            spatial,
                            asset_handle,
                            material_overrides: obj.material_overrides.clone(),
                            ..Default::default()
                        }));
                        if let Some(path) = deferred_voxel_path {
                            self.pending_asset_loads.push_back(PendingAssetLoad {
                                entity: e,
                                full_path: path,
                            });
                        } else {
                            self.geometry_dirty.mark_all();
                        }
                        Some(e)
                    } else if obj.primitive.as_deref() == Some("procedural") {
                        // Procedural without an on-disk bake cache.
                        // The proxy-mesh bake path (the only bake
                        // procedurals take today) doesn't persist its
                        // output to .arvx — the cache branch above only
                        // fires for generator children — so this is the
                        // common case for any saved procedural. Spawn
                        // an empty entity here; the third pass attaches
                        // ProceduralGeometry from `obj.components`, and
                        // the fifth pass marks it `pending_bake` so the
                        // proxy mesh regenerates on the next tick.
                        let e = self.world.spawn((transform, meta, Renderable {
                            primitive: Some("procedural".to_string()),
                            material_id: obj.material_id,
                            voxel_count: 0,
                            spatial: None,
                            asset_handle: None,
                            material_overrides: obj.material_overrides.clone(),
                            ..Default::default()
                        }));
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
                            // Diagnostic: a registered component whose
                            // hecs layout has gone insane (huge size,
                            // bad align) panics inside `insert_one`
                            // with a `LayoutError` and no context.
                            // Log the about-to-insert name so a crash
                            // report tells us which component to
                            // investigate. Stays in place — cheap.
                            eprintln!(
                                "[scene-load] restoring component '{}' on '{}'",
                                comp_name, obj.name,
                            );
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
                // needed rebaking.
                let proc_entities_with_spatial: Vec<hecs::Entity> = self
                    .world
                    .query::<(&ProceduralGeometry, &Renderable)>()
                    .iter()
                    .filter(|(_, (_, r))| r.spatial.is_some())
                    .map(|(e, _)| e)
                    .collect();
                for entity in proc_entities_with_spatial {
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

                // Procedurals without a loaded spatial — the common
                // case, since the proxy-mesh bake path doesn't write
                // cache files. Schedule them for an auto-bake on the
                // next tick (same flags `SpawnProceduralObject` uses).
                // Without this they'd render invisible until the user
                // manually nudges a parameter or hits Bake.
                let proc_entities_without_spatial: Vec<hecs::Entity> = self
                    .world
                    .query::<(&ProceduralGeometry, &Renderable)>()
                    .iter()
                    .filter(|(_, (_, r))| r.spatial.is_none())
                    .map(|(e, _)| e)
                    .collect();
                let now = std::time::Instant::now();
                for entity in proc_entities_without_spatial {
                    if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        // Seed `last_evaluated_root_scale` too — needed
                        // before the first interaction even if no
                        // spatial is loaded yet.
                        let root_id = pg.tree.root();
                        if let Some(root) = pg.tree.get(root_id) {
                            let (s, _, _) = root.transform.to_scale_rotation_translation();
                            pg.last_evaluated_root_scale = s;
                        }
                        pg.dirty = false;
                        pg.pending_bake = true;
                        pg.bake_dirty_at = Some(now);
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

                // If the loaded scene contains a Terrain singleton,
                // create its runtime so the streamer actually ticks.
                // Without this, `tick_terrain_streamer` early-returns
                // on `self.terrain.is_none()` and the user has to
                // manually click "+ Terrain" (which the singleton
                // check then rejects, leaving them stuck). The helper
                // also hydrates `runtime.diffs` from any
                // `.arvxsculpt` sidecars saved with this scene so
                // prior-session sculpts replay onto the fresh bakes.
                //
                // Singleton enforced by taking the first match — the
                // save path writes exactly one Terrain component, and
                // `SpawnTerrain` rejects a second.
                if self.terrain.is_none() {
                    let terrain_entity: Option<hecs::Entity> = self
                        .world
                        .query::<&arvx_terrain::Terrain>()
                        .iter()
                        .next()
                        .map(|(e, _)| e);
                    if let Some(e) = terrain_entity {
                        self.init_terrain_runtime(e);
                    }
                }

                self.scene_dirty.mark_all();
                self.gpu_objects_dirty.mark_all();
            }
            Err(e) => self.console.error(format!("Load scene failed: {e}")),
        }
    }

    /// Integrate a bounded budget of deferred voxel-asset loads per
    /// tick (see [`EngineState::pending_asset_loads`]). Each
    /// `acquire_asset` reads + integrates one `.arvx` into the shared
    /// pools under the `scene_mgr` lock; spreading them across ticks
    /// keeps the viewport shipping frames during a heavy scene load
    /// instead of freezing for seconds while every asset integrates.
    pub(crate) fn drain_pending_asset_loads(&mut self) {
        if self.pending_asset_loads.is_empty() {
            return;
        }
        // Per-tick budget: finish the current asset even if it overruns
        // (one integrate can't be sub-divided), but stop pulling new
        // ones past the budget so a frame can ship between heavy assets.
        const BUDGET: std::time::Duration = std::time::Duration::from_millis(6);
        let start = std::time::Instant::now();
        while let Some(p) = self.pending_asset_loads.pop_front() {
            if !self.world.contains(p.entity) {
                continue; // entity deleted before its deferred load ran
            }
            let acquired = self
                .scene_mgr
                .lock()
                .unwrap()
                .acquire_asset(&p.full_path.to_string_lossy());
            match acquired {
                Ok((handle, info)) => {
                    let spatial = spatial_from_handle(
                        &info.spatial,
                        info.voxel_size,
                        &info.aabb,
                        info.grid_origin,
                        info.leaf_attr_slot_start,
                        info.leaf_attr_slot_count,
                        Vec::new(),
                    );
                    // Snapshot overrides before the mutable Renderable borrow.
                    let overrides: Vec<(u16, u16)> = self
                        .world
                        .get::<&crate::components::Renderable>(p.entity)
                        .map(|r| r.material_overrides.clone())
                        .unwrap_or_default();
                    if let Ok(mut r) =
                        self.world.get::<&mut crate::components::Renderable>(p.entity)
                    {
                        r.spatial = Some(crate::components::RenderGeometry::Octree(spatial));
                        r.asset_handle = Some(handle);
                        r.voxel_count = info.voxel_count;
                    }
                    // Replay persisted material overrides now that the
                    // voxels exist (the load-time first-pass replay was a
                    // no-op against the not-yet-loaded geometry).
                    for (from, to) in overrides {
                        self.remap_entity_material(p.entity, from, to);
                    }
                    self.geometry_dirty.mark_all();
                    self.gpu_objects_dirty.mark_all();
                    self.scene_dirty.mark_entity(p.entity);
                }
                Err(e) => {
                    self.console.warn(format!(
                        "Deferred asset load failed for {}: {e}",
                        p.full_path.display(),
                    ));
                }
            }
            if start.elapsed() >= BUDGET {
                break;
            }
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
            // Skip runtime-spawned terrain tile entities. Phase 2
            // doesn't persist tiles — they regenerate from the
            // Terrain entity's `TerrainFn` on next load. The Terrain
            // entity itself has no `TerrainTile` component, so the
            // singleton still flows through normal save.
            if self
                .world
                .get::<&arvx_terrain::TerrainTile>(entity)
                .is_ok()
            {
                continue;
            }
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
            // load can restore them without re-running anything. Three
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
            //   3. Convert / Copy of a procedural: writes to the same
            //      `{scene}.bakes/{uuid}.arvx` path so the resulting
            //      static voxel object doesn't pollute the Models
            //      panel (which scans `assets/`). The entity has
            //      neither `ProceduralGeometry` nor `GeneratorOwned`
            //      at save time; we detect it by checking that the
            //      uuid-keyed bake file exists on disk.
            //
            // Either way, only emit when the file actually exists. An
            // unsaved scratch scene (no `scene_path`) or a never-baked
            // entity won't have one.
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
                            // Generator children come in two flavours:
                            //   - Proxy-mesh (`emit_child`) → `.arvxproxy`
                            //   - Voxelized (`emit_child_artifact`) → `.arvx`
                            // Both share the `gen_<parent>_<slot>` stem;
                            // probe proxy first (the dominant path),
                            // fall through to voxel for the rare case.
                            // The `abs.exists()` gate below filters
                            // unbacked entries.
                            let bakes = dir.join(format!("{stem}.bakes"));
                            let proxy_path = crate::generator::child_cache_path(
                                &bakes,
                                owned.parent_uuid,
                                &owned.slot_key,
                                "arvxproxy",
                            );
                            if proxy_path.exists() {
                                Some(proxy_path)
                            } else {
                                Some(crate::generator::child_cache_path(
                                    &bakes,
                                    owned.parent_uuid,
                                    &owned.slot_key,
                                    "arvx",
                                ))
                            }
                        }
                        _ => None,
                    }
                } else {
                    // Convert/Copy-produced static voxel object: the
                    // bake file (if any) lives at the uuid-keyed path
                    // and we fall through to the `abs.exists()` gate
                    // below — library-asset entities (asset_path set,
                    // no bake file) emit None as before.
                    self.procedural_cache_path(entity)
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
