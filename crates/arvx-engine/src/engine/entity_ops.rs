//! Entity CRUD, UUIDs, unique-name resolution.
//!
//! Plain ECS-side operations on the `EngineState.world`. These methods
//! don't touch GPU state directly — they mutate the world + UUID maps +
//! tree order side-map and flag `scene_dirty` / `gpu_objects_dirty` so
//! the next tick picks up the change.

use super::state::EngineState;
use super::model_scan::spatial_from_handle;

impl EngineState {
    /// Resolve a Uuid (from UI) to an hecs::Entity.
    pub(crate) fn resolve_entity(&self, uuid: &uuid::Uuid) -> Option<hecs::Entity> {
        self.uuid_to_entity.get(uuid).copied()
    }

    /// Get the stable UUID for an hecs Entity.
    pub(crate) fn get_entity_uuid(&self, entity: hecs::Entity) -> uuid::Uuid {
        self.entity_uuids.get(&entity).copied()
            .unwrap_or_else(uuid::Uuid::nil)
    }

    /// Generate a unique entity name. If `base` already exists, appends a number.
    pub(crate) fn unique_name(&self, base: &str) -> String {
        let existing: std::collections::HashSet<String> = self.world
            .query::<&crate::components::EditorMetadata>()
            .iter()
            .map(|(_, m)| m.name.clone())
            .collect();
        if !existing.contains(base) {
            return base.to_string();
        }
        for i in 1.. {
            let candidate = format!("{base} ({i})");
            if !existing.contains(&candidate) {
                return candidate;
            }
        }
        base.to_string()
    }

    /// Extract an intelligent display name from an asset path.
    /// Uses parent directory name if the filename is generic (scene, model, etc.).
    pub(crate) fn display_name_from_path(path: &str) -> String {
        let p = std::path::Path::new(path);
        let stem = p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // If the filename is generic, use the parent directory name.
        let generic_names = ["scene", "model", "mesh", "object", "default", "untitled"];
        if generic_names.iter().any(|g| stem.eq_ignore_ascii_case(g)) {
            if let Some(parent) = p.parent().and_then(|p| p.file_name()) {
                let parent_name = parent.to_string_lossy().into_owned();
                // Don't use generic parent names either.
                if !generic_names.iter().any(|g| parent_name.eq_ignore_ascii_case(g))
                    && parent_name != "objects" && parent_name != "assets" && parent_name != "models"
                {
                    return parent_name;
                }
            }
        }
        stem
    }

    /// Spawn an .arvx asset at a world-space position. The passed `pos`
    /// is interpreted as the surface point the user wants the asset to
    /// stand on — the asset's AABB bottom is snapped there (i.e.
    /// `transform.position.y = pos.y - info.aabb.min.y`), matching
    /// rkifield's drop-on-geometry behaviour.
    pub(crate) fn spawn_asset(&mut self, path: &str, pos: glam::Vec3) {
        let _ = self.spawn_asset_ex(path, pos, true);
    }

    /// Spawn an .arvx asset and return (entity, aabb_min_y) — the latter
    /// is cached by the drag-preview so every subsequent pick-result
    /// update can apply the same AABB-bottom snap without reloading
    /// the asset info. `verbose` gates the console log; drag-preview
    /// spawns are noisy without it.
    pub(crate) fn spawn_asset_ex(&mut self, path: &str, pos: glam::Vec3, verbose: bool)
        -> Option<(hecs::Entity, f32)>
    {
        use crate::components::*;
        let acquired = self.scene_mgr.lock().unwrap().acquire_asset(path);
        match acquired {
            Ok((handle, info)) => {
                let raw_name = Self::display_name_from_path(path);
                let name = self.unique_name(&raw_name);
                let spatial = spatial_from_handle(
                    &info.spatial, info.voxel_size, &info.aabb, info.grid_origin,
                    info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new(),
                );
                let mut transform = Transform::default();
                transform.position = glam::Vec3::new(pos.x, pos.y - info.aabb.min.y, pos.z);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    Renderable {
                        asset_path: Some(path.to_string()),
                        voxel_count: info.voxel_count,
                        spatial: Some(crate::components::RenderGeometry::Octree(spatial)),
                        asset_handle: Some(handle),
                        ..Default::default()
                    },
                ));
                self.assign_entity_uuid(entity);
                self.geometry_dirty.mark_all();
                self.scene_dirty.mark_entity(entity);
                self.gpu_objects_dirty.mark_all();
                if verbose {
                    self.console.info(format!("Loaded '{name}': {} voxels", info.voxel_count));
                    // User-committed spawn — auto-select so the
                    // Properties / Inspector panel reflects the new
                    // object without an extra click. The verbose=false
                    // path (drag-preview ghost) deliberately skips this
                    // to avoid the selection flicker; the drop site in
                    // DragPreviewCommit selects the final entity there.
                    self.selected_entity = Some(entity);
                }
                Some((entity, info.aabb.min.y))
            }
            Err(e) => {
                self.console.error(format!("Failed to load '{path}': {e}"));
                None
            }
        }
    }

    /// Assign a stable UUID to an entity.
    pub(crate) fn assign_entity_uuid(&mut self, entity: hecs::Entity) -> uuid::Uuid {
        let uuid = uuid::Uuid::new_v4();
        self.entity_uuids.insert(entity, uuid);
        self.uuid_to_entity.insert(uuid, entity);
        // Fresh spawns append to the end of the tree. `entry` makes
        // the assignment idempotent: if a caller pre-seeded an order
        // (e.g. scene-load from a persisted value), we keep it.
        self.entity_tree_order.entry(entity).or_insert_with(|| {
            let key = self.next_tree_order;
            self.next_tree_order += 1.0;
            key
        });
        uuid
    }

    /// Bind an entity to a specific (pre-existing) UUID. Used by scene
    /// load so entities keep the ID they had when the scene was saved,
    /// not a fresh random one. Keeping IDs stable is what lets paths
    /// derived from UUIDs — like procedural bake sidecars — survive a
    /// reload.
    pub(crate) fn set_entity_uuid(&mut self, entity: hecs::Entity, uuid: uuid::Uuid) {
        self.entity_uuids.insert(entity, uuid);
        self.uuid_to_entity.insert(uuid, entity);
    }

    pub(crate) fn delete_entity(&mut self, entity: hecs::Entity) {
        // Get name for logging.
        let name = self.world.get::<&crate::components::EditorMetadata>(entity)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());

        // Phase 5.6: capture the stamp's AABB before despawn so the
        // post-delete sync invalidates the right tiles. Pre-delete
        // because the Stamp component is gone after `world.despawn`.
        let stamp_aabb_to_invalidate =
            self.capture_stamp_aabb_before_delete(entity);
        // Phase 7: same idea for regions — capture pre-delete so we
        // can invalidate the tiles the deleted region was covering.
        let region_aabb_to_invalidate =
            self.capture_region_aabb_before_delete(entity);

        // If this is a generator entity, cancel any in-flight run and
        // recursively delete its owned children first. Children hold
        // their own pool allocations that need the standard cleanup
        // path, so we route them back through `delete_entity` rather
        // than raw despawn.
        let owned_children: Vec<hecs::Entity> = if let Some(pu) = self.entity_uuids.get(&entity).copied() {
            self.world
                .query::<&crate::generator::GeneratorOwned>()
                .iter()
                .filter(|(_, owned)| owned.parent_uuid == pu)
                .map(|(e, _)| e)
                .collect()
        } else {
            Vec::new()
        };
        self.generator_system.forget(entity);
        self.pending_generator_slot_keys.remove(&entity);
        for child in owned_children {
            self.delete_entity(child);
        }

        // Release geometry. Asset-backed entities go through the cache so
        // their leaf_attr/brick/octree ranges only free on the last instance
        // release. Procedural entities (no asset_handle) free their own
        // octree + leaf_attr range + brick ids via `deallocate_geometry`.
        if let Ok(renderable) = self.world.get::<&crate::components::Renderable>(entity) {
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

        // Clear selection if this was selected.
        if self.selected_entity == Some(entity) {
            self.selected_entity = None;
        }

        // Reparent children to root (remove Parent component).
        let entity_uuid = self.entity_uuids.get(&entity).copied();
        if let Some(uuid) = entity_uuid {
            let children: Vec<hecs::Entity> = self.world
                .query::<&crate::components::Parent>()
                .iter()
                .filter(|(_, p)| p.parent_id == uuid)
                .map(|(e, _)| e)
                .collect();
            for child in children {
                let _ = self.world.remove_one::<crate::components::Parent>(child);
            }
        }

        // Remove UUID mappings.
        if let Some(uuid) = self.entity_uuids.remove(&entity) {
            self.uuid_to_entity.remove(&uuid);
        }
        self.entity_tree_order.remove(&entity);
        // PERF_DEBT.md D2/D3: removing an entity with a non-empty
        // overlay/sculpt drops bytes out of the concatenated GPU
        // buffer (and shifts every later slice). Mark both dirty
        // when the entry actually existed so an idle entity remove
        // doesn't flip the flag unnecessarily.
        if self.paint_overlays.remove(&entity).is_some_and(|o| !o.is_empty()) {
            self.gpu_instance_overlays_dirty = true;
        }
        if self.sculpt_overlays.remove(&entity).is_some_and(|s| !s.is_empty()) {
            self.gpu_instance_sculpts_dirty = true;
        }
        // Drop any cached painted-material walk results so the next
        // flat-rebuild doesn't carry phantom leaves for this entity.
        // The dirty set entry (if present) would also resolve to a
        // remove on the next walk via `world.contains`, but pulling
        // it out here keeps the dirty set tight.
        self.painted_per_entity.remove(&entity);
        self.painted_dirty_entities.remove(&entity);
        self.painted_dirty_regions.remove(&entity);

        // Phase A1 mutation event — see docs/PERF_DEBT.md. Scaffolding
        // only (no consumers yet).
        self.mutation_log.push(
            super::mutation_log::MutationEvent::EntityRemoved { entity },
        );

        // Despawn from ECS.
        let _ = self.world.despawn(entity);

        // Phase 5.6: if this was a stamp, the Terrain.stamps index now
        // contains a stale entry. Re-sync from the live ECS (the
        // entity is gone, so the new index won't include it) and
        // invalidate the AABB we captured before despawn.
        if let Some(aabb) = stamp_aabb_to_invalidate {
            self.sync_terrain_stamps_and_invalidate(Some(aabb));
        }
        // Phase 7: mirror for regions.
        if let Some(aabb) = region_aabb_to_invalidate {
            self.sync_terrain_regions_and_invalidate(Some(aabb));
        }

        self.console.info(format!("Deleted '{name}'"));
        self.geometry_dirty.mark_all();
        self.scene_dirty.mark_entity(entity);
        self.gpu_objects_dirty.mark_all();
    }

    /// Duplicate an entity — copies every registered component via the
    /// component registry's serialize/deserialize round-trip, so any
    /// component type the registry knows about (including gameplay
    /// components from the hot-reloaded dylib) is carried across
    /// automatically. Replaces the previous hand-maintained whitelist,
    /// which silently dropped ProceduralGeometry, SpotLight, RigidBody,
    /// Skeleton, AnimationPlayer, and any user-added gameplay component.
    pub(crate) fn duplicate_entity(&mut self, source: hecs::Entity) {
        use crate::components::*;

        // Capture the source's name so we can stamp a unique one on the copy.
        let src_name = self.world.get::<&EditorMetadata>(source)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());
        let new_name = self.unique_name(&src_name);

        // Phase 1 (read-only): serialize every present component into JSON.
        // We materialise an owned (name, json) vec so the registry borrow is
        // dropped before we take &mut world to spawn/insert.
        let pairs: Vec<(String, String)> = {
            let entries = self.registry.components_on(&self.world, source);
            entries
                .iter()
                .filter_map(|e| {
                    (e.serialize)(&self.world, source).map(|json| (e.name.to_string(), json))
                })
                .collect()
        };

        // Phase 2: spawn empty entity, re-insert each serialized component.
        let entity = self.world.spawn(());
        for (name, json) in &pairs {
            if let Some(entry) = self.registry.get(name) {
                if let Err(err) = (entry.deserialize_insert)(&mut self.world, entity, json) {
                    self.console.warn(format!(
                        "duplicate_entity: component '{name}' failed to clone: {err}"
                    ));
                }
            }
        }

        // Phase 3: stamp unique identity. Transform is left exactly
        // as the source — the copy occupies the same world position
        // and only becomes visible once the user moves it (this
        // matches DCC tools like Blender/Maya where Ctrl+D stacks).
        if let Ok(mut md) = self.world.get::<&mut EditorMetadata>(entity) {
            md.name = new_name.clone();
        }
        self.assign_entity_uuid(entity);

        // Phase 4: hydrate runtime-only fields the registry's
        // serialize/deserialize couldn't carry over.
        //
        // `Renderable.spatial` and `Renderable.asset_handle` are
        // `#[serde(skip)]` — they're references into the scene
        // manager's runtime pools and the asset cache, not data we
        // store on disk. After serde-deserialize the cloned entity
        // has `asset_path: Some(...)` but neither a spatial nor a
        // handle, so it would render as empty space. Re-acquire the
        // asset to bump its refcount and populate the runtime
        // fields. Mirrors the load-from-disk path in `load_scene`.
        let asset_path_to_acquire = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| r.asset_path.clone());
        if let Some(asset_path) = asset_path_to_acquire {
            let full_path = self
                .project_dir
                .as_ref()
                .map(|d| d.join("assets").join(&asset_path))
                .unwrap_or_else(|| std::path::PathBuf::from(&asset_path));
            let acquired = self
                .scene_mgr
                .lock()
                .unwrap()
                .acquire_asset(&full_path.to_string_lossy());
            match acquired {
                Ok((handle, info)) => {
                    let new_spatial = spatial_from_handle(
                        &info.spatial,
                        info.voxel_size,
                        &info.aabb,
                        info.grid_origin,
                        info.leaf_attr_slot_start,
                        info.leaf_attr_slot_count,
                        Vec::new(),
                    );
                    if let Ok(mut r) = self.world.get::<&mut Renderable>(entity) {
                        r.spatial = Some(crate::components::RenderGeometry::Octree(new_spatial));
                        r.asset_handle = Some(handle);
                        r.voxel_count = info.voxel_count;
                    }
                }
                Err(e) => {
                    self.console.warn(format!(
                        "Duplicate: couldn't acquire asset '{asset_path}' for clone of '{src_name}': {e}",
                    ));
                }
            }
        }

        // Procedural duplicates: the source's bake-cache file is
        // keyed by the SOURCE entity's UUID, not the new one, so
        // the new entity has no on-disk cache and no runtime
        // spatial yet. Mark it dirty so the bake_worker schedules
        // a fresh bake on the next tick — same treatment a brand-
        // new procedural gets.
        if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
            pg.dirty = true;
        }

        self.selected_entity = Some(entity);

        self.console.info(format!("Duplicated '{src_name}' → '{new_name}'"));
        self.geometry_dirty.mark_all();
        self.scene_dirty.mark_entity(entity);
        self.gpu_objects_dirty.mark_all();
    }

    pub(crate) fn clear_scene(&mut self) {
        self.world.clear();
        self.entity_uuids.clear();
        self.uuid_to_entity.clear();
        self.next_entity_uuid = 1;
        std::sync::Arc::make_mut(&mut self.gpu_assets).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instances).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instance_overlays).clear();
        std::sync::Arc::make_mut(&mut self.gpu_instance_sculpts).clear();
        self.gpu_to_entity.clear();
        // PERF_DEBT.md D2/D3: clearing wipes all overlay/sculpt
        // entries; mark dirty so the snapshot ships a non-empty
        // DirtyRanges (which write_with_dirty resolves to a single
        // upload of the post-clear empty buffer — or skipped entirely
        // if the buffer length is zero).
        if !self.paint_overlays.is_empty() {
            self.gpu_instance_overlays_dirty = true;
        }
        if !self.sculpt_overlays.is_empty() {
            self.gpu_instance_sculpts_dirty = true;
        }
        self.paint_overlays.clear();
        self.sculpt_overlays.clear();
        self.painted_per_entity.clear();
        self.painted_dirty_entities.clear();
        self.painted_dirty_regions.clear();
        // Phase A1 mutation event for project/scene reset. Scaffolding
        // only — see docs/PERF_DEBT.md.
        self.mutation_log.push(super::mutation_log::MutationEvent::WorldReset);
        // `clear()` wipes every pool but preserves the epoch atomic
        // identity — replacing the whole manager here would orphan
        // sim's `geometry_epoch_handle`, breaking the lock-free
        // epoch read so render never sees future bumps and stops
        // uploading geometry (everything renders as the raw AABB
        // cubes after a project close+open).
        self.scene_mgr.lock().unwrap().clear(1_000_000);
        self.selected_entity = None;
        self.geometry_dirty.mark_all();
        self.scene_dirty.mark_all();
        self.gpu_objects_dirty.mark_all();
    }
}
