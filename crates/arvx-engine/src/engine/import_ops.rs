//! Mesh-import pipeline + asset hot-reload.
//!
//! Scans the project `assets/` tree for importable meshes, submits them
//! to the background import worker (`ImportWorker`), drains progress +
//! completion events, and reloads in-scene entities whose `.arvx` output
//! has been regenerated. Also owns the file-watcher wiring.

use super::state::EngineState;
use super::model_scan::spatial_from_handle;

impl EngineState {
    /// Scan for importable mesh files and auto-import any that don't have .arvx outputs.
    pub(crate) fn auto_import_meshes(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if !assets_dir.exists() { return; }

            // Scan recursively for mesh files.
            let mut meshes = Vec::new();
            Self::scan_meshes_recursive(&assets_dir, &mut meshes);

            for source in meshes {
                let output = crate::import_worker::arvx_output_path(&source);
                // Only import if .arvx doesn't exist or is older than source.
                let needs_import = if output.exists() {
                    let src_mod = std::fs::metadata(&source)
                        .and_then(|m| m.modified()).ok();
                    let out_mod = std::fs::metadata(&output)
                        .and_then(|m| m.modified()).ok();
                    match (src_mod, out_mod) {
                        (Some(s), Some(o)) => s > o,
                        _ => true,
                    }
                } else {
                    true
                };

                if needs_import {
                    eprintln!("[ArvxEngine] auto-importing: {}", source.display());
                    self.import_worker.submit(crate::import_worker::ImportRequest {
                        source_path: source,
                        output_path: output,
                        config: crate::import_worker::default_import_config(),
                    });
                }
            }
        }
    }

    pub(crate) fn scan_meshes_recursive(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_meshes_recursive(&path, out);
            } else {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "glb" | "gltf" | "obj" | "fbx") {
                    out.push(path);
                }
            }
        }
    }

    pub(crate) fn init_file_watcher(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if assets_dir.exists() {
                match crate::file_watcher::ArvxFileWatcher::new(&[assets_dir.as_path()]) {
                    Ok(watcher) => {
                        self.file_watcher = Some(watcher);
                        eprintln!("[ArvxEngine] file watcher started on {}", assets_dir.display());
                    }
                    Err(e) => eprintln!("[ArvxEngine] file watcher failed: {e}"),
                }
            }
        }
    }

    pub(crate) fn process_file_events(&mut self) {
        let events = match self.file_watcher {
            Some(ref watcher) => watcher.poll_events(),
            None => return,
        };

        for event in events {
            use crate::file_watcher::FileEvent;
            match event {
                FileEvent::ModelChanged(path) => {
                    eprintln!("[ArvxEngine] model changed: {}", path.display());
                    self.scan_models();
                    let path_str = path.to_string_lossy().to_string();
                    self.reload_asset(&path_str);
                }
                FileEvent::ShaderChanged(path) => {
                    eprintln!("[ArvxEngine] shader changed: {}", path.display());
                    // Only `assets/shaders/*.wgsl` files participate in
                    // user-shader composition; other .wgsl in the project
                    // (engine assets, debug tools) are not composed and
                    // shouldn't trigger a registry rescan. Filter by the
                    // canonical shaders dir.
                    if let Some(shaders_dir) = self.shaders_dir() {
                        if path.starts_with(&shaders_dir) {
                            let _ = self.reload_user_shaders();
                        }
                    }
                }
                FileEvent::MaterialChanged(path) => {
                    eprintln!("[ArvxEngine] material changed: {}", path.display());
                    self.material_lib.reload(&path);
                }
                FileEvent::MeshSourceChanged(path) => {
                    eprintln!("[ArvxEngine] mesh source changed: {}", path.display());
                    let output = crate::import_worker::arvx_output_path(&path);
                    self.import_worker.submit(crate::import_worker::ImportRequest {
                        source_path: path,
                        output_path: output,
                        config: crate::import_worker::default_import_config(),
                    });
                }
                FileEvent::ScriptChanged(path) => {
                    eprintln!("[ArvxEngine] script changed: {}", path.display());
                    self.scaffold_and_build_gameplay();
                }
            }
        }
    }

    /// Drain queued `ImportEvent`s from the worker and reduce them
    /// into `importing_progress`. Called each tick before
    /// `poll_import_completions` so a completion's final
    /// `StageEnd` / `Error` event lands in `importing_progress`
    /// before the entry is removed on completion.
    pub(crate) fn pump_import_events(&mut self) {
        use crate::snapshot::ImportProgressInfo;
        use arvx_import::ImportEvent;

        let events = self.import_worker.poll_events();
        for tagged in events {
            let source_key = tagged.source_path.to_string_lossy().into_owned();
            let entry = self
                .importing_progress
                .entry(source_key.clone())
                .or_insert_with(|| ImportProgressInfo {
                    source_path: source_key,
                    ..Default::default()
                });
            match tagged.event {
                ImportEvent::StageStart { stage, message } => {
                    entry.stage = stage.to_string();
                    entry.message = message;
                    entry.done = 0;
                    entry.total = 0;
                }
                ImportEvent::StageProgress { stage, done, total } => {
                    // Ignore stale progress events from a stage that
                    // already ended (shouldn't happen given the
                    // worker is single-threaded, but cheap to guard).
                    if entry.stage == stage {
                        entry.done = done;
                        entry.total = total;
                    }
                }
                ImportEvent::StageEnd { stage } => {
                    if entry.stage == stage && entry.total > 0 {
                        entry.done = entry.total;
                    }
                }
                ImportEvent::Warn { message } => {
                    entry.warnings.push(message);
                }
                ImportEvent::Error { message } => {
                    entry.error = Some(message);
                }
            }
        }
    }

    pub(crate) fn poll_import_completions(&mut self) {
        let completions = self.import_worker.poll_completions();
        for completion in completions {
            let source_key = completion.source_path.to_string_lossy().into_owned();
            if self.importing_sources.remove(&source_key) {
                self.importing_dirty = true;
            }
            self.importing_progress.remove(&source_key);
            match completion.result {
                Ok(result) => {
                    let name = completion.source_path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    self.console.info(format!(
                        "Import complete: {name} ({} voxels)",
                        result.shell_voxels,
                    ));
                    self.refresh_reimported_asset(&completion.output_path);
                    self.scan_models();
                }
                Err(e) => {
                    let name = completion.source_path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    self.console.error(format!("Import failed: {name} — {e}"));
                }
            }
        }
    }

    pub(crate) fn reload_asset(&mut self, path: &str) {
        // Find any scene objects that reference this asset and reload them.
        // For now, log that we detected the change.
        eprintln!("[ArvxEngine] hot-reload asset: {path}");
        // TODO: remove old GPU objects for this asset, re-load from file,
        // rebuild faces, re-upload geometry.
    }

    /// After a re-import has rewritten the `.arvx` on disk, refresh the
    /// scene manager's cached copy and point any entities that were
    /// referencing it at the new geometry. No-op when the asset isn't
    /// currently loaded into the scene.
    pub(crate) fn refresh_reimported_asset(&mut self, output_path: &std::path::Path) {
        let path_str = output_path.to_string_lossy().into_owned();
        let reload = match self.scene_mgr.lock().unwrap().reload_asset(&path_str) {
            Ok(Some(r)) => r,
            Ok(None) => {
                eprintln!(
                    "[ArvxEngine] refresh_reimported_asset: {} not in asset cache — \
                     no scene entities to refresh",
                    output_path.display(),
                );
                return;
            }
            Err(e) => {
                self.console.error(format!("Reload after import failed: {e}"));
                return;
            }
        };

        let entities_to_update: Vec<hecs::Entity> = self.world
            .query::<&crate::components::Renderable>()
            .iter()
            .filter_map(|(e, r)| (r.asset_handle == Some(reload.old_handle)).then_some(e))
            .collect();

        eprintln!(
            "[ArvxEngine] refresh_reimported_asset: {} → {} entities to update \
             (old_handle={:?}, new_handle={:?}, voxels={})",
            output_path.display(),
            entities_to_update.len(),
            reload.old_handle,
            reload.new_handle,
            reload.info.voxel_count,
        );

        for entity in entities_to_update {
            if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                let spatial = spatial_from_handle(
                    &reload.info.spatial,
                    reload.info.voxel_size,
                    &reload.info.aabb,
                    reload.info.grid_origin,
                    reload.info.leaf_attr_slot_start,
                    reload.info.leaf_attr_slot_count,
                    Vec::new(),
                );
                r.asset_handle = Some(reload.new_handle);
                r.spatial = Some(crate::components::RenderGeometry::Octree(spatial));
                r.voxel_count = reload.info.voxel_count;
            }
        }
        // geometry_dirty: re-upload pools. gpu_objects_dirty: rebuild the
        // per-entity GpuObject list so the new AABB / octree offsets land
        // on the GPU (target_size, rotation offsets, etc. only show up in
        // the render once this runs).
        self.geometry_dirty.mark_all();
        self.gpu_objects_dirty.mark_all();
    }

    /// If a sibling `.arvxskel` exists alongside the `.arvx` path, load it
    /// into the animation cache and attach `Skeleton` + a default
    /// paused `AnimationPlayer` to the entity. Missing sidecar is not
    /// an error — static meshes are expected.
    pub(crate) fn try_attach_skeleton(&mut self, entity: hecs::Entity, arvx_path: &std::path::Path) {
        let rkskel_path = arvx_path.with_extension("arvxskel");
        if !rkskel_path.exists() {
            return;
        }
        match self.animation_cache.get_or_load(&rkskel_path) {
            Ok(asset) => {
                // Skeleton is transient — always attach/replace with the
                // freshly-loaded asset so the `current_pose` matches the
                // current bone count.
                //
                // Fold the grid-frame offset into the skeleton's
                // glTF→local transform: `rest_bone_aabbs` and the
                // scatter's `rest_pos` live in grid frame (octree
                // corner at 0, range [0, extent]), so the pose
                // produced by `animation::tick` must operate on grid-
                // frame positions too. The offset is
                // `half_extent = base_voxel_size × 2^depth / 2`,
                // available from the entity's spatial data (present
                // because `LoadAsset` populated `Renderable` before
                // this function runs).
                let grid_offset = self.world
                    .get::<&crate::components::Renderable>(entity)
                    .ok()
                    .and_then(|r| r.spatial.as_ref().and_then(|g| g.as_octree()).map(|s| {
                        let he = 0.5 * s.base_voxel_size * (1u32 << s.depth) as f32;
                        glam::Vec3::splat(he)
                    }))
                    .unwrap_or(glam::Vec3::ZERO);
                let skeleton = crate::animation::skeleton_component(
                    asset.clone(), rkskel_path.clone(), grid_offset,
                );
                if let Err(e) = self.world.insert_one(entity, skeleton) {
                    eprintln!("[ArvxEngine] attach Skeleton: world.insert_one failed: {e}");
                    return;
                }
                // Only attach a default `AnimationPlayer` if the entity
                // doesn't already have one — scene load may have
                // deserialized a persisted player with user-chosen clip
                // / time / loop mode.
                let has_player = self.world.get::<&crate::components::AnimationPlayer>(entity).is_ok();
                if !has_player {
                    let player = crate::animation::default_player(&asset);
                    if let Err(e) = self.world.insert_one(entity, player) {
                        eprintln!("[ArvxEngine] attach AnimationPlayer: world.insert_one failed: {e}");
                    }
                }
                eprintln!(
                    "[ArvxEngine] attached skeleton ({} bones, {} clips) from {}",
                    asset.skeleton.bones.len(),
                    asset.clips.len(),
                    rkskel_path.display(),
                );
            }
            Err(e) => {
                self.console.warn(format!("load .arvxskel failed: {e}"));
            }
        }
    }
}
