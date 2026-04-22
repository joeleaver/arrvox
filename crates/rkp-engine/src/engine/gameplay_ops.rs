//! Gameplay dylib hot-reload orchestration.
//!
//! Scaffolds a fresh `gameplay/` crate inside a project, invokes cargo
//! to build the cdylib, loads/unloads the resulting library via
//! `GameplayLoader`, and reacts to file-watcher events that require a
//! rebuild. Actual dylib parsing lives in `crate::gameplay_loader`;
//! these methods are the EngineState-side glue.

use super::state::EngineState;

impl EngineState {
    /// Scaffold the gameplay crate from project scripts and trigger a build.
    pub(crate) fn scaffold_and_build_gameplay(&mut self) {
        let Some(ref project_dir) = self.project_dir else { return };

        // Create assets/scripts directories if they don't exist (new projects).
        let scripts_dir = project_dir.join("assets/scripts");
        let _ = std::fs::create_dir_all(scripts_dir.join("components"));
        let _ = std::fs::create_dir_all(scripts_dir.join("systems"));

        // Generate the gameplay crate.
        match crate::scaffold::generate_gameplay_crate(project_dir) {
            Ok(crate_dir) => {
                self.console.info("Scaffolded gameplay crate");
                // Build the dylib.
                self.build_gameplay_crate(&crate_dir);
            }
            Err(e) => {
                self.console.error(format!("Scaffold failed: {e}"));
            }
        }
    }

    /// Build the scaffolded gameplay crate and load the resulting dylib.
    pub(crate) fn build_gameplay_crate(&mut self, crate_dir: &std::path::Path) {
        self.console.info("Building gameplay scripts...");
        let output = std::process::Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(crate_dir.join("Cargo.toml"))
            .output();

        match output {
            Ok(out) if out.status.success() => {
                self.console.info("Gameplay scripts compiled");
                // Load the built dylib.
                if let Some(ref project_dir) = self.project_dir {
                    let dylib_path = crate::scaffold::gameplay_dylib_path(project_dir);
                    self.load_gameplay_dylib(&dylib_path);
                }
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                self.console.error(format!("Gameplay build failed:\n{stderr}"));
            }
            Err(e) => {
                self.console.error(format!("Failed to run cargo: {e}"));
            }
        }
    }

    /// Load a gameplay dylib and register its components + systems.
    pub(crate) fn load_gameplay_dylib(&mut self, path: &std::path::Path) {
        if !path.exists() {
            return;
        }
        match self.gameplay_loader.load(path) {
            Ok(entries) => {
                let names: Vec<&str> = entries.iter().map(|e| e.name).collect();
                self.console.info(format!(
                    "Loaded gameplay: {} components ({})",
                    entries.len(),
                    names.join(", "),
                ));
                for &entry in entries {
                    self.registry.register_gameplay(entry);
                }
                self.gameplay_systems = self.gameplay_loader.system_entries().to_vec();
                if !self.gameplay_systems.is_empty() {
                    self.console.info(format!(
                        "Loaded {} gameplay systems",
                        self.gameplay_systems.len(),
                    ));
                }
                let gen_entries = self.gameplay_loader.generator_entries();
                if !gen_entries.is_empty() {
                    self.console.info(format!(
                        "Loaded {} generators: {}",
                        gen_entries.len(),
                        gen_entries.iter().map(|e| e.name).collect::<Vec<_>>().join(", "),
                    ));
                }
                self.generator_system.register_gameplay(gen_entries);
                self.generators_dirty = true;
                self.scene_dirty = true;
            }
            Err(e) => {
                self.console.error(format!("Failed to load gameplay dylib: {e}"));
            }
        }
    }

    /// Try to load an already-built gameplay dylib for the current project.
    pub(crate) fn try_load_gameplay_dylib(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let dylib_path = crate::scaffold::gameplay_dylib_path(project_dir);
            self.load_gameplay_dylib(&dylib_path);
        }
    }

    /// Check if the gameplay dylib needs hot-reloading.
    pub(crate) fn check_gameplay_reload(&mut self) {
        if !self.gameplay_loader.needs_reload() {
            return;
        }

        self.console.info("Hot-reloading gameplay dylib...");

        // 1. Serialize all gameplay component data.
        let saved = self.gameplay_loader.serialize_all(&self.world, &self.entity_uuids);
        self.console.info(format!("Serialized {} gameplay component instances", saved.len()));

        // 2. Remove all gameplay components from entities.
        self.gameplay_loader.remove_all_gameplay_components(&mut self.world, &self.entity_uuids);

        // 3. Clear gameplay entries from registry.
        self.registry.clear_gameplay();
        self.generator_system.clear_gameplay_generators();
        self.generators_dirty = true;

        // 4. Unload old dylib.
        let dylib_path = self.gameplay_loader.dylib_path().map(|p| p.to_owned());
        self.gameplay_loader.unload();

        // 5. Load new dylib.
        if let Some(path) = dylib_path {
            // Small delay to ensure the file is fully written.
            std::thread::sleep(std::time::Duration::from_millis(100));

            match self.gameplay_loader.load(&path) {
                Ok(entries) => {
                    let names: Vec<&str> = entries.iter().map(|e| e.name).collect();
                    self.console.info(format!(
                        "Reloaded: {} components ({})",
                        entries.len(),
                        names.join(", "),
                    ));

                    // 6. Re-register gameplay entries.
                    for &entry in entries {
                        self.registry.register_gameplay(entry);
                    }

                    // 6b. Re-register gameplay generators. Without
                    // this every live generator entity stays Pending
                    // forever after a reload — `scan_and_submit`
                    // looks them up by name and finds nothing.
                    let gen_entries = self.gameplay_loader.generator_entries();
                    self.generator_system.register_gameplay(gen_entries);
                    self.generators_dirty = true;
                    if !gen_entries.is_empty() {
                        self.console.info(format!(
                            "Reloaded {} generators: {}",
                            gen_entries.len(),
                            gen_entries
                                .iter()
                                .map(|e| e.name)
                                .collect::<Vec<_>>()
                                .join(", "),
                        ));
                    }

                    // 7. Deserialize component data back.
                    let restored = self.gameplay_loader.deserialize_all(
                        &mut self.world,
                        &self.uuid_to_entity,
                        &saved,
                    );
                    self.console.info(format!("Restored {restored}/{} component instances", saved.len()));

                    // 7b. Force every live generator to re-run against
                    // the new code. Param-hash equality wouldn't catch
                    // a code change, so we explicitly mark stale.
                    let live_generators: Vec<hecs::Entity> = self
                        .world
                        .query::<&crate::generator::GeneratorState>()
                        .iter()
                        .map(|(e, _)| e)
                        .collect();
                    for entity in live_generators {
                        self.generator_system.force_regenerate(entity, &mut self.world);
                    }

                    // 8. Reload system entries and rebuild executor.
                    self.gameplay_systems = self.gameplay_loader.system_entries().to_vec();
                    if let Some(ref mut executor) = self.behavior_executor {
                        if let Err(e) = executor.rebuild(&self.gameplay_systems) {
                            self.console.error(format!("Failed to rebuild system schedule: {e}"));
                        } else {
                            self.console.info(format!(
                                "Rebuilt schedule: {} systems",
                                self.gameplay_systems.len(),
                            ));
                        }
                    }
                }
                Err(e) => {
                    self.console.error(format!("Hot-reload failed: {e}"));
                }
            }
        }

        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }
}
