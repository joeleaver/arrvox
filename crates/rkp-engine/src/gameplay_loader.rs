//! Gameplay dylib loader — loads and hot-reloads the gameplay cdylib.
//!
//! The gameplay crate compiles as a `.so`/`.dylib`/`.dll` containing user-defined
//! components. The engine loads it at startup and reloads when the file changes.
//!
//! Hot-reload flow:
//! 1. Serialize all gameplay component data (per-entity, per-component JSON)
//! 2. Remove gameplay entries from the registry
//! 3. Unload old dylib
//! 4. Load new dylib
//! 5. Call `rkp_gameplay_entries()` to discover new component entries
//! 6. Add entries to the registry
//! 7. Deserialize component data back onto entities

use std::path::{Path, PathBuf};

use crate::behavior::SystemEntry;
use crate::component_registry::{ComponentEntry, ComponentRegistry};

/// Manages the gameplay dylib lifecycle.
pub struct GameplayLoader {
    /// Path to the dylib file.
    dylib_path: Option<PathBuf>,
    /// Loaded library handle.
    lib: Option<libloading::Library>,
    /// Component entries from the current dylib (borrowed from the lib's static memory).
    gameplay_entries: Vec<&'static ComponentEntry>,
    /// System entries from the current dylib.
    system_entries: Vec<&'static SystemEntry>,
    /// Last modification time of the dylib file.
    last_modified: Option<std::time::SystemTime>,
}

/// Saved component data for hot-reload (entity UUID → component name → JSON).
pub type SavedComponents = Vec<(uuid::Uuid, String, String)>;

impl GameplayLoader {
    pub fn new() -> Self {
        Self {
            dylib_path: None,
            lib: None,
            gameplay_entries: Vec::new(),
            system_entries: Vec::new(),
            last_modified: None,
        }
    }

    /// Attempt to load the gameplay dylib from the standard build output path.
    /// Returns the component entries that were discovered.
    pub fn load(&mut self, path: &Path) -> Result<&[&'static ComponentEntry], String> {
        eprintln!("[GameplayLoader] loading {}", path.display());

        // Record modification time.
        self.last_modified = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        self.dylib_path = Some(path.to_owned());

        // Load the library.
        let lib = unsafe {
            libloading::Library::new(path)
                .map_err(|e| format!("load dylib: {e}"))?
        };

        // Look up the entry point.
        type EntryFn = extern "C" fn() -> crate::gameplay_loader::GameplayEntries;
        let entries_fn: libloading::Symbol<EntryFn> = unsafe {
            lib.get(b"rkp_gameplay_entries")
                .map_err(|e| format!("symbol lookup: {e}"))?
        };

        // Call it to get the component entries.
        let result = entries_fn();
        let entries: Vec<&'static ComponentEntry> = unsafe {
            std::slice::from_raw_parts(result.ptr, result.len).to_vec()
        };

        eprintln!(
            "[GameplayLoader] discovered {} gameplay components: {}",
            entries.len(),
            entries.iter().map(|e| e.name).collect::<Vec<_>>().join(", "),
        );

        self.gameplay_entries = entries;

        // Also discover system entries (optional — older dylibs may not export this).
        type SystemsFn = extern "C" fn() -> GameplaySystems;
        let systems: Vec<&'static SystemEntry> = unsafe {
            if let Ok(systems_fn) = lib.get::<SystemsFn>(b"rkp_gameplay_systems") {
                let result = systems_fn();
                std::slice::from_raw_parts(result.ptr, result.len).to_vec()
            } else {
                Vec::new()
            }
        };

        if !systems.is_empty() {
            eprintln!(
                "[GameplayLoader] discovered {} gameplay systems: {}",
                systems.len(),
                systems.iter().map(|e| e.name).collect::<Vec<_>>().join(", "),
            );
        }

        self.system_entries = systems;
        self.lib = Some(lib);

        Ok(&self.gameplay_entries)
    }

    /// Check if the dylib has been modified since last load.
    pub fn needs_reload(&self) -> bool {
        let Some(ref path) = self.dylib_path else { return false };
        let Some(last) = self.last_modified else { return false };
        let Ok(meta) = std::fs::metadata(path) else { return false };
        let Ok(current) = meta.modified() else { return false };
        current > last
    }

    /// Get the currently loaded gameplay component entries.
    pub fn entries(&self) -> &[&'static ComponentEntry] {
        &self.gameplay_entries
    }

    /// Get the currently loaded gameplay system entries.
    pub fn system_entries(&self) -> &[&'static SystemEntry] {
        &self.system_entries
    }

    /// Whether a component name belongs to the gameplay dylib.
    pub fn is_gameplay_component(&self, name: &str) -> bool {
        self.gameplay_entries.iter().any(|e| e.name == name)
    }

    /// Serialize all gameplay component data from the ECS world.
    /// Returns a list of (entity_uuid, component_name, json_data).
    pub fn serialize_all(
        &self,
        world: &hecs::World,
        entity_uuids: &std::collections::HashMap<hecs::Entity, uuid::Uuid>,
    ) -> SavedComponents {
        let mut saved = Vec::new();
        for (entity, uuid) in entity_uuids {
            for entry in &self.gameplay_entries {
                if (entry.has)(world, *entity) {
                    if let Some(json) = (entry.serialize)(world, *entity) {
                        saved.push((*uuid, entry.name.to_string(), json));
                    }
                }
            }
        }
        saved
    }

    /// Deserialize saved component data back into the ECS world.
    pub fn deserialize_all(
        &self,
        world: &mut hecs::World,
        uuid_to_entity: &std::collections::HashMap<uuid::Uuid, hecs::Entity>,
        saved: &SavedComponents,
    ) -> u32 {
        let mut restored = 0u32;
        for (uuid, comp_name, json) in saved {
            let Some(&entity) = uuid_to_entity.get(uuid) else { continue };
            let Some(entry) = self.gameplay_entries.iter().find(|e| e.name == *comp_name) else {
                eprintln!("[GameplayLoader] component '{comp_name}' no longer exists after reload");
                continue;
            };
            match (entry.deserialize_insert)(world, entity, json) {
                Ok(()) => restored += 1,
                Err(e) => eprintln!("[GameplayLoader] restore {comp_name}: {e}"),
            }
        }
        restored
    }

    /// Remove all gameplay components from all entities.
    pub fn remove_all_gameplay_components(
        &self,
        world: &mut hecs::World,
        entity_uuids: &std::collections::HashMap<hecs::Entity, uuid::Uuid>,
    ) {
        for (entity, _) in entity_uuids {
            for entry in &self.gameplay_entries {
                if (entry.has)(world, *entity) {
                    let _ = (entry.remove)(world, *entity);
                }
            }
        }
    }

    /// Unload the current dylib. Must be called AFTER serializing and removing components.
    pub fn unload(&mut self) {
        self.gameplay_entries.clear();
        self.system_entries.clear();
        self.lib = None;
        eprintln!("[GameplayLoader] unloaded gameplay dylib");
    }

    /// Whether a dylib is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.lib.is_some()
    }

    /// Path to the currently loaded dylib.
    pub fn dylib_path(&self) -> Option<&Path> {
        self.dylib_path.as_deref()
    }
}

/// FFI-safe container for component entries returned by the gameplay dylib.
#[repr(C)]
pub struct GameplayEntries {
    pub ptr: *const &'static ComponentEntry,
    pub len: usize,
    _storage: Vec<&'static ComponentEntry>,
}

impl GameplayEntries {
    /// Construct from a collected inventory iterator.
    pub fn from_iter(entries: Vec<&'static ComponentEntry>) -> Self {
        Self {
            ptr: entries.as_ptr(),
            len: entries.len(),
            _storage: entries,
        }
    }
}

/// FFI-safe container for system entries returned by the gameplay dylib.
#[repr(C)]
pub struct GameplaySystems {
    pub ptr: *const &'static SystemEntry,
    pub len: usize,
    _storage: Vec<&'static SystemEntry>,
}

impl GameplaySystems {
    /// Construct from a collected inventory iterator.
    pub fn from_iter(entries: Vec<&'static SystemEntry>) -> Self {
        Self {
            ptr: entries.as_ptr(),
            len: entries.len(),
            _storage: entries,
        }
    }
}
