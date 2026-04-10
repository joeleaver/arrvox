//! Material library — manages `.rkmat` asset files and the runtime material palette.
//!
//! Each `.rkmat` file is a JSON-serialized `MaterialDef`. The library scans a
//! directory, assigns stable u16 IDs (alphabetical order for determinism), and
//! builds a `Vec<GpuMaterial>` palette for GPU upload.
//!
//! Slot 0 is always the built-in default material (not backed by a file).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rkp_render::rkp_shade::GpuMaterial;
use serde::{Deserialize, Serialize};

// ── On-disk format (.rkmat) ──────────────────────────────────────────────

fn default_base_color() -> [f32; 4] {
    [0.8, 0.8, 0.8, 1.0]
}
fn default_roughness() -> f32 {
    0.5
}
fn default_opacity() -> f32 {
    1.0
}

/// Material definition — serialized to/from `.rkmat` JSON files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialDef {
    pub name: String,
    /// Path to a `.rkshader` file (future use — stored but not compiled yet).
    #[serde(default)]
    pub shader: Option<String>,
    #[serde(default = "default_base_color")]
    pub base_color: [f32; 4],
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metallic: f32,
    #[serde(default)]
    pub emission_strength: f32,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    /// Shader-specific parameters (future use).
    #[serde(default)]
    pub shader_params: HashMap<String, serde_json::Value>,
}

impl Default for MaterialDef {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            shader: None,
            base_color: default_base_color(),
            roughness: default_roughness(),
            metallic: 0.0,
            emission_strength: 0.0,
            opacity: default_opacity(),
            shader_params: HashMap::new(),
        }
    }
}

impl MaterialDef {
    /// Convert to the 32-byte GPU struct.
    pub fn to_gpu(&self) -> GpuMaterial {
        GpuMaterial {
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            emission_strength: self.emission_strength,
            opacity: self.opacity,
        }
    }
}

// ── UI info snapshot ─────────────────────────────────────────────────────

/// Lightweight material info for the editor UI.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaterialInfo {
    pub id: u16,
    pub name: String,
    /// Path relative to project root (e.g. "assets/materials/wood.rkmat").
    pub path: String,
    pub base_color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emission_strength: f32,
    pub opacity: f32,
}

// ── Internal slot representation ─────────────────────────────────────────

enum MaterialSlot {
    /// Slot 0 only — the built-in default material.
    Default,
    /// Loaded from a `.rkmat` file.
    Loaded { path: PathBuf, def: MaterialDef },
    /// File was deleted — renders as default, ID never reused in this session.
    Tombstone,
}

// ── Material library ─────────────────────────────────────────────────────

/// Manages the mapping from `.rkmat` asset paths to runtime u16 IDs.
pub struct MaterialLibrary {
    slots: Vec<MaterialSlot>,
    path_to_slot: HashMap<PathBuf, u16>,
    /// GPU palette needs re-upload.
    dirty: bool,
    /// UI needs updated material list.
    ui_dirty: bool,
    /// Root directory for materials (e.g. `<project>/assets/materials/`).
    materials_dir: Option<PathBuf>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self {
            slots: vec![MaterialSlot::Default],
            path_to_slot: HashMap::new(),
            dirty: true,
            ui_dirty: true,
            materials_dir: None,
        }
    }

    /// Scan a directory recursively for `.rkmat` files.
    /// Clears existing loaded materials and reassigns IDs in alphabetical order.
    pub fn scan(&mut self, materials_dir: &Path) {
        self.materials_dir = Some(materials_dir.to_owned());

        // Collect all .rkmat files.
        let mut paths = Vec::new();
        Self::collect_rkmat_files(materials_dir, &mut paths);
        paths.sort();

        // Clear existing loaded materials (keep slot 0 = Default).
        self.slots.truncate(1);
        self.path_to_slot.clear();

        // Load each file and assign sequential IDs.
        for path in paths {
            match Self::load_rkmat(&path) {
                Ok(def) => {
                    let id = self.slots.len() as u16;
                    self.path_to_slot.insert(path.clone(), id);
                    self.slots.push(MaterialSlot::Loaded { path, def });
                }
                Err(e) => {
                    eprintln!(
                        "[MaterialLibrary] failed to load {}: {e}",
                        path.display()
                    );
                }
            }
        }

        eprintln!(
            "[MaterialLibrary] scanned {} materials",
            self.slots.len() - 1
        );
        self.dirty = true;
        self.ui_dirty = true;
    }

    /// Reload a single `.rkmat` file. If the file is new, assigns a new slot.
    /// If it already has a slot, updates in place. If the file doesn't exist,
    /// tombstones the slot.
    pub fn reload(&mut self, path: &Path) {
        let canonical = path.to_owned();

        if !path.exists() {
            // File deleted — tombstone if we had it.
            self.remove(&canonical);
            return;
        }

        match Self::load_rkmat(path) {
            Ok(def) => {
                if let Some(&slot_id) = self.path_to_slot.get(&canonical) {
                    // Existing slot — update in place.
                    self.slots[slot_id as usize] =
                        MaterialSlot::Loaded { path: canonical, def };
                    eprintln!(
                        "[MaterialLibrary] reloaded slot {slot_id}: {}",
                        path.display()
                    );
                } else {
                    // New file — assign next slot.
                    let id = self.slots.len() as u16;
                    self.path_to_slot.insert(canonical.clone(), id);
                    self.slots.push(MaterialSlot::Loaded {
                        path: canonical,
                        def,
                    });
                    eprintln!(
                        "[MaterialLibrary] added new material as slot {id}: {}",
                        path.display()
                    );
                }
                self.dirty = true;
                self.ui_dirty = true;
            }
            Err(e) => {
                eprintln!(
                    "[MaterialLibrary] failed to reload {}: {e}",
                    path.display()
                );
            }
        }
    }

    /// Tombstone a material slot (file was deleted).
    pub fn remove(&mut self, path: &Path) {
        if let Some(&slot_id) = self.path_to_slot.get(path) {
            self.slots[slot_id as usize] = MaterialSlot::Tombstone;
            self.path_to_slot.remove(path);
            eprintln!(
                "[MaterialLibrary] tombstoned slot {slot_id}: {}",
                path.display()
            );
            self.dirty = true;
            self.ui_dirty = true;
        }
    }

    /// Create a new material with the given name, write it to disk, assign a slot.
    /// Returns the new material's u16 ID.
    pub fn create(&mut self, name: &str) -> Result<u16, String> {
        let dir = self
            .materials_dir
            .as_ref()
            .ok_or_else(|| "no materials directory set".to_string())?;
        let _ = std::fs::create_dir_all(dir);

        // Generate a unique filename from the name.
        let slug = name
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>();
        let mut filename = format!("{slug}.rkmat");
        let mut counter = 2u32;
        while dir.join(&filename).exists() {
            filename = format!("{slug}_{counter}.rkmat");
            counter += 1;
        }

        let path = dir.join(&filename);
        let def = MaterialDef {
            name: name.to_string(),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&def)
            .map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, &json).map_err(|e| format!("write {}: {e}", path.display()))?;

        let id = self.slots.len() as u16;
        self.path_to_slot.insert(path.clone(), id);
        self.slots.push(MaterialSlot::Loaded { path, def });
        self.dirty = true;
        self.ui_dirty = true;

        eprintln!("[MaterialLibrary] created material '{name}' as slot {id}");
        Ok(id)
    }

    /// Save a material's current definition back to its `.rkmat` file.
    pub fn save(&self, id: u16) -> Result<(), String> {
        match self.slots.get(id as usize) {
            Some(MaterialSlot::Loaded { path, def }) => {
                let json = serde_json::to_string_pretty(def)
                    .map_err(|e| format!("serialize: {e}"))?;
                std::fs::write(path, &json)
                    .map_err(|e| format!("write {}: {e}", path.display()))?;
                Ok(())
            }
            _ => Err(format!("slot {id} is not a loaded material")),
        }
    }

    /// Look up the slot ID for a given file path. Returns 0 (default) if not found.
    pub fn id_for_path(&self, path: &Path) -> u16 {
        self.path_to_slot.get(path).copied().unwrap_or(0)
    }

    /// Look up the file path for a given slot ID.
    pub fn path_for_id(&self, id: u16) -> Option<&Path> {
        match self.slots.get(id as usize) {
            Some(MaterialSlot::Loaded { path, .. }) => Some(path),
            _ => None,
        }
    }

    /// Get an immutable reference to a material definition by ID.
    pub fn get_def(&self, id: u16) -> Option<&MaterialDef> {
        match self.slots.get(id as usize) {
            Some(MaterialSlot::Loaded { def, .. }) => Some(def),
            _ => None,
        }
    }

    /// Get a mutable reference to a material definition by ID.
    pub fn get_def_mut(&mut self, id: u16) -> Option<&mut MaterialDef> {
        match self.slots.get_mut(id as usize) {
            Some(MaterialSlot::Loaded { def, .. }) => Some(def),
            _ => None,
        }
    }

    /// Build the GPU palette array. Slot 0 = default, tombstoned slots = default.
    pub fn build_palette(&self) -> Vec<GpuMaterial> {
        let default_gpu = MaterialDef::default().to_gpu();
        self.slots
            .iter()
            .map(|slot| match slot {
                MaterialSlot::Default | MaterialSlot::Tombstone => default_gpu,
                MaterialSlot::Loaded { def, .. } => def.to_gpu(),
            })
            .collect()
    }

    /// Build lightweight info for the UI. Excludes slot 0 (default) and tombstones.
    pub fn build_info(&self) -> Vec<MaterialInfo> {
        let project_root = self
            .materials_dir
            .as_ref()
            .and_then(|d| d.parent()) // assets/
            .and_then(|d| d.parent()); // project root

        let mut infos = Vec::new();

        // Include slot 0 (default) so the UI can show it.
        infos.push(MaterialInfo {
            id: 0,
            name: "Default".into(),
            path: String::new(),
            base_color: default_base_color(),
            roughness: default_roughness(),
            metallic: 0.0,
            emission_strength: 0.0,
            opacity: default_opacity(),
        });

        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if let MaterialSlot::Loaded { path, def } = slot {
                let rel_path = project_root
                    .and_then(|root| path.strip_prefix(root).ok())
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());

                infos.push(MaterialInfo {
                    id: i as u16,
                    name: def.name.clone(),
                    path: rel_path,
                    base_color: def.base_color,
                    roughness: def.roughness,
                    metallic: def.metallic,
                    emission_strength: def.emission_strength,
                    opacity: def.opacity,
                });
            }
        }

        infos
    }

    /// Whether the GPU palette needs re-upload.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the GPU dirty flag.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Whether the UI needs an updated material list.
    pub fn is_ui_dirty(&self) -> bool {
        self.ui_dirty
    }

    /// Clear the UI dirty flag.
    pub fn clear_ui_dirty(&mut self) {
        self.ui_dirty = false;
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn collect_rkmat_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::collect_rkmat_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rkmat") {
                out.push(path);
            }
        }
    }

    fn load_rkmat(path: &Path) -> Result<MaterialDef, String> {
        let json =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&json).map_err(|e| format!("parse {}: {e}", path.display()))
    }
}

// ── Starter materials ────────────────────────────────────────────────────

/// Starter material definitions written to new projects.
/// Each is a (filename, MaterialDef) pair.
fn starter_materials() -> Vec<(&'static str, MaterialDef)> {
    vec![
        (
            "stone.rkmat",
            MaterialDef {
                name: "Stone".into(),
                base_color: [0.55, 0.53, 0.50, 1.0],
                roughness: 0.85,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "wood.rkmat",
            MaterialDef {
                name: "Wood".into(),
                base_color: [0.55, 0.35, 0.18, 1.0],
                roughness: 0.75,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "metal.rkmat",
            MaterialDef {
                name: "Metal".into(),
                base_color: [0.77, 0.78, 0.78, 1.0],
                roughness: 0.25,
                metallic: 1.0,
                ..Default::default()
            },
        ),
        (
            "gold.rkmat",
            MaterialDef {
                name: "Gold".into(),
                base_color: [1.0, 0.84, 0.0, 1.0],
                roughness: 0.3,
                metallic: 1.0,
                ..Default::default()
            },
        ),
        (
            "brick.rkmat",
            MaterialDef {
                name: "Brick".into(),
                base_color: [0.65, 0.30, 0.22, 1.0],
                roughness: 0.9,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "glass.rkmat",
            MaterialDef {
                name: "Glass".into(),
                base_color: [0.9, 0.95, 1.0, 1.0],
                roughness: 0.05,
                metallic: 0.0,
                opacity: 0.3,
                ..Default::default()
            },
        ),
        (
            "emissive.rkmat",
            MaterialDef {
                name: "Emissive".into(),
                base_color: [1.0, 0.95, 0.8, 1.0],
                roughness: 0.5,
                metallic: 0.0,
                emission_strength: 5.0,
                ..Default::default()
            },
        ),
        (
            "white.rkmat",
            MaterialDef {
                name: "White".into(),
                base_color: [0.95, 0.95, 0.95, 1.0],
                roughness: 0.5,
                metallic: 0.0,
                ..Default::default()
            },
        ),
    ]
}

/// Write starter `.rkmat` files into a project's materials directory.
/// Skips any files that already exist (won't overwrite user edits).
pub fn write_starter_materials(materials_dir: &Path) {
    let _ = std::fs::create_dir_all(materials_dir);
    for (filename, def) in starter_materials() {
        let path = materials_dir.join(filename);
        if path.exists() {
            continue;
        }
        match serde_json::to_string_pretty(&def) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, &json) {
                    eprintln!("[MaterialLibrary] failed to write starter {filename}: {e}");
                }
            }
            Err(e) => {
                eprintln!("[MaterialLibrary] failed to serialize starter {filename}: {e}");
            }
        }
    }
}
