//! Material library — manages `.arvxmat` asset files and the runtime material palette.
//!
//! Each `.arvxmat` file is a JSON-serialized `MaterialDef`. The library scans a
//! directory, assigns stable u16 IDs (alphabetical order for determinism), and
//! builds a `Vec<GpuMaterial>` palette for GPU upload.
//!
//! Slot 0 is always the built-in default material (not backed by a file).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use arvx_render::arvx_shade::GpuMaterial;
use serde::{Deserialize, Serialize};

// ── On-disk format (.arvxmat) ──────────────────────────────────────────────

fn default_albedo() -> [f32; 3] {
    [0.8, 0.8, 0.8]
}
fn default_roughness() -> f32 {
    0.5
}
fn default_emission_color() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}
fn default_subsurface_color() -> [f32; 3] {
    [1.0, 0.8, 0.6]
}
fn default_opacity() -> f32 {
    1.0
}
fn default_ior() -> f32 {
    1.5
}

/// Material definition — serialized to/from `.arvxmat` JSON files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialDef {
    pub name: String,
    /// Path to a `.arvxshader` file (future use — stored but not compiled yet).
    #[serde(default)]
    pub shader: Option<String>,

    // PBR baseline
    #[serde(default = "default_albedo")]
    pub albedo: [f32; 3],
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "default_emission_color")]
    pub emission_color: [f32; 3],
    #[serde(default)]
    pub emission_strength: f32,

    // Subsurface + translucency
    #[serde(default)]
    pub subsurface: f32,
    #[serde(default = "default_subsurface_color")]
    pub subsurface_color: [f32; 3],
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    #[serde(default = "default_ior")]
    pub ior: f32,

    // Procedural variation
    #[serde(default)]
    pub noise_scale: f32,
    #[serde(default)]
    pub noise_strength: f32,
    /// Bit 0 = albedo, bit 1 = roughness, bit 2 = normal perturbation.
    #[serde(default)]
    pub noise_channels: u32,

    /// Shader-specific parameters (future use).
    #[serde(default)]
    pub shader_params: HashMap<String, serde_json::Value>,
}

impl Default for MaterialDef {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            shader: None,
            albedo: default_albedo(),
            roughness: default_roughness(),
            metallic: 0.0,
            emission_color: default_emission_color(),
            emission_strength: 0.0,
            subsurface: 0.0,
            subsurface_color: default_subsurface_color(),
            opacity: default_opacity(),
            ior: default_ior(),
            noise_scale: 0.0,
            noise_strength: 0.0,
            noise_channels: 0,
            shader_params: HashMap::new(),
        }
    }
}

impl MaterialDef {
    /// Convert to the 96-byte GPU struct, resolving `shader: Option<String>`
    /// to a numeric `shader_id` via the supplied resolver. The resolver
    /// is typically `|name| registry.resolve(name)` from
    /// [`arvx_render::shader_composer::UserShaderRegistry`]. Missing or
    /// unregistered shader names produce `shader_id = 0`, which the
    /// shade-pass dispatcher's identity arm handles as the standard
    /// PBR path.
    pub fn to_gpu(
        &self,
        shader_id_resolver: &dyn Fn(&str) -> Option<u32>,
        instance_shader_id_resolver: &dyn Fn(&str) -> Option<u32>,
    ) -> GpuMaterial {
        let shader_id = self
            .shader
            .as_deref()
            .and_then(shader_id_resolver)
            .unwrap_or(0);
        let instance_shader_id = self
            .shader
            .as_deref()
            .and_then(instance_shader_id_resolver)
            .unwrap_or(0);
        GpuMaterial {
            albedo: self.albedo,
            roughness: self.roughness,
            metallic: self.metallic,
            emission_color: self.emission_color,
            emission_strength: self.emission_strength,
            subsurface: self.subsurface,
            subsurface_color: self.subsurface_color,
            opacity: self.opacity,
            ior: self.ior,
            noise_scale: self.noise_scale,
            noise_strength: self.noise_strength,
            noise_channels: self.noise_channels,
            shader_id,
            instance_shader_id,
            _padding: [0.0; 4],
        }
    }
}

// ── UI info snapshot ─────────────────────────────────────────────────────

/// Lightweight material info for the editor UI.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaterialInfo {
    pub id: u16,
    pub name: String,
    /// Path relative to project root (e.g. "assets/materials/wood.arvxmat").
    pub path: String,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission_color: [f32; 3],
    pub emission_strength: f32,
    pub subsurface: f32,
    pub subsurface_color: [f32; 3],
    pub opacity: f32,
    pub ior: f32,
    pub noise_scale: f32,
    pub noise_strength: f32,
    pub noise_channels: u32,
    /// User-shader name reference (matches a stem in
    /// `<project>/assets/shaders/`). `None` means "use built-in PBR".
    pub shader: Option<String>,
    /// Current values for shader-declared `@param` slots, keyed by
    /// param name. The editor renders one slider per param using the
    /// shader's `ParamDef` schema; missing keys fall back to the
    /// shader's declared default.
    pub shader_params: std::collections::HashMap<String, f32>,
}

// ── Internal slot representation ─────────────────────────────────────────

enum MaterialSlot {
    /// Slot 0 only — the built-in default material. Immutable fallback;
    /// not editable. Users create real materials via `create()`.
    Default,
    /// Loaded from a `.arvxmat` file.
    Loaded { path: PathBuf, def: MaterialDef },
    /// File was deleted — renders as default, ID never reused in this session.
    Tombstone,
}

// ── Material library ─────────────────────────────────────────────────────

/// Manages the mapping from `.arvxmat` asset paths to runtime u16 IDs.
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

    /// Scan a directory recursively for `.arvxmat` files.
    /// Clears existing loaded materials and reassigns IDs in alphabetical order.
    pub fn scan(&mut self, materials_dir: &Path) {
        self.materials_dir = Some(materials_dir.to_owned());

        // Collect all .arvxmat files.
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

    /// Reload a single `.arvxmat` file. If the file is new, assigns a new slot.
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
        let mut filename = format!("{slug}.arvxmat");
        let mut counter = 2u32;
        while dir.join(&filename).exists() {
            filename = format!("{slug}_{counter}.arvxmat");
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

    /// Save a material's current definition back to its `.arvxmat` file.
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

    /// Project root the library was scanned against, derived as the
    /// `materials_dir`'s grandparent (`materials_dir` is conventionally
    /// `<root>/assets/materials/`). `None` until `scan` has been
    /// called or if the layout doesn't conform.
    pub fn project_root(&self) -> Option<&Path> {
        self.materials_dir
            .as_ref()
            .and_then(|d| d.parent()) // assets/
            .and_then(|d| d.parent()) // project root
    }

    /// Look up the file path for a given slot ID.
    pub fn path_for_id(&self, id: u16) -> Option<&Path> {
        match self.slots.get(id as usize) {
            Some(MaterialSlot::Loaded { path, .. }) => Some(path),
            _ => None,
        }
    }

    /// Get an immutable reference to a material definition by ID.
    /// Total slot count including the default (slot 0) and tombstoned
    /// slots. Iteration `0..slot_count()` covers every id ever
    /// assigned in the current session.
    pub fn slot_count(&self) -> usize { self.slots.len() }

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
    /// `shader_id_resolver` resolves each material's `shader: Option<String>`
    /// to a numeric dispatch id (typically `|n| registry.resolve(n)` from
    /// the engine's `UserShaderRegistry`). Pass `&|_| None` for an
    /// "all identity" palette in tests or pre-registry init.
    pub fn build_palette(
        &self,
        shader_id_resolver: &dyn Fn(&str) -> Option<u32>,
        instance_shader_id_resolver: &dyn Fn(&str) -> Option<u32>,
    ) -> Vec<GpuMaterial> {
        let default_gpu = MaterialDef::default()
            .to_gpu(shader_id_resolver, instance_shader_id_resolver);
        self.slots
            .iter()
            .map(|slot| match slot {
                MaterialSlot::Default | MaterialSlot::Tombstone => default_gpu,
                MaterialSlot::Loaded { def, .. } => {
                    def.to_gpu(shader_id_resolver, instance_shader_id_resolver)
                }
            })
            .collect()
    }

    /// Build the per-material shader-params buffer — one fixed-size
    /// 8 × f32 slot per material, parallel to `build_palette`. For each
    /// material, looks up its shader in the registry and packs its
    /// `MaterialDef.shader_params` values in the order the shader's
    /// metadata declared them. Missing values fall back to the
    /// shader's declared default. Materials with no shader (or an
    /// unregistered shader name) get an all-zeros slot.
    ///
    /// Phase A produces this buffer; Phase B's shade pipeline binds
    /// it and reads `params[material_id][i]` for each named param.
    pub fn build_shader_params(
        &self,
        registry: &arvx_render::shader_composer::UserShaderRegistry,
    ) -> Vec<[f32; 8]> {
        let default_def = MaterialDef::default();
        let pack = |def: &MaterialDef| -> [f32; 8] {
            let Some(name) = def.shader.as_deref() else {
                return [0.0; 8];
            };
            // Find the registered shader by name; if it isn't
            // registered, all slots zero (the shade dispatcher will
            // fall through to identity anyway since shader_id == 0).
            let Some(entry) = registry.entries().iter().find(|e| e.name == name)
            else {
                return [0.0; 8];
            };
            let mut out = [0.0f32; 8];
            for (i, param) in entry.metadata.params.iter().take(8).enumerate() {
                out[i] = def
                    .shader_params
                    .get(&param.name)
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(param.default);
            }
            out
        };
        self.slots
            .iter()
            .map(|slot| match slot {
                MaterialSlot::Default | MaterialSlot::Tombstone => pack(&default_def),
                MaterialSlot::Loaded { def, .. } => pack(def),
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
        let d = MaterialDef::default();
        infos.push(MaterialInfo {
            id: 0,
            name: "Default".into(),
            path: String::new(),
            albedo: d.albedo,
            roughness: d.roughness,
            metallic: d.metallic,
            emission_color: d.emission_color,
            emission_strength: d.emission_strength,
            subsurface: d.subsurface,
            subsurface_color: d.subsurface_color,
            opacity: d.opacity,
            ior: d.ior,
            noise_scale: d.noise_scale,
            noise_strength: d.noise_strength,
            noise_channels: d.noise_channels,
            shader: None,
            shader_params: std::collections::HashMap::new(),
        });

        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if let MaterialSlot::Loaded { path, def } = slot {
                let rel_path = project_root
                    .and_then(|root| path.strip_prefix(root).ok())
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());

                let shader_params: std::collections::HashMap<String, f32> = def
                    .shader_params
                    .iter()
                    .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
                    .collect();
                infos.push(MaterialInfo {
                    id: i as u16,
                    name: def.name.clone(),
                    path: rel_path,
                    albedo: def.albedo,
                    roughness: def.roughness,
                    metallic: def.metallic,
                    emission_color: def.emission_color,
                    emission_strength: def.emission_strength,
                    subsurface: def.subsurface,
                    subsurface_color: def.subsurface_color,
                    opacity: def.opacity,
                    ior: def.ior,
                    noise_scale: def.noise_scale,
                    noise_strength: def.noise_strength,
                    noise_channels: def.noise_channels,
                    shader: def.shader.clone(),
                    shader_params,
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

    /// Mark both the GPU palette and UI material list as stale. Called
    /// after any external field edit via `get_def_mut`.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.ui_dirty = true;
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
            } else if path.extension().is_some_and(|e| e == "arvxmat") {
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

// ── MaterialLibraryLookup impl ───────────────────────────────────────────

/// Make the live library queryable via the trait `arvx-terrain` (and
/// any other procedural source crate) uses to resolve
/// `MaterialRef::Path` → slot id. Paths in the lookup are
/// **project-root-relative** (`"assets/materials/rock.arvxmat"`); we
/// join against `project_root()` to match the canonical absolute
/// paths in `path_to_slot`.
impl arvx_core::MaterialLibraryLookup for MaterialLibrary {
    fn resolve_path(&self, path: &Path) -> Option<u16> {
        // Already absolute? Hit the map directly.
        if path.is_absolute() {
            return self.path_to_slot.get(path).copied();
        }
        // Project-root-relative form — the common case for paths
        // authored in FBM defaults / scene JSON.
        let root = self.project_root()?;
        let abs = root.join(path);
        self.path_to_slot.get(&abs).copied()
    }
}

// ── Starter materials ────────────────────────────────────────────────────

/// Starter material definitions written to new projects.
/// Each is a (filename, MaterialDef) pair.
fn starter_materials() -> Vec<(&'static str, MaterialDef)> {
    vec![
        (
            "stone.arvxmat",
            MaterialDef {
                name: "Stone".into(),
                albedo: [0.55, 0.53, 0.50],
                roughness: 0.85,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "wood.arvxmat",
            MaterialDef {
                name: "Wood".into(),
                albedo: [0.55, 0.35, 0.18],
                roughness: 0.75,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "metal.arvxmat",
            MaterialDef {
                name: "Metal".into(),
                albedo: [0.77, 0.78, 0.78],
                roughness: 0.25,
                metallic: 1.0,
                ..Default::default()
            },
        ),
        (
            "gold.arvxmat",
            MaterialDef {
                name: "Gold".into(),
                albedo: [1.0, 0.84, 0.0],
                roughness: 0.3,
                metallic: 1.0,
                ..Default::default()
            },
        ),
        (
            "brick.arvxmat",
            MaterialDef {
                name: "Brick".into(),
                albedo: [0.65, 0.30, 0.22],
                roughness: 0.9,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "glass.arvxmat",
            MaterialDef {
                name: "Glass".into(),
                albedo: [0.9, 0.95, 1.0],
                roughness: 0.05,
                metallic: 0.0,
                opacity: 0.3,
                ..Default::default()
            },
        ),
        // Terrain-FBM material defaults — pair with
        // `FbmTerrainFn::default()` material paths so a fresh project's
        // procedural terrain renders correctly without any user setup.
        (
            "grass.arvxmat",
            MaterialDef {
                name: "Grass".into(),
                albedo: [0.32, 0.48, 0.18],
                roughness: 0.85,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "rock.arvxmat",
            MaterialDef {
                name: "Rock".into(),
                albedo: [0.42, 0.40, 0.38],
                roughness: 0.92,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "sand.arvxmat",
            MaterialDef {
                name: "Sand".into(),
                albedo: [0.76, 0.69, 0.50],
                roughness: 0.88,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "snow.arvxmat",
            MaterialDef {
                name: "Snow".into(),
                albedo: [0.93, 0.94, 0.96],
                roughness: 0.55,
                metallic: 0.0,
                ..Default::default()
            },
        ),
        (
            "emissive.arvxmat",
            MaterialDef {
                name: "Emissive".into(),
                albedo: [1.0, 0.95, 0.8],
                emission_color: [1.0, 0.95, 0.8],
                roughness: 0.5,
                metallic: 0.0,
                emission_strength: 5.0,
                ..Default::default()
            },
        ),
        (
            "white.arvxmat",
            MaterialDef {
                name: "White".into(),
                albedo: [0.95, 0.95, 0.95],
                roughness: 0.5,
                metallic: 0.0,
                ..Default::default()
            },
        ),
    ]
}

/// Write starter `.arvxmat` files into a project's materials directory.
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
