//! Import profile — per-asset import settings stored as `.rkimport` sidecar files.
//!
//! When a mesh file (e.g. `bunny.glb`) is imported, its settings are stored
//! alongside it as `bunny.glb.rkimport`. This lets the user tweak resolution,
//! rotation, naming, etc. and re-import without losing their configuration.

use std::path::{Path, PathBuf};

use rkf_import::pipeline::ImportConfig;
use serde::{Deserialize, Serialize};

/// Per-asset import profile — serialized to `.rkimport` JSON sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportProfile {
    /// Display name override. None = auto-derive from filename.
    #[serde(default)]
    pub display_name: Option<String>,
    /// Finest voxel size in metres. None = auto-detect from mesh extent.
    #[serde(default)]
    pub voxel_size: Option<f32>,
    /// Normalize longest axis to this size (metres).
    #[serde(default = "default_target_size")]
    pub target_size: f32,
    /// Skip normalization — keep original mesh coordinates.
    #[serde(default)]
    pub no_normalize: bool,
    /// Sample per-voxel color from mesh textures.
    #[serde(default = "default_true")]
    pub import_colors: bool,
    /// Euler rotation offset in degrees [X, Y, Z].
    #[serde(default)]
    pub rotation_offset: [f32; 3],
    /// Uniform scale multiplier (applied after normalization).
    #[serde(default)]
    pub scale_override: Option<f32>,
}

fn default_target_size() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

impl Default for ImportProfile {
    fn default() -> Self {
        Self {
            display_name: None,
            voxel_size: None,
            target_size: 1.0,
            no_normalize: false,
            import_colors: true,
            rotation_offset: [0.0; 3],
            scale_override: None,
        }
    }
}

impl ImportProfile {
    /// Convert to the rkf-import ImportConfig used by the import pipeline.
    pub fn to_import_config(&self) -> ImportConfig {
        ImportConfig {
            voxel_size: self.voxel_size,
            lod_levels: 1,
            target_size: self.target_size,
            no_normalize: self.no_normalize,
            material_id_override: None,
            import_colors: self.import_colors,
            rotation_offset: self.rotation_offset,
            scale_override: self.scale_override,
            pool_size: 65536,
            verbose: true,
        }
    }

    /// Sidecar path for a source mesh file (e.g. `bunny.glb` → `bunny.glb.rkimport`).
    pub fn sidecar_path(source: &Path) -> PathBuf {
        let mut p = source.as_os_str().to_owned();
        p.push(".rkimport");
        PathBuf::from(p)
    }

    /// Load from sidecar, or return default if not found.
    pub fn load_or_default(source: &Path) -> Self {
        let sidecar = Self::sidecar_path(source);
        if sidecar.exists() {
            match std::fs::read_to_string(&sidecar) {
                Ok(json) => match serde_json::from_str(&json) {
                    Ok(profile) => return profile,
                    Err(e) => eprintln!(
                        "[ImportProfile] parse error {}: {e}",
                        sidecar.display()
                    ),
                },
                Err(e) => eprintln!(
                    "[ImportProfile] read error {}: {e}",
                    sidecar.display()
                ),
            }
        }
        Self::default()
    }

    /// Save to sidecar file.
    pub fn save_for(&self, source: &Path) -> Result<(), String> {
        let sidecar = Self::sidecar_path(source);
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&sidecar, &json)
            .map_err(|e| format!("write {}: {e}", sidecar.display()))
    }
}
