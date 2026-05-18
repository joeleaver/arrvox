//! `.arvxgen` generator presets — named param overrides.
//!
//! A preset is a JSON file that names a registered generator and a set
//! of parameter overrides. Drop one in the asset browser to spawn the
//! generator with those params instead of the defaults.
//!
//! File format:
//!
//! ```json
//! {
//!   "generator": "building",
//!   "name": "Tall Tower",
//!   "params": {
//!     "floors": 8.0,
//!     "width": 14.0,
//!     "depth": 10.0
//!   }
//! }
//! ```
//!
//! Fields not listed in `params` fall back to the generator's
//! `Default` impl. Field types must match the registered component's
//! `FieldType` (Float / Int / Bool / String / Vec3 / Color).
//!
//! No editor UX yet for creating / editing presets — hand-edit the
//! JSON file and the engine picks it up on the next project rescan.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorAssetConfig {
    /// Registered generator name (e.g. `"building"`). Must match an
    /// entry in the gameplay dylib's generator registry.
    pub generator: String,
    /// Display name in the models panel (e.g. `"Warehouse"`).
    pub name: String,
    /// Per-field overrides. Keys are field names on the generator's
    /// param component; values are JSON values matching the field's
    /// declared type. Missing fields use the component's `Default`.
    #[serde(default)]
    pub params: serde_json::Map<String, serde_json::Value>,
}

impl GeneratorAssetConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
}

/// One discovered preset on disk. Surfaced in the snapshot to the
/// editor so the models panel can render a row for it.
#[derive(Debug, Clone)]
pub struct GeneratorPresetInfo {
    /// Absolute path to the `.arvxgen` file.
    pub path: PathBuf,
    /// Display name from the file's `name` field.
    pub display_name: String,
    /// Generator the preset targets — useful for tooltips.
    pub generator_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_minimal() {
        let cfg = GeneratorAssetConfig {
            generator: "building".into(),
            name: "Warehouse".into(),
            params: {
                let mut m = serde_json::Map::new();
                m.insert("floors".into(), serde_json::json!(5.0));
                m.insert("width".into(), serde_json::json!(20.0));
                m
            },
        };
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let back: GeneratorAssetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generator, "building");
        assert_eq!(back.name, "Warehouse");
        assert_eq!(back.params.len(), 2);
    }

    #[test]
    fn missing_params_defaults_empty() {
        let json = r#"{ "generator": "hello", "name": "Hi" }"#;
        let cfg: GeneratorAssetConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.params.is_empty());
    }
}
