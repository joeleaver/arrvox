//! Scene file format — .arvxscene files.
//!
//! A scene contains objects with transforms, asset references,
//! camera state, and lights. Serialized as JSON.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A scene file — serialized to `.arvxscene` as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub objects: Vec<SceneObject>,
    pub camera: CameraState,
    #[serde(default)]
    pub lights: Vec<SceneLight>,
    #[serde(default)]
    pub environment: Option<crate::environment::EnvironmentSettings>,
}

/// An object in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: Uuid,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    /// Scene-tree display order. `None` on old saves / brand-new
    /// objects; the engine reseeds its counter past the max loaded
    /// value so post-load spawns still append at the bottom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_order: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    /// Path to the .arvx asset file (relative to project assets/).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_path: Option<String>,
    /// Primitive type if this is an analytical object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive: Option<String>,
    /// Relative path (from the scene file's directory) to a `.arvx`
    /// sidecar holding the most recent procedural bake for this entity.
    /// Set when a procedural's bake worker persisted its artifact and
    /// that file still exists at save time; load uses it to restore
    /// the voxel data without auto-rebaking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub procedural_cache: Option<String>,
    #[serde(default)]
    pub material_id: u16,
    /// Per-voxel material remaps applied to this entity's shared
    /// asset/cache geometry. Each pair is `(original_material_id,
    /// current_material_id)`. Empty for entities where the user
    /// hasn't dragged a material onto them; non-empty entries are
    /// replayed via `remap_entity_material` after load.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub material_overrides: Vec<(u16, u16)>,
    /// PointLight component data (if entity has one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_light: Option<ScenePointLight>,
    /// Camera component data (if entity has one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<SceneCamera>,
    /// Generic component data — maps component name → JSON string.
    /// Used for gameplay components and any components not covered by
    /// the hardcoded fields above.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub components: std::collections::HashMap<String, String>,
}

/// Saved PointLight component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenePointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    #[serde(default = "default_true")]
    pub cast_shadow: bool,
}

fn default_true() -> bool { true }

/// Saved Camera component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneCamera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub active: bool,
}

/// Camera state stored in a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraState {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: [0.0, 2.0, 5.0],
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0,
        }
    }
}

/// A light in the scene (legacy format — new scenes use components on SceneObject).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLight {
    pub id: Uuid,
    pub name: String,
    pub light_type: String,
    pub position: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

impl SceneFile {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            camera: CameraState::default(),
            lights: Vec::new(),
            environment: None,
        }
    }
}

/// Save a scene to disk.
pub fn save_scene(scene: &SceneFile, path: &std::path::Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(scene)
        .map_err(|e| format!("serialize scene: {e}"))?;
    std::fs::write(path, &json)
        .map_err(|e| format!("write scene: {e}"))?;
    eprintln!("[ArvxEngine] saved scene to {}", path.display());
    Ok(())
}

/// Load a scene from disk.
pub fn load_scene(path: &std::path::Path) -> Result<SceneFile, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("read scene: {e}"))?;
    let scene: SceneFile = serde_json::from_str(&json)
        .map_err(|e| format!("parse scene: {e}"))?;
    eprintln!("[ArvxEngine] loaded scene from {} ({} objects)", path.display(), scene.objects.len());
    Ok(scene)
}
