//! Scene file format — .rkscene files.
//!
//! A scene contains objects with transforms, asset references,
//! camera state, and lights. Serialized as JSON.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A scene file — serialized to `.rkscene` as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    pub objects: Vec<SceneObject>,
    pub camera: CameraState,
    #[serde(default)]
    pub lights: Vec<SceneLight>,
}

/// An object in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: Uuid,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub parent_id: Option<Uuid>,
    /// Path to the .rkp asset file (relative to project assets/).
    pub asset_path: Option<String>,
    /// Primitive type if this is an analytical object.
    pub primitive: Option<String>,
    pub material_id: u16,
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

/// A light in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLight {
    pub id: Uuid,
    pub name: String,
    pub light_type: String, // "point" or "spot"
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
        }
    }
}

/// Save a scene to disk.
pub fn save_scene(scene: &SceneFile, path: &std::path::Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(scene)
        .map_err(|e| format!("serialize scene: {e}"))?;
    std::fs::write(path, &json)
        .map_err(|e| format!("write scene: {e}"))?;
    eprintln!("[RkpEngine] saved scene to {}", path.display());
    Ok(())
}

/// Load a scene from disk.
pub fn load_scene(path: &std::path::Path) -> Result<SceneFile, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("read scene: {e}"))?;
    let scene: SceneFile = serde_json::from_str(&json)
        .map_err(|e| format!("parse scene: {e}"))?;
    eprintln!("[RkpEngine] loaded scene from {} ({} objects)", path.display(), scene.objects.len());
    Ok(scene)
}
