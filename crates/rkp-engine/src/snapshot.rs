//! Engine state snapshot — plain data the engine publishes each tick.
//!
//! No GUI types, no signals, no rinch dependency. The engine pushes this
//! via a callback. The editor (or any client) converts it to whatever
//! reactive system it uses.

use glam::Vec3;
use uuid::Uuid;

/// Lightweight scene object info for UI display.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneObjectInfo {
    pub id: Uuid,
    pub name: String,
    pub parent_id: Option<Uuid>,
    pub is_camera: bool,
    pub is_light: bool,
}

/// State the engine publishes at the end of each tick.
///
/// This is a plain struct — the engine doesn't know how the client
/// uses it. The client receives it via callback and can push to
/// reactive signals, log it, ignore it, etc.
#[derive(Debug, Clone)]
pub struct StateUpdate {
    pub fps: f32,
    pub gpu_object_count: u32,
    pub camera_position: Vec3,
    pub play_mode: bool,
    pub selected_entity: Option<Uuid>,
    /// Scene objects — only sent when the scene changes.
    /// `None` means "unchanged since last update."
    pub objects: Option<Vec<SceneObjectInfo>>,
    /// Project loaded state — only sent when it changes.
    pub project_loaded: Option<bool>,
    /// Project name — only sent when it changes.
    pub project_name: Option<String>,
    /// Available model files — only sent when the list changes.
    pub available_models: Option<Vec<ModelInfo>>,
    /// Inspector data for the selected entity — sent when selection changes.
    pub inspector: Option<crate::inspector::InspectorSnapshot>,
}

/// Info about an available model file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelInfo {
    /// Display name (filename without extension).
    pub name: String,
    /// Full path to the .rkp file.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
}
