//! Engine state snapshot — plain data the engine publishes each tick.
//!
//! No GUI types, no signals, no rinch dependency. The engine pushes this
//! via a callback. The editor (or any client) converts it to whatever
//! reactive system it uses.

use glam::Vec3;
use uuid::Uuid;

/// Live progress of one in-flight mesh import. Reduced by the
/// engine from the raw `ImportEvent` stream emitted by `rkp-import`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportProgressInfo {
    /// Source path as a string (matches `importing_models` entries).
    pub source_path: String,
    /// Machine-friendly name of the most recently started stage
    /// (`load_mesh`, `build_bvh`, `voxelize_surface`, ...).
    pub stage: String,
    /// Human-readable status line to display alongside the progress bar.
    pub message: String,
    /// Work units completed within the current stage.
    pub done: u64,
    /// Total work units for the current stage, or 0 if indeterminate.
    pub total: u64,
    /// Warning messages accumulated so far. Surfaced in the console panel.
    pub warnings: Vec<String>,
    /// Set once [`ImportEvent::Error`] arrives — the import has failed
    /// but the completion message may not have been delivered yet.
    pub error: Option<String>,
}

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
    /// Project root directory as an absolute path string — only sent
    /// when it changes (project open / close). The UI uses this to
    /// strip the prefix from displayed paths so users see
    /// `assets/bunny.obj` instead of `/home/joe/dev/rkipatch/assets/bunny.obj`.
    /// Outer `Option` = "this tick carries a project_dir update";
    /// inner `Option` = "is a project loaded" (None on close).
    pub project_dir: Option<Option<String>>,
    /// Available model files — only sent when the list changes.
    pub available_models: Option<Vec<ModelInfo>>,
    /// Source paths currently being re-imported. Sent whenever the set
    /// changes — on submit (grows) or completion (shrinks). The UI uses
    /// this to show a progress indicator in place of the Re-import button.
    pub importing_models: Option<Vec<String>>,
    /// Live per-source import progress — sent every tick while any
    /// import is in flight, so the UI can render a real stage/progress
    /// bar instead of a spinner. `None` means "no imports active this
    /// tick, don't re-render".
    pub import_progress: Option<Vec<ImportProgressInfo>>,
    /// Editor layout blob round-tripped from `.rkproject`. Sent once on
    /// project open so the editor can hydrate its docking state; the
    /// outer `Option` is "is this tick carrying a layout update?", the
    /// inner `Option` is "was one stored?" (None = pre-persistence
    /// project, editor should reset to its default layout).
    pub editor_layout: Option<Option<String>>,
    /// Inspector data for the selected entity — sent when selection changes.
    pub inspector: Option<crate::inspector::InspectorSnapshot>,
    /// Component names that can be added to the selected entity.
    pub available_components: Option<Vec<String>>,
    /// Recent projects list (sent once on startup).
    pub recent_projects: Option<Vec<crate::recent_projects::RecentProject>>,
    /// Available materials — sent when the material list changes.
    pub materials: Option<Vec<crate::material_library::MaterialInfo>>,
    /// Currently selected material in the materials panel.
    pub selected_material: Option<u16>,
    /// Currently selected model path (for Asset Properties).
    pub selected_model: Option<String>,
    /// Environment settings (sent when changed or on first frame).
    pub environment: Option<crate::environment::EnvironmentSettings>,
    /// Procedural object snapshot for the selected entity (if it has ProceduralGeometry).
    pub procedural: Option<crate::procedural_snapshot::ProceduralSnapshot>,
    /// New console log entries since last tick.
    pub console_entries: Vec<crate::console::LogEntry>,
}

/// Info about an available model file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelInfo {
    /// Display name (from import profile, or filename without extension).
    pub name: String,
    /// Full path to the .rkp file.
    pub path: String,
    /// Source mesh path (.glb/.obj/.fbx) if this was auto-imported.
    pub source_path: String,
    /// File size in bytes.
    pub size: u64,
    /// Total shell voxel count read from the .rkp header. Displayed
    /// in the Asset Properties panel so users can judge LOD / storage
    /// tradeoffs at a glance. Zero if the header couldn't be read.
    pub voxel_count: u32,
    /// Import profile (for editing in Asset Properties).
    pub import_profile: Option<crate::import_profile::ImportProfile>,
}
