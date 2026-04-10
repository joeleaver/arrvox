//! Engine commands — the async API between the editor (or any client) and the engine.
//!
//! Commands are sent via `crossbeam::channel::Sender<EngineCommand>` and drained
//! each tick by the engine. They represent game-relevant state mutations: spawn,
//! delete, transform, load, render, physics. Editor-only state (selection, gizmo
//! mode, debug views, tool settings) is NOT here — it stays in the editor client.

use glam::Vec3;
use rkf_runtime::input::{InputKeyCode, InputMouseButton};
use uuid::Uuid;

/// A command sent to the engine from the editor or any other client.
///
/// The engine drains these once per tick. Commands are cheap to clone and safe
/// to send across threads.
#[derive(Debug, Clone)]
pub enum EngineCommand {
    // ── Object lifecycle ─────────────────────────────────────────────

    /// Spawn an analytical primitive (box, sphere, capsule, etc.).
    SpawnPrimitive {
        name: String,
    },

    /// Spawn a camera entity.
    SpawnCamera,

    /// Spawn a point light.
    SpawnPointLight,

    /// Spawn a spot light.
    SpawnSpotLight,

    /// Place an imported model at the camera position.
    PlaceModel {
        asset_path: String,
    },

    /// Delete an object by entity ID.
    DeleteObject {
        entity_id: Uuid,
    },

    /// Duplicate an object by entity ID.
    DuplicateObject {
        entity_id: Uuid,
    },

    // ── Transforms ───────────────────────────────────────────────────

    /// Set an object's local position.
    SetObjectPosition {
        entity_id: Uuid,
        position: Vec3,
    },

    /// Set an object's local rotation (Euler degrees).
    SetObjectRotation {
        entity_id: Uuid,
        rotation: Vec3,
    },

    /// Set an object's local scale.
    SetObjectScale {
        entity_id: Uuid,
        scale: Vec3,
    },

    /// Set parent-child relationship.
    SetParent {
        child: Uuid,
        new_parent: Option<Uuid>,
    },

    // ── Geometry operations ──────────────────────────────────────────

    /// Convert an analytical primitive to a voxelized object.
    ConvertToVoxel {
        object_id: Uuid,
        voxel_size: f32,
    },

    /// Remap material on a voxelized object.
    RemapMaterial {
        object_id: Uuid,
        from_material: u16,
        to_material: u16,
    },

    /// Set the material of an analytical primitive.
    SetPrimitiveMaterial {
        object_id: Uuid,
        material_id: u16,
    },

    // ── Materials ────────────────────────────────────────────────────

    /// Create a new material with the given name.
    CreateMaterial {
        name: String,
    },

    /// Update a field on a material definition.
    UpdateMaterialField {
        material_id: u16,
        field: String,
        value: String,
    },

    /// Delete a material by its runtime ID.
    DeleteMaterial {
        material_id: u16,
    },

    /// Assign a material to an entity (sets Renderable.material_id).
    AssignMaterial {
        entity_id: Uuid,
        material_id: u16,
    },

    /// Select a material in the materials panel.
    SelectMaterial {
        material_id: Option<u16>,
    },

    /// Select a model in the models panel (for Asset Properties).
    SelectModel {
        path: Option<String>,
    },

    /// Update a field on a model's import profile.
    UpdateImportField {
        /// Source mesh path (identifies which model).
        source_path: String,
        field: String,
        value: String,
    },

    /// Save the import profile and re-import the model.
    ReimportModel {
        source_path: String,
    },

    /// Update an environment setting.
    UpdateEnvironment {
        field: String,
        value: String,
    },

    // ── Sculpt / Paint ───────────────────────────────────────────────

    /// Apply a sculpt brush stroke.
    Sculpt {
        position: Vec3,
        normal: Vec3,
        radius: f32,
        strength: f32,
        mode: SculptMode,
    },

    /// Apply a paint brush stroke.
    Paint {
        position: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 3],
        strength: f32,
        mode: PaintMode,
    },

    // ── Asset I/O ────────────────────────────────────────────────────

    /// Import and place an asset from a file path.
    ImportAsset {
        source_path: String,
    },

    /// Load an .rkp asset directly into the scene.
    LoadAsset {
        path: String,
        position: Vec3,
    },

    /// Create a new project at the given path.
    NewProject {
        path: String,
    },

    /// Open an existing project (.rkproject file).
    OpenProject {
        path: String,
    },

    /// Load a scene from disk.
    LoadScene {
        path: String,
    },

    /// Save the current scene (None = overwrite current path).
    SaveScene {
        path: Option<String>,
    },

    /// Save the project metadata (.rkproject file).
    SaveProject,

    // ── Play mode ────────────────────────────────────────────────────

    /// Enter play mode — start physics + behaviors.
    PlayStart,

    /// Exit play mode — restore edit state.
    PlayStop,

    // ── ECS component mutations ──────────────────────────────────────

    /// Set a field on an entity's component.
    SetComponentField {
        entity_id: Uuid,
        component_name: String,
        field_name: String,
        value: String, // Serialized — engine deserializes via GameValue.
    },

    /// Add a component to an entity.
    AddComponent {
        entity_id: Uuid,
        component_name: String,
    },

    /// Remove a component from an entity.
    RemoveComponent {
        entity_id: Uuid,
        component_name: String,
    },

    // ── Camera ───────────────────────────────────────────────────────

    /// Set the engine camera position/rotation directly (from editor camera controls).
    SetCamera {
        position: Vec3,
        yaw: f32,
        pitch: f32,
        fov: f32,
    },

    // ── Viewport ─────────────────────────────────────────────────────

    /// Resize the render viewport.
    Resize {
        width: u32,
        height: u32,
    },

    /// Select an entity (for UI highlight and inspector).
    SelectEntity {
        entity_id: Uuid,
    },

    // ── Picking ───────────────────────────────────────────────────

    /// Pick the object at viewport pixel (x, y).
    /// Engine reads the G-buffer and updates selection.
    Pick {
        x: u32,
        y: u32,
    },

    // ── Raw input (fed from surface events) ────────────────────────

    /// Mouse moved — absolute position + delta in pixels.
    MouseMove {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
    },

    /// Mouse button pressed/released.
    MouseButton {
        button: InputMouseButton,
        pressed: bool,
    },

    /// Scroll wheel.
    Scroll {
        delta: f32,
    },

    /// Key pressed.
    KeyDown {
        key: InputKeyCode,
    },

    /// Key released.
    KeyUp {
        key: InputKeyCode,
    },

    /// Shut down the engine.
    Shutdown,
}

/// Sculpt brush mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptMode {
    Raise,
    Carve,
    Smooth,
    Flatten,
}

/// Paint brush mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    Color,
    Material,
    Erase,
}
