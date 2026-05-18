//! Engine commands — the async API between the editor (or any client) and the engine.
//!
//! Commands are sent via `crossbeam::channel::Sender<EngineCommand>` and drained
//! each tick by the engine. They represent game-relevant state mutations: spawn,
//! delete, transform, load, render, physics. Editor-only state (selection, gizmo
//! mode, debug views, tool settings) is NOT here — it stays in the editor client.

use glam::Vec3;
use rkp_runtime::input::{InputKeyCode, InputMouseButton};
use uuid::Uuid;

use crate::viewport::ViewportId;

/// A command sent to the engine from the editor or any other client.
///
/// The engine drains these once per tick. Commands are cheap to clone and safe
/// to send across threads.
#[derive(Debug, Clone)]
pub enum EngineCommand {
    // ── Object lifecycle ─────────────────────────────────────────────

    /// Spawn a procedural object. `leaf_kind` picks the initial child under
    /// Root — any primitive name accepted by `parse_node_kind` (Sphere, Box,
    /// Capsule, Cylinder, Torus, Plane, Ramp). `None` defaults to Sphere.
    SpawnProceduralObject {
        name: String,
        leaf_kind: Option<String>,
    },

    /// Spawn a camera entity.
    SpawnCamera,

    /// Spawn a point light.
    SpawnPointLight,

    /// Spawn a spot light.
    SpawnSpotLight,

    /// Spawn a generator-driven entity. `generator_name` must match a
    /// registered generator from the gameplay dylib. The entity gets a
    /// Transform, EditorMetadata, GeneratorState, and a default instance
    /// of the generator's param component — the tick driver picks it up
    /// next frame and submits the first run.
    SpawnGenerator {
        generator_name: String,
    },

    /// Spawn a generator entity from a `.rkgen` preset on disk. The
    /// engine loads the preset's generator name + per-field overrides,
    /// spawns the entity with default params, then applies each
    /// override via `ComponentEntry::set_field`. Missing fields keep
    /// their default values.
    SpawnGeneratorPreset {
        /// Absolute path to the `.rkgen` file (round-tripped from
        /// `StateUpdate::available_generator_presets`).
        path: String,
    },

    /// Drop a generator at a viewport pixel. Drag-and-drop variant of
    /// `SpawnGenerator` — the engine issues a position-readback pick
    /// and spawns at the hit point (ground-plane fallback on sky).
    DropGenerator {
        id: ViewportId,
        generator_name: String,
        x: u32,
        y: u32,
    },

    /// Drop a `.rkgen` preset at a viewport pixel. Drag-and-drop
    /// variant of `SpawnGeneratorPreset`.
    DropGeneratorPreset {
        id: ViewportId,
        path: String,
        x: u32,
        y: u32,
    },

    /// Place an imported model at the camera position.
    PlaceModel {
        asset_path: String,
    },

    /// Delete an object by entity ID.
    DeleteObject {
        entity_id: Uuid,
    },

    /// Delete the currently selected object.
    DeleteSelected,

    /// Duplicate an object by entity ID.
    DuplicateObject {
        entity_id: Uuid,
    },

    /// Duplicate the currently selected object.
    DuplicateSelected,

    // ── Procedural editing ────────────────────────────────────────────

    /// Select a node within the currently selected procedural object.
    SelectProceduralNode {
        node_id: Option<u32>,
    },

    /// Add a child node to the selected procedural object.
    AddProceduralNode {
        parent_node_id: u32,
        kind: String,
    },

    /// Remove a node from the selected procedural object.
    RemoveProceduralNode {
        node_id: u32,
    },

    /// Move a procedural node earlier among its siblings.
    MoveProceduralNodeUp {
        node_id: u32,
    },

    /// Move a procedural node later among its siblings.
    MoveProceduralNodeDown {
        node_id: u32,
    },

    /// Reparent a procedural node to a different combinator.
    ReparentProceduralNode {
        node_id: u32,
        new_parent_id: u32,
    },

    /// Move a procedural node to a new parent at a specific child index.
    /// Supersedes the reparent/move-up/move-down triad for drag-and-drop:
    /// one command encodes both the destination parent and the visual
    /// insertion position. Same parent + different index = pure reorder.
    MoveProceduralNode {
        node_id: u32,
        new_parent_id: u32,
        index: u32,
    },

    /// Deep-clone a procedural node and its subtree, inserting the copy
    /// as the next sibling of the source. Selection moves to the clone.
    DuplicateProceduralNode {
        node_id: u32,
    },

    /// Change a combinator node's kind (Union / Intersect / Subtract)
    /// in place. Children, transform, and position in the tree are
    /// preserved; `material_combine` is carried across Union↔Intersect
    /// and defaulted to Winner when moving into Subtract (which has
    /// no material_combine). Ignored for leaf nodes / unknown kinds.
    SetProceduralNodeCombinator {
        node_id: u32,
        kind: String,
    },

    /// Set the render voxel size tier on the selected procedural object.
    /// Value must be one of: "0.005", "0.02", "0.08", "0.32".
    SetProceduralVoxelSize {
        tier: String,
    },

    /// Set the local position of a procedural node. Rotation + scale
    /// on the node are preserved.
    SetProceduralNodePosition {
        node_id: u32,
        position: Vec3,
    },

    /// Set the local rotation of a procedural node (Euler degrees,
    /// XYZ order). Position + scale are preserved.
    SetProceduralNodeRotation {
        node_id: u32,
        rotation_deg: Vec3,
    },

    /// Set the local scale factor of a procedural node. Position +
    /// rotation are preserved.
    SetProceduralNodeScale {
        node_id: u32,
        scale: Vec3,
    },

    /// Set a parameter on a procedural node.
    SetProceduralNodeParam {
        node_id: u32,
        param_name: String,
        value: String,
    },

    /// Voxelize the given procedural entity now, regardless of the
    /// auto-bake policy. Called by the "Bake" action in the build panel
    /// (or any other explicit request). Interactive edits mark the tree
    /// dirty but do not themselves trigger a bake — the user decides when
    /// to pay the voxelization cost.
    BakeProceduralEntity {
        entity_id: Uuid,
    },

    /// Voxelize every procedural entity whose tree is currently dirty.
    /// Convenience for "bake everything I've changed."
    BakeAllDirtyProcedurals,

    /// Convert a procedural entity in place: drop the
    /// `ProceduralGeometry`, keep the currently-baked voxels. No new
    /// entity, no extra GPU allocation — just a component removal.
    /// UI flags this as destructive and confirms with a modal.
    /// No-op (console warning) if the bake isn't clean.
    ConvertProceduralToVoxel {
        entity_id: Uuid,
    },

    /// Copy a procedural entity's baked voxels into a NEW voxel
    /// entity. The original stays procedural and editable; the copy
    /// is a static voxel object spawned next to it. Shares the same
    /// octree / leaf_attr / brick allocations via `refcount` on the
    /// asset cache... except this path doesn't go through the asset
    /// cache — it re-bakes the current tree into a fresh scene
    /// allocation owned by the new entity. Same gating as convert.
    CopyProceduralToNewVoxel {
        entity_id: Uuid,
    },

    /// Switch the build viewport's primary-visibility source between
    /// the voxel march (shows the baked result) and the procedural
    /// CSG raymarcher (shows the live tree, no bake required). The
    /// procedural being previewed is the currently-selected entity.
    SetBuildPreviewMode {
        mode: rkp_render::BuildPreviewMode,
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

    /// Scene-tree reorder: set `entity`'s parent and tree-order key
    /// in one step. The editor computes both values locally from
    /// the cursor's drop target, the target's expanded state, and
    /// its neighbors' orders — this command is a dumb applier.
    ///
    /// `new_parent = None` means "scene root" (clears the `Parent`
    /// component). `new_order` should sit between the surrounding
    /// siblings' tree_order values at the resolved parent level.
    /// Engine validates cycle + self-drop and applies; no ordering
    /// math here.
    ReorderEntity {
        entity: Uuid,
        new_parent: Option<Uuid>,
        new_order: f64,
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

    /// Set the user shader assigned to this material. `None` clears
    /// the assignment (PBR baseline). The string must match the file
    /// stem of a shader registered in
    /// `<project>/assets/shaders/<name>.wgsl`; unrecognized names map
    /// to `shader_id = 0` (identity dispatch) on the next material
    /// upload, so a typo renders as PBR rather than failing silently.
    SetMaterialShader {
        material_id: u16,
        shader_name: Option<String>,
    },

    /// Update one of the shader's named parameter values for this
    /// material. The param name comes from the shader's
    /// `// @param <name>: ...` schema; unknown names are accepted but
    /// the GPU side ignores them (the buffer only packs the schema's
    /// declared params).
    SetMaterialShaderParam {
        material_id: u16,
        name: String,
        value: f32,
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

    /// Set a view option (e.g., show_colliders).
    SetViewOption {
        option: String,
        enabled: bool,
    },

    /// Clear the console log.
    ClearConsole,

    /// Update an environment setting.
    UpdateEnvironment {
        field: String,
        value: String,
    },

    // ── Sculpt / Paint ───────────────────────────────────────────────

    /// Apply a sculpt brush stroke at an already-known world position.
    /// Used by tests and any caller that has already resolved the hit
    /// point. The normal UI flow uses [`SculptAtPixel`] instead, which
    /// drives a GPU pick readback to resolve position + entity.
    Sculpt {
        position: Vec3,
        normal: Vec3,
        radius: f32,
        strength: f32,
        mode: SculptMode,
    },

    /// Toggle sculpt mode + update the cached brush radius. Mirrors
    /// [`SetPaintActive`]. Mutually exclusive with paint mode — the
    /// editor's toggle handler clears the other before sending.
    SetSculptActive { active: bool, radius: f32 },

    /// Apply a sculpt brush stamp at a viewport pixel. Mirrors
    /// [`PaintAtPixel`]: the engine issues a GPU pick readback to
    /// resolve (entity, world_pos), then routes the result through
    /// the sculpt stamp path with these settings. Coalescing via
    /// single-in-flight pick is automatic.
    SculptAtPixel {
        id: ViewportId,
        x: u32,
        y: u32,
        radius: f32,
        /// 1-D falloff curve from `d/r ∈ [0, 1]` (center → rim) to
        /// strength `∈ [0, 1]`. Used by Inflate / Deflate to shape
        /// the per-cell thickness profile; Carve / Raise ignore it.
        falloff_curve: rkp_core::sculpt::FalloffCurve,
        /// Max-thickness amplitude in finest-voxel units (used by
        /// Inflate / Deflate; ignored by Carve / Raise).
        strength: f32,
        /// Monotonic stroke identifier — bumped by the editor on
        /// every LMB-down (start of a new sculpt stroke). The scene
        /// manager uses this to detect stroke boundaries and clear
        /// its per-stroke "already touched" cell set, so each new
        /// stroke starts fresh ("one layer per stroke" feel,
        /// matching Blender's no-accumulate brush behaviour). Stamps
        /// within the same stroke share the same value.
        stroke_seq: u64,
        mode: SculptMode,
        /// Material to assign to leaves that *transition* under the
        /// brush (Empty→Mixed for Raise, Interior→Mixed for newly-
        /// exposed surfaces in Carve). Pre-existing surface leaves
        /// keep their material — sculpt is not paint.
        material_id: u16,
    },

    /// Apply a paint brush stroke at an already-known world position.
    /// Used by tests and any caller that has already resolved the hit
    /// point. The normal UI flow uses [`PaintAtPixel`] instead, which
    /// drives a GPU pick readback to resolve position + entity.
    Paint {
        position: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 3],
        strength: f32,
        mode: PaintMode,
    },

    /// Toggle paint mode + update the cached brush radius. Editor
    /// sends this when the user clicks the Paint toolbar button or
    /// presses 'P', and on every radius-slider change once that ships.
    /// Engine uses this to drive the cursor wireframe (so it knows
    /// when and at what size to render the sphere) without re-reading
    /// editor state every frame.
    SetPaintActive { active: bool, radius: f32 },

    /// Apply a paint brush stamp at a viewport pixel. The engine issues
    /// a GPU pick readback to resolve the (entity, world_pos) under the
    /// cursor, then routes the result through `apply_paint_stamp` with
    /// these settings. Editor fires this on LMB press + every
    /// `MouseMove` while LMB held in paint mode. One pick is allowed
    /// in flight; repeated `PaintAtPixel` calls from a drag replace the
    /// prior in-flight settings (latest sample wins).
    PaintAtPixel {
        id: ViewportId,
        x: u32,
        y: u32,
        radius: f32,
        color: [f32; 3],
        strength: f32,
        /// Smoothstep shoulder [0, 1]. 0 = hard edge, 1 = smoothstep
        /// from center outward.
        falloff: f32,
        mode: PaintMode,
        /// Primary material id when `mode == Material`. Ignored otherwise.
        material_id: u16,
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

    /// Drop an .rkp asset at a viewport pixel. The engine issues a
    /// position-readback pick and spawns the asset at the hit point
    /// (ground plane Y=0 fallback if the ray missed geometry). The
    /// asset's AABB bottom is snapped onto the surface.
    DropAsset {
        id: ViewportId,
        path: String,
        x: u32,
        y: u32,
    },

    /// Begin a live drag-preview. Spawns the appropriate kind of
    /// scene object (asset, generator, or preset) as a preview and
    /// starts position-tracking the cursor via continuous pick
    /// readbacks. The preview follows the drop pixel's surface snap
    /// until `DragPreviewCommit` (drop) or `DragPreviewCancel`
    /// (dragleave). Source identifies what's being dragged; the engine
    /// dispatches on it to spawn the right entity type.
    DragPreviewEnter {
        id: ViewportId,
        source: DragPreviewSource,
        x: u32,
        y: u32,
    },

    /// Update the drag preview's target pixel. Fires on every `dragover`
    /// event; the engine coalesces these to one pick readback per frame
    /// via the existing `pending_pick` → `pick_in_flight` gate.
    DragPreviewOver {
        id: ViewportId,
        x: u32,
        y: u32,
    },

    /// Finalize the drag preview in-place — the preview stays wherever
    /// the last pick result placed it. No-op if no preview is active.
    DragPreviewCommit,

    /// Abort the drag preview: despawn the preview entity (and any
    /// generator-owned children) and clear drag state. No-op if no
    /// preview is active (so the `DragLeave` that fires after every
    /// `Drop` doesn't resurrect stale state).
    DragPreviewCancel,

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

    /// Cache editor-side layout state (docking, splitter sizes, etc.).
    /// Engine treats the payload as opaque JSON — the editor is the only
    /// code that understands it. Persisted to `.rkproject` on save and
    /// echoed back through `StateUpdate.editor_layout` on project open.
    SetEditorLayout {
        json: String,
    },

    // ── Play mode ────────────────────────────────────────────────────

    /// Set the gizmo mode (translate/rotate/scale).
    SetGizmoMode {
        mode: crate::gizmo::GizmoMode,
    },

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

    /// Set the editor camera state on the given viewport. The viewport's
    /// `editor_camera` is updated regardless of whether a runtime override
    /// is active — so on play-stop the camera lands where edit-mode left it.
    SetCamera {
        id: ViewportId,
        position: Vec3,
        yaw: f32,
        pitch: f32,
        fov: f32,
    },

    // ── Viewport ─────────────────────────────────────────────────────

    /// Resize a viewport's render target.
    Resize {
        id: ViewportId,
        width: u32,
        height: u32,
    },

    /// Toggle whether a viewport renders this frame. Hidden viewports skip
    /// the render pipeline entirely.
    SetViewportVisible {
        id: ViewportId,
        visible: bool,
    },

    /// Replace a viewport's `SceneFilter` — the layer mask it sees plus an
    /// optional always-included focus entity (matched by stable UUID).
    SetViewportFilter {
        id: ViewportId,
        base_layers: u32,
        focus_entity_id: Option<Uuid>,
    },

    /// Set a runtime camera override on a viewport — the viewport renders
    /// from `entity_id`'s Camera/Transform until the override is cleared.
    /// The viewport's `editor_camera` is preserved untouched.
    SetViewportCamera {
        id: ViewportId,
        entity_id: Uuid,
    },

    /// Clear a viewport's runtime override; rendering falls back to the
    /// persistent `editor_camera`.
    ClearViewportCamera {
        id: ViewportId,
    },

    /// Switch a viewport between full-pipeline (`InSitu`) and stripped
    /// preview (`Isolation`) rendering. Drives pass gating + grid overlay.
    SetViewportMode {
        id: ViewportId,
        mode: rkp_render::RenderMode,
    },

    /// Select an entity (for UI highlight and inspector).
    SelectEntity {
        entity_id: Uuid,
    },

    // ── Picking ───────────────────────────────────────────────────

    /// Pick the object at the given viewport's pixel (x, y).
    /// Engine reads that viewport's G-buffer and updates selection.
    Pick {
        id: ViewportId,
        x: u32,
        y: u32,
    },

    // ── Raw input (fed from surface events) ────────────────────────

    /// Mouse moved over a viewport — absolute position + delta in pixels.
    MouseMove {
        id: ViewportId,
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
    },

    /// Mouse button pressed/released over a viewport.
    MouseButton {
        id: ViewportId,
        button: InputMouseButton,
        pressed: bool,
    },

    /// Scroll wheel over a viewport.
    Scroll {
        id: ViewportId,
        delta: f32,
    },

    /// Key pressed (global — keys aren't viewport-scoped).
    KeyDown {
        key: InputKeyCode,
    },

    /// Key released (global).
    KeyUp {
        key: InputKeyCode,
    },

    /// Shut down the engine.
    Shutdown,
}

/// Sculpt brush mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SculptMode {
    /// Hard SDF union — fill the brush sphere with material.
    Raise,
    /// Hard SDF subtract — empty the brush sphere.
    Carve,
    /// Soft outward dilation — add a falloff-shaped thickness band
    /// outside the existing surface, masked by the brush radius.
    /// Blender's Draw / Inflate equivalent.
    Inflate,
    /// Soft inward erosion — remove a falloff-shaped thickness band
    /// from inside the existing surface, masked by the brush radius.
    /// The "soft Carve" most sculpt programs default to.
    Deflate,
    /// (P2) Normal-blend smoothing.
    Smooth,
    /// (P3) Flatten to local average plane.
    Flatten,
}

/// Paint brush mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    Color,
    Material,
    Erase,
}

/// What's being dragged into the viewport for a live preview. The
/// engine picks the spawn + snap rules based on this (models have a
/// known AABB so we bottom-snap; generators bake asynchronously and
/// use the raw surface position with child-tracking for self-hit).
#[derive(Debug, Clone)]
pub enum DragPreviewSource {
    /// `.rkp` asset file.
    Asset { path: String },
    /// Bare generator registered by the gameplay dylib.
    Generator { name: String },
    /// `.rkgen` preset on disk — loads a generator + overrides.
    GeneratorPreset { path: String },
}
