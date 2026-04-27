//! `EngineState` — the internal runtime state of `RkpEngine`.
//!
//! Single source of truth for the engine's per-tick state: ECS,
//! scene_mgr handles, render/bake workers, input, camera, project state,
//! gizmo state, per-frame dirty flags. Every `impl EngineState` block in
//! a sibling `engine/*_ops.rs` file mutates fields on this struct.
//!
//! The struct and all fields are `pub(crate)` so the sibling modules
//! (entity_ops, procedural_ops, command_handler, etc.) can access them
//! directly. Nothing outside the `rkp_engine` crate touches this type.

use rkp_render::rkp_gpu_object::RkpGpuObject;
use rkp_render::rkp_scene_manager::RkpSceneManager;

use crate::camera::CameraControlState;

use super::{CameraState, EngineConfig, FrameCallback};

/// A click-pick awaiting a G-buffer readback. The viewport tag
/// determines interpretation: MAIN resolves to a scene entity; BUILD
/// resolves to a procedural NodeId (when the build viewport is in
/// raymarch preview mode — the shader packs the hit primitive's
/// NodeId into the material G-buffer).
///
/// `ghost_pick_node_id` holds the result of a synchronous CPU raycast
/// performed at click time against the tree's ghost-role primitives
/// (cutters, Intersect operands). When `Some`, it overrides the
/// G-buffer decode — the visual rule is "if a ghost silhouette is
/// drawn at the click pixel, clicking picks it" even when a solid
/// surface is closer along the ray, because the ghost pass renders
/// depth-free on top of everything.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingPick {
    pub(crate) viewport: crate::viewport::ViewportId,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) ghost_pick_node_id: Option<u32>,
}

/// A queued drag-drop action awaiting the position-readback result from
/// the pick pipeline. When the matching `PickResult` returns with a
/// world-space position, `process_pick_result` consumes this instead of
/// running the usual selection update — the drop spawns the asset /
/// generator / preset at the hit point (or ground-plane fallback).
///
/// We reuse the pick pipeline (instead of adding a separate
/// position-readback route) because it already handles the
/// async-one-frame-later readback timing and we only need a single
/// coordinate query per drop.
#[derive(Debug, Clone)]
pub(crate) enum PendingDropAction {
    Asset { path: String },
    Generator { name: String },
    GeneratorPreset { path: String },
}

#[derive(Debug, Clone)]
pub(crate) struct PendingDrop {
    pub(crate) viewport: crate::viewport::ViewportId,
    /// The screen pixel that was dropped on. Used to cast a ground-plane
    /// fallback ray when the pick result's `position` is `None` (sky hit).
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) action: PendingDropAction,
}

/// Brush settings captured when a paint stroke sample fires its pick
/// readback. One instance lives in [`EngineState::paint_pick_settings`]
/// at a time — the editor throttles stamps to one pick in flight, and
/// a fresh `PaintAtPixel` replaces the previous settings if the pick
/// hasn't returned yet (latest sample wins, matching drag-preview).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintPickSettings {
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) strength: f32,
    pub(crate) falloff: f32,
    pub(crate) mode: crate::command::PaintMode,
    pub(crate) material_id: u16,
}

/// A live drag-and-drop preview. While active, each `DragPreviewOver`
/// (re)sets the pending pick at the cursor pixel. What happens when
/// the pick result arrives depends on `kind`:
///
/// * **Model** — a real asset entity is already spawned; its transform
///   gets updated to the new surface snap (AABB-bottom-snapped).
/// * **Generator** — nothing is spawned yet. A wireframe AABB gizmo is
///   drawn at the cached position every frame; on commit we spawn the
///   real generator there. Baking a live generator while the user
///   drags would produce a trail of stale children at each emit
///   position.
#[derive(Debug, Clone)]
pub(crate) struct DragPreviewState {
    pub(crate) viewport: crate::viewport::ViewportId,
    pub(crate) kind: DragPreviewKind,
    /// Most recent valid surface hit. Reused when the next pick returns
    /// a sky miss or a self-hit to avoid flickering to the ground plane.
    pub(crate) last_surface_pos: Option<glam::Vec3>,
    /// Last pixel the editor asked us to preview at. Used to cast the
    /// ground-plane fallback ray when we never got a valid surface hit.
    pub(crate) last_cursor: (u32, u32),
}

#[derive(Debug, Clone)]
pub(crate) enum DragPreviewKind {
    /// An .rkp asset — spawned as a real entity on DragEnter that
    /// tracks the cursor. Surface hits are bottom-snapped by
    /// subtracting `aabb_min_y` from the hit Y.
    Model {
        entity: hecs::Entity,
        aabb_min_y: f32,
    },
    /// A generator or preset — NO entity is spawned during drag. Just
    /// a gizmo wireframe (half-size `gizmo_half`) centered at
    /// `last_surface_pos`. On commit we spawn the real source.
    Generator {
        source: crate::command::DragPreviewSource,
        gizmo_half: glam::Vec3,
    },
}

pub(crate) struct EngineState {
    /// Render thread handle. Owns the wgpu device/queue/renderer/
    /// viewport renderers; sim communicates only via `render_worker.inbox`
    /// (per-frame `RenderFrame` snapshots), `render_worker.outbox`
    /// (per-frame `RenderResult` returns: pick + atten + GPU timings),
    /// and `render_worker.commands` (aperiodic events: resize, etc.).
    /// Dropping this triggers the render thread to shut down.
    pub(crate) render_worker: crate::render_worker::RenderWorker,

    // Scene management (CPU). Wrapped in `Arc<Mutex<>>` so the bake
    // worker can run the integrate pass (dealloc-prev + memcpy +
    // remap) against the shared pools directly, without shipping
    // artifacts back to the main thread for a 75+ ms copy. The lock
    // is uncontended on most frames — only the render thread's
    // per-frame geometry upload and the sim thread's asset loads
    // touch it, and both finish in a ms or two. See
    // `bake_worker::run_loop` for the worker-side lock scope.
    pub(crate) scene_mgr: std::sync::Arc<std::sync::Mutex<RkpSceneManager>>,

    /// Lock-free handle on `scene_mgr.geometry_epoch()` so per-tick
    /// reads (in `submit_render_frame`) don't take the scene_mgr
    /// Mutex. Without this, every sim tick blocks for the duration
    /// of any in-progress bake_worker integrate (~50 ms+), dropping
    /// sim from 60 Hz to ~20 Hz and making animation/camera feel
    /// like 0.5 fps.
    pub(crate) geometry_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

    /// Brush-overlay epoch handle. Bumped by
    /// `RkpSceneManager::update_brush_overlay` / `clear_brush_overlay`.
    /// Sim reads it lock-free to decide whether the next snapshot
    /// needs to ship a fresh brush-overlay upload.
    pub(crate) brush_overlay_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

    /// Paint-data epoch handle. Bumped by `apply_paint_sphere`
    /// whenever a stroke writes to leaf_attr / color. Separate from
    /// `geometry_epoch` so paint doesn't re-upload octree + brick
    /// buffers — render only slice-uploads the dirty slot range.
    pub(crate) paint_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

    /// Sim-side cache of per-asset skinning data. Built lazily under
    /// the scene_mgr lock only when `skinning_data_cache_epoch` falls
    /// behind the current `geometry_epoch_handle`, i.e. when a bake
    /// completes or an asset is (re)loaded. On most ticks there's no
    /// epoch change, so sim reads the cache directly without touching
    /// the scene_mgr Mutex — even when bake_worker is mid-integrate
    /// holding the lock for 100 ms+.
    ///
    /// The previous pattern was `scene_mgr.lock().unwrap().skinning_data(...)`
    /// once per skinned entity inside `update_scene_gpu`, so any bake
    /// in flight would stall sim for the full duration of the bake's
    /// `integrate_artifact`. Dropped sim from 60 Hz to ~5 Hz with
    /// multiple bakes and a few skinned entities — visible as
    /// "0.5 fps animation and camera" even though render reported
    /// 170 fps.
    pub(crate) skinning_data_cache: std::collections::HashMap<
        rkp_render::AssetHandle,
        rkp_render::SkinningAssetData,
    >,
    /// Last `geometry_epoch` we used to build [`Self::skinning_data_cache`].
    /// When `geometry_epoch_handle > this`, the cache is stale and
    /// gets rebuilt next tick.
    pub(crate) skinning_data_cache_epoch: u64,

    /// Async bake pipeline. The worker owns its own GpuEvaluator and
    /// private pools; the engine sends requests + drains results on
    /// each tick via `drain_bake_results`. The same worker also
    /// handles generator jobs (see `generator_system`).
    pub(crate) bake_worker: crate::bake_worker::BakeWorker,

    /// Generator tick driver. Scans entities with `GeneratorState`,
    /// hashes params to detect edits, submits stale runs to the
    /// bake worker's generator channel, and updates the ECS as
    /// results arrive.
    pub(crate) generator_system: crate::generator::GeneratorSystem,

    /// User-shader registry — populated by scanning
    /// `<project_root>/assets/shaders/` on project load and on
    /// filesystem change. The engine resolves `UserShader.shader_name`
    /// via this registry when flattening trees, and the bake worker's
    /// `GpuEvaluator` is recompiled whenever this registry changes
    /// (engine sends `WorkerControl::ReloadUserShaders`).
    pub(crate) user_shader_registry: rkp_render::shader_composer::UserShaderRegistry,

    // Input + Camera
    pub(crate) input_system: rkp_runtime::input::InputSystem,
    pub(crate) camera_control: CameraControlState,
    pub(crate) camera: CameraState,
    /// Viewport scaffolding (Phase 1). Populated with one entry for
    /// `ViewportId::MAIN` and kept in step with `camera` field on writes,
    /// but rendering still consults `camera` directly. Future phases route
    /// reads through here as the renderer is split per-viewport.
    pub(crate) viewports: crate::viewport::Viewports,

    // ECS — the source of truth for scene state.
    pub(crate) world: hecs::World,
    pub(crate) registry: crate::component_registry::ComponentRegistry,
    /// Stable UUID ↔ hecs Entity mapping.
    pub(crate) entity_uuids: std::collections::HashMap<hecs::Entity, uuid::Uuid>,
    pub(crate) uuid_to_entity: std::collections::HashMap<uuid::Uuid, hecs::Entity>,
    /// UUID counter for generating stable IDs.
    pub(crate) next_entity_uuid: u64,
    /// Per-entity scene-tree order key. `f64` so the drag-reorder path
    /// can insert between two neighbors (mid = (a + b) * 0.5) without
    /// renumbering anything. Persisted in the scene file — user-
    /// arranged ordering survives save / reload.
    ///
    /// Deliberately a side-map rather than a hecs component: the
    /// properties panel reflects over components via `ComponentRegistry`,
    /// and we don't want this editor-ordering concern to show up as a
    /// field there.
    pub(crate) entity_tree_order: std::collections::HashMap<hecs::Entity, f64>,
    /// Next TreeOrder value to hand out on a fresh spawn. Monotonic;
    /// reseeds past `max(loaded)` after a scene load so new spawns
    /// still append at the bottom.
    pub(crate) next_tree_order: f64,
    /// Currently selected entity.
    pub(crate) selected_entity: Option<hecs::Entity>,
    /// Currently selected procedural node (within the selected entity's ProceduralGeometry).
    pub(crate) selected_procedural_node: Option<u32>,

    // Derived GPU data — rebuilt from world each frame.
    pub(crate) gpu_objects: Vec<RkpGpuObject>,
    /// Maps gpu_object index → hecs Entity (for pick resolution).
    pub(crate) gpu_to_entity: Vec<hecs::Entity>,
    /// Maps hecs Entity → gpu_object index.
    pub(crate) entity_to_gpu: std::collections::HashMap<hecs::Entity, usize>,

    // Project state
    pub(crate) project_loaded: bool,
    pub(crate) project_name: String,
    pub(crate) project_dir: Option<std::path::PathBuf>,
    pub(crate) project_path: Option<std::path::PathBuf>,
    pub(crate) scene_path: Option<std::path::PathBuf>,
    pub(crate) project_dirty: bool,
    /// Available .rkp model files in the project.
    pub(crate) available_models: Vec<crate::snapshot::ModelInfo>,
    pub(crate) models_dirty: bool,
    /// Set whenever the gameplay dylib load/unload changes the set of
    /// registered generators. Drives the next snapshot's
    /// `available_generators` field.
    pub(crate) generators_dirty: bool,
    /// Discovered `.rkgen` preset files in the project's
    /// `assets/generators/` directory. Repopulated on project open.
    pub(crate) available_generator_presets: Vec<crate::generator::GeneratorPresetInfo>,
    /// Set when the preset list changed; drives the next snapshot.
    pub(crate) generator_presets_dirty: bool,
    /// Per-generator-entity set of slot keys that the *current* run
    /// has emitted so far. Reset on `WillResubmit`; consulted on
    /// `Completed` to delete persistent children whose key wasn't
    /// re-emitted (orphans from the previous generation).
    pub(crate) pending_generator_slot_keys:
        std::collections::HashMap<hecs::Entity, std::collections::HashSet<String>>,
    /// Source paths currently being re-imported. The UI consults this set
    /// to show a progress indicator in place of the Re-import button.
    /// Populated on `ReimportModel` submission, drained on completion.
    pub(crate) importing_sources: std::collections::HashSet<String>,
    /// Publish `importing_sources` to the UI on the next snapshot.
    pub(crate) importing_dirty: bool,
    /// Live per-import progress state keyed by source path string —
    /// reduced from the `ImportEvent` stream each tick, published to
    /// the UI through `StateUpdate.import_progress`. Entries are
    /// removed when the matching completion lands.
    pub(crate) importing_progress: std::collections::HashMap<String, crate::snapshot::ImportProgressInfo>,
    /// Latest editor layout JSON pushed up from the editor. Opaque to
    /// the engine — it just round-trips this through `.rkproject`.
    pub(crate) editor_layout_json: Option<String>,
    /// Ship `editor_layout_json` to the editor on the next snapshot.
    /// Set on project load so the editor can hydrate its signals; never
    /// set for echoes from the editor itself (no feedback loop).
    pub(crate) editor_layout_pending: bool,

    /// Shared cache of loaded `.rkskel` skeleton assets. Multiple
    /// entities loaded from the same `.rkp` share a single `Arc`.
    pub(crate) animation_cache: crate::animation::AnimationAssetCache,
    /// Per-frame allocator that packs every skinned entity's
    /// `Skeleton.current_pose` into one contiguous byte buffer for GPU
    /// upload. Rebuilt whenever `update_scene_gpu` runs.
    pub(crate) bone_matrix_allocator: crate::scene_sync::BoneMatrixAllocator,
    /// Per-frame scatter dispatches — one per skinned entity with a
    /// resolved skinning asset. Rebuilt in `update_scene_gpu`.
    pub(crate) skin_dispatches: Vec<crate::scene_sync::PlannedSkinDispatch>,
    /// Reusable per-frame scratch that concatenates every
    /// `skin_dispatches` entry into the single batched compute
    /// dispatch `scatter_skin_batch` fires.
    pub(crate) skin_batch: rkp_render::SkinBatchScratch,
    /// Total bytes required in `scene.bone_field_buffer` this frame;
    /// drives the per-frame grow+clear.
    pub(crate) skin_bone_field_bytes: u64,
    /// Total bytes required in `scene.bone_field_occ_buffer` this
    /// frame (packed 1-bit-per-brick occupancy bitmap paired with
    /// `bone_field_buffer`).
    pub(crate) skin_bone_field_occ_bytes: u64,
    /// Per-skinned-entity cache of last frame's `current_pose`. Used
    /// by the pause-aware scatter-skip — if every entity's pose is
    /// byte-identical to last frame and the set of skinned entities
    /// hasn't changed, the previous frame's `bone_field` buffer is
    /// still valid, so both the clear and the scatter dispatch get
    /// skipped. Big win when the user pauses an animation.
    pub(crate) last_skin_poses: std::collections::HashMap<hecs::Entity, Vec<glam::Mat4>>,
    /// `true` this frame iff the scatter can be skipped. Computed in
    /// `update_scene_gpu` after `plan_skin_dispatch` runs.
    pub(crate) skin_reuse: bool,
    /// Master toggle — when false, skip scatter + fall the march back
    /// to its rigid path. Driven by the AnimationPanel checkbox.
    pub(crate) skinning_enabled: bool,
    /// `true` → Dual-Quaternion Skinning in the scatter pass; `false`
    /// → Linear Blend Skinning. DQS preserves joint volume and fixes
    /// axial-twist candy-wrapper at ~+13% scatter cost. The visible
    /// payoff on gentle clips (Mixamo walks) is subtle; defaults off
    /// so the fast path is the common path. Flip on for extreme
    /// poses (crouch, acrobatic, twist-heavy clips) or to A/B compare.
    pub(crate) dqs_enabled: bool,

    /// Latest cloud-sun attenuation read from MAIN's volumetric pass,
    /// fed back over the render→sim result channel each frame. Sim
    /// uses it as the *target* of an EMA into [`Self::cloud_sun_atten`]
    /// (which is what actually scales the sun light on the next
    /// frame). NaN sentinel = render hasn't published a value yet
    /// (e.g. during the first frame or while MAIN is hidden); sim
    /// holds the previous target in that case.
    pub(crate) last_cloud_sun_atten_raw: f32,

    /// Sim-side stash for the most recently submitted pick's
    /// CPU-resolved ghost hint. Rendering is GPU-only; the ghost
    /// priority logic stays sim-side because it depends on the
    /// procedural tree (sim-owned). When the matching `PickResult`
    /// arrives back from render, sim consults this to decide whether
    /// the ghost win overrides the GPU-decoded NodeId.
    pub(crate) in_flight_pick_ghost: Option<u32>,

    /// Material library — manages .rkmat files and runtime palette.
    pub(crate) material_lib: crate::material_library::MaterialLibrary,
    /// Currently selected material in the materials panel.
    pub(crate) selected_material: Option<u16>,
    /// Currently selected model path (source mesh) for Asset Properties.
    pub(crate) selected_model: Option<String>,

    /// Environment settings (sky, lighting, shadows, tone mapping).
    pub(crate) environment: crate::environment::EnvironmentSettings,
    /// Whether environment settings changed and need GPU update.
    pub(crate) environment_dirty: bool,
    /// Whether the editor UI needs the latest environment (cleared by build_state_update).
    pub(crate) environment_ui_dirty: bool,

    /// Console log buffer.
    pub(crate) console: crate::console::ConsoleLog,
    /// Gameplay dylib loader (hot-reload).
    pub(crate) gameplay_loader: crate::gameplay_loader::GameplayLoader,
    /// Behavior system executor (created when play starts).
    pub(crate) behavior_executor: Option<crate::behavior::BehaviorExecutor>,
    /// Command queue for deferred ECS mutations from gameplay systems.
    pub(crate) behavior_commands: crate::behavior::CommandQueue,
    /// Key-value game state store + event bus.
    pub(crate) game_store: crate::behavior::GameStore,
    /// System entries from the gameplay dylib.
    pub(crate) gameplay_systems: Vec<&'static crate::behavior::SystemEntry>,
    /// Monotonic total play time.
    pub(crate) play_total_time: f64,
    /// Monotonic play frame counter.
    pub(crate) play_frame_count: u64,
    /// Play mode state (None = edit mode).
    pub(crate) play_state: Option<crate::play_mode::PlayModeState>,
    /// View options.
    pub(crate) show_colliders: bool,
    /// Collider caches need rebuild.
    pub(crate) collider_caches_dirty: bool,
    /// EMA of true tick rate (1 / wall-clock tick interval, including the
    /// 60-Hz pacing sleep). Distinct from `fps` in the state update, which
    /// is `1 / frame_work_time` and ignores sleep — useful for profiling but
    /// not what the user perceives.
    pub(crate) tick_hz_ema: f32,
    /// EMA of physics substeps per second across the engine tick. When
    /// physics is stepping at the target 60 Hz this sits near 60.
    pub(crate) physics_hz_ema: f32,
    /// EMA of the render thread's actual iteration rate, in Hz. Fed
    /// from `RenderResult::render_dt_ms` each time sim drains the
    /// render outbox. This is the "FPS" the editor displays — it
    /// reflects the on-screen production cadence, not sim CPU
    /// headroom (which `1 / cpu_total_ms` would be).
    pub(crate) render_hz_ema: f32,
    /// Rate at which fresh pixel frames actually reach the editor
    /// surface, EMA-smoothed. Updated from
    /// `RenderResult::delivered_dt_ms` whenever the render thread
    /// reports a successful pixel ship. Diverges from
    /// `render_hz_ema` whenever render iterates faster than it ships
    /// (interp re-renders, `MIN_FRAME_CALLBACK_INTERVAL`, sim
    /// upstream bottleneck). This is the honest "what did the user
    /// see" number.
    pub(crate) delivered_hz_ema: f32,
    /// Last inspector snapshot we sent to the editor. Used to skip pushing
    /// an identical snapshot every tick — without this, the panel re-renders
    /// 60Hz when physics writes Transform on a selected RigidBody, which
    /// chunks the UI thread.
    pub(crate) prev_inspector: Option<crate::inspector::InspectorSnapshot>,
    /// Cached per-entity `MaterialUsage` list. Computing it walks every
    /// leaf slot in the entity's subtree (including every brick cell
    /// and every prefilter-LOD slot), which is O(voxels) — trivial
    /// before the brick-descent fix, but a tick-killing 50 ms+ per
    /// call on high-voxel entities once bricks were included. Since
    /// the list only changes when the selection or geometry changes,
    /// cache it keyed on `(entity, geometry_epoch)` and reuse across
    /// ticks.
    pub(crate) cached_material_usage:
        Option<(hecs::Entity, u64, Vec<crate::inspector::MaterialUsage>)>,
    /// Same change-detection cache for the procedural snapshot.
    pub(crate) prev_procedural: Option<crate::procedural_snapshot::ProceduralSnapshot>,
    /// Last environment we shipped to the editor — diff-suppression
    /// avoids env-panel churn from any path that pushes env on a
    /// no-op (and means we no longer need the env_ui_dirty gate's
    /// "don't echo back during slider drag" workaround).
    pub(crate) prev_environment: Option<crate::environment::EnvironmentSettings>,
    /// Tracks the source hash of the user-shader registry the editor
    /// most recently saw, so `build_state_update` only sends a fresh
    /// `user_shaders` list when the registry has actually changed.
    /// `0` = "editor hasn't been told yet" (matches the empty-registry
    /// hash, but the first tick always sends regardless to populate it).
    pub(crate) prev_user_shader_hash: u64,
    pub(crate) user_shader_first_send: bool,

    /// File watcher for hot-reload (watches project assets/ directory).
    pub(crate) file_watcher: Option<crate::file_watcher::RkpFileWatcher>,
    /// Background import worker for mesh → .rkp conversion.
    pub(crate) import_worker: crate::import_worker::ImportWorker,

    // Geometry dirty flag
    pub(crate) geometry_dirty: bool,
    /// Scene structure changed — push objects list to UI.
    pub(crate) scene_dirty: bool,
    /// GPU objects / transforms changed — rebuild gpu_objects + re-upload.
    pub(crate) gpu_objects_dirty: bool,

    // Frame counter
    pub(crate) frame_index: u64,

    /// Ring buffer of per-frame CPU + GPU timings. Fed from the frame
    /// work at the end of `tick`, read by the editor (via `StateUpdate`)
    /// and by MCP once wired.
    pub(crate) profiling: crate::profiling::ProfilingHistory,

    /// Behavior `FixedUpdate` accumulator. Mirrors physics' Rapier-side
    /// accumulator so behavior code that registers in the FixedUpdate
    /// phase ticks at exactly 60 Hz regardless of render rate. We carry
    /// it here (not inside the executor) because the executor is
    /// optional — it doesn't exist before a project loads — and the
    /// accumulator must persist across executor (re)creation so we
    /// don't lose simulation time on hot-reload.
    pub(crate) behavior_fixed_accumulator: f32,

    // Temporally smoothed cloud-sun attenuation (camera→sun ray through the
    // cloud layer). Lerps toward the target each frame so a single noisy ray
    // through FBM doesn't flicker sun intensity.
    pub(crate) cloud_sun_atten: f32,

    // Render dimensions
    pub(crate) width: u32,
    pub(crate) height: u32,

    // (Per-viewport readback / composite / wireframe live in
    // `viewport_renderers[MAIN]` — see `rkp_render::ViewportRenderer`.)

    // Gizmo state
    pub(crate) gizmo: crate::gizmo::GizmoState,
    /// Gizmo state for the BUILD viewport — targets the selected
    /// procedural node's transform rather than an entity Transform.
    /// Separate from `gizmo` so a drag on BUILD doesn't fight a hover
    /// on MAIN (or vice versa).
    pub(crate) proc_gizmo: crate::gizmo::GizmoState,
    /// BUILD viewport cursor position (in BUILD's local pixel space).
    pub(crate) build_mouse_pos: glam::Vec2,
    /// BUILD viewport left-button pressed state. Tracked directly
    /// (rather than feeding `input_system`) so BUILD input doesn't
    /// fight MAIN's WASD/fly camera input.
    pub(crate) build_mouse_left: bool,
    /// Previous tick's value of `build_mouse_left` — used for edge
    /// detection so picking fires once per click rather than every
    /// frame the button is held.
    /// Parent-world transform of the procedural node at drag start —
    /// used to project world-space gizmo deltas back into the node's
    /// local (parent-relative) transform on each frame. Identity when
    /// no drag is active.
    pub(crate) proc_gizmo_parent_world: glam::Affine3A,
    /// Node's local SRT components at drag start. Held separately from
    /// `proc_gizmo.initial_*` (which track world-space) so we can
    /// rebuild the node's Affine3A correctly without redoing the
    /// decompose per frame.
    pub(crate) proc_gizmo_initial_local: (glam::Vec3, glam::Quat, glam::Vec3),
    /// Mouse position in viewport pixels (for gizmo hover).
    pub(crate) mouse_pos: glam::Vec2,

    /// Pending pixel-pick: a (viewport, x, y) plus optional CPU-resolved
    /// ghost-priority hint. Sim populates this on click; it travels in
    /// the next [`crate::render_frame::RenderFrame`] to the render
    /// thread, which encodes the G-buffer copy. The render thread
    /// returns the raw payload via `RenderResult::pick_result`; sim
    /// resolves the final entity / NodeId in `process_pick_result`.
    pub(crate) pending_pick: Option<PendingPick>,
    /// Queued drag-drop. Populated on `DropAsset` / `DropGenerator` /
    /// `DropGeneratorPreset`; consumed when the paired pick readback
    /// returns with a world-space position.
    pub(crate) pending_drop: Option<PendingDrop>,
    /// Active drag-preview: the preview entity + cached AABB offset +
    /// last-known-good surface pos. Populated on `DragAssetEnter`, kept
    /// up-to-date by pick readbacks during `DragAssetOver`, cleared on
    /// commit or cancel.
    pub(crate) drag_preview: Option<DragPreviewState>,
    /// Paint stroke that issued the current pick. When set, the next
    /// pick result bypasses selection / drag-preview handling and is
    /// routed to `apply_paint_stamp` with these settings. Populated by
    /// `EngineCommand::PaintAtPixel` (sim); taken out and consumed by
    /// `process_pick_result` when the matching readback returns.
    pub(crate) paint_pick_settings: Option<PaintPickSettings>,
    /// When set, the next pick result updates `paint_cursor_world`
    /// without applying any paint — the editor uses this to keep the
    /// cursor sphere tracking the cursor while the user is just hovering
    /// in paint mode. Mutually exclusive in practice with
    /// `paint_pick_settings` (LMB held → stamp path fires instead).
    pub(crate) paint_hover_pending: Option<crate::viewport::ViewportId>,
    /// `true` while the editor is in paint mode. Drives the cursor
    /// wireframe — when false, the cursor sphere is never drawn even
    /// if `paint_cursor_world` still carries a stale last-hover value.
    /// Updated by `SetPaintActive` commands.
    pub(crate) paint_mode_active: bool,
    /// Brush world-space radius while paint mode is active. Cached on
    /// the engine so the cursor sphere renders at the same size the
    /// next stamp will use. Updated by `SetPaintActive`.
    pub(crate) paint_mode_radius: f32,
    /// Most recent world-space hit under the paint cursor. Updated on
    /// every hover pick and on every successful paint stamp. `None`
    /// when no pick has returned with a valid surface hit yet.
    pub(crate) paint_cursor_world: Option<glam::Vec3>,
    /// Entity the cursor is currently over — needed to re-run the
    /// geodesic flood fill when the radius changes while the cursor
    /// is stationary. Set by pick results; cleared on paint-off.
    pub(crate) paint_cursor_entity: Option<hecs::Entity>,
    /// Cached light count for march pass (set in light upload block, used in render).
    pub(crate) num_lights_cache: u32,
    /// Base ShadeParams (recomputed once per frame from environment +
    /// light list). The per-viewport loop writes this into the shared
    /// shade_params buffer with the VR's `isolation` flag overlaid,
    /// just before that VR's submit.
    pub(crate) shade_params_base: rkp_render::rkp_shade::ShadeParams,
    /// Prefiltered-LOD early-exit toggle. On by default; flipped off for
    /// A/B correctness comparison against the pre-LOD descent behavior.
    pub(crate) lod_enabled: bool,
    /// Surface-Nets render-time normal reconstruction (POC). Off by
    /// default — flip on via `set_surfacenet_enabled` for A/B.
    pub(crate) surfacenet_enabled: bool,
}

impl EngineState {
    /// Flip the prefiltered-LOD march early-exit on or off. Public API
    /// exists mainly for A/B correctness tests and debugging — no UI
    /// wires it yet, but tests and MCP may poke it.
    #[allow(dead_code)]
    pub fn set_lod_enabled(&mut self, enabled: bool) {
        self.lod_enabled = enabled;
    }

    /// Current LOD toggle state.
    #[allow(dead_code)]
    pub fn lod_enabled(&self) -> bool {
        self.lod_enabled
    }

    /// Flip the Surface-Nets normal reconstruction on or off. When on,
    /// the march computes per-voxel normals from the 3³ in-brick
    /// occupancy neighborhood instead of reading the baked octahedral
    /// `LeafAttr.normal_oct`. Dormant infrastructure for the upcoming
    /// sculpt path — runtime normal reconstruction is what sculpting
    /// will need when voxels mutate between bakes.
    #[allow(dead_code)]
    pub fn set_surfacenet_enabled(&mut self, enabled: bool) {
        self.surfacenet_enabled = enabled;
    }

    #[allow(dead_code)]
    pub fn surfacenet_enabled(&self) -> bool {
        self.surfacenet_enabled
    }

    pub(crate) fn new(config: &EngineConfig, frame_callback: FrameCallback) -> Self {
        let ctx = rkp_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let scene_mgr = std::sync::Arc::new(std::sync::Mutex::new(
            RkpSceneManager::new(1_000_000),
        ));
        // Clone the lock-free epoch handle ONCE at startup; sim
        // reads it every tick to detect geometry changes without
        // having to take the scene_mgr Mutex.
        let (geometry_epoch_handle, brush_overlay_epoch_handle, paint_epoch_handle) = {
            let sm = scene_mgr.lock().expect("scene_mgr poisoned");
            (sm.epoch_handle(), sm.brush_overlay_epoch_handle(), sm.paint_epoch_handle())
        };

        // Input system with default action map.
        let mut input_system = rkp_runtime::input::InputSystem::new();
        input_system.add_map(crate::camera::default_action_map());
        input_system.set_active_map("editor");
        let camera_control = CameraControlState::default();

        // Bake worker: shares device/queue clones (cheap — wgpu wraps
        // the underlying objects in Arcs internally) and the same
        // scene_mgr we'll hand to the render thread below.
        let bake_worker = crate::bake_worker::BakeWorker::spawn(
            device.clone(),
            queue.clone(),
            scene_mgr.clone(),
        );
        let generator_system = crate::generator::GeneratorSystem::new(
            crate::generator::GeneratorRegistry::new(),
            bake_worker.tx_generator.clone(),
            bake_worker.rx_generator.clone(),
        );

        // Render worker — takes ownership of `device` + `queue` and
        // builds the renderer + per-VR pass chains on the render
        // thread. Sim never touches wgpu after this point; everything
        // GPU goes through the snapshot/result/command channels on
        // `render_worker`.
        let render_init = crate::render_frame::RenderInit {
            device,
            queue,
            initial_width: width,
            initial_height: height,
            scene_mgr: scene_mgr.clone(),
            render_pacing: config.render_pacing,
        };
        let render_worker = crate::render_worker::RenderWorker::spawn(
            render_init,
            frame_callback,
        );

        Self {
            bake_worker,
            generator_system,
            user_shader_registry: rkp_render::shader_composer::UserShaderRegistry::empty(),
            render_worker,
            scene_mgr,
            geometry_epoch_handle,
            brush_overlay_epoch_handle,
            paint_epoch_handle,
            skinning_data_cache: std::collections::HashMap::new(),
            // 0 → cache rebuilds the first time epoch > 0 (any
            // geometry mutation triggers it). Until then the cache
            // is empty, which matches a freshly-spawned scene.
            skinning_data_cache_epoch: 0,
            input_system,
            camera_control,
            camera: CameraState::default(),
            viewports: {
                let mut v = crate::viewport::Viewports::new();
                v.insert(crate::viewport::Viewport::new_main(width, height));
                // BUILD starts hidden (new_build sets visible: false) and
                // the editor flips it via SetViewportVisible when the user
                // opens the procedural preview.
                v.insert(crate::viewport::Viewport::new_build(800, 600));
                v
            },
            world: hecs::World::new(),
            registry: {
                let mut r = crate::component_registry::ComponentRegistry::new();
                crate::component_registry::register_builtins(&mut r);
                r
            },
            entity_uuids: std::collections::HashMap::new(),
            uuid_to_entity: std::collections::HashMap::new(),
            next_entity_uuid: 1,
            entity_tree_order: std::collections::HashMap::new(),
            next_tree_order: 0.0,
            selected_entity: None,
            selected_procedural_node: None,
            gpu_objects: Vec::new(),
            gpu_to_entity: Vec::new(),
            entity_to_gpu: std::collections::HashMap::new(),
            project_loaded: false,
            project_name: String::new(),
            project_dir: None,
            project_path: None,
            scene_path: None,
            project_dirty: true, // push initial state
            available_models: Vec::new(),
            models_dirty: false,
            generators_dirty: true,
            available_generator_presets: Vec::new(),
            generator_presets_dirty: true,
            pending_generator_slot_keys: std::collections::HashMap::new(),
            importing_sources: std::collections::HashSet::new(),
            importing_progress: std::collections::HashMap::new(),
            importing_dirty: false,
            editor_layout_json: None,
            editor_layout_pending: false,
            animation_cache: crate::animation::AnimationAssetCache::new(),
            bone_matrix_allocator: crate::scene_sync::BoneMatrixAllocator::new(),
            skin_dispatches: Vec::new(),
            skin_batch: rkp_render::SkinBatchScratch::default(),
            skin_bone_field_bytes: 0,
            skin_bone_field_occ_bytes: 0,
            last_skin_poses: std::collections::HashMap::new(),
            skin_reuse: false,
            skinning_enabled: true,
            dqs_enabled: false,
            last_cloud_sun_atten_raw: f32::NAN,
            in_flight_pick_ghost: None,
            material_lib: crate::material_library::MaterialLibrary::new(),
            selected_material: None,
            selected_model: None,
            environment: crate::environment::EnvironmentSettings::default(),
            environment_dirty: true, // upload on first frame
            environment_ui_dirty: true,
            console: crate::console::ConsoleLog::new(),
            gameplay_loader: crate::gameplay_loader::GameplayLoader::new(),
            behavior_executor: None,
            behavior_commands: crate::behavior::CommandQueue::new(),
            game_store: crate::behavior::GameStore::new(),
            gameplay_systems: Vec::new(),
            play_total_time: 0.0,
            play_frame_count: 0,
            play_state: None,
            show_colliders: false,
            collider_caches_dirty: true,
            tick_hz_ema: 60.0,
            physics_hz_ema: 0.0,
            render_hz_ema: 0.0,
            delivered_hz_ema: 0.0,
            prev_inspector: None,
            cached_material_usage: None,
            prev_procedural: None,
            prev_environment: None,
            prev_user_shader_hash: 0,
            user_shader_first_send: true,
            file_watcher: None,
            import_worker: crate::import_worker::ImportWorker::new(),
            geometry_dirty: false,
            scene_dirty: false,
            gpu_objects_dirty: true,
            frame_index: 0,
            profiling: crate::profiling::ProfilingHistory::default(),
            behavior_fixed_accumulator: 0.0,
            cloud_sun_atten: 1.0,
            width,
            height,
            gizmo: crate::gizmo::GizmoState::new(),
            proc_gizmo: crate::gizmo::GizmoState::new(),
            build_mouse_pos: glam::Vec2::ZERO,
            build_mouse_left: false,
            proc_gizmo_parent_world: glam::Affine3A::IDENTITY,
            proc_gizmo_initial_local: (glam::Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec3::ONE),
            mouse_pos: glam::Vec2::ZERO,
            pending_pick: None,
            pending_drop: None,
            drag_preview: None,
            paint_pick_settings: None,
            paint_hover_pending: None,
            paint_mode_active: false,
            paint_mode_radius: 0.5,
            paint_cursor_world: None,
            paint_cursor_entity: None,
            num_lights_cache: 1,
            shade_params_base: rkp_render::rkp_shade::ShadeParams::default(),
            lod_enabled: true,
            // Bake-time Laplacian smoothing of stored normals (see
            // `load_asset` → `smooth_shell_normals`) makes the shader-
            // time centroid reconstruction redundant. Default OFF so
            // the shader uses the smoothed baked normal via its
            // existing 1-fetch path.
            surfacenet_enabled: false,
        }
    }

    /// Access the profiling ring buffer. Intended for MCP tools and
    /// any other read-only consumer outside the editor snapshot path.
    #[allow(dead_code)]
    pub fn profiling_history(&self) -> &crate::profiling::ProfilingHistory {
        &self.profiling
    }
}
