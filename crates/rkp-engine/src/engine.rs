//! RkpEngine — the self-contained game engine.
//!
//! Owns the tick loop, scene state, renderer, and all GPU resources.
//! Runs on its own thread. Communicates with clients via command channel
//! and shared snapshot.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::Receiver;

use rkp_render::rkp_gpu_object::RkpGpuObject;
use rkp_render::rkp_scene_manager::RkpSceneManager;

use crate::camera::CameraControlState;
use crate::command::EngineCommand;
use crate::components::SpatialData;
use crate::snapshot::StateUpdate;

/// Collect all leaf voxel-pool slots from an octree in the packed node buffer.
///
/// Branch offsets in the packed buffer are ABSOLUTE indices. This function
/// traverses from `node_idx` directly in `all_nodes` without sub-slicing,
/// avoiding the offset-rebasing problem that `SparseOctree::from_raw` has
/// when given a sub-slice.
/// Coerce a JSON preset value into a `FieldValue` whose variant matches
/// the field's declared `FieldType`. Returns a descriptive error if the
/// types don't line up — the caller logs and continues.
fn json_to_field_value(
    value: &serde_json::Value,
    field_name: &str,
    comp: &crate::component_registry::ComponentEntry,
) -> Result<crate::inspector::FieldValue, String> {
    use crate::inspector::{FieldType, FieldValue};
    let meta = comp
        .meta
        .iter()
        .find(|m| m.name == field_name)
        .ok_or_else(|| format!("unknown field '{field_name}'"))?;
    match meta.field_type {
        FieldType::Float => value
            .as_f64()
            .map(FieldValue::Float)
            .ok_or_else(|| format!("expected number for {field_name}")),
        FieldType::Int => value
            .as_i64()
            .map(FieldValue::Int)
            .ok_or_else(|| format!("expected integer for {field_name}")),
        FieldType::Bool => value
            .as_bool()
            .map(FieldValue::Bool)
            .ok_or_else(|| format!("expected boolean for {field_name}")),
        FieldType::String => value
            .as_str()
            .map(|s| FieldValue::String(s.to_string()))
            .ok_or_else(|| format!("expected string for {field_name}")),
        FieldType::Vec3 => {
            let arr = value
                .as_array()
                .filter(|a| a.len() == 3)
                .ok_or_else(|| format!("expected [x,y,z] for {field_name}"))?;
            let mut out = [0.0f32; 3];
            for (i, v) in arr.iter().enumerate() {
                out[i] = v
                    .as_f64()
                    .ok_or_else(|| format!("non-number in {field_name}[{i}]"))?
                    as f32;
            }
            Ok(FieldValue::Vec3(out))
        }
        FieldType::Color => {
            let arr = value
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or_else(|| format!("expected [r,g,b,a] for {field_name}"))?;
            let mut out = [0.0f32; 4];
            for (i, v) in arr.iter().enumerate() {
                out[i] = v
                    .as_f64()
                    .ok_or_else(|| format!("non-number in {field_name}[{i}]"))?
                    as f32;
            }
            Ok(FieldValue::Color(out))
        }
    }
}

/// Compose a generator's parent transform with a child's local transform
/// to produce the child's absolute world transform. `Transform.rotation`
/// is Euler XYZ degrees (engine convention).
fn compose_generator_transforms(
    parent: &crate::components::Transform,
    child: &crate::components::Transform,
) -> crate::components::Transform {
    let parent_rot = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        parent.rotation.x.to_radians(),
        parent.rotation.y.to_radians(),
        parent.rotation.z.to_radians(),
    );
    let child_rot = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        child.rotation.x.to_radians(),
        child.rotation.y.to_radians(),
        child.rotation.z.to_radians(),
    );
    let world_rot = parent_rot * child_rot;
    let (ex, ey, ez) = world_rot.to_euler(glam::EulerRot::XYZ);
    let scaled_child_pos = parent_rot * (parent.scale * child.position);
    crate::components::Transform {
        position: parent.position + scaled_child_pos,
        rotation: glam::Vec3::new(ex.to_degrees(), ey.to_degrees(), ez.to_degrees()),
        scale: parent.scale * child.scale,
    }
}

fn collect_leaf_slots(all_nodes: &[u32], node_idx: usize, out: &mut Vec<u32>) {
    if node_idx >= all_nodes.len() {
        return;
    }
    let node = all_nodes[node_idx];
    if node == rkp_core::sparse_octree::EMPTY_NODE || node == rkp_core::sparse_octree::INTERIOR_NODE {
        return;
    }
    if rkp_core::sparse_octree::is_leaf(node) {
        out.push(rkp_core::sparse_octree::leaf_slot(node));
        return;
    }
    // Branch — value is absolute offset to 8 contiguous children.
    let children_offset = node as usize;
    for octant in 0..8 {
        collect_leaf_slots(all_nodes, children_offset + octant, out);
    }
}

/// Convert a SpatialHandle from rkp_render into our SpatialData component.
fn spatial_from_handle(
    handle: &rkp_core::scene_node::SpatialHandle,
    voxel_size: f32,
    aabb: &rkp_core::Aabb,
    grid_origin: glam::Vec3,
    voxel_slot_start: u32,
    voxel_slot_count: u32,
    brick_ids: Vec<u32>,
) -> SpatialData {
    if let rkp_core::scene_node::SpatialHandle::Octree {
        root_offset, len, depth, base_voxel_size,
    } = handle
    {
        SpatialData {
            root_offset: *root_offset,
            len: *len,
            depth: *depth,
            base_voxel_size: *base_voxel_size,
            aabb: *aabb,
            voxel_size,
            grid_origin,
            voxel_slot_start,
            voxel_slot_count,
            brick_ids,
        }
    } else {
        SpatialData {
            root_offset: 0, len: 0, depth: 0, base_voxel_size: voxel_size,
            aabb: *aabb, voxel_size,
            grid_origin,
            voxel_slot_start, voxel_slot_count,
            brick_ids,
        }
    }
}

/// Frame delivery callback — called once per visible viewport each tick.
/// `id` identifies which viewport this frame belongs to (the editor maps
/// each `ViewportId` to its own `RenderSurface`). RGBA8 pixels, length
/// `width * height * 4`.
pub type FrameCallback = Box<dyn Fn(crate::viewport::ViewportId, &[u8], u32, u32) + Send>;

/// State update callback — called each tick with engine state.
pub type StateCallback = Box<dyn Fn(&StateUpdate) + Send>;

/// How aggressively a thread loop should pace itself.
///
/// Used by both the sim tick loop ([`EngineConfig::sim_pacing`]) and
/// the render thread loop ([`EngineConfig::render_pacing`]).
///
/// - `Uncapped` runs as fast as the CPU/GPU can sustain. Right for
///   game builds shipped to players, or whenever you want maximum
///   throughput at the cost of CPU.
/// - `TargetHz(N)` sleeps each loop iteration's remainder to hold at
///   most `N` iterations per second. Right for the editor (60 Hz keeps
///   battery / fan reasonable), or to cap render at a display refresh
///   rate.
///
/// Sim correctness is independent of these knobs: physics and behavior
/// `FixedUpdate` run via accumulators on real wall-clock dt and tick
/// at the same simulation rate regardless of any pacing. Per-frame
/// systems (animation, camera/input, behavior `Update` / `LateUpdate`)
/// advance by real_dt and stay frame-rate-correct.
///
/// Display-rate vsync is *not* a value in this enum: the engine
/// renders headless to an offscreen texture and ships pixels to the
/// editor, where the actual presentation (and any vsync) is owned by
/// the rinch surface chain. To approximate vsync at the engine level,
/// set `render_pacing: TargetHz(display_refresh_hz)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// Run as fast as possible.
    Uncapped,
    /// Sleep at the end of each loop iteration so the loop holds at
    /// most this many iterations per second.
    TargetHz(u32),
}

impl PacingMode {
    /// Sleep target as a Duration, or None for uncapped.
    pub fn target_interval(&self) -> Option<Duration> {
        match *self {
            PacingMode::Uncapped => None,
            PacingMode::TargetHz(0) => None,
            PacingMode::TargetHz(hz) => {
                Some(Duration::from_nanos(1_000_000_000u64 / hz as u64))
            }
        }
    }
}

/// Backwards-compatibility alias. New code should use [`PacingMode`].
#[deprecated(note = "use `PacingMode` — `RenderPacing` was a misleading name when only the sim loop used it")]
pub type RenderPacing = PacingMode;

/// Configuration for spawning the engine.
pub struct EngineConfig {
    /// Initial render width.
    pub width: u32,
    /// Initial render height.
    pub height: u32,
    /// Sim tick-loop pacing. Drives ECS, physics, behavior, animation,
    /// snapshot construction. `TargetHz(60)` is the editor default;
    /// games typically run sim at a fixed step (60 or 120 Hz).
    pub sim_pacing: PacingMode,
    /// Render thread pacing. Independent of sim. When `render_pacing`'s
    /// rate exceeds `sim_pacing`'s, render interpolates between the
    /// last two snapshots so visuals stay smooth at the higher rate
    /// instead of strobing the same sim state. `TargetHz(60)` is the
    /// editor default; games can set `Uncapped` or `TargetHz(144)` /
    /// `TargetHz(240)` to match a high-refresh display.
    pub render_pacing: PacingMode,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            // Sim caps at 60 Hz: physics + behavior FixedUpdate both
            // accumulate against fixed 1/60 steps, so an uncapped sim
            // would spin doing zero work most ticks. 60 Hz matches
            // the fixed-step rate exactly — every tick produces one
            // physics step on average.
            sim_pacing: PacingMode::TargetHz(60),
            // Render is uncapped by default. This is a game engine —
            // players on 240 Hz monitors should get 240 fps. The
            // render thread interpolates between sim snapshots so
            // visuals stay smooth even though sim is locked at 60 Hz.
            // Editor / dev tooling can override to TargetHz(N) if
            // they want a softer cap (battery, fans, etc.).
            render_pacing: PacingMode::Uncapped,
        }
    }
}

/// The RKIPatch game engine.
///
/// Created via [`RkpEngine::spawn`], which starts the engine on a background thread.
/// The caller communicates via the command channel and receives state via callbacks.
pub struct RkpEngine {
    /// Handle to the engine thread.
    thread: Option<JoinHandle<()>>,
    /// Send commands to the engine.
    pub cmd_tx: crossbeam::channel::Sender<EngineCommand>,
}

impl RkpEngine {
    /// Spawn the engine on a background thread.
    ///
    /// - `frame_callback`: called each tick with RGBA8 pixels (`width * height * 4` bytes)
    /// - `state_callback`: called each tick with current engine state
    pub fn spawn(
        config: EngineConfig,
        frame_callback: FrameCallback,
        state_callback: StateCallback,
    ) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded();

        let thread = std::thread::Builder::new()
            .name("rkp-engine".into())
            .spawn(move || {
                tick_loop(cmd_rx, frame_callback, state_callback, config);
            })
            .expect("failed to spawn engine thread");

        Self {
            thread: Some(thread),
            cmd_tx,
        }
    }

    /// Send a command to the engine (non-blocking).
    pub fn send(&self, cmd: EngineCommand) {
        let _ = self.cmd_tx.send(cmd);
    }
}

impl Drop for RkpEngine {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(EngineCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Camera state tracked by the engine.
#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    pub position: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: glam::Vec3::new(0.0, 2.0, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            fov: 60.0,
            near: 0.01,
            far: 1000.0,
        }
    }
}

// ── Internal: engine state ───────────────────────────────────────────

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
struct PendingPick {
    viewport: crate::viewport::ViewportId,
    x: u32,
    y: u32,
    ghost_pick_node_id: Option<u32>,
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
enum PendingDropAction {
    Asset { path: String },
    Generator { name: String },
    GeneratorPreset { path: String },
}

#[derive(Debug, Clone)]
struct PendingDrop {
    viewport: crate::viewport::ViewportId,
    /// The screen pixel that was dropped on. Used to cast a ground-plane
    /// fallback ray when the pick result's `position` is `None` (sky hit).
    x: u32,
    y: u32,
    action: PendingDropAction,
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
struct DragPreviewState {
    viewport: crate::viewport::ViewportId,
    kind: DragPreviewKind,
    /// Most recent valid surface hit. Reused when the next pick returns
    /// a sky miss or a self-hit to avoid flickering to the ground plane.
    last_surface_pos: Option<glam::Vec3>,
    /// Last pixel the editor asked us to preview at. Used to cast the
    /// ground-plane fallback ray when we never got a valid surface hit.
    last_cursor: (u32, u32),
}

#[derive(Debug, Clone)]
enum DragPreviewKind {
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

struct EngineState {
    /// Render thread handle. Owns the wgpu device/queue/renderer/
    /// viewport renderers; sim communicates only via `render_worker.inbox`
    /// (per-frame `RenderFrame` snapshots), `render_worker.outbox`
    /// (per-frame `RenderResult` returns: pick + atten + GPU timings),
    /// and `render_worker.commands` (aperiodic events: resize, etc.).
    /// Dropping this triggers the render thread to shut down.
    render_worker: crate::render_worker::RenderWorker,

    // Scene management (CPU). Wrapped in `Arc<Mutex<>>` so the bake
    // worker can run the integrate pass (dealloc-prev + memcpy +
    // remap) against the shared pools directly, without shipping
    // artifacts back to the main thread for a 75+ ms copy. The lock
    // is uncontended on most frames — only the render thread's
    // per-frame geometry upload and the sim thread's asset loads
    // touch it, and both finish in a ms or two. See
    // `bake_worker::run_loop` for the worker-side lock scope.
    scene_mgr: std::sync::Arc<std::sync::Mutex<RkpSceneManager>>,

    /// Lock-free handle on `scene_mgr.geometry_epoch()` so per-tick
    /// reads (in `submit_render_frame`) don't take the scene_mgr
    /// Mutex. Without this, every sim tick blocks for the duration
    /// of any in-progress bake_worker integrate (~50 ms+), dropping
    /// sim from 60 Hz to ~20 Hz and making animation/camera feel
    /// like 0.5 fps.
    geometry_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

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
    skinning_data_cache: std::collections::HashMap<
        rkp_render::AssetHandle,
        rkp_render::SkinningAssetData,
    >,
    /// Last `geometry_epoch` we used to build [`Self::skinning_data_cache`].
    /// When `geometry_epoch_handle > this`, the cache is stale and
    /// gets rebuilt next tick.
    skinning_data_cache_epoch: u64,

    /// Async bake pipeline. The worker owns its own GpuEvaluator and
    /// private pools; the engine sends requests + drains results on
    /// each tick via `drain_bake_results`. The same worker also
    /// handles generator jobs (see `generator_system`).
    bake_worker: crate::bake_worker::BakeWorker,

    /// Generator tick driver. Scans entities with `GeneratorState`,
    /// hashes params to detect edits, submits stale runs to the
    /// bake worker's generator channel, and updates the ECS as
    /// results arrive.
    generator_system: crate::generator::GeneratorSystem,

    // Input + Camera
    input_system: rkp_runtime::input::InputSystem,
    camera_control: CameraControlState,
    camera: CameraState,
    /// Viewport scaffolding (Phase 1). Populated with one entry for
    /// `ViewportId::MAIN` and kept in step with `camera` field on writes,
    /// but rendering still consults `camera` directly. Future phases route
    /// reads through here as the renderer is split per-viewport.
    viewports: crate::viewport::Viewports,

    // ECS — the source of truth for scene state.
    world: hecs::World,
    registry: crate::component_registry::ComponentRegistry,
    /// Stable UUID ↔ hecs Entity mapping.
    entity_uuids: std::collections::HashMap<hecs::Entity, uuid::Uuid>,
    uuid_to_entity: std::collections::HashMap<uuid::Uuid, hecs::Entity>,
    /// UUID counter for generating stable IDs.
    next_entity_uuid: u64,
    /// Per-entity scene-tree order key. `f64` so the drag-reorder path
    /// can insert between two neighbors (mid = (a + b) * 0.5) without
    /// renumbering anything. Persisted in the scene file — user-
    /// arranged ordering survives save / reload.
    ///
    /// Deliberately a side-map rather than a hecs component: the
    /// properties panel reflects over components via `ComponentRegistry`,
    /// and we don't want this editor-ordering concern to show up as a
    /// field there.
    entity_tree_order: std::collections::HashMap<hecs::Entity, f64>,
    /// Next TreeOrder value to hand out on a fresh spawn. Monotonic;
    /// reseeds past `max(loaded)` after a scene load so new spawns
    /// still append at the bottom.
    next_tree_order: f64,
    /// Currently selected entity.
    selected_entity: Option<hecs::Entity>,
    /// Currently selected procedural node (within the selected entity's ProceduralGeometry).
    selected_procedural_node: Option<u32>,

    // Derived GPU data — rebuilt from world each frame.
    gpu_objects: Vec<RkpGpuObject>,
    /// Maps gpu_object index → hecs Entity (for pick resolution).
    gpu_to_entity: Vec<hecs::Entity>,
    /// Maps hecs Entity → gpu_object index.
    entity_to_gpu: std::collections::HashMap<hecs::Entity, usize>,

    // Project state
    project_loaded: bool,
    project_name: String,
    project_dir: Option<std::path::PathBuf>,
    project_path: Option<std::path::PathBuf>,
    scene_path: Option<std::path::PathBuf>,
    project_dirty: bool,
    /// Available .rkp model files in the project.
    available_models: Vec<crate::snapshot::ModelInfo>,
    models_dirty: bool,
    /// Set whenever the gameplay dylib load/unload changes the set of
    /// registered generators. Drives the next snapshot's
    /// `available_generators` field.
    generators_dirty: bool,
    /// Discovered `.rkgen` preset files in the project's
    /// `assets/generators/` directory. Repopulated on project open.
    available_generator_presets: Vec<crate::generator::GeneratorPresetInfo>,
    /// Set when the preset list changed; drives the next snapshot.
    generator_presets_dirty: bool,
    /// Per-generator-entity set of slot keys that the *current* run
    /// has emitted so far. Reset on `WillResubmit`; consulted on
    /// `Completed` to delete persistent children whose key wasn't
    /// re-emitted (orphans from the previous generation).
    pending_generator_slot_keys:
        std::collections::HashMap<hecs::Entity, std::collections::HashSet<String>>,
    /// Source paths currently being re-imported. The UI consults this set
    /// to show a progress indicator in place of the Re-import button.
    /// Populated on `ReimportModel` submission, drained on completion.
    importing_sources: std::collections::HashSet<String>,
    /// Publish `importing_sources` to the UI on the next snapshot.
    importing_dirty: bool,
    /// Live per-import progress state keyed by source path string —
    /// reduced from the `ImportEvent` stream each tick, published to
    /// the UI through `StateUpdate.import_progress`. Entries are
    /// removed when the matching completion lands.
    importing_progress: std::collections::HashMap<String, crate::snapshot::ImportProgressInfo>,
    /// Latest editor layout JSON pushed up from the editor. Opaque to
    /// the engine — it just round-trips this through `.rkproject`.
    editor_layout_json: Option<String>,
    /// Ship `editor_layout_json` to the editor on the next snapshot.
    /// Set on project load so the editor can hydrate its signals; never
    /// set for echoes from the editor itself (no feedback loop).
    editor_layout_pending: bool,

    /// Shared cache of loaded `.rkskel` skeleton assets. Multiple
    /// entities loaded from the same `.rkp` share a single `Arc`.
    animation_cache: crate::animation::AnimationAssetCache,
    /// Per-frame allocator that packs every skinned entity's
    /// `Skeleton.current_pose` into one contiguous byte buffer for GPU
    /// upload. Rebuilt whenever `update_scene_gpu` runs.
    bone_matrix_allocator: crate::scene_sync::BoneMatrixAllocator,
    /// Per-frame scatter dispatches — one per skinned entity with a
    /// resolved skinning asset. Rebuilt in `update_scene_gpu`.
    skin_dispatches: Vec<crate::scene_sync::PlannedSkinDispatch>,
    /// Reusable per-frame scratch that concatenates every
    /// `skin_dispatches` entry into the single batched compute
    /// dispatch `scatter_skin_batch` fires.
    skin_batch: rkp_render::SkinBatchScratch,
    /// Total bytes required in `scene.bone_field_buffer` this frame;
    /// drives the per-frame grow+clear.
    skin_bone_field_bytes: u64,
    /// Total bytes required in `scene.bone_field_occ_buffer` this
    /// frame (packed 1-bit-per-brick occupancy bitmap paired with
    /// `bone_field_buffer`).
    skin_bone_field_occ_bytes: u64,
    /// Per-skinned-entity cache of last frame's `current_pose`. Used
    /// by the pause-aware scatter-skip — if every entity's pose is
    /// byte-identical to last frame and the set of skinned entities
    /// hasn't changed, the previous frame's `bone_field` buffer is
    /// still valid, so both the clear and the scatter dispatch get
    /// skipped. Big win when the user pauses an animation.
    last_skin_poses: std::collections::HashMap<hecs::Entity, Vec<glam::Mat4>>,
    /// `true` this frame iff the scatter can be skipped. Computed in
    /// `update_scene_gpu` after `plan_skin_dispatch` runs.
    skin_reuse: bool,
    /// Master toggle — when false, skip scatter + fall the march back
    /// to its rigid path. Driven by the AnimationPanel checkbox.
    skinning_enabled: bool,
    /// `true` → Dual-Quaternion Skinning in the scatter pass; `false`
    /// → Linear Blend Skinning. DQS preserves joint volume and fixes
    /// axial-twist candy-wrapper at ~+13% scatter cost. The visible
    /// payoff on gentle clips (Mixamo walks) is subtle; defaults off
    /// so the fast path is the common path. Flip on for extreme
    /// poses (crouch, acrobatic, twist-heavy clips) or to A/B compare.
    dqs_enabled: bool,

    /// Latest cloud-sun attenuation read from MAIN's volumetric pass,
    /// fed back over the render→sim result channel each frame. Sim
    /// uses it as the *target* of an EMA into [`Self::cloud_sun_atten`]
    /// (which is what actually scales the sun light on the next
    /// frame). NaN sentinel = render hasn't published a value yet
    /// (e.g. during the first frame or while MAIN is hidden); sim
    /// holds the previous target in that case.
    last_cloud_sun_atten_raw: f32,

    /// Sim-side stash for the most recently submitted pick's
    /// CPU-resolved ghost hint. Rendering is GPU-only; the ghost
    /// priority logic stays sim-side because it depends on the
    /// procedural tree (sim-owned). When the matching `PickResult`
    /// arrives back from render, sim consults this to decide whether
    /// the ghost win overrides the GPU-decoded NodeId.
    in_flight_pick_ghost: Option<u32>,

    /// Material library — manages .rkmat files and runtime palette.
    material_lib: crate::material_library::MaterialLibrary,
    /// Currently selected material in the materials panel.
    selected_material: Option<u16>,
    /// Currently selected model path (source mesh) for Asset Properties.
    selected_model: Option<String>,

    /// Environment settings (sky, lighting, shadows, tone mapping).
    environment: crate::environment::EnvironmentSettings,
    /// Whether environment settings changed and need GPU update.
    environment_dirty: bool,
    /// Whether the editor UI needs the latest environment (cleared by build_state_update).
    environment_ui_dirty: bool,

    /// Console log buffer.
    console: crate::console::ConsoleLog,
    /// Gameplay dylib loader (hot-reload).
    gameplay_loader: crate::gameplay_loader::GameplayLoader,
    /// Behavior system executor (created when play starts).
    behavior_executor: Option<crate::behavior::BehaviorExecutor>,
    /// Command queue for deferred ECS mutations from gameplay systems.
    behavior_commands: crate::behavior::CommandQueue,
    /// Key-value game state store + event bus.
    game_store: crate::behavior::GameStore,
    /// System entries from the gameplay dylib.
    gameplay_systems: Vec<&'static crate::behavior::SystemEntry>,
    /// Monotonic total play time.
    play_total_time: f64,
    /// Monotonic play frame counter.
    play_frame_count: u64,
    /// Play mode state (None = edit mode).
    play_state: Option<crate::play_mode::PlayModeState>,
    /// View options.
    show_colliders: bool,
    /// Collider caches need rebuild.
    collider_caches_dirty: bool,
    /// EMA of true tick rate (1 / wall-clock tick interval, including the
    /// 60-Hz pacing sleep). Distinct from `fps` in the state update, which
    /// is `1 / frame_work_time` and ignores sleep — useful for profiling but
    /// not what the user perceives.
    tick_hz_ema: f32,
    /// EMA of physics substeps per second across the engine tick. When
    /// physics is stepping at the target 60 Hz this sits near 60.
    physics_hz_ema: f32,
    /// EMA of the render thread's actual iteration rate, in Hz. Fed
    /// from `RenderResult::render_dt_ms` each time sim drains the
    /// render outbox. This is the "FPS" the editor displays — it
    /// reflects the on-screen production cadence, not sim CPU
    /// headroom (which `1 / cpu_total_ms` would be).
    render_hz_ema: f32,
    /// Last inspector snapshot we sent to the editor. Used to skip pushing
    /// an identical snapshot every tick — without this, the panel re-renders
    /// 60Hz when physics writes Transform on a selected RigidBody, which
    /// chunks the UI thread.
    prev_inspector: Option<crate::inspector::InspectorSnapshot>,
    /// Same change-detection cache for the procedural snapshot.
    prev_procedural: Option<crate::procedural_snapshot::ProceduralSnapshot>,
    /// Last environment we shipped to the editor — diff-suppression
    /// avoids env-panel churn from any path that pushes env on a
    /// no-op (and means we no longer need the env_ui_dirty gate's
    /// "don't echo back during slider drag" workaround).
    prev_environment: Option<crate::environment::EnvironmentSettings>,

    /// File watcher for hot-reload (watches project assets/ directory).
    file_watcher: Option<crate::file_watcher::RkpFileWatcher>,
    /// Background import worker for mesh → .rkp conversion.
    import_worker: crate::import_worker::ImportWorker,

    // Geometry dirty flag
    geometry_dirty: bool,
    /// Scene structure changed — push objects list to UI.
    scene_dirty: bool,
    /// GPU objects / transforms changed — rebuild gpu_objects + re-upload.
    gpu_objects_dirty: bool,

    // Frame counter
    frame_index: u64,

    /// Ring buffer of per-frame CPU + GPU timings. Fed from the frame
    /// work at the end of `tick`, read by the editor (via `StateUpdate`)
    /// and by MCP once wired.
    profiling: crate::profiling::ProfilingHistory,

    /// Behavior `FixedUpdate` accumulator. Mirrors physics' Rapier-side
    /// accumulator so behavior code that registers in the FixedUpdate
    /// phase ticks at exactly 60 Hz regardless of render rate. We carry
    /// it here (not inside the executor) because the executor is
    /// optional — it doesn't exist before a project loads — and the
    /// accumulator must persist across executor (re)creation so we
    /// don't lose simulation time on hot-reload.
    behavior_fixed_accumulator: f32,

    // Temporally smoothed cloud-sun attenuation (camera→sun ray through the
    // cloud layer). Lerps toward the target each frame so a single noisy ray
    // through FBM doesn't flicker sun intensity.
    cloud_sun_atten: f32,

    // Render dimensions
    width: u32,
    height: u32,

    // (Per-viewport readback / composite / wireframe live in
    // `viewport_renderers[MAIN]` — see `rkp_render::ViewportRenderer`.)

    // Gizmo state
    gizmo: crate::gizmo::GizmoState,
    /// Gizmo state for the BUILD viewport — targets the selected
    /// procedural node's transform rather than an entity Transform.
    /// Separate from `gizmo` so a drag on BUILD doesn't fight a hover
    /// on MAIN (or vice versa).
    proc_gizmo: crate::gizmo::GizmoState,
    /// BUILD viewport cursor position (in BUILD's local pixel space).
    build_mouse_pos: glam::Vec2,
    /// BUILD viewport left-button pressed state. Tracked directly
    /// (rather than feeding `input_system`) so BUILD input doesn't
    /// fight MAIN's WASD/fly camera input.
    build_mouse_left: bool,
    /// Previous tick's value of `build_mouse_left` — used for edge
    /// detection so picking fires once per click rather than every
    /// frame the button is held.
    /// Parent-world transform of the procedural node at drag start —
    /// used to project world-space gizmo deltas back into the node's
    /// local (parent-relative) transform on each frame. Identity when
    /// no drag is active.
    proc_gizmo_parent_world: glam::Affine3A,
    /// Node's local SRT components at drag start. Held separately from
    /// `proc_gizmo.initial_*` (which track world-space) so we can
    /// rebuild the node's Affine3A correctly without redoing the
    /// decompose per frame.
    proc_gizmo_initial_local: (glam::Vec3, glam::Quat, glam::Vec3),
    /// Mouse position in viewport pixels (for gizmo hover).
    mouse_pos: glam::Vec2,

    /// Pending pixel-pick: a (viewport, x, y) plus optional CPU-resolved
    /// ghost-priority hint. Sim populates this on click; it travels in
    /// the next [`crate::render_frame::RenderFrame`] to the render
    /// thread, which encodes the G-buffer copy. The render thread
    /// returns the raw payload via `RenderResult::pick_result`; sim
    /// resolves the final entity / NodeId in `process_pick_result`.
    pending_pick: Option<PendingPick>,
    /// Queued drag-drop. Populated on `DropAsset` / `DropGenerator` /
    /// `DropGeneratorPreset`; consumed when the paired pick readback
    /// returns with a world-space position.
    pending_drop: Option<PendingDrop>,
    /// Active drag-preview: the preview entity + cached AABB offset +
    /// last-known-good surface pos. Populated on `DragAssetEnter`, kept
    /// up-to-date by pick readbacks during `DragAssetOver`, cleared on
    /// commit or cancel.
    drag_preview: Option<DragPreviewState>,
    /// Cached light count for march pass (set in light upload block, used in render).
    num_lights_cache: u32,
    /// Base ShadeParams (recomputed once per frame from environment +
    /// light list). The per-viewport loop writes this into the shared
    /// shade_params buffer with the VR's `isolation` flag overlaid,
    /// just before that VR's submit.
    shade_params_base: rkp_render::rkp_shade::ShadeParams,
    /// Prefiltered-LOD early-exit toggle. On by default; flipped off for
    /// A/B correctness comparison against the pre-LOD descent behavior.
    lod_enabled: bool,
    /// Surface-Nets render-time normal reconstruction (POC). Off by
    /// default — flip on via `set_surfacenet_enabled` for A/B.
    surfacenet_enabled: bool,
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

    fn new(config: &EngineConfig, frame_callback: FrameCallback) -> Self {
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
        let geometry_epoch_handle = scene_mgr
            .lock()
            .expect("scene_mgr poisoned")
            .epoch_handle();

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
            render_worker,
            scene_mgr,
            geometry_epoch_handle,
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
            prev_inspector: None,
            prev_procedural: None,
            prev_environment: None,
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

    /// Build a [`RenderFrame`] snapshot from current ECS / environment
    /// state and submit it to the render thread.
    ///
    /// Sim does no GPU work directly anymore — every per-frame thing the
    /// renderer used to read off `EngineState` is now packaged into a
    /// snapshot and shipped over `render_worker.inbox`. The render
    /// thread consumes, encodes, submits, and returns a
    /// [`RenderResult`] back via `render_worker.outbox` (which we drain
    /// in [`Self::drain_render_results`] called from the tick loop).
    ///
    /// Returns the CPU phases for this submission (setup vs. snapshot
    /// build vs. submit-handoff). The post-submit bucket reflects the
    /// time spent waiting for render-thread results, which is also a
    /// proxy for GPU backpressure.
    ///
    /// Originally a 700-line method that owned both the build *and* the
    /// GPU work. The latter migrated to [`crate::render_worker`]; what
    /// remains here is purely sim-side data assembly.
    fn submit_render_frame(&mut self) {
        use crate::viewport::ViewportId;
        let frame_start = std::time::Instant::now();

        // 0. Drain RenderResults that landed since last submit. The
        //    render thread runs on its own pace; the latest result it
        //    finished publishing carries the freshest pick decoding,
        //    cloud-sun atten, and GPU pass timings for us to fold back
        //    into sim state before we build the next snapshot.
        self.drain_render_results();

        // 0a. Material palette — built every tick and shipped in the
        //     snapshot. Render uploads every frame. Cheap (small Vec)
        //     and robust to snapshot drops; the old "ship only when
        //     dirty" pattern could lose the upload if its carrying
        //     snapshot was dropped by the newest-wins inbox before
        //     render saw it.
        let materials = self.material_lib.build_palette();
        // Clear the dirty flag so any other consumers (UI, etc.)
        // know the palette they observed has been published. We
        // ship every tick regardless, so the flag is purely for
        // outside-of-render bookkeeping now.
        self.material_lib.clear_dirty();

        // MAIN camera first: atmosphere LUTs + sun-light tinting both
        // depend on its altitude (scene-wide values shared across VRs).
        let main_cam = self.build_camera_uniforms(ViewportId::MAIN);
        let cam_y = main_cam.position[1];

        // Cloud-sun atten: smooth toward the latest render-thread
        // readback (fed in via `last_cloud_sun_atten_raw` by
        // `drain_render_results`). NaN sentinel = render hasn't
        // published one yet (first frame, MAIN hidden), so we hold the
        // last EMA target.
        let target_atten = if self.environment.attenuate_sun_by_clouds
            && self.environment.clouds_enabled
        {
            if self.last_cloud_sun_atten_raw.is_nan() {
                self.cloud_sun_atten
            } else {
                self.last_cloud_sun_atten_raw
            }
        } else {
            1.0
        };
        self.cloud_sun_atten += (target_atten - self.cloud_sun_atten) * 0.04;

        // Sun + entity-driven point/spot lights, all in the order the
        // shade shader expects (entry 0 = sun).
        let mut sun_light = self.environment.to_gpu_light(cam_y);
        sun_light.color[0] *= self.cloud_sun_atten;
        sun_light.color[1] *= self.cloud_sun_atten;
        sun_light.color[2] *= self.cloud_sun_atten;
        let mut gpu_lights = vec![sun_light];
        for (_entity, (transform, pl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::PointLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 1.0],
                color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                direction: [0.0, 0.0, 0.0, 0.0],
                params: [pl.range, 0.0, 0.0, if pl.cast_shadow { 1.0 } else { 0.0 }],
            });
        }
        for (_entity, (transform, sl)) in self
            .world
            .query::<(&crate::components::Transform, &crate::components::SpotLight)>()
            .iter()
        {
            gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                position: [transform.position.x, transform.position.y, transform.position.z, 2.0],
                color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                direction: [
                    sl.direction.x,
                    sl.direction.y,
                    sl.direction.z,
                    sl.outer_angle.to_radians(),
                ],
                params: [
                    sl.range,
                    sl.inner_angle.to_radians(),
                    0.0,
                    if sl.cast_shadow { 1.0 } else { 0.0 },
                ],
            });
        }

        let mut shade_params = self.environment.to_shade_params(cam_y);
        shade_params.num_lights = gpu_lights.len() as u32;
        self.shade_params_base = shade_params;
        self.num_lights_cache = shade_params.num_lights;

        // Env update — shipped every tick (cheap; render writes a few
        // u32-sized queue.write_buffers). Same drop-safety rationale
        // as `materials`.
        let env_update = crate::render_frame::EnvUpdate {
            exposure: self.environment.exposure,
            bloom_threshold: self.environment.bloom_threshold,
            bloom_knee: self.environment.bloom_knee,
            bloom_intensity: self.environment.bloom_intensity,
        };
        // Clear the legacy flag for other consumers; render no longer
        // gates on it.
        self.environment_dirty = false;

        // 0c. Rebuild GPU objects from ECS world only when
        //     transforms/objects/membership changed.
        let gpu_objects_dirty_this_frame = self.gpu_objects_dirty;
        if self.gpu_objects_dirty {
            self.update_scene_gpu();
            self.gpu_objects_dirty = false;
        }

        let t_cpu_setup = frame_start.elapsed();

        // 1. Geometry epoch — read lock-free via the shared atomic
        //    handle. Render compares against its own last-uploaded
        //    epoch and re-uploads when behind. Robust to dropped
        //    snapshots: the next snapshot still carries the latest
        //    epoch, so render always catches up.
        //
        //    The lock-free read is what keeps sim at 60 Hz while
        //    bake_worker is busy — taking `scene_mgr.lock()` here
        //    would block sim for the full duration of any bake
        //    integrate (50 ms+).
        //
        //    The legacy `self.geometry_dirty` flag is kept for collider
        //    rebuild scheduling (independent of GPU upload). It's set
        //    by every code path that mutates scene geometry.
        let geometry_epoch = self
            .geometry_epoch_handle
            .load(std::sync::atomic::Ordering::Acquire);
        if self.geometry_dirty {
            self.collider_caches_dirty = true;
            self.geometry_dirty = false;
        }
        if self.collider_caches_dirty {
            self.rebuild_collider_caches();
            self.collider_caches_dirty = false;
        }

        // 2. Bone matrix bytes for shading (LBS + DQ paths).
        let bone_matrix_lbs = self.bone_matrix_allocator.bytes().to_vec();
        let bone_matrix_dqs = self.bone_matrix_allocator.bytes_dq().to_vec();

        // 2b. Skin scatter — fold per-entity dispatches into one
        //     batched compute dispatch sim-side; render fires the
        //     batch on its thread. `skin_reuse` short-circuits when
        //     every skinned pose was byte-identical to the previous
        //     frame (paused animation), in which case the bone_field
        //     buffer from last frame is still valid and the scatter
        //     can skip entirely.
        let skin = if self.skinning_enabled
            && !self.skin_dispatches.is_empty()
            && !self.skin_reuse
        {
            self.skin_batch.clear();
            for plan in &self.skin_dispatches {
                let d = rkp_render::SkinDispatch {
                    uniforms: plan.uniforms,
                    bricks: &plan.bricks,
                };
                self.skin_batch.push(&d);
            }
            Some(crate::render_frame::RenderSkin {
                bone_field_bytes: self.skin_bone_field_bytes,
                bone_field_occ_bytes: self.skin_bone_field_occ_bytes,
                batch: self.skin_batch.clone(),
            })
        } else {
            if self.skinning_enabled && self.frame_index % 60 == 0 {
                // Once a second, log why scatter isn't running when
                // the user has the toggle on — most common reason is
                // a stale `.rkp` without the new skin-meta section.
                let skinned_entities = self
                    .world
                    .query::<&crate::components::Skeleton>()
                    .iter()
                    .count();
                if skinned_entities > 0 {
                    eprintln!(
                        "[RkpEngine] skinning enabled, {} skinned entities, but 0 scatter dispatches this frame. \
                         Likely cause: stale .rkp without skin-meta section — re-import the asset.",
                        skinned_entities,
                    );
                }
            }
            None
        };

        // 3. Per-viewport snapshot build — derive every per-VR
        //    parameter the render thread needs from current sim state
        //    and stash it in `viewports` for the snapshot. No GPU
        //    calls; the render thread does all the actual encoding
        //    and submission against this data.
        let visible_ids: Vec<ViewportId> = self
            .viewports
            .iter()
            .filter(|(_, v)| v.visible)
            .map(|(id, _)| *id)
            .collect();

        // Gizmo overlay is drawn on MAIN only — selection state is global.
        let gizmo_verts_main = self.build_gizmo_wireframe();
        let mut vp_list: Vec<crate::render_frame::RenderViewport> =
            Vec::with_capacity(visible_ids.len());

        for &viewport_id in &visible_ids {
            let cam_uniforms = self.build_camera_uniforms(viewport_id);
            let (vp_w, vp_h) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.width, v.height))
                .expect("viewport must exist");

            // Per-viewport screen-AABBs (camera-dependent) for tile cull.
            let vp_matrix = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            let screen_aabbs = crate::scene_sync::compute_screen_aabbs(
                &self.gpu_objects,
                &vp_matrix,
                vp_w as f32,
                vp_h as f32,
            );
            let screen_aabbs_bytes: Vec<u8> = bytemuck::cast_slice(&screen_aabbs).to_vec();
            // Per-tile object lists — replaces the 32-object bitmask so
            // the march shader handles arbitrary scene object counts.
            let tile_lists = crate::scene_sync::build_tile_lists(
                &screen_aabbs, vp_w, vp_h,
            );
            let tile_offsets_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.offsets).to_vec();
            let tile_object_ids_bytes: Vec<u8> =
                bytemuck::cast_slice(&tile_lists.object_ids).to_vec();
            let tile_count_x = tile_lists.tile_count_x;

            // Per-VR vol/cloud/atmo/god-ray params — derived from
            // environment + this VR's camera. Render writes them into
            // the corresponding per-VR uniform buffers right before
            // submit (one submit per VR keeps the writes correctly
            // paired with their dispatches).
            let vol_params = self.environment.to_volumetric_params(
                &cam_uniforms,
                vp_w,
                vp_h,
                self.frame_index as u32,
            );
            let cloud_params =
                self.environment.to_cloud_params(self.frame_index as f32 / 60.0);

            let sun_d = self.environment.sun_direction();
            let cam_y_vp = cam_uniforms.position[1];
            let atmo_frame = rkp_render::rkp_atmosphere::AtmosphereFrameParams {
                sun_dir: [-sun_d[0], -sun_d[1], -sun_d[2]],
                sun_intensity: self.environment.sun_intensity,
                camera_altitude: self.environment.effective_altitude(cam_y_vp),
                ground_albedo: self.environment.ground_albedo,
                cam_pos: [
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                ],
                _pad1b: 0.0,
                cam_forward: [
                    cam_uniforms.forward[0],
                    cam_uniforms.forward[1],
                    cam_uniforms.forward[2],
                ],
                _pad2: 0.0,
                cam_right: [
                    cam_uniforms.right[0],
                    cam_uniforms.right[1],
                    cam_uniforms.right[2],
                ],
                _pad3: 0.0,
                cam_up: [
                    cam_uniforms.up[0],
                    cam_uniforms.up[1],
                    cam_uniforms.up[2],
                ],
                _pad4: 0.0,
            };

            let god_ray_params = {
                let sun_toward = [-sun_d[0], -sun_d[1], -sun_d[2]];
                let sun_world = glam::Vec3::new(
                    cam_uniforms.position[0] + sun_toward[0] * 1000.0,
                    cam_uniforms.position[1] + sun_toward[1] * 1000.0,
                    cam_uniforms.position[2] + sun_toward[2] * 1000.0,
                );
                let clip = vp_matrix * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);
                let sun_on_screen = if clip.w > 0.0 { 1.0 } else { 0.0 };
                let ndc = if clip.w > 0.0 {
                    glam::Vec2::new(clip.x / clip.w, clip.y / clip.w)
                } else {
                    glam::Vec2::ZERO
                };
                let sun_uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];
                rkp_render::rkp_god_rays::GodRayParams {
                    sun_screen_pos: sun_uv,
                    sun_on_screen,
                    density: self.environment.god_ray_density,
                    weight: self.environment.god_ray_weight,
                    decay: self.environment.god_ray_decay,
                    exposure: self.environment.god_ray_exposure,
                    num_samples: 64,
                    sun_color: self.environment.sun_tint(cam_y_vp),
                    _pad: 0.0,
                }
            };

            let (vp_mode, vp_preview_mode) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.mode, v.preview_mode))
                .unwrap_or((
                    rkp_render::RenderMode::InSitu,
                    rkp_render::BuildPreviewMode::Voxel,
                ));

            // The procedural being previewed in raymarch mode is
            // always the currently-selected entity — keeps the
            // preview following selection automatically.
            let vp_preview_entity = self.selected_entity.and_then(|entity| {
                if self
                    .world
                    .get::<&crate::components::ProceduralGeometry>(entity)
                    .is_ok()
                {
                    self.entity_uuids.get(&entity).copied()
                } else {
                    None
                }
            });

            // Per-VR shade params: same scene-wide values plus the
            // per-VR `isolation` flag and a clamp on the light count
            // when isolated (so the BUILD preview doesn't pick up
            // the main scene's point lights).
            let isolation = matches!(vp_mode, rkp_render::RenderMode::Isolation);
            let mut shade_params_vr = self.shade_params_base;
            shade_params_vr.isolation = isolation as u32;
            if isolation {
                shade_params_vr.num_lights = shade_params_vr.num_lights.min(1);
            }
            let bloom_composite_intensity = if isolation {
                0.0
            } else {
                self.environment.bloom_intensity
            };

            // Procedural raymarch state — only when this VR is in
            // raymarch preview mode AND a procedural entity is
            // selected. Sim flattens the tree, builds the AABB, and
            // pre-filters ghost primitives; render uploads + binds.
            let proc_raymarch =
                if matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
                    let entity = vp_preview_entity.and_then(|uuid| {
                        self.entity_uuids
                            .iter()
                            .find_map(|(e, u)| (*u == uuid).then_some(*e))
                    });

                    let (instructions, aabb_min, aabb_max) = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| {
                                    let ins = rkp_procedural::flatten_tree(&pg.tree);
                                    let bounds = rkp_procedural::compute_bounds(&pg.tree);
                                    (ins, bounds.min, bounds.max)
                                })
                        })
                        // Empty-AABB sentinel: -1..+1 degenerate box
                        // any sane ray-AABB slab test fails. Covers
                        // "raymarch enabled but no procedural entity
                        // selected" so we don't get a bogus hit.
                        .unwrap_or_else(|| {
                            (Vec::new(), glam::Vec3::splat(1.0), glam::Vec3::splat(-1.0))
                        });

                    // Any stable per-entity u32 works — the shader
                    // packs it into the material G channel for the
                    // (now-unused) old 8-bit pick byte; retained here
                    // only as a non-breaking placeholder until
                    // `ProcRaymarchParams.object_id` gets cleaned up.
                    let object_id = entity.map(|e| e.to_bits().get() as u32).unwrap_or(0);

                    let entity_world = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::Transform>(e)
                                .ok()
                                .map(|xf| {
                                    glam::Affine3A::from_scale_rotation_translation(
                                        xf.scale,
                                        glam::Quat::from_euler(
                                            glam::EulerRot::XYZ,
                                            xf.rotation.x.to_radians(),
                                            xf.rotation.y.to_radians(),
                                            xf.rotation.z.to_radians(),
                                        ),
                                        xf.position,
                                    )
                                })
                        })
                        .unwrap_or(glam::Affine3A::IDENTITY);

                    // Ghost overlay: every cutter-role primitive,
                    // regardless of selection. Filter the flattened
                    // instruction stream so ghost renders use the
                    // same composed transforms the main raymarch does.
                    let ghost_ids = entity
                        .and_then(|e| {
                            self.world
                                .get::<&crate::components::ProceduralGeometry>(e)
                                .ok()
                                .map(|pg| collect_ghost_primitives(&pg.tree))
                        })
                        .unwrap_or_default();
                    let ghost_set: std::collections::HashSet<u32> =
                        ghost_ids.into_iter().collect();
                    let ghost_instructions: Vec<rkp_procedural::ProcInstruction> = instructions
                        .iter()
                        .filter(|ins| ghost_set.contains(&ins.node_id))
                        .copied()
                        .collect();

                    Some(crate::render_frame::RenderProcRaymarch {
                        instructions,
                        ghost_instructions,
                        object_id,
                        entity_world,
                        aabb_min,
                        aabb_max,
                        selected_node: self.selected_procedural_node,
                    })
                } else {
                    None
                };

            // Wireframe verts: gizmo on MAIN, procedural-node gizmo
            // on BUILD when in raymarch preview. The procedural-node
            // gizmo is only meaningful in raymarch mode — in voxel
            // mode the user sees the baked result and any drag would
            // silently edit the tree without visual feedback.
            let wireframe_verts = if viewport_id == ViewportId::MAIN {
                gizmo_verts_main.clone()
            } else if viewport_id == ViewportId::BUILD
                && matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch)
            {
                let cam_pos = glam::Vec3::new(
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                );
                self.build_procedural_gizmo_wireframe(cam_pos)
            } else {
                Vec::new()
            };

            // Editor-overlay gate. MAIN gates on the EDITOR_ONLY
            // layer bit (off in play mode); BUILD always shows its
            // proc-gizmo when one's present.
            let show_editor_overlays = if viewport_id == ViewportId::MAIN {
                self.viewports
                    .get(ViewportId::MAIN)
                    .map(|v| v.filter.base_layers & crate::viewport::layer::EDITOR_ONLY != 0)
                    .unwrap_or(false)
            } else {
                true
            };

            // BUILD: pin the studio-floor grid under the previewed
            // entity instead of world origin. Without this, moving
            // the entity in world-Y leaves the grid at y=0 while the
            // camera orbits around the entity, so the object floats
            // relative to the grid.
            let grid_override = if viewport_id == ViewportId::BUILD {
                let p = proc_raymarch
                    .as_ref()
                    .map(|p| p.entity_world.translation)
                    .unwrap_or(glam::Vec3A::ZERO);
                Some(rkp_render::rkp_grid::GridParams {
                    plane_origin: [p.x, p.y, p.z, 0.0],
                    ..Default::default()
                })
            } else {
                None
            };

            vp_list.push(crate::render_frame::RenderViewport {
                id: viewport_id,
                width: vp_w,
                height: vp_h,
                mode: vp_mode,
                preview_mode: vp_preview_mode,
                camera: cam_uniforms,
                screen_aabbs_bytes,
                tile_offsets_bytes,
                tile_object_ids_bytes,
                tile_count_x,
                vp_matrix,
                vol_params,
                cloud_params,
                atmo_frame,
                god_ray_params,
                shade_params: shade_params_vr,
                bloom_composite_intensity,
                grid_override,
                wireframe_verts,
                show_editor_overlays,
                proc_raymarch,
            });

            // Update sim-side `prev_view_proj` so next frame's
            // CameraUniforms carry the right reprojection matrix for
            // cloud TAA / temporal upscale.
            if let Some(v) = self.viewports.get_mut(viewport_id) {
                v.prev_view_proj = cam_uniforms.view_proj;
            }
        }

        // 4. Pending pick — convert sim's `PendingPick` (which carries
        //    a CPU-resolved ghost hint) to the render-side struct.
        //    Ghost hint stays sim-side; we'll re-apply it when the
        //    matching `PickResult` comes back.
        //
        //    Re-ship every snapshot until [`process_pick_result`]
        //    clears `self.pending_pick`. Picks used to be cleared
        //    eagerly with `take()`, but the GPU-backpressure backoff
        //    in `render_worker` now causes the inbox (newest-wins) to
        //    drop a sizeable fraction of snapshots before render sees
        //    them — eager-clearing meant the click was lost forever
        //    whenever its carrier snapshot got dropped. Re-shipping
        //    is safe because render's `pick_in_flight` gate dedupes
        //    duplicates: at most one map_async is ever in flight per
        //    pick request.
        let pending_pick = if let Some(pp) = self.pending_pick.as_ref() {
            // Map viewport+preview-mode → kind. BUILD raymarch decodes
            // the gbuf_pick texture for procedural NodeIds; everything
            // else (MAIN voxel, BUILD voxel) decodes gbuf_material for
            // the entity scene_id.
            let kind = if pp.viewport == ViewportId::BUILD
                && self
                    .viewports
                    .get(ViewportId::BUILD)
                    .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                    .unwrap_or(false)
            {
                crate::render_frame::PickKind::ProceduralNode
            } else {
                crate::render_frame::PickKind::Material
            };
            self.in_flight_pick_ghost = pp.ghost_pick_node_id;
            Some(crate::render_frame::PendingPick {
                viewport: pp.viewport,
                x: pp.x,
                y: pp.y,
                kind,
            })
        } else {
            None
        };

        // 5. Build + submit the snapshot. `submit` is non-blocking;
        //    if render hadn't consumed the previous snapshot yet,
        //    that one is dropped (newest-wins). Sim never stalls on
        //    render's GPU rate.
        let frame = crate::render_frame::RenderFrame {
            frame_index: self.frame_index,
            gpu_objects: self.gpu_objects.clone(),
            gpu_objects_dirty: gpu_objects_dirty_this_frame,
            geometry_epoch,
            materials,
            lights: gpu_lights,
            shade_params_base: self.shade_params_base,
            env_update,
            viewports: vp_list,
            skin,
            bone_matrix_lbs,
            bone_matrix_dqs,
            pending_pick,
            cloud_sun_atten: self.cloud_sun_atten,
            lod_enabled: self.lod_enabled,
            surfacenet_enabled: self.surfacenet_enabled,
            shadow_steps: self.environment.shadow_steps,
        };

        let t_encode = frame_start.elapsed();
        self.render_worker.inbox.submit(frame);
        let t_frame_end = frame_start.elapsed();

        // 6. Push CPU-side timings into profiling history. GPU pass
        //    timings get stitched into the most-recent sample by
        //    `drain_render_results` once the render thread publishes
        //    them (typically 1-2 frames behind sim).
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        let cpu = crate::profiling::CpuPhaseTimings {
            setup_ms: ms(t_cpu_setup),
            snapshot_ms: ms(t_encode - t_cpu_setup),
            submit_ms: ms(t_frame_end - t_encode),
            total_ms: ms(t_frame_end),
        };
        self.profiling.push(crate::profiling::FrameSample {
            frame_idx: self.frame_index,
            cpu,
            // Both filled in by `drain_render_results` once the render
            // thread publishes the matching frame's `RenderResult`.
            // Lag is typically 1-2 frames, fine for display.
            gpu_passes: Vec::new(),
            render_dt_ms: 0.0,
            gpu_object_count: self.gpu_objects.len() as u32,
        });

        self.frame_index += 1;
    }

    /// Drain every [`RenderResult`] the render thread has published
    /// since the previous tick. Applies pick decoding, updates the
    /// smoothed-cloud-sun-atten target, and stitches GPU pass timings
    /// into the most-recent profiling sample.
    ///
    /// Called from the top of [`Self::render_frame`]; safe to call
    /// when the channel is empty (no-op).
    fn drain_render_results(&mut self) {
        // Take a Vec rather than reborrow `self.render_worker` inside
        // the loop — pick processing wants `&mut self`, which would
        // otherwise alias the channel borrow.
        let mut latest_atten: Option<f32> = None;
        // (frame_idx, gpu_passes, render_dt_ms) — kept together so a
        // single `attach_render_data` call updates both fields on the
        // matching ring entry.
        let mut latest_render_data: Option<(u64, Vec<(String, f32)>, Option<f32>)> = None;
        let mut pick_results: Vec<crate::render_frame::PickResult> = Vec::new();
        // EMA alpha for the render-FPS readout. 0.1 = ~25-tick
        // settling at 60 Hz; same time-constant the tick/physics
        // EMAs use, keeps the panel readouts feeling consistent.
        const RENDER_HZ_EMA_ALPHA: f32 = 0.1;
        while let Ok(result) = self.render_worker.outbox.try_recv() {
            if !result.cloud_sun_atten_raw.is_nan() {
                latest_atten = Some(result.cloud_sun_atten_raw);
            }
            // Track latest result that carries either GPU passes OR a
            // render dt — both are stitched onto the matching frame
            // sample. We always overwrite so the "latest" wins on
            // multi-result drains; correlation by frame_index keeps
            // attribution honest even if results arrive out of order
            // (which they shouldn't, but the API doesn't forbid it).
            if !result.gpu_passes.is_empty() || result.render_dt_ms.is_some() {
                latest_render_data = Some((
                    result.frame_index,
                    result.gpu_passes,
                    result.render_dt_ms,
                ));
            }
            if let Some(pr) = result.pick_result {
                pick_results.push(pr);
            }
            // Fold render thread's observed iteration interval into
            // the FPS EMA. Skip the first iteration (`None`) and
            // any zero/negative dt (paranoia — clock can rarely tie
            // on the same nanosecond).
            if let Some(dt_ms) = result.render_dt_ms {
                if dt_ms > 0.0 {
                    let inst_hz = 1000.0 / dt_ms;
                    self.render_hz_ema = self.render_hz_ema * (1.0 - RENDER_HZ_EMA_ALPHA)
                        + inst_hz * RENDER_HZ_EMA_ALPHA;
                }
            }
        }
        if let Some(a) = latest_atten {
            self.last_cloud_sun_atten_raw = a;
        }
        if let Some((frame_idx, passes, dt)) = latest_render_data {
            self.profiling.attach_render_data(frame_idx, passes, dt);
        }
        for pr in pick_results {
            self.process_pick_result(pr);
        }
    }

    /// Apply a [`PickResult`] from the render thread. Mirrors the old
    /// `drain_pick_result` logic: ghost priority on BUILD raymarch,
    /// otherwise scene_id → entity for MAIN voxel and `selected_entity` /
    /// `selected_procedural_node` updates accordingly.
    fn process_pick_result(&mut self, pr: crate::render_frame::PickResult) {
        use crate::render_frame::PickKind;
        use crate::viewport::ViewportId;

        // Acknowledge: this pick request has been served, so stop
        // re-shipping it in subsequent snapshots. (See the "re-ship
        // every snapshot" rationale in `submit_render_frame`.) A
        // brand-new click later will set `pending_pick` again.
        self.pending_pick = None;

        // Drop-on-geometry: if a drag-drop is queued for this viewport,
        // consume it instead of running selection. The pick was issued
        // purely to get the world-space surface position at the drop
        // pixel; selection should not change from a drop.
        if let Some(drop) = self.pending_drop.as_ref() {
            if drop.viewport == pr.viewport {
                let drop = self.pending_drop.take().unwrap();
                self.handle_drop(drop, pr.position);
                self.in_flight_pick_ghost = None;
                return;
            }
        }

        // Drag-preview: move the preview to the freshest surface snap.
        // Skip the selection path entirely — picks issued while
        // dragging are purely for positioning.
        if let Some(preview) = self.drag_preview.as_ref() {
            if preview.viewport == pr.viewport {
                let kind = preview.kind.clone();
                let (cx, cy) = preview.last_cursor;
                let vp = preview.viewport;

                // Self-hit detection only matters for the model path —
                // generators have nothing spawned yet to self-hit. For
                // models, ignore picks that land on the preview entity.
                // `raw_payload[0]` is the 32-bit pick channel — gpu_idx
                // on hit, 0xFFFFFFFF on sky.
                let hit_gpu_idx = if pr.raw_payload[0] != u32::MAX {
                    Some(pr.raw_payload[0] as usize)
                } else {
                    None
                };
                let hit_self = match &kind {
                    DragPreviewKind::Model { entity, .. } => {
                        hit_gpu_idx.is_some_and(|idx| {
                            self.entity_to_gpu.get(entity).copied() == Some(idx)
                        })
                    }
                    DragPreviewKind::Generator { .. } => false,
                };

                // Resolve target world position:
                //   1. Valid surface hit (not self) → use it, cache it.
                //   2. Self-hit or sky miss with a cached pos → keep that.
                //   3. No cache yet → ground-plane ray at the cursor.
                let new_pos = if let Some(hit) = pr.position.filter(|_| !hit_self) {
                    self.drag_preview.as_mut().unwrap().last_surface_pos = Some(hit);
                    Some(hit)
                } else if let Some(cached) = self.drag_preview.as_ref()
                    .and_then(|p| p.last_surface_pos)
                {
                    Some(cached)
                } else {
                    let (ro, rd) = self.screen_to_ray_for_viewport(vp, cx as f32, cy as f32);
                    if rd.y.abs() > 1e-6 {
                        let t = -ro.y / rd.y;
                        if t > 0.0 { Some(ro + rd * t) } else { None }
                    } else { None }
                };

                if let Some(p) = new_pos {
                    match kind {
                        DragPreviewKind::Model { entity, aabb_min_y } => {
                            // Bottom-snap the asset so its feet sit on
                            // the surface under the cursor.
                            if let Ok(mut t) = self.world
                                .get::<&mut crate::components::Transform>(entity)
                            {
                                t.position = glam::Vec3::new(p.x, p.y - aabb_min_y, p.z);
                            }
                            self.gpu_objects_dirty = true;
                        }
                        DragPreviewKind::Generator { .. } => {
                            // Gizmo-only — `last_surface_pos` is the
                            // whole state. Update and redraw at frame
                            // start via `build_gizmo_wireframe`.
                            self.drag_preview.as_mut().unwrap().last_surface_pos = Some(p);
                        }
                    }
                }
                self.in_flight_pick_ghost = None;
                return;
            }
        }

        match pr.kind {
            PickKind::ProceduralNode => {
                // Ghost priority: if the CPU raycast at click time
                // found a ghost primitive on the ray, that wins —
                // matches "translucent overlay on top owns the click."
                if let Some(ghost_id) = self.in_flight_pick_ghost.take() {
                    self.selected_procedural_node = Some(ghost_id);
                    return;
                }
                // Proc raymarch writes the primitive NodeId into the
                // low 16 bits of the 32-bit pick channel, with 0xFFFF
                // as the sky sentinel.
                let node_id_16 = pr.raw_payload[0] & 0xFFFFu32;
                if node_id_16 != 0xFFFFu32 {
                    self.selected_procedural_node = Some(node_id_16);
                } else {
                    self.selected_procedural_node = None;
                }
            }
            PickKind::Material => {
                // Voxel march writes `gpu_idx` to the 32-bit pick
                // channel, 0xFFFFFFFF on sky. Direct lookup — no
                // bit-unpacking, no 255-object cap.
                let pick = pr.raw_payload[0];
                if pick != u32::MAX {
                    let gpu_idx = pick as usize;
                    if gpu_idx < self.gpu_to_entity.len() {
                        self.selected_entity = Some(self.gpu_to_entity[gpu_idx]);
                    }
                } else {
                    self.selected_entity = None;
                }
                // Discard the ghost hint either way — Material picks
                // never hit ghost-primitive priority.
                self.in_flight_pick_ghost = None;
                let _ = ViewportId::MAIN;
            }
        }
    }

    /// Apply a queued drop. `surface_pos` is `Some(hit)` when the pick
    /// readback sampled valid geometry (hit_distance < 1e9); otherwise
    /// we cast a ray through the drop pixel and intersect the Y=0
    /// ground plane as a fallback. If that ray also misses (looking
    /// up, no floor), log and skip — no silent spawn at the origin.
    fn handle_drop(&mut self, drop: PendingDrop, surface_pos: Option<glam::Vec3>) {
        let pos = if let Some(p) = surface_pos {
            p
        } else {
            let (ray_o, ray_d) = self.screen_to_ray_for_viewport(
                drop.viewport, drop.x as f32, drop.y as f32,
            );
            // Ground plane at y=0. `t > 0` means "plane is ahead of
            // the camera along the ray"; negative t (looking up) is a
            // miss.
            if ray_d.y.abs() > 1e-6 {
                let t = -ray_o.y / ray_d.y;
                if t > 0.0 {
                    ray_o + ray_d * t
                } else {
                    self.console.warn(format!(
                        "Drop missed geometry and the ground plane is behind the camera — skipping."
                    ));
                    return;
                }
            } else {
                self.console.warn(format!(
                    "Drop ray parallel to ground plane — skipping."
                ));
                return;
            }
        };
        match drop.action {
            PendingDropAction::Asset { path } => {
                self.spawn_asset(&path, pos);
            }
            PendingDropAction::Generator { name } => {
                self.spawn_generator(&name, Some(pos));
            }
            PendingDropAction::GeneratorPreset { path } => {
                self.spawn_generator_preset(&path, Some(pos));
            }
        }
    }

    /// On PlayStart: hand the MAIN viewport over to the active scene
    /// camera (if one exists) and flip its layer mask so editor-only
    /// helpers vanish and HUD becomes visible.
    fn enter_play_mode_viewports(&mut self) {
        use crate::components::{Camera, EditorMetadata};
        use crate::viewport::{layer, CameraSource, SceneFilter, ViewportId};

        // Find the scene camera flagged active. If multiple are flagged,
        // pick the first the iteration yields — scene authoring should
        // ensure exactly one.
        let scene_cam = self.world.query::<&Camera>().iter()
            .find(|(_, c)| c.active)
            .map(|(e, _)| e);

        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            if let Some(entity) = scene_cam {
                let name = self.world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{entity:?}"));
                self.console.info(format!("Play mode: camera → '{name}'"));
                main.runtime_override = Some(CameraSource::Entity(entity));
            } else {
                self.console.warn("Play mode: no active scene camera found, \
                                   keeping editor camera");
            }
            main.filter = SceneFilter {
                base_layers: layer::DEFAULT | layer::UI,
                focus_entity: None,
            };
        }
    }

    /// Apply a scripted viewport request (from the behavior system).
    /// Currently all requests target MAIN; per-viewport routing would use
    /// a `ViewportId` payload on the request enum.
    fn apply_viewport_request(&mut self, req: crate::behavior::ViewportRequest) {
        use crate::behavior::ViewportRequest;
        use crate::viewport::{CameraSource, ViewportId};
        let Some(main) = self.viewports.get_mut(ViewportId::MAIN) else { return };
        match req {
            ViewportRequest::SetActiveCamera(entity) => {
                main.runtime_override = Some(CameraSource::Entity(entity));
            }
            ViewportRequest::ClearActiveCamera => {
                main.runtime_override = None;
            }
        }
    }

    /// On PlayStop: clear the runtime override and restore the editor
    /// layer mask. The editor camera state was untouched throughout play
    /// mode, so the user lands exactly where they left off.
    fn exit_play_mode_viewports(&mut self) {
        use crate::viewport::{layer, SceneFilter, ViewportId};
        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            main.runtime_override = None;
            main.filter = SceneFilter {
                base_layers: layer::DEFAULT | layer::EDITOR_ONLY,
                focus_entity: None,
            };
        }
    }

    /// Read the `(position, yaw, pitch, fov, near, far)` 6-tuple from a
    /// scene-camera entity. Returns `None` if the entity is missing
    /// either a `Transform` or `Camera` component, or has been despawned —
    /// callers fall back to the editor camera in that case.
    ///
    /// Yaw/pitch derive from the Transform's Euler rotation: yaw is Y in
    /// radians, pitch is X in radians (matching the editor's fly-camera
    /// convention so play-mode → edit-mode "Look Through" stays continuous).
    fn read_entity_camera(&self, entity: hecs::Entity)
        -> Option<(glam::Vec3, f32, f32, f32, f32, f32)>
    {
        use crate::components::{Camera, Transform};
        let transform = self.world.get::<&Transform>(entity).ok()?;
        let cam = self.world.get::<&Camera>(entity).ok()?;
        let yaw = transform.rotation.y.to_radians();
        let pitch = transform.rotation.x.to_radians();
        Some((transform.position, yaw, pitch, cam.fov, cam.near, cam.far))
    }

    /// Mirror the legacy `self.camera` state into `viewports[MAIN].editor_camera`.
    /// Phase 1 keeps the legacy field as the source of truth; the viewport copy
    /// is kept in sync so later phases can flip the dependency direction
    /// without surprises.
    fn sync_main_viewport_from_legacy_camera(&mut self) {
        use crate::viewport::{EditorCamera, FlyCameraState, ViewportId};
        if let Some(main) = self.viewports.get_mut(ViewportId::MAIN) {
            main.editor_camera = EditorCamera::Fly(FlyCameraState {
                position: self.camera.position,
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
                fov: self.camera.fov,
                near: self.camera.near,
                far: self.camera.far,
            });
        }
    }

    fn build_camera_uniforms(&self, viewport_id: crate::viewport::ViewportId)
        -> rkp_render::rkp_scene::CameraUniforms
    {
        use crate::viewport::{CameraSource, EditorCamera, ViewportId};
        let viewport = self.viewports
            .get(viewport_id)
            .expect("build_camera_uniforms: viewport must exist");

        // Camera resolution priority (Phase 5):
        //   1. runtime_override → entity's Transform + Camera components
        //   2. MAIN: legacy `self.camera` (still source of truth, synced
        //      into editor_camera by sync_main_viewport_from_legacy_camera)
        //   3. Other viewports: their own editor_camera
        let from_entity = viewport.runtime_override.and_then(|src| match src {
            CameraSource::Entity(entity) => self.read_entity_camera(entity),
        });
        let (position, yaw, pitch, fov, near, far) = if let Some(c) = from_entity {
            c
        } else if viewport_id == ViewportId::MAIN {
            (self.camera.position, self.camera.yaw, self.camera.pitch,
             self.camera.fov, self.camera.near, self.camera.far)
        } else {
            match viewport.editor_camera {
                EditorCamera::Fly(s) => (s.position, s.yaw, s.pitch, s.fov, s.near, s.far),
                EditorCamera::Turntable(t) => {
                    // Convert orbit (yaw/pitch + distance about target) to
                    // equivalent eye-position + look direction.
                    let dir = glam::Vec3::new(
                        -t.yaw.sin() * t.pitch.cos(),
                        t.pitch.sin(),
                        -t.yaw.cos() * t.pitch.cos(),
                    );
                    let position = t.target - dir * t.distance;
                    (position, t.yaw, t.pitch, t.fov, t.near, t.far)
                }
            }
        };

        let forward = glam::Vec3::new(
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ).normalize();
        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        let fov_rad = fov.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = viewport.width as f32 / viewport.height.max(1) as f32;

        let view = glam::Mat4::look_to_rh(position, forward, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(fov_rad, aspect, near, far);
        let view_proj = proj * view;

        // Render-layer + focus filter from this viewport's SceneFilter.
        // u32::MAX defaults pass everything (no real object_id is u32::MAX
        // since they're sequential from 0).
        let focus_object_id = viewport.filter.focus_entity
            .and_then(|e| self.entity_to_gpu.get(&e).copied())
            .map(|idx| idx as u32)
            .unwrap_or(u32::MAX);
        let layer_mask = viewport.filter.base_layers;

        rkp_render::rkp_scene::CameraUniforms {
            position: [position.x, position.y, position.z, 1.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            right: [right.x * half_fov_tan * aspect, right.y * half_fov_tan * aspect, right.z * half_fov_tan * aspect, 0.0],
            up: [up.x * half_fov_tan, up.y * half_fov_tan, up.z * half_fov_tan, 0.0],
            resolution: [viewport.width as f32, viewport.height as f32],
            jitter: [0.0, 0.0],
            layer_mask,
            focus_object_id,
            _pad: [0; 2],
            prev_vp: viewport.prev_view_proj,
            view_proj: view_proj.to_cols_array_2d(),
        }
    }

    /// Read-modify-write a procedural node's local `Affine3A`.
    ///
    /// The stored transform is a full SRT compose, not just a
    /// translation — so each of the three `SetProceduralNode*` commands
    /// must preserve the two components it doesn't own. `f` takes
    /// `(scale, rotation, translation)` as decomposed from the current
    /// transform and returns the new triple.
    fn update_procedural_node_transform(
        &mut self,
        node_id: u32,
        f: impl FnOnce(glam::Vec3, glam::Quat, glam::Vec3) -> (glam::Vec3, glam::Quat, glam::Vec3),
    ) {
        let entity = match self.selected_entity {
            Some(e) => e,
            None => return,
        };
        let Ok(mut proc_geo) = self
            .world
            .get::<&mut crate::components::ProceduralGeometry>(entity)
        else {
            return;
        };
        let id = rkp_procedural::NodeId(node_id);
        let current = match proc_geo.tree.get(id) {
            Some(n) => n.transform,
            None => return,
        };

        // Decompose current transform.
        let t = current.translation.into();
        let m = current.matrix3;
        let sx = glam::Vec3::from(m.x_axis).length();
        let sy = glam::Vec3::from(m.y_axis).length();
        let sz = glam::Vec3::from(m.z_axis).length();
        let scale = glam::Vec3::new(sx.max(1e-8), sy.max(1e-8), sz.max(1e-8));
        let rot_mat = glam::Mat3::from_cols(
            (glam::Vec3::from(m.x_axis) / scale.x).into(),
            (glam::Vec3::from(m.y_axis) / scale.y).into(),
            (glam::Vec3::from(m.z_axis) / scale.z).into(),
        );
        let rotation = glam::Quat::from_mat3(&rot_mat);

        let (new_scale, new_rot, new_t) = f(scale, rotation, t);
        let new_affine =
            glam::Affine3A::from_scale_rotation_translation(new_scale, new_rot, new_t);
        proc_geo.tree.set_transform(id, new_affine);
        proc_geo.dirty = true;
    }

    fn process_command(&mut self, cmd: EngineCommand) -> bool {
        match cmd {
            EngineCommand::Shutdown => return false,

            EngineCommand::SetCamera { id, position, yaw, pitch, fov } => {
                // Phase 3: only MAIN is wired to the legacy `self.camera`.
                // Non-MAIN viewports update their own editor_camera once
                // multi-viewport rendering lands (Phase 4+).
                if id == crate::viewport::ViewportId::MAIN {
                    self.camera.position = position;
                    self.camera.yaw = yaw;
                    self.camera.pitch = pitch;
                    self.camera.fov = fov;
                    self.sync_main_viewport_from_legacy_camera();
                } else if let Some(vp) = self.viewports.get_mut(id) {
                    use crate::viewport::{EditorCamera, FlyCameraState};
                    vp.editor_camera = EditorCamera::Fly(FlyCameraState {
                        position, yaw, pitch, fov,
                        near: 0.01, far: 1000.0,
                    });
                }
            }

            EngineCommand::Resize { id, width, height } => {
                // Each VR has its own per-resolution pass chain now, so
                // Resize is per-viewport. Resizing BUILD doesn't affect
                // MAIN (and vice versa). The editor sends Resize on
                // every event (mouse move etc.) and relies on this
                // handler to no-op when the size hasn't actually changed
                // — without that guard `vr.resize` rebuilds bloom /
                // tonemap each frame and `environment_dirty` ticks every
                // tick.
                let unchanged = self
                    .viewports
                    .get(id)
                    .map(|vp| vp.width == width && vp.height == height)
                    .unwrap_or(false);
                if unchanged {
                    return true;
                }
                let _ = self.render_worker.commands.send(
                    crate::render_frame::RenderCommand::ResizeViewport { id, width, height },
                );
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.width = width;
                    vp.height = height;
                }
                // `vr.resize` reconstructs the bloom + tonemap passes with
                // their hard-coded defaults, so the scene's exposure and
                // bloom knobs have to be re-uploaded afterwards on EVERY
                // resize — not just MAIN. Previously this was gated to
                // MAIN and BUILD's first Resize (sent by the editor when
                // the build panel sizes up) left it running with default
                // exposure → blown-out preview until something else
                // flipped environment_dirty back on.
                self.environment_dirty = true;
                if id == crate::viewport::ViewportId::MAIN {
                    // MAIN drives the legacy width/height on EngineState
                    // for hot paths that haven't migrated (sculpt/paint
                    // ray math).
                    self.width = width;
                    self.height = height;
                    self.environment_dirty = true;
                    self.environment_ui_dirty = true;
                    eprintln!("[RkpEngine] MAIN resized to {}x{}", width, height);
                }
            }

            EngineCommand::SetViewportVisible { id, visible } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.visible = visible;
                }
            }

            EngineCommand::SetViewportFilter { id, base_layers, focus_entity_id } => {
                let focus_entity = focus_entity_id
                    .and_then(|uuid| self.uuid_to_entity.get(&uuid).copied());
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.filter = crate::viewport::SceneFilter {
                        base_layers,
                        focus_entity,
                    };
                }
            }

            EngineCommand::SetViewportCamera { id, entity_id } => {
                if let Some(entity) = self.uuid_to_entity.get(&entity_id).copied() {
                    if let Some(vp) = self.viewports.get_mut(id) {
                        vp.runtime_override =
                            Some(crate::viewport::CameraSource::Entity(entity));
                    }
                }
            }

            EngineCommand::ClearViewportCamera { id } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.runtime_override = None;
                }
            }

            EngineCommand::SetViewportMode { id, mode } => {
                if let Some(vp) = self.viewports.get_mut(id) {
                    vp.mode = mode;
                }
            }

            EngineCommand::SpawnProceduralObject { name, leaf_kind } => {
                use crate::components::*;
                let name = self.unique_name(&name);
                let mut proc_geo = match leaf_kind {
                    Some(kind) => ProceduralGeometry::with_leaf(parse_node_kind(&kind)),
                    None => ProceduralGeometry::default_sphere(),
                };
                // Freshly-spawned procedurals should bake immediately so
                // the user sees a visible object. We set `pending_bake`
                // (not just `dirty`) so the debounced auto-bake path in
                // `update_dirty_procedurals` picks it up — scene *load*
                // deliberately never auto-bakes, so riding on `dirty`
                // alone would leave the spawn invisible.
                proc_geo.dirty = false;
                proc_geo.pending_bake = true;
                proc_geo.bake_dirty_at = Some(std::time::Instant::now());
                let entity = self.world.spawn((
                    Transform::default(),
                    EditorMetadata { name: name.clone() },
                    Renderable {
                        primitive: Some("procedural".to_string()),
                        voxel_count: 0,
                        spatial: None,
                        ..Default::default()
                    },
                    proc_geo,
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned procedural '{name}' (baking…)"));
            }

            EngineCommand::SelectProceduralNode { node_id } => {
                self.selected_procedural_node = node_id;
            }

            EngineCommand::SetProceduralVoxelSize { tier } => {
                const VOXEL_TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if let Ok(vs) = tier.parse::<f32>() {
                            let snapped = VOXEL_TIERS.iter()
                                .min_by(|a, b| ((**a) - vs).abs().partial_cmp(&((**b) - vs).abs()).unwrap())
                                .copied()
                                .unwrap_or(0.02);
                            if (snapped - proc_geo.voxel_size).abs() > 1e-6 {
                                proc_geo.voxel_size = snapped;
                                // Auto-bake — voxel-size changes are
                                // single-click tier flips; the debounce
                                // window absorbs rapid double-clicks but
                                // otherwise the user expects an immediate
                                // rebake.
                                proc_geo.pending_bake = true;
                                proc_geo.bake_dirty_at =
                                    Some(std::time::Instant::now());
                            }
                        }
                    }
                }
            }

            EngineCommand::AddProceduralNode { parent_node_id, kind } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let parent = rkp_procedural::NodeId(parent_node_id);
                        let node_kind = parse_node_kind(&kind);
                        // Root accepts children directly — no
                        // auto-promote, no special cases. Drops onto
                        // a leaf are rejected by the UI (is_leaf →
                        // no "+" affordance).
                        let new_id = proc_geo.tree.add_child(parent, node_kind);
                        proc_geo.dirty = true;
                        self.selected_procedural_node = Some(new_id.0);
                    }
                }
            }

            EngineCommand::RemoveProceduralNode { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = rkp_procedural::NodeId(node_id);
                        if proc_geo.tree.remove(id) {
                            proc_geo.dirty = true;
                            if self.selected_procedural_node == Some(node_id) {
                                self.selected_procedural_node = None;
                            }
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNodeUp { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_up(rkp_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNodeDown { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_down(rkp_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::ReparentProceduralNode { node_id, new_parent_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.reparent(
                            rkp_procedural::NodeId(node_id),
                            rkp_procedural::NodeId(new_parent_id),
                        ) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::MoveProceduralNode { node_id, new_parent_id, index } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if proc_geo.tree.move_to(
                            rkp_procedural::NodeId(node_id),
                            rkp_procedural::NodeId(new_parent_id),
                            index as usize,
                        ) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::DuplicateProceduralNode { node_id } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        if let Some(new_id) = proc_geo.tree.duplicate(rkp_procedural::NodeId(node_id)) {
                            proc_geo.dirty = true;
                            self.selected_procedural_node = Some(new_id.0);
                        }
                    }
                }
            }

            EngineCommand::SetProceduralNodeCombinator { node_id, kind } => {
                // Local helper — returns true when a kind change was
                // actually applied. Early-returns via `?` / plain
                // `return None` keep the body flat and side-step the
                // `continue` footgun (there's no outer loop here,
                // this is a one-off match arm).
                fn swap_kind(
                    proc_geo: &mut crate::components::ProceduralGeometry,
                    id: rkp_procedural::NodeId,
                    kind: &str,
                ) -> bool {
                    let node = match proc_geo.tree.get_mut(id) {
                        Some(n) => n,
                        None => return false,
                    };
                    // Only swap between combinators; silently ignore on
                    // leaves (UI should hide the menu there anyway, but
                    // defend at the boundary).
                    let current_mc = match &node.kind {
                        rkp_procedural::NodeKind::Union { material_combine }
                        | rkp_procedural::NodeKind::Intersect { material_combine } => {
                            Some(*material_combine)
                        }
                        rkp_procedural::NodeKind::Subtract => None,
                        _ => return false, // leaf
                    };
                    let new_kind = match kind {
                        "Union" => rkp_procedural::NodeKind::Union {
                            material_combine: current_mc
                                .unwrap_or(rkp_procedural::MaterialCombine::Winner),
                        },
                        "Intersect" => rkp_procedural::NodeKind::Intersect {
                            material_combine: current_mc
                                .unwrap_or(rkp_procedural::MaterialCombine::Winner),
                        },
                        "Subtract" => rkp_procedural::NodeKind::Subtract,
                        _ => return false,
                    };
                    // No-op when the user re-picks the current kind —
                    // without this the version bump would force a rebake.
                    let same_kind = matches!(
                        (&node.kind, &new_kind),
                        (rkp_procedural::NodeKind::Union { .. }, rkp_procedural::NodeKind::Union { .. })
                            | (rkp_procedural::NodeKind::Intersect { .. }, rkp_procedural::NodeKind::Intersect { .. })
                            | (rkp_procedural::NodeKind::Subtract, rkp_procedural::NodeKind::Subtract)
                    );
                    if same_kind {
                        return false;
                    }
                    node.kind = new_kind;
                    true
                }

                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = rkp_procedural::NodeId(node_id);
                        if swap_kind(&mut proc_geo, id, &kind) {
                            proc_geo.tree.bump_version(id);
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::SetProceduralNodePosition { node_id, position } => {
                self.update_procedural_node_transform(node_id, |s, r, _| (s, r, position));
            }

            EngineCommand::SetProceduralNodeRotation { node_id, rotation_deg } => {
                let rot = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    rotation_deg.x.to_radians(),
                    rotation_deg.y.to_radians(),
                    rotation_deg.z.to_radians(),
                );
                self.update_procedural_node_transform(node_id, |s, _, t| (s, rot, t));
            }

            EngineCommand::SetProceduralNodeScale { node_id, scale } => {
                self.update_procedural_node_transform(node_id, |_, r, t| (scale, r, t));
            }

            EngineCommand::SetProceduralNodeParam { node_id, param_name, value } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let id = rkp_procedural::NodeId(node_id);
                        if apply_procedural_param(&mut proc_geo.tree, id, &param_name, &value) {
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::BakeProceduralEntity { entity_id } => {
                let entity = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e));
                if let Some(entity) = entity {
                    self.enqueue_bake(entity);
                }
            }

            EngineCommand::BakeAllDirtyProcedurals => {
                use crate::components::*;
                let dirty: Vec<hecs::Entity> = self
                    .world
                    .query::<&ProceduralGeometry>()
                    .iter()
                    .filter(|(_, p)| p.dirty)
                    .map(|(e, _)| e)
                    .collect();
                for entity in dirty {
                    self.enqueue_bake(entity);
                }
            }

            EngineCommand::ConvertProceduralToVoxel { entity_id } => {
                use crate::components::*;
                let Some(entity) = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e))
                else {
                    self.console.warn("Convert: entity not found".to_string());
                    return true;
                };
                // Gate on a clean bake state — a pending/in-flight
                // bake means the voxels aren't what the user just
                // asked for.
                let can_convert = self
                    .world
                    .get::<&ProceduralGeometry>(entity)
                    .map(|pg| !pg.bake_in_flight && !pg.pending_bake && !pg.dirty)
                    .unwrap_or(false);
                if !can_convert {
                    self.console.warn(
                        "Convert: bake pending or in flight — let it settle first".to_string(),
                    );
                    return true;
                }
                // Hard requirements for promoting the procedural to a
                // first-class asset: an open project (so we have an
                // assets/ directory to write to) and a saved scene
                // (so the bake worker has been writing the cache to a
                // known location). Without either, we can't produce
                // a persistent on-disk asset for the converted voxels.
                let Some(project_dir) = self.project_dir.clone() else {
                    self.console.warn(
                        "Convert: open or save a project first so the converted asset has somewhere to live.".to_string(),
                    );
                    return true;
                };
                let Some(cache_path) = self.procedural_cache_path(entity) else {
                    self.console.warn(
                        "Convert: save the scene first — the bake cache is keyed off the scene path.".to_string(),
                    );
                    return true;
                };
                if !cache_path.exists() {
                    self.console.warn(format!(
                        "Convert: bake cache '{}' missing — re-bake first.",
                        cache_path.display(),
                    ));
                    return true;
                }

                let name = self
                    .world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{entity:?}"));

                // Sanitize the entity name into a filename-safe slug:
                // lowercase, [a-z0-9_-] only, collapse runs of '_'.
                let mut slug: String = name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect();
                // Trim leading/trailing underscores; collapse runs.
                while slug.contains("__") {
                    slug = slug.replace("__", "_");
                }
                let slug = slug.trim_matches('_').to_string();
                let slug = if slug.is_empty() { "converted".to_string() } else { slug };

                // Drop converted assets under `assets/converted/` so
                // they're discoverable from the Models panel (which
                // recursively scans `assets/`) but visually grouped
                // separately from imported meshes and authored .rkp
                // files. The directory is created lazily.
                let target_dir = project_dir.join("assets").join("converted");
                if let Err(e) = std::fs::create_dir_all(&target_dir) {
                    self.console.error(format!(
                        "Convert: failed to create '{}': {e}",
                        target_dir.display(),
                    ));
                    return true;
                }
                let mut target = target_dir.join(format!("{slug}.rkp"));
                let mut suffix = 1u32;
                while target.exists() {
                    target = target_dir.join(format!("{slug}_{suffix}.rkp"));
                    suffix += 1;
                }

                if let Err(e) = std::fs::copy(&cache_path, &target) {
                    self.console.error(format!(
                        "Convert: failed to write asset '{}': {e}",
                        target.display(),
                    ));
                    return true;
                }

                // Acquire the new file as a regular asset. This
                // gives us a fresh OctreeHandle living in the asset
                // cache; the procedural's previous scene-pool
                // allocation still exists and is now orphaned (a
                // small bounded leak — bake_worker would have
                // re-used it on the next bake, but there isn't
                // going to be a next bake). We accept the leak
                // rather than risk freeing a slot the renderer or
                // a stale snapshot is mid-read of.
                let acquired = self
                    .scene_mgr
                    .lock()
                    .unwrap()
                    .acquire_asset(&target.to_string_lossy());
                let (handle, info) = match acquired {
                    Ok(t) => t,
                    Err(e) => {
                        self.console.error(format!(
                            "Convert: failed to load new asset '{}': {e}",
                            target.display(),
                        ));
                        return true;
                    }
                };
                let new_spatial = spatial_from_handle(
                    &info.spatial,
                    info.voxel_size,
                    &info.aabb,
                    info.grid_origin,
                    info.leaf_attr_slot_start,
                    info.leaf_attr_slot_count,
                    Vec::new(),
                );
                // Path stored in the scene file is relative to the
                // project's assets/ directory — same convention as
                // imported meshes. e.g. "converted/sphere_1.rkp".
                let rel_path = target
                    .strip_prefix(project_dir.join("assets"))
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| target.to_string_lossy().to_string());

                if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
                    renderable.primitive = None;
                    renderable.spatial = Some(new_spatial);
                    renderable.asset_handle = Some(handle);
                    renderable.asset_path = Some(rel_path.clone());
                    renderable.voxel_count = info.voxel_count;
                }
                let _ = self.world.remove_one::<ProceduralGeometry>(entity);
                if self.selected_entity == Some(entity) {
                    self.selected_procedural_node = None;
                }
                self.scene_dirty = true;
                self.geometry_dirty = true;
                self.gpu_objects_dirty = true;
                // Surface the new asset in the Models panel right
                // away so it can be re-spawned later.
                self.scan_models();
                self.console.info(format!(
                    "Converted '{name}' to voxel asset → assets/{rel_path} ({} voxels).",
                    info.voxel_count,
                ));
            }

            EngineCommand::CopyProceduralToNewVoxel { entity_id } => {
                use crate::components::*;
                let Some(src_entity) = self
                    .entity_uuids
                    .iter()
                    .find_map(|(e, u)| (*u == entity_id).then_some(*e))
                else {
                    self.console.warn("Copy: entity not found".to_string());
                    return true;
                };
                // Same gate: won't copy a snapshot the user didn't
                // ask for.
                let can_copy = self
                    .world
                    .get::<&ProceduralGeometry>(src_entity)
                    .map(|pg| !pg.bake_in_flight && !pg.pending_bake && !pg.dirty)
                    .unwrap_or(false);
                if !can_copy {
                    self.console.warn(
                        "Copy: bake pending or in flight — let it settle first".to_string(),
                    );
                    return true;
                }
                // Read what we need from the source entity. The
                // baked voxels live in shared scene pools, so we
                // re-voxelize the tree for the new entity rather
                // than refcounting the source's allocation — two
                // entities refcounting the same octree isn't what
                // we've got today (asset_cache is path-keyed, not
                // generalized). A second bake of the same tree
                // reuses the GPU evaluator's warmed pipelines and is
                // fast; we also go through the async path so the
                // engine tick stays smooth.
                let (src_name, src_transform, src_scale_for_bake, src_tree, src_voxel_size) = {
                    let name = self
                        .world
                        .get::<&EditorMetadata>(src_entity)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|_| "Procedural".to_string());
                    let transform = self
                        .world
                        .get::<&Transform>(src_entity)
                        .map(|t| (*t).clone())
                        .unwrap_or_else(|_| Transform::default());
                    let proc_geo = match self.world.get::<&ProceduralGeometry>(src_entity) {
                        Ok(pg) => pg,
                        Err(_) => {
                            self.console.warn("Copy: source has no ProceduralGeometry".to_string());
                            return true;
                        }
                    };
                    let root_scale = proc_geo
                        .tree
                        .get(proc_geo.tree.root())
                        .map(|n| n.transform.to_scale_rotation_translation().0)
                        .unwrap_or(glam::Vec3::ONE);
                    (
                        name,
                        transform,
                        root_scale,
                        proc_geo.tree.clone(),
                        proc_geo.voxel_size,
                    )
                };

                // Spawn the destination entity. No ProceduralGeometry —
                // this is the static voxel copy. Starts with
                // spatial=None; the bake we enqueue below fills it.
                let new_name = self.unique_name(&format!("{src_name} (copy)"));
                let new_entity = self.world.spawn((
                    src_transform,
                    EditorMetadata { name: new_name.clone() },
                    Renderable {
                        primitive: None,
                        voxel_count: 0,
                        spatial: None,
                        ..Default::default()
                    },
                ));
                self.assign_entity_uuid(new_entity);
                self.scene_dirty = true;

                // Enqueue a bake for the copy. We're reusing the
                // async bake pipeline but the target entity has no
                // ProceduralGeometry — so `enqueue_bake` won't
                // accept it. Build the request by hand.
                let (aabb, voxel_size) = procedural_voxel_params(&src_tree, src_voxel_size);
                let instructions = rkp_procedural::flatten_tree(&src_tree);
                // `generation: 0` so the staleness check in
                // `drain_bake_results` (which reads the target
                // entity's current generation and defaults to 0 when
                // the entity has no ProceduralGeometry) matches and
                // we actually apply the result. Copy targets never
                // re-bake, so a single-shot value is fine.
                let req = crate::bake_worker::BakeRequest {
                    entity: new_entity,
                    generation: 0,
                    input: crate::bake_worker::BakeInput::Procedural(instructions),
                    aabb,
                    voxel_size,
                    root_scale: src_scale_for_bake,
                    prev_spatial: None,
                    // Copy targets have no ProceduralGeometry, so no
                    // scene reload would look for a sidecar here.
                    cache_output_path: None,
                    generator_child: None,
                };
                if self.bake_worker.tx_request.send(req).is_err() {
                    self.console.warn("Copy: bake worker channel closed".to_string());
                    return true;
                }
                self.console.info(format!(
                    "Copied '{src_name}' → '{new_name}' (baking…)",
                ));
            }

            EngineCommand::SetBuildPreviewMode { mode } => {
                if let Some(vp) = self.viewports.get_mut(crate::viewport::ViewportId::BUILD) {
                    vp.preview_mode = mode;
                    eprintln!("[preview] build viewport preview_mode -> {mode:?}");
                } else {
                    eprintln!("[preview] SetBuildPreviewMode but no BUILD viewport registered");
                }
            }

            EngineCommand::SpawnPointLight => {
                use crate::components::*;
                let name = self.unique_name("Point Light");
                let mut transform = Transform::default();
                transform.position = self.camera.position + glam::Vec3::new(0.0, 2.0, 0.0);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    PointLight::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnSpotLight => {
                use crate::components::*;
                let name = self.unique_name("Spot Light");
                let mut transform = Transform::default();
                transform.position = self.camera.position + glam::Vec3::new(0.0, 3.0, 0.0);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    SpotLight::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnCamera => {
                use crate::components::*;
                let name = self.unique_name("Camera");
                let mut transform = Transform::default();
                transform.position = self.camera.position;
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    Camera::default(),
                ));
                self.assign_entity_uuid(entity);
                self.scene_dirty = true;
                self.console.info(format!("Spawned '{name}'"));
            }

            EngineCommand::SpawnGenerator { generator_name } => {
                self.spawn_generator(&generator_name, None);
            }

            EngineCommand::SpawnGeneratorPreset { path } => {
                self.spawn_generator_preset(&path, None);
            }

            EngineCommand::DropGenerator { id, generator_name, x, y } => {
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::Generator { name: generator_name },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
            }

            EngineCommand::DropGeneratorPreset { id, path, x, y } => {
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::GeneratorPreset { path },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y, ghost_pick_node_id: None,
                });
            }

            EngineCommand::LoadAsset { path, position } => {
                self.spawn_asset(&path, position);
            }

            EngineCommand::DropAsset { id, path, x, y } => {
                // Drag-drop placement: issue a position-readback pick at
                // the drop pixel, queue a pending drop, and spawn when
                // the pick result arrives (process_pick_result handles
                // it — see `PendingDrop`).
                self.pending_drop = Some(PendingDrop {
                    viewport: id, x, y,
                    action: PendingDropAction::Asset { path },
                });
                self.pending_pick = Some(PendingPick {
                    viewport: id, x, y,
                    ghost_pick_node_id: None,
                });
            }

            EngineCommand::DragPreviewEnter { id, source, x, y } => {
                // Clean up any orphaned preview from a previous drag
                // (two DragEnters with no Cancel / Commit between).
                if let Some(prev) = self.drag_preview.take() {
                    if let DragPreviewKind::Model { entity, .. } = prev.kind {
                        self.delete_entity(entity);
                    }
                }
                // Initial position: ground-plane raycast at the cursor
                // so the preview doesn't flash at the origin before the
                // first pick readback lands. Falls back to 3m in front
                // of the camera for rays that miss the plane.
                let provisional = {
                    let (ro, rd) = self.screen_to_ray_for_viewport(id, x as f32, y as f32);
                    if rd.y.abs() > 1e-6 {
                        let t = -ro.y / rd.y;
                        if t > 0.0 { ro + rd * t }
                        else { self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0) }
                    } else {
                        self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
                    }
                };
                let kind = match source {
                    crate::command::DragPreviewSource::Asset { path } => {
                        // Models: spawn the real asset now. The first
                        // pick readback snaps it to the cursor.
                        match self.spawn_asset_ex(&path, provisional, false) {
                            Some((entity, aabb_min_y)) => {
                                Some(DragPreviewKind::Model { entity, aabb_min_y })
                            }
                            None => None,
                        }
                    }
                    src @ (crate::command::DragPreviewSource::Generator { .. }
                        | crate::command::DragPreviewSource::GeneratorPreset { .. }) => {
                        // Generators: no spawn yet — the real entity
                        // only materialises on commit. Meanwhile draw a
                        // 1 m half-extent wireframe box at the cursor.
                        // We don't know the baked bounds until after a
                        // run, so a single conservative default beats
                        // introspecting parameters per-generator.
                        Some(DragPreviewKind::Generator {
                            source: src,
                            gizmo_half: glam::Vec3::splat(0.5),
                        })
                    }
                };
                if let Some(kind) = kind {
                    self.drag_preview = Some(DragPreviewState {
                        viewport: id,
                        kind,
                        last_surface_pos: Some(provisional),
                        last_cursor: (x, y),
                    });
                    self.pending_pick = Some(PendingPick {
                        viewport: id, x, y, ghost_pick_node_id: None,
                    });
                }
            }

            EngineCommand::DragPreviewOver { id, x, y } => {
                if let Some(preview) = self.drag_preview.as_mut() {
                    if preview.viewport == id {
                        preview.last_cursor = (x, y);
                        // Overwrite any in-flight request with the
                        // freshest pixel. Render-side `pick_in_flight`
                        // gate throttles to one readback per frame, so
                        // newer coords win naturally.
                        self.pending_pick = Some(PendingPick {
                            viewport: id, x, y, ghost_pick_node_id: None,
                        });
                    }
                }
            }

            EngineCommand::DragPreviewCommit => {
                if let Some(preview) = self.drag_preview.take() {
                    match preview.kind {
                        // Models: entity is already live at the final
                        // position. Just retire the preview state —
                        // subsequent pick results won't touch it.
                        DragPreviewKind::Model { .. } => {}
                        // Generators: now spawn the real source at the
                        // last-known surface position. Falls back to a
                        // ground-plane cast at the final cursor pixel
                        // if no valid surface hit ever landed.
                        DragPreviewKind::Generator { source, .. } => {
                            let pos = preview.last_surface_pos.unwrap_or_else(|| {
                                let (cx, cy) = preview.last_cursor;
                                let (ro, rd) = self.screen_to_ray_for_viewport(
                                    preview.viewport, cx as f32, cy as f32,
                                );
                                if rd.y.abs() > 1e-6 {
                                    let t = -ro.y / rd.y;
                                    if t > 0.0 { ro + rd * t }
                                    else { self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0) }
                                } else {
                                    self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
                                }
                            });
                            match source {
                                crate::command::DragPreviewSource::Generator { name } => {
                                    self.spawn_generator(&name, Some(pos));
                                }
                                crate::command::DragPreviewSource::GeneratorPreset { path } => {
                                    self.spawn_generator_preset(&path, Some(pos));
                                }
                                crate::command::DragPreviewSource::Asset { .. } => {
                                    // Unreachable — Asset paths produce
                                    // `DragPreviewKind::Model`, handled
                                    // above.
                                }
                            }
                        }
                    }
                }
            }

            EngineCommand::DragPreviewCancel => {
                if let Some(preview) = self.drag_preview.take() {
                    // Only the model path has a live entity to delete;
                    // generators never spawned anything during drag.
                    if let DragPreviewKind::Model { entity, .. } = preview.kind {
                        self.delete_entity(entity);
                    }
                }
            }

            EngineCommand::Pick { id, x, y } => {
                // BUILD + Voxel: picking doesn't make sense. The G-buffer
                // slot the raymarch uses for NodeId is occupied by
                // secondary_material_id in voxel mode, so decoding
                // would return arbitrary node ids. Skip entirely —
                // the user selects tree nodes via the build panel
                // in voxel mode.
                if id == crate::viewport::ViewportId::BUILD {
                    let is_raymarch = self
                        .viewports
                        .get(crate::viewport::ViewportId::BUILD)
                        .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                        .unwrap_or(false);
                    if !is_raymarch {
                        return true;
                    }
                }

                // Route the pick by viewport — MAIN picks scene entities
                // (old path), BUILD picks procedural primitives when in
                // raymarch preview. Either way, a click landing on a
                // gizmo axis should NOT fall through to pick — that
                // deselects the currently-manipulated object and
                // prevents the drag from starting. Each viewport has
                // its own gizmo state; pick the right one.
                let gizmo_blocking = match id {
                    crate::viewport::ViewportId::MAIN => {
                        self.gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
                            || self.gizmo.dragging
                    }
                    crate::viewport::ViewportId::BUILD => {
                        self.proc_gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
                            || self.proc_gizmo.dragging
                    }
                    _ => false,
                };
                if !gizmo_blocking {
                    // Ghost-priority pick: on BUILD in raymarch mode,
                    // CPU-raycast the tree's ghost-role primitives at
                    // the click ray. If any hits, remember which one —
                    // it takes priority over the G-buffer decode
                    // (matches the visual rule that a ghost painted
                    // on the pixel owns the click).
                    let ghost_pick_node_id = self
                        .compute_ghost_pick(id, x, y);
                    self.pending_pick = Some(PendingPick {
                        viewport: id, x, y, ghost_pick_node_id,
                    });
                }
            }

            EngineCommand::ImportAsset { source_path } => {
                let source = std::path::PathBuf::from(&source_path);
                let output = crate::import_worker::rkp_output_path(&source);
                self.import_worker.submit(crate::import_worker::ImportRequest {
                    source_path: source,
                    output_path: output,
                    config: crate::import_worker::default_import_config(),
                });
            }

            EngineCommand::SetObjectPosition { entity_id, position } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.position = position;
                    }
                }
            }

            EngineCommand::SetObjectRotation { entity_id, rotation } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.rotation = rotation;
                    }
                }
            }

            EngineCommand::SetObjectScale { entity_id, scale } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(entity) {
                        t.scale = scale;
                    }
                }
            }

            EngineCommand::SelectEntity { entity_id } => {
                self.selected_entity = self.resolve_entity(&entity_id);
            }

            EngineCommand::DeleteObject { entity_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    self.delete_entity(entity);
                }
            }

            EngineCommand::ReorderEntity { entity, new_parent, new_order } => {
                self.handle_reorder(entity, new_parent, new_order);
            }

            EngineCommand::DeleteSelected => {
                if let Some(entity) = self.selected_entity {
                    self.delete_entity(entity);
                }
            }

            EngineCommand::DuplicateObject { entity_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    self.duplicate_entity(entity);
                }
            }

            EngineCommand::DuplicateSelected => {
                if let Some(entity) = self.selected_entity {
                    self.duplicate_entity(entity);
                }
            }

            EngineCommand::NewProject { path } => {
                let path = std::path::PathBuf::from(&path);
                match crate::project::create_project(&path) {
                    Ok(project_dir) => {
                        self.clear_scene();
                        let project_name = project_dir.file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let project_file = project_dir.join(format!("{project_name}.rkproject"));
                        self.project_dir = Some(project_dir.clone());
                        self.project_path = Some(project_file);
                        self.scene_path = Some(project_dir.join("scenes/default.rkscene"));
                        self.project_name = project_name;
                        self.project_loaded = true;
                        self.project_dirty = true;
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                        self.scan_models();
                        if let Some(ref dir) = self.project_dir {
                            // Write starter materials before scanning.
                            crate::material_library::write_starter_materials(
                                &dir.join("assets/materials"),
                            );
                            self.material_lib.scan(&dir.join("assets/materials"));
                        }
                        self.init_file_watcher();
                        self.scaffold_and_build_gameplay();
                        self.auto_import_meshes();
                        if let Some(ref pp) = self.project_path {
                            crate::recent_projects::add_recent(&self.project_name, &pp.to_string_lossy());
                        }
                    }
                    Err(e) => eprintln!("[RkpEngine] new project failed: {e}"),
                }
            }

            EngineCommand::OpenProject { path } => {
                let path = std::path::PathBuf::from(&path);
                match crate::project::load_project(&path) {
                    Ok((project, project_dir)) => {
                        self.clear_scene();
                        self.project_dir = Some(project_dir.clone());
                        self.project_path = Some(path);
                        self.project_name = project.name;
                        self.project_loaded = true;
                        self.project_dirty = true;
                        // Cache + flag the editor layout so the editor
                        // hydrates its docking state on the next tick.
                        // `None` is meaningful — it means "reset to
                        // default" for projects saved pre-persistence.
                        self.editor_layout_json = project.editor_layout;
                        self.editor_layout_pending = true;

                        // Scaffold + build gameplay BEFORE loading the scene,
                        // so gameplay components (Spin, Health, etc.) are registered
                        // and can be deserialized from the scene file.
                        self.scaffold_and_build_gameplay();

                        // `scene_path` must be set BEFORE loading so
                        // `load_scene_from_file` can resolve
                        // procedural bake-cache sidecars relative to
                        // the scene file's directory.
                        let scene_path = project_dir.join(format!("scenes/{}.rkscene", project.default_scene));
                        self.scene_path = Some(scene_path.clone());
                        if scene_path.exists() {
                            self.load_scene_from_file(&scene_path);
                        }

                        self.scan_models();
                        if let Some(ref dir) = self.project_dir {
                            self.material_lib.scan(&dir.join("assets/materials"));
                        }
                        self.init_file_watcher();
                        self.auto_import_meshes();
                        if let Some(ref pp) = self.project_path {
                            crate::recent_projects::add_recent(&self.project_name, &pp.to_string_lossy());
                        }
                    }
                    Err(e) => eprintln!("[RkpEngine] open project failed: {e}"),
                }
            }

            EngineCommand::SaveScene { path } => {
                let save_path = path.map(std::path::PathBuf::from)
                    .or_else(|| self.scene_path.clone());
                if let Some(save_path) = save_path {
                    let scene = self.build_scene_file();
                    if let Err(e) = crate::scene_io::save_scene(&scene, &save_path) {
                        eprintln!("[RkpEngine] save scene failed: {e}");
                    }
                    self.scene_path = Some(save_path);
                }
                // Persist the project descriptor alongside the scene so
                // the cached editor layout (and anything else on
                // ProjectFile) actually hits disk on Ctrl+S. Without
                // this, layout state would only be written by explicit
                // SaveProject, which the UI doesn't wire up.
                self.save_project_file();
            }

            EngineCommand::SaveProject => {
                self.save_project_file();
            }

            EngineCommand::SetEditorLayout { json } => {
                // Cache only — actual write happens on save. Don't echo
                // back to the editor; it's the source of truth for this.
                self.editor_layout_json = Some(json);
            }

            // ── Raw input → feed to InputSystem ──────────────────────
            // Phase 3: only MAIN viewport input drives the camera-controller
            // input system. Build viewport / PiP wiring lands in Phase 6.
            EngineCommand::MouseMove { id, x, y, dx, dy } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.mouse_pos = glam::Vec2::new(x, y);
                    self.input_system.feed_mouse_delta(glam::Vec2::new(dx, dy));
                } else if id == crate::viewport::ViewportId::BUILD {
                    self.build_mouse_pos = glam::Vec2::new(x, y);
                    let _ = (dx, dy);
                }
            }
            EngineCommand::MouseButton { id, button, pressed } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.input_system.feed_mouse_button(button, pressed);
                } else if id == crate::viewport::ViewportId::BUILD {
                    if button == rkp_runtime::input::InputMouseButton::Left {
                        self.build_mouse_left = pressed;
                    }
                }
            }
            EngineCommand::Scroll { id, delta } => {
                if id == crate::viewport::ViewportId::MAIN {
                    self.input_system.feed_scroll(delta);
                }
            }
            EngineCommand::KeyDown { key } => {
                self.input_system.feed_key_down(key);
            }
            EngineCommand::KeyUp { key } => {
                self.input_system.feed_key_up(key);
            }

            EngineCommand::SetComponentField { entity_id, component_name, field_name, value } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Some(entry) = self.registry.get(&component_name) {
                        if let Ok(fv) = serde_json::from_str::<crate::inspector::FieldValue>(&value) {
                            if let Err(e) = (entry.set_field)(&mut self.world, entity, &field_name, fv) {
                                eprintln!("[RkpEngine] set_field failed: {e}");
                            } else {
                                if component_name == "Transform" {
                                    // Procedural entities treat Transform.scale as
                                    // an alias for the Root node's scale: bake the
                                    // value into the tree, reset the entity scale,
                                    // and queue an auto-bake. Keeps procedural
                                    // entities at world scale 1 so colliders /
                                    // gizmos / physics aren't double-scaled, and
                                    // makes the bake actually produce voxels at
                                    // the right density (the entity-level scale
                                    // path was a no-op visually — same voxels,
                                    // just stretched at render time).
                                    if field_name == "scale" {
                                        self.redirect_transform_scale_to_root(entity);
                                    }
                                    self.gpu_objects_dirty = true;
                                }
                                if component_name == "RigidBody" {
                                    self.collider_caches_dirty = true;
                                }
                            }
                        }
                    }
                }
            }

            EngineCommand::AddComponent { entity_id, component_name } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    // Skeleton needs more context than the registry's
                    // plain (World, Entity) `add_default` — it has to
                    // find the sibling `.rkskel` next to the entity's
                    // Renderable asset and load it. Route here first;
                    // the attach helper also inserts an
                    // AnimationPlayer alongside (components are
                    // bundled — you never want one without the other).
                    if component_name == "Skeleton" {
                        let rkp_path = self.world
                            .get::<&crate::components::Renderable>(entity)
                            .ok()
                            .and_then(|r| r.asset_path.clone());
                        match rkp_path {
                            Some(p) => self.try_attach_skeleton(entity, std::path::Path::new(&p)),
                            None => self.console.warn(
                                "Add Skeleton: entity has no Renderable asset — attach a model first".to_string(),
                            ),
                        }
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                    } else if let Some(entry) = self.registry.get(&component_name) {
                        if let Err(e) = (entry.add_default)(&mut self.world, entity) {
                            eprintln!("[RkpEngine] add component failed: {e}");
                        }
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                        if component_name == "RigidBody" {
                            self.collider_caches_dirty = true;
                        }
                    }
                }
            }

            EngineCommand::RemoveComponent { entity_id, component_name } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Some(entry) = self.registry.get(&component_name) {
                        if let Err(e) = (entry.remove)(&mut self.world, entity) {
                            eprintln!("[RkpEngine] remove component failed: {e}");
                        }
                        // Skeleton + AnimationPlayer are bundled —
                        // pulling the skeleton also pulls the player
                        // (ui treats AnimationPlayer as part of the
                        // Skeleton section, so an orphaned player
                        // would be invisible and confusing).
                        if component_name == "Skeleton" {
                            let _ = self.world.remove_one::<crate::components::AnimationPlayer>(entity);
                        }
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                        if component_name == "RigidBody" {
                            self.collider_caches_dirty = true;
                        }
                    }
                }
            }

            EngineCommand::CreateMaterial { name } => {
                match self.material_lib.create(&name) {
                    Ok(id) => {
                        eprintln!("[RkpEngine] created material '{name}' as id {id}");
                        self.selected_material = Some(id);
                    }
                    Err(e) => eprintln!("[RkpEngine] create material failed: {e}"),
                }
            }

            EngineCommand::UpdateMaterialField { material_id, field, value } => {
                if let Some(def) = self.material_lib.get_def_mut(material_id) {
                    match field.as_str() {
                        "name" => { def.name = value; }
                        "base_color" => {
                            if let Ok(v) = serde_json::from_str::<[f32; 4]>(&value) {
                                def.base_color = v;
                            }
                        }
                        "roughness" => {
                            if let Ok(v) = value.parse::<f32>() { def.roughness = v; }
                        }
                        "metallic" => {
                            if let Ok(v) = value.parse::<f32>() { def.metallic = v; }
                        }
                        "emission_strength" => {
                            if let Ok(v) = value.parse::<f32>() { def.emission_strength = v; }
                        }
                        "opacity" => {
                            if let Ok(v) = value.parse::<f32>() { def.opacity = v; }
                        }
                        _ => { eprintln!("[RkpEngine] unknown material field: {field}"); }
                    }
                    let _ = self.material_lib.save(material_id);
                }
            }

            EngineCommand::DeleteMaterial { material_id } => {
                if let Some(path) = self.material_lib.path_for_id(material_id).map(|p| p.to_owned()) {
                    let _ = std::fs::remove_file(&path);
                    self.material_lib.remove(&path);
                    if self.selected_material == Some(material_id) {
                        self.selected_material = None;
                    }
                }
            }

            EngineCommand::AssignMaterial { entity_id, material_id } => {
                if let Some(entity) = self.resolve_entity(&entity_id) {
                    if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                        r.material_id = material_id;
                        self.gpu_objects_dirty = true;
                    }
                }
            }

            EngineCommand::SelectMaterial { material_id } => {
                // The Asset Properties panel inspects one thing at a time —
                // picking a material drops any prior model selection so the
                // panel swaps over instead of staying stuck on the model (or
                // vice versa).
                self.selected_material = material_id;
                if material_id.is_some() {
                    self.selected_model = None;
                }
            }

            EngineCommand::RemapMaterial { object_id, from_material, to_material } => {
                if let Some(entity) = self.resolve_entity(&object_id) {
                    let count = self.remap_entity_material(entity, from_material, to_material);
                    if count > 0 {
                        eprintln!("[RkpEngine] remapped {count} voxels from material {from_material} to {to_material}");
                        self.geometry_dirty = true;
                    }
                }
            }

            EngineCommand::SetPrimitiveMaterial { object_id, material_id } => {
                if let Some(entity) = self.resolve_entity(&object_id) {
                    if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                        r.material_id = material_id;
                        self.gpu_objects_dirty = true;
                    }
                }
            }

            EngineCommand::SelectModel { path } => {
                self.selected_model = path;
                if self.selected_model.is_some() {
                    self.selected_material = None;
                }
            }

            EngineCommand::UpdateImportField { source_path, field, value } => {
                // Find the model info, update its import profile, save sidecar.
                let source = std::path::PathBuf::from(&source_path);
                let mut profile = crate::import_profile::ImportProfile::load_or_default(&source);
                match field.as_str() {
                    "display_name" => {
                        profile.display_name = if value.is_empty() { None } else { Some(value) };
                    }
                    "voxel_size" => {
                        profile.voxel_size = value.parse::<f32>().ok().filter(|&v| v > 0.0);
                    }
                    "target_size" => {
                        if let Ok(v) = value.parse::<f32>() { profile.target_size = v; }
                    }
                    "no_normalize" => {
                        profile.no_normalize = value == "true";
                    }
                    "import_colors" => {
                        profile.import_colors = value == "true";
                    }
                    "rotation_x" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[0] = v; }
                    }
                    "rotation_y" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[1] = v; }
                    }
                    "rotation_z" => {
                        if let Ok(v) = value.parse::<f32>() { profile.rotation_offset[2] = v; }
                    }
                    _ => {
                        eprintln!("[RkpEngine] unknown import field: {field}");
                    }
                }
                if let Err(e) = profile.save_for(&source) {
                    eprintln!("[RkpEngine] save import profile failed: {e}");
                }
                // Update the in-memory model info.
                if let Some(mi) = self.available_models.iter_mut().find(|m| m.source_path == source_path) {
                    if let Some(ref name) = profile.display_name {
                        mi.name = name.clone();
                    }
                    mi.import_profile = Some(profile);
                }
                self.models_dirty = true;
            }

            EngineCommand::ReimportModel { source_path } => {
                let source = std::path::PathBuf::from(&source_path);
                let source_key = source.to_string_lossy().into_owned();
                // Drop the request if this source already has an import
                // in flight. Without the guard a double-click would queue
                // two identical jobs, and the spinner would clear halfway
                // through while the second still ran in the background.
                if self.importing_sources.contains(&source_key) {
                    eprintln!(
                        "[RkpEngine] re-import already in flight for {} — ignoring",
                        source.display(),
                    );
                    return true;
                }
                let profile = crate::import_profile::ImportProfile::load_or_default(&source);
                let config = profile.to_import_config();
                let output = crate::import_worker::rkp_output_path(&source);
                eprintln!(
                    "[RkpEngine] re-importing {} → {} \
                     (target_size={}, voxel_size={:?}, rotation={:?}, import_colors={})",
                    source.display(), output.display(),
                    config.target_size, config.voxel_size,
                    config.rotation_offset, config.import_colors,
                );
                let name = source.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.console.info(format!("Re-importing '{name}'…"));
                self.importing_sources.insert(source_key);
                self.importing_dirty = true;
                self.import_worker.submit(crate::import_worker::ImportRequest {
                    source_path: source,
                    output_path: output,
                    config,
                });
            }

            EngineCommand::SetViewOption { option, enabled } => {
                match option.as_str() {
                    "show_colliders" => self.show_colliders = enabled,
                    "skinning" => self.skinning_enabled = enabled,
                    "dqs" => self.dqs_enabled = enabled,
                    _ => eprintln!("[RkpEngine] unknown view option: {option}"),
                }
            }

            EngineCommand::ClearConsole => {
                self.console.clear();
            }

            EngineCommand::UpdateEnvironment { field, value } => {
                let env = &mut self.environment;
                match field.as_str() {
                    "sky_color_top_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_top_override = Some(v); }
                    }
                    "sky_color_top_override_enabled" => {
                        if value == "false" { env.sky_color_top_override = None; }
                    }
                    "sky_color_horizon_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_horizon_override = Some(v); }
                    }
                    "sky_color_horizon_override_enabled" => {
                        if value == "false" { env.sky_color_horizon_override = None; }
                    }
                    "sun_color_override" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sun_color_override = Some(v); }
                    }
                    "sun_color_override_enabled" => {
                        if value == "false" { env.sun_color_override = None; }
                    }
                    "ambient_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.ambient_intensity = v; }
                    }
                    "sun_azimuth" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_azimuth = v; }
                    }
                    "sun_elevation" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_elevation = v; }
                    }
                    "sun_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.sun_intensity = v; }
                    }
                    "shadow_steps" => {
                        if let Ok(v) = value.parse::<u32>() { env.shadow_steps = v; }
                    }
                    "ao_radius" => {
                        if let Ok(v) = value.parse::<f32>() { env.ao_radius = v; }
                    }
                    "ao_steps" => {
                        if let Ok(v) = value.parse::<u32>() { env.ao_steps = v; }
                    }
                    "exposure" => {
                        if let Ok(v) = value.parse::<f32>() { env.exposure = v; }
                    }
                    "bloom_threshold" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_threshold = v; }
                    }
                    "bloom_knee" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_knee = v; }
                    }
                    "bloom_intensity" => {
                        if let Ok(v) = value.parse::<f32>() { env.bloom_intensity = v; }
                    }
                    "god_ray_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_density = v; }
                    }
                    "god_ray_weight" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_weight = v; }
                    }
                    "god_ray_decay" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_decay = v; }
                    }
                    "god_ray_exposure" => {
                        if let Ok(v) = value.parse::<f32>() { env.god_ray_exposure = v; }
                    }
                    "scene_elevation" => {
                        if let Ok(v) = value.parse::<f32>() { env.scene_elevation = v; }
                    }
                    "ground_albedo" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.ground_albedo = v; }
                    }
                    // Fog
                    "fog_color" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.fog_color = v; }
                    }
                    "height_fog_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.height_fog_density = v; }
                    }
                    "fog_base_height" => {
                        if let Ok(v) = value.parse::<f32>() { env.fog_base_height = v; }
                    }
                    "fog_height_falloff" => {
                        if let Ok(v) = value.parse::<f32>() { env.fog_height_falloff = v; }
                    }
                    "vol_far" => {
                        if let Ok(v) = value.parse::<f32>() { env.vol_far = v; }
                    }
                    // Clouds
                    "clouds_enabled" => {
                        env.clouds_enabled = value == "true" || value == "1";
                    }
                    "attenuate_sun_by_clouds" => {
                        env.attenuate_sun_by_clouds = value == "true" || value == "1";
                    }
                    "cloud_slab_steps" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_slab_steps = (v as u32).clamp(8, 128);
                        }
                    }
                    "cloud_shadow_steps" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_shadow_steps = (v as u32).clamp(1, 8);
                        }
                    }
                    "cloud_detail_octaves" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_detail_octaves = (v as u32).clamp(1, 6);
                        }
                    }
                    "cloud_ms_octaves" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_ms_octaves = (v as u32).clamp(1, 5);
                        }
                    }
                    "cloud_taa_alpha" => {
                        if let Ok(v) = value.parse::<f32>() {
                            env.cloud_taa_alpha = v.clamp(0.05, 0.7);
                        }
                    }
                    "cloud_altitude_min" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_altitude_min = v; }
                    }
                    "cloud_altitude_max" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_altitude_max = v; }
                    }
                    "cloud_coverage" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_coverage = v; }
                    }
                    "cloud_density_scale" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_density_scale = v; }
                    }
                    "cloud_wind_speed" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_wind_speed = v; }
                    }
                    "cloud_wind_dir" => {
                        if let Ok(v) = value.parse::<f32>() { env.cloud_wind_dir = v; }
                    }
                    _ => { eprintln!("[RkpEngine] unknown environment field: {field}"); }
                }
                self.environment_dirty = true;
                // Deliberately do NOT set environment_ui_dirty: the UI already holds
                // the authoritative value (it just sent it). Echoing back would cause
                // the form to remount mid-drag on every slider tick.
            }

            EngineCommand::SetGizmoMode { mode } => {
                self.gizmo.mode = mode;
            }

            EngineCommand::PlayStart => {
                if self.play_state.is_none() {
                    // Ensure collider caches are up to date before entering play mode.
                    if self.collider_caches_dirty {
                        self.rebuild_collider_caches();
                        self.collider_caches_dirty = false;
                    }
                    let play = crate::play_mode::PlayModeState::start(&mut self.world);
                    self.play_state = Some(play);
                    // Build behavior executor from gameplay systems.
                    match crate::behavior::BehaviorExecutor::new(&self.gameplay_systems) {
                        Ok(executor) => {
                            self.behavior_executor = Some(executor);
                            self.console.info(format!(
                                "Play mode started ({} systems)",
                                self.gameplay_systems.len(),
                            ));
                        }
                        Err(e) => {
                            self.behavior_executor = None;
                            self.console.error(format!("Failed to build system schedule: {e}"));
                            self.console.info("Play mode started (no systems)");
                        }
                    }
                    self.play_total_time = 0.0;
                    self.play_frame_count = 0;
                    // Reset FixedUpdate accumulator so play mode
                    // starts from a clean zero rather than firing a
                    // burst of catch-up steps.
                    self.behavior_fixed_accumulator = 0.0;
                    self.enter_play_mode_viewports();
                }
            }

            EngineCommand::PlayStop => {
                if let Some(play) = self.play_state.take() {
                    play.stop(&mut self.world);
                    self.behavior_executor = None;
                    self.gpu_objects_dirty = true;
                    self.console.info("Play mode stopped — transforms restored");
                    self.exit_play_mode_viewports();
                }
            }

            _ => {
                eprintln!("[RkpEngine] unhandled command: {cmd:?}");
            }
        }

        true
    }

    /// Scan for importable mesh files and auto-import any that don't have .rkp outputs.
    fn auto_import_meshes(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if !assets_dir.exists() { return; }

            // Scan recursively for mesh files.
            let mut meshes = Vec::new();
            Self::scan_meshes_recursive(&assets_dir, &mut meshes);

            for source in meshes {
                let output = crate::import_worker::rkp_output_path(&source);
                // Only import if .rkp doesn't exist or is older than source.
                let needs_import = if output.exists() {
                    let src_mod = std::fs::metadata(&source)
                        .and_then(|m| m.modified()).ok();
                    let out_mod = std::fs::metadata(&output)
                        .and_then(|m| m.modified()).ok();
                    match (src_mod, out_mod) {
                        (Some(s), Some(o)) => s > o,
                        _ => true,
                    }
                } else {
                    true
                };

                if needs_import {
                    eprintln!("[RkpEngine] auto-importing: {}", source.display());
                    self.import_worker.submit(crate::import_worker::ImportRequest {
                        source_path: source,
                        output_path: output,
                        config: crate::import_worker::default_import_config(),
                    });
                }
            }
        }
    }

    fn scan_meshes_recursive(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_meshes_recursive(&path, out);
            } else {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "glb" | "gltf" | "obj" | "fbx") {
                    out.push(path);
                }
            }
        }
    }

    fn init_file_watcher(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if assets_dir.exists() {
                match crate::file_watcher::RkpFileWatcher::new(&[assets_dir.as_path()]) {
                    Ok(watcher) => {
                        self.file_watcher = Some(watcher);
                        eprintln!("[RkpEngine] file watcher started on {}", assets_dir.display());
                    }
                    Err(e) => eprintln!("[RkpEngine] file watcher failed: {e}"),
                }
            }
        }
    }

    /// Scaffold the gameplay crate from project scripts and trigger a build.
    fn scaffold_and_build_gameplay(&mut self) {
        let Some(ref project_dir) = self.project_dir else { return };

        // Create assets/scripts directories if they don't exist (new projects).
        let scripts_dir = project_dir.join("assets/scripts");
        let _ = std::fs::create_dir_all(scripts_dir.join("components"));
        let _ = std::fs::create_dir_all(scripts_dir.join("systems"));

        // Generate the gameplay crate.
        match crate::scaffold::generate_gameplay_crate(project_dir) {
            Ok(crate_dir) => {
                self.console.info("Scaffolded gameplay crate");
                // Build the dylib.
                self.build_gameplay_crate(&crate_dir);
            }
            Err(e) => {
                self.console.error(format!("Scaffold failed: {e}"));
            }
        }
    }

    /// Build the scaffolded gameplay crate and load the resulting dylib.
    fn build_gameplay_crate(&mut self, crate_dir: &std::path::Path) {
        self.console.info("Building gameplay scripts...");
        let output = std::process::Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(crate_dir.join("Cargo.toml"))
            .output();

        match output {
            Ok(out) if out.status.success() => {
                self.console.info("Gameplay scripts compiled");
                // Load the built dylib.
                if let Some(ref project_dir) = self.project_dir {
                    let dylib_path = crate::scaffold::gameplay_dylib_path(project_dir);
                    self.load_gameplay_dylib(&dylib_path);
                }
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                self.console.error(format!("Gameplay build failed:\n{stderr}"));
            }
            Err(e) => {
                self.console.error(format!("Failed to run cargo: {e}"));
            }
        }
    }

    /// Load a gameplay dylib and register its components + systems.
    fn load_gameplay_dylib(&mut self, path: &std::path::Path) {
        if !path.exists() {
            return;
        }
        match self.gameplay_loader.load(path) {
            Ok(entries) => {
                let names: Vec<&str> = entries.iter().map(|e| e.name).collect();
                self.console.info(format!(
                    "Loaded gameplay: {} components ({})",
                    entries.len(),
                    names.join(", "),
                ));
                for &entry in entries {
                    self.registry.register_gameplay(entry);
                }
                self.gameplay_systems = self.gameplay_loader.system_entries().to_vec();
                if !self.gameplay_systems.is_empty() {
                    self.console.info(format!(
                        "Loaded {} gameplay systems",
                        self.gameplay_systems.len(),
                    ));
                }
                let gen_entries = self.gameplay_loader.generator_entries();
                if !gen_entries.is_empty() {
                    self.console.info(format!(
                        "Loaded {} generators: {}",
                        gen_entries.len(),
                        gen_entries.iter().map(|e| e.name).collect::<Vec<_>>().join(", "),
                    ));
                }
                self.generator_system.register_gameplay(gen_entries);
                self.generators_dirty = true;
                self.scene_dirty = true;
            }
            Err(e) => {
                self.console.error(format!("Failed to load gameplay dylib: {e}"));
            }
        }
    }

    /// Try to load an already-built gameplay dylib for the current project.
    fn try_load_gameplay_dylib(&mut self) {
        if let Some(ref project_dir) = self.project_dir {
            let dylib_path = crate::scaffold::gameplay_dylib_path(project_dir);
            self.load_gameplay_dylib(&dylib_path);
        }
    }

    /// Check if the gameplay dylib needs hot-reloading.
    fn check_gameplay_reload(&mut self) {
        if !self.gameplay_loader.needs_reload() {
            return;
        }

        self.console.info("Hot-reloading gameplay dylib...");

        // 1. Serialize all gameplay component data.
        let saved = self.gameplay_loader.serialize_all(&self.world, &self.entity_uuids);
        self.console.info(format!("Serialized {} gameplay component instances", saved.len()));

        // 2. Remove all gameplay components from entities.
        self.gameplay_loader.remove_all_gameplay_components(&mut self.world, &self.entity_uuids);

        // 3. Clear gameplay entries from registry.
        self.registry.clear_gameplay();
        self.generator_system.clear_gameplay_generators();
        self.generators_dirty = true;

        // 4. Unload old dylib.
        let dylib_path = self.gameplay_loader.dylib_path().map(|p| p.to_owned());
        self.gameplay_loader.unload();

        // 5. Load new dylib.
        if let Some(path) = dylib_path {
            // Small delay to ensure the file is fully written.
            std::thread::sleep(std::time::Duration::from_millis(100));

            match self.gameplay_loader.load(&path) {
                Ok(entries) => {
                    let names: Vec<&str> = entries.iter().map(|e| e.name).collect();
                    self.console.info(format!(
                        "Reloaded: {} components ({})",
                        entries.len(),
                        names.join(", "),
                    ));

                    // 6. Re-register gameplay entries.
                    for &entry in entries {
                        self.registry.register_gameplay(entry);
                    }

                    // 6b. Re-register gameplay generators. Without
                    // this every live generator entity stays Pending
                    // forever after a reload — `scan_and_submit`
                    // looks them up by name and finds nothing.
                    let gen_entries = self.gameplay_loader.generator_entries();
                    self.generator_system.register_gameplay(gen_entries);
                    self.generators_dirty = true;
                    if !gen_entries.is_empty() {
                        self.console.info(format!(
                            "Reloaded {} generators: {}",
                            gen_entries.len(),
                            gen_entries
                                .iter()
                                .map(|e| e.name)
                                .collect::<Vec<_>>()
                                .join(", "),
                        ));
                    }

                    // 7. Deserialize component data back.
                    let restored = self.gameplay_loader.deserialize_all(
                        &mut self.world,
                        &self.uuid_to_entity,
                        &saved,
                    );
                    self.console.info(format!("Restored {restored}/{} component instances", saved.len()));

                    // 7b. Force every live generator to re-run against
                    // the new code. Param-hash equality wouldn't catch
                    // a code change, so we explicitly mark stale.
                    let live_generators: Vec<hecs::Entity> = self
                        .world
                        .query::<&crate::generator::GeneratorState>()
                        .iter()
                        .map(|(e, _)| e)
                        .collect();
                    for entity in live_generators {
                        self.generator_system.force_regenerate(entity, &mut self.world);
                    }

                    // 8. Reload system entries and rebuild executor.
                    self.gameplay_systems = self.gameplay_loader.system_entries().to_vec();
                    if let Some(ref mut executor) = self.behavior_executor {
                        if let Err(e) = executor.rebuild(&self.gameplay_systems) {
                            self.console.error(format!("Failed to rebuild system schedule: {e}"));
                        } else {
                            self.console.info(format!(
                                "Rebuilt schedule: {} systems",
                                self.gameplay_systems.len(),
                            ));
                        }
                    }
                }
                Err(e) => {
                    self.console.error(format!("Hot-reload failed: {e}"));
                }
            }
        }

        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn process_file_events(&mut self) {
        let events = match self.file_watcher {
            Some(ref watcher) => watcher.poll_events(),
            None => return,
        };

        for event in events {
            use crate::file_watcher::FileEvent;
            match event {
                FileEvent::ModelChanged(path) => {
                    eprintln!("[RkpEngine] model changed: {}", path.display());
                    self.scan_models();
                    let path_str = path.to_string_lossy().to_string();
                    self.reload_asset(&path_str);
                }
                FileEvent::ShaderChanged(path) => {
                    eprintln!("[RkpEngine] shader changed: {}", path.display());
                    // TODO: recompile GPU pipelines
                }
                FileEvent::MaterialChanged(path) => {
                    eprintln!("[RkpEngine] material changed: {}", path.display());
                    self.material_lib.reload(&path);
                }
                FileEvent::MeshSourceChanged(path) => {
                    eprintln!("[RkpEngine] mesh source changed: {}", path.display());
                    let output = crate::import_worker::rkp_output_path(&path);
                    self.import_worker.submit(crate::import_worker::ImportRequest {
                        source_path: path,
                        output_path: output,
                        config: crate::import_worker::default_import_config(),
                    });
                }
                FileEvent::ScriptChanged(path) => {
                    eprintln!("[RkpEngine] script changed: {}", path.display());
                    self.scaffold_and_build_gameplay();
                }
            }
        }
    }

    fn build_inspector_snapshot(&self) -> Option<crate::inspector::InspectorSnapshot> {
        let selected = self.selected_entity?;
        if !self.world.contains(selected) {
            return None;
        }

        let name = self.world.get::<&crate::components::EditorMetadata>(selected)
            .map(|m| m.name.clone())
            .unwrap_or_default();

        use crate::inspector::*;

        // For procedural entities, the Transform.scale slider is a
        // proxy for Root.transform.scale (see
        // `redirect_transform_scale_to_root`). Pull the displayed value
        // from the tree so the slider reflects what's actually baked.
        let proc_root_scale: Option<[f32; 3]> = self
            .world
            .get::<&crate::components::ProceduralGeometry>(selected)
            .ok()
            .and_then(|pg| {
                let root = pg.tree.root();
                pg.tree.get(root).map(|node| {
                    node.transform
                        .to_scale_rotation_translation()
                        .0
                        .to_array()
                })
            });

        // Build component snapshots from the registry.
        let mut components = Vec::new();
        for entry in self.registry.components_on(&self.world, selected) {
            let fields: Vec<FieldSnapshot> = entry.meta.iter().map(|meta| {
                let mut value = (entry.get_field)(&self.world, selected, meta.name)
                    .unwrap_or(FieldValue::String("<error>".into()));
                if entry.name == "Transform"
                    && meta.name == "scale"
                    && let Some(s) = proc_root_scale
                {
                    value = FieldValue::Vec3(s);
                }
                FieldSnapshot {
                    name: meta.name.to_string(),
                    field_type: meta.field_type,
                    value,
                    range: meta.range,
                    transient: meta.transient,
                    enum_options: meta.enum_options
                        .map(|opts| opts.iter().map(|(v, l)| (v.to_string(), l.to_string())).collect())
                        .unwrap_or_default(),
                    scrub: meta.scrub,
                    ..Default::default()
                }
            }).collect();
            components.push(ComponentSnapshot {
                name: entry.name.to_string(),
                fields,
                removable: !entry.mandatory,
            });
        }

        // Extract position/rotation/scale from Transform if present.
        let transform = self.world.get::<&crate::components::Transform>(selected).ok();
        let pos = transform.as_ref().map(|t| t.position.to_array()).unwrap_or([0.0; 3]);
        let rot = transform.as_ref().map(|t| t.rotation.to_array()).unwrap_or([0.0; 3]);
        let scl = transform.as_ref().map(|t| t.scale.to_array()).unwrap_or([1.0; 3]);

        // Count per-material voxel usage if entity has spatial data.
        let material_usage = self.count_material_usage(selected);

        // Skeleton sidecar — clip + bone metadata for the dedicated
        // animation panel. Skipped when the entity has no skeleton.
        let skeleton = self.world.get::<&crate::components::Skeleton>(selected).ok()
            .map(|skel| {
                let bone_names: Vec<String> = skel.asset.skeleton.bones.iter().map(|b| b.name.clone()).collect();
                let bone_parents: Vec<i32> = skel.asset.skeleton.hierarchy.clone();
                let clips: Vec<ClipInfo> = skel.asset.clips.iter().map(|c| ClipInfo {
                    name: c.name.clone(),
                    duration: c.duration,
                }).collect();
                SkeletonInspector {
                    path: skel.path.to_string_lossy().into_owned(),
                    bone_names,
                    bone_parents,
                    clips,
                }
            });

        Some(InspectorSnapshot {
            entity_name: name,
            entity_id: format!("{}", self.get_entity_uuid(selected).as_simple()),
            position: pos,
            rotation: rot,
            scale: scl,
            components,
            material_usage,
            skeleton,
        })
    }

    /// Count per-material voxel usage for an entity's octree.
    fn count_material_usage(&self, entity: hecs::Entity) -> Vec<crate::inspector::MaterialUsage> {
        let renderable = match self.world.get::<&crate::components::Renderable>(entity) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let spatial = match &renderable.spatial {
            Some(s) => s,
            None => return Vec::new(),
        };

        // Collect leaf voxel slots from the packed octree buffer.
        // Branch offsets in the packed buffer are ABSOLUTE, so we traverse
        // the full buffer starting at root_offset (not a sub-slice).
        let sm = self.scene_mgr.lock().unwrap();
        let all_nodes = sm.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, spatial.root_offset as usize, &mut leaf_slots);

        // Count material IDs across all leaf slots. Every leaf is a surface
        // voxel now — no opacity gate.
        let pool_size = sm.leaf_attr_pool.allocated_count();
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        for slot in leaf_slots {
            if slot >= pool_size {
                continue; // stale or invalid slot — skip
            }
            let attr = sm.leaf_attr_pool.get(slot);
            *counts.entry(attr.material_primary).or_insert(0) += 1;
        }

        // Sort by voxel count descending.
        let mut usage: Vec<crate::inspector::MaterialUsage> = counts
            .into_iter()
            .map(|(material_id, voxel_count)| crate::inspector::MaterialUsage {
                material_id,
                voxel_count,
            })
            .collect();
        usage.sort_by(|a, b| b.voxel_count.cmp(&a.voxel_count));
        usage
    }

    /// Remap all voxels on an entity from one material to another.
    /// Returns the number of voxels changed.
    fn remap_entity_material(
        &mut self,
        entity: hecs::Entity,
        from_material: u16,
        to_material: u16,
    ) -> u32 {
        let renderable = match self.world.get::<&crate::components::Renderable>(entity) {
            Ok(r) => r.clone(),
            Err(_) => return 0,
        };
        let spatial = match &renderable.spatial {
            Some(s) => s.clone(),
            None => return 0,
        };

        // Collect leaf slots using absolute offsets in the packed buffer.
        let mut sm = self.scene_mgr.lock().unwrap();
        let all_nodes = sm.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, spatial.root_offset as usize, &mut leaf_slots);

        let pool_size = sm.leaf_attr_pool.allocated_count();
        let mut count = 0u32;
        for slot in leaf_slots {
            if slot >= pool_size { continue; }
            let attr = sm.leaf_attr_pool.get(slot);
            let primary = attr.material_primary;
            let secondary = attr.material_secondary();
            let mut changed = false;

            if primary == from_material {
                let m = sm.leaf_attr_pool.get_mut(slot);
                m.material_primary = to_material;
                changed = true;
            }
            if secondary == from_material {
                // Re-pack secondary + blend, since both share material_secondary_blend.
                let attr = *sm.leaf_attr_pool.get(slot);
                let blend = attr.blend_weight();
                let m = sm.leaf_attr_pool.get_mut(slot);
                let secondary_bits = (to_material & 0x0FFF) as u16;
                let blend_bits = ((blend as u16) & 0x0F) << 12;
                m.material_secondary_blend = secondary_bits | blend_bits;
                changed = true;
            }
            if changed {
                count += 1;
            }
        }
        count
    }

    /// Drain queued `ImportEvent`s from the worker and reduce them
    /// into `importing_progress`. Called each tick before
    /// `poll_import_completions` so a completion's final
    /// `StageEnd` / `Error` event lands in `importing_progress`
    /// before the entry is removed on completion.
    fn pump_import_events(&mut self) {
        use crate::snapshot::ImportProgressInfo;
        use rkp_import::ImportEvent;

        let events = self.import_worker.poll_events();
        for tagged in events {
            let source_key = tagged.source_path.to_string_lossy().into_owned();
            let entry = self
                .importing_progress
                .entry(source_key.clone())
                .or_insert_with(|| ImportProgressInfo {
                    source_path: source_key,
                    ..Default::default()
                });
            match tagged.event {
                ImportEvent::StageStart { stage, message } => {
                    entry.stage = stage.to_string();
                    entry.message = message;
                    entry.done = 0;
                    entry.total = 0;
                }
                ImportEvent::StageProgress { stage, done, total } => {
                    // Ignore stale progress events from a stage that
                    // already ended (shouldn't happen given the
                    // worker is single-threaded, but cheap to guard).
                    if entry.stage == stage {
                        entry.done = done;
                        entry.total = total;
                    }
                }
                ImportEvent::StageEnd { stage } => {
                    if entry.stage == stage && entry.total > 0 {
                        entry.done = entry.total;
                    }
                }
                ImportEvent::Warn { message } => {
                    entry.warnings.push(message);
                }
                ImportEvent::Error { message } => {
                    entry.error = Some(message);
                }
            }
        }
    }

    fn poll_import_completions(&mut self) {
        let completions = self.import_worker.poll_completions();
        for completion in completions {
            let source_key = completion.source_path.to_string_lossy().into_owned();
            if self.importing_sources.remove(&source_key) {
                self.importing_dirty = true;
            }
            self.importing_progress.remove(&source_key);
            match completion.result {
                Ok(result) => {
                    let name = completion.source_path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    self.console.info(format!(
                        "Import complete: {name} ({} voxels)",
                        result.shell_voxels,
                    ));
                    self.refresh_reimported_asset(&completion.output_path);
                    self.scan_models();
                }
                Err(e) => {
                    let name = completion.source_path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    self.console.error(format!("Import failed: {name} — {e}"));
                }
            }
        }
    }

    fn reload_asset(&mut self, path: &str) {
        // Find any scene objects that reference this asset and reload them.
        // For now, log that we detected the change.
        eprintln!("[RkpEngine] hot-reload asset: {path}");
        // TODO: remove old GPU objects for this asset, re-load from file,
        // rebuild faces, re-upload geometry.
    }

    /// After a re-import has rewritten the `.rkp` on disk, refresh the
    /// scene manager's cached copy and point any entities that were
    /// referencing it at the new geometry. No-op when the asset isn't
    /// currently loaded into the scene.
    fn refresh_reimported_asset(&mut self, output_path: &std::path::Path) {
        let path_str = output_path.to_string_lossy().into_owned();
        let reload = match self.scene_mgr.lock().unwrap().reload_asset(&path_str) {
            Ok(Some(r)) => r,
            Ok(None) => {
                eprintln!(
                    "[RkpEngine] refresh_reimported_asset: {} not in asset cache — \
                     no scene entities to refresh",
                    output_path.display(),
                );
                return;
            }
            Err(e) => {
                self.console.error(format!("Reload after import failed: {e}"));
                return;
            }
        };

        let entities_to_update: Vec<hecs::Entity> = self.world
            .query::<&crate::components::Renderable>()
            .iter()
            .filter_map(|(e, r)| (r.asset_handle == Some(reload.old_handle)).then_some(e))
            .collect();

        eprintln!(
            "[RkpEngine] refresh_reimported_asset: {} → {} entities to update \
             (old_handle={:?}, new_handle={:?}, voxels={})",
            output_path.display(),
            entities_to_update.len(),
            reload.old_handle,
            reload.new_handle,
            reload.info.voxel_count,
        );

        for entity in entities_to_update {
            if let Ok(mut r) = self.world.get::<&mut crate::components::Renderable>(entity) {
                let spatial = spatial_from_handle(
                    &reload.info.spatial,
                    reload.info.voxel_size,
                    &reload.info.aabb,
                    reload.info.grid_origin,
                    reload.info.leaf_attr_slot_start,
                    reload.info.leaf_attr_slot_count,
                    Vec::new(),
                );
                r.asset_handle = Some(reload.new_handle);
                r.spatial = Some(spatial);
                r.voxel_count = reload.info.voxel_count;
            }
        }
        // geometry_dirty: re-upload pools. gpu_objects_dirty: rebuild the
        // per-entity GpuObject list so the new AABB / octree offsets land
        // on the GPU (target_size, rotation offsets, etc. only show up in
        // the render once this runs).
        self.geometry_dirty = true;
        self.gpu_objects_dirty = true;
    }

    /// Resolve a Uuid (from UI) to an hecs::Entity.
    fn resolve_entity(&self, uuid: &uuid::Uuid) -> Option<hecs::Entity> {
        self.uuid_to_entity.get(uuid).copied()
    }

    /// Get the stable UUID for an hecs Entity.
    fn get_entity_uuid(&self, entity: hecs::Entity) -> uuid::Uuid {
        self.entity_uuids.get(&entity).copied()
            .unwrap_or_else(uuid::Uuid::nil)
    }

    /// Generate a unique entity name. If `base` already exists, appends a number.
    fn unique_name(&self, base: &str) -> String {
        let existing: std::collections::HashSet<String> = self.world
            .query::<&crate::components::EditorMetadata>()
            .iter()
            .map(|(_, m)| m.name.clone())
            .collect();
        if !existing.contains(base) {
            return base.to_string();
        }
        for i in 1.. {
            let candidate = format!("{base} ({i})");
            if !existing.contains(&candidate) {
                return candidate;
            }
        }
        base.to_string()
    }

    /// Extract an intelligent display name from an asset path.
    /// Uses parent directory name if the filename is generic (scene, model, etc.).
    fn display_name_from_path(path: &str) -> String {
        let p = std::path::Path::new(path);
        let stem = p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        // If the filename is generic, use the parent directory name.
        let generic_names = ["scene", "model", "mesh", "object", "default", "untitled"];
        if generic_names.iter().any(|g| stem.eq_ignore_ascii_case(g)) {
            if let Some(parent) = p.parent().and_then(|p| p.file_name()) {
                let parent_name = parent.to_string_lossy().into_owned();
                // Don't use generic parent names either.
                if !generic_names.iter().any(|g| parent_name.eq_ignore_ascii_case(g))
                    && parent_name != "objects" && parent_name != "assets" && parent_name != "models"
                {
                    return parent_name;
                }
            }
        }
        stem
    }

    /// Spawn an .rkp asset at a world-space position. The passed `pos`
    /// is interpreted as the surface point the user wants the asset to
    /// stand on — the asset's AABB bottom is snapped there (i.e.
    /// `transform.position.y = pos.y - info.aabb.min.y`), matching
    /// rkifield's drop-on-geometry behaviour.
    fn spawn_asset(&mut self, path: &str, pos: glam::Vec3) {
        let _ = self.spawn_asset_ex(path, pos, true);
    }

    /// Spawn an .rkp asset and return (entity, aabb_min_y) — the latter
    /// is cached by the drag-preview so every subsequent pick-result
    /// update can apply the same AABB-bottom snap without reloading
    /// the asset info. `verbose` gates the console log; drag-preview
    /// spawns are noisy without it.
    fn spawn_asset_ex(&mut self, path: &str, pos: glam::Vec3, verbose: bool)
        -> Option<(hecs::Entity, f32)>
    {
        use crate::components::*;
        let acquired = self.scene_mgr.lock().unwrap().acquire_asset(path);
        match acquired {
            Ok((handle, info)) => {
                let raw_name = Self::display_name_from_path(path);
                let name = self.unique_name(&raw_name);
                let spatial = spatial_from_handle(
                    &info.spatial, info.voxel_size, &info.aabb, info.grid_origin,
                    info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new(),
                );
                let mut transform = Transform::default();
                transform.position = glam::Vec3::new(pos.x, pos.y - info.aabb.min.y, pos.z);
                let entity = self.world.spawn((
                    transform,
                    EditorMetadata { name: name.clone() },
                    Renderable {
                        asset_path: Some(path.to_string()),
                        voxel_count: info.voxel_count,
                        spatial: Some(spatial),
                        asset_handle: Some(handle),
                        ..Default::default()
                    },
                ));
                self.assign_entity_uuid(entity);
                self.geometry_dirty = true;
                self.scene_dirty = true;
                self.gpu_objects_dirty = true;
                if verbose {
                    self.console.info(format!("Loaded '{name}': {} voxels", info.voxel_count));
                }
                Some((entity, info.aabb.min.y))
            }
            Err(e) => {
                self.console.error(format!("Failed to load '{path}': {e}"));
                None
            }
        }
    }

    /// If a sibling `.rkskel` exists alongside the `.rkp` path, load it
    /// into the animation cache and attach `Skeleton` + a default
    /// paused `AnimationPlayer` to the entity. Missing sidecar is not
    /// an error — static meshes are expected.
    fn try_attach_skeleton(&mut self, entity: hecs::Entity, rkp_path: &std::path::Path) {
        let rkskel_path = rkp_path.with_extension("rkskel");
        if !rkskel_path.exists() {
            return;
        }
        match self.animation_cache.get_or_load(&rkskel_path) {
            Ok(asset) => {
                // Skeleton is transient — always attach/replace with the
                // freshly-loaded asset so the `current_pose` matches the
                // current bone count.
                //
                // Fold the grid-frame offset into the skeleton's
                // glTF→local transform: `rest_bone_aabbs` and the
                // scatter's `rest_pos` live in grid frame (octree
                // corner at 0, range [0, extent]), so the pose
                // produced by `animation::tick` must operate on grid-
                // frame positions too. The offset is
                // `half_extent = base_voxel_size × 2^depth / 2`,
                // available from the entity's spatial data (present
                // because `LoadAsset` populated `Renderable` before
                // this function runs).
                let grid_offset = self.world
                    .get::<&crate::components::Renderable>(entity)
                    .ok()
                    .and_then(|r| r.spatial.as_ref().map(|s| {
                        let he = 0.5 * s.base_voxel_size * (1u32 << s.depth) as f32;
                        glam::Vec3::splat(he)
                    }))
                    .unwrap_or(glam::Vec3::ZERO);
                let skeleton = crate::animation::skeleton_component(
                    asset.clone(), rkskel_path.clone(), grid_offset,
                );
                if let Err(e) = self.world.insert_one(entity, skeleton) {
                    eprintln!("[RkpEngine] attach Skeleton: world.insert_one failed: {e}");
                    return;
                }
                // Only attach a default `AnimationPlayer` if the entity
                // doesn't already have one — scene load may have
                // deserialized a persisted player with user-chosen clip
                // / time / loop mode.
                let has_player = self.world.get::<&crate::components::AnimationPlayer>(entity).is_ok();
                if !has_player {
                    let player = crate::animation::default_player(&asset);
                    if let Err(e) = self.world.insert_one(entity, player) {
                        eprintln!("[RkpEngine] attach AnimationPlayer: world.insert_one failed: {e}");
                    }
                }
                eprintln!(
                    "[RkpEngine] attached skeleton ({} bones, {} clips) from {}",
                    asset.skeleton.bones.len(),
                    asset.clips.len(),
                    rkskel_path.display(),
                );
            }
            Err(e) => {
                self.console.warn(format!("load .rkskel failed: {e}"));
            }
        }
    }

    /// Assign a stable UUID to an entity.
    fn assign_entity_uuid(&mut self, entity: hecs::Entity) -> uuid::Uuid {
        let uuid = uuid::Uuid::new_v4();
        self.entity_uuids.insert(entity, uuid);
        self.uuid_to_entity.insert(uuid, entity);
        // Fresh spawns append to the end of the tree. `entry` makes
        // the assignment idempotent: if a caller pre-seeded an order
        // (e.g. scene-load from a persisted value), we keep it.
        self.entity_tree_order.entry(entity).or_insert_with(|| {
            let key = self.next_tree_order;
            self.next_tree_order += 1.0;
            key
        });
        uuid
    }

    /// Bind an entity to a specific (pre-existing) UUID. Used by scene
    /// load so entities keep the ID they had when the scene was saved,
    /// not a fresh random one. Keeping IDs stable is what lets paths
    /// derived from UUIDs — like procedural bake sidecars — survive a
    /// reload.
    fn set_entity_uuid(&mut self, entity: hecs::Entity, uuid: uuid::Uuid) {
        self.entity_uuids.insert(entity, uuid);
        self.uuid_to_entity.insert(uuid, entity);
    }

    /// Rebuild GPU objects from the hecs world.
    /// Per-tick procedural maintenance. Bakes any entity whose
    /// `pending_bake` has settled past the debounce window. We
    /// deliberately do NOT auto-bake on `dirty` alone: scene load
    /// restores procedurals with a cached spatial and a clean flag,
    /// but if a rogue edit path left `dirty = true` we'd silently
    /// re-run a potentially huge bake at startup — historically a
    /// source of UI freezes and crashes. Manual bakes (the build
    /// panel's Bake button, `BakeProceduralEntity`, `BakeAllDirty`)
    /// explicitly set `pending_bake` so they still ride this path.
    fn update_dirty_procedurals(&mut self) {
        use crate::components::*;

        let mut to_update: Vec<hecs::Entity> = Vec::new();

        // Debounce window for `pending_bake` — long enough to suppress
        // bakes mid-scrub on a slider, short enough to feel immediate
        // when the user releases.
        const BAKE_DEBOUNCE: std::time::Duration =
            std::time::Duration::from_millis(150);
        let now = std::time::Instant::now();
        // Bakes are sync and can take ~1s on big objects. Firing one
        // mid-drag freezes the engine tick and the gizmo can't track
        // the cursor for the duration — looks like the bake "ate" the
        // drag motion when the queued events finally drain. Defer
        // until the gizmo is released; the existing debounce timestamp
        // was bumped by the last drag tick, so this only delays the
        // bake by however long the user keeps dragging.
        let drag_active = self.gizmo.dragging || self.proc_gizmo.dragging;

        for (entity, (_renderable, proc_geo)) in self
            .world
            .query::<(&Renderable, &ProceduralGeometry)>()
            .iter()
        {
            // Only one bake per entity in flight at a time — the
            // worker channel could otherwise bloat with dozens of
            // requests during a long bake, all destined to be stale.
            // New edits while a bake runs will queue a fresh request
            // on the tick after the current one returns, via the
            // preserved `dirty` / `pending_bake` flags.
            if proc_geo.bake_in_flight {
                continue;
            }
            let pending_settled = !drag_active
                && proc_geo.pending_bake
                && proc_geo
                    .bake_dirty_at
                    .map(|t| now.duration_since(t) >= BAKE_DEBOUNCE)
                    .unwrap_or(true);
            if pending_settled {
                to_update.push(entity);
            }
        }

        for entity in to_update {
            self.enqueue_bake(entity);
        }
    }

    /// Move the just-set `Transform.scale` onto the procedural Root
    /// node (preserving Root's existing rotation / translation), set
    /// `Transform.scale` to the preview multiplier
    /// `new_root / last_evaluated_root` so the still-old baked voxels
    /// stretch to the user's target size during the debounce window,
    /// and queue an auto-bake. No-op for non-procedural entities.
    ///
    /// **Invariant**: after every call, for procedural entities,
    /// `Transform.scale == Root.scale / last_evaluated_root_scale`. The
    /// caller is expected to have written the user's intended absolute
    /// scale into `Transform.scale` first; this method captures it,
    /// stores it on the tree, and overwrites `Transform.scale` with the
    /// preview multiplier. Skipping this overwrite — even on a "no
    /// change" tick — causes a visual jump because the rendered size
    /// is `Transform.scale × baked_voxels`, and a stale absolute value
    /// in `Transform.scale` will multiply the already-baked-up voxels
    /// a second time.
    fn redirect_transform_scale_to_root(&mut self, entity: hecs::Entity) {
        use crate::components::*;
        // Hard cap on Root.scale per axis. The voxel budget scales as
        // the squared surface area (roughly) and the bake wall time
        // follows suit, so uncapped scaling blows through GPU memory
        // and wall-clock quickly. 20× the default primitive's extent
        // puts a 0.35 m sphere at 7 m radius, well inside the octree's
        // depth-11 cap and the 4.2 M-per-dispatch GPU chunking. Tune
        // in one place; the field meta's slider range below is kept
        // in lockstep.
        const SCALE_MIN: f32 = 0.01;
        const SCALE_MAX: f32 = 20.0;

        let user_scale = match self.world.get::<&Transform>(entity) {
            Ok(t) => t.scale,
            Err(_) => return,
        };
        // Procedurals-only: clamp here so the property slider + gizmo
        // both hit the same ceiling. Non-procedurals keep whatever
        // scale they were given.
        let is_procedural = self.world.get::<&ProceduralGeometry>(entity).is_ok();
        let user_scale = if is_procedural {
            glam::Vec3::new(
                user_scale.x.clamp(SCALE_MIN, SCALE_MAX),
                user_scale.y.clamp(SCALE_MIN, SCALE_MAX),
                user_scale.z.clamp(SCALE_MIN, SCALE_MAX),
            )
        } else {
            user_scale
        };
        let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) else {
            // Non-procedural entity — leave Transform.scale alone.
            return;
        };
        let root_id = proc_geo.tree.root();
        let root_xf = proc_geo
            .tree
            .get(root_id)
            .map(|n| n.transform)
            .unwrap_or(glam::Affine3A::IDENTITY);
        let (current_root_scale, rot, trans) = root_xf.to_scale_rotation_translation();
        // Push to Root + queue an auto-bake only when the value
        // actually changed; spammy slider events that re-write the
        // same scale shouldn't bump the debounce timestamp.
        if (user_scale - current_root_scale).length() >= 1e-6 {
            let new_root =
                glam::Affine3A::from_scale_rotation_translation(user_scale, rot, trans);
            proc_geo.tree.set_transform(root_id, new_root);
            proc_geo.pending_bake = true;
            proc_geo.bake_dirty_at = Some(std::time::Instant::now());
        }
        // Always restore the preview multiplier — see invariant above.
        let baked = proc_geo.last_evaluated_root_scale;
        let safe = |a: f32, b: f32| if b.abs() > 1e-6 { a / b } else { 1.0 };
        let preview = glam::Vec3::new(
            safe(user_scale.x, baked.x),
            safe(user_scale.y, baked.y),
            safe(user_scale.z, baked.z),
        );
        drop(proc_geo);
        if let Ok(mut t) = self.world.get::<&mut Transform>(entity) {
            t.scale = preview;
        }
    }

    /// Compute the bake-cache sidecar path for a procedural entity:
    /// `{scene_dir}/{scene_stem}.bakes/{uuid}.rkp`. Returns `None` when
    /// the scene has no on-disk path yet (unsaved scratch session) or
    /// the entity has no UUID. The relative form
    /// (`{scene_stem}.bakes/{uuid}.rkp`) is what `SceneObject.procedural_cache`
    /// stores — use [`procedural_cache_relative`] for that.
    fn procedural_cache_path(&self, entity: hecs::Entity) -> Option<std::path::PathBuf> {
        let uuid = self.entity_uuids.get(&entity).copied()?;
        let scene_path = self.scene_path.as_ref()?;
        let parent = scene_path.parent()?;
        let stem = scene_path.file_stem()?;
        let mut dir = parent.to_path_buf();
        dir.push(format!("{}.bakes", stem.to_string_lossy()));
        dir.push(format!("{}.rkp", uuid));
        Some(dir)
    }

    /// Enqueue an async bake for a procedural entity. Bumps the
    /// entity's `bake_generation` and sends a [`BakeRequest`] to the
    /// worker thread. The result (an integrate-able [`BakeArtifact`])
    /// is picked up later by `drain_bake_results`. If the user keeps
    /// editing before the bake finishes, subsequent calls bump the
    /// generation and the old result gets dropped on arrival.
    ///
    /// Returns the generation number assigned, or `None` if the entity
    /// isn't a procedural.
    fn enqueue_bake(&mut self, entity: hecs::Entity) -> Option<u64> {
        use crate::components::*;

        let (tree_clone, base_voxel_size, generation) = {
            let mut proc_geo = self.world.get::<&mut ProceduralGeometry>(entity).ok()?;
            proc_geo.bake_generation = proc_geo.bake_generation.wrapping_add(1);
            // Clear the edit flags that triggered this bake — the
            // request captures the current state, so if no new edit
            // follows, we shouldn't re-fire next tick. A subsequent
            // edit will set `dirty` / `pending_bake` again, and the
            // NEXT tick after the bake returns will pick those up.
            proc_geo.dirty = false;
            proc_geo.pending_bake = false;
            proc_geo.bake_dirty_at = None;
            proc_geo.bake_in_flight = true;
            (proc_geo.tree.clone(), proc_geo.voxel_size, proc_geo.bake_generation)
        };
        // Worker needs the previous allocation to free it under the
        // integrate lock — pull it now so the worker doesn't round-
        // trip through the ECS.
        let prev_spatial = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| r.spatial.clone());

        let (aabb, voxel_size) = procedural_voxel_params(&tree_clone, base_voxel_size);
        let instructions = rkp_procedural::flatten_tree(&tree_clone);
        let root_scale = tree_clone
            .get(tree_clone.root())
            .map(|n| n.transform.to_scale_rotation_translation().0)
            .unwrap_or(glam::Vec3::ONE);

        // Build the sidecar .rkp path: `{scene_dir}/{scene_stem}.bakes/{uuid}.rkp`.
        // Both the scene path and entity UUID must be known; unsaved
        // scratch scenes skip caching and just rely on next-spawn re-bake.
        let cache_output_path = self.procedural_cache_path(entity);

        let req = crate::bake_worker::BakeRequest {
            entity,
            generation,
            input: crate::bake_worker::BakeInput::Procedural(instructions),
            aabb,
            voxel_size,
            root_scale,
            prev_spatial,
            cache_output_path,
            generator_child: None,
        };
        if self.bake_worker.tx_request.send(req).is_err() {
            self.console.warn("bake worker channel closed".to_string());
            // Revert the in-flight flag — otherwise a permanently
            // dead channel would lock the entity out of future bakes.
            if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                proc_geo.bake_in_flight = false;
            }
            return None;
        }
        Some(generation)
    }

    /// Drain any finished bake results from the worker and integrate
    /// each one whose generation still matches the entity's latest
    /// request (stale results from superseded edits get silently
    /// dropped). Called once per tick, before rendering.
    fn drain_bake_results(&mut self) {
        use crate::components::*;
        use crate::bake_worker::BakeOutcome;

        // Drain everything the worker has produced since the last
        // tick. `try_recv` is non-blocking — we never wait here.
        while let Ok(result) = self.bake_worker.rx_result.try_recv() {
            // Generator-emitted child: spawn a new entity (anonymous)
            // or update an existing entity (persistent slot_key). The
            // `entity` field on the request is the generator (parent);
            // the spec carries everything needed downstream.
            if let Some(spec) = result.generator_child {
                match result.outcome {
                    BakeOutcome::Ok { spatial, voxel_count } => {
                        self.spawn_or_update_generated_child(
                            spec.parent_entity,
                            spec.local_transform,
                            spec.generation,
                            spec.slot_key,
                            spatial,
                            voxel_count,
                            spec.name_hint,
                        );
                    }
                    BakeOutcome::Failed => {
                        self.console.warn(format!(
                            "Generator child voxelization failed (parent={:?}, vs={:.4}).",
                            spec.parent_entity, result.voxel_size,
                        ));
                    }
                }
                continue;
            }

            // Regular procedural-entity bake below.
            let entity = result.entity;
            if !self.world.contains(entity) {
                continue;
            }

            // Every result clears the in-flight gate. If the user
            // edited after the request was sent, `dirty` /
            // `pending_bake` will already be set again and the next
            // tick's `update_dirty_procedurals` will enqueue a fresh
            // bake. We deliberately do NOT clear those flags here —
            // that would swallow the new edit.
            if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                proc_geo.bake_in_flight = false;
            }

            // Generation-mismatch = stale, drop.
            let current_gen = self
                .world
                .get::<&ProceduralGeometry>(entity)
                .map(|pg| pg.bake_generation)
                .unwrap_or(0);
            if result.generation != current_gen {
                continue;
            }

            match result.outcome {
                BakeOutcome::Ok { spatial, voxel_count } => {
                    self.apply_bake_result(
                        entity,
                        result.root_scale,
                        spatial,
                        voxel_count,
                    );
                }
                BakeOutcome::Failed => {
                    // Keep `dirty` / `pending_bake` intact so the user
                    // can retry (via a new edit or the Bake button) —
                    // clearing them would pretend the bake succeeded.
                    self.console.warn(format!(
                        "Procedural voxelization failed (voxel_size={:.4}, extent={:.1}).",
                        result.voxel_size,
                        (result.aabb.max - result.aabb.min).length(),
                    ));
                }
            }
        }
    }

    /// Pump the generator system once per frame.
    ///
    /// Surfaces notable lifecycle events (submit / complete / fail /
    /// cancel) to the console so the user sees them there even when
    /// the generator panel is hidden. Per-entity status updates land
    /// on the ECS via `tick()` itself.
    fn tick_generators(&mut self) {
        // Compute the per-session bake-cache directory for emitted
        // children: `{scene_dir}/{scene_stem}.bakes/`. Same directory
        // procedurals use for their per-entity caches — generator
        // children get filenames keyed by `(parent_uuid, slot_key)` so
        // the two never collide. `None` here means the scene is
        // unsaved; persistent emits run but won't write a cache, so a
        // save+reload of an unsaved-then-saved scene will trigger a
        // one-time regen until the first bake completes after save.
        let child_cache_dir = self
            .scene_path
            .as_ref()
            .and_then(|p| {
                let parent = p.parent()?;
                let stem = p.file_stem()?;
                Some(parent.join(format!("{}.bakes", stem.to_string_lossy())))
            });
        if let Some(ref dir) = child_cache_dir {
            // Best-effort create — same lazy create pattern used by
            // procedural caches.
            let _ = std::fs::create_dir_all(dir);
        }
        let events = self.generator_system.tick(
            &mut self.world,
            &self.registry,
            &self.entity_uuids,
            child_cache_dir.as_deref(),
        );
        for ev in events {
            use crate::generator::GeneratorEvent;
            match ev {
                GeneratorEvent::WillResubmit { entity, name: _ } => {
                    // Reset the per-generation slot-key tracker before
                    // the new run's emits land. The spawn-or-update
                    // path repopulates it as each child arrives, and
                    // the Completed handler diffs the resulting set
                    // against the existing children to despawn slots
                    // the generator no longer emits.
                    self.pending_generator_slot_keys.remove(&entity);
                }
                GeneratorEvent::Submitted { entity, name } => {
                    eprintln!("[gen] submit entity={entity:?} name={name}");
                }
                GeneratorEvent::Completed { entity, name } => {
                    eprintln!("[gen] completed entity={entity:?} name={name}");
                    // Orphan cleanup: delete persistent children whose
                    // slot_key wasn't emitted in this generation.
                    let seen = self
                        .pending_generator_slot_keys
                        .remove(&entity)
                        .unwrap_or_default();
                    let parent_uuid = self.entity_uuids.get(&entity).copied();
                    let orphans: Vec<hecs::Entity> = if let Some(pu) = parent_uuid {
                        self.world
                            .query::<&crate::generator::GeneratorOwned>()
                            .iter()
                            .filter(|(_, owned)| {
                                owned.parent_uuid == pu && !seen.contains(&owned.slot_key)
                            })
                            .map(|(e, _)| e)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    for child in orphans {
                        self.delete_entity(child);
                    }
                    self.scene_dirty = true;
                }
                GeneratorEvent::Failed { entity, name, error } => {
                    self.console.error(format!(
                        "Generator '{name}' on {entity:?} failed: {error}"
                    ));
                }
                GeneratorEvent::Cancelled { entity, name } => {
                    eprintln!("[gen] cancelled entity={entity:?} name={name}");
                }
            }
        }
    }

    /// Spawn or update a child entity for a generator's emitted bake.
    ///
    /// * Anonymous (`slot_key.is_none()`): always spawn a brand-new
    ///   entity. The previous generation's anonymous children were
    ///   blown away in `tick_generators`'s `WillResubmit` handler.
    /// * Persistent (`slot_key.is_some()`): look for an existing
    ///   `(parent, slot_key)` match. If found, replace its Transform +
    ///   Renderable.spatial in place — preserves any user-attached
    ///   components (lights, scripts, colliders). If not, spawn a new
    ///   entity tagged with the slot key.
    ///
    /// Either way, the world transform is composed against the
    /// parent's *current* transform (not a stale snapshot from the
    /// worker), so dragging the generator between emit and spawn
    /// still places the child correctly.
    fn spawn_or_update_generated_child(
        &mut self,
        generator_entity: hecs::Entity,
        local_transform: crate::components::Transform,
        generation: u64,
        slot_key: String,
        spatial: crate::components::SpatialData,
        voxel_count: u32,
        name_hint: Option<String>,
    ) {
        use crate::components::*;
        if !self.world.contains(generator_entity) {
            return;
        }

        let parent_transform = self
            .world
            .get::<&Transform>(generator_entity)
            .map(|t| (*t).clone())
            .unwrap_or_default();
        let world_transform = compose_generator_transforms(
            &parent_transform,
            &local_transform,
        );

        // Track that this slot was emitted this generation, so the
        // Completed handler knows which children survived (vs. which
        // to despawn as orphans because the generator stopped emitting
        // them).
        self.pending_generator_slot_keys
            .entry(generator_entity)
            .or_default()
            .insert(slot_key.clone());

        if let Some(existing) = self.find_persistent_child(generator_entity, &slot_key) {
            // Reuse: free the old geometry, swap the Renderable's
            // spatial in place, refresh transform + generation. Other
            // components stay → user-attached lights / scripts survive
            // across regens.
            self.release_renderable_geometry(existing);
            if let Ok(mut t) = self.world.get::<&mut Transform>(existing) {
                *t = world_transform;
            }
            if let Ok(mut r) = self.world.get::<&mut Renderable>(existing) {
                r.spatial = Some(spatial);
                r.voxel_count = voxel_count;
                // Reload-from-cache populates `asset_handle` (children
                // round-trip through the asset cache on load). The
                // fresh bake hands us a raw scene-pool allocation,
                // not an asset — clear the stale handle so the NEXT
                // regen's release_renderable_geometry takes the
                // deallocate-spatial path instead of releasing an
                // asset that was already released up above.
                r.asset_handle = None;
            }
            if let Ok(mut owned) =
                self.world.get::<&mut crate::generator::GeneratorOwned>(existing)
            {
                owned.generation = generation;
            }
            self.scene_dirty = true;
            self.geometry_dirty = true;
            self.gpu_objects_dirty = true;
            eprintln!(
                "[gen] reused child entity={existing:?} parent={generator_entity:?} \
                 slot='{slot_key}' voxels={voxel_count} gen={generation}"
            );
            return;
        }

        let base_name = name_hint.unwrap_or_else(|| {
            self.world
                .get::<&EditorMetadata>(generator_entity)
                .map(|m| format!("{}.child", m.name))
                .unwrap_or_else(|_| "child".into())
        });
        let name = self.unique_name(&base_name);
        let renderable = Renderable {
            asset_path: None,
            primitive: None,
            material_id: 0,
            voxel_count,
            spatial: Some(spatial),
            ..Default::default()
        };
        let parent_uuid = self.entity_uuids.get(&generator_entity).copied();
        // GeneratorOwned needs the parent's UUID; without it the marker
        // can't survive a save/load (queries match by UUID, not Entity).
        // If the generator entity has somehow lost its UUID, fail loud
        // instead of silently spawning an orphan.
        let owned_parent_uuid = match parent_uuid {
            Some(u) => u,
            None => {
                eprintln!(
                    "[gen] spawn_or_update_generated_child: generator entity {generator_entity:?} has no UUID; dropping child",
                );
                return;
            }
        };
        let child = self.world.spawn((
            world_transform,
            EditorMetadata { name: name.clone() },
            renderable,
            crate::generator::GeneratorOwned {
                parent_uuid: owned_parent_uuid,
                generation,
                slot_key: slot_key.clone(),
            },
        ));
        // Attach Parent for the scene tree. The transform stored on
        // `child` is already absolute world — we don't compose on GPU
        // build — but Parent makes the scene_tree panel show the child
        // indented under its generator.
        if let Some(uuid) = parent_uuid {
            let _ = self.world.insert_one(
                child,
                crate::components::Parent { parent_id: uuid },
            );
        }
        self.assign_entity_uuid(child);
        self.scene_dirty = true;
        self.geometry_dirty = true;
        self.gpu_objects_dirty = true;
        eprintln!(
            "[gen] spawned child entity={child:?} parent={generator_entity:?} \
             name='{name}' slot='{slot_key}' voxels={voxel_count} gen={generation}"
        );
    }

    /// Find the existing persistent child of `parent` matching
    /// `slot_key`, if any.
    fn find_persistent_child(
        &self,
        parent: hecs::Entity,
        slot_key: &str,
    ) -> Option<hecs::Entity> {
        let parent_uuid = self.entity_uuids.get(&parent).copied()?;
        self.world
            .query::<&crate::generator::GeneratorOwned>()
            .iter()
            .find(|(_, owned)| {
                owned.parent_uuid == parent_uuid && owned.slot_key == slot_key
            })
            .map(|(e, _)| e)
    }

    /// Release the GPU pool slots held by `entity`'s `Renderable`,
    /// without despawning the entity. Used by the persistent-child
    /// reuse path: the entity stays, only the geometry is swapped.
    /// Mirrors the asset/spatial branches of `delete_entity`.
    fn release_renderable_geometry(&mut self, entity: hecs::Entity) {
        use crate::components::*;
        if let Ok(renderable) = self.world.get::<&Renderable>(entity) {
            if let Some(handle) = renderable.asset_handle {
                drop(renderable);
                self.scene_mgr.lock().unwrap().release_asset(handle);
            } else if let Some(ref spatial) = renderable.spatial {
                let handle = rkp_core::OctreeHandle {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let slot_start = spatial.voxel_slot_start;
                let slot_count = spatial.voxel_slot_count;
                let brick_ids = spatial.brick_ids.clone();
                drop(renderable);
                self.scene_mgr.lock().unwrap().deallocate_geometry(
                    &handle, slot_start, slot_count, &brick_ids,
                );
            }
        }
        // Wipe the spatial so the renderer doesn't see stale data
        // between this point and the caller's swap-in.
        if let Ok(mut r) = self.world.get::<&mut Renderable>(entity) {
            r.spatial = None;
        }
    }

    /// Apply a completed (already-integrated) bake's result to the
    /// ECS. The heavy work — dealloc, artifact remap, pool writes —
    /// already happened on the bake worker under the `scene_mgr`
    /// lock; this runs on the engine tick and only touches the ECS,
    /// so it's microseconds.
    fn apply_bake_result(
        &mut self,
        entity: hecs::Entity,
        baked_root_scale: glam::Vec3,
        spatial: crate::components::SpatialData,
        voxel_count: u32,
    ) {
        use crate::components::*;

        if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
            renderable.voxel_count = voxel_count;
            renderable.spatial = Some(spatial);
        }

        // Recompute `Transform.scale` as `current_root / baked_root`
        // so the visual size stays equal to the user's latest intent
        // across the integrate:
        //     visual = Transform.scale × baked_voxels_world_scale
        //            = (current_root / baked_root) × baked_root
        //            = current_root
        // If no mid-bake edit happened, current_root == baked_root
        // and `Transform.scale` collapses to 1.
        let current_root_scale = self
            .world
            .get::<&ProceduralGeometry>(entity)
            .ok()
            .and_then(|pg| {
                pg.tree
                    .get(pg.tree.root())
                    .map(|n| n.transform.to_scale_rotation_translation().0)
            })
            .unwrap_or(baked_root_scale);
        let safe = |a: f32, b: f32| if b.abs() > 1e-6 { a / b } else { 1.0 };
        let new_transform_scale = glam::Vec3::new(
            safe(current_root_scale.x, baked_root_scale.x),
            safe(current_root_scale.y, baked_root_scale.y),
            safe(current_root_scale.z, baked_root_scale.z),
        );

        if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
            // `dirty` / `pending_bake` / `bake_dirty_at` were cleared
            // at enqueue time. If they're set *now*, the user edited
            // after the request was sent — preserve the new intent so
            // the next tick re-enqueues.
            proc_geo.last_evaluated_root_scale = baked_root_scale;
        }
        if let Ok(mut t) = self.world.get::<&mut Transform>(entity) {
            t.scale = new_transform_scale;
        }

        self.geometry_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn update_scene_gpu(&mut self) {
        use crate::components::*;

        // Re-pack every skinned entity's current pose into the scene
        // bone buffer. Empty when no animated entities are loaded.
        self.bone_matrix_allocator.rebuild(&self.world);
        // Wipe last frame's scatter plan — rebuilt below per skinned
        // entity. Bone-field cells are u64s of offset; cells × 8 B =
        // total bone-field bytes.
        self.skin_dispatches.clear();
        let mut running_bone_field_cells: u32 = 0;
        let mut running_bone_field_occ_u32s: u32 = 0;

        // Pose cache for the scatter-skip check. Built in lock-step
        // with the planning loop and swapped with `last_skin_poses`
        // at the end of this function. Equal maps → skip scatter.
        let mut this_frame_poses: std::collections::HashMap<hecs::Entity, Vec<glam::Mat4>>
            = std::collections::HashMap::new();

        self.gpu_objects.clear();
        self.gpu_to_entity.clear();
        self.entity_to_gpu.clear();

        // Refresh the sim-side skinning_data cache only when a
        // geometry mutation has happened since we last built it.
        // Steady-state (no bakes / no asset (un)loads) does ZERO
        // scene_mgr lock acquisitions per tick — so even a busy
        // bake_worker holding the Mutex for hundreds of ms can't
        // stall sim. When the epoch does advance (a bake completed,
        // an asset was loaded), we lock once to refresh; that one
        // wait might be long if bake_worker has *already started*
        // the next bake, but we pay it once per epoch bump rather
        // than once per skinned entity per tick.
        if self.skinning_enabled {
            let current_epoch = self
                .geometry_epoch_handle
                .load(std::sync::atomic::Ordering::Acquire);
            if current_epoch > self.skinning_data_cache_epoch {
                let asset_handles: std::collections::HashSet<rkp_render::AssetHandle> = self
                    .world
                    .query::<&Renderable>()
                    .iter()
                    .filter_map(|(_, r)| r.asset_handle)
                    .collect();
                self.skinning_data_cache.clear();
                if !asset_handles.is_empty() {
                    let sm = self.scene_mgr.lock().unwrap();
                    for h in asset_handles {
                        if let Some(data) = sm.skinning_data(h).cloned() {
                            self.skinning_data_cache.insert(h, data);
                        }
                    }
                }
                self.skinning_data_cache_epoch = current_epoch;
            }
        }

        // Collect renderable entities and sort by `Entity::to_bits()`
        // — hecs assigns monotonically-increasing bits to every spawn
        // (generation << 32 | index), so this is stable per entity
        // while alive AND gives newest-at-bottom ordering naturally.
        // hecs query iteration order follows archetype layout, which
        // shifts when a new archetype appears — without this sort,
        // gpu vec positions of existing entities reshuffle on every
        // such event. Since render-side `interpolate_gpu_objects`
        // matches prev↔curr by `object_id == gpu_idx`, any shift
        // would blend each entity against some unrelated entity's
        // previous world matrix (visible smear, then pop-back).
        let mut ordered: Vec<hecs::Entity> = self.world
            .query::<(&Transform, &Renderable)>()
            .iter()
            .filter_map(|(entity, (_, r))| {
                if r.spatial.is_some() { Some(entity) } else { None }
            })
            .collect();
        ordered.sort_by_key(|e| e.to_bits());

        for entity in ordered {
            let Ok(transform) = self.world.get::<&Transform>(entity) else { continue };
            let Ok(renderable) = self.world.get::<&Renderable>(entity) else { continue };
            let transform = (*transform).clone();
            if let Some(ref spatial) = renderable.spatial {
                let world_matrix = glam::Mat4::from_scale_rotation_translation(
                    transform.scale,
                    glam::Quat::from_euler(
                        glam::EulerRot::XYZ,
                        transform.rotation.x.to_radians(),
                        transform.rotation.y.to_radians(),
                        transform.rotation.z.to_radians(),
                    ),
                    transform.position,
                );
                let gpu_idx = self.gpu_objects.len() as u32;
                let spatial_handle = rkp_core::scene_node::SpatialHandle::Octree {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let mut skinning = self.bone_matrix_allocator.binding(entity);
                // Plan the skin-deform scatter for this entity if it's
                // animated AND the asset has baked skinning metadata.
                if self.skinning_enabled {
                    if let (Some(bind), Some(handle)) = (skinning, renderable.asset_handle) {
                        // Cache lookup — no scene_mgr lock here. The
                        // cache is refreshed at the top of this fn
                        // only when geometry epoch advances.
                        if let (Some(skel), Some(skin_data)) = (
                            self.world.get::<&crate::components::Skeleton>(entity).ok(),
                            self.skinning_data_cache.get(&handle),
                        ) {
                            if let Some(plan) = crate::scene_sync::plan_skin_dispatch(
                                bind.bone_buffer_offset,
                                bind.bone_count,
                                &skel.current_pose,
                                skin_data,
                                spatial.voxel_size,
                                &mut running_bone_field_cells,
                                &mut running_bone_field_occ_u32s,
                                if self.dqs_enabled { 1 } else { 0 },
                                bind.bone_dq_offset,
                            ) {
                                // Copy the plan's bone-field geometry
                                // into the SkinnedBinding so the GPU
                                // object carries the same coords the
                                // scatter wrote to. Without this the
                                // march would descend a bone field
                                // sized in one frame and origin'd from
                                // another.
                                skinning = Some(crate::scene_sync::SkinnedBinding {
                                    bone_count: bind.bone_count,
                                    bone_buffer_offset: bind.bone_buffer_offset,
                                    bone_field_offset: plan.uniforms.bone_field_offset,
                                    bone_field_dims: [
                                        plan.uniforms.bone_field_dim_x,
                                        plan.uniforms.bone_field_dim_y,
                                        plan.uniforms.bone_field_dim_z,
                                    ],
                                    bone_field_origin: [
                                        plan.uniforms.grid_origin_x,
                                        plan.uniforms.grid_origin_y,
                                        plan.uniforms.grid_origin_z,
                                    ],
                                    bone_field_occ_offset: plan.uniforms.bone_field_occ_offset,
                                    bone_dq_offset: bind.bone_dq_offset,
                                });
                                self.skin_dispatches.push(plan);
                                // Cache this entity's pose for the
                                // scatter-skip check at the end of the
                                // function. Only records entities that
                                // made it to a plan — a plan bail
                                // below treats this entity as "not
                                // animated this frame", same as last
                                // frame's cache if it was also missing.
                                this_frame_poses.insert(entity, skel.current_pose.clone());
                            } else {
                                // Plan bailed (no extent, or dims > cap).
                                // Leave skinning = None so march falls
                                // back to the rigid path for this
                                // entity.
                                skinning = None;
                            }
                        }
                    }
                }
                let mut gpu_obj = crate::scene_sync::build_gpu_object(
                    &world_matrix,
                    &spatial.aabb,
                    spatial.grid_origin,
                    &spatial_handle,
                    spatial.voxel_size,
                    renderable.material_id,
                    gpu_idx,
                    skinning,
                );
                // Render-layer mask — entity opt-in via RenderLayer
                // component, otherwise the system DEFAULT bit.
                gpu_obj.layer_mask = self
                    .world
                    .get::<&crate::viewport::RenderLayer>(entity)
                    .map(|l| l.mask)
                    .unwrap_or(crate::viewport::layer::DEFAULT);
                self.entity_to_gpu.insert(entity, self.gpu_objects.len());
                self.gpu_to_entity.push(entity);
                self.gpu_objects.push(gpu_obj);
            }
        }

        // Each bone-field cell is a `vec2<u32>` (packed bone indices +
        // weights) = 8 bytes. Used by the render loop to size the
        // scene's bone_field_buffer before the scatter dispatch.
        self.skin_bone_field_bytes = (running_bone_field_cells as u64).saturating_mul(8);
        self.skin_bone_field_occ_bytes = (running_bone_field_occ_u32s as u64).saturating_mul(4);


        // Pause-aware scatter skip: if the set of skinned entities and
        // their per-bone matrices are byte-identical to last frame,
        // the `bone_field` buffer still holds valid data — render loop
        // skips both the clear and the scatter dispatch. Big win when
        // the user pauses the animation to inspect a frame.
        //
        // Empty-to-empty doesn't count as a reuse opportunity: there
        // was nothing to clear last frame either, so the render loop
        // already skips via `skin_dispatches.is_empty()`.
        self.skin_reuse = !this_frame_poses.is_empty()
            && this_frame_poses == self.last_skin_poses;
        self.last_skin_poses = this_frame_poses;

    }

    fn scan_models(&mut self) {
        self.available_models.clear();
        if let Some(ref project_dir) = self.project_dir {
            let assets_dir = project_dir.join("assets");
            if assets_dir.exists() {
                Self::scan_rkp_recursive(&assets_dir, &mut self.available_models);
            }
            self.available_models.sort_by(|a, b| a.name.cmp(&b.name));
            self.models_dirty = true;
            eprintln!("[RkpEngine] scanned {} models", self.available_models.len());
        }
        // Same lifecycle — scan presets alongside models so a project
        // open / model-watcher refresh picks up new `.rkgen` files too.
        self.scan_generator_presets();
    }

    /// Load a `.rkgen` preset and spawn the generator entity it
    /// describes. Param overrides flow through the component
    /// registry's typed `set_field` so partial preset files work —
    /// any field absent from `params` keeps its `Default` value.
    /// Spawn a bare generator (no preset overrides). `pos = None` uses
    /// the click-path default of 3m in front of the camera; `Some(p)`
    /// places the generator's origin at `p` (drop-on-geometry path).
    fn spawn_generator(&mut self, generator_name: &str, pos: Option<glam::Vec3>) {
        let _ = self.spawn_generator_ex(generator_name, pos, true);
    }

    /// Spawn helper that returns the entity so drag-preview can track
    /// it. `verbose` gates the console log — drag-preview spawns are
    /// transient and shouldn't spam the log on every cancel/recreate.
    fn spawn_generator_ex(
        &mut self,
        generator_name: &str,
        pos: Option<glam::Vec3>,
        verbose: bool,
    ) -> Option<hecs::Entity> {
        use crate::components::*;
        let Some(entry) = self.generator_system.registry().get(generator_name) else {
            self.console.error(format!(
                "Unknown generator '{generator_name}' — not registered in gameplay dylib"
            ));
            return None;
        };
        let name = self.unique_name(generator_name);
        let mut transform = Transform::default();
        transform.position = pos.unwrap_or_else(|| {
            self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
        });
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: name.clone() },
            crate::generator::GeneratorState::new(generator_name),
        ));
        (entry.insert_default_params)(&mut self.world, entity);
        self.assign_entity_uuid(entity);
        self.scene_dirty = true;
        if verbose {
            self.console.info(format!("Spawned generator '{name}'"));
        }
        Some(entity)
    }

    fn spawn_generator_preset(&mut self, path: &str, pos: Option<glam::Vec3>) {
        let _ = self.spawn_generator_preset_ex(path, pos, true);
    }

    fn spawn_generator_preset_ex(
        &mut self,
        path: &str,
        pos: Option<glam::Vec3>,
        verbose: bool,
    ) -> Option<hecs::Entity> {
        use crate::components::*;
        let preset_path = std::path::PathBuf::from(path);
        let cfg = match crate::generator::GeneratorAssetConfig::load(&preset_path) {
            Ok(c) => c,
            Err(e) => {
                self.console.error(format!("Load preset failed: {e}"));
                return None;
            }
        };
        let Some(entry) = self.generator_system.registry().get(&cfg.generator) else {
            self.console.error(format!(
                "Preset '{}' targets unknown generator '{}'",
                cfg.name, cfg.generator,
            ));
            return None;
        };
        let display_name = self.unique_name(&cfg.name);
        let mut transform = Transform::default();
        transform.position = pos.unwrap_or_else(|| {
            self.camera.position + glam::Vec3::new(0.0, 0.0, -3.0)
        });
        let entity = self.world.spawn((
            transform,
            EditorMetadata { name: display_name.clone() },
            crate::generator::GeneratorState::new(&cfg.generator),
        ));
        (entry.insert_default_params)(&mut self.world, entity);
        if !cfg.params.is_empty() {
            if let Some(comp_entry) = self.registry.get(entry.param_component_name) {
                for (field_name, value) in &cfg.params {
                    match json_to_field_value(value, field_name, comp_entry) {
                        Ok(fv) => {
                            if let Err(e) = (comp_entry.set_field)(
                                &mut self.world, entity, field_name, fv,
                            ) {
                                self.console.warn(format!(
                                    "Preset '{}': set {field_name} failed: {e}",
                                    display_name,
                                ));
                            }
                        }
                        Err(e) => {
                            self.console.warn(format!(
                                "Preset '{}': skip {field_name}: {e}",
                                display_name,
                            ));
                        }
                    }
                }
            } else {
                self.console.warn(format!(
                    "Preset '{}': param component '{}' not registered",
                    display_name, entry.param_component_name,
                ));
            }
        }
        self.assign_entity_uuid(entity);
        self.scene_dirty = true;
        if verbose {
            self.console.info(format!(
                "Spawned preset '{}' ({}) with {} override(s)",
                display_name, cfg.generator, cfg.params.len(),
            ));
        }
        Some(entity)
    }

    fn scan_generator_presets(&mut self) {
        self.available_generator_presets.clear();
        let Some(ref project_dir) = self.project_dir else { return };
        let presets_dir = project_dir.join("assets/generators");
        if !presets_dir.exists() {
            self.generator_presets_dirty = true;
            return;
        }
        let Ok(entries) = std::fs::read_dir(&presets_dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "rkgen").unwrap_or(false) {
                match crate::generator::GeneratorAssetConfig::load(&path) {
                    Ok(cfg) => {
                        self.available_generator_presets.push(
                            crate::generator::GeneratorPresetInfo {
                                path: path.clone(),
                                display_name: cfg.name,
                                generator_name: cfg.generator,
                            },
                        );
                    }
                    Err(e) => {
                        self.console.warn(format!(
                            "Skipping malformed preset {}: {e}",
                            path.display(),
                        ));
                    }
                }
            }
        }
        self.available_generator_presets
            .sort_by(|a, b| a.display_name.cmp(&b.display_name));
        self.generator_presets_dirty = true;
        eprintln!(
            "[RkpEngine] scanned {} generator presets",
            self.available_generator_presets.len(),
        );
    }

    fn scan_rkp_recursive(dir: &std::path::Path, out: &mut Vec<crate::snapshot::ModelInfo>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_rkp_recursive(&path, out);
            } else if path.extension().map(|e| e == "rkp").unwrap_or(false) {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let rkp_path = path.to_string_lossy().into_owned();

                // Try to find the source mesh file (the .rkp was generated from it).
                // Convention: source.glb → source.rkp, so source = rkp with mesh extension.
                let source_path = Self::find_source_for_rkp(&path);
                let profile = source_path.as_ref().map(|sp| {
                    crate::import_profile::ImportProfile::load_or_default(sp)
                });

                // Display name: profile override → filename stem.
                let name = profile.as_ref()
                    .and_then(|p| p.display_name.clone())
                    .unwrap_or_else(|| {
                        path.file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    });

                // Read just the header to surface the voxel count in
                // the Asset Properties panel. Header is the first
                // bytes of the file — one small seek per asset during
                // the scan, negligible vs the full .rkp load.
                let voxel_count = read_rkp_voxel_count(&path).unwrap_or(0);

                out.push(crate::snapshot::ModelInfo {
                    name,
                    path: rkp_path,
                    source_path: source_path
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    size,
                    voxel_count,
                    import_profile: profile,
                });
            }
        }
    }

    /// Find the source mesh file for a .rkp output.
    /// Convention: bunny.rkp was generated from bunny.glb (or .gltf, .obj, .fbx).
    fn find_source_for_rkp(rkp_path: &std::path::Path) -> Option<std::path::PathBuf> {
        let stem = rkp_path.with_extension("");
        for ext in &["glb", "gltf", "obj", "fbx"] {
            let candidate = stem.with_extension(ext);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }

    /// Apply an editor-computed reorder: set `entity`'s parent and
    /// tree_order. All ordering math is editor-side (it has the visual
    /// tree context); this handler is a thin applier. Validates cycle
    /// (reject dropping inside own subtree) and self-parent.
    fn handle_reorder(
        &mut self,
        entity_id: uuid::Uuid,
        new_parent: Option<uuid::Uuid>,
        new_order: f64,
    ) {
        if Some(entity_id) == new_parent {
            return; // self-parent
        }
        let Some(entity) = self.uuid_to_entity.get(&entity_id).copied() else { return };

        // Cycle check: walk the proposed parent chain upward, rejecting
        // if we pass through `entity`. No-op when parent is root.
        if let Some(pid) = new_parent {
            let mut cursor = self.uuid_to_entity.get(&pid).copied();
            while let Some(c) = cursor {
                if c == entity {
                    return;
                }
                cursor = self
                    .world
                    .get::<&crate::components::Parent>(c)
                    .ok()
                    .and_then(|p| self.uuid_to_entity.get(&p.parent_id).copied());
            }
        }

        self.entity_tree_order.insert(entity, new_order);
        if new_order + 1.0 > self.next_tree_order {
            self.next_tree_order = new_order + 1.0;
        }
        match new_parent {
            Some(pid) => {
                // hecs `insert_one` replaces an existing component.
                let _ = self.world.insert_one(
                    entity,
                    crate::components::Parent { parent_id: pid },
                );
            }
            None => {
                // Only remove if present — removing an absent component
                // returns MissingComponent, which we'd discard anyway.
                if self.world.get::<&crate::components::Parent>(entity).is_ok() {
                    let _ = self.world.remove_one::<crate::components::Parent>(entity);
                }
            }
        }

        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn delete_entity(&mut self, entity: hecs::Entity) {
        // Get name for logging.
        let name = self.world.get::<&crate::components::EditorMetadata>(entity)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());

        // If this is a generator entity, cancel any in-flight run and
        // recursively delete its owned children first. Children hold
        // their own pool allocations that need the standard cleanup
        // path, so we route them back through `delete_entity` rather
        // than raw despawn.
        let owned_children: Vec<hecs::Entity> = if let Some(pu) = self.entity_uuids.get(&entity).copied() {
            self.world
                .query::<&crate::generator::GeneratorOwned>()
                .iter()
                .filter(|(_, owned)| owned.parent_uuid == pu)
                .map(|(e, _)| e)
                .collect()
        } else {
            Vec::new()
        };
        self.generator_system.forget(entity);
        self.pending_generator_slot_keys.remove(&entity);
        for child in owned_children {
            self.delete_entity(child);
        }

        // Release geometry. Asset-backed entities go through the cache so
        // their leaf_attr/brick/octree ranges only free on the last instance
        // release. Procedural entities (no asset_handle) free their own
        // octree + leaf_attr range + brick ids via `deallocate_geometry`.
        if let Ok(renderable) = self.world.get::<&crate::components::Renderable>(entity) {
            if let Some(handle) = renderable.asset_handle {
                drop(renderable);
                self.scene_mgr.lock().unwrap().release_asset(handle);
            } else if let Some(ref spatial) = renderable.spatial {
                let handle = rkp_core::OctreeHandle {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let slot_start = spatial.voxel_slot_start;
                let slot_count = spatial.voxel_slot_count;
                let brick_ids = spatial.brick_ids.clone();
                drop(renderable);
                self.scene_mgr.lock().unwrap().deallocate_geometry(
                    &handle, slot_start, slot_count, &brick_ids,
                );
            }
        }

        // Clear selection if this was selected.
        if self.selected_entity == Some(entity) {
            self.selected_entity = None;
        }

        // Reparent children to root (remove Parent component).
        let entity_uuid = self.entity_uuids.get(&entity).copied();
        if let Some(uuid) = entity_uuid {
            let children: Vec<hecs::Entity> = self.world
                .query::<&crate::components::Parent>()
                .iter()
                .filter(|(_, p)| p.parent_id == uuid)
                .map(|(e, _)| e)
                .collect();
            for child in children {
                let _ = self.world.remove_one::<crate::components::Parent>(child);
            }
        }

        // Remove UUID mappings.
        if let Some(uuid) = self.entity_uuids.remove(&entity) {
            self.uuid_to_entity.remove(&uuid);
        }
        self.entity_tree_order.remove(&entity);

        // Despawn from ECS.
        let _ = self.world.despawn(entity);

        self.console.info(format!("Deleted '{name}'"));
        self.geometry_dirty = true;
        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    /// Duplicate an entity — copies every registered component via the
    /// component registry's serialize/deserialize round-trip, so any
    /// component type the registry knows about (including gameplay
    /// components from the hot-reloaded dylib) is carried across
    /// automatically. Replaces the previous hand-maintained whitelist,
    /// which silently dropped ProceduralGeometry, SpotLight, RigidBody,
    /// Skeleton, AnimationPlayer, and any user-added gameplay component.
    fn duplicate_entity(&mut self, source: hecs::Entity) {
        use crate::components::*;

        // Capture the source's name so we can stamp a unique one on the copy.
        let src_name = self.world.get::<&EditorMetadata>(source)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());
        let new_name = self.unique_name(&src_name);

        // Phase 1 (read-only): serialize every present component into JSON.
        // We materialise an owned (name, json) vec so the registry borrow is
        // dropped before we take &mut world to spawn/insert.
        let pairs: Vec<(String, String)> = {
            let entries = self.registry.components_on(&self.world, source);
            entries
                .iter()
                .filter_map(|e| {
                    (e.serialize)(&self.world, source).map(|json| (e.name.to_string(), json))
                })
                .collect()
        };

        // Phase 2: spawn empty entity, re-insert each serialized component.
        let entity = self.world.spawn(());
        for (name, json) in &pairs {
            if let Some(entry) = self.registry.get(name) {
                if let Err(err) = (entry.deserialize_insert)(&mut self.world, entity, json) {
                    self.console.warn(format!(
                        "duplicate_entity: component '{name}' failed to clone: {err}"
                    ));
                }
            }
        }

        // Phase 3: stamp unique identity. Transform is left exactly
        // as the source — the copy occupies the same world position
        // and only becomes visible once the user moves it (this
        // matches DCC tools like Blender/Maya where Ctrl+D stacks).
        if let Ok(mut md) = self.world.get::<&mut EditorMetadata>(entity) {
            md.name = new_name.clone();
        }
        self.assign_entity_uuid(entity);

        // Phase 4: hydrate runtime-only fields the registry's
        // serialize/deserialize couldn't carry over.
        //
        // `Renderable.spatial` and `Renderable.asset_handle` are
        // `#[serde(skip)]` — they're references into the scene
        // manager's runtime pools and the asset cache, not data we
        // store on disk. After serde-deserialize the cloned entity
        // has `asset_path: Some(...)` but neither a spatial nor a
        // handle, so it would render as empty space. Re-acquire the
        // asset to bump its refcount and populate the runtime
        // fields. Mirrors the load-from-disk path in `load_scene`.
        let asset_path_to_acquire = self
            .world
            .get::<&Renderable>(entity)
            .ok()
            .and_then(|r| r.asset_path.clone());
        if let Some(asset_path) = asset_path_to_acquire {
            let full_path = self
                .project_dir
                .as_ref()
                .map(|d| d.join("assets").join(&asset_path))
                .unwrap_or_else(|| std::path::PathBuf::from(&asset_path));
            let acquired = self
                .scene_mgr
                .lock()
                .unwrap()
                .acquire_asset(&full_path.to_string_lossy());
            match acquired {
                Ok((handle, info)) => {
                    let new_spatial = spatial_from_handle(
                        &info.spatial,
                        info.voxel_size,
                        &info.aabb,
                        info.grid_origin,
                        info.leaf_attr_slot_start,
                        info.leaf_attr_slot_count,
                        Vec::new(),
                    );
                    if let Ok(mut r) = self.world.get::<&mut Renderable>(entity) {
                        r.spatial = Some(new_spatial);
                        r.asset_handle = Some(handle);
                        r.voxel_count = info.voxel_count;
                    }
                }
                Err(e) => {
                    self.console.warn(format!(
                        "Duplicate: couldn't acquire asset '{asset_path}' for clone of '{src_name}': {e}",
                    ));
                }
            }
        }

        // Procedural duplicates: the source's bake-cache file is
        // keyed by the SOURCE entity's UUID, not the new one, so
        // the new entity has no on-disk cache and no runtime
        // spatial yet. Mark it dirty so the bake_worker schedules
        // a fresh bake on the next tick — same treatment a brand-
        // new procedural gets.
        if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
            pg.dirty = true;
        }

        self.selected_entity = Some(entity);

        self.console.info(format!("Duplicated '{src_name}' → '{new_name}'"));
        self.geometry_dirty = true;
        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn clear_scene(&mut self) {
        self.world.clear();
        self.entity_uuids.clear();
        self.uuid_to_entity.clear();
        self.next_entity_uuid = 1;
        self.gpu_objects.clear();
        self.gpu_to_entity.clear();
        // `clear()` wipes every pool but preserves the epoch atomic
        // identity — replacing the whole manager here would orphan
        // sim's `geometry_epoch_handle`, breaking the lock-free
        // epoch read so render never sees future bumps and stops
        // uploading geometry (everything renders as the raw AABB
        // cubes after a project close+open).
        self.scene_mgr.lock().unwrap().clear(1_000_000);
        self.selected_entity = None;
        self.geometry_dirty = true;
        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn load_scene_from_file(&mut self, path: &std::path::Path) {
        // Resolve the scene directory from the passed-in path rather
        // than `self.scene_path` — we used to rely on the latter being
        // set before load, but order-of-operations bugs there silently
        // broke procedural bake-cache restoration. The path we're
        // loading is authoritative.
        let scene_dir = path.parent().map(|p| p.to_path_buf());
        match crate::scene_io::load_scene(path) {
            Ok(scene) => {
                // Restore camera.
                self.camera.position = glam::Vec3::from_array(scene.camera.position);
                self.camera.yaw = scene.camera.yaw;
                self.camera.pitch = scene.camera.pitch;
                self.camera.fov = scene.camera.fov;
                self.sync_main_viewport_from_legacy_camera();

                // Restore environment.
                if let Some(ref env) = scene.environment {
                    self.environment = env.clone();
                    self.environment_dirty = true;
                    self.environment_ui_dirty = true;
                }

                // Load objects as hecs entities.
                // First pass: create entities + map scene UUID → hecs entity.
                use crate::components::*;
                let mut uuid_to_hecs: std::collections::HashMap<uuid::Uuid, hecs::Entity> =
                    std::collections::HashMap::new();

                for obj in &scene.objects {
                    let transform = Transform {
                        position: glam::Vec3::from_array(obj.position),
                        rotation: glam::Vec3::from_array(obj.rotation),
                        scale: glam::Vec3::from_array(obj.scale),
                    };
                    let meta = EditorMetadata { name: obj.name.clone() };

                    let entity = if let Some(ref asset_path) = obj.asset_path {
                        let full_path = self.project_dir.as_ref()
                            .map(|d| d.join("assets").join(asset_path))
                            .unwrap_or_else(|| std::path::PathBuf::from(asset_path));
                        match self.scene_mgr.lock().unwrap().acquire_asset(&full_path.to_string_lossy()) {
                            Ok((handle, info)) => {
                                let spatial = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.grid_origin, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
                                let e = self.world.spawn((transform, meta, Renderable {
                                    asset_path: Some(asset_path.clone()),
                                    material_id: obj.material_id,
                                    voxel_count: info.voxel_count,
                                    spatial: Some(spatial),
                                    asset_handle: Some(handle),
                                    ..Default::default()
                                }));
                                self.geometry_dirty = true;
                                Some(e)
                            }
                            Err(_) => None,
                        }
                    } else if obj.procedural_cache.is_some() {
                        // Two cases land here:
                        //   - Procedurals (`primitive == Some("procedural")`):
                        //     the tree component arrives via the generic
                        //     components pass (ProceduralGeometry). The
                        //     cache provides the pre-baked voxels so
                        //     reload is instant.
                        //   - Persistent generator children (`primitive
                        //     == None`, GeneratorOwned arrives via the
                        //     generic components pass): the cache is
                        //     their only geometry source. Without
                        //     loading it here the child is invisible
                        //     until the generator regens.
                        //
                        // Either way, attach the baked spatial. Missing
                        // or unreadable cache leaves the entity empty —
                        // for procedurals that's recoverable via Bake;
                        // for generator children the parent generator's
                        // next tick will detect the missing slot and
                        // re-emit it.
                        let (spatial, asset_handle, voxel_count) = match (&obj.procedural_cache, &scene_dir) {
                            (Some(rel), Some(dir)) => {
                                let full = dir.join(rel);
                                if full.exists() {
                                    match self.scene_mgr.lock().unwrap().acquire_asset(&full.to_string_lossy()) {
                                        Ok((handle, info)) => {
                                            let sp = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.grid_origin, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
                                            (Some(sp), Some(handle), info.voxel_count)
                                        }
                                        Err(e) => {
                                            self.console.warn(format!(
                                                "Failed to load procedural cache '{rel}' for '{}': {e}",
                                                obj.name,
                                            ));
                                            (None, None, 0)
                                        }
                                    }
                                } else {
                                    self.console.warn(format!(
                                        "Procedural cache '{rel}' referenced by '{}' not found — entity will load unbaked",
                                        obj.name,
                                    ));
                                    (None, None, 0)
                                }
                            }
                            _ => (None, None, 0),
                        };
                        let e = self.world.spawn((transform, meta, Renderable {
                            // Preserve the saved primitive tag —
                            // `Some("procedural")` for un-converted
                            // procedurals (so the inspector still
                            // recognises them and the components pass
                            // attaches the tree); `None` for generator
                            // children (no tree, no procedural
                            // affordances, just baked voxels).
                            primitive: obj.primitive.clone(),
                            material_id: obj.material_id,
                            voxel_count,
                            spatial,
                            asset_handle,
                            ..Default::default()
                        }));
                        self.geometry_dirty = true;
                        Some(e)
                    } else if let Some(ref prim_name) = obj.primitive {
                        let primitive = match prim_name.as_str() {
                            "box" => rkp_core::scene_node::SdfPrimitive::Box {
                                half_extents: glam::Vec3::from_array(obj.scale) * 0.5,
                            },
                            "sphere" => rkp_core::scene_node::SdfPrimitive::Sphere {
                                radius: obj.scale[0] * 0.5,
                            },
                            _ => continue,
                        };
                        // `object_id` is only forwarded to the retired
                        // `pending_faces` emit path; pass 0 to indicate
                        // "no pickable identity" until we either revive
                        // face emission or drop the parameter.
                        self.scene_mgr.lock().unwrap().voxelize_primitive(
                            &primitive, obj.material_id, 0.05, glam::Vec3::ONE, 0,
                        ).map(|result| {
                            let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.grid_origin, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
                            let e = self.world.spawn((transform, meta, Renderable {
                                primitive: Some(prim_name.clone()),
                                material_id: obj.material_id,
                                voxel_count: result.voxel_count,
                                spatial: Some(spatial),
                                ..Default::default()
                            }));
                            self.geometry_dirty = true;
                            e
                        })
                    } else {
                        // Entity with no renderable (e.g. empty transform node).
                        Some(self.world.spawn((transform, meta)))
                    };

                    if let Some(e) = entity {
                        // Keep the UUID from the scene file — freshly
                        // generating a new one would orphan anything
                        // keyed off the ID (bake-cache sidecars, MCP
                        // references, per-entity persisted data).
                        self.set_entity_uuid(e, obj.id);
                        uuid_to_hecs.insert(obj.id, e);
                        // Tree order: prefer the persisted value.
                        // Legacy saves without `tree_order` get a
                        // fresh monotonic key *in file order* — the
                        // file lists objects in tree order, which is
                        // what the user last saw. The alternative
                        // (backfilling later via hecs query iteration)
                        // would reorder in archetype order, which
                        // feels arbitrary to the user.
                        match obj.tree_order {
                            Some(k) => {
                                self.entity_tree_order.insert(e, k);
                            }
                            None => {
                                let k = self.next_tree_order;
                                self.next_tree_order += 1.0;
                                self.entity_tree_order.insert(e, k);
                            }
                        }

                        // Restore PointLight component.
                        if let Some(ref pl) = obj.point_light {
                            let _ = self.world.insert_one(e, PointLight {
                                color: pl.color,
                                intensity: pl.intensity,
                                range: pl.range,
                                cast_shadow: pl.cast_shadow,
                            });
                        }

                        // Restore Camera component.
                        if let Some(ref cam) = obj.camera {
                            let _ = self.world.insert_one(e, Camera {
                                fov: cam.fov,
                                near: cam.near,
                                far: cam.far,
                                active: cam.active,
                            });
                        }
                    }
                }

                // Second pass: restore parent-child relationships.
                for obj in &scene.objects {
                    if let Some(parent_uuid) = obj.parent_id {
                        if let Some(&entity) = uuid_to_hecs.get(&obj.id) {
                            let _ = self.world.insert_one(entity, Parent { parent_id: parent_uuid });
                        }
                    }
                }

                // Third pass: restore generic components via registry.
                // Skeleton is deferred to a fourth pass because it
                // depends on sibling `.rkskel` discovery off the
                // Renderable's asset path, and on `AnimationPlayer`
                // already being in place so `try_attach_skeleton`
                // doesn't overwrite the restored playback state.
                for obj in &scene.objects {
                    if obj.components.is_empty() {
                        continue;
                    }
                    let Some(&entity) = uuid_to_hecs.get(&obj.id) else { continue };
                    for (comp_name, json) in &obj.components {
                        if comp_name == "Skeleton" {
                            continue; // handled in the fourth pass below
                        }
                        if let Some(entry) = self.registry.get(comp_name) {
                            if let Err(e) = (entry.deserialize_insert)(&mut self.world, entity, json) {
                                self.console.warn(format!(
                                    "Failed to restore component '{comp_name}' on '{}': {e}",
                                    obj.name,
                                ));
                            }
                        } else {
                            self.console.warn(format!(
                                "Unknown component '{comp_name}' on '{}' — skipped (gameplay dylib not loaded?)",
                                obj.name,
                            ));
                        }
                    }
                }

                // Fourth pass: re-attach Skeleton (+ bundled
                // AnimationPlayer, preserving the restored-from-disk
                // player state). Uses the same engine-side helper the
                // AddComponent command routes through, so the asset
                // cache + grid-offset derivation stay in one place.
                for obj in &scene.objects {
                    if !obj.components.iter().any(|(n, _)| n == "Skeleton") {
                        continue;
                    }
                    let Some(&entity) = uuid_to_hecs.get(&obj.id) else { continue };
                    let Some(ref asset_path) = obj.asset_path else {
                        self.console.warn(format!(
                            "Restore Skeleton on '{}': no Renderable asset — skipped",
                            obj.name,
                        ));
                        continue;
                    };
                    let full_path = self.project_dir.as_ref()
                        .map(|d| d.join("assets").join(asset_path))
                        .unwrap_or_else(|| std::path::PathBuf::from(asset_path));
                    self.try_attach_skeleton(entity, &full_path);
                }

                // Fifth pass: reconcile ProceduralGeometry.dirty with
                // whether a bake cache actually loaded. Deserialization
                // defaults `dirty = true` to cover legacy scenes with
                // no cache concept; after the cache load we flip that
                // to `false` on entities whose Renderable has a spatial
                // — otherwise the properties panel would mislead the
                // user into thinking a freshly-restored procedural
                // needed rebaking. Entities without a loaded spatial
                // stay dirty so the UI's unbaked chip is accurate.
                let proc_entities: Vec<hecs::Entity> = self
                    .world
                    .query::<(&ProceduralGeometry, &Renderable)>()
                    .iter()
                    .filter(|(_, (_, r))| r.spatial.is_some())
                    .map(|(e, _)| e)
                    .collect();
                for entity in proc_entities {
                    if let Ok(mut pg) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        pg.dirty = false;
                        // Seed `last_evaluated_root_scale` from the
                        // tree's Root so `redirect_transform_scale_to_root`
                        // computes a sane preview multiplier on the
                        // first interaction.
                        let root_id = pg.tree.root();
                        if let Some(root) = pg.tree.get(root_id) {
                            let (s, _, _) = root.transform.to_scale_rotation_translation();
                            pg.last_evaluated_root_scale = s;
                        }
                    }
                }

                // Reseed `next_tree_order` past the max value loaded
                // from the scene file so post-load spawns continue to
                // append at the bottom. Entities missing a persisted
                // `tree_order` already got fresh monotonic keys in
                // file order in the spawn loop above — no second pass
                // here would help, and a hecs-query iteration would
                // actively hurt (archetype order ≠ file order).
                let max_loaded = self
                    .entity_tree_order
                    .values()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_loaded.is_finite() {
                    self.next_tree_order = max_loaded + 1.0;
                }

                self.scene_dirty = true;
                self.gpu_objects_dirty = true;
            }
            Err(e) => self.console.error(format!("Load scene failed: {e}")),
        }
    }

    /// Write the current project descriptor to disk, folding in the
    /// latest editor layout blob. No-op when no project is loaded
    /// (prevents the unnamed-scratch-session case from spraying files).
    fn save_project_file(&self) {
        let (Some(project_path), Some(_)) = (&self.project_path, &self.project_dir) else {
            return;
        };
        let project = crate::project::ProjectFile {
            name: self.project_name.clone(),
            default_scene: "default".to_string(),
            recent_scenes: Vec::new(),
            editor_layout: self.editor_layout_json.clone(),
        };
        if let Err(e) = crate::project::save_project(&project, project_path) {
            eprintln!("[RkpEngine] save project failed: {e}");
        }
    }

    fn build_scene_file(&self) -> crate::scene_io::SceneFile {
        use crate::components::*;
        let mut objects = Vec::new();
        let scene_dir = self
            .scene_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        for (entity, (transform, meta)) in self.world.query::<(&Transform, &EditorMetadata)>().iter() {
            let renderable = self.world.get::<&Renderable>(entity).ok();
            let parent = self.world.get::<&Parent>(entity).ok();
            let point_light = self.world.get::<&PointLight>(entity).ok();
            let camera = self.world.get::<&Camera>(entity).ok();

            // Serialize extra components (gameplay + any non-hardcoded) via registry.
            let hardcoded = ["Transform", "EditorMetadata", "Renderable", "PointLight", "Camera", "Parent"];
            let mut components = std::collections::HashMap::new();
            for entry in self.registry.components_on(&self.world, entity) {
                if hardcoded.contains(&entry.name) {
                    continue;
                }
                if let Some(json) = (entry.serialize)(&self.world, entity) {
                    components.insert(entry.name.to_string(), json);
                }
            }

            // Procedural bake cache reference — points at the .rkp
            // sidecar that holds this entity's pre-baked voxels so
            // load can restore them without re-running anything. Two
            // sources flow through this same field:
            //
            //   1. Procedurals: `procedural_cache_path()` →
            //      `{scene}.bakes/{uuid}.rkp` written by the bake
            //      worker on every procedural bake.
            //   2. Persistent generator children: derived from
            //      `(parent_uuid, slot_key)` →
            //      `{scene}.bakes/gen_{parent}_{slot}.rkp` written by
            //      the bake worker via the `cache_output_path` set on
            //      the BakeRequest by `enqueue_child_bake`.
            //
            // Either way, only emit when the file actually exists. An
            // unsaved scratch scene (no `scene_path`) or a never-baked
            // entity won't have one. Converted procedurals took a
            // different route (`assets/converted/*.rkp` via
            // `asset_path`) so they don't appear here.
            let procedural_cache = {
                let abs = if components.contains_key("ProceduralGeometry") {
                    self.procedural_cache_path(entity)
                } else if let Ok(owned) = self.world.get::<&crate::generator::GeneratorOwned>(entity) {
                    let stem_opt = self
                        .scene_path
                        .as_ref()
                        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()));
                    match (&scene_dir, stem_opt) {
                        (Some(dir), Some(stem)) => {
                            let bakes = dir.join(format!("{stem}.bakes"));
                            Some(crate::generator::child_cache_path(
                                &bakes,
                                owned.parent_uuid,
                                &owned.slot_key,
                            ))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match (abs, &scene_dir) {
                    (Some(abs), Some(dir)) if abs.exists() => abs
                        .strip_prefix(dir)
                        .ok()
                        .map(|rel| rel.to_string_lossy().to_string()),
                    _ => None,
                }
            };

            objects.push(crate::scene_io::SceneObject {
                id: self.get_entity_uuid(entity),
                name: meta.name.clone(),
                position: transform.position.to_array(),
                rotation: transform.rotation.to_array(),
                scale: transform.scale.to_array(),
                tree_order: self.entity_tree_order.get(&entity).copied(),
                parent_id: parent.map(|p| p.parent_id),
                asset_path: renderable.as_ref().and_then(|r| r.asset_path.clone()),
                primitive: renderable.as_ref().and_then(|r| r.primitive.clone()),
                procedural_cache,
                material_id: renderable.map(|r| r.material_id).unwrap_or(0),
                point_light: point_light.map(|l| crate::scene_io::ScenePointLight {
                    color: l.color,
                    intensity: l.intensity,
                    range: l.range,
                    cast_shadow: l.cast_shadow,
                }),
                camera: camera.map(|c| crate::scene_io::SceneCamera {
                    fov: c.fov,
                    near: c.near,
                    far: c.far,
                    active: c.active,
                }),
                components,
            });
        }

        crate::scene_io::SceneFile {
            objects,
            camera: crate::scene_io::CameraState {
                position: self.camera.position.to_array(),
                yaw: self.camera.yaw,
                pitch: self.camera.pitch,
                fov: self.camera.fov,
            },
            lights: Vec::new(),
            environment: Some(self.environment.clone()),
        }
    }

    fn update_gizmo(&mut self) {
        let Some(selected) = self.selected_entity else {
            self.gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
            if self.gizmo.dragging {
                self.gizmo.end_drag();
            }
            return;
        };

        let center = match self.world.get::<&crate::components::Transform>(selected) {
            Ok(t) => t.position,
            Err(_) => return,
        };
        let cam_dist = (center - self.camera.position).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let (ray_o, ray_d) = self.screen_to_ray(self.mouse_pos.x, self.mouse_pos.y);

        let left_pressed = self.input_system.raw_state().is_mouse_button_pressed(rkp_runtime::input::InputMouseButton::Left);

        if self.gizmo.dragging {
            // Update drag.
            match self.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => {
                    let delta = crate::gizmo::compute_translate_delta(&self.gizmo, ray_o, ray_d);
                    let new_pos = self.gizmo.initial_position + delta;
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.position = new_pos;
                        self.gpu_objects_dirty = true;
                    }
                }
                crate::gizmo::GizmoMode::Rotate => {
                    let delta = crate::gizmo::compute_rotate_delta(&self.gizmo, ray_o, ray_d, center);
                    let new_rot = delta * self.gizmo.initial_rotation;
                    // Convert quaternion back to Euler degrees for storage.
                    let (y, x, z) = new_rot.to_euler(glam::EulerRot::YXZ);
                    let euler_deg = glam::Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.rotation = euler_deg;
                        self.gpu_objects_dirty = true;
                    }
                }
                crate::gizmo::GizmoMode::Scale => {
                    let delta = crate::gizmo::compute_scale_delta(&self.gizmo, ray_o, ray_d);
                    let new_scale = self.gizmo.initial_scale * delta;
                    if let Ok(mut t) = self.world.get::<&mut crate::components::Transform>(selected) {
                        t.scale = new_scale;
                        self.gpu_objects_dirty = true;
                    }
                    // Same path as the properties-panel scale slider:
                    // route procedural entities' scale onto Root,
                    // queue a debounced bake, and convert what we just
                    // wrote into a render-time preview multiplier.
                    self.redirect_transform_scale_to_root(selected);
                }
            }

            if !left_pressed {
                // Drag ended this tick. For Scale mode on procedural
                // entities, clear `bake_dirty_at` so the next tick's
                // `pending_settled` check fires immediately instead of
                // waiting out the 150 ms slider debounce — mouse-up is
                // an unambiguous "done" signal that a slider doesn't
                // have, no reason to sit on it.
                if matches!(self.gizmo.mode, crate::gizmo::GizmoMode::Scale) {
                    if let Ok(mut pg) = self
                        .world
                        .get::<&mut crate::components::ProceduralGeometry>(selected)
                    {
                        if pg.pending_bake {
                            pg.bake_dirty_at = None;
                        }
                    }
                }
                self.gizmo.end_drag();
            }
        } else {
            // Update hover.
            self.gizmo.hovered_axis = crate::gizmo::pick_gizmo_axis_for_mode(
                ray_o, ray_d, center, gizmo_size, self.gizmo.mode,
            );

            // Start drag if left mouse is pressed on a gizmo handle.
            if left_pressed && self.gizmo.hovered_axis != crate::gizmo::GizmoAxis::None {
                let start_point = match (self.gizmo.mode, self.gizmo.hovered_axis) {
                    // Rotation: project onto the plane perpendicular to the rotation axis.
                    (crate::gizmo::GizmoMode::Rotate, crate::gizmo::GizmoAxis::X | crate::gizmo::GizmoAxis::Y | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.gizmo.hovered_axis.direction();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, axis_dir).unwrap_or(center)
                    }
                    // Plane handles (XY/XZ/YZ): project onto the constraint plane.
                    (_, crate::gizmo::GizmoAxis::XY | crate::gizmo::GizmoAxis::XZ | crate::gizmo::GizmoAxis::YZ) => {
                        let normal = self.gizmo.hovered_axis.plane_normal();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, normal).unwrap_or(center)
                    }
                    // Single-axis translate / scale: closest point on the axis line.
                    (_, crate::gizmo::GizmoAxis::X | crate::gizmo::GizmoAxis::Y | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.gizmo.hovered_axis.direction();
                        let t = crate::gizmo::ray_axis_closest_point(ray_o, ray_d, center, axis_dir);
                        center + axis_dir * t
                    }
                    _ => {
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, -ray_d).unwrap_or(center)
                    }
                };
                let forward = (center - self.camera.position).normalize();
                let rotation = self.world.get::<&crate::components::Transform>(selected)
                    .map(|t| {
                        let r = t.rotation;
                        glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            r.y.to_radians(), r.x.to_radians(), r.z.to_radians(),
                        )
                    })
                    .unwrap_or(glam::Quat::IDENTITY);
                // For procedural entities the user-visible scale lives
                // on Root.transform (Transform.scale stays ~1 between
                // bakes / momentarily holds the preview multiplier
                // mid-debounce). Drag math is multiplicative against
                // `initial_scale`, so capturing Transform.scale would
                // make the first frame of a drag interpret the object
                // as scale 1 and snap it back to its baseline size.
                let scale = self.world.get::<&crate::components::ProceduralGeometry>(selected)
                    .ok()
                    .and_then(|pg| {
                        let root = pg.tree.root();
                        pg.tree
                            .get(root)
                            .map(|n| n.transform.to_scale_rotation_translation().0)
                    })
                    .or_else(|| {
                        self.world
                            .get::<&crate::components::Transform>(selected)
                            .ok()
                            .map(|t| t.scale)
                    })
                    .unwrap_or(glam::Vec3::ONE);
                self.gizmo.pivot = center;
                self.gizmo.begin_drag(
                    self.gizmo.hovered_axis,
                    start_point,
                    center,
                    rotation,
                    scale,
                    forward,
                );
            }
        }
    }

    /// Emit one line segment per parent→child bone pair for each
    /// animated entity. The skinning palette already encodes
    /// `current_world * inverse_bind`, so premultiplying by the
    /// bone's bind-pose local origin cancels the inverse-bind and
    /// gives us the animated world origin directly. Cheap (one mat4
    /// × vec3 per bone) and stateless — runs in the same pass as
    /// selection/light gizmos.
    fn build_bone_wireframes(&self) -> Vec<rkp_render::LineVertex> {
        use glam::{Mat4, Vec3, Vec4};
        let mut verts = Vec::new();
        let bright = [0.5, 0.9, 1.0, 1.0];
        // Bones are editor chrome for the currently-selected rig.
        // Play mode has no selection (selection is an edit-mode
        // concept), and non-selected entities clutter the viewport
        // when multiple animated characters are on screen — so we
        // draw bones for the selected entity only.
        let Some(selected) = self.selected_entity else { return verts };
        let query_result = self.world.query_one::<(&crate::components::Transform, &crate::components::Skeleton)>(selected);
        let Ok(mut query) = query_result else { return verts };
        let Some((transform, skeleton)) = query.get() else { return verts };
        {
            let color = bright;

            // Entity's root world transform (same one the renderer uses
            // for this entity). Bone origins are in object-local space;
            // multiply by this to lift them into world space.
            let root_world = Mat4::from_scale_rotation_translation(
                transform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    transform.rotation.x.to_radians(),
                    transform.rotation.y.to_radians(),
                    transform.rotation.z.to_radians(),
                ),
                transform.position,
            );

            let bones = &skeleton.asset.skeleton;
            let pose = &skeleton.current_pose;
            let bind_origins = &skeleton.bind_world_origins;
            // Defensive: if evaluate() hasn't run yet (pose is all
            // identity) or the pose is the wrong size, fall back to
            // bind-pose origins so the bones still render at rest.
            let use_pose = pose.len() == bones.bones.len();
            // `current_pose` + `bind_world_origins` are in grid frame
            // (origin at octree corner). Undo the grid offset before
            // handing the position to `root_world`, which expects
            // mesh-frame (origin at object centre).
            let grid_offset = skeleton.grid_offset;

            let animated_origin = |i: usize| -> Vec3 {
                let bind = bind_origins.get(i).copied().unwrap_or(Vec3::ZERO);
                let animated_grid = if use_pose {
                    let p = Vec4::new(bind.x, bind.y, bind.z, 1.0);
                    let v = pose[i] * p;
                    Vec3::new(v.x, v.y, v.z)
                } else {
                    bind
                };
                let animated_mesh = animated_grid - grid_offset;
                root_world.transform_point3(animated_mesh)
            };

            // Parent→child line per bone. Root bones get a tiny
            // crosshair so isolated roots (skeleton with 1 bone, or
            // detached rigs) still render something.
            for (i, &parent) in bones.hierarchy.iter().enumerate() {
                let child = animated_origin(i);
                if parent >= 0 && (parent as usize) < bones.bones.len() {
                    let parent_pos = animated_origin(parent as usize);
                    verts.push(rkp_render::LineVertex {
                        position: parent_pos.to_array(),
                        color,
                    });
                    verts.push(rkp_render::LineVertex {
                        position: child.to_array(),
                        color,
                    });
                } else {
                    // Root bone — little crosshair so it's visible even
                    // with no child.
                    verts.extend(rkp_render::wireframe::crosshair(child, 0.05, color));
                }
            }
        }
        verts
    }

    fn build_gizmo_wireframe(&self) -> Vec<rkp_render::LineVertex> {
        let mut verts = Vec::new();

        // Light gizmos — always visible for all light entities.
        let light_color = [1.0, 0.9, 0.5, 0.5]; // warm yellow, semi-transparent
        let selected_light_color = [1.0, 0.9, 0.5, 1.0]; // bright when selected

        for (entity, (transform, pl)) in self.world.query::<(&crate::components::Transform, &crate::components::PointLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            // Always show crosshair icon.
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkp_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Range sphere only when selected.
            if selected {
                verts.extend(rkp_render::wireframe::point_light_wireframe(
                    transform.position, pl.range, selected_light_color,
                ));
            }
        }

        for (entity, (transform, sl)) in self.world.query::<(&crate::components::Transform, &crate::components::SpotLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkp_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Cone only when selected.
            if selected {
                verts.extend(rkp_render::wireframe::spot_light_wireframe(
                    transform.position, sl.direction, sl.range, sl.outer_angle.to_radians(), selected_light_color,
                ));
            }
        }

        // Physics collider wireframes.
        if self.show_colliders {
            verts.extend(self.build_collider_wireframes());
        }

        // Bone gizmo — one set of line segments per skinned entity,
        // drawn from animated bone origins. Selected entity gets a
        // brighter palette so it pops against a scene with multiple
        // animated characters.
        verts.extend(self.build_bone_wireframes());

        // Drag-preview gizmo for generators — a wireframe AABB sized
        // by the preview's `gizmo_half`, centered on the cached surface
        // hit. The generator itself only spawns on commit; this box is
        // the user's visual anchor while dragging.
        if let Some(preview) = self.drag_preview.as_ref() {
            if let DragPreviewKind::Generator { gizmo_half, .. } = &preview.kind {
                if let Some(center) = preview.last_surface_pos {
                    // Sit the box on the surface so the bottom face is
                    // flush with the drop point — matches how model
                    // previews bottom-snap.
                    let min = glam::Vec3::new(
                        center.x - gizmo_half.x,
                        center.y,
                        center.z - gizmo_half.z,
                    );
                    let max = glam::Vec3::new(
                        center.x + gizmo_half.x,
                        center.y + 2.0 * gizmo_half.y,
                        center.z + gizmo_half.z,
                    );
                    // Soft cyan, semi-transparent — the same palette
                    // the editor uses for "pending" overlays.
                    let color = [0.4, 0.9, 1.0, 0.7];
                    verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                }
            }
        }

        // Transform gizmo — only for the selected entity.
        let Some(selected) = self.selected_entity else {
            return verts;
        };

        let center = match self.world.get::<&crate::components::Transform>(selected) {
            Ok(t) => t.position,
            Err(_) => return verts,
        };

        let cam_dist = (center - self.camera.position).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let gizmo_verts = match self.gizmo.mode {
            crate::gizmo::GizmoMode::Translate => {
                crate::wireframe_builders::translate_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
            crate::gizmo::GizmoMode::Rotate => {
                crate::wireframe_builders::rotate_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
            crate::gizmo::GizmoMode::Scale => {
                crate::wireframe_builders::scale_gizmo_wireframe(
                    center, gizmo_size, self.gizmo.hovered_axis, self.camera.position,
                )
            }
        };
        verts.extend(gizmo_verts);
        verts
    }

    /// BUILD-viewport gizmo: hover + drag for the selected procedural
    /// node's transform. Mirrors `update_gizmo` but reads BUILD mouse
    /// state, casts rays through BUILD's camera, and writes to the
    /// node's Affine3A instead of an entity Transform.
    fn update_procedural_gizmo(&mut self) {
        // Voxel preview mode: the gizmo would edit the tree without
        // any live visual update in the build viewport, so disable
        // it entirely. Clear any in-flight drag and reset hover so a
        // mode flip mid-interaction doesn't leave stale state.
        let raymarch = self.viewports
            .get(crate::viewport::ViewportId::BUILD)
            .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
            .unwrap_or(false);
        if !raymarch {
            self.proc_gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
            if self.proc_gizmo.dragging {
                self.proc_gizmo.end_drag();
            }
            return;
        }

        let (node_id, entity) = match (self.selected_procedural_node, self.selected_entity) {
            (Some(n), Some(e)) => (n, e),
            _ => {
                self.proc_gizmo.hovered_axis = crate::gizmo::GizmoAxis::None;
                if self.proc_gizmo.dragging {
                    self.proc_gizmo.end_drag();
                }
                return;
            }
        };

        // Resolve parent-world and current local transform from the tree.
        let (parent_world, current_local) = {
            let Ok(proc_geo) =
                self.world.get::<&crate::components::ProceduralGeometry>(entity)
            else {
                return;
            };
            let Ok(entity_xform) =
                self.world.get::<&crate::components::Transform>(entity)
            else {
                return;
            };
            let target = rkp_procedural::NodeId(node_id);
            let mut path = Vec::new();
            if !find_path(&proc_geo.tree, proc_geo.tree.root(), target, &mut path) {
                return;
            }
            let entity_world = glam::Affine3A::from_scale_rotation_translation(
                entity_xform.scale,
                glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    entity_xform.rotation.x.to_radians(),
                    entity_xform.rotation.y.to_radians(),
                    entity_xform.rotation.z.to_radians(),
                ),
                entity_xform.position,
            );
            let mut parent_world = entity_world;
            for id in &path[..path.len() - 1] {
                if let Some(n) = proc_geo.tree.get(*id) {
                    parent_world = parent_world * n.transform;
                }
            }
            let current_local = proc_geo
                .tree
                .get(target)
                .map(|n| n.transform)
                .unwrap_or(glam::Affine3A::IDENTITY);
            (parent_world, current_local)
        };

        let world_transform = parent_world * current_local;
        let center = world_transform.transform_point3(glam::Vec3::ZERO);

        let cam_uniforms =
            self.build_camera_uniforms(crate::viewport::ViewportId::BUILD);
        let cam_pos = glam::Vec3::new(
            cam_uniforms.position[0],
            cam_uniforms.position[1],
            cam_uniforms.position[2],
        );
        let cam_dist = (center - cam_pos).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;

        let (ray_o, ray_d) = self.screen_to_ray_for_viewport(
            crate::viewport::ViewportId::BUILD,
            self.build_mouse_pos.x,
            self.build_mouse_pos.y,
        );

        if self.proc_gizmo.dragging {
            // Apply deltas relative to the drag-start SRT, then write
            // back to the node's local transform in parent-relative
            // space. `parent_world.inverse()` handles the conversion
            // back from world deltas.
            let (init_local_t, init_local_r, init_local_s) = self.proc_gizmo_initial_local;
            let parent_inv = self.proc_gizmo_parent_world.inverse();
            let parent_rot = decompose_affine_rotation(&self.proc_gizmo_parent_world);

            let new_local = match self.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => {
                    let world_delta = crate::gizmo::compute_translate_delta(
                        &self.proc_gizmo, ray_o, ray_d,
                    );
                    let new_world_pos = self.proc_gizmo.initial_position + world_delta;
                    let new_local_t = parent_inv.transform_point3(new_world_pos);
                    glam::Affine3A::from_scale_rotation_translation(
                        init_local_s, init_local_r, new_local_t,
                    )
                }
                crate::gizmo::GizmoMode::Rotate => {
                    let world_delta = crate::gizmo::compute_rotate_delta(
                        &self.proc_gizmo, ray_o, ray_d, center,
                    );
                    let new_world_rot = world_delta * self.proc_gizmo.initial_rotation;
                    let new_local_r = parent_rot.inverse() * new_world_rot;
                    glam::Affine3A::from_scale_rotation_translation(
                        init_local_s, new_local_r, init_local_t,
                    )
                }
                crate::gizmo::GizmoMode::Scale => {
                    let delta = crate::gizmo::compute_scale_delta(
                        &self.proc_gizmo, ray_o, ray_d,
                    );
                    let new_local_s = init_local_s * delta;
                    glam::Affine3A::from_scale_rotation_translation(
                        new_local_s, init_local_r, init_local_t,
                    )
                }
            };

            if let Ok(mut proc_geo) =
                self.world.get::<&mut crate::components::ProceduralGeometry>(entity)
            {
                proc_geo.tree.set_transform(
                    rkp_procedural::NodeId(node_id),
                    new_local,
                );
                proc_geo.dirty = true;
            }

            if !self.build_mouse_left {
                self.proc_gizmo.end_drag();
            }
        } else {
            self.proc_gizmo.hovered_axis = crate::gizmo::pick_gizmo_axis_for_mode(
                ray_o, ray_d, center, gizmo_size, self.gizmo.mode,
            );

            if self.build_mouse_left
                && self.proc_gizmo.hovered_axis != crate::gizmo::GizmoAxis::None
            {
                // Capture starting state. Same branching as the entity
                // gizmo — the start point depends on which handle was
                // grabbed so drag math projects from the right origin.
                let start_point = match (self.gizmo.mode, self.proc_gizmo.hovered_axis) {
                    (crate::gizmo::GizmoMode::Rotate,
                     crate::gizmo::GizmoAxis::X
                     | crate::gizmo::GizmoAxis::Y
                     | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.proc_gizmo.hovered_axis.direction();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, axis_dir)
                            .unwrap_or(center)
                    }
                    (_,
                     crate::gizmo::GizmoAxis::XY
                     | crate::gizmo::GizmoAxis::XZ
                     | crate::gizmo::GizmoAxis::YZ) => {
                        let normal = self.proc_gizmo.hovered_axis.plane_normal();
                        crate::gizmo::project_to_plane(ray_o, ray_d, center, normal)
                            .unwrap_or(center)
                    }
                    (_,
                     crate::gizmo::GizmoAxis::X
                     | crate::gizmo::GizmoAxis::Y
                     | crate::gizmo::GizmoAxis::Z) => {
                        let axis_dir = self.proc_gizmo.hovered_axis.direction();
                        let t = crate::gizmo::ray_axis_closest_point(
                            ray_o, ray_d, center, axis_dir,
                        );
                        center + axis_dir * t
                    }
                    _ => crate::gizmo::project_to_plane(ray_o, ray_d, center, -ray_d)
                        .unwrap_or(center),
                };

                // Decompose current LOCAL transform once for later
                // reconstruction during drag.
                let local_t = current_local.translation.into();
                let m = current_local.matrix3;
                let sx = glam::Vec3::from(m.x_axis).length();
                let sy = glam::Vec3::from(m.y_axis).length();
                let sz = glam::Vec3::from(m.z_axis).length();
                let local_s = glam::Vec3::new(sx.max(1e-8), sy.max(1e-8), sz.max(1e-8));
                let rot_mat = glam::Mat3::from_cols(
                    (glam::Vec3::from(m.x_axis) / local_s.x).into(),
                    (glam::Vec3::from(m.y_axis) / local_s.y).into(),
                    (glam::Vec3::from(m.z_axis) / local_s.z).into(),
                );
                let local_r = glam::Quat::from_mat3(&rot_mat);

                let parent_rot = decompose_affine_rotation(&parent_world);
                let world_rot = parent_rot * local_r;
                let forward = (center - cam_pos).normalize();

                self.proc_gizmo_parent_world = parent_world;
                self.proc_gizmo_initial_local = (local_t, local_r, local_s);
                self.proc_gizmo.pivot = center;
                self.proc_gizmo.begin_drag(
                    self.proc_gizmo.hovered_axis,
                    start_point,
                    center,
                    world_rot,
                    local_s,
                    forward,
                );
            }
        }
    }

    /// Wireframe for the procedural-node gizmo drawn on the BUILD viewport.
    ///
    /// Returns an empty vec when:
    /// - no entity selected,
    /// - the selected entity has no `ProceduralGeometry`,
    /// - no procedural node is selected,
    /// - the selected node can't be reached from the root (stale snapshot).
    ///
    /// The gizmo sits at the node's origin in world space — entity world
    /// transform × accumulated parent transforms × the node's own
    /// transform, all applied to (0,0,0). Axes stay world-aligned
    /// (matches the entity gizmo's convention).
    fn build_procedural_gizmo_wireframe(
        &self,
        cam_pos: glam::Vec3,
    ) -> Vec<rkp_render::LineVertex> {
        let node_id = match self.selected_procedural_node {
            Some(id) => id,
            None => return Vec::new(),
        };
        let entity = match self.selected_entity {
            Some(e) => e,
            None => return Vec::new(),
        };

        let Ok(proc_geo) = self
            .world
            .get::<&crate::components::ProceduralGeometry>(entity)
        else {
            return Vec::new();
        };
        let Ok(entity_xform) = self
            .world
            .get::<&crate::components::Transform>(entity)
        else {
            return Vec::new();
        };

        // Walk root → selected node, accumulating parent transforms.
        let tree = &proc_geo.tree;
        let target = rkp_procedural::NodeId(node_id);
        let mut path: Vec<rkp_procedural::NodeId> = Vec::new();
        if !find_path(tree, tree.root(), target, &mut path) {
            return Vec::new();
        }

        // Compose entity world × each transform on the path. Path is
        // root-first and includes the target node, so the last multiply
        // pulls in the target's own local transform — which is what we
        // want: gizmo sits at the node's rotated/scaled/translated origin.
        let entity_world = glam::Affine3A::from_scale_rotation_translation(
            entity_xform.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                entity_xform.rotation.x.to_radians(),
                entity_xform.rotation.y.to_radians(),
                entity_xform.rotation.z.to_radians(),
            ),
            entity_xform.position,
        );
        let mut accum = entity_world;
        for id in &path {
            if let Some(n) = tree.get(*id) {
                accum = accum * n.transform;
            }
        }
        let center = accum.transform_point3(glam::Vec3::ZERO);

        let cam_dist = (center - cam_pos).length().max(0.1);
        let gizmo_size = cam_dist * 0.15;
        // Use proc_gizmo's hover/drag axis so the handle highlights
        // correctly while the user is interacting with BUILD — gizmo
        // mode itself is shared with MAIN's toolbar.
        let hovered = if self.proc_gizmo.dragging {
            self.proc_gizmo.active_axis
        } else {
            self.proc_gizmo.hovered_axis
        };
        match self.gizmo.mode {
            crate::gizmo::GizmoMode::Translate => {
                crate::wireframe_builders::translate_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
            crate::gizmo::GizmoMode::Rotate => {
                crate::wireframe_builders::rotate_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
            crate::gizmo::GizmoMode::Scale => {
                crate::wireframe_builders::scale_gizmo_wireframe(
                    center, gizmo_size, hovered, cam_pos,
                )
            }
        }
    }

    /// Rebuild collider caches for all entities with RigidBody.
    /// Called when geometry changes, RigidBody is added/modified, etc.
    fn rebuild_collider_caches(&mut self) {
        use crate::components::*;

        // Collect entities that need cache rebuild.
        let entities: Vec<(hecs::Entity, RigidBody, Option<SpatialData>, glam::Vec3)> = self.world
            .query::<(&RigidBody, Option<&Renderable>, &Transform)>()
            .iter()
            .map(|(e, (rb, r, t))| {
                (e, rb.clone(), r.and_then(|r| r.spatial.clone()), t.scale)
            })
            .collect();

        let sm_guard = self.scene_mgr.lock().unwrap();
        let all_nodes = sm_guard.octree.data();

        for (entity, rb, spatial, scale) in entities {
            let name = self.world.get::<&EditorMetadata>(entity)
                .map(|m| m.name.clone()).unwrap_or_default();
            let pos = self.world.get::<&Transform>(entity)
                .map(|t| t.position).unwrap_or_default();

            // Derive the fitted-shape bounds from the actually occupied
            // voxels. The padded `SpatialData.aabb` overshoots by ~14 voxels
            // per side (boundary-sampling margin) — fine for the renderer,
            // wrong for Box/Sphere/Capsule sizing.
            let tight_local = spatial.as_ref().and_then(|sp| {
                crate::play_mode::compute_tight_local_aabb(
                    all_nodes,
                    &sm_guard.brick_pool,
                    sp.root_offset as usize,
                    sp.depth,
                    sp.len,
                    sp.base_voxel_size,
                    sp.grid_origin,
                )
            });

            let (aabb_half, local_center) = match tight_local {
                Some(t) => (t.half_extents() * scale, (t.min + t.max) * 0.5 * scale),
                None => (glam::Vec3::splat(0.5), glam::Vec3::ZERO),
            };

            if let Some(ref sp) = spatial {
                eprintln!(
                    "[ColliderCache] '{name}' pos={pos:?} scale={scale:?} \
                     padded_aabb={:?}..{:?} tight_local={tight_local:?} \
                     aabb_half={aabb_half:?} local_center={local_center:?}",
                    sp.aabb.min, sp.aabb.max,
                );
            }

            let (resolved_shape, voxel_coords, voxel_size) = match rb.collider_shape {
                rkp_physics::rigid_body::ColliderShape::Auto => {
                    if let Some(ref sp) = spatial {
                        let (coords, cell_size) = crate::play_mode::build_coarse_collider(
                            all_nodes,
                            &sm_guard.brick_pool,
                            sp.root_offset as usize,
                            sp.depth,
                            sp.len,
                            sp.base_voxel_size,
                            rb.collider_cell_size,
                        );
                        if coords.is_empty() {
                            (rkp_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                        } else {
                            (rkp_physics::rigid_body::ColliderShape::Auto, coords, cell_size)
                        }
                    } else {
                        (rkp_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                    }
                }
                other => (other.clone(), Vec::new(), 0.0),
            };

            let (grid_origin, tree_depth) = match spatial.as_ref() {
                Some(sp) => (sp.grid_origin, sp.depth),
                None => (glam::Vec3::ZERO, 0),
            };

            let cache = ColliderCache {
                shape: resolved_shape,
                voxel_coords,
                collider_cell_size: voxel_size, // actually the coarse cell size from build_coarse_collider
                aabb_half,
                local_center,
                grid_origin,
                tree_depth,
            };

            // Insert or replace the cache component.
            if self.world.get::<&ColliderCache>(entity).is_ok() {
                let _ = self.world.remove_one::<ColliderCache>(entity);
            }
            let _ = self.world.insert_one(entity, cache);
        }
    }

    /// Build wireframe visualization for all physics colliders from cached data.
    fn build_collider_wireframes(&self) -> Vec<rkp_render::LineVertex> {
        use rkp_physics::rigid_body::{BodyType, ColliderShape};
        let mut verts = Vec::new();

        for (_entity, (transform, rb, cache)) in self.world.query::<(
            &crate::components::Transform,
            &crate::components::RigidBody,
            &crate::components::ColliderCache,
        )>().iter() {
            let color = match rb.body_type {
                BodyType::Dynamic => [0.2, 0.8, 0.2, 0.6],
                BodyType::Static => [0.5, 0.5, 0.8, 0.6],
                BodyType::KinematicPosition | BodyType::KinematicVelocity => [0.9, 0.6, 0.2, 0.6],
            };

            // Fitted shapes sit at `transform.position + local_center`, not
            // at `transform.position`, so they line up with off-center bakes.
            let center = transform.position + cache.local_center;
            match cache.shape {
                ColliderShape::Box => {
                    let min = center - cache.aabb_half;
                    let max = center + cache.aabb_half;
                    verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                }
                ColliderShape::Sphere => {
                    let r = cache.aabb_half.max_element();
                    verts.extend(rkp_render::wireframe::sphere_wireframe(center, r, color));
                }
                ColliderShape::Capsule => {
                    let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
                    let hh = (cache.aabb_half.y - r).max(0.01);
                    let top = center + glam::Vec3::new(0.0, hh, 0.0);
                    let bot = center - glam::Vec3::new(0.0, hh, 0.0);
                    verts.extend(rkp_render::wireframe::sphere_wireframe(top, r, color));
                    verts.extend(rkp_render::wireframe::sphere_wireframe(bot, r, color));
                    for angle in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2] {
                        let offset = glam::Vec3::new(angle.cos() * r, 0.0, angle.sin() * r);
                        verts.push(rkp_render::LineVertex { position: (top + offset).to_array(), color });
                        verts.push(rkp_render::LineVertex { position: (bot + offset).to_array(), color });
                    }
                }
                ColliderShape::Auto => {
                    // 24 line vertices per coarse cell; cap so the wireframe
                    // pass never asks for a vertex buffer the GPU can't allocate.
                    // Above this, fall back to the AABB outline.
                    const MAX_WIRE_CELLS: usize = 32_768;
                    if !cache.voxel_coords.is_empty()
                        && cache.voxel_coords.len() <= MAX_WIRE_CELLS
                    {
                        let cs = cache.collider_cell_size;

                        let offset = cache.grid_origin * transform.scale;
                        for coord in &cache.voxel_coords {
                            // Match Rapier: min = coord * cell_size, max = (coord+1) * cell_size,
                            // plus grid_origin offset to align with rendered geometry.
                            let local_min = glam::Vec3::new(
                                coord.x as f32 * cs,
                                coord.y as f32 * cs,
                                coord.z as f32 * cs,
                            );
                            let local_max = glam::Vec3::new(
                                (coord.x + 1) as f32 * cs,
                                (coord.y + 1) as f32 * cs,
                                (coord.z + 1) as f32 * cs,
                            );
                            let world_min = transform.position + offset + local_min * transform.scale;
                            let world_max = transform.position + offset + local_max * transform.scale;
                            verts.extend(rkp_render::wireframe::aabb_wireframe(world_min, world_max, color));
                        }
                    } else {
                        let min = transform.position - cache.aabb_half;
                        let max = transform.position + cache.aabb_half;
                        verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                    }
                }
            }
        }

        verts
    }

    /// Screen-space ray from pixel coordinates.
    /// Phase 4: rays come from MAIN's camera — sculpt/paint/picking are
    /// MAIN-only operations.
    fn screen_to_ray(&self, px: f32, py: f32) -> (glam::Vec3, glam::Vec3) {
        self.screen_to_ray_for_viewport(crate::viewport::ViewportId::MAIN, px, py)
    }

    /// Unproject a pixel position to a world-space ray through the
    /// given viewport's camera. Each viewport has its own camera +
    /// resolution — BUILD's turntable ray lands on the procedural's
    /// gizmo handles, not on MAIN's fly-cam scene.
    fn screen_to_ray_for_viewport(
        &self,
        viewport_id: crate::viewport::ViewportId,
        px: f32,
        py: f32,
    ) -> (glam::Vec3, glam::Vec3) {
        let cam = self.build_camera_uniforms(viewport_id);
        let (vw, vh) = self
            .viewports
            .get(viewport_id)
            .map(|v| (v.width as f32, v.height as f32))
            .unwrap_or((self.width as f32, self.height as f32));

        let vp = glam::Mat4::from_cols_array_2d(&cam.view_proj);
        let inv_vp = vp.inverse();

        let ndc_x = (px / vw) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / vh) * 2.0;

        let near = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
        let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize();
        let origin = glam::Vec3::new(cam.position[0], cam.position[1], cam.position[2]);
        (origin, dir)
    }

    /// CPU raycast against tree-wide ghost primitives at a BUILD-viewport
    /// click pixel. Returns the nearest ghost hit's NodeId (or `None`
    /// if no ghost is on the ray, the viewport isn't in raymarch
    /// mode, or the click isn't on BUILD). The ghost pass renders
    /// depth-free, so "nearest ghost along the ray" matches "ghost
    /// silhouette visible at this pixel."
    fn compute_ghost_pick(
        &self,
        viewport_id: crate::viewport::ViewportId,
        x: u32,
        y: u32,
    ) -> Option<u32> {
        if viewport_id != crate::viewport::ViewportId::BUILD {
            return None;
        }
        let build_vp = self.viewports.get(viewport_id)?;
        if !matches!(build_vp.preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
            return None;
        }

        // Resolve the procedural entity: either the viewport's focus
        // target (the build viewport pins focus to whatever procedural
        // is under edit) or fall back to the editor's global selection.
        let entity = build_vp.filter.focus_entity.or(self.selected_entity)?;

        let proc_geo = self.world
            .get::<&crate::components::ProceduralGeometry>(entity).ok()?;
        let transform = self.world
            .get::<&crate::components::Transform>(entity).ok()?;

        let entity_world = glam::Affine3A::from_scale_rotation_translation(
            transform.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                transform.rotation.x.to_radians(),
                transform.rotation.y.to_radians(),
                transform.rotation.z.to_radians(),
            ),
            transform.position,
        );

        let (ray_o, ray_d) = self
            .screen_to_ray_for_viewport(viewport_id, x as f32, y as f32);

        nearest_ghost_hit(
            &proc_geo.tree, entity_world, ray_o, ray_d, proc_geo.voxel_size,
        )
        .map(|(id, _t)| id)
    }

    // `drain_pick_result` was retired — pick decoding now lives in
    // `process_pick_result`, called from `drain_render_results` when
    // the render thread publishes a `PickResult`. The CPU-resolved
    // ghost hint is stashed in `in_flight_pick_ghost` between submit
    // and result.

    /// Access the profiling ring buffer. Intended for MCP tools and
    /// any other read-only consumer outside the editor snapshot path.
    #[allow(dead_code)]
    pub fn profiling_history(&self) -> &crate::profiling::ProfilingHistory {
        &self.profiling
    }

    fn build_state_update(&mut self, _sim_frame_time: Duration) -> StateUpdate {
        // FPS = render thread's actual iteration rate, EMA-smoothed.
        // The previous formula was `1 / sim_cpu_work_time`, which
        // measured sim CPU headroom rather than what's on screen.
        // After the sim/render thread split they're independent
        // numbers — sim might be at 600 Hz "could do" while render
        // is paced to 60 Hz. The user-visible FPS is the render rate.
        let fps = self.render_hz_ema;

        let objects = if self.scene_dirty {
            self.scene_dirty = false;
            // Sort by `entity_tree_order` — user-arrangeable (via a
            // future drag-reorder command) and persisted in the scene
            // file, so the arrangement survives save/reload. Entities
            // missing from the map (transient edge cases) fall back
            // to `Entity::to_bits()` as a tiebreaker, which preserves
            // spawn order for them.
            //
            // The editor's scene-tree panel groups by `parent_id`
            // after the fact, so children of the same parent end up
            // displayed in their shared TreeOrder order naturally.
            let mut ordered: Vec<hecs::Entity> = self.world
                .query::<&crate::components::EditorMetadata>()
                .iter()
                .map(|(entity, _)| entity)
                .collect();
            ordered.sort_by(|a, b| {
                let ka = self.entity_tree_order.get(a).copied();
                let kb = self.entity_tree_order.get(b).copied();
                match (ka, kb) {
                    (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.to_bits().cmp(&b.to_bits()),
                }
            });

            let mut objs = Vec::with_capacity(ordered.len());
            for entity in ordered {
                let Ok(meta) = self.world.get::<&crate::components::EditorMetadata>(entity) else {
                    continue;
                };
                let is_light = self.world.get::<&crate::components::PointLight>(entity).is_ok()
                    || self.world.get::<&crate::components::SpotLight>(entity).is_ok();
                let is_camera = self.world.get::<&crate::components::Camera>(entity).is_ok();
                let is_procedural = self
                    .world
                    .get::<&crate::components::ProceduralGeometry>(entity)
                    .is_ok();
                let parent_id = self.world.get::<&crate::components::Parent>(entity)
                    .ok()
                    .map(|p| p.parent_id);
                objs.push(crate::snapshot::SceneObjectInfo {
                    id: self.get_entity_uuid(entity),
                    name: meta.name.clone(),
                    parent_id,
                    tree_order: self.entity_tree_order.get(&entity).copied().unwrap_or(0.0),
                    is_camera,
                    is_light,
                    is_procedural,
                });
            }
            Some(objs)
        } else {
            None
        };

        let project = if self.project_dirty {
            self.project_dirty = false;
            Some(self.project_loaded)
        } else {
            None
        };

        let project_name = if project.is_some() {
            Some(self.project_name.clone())
        } else {
            None
        };

        // Ride the same `project_dirty` flag — project_dir changes
        // exactly when project_loaded / project_name do.
        let project_dir = if project.is_some() {
            Some(
                self.project_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned()),
            )
        } else {
            None
        };

        let models = if self.models_dirty {
            self.models_dirty = false;
            Some(self.available_models.clone())
        } else {
            None
        };
        let generators = if self.generators_dirty {
            self.generators_dirty = false;
            let mut names: Vec<String> = self
                .generator_system
                .registry()
                .names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            names.sort();
            Some(names)
        } else {
            None
        };
        let generator_presets = if self.generator_presets_dirty {
            self.generator_presets_dirty = false;
            Some(
                self.available_generator_presets
                    .iter()
                    .map(|p| crate::snapshot::GeneratorPresetEntry {
                        path: p.path.to_string_lossy().into_owned(),
                        display_name: p.display_name.clone(),
                        generator_name: p.generator_name.clone(),
                    })
                    .collect(),
            )
        } else {
            None
        };
        let importing = if self.importing_dirty {
            self.importing_dirty = false;
            Some(self.importing_sources.iter().cloned().collect())
        } else {
            None
        };
        // Send live progress every tick while any import is in flight.
        // Outside an active import this is `None` so the UI skips
        // re-rendering the panel.
        let import_progress = if self.importing_progress.is_empty() {
            None
        } else {
            Some(self.importing_progress.values().cloned().collect())
        };
        let editor_layout = if self.editor_layout_pending {
            self.editor_layout_pending = false;
            Some(self.editor_layout_json.clone())
        } else {
            None
        };

        // Inspector + procedural: send only on change. Both rebuild every
        // tick (cheap) but the panel re-render they trigger on the editor
        // thread is not — sending an identical snapshot 60Hz used to chunk
        // the UI when physics drove a selected RigidBody's Transform.
        let new_inspector = self.build_inspector_snapshot();
        let inspector_update = if new_inspector != self.prev_inspector {
            self.prev_inspector = new_inspector.clone();
            Some(new_inspector)
        } else {
            None
        };
        let new_procedural = self.build_procedural_snapshot();
        let procedural_update = if new_procedural != self.prev_procedural {
            self.prev_procedural = new_procedural.clone();
            Some(new_procedural)
        } else {
            None
        };

        StateUpdate {
            fps,
            tick_hz: self.tick_hz_ema,
            physics_hz: self.physics_hz_ema,
            gpu_object_count: self.gpu_objects.len() as u32,
            camera_position: self.camera.position,
            play_mode: self.play_state.is_some(),
            selected_entity: self.selected_entity.map(|e| self.get_entity_uuid(e)),
            objects,
            project_loaded: project,
            project_name,
            project_dir,
            available_models: models,
            available_generators: generators,
            available_generator_presets: generator_presets,
            importing_models: importing,
            import_progress,
            editor_layout,
            inspector: inspector_update,
            recent_projects: if self.frame_index == 1 {
                Some(crate::recent_projects::load_recent())
            } else {
                None
            },
            available_components: self.selected_entity.map(|entity| {
                self.registry.available_for(&self.world, entity)
                    .iter()
                    .map(|e| e.name.to_string())
                    .collect()
            }),
            materials: if self.material_lib.is_ui_dirty() {
                self.material_lib.clear_ui_dirty();
                Some(self.material_lib.build_info())
            } else {
                None
            },
            selected_material: self.selected_material,
            selected_model: self.selected_model.clone(),
            environment: {
                // Always build, diff-suppress. The old `environment_ui_dirty`
                // gate explicitly avoided echoing slider edits back to the
                // editor (it would have remounted the form mid-drag);
                // diff-suppression makes that hack unnecessary because user
                // edits round-trip back as exact-match no-ops here. With the
                // env panel reading per-field Memos against `store.environment`,
                // these pushes also don't remount anything — only the changed
                // field's DOM updates.
                let _ = self.environment_ui_dirty; // legacy flag, no longer gates
                self.environment_ui_dirty = false;
                if Some(&self.environment) != self.prev_environment.as_ref() {
                    self.prev_environment = Some(self.environment.clone());
                    Some(self.environment.clone())
                } else {
                    None
                }
            },
            procedural: procedural_update,
            console_entries: self.console.drain_new(),
            // Pull the most recent sample whose render-thread data
            // has been stitched in. Render publishes 1-2 frames
            // behind sim — `latest()` would return a still-empty
            // sample sim just pushed and the panel would show no GPU
            // / frame-time data. Falls back to `latest()` during the
            // first few frames before any render results land.
            profiling: self
                .profiling
                .latest_with_render_data()
                .or_else(|| self.profiling.latest())
                .cloned(),
        }
    }

    fn build_procedural_snapshot(&self) -> Option<crate::procedural_snapshot::ProceduralSnapshot> {
        let entity = self.selected_entity?;
        let proc_geo = self.world.get::<&crate::components::ProceduralGeometry>(entity).ok()?;
        let uuid = self.get_entity_uuid(entity);
        let vs = proc_geo.voxel_size;
        // Renderable carries the post-bake voxel count. Procedurals
        // always have one paired with their ProceduralGeometry, but
        // defend with 0 rather than panic if something gets out of
        // sync mid-edit.
        let voxel_count = self
            .world
            .get::<&crate::components::Renderable>(entity)
            .map(|r| r.voxel_count)
            .unwrap_or(0);
        Some(crate::procedural_snapshot::build_procedural_snapshot(
            uuid,
            &proc_geo,
            self.selected_procedural_node,
            vs,
            voxel_count,
        ))
    }
}

// ── Procedural helpers ───────────────────────────────────────────────

/// Compute a safe AABB and voxel size for procedural voxelization.
///
/// Adds margin around the tight bounds and ensures the octree depth won't
/// exceed MAX_DEPTH (11). If the object is too large for the requested voxel
/// size, the voxel size is increased to fit.
/// Walk the procedural tree to find the path from `start` to `target`.
///
/// Writes the sequence of node IDs (including both endpoints) into
/// `out_path` when a match is found and returns `true`. Returns `false`
/// (and leaves `out_path` empty) if `target` isn't reachable from
/// `start` — a possible state if the snapshot a caller is holding has
/// drifted from the current tree.
/// Every leaf NodeId in the tree that plays a "ghost" role — a primitive
/// whose surface can go invisible in the final CSG output, specifically:
///   - non-primary children of a `Subtract` (the cutters) and everything
///     beneath them,
///   - every child of an `Intersect` and everything beneath it,
///   - transitively: once a subtree is in a ghost role, all its leaves
///     inherit the role regardless of further combinators below.
/// Primary children of `Subtract` and all children of `Union` are fully
/// visible in the main raymarch and aren't ghosted.
///
/// Computed tree-wide (not per-selection) so ghosts act like a constant
/// editing aid — you can see every cutter in the scene at all times
/// and click on one to pick it even when it's fully carved away.
fn collect_ghost_primitives(tree: &rkp_procedural::ProceduralObject) -> Vec<u32> {
    let mut out = Vec::new();
    collect_ghosts_rec(tree, tree.root(), false, &mut out);
    out
}

/// CPU sphere-trace against a single procedural primitive's SDF.
///
/// Evaluates one leaf (analytical SDF) against a world-space ray by
/// transforming the ray into the primitive's local frame via the
/// composed ancestor transform, then sphere-tracing up to `MAX_STEPS`
/// iterations. Returns the nearest positive-t hit or `None`.
///
/// Used by the click-to-pick-ghost path. The GPU raymarch shader
/// handles the common case (pixel hits a visible primitive) via
/// `gbuf_pick`; this CPU walk is only for fully-carved cutters whose
/// silhouette has no surface in the G-buffer.
fn raycast_leaf_primitive(
    tree: &rkp_procedural::ProceduralObject,
    leaf_id: rkp_procedural::NodeId,
    ancestor_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> Option<f32> {
    const MAX_STEPS: u32 = 64;
    const MAX_DIST: f32 = 500.0;
    const SURFACE_EPS: f32 = 0.001;

    let node = tree.get(leaf_id)?;
    if !node.kind.is_leaf() { return None; }

    // Local-frame ray. Non-uniform scale on the transform chain would
    // make `ray_d_local` non-unit; for the current editor workflow
    // transforms are uniform-scale-only from the gizmo, so this is
    // fine. If that ever changes, normalize and scale `t` back out.
    let world = ancestor_world * node.transform;
    let inv = world.inverse();
    let ro = inv.transform_point3(ray_origin);
    let rd = inv.transform_vector3(ray_dir);

    let mut t: f32 = 0.0;
    for _ in 0..MAX_STEPS {
        let p = ro + rd * t;
        let d = rkp_procedural::eval_leaf_distance(p, &node.kind);
        if d < SURFACE_EPS { return Some(t); }
        t += d.max(SURFACE_EPS);
        if t > MAX_DIST { return None; }
    }
    None
}

/// Find the closest ghost-role primitive along a world-space ray.
/// Returns `Some((node_id, t))` for the nearest hit, or `None` if no
/// ghost is on the ray. Composed-transform walk mirrors
/// `collect_ghost_primitives` so the same inheritance rules apply.
fn nearest_ghost_hit(
    tree: &rkp_procedural::ProceduralObject,
    entity_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    voxel_size: f32,
) -> Option<(u32, f32)> {
    let mut best: Option<(u32, f32)> = None;
    nearest_ghost_hit_rec(
        tree, tree.root(), false, entity_world,
        ray_origin, ray_dir, voxel_size, &mut best,
    );
    best
}

#[allow(clippy::too_many_arguments)]
fn nearest_ghost_hit_rec(
    tree: &rkp_procedural::ProceduralObject,
    id: rkp_procedural::NodeId,
    is_ghost: bool,
    ancestor_world: glam::Affine3A,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    voxel_size: f32,
    best: &mut Option<(u32, f32)>,
) {
    use rkp_procedural::NodeKind;
    let Some(node) = tree.get(id) else { return };
    let this_world = ancestor_world * node.transform;

    if node.kind.is_leaf() {
        if is_ghost {
            // Raycast in the LEAF's frame — its own transform is
            // already composed into `this_world`, so the caller of
            // raycast_leaf_primitive passes `ancestor_world` = the
            // parent's world (i.e. without leaf.transform) and the
            // function re-composes. To avoid double-applying, use
            // the ancestor_world we were called with, not this_world.
            if let Some(t) = raycast_leaf_primitive(
                tree, id, ancestor_world, ray_origin, ray_dir,
            ) {
                match *best {
                    None => *best = Some((id.0, t)),
                    Some((_, bt)) if t < bt => *best = Some((id.0, t)),
                    _ => {}
                }
            }
        }
        return;
    }
    match &node.kind {
        NodeKind::Union { .. } => {
            for &c in &node.children {
                nearest_ghost_hit_rec(
                    tree, c, is_ghost, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        NodeKind::Intersect { .. } => {
            for &c in &node.children {
                nearest_ghost_hit_rec(
                    tree, c, true, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        NodeKind::Subtract => {
            for (i, &c) in node.children.iter().enumerate() {
                let child_ghost = is_ghost || i > 0;
                nearest_ghost_hit_rec(
                    tree, c, child_ghost, this_world,
                    ray_origin, ray_dir, voxel_size, best,
                );
            }
        }
        _ => {}
    }
}

fn collect_ghosts_rec(
    tree: &rkp_procedural::ProceduralObject,
    id: rkp_procedural::NodeId,
    is_ghost: bool,
    out: &mut Vec<u32>,
) {
    use rkp_procedural::NodeKind;
    let Some(node) = tree.get(id) else { return };
    if node.kind.is_leaf() {
        if is_ghost { out.push(id.0); }
        return;
    }
    match &node.kind {
        NodeKind::Union { .. } => {
            for &c in &node.children {
                collect_ghosts_rec(tree, c, is_ghost, out);
            }
        }
        NodeKind::Intersect { .. } => {
            // All children of an Intersect are "operands that can go
            // invisible where the others aren't." Flip on the ghost
            // flag for every descendant branch.
            for &c in &node.children {
                collect_ghosts_rec(tree, c, true, out);
            }
        }
        NodeKind::Subtract => {
            // First child (minuend) stays whatever its ancestor context
            // made it. Later children (cutters) become ghosts.
            for (i, &c) in node.children.iter().enumerate() {
                let child_ghost = is_ghost || i > 0;
                collect_ghosts_rec(tree, c, child_ghost, out);
            }
        }
        _ => {}
    }
}

fn find_path(
    tree: &rkp_procedural::ProceduralObject,
    start: rkp_procedural::NodeId,
    target: rkp_procedural::NodeId,
    out_path: &mut Vec<rkp_procedural::NodeId>,
) -> bool {
    out_path.push(start);
    if start == target {
        return true;
    }
    if let Some(node) = tree.get(start) {
        for &child in &node.children {
            if find_path(tree, child, target, out_path) {
                return true;
            }
        }
    }
    out_path.pop();
    false
}

/// Extract the rotation component from an `Affine3A` by normalizing
/// the 3×3 matrix's columns to remove per-axis scale. Matches the
/// decomposition used in `procedural_snapshot::decompose_affine`.
fn decompose_affine_rotation(t: &glam::Affine3A) -> glam::Quat {
    let m = t.matrix3;
    let sx = glam::Vec3::from(m.x_axis).length().max(1e-8);
    let sy = glam::Vec3::from(m.y_axis).length().max(1e-8);
    let sz = glam::Vec3::from(m.z_axis).length().max(1e-8);
    let rot_mat = glam::Mat3::from_cols(
        (glam::Vec3::from(m.x_axis) / sx).into(),
        (glam::Vec3::from(m.y_axis) / sy).into(),
        (glam::Vec3::from(m.z_axis) / sz).into(),
    );
    glam::Quat::from_mat3(&rot_mat)
}

fn procedural_voxel_params(tree: &rkp_procedural::ProceduralObject, base_voxel_size: f32) -> (rkp_core::Aabb, f32) {
    let tight = rkp_procedural::compute_bounds(tree);

    // Add margin for boundary sampling (same approach as voxelize_primitive).
    // Grid placement is handled by threading `grid_origin` through to the
    // shader (`local_origin - grid_origin` replaces the old
    // `local_origin + extent/2`), so we can return a tight AABB here
    // without wasting voxel budget on symmetric padding around the origin.
    let margin = base_voxel_size * 8.0 * 1.8 + base_voxel_size;
    let aabb = rkp_core::Aabb {
        min: tight.min - glam::Vec3::splat(margin),
        max: tight.max + glam::Vec3::splat(margin),
    };

    // Ensure depth won't exceed MAX_DEPTH (11). Max voxels per axis = 2^11 = 2048.
    let extent = aabb.max - aabb.min;
    let max_dim = extent.x.max(extent.y).max(extent.z);
    let max_voxels = 2048.0_f32; // 2^11
    let min_voxel_size = max_dim / max_voxels;
    let voxel_size = base_voxel_size.max(min_voxel_size);

    (aabb, voxel_size)
}

/// Parse a node kind name into a `NodeKind`.
fn parse_node_kind(kind: &str) -> rkp_procedural::NodeKind {
    use rkp_procedural::node_kind::*;
    match kind {
        "Sphere" => rkp_procedural::NodeKind::Sphere(SphereParams::default()),
        "Box" => rkp_procedural::NodeKind::Box(BoxParams::default()),
        "Capsule" => rkp_procedural::NodeKind::Capsule(CapsuleParams::default()),
        "Cylinder" => rkp_procedural::NodeKind::Cylinder(CylinderParams::default()),
        "Torus" => rkp_procedural::NodeKind::Torus(TorusParams::default()),
        "Plane" => rkp_procedural::NodeKind::Plane(PlaneParams::default()),
        "Ramp" => rkp_procedural::NodeKind::Ramp(RampParams::default()),
        "Union" => rkp_procedural::NodeKind::Union {
            material_combine: rkp_procedural::MaterialCombine::Winner,
        },
        "Intersect" => rkp_procedural::NodeKind::Intersect {
            material_combine: rkp_procedural::MaterialCombine::Winner,
        },
        "Subtract" => rkp_procedural::NodeKind::Subtract,
        "NoiseDisplace" => {
            rkp_procedural::NodeKind::NoiseDisplace(NoiseDisplaceParams::default())
        }
        "Mirror" => rkp_procedural::NodeKind::Mirror(MirrorParams::default()),
        "MaterialByHeight" => {
            rkp_procedural::NodeKind::MaterialByHeight(MaterialByHeightParams::default())
        }
        "ColorByHeight" => {
            rkp_procedural::NodeKind::ColorByHeight(ColorByHeightParams::default())
        }
        "MaterialByNoise" => {
            rkp_procedural::NodeKind::MaterialByNoise(MaterialByNoiseParams::default())
        }
        "ColorByNoise" => {
            rkp_procedural::NodeKind::ColorByNoise(ColorByNoiseParams::default())
        }
        "Array" => rkp_procedural::NodeKind::Array(ArrayParams::default()),
        _ => rkp_procedural::NodeKind::Sphere(SphereParams::default()),
    }
}

/// Apply a parameter value to a procedural node. Returns true if the param was found.
fn apply_procedural_param(
    tree: &mut rkp_procedural::ProceduralObject,
    id: rkp_procedural::NodeId,
    param_name: &str,
    value: &str,
) -> bool {
    use rkp_procedural::NodeKind;

    let node = match tree.get_mut(id) {
        Some(n) => n,
        None => return false,
    };

    match &mut node.kind {
        // Root has no editable params. Present a row with no fields
        // in the inspector; silently no-op any set attempts.
        NodeKind::Root => false,
        NodeKind::Sphere(p) => match param_name {
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Box(p) => match param_name {
            "half_extents" => { if let Some(v) = parse_vec3(value) { p.half_extents = v; } true }
            "rounding" => { p.rounding = value.parse().unwrap_or(p.rounding); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Capsule(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Cylinder(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Torus(p) => match param_name {
            "major_radius" => { p.major_radius = value.parse().unwrap_or(p.major_radius); true }
            // `tube_radius` is the UI-visible name; `minor_radius` is kept
            // as an alias so the raw field name still works from MCP/scripts.
            "minor_radius" | "tube_radius" => { p.minor_radius = value.parse().unwrap_or(p.minor_radius); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Plane(p) => match param_name {
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Ramp(p) => match param_name {
            "half_length" => { p.half_length = value.parse().unwrap_or(p.half_length); true }
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "half_width" => { p.half_width = value.parse().unwrap_or(p.half_width); true }
            "material_id" | "material" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Union { material_combine } | NodeKind::Intersect { material_combine } => {
            if param_name == "material_combine" {
                *material_combine = match value {
                    "Layered" => rkp_procedural::MaterialCombine::Layered,
                    "Blend" => rkp_procedural::MaterialCombine::Blend { radius: 0.1 },
                    _ => rkp_procedural::MaterialCombine::Winner,
                };
                true
            } else {
                false
            }
        }
        NodeKind::Subtract => false,
        NodeKind::NoiseDisplace(p) => match param_name {
            "amplitude" => { p.amplitude = value.parse().unwrap_or(p.amplitude); true }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            // Octaves + seed come in as floats via the UI's Float
            // scrub control — round to u32 and clamp octaves to the
            // same bound `fbm_3d_vec` enforces so the stored value
            // matches what the evaluator actually uses.
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::Mirror(p) => match param_name {
            "axis" => {
                use rkp_procedural::node_kind::MirrorAxis;
                p.axis = match value {
                    "Y" => MirrorAxis::Y,
                    "Z" => MirrorAxis::Z,
                    _ => MirrorAxis::X,
                };
                true
            }
            _ => false,
        },
        NodeKind::MaterialByHeight(p) => match param_name {
            "low_material" => { p.low_material = value.parse().unwrap_or(p.low_material); true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_material" => { p.mid_material = value.parse().unwrap_or(p.mid_material); true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_material" => { p.high_material = value.parse().unwrap_or(p.high_material); true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            _ => false,
        },
        NodeKind::ColorByHeight(p) => match param_name {
            "low_color" => { if let Some(v) = parse_vec3(value) { p.low_color = v; } true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_color" => { if let Some(v) = parse_vec3(value) { p.mid_color = v; } true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_color" => { if let Some(v) = parse_vec3(value) { p.high_color = v; } true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            _ => false,
        },
        NodeKind::MaterialByNoise(p) => match param_name {
            "low_material" => { p.low_material = value.parse().unwrap_or(p.low_material); true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_material" => { p.mid_material = value.parse().unwrap_or(p.mid_material); true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_material" => { p.high_material = value.parse().unwrap_or(p.high_material); true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::ColorByNoise(p) => match param_name {
            "low_color" => { if let Some(v) = parse_vec3(value) { p.low_color = v; } true }
            "low_to_mid" => { p.low_to_mid = value.parse().unwrap_or(p.low_to_mid); true }
            "mid_color" => { if let Some(v) = parse_vec3(value) { p.mid_color = v; } true }
            "mid_to_high" => { p.mid_to_high = value.parse().unwrap_or(p.mid_to_high); true }
            "high_color" => { if let Some(v) = parse_vec3(value) { p.high_color = v; } true }
            "transition_width" => {
                p.transition_width = value.parse::<f32>().unwrap_or(p.transition_width).max(0.0);
                true
            }
            "frequency" => { p.frequency = value.parse().unwrap_or(p.frequency); true }
            "octaves" => {
                let f: f32 = value.parse().unwrap_or(p.octaves as f32);
                p.octaves = (f.max(0.0) as u32).clamp(1, 8);
                true
            }
            "seed" => {
                let f: f32 = value.parse().unwrap_or(p.seed as f32);
                p.seed = f.max(0.0) as u32;
                true
            }
            _ => false,
        },
        NodeKind::Array(p) => {
            // Counts are per-axis u32s but the UI Float widget hands
            // us strings — round and clamp to ≥ 1 (0 would divide-by-
            // zero in the flatten emit).
            let set_count = |p_slot: &mut u32, v: &str| {
                let f: f32 = v.parse().unwrap_or(*p_slot as f32);
                *p_slot = (f.round().max(1.0) as u32).max(1);
            };
            match param_name {
                "count_x" => { set_count(&mut p.counts[0], value); true }
                "count_y" => { set_count(&mut p.counts[1], value); true }
                "count_z" => { set_count(&mut p.counts[2], value); true }
                "spacing_x" => {
                    p.spacings[0] = value.parse::<f32>().unwrap_or(p.spacings[0]).max(1e-4);
                    true
                }
                "spacing_y" => {
                    p.spacings[1] = value.parse::<f32>().unwrap_or(p.spacings[1]).max(1e-4);
                    true
                }
                "spacing_z" => {
                    p.spacings[2] = value.parse::<f32>().unwrap_or(p.spacings[2]).max(1e-4);
                    true
                }
                _ => false,
            }
        }
    }
}

fn parse_vec3(value: &str) -> Option<glam::Vec3> {
    // Accept "x,y,z" or "[x,y,z]"
    let cleaned = value.trim_matches(|c| c == '[' || c == ']' || c == ' ');
    let parts: Vec<f32> = cleaned.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if parts.len() == 3 {
        Some(glam::Vec3::new(parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

// ── Tick loop ────────────────────────────────────────────────────────

fn tick_loop(
    cmd_rx: Receiver<EngineCommand>,
    frame_callback: FrameCallback,
    state_callback: StateCallback,
    config: EngineConfig,
) {
    // Hand the frame_callback to the render thread (constructed inside
    // EngineState::new). Sim no longer fires it directly — pixel
    // callbacks happen on the render thread after each VR's readback
    // drain. The callback closure is `Send`, so this just transfers
    // ownership across the thread boundary at spawn time.
    let mut state = EngineState::new(&config, frame_callback);
    state.console.info(format!("Engine started ({}x{})", config.width, config.height));

    // Try to load a pre-built gameplay dylib (if project is already set).
    // Normally the dylib is scaffolded + built when a project is opened.
    state.try_load_gameplay_dylib();

    let mut last_tick_start: Option<Instant> = None;

    loop {
        let frame_start = Instant::now();

        // Real wall-clock time since the previous tick started. Cap at 100ms
        // so a one-off hitch (e.g. an asset load) doesn't catapult dynamic
        // bodies; physics will fall behind real time during the hitch and
        // catch up on the next normal tick. First tick uses the target
        // pacing interval since there's no prior tick to measure against.
        let real_dt = match last_tick_start {
            Some(prev) => frame_start.duration_since(prev).as_secs_f32().min(0.1),
            None => 1.0 / 60.0,
        };
        last_tick_start = Some(frame_start);
        let inst_tick_hz = if real_dt > 0.0 { 1.0 / real_dt } else { 0.0 };
        state.tick_hz_ema = state.tick_hz_ema * 0.9 + inst_tick_hz * 0.1;

        // 1. Drain command queue.
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !state.process_command(cmd) {
                eprintln!("[RkpEngine] shutdown");
                return;
            }
        }

        // 1b. Process file watcher events + import events/completions + gameplay reload.
        state.process_file_events();
        state.pump_import_events();
        state.poll_import_completions();
        state.check_gameplay_reload();

        // 1b2. Integrate finished async bakes, then enqueue any new
        // work. Drain first so a bake that just completed gets applied
        // before its entity's `pending_bake` gets re-queued — avoids
        // an otherwise-harmless one-tick stale queue entry.
        state.drain_bake_results();
        state.update_dirty_procedurals();

        // 1b3. Generator system — poll finished generator jobs,
        // detect param edits, submit stale jobs. Lives alongside the
        // bake pump because both flow through the same worker.
        state.tick_generators();

        // 1c. Step gameplay systems + physics if in play mode.
        //
        // Frame order: Update → flush → FixedUpdate → flush → Physics → LateUpdate → flush
        //
        // Gameplay runs before physics so scripts can set transforms on kinematic
        // bodies before physics reads them. Dynamic bodies have their transforms
        // overwritten by physics afterward (physics owns dynamic bodies).
        if state.play_state.is_none() {
            // Decay physics readout when not stepping so a stale 60 Hz doesn't
            // persist in the profiler after Stop.
            state.physics_hz_ema *= 0.9;
        }
        if state.play_state.is_some() {
            let dt = real_dt;
            /// Fixed-step duration for behavior FixedUpdate. Kept in
            /// sync with rkp-physics's default timestep on purpose —
            /// running them at the same rate means a FixedUpdate
            /// system that manipulates a kinematic body sees the
            /// physics world integrated at matching cadence.
            const FIXED_DT: f32 = 1.0 / 60.0;
            /// Cap on fixed steps per render frame. Mirrors physics'
            /// "spiral of death" guard — if we ever fall this far
            /// behind, drop the residual rather than try to catch up.
            const MAX_FIXED_STEPS: u32 = 8;

            state.play_total_time += dt as f64;
            state.play_frame_count += 1;

            // Phase 1: Update — runs once per render frame at real_dt.
            if let Some(ref mut executor) = state.behavior_executor {
                executor.tick_update(
                    &state.gameplay_systems,
                    &mut state.world,
                    &mut state.behavior_commands,
                    &mut state.game_store,
                    dt,
                    state.play_total_time,
                    state.play_frame_count,
                );
                state.gpu_objects_dirty = true;
            }

            // Phase 2: FixedUpdate — accumulator-driven. Runs zero or
            // more times at exactly FIXED_DT each. A 60 Hz render has
            // exactly one step per frame in steady state; a 240 Hz
            // render has a step every ~4 frames; a heavy hitch (say
            // 100 ms) would fire up to MAX_FIXED_STEPS back-to-back
            // and then drop the rest.
            if let Some(ref mut executor) = state.behavior_executor {
                state.behavior_fixed_accumulator += dt;
                let mut steps = 0u32;
                while state.behavior_fixed_accumulator >= FIXED_DT
                    && steps < MAX_FIXED_STEPS
                {
                    executor.tick_fixed_update(
                        &state.gameplay_systems,
                        &mut state.world,
                        &mut state.behavior_commands,
                        &mut state.game_store,
                        FIXED_DT,
                        state.play_total_time,
                        state.play_frame_count,
                    );
                    state.behavior_fixed_accumulator -= FIXED_DT;
                    steps += 1;
                }
                if steps == MAX_FIXED_STEPS {
                    // Spiral-of-death guard: drop residual so we
                    // don't keep trying to catch up next frame.
                    state.behavior_fixed_accumulator = 0.0;
                }
            }

            // Physics step (between FixedUpdate and LateUpdate).
            // Physics has its own Rapier-side accumulator so passing
            // real_dt is correct regardless of render rate.
            if let Some(ref mut play) = state.play_state {
                if play.step(dt, &mut state.world) {
                    state.gpu_objects_dirty = true;
                }
                let substeps = play.last_step_substeps() as f32;
                let inst_hz = if dt > 0.0 { substeps / dt } else { 0.0 };
                state.physics_hz_ema = state.physics_hz_ema * 0.9 + inst_hz * 0.1;
            }

            // Phase 3: LateUpdate — once per render frame at real_dt.
            if let Some(ref mut executor) = state.behavior_executor {
                executor.tick_late(
                    &state.gameplay_systems,
                    &mut state.world,
                    &mut state.behavior_commands,
                    &mut state.game_store,
                    dt,
                    state.play_total_time,
                    state.play_frame_count,
                );
            }

            // Drain viewport requests from behaviors (e.g. set_active_camera).
            // The executor only touches the ECS world; viewport state lives
            // on EngineState so we apply these after the phases complete.
            let requests = state.behavior_commands.take_viewport_requests();
            for req in requests {
                state.apply_viewport_request(req);
            }
        }

        // 1d. Advance skeletal animations on real wall-clock dt so
        // playback rate is independent of render rate. Runs every
        // frame in both edit and play modes so animated characters
        // preview correctly in the editor.
        if crate::animation::tick(&mut state.world, real_dt) {
            state.gpu_objects_dirty = true;
        }

        // 2. Update input system + camera with real_dt — fly mode
        // uses dt to scale velocity, so anything other than real_dt
        // makes the camera move at the wrong speed when the render
        // rate diverges from 60 Hz.
        state.input_system.evaluate();
        state.camera_control.update(
            &state.input_system,
            real_dt,
            &mut state.camera.position,
            &mut state.camera.yaw,
            &mut state.camera.pitch,
        );
        state.sync_main_viewport_from_legacy_camera();

        // 3. Update gizmo hover + drag — MAIN targets the entity
        // transform, BUILD targets the selected procedural node.
        state.update_gizmo();
        state.update_procedural_gizmo();
        // BUILD left-click picking is handled by `process_pick_result`
        // reading the GPU `gbuf_pick` texture (written every frame by
        // `proc_raymarch.wgsl`). Ghost-pick priority is applied there
        // too — see the BUILD+Raymarch branch in process_pick_result.

        // 4. Build the render snapshot and submit to the render thread.
        //    Pixel `frame_callback` fires from the render thread after
        //    each VR's composite readback drains; sim no longer touches
        //    it directly.
        state.submit_render_frame();

        // 6. Push state to client.
        let frame_time = frame_start.elapsed();
        let update = state.build_state_update(frame_time);
        state_callback(&update);

        // 7. Clear per-frame input state for next tick.
        state.input_system.begin_frame();

        // 8. Sim-loop pacing — sleep the remainder of the configured
        // sim target interval. `Uncapped` skips the sleep entirely
        // and lets the sim loop run as fast as its work allows.
        // Render pacing is separate and handled on the render thread.
        if let Some(target) = config.sim_pacing.target_interval() {
            let elapsed = frame_start.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
    }
}

/// Read just the voxel count from a `.rkp` header. Opens the file,
/// parses the header (cheap — header carries `voxel_count` directly
/// near the start), then drops the reader. None on any I/O or format
/// error; callers fall back to 0 (unknown).
fn read_rkp_voxel_count(path: &std::path::Path) -> Option<u32> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let header = rkp_core::asset_file::read_rkp_header(&mut reader).ok()?;
    Some(header.voxel_count)
}
