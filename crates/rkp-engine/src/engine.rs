//! RkpEngine — the self-contained game engine.
//!
//! Owns the tick loop, scene state, renderer, and all GPU resources.
//! Runs on its own thread. Communicates with clients via command channel
//! and shared snapshot.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::Receiver;

use rkp_render::rkp_gpu_object::RkpGpuObject;
use rkp_render::rkp_renderer::RkpRenderer;
use rkp_render::rkp_scene::FrameUpload;
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
    handle: &rkf_core::scene_node::SpatialHandle,
    voxel_size: f32,
    aabb: &rkf_core::Aabb,
    grid_origin: glam::Vec3,
    voxel_slot_start: u32,
    voxel_slot_count: u32,
    brick_ids: Vec<u32>,
) -> SpatialData {
    if let rkf_core::scene_node::SpatialHandle::Octree {
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

/// Configuration for spawning the engine.
pub struct EngineConfig {
    /// Initial render width.
    pub width: u32,
    /// Initial render height.
    pub height: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
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

struct EngineState {
    // GPU
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Rendering pipeline
    renderer: RkpRenderer,
    /// Per-viewport render targets + post-process state. Single MAIN entry
    /// in Phase 2; later phases will key BUILD / PiP / minimap viewports
    /// alongside.
    viewport_renderers: std::collections::HashMap<crate::viewport::ViewportId, rkp_render::ViewportRenderer>,

    // Scene management (CPU). Wrapped in `Arc<Mutex<>>` so the bake
    // worker can run the integrate pass (dealloc-prev + memcpy +
    // remap) against the shared pools directly, without shipping
    // artifacts back to the main thread for a 75+ ms copy. The lock
    // is uncontended on most frames — only `render_frame`'s
    // geometry upload + asset loads touch the scene_mgr, and those
    // finish in a ms or two. See `bake_worker::run_loop` for the
    // worker-side lock scope.
    scene_mgr: std::sync::Arc<std::sync::Mutex<RkpSceneManager>>,

    /// GPU-backed evaluator for procedural trees. Lazy-initialized on
    /// the first procedural bake. All procedural voxelization (spawn,
    /// explicit bake, transform-scale re-bake) flows through this —
    /// the CPU tree evaluator was removed; the GPU is the only path.
    gpu_evaluator: Option<rkp_render::proc_sample::GpuEvaluator>,

    /// Async bake pipeline. The worker owns its own GpuEvaluator and
    /// private pools; the engine sends requests + drains results on
    /// each tick via `drain_bake_results`.
    bake_worker: crate::bake_worker::BakeWorker,

    // Input + Camera
    input_system: rkf_runtime::input::InputSystem,
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
    /// Stable scene object IDs for face emission (entity → scene_obj_id).
    entity_scene_ids: std::collections::HashMap<hecs::Entity, u32>,
    next_scene_id: u32,
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
    /// Maps scene_id (from face emission) → gpu object index (this frame).
    scene_id_to_gpu: std::collections::HashMap<u32, u32>,

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

    // Pick readback (8 bytes for 1 pixel of Rg32Uint material texture)
    pick_readback_buffer: wgpu::Buffer,
    /// Pending pixel-pick: a (viewport, x, y) issued by a click in the
    /// corresponding viewport, awaiting a G-buffer readback. The
    /// viewport tag chooses how to interpret the readback — MAIN
    /// decodes entity scene_id (old path), BUILD decodes per-primitive
    /// NodeId when in raymarch mode (new path, enables
    /// click-to-select-primitive in the build viewport).
    pending_pick: Option<PendingPick>,
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
    /// exists mainly for A/B correctness tests and debugging.
    pub fn set_lod_enabled(&mut self, enabled: bool) {
        self.lod_enabled = enabled;
    }

    /// Current LOD toggle state.
    pub fn lod_enabled(&self) -> bool {
        self.lod_enabled
    }

    /// Flip the Surface-Nets normal reconstruction on or off. When on,
    /// the march computes per-voxel normals from the 3³ in-brick
    /// occupancy neighborhood instead of reading the baked octahedral
    /// `LeafAttr.normal_oct`. POC — see the `[surfnet]` log lines for
    /// coverage metrics.
    pub fn set_surfacenet_enabled(&mut self, enabled: bool) {
        self.surfacenet_enabled = enabled;
    }

    pub fn surfacenet_enabled(&self) -> bool {
        self.surfacenet_enabled
    }

    fn new(config: &EngineConfig) -> Self {
        let ctx = rkf_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let mut renderer = RkpRenderer::new(&device, &queue, width, height);

        // Build the main viewport renderer — owns its full per-resolution
        // pass chain (march/shade/ssao/etc.), gbuffer, bloom chain,
        // tone-map, composite, readback, wireframe-overlay state, plus
        // its own camera buffer + scene bind group.
        let main_viewport_renderer = rkp_render::ViewportRenderer::new(
            &device, &queue, &mut renderer, width, height,
        );
        // Pre-create the BUILD viewport renderer at its default size. It
        // starts invisible (Viewport::new_build) so render_to skips it
        // until the editor enables it via SetViewportVisible. Allocating
        // up-front (~20 MiB) is cheaper than the latency hit of creating
        // it when the user opens the build surface mid-session.
        let build_viewport_renderer = rkp_render::ViewportRenderer::new(
            &device, &queue, &mut renderer, 800, 600,
        );
        let mut viewport_renderers = std::collections::HashMap::new();
        viewport_renderers.insert(crate::viewport::ViewportId::MAIN, main_viewport_renderer);
        viewport_renderers.insert(crate::viewport::ViewportId::BUILD, build_viewport_renderer);

        let scene_mgr = std::sync::Arc::new(std::sync::Mutex::new(
            RkpSceneManager::new(1_000_000),
        ));

        // Input system with default action map.
        let mut input_system = rkf_runtime::input::InputSystem::new();
        input_system.add_map(crate::camera::default_action_map());
        input_system.set_active_map("editor");
        let camera_control = CameraControlState::default();

        // Pick readback buffer — two 1-pixel slots packed at 256-byte
        // aligned offsets: material (Rg32Uint, 8 bytes) at 0..8, pick
        // (R32Uint, 4 bytes) at 256..260. `process_pick_result` reads
        // both to resolve voxel / procedural picks cleanly now that
        // primitive_node_id lives in its own texture.
        let pick_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp pick readback"),
            size: 512,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            bake_worker: crate::bake_worker::BakeWorker::spawn(
                device.clone(),
                queue.clone(),
                scene_mgr.clone(),
            ),
            device,
            queue,
            renderer,
            viewport_renderers,
            scene_mgr,
            gpu_evaluator: None,
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
            entity_scene_ids: std::collections::HashMap::new(),
            next_scene_id: 0,
            selected_entity: None,
            selected_procedural_node: None,
            gpu_objects: Vec::new(),
            gpu_to_entity: Vec::new(),
            entity_to_gpu: std::collections::HashMap::new(),
            scene_id_to_gpu: std::collections::HashMap::new(),
            project_loaded: false,
            project_name: String::new(),
            project_dir: None,
            project_path: None,
            scene_path: None,
            project_dirty: true, // push initial state
            available_models: Vec::new(),
            models_dirty: false,
            importing_sources: std::collections::HashSet::new(),
            importing_progress: std::collections::HashMap::new(),
            importing_dirty: false,
            editor_layout_json: None,
            editor_layout_pending: false,
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
            file_watcher: None,
            import_worker: crate::import_worker::ImportWorker::new(),
            geometry_dirty: false,
            scene_dirty: false,
            gpu_objects_dirty: true,
            frame_index: 0,
            width,
            height,
            gizmo: crate::gizmo::GizmoState::new(),
            proc_gizmo: crate::gizmo::GizmoState::new(),
            build_mouse_pos: glam::Vec2::ZERO,
            build_mouse_left: false,
            proc_gizmo_parent_world: glam::Affine3A::IDENTITY,
            proc_gizmo_initial_local: (glam::Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec3::ONE),
            mouse_pos: glam::Vec2::ZERO,
            pick_readback_buffer,
            pending_pick: None,
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

    /// Render every visible viewport this tick. Once-per-frame work (light
    /// upload, geometry upload, env params) happens once at the top; the
    /// per-viewport block (camera build, screen-aabbs, frame upload, render,
    /// pick/wireframe on MAIN, readback copy) iterates the visible set. One
    /// encoder + one submit covers the whole frame; readback is then mapped
    /// per viewport and delivered via `frame_callback`.
    ///
    /// Phase 4 caveat: only MAIN is visible by default and the renderer's
    /// internal pass resources stay sized to MAIN. A second visible viewport
    /// at a different resolution would need either a per-renderer-resize per
    /// iteration or a pass-internal split — both deferred to Phase 6 since
    /// no UI surfaces a second viewport yet.
    fn render_frame(&mut self, frame_callback: &FrameCallback) {
        use crate::viewport::ViewportId;
        let frame_start = std::time::Instant::now();

        // 0a. Upload material palette if dirty.
        if self.material_lib.is_dirty() {
            let palette = self.material_lib.build_palette();
            self.renderer.update_materials(&self.queue, &palette);
            self.material_lib.clear_dirty();
        }

        // 0b. Upload environment + lights.
        // Always rebuild lights array (entity lights may have moved).
        {
            let mut gpu_lights = vec![self.environment.to_gpu_light()]; // [0] = sun

            // Collect point lights from entities.
            for (_entity, (transform, pl)) in self.world.query::<(&crate::components::Transform, &crate::components::PointLight)>().iter() {
                gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                    position: [transform.position.x, transform.position.y, transform.position.z, 1.0], // w=1 = point
                    color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                    direction: [0.0, 0.0, 0.0, 0.0],
                    params: [pl.range, 0.0, 0.0, if pl.cast_shadow { 1.0 } else { 0.0 }],
                });
            }

            // Collect spot lights from entities.
            for (_entity, (transform, sl)) in self.world.query::<(&crate::components::Transform, &crate::components::SpotLight)>().iter() {
                gpu_lights.push(rkp_render::rkp_shade::GpuLight {
                    position: [transform.position.x, transform.position.y, transform.position.z, 2.0], // w=2 = spot
                    color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                    direction: [sl.direction.x, sl.direction.y, sl.direction.z, sl.outer_angle.to_radians()],
                    params: [sl.range, sl.inner_angle.to_radians(), 0.0, if sl.cast_shadow { 1.0 } else { 0.0 }],
                });
            }

            let mut shade_params = self.environment.to_shade_params();
            shade_params.num_lights = gpu_lights.len() as u32;
            // Stash the base shade_params; the per-viewport loop below
            // writes it (with the per-VR `isolation` flag set) into the
            // shared shade_params buffer just before each VR's submit.
            // Same shared-buffer-per-VR-submit pattern as vol/cloud/atmo.
            self.shade_params_base = shade_params;
            self.renderer.update_lights(&self.queue, &gpu_lights);
            self.num_lights_cache = shade_params.num_lights;

            if self.environment_dirty {
                // Bloom/tonemap params apply to every viewport (each VR
                // owns its own bloom + tonemap pass; no per-viewport
                // overrides today, everybody shares the scene's env).
                // If we skip non-MAIN here, BUILD falls back to default
                // bloom intensity/exposure and the preview looks
                // massively over-bloomed — same shared-state bug class
                // as Phase 6a's camera buffer.
                let env = &self.environment;
                let queue = &self.queue;
                let vr_ids: Vec<_> = self.viewport_renderers.keys().copied().collect();
                for vr_id in vr_ids {
                    let vr = self.viewport_renderers
                        .get_mut(&vr_id)
                        .expect("viewport renderer must exist");
                    vr.tone_map.set_exposure(queue, env.exposure);
                    vr.bloom.set_threshold(queue, env.bloom_threshold, env.bloom_knee);
                    vr.bloom_composite.set_intensity(queue, env.bloom_intensity);
                }
                self.environment_dirty = false;
            }
        }

        // 0c. Rebuild GPU objects from ECS world only when transforms/objects changed.
        if self.gpu_objects_dirty {
            self.update_scene_gpu();
            self.gpu_objects_dirty = false;
        }

        let t_cpu_setup = frame_start.elapsed();

        // 1. Upload geometry + per-frame objects once (queue-side, no
        // encoder). Before any submit so all viewports see the same
        // scene data.
        if self.geometry_dirty {
            let sm = self.scene_mgr.lock().unwrap();
            let geo = sm.geometry_upload();
            self.renderer.upload_geometry(&self.queue, &geo);
            drop(sm);
            self.geometry_dirty = false;
            self.collider_caches_dirty = true;
        }
        if self.collider_caches_dirty {
            self.rebuild_collider_caches();
            self.collider_caches_dirty = false;
        }
        self.renderer.upload_frame(&self.queue, &FrameUpload {
            objects: &self.gpu_objects,
        });

        let t_upload = frame_start.elapsed();

        // ── Per-viewport rendering with one submit per viewport ─────────
        // `queue.write_buffer` is queue-global — only the last write
        // before a submit is visible. Per-frame params (vol/cloud/
        // god_ray/atmo) are a shared buffer that each viewport writes
        // its own values to. Without a per-viewport submit boundary,
        // MAIN's dispatches would read BUILD's params (or vice versa)
        // and both viewports would render wrong. One submit per
        // viewport keeps each VR's `queue.write_buffer` correctly
        // paired with its own encoded dispatches.
        let visible_ids: Vec<ViewportId> = self.viewports
            .iter()
            .filter(|(_, v)| v.visible)
            .map(|(id, _)| *id)
            .collect();

        let object_count = self.gpu_objects.len() as u32;
        let shadow_steps = self.environment.shadow_steps;
        let num_lights = self.num_lights_cache;
        // Gizmo overlay is drawn on MAIN only — selection state is global.
        let gizmo_verts = self.build_gizmo_wireframe();
        let mut pick_issued = false;

        for &viewport_id in &visible_ids {
            let cam_uniforms = self.build_camera_uniforms(viewport_id);
            let (vp_w, vp_h) = self.viewports
                .get(viewport_id)
                .map(|v| (v.width, v.height))
                .expect("viewport must exist");

            // Per-viewport screen-AABBs (camera-dependent).
            let vp_matrix = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            let screen_aabbs = crate::scene_sync::compute_screen_aabbs(
                &self.gpu_objects, &vp_matrix, vp_w as f32, vp_h as f32,
            );
            let screen_aabbs_bytes: &[u8] = bytemuck::cast_slice(&screen_aabbs);

            // Per-viewport camera upload (own buffer per VR).
            {
                let vr = self.viewport_renderers
                    .get(&viewport_id)
                    .expect("viewport renderer must exist");
                vr.upload_camera(&self.queue, &cam_uniforms);
            }

            // Refresh this VR's scene + lights/materials bind groups if
            // the corresponding shared buffers reallocated. No-op when
            // epochs match.
            {
                let vr = self.viewport_renderers
                    .get_mut(&viewport_id)
                    .expect("viewport renderer must exist");
                vr.refresh_bindings(&self.device, &self.renderer);
            }

            // Per-viewport vol/cloud/god-ray params — written directly
            // to this VR's own pass buffers. No shared-state race now.
            let vol_params = self.environment.to_volumetric_params(
                &cam_uniforms, vp_w, vp_h, self.frame_index as u32,
            );
            let cloud_params = self.environment.to_cloud_params(self.frame_index as f32 / 60.0);

            let sun_d = self.environment.sun_direction();
            let atmo_frame = rkp_render::rkp_atmosphere::AtmosphereFrameParams {
                sun_dir: [-sun_d[0], -sun_d[1], -sun_d[2]],
                sun_intensity: self.environment.sun_intensity,
                camera_altitude: self.environment.camera_altitude,
                _pad: [0.0; 3],
                cam_pos: [cam_uniforms.position[0], cam_uniforms.position[1], cam_uniforms.position[2]],
                _pad1b: 0.0,
                cam_forward: [cam_uniforms.forward[0], cam_uniforms.forward[1], cam_uniforms.forward[2]],
                _pad2: 0.0,
                cam_right: [cam_uniforms.right[0], cam_uniforms.right[1], cam_uniforms.right[2]],
                _pad3: 0.0,
                cam_up: [cam_uniforms.up[0], cam_uniforms.up[1], cam_uniforms.up[2]],
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
                }
            };
            {
                let vr = self.viewport_renderers
                    .get_mut(&viewport_id)
                    .expect("viewport renderer must exist");
                vr.volumetric.update_params(&self.queue, &vol_params);
                vr.volumetric.update_cloud_params(&self.queue, &cloud_params);
                vr.god_rays.update_params(&self.queue, &god_ray_params);
            }

            // Per-VR shade params: same scene-wide values, isolation bit
            // set per the viewport's mode. Written into the shared
            // shade_params buffer right before this VR's submit so the
            // shade pass reads this VR's flag, not another's.
            let (vp_mode, vp_preview_mode) = self
                .viewports
                .get(viewport_id)
                .map(|v| (v.mode, v.preview_mode))
                .unwrap_or((
                    rkp_render::RenderMode::InSitu,
                    rkp_render::BuildPreviewMode::Voxel,
                ));
            // The procedural being previewed in raymarch mode is always
            // the currently-selected entity — the same thing the build
            // viewport's focus filter already tracks. Keeps the preview
            // follow selection automatically with no extra UI state.
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
            let isolation = matches!(vp_mode, rkp_render::RenderMode::Isolation);
            {
                let mut sp = self.shade_params_base;
                sp.isolation = isolation as u32;
                // The lights buffer is scene-wide (one buffer, shared by
                // every VR), and the shade shader walks the first
                // `num_lights` entries regardless of `isolation`. Entry 0
                // is the sun; 1..N are scene point/spot lights. In
                // isolation we want a neutral studio, so clamp to just
                // the sun — otherwise the main scene's point lights
                // light up the build-viewport preview, which visually
                // contradicts "isolation."
                if isolation {
                    sp.num_lights = sp.num_lights.min(1);
                }
                self.renderer.update_shade_params(&self.queue, &sp);
            }
            // Per-VR bloom intensity. In isolation we run bloom_composite
            // as a passthrough (intensity = 0) since the bloom mips are
            // not refreshed; in-situ uses the env-configured intensity.
            {
                let vr = self.viewport_renderers
                    .get(&viewport_id)
                    .expect("viewport renderer must exist");
                let intensity = if isolation { 0.0 } else { self.environment.bloom_intensity };
                vr.bloom_composite.set_intensity(&self.queue, intensity);
            }

            // Compute the procedural-node gizmo wireframe before we
            // take the mutable VR borrow below — build_procedural_
            // gizmo_wireframe reads from `self.world`, which shares
            // scope with `viewport_renderers`.
            // The procedural-node gizmo is only meaningful when the
            // viewport shows the live tree (raymarch). In voxel mode
            // the user sees the baked result and any gizmo drag would
            // silently edit the tree without visual feedback — drawing
            // the gizmo there invites the user to interact with
            // something that does nothing they can see until they
            // re-bake, which is worse than not showing it at all.
            let proc_gizmo_verts = if viewport_id == ViewportId::BUILD
                && matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch)
            {
                let build_cam_pos = glam::Vec3::new(
                    cam_uniforms.position[0],
                    cam_uniforms.position[1],
                    cam_uniforms.position[2],
                );
                self.build_procedural_gizmo_wireframe(build_cam_pos)
            } else {
                Vec::new()
            };

            // Per-viewport encoder.
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rkp viewport"),
            });

            // When this viewport is in procedural raymarch preview mode,
            // flatten the selected procedural's tree into an RPN stream
            // and push it to the VR's raymarch pass before dispatch. We
            // flatten every frame the mode is active — the cost is O(tree
            // nodes), microseconds, and sidesteps having to track dirty
            // state through another layer. Skip entirely when the mode is
            // Voxel so we don't touch the raymarch pass's buffers.
            // Capture the tree's entity-local AABB alongside the
            // flattened instruction stream — the raymarch shader
            // uses it to reject rays that can't possibly hit before
            // paying the sphere-trace loop.
            let (proc_instructions, proc_aabb): (
                Vec<rkp_procedural::ProcInstruction>,
                Option<rkf_core::Aabb>,
            ) = if matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
                vp_preview_entity
                    .and_then(|uuid| {
                        self.entity_uuids
                            .iter()
                            .find_map(|(e, u)| (*u == uuid).then_some(*e))
                    })
                    .and_then(|entity| {
                        self.world
                            .get::<&crate::components::ProceduralGeometry>(entity)
                            .ok()
                            .map(|pg| {
                                (
                                    rkp_procedural::flatten_tree(&pg.tree),
                                    Some(rkp_procedural::compute_bounds(&pg.tree)),
                                )
                            })
                    })
                    .unwrap_or_default()
            } else {
                (Vec::new(), None)
            };
            let proc_object_id = vp_preview_entity
                .and_then(|uuid| {
                    self.entity_uuids
                        .iter()
                        .find_map(|(e, u)| (*u == uuid).then_some(*e))
                })
                .and_then(|entity| self.entity_scene_ids.get(&entity).copied())
                .unwrap_or(0);

            // World transform of the previewed procedural entity. The
            // raymarch shader uses this to pin the tree at the entity's
            // world position (matching what the voxel path does via the
            // octree's per-object inverse_world); without it, moving the
            // entity with the MAIN gizmo shifts the BUILD preview off
            // the camera.
            let proc_entity_world = vp_preview_entity
                .and_then(|uuid| {
                    self.entity_uuids
                        .iter()
                        .find_map(|(e, u)| (*u == uuid).then_some(*e))
                })
                .and_then(|entity| {
                    self.world
                        .get::<&crate::components::Transform>(entity)
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

            let vr = self.viewport_renderers
                .get_mut(&viewport_id)
                .expect("viewport renderer must exist");

            // Pin the BUILD grid under the previewed entity. Without
            // this, moving the entity in world-space Y (or X/Z, though
            // that was less obvious thanks to the camera-follow) leaves
            // the grid at world y=0 while the camera orbits around the
            // entity's actual position — the object ends up floating
            // relative to the grid. The grid plane following the entity
            // keeps the "studio floor" always under the object. MAIN
            // stays at its default (world origin) so scene-wide layout
            // stays legible there.
            if viewport_id == crate::viewport::ViewportId::BUILD {
                let p = proc_entity_world.translation;
                let grid_params = rkp_render::rkp_grid::GridParams {
                    plane_origin: [p.x, p.y, p.z, 0.0],
                    ..Default::default()
                };
                vr.grid.update_params(&self.queue, &grid_params);
            }

            if matches!(vp_preview_mode, rkp_render::BuildPreviewMode::Raymarch) {
                // Log on transitions / interesting changes so we can
                // tell from stderr whether the pass is wired up
                // without flooding on every frame. Key = (viewport,
                // instruction_count, entity presence).
                static LAST_LOG: std::sync::Mutex<Option<(u32, bool)>> =
                    std::sync::Mutex::new(None);
                let key = (
                    proc_instructions.len() as u32,
                    vp_preview_entity.is_some(),
                );
                let mut guard = LAST_LOG.lock().unwrap();
                if guard.as_ref() != Some(&key) {
                    *guard = Some(key);
                    eprintln!(
                        "[preview] raymarch dispatch viewport={:?} instructions={} entity_present={}",
                        viewport_id,
                        proc_instructions.len(),
                        vp_preview_entity.is_some(),
                    );
                }
                vr.proc_raymarch.upload_instructions(&self.device, &self.queue, &proc_instructions);
                // Empty-AABB sentinel: -1..+1 degenerate box that
                // any sane ray-AABB slab test fails. Covers the
                // "raymarch pass enabled but no procedural entity
                // is selected" transient — avoids a bogus hit.
                let (aabb_min, aabb_max) = match proc_aabb {
                    Some(a) => (a.min, a.max),
                    None => (glam::Vec3::splat(1.0), glam::Vec3::splat(-1.0)),
                };
                vr.proc_raymarch.set_params(
                    &self.queue,
                    proc_instructions.len() as u32,
                    proc_object_id + 1,
                    proc_entity_world,
                    aabb_min,
                    aabb_max,
                );
                // Push the currently-selected procedural NodeId to the
                // outline overlay. Sentinel (`u32::MAX`) when nothing
                // is selected — the shader early-discards on that and
                // the pass becomes free.
                let outline_params = match self.selected_procedural_node {
                    Some(n) => rkp_render::proc_outline::OutlineParams::new(
                        n,
                        // Warm orange highlight, fully opaque. Alpha
                        // is shader-premultiplied into the emitted
                        // color so the pipeline's One/OneMinusSrcAlpha
                        // blend gives the right "over" composite.
                        [1.0, 0.55, 0.15, 1.0],
                    ),
                    None => rkp_render::proc_outline::OutlineParams::NONE,
                };
                vr.proc_outline.update_params(&self.queue, &outline_params);

                // Ghost overlay: every cutter-role primitive in the
                // tree, regardless of selection. Filters the already-
                // flattened instruction stream so ghost renders use the
                // same composed transforms the main raymarch does.
                // Ghost pass early-outs on zero-length upload.
                let ghost_ids = vp_preview_entity.and_then(|uuid| {
                    let entity = self.entity_uuids.iter().find_map(
                        |(e, u)| (*u == uuid).then_some(*e),
                    )?;
                    let proc_geo = self.world
                        .get::<&crate::components::ProceduralGeometry>(entity).ok()?;
                    Some(collect_ghost_primitives(&proc_geo.tree))
                }).unwrap_or_default();
                let ghost_set: std::collections::HashSet<u32> = ghost_ids.into_iter().collect();
                let ghost_instructions: Vec<rkp_procedural::ProcInstruction> =
                    proc_instructions.iter()
                        .filter(|ins| ghost_set.contains(&ins.node_id))
                        .copied()
                        .collect();
                vr.proc_ghost.upload_instructions(
                    &self.device, &self.queue, &ghost_instructions,
                );
                vr.proc_ghost.update_params(
                    &self.queue,
                    &rkp_render::proc_ghost::GhostParams::new(
                        ghost_instructions.len() as u32,
                        // Cool translucent cyan — distinct from the
                        // outline's warm orange so combined visuals
                        // don't muddle when a ghost is also selected.
                        [0.25, 0.7, 1.0, 0.35],
                    ),
                );
            }
            self.renderer.render_to(
                &mut encoder, &self.queue, vr,
                object_count, shadow_steps, num_lights,
                self.lod_enabled, self.surfacenet_enabled,
                screen_aabbs_bytes, &atmo_frame,
                vp_mode,
                vp_preview_mode,
            );

            // Issue the G-buffer readback from whichever viewport the
            // pending pick targets. Only one pick is in flight at a time;
            // the viewport tag + raymarch/voxel mode drives decoding
            // inside `process_pick_result`.
            //
            // Two copies into the same 256-byte aligned readback buffer:
            //   offset 0..8   : gbuf_material pixel (packed_r + packed_g)
            //   offset 128..132: gbuf_pick pixel (primitive_node_id)
            // The second offset is 128 because wgpu requires each copy's
            // destination to be 256-byte aligned and bytes_per_row ≥ 256,
            // which pads each 1×1 region to 256 bytes. Using a second
            // buffer would also work; packing into one is simpler.
            if let Some(pp) = self.pending_pick {
                if pp.viewport == viewport_id && pp.x < vr.width && pp.y < vr.height {
                    pick_issued = true;
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &vr.gbuffer.material_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &self.pick_readback_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(256),
                                rows_per_image: Some(1),
                            },
                        },
                        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    );
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &vr.pick_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: &self.pick_readback_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 256,
                                bytes_per_row: Some(256),
                                rows_per_image: Some(1),
                            },
                        },
                        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    );
                }
            }
            if viewport_id == ViewportId::MAIN {
                let show_editor_overlays = self.viewports
                    .get(ViewportId::MAIN)
                    .map(|v| v.filter.base_layers & crate::viewport::layer::EDITOR_ONLY != 0)
                    .unwrap_or(false);
                if show_editor_overlays && !gizmo_verts.is_empty() {
                    let composite_view = &vr.composite_view;
                    let vw = vr.width as f32;
                    let vh = vr.height as f32;
                    vr.wireframe_pass.draw(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        composite_view,
                        vp_matrix,
                        (0.0, 0.0, vw, vh),
                        &gizmo_verts,
                    );
                }
            }

            // BUILD viewport: procedural-node gizmo overlay. Separate
            // wireframe from the entity gizmo on MAIN — same wireframe
            // pass, different source geometry (the selected procedural
            // node's transform, not the entity's).
            if viewport_id == ViewportId::BUILD && !proc_gizmo_verts.is_empty() {
                let composite_view = &vr.composite_view;
                let vw = vr.width as f32;
                let vh = vr.height as f32;
                vr.wireframe_pass.draw(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    composite_view,
                    vp_matrix,
                    (0.0, 0.0, vw, vh),
                    &proc_gizmo_verts,
                );
            }

            vr.copy_composite_to_readback(&mut encoder);
            self.renderer.resolve_profiler_queries(&mut encoder);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let t_encode = frame_start.elapsed();
        let t_post = t_encode;
        let t_submit = t_encode;

        // Pick result process — depends on this frame's pick copy completing.
        if pick_issued {
            self.process_pick_result();
        }

        // ── Per-viewport readback + delivery ────────────────────────────
        for &viewport_id in &visible_ids {
            let (read_index, w, h) = {
                let vr = self.viewport_renderers
                    .get(&viewport_id)
                    .expect("viewport renderer must exist");
                let read_index = if vr.readback_ready {
                    1 - vr.readback_index
                } else {
                    // First frame: no previous data, read current (blocking).
                    vr.readback_index
                };
                (read_index, vr.width, vr.height)
            };
            let pixels = self.map_readback(viewport_id, read_index);
            frame_callback(viewport_id, &pixels, w, h);
            self.viewport_renderers
                .get_mut(&viewport_id)
                .expect("viewport renderer must exist")
                .advance_readback();
        }

        // GPU profiler — process finished frames (logs every 60 frames).
        self.renderer.end_profiler_frame(self.frame_index, self.width, self.height);

        let t_frame_end = frame_start.elapsed();

        if self.frame_index % 60 == 0 && self.frame_index > 0 {
            eprintln!(
                "[perf] cpu_setup={:.1}ms upload={:.1}ms encode={:.1}ms post={:.1}ms submit={:.1}ms gpu_wait={:.1}ms total={:.1}ms",
                t_cpu_setup.as_secs_f64() * 1000.0,
                (t_upload - t_cpu_setup).as_secs_f64() * 1000.0,
                (t_encode - t_upload).as_secs_f64() * 1000.0,
                (t_post - t_encode).as_secs_f64() * 1000.0,
                (t_submit - t_post).as_secs_f64() * 1000.0,
                (t_frame_end - t_submit).as_secs_f64() * 1000.0,
                t_frame_end.as_secs_f64() * 1000.0,
            );
        }

        self.frame_index += 1;
    }

    /// Read from a viewport's readback buffer. With double-buffering we
    /// read the previous frame's buffer, so the GPU work is already
    /// complete and `wait_indefinitely` returns near-instantly.
    fn map_readback(&self, viewport_id: crate::viewport::ViewportId, index: usize) -> Vec<u8> {
        let vr = self.viewport_renderers
            .get(&viewport_id)
            .expect("viewport renderer must exist");
        let w = vr.width;
        let h = vr.height;
        let padded_row = vr.readback_padded_row();

        let buffer_slice = vr.readback_buffers[index].slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let mut rgba8 = vec![0u8; (w * h * 4) as usize];
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..h as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * w as usize * 4;
                let row_bytes = w as usize * 4;
                if src_offset + row_bytes <= data.len()
                    && dst_offset + row_bytes <= rgba8.len()
                {
                    rgba8[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&data[src_offset..src_offset + row_bytes]);
                }
            }
            drop(data);
            vr.readback_buffers[index].unmap();
        }

        rgba8
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
            prev_vp: view_proj.to_cols_array_2d(),
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
                if let Some(vr) = self.viewport_renderers.get_mut(&id) {
                    vr.resize(&self.device, &mut self.renderer, width, height);
                }
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
                let proc_geo = match leaf_kind {
                    Some(kind) => ProceduralGeometry::with_leaf(parse_node_kind(&kind)),
                    None => ProceduralGeometry::default_sphere(),
                };
                // Spawn the entity with no spatial yet; `dirty = true`
                // on the fresh `ProceduralGeometry` is what makes
                // `update_dirty_procedurals` enqueue an initial bake
                // next tick. Keeps spawn + edit flows on the same
                // async code path.
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;
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
                self.entity_scene_ids.insert(entity, scene_id);
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
                let name = self
                    .world
                    .get::<&EditorMetadata>(entity)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{entity:?}"));
                let voxel_count = self
                    .world
                    .get::<&Renderable>(entity)
                    .map(|r| r.voxel_count)
                    .unwrap_or(0);
                // Drop the tree. Renderable.spatial keeps pointing at
                // the same scene-pool allocation, so rendering
                // continues unchanged — the entity just loses its
                // editing affordances. Clear the "procedural" tag so
                // save/load + inspector treat it as a plain voxel.
                let _ = self.world.remove_one::<ProceduralGeometry>(entity);
                if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
                    renderable.primitive = None;
                }
                if self.selected_entity == Some(entity) {
                    self.selected_procedural_node = None;
                }
                self.scene_dirty = true;
                self.console.info(format!(
                    "Converted '{name}' to voxel object ({voxel_count} voxels).",
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
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;
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
                self.entity_scene_ids.insert(new_entity, scene_id);
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
                    scene_id,
                    instructions,
                    aabb,
                    voxel_size,
                    root_scale: src_scale_for_bake,
                    prev_spatial: None,
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

            EngineCommand::LoadAsset { path, .. } => {
                use crate::components::*;
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;
                let acquired = self.scene_mgr.lock().unwrap().acquire_asset(&path);
                match acquired {
                    Ok((handle, info)) => {
                        let raw_name = Self::display_name_from_path(&path);
                        let name = self.unique_name(&raw_name);
                        // Asset-backed entity — brick_ids stays empty.
                        // Asset cache owns the shared brick range and frees
                        // it on the final release_asset.
                        let spatial = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.grid_origin, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
                        let entity = self.world.spawn((
                            Transform::default(),
                            EditorMetadata { name: name.clone() },
                            Renderable {
                                asset_path: Some(path.clone()),
                                voxel_count: info.voxel_count,
                                spatial: Some(spatial),
                                asset_handle: Some(handle),
                                ..Default::default()
                            },
                        ));
                        self.assign_entity_uuid(entity);
                        self.entity_scene_ids.insert(entity, scene_id);
                        self.geometry_dirty = true;
                        self.scene_dirty = true;
                        self.gpu_objects_dirty = true;
                        self.console.info(format!("Loaded '{name}': {} voxels", info.voxel_count));
                    }
                    Err(e) => {
                        self.console.error(format!("Failed to load '{path}': {e}"));
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

                        let scene_path = project_dir.join(format!("scenes/{}.rkscene", project.default_scene));
                        if scene_path.exists() {
                            self.load_scene_from_file(&scene_path);
                        }
                        self.scene_path = Some(scene_path);

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
                    if button == rkf_runtime::input::InputMouseButton::Left {
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
                    if let Some(entry) = self.registry.get(&component_name) {
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
                    "camera_altitude" => {
                        if let Ok(v) = value.parse::<f32>() { env.camera_altitude = v; }
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
                    "distance_fog_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.distance_fog_density = v; }
                    }
                    "distance_fog_falloff" => {
                        if let Ok(v) = value.parse::<f32>() { env.distance_fog_falloff = v; }
                    }
                    "dust_density" => {
                        if let Ok(v) = value.parse::<f32>() { env.dust_density = v; }
                    }
                    "dust_asymmetry" => {
                        if let Ok(v) = value.parse::<f32>() { env.dust_asymmetry = v; }
                    }
                    "vol_far" => {
                        if let Ok(v) = value.parse::<f32>() { env.vol_far = v; }
                    }
                    // Clouds
                    "clouds_enabled" => {
                        env.clouds_enabled = value == "true" || value == "1";
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
                self.environment_ui_dirty = true;
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

                    // 7. Deserialize component data back.
                    let restored = self.gameplay_loader.deserialize_all(
                        &mut self.world,
                        &self.uuid_to_entity,
                        &saved,
                    );
                    self.console.info(format!("Restored {restored}/{} component instances", saved.len()));

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

        Some(InspectorSnapshot {
            entity_name: name,
            entity_id: format!("{}", self.get_entity_uuid(selected).as_simple()),
            position: pos,
            rotation: rot,
            scale: scl,
            components,
            material_usage,
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

    /// Get or assign a stable scene object ID for face emission.
    fn get_scene_id(&mut self, entity: hecs::Entity) -> u32 {
        if let Some(&id) = self.entity_scene_ids.get(&entity) {
            id
        } else {
            let id = self.next_scene_id;
            self.next_scene_id += 1;
            self.entity_scene_ids.insert(entity, id);
            id
        }
    }

    /// Assign a stable UUID to an entity.
    fn assign_entity_uuid(&mut self, entity: hecs::Entity) -> uuid::Uuid {
        let uuid = uuid::Uuid::new_v4();
        self.entity_uuids.insert(entity, uuid);
        self.uuid_to_entity.insert(uuid, entity);
        uuid
    }

    /// Rebuild GPU objects from the hecs world.
    /// Per-tick procedural maintenance. Bakes any entity that needs an
    /// initial bake (freshly spawned, spatial == None) or has a settled
    /// `pending_bake` (last edit was at least `BAKE_DEBOUNCE` ago).
    /// Interactive build-panel param edits mark `dirty` but sit until
    /// the user clicks "Bake" — only the properties-panel scale slider
    /// (via `redirect_transform_scale_to_root`) currently sets the
    /// auto-bake flag.
    fn update_dirty_procedurals(&mut self) {
        use crate::components::*;

        let mut to_update: Vec<hecs::Entity> = Vec::new();

        // Debounce window for `pending_bake` — long enough to suppress
        // bakes mid-scrub on a slider, short enough to feel immediate
        // when the user releases. Initial bakes don't debounce.
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

        for (entity, (renderable, proc_geo)) in self
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
            let needs_initial_bake = renderable.spatial.is_none() && proc_geo.dirty;
            let pending_settled = !drag_active
                && proc_geo.pending_bake
                && proc_geo
                    .bake_dirty_at
                    .map(|t| now.duration_since(t) >= BAKE_DEBOUNCE)
                    .unwrap_or(true);
            if needs_initial_bake || pending_settled {
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

        let scene_id = self.entity_scene_ids.get(&entity).copied().unwrap_or(0);
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

        let req = crate::bake_worker::BakeRequest {
            entity,
            generation,
            scene_id,
            instructions,
            aabb,
            voxel_size,
            root_scale,
            prev_spatial,
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
            let entity = result.entity;
            // Entity gone? Discard quietly.
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

        self.gpu_objects.clear();
        self.gpu_to_entity.clear();
        self.entity_to_gpu.clear();
        self.scene_id_to_gpu.clear();

        for (entity, (transform, renderable)) in self.world.query::<(&Transform, &Renderable)>().iter() {
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
                let spatial_handle = rkf_core::scene_node::SpatialHandle::Octree {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                let mut gpu_obj = crate::scene_sync::build_gpu_object(
                    &world_matrix,
                    &spatial.aabb,
                    spatial.grid_origin,
                    &spatial_handle,
                    spatial.voxel_size,
                    renderable.material_id,
                    gpu_idx,
                );
                // Render-layer mask — entity opt-in via RenderLayer
                // component, otherwise the system DEFAULT bit.
                gpu_obj.layer_mask = self
                    .world
                    .get::<&crate::viewport::RenderLayer>(entity)
                    .map(|l| l.mask)
                    .unwrap_or(crate::viewport::layer::DEFAULT);
                if let Some(&scene_id) = self.entity_scene_ids.get(&entity) {
                    self.scene_id_to_gpu.insert(scene_id, gpu_idx);
                }
                self.entity_to_gpu.insert(entity, self.gpu_objects.len());
                self.gpu_to_entity.push(entity);
                self.gpu_objects.push(gpu_obj);
            }
        }
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

                out.push(crate::snapshot::ModelInfo {
                    name,
                    path: rkp_path,
                    source_path: source_path
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    size,
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

    /// Delete an entity and clean up all associated data.
    fn delete_entity(&mut self, entity: hecs::Entity) {
        // Get name for logging.
        let name = self.world.get::<&crate::components::EditorMetadata>(entity)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());

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
        self.entity_scene_ids.remove(&entity);

        // Despawn from ECS.
        let _ = self.world.despawn(entity);

        self.console.info(format!("Deleted '{name}'"));
        self.geometry_dirty = true;
        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    /// Duplicate an entity (deep copy of all components).
    fn duplicate_entity(&mut self, source: hecs::Entity) {
        use crate::components::*;

        let name = self.world.get::<&EditorMetadata>(source)
            .map(|m| m.name.clone())
            .unwrap_or_else(|_| "unknown".into());
        let new_name = self.unique_name(&name);

        // Read components from source.
        let transform = self.world.get::<&Transform>(source)
            .map(|t| (*t).clone())
            .unwrap_or_else(|_| Transform::default());
        let renderable = self.world.get::<&Renderable>(source)
            .map(|r| (*r).clone())
            .ok();
        let point_light = self.world.get::<&PointLight>(source)
            .map(|l| (*l).clone())
            .ok();
        let camera = self.world.get::<&Camera>(source)
            .map(|c| (*c).clone())
            .ok();
        let parent = self.world.get::<&Parent>(source)
            .map(|p| (*p).clone())
            .ok();

        // Offset the duplicate slightly so it's visible.
        let mut new_transform = transform;
        new_transform.position += glam::Vec3::new(0.5, 0.0, 0.5);

        // Spawn the new entity with the same components.
        let entity = self.world.spawn((
            new_transform,
            EditorMetadata { name: new_name.clone() },
        ));

        if let Some(r) = renderable {
            // Sharing the spatial data is fine — voxels are immutable once created.
            // The duplicate gets its own scene_id for face emission.
            let scene_id = self.next_scene_id;
            self.next_scene_id += 1;
            self.entity_scene_ids.insert(entity, scene_id);
            let _ = self.world.insert_one(entity, r);
        }
        if let Some(l) = point_light {
            let _ = self.world.insert_one(entity, l);
        }
        if let Some(c) = camera {
            let _ = self.world.insert_one(entity, c);
        }
        if let Some(p) = parent {
            let _ = self.world.insert_one(entity, p);
        }

        self.assign_entity_uuid(entity);
        self.selected_entity = Some(entity);

        self.console.info(format!("Duplicated '{name}' → '{new_name}'"));
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
        *self.scene_mgr.lock().unwrap() = RkpSceneManager::new(1_000_000);
        self.selected_entity = None;
        self.geometry_dirty = true;
        self.scene_dirty = true;
        self.gpu_objects_dirty = true;
    }

    fn load_scene_from_file(&mut self, path: &std::path::Path) {
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
                        let sid = self.next_scene_id;
                        self.next_scene_id += 1;
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
                                self.entity_scene_ids.insert(e, sid);
                                self.geometry_dirty = true;
                                Some(e)
                            }
                            Err(_) => None,
                        }
                    } else if let Some(ref prim_name) = obj.primitive {
                        let primitive = match prim_name.as_str() {
                            "box" => rkf_core::scene_node::SdfPrimitive::Box {
                                half_extents: glam::Vec3::from_array(obj.scale) * 0.5,
                            },
                            "sphere" => rkf_core::scene_node::SdfPrimitive::Sphere {
                                radius: obj.scale[0] * 0.5,
                            },
                            _ => continue,
                        };
                        let sid = self.next_scene_id;
                        self.next_scene_id += 1;
                        self.scene_mgr.lock().unwrap().voxelize_primitive(
                            &primitive, obj.material_id, 0.05, glam::Vec3::ONE, sid,
                        ).map(|result| {
                            let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.grid_origin, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
                            let e = self.world.spawn((transform, meta, Renderable {
                                primitive: Some(prim_name.clone()),
                                material_id: obj.material_id,
                                voxel_count: result.voxel_count,
                                spatial: Some(spatial),
                                ..Default::default()
                            }));
                            self.entity_scene_ids.insert(e, sid);
                            self.geometry_dirty = true;
                            e
                        })
                    } else {
                        // Entity with no renderable (e.g. empty transform node).
                        Some(self.world.spawn((transform, meta)))
                    };

                    if let Some(e) = entity {
                        self.assign_entity_uuid(e);
                        uuid_to_hecs.insert(obj.id, e);

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
                for obj in &scene.objects {
                    if obj.components.is_empty() {
                        continue;
                    }
                    let Some(&entity) = uuid_to_hecs.get(&obj.id) else { continue };
                    for (comp_name, json) in &obj.components {
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

            objects.push(crate::scene_io::SceneObject {
                id: self.get_entity_uuid(entity),
                name: meta.name.clone(),
                position: transform.position.to_array(),
                rotation: transform.rotation.to_array(),
                scale: transform.scale.to_array(),
                parent_id: parent.map(|p| p.parent_id),
                asset_path: renderable.as_ref().and_then(|r| r.asset_path.clone()),
                primitive: renderable.as_ref().and_then(|r| r.primitive.clone()),
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

        let left_pressed = self.input_system.raw_state().is_mouse_button_pressed(rkf_runtime::input::InputMouseButton::Left);

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

    fn build_gizmo_wireframe(&self) -> Vec<rkf_render::LineVertex> {
        let mut verts = Vec::new();

        // Light gizmos — always visible for all light entities.
        let light_color = [1.0, 0.9, 0.5, 0.5]; // warm yellow, semi-transparent
        let selected_light_color = [1.0, 0.9, 0.5, 1.0]; // bright when selected

        for (entity, (transform, pl)) in self.world.query::<(&crate::components::Transform, &crate::components::PointLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            // Always show crosshair icon.
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkf_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Range sphere only when selected.
            if selected {
                verts.extend(rkf_render::wireframe::point_light_wireframe(
                    transform.position, pl.range, selected_light_color,
                ));
            }
        }

        for (entity, (transform, sl)) in self.world.query::<(&crate::components::Transform, &crate::components::SpotLight)>().iter() {
            let selected = self.selected_entity == Some(entity);
            let icon_color = if selected { selected_light_color } else { light_color };
            verts.extend(rkf_render::wireframe::crosshair(transform.position, 0.2, icon_color));
            // Cone only when selected.
            if selected {
                verts.extend(rkf_render::wireframe::spot_light_wireframe(
                    transform.position, sl.direction, sl.range, sl.outer_angle.to_radians(), selected_light_color,
                ));
            }
        }

        // Physics collider wireframes.
        if self.show_colliders {
            verts.extend(self.build_collider_wireframes());
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
    ) -> Vec<rkf_render::LineVertex> {
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

            let aabb_half = spatial.as_ref()
                .map(|s| s.aabb.half_extents() * scale)
                .unwrap_or(glam::Vec3::splat(0.5));

            if let Some(ref sp) = spatial {
                let extent = (1u32 << sp.depth) as f32 * sp.base_voxel_size;
                eprintln!("[ColliderCache] '{name}' pos={pos:?} scale={scale:?} aabb={:?}..{:?} aabb_half={aabb_half:?} extent={extent}",
                    sp.aabb.min, sp.aabb.max);
            }

            let (resolved_shape, voxel_coords, voxel_size) = match rb.collider_shape {
                rkf_physics::rigid_body::ColliderShape::Auto => {
                    if let Some(ref sp) = spatial {
                        let (coords, cell_size) = crate::play_mode::build_coarse_collider(
                            all_nodes,
                            sp.root_offset as usize,
                            sp.depth,
                            sp.len,
                            sp.base_voxel_size,
                            rb.collider_cell_size,
                        );
                        if coords.is_empty() {
                            (rkf_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                        } else {
                            (rkf_physics::rigid_body::ColliderShape::Auto, coords, cell_size)
                        }
                    } else {
                        (rkf_physics::rigid_body::ColliderShape::Box, Vec::new(), 0.0)
                    }
                }
                other => (other.clone(), Vec::new(), 0.0),
            };

            // Compute grid origin: aabb_center - extent/2 (same as voxelization).
            let (grid_origin, tree_depth) = if let Some(ref sp) = spatial {
                let aabb_center = (sp.aabb.min + sp.aabb.max) * 0.5;
                let extent = (1u32 << sp.depth) as f32 * sp.voxel_size;
                (aabb_center - glam::Vec3::splat(extent * 0.5), sp.depth)
            } else {
                (glam::Vec3::ZERO, 0)
            };

            let cache = ColliderCache {
                shape: resolved_shape,
                voxel_coords,
                collider_cell_size: voxel_size, // actually the coarse cell size from build_coarse_collider
                aabb_half,
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
    fn build_collider_wireframes(&self) -> Vec<rkf_render::LineVertex> {
        use rkf_physics::rigid_body::{BodyType, ColliderShape};
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

            match cache.shape {
                ColliderShape::Box => {
                    let min = transform.position - cache.aabb_half;
                    let max = transform.position + cache.aabb_half;
                    verts.extend(rkf_render::wireframe::aabb_wireframe(min, max, color));
                }
                ColliderShape::Sphere => {
                    let r = cache.aabb_half.max_element();
                    verts.extend(rkf_render::wireframe::sphere_wireframe(transform.position, r, color));
                }
                ColliderShape::Capsule => {
                    let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
                    let hh = (cache.aabb_half.y - r).max(0.01);
                    let top = transform.position + glam::Vec3::new(0.0, hh, 0.0);
                    let bot = transform.position - glam::Vec3::new(0.0, hh, 0.0);
                    verts.extend(rkf_render::wireframe::sphere_wireframe(top, r, color));
                    verts.extend(rkf_render::wireframe::sphere_wireframe(bot, r, color));
                    for angle in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2] {
                        let offset = glam::Vec3::new(angle.cos() * r, 0.0, angle.sin() * r);
                        verts.push(rkf_render::LineVertex { position: (top + offset).to_array(), color });
                        verts.push(rkf_render::LineVertex { position: (bot + offset).to_array(), color });
                    }
                }
                ColliderShape::Auto => {
                    if !cache.voxel_coords.is_empty() {
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
                            verts.extend(rkf_render::wireframe::aabb_wireframe(world_min, world_max, color));
                        }
                    } else {
                        let min = transform.position - cache.aabb_half;
                        let max = transform.position + cache.aabb_half;
                        verts.extend(rkf_render::wireframe::aabb_wireframe(min, max, color));
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

    fn process_pick_result(&mut self) {
        let pending = self.pending_pick.take();
        let slice = self.pick_readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            if data.len() >= 260 {
                // Material G-buffer (Rg32Uint, at offset 0):
                //   R = primary (low16) | secondary (high16)
                //   G = blend (lo8) | object_id+1 (8-15) | color_rgb565 (16-31)
                // Pick G-buffer (R32Uint, at offset 256):
                //   primitive_node_id (low16 for procedural hits,
                //   0xFFFF for misses / combinators / voxel hits).
                let r_channel = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let g_channel = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                let pick_channel = u32::from_le_bytes([data[256], data[257], data[258], data[259]]);
                let _ = r_channel; // primary/secondary material ids not needed for pick
                let obj_raw = (g_channel >> 8) & 0xFF;

                // Dispatch by viewport. BUILD + raymarch resolves the
                // NodeId from the dedicated pick texture. Other paths
                // (MAIN + voxel, or BUILD + voxel) fall back to the
                // same scene_id table — entity-granularity pick.
                let is_build_raymarch = pending.map(|pp| {
                    pp.viewport == crate::viewport::ViewportId::BUILD
                }).unwrap_or(false)
                    && self.viewports
                        .get(crate::viewport::ViewportId::BUILD)
                        .map(|v| matches!(v.preview_mode, rkp_render::BuildPreviewMode::Raymarch))
                        .unwrap_or(false);

                if is_build_raymarch {
                    // Ghost priority: if the CPU raycast at click time
                    // found a ghost primitive on the ray, that wins —
                    // matches the "translucent overlay on top owns the
                    // click" rule, and catches cutters fully carved
                    // away by their parent (no visible surface, so the
                    // G-buffer has nothing for them).
                    if let Some(ghost_id) = pending.and_then(|pp| pp.ghost_pick_node_id) {
                        self.selected_procedural_node = Some(ghost_id);
                    } else {
                        let node_id_16 = pick_channel & 0xFFFFu32;
                        // Shader writes 0xFFFF for misses/combinators;
                        // treat as "no selection" and clear any
                        // previously-selected node so click-on-empty
                        // deselects.
                        if obj_raw > 0 && node_id_16 != 0xFFFFu32 {
                            self.selected_procedural_node = Some(node_id_16);
                        } else {
                            self.selected_procedural_node = None;
                        }
                    }
                } else if obj_raw > 0 {
                    let gpu_idx = (obj_raw - 1) as usize;
                    if gpu_idx < self.gpu_to_entity.len() {
                        self.selected_entity = Some(self.gpu_to_entity[gpu_idx]);
                    }
                } else {
                    self.selected_entity = None;
                }
            }
            drop(data);
            self.pick_readback_buffer.unmap();
        }
    }

    fn build_state_update(&mut self, frame_time: Duration) -> StateUpdate {
        let fps = if frame_time.as_secs_f32() > 0.0 {
            1.0 / frame_time.as_secs_f32()
        } else {
            0.0
        };

        let objects = if self.scene_dirty {
            self.scene_dirty = false;
            let mut objs = Vec::new();
            for (entity, meta) in self.world.query::<&crate::components::EditorMetadata>().iter() {
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

        let models = if self.models_dirty {
            self.models_dirty = false;
            Some(self.available_models.clone())
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

        StateUpdate {
            fps,
            gpu_object_count: self.gpu_objects.len() as u32,
            camera_position: self.camera.position,
            play_mode: self.play_state.is_some(),
            selected_entity: self.selected_entity.map(|e| self.get_entity_uuid(e)),
            objects,
            project_loaded: project,
            project_name,
            available_models: models,
            importing_models: importing,
            import_progress,
            editor_layout,
            inspector: self.build_inspector_snapshot(),
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
            environment: if self.frame_index <= 1 || self.environment_ui_dirty {
                self.environment_ui_dirty = false;
                Some(self.environment.clone())
            } else {
                None
            },
            procedural: self.build_procedural_snapshot(),
            console_entries: self.console.drain_new(),
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

fn procedural_voxel_params(tree: &rkp_procedural::ProceduralObject, base_voxel_size: f32) -> (rkf_core::Aabb, f32) {
    let tight = rkp_procedural::compute_bounds(tree);

    // Add margin for boundary sampling (same approach as voxelize_primitive).
    // Grid placement is handled by threading `grid_origin` through to the
    // shader (`local_origin - grid_origin` replaces the old
    // `local_origin + extent/2`), so we can return a tight AABB here
    // without wasting voxel budget on symmetric padding around the origin.
    let margin = base_voxel_size * 8.0 * 1.8 + base_voxel_size;
    let aabb = rkf_core::Aabb {
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
    let mut state = EngineState::new(&config);
    state.console.info(format!("Engine started ({}x{})", config.width, config.height));

    // Try to load a pre-built gameplay dylib (if project is already set).
    // Normally the dylib is scaffolded + built when a project is opened.
    state.try_load_gameplay_dylib();

    loop {
        let frame_start = Instant::now();

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

        // 1c. Step gameplay systems + physics if in play mode.
        //
        // Frame order: Update → flush → FixedUpdate → flush → Physics → LateUpdate → flush
        //
        // Gameplay runs before physics so scripts can set transforms on kinematic
        // bodies before physics reads them. Dynamic bodies have their transforms
        // overwritten by physics afterward (physics owns dynamic bodies).
        if state.play_state.is_some() {
            let dt = 1.0 / 60.0;
            let fixed_dt = 1.0 / 60.0;
            state.play_total_time += dt as f64;
            state.play_frame_count += 1;

            // Update + FixedUpdate phases
            if let Some(ref mut executor) = state.behavior_executor {
                executor.tick(
                    &state.gameplay_systems,
                    &mut state.world,
                    &mut state.behavior_commands,
                    &mut state.game_store,
                    dt, fixed_dt,
                    state.play_total_time,
                    state.play_frame_count,
                );
                state.gpu_objects_dirty = true;
            }

            // Physics step (between FixedUpdate and LateUpdate)
            if let Some(ref mut play) = state.play_state {
                if play.step(dt, &mut state.world) {
                    state.gpu_objects_dirty = true;
                }
            }

            // LateUpdate phase
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

        // 2. Update input system + camera.
        let dt = 1.0 / 60.0; // TODO: use actual delta time
        state.input_system.evaluate();
        state.camera_control.update(
            &state.input_system,
            dt,
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

        // 4. Render frame — fires `frame_callback` once per visible viewport.
        state.render_frame(&frame_callback);

        // 6. Push state to client.
        let frame_time = frame_start.elapsed();
        let update = state.build_state_update(frame_time);
        state_callback(&update);

        // 7. Clear per-frame input state for next tick.
        state.input_system.begin_frame();

        // 8. Frame pacing — target ~60 FPS.
        let target = Duration::from_micros(16_667);
        let elapsed = frame_start.elapsed();
        if elapsed < target {
            std::thread::sleep(target - elapsed);
        }
    }
}

/// Extract 6 frustum planes from a view-projection matrix.
/// Each plane is (nx, ny, nz, d) where nx*x + ny*y + nz*z + d >= 0 means inside.
fn extract_frustum_planes(vp: &glam::Mat4) -> [glam::Vec4; 6] {
    let r0 = vp.row(0);
    let r1 = vp.row(1);
    let r2 = vp.row(2);
    let r3 = vp.row(3);
    [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r3 + r2, // near
        r3 - r2, // far
    ]
}

/// Test if an AABB (center + half-extents) is inside or intersects the frustum.
fn aabb_in_frustum(planes: &[glam::Vec4; 6], center: glam::Vec3, half: glam::Vec3) -> bool {
    for plane in planes {
        let n = glam::Vec3::new(plane.x, plane.y, plane.z);
        let d = plane.w;
        // Effective radius: project half-extents onto the plane normal.
        let r = half.x * n.x.abs() + half.y * n.y.abs() + half.z * n.z.abs();
        // If the center is further than r behind the plane, the AABB is fully outside.
        if n.dot(center) + d + r < 0.0 {
            return false;
        }
    }
    true
}
