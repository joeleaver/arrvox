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
    handle: &rkp_core::scene_node::SpatialHandle,
    voxel_size: f32,
    aabb: &rkp_core::Aabb,
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
            voxel_slot_start,
            voxel_slot_count,
            brick_ids,
        }
    } else {
        SpatialData {
            root_offset: 0, len: 0, depth: 0, base_voxel_size: voxel_size,
            aabb: *aabb, voxel_size,
            voxel_slot_start, voxel_slot_count,
            brick_ids,
        }
    }
}

/// Frame delivery callback — called each tick with RGBA8 pixels.
pub type FrameCallback = Box<dyn Fn(&[u8], u32, u32) + Send>;

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

struct EngineState {
    // GPU
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Rendering pipeline
    renderer: RkpRenderer,
    gbuffer: rkp_render::GBuffer,
    bloom: rkp_render::BloomPass,
    bloom_composite: rkp_render::BloomCompositePass,
    tone_map: rkp_render::ToneMapPass,

    // Scene management (CPU)
    scene_mgr: RkpSceneManager,

    // Input + Camera
    input_system: rkp_runtime::input::InputSystem,
    camera_control: CameraControlState,
    camera: CameraState,

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

    // Previous frame's view-projection matrix, used for temporal reprojection
    // (e.g. the volumetric cloud pass). Identity on the first frame.
    prev_view_proj: [[f32; 4]; 4],

    // Temporally smoothed cloud-sun attenuation (camera→sun ray through the
    // cloud layer). Lerps toward the target each frame so a single noisy ray
    // through FBM doesn't flicker sun intensity.
    cloud_sun_atten: f32,

    // Render dimensions
    width: u32,
    height: u32,

    // Double-buffered readback: copy to one buffer this frame, read from the other
    // (which completed last frame). Avoids blocking CPU waiting for GPU.
    readback_buffers: [wgpu::Buffer; 2],
    readback_index: usize, // which buffer to copy INTO this frame
    readback_ready: bool,  // false on first frame (no previous data yet)

    // Wireframe overlay
    wireframe_pass: rkp_render::WireframePass,
    /// Composite texture — LDR + wireframe overlay. Rgba8Unorm with RENDER_ATTACHMENT.
    composite_texture: wgpu::Texture,
    composite_view: wgpu::TextureView,

    // Gizmo state
    gizmo: crate::gizmo::GizmoState,
    /// Mouse position in viewport pixels (for gizmo hover).
    mouse_pos: glam::Vec2,

    // Pick readback (8 bytes for 1 pixel of Rg32Uint material texture)
    pick_readback_buffer: wgpu::Buffer,
    pending_pick: Option<(u32, u32)>,
    /// Cached light count for march pass (set in light upload block, used in render).
    num_lights_cache: u32,
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
        let ctx = rkp_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let gbuffer = rkp_render::GBuffer::new(&device, width, height);
        let mut renderer = RkpRenderer::new(&device, &queue, width, height);

        // Wire G-buffer into renderer.
        renderer.set_gbuffer(&gbuffer);

        // Bloom: extract bright pixels from volumetric HDR output, blur, composite.
        let bloom = rkp_render::BloomPass::new(
            &device,
            &renderer.god_rays.output_view,
            width,
            height,
        );
        let bloom_composite = rkp_render::BloomCompositePass::new(
            &device,
            &renderer.god_rays.output_view,
            bloom.mip_views(),
            width,
            height,
        );

        // Tone mapping: bloom composite HDR → LDR (Rgba8Unorm).
        let tone_map = rkp_render::ToneMapPass::new(
            &device,
            &bloom_composite.output_view,
            width,
            height,
        );

        let scene_mgr = RkpSceneManager::new(1_000_000);

        // Input system with default action map.
        let mut input_system = rkp_runtime::input::InputSystem::new();
        input_system.add_map(crate::camera::default_action_map());
        input_system.set_active_map("editor");
        let camera_control = CameraControlState::default();

        // Double-buffered readback — avoids blocking CPU for GPU completion.
        let readback_buffers = [
            Self::create_readback_buffer(&device, width, height),
            Self::create_readback_buffer(&device, width, height),
        ];

        // Wireframe pass for gizmo overlay.
        let wireframe_pass = rkp_render::WireframePass::new(&device, rkp_render::LDR_FORMAT);

        // Composite texture: LDR + wireframes. Needs RENDER_ATTACHMENT for wireframe draw.
        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: rkp_render::LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let composite_view = composite_texture.create_view(&Default::default());

        // Pick readback buffer — 1 pixel of Rg32Uint (8 bytes), 256-byte aligned.
        let pick_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp pick readback"),
            size: 256, // wgpu requires COPY_DST alignment
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            renderer,
            gbuffer,
            bloom,
            bloom_composite,
            tone_map,
            scene_mgr,
            input_system,
            camera_control,
            camera: CameraState::default(),
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
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cloud_sun_atten: 1.0,
            width,
            height,
            readback_buffers,
            readback_index: 0,
            readback_ready: false,
            wireframe_pass,
            composite_texture,
            composite_view,
            gizmo: crate::gizmo::GizmoState::new(),
            mouse_pos: glam::Vec2::ZERO,
            pick_readback_buffer,
            pending_pick: None,
            num_lights_cache: 1,
            lod_enabled: true,
            // Bake-time Laplacian smoothing of stored normals (see
            // `load_asset` → `smooth_shell_normals`) makes the shader-
            // time centroid reconstruction redundant. Default OFF so
            // the shader uses the smoothed baked normal via its
            // existing 1-fetch path.
            surfacenet_enabled: false,
        }
    }

    fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let padded_row = (width * 4 + 255) & !255;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn render_frame(&mut self) -> (Vec<u8>, u32, u32) {
        let frame_start = std::time::Instant::now();

        // 0a. Upload material palette if dirty.
        if self.material_lib.is_dirty() {
            let palette = self.material_lib.build_palette();
            self.renderer.update_materials(&self.queue, &palette);
            self.material_lib.clear_dirty();
        }

        // Camera uniforms are needed now for altitude-dependent atmosphere
        // params (sun transmittance, sky colors). Building them early — the
        // actual per-frame upload still happens at step 2.
        let cam_uniforms = self.build_camera_uniforms();
        let cam_y = cam_uniforms.position[1];

        // 0b. Upload environment + lights.
        // Always rebuild lights array (entity lights may have moved).
        {
            // Cloud → sun attenuation comes from a GPU readback of the dedicated
            // compute pass. `sun_atten_value()` is the last-received exp(-τ); it
            // lags 1–2 frames but the temporal lerp hides that.
            let target_atten = if self.environment.attenuate_sun_by_clouds && self.environment.clouds_enabled {
                self.renderer.volumetric.sun_atten_value()
            } else {
                1.0
            };
            // Lerp slowly toward the readback value — the GPU integral can still
            // swing a bit when a cloud edge crosses the camera→sun ray, and a
            // multi-frame fade reads as "cloud rolling over the sun" rather than
            // a per-frame flicker.
            self.cloud_sun_atten = self.cloud_sun_atten + (target_atten - self.cloud_sun_atten) * 0.04;

            let mut sun_light = self.environment.to_gpu_light(cam_y);
            sun_light.color[0] *= self.cloud_sun_atten;
            sun_light.color[1] *= self.cloud_sun_atten;
            sun_light.color[2] *= self.cloud_sun_atten;
            let mut gpu_lights = vec![sun_light]; // [0] = sun

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

            let mut shade_params = self.environment.to_shade_params(cam_y);
            shade_params.num_lights = gpu_lights.len() as u32;
            self.renderer.update_shade_params(&self.queue, &shade_params);
            self.renderer.update_lights(&self.queue, &gpu_lights);
            self.num_lights_cache = shade_params.num_lights;

            if self.environment_dirty {
                self.tone_map.set_exposure(&self.queue, self.environment.exposure);
                self.bloom.set_threshold(&self.queue, self.environment.bloom_threshold, self.environment.bloom_knee);
                self.bloom_composite.set_intensity(&self.queue, self.environment.bloom_intensity);
                self.environment_dirty = false;
            }
        }

        // 0c. Rebuild GPU objects from ECS world only when transforms/objects changed.
        if self.gpu_objects_dirty {
            self.update_scene_gpu();
            self.gpu_objects_dirty = false;
        }

        let t_cpu_setup = frame_start.elapsed();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rkp frame"),
        });

        // 1. Upload geometry if dirty.
        if self.geometry_dirty {
            let geo = self.scene_mgr.geometry_upload();
            self.renderer.upload_geometry(&self.queue, &geo);
            self.geometry_dirty = false;
            self.collider_caches_dirty = true;
        }

        // 1e. Rebuild collider caches if needed.
        if self.collider_caches_dirty {
            self.rebuild_collider_caches();
            self.collider_caches_dirty = false;
        }

        // 2. Upload per-frame data (objects + camera + bone matrices).
        // (cam_uniforms was built above for the altitude-dependent params.)
        let frame = FrameUpload {
            objects: &self.gpu_objects,
            camera: &cam_uniforms,
            bone_matrices: self.bone_matrix_allocator.bytes(),
            bone_dual_quats: self.bone_matrix_allocator.bytes_dq(),
        };
        self.renderer.upload_frame(&self.queue, &frame);

        // 2b. Skeletal skin-deform scatter. Folds every skinned
        // entity into one batched compute dispatch so we fire exactly
        // one `write_buffer` per input — side-stepping the write-
        // ordering pitfall of multiple per-entity dispatches sharing
        // a single uniform/brick buffer inside one submission.
        if self.skinning_enabled && !self.skin_dispatches.is_empty() && !self.skin_reuse {
            let q = self.renderer.profiler.begin_query("skin_deform", &mut encoder);
            self.renderer.prepare_bone_field(
                &self.queue,
                &mut encoder,
                self.skin_bone_field_bytes,
                self.skin_bone_field_occ_bytes,
            );
            self.skin_batch.clear();
            for plan in &self.skin_dispatches {
                let d = rkp_render::SkinDispatch {
                    uniforms: plan.uniforms,
                    bricks: &plan.bricks,
                };
                self.skin_batch.push(&d);
            }
            self.renderer.scatter_skin_batch(&self.queue, &mut encoder, &self.skin_batch);
            self.renderer.profiler.end_query(&mut encoder, q);
        } else if self.skinning_enabled && self.frame_index % 60 == 0 {
            // Once a second, log why scatter isn't running when the
            // user has the toggle on — most common reason is a stale
            // `.rkp` without the new skin-meta section, or no skinned
            // entities in the scene.
            let skinned_entities = self.world.query::<&crate::components::Skeleton>().iter().count();
            if skinned_entities > 0 {
                eprintln!(
                    "[RkpEngine] skinning enabled, {} skinned entities, but 0 scatter dispatches this frame. \
                     Likely cause: stale .rkp without skin-meta section — re-import the asset.",
                    skinned_entities,
                );
            }
        }

        let t_upload = frame_start.elapsed();

        // 3. Render: march (+ per-light shadow) → SSAO → shade → volumetrics.
        let object_count = self.gpu_objects.len() as u32;
        let shadow_steps = self.environment.shadow_steps;
        let num_lights = self.num_lights_cache;
        let vp = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
        let screen_aabbs = crate::scene_sync::compute_screen_aabbs(
            &self.gpu_objects, &vp, self.width as f32, self.height as f32,
        );
        let screen_aabbs_bytes: &[u8] = bytemuck::cast_slice(&screen_aabbs);

        // Upload volumetric params.
        let vol_params = self.environment.to_volumetric_params(
            &cam_uniforms, self.width, self.height, self.frame_index as u32,
        );
        self.renderer.update_volumetric_params(&self.queue, &vol_params);
        let cloud_params = self.environment.to_cloud_params(self.frame_index as f32 / 60.0);
        self.renderer.update_cloud_params(&self.queue, &cloud_params);

        // Atmosphere per-frame params.
        let sun_d = self.environment.sun_direction();
        let atmo_frame = rkp_render::rkp_atmosphere::AtmosphereFrameParams {
            sun_dir: [-sun_d[0], -sun_d[1], -sun_d[2]],
            sun_intensity: self.environment.sun_intensity,
            camera_altitude: self.environment.effective_altitude(cam_y),
            ground_albedo: self.environment.ground_albedo,
            cam_pos: [cam_uniforms.position[0], cam_uniforms.position[1], cam_uniforms.position[2]],
            _pad1b: 0.0,
            cam_forward: [cam_uniforms.forward[0], cam_uniforms.forward[1], cam_uniforms.forward[2]],
            _pad2: 0.0,
            cam_right: [cam_uniforms.right[0], cam_uniforms.right[1], cam_uniforms.right[2]],
            _pad3: 0.0,
            cam_up: [cam_uniforms.up[0], cam_uniforms.up[1], cam_uniforms.up[2]],
            _pad4: 0.0,
        };

        // God ray params: project sun position to screen space.
        {
            let sun_toward = [-sun_d[0], -sun_d[1], -sun_d[2]];
            // Sun is infinitely far — project direction as a point far along sun_dir.
            let sun_world = glam::Vec3::new(
                cam_uniforms.position[0] + sun_toward[0] * 1000.0,
                cam_uniforms.position[1] + sun_toward[1] * 1000.0,
                cam_uniforms.position[2] + sun_toward[2] * 1000.0,
            );
            let vp = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            let clip = vp * glam::Vec4::new(sun_world.x, sun_world.y, sun_world.z, 1.0);
            let sun_on_screen = if clip.w > 0.0 { 1.0 } else { 0.0 };
            let ndc = if clip.w > 0.0 {
                glam::Vec2::new(clip.x / clip.w, clip.y / clip.w)
            } else {
                glam::Vec2::ZERO
            };
            let sun_uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];

            // Atmospherically-tinted sun colour (plain 0-1 linear RGB). HDR
            // magnitude comes from reading the composite's sun-disc luminance
            // inside the shader; the tint here just biases the ray hue.
            let sun_tint = self.environment.sun_tint(cam_y);

            let god_ray_params = rkp_render::rkp_god_rays::GodRayParams {
                sun_screen_pos: sun_uv,
                sun_on_screen,
                density: self.environment.god_ray_density,
                weight: self.environment.god_ray_weight,
                decay: self.environment.god_ray_decay,
                exposure: self.environment.god_ray_exposure,
                num_samples: 64,
                sun_color: sun_tint,
                _pad: 0.0,
            };
            self.renderer.god_rays.update_params(&self.queue, &god_ray_params);
        }

        self.renderer.render(&mut encoder, &self.queue, object_count, self.width, self.height, shadow_steps, num_lights, self.lod_enabled, self.surfacenet_enabled, screen_aabbs_bytes, &atmo_frame);

        let t_encode = frame_start.elapsed();

        // 4b. Pick: copy material texture (object_id+1 in G bits 8-15, 0 = no hit).
        let pick_issued = self.pending_pick.is_some();
        if let Some((px, py)) = self.pending_pick.take() {
            if px < self.width && py < self.height {
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.gbuffer.material_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: px, y: py, z: 0 },
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
            }
        }

        // 5a. Bloom: extract bright pixels + multi-level blur.
        {
            let q = self.renderer.profiler.begin_query("bloom", &mut encoder);
            self.bloom.dispatch(&mut encoder);
            self.renderer.profiler.end_query(&mut encoder, q);
        }

        // 5b. Bloom composite: blend blurred bloom back onto HDR.
        {
            let q = self.renderer.profiler.begin_query("bloom_composite", &mut encoder);
            self.bloom_composite.dispatch(&mut encoder);
            self.renderer.profiler.end_query(&mut encoder, q);
        }

        // 5c. Tone mapping: bloom composite HDR → LDR (Rgba8Unorm).
        {
            let q = self.renderer.profiler.begin_query("tone_map", &mut encoder);
            self.tone_map.dispatch(&mut encoder);
            self.renderer.profiler.end_query(&mut encoder, q);
        }

        // 6. Copy LDR to composite texture, draw gizmo wireframes, readback.
        let q_post = self.renderer.profiler.begin_query("post", &mut encoder);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.tone_map.ldr_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // 6b. Draw gizmo wireframe if an object is selected.
        let gizmo_verts = self.build_gizmo_wireframe();
        if !gizmo_verts.is_empty() {
            let cam_uniforms = self.build_camera_uniforms();
            let vp_matrix = glam::Mat4::from_cols_array_2d(&cam_uniforms.view_proj);
            self.wireframe_pass.draw(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.composite_view,
                vp_matrix,
                (0.0, 0.0, self.width as f32, self.height as f32),
                &gizmo_verts,
            );
        }

        // 6c. Copy composite to readback buffer.
        let padded_row = (self.width * 4 + 255) & !255;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffers[self.readback_index],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.renderer.profiler.end_query(&mut encoder, q_post);

        // 7b. Resolve all profiler queries now that every pass has been encoded.
        self.renderer.resolve_profiler_queries(&mut encoder);

        let t_post = frame_start.elapsed();

        // 8. Submit GPU work.
        self.queue.submit(std::iter::once(encoder.finish()));

        // Kick off the cloud→sun atten readback, then drive a non-blocking poll
        // so previously-issued map_async callbacks can fire.
        self.renderer.volumetric.issue_sun_atten_map();
        let _ = self.device.poll(wgpu::PollType::Poll);

        let t_submit = frame_start.elapsed();

        // 9. Process pick readback if we just issued one.
        if pick_issued {
            self.process_pick_result();
        }

        // 10. Read from the OTHER buffer (completed last frame).
        // Double-buffered: we read last frame's data while this frame's copy runs.
        // wait_indefinitely returns near-instantly since last frame's GPU work is done.
        let read_index = 1 - self.readback_index;
        let pixels = if self.readback_ready {
            self.map_readback(read_index)
        } else {
            // First frame: no previous data, read current (blocking).
            self.readback_ready = true;
            self.map_readback(self.readback_index)
        };
        self.readback_index = read_index; // swap for next frame

        // 11. GPU profiler — process finished frames (logs every 60 frames).
        self.renderer.end_profiler_frame(self.frame_index, self.width, self.height);

        let t_frame_end = frame_start.elapsed();

        // Log timing every 60 frames.
        // gpu_wait = CPU blocking on last frame's GPU work to finish so we can
        // map the composite buffer for the editor. The copy itself is cheap;
        // the cost is the sync.
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

        // Remember this frame's view-projection so next frame can reproject into it.
        self.prev_view_proj = cam_uniforms.view_proj;

        self.frame_index += 1;

        (pixels, self.width, self.height)
    }

    /// Read from a readback buffer. With double-buffering we read last frame's
    /// buffer, so the GPU work is already complete and wait returns near-instantly.
    fn map_readback(&self, index: usize) -> Vec<u8> {
        let w = self.width;
        let h = self.height;
        let padded_row = (w * 4 + 255) & !255;

        let buffer_slice = self.readback_buffers[index].slice(..);
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
            self.readback_buffers[index].unmap();
        }

        rgba8
    }

    fn build_camera_uniforms(&self) -> rkp_render::rkp_scene::CameraUniforms {
        // yaw/pitch are in radians (set by camera controller).
        let yaw = self.camera.yaw;
        let pitch = self.camera.pitch;

        // Same fly_direction formula as the camera controller.
        let forward = glam::Vec3::new(
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ).normalize();
        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        let fov_rad = self.camera.fov.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = self.width as f32 / self.height.max(1) as f32;

        let view = glam::Mat4::look_to_rh(self.camera.position, forward, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(fov_rad, aspect, self.camera.near, self.camera.far);
        let view_proj = proj * view;

        rkp_render::rkp_scene::CameraUniforms {
            position: [self.camera.position.x, self.camera.position.y, self.camera.position.z, 1.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            right: [right.x * half_fov_tan * aspect, right.y * half_fov_tan * aspect, right.z * half_fov_tan * aspect, 0.0],
            up: [up.x * half_fov_tan, up.y * half_fov_tan, up.z * half_fov_tan, 0.0],
            resolution: [self.width as f32, self.height as f32],
            jitter: [0.0, 0.0],
            prev_vp: self.prev_view_proj,
            view_proj: view_proj.to_cols_array_2d(),
        }
    }

    fn process_command(&mut self, cmd: EngineCommand) -> bool {
        match cmd {
            EngineCommand::Shutdown => return false,

            EngineCommand::SetCamera { position, yaw, pitch, fov } => {
                self.camera.position = position;
                self.camera.yaw = yaw;
                self.camera.pitch = pitch;
                self.camera.fov = fov;
            }

            EngineCommand::Resize { width, height } => {
                if width != self.width || height != self.height {
                    self.width = width;
                    self.height = height;
                    self.gbuffer = rkp_render::GBuffer::new(&self.device, width, height);
                    self.renderer.resize(width, height);
                    self.renderer.set_gbuffer(&self.gbuffer);
                    self.bloom = rkp_render::BloomPass::new(
                        &self.device,
                        &self.renderer.god_rays.output_view,
                        width,
                        height,
                    );
                    self.bloom_composite = rkp_render::BloomCompositePass::new(
                        &self.device,
                        &self.renderer.god_rays.output_view,
                        self.bloom.mip_views(),
                        width,
                        height,
                    );
                    self.tone_map = rkp_render::ToneMapPass::new(
                        &self.device,
                        &self.bloom_composite.output_view,
                        width,
                        height,
                    );
                    self.readback_buffers = [
                        Self::create_readback_buffer(&self.device, width, height),
                        Self::create_readback_buffer(&self.device, width, height),
                    ];
                    self.readback_ready = false;
                    self.composite_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("rkp composite"),
                        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: rkp_render::LDR_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    self.composite_view = self.composite_texture.create_view(&Default::default());
                    self.environment_dirty = true;
                    self.environment_ui_dirty = true;
                    eprintln!("[RkpEngine] resized to {}x{}", width, height);
                }
            }

            EngineCommand::SpawnPrimitive { name } => {
                use crate::components::*;
                let name = self.unique_name(&name);
                let primitive = rkp_core::scene_node::SdfPrimitive::Box {
                    half_extents: glam::Vec3::splat(0.5),
                };
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;
                let result = self.scene_mgr.voxelize_primitive(
                    &primitive, 0, 0.05, glam::Vec3::ONE, scene_id,
                );
                if let Some(result) = result {
                    let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
                    let entity = self.world.spawn((
                        Transform::default(),
                        EditorMetadata { name: name.clone() },
                        Renderable {
                            primitive: Some("box".to_string()),
                            voxel_count: result.voxel_count,
                            spatial: Some(spatial),
                            ..Default::default()
                        },
                    ));
                    self.assign_entity_uuid(entity);
                    self.entity_scene_ids.insert(entity, scene_id);
                    self.geometry_dirty = true;
                    self.scene_dirty = true;
                    self.gpu_objects_dirty = true;
                    self.console.info(format!("Spawned '{name}': {} voxels", result.voxel_count));
                }
            }

            EngineCommand::SpawnProceduralObject { name } => {
                use crate::components::*;
                let name = self.unique_name(&name);
                let proc_geo = ProceduralGeometry::default_sphere();
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;

                // Compute bounds and voxelize the procedural tree.
                let (aabb, voxel_size) = procedural_voxel_params(&proc_geo.tree, proc_geo.voxel_size);
                let tree_ref = &proc_geo.tree;
                let sdf_fn = |pos: glam::Vec3| -> (f32, u16) {
                    let sample = rkp_procedural::sample_tree(tree_ref, pos, voxel_size);
                    (sample.distance, sample.material_id)
                };

                let result = self.scene_mgr.voxelize_sdf_fn(
                    sdf_fn, &aabb, voxel_size, scene_id,
                );
                if let Some(result) = result {
                    let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
                    let entity = self.world.spawn((
                        Transform::default(),
                        EditorMetadata { name: name.clone() },
                        Renderable {
                            primitive: Some("procedural".to_string()),
                            voxel_count: result.voxel_count,
                            spatial: Some(spatial),
                            ..Default::default()
                        },
                        proc_geo,
                    ));
                    self.assign_entity_uuid(entity);
                    self.entity_scene_ids.insert(entity, scene_id);
                    self.geometry_dirty = true;
                    self.scene_dirty = true;
                    self.gpu_objects_dirty = true;
                    self.console.info(format!("Spawned procedural '{name}': {} voxels", result.voxel_count));
                }
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
                            proc_geo.voxel_size = snapped;
                            proc_geo.dirty = true;
                        }
                    }
                }
            }

            EngineCommand::AddProceduralNode { parent_node_id, kind } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        let parent = rkp_procedural::NodeId(parent_node_id);
                        let node_kind = parse_node_kind(&kind);
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

            EngineCommand::SetProceduralNodePosition { node_id, position } => {
                if let Some(entity) = self.selected_entity {
                    if let Ok(mut proc_geo) = self.world.get::<&mut crate::components::ProceduralGeometry>(entity) {
                        proc_geo.tree.set_transform(
                            rkp_procedural::NodeId(node_id),
                            glam::Affine3A::from_translation(position),
                        );
                        proc_geo.dirty = true;
                    }
                }
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
                match self.scene_mgr.acquire_asset(&path) {
                    Ok((handle, info)) => {
                        let raw_name = Self::display_name_from_path(&path);
                        let name = self.unique_name(&raw_name);
                        // Asset-backed entity — brick_ids stays empty.
                        // Asset cache owns the shared brick range and frees
                        // it on the final release_asset.
                        let spatial = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
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
                        // No auto-attach — the user adds `Skeleton`
                        // manually via the Add Component menu when they
                        // want animation. Helper below in
                        // `try_attach_skeleton` is invoked from the
                        // AddComponent command handler.
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

            EngineCommand::Pick { x, y } => {
                // Don't pick if clicking on a gizmo handle — let the drag start instead.
                if self.gizmo.hovered_axis == crate::gizmo::GizmoAxis::None {
                    self.pending_pick = Some((x, y));
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
            EngineCommand::MouseMove { x, y, dx, dy } => {
                self.mouse_pos = glam::Vec2::new(x, y);
                self.input_system.feed_mouse_delta(glam::Vec2::new(dx, dy));
            }
            EngineCommand::MouseButton { button, pressed } => {
                self.input_system.feed_mouse_button(button, pressed);
            }
            EngineCommand::Scroll { delta } => {
                self.input_system.feed_scroll(delta);
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
                }
            }

            EngineCommand::PlayStop => {
                if let Some(play) = self.play_state.take() {
                    play.stop(&mut self.world);
                    self.behavior_executor = None;
                    self.gpu_objects_dirty = true;
                    self.console.info("Play mode stopped — transforms restored");
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

        // Build component snapshots from the registry.
        let mut components = Vec::new();
        for entry in self.registry.components_on(&self.world, selected) {
            let fields: Vec<FieldSnapshot> = entry.meta.iter().map(|meta| {
                let value = (entry.get_field)(&self.world, selected, meta.name)
                    .unwrap_or(FieldValue::String("<error>".into()));
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
        let all_nodes = self.scene_mgr.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, spatial.root_offset as usize, &mut leaf_slots);

        // Count material IDs across all leaf slots. Every leaf is a surface
        // voxel now — no opacity gate.
        let pool_size = self.scene_mgr.leaf_attr_pool.allocated_count();
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        for slot in leaf_slots {
            if slot >= pool_size {
                continue; // stale or invalid slot — skip
            }
            let attr = self.scene_mgr.leaf_attr_pool.get(slot);
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
        let all_nodes = self.scene_mgr.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, spatial.root_offset as usize, &mut leaf_slots);

        let pool_size = self.scene_mgr.leaf_attr_pool.allocated_count();
        let mut count = 0u32;
        for slot in leaf_slots {
            if slot >= pool_size { continue; }
            let attr = self.scene_mgr.leaf_attr_pool.get(slot);
            let primary = attr.material_primary;
            let secondary = attr.material_secondary();
            let mut changed = false;

            if primary == from_material {
                let m = self.scene_mgr.leaf_attr_pool.get_mut(slot);
                m.material_primary = to_material;
                changed = true;
            }
            if secondary == from_material {
                // Re-pack secondary + blend, since both share material_secondary_blend.
                let attr = *self.scene_mgr.leaf_attr_pool.get(slot);
                let blend = attr.blend_weight();
                let m = self.scene_mgr.leaf_attr_pool.get_mut(slot);
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
        let reload = match self.scene_mgr.reload_asset(&path_str) {
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
    /// Re-voxelize any procedural objects that are dirty or whose entity scale changed.
    fn update_dirty_procedurals(&mut self) {
        use crate::components::*;

        // Collect entities that need re-evaluation. We can't mutate world + scene_mgr
        // simultaneously in the query, so collect first.
        let mut to_update: Vec<(hecs::Entity, u32)> = Vec::new();

        for (entity, (transform, proc_geo)) in self
            .world
            .query::<(&Transform, &ProceduralGeometry)>()
            .iter()
        {
            let scale_changed = (transform.scale - proc_geo.last_evaluated_scale).length() > 1e-5;
            if proc_geo.dirty || scale_changed {
                let scene_id = self.entity_scene_ids.get(&entity).copied().unwrap_or(0);
                to_update.push((entity, scene_id));
            }
        }

        for (entity, scene_id) in to_update {
            // Read the procedural tree, voxel size, and current scale.
            let (tree_clone, base_voxel_size, scale) = {
                let proc_geo = self.world.get::<&ProceduralGeometry>(entity).unwrap();
                let transform = self.world.get::<&Transform>(entity).unwrap();
                (proc_geo.tree.clone(), proc_geo.voxel_size, transform.scale)
            };

            // Free previous geometry allocation (if any) before re-voxelizing.
            // Without this, every re-voxelization leaks voxel slots and octree
            // entries until the pool is exhausted.
            let prev_spatial = self
                .world
                .get::<&Renderable>(entity)
                .ok()
                .and_then(|r| r.spatial.clone());
            if let Some(prev) = prev_spatial {
                let handle = rkp_core::OctreeHandle {
                    root_offset: prev.root_offset,
                    len: prev.len,
                    depth: prev.depth,
                    base_voxel_size: prev.base_voxel_size,
                };
                self.scene_mgr.deallocate_geometry(
                    &handle, prev.voxel_slot_start, prev.voxel_slot_count, &prev.brick_ids,
                );
            }

            let (aabb, voxel_size) = procedural_voxel_params(&tree_clone, base_voxel_size);
            let sdf_fn = |pos: glam::Vec3| -> (f32, u16) {
                let sample = rkp_procedural::sample_tree(&tree_clone, pos, voxel_size);
                (sample.distance, sample.material_id)
            };

            match self.scene_mgr.voxelize_sdf_fn(
                sdf_fn, &aabb, voxel_size, scene_id,
            ) {
                Some(result) => {
                    let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);

                    if let Ok(mut renderable) = self.world.get::<&mut Renderable>(entity) {
                        renderable.voxel_count = result.voxel_count;
                        renderable.spatial = Some(spatial);
                    }
                    if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        proc_geo.dirty = false;
                        proc_geo.last_evaluated_scale = scale;
                    }

                    self.geometry_dirty = true;
                    self.gpu_objects_dirty = true;
                }
                None => {
                    // Voxelization failed (pool full, empty result, etc.).
                    // Clear dirty to avoid retrying every frame.
                    if let Ok(mut proc_geo) = self.world.get::<&mut ProceduralGeometry>(entity) {
                        proc_geo.dirty = false;
                    }
                    self.console.warn(format!(
                        "Procedural voxelization failed. \
                         Voxel size: {voxel_size:.4}, AABB extent: {:.1}",
                        (aabb.max - aabb.min).length()
                    ));
                }
            }
        }
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
                        if let (Some(skel), Some(skin_data)) = (
                            self.world.get::<&crate::components::Skeleton>(entity).ok(),
                            self.scene_mgr.skinning_data(handle),
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
                let gpu_obj = crate::scene_sync::build_gpu_object(
                    &world_matrix,
                    &spatial.aabb,
                    &spatial_handle,
                    spatial.voxel_size,
                    renderable.material_id,
                    gpu_idx,
                    skinning,
                );
                if let Some(&scene_id) = self.entity_scene_ids.get(&entity) {
                    self.scene_id_to_gpu.insert(scene_id, gpu_idx);
                }
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

        // One-second heartbeat so we can tell whether the GPU object
        // the march shader sees actually carries skinning fields.
        // Diagnoses: "scatter dispatched but march renders rigid" —
        // if `is_skinned=1 count < skin_dispatches.len()` or
        // `dim_x>0 count < dispatches`, the GPU object didn't get
        // populated even though the plan succeeded.
        if self.frame_index % 60 == 0 && !self.skin_dispatches.is_empty() {
            let (mut skinned, mut with_dims) = (0u32, 0u32);
            let mut first_dims = [0u32; 3];
            for obj in &self.gpu_objects {
                if obj.is_skinned != 0 { skinned += 1; }
                if obj.bone_field_dim_x > 0 {
                    if with_dims == 0 {
                        first_dims = [obj.bone_field_dim_x, obj.bone_field_dim_y, obj.bone_field_dim_z];
                    }
                    with_dims += 1;
                }
            }
            eprintln!(
                "[skin] plans={} gpu_objs={} is_skinned={} with_dims={} first_dims={:?} bone_field_bytes={}",
                self.skin_dispatches.len(),
                self.gpu_objects.len(),
                skinned,
                with_dims,
                first_dims,
                self.skin_bone_field_bytes,
            );
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
                self.scene_mgr.release_asset(handle);
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
                self.scene_mgr.deallocate_geometry(&handle, slot_start, slot_count, &brick_ids);
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
        self.scene_mgr = RkpSceneManager::new(1_000_000);
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
                        match self.scene_mgr.acquire_asset(&full_path.to_string_lossy()) {
                            Ok((handle, info)) => {
                                let spatial = spatial_from_handle(&info.spatial, info.voxel_size, &info.aabb, info.leaf_attr_slot_start, info.leaf_attr_slot_count, Vec::new());
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
                            "box" => rkp_core::scene_node::SdfPrimitive::Box {
                                half_extents: glam::Vec3::from_array(obj.scale) * 0.5,
                            },
                            "sphere" => rkp_core::scene_node::SdfPrimitive::Sphere {
                                radius: obj.scale[0] * 0.5,
                            },
                            _ => continue,
                        };
                        let sid = self.next_scene_id;
                        self.next_scene_id += 1;
                        self.scene_mgr.voxelize_primitive(
                            &primitive, obj.material_id, 0.05, glam::Vec3::ONE, sid,
                        ).map(|result| {
                            let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb, result.leaf_attr_slot_start, result.leaf_attr_slot_count, result.brick_ids);
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
                }
            }

            if !left_pressed {
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
                let (rotation, scale) = self.world.get::<&crate::components::Transform>(selected)
                    .map(|t| {
                        let r = t.rotation;
                        let q = glam::Quat::from_euler(
                            glam::EulerRot::YXZ,
                            r.y.to_radians(), r.x.to_radians(), r.z.to_radians(),
                        );
                        (q, t.scale)
                    })
                    .unwrap_or((glam::Quat::IDENTITY, glam::Vec3::ONE));
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

        let all_nodes = self.scene_mgr.octree.data();

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
                rkp_physics::rigid_body::ColliderShape::Auto => {
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

            match cache.shape {
                ColliderShape::Box => {
                    let min = transform.position - cache.aabb_half;
                    let max = transform.position + cache.aabb_half;
                    verts.extend(rkp_render::wireframe::aabb_wireframe(min, max, color));
                }
                ColliderShape::Sphere => {
                    let r = cache.aabb_half.max_element();
                    verts.extend(rkp_render::wireframe::sphere_wireframe(transform.position, r, color));
                }
                ColliderShape::Capsule => {
                    let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
                    let hh = (cache.aabb_half.y - r).max(0.01);
                    let top = transform.position + glam::Vec3::new(0.0, hh, 0.0);
                    let bot = transform.position - glam::Vec3::new(0.0, hh, 0.0);
                    verts.extend(rkp_render::wireframe::sphere_wireframe(top, r, color));
                    verts.extend(rkp_render::wireframe::sphere_wireframe(bot, r, color));
                    for angle in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 3.0 * std::f32::consts::FRAC_PI_2] {
                        let offset = glam::Vec3::new(angle.cos() * r, 0.0, angle.sin() * r);
                        verts.push(rkp_render::LineVertex { position: (top + offset).to_array(), color });
                        verts.push(rkp_render::LineVertex { position: (bot + offset).to_array(), color });
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
    fn screen_to_ray(&self, px: f32, py: f32) -> (glam::Vec3, glam::Vec3) {
        let cam = self.build_camera_uniforms();
        let vp = glam::Mat4::from_cols_array_2d(&cam.view_proj);
        let inv_vp = vp.inverse();

        let ndc_x = (px / self.width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / self.height as f32) * 2.0;

        let near = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
        let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far - near).normalize();
        (self.camera.position, dir)
    }

    fn process_pick_result(&mut self) {
        let slice = self.pick_readback_buffer.slice(..256);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            if data.len() >= 8 {
                // Material texture (Rg32Uint): R = material ids, G = blend|object_id+1|color.
                let g_channel = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                // object_id+1 in bits 8-15. 0 means no geometry.
                let raw_id = (g_channel >> 8) & 0xFF;

                if raw_id > 0 {
                    let gpu_idx = (raw_id - 1) as usize;
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
                let parent_id = self.world.get::<&crate::components::Parent>(entity)
                    .ok()
                    .map(|p| p.parent_id);
                objs.push(crate::snapshot::SceneObjectInfo {
                    id: self.get_entity_uuid(entity),
                    name: meta.name.clone(),
                    parent_id,
                    is_camera,
                    is_light,
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
            project_dir,
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
        Some(crate::procedural_snapshot::build_procedural_snapshot(
            uuid,
            &proc_geo,
            self.selected_procedural_node,
            vs,
        ))
    }
}

// ── Procedural helpers ───────────────────────────────────────────────

/// Compute a safe AABB and voxel size for procedural voxelization.
///
/// Adds margin around the tight bounds and ensures the octree depth won't
/// exceed MAX_DEPTH (11). If the object is too large for the requested voxel
/// size, the voxel size is increased to fit.
fn procedural_voxel_params(tree: &rkp_procedural::ProceduralObject, base_voxel_size: f32) -> (rkp_core::Aabb, f32) {
    let tight = rkp_procedural::compute_bounds(tree);

    // Add margin for boundary sampling (same approach as voxelize_primitive).
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
        NodeKind::Sphere(p) => match param_name {
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Box(p) => match param_name {
            "half_extents" => { if let Some(v) = parse_vec3(value) { p.half_extents = v; } true }
            "rounding" => { p.rounding = value.parse().unwrap_or(p.rounding); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            "color" => { if let Some(v) = parse_vec3(value) { p.color = v; } true }
            _ => false,
        },
        NodeKind::Capsule(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Cylinder(p) => match param_name {
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "radius" => { p.radius = value.parse().unwrap_or(p.radius); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Torus(p) => match param_name {
            "major_radius" => { p.major_radius = value.parse().unwrap_or(p.major_radius); true }
            "minor_radius" => { p.minor_radius = value.parse().unwrap_or(p.minor_radius); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Plane(p) => match param_name {
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
            _ => false,
        },
        NodeKind::Ramp(p) => match param_name {
            "half_length" => { p.half_length = value.parse().unwrap_or(p.half_length); true }
            "half_height" => { p.half_height = value.parse().unwrap_or(p.half_height); true }
            "half_width" => { p.half_width = value.parse().unwrap_or(p.half_width); true }
            "material_id" => { p.material_id = value.parse().unwrap_or(p.material_id); true }
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

        // 1b2. Re-evaluate dirty procedural objects.
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
        }

        // 1d. Advance skeletal animations. Runs every frame in both edit
        // and play modes so animated characters preview correctly in the
        // editor. Uses the same fixed 60 Hz step as gameplay for now.
        let anim_dt = 1.0 / 60.0;
        if crate::animation::tick(&mut state.world, anim_dt) {
            state.gpu_objects_dirty = true;
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

        // 3. Update gizmo hover + drag.
        state.update_gizmo();

        // 4. Render frame.
        let (pixels, w, h) = state.render_frame();

        // 5. Deliver frame to client.
        frame_callback(&pixels, w, h);

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
