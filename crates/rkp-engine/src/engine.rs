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
        }
    } else {
        SpatialData {
            root_offset: 0, len: 0, depth: 0, base_voxel_size: voxel_size,
            aabb: *aabb, voxel_size,
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
    gbuffer: rkf_render::GBuffer,
    tone_map: rkf_render::ToneMapPass,

    // Scene management (CPU)
    scene_mgr: RkpSceneManager,

    // Input + Camera
    input_system: rkf_runtime::input::InputSystem,
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

    /// Console log buffer.
    console: crate::console::ConsoleLog,
    /// Gameplay dylib loader (hot-reload).
    gameplay_loader: crate::gameplay_loader::GameplayLoader,
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

    // Double-buffered readback: copy to one buffer this frame, read from the other
    // (which completed last frame). Avoids blocking CPU waiting for GPU.
    readback_buffers: [wgpu::Buffer; 2],
    readback_index: usize, // which buffer to copy INTO this frame
    readback_ready: bool,  // false on first frame (no previous data yet)

    // Wireframe overlay
    wireframe_pass: rkf_render::WireframePass,
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
}

impl EngineState {
    fn new(config: &EngineConfig) -> Self {
        let ctx = rkf_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let gbuffer = rkf_render::GBuffer::new(&device, width, height);
        let mut renderer = RkpRenderer::new(&device, &queue, width, height);

        // Wire G-buffer into renderer.
        renderer.set_gbuffer(&gbuffer);

        // Tone mapping: HDR volumetric output → LDR (Rgba8Unorm).
        let tone_map = rkf_render::ToneMapPass::new(
            &device,
            &renderer.volumetric.output_view,
            width,
            height,
        );

        let scene_mgr = RkpSceneManager::new(1_000_000);

        // Input system with default action map.
        let mut input_system = rkf_runtime::input::InputSystem::new();
        input_system.add_map(crate::camera::default_action_map());
        input_system.set_active_map("editor");
        let camera_control = CameraControlState::default();

        // Double-buffered readback — avoids blocking CPU for GPU completion.
        let readback_buffers = [
            Self::create_readback_buffer(&device, width, height),
            Self::create_readback_buffer(&device, width, height),
        ];

        // Wireframe pass for gizmo overlay.
        let wireframe_pass = rkf_render::WireframePass::new(&device, rkf_render::LDR_FORMAT);

        // Composite texture: LDR + wireframes. Needs RENDER_ATTACHMENT for wireframe draw.
        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: rkf_render::LDR_FORMAT,
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
            material_lib: crate::material_library::MaterialLibrary::new(),
            selected_material: None,
            selected_model: None,
            environment: crate::environment::EnvironmentSettings::default(),
            environment_dirty: true, // upload on first frame
            console: crate::console::ConsoleLog::new(),
            gameplay_loader: crate::gameplay_loader::GameplayLoader::new(),
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
            self.renderer.update_shade_params(&self.queue, &shade_params);
            self.renderer.update_lights(&self.queue, &gpu_lights);
            self.num_lights_cache = shade_params.num_lights;

            if self.environment_dirty {
                self.tone_map.set_exposure(&self.queue, self.environment.exposure);
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

        // 2. Upload per-frame data (objects + camera).
        let cam_uniforms = self.build_camera_uniforms();
        let frame = FrameUpload {
            objects: &self.gpu_objects,
            camera: &cam_uniforms,
        };
        self.renderer.upload_frame(&self.queue, &frame);

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

        self.renderer.render(&mut encoder, &self.queue, object_count, self.width, self.height, shadow_steps, num_lights, screen_aabbs_bytes);

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

        // 5. Tone mapping: HDR → LDR (Rgba8Unorm).
        self.tone_map.dispatch(&mut encoder);

        // 6. Copy LDR to composite texture, draw gizmo wireframes, readback.
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

        let t_post = frame_start.elapsed();

        // 8. Submit GPU work.
        self.queue.submit(std::iter::once(encoder.finish()));

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

        let t_readback = frame_start.elapsed();

        // Log timing every 60 frames.
        if self.frame_index % 60 == 0 && self.frame_index > 0 {
            eprintln!(
                "[perf] cpu_setup={:.1}ms upload={:.1}ms encode={:.1}ms post={:.1}ms submit={:.1}ms readback={:.1}ms total={:.1}ms",
                t_cpu_setup.as_secs_f64() * 1000.0,
                (t_upload - t_cpu_setup).as_secs_f64() * 1000.0,
                (t_encode - t_upload).as_secs_f64() * 1000.0,
                (t_post - t_encode).as_secs_f64() * 1000.0,
                (t_submit - t_post).as_secs_f64() * 1000.0,
                (t_readback - t_submit).as_secs_f64() * 1000.0,
                t_readback.as_secs_f64() * 1000.0,
            );
        }

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
            prev_vp: view_proj.to_cols_array_2d(),
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
                    self.gbuffer = rkf_render::GBuffer::new(&self.device, width, height);
                    self.renderer.resize(width, height);
                    self.renderer.set_gbuffer(&self.gbuffer);
                    self.tone_map = rkf_render::ToneMapPass::new(
                        &self.device,
                        &self.renderer.volumetric.output_view,
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
                        format: rkf_render::LDR_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    self.composite_view = self.composite_texture.create_view(&Default::default());
                    eprintln!("[RkpEngine] resized to {}x{}", width, height);
                }
            }

            EngineCommand::SpawnPrimitive { name } => {
                use crate::components::*;
                let name = self.unique_name(&name);
                let primitive = rkf_core::scene_node::SdfPrimitive::Box {
                    half_extents: glam::Vec3::splat(0.5),
                };
                let scene_id = self.next_scene_id;
                self.next_scene_id += 1;
                let result = self.scene_mgr.voxelize_primitive(
                    &primitive, 0, 0.05, glam::Vec3::ONE, scene_id,
                );
                if let Some(result) = result {
                    let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb);
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
                match self.scene_mgr.load_rkp(&path, scene_id) {
                    Ok(result) => {
                        let raw_name = Self::display_name_from_path(&path);
                        let name = self.unique_name(&raw_name);
                        let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb);
                        let entity = self.world.spawn((
                            Transform::default(),
                            EditorMetadata { name: name.clone() },
                            Renderable {
                                asset_path: Some(path.clone()),
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
                    self.console.info(format!("Loaded '{name}': {} voxels", result.voxel_count));
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
                        let scene_path = project_dir.join(format!("scenes/{}.rkscene", project.default_scene));
                        if scene_path.exists() {
                            self.load_scene_from_file(&scene_path);
                        }
                        self.project_dir = Some(project_dir);
                        self.project_path = Some(path);
                        self.scene_path = Some(scene_path);
                        self.project_name = project.name;
                        self.project_loaded = true;
                        self.project_dirty = true;
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
            }

            EngineCommand::SaveProject => {
                if let (Some(project_path), Some(_project_dir)) = (&self.project_path, &self.project_dir) {
                    let project = crate::project::ProjectFile {
                        name: self.project_name.clone(),
                        default_scene: "default".to_string(),
                        recent_scenes: Vec::new(),
                    };
                    if let Err(e) = crate::project::save_project(&project, project_path) {
                        eprintln!("[RkpEngine] save project failed: {e}");
                    }
                }
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
                self.selected_material = material_id;
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
                let profile = crate::import_profile::ImportProfile::load_or_default(&source);
                let output = crate::import_worker::rkp_output_path(&source);
                eprintln!("[RkpEngine] re-importing {} → {}", source.display(), output.display());
                self.import_worker.submit(crate::import_worker::ImportRequest {
                    source_path: source,
                    output_path: output,
                    config: profile.to_import_config(),
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
                    "sky_color_top" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_top = v; }
                    }
                    "sky_color_horizon" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sky_color_horizon = v; }
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
                    "sun_color" => {
                        if let Ok(v) = serde_json::from_str::<[f32; 3]>(&value) { env.sun_color = v; }
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
                    self.console.info("Play mode started");
                }
            }

            EngineCommand::PlayStop => {
                if let Some(play) = self.play_state.take() {
                    play.stop(&mut self.world);
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

    /// Try to find and load the gameplay dylib from the workspace target directory.
    fn try_load_gameplay_dylib(&mut self) {
        // Look for the cdylib in the standard cargo output locations.
        let candidates = [
            // Release build (most common for the editor)
            std::path::PathBuf::from("target/release/librkp_gameplay.so"),
            std::path::PathBuf::from("target/release/librkp_gameplay.dylib"),
            std::path::PathBuf::from("target/release/rkp_gameplay.dll"),
            // Debug build
            std::path::PathBuf::from("target/debug/librkp_gameplay.so"),
            std::path::PathBuf::from("target/debug/librkp_gameplay.dylib"),
            std::path::PathBuf::from("target/debug/rkp_gameplay.dll"),
        ];

        for path in &candidates {
            if path.exists() {
                match self.gameplay_loader.load(path) {
                    Ok(entries) => {
                        let names: Vec<&str> = entries.iter().map(|e| e.name).collect();
                        self.console.info(format!(
                            "Loaded gameplay dylib: {} components ({})",
                            entries.len(),
                            names.join(", "),
                        ));
                        // Register gameplay entries in the component registry.
                        for &entry in entries {
                            self.registry.register_gameplay(entry);
                        }
                        self.scene_dirty = true;
                        return;
                    }
                    Err(e) => {
                        self.console.error(format!("Failed to load gameplay dylib: {e}"));
                    }
                }
            }
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
        let all_nodes = self.scene_mgr.octree.data();
        let mut leaf_slots = Vec::new();
        collect_leaf_slots(all_nodes, spatial.root_offset as usize, &mut leaf_slots);

        // Count material IDs across all leaf voxels (with bounds check).
        let pool_size = self.scene_mgr.voxel_pool.allocated_count();
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
        for slot in leaf_slots {
            if slot >= pool_size {
                continue; // stale or invalid slot — skip
            }
            let voxel = self.scene_mgr.voxel_pool.get(slot);
            if voxel.opacity_f32() > 0.01 {
                *counts.entry(voxel.material_id()).or_insert(0) += 1;
            }
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

        let pool_size = self.scene_mgr.voxel_pool.allocated_count();
        let mut count = 0u32;
        for slot in leaf_slots {
            if slot >= pool_size { continue; }
            let voxel = self.scene_mgr.voxel_pool.get(slot);
            let primary = voxel.material_id();
            let secondary = voxel.secondary_material_id();
            let mut changed = false;

            if primary == from_material {
                let mut v = *voxel;
                v.set_material_id(to_material);
                *self.scene_mgr.voxel_pool.get_mut(slot) = v;
                changed = true;
            }
            if secondary == from_material {
                let mut v = *self.scene_mgr.voxel_pool.get(slot);
                v.set_secondary_material_id(to_material);
                *self.scene_mgr.voxel_pool.get_mut(slot) = v;
                changed = true;
            }
            if changed {
                count += 1;
            }
        }
        count
    }

    fn poll_import_completions(&mut self) {
        let completions = self.import_worker.poll_completions();
        for completion in completions {
            match completion.result {
                Ok(result) => {
                    let name = completion.source_path.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    self.console.info(format!(
                        "Import complete: {name} ({} voxels)",
                        result.total_bricks,
                    ));
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
                let gpu_obj = crate::scene_sync::build_gpu_object(
                    &world_matrix,
                    &spatial.aabb,
                    &spatial_handle,
                    spatial.voxel_size,
                    renderable.material_id,
                    gpu_idx,
                );
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

        // Deallocate octree if entity has spatial data.
        if let Ok(renderable) = self.world.get::<&crate::components::Renderable>(entity) {
            if let Some(ref spatial) = renderable.spatial {
                let handle = rkp_core::OctreeHandle {
                    root_offset: spatial.root_offset,
                    len: spatial.len,
                    depth: spatial.depth,
                    base_voxel_size: spatial.base_voxel_size,
                };
                self.scene_mgr.octree.deallocate(handle);
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
                if let Some(ref env_state) = scene.environment {
                    self.environment = env_state.to_settings();
                    self.environment_dirty = true;
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
                        match self.scene_mgr.load_rkp(&full_path.to_string_lossy(), sid) {
                            Ok(result) => {
                                let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb);
                                let e = self.world.spawn((transform, meta, Renderable {
                                    asset_path: Some(asset_path.clone()),
                                    material_id: obj.material_id,
                                    voxel_count: result.voxel_count,
                                    spatial: Some(spatial),
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
                        self.scene_mgr.voxelize_primitive(
                            &primitive, obj.material_id, 0.05, glam::Vec3::ONE, sid,
                        ).map(|result| {
                            let spatial = spatial_from_handle(&result.spatial, result.voxel_size, &result.aabb);
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
            environment: Some(crate::scene_io::EnvironmentState::from_settings(&self.environment)),
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
                let start_point = match self.gizmo.hovered_axis {
                    crate::gizmo::GizmoAxis::X | crate::gizmo::GizmoAxis::Y | crate::gizmo::GizmoAxis::Z => {
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
                rkf_physics::rigid_body::ColliderShape::Auto => {
                    if let Some(ref sp) = spatial {
                        let (coords, cell_size) = crate::play_mode::build_coarse_collider(
                            all_nodes,
                            &self.scene_mgr.voxel_pool,
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

        let models = if self.models_dirty {
            self.models_dirty = false;
            Some(self.available_models.clone())
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
            environment: if self.frame_index <= 1 || self.environment_dirty {
                Some(self.environment.clone())
            } else {
                None
            },
            console_entries: self.console.drain_new(),
        }
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

    // Try to load gameplay dylib from the standard build output.
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

        // 1b. Process file watcher events + import completions + gameplay reload.
        state.process_file_events();
        state.poll_import_completions();
        state.check_gameplay_reload();

        // 1c. Step physics if in play mode.
        if let Some(ref mut play) = state.play_state {
            let physics_dt = 1.0 / 60.0;
            if play.step(physics_dt, &mut state.world) {
                state.gpu_objects_dirty = true;
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
