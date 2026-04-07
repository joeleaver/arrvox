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

use crate::command::EngineCommand;
use crate::snapshot::StateUpdate;

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

    // Camera
    camera: CameraState,

    // GPU objects (built from scene each frame)
    gpu_objects: Vec<RkpGpuObject>,
    /// Object names for UI display (parallel to gpu_objects).
    object_names: Vec<String>,
    /// Currently selected entity.
    selected_entity: Option<uuid::Uuid>,

    // Geometry dirty flag
    geometry_dirty: bool,
    /// Scene structure changed — push objects list to UI.
    scene_dirty: bool,

    // Frame counter
    frame_index: u64,

    // Render dimensions
    width: u32,
    height: u32,

    // Readback buffer (reads from tone_map.ldr_texture, Rgba8Unorm)
    readback_buffer: wgpu::Buffer,
}

impl EngineState {
    fn new(config: &EngineConfig) -> Self {
        let ctx = rkf_render::RenderContext::new_headless();
        let device = ctx.device;
        let queue = ctx.queue;

        let width = config.width;
        let height = config.height;

        let gbuffer = rkf_render::GBuffer::new(&device, width, height);
        let mut renderer = RkpRenderer::new(&device, width, height);

        // Wire G-buffer into renderer.
        renderer.set_gbuffer(
            &gbuffer.position_view,
            &gbuffer.normal_view,
            &gbuffer.material_view,
        );

        // Tone mapping: HDR shade output → LDR (Rgba8Unorm).
        let tone_map = rkf_render::ToneMapPass::new(
            &device,
            &renderer.shade.output_view,
            width,
            height,
        );

        let scene_mgr = RkpSceneManager::new(1_000_000);

        // Readback buffer — reads from tone_map.ldr_texture (Rgba8Unorm).
        let readback_buffer = Self::create_readback_buffer(&device, width, height);

        Self {
            device,
            queue,
            renderer,
            gbuffer,
            tone_map,
            scene_mgr,
            camera: CameraState::default(),
            gpu_objects: Vec::new(),
            object_names: Vec::new(),
            selected_entity: None,
            geometry_dirty: false,
            scene_dirty: false,
            frame_index: 0,
            width,
            height,
            readback_buffer,
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
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rkp frame"),
        });

        // 1. Upload geometry if dirty.
        if self.geometry_dirty {
            let geo = self.scene_mgr.geometry_upload();
            self.renderer.upload_geometry(&self.queue, &geo);
            self.geometry_dirty = false;
        }

        // 2. Upload per-frame data (objects + camera).
        let cam_uniforms = self.build_camera_uniforms();
        let frame = FrameUpload {
            objects: &self.gpu_objects,
            camera: &cam_uniforms,
        };
        self.renderer.upload_frame(&self.queue, &frame);

        // 3. Upload face instances.
        let faces = self.scene_mgr.pending_faces().to_vec();
        self.renderer.upload_faces(&mut encoder, &faces);

        // 4. Render: raster → shadow/AO → shade.
        let shadow_params = rkp_render::rkp_shadow_ao::ShadowAoParams::default();
        self.renderer.render(&mut encoder, &self.gbuffer, &self.queue, &shadow_params);

        // 5. Tone mapping: HDR → LDR (Rgba8Unorm).
        self.tone_map.dispatch(&mut encoder);

        // 6. Copy LDR output to readback buffer.
        let padded_row = (self.width * 4 + 255) & !255;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.tone_map.ldr_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
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

        // 7. Submit GPU work.
        self.queue.submit(std::iter::once(encoder.finish()));

        // 8. Map readback buffer and extract pixels.
        let pixels = self.map_readback();

        self.frame_index += 1;

        (pixels, self.width, self.height)
    }

    fn map_readback(&self) -> Vec<u8> {
        let w = self.width;
        let h = self.height;
        let padded_row = (w * 4 + 255) & !255;

        let buffer_slice = self.readback_buffer.slice(..);
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
            self.readback_buffer.unmap();
        }

        rgba8
    }

    fn build_camera_uniforms(&self) -> rkp_render::rkp_scene::CameraUniforms {
        let yaw_rad = self.camera.yaw.to_radians();
        let pitch_rad = self.camera.pitch.to_radians();

        let forward = glam::Vec3::new(
            yaw_rad.sin() * pitch_rad.cos(),
            pitch_rad.sin(),
            -yaw_rad.cos() * pitch_rad.cos(),
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
                    self.renderer.set_gbuffer(
                        &self.gbuffer.position_view,
                        &self.gbuffer.normal_view,
                        &self.gbuffer.material_view,
                    );
                    self.tone_map = rkf_render::ToneMapPass::new(
                        &self.device,
                        &self.renderer.shade.output_view,
                        width,
                        height,
                    );
                    self.readback_buffer = Self::create_readback_buffer(&self.device, width, height);
                }
            }

            EngineCommand::SpawnPrimitive { name } => {
                let primitive = rkf_core::scene_node::SdfPrimitive::Box {
                    half_extents: glam::Vec3::splat(0.5),
                };
                let obj_id = self.gpu_objects.len() as u32;
                let result = self.scene_mgr.voxelize_primitive(
                    &primitive, 0, 0.05, glam::Vec3::ONE, obj_id,
                );
                if let Some(result) = result {
                    let gpu_obj = crate::scene_sync::build_gpu_object(
                        &glam::Mat4::IDENTITY,
                        &result.aabb,
                        &result.spatial,
                        result.voxel_size,
                        0,
                        obj_id,
                    );
                    self.gpu_objects.push(gpu_obj);
                    self.object_names.push(name.clone());
                    self.geometry_dirty = true;
                    self.scene_dirty = true;
                    eprintln!("[RkpEngine] spawned primitive '{name}': {} voxels", result.voxel_count);
                }
            }

            EngineCommand::LoadAsset { path, .. } => {
                let object_id = self.gpu_objects.len() as u32;
                match self.scene_mgr.load_rkp(&path, object_id) {
                    Ok(result) => {
                        let gpu_obj = crate::scene_sync::build_gpu_object(
                            &glam::Mat4::IDENTITY,
                            &result.aabb,
                            &result.spatial,
                            result.voxel_size,
                            0,
                            object_id,
                        );
                        // Use filename as display name.
                        let name = std::path::Path::new(&path)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| path.clone());
                        self.gpu_objects.push(gpu_obj);
                        self.object_names.push(name);
                        self.geometry_dirty = true;
                        self.scene_dirty = true;
                        eprintln!("[RkpEngine] loaded asset '{path}': {} voxels", result.voxel_count);
                    }
                    Err(e) => {
                        eprintln!("[RkpEngine] failed to load '{path}': {e}");
                    }
                }
            }

            EngineCommand::SelectEntity { entity_id } => {
                self.selected_entity = Some(entity_id);
            }

            _ => {
                eprintln!("[RkpEngine] unhandled command: {cmd:?}");
            }
        }

        true
    }

    fn build_state_update(&mut self, frame_time: Duration) -> StateUpdate {
        let fps = if frame_time.as_secs_f32() > 0.0 {
            1.0 / frame_time.as_secs_f32()
        } else {
            0.0
        };

        let objects = if self.scene_dirty {
            self.scene_dirty = false;
            Some(
                self.object_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| crate::snapshot::SceneObjectInfo {
                        id: uuid::Uuid::from_u128(i as u128),
                        name: name.clone(),
                        parent_id: None,
                        is_camera: false,
                        is_light: false,
                    })
                    .collect(),
            )
        } else {
            None
        };

        StateUpdate {
            fps,
            gpu_object_count: self.gpu_objects.len() as u32,
            camera_position: self.camera.position,
            play_mode: false,
            selected_entity: self.selected_entity,
            objects,
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
    eprintln!("[RkpEngine] starting tick loop ({}x{})", config.width, config.height);

    let mut state = EngineState::new(&config);

    loop {
        let frame_start = Instant::now();

        // 1. Drain command queue.
        while let Ok(cmd) = cmd_rx.try_recv() {
            if !state.process_command(cmd) {
                eprintln!("[RkpEngine] shutdown");
                return;
            }
        }

        // 2. Render frame.
        let (pixels, w, h) = state.render_frame();

        // 3. Deliver frame to client.
        frame_callback(&pixels, w, h);

        // 4. Push state to client.
        let frame_time = frame_start.elapsed();
        let update = state.build_state_update(frame_time);
        state_callback(&update);

        // 5. Frame pacing — target ~60 FPS.
        let target = Duration::from_micros(16_667);
        let elapsed = frame_start.elapsed();
        if elapsed < target {
            std::thread::sleep(target - elapsed);
        }
    }
}
