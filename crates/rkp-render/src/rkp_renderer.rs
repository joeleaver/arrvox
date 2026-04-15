//! Shared GPU state + orchestration for per-viewport rendering.
//!
//! [`RkpRenderer`] holds the scene-wide buffers (scene, atmosphere LUTs,
//! shade params, lights, materials) plus the GPU profiler. The
//! resolution-coupled passes (march, shadow_trace, ssao, shade,
//! volumetric, god_rays) live in [`ViewportRenderer`] so each viewport
//! can render at its own resolution without clashing on shared textures.
//!
//! The caller (rkp-editor) drives each frame:
//! 1. Build RkpGpuObjects from ECS/scene state.
//! 2. Call `upload_frame(objects)` — cheap, every frame.
//! 3. Call `upload_geometry(...)` — only when geometry changes.
//! 4. For each visible viewport:
//!    - `vr.upload_camera(queue, cam_uniforms)` and
//!      `vr.refresh_bindings(device, renderer)` to catch up with any
//!      shared-buffer reallocations.
//!    - `renderer.render_to(encoder, queue, vr, ...)`.

use crate::rkp_scene::{RkpScene, GeometryUpload, FrameUpload};
use crate::rkp_shade::{ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_volumetric::{VolumetricParams, CloudParams};
use crate::rkp_atmosphere::RkpAtmospherePass;
use wgpu_profiler::GpuProfiler;

/// The RKIPatch renderer — shared state only. Per-viewport passes live
/// in [`ViewportRenderer`].
pub struct RkpRenderer {
    /// Scene GPU buffers (shared across viewports).
    pub scene: RkpScene,
    /// Atmosphere LUTs (transmittance + multi-scattering + sky-view + AP).
    /// Computed from sun/camera, consumed by shade.
    pub atmosphere: RkpAtmospherePass,
    /// Scene-wide shade params (env-colors, ambient intensity, etc.).
    pub shade_params_buffer: wgpu::Buffer,
    /// Scene-wide light list (directional + point + spot).
    pub lights_buffer: wgpu::Buffer,
    /// Scene-wide material palette.
    pub materials_buffer: wgpu::Buffer,
    /// Bumps when `lights_buffer` or `materials_buffer` reallocates so
    /// ViewportRenderers know to rebuild their march + shade bindings.
    lights_materials_epoch: u64,
    /// Device for buffer operations.
    pub device: wgpu::Device,
    /// GPU profiler (wgpu-profiler).
    pub profiler: GpuProfiler,
    timestamp_period: f32,
}

impl RkpRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, _width: u32, _height: u32) -> Self {
        let scene = RkpScene::new(device);

        let profiler = GpuProfiler::new(device, wgpu_profiler::GpuProfilerSettings {
            enable_timer_queries: true,
            enable_debug_groups: true,
            max_num_pending_frames: 4,
        }).expect("failed to create GPU profiler");
        let timestamp_period = queue.get_timestamp_period();

        let default_params = ShadeParams { num_lights: 1, ..ShadeParams::default() };
        let default_light = GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [1.0, 0.95, 0.9, 2.0],
            direction: [-0.5, -0.8, -0.3, 0.0],
            params: [0.0; 4],
        };
        let default_material = GpuMaterial {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emission_strength: 0.0,
            opacity: 1.0,
        };

        let shade_params_buffer = Self::create_init_buffer(device, "rkp_shade_params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_params));
        let lights_buffer = Self::create_init_buffer(device, "rkp_shade_lights",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_light));
        let materials_buffer = Self::create_init_buffer(device, "rkp_shade_materials",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_material));

        let atmosphere = RkpAtmospherePass::new(device);

        Self {
            scene, atmosphere,
            shade_params_buffer, lights_buffer, materials_buffer,
            lights_materials_epoch: 0,
            device: device.clone(),
            profiler, timestamp_period,
        }
    }

    /// Current lights/materials epoch — ViewportRenderers compare against
    /// this to detect when their march/shade bindings have gone stale
    /// (shared buffer reallocated under them).
    pub fn lights_materials_epoch(&self) -> u64 {
        self.lights_materials_epoch
    }

    pub fn upload_geometry(&mut self, queue: &wgpu::Queue, data: &GeometryUpload) {
        self.scene.upload_geometry(&self.device, queue, data);
    }

    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        self.scene.upload_frame(&self.device, queue, data);
    }

    /// Render one frame into `viewport`. Dispatches into the VR's own
    /// per-resolution passes (march → shadow → ssao → shade → vol →
    /// god_rays → bloom → bloom_composite → tone_map) + copies tone_map's
    /// LDR output into the VR's composite texture. Caller still owns the
    /// optional wireframe overlay + readback copy + submit.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        object_count: u32,
        shadow_steps: u32,
        num_lights: u32,
        screen_aabbs: &[u8],
        atmo_frame_params: &crate::rkp_atmosphere::AtmosphereFrameParams,
    ) {
        // Upload per-viewport tile-cull screen-space AABBs into the VR's march.
        viewport.march.upload_screen_aabbs(queue, screen_aabbs);

        // 0. Atmosphere LUTs (precomputed + per-frame, shared — happens once
        // per viewport because the sun/camera change per render).
        self.atmosphere.dispatch_if_dirty(encoder);
        {
            let q = self.profiler.begin_query("atmo", encoder);
            self.atmosphere.dispatch_per_frame(encoder, queue, atmo_frame_params);
            self.profiler.end_query(encoder, q);
        }

        // 1. Octree ray march → G-buffer (primary visibility only).
        viewport.march.clear_stats(encoder);
        {
            let q = self.profiler.begin_query("march", encoder);
            viewport.march.dispatch(
                encoder, queue, &viewport.scene_bind_group,
                object_count, viewport.width, viewport.height, 0,
                shadow_steps, num_lights, None,
            );
            self.profiler.end_query(encoder, q);
        }

        // 1b. Half-res shadow trace. Uses march's params bind group.
        if let Some(params_bg) = viewport.march.params_bind_group() {
            let q = self.profiler.begin_query("shadow", encoder);
            viewport.shadow_trace.dispatch(encoder, &viewport.scene_bind_group, params_bg);
            self.profiler.end_query(encoder, q);
        }
        viewport.march.copy_stats(encoder);

        // 2. SSAO (half-res).
        {
            let q = self.profiler.begin_query("ssao", encoder);
            viewport.ssao.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 3. Deferred PBR shading.
        {
            let q = self.profiler.begin_query("shade", encoder);
            viewport.shade.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4. Volumetric march (half-res) + composite (full-res).
        {
            let q = self.profiler.begin_query("vol", encoder);
            viewport.volumetric.dispatch_march(encoder);
            viewport.volumetric.dispatch_composite(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4b. Screen-space god rays.
        {
            let q = self.profiler.begin_query("god_rays", encoder);
            viewport.god_rays.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 5. Bloom + bloom composite + tone_map (VR-level).
        {
            let q = self.profiler.begin_query("bloom", encoder);
            viewport.bloom.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }
        {
            let q = self.profiler.begin_query("bloom_composite", encoder);
            viewport.bloom_composite.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }
        {
            let q = self.profiler.begin_query("tone_map", encoder);
            viewport.tone_map.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // Copy LDR into composite for any overlay passes (wireframe) the caller runs.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: viewport.tone_map.ldr_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &viewport.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: viewport.width,
                height: viewport.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn resolve_profiler_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.resolve_queries(encoder);
    }

    pub fn end_profiler_frame(&mut self, frame_idx: u64, _width: u32, _height: u32) {
        // Note: `march.read_stats` used to live here but march is now per-VR.
        // The editor's profiler panel should query per-VR stats; for now we
        // just flush the frame without the march-specific stat readback.
        if let Err(e) = self.profiler.end_frame() {
            if frame_idx > 10 {
                eprintln!("[profiler] end_frame: {e}");
            }
        }
        if let Some(results) = self.profiler.process_finished_frame(self.timestamp_period) {
            if frame_idx % 60 == 0 && frame_idx > 0 {
                eprint!("[gpu]");
                for r in &results {
                    let ms = r.time.as_ref().map(|t| (t.end - t.start) * 1000.0).unwrap_or(0.0);
                    eprint!(" {}={:.2}ms", r.label, ms);
                }
                eprintln!();
            }
        }
    }

    pub fn update_shade_params(&self, queue: &wgpu::Queue, params: &ShadeParams) {
        queue.write_buffer(&self.shade_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_lights(&mut self, queue: &wgpu::Queue, lights: &[GpuLight]) {
        let data: &[u8] = bytemuck::cast_slice(lights);
        let needed = data.len() as u64;
        if needed > self.lights_buffer.size() {
            self.lights_buffer = Self::create_init_buffer(
                &self.device,
                "rkp_shade_lights",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                data,
            );
            self.lights_materials_epoch += 1;
        } else {
            queue.write_buffer(&self.lights_buffer, 0, data);
        }
    }

    pub fn update_materials(&mut self, queue: &wgpu::Queue, materials: &[GpuMaterial]) {
        let data: &[u8] = bytemuck::cast_slice(materials);
        let needed = data.len() as u64;

        if needed > self.materials_buffer.size() {
            self.materials_buffer = Self::create_init_buffer(
                &self.device,
                "rkp_shade_materials",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                data,
            );
            self.lights_materials_epoch += 1;
        } else {
            queue.write_buffer(&self.materials_buffer, 0, data);
        }
    }

    fn create_init_buffer(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages, data: &[u8]) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().copy_from_slice(data);
        buf.unmap();
        buf
    }
}
