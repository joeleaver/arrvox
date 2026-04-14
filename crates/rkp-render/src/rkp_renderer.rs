//! RKIPatch renderer — owns the GPU scene and all render passes.
//!
//! Phase 4 pipeline: triangle raster → SSAO → shade → volumetric → god rays.
//! The compute-era ray march is gone; every renderable object comes from the
//! mesh pool.
//!
//! Caller (rkp-editor) drives the renderer each frame:
//! 1. Build `RkpGpuObject`s from ECS/scene state
//! 2. Upload extracted meshes via `renderer.mesh_pool.upload_mesh` / `.flush`
//! 3. Call `upload_frame(objects, camera)` every frame (cheap)
//! 4. Call `render(encoder, gbuffer, mesh_draws, ...)`

use crate::rkp_scene::{RkpScene, FrameUpload};
use crate::rkp_ssao::RkpSsaoPass;
use crate::rkp_shade::{RkpShadePass, ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_volumetric::{RkpVolumetricPass, VolumetricParams, CloudParams};
use crate::rkp_atmosphere::RkpAtmospherePass;
use crate::rkp_god_rays::RkpGodRayPass;
use crate::mesh_pool::MeshPool;
use crate::triangle_gbuffer::{MeshDraw, TriangleGBufferPass};
use wgpu_profiler::GpuProfiler;

/// The complete RKIPatch renderer.
pub struct RkpRenderer {
    /// Scene GPU buffers (objects + camera).
    pub scene: RkpScene,
    /// Triangle rasterization pass — primary visibility for mesh objects.
    pub triangle: TriangleGBufferPass,
    /// GPU vertex/index storage for extracted marching-cubes meshes.
    pub mesh_pool: MeshPool,
    /// Half-res screen-space ambient occlusion compute pass.
    pub ssao: RkpSsaoPass,
    /// Deferred PBR shading compute pass.
    pub shade: RkpShadePass,
    /// Default light/material buffers.
    shade_params_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    /// Atmosphere LUTs (transmittance + multi-scattering).
    pub atmosphere: RkpAtmospherePass,
    /// Volumetric rendering pass (fog + dust + clouds).
    pub volumetric: RkpVolumetricPass,
    /// Screen-space god rays.
    pub god_rays: RkpGodRayPass,
    /// Device for buffer operations.
    device: wgpu::Device,
    /// GPU profiler (wgpu-profiler).
    pub profiler: GpuProfiler,
    timestamp_period: f32,
    width: u32,
    height: u32,
}

impl RkpRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let scene = RkpScene::new(device);
        let triangle = TriangleGBufferPass::new(device, &scene.bind_group_layout);
        let mesh_pool = MeshPool::new(device);

        let profiler = GpuProfiler::new(device, wgpu_profiler::GpuProfilerSettings {
            enable_timer_queries: true,
            enable_debug_groups: true,
            max_num_pending_frames: 4,
        }).expect("failed to create GPU profiler");
        let timestamp_period = queue.get_timestamp_period();
        let ssao = RkpSsaoPass::new(device, queue, width, height);
        let mut shade = RkpShadePass::new(device, width, height);

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

        shade.set_shade_data(device, &shade_params_buffer, &lights_buffer, &materials_buffer);
        shade.set_camera(device, &scene.camera_buffer);
        shade.set_atmosphere_luts(device, &atmosphere.transmittance_view, &atmosphere.multiscatter_view, &atmosphere.lut_sampler, &atmosphere.sky_view_view, &atmosphere.ap_view);

        let volumetric = RkpVolumetricPass::new(device, width, height);
        let mut god_rays = RkpGodRayPass::new(device, width, height);
        god_rays.set_input(device, &volumetric.output_view);

        Self {
            scene, triangle, mesh_pool, ssao, shade, atmosphere, volumetric, god_rays,
            shade_params_buffer, lights_buffer, materials_buffer,
            device: device.clone(),
            profiler, timestamp_period,
            width, height,
        }
    }

    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        self.scene.upload_frame(&self.device, queue, data);
    }

    /// Set G-buffer views. Call after G-buffer creation or resize.
    pub fn set_gbuffer(&mut self, gbuffer: &rkf_render::GBuffer) {
        self.ssao.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view);
        self.shade.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);
        self.shade.set_ssao(&self.device, &self.ssao.output_view);
        self.volumetric.set_depth_view(&self.device, &gbuffer.position_view);
        self.volumetric.set_scene_hdr_view(&self.device, &self.shade.output_view);
        self.god_rays.set_input(&self.device, &self.volumetric.output_view);
    }

    pub fn set_hdr_output(&mut self, view: &wgpu::TextureView) {
        self.shade.set_output_view(&self.device, view);
    }

    /// Render: atmosphere LUTs → triangle G-buffer → SSAO → shade → volumetric → god rays.
    ///
    /// `mesh_draws` is an ordered list of `(gpu_object_index, allocation)` for
    /// every object whose geometry has been extracted into the mesh pool.
    /// The triangle pass always runs (clearing the G-buffer) so downstream
    /// passes see a defined surface.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        gbuffer: &rkf_render::GBuffer,
        mesh_draws: &[MeshDraw],
        atmo_frame_params: &crate::rkp_atmosphere::AtmosphereFrameParams,
    ) {
        // 0. Atmosphere LUTs (precomputed + per-frame).
        self.atmosphere.dispatch_if_dirty(encoder);
        {
            let q = self.profiler.begin_query("atmo", encoder);
            self.atmosphere.dispatch_per_frame(encoder, queue, atmo_frame_params);
            self.profiler.end_query(encoder, q);
        }

        // 1. Triangle G-buffer pass — owns the G-buffer (clears + draws).
        //    Always runs even with zero mesh draws so the cleared surface
        //    propagates to SSAO / shade as "sky" (hit_dist = 0).
        {
            let q = self.profiler.begin_query("triangle", encoder);
            self.triangle.dispatch(
                encoder,
                &self.scene.bind_group,
                gbuffer,
                &self.mesh_pool,
                mesh_draws,
                true, // always clear — triangle pass owns the G-buffer post-Phase-4
            );
            self.profiler.end_query(encoder, q);
        }

        // 2. SSAO at half resolution.
        {
            let q = self.profiler.begin_query("ssao", encoder);
            self.ssao.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 3. Deferred PBR shading.
        {
            let q = self.profiler.begin_query("shade", encoder);
            self.shade.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4. Volumetric march (half-res) + composite (full-res).
        {
            let q = self.profiler.begin_query("vol", encoder);
            self.volumetric.dispatch_march(encoder);
            self.volumetric.dispatch_composite(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4b. Screen-space god rays.
        {
            let q = self.profiler.begin_query("god_rays", encoder);
            self.god_rays.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // Profiler queries are resolved by `resolve_profiler_queries` — the
        // caller runs extra passes after this (bloom/tone/composite) and
        // wants them profiled too, so the resolve happens once at the end.
    }

    /// Resolve all profiler queries issued this frame. Call after *all* passes
    /// (including any issued by the caller after `render`) are encoded.
    pub fn resolve_profiler_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.resolve_queries(encoder);
    }

    /// End frame + process profiler results. Call after submit.
    pub fn end_profiler_frame(&mut self, frame_idx: u64, _width: u32, _height: u32) {
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

    /// Update shade params (sky colors, ambient intensity).
    pub fn update_shade_params(&self, queue: &wgpu::Queue, params: &ShadeParams) {
        queue.write_buffer(&self.shade_params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update the lights buffer (directional/point lights).
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
            self.shade.set_shade_data(
                &self.device,
                &self.shade_params_buffer,
                &self.lights_buffer,
                &self.materials_buffer,
            );
        } else {
            queue.write_buffer(&self.lights_buffer, 0, data);
        }
    }

    /// Replace the GPU materials palette.
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
            self.shade.set_shade_data(
                &self.device,
                &self.shade_params_buffer,
                &self.lights_buffer,
                &self.materials_buffer,
            );
        } else {
            queue.write_buffer(&self.materials_buffer, 0, data);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
        }
        self.ssao.resize(&self.device, width, height);
        self.shade.resize(&self.device, width, height);
        self.volumetric.resize(&self.device, width, height);
        self.god_rays.resize(&self.device, width, height);
    }

    /// Update volumetric parameters (fog, dust, march settings).
    pub fn update_volumetric_params(&self, queue: &wgpu::Queue, params: &VolumetricParams) {
        self.volumetric.update_params(queue, params);
    }

    /// Update cloud parameters.
    pub fn update_cloud_params(&self, queue: &wgpu::Queue, cloud: &CloudParams) {
        self.volumetric.update_cloud_params(queue, cloud);
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
