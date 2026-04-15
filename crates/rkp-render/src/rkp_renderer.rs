//! RKIPatch renderer — owns the GPU scene and all render passes.
//!
//! The caller (rkp-editor) drives the renderer each frame:
//! 1. Build RkpGpuObjects from ECS/scene state
//! 2. Call `upload_frame(objects, camera)` — cheap, every frame
//! 3. Call `upload_geometry(...)` — only when geometry changes
//! 4. Call `render(encoder, width, height, ...)` — march (+ shadow) + SSAO + shade

use crate::rkp_scene::{RkpScene, GeometryUpload, FrameUpload, CameraUniforms};
use crate::rkp_ssao::RkpSsaoPass;
use crate::rkp_shade::{RkpShadePass, ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_volumetric::{RkpVolumetricPass, VolumetricParams, CloudParams};
use crate::rkp_atmosphere::RkpAtmospherePass;
use crate::rkp_god_rays::RkpGodRayPass;
use crate::rkp_gpu_object::RkpGpuObject;
use crate::octree_march::OctreeMarchPass;
use crate::rkp_shadow_trace::ShadowTracePass;
use wgpu_profiler::GpuProfiler;

/// The complete RKIPatch renderer.
pub struct RkpRenderer {
    /// Scene GPU buffers.
    pub scene: RkpScene,
    /// Octree ray march compute pass — primary visibility only; shadows
    /// run in `shadow_trace`.
    pub march: OctreeMarchPass,
    /// Half-resolution shadow trace (up to 4 shadow-casting lights).
    pub shadow_trace: ShadowTracePass,
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
        let mut march = OctreeMarchPass::new(device, &scene.bind_group_layout);

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
        // Camera is per-viewport now; ViewportRenderer::new re-points
        // shade's camera binding at its own `camera_buffer`. We leave it
        // uninitialized here — a renderer without a VR doesn't render.
        shade.set_atmosphere_luts(device, &atmosphere.transmittance_view, &atmosphere.multiscatter_view, &atmosphere.lut_sampler, &atmosphere.sky_view_view, &atmosphere.ap_view);
        march.set_materials(device, &materials_buffer);
        march.set_lights(device, &lights_buffer);

        let shadow_trace = ShadowTracePass::new(
            device, width, height,
            &scene.bind_group_layout,
            march.params_bind_group_layout(),
        );
        let volumetric = RkpVolumetricPass::new(device, width, height);
        let mut god_rays = RkpGodRayPass::new(device, width, height);
        god_rays.set_input(device, &volumetric.output_view);

        Self {
            scene, march, shadow_trace, ssao, shade, atmosphere, volumetric, god_rays,
            shade_params_buffer, lights_buffer, materials_buffer,
            device: device.clone(),
            profiler, timestamp_period,
            width, height,
        }
    }

    pub fn upload_geometry(&mut self, queue: &wgpu::Queue, data: &GeometryUpload) {
        self.scene.upload_geometry(&self.device, queue, data);
    }

    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        self.scene.upload_frame(&self.device, queue, data);
    }

    /// Set G-buffer views. Call after G-buffer creation or resize.
    pub fn set_gbuffer(
        &mut self,
        gbuffer: &rkf_render::GBuffer,
    ) {
        self.march.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);
        self.shadow_trace.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view);
        self.ssao.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view);
        self.shade.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);
        self.shade.set_shadow_and_ssao(&self.device, &self.shadow_trace.output_view, &self.ssao.output_view);
        self.volumetric.set_depth_view(&self.device, &gbuffer.position_view);
        self.volumetric.set_scene_hdr_view(&self.device, &self.shade.output_view);
        self.god_rays.set_input(&self.device, &self.volumetric.output_view);
    }

    pub fn set_hdr_output(&mut self, view: &wgpu::TextureView) {
        self.shade.set_output_view(&self.device, view);
    }

    /// Render: march (+ per-light shadow) → SSAO → shade.
    /// `scene_bind_group` comes from the calling viewport (each VR owns
    /// its own, tying the shared scene buffers to its own camera).
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        scene_bind_group: &wgpu::BindGroup,
        object_count: u32,
        width: u32,
        height: u32,
        shadow_steps: u32,
        num_lights: u32,
        screen_aabbs: &[u8],
        atmo_frame_params: &crate::rkp_atmosphere::AtmosphereFrameParams,
    ) {
        // Upload screen-space AABBs for tile culling.
        self.march.upload_screen_aabbs(queue, screen_aabbs);

        // 0. Atmosphere LUTs (precomputed + per-frame).
        self.atmosphere.dispatch_if_dirty(encoder);
        {
            let q = self.profiler.begin_query("atmo", encoder);
            self.atmosphere.dispatch_per_frame(encoder, queue, atmo_frame_params);
            self.profiler.end_query(encoder, q);
        }

        // 1. Octree ray march → G-buffer (surface only; shadow is its own pass).
        self.march.clear_stats(encoder);
        {
            let q = self.profiler.begin_query("march", encoder);
            self.march.dispatch(
                encoder, queue, scene_bind_group,
                object_count, width, height, 0,
                shadow_steps, num_lights, None,
            );
            self.profiler.end_query(encoder, q);
        }

        // 1b. Half-res shadow trace: reads gbuf, writes half-res shadow.
        // Runs at 1/4 the thread count of primary march — bilateral
        // upsample happens inline in the shade pass.
        if let Some(params_bg) = self.march.params_bind_group() {
            let q = self.profiler.begin_query("shadow", encoder);
            self.shadow_trace.dispatch(encoder, scene_bind_group, params_bg);
            self.profiler.end_query(encoder, q);
        }
        self.march.copy_stats(encoder);

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

        // Note: profiler queries are resolved by `resolve_profiler_queries` —
        // the caller runs extra passes after this (bloom/tone/composite) and
        // wants them profiled too, so the resolve happens once at the end.
    }

    /// Render one frame for the given viewport. Encodes the entire renderer
    /// pipeline (march → shadow → ssao → shade → volumetric → god_rays →
    /// bloom → bloom_composite → tone_map → composite copy) into `encoder`.
    /// The caller still owns the wireframe overlay + readback copy +
    /// submit so engine-specific concerns stay engine-side.
    ///
    /// In Phase 2 there's only one viewport so the renderer's internal
    /// resolution is always in step with `viewport.width/height`. Multi-
    /// viewport phases will need to resize the renderer in here.
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
        // Point shade's camera binding at THIS viewport's camera buffer.
        // Single-viewport case this is redundant (already wired at VR
        // construction), but with multiple visible viewports each pass
        // of render_to must re-aim shade at the current target.
        self.shade.set_camera(&self.device, &viewport.camera_buffer);

        // Renderer-internal pipeline (march, shadow, ssao, shade, vol, god_rays).
        self.render(
            encoder, queue, &viewport.scene_bind_group,
            object_count, viewport.width, viewport.height,
            shadow_steps, num_lights, screen_aabbs, atmo_frame_params,
        );

        // Per-viewport bloom + tonemap chain.
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

        // Copy LDR into the composite texture so the caller can overlay
        // wireframes on it before the readback copy.
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

    /// Resolve all profiler queries issued this frame. Call after *all* passes
    /// (including any issued by the caller after `render`) are encoded.
    pub fn resolve_profiler_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.resolve_queries(encoder);
    }

    /// End frame + process profiler results. Call after submit.
    pub fn end_profiler_frame(&mut self, frame_idx: u64, width: u32, height: u32) {
        self.march.read_stats(&self.device, width * height, frame_idx);
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
            self.march.set_lights(&self.device, &self.lights_buffer);
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
            self.march.set_materials(&self.device, &self.materials_buffer);
        } else {
            queue.write_buffer(&self.materials_buffer, 0, data);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
        }
        self.shadow_trace.resize(&self.device, width, height);
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
