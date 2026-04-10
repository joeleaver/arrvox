//! RKIPatch renderer — owns the GPU scene and all render passes.
//!
//! The caller (rkp-editor) drives the renderer each frame:
//! 1. Build RkpGpuObjects from ECS/scene state
//! 2. Call `upload_frame(objects, camera)` — cheap, every frame
//! 3. Call `upload_geometry(...)` — only when geometry changes
//! 4. Call `render(encoder, width, height, ...)` — march + shadow/AO + shade

use crate::rkp_scene::{RkpScene, GeometryUpload, FrameUpload, CameraUniforms};
use crate::rkp_shadow_ao::{RkpShadowAoPass, ShadowAoParams};
use crate::rkp_shade::{RkpShadePass, ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_gpu_object::RkpGpuObject;
use crate::octree_march::OctreeMarchPass;
use crate::gpu_profiler::GpuProfiler;

/// The complete RKIPatch renderer.
pub struct RkpRenderer {
    /// Scene GPU buffers.
    pub scene: RkpScene,
    /// Octree ray march compute pass — primary visibility.
    pub march: OctreeMarchPass,
    /// Half-res shadow + AO compute pass.
    pub shadow_ao: RkpShadowAoPass,
    /// Deferred PBR shading compute pass.
    pub shade: RkpShadePass,
    /// Default light/material buffers.
    shade_params_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    /// Device for buffer operations.
    device: wgpu::Device,
    /// GPU timestamp profiler.
    pub profiler: GpuProfiler,
    /// Pass slot indices.
    slot_march: u32,
    slot_normal: u32,
    slot_shadow: u32,
    slot_shade: u32,
}

impl RkpRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let scene = RkpScene::new(device);
        let mut march = OctreeMarchPass::new(device, &scene.bind_group_layout);

        let mut profiler = GpuProfiler::new(device, 0.0); // disabled for now
        let slot_march = profiler.register_pass("march_hit");
        let slot_normal = profiler.register_pass("march_nrm");
        let slot_shadow = profiler.register_pass("shadow");
        let slot_shade = profiler.register_pass("shade");
        let shadow_ao = RkpShadowAoPass::new(device, &scene, width, height);
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

        shade.set_shade_data(device, &shade_params_buffer, &lights_buffer, &materials_buffer);
        shade.set_camera(device, &scene.camera_buffer);
        march.set_materials(device, &materials_buffer);

        Self {
            scene, march, shadow_ao, shade,
            shade_params_buffer, lights_buffer, materials_buffer,
            device: device.clone(),
            profiler, slot_march, slot_normal, slot_shadow, slot_shade,
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
        self.shadow_ao.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view);
        self.shade.set_gbuffer(&self.device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);
        self.shade.set_shadow_ao(&self.device, &self.shadow_ao.output_view);
    }

    pub fn set_hdr_output(&mut self, view: &wgpu::TextureView) {
        self.shade.set_output_view(&self.device, view);
    }

    /// Render: march → shadow/AO → shade.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        object_count: u32,
        width: u32,
        height: u32,
        shadow_ao_params: &ShadowAoParams,
    ) {
        // 1. Octree ray march → G-buffer.
        self.march.dispatch(
            encoder, queue, &self.scene.bind_group,
            object_count, width, height, 0,
            self.profiler.compute_timestamps(self.slot_march),
        );

        // 2. Shadow + AO at half resolution.
        self.shadow_ao.update_params(queue, shadow_ao_params);
        self.shadow_ao.dispatch_with_timestamps(encoder, &self.scene,
            self.profiler.compute_timestamps(self.slot_shadow));

        // 3. Deferred PBR shading.
        self.shade.dispatch_with_timestamps(encoder,
            self.profiler.compute_timestamps(self.slot_shade));

        // 4. Resolve profiler timestamps.
        self.profiler.resolve(encoder);
    }

    /// Read and log GPU profiler results. Call after submit + poll.
    pub fn log_profiler(&mut self) {
        self.profiler.read_and_log(&self.device, 60);
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
        self.shadow_ao.resize(&self.device, width, height);
        self.shade.resize(&self.device, width, height);
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
