//! RKIPatch renderer — owns the GPU scene and all render passes.
//!
//! The caller (rkp-editor) drives the renderer each frame:
//! 1. Build RkpGpuObjects from ECS/scene state
//! 2. Call `upload_frame(objects, camera)` — cheap, every frame
//! 3. Call `upload_geometry(...)` — only when geometry changes
//! 4. Call `render(encoder, gbuffer)` — raster + shadow/AO + shade
//!
//! No MarchPass trait. No callbacks. No incremental caching.

use crate::rkp_scene::{RkpScene, GeometryUpload, FrameUpload, CameraUniforms};
use crate::rkp_shadow_ao::{RkpShadowAoPass, ShadowAoParams};
use crate::rkp_shade::{RkpShadePass, ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_gpu_object::RkpGpuObject;
use crate::splat_raster::SplatRasterPipeline;
use crate::splat_emit::SplatEmitPass;

/// The complete RKIPatch renderer.
pub struct RkpRenderer {
    /// Scene GPU buffers.
    pub scene: RkpScene,
    /// Face emit pass (CPU-driven, writes face instance buffer).
    pub emit: SplatEmitPass,
    /// Rasterization pipeline (vertex/fragment, writes G-buffer).
    pub raster: SplatRasterPipeline,
    /// Half-res shadow + AO compute pass.
    pub shadow_ao: RkpShadowAoPass,
    /// Deferred PBR shading compute pass.
    pub shade: RkpShadePass,
    /// Default light/material buffers (created at init, updated later).
    shade_params_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    /// Device for buffer operations.
    device: wgpu::Device,
}

impl RkpRenderer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let scene = RkpScene::new(device);
        let emit = SplatEmitPass::new(device, &scene.bind_group_layout);
        let raster = SplatRasterPipeline::new(device, &scene.bind_group_layout, &emit);
        let shadow_ao = RkpShadowAoPass::new(device, &scene, width, height);
        let mut shade = RkpShadePass::new(device, width, height);

        // Create default light + material buffers.
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

        Self {
            scene, emit, raster, shadow_ao, shade,
            shade_params_buffer, lights_buffer, materials_buffer,
            device: device.clone(),
        }
    }

    /// Upload geometry data. Call only when geometry changes.
    pub fn upload_geometry(&mut self, queue: &wgpu::Queue, data: &GeometryUpload) {
        self.scene.upload_geometry(&self.device, queue, data);
    }

    /// Upload per-frame data. Call every frame.
    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        self.scene.upload_frame(&self.device, queue, data);
    }

    /// Set G-buffer views. Call after G-buffer creation or resize.
    pub fn set_gbuffer(
        &mut self,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
    ) {
        self.shadow_ao.set_gbuffer(&self.device, position_view, normal_view);
        self.shade.set_gbuffer(&self.device, position_view, normal_view, material_view);
        self.shade.set_shadow_ao(&self.device, &self.shadow_ao.output_view);
    }

    /// Point the shade pass at an external HDR output texture (e.g., engine's shading HDR).
    pub fn set_hdr_output(&mut self, view: &wgpu::TextureView) {
        self.shade.set_output_view(&self.device, view);
    }

    /// Upload face instances and indirect draw args.
    pub fn upload_faces(&self, encoder: &mut wgpu::CommandEncoder, faces: &[crate::splat_emit::FaceInstance]) {
        if faces.is_empty() {
            return;
        }
        let face_bytes: &[u8] = bytemuck::cast_slice(faces);

        // Grow face buffer if needed.
        {
            let buf = self.emit.face_buffer.borrow();
            if face_bytes.len() as u64 > buf.size() {
                let new_size = (face_bytes.len() as u64).max(buf.size() * 2);
                drop(buf);
                let new_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("emit face instances"),
                    size: new_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                *self.raster.face_bind_group.borrow_mut() = self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("raster face bind group"),
                        layout: &self.raster.face_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: new_buf.as_entire_binding(),
                        }],
                    },
                );
                *self.emit.face_buffer.borrow_mut() = new_buf;
            }
        }

        // Upload via staging.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("face staging"),
            size: face_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        staging.slice(..).get_mapped_range_mut().copy_from_slice(face_bytes);
        staging.unmap();
        encoder.copy_buffer_to_buffer(&staging, 0, &self.emit.face_buffer.borrow(), 0, face_bytes.len() as u64);

        // Set indirect draw args.
        let draw_args: [u32; 4] = [6, faces.len() as u32, 0, 0];
        let args_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect args staging"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        args_staging.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&draw_args));
        args_staging.unmap();
        encoder.copy_buffer_to_buffer(&args_staging, 0, &self.emit.indirect_buffer, 0, 16);
    }

    /// Render: raster G-buffer + shadow/AO + shade.
    /// Faces must have been uploaded via upload_faces() before this call.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &rkf_render::GBuffer,
        queue: &wgpu::Queue,
        shadow_ao_params: &ShadowAoParams,
    ) {
        // 1. Raster: face quads → G-buffer.
        {
            let mut render_pass = SplatRasterPipeline::begin_render_pass(encoder, gbuffer);
            self.raster.draw(&mut render_pass, &self.scene.bind_group, &self.emit.indirect_buffer);
        }

        // 2. Shadow + AO at half resolution.
        self.shadow_ao.update_params(queue, shadow_ao_params);
        self.shadow_ao.dispatch(encoder, &self.scene);

        // 3. Deferred PBR shading.
        self.shade.dispatch(encoder);
    }

    /// Resize resolution-dependent resources.
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
