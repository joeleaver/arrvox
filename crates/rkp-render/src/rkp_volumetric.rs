//! RKIPatch volumetric rendering — fog, dust, and procedural clouds.
//!
//! Two compute passes:
//! 1. Volumetric march (half-res): marches view rays through atmosphere
//! 2. Volumetric composite (full-res): blends scatter over scene HDR
//!
//! Plus a tiny 1-thread compute pass that integrates cloud density along
//! the camera→sun ray and writes exp(-τ) to a buffer for CPU readback,
//! so the engine can dim direct sunlight when clouds overhead block the sun.

use std::sync::{Arc, atomic::{AtomicU32, AtomicBool, Ordering}};

use crate::validate_wgsl;

/// Uniform parameters for the volumetric march.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricParams {
    pub cam_pos: [f32; 4],
    pub cam_forward: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub width: u32,
    pub height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub max_steps: u32,
    pub step_size: f32,
    pub near: f32,
    pub far: f32,
    pub fog_color: [f32; 4],
    pub fog_height: [f32; 4],
    pub frame_index: u32,
    pub vol_ambient_r: f32,
    pub vol_ambient_g: f32,
    pub vol_ambient_b: f32,
    pub prev_view_proj: [[f32; 4]; 4],
}

/// Cloud parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParams {
    pub altitude: [f32; 4],
    pub noise: [f32; 4],
    pub wind: [f32; 4],
    pub flags: [f32; 4],
    /// x = slab_steps, y = shadow_steps, z = detail_octaves, w = ms_octaves.
    pub quality: [f32; 4],
    /// x = taa_alpha, y..w reserved.
    pub quality2: [f32; 4],
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            altitude: [1000.0, 3000.0, 0.1, 1.0],
            noise: [0.0003, 0.002, 0.3, 10000.0],
            wind: [1.0, 0.0, 5.0, 0.0],
            flags: [0.0, 0.0, 0.0, 0.0], // disabled by default
            quality: [32.0, 4.0, 4.0, 3.0],
            quality2: [0.25, 0.0, 0.0, 0.0],
        }
    }
}

/// Volumetric rendering pass (march + composite).
pub struct RkpVolumetricPass {
    march_pipeline: wgpu::ComputePipeline,
    march_bind_group_layout: wgpu::BindGroupLayout,
    march_bind_group: Option<wgpu::BindGroup>,

    composite_pipeline: wgpu::ComputePipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: Option<wgpu::BindGroup>,

    params_buffer: wgpu::Buffer,
    cloud_params_buffer: wgpu::Buffer,

    /// Half-res cloud scatter+transmittance output. Identity (0,0,0,1) on
    /// non-sky pixels so the cloud composite step is a no-op there.
    pub cloud_texture: wgpu::Texture,
    pub cloud_view: wgpu::TextureView,

    /// Half-res fog scatter+transmittance output (all pixels, no TAA).
    pub fog_texture: wgpu::Texture,
    pub fog_view: wgpu::TextureView,

    /// Previous-frame cloud buffer — sampled in the march for temporal accumulation.
    history_texture: wgpu::Texture,
    history_view: wgpu::TextureView,
    history_sampler: wgpu::Sampler,

    /// Full-res composited HDR output (replaces shade output for tone mapping).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    half_width: u32,
    half_height: u32,
    width: u32,
    height: u32,

    depth_view_set: bool,

    // Selective history update — copies current scatter into history only for
    // sky pixels, leaving object pixels untouched so their stale values don't
    // bleed into sky reprojection next frame.
    history_update_pipeline: wgpu::ComputePipeline,
    history_update_bind_group: Option<wgpu::BindGroup>,
    history_update_bind_group_layout: wgpu::BindGroupLayout,

    // Cloud → sun attenuation: tiny compute pass + async readback.
    sun_atten_pipeline: wgpu::ComputePipeline,
    sun_atten_bind_group: wgpu::BindGroup,
    sun_atten_storage: wgpu::Buffer,  // GPU-only, written by compute
    sun_atten_staging: wgpu::Buffer,  // MAP_READ copy of storage
    /// Latest received exp(-τ·k) value as f32-bits. Updated by map_async callback.
    sun_atten_value: Arc<AtomicU32>,
    /// Raw τ from last readback, for debugging.
    sun_atten_tau_bits: Arc<AtomicU32>,
    /// True while a map_async call is in flight. Blocks re-issuing the readback
    /// until the previous map completes (single-buffer design, lags 1–2 frames
    /// under normal GPU pacing).
    sun_atten_map_pending: Arc<AtomicBool>,
}

impl RkpVolumetricPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // March bind group layout (group 0).
        let march_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol march layout"),
                entries: &[
                    // 0: VolumetricParams uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1: depth buffer (G-buffer position texture, read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: scatter output (half-res, write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 3: CloudParams uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: previous-frame scatter history (filterable, sampled)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 5: history sampler (linear)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 6: fog scatter output (half-res, write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // Composite bind group layout (group 0).
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol composite layout"),
                entries: &[
                    // 0: scene HDR (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: cloud scatter (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: composited output (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 3: fog scatter (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Buffers.
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vol params"),
            size: std::mem::size_of::<VolumetricParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cloud_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vol cloud params"),
            size: std::mem::size_of::<CloudParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Textures. Cloud + fog are separate half-res buffers; history stores
        // previous-frame cloud for TAA.
        let (cloud_texture, cloud_view) = Self::create_march_output_texture(
            device, "vol cloud", half_width, half_height,
        );
        let (fog_texture, fog_view) = Self::create_march_output_texture(
            device, "vol fog", half_width, half_height,
        );
        let (history_texture, history_view) = Self::create_history_texture(
            device, "vol cloud history", half_width, half_height,
        );
        let history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vol history sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (output_texture, output_view) =
            Self::create_texture(device, "vol output", width, height, wgpu::TextureFormat::Rgba16Float);

        // March pipeline.
        let march_src = include_str!("shaders/rkp_volumetric.wgsl");
        validate_wgsl(march_src, "rkp_volumetric");
        let march_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_volumetric"),
            source: wgpu::ShaderSource::Wgsl(march_src.into()),
        });
        let march_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol march pipeline"),
            bind_group_layouts: &[Some(&march_bind_group_layout)],
            immediate_size: 0,
        });
        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol march"),
            layout: Some(&march_layout),
            module: &march_module,
            entry_point: Some("march_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Composite pipeline.
        let composite_src = include_str!("shaders/rkp_vol_composite.wgsl");
        validate_wgsl(composite_src, "rkp_vol_composite");
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_vol_composite"),
            source: wgpu::ShaderSource::Wgsl(composite_src.into()),
        });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol composite pipeline"),
            bind_group_layouts: &[Some(&composite_bind_group_layout)],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol composite"),
            layout: Some(&composite_layout),
            module: &composite_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // History-update pipeline — per-pixel copy of current scatter into history,
        // gated by depth so non-sky pixels don't contaminate the history buffer.
        let history_update_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol history update layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let history_update_src = include_str!("shaders/rkp_vol_history_update.wgsl");
        validate_wgsl(history_update_src, "rkp_vol_history_update");
        let history_update_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_vol_history_update"),
            source: wgpu::ShaderSource::Wgsl(history_update_src.into()),
        });
        let history_update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol history update pipeline layout"),
            bind_group_layouts: &[Some(&history_update_bind_group_layout)],
            immediate_size: 0,
        });
        let history_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol history update"),
            layout: Some(&history_update_pipeline_layout),
            module: &history_update_module,
            entry_point: Some("update_history"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Sun-atten pipeline — 1-thread compute + storage + readback.
        let sun_atten_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol sun atten layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sun_atten_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vol sun atten storage"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sun_atten_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vol sun atten staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sun_atten_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol sun atten bg"),
            layout: &sun_atten_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cloud_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sun_atten_storage.as_entire_binding() },
            ],
        });
        let sun_atten_src = include_str!("shaders/rkp_cloud_sun_atten.wgsl");
        validate_wgsl(sun_atten_src, "rkp_cloud_sun_atten");
        let sun_atten_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_cloud_sun_atten"),
            source: wgpu::ShaderSource::Wgsl(sun_atten_src.into()),
        });
        let sun_atten_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol sun atten pipeline layout"),
            bind_group_layouts: &[Some(&sun_atten_layout)],
            immediate_size: 0,
        });
        let sun_atten_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol sun atten"),
            layout: Some(&sun_atten_pipeline_layout),
            module: &sun_atten_module,
            entry_point: Some("sun_atten_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            march_pipeline,
            march_bind_group_layout,
            march_bind_group: None,
            composite_pipeline,
            composite_bind_group_layout,
            composite_bind_group: None,
            params_buffer,
            cloud_params_buffer,
            cloud_texture,
            cloud_view,
            fog_texture,
            fog_view,
            history_texture,
            history_view,
            history_sampler,
            output_texture,
            output_view,
            half_width,
            half_height,
            width,
            height,
            depth_view_set: false,
            history_update_pipeline,
            history_update_bind_group: None,
            history_update_bind_group_layout,
            sun_atten_pipeline,
            sun_atten_bind_group,
            sun_atten_storage,
            sun_atten_staging,
            sun_atten_value: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            sun_atten_tau_bits: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            sun_atten_map_pending: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the shadow map texture for volumetric shadow sampling.
    /// Set the depth view (G-buffer position texture). Rebuilds march bind group.
    pub fn set_depth_view(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.march_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol march bg"),
            layout: &self.march_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cloud_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.history_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.history_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.fog_view) },
            ],
        }));
        self.history_update_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol history update bg"),
            layout: &self.history_update_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.history_view) },
            ],
        }));
        self.depth_view_set = true;
    }

    /// Copy the current scatter output into the history buffer for next-frame
    /// reprojection — but only for sky pixels. Object pixels leave history
    /// untouched so their transient values don't bleed into sky reprojection.
    pub fn update_history(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(bg) = &self.history_update_bind_group else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol history update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.history_update_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.half_width + 7) / 8,
            (self.half_height + 7) / 8,
            1,
        );
    }

    /// Set the scene HDR view (shade pass output). Rebuilds composite bind group.
    pub fn set_scene_hdr_view(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        self.composite_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol composite bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.cloud_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.fog_view) },
            ],
        }));
    }

    /// Update volumetric parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &VolumetricParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update cloud parameters.
    pub fn update_cloud_params(&self, queue: &wgpu::Queue, cloud: &CloudParams) {
        queue.write_buffer(&self.cloud_params_buffer, 0, bytemuck::bytes_of(cloud));
    }

    /// Dispatch the volumetric march (half-res).
    pub fn dispatch_march(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.march_bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.march_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.half_width + 7) / 8,
            (self.half_height + 7) / 8,
            1,
        );
    }

    /// Dispatch the 1-thread sun-attenuation compute pass and queue the GPU→CPU
    /// copy into the staging buffer. The value becomes readable after the next
    /// submit completes and the map_async callback fires (see `issue_sun_atten_map`).
    pub fn dispatch_sun_atten(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vol sun atten"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sun_atten_pipeline);
            pass.set_bind_group(0, &self.sun_atten_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.sun_atten_storage, 0, &self.sun_atten_staging, 0, 16);
    }

    /// After submit, issue a non-blocking map on the staging buffer. The callback
    /// writes the f32 bits into `sun_atten_value` when the GPU catches up. Skipped
    /// if a prior map is still pending (single-buffer design — one read at a time).
    pub fn issue_sun_atten_map(&self) {
        if self.sun_atten_map_pending.load(Ordering::Acquire) {
            return;
        }
        self.sun_atten_map_pending.store(true, Ordering::Release);

        let value = self.sun_atten_value.clone();
        let tau_bits = self.sun_atten_tau_bits.clone();
        let pending = self.sun_atten_map_pending.clone();
        let staging_for_cb = self.sun_atten_staging.clone();
        self.sun_atten_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                let data = staging_for_cb.slice(..).get_mapped_range();
                let atten = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let tau = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                value.store(atten, Ordering::Release);
                tau_bits.store(tau, Ordering::Release);
                drop(data);
                staging_for_cb.unmap();
            }
            pending.store(false, Ordering::Release);
        });
    }

    /// Latest received exp(-τ) value (updated asynchronously by `issue_sun_atten_map`).
    pub fn sun_atten_value(&self) -> f32 {
        f32::from_bits(self.sun_atten_value.load(Ordering::Acquire))
    }

    /// Raw τ from the last readback, for debugging. Staging buffer holds
    /// (exp(-τ·k), τ, 0, 0).
    pub fn sun_atten_tau_debug(&self) -> f32 {
        f32::from_bits(self.sun_atten_tau_bits.load(Ordering::Acquire))
    }

    /// Dispatch the volumetric composite (full-res).
    pub fn dispatch_composite(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.composite_bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol composite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (self.width + 7) / 8,
            (self.height + 7) / 8,
            1,
        );
    }

    /// Resize textures. Call when window resizes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let hw = (width / 2).max(1);
        let hh = (height / 2).max(1);
        if hw == self.half_width && hh == self.half_height { return; }
        self.half_width = hw;
        self.half_height = hh;
        self.width = width;
        self.height = height;
        let (ct, cv) = Self::create_march_output_texture(device, "vol cloud", hw, hh);
        self.cloud_texture = ct;
        self.cloud_view = cv;
        let (ft, fv) = Self::create_march_output_texture(device, "vol fog", hw, hh);
        self.fog_texture = ft;
        self.fog_view = fv;
        let (ht, hv) = Self::create_history_texture(device, "vol cloud history", hw, hh);
        self.history_texture = ht;
        self.history_view = hv;
        let (ot, ov) = Self::create_texture(device, "vol output", width, height, wgpu::TextureFormat::Rgba16Float);
        self.output_texture = ot;
        self.output_view = ov;
        // Bind groups need rebuild — caller must call set_depth_view + set_scene_hdr_view.
    }

    fn create_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_march_output_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_history_texture(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }
}
