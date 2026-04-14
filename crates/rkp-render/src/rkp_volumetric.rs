//! RKIPatch volumetric rendering — fog, dust, and procedural clouds.
//!
//! Two compute passes:
//! 1. Volumetric march (half-res): marches view rays through atmosphere
//! 2. Volumetric composite (full-res): blends scatter over scene HDR

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
    pub fog_distance: [f32; 4],
    pub frame_index: u32,
    pub vol_ambient_r: f32,
    pub vol_ambient_g: f32,
    pub vol_ambient_b: f32,
}

/// Cloud parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParams {
    pub altitude: [f32; 4],
    pub noise: [f32; 4],
    pub wind: [f32; 4],
    pub flags: [f32; 4],
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            altitude: [1000.0, 3000.0, 0.1, 1.0],
            noise: [0.0003, 0.002, 0.3, 10000.0],
            wind: [1.0, 0.0, 5.0, 0.0],
            flags: [0.0, 0.0, 0.0, 0.0], // disabled by default
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

    /// Half-res scatter+transmittance output.
    pub scatter_texture: wgpu::Texture,
    pub scatter_view: wgpu::TextureView,

    /// Full-res composited HDR output (replaces shade output for tone mapping).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    half_width: u32,
    half_height: u32,
    width: u32,
    height: u32,

    depth_view_set: bool,
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
                    // 1: vol scatter (read)
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

        // Textures.
        let (scatter_texture, scatter_view) =
            Self::create_texture(device, "vol scatter", half_width, half_height, wgpu::TextureFormat::Rgba16Float);
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

        Self {
            march_pipeline,
            march_bind_group_layout,
            march_bind_group: None,
            composite_pipeline,
            composite_bind_group_layout,
            composite_bind_group: None,
            params_buffer,
            cloud_params_buffer,
            scatter_texture,
            scatter_view,
            output_texture,
            output_view,
            half_width,
            half_height,
            width,
            height,
            depth_view_set: false,
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
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scatter_view) },
                wgpu::BindGroupEntry { binding: 3, resource: self.cloud_params_buffer.as_entire_binding() },
            ],
        }));
        self.depth_view_set = true;
    }

    /// Set the scene HDR view (shade pass output). Rebuilds composite bind group.
    pub fn set_scene_hdr_view(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        self.composite_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol composite bg"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.scatter_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.output_view) },
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
        let (st, sv) = Self::create_texture(device, "vol scatter", hw, hh, wgpu::TextureFormat::Rgba16Float);
        self.scatter_texture = st;
        self.scatter_view = sv;
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
}
