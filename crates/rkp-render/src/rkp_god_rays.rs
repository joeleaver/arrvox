//! Screen-space god rays — radial blur of a sun-visibility mask.
//!
//! Post-process compute pass. Reads (composite HDR, G-buffer depth, cloud
//! transmittance) and writes the composite with additive light shafts.

use crate::validate_wgsl;

/// GPU parameters for the god ray pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GodRayParams {
    pub sun_screen_pos: [f32; 2],
    pub sun_on_screen: f32,
    pub density: f32,
    pub weight: f32,
    pub decay: f32,
    pub exposure: f32,
    pub num_samples: u32,
    /// Linear-RGB sun tint (atmospherically extinguished). Intensity is
    /// applied separately via `exposure`.
    pub sun_color: [f32; 3],
    pub _pad: f32,
}

impl Default for GodRayParams {
    fn default() -> Self {
        Self {
            sun_screen_pos: [0.5, 0.5],
            sun_on_screen: 0.0,
            density: 1.0,
            // Mask is scene HDR luminance (up to ~65500 at the sun disc).
            // Weight stays small because per-sample contribution is large.
            weight: 0.01,
            decay: 0.95,
            exposure: 0.1,
            num_samples: 64,
            sun_color: [1.0, 0.95, 0.9],
            _pad: 0.0,
        }
    }
}

/// Screen-space god ray compute pass.
pub struct RkpGodRayPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RkpGodRayPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let sampled_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("god_rays layout"),
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
                sampled_tex(1), // composite
                sampled_tex(2), // gbuf_position
                sampled_tex(3), // cloud_tex (half-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("god_ray params"),
            size: std::mem::size_of::<GodRayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (output_texture, output_view) = Self::create_output(device, width, height);

        let shader_src = include_str!("shaders/rkp_god_rays.wgsl");
        validate_wgsl(shader_src, "rkp_god_rays");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_god_rays"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("god_rays"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("god_rays layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            })),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group: None,
            params_buffer,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Bind the four input views the god-rays pass reads from. Call whenever
    /// any source view is re-created (resize, G-buffer rebuild, etc.).
    pub fn set_inputs(
        &mut self,
        device: &wgpu::Device,
        composite_view: &wgpu::TextureView,
        gbuf_position_view: &wgpu::TextureView,
        cloud_view: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("god_rays bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(composite_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(gbuf_position_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(cloud_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.output_view) },
            ],
        }));
    }

    /// Update god ray parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &GodRayParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the god ray pass.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("god_rays"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height { return; }
        self.width = width;
        self.height = height;
        let (t, v) = Self::create_output(device, width, height);
        self.output_texture = t;
        self.output_view = v;
    }

    fn create_output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("god_rays output"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }
}
