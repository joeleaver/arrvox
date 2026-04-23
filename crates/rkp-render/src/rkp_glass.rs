//! Glass composite post-pass — runs after `rkp_shade` and before the
//! rest of the post-process chain.
//!
//! Reads the shaded HDR + `gbuf_glass` and, for any pixel whose primary
//! ray passed through glass, does a screen-space refraction sample to
//! compose the bent-behind color with Beer absorption and a Fresnel-
//! weighted sky reflection. Non-glass pixels pass through unchanged.
//!
//! Runs in one full-screen compute dispatch. Cost: ~1 texture load +
//! branch for the ~90 % of pixels without glass; ~4 samples + Snell /
//! Beer math for glass pixels. Sub-ms at 1920×1080 on desktop GPUs.

use crate::validate_wgsl;

/// Glass composite compute pass.
pub struct RkpGlassPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RkpGlassPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let sampled_float = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampled_uint = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rkp_glass layout"),
            entries: &[
                // 0: HDR input (shaded behind).
                sampled_float(0),
                // 1: gbuf_glass (oct-packed normal + packed thickness / material_id).
                sampled_uint(1),
                // 2: HDR output.
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
                // 3: camera uniform (for ray reconstruction + projection).
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
                // 4: materials palette.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let (output_texture, output_view) = Self::create_output(device, width, height);

        let shader_src = include_str!("shaders/rkp_glass.wgsl");
        validate_wgsl(shader_src, "rkp_glass");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_glass"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_glass"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rkp_glass pipeline layout"),
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
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Rebind every input + output. Call on init, resize, and any time
    /// a source view is re-created (G-buffer rebuild, HDR recreate,
    /// materials buffer reallocation).
    pub fn set_inputs(
        &mut self,
        device: &wgpu::Device,
        hdr_in_view: &wgpu::TextureView,
        gbuf_glass_view: &wgpu::TextureView,
        camera_buffer: &wgpu::Buffer,
        materials_buffer: &wgpu::Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_glass bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(hdr_in_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuf_glass_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: materials_buffer.as_entire_binding() },
            ],
        }));
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let bg = match &self.bind_group { Some(bg) => bg, None => return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_glass"),
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
            label: Some("rkp_glass output"),
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
