//! RKIPatch shadow + AO compute pass.
//!
//! Half-resolution compute shader that traces shadow rays and AO probes through
//! the per-voxel octree. Writes to a half-res Rgba8Unorm storage texture
//! (R=shadow, G=AO). Consumed by the shading pass.

use crate::rkp_scene::RkpScene;

/// Uniform parameters for the shadow/AO pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowAoParams {
    pub light_dir: [f32; 3],
    pub num_objects: u32,
    pub light_intensity: f32,
    pub ao_radius: f32,
    pub ao_steps: u32,
    pub shadow_steps: u32,
}

impl Default for ShadowAoParams {
    fn default() -> Self {
        Self {
            light_dir: [0.5, 0.8, 0.3],
            num_objects: 0,
            light_intensity: 1.0,
            ao_radius: 0.1,
            ao_steps: 5,
            shadow_steps: 32,
        }
    }
}

/// Half-resolution shadow + AO compute pass.
pub struct RkpShadowAoPass {
    pipeline: wgpu::ComputePipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    /// Output texture (half-res, Rgba8Unorm).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    output_bind_group: wgpu::BindGroup,
    /// G-buffer bind group (recreated when resolution changes).
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    half_width: u32,
    half_height: u32,
}

impl RkpShadowAoPass {
    pub fn new(
        device: &wgpu::Device,
        scene: &RkpScene,
        width: u32,
        height: u32,
    ) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // G-buffer bind group layout (group 1): position + normal textures.
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shadow_ao gbuf layout"),
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
                ],
            });

        // Output bind group layout (group 2): storage texture write.
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shadow_ao output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Params bind group layout (group 3): uniform.
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shadow_ao params layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_shadow_ao params"),
            size: std::mem::size_of::<ShadowAoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shadow_ao params bg"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Output texture.
        let (output_texture, output_view) = Self::create_output(device, half_width, half_height);
        let output_bind_group = Self::create_output_bind_group(device, &output_bind_group_layout, &output_view);

        // Pipeline.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_shadow_ao"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rkp_shadow_ao.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_shadow_ao pipeline"),
            bind_group_layouts: &[
                &scene.bind_group_layout,    // group 0: scene
                &gbuffer_bind_group_layout,  // group 1: g-buffer
                &output_bind_group_layout,   // group 2: output
                &params_bind_group_layout,   // group 3: params
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_shadow_ao"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            gbuffer_bind_group_layout,
            output_bind_group_layout,
            params_bind_group_layout,
            params_buffer,
            params_bind_group,
            output_texture,
            output_view,
            output_bind_group,
            gbuffer_bind_group: None,
            half_width,
            half_height,
        }
    }

    /// Update the G-buffer bind group (call when G-buffer textures are created/resized).
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shadow_ao gbuf bg"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
            ],
        }));
    }

    /// Update parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &ShadowAoParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the shadow/AO compute pass.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RkpScene,
    ) {
        let gbuf_bg = match &self.gbuffer_bind_group {
            Some(bg) => bg,
            None => return, // G-buffer not set yet.
        };

        let wg_x = (self.half_width + 7) / 8;
        let wg_y = (self.half_height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_shadow_ao"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, gbuf_bg, &[]);
        pass.set_bind_group(2, &self.output_bind_group, &[]);
        pass.set_bind_group(3, &self.params_bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Resize the output texture (call when window resizes).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let hw = (width / 2).max(1);
        let hh = (height / 2).max(1);
        if hw == self.half_width && hh == self.half_height {
            return;
        }
        self.half_width = hw;
        self.half_height = hh;
        let (tex, view) = Self::create_output(device, hw, hh);
        self.output_texture = tex;
        self.output_view = view;
        self.output_bind_group =
            Self::create_output_bind_group(device, &self.output_bind_group_layout, &self.output_view);
    }

    fn create_output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp_shadow_ao output"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_output_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shadow_ao output bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }
}
