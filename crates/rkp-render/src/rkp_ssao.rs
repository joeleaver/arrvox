//! RKIPatch screen-space ambient occlusion (SSAO) compute pass.
//!
//! Half-resolution compute shader that samples a hemisphere around each pixel's
//! normal in screen space to estimate ambient occlusion. Reads G-buffer position
//! + normal, writes AO factor to a half-res R8Unorm storage texture.

/// Uniform parameters for the SSAO pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoParams {
    pub radius: f32,
    pub bias: f32,
    pub intensity: f32,
    pub _pad: u32,
    pub kernel: [[f32; 4]; 16], // hemisphere samples
}

impl Default for SsaoParams {
    fn default() -> Self {
        Self {
            radius: 0.5,
            bias: 0.025,
            intensity: 1.5,
            _pad: 0,
            kernel: Self::generate_kernel(),
        }
    }
}

impl SsaoParams {
    /// Generate a cosine-weighted hemisphere sample kernel.
    /// Uses a deterministic hash sequence (no rand dependency).
    fn generate_kernel() -> [[f32; 4]; 16] {
        let mut kernel = [[0.0f32; 4]; 16];
        for i in 0..16 {
            // Deterministic quasi-random using simple hash.
            let fi = i as f32;
            let phi = 2.4 * fi; // golden angle
            let cos_theta = 1.0 - (fi + 0.5) / 16.0; // uniform cos(theta)
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

            let x = sin_theta * phi.cos();
            let y = sin_theta * phi.sin();
            let z = cos_theta; // hemisphere: z >= 0

            // Scale: samples closer to origin are more important (accelerating distribution).
            let scale = (fi / 16.0).powi(2) * 0.9 + 0.1; // lerp(0.1, 1.0, (i/16)^2)

            kernel[i] = [x * scale, y * scale, z * scale, 0.0];
        }
        kernel
    }
}

/// Half-resolution SSAO compute pass.
pub struct RkpSsaoPass {
    pipeline: wgpu::ComputePipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    output_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    /// Output texture (half-res, R8Unorm).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    output_bind_group: wgpu::BindGroup,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    half_width: u32,
    half_height: u32,
}

impl RkpSsaoPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // Group 0: G-buffer (position + normal textures, read).
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_ssao gbuf layout"),
                entries: &[
                    bgl_texture(0),
                    bgl_texture(1),
                ],
            });

        // Group 1: output AO texture (write).
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_ssao output layout"),
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

        // Group 2: params uniform + noise texture.
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_ssao params layout"),
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
                    bgl_texture(1),
                ],
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_ssao params"),
            size: std::mem::size_of::<SsaoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create and upload 4x4 noise texture. Only `noise_view` is
        // bound into the SSAO params bind group; the underlying texture
        // stays alive via wgpu's internal Arc on the view.
        let (_noise_texture, noise_view) = Self::create_noise_texture(device, queue);

        let default_params = SsaoParams::default();
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&default_params));

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_ssao params bg"),
            layout: &params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&noise_view),
                },
            ],
        });

        // Output texture.
        let (output_texture, output_view) = Self::create_output(device, half_width, half_height);
        let output_bind_group =
            Self::create_output_bind_group(device, &output_bind_group_layout, &output_view);

        // Pipeline.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_ssao"),
            source: wgpu::ShaderSource::Wgsl(wesl::include_wesl!("rkp_ssao").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_ssao pipeline"),
            bind_group_layouts: &[
                Some(&gbuffer_bind_group_layout), // group 0
                Some(&output_bind_group_layout),  // group 1
                Some(&params_bind_group_layout),  // group 2
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_ssao"),
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
            label: Some("rkp_ssao gbuf bg"),
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

    /// Update SSAO parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &SsaoParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Dispatch the SSAO compute pass.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let gbuf_bg = match &self.gbuffer_bind_group {
            Some(bg) => bg,
            None => return,
        };

        let wg_x = (self.half_width + 7) / 8;
        let wg_y = (self.half_height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_ssao"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gbuf_bg, &[]);
        pass.set_bind_group(1, &self.output_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bind_group, &[]);
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
            label: Some("rkp_ssao output"),
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
            label: Some("rkp_ssao output bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    /// Create a 4x4 noise texture with deterministic random unit vectors.
    fn create_noise_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp_ssao noise"),
            size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Deterministic noise: 16 quasi-random unit vectors packed as Rg8Unorm.
        let mut data = [0u8; 4 * 4 * 2]; // 4x4 pixels, 2 bytes each (RG)
        for i in 0..16 {
            let angle = 2.0 * std::f32::consts::PI * (i as f32 * 0.618033988749895); // golden ratio
            let x = (angle.cos() * 0.5 + 0.5).clamp(0.0, 1.0);
            let y = (angle.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            data[i * 2] = (x * 255.0) as u8;
            data[i * 2 + 1] = (y * 255.0) as u8;
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * 2), // 4 pixels * 2 bytes
                rows_per_image: Some(4),
            },
            wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }
}

fn bgl_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
