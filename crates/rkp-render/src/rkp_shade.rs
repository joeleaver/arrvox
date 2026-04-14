//! RKIPatch deferred PBR shading compute pass.
//!
//! Reads G-buffer + shadow/AO texture, evaluates Cook-Torrance PBR with direct
//! lighting, hemisphere ambient, AO, and emission. Writes final HDR color.

/// The deferred PBR shading pass.
pub struct RkpShadePass {
    pipeline: wgpu::ComputePipeline,
    pub gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    pub ssao_bind_group_layout: wgpu::BindGroupLayout,
    pub output_bind_group_layout: wgpu::BindGroupLayout,
    pub shade_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub atmo_bind_group_layout: wgpu::BindGroupLayout,
    /// HDR output texture (full-res, Rgba16Float).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    output_bind_group: wgpu::BindGroup,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    ssao_bind_group: Option<wgpu::BindGroup>,
    shade_bind_group: Option<wgpu::BindGroup>,
    camera_bind_group: Option<wgpu::BindGroup>,
    atmo_bind_group: Option<wgpu::BindGroup>,
    width: u32,
    height: u32,
}

/// Shading parameters uniform.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadeParams {
    pub num_lights: u32,
    pub ambient_intensity: f32,
    pub camera_altitude: f32,
    pub sun_intensity: f32,
    pub sky_color_top: [f32; 3],
    pub _pad0: f32,
    pub sky_color_horizon: [f32; 3],
    pub _pad1: f32,
    pub sun_dir: [f32; 3],
    pub _pad2: f32,
    pub ambient_color: [f32; 3],
    pub _pad3: f32,
}

impl Default for ShadeParams {
    fn default() -> Self {
        Self {
            num_lights: 0,
            ambient_intensity: 0.3,
            camera_altitude: 100.0,
            sun_intensity: 20.0,
            sky_color_top: [0.4, 0.6, 1.0],
            _pad0: 0.0,
            sky_color_horizon: [0.8, 0.85, 0.9],
            _pad1: 0.0,
            sun_dir: [0.5, 0.7, 0.5],
            _pad2: 0.0,
            ambient_color: [0.1, 0.15, 0.25],
            _pad3: 0.0,
        }
    }
}

/// Per-light GPU data.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub direction: [f32; 4],
    pub params: [f32; 4],
}

/// Per-material GPU data.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emission_strength: f32,
    pub opacity: f32,
}

impl RkpShadePass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uint_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let depth_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // Group 0: G-buffer (depth, normal, material). World position is
        // reconstructed from depth in the shader.
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade gbuf"),
                entries: &[depth_texture_entry(0), texture_entry(1), uint_texture_entry(2)],
            });

        // Group 1: SSAO texture (shadow was removed alongside the compute
        // march; a future triangle-based shadow pass will reintroduce it).
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade ssao"),
                entries: &[texture_entry(0)],
            });

        // Group 2: output HDR texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade output"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Group 3: shade params + lights + materials
        let shade_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade params"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        // Group 4: camera
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade camera"),
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

        // Group 5: atmosphere LUTs
        let atmo_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade atmo"),
                entries: &[
                    // Atmosphere LUTs — all filterable.
                    Self::filterable_tex_2d(0),  // transmittance LUT
                    Self::filterable_tex_2d(1),  // multi-scatter LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    Self::filterable_tex_2d(3),  // sky view LUT
                    Self::filterable_tex_3d(4),  // aerial perspective LUT
                ],
            });

        // Output texture.
        let (output_texture, output_view) = Self::create_output(device, width, height);
        let output_bind_group = Self::create_output_bind_group(device, &output_bind_group_layout, &output_view);

        // Pipeline.
        let shade_src = include_str!("shaders/rkp_shade.wgsl");
        crate::validate_wgsl(shade_src, "rkp_shade");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_shade"),
            source: wgpu::ShaderSource::Wgsl(shade_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_shade pipeline"),
            bind_group_layouts: &[
                Some(&gbuffer_bind_group_layout),
                Some(&ssao_bind_group_layout),
                Some(&output_bind_group_layout),
                Some(&shade_bind_group_layout),
                Some(&camera_bind_group_layout),
                Some(&atmo_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_shade"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            gbuffer_bind_group_layout,
            ssao_bind_group_layout,
            output_bind_group_layout,
            shade_bind_group_layout,
            camera_bind_group_layout,
            atmo_bind_group_layout,
            output_texture,
            output_view,
            output_bind_group,
            gbuffer_bind_group: None,
            ssao_bind_group: None,
            shade_bind_group: None,
            camera_bind_group: None,
            atmo_bind_group: None,
            width,
            height,
        }
    }

    /// Set G-buffer views. `depth_view` replaces the old position view —
    /// world position is reconstructed from depth + inverse_view_proj.
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade gbuf bg"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(material_view) },
            ],
        }));
    }

    /// Set the SSAO texture view.
    pub fn set_ssao(&mut self, device: &wgpu::Device, ssao_view: &wgpu::TextureView) {
        self.ssao_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade ssao bg"),
            layout: &self.ssao_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(ssao_view),
            }],
        }));
    }

    /// Set shading data (params uniform, lights buffer, materials buffer).
    pub fn set_shade_data(
        &mut self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        materials_buffer: &wgpu::Buffer,
    ) {
        self.shade_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade params bg"),
            layout: &self.shade_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: materials_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Set camera uniform buffer.
    /// Set atmosphere LUT textures (all 4 LUTs + sampler).
    pub fn set_atmosphere_luts(
        &mut self,
        device: &wgpu::Device,
        transmittance_view: &wgpu::TextureView,
        multiscatter_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        sky_view_view: &wgpu::TextureView,
        ap_view: &wgpu::TextureView,
    ) {
        self.atmo_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade atmo bg"),
            layout: &self.atmo_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(transmittance_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(multiscatter_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sky_view_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ap_view) },
            ],
        }));
    }

    pub fn set_camera(&mut self, device: &wgpu::Device, camera_buffer: &wgpu::Buffer) {
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade camera bg"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Dispatch the shading pass.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_with_timestamps(encoder, None);
    }

    pub fn dispatch_with_timestamps(&self, encoder: &mut wgpu::CommandEncoder, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>) {
        let gbuf = match &self.gbuffer_bind_group { Some(bg) => bg, None => return };
        let sao = match &self.ssao_bind_group { Some(bg) => bg, None => return };
        let shade = match &self.shade_bind_group { Some(bg) => bg, None => return };
        let cam = match &self.camera_bind_group { Some(bg) => bg, None => return };
        let atmo = match &self.atmo_bind_group { Some(bg) => bg, None => return };

        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_shade"),
            timestamp_writes: timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gbuf, &[]);
        pass.set_bind_group(1, sao, &[]);
        pass.set_bind_group(2, &self.output_bind_group, &[]);
        pass.set_bind_group(3, shade, &[]);
        pass.set_bind_group(4, cam, &[]);
        pass.set_bind_group(5, atmo, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Point the shade pass at an external output texture (e.g., the engine's
    /// shading HDR texture). Rebuilds the output bind group to write there.
    pub fn set_output_view(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) {
        self.output_bind_group =
            Self::create_output_bind_group(device, &self.output_bind_group_layout, view);
    }

    /// Resize output texture.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (tex, view) = Self::create_output(device, width, height);
        self.output_texture = tex;
        self.output_view = view;
        self.output_bind_group =
            Self::create_output_bind_group(device, &self.output_bind_group_layout, &self.output_view);
    }

    fn filterable_tex_2d(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
            }, count: None,
        }
    }

    fn filterable_tex_3d(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
            }, count: None,
        }
    }

    fn create_output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp_shade output"),
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

    fn create_output_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade output bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }
}
