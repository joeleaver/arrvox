//! Atmosphere LUT management — precomputed transmittance and multi-scattering.
//!
//! Creates and dispatches compute shaders for:
//! - Transmittance LUT (256×64, rgba16float) — precomputed once
//! - Multi-Scattering LUT (32×32, rgba16float) — precomputed once
//!
//! Both are recomputed only when atmosphere parameters change (which is rare).
//! The shade pass samples these LUTs instead of per-pixel ray marching.

use crate::compile_pass_shader;

const TRANSMITTANCE_W: u32 = 256;
const TRANSMITTANCE_H: u32 = 64;
const MULTISCATTER_W: u32 = 32;
const MULTISCATTER_H: u32 = 32;
const SKY_VIEW_W: u32 = 192;
const SKY_VIEW_H: u32 = 108;
const AP_SIZE: u32 = 32;

/// Atmosphere params for per-frame LUTs (sky view + aerial perspective).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmosphereFrameParams {
    pub sun_dir: [f32; 3],
    pub sun_intensity: f32,
    pub camera_altitude: f32,
    /// Linear-RGB albedo used for below-horizon ground radiance in the
    /// sky-view LUT. Lets empty scenes show "ground-through-atmosphere"
    /// rather than a black void past the geometric horizon.
    pub ground_albedo: [f32; 3],
    pub cam_pos: [f32; 3],
    pub _pad1b: f32,
    pub cam_forward: [f32; 3],
    pub _pad2: f32,
    pub cam_right: [f32; 3],
    pub _pad3: f32,
    pub cam_up: [f32; 3],
    pub _pad4: f32,
}

/// Atmosphere LUT pass — owns transmittance and multi-scattering textures.
pub struct RkpAtmospherePass {
    // Transmittance LUT.
    transmittance_pipeline: wgpu::ComputePipeline,
    transmittance_bind_group: wgpu::BindGroup,
    pub transmittance_texture: wgpu::Texture,
    pub transmittance_view: wgpu::TextureView,

    // Multi-scattering LUT.
    multiscatter_pipeline: wgpu::ComputePipeline,
    multiscatter_bind_group: wgpu::BindGroup,
    pub multiscatter_texture: wgpu::Texture,
    pub multiscatter_view: wgpu::TextureView,

    /// Sampler for LUT lookups (linear, clamp-to-edge).
    pub lut_sampler: wgpu::Sampler,

    // Sky View LUT (per-frame).
    sky_view_pipeline: wgpu::ComputePipeline,
    sky_view_bind_group: wgpu::BindGroup,
    pub sky_view_texture: wgpu::Texture,
    pub sky_view_view: wgpu::TextureView,

    // Aerial Perspective LUT (per-frame).
    ap_pipeline: wgpu::ComputePipeline,
    ap_bind_group: wgpu::BindGroup,
    pub ap_texture: wgpu::Texture,
    pub ap_view: wgpu::TextureView,

    // Shared params buffer for per-frame LUTs.
    frame_params_buffer: wgpu::Buffer,

    /// Whether precomputed LUTs need recomputation.
    dirty: bool,
}

impl RkpAtmospherePass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Create textures.
        let (transmittance_texture, transmittance_view) =
            Self::create_lut(device, "transmittance_lut", TRANSMITTANCE_W, TRANSMITTANCE_H);
        let (multiscatter_texture, multiscatter_view) =
            Self::create_lut(device, "multiscatter_lut", MULTISCATTER_W, MULTISCATTER_H);

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmo_lut_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // --- Transmittance pipeline ---
        let transmittance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transmittance_lut layout"),
                entries: &[
                    // binding 0: output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
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

        let trans_module = compile_pass_shader(
            device,
            wesl::include_wesl!("rkp_transmittance_lut"),
            "rkp_transmittance_lut",
        );
        let transmittance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("transmittance_lut"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transmittance_lut layout"),
                bind_group_layouts: &[Some(&transmittance_bind_group_layout)],
                immediate_size: 0,
            })),
            module: &trans_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let transmittance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transmittance_lut bg"),
            layout: &transmittance_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&transmittance_view),
            }],
        });

        // --- Multi-scattering pipeline ---
        let multiscatter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("multiscatter_lut layout"),
                entries: &[
                    // binding 0: transmittance LUT (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // binding 2: output storage texture
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

        let ms_module = compile_pass_shader(
            device,
            wesl::include_wesl!("rkp_multiscatter_lut"),
            "rkp_multiscatter_lut",
        );
        let multiscatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("multiscatter_lut"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("multiscatter_lut layout"),
                bind_group_layouts: &[Some(&multiscatter_bind_group_layout)],
                immediate_size: 0,
            })),
            module: &ms_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let multiscatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("multiscatter_lut bg"),
            layout: &multiscatter_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&multiscatter_view),
                },
            ],
        });

        // --- Shared per-frame params buffer ---
        let frame_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo_frame_params"),
            size: std::mem::size_of::<AtmosphereFrameParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shared bind group layout for per-frame LUTs (sky view + aerial perspective).
        let per_frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo per-frame layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });

        // --- Sky View LUT ---
        let (sky_view_texture, sky_view_view) = Self::create_lut(device, "sky_view_lut", SKY_VIEW_W, SKY_VIEW_H);

        let sv_module = compile_pass_shader(
            device,
            wesl::include_wesl!("rkp_sky_view_lut"),
            "rkp_sky_view_lut",
        );
        let sky_view_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sky_view_lut"), layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky_view layout"), bind_group_layouts: &[Some(&per_frame_layout)], immediate_size: 0,
            })), module: &sv_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        let sky_view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_view bg"), layout: &per_frame_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&transmittance_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&multiscatter_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&lut_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&sky_view_view) },
            ],
        });

        // --- Aerial Perspective LUT ---
        let ap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aerial_perspective_lut"),
            size: wgpu::Extent3d { width: AP_SIZE, height: AP_SIZE, depth_or_array_layers: AP_SIZE },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ap_view = ap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // AP uses a different layout (3D output texture).
        let ap_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aerial_perspective layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 }, count: None },
            ],
        });

        let ap_module = compile_pass_shader(
            device,
            wesl::include_wesl!("rkp_aerial_perspective_lut"),
            "rkp_aerial_perspective_lut",
        );
        let ap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aerial_perspective_lut"), layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aerial_perspective layout"), bind_group_layouts: &[Some(&ap_bind_group_layout)], immediate_size: 0,
            })), module: &ap_module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        let ap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aerial_perspective bg"), layout: &ap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&transmittance_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&multiscatter_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&lut_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&ap_view) },
            ],
        });

        let _ = per_frame_layout;
        Self {
            transmittance_pipeline,
            transmittance_bind_group,
            transmittance_texture,
            transmittance_view,
            multiscatter_pipeline,
            multiscatter_bind_group,
            multiscatter_texture,
            multiscatter_view,
            lut_sampler,
            sky_view_pipeline,
            sky_view_bind_group,
            sky_view_texture,
            sky_view_view,
            ap_pipeline,
            ap_bind_group,
            ap_texture,
            ap_view,
            frame_params_buffer,
            dirty: true,
        }
    }

    /// Mark LUTs for recomputation (call when atmosphere params change).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Dispatch LUT computation if needed. Call before rendering.
    pub fn dispatch_if_dirty(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.dirty { return; }
        self.dirty = false;

        // 1. Transmittance LUT.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transmittance_lut"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transmittance_pipeline);
            pass.set_bind_group(0, &self.transmittance_bind_group, &[]);
            pass.dispatch_workgroups(
                (TRANSMITTANCE_W + 7) / 8,
                (TRANSMITTANCE_H + 7) / 8,
                1,
            );
        }

        // 2. Multi-scattering LUT (depends on transmittance LUT).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multiscatter_lut"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.multiscatter_pipeline);
            pass.set_bind_group(0, &self.multiscatter_bind_group, &[]);
            // Each pixel dispatches 64 threads (workgroup size 1×1×64).
            pass.dispatch_workgroups(MULTISCATTER_W, MULTISCATTER_H, 1);
        }
    }

    /// Dispatch per-frame LUTs (sky view + aerial perspective). Call every frame.
    pub fn dispatch_per_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &AtmosphereFrameParams,
    ) {
        queue.write_buffer(&self.frame_params_buffer, 0, bytemuck::bytes_of(params));

        // Sky View LUT.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky_view_lut"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sky_view_pipeline);
            pass.set_bind_group(0, &self.sky_view_bind_group, &[]);
            pass.dispatch_workgroups(
                (SKY_VIEW_W + 7) / 8,
                (SKY_VIEW_H + 7) / 8,
                1,
            );
        }

        // Aerial Perspective LUT.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aerial_perspective_lut"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ap_pipeline);
            pass.set_bind_group(0, &self.ap_bind_group, &[]);
            pass.dispatch_workgroups(
                (AP_SIZE + 7) / 8,
                (AP_SIZE + 7) / 8,
                AP_SIZE,
            );
        }
    }

    fn create_lut(
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
}
