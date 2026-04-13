//! Atmosphere LUT management — precomputed transmittance and multi-scattering.
//!
//! Creates and dispatches compute shaders for:
//! - Transmittance LUT (256×64, rgba16float) — precomputed once
//! - Multi-Scattering LUT (32×32, rgba16float) — precomputed once
//!
//! Both are recomputed only when atmosphere parameters change (which is rare).
//! The shade pass samples these LUTs instead of per-pixel ray marching.

use crate::validate_wgsl;

const TRANSMITTANCE_W: u32 = 256;
const TRANSMITTANCE_H: u32 = 64;
const MULTISCATTER_W: u32 = 32;
const MULTISCATTER_H: u32 = 32;

/// Atmosphere LUT pass — owns transmittance and multi-scattering textures.
pub struct RkpAtmospherePass {
    // Transmittance LUT.
    transmittance_pipeline: wgpu::ComputePipeline,
    transmittance_bind_group_layout: wgpu::BindGroupLayout,
    transmittance_bind_group: wgpu::BindGroup,
    pub transmittance_texture: wgpu::Texture,
    pub transmittance_view: wgpu::TextureView,

    // Multi-scattering LUT.
    multiscatter_pipeline: wgpu::ComputePipeline,
    multiscatter_bind_group_layout: wgpu::BindGroupLayout,
    multiscatter_bind_group: wgpu::BindGroup,
    pub multiscatter_texture: wgpu::Texture,
    pub multiscatter_view: wgpu::TextureView,

    /// Sampler for LUT lookups (linear, clamp-to-edge).
    pub lut_sampler: wgpu::Sampler,

    /// Whether LUTs need recomputation.
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

        let trans_src = include_str!("shaders/rkp_transmittance_lut.wgsl");
        validate_wgsl(trans_src, "rkp_transmittance_lut");
        let trans_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_transmittance_lut"),
            source: wgpu::ShaderSource::Wgsl(trans_src.into()),
        });
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

        let ms_src = include_str!("shaders/rkp_multiscatter_lut.wgsl");
        validate_wgsl(ms_src, "rkp_multiscatter_lut");
        let ms_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_multiscatter_lut"),
            source: wgpu::ShaderSource::Wgsl(ms_src.into()),
        });
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

        Self {
            transmittance_pipeline,
            transmittance_bind_group_layout,
            transmittance_bind_group,
            transmittance_texture,
            transmittance_view,
            multiscatter_pipeline,
            multiscatter_bind_group_layout,
            multiscatter_bind_group,
            multiscatter_texture,
            multiscatter_view,
            lut_sampler,
            dirty: true, // compute on first frame
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
