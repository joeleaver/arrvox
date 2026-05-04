//! `RkpVolumetricPass::new` — wgpu pipeline + buffer + texture setup
//! for fog march, cloud march, sun-atten, history update, and composite.

use std::sync::{Arc, atomic::{AtomicBool, AtomicU32}};

use crate::compile_pass_shader;

use super::{CloudParams, RkpVolumetricPass, VolumetricParams};

impl RkpVolumetricPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // Fog march layout (3 bindings): params, depth, fog_out.
        let fog_march_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol fog march layout"),
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

        // Cloud march layout (6 bindings): params, depth, cloud_out, cloud_params,
        // history, history_sampler.
        let cloud_march_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol cloud march layout"),
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
                    // 1: depth buffer
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
                    // 2: cloud output
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
                    // 4: history texture — bilateral-gathered manually, so
                    // filterable=false (the shader uses textureLoad).
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
        let (output_texture, output_view) =
            Self::create_texture(device, "vol output", width, height, wgpu::TextureFormat::Rgba16Float);

        // Fog march pipeline.
        let fog_module = compile_pass_shader(device, wesl::include_wesl!("rkp_fog_march"), "rkp_fog_march");
        let fog_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol fog march"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vol fog march pipeline"),
                bind_group_layouts: &[Some(&fog_march_bind_group_layout)],
                immediate_size: 0,
            })),
            module: &fog_module,
            entry_point: Some("fog_march"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Cloud march pipeline.
        let cloud_module = compile_pass_shader(device, wesl::include_wesl!("rkp_cloud_march"), "rkp_cloud_march");
        let cloud_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol cloud march"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vol cloud march pipeline"),
                bind_group_layouts: &[Some(&cloud_march_bind_group_layout)],
                immediate_size: 0,
            })),
            module: &cloud_module,
            entry_point: Some("cloud_march"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Composite pipeline.
        let composite_module = compile_pass_shader(device, wesl::include_wesl!("rkp_vol_composite"), "rkp_vol_composite");
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
        let history_update_module = compile_pass_shader(
            device, wesl::include_wesl!("rkp_vol_history_update"), "rkp_vol_history_update",
        );
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
        let sun_atten_module = compile_pass_shader(
            device, wesl::include_wesl!("rkp_cloud_sun_atten"), "rkp_cloud_sun_atten",
        );
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
            fog_march_pipeline,
            fog_march_bind_group_layout,
            fog_march_bind_group: None,
            cloud_march_pipeline,
            cloud_march_bind_group_layout,
            cloud_march_bind_group: None,
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
}
