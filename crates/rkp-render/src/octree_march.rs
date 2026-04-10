//! Octree-accelerated compute ray marcher.
//!
//! Single compute dispatch per frame — one thread per pixel. Each thread casts
//! a camera ray, traverses the octree hierarchy for each object, and writes
//! the closest hit to the G-buffer.

use crate::validate_wgsl;

/// Uniform parameters for the march shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchParams {
    pub object_count: u32,
    pub mode: u32,     // 0 = full (hit + normal), 1 = normal-only (reads position from G-buffer)
    pub shadow_max_steps: u32,
    pub _pad: u32,
    pub light_dir: [f32; 3],
    pub _pad2: f32,
}

/// The octree ray march compute pass.
pub struct OctreeMarchPass {
    pipeline: wgpu::ComputePipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: Option<wgpu::BindGroup>,
    /// Stats buffer for profiling (4 atomic u32s).
    stats_buffer: wgpu::Buffer,
    stats_readback: wgpu::Buffer,
    /// Screen-space AABB buffer for tile culling.
    screen_aabbs_buffer: wgpu::Buffer,
}

impl OctreeMarchPass {
    /// Create the march pass.
    ///
    /// `scene_bind_group_layout`: group 0 layout (from RkpScene).
    pub fn new(
        device: &wgpu::Device,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Group 1: G-buffer storage textures (write-only).
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("march gbuffer layout"),
                entries: &[
                    bgl_storage_tex(0, wgpu::TextureFormat::Rgba32Float),
                    bgl_storage_tex(1, wgpu::TextureFormat::Rgba16Float),
                    bgl_storage_tex(2, wgpu::TextureFormat::Rg32Uint),
                ],
            });

        // Group 2: march params + materials palette.
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("march params layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("march params"),
            size: std::mem::size_of::<MarchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pipeline.
        let shader_src = include_str!("shaders/octree_march.wgsl");
        validate_wgsl(shader_src, "octree_march");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("octree_march"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("octree_march pipeline layout"),
            bind_group_layouts: &[
                Some(scene_bind_group_layout),         // group 0
                Some(&gbuffer_bind_group_layout),      // group 1
                Some(&params_bind_group_layout),       // group 2
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("octree_march"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            gbuffer_bind_group_layout,
            gbuffer_bind_group: None,
            params_bind_group_layout,
            params_buffer,
            params_bind_group: None,
            stats_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march stats"),
                size: 16, // 4 × u32
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stats_readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march stats readback"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            screen_aabbs_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march screen_aabbs"),
                size: 16 * 32, // 32 objects × vec4<f32>
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Set the materials buffer. Call after materials are uploaded/resized.
    pub fn set_materials(&mut self, device: &wgpu::Device, materials_buffer: &wgpu::Buffer) {
        self.rebuild_params_bind_group(device, materials_buffer);
    }

    fn rebuild_params_bind_group(&mut self, device: &wgpu::Device, materials_buffer: &wgpu::Buffer) {
        self.params_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("march params+materials bind group"),
            layout: &self.params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.screen_aabbs_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Upload screen-space AABBs for tile culling. Call each frame before dispatch.
    pub fn upload_screen_aabbs(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(&self.screen_aabbs_buffer, 0, data);
    }

    /// Set the G-buffer textures. Call on init and after resize.
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("march gbuffer bind group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(material_view) },
            ],
        }));
    }

    /// Clear stats buffer before dispatch.
    pub fn clear_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.stats_buffer, 0, None);
    }

    /// Copy stats to readback buffer after dispatch.
    pub fn copy_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.stats_buffer, 0, &self.stats_readback, 0, 16);
    }

    /// Read stats from readback buffer. Call after device.poll().
    pub fn read_stats(&self, device: &wgpu::Device, total_pixels: u32, frame_idx: u64) {
        if frame_idx % 60 != 0 || frame_idx == 0 { return; }
        let slice = self.stats_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::Poll);
        if let Ok(Ok(())) = rx.try_recv() {
            let data = slice.get_mapped_range();
            let vals: &[u32] = bytemuck::cast_slice(&data);
            let total_steps = vals[0];
            let total_lookups = vals[1];
            let hit_pixels = vals[2];
            let max_steps = vals[3];
            let avg_steps = if total_pixels > 0 { total_steps as f32 / total_pixels as f32 } else { 0.0 };
            let avg_lookups = if total_pixels > 0 { total_lookups as f32 / total_pixels as f32 } else { 0.0 };
            eprintln!(
                "[march stats] avg_steps={:.1} avg_lookups={:.1} max_steps={} hit_pixels={}/{} total_lookups={}M",
                avg_steps, avg_lookups, max_steps, hit_pixels, total_pixels,
                total_lookups / 1_000_000,
            );
            drop(data);
            self.stats_readback.unmap();
        }
    }

    /// Update params and dispatch the march.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        scene_bind_group: &wgpu::BindGroup,
        object_count: u32,
        width: u32,
        height: u32,
        mode: u32,
        light_dir: [f32; 3],
        shadow_max_steps: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        // Update params.
        let params = MarchParams {
            object_count,
            mode,
            shadow_max_steps,
            _pad: 0,
            light_dir,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Dispatch.
        if self.gbuffer_bind_group.is_none() || self.params_bind_group.is_none() {
            eprintln!("[OctreeMarch] SKIP: gbuf={} params={}", self.gbuffer_bind_group.is_some(), self.params_bind_group.is_some());
        }
        if let (Some(gbuffer_bg), Some(params_bg)) = (&self.gbuffer_bind_group, &self.params_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("octree_march"),
                timestamp_writes: timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, scene_bind_group, &[]);
            pass.set_bind_group(1, gbuffer_bg, &[]);
            pass.set_bind_group(2, params_bg, &[]);
            pass.dispatch_workgroups(
                (width + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
    }
}

fn create_prev_texture(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn bgl_texture(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_storage_tex_rw(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::ReadWrite,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn bgl_storage_tex(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
