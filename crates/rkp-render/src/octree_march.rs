//! Octree-accelerated compute ray marcher.
//!
//! Single compute dispatch per frame — one thread per pixel. Each thread casts
//! a camera ray, traverses the octree hierarchy for each object, and writes
//! the closest hit to the G-buffer.

use crate::validate_wgsl;

/// Stats buffer size in bytes (52 × u32). See the `stats` binding in
/// `shaders/octree_march.wgsl` for the layout.
const STATS_U32_COUNT: usize = 52;
const STATS_BYTES: u64 = (STATS_U32_COUNT * 4) as u64;

/// Uniform parameters for the march shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchParams {
    pub object_count: u32,
    pub mode: u32,
    pub shadow_max_steps: u32,
    pub num_lights: u32,
}

/// The octree ray march compute pass.
pub struct OctreeMarchPass {
    pipeline: wgpu::ComputePipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: Option<wgpu::BindGroup>,
    /// Stats buffer for profiling (44 atomic u32s — see shader comment at stats binding).
    stats_buffer: wgpu::Buffer,
    stats_readback: wgpu::Buffer,
    /// Screen-space AABB buffer for tile culling.
    screen_aabbs_buffer: wgpu::Buffer,
    /// Lights buffer (shared with shade pass).
    lights_buffer: Option<wgpu::Buffer>,
    /// Materials buffer reference for bind group rebuild.
    materials_buffer: Option<wgpu::Buffer>,
}

impl OctreeMarchPass {
    /// Create the march pass.
    ///
    /// `scene_bind_group_layout`: group 0 layout (from RkpScene).
    pub fn new(
        device: &wgpu::Device,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Group 1: G-buffer storage textures (write-only). Shadow output
        // moved to the rkp_shadow_trace pass (half-res).
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
                size: STATS_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stats_readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march stats readback"),
                size: STATS_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            screen_aabbs_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march screen_aabbs"),
                size: 16 * 32, // 32 objects × vec4<f32>
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            lights_buffer: None,
            materials_buffer: None,
        }
    }

    /// Set the materials buffer. Call after materials are uploaded/resized.
    pub fn set_materials(&mut self, device: &wgpu::Device, materials_buffer: &wgpu::Buffer) {
        self.materials_buffer = Some(materials_buffer.clone());
        self.try_rebuild_params_bind_group(device);
    }

    /// Set the lights buffer. Call after lights are uploaded/resized.
    pub fn set_lights(&mut self, device: &wgpu::Device, lights_buffer: &wgpu::Buffer) {
        self.lights_buffer = Some(lights_buffer.clone());
        self.try_rebuild_params_bind_group(device);
    }

    fn try_rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        let (Some(materials_buffer), Some(lights_buffer)) =
            (&self.materials_buffer, &self.lights_buffer) else { return };
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lights_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Upload screen-space AABBs for tile culling. Call each frame before dispatch.
    pub fn upload_screen_aabbs(&self, queue: &wgpu::Queue, data: &[u8]) {
        queue.write_buffer(&self.screen_aabbs_buffer, 0, data);
    }

    /// Expose the params bind group layout so the shadow_trace pass can
    /// share the march's params + materials + stats + lights bindings.
    pub fn params_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.params_bind_group_layout
    }

    /// The params bind group itself, for external passes that dispatch
    /// with the same layout (currently rkp_shadow_trace).
    pub fn params_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.params_bind_group.as_ref()
    }

    /// Set the G-buffer textures. Call on init and after resize. Shadows are
    /// traced in a separate half-res pass (`rkp_shadow_trace`) that no longer
    /// lives in this pipeline.
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
        encoder.copy_buffer_to_buffer(&self.stats_buffer, 0, &self.stats_readback, 0, STATS_BYTES);
    }

    /// Read stats from readback buffer. Call after device.poll().
    ///
    /// Prints once per second (assuming 60fps) with per-phase depth histograms
    /// and a hit-footprint histogram. This is intentionally verbose — the goal
    /// is to drive the mipped-octree / LOD-cutoff decision, not light telemetry.
    pub fn read_stats(&self, device: &wgpu::Device, total_pixels: u32, frame_idx: u64) {
        if frame_idx % 60 != 0 || frame_idx == 0 { return; }
        let slice = self.stats_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let vals: &[u32] = bytemuck::cast_slice(&data);
            let total_steps = vals[0];
            let hit_pixels = vals[2];
            let max_steps = vals[3];

            let surface: &[u32] = &vals[4..16];
            let normal:  &[u32] = &vals[16..28];
            let shadow:  &[u32] = &vals[28..40];
            let foot:    &[u32] = &vals[40..44];
            let leaf_attr_reads = vals[44] as u64;
            let voxel_pool_reads = vals[45] as u64;
            let color_pool_reads = vals[46] as u64;
            let materials_reads = vals[47] as u64;

            let sum = |h: &[u32]| -> u64 { h.iter().map(|&x| x as u64).sum() };
            let weighted = |h: &[u32]| -> f64 {
                let s = sum(h);
                if s == 0 { return 0.0; }
                let w: u64 = h.iter().enumerate().map(|(i, &c)| i as u64 * c as u64).sum();
                w as f64 / s as f64
            };
            // Total node reads for a phase: sum over buckets of (level+1)*count.
            // (Each octree_lookup that terminates at level L reads L+1 nodes.)
            let phase_node_reads = |h: &[u32]| -> u64 {
                h.iter().enumerate().map(|(i, &c)| (i as u64 + 1) * c as u64).sum()
            };

            let total_lookups = sum(surface) + sum(normal) + sum(shadow);
            let avg_steps = if total_pixels > 0 { total_steps as f32 / total_pixels as f32 } else { 0.0 };

            eprintln!(
                "[march] hits {}/{}  avg_steps {:.1}  max_steps {}  total_lookups {}M",
                hit_pixels, total_pixels, avg_steps, max_steps, total_lookups / 1_000_000,
            );
            eprintln!(
                "[descents surface] avg_depth {:.2}  {}",
                weighted(surface), format_histogram(surface),
            );
            eprintln!(
                "[descents normal ] avg_depth {:.2}  {}",
                weighted(normal), format_histogram(normal),
            );
            eprintln!(
                "[descents shadow ] avg_depth {:.2}  {}",
                weighted(shadow), format_histogram(shadow),
            );
            let foot_total: u64 = foot.iter().map(|&x| x as u64).sum();
            let pct = |n: u32| -> f32 {
                if foot_total == 0 { 0.0 } else { 100.0 * n as f32 / foot_total as f32 }
            };
            eprintln!(
                "[footprint] <1px:{:.0}%  1-2px:{:.0}%  2-4px:{:.0}%  >=4px:{:.0}%  (n={})",
                pct(foot[0]), pct(foot[1]), pct(foot[2]), pct(foot[3]), foot_total,
            );

            // Per-buffer byte traffic per frame. Octree reads come from the
            // depth histograms; other buffers have direct atomic counters.
            let octree_reads = phase_node_reads(surface) + phase_node_reads(normal) + phase_node_reads(shadow);
            let mb = |bytes: u64| -> f64 { bytes as f64 / (1024.0 * 1024.0) };
            let octree_bytes     = octree_reads       * 4;  // 4 B per node
            let leaf_attr_bytes  = leaf_attr_reads    * 8;  // LeafAttr = 8 B
            let voxel_bytes      = voxel_pool_reads   * 8;  // VoxelSample = 8 B
            let color_bytes      = color_pool_reads   * 4;  // packed color u32
            let materials_bytes  = materials_reads    * 32; // GpuMaterial ~32 B
            let total_bytes      = octree_bytes + leaf_attr_bytes + voxel_bytes + color_bytes + materials_bytes;
            eprintln!(
                "[bandwidth/frame] octree {:>6.1}M reads {:>6.1} MiB  leaf_attr {:>6.1}M {:>6.1} MiB  voxel {:>6.1}M {:>6.1} MiB  color {:>6.1}M {:>5.1} MiB  mat {:>6.1}M {:>6.1} MiB",
                octree_reads as f64 / 1e6, mb(octree_bytes),
                leaf_attr_reads as f64 / 1e6, mb(leaf_attr_bytes),
                voxel_pool_reads as f64 / 1e6, mb(voxel_bytes),
                color_pool_reads as f64 / 1e6, mb(color_bytes),
                materials_reads as f64 / 1e6, mb(materials_bytes),
            );
            eprintln!(
                "[bandwidth/frame] total {:.1} MiB (at 60 fps → {:.1} GB/s if uncached)",
                mb(total_bytes), (total_bytes as f64 * 60.0) / 1e9,
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
        shadow_max_steps: u32,
        num_lights: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        // Update params.
        let params = MarchParams {
            object_count,
            mode,
            shadow_max_steps,
            num_lights,
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

#[cfg(test)]
mod tests {
    #[test]
    fn octree_march_shader_is_valid_wgsl() {
        let src = include_str!("shaders/octree_march.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}

/// Render a 12-bucket histogram as "L0:N L1:N ... L11:N", skipping empty tail buckets.
fn format_histogram(h: &[u32]) -> String {
    let last_nonzero = h.iter().rposition(|&x| x > 0).unwrap_or(0);
    let mut s = String::new();
    for (i, &v) in h.iter().take(last_nonzero + 1).enumerate() {
        if !s.is_empty() { s.push(' '); }
        if v >= 1_000_000 {
            s.push_str(&format!("L{}:{:.1}M", i, v as f32 / 1_000_000.0));
        } else if v >= 1_000 {
            s.push_str(&format!("L{}:{}k", i, v / 1_000));
        } else {
            s.push_str(&format!("L{}:{}", i, v));
        }
    }
    s
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
