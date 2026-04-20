//! Octree-accelerated compute ray marcher.
//!
//! Single compute dispatch per frame — one thread per pixel. Each thread casts
//! a camera ray, traverses the octree hierarchy for each object, and writes
//! the closest hit to the G-buffer.

use crate::validate_wgsl;

/// Stats buffer size in bytes (64 × u32). See the `stats` binding in
/// `shaders/octree_march.wgsl` for the layout. Expanded from 52 when
/// the Surface-Nets normal-reconstruction POC added counters at
/// stats[52..55].
const STATS_U32_COUNT: usize = 64;
const STATS_BYTES: u64 = (STATS_U32_COUNT * 4) as u64;

/// Uniform parameters for the march shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchParams {
    pub object_count: u32,
    pub mode: u32,
    pub shadow_max_steps: u32,
    pub num_lights: u32,
    /// LOD gate: `1` → read `.y` from `octree_nodes` at branches and
    /// early-exit when the node's projected screen footprint falls
    /// below ~1 pixel. `0` → descend every branch to a terminal node
    /// (pre-LOD behavior, kept as an A/B lever for correctness tests
    /// and as a runtime kill-switch).
    pub lod_enabled: u32,
    /// Surface-Nets normal gate: `1` → reconstruct per-voxel normal at
    /// render time from the 3³ in-brick occupancy neighborhood. `0` →
    /// use the baked octahedral normal from `LeafAttr`. A/B toggle for
    /// the POC.
    pub surfacenet_enabled: u32,
    pub _pad: [u32; 2],
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
    /// Screen-space AABB buffer for tile culling. Grown on upload to
    /// fit the scene's current GpuObject count.
    screen_aabbs_buffer: wgpu::Buffer,
    /// Tracked capacity for `screen_aabbs_buffer` (bytes). Avoids
    /// querying `buffer.size()` so a stale validation handle can't
    /// trip a false-overrun error.
    screen_aabbs_capacity: u64,
    /// Cached params bind group needs to be invalidated when
    /// `screen_aabbs_buffer` is recreated since it holds a ref to the
    /// old handle.
    screen_aabbs_dirty: bool,
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
                // Initial capacity for 32 objects × vec4<f32>. Grown
                // by `upload_screen_aabbs` when the scene exceeds it.
                size: 16 * 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            screen_aabbs_capacity: 16 * 32,
            screen_aabbs_dirty: false,
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

    /// Upload screen-space AABBs for tile culling. Call each frame
    /// before dispatch. Grows the buffer if `data` exceeds the current
    /// capacity (each new GpuObject = +16 bytes), and rebuilds the
    /// params bind group so it points at the new handle.
    pub fn upload_screen_aabbs(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let needed = data.len() as u64;
        if needed > self.screen_aabbs_capacity {
            let mut new_cap = self.screen_aabbs_capacity.max(16 * 32);
            while new_cap < needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.screen_aabbs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march screen_aabbs"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.screen_aabbs_capacity = new_cap;
            self.screen_aabbs_dirty = true;
            // Cached bind group references the OLD buffer — invalidate
            // so the next `try_rebuild_params_bind_group` call (via
            // set_materials / set_lights at frame start, or our own
            // explicit rebuild below) picks up the new handle.
            self.params_bind_group = None;
            self.try_rebuild_params_bind_group(device);
        }
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

    // NOTE: the march shader writes verbose per-frame counters
    // (descent histograms, bandwidth tallies, skin-march probe) into
    // `stats_buffer`; they used to be blocking-read and eprintln'd
    // here every 60 frames. That log has been retired in favor of the
    // engine-side `ProfilingHistory`. The GPU scaffolding (stats
    // buffer + clear/copy) is still wired so the shader keeps
    // compiling; a future change will route these counters through
    // async readback into `ProfilingHistory::counters` for the panel
    // and MCP.

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
        lod_enabled: bool,
        surfacenet_enabled: bool,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        // Update params.
        let params = MarchParams {
            object_count,
            mode,
            shadow_max_steps,
            num_lights,
            lod_enabled: if lod_enabled { 1 } else { 0 },
            surfacenet_enabled: if surfacenet_enabled { 1 } else { 0 },
            _pad: [0; 2],
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
