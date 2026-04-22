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
    /// Per-tile list grid width in tiles (render_width / 8, rounded up).
    /// Shader uses this to compute `tile_idx` from a pixel coordinate.
    pub tile_count_x: u32,
    pub _pad: u32,
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
    /// Per-tile object-list offsets (prefix sum, u32). One entry per
    /// tile plus a trailing sentinel, so `tile_offsets[t+1]` is always
    /// a valid read bound for tile `t`'s list.
    tile_offsets_buffer: wgpu::Buffer,
    tile_offsets_capacity: u64,
    /// Flat list of object indices, grouped by tile. Length = sum of
    /// all per-tile counts, i.e. `tile_offsets[num_tiles]`.
    tile_object_ids_buffer: wgpu::Buffer,
    tile_object_ids_capacity: u64,
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
                    // Binding 3: lights.
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
                    // Binding 4: per-tile object-list offsets (prefix sum).
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
                    // Binding 5: per-tile object-ids flat list.
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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
            // Tile-list buffers — sized for 1 tile at start (2 u32s =
            // 8 B; sentinel offset + a trivially empty object list).
            // Both grow on `upload_tile_lists` when needed.
            tile_offsets_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march tile_offsets"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tile_offsets_capacity: 256,
            tile_object_ids_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march tile_object_ids"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tile_object_ids_capacity: 256,
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
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.tile_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.tile_object_ids_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Upload per-tile object lists. Call each frame before dispatch.
    /// Grows both buffers if needed; a reallocation invalidates the
    /// cached params bind group so we rebuild it before the next use.
    pub fn upload_tile_lists(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        offsets: &[u8],
        object_ids: &[u8],
    ) {
        let mut dirty = false;
        if (offsets.len() as u64) > self.tile_offsets_capacity {
            let mut new_cap = self.tile_offsets_capacity.max(256);
            while new_cap < offsets.len() as u64 {
                new_cap = new_cap.saturating_mul(2);
            }
            self.tile_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march tile_offsets"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.tile_offsets_capacity = new_cap;
            dirty = true;
        }
        // `object_ids` can legitimately be empty (no objects visible);
        // wgpu requires a non-zero buffer size, so we always keep the
        // tracked capacity at ≥256 B. Skip the write on an empty input —
        // shaders only read from it via `tile_offsets[t]..[t+1]`, which
        // will be (0..0) and short-circuit the loop.
        let obj_needed = object_ids.len().max(4) as u64;
        if obj_needed > self.tile_object_ids_capacity {
            let mut new_cap = self.tile_object_ids_capacity.max(256);
            while new_cap < obj_needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.tile_object_ids_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march tile_object_ids"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.tile_object_ids_capacity = new_cap;
            dirty = true;
        }
        if dirty {
            self.params_bind_group = None;
            self.try_rebuild_params_bind_group(device);
        }
        if !offsets.is_empty() {
            queue.write_buffer(&self.tile_offsets_buffer, 0, offsets);
        }
        if !object_ids.is_empty() {
            queue.write_buffer(&self.tile_object_ids_buffer, 0, object_ids);
        }
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
        tile_count_x: u32,
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
            tile_count_x,
            _pad: 0,
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
