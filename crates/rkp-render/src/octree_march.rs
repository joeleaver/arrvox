//! Octree-accelerated compute ray marcher.
//!
//! Single compute dispatch per frame — one thread per pixel. Each thread casts
//! a camera ray, traverses the octree hierarchy for each object, and writes
//! the closest hit to the G-buffer.

use crate::compile_pass_shader;

/// Stats buffer size in bytes. See the `stats` binding in
/// `shaders/octree_march.wgsl` for the layout. Expanded from 52 → 64
/// when Surface-Nets normal-reconstruction added counters at
/// stats[52..55]; 64 → 80 added the user-shader descend-body
/// breakdown counters at stats[64..72] (k-loop, AABB rejected, descent
/// run, descent miss, hit) for measuring the band-cell descent cost.
pub const STATS_U32_COUNT: usize = 80;
pub const STATS_BYTES: u64 = (STATS_U32_COUNT * 4) as u64;

/// State machine for the async stats readback. Single buffer rather
/// than a ring — stats are sampled sparsely (debug eprintln), so
/// "skip if previous map_async still in flight" is sufficient.
const STATS_MAP_IDLE: u8 = 0;
const STATS_MAP_PENDING: u8 = 1;
const STATS_MAP_READY: u8 = 2;
const STATS_MAP_FAILED: u8 = 3;

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
    /// Phase 7 Session 4b — TLAS node count. Shadow trace uses this to
    /// gate on empty TLAS (zero → skip the BVH traversal). March
    /// itself doesn't read TLAS yet; the field replaces the previous
    /// `_pad` slot.
    pub tlas_node_count: u32,
    /// Phase 8 — `1` ⇒ engine has dispatched the shadow-map march
    /// this frame; shadow trace's directional rays are now served
    /// by the shadow map sample in shade. The shadow trace skips
    /// any `light_type == 0u` light when this is set, writing 1.0
    /// into its `shadow_data` slot as an unused sentinel (shade
    /// reads `sample_shadow_map` for that branch). Spot/point
    /// lights keep the per-pixel ray-traced path either way.
    pub shadow_map_enabled: u32,
    /// Phase B-redux Phase 3a — frame time threaded into `ctx.time`
    /// so user shaders' `instance_at` hooks can derive
    /// time-dependent parameters (wind sway, etc.) without per-frame
    /// re-bake. Engine populates from `frame.shade_params_base.time`.
    pub time: f32,
    /// Phase B-redux Phase 3a — number of records in `assets[]`. The
    /// host march scans the array to find an asset whose
    /// `asset.shader_id` matches the painted leaf's material
    /// `shader_id`, mapping shader_id → proto octree at march time.
    /// Engine sets to combined_assets.len() each frame.
    pub asset_count: u32,
    /// Trailing pad to round the struct size up to the next 16-byte
    /// multiple (uniform-storage layout requirement). 11 u32s × 4 =
    /// 44 → pad to 48. Named so any future additions slot in cleanly.
    pub _pad0: u32,
}

/// The octree ray march compute pass.
pub struct OctreeMarchPass {
    pipeline: wgpu::ComputePipeline,
    /// Kept around so `reload_user_shaders` can rebuild the pipeline
    /// against the same bind-group layouts when user-shader chunks
    /// change. Phase 4c.
    pipeline_layout: wgpu::PipelineLayout,
    /// Hash of the user-shader source mix this pipeline was last
    /// built against. Comparing to the registry's `source_hash`
    /// decides whether a rebuild is needed. 0 = "default identity
    /// stubs", which is what the static template ships with.
    user_shader_source_hash: u64,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: Option<wgpu::BindGroup>,
    /// Stats buffer for profiling. See the `stats` binding doc in
    /// `shaders/octree_march.wgsl` for the slot layout.
    stats_buffer: wgpu::Buffer,
    stats_readback: wgpu::Buffer,
    /// Map state for the async stats readback. Held in an `Arc` so the
    /// `map_async` callback can flip it from the wgpu-internal worker.
    stats_map_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// Per-tile object-list offsets (prefix sum, u32). One entry per
    /// tile plus a trailing sentinel, so `tile_offsets[t+1]` is always
    /// a valid read bound for tile `t`'s list.
    tile_offsets_buffer: wgpu::Buffer,
    tile_offsets_capacity: u64,
    /// Flat list of object indices, grouped by tile. Length = sum of
    /// all per-tile counts, i.e. `tile_offsets[num_tiles]`.
    tile_object_ids_buffer: wgpu::Buffer,
    tile_object_ids_capacity: u64,
    /// Phase 7 Session 4b — TLAS node + leaf buffers, shared with
    /// `state.tlas_pass`. Shadow trace reads bindings 8 + 9 to
    /// traverse the BVH. March itself doesn't currently read them
    /// (naga DCE drops them from the march pipeline) but the layout
    /// declares them so the shared bind group can hold them for
    /// shadow trace's use.
    tlas_nodes_buffer: Option<wgpu::Buffer>,
    tlas_leaves_buffer: Option<wgpu::Buffer>,
    /// Lights buffer (shared with shade pass).
    lights_buffer: Option<wgpu::Buffer>,
    /// Materials buffer reference for bind group rebuild.
    materials_buffer: Option<wgpu::Buffer>,
    /// Phase B-redux Phase 3a — shader_params buffer (per-material
    /// 8 × f32 slots), shared with rkp_shade. Used by the host
    /// march's `instance_at` dispatcher to populate `ctx.params` for
    /// derivation.
    shader_params_buffer: Option<wgpu::Buffer>,
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
        // moved to the rkp_shadow_trace pass (half-res). Binding 3 is
        // the dedicated 32-bit pick channel — replaces the old 8-bit
        // object_id packed into the material G channel (which capped
        // the scene at 255 pickable entries).
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("march gbuffer layout"),
                entries: &[
                    bgl_storage_tex(0, wgpu::TextureFormat::Rgba32Float),
                    bgl_storage_tex(1, wgpu::TextureFormat::Rgba16Float),
                    bgl_storage_tex(2, wgpu::TextureFormat::Rg32Uint),
                    bgl_storage_tex(3, wgpu::TextureFormat::R32Uint),
                    // Glass info target — oct-packed entry normal +
                    // packed (thickness_mm, material_id). Written only
                    // when the primary ray passes through a transparent
                    // voxel; rkp_glass gates on `thickness_mm != 0`.
                    bgl_storage_tex(4, wgpu::TextureFormat::Rg32Uint),
                    // Leaf-slot target — primary hit's leaf_attr_slot.
                    // Read by rkp_shade's geodesic paint cursor.
                    bgl_storage_tex(5, wgpu::TextureFormat::R32Uint),
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
                    // Binding 8: TLAS nodes (Phase 7 Session 4b).
                    // Read by shadow trace; march doesn't use it
                    // (naga DCE drops the binding from march SPIR-V).
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 9: TLAS leaves (Phase 7 Session 4b).
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 10: shader_params — flat per-material
                    // f32 array (8 floats per material) keyed by
                    // material_id. Phase B-redux Phase 3a wires this
                    // so user-shader `instance_at` derivation can read
                    // ctx.params from the live material's slider
                    // values. Already the same buffer the shade pass
                    // binds; this just exposes it to march too.
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
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
        let module = compile_pass_shader(device, wesl::include_wesl!("octree_march"), "octree_march");

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
            pipeline_layout,
            user_shader_source_hash: 0,
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
            stats_map_state: std::sync::Arc::new(
                std::sync::atomic::AtomicU8::new(STATS_MAP_IDLE),
            ),
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
            tlas_nodes_buffer: None,
            tlas_leaves_buffer: None,
            lights_buffer: None,
            materials_buffer: None,
            shader_params_buffer: None,
        }
    }

    /// Phase 7 Session 4b — set the TLAS buffer handles. Engine calls
    /// this whenever the buffers change (typically on
    /// `tlas_pass.build_tlas` reallocation). The bind group rebuild
    /// follows the same pattern as `set_lights` / `set_materials`.
    pub fn set_tlas_buffers(
        &mut self,
        device: &wgpu::Device,
        nodes_buffer: &wgpu::Buffer,
        leaves_buffer: &wgpu::Buffer,
    ) {
        self.tlas_nodes_buffer = Some(nodes_buffer.clone());
        self.tlas_leaves_buffer = Some(leaves_buffer.clone());
        self.try_rebuild_params_bind_group(device);
    }

    /// Re-build the compute pipeline against the spliced user-shader
    /// `inst_to_local` + `inst_aabb` chunks. Returns `true` if rebuilt,
    /// `false` if `source_hash` matched and the existing pipeline was
    /// kept. Empty chunks restore the default identity-arm stubs (the
    /// "no user shader registered" path). Phase 4c.
    ///
    /// Mirrors `PrototypeBakePass::reload_user_shaders` exactly so the
    /// engine can call both with the same `frame.user_shader_source_hash`
    /// without having to track per-pass hashes.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        instance_at_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.user_shader_source_hash {
            return false;
        }
        let template = wesl::include_wesl!("octree_march");
        let source = crate::shader_composer::splice_inst_chunks(
            template, instance_at_chunk,
        );
        let module = compile_pass_shader(device, &source, "octree_march");
        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("octree_march"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.user_shader_source_hash = source_hash;
        true
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

    /// Phase B-redux Phase 3a — set the shader_params buffer (the
    /// per-material 8 × f32 slot array shared with `rkp_shade`).
    /// `instance_at` derivation reads `ctx.params` from this binding;
    /// the bind group is rebuilt to include it when both materials
    /// and lights are also set.
    pub fn set_shader_params(
        &mut self, device: &wgpu::Device, shader_params_buffer: &wgpu::Buffer,
    ) {
        self.shader_params_buffer = Some(shader_params_buffer.clone());
        self.try_rebuild_params_bind_group(device);
    }

    fn try_rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        let (
            Some(materials_buffer),
            Some(lights_buffer),
            Some(tlas_nodes),
            Some(tlas_leaves),
            Some(shader_params),
        ) = (
            &self.materials_buffer,
            &self.lights_buffer,
            &self.tlas_nodes_buffer,
            &self.tlas_leaves_buffer,
            &self.shader_params_buffer,
        ) else { return };
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
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: tlas_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: tlas_leaves.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: shader_params.as_entire_binding(),
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
    #[allow(clippy::too_many_arguments)]
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
        leaf_slot_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("march gbuffer bind group"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(material_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(pick_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(glass_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(leaf_slot_view) },
            ],
        }));
    }

    /// Clear stats buffer before dispatch.
    pub fn clear_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.stats_buffer, 0, None);
    }

    /// Copy stats to readback buffer after dispatch. Skipped when a
    /// previous `submit_stats_readback`'s map_async is still pending or
    /// the result hasn't been drained yet — encoding a copy into a
    /// mapped (or about-to-be-mapped) buffer is a wgpu validation
    /// error. The drain-then-submit cycle in the engine clears the
    /// state each frame, so under steady state we get fresh data
    /// every frame the env var is enabled; without the env var the
    /// state stays IDLE and the copy fires every frame.
    pub fn copy_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering;
        let state = self.stats_map_state.load(Ordering::Acquire);
        if state == STATS_MAP_PENDING || state == STATS_MAP_READY {
            return;
        }
        encoder.copy_buffer_to_buffer(&self.stats_buffer, 0, &self.stats_readback, 0, STATS_BYTES);
    }

    /// Schedule async readback of the stats buffer. Call AFTER
    /// `queue.submit` of the encoder containing `copy_stats`. No-op
    /// (returns immediately) if a previous map_async is still pending
    /// or hasn't been drained yet — caller can poll every frame and
    /// the readback will progress at GPU/driver pace.
    pub fn submit_stats_readback(&self) {
        use std::sync::atomic::Ordering;
        let state = self.stats_map_state.load(Ordering::Acquire);
        if state == STATS_MAP_PENDING || state == STATS_MAP_READY {
            return;
        }
        self.stats_map_state.store(STATS_MAP_PENDING, Ordering::Release);
        let state_arc = std::sync::Arc::clone(&self.stats_map_state);
        let slice = self.stats_readback.slice(0..STATS_BYTES);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let next = if result.is_ok() { STATS_MAP_READY } else { STATS_MAP_FAILED };
            state_arc.store(next, Ordering::Release);
        });
    }

    /// Try to drain the stats readback. Returns the most-recent stats
    /// snapshot if the previous `submit_stats_readback` has resolved,
    /// or `None` if no resolution is ready. After draining, the slot
    /// is freed for the next `submit_stats_readback`.
    pub fn try_drain_stats(&self) -> Option<Vec<u32>> {
        use std::sync::atomic::Ordering;
        let state = self.stats_map_state.load(Ordering::Acquire);
        if state == STATS_MAP_FAILED {
            self.stats_map_state.store(STATS_MAP_IDLE, Ordering::Release);
            return None;
        }
        if state != STATS_MAP_READY {
            return None;
        }
        let slice = self.stats_readback.slice(0..STATS_BYTES);
        let counts: Vec<u32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        self.stats_readback.unmap();
        self.stats_map_state.store(STATS_MAP_IDLE, Ordering::Release);
        Some(counts)
    }

    /// Update params and dispatch the march.
    #[allow(clippy::too_many_arguments)]
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
        tlas_node_count: u32,
        shadow_map_enabled: bool,
        time: f32,
        asset_count: u32,
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
            tlas_node_count,
            shadow_map_enabled: u32::from(shadow_map_enabled),
            time,
            asset_count,
            _pad0: 0,
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
        let src = wesl::include_wesl!("octree_march");
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
