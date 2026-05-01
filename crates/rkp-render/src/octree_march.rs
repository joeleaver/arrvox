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

/// Per-tile user-shader entry (Phase 6). Each entry carries enough
/// metadata for the host march to dispatch the user-shader path
/// without going through a per-instance `RkpGpuInstance` — those are
/// gone for user-shader paths in Phase 6.
///
/// Wire format must match the WGSL struct in
/// `octree_march.wgsl::UserShaderTileEntry`. 16 bytes — std140
/// alignment puts the four `u32`s tightly packed.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UserShaderTileEntry {
    /// Index into the scene's `assets[]` array. Looks up the user-
    /// shader proto (octree_root, max_depth, voxel_size, shader_id).
    pub asset_id: u32,
    /// u32 offset into `instance_pool` for this instance's per-instance
    /// state. The user's hooks read from this offset.
    pub instance_state_offset: u32,
    /// Painted host material id (low 16 bits). The proto bake writes
    /// `material_primary = 0` into its leaf_attrs; the host march
    /// overrides with this on hit (locked V1 host-material inheritance).
    pub material_id: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<UserShaderTileEntry>() == 16);

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

    // ── Phase 7d — directional shadow tile cull ──────────────────
    /// `1` if the engine populated the shadow_tile_bitmap this
    /// frame; `0` skips the cull (e.g., no directional light, or
    /// the bitmap pass was disabled). Shadow trace gates on this.
    pub shadow_tile_enabled: u32,
    /// Index in `lights[]` of the directional light the bitmap was
    /// built for. Other lights take the full BVH path.
    pub shadow_tile_light_idx: u32,
    pub shadow_tile_grid_w: u32,
    pub shadow_tile_grid_h: u32,
    /// World-space origin of the light-space coordinate system.
    /// Mirrors `ShadowTileUniform.light_origin`.
    pub shadow_tile_origin: [f32; 3],
    /// Tile size in world units.
    pub shadow_tile_size: f32,
    pub shadow_tile_right: [f32; 3],
    pub _pad0: u32,
    pub shadow_tile_up: [f32; 3],
    pub _pad1: u32,
}

/// Phase 7d — packed shadow tile cull parameters passed to
/// `OctreeMarchPass::dispatch` per frame. Mirrors the same fields
/// the engine uploads into [`crate::shadow_tile_cull_pass::ShadowTileUniform`]
/// so the mark pass and shadow trace see consistent geometry.
#[derive(Debug, Clone, Copy)]
pub struct ShadowTileCullParams {
    pub enabled: u32,
    pub light_idx: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub origin: [f32; 3],
    pub tile_size: f32,
    pub right: [f32; 3],
    pub up: [f32; 3],
}

impl ShadowTileCullParams {
    /// Sentinel for frames without an active shadow-tile bitmap
    /// (no directional light, or the cull pass was skipped). The
    /// shadow trace gates on `enabled == 0` and falls back to the
    /// full BVH path.
    pub fn disabled() -> Self {
        Self {
            enabled: 0,
            light_idx: 0,
            grid_w: 0,
            grid_h: 0,
            origin: [0.0; 3],
            tile_size: 1.0,
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }
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
    /// Phase 6 — per-tile user-shader entry offsets (prefix sum).
    /// GPU-built each frame by the user-shader tile-cull pass. One
    /// `u32` per tile + a sentinel; slice for tile `t` is
    /// `us_tile_entries[us_tile_offsets[t]..us_tile_offsets[t+1]]`.
    /// Sized in pair with `tile_offsets_buffer` so the host march can
    /// loop both in parallel without separate dispatches.
    pub us_tile_offsets_buffer: wgpu::Buffer,
    pub us_tile_offsets_capacity: u64,
    /// Phase 6 — flat per-tile `UserShaderTileEntry` array
    /// (16 B each: asset_id, instance_state_offset, material_id,
    /// _pad). Built GPU-side by the tile-cull scatter pass; each entry
    /// carries enough metadata for the host march to dispatch the
    /// user-shader path without going through a per-instance
    /// `RkpGpuInstance` (those are gone for user-shader paths in
    /// Phase 6).
    pub us_tile_entries_buffer: wgpu::Buffer,
    pub us_tile_entries_capacity: u64,
    /// Phase 6 Session 3 — per-tile atomic counts populated by the
    /// tile-cull count pass; consumed by the prefix-sum pass to build
    /// `us_tile_offsets`. Reset to zero every frame. One `u32` per tile.
    pub us_tile_counts_buffer: wgpu::Buffer,
    pub us_tile_counts_capacity: u64,
    /// Phase 6 Session 3 — per-tile atomic cursor for the scatter pass.
    /// Engine initializes to a copy of `us_tile_offsets[..tile_count]`
    /// before scatter; the pass `atomicAdd`s into it to claim slots
    /// inside `us_tile_entries[]`. One `u32` per tile.
    pub us_tile_scatter_cursor_buffer: wgpu::Buffer,
    pub us_tile_scatter_cursor_capacity: u64,
    /// Phase 7 Session 4b — TLAS node + leaf buffers, shared with
    /// `state.tlas_pass`. Shadow trace reads bindings 8 + 9 to
    /// traverse the BVH. March itself doesn't currently read them
    /// (naga DCE drops them from the march pipeline) but the layout
    /// declares them so the shared bind group can hold them for
    /// shadow trace's use.
    tlas_nodes_buffer: Option<wgpu::Buffer>,
    tlas_leaves_buffer: Option<wgpu::Buffer>,
    /// Phase 7d — shadow tile cull bitmap. The shadow trace's
    /// directional-light path looks up its tile bit before
    /// descending the BVH; bit clear → skip.
    shadow_tile_bitmap_buffer: Option<wgpu::Buffer>,
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
                    // Binding 6: per-tile user-shader entry offsets
                    // (prefix sum). Phase 6 — GPU-built, parallel to
                    // bindings 4/5 but for user-shader instances.
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 7: per-tile user-shader entries
                    // (`UserShaderTileEntry` records — 16 B each:
                    // asset_id, instance_state_offset, material_id,
                    // _pad). Slice for tile `t` is
                    // `us_tile_entries[us_tile_offsets[t]..us_tile_offsets[t+1]]`.
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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
                    // Binding 10: shadow tile bitmap (Phase 7d).
                    // 2 KB u32 array. Per-pixel directional shadow
                    // ray short-circuits the BVH descent when its
                    // tile bit is 0. Read by the shadow trace; the
                    // primary octree march doesn't read it (naga
                    // DCE drops the binding from the march SPIR-V).
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
            // Phase 6 — start at single-entry placeholders; tile-cull
            // GPU pass writes these per frame and grows on overflow.
            // Initial layout encodes "no user-shader instances" (one
            // tile, one zero offset → empty entry list).
            us_tile_offsets_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_offsets"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            us_tile_offsets_capacity: 256,
            us_tile_entries_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_entries"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            us_tile_entries_capacity: 256,
            us_tile_counts_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_counts"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            us_tile_counts_capacity: 256,
            us_tile_scatter_cursor_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_scatter_cursor"),
                size: 256,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            us_tile_scatter_cursor_capacity: 256,
            tlas_nodes_buffer: None,
            tlas_leaves_buffer: None,
            shadow_tile_bitmap_buffer: None,
            lights_buffer: None,
            materials_buffer: None,
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

    /// Phase 7d — set the shadow tile bitmap buffer. Call after
    /// `ShadowTileCullPass::new`; the buffer handle is stable so a
    /// single call suffices unless the engine reconstructs the
    /// pass.
    pub fn set_shadow_tile_buffer(
        &mut self,
        device: &wgpu::Device,
        bitmap_buffer: &wgpu::Buffer,
    ) {
        self.shadow_tile_bitmap_buffer = Some(bitmap_buffer.clone());
        self.try_rebuild_params_bind_group(device);
    }

    /// Re-build the compute pipeline against the spliced user-shader
    /// `inst_to_local` + `inst_aabb` chunks. Returns `true` if rebuilt,
    /// `false` if `source_hash` matched and the existing pipeline was
    /// kept. Empty chunks restore the default identity-arm stubs (the
    /// "no user shader registered" path). Phase 4c.
    ///
    /// Mirrors `InstanceMarchPass::reload_user_shaders` /
    /// `PrototypeBakePass::reload_user_shaders` exactly so the engine
    /// can call all three with the same `frame.user_shader_source_hash`
    /// without having to track per-pass hashes.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        inst_to_local_chunk: &str,
        inst_aabb_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.user_shader_source_hash {
            return false;
        }
        let template = include_str!("shaders/octree_march.wgsl");
        let source = crate::shader_composer::splice_inst_chunks(
            template, inst_to_local_chunk, inst_aabb_chunk,
        );
        validate_wgsl(&source, "octree_march");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("octree_march"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
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

    fn try_rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        let (
            Some(materials_buffer),
            Some(lights_buffer),
            Some(tlas_nodes),
            Some(tlas_leaves),
            Some(shadow_tile_bitmap),
        ) = (
            &self.materials_buffer,
            &self.lights_buffer,
            &self.tlas_nodes_buffer,
            &self.tlas_leaves_buffer,
            &self.shadow_tile_bitmap_buffer,
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
                    binding: 6,
                    resource: self.us_tile_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.us_tile_entries_buffer.as_entire_binding(),
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
                    resource: shadow_tile_bitmap.as_entire_binding(),
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

    /// Phase 6 Session 3 — grow the per-tile user-shader buffers to fit
    /// `tile_count` tiles. The four buffers are sized in lock-step:
    ///
    /// * `us_tile_offsets`  — `(tile_count + 1) × 4 B` (prefix sum).
    /// * `us_tile_counts`   — `tile_count × 4 B` (per-tile atomic).
    /// * `us_tile_scatter_cursor` — `tile_count × 4 B` (per-tile atomic).
    /// * `us_tile_entries`  — sized separately by `ensure_us_tile_entries_capacity`
    ///   since entry count is determined by post-cull totals, not tile count.
    ///
    /// Returns `true` if any buffer was reallocated. The caller is
    /// responsible for invalidating any cached bind group that references
    /// these buffers (`params_bind_group` is rebuilt automatically by
    /// `try_rebuild_params_bind_group`).
    pub fn ensure_us_tile_grid_capacity(
        &mut self,
        device: &wgpu::Device,
        tile_count: u32,
    ) -> bool {
        let offsets_needed = ((tile_count as u64) + 1) * 4;
        let cells_needed = (tile_count as u64) * 4;
        let mut dirty = false;
        if offsets_needed > self.us_tile_offsets_capacity {
            let mut new_cap = self.us_tile_offsets_capacity.max(256);
            while new_cap < offsets_needed { new_cap = new_cap.saturating_mul(2); }
            self.us_tile_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_offsets"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.us_tile_offsets_capacity = new_cap;
            dirty = true;
        }
        if cells_needed > self.us_tile_counts_capacity {
            let mut new_cap = self.us_tile_counts_capacity.max(256);
            while new_cap < cells_needed { new_cap = new_cap.saturating_mul(2); }
            self.us_tile_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_counts"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.us_tile_counts_capacity = new_cap;
            dirty = true;
        }
        if cells_needed > self.us_tile_scatter_cursor_capacity {
            let mut new_cap = self.us_tile_scatter_cursor_capacity.max(256);
            while new_cap < cells_needed { new_cap = new_cap.saturating_mul(2); }
            self.us_tile_scatter_cursor_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march us_tile_scatter_cursor"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.us_tile_scatter_cursor_capacity = new_cap;
            dirty = true;
        }
        if dirty {
            self.params_bind_group = None;
            self.try_rebuild_params_bind_group(device);
        }
        dirty
    }

    /// Phase 6 Session 3 — grow `us_tile_entries_buffer` to fit
    /// `entry_count` × 16 B `UserShaderTileEntry` records. Sized
    /// separately from the per-tile buffers because entry count comes
    /// from post-cull totals (sum of per-tile counts), not tile count.
    /// Returns `true` if reallocated.
    pub fn ensure_us_tile_entries_capacity(
        &mut self,
        device: &wgpu::Device,
        entry_count: u32,
    ) -> bool {
        let needed = (entry_count.max(1) as u64) * 16;
        if needed <= self.us_tile_entries_capacity {
            return false;
        }
        let mut new_cap = self.us_tile_entries_capacity.max(256);
        while new_cap < needed { new_cap = new_cap.saturating_mul(2); }
        self.us_tile_entries_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("march us_tile_entries"),
            size: new_cap,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.us_tile_entries_capacity = new_cap;
        self.params_bind_group = None;
        self.try_rebuild_params_bind_group(device);
        true
    }

    /// Phase 6 Session 3 — copy the prefix-summed `us_tile_offsets`
    /// values for tiles `[0, tile_count)` into `us_tile_scatter_cursor`.
    /// Each tile's slot starts at its prefix offset; the scatter pass's
    /// atomicAdd produces slots in `[offset[t], offset[t+1])`.
    pub fn init_scatter_cursor(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tile_count: u32,
    ) {
        let bytes = (tile_count as u64) * 4;
        if bytes == 0 { return; }
        encoder.copy_buffer_to_buffer(
            &self.us_tile_offsets_buffer, 0,
            &self.us_tile_scatter_cursor_buffer, 0,
            bytes,
        );
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
        shadow_tile_cull: ShadowTileCullParams,
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
            shadow_tile_enabled: shadow_tile_cull.enabled,
            shadow_tile_light_idx: shadow_tile_cull.light_idx,
            shadow_tile_grid_w: shadow_tile_cull.grid_w,
            shadow_tile_grid_h: shadow_tile_cull.grid_h,
            shadow_tile_origin: shadow_tile_cull.origin,
            shadow_tile_size: shadow_tile_cull.tile_size,
            shadow_tile_right: shadow_tile_cull.right,
            _pad0: 0,
            shadow_tile_up: shadow_tile_cull.up,
            _pad1: 0,
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
