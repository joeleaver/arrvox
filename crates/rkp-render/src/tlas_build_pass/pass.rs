//! `TlasBuildPass` — owns all GPU pipelines + buffers for the Phase 7c
//! GPU TLAS build (assemble → Morton → radix sort → Karras → propagate).
//!
//! [`TlasBuildPass::new`] builds every pipeline + storage buffer at
//! startup; [`TlasBuildPass::dispatch`] encodes the per-frame chain
//! against [`GpuTlasBuildInputs`].
//!
//! Buffer-capacity growth: `ensure_prims_capacity` /
//! `ensure_prim_keys_capacity` etc. realloc storage buffers to fit
//! `prim_count` for the frame. Initial sizes match the wgpu binding-
//! validation rule that storage buffers must exist before the bind
//! group is built (see [`super::types::TLAS_PRIMS_INITIAL_ENTRIES`]).

use super::types::{
    AssembleHostUniform, KarrasUniform, MortonUniform, RadixUniform, TlasPrim, RADIX_BUCKETS,
    RADIX_PASSES, TLAS_PRIMS_INITIAL_ENTRIES,
};

/// Pipeline holder for Phase 7c GPU TLAS build. Session 1 owns the
/// primitive-assembly pipelines and the shared output
/// (`tlas_prims_buffer` + `tlas_prim_count_buffer`); Session 2
/// adds the Morton + radix-sort pipelines and ping-pong key/value
/// buffers; later sessions extend with the Karras tree builder
/// and AABB propagation.
pub struct TlasBuildPass {
    // ── Session 1 ──────────────────────────────────────────────────
    pub host_pipeline: wgpu::ComputePipeline,
    pub host_g0_layout: wgpu::BindGroupLayout,
    pub host_g1_layout: wgpu::BindGroupLayout,
    /// Packed `array<TlasPrim>`. Capacity grows; `tlas_prim_count`
    /// holds the per-frame live count after the host assembly
    /// dispatch finishes.
    pub tlas_prims_buffer: wgpu::Buffer,
    pub tlas_prims_capacity: u32,
    /// Single-element `array<atomic<u32>>` — the assembly pass
    /// `atomicAdd`s into slot 0. Engine zeroes per frame before
    /// dispatch.
    pub tlas_prim_count_buffer: wgpu::Buffer,
    /// Per-dispatch uniform — re-uploaded per frame.
    pub host_uniform_buffer: wgpu::Buffer,

    // ── Session 2 — Morton + radix sort ───────────────────────────
    pub morton_pipeline: wgpu::ComputePipeline,
    pub morton_g0_layout: wgpu::BindGroupLayout,
    pub morton_g1_layout: wgpu::BindGroupLayout,
    pub morton_uniform_buffer: wgpu::Buffer,

    pub radix_count_pipeline: wgpu::ComputePipeline,
    pub radix_scan_pipeline: wgpu::ComputePipeline,
    pub radix_scatter_pipeline: wgpu::ComputePipeline,
    pub radix_g0_layout: wgpu::BindGroupLayout,
    pub radix_g1_layout: wgpu::BindGroupLayout,
    /// One uniform per radix pass (4 of them, one per digit shift).
    /// Engine writes all four contiguously per frame; bind group
    /// uses dynamic offset to select the right one. 16 B each;
    /// 256 B per slot for wgpu's dynamic-offset alignment.
    pub radix_uniform_buffer: wgpu::Buffer,

    /// Ping-pong key buffers. After Morton compute, `keys_a` holds
    /// the input; pass 0 writes to `keys_b`; pass 1 to `keys_a`;
    /// passes 2 and 3 alternate. With 4 passes (even count), the
    /// final sorted output lands in `keys_a`. Same for `vals_a`/
    /// `vals_b`.
    pub keys_a_buffer: wgpu::Buffer,
    pub keys_b_buffer: wgpu::Buffer,
    pub vals_a_buffer: wgpu::Buffer,
    pub vals_b_buffer: wgpu::Buffer,
    /// Capacity of each ping-pong buffer in u32 entries.
    pub keys_capacity: u32,

    /// Per-WG histogram. Size `histogram_capacity_workgroups × 256`
    /// u32s (atomic). Re-zeroed between radix passes.
    pub histogram_buffer: wgpu::Buffer,
    /// Per-WG starting offsets after the scan pass. Same shape as
    /// histogram. Mutated in-place by scatter atomics.
    pub scan_offsets_buffer: wgpu::Buffer,
    /// Capacity of histogram + scan_offsets, in workgroup slots.
    pub histogram_capacity_workgroups: u32,

    // ── Session 3 — Karras radix tree ─────────────────────────────
    pub karras_leaves_pipeline: wgpu::ComputePipeline,
    pub karras_internal_pipeline: wgpu::ComputePipeline,
    pub karras_g0_layout: wgpu::BindGroupLayout,
    pub karras_g1_layout: wgpu::BindGroupLayout,
    pub karras_uniform_buffer: wgpu::Buffer,

    // ── Session 4 — bottom-up AABB propagation ───────────────────
    /// Phase 7c.6 — atomic AABB-accumulator propagation. Three
    /// pipelines on the same module + bind layout: init clears
    /// accumulators to ±∞ sentinels, propagate walks each leaf
    /// up the parent chain applying atomicMin/Max, decode reads
    /// the final accumulator values and writes `tlas_nodes` AABBs.
    pub init_atomic_aabb_pipeline: wgpu::ComputePipeline,
    pub propagate_atomic_pipeline: wgpu::ComputePipeline,
    pub decode_aabb_pipeline: wgpu::ComputePipeline,
    /// `array<u32>` of length `2N-1`. Engine fills with
    /// `0xFFFFFFFF` (`PARENT_SENTINEL`) before each frame's tree
    /// build; `build_internal_main` overwrites with parent indices
    /// for non-root nodes.
    pub parents_buffer: wgpu::Buffer,
    /// `array<atomic<u32>>` of length `3 × (2N-1)`. One u32 slot
    /// per (node, axis) for the min accumulator. Init pass fills
    /// with the encoded +∞ sentinel; each leaf-walk-up applies
    /// `atomicMin`; decode pass reads + writes back into
    /// `tlas_nodes[i].aabb_min`.
    pub aabb_min_atomic_buffer: wgpu::Buffer,
    /// Same shape as `aabb_min_atomic_buffer` but for max.
    pub aabb_max_atomic_buffer: wgpu::Buffer,
    /// Capacity of `parents_buffer` in u32 entries.
    pub parents_capacity: u32,
    /// Capacity of each atomic-AABB buffer in u32 entries
    /// (= 3 × (2N-1) target).
    pub aabb_atomic_capacity: u32,
    /// 4-byte staging buffer used to read `tlas_prim_count[0]`
    /// back to CPU after the assembly stage. The Phase 7c V1
    /// uses a synchronous readback to drive the rest of the
    /// pipeline; this buffer is the readback target.
    pub count_staging_buffer: wgpu::Buffer,
}

/// Inputs to [`TlasBuildPass::build_gpu_tlas`]. All buffers are
/// borrowed from external owners; the build pass binds them as
/// pipeline inputs but doesn't take ownership.
pub struct GpuTlasBuildInputs<'a> {
    /// `state.renderer.scene.objects_buffer` — `array<RkpGpuInstance>`.
    pub instances_buffer: &'a wgpu::Buffer,
    pub instance_count: u32,
    /// `state.renderer.scene.assets_buffer` — `array<RkpGpuAsset>`.
    pub assets_buffer: &'a wgpu::Buffer,
    pub asset_count: u32,
    /// CPU-derived scene AABB for Morton normalization.
    /// Conservative is fine — Morton sort just needs a stable
    /// coordinate system.
    pub scene_min: [f32; 3],
    pub scene_max: [f32; 3],
}

impl TlasBuildPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let host_g0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_host g0"),
                entries: &[
                    ro_storage(0), // host_instances
                    ro_storage(1), // host_assets
                    rw_storage(2), // tlas_prims
                    rw_storage(3), // tlas_prim_count
                ],
            });
        let host_g1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_host g1"),
                entries: &[uniform_entry(0, std::mem::size_of::<AssembleHostUniform>() as u64)],
            });
        let host_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_assemble_host pipeline layout"),
            bind_group_layouts: &[Some(&host_g0_layout), Some(&host_g1_layout)],
            immediate_size: 0,
        });
        let host_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_assemble_host"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tlas_assemble_host.wgsl").into(),
            ),
        });
        let host_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_assemble_host"),
            layout: Some(&host_layout),
            module: &host_module,
            entry_point: Some("assemble_host_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prims_bytes = (TLAS_PRIMS_INITIAL_ENTRIES as u64)
            * (std::mem::size_of::<TlasPrim>() as u64);
        let tlas_prims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prims"),
            size: prims_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tlas_prim_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prim_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let host_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_assemble_host uniform"),
            size: std::mem::size_of::<AssembleHostUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Session 2 — Morton + radix sort ───────────────────────
        let morton_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_morton g0"),
            entries: &[
                ro_storage(0), // tlas_prims
                rw_storage(1), // keys_a
                rw_storage(2), // vals_a
            ],
        });
        let morton_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_morton g1"),
            entries: &[uniform_entry(0, std::mem::size_of::<MortonUniform>() as u64)],
        });
        let morton_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_morton pipeline layout"),
            bind_group_layouts: &[Some(&morton_g0_layout), Some(&morton_g1_layout)],
            immediate_size: 0,
        });
        let morton_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_morton"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tlas_morton.wgsl").into()),
        });
        let morton_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_morton"),
            layout: Some(&morton_layout),
            module: &morton_module,
            entry_point: Some("compute_morton_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let morton_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_morton uniform"),
            size: std::mem::size_of::<MortonUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radix_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_radix g0"),
            entries: &[
                ro_storage(0), // keys_in
                ro_storage(1), // vals_in
                rw_storage(2), // keys_out
                rw_storage(3), // vals_out
                rw_storage(4), // histogram
                rw_storage(5), // scan_offsets
            ],
        });
        let radix_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_radix g1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                },
                count: None,
            }],
        });
        let radix_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_radix pipeline layout"),
            bind_group_layouts: &[Some(&radix_g0_layout), Some(&radix_g1_layout)],
            immediate_size: 0,
        });
        let radix_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_radix_sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tlas_radix_sort.wgsl").into()),
        });
        let radix_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_count"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("count_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let radix_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_scan"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("scan_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let radix_scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_scatter"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Four uniforms, one per radix pass, laid out 256 B apart
        // for wgpu's dynamic-offset alignment.
        let radix_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_radix uniform"),
            size: 256 * RADIX_PASSES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_keys_capacity: u32 = 1;
        let keys_buffer_descriptor = |label: &'static str, cap: u32| wgpu::BufferDescriptor {
            label: Some(label),
            size: (cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        let keys_a_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_keys_a", initial_keys_capacity));
        let keys_b_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_keys_b", initial_keys_capacity));
        let vals_a_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_vals_a", initial_keys_capacity));
        let vals_b_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_vals_b", initial_keys_capacity));

        let initial_histogram_workgroups: u32 = 1;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_histogram"),
            size: (initial_histogram_workgroups as u64) * (RADIX_BUCKETS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scan_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_scan_offsets"),
            size: (initial_histogram_workgroups as u64) * (RADIX_BUCKETS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Session 3-4 — Karras tree + AABB propagation ──────────
        let karras_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_karras g0"),
            entries: &[
                ro_storage(0), // sorted_keys
                ro_storage(1), // sorted_vals
                ro_storage(2), // tlas_prims
                rw_storage(3), // tlas_nodes
                rw_storage(4), // tlas_leaves
                rw_storage(5), // parents
                rw_storage(6), // aabb_min_atomic
                rw_storage(7), // aabb_max_atomic
            ],
        });
        let karras_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_karras g1"),
            entries: &[uniform_entry(0, std::mem::size_of::<KarrasUniform>() as u64)],
        });
        let karras_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_karras pipeline layout"),
            bind_group_layouts: &[Some(&karras_g0_layout), Some(&karras_g1_layout)],
            immediate_size: 0,
        });
        let karras_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_karras"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tlas_karras.wgsl").into()),
        });
        let karras_leaves_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_karras_leaves"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("build_leaves_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let karras_internal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_karras_internal"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("build_internal_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let karras_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_karras uniform"),
            size: std::mem::size_of::<KarrasUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Session 4 — AABB propagation ──────────────────────────
        let init_atomic_aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_init_atomic_aabb"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("init_atomic_aabb_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let propagate_atomic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_propagate_atomic"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("propagate_atomic_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let decode_aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_decode_aabb"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("decode_aabb_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let initial_parents_capacity: u32 = 1;
        let initial_aabb_atomic_capacity: u32 = 3;
        let parents_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_parents"),
            size: (initial_parents_capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let aabb_atomic_descriptor = |label: &'static str, cap: u32| wgpu::BufferDescriptor {
            label: Some(label),
            size: (cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        let aabb_min_atomic_buffer = device.create_buffer(&aabb_atomic_descriptor(
            "tlas_aabb_min_atomic",
            initial_aabb_atomic_capacity,
        ));
        let aabb_max_atomic_buffer = device.create_buffer(&aabb_atomic_descriptor(
            "tlas_aabb_max_atomic",
            initial_aabb_atomic_capacity,
        ));
        let count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prim_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            host_pipeline,
            host_g0_layout,
            host_g1_layout,
            tlas_prims_buffer,
            tlas_prims_capacity: TLAS_PRIMS_INITIAL_ENTRIES,
            tlas_prim_count_buffer,
            host_uniform_buffer,
            morton_pipeline,
            morton_g0_layout,
            morton_g1_layout,
            morton_uniform_buffer,
            radix_count_pipeline,
            radix_scan_pipeline,
            radix_scatter_pipeline,
            radix_g0_layout,
            radix_g1_layout,
            radix_uniform_buffer,
            keys_a_buffer,
            keys_b_buffer,
            vals_a_buffer,
            vals_b_buffer,
            keys_capacity: initial_keys_capacity,
            histogram_buffer,
            scan_offsets_buffer,
            histogram_capacity_workgroups: initial_histogram_workgroups,
            karras_leaves_pipeline,
            karras_internal_pipeline,
            karras_g0_layout,
            karras_g1_layout,
            karras_uniform_buffer,
            init_atomic_aabb_pipeline,
            propagate_atomic_pipeline,
            decode_aabb_pipeline,
            parents_buffer,
            aabb_min_atomic_buffer,
            aabb_max_atomic_buffer,
            parents_capacity: initial_parents_capacity,
            aabb_atomic_capacity: initial_aabb_atomic_capacity,
            count_staging_buffer,
        }
    }

    /// Grow `parents_buffer` to fit `entries` u32s. Allocation
    /// target = `2N-1` (one parent slot per node, including leaves).
    pub fn ensure_parents_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.parents_capacity {
            return false;
        }
        let mut new_cap = self.parents_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        self.parents_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_parents"),
            size: (new_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.parents_capacity = new_cap;
        true
    }

    /// Drive the full GPU TLAS build (Sessions 1-4) end to end.
    /// Encodes assembly → readback → Morton → 4× radix → Karras
    /// leaves + internal → AABB propagation, writing the final
    /// `tlas_nodes` + `tlas_leaves` into the supplied
    /// [`crate::tlas_pass::TlasPass`] buffers (which the shadow
    /// trace already binds).
    ///
    /// Returns the actual primitive count after assembly (= number
    /// of leaves in the built TLAS). Caller stamps this into
    /// `tlas_pass.last_node_count = 2N-1` and
    /// `tlas_pass.last_leaf_count = N` so the shadow trace's empty-
    /// scene skip works (the WGSL early-outs when `tlas_node_count
    /// == 0`).
    ///
    /// V1 uses a synchronous readback between assembly and the
    /// downstream chain — `device.poll(wait_indefinitely)` blocks
    /// the calling thread for ~1 ms per frame. Acceptable for V1
    /// (we're trading 1 ms here to save 30+ ms of shadow trace);
    /// future refactor to indirect dispatch would remove the stall.
    pub fn build_gpu_tlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &GpuTlasBuildInputs,
        tlas_pass: &mut crate::tlas_pass::TlasPass,
    ) -> u32 {
        let upper_bound = inputs.instance_count;
        if upper_bound == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the assembly stage.
        self.ensure_prims_capacity(device, upper_bound);

        // Assemble + count readback. Single submit + map_async +
        // device.poll. The blocking poll is the 1 ms stall.
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build assemble"),
        });
        enc1.clear_buffer(&self.tlas_prim_count_buffer, 0, Some(4));

        // Upload assembly uniform.
        queue.write_buffer(
            &self.host_uniform_buffer,
            0,
            bytemuck::bytes_of(&AssembleHostUniform {
                instance_count: inputs.instance_count,
                asset_count: inputs.asset_count,
                prims_capacity: self.tlas_prims_capacity,
                _pad: 0,
            }),
        );

        // Host assembly bind groups + dispatch.
        if inputs.instance_count > 0 {
            let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g0"),
                layout: &self.host_g0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: inputs.instances_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: inputs.assets_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.tlas_prim_count_buffer.as_entire_binding() },
                ],
            });
            let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g1"),
                layout: &self.host_g1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.host_uniform_buffer.as_entire_binding(),
                }],
            });
            let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("assemble_host_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.host_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            let wgs = ((inputs.instance_count + 63) / 64).max(1);
            cpass.dispatch_workgroups(wgs, 1, 1);
        }

        // Copy count to staging for readback.
        enc1.copy_buffer_to_buffer(&self.tlas_prim_count_buffer, 0, &self.count_staging_buffer, 0, 4);
        queue.submit(std::iter::once(enc1.finish()));

        // Synchronous readback. Stalls the engine thread ~1 ms.
        let slice = self.count_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll for tlas count readback");
        let raw_count = {
            let view = slice.get_mapped_range();
            let c = u32::from_le_bytes(view[0..4].try_into().unwrap());
            drop(view);
            self.count_staging_buffer.unmap();
            c
        };
        // Clamp the readback to actual capacity. If the assembly
        // atomic recorded more attempted writes than the buffer
        // holds (overflow), the writes themselves were gated by
        // `if (slot >= u.prims_capacity) return;`; the counter
        // just kept incrementing for telemetry.
        let actual_count = raw_count.min(self.tlas_prims_capacity);
        let upper_bound = inputs.instance_count;
        if raw_count > upper_bound || (raw_count == 0 && upper_bound > 0) {
            eprintln!(
                "[tlas_build] suspect raw={raw_count} upper={upper_bound} host={}",
                inputs.instance_count,
            );
        }

        if actual_count == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the downstream chain.
        self.ensure_keys_capacity(device, actual_count);
        let radix_workgroups = ((actual_count + 63) / 64).max(1);
        self.ensure_histogram_capacity(device, radix_workgroups);
        self.ensure_parents_capacity(device, 2 * actual_count - 1);
        // Phase 7c.6 — atomic AABB accumulators sized 3 × (2N-1)
        // u32s each (one per axis per node). Init pass clears
        // them to ±∞ sentinels at frame start; `propagate_atomic_main`
        // walks each leaf up the parent chain applying atomicMin/Max.
        let total_nodes = (2 * actual_count).saturating_sub(1).max(1);
        let aabb_atomic_entries = total_nodes.saturating_mul(3).max(3);
        self.ensure_aabb_atomic_capacity(device, aabb_atomic_entries);
        tlas_pass.ensure_capacity(device, 2 * actual_count - 1, actual_count);

        // Upload uniforms for the chain.
        queue.write_buffer(
            &self.morton_uniform_buffer,
            0,
            bytemuck::bytes_of(&MortonUniform {
                scene_min: inputs.scene_min,
                _pad0: 0,
                scene_max: inputs.scene_max,
                prim_count: actual_count,
            }),
        );
        let radix_stride: u64 = 256;
        let mut radix_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_stride) as usize];
        for p in 0..RADIX_PASSES {
            let u = RadixUniform {
                prim_count: actual_count,
                digit_shift: p * 8,
                num_workgroups: radix_workgroups,
                _pad: 0,
            };
            let off = (p as u64 * radix_stride) as usize;
            radix_bytes[off..off + std::mem::size_of::<RadixUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        queue.write_buffer(&self.radix_uniform_buffer, 0, &radix_bytes);
        queue.write_buffer(
            &self.karras_uniform_buffer,
            0,
            bytemuck::bytes_of(&KarrasUniform {
                prim_count: actual_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        // Init parents (sentinel) + visit_counter (zero).
        let parents_init: Vec<u32> = vec![0xFFFFFFFFu32; (2 * actual_count - 1) as usize];
        queue.write_buffer(&self.parents_buffer, 0, bytemuck::cast_slice(&parents_init));
        // Phase 7c.6 — `init_atomic_aabb_main` (in the dispatch
        // chain below) writes ±∞ sentinels to both atomic
        // buffers; no CPU pre-fill needed. Just declare we're
        // about to use them at the new size.
        let _ = total_nodes;

        // Encode the chain.
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build chain"),
        });

        // Morton compute.
        let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g0"),
            layout: &self.morton_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.vals_a_buffer.as_entire_binding() },
            ],
        });
        let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g1"),
            layout: &self.morton_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.morton_uniform_buffer.as_entire_binding(),
            }],
        });
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_morton_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.morton_pipeline);
            cpass.set_bind_group(0, &morton_g0, &[]);
            cpass.set_bind_group(1, &morton_g1, &[]);
            cpass.dispatch_workgroups(radix_workgroups, 1, 1);
        }

        // Radix sort — 4 passes ping-ponging a→b→a→b.
        let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 a→b"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 b→a"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g1"),
            layout: &self.radix_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            }],
        });
        let histogram_bytes = (radix_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
        for p in 0..RADIX_PASSES {
            let g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
            let dyn_off = (p as u64 * radix_stride) as u32;
            enc2.clear_buffer(&self.histogram_buffer, 0, Some(histogram_bytes));
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix count_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_count_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scan_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scan_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scatter_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scatter_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
        }

        // Karras tree + AABB propagation. Output goes into the
        // shadow-trace consumer buffers (`tlas_pass.{nodes,leaves}_buffer`).
        let karras_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g0"),
            layout: &self.karras_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tlas_pass.nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tlas_pass.leaves_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.parents_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.aabb_min_atomic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.aabb_max_atomic_buffer.as_entire_binding() },
            ],
        });
        let karras_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g1"),
            layout: &self.karras_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.karras_uniform_buffer.as_entire_binding(),
            }],
        });
        let leaf_wgs = ((actual_count + 63) / 64).max(1);
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_leaves_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.karras_leaves_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups(leaf_wgs, 1, 1);
        }
        if actual_count >= 2 {
            let internal_wgs = (((actual_count - 1) + 63) / 64).max(1);
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("build_internal_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.karras_internal_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }
        // Phase 7c.6 — atomic AABB propagation. Three passes:
        //   1. init: clear accumulators to ±∞ sentinels.
        //   2. propagate: each leaf walks up to root, atomic-min/max
        //      into ancestors. Commutative — no thread ordering
        //      issues, no cross-buffer memory visibility needed.
        //   3. decode: read accumulators, write tlas_nodes AABBs.
        let total_node_wgs = (((2 * actual_count - 1) + 63) / 64).max(1);
        let internal_wgs = if actual_count >= 2 {
            (((actual_count - 1) + 63) / 64).max(1)
        } else {
            1
        };
        if actual_count >= 2 {
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("init_atomic_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.init_atomic_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(total_node_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("propagate_atomic_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.propagate_atomic_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(leaf_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("decode_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.decode_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }

        queue.submit(std::iter::once(enc2.finish()));

        tlas_pass.last_node_count = 2 * actual_count - 1;
        tlas_pass.last_leaf_count = actual_count;
        actual_count
    }

    /// Grow each `aabb_*_atomic_buffer` to fit `entries` u32s
    /// (= 3 × node count). Both buffers share the same capacity.
    pub fn ensure_aabb_atomic_capacity(
        &mut self,
        device: &wgpu::Device,
        entries: u32,
    ) -> bool {
        if entries <= self.aabb_atomic_capacity {
            return false;
        }
        let mut new_cap = self.aabb_atomic_capacity.max(3);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let descriptor = |label: &'static str| wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        self.aabb_min_atomic_buffer = device.create_buffer(&descriptor("tlas_aabb_min_atomic"));
        self.aabb_max_atomic_buffer = device.create_buffer(&descriptor("tlas_aabb_max_atomic"));
        self.aabb_atomic_capacity = new_cap;
        true
    }

    /// Grow `keys_*` and `vals_*` ping-pong buffers to fit
    /// `entries` u32s each. Grows with capacity doubling.
    pub fn ensure_keys_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.keys_capacity {
            return false;
        }
        let mut new_cap = self.keys_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * 4;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let descriptor = |label: &'static str| wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        };
        self.keys_a_buffer = device.create_buffer(&descriptor("tlas_keys_a"));
        self.keys_b_buffer = device.create_buffer(&descriptor("tlas_keys_b"));
        self.vals_a_buffer = device.create_buffer(&descriptor("tlas_vals_a"));
        self.vals_b_buffer = device.create_buffer(&descriptor("tlas_vals_b"));
        self.keys_capacity = new_cap;
        true
    }

    /// Grow `histogram_buffer` and `scan_offsets_buffer` to fit
    /// `workgroups` per-WG slices (each `RADIX_BUCKETS` u32 wide).
    pub fn ensure_histogram_capacity(&mut self, device: &wgpu::Device, workgroups: u32) -> bool {
        if workgroups <= self.histogram_capacity_workgroups {
            return false;
        }
        let mut new_cap = self.histogram_capacity_workgroups.max(1);
        while new_cap < workgroups {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * (RADIX_BUCKETS as u64) * 4;
        self.histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_histogram"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.scan_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_scan_offsets"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.histogram_capacity_workgroups = new_cap;
        true
    }

    /// Grow `tlas_prims_buffer` to fit `entries` primitives. Grows
    /// with capacity doubling (mirrors the pattern in
    /// `OctreeMarchPass::ensure_us_tile_grid_capacity`). Returns
    /// `true` if the buffer reallocated — caller is responsible for
    /// invalidating any cached bind groups that referenced the old
    /// handle.
    pub fn ensure_prims_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.tlas_prims_capacity {
            return false;
        }
        let mut new_cap = self.tlas_prims_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * (std::mem::size_of::<TlasPrim>() as u64);
        self.tlas_prims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prims"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.tlas_prims_capacity = new_cap;
        true
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(min_size),
        },
        count: None,
    }
}
