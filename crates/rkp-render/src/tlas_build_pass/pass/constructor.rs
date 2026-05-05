//! `TlasBuildPass::new` — wgpu pipeline + buffer setup for all four
//! TLAS build sessions: assembly, Morton/radix sort, Karras tree, AABB propagation.

use super::super::types::{
    AssembleHostUniform, MortonUniform, RadixUniform, TlasPrim, TlasState, RADIX_BUCKETS,
    RADIX_PASSES, TLAS_DISPATCH_ARG_SLOTS, TLAS_DISPATCH_ARG_STRIDE, TLAS_PRIMS_INITIAL_ENTRIES,
};

use super::TlasBuildPass;
use super::{ro_storage, rw_storage, uniform_entry};

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
        let host_module = crate::compile_pass_shader(
            device, wesl::include_wesl!("tlas_assemble_host"), "tlas_assemble_host",
        );
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
            entries: &[
                uniform_entry(0, std::mem::size_of::<MortonUniform>() as u64),
                // `tlas_state` is bound here as UNIFORM rather than
                // STORAGE so the karras pass (which already binds 8
                // storage buffers — wgpu's default per-stage limit)
                // can fit it without bumping the device limit.
                uniform_entry(1, std::mem::size_of::<TlasState>() as u64),
            ],
        });
        let morton_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_morton pipeline layout"),
            bind_group_layouts: &[Some(&morton_g0_layout), Some(&morton_g1_layout)],
            immediate_size: 0,
        });
        let morton_module = crate::compile_pass_shader(
            device, wesl::include_wesl!("tlas_morton"), "tlas_morton",
        );
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                    },
                    count: None,
                },
                uniform_entry(1, std::mem::size_of::<TlasState>() as u64), // tlas_state
            ],
        });
        let radix_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_radix pipeline layout"),
            bind_group_layouts: &[Some(&radix_g0_layout), Some(&radix_g1_layout)],
            immediate_size: 0,
        });
        let radix_module = crate::compile_pass_shader(
            device, wesl::include_wesl!("tlas_radix_sort"), "tlas_radix_sort",
        );
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
            entries: &[uniform_entry(0, std::mem::size_of::<TlasState>() as u64)], // tlas_state
        });
        let karras_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_karras pipeline layout"),
            bind_group_layouts: &[Some(&karras_g0_layout), Some(&karras_g1_layout)],
            immediate_size: 0,
        });
        let karras_module = crate::compile_pass_shader(
            device, wesl::include_wesl!("tlas_karras"), "tlas_karras",
        );
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
        // ── tlas_state — shared dynamic-counts buffer ────────────────
        // Written by `tlas_compute_dispatch_args.wesl` once per frame;
        // read by morton/radix/karras shaders. 16 B `TlasState`.
        //
        // Carries BOTH `STORAGE` and `UNIFORM` usages: the writer
        // pipeline binds it as `var<storage, read_write>` (so it
        // can write the new counts), while the consumer pipelines
        // bind it as `var<uniform>` to keep within wgpu's default
        // 8-storage-buffer-per-stage limit (the karras pass already
        // saturates that limit with its 8 storage buffers in g0).
        let tlas_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_state"),
            size: std::mem::size_of::<TlasState>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── tlas_dispatch_args — packed indirect-dispatch buffer ─────
        // 7 slots × 12 B (u32 x/y/z workgroup counts) = 84 B. STORAGE
        // for the writer; INDIRECT for the chain dispatches' calls
        // to `dispatch_workgroups_indirect`.
        let dispatch_args_bytes = TLAS_DISPATCH_ARG_SLOTS * TLAS_DISPATCH_ARG_STRIDE;
        let tlas_dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_dispatch_args"),
            size: dispatch_args_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── dispatch_args pipeline ───────────────────────────────────
        let dispatch_args_g0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_compute_dispatch_args g0"),
                entries: &[
                    rw_storage(0), // tlas_prim_count (atomic u32)
                    rw_storage(1), // tlas_state
                    rw_storage(2), // tlas_dispatch_args
                ],
            });
        let dispatch_args_g1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_compute_dispatch_args g1"),
                entries: &[uniform_entry(0, std::mem::size_of::<AssembleHostUniform>() as u64)],
            });
        let dispatch_args_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tlas_compute_dispatch_args pipeline layout"),
                bind_group_layouts: &[
                    Some(&dispatch_args_g0_layout),
                    Some(&dispatch_args_g1_layout),
                ],
                immediate_size: 0,
            });
        let dispatch_args_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("tlas_compute_dispatch_args"),
            "tlas_compute_dispatch_args",
        );
        let dispatch_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tlas_compute_dispatch_args"),
                layout: Some(&dispatch_args_pipeline_layout),
                module: &dispatch_args_module,
                entry_point: Some("compute_dispatch_args_main"),
                compilation_options: Default::default(),
                cache: None,
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
            init_atomic_aabb_pipeline,
            propagate_atomic_pipeline,
            decode_aabb_pipeline,
            parents_buffer,
            aabb_min_atomic_buffer,
            aabb_max_atomic_buffer,
            parents_capacity: initial_parents_capacity,
            aabb_atomic_capacity: initial_aabb_atomic_capacity,
            tlas_state_buffer,
            tlas_dispatch_args_buffer,
            dispatch_args_pipeline,
            dispatch_args_g0_layout,
            dispatch_args_g1_layout,
        }
    }
}
