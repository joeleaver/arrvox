//! BFS dispatcher — owns the GPU pipelines, the transient buffers
//! they read/write (active queue, fill_task pool, overflow counters,
//! per-region atomic counters, region uniforms, level uniforms), and
//! the per-frame `dispatch_regions` encoder routine.
//!
//! ## Pipeline shape
//!
//! Two compute pipelines off one shader module:
//!   - `classify_main` (per-level — workgroup size 64) — reads the
//!     active queue at level L, classifies each cell, atomically
//!     allocates octree-node / brick / leaf-attr / fill-task slots
//!     out of the SHARED global pool, writes child cells into the
//!     active queue at L+1.
//!   - `brick_fill_main` (one workgroup per fill task) — reads the
//!     `fill_task_pool` filled by classify, runs the user's
//!     `dispatch_user_generate` per occupied cell, writes brick + leaf-attr.
//!
//! Both share group-0 (ten storage bindings: octree_nodes, brick_pool,
//! leaf_attr_pool, four atomic counters, active queue + count,
//! fill_task pool + counter, overflow counters), group-1 (region
//! uniforms array), group-2 (per-level uniform with dynamic offset).
//!
//! ## Per-frame contract
//!
//! `dispatch_regions(region_uniforms, topology_dirty_count, …)` — the
//! caller passes regions ordered as
//! `[topology_dirty | fill_only_dirty]` (clean regions are dropped).
//! Topology-dirty regions get an initial `ActiveCell` seeded into
//! `active_queue[L=0]` plus reset octree + fill-task counters;
//! fill-only regions reuse the cached classify output and only get
//! their brick / leaf-attr counters reset.

use crate::shader_composer::UserShaderInfo;
use crate::validate_wgsl;

use super::cache::{FILL_TASK_BUCKET_MAX, MAX_GLOBAL_FILL_TASKS};
use super::overflow::{
    OverflowReadback, OVERFLOW_BUFFER_BYTES, OVERFLOW_COUNTER_COUNT,
};
use super::region::RegionUniform;

/// Hard ceiling on octree depth. Mirrors `MAX_DEPTH` in
/// `user_shader_geom.wgsl` — the active-queue / counter buffers are
/// sized for this many levels (+1 for the root). Bumping requires
/// growing the queue buffers and the WGSL constant in lockstep.
pub const MAX_DEPTH: u32 = 8;

/// Worst-case active cells held per level in the BFS queue,
/// multiplexed across all regions processed in one frame. Multiplied
/// by `MAX_DEPTH+1` for the global queue buffer.
///
/// 2 M cells per level. With V13 collapse-deepest into inline
/// children, the dominant queue is at L=max_depth-1 (e.g. L=4 for
/// grass at depth 5). For 512 tiled paint regions at depth 5 with
/// band 0.5 m, L=4 cells ≈ 16² × 10 layers = 2.5K per tile → 1.3 M
/// total. 1 M was right at the threshold; 2 M gives margin.
///
/// Total queue storage: 9 × 2 M × 32 B = 576 MB. Heavy but
/// allocated once at startup.
const PER_LEVEL_QUEUE_CAP: u32 = 2097152;

/// Maximum simultaneous regions in a single frame. Bound here (not in
/// the engine layer) because `RegionUniform` is bound as a
/// `array<RegionUniform>` storage binding and we need a fixed cap to
/// keep the buffer under wgpu's `max_storage_buffer_binding_size`. At
/// 240 B per uniform × 1024 = 240 KB — trivial.
pub const MAX_REGIONS: u32 = 1024;

/// Workgroup size for `classify_main`. Determines how many workgroups
/// we dispatch per level: `PER_LEVEL_QUEUE_CAP / CLASSIFY_WG_SIZE`
/// regardless of real active count (per-thread early-out).
const CLASSIFY_WG_SIZE: u32 = 64;

/// Stride between per-level uniforms in `level_uniforms_buffer`.
/// 256 B fits typical wgpu min-uniform-buffer-offset-alignment.
const LEVEL_UNIFORM_STRIDE: u64 = 256;

/// Sentinel value the brick_fill kernel checks in the first u32 of a
/// `BrickFillTask` slot to early-out on unfilled tail entries. CPU
/// pre-fills the topology-dirty region's `[0, fill_task_block_size)`
/// to this value; classify overwrites valid slots, leaving the tail
/// for fill workgroups to skip.
const FILL_TASK_SENTINEL: u32 = 0xFFFFFFFE;

/// Per-dispatch state for `classify_main`. Re-uploaded between levels.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LevelUniform {
    pub current_level: u32,
    pub per_level_cap: u32,
    pub max_active_per_level: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<LevelUniform>() == 16);

/// One BFS-side initial active cell (one per region per dispatch
/// chain). Mirrors the WGSL `ActiveCell` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ActiveCell {
    octree_offset: u32,
    region_index: u32,
    center: [f32; 3],
    half_extent: f32,
    _pad0: u32,
    _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<ActiveCell>() == 32);

/// Compose the geom-build WGSL with a user `generate` chunk. Empty
/// chunk leaves the in-tree identity stub in place.
pub fn compose_geom_source(user_chunk: &str) -> String {
    let geom_src = include_str!("../shaders/user_shader_geom.wgsl");
    crate::shader_composer::splice_const_marker(
        geom_src,
        "USER_GENERATE_DISPATCH",
        user_chunk,
    )
}

/// Holds the BFS pipelines, the transient resources they own, and the
/// per-frame buffers (active queue, fill_task_pool, level uniforms,
/// overflow counters).
pub struct UserShaderPass {
    group0_layout: wgpu::BindGroupLayout,
    group1_layout: wgpu::BindGroupLayout,
    group2_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    classify_pipeline: wgpu::ComputePipeline,
    fill_pipeline: wgpu::ComputePipeline,

    /// Per-region atomic counters. Each is `array<atomic<u32>>` of
    /// length `MAX_REGIONS`. Indexed by `region_index` in the
    /// dispatch's regions array.
    octree_alloc_buffer: wgpu::Buffer,
    brick_alloc_buffer: wgpu::Buffer,
    leaf_attr_alloc_buffer: wgpu::Buffer,
    fill_task_alloc_buffer: wgpu::Buffer,

    /// Active-cell queue — `(MAX_DEPTH+1) * PER_LEVEL_QUEUE_CAP`
    /// `ActiveCell`s. Written by CPU at level 0 (only for
    /// topology-dirty regions) and by `classify_main` at L > 0.
    active_queue_buffer: wgpu::Buffer,
    active_count_buffer: wgpu::Buffer,

    /// Persistent fill-task pool — `MAX_GLOBAL_FILL_TASKS` `BrickFillTask`
    /// slots. Each cached region owns a contiguous block within this
    /// buffer. classify writes tasks into the region's block;
    /// brick_fill reads from there.
    fill_task_pool_buffer: wgpu::Buffer,

    /// Overflow counters — async readback ring on the CPU side.
    overflow_buffer: wgpu::Buffer,
    overflow_readback: OverflowReadback,

    /// Region uniforms — `array<RegionUniform>` storage binding,
    /// sized for `MAX_REGIONS` slots.
    regions_buffer: wgpu::Buffer,

    /// Per-level uniform buffer with `MAX_DEPTH+1` slots at
    /// `LEVEL_UNIFORM_STRIDE` apart.
    level_uniforms_buffer: wgpu::Buffer,

    source_hash: u64,
    group0_bind_group: Option<wgpu::BindGroup>,
    group0_buffers_epoch: u64,
}

impl UserShaderPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group0"),
            entries: &[
                rw_storage(0),  // octree_nodes
                rw_storage(1),  // brick_pool
                rw_storage(2),  // leaf_attr_pool
                rw_storage(3),  // octree_alloc (per-region atomic array)
                rw_storage(4),  // brick_alloc  (per-region atomic array)
                rw_storage(5),  // leaf_attr_alloc (per-region atomic array)
                rw_storage(6),  // active_queue
                rw_storage(7),  // active_count
                rw_storage(8),  // fill_task_pool
                rw_storage(9),  // fill_task_alloc (per-region atomic array)
                rw_storage(10), // overflow counters
                ro_storage(11), // instance_overlay (per-instance paint)
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<RegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<LevelUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_geom pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
            ],
            immediate_size: 0,
        });
        let (classify_pipeline, fill_pipeline) = build_pipelines(device, &pipeline_layout, "");

        // Per-region atomic counters: array<atomic<u32>, MAX_REGIONS>.
        // Sized at MAX_REGIONS u32s. Reset per frame for dirty regions.
        let alloc_buf_size = (MAX_REGIONS as u64) * 4;
        let make_alloc_buf = |label| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: alloc_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let octree_alloc_buffer = make_alloc_buf("user_shader_geom octree_alloc");
        let brick_alloc_buffer = make_alloc_buf("user_shader_geom brick_alloc");
        let leaf_attr_alloc_buffer = make_alloc_buf("user_shader_geom leaf_attr_alloc");
        let fill_task_alloc_buffer = make_alloc_buf("user_shader_geom fill_task_alloc");

        let overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom overflow"),
            size: OVERFLOW_BUFFER_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let overflow_readback = OverflowReadback::new(device);

        let queue_size_bytes =
            (MAX_DEPTH + 1) as u64 * PER_LEVEL_QUEUE_CAP as u64 * std::mem::size_of::<ActiveCell>() as u64;
        let active_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_queue"),
            size: queue_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_count"),
            size: (MAX_DEPTH + 1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Persistent fill-task pool. ~48 MB at MAX_GLOBAL_FILL_TASKS.
        let fill_task_pool_size = MAX_GLOBAL_FILL_TASKS as u64 * 32;
        let fill_task_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom fill_task_pool"),
            size: fill_task_pool_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let regions_capacity: u64 =
            std::mem::size_of::<RegionUniform>() as u64 * MAX_REGIONS as u64;
        let regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom regions"),
            size: regions_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let level_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom level_uniforms"),
            size: LEVEL_UNIFORM_STRIDE * (MAX_DEPTH + 1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            pipeline_layout,
            classify_pipeline,
            fill_pipeline,
            octree_alloc_buffer,
            brick_alloc_buffer,
            leaf_attr_alloc_buffer,
            fill_task_alloc_buffer,
            active_queue_buffer,
            active_count_buffer,
            fill_task_pool_buffer,
            overflow_buffer,
            overflow_readback,
            regions_buffer,
            level_uniforms_buffer,
            source_hash: 0,
            group0_bind_group: None,
            group0_buffers_epoch: 0,
        }
    }

    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        user_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        let (classify, fill) = build_pipelines(device, &self.pipeline_layout, user_chunk);
        self.classify_pipeline = classify;
        self.fill_pipeline = fill;
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 { self.source_hash }

    /// Test accessor — exposes the leaf-attr-alloc counter buffer so a
    /// reference test can read back exactly how many occupied cells
    /// the BFS wrote without having to scan the brick pool. Not part
    /// of the production API; only used by `tests/user_shader_geom_bfs.rs`.
    #[doc(hidden)]
    pub fn test_leaf_attr_alloc_buffer(&self) -> &wgpu::Buffer {
        &self.leaf_attr_alloc_buffer
    }
    #[doc(hidden)]
    pub fn test_brick_alloc_buffer(&self) -> &wgpu::Buffer {
        &self.brick_alloc_buffer
    }
    #[doc(hidden)]
    pub fn test_fill_task_alloc_buffer(&self) -> &wgpu::Buffer {
        &self.fill_task_alloc_buffer
    }
    #[doc(hidden)]
    pub fn test_active_count_buffer(&self) -> &wgpu::Buffer {
        &self.active_count_buffer
    }

    pub fn ensure_group0(
        &mut self,
        device: &wgpu::Device,
        octree_nodes_buffer: &wgpu::Buffer,
        brick_pool_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        instance_overlay_buffer: &wgpu::Buffer,
        buffers_epoch: u64,
    ) {
        if self.group0_bind_group.is_some() && buffers_epoch == self.group0_buffers_epoch {
            return;
        }
        self.group0_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group0 bg"),
            layout: &self.group0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: octree_nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brick_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.octree_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.brick_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.leaf_attr_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.active_queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.active_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.fill_task_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.fill_task_alloc_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.overflow_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: instance_overlay_buffer.as_entire_binding() },
            ],
        }));
        self.group0_buffers_epoch = buffers_epoch;
    }

    /// Encode the BFS dispatch chain.
    ///
    /// Convention on `region_uniforms` ordering:
    ///   - `[0, topology_dirty_count)` — regions needing classify+fill.
    ///   - `[topology_dirty_count, fill_dirty_count)` — regions needing
    ///     fill only (cached topology, dirty fill).
    ///   - Indices `>= fill_dirty_count` should not be in the array;
    ///     the caller drops fully-clean (skipped) regions before
    ///     calling here.
    ///
    /// `region_uniforms[i].region_index` is implicit in array order.
    /// Topology-dirty regions get an initial active cell seeded into
    /// `active_queue[level=0]` and have their per-region atomic
    /// counters zeroed; fill-only regions only get their brick /
    /// leaf-attr counters reset (their `fill_task_extent` is reused
    /// from the cached bake).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        region_uniforms: &[RegionUniform],
        topology_dirty_count: u32,
        max_max_depth: u32,
        octree_nodes_buffer: &wgpu::Buffer,
        brick_pool_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        instance_overlay_buffer: &wgpu::Buffer,
        buffers_epoch: u64,
    ) {
        if region_uniforms.is_empty() {
            return;
        }
        debug_assert!(
            topology_dirty_count as usize <= region_uniforms.len(),
            "topology_dirty_count must be within region_uniforms",
        );
        let fill_dirty_count = region_uniforms.len() as u32;

        self.ensure_group0(
            device,
            octree_nodes_buffer,
            brick_pool_buffer,
            leaf_attr_pool_buffer,
            instance_overlay_buffer,
            buffers_epoch,
        );

        // ---- Per-region atomic counter resets ----
        //
        // Regions in [0, fill_dirty_count) all need fresh brick /
        // leaf-attr atomic counters so this frame's fill repopulates
        // their brick blocks at the same offsets the cached BFS
        // expected. Topology-dirty regions [0, topology_dirty_count)
        // additionally reset octree (seed=1, root sits at slot 0
        // within the block) and fill_task counters.
        //
        // We zero out positions [0, fill_dirty_count) for
        // brick/leaf_attr; positions [0, topology_dirty_count) for
        // fill_task; and seed octree[0..topology_dirty_count) = 1.
        // queue.write_buffer of small slices is essentially free.
        let zero_fill: Vec<u32> = vec![0u32; fill_dirty_count as usize];
        queue.write_buffer(&self.brick_alloc_buffer, 0, bytemuck::cast_slice(&zero_fill));
        queue.write_buffer(&self.leaf_attr_alloc_buffer, 0, bytemuck::cast_slice(&zero_fill));
        if topology_dirty_count > 0 {
            let topo = topology_dirty_count as usize;
            let zero_topo: Vec<u32> = vec![0u32; topo];
            queue.write_buffer(
                &self.fill_task_alloc_buffer, 0,
                bytemuck::cast_slice(&zero_topo),
            );
            let one_topo: Vec<u32> = vec![1u32; topo];
            queue.write_buffer(
                &self.octree_alloc_buffer, 0,
                bytemuck::cast_slice(&one_topo),
            );
        }

        // Reset overflow counters.
        let zero_overflow: Vec<u32> = vec![0u32; OVERFLOW_COUNTER_COUNT];
        queue.write_buffer(
            &self.overflow_buffer, 0,
            bytemuck::cast_slice(&zero_overflow),
        );

        // ---- Pre-fill sentinels for topology-dirty regions' fill_task extents ----
        //
        // Classify will overwrite `[0, count)` with valid tasks; the
        // suffix `[count, block_size)` must remain SENTINEL so
        // brick_fill workgroups can early-out instead of reading
        // garbage. CPU writes once before classify; for fill-only
        // regions the cached extent is already in this state.
        for ru in &region_uniforms[..topology_dirty_count as usize] {
            let block_bytes = ru.fill_task_block_size as u64 * 32;
            let offset_bytes = ru.fill_task_block_offset as u64 * 32;
            // BrickFillTask is 8 u32s (32 B); first u32 is octree_offset
            // which the fill kernel checks against the sentinel. We
            // initialize the entire block to sentinels; that overwrites
            // every field with 0xFE bytes but the only field the fill
            // kernel reads as a sentinel-check is octree_offset. The
            // rest gets re-written by classify for valid slots.
            let sentinel_data: Vec<u32> = vec![FILL_TASK_SENTINEL; (block_bytes / 4) as usize];
            queue.write_buffer(
                &self.fill_task_pool_buffer,
                offset_bytes,
                bytemuck::cast_slice(&sentinel_data),
            );
        }

        // ---- Region uniforms upload ----
        queue.write_buffer(
            &self.regions_buffer, 0,
            bytemuck::cast_slice(region_uniforms),
        );

        // ---- Initial active cells (topology-dirty regions only) ----
        let mut initial: Vec<ActiveCell> = Vec::with_capacity(topology_dirty_count as usize);
        for (i, ru) in region_uniforms[..topology_dirty_count as usize].iter().enumerate() {
            let center = [
                0.5 * (ru.aabb_min[0] + ru.aabb_max[0]),
                0.5 * (ru.aabb_min[1] + ru.aabb_max[1]),
                0.5 * (ru.aabb_min[2] + ru.aabb_max[2]),
            ];
            let half = 0.5 * (ru.aabb_max[0] - ru.aabb_min[0]);
            initial.push(ActiveCell {
                // Region's octree root sits at the start of its block.
                octree_offset: ru.octree_block_offset,
                region_index: i as u32,
                center,
                half_extent: half,
                _pad0: 0,
                _pad1: 0,
            });
        }
        if !initial.is_empty() {
            queue.write_buffer(
                &self.active_queue_buffer, 0,
                bytemuck::cast_slice(&initial),
            );
        }
        // active_count: [0] = topology_dirty_count, rest = 0.
        let mut init_active_count = vec![0u32; (MAX_DEPTH + 1) as usize];
        init_active_count[0] = topology_dirty_count;
        queue.write_buffer(
            &self.active_count_buffer, 0,
            bytemuck::cast_slice(&init_active_count),
        );

        // ---- Per-level uniform packs ----
        let mut level_packed = vec![0u8; LEVEL_UNIFORM_STRIDE as usize * (MAX_DEPTH + 1) as usize];
        for level in 0..=MAX_DEPTH {
            let lu = LevelUniform {
                current_level: level,
                per_level_cap: PER_LEVEL_QUEUE_CAP,
                max_active_per_level: PER_LEVEL_QUEUE_CAP,
                _pad: 0,
            };
            let off = level as usize * LEVEL_UNIFORM_STRIDE as usize;
            level_packed[off..off + std::mem::size_of::<LevelUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&lu));
        }
        queue.write_buffer(&self.level_uniforms_buffer, 0, &level_packed);

        let group0 = match &self.group0_bind_group {
            Some(bg) => bg,
            None => {
                eprintln!("[user_shader_pass] dispatch skipped — group 0 not bound");
                return;
            }
        };
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group1 bg"),
            layout: &self.group1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.regions_buffer.as_entire_binding(),
            }],
        });
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_geom group2 bg"),
            layout: &self.group2_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.level_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(
                        std::mem::size_of::<LevelUniform>() as u64,
                    ),
                }),
            }],
        });

        // ---- Classify dispatch chain (only if any topology-dirty) ----
        if topology_dirty_count > 0 {
            let workgroups_per_level = PER_LEVEL_QUEUE_CAP / CLASSIFY_WG_SIZE;
            let max_level = max_max_depth.min(MAX_DEPTH);
            for level in 0..=max_level {
                let dynamic_offset = (level as u64 * LEVEL_UNIFORM_STRIDE) as u32;
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("user_shader_geom classify"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.classify_pipeline);
                pass.set_bind_group(0, group0, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.set_bind_group(2, &group2, &[dynamic_offset]);
                pass.dispatch_workgroups(workgroups_per_level, 1, 1);
            }
        }

        // ---- Brick fill dispatch ----
        //
        // Workgroup grid: (max_block_size, fill_dirty_count, 1).
        //   wid.x = task index within region's fill_task block.
        //   wid.y = region index (regions array slot).
        // Workgroups past the region's actual emitted task count hit
        // `FILL_TASK_SENTINEL` in the task slot and early-out. Workgroups
        // past `region.fill_task_block_size` early-out via the cap check.
        let max_block_x = FILL_TASK_BUCKET_MAX;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("user_shader_geom brick_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fill_pipeline);
        pass.set_bind_group(0, group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.set_bind_group(2, &group2, &[0]);
        pass.dispatch_workgroups(max_block_x, fill_dirty_count, 1);
        drop(pass);

        // ---- Overflow readback copy ----
        if let Some(stage) = self.overflow_readback.next_write_buffer() {
            encoder.copy_buffer_to_buffer(
                &self.overflow_buffer, 0, stage, 0, OVERFLOW_BUFFER_BYTES,
            );
        }
    }

    /// Submit map_async on the slot we just wrote into (must run AFTER
    /// the encoder's commands have been queued.submit'd) and drain any
    /// already-mapped slots, logging non-zero counters. Idempotent
    /// across frames; safe to skip-call if no dispatch ran.
    pub fn submit_overflow_readback(&mut self) {
        self.overflow_readback.advance();
        self.overflow_readback.drain_and_log();
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

fn build_pipelines(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    user_chunk: &str,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let source = compose_geom_source(user_chunk);
    validate_wgsl(&source, "user_shader_geom");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_geom"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let classify = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom classify"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("classify_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let fill = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom fill"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("brick_fill_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (classify, fill)
}

/// Resolve a `shader_name` to the registry's `shader_id`. `0` =
/// identity / unregistered.
pub fn resolve_shader_id(infos: &[UserShaderInfo], name: &str) -> u32 {
    if name.is_empty() {
        return 0;
    }
    let mut sorted: Vec<&UserShaderInfo> = infos.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for (i, info) in sorted.iter().enumerate() {
        if info.name == name {
            return (i as u32) + 1;
        }
    }
    0
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
