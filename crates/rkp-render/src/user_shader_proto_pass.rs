//! Option B prototype bake pipeline.
//!
//! Owns the GPU pipeline + cache that materialises each user shader's
//! `proto_sample_at(uvw)` hook into a small dedicated octree+brick+leaf-attr
//! triple. The bake runs ONCE per shader (cached by source hash);
//! every instance the shader emits at march time reads back into the
//! same prototype voxels, regardless of the instance's transform.
//!
//! ## Lifecycle
//!
//! 1. `PrototypeCache::lookup_or_allocate(shader_id, source_hash, max_depth)`
//!    returns an entry. If `source_hash` matches the cached value, the
//!    entry is clean — bake is skipped. Otherwise the old extents are
//!    freed and fresh ones allocated; the entry is marked dirty.
//! 2. For each dirty entry, [`PrototypeBakePass`] uploads a
//!    [`PrototypeUniform`] and dispatches `proto_bake_main` with
//!    `(2^max_depth)³` workgroups. The compute shader writes leaf-level
//!    octree nodes + bricks + leaf-attrs.
//! 3. CPU pre-builds the prototype's INTERNAL octree levels (0..max_depth-1)
//!    at allocation time — they're a fixed dense structure determined
//!    entirely by `max_depth`. Stage 5's march descends through both
//!    pre-built and baked levels uniformly.
//!
//! ## Pool storage
//!
//! Prototypes live in the same `octree_nodes`, `brick_pool`,
//! `leaf_attr_pool` GPU buffers that the per-region cache and the main
//! march pass already bind, just at a disjoint byte range owned by
//! this cache's allocators. Consumers (the engine layer wiring this up
//! in Stage 5+) hand each cache its own base offsets so the ranges
//! don't overlap.

use std::collections::HashMap;

use crate::user_shader_pass::{
    BucketPoolAllocator, BRICK_BUCKET_MAX, BRICK_BUCKET_MIN, BRICK_CELLS,
    LEAF_ATTR_BUCKET_MAX, LEAF_ATTR_BUCKET_MIN, OCTREE_BUCKET_MAX, OCTREE_BUCKET_MIN,
};

/// Default prototype octree depth. With depth 2 the prototype is a
/// 16-cell-per-axis cube (4 bricks per axis, 64 max bricks) — enough
/// resolution for grass blades / pebbles / coarse foliage. Authors can
/// override per-shader via `@proto_max_depth`.
pub const DEFAULT_PROTO_MAX_DEPTH: u32 = 2;

/// Hard ceiling on prototype octree depth. Larger prototypes burn
/// per-shader pool space without a corresponding density gain — the
/// instance pipeline's win comes from MANY instances of a SMALL
/// prototype, not from per-prototype detail. Keeping this at 4 caps a
/// single prototype's brick reservation at 8³ = 512 max bricks, so
/// even a fully-solid prototype fits in the bucket allocator.
pub const MAX_PROTO_MAX_DEPTH: u32 = 4;

/// Total octree nodes in a fully-built dense tree at given depth.
/// Sum of geometric series 1 + 8 + 64 + ... + 8^depth = (8^(depth+1) - 1) / 7.
pub const fn octree_node_count_for_depth(max_depth: u32) -> u32 {
    let mut acc: u32 = 0;
    let mut level_size: u32 = 1;
    let mut k: u32 = 0;
    while k <= max_depth {
        acc += level_size;
        level_size *= 8;
        k += 1;
    }
    acc
}

/// Cached prototype state for one shader.
#[derive(Debug, Clone)]
pub struct PrototypeEntry {
    pub shader_id: u32,
    pub source_hash: u64,
    pub max_depth: u32,
    /// `(offset, size)` extents in each global pool. Offsets are
    /// RELATIVE to the cache's pool bases — add `pool_X_base` to get
    /// an absolute GPU index.
    pub octree_extent: (u32, u32),
    pub brick_extent: (u32, u32),
    pub leaf_attr_extent: (u32, u32),
    /// `true` after `begin_frame`; lookups touch the entry, so untouched
    /// entries are evicted at end of frame.
    pub touched_this_frame: bool,
}

impl PrototypeEntry {
    /// Absolute pool offset of the prototype's octree root (level 0).
    pub fn octree_root(&self, pool_octree_base: u32) -> u32 {
        pool_octree_base + self.octree_extent.0
    }

    /// Absolute pool offset of the prototype's leaf-level octree slots.
    /// The bake's workgroup_id (3D) Morton-encoded into a linear index
    /// lands at this offset.
    pub fn octree_leaf_offset(&self, pool_octree_base: u32) -> u32 {
        pool_octree_base
            + self.octree_extent.0
            + level_starts_inclusive(self.max_depth)[self.max_depth as usize]
    }
}

/// Returns `levels[k] = count of nodes at levels 0..k` for k in 0..=max_depth+1.
/// Length is `max_depth + 2`.
pub fn level_starts_inclusive(max_depth: u32) -> Vec<u32> {
    let n = max_depth as usize + 2;
    let mut v = Vec::with_capacity(n);
    let mut acc: u32 = 0;
    let mut level_size: u32 = 1;
    for _ in 0..=max_depth + 1 {
        v.push(acc);
        acc = acc.saturating_add(level_size);
        level_size = level_size.saturating_mul(8);
    }
    v
}

/// Conservative upper bound on bricks for a depth-`max_depth` prototype.
/// Equal to the leaf-level octree slot count = 8^max_depth.
pub fn max_bricks_for_depth(max_depth: u32) -> u32 {
    8u32.saturating_pow(max_depth)
}

/// Conservative upper bound on leaf-attr slots: every cell solid =
/// `BRICK_CELLS * max_bricks`.
pub fn max_leaf_attrs_for_depth(max_depth: u32) -> u32 {
    BRICK_CELLS.saturating_mul(max_bricks_for_depth(max_depth))
}

/// Cap on prototypes simultaneously cached. 256 is generous —
/// projects rarely have more than a few dozen instance shaders.
pub const MAX_PROTOTYPES: u32 = 256;

/// Default capacity of the prototype-only octree sub-pool.
/// `MAX_PROTOTYPES × octree_node_count_for_depth(MAX_PROTO_MAX_DEPTH)` =
/// 256 × 4681 ≈ 1.2 M nodes (~9 MB at 8 B/node). Modest reservation
/// from the global octree pool.
pub const PROTO_OCTREE_POOL_CAPACITY: u32 = 1_300_000;

/// Default capacity of the prototype-only brick sub-pool.
/// `MAX_PROTOTYPES × 8^MAX_PROTO_MAX_DEPTH` = 256 × 4096 ≈ 1 M bricks
/// (~256 MB). Bigger than expected real usage but fits comfortably
/// alongside the per-region cache's 768 MB reservation.
pub const PROTO_BRICK_POOL_CAPACITY: u32 = 1_048_576;

/// Default capacity of the prototype-only leaf-attr sub-pool.
/// `MAX_PROTOTYPES × BRICK_CELLS × 8^MAX_PROTO_MAX_DEPTH / 2` (half-occupancy)
/// = 256 × 64 × 4096 / 2 ≈ 33 M slots (~264 MB).
pub const PROTO_LEAF_ATTR_POOL_CAPACITY: u32 = 33_554_432;

/// Persistent prototype cache + variable-size pool allocator. Mirrors
/// `UserShaderObjectCache`'s shape but keyed by `shader_id` and with
/// no per-frame topology/fill split — prototypes only re-bake when
/// the shader source changes.
pub struct PrototypeCache {
    entries: HashMap<u32, PrototypeEntry>,
    octree_alloc: BucketPoolAllocator,
    brick_alloc: BucketPoolAllocator,
    leaf_attr_alloc: BucketPoolAllocator,
    pool_octree_base: u32,
    pool_brick_base: u32,
    pool_leaf_attr_base: u32,
    pool_octree_capacity: u32,
    pool_brick_capacity: u32,
    pool_leaf_attr_capacity: u32,
}

impl PrototypeCache {
    pub fn new() -> Self {
        Self::with_capacities(
            PROTO_OCTREE_POOL_CAPACITY,
            PROTO_BRICK_POOL_CAPACITY,
            PROTO_LEAF_ATTR_POOL_CAPACITY,
        )
    }

    pub fn with_capacities(
        octree_capacity: u32,
        brick_capacity: u32,
        leaf_attr_capacity: u32,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            octree_alloc: BucketPoolAllocator::new(
                octree_capacity, OCTREE_BUCKET_MIN, OCTREE_BUCKET_MAX,
            ),
            brick_alloc: BucketPoolAllocator::new(
                brick_capacity, BRICK_BUCKET_MIN, BRICK_BUCKET_MAX,
            ),
            leaf_attr_alloc: BucketPoolAllocator::new(
                leaf_attr_capacity, LEAF_ATTR_BUCKET_MIN, LEAF_ATTR_BUCKET_MAX,
            ),
            pool_octree_base: 0,
            pool_brick_base: 0,
            pool_leaf_attr_base: 0,
            pool_octree_capacity: octree_capacity,
            pool_brick_capacity: brick_capacity,
            pool_leaf_attr_capacity: leaf_attr_capacity,
        }
    }

    /// Configure the GPU offsets where the prototype-only sub-pool
    /// begins. Coordinated by the engine layer (Stage 5+) so the
    /// prototype range is disjoint from the per-region transient
    /// range.
    pub fn set_pool_bases(
        &mut self,
        pool_octree_base: u32,
        pool_brick_base: u32,
        pool_leaf_attr_base: u32,
    ) {
        if self.pool_octree_base == pool_octree_base
            && self.pool_brick_base == pool_brick_base
            && self.pool_leaf_attr_base == pool_leaf_attr_base
        {
            return;
        }
        self.flush();
        self.pool_octree_base = pool_octree_base;
        self.pool_brick_base = pool_brick_base;
        self.pool_leaf_attr_base = pool_leaf_attr_base;
    }

    pub fn pool_octree_base(&self) -> u32 { self.pool_octree_base }
    pub fn pool_brick_base(&self) -> u32 { self.pool_brick_base }
    pub fn pool_leaf_attr_base(&self) -> u32 { self.pool_leaf_attr_base }

    /// Drop every entry and reset all allocators.
    pub fn flush(&mut self) {
        self.entries.clear();
        self.octree_alloc = BucketPoolAllocator::new(
            self.pool_octree_capacity, OCTREE_BUCKET_MIN, OCTREE_BUCKET_MAX,
        );
        self.brick_alloc = BucketPoolAllocator::new(
            self.pool_brick_capacity, BRICK_BUCKET_MIN, BRICK_BUCKET_MAX,
        );
        self.leaf_attr_alloc = BucketPoolAllocator::new(
            self.pool_leaf_attr_capacity, LEAF_ATTR_BUCKET_MIN, LEAF_ATTR_BUCKET_MAX,
        );
    }

    /// Mark every entry untouched at the start of a frame.
    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.touched_this_frame = false;
        }
    }

    /// Look up `shader_id` against the cache. Returns `(slot, dirty)`:
    /// dirty=true means the bake compute must run for this entry.
    /// Returns `None` only when allocation fails (pool exhaustion); the
    /// caller should log overflow and proceed without the prototype.
    pub fn lookup_or_allocate(
        &mut self,
        shader_id: u32,
        source_hash: u64,
        max_depth: u32,
    ) -> Option<(PrototypeEntry, bool)> {
        debug_assert!(
            max_depth <= MAX_PROTO_MAX_DEPTH,
            "max_depth {max_depth} exceeds MAX_PROTO_MAX_DEPTH",
        );

        let estimate = PrototypePoolEstimate::for_depth(max_depth);

        if let Some(entry) = self.entries.get_mut(&shader_id) {
            let extents_fit = entry.octree_extent.1 >= estimate.octree
                && entry.brick_extent.1 >= estimate.bricks
                && entry.leaf_attr_extent.1 >= estimate.leaf_attrs;
            let depth_match = entry.max_depth == max_depth;
            if extents_fit && depth_match {
                let dirty = entry.source_hash != source_hash;
                if dirty {
                    entry.source_hash = source_hash;
                }
                entry.touched_this_frame = true;
                return Some((entry.clone(), dirty));
            }
            // Stale extents (depth changed or extents too small) — free
            // and fall through to fresh alloc.
            self.octree_alloc.free(entry.octree_extent.0, entry.octree_extent.1);
            self.brick_alloc.free(entry.brick_extent.0, entry.brick_extent.1);
            self.leaf_attr_alloc.free(entry.leaf_attr_extent.0, entry.leaf_attr_extent.1);
            self.entries.remove(&shader_id);
        }

        let octree_extent = self.octree_alloc.alloc(estimate.octree)?;
        let brick_extent = match self.brick_alloc.alloc(estimate.bricks) {
            Some(e) => e,
            None => {
                self.octree_alloc.free(octree_extent.0, octree_extent.1);
                return None;
            }
        };
        let leaf_attr_extent = match self.leaf_attr_alloc.alloc(estimate.leaf_attrs) {
            Some(e) => e,
            None => {
                self.octree_alloc.free(octree_extent.0, octree_extent.1);
                self.brick_alloc.free(brick_extent.0, brick_extent.1);
                return None;
            }
        };

        let entry = PrototypeEntry {
            shader_id,
            source_hash,
            max_depth,
            octree_extent,
            brick_extent,
            leaf_attr_extent,
            touched_this_frame: true,
        };
        self.entries.insert(shader_id, entry.clone());
        Some((entry, true))
    }

    /// Drop entries not referenced this frame and return their extents
    /// to the bucket allocators' free lists.
    pub fn evict_untouched(&mut self) {
        let to_remove: Vec<u32> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.touched_this_frame)
            .map(|(k, _)| *k)
            .collect();
        for k in to_remove {
            if let Some(entry) = self.entries.remove(&k) {
                self.octree_alloc.free(entry.octree_extent.0, entry.octree_extent.1);
                self.brick_alloc.free(entry.brick_extent.0, entry.brick_extent.1);
                self.leaf_attr_alloc.free(entry.leaf_attr_extent.0, entry.leaf_attr_extent.1);
            }
        }
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn get(&self, shader_id: u32) -> Option<&PrototypeEntry> {
        self.entries.get(&shader_id)
    }
    pub fn octree_high_water(&self) -> u32 { self.octree_alloc.high_water() }
    pub fn brick_high_water(&self) -> u32 { self.brick_alloc.high_water() }
    pub fn leaf_attr_high_water(&self) -> u32 { self.leaf_attr_alloc.high_water() }
}

impl Default for PrototypeCache {
    fn default() -> Self { Self::new() }
}

/// Pool size estimate for one prototype at a given depth. Always sized
/// to the worst case (every cell solid) — prototypes are tiny and
/// over-estimation is cheap.
#[derive(Debug, Clone, Copy)]
pub struct PrototypePoolEstimate {
    pub octree: u32,
    pub bricks: u32,
    pub leaf_attrs: u32,
}

impl PrototypePoolEstimate {
    pub fn for_depth(max_depth: u32) -> Self {
        Self {
            octree: octree_node_count_for_depth(max_depth)
                .clamp(OCTREE_BUCKET_MIN, OCTREE_BUCKET_MAX),
            bricks: max_bricks_for_depth(max_depth)
                .clamp(BRICK_BUCKET_MIN, BRICK_BUCKET_MAX),
            leaf_attrs: max_leaf_attrs_for_depth(max_depth)
                .clamp(LEAF_ATTR_BUCKET_MIN, LEAF_ATTR_BUCKET_MAX),
        }
    }
}

/// Pre-build the internal levels (0..max_depth-1) of a dense octree
/// rooted at byte offset `octree_block_offset` (relative to its pool).
/// Internal node values are absolute pool offsets when written into
/// `octree_nodes` because that's what the march reads directly.
///
/// Output layout — entries in source order:
///   * level 0: 1 node, value = pool_octree_base + octree_block_offset + level_starts[1]
///   * level 1: 8 nodes, each value = ...+ level_starts[2] + i * 8
///   * ...
///   * level max_depth-1: 8^(max_depth-1) nodes
///   * level max_depth: 8^max_depth nodes, all OCTREE_EMPTY (bake fills)
///
/// `pool_octree_base` is the absolute offset of byte 0 of the
/// prototype-only sub-pool; `octree_block_offset` is this prototype's
/// extent offset within that sub-pool. The two sum is the absolute
/// pool index of the prototype's root.
pub fn build_internal_levels(
    pool_octree_base: u32,
    octree_block_offset: u32,
    max_depth: u32,
) -> Vec<[u32; 2]> {
    let levels = level_starts_inclusive(max_depth);
    let total = levels[max_depth as usize + 1] as usize;
    let block_root = pool_octree_base + octree_block_offset;
    let mut out: Vec<[u32; 2]> = Vec::with_capacity(total);
    // Internal levels 0..max_depth-1: each node is a branch pointing to
    // 8 children at the next level.
    for k in 0..max_depth {
        let level_size = 8u32.saturating_pow(k);
        for i in 0..level_size {
            let first_child = block_root + levels[(k + 1) as usize] + i * 8;
            out.push([first_child, INTERNAL_ATTR_NONE]);
        }
    }
    // Leaf level: bake fills these in. Initialize to EMPTY.
    let leaf_level_size = 8u32.saturating_pow(max_depth);
    for _ in 0..leaf_level_size {
        out.push([OCTREE_EMPTY, INTERNAL_ATTR_NONE]);
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// Constants mirrored from `user_shader_proto.wgsl`. Kept in Rust so
/// the CPU pre-builder doesn't have to read the WGSL.
pub const OCTREE_EMPTY: u32 = 0xFFFFFFFFu32;
pub const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu32;

/// GPU prototype uniform — must match `PrototypeUniform` in
/// `user_shader_proto.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrototypeUniform {
    pub shader_id: u32,
    pub max_depth: u32,
    pub brick_block_offset: u32,
    pub brick_block_size: u32,
    pub leaf_attr_block_offset: u32,
    pub leaf_attr_block_size: u32,
    pub octree_leaf_offset: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<PrototypeUniform>() == 32);

impl PrototypeUniform {
    pub fn from_entry(entry: &PrototypeEntry, cache: &PrototypeCache) -> Self {
        Self {
            shader_id: entry.shader_id,
            max_depth: entry.max_depth,
            brick_block_offset: cache.pool_brick_base + entry.brick_extent.0,
            brick_block_size: entry.brick_extent.1,
            leaf_attr_block_offset: cache.pool_leaf_attr_base + entry.leaf_attr_extent.0,
            leaf_attr_block_size: entry.leaf_attr_extent.1,
            octree_leaf_offset: entry.octree_leaf_offset(cache.pool_octree_base),
            _pad: 0,
        }
    }
}

/// Cap on per-frame prototype atomic-counter slots. Each prototype
/// dispatch resets slot 0 of the counter buffer; the cap exists so the
/// buffer has fixed size for binding. V1 uses one dispatch per dirty
/// prototype (slot 0 only); batched multi-prototype dispatches in a
/// later phase would index 0..MAX_PROTO_DISPATCH_BATCH.
pub const MAX_PROTO_DISPATCH_BATCH: u32 = 16;

/// GPU pipeline owner for the prototype bake compute shader. Mirrors
/// the construction shape of [`crate::user_shader_pass::UserShaderPass`]
/// but is much smaller — prototype bakes don't need the BFS classify
/// step, the active queue, or per-region atomic counters.
pub struct PrototypeBakePass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub bake_pipeline: wgpu::ComputePipeline,
    /// Per-batch-slot atomic counter for `proto_brick_alloc`.
    /// `MAX_PROTO_DISPATCH_BATCH * 4` bytes.
    pub proto_brick_alloc_buffer: wgpu::Buffer,
    /// Per-batch-slot atomic counter for `proto_leaf_attr_alloc`.
    pub proto_leaf_attr_alloc_buffer: wgpu::Buffer,
    /// Overflow counters — same layout the per-region pass uses
    /// (only `OVERFLOW_BRICK` and `OVERFLOW_LEAF_ATTR` are written by
    /// this shader). The proto pass owns its own buffer rather than
    /// sharing with the per-region one because resets and readbacks
    /// are scheduled independently.
    pub overflow_buffer: wgpu::Buffer,
    /// Uniform buffer for `PrototypeUniform`. Bound at group(1).
    pub proto_uniform_buffer: wgpu::Buffer,
    /// Hash of the user-shader source mix the bake pipeline was
    /// last built against. Comparing against the registry's
    /// `source_hash` decides whether the pipeline needs rebuilding.
    pub source_hash: u64,
}

impl PrototypeBakePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_proto group0"),
            entries: &[
                rw_storage(0), // octree_nodes
                rw_storage(1), // brick_pool
                rw_storage(2), // leaf_attr_pool
                rw_storage(3), // proto_brick_alloc
                rw_storage(4), // proto_leaf_attr_alloc
                rw_storage(5), // overflow
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_proto group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<PrototypeUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_proto pipeline layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let bake_pipeline = build_proto_pipeline(device, &pipeline_layout, "");

        let alloc_size = (MAX_PROTO_DISPATCH_BATCH as u64) * 4;
        let make_alloc = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: alloc_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let proto_brick_alloc_buffer = make_alloc("user_shader_proto brick_alloc");
        let proto_leaf_attr_alloc_buffer = make_alloc("user_shader_proto leaf_attr_alloc");

        // Overflow buffer — must be at least as large as the highest
        // index the WGSL writes (OVERFLOW_LEAF_ATTR = 2). Match the
        // per-region pass's OVERFLOW_COUNTER_COUNT (=12 at MAX_DEPTH=8)
        // so a future shared-buffer setup is trivial.
        let overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_proto overflow"),
            size: 12 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let proto_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_proto uniform"),
            size: std::mem::size_of::<PrototypeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            pipeline_layout,
            bake_pipeline,
            proto_brick_alloc_buffer,
            proto_leaf_attr_alloc_buffer,
            overflow_buffer,
            proto_uniform_buffer,
            source_hash: 0,
        }
    }

    /// Re-build the compute pipeline against a fresh user-shader chunk.
    /// Returns `true` if rebuilt, `false` if the hash matched and the
    /// existing pipeline was kept. Mirrors
    /// `UserShaderPass::reload_user_shaders`.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        proto_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.bake_pipeline = build_proto_pipeline(device, &self.pipeline_layout, proto_chunk);
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
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

fn build_proto_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    proto_chunk: &str,
) -> wgpu::ComputePipeline {
    let source = compose_proto_source(proto_chunk);
    crate::validate_wgsl(&source, "user_shader_proto");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_proto"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_proto bake"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("proto_bake_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Splice the composer's `proto` chunk into the bake shader source.
/// Empty chunk returns the in-tree default (which has its own
/// identity stub between the markers); non-empty chunk REPLACES the
/// stub and the markers themselves with `proto_chunk`. Mirrors
/// `compose_geom_source` in `user_shader_pass.rs`.
pub fn compose_proto_source(proto_chunk: &str) -> String {
    let template = include_str!("shaders/user_shader_proto.wgsl");
    if proto_chunk.is_empty() {
        return template.to_string();
    }
    // Marker strings constructed via concat so the `find` below isn't
    // fooled by literal occurrences in this docstring or elsewhere in
    // the Rust source.
    let begin_marker = concat!("USER_PROTO_DISPATCH", "_BEGIN");
    let end_marker = concat!("USER_PROTO_DISPATCH", "_END");
    let begin = template
        .find(begin_marker)
        .expect("user_shader_proto.wgsl missing USER_PROTO_DISPATCH_BEGIN marker");
    let end_after = template[begin..]
        .find(end_marker)
        .map(|off| begin + off + end_marker.len())
        .expect("user_shader_proto.wgsl missing USER_PROTO_DISPATCH_END marker");
    let mut out = String::with_capacity(template.len() + proto_chunk.len());
    out.push_str(&template[..begin]);
    out.push_str(proto_chunk);
    out.push_str(&template[end_after..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_wgsl_valid(source: &str, label: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("[{label}] parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
    }

    #[test]
    fn octree_node_count_matches_geometric_series() {
        // Sum 1 + 8 + 64 + ... + 8^d = (8^(d+1) - 1) / 7
        assert_eq!(octree_node_count_for_depth(0), 1);
        assert_eq!(octree_node_count_for_depth(1), 9);
        assert_eq!(octree_node_count_for_depth(2), 73);
        assert_eq!(octree_node_count_for_depth(3), 585);
        assert_eq!(octree_node_count_for_depth(4), 4681);
    }

    #[test]
    fn level_starts_are_cumulative_sizes() {
        // For max_depth=2: [0, 1, 9, 73, 585]
        // (level 0 starts at 0, level 1 at 1, level 2 at 9, leaf-level
        // ends at 73 which is the total).
        let lv = level_starts_inclusive(2);
        assert_eq!(lv, vec![0, 1, 9, 73]);
        let lv = level_starts_inclusive(3);
        assert_eq!(lv, vec![0, 1, 9, 73, 585]);
    }

    #[test]
    fn max_bricks_and_leaf_attrs_at_depth() {
        // 8^max_depth bricks, 64 cells each.
        assert_eq!(max_bricks_for_depth(0), 1);
        assert_eq!(max_bricks_for_depth(2), 64);
        assert_eq!(max_bricks_for_depth(4), 4096);
        assert_eq!(max_leaf_attrs_for_depth(2), 64 * 64);
        assert_eq!(max_leaf_attrs_for_depth(4), 64 * 4096);
    }

    #[test]
    fn pool_estimate_clamps_to_bucket_range() {
        // depth 0 → tiny; estimate must clamp up to OCTREE_BUCKET_MIN
        let e = PrototypePoolEstimate::for_depth(0);
        assert!(e.octree >= OCTREE_BUCKET_MIN);
        assert!(e.bricks >= BRICK_BUCKET_MIN);
        assert!(e.leaf_attrs >= LEAF_ATTR_BUCKET_MIN);
        // depth 4 → larger; should fit comfortably in bucket maxes
        let e = PrototypePoolEstimate::for_depth(4);
        assert!(e.octree <= OCTREE_BUCKET_MAX);
        assert!(e.bricks <= BRICK_BUCKET_MAX);
        assert!(e.leaf_attrs <= LEAF_ATTR_BUCKET_MAX);
    }

    #[test]
    fn build_internal_levels_layout_for_depth_2() {
        // pool_octree_base = 1000, block_offset = 50.
        // Block root = 1050.
        // levels for depth 2: [0, 1, 9, 73].
        // Total nodes: 73.
        // Level 0 (1 node at slot 0): value = 1050 + 1 = 1051
        // Level 1 (8 nodes at slots 1..9): values = 1050 + 9 + i*8 for i in 0..8
        //   → 1059, 1067, 1075, 1083, 1091, 1099, 1107, 1115
        // Level 2 (64 nodes at slots 9..73): all OCTREE_EMPTY
        let nodes = build_internal_levels(1000, 50, 2);
        assert_eq!(nodes.len(), 73);
        assert_eq!(nodes[0], [1051, INTERNAL_ATTR_NONE]);
        for i in 0..8u32 {
            assert_eq!(
                nodes[1 + i as usize],
                [1050 + 9 + i * 8, INTERNAL_ATTR_NONE],
                "level-1 node {i} mismatch",
            );
        }
        for (idx, node) in nodes.iter().enumerate().skip(9) {
            assert_eq!(
                *node,
                [OCTREE_EMPTY, INTERNAL_ATTR_NONE],
                "leaf-level slot {idx} should start empty",
            );
        }
    }

    #[test]
    fn build_internal_levels_root_only_for_depth_0() {
        let nodes = build_internal_levels(0, 0, 0);
        // depth 0: only the leaf level exists, 1 node, EMPTY.
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], [OCTREE_EMPTY, INTERNAL_ATTR_NONE]);
    }

    #[test]
    fn cache_first_lookup_is_dirty() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let (entry, dirty) = cache.lookup_or_allocate(1, 0xDEAD_BEEFu64, 2).unwrap();
        assert!(dirty);
        assert_eq!(entry.shader_id, 1);
        assert_eq!(entry.source_hash, 0xDEAD_BEEFu64);
        assert_eq!(entry.max_depth, 2);
    }

    #[test]
    fn cache_repeat_lookup_with_same_hash_is_clean() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xDEAD, 2).unwrap();
        let (_, dirty) = cache.lookup_or_allocate(1, 0xDEAD, 2).unwrap();
        assert!(!dirty);
    }

    #[test]
    fn cache_source_change_re_dirties_without_re_allocating() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        let oct_hw_after_first = cache.octree_high_water();
        let (e2, dirty) = cache.lookup_or_allocate(1, 0xBBBB, 2).unwrap();
        assert!(dirty);
        // Same extents → same offsets. Re-bake reuses the slot.
        assert_eq!(e1.octree_extent, e2.octree_extent);
        assert_eq!(e1.brick_extent, e2.brick_extent);
        assert_eq!(e1.leaf_attr_extent, e2.leaf_attr_extent);
        assert_eq!(cache.octree_high_water(), oct_hw_after_first);
    }

    #[test]
    fn cache_distinct_shader_ids_get_distinct_extents() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        let (e2, _) = cache.lookup_or_allocate(2, 0xBBBB, 2).unwrap();
        assert_ne!(e1.octree_extent.0, e2.octree_extent.0);
        assert_ne!(e1.brick_extent.0, e2.brick_extent.0);
    }

    #[test]
    fn cache_evicts_untouched_entries() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        let _ = cache.lookup_or_allocate(2, 0xBBBB, 2).unwrap();
        assert_eq!(cache.entry_count(), 2);
        cache.begin_frame();
        // Touch only shader 1 this frame.
        let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        cache.evict_untouched();
        assert_eq!(cache.entry_count(), 1);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
    }

    #[test]
    fn cache_depth_change_reallocs_extents() {
        // Depth 4's leaf-attr estimate clamps to LEAF_ATTR_BUCKET_MAX
        // (131 072), so the test pool needs at least that much.
        let mut cache = PrototypeCache::with_capacities(20_000, 8192, 200_000);
        cache.set_pool_bases(0, 0, 0);
        let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        let (e2, dirty) = cache.lookup_or_allocate(1, 0xAAAA, 4).unwrap();
        assert!(dirty);
        // Stale extents freed, new (likely larger) extents allocated.
        assert_eq!(e2.max_depth, 4);
        // depth 4 needs more bricks than depth 2 — extent size grows.
        assert!(e2.brick_extent.1 >= e1.brick_extent.1);
    }

    #[test]
    fn cache_pool_base_change_flushes() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        cache.set_pool_bases(100, 0, 0);
        // Flush dropped the entry.
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        // Two prototypes at depth 2 require: octree bucket = 128 each
        // (depth-2 estimate is 73 rounded up to OCTREE_BUCKET_MIN=64
        // → next is 128), brick bucket = 64, leaf-attr bucket = 4096.
        // Sized for exactly one — second fails.
        let mut cache = PrototypeCache::with_capacities(128, 64, 4096);
        cache.set_pool_bases(0, 0, 0);
        let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
        // Second prototype at depth 2 won't fit — octree pool out of room
        // (one prototype already consumed bucket-rounded extent).
        assert!(cache.lookup_or_allocate(2, 0xBBBB, 2).is_none());
    }

    #[test]
    fn proto_uniform_size_is_32() {
        assert_eq!(std::mem::size_of::<PrototypeUniform>(), 32);
    }

    #[test]
    fn proto_uniform_offsets_match_entry_extents() {
        let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
        cache.set_pool_bases(1000, 2000, 3000);
        let (entry, _) = cache.lookup_or_allocate(7, 0xCAFE, 2).unwrap();
        let u = PrototypeUniform::from_entry(&entry, &cache);
        assert_eq!(u.shader_id, 7);
        assert_eq!(u.max_depth, 2);
        assert_eq!(u.brick_block_offset, 2000 + entry.brick_extent.0);
        assert_eq!(u.brick_block_size, entry.brick_extent.1);
        assert_eq!(u.leaf_attr_block_offset, 3000 + entry.leaf_attr_extent.0);
        assert_eq!(u.leaf_attr_block_size, entry.leaf_attr_extent.1);
        // octree_leaf_offset = pool_octree_base + extent.0 + level_starts[max_depth]
        let level_starts = level_starts_inclusive(2);
        let expected = 1000 + entry.octree_extent.0 + level_starts[2];
        assert_eq!(u.octree_leaf_offset, expected);
    }

    #[test]
    fn proto_shader_validates_with_empty_chunk() {
        // Empty proto chunk should still produce valid WGSL — the
        // identity stub `dispatch_user_proto` is the default.
        let source = compose_proto_source("");
        assert_wgsl_valid(&source, "user_shader_proto");
        assert!(source.contains("proto_bake_main"));
    }

    #[test]
    fn proto_shader_validates_with_nonempty_chunk() {
        // Splice in a minimal user dispatch chunk and confirm the
        // composed source is valid WGSL. The chunk has to provide its
        // own dispatch_user_proto definition (the splice removes the
        // identity stub between the markers).
        let chunk = r#"
fn rkp_user_1_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 1u;
    v.normal = vec3<f32>(0.0, 1.0, 0.0);
    return v;
}
fn dispatch_user_proto(shader_id: u32, uvw: vec3<f32>) -> VoxelEmit {
    switch shader_id {
        case 1u: { return rkp_user_1_proto(uvw); }
        default: { return voxel_emit_skip(); }
    }
}
"#;
        let source = compose_proto_source(chunk);
        assert_wgsl_valid(&source, "user_shader_proto.spliced");
        assert!(source.contains("rkp_user_1_proto"));
    }
}
