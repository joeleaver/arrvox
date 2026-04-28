//! Sparse BFS GPU runtime geometry from user shaders, global-pool variant.
//!
//! Owns the GPU pipelines that build transient octrees by atomically
//! allocating nodes / bricks / leaf-attrs from a SINGLE global pool
//! shared across all regions in the frame. Memory and compute scale
//! with painted surface area rather than the (4·2^depth)³ cube the
//! original dense-brick model demanded.
//!
//! ## Pool layout
//!
//! All regions in a frame draw from the same three flat tails:
//!   - `octree_nodes` — `MAX_GLOBAL_OCTREE_NODES` slots
//!   - `brick_pool`   — `MAX_GLOBAL_BRICKS` bricks
//!   - `leaf_attr_pool` — `MAX_GLOBAL_LEAF_ATTRS` slots
//!
//! These tails live in the scene's flat pools past the CPU-managed
//! head (same buffers the march reads). `RkpScene::ensure_user_shader_capacity`
//! grows the buffers once at startup and they stay stable.
//!
//! Three GLOBAL atomic counters (`octree_alloc`, `brick_alloc`,
//! `leaf_attr_alloc`) — single u32 each — bump-allocate within those
//! ranges as the BFS expands. Overflow degrades to `OCTREE_EMPTY` at
//! the offending node and increments a per-pool counter in the
//! `overflow` buffer; CPU reads that buffer asynchronously and logs
//! when caps are hit.
//!
//! ## No persistent cache
//!
//! Every region rebakes from scratch every frame. There's no
//! cross-frame state to preserve. Eliminating the cache (vs. V10's
//! per-tile slabs) removes ~50× over-reservation: 400 fully-painted
//! grass tiles drop from ~1.7 GB transient pool to ~50 MB actually-used
//! while keeping the same ~2 ms BFS bake cost.
//!
//! ## Dispatch chain (per frame)
//!
//! 1. Counters reset: `octree_alloc = region_count` (one root per
//!    region, sequentially placed at the head of the transient
//!    octree slice), `brick_alloc = 0`, `leaf_attr_alloc = 0`,
//!    `fill_count = 0`, `active_count[*] = 0`, `overflow[*] = 0`.
//! 2. Active queue seeded with one root cell per region into
//!    `active_queue[level=0]`, `active_count[0] = region_count`.
//!    Each cell's `octree_offset = pool_octree_base + region_index`.
//! 3. For L in 0..=max(max_depth across regions): one
//!    `classify_main` dispatch with `level_u.current_level = L`. Threads
//!    past `active_count[L]` early-out, so we always issue
//!    `(per_level_cap / 64)` workgroups regardless of true active count
//!    (saves a build-indirect-args dispatch).
//! 4. One `brick_fill_main` dispatch over the surviving fill_queue.
//!
//! All dispatches share group-0 (scene + global counters + overflow
//! buffer) and group-1 (region storage array). Group-2 holds the
//! per-dispatch level uniform — which now also carries the global
//! pool bases and caps — at a dynamic offset.
//!
//! ## Compose contract
//!
//! Unchanged — `compose_geom_source` splices the composer's
//! `generate` chunk between `// USER_GENERATE_DISPATCH_BEGIN/_END`
//! markers in `user_shader_geom.wgsl`. User shaders that called
//! `host_sample_at(world_pos)` keep working unchanged.

use std::collections::HashMap;

use crate::rkp_gpu_object::{geom_type, RkpGpuObject};
use crate::shader_composer::UserShaderInfo;
use crate::validate_wgsl;

// ============================================================
// Variable-size bucketed pool allocator
// ============================================================

/// Free-list allocator over a fixed-capacity pool, bucketed by power
/// of 2. Per-pool bucket range is configurable; `alloc(n)` rounds the
/// request up to the smallest bucket that fits and returns
/// `(offset, allocated_size)`. `free(offset, allocated_size)` pushes
/// the extent back onto its bucket's free list. Bumps a global
/// high-water mark when the matching free list is empty.
///
/// Internal waste is at most 2× per region (worst case: request just
/// over a bucket boundary, get the next bucket). Total reservation
/// matches actual aggregate usage instead of `regions × max_per_region`.
///
/// Not thread-safe — all calls are CPU-side from the render thread.
#[derive(Debug, Clone)]
pub struct BucketPoolAllocator {
    capacity: u32,
    min_bucket: u32,
    max_bucket: u32,
    high_water: u32,
    /// `free_lists[i]` holds free-extent offsets for bucket size
    /// `min_bucket << i`. Length is `log2(max_bucket / min_bucket) + 1`.
    free_lists: Vec<Vec<u32>>,
}

impl BucketPoolAllocator {
    /// Build an allocator over `capacity` slots with bucket sizes
    /// `[min_bucket, 2*min_bucket, …, max_bucket]`. Both bounds must
    /// be powers of 2 and `min_bucket <= max_bucket <= capacity`.
    pub fn new(capacity: u32, min_bucket: u32, max_bucket: u32) -> Self {
        assert!(min_bucket.is_power_of_two(), "min_bucket must be power of 2");
        assert!(max_bucket.is_power_of_two(), "max_bucket must be power of 2");
        assert!(min_bucket <= max_bucket, "min_bucket must be <= max_bucket");
        let n_buckets =
            (max_bucket.trailing_zeros() - min_bucket.trailing_zeros() + 1) as usize;
        Self {
            capacity,
            min_bucket,
            max_bucket,
            high_water: 0,
            free_lists: vec![Vec::new(); n_buckets],
        }
    }

    /// Smallest bucket size at least `requested`, clamped to
    /// `[min_bucket, max_bucket]`.
    fn bucket_for(&self, requested: u32) -> u32 {
        requested
            .max(1)
            .next_power_of_two()
            .max(self.min_bucket)
            .min(self.max_bucket)
    }

    fn bucket_idx(&self, bucket: u32) -> usize {
        (bucket.trailing_zeros() - self.min_bucket.trailing_zeros()) as usize
    }

    /// Allocate at least `requested` slots. Returns `(offset, allocated_size)`
    /// where `allocated_size >= requested` is the bucket size, or
    /// `None` if the request exceeds `max_bucket` or the pool is
    /// exhausted (no matching free extent + no room to bump).
    pub fn alloc(&mut self, requested: u32) -> Option<(u32, u32)> {
        if requested > self.max_bucket {
            return None;
        }
        let bucket = self.bucket_for(requested);
        let idx = self.bucket_idx(bucket);
        if let Some(offset) = self.free_lists[idx].pop() {
            return Some((offset, bucket));
        }
        if self.high_water + bucket > self.capacity {
            return None;
        }
        let offset = self.high_water;
        self.high_water += bucket;
        Some((offset, bucket))
    }

    /// Return an extent to its bucket's free list. `allocated_size`
    /// must match the value returned by the corresponding `alloc`.
    pub fn free(&mut self, offset: u32, allocated_size: u32) {
        debug_assert!(
            allocated_size.is_power_of_two()
                && allocated_size >= self.min_bucket
                && allocated_size <= self.max_bucket,
            "free: allocated_size must come from a previous alloc()",
        );
        let idx = self.bucket_idx(allocated_size);
        self.free_lists[idx].push(offset);
    }

    pub fn high_water(&self) -> u32 { self.high_water }
    pub fn capacity(&self) -> u32 { self.capacity }
    pub fn min_bucket(&self) -> u32 { self.min_bucket }
    pub fn max_bucket(&self) -> u32 { self.max_bucket }

    /// Number of free extents currently held across all buckets —
    /// for diagnostics and tests.
    pub fn free_count(&self) -> usize {
        self.free_lists.iter().map(|l| l.len()).sum()
    }
}

/// One materialization request from sim → render. Stable across frames
/// for cache hit; rebuilt by sim each tick from the ECS scan.
#[derive(Debug, Clone)]
pub struct ShaderRegionRequest {
    /// Stable identifier — typically the host entity's scene id or a
    /// synthetic id for free-standing regions. Used as the cache key
    /// alongside `material_id`.
    pub host_object_id: u32,
    /// The host's leaf-level material that triggered this region. Used
    /// for cache keying so the same host with two shader-using
    /// materials gets two cache entries.
    pub material_id: u32,
    /// Shader name (file stem). Resolved against the registry to a
    /// `shader_id` at dispatch time. Empty / unregistered names skip
    /// the request.
    pub shader_name: String,
    /// Per-material shader params, packed in the shader's declared
    /// order. Length matches the shader's `params` schema; longer is
    /// truncated, shorter is zero-padded. The first 8 entries land in
    /// the GPU param array.
    pub params: Vec<f32>,
    /// World-space AABB the user's `generate` hook is sampled across.
    /// Must be a cube — the BFS subdivides isotropically.
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    /// Voxel size at the deepest level. Sim derives this from the
    /// shader's `@cell_size` directive (clamped against the cube's
    /// extent so the implied depth fits within `max_depth`).
    pub cell_size: f32,
    /// Folded with shader source hash + host geometry epoch into the
    /// cache key. Bumped by sim whenever any input the cache should
    /// invalidate on changes.
    pub input_hash: u64,
    /// `@animated` — regenerate every frame, ignoring the hash.
    pub animated: bool,
    /// `@region_thickness` — Lipschitz band around host surface within
    /// which the classifier keeps cells live. 0 disables the gate.
    pub region_thickness: f32,
    /// Octree depth — derived sim-side as
    /// `ceil(log2(extent / (cell_size * BRICK_DIM)))` and clamped to
    /// the shader's `@max_depth` cap (default 8).
    pub max_depth: u32,
    /// Painted-leaf count from the host scan that produced this region.
    /// Drives per-region pool sizing — more painted leaves means a
    /// larger surface area and more sparse-octree expansion. 0 falls
    /// back to a small floor so test/free-standing regions still get a
    /// usable reservation.
    pub painted_leaf_count: u32,
    /// V10 tile coordinate. For shaders with `@tile_size`, this is
    /// the host-local tile index `floor(painted_leaf_pos / tile_size)`.
    /// For shaders without tiling, set to `NO_TILE` (sentinel).
    /// Folded into the cache key so two tiles on the same
    /// (object, material) get distinct cache entries + pool slices.
    pub tile_index: [i32; 3],
    /// Host octree info for `host_sample_at(world_pos)` queries from
    /// inside the user shader. `host_octree_root == 0xFFFFFFFF` means
    /// "no host" (region is free-standing); `host_sample_at` returns
    /// `(+inf, +Y)` in that case.
    pub host_octree_root: u32,
    pub host_octree_depth: u32,
    pub host_octree_extent: f32,
    pub host_grid_origin: [f32; 3],
    pub host_inverse_world: [[f32; 4]; 4],
}

/// Sentinel "no host" value matching `HOST_NO_HOST_SENTINEL` in WGSL.
pub const HOST_NO_HOST_SENTINEL: u32 = 0xFFFF_FFFFu32;

/// Sentinel `tile_index` value used for non-tiled shaders (those
/// without an `@tile_size` directive). One cache entry per
/// (object, material) pair, V9 behaviour.
pub const NO_TILE: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Cells per brick — must match `rkp_core::brick_pool::BRICK_CELLS`.
pub const BRICK_CELLS: u32 = 64;

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
/// 176 B per uniform × 1024 = 180 KB — trivial.
pub const MAX_REGIONS: u32 = 1024;

/// Global brick pool capacity for user-shader transient geometry.
/// 3 M bricks × 64 cells × 4 B = 768 MB. Combined with typical scene
/// CPU brick pool (~250 MB) this stays under wgpu's 1 GB binding
/// limit with headroom for moderate CPU growth. Bumping past this
/// (without a separate user-shader brick buffer) would push the
/// brick_pool binding past the limit and trigger silent clamping.
pub const MAX_GLOBAL_BRICKS: u32 = 3_000_000;

/// Global leaf-attr pool capacity. Sized at `MAX_GLOBAL_BRICKS ×
/// BRICK_CELLS / 2` — assumes ~half of each brick's cells are surface
/// (occupied) on average. At 112 M slots × 8 B = 896 MB.
pub const MAX_GLOBAL_LEAF_ATTRS: u32 = MAX_GLOBAL_BRICKS * (BRICK_CELLS / 2);

/// Global octree node capacity. 50 M slots × 8 B = 400 MB. Sized to
/// fit ~400 typical regions at OCTREE_BUCKET_MAX = 131 072.
pub const MAX_GLOBAL_OCTREE_NODES: u32 = 50_000_000;

/// Persistent fill-task pool capacity. Each `BrickFillTask` is 32 B.
/// At depth 5, each 1 m tile produces up to ~32 K fill tasks worst
/// case (geometry-driven, NOT paint-driven). 16 M tasks × 32 B =
/// 512 MB. Fits ~500 worst-case tiles per frame, more for typical
/// density.
pub const MAX_GLOBAL_FILL_TASKS: u32 = 16_000_000;

// ---------- Bucket sizes per pool ---------- //
//
// Allocator buckets are powers of 2 from MIN to MAX inclusive, in the
// pool's native unit (octree nodes / bricks / leaf-attrs / fill tasks).
//
// Sized to fit a fully-populated tile at depth 5 + region_thickness
// 0.5 m + 1 m extent without overflow:
//   - Brick-parent cells per tile: (extent / cell_size / 4)^3 ≈ 32^3 =
//     32 768. With proximity gate keeping ~half, ~16 K fill tasks.
//   - Of those, V12 deferred allocates a brick only where cells emit;
//     for grass-density paint, ~half allocate → ~8 K bricks per tile.
//   - Octree allocations across all levels for a fully-MIXED tile at
//     depth 5: 8 + 64 + 512 + 4096 + 32 768 ≈ 37 K. With proximity
//     gating, ~20 K typical.
//   - Leaf-attrs: each brick has up to BRICK_CELLS = 64 cells with
//     attrs; for grass, average ~10-30 emissions per brick.

/// Brick pool buckets — region claims a contiguous block of this many
/// bricks. {16, 32, 64, …, 16384}. Sized so 3 M / 16384 = 183
/// max-bucket regions fit globally. Larger per-region buckets would
/// pack fewer regions into the global brick pool and cause whole
/// tiles to silently drop when paint coverage grows.
pub const BRICK_BUCKET_MIN: u32 = 16;
pub const BRICK_BUCKET_MAX: u32 = 16384;

/// Octree pool buckets — fully-MIXED tile at depth 5 uses ~37 K
/// nodes. Bucket up to 131 072 leaves headroom for whatever the
/// classifier actually produces under high paint density.
pub const OCTREE_BUCKET_MIN: u32 = 64;
pub const OCTREE_BUCKET_MAX: u32 = 131072;

/// Leaf-attr pool buckets. Smallest bucket uses `BRICK_BUCKET_MIN ×
/// BRICK_CELLS / 2`. Max is capped at 131 072 = enough for 4 K bricks
/// at full occupancy or 8 K at half occupancy — covers typical grass
/// densities. Above this, regions overflow leaf-attrs gracefully (the
/// `OVERFLOW_LEAF_ATTR` counter logs the event; the brick still
/// renders with `BRICK_CELL_EMPTY` for the overflowing cells).
pub const LEAF_ATTR_BUCKET_MIN: u32 = BRICK_BUCKET_MIN * BRICK_CELLS / 2;       // 512
pub const LEAF_ATTR_BUCKET_MAX: u32 = 131_072;

/// Fill-task pool buckets — one task per brick-parent cell. Sized to
/// hold a fully-populated 1 m tile at depth 5 (~16 K), with bucket up
/// to 32 768 to absorb above-average density.
pub const FILL_TASK_BUCKET_MIN: u32 = BRICK_BUCKET_MIN;
pub const FILL_TASK_BUCKET_MAX: u32 = 32768;

/// Number of overflow counter slots in the GPU `overflow` buffer.
/// Layout (must match `OVERFLOW_*` constants in user_shader_geom.wgsl):
///   [0]    = octree pool overflow
///   [1]    = brick pool overflow
///   [2]    = leaf-attr pool overflow
///   [3]    = fill queue overflow
///   [4..4+MAX_DEPTH+1] = active-queue overflow per level
const OVERFLOW_COUNTER_COUNT: usize = 4 + (MAX_DEPTH as usize + 1);
const OVERFLOW_BUFFER_BYTES: u64 = OVERFLOW_COUNTER_COUNT as u64 * 4;

/// Number of frames we keep readback staging buffers in flight for the
/// overflow counters. 3 frames matches the typical wgpu queue depth so
/// readbacks don't stall the GPU; at any given moment we have one
/// "current" staging buffer being copied into and two pending
/// `map_async` results.
const OVERFLOW_READBACK_FRAMES: usize = 3;

/// Workgroup size for `classify_main`. Determines how many workgroups
/// we dispatch per level: `PER_LEVEL_QUEUE_CAP / CLASSIFY_WG_SIZE`
/// regardless of real active count (per-thread early-out).
const CLASSIFY_WG_SIZE: u32 = 64;

/// Persistent per-region cache entry. Survives across frames. Each
/// entry holds variable-size extents in the four global pools
/// (octree, brick, leaf-attr, fill-task) plus the two hashes used to
/// decide whether the region's GPU contents are still valid.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Global pool offset of this region's octree root. Same as
    /// `octree_block_offset` (sequential allocation puts the root
    /// at the start of the block).
    octree_root: u32,
    /// `(offset, size)` extents in each global pool, in pool-native
    /// units (octree nodes / bricks / LeafAttr slots / FillTask slots).
    octree_extent: (u32, u32),
    brick_extent: (u32, u32),
    leaf_attr_extent: (u32, u32),
    fill_task_extent: (u32, u32),
    /// Hash of inputs that affect the BFS topology (host geometry,
    /// region thickness, max_depth, aabb, cell_size). Unchanged →
    /// classify dispatch can be skipped.
    topology_hash: u64,
    /// Hash of inputs that affect per-cell shader output (params,
    /// time-if-animated, paint epoch, shader source). Unchanged →
    /// fill dispatch can be skipped.
    fill_hash: u64,
    /// Stable across frames for the same cache key — used as the
    /// `RkpGpuObject.object_id` so tile/cull lists key consistently.
    object_id: u32,
    max_depth: u32,
    aabb_min: [f32; 3],
    cell_size: f32,
    /// `true` after `begin_frame` until the entry is hit by a
    /// lookup this frame. End-of-frame `evict_untouched` drops
    /// entries still false.
    touched_this_frame: bool,
}

/// Persistent cache + variable-size pool allocator for user-shader
/// transient geometry.
///
/// Cache key: `(host_object_id, material_id, tile_index)`. Each entry
/// owns a contiguous extent in each of the four pools — so a region's
/// bricks and leaf-attrs stay together in memory, preserving cache
/// locality for the march pass.
///
/// Per-frame flow:
///   1. `begin_frame()` marks every entry untouched.
///   2. `lookup_or_allocate(request, topology_hash, fill_hash)` returns
///      a `CachedSlot` carrying the region's extents and two dirty
///      bits (`topology_dirty`, `fill_dirty`):
///        - `topology_dirty=false, fill_dirty=false` → caller skips
///          GPU dispatch; data from prior frame is still valid.
///        - `topology_dirty=false, fill_dirty=true` → caller dispatches
///          fill only (reuse classify output from prior frame).
///        - `topology_dirty=true` → caller dispatches classify + fill.
///   3. `evict_untouched()` returns extents from un-touched entries
///      to the bucket allocators' free lists.
///   4. `build_transient_objects()` emits one `RkpGpuObject` per
///      cached entry so the march pass can find them.
pub struct UserShaderObjectCache {
    entries: HashMap<(u32, u32, [i32; 3]), CacheEntry>,
    /// Variable-size bucket allocators, one per pool. All work in
    /// pool-native units (e.g. brick allocator works in BRICKS).
    octree_alloc: BucketPoolAllocator,
    brick_alloc: BucketPoolAllocator,
    leaf_attr_alloc: BucketPoolAllocator,
    fill_task_alloc: BucketPoolAllocator,
    /// Element offsets into the GPU buffers where each pool's
    /// user-shader transient region begins. Added to allocator
    /// outputs to produce absolute GPU offsets.
    pool_octree_base: u32,
    pool_brick_base: u32,
    pool_leaf_attr_base: u32,
    /// `MAX_GLOBAL_*` clamped to the device's binding limit, seen
    /// last set_pool_bases call. If they change (CPU heads moved),
    /// the cache is flushed.
    pool_octree_capacity: u32,
    pool_brick_capacity: u32,
    pool_leaf_attr_capacity: u32,
    pool_fill_task_capacity: u32,
    next_object_id: u32,
    last_seen_geometry_epoch: u64,
}

const USER_SHADER_OBJECT_ID_BASE: u32 = 0xF000_0000;

impl UserShaderObjectCache {
    pub fn new() -> Self {
        Self::with_capacities(
            MAX_GLOBAL_OCTREE_NODES,
            MAX_GLOBAL_BRICKS,
            MAX_GLOBAL_LEAF_ATTRS,
            MAX_GLOBAL_FILL_TASKS,
        )
    }

    /// Build a cache with explicit pool capacities. Used by tests to
    /// size things tighter without changing the production constants.
    pub fn with_capacities(
        octree_capacity: u32,
        brick_capacity: u32,
        leaf_attr_capacity: u32,
        fill_task_capacity: u32,
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
            fill_task_alloc: BucketPoolAllocator::new(
                fill_task_capacity, FILL_TASK_BUCKET_MIN, FILL_TASK_BUCKET_MAX,
            ),
            pool_octree_base: 0,
            pool_brick_base: 0,
            pool_leaf_attr_base: 0,
            pool_octree_capacity: octree_capacity,
            pool_brick_capacity: brick_capacity,
            pool_leaf_attr_capacity: leaf_attr_capacity,
            pool_fill_task_capacity: fill_task_capacity,
            next_object_id: USER_SHADER_OBJECT_ID_BASE,
            last_seen_geometry_epoch: 0,
        }
    }

    /// Configure the GPU offsets into each pool buffer where the
    /// transient region begins. Called once per frame; if any base
    /// changes (CPU pool head moved), the cache flushes.
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
    pub fn pool_octree_capacity(&self) -> u32 { self.pool_octree_capacity }
    pub fn pool_brick_capacity(&self) -> u32 { self.pool_brick_capacity }
    pub fn pool_leaf_attr_capacity(&self) -> u32 { self.pool_leaf_attr_capacity }
    pub fn pool_fill_task_capacity(&self) -> u32 { self.pool_fill_task_capacity }

    /// Drop every entry and reset all allocators. Used when the
    /// underlying GPU pool buffers reallocate or the host geometry
    /// epoch bumps.
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
        self.fill_task_alloc = BucketPoolAllocator::new(
            self.pool_fill_task_capacity, FILL_TASK_BUCKET_MIN, FILL_TASK_BUCKET_MAX,
        );
    }

    /// If the host geometry epoch advanced since last frame, every
    /// region's `topology_hash` is stale (since topology depends on
    /// host geometry). Flush rather than try to invalidate
    /// individually. Returns `true` if a flush happened.
    pub fn reconcile_epoch(&mut self, geometry_epoch: u64) -> bool {
        if geometry_epoch <= self.last_seen_geometry_epoch {
            return false;
        }
        self.last_seen_geometry_epoch = geometry_epoch;
        if !self.entries.is_empty() {
            self.flush();
            return true;
        }
        false
    }

    /// Mark every entry untouched at the start of a frame. Lookups
    /// touch the entries they hit; `evict_untouched` at the end
    /// drops entries that didn't get a request this frame.
    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.touched_this_frame = false;
        }
    }

    /// Look up or allocate a cache slot. Returns `Some` with the
    /// region's extents + dirty bits derived from comparing the
    /// supplied hashes against the cached values, or `None` on pool
    /// exhaustion (no free extent in the right bucket and no room to
    /// bump high-water).
    pub fn lookup_or_allocate(
        &mut self,
        request: &ShaderRegionRequest,
        topology_hash: u64,
        fill_hash: u64,
        estimate: &PoolEstimate,
    ) -> Option<CachedSlot> {
        let key = (request.host_object_id, request.material_id, request.tile_index);

        // Cache hit: maybe just dirty-bit accounting; maybe re-alloc
        // if the cached extents are too small for the new estimate.
        if let Some(entry) = self.entries.get_mut(&key) {
            let extents_fit = entry.octree_extent.1 >= estimate.octree
                && entry.brick_extent.1 >= estimate.bricks
                && entry.leaf_attr_extent.1 >= estimate.leaf_attrs
                && entry.fill_task_extent.1 >= estimate.fill_tasks;
            let max_depth_match = entry.max_depth == request.max_depth;

            if extents_fit && max_depth_match {
                let topology_dirty = entry.topology_hash != topology_hash;
                let fill_dirty = topology_dirty || entry.fill_hash != fill_hash;
                entry.aabb_min = request.aabb_min;
                entry.cell_size = request.cell_size;
                entry.touched_this_frame = true;
                if topology_dirty {
                    entry.topology_hash = topology_hash;
                }
                if fill_dirty {
                    entry.fill_hash = fill_hash;
                }
                return Some(slot_from_entry(
                    entry, self.pool_octree_base, self.pool_brick_base,
                    self.pool_leaf_attr_base, topology_dirty, fill_dirty,
                ));
            }
            // Stale extents — free them and fall through to alloc.
            free_entry_extents(
                entry,
                &mut self.octree_alloc,
                &mut self.brick_alloc,
                &mut self.leaf_attr_alloc,
                &mut self.fill_task_alloc,
            );
            self.entries.remove(&key);
        }

        // Allocate fresh extents.
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
        let fill_task_extent = match self.fill_task_alloc.alloc(estimate.fill_tasks) {
            Some(e) => e,
            None => {
                self.octree_alloc.free(octree_extent.0, octree_extent.1);
                self.brick_alloc.free(brick_extent.0, brick_extent.1);
                self.leaf_attr_alloc.free(leaf_attr_extent.0, leaf_attr_extent.1);
                return None;
            }
        };

        let object_id = self.next_object_id;
        self.next_object_id = self.next_object_id.wrapping_add(1);

        let entry = CacheEntry {
            octree_root: self.pool_octree_base + octree_extent.0,
            octree_extent,
            brick_extent,
            leaf_attr_extent,
            fill_task_extent,
            topology_hash,
            fill_hash,
            object_id,
            max_depth: request.max_depth,
            aabb_min: request.aabb_min,
            cell_size: request.cell_size,
            touched_this_frame: true,
        };
        let result = slot_from_entry(
            &entry, self.pool_octree_base, self.pool_brick_base,
            self.pool_leaf_attr_base, true, true,
        );
        self.entries.insert(key, entry);
        Some(result)
    }

    /// Drop entries not referenced this frame and return their
    /// extents to the bucket allocators' free lists.
    pub fn evict_untouched(&mut self) {
        let mut to_remove: Vec<(u32, u32, [i32; 3])> = Vec::new();
        for (key, entry) in self.entries.iter() {
            if !entry.touched_this_frame {
                to_remove.push(*key);
            }
        }
        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) {
                self.octree_alloc.free(entry.octree_extent.0, entry.octree_extent.1);
                self.brick_alloc.free(entry.brick_extent.0, entry.brick_extent.1);
                self.leaf_attr_alloc.free(entry.leaf_attr_extent.0, entry.leaf_attr_extent.1);
                self.fill_task_alloc.free(entry.fill_task_extent.0, entry.fill_task_extent.1);
            }
        }
    }

    /// Iterate cached entries (touched this frame) and emit one
    /// `RkpGpuObject` each so the march pass finds the geometry.
    pub fn build_transient_objects(&self) -> Vec<RkpGpuObject> {
        self.entries
            .values()
            .filter(|e| e.touched_this_frame)
            .map(transient_gpu_object)
            .collect()
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn brick_high_water(&self) -> u32 { self.brick_alloc.high_water() }
    pub fn octree_high_water(&self) -> u32 { self.octree_alloc.high_water() }
    pub fn leaf_attr_high_water(&self) -> u32 { self.leaf_attr_alloc.high_water() }
    pub fn fill_task_high_water(&self) -> u32 { self.fill_task_alloc.high_water() }
}

impl Default for UserShaderObjectCache {
    fn default() -> Self { Self::new() }
}

fn slot_from_entry(
    entry: &CacheEntry,
    pool_octree_base: u32,
    pool_brick_base: u32,
    pool_leaf_attr_base: u32,
    topology_dirty: bool,
    fill_dirty: bool,
) -> CachedSlot {
    CachedSlot {
        region_index: 0, // populated by the caller after gather_dirty_regions
        octree_root: entry.octree_root,
        octree_block_offset: pool_octree_base + entry.octree_extent.0,
        octree_block_size: entry.octree_extent.1,
        brick_block_offset: pool_brick_base + entry.brick_extent.0,
        brick_block_size: entry.brick_extent.1,
        leaf_attr_block_offset: pool_leaf_attr_base + entry.leaf_attr_extent.0,
        leaf_attr_block_size: entry.leaf_attr_extent.1,
        fill_task_block_offset: entry.fill_task_extent.0,
        fill_task_block_size: entry.fill_task_extent.1,
        object_id: entry.object_id,
        max_depth: entry.max_depth,
        topology_dirty,
        fill_dirty,
    }
}

fn free_entry_extents(
    entry: &CacheEntry,
    octree_alloc: &mut BucketPoolAllocator,
    brick_alloc: &mut BucketPoolAllocator,
    leaf_attr_alloc: &mut BucketPoolAllocator,
    fill_task_alloc: &mut BucketPoolAllocator,
) {
    octree_alloc.free(entry.octree_extent.0, entry.octree_extent.1);
    brick_alloc.free(entry.brick_extent.0, entry.brick_extent.1);
    leaf_attr_alloc.free(entry.leaf_attr_extent.0, entry.leaf_attr_extent.1);
    fill_task_alloc.free(entry.fill_task_extent.0, entry.fill_task_extent.1);
}

/// Per-region pool size estimate driving extent allocation. The
/// bucket allocator rounds up to the next bucket, so over-estimation
/// is cheap; under-estimation drops bricks and leaves visual holes.
#[derive(Debug, Clone, Copy)]
pub struct PoolEstimate {
    pub octree: u32,
    pub bricks: u32,
    pub leaf_attrs: u32,
    pub fill_tasks: u32,
}

/// Estimate per-region pool needs from TILE GEOMETRY.
///
/// The BFS classifier descends every cell within `region_thickness`
/// of the host surface, regardless of paint material. So `fill_tasks`
/// and `octree` counts scale with **tile dimensions × proximity-band
/// fraction**, NOT painted-cell count.
///
/// `bricks` and `leaf_attrs` ARE paint-driven: V12 deferred allocation
/// only consumes a brick slot when the user shader emits at least one
/// occupied cell. We use `painted_leaf_count` as a paint-density
/// proxy and clamp at the geometric upper bound.
///
/// Inputs all come from `ShaderRegionRequest` — `aabb_min/max`,
/// `cell_size`, `max_depth`, `region_thickness`, `painted_leaf_count`.
pub fn estimate_region_pool(request: &ShaderRegionRequest) -> PoolEstimate {
    // Brick-parent cells per axis at depth `max_depth`. Each
    // brick-parent spans `cell_size * BRICK_DIM = cell_size * 4`.
    let extent = (request.aabb_max[0] - request.aabb_min[0]).max(1e-6);
    let bp_cell = (request.cell_size * 4.0).max(1e-6);
    let bp_per_axis = ((extent / bp_cell).ceil() as u32).max(1);
    let bp_total = bp_per_axis.saturating_mul(bp_per_axis).saturating_mul(bp_per_axis);

    // Proximity-band fraction. With band B + half-cell-diag headroom
    // the gate keeps cells whose Lipschitz lower bound puts them
    // within ±(B + diag) of the surface. For a roughly-flat host
    // surface this is approximately
    //   fraction ≈ min(1, 2 * (B + diag) / extent)
    // Round generously up — over-estimating fill tasks is cheap.
    let band = request.region_thickness;
    let bp_diag_half = bp_cell * 0.866_025_4; // sqrt(3)/2
    let band_thickness = band + bp_diag_half;
    let band_fraction = if band > 0.0 {
        ((2.0 * band_thickness / extent).min(1.0)).max(0.5)
    } else {
        // No proximity gate → every cell is MIXED.
        1.0
    };
    // Estimates are clamped at the corresponding bucket-max so the
    // allocator can always satisfy the request. A region that
    // legitimately needs more than the max bucket falls back to the
    // GPU-side overflow counters (graceful degradation: the relevant
    // pool's overflow counter increments and individual bricks /
    // cells / branches drop to OCTREE_EMPTY).
    let fill_tasks = ((bp_total as f32 * band_fraction).ceil() as u32)
        .max(FILL_TASK_BUCKET_MIN)
        .min(FILL_TASK_BUCKET_MAX);

    // Octree allocations: sum across levels of (MIXED-cells × 8).
    // Conservative: 1.2 × fill_tasks (deepest level dominates) +
    // a constant for the spine.
    let depth_overhead = (request.max_depth.max(1) + 1) * 8;
    let octree = ((fill_tasks as u64 * 12 / 10) as u32)
        .saturating_add(depth_overhead)
        .max(OCTREE_BUCKET_MIN)
        .min(OCTREE_BUCKET_MAX);

    // Bricks: paint-driven via V12 deferred allocation. For grass-style
    // shaders each painted host cell projects to several brick-parent
    // cells (vertical extent of blade × thinness × cluster density).
    // Multiplier 12 sizes per-region brick blocks at bucket 8192 for
    // typical 1 m grass tiles, leaving 3 M / 8192 = 366 typical
    // regions fitting globally. Higher multiplier reduces per-region
    // overflow but eats more of the global pool — at 1 GB binding
    // limit, that means whole-tile drops once paint coverage grows.
    // Per-region overflow degrades gracefully (a few missing bricks
    // per tile, visible as small holes); whole-tile drops are much
    // worse visually.
    let painted = request.painted_leaf_count.max(BRICK_BUCKET_MIN);
    let bricks = painted
        .saturating_mul(12)
        .min(fill_tasks)
        .max(BRICK_BUCKET_MIN)
        .min(BRICK_BUCKET_MAX);

    // Leaf-attrs: each emitting brick has up to ~BRICK_CELLS / 2 = 32
    // occupied cells for grass-density shaders. Higher-density shaders
    // (full solids) overflow gracefully via the overflow counter.
    let leaf_attrs = bricks
        .saturating_mul(BRICK_CELLS / 2)
        .max(LEAF_ATTR_BUCKET_MIN)
        .min(LEAF_ATTR_BUCKET_MAX);

    PoolEstimate {
        octree,
        bricks,
        leaf_attrs,
        fill_tasks,
    }
}

fn transient_gpu_object(entry: &CacheEntry) -> RkpGpuObject {
    // Brick grid spans `cell_size * BRICK_DIM * 2^max_depth` per axis
    // from `aabb_min` — the BFS still yields a (4 * 2^max_depth)³ cube
    // at full depth.
    let bricks_per_axis = (1u32 << entry.max_depth) as f32;
    let extent = entry.cell_size * 4.0 * bricks_per_axis;
    let aabb_max_cube = [
        entry.aabb_min[0] + extent,
        entry.aabb_min[1] + extent,
        entry.aabb_min[2] + extent,
    ];
    let identity: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    RkpGpuObject {
        world: identity,
        aabb_min: entry.aabb_min,
        octree_root: entry.octree_root,
        aabb_max: aabb_max_cube,
        octree_depth: entry.max_depth,
        octree_extent_bits: extent.to_bits(),
        voxel_size: entry.cell_size,
        material_id: 0,
        object_id: entry.object_id,
        geom_type: geom_type::VOXELIZED,
        is_skinned: 0,
        bone_count: 0,
        bone_buffer_offset: 0,
        rest_octree_root: 0,
        rest_octree_depth: 0,
        rest_octree_extent_bits: 0,
        bone_field_offset: 0,
        layer_mask: u32::MAX,
        bone_field_dim_x: 0,
        bone_field_dim_y: 0,
        bone_field_dim_z: 0,
        bone_field_origin_x: 0.0,
        bone_field_origin_y: 0.0,
        bone_field_origin_z: 0.0,
        bone_field_occ_offset: 0,
        grid_origin: entry.aabb_min,
        _post_grid_pad: 0,
        inverse_world: identity,
    }
}

/// Slot descriptor returned from `lookup_or_allocate`. Carries the
/// per-region state the host needs to populate its `RegionUniform`
/// upload, plus dirty bits indicating whether classify, fill, or
/// neither needs to dispatch this frame.
#[derive(Debug, Clone, Copy)]
pub struct CachedSlot {
    /// Region index in this frame's dispatch arrays. Populated by the
    /// caller after gathering all dirty slots — `lookup_or_allocate`
    /// returns 0 here; the caller assigns sequential indices and
    /// updates the underlying entry.
    pub region_index: u32,
    /// Global pool offset where this region's octree root lives.
    pub octree_root: u32,
    /// Per-pool block offsets (absolute, ready for the GPU) and sizes.
    pub octree_block_offset: u32,
    pub octree_block_size: u32,
    pub brick_block_offset: u32,
    pub brick_block_size: u32,
    pub leaf_attr_block_offset: u32,
    pub leaf_attr_block_size: u32,
    /// Fill-task pool offset is in FillTask units; relative to the
    /// fill-task pool buffer (no separate "base" — the pool is
    /// owned entirely by the user-shader pass).
    pub fill_task_block_offset: u32,
    pub fill_task_block_size: u32,
    pub object_id: u32,
    pub max_depth: u32,
    /// `true` when topology inputs differ from the cached values —
    /// classify must re-run.
    pub topology_dirty: bool,
    /// `true` when fill inputs differ. Always `true` when
    /// `topology_dirty` is.
    pub fill_dirty: bool,
}

/// Per-region uniform — laid out to match WGSL's std430 storage layout
/// for `array<RegionUniform>`.
///
/// 208 bytes. Carries per-region pool block offsets/sizes (allocator
/// output) so each region's allocator atomicAdd composes a global
/// pool offset as `block_offset + atomic_slot`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionUniform {
    pub aabb_min: [f32; 3],                // offset  0
    pub cell_size: f32,                     // offset 12
    pub aabb_max: [f32; 3],                 // offset 16
    pub shader_id: u32,                     // offset 28
    pub max_depth: u32,                     // offset 32
    pub time: f32,                          // offset 36
    pub material_id: u32,                   // offset 40
    pub region_thickness: f32,              // offset 44
    pub host_octree_root: u32,              // offset 48
    pub host_octree_depth: u32,             // offset 52
    pub host_octree_extent: f32,            // offset 56
    /// Per-region pool block offsets + sizes. Offsets are absolute
    /// GPU-buffer indices; sizes are the bucket-rounded extents the
    /// allocator handed out. Units: octree_block in vec2<u32>;
    /// brick_block in BRICKS; leaf_attr_block in LeafAttr;
    /// fill_task_block in BrickFillTask.
    pub octree_block_offset: u32,           // offset 60
    pub octree_block_size: u32,             // offset 64
    pub brick_block_offset: u32,            // offset 68
    pub brick_block_size: u32,              // offset 72
    pub leaf_attr_block_offset: u32,        // offset 76
    pub leaf_attr_block_size: u32,          // offset 80
    pub fill_task_block_offset: u32,        // offset 84
    pub fill_task_block_size: u32,          // offset 88
    /// Pad so `host_grid_origin` (vec3, 16-byte aligned in WGSL)
    /// lands at the next 16-aligned offset (96).
    pub _pad_host: u32,                     // offset 92
    pub host_grid_origin: [f32; 3],         // offset 96
    /// Pad so `params` (vec4) lands at offset 112.
    pub _pad_grid: f32,                     // offset 108
    pub params: [[f32; 4]; 2],              // offset 112
    pub host_inverse_world: [[f32; 4]; 4],  // offset 144
}

const _: () = assert!(std::mem::size_of::<RegionUniform>() == 208);

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

const LEVEL_UNIFORM_STRIDE: u64 = 256;

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
    let geom_src = include_str!("shaders/user_shader_geom.wgsl");
    if user_chunk.is_empty() {
        return geom_src.to_string();
    }
    const BEGIN: &str = "// USER_GENERATE_DISPATCH_BEGIN";
    const END: &str = "// USER_GENERATE_DISPATCH_END";
    let begin = geom_src.find(BEGIN).expect(
        "user_shader_geom.wgsl missing USER_GENERATE_DISPATCH_BEGIN marker",
    );
    let end_after = geom_src[begin..]
        .find(END)
        .map(|off| begin + off + END.len())
        .expect("user_shader_geom.wgsl missing USER_GENERATE_DISPATCH_END marker");
    let mut out = String::with_capacity(geom_src.len() + user_chunk.len());
    out.push_str(&geom_src[..begin]);
    out.push_str(user_chunk);
    out.push_str(&geom_src[end_after..]);
    out
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

/// CPU-side machinery for reading back the GPU overflow counters with
/// 3-frame ring buffering. We copy `overflow_buffer` into `slots[i]`
/// each frame, then `map_async` it. Frames in flight don't stall the
/// GPU; we drain the oldest mapped slot each frame and log if any
/// counter is non-zero.
struct OverflowReadback {
    slots: [OverflowReadbackSlot; OVERFLOW_READBACK_FRAMES],
    next_write: usize,
    /// True if a slot's `map_async` has resolved and the buffer is
    /// ready to read. The flag is shared with the map callback via
    /// an `Arc<AtomicBool>`.
    map_states: [std::sync::Arc<std::sync::atomic::AtomicU8>; OVERFLOW_READBACK_FRAMES],
}

struct OverflowReadbackSlot {
    buffer: wgpu::Buffer,
    in_flight: bool,
}

const MAP_STATE_IDLE: u8 = 0;
const MAP_STATE_PENDING: u8 = 1;
const MAP_STATE_READY: u8 = 2;
const MAP_STATE_FAILED: u8 = 3;

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
        const FILL_TASK_SENTINEL: u32 = 0xFFFFFFFE;
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

impl OverflowReadback {
    fn new(device: &wgpu::Device) -> Self {
        let make_slot = |i: usize| OverflowReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("user_shader_geom overflow stage {i}")),
                size: OVERFLOW_BUFFER_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            in_flight: false,
        };
        let make_state = || std::sync::Arc::new(
            std::sync::atomic::AtomicU8::new(MAP_STATE_IDLE),
        );
        Self {
            slots: [make_slot(0), make_slot(1), make_slot(2)],
            map_states: [make_state(), make_state(), make_state()],
            next_write: 0,
        }
    }

    /// Returns the staging buffer to copy into this frame, or `None`
    /// if the slot's previous map_async hasn't completed yet (we don't
    /// double-book).
    fn next_write_buffer(&mut self) -> Option<&wgpu::Buffer> {
        let idx = self.next_write;
        let state = self.map_states[idx].load(std::sync::atomic::Ordering::Acquire);
        // IDLE → never used yet, free to write.
        // FAILED → previous map errored; we already reset the buffer
        //   in drain_and_log so it's free again.
        // PENDING → in flight; skip this frame to avoid clobbering.
        // READY → drain_and_log not yet called; skip too.
        if state == MAP_STATE_IDLE || state == MAP_STATE_FAILED {
            self.slots[idx].in_flight = true;
            Some(&self.slots[idx].buffer)
        } else {
            None
        }
    }

    /// Schedule map_async on the slot we just copied into. Call AFTER
    /// the queue.submit so the copy is in flight.
    fn advance(&mut self) {
        let idx = self.next_write;
        if !self.slots[idx].in_flight {
            return;
        }
        self.slots[idx].in_flight = false;
        let state = std::sync::Arc::clone(&self.map_states[idx]);
        state.store(MAP_STATE_PENDING, std::sync::atomic::Ordering::Release);
        let buffer = &self.slots[idx].buffer;
        let slice = buffer.slice(0..OVERFLOW_BUFFER_BYTES);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let next = if result.is_ok() {
                MAP_STATE_READY
            } else {
                MAP_STATE_FAILED
            };
            state.store(next, std::sync::atomic::Ordering::Release);
        });
        self.next_write = (idx + 1) % OVERFLOW_READBACK_FRAMES;
    }

    /// Walk every slot; for each that's READY, read its bytes, log if
    /// non-zero, unmap, mark IDLE.
    fn drain_and_log(&mut self) {
        for idx in 0..OVERFLOW_READBACK_FRAMES {
            let state = self.map_states[idx].load(std::sync::atomic::Ordering::Acquire);
            if state == MAP_STATE_READY {
                let buffer = &self.slots[idx].buffer;
                let slice = buffer.slice(0..OVERFLOW_BUFFER_BYTES);
                let counts: Vec<u32> = {
                    let view = slice.get_mapped_range();
                    bytemuck::cast_slice::<u8, u32>(&view).to_vec()
                };
                buffer.unmap();
                self.map_states[idx].store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
                if counts.iter().any(|c| *c != 0) {
                    log_overflow(&counts);
                }
            } else if state == MAP_STATE_FAILED {
                eprintln!("[user_shader_pass] overflow map_async failed in slot {idx}");
                self.map_states[idx].store(MAP_STATE_IDLE, std::sync::atomic::Ordering::Release);
            }
        }
    }
}

fn log_overflow(counts: &[u32]) {
    let octree = counts.first().copied().unwrap_or(0);
    let brick = counts.get(1).copied().unwrap_or(0);
    let leaf_attr = counts.get(2).copied().unwrap_or(0);
    let fill_queue = counts.get(3).copied().unwrap_or(0);
    eprintln!(
        "[user_shader_pass] OVERFLOW — octree:{octree} brick:{brick} \
         leaf_attr:{leaf_attr} fill_queue:{fill_queue}",
    );
    for level in 0..=(MAX_DEPTH as usize) {
        let c = counts.get(4 + level).copied().unwrap_or(0);
        if c != 0 {
            eprintln!("[user_shader_pass] OVERFLOW — active_queue[L={level}]:{c}");
        }
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

/// Build the per-region uniform from a request + region slot.
pub fn build_region_uniform(
    request: &ShaderRegionRequest,
    slot: &CachedSlot,
    shader_id: u32,
    time_seconds: f32,
) -> RegionUniform {
    let mut params = [[0.0f32; 4]; 2];
    for (i, &v) in request.params.iter().take(8).enumerate() {
        params[i / 4][i % 4] = v;
    }
    RegionUniform {
        aabb_min: request.aabb_min,
        cell_size: request.cell_size,
        aabb_max: request.aabb_max,
        shader_id,
        max_depth: slot.max_depth,
        time: time_seconds,
        material_id: request.material_id,
        region_thickness: request.region_thickness,
        host_octree_root: request.host_octree_root,
        host_octree_depth: request.host_octree_depth,
        host_octree_extent: request.host_octree_extent,
        octree_block_offset: slot.octree_block_offset,
        octree_block_size: slot.octree_block_size,
        brick_block_offset: slot.brick_block_offset,
        brick_block_size: slot.brick_block_size,
        leaf_attr_block_offset: slot.leaf_attr_block_offset,
        leaf_attr_block_size: slot.leaf_attr_block_size,
        fill_task_block_offset: slot.fill_task_block_offset,
        fill_task_block_size: slot.fill_task_block_size,
        _pad_host: 0,
        host_grid_origin: request.host_grid_origin,
        _pad_grid: 0.0,
        params,
        host_inverse_world: request.host_inverse_world,
    }
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
mod tests {
    use super::*;

    fn req(host: u32, mat: u32) -> ShaderRegionRequest {
        ShaderRegionRequest {
            host_object_id: host,
            material_id: mat,
            shader_name: "x".to_string(),
            params: vec![],
            aabb_min: [0.0; 3],
            aabb_max: [1.0; 3],
            cell_size: 0.25,
            input_hash: 0,
            animated: false,
            region_thickness: 0.0,
            max_depth: 4,
            painted_leaf_count: 8,
            host_octree_root: HOST_NO_HOST_SENTINEL,
            host_octree_depth: 0,
            host_octree_extent: 0.0,
            host_grid_origin: [0.0; 3],
            host_inverse_world: [[0.0; 4]; 4],
            tile_index: NO_TILE,
        }
    }

    #[test]
    fn geom_shader_validates_with_empty_chunk() {
        let src = compose_geom_source("");
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
            panic!("parse error:\n{}", e.emit_to_string(&src))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }

    #[test]
    fn compose_splices_user_chunk() {
        let chunk = "fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit { return voxel_emit_skip(); }";
        let src = compose_geom_source(chunk);
        assert!(src.contains("dispatch_user_generate"));
        assert!(!src.contains("Default identity stub"));
    }

    fn small_cache() -> UserShaderObjectCache {
        // Tight test pool: 1024 octree, 256 bricks, 4096 leaf-attrs,
        // 256 fill tasks. Big enough for a handful of small regions.
        UserShaderObjectCache::with_capacities(1024, 256, 4096, 256)
    }

    fn small_estimate() -> PoolEstimate {
        // Small enough to fit several entries in `small_cache`.
        PoolEstimate {
            octree: 64,
            bricks: 16,
            leaf_attrs: 512,
            fill_tasks: 16,
        }
    }

    #[test]
    fn cache_first_lookup_is_topology_and_fill_dirty() {
        let mut c = small_cache();
        let s = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
        assert!(s.topology_dirty);
        assert!(s.fill_dirty);
    }

    #[test]
    fn cache_second_lookup_with_same_hashes_is_clean() {
        let mut c = small_cache();
        let s1 = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
        let s2 = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
        // Cache hit, both hashes match → both flags clean.
        assert!(!s2.topology_dirty);
        assert!(!s2.fill_dirty);
        // Same physical extents reused.
        assert_eq!(s1.octree_root, s2.octree_root);
        assert_eq!(s1.brick_block_offset, s2.brick_block_offset);
        assert_eq!(s1.object_id, s2.object_id);
    }

    #[test]
    fn cache_topology_unchanged_fill_changed_yields_fill_only() {
        let mut c = small_cache();
        c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
        // Different fill hash, same topology hash.
        let s = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xCC, &small_estimate()).unwrap();
        assert!(!s.topology_dirty);
        assert!(s.fill_dirty);
    }

    #[test]
    fn cache_topology_changed_yields_full_rebake() {
        let mut c = small_cache();
        c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
        let s = c.lookup_or_allocate(&req(1, 1), 0xCC, 0xBB, &small_estimate()).unwrap();
        assert!(s.topology_dirty);
        assert!(s.fill_dirty);
    }

    #[test]
    fn cache_distinguishes_keys() {
        let mut c = small_cache();
        let s1 = c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        let s2 = c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
        // Different (object, material) → different extents.
        assert_ne!(s1.octree_root, s2.octree_root);
        assert_ne!(s1.brick_block_offset, s2.brick_block_offset);
    }

    #[test]
    fn evict_untouched_returns_extents_to_free_list() {
        let mut c = small_cache();
        c.begin_frame();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
        let pre_brick_high = c.brick_high_water();
        // Frame 2 — only touch one of the two entries.
        c.begin_frame();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        c.evict_untouched();
        // Untouched entry's extents are now in free lists; brick
        // high-water shouldn't have advanced.
        assert_eq!(c.brick_high_water(), pre_brick_high);
        assert_eq!(c.entry_count(), 1);
        // Frame 3 — request a NEW key; should reuse a freed bucket
        // before bumping high-water.
        c.begin_frame();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        c.lookup_or_allocate(&req(1, 9), 0, 0, &small_estimate()).unwrap();
        assert_eq!(c.brick_high_water(), pre_brick_high);
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        // Pool sized for exactly one tiny region.
        let mut c = UserShaderObjectCache::with_capacities(64, 16, 512, 16);
        assert!(c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).is_some());
        // Second allocation has nothing left.
        assert!(c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).is_none());
    }

    #[test]
    fn build_transient_objects_includes_only_touched() {
        let mut c = small_cache();
        c.begin_frame();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
        // Frame 2 — touch only one.
        c.begin_frame();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        let objs = c.build_transient_objects();
        // Only the touched entry shows up; the untouched one is
        // pending eviction at end-of-frame and shouldn't render.
        assert_eq!(objs.len(), 1);
    }

    #[test]
    fn flush_on_geometry_epoch_bump() {
        let mut c = small_cache();
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        assert_eq!(c.entry_count(), 1);
        assert!(c.reconcile_epoch(1));
        assert_eq!(c.entry_count(), 0);
        // Subsequent lookup is a fresh allocation.
        let s = c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        assert!(s.topology_dirty);
    }

    #[test]
    fn flush_on_pool_base_change() {
        let mut c = small_cache();
        c.set_pool_bases(0, 0, 0);
        c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
        assert_eq!(c.entry_count(), 1);
        // Different bases → flush.
        c.set_pool_bases(100, 200, 300);
        assert_eq!(c.entry_count(), 0);
    }

    #[test]
    fn region_uniform_size_is_208() {
        assert_eq!(std::mem::size_of::<RegionUniform>(), 208);
    }

    #[test]
    fn level_uniform_size_is_16() {
        assert_eq!(std::mem::size_of::<LevelUniform>(), 16);
    }

    #[test]
    fn active_cell_size_is_32() {
        assert_eq!(std::mem::size_of::<ActiveCell>(), 32);
    }

    #[test]
    fn allocator_rounds_up_to_next_bucket() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        // Request 17 → bucket 32.
        let (o, s) = a.alloc(17).unwrap();
        assert_eq!(o, 0);
        assert_eq!(s, 32);
        // Request exactly 32 → bucket 32.
        let (o2, s2) = a.alloc(32).unwrap();
        assert_eq!(o2, 32);
        assert_eq!(s2, 32);
        // Request 200 → bucket 256.
        let (o3, s3) = a.alloc(200).unwrap();
        assert_eq!(o3, 64);
        assert_eq!(s3, 256);
    }

    #[test]
    fn allocator_clamps_below_min_bucket() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        // Request 1 → still get bucket 16.
        let (_, s) = a.alloc(1).unwrap();
        assert_eq!(s, 16);
    }

    #[test]
    fn allocator_rejects_above_max_bucket() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        // Request 257 → exceeds max bucket → reject.
        assert!(a.alloc(257).is_none());
    }

    #[test]
    fn allocator_reuses_freed_extents_per_bucket() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        let (o1, s1) = a.alloc(20).unwrap();
        assert_eq!(s1, 32);
        let pre_high = a.high_water();
        a.free(o1, s1);
        assert_eq!(a.free_count(), 1);
        // Re-alloc same bucket → reuses freed offset, doesn't bump high-water.
        let (o2, s2) = a.alloc(20).unwrap();
        assert_eq!(o2, o1);
        assert_eq!(s2, 32);
        assert_eq!(a.high_water(), pre_high);
        assert_eq!(a.free_count(), 0);
    }

    #[test]
    fn allocator_separate_free_lists_per_bucket() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        let (o16, s16) = a.alloc(16).unwrap();
        let (_o32, s32) = a.alloc(32).unwrap();
        a.free(o16, s16);
        // Asking for 32 should NOT pop the 16-bucket free list.
        let (o32b, s32b) = a.alloc(32).unwrap();
        assert_eq!(s32b, 32);
        assert_ne!(o32b, o16);
        // Asking for 16 picks up the freed 16.
        let (o16b, s16b) = a.alloc(16).unwrap();
        assert_eq!(o16b, o16);
        assert_eq!(s16b, 16);
        let _ = (s16, s32);
    }

    #[test]
    fn allocator_exhaustion_returns_none() {
        let mut a = BucketPoolAllocator::new(64, 16, 64);
        assert!(a.alloc(64).is_some()); // claim 0..64
        assert!(a.alloc(16).is_none());  // no room
    }

    #[test]
    fn allocator_high_water_only_advances_on_fresh_alloc() {
        let mut a = BucketPoolAllocator::new(1024, 16, 256);
        let (_, s1) = a.alloc(20).unwrap();
        let after_first = a.high_water();
        a.free(0, s1);
        let _ = a.alloc(20).unwrap(); // should reuse, not advance
        assert_eq!(a.high_water(), after_first);
    }

    #[test]
    fn resolve_shader_id_alphabetical_one_based() {
        let infos = vec![
            UserShaderInfo { name: "zeta".into(), ..Default::default() },
            UserShaderInfo { name: "alpha".into(), ..Default::default() },
            UserShaderInfo { name: "mu".into(), ..Default::default() },
        ];
        assert_eq!(resolve_shader_id(&infos, "alpha"), 1);
        assert_eq!(resolve_shader_id(&infos, "mu"), 2);
        assert_eq!(resolve_shader_id(&infos, "zeta"), 3);
        assert_eq!(resolve_shader_id(&infos, ""), 0);
        assert_eq!(resolve_shader_id(&infos, "missing"), 0);
    }
}
