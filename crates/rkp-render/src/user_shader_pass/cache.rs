//! Variable-size pool allocator + persistent per-region cache for the
//! user-shader BFS bake.
//!
//! Owns:
//! - [`BucketPoolAllocator`] — bump-allocates power-of-2 extents from a
//!   fixed-capacity pool, with per-bucket free lists for return.
//! - [`UserShaderObjectCache`] — keys `(host_object_id, material_id,
//!   tile_index) → CacheEntry`; carries each region's extents in the
//!   four flat pools (octree / brick / leaf-attr / fill-task), plus the
//!   topology + fill hashes that decide whether GPU dispatch can be
//!   skipped this frame.
//! - [`ShaderRegionRequest`] — the sim → render API the cache keys on.
//! - [`PoolEstimate`] + [`estimate_region_pool`] — sizing the per-region
//!   extents from tile dimensions + paint density.
//! - [`CachedSlot`] — what `lookup_or_allocate` hands back: GPU-absolute
//!   pool offsets + dirty bits.
//! - Pool-capacity + bucket-bound constants shared with the dispatch.

use std::collections::HashMap;

use crate::rkp_gpu_object::{geom_type, RkpGpuAsset, RkpGpuInstance};

/// Sentinel "no host" value matching `HOST_NO_HOST_SENTINEL` in WGSL.
pub const HOST_NO_HOST_SENTINEL: u32 = 0xFFFF_FFFFu32;

/// Sentinel `tile_index` value used for non-tiled shaders (those
/// without an `@tile_size` directive). One cache entry per
/// (object, material) pair, V9 behaviour.
pub const NO_TILE: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Cells per brick — must match `rkp_core::brick_pool::BRICK_CELLS`.
pub const BRICK_CELLS: u32 = 64;

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

/// Phase B-redux 3b — global band-cell pool capacity. Each
/// `GpuBandCell` is 16 B carrying `(anchor_world_pos: vec3, region_index: u32)`.
/// Sized for ~100 active painted tiles × ~100K band cells/tile = 10M
/// cells = 160 MB. The BFS bake bumps a global atomic counter into
/// this pool whenever an `instance_at` shader's region produces a
/// max-depth band cell.
pub const MAX_GLOBAL_BAND_CELLS: u32 = 16_000_000;

// Note: an earlier draft of 3b carried a separate `GpuBandRegion`
// table indexed by `BandCell.region_index`. V1 ships material_id
// directly on the BandCell instead — the table was dropped to
// avoid a new march binding for one tiny lookup.

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

/// Base for cache-allocated `object_id`s. Each new entry bumps from
/// here; the high bit keeps user-shader transient ids out of the host
/// object id space.
const USER_SHADER_OBJECT_ID_BASE: u32 = 0xF000_0000;

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

// ============================================================
// Sim → render request
// ============================================================

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
    /// Phase B-redux 3b — `true` when the BFS should bake band cells
    /// (with `instance_at` derivation hook) instead of voxel bricks.
    /// Routed by sim from `UserShaderInfo.has_instance_at`. Mutually
    /// exclusive with the voxel-emit path within one region; a shader
    /// that has both `generate` and `instance_at` is rejected by the
    /// composer.
    pub is_band_region: bool,
    /// Phase B-redux band-cell anchor projection target. World-space
    /// y of the painted surface; the BFS uses this directly as the
    /// anchor's y when `is_band_region == true`. Replaces V1's
    /// `host.distance` projection which was unreliable on hosts whose
    /// octree subdivides finely above the surface (returns ~0 for
    /// cells in BRICK_CELL_EMPTY, gating off and producing scattered
    /// blades). Computed CPU-side from the painted leaves' world-space
    /// y. Flat-surface only for V1; sloped/curved hosts need a more
    /// expressive scheme (per-cell projection or multi-source BFS).
    pub host_surface_y: f32,
    /// Phase B-redux V1.1 — world-space AABB of the painted leaves.
    /// The BFS uses this on the band path to gate horizontally:
    /// cells whose x/z fall outside `[painted_min - band, painted_max + band]`
    /// are rejected. Without this gate, blades fill the whole tile
    /// cube horizontally instead of just the painted area.
    pub painted_world_min: [f32; 3],
    pub painted_world_max: [f32; 3],
}

// ============================================================
// Per-region cache
// ============================================================

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
    /// (asset, instance) pair each so the march pass finds the geometry.
    /// Each transient region is its own asset (unique octree slot — no
    /// dedupe with persistent assets or among transients), and one
    /// instance per asset since transients aren't multi-instanced.
    pub fn build_transient_assets_and_instances(
        &self,
        asset_id_base: u32,
    ) -> (Vec<RkpGpuAsset>, Vec<RkpGpuInstance>) {
        let mut assets: Vec<RkpGpuAsset> = Vec::new();
        let mut instances: Vec<RkpGpuInstance> = Vec::new();
        for e in self.entries.values().filter(|e| e.touched_this_frame) {
            let (a, mut i) = transient_asset_and_instance(e);
            i.asset_id = asset_id_base + assets.len() as u32;
            assets.push(a);
            instances.push(i);
        }
        (assets, instances)
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

// ============================================================
// Pool-size estimator
// ============================================================

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

fn transient_asset_and_instance(entry: &CacheEntry) -> (RkpGpuAsset, RkpGpuInstance) {
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
    let asset = RkpGpuAsset {
        aabb_min: entry.aabb_min,
        octree_root: entry.octree_root,
        aabb_max: aabb_max_cube,
        octree_depth: entry.max_depth,
        octree_extent_bits: extent.to_bits(),
        voxel_size: entry.cell_size,
        geom_type: geom_type::VOXELIZED,
        bone_count: 0,
        grid_origin: entry.aabb_min,
        rest_octree_root: 0,
        rest_octree_depth: 0,
        rest_octree_extent_bits: 0,
        // Phase C transient regions are NOT user-shader instance
        // protos — they're the per-region geometry-build pass output.
        // Standard host-march descent, no per-instance hooks.
        shader_id: 0,
        _pad: 0,
    };
    let instance = RkpGpuInstance {
        world: identity,
        asset_id: 0, // caller assigns the actual slot index
        material_id: 0,
        object_id: entry.object_id,
        layer_mask: u32::MAX,
        is_skinned: 0,
        bone_buffer_offset: 0,
        bone_field_offset: 0,
        bone_field_occ_offset: 0,
        bone_field_dim_x: 0,
        bone_field_dim_y: 0,
        bone_field_dim_z: 0,
        bone_field_origin_x: 0.0,
        bone_field_origin_y: 0.0,
        bone_field_origin_z: 0.0,
        // Transient user-shader regions never carry per-instance paint
        // overlays — they're rebuilt each frame from the procedural
        // pass, so paint can't accumulate on them.
        overlay_offset: 0,
        overlay_count: 0,
    };
    (asset, instance)
}

// ============================================================
// Cache lookup result
// ============================================================

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

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
