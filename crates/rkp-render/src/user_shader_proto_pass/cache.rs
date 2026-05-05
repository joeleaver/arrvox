//! `PrototypeCache` — persistent per-shader prototype slot tracking +
//! octree-extent allocator with free-list reuse on re-bake.
//!
//! Per-frame flow: `begin_frame` marks all entries untouched →
//! per-instance-shader `lookup_or_allocate` flips the touched bit and
//! returns the slot + dirty bit → `evict_untouched` at frame end frees
//! the orphan extents back to the free-list.
//!
//! Brick + leaf-attr cursors live on [`super::pass::PrototypeBakePass`]
//! (global cursors that interleave across prototypes); only the octree
//! extent is per-prototype tracked here.

use std::collections::HashMap;

use super::types::{
    level_starts_inclusive, octree_node_count_for_depth, PrototypeEntry, INTERNAL_ATTR_NONE,
    MAX_PROTO_MAX_DEPTH, OCTREE_EMPTY, PROTO_BRICK_POOL_CAPACITY, PROTO_LEAF_ATTR_POOL_CAPACITY,
    PROTO_OCTREE_POOL_CAPACITY,
};

/// Persistent prototype cache.
///
/// Octree extents are per-prototype contiguous (the dense spine demands
/// it) — allocated from a bump cursor + a free-list keyed on extent
/// size for re-bake reuse. Brick + leaf-attr extents are NOT tracked
/// per-prototype; the GPU bake bumps global cursors live in
/// [`PrototypeBakePass`], and the bake's overflow check uses the
/// `*_pool_capacity` values stored here.
///
/// Eviction (LRU-style via `begin_frame` + `evict_untouched`) drops
/// entries that weren't touched this frame and returns their octree
/// extent to the free-list. Brick + leaf-attr slots used by an evicted
/// prototype are orphaned in the GPU pool; the cursors don't reclaim.
/// When pools fill, the engine triggers `full_reset` to mark every
/// remaining entry dirty + zero the GPU cursors, restarting from a
/// blank slate.
pub struct PrototypeCache {
    pub(super) entries: HashMap<u32, PrototypeEntry>,
    /// Bump cursor for the octree pool. Free-list reuses extents on
    /// re-bake, so this only grows monotonically when a NEW (shader,
    /// depth) tuple appears.
    pub(super) octree_high_water: u32,
    /// Free extents (offset, size) returned by re-bakes / eviction;
    /// preferred over a fresh bump when an extent of matching size is
    /// available. Linear scan is fine — entries cap at MAX_PROTOTYPES
    /// (256).
    pub(super) octree_free_list: Vec<(u32, u32)>,
    pub(super) pool_octree_base: u32,
    pub(super) pool_brick_base: u32,
    pub(super) pool_leaf_attr_base: u32,
    pub(super) pool_octree_capacity: u32,
    pub(super) pool_brick_capacity: u32,
    pub(super) pool_leaf_attr_capacity: u32,
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
            octree_high_water: 0,
            octree_free_list: Vec::new(),
            pool_octree_base: 0,
            pool_brick_base: 0,
            pool_leaf_attr_base: 0,
            pool_octree_capacity: octree_capacity,
            pool_brick_capacity: brick_capacity,
            pool_leaf_attr_capacity: leaf_attr_capacity,
        }
    }

    /// Configure the GPU offsets where the prototype sub-pool begins.
    /// Coordinated by the engine layer so the prototype range is
    /// disjoint from the per-region transient range.
    ///
    /// Phase 4 — bases now point into the host scene's main pool tails
    /// (not the dedicated proto buffers, which become dead code in
    /// Phase 5). When bases shift (e.g. CPU geometry data grows past
    /// the previous proto base), the cache flushes and every entry
    /// re-bakes at the new base.
    ///
    /// Returns `true` iff the bases changed — the caller must also
    /// reset the GPU cursor buffer to `(brick_base, leaf_attr_base)` so
    /// the next bake's atomic-bumps land at the new offsets.
    pub fn set_pool_bases(
        &mut self,
        pool_octree_base: u32,
        pool_brick_base: u32,
        pool_leaf_attr_base: u32,
    ) -> bool {
        if self.pool_octree_base == pool_octree_base
            && self.pool_brick_base == pool_brick_base
            && self.pool_leaf_attr_base == pool_leaf_attr_base
        {
            return false;
        }
        self.flush();
        self.pool_octree_base = pool_octree_base;
        self.pool_brick_base = pool_brick_base;
        self.pool_leaf_attr_base = pool_leaf_attr_base;
        true
    }

    pub fn pool_octree_base(&self) -> u32 { self.pool_octree_base }
    pub fn pool_brick_base(&self) -> u32 { self.pool_brick_base }
    pub fn pool_leaf_attr_base(&self) -> u32 { self.pool_leaf_attr_base }

    pub fn brick_pool_capacity(&self) -> u32 { self.pool_brick_capacity }
    pub fn leaf_attr_pool_capacity(&self) -> u32 { self.pool_leaf_attr_capacity }

    /// Drop every entry and reset the octree cursor + free-list. The
    /// engine should also zero the GPU brick + leaf-attr cursor buffers
    /// when calling this (the cache doesn't own them).
    pub fn flush(&mut self) {
        self.entries.clear();
        self.octree_high_water = 0;
        self.octree_free_list.clear();
    }

    /// Mark every retained entry dirty (forces re-bake on next lookup)
    /// without freeing extents. Used by [`Self::full_reset`] alongside
    /// the engine zeroing GPU cursors so live prototypes get fresh
    /// brick + leaf-attr slots in the new global pool.
    pub fn dirty_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.source_hash = entry.source_hash.wrapping_add(1);
        }
    }

    /// Mark every entry untouched at the start of a frame.
    pub fn begin_frame(&mut self) {
        for entry in self.entries.values_mut() {
            entry.touched_this_frame = false;
        }
    }

    /// Look up `shader_id` against the cache. Returns `(entry, dirty)`:
    /// dirty=true means the bake compute must run for this entry.
    /// Returns `None` only when the octree pool is exhausted (free-list
    /// empty + cursor would overrun capacity); the caller should log
    /// overflow and proceed without the prototype.
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

        let needed = octree_node_count_for_depth(max_depth);

        if let Some(entry) = self.entries.get_mut(&shader_id) {
            if entry.max_depth == max_depth {
                let dirty = entry.source_hash != source_hash;
                if dirty {
                    entry.source_hash = source_hash;
                }
                entry.touched_this_frame = true;
                return Some((entry.clone(), dirty));
            }
            // Depth changed — return the old extent to the free-list and
            // fall through to fresh alloc.
            let old = entry.octree_extent;
            self.octree_free_list.push(old);
            self.entries.remove(&shader_id);
        }

        let octree_extent = self.alloc_octree(needed)?;

        let entry = PrototypeEntry {
            shader_id,
            source_hash,
            max_depth,
            octree_extent,
            touched_this_frame: true,
        };
        self.entries.insert(shader_id, entry.clone());
        Some((entry, true))
    }

    /// Drop entries not referenced this frame and return their octree
    /// extents to the free-list.
    pub fn evict_untouched(&mut self) {
        let to_remove: Vec<u32> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.touched_this_frame)
            .map(|(k, _)| *k)
            .collect();
        for k in to_remove {
            if let Some(entry) = self.entries.remove(&k) {
                self.octree_free_list.push(entry.octree_extent);
            }
        }
    }

    /// Reuse a same-sized extent from the free-list if one is available;
    /// otherwise bump the cursor. Returns `None` if neither path can
    /// satisfy the request.
    fn alloc_octree(&mut self, size: u32) -> Option<(u32, u32)> {
        if let Some(idx) = self
            .octree_free_list
            .iter()
            .position(|(_, s)| *s == size)
        {
            return Some(self.octree_free_list.swap_remove(idx));
        }
        if self.octree_high_water + size > self.pool_octree_capacity {
            return None;
        }
        let offset = self.octree_high_water;
        self.octree_high_water += size;
        Some((offset, size))
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn get(&self, shader_id: u32) -> Option<&PrototypeEntry> {
        self.entries.get(&shader_id)
    }
    pub fn octree_high_water(&self) -> u32 { self.octree_high_water }
}

impl Default for PrototypeCache {
    fn default() -> Self { Self::new() }
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
) -> Vec<[u32; 4]> {
    let levels = level_starts_inclusive(max_depth);
    let total = levels[max_depth as usize + 1] as usize;
    let block_root = pool_octree_base + octree_block_offset;
    let mut out: Vec<[u32; 4]> = Vec::with_capacity(total);
    // Internal levels 0..max_depth-1: each node is a branch pointing to
    // 8 children at the next level. The trailing two u32 lanes are
    // the per-node tight-AABB slots — zeroed during Step 1 of the
    // rollout; bake/rollup fill them in Step 2.
    for k in 0..max_depth {
        let level_size = 8u32.saturating_pow(k);
        for i in 0..level_size {
            let first_child = block_root + levels[(k + 1) as usize] + i * 8;
            out.push([first_child, INTERNAL_ATTR_NONE, 0u32, 0u32]);
        }
    }
    // Leaf level: bake fills these in. Initialize to EMPTY.
    let leaf_level_size = 8u32.saturating_pow(max_depth);
    for _ in 0..leaf_level_size {
        out.push([OCTREE_EMPTY, INTERNAL_ATTR_NONE, 0u32, 0u32]);
    }
    debug_assert_eq!(out.len(), total);
    out
}
