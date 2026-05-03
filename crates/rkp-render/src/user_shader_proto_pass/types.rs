//! Constants + types + pure helpers for the prototype bake pipeline.
//!
//! No logic beyond pool sizing math:
//! - Per-prototype octree-depth caps + the depth → node/brick/leaf-attr
//!   counts the cache uses to allocate.
//! - Pool capacity constants (`PROTO_*_POOL_CAPACITY`,
//!   `PROTO_TAIL_*_BYTES`).
//! - [`PrototypeEntry`] — one cached prototype's record (octree extent,
//!   max_depth, source hash).
//!
//! Larger types (`PrototypeUniform`, `PrototypeCache`, `PrototypeBakePass`)
//! live in their respective sibling modules.

use crate::user_shader_pass::BRICK_CELLS;

/// Default prototype octree depth. With depth 2 the prototype is a
/// 16-cell-per-axis cube (4 bricks per axis, 64 max bricks) — enough
/// resolution for grass blades / pebbles / coarse foliage. Authors can
/// override per-shader via `@proto_max_depth`.
pub const DEFAULT_PROTO_MAX_DEPTH: u32 = 2;

/// Hard ceiling on prototype octree depth. With sparse global brick +
/// leaf-attr cursors (no per-prototype worst-case reservation), the
/// real cost of higher depths is the dense octree spine. Cumulative
/// nodes at depth 8 = (8^9 − 1) / 7 ≈ 19.2 M nodes × 8 B = 153 MB per
/// prototype, fits in the dedicated proto octree pool sized for it.
/// Going higher would push spine cost past the budget; for finer
/// resolution than depth 8 gives (~0.4 mm cells on a 0.4 m blade),
/// switch to sparse-spine BFS first.
pub const MAX_PROTO_MAX_DEPTH: u32 = 8;

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
///
/// Carries the OCTREE extent only — bricks and leaf-attrs are allocated
/// at GPU bake time from global cursors, so an entry doesn't reserve
/// brick / leaf-attr range up front. The slots a bake actually writes
/// are referenced indirectly via the octree's leaf-level brick_id
/// pointers, so per-prototype layout is unnecessary.
#[derive(Debug, Clone)]
pub struct PrototypeEntry {
    pub shader_id: u32,
    pub source_hash: u64,
    pub max_depth: u32,
    /// `(offset, size)` extent in the octree pool. Offset is RELATIVE
    /// to `pool_octree_base` — add it to get an absolute GPU index.
    /// Size is exactly `octree_node_count_for_depth(max_depth)`.
    pub octree_extent: (u32, u32),
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

/// Default octree-pool capacity, in nodes. Sized to fit a handful of
/// depth-8 prototypes (each spine ~19.2 M nodes) plus headroom for the
/// usual mix of smaller depths. 32 M nodes × 8 B = 256 MB.
pub const PROTO_OCTREE_POOL_CAPACITY: u32 = 32 * 1024 * 1024;

/// Default brick-pool capacity, in bricks (a brick = 64 cells × 4 B).
/// A depth-8 sparse blade is ~18 K bricks; 256 K bricks = 64 MB =
/// headroom for ~14 such prototypes baked simultaneously. Bricks are
/// allocated globally at GPU bake time; the cap exists for overflow
/// gating, not per-prototype reservation.
pub const PROTO_BRICK_POOL_CAPACITY: u32 = 256 * 1024;

/// Default leaf-attr-pool capacity, in slots (each slot = 8 B).
/// A depth-8 sparse blade is ~1.2 M slots; 4 M slots = 32 MB = a few
/// depth-8 prototypes' worth, more if shallower. Leaf-attrs are
/// allocated globally at GPU bake time.
pub const PROTO_LEAF_ATTR_POOL_CAPACITY: u32 = 4 * 1024 * 1024;

/// Phase 4 — proto-tail reservation in the host scene buffers.
///
/// Sized tight enough that adding `cpu_*_bytes + proto_tail + Phase C
/// transient` doesn't breach `max_storage_buffer_binding_size` (1 GB).
/// Phase C's transients are huge (~768 MB on the brick buffer at
/// MAX_GLOBAL_BRICKS = 3M), so the proto tail must be small.
///
/// Capacity for a typical project: a few `@instance_proto` shaders at
/// depth 4-6 (grass, foliage, pebbles). Depth 6 sparse = ~300 K octree
/// nodes / ~10 K bricks / ~80 K leaf-attrs per shader. The reservations
/// below comfortably hold ~5 such shaders simultaneously.
///
/// Depth-8 prototypes overflow these — that's intentional. The dedicated
/// proto buffers (PROTO_*_CAPACITY_BYTES) sized for depth-8 worst-case
/// are dead code post-Phase 5; proto data now lives in the host pools.
/// Authors needing depth-8 should instead reduce paint area or cell_size.
///
/// 2 M octree nodes × 8 B = 16 MB.
pub const PROTO_TAIL_OCTREE_BYTES: u64 = 16 * 1024 * 1024;
/// 64 K bricks × 64 cells × 4 B = 16 MB.
pub const PROTO_TAIL_BRICK_BYTES: u64 = 16 * 1024 * 1024;
/// 1 M leaf-attr slots × 8 B = 8 MB.
pub const PROTO_TAIL_LEAF_ATTR_BYTES: u64 = 8 * 1024 * 1024;

/// Constants mirrored from `user_shader_proto.wgsl`. Kept in Rust so
/// the CPU pre-builder doesn't have to read the WGSL.
pub const OCTREE_EMPTY: u32 = 0xFFFFFFFFu32;
pub const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu32;
