//! Sparse octree spatial structure for organizing voxels.
//!
//! Replaces the flat [`BrickMap`](crate::brick_map::BrickMap) with a
//! compact tree where uniform regions (all-empty or all-solid) collapse to single
//! nodes. Variable-depth leaves provide built-in LOD — coarser leaves at shallower
//! depths represent larger spatial extents.
//!
//! Each leaf is an individual voxel (no bricks). Octree traversal lands directly
//! on a voxel pool slot — no within-brick indexing, no brick boundaries.
//!
//! # Node encoding
//!
//! Each node is a single `u32`:
//!
//! | Value | Meaning |
//! |-------|---------|
//! | `0xFFFF_FFFF` | **EMPTY** — entire subtree is empty air |
//! | `0xFFFF_FFFE` | **INTERIOR** — entire subtree is fully opaque |
//! | Bit 31 clear | **BRANCH** — value is offset to 8 contiguous child nodes |
//! | Bit 31 set, < `0xFFFF_FFFE` | **LEAF** — `value & 0x7FFF_FFFF` is voxel pool slot |
//!
//! Branch nodes always store 8 contiguous children (no child mask). This trades
//! ~7× overhead for sparse branches in exchange for simple GPU traversal —
//! `children[octant]` is a direct index, no popcount needed.

use glam::UVec3;
use crate::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};

mod insert;
mod mutate;
mod mutation_log;
mod optimize;
mod query;

pub use mutate::{BrickPathCache, CellState, CellStateCache};
pub use mutation_log::OctreeMutationLog;

#[cfg(test)]
mod tests;

/// Sentinel: entire subtree is empty (no geometry).
pub const EMPTY_NODE: u32 = 0xFFFF_FFFF;

/// Sentinel: entire subtree is fully opaque interior.
pub const INTERIOR_NODE: u32 = 0xFFFF_FFFE;

/// Bit flag indicating a leaf-like node (LEAF or BRICK; not a branch and not
/// a sentinel). Both LEAF_BIT and the sentinels share bit 31.
pub const LEAF_BIT: u32 = 0x8000_0000;

/// Bit flag distinguishing BRICK from LEAF (only meaningful when LEAF_BIT is
/// also set). LEAF: `LEAF_BIT | leaf_attr_id`. BRICK: `LEAF_BIT | BRICK_BIT |
/// brick_id`. Sentinels also have both bits set, so brick/sentinel distinction
/// requires checking the sentinel values too.
pub const BRICK_BIT: u32 = 0x4000_0000;

/// Maximum supported octree depth (2^11 = 2048 voxels per axis).
pub const MAX_DEPTH: u8 = 11;

/// Sentinel for [`SparseOctree::internal_attr_index`]: "no prefiltered
/// LOD attr for this node." Meaningful only at slots whose `nodes` value
/// is a branch; for leaf / empty / interior slots the parallel entry is
/// always this sentinel (never read by the shader).
pub const INTERNAL_ATTR_NONE: u32 = 0xFFFF_FFFF;

/// Returns `true` if the node value represents a regular leaf — leaf_attr_id
/// in the low 30 bits. Excludes BRICK references and sentinels.
#[inline]
pub fn is_leaf(node: u32) -> bool {
    (node & LEAF_BIT) != 0
        && (node & BRICK_BIT) == 0
        && node != EMPTY_NODE
        && node != INTERIOR_NODE
}

/// Returns `true` if the node value represents a brick reference — brick_id
/// in the low 30 bits, with BRICK_BIT set.
#[inline]
pub fn is_brick(node: u32) -> bool {
    (node & LEAF_BIT) != 0
        && (node & BRICK_BIT) != 0
        && node != EMPTY_NODE
        && node != INTERIOR_NODE
}

/// Returns `true` if the node value represents a branch (offset to children).
#[inline]
pub fn is_branch(node: u32) -> bool {
    (node & LEAF_BIT) == 0 && node != EMPTY_NODE && node != INTERIOR_NODE
}

/// Returns `true` if the node value is leaf-like (LEAF or BRICK or sentinel)
/// — i.e. a non-branch terminator that traversal stops at.
#[inline]
pub fn is_terminator(node: u32) -> bool {
    !is_branch(node)
}

/// Extract the leaf_attr_id from a regular LEAF node.
#[inline]
pub fn leaf_slot(node: u32) -> u32 {
    debug_assert!(is_leaf(node));
    node & !(LEAF_BIT | BRICK_BIT)
}

/// Extract the brick_id from a BRICK node.
#[inline]
pub fn brick_id(node: u32) -> u32 {
    debug_assert!(is_brick(node));
    node & !(LEAF_BIT | BRICK_BIT)
}

/// Encode a leaf_attr_id as a LEAF node.
#[inline]
pub fn make_leaf(slot: u32) -> u32 {
    debug_assert!(slot < BRICK_BIT, "leaf_attr_id too large for 30-bit leaf encoding");
    slot | LEAF_BIT
}

/// Encode a brick_id as a BRICK node.
#[inline]
pub fn make_brick(id: u32) -> u32 {
    debug_assert!(id < BRICK_BIT, "brick_id too large for 30-bit brick encoding");
    let v = id | LEAF_BIT | BRICK_BIT;
    debug_assert!(v != EMPTY_NODE && v != INTERIOR_NODE,
        "brick_id collides with sentinel encoding");
    v
}

/// Compute the octant index (0–7) for a voxel coordinate at a given level.
///
/// At each level, the coordinate space is halved. The octant is determined by
/// which half the coordinate falls in along each axis.
#[inline]
fn octant_for_coord(coord: UVec3, level: u8, depth: u8) -> u32 {
    // At this level, each child covers `half` bricks per axis.
    let half = 1u32 << (depth - level - 1);
    let x = if coord.x & half != 0 { 1u32 } else { 0 };
    let y = if coord.y & half != 0 { 1u32 } else { 0 };
    let z = if coord.z & half != 0 { 1u32 } else { 0 };
    x + y * 2 + z * 4
}

/// A sparse octree organizing voxel pool slots in 3D space.
///
/// The tree covers a cube of `2^depth` voxels per axis. Each leaf holds a voxel
/// pool slot. Uniform regions collapse to [`EMPTY_NODE`] or [`INTERIOR_NODE`].
#[derive(Debug, Clone)]
pub struct SparseOctree {
    /// Packed node buffer. The root is at index 0.
    nodes: Vec<u32>,
    /// Parallel prefiltered-LOD attr index, same length as `nodes`.
    /// Entry `i` is a `leaf_attr_id` for the prefiltered surface of the
    /// subtree rooted at branch node `i`, or [`INTERNAL_ATTR_NONE`] when
    /// node `i` isn't a branch (or the prefilter pass hasn't run yet).
    /// See the LOD plan: the GPU march uses this to early-exit descent
    /// once the node's projected screen footprint drops below 1 pixel.
    internal_attr_index: Vec<u32>,
    /// Maximum depth (0 = single root node, 8 = 256³ voxels per axis).
    depth: u8,
    /// Voxel size at the finest (deepest) level.
    base_voxel_size: f32,
    /// Active mutation log (when `Some`). The mutation primitives in
    /// [`mutate`] route writes through `record_node_write` /
    /// `record_attr_write`; callers that need a log opt in via
    /// [`begin_mutation_log`] and take it back with
    /// [`take_mutation_log`]. Boxed so the field stays cheap when
    /// unused.
    mutation_log: Option<Box<OctreeMutationLog>>,
}

impl SparseOctree {
    /// Create a new octree with the given depth, initially all EMPTY.
    ///
    /// `depth`: tree depth. The octree covers `2^depth` voxels per axis.
    /// `base_voxel_size`: voxel size at the finest level.
    pub fn new(depth: u8, base_voxel_size: f32) -> Self {
        assert!(depth <= MAX_DEPTH, "depth {depth} exceeds MAX_DEPTH {MAX_DEPTH}");
        Self {
            nodes: vec![EMPTY_NODE],
            internal_attr_index: vec![INTERNAL_ATTR_NONE],
            depth,
            base_voxel_size,
            mutation_log: None,
        }
    }

    /// Create from raw node data (for file loading).
    ///
    /// The nodes must have valid internal structure (branch offsets are 0-based
    /// within the node array). `internal_attr_index` is initialized to
    /// sentinels — callers that load a prefiltered asset (e.g. .rkp v5+)
    /// should follow with [`set_internal_attr_index`](Self::set_internal_attr_index).
    pub fn from_raw(nodes: &[u32], depth: u8, base_voxel_size: f32) -> Self {
        Self {
            internal_attr_index: vec![INTERNAL_ATTR_NONE; nodes.len()],
            nodes: nodes.to_vec(),
            depth,
            base_voxel_size,
            mutation_log: None,
        }
    }

    /// Tree depth (0 = single root node).
    #[inline]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Voxel size at the finest level.
    #[inline]
    pub fn base_voxel_size(&self) -> f32 {
        self.base_voxel_size
    }

    /// Number of voxels per axis at the finest level.
    #[inline]
    pub fn extent(&self) -> u32 {
        1u32 << self.depth
    }

    /// World-space extent of the root node (one axis).
    #[inline]
    pub fn extent_world(&self) -> f32 {
        self.extent() as f32 * self.base_voxel_size
    }

    /// Total number of nodes in the packed buffer.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Access the raw packed node buffer (for GPU upload).
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.nodes
    }

    /// Mutable access to the flat node storage. Used by loaders that need to
    /// rewrite BRICK node ids in place after allocating scene-local bricks.
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        &mut self.nodes
    }

    /// Slice of prefiltered-LOD attr ids, one per node slot (for GPU upload).
    /// Length matches [`as_slice`](Self::as_slice). Entry is [`INTERNAL_ATTR_NONE`]
    /// at slots that aren't branches or haven't been populated by the prefilter.
    #[inline]
    pub fn internal_attr_slice(&self) -> &[u32] {
        &self.internal_attr_index
    }

    /// Get the prefiltered-LOD attr id at `node_idx`. Returns
    /// [`INTERNAL_ATTR_NONE`] when no prefilter is available for that slot.
    #[inline]
    pub fn internal_attr(&self, node_idx: u32) -> u32 {
        self.internal_attr_index[node_idx as usize]
    }

    /// Set the prefiltered-LOD attr id at `node_idx`. Call during the
    /// bottom-up prefilter pass after `compact`/`deduplicate_subtrees`
    /// have stabilized the node buffer — or before any rewriting pass if
    /// you want it to ride along through those passes.
    #[inline]
    pub fn set_internal_attr(&mut self, node_idx: u32, attr_id: u32) {
        self.internal_attr_index[node_idx as usize] = attr_id;
    }

    /// Replace the entire prefilter index buffer. Must have length equal to
    /// [`node_count`](Self::node_count). Used when loading a .rkp asset that
    /// already has prefiltered LOD baked in.
    pub fn set_internal_attr_index(&mut self, index: Vec<u32>) {
        assert_eq!(
            index.len(),
            self.nodes.len(),
            "internal_attr_index length must match nodes length"
        );
        self.internal_attr_index = index;
    }

    /// Start a [`OctreeMutationLog`] capturing every subsequent write
    /// to `nodes[]` and `internal_attr_index[]` made via the
    /// `set_cell_*` primitives. Replaces any prior active log. The
    /// log's `initial_node_count` is snapshotted now so the caller can
    /// detect growth.
    pub fn begin_mutation_log(&mut self) {
        self.mutation_log = Some(Box::new(OctreeMutationLog::new(self.nodes.len() as u32)));
    }

    /// Take the active mutation log, clearing it from the tree. Returns
    /// `None` when no log was active.
    pub fn take_mutation_log(&mut self) -> Option<OctreeMutationLog> {
        self.mutation_log.take().map(|b| *b)
    }

    /// Internal helper used by mutate primitives — record a single node
    /// write if the log is active. The actual write to `self.nodes[i]`
    /// is the caller's responsibility (kept separate to keep the
    /// fast-path non-logging mutation a single store).
    #[inline]
    pub(super) fn record_node_write(&mut self, local_idx: u32, value: u32) {
        if let Some(log) = self.mutation_log.as_mut() {
            log.node_writes.push((local_idx, value));
        }
    }

    /// Internal helper — record a single internal_attr_index write.
    #[inline]
    pub(super) fn record_attr_write(&mut self, local_idx: u32, value: u32) {
        if let Some(log) = self.mutation_log.as_mut() {
            log.attr_writes.push((local_idx, value));
        }
    }

    /// Check if a voxel coordinate is in bounds for this tree.
    #[inline]
    fn in_bounds(&self, coord: UVec3) -> bool {
        let ext = self.extent();
        coord.x < ext && coord.y < ext && coord.z < ext
    }

    /// Build a `SparseOctree` from an existing flat `BrickMap`.
    ///
    /// This is the migration path for loading `.rkf` files: load the flat map,
    /// convert to octree for GPU upload.
    pub fn from_brick_map(map: &BrickMap, base_voxel_size: f32) -> Self {
        // Determine depth: smallest power of 2 that covers all dimensions.
        let max_dim = map.dims.x.max(map.dims.y).max(map.dims.z);
        let depth = if max_dim == 0 {
            0
        } else {
            (32 - (max_dim - 1).leading_zeros()) as u8
        };
        let depth = depth.max(1); // minimum depth 1

        let mut tree = SparseOctree::new(depth, base_voxel_size);

        for bz in 0..map.dims.z {
            for by in 0..map.dims.y {
                for bx in 0..map.dims.x {
                    let entry = map.get(bx, by, bz).unwrap();
                    let coord = UVec3::new(bx, by, bz);
                    match entry {
                        EMPTY_SLOT => {} // skip, tree is already EMPTY
                        INTERIOR_SLOT => tree.insert_interior(coord),
                        slot => tree.insert(coord, slot),
                    }
                }
            }
        }

        tree
    }

    /// Count live (reachable) nodes — excludes dead space from collapsed branches.
    pub fn live_node_count(&self) -> usize {
        self.count_live(0)
    }

    fn count_live(&self, node_idx: usize) -> usize {
        let node = self.nodes[node_idx];
        if !is_branch(node) {
            return 1;
        }
        let children_offset = node as usize;
        1 + (0..8usize).map(|i| self.count_live(children_offset + i)).sum::<usize>()
    }
}

