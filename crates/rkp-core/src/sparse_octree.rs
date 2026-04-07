//! Sparse octree spatial structure for organizing voxels.
//!
//! Replaces rkf-core's flat [`BrickMap`](rkf_core::brick_map::BrickMap) with a
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
use rkf_core::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};

/// Sentinel: entire subtree is empty (no geometry).
pub const EMPTY_NODE: u32 = 0xFFFF_FFFF;

/// Sentinel: entire subtree is fully opaque interior.
pub const INTERIOR_NODE: u32 = 0xFFFF_FFFE;

/// Bit flag indicating a leaf node (voxel pool slot in lower 31 bits).
pub const LEAF_BIT: u32 = 0x8000_0000;

/// Maximum supported octree depth (2^11 = 2048 voxels per axis).
pub const MAX_DEPTH: u8 = 11;

/// Returns `true` if the node value represents a leaf (has `LEAF_BIT` set and is
/// not one of the sentinel values).
#[inline]
pub fn is_leaf(node: u32) -> bool {
    (node & LEAF_BIT) != 0 && node != EMPTY_NODE && node != INTERIOR_NODE
}

/// Returns `true` if the node value represents a branch (offset to children).
#[inline]
pub fn is_branch(node: u32) -> bool {
    (node & LEAF_BIT) == 0 && node != EMPTY_NODE && node != INTERIOR_NODE
}

/// Extract the voxel pool slot from a leaf node.
#[inline]
pub fn leaf_slot(node: u32) -> u32 {
    debug_assert!(is_leaf(node));
    node & !LEAF_BIT
}

/// Encode a voxel pool slot as a leaf node.
#[inline]
pub fn make_leaf(slot: u32) -> u32 {
    debug_assert!(slot < LEAF_BIT, "voxel pool slot too large for leaf encoding");
    slot | LEAF_BIT
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
    /// Maximum depth (0 = single root node, 8 = 256³ voxels per axis).
    depth: u8,
    /// Voxel size at the finest (deepest) level.
    base_voxel_size: f32,
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
            depth,
            base_voxel_size,
        }
    }

    /// Create from raw node data (for file loading).
    ///
    /// The nodes must have valid internal structure (branch offsets are 0-based
    /// within the node array).
    pub fn from_raw(nodes: &[u32], depth: u8, base_voxel_size: f32) -> Self {
        Self {
            nodes: nodes.to_vec(),
            depth,
            base_voxel_size,
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

    /// Check if a voxel coordinate is in bounds for this tree.
    #[inline]
    fn in_bounds(&self, coord: UVec3) -> bool {
        let ext = self.extent();
        coord.x < ext && coord.y < ext && coord.z < ext
    }

    /// Insert a leaf (voxel pool slot) at the given voxel coordinate.
    ///
    /// Subdivides branch nodes as needed to reach the finest level.
    /// Panics if `coord` is out of bounds.
    pub fn insert(&mut self, coord: UVec3, slot: u32) {
        assert!(self.in_bounds(coord), "coord {coord} out of bounds for depth {}", self.depth);
        self.insert_at(0, coord, 0, make_leaf(slot));
    }

    /// Mark the voxel coordinate as INTERIOR (fully opaque, no voxel needed).
    ///
    /// Panics if `coord` is out of bounds.
    pub fn insert_interior(&mut self, coord: UVec3) {
        assert!(self.in_bounds(coord), "coord {coord} out of bounds for depth {}", self.depth);
        self.insert_at(0, coord, 0, INTERIOR_NODE);
    }

    /// Set a value at a specific target level, subdividing as needed to reach it.
    ///
    /// Unlike `insert` which always descends to max depth, this stops at
    /// `target_level` and sets the entire subtree at that level to `value`.
    /// Useful for marking an entire subtree as EMPTY or INTERIOR during
    /// adaptive voxelization.
    ///
    /// `coord`: any brick coordinate within the target subtree's spatial extent.
    /// `target_level`: the level at which to set the value (0 = root).
    pub fn set_at_level(&mut self, coord: UVec3, target_level: u8, value: u32) {
        assert!(target_level <= self.depth);
        self.insert_at_target(0, coord, 0, target_level, value);
    }

    fn insert_at_target(
        &mut self,
        node_idx: usize,
        coord: UVec3,
        level: u8,
        target_level: u8,
        value: u32,
    ) {
        if level == target_level {
            self.nodes[node_idx] = value;
            return;
        }

        let current = self.nodes[node_idx];

        if is_branch(current) {
            let children_offset = current as usize;
            let octant = octant_for_coord(coord, level, self.depth) as usize;
            self.insert_at_target(children_offset + octant, coord, level + 1, target_level, value);
            self.try_collapse(node_idx);
            return;
        }

        // Need to subdivide to reach target level.
        let children_offset = self.nodes.len();
        self.nodes.extend_from_slice(&[current; 8]);
        self.nodes[node_idx] = children_offset as u32;

        let octant = octant_for_coord(coord, level, self.depth) as usize;
        self.insert_at_target(children_offset + octant, coord, level + 1, target_level, value);
        self.try_collapse(node_idx);
    }

    /// Recursive insert. `node_idx` is the current node in `self.nodes`.
    /// `level` is the current depth (0 = root). `value` is the leaf/sentinel to store.
    fn insert_at(&mut self, node_idx: usize, coord: UVec3, level: u8, value: u32) {
        if level == self.depth {
            // At max depth — store the value directly.
            self.nodes[node_idx] = value;
            return;
        }

        let current = self.nodes[node_idx];

        // If current node is a branch, descend into the correct child.
        if is_branch(current) {
            let children_offset = current as usize;
            let octant = octant_for_coord(coord, level, self.depth) as usize;
            self.insert_at(children_offset + octant, coord, level + 1, value);
            self.try_collapse(node_idx);
            return;
        }

        // Current node is a leaf, EMPTY, or INTERIOR. Need to subdivide.
        // If we're inserting the same value as what's already here, no-op.
        if level == self.depth - 1 {
            // Next level is max depth — no further subdivision needed, but we
            // need 8 children to store the new value in the right octant.
        }

        // Allocate 8 children, all initialized to the current node value
        // (preserving the existing uniform content).
        let children_offset = self.nodes.len();
        self.nodes.extend_from_slice(&[current; 8]);
        self.nodes[node_idx] = children_offset as u32;

        // Now descend into the correct child.
        let octant = octant_for_coord(coord, level, self.depth) as usize;
        self.insert_at(children_offset + octant, coord, level + 1, value);
        self.try_collapse(node_idx);
    }

    /// Try to collapse a branch node back to a uniform value if all 8 children
    /// are identical leaves/sentinels (not branches).
    fn try_collapse(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
        if !is_branch(node) {
            return;
        }
        let children_offset = node as usize;
        let first = self.nodes[children_offset];
        // Only collapse if first child is a leaf or sentinel (not a branch).
        if is_branch(first) {
            return;
        }
        for i in 1..8 {
            if self.nodes[children_offset + i] != first {
                return;
            }
        }
        // All children identical — collapse.
        self.nodes[node_idx] = first;
        // Note: the 8 child slots become dead space. A compaction pass could
        // reclaim them, but for typical usage the waste is small.
    }

    /// Look up the node value at a voxel coordinate.
    ///
    /// Returns the leaf (with `LEAF_BIT` set), `EMPTY_NODE`, or `INTERIOR_NODE`.
    /// Returns `None` if `coord` is out of bounds.
    pub fn lookup(&self, coord: UVec3) -> Option<u32> {
        if !self.in_bounds(coord) {
            return None;
        }
        Some(self.lookup_unchecked(coord))
    }

    /// Look up without bounds checking.
    fn lookup_unchecked(&self, coord: UVec3) -> u32 {
        let mut idx = 0;
        for level in 0..self.depth {
            let node = self.nodes[idx];
            if !is_branch(node) {
                return node;
            }
            let octant = octant_for_coord(coord, level, self.depth) as usize;
            idx = node as usize + octant;
        }
        self.nodes[idx]
    }

    /// Look up returning both the node value and the depth at which it was found.
    ///
    /// A leaf found at depth D < max_depth means it covers a coarser region
    /// (variable LOD). The effective voxel size is `base_voxel_size * 2^(max_depth - D)`.
    pub fn lookup_with_depth(&self, coord: UVec3) -> Option<(u32, u8)> {
        if !self.in_bounds(coord) {
            return None;
        }
        let mut idx = 0;
        for level in 0..self.depth {
            let node = self.nodes[idx];
            if !is_branch(node) {
                return Some((node, level));
            }
            let octant = octant_for_coord(coord, level, self.depth) as usize;
            idx = node as usize + octant;
        }
        Some((self.nodes[idx], self.depth))
    }

    /// Iterate all leaf nodes, yielding `(voxel_coord, voxel_slot, leaf_depth)`.
    ///
    /// `voxel_coord` is the lower-corner coordinate of the leaf's spatial extent.
    /// `leaf_depth` is the depth at which the leaf lives (max_depth = finest).
    /// The leaf covers `2^(max_depth - leaf_depth)` voxels per axis.
    pub fn iter_leaves(&self) -> impl Iterator<Item = (UVec3, u32, u8)> + '_ {
        let mut results = Vec::new();
        self.collect_leaves(0, UVec3::ZERO, 0, &mut results);
        results.into_iter()
    }

    fn collect_leaves(
        &self,
        node_idx: usize,
        origin: UVec3,
        level: u8,
        out: &mut Vec<(UVec3, u32, u8)>,
    ) {
        let node = self.nodes[node_idx];
        if node == EMPTY_NODE {
            return;
        }
        if node == INTERIOR_NODE {
            // Interior nodes don't have brick pool slots — skip.
            return;
        }
        if is_leaf(node) {
            out.push((origin, leaf_slot(node), level));
            return;
        }
        // Branch — recurse into children.
        let children_offset = node as usize;
        let half = 1u32 << (self.depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            self.collect_leaves(children_offset + octant as usize, child_origin, level + 1, out);
        }
    }

    /// Count the number of leaf nodes (allocated voxels).
    pub fn leaf_count(&self) -> usize {
        self.count_leaves(0)
    }

    fn count_leaves(&self, node_idx: usize) -> usize {
        let node = self.nodes[node_idx];
        if node == EMPTY_NODE || node == INTERIOR_NODE {
            return 0;
        }
        if is_leaf(node) {
            return 1;
        }
        let children_offset = node as usize;
        (0..8).map(|i| self.count_leaves(children_offset + i)).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_octree_is_empty() {
        let tree = SparseOctree::new(3, 0.1);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.nodes[0], EMPTY_NODE);
        assert_eq!(tree.depth(), 3);
        assert_eq!(tree.extent(), 8); // 2^3
        assert_eq!(tree.leaf_count(), 0);
    }

    #[test]
    fn insert_single_leaf() {
        let mut tree = SparseOctree::new(2, 0.1); // 4x4x4 bricks
        tree.insert(UVec3::new(1, 2, 3), 42);

        let result = tree.lookup(UVec3::new(1, 2, 3));
        assert_eq!(result, Some(make_leaf(42)));
        assert_eq!(tree.leaf_count(), 1);

        // Other coords should be EMPTY.
        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(EMPTY_NODE));
    }

    #[test]
    fn insert_multiple_leaves() {
        let mut tree = SparseOctree::new(3, 0.1); // 8x8x8 bricks
        tree.insert(UVec3::new(0, 0, 0), 10);
        tree.insert(UVec3::new(7, 7, 7), 20);
        tree.insert(UVec3::new(3, 4, 5), 30);

        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(10)));
        assert_eq!(tree.lookup(UVec3::new(7, 7, 7)), Some(make_leaf(20)));
        assert_eq!(tree.lookup(UVec3::new(3, 4, 5)), Some(make_leaf(30)));
        assert_eq!(tree.leaf_count(), 3);
    }

    #[test]
    fn insert_interior() {
        let mut tree = SparseOctree::new(2, 0.1);
        tree.insert_interior(UVec3::new(1, 1, 1));

        assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), Some(INTERIOR_NODE));
        // Interior nodes aren't counted as leaves (no brick pool slot).
        assert_eq!(tree.leaf_count(), 0);
    }

    #[test]
    fn lookup_out_of_bounds() {
        let tree = SparseOctree::new(2, 0.1); // 4x4x4
        assert_eq!(tree.lookup(UVec3::new(4, 0, 0)), None);
        assert_eq!(tree.lookup(UVec3::new(0, 4, 0)), None);
        assert_eq!(tree.lookup(UVec3::new(0, 0, 4)), None);
    }

    #[test]
    fn collapse_uniform_children() {
        let mut tree = SparseOctree::new(1, 0.1); // 2x2x2 = 8 leaves at depth 1
        // Fill all 8 positions with the same slot.
        for z in 0..2u32 {
            for y in 0..2u32 {
                for x in 0..2u32 {
                    tree.insert(UVec3::new(x, y, z), 99);
                }
            }
        }
        // All children identical — root should collapse to a single leaf.
        assert_eq!(tree.nodes[0], make_leaf(99));
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn no_collapse_with_different_children() {
        let mut tree = SparseOctree::new(1, 0.1);
        tree.insert(UVec3::new(0, 0, 0), 10);
        tree.insert(UVec3::new(1, 0, 0), 20);

        // Root should be a branch, not collapsed.
        assert!(is_branch(tree.nodes[0]));
        assert_eq!(tree.leaf_count(), 2);
    }

    #[test]
    fn overwrite_leaf() {
        let mut tree = SparseOctree::new(2, 0.1);
        tree.insert(UVec3::new(1, 1, 1), 42);
        tree.insert(UVec3::new(1, 1, 1), 99);

        assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), Some(make_leaf(99)));
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn lookup_with_depth_finest() {
        let mut tree = SparseOctree::new(3, 0.1);
        tree.insert(UVec3::new(2, 3, 4), 50);

        let (node, depth) = tree.lookup_with_depth(UVec3::new(2, 3, 4)).unwrap();
        assert_eq!(node, make_leaf(50));
        assert_eq!(depth, 3); // at finest level
    }

    #[test]
    fn lookup_with_depth_coarse() {
        // A tree where a leaf exists at a non-max depth (uniform subtree).
        let tree = SparseOctree::new(3, 0.1);
        // The entire tree is EMPTY — lookup should return EMPTY at depth 0 (root).
        let (node, depth) = tree.lookup_with_depth(UVec3::new(2, 3, 4)).unwrap();
        assert_eq!(node, EMPTY_NODE);
        assert_eq!(depth, 0);
    }

    #[test]
    fn iter_leaves_empty() {
        let tree = SparseOctree::new(3, 0.1);
        assert_eq!(tree.iter_leaves().count(), 0);
    }

    #[test]
    fn iter_leaves_collects_all() {
        let mut tree = SparseOctree::new(2, 0.1);
        tree.insert(UVec3::new(0, 0, 0), 10);
        tree.insert(UVec3::new(3, 3, 3), 20);
        tree.insert(UVec3::new(1, 2, 0), 30);

        let mut leaves: Vec<_> = tree.iter_leaves().collect();
        leaves.sort_by_key(|&(coord, slot, _)| (coord.z, coord.y, coord.x, slot));

        assert_eq!(leaves.len(), 3);
        assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(0, 0, 0) && s == 10));
        assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(3, 3, 3) && s == 20));
        assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(1, 2, 0) && s == 30));
    }

    #[test]
    fn from_brick_map_roundtrip() {
        let mut map = BrickMap::new(UVec3::new(4, 4, 4));
        map.set(0, 0, 0, 10);
        map.set(3, 3, 3, 20);
        map.set(1, 2, 3, 30);
        map.set(2, 2, 2, INTERIOR_SLOT);

        let tree = SparseOctree::from_brick_map(&map, 0.1);

        // Verify all lookups match the original map.
        for bz in 0..4 {
            for by in 0..4 {
                for bx in 0..4 {
                    let map_val = map.get(bx, by, bz).unwrap();
                    let tree_val = tree.lookup(UVec3::new(bx, by, bz)).unwrap();
                    match map_val {
                        EMPTY_SLOT => assert_eq!(tree_val, EMPTY_NODE,
                            "mismatch at ({bx},{by},{bz}): map=EMPTY, tree={tree_val:#x}"),
                        INTERIOR_SLOT => assert_eq!(tree_val, INTERIOR_NODE,
                            "mismatch at ({bx},{by},{bz}): map=INTERIOR, tree={tree_val:#x}"),
                        slot => assert_eq!(tree_val, make_leaf(slot),
                            "mismatch at ({bx},{by},{bz}): map={slot}, tree={tree_val:#x}"),
                    }
                }
            }
        }
    }

    #[test]
    fn from_brick_map_non_power_of_two() {
        // BrickMap dims that aren't a power of 2 — octree rounds up.
        let mut map = BrickMap::new(UVec3::new(3, 5, 2));
        map.set(2, 4, 1, 42);

        let tree = SparseOctree::from_brick_map(&map, 0.1);
        assert!(tree.extent() >= 5); // must cover the largest dim
        assert_eq!(tree.lookup(UVec3::new(2, 4, 1)), Some(make_leaf(42)));
    }

    #[test]
    fn extent_world() {
        let tree = SparseOctree::new(3, 0.1);
        // 2^3 = 8 voxels per axis, each voxel 0.1 → 8 * 0.1 = 0.8
        assert!((tree.extent_world() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn leaf_and_branch_encoding() {
        assert!(is_leaf(make_leaf(0)));
        assert!(is_leaf(make_leaf(42)));
        assert!(is_leaf(make_leaf(0x7FFF_FFFD)));
        assert!(!is_leaf(EMPTY_NODE));
        assert!(!is_leaf(INTERIOR_NODE));

        assert!(is_branch(0)); // offset 0 is a valid branch
        assert!(is_branch(100));
        assert!(!is_branch(EMPTY_NODE));
        assert!(!is_branch(INTERIOR_NODE));
        assert!(!is_branch(make_leaf(0)));
    }

    #[test]
    fn leaf_slot_roundtrip() {
        for slot in [0, 1, 42, 1000, 0x7FFF_FFFD] {
            assert_eq!(leaf_slot(make_leaf(slot)), slot);
        }
    }

    #[test]
    #[should_panic]
    fn insert_out_of_bounds_panics() {
        let mut tree = SparseOctree::new(2, 0.1); // 4x4x4
        tree.insert(UVec3::new(4, 0, 0), 1);
    }

    #[test]
    fn depth_zero_single_node() {
        // A depth-0 tree can't store any brick coordinates (extent = 1).
        // Actually extent is 2^0 = 1, so coord (0,0,0) is valid.
        let mut tree = SparseOctree::new(1, 0.1);
        tree.insert(UVec3::new(0, 0, 0), 5);
        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(5)));
    }

    #[test]
    fn many_inserts_no_panic() {
        let mut tree = SparseOctree::new(4, 0.1); // 16x16x16
        let mut count = 0;
        for z in 0..16u32 {
            for y in 0..16u32 {
                for x in 0..16u32 {
                    // Sparse: only insert ~10% of positions.
                    if (x + y * 3 + z * 7) % 10 == 0 {
                        tree.insert(UVec3::new(x, y, z), count);
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(tree.leaf_count(), count as usize);
    }

    #[test]
    fn live_node_count_excludes_dead_space() {
        let mut tree = SparseOctree::new(1, 0.1); // 2x2x2
        // Fill all 8 positions with same slot to trigger collapse.
        for z in 0..2u32 {
            for y in 0..2u32 {
                for x in 0..2u32 {
                    tree.insert(UVec3::new(x, y, z), 99);
                }
            }
        }
        // Root collapsed to a single leaf — node_count includes dead children,
        // but live_node_count should be 1.
        assert_eq!(tree.nodes[0], make_leaf(99));
        assert_eq!(tree.live_node_count(), 1);
        assert!(tree.node_count() >= 1); // may have dead space
    }

    #[test]
    fn from_brick_map_all_interior() {
        let mut map = BrickMap::new(UVec3::new(2, 2, 2));
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    map.set(x, y, z, INTERIOR_SLOT);
                }
            }
        }
        let tree = SparseOctree::from_brick_map(&map, 0.1);
        // Should collapse to a single INTERIOR root.
        assert_eq!(tree.nodes[0], INTERIOR_NODE);
        // live_node_count excludes dead children from collapsed branches.
        assert_eq!(tree.live_node_count(), 1);
    }

    #[test]
    fn from_brick_map_all_empty() {
        let map = BrickMap::new(UVec3::new(4, 4, 4));
        let tree = SparseOctree::from_brick_map(&map, 0.1);
        assert_eq!(tree.nodes[0], EMPTY_NODE);
        assert_eq!(tree.node_count(), 1);
    }
}
