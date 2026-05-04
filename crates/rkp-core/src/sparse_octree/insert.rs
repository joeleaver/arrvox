//! Insertion + opportunistic-collapse routines on the octree node buffer.

use glam::UVec3;

use super::{
    INTERIOR_NODE, INTERNAL_ATTR_NONE, SparseOctree, is_branch, make_leaf, octant_for_coord,
};

impl SparseOctree {
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
        self.internal_attr_index.extend_from_slice(&[INTERNAL_ATTR_NONE; 8]);
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
        self.internal_attr_index.extend_from_slice(&[INTERNAL_ATTR_NONE; 8]);
        self.nodes[node_idx] = children_offset as u32;

        // Now descend into the correct child.
        let octant = octant_for_coord(coord, level, self.depth) as usize;
        self.insert_at(children_offset + octant, coord, level + 1, value);
        self.try_collapse(node_idx);
    }

    /// Try to collapse a branch node back to a uniform value if all 8 children
    /// are identical leaves/sentinels (not branches).
    ///
    /// The 8 orphaned child slots are not reclaimed here — call [`compact`] to
    /// produce a dense representation for GPU upload.
    ///
    /// [`compact`]: SparseOctree::compact
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
    }

    /// Walk the tree bottom-up and collapse every branch whose 8 children are
    /// all identical (leaves or sentinels).
    ///
    /// `try_collapse` runs opportunistically during `insert`, but if leaf
    /// values are edited after insertion (for example, by remapping slot
    /// indices after a dedup pass on a loaded .rkp) the collapse opportunity
    /// is missed. This pass catches those. It does not reclaim storage —
    /// follow it with [`compact`](Self::compact).
    pub fn collapse_all(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        Self::collapse_recursive(&mut self.nodes, 0);
    }

    fn collapse_recursive(nodes: &mut [u32], idx: usize) {
        let node = nodes[idx];
        if !is_branch(node) {
            return;
        }
        let offset = node as usize;
        for i in 0..8 {
            Self::collapse_recursive(nodes, offset + i);
        }
        // After children are collapsed, check for uniformity.
        let first = nodes[offset];
        if is_branch(first) {
            return;
        }
        for i in 1..8 {
            if nodes[offset + i] != first {
                return;
            }
        }
        nodes[idx] = first;
    }
}
