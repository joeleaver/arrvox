//! Read-only traversal: coordinate lookup + leaf/brick iteration.

use glam::UVec3;

use super::{
    EMPTY_NODE, INTERIOR_NODE, SparseOctree, brick_id, is_branch, is_brick, is_leaf, leaf_slot,
    octant_for_coord,
};

impl SparseOctree {
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

    /// Walk the tree and emit every BRICK node as `(brick_origin, brick_id)`.
    /// `brick_origin` is in finest-voxel grid units — divide by
    /// `BRICK_DIM` (=8) to get brick-grid units. Used by the
    /// skin-deform scatter pass, which needs each brick's object-local
    /// position to forward-skin the voxels it contains.
    pub fn iter_bricks(&self) -> impl Iterator<Item = (UVec3, u32)> + '_ {
        let mut results = Vec::new();
        self.collect_bricks(0, UVec3::ZERO, 0, &mut results);
        results.into_iter()
    }

    fn collect_bricks(
        &self,
        node_idx: usize,
        origin: UVec3,
        level: u8,
        out: &mut Vec<(UVec3, u32)>,
    ) {
        let node = self.nodes[node_idx];
        if node == EMPTY_NODE || node == INTERIOR_NODE {
            return;
        }
        if is_leaf(node) {
            return;
        }
        if is_brick(node) {
            out.push((origin, brick_id(node)));
            return;
        }
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
            self.collect_bricks(children_offset + octant as usize, child_origin, level + 1, out);
        }
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
        if is_brick(node) {
            // BRICKs hold a flat array of leaf_attr_ids covering several
            // voxels at once; iter_leaves can't expand them without
            // BrickPool access. Callers that need to enumerate brick
            // contents should iterate the BrickPool directly. Skip here.
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
}
