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

    /// Mutable access to the flat node storage. Used by loaders that need to
    /// rewrite BRICK node ids in place after allocating scene-local bricks.
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        &mut self.nodes
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

    /// Deduplicate identical subtrees in place, converting the octree into a
    /// sparse voxel DAG.
    ///
    /// Two branches whose 8 children resolve to the same canonical values
    /// share a single copy of their 8-child block — the duplicate branch's
    /// parent is rewritten to point at the canonical offset. Applied bottom-up
    /// so sharing cascades: deep subtrees merge first, which makes the next
    /// level up more likely to find matches, and so on.
    ///
    /// For geometry with any repetition — a cube's 6 identical face gradients,
    /// a procedural tree, tiled patterns — the storage savings are typically
    /// 10–1000×. The shader sees an ordinary offset-indexed octree; it has no
    /// idea some offsets are referenced by multiple parents.
    ///
    /// Produces a compact output directly (no orphans), so no subsequent
    /// `compact()` call is needed. Safe to call on any tree; trivially
    /// returns for leaf-only and empty roots.
    ///
    /// ## Correctness with shared subtrees
    ///
    /// All iteration in this module is path-based — `iter_leaves` computes
    /// coord from the parent traversal, not from the node's buffer index —
    /// so a shared subtree is correctly visited once per reference, yielding
    /// distinct coords on each visit. `try_collapse` and `insert` were
    /// designed for pre-dedup trees; do not call them after this pass.
    pub fn deduplicate_subtrees(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let root = self.nodes[0];
        if !is_branch(root) {
            // Trivial root (leaf / empty / interior). Nothing to share; also
            // reclaim any orphan tail the builder left behind.
            self.nodes.truncate(1);
            return;
        }

        let mut new_nodes: Vec<u32> = Vec::new();
        new_nodes.push(0); // reserve position [0] for root's value

        let mut seen: std::collections::HashMap<[u32; 8], u32> =
            std::collections::HashMap::new();

        let canonical_root = Self::dedup_value(&self.nodes, root, &mut new_nodes, &mut seen);
        new_nodes[0] = canonical_root;

        self.nodes = new_nodes;
    }

    /// Reorder nodes into BFS/Morton order so that cousins of the same depth
    /// sit contiguously in memory. Within a level, children of sibling
    /// branches are placed next to each other — a depth-N descent across a
    /// warp of pixels lands in a compact byte range instead of scattered
    /// blocks left behind by the depth-first builder.
    ///
    /// DAG sharing from `deduplicate_subtrees` is preserved: if two parents
    /// reference the same old children block, they keep referencing the same
    /// new block after reorder. This is tracked by a map from
    /// `old_children_offset → new_children_offset`.
    ///
    /// Typical effect: for a sphere with ~7.8M reachable nodes, warp-level
    /// cache hit rate at mid-depths improves because sibling subtrees share
    /// cache lines instead of straddling thousands of nodes apart.
    ///
    /// Must be called after `compact` and `deduplicate_subtrees`. Does not
    /// change the set of reachable nodes, only their buffer positions.
    pub fn morton_reorder(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let root = self.nodes[0];
        if !is_branch(root) {
            // Trivial root (leaf / empty / interior) — nothing to reorder.
            self.nodes.truncate(1);
            return;
        }

        let old = std::mem::take(&mut self.nodes);
        let mut new_nodes: Vec<u32> = Vec::with_capacity(old.len());

        // Root lives at new offset 0. Reserve its slot; the branch loop below
        // will write the correct offset to its 8 children.
        new_nodes.push(0);

        // Map: old children-block offset → new children-block offset. Ensures
        // DAG-shared subtrees remain shared after reorder.
        let mut branch_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

        // Queue entries are (old_offset, new_offset) of a single node that
        // needs its value written into `new_nodes[new_offset]`. The BFS order
        // of queue processing is what gives us Morton-like contiguity.
        let mut queue: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();
        queue.push_back((0u32, 0u32));

        while let Some((old_off, new_off)) = queue.pop_front() {
            let node = old[old_off as usize];

            if !is_branch(node) {
                new_nodes[new_off as usize] = node;
                continue;
            }

            let old_children = node;
            let new_children = if let Some(&existing) = branch_map.get(&old_children) {
                // This children block was already allocated by an earlier
                // parent (DAG share). Reuse the same target block — the
                // child nodes inside it are (being) filled in by that
                // earlier traversal, nothing more to do here.
                existing
            } else {
                let start = new_nodes.len() as u32;
                // Reserve 8 slots contiguously; children enqueue below will
                // fill them.
                new_nodes.extend(std::iter::repeat(0u32).take(8));
                branch_map.insert(old_children, start);
                for i in 0..8u32 {
                    queue.push_back((old_children + i, start + i));
                }
                start
            };

            new_nodes[new_off as usize] = new_children;
        }

        self.nodes = new_nodes;
    }

    /// Canonicalize a single node value. Leaves and sentinels pass through
    /// unchanged; branches are expanded, their 8 children are recursively
    /// canonicalized, and the resulting 8-child block is inserted into (or
    /// fetched from) the dedup map.
    fn dedup_value(
        old_nodes: &[u32],
        node_value: u32,
        new_nodes: &mut Vec<u32>,
        seen: &mut std::collections::HashMap<[u32; 8], u32>,
    ) -> u32 {
        if !is_branch(node_value) {
            return node_value;
        }
        let children_offset = node_value as usize;
        let mut canonical_children: [u32; 8] = [0; 8];
        for i in 0..8 {
            let child = old_nodes[children_offset + i];
            canonical_children[i] = Self::dedup_value(old_nodes, child, new_nodes, seen);
        }
        if let Some(&existing) = seen.get(&canonical_children) {
            return existing;
        }
        let new_offset = new_nodes.len() as u32;
        new_nodes.extend_from_slice(&canonical_children);
        seen.insert(canonical_children, new_offset);
        new_offset
    }

    /// Rebuild the node buffer keeping only nodes reachable from the root.
    ///
    /// During construction, [`try_collapse`] merges uniform 8-child subtrees
    /// into a single value at the parent, but the 8 orphaned child slots stay
    /// in the buffer as dead weight. For a large collapsed tree that waste
    /// compounds — every call to `extend_from_slice` in `insert_at` was only
    /// ever going to produce 1-byte storage for 32 bytes of allocation in the
    /// worst case.
    ///
    /// This pass walks the reachable subtree, copying it into a fresh buffer
    /// and rewriting branch offsets as it goes. After this, `node_count()`
    /// equals the number of nodes that GPU traversal could actually reach.
    pub fn compact(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let root = self.nodes[0];

        // Trivial roots (leaf / empty / interior) don't reference anything
        // else — drop the whole tail.
        if !is_branch(root) {
            self.nodes.truncate(1);
            return;
        }

        let mut new_nodes: Vec<u32> = Vec::with_capacity(self.nodes.len());
        new_nodes.push(0); // placeholder for root; filled in below

        // Work queue holds pairs (old_node_value, new_slot_idx). For each
        // entry, we read the node at `old_node_value` from the old buffer;
        // if it's a branch we allocate 8 new children slots and enqueue them.
        //
        // The root is handled specially since its value lives at `nodes[0]`.
        let mut queue: std::collections::VecDeque<(u32, u32)> =
            std::collections::VecDeque::new();

        // Allocate 8 slots for the root's children in the new buffer.
        let root_children_new = new_nodes.len() as u32;
        new_nodes.extend(std::iter::repeat(0u32).take(8));
        new_nodes[0] = root_children_new;
        let root_children_old = root;
        for i in 0..8u32 {
            queue.push_back((root_children_old + i, root_children_new + i));
        }

        while let Some((old_idx, new_idx)) = queue.pop_front() {
            let node = self.nodes[old_idx as usize];
            if is_branch(node) {
                // Allocate 8 new children slots.
                let children_new = new_nodes.len() as u32;
                new_nodes.extend(std::iter::repeat(0u32).take(8));
                new_nodes[new_idx as usize] = children_new;
                let children_old = node;
                for i in 0..8u32 {
                    queue.push_back((children_old + i, children_new + i));
                }
            } else {
                // Leaf or sentinel — copy verbatim, no children to follow.
                new_nodes[new_idx as usize] = node;
            }
        }

        self.nodes = new_nodes;
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
    fn compact_drops_orphan_slots_after_collapse() {
        // Build a tree that will have orphaned slots post-collapse, then
        // compact and verify the buffer shrinks but lookups still work.
        let mut tree = SparseOctree::new(2, 0.1); // 4x4x4
        for z in 0..4u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    tree.insert(UVec3::new(x, y, z), 42);
                }
            }
        }
        // Fully uniform — should have collapsed to a single LEAF at the root,
        // but the intermediate branch allocations are still in `nodes`.
        assert_eq!(tree.nodes[0], make_leaf(42));
        assert!(tree.node_count() > 1, "should have orphaned slots before compact");

        tree.compact();
        // Only the root remains.
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.nodes[0], make_leaf(42));

        // Lookups still work.
        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(42)));
        assert_eq!(tree.lookup(UVec3::new(3, 3, 3)), Some(make_leaf(42)));
    }

    #[test]
    fn compact_preserves_tree_with_no_orphans() {
        // A tree with distinct children per octant has nothing to collapse.
        // compact() should produce a buffer of the same shape.
        let mut tree = SparseOctree::new(1, 0.1);
        tree.insert(UVec3::new(0, 0, 0), 10);
        tree.insert(UVec3::new(1, 1, 1), 20);

        let before_count = tree.node_count();
        let before_lookup_000 = tree.lookup(UVec3::new(0, 0, 0));
        let before_lookup_111 = tree.lookup(UVec3::new(1, 1, 1));

        tree.compact();

        // Same number of reachable nodes (nothing to reclaim).
        assert_eq!(tree.node_count(), before_count);
        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), before_lookup_000);
        assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), before_lookup_111);
    }

    #[test]
    fn deduplicate_shares_identical_subtrees() {
        // Build a depth-2 tree where each of the root's 8 children is an
        // identical branch: a branch whose 8 leaves all point to slot 99.
        // After dedup, those 8 parent-branches all reference the same 8-leaf
        // block, AND that block gets collapsed into a single LEAF by
        // try_collapse (so the tree is actually just a single LEAF at the
        // root after `insert` fires collapse).
        //
        // To specifically exercise DAG sharing (subtrees that don't themselves
        // collapse), build a non-uniform child and place it at the same
        // octant in every root-child.
        let mut tree = SparseOctree::new(2, 0.1); // 4x4x4

        // Fill octant 0 of each of the root's 8 quadrants with slot 7,
        // others with slot 11. So each of the 8 root-child branches has the
        // same internal structure — but because it's non-uniform, the branch
        // itself can't collapse into a single leaf.
        for root_oct in 0..8u32 {
            let dx = root_oct & 1;
            let dy = (root_oct >> 1) & 1;
            let dz = (root_oct >> 2) & 1;
            let base = UVec3::new(dx * 2, dy * 2, dz * 2);
            for inner_oct in 0..8u32 {
                let ix = inner_oct & 1;
                let iy = (inner_oct >> 1) & 1;
                let iz = (inner_oct >> 2) & 1;
                let coord = UVec3::new(base.x + ix, base.y + iy, base.z + iz);
                let slot = if inner_oct == 0 { 7 } else { 11 };
                tree.insert(coord, slot);
            }
        }

        let before = tree.node_count();
        tree.deduplicate_subtrees();
        let after = tree.node_count();

        // All 8 root-children are structurally identical; they should all
        // reference a single shared 8-child block after dedup.
        assert!(
            after < before,
            "dedup should shrink: {} -> {}", before, after,
        );

        // Sanity: every lookup returns the correct slot.
        for root_oct in 0..8u32 {
            let dx = root_oct & 1;
            let dy = (root_oct >> 1) & 1;
            let dz = (root_oct >> 2) & 1;
            let base = UVec3::new(dx * 2, dy * 2, dz * 2);
            for inner_oct in 0..8u32 {
                let ix = inner_oct & 1;
                let iy = (inner_oct >> 1) & 1;
                let iz = (inner_oct >> 2) & 1;
                let coord = UVec3::new(base.x + ix, base.y + iy, base.z + iz);
                let expected = if inner_oct == 0 {
                    make_leaf(7)
                } else {
                    make_leaf(11)
                };
                assert_eq!(
                    tree.lookup(coord),
                    Some(expected),
                    "wrong lookup at {:?}", coord,
                );
            }
        }
    }

    #[test]
    fn deduplicate_preserves_unique_subtrees() {
        // A tree whose 8 root-children are all structurally different should
        // not shrink (nothing to share).
        let mut tree = SparseOctree::new(1, 0.1);
        for i in 0..8u32 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = (i >> 2) & 1;
            // Each position gets a unique slot.
            tree.insert(UVec3::new(x, y, z), 100 + i);
        }

        // Verify each lookup is distinct and correct BEFORE dedup.
        for i in 0..8u32 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = (i >> 2) & 1;
            assert_eq!(
                tree.lookup(UVec3::new(x, y, z)),
                Some(make_leaf(100 + i)),
            );
        }

        tree.deduplicate_subtrees();

        // Lookups still correct.
        for i in 0..8u32 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = (i >> 2) & 1;
            assert_eq!(
                tree.lookup(UVec3::new(x, y, z)),
                Some(make_leaf(100 + i)),
                "post-dedup lookup wrong at i={i}",
            );
        }
    }

    #[test]
    fn deduplicate_handles_trivial_root() {
        // A single-leaf tree: no branches, nothing to dedup, but shouldn't
        // crash and should leave the tree valid.
        let mut tree = SparseOctree::new(3, 0.1);
        // The default root is EMPTY_NODE. Dedup should be a no-op.
        tree.deduplicate_subtrees();
        assert_eq!(tree.nodes[0], EMPTY_NODE);
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn deduplicate_recursive_self_similar_pattern() {
        // Build a "corner" pattern: at every level of subdivision, octant 0
        // gets subdivided the same way. This creates nested self-similar
        // structure — dedup should collapse it dramatically.
        let mut tree = SparseOctree::new(4, 0.1); // 16x16x16

        // Insert a single voxel at (0,0,0) and another at (15,15,15).
        // This forces subdivision along two diagonal chains. The empty
        // octants at each level of the chain share structure (all EMPTY).
        tree.insert(UVec3::new(0, 0, 0), 1);
        tree.insert(UVec3::new(15, 15, 15), 2);

        let before = tree.node_count();
        tree.deduplicate_subtrees();
        let after = tree.node_count();

        // Lookups preserved.
        assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(1)));
        assert_eq!(tree.lookup(UVec3::new(15, 15, 15)), Some(make_leaf(2)));
        assert_eq!(tree.lookup(UVec3::new(5, 5, 5)), Some(EMPTY_NODE));

        // Even without obvious symmetry, there's enough shared sentinel
        // structure that dedup shouldn't grow the tree.
        assert!(after <= before, "dedup should not grow: {} -> {}", before, after);
    }

    #[test]
    fn compact_handles_mixed_orphans_and_reachable() {
        // Insert enough to create nested branches, then insert more causing
        // some subtrees to collapse — producing orphans — while leaving other
        // subtrees intact. Compact should drop the orphans but preserve the
        // rest.
        let mut tree = SparseOctree::new(2, 0.1);
        // Half of the tree gets uniform data (will collapse); the other half
        // gets two distinct values (can't collapse).
        for z in 0..2u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    tree.insert(UVec3::new(x, y, z), 7);
                }
            }
        }
        tree.insert(UVec3::new(0, 0, 3), 100);
        tree.insert(UVec3::new(1, 1, 3), 200);

        let before_count = tree.node_count();
        tree.compact();
        let after_count = tree.node_count();

        assert!(after_count < before_count, "compact should shrink when orphans exist ({} -> {})", before_count, after_count);

        // All original lookups must still succeed with the same values.
        assert_eq!(tree.lookup(UVec3::new(2, 2, 0)), Some(make_leaf(7)));
        assert_eq!(tree.lookup(UVec3::new(3, 3, 1)), Some(make_leaf(7)));
        assert_eq!(tree.lookup(UVec3::new(0, 0, 3)), Some(make_leaf(100)));
        assert_eq!(tree.lookup(UVec3::new(1, 1, 3)), Some(make_leaf(200)));
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

    /// GPU-style position-based lookup (mirrors octree_lookup in WGSL).
    /// Uses floating-point comparisons instead of integer bit tests.
    fn gpu_style_lookup(tree: &SparseOctree, pos: glam::Vec3) -> (u32, u8) {
        let extent = tree.extent() as f32 * tree.base_voxel_size();
        let mut offset = 0usize;
        let mut half = extent * 0.5;
        let mut center = glam::Vec3::splat(half);

        for level in 0..tree.depth() {
            let node = tree.as_slice()[offset];
            if node == EMPTY_NODE { return (EMPTY_NODE, level); }
            if node == INTERIOR_NODE { return (INTERIOR_NODE, level); }
            if is_leaf(node) { return (leaf_slot(node), level); }

            // Branch — same logic as GPU shader
            let gx = if pos.x >= center.x { 1u32 } else { 0 };
            let gy = if pos.y >= center.y { 1u32 } else { 0 };
            let gz = if pos.z >= center.z { 1u32 } else { 0 };
            let child = (gx + gy * 2 + gz * 4) as usize;
            offset = node as usize + child;

            half *= 0.5;
            center.x += if pos.x >= center.x { half } else { -half };
            center.y += if pos.y >= center.y { half } else { -half };
            center.z += if pos.z >= center.z { half } else { -half };
        }

        let node = tree.as_slice()[offset];
        if node == EMPTY_NODE { return (EMPTY_NODE, tree.depth()); }
        if node == INTERIOR_NODE { return (INTERIOR_NODE, tree.depth()); }
        if is_leaf(node) { return (leaf_slot(node), tree.depth()); }
        (EMPTY_NODE, tree.depth())
    }

    #[test]
    fn gpu_lookup_matches_coord_lookup() {
        // Build a small sphere octree (depth low enough that bricks don't
        // activate) and verify every leaf is reachable by position. Brick
        // path is exercised by tests in voxelize_octree.
        let mut attrs = crate::LeafAttrPool::new(100_000);
        let mut bricks = crate::BrickPool::new(64);
        let r = crate::voxelize_octree::voxelize_sphere_octree(
            glam::Vec3::ZERO, 0.4, 0, 0.4, &mut attrs, &mut bricks,
        ).unwrap();
        let tree = &r.octree;
        let _voxel_count = r.voxel_count;

        let vs = tree.base_voxel_size();
        let extent = tree.extent() as f32 * vs;
        let mut mismatches = 0u32;
        let mut total = 0u32;

        for (coord, slot, leaf_depth) in tree.iter_leaves() {
            total += 1;
            let depth_diff = tree.depth() - leaf_depth;
            let leaf_vs = vs * (1u32 << depth_diff) as f32;
            // Position at center of the leaf voxel
            let pos = glam::Vec3::new(
                coord.x as f32 * vs + leaf_vs * 0.5,
                coord.y as f32 * vs + leaf_vs * 0.5,
                coord.z as f32 * vs + leaf_vs * 0.5,
            );

            let (gpu_slot, gpu_depth) = gpu_style_lookup(&tree, pos);
            let (coord_node, _) = tree.lookup_with_depth(coord).unwrap();
            let coord_slot = if is_leaf(coord_node) { leaf_slot(coord_node) } else { coord_node };

            if gpu_slot != slot {
                if mismatches < 5 {
                    eprintln!(
                        "MISMATCH at coord={:?} pos={:?}: coord_lookup_slot={} gpu_slot={} (expected {})",
                        coord, pos, coord_slot, gpu_slot, slot
                    );
                }
                mismatches += 1;
            }
        }

        eprintln!("GPU lookup test: {total} leaves, {mismatches} mismatches");
        assert_eq!(mismatches, 0, "{mismatches}/{total} leaves unreachable by GPU-style position lookup");
    }

    #[test]
    fn gpu_lookup_matches_rkp_file() {
        // Test with an actual .rkp file if available.
        let path = "/home/joe/dev/rkifield_game/splat5/assets/models/bunny_pbr/scene.rkp";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping .rkp test — file not found: {path}");
            return;
        }

        let mut file = std::fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(&mut file);
        let header = match crate::asset_file::read_rkp_header(&mut reader) {
            Ok(h) => h,
            Err(e) => { eprintln!("Skipping .rkp test — header error: {e}"); return; }
        };
        let octree_nodes = crate::asset_file::read_rkp_octree(&mut reader, &header).unwrap();

        let depth = header.octree_depth as u8;
        let vs = header.base_voxel_size;
        let tree = SparseOctree::from_raw(&octree_nodes, depth, vs);

        let voxel_data = crate::asset_file::read_rkp_voxels(&mut reader, &header).unwrap();

        let extent = tree.extent() as f32 * vs;
        let mut mismatches = 0u32;
        let mut total = 0u32;

        for (coord, slot, leaf_depth) in tree.iter_leaves() {
            total += 1;
            let depth_diff = tree.depth() - leaf_depth;
            let leaf_vs = vs * (1u32 << depth_diff) as f32;
            let pos = glam::Vec3::new(
                coord.x as f32 * vs + leaf_vs * 0.5,
                coord.y as f32 * vs + leaf_vs * 0.5,
                coord.z as f32 * vs + leaf_vs * 0.5,
            );

            let (gpu_slot, _) = gpu_style_lookup(&tree, pos);
            if gpu_slot != slot {
                if mismatches < 10 {
                    eprintln!(
                        "MISMATCH coord={:?} pos={:?}: expected slot={} got gpu_slot={}",
                        coord, pos, slot, gpu_slot
                    );
                }
                mismatches += 1;
            }
        }

        eprintln!("GPU lookup .rkp test: {total} leaves, {mismatches} mismatches (extent={extent}, depth={depth}, vs={vs})");
        assert_eq!(mismatches, 0, "{mismatches}/{total} leaves unreachable by GPU-style lookup");
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
        assert!(is_leaf(make_leaf(0x3FFF_FFFD))); // max leaf_attr_id (30 bits - 2 reserved)
        assert!(!is_leaf(EMPTY_NODE));
        assert!(!is_leaf(INTERIOR_NODE));
        assert!(!is_leaf(make_brick(0)));
        assert!(!is_leaf(make_brick(42)));

        assert!(is_branch(0)); // offset 0 is a valid branch
        assert!(is_branch(100));
        assert!(!is_branch(EMPTY_NODE));
        assert!(!is_branch(INTERIOR_NODE));
        assert!(!is_branch(make_leaf(0)));
        assert!(!is_branch(make_brick(0)));
    }

    #[test]
    fn brick_encoding() {
        assert!(is_brick(make_brick(0)));
        assert!(is_brick(make_brick(42)));
        assert!(is_brick(make_brick(0x3FFF_FFFD)));
        assert!(!is_brick(EMPTY_NODE));
        assert!(!is_brick(INTERIOR_NODE));
        assert!(!is_brick(make_leaf(0)));
        assert!(!is_brick(0)); // branch
    }

    #[test]
    fn leaf_slot_roundtrip() {
        for slot in [0u32, 1, 42, 1000, 0x3FFF_FFFD] {
            assert_eq!(leaf_slot(make_leaf(slot)), slot);
        }
    }

    #[test]
    fn brick_id_roundtrip() {
        for id in [0u32, 1, 42, 1000, 0x3FFF_FFFD] {
            assert_eq!(brick_id(make_brick(id)), id);
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
