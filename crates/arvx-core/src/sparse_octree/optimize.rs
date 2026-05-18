//! In-place optimization passes: dedup → DAG, Morton-order rewrite, compact.
//!
//! These passes rewrite the node buffer; they should be run after construction
//! is complete and before GPU upload. They preserve the parallel
//! `internal_attr_index` buffer so that prefiltered-LOD ids ride along through
//! every rewrite.

use super::{INTERNAL_ATTR_NONE, SparseOctree, is_branch};

impl SparseOctree {
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
            self.internal_attr_index.truncate(1);
            return;
        }

        let mut new_nodes: Vec<u32> = Vec::new();
        new_nodes.push(0); // reserve position [0] for root's value

        // Parallel prefilter buffer for the rewrite. Root's entry is a
        // direct copy from the old buffer; internal branch slots carry
        // their prefilter-id through the recursion (see dedup_value).
        let mut new_prefilter: Vec<u32> = Vec::with_capacity(self.internal_attr_index.len());
        new_prefilter.push(self.internal_attr_index[0]);

        let mut seen: std::collections::HashMap<[u32; 8], u32> =
            std::collections::HashMap::new();

        let canonical_root = Self::dedup_value(
            &self.nodes,
            &self.internal_attr_index,
            0,
            &mut new_nodes,
            &mut new_prefilter,
            &mut seen,
        );
        new_nodes[0] = canonical_root;

        debug_assert_eq!(new_nodes.len(), new_prefilter.len());
        self.nodes = new_nodes;
        self.internal_attr_index = new_prefilter;
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
            self.internal_attr_index.truncate(1);
            return;
        }

        let old = std::mem::take(&mut self.nodes);
        let old_prefilter = std::mem::take(&mut self.internal_attr_index);
        let mut new_nodes: Vec<u32> = Vec::with_capacity(old.len());
        let mut new_prefilter: Vec<u32> = Vec::with_capacity(old.len());

        // Root lives at new offset 0. Reserve its slot; the branch loop below
        // will write the correct offset to its 8 children.
        new_nodes.push(0);
        new_prefilter.push(INTERNAL_ATTR_NONE);

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

            // Every visited slot — branch or not — copies its prefilter-id
            // from old_prefilter[old_off] to new_prefilter[new_off]. DAG
            // sharing is honored because two different parent slots pointing
            // to the same old children-block end up writing into the same
            // new children-block slots; the prefilter values at those slots
            // are identical (prefilter is a pure function of the subtree).
            new_prefilter[new_off as usize] = old_prefilter[old_off as usize];

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
                new_prefilter.extend(std::iter::repeat(INTERNAL_ATTR_NONE).take(8));
                branch_map.insert(old_children, start);
                for i in 0..8u32 {
                    queue.push_back((old_children + i, start + i));
                }
                start
            };

            new_nodes[new_off as usize] = new_children;
        }

        debug_assert_eq!(new_nodes.len(), new_prefilter.len());
        self.nodes = new_nodes;
        self.internal_attr_index = new_prefilter;
    }

    /// Canonicalize the node value at `slot_in_old`. Leaves and sentinels
    /// pass through unchanged; branches expand, their 8 children are
    /// recursively canonicalized, and the resulting 8-child block is
    /// inserted into (or fetched from) the dedup map.
    ///
    /// Prefilter-ids ride along through the parallel `old_prefilter` array:
    /// when we emit a fresh 8-child block to `new_nodes`, we also write the
    /// 8 per-slot prefilter-ids from the old buffer into `new_prefilter`.
    /// The function does *not* write into `new_nodes[slot_in_old]` or the
    /// corresponding new_prefilter slot — that's the caller's responsibility
    /// (the parent's 8-tuple write, or the outer deduplicate_subtrees for
    /// the root).
    fn dedup_value(
        old_nodes: &[u32],
        old_prefilter: &[u32],
        slot_in_old: u32,
        new_nodes: &mut Vec<u32>,
        new_prefilter: &mut Vec<u32>,
        seen: &mut std::collections::HashMap<[u32; 8], u32>,
    ) -> u32 {
        let node_value = old_nodes[slot_in_old as usize];
        if !is_branch(node_value) {
            return node_value;
        }
        let children_offset = node_value as usize;
        let mut canonical_children: [u32; 8] = [0; 8];
        let mut child_prefilters: [u32; 8] = [INTERNAL_ATTR_NONE; 8];
        for i in 0..8 {
            canonical_children[i] = Self::dedup_value(
                old_nodes,
                old_prefilter,
                (children_offset + i) as u32,
                new_nodes,
                new_prefilter,
                seen,
            );
            child_prefilters[i] = old_prefilter[children_offset + i];
        }
        if let Some(&existing) = seen.get(&canonical_children) {
            return existing;
        }
        let new_offset = new_nodes.len() as u32;
        new_nodes.extend_from_slice(&canonical_children);
        new_prefilter.extend_from_slice(&child_prefilters);
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
            self.internal_attr_index.truncate(1);
            return;
        }

        let mut new_nodes: Vec<u32> = Vec::with_capacity(self.nodes.len());
        let mut new_prefilter: Vec<u32> = Vec::with_capacity(self.nodes.len());
        new_nodes.push(0); // placeholder for root; filled in below
        new_prefilter.push(self.internal_attr_index[0]);

        // Work queue holds pairs (old_node_idx, new_slot_idx). For each
        // entry, we read the node at `old_node_idx` from the old buffer;
        // if it's a branch we allocate 8 new children slots and enqueue them.
        //
        // The root is handled specially since its value lives at `nodes[0]`.
        let mut queue: std::collections::VecDeque<(u32, u32)> =
            std::collections::VecDeque::new();

        // Allocate 8 slots for the root's children in the new buffer.
        let root_children_new = new_nodes.len() as u32;
        new_nodes.extend(std::iter::repeat(0u32).take(8));
        new_prefilter.extend(std::iter::repeat(INTERNAL_ATTR_NONE).take(8));
        new_nodes[0] = root_children_new;
        let root_children_old = root;
        for i in 0..8u32 {
            queue.push_back((root_children_old + i, root_children_new + i));
        }

        while let Some((old_idx, new_idx)) = queue.pop_front() {
            // Every visited slot copies its prefilter-id. Non-branch slots
            // carry INTERNAL_ATTR_NONE (harmless); branch slots carry their
            // prefiltered LeafAttr id.
            new_prefilter[new_idx as usize] = self.internal_attr_index[old_idx as usize];

            let node = self.nodes[old_idx as usize];
            if is_branch(node) {
                // Allocate 8 new children slots.
                let children_new = new_nodes.len() as u32;
                new_nodes.extend(std::iter::repeat(0u32).take(8));
                new_prefilter.extend(std::iter::repeat(INTERNAL_ATTR_NONE).take(8));
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

        debug_assert_eq!(new_nodes.len(), new_prefilter.len());
        self.nodes = new_nodes;
        self.internal_attr_index = new_prefilter;
    }
}
