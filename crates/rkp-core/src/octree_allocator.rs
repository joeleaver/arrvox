//! Allocator that packs multiple [`SparseOctree`]s into a single contiguous
//! buffer suitable for GPU upload.
//!
//! Mirrors the design of rkf-core's `BrickMapAllocator` — each octree occupies
//! a contiguous region tracked by an [`OctreeHandle`]. Deallocated regions go
//! onto a free list for reuse.

use crate::sparse_octree::SparseOctree;

/// Handle to an allocated octree region in the packed buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OctreeHandle {
    /// Offset (in u32 entries) to the root node in the packed buffer.
    pub root_offset: u32,
    /// Number of u32 entries occupied by this octree.
    pub len: u32,
    /// Tree depth.
    pub depth: u8,
    /// Voxel size at the finest level.
    pub base_voxel_size: f32,
}

/// Packs multiple octrees into a single `Vec<u32>` for GPU upload.
#[derive(Debug)]
pub struct OctreeAllocator {
    data: Vec<u32>,
    free_list: Vec<(u32, u32)>, // (offset, length)
}

impl OctreeAllocator {
    /// Create a new empty allocator.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Create an allocator with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    /// Allocate space for an octree and copy its nodes into the packed buffer.
    ///
    /// The octree's internal node offsets are rebased to be absolute within the
    /// packed buffer (by adding the allocation offset to all branch pointers).
    pub fn allocate(&mut self, octree: &SparseOctree) -> OctreeHandle {
        let nodes = octree.as_slice();
        let len = nodes.len() as u32;

        let offset = if let Some(idx) = self.find_free_region(len) {
            let (free_offset, free_len) = self.free_list[idx];
            let start = free_offset as usize;
            self.data[start..start + len as usize].copy_from_slice(nodes);
            if free_len > len {
                self.free_list[idx] = (free_offset + len, free_len - len);
            } else {
                self.free_list.swap_remove(idx);
            }
            free_offset
        } else {
            let offset = self.data.len() as u32;
            self.data.extend_from_slice(nodes);
            offset
        };

        // Rebase branch pointers: all internal offsets are relative to the
        // octree's start (0-based). Shift them by `offset` so they become
        // absolute indices into the packed buffer.
        self.rebase_branches(offset as usize, len as usize, offset);

        OctreeHandle {
            root_offset: offset,
            len,
            depth: octree.depth(),
            base_voxel_size: octree.base_voxel_size(),
        }
    }

    /// Deallocate an octree region, adding it to the free list.
    ///
    /// Coalesces adjacent free regions and shrinks the backing buffer if the
    /// coalesced tail reaches `data.len()`. Without this, repeated re-voxelize
    /// of procedural objects whose new allocation is even slightly larger than
    /// any single freed region causes the buffer to grow monotonically.
    pub fn deallocate(&mut self, handle: OctreeHandle) {
        let start = handle.root_offset as usize;
        let end = start + handle.len as usize;
        if end > self.data.len() {
            return;
        }
        // Clear to EMPTY so stale branch offsets can't be misread.
        for entry in &mut self.data[start..end] {
            *entry = crate::sparse_octree::EMPTY_NODE;
        }
        self.free_list.push((handle.root_offset, handle.len));
        self.coalesce_and_shrink();
    }

    /// Merge adjacent regions in the free list, then truncate `data` if the
    /// merged tail ends at the current buffer length.
    fn coalesce_and_shrink(&mut self) {
        if self.free_list.is_empty() {
            return;
        }
        // Sort by offset so adjacency can be detected in one pass.
        self.free_list.sort_unstable_by_key(|&(off, _)| off);

        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.free_list.len());
        for &(off, len) in &self.free_list {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == off => last.1 += len,
                _ => merged.push((off, len)),
            }
        }
        self.free_list = merged;

        // If the last free region now ends at the buffer tail, drop it from
        // both the free list and the backing vec — truly reclaiming space.
        while let Some(&(off, len)) = self.free_list.last() {
            if (off + len) as usize == self.data.len() {
                self.data.truncate(off as usize);
                self.free_list.pop();
            } else {
                break;
            }
        }
    }

    /// Total length of the packed buffer (in u32 entries).
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    /// Backing slice for GPU buffer upload.
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    /// Number of free-list regions.
    #[inline]
    pub fn free_region_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total free entries across all free-list regions.
    pub fn total_free_entries(&self) -> u32 {
        self.free_list.iter().map(|&(_, len)| len).sum()
    }

    /// Rebase branch pointers in the allocated region.
    ///
    /// Branch node values are offsets into the octree's local node array (0-based).
    /// After packing at `base_offset` in the global buffer, they need to become
    /// `local_offset + base_offset`.
    fn rebase_branches(&mut self, start: usize, len: usize, base_offset: u32) {
        if base_offset == 0 {
            return; // No rebasing needed for offset 0.
        }
        use crate::sparse_octree::{is_branch, EMPTY_NODE, INTERIOR_NODE};
        for i in start..start + len {
            let node = self.data[i];
            if node == EMPTY_NODE || node == INTERIOR_NODE {
                continue;
            }
            if is_branch(node) {
                self.data[i] = node + base_offset;
            }
            // Leaves don't contain offsets — they store brick pool slots.
        }
    }

    fn find_free_region(&self, needed: u32) -> Option<usize> {
        self.free_list.iter().position(|&(_, len)| len >= needed)
    }
}

impl Default for OctreeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::{is_branch, EMPTY_NODE};
    use glam::UVec3;

    fn make_test_octree() -> SparseOctree {
        let mut tree = SparseOctree::new(2, 0.1);
        tree.insert(UVec3::new(0, 0, 0), 10);
        tree.insert(UVec3::new(3, 3, 3), 20);
        tree
    }

    #[test]
    fn allocate_single() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let handle = alloc.allocate(&tree);

        assert_eq!(handle.root_offset, 0);
        assert_eq!(handle.depth, 2);
        assert!(handle.len > 0);
        assert_eq!(alloc.buffer_len(), handle.len as usize);
    }

    #[test]
    fn allocate_preserves_lookups() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let handle = alloc.allocate(&tree);

        // Manually traverse the packed buffer to verify lookups work.
        let buf = alloc.as_slice();
        let root = buf[handle.root_offset as usize];

        // Root should be a branch (two different leaves).
        assert!(is_branch(root), "root should be branch, got {root:#x}");
    }

    #[test]
    fn allocate_multiple_non_overlapping() {
        let mut alloc = OctreeAllocator::new();

        let tree1 = make_test_octree();
        let tree2 = {
            let mut t = SparseOctree::new(1, 0.2);
            t.insert(UVec3::new(0, 0, 0), 99);
            t
        };

        let h1 = alloc.allocate(&tree1);
        let h2 = alloc.allocate(&tree2);

        // Non-overlapping regions.
        assert!(h2.root_offset >= h1.root_offset + h1.len);
        assert_eq!(alloc.buffer_len(), (h1.len + h2.len) as usize);
    }

    #[test]
    fn rebased_branches_are_absolute() {
        let mut alloc = OctreeAllocator::new();

        // Allocate a dummy first to push the second octree to a non-zero offset.
        let dummy = SparseOctree::new(1, 0.1);
        let _h_dummy = alloc.allocate(&dummy);

        let tree = make_test_octree();
        let handle = alloc.allocate(&tree);

        let buf = alloc.as_slice();
        let root = buf[handle.root_offset as usize];

        if is_branch(root) {
            let children_offset = root as usize;
            // Children offset should be absolute (>= handle.root_offset).
            assert!(
                children_offset >= handle.root_offset as usize,
                "branch offset {children_offset} should be >= handle offset {}",
                handle.root_offset
            );
            // And within the handle's region.
            assert!(
                children_offset < (handle.root_offset + handle.len) as usize,
                "branch offset {children_offset} should be < handle end {}",
                handle.root_offset + handle.len
            );
        }
    }

    #[test]
    fn deallocate_and_reuse() {
        // Freeing a non-tail region leaves a gap that the next small
        // allocation should reuse. Set up A, B; free A (middle); alloc small
        // into A's old slot. B pins the tail so coalesce-and-shrink can't
        // reclaim the region yet.
        let mut alloc = OctreeAllocator::new();

        let tree1 = make_test_octree();
        let h_a = alloc.allocate(&tree1);
        let _h_b = alloc.allocate(&tree1);
        let buf_len_with_both = alloc.buffer_len();

        alloc.deallocate(h_a);
        // A is a non-tail region — can't shrink, stays on free list.
        assert_eq!(alloc.free_region_count(), 1);
        assert_eq!(alloc.buffer_len(), buf_len_with_both);

        // Allocate a small tree — should reuse the freed region at offset 0.
        let small = SparseOctree::new(1, 0.1);
        let h_c = alloc.allocate(&small);
        assert_eq!(h_c.root_offset, 0, "should reuse freed region at offset 0");

        // Buffer didn't grow (reused space, B still pins the tail).
        assert_eq!(alloc.buffer_len(), buf_len_with_both);
    }

    #[test]
    fn deallocate_clears_to_empty() {
        // A non-tail region's slots must be cleared to EMPTY_NODE so stale
        // branch pointers can't be followed by accident. Pin the tail with B
        // so A's dealloc doesn't get truncated away.
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let h_a = alloc.allocate(&tree);
        let _h_b = alloc.allocate(&tree);

        alloc.deallocate(h_a);

        for i in 0..h_a.len as usize {
            assert_eq!(alloc.as_slice()[i], EMPTY_NODE);
        }
    }

    /// GPU-style position lookup on the packed (rebased) allocator buffer.
    fn gpu_lookup_packed(buf: &[u32], root: u32, depth: u8, extent: f32, vs: f32, pos: glam::Vec3) -> u32 {
        use crate::sparse_octree::{EMPTY_NODE, INTERIOR_NODE, is_leaf, leaf_slot};
        let mut offset = root as usize;
        let mut half = extent * 0.5;
        let mut center = glam::Vec3::splat(half);

        for _ in 0..depth {
            let node = buf[offset];
            if node == EMPTY_NODE { return EMPTY_NODE; }
            if node == INTERIOR_NODE { return INTERIOR_NODE; }
            if is_leaf(node) { return leaf_slot(node); }

            let gx = if pos.x >= center.x { 1u32 } else { 0 };
            let gy = if pos.y >= center.y { 1u32 } else { 0 };
            let gz = if pos.z >= center.z { 1u32 } else { 0 };
            let child = (gx + gy * 2 + gz * 4) as usize;
            offset = node as usize + child; // absolute offset (rebased)

            half *= 0.5;
            center.x += if pos.x >= center.x { half } else { -half };
            center.y += if pos.y >= center.y { half } else { -half };
            center.z += if pos.z >= center.z { half } else { -half };
        }

        let node = buf[offset];
        if is_leaf(node) { return leaf_slot(node); }
        if node == INTERIOR_NODE { return INTERIOR_NODE; }
        EMPTY_NODE
    }

    #[test]
    fn gpu_lookup_on_packed_rkp() {
        // Load bunny .rkp, allocate into packed buffer (with a dummy before it to
        // force non-zero offset), verify GPU-style position lookups match.
        let path = "/home/joe/dev/rkifield_game/splat5/assets/models/bunny_pbr/scene.rkp";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping packed .rkp test — file not found");
            return;
        }

        use crate::sparse_octree::SparseOctree;

        let mut file = std::fs::File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(&mut file);
        let header = match crate::asset_file::read_rkp_header(&mut reader) {
            Ok(h) => h,
            Err(e) => { eprintln!("Skipping packed .rkp test — header error: {e}"); return; }
        };
        let octree_nodes = crate::asset_file::read_rkp_octree(&mut reader, &header).unwrap();
        let depth = header.octree_depth as u8;
        let vs = header.base_voxel_size;

        // Allocate a dummy first to push the bunny to a non-zero offset.
        let mut alloc = OctreeAllocator::new();
        let dummy = SparseOctree::new(1, 0.1);
        let _h_dummy = alloc.allocate(&dummy);

        let tree = SparseOctree::from_raw(&octree_nodes, depth, vs);
        let handle = alloc.allocate(&tree);

        let extent = tree.extent() as f32 * vs;
        let buf = alloc.as_slice();
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

            let gpu_slot = gpu_lookup_packed(buf, handle.root_offset, depth, extent, vs, pos);
            // Note: slot is the ORIGINAL (0-based) pool slot from the tree.
            // The packed buffer has the same leaf slot values (they're not rebased).
            if gpu_slot != slot {
                if mismatches < 10 {
                    eprintln!(
                        "PACKED MISMATCH coord={:?} pos={:.4?}: expected slot={} got gpu_slot={} (root_offset={})",
                        coord, pos, slot, gpu_slot, handle.root_offset
                    );
                }
                mismatches += 1;
            }
        }

        eprintln!("Packed GPU lookup: {total} leaves, {mismatches} mismatches (root_offset={}, extent={extent})", handle.root_offset);
        assert_eq!(mismatches, 0, "{mismatches}/{total} leaves unreachable in packed buffer");
    }

    #[test]
    fn empty_allocator() {
        let alloc = OctreeAllocator::new();
        assert_eq!(alloc.buffer_len(), 0);
        assert_eq!(alloc.free_region_count(), 0);
        assert_eq!(alloc.total_free_entries(), 0);
    }

    #[test]
    fn with_capacity() {
        let alloc = OctreeAllocator::with_capacity(1024);
        assert_eq!(alloc.buffer_len(), 0);
    }

    #[test]
    fn coalesces_adjacent_free_regions() {
        // Allocate A, B, C sequentially. Freeing A and B should coalesce their
        // free regions into one, enabling a later allocation larger than either
        // individual freed region to fit without bumping the tail.
        let mut alloc = OctreeAllocator::new();
        let t_a = { let mut t = SparseOctree::new(1, 0.1); t.insert(UVec3::new(0,0,0), 1); t };
        let t_b = { let mut t = SparseOctree::new(1, 0.1); t.insert(UVec3::new(0,0,0), 2); t };
        let t_c = { let mut t = SparseOctree::new(1, 0.1); t.insert(UVec3::new(0,0,0), 3); t };
        let h_a = alloc.allocate(&t_a);
        let h_b = alloc.allocate(&t_b);
        let _h_c = alloc.allocate(&t_c);
        assert!(h_a.len > 0 && h_b.len > 0);

        alloc.deallocate(h_a);
        alloc.deallocate(h_b);
        // A + B are contiguous and C is still at the tail, so the free list
        // should contain exactly one coalesced entry covering both.
        assert_eq!(alloc.free_region_count(), 1);
        assert_eq!(alloc.total_free_entries(), h_a.len + h_b.len);
    }

    #[test]
    fn shrinks_buffer_when_tail_freed() {
        // Deallocating the tail region should shrink `data` — otherwise the
        // GPU upload size never decreases even when content drops.
        let mut alloc = OctreeAllocator::new();
        let t1 = make_test_octree();
        let h1 = alloc.allocate(&t1);
        let len_after_first = alloc.buffer_len();

        let t2 = make_test_octree();
        let h2 = alloc.allocate(&t2);
        let len_after_second = alloc.buffer_len();
        assert!(len_after_second > len_after_first);

        alloc.deallocate(h2);
        // Tail freed and coalesced — buffer should shrink back to h1's extent.
        assert_eq!(alloc.buffer_len(), len_after_first);
        assert_eq!(alloc.free_region_count(), 0);

        alloc.deallocate(h1);
        // Now everything is freed and was at the tail — buffer empty.
        assert_eq!(alloc.buffer_len(), 0);
        assert_eq!(alloc.free_region_count(), 0);
    }

    #[test]
    fn non_tail_dealloc_keeps_buffer_len_but_coalesces_later() {
        // Free a middle region — buffer can't shrink yet. Then free the tail;
        // coalescing should now drop the whole thing.
        let mut alloc = OctreeAllocator::new();
        let t = make_test_octree();
        let h1 = alloc.allocate(&t);
        let h2 = alloc.allocate(&t);
        let h3 = alloc.allocate(&t);
        let peak = alloc.buffer_len();

        alloc.deallocate(h2);
        assert_eq!(alloc.buffer_len(), peak);
        assert_eq!(alloc.free_region_count(), 1);

        alloc.deallocate(h3);
        // h2 + h3 now adjacent and at the tail — both should be reclaimed.
        assert_eq!(alloc.buffer_len(), h1.len as usize);
        assert_eq!(alloc.free_region_count(), 0);
    }
}
