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
    pub fn deallocate(&mut self, handle: OctreeHandle) {
        let start = handle.root_offset as usize;
        let end = start + handle.len as usize;
        if end <= self.data.len() {
            // Clear to EMPTY to avoid stale data.
            for entry in &mut self.data[start..end] {
                *entry = crate::sparse_octree::EMPTY_NODE;
            }
            self.free_list.push((handle.root_offset, handle.len));
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
        let mut alloc = OctreeAllocator::new();

        let tree1 = make_test_octree();
        let h1 = alloc.allocate(&tree1);
        let buf_len_after_first = alloc.buffer_len();

        alloc.deallocate(h1);
        assert_eq!(alloc.free_region_count(), 1);

        // Allocate a small tree — should reuse the freed region.
        let small = SparseOctree::new(1, 0.1);
        let h2 = alloc.allocate(&small);
        assert_eq!(h2.root_offset, 0, "should reuse freed region at offset 0");

        // Buffer shouldn't have grown (reused space).
        assert_eq!(alloc.buffer_len(), buf_len_after_first);
    }

    #[test]
    fn deallocate_clears_to_empty() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let handle = alloc.allocate(&tree);

        alloc.deallocate(handle);

        // All entries in the deallocated region should be EMPTY_NODE.
        for i in 0..handle.len as usize {
            assert_eq!(alloc.as_slice()[i], EMPTY_NODE);
        }
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
}
