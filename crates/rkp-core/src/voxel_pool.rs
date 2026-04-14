//! Flat pool of individual [`SplatVoxel`] entries with parallel color data.
//!
//! Replaces `BrickPool` for per-voxel octree leaves. Each slot holds one
//! `SplatVoxel` (8 bytes) instead of a 512-voxel brick (4096 bytes).
//! Octree leaves index directly into this pool — no within-brick addressing.
//!
//! Color is stored in a parallel array at the same index: `colors[slot]`.
//! No companion map needed — the 1:1 relationship is implicit.

use crate::SplatVoxel;

/// A flat pool of individual voxels, indexed by slot number.
///
/// Parallel arrays: `data[slot]` = voxel, `colors[slot]` = packed RGBA.
/// Same index, no indirection.
pub struct VoxelPool {
    data: Vec<SplatVoxel>,
    /// Parallel color array: packed R|G|B|intensity (same as ColorVoxel).
    /// 0 = no color assigned.
    colors: Vec<u32>,
    /// Next unallocated slot (bump pointer).
    next_free: u32,
    /// Free list of reclaimed ranges — `(start, count)` pairs. Allocations
    /// prefer free ranges over bumping `next_free` to prevent leaks from
    /// re-voxelized procedural objects.
    free_list: Vec<(u32, u32)>,
}

impl VoxelPool {
    /// Create a pool with the given initial capacity (number of voxels).
    pub fn new(capacity: u32) -> Self {
        Self {
            data: vec![SplatVoxel::EMPTY; capacity as usize],
            colors: vec![0u32; capacity as usize],
            next_free: 0,
            free_list: Vec::new(),
        }
    }

    /// Allocate `count` contiguous voxel slots.
    ///
    /// Prefers reusing an entry from `free_list` (first-fit) so re-voxelized
    /// procedural objects don't leak their old range. Falls back to the bump
    /// pointer (growing the pool if needed) when no free range is big enough.
    /// Only returns `None` for count == 0 or if growth would overflow u32.
    pub fn allocate_range(&mut self, count: u32) -> Option<Vec<u32>> {
        if count == 0 {
            return Some(Vec::new());
        }

        // First-fit reuse of a previously-freed range.
        if let Some(idx) = self.free_list.iter().position(|(_, c)| *c >= count) {
            let (start, free_count) = self.free_list[idx];
            if free_count == count {
                self.free_list.swap_remove(idx);
            } else {
                // Split: keep the unused tail of this free range.
                self.free_list[idx] = (start + count, free_count - count);
            }
            return Some((start..start + count).collect());
        }

        let start = self.next_free;
        let end = start.checked_add(count)?;
        if end as usize > self.data.len() {
            // Grow: double or fit, whichever is larger.
            let new_cap = (self.data.len() as u32 * 2).max(end);
            self.grow(new_cap);
        }
        self.next_free = end;
        Some((start..end).collect())
    }

    /// Allocate a single voxel slot.
    ///
    /// Prefers a 1-slot entry from `free_list` before bumping. Only returns
    /// `None` if growth would overflow u32 capacity (~4B voxels).
    pub fn allocate(&mut self) -> Option<u32> {
        // First-fit reuse: split a 1-slot off any free range.
        if let Some(idx) = self.free_list.iter().position(|(_, c)| *c >= 1) {
            let (start, free_count) = self.free_list[idx];
            if free_count == 1 {
                self.free_list.swap_remove(idx);
            } else {
                self.free_list[idx] = (start + 1, free_count - 1);
            }
            return Some(start);
        }

        if (self.next_free as usize) >= self.data.len() {
            // Grow by doubling.
            let new_cap = (self.data.len() as u32).checked_mul(2)?;
            self.grow(new_cap);
        }
        let slot = self.next_free;
        self.next_free += 1;
        Some(slot)
    }

    /// Return a contiguous range of slots for reuse.
    ///
    /// Clears the voxel data (opacity=0) and color at those slots so stale
    /// values don't leak. If the range sits at the bump pointer, shrinks it
    /// to reclaim space immediately. This is the common case for procedural
    /// re-voxelization: dealloc old range → bump shrinks → alloc new range
    /// starting at the same base.
    pub fn deallocate_range(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let end = (start + count) as usize;
        if end > self.data.len() {
            return;
        }
        for s in start as usize..end {
            self.data[s] = SplatVoxel::EMPTY;
            self.colors[s] = 0;
        }
        if start + count == self.next_free {
            self.next_free = start;
            // Opportunistically coalesce: if a free range ends exactly where
            // next_free now is, absorb it too.
            loop {
                let idx = self.free_list.iter().position(|(s, c)| s + c == self.next_free);
                match idx {
                    Some(i) => {
                        let (s, _) = self.free_list.swap_remove(i);
                        self.next_free = s;
                    }
                    None => break,
                }
            }
        } else {
            // Non-contiguous — add to free list (may leak until a later dealloc
            // touches the bump pointer and coalesces).
            self.free_list.push((start, count));
        }
    }

    /// Read a voxel by slot index.
    #[inline]
    pub fn get(&self, slot: u32) -> &SplatVoxel {
        &self.data[slot as usize]
    }

    /// Write a voxel by slot index.
    #[inline]
    pub fn get_mut(&mut self, slot: u32) -> &mut SplatVoxel {
        &mut self.data[slot as usize]
    }

    /// Number of allocated voxels.
    #[inline]
    pub fn allocated_count(&self) -> u32 {
        self.next_free
    }

    /// Total capacity in voxels.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.data.len() as u32
    }

    /// Number of free slots remaining.
    #[inline]
    pub fn free_count(&self) -> u32 {
        self.capacity() - self.next_free
    }

    /// Read the packed color (R|G|B|intensity) at a slot.
    #[inline]
    pub fn color(&self, slot: u32) -> u32 {
        self.colors[slot as usize]
    }

    /// Set the packed color at a slot.
    #[inline]
    pub fn set_color(&mut self, slot: u32, packed: u32) {
        self.colors[slot as usize] = packed;
    }

    /// Grow the pool to at least `new_cap` voxels. Preserves existing data.
    pub fn grow(&mut self, new_cap: u32) {
        if new_cap as usize <= self.data.len() {
            return;
        }
        self.data.resize(new_cap as usize, SplatVoxel::EMPTY);
        self.colors.resize(new_cap as usize, 0);
    }

    /// Raw byte slice of the allocated region (for GPU upload).
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.next_free as usize;
        if count == 0 {
            return &[];
        }
        let ptr = self.data.as_ptr() as *const u8;
        let byte_len = count * std::mem::size_of::<SplatVoxel>();
        unsafe { std::slice::from_raw_parts(ptr, byte_len) }
    }

    /// Raw slice of all allocated voxels.
    pub fn as_slice(&self) -> &[SplatVoxel] {
        &self.data[..self.next_free as usize]
    }

    /// Raw byte slice of allocated color data (for GPU upload).
    pub fn color_bytes(&self) -> &[u8] {
        let count = self.next_free as usize;
        if count == 0 {
            return &[];
        }
        bytemuck::cast_slice(&self.colors[..count])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_is_empty() {
        let pool = VoxelPool::new(100);
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.capacity(), 100);
        assert_eq!(pool.free_count(), 100);
    }

    #[test]
    fn allocate_single() {
        let mut pool = VoxelPool::new(10);
        let slot = pool.allocate().unwrap();
        assert_eq!(slot, 0);
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.free_count(), 9);
    }

    #[test]
    fn allocate_range() {
        let mut pool = VoxelPool::new(100);
        let slots = pool.allocate_range(10).unwrap();
        assert_eq!(slots.len(), 10);
        assert_eq!(slots[0], 0);
        assert_eq!(slots[9], 9);
        assert_eq!(pool.allocated_count(), 10);
    }

    #[test]
    fn allocate_grows_pool_when_full() {
        let mut pool = VoxelPool::new(5);
        assert!(pool.allocate_range(5).is_some());
        assert_eq!(pool.capacity(), 5);
        // Beyond capacity — pool should grow automatically.
        assert!(pool.allocate().is_some());
        assert!(pool.capacity() > 5);
        // Range allocation beyond current capacity also grows.
        assert!(pool.allocate_range(100).is_some());
        assert!(pool.capacity() >= pool.allocated_count());
    }

    #[test]
    fn get_set_roundtrip() {
        let mut pool = VoxelPool::new(10);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = SplatVoxel::new(0.75, 42);
        assert!((pool.get(slot).opacity_f32() - 0.75).abs() < 0.01);
        assert_eq!(pool.get(slot).material_id(), 42);
    }

    #[test]
    fn grow_preserves_data() {
        let mut pool = VoxelPool::new(5);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = SplatVoxel::new(1.0, 99);
        pool.grow(100);
        assert_eq!(pool.capacity(), 100);
        assert!((pool.get(slot).opacity_f32() - 1.0).abs() < 0.01);
        assert_eq!(pool.get(slot).material_id(), 99);
    }

    #[test]
    fn as_bytes_length() {
        let mut pool = VoxelPool::new(10);
        pool.allocate_range(3).unwrap();
        assert_eq!(pool.as_bytes().len(), 3 * 8); // 3 voxels * 8 bytes each
    }

    #[test]
    fn allocate_range_zero() {
        let mut pool = VoxelPool::new(10);
        let slots = pool.allocate_range(0).unwrap();
        assert!(slots.is_empty());
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn reuses_freed_range_instead_of_bumping() {
        // Two objects: A gets slots [0,10), B gets [10,20). Freeing A leaves
        // a non-tail free range. A re-allocation of the same size should
        // reuse it instead of bumping next_free past 20.
        let mut pool = VoxelPool::new(100);
        let a = pool.allocate_range(10).unwrap();
        let _b = pool.allocate_range(10).unwrap();
        assert_eq!(pool.allocated_count(), 20);

        pool.deallocate_range(a[0], 10);
        // A wasn't at the tail, so its range went to the free list.
        assert_eq!(pool.allocated_count(), 20);

        let a2 = pool.allocate_range(10).unwrap();
        // Reused A's old range — no bump.
        assert_eq!(a2[0], 0);
        assert_eq!(pool.allocated_count(), 20);
    }

    #[test]
    fn reuses_partial_free_range() {
        // A free range larger than requested is split; the unused tail
        // remains on the free list for a later allocation.
        let mut pool = VoxelPool::new(100);
        let a = pool.allocate_range(20).unwrap();
        let _b = pool.allocate_range(5).unwrap();
        pool.deallocate_range(a[0], 20);

        let small = pool.allocate_range(7).unwrap();
        assert_eq!(small[0], 0);
        // 13 slots should remain on the free list — another alloc reuses them.
        let rest = pool.allocate_range(13).unwrap();
        assert_eq!(rest[0], 7);
        // Still nothing new past the original tail.
        assert_eq!(pool.allocated_count(), 25);
    }

    #[test]
    fn single_allocate_reuses_free_list() {
        let mut pool = VoxelPool::new(100);
        let a = pool.allocate_range(3).unwrap();
        let _b = pool.allocate_range(2).unwrap();
        pool.deallocate_range(a[0], 3);

        let s1 = pool.allocate().unwrap();
        let s2 = pool.allocate().unwrap();
        let s3 = pool.allocate().unwrap();
        // Three consecutive single allocs come from the freed 3-slot range,
        // not from bumping past the tail.
        assert_eq!([s1, s2, s3], [0, 1, 2]);
        assert_eq!(pool.allocated_count(), 5);
    }
}
