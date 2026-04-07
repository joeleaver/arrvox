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
    /// Next unallocated slot.
    next_free: u32,
}

impl VoxelPool {
    /// Create a pool with the given initial capacity (number of voxels).
    pub fn new(capacity: u32) -> Self {
        Self {
            data: vec![SplatVoxel::EMPTY; capacity as usize],
            colors: vec![0u32; capacity as usize],
            next_free: 0,
        }
    }

    /// Allocate `count` contiguous voxel slots.
    ///
    /// Returns `None` if the pool is full. Slots are returned as a contiguous
    /// range `[start .. start + count)`.
    pub fn allocate_range(&mut self, count: u32) -> Option<Vec<u32>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let start = self.next_free;
        let end = start + count;
        if end as usize > self.data.len() {
            return None;
        }
        self.next_free = end;
        Some((start..end).collect())
    }

    /// Allocate a single voxel slot.
    pub fn allocate(&mut self) -> Option<u32> {
        if (self.next_free as usize) >= self.data.len() {
            return None;
        }
        let slot = self.next_free;
        self.next_free += 1;
        Some(slot)
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
    fn allocate_fills_pool() {
        let mut pool = VoxelPool::new(5);
        assert!(pool.allocate_range(5).is_some());
        assert!(pool.allocate().is_none());
        assert!(pool.allocate_range(1).is_none());
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
}
