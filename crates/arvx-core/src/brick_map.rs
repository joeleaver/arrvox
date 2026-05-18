//! Per-object brick maps — flat 3D arrays mapping brick coordinates to pool slots.
//!
//! In v2, each voxelized object owns a compact [`BrickMap`] that maps 3D brick
//! coordinates to slots in the global brick pool. This replaces the v1 chunk-based
//! spatial index with a per-object O(1) lookup.
//!
//! [`BrickMapAllocator`] packs multiple `BrickMap`s contiguously into a single
//! buffer suitable for GPU upload. Each allocation is tracked by a
//! [`BrickMapHandle`] (defined in [`crate::scene_node`]).
//!
//! # GPU layout
//!
//! All brick maps are packed into one `Vec<u32>` storage buffer. A voxelized
//! object stores its `brick_map_offset` and `brick_map_dims` in the GPU object
//! metadata. The shader computes:
//!
//! ```text
//! map_index = object.brick_map_offset + flatten(brick_coord, object.brick_dims)
//! brick_slot = brick_maps[map_index]
//! if brick_slot == EMPTY_SLOT: return MAX_DIST
//! ```

use glam::UVec3;

use crate::scene_node::BrickMapHandle;

/// Sentinel value indicating an empty (unallocated) brick map entry.
///
/// No brick is stored at this position — the object has no geometry there.
/// On the GPU, returns `+vs*4.0` (exterior/positive distance).
pub const EMPTY_SLOT: u32 = u32::MAX;

/// Interior empty slot — no brick allocated, but known to be inside the object.
/// On the GPU, returns `-vs*4.0` (interior/negative distance).
/// Used after sculpting to tell the GPU the correct sign for unallocated
/// interior bricks without wasting brick pool memory on constant-fill data.
pub const INTERIOR_SLOT: u32 = u32::MAX - 1;

/// A per-object brick map — flat 3D array of brick pool slot indices.
///
/// Maps 3D brick coordinates `(bx, by, bz)` within an object's local grid
/// to `u32` slot indices in the global brick pool. Entries equal to
/// [`EMPTY_SLOT`] indicate no brick at that position (empty air or deep interior).
///
/// Layout is row-major: `index = bx + by * dims.x + bz * dims.x * dims.y`.
#[derive(Debug, Clone)]
pub struct BrickMap {
    /// Dimensions of the 3D brick grid (number of bricks in each axis).
    pub dims: UVec3,
    /// Flat array of brick pool slot indices. Length = dims.x * dims.y * dims.z.
    pub entries: Vec<u32>,
}

impl BrickMap {
    /// Create a new brick map with given dimensions, all entries set to [`EMPTY_SLOT`].
    pub fn new(dims: UVec3) -> Self {
        let len = (dims.x * dims.y * dims.z) as usize;
        Self {
            dims,
            entries: vec![EMPTY_SLOT; len],
        }
    }

    /// Total number of entries in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the map has zero entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Flatten a 3D brick coordinate to a linear index.
    ///
    /// Returns `None` if any coordinate is out of bounds.
    #[inline]
    pub fn flatten(&self, bx: u32, by: u32, bz: u32) -> Option<usize> {
        if bx >= self.dims.x || by >= self.dims.y || bz >= self.dims.z {
            return None;
        }
        Some((bx + by * self.dims.x + bz * self.dims.x * self.dims.y) as usize)
    }

    /// Get the brick pool slot at `(bx, by, bz)`.
    ///
    /// Returns `None` if coordinates are out of bounds.
    #[inline]
    pub fn get(&self, bx: u32, by: u32, bz: u32) -> Option<u32> {
        self.flatten(bx, by, bz).map(|i| self.entries[i])
    }

    /// Set the brick pool slot at `(bx, by, bz)`.
    ///
    /// Returns `false` if coordinates are out of bounds.
    #[inline]
    pub fn set(&mut self, bx: u32, by: u32, bz: u32, slot: u32) -> bool {
        if let Some(i) = self.flatten(bx, by, bz) {
            self.entries[i] = slot;
            true
        } else {
            false
        }
    }

    /// Count the number of non-empty (allocated) entries.
    pub fn allocated_count(&self) -> usize {
        self.entries.iter().filter(|&&e| e != EMPTY_SLOT).count()
    }
}

/// Allocator that packs multiple [`BrickMap`]s into a single contiguous buffer.
///
/// Each allocation occupies a contiguous region in the packed buffer and is
/// tracked by a [`BrickMapHandle`]. Deallocated regions are added to a free
/// list for reuse.
///
/// The backing `data` vec is suitable for direct GPU upload as a storage buffer.
#[derive(Debug)]
pub struct BrickMapAllocator {
    /// Packed buffer of all brick map entries.
    data: Vec<u32>,
    /// Free list of `(offset, length)` pairs — previously deallocated regions.
    free_list: Vec<(u32, u32)>,
}

impl BrickMapAllocator {
    /// Create a new empty allocator.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Create an allocator pre-allocated with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    /// Allocate space for a brick map and copy its entries into the packed buffer.
    ///
    /// Returns a [`BrickMapHandle`] pointing to the allocation.
    pub fn allocate(&mut self, map: &BrickMap) -> BrickMapHandle {
        let len = map.entries.len() as u32;

        // Try to find a free-list region that fits.
        if let Some(idx) = self.find_free_region(len) {
            let (offset, free_len) = self.free_list[idx];

            // Copy entries into the free region.
            let start = offset as usize;
            self.data[start..start + len as usize].copy_from_slice(&map.entries);

            // If the free region is larger, shrink it; otherwise remove it.
            if free_len > len {
                self.free_list[idx] = (offset + len, free_len - len);
            } else {
                self.free_list.swap_remove(idx);
            }

            BrickMapHandle {
                offset,
                dims: map.dims,
            }
        } else {
            // Append to the end of the data buffer.
            let offset = self.data.len() as u32;
            self.data.extend_from_slice(&map.entries);
            BrickMapHandle {
                offset,
                dims: map.dims,
            }
        }
    }

    /// Deallocate a brick map region, adding it to the free list.
    ///
    /// The region's entries are reset to [`EMPTY_SLOT`].
    pub fn deallocate(&mut self, handle: BrickMapHandle) {
        let len = handle.dims.x * handle.dims.y * handle.dims.z;
        let start = handle.offset as usize;
        let end = start + len as usize;

        // Reset entries to EMPTY_SLOT.
        if end <= self.data.len() {
            for entry in &mut self.data[start..end] {
                *entry = EMPTY_SLOT;
            }
            self.free_list.push((handle.offset, len));
        }
    }

    /// Update a single entry in the packed buffer.
    ///
    /// `handle` identifies the brick map, `(bx, by, bz)` the coordinate within it.
    /// Returns `false` if out of bounds.
    pub fn set_entry(
        &mut self,
        handle: &BrickMapHandle,
        bx: u32,
        by: u32,
        bz: u32,
        slot: u32,
    ) -> bool {
        let dims = handle.dims;
        if bx >= dims.x || by >= dims.y || bz >= dims.z {
            return false;
        }
        let local_idx = bx + by * dims.x + bz * dims.x * dims.y;
        let global_idx = handle.offset + local_idx;
        if (global_idx as usize) < self.data.len() {
            self.data[global_idx as usize] = slot;
            true
        } else {
            false
        }
    }

    /// Read a single entry from the packed buffer.
    ///
    /// Returns `None` if out of bounds.
    pub fn get_entry(&self, handle: &BrickMapHandle, bx: u32, by: u32, bz: u32) -> Option<u32> {
        let dims = handle.dims;
        if bx >= dims.x || by >= dims.y || bz >= dims.z {
            return None;
        }
        let local_idx = bx + by * dims.x + bz * dims.x * dims.y;
        let global_idx = (handle.offset + local_idx) as usize;
        self.data.get(global_idx).copied()
    }

    /// Total length of the packed buffer (in u32 entries).
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    /// Backing slice — suitable for GPU buffer upload.
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    /// Mutable backing slice — for bulk operations like fill.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        &mut self.data
    }

    /// Fill a handle's region with a constant value (e.g., EMPTY_SLOT).
    /// Much faster than per-entry set_entry calls for clearing.
    #[inline]
    pub fn fill_region(&mut self, handle: &BrickMapHandle, value: u32) {
        let start = handle.offset as usize;
        let count = (handle.dims.x * handle.dims.y * handle.dims.z) as usize;
        if start + count <= self.data.len() {
            self.data[start..start + count].fill(value);
        }
    }

    /// Number of entries on the free list.
    #[inline]
    pub fn free_region_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of free entries across all free-list regions.
    pub fn total_free_entries(&self) -> u32 {
        self.free_list.iter().map(|&(_, len)| len).sum()
    }

    /// Find the first free-list region that can fit `needed` entries.
    fn find_free_region(&self, needed: u32) -> Option<usize> {
        self.free_list
            .iter()
            .position(|&(_, len)| len >= needed)
    }
}

impl Default for BrickMapAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── BrickMap tests ──────────────────────────────────────────────────

    #[test]
    fn brick_map_new_all_empty() {
        let map = BrickMap::new(UVec3::new(4, 4, 4));
        assert_eq!(map.len(), 64);
        assert_eq!(map.allocated_count(), 0);
        for &entry in &map.entries {
            assert_eq!(entry, EMPTY_SLOT);
        }
    }

    #[test]
    fn brick_map_zero_dims() {
        let map = BrickMap::new(UVec3::ZERO);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn brick_map_get_set_roundtrip() {
        let mut map = BrickMap::new(UVec3::new(4, 3, 2));
        assert_eq!(map.get(0, 0, 0), Some(EMPTY_SLOT));

        assert!(map.set(2, 1, 0, 42));
        assert_eq!(map.get(2, 1, 0), Some(42));
        assert_eq!(map.allocated_count(), 1);
    }

    #[test]
    fn brick_map_out_of_bounds() {
        let mut map = BrickMap::new(UVec3::new(2, 2, 2));
        assert_eq!(map.get(2, 0, 0), None);
        assert_eq!(map.get(0, 2, 0), None);
        assert_eq!(map.get(0, 0, 2), None);
        assert!(!map.set(2, 0, 0, 1));
    }

    #[test]
    fn brick_map_flatten_correctness() {
        let map = BrickMap::new(UVec3::new(3, 4, 5));
        // (0,0,0) = 0
        assert_eq!(map.flatten(0, 0, 0), Some(0));
        // (2,0,0) = 2
        assert_eq!(map.flatten(2, 0, 0), Some(2));
        // (0,1,0) = 3
        assert_eq!(map.flatten(0, 1, 0), Some(3));
        // (0,0,1) = 3*4 = 12
        assert_eq!(map.flatten(0, 0, 1), Some(12));
        // (2,3,4) = 2 + 3*3 + 4*3*4 = 2 + 9 + 48 = 59
        assert_eq!(map.flatten(2, 3, 4), Some(59));
    }

    #[test]
    fn brick_map_all_entries_unique_index() {
        let map = BrickMap::new(UVec3::new(3, 4, 5));
        let mut seen = std::collections::HashSet::new();
        for z in 0..5 {
            for y in 0..4 {
                for x in 0..3 {
                    let idx = map.flatten(x, y, z).unwrap();
                    assert!(seen.insert(idx), "duplicate index {idx} at ({x},{y},{z})");
                }
            }
        }
        assert_eq!(seen.len(), 60);
    }

    #[test]
    fn brick_map_allocated_count() {
        let mut map = BrickMap::new(UVec3::new(2, 2, 2));
        assert_eq!(map.allocated_count(), 0);
        map.set(0, 0, 0, 10);
        map.set(1, 1, 1, 20);
        assert_eq!(map.allocated_count(), 2);
        // Setting back to EMPTY_SLOT decrements count.
        map.set(0, 0, 0, EMPTY_SLOT);
        assert_eq!(map.allocated_count(), 1);
    }

    // ── BrickMapAllocator tests ─────────────────────────────────────────

    #[test]
    fn allocator_empty() {
        let alloc = BrickMapAllocator::new();
        assert_eq!(alloc.buffer_len(), 0);
        assert_eq!(alloc.free_region_count(), 0);
    }

    #[test]
    fn allocator_single_allocation() {
        let mut alloc = BrickMapAllocator::new();
        let mut map = BrickMap::new(UVec3::new(2, 2, 2));
        map.set(0, 0, 0, 100);
        map.set(1, 1, 1, 200);

        let handle = alloc.allocate(&map);
        assert_eq!(handle.offset, 0);
        assert_eq!(handle.dims, UVec3::new(2, 2, 2));
        assert_eq!(alloc.buffer_len(), 8);

        // Verify entries in packed buffer.
        assert_eq!(alloc.get_entry(&handle, 0, 0, 0), Some(100));
        assert_eq!(alloc.get_entry(&handle, 1, 1, 1), Some(200));
        assert_eq!(alloc.get_entry(&handle, 1, 0, 0), Some(EMPTY_SLOT));
    }

    #[test]
    fn allocator_multiple_allocations() {
        let mut alloc = BrickMapAllocator::new();

        let map1 = BrickMap::new(UVec3::new(2, 2, 2)); // 8 entries
        let map2 = BrickMap::new(UVec3::new(3, 3, 3)); // 27 entries

        let h1 = alloc.allocate(&map1);
        let h2 = alloc.allocate(&map2);

        assert_eq!(h1.offset, 0);
        assert_eq!(h2.offset, 8);
        assert_eq!(alloc.buffer_len(), 8 + 27);
    }

    #[test]
    fn allocator_deallocate_and_reuse() {
        let mut alloc = BrickMapAllocator::new();

        let map1 = BrickMap::new(UVec3::new(2, 2, 2)); // 8 entries
        let map2 = BrickMap::new(UVec3::new(3, 3, 3)); // 27 entries

        let h1 = alloc.allocate(&map1);
        let _h2 = alloc.allocate(&map2);

        // Deallocate the first map.
        alloc.deallocate(h1);
        assert_eq!(alloc.free_region_count(), 1);
        assert_eq!(alloc.total_free_entries(), 8);

        // Entries should be reset to EMPTY_SLOT.
        for i in 0..8 {
            assert_eq!(alloc.as_slice()[i], EMPTY_SLOT);
        }

        // Allocate a new map that fits in the freed region.
        let mut map3 = BrickMap::new(UVec3::new(2, 2, 2)); // 8 entries — exact fit
        map3.set(0, 0, 0, 999);
        let h3 = alloc.allocate(&map3);

        // Should reuse the freed region.
        assert_eq!(h3.offset, 0);
        assert_eq!(alloc.free_region_count(), 0);
        assert_eq!(alloc.get_entry(&h3, 0, 0, 0), Some(999));
    }

    #[test]
    fn allocator_partial_free_region_reuse() {
        let mut alloc = BrickMapAllocator::new();

        // Allocate a large region.
        let big_map = BrickMap::new(UVec3::new(4, 4, 4)); // 64 entries
        let h_big = alloc.allocate(&big_map);

        // Deallocate it.
        alloc.deallocate(h_big);
        assert_eq!(alloc.total_free_entries(), 64);

        // Allocate a smaller map — should partially consume the free region.
        let small_map = BrickMap::new(UVec3::new(2, 2, 2)); // 8 entries
        let h_small = alloc.allocate(&small_map);
        assert_eq!(h_small.offset, 0);
        // Free region should shrink.
        assert_eq!(alloc.total_free_entries(), 56);
    }

    #[test]
    fn allocator_set_entry() {
        let mut alloc = BrickMapAllocator::new();
        let map = BrickMap::new(UVec3::new(2, 2, 2));
        let handle = alloc.allocate(&map);

        assert!(alloc.set_entry(&handle, 1, 0, 0, 42));
        assert_eq!(alloc.get_entry(&handle, 1, 0, 0), Some(42));
    }

    #[test]
    fn allocator_set_entry_out_of_bounds() {
        let mut alloc = BrickMapAllocator::new();
        let map = BrickMap::new(UVec3::new(2, 2, 2));
        let handle = alloc.allocate(&map);

        assert!(!alloc.set_entry(&handle, 2, 0, 0, 42));
    }

    #[test]
    fn allocator_get_entry_out_of_bounds() {
        let alloc = BrickMapAllocator::new();
        let handle = BrickMapHandle {
            offset: 0,
            dims: UVec3::new(2, 2, 2),
        };
        assert_eq!(alloc.get_entry(&handle, 0, 0, 0), None);
    }

    #[test]
    fn allocator_with_capacity() {
        let alloc = BrickMapAllocator::with_capacity(1024);
        assert_eq!(alloc.buffer_len(), 0);
        // Internal capacity is at least 1024 (Vec guarantees).
    }

    #[test]
    fn allocator_many_allocations() {
        let mut alloc = BrickMapAllocator::new();
        let mut handles = Vec::new();

        for i in 0..50 {
            let mut map = BrickMap::new(UVec3::new(2, 2, 2));
            map.set(0, 0, 0, i);
            handles.push(alloc.allocate(&map));
        }

        // Verify all 50 maps are accessible.
        for (i, handle) in handles.iter().enumerate() {
            assert_eq!(alloc.get_entry(handle, 0, 0, 0), Some(i as u32));
        }

        assert_eq!(alloc.buffer_len(), 50 * 8);
    }

    #[test]
    fn allocator_dealloc_then_larger_alloc_appends() {
        let mut alloc = BrickMapAllocator::new();

        let small = BrickMap::new(UVec3::new(2, 2, 2)); // 8 entries
        let h = alloc.allocate(&small);
        alloc.deallocate(h);

        // Allocate something larger than the freed region.
        let big = BrickMap::new(UVec3::new(3, 3, 3)); // 27 entries — doesn't fit in 8
        let h_big = alloc.allocate(&big);

        // Should append, not reuse the 8-entry region.
        assert_eq!(h_big.offset, 8);
        // Free list still has the small region.
        assert_eq!(alloc.free_region_count(), 1);
    }
}
