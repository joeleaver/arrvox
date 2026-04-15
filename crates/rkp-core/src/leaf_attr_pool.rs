//! Flat pool of [`LeafAttr`] entries with parallel per-leaf color.
//!
//! Each pool slot holds a [`LeafAttr`] (material IDs + normal, 8 B) plus a
//! packed u32 color (4 B), same index on both arrays. Color is 0 for leaves
//! where the material's own base_color is sufficient (procedural primitives);
//! mesh imports override with per-voxel texture-baked RGB.
//!
//! Allocation discipline: bump-allocated with a free list of reclaimed
//! ranges, opportunistic tail coalescing on deallocate. Procedural
//! re-voxelization reclaims storage predictably.

use crate::leaf_attr::LeafAttr;

/// A flat pool of [`LeafAttr`] entries indexed by slot number, with a
/// parallel color array at the same index.
pub struct LeafAttrPool {
    data: Vec<LeafAttr>,
    /// Parallel color array: packed R|G|B|A u32 (A reserved / intensity).
    /// 0 = no override, fall back to material base_color.
    colors: Vec<u32>,
    /// Next unallocated slot (bump pointer).
    next_free: u32,
    /// Free list of reclaimed ranges — `(start, count)` pairs.
    free_list: Vec<(u32, u32)>,
}

impl LeafAttrPool {
    /// Create with the given initial capacity (number of entries).
    pub fn new(capacity: u32) -> Self {
        Self {
            data: vec![LeafAttr::EMPTY; capacity as usize],
            colors: vec![0u32; capacity as usize],
            next_free: 0,
            free_list: Vec::new(),
        }
    }

    /// Allocate `count` contiguous slots. First-fit in free list, else bump.
    pub fn allocate_range(&mut self, count: u32) -> Option<Vec<u32>> {
        if count == 0 {
            return Some(Vec::new());
        }

        if let Some(idx) = self.free_list.iter().position(|(_, c)| *c >= count) {
            let (start, free_count) = self.free_list[idx];
            if free_count == count {
                self.free_list.swap_remove(idx);
            } else {
                self.free_list[idx] = (start + count, free_count - count);
            }
            return Some((start..start + count).collect());
        }

        let start = self.next_free;
        let end = start.checked_add(count)?;
        if end as usize > self.data.len() {
            let new_cap = (self.data.len() as u32 * 2).max(end);
            self.grow(new_cap);
        }
        self.next_free = end;
        Some((start..end).collect())
    }

    /// Allocate a single slot.
    pub fn allocate(&mut self) -> Option<u32> {
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
            let new_cap = (self.data.len() as u32).checked_mul(2)?;
            self.grow(new_cap);
        }
        let slot = self.next_free;
        self.next_free += 1;
        Some(slot)
    }

    /// Return a contiguous range for reuse. Tail ranges shrink `next_free`
    /// directly; interior ranges go on the free list. Clears the attr and
    /// color at freed slots so stale data doesn't leak.
    pub fn deallocate_range(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let end = (start + count) as usize;
        if end > self.data.len() {
            return;
        }
        for s in start as usize..end {
            self.data[s] = LeafAttr::EMPTY;
            self.colors[s] = 0;
        }
        if start + count == self.next_free {
            self.next_free = start;
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
            self.free_list.push((start, count));
        }
    }

    #[inline]
    pub fn get(&self, slot: u32) -> &LeafAttr { &self.data[slot as usize] }

    #[inline]
    pub fn get_mut(&mut self, slot: u32) -> &mut LeafAttr { &mut self.data[slot as usize] }

    #[inline]
    pub fn color(&self, slot: u32) -> u32 { self.colors[slot as usize] }

    #[inline]
    pub fn set_color(&mut self, slot: u32, packed: u32) {
        self.colors[slot as usize] = packed;
    }

    #[inline]
    pub fn allocated_count(&self) -> u32 { self.next_free }

    #[inline]
    pub fn capacity(&self) -> u32 { self.data.len() as u32 }

    pub fn grow(&mut self, new_cap: u32) {
        if new_cap as usize <= self.data.len() {
            return;
        }
        self.data.resize(new_cap as usize, LeafAttr::EMPTY);
        self.colors.resize(new_cap as usize, 0);
    }

    /// Raw byte slice of the allocated attr region (for GPU upload).
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.next_free as usize;
        if count == 0 {
            return &[];
        }
        let ptr = self.data.as_ptr() as *const u8;
        let byte_len = count * std::mem::size_of::<LeafAttr>();
        unsafe { std::slice::from_raw_parts(ptr, byte_len) }
    }

    /// Raw byte slice of the parallel color array (for GPU upload).
    pub fn color_bytes(&self) -> &[u8] {
        let count = self.next_free as usize;
        if count == 0 {
            return &[];
        }
        bytemuck::cast_slice(&self.colors[..count])
    }

    /// Raw slice of allocated entries.
    pub fn as_slice(&self) -> &[LeafAttr] {
        &self.data[..self.next_free as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn new_pool_is_empty() {
        let pool = LeafAttrPool::new(100);
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.capacity(), 100);
    }

    #[test]
    fn allocate_and_retrieve() {
        let mut pool = LeafAttrPool::new(10);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = LeafAttr::new(Vec3::Y, 42);
        pool.set_color(slot, 0x00112233);
        let back = pool.get(slot);
        assert_eq!(back.material_primary, 42);
        assert_eq!(pool.color(slot), 0x00112233);
    }

    #[test]
    fn grow_preserves_data() {
        let mut pool = LeafAttrPool::new(4);
        let s = pool.allocate().unwrap();
        *pool.get_mut(s) = LeafAttr::new(Vec3::X, 99);
        pool.set_color(s, 0xAABBCCDD);
        pool.grow(100);
        assert_eq!(pool.capacity(), 100);
        assert_eq!(pool.get(s).material_primary, 99);
        assert_eq!(pool.color(s), 0xAABBCCDD);
    }

    #[test]
    fn bytes_roundtrip() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(3).unwrap();
        *pool.get_mut(0) = LeafAttr::new(Vec3::X, 1);
        *pool.get_mut(1) = LeafAttr::new(Vec3::Y, 2);
        *pool.get_mut(2) = LeafAttr::new(Vec3::Z, 3);
        let bytes = pool.as_bytes();
        assert_eq!(bytes.len(), 3 * 8);
        let reread: &[LeafAttr] = bytemuck::cast_slice(bytes);
        assert_eq!(reread[0].material_primary, 1);
        assert_eq!(reread[2].material_primary, 3);
    }

    #[test]
    fn color_bytes_roundtrip() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(3).unwrap();
        pool.set_color(0, 0x11);
        pool.set_color(1, 0x22);
        pool.set_color(2, 0x33);
        let cb = pool.color_bytes();
        assert_eq!(cb.len(), 3 * 4);
        let back: &[u32] = bytemuck::cast_slice(cb);
        assert_eq!(back, &[0x11, 0x22, 0x33]);
    }

    #[test]
    fn deallocate_tail_shrinks_next_free() {
        let mut pool = LeafAttrPool::new(16);
        pool.allocate_range(10).unwrap();
        pool.deallocate_range(5, 5);
        assert_eq!(pool.allocated_count(), 5);
    }

    #[test]
    fn deallocate_interior_goes_to_free_list_and_reuses() {
        let mut pool = LeafAttrPool::new(100);
        let a = pool.allocate_range(10).unwrap();
        let _b = pool.allocate_range(5).unwrap();
        pool.deallocate_range(a[0], 10);
        assert_eq!(pool.allocated_count(), 15);
        let reused = pool.allocate_range(10).unwrap();
        assert_eq!(reused[0], 0);
        assert_eq!(pool.allocated_count(), 15);
    }
}
