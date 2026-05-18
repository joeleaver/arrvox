//! Flat pool of [`LeafAttr`] entries with parallel per-leaf color and
//! per-leaf skinning weights.
//!
//! Each pool slot holds a [`LeafAttr`] (material IDs + normal, 8 B), a
//! packed u32 color (4 B), and a [`BoneVoxel`] (4 bone indices + 4
//! weights = 8 B), all at the same slot index. Color is 0 for leaves
//! where the material's own base_color is sufficient; bone weights are
//! a zeroed `BoneVoxel` for unskinned assets.
//!
//! Allocation discipline: bump-allocated with a free list of reclaimed
//! ranges, opportunistic tail coalescing on deallocate. Procedural
//! re-voxelization reclaims storage predictably.

use crate::companion::BoneVoxel;

use crate::leaf_attr::LeafAttr;
use crate::DirtyRanges;

const LEAF_ATTR_STRIDE: u32 = std::mem::size_of::<LeafAttr>() as u32;
const COLOR_STRIDE: u32 = std::mem::size_of::<u32>() as u32;
const BONE_STRIDE: u32 = std::mem::size_of::<BoneVoxel>() as u32;

/// A flat pool of [`LeafAttr`] entries indexed by slot number, with
/// parallel color and bone-weight arrays at the same indices.
pub struct LeafAttrPool {
    /// Per-slot LeafAttr storage. Wrapped in `Arc<Vec<LeafAttr>>` (not
    /// plain `Vec<LeafAttr>`) so the painted-material walk's
    /// `WalkSnapshot` can share the buffer via constant-time
    /// `Arc::clone` instead of paying the ~65 MB memcpy of the prior
    /// `.to_vec()` design. Internal mutations route through
    /// [`Self::data_mut`] (`Arc::make_mut`): refcount=1 in steady state,
    /// so writes are in-place; an outstanding snapshot triggers a
    /// one-time clone-on-write that the next snapshot's caller would
    /// have paid anyway. See PERF_DEBT.md A2. `colors` and `bones` stay
    /// plain `Vec` — they aren't part of the walk_snapshot.
    data: std::sync::Arc<Vec<LeafAttr>>,
    /// Parallel color array: packed R|G|B|A u32 (A reserved / intensity).
    /// 0 = no override, fall back to material base_color.
    colors: Vec<u32>,
    /// Parallel bone-weight array. Default `BoneVoxel` is zero indices +
    /// zero weights — shader treats it as "no skinning influence" so
    /// unskinned assets cost nothing beyond the 8 B per slot.
    bones: Vec<BoneVoxel>,
    /// Next unallocated slot (bump pointer).
    next_free: u32,
    /// Free list of reclaimed ranges — `(start, count)` pairs.
    free_list: Vec<(u32, u32)>,
    /// Byte ranges in `data` mutated since the last GPU upload.
    dirty_attrs: DirtyRanges,
    /// Byte ranges in `colors` mutated since the last GPU upload.
    dirty_colors: DirtyRanges,
    /// Byte ranges in `bones` mutated since the last GPU upload.
    dirty_bones: DirtyRanges,
}

impl LeafAttrPool {
    /// Create with the given initial capacity (number of entries).
    pub fn new(capacity: u32) -> Self {
        Self {
            data: std::sync::Arc::new(vec![LeafAttr::EMPTY; capacity as usize]),
            colors: vec![0u32; capacity as usize],
            bones: vec![BoneVoxel::default(); capacity as usize],
            next_free: 0,
            free_list: Vec::new(),
            dirty_attrs: DirtyRanges::new(),
            dirty_colors: DirtyRanges::new(),
            dirty_bones: DirtyRanges::new(),
        }
    }

    /// Mutable access to the inner `data` storage for in-pool writes.
    /// Routes through `Arc::make_mut` so an outstanding `WalkSnapshot`
    /// clone causes a one-time copy-on-write rather than corrupting
    /// the snapshot. In steady state (refcount=1) this is a free
    /// `&mut Vec<LeafAttr>`.
    #[inline]
    fn data_mut(&mut self) -> &mut Vec<LeafAttr> {
        std::sync::Arc::make_mut(&mut self.data)
    }

    /// Cheap shareable handle to the LeafAttr data, used by
    /// `ArvxSceneManager::walk_snapshot` to hand pool storage to the
    /// painted-material walk without copying.
    #[inline]
    pub fn data_arc(&self) -> std::sync::Arc<Vec<LeafAttr>> {
        self.data.clone()
    }

    #[inline]
    fn mark_slot_range_all(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        self.dirty_attrs.mark(start * LEAF_ATTR_STRIDE, count * LEAF_ATTR_STRIDE);
        self.dirty_colors.mark(start * COLOR_STRIDE, count * COLOR_STRIDE);
        self.dirty_bones.mark(start * BONE_STRIDE, count * BONE_STRIDE);
    }

    #[inline]
    fn mark_attr_slot(&mut self, slot: u32) {
        self.dirty_attrs.mark(slot * LEAF_ATTR_STRIDE, LEAF_ATTR_STRIDE);
    }

    #[inline]
    fn mark_color_slot(&mut self, slot: u32) {
        self.dirty_colors.mark(slot * COLOR_STRIDE, COLOR_STRIDE);
    }

    #[inline]
    fn mark_bone_slot(&mut self, slot: u32) {
        self.dirty_bones.mark(slot * BONE_STRIDE, BONE_STRIDE);
    }

    /// Read-only views of the per-pool dirty range trackers.
    #[inline]
    pub fn dirty_attrs(&self) -> &DirtyRanges { &self.dirty_attrs }
    #[inline]
    pub fn dirty_colors(&self) -> &DirtyRanges { &self.dirty_colors }
    #[inline]
    pub fn dirty_bones(&self) -> &DirtyRanges { &self.dirty_bones }

    /// Mutable views of the per-pool dirty range trackers. The upload
    /// path calls `clear()` on these after writing the deltas.
    #[inline]
    pub fn dirty_attrs_mut(&mut self) -> &mut DirtyRanges { &mut self.dirty_attrs }
    #[inline]
    pub fn dirty_colors_mut(&mut self) -> &mut DirtyRanges { &mut self.dirty_colors }
    #[inline]
    pub fn dirty_bones_mut(&mut self) -> &mut DirtyRanges { &mut self.dirty_bones }

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
            self.mark_slot_range_all(start, count);
            return Some((start..start + count).collect());
        }

        let start = self.next_free;
        let end = start.checked_add(count)?;
        if end as usize > self.data.len() {
            let new_cap = (self.data.len() as u32 * 2).max(end);
            self.grow(new_cap);
        }
        self.next_free = end;
        self.mark_slot_range_all(start, count);
        Some((start..end).collect())
    }

    /// Reserve `count` contiguous slots by bumping past any current
    /// free-list entries. Returns the start index. Never reuses freed
    /// ranges, which keeps the returned range contiguous regardless of
    /// prior deallocations — important for asset loaders that need
    /// `(start, count)` to describe the range for later release.
    pub fn allocate_contiguous_bump(&mut self, count: u32) -> Option<u32> {
        if count == 0 { return Some(self.next_free); }
        let start = self.next_free;
        let end = start.checked_add(count)?;
        if end as usize > self.data.len() {
            let new_cap = (self.data.len() as u32 * 2).max(end);
            self.grow(new_cap);
        }
        self.next_free = end;
        self.mark_slot_range_all(start, count);
        Some(start)
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
            self.mark_slot_range_all(start, 1);
            return Some(start);
        }

        if (self.next_free as usize) >= self.data.len() {
            let new_cap = (self.data.len() as u32).checked_mul(2)?;
            self.grow(new_cap);
        }
        let slot = self.next_free;
        self.next_free += 1;
        self.mark_slot_range_all(slot, 1);
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
        let data = self.data_mut();
        for s in start as usize..end {
            data[s] = LeafAttr::EMPTY;
        }
        for s in start as usize..end {
            self.colors[s] = 0;
            self.bones[s] = BoneVoxel::default();
        }
        self.mark_slot_range_all(start, count);
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

    /// Mutable handle to the [`LeafAttr`] at `slot`. Conservatively
    /// marks the slot dirty — caller intent is mutation, and we cannot
    /// observe whether the caller actually writes.
    #[inline]
    pub fn get_mut(&mut self, slot: u32) -> &mut LeafAttr {
        self.mark_attr_slot(slot);
        &mut self.data_mut()[slot as usize]
    }

    #[inline]
    pub fn color(&self, slot: u32) -> u32 { self.colors[slot as usize] }

    #[inline]
    pub fn set_color(&mut self, slot: u32, packed: u32) {
        self.colors[slot as usize] = packed;
        self.mark_color_slot(slot);
    }

    #[inline]
    pub fn bone(&self, slot: u32) -> BoneVoxel { self.bones[slot as usize] }

    #[inline]
    pub fn set_bone(&mut self, slot: u32, bv: BoneVoxel) {
        self.bones[slot as usize] = bv;
        self.mark_bone_slot(slot);
    }

    #[inline]
    pub fn allocated_count(&self) -> u32 { self.next_free }

    #[inline]
    pub fn capacity(&self) -> u32 { self.data.len() as u32 }

    pub fn grow(&mut self, new_cap: u32) {
        if new_cap as usize <= self.data.len() {
            return;
        }
        self.data_mut().resize(new_cap as usize, LeafAttr::EMPTY);
        self.colors.resize(new_cap as usize, 0);
        self.bones.resize(new_cap as usize, BoneVoxel::default());
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

    /// Raw byte slice of the parallel bone-weight array (for GPU upload).
    /// Unskinned assets leave this zero-filled; the shader reads the
    /// per-object `is_skinned` flag to decide whether to consume it.
    pub fn bone_bytes(&self) -> &[u8] {
        let count = self.next_free as usize;
        if count == 0 {
            return &[];
        }
        bytemuck::cast_slice(&self.bones[..count])
    }

    /// Typed slice of the parallel bone-weight array. Indexed by the
    /// same slot id as [`as_slice`] / [`bone`]; unskinned slots stay
    /// zero-default. Used by the surface-mesh extractor to bake bone
    /// quads into `MeshVertex` at extract time.
    pub fn bones_as_slice(&self) -> &[BoneVoxel] {
        &self.bones[..self.next_free as usize]
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

    #[test]
    fn allocate_marks_all_pools() {
        let mut pool = LeafAttrPool::new(16);
        assert!(pool.dirty_attrs().is_empty());
        let slot = pool.allocate().unwrap();
        let a: Vec<_> = pool.dirty_attrs().iter().collect();
        let c: Vec<_> = pool.dirty_colors().iter().collect();
        let b: Vec<_> = pool.dirty_bones().iter().collect();
        assert_eq!(a, vec![(slot * LEAF_ATTR_STRIDE, LEAF_ATTR_STRIDE)]);
        assert_eq!(c, vec![(slot * COLOR_STRIDE, COLOR_STRIDE)]);
        assert_eq!(b, vec![(slot * BONE_STRIDE, BONE_STRIDE)]);
    }

    #[test]
    fn allocate_range_marks_full_range() {
        let mut pool = LeafAttrPool::new(16);
        let slots = pool.allocate_range(4).unwrap();
        let start = slots[0];
        let a: Vec<_> = pool.dirty_attrs().iter().collect();
        assert_eq!(a, vec![(start * LEAF_ATTR_STRIDE, 4 * LEAF_ATTR_STRIDE)]);
    }

    #[test]
    fn deallocate_range_marks_all_pools() {
        let mut pool = LeafAttrPool::new(16);
        let slots = pool.allocate_range(3).unwrap();
        pool.dirty_attrs_mut().clear();
        pool.dirty_colors_mut().clear();
        pool.dirty_bones_mut().clear();
        pool.deallocate_range(slots[0], 3);
        // Tail-shrink path still marks per spec.
        let a: Vec<_> = pool.dirty_attrs().iter().collect();
        assert_eq!(a, vec![(slots[0] * LEAF_ATTR_STRIDE, 3 * LEAF_ATTR_STRIDE)]);
    }

    #[test]
    fn get_mut_marks_attr_only() {
        let mut pool = LeafAttrPool::new(16);
        let slot = pool.allocate().unwrap();
        pool.dirty_attrs_mut().clear();
        pool.dirty_colors_mut().clear();
        pool.dirty_bones_mut().clear();
        let _ = pool.get_mut(slot);
        assert!(!pool.dirty_attrs().is_empty());
        assert!(pool.dirty_colors().is_empty());
        assert!(pool.dirty_bones().is_empty());
    }

    #[test]
    fn set_color_marks_color_only() {
        let mut pool = LeafAttrPool::new(16);
        let slot = pool.allocate().unwrap();
        pool.dirty_attrs_mut().clear();
        pool.dirty_colors_mut().clear();
        pool.dirty_bones_mut().clear();
        pool.set_color(slot, 0xCC);
        assert!(pool.dirty_attrs().is_empty());
        let c: Vec<_> = pool.dirty_colors().iter().collect();
        assert_eq!(c, vec![(slot * COLOR_STRIDE, COLOR_STRIDE)]);
        assert!(pool.dirty_bones().is_empty());
    }

    #[test]
    fn set_bone_marks_bone_only() {
        let mut pool = LeafAttrPool::new(16);
        let slot = pool.allocate().unwrap();
        pool.dirty_attrs_mut().clear();
        pool.dirty_colors_mut().clear();
        pool.dirty_bones_mut().clear();
        pool.set_bone(slot, BoneVoxel::default());
        assert!(pool.dirty_attrs().is_empty());
        assert!(pool.dirty_colors().is_empty());
        let b: Vec<_> = pool.dirty_bones().iter().collect();
        assert_eq!(b, vec![(slot * BONE_STRIDE, BONE_STRIDE)]);
    }
}
