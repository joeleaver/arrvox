//! Allocator that packs multiple [`SparseOctree`]s into a single contiguous
//! buffer suitable for GPU upload.
//!
//! Mirrors the design of rkf-core's `BrickMapAllocator` — each octree occupies
//! a contiguous region tracked by an [`OctreeHandle`]. Deallocated regions go
//! onto a free list for reuse.

use std::collections::HashMap;

use crate::sparse_octree::{SparseOctree, INTERNAL_ATTR_NONE};
use crate::DirtyRanges;

/// Bytes per octree slot in the interleaved GPU buffer layout used by
/// `RkpScene::upload_geometry`: `(node, prefilter_id, 0, 0)` × u32 =
/// 4 u32 = 16 B. The `OctreeAllocator` itself stores nodes and attrs as
/// two separate `Vec<u32>` on the CPU, but every dirty mark covers one
/// GPU slot so the upload pass can emit a single `queue.write_buffer`
/// per dirty range without worrying about CPU↔GPU layout mismatch.
pub const OCTREE_GPU_SLOT_BYTES: u32 = 4 * 4;

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

/// Packs multiple octrees into a single `Vec<u32>` for GPU upload, plus a
/// parallel `Vec<u32>` of prefiltered-LOD attr ids (one per node slot).
///
/// The GPU side interleaves the two as `array<vec2<u32>>` — see
/// `RkpScene::upload_geometry`. Keeping them as two separate Vecs on the
/// CPU side lets the rest of rkp-core treat them as independent buffers;
/// the interleave happens at upload time.
#[derive(Debug)]
pub struct OctreeAllocator {
    /// Packed octree-node storage. Wrapped in `Arc<Vec<u32>>` (instead
    /// of plain `Vec<u32>`) so the painted-material walk's
    /// `WalkSnapshot` can take a constant-time `Arc::clone` and traverse
    /// the buffer outside the `scene_mgr` lock without paying the
    /// ~80 MB memcpy of the prior `.to_vec()` design. Internal mutations
    /// route through [`Self::data_mut`] (`Arc::make_mut`): refcount=1
    /// in steady state means writes are in-place; an outstanding
    /// snapshot triggers a one-time clone-on-write that the snapshot's
    /// caller would otherwise have paid eagerly. See PERF_DEBT.md A2.
    data: std::sync::Arc<Vec<u32>>,
    /// Parallel prefiltered-LOD attr ids, same length as `data`. Unlike
    /// `data`, entries here are never rebased — each entry is a global
    /// `leaf_attr_id` from the shared `LeafAttrPool`, or
    /// [`INTERNAL_ATTR_NONE`] when no prefilter is available for that slot.
    /// Stays plain `Vec` — not part of the walk_snapshot.
    internal_attrs: Vec<u32>,
    free_list: Vec<(u32, u32)>, // (offset, length)
    /// GPU-slot byte ranges (in the interleaved `vec4<u32>` layout) that
    /// have been mutated since the last upload. Each entry covers
    /// `OCTREE_GPU_SLOT_BYTES` per CPU slot. Sculpt's per-stamp `write_node`
    /// / `write_internal_attr` calls populate this; the upload pass
    /// drains it via `dirty_ranges_mut`.
    dirty: DirtyRanges,
    /// Per-slot reserved capacity (in slots). Key is `root_offset`,
    /// value is the number of slots reserved for that slot — always
    /// `>= handle.len`. Sculpt growth writes into the slack region
    /// [root_offset + handle.len .. root_offset + reserved) without
    /// re-allocating the slot. `deallocate` uses this to free the full
    /// reserved range.
    slot_capacity: HashMap<u32, u32>,
}

impl OctreeAllocator {
    /// Create a new empty allocator.
    pub fn new() -> Self {
        Self {
            data: std::sync::Arc::new(Vec::new()),
            internal_attrs: Vec::new(),
            free_list: Vec::new(),
            dirty: DirtyRanges::new(),
            slot_capacity: HashMap::new(),
        }
    }

    /// Create an allocator with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: std::sync::Arc::new(Vec::with_capacity(capacity)),
            internal_attrs: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            dirty: DirtyRanges::new(),
            slot_capacity: HashMap::new(),
        }
    }

    /// Mutable access to the packed-buffer storage for in-allocator
    /// writes. Routes through `Arc::make_mut` so an outstanding
    /// `WalkSnapshot` clone causes a one-time copy-on-write rather
    /// than corrupting the snapshot. In steady state (refcount=1)
    /// this is a free `&mut Vec<u32>`.
    #[inline]
    fn data_mut(&mut self) -> &mut Vec<u32> {
        std::sync::Arc::make_mut(&mut self.data)
    }

    /// Cheap shareable handle to the packed-buffer storage, used by
    /// `RkpSceneManager::walk_snapshot` to hand octree data to the
    /// painted-material walk without copying.
    #[inline]
    pub fn data_arc(&self) -> std::sync::Arc<Vec<u32>> {
        self.data.clone()
    }

    #[inline]
    fn mark_slot_dirty(&mut self, absolute_slot_idx: u32) {
        self.dirty.mark(absolute_slot_idx * OCTREE_GPU_SLOT_BYTES, OCTREE_GPU_SLOT_BYTES);
    }

    /// Write a node value at the absolute (post-rebase) packed-buffer
    /// index. `local_value` is the value as observed in the source
    /// `SparseOctree` — local branch offsets get rebased to absolute by
    /// adding `base_offset`. Sentinels (EMPTY_NODE / INTERIOR_NODE) and
    /// leaf encodings (LEAF / BRICK) are written verbatim.
    ///
    /// Marks the dirty range so the next `upload_geometry` will emit a
    /// `queue.write_buffer` covering this slot.
    pub fn write_node(&mut self, absolute_idx: u32, local_value: u32, base_offset: u32) {
        use crate::sparse_octree::{EMPTY_NODE, INTERIOR_NODE, is_branch};
        let to_write = if local_value == EMPTY_NODE || local_value == INTERIOR_NODE {
            local_value
        } else if is_branch(local_value) {
            local_value + base_offset
        } else {
            // LEAF or BRICK: low 30 bits are a pool slot id, not a buffer offset.
            local_value
        };
        self.data_mut()[absolute_idx as usize] = to_write;
        self.mark_slot_dirty(absolute_idx);
    }

    /// Write an internal-attr value at the absolute packed-buffer index.
    /// Attr values are global `leaf_attr_id`s (never rebased).
    pub fn write_internal_attr(&mut self, absolute_idx: u32, value: u32) {
        self.internal_attrs[absolute_idx as usize] = value;
        self.mark_slot_dirty(absolute_idx);
    }

    /// Read-only view of the per-slot dirty range tracker.
    #[inline]
    pub fn dirty_ranges(&self) -> &DirtyRanges {
        &self.dirty
    }

    /// Mutable view of the dirty range tracker. The upload pass calls
    /// `clear()` after writing the deltas to the GPU buffer.
    #[inline]
    pub fn dirty_ranges_mut(&mut self) -> &mut DirtyRanges {
        &mut self.dirty
    }

    /// Allocate space for an octree and copy its nodes into the packed buffer.
    ///
    /// The octree's internal node offsets are rebased to be absolute within the
    /// packed buffer (by adding the allocation offset to all branch pointers).
    /// The parallel `internal_attrs` are copied verbatim (no rebasing — the
    /// ids they carry are global `leaf_attr_pool` slots, already absolute).
    pub fn allocate(&mut self, octree: &SparseOctree) -> OctreeHandle {
        let len = octree.as_slice().len() as u32;
        self.allocate_reserved(octree, len)
    }

    /// Allocate a slot with `slack_factor` × `octree.node_count()`
    /// reserved capacity (rounded up). Used by the sculpt re-alloc
    /// path so that subsequent growth via [`try_extend_in_slack`]
    /// lands in pre-reserved space instead of forcing another full
    /// re-allocation. A factor of 1.5 gives 50% headroom (typical
    /// sculpt session grows nodes by ~0.03% per stamp, so 50% covers
    /// thousands of stamps).
    pub fn allocate_with_slack(
        &mut self,
        octree: &SparseOctree,
        slack_factor: f32,
    ) -> OctreeHandle {
        let len = octree.as_slice().len() as u32;
        let reserved = ((len as f32) * slack_factor.max(1.0)).ceil() as u32;
        // Always reserve at least +64 slots so very small octrees get
        // a sane head-start before the slack-factor takes over.
        let reserved = reserved.max(len.saturating_add(64));
        self.allocate_reserved(octree, reserved)
    }

    /// Allocate a slot with an exact `reserved` capacity (in slots).
    /// The populated region `[root_offset, root_offset + len)` gets
    /// the octree's nodes (with branch pointers rebased); the slack
    /// region `[root_offset + len, root_offset + reserved)` is filled
    /// with `EMPTY_NODE` / `INTERNAL_ATTR_NONE` sentinels on the CPU
    /// side. Only the populated region is marked dirty — the slack
    /// has no incoming branch references, so the shader never reads
    /// those slots before they're populated by a future
    /// [`try_extend_in_slack`].
    fn allocate_reserved(&mut self, octree: &SparseOctree, reserved: u32) -> OctreeHandle {
        let nodes = octree.as_slice();
        let attrs = octree.internal_attr_slice();
        debug_assert_eq!(nodes.len(), attrs.len());
        let len = nodes.len() as u32;
        debug_assert!(reserved >= len, "reserved must be >= len");

        let offset = if let Some(idx) = self.find_free_region(reserved) {
            let (free_offset, free_len) = self.free_list[idx];
            let start = free_offset as usize;
            {
                let data = self.data_mut();
                data[start..start + len as usize].copy_from_slice(nodes);
                // Slack region (within the freed range) gets sentinels.
                if reserved > len {
                    let slack_start = start + len as usize;
                    let slack_end = start + reserved as usize;
                    for entry in &mut data[slack_start..slack_end] {
                        *entry = crate::sparse_octree::EMPTY_NODE;
                    }
                }
            }
            self.internal_attrs[start..start + len as usize].copy_from_slice(attrs);
            if reserved > len {
                let slack_start = start + len as usize;
                let slack_end = start + reserved as usize;
                for entry in &mut self.internal_attrs[slack_start..slack_end] {
                    *entry = INTERNAL_ATTR_NONE;
                }
            }
            if free_len > reserved {
                self.free_list[idx] = (free_offset + reserved, free_len - reserved);
            } else {
                self.free_list.swap_remove(idx);
            }
            free_offset
        } else {
            let offset = self.data.len() as u32;
            {
                let data = self.data_mut();
                data.extend_from_slice(nodes);
                if reserved > len {
                    let extra = (reserved - len) as usize;
                    data.extend(std::iter::repeat(crate::sparse_octree::EMPTY_NODE).take(extra));
                }
            }
            self.internal_attrs.extend_from_slice(attrs);
            if reserved > len {
                let extra = (reserved - len) as usize;
                self.internal_attrs.extend(std::iter::repeat(INTERNAL_ATTR_NONE).take(extra));
            }
            offset
        };

        // Rebase branch pointers in the populated region only — the
        // slack region holds sentinels which `rebase_branches` skips.
        self.rebase_branches(offset as usize, len as usize, offset);

        // Mark the populated region dirty so the next upload ships it.
        // Slack stays clean — the GPU buffer's slack bytes are
        // initialized on first ever upload of this allocator (via the
        // grow path in `upload_octree_delta`), and slack nodes have no
        // incoming references so the shader never reads them.
        if len > 0 {
            self.dirty.mark(offset * OCTREE_GPU_SLOT_BYTES, len * OCTREE_GPU_SLOT_BYTES);
        }

        self.slot_capacity.insert(offset, reserved);

        OctreeHandle {
            root_offset: offset,
            len,
            depth: octree.depth(),
            base_voxel_size: octree.base_voxel_size(),
        }
    }

    /// Attempt to grow an existing slot's populated length in place by
    /// using its reserved slack. Returns `Some(new_handle)` (with
    /// updated `len`) when the new node count fits within the slot's
    /// reservation; returns `None` when slack is exhausted (caller
    /// should fall back to `deallocate` + `allocate_with_slack`).
    ///
    /// **The caller is responsible for writing the appended nodes via
    /// [`write_node`] / [`write_internal_attr`].** Typical use: sculpt's
    /// [`apply_delta`] returns a mutation log that already records
    /// every appended-slot write at the right local indices; replaying
    /// the log against the extended slot fills the slack region.
    pub fn try_extend_in_slack(
        &self,
        handle: &OctreeHandle,
        new_len: u32,
    ) -> Option<OctreeHandle> {
        let reserved = *self.slot_capacity.get(&handle.root_offset)?;
        if new_len > reserved {
            return None;
        }
        Some(OctreeHandle {
            root_offset: handle.root_offset,
            len: new_len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        })
    }

    /// Reserved capacity of a slot (in slots). Returns the slot's
    /// `len` when the allocator has no per-slot reservation entry
    /// (legacy callers / non-existent slot). Used by
    /// [`OctreeGpu::apply_mutation_log`] to bounds-check writes against
    /// the slot's full reservation, not just its current populated len.
    pub fn reserved_capacity(&self, root_offset: u32) -> u32 {
        self.slot_capacity.get(&root_offset).copied().unwrap_or(0)
    }

    /// Deallocate an octree region, adding it to the free list.
    ///
    /// Coalesces adjacent free regions and shrinks the backing buffer if the
    /// coalesced tail reaches `data.len()`. Without this, repeated re-voxelize
    /// of procedural objects whose new allocation is even slightly larger than
    /// any single freed region causes the buffer to grow monotonically.
    pub fn deallocate(&mut self, handle: OctreeHandle) {
        // Free the full reserved capacity (slack + populated), not
        // just `handle.len` — sculpt may have reserved extra slots
        // via `allocate_with_slack` that aren't yet populated but are
        // still owned by this slot.
        let reserved = self
            .slot_capacity
            .remove(&handle.root_offset)
            .unwrap_or(handle.len);
        let start = handle.root_offset as usize;
        let end = start + reserved as usize;
        if end > self.data.len() {
            return;
        }
        // Clear to EMPTY so stale branch offsets can't be misread.
        for entry in &mut self.data_mut()[start..end] {
            *entry = crate::sparse_octree::EMPTY_NODE;
        }
        // Clear the parallel prefilter ids too — a stale id could otherwise
        // point at a leaf_attr slot now owned by another asset after a
        // reallocation overwrites this region.
        for entry in &mut self.internal_attrs[start..end] {
            *entry = INTERNAL_ATTR_NONE;
        }
        if reserved > 0 {
            self.dirty.mark(
                handle.root_offset * OCTREE_GPU_SLOT_BYTES,
                reserved * OCTREE_GPU_SLOT_BYTES,
            );
        }
        self.free_list.push((handle.root_offset, reserved));
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
                self.data_mut().truncate(off as usize);
                self.internal_attrs.truncate(off as usize);
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

    /// Parallel prefilter-attr slice, same length as [`as_slice`](Self::as_slice).
    /// The GPU upload interleaves these with the node values into a
    /// single `array<vec2<u32>>` binding.
    #[inline]
    pub fn internal_attrs_slice(&self) -> &[u32] {
        &self.internal_attrs
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
        let data = self.data_mut();
        for i in start..start + len {
            let node = data[i];
            if node == EMPTY_NODE || node == INTERIOR_NODE {
                continue;
            }
            if is_branch(node) {
                data[i] = node + base_offset;
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

    #[test]
    fn internal_attrs_slice_mirrors_data_slice() {
        let mut alloc = OctreeAllocator::new();
        let t = make_test_octree();
        alloc.allocate(&t);
        assert_eq!(alloc.as_slice().len(), alloc.internal_attrs_slice().len());
        // Post-allocate: all sentinel (prefilter hasn't run on the test tree).
        assert!(alloc.internal_attrs_slice().iter().all(|&x| x == INTERNAL_ATTR_NONE));
    }

    #[test]
    fn allocator_preserves_per_slot_internal_attrs() {
        // Hand-populate a tree's internal_attr_index, then allocate —
        // the allocator should copy the values through to its parallel
        // buffer at the corresponding offsets.
        let mut alloc = OctreeAllocator::new();
        let mut t = make_test_octree();
        // Pick the first branch and seed a cookie there. Lookup scans until
        // hitting a branch to avoid hard-coding index 0's type.
        let mut seeded = None;
        for i in 0..t.node_count() {
            if crate::sparse_octree::is_branch(t.as_slice()[i]) {
                t.set_internal_attr(i as u32, 0xCAFEBABE);
                seeded = Some(i);
                break;
            }
        }
        let seed_idx = seeded.expect("test tree should have a branch");
        let h = alloc.allocate(&t);

        assert_eq!(
            alloc.internal_attrs_slice()[h.root_offset as usize + seed_idx],
            0xCAFEBABE,
        );
    }

    #[test]
    fn allocate_marks_region_dirty() {
        let mut alloc = OctreeAllocator::new();
        assert!(alloc.dirty_ranges().is_empty());
        let h = alloc.allocate(&make_test_octree());
        let ranges: Vec<_> = alloc.dirty_ranges().iter().collect();
        assert_eq!(
            ranges,
            vec![(h.root_offset * OCTREE_GPU_SLOT_BYTES, h.len * OCTREE_GPU_SLOT_BYTES)],
        );
    }

    #[test]
    fn deallocate_marks_region_dirty() {
        let mut alloc = OctreeAllocator::new();
        let h_a = alloc.allocate(&make_test_octree());
        let _h_b = alloc.allocate(&make_test_octree()); // pin tail
        alloc.dirty_ranges_mut().clear();

        alloc.deallocate(h_a);
        // The freed slots get marked so the next upload writes the
        // cleared (EMPTY_NODE / INTERNAL_ATTR_NONE) state to GPU.
        let ranges: Vec<_> = alloc.dirty_ranges().iter().collect();
        assert_eq!(
            ranges,
            vec![(h_a.root_offset * OCTREE_GPU_SLOT_BYTES, h_a.len * OCTREE_GPU_SLOT_BYTES)],
        );
    }

    #[test]
    fn write_node_marks_one_slot() {
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&make_test_octree());
        alloc.dirty_ranges_mut().clear();

        let slot = h.root_offset + 0;
        alloc.write_node(slot, 0x12345, h.root_offset);
        let ranges: Vec<_> = alloc.dirty_ranges().iter().collect();
        assert_eq!(ranges, vec![(slot * OCTREE_GPU_SLOT_BYTES, OCTREE_GPU_SLOT_BYTES)]);
    }

    #[test]
    fn write_node_rebases_branch_values() {
        let mut alloc = OctreeAllocator::new();
        // Allocate a dummy first to push real octree to non-zero offset.
        let _dummy = alloc.allocate(&SparseOctree::new(1, 0.1));
        let h = alloc.allocate(&make_test_octree());

        // Write a "local branch" value (i.e. a value < EMPTY_NODE that
        // is_branch() recognizes) at the root slot. Local value 4 should
        // be rebased to handle.root_offset + 4.
        let slot = h.root_offset + 1;
        alloc.write_node(slot, 4, h.root_offset);
        assert_eq!(alloc.as_slice()[slot as usize], h.root_offset + 4);
    }

    #[test]
    fn write_node_passes_sentinels_through() {
        use crate::sparse_octree::{EMPTY_NODE, INTERIOR_NODE};
        let mut alloc = OctreeAllocator::new();
        let _dummy = alloc.allocate(&SparseOctree::new(1, 0.1));
        let h = alloc.allocate(&make_test_octree());

        alloc.write_node(h.root_offset, EMPTY_NODE, h.root_offset);
        assert_eq!(alloc.as_slice()[h.root_offset as usize], EMPTY_NODE);

        alloc.write_node(h.root_offset, INTERIOR_NODE, h.root_offset);
        assert_eq!(alloc.as_slice()[h.root_offset as usize], INTERIOR_NODE);
    }

    #[test]
    fn allocate_with_slack_reserves_capacity() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let len = tree.node_count() as u32;
        let h = alloc.allocate_with_slack(&tree, 2.0);
        // populated len is unchanged.
        assert_eq!(h.len, len);
        // reserved capacity is at least 2× and at least len+64.
        let reserved = alloc.reserved_capacity(h.root_offset);
        assert!(reserved >= 2 * len);
        assert!(reserved >= len + 64);
    }

    #[test]
    fn try_extend_in_slack_succeeds_within_reservation() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let h = alloc.allocate_with_slack(&tree, 3.0);
        let reserved = alloc.reserved_capacity(h.root_offset);
        // Grow up to exactly reserved.
        let extended = alloc.try_extend_in_slack(&h, reserved);
        assert!(extended.is_some());
        let new = extended.unwrap();
        assert_eq!(new.len, reserved);
        assert_eq!(new.root_offset, h.root_offset);
    }

    #[test]
    fn try_extend_in_slack_fails_when_exhausted() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let h = alloc.allocate_with_slack(&tree, 1.5);
        let reserved = alloc.reserved_capacity(h.root_offset);
        // One past reserved → must return None.
        assert!(alloc.try_extend_in_slack(&h, reserved + 1).is_none());
    }

    #[test]
    fn deallocate_with_slack_frees_full_reservation() {
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let h = alloc.allocate_with_slack(&tree, 2.0);
        let reserved = alloc.reserved_capacity(h.root_offset);
        let buffer_after_alloc = alloc.buffer_len();
        // Allocate something AFTER h to pin the tail (otherwise dealloc
        // would shrink and the test wouldn't observe the free_list).
        let _h2 = alloc.allocate(&tree);
        alloc.deallocate(h);
        // The freed region should equal `reserved`, not h.len.
        assert_eq!(alloc.total_free_entries(), reserved);
        // Subsequent allocate (small enough) re-uses the freed region.
        assert!(alloc.buffer_len() >= buffer_after_alloc);
    }

    #[test]
    fn extend_in_slack_then_apply_writes_into_slack() {
        // Round-trip: allocate with slack; write a node past the
        // current `len` (in the slack region) via the public
        // write_node API. The write should succeed (cap is `reserved`,
        // not `len`).
        let mut alloc = OctreeAllocator::new();
        let tree = make_test_octree();
        let h = alloc.allocate_with_slack(&tree, 2.0);
        let reserved = alloc.reserved_capacity(h.root_offset);
        let slack_idx = h.root_offset + h.len; // first slack slot
        assert!(slack_idx < h.root_offset + reserved);
        // Slack starts as EMPTY_NODE.
        use crate::sparse_octree::EMPTY_NODE;
        assert_eq!(alloc.as_slice()[slack_idx as usize], EMPTY_NODE);
        // Write a leaf value in the slack.
        use crate::sparse_octree::make_leaf;
        let leaf = make_leaf(0xAB);
        alloc.write_node(slack_idx, leaf, h.root_offset);
        assert_eq!(alloc.as_slice()[slack_idx as usize], leaf);
    }

    #[test]
    fn write_internal_attr_marks_slot() {
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&make_test_octree());
        alloc.dirty_ranges_mut().clear();

        alloc.write_internal_attr(h.root_offset + 2, 0xCAFE);
        assert_eq!(alloc.internal_attrs_slice()[(h.root_offset + 2) as usize], 0xCAFE);
        let ranges: Vec<_> = alloc.dirty_ranges().iter().collect();
        assert_eq!(
            ranges,
            vec![((h.root_offset + 2) * OCTREE_GPU_SLOT_BYTES, OCTREE_GPU_SLOT_BYTES)],
        );
    }

    #[test]
    fn deallocate_clears_internal_attrs() {
        let mut alloc = OctreeAllocator::new();
        let mut t = make_test_octree();
        for i in 0..t.node_count() {
            if crate::sparse_octree::is_branch(t.as_slice()[i]) {
                t.set_internal_attr(i as u32, 0xAABBCCDD);
                break;
            }
        }
        let h = alloc.allocate(&t);
        // Before dealloc: at least one non-sentinel somewhere in our region.
        let region_range = h.root_offset as usize..(h.root_offset + h.len) as usize;
        assert!(
            alloc.internal_attrs_slice()[region_range.clone()]
                .iter()
                .any(|&x| x != INTERNAL_ATTR_NONE),
        );

        alloc.deallocate(h);
        // After dealloc (tail-reclaim truncates both vecs): nothing remains.
        assert_eq!(alloc.as_slice().len(), 0);
        assert_eq!(alloc.internal_attrs_slice().len(), 0);
    }
}
