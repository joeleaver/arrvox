//! Fixed-size voxel bricks — the flat leaf-level storage that terminates the
//! octree's deep levels.
//!
//! A brick is a 4³ = 64-cell flat array. The octree's deepest branches point
//! at a brick; the ray-march shader finds the brick via one octree-leaf read
//! and then advances through its cells with pure flat-array reads (no more
//! descent). This is what makes the post-bricks march dramatically faster
//! than per-step root-to-leaf descents.
//!
//! # Cell encoding
//!
//! Each cell is a `u32`:
//!
//! * [`BRICK_EMPTY`] (`0xFFFF_FFFF`) — nothing here. Ray continues.
//! * otherwise — a `leaf_attr_id` indexing into
//!   [`LeafAttrPool`](crate::leaf_attr_pool::LeafAttrPool), same as an octree
//!   leaf today.
//!
//! Bricks do not carry per-cell distance or occupancy-mask metadata; for a 4³
//! brick (256 bytes = 2 cache lines) the flat read is cheap enough that a
//! dedicated occupancy bitmap would cost more cache reads than it saves.
//! Larger bricks may want one.
//!
//! # Linear layout
//!
//! `index = x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM`. No Morton — for
//! 4³ bricks both layouts fit the same two cache lines, so the flat linear
//! index wins on shader simplicity.

/// Brick edge length in voxels. Each brick covers BRICK_DIM^3 voxels.
pub const BRICK_DIM: u32 = 4;

/// Cells per brick.
pub const BRICK_CELLS: u32 = BRICK_DIM * BRICK_DIM * BRICK_DIM;

/// Bytes per brick on the GPU (matches the shader's `array<u32>` view).
pub const BRICK_BYTES: u32 = BRICK_CELLS * 4;

/// Number of octree levels a brick replaces. log2(BRICK_DIM).
pub const BRICK_LEVELS: u8 = 2;

/// Cell sentinel: no voxel at this cell (empty air).
pub const BRICK_EMPTY: u32 = 0xFFFF_FFFFu32;

/// Cell sentinel: the cell sits inside the solid of a mesh-imported
/// asset — there's no visible surface here (the march skips past it
/// identically to `BRICK_EMPTY`), but it counts as *occupied mass* for
/// any neighborhood-averaging pass that needs volumetric information
/// (see `rkp-render/src/shaders/octree_march.wgsl::reconstruct_normal_surfacenet`).
///
/// Costs zero memory: occupies one of the 64 pre-allocated u32 slots in
/// a brick that was already allocated for its shell cells. Distinct
/// from `BRICK_EMPTY` purely so neighborhood kernels can distinguish
/// "inside bulk" from "exterior air" without needing the SDF at render
/// time.
pub const BRICK_INTERIOR: u32 = 0xFFFF_FFFDu32;

/// Convert (x, y, z) within a brick to its flat cell index.
#[inline]
pub const fn brick_flat_index(x: u32, y: u32, z: u32) -> u32 {
    x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM
}

/// A flat pool of bricks. Each brick occupies `BRICK_CELLS` consecutive u32
/// slots. A brick's `id` is its starting slot divided by `BRICK_CELLS`.
pub struct BrickPool {
    /// Flat cell storage: `data[brick_id * BRICK_CELLS + flat_index]`.
    data: Vec<u32>,
    /// Number of allocated bricks (bump pointer, in bricks, not cells).
    next_free_brick: u32,
    /// Free list of reclaimed brick ranges — `(brick_start, brick_count)`.
    free_list: Vec<(u32, u32)>,
}

impl BrickPool {
    /// Create with capacity for `capacity_bricks` bricks.
    pub fn new(capacity_bricks: u32) -> Self {
        let capacity_cells = (capacity_bricks as usize) * (BRICK_CELLS as usize);
        Self {
            data: vec![BRICK_EMPTY; capacity_cells],
            next_free_brick: 0,
            free_list: Vec::new(),
        }
    }

    /// Reserve `count` contiguous bricks by bumping past any freed entries.
    /// Returns the starting brick_id. Used by asset loaders that need a
    /// contiguous range so the octree's BRICK nodes can be remapped by a
    /// single offset add.
    pub fn allocate_contiguous_bump(&mut self, count: u32) -> Option<u32> {
        if count == 0 { return Some(self.next_free_brick); }
        let start = self.next_free_brick;
        let end = start.checked_add(count)?;
        if end as usize > self.capacity_bricks() as usize {
            let new_cap = (self.capacity_bricks() as u32 * 2).max(end).max(64);
            self.grow(new_cap);
        }
        self.next_free_brick = end;
        Some(start)
    }

    /// Allocate one brick and return its id (0-based). Initial contents are
    /// all-EMPTY. Panics if the pool can't grow further (would require
    /// allocating >2^31 cells).
    pub fn allocate(&mut self) -> Option<u32> {
        if let Some(idx) = self.free_list.iter().position(|(_, c)| *c >= 1) {
            let (start, count) = self.free_list[idx];
            if count == 1 {
                self.free_list.swap_remove(idx);
            } else {
                self.free_list[idx] = (start + 1, count - 1);
            }
            return Some(start);
        }

        if self.next_free_brick as usize >= self.capacity_bricks() as usize {
            let new_cap = (self.capacity_bricks() as u32).checked_mul(2)?.max(64);
            self.grow(new_cap);
        }
        let id = self.next_free_brick;
        self.next_free_brick += 1;
        Some(id)
    }

    /// Free many bricks at once, O(n) in the batch size.
    ///
    /// The per-brick [`deallocate`] has a tail-coalescing loop that
    /// runs `position` scans of the free list. Freeing the last-
    /// allocated brick of a long contiguous range then pays O(m) per
    /// brick to walk the free-list, which is O(n²) total for the
    /// batch. Real-world procedural rebakes hit this every time —
    /// the batch freed is 100k–1M bricks contiguous from the previous
    /// bake. This method handles the whole batch in one pass:
    ///
    /// 1. Clear every brick's cells (unavoidable memory writes).
    /// 2. Sort ids, merge into contiguous ranges.
    /// 3. Either extend `next_free_brick` (ranges at the tail) or
    ///    push one `(start, count)` entry per disjoint range.
    /// 4. One coalescing pass to absorb any adjacent free-list
    ///    entries into `next_free_brick`.
    pub fn deallocate_batch(&mut self, brick_ids: &[u32]) {
        if brick_ids.is_empty() {
            return;
        }

        // Clear cells first — unordered is fine.
        for &id in brick_ids {
            let start = id as usize * BRICK_CELLS as usize;
            let end = start + BRICK_CELLS as usize;
            if end > self.data.len() {
                continue;
            }
            for cell in &mut self.data[start..end] {
                *cell = BRICK_EMPTY;
            }
        }

        // Sort + group into (start, count) ranges.
        let mut sorted: Vec<u32> = brick_ids.to_vec();
        sorted.sort_unstable();
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        let mut range_start = sorted[0];
        let mut range_count: u32 = 1;
        for &id in &sorted[1..] {
            if id == range_start + range_count {
                range_count += 1;
            } else if id == range_start + range_count - 1 {
                // Duplicate id — ignore. `deallocate` also silently
                // no-ops on double-free above the data-len check.
                continue;
            } else {
                ranges.push((range_start, range_count));
                range_start = id;
                range_count = 1;
            }
        }
        ranges.push((range_start, range_count));

        // Walk the ranges. Any range that butts up against
        // `next_free_brick` shrinks the tail directly; the rest go on
        // the free list.
        for (s, c) in ranges {
            if s + c == self.next_free_brick {
                self.next_free_brick = s;
            } else {
                self.free_list.push((s, c));
            }
        }

        // One-shot tail coalesce: repeatedly absorb any free-list
        // range whose end meets the current tail. Position-scan is
        // O(free_list.len()) per iteration, but we stop as soon as
        // nothing matches, so across all iterations we visit each
        // entry at most twice.
        loop {
            let idx = self
                .free_list
                .iter()
                .position(|&(s, c)| s + c == self.next_free_brick);
            match idx {
                Some(i) => {
                    let (s, _) = self.free_list.swap_remove(i);
                    self.next_free_brick = s;
                }
                None => break,
            }
        }
    }

    /// Return a brick to the pool. Its cells are zeroed to EMPTY so no stale
    /// leaf_attr ids leak into a future caller.
    pub fn deallocate(&mut self, brick_id: u32) {
        let start = brick_id as usize * BRICK_CELLS as usize;
        let end = start + BRICK_CELLS as usize;
        if end > self.data.len() {
            return;
        }
        for cell in &mut self.data[start..end] {
            *cell = BRICK_EMPTY;
        }
        if brick_id + 1 == self.next_free_brick {
            self.next_free_brick = brick_id;
            loop {
                let idx = self.free_list.iter().position(|(s, c)| s + c == self.next_free_brick);
                match idx {
                    Some(i) => {
                        let (s, _) = self.free_list.swap_remove(i);
                        self.next_free_brick = s;
                    }
                    None => break,
                }
            }
        } else {
            self.free_list.push((brick_id, 1));
        }
    }

    /// Read a cell at (x, y, z) within the given brick.
    #[inline]
    pub fn get_cell(&self, brick_id: u32, x: u32, y: u32, z: u32) -> u32 {
        let offset = brick_id as usize * BRICK_CELLS as usize
            + brick_flat_index(x, y, z) as usize;
        self.data[offset]
    }

    /// Write a cell at (x, y, z) within the given brick.
    #[inline]
    pub fn set_cell(&mut self, brick_id: u32, x: u32, y: u32, z: u32, value: u32) {
        let offset = brick_id as usize * BRICK_CELLS as usize
            + brick_flat_index(x, y, z) as usize;
        self.data[offset] = value;
    }

    /// Raw slice of a brick's 64 cells.
    #[inline]
    pub fn brick_cells(&self, brick_id: u32) -> &[u32] {
        let start = brick_id as usize * BRICK_CELLS as usize;
        &self.data[start..start + BRICK_CELLS as usize]
    }

    /// Mutable slice of a brick's 64 cells.
    #[inline]
    pub fn brick_cells_mut(&mut self, brick_id: u32) -> &mut [u32] {
        let start = brick_id as usize * BRICK_CELLS as usize;
        &mut self.data[start..start + BRICK_CELLS as usize]
    }

    /// Number of bricks currently allocated (inclusive of any holes on the
    /// free list — this is the high-water mark, what would be uploaded).
    #[inline]
    pub fn allocated_count(&self) -> u32 { self.next_free_brick }

    /// Current pool capacity measured in bricks.
    #[inline]
    pub fn capacity_bricks(&self) -> u32 {
        (self.data.len() / BRICK_CELLS as usize) as u32
    }

    /// Grow the pool to at least `new_cap` bricks. Preserves existing data.
    pub fn grow(&mut self, new_cap: u32) {
        let target_cells = (new_cap as usize) * (BRICK_CELLS as usize);
        if target_cells <= self.data.len() {
            return;
        }
        self.data.resize(target_cells, BRICK_EMPTY);
    }

    /// Raw byte view of the allocated region (for GPU upload).
    pub fn as_bytes(&self) -> &[u8] {
        let count = self.next_free_brick as usize * BRICK_CELLS as usize;
        if count == 0 {
            return &[];
        }
        bytemuck::cast_slice(&self.data[..count])
    }

    /// Raw slice of allocated u32 cells across all bricks.
    pub fn as_slice(&self) -> &[u32] {
        let count = self.next_free_brick as usize * BRICK_CELLS as usize;
        &self.data[..count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_consistent() {
        assert_eq!(BRICK_DIM.pow(3), BRICK_CELLS);
        assert_eq!(BRICK_LEVELS, BRICK_DIM.ilog2() as u8);
        assert_eq!(BRICK_BYTES, BRICK_CELLS * 4);
    }

    #[test]
    fn flat_index_linear() {
        assert_eq!(brick_flat_index(0, 0, 0), 0);
        assert_eq!(brick_flat_index(1, 0, 0), 1);
        assert_eq!(brick_flat_index(0, 1, 0), 4);
        assert_eq!(brick_flat_index(0, 0, 1), 16);
        assert_eq!(brick_flat_index(3, 3, 3), 63);
    }

    #[test]
    fn new_pool_is_empty() {
        let pool = BrickPool::new(4);
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.capacity_bricks(), 4);
    }

    #[test]
    fn allocate_sets_all_cells_empty() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        assert_eq!(id, 0);
        for cell in pool.brick_cells(id) {
            assert_eq!(*cell, BRICK_EMPTY);
        }
    }

    #[test]
    fn set_get_cell_roundtrip() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        pool.set_cell(id, 1, 2, 3, 42);
        assert_eq!(pool.get_cell(id, 1, 2, 3), 42);
        // Other cells remain EMPTY.
        assert_eq!(pool.get_cell(id, 0, 0, 0), BRICK_EMPTY);
    }

    #[test]
    fn allocate_grows() {
        let mut pool = BrickPool::new(1);
        assert_eq!(pool.allocate().unwrap(), 0);
        // Pool of size 1 is full — next allocation should grow.
        let id2 = pool.allocate().unwrap();
        assert_eq!(id2, 1);
        assert!(pool.capacity_bricks() >= 2);
    }

    #[test]
    fn deallocate_tail_shrinks() {
        let mut pool = BrickPool::new(4);
        let a = pool.allocate().unwrap();
        let _b = pool.allocate().unwrap();
        pool.deallocate(1);
        // Tail deallocation — next_free_brick shrinks.
        assert_eq!(pool.allocated_count(), 1);
        let _ = a;
    }

    #[test]
    fn deallocate_interior_reusable() {
        let mut pool = BrickPool::new(4);
        let a = pool.allocate().unwrap();
        let _b = pool.allocate().unwrap();
        pool.deallocate(a);
        // a was interior (b still live at tail).
        let reused = pool.allocate().unwrap();
        assert_eq!(reused, a);
    }

    #[test]
    fn as_bytes_sized_correctly() {
        let mut pool = BrickPool::new(4);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert_eq!(pool.as_bytes().len(), 2 * BRICK_BYTES as usize);
    }

    #[test]
    fn brick_cells_covers_full_slice() {
        let mut pool = BrickPool::new(2);
        let id = pool.allocate().unwrap();
        pool.brick_cells_mut(id)[7] = 99;
        assert_eq!(pool.get_cell(id, 3, 1, 0), 99); // 3 + 1*4 + 0*16 = 7
    }
}
