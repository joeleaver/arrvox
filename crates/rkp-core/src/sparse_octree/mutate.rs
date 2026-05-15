//! Brick-aware single-cell mutation primitives on [`SparseOctree`].
//!
//! Where [`insert`](super::insert) handles bulk voxelization driven by a
//! single owner (an asset loader, procedural baker), this module handles
//! **incremental edits** against an already-baked octree: sculpt, paint
//! ops that change occupancy, destruction, runtime terrain edits.
//!
//! The three primitives are:
//!
//! * [`SparseOctree::set_cell_solid`] — make a finest-grid cell SOLID at
//!   a given leaf_attr slot.
//! * [`SparseOctree::set_cell_empty`] — make a finest-grid cell EMPTY.
//! * [`SparseOctree::set_cell_interior`] — make a finest-grid cell
//!   INTERIOR (occupied bulk, no visible surface).
//!
//! All three are **brick-aware**: they materialize a [`BrickPool`] entry
//! on demand when the target cell sits inside an EMPTY_NODE / INTERIOR_NODE
//! terminator at brick depth, and they free the brick when its 64 cells
//! collapse back to a uniform sentinel. They also opportunistically
//! collapse ancestor branches whose 8 children become identical.
//!
//! ## Return value contract
//!
//! Each primitive returns `Option<u32>` — the **previous** leaf_attr slot
//! that lived at the cell, if any. The caller owns the
//! [`LeafAttrPool`](crate::leaf_attr_pool::LeafAttrPool) and is
//! responsible for freeing the returned slot. `None` is returned when
//! the cell was previously EMPTY, INTERIOR, BRICK_EMPTY, or BRICK_INTERIOR
//! — none of those reference a leaf_attr slot.
//!
//! The caller hands in `leaf_attr_id` already allocated and written.
//! That keeps the primitive free of any allocation policy (single vs.
//! range, free-list vs. bump) — see the existing `apply_delta` in
//! `sculpt.rs` for the orchestration pattern.

use glam::UVec3;

use crate::brick_pool::{
    BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR, BRICK_LEVELS, BrickPool,
    brick_flat_index,
};

use super::{
    EMPTY_NODE, INTERIOR_NODE, INTERNAL_ATTR_NONE, SparseOctree, brick_id, is_branch, is_brick,
    is_leaf, leaf_slot, make_brick, make_leaf, octant_for_coord,
};

/// Local coord within a brick at finest grid `coord`. Equivalent to
/// `(coord & (BRICK_DIM - 1))` since BRICK_DIM is a power of two.
#[inline]
fn brick_local(coord: UVec3) -> (u32, u32, u32) {
    let mask = BRICK_DIM - 1;
    (coord.x & mask, coord.y & mask, coord.z & mask)
}

/// Per-call cache for [`SparseOctree::set_cell_solid_cached`] and the
/// other `*_cached` mutation primitives (D5.b). Amortises the 9-level
/// octree descent across consecutive cell mutations that fall in the
/// same brick.
///
/// `apply_delta` instantiates one of these and threads it through the
/// per-edit loop. Consecutive edits inside the same brick (the common
/// case under a brush stamp, where the kernel walks the brush region
/// in row-major (x, y, z) order — i.e. up to 4 cells in a row inside
/// one brick before crossing a brick boundary) hit the fast path,
/// skipping the descent.
///
/// `Default` produces an empty cache equivalent to a fresh one — no
/// fast-path possible on the first call.
#[derive(Debug, Default, Clone)]
pub struct BrickPathCache {
    /// Coord of the brick whose path is cached, in
    /// `coord >> BRICK_LEVELS` units. `None` means "no fast-path
    /// available" (cache is empty or was invalidated).
    brick_coord: Option<UVec3>,
    /// Node index in [`SparseOctree::nodes`] where the brick's
    /// terminator (BRICK / EMPTY_NODE / INTERIOR_NODE) lives. Only
    /// meaningful when `brick_coord.is_some()`.
    brick_node_idx: usize,
    /// Indices of the branch nodes from root down to (but not
    /// including) the brick. Reused on cache hits to walk back up for
    /// `try_collapse_after_mutate`.
    ancestor_path: Vec<usize>,
}

impl BrickPathCache {
    /// Fresh cache with no entry. Same as `Default::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the cache so the next mutation falls back to a full
    /// descent. Called after a collapse cascade detaches the cached
    /// brick from the live tree.
    pub fn invalidate(&mut self) {
        self.brick_coord = None;
        self.ancestor_path.clear();
    }
}

/// Occupancy state of a single finest-grid cell, resolved across the
/// octree's terminator types and (when applicable) the brick pool.
///
/// Sculpt / paint / runtime-edit kernels read this to decide what
/// edits a cell is eligible for; the variants line up with the
/// mutation primitives ([`SparseOctree::set_cell_solid`] etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Out of the octree's spatial bounds.
    OutOfBounds,
    /// Cell is air. No leaf_attr slot, no visible surface.
    Empty,
    /// Cell is occupied bulk. No visible surface; counts as mass for
    /// neighborhood / surface-net reconstruction.
    Interior,
    /// Cell has a surface attribute at the given LeafAttrPool slot.
    Solid(u32),
}

impl SparseOctree {
    /// Resolve the occupancy state of a finest-grid cell.
    ///
    /// Walks the octree to the cell's terminator and (if BRICK) reads
    /// the per-cell value from the supplied [`BrickPool`]. Returns
    /// [`CellState::OutOfBounds`] when `coord` falls outside the tree.
    ///
    /// This is the kernel-side counterpart to [`Self::set_cell_solid`]
    /// / [`Self::set_cell_empty`] / [`Self::set_cell_interior`]: read
    /// the state, classify against the brush, decide the edit.
    pub fn cell_state(&self, coord: UVec3, brick_pool: &BrickPool) -> CellState {
        if !self.in_bounds(coord) {
            return CellState::OutOfBounds;
        }
        let (node, _depth) = self.lookup_with_depth(coord).expect("in bounds checked above");
        if node == EMPTY_NODE {
            return CellState::Empty;
        }
        if node == INTERIOR_NODE {
            return CellState::Interior;
        }
        if is_leaf(node) {
            return CellState::Solid(leaf_slot(node));
        }
        if is_brick(node) {
            let bid = brick_id(node);
            let (lx, ly, lz) = brick_local(coord);
            let cell = brick_pool.get_cell(bid, lx, ly, lz);
            return match cell {
                BRICK_EMPTY => CellState::Empty,
                BRICK_INTERIOR => CellState::Interior,
                slot => CellState::Solid(slot),
            };
        }
        // Branch reached lookup_with_depth — shouldn't happen since
        // lookup_with_depth returns the deepest terminator. Fall back
        // to Empty defensively.
        debug_assert!(false, "cell_state hit a branch node at {coord} — lookup invariant broken");
        CellState::Empty
    }
}

/// Return `Some(prev_slot)` if `cell` is a real leaf_attr id, else `None`.
/// Used by the brick-cell write path to surface the slot the caller must
/// free.
#[inline]
fn brick_cell_prev_slot(cell: u32) -> Option<u32> {
    if cell == BRICK_EMPTY || cell == BRICK_INTERIOR {
        None
    } else {
        Some(cell)
    }
}

/// Inspect the 64 cells of a brick after a write. If all cells equal a
/// single sentinel (BRICK_EMPTY or BRICK_INTERIOR), return the
/// corresponding octree-level sentinel (EMPTY_NODE / INTERIOR_NODE) so
/// the caller can collapse the BRICK back. Mixed bricks return `None`.
#[inline]
fn brick_collapse_target(brick_pool: &BrickPool, bid: u32) -> Option<u32> {
    let cells = brick_pool.brick_cells(bid);
    let first = cells[0];
    if first != BRICK_EMPTY && first != BRICK_INTERIOR {
        return None;
    }
    for &c in &cells[1..] {
        if c != first {
            return None;
        }
    }
    Some(if first == BRICK_INTERIOR {
        INTERIOR_NODE
    } else {
        EMPTY_NODE
    })
}

/// Allocate a brick, fill all 64 cells with `fill`, then patch the
/// target cell to `target_value`. Returns the new brick id.
#[inline]
fn materialize_brick(
    brick_pool: &mut BrickPool,
    fill: u32,
    local: (u32, u32, u32),
    target_value: u32,
) -> u32 {
    let bid = brick_pool
        .allocate()
        .expect("brick_pool allocation exhausted (cannot grow beyond u32 cells)");
    let cells = brick_pool.brick_cells_mut(bid);
    for c in cells.iter_mut() {
        *c = fill;
    }
    let flat = brick_flat_index(local.0, local.1, local.2) as usize;
    cells[flat] = target_value;
    bid
}

impl SparseOctree {
    /// Set the finest-grid cell at `coord` to SOLID with leaf attrs at
    /// `leaf_attr_id`. Returns the previous leaf_attr slot if the cell
    /// already held one (caller must free it from its
    /// [`LeafAttrPool`](crate::leaf_attr_pool::LeafAttrPool)).
    ///
    /// Materializes a brick on demand if the cell sits inside an
    /// EMPTY_NODE / INTERIOR_NODE region at brick depth. Subdivides
    /// branches as needed. Collapses ancestors that become uniform.
    ///
    /// Panics if `coord` is out of bounds for this tree.
    pub fn set_cell_solid(
        &mut self,
        coord: UVec3,
        leaf_attr_id: u32,
        brick_pool: &mut BrickPool,
    ) -> Option<u32> {
        assert!(self.in_bounds(coord), "coord {coord} out of bounds for depth {}", self.depth);
        self.mutate_at(0, coord, 0, brick_pool, CellOp::Solid(leaf_attr_id))
    }

    /// Set the finest-grid cell at `coord` to EMPTY. Returns the
    /// previous leaf_attr slot if the cell held one. Frees the
    /// containing brick if it becomes all-empty. Triggers branch
    /// collapse on ancestors.
    ///
    /// Panics if `coord` is out of bounds.
    pub fn set_cell_empty(
        &mut self,
        coord: UVec3,
        brick_pool: &mut BrickPool,
    ) -> Option<u32> {
        assert!(self.in_bounds(coord), "coord {coord} out of bounds for depth {}", self.depth);
        self.mutate_at(0, coord, 0, brick_pool, CellOp::Empty)
    }

    /// Set the finest-grid cell at `coord` to INTERIOR (occupied bulk,
    /// no visible surface; counts as mass for the surface-net
    /// reconstruction). Returns the previous leaf_attr slot if the
    /// cell held one. Frees the containing brick if it becomes
    /// uniformly INTERIOR and writes [`INTERIOR_NODE`] at the octree
    /// level.
    ///
    /// Panics if `coord` is out of bounds.
    pub fn set_cell_interior(
        &mut self,
        coord: UVec3,
        brick_pool: &mut BrickPool,
    ) -> Option<u32> {
        assert!(self.in_bounds(coord), "coord {coord} out of bounds for depth {}", self.depth);
        self.mutate_at(0, coord, 0, brick_pool, CellOp::Interior)
    }

    // ─────────────── Cached cell mutation (D5.b) ──────────────────

    /// [`set_cell_solid`](Self::set_cell_solid) with a per-call brick
    /// path cache. Equivalent semantics; the cache amortizes the
    /// 9-level descent across consecutive mutations in the same brick.
    pub fn set_cell_solid_cached(
        &mut self,
        coord: UVec3,
        leaf_attr_id: u32,
        brick_pool: &mut BrickPool,
        cache: &mut BrickPathCache,
    ) -> Option<u32> {
        self.mutate_cell_cached(coord, CellOp::Solid(leaf_attr_id), brick_pool, cache)
    }

    /// [`set_cell_empty`](Self::set_cell_empty) with a per-call brick
    /// path cache. See [`set_cell_solid_cached`](Self::set_cell_solid_cached).
    pub fn set_cell_empty_cached(
        &mut self,
        coord: UVec3,
        brick_pool: &mut BrickPool,
        cache: &mut BrickPathCache,
    ) -> Option<u32> {
        self.mutate_cell_cached(coord, CellOp::Empty, brick_pool, cache)
    }

    /// [`set_cell_interior`](Self::set_cell_interior) with a per-call
    /// brick path cache.
    pub fn set_cell_interior_cached(
        &mut self,
        coord: UVec3,
        brick_pool: &mut BrickPool,
        cache: &mut BrickPathCache,
    ) -> Option<u32> {
        self.mutate_cell_cached(coord, CellOp::Interior, brick_pool, cache)
    }

    /// Cached cell mutation — see [`BrickPathCache`].
    ///
    /// **Fast path** (cache hit): the previous call mutated a cell in
    /// the same brick, so the brick's `node_idx` and ancestor branch
    /// path are still in [`cache`]. We skip the descent entirely and
    /// call `mutate_at_brick` directly, then walk `cache.ancestor_path`
    /// in reverse for `try_collapse_after_mutate`. If any ancestor's
    /// node value changes during the collapse walk, the path is now
    /// invalid (the brick is orphaned) and the cache is cleared so the
    /// next call re-descends.
    ///
    /// **Slow path** (cache miss / shallow tree / collapsed cache): an
    /// iterative descent equivalent to [`mutate_at`], but recording
    /// every branch / freshly-subdivided node into
    /// `cache.ancestor_path`. The brick's node_idx is then stashed in
    /// the cache for the next call.
    fn mutate_cell_cached(
        &mut self,
        coord: UVec3,
        op: CellOp,
        brick_pool: &mut BrickPool,
        cache: &mut BrickPathCache,
    ) -> Option<u32> {
        assert!(
            self.in_bounds(coord),
            "coord {coord} out of bounds for depth {}",
            self.depth
        );

        let bricks_enabled = self.depth > BRICK_LEVELS;
        if !bricks_enabled {
            // Shallow trees don't have bricks; the cache wouldn't help
            // (one level of descent, no brick boundary). Fall back.
            cache.invalidate();
            return self.mutate_at(0, coord, 0, brick_pool, op);
        }

        let brick_depth = self.depth - BRICK_LEVELS;
        let brick_coord = UVec3::new(
            coord.x >> BRICK_LEVELS,
            coord.y >> BRICK_LEVELS,
            coord.z >> BRICK_LEVELS,
        );

        // ── Fast path: cache hit ──────────────────────────────────
        if cache.brick_coord == Some(brick_coord) {
            let n = self.nodes[cache.brick_node_idx];
            // Cache is valid only if the slot still holds a brick-depth
            // terminator. If the brick collapsed and an ancestor split
            // mid-batch (rare but possible across cache-invalidating
            // events we didn't observe), fall back to a full descent.
            if is_brick(n) || n == EMPTY_NODE || n == INTERIOR_NODE {
                let prev = self.mutate_at_brick(
                    cache.brick_node_idx,
                    n,
                    coord,
                    brick_pool,
                    op,
                );
                let still_valid = self.run_ancestor_collapse(&cache.ancestor_path);
                if !still_valid {
                    // An ancestor collapsed → our brick_node_idx is now
                    // orphaned (the parent now points at a different
                    // terminator). Subsequent same-brick edits must
                    // re-descend.
                    cache.invalidate();
                }
                return prev;
            }
            // Cache is stale: the slot was overwritten in some way we
            // didn't anticipate. Fall through to the slow path.
            cache.invalidate();
        }

        // ── Slow path: iterative descent, capturing path ──────────
        //
        // Mirrors `mutate_at`'s control flow exactly: branches are
        // descended first (regardless of level — trees deeper than
        // `brick_depth` legitimately exist after `mutate_at`-driven
        // subdivision in shallower regions of the same tree); brick
        // depth is checked only when the current node is a
        // non-branch terminator. Inverting the order is a real bug —
        // it tries to invoke `mutate_at_brick` on a BRANCH and panics
        // the brick-depth assertion.
        cache.ancestor_path.clear();
        let mut node_idx: usize = 0;
        let mut level: u8 = 0;

        loop {
            let current = self.nodes[node_idx];

            if is_branch(current) {
                cache.ancestor_path.push(node_idx);
                let children_offset = current as usize;
                let octant = octant_for_coord(coord, level, self.depth) as usize;
                node_idx = children_offset + octant;
                level += 1;
                continue;
            }

            if level == brick_depth {
                // At brick depth and the node is a terminator — apply
                // the op via the shared brick logic.
                let prev = self.mutate_at_brick(node_idx, current, coord, brick_pool, op);
                let still_valid = self.run_ancestor_collapse(&cache.ancestor_path);
                let new_n = self.nodes[node_idx];
                if still_valid
                    && (is_brick(new_n) || new_n == EMPTY_NODE || new_n == INTERIOR_NODE)
                {
                    cache.brick_coord = Some(brick_coord);
                    cache.brick_node_idx = node_idx;
                    // ancestor_path already populated.
                } else {
                    cache.invalidate();
                }
                return prev;
            }

            if level == self.depth {
                // The descent went past `brick_depth` via earlier
                // branches — this happens on trees that inserted at
                // finest level above the brick boundary (synthetic
                // tests, shallow-tree paths). Apply via
                // `mutate_at_finest`; the cell isn't actually
                // brick-backed so we don't update the cache.
                let prev = self.mutate_at_finest(node_idx, current, op);
                let _ = self.run_ancestor_collapse(&cache.ancestor_path);
                cache.invalidate();
                return prev;
            }

            // Intermediate level, non-branch terminator. Subdivide
            // (mirrors the subdivide block in `mutate_at`). After this
            // node becomes a branch, the next loop iteration descends.
            if is_leaf(current) {
                debug_assert!(
                    false,
                    "mutate_cell_cached hit a LEAF at intermediate level {} \
                     (depth={}, brick_depth={}). Single-cell refinement of a \
                     coarse LEAF is not supported in R1.",
                    level, self.depth, brick_depth
                );
                // Release fallback: treat as the subdivide path below.
            }

            cache.ancestor_path.push(node_idx);
            let children_offset = self.nodes.len();
            self.nodes.extend_from_slice(&[current; 8]);
            self.internal_attr_index
                .extend_from_slice(&[INTERNAL_ATTR_NONE; 8]);
            for i in 0..8u32 {
                self.record_node_write(children_offset as u32 + i, current);
                self.record_attr_write(children_offset as u32 + i, INTERNAL_ATTR_NONE);
            }
            self.nodes[node_idx] = children_offset as u32;
            self.record_node_write(node_idx as u32, children_offset as u32);
            self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
            self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);

            let octant = octant_for_coord(coord, level, self.depth) as usize;
            node_idx = children_offset + octant;
            level += 1;
        }
    }

    /// Walk an ancestor path bottom-up, calling
    /// [`try_collapse_after_mutate`](Self::try_collapse_after_mutate) at
    /// each branch. Returns `true` if no ancestor's value changed
    /// (path still valid for cache reuse), `false` if any ancestor
    /// collapsed (the path is broken and the cache must be invalidated).
    fn run_ancestor_collapse(&mut self, ancestor_path: &[usize]) -> bool {
        let mut changed = false;
        for &idx in ancestor_path.iter().rev() {
            let pre = self.nodes[idx];
            self.try_collapse_after_mutate(idx);
            if self.nodes[idx] != pre {
                changed = true;
            }
        }
        !changed
    }

    /// Recursive descent. Returns the previous leaf_attr slot at the
    /// target cell (if any), bubbled all the way back up.
    fn mutate_at(
        &mut self,
        node_idx: usize,
        coord: UVec3,
        level: u8,
        brick_pool: &mut BrickPool,
        op: CellOp,
    ) -> Option<u32> {
        // Brick-depth math. `BRICK_LEVELS` is u8 const (=2).
        // `brick_depth` is the level at which a BRICK terminator can
        // live; below this we descend into per-brick cells. For
        // shallow trees where `depth < BRICK_LEVELS`, bricks are
        // disabled (matches the asset-bake convention in
        // `voxelize_octree/emit.rs::emit_leaves_batched`).
        let bricks_enabled = self.depth > BRICK_LEVELS;
        let brick_depth = if bricks_enabled {
            self.depth - BRICK_LEVELS
        } else {
            u8::MAX // unreachable since `level <= depth`
        };

        let current = self.nodes[node_idx];

        // ── Branch: descend. ──────────────────────────────────────
        if is_branch(current) {
            let children_offset = current as usize;
            let octant = octant_for_coord(coord, level, self.depth) as usize;
            let prev =
                self.mutate_at(children_offset + octant, coord, level + 1, brick_pool, op);
            self.try_collapse_after_mutate(node_idx);
            return prev;
        }

        // ── At brick depth: brick-cell write or materialization. ─
        if bricks_enabled && level == brick_depth {
            return self.mutate_at_brick(node_idx, current, coord, brick_pool, op);
        }

        // ── At finest depth (shallow trees only): direct LEAF write.
        if level == self.depth {
            return self.mutate_at_finest(node_idx, current, op);
        }

        // ── Above brick depth (or above finest, for shallow trees):
        // current is a non-branch terminator (EMPTY/INTERIOR/LEAF).
        // EMPTY/INTERIOR may need to subdivide; LEAF at intermediate
        // levels is a coarser-LOD slot which we don't support
        // refining via this primitive in R1.
        if is_leaf(current) {
            // A LEAF at an intermediate level represents a coarse-LOD
            // node (`internal_attr_index` is the standard prefilter
            // path; this is the rarer "direct write into nodes[]"
            // case, e.g. from `set_at_level`). Refining it through
            // single-cell mutation isn't well-defined: do we want
            // 511 sibling cells filled with the LEAF's slot, all
            // INTERIOR, or all EMPTY? Asset / procedural bakes never
            // produce this state at the moment. Assert so the case
            // surfaces if some future caller does.
            debug_assert!(
                false,
                "set_cell_* hit a LEAF at intermediate level {} (depth={}, brick_depth={}). \
                 Single-cell refinement of a coarse LEAF is not supported in R1.",
                level, self.depth, brick_depth
            );
            // In release builds: treat as EMPTY_NODE for the subdivide
            // step; the LEAF's slot is leaked. This is a best-effort
            // fallback rather than corrupting the tree.
        }

        // Subdivide. The 8 new children inherit `current` (uniform
        // fill — EMPTY_NODE or INTERIOR_NODE).
        let children_offset = self.nodes.len();
        self.nodes.extend_from_slice(&[current; 8]);
        self.internal_attr_index
            .extend_from_slice(&[INTERNAL_ATTR_NONE; 8]);
        for i in 0..8u32 {
            self.record_node_write(children_offset as u32 + i, current);
            self.record_attr_write(children_offset as u32 + i, INTERNAL_ATTR_NONE);
        }
        self.nodes[node_idx] = children_offset as u32;
        self.record_node_write(node_idx as u32, children_offset as u32);
        // The slot that was a terminator becomes a branch; its
        // (stale) prefilter attr at this index should now be NONE.
        self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
        self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);

        let octant = octant_for_coord(coord, level, self.depth) as usize;
        let prev =
            self.mutate_at(children_offset + octant, coord, level + 1, brick_pool, op);
        self.try_collapse_after_mutate(node_idx);
        prev
    }

    /// Apply `op` at a BRICK / EMPTY_NODE / INTERIOR_NODE terminator
    /// sitting at `brick_depth`. Materializes / writes / frees the
    /// brick as needed; updates the octree slot to BRICK / EMPTY_NODE
    /// / INTERIOR_NODE per the resulting state.
    fn mutate_at_brick(
        &mut self,
        node_idx: usize,
        current: u32,
        coord: UVec3,
        brick_pool: &mut BrickPool,
        op: CellOp,
    ) -> Option<u32> {
        let local = brick_local(coord);

        // Write target value into the brick cell.
        let target_value = match op {
            CellOp::Solid(slot) => slot,
            CellOp::Empty => BRICK_EMPTY,
            CellOp::Interior => BRICK_INTERIOR,
        };

        // Case 1: already a BRICK — patch the cell in place.
        if is_brick(current) {
            let bid = brick_id(current);
            let flat = brick_flat_index(local.0, local.1, local.2);
            let prev_cell = brick_pool.get_cell(bid, local.0, local.1, local.2);
            // Short-circuit if no change.
            if prev_cell == target_value {
                return brick_cell_prev_slot(prev_cell);
            }
            // Write via the typed slice to avoid a redundant flat-index
            // recompute.
            brick_pool.brick_cells_mut(bid)[flat as usize] = target_value;

            // Did the brick collapse to a uniform sentinel?
            if let Some(uniform) = brick_collapse_target(brick_pool, bid) {
                brick_pool.deallocate(bid);
                self.nodes[node_idx] = uniform;
                self.record_node_write(node_idx as u32, uniform);
                // Branch ancestors will be checked on the recursion's
                // way up; this node's own internal_attr is now stale
                // for a non-branch slot.
                self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
                self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);
            }
            return brick_cell_prev_slot(prev_cell);
        }

        // Case 2: EMPTY_NODE / INTERIOR_NODE — may need materialization.
        debug_assert!(
            current == EMPTY_NODE || current == INTERIOR_NODE,
            "expected EMPTY_NODE or INTERIOR_NODE at brick_depth, got 0x{:08X}",
            current
        );
        let fill = if current == INTERIOR_NODE {
            BRICK_INTERIOR
        } else {
            BRICK_EMPTY
        };

        // Short-circuit: writing the same value as the uniform fill
        // leaves the region unchanged. No brick materialization, no
        // slot to free.
        if target_value == fill {
            return None;
        }

        let bid = materialize_brick(brick_pool, fill, local, target_value);

        // Sanity: the freshly-materialized brick can't collapse, since
        // we just made it non-uniform.
        debug_assert!(brick_collapse_target(brick_pool, bid).is_none());

        let brick_node = make_brick(bid);
        self.nodes[node_idx] = brick_node;
        self.record_node_write(node_idx as u32, brick_node);
        self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
        self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);
        None
    }

    /// Apply `op` at a finest-level terminator (only reached for
    /// shallow trees where bricks are disabled, or in the depth==0
    /// pathological case). LEAF gets replaced; EMPTY/INTERIOR gets
    /// overwritten with the new terminator.
    fn mutate_at_finest(
        &mut self,
        node_idx: usize,
        current: u32,
        op: CellOp,
    ) -> Option<u32> {
        let prev = if is_leaf(current) {
            Some(leaf_slot(current))
        } else {
            None
        };
        let new_node = match op {
            CellOp::Solid(slot) => make_leaf(slot),
            CellOp::Empty => EMPTY_NODE,
            CellOp::Interior => INTERIOR_NODE,
        };
        if self.nodes[node_idx] != new_node {
            self.nodes[node_idx] = new_node;
            self.record_node_write(node_idx as u32, new_node);
            self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
            self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);
        }
        prev
    }

    /// Like the existing [`try_collapse`](super::insert) helper but
    /// scoped to the mutate path. Collapses a branch whose 8 children
    /// are identical non-branch terminators, then resets the parent
    /// slot's prefilter attr (stale once the slot stops being a
    /// branch).
    fn try_collapse_after_mutate(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];
        if !is_branch(node) {
            return;
        }
        let children_offset = node as usize;
        let first = self.nodes[children_offset];
        if is_branch(first) {
            return;
        }
        for i in 1..8 {
            if self.nodes[children_offset + i] != first {
                return;
            }
        }
        self.nodes[node_idx] = first;
        self.record_node_write(node_idx as u32, first);
        self.internal_attr_index[node_idx] = INTERNAL_ATTR_NONE;
        self.record_attr_write(node_idx as u32, INTERNAL_ATTR_NONE);
    }
}

/// What to write at a finest-grid cell.
#[derive(Debug, Clone, Copy)]
enum CellOp {
    Solid(u32),
    Empty,
    Interior,
}

#[cfg(not(debug_assertions))]
#[allow(dead_code)]
const _: () = {
    // Silence unused-import warnings for items that only the
    // debug_assert path inside `mutate_at` references in non-release
    // builds. Keeping the constants reachable here keeps imports
    // honest if release is the only target compiled.
};

// Compile-time sanity: the brick local mask logic assumes BRICK_DIM
// is a power of two AND BRICK_DIM == 1 << BRICK_LEVELS. If either
// breaks, the brick_local() bit-mask is wrong and the descent
// boundary between brick_depth and finest is wrong too.
const _: () = {
    assert!(BRICK_DIM.is_power_of_two());
    assert!(BRICK_DIM == 1 << BRICK_LEVELS);
    assert!(BRICK_CELLS == BRICK_DIM * BRICK_DIM * BRICK_DIM);
};

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the tree and assert that every brick referenced by the
    /// nodes[] array has its corresponding `brick_pool.brick_cells`
    /// slot showing at least one non-sentinel cell (i.e., the brick is
    /// "live" — collapsed bricks should have been deallocated and
    /// replaced with EMPTY_NODE / INTERIOR_NODE).
    fn assert_no_collapsed_bricks_left_behind(tree: &SparseOctree, brick_pool: &BrickPool) {
        for &node in tree.as_slice() {
            if is_brick(node) {
                let bid = brick_id(node);
                let cells = brick_pool.brick_cells(bid);
                // At least one cell should be a non-sentinel slot OR
                // the brick should be a legitimate mixed sentinel
                // (BRICK_EMPTY + BRICK_INTERIOR mix). If it's a fully
                // uniform sentinel, our collapse code should have
                // freed it.
                let collapse = brick_collapse_target(brick_pool, bid);
                assert!(
                    collapse.is_none(),
                    "BRICK at bid={bid} should have collapsed to {:?}",
                    collapse
                );
                // Suppress unused if assertion is disabled.
                let _ = cells.len();
            }
        }
    }

    /// Common deep-tree harness. depth=4 → 16³ finest, brick_depth=2,
    /// one brick covers a 4³ region.
    fn deep_tree() -> (SparseOctree, BrickPool) {
        let tree = SparseOctree::new(4, 1.0);
        let pool = BrickPool::new(16);
        (tree, pool)
    }

    // ── Shallow-tree path (no bricks) ─────────────────────────────

    #[test]
    fn shallow_set_solid_writes_finest_leaf() {
        // depth=2 ≤ BRICK_LEVELS → bricks disabled; LEAF lives at finest.
        let mut tree = SparseOctree::new(2, 1.0);
        let mut pool = BrickPool::new(4);
        let coord = UVec3::new(1, 2, 3);

        let prev = tree.set_cell_solid(coord, 42, &mut pool);
        assert!(prev.is_none());
        assert_eq!(tree.lookup(coord), Some(make_leaf(42)));
        assert_eq!(pool.allocated_count(), 0, "no brick should have been allocated");

        // Overwriting returns the previous slot.
        let prev2 = tree.set_cell_solid(coord, 99, &mut pool);
        assert_eq!(prev2, Some(42));
        assert_eq!(tree.lookup(coord), Some(make_leaf(99)));
    }

    #[test]
    fn shallow_set_empty_clears_leaf() {
        let mut tree = SparseOctree::new(2, 1.0);
        let mut pool = BrickPool::new(4);
        let coord = UVec3::new(0, 0, 0);
        tree.set_cell_solid(coord, 7, &mut pool);

        let prev = tree.set_cell_empty(coord, &mut pool);
        assert_eq!(prev, Some(7));
        assert_eq!(tree.lookup(coord), Some(EMPTY_NODE));
        // Whole tree collapses back to a single EMPTY_NODE root. The
        // orphaned subdivision slots are not reclaimed (callers run
        // `compact` for that); root must read as EMPTY_NODE though.
        assert_eq!(tree.as_slice()[0], EMPTY_NODE);
        assert_eq!(tree.live_node_count(), 1);
    }

    #[test]
    fn shallow_set_interior_writes_finest_terminator() {
        let mut tree = SparseOctree::new(2, 1.0);
        let mut pool = BrickPool::new(4);
        let coord = UVec3::new(2, 2, 2);

        let prev = tree.set_cell_interior(coord, &mut pool);
        assert!(prev.is_none());
        assert_eq!(tree.lookup(coord), Some(INTERIOR_NODE));
    }

    // ── Deep-tree: EMPTY_NODE → brick materialization ─────────────

    #[test]
    fn deep_solid_into_empty_materializes_brick() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 6, 7); // inside the brick covering (4..8)³

        let prev = tree.set_cell_solid(coord, 42, &mut pool);
        assert!(prev.is_none());

        // Cell lookup returns the BRICK node value (the octree reports
        // the terminator at the brick level — the leaf_attr_id lives
        // inside the BrickPool, not in nodes[]).
        let node = tree.lookup(coord).unwrap();
        assert!(is_brick(node), "expected BRICK terminator, got 0x{:08X}", node);

        let bid = brick_id(node);
        let (lx, ly, lz) = brick_local(coord);
        assert_eq!(pool.get_cell(bid, lx, ly, lz), 42);

        // Other 63 cells in the brick are BRICK_EMPTY.
        let mut empties = 0;
        for cell in pool.brick_cells(bid) {
            if *cell == BRICK_EMPTY {
                empties += 1;
            }
        }
        assert_eq!(empties, 63);
        assert_eq!(pool.allocated_count(), 1);
    }

    #[test]
    fn deep_solid_into_interior_materializes_brick_with_interior_fill() {
        let (mut tree, mut pool) = deep_tree();
        // Mark a sub-region INTERIOR so we descend through interior
        // territory. Mark a single coord interior; the rest of the
        // brick will be EMPTY (because the only path that hits
        // INTERIOR_NODE at brick_depth is when an ancestor was
        // INTERIOR_NODE). Easier: paint the whole tree INTERIOR via
        // set_at_level at root, which is exactly the state a fully-
        // solid object would land in.
        tree.as_slice_mut()[0] = INTERIOR_NODE;

        let coord = UVec3::new(3, 3, 3);
        let prev = tree.set_cell_solid(coord, 88, &mut pool);
        assert!(prev.is_none());

        // The brick covering (0..4)³ was materialized with INTERIOR
        // fill + a solid slot at (3,3,3).
        let node = tree.lookup(coord).unwrap();
        assert!(is_brick(node));
        let bid = brick_id(node);
        let (lx, ly, lz) = brick_local(coord);
        assert_eq!(pool.get_cell(bid, lx, ly, lz), 88);

        let mut interior_count = 0;
        for cell in pool.brick_cells(bid) {
            if *cell == BRICK_INTERIOR {
                interior_count += 1;
            }
        }
        assert_eq!(interior_count, 63);

        // The rest of the tree still surfaces as INTERIOR_NODE at
        // other coords' brick-depth lookups.
        let other = UVec3::new(8, 8, 8);
        assert_eq!(tree.lookup(other), Some(INTERIOR_NODE));
    }

    #[test]
    fn deep_solid_into_interior_short_circuits_when_writing_interior() {
        // Writing INTERIOR into INTERIOR is a no-op — no brick should
        // be materialized.
        let (mut tree, mut pool) = deep_tree();
        tree.as_slice_mut()[0] = INTERIOR_NODE;
        let coord = UVec3::new(3, 3, 3);

        let prev = tree.set_cell_interior(coord, &mut pool);
        assert!(prev.is_none());
        assert_eq!(pool.allocated_count(), 0, "no brick should be allocated");
        assert_eq!(tree.lookup(coord), Some(INTERIOR_NODE));
    }

    #[test]
    fn deep_empty_into_empty_short_circuits() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 5, 5);
        let prev = tree.set_cell_empty(coord, &mut pool);
        assert!(prev.is_none());
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(tree.lookup(coord), Some(EMPTY_NODE));
    }

    #[test]
    fn deep_empty_into_interior_materializes_brick_with_one_hole() {
        // INTERIOR_NODE at root, set one cell EMPTY → brick filled
        // with INTERIOR + one BRICK_EMPTY.
        let (mut tree, mut pool) = deep_tree();
        tree.as_slice_mut()[0] = INTERIOR_NODE;
        let coord = UVec3::new(1, 1, 1);

        let prev = tree.set_cell_empty(coord, &mut pool);
        assert!(prev.is_none());

        let node = tree.lookup(coord).unwrap();
        assert!(is_brick(node));
        let bid = brick_id(node);
        let (lx, ly, lz) = brick_local(coord);
        assert_eq!(pool.get_cell(bid, lx, ly, lz), BRICK_EMPTY);
        let interior_count = pool
            .brick_cells(bid)
            .iter()
            .filter(|c| **c == BRICK_INTERIOR)
            .count();
        assert_eq!(interior_count, 63);
    }

    // ── BRICK cell mutations ──────────────────────────────────────

    #[test]
    fn brick_cell_solid_to_empty_to_solid_roundtrip() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 6, 7);

        tree.set_cell_solid(coord, 42, &mut pool);
        let bid_before = brick_id(tree.lookup(coord).unwrap());

        // Add a second slot in the same brick so it stays alive after
        // clearing the first.
        let other = UVec3::new(4, 4, 4);
        tree.set_cell_solid(other, 11, &mut pool);

        // Clear the first cell.
        let prev = tree.set_cell_empty(coord, &mut pool);
        assert_eq!(prev, Some(42));
        let bid_after = brick_id(tree.lookup(coord).unwrap());
        assert_eq!(bid_before, bid_after, "brick should not be freed yet");

        // Re-fill — should land on the same brick id, no reallocation.
        let prev2 = tree.set_cell_solid(coord, 77, &mut pool);
        assert!(prev2.is_none());
        let bid_finally = brick_id(tree.lookup(coord).unwrap());
        assert_eq!(bid_finally, bid_before);

        // Both cells live.
        let (cx, cy, cz) = brick_local(coord);
        let (ox, oy, oz) = brick_local(other);
        assert_eq!(pool.get_cell(bid_finally, cx, cy, cz), 77);
        assert_eq!(pool.get_cell(bid_finally, ox, oy, oz), 11);
    }

    #[test]
    fn brick_collapses_to_empty_node_after_last_cell_cleared() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 6, 7);

        tree.set_cell_solid(coord, 42, &mut pool);
        assert_eq!(pool.allocated_count(), 1);

        let prev = tree.set_cell_empty(coord, &mut pool);
        assert_eq!(prev, Some(42));
        // Brick was deallocated; node became EMPTY_NODE; ancestor
        // branches with all-EMPTY children collapsed all the way up.
        assert_eq!(tree.lookup(coord), Some(EMPTY_NODE));
        // Tree is back to a single EMPTY_NODE root.
        assert_eq!(tree.as_slice()[0], EMPTY_NODE);
        // Brick pool tail shrank.
        assert_eq!(pool.allocated_count(), 0);
        assert_no_collapsed_bricks_left_behind(&tree, &pool);
    }

    #[test]
    fn brick_collapses_to_interior_node_when_fully_filled_interior() {
        let (mut tree, mut pool) = deep_tree();
        // Start fresh (EMPTY everywhere). Mark all 64 cells of the
        // (4..8)³ brick INTERIOR. Brick should materialize on first
        // touch, fill with BRICK_INTERIOR as we write each cell, and
        // collapse back to INTERIOR_NODE on the 64th write.
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    tree.set_cell_interior(UVec3::new(x, y, z), &mut pool);
                }
            }
        }
        // Brick should be gone; the node should be INTERIOR_NODE.
        // Pick any coord in the region.
        assert_eq!(tree.lookup(UVec3::new(5, 5, 5)), Some(INTERIOR_NODE));
        assert_no_collapsed_bricks_left_behind(&tree, &pool);
    }

    // ── Ancestor branch collapse ─────────────────────────────────

    #[test]
    fn ancestor_branches_collapse_after_last_brick_freed() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 6, 7);

        tree.set_cell_solid(coord, 42, &mut pool);
        // Tree now has branches all the way down to brick_depth.
        let nodes_with_brick = tree.node_count();
        assert!(nodes_with_brick > 1, "expected branch nodes to be present");

        tree.set_cell_empty(coord, &mut pool);
        // After collapse: root is EMPTY_NODE; branch slots are
        // orphaned but try_collapse rolled them up so the root reads
        // as EMPTY_NODE.
        assert_eq!(tree.as_slice()[0], EMPTY_NODE);
    }

    #[test]
    fn ancestor_branches_collapse_to_interior_after_uniform_fill() {
        let (mut tree, mut pool) = deep_tree();
        // Fill the entire 16³ region with INTERIOR.
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    tree.set_cell_interior(UVec3::new(x, y, z), &mut pool);
                }
            }
        }
        // Should collapse all the way to the root.
        assert_eq!(tree.as_slice()[0], INTERIOR_NODE);
        assert_eq!(pool.allocated_count(), 0, "all bricks should have been freed during collapse");
    }

    // ── set_cell_solid replacing an existing brick cell slot ─────

    #[test]
    fn brick_cell_replace_returns_prev_slot() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 5, 5);
        tree.set_cell_solid(coord, 100, &mut pool);
        let prev = tree.set_cell_solid(coord, 200, &mut pool);
        assert_eq!(prev, Some(100));
        let bid = brick_id(tree.lookup(coord).unwrap());
        let (lx, ly, lz) = brick_local(coord);
        assert_eq!(pool.get_cell(bid, lx, ly, lz), 200);
    }

    #[test]
    fn brick_cell_to_interior_then_back_to_solid() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 5, 5);
        let other = UVec3::new(4, 4, 4);
        tree.set_cell_solid(coord, 100, &mut pool);
        tree.set_cell_solid(other, 11, &mut pool);

        let prev = tree.set_cell_interior(coord, &mut pool);
        assert_eq!(prev, Some(100));
        let bid = brick_id(tree.lookup(coord).unwrap());
        let (lx, ly, lz) = brick_local(coord);
        assert_eq!(pool.get_cell(bid, lx, ly, lz), BRICK_INTERIOR);

        // Other cell unchanged.
        let (ox, oy, oz) = brick_local(other);
        assert_eq!(pool.get_cell(bid, ox, oy, oz), 11);

        // Re-solid the cell.
        let prev2 = tree.set_cell_solid(coord, 222, &mut pool);
        assert!(prev2.is_none(), "BRICK_INTERIOR doesn't reference a slot, so prev is None");
        assert_eq!(pool.get_cell(bid, lx, ly, lz), 222);
    }

    // ── No-op short-circuits ─────────────────────────────────────

    #[test]
    fn writing_same_slot_returns_same_slot_and_no_brick_growth() {
        let (mut tree, mut pool) = deep_tree();
        let coord = UVec3::new(5, 5, 5);
        tree.set_cell_solid(coord, 42, &mut pool);
        let alloc_before = pool.allocated_count();
        let prev = tree.set_cell_solid(coord, 42, &mut pool);
        assert_eq!(prev, Some(42));
        assert_eq!(pool.allocated_count(), alloc_before);
    }

    // ── Bounds check ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_solid_oob_panics() {
        let (mut tree, mut pool) = deep_tree();
        tree.set_cell_solid(UVec3::new(16, 0, 0), 0, &mut pool);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_empty_oob_panics() {
        let (mut tree, mut pool) = deep_tree();
        tree.set_cell_empty(UVec3::new(0, 16, 0), &mut pool);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_interior_oob_panics() {
        let (mut tree, mut pool) = deep_tree();
        tree.set_cell_interior(UVec3::new(0, 0, 16), &mut pool);
    }

    // ── Internal_attr_index stays in sync ────────────────────────

    #[test]
    fn internal_attr_index_length_stays_in_sync_with_nodes() {
        let (mut tree, mut pool) = deep_tree();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    tree.set_cell_solid(UVec3::new(x, y, z), x + y * 4 + z * 16, &mut pool);
                }
            }
        }
        assert_eq!(tree.as_slice().len(), tree.internal_attr_slice().len());

        // After collapse, the parallel buffers must still match.
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    tree.set_cell_empty(UVec3::new(x, y, z), &mut pool);
                }
            }
        }
        assert_eq!(tree.as_slice().len(), tree.internal_attr_slice().len());
    }

    // ── D5.b: BrickPathCache ──────────────────────────────────────

    /// 64 cells in the same brick mutated in sequence through the
    /// cached path must produce an identical octree + brick-pool to
    /// the same edits through the uncached path. The cache should hit
    /// on all but the first edit of each pass; the result is
    /// observable through `lookup` and the cell values.
    #[test]
    fn cached_set_solid_matches_uncached_within_one_brick() {
        // depth=4 → brick_depth=2; the brick at brick_coord=(0,0,0)
        // covers cells (0..4, 0..4, 0..4) — 64 cells in one brick.
        let (mut t_ref, mut p_ref) = deep_tree();
        let (mut t_cached, mut p_cached) = deep_tree();

        let mut cache = BrickPathCache::new();
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let coord = UVec3::new(x, y, z);
                    let slot = x + y * 4 + z * 16 + 1;
                    let r1 = t_ref.set_cell_solid(coord, slot, &mut p_ref);
                    let r2 =
                        t_cached.set_cell_solid_cached(coord, slot, &mut p_cached, &mut cache);
                    assert_eq!(r1, r2, "prev slot mismatch at {coord}");
                }
            }
        }

        // Compare every cell.
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let coord = UVec3::new(x, y, z);
                    assert_eq!(
                        t_ref.cell_state(coord, &p_ref),
                        t_cached.cell_state(coord, &p_cached),
                        "cell mismatch at {coord}",
                    );
                }
            }
        }
        assert_eq!(t_ref.as_slice(), t_cached.as_slice(), "node arrays diverged");
    }

    /// Mutating cells across two different bricks with one shared
    /// cache must still produce the correct state. The cache should
    /// miss on the first edit of the second brick and re-walk.
    #[test]
    fn cached_path_handles_brick_boundary_crossings() {
        let (mut t_ref, mut p_ref) = deep_tree();
        let (mut t_cached, mut p_cached) = deep_tree();
        let mut cache = BrickPathCache::new();

        // Alternate between two bricks: (0,0,0)-(3,3,3) and (4,0,0)-(7,3,3).
        let coords: Vec<UVec3> = (0..8)
            .map(|i| UVec3::new(i, 0, 0))
            .chain((0..8).map(|i| UVec3::new(i, 1, 0)))
            .collect();

        for (idx, &coord) in coords.iter().enumerate() {
            let slot = idx as u32 + 1;
            let r1 = t_ref.set_cell_solid(coord, slot, &mut p_ref);
            let r2 = t_cached.set_cell_solid_cached(coord, slot, &mut p_cached, &mut cache);
            assert_eq!(r1, r2);
        }

        for &coord in &coords {
            assert_eq!(
                t_ref.cell_state(coord, &p_ref),
                t_cached.cell_state(coord, &p_cached),
            );
        }
    }

    /// Mid-batch brick collapse: fill a brick fully, then empty every
    /// cell. The brick should collapse mid-batch and the cache should
    /// invalidate cleanly without panicking or leaving the tree in a
    /// bad state.
    #[test]
    fn cached_path_handles_mid_batch_brick_collapse() {
        let (mut t_ref, mut p_ref) = deep_tree();
        let (mut t_cached, mut p_cached) = deep_tree();
        let mut cache = BrickPathCache::new();

        // Phase 1: fill the brick at (0,0,0).
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let coord = UVec3::new(x, y, z);
                    let slot = x + y * 4 + z * 16 + 1;
                    t_ref.set_cell_solid(coord, slot, &mut p_ref);
                    t_cached.set_cell_solid_cached(coord, slot, &mut p_cached, &mut cache);
                }
            }
        }

        // Phase 2: empty every cell in order. Both paths should end
        // with the brick collapsed back to EMPTY_NODE.
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let coord = UVec3::new(x, y, z);
                    t_ref.set_cell_empty(coord, &mut p_ref);
                    t_cached.set_cell_empty_cached(coord, &mut p_cached, &mut cache);
                }
            }
        }

        assert_eq!(
            t_ref.as_slice(),
            t_cached.as_slice(),
            "node arrays diverged after collapse",
        );
        assert_no_collapsed_bricks_left_behind(&t_cached, &p_cached);
    }

    /// Cached mutations through an INTERIOR_NODE region exercise the
    /// brick materialization path. Same behaviour as `set_cell_empty`
    /// against a fresh INTERIOR_NODE.
    #[test]
    fn cached_path_materializes_brick_in_interior_region() {
        let (mut t_ref, mut p_ref) = deep_tree();
        let (mut t_cached, mut p_cached) = deep_tree();

        // Seed the trees with a 4³ INTERIOR region.
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    t_ref.insert_interior(UVec3::new(x, y, z));
                    t_cached.insert_interior(UVec3::new(x, y, z));
                }
            }
        }

        let mut cache = BrickPathCache::new();
        // Empty out two cells within the brick. The brick must
        // materialize and the cells must read back as EMPTY.
        for &coord in &[UVec3::new(1, 1, 1), UVec3::new(2, 1, 1)] {
            t_ref.set_cell_empty(coord, &mut p_ref);
            t_cached.set_cell_empty_cached(coord, &mut p_cached, &mut cache);
        }

        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let coord = UVec3::new(x, y, z);
                    assert_eq!(
                        t_ref.cell_state(coord, &p_ref),
                        t_cached.cell_state(coord, &p_cached),
                    );
                }
            }
        }
    }
}
