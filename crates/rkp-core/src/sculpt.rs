//! Sculpt brush kernel — pure CPU octree mutation.
//!
//! Given a baked [`SparseOctree`] and a [`BrushOp`] in finest-voxel
//! grid coordinates, this module produces:
//!
//! * [`compute_brush_edits`] — a [`SculptDelta`] listing every leaf-level
//!   add/remove the brush would produce. No octree mutation, no slot
//!   allocation. Pure and deterministic; safe to call on a shared
//!   octree reference (e.g. an `Arc<SparseOctree>` shared across
//!   instances).
//! * [`apply_delta`] — applies a delta to a `&mut SparseOctree`, calling
//!   the caller's `alloc_slot` for each `Add` edit. Returns an
//!   [`AppliedDelta`] reporting (slot_id, attrs) pairs the caller must
//!   write into its [`LeafAttrPool`], and the slot ids that were freed
//!   by `Remove` edits.
//!
//! ## Coordinate convention
//!
//! Everything in this module is in **finest-voxel grid units**: brush
//! center / radius / cell coords are all on a `[0, 2^depth)` axis-aligned
//! integer grid. The caller (engine-side glue) converts from world
//! space using the entity transform + `grid_origin` + `voxel_size`
//! before calling.
//!
//! ## Phase 1 semantics (minimal)
//!
//! * **Carve (Subtract):** finest-voxel cells whose center lies inside
//!   the brush sphere AND whose current state is a leaf (Mixed) are
//!   marked `Remove` — the leaf disappears. Cells that are currently
//!   `Interior` (fully solid) stay `Interior`; newly-exposed surface
//!   from carving past the surface shell is *not* emitted in this
//!   phase. That leaves blocky edges around carved interiors —
//!   acceptable for the POC, and the natural place to handle it is
//!   the Phase 2 cluster re-bake (Surface Nets resamples the carved
//!   region from the mutated classification).
//! * **Raise (Add):** finest-voxel cells whose center lies inside the
//!   brush sphere AND whose current state is `Empty` are marked
//!   `Add { material, normal }`. The normal is the outward gradient
//!   of the brush sphere at the cell center — fine as a first cut;
//!   the Phase 2 re-bake will recompute via Surface Nets on the
//!   merged classification.
//! * Cells in any other state under either mode are no-ops.
//!
//! ## Out of scope for Phase 1
//!
//! * Newly-exposed interior surfaces from Carve (see above).
//! * Surface-Nets-style smoothed normals at the brush boundary.
//! * Material/normal carryover from neighbors (sculpt always uses the
//!   brush's chosen material — that's the design pillar).
//! * Slot allocation, scene_mgr integration, clone-on-write,
//!   geometry-epoch bump — all Phase 2 glue.

use glam::{UVec3, Vec3};

use crate::leaf_attr::{LeafAttr, pack_oct};
use crate::sparse_octree::{
    EMPTY_NODE, INTERIOR_NODE, SparseOctree, is_leaf, leaf_slot, make_leaf,
};

/// Add (clay) vs Subtract (dig). Matches the engine-side
/// `SculptMode::Raise` / `SculptMode::Carve` variants; this enum is
/// duplicated rather than imported to keep `rkp-core` free of any
/// dependency on `rkp-engine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushMode {
    /// Add geometry where the brush extends past the surface.
    Raise,
    /// Remove geometry where the brush overlaps the surface.
    Carve,
}

/// Brush parameters in finest-voxel grid coordinates.
///
/// `center` and `radius` are both on the `[0, 2^depth)` integer grid
/// (sub-voxel-cell precision is allowed via the `Vec3`). The caller
/// converts from world space using the entity transform +
/// `grid_origin` + `voxel_size` before calling [`compute_brush_edits`].
#[derive(Debug, Clone, Copy)]
pub struct BrushOp {
    pub center: Vec3,
    /// Brush radius in finest-voxel units. A radius of `1.0` covers
    /// roughly one cell along each axis.
    pub radius: f32,
    /// Smoothstep shoulder \[0, 1\]. `0.0` = hard sphere boundary at
    /// `radius`; `1.0` = smoothstep all the way from `0` to `radius`.
    /// Phase 1 uses this only to gate cells (`cell_center` is
    /// considered "in brush" iff `falloff_weight > 0`).
    pub falloff: f32,
    pub mode: BrushMode,
    /// Material assigned to leaves added by `Raise`. Carve doesn't
    /// consume this — removed leaves just disappear and the field is
    /// ignored.
    pub material: u16,
}

/// What to do at a single finest-voxel cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeafEditOp {
    /// Drop the leaf at this coord — node becomes [`EMPTY_NODE`].
    Remove,
    /// Add a leaf at this coord. Phase 2 glue allocates a slot via the
    /// caller-supplied `alloc_slot` and writes `material` + `normal`
    /// into the slot's [`LeafAttr`].
    Add { material: u16, normal: Vec3 },
}

/// A single leaf-level edit produced by [`compute_brush_edits`].
#[derive(Debug, Clone, Copy)]
pub struct LeafEdit {
    pub coord: UVec3,
    pub op: LeafEditOp,
}

/// Output of [`compute_brush_edits`]: the list of leaf-level changes
/// the brush would produce against the octree.
#[derive(Debug, Clone, Default)]
pub struct SculptDelta {
    pub edits: Vec<LeafEdit>,
}

impl SculptDelta {
    pub fn is_empty(&self) -> bool { self.edits.is_empty() }
    pub fn len(&self) -> usize { self.edits.len() }
    pub fn count_added(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count()
    }
    pub fn count_removed(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::Remove)).count()
    }
}

/// Attributes to write into a freshly-allocated leaf slot. Returned by
/// [`apply_delta`] paired with the slot id the caller's `alloc_slot`
/// produced.
#[derive(Debug, Clone, Copy)]
pub struct LeafEditAttrs {
    pub material: u16,
    pub normal: Vec3,
}

impl LeafEditAttrs {
    /// Build a [`LeafAttr`] ready for [`LeafAttrPool`] write: packs the
    /// normal octahedrally and uses the configured `material` as the
    /// primary (secondary stays 0, blend weight stays 0 — sculpt-added
    /// leaves don't blend).
    pub fn to_leaf_attr(&self) -> LeafAttr {
        LeafAttr {
            normal_oct: pack_oct(self.normal),
            material_primary: self.material,
            material_secondary_blend: 0,
        }
    }
}

/// Result of [`apply_delta`]. Caller still owes the pool the matching
/// writes / deallocations.
#[derive(Debug, Default)]
pub struct AppliedDelta {
    /// One entry per `Add` edit, in delta order. `slot_id` is whatever
    /// the caller's `alloc_slot` produced; `attrs` is the data the
    /// caller must write into that slot's [`LeafAttr`].
    pub allocated_slots: Vec<(u32, LeafEditAttrs)>,
    /// Slot ids freed by `Remove` edits. Caller should
    /// `pool.deallocate_range(slot, 1)` for each (or batch-collect into
    /// `(start, count)` ranges).
    pub freed_slots: Vec<u32>,
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Brush SDF at point `p`: negative inside the brush, positive outside,
/// zero on the boundary. Used both to gate "is this cell affected" and
/// to derive a per-cell normal (Add case).
#[inline]
fn brush_sdf(p: Vec3, op: &BrushOp) -> f32 {
    (p - op.center).length() - op.radius
}

/// Brush outward normal at `p`: unit vector pointing away from
/// `op.center`. Defined for any `p != op.center`; degenerate at the
/// center (returns +Y as a stable fallback so packed-normal encoding
/// never produces NaN).
#[inline]
fn brush_outward_normal(p: Vec3, op: &BrushOp) -> Vec3 {
    let d = p - op.center;
    let len_sq = d.length_squared();
    if len_sq < 1e-12 {
        Vec3::Y
    } else {
        d * len_sq.sqrt().recip()
    }
}

/// Conservative axis-aligned brush bounds in finest-voxel grid units,
/// clamped to `[0, extent)`. Returned as `(min_inclusive, max_exclusive)`.
fn brush_cell_range(op: &BrushOp, extent: u32) -> (UVec3, UVec3) {
    let min_f = op.center - Vec3::splat(op.radius);
    let max_f = op.center + Vec3::splat(op.radius);
    let lo = UVec3::new(
        min_f.x.floor().max(0.0) as u32,
        min_f.y.floor().max(0.0) as u32,
        min_f.z.floor().max(0.0) as u32,
    );
    let hi = UVec3::new(
        (max_f.x.ceil().max(0.0) as u32 + 1).min(extent),
        (max_f.y.ceil().max(0.0) as u32 + 1).min(extent),
        (max_f.z.ceil().max(0.0) as u32 + 1).min(extent),
    );
    // Empty-range guard: any axis where lo >= hi short-circuits to a
    // zero-cell range. Caller's nested loop naturally yields no cells
    // for that range.
    (lo, hi)
}

// ── Public API ───────────────────────────────────────────────────────

/// Walk every finest-voxel cell intersecting the brush AABB and emit a
/// [`LeafEdit`] when the brush's effect on that cell is non-trivial.
/// Pure CPU; doesn't allocate slots or touch the octree.
///
/// Cells whose center sits inside the brush sphere are classified
/// against the current octree state at that coord:
///
/// * **Carve** on `Mixed` (leaf) → `Remove`
/// * **Raise** on `Empty` → `Add { brush.material, outward normal }`
/// * any other combination → no edit
///
/// The edit list is emitted in row-major Z-Y-X order — stable across
/// runs, which keeps slot allocation deterministic when paired with a
/// monotonic `alloc_slot`.
pub fn compute_brush_edits(octree: &SparseOctree, op: BrushOp) -> SculptDelta {
    let extent = octree.extent();
    let (lo, hi) = brush_cell_range(&op, extent);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let mut edits = Vec::new();
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let coord = UVec3::new(x, y, z);
                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                if brush_sdf(cell_center, &op) > 0.0 {
                    continue;
                }
                // Cell center is inside the brush sphere. Classify
                // against current octree state and emit an edit when
                // the brush has work to do here.
                let Some(node) = octree.lookup(coord) else { continue };
                match op.mode {
                    BrushMode::Carve => {
                        if is_leaf(node) {
                            edits.push(LeafEdit { coord, op: LeafEditOp::Remove });
                        }
                        // Interior cells under Carve stay Interior in
                        // Phase 1 — the newly-exposed-surface case is
                        // Phase 2's cluster re-bake job.
                    }
                    BrushMode::Raise => {
                        if node == EMPTY_NODE {
                            let normal = brush_outward_normal(cell_center, &op);
                            edits.push(LeafEdit {
                                coord,
                                op: LeafEditOp::Add {
                                    material: op.material,
                                    normal,
                                },
                            });
                        }
                        // Mixed (already a surface leaf) under Raise:
                        // keep existing leaf. Sculpt is not paint —
                        // the brush material doesn't overwrite the
                        // surface, the user picks the paint tool for
                        // that. Interior under Raise: cell is already
                        // solid; no-op.
                    }
                }
                // Suppress the unused-field warning for `falloff` — it
                // currently only gates as the binary "is the cell
                // inside the sphere" test above; later phases will use
                // it to weight the smoothstep boundary.
                let _ = op.falloff;
                let _ = INTERIOR_NODE; // referenced via re-export above; silences unused-import warning if logic later drops it.
            }
        }
    }
    SculptDelta { edits }
}

/// Apply a [`SculptDelta`] to a mutable octree.
///
/// For each `Remove` edit: look up the leaf's current slot id (record
/// it in `freed_slots`) and write `EMPTY_NODE` at the finest level.
/// For each `Add` edit: call `alloc_slot()` to get a fresh slot id,
/// record `(slot, attrs)` in `allocated_slots`, and write a `LEAF`
/// node referencing that slot.
///
/// The caller is responsible for:
/// * Writing `attrs.to_leaf_attr()` (and a default color, if used)
///   into each freshly-allocated slot in its [`LeafAttrPool`].
/// * Deallocating the slot ids in `freed_slots`.
/// * Bumping the scene's geometry epoch so the renderer re-uploads.
pub fn apply_delta(
    octree: &mut SparseOctree,
    delta: &SculptDelta,
    mut alloc_slot: impl FnMut() -> u32,
) -> AppliedDelta {
    let depth = octree.depth();
    let mut allocated_slots = Vec::with_capacity(delta.count_added());
    let mut freed_slots = Vec::with_capacity(delta.count_removed());

    for edit in &delta.edits {
        match edit.op {
            LeafEditOp::Remove => {
                if let Some(node) = octree.lookup(edit.coord) {
                    if is_leaf(node) {
                        freed_slots.push(leaf_slot(node));
                    }
                }
                octree.set_at_level(edit.coord, depth, EMPTY_NODE);
            }
            LeafEditOp::Add { material, normal } => {
                let slot = alloc_slot();
                allocated_slots.push((slot, LeafEditAttrs { material, normal }));
                octree.set_at_level(edit.coord, depth, make_leaf(slot));
            }
        }
    }

    AppliedDelta { allocated_slots, freed_slots }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a single-leaf octree at the given coord with a
    /// known slot id, on a tree of the given depth. Everything else
    /// stays EMPTY.
    fn one_leaf(depth: u8, coord: UVec3, slot: u32) -> SparseOctree {
        let mut t = SparseOctree::new(depth, 1.0);
        t.insert(coord, slot);
        t
    }

    /// Helper: count leaves in the tree.
    fn leaf_count(t: &SparseOctree) -> usize { t.leaf_count() }

    /// Helper: assert a coord is EMPTY in the tree.
    fn assert_empty(t: &SparseOctree, coord: UVec3) {
        assert_eq!(t.lookup(coord), Some(EMPTY_NODE), "expected EMPTY at {coord}");
    }
    /// Helper: assert a coord holds a leaf with the given slot id.
    fn assert_leaf(t: &SparseOctree, coord: UVec3, slot: u32) {
        let node = t.lookup(coord).expect("in bounds");
        assert!(is_leaf(node), "expected leaf at {coord}, got 0x{:08X}", node);
        assert_eq!(leaf_slot(node), slot, "wrong slot at {coord}");
    }

    /// Brush radius ≥ √3/2 covers a unit cell's diagonal — used to
    /// build brushes that surely include a particular cell center.
    const CELL_DIAG_HALF: f32 = 0.8660254; // > √3/2

    #[test]
    fn carve_drops_matching_leaf() {
        // depth=3 → 8³ tree. One leaf at (4, 4, 4) with slot=42.
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        assert_eq!(leaf_count(&t), 1);

        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, op);
        assert_eq!(delta.count_removed(), 1);
        assert_eq!(delta.count_added(), 0);

        let applied = apply_delta(&mut t, &delta, || panic!("no allocations expected"));
        assert_eq!(applied.freed_slots, vec![42]);
        assert_eq!(leaf_count(&t), 0);
        assert_empty(&t, UVec3::new(4, 4, 4));
    }

    #[test]
    fn carve_misses_when_brush_outside() {
        // Brush far from the leaf — no edits.
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        let op = BrushOp {
            center: Vec3::new(0.5, 0.5, 0.5),
            radius: 1.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, op);
        assert!(delta.is_empty());
        let applied = apply_delta(&mut t, &delta, || panic!("no alloc"));
        assert!(applied.allocated_slots.is_empty());
        assert!(applied.freed_slots.is_empty());
        assert_eq!(leaf_count(&t), 1);
    }

    #[test]
    fn carve_skips_interior_in_phase1() {
        // depth=2 → 4³ tree, all INTERIOR. Carve should produce *no*
        // edits in Phase 1 — newly-exposed surfaces come from the
        // cluster re-bake (Phase 2), not the kernel.
        let mut t = SparseOctree::new(2, 1.0);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    t.insert_interior(UVec3::new(x, y, z));
                }
            }
        }
        // The whole tree collapses to a single INTERIOR_NODE.
        assert_eq!(t.lookup(UVec3::new(2, 2, 2)), Some(INTERIOR_NODE));

        let op = BrushOp {
            center: Vec3::new(2.0, 2.0, 2.0),
            radius: 1.5,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, op);
        assert!(delta.is_empty(), "Phase 1 Carve emits no edits on pure-Interior cells");

        let _ = apply_delta(&mut t, &delta, || panic!("no alloc"));
        // Tree is unchanged.
        assert_eq!(t.lookup(UVec3::new(2, 2, 2)), Some(INTERIOR_NODE));
    }

    #[test]
    fn raise_adds_into_empty() {
        // depth=3 → 8³ tree, all EMPTY.
        let mut t = SparseOctree::new(3, 1.0);
        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 7,
        };
        let delta = compute_brush_edits(&t, op);
        assert_eq!(delta.count_added(), 1);
        assert_eq!(delta.count_removed(), 0);

        // Allocator hands out monotonically increasing ids.
        let mut next = 100u32;
        let applied = apply_delta(&mut t, &delta, || {
            let s = next;
            next += 1;
            s
        });
        assert_eq!(applied.allocated_slots.len(), 1);
        let (slot, attrs) = applied.allocated_slots[0];
        assert_eq!(slot, 100);
        assert_eq!(attrs.material, 7);
        // Center cell is at the brush center → degenerate normal
        // falls back to +Y (see `brush_outward_normal`).
        assert!((attrs.normal - Vec3::Y).length() < 1e-6);

        // The kernel did write the leaf into the tree.
        assert_leaf(&t, UVec3::new(4, 4, 4), 100);
        assert_eq!(leaf_count(&t), 1);
    }

    #[test]
    fn raise_skips_mixed_keeps_existing_leaf() {
        // The cell already has a surface leaf — Raise must not
        // overwrite it (sculpt is not paint).
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 99,
        };
        let delta = compute_brush_edits(&t, op);
        assert!(delta.is_empty());
        let applied = apply_delta(&mut t, &delta, || panic!("no alloc"));
        assert!(applied.allocated_slots.is_empty());
        assert_leaf(&t, UVec3::new(4, 4, 4), 42);
    }

    #[test]
    fn raise_skips_interior() {
        let mut t = SparseOctree::new(3, 1.0);
        for z in 3..5 { for y in 3..5 { for x in 3..5 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(4.0, 4.0, 4.0),
            radius: 0.5,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, op);
        assert!(delta.is_empty(), "Raise on Interior cells is a no-op");
    }

    #[test]
    fn brush_completely_outside_bounds() {
        let mut t = SparseOctree::new(3, 1.0); // extent = 8
        // Brush centered well outside the cube — no cells in range.
        let op = BrushOp {
            center: Vec3::new(-10.0, -10.0, -10.0),
            radius: 1.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, op);
        assert!(delta.is_empty());

        // And on the far side.
        let op2 = BrushOp { center: Vec3::splat(100.0), ..op };
        let delta2 = compute_brush_edits(&t, op2);
        assert!(delta2.is_empty());

        // Sanity: applying empty delta leaves the tree untouched.
        let applied = apply_delta(&mut t, &delta, || panic!());
        assert!(applied.allocated_slots.is_empty());
    }

    #[test]
    fn carve_shell_drops_only_intersected_leaves() {
        // Build a small "surface shell" of leaves at z=4 across x=2..6, y=2..6.
        // depth=3 → 8³ tree.
        let mut t = SparseOctree::new(3, 1.0);
        let mut slot = 0u32;
        for y in 2..6 {
            for x in 2..6 {
                t.insert(UVec3::new(x, y, 4), slot);
                slot += 1;
            }
        }
        let total = leaf_count(&t);
        assert_eq!(total, 16);

        // Carve a small brush at (3.5, 3.5, 4.5) with radius ~1.5 —
        // should overlap a few cells of the shell.
        let op = BrushOp {
            center: Vec3::new(3.5, 3.5, 4.5),
            radius: 1.5,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, op);
        let n_removed = delta.count_removed();
        assert!(n_removed > 0 && n_removed < total,
            "expected partial overlap, got {n_removed} of {total}");

        let applied = apply_delta(&mut t, &delta, || panic!("no alloc"));
        assert_eq!(applied.freed_slots.len(), n_removed);
        // The remaining leaf count matches.
        assert_eq!(leaf_count(&t), total - n_removed);
    }

    #[test]
    fn delta_count_matches_apply() {
        // Mixed delta (both adds and removes) → applied counts match.
        let mut t = SparseOctree::new(3, 1.0);
        // Half: leaves at z=0 row; half: empty everywhere else.
        for x in 0..4u32 {
            t.insert(UVec3::new(x, 0, 0), 1000 + x);
        }

        // Build a synthetic mixed delta manually.
        let delta = SculptDelta { edits: vec![
            LeafEdit { coord: UVec3::new(0, 0, 0), op: LeafEditOp::Remove },
            LeafEdit { coord: UVec3::new(1, 0, 0), op: LeafEditOp::Remove },
            LeafEdit { coord: UVec3::new(0, 1, 0), op: LeafEditOp::Add {
                material: 5, normal: Vec3::Y,
            }},
            LeafEdit { coord: UVec3::new(1, 1, 0), op: LeafEditOp::Add {
                material: 5, normal: Vec3::Y,
            }},
        ]};
        assert_eq!(delta.count_removed(), 2);
        assert_eq!(delta.count_added(), 2);

        let mut next = 7000u32;
        let applied = apply_delta(&mut t, &delta, || {
            let s = next; next += 1; s
        });
        assert_eq!(applied.allocated_slots.len(), 2);
        assert_eq!(applied.freed_slots.len(), 2);
        // Old slots came back in delta order.
        assert_eq!(applied.freed_slots, vec![1000, 1001]);
        // New slots got allocated in delta order.
        assert_eq!(applied.allocated_slots[0].0, 7000);
        assert_eq!(applied.allocated_slots[1].0, 7001);
        // Final state: rows swapped.
        assert_empty(&t, UVec3::new(0, 0, 0));
        assert_empty(&t, UVec3::new(1, 0, 0));
        assert_leaf(&t, UVec3::new(0, 1, 0), 7000);
        assert_leaf(&t, UVec3::new(1, 1, 0), 7001);
        // The untouched part of the row stayed.
        assert_leaf(&t, UVec3::new(2, 0, 0), 1002);
        assert_leaf(&t, UVec3::new(3, 0, 0), 1003);
    }

    #[test]
    fn raise_outward_normal_points_away_from_brush_center() {
        // depth=4 → 16³. Brush at (8, 8, 8), big enough radius to span
        // multiple cells. One cell at (10, 8, 8) → its center is at
        // (10.5, 8.5, 8.5) so the outward normal must point in +X.
        let t = SparseOctree::new(4, 1.0);
        let op = BrushOp {
            center: Vec3::new(8.0, 8.5, 8.5),
            radius: 4.0,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, op);

        let edge_edit = delta.edits.iter().find(|e| e.coord == UVec3::new(10, 8, 8))
            .expect("brush reaches (10, 8, 8)");
        let LeafEditOp::Add { normal, .. } = edge_edit.op else {
            panic!("expected Add at (10, 8, 8)");
        };
        assert!(normal.x > 0.95, "outward normal should point +X, got {normal:?}");
        assert!(normal.y.abs() < 0.2);
        assert!(normal.z.abs() < 0.2);
    }
}
