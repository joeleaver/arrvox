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
//! ## Kernel rule set (neighbor-adjacency, post-2026-05-15)
//!
//! Conceptually Carve and Raise are SDF operations on the asset's
//! occupancy field:
//!
//! ```text
//!   Carve:  D_new = max(D_obj, -D_brush)
//!   Raise:  D_new = min(D_obj,  D_brush)
//! ```
//!
//! Discretised on the finest-voxel grid, the rules become:
//!
//! **Carve:**
//! * Inside-brush + `Solid` → `Remove` (carve the surface leaf).
//! * Inside-brush + `Interior` → `Empty` (carve the bulk; deep carve
//!   straight through).
//! * Inside-brush + `Empty` → no-op (already air).
//! * **Outside-brush** + `Interior` + has any 6-face neighbor that's
//!   inside-brush AND will flip to `Empty` (i.e. was `Solid` or
//!   `Interior`) → `Add { material }` (cavity wall: a previously-bulk
//!   cell becomes the newly-exposed surface on the cavity rim).
//!
//! **Raise:**
//! * Inside-brush + `Empty` → `SetInterior` (new clay bulk).
//! * Inside-brush + `Solid`/`Interior` → no-op (sculpt isn't paint).
//! * **Outside-brush** + `Empty` + has any 6-face neighbor that's
//!   inside-brush AND will flip to `Interior` (i.e. was `Empty`) →
//!   `Add { material }` (dome surface: the outside-rim air cell
//!   becomes the new surface against the newly-raised bulk).
//!
//! The earlier ½-voxel-band rule placed the surface band *inside* the
//! brush and emitted `Add` for cells whose SDF distance fell within
//! `[-0.5, 0]`. That fuzzy threshold produced pinholes (the discrete
//! cell grid would have neighbours randomly fall in or out of the
//! band) and never created cavity walls on thin-shell assets (which
//! have no `Interior` cells at all). The neighbour-adjacency rule
//! above is a discrete yes/no per cell — no pinholes — and the
//! cavity-wall band lives on the *outside* rim of the brush, where
//! the surrounding asset bulk actually is. The kernel walks an AABB
//! that's padded by 1 cell to cover that outside rim.
//!
//! ## Out of scope for the current rule set
//!
//! * Brush falloff / per-stamp strength — every inside-brush cell is
//!   treated uniformly, so each Raise stamp adds bulk equal to the
//!   brush hemisphere volume. Real sculpting brushes feather. Follow-up.
//! * Material / normal carryover from neighbours — sculpt always uses
//!   the brush's chosen material (design pillar). Normals are
//!   recomputed from the post-stamp occupancy gradient in
//!   [`apply_delta`].
//! * Slot allocation, scene_mgr integration, clone-on-write,
//!   geometry-epoch bump — all glue lives at the caller.

use glam::{IVec3, UVec3, Vec3};

use crate::brick_pool::BrickPool;
use crate::leaf_attr::{LeafAttr, pack_oct};
use crate::sparse_octree::{CellState, OctreeMutationLog, SparseOctree};
#[cfg(test)]
use crate::sparse_octree::INTERIOR_NODE;

// The `assert_*` helpers in this file's test module reference the
// brick-cell sentinels and the LEAF / BRICK predicates directly.
// `#[cfg(test)]` keeps the imports out of the release build's
// unused-import set.
#[cfg(test)]
use crate::brick_pool::{BRICK_DIM, BRICK_EMPTY};
#[cfg(test)]
use crate::sparse_octree::{EMPTY_NODE, brick_id, is_brick, is_leaf, leaf_slot};

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
    /// Drop the surface leaf at this coord — cell becomes EMPTY. The
    /// caller's [`LeafAttrPool`] reclaims the prior slot id surfaced
    /// in [`AppliedDelta::freed_slots`].
    Remove,
    /// Make the cell EMPTY regardless of prior state. Semantically the
    /// same as `Remove` for kernel V1; the variant exists so future
    /// kernel variants can distinguish "drop the surface I knew about"
    /// from "clear bulk to air" (deep-Carve through INTERIOR).
    Empty,
    /// Mark the cell as INTERIOR (occupied bulk, no visible surface).
    /// Used for deep-Raise to add solid mass beyond the brush boundary.
    SetInterior,
    /// Add a surface leaf at this coord. The caller's `alloc_slot`
    /// produces the slot id; the primitive writes the LEAF/BRICK_cell
    /// encoding referencing that slot. The normal stored on the
    /// resulting [`LeafAttr`] is computed by [`apply_delta`] as a
    /// post-stamp 6-tap occupancy gradient — that matches what
    /// Surface Nets uses for vertex normals on the same cells.
    Add { material: u16 },
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
    /// **D9.a** — per-call breakdown of the work
    /// [`compute_brush_edits`] did to produce `edits`. Lets the caller
    /// log how much of the `edits` phase budget went to outer-loop
    /// iteration vs cell-state lookups vs neighbor probes.
    pub timing: ComputeBrushEditsTiming,
}

impl SculptDelta {
    pub fn is_empty(&self) -> bool { self.edits.is_empty() }
    pub fn len(&self) -> usize { self.edits.len() }
    pub fn count_added(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count()
    }
    pub fn count_removed(&self) -> usize {
        self.edits
            .iter()
            .filter(|e| matches!(e.op, LeafEditOp::Remove | LeafEditOp::Empty))
            .count()
    }
    pub fn count_interior(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::SetInterior)).count()
    }
}

/// **D9.a** — instrumentation captured during [`compute_brush_edits`].
/// All counters are per-call.
///
/// The dominant unknowns are which sub-cost dominates the 1.6-2.4 ms
/// edits phase observed on splat5 Carve drags: the AABB walk itself,
/// the per-cell SDF cull, the [`SparseOctree::cell_state`] octree
/// walks, or the per-Empty-cell [`has_outside_solid_neighbor`] (which
/// fires up to 6 extra cell_state calls).
#[derive(Debug, Default, Clone, Copy)]
pub struct ComputeBrushEditsTiming {
    /// Total cells the outer `(z, y, x)` triple-loop iterated.
    pub n_aabb_cells: u32,
    /// Cells that survived the SDF brush-sphere cull (`d <= 0`).
    pub n_inside_sphere: u32,
    /// Cells whose state was actually resolved via
    /// [`SparseOctree::cell_state`] (= `n_inside_sphere`, minus any
    /// `OutOfBounds` early-outs).
    pub n_cell_state_calls: u32,
    /// Invocations of [`has_outside_solid_neighbor`] from the Carve
    /// path's "Empty cell" branch. Each fires up to 6 extra
    /// `cell_state` calls (face-neighbor checks).
    pub n_neighbor_calls: u32,
    /// Wall-clock time of the whole `compute_brush_edits` call body.
    pub t_total_ns: u64,
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
    /// Pre-existing SOLID slots whose post-stamp gradient differs from
    /// the stored normal — typically `(slot, new_normal)` pairs for
    /// SOLID cells that border this stamp's mutations. Without this, a
    /// Raise that refills a prior Carve cavity leaves the Carve's
    /// cavity-wall cells with inward-pointing normals (computed when
    /// the cavity was empty), so they back-face-cull from the new
    /// dome and half the surface vanishes. The caller patches each
    /// slot's `LeafAttr.normal_oct` in place; material stays.
    pub renormalized_slots: Vec<(u32, Vec3)>,
    /// Slot ids freed by `Remove` edits. Caller should
    /// `pool.deallocate_range(slot, 1)` for each (or batch-collect into
    /// `(start, count)` ranges).
    pub freed_slots: Vec<u32>,
    /// Every write made to the octree's `nodes[]` / `internal_attr_index[]`
    /// during this `apply_delta`. The caller (typically the render-side
    /// scene manager) replays these writes into its packed GPU buffer
    /// so the CPU and GPU octrees stay in sync. The log's
    /// `initial_node_count` lets the caller detect growth — if the
    /// tree's `node_count()` after `apply_delta` exceeds the original,
    /// the existing GPU allocator slot is too small and a re-allocation
    /// is required.
    pub octree_log: OctreeMutationLog,
    /// **D5.a** — per-sub-phase wall-clock timing collected inside
    /// [`apply_delta`]. Lets the caller log a breakdown that splits
    /// the mutation-log setup/take from the actual edit-application
    /// loop, with per-op-type loop times.
    pub timing: ApplyDeltaTiming,
}

/// **D5.a** — wall-clock timing breakdown for [`apply_delta`]. All
/// fields are nanoseconds.
///
/// The loop is split into three sequential sub-passes (one per
/// op-type) so each can be timed independently. compute_brush_edits
/// emits at most one edit per cell so reordering Empty → Interior
/// → Add across cells is correctness-neutral; within each pass
/// edit order is preserved.
///
/// The mutation log records every node write in occurrence order;
/// reordering doesn't affect the final octree state but can change
/// the log's intermediate sequence. Downstream consumers
/// (`OctreeGpu::apply_mutation_log`) apply writes in order and
/// converge to the same final state.
#[derive(Debug, Default, Clone, Copy)]
pub struct ApplyDeltaTiming {
    pub t_log_setup_ns: u64,
    pub t_loop_empty_ns: u64,
    pub t_loop_interior_ns: u64,
    pub t_loop_add_ns: u64,
    pub t_log_take_ns: u64,
    pub n_empty: u32,
    pub n_interior: u32,
    pub n_add: u32,
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Brush SDF at point `p`: negative inside the brush, positive outside,
/// zero on the boundary.
#[inline]
fn brush_sdf(p: Vec3, op: &BrushOp) -> f32 {
    (p - op.center).length() - op.radius
}

/// Conservative axis-aligned brush bounds in finest-voxel grid units,
/// clamped to `[0, extent)`. Returned as `(min_inclusive, max_exclusive)`.
///
/// This is the "brush footprint" range — cells whose center sits inside
/// the brush sphere live entirely within `[lo, hi)`. Used by external
/// callers (cluster spatial-index queries, etc.) that only need to know
/// which cells the brush directly affects.
///
/// [`compute_brush_edits`] uses a +1-cell-padded version of this range
/// to walk the *outside* rim too — the cavity-wall / dome-surface band
/// lives on the outside-rim cells adjacent to the brush interior.
pub fn brush_cell_range(op: &BrushOp, extent: u32) -> (UVec3, UVec3) {
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

/// 6-tap central-difference gradient of the binary occupancy field
/// (`1.0` if SOLID/INTERIOR/OutOfBounds-treated-as-bulk... wait, see
/// below) sampled at `coord`'s 6 face-neighbours. Returns a unit
/// vector pointing from solid into empty — the outward surface normal
/// for both cavity walls (which face into the carved-out interior)
/// and dome surfaces (which face away from raised bulk).
///
/// **OOB treatment.** Cells outside the octree extent count as EMPTY,
/// so a cell on the asset boundary gets a normal pointing outward
/// (toward the OOB side). That matches the "outside is air" mental
/// model SN uses.
///
/// **Degenerate case.** If all six neighbours have identical
/// occupancy, the gradient is zero. We fall back to `+Y` — same
/// stable fallback the old brush-radial helper used at the brush
/// center.
fn gradient_normal(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    coord: UVec3,
    cache: &mut crate::sparse_octree::CellStateCache,
) -> Vec3 {
    let extent = octree.extent() as i32;
    let c = IVec3::new(coord.x as i32, coord.y as i32, coord.z as i32);
    let mut occ = |p: IVec3| -> f32 {
        if p.x < 0 || p.y < 0 || p.z < 0 || p.x >= extent || p.y >= extent || p.z >= extent {
            return 0.0;
        }
        let pu = UVec3::new(p.x as u32, p.y as u32, p.z as u32);
        match octree.cell_state_cached(pu, brick_pool, cache) {
            CellState::Empty | CellState::OutOfBounds => 0.0,
            CellState::Solid(_) | CellState::Interior => 1.0,
        }
    };
    let gx = occ(c - IVec3::X) - occ(c + IVec3::X);
    let gy = occ(c - IVec3::Y) - occ(c + IVec3::Y);
    let gz = occ(c - IVec3::Z) - occ(c + IVec3::Z);
    let g = Vec3::new(gx, gy, gz);
    let len_sq = g.length_squared();
    if len_sq < 1e-6 { Vec3::Y } else { g * len_sq.sqrt().recip() }
}

/// Same as [`brush_cell_range`] but inflated by `pad` cells on each
/// side. The compute kernel uses this with `pad=1` so the walk covers
/// the outside-rim cells whose state matters for the cavity-wall /
/// dome-surface rules.
fn brush_cell_range_padded(op: &BrushOp, extent: u32, pad: u32) -> (UVec3, UVec3) {
    let (lo, hi) = brush_cell_range(op, extent);
    let lo_p = UVec3::new(lo.x.saturating_sub(pad), lo.y.saturating_sub(pad), lo.z.saturating_sub(pad));
    let hi_p = UVec3::new((hi.x + pad).min(extent), (hi.y + pad).min(extent), (hi.z + pad).min(extent));
    (lo_p, hi_p)
}

// ── Public API ───────────────────────────────────────────────────────

/// Walk every finest-voxel cell intersecting the brush AABB and emit a
/// [`LeafEdit`] per Phase B's SDF rule set.
///
/// **Carve (`D_new = max(D_obj, -D_brush)`):**
/// * `Solid` cell inside the brush → `Remove`. The surface leaf goes
///   away.
/// * `Interior` cell inside the brush, within ½ voxel of the brush
///   boundary → `Add { material, outward normal }`. This is the
///   newly-exposed cavity-wall cell on a solid body — the
///   previously-bulk material becomes the new surface.
/// * `Interior` cell inside the brush, **deeper** than ½ voxel from
///   the boundary → `Empty`. The brush has carved through bulk; the
///   cell becomes air. (Surface Nets on the mutated occupancy will
///   regenerate the cavity wall at the ½-voxel band picked up by the
///   rule above.)
/// * `Empty` cell inside the brush AND with at least one 6-face
///   neighbor that is `Solid`-or-`Interior`-AND-outside-the-brush →
///   `Add` (thin-shell cavity-wall case; closes the brush sphere as
///   it crosses an EMPTY interior cavity).
///
/// **Raise (`D_new = min(D_obj, D_brush)`):**
/// * `Empty` cell inside the brush, within ½ voxel of boundary →
///   `Add { material, outward normal }` (new clay surface along the
///   brush sphere).
/// * `Empty` cell inside the brush, deeper than ½ voxel from the
///   boundary → `SetInterior` (new clay bulk that Surface Nets will
///   surface around).
/// * `Solid` / `Interior` cells under Raise stay unchanged — sculpt is
///   not paint; the brush material doesn't overwrite existing surface.
///
/// The edit list is emitted in row-major Z-Y-X order — stable across
/// runs, which keeps slot allocation deterministic when paired with a
/// monotonic `alloc_slot`.
pub fn compute_brush_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    op: BrushOp,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    // The walker covers the brush footprint PLUS one cell of padding so
    // the outside-rim cells whose state drives the cavity-wall /
    // dome-surface rules are in scope.
    let (lo, hi) = brush_cell_range_padded(&op, extent, 1);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let t_start = Instant::now();
    let mut edits = Vec::new();
    let mut n_aabb_cells = 0u32;
    let mut n_inside_sphere = 0u32;
    let mut n_cell_state_calls = 0u32;
    let mut n_neighbor_calls = 0u32;
    // Shared brick cache for the outer cell-state lookup and the
    // per-outside-rim 6-face probes. Row-major walk shares bricks for
    // runs of up to BRICK_DIM=4 cells; neighbour probes mostly stay
    // inside the source cell's brick (5-of-6 face-neighbors of an
    // interior cell are same-brick).
    let mut cache = crate::sparse_octree::CellStateCache::new();
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let coord = UVec3::new(x, y, z);
                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let d = brush_sdf(cell_center, &op);
                let inside = d <= 0.0;
                let state = octree.cell_state_cached(coord, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                if matches!(state, CellState::OutOfBounds) {
                    continue;
                }
                if inside {
                    n_inside_sphere += 1;
                }
                match (op.mode, inside) {
                    (BrushMode::Carve, true) => emit_carve_inside(&mut edits, coord, state),
                    (BrushMode::Carve, false) => emit_carve_outside_rim(
                        &mut edits, coord, state, &op, octree, brick_pool,
                        &mut cache, &mut n_neighbor_calls,
                    ),
                    (BrushMode::Raise, true) => emit_raise_inside(&mut edits, coord, state),
                    (BrushMode::Raise, false) => emit_raise_outside_rim(
                        &mut edits, coord, state, &op, octree, brick_pool,
                        &mut cache, &mut n_neighbor_calls,
                    ),
                }
                let _ = op.falloff;
            }
        }
    }

    SculptDelta {
        edits,
        timing: ComputeBrushEditsTiming {
            n_aabb_cells,
            n_inside_sphere,
            n_cell_state_calls,
            n_neighbor_calls,
            t_total_ns: t_start.elapsed().as_nanos() as u64,
        },
    }
}

/// Carve, inside-brush cell. `Solid` → Remove; `Interior` → Empty (deep
/// carve through bulk). EMPTY stays as-is.
#[inline]
fn emit_carve_inside(edits: &mut Vec<LeafEdit>, coord: UVec3, state: CellState) {
    match state {
        CellState::Solid(_) => edits.push(LeafEdit { coord, op: LeafEditOp::Remove }),
        CellState::Interior => edits.push(LeafEdit { coord, op: LeafEditOp::Empty }),
        _ => {}
    }
}

/// Carve, outside-brush cell. Only `Interior` cells with an inside-brush
/// 6-neighbour that will flip to `Empty` (Solid or Interior pre-stamp)
/// become cavity-wall `Add` cells.
#[inline]
fn emit_carve_outside_rim(
    edits: &mut Vec<LeafEdit>,
    coord: UVec3,
    state: CellState,
    op: &BrushOp,
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    cache: &mut crate::sparse_octree::CellStateCache,
    n_neighbor_calls: &mut u32,
) {
    if !matches!(state, CellState::Interior) {
        return;
    }
    *n_neighbor_calls += 1;
    if has_inside_brush_neighbor_in(
        coord, op, octree, brick_pool, cache,
        |s| matches!(s, CellState::Solid(_) | CellState::Interior),
    ) {
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material } });
    }
}

/// Raise, inside-brush cell. `Empty` → SetInterior (clay bulk).
/// Solid/Interior cells stay (sculpt isn't paint).
#[inline]
fn emit_raise_inside(edits: &mut Vec<LeafEdit>, coord: UVec3, state: CellState) {
    if matches!(state, CellState::Empty) {
        edits.push(LeafEdit { coord, op: LeafEditOp::SetInterior });
    }
}

/// Raise, outside-brush cell. Only `Empty` cells with an inside-brush
/// 6-neighbour that will flip to `Interior` (Empty pre-stamp) become
/// dome-surface `Add` cells.
#[inline]
fn emit_raise_outside_rim(
    edits: &mut Vec<LeafEdit>,
    coord: UVec3,
    state: CellState,
    op: &BrushOp,
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    cache: &mut crate::sparse_octree::CellStateCache,
    n_neighbor_calls: &mut u32,
) {
    if !matches!(state, CellState::Empty) {
        return;
    }
    *n_neighbor_calls += 1;
    if has_inside_brush_neighbor_in(
        coord, op, octree, brick_pool, cache,
        |s| matches!(s, CellState::Empty),
    ) {
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material } });
    }
}

/// Return `true` if any of the 6 face-neighbours of `coord` sits
/// *inside* the brush sphere AND its current cell state satisfies the
/// caller-supplied predicate. The predicate lets Carve and Raise share
/// the same probe logic — Carve wants neighbours that will flip to
/// EMPTY (currently Solid/Interior); Raise wants neighbours that will
/// flip to INTERIOR (currently Empty).
#[inline]
fn has_inside_brush_neighbor_in(
    coord: UVec3,
    op: &BrushOp,
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    cache: &mut crate::sparse_octree::CellStateCache,
    pred: impl Fn(CellState) -> bool,
) -> bool {
    const FACE_DIRS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];
    let extent = octree.extent() as i32;
    let c = IVec3::new(coord.x as i32, coord.y as i32, coord.z as i32);
    for dir in FACE_DIRS {
        let n = c + dir;
        if n.x < 0 || n.y < 0 || n.z < 0 || n.x >= extent || n.y >= extent || n.z >= extent {
            continue;
        }
        let n_u = UVec3::new(n.x as u32, n.y as u32, n.z as u32);
        let n_center = Vec3::new(n.x as f32 + 0.5, n.y as f32 + 0.5, n.z as f32 + 0.5);
        if brush_sdf(n_center, op) > 0.0 {
            // Neighbour is also outside-brush — doesn't anchor the rim.
            continue;
        }
        let n_state = octree.cell_state_cached(n_u, brick_pool, cache);
        if pred(n_state) {
            return true;
        }
    }
    false
}

/// Apply a [`SculptDelta`] to a mutable octree + brick pool.
///
/// For each `Remove` edit: clear the cell to EMPTY via
/// [`SparseOctree::set_cell_empty`] and surface the previous
/// leaf_attr slot in `freed_slots` so the caller can release it
/// from its [`LeafAttrPool`].
///
/// For each `Add` edit: call `alloc_slot()` to obtain a fresh
/// slot id, write SOLID at the cell via
/// [`SparseOctree::set_cell_solid`], and record `(slot, attrs)` in
/// `allocated_slots` so the caller can write the [`LeafAttr`].
/// If the cell already held a slot (rare for brush ops, but possible
/// for cavity-wall edits that overwrite an INTERIOR-just-promoted-to-
/// SOLID cell), the prior slot id is also surfaced in `freed_slots`.
///
/// `Empty` and `SetInterior` mirror Remove/Add for the cavity-wall
/// rule set: they let the kernel emit "this cell is now bulk" or
/// "this cell is now air" edits independent of whether it previously
/// held a surface slot.
///
/// The caller is responsible for:
/// * Writing `attrs.to_leaf_attr()` (and a default color, if used)
///   into each freshly-allocated slot in its [`LeafAttrPool`].
/// * Deallocating the slot ids in `freed_slots`.
/// * Bumping the scene's geometry epoch so the renderer re-uploads.
pub fn apply_delta(
    octree: &mut SparseOctree,
    brick_pool: &mut BrickPool,
    delta: &SculptDelta,
    mut alloc_slot: impl FnMut() -> u32,
) -> AppliedDelta {
    use crate::sparse_octree::BrickPathCache;
    use std::time::Instant;

    let mut allocated_slots = Vec::with_capacity(delta.count_added());
    let mut freed_slots = Vec::with_capacity(delta.count_removed());

    // ── Setup ────────────────────────────────────────────────────
    let t0 = Instant::now();
    octree.begin_mutation_log();
    let t_setup_ns = t0.elapsed().as_nanos() as u64;

    // ── Loop, split by op-type so each can be timed independently.
    //
    // compute_brush_edits guarantees one edit per cell, so reordering
    // across op-types is correctness-neutral; within each pass the
    // original edit order is preserved. The mutation log's intermediate
    // sequence changes but the final octree state — and therefore the
    // replayed GPU state — is identical.
    //
    // **D5.b** — each pass threads a [`BrickPathCache`] through the
    // `*_cached` mutation primitives. compute_brush_edits walks the
    // brush region in row-major (x, y, z) order so consecutive edits
    // typically share a brick (up to 4 cells in a row, plus same-row
    // neighbors in (y, z)). Cache hits skip the 9-level descent and
    // call `mutate_at_brick` directly — measured per-op cost on D5.a
    // was ~60 ns (descent-dominated) on splat5 elephant Carve.
    //
    // A fresh cache per pass avoids cross-op-type stale entries
    // (e.g., the Empty pass leaves the cache pointing at a brick
    // whose state the Add pass might invalidate).

    let mut n_empty = 0u32;
    let t_empty = Instant::now();
    {
        let mut cache = BrickPathCache::new();
        for edit in &delta.edits {
            if matches!(edit.op, LeafEditOp::Remove | LeafEditOp::Empty) {
                n_empty += 1;
                if let Some(prev) =
                    octree.set_cell_empty_cached(edit.coord, brick_pool, &mut cache)
                {
                    freed_slots.push(prev);
                }
            }
        }
    }
    let t_loop_empty_ns = t_empty.elapsed().as_nanos() as u64;

    let mut n_interior = 0u32;
    let t_interior = Instant::now();
    {
        let mut cache = BrickPathCache::new();
        for edit in &delta.edits {
            if matches!(edit.op, LeafEditOp::SetInterior) {
                n_interior += 1;
                if let Some(prev) =
                    octree.set_cell_interior_cached(edit.coord, brick_pool, &mut cache)
                {
                    freed_slots.push(prev);
                }
            }
        }
    }
    let t_loop_interior_ns = t_interior.elapsed().as_nanos() as u64;

    let mut n_add = 0u32;
    let t_add = Instant::now();
    let mut add_coords: Vec<UVec3> = Vec::with_capacity(delta.count_added());
    {
        let mut cache = BrickPathCache::new();
        for edit in &delta.edits {
            if let LeafEditOp::Add { material } = edit.op {
                n_add += 1;
                let slot = alloc_slot();
                // Normal is filled in below from a post-stamp gradient;
                // +Y is a safe placeholder if the gradient is degenerate.
                allocated_slots.push((slot, LeafEditAttrs { material, normal: Vec3::Y }));
                add_coords.push(edit.coord);
                if let Some(prev) =
                    octree.set_cell_solid_cached(edit.coord, slot, brick_pool, &mut cache)
                {
                    // Caller must free the displaced slot too.
                    freed_slots.push(prev);
                }
            }
        }
    }
    // Gradient-normal pass: now that every mutation is applied, the
    // octree's occupancy field reflects the post-stamp surface. A 6-tap
    // central-difference reads the local gradient — pointing from
    // SOLID/INTERIOR toward EMPTY, which is the outward-facing normal
    // both cavity walls and dome surfaces want. This matches the
    // gradient Surface Nets uses for its mesh vertex normals on the
    // same cells, so per-pixel resolved normals and SN-extracted
    // vertex normals agree at the sculpt-region seam.
    {
        let mut grad_cache = crate::sparse_octree::CellStateCache::new();
        // `allocated_slots` is only pushed during the Add pass above,
        // so indices line up 1:1 with `add_coords`.
        for (idx, coord) in add_coords.iter().enumerate() {
            allocated_slots[idx].1.normal =
                gradient_normal(octree, brick_pool, *coord, &mut grad_cache);
        }
    }
    let t_loop_add_ns = t_add.elapsed().as_nanos() as u64;

    // Renormalize pass deliberately removed (2026-05-15).
    // Naive 6-face renormalize of any SOLID neighbour of an edit
    // inverts the ground's normal whenever a Raise piles INTERIOR
    // bulk above it: a ground cell at y=0 originally had +Y EMPTY
    // and -Y OOB → gradient.y=0, fallback +Y. After the Raise, +Y
    // becomes INTERIOR while -Y stays OOB → gradient.y=-1 → normal
    // flips to -Y. The ground back-face-culls and disappears under
    // the dome. The right fix needs to discriminate "this cell's
    // surface lives on the face whose state changed" from "this
    // cell's surface lives elsewhere on the cell" — punted until we
    // have a real repro of the underlying stale-normal case.
    let renormalized_slots: Vec<(u32, Vec3)> = Vec::new();

    // ── Teardown ─────────────────────────────────────────────────
    let t_take = Instant::now();
    let octree_log = octree.take_mutation_log().unwrap_or_default();
    let t_log_take_ns = t_take.elapsed().as_nanos() as u64;

    AppliedDelta {
        allocated_slots,
        renormalized_slots,
        freed_slots,
        octree_log,
        timing: ApplyDeltaTiming {
            t_log_setup_ns: t_setup_ns,
            t_loop_empty_ns,
            t_loop_interior_ns,
            t_loop_add_ns,
            t_log_take_ns,
            n_empty,
            n_interior,
            n_add,
        },
    }
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

    /// Helper: assert a coord is EMPTY in the tree (works for LEAF or
    /// BRICK encoding).
    fn assert_empty(t: &SparseOctree, brick_pool: &BrickPool, coord: UVec3) {
        let node = t.lookup(coord).expect("in bounds");
        if node == EMPTY_NODE {
            return;
        }
        if is_brick(node) {
            let bid = brick_id(node);
            let mask = BRICK_DIM - 1;
            let cell = brick_pool.get_cell(bid, coord.x & mask, coord.y & mask, coord.z & mask);
            assert_eq!(cell, BRICK_EMPTY, "expected BRICK_EMPTY at {coord}, got 0x{cell:08X}");
            return;
        }
        panic!("expected EMPTY at {coord}, got 0x{node:08X}");
    }

    /// Helper: assert a coord holds a surface slot with the given id.
    /// Resolves LEAF nodes directly and BRICK nodes via the brick pool.
    fn assert_leaf(t: &SparseOctree, brick_pool: &BrickPool, coord: UVec3, slot: u32) {
        let node = t.lookup(coord).expect("in bounds");
        if is_leaf(node) {
            assert_eq!(leaf_slot(node), slot, "wrong LEAF slot at {coord}");
            return;
        }
        if is_brick(node) {
            let bid = brick_id(node);
            let mask = BRICK_DIM - 1;
            let cell = brick_pool.get_cell(bid, coord.x & mask, coord.y & mask, coord.z & mask);
            assert_eq!(cell, slot, "wrong BRICK cell slot at {coord}");
            return;
        }
        panic!("expected surface slot at {coord}, got 0x{node:08X}");
    }

    /// Brush radius ≥ √3/2 covers a unit cell's diagonal — used to
    /// build brushes that surely include a particular cell center.
    const CELL_DIAG_HALF: f32 = 0.8660254; // > √3/2

    /// Brick pool sized for the small synthetic trees in these tests.
    /// Sculpt edits on depth-3 trees don't trigger brick materialization
    /// (LEAFs live at finest depth, where mutate_at_finest handles them),
    /// so the pool stays unused for most tests — present only because
    /// `apply_delta` takes `&mut BrickPool` as a structural parameter.
    fn fresh_pool() -> BrickPool { BrickPool::new(8) }

    #[test]
    fn carve_drops_matching_leaf() {
        // depth=3 → 8³ tree. One leaf at (4, 4, 4) with slot=42.
        // (`insert` writes LEAF at finest depth; brick-aware mutate
        // descends past brick_depth=1 and lands on the LEAF directly.)
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        let mut pool = fresh_pool();
        assert_eq!(leaf_count(&t), 1);

        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert_eq!(delta.count_removed(), 1);
        assert_eq!(delta.count_added(), 0);

        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!("no allocations expected"));
        assert_eq!(applied.freed_slots, vec![42]);
        assert_eq!(leaf_count(&t), 0);
        assert_empty(&t, &pool, UVec3::new(4, 4, 4));
    }

    #[test]
    fn carve_misses_when_brush_outside() {
        // Brush far from the leaf — no edits.
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(0.5, 0.5, 0.5),
            radius: 1.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.is_empty());
        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!("no alloc"));
        assert!(applied.allocated_slots.is_empty());
        assert!(applied.freed_slots.is_empty());
        assert_eq!(leaf_count(&t), 1);
    }

    #[test]
    fn carve_solid_cube_makes_hemispherical_cavity() {
        // Carve on a solid INTERIOR body. Every inside-brush cell flips
        // to EMPTY (deep carve); cavity-wall cells form on the OUTSIDE
        // rim where outside-brush INTERIOR cells touch the carved-out
        // interior. depth=4 → 16³ tree, all INTERIOR_NODE.
        let mut t = SparseOctree::new(4, 1.0);
        let pool = fresh_pool();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    t.insert_interior(UVec3::new(x, y, z));
                }
            }
        }
        assert_eq!(t.lookup(UVec3::new(8, 8, 8)), Some(INTERIOR_NODE));

        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(
            delta.count_added() > 0,
            "expected outside-rim cavity-wall ADD edits; got {} adds / {} empties",
            delta.count_added(), delta.count_removed(),
        );
        assert!(
            delta.count_removed() > 0,
            "expected deep-carve Empty edits inside the brush",
        );
        // No SetInterior edits — those are Raise-only.
        assert_eq!(delta.count_interior(), 0);
    }

    #[test]
    fn raise_adds_into_empty() {
        // depth=3 → 8³ tree, all EMPTY. brick_depth=1, so apply_delta
        // materializes a brick at level 1 covering (4..8, 4..8, 4..8).
        //
        // New rule: inside-brush EMPTY → SetInterior; outside-brush
        // EMPTY with inside-brush EMPTY neighbor → Add (dome surface).
        // A radius-0.4 brush centered on (4.5,4.5,4.5) catches exactly
        // one inside-brush cell (4,4,4) and its six face-neighbours
        // become outside-rim Add cells.
        let mut t = SparseOctree::new(3, 1.0);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: 0.4,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert_eq!(delta.count_interior(), 1, "expected one SetInterior bulk cell");
        assert_eq!(delta.count_added(), 6, "expected one face-neighbour Add per axis");
        assert_eq!(delta.count_removed(), 0);

        // Allocator hands out monotonically increasing ids.
        let mut next = 100u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || {
            let s = next;
            next += 1;
            s
        });
        assert_eq!(applied.allocated_slots.len(), 6);
        for (slot, attrs) in &applied.allocated_slots {
            assert!(*slot >= 100);
            assert_eq!(attrs.material, 7);
        }
        // Each Add cell sits adjacent to the newly-INTERIOR centre, so
        // its gradient normal points along the brush radius (away from
        // (4,4,4)).
        let add_idx = delta.edits.iter().position(|e| e.coord == UVec3::new(5, 4, 4) && matches!(e.op, LeafEditOp::Add { .. }));
        let add_idx = add_idx.expect("cell (5,4,4) should be an Add");
        // The Add pass iterates edits in order and pushes to
        // allocated_slots in the same order. Find which Add-pass index
        // (5,4,4) lives at by counting prior Adds in the delta.
        let add_pass_idx = delta.edits[..add_idx].iter().filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count();
        let (_, attrs) = applied.allocated_slots[add_pass_idx];
        assert!(attrs.normal.x > 0.95, "+X dome cell should have +X normal; got {:?}", attrs.normal);
        // Cell (4,4,4) is now INTERIOR bulk.
        let center_state = t.cell_state(UVec3::new(4, 4, 4), &pool);
        assert!(matches!(center_state, CellState::Interior), "centre cell should be INTERIOR, got {:?}", center_state);
    }

    #[test]
    fn raise_skips_mixed_keeps_existing_leaf() {
        // The cell already has a surface leaf — Raise must not
        // overwrite it (sculpt is not paint).
        let mut t = one_leaf(3, UVec3::new(4, 4, 4), 42);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 99,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.is_empty());
        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!("no alloc"));
        assert!(applied.allocated_slots.is_empty());
        assert_leaf(&t, &pool, UVec3::new(4, 4, 4), 42);
    }

    #[test]
    fn raise_skips_interior() {
        let mut t = SparseOctree::new(3, 1.0);
        let pool = fresh_pool();
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
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.is_empty(), "Raise on Interior cells is a no-op");
    }

    #[test]
    fn brush_completely_outside_bounds() {
        let mut t = SparseOctree::new(3, 1.0); // extent = 8
        let pool = fresh_pool();
        // Brush centered well outside the cube — no cells in range.
        let op = BrushOp {
            center: Vec3::new(-10.0, -10.0, -10.0),
            radius: 1.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.is_empty());

        // And on the far side.
        let op2 = BrushOp { center: Vec3::splat(100.0), ..op };
        let delta2 = compute_brush_edits(&t, &pool, op2);
        assert!(delta2.is_empty());

        // Sanity: applying empty delta leaves the tree untouched.
        let mut pool = fresh_pool();
        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!());
        assert!(applied.allocated_slots.is_empty());
    }

    #[test]
    fn carve_shell_drops_only_intersected_leaves() {
        // Build a small "surface shell" of leaves at z=4 across x=2..6, y=2..6.
        // depth=3 → 8³ tree.
        let mut t = SparseOctree::new(3, 1.0);
        let mut pool = fresh_pool();
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
        let delta = compute_brush_edits(&t, &pool, op);
        let n_removed = delta.count_removed();
        assert!(n_removed > 0 && n_removed < total,
            "expected partial overlap, got {n_removed} of {total}");

        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!("no alloc"));
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
        let delta = SculptDelta { timing: Default::default(), edits: vec![
            LeafEdit { coord: UVec3::new(0, 0, 0), op: LeafEditOp::Remove },
            LeafEdit { coord: UVec3::new(1, 0, 0), op: LeafEditOp::Remove },
            LeafEdit { coord: UVec3::new(0, 1, 0), op: LeafEditOp::Add { material: 5 } },
            LeafEdit { coord: UVec3::new(1, 1, 0), op: LeafEditOp::Add { material: 5 } },
        ]};
        assert_eq!(delta.count_removed(), 2);
        assert_eq!(delta.count_added(), 2);

        let mut pool = fresh_pool();
        let mut next = 7000u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || {
            let s = next; next += 1; s
        });
        assert_eq!(applied.allocated_slots.len(), 2);
        assert_eq!(applied.freed_slots.len(), 2);
        // Old slots came back in delta order.
        assert_eq!(applied.freed_slots, vec![1000, 1001]);
        // New slots got allocated in delta order.
        assert_eq!(applied.allocated_slots[0].0, 7000);
        assert_eq!(applied.allocated_slots[1].0, 7001);
        // Final state: rows swapped. Reading via the brick-aware
        // helpers since Adds may have landed in a BRICK at brick_depth.
        assert_empty(&t, &pool, UVec3::new(0, 0, 0));
        assert_empty(&t, &pool, UVec3::new(1, 0, 0));
        assert_leaf(&t, &pool, UVec3::new(0, 1, 0), 7000);
        assert_leaf(&t, &pool, UVec3::new(1, 1, 0), 7001);
        // The untouched part of the row stayed.
        assert_leaf(&t, &pool, UVec3::new(2, 0, 0), 1002);
        assert_leaf(&t, &pool, UVec3::new(3, 0, 0), 1003);
    }

    #[test]
    fn raise_outward_normal_points_away_from_brush_center() {
        // depth=4 → 16³. Brush at (8, 8.5, 8.5), radius 4.0. The
        // outside-rim Add cell at (12, 8, 8) lies just outside the
        // brush's +X face. Its gradient normal — computed in
        // `apply_delta` from the post-stamp occupancy — should point
        // in +X (away from the newly-INTERIOR bulk).
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(8.0, 8.5, 8.5),
            radius: 4.0,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        // Find the index of (12, 8, 8) in the Add subsequence so we
        // can pull its post-stamp normal from `allocated_slots`.
        let target = UVec3::new(12, 8, 8);
        let add_pos_in_edits = delta.edits.iter().position(|e| {
            e.coord == target && matches!(e.op, LeafEditOp::Add { .. })
        }).expect("expected an Add at (12, 8, 8) on the +X brush rim");
        let add_pass_idx = delta.edits[..add_pos_in_edits]
            .iter().filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count();

        let mut next = 0u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || { let s = next; next += 1; s });
        let (_, attrs) = applied.allocated_slots[add_pass_idx];
        assert!(attrs.normal.x > 0.85, "outward normal should point ~+X, got {:?}", attrs.normal);
    }

    // ── R2b cavity-wall + INTERIOR tests ─────────────────────────

    #[test]
    fn raise_deep_into_empty_emits_set_interior() {
        // depth=4 → 16³. Brush at (8,8,8) radius 3.0. Inside-brush
        // EMPTY cells become SetInterior (bulk); the outside-rim ring
        // of cells touching that bulk becomes Add (dome surface).
        let t = SparseOctree::new(4, 1.0);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.count_interior() > 0, "expected SetInterior edits for deep Raise");
        assert!(delta.count_added() > 0, "expected outside-rim dome-surface Add edits");
        assert_eq!(delta.count_removed(), 0);
    }

    #[test]
    fn carve_thin_shell_punches_clean_hole_no_cavity_walls() {
        // A "thin shell" asset: one SOLID layer at z=3, EMPTY
        // everywhere else (no INTERIOR bulk). Carving through the
        // shell should remove the inside-brush shell cells and leave
        // ZERO cavity-wall Adds — the rule only produces cavity walls
        // from outside-brush INTERIOR cells, and there are none. This
        // is the "tunnelling through is fine" case for thin shells.
        let mut t = SparseOctree::new(3, 1.0);
        let pool = fresh_pool();
        let mut slot = 0u32;
        for y in 0..8 {
            for x in 0..8 {
                t.insert(UVec3::new(x, y, 3), slot);
                slot += 1;
            }
        }
        let op = BrushOp {
            center: Vec3::new(4.0, 4.0, 3.5),
            radius: 1.5,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        assert!(delta.count_removed() > 0, "expected Remove edits on the shell");
        assert_eq!(delta.count_added(), 0, "thin shell has no INTERIOR to form cavity walls");
    }

    #[test]
    fn apply_delta_captures_octree_writes() {
        // Sculpt into a tree with a brick — apply_delta should record
        // the brick-cell mutation paths (subdivision, brick materialize,
        // brick collapse). For a shallow tree with bricks disabled,
        // single set_cell writes still record through the finest-leaf
        // path.
        let mut t = SparseOctree::new(2, 1.0); // shallow → no bricks
        let mut pool = fresh_pool();
        // Pre-seed a leaf so its Remove path records the finest-leaf write.
        t.insert(UVec3::new(0, 0, 0), 42);

        let delta = SculptDelta { timing: Default::default(), edits: vec![
            LeafEdit { coord: UVec3::new(0, 0, 0), op: LeafEditOp::Remove },
            LeafEdit { coord: UVec3::new(1, 0, 0), op: LeafEditOp::Add { material: 7 } },
        ]};
        let applied = apply_delta(&mut t, &mut pool, &delta, || 99);
        // Either path mutated nodes — log must have writes.
        assert!(
            !applied.octree_log.node_writes.is_empty(),
            "expected node writes to be recorded, got {applied:?}",
        );
        assert!(applied.octree_log.initial_node_count > 0);
    }

    #[test]
    fn apply_delta_log_detects_growth() {
        // Sculpting into a virgin EMPTY_NODE tree forces subdivision
        // down to brick_depth — that grows nodes.len() past the initial
        // count. The log's `grew()` flag should fire.
        let mut t = SparseOctree::new(4, 1.0); // depth 4 → grows on first mutation
        let mut pool = fresh_pool();
        let initial = t.node_count();
        let delta = SculptDelta { timing: Default::default(), edits: vec![
            LeafEdit { coord: UVec3::new(2, 2, 2), op: LeafEditOp::Add { material: 1 } },
        ]};
        let applied = apply_delta(&mut t, &mut pool, &delta, || 0);
        assert_eq!(applied.octree_log.initial_node_count as usize, initial);
        // Tree grew via subdivision of the root EMPTY_NODE.
        assert!(applied.octree_log.grew(t.node_count() as u32));
    }

    #[test]
    fn raise_dome_outside_rim_is_pinhole_free() {
        // Stamp a Raise brush into an all-EMPTY 16³ tree. The outside-
        // rim Add cells should form a continuous shell — every face-
        // neighbour of a SetInterior bulk cell that sits outside the
        // brush gets promoted. With the new adjacency rule there's no
        // ½-voxel-band distance threshold to fall in and out of, so
        // the shell is hole-free by construction.
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff: 0.0,
            mode: BrushMode::Raise,
            material: 4,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        let mut next = 0u32;
        let _applied = apply_delta(&mut t, &mut pool, &delta, || { let s = next; next += 1; s });

        // Every Add cell sits on the outside rim. Walk each one and
        // verify it has at least one INTERIOR face-neighbour (the bulk
        // it bounds). Any "floating" Add cell with no INTERIOR neighbour
        // would be a pinhole-adjacent ghost — the regression case the
        // old ½-voxel-band rule produced.
        let mut adds = 0;
        for edit in &delta.edits {
            if !matches!(edit.op, LeafEditOp::Add { .. }) {
                continue;
            }
            adds += 1;
            let c = edit.coord;
            let mut has_interior = false;
            for d in [IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z] {
                let n = IVec3::new(c.x as i32, c.y as i32, c.z as i32) + d;
                if n.x < 0 || n.y < 0 || n.z < 0 || n.x >= 16 || n.y >= 16 || n.z >= 16 { continue; }
                let nu = UVec3::new(n.x as u32, n.y as u32, n.z as u32);
                if matches!(t.cell_state(nu, &pool), CellState::Interior) {
                    has_interior = true;
                    break;
                }
            }
            assert!(has_interior, "Add cell {c} has no INTERIOR neighbour — would be a pinhole");
        }
        assert!(adds >= 30, "expected a dense dome-surface ring, got only {adds} Add cells");
    }

    #[test]
    fn gradient_normal_cavity_wall_points_into_cavity() {
        // Carve a solid INTERIOR cube. A cavity-wall cell on the +X
        // side of the brush should have its gradient normal pointing
        // in -X (toward brush center = into the carved-out cavity).
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        // Pick an Add edit on the +X cavity rim. Brush radius 3.0 at
        // (8,8,8) makes cell (10,8,8) the outermost inside-brush cell
        // along +X (centre at (10.5,8.5,8.5), distance ≈2.6 from brush
        // centre). The outside-rim cavity-wall cell is its +X neighbour
        // (11,8,8) — outside-brush INTERIOR with an inside-brush
        // INTERIOR neighbour that's about to flip to EMPTY.
        let target = UVec3::new(11, 8, 8);
        let pos = delta.edits.iter().position(|e| {
            e.coord == target && matches!(e.op, LeafEditOp::Add { .. })
        }).expect("expected Add at +X rim (11,8,8)");
        let add_pass_idx = delta.edits[..pos]
            .iter().filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count();
        let mut next = 0u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || { let s = next; next += 1; s });
        let (_, attrs) = applied.allocated_slots[add_pass_idx];
        assert!(attrs.normal.x < -0.85, "+X cavity wall should face -X (into cavity); got {:?}", attrs.normal);
    }

    #[test]
    fn carve_solid_block_makes_cavity_walls_at_boundary() {
        // Stamping a brush into a solid (INTERIOR) cube produces a
        // hemispherical cavity. Every inside-brush cell flips to
        // EMPTY (Empty edit); the outside-brush INTERIOR rim cells
        // that touch the carved region become Add (cavity wall).
        // depth=3 (8³), all INTERIOR. Brush at (4, 4, 4) radius 2.0.
        let mut t = SparseOctree::new(3, 1.0);
        let pool = fresh_pool();
        for z in 0..8 { for y in 0..8 { for x in 0..8 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        assert_eq!(t.lookup(UVec3::new(0, 0, 0)), Some(INTERIOR_NODE));

        let op = BrushOp {
            center: Vec3::new(4.0, 4.0, 4.0),
            radius: 2.0,
            falloff: 0.0,
            mode: BrushMode::Carve,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, op);
        // Walls at the brush boundary band + holes in the deep
        // interior.
        let walls = delta.count_added();
        let holes = delta.count_removed();
        assert!(walls > 0, "expected cavity-wall ADD edits");
        assert!(holes > 0, "expected deep-carve Empty edits");
        // All ADD edits use the brush material.
        for edit in &delta.edits {
            if let LeafEditOp::Add { material, .. } = edit.op {
                assert_eq!(material, 9);
            }
        }
    }
}
