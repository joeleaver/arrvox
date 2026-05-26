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
use crate::leaf_attr::{LeafAttr, pack_oct, unpack_oct};
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
/// duplicated rather than imported to keep `arvx-core` free of any
/// dependency on `arvx-engine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushMode {
    /// Hard SDF union — fill the brush sphere with material.
    Raise,
    /// Hard SDF subtract — empty the brush sphere.
    Carve,
    /// Soft outward dilation. For each cell in the brush AABB, the
    /// effective per-cell thickness in voxels is `ceil(falloff(t) *
    /// strength)` with `t = 1 − d/r`; empty cells within that many
    /// 6-face hops of the existing surface flip to solid.
    Inflate,
    /// Soft inward erosion. Mirror of [`Inflate`]: surface cells
    /// within `ceil(falloff(t) * strength)` 6-face hops of empty
    /// space flip to empty, and the newly-exposed interior cells
    /// emit cavity-wall Adds on the outside rim.
    Deflate,
    /// Clay strips — deposits a fixed-height flat-topped strip above
    /// the pre-stroke surface, swept along the capsule axis. Width =
    /// brush radius, height = `strength` cells. The cross-section has
    /// a flat top (75% of radius) with falloff-shaped shoulders at
    /// the lateral edges. Overlapping strokes stack: each adds its
    /// fixed height on top of whatever surface was there when the
    /// stroke began. Uses the same brushfire infrastructure as Inflate
    /// but with a flat-top thickness profile instead of a dome.
    ClayStrip,
    /// Geometry-and-normal smoothing. Per cell inside the brush capsule
    /// the kernel reads pre-stamp 6-neighbour occupancy from a dense
    /// scratch grid and applies one of three rules:
    ///
    /// * **Cavity-fill** — `Empty` cell with `≥ 4` occupied 6-neighbours
    ///   (Solid or Interior) flips to `Solid`. Fills isolated pits and
    ///   thin concavities back into the bulk.
    /// * **Bump-shave** — `Solid` cell with `≤ 2` occupied 6-neighbours
    ///   (equivalently `≥ 4` Empty) is `Remove`d. Erases isolated
    ///   1-voxel bumps and thin protrusions.
    /// * **Normal blend** — `Solid` cells that survive the morph rule
    ///   keep their slot but blend their `LeafAttr.normal_oct` toward
    ///   the local 6-neighbour normal average, weighted by
    ///   `falloff_curve(t) * (strength / 32)`.
    ///
    /// A second pass walks the +1 rim outside the capsule and emits
    /// cavity-wall `Add`s for any `Interior` cell whose 6-neighbour
    /// just got `Remove`d — same pattern Carve / Deflate use to keep
    /// the bulk watertight when its surface shell is breached.
    ///
    /// Multiple stamps along a stroke accumulate: a slow drag converges
    /// toward a locally-smooth silhouette (morph cleans up bumps; the
    /// surviving surface gets its normals averaged). Unlike Inflate /
    /// Deflate, Smooth has no transit-brushfire cap — each stamp is an
    /// independent local relaxation step.
    Smooth,
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
    /// Start of the brush segment for capsule-sweep stamps. Equal to
    /// `center` for single stamps and the first stamp of a stroke; on
    /// subsequent drag stamps it carries the previous stamp's `center`
    /// so the kernel evaluates the cell against the *swept volume*
    /// from `segment_start` to `center` rather than an isolated sphere.
    /// Adjacent spheres along a drag produce visible meeting-circle
    /// creases (see the screenshot in `project_sculpt_phase1_session_
    /// 2026_05_15c`); a capsule eliminates them — its surface is
    /// smooth across the cylindrical body and the two hemispherical
    /// caps. When `segment_start == center` every capsule operation
    /// collapses to the sphere it used to be, so existing tests + the
    /// first stamp of a stroke keep working unchanged.
    pub segment_start: Vec3,
    /// Brush radius in finest-voxel units. A radius of `1.0` covers
    /// roughly one cell along each axis.
    pub radius: f32,
    /// 1-D falloff curve from `d/r ∈ [0, 1]` (center → rim) to
    /// strength `∈ [0, 1]`. Carve / Raise use this to taper the
    /// effective brush radius (currently they're hard Boolean
    /// operations so the curve only gates inside-or-outside);
    /// Inflate / Deflate use it to shape the per-cell thickness
    /// profile of the dilation / erosion band.
    pub falloff_curve: FalloffCurve,
    /// Max-thickness amplitude in finest-voxel units. Inflate adds
    /// up to `ceil(falloff_curve(t) * strength)` cells of material;
    /// Deflate erodes by the same. Carve / Raise ignore this — they
    /// flip every cell inside the brush radius regardless.
    pub strength: f32,
    pub mode: BrushMode,
    /// Material assigned to leaves added by `Raise` / `Inflate` /
    /// Deflate's cavity-wall rim. Carve doesn't consume this —
    /// removed leaves just disappear and the field is ignored.
    pub material: u16,
}

/// 1-D falloff function, evaluated on the normalised brush parameter
/// `t = max(0, 1 − d/r)` — `t = 1` at the brush center, `t = 0` at
/// the rim. Returns strength `∈ [0, 1]`. Mirrors Blender's brush
/// curve presets so the visual feel translates 1:1.
///
/// * `Constant` — `1.0` everywhere inside the radius (the legacy
///   hard-sphere brush). Use for Boolean Carve / Raise where the
///   strength profile doesn't matter.
/// * `Smooth` — cosine bell: `0.5 · (1 − cos(π · t))`. Blender's
///   default Draw / Inflate curve.
/// * `Smoothstep` — Hermite `t² · (3 − 2t)`. Slightly steeper
///   shoulder than Smooth.
/// * `Sharp` — `t²`. Concentrates strength toward the center.
/// * `Linear` — `t`. Even tapering rim → center.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalloffCurve {
    Constant,
    Smooth,
    Smoothstep,
    Sharp,
    Linear,
}

impl FalloffCurve {
    /// Evaluate the curve at `t ∈ [0, 1]`. Caller is responsible for
    /// clamping `t` if needed; this function does not branch on it.
    #[inline]
    pub fn evaluate(self, t: f32) -> f32 {
        match self {
            FalloffCurve::Constant => 1.0,
            FalloffCurve::Smooth => 0.5 - 0.5 * (std::f32::consts::PI * t).cos(),
            FalloffCurve::Smoothstep => t * t * (3.0 - 2.0 * t),
            FalloffCurve::Sharp => t * t,
            FalloffCurve::Linear => t,
        }
    }
}

impl Default for FalloffCurve {
    fn default() -> Self { FalloffCurve::Smooth }
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
    /// encoding referencing that slot.
    ///
    /// **`normal`** carries the analytical brush-SDF gradient at this
    /// cell, computed by [`compute_brush_edits`] from the brush
    /// primitive directly (e.g. `normalize(cell − center)` for a
    /// sphere). The kernel's Add rules guarantee every cell tagged
    /// with this op sits on the brush boundary by construction
    /// (cavity walls for Carve, dome cells for Raise), so the brush
    /// gradient *is* the correct outward surface normal — real-valued
    /// and continuous across the brush footprint, with none of the
    /// lattice quantization a stencil over binary occupancy
    /// introduces. Sign convention matches the
    /// [`LeafAttr::normal_oct`] one already used by bake (outward —
    /// away from solid bulk, into empty space): for Raise dome cells
    /// this points away from the brush center; for Carve cavity walls
    /// it points toward the brush center (into the carved cavity).
    Add { material: u16, normal: Vec3 },
    /// Rewrite an existing surface cell's `LeafAttr.normal_oct`
    /// without touching its occupancy or material. The Smooth brush
    /// emits these to nudge surface normals toward the local
    /// neighbourhood average. `slot` is the cell's pre-resolved
    /// `LeafAttrPool` slot id — compute time has already done the
    /// octree+brick lookup, so [`apply_delta`] can write the pool
    /// directly without re-traversing.
    SetNormal { slot: u32, normal: Vec3 },
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
    pub fn count_set_normal(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::SetNormal { .. })).count()
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

/// Closest point on the brush's capsule axis (the segment from
/// `segment_start` to `center`) to `p`. Used by the SDF, brush-cell
/// AABB, and the per-cell normal — all sphere-vs-segment math
/// collapses to "the cell sees a sphere of radius `op.radius` around
/// this point". When the brush is a single sphere (`segment_start ==
/// center`), returns `op.center`.
#[inline]
fn closest_on_axis(p: Vec3, op: &BrushOp) -> Vec3 {
    let ab = op.center - op.segment_start;
    let ab_len_sq = ab.length_squared();
    if ab_len_sq < 1e-12 {
        return op.center;
    }
    let t = ((p - op.segment_start).dot(ab) / ab_len_sq).clamp(0.0, 1.0);
    op.segment_start + ab * t
}

/// Perpendicular distance from `p` to the brush axis (the segment, not
/// the capsule surface). Subtract `op.radius` to get the SDF; clamp to
/// `op.radius` to get the falloff parameter Inflate / Deflate need.
#[inline]
fn axis_distance(p: Vec3, op: &BrushOp) -> f32 {
    (p - closest_on_axis(p, op)).length()
}

/// Brush SDF at point `p`: negative inside the brush, positive outside,
/// zero on the boundary. For a single-stamp sphere this is the
/// classical `|p − center| − radius`; for a drag capsule it's
/// `dist_to_segment(p) − radius`, which collapses to the sphere form
/// when `segment_start == center`.
#[inline]
fn brush_sdf(p: Vec3, op: &BrushOp) -> f32 {
    axis_distance(p, op) - op.radius
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
    // Capsule AABB = union of the segment endpoints' bounding boxes
    // (each a sphere of radius `op.radius` around the endpoint). For
    // a sphere stamp (`segment_start == center`) this collapses to
    // the prior single-sphere AABB.
    let a_min = op.segment_start - Vec3::splat(op.radius);
    let a_max = op.segment_start + Vec3::splat(op.radius);
    let b_min = op.center - Vec3::splat(op.radius);
    let b_max = op.center + Vec3::splat(op.radius);
    let min_f = a_min.min(b_min);
    let max_f = a_max.max(b_max);
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
    leaf_attr_pool: &[LeafAttr],
    op: BrushOp,
) -> SculptDelta {
    // Inflate / Deflate paths default `is_stroke_edit` to "always
    // false" — meaning the kernel treats every cell's current state as
    // its pre-stroke state. Callers that need the one-layer-per-stroke
    // semantic (the scene manager's drag path) go through
    // `compute_brush_edits_in_stroke` and supply the actual closure.
    compute_brush_edits_in_stroke(octree, brick_pool, leaf_attr_pool, op, |_| false)
}

/// Same as [`compute_brush_edits`] but with an explicit "was this cell
/// already edited in the current stroke?" predicate. The Inflate /
/// Deflate kernels use this to seed brushfire only from PRE-STROKE
/// empty (Deflate) / pre-stroke solid (Inflate) cells, capping the
/// stroke's depth at one `target_thickness` layer while still letting
/// brushfire propagate through stroke-edited cells to reach and clean
/// up earlier-stamp cavity walls. Carve / Raise ignore the closure —
/// their kernels don't use brushfire.
pub fn compute_brush_edits_in_stroke(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    op: BrushOp,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    match op.mode {
        BrushMode::Inflate => {
            return compute_inflate_edits(
                octree,
                brick_pool,
                leaf_attr_pool,
                &op,
                is_stroke_edit,
            );
        }
        BrushMode::Deflate => {
            return compute_deflate_edits(
                octree,
                brick_pool,
                leaf_attr_pool,
                &op,
                is_stroke_edit,
            );
        }
        BrushMode::Smooth => {
            return compute_smooth_edits(octree, brick_pool, leaf_attr_pool, &op);
        }
        BrushMode::ClayStrip => {
            return compute_clay_strip_edits(
                octree,
                brick_pool,
                leaf_attr_pool,
                &op,
                is_stroke_edit,
            );
        }
        BrushMode::Carve | BrushMode::Raise => {}
    }
    let _ = leaf_attr_pool;
    let _ = is_stroke_edit;

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
                // `falloff_curve` and `strength` are only read by the
                // Inflate / Deflate paths which dispatched above; here
                // Carve and Raise are hard Boolean SDF ops with no
                // strength axis.
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
                    (BrushMode::Inflate | BrushMode::Deflate | BrushMode::Smooth | BrushMode::ClayStrip, _) => {
                        unreachable!(
                            "Inflate / Deflate / Smooth / ClayStrip are handled by the top-level dispatch above"
                        )
                    }
                }
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

/// Soft outward dilation. For each Empty cell in the brush AABB,
/// compute its 6-face-hop distance to the nearest Solid / Interior
/// seed (a brushfire propagation over a dense scratch grid), then
/// emit it as an `Add` if that distance is within the brush's
/// falloff-shaped target thickness at the cell. The result is a
/// dilation shell whose thickness profile matches the falloff curve
/// — full strength at center, tapering to zero at the rim.
///
/// **Walk shape.** AABB is padded by `max_k = ceil(strength)` along
/// each axis so the brushfire seed set includes every surface cell
/// that could legitimately reach an addable cell within the brush
/// radius. Total cost: `O(max_k · N³)` per stamp, with `N ≈ 2 · r +
/// 2 · max_k` cells per axis. For a `r = 25`, `strength = 8` brush
/// that's `~6 M` `u8` reads — well under a millisecond.
///
/// **Normals — surface-following inheritance.** Each newly-added
/// cell inherits its normal from the existing-surface (Solid) cell
/// the brushfire reached out from. The init pass unpacks
/// `LeafAttr.normal_oct` for every Solid seed and stores it in a
/// parallel `seed_normal` grid; brushfire propagation carries the
/// value forward — a cell at distance `step` inherits from any 6-
/// neighbour at distance `step − 1` that already has a non-zero
/// seed normal (preferring those over Interior seeds with no
/// stored normal). When no surface seed reaches a cell — purely
/// Interior brushfire path, or a degenerate case — the kernel
/// falls back to [`brush_add_normal`].
///
/// This is the difference between a dome-shaped Inflate (every
/// added cell shading like a sphere stamped on top) and a true
/// soft-Inflate (added cells continue the underlying surface
/// curvature), and it removes the per-stamp normal stripes that
/// brush-radial normals produce along a drag stroke.
fn compute_inflate_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    op: &BrushOp,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    let max_k = op.strength.max(0.0).ceil() as u32;
    if max_k == 0 || op.radius <= 0.0 {
        return SculptDelta::default();
    }
    // u8 distance field; capping at 254 leaves u8::MAX as the
    // unreached sentinel. Practical brushes stay far below this.
    let max_k = max_k.min(254) as u8;

    let (lo, hi) = brush_cell_range_padded(op, extent, max_k as u32);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let t_start = Instant::now();
    let size = hi - lo;
    let stride_y = size.x as usize;
    let stride_z = (size.x * size.y) as usize;
    let total = (size.x * size.y * size.z) as usize;
    let idx = |c: UVec3| -> usize {
        let l = c - lo;
        (l.x as usize) + (l.y as usize) * stride_y + (l.z as usize) * stride_z
    };

    let mut dist: Vec<u8> = vec![u8::MAX; total];
    let mut is_addable: Vec<bool> = vec![false; total];
    // Transit cells — currently Solid/Interior (so brushfire CAN walk
    // through them) but their state was reached this stroke (so they
    // must NOT seed brushfire at dist=0). Without this distinction the
    // kernel would treat stamp N-1's added cells as fresh surface and
    // stamp N would puff *another* target_thickness layer past them —
    // exactly the compounding the user reports as "keeps puffing the
    // longer I hold the mouse". With it, stamp N's brushfire depth is
    // measured from the ORIGINAL pre-stroke surface, not the stamp-
    // accumulated one, so total stroke depth is capped at one
    // `target_thickness`.
    let mut is_transit: Vec<bool> = vec![false; total];
    // Per-cell inherited surface normal; `ZERO` is the sentinel for
    // "no surface seed has reached this cell yet" (and never a valid
    // unit normal). Solid cells in the init pass seed their unpacked
    // `LeafAttr.normal_oct` here; brushfire propagation carries the
    // value to reached cells.
    let mut seed_normal: Vec<Vec3> = vec![Vec3::ZERO; total];

    let mut cache = crate::sparse_octree::CellStateCache::new();
    let mut n_cell_state_calls = 0u32;
    let mut n_inside_sphere = 0u32;
    let mut n_aabb_cells = 0u32;

    // ── Init pass: classify every cell in the AABB. ──
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let c = UVec3::new(x, y, z);
                let state = octree.cell_state_cached(c, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                let i = idx(c);
                let edited = is_stroke_edit(c);
                match state {
                    CellState::Solid(slot) => {
                        if edited {
                            // Added this stroke — brushfire transits
                            // (so deeper cells can be reached at the
                            // correct distance) but the cell itself
                            // doesn't seed at dist=0.
                            is_transit[i] = true;
                        } else {
                            dist[i] = 0;
                            if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                                seed_normal[i] = unpack_oct(attr.normal_oct);
                            }
                        }
                    }
                    CellState::Interior => {
                        if edited {
                            is_transit[i] = true;
                        } else {
                            dist[i] = 0;
                        }
                    }
                    CellState::Empty => is_addable[i] = true,
                    CellState::OutOfBounds => {}
                }
            }
        }
    }

    // ── Brushfire: `max_k` passes, each fills cells whose
    // immediate 6-neighbour has dist == step - 1.
    //
    // **Seed-normal averaging.** Every dist==prev predecessor with
    // a non-zero `seed_normal` contributes one vote; the cell's new
    // normal is the L2-normalised sum. At a corner of an asset
    // (three orthogonal faces meet the brushfire at the same depth)
    // the three face-normals sum to roughly `(1, 1, 1)/√3` — a
    // smooth corner direction that adjacent cells in the bulk all
    // converge on. Picking "first non-zero" instead would let
    // adjacent cells inherit from different faces 90° apart and
    // produce visible scatter on the cavity wall. ──
    for step in 1..=max_k {
        let prev = step - 1;
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = UVec3::new(x, y, z);
                    let i = idx(c);
                    // Propagation reaches three kinds of cells:
                    // - `is_addable` Empty targets (will potentially
                    //   emit Add in the next pass);
                    // - `is_transit` cells (currently Solid/Interior
                    //   added this stroke — they carry distance so
                    //   cells beyond them get the correct value, but
                    //   they don't emit anything themselves).
                    if dist[i] != u8::MAX || (!is_addable[i] && !is_transit[i]) {
                        continue;
                    }
                    let predecessors = [
                        (x > lo.x).then(|| idx(UVec3::new(x - 1, y, z))),
                        (x + 1 < hi.x).then(|| idx(UVec3::new(x + 1, y, z))),
                        (y > lo.y).then(|| idx(UVec3::new(x, y - 1, z))),
                        (y + 1 < hi.y).then(|| idx(UVec3::new(x, y + 1, z))),
                        (z > lo.z).then(|| idx(UVec3::new(x, y, z - 1))),
                        (z + 1 < hi.z).then(|| idx(UVec3::new(x, y, z + 1))),
                    ];
                    let mut found = false;
                    let mut sum = Vec3::ZERO;
                    for p in predecessors.iter().flatten() {
                        if dist[*p] != prev {
                            continue;
                        }
                        found = true;
                        let n = seed_normal[*p];
                        if n != Vec3::ZERO {
                            sum += n;
                        }
                    }
                    if found {
                        dist[i] = step;
                        let len_sq = sum.length_squared();
                        seed_normal[i] = if len_sq > 1e-6 {
                            sum * len_sq.sqrt().recip()
                        } else {
                            Vec3::ZERO
                        };
                    }
                }
            }
        }
    }

    // ── Emit pass: addable cells whose distance is within the
    // falloff-shaped target thickness become Adds. ──
    let mut edits = Vec::new();
    let inv_r = 1.0 / op.radius;
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if !is_addable[i] {
                    continue;
                }
                let d_step = dist[i];
                if d_step == 0 || d_step == u8::MAX {
                    continue;
                }

                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                // Capsule axis distance — collapses to sphere distance
                // when `segment_start == center`. Falloff parameter is
                // the same `1 − d/r` shape across the cylindrical
                // body, so a drag stroke gets a uniform-thickness
                // ribbon rather than a sequence of meeting-circle
                // creases between adjacent spheres.
                let dist_from_brush = axis_distance(cell_center, op);
                if dist_from_brush > op.radius {
                    continue;
                }
                n_inside_sphere += 1;
                let t = (1.0 - dist_from_brush * inv_r).clamp(0.0, 1.0);
                let s = op.falloff_curve.evaluate(t);
                // ceil so any non-zero strength reaches at least one
                // voxel — keeps the brush from "disappearing" at low
                // strength values.
                let target_thickness = (s * op.strength).ceil() as u8;
                if d_step <= target_thickness {
                    let inherited = seed_normal[i];
                    let normal = if inherited != Vec3::ZERO {
                        inherited
                    } else {
                        brush_add_normal(c, op)
                    };
                    edits.push(LeafEdit {
                        coord: c,
                        op: LeafEditOp::Add { material: op.material, normal },
                    });
                }
            }
        }
    }

    SculptDelta {
        edits,
        timing: ComputeBrushEditsTiming {
            n_aabb_cells,
            n_inside_sphere,
            n_cell_state_calls,
            n_neighbor_calls: 0,
            t_total_ns: t_start.elapsed().as_nanos() as u64,
        },
    }
}

/// Flat-top cross-section profile for clay strips. Returns the target
/// thickness in cells at perpendicular distance `d` from the capsule axis.
/// Flat at full `strength` across the inner 75% of the radius, then
/// tapers to zero over the 25% shoulder via the falloff curve.
#[inline]
pub fn clay_strip_profile(d: f32, radius: f32, strength: f32, falloff: FalloffCurve) -> f32 {
    let shoulder = 0.25 * radius;
    let flat = radius - shoulder;
    if d <= flat {
        strength
    } else if d <= radius {
        let t = (radius - d) / shoulder;
        strength * falloff.evaluate(t)
    } else {
        0.0
    }
}

/// Analytical normal for a clay strip cell. Flat-top cells get (0,1,0);
/// shoulder cells get an outward-tilted normal derived from the profile
/// gradient. `horiz_dist` is the horizontal distance from the axis,
/// `diff` is the full vector from the nearest axis point to the cell.
#[inline]
fn clay_strip_normal(horiz_dist: f32, diff: Vec3, op: &BrushOp) -> Vec3 {
    let shoulder = 0.25 * op.radius;
    let flat = op.radius - shoulder;

    if horiz_dist <= flat {
        return Vec3::Y;
    }
    // Shoulder: compute the profile slope via finite difference.
    let eps = 0.5;
    let h0 = clay_strip_profile(horiz_dist - eps, op.radius, op.strength, op.falloff_curve);
    let h1 = clay_strip_profile(horiz_dist + eps, op.radius, op.strength, op.falloff_curve);
    let slope = (h0 - h1) / (2.0 * eps);

    // Normal = (-slope * outward_horizontal, 1, 0), normalized.
    let horiz_vec = Vec3::new(diff.x, 0.0, diff.z);
    let horiz_len = horiz_vec.length();
    if horiz_len < 1e-6 {
        return Vec3::Y;
    }
    let outward = horiz_vec / horiz_len;
    Vec3::new(-slope * outward.x, 1.0, -slope * outward.z).normalize()
}

/// Clay strip deposit. Same brushfire infrastructure as
/// [`compute_inflate_edits`] but with a flat-top thickness profile:
/// full `strength` cells across the inner 75% of the brush radius,
/// tapering to zero over a 25% shoulder at the lateral edges.
fn compute_clay_strip_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    op: &BrushOp,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    let max_k = op.strength.max(0.0).ceil() as u32;
    if max_k == 0 || op.radius <= 0.0 {
        return SculptDelta::default();
    }
    let max_k = max_k.min(254) as u8;

    let (lo, hi) = brush_cell_range_padded(op, extent, max_k as u32);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let t_start = Instant::now();
    let size = hi - lo;
    let stride_y = size.x as usize;
    let stride_z = (size.x * size.y) as usize;
    let total = (size.x * size.y * size.z) as usize;
    let idx = |c: UVec3| -> usize {
        let l = c - lo;
        (l.x as usize) + (l.y as usize) * stride_y + (l.z as usize) * stride_z
    };

    let mut dist: Vec<u8> = vec![u8::MAX; total];
    let mut is_addable: Vec<bool> = vec![false; total];
    let mut is_transit: Vec<bool> = vec![false; total];
    let mut seed_normal: Vec<Vec3> = vec![Vec3::ZERO; total];

    let mut cache = crate::sparse_octree::CellStateCache::new();
    let mut n_cell_state_calls = 0u32;
    let mut n_inside_sphere = 0u32;
    let mut n_aabb_cells = 0u32;

    // ── Init pass ──
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let c = UVec3::new(x, y, z);
                let state = octree.cell_state_cached(c, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                let i = idx(c);
                let edited = is_stroke_edit(c);
                match state {
                    CellState::Solid(slot) => {
                        if edited {
                            is_transit[i] = true;
                        } else {
                            dist[i] = 0;
                            if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                                seed_normal[i] = unpack_oct(attr.normal_oct);
                            }
                        }
                    }
                    CellState::Interior => {
                        if edited {
                            is_transit[i] = true;
                        } else {
                            dist[i] = 0;
                        }
                    }
                    CellState::Empty => is_addable[i] = true,
                    CellState::OutOfBounds => {}
                }
            }
        }
    }

    // ── Brushfire ──
    for step in 1..=max_k {
        let prev = step - 1;
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = UVec3::new(x, y, z);
                    let i = idx(c);
                    if dist[i] != u8::MAX || (!is_addable[i] && !is_transit[i]) {
                        continue;
                    }
                    let predecessors = [
                        (x > lo.x).then(|| idx(UVec3::new(x - 1, y, z))),
                        (x + 1 < hi.x).then(|| idx(UVec3::new(x + 1, y, z))),
                        (y > lo.y).then(|| idx(UVec3::new(x, y - 1, z))),
                        (y + 1 < hi.y).then(|| idx(UVec3::new(x, y + 1, z))),
                        (z > lo.z).then(|| idx(UVec3::new(x, y, z - 1))),
                        (z + 1 < hi.z).then(|| idx(UVec3::new(x, y, z + 1))),
                    ];
                    let mut found = false;
                    let mut sum = Vec3::ZERO;
                    for p in predecessors.iter().flatten() {
                        if dist[*p] != prev {
                            continue;
                        }
                        found = true;
                        let n = seed_normal[*p];
                        if n != Vec3::ZERO {
                            sum += n;
                        }
                    }
                    if found {
                        dist[i] = step;
                        let len_sq = sum.length_squared();
                        seed_normal[i] = if len_sq > 1e-6 {
                            sum * len_sq.sqrt().recip()
                        } else {
                            Vec3::ZERO
                        };
                    }
                }
            }
        }
    }

    // ── Emit pass: flat-top profile instead of dome ──
    let mut edits = Vec::new();
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if !is_addable[i] {
                    continue;
                }
                let d_step = dist[i];
                if d_step == 0 || d_step == u8::MAX {
                    continue;
                }

                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                // Horizontal distance from the capsule axis — use
                // only XZ so the strip profile evaluates the cross-
                // section width correctly. 3D distance would make
                // the flat top narrower at higher cells because the
                // vertical offset inflates the distance.
                let closest = closest_on_axis(cell_center, op);
                let diff = cell_center - closest;
                let horiz_dist = Vec3::new(diff.x, 0.0, diff.z).length();
                if horiz_dist > op.radius {
                    continue;
                }
                n_inside_sphere += 1;
                let target_thickness = clay_strip_profile(
                    horiz_dist, op.radius, op.strength, op.falloff_curve,
                ).ceil() as u8;
                if d_step <= target_thickness {
                    let normal = clay_strip_normal(horiz_dist, diff, op);
                    edits.push(LeafEdit {
                        coord: c,
                        op: LeafEditOp::Add { material: op.material, normal },
                    });
                }
            }
        }
    }

    SculptDelta {
        edits,
        timing: ComputeBrushEditsTiming {
            n_aabb_cells,
            n_inside_sphere,
            n_cell_state_calls,
            n_neighbor_calls: 0,
            t_total_ns: t_start.elapsed().as_nanos() as u64,
        },
    }
}

/// Soft inward erosion. Mirror of [`compute_inflate_edits`]: brushfire
/// seeds from Empty cells and propagates inward into Solid / Interior
/// bulk; cells whose distance-to-empty falls within the falloff-shaped
/// target thickness are removed. After the erosion pocket is
/// determined, every Interior cell that borders the pocket becomes a
/// cavity-wall surface leaf — same rule Carve uses on its outside-rim
/// Interior cells, just driven by the per-cell erosion mask instead
/// of a hard sphere boundary.
///
/// **Ops emitted.** Solid cells inside the pocket → `Remove`;
/// Interior cells inside the pocket → `Empty`; Interior cells
/// outside the pocket with a pocket-adjacent 6-neighbour → `Add
/// { material, normal }`. Existing Solid surface cells that aren't
/// eroded keep their bake-time normal — they were already surface
/// and the bake gradient is still a better direction than the
/// brush-sphere gradient would be.
///
/// **Cavity-wall normals — surface-following inheritance.** Same
/// principle as [`compute_inflate_edits`], inverted seed direction.
/// The brushfire propagates from Empty (and OOB-treated-as-Empty)
/// seeds inward into the bulk; when it reaches a Solid cell at
/// distance N, that cell stamps the `seed_normal` grid with its own
/// bake-time `LeafAttr.normal_oct` (which points outward from the
/// bulk). As the walk continues to Interior cells at distance N+1,
/// N+2, ..., they inherit that direction from their predecessor.
/// The cavity wall — the first Interior cell beyond the erosion
/// target_thickness — ends up holding the original surface's
/// outward direction, which is also the new "into the pit"
/// direction since the pit is now the empty volume on the bulk's
/// outer side. The brushfire runs an extra step (`max_k + 1`) so
/// cavity walls one beyond the maximum erosion depth still get
/// reached.
///
/// Without this, cavity walls fall back to [`brush_add_normal`]
/// (the brush-radial direction), which varies per stamp center —
/// adjacent cavity walls from different stamps along a drag stroke
/// pick up subtly different normals and produce visible shading
/// stripes. Inheritance ties the cavity wall's direction to the
/// existing surface, which doesn't move per-stamp.
///
/// **Walk shape & cost.** Same as Inflate: AABB padded by
/// `ceil(strength)`, four sequential walks over the dense scratch
/// grid (init / extent-edge prime / brushfire / erosion-mask +
/// emit). Memory: ~28 bytes per cell (dist + was_solid +
/// was_interior + in_erosion + seed_normal + original_normal).
fn compute_deflate_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    op: &BrushOp,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    let max_k = op.strength.max(0.0).ceil() as u32;
    if max_k == 0 || op.radius <= 0.0 {
        return SculptDelta::default();
    }
    let max_k = max_k.min(254) as u8;

    let (lo, hi) = brush_cell_range_padded(op, extent, max_k as u32);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let t_start = Instant::now();
    let size = hi - lo;
    let stride_y = size.x as usize;
    let stride_z = (size.x * size.y) as usize;
    let total = (size.x * size.y * size.z) as usize;
    let idx = |c: UVec3| -> usize {
        let l = c - lo;
        (l.x as usize) + (l.y as usize) * stride_y + (l.z as usize) * stride_z
    };

    let mut dist: Vec<u8> = vec![u8::MAX; total];
    let mut was_solid: Vec<bool> = vec![false; total];
    let mut was_interior: Vec<bool> = vec![false; total];
    // Transit cells — currently Empty (so the kernel doesn't emit
    // anything for them) but emptied by a previous stamp in this
    // stroke (so they must NOT seed brushfire at dist=0). Without this
    // distinction stamp N's brushfire starts from stamp N-1's already-
    // eroded region and reaches one layer deeper into the bulk every
    // stamp — the "Deflate keeps digging" report. Treating them as
    // transit makes the brushfire propagate THROUGH them while still
    // measuring distance from the original pre-stroke surface, so the
    // stroke's total depth is capped at one `target_thickness`. Bonus:
    // earlier-stamp cavity walls (currently Solid, in the `was_solid`
    // bucket below regardless) become reachable from the original
    // outside through transit cells AND through the asset's
    // pre-stroke air, so stamp N can clean up stamp N-1's walls and
    // the channel comes out smooth.
    let mut is_transit: Vec<bool> = vec![false; total];
    // Per-cell propagated seed normal — see docstring. `ZERO` means
    // "no surface seed has been carried here yet"; ZERO never names
    // a valid unit normal so it's safe as a sentinel.
    let mut seed_normal: Vec<Vec3> = vec![Vec3::ZERO; total];
    // Pre-unpacked bake-time normal for every Solid cell in the
    // AABB. Pulled out so the brushfire loop can stamp seed_normal
    // when it reaches the cell without re-touching the
    // `leaf_attr_pool`.
    let mut original_normal: Vec<Vec3> = vec![Vec3::ZERO; total];

    let mut cache = crate::sparse_octree::CellStateCache::new();
    let mut n_cell_state_calls = 0u32;
    let mut n_aabb_cells = 0u32;
    let mut n_inside_sphere = 0u32;

    // ── Init: Empty cells seed the brushfire; Solid / Interior
    // record their original kind for the emit + cavity-wall passes.
    //
    // **OOB-as-Empty.** Cells outside the octree extent are air
    // conceptually, so they seed the brushfire alongside actual
    // Empty cells. The AABB is clipped to the extent so OOB cells
    // never appear in the inner loop directly — instead, any
    // Solid / Interior cell at the asset extent boundary is
    // pre-marked with `dist = 1` (i.e. "one step from a virtual
    // empty seed just past the extent"). Lets Deflate erode the
    // asset's outermost surface even when the brush sits right at
    // the extent boundary; without this an all-Interior asset
    // would have no seeds inside the AABB and Deflate would no-op. ──
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let c = UVec3::new(x, y, z);
                let state = octree.cell_state_cached(c, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                let i = idx(c);
                let edited = is_stroke_edit(c);
                match state {
                    CellState::Empty => {
                        if edited {
                            // Emptied this stroke — brushfire transits
                            // through but doesn't restart distance at 0.
                            is_transit[i] = true;
                        } else {
                            dist[i] = 0;
                        }
                    }
                    CellState::OutOfBounds => dist[i] = 0,
                    CellState::Solid(slot) => {
                        was_solid[i] = true;
                        if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                            original_normal[i] = unpack_oct(attr.normal_oct);
                        }
                    }
                    CellState::Interior => was_interior[i] = true,
                }
            }
        }
    }
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if !was_solid[i] && !was_interior[i] {
                    continue;
                }
                let touches_extent_edge = x == 0
                    || y == 0
                    || z == 0
                    || x + 1 == extent
                    || y + 1 == extent
                    || z + 1 == extent;
                if touches_extent_edge {
                    dist[i] = 1;
                    // Same self-stamp rule as the brushfire below:
                    // a pre-marked Solid cell seeds the inheritance
                    // chain with its own bake-time normal.
                    if was_solid[i] && original_normal[i] != Vec3::ZERO {
                        seed_normal[i] = original_normal[i];
                    }
                }
            }
        }
    }

    // ── Brushfire into Solid / Interior cells. Runs to `max_k + 1`
    // (one beyond the maximum possible target_thickness) so the
    // cavity wall — the first Interior cell beyond the erosion
    // pocket — gets a propagated `seed_normal`. Each reached cell
    // either stamps its own bake-time normal (if it's a Solid
    // surface) or averages from its predecessors (Interior bulk).
    //
    // **Seed-normal averaging.** Sums non-zero `seed_normal` values
    // from every `dist == prev` predecessor and normalises. At a
    // box corner where +X, +Y, +Z surface faces all reach the
    // brushfire at the same depth, the three face-normals average
    // to `(1, 1, 1)/√3` — a smooth corner direction that adjacent
    // cells deeper in the bulk all converge on. "First-found" would
    // let adjacent cavity walls inherit from different faces 90°
    // apart and produce visible scatter on the eroded surface. ──
    let brushfire_steps: u8 = max_k.saturating_add(1).min(254);
    for step in 1..=brushfire_steps {
        let prev = step - 1;
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = UVec3::new(x, y, z);
                    let i = idx(c);
                    // Propagation extends to `was_solid` / `was_interior`
                    // targets (these emit erosion ops below) AND to
                    // `is_transit` cells (currently Empty, emptied
                    // this stroke — they carry distance so cells
                    // beyond them get the correct value, but they
                    // don't emit anything themselves).
                    if dist[i] != u8::MAX
                        || (!was_solid[i] && !was_interior[i] && !is_transit[i])
                    {
                        continue;
                    }
                    let predecessors = [
                        (x > lo.x).then(|| idx(UVec3::new(x - 1, y, z))),
                        (x + 1 < hi.x).then(|| idx(UVec3::new(x + 1, y, z))),
                        (y > lo.y).then(|| idx(UVec3::new(x, y - 1, z))),
                        (y + 1 < hi.y).then(|| idx(UVec3::new(x, y + 1, z))),
                        (z > lo.z).then(|| idx(UVec3::new(x, y, z - 1))),
                        (z + 1 < hi.z).then(|| idx(UVec3::new(x, y, z + 1))),
                    ];
                    let mut found = false;
                    let mut sum = Vec3::ZERO;
                    for p in predecessors.iter().flatten() {
                        if dist[*p] != prev {
                            continue;
                        }
                        found = true;
                        let n = seed_normal[*p];
                        if n != Vec3::ZERO {
                            sum += n;
                        }
                    }
                    if found {
                        dist[i] = step;
                        let len_sq = sum.length_squared();
                        let inherited = if len_sq > 1e-6 {
                            sum * len_sq.sqrt().recip()
                        } else {
                            Vec3::ZERO
                        };
                        // Solid cells override the inherited chain
                        // with their own bake-time normal — they
                        // ARE a real surface so they're the more
                        // authoritative source for everything
                        // deeper than them.
                        seed_normal[i] = if was_solid[i] && original_normal[i] != Vec3::ZERO {
                            original_normal[i]
                        } else {
                            inherited
                        };
                    }
                }
            }
        }
    }

    // ── Determine erosion mask (cells the brush will actually
    // remove). Decoupling this from emit so the cavity-wall pass
    // below can read it. ──
    let mut in_erosion: Vec<bool> = vec![false; total];
    let inv_r = 1.0 / op.radius;
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if !was_solid[i] && !was_interior[i] {
                    continue;
                }
                let d_step = dist[i];
                if d_step == 0 || d_step == u8::MAX {
                    continue;
                }
                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                // Capsule axis distance — see Inflate's emit pass for
                // the longer rationale. Collapses to sphere distance
                // for single stamps; gives a uniform-thickness ribbon
                // along a drag.
                let dist_from_brush = axis_distance(cell_center, op);
                if dist_from_brush > op.radius {
                    continue;
                }
                n_inside_sphere += 1;
                let t = (1.0 - dist_from_brush * inv_r).clamp(0.0, 1.0);
                let s = op.falloff_curve.evaluate(t);
                let target_thickness = (s * op.strength).ceil() as u8;
                if d_step <= target_thickness {
                    in_erosion[i] = true;
                }
            }
        }
    }

    // ── Emit pass: Removes / Empties for in-erosion cells, plus
    // cavity-wall Adds for Interior cells that border the pocket. ──
    let mut edits = Vec::new();
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if in_erosion[i] {
                    if was_solid[i] {
                        edits.push(LeafEdit { coord: c, op: LeafEditOp::Remove });
                    } else if was_interior[i] {
                        edits.push(LeafEdit { coord: c, op: LeafEditOp::Empty });
                    }
                    continue;
                }
                // Cavity-wall rule: an Interior cell that's NOT in
                // the erosion pocket but has a 6-neighbour that IS
                // becomes a freshly-exposed surface leaf. The same
                // outside-rim pattern Carve uses, driven here by
                // the soft erosion mask instead of a hard sphere.
                if !was_interior[i] {
                    continue;
                }
                let adj_eroded =
                    (x > lo.x && in_erosion[idx(UVec3::new(x - 1, y, z))])
                        || (x + 1 < hi.x && in_erosion[idx(UVec3::new(x + 1, y, z))])
                        || (y > lo.y && in_erosion[idx(UVec3::new(x, y - 1, z))])
                        || (y + 1 < hi.y && in_erosion[idx(UVec3::new(x, y + 1, z))])
                        || (z > lo.z && in_erosion[idx(UVec3::new(x, y, z - 1))])
                        || (z + 1 < hi.z && in_erosion[idx(UVec3::new(x, y, z + 1))]);
                if adj_eroded {
                    // Cavity-wall normal: prefer the brushfire-
                    // propagated `seed_normal`, which carries the
                    // original surface's outward direction inward
                    // through the brushfire propagation. The
                    // surface doesn't move per-stamp, so adjacent
                    // walls along a drag inherit the SAME direction
                    // and shade consistently — fixes the per-stamp
                    // shading stripe that `brush_add_normal` alone
                    // produces (its direction varies with the brush
                    // center, so each stamp's walls pick up subtly
                    // different normals).
                    //
                    // Sometimes the brushfire chain into this cell
                    // is truncated by the AABB edge or by a normal-
                    // cancellation degenerate (two predecessors with
                    // opposite surface normals), and `seed_normal[i]`
                    // ends up ZERO. Before falling back to
                    // `brush_add_normal` — which produces the
                    // per-stamp variation the inheritance is meant
                    // to avoid — average any non-zero seed_normals
                    // among the 6-neighbours; the adj_eroded check
                    // guarantees at least one of them is an in_erosion
                    // cell that brushfire reached, so a useful
                    // direction is almost always available. Only
                    // genuinely isolated cells (no neighbour has a
                    // non-zero normal) fall through to the radial
                    // gradient.
                    let inherited = seed_normal[i];
                    let normal = if inherited != Vec3::ZERO {
                        inherited
                    } else {
                        let mut sum = Vec3::ZERO;
                        if x > lo.x {
                            let n = seed_normal[idx(UVec3::new(x - 1, y, z))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        if x + 1 < hi.x {
                            let n = seed_normal[idx(UVec3::new(x + 1, y, z))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        if y > lo.y {
                            let n = seed_normal[idx(UVec3::new(x, y - 1, z))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        if y + 1 < hi.y {
                            let n = seed_normal[idx(UVec3::new(x, y + 1, z))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        if z > lo.z {
                            let n = seed_normal[idx(UVec3::new(x, y, z - 1))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        if z + 1 < hi.z {
                            let n = seed_normal[idx(UVec3::new(x, y, z + 1))];
                            if n != Vec3::ZERO { sum += n; }
                        }
                        let len_sq = sum.length_squared();
                        if len_sq > 1e-6 {
                            sum * len_sq.sqrt().recip()
                        } else {
                            brush_add_normal(c, op)
                        }
                    };
                    edits.push(LeafEdit {
                        coord: c,
                        op: LeafEditOp::Add { material: op.material, normal },
                    });
                }
            }
        }
    }

    SculptDelta {
        edits,
        timing: ComputeBrushEditsTiming {
            n_aabb_cells,
            n_inside_sphere,
            n_cell_state_calls,
            n_neighbor_calls: 0,
            t_total_ns: t_start.elapsed().as_nanos() as u64,
        },
    }
}

/// Geometry-and-normal smoothing. Three rules over the brush capsule
/// (all from pre-stamp 6-neighbour occupancy in a dense scratch grid):
///
/// * **Cavity-fill.** `Empty` cell with `≥ 4` of its 6 face-neighbours
///   occupied (Solid or Interior) flips to `Solid`. Used to fill
///   isolated 1-voxel pits and thin concavities. The new cell's
///   `LeafAttr.normal_oct` points toward where the Empty neighbours
///   were (averaged direction-to-empty-neighbour), so the new surface
///   aligns with the local outward direction rather than the brush
///   axis.
/// * **Bump-shave.** `Solid` cell with `≤ 2` occupied 6-neighbours is
///   `Remove`d. Erodes isolated 1-voxel bumps and thin Solid spikes.
/// * **Normal blend.** Surviving `Solid` cells (those NOT being
///   `Remove`d) blend their existing normal toward the average of
///   their Solid 6-neighbours' normals by `falloff_curve(t) *
///   (strength / 32)`. Same V1 rule, just gated by morph survival.
///
/// **Cavity walls.** After the in-capsule pass, a +1-rim pass emits
/// `Add` edits for `Interior` cells with a `Solid` 6-neighbour that
/// just got `Remove`d. Direction of the new surface normal: average
/// of unit-vectors toward the just-removed neighbours. Same pattern
/// Carve / Deflate use to keep the bulk watertight when the surface
/// shell breaks.
///
/// **Why a dense scratch grid.** The morph rule is "majority of
/// pre-stamp neighbours", so every cell must read consistent
/// pre-stamp state regardless of walk order. A scratch grid pinned to
/// the +2-padded AABB lets the morph + blend + cavity-wall passes all
/// read the same snapshot without re-traversing the octree, and folds
/// the per-Solid `LeafAttr` unpack into a single init pass (the V1
/// kernel re-unpacked the same `LeafAttr` once per probing
/// neighbour).
///
/// **Threshold choice.** `≥ 4 of 6` for fill and `≤ 2 of 6` for shave
/// is "strict majority" of the 6 face-neighbours, which is the
/// roughness-eroding rule. It's conservative enough that flat-surface
/// patches and short grooves survive (their interior cells have
/// 3–5 occupied neighbours, on neither side of either threshold), but
/// aggressive enough that isolated bumps / pits flip immediately. The
/// `falloff_curve` and `strength` slider modulate ONLY the normal
/// blend — the morph rule is binary by design, since "partially
/// flipping" a cell isn't a representable state.
///
/// **Strength mapping.** Editor strength slider is 1..32, mapped to
/// `0..1` blend rate per stamp for the normal pass. Strength does not
/// affect morph aggression — the threshold is fixed at 4 / 2.
fn compute_smooth_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    op: &BrushOp,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    if op.radius <= 0.0 {
        return SculptDelta::default();
    }
    // +2 pad: +1 covers the cavity-wall rim cells outside the capsule,
    // +1 more so those rim cells' 6-neighbour lookups (used to spot
    // just-removed Solids) stay inside the scratch grid.
    let (lo, hi) = brush_cell_range_padded(op, extent, 2);
    if lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z {
        return SculptDelta::default();
    }

    let t_start = Instant::now();
    let size = hi - lo;
    let stride_y = size.x as usize;
    let stride_z = (size.x * size.y) as usize;
    let total = (size.x * size.y * size.z) as usize;
    let idx = |c: UVec3| -> usize {
        let l = c - lo;
        (l.x as usize) + (l.y as usize) * stride_y + (l.z as usize) * stride_z
    };

    // Pre-stamp state grid: 0 = Empty, 1 = Solid, 2 = Interior,
    // 3 = OutOfBounds. Default to OOB so cells the init loop skips
    // (clipped against extent) are treated as "no neighbour" by both
    // the morph rule and the cavity-wall pass.
    const ST_EMPTY: u8 = 0;
    const ST_SOLID: u8 = 1;
    const ST_INTERIOR: u8 = 2;
    const ST_OOB: u8 = 3;
    let mut state_grid: Vec<u8> = vec![ST_OOB; total];
    // Slot id of each Solid cell, for the normal-blend SetNormal emit.
    // u32::MAX = no slot stored at this cell.
    let mut slot_grid: Vec<u32> = vec![u32::MAX; total];
    // Pre-stamp surface normal for each Solid cell, unpacked once.
    let mut normal_grid: Vec<Vec3> = vec![Vec3::ZERO; total];

    // ── Init pass: classify every cell in the +2-padded AABB. ──
    let mut cache = crate::sparse_octree::CellStateCache::new();
    let mut n_aabb_cells = 0u32;
    let mut n_cell_state_calls = 0u32;
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let c = UVec3::new(x, y, z);
                let state = octree.cell_state_cached(c, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                let i = idx(c);
                match state {
                    CellState::Empty => state_grid[i] = ST_EMPTY,
                    CellState::Solid(slot) => {
                        state_grid[i] = ST_SOLID;
                        slot_grid[i] = slot;
                        if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                            normal_grid[i] = unpack_oct(attr.normal_oct);
                        }
                    }
                    CellState::Interior => state_grid[i] = ST_INTERIOR,
                    CellState::OutOfBounds => state_grid[i] = ST_OOB,
                }
            }
        }
    }

    // 6 face-neighbour offsets, indexed in pairs (+x, -x, +y, -y,
    // +z, -z) for both occupancy counting and Add-normal averaging.
    const FACE_DIRS: [IVec3; 6] = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];

    // Lookup helper: state at coord, or OOB if outside the scratch grid.
    let in_grid = |c: IVec3| -> bool {
        c.x >= lo.x as i32 && c.y >= lo.y as i32 && c.z >= lo.z as i32
            && c.x < hi.x as i32 && c.y < hi.y as i32 && c.z < hi.z as i32
    };
    let state_at = |c: IVec3, grid: &[u8]| -> u8 {
        if !in_grid(c) { ST_OOB } else { grid[idx(UVec3::new(c.x as u32, c.y as u32, c.z as u32))] }
    };

    let mut edits: Vec<LeafEdit> = Vec::new();
    // Tracks which Solid cells the morph pass marks for removal — the
    // cavity-wall pass reads this to spot Interior cells whose Solid
    // neighbour just vanished.
    let mut will_remove: Vec<bool> = vec![false; total];

    let inv_r = 1.0 / op.radius;
    let strength_scaled = (op.strength / 32.0).clamp(0.0, 1.0);
    let mut n_inside_sphere = 0u32;
    let mut n_neighbor_calls = 0u32;

    // ── Morph + normal-blend pass: cells inside the brush capsule. ──
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let cell_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let dist_from_brush = axis_distance(cell_center, op);
                if dist_from_brush > op.radius {
                    continue;
                }
                n_inside_sphere += 1;

                let i = idx(c);
                let s = state_grid[i];
                if s != ST_EMPTY && s != ST_SOLID {
                    // Interior / OOB cells don't drive morph here —
                    // Interior cavity walls are emitted by the rim
                    // pass below; OOB doesn't have an octree slot to
                    // operate on.
                    continue;
                }

                // Count occupied (Solid|Interior) 6-neighbours. OOB
                // neighbours count as neither occupied nor empty —
                // they just lower the maximum the rule sees. At the
                // grid edge this means cells lose at most 1–2
                // potential occupied votes, which is acceptable noise.
                let mut occupied = 0u32;
                let mut empty_neighbour_dirs: [Option<IVec3>; 6] = [None; 6];
                let ci = IVec3::new(x as i32, y as i32, z as i32);
                for (dir_idx, dir) in FACE_DIRS.iter().enumerate() {
                    n_neighbor_calls += 1;
                    let ns = state_at(ci + *dir, &state_grid);
                    if ns == ST_SOLID || ns == ST_INTERIOR {
                        occupied += 1;
                    } else if ns == ST_EMPTY {
                        empty_neighbour_dirs[dir_idx] = Some(*dir);
                    }
                }

                if s == ST_EMPTY {
                    // Cavity-fill: ≥ 4 occupied → flip to Solid.
                    if occupied >= 4 {
                        // Outward normal = averaged direction toward
                        // Empty neighbours (which is where the new
                        // surface faces).
                        let mut sum = Vec3::ZERO;
                        for d in empty_neighbour_dirs.iter().flatten() {
                            sum += Vec3::new(d.x as f32, d.y as f32, d.z as f32);
                        }
                        let len_sq = sum.length_squared();
                        let normal = if len_sq < 1e-6 {
                            // All 6 occupied (or only OOB on the
                            // Empty side) — defensive fallback to the
                            // analytical brush gradient.
                            brush_add_normal(c, op)
                        } else {
                            sum * len_sq.sqrt().recip()
                        };
                        edits.push(LeafEdit {
                            coord: c,
                            op: LeafEditOp::Add { material: op.material, normal },
                        });
                    }
                    continue;
                }

                // Solid cell. Decide bump-shave vs normal-blend.
                if occupied <= 2 {
                    edits.push(LeafEdit { coord: c, op: LeafEditOp::Remove });
                    will_remove[i] = true;
                    continue;
                }

                // Normal blend on cells that survive the morph rule.
                let mut sum = Vec3::ZERO;
                for dir in FACE_DIRS.iter() {
                    let n_coord = ci + *dir;
                    if !in_grid(n_coord) { continue; }
                    let n_i = idx(UVec3::new(n_coord.x as u32, n_coord.y as u32, n_coord.z as u32));
                    if state_grid[n_i] != ST_SOLID { continue; }
                    let n_normal = normal_grid[n_i];
                    if n_normal != Vec3::ZERO {
                        sum += n_normal;
                    }
                }
                let target_len_sq = sum.length_squared();
                if target_len_sq < 1e-6 {
                    // No Solid neighbour with a real normal — leave
                    // the cell's normal alone.
                    continue;
                }
                let target = sum * target_len_sq.sqrt().recip();

                let t_curve = (1.0 - dist_from_brush * inv_r).clamp(0.0, 1.0);
                let s_curve = op.falloff_curve.evaluate(t_curve);
                let blend = (s_curve * strength_scaled).clamp(0.0, 1.0);
                if blend <= 0.0 {
                    continue;
                }
                let current = normal_grid[i];
                let blended = current.lerp(target, blend);
                let blended_len_sq = blended.length_squared();
                if blended_len_sq < 1e-6 {
                    continue;
                }
                let new_normal = blended * blended_len_sq.sqrt().recip();
                edits.push(LeafEdit {
                    coord: c,
                    op: LeafEditOp::SetNormal { slot: slot_grid[i], normal: new_normal },
                });
            }
        }
    }

    // ── Cavity-wall pass: Interior cells with a just-Removed Solid
    // 6-neighbour become the new surface shell. Walks the +1 padded
    // range (so a Solid removed AT the capsule boundary can still
    // expose its outside-the-capsule Interior neighbour). Skipping
    // the outermost +1 ring of the +2 scratch grid keeps neighbour
    // lookups in-bounds without an extra branch.
    let walk_lo = UVec3::new(
        (lo.x + 1).min(hi.x),
        (lo.y + 1).min(hi.y),
        (lo.z + 1).min(hi.z),
    );
    let walk_hi = UVec3::new(
        hi.x.saturating_sub(1).max(walk_lo.x),
        hi.y.saturating_sub(1).max(walk_lo.y),
        hi.z.saturating_sub(1).max(walk_lo.z),
    );
    for z in walk_lo.z..walk_hi.z {
        for y in walk_lo.y..walk_hi.y {
            for x in walk_lo.x..walk_hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if state_grid[i] != ST_INTERIOR {
                    continue;
                }
                let ci = IVec3::new(x as i32, y as i32, z as i32);
                // Average direction toward the just-removed Solid
                // neighbours — the cavity-wall surface faces the
                // empty space those Solids used to occupy.
                let mut sum = Vec3::ZERO;
                let mut has_removed = false;
                for dir in FACE_DIRS.iter() {
                    let n_coord = ci + *dir;
                    let n_i = idx(UVec3::new(n_coord.x as u32, n_coord.y as u32, n_coord.z as u32));
                    if will_remove[n_i] {
                        has_removed = true;
                        sum += Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
                    }
                }
                if !has_removed {
                    continue;
                }
                let len_sq = sum.length_squared();
                let normal = if len_sq < 1e-6 {
                    // Opposite-sign removals cancelled — fall back to
                    // the brush gradient (carve-style cavity wall
                    // direction).
                    -brush_add_normal(c, op)
                } else {
                    sum * len_sq.sqrt().recip()
                };
                edits.push(LeafEdit {
                    coord: c,
                    op: LeafEditOp::Add { material: op.material, normal },
                });
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
        let normal = brush_add_normal(coord, op);
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material, normal } });
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
        let normal = brush_add_normal(coord, op);
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material, normal } });
    }
}

/// Outward-pointing surface normal for a freshly-Added cell, derived
/// analytically from the brush primitive's SDF gradient at the cell
/// center. Cells emitted by [`emit_carve_outside_rim`] /
/// [`emit_raise_outside_rim`] sit on the brush boundary by
/// construction (the outside-rim rule requires an inside-brush
/// neighbour about to flip occupancy), so the brush's own gradient
/// *is* the surface normal at that cell.
///
/// **Sphere brush** (the only [`BrushOp`] shape today): the SDF is
/// `|c − center| − radius`, so the outward gradient is
/// `normalize(c − center)`. For
///
/// * `Raise`: the new dome cell is outside the brush, the bulk is
///   inside → outward (into empty) = `+(c − center)`.
/// * `Carve`: the new cavity wall is outside the brush, the cavity
///   (empty) is inside → outward (into the carved cavity) =
///   `−(c − center) = (center − c)`.
///
/// `c` is taken at the cell's center (integer coord + 0.5 on each
/// axis) — the same convention Surface Nets uses for its corner
/// classification. Degenerate (cell center coincident with brush
/// center) falls back to `+Y`, same stable default the prior
/// occupancy-gradient helpers used.
///
/// When the brush primitive grows beyond a sphere (box, capsule,
/// custom user SDF), this function becomes the single place that
/// dispatches on the primitive shape — same pattern the bake-time
/// procedural pipeline already uses.
#[inline]
fn brush_add_normal(coord: UVec3, op: &BrushOp) -> Vec3 {
    let cell_center = Vec3::new(
        coord.x as f32 + 0.5,
        coord.y as f32 + 0.5,
        coord.z as f32 + 0.5,
    );
    // For a drag capsule the gradient points radially from the
    // *closest point on the segment*, not the segment's endpoint —
    // otherwise the normals along the cylindrical body would have a
    // longitudinal component that creases the lighting. The
    // closest-point reduction collapses to `op.center` for a sphere
    // stamp.
    let from_axis = cell_center - closest_on_axis(cell_center, op);
    let len_sq = from_axis.length_squared();
    if len_sq < 1e-6 {
        return Vec3::Y;
    }
    let outward_from_brush = from_axis * len_sq.sqrt().recip();
    match op.mode {
        // Dome / Inflate surface faces away from the bulk (out of
        // the brush). Inflate cells sit a few voxels outside the
        // pre-existing surface and bound a freshly-added clay layer
        // — the analytical brush gradient is still the right
        // direction because the added layer is centered on the
        // brush, not on the underlying surface.
        BrushMode::Raise | BrushMode::Inflate | BrushMode::ClayStrip => outward_from_brush,
        // Cavity wall / Deflate-revealed wall faces into the carved
        // pocket (toward the brush center, where the empty volume
        // now lives).
        BrushMode::Carve | BrushMode::Deflate => -outward_from_brush,
        // Smooth's morph pass computes Add normals directly in
        // compute_smooth_edits (averaging direction-to-neighbour
        // vectors), so this fallback only fires when the in-kernel
        // computation degenerates to zero magnitude — defensive.
        BrushMode::Smooth => outward_from_brush,
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
    {
        let mut cache = BrickPathCache::new();
        for edit in &delta.edits {
            if let LeafEditOp::Add { material, normal } = edit.op {
                n_add += 1;
                let slot = alloc_slot();
                // The normal carried on the Add op is the analytical
                // brush-SDF gradient at this cell, computed by
                // compute_brush_edits from the brush primitive
                // (sphere SDF → `normalize(cell - center)`, signed for
                // Raise vs Carve). No post-stamp gradient pass is
                // needed — the brush boundary IS the surface for
                // every Add the kernel emits, and the brush gradient
                // is real-valued and continuous across the brush
                // footprint, eliminating the lattice quantization a
                // stencil over binary occupancy used to introduce.
                allocated_slots.push((slot, LeafEditAttrs { material, normal }));
                if let Some(prev) =
                    octree.set_cell_solid_cached(edit.coord, slot, brick_pool, &mut cache)
                {
                    // Caller must free the displaced slot too.
                    freed_slots.push(prev);
                }
            }
        }
    }
    let t_loop_add_ns = t_add.elapsed().as_nanos() as u64;

    // Smooth's SetNormal edits land here: those cells stay Solid (no
    // octree mutation, no slot alloc), only their `LeafAttr.normal_oct`
    // changes. The scene-manager consumer drains `renormalized_slots`
    // into the LeafAttrPool unchanged. Smooth's morph pass ALSO emits
    // Remove + Add (cavity-fill / bump-shave / cavity-wall) edits,
    // which flow through the regular `set_cell_empty_cached` /
    // `set_cell_solid_cached` paths above just like Carve / Raise.
    let renormalized_slots: Vec<(u32, Vec3)> = delta
        .edits
        .iter()
        .filter_map(|edit| {
            if let LeafEditOp::SetNormal { slot, normal } = edit.op {
                Some((slot, normal))
            } else {
                None
            }
        })
        .collect();

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
            segment_start: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(0.5, 0.5, 0.5),
            radius: 1.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(4.5, 4.5, 4.5),
            radius: 0.4,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(4.5, 4.5, 4.5),
            radius: CELL_DIAG_HALF,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 99,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(4.0, 4.0, 4.0),
            radius: 0.5,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert!(delta.is_empty(), "Raise on Interior cells is a no-op");
    }

    #[test]
    fn brush_completely_outside_bounds() {
        let mut t = SparseOctree::new(3, 1.0); // extent = 8
        let pool = fresh_pool();
        // Brush centered well outside the cube — no cells in range.
        let op = BrushOp {
            center: Vec3::new(-10.0, -10.0, -10.0),
            segment_start: Vec3::new(-10.0, -10.0, -10.0),
            radius: 1.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert!(delta.is_empty());

        // And on the far side.
        let op2 = BrushOp { center: Vec3::splat(100.0), ..op };
        let delta2 = compute_brush_edits(&t, &pool, &[], op2);
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
            segment_start: Vec3::new(3.5, 3.5, 4.5),
            radius: 1.5,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            LeafEdit { coord: UVec3::new(0, 1, 0), op: LeafEditOp::Add { material: 5, normal: Vec3::Y } },
            LeafEdit { coord: UVec3::new(1, 1, 0), op: LeafEditOp::Add { material: 5, normal: Vec3::Y } },
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
        // brush's +X face. Its analytical brush-SDF normal — carried
        // on the `LeafEditOp::Add` op by `compute_brush_edits` —
        // should point in +X (away from the newly-INTERIOR bulk).
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(8.0, 8.5, 8.5),
            segment_start: Vec3::new(8.0, 8.5, 8.5),
            radius: 4.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(4.0, 4.0, 3.5),
            radius: 1.5,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            LeafEdit { coord: UVec3::new(1, 0, 0), op: LeafEditOp::Add { material: 7, normal: Vec3::Y } },
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
            LeafEdit { coord: UVec3::new(2, 2, 2), op: LeafEditOp::Add { material: 1, normal: Vec3::Y } },
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
            segment_start: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Raise,
            material: 4,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
    fn cavity_wall_normal_points_into_cavity() {
        // Carve a solid INTERIOR cube. A cavity-wall cell on the +X
        // side of the brush should have its analytical brush-SDF
        // normal pointing in -X (toward brush center = into the
        // carved-out cavity).
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            segment_start: Vec3::new(8.0, 8.0, 8.0),
            radius: 3.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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
            segment_start: Vec3::new(4.0, 4.0, 4.0),
            radius: 2.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
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

    // ── P1 Inflate / Deflate tests ───────────────────────────────

    #[test]
    fn inflate_zero_strength_is_noop() {
        // Strength = 0 → max_k = 0 → kernel returns empty delta
        // without doing any brushfire work. Verifies the early bail.
        let mut t = SparseOctree::new(4, 1.0);
        let pool = fresh_pool();
        for y in 0..16 { for x in 0..16 { t.insert(UVec3::new(x, y, 0), 0); } }
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 1.0),
            segment_start: Vec3::new(8.0, 8.0, 1.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 0.0,
            mode: BrushMode::Inflate,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert_eq!(delta.count_added(), 0);
        assert_eq!(delta.count_removed(), 0);
    }

    #[test]
    fn inflate_flat_slab_raises_falloff_shaped_ridge() {
        // Build a 32³ tree with a solid floor at z=0. Inflate at the
        // floor center with radius 8, strength 6. The brushfire
        // should walk upward from the floor; falloff caps thickness
        // by horizontal distance, so the column under brush center
        // gets ~strength voxels of new material and rim columns get
        // 0-1 voxels.
        let mut t = SparseOctree::new(5, 1.0);
        let pool = fresh_pool();
        let mut slot = 0u32;
        for y in 0..32 {
            for x in 0..32 {
                t.insert(UVec3::new(x, y, 0), slot);
                slot += 1;
            }
        }
        let op = BrushOp {
            center: Vec3::new(16.0, 16.0, 0.5),
            segment_start: Vec3::new(16.0, 16.0, 0.5),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 6.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert!(delta.count_added() > 0, "expected Inflate adds on the slab");
        assert_eq!(delta.count_removed(), 0, "Inflate must not remove anything");

        let count_column = |cx: u32, cy: u32| {
            delta.edits.iter().filter(|e| {
                e.coord.x == cx && e.coord.y == cy
                    && matches!(e.op, LeafEditOp::Add { .. })
            }).count()
        };
        let center_height = count_column(16, 16);
        let rim_height = count_column(22, 16);
        assert!(
            center_height > rim_height,
            "center column should be taller than rim column: center={center_height} rim={rim_height}",
        );
        assert!(
            center_height <= 6,
            "center thickness must not exceed strength=6, got {center_height}",
        );

        // Every Add must use the brush material.
        for edit in &delta.edits {
            if let LeafEditOp::Add { material, .. } = edit.op {
                assert_eq!(material, 9);
            }
        }
    }

    #[test]
    fn inflate_pinhole_free() {
        // Every Inflate-emitted Add must have at least one
        // pre-existing SOLID 6-neighbour — that's the "attached to
        // the surface" invariant the brushfire seeds enforce. A
        // floating Add would mean the kernel grew material across a
        // void, which would produce a disconnected ghost shell.
        let mut t = SparseOctree::new(5, 1.0);
        let pool = fresh_pool();
        let mut slot = 0u32;
        for y in 0..32 {
            for x in 0..32 {
                t.insert(UVec3::new(x, y, 0), slot);
                slot += 1;
            }
        }
        let op = BrushOp {
            center: Vec3::new(16.0, 16.0, 0.5),
            segment_start: Vec3::new(16.0, 16.0, 0.5),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        let solid_at = |t: &SparseOctree, brick_pool: &BrickPool, c: UVec3| -> bool {
            !matches!(
                t.cell_state(c, brick_pool),
                CellState::Empty | CellState::OutOfBounds,
            )
        };
        // Adds form a multi-layer shell; some second-layer cells
        // attach via first-layer Adds, not the original surface.
        // Walk in dist order: any Add cell must have a solid
        // 6-neighbour among (original-floor ∪ earlier Adds in this
        // delta). The kernel emits Adds in row-major (z, y, x)
        // order with z increasing → first-layer adds (z=1) appear
        // before second-layer adds (z=2) etc.
        let mut applied_adds: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        for edit in &delta.edits {
            if !matches!(edit.op, LeafEditOp::Add { .. }) {
                continue;
            }
            let c = edit.coord;
            let neighbors = [
                IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z,
            ];
            let has_anchor = neighbors.iter().any(|&d| {
                let n = IVec3::new(c.x as i32, c.y as i32, c.z as i32) + d;
                if n.x < 0 || n.y < 0 || n.z < 0 { return false; }
                let nu = UVec3::new(n.x as u32, n.y as u32, n.z as u32);
                solid_at(&t, &pool, nu) || applied_adds.contains(&nu)
            });
            assert!(has_anchor, "floating Inflate cell at {c}");
            applied_adds.insert(c);
        }
    }

    #[test]
    fn deflate_zero_strength_is_noop() {
        let mut t = SparseOctree::new(4, 1.0);
        let pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 16.0),
            segment_start: Vec3::new(8.0, 8.0, 16.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 0.0,
            mode: BrushMode::Deflate,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert_eq!(delta.count_added(), 0);
        assert_eq!(delta.count_removed(), 0);
    }

    #[test]
    fn deflate_solid_block_carves_smooth_pit() {
        // 16³ all-Interior block, Deflate at the +Z face center
        // (just outside the asset extent so OOB-as-Empty seeds the
        // brushfire downward into the bulk). Falloff curve makes the
        // pit deepest at center and shallowest at rim.
        let mut t = SparseOctree::new(4, 1.0);
        let pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 16.0),
            segment_start: Vec3::new(8.0, 8.0, 16.0),
            radius: 5.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Deflate,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert!(delta.count_removed() > 0, "Deflate must erode some cells");
        assert!(delta.count_added() > 0, "Deflate must add cavity walls on the pit rim");
        // Pit goes deeper at center than at rim. Walk each column
        // and find the deepest eroded cell — center column should
        // reach further into the bulk than a rim column.
        let pit_depth_at = |cx: u32, cy: u32| -> Option<u32> {
            delta.edits.iter()
                .filter(|e| e.coord.x == cx && e.coord.y == cy
                    && matches!(e.op, LeafEditOp::Empty | LeafEditOp::Remove))
                .map(|e| 16u32.saturating_sub(e.coord.z) - 1)
                .max()
        };
        let center_depth = pit_depth_at(8, 8).unwrap_or(0);
        let rim_depth = pit_depth_at(12, 8).unwrap_or(0);
        assert!(
            center_depth > rim_depth,
            "pit should be deeper at center than at rim: center_depth={center_depth} rim_depth={rim_depth}",
        );
    }

    #[test]
    fn deflate_cavity_wall_normal_points_into_pit() {
        // 16³ all-Interior block, Deflate at the +Z face. Pick a
        // cavity-wall cell on the floor of the pit (below the brush
        // center) — its outward normal should point toward +Z (out
        // of the bulk, into the carved-out volume).
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 16.0),
            segment_start: Vec3::new(8.0, 8.0, 16.0),
            radius: 5.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Deflate,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        // Find an Add whose coord is along the brush axis on the
        // pit floor — directly under the brush center.
        let add_under_center = delta.edits.iter()
            .find(|e| e.coord.x == 8 && e.coord.y == 8 && matches!(e.op, LeafEditOp::Add { .. }))
            .expect("expected cavity-wall Add along brush axis");
        let pos_in_edits = delta.edits.iter().position(|e| std::ptr::eq(e, *&add_under_center))
            .or_else(|| delta.edits.iter()
                .position(|e| e.coord == add_under_center.coord
                    && matches!(e.op, LeafEditOp::Add { .. })))
            .unwrap();
        let add_pass_idx = delta.edits[..pos_in_edits].iter()
            .filter(|e| matches!(e.op, LeafEditOp::Add { .. })).count();
        let mut next = 0u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || { let s = next; next += 1; s });
        let (_, attrs) = applied.allocated_slots[add_pass_idx];
        // Brush center is at (8, 8, 16); a cavity-wall cell at
        // (8, 8, z<16) sits below the brush. brush_add_normal with
        // BrushMode::Deflate returns -(cell - center) normalised, so
        // along the -z direction from cell to center → +Z. The wall
        // faces up into the pit.
        assert!(attrs.normal.z > 0.85,
            "cavity-wall normal under brush center should point +Z, got {:?}", attrs.normal);
    }

    #[test]
    fn inflate_far_from_surface_emits_nothing() {
        // A brush hovering in empty space far from any surface
        // shouldn't add anything — the brushfire never reaches a
        // seed, so every empty cell stays at u8::MAX.
        let t = SparseOctree::new(5, 1.0);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(16.0, 16.0, 16.0),
            segment_start: Vec3::new(16.0, 16.0, 16.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Inflate,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &[], op);
        assert_eq!(delta.count_added(), 0);
    }

    // ── Capsule-sweep helpers ────────────────────────────────────────

    fn sphere_op(center: Vec3, radius: f32) -> BrushOp {
        BrushOp {
            center,
            segment_start: center,
            radius,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        }
    }

    fn capsule_op(start: Vec3, end: Vec3, radius: f32) -> BrushOp {
        BrushOp {
            center: end,
            segment_start: start,
            radius,
            falloff_curve: FalloffCurve::Constant,
            strength: 0.0,
            mode: BrushMode::Carve,
            material: 0,
        }
    }

    /// Degenerate capsule (`segment_start == center`) must reproduce
    /// the sphere-SDF behaviour bit-for-bit so the 481 existing
    /// kernel tests stay valid.
    #[test]
    fn capsule_degenerates_to_sphere_when_endpoints_match() {
        let op = sphere_op(Vec3::new(10.0, 10.0, 10.0), 4.0);
        // Inside.
        assert!(brush_sdf(Vec3::new(11.0, 10.0, 10.0), &op) < 0.0);
        // On boundary (within float epsilon).
        assert!(brush_sdf(Vec3::new(14.0, 10.0, 10.0), &op).abs() < 1e-4);
        // Outside.
        assert!(brush_sdf(Vec3::new(20.0, 10.0, 10.0), &op) > 0.0);
        // closest_on_axis pins to the (degenerate) segment endpoint.
        assert_eq!(closest_on_axis(Vec3::new(11.0, 10.0, 10.0), &op), op.center);
    }

    /// Cell directly perpendicular to the midpoint of the segment is
    /// inside the capsule iff its perpendicular distance < radius —
    /// the cylindrical body case where overlapping-spheres math would
    /// produce a meeting-circle crease.
    #[test]
    fn capsule_cylindrical_body_inside_when_axis_distance_under_radius() {
        let op = capsule_op(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            3.0,
        );
        // Midpoint of segment, 2 units perpendicular → axis_distance
        // = 2.0 < 3.0 → inside (SDF negative).
        let p_inside = Vec3::new(5.0, 2.0, 0.0);
        assert!(axis_distance(p_inside, &op) < op.radius);
        assert!(brush_sdf(p_inside, &op) < 0.0);
        // Same X, 3.5 perpendicular → outside.
        let p_outside = Vec3::new(5.0, 3.5, 0.0);
        assert!(brush_sdf(p_outside, &op) > 0.0);
        // Closest point on axis at the midpoint X (clamped t = 0.5).
        let closest = closest_on_axis(p_inside, &op);
        assert!((closest - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4);
    }

    /// Past the segment endpoints the capsule degenerates to the
    /// hemispherical caps — distance computed against the *endpoint*
    /// (clamped t = 0 or t = 1), not the infinite axis line.
    #[test]
    fn capsule_endpoint_caps_behave_like_spheres() {
        let op = capsule_op(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            3.0,
        );
        // Point beyond the +X cap, 2 units past on the axis. Closest
        // point on segment is the (10, 0, 0) endpoint. Distance = 2.
        let p = Vec3::new(12.0, 0.0, 0.0);
        assert!((axis_distance(p, &op) - 2.0).abs() < 1e-4);
        assert!(brush_sdf(p, &op) < 0.0);
        // 4 units past → outside.
        let p_out = Vec3::new(14.0, 0.0, 0.0);
        assert!(brush_sdf(p_out, &op) > 0.0);
        // closest_on_axis clamps t to 1.0 at the +X end.
        assert!((closest_on_axis(p, &op) - op.center).length() < 1e-4);
    }

    /// `brush_cell_range` must enclose every cell the capsule can
    /// touch — i.e. the union of the two endpoint spheres' AABBs.
    /// Bug here would let the kernel walk a too-small footprint and
    /// silently drop edits at the leading or trailing endpoint.
    #[test]
    fn capsule_aabb_covers_both_endpoint_spheres() {
        let op = capsule_op(
            Vec3::new(20.0, 30.0, 40.0),
            Vec3::new(60.0, 30.0, 40.0),
            5.0,
        );
        let (lo, hi) = brush_cell_range(&op, 128);
        // Lo bound = min(start - r, end - r) = (15, 25, 35) floored.
        assert_eq!(lo, UVec3::new(15, 25, 35));
        // Hi bound is exclusive and ceiled +1 for the cell that
        // contains the rightmost float: +X = ceil(65)+1 = 66, +Y/+Z
        // = ceil(35/45)+1 = 36 / 46.
        assert_eq!(hi, UVec3::new(66, 36, 46));
    }

    /// `brush_add_normal` on the cylindrical body must point purely
    /// perpendicular to the axis — no longitudinal component (that
    /// would tilt the wall normal forward/backward along the drag
    /// direction and re-introduce visible banding).
    #[test]
    fn capsule_brush_add_normal_is_purely_radial_on_cylinder() {
        // Brush axis runs through (y=0.5, z=0.5) — cell-center
        // height. Without this offset, the cell at (10, 6, 0) has
        // center (10.5, 6.5, 0.5), 0.5 above the y=0 axis in Z, and
        // the resulting normal picks up a Z component. The point of
        // the test is the cylindrical-radial direction, not the
        // sub-voxel cell-center offset, so align the axis to the
        // cell-center grid.
        let mut op = capsule_op(
            Vec3::new(0.0, 0.5, 0.5),
            Vec3::new(20.0, 0.5, 0.5),
            5.0,
        );
        op.mode = BrushMode::Deflate;
        let n = brush_add_normal(UVec3::new(10, 6, 0), &op);
        assert!(n.x.abs() < 1e-4, "x component should be ~0 on cylinder body, got {}", n.x);
        assert!(n.y < -0.99, "y component should point toward the axis (−Y), got {}", n.y);
        assert!(n.z.abs() < 1e-4, "z component should be ~0 when axis is aligned, got {}", n.z);
    }

    // ── Transit-brushfire (pre-stroke seed restriction) ──────────────

    /// Stamp 2 at the SAME position with stamp 1's edited cells in the
    /// stroke-edit set must NOT deepen the pit. The capped-depth
    /// invariant: brushfire seeds only from pre-stroke Empty (the +Z
    /// face), so the second stamp's brushfire can't restart distance
    /// at 0 inside stamp 1's eroded region — every reachable Solid
    /// /Interior cell is already past the target_thickness boundary.
    #[test]
    fn deflate_stationary_stamp_does_not_deepen_with_stroke_edit_set() {
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 16.0),
            segment_start: Vec3::new(8.0, 8.0, 16.0),
            radius: 5.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Deflate,
            material: 7,
        };
        // Stamp 1 — no stroke history. Apply to mutate the octree.
        let delta1 = compute_brush_edits(&t, &pool, &[], op);
        assert!(delta1.count_removed() > 0, "stamp 1 must erode some cells");
        let mut touched: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        for e in &delta1.edits {
            touched.insert(e.coord);
        }
        let mut next = 0u32;
        apply_delta(&mut t, &mut pool, &delta1, || { let s = next; next += 1; s });

        // Stamp 2 — same position, but with the stroke-edit closure
        // identifying every cell stamp 1 touched. Without the
        // restriction, brushfire would reseed from stamp 1's newly-
        // empty cells and emit a fresh layer of `Empty` edits one
        // deeper.
        let delta2 = compute_brush_edits_in_stroke(
            &t, &pool, &[], op, |c| touched.contains(&c),
        );
        // The only edits stamp 2 may legitimately emit are walls
        // that stamp 1 ALREADY emitted at the same coord (the
        // brushfire still reaches them at dist = 1, which is ≤
        // target_thickness for any positive strength). Empty edits
        // would mean we're carving a deeper layer — that's the bug.
        let new_empties = delta2.edits.iter().filter(|e| {
            matches!(e.op, LeafEditOp::Empty | LeafEditOp::Remove)
        }).count();
        assert_eq!(
            new_empties, 0,
            "stationary stamp 2 must not emit new Remove/Empty edits — \
             that means depth compounding, the original bug. \
             new_empties = {new_empties}",
        );
    }

    /// First stamp of a stroke (touched set empty) must behave
    /// identically to today's behaviour — no edit-set means every
    /// pre-stroke Empty cell seeds brushfire as before.
    #[test]
    fn deflate_first_stamp_matches_legacy_compute_brush_edits() {
        let mut t = SparseOctree::new(4, 1.0);
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(8.0, 8.0, 16.0),
            segment_start: Vec3::new(8.0, 8.0, 16.0),
            radius: 5.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 4.0,
            mode: BrushMode::Deflate,
            material: 0,
        };
        let legacy = compute_brush_edits(&t, &pool, &[], op);
        let in_stroke = compute_brush_edits_in_stroke(&t, &pool, &[], op, |_| false);
        // Edit lists must be byte-identical: same cells, same ops,
        // same order. Any divergence would mean the transit code
        // path leaked into the no-stroke-history case.
        assert_eq!(legacy.edits.len(), in_stroke.edits.len());
        for (a, b) in legacy.edits.iter().zip(in_stroke.edits.iter()) {
            assert_eq!(a.coord, b.coord);
            assert_eq!(
                std::mem::discriminant(&a.op),
                std::mem::discriminant(&b.op),
            );
        }
    }

    /// During a drag, stamp 2 at an OFFSET position can reach into
    /// stamp 1's wall cells (currently Solid, in the touched set)
    /// and emit Remove for them. That's the wall-cleanup behaviour
    /// that produces a smooth channel instead of scallop ridges.
    #[test]
    fn deflate_drag_stamp_removes_previous_stamp_walls() {
        let mut t = SparseOctree::new(4, 1.0);
        let mut pool = fresh_pool();
        for z in 0..16 { for y in 0..16 { for x in 0..16 {
            t.insert_interior(UVec3::new(x, y, z));
        }}}
        let op1 = BrushOp {
            center: Vec3::new(6.0, 8.0, 16.0),
            segment_start: Vec3::new(6.0, 8.0, 16.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 3.0,
            mode: BrushMode::Deflate,
            material: 9,
        };
        let delta1 = compute_brush_edits(&t, &pool, &[], op1);
        let mut touched: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        let mut walled_coords: Vec<UVec3> = Vec::new();
        for e in &delta1.edits {
            touched.insert(e.coord);
            if matches!(e.op, LeafEditOp::Add { .. }) {
                walled_coords.push(e.coord);
            }
        }
        assert!(!walled_coords.is_empty(), "stamp 1 must produce cavity walls");
        let mut next = 0u32;
        apply_delta(&mut t, &mut pool, &delta1, || { let s = next; next += 1; s });

        // Stamp 2 offset by +X (drag direction). Its capsule covers
        // stamp 1's region + a new region. Walls between the two
        // stamp centres should be removable.
        let op2 = BrushOp {
            center: Vec3::new(10.0, 8.0, 16.0),
            segment_start: Vec3::new(6.0, 8.0, 16.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 3.0,
            mode: BrushMode::Deflate,
            material: 9,
        };
        let delta2 = compute_brush_edits_in_stroke(
            &t, &pool, &[], op2, |c| touched.contains(&c),
        );
        // At least one of stamp 1's wall cells must be Removed by
        // stamp 2 — otherwise we'd see scallop ridges between the
        // two stamp centres.
        let removed_walls: usize = walled_coords.iter().filter(|w| {
            delta2.edits.iter().any(|e| e.coord == **w && matches!(e.op, LeafEditOp::Remove))
        }).count();
        assert!(
            removed_walls > 0,
            "stamp 2 must Remove at least one of stamp 1's {} wall cells",
            walled_coords.len(),
        );
    }

    // ── Smooth brush ─────────────────────────────────────────────────

    /// Build a Smooth BrushOp at the given centre with full
    /// strength (32 → blend = 1.0 at the centre) and Constant
    /// falloff so the blend rate is uniform across the brush.
    fn smooth_op(center: Vec3, radius: f32) -> BrushOp {
        BrushOp {
            center,
            segment_start: center,
            radius,
            falloff_curve: FalloffCurve::Constant,
            strength: 32.0,
            mode: BrushMode::Smooth,
            material: 0,
        }
    }

    /// Set up a LeafAttrPool with `attrs` at slot indices 0..N.
    /// Returns a `Vec<LeafAttr>` ready to feed `compute_brush_edits`.
    fn pool_with_normals(normals: &[Vec3]) -> Vec<LeafAttr> {
        normals
            .iter()
            .map(|n| LeafAttr {
                material_primary: 1,
                material_secondary_blend: 0,
                normal_oct: pack_oct(*n),
            })
            .collect()
    }

    /// Build a 3×3×3 cube of Solid cells centred at (8,8,8). Slots
    /// are assigned in row-major (x, y, z) order, giving each cell a
    /// unique slot id 0..27. The centre cell (8,8,8) lands at slot 13.
    /// Surrounding the centre in three dimensions means it has 6 Solid
    /// face-neighbours and survives the morph rule (`occupied = 6 > 2`),
    /// so V1-style normal-blend tests can target the centre without
    /// the cube cells being eroded.
    fn cube_3x3x3() -> (SparseOctree, [u32; 27]) {
        let mut t = SparseOctree::new(4, 1.0);
        let mut slots = [0u32; 27];
        let mut next = 0u32;
        for dz in 0..3u32 {
            for dy in 0..3u32 {
                for dx in 0..3u32 {
                    let coord = UVec3::new(7 + dx, 7 + dy, 7 + dz);
                    let slot = next;
                    next += 1;
                    t.insert(coord, slot);
                    let i = (dz * 9 + dy * 3 + dx) as usize;
                    slots[i] = slot;
                }
            }
        }
        (t, slots)
    }

    /// Slot id of the centre cell (8,8,8) in the [`cube_3x3x3`]
    /// layout. Row-major (x, y, z): cell at (dx=1, dy=1, dz=1) is at
    /// linear index 9 + 3 + 1 = 13.
    const CUBE_CENTRE_SLOT: u32 = 13;

    /// Build a `LeafAttrPool` for a 3×3×3 cube where every slot gets
    /// the same `default_normal` except the centre, which gets
    /// `centre_normal`. The cube's slots run 0..27 with the centre at
    /// [`CUBE_CENTRE_SLOT`].
    fn cube_normals(default_normal: Vec3, centre_normal: Vec3) -> Vec<LeafAttr> {
        (0..27u32)
            .map(|s| LeafAttr {
                material_primary: 1,
                material_secondary_blend: 0,
                normal_oct: pack_oct(
                    if s == CUBE_CENTRE_SLOT { centre_normal } else { default_normal },
                ),
            })
            .collect()
    }

    /// Isolated Solid cell (no occupied 6-neighbours): the morph rule
    /// triggers bump-shave (`occupied = 0 ≤ 2`) and emits a Remove,
    /// since one floating voxel IS the kind of noise Smooth should
    /// erase. V1 normal-only smoothing was a no-op here; V2 explicitly
    /// removes the cell.
    #[test]
    fn smooth_morph_erodes_isolated_cell() {
        let t = one_leaf(4, UVec3::new(8, 8, 8), 0);
        let pool_data = pool_with_normals(&[Vec3::Y]);
        let pool = fresh_pool();
        let op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 2.0);
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        let removed = delta
            .edits
            .iter()
            .filter(|e| e.coord == UVec3::new(8, 8, 8) && matches!(e.op, LeafEditOp::Remove))
            .count();
        assert_eq!(removed, 1, "isolated cell must morph-shave to a single Remove");
        // No SetNormal — the cell is being deleted, not blended.
        let set_normal_count = delta
            .edits
            .iter()
            .filter(|e| matches!(e.op, LeafEditOp::SetNormal { .. }))
            .count();
        assert_eq!(set_normal_count, 0, "shaved cell must not also emit SetNormal");
    }

    /// 3×3×3 cube with every cell's normal = +Y. The centre cell has
    /// 6 Solid neighbours (one per face) all carrying the same normal,
    /// so the neighbour average IS +Y and the blend is identity. The
    /// centre survives the morph rule (`occupied = 6 > 2`).
    #[test]
    fn smooth_uniform_normals_is_noop() {
        let (t, _slots) = cube_3x3x3();
        let pool_data = cube_normals(Vec3::Y, Vec3::Y);
        let pool = fresh_pool();
        let op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 1.5);
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        // Whatever SetNormals fire, every result must point ≈ +Y.
        let mut any = false;
        for edit in &delta.edits {
            if let LeafEditOp::SetNormal { normal, .. } = edit.op {
                any = true;
                let dot = normal.dot(Vec3::Y);
                assert!(
                    dot > 0.9999,
                    "uniform-normal smoothing should leave the normal pointing +Y, got {normal:?}",
                );
            }
        }
        assert!(any, "cube centre should emit at least one SetNormal");
        // Sanity: nothing eroded — every in-brush cube cell has ≥ 3
        // occupied 6-neighbours.
        assert_eq!(delta.count_removed(), 0);
        assert_eq!(delta.count_added(), 0);
    }

    /// 3×3×3 cube. Centre normal = +Y, every surrounding cell's
    /// normal = +Z. Smooth at full strength (32 → blend = 1.0) must
    /// rewrite the centre's normal to ≈ +Z (the local average).
    #[test]
    fn smooth_blends_toward_neighbour_average() {
        let (t, slots) = cube_3x3x3();
        let pool_data = cube_normals(Vec3::Z, Vec3::Y);
        let pool = fresh_pool();
        let op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 1.5);
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        let centre = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::SetNormal { slot, normal } = e.op {
                    return Some((slot, normal));
                }
            }
            None
        });
        let (slot, normal) =
            centre.expect("centre cell must emit a SetNormal at full-strength blend");
        assert_eq!(slot, slots[13]);
        assert!(
            normal.dot(Vec3::Z) > 0.99,
            "centre normal should be ≈ +Z after blend, got {normal:?}",
        );
    }

    /// Constant falloff + strength=16 puts blend at exactly 0.5.
    /// Centre normal +Y, neighbours +Z → blended result ≈ normalize(+Y + +Z).
    /// Both Y and Z components must be positive and roughly equal magnitude.
    #[test]
    fn smooth_half_strength_produces_midway_blend() {
        let (t, _slots) = cube_3x3x3();
        let pool_data = cube_normals(Vec3::Z, Vec3::Y);
        let pool = fresh_pool();
        let mut op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 1.5);
        op.strength = 16.0;
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        let centre_normal = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::SetNormal { normal, .. } = e.op {
                    return Some(normal);
                }
            }
            None
        }).expect("centre cell must emit SetNormal");
        assert!(centre_normal.y > 0.3 && centre_normal.y < 0.8);
        assert!(centre_normal.z > 0.3 && centre_normal.z < 0.8);
    }

    /// 1-voxel bump above a 5×5 flat Solid plane → bump-shave. The
    /// brush capsule is shrunk to radius 0.6 around the bump so only
    /// the bump cell falls inside; plane corners (which would
    /// otherwise morph-erode in this thin-plane test scene) stay
    /// untouched.
    #[test]
    fn smooth_morph_shaves_isolated_bump() {
        let mut t = SparseOctree::new(4, 1.0);
        let mut next = 0u32;
        for x in 6..11u32 {
            for y in 6..11u32 {
                t.insert(UVec3::new(x, y, 8), next);
                next += 1;
            }
        }
        // The bump.
        let bump_slot = next;
        t.insert(UVec3::new(8, 8, 9), bump_slot);
        let pool_data: Vec<LeafAttr> = (0..=bump_slot)
            .map(|_| LeafAttr {
                material_primary: 1,
                material_secondary_blend: 0,
                normal_oct: pack_oct(Vec3::Z),
            })
            .collect();
        let pool = fresh_pool();
        let mut op = smooth_op(Vec3::new(8.5, 8.5, 9.5), 0.6);
        op.material = 7;
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        let bump_removed = delta.edits.iter().any(|e| {
            e.coord == UVec3::new(8, 8, 9) && matches!(e.op, LeafEditOp::Remove)
        });
        assert!(bump_removed, "1-voxel bump must morph-shave to Remove");
        assert_eq!(delta.count_added(), 0, "no Adds expected for bump-shave-only stamp");
    }

    /// 1-voxel pit in a 5×5 Solid plane with Interior bulk below →
    /// cavity-fill. The pit cell at (8,8,8) is Empty; its 6 neighbours
    /// are 4 lateral Solid + 1 Empty above + 1 Interior below =
    /// 5 occupied ≥ 4 → flip to Solid.
    #[test]
    fn smooth_morph_fills_isolated_pit() {
        let mut t = SparseOctree::new(4, 1.0);
        let mut next = 0u32;
        for x in 6..11u32 {
            for y in 6..11u32 {
                if x == 8 && y == 8 {
                    continue;
                }
                t.insert(UVec3::new(x, y, 8), next);
                next += 1;
            }
        }
        // Interior bulk below the plane.
        for x in 6..11u32 {
            for y in 6..11u32 {
                for z in 6..8u32 {
                    t.insert_interior(UVec3::new(x, y, z));
                }
            }
        }
        let pool_data: Vec<LeafAttr> = (0..next)
            .map(|_| LeafAttr {
                material_primary: 1,
                material_secondary_blend: 0,
                normal_oct: pack_oct(Vec3::Z),
            })
            .collect();
        let pool = fresh_pool();
        let mut op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 0.6);
        op.material = 7;
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        let fill = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::Add { material, normal } = e.op {
                    return Some((material, normal));
                }
            }
            None
        });
        let (material, normal) = fill.expect("pit must morph-fill to Add");
        assert_eq!(material, 7, "fill Add should carry op.material");
        // Outward direction is toward the Empty neighbour (above, +z).
        assert!(
            normal.z > 0.5,
            "fill normal should point +z (the empty side), got {normal:?}",
        );
        assert_eq!(delta.count_removed(), 0, "no Removes expected for cavity-fill-only stamp");
    }

    /// Isolated Solid "thumb" with Interior bulk beneath. The thumb at
    /// (8,8,9) has 1 occupied neighbour (the Interior below) → morph-shave
    /// to Remove. The Interior cell at (8,8,8) then has a will-remove
    /// 6-neighbour (+z) → cavity-wall Add, with the new surface normal
    /// pointing toward where the thumb used to be (+z).
    #[test]
    fn smooth_morph_emits_cavity_wall_when_solid_exposes_interior() {
        let mut t = SparseOctree::new(4, 1.0);
        t.insert(UVec3::new(8, 8, 9), 0);
        t.insert_interior(UVec3::new(8, 8, 8));
        t.insert_interior(UVec3::new(8, 8, 7));
        t.insert_interior(UVec3::new(8, 8, 6));
        let pool_data = pool_with_normals(&[Vec3::Z]);
        let pool = fresh_pool();
        let mut op = smooth_op(Vec3::new(8.5, 8.5, 9.0), 1.0);
        op.material = 7;
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);

        let bump_removed = delta.edits.iter().any(|e| {
            e.coord == UVec3::new(8, 8, 9) && matches!(e.op, LeafEditOp::Remove)
        });
        assert!(bump_removed, "isolated bump must morph-shave");

        let wall = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::Add { normal, material } = e.op {
                    return Some((normal, material));
                }
            }
            None
        });
        let (normal, material) =
            wall.expect("Interior beneath the removed bump must emit cavity-wall Add");
        assert_eq!(material, 7, "cavity-wall Add should carry op.material");
        assert!(
            normal.z > 0.9,
            "cavity-wall normal should point +z toward the removed bump, got {normal:?}",
        );
    }

    /// apply_delta round-trip: a 3×3×3 cube under Smooth produces
    /// only SetNormal edits (no morph flips because every in-brush
    /// cell survives), and those SetNormals show up in
    /// `renormalized_slots` for the scene-manager consumer.
    #[test]
    fn apply_delta_surfaces_smooth_in_renormalized_slots() {
        let (mut t, _slots) = cube_3x3x3();
        let pool_data = cube_normals(Vec3::Z, Vec3::Y);
        let mut pool = fresh_pool();
        let op = smooth_op(Vec3::new(8.5, 8.5, 8.5), 1.5);
        let delta = compute_brush_edits(&t, &pool, &pool_data, op);
        // The cube test scene has no morph flips; only SetNormals fire.
        assert_eq!(delta.count_removed(), 0);
        assert_eq!(delta.count_added(), 0);
        let applied = apply_delta(&mut t, &mut pool, &delta, || panic!("no alloc expected"));
        assert!(applied.allocated_slots.is_empty());
        assert!(applied.freed_slots.is_empty());
        let set_normals: Vec<(u32, Vec3)> = delta
            .edits
            .iter()
            .filter_map(|e| {
                if let LeafEditOp::SetNormal { slot, normal } = e.op {
                    Some((slot, normal))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(applied.renormalized_slots.len(), set_normals.len());
        for ((a_slot, a_n), (e_slot, e_n)) in
            applied.renormalized_slots.iter().zip(set_normals.iter())
        {
            assert_eq!(a_slot, e_slot);
            assert!((*a_n - *e_n).length() < 1e-4);
        }
    }
}
