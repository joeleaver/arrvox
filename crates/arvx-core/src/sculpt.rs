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
    /// Soft outward dilation as a signed-distance-field OFFSET:
    /// `sd'(p) = sd(p) − offset(p)`, `offset = falloff(t)·strength`,
    /// `t = 1 − axis_dist/radius`. Cells with `sd' ≤ 0` become solid; the
    /// new surface is re-discretised and meshed through the same Manifold-DC
    /// path as terrain / Raise / Carve. Smooth by construction — replaces the
    /// retired pre-SDF brushfire kernel (which stamped brush-sphere distance
    /// on an occupancy shell and shattered DC). See [`compute_offset_edits`].
    Inflate,
    /// Soft inward erosion — the [`Inflate`] mirror with `sd'(p) = sd(p) +
    /// offset(p)`. The receded surface is re-discretised and DC-meshed the
    /// same way. See [`compute_offset_edits`].
    Deflate,
    /// Clay strips — deposits a fixed-height flat-topped strip above
    /// the pre-stroke surface, swept along the capsule axis. Width =
    /// brush radius, height = `strength` cells. The cross-section has
    /// a flat top (75% of radius) with falloff-shaped shoulders at
    /// the lateral edges. Overlapping strokes stack: each adds its
    /// fixed height on top of whatever surface was there when the
    /// stroke began. Uses a brushfire-shell kernel with a flat-top thickness
    /// profile instead of a dome (Inflate/Deflate now use the SDF-offset
    /// [`compute_offset_edits`] path; ClayStrip still uses brushfire).
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

    /// Derivative `d(evaluate)/dt` at `t ∈ [0, 1]`. The SDF-offset
    /// Inflate/Deflate kernel needs this to compute the analytic surface
    /// normal of the offset field: `offset(p) = evaluate(t)·strength` with
    /// `t = 1 − axis_dist/radius`, so the surface normal is
    /// `normalize(∇sd ∓ ∇offset)` where `∇offset` has magnitude
    /// `strength · evaluate_deriv(t) · (1/radius)` along the brush-radial
    /// direction. Deriving the normal analytically (vs finite-differencing
    /// the discretised field) keeps it speckle-free.
    #[inline]
    pub fn evaluate_deriv(self, t: f32) -> f32 {
        match self {
            FalloffCurve::Constant => 0.0,
            FalloffCurve::Smooth => 0.5 * std::f32::consts::PI * (std::f32::consts::PI * t).sin(),
            FalloffCurve::Smoothstep => 6.0 * t * (1.0 - t),
            FalloffCurve::Sharp => 2.0 * t,
            FalloffCurve::Linear => 1.0,
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
    /// **`dist`** is the brush-SDF signed distance from this cell's center
    /// to the new (brush) surface, in VOXEL units, sign-matched to `normal`
    /// (negative inside the new solid). It is the QEF-Hermite companion to
    /// `normal` — together they place the dual vertex exactly on the brush
    /// surface (`p_surf = center − dist·normal`) so a carved/raised face is
    /// smooth, not staircased to the cell center. Euclidean (`|∇|=1`) so it
    /// is stored raw (the terrain/sculpt asymmetry — terrain re-normalizes,
    /// sculpt does not). `0.0` for replayed-from-disk edits (the
    /// `.arvxsculpt` sidecar does not persist it; they re-extract at the
    /// cell center until re-sculpted).
    Add { material: u16, normal: Vec3, dist: f32 },
    /// Rewrite an existing surface cell's `LeafAttr.normal_oct`
    /// without touching its occupancy or material. The Smooth brush
    /// emits these to nudge surface normals toward the local
    /// neighbourhood average. `slot` is the cell's pre-resolved
    /// `LeafAttrPool` slot id — compute time has already done the
    /// octree+brick lookup, so [`apply_delta`] can write the pool
    /// directly without re-traversing.
    SetNormal { slot: u32, normal: Vec3 },
    /// Rewrite an existing surface cell's stored signed distance
    /// (`LeafAttrPool` dist) in place, keeping its occupancy / material /
    /// slot. The SDF-offset Inflate/Deflate brushes emit these for cells
    /// that stay surface leaves but whose distance to the (offset) surface
    /// changed — without this, an old surface cell that the offset pushed
    /// sub-surface keeps its stale near-zero distance and Manifold-DC emits
    /// a spurious surface there (the shatter the brushfire produced).
    /// `slot` is pre-resolved at compute time (like [`Self::SetNormal`]);
    /// `dist` is the new signed distance in voxel units. May also carry a
    /// refreshed `normal` (the offset preserves direction, but a re-derived
    /// normal keeps the band consistent).
    SetDist { slot: u32, normal: Vec3, dist: f32 },
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

    /// Count of in-place distance rewrites (SDF-offset Inflate/Deflate). These
    /// mutate an existing leaf's stored distance without changing occupancy, so
    /// a delta of only `SetDist` edits IS a real mutation — callers gating on
    /// "did anything change?" must include this (else a gentle Inflate/Deflate
    /// that only re-distances existing surface leaves is silently dropped).
    pub fn count_set_dist(&self) -> usize {
        self.edits.iter().filter(|e| matches!(e.op, LeafEditOp::SetDist { .. })).count()
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
    /// Brush-SDF signed distance (voxel units) for the QEF-Hermite
    /// re-extract; the render consumer writes it via `LeafAttrPool::set_dist`
    /// alongside `to_leaf_attr()`. See [`LeafEditOp::Add`]'s `dist`.
    pub dist: f32,
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
    /// Pre-existing SOLID slots whose stored DISTANCE (and normal) the
    /// SDF-offset Inflate/Deflate moved — `(slot, new_normal, new_dist)`.
    /// The caller writes `pool.set_dist(slot, dist)` + `normal_oct` in
    /// place; occupancy / material / slot unchanged. Keeps an old surface
    /// cell's stored field consistent with the offset surface so DC does
    /// not emit a stale surface at its former position.
    pub redist_slots: Vec<(u32, Vec3, f32)>,
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
    dists: &[i16],
    op: BrushOp,
) -> SculptDelta {
    // Inflate / Deflate paths default `is_stroke_edit` to "always
    // false" — meaning the kernel treats every cell's current state as
    // its pre-stroke state. Callers that need the one-layer-per-stroke
    // semantic (the scene manager's drag path) go through
    // `compute_brush_edits_in_stroke` and supply the actual closure.
    compute_brush_edits_in_stroke(octree, brick_pool, leaf_attr_pool, dists, op, |_| false)
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
    dists: &[i16],
    op: BrushOp,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    match op.mode {
        BrushMode::Inflate => {
            return compute_offset_edits(
                octree,
                brick_pool,
                leaf_attr_pool,
                dists,
                &op,
                OffsetSign::Inflate,
                is_stroke_edit,
            );
        }
        BrushMode::Deflate => {
            return compute_offset_edits(
                octree,
                brick_pool,
                leaf_attr_pool,
                dists,
                &op,
                OffsetSign::Deflate,
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
    let _ = dists;
    let _ = is_stroke_edit;

    use std::time::Instant;

    let extent = octree.extent();
    // The walker covers the brush footprint PLUS one cell of padding so
    // the outside-rim cells whose state drives the cavity-wall /
    // dome-surface rules are in scope. The inside surface band
    // ([`emit_raise_inside`]) lives within the footprint, so one cell of
    // pad still suffices.
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
                    (BrushMode::Raise, true) => emit_raise_inside(&mut edits, coord, state, d, &op),
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

/// Direction of a soft SDF-field offset brush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OffsetSign {
    /// Grow solid outward: `sd' = sd − offset`.
    Inflate,
    /// Erode solid inward: `sd' = sd + offset`.
    Deflate,
}

/// **SDF-offset Inflate / Deflate** — the SDF-native replacement for the
/// retired brushfire kernel.
///
/// The brushfire kernel grew/eroded occupancy by integer 6-face hop counts
/// and then stamped each touched cell with its distance to the brush
/// *sphere* (`brush_add_dist`). That distance disagreed with the occupancy
/// boundary (every shell cell is "inside" the sphere → all negative), so
/// Manifold-DC placed vertices at the wrong surface and the result
/// shattered. This kernel instead treats Inflate / Deflate as a genuine
/// **offset of the signed-distance field** and re-discretises it, meshing
/// through the same DC path as terrain / Raise / Carve — smooth by
/// construction, one unified surface model.
///
/// Pipeline (all in finest-voxel grid units):
///
/// 1. **Reconstruct a continuous SDF over the brush region.** Only surface
///    *leaves* store a signed distance + normal. Each leaf defines a Hermite
///    tangent plane through its sub-voxel surface point
///    `p_surf = center − dist·n` (the same plane Manifold-DC extrapolates
///    from). A vector distance transform (6-neighbour relaxation, two
///    sweeps per pass, looped to convergence) propagates the nearest leaf's
///    `(p_surf, n)` to every region cell, so `sd(c) = (c − p_surf)·n` is a
///    continuous reconstruction of the field — exactly what DC already
///    trusts. Degrades gracefully when distances are absent (`dist` → 0,
///    plane through the cell center) or a normal is degenerate (→ +Y).
///
/// 2. **Offset.** `offset(c) = falloff(t)·strength` with
///    `t = 1 − axis_dist/radius` (capsule-aware). Inflate: `sd' = sd −
///    offset`; Deflate: `sd' = sd + offset`. Cells with `sd' ≤ 0` are solid.
///
/// 3. **Re-discretise to leaf edits.** Within a `SCULPT_BAND_HALF_WIDTH`
///    band of the new zero-crossing each solid cell becomes a surface leaf
///    carrying `(dist = sd', normal)`; deeper solid is interior; cells the
///    offset pushed outside go empty. Crucially, a *pre-existing* surface
///    leaf the offset pushed sub-surface is updated in place via
///    [`LeafEditOp::SetDist`] (deep-negative distance) so DC no longer emits
///    a spurious surface at its former position — the shatter the brushfire
///    produced.
///
/// **Normal.** The surface leaf's normal is the analytic gradient of the
/// offset field, `normalize(∇sd ± ∇offset)` (`−` Inflate, `+` Deflate),
/// where `∇sd = n` (nearest-leaf plane) and `∇offset` is the brush-radial
/// gradient with magnitude `strength·falloff'(t)/radius`. Deriving it
/// analytically (vs finite-differencing the discretised field) keeps the
/// dome / pit flanks speckle-free and correctly tilted on the falloff
/// shoulder. On a flat ground (`n = +Y`) this reproduces a clean dome.
///
/// **Stroke capping.** `is_stroke_edit(c)` excludes cells already edited
/// this stroke from *seeding* the nearest-surface field, so `sd0` is always
/// reconstructed from the pristine pre-stroke surface and the offset applies
/// once however many stamps cover a cell — a held / dragged brush converges
/// to one offset layer instead of compounding. For a single stamp the
/// predicate is always false and every surface leaf seeds.
#[allow(clippy::too_many_arguments)]
fn compute_offset_edits(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &[LeafAttr],
    dists: &[i16],
    op: &BrushOp,
    sign: OffsetSign,
    is_stroke_edit: impl Fn(UVec3) -> bool,
) -> SculptDelta {
    use std::time::Instant;

    let extent = octree.extent();
    // Max reach in voxels — used for region padding (so the surrounding
    // surface is in scope as seeds) and as a convergence bound.
    let max_k = (op.strength.max(0.0).ceil() as u32).min(254);
    if max_k == 0 || op.radius <= 0.0 {
        return SculptDelta::default();
    }
    // Pad by reach + 2 so seeds around the brush are captured and the
    // nearest-surface transform can reach every emit cell from a seed.
    let (lo, hi) = brush_cell_range_padded(op, extent, max_k + 2);
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
    let cell_center =
        |c: UVec3| Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5);

    // Per-cell state: 0=Empty, 1=Interior, 2=Solid(slot), 3=OutOfBounds.
    let mut state_kind: Vec<u8> = vec![0; total];
    let mut slot_grid: Vec<u32> = vec![u32::MAX; total];
    // Nearest-surface vector field: squared euclidean distance to the
    // nearest seed leaf's sub-voxel surface point, that point, and the
    // leaf's outward normal. `INFINITY` best_d2 = "no surface reached yet".
    let mut best_d2: Vec<f32> = vec![f32::INFINITY; total];
    let mut p_surf_grid: Vec<Vec3> = vec![Vec3::ZERO; total];
    let mut normal_grid: Vec<Vec3> = vec![Vec3::ZERO; total];

    let mut cache = crate::sparse_octree::CellStateCache::new();
    let mut n_cell_state_calls = 0u32;
    let mut n_aabb_cells = 0u32;

    // ── Init: classify every region cell + seed the nearest-surface field
    //    from surface leaves. ──
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                n_aabb_cells += 1;
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                let st = octree.cell_state_cached(c, brick_pool, &mut cache);
                n_cell_state_calls += 1;
                match st {
                    CellState::Empty => state_kind[i] = 0,
                    CellState::Interior => state_kind[i] = 1,
                    CellState::OutOfBounds => state_kind[i] = 3,
                    CellState::Solid(slot) => {
                        state_kind[i] = 2;
                        slot_grid[i] = slot;
                        // STROKE CAPPING: a cell already edited this stroke is
                        // NOT a seed. The nearest-surface field then propagates
                        // in from the *un-touched* (pristine, pre-stroke)
                        // surface around the footprint, so `sd0` always measures
                        // from the ORIGINAL surface — the offset is applied once
                        // no matter how many stamps cover the cell. Without this
                        // a held / dragged brush would re-offset the already-
                        // offset surface every stamp ("keeps puffing the longer
                        // I hold the mouse"). For a single stamp the predicate
                        // is always false, so every surface leaf seeds.
                        if is_stroke_edit(c) {
                            continue;
                        }
                        let n = leaf_attr_pool
                            .get(slot as usize)
                            .map(|a| unpack_oct(a.normal_oct))
                            .filter(|n| n.length_squared() > 1e-10)
                            .map(|n| n.normalize())
                            .unwrap_or(Vec3::Y);
                        let d = dists
                            .get(slot as usize)
                            .copied()
                            .map(crate::LeafAttrPool::dequantize_dist)
                            .unwrap_or(0.0);
                        let cc = cell_center(c);
                        let p_surf = cc - d * n;
                        best_d2[i] = (cc - p_surf).length_squared();
                        p_surf_grid[i] = p_surf;
                        normal_grid[i] = n;
                    }
                }
            }
        }
    }

    // ── Vector distance transform: relax until no cell adopts a closer
    //    neighbour's surface. Forward sweep checks the −x/−y/−z neighbours,
    //    backward checks +x/+y/+z; the pair covers all 6 directions per
    //    iteration. Bounded by the region span (it always converges well
    //    before — the surface sheet runs through the brush center). ──
    let max_passes = (size.x + size.y + size.z).max(1);
    let mut passes = 0u32;
    let mut changed = true;
    while changed && passes < max_passes {
        changed = false;
        passes += 1;
        // Forward.
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = UVec3::new(x, y, z);
                    let i = idx(c);
                    let cc = cell_center(c);
                    let neigh = [
                        (x > lo.x).then(|| idx(UVec3::new(x - 1, y, z))),
                        (y > lo.y).then(|| idx(UVec3::new(x, y - 1, z))),
                        (z > lo.z).then(|| idx(UVec3::new(x, y, z - 1))),
                    ];
                    for ni in neigh.iter().flatten() {
                        if !best_d2[*ni].is_finite() {
                            continue;
                        }
                        let d2 = (cc - p_surf_grid[*ni]).length_squared();
                        if d2 < best_d2[i] {
                            best_d2[i] = d2;
                            p_surf_grid[i] = p_surf_grid[*ni];
                            normal_grid[i] = normal_grid[*ni];
                            changed = true;
                        }
                    }
                }
            }
        }
        // Backward.
        for z in (lo.z..hi.z).rev() {
            for y in (lo.y..hi.y).rev() {
                for x in (lo.x..hi.x).rev() {
                    let c = UVec3::new(x, y, z);
                    let i = idx(c);
                    let cc = cell_center(c);
                    let neigh = [
                        (x + 1 < hi.x).then(|| idx(UVec3::new(x + 1, y, z))),
                        (y + 1 < hi.y).then(|| idx(UVec3::new(x, y + 1, z))),
                        (z + 1 < hi.z).then(|| idx(UVec3::new(x, y, z + 1))),
                    ];
                    for ni in neigh.iter().flatten() {
                        if !best_d2[*ni].is_finite() {
                            continue;
                        }
                        let d2 = (cc - p_surf_grid[*ni]).length_squared();
                        if d2 < best_d2[i] {
                            best_d2[i] = d2;
                            p_surf_grid[i] = p_surf_grid[*ni];
                            normal_grid[i] = normal_grid[*ni];
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // ── Emit: walk the brush footprint, re-discretise the offset field. ──
    let mut edits = Vec::new();
    let mut n_inside_sphere = 0u32;
    let inv_r = 1.0 / op.radius;
    let band = SCULPT_BAND_HALF_WIDTH;
    let sgn = match sign {
        OffsetSign::Inflate => -1.0_f32,
        OffsetSign::Deflate => 1.0_f32,
    };

    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = UVec3::new(x, y, z);
                let i = idx(c);
                if state_kind[i] == 3 || !best_d2[i].is_finite() {
                    continue; // OOB, or no surface reached this cell
                }
                let cc = cell_center(c);
                let axis_d = axis_distance(cc, op);
                if axis_d > op.radius {
                    continue; // outside the brush footprint → no influence
                }
                let t = (1.0 - axis_d * inv_r).clamp(0.0, 1.0);
                let off = op.falloff_curve.evaluate(t) * op.strength;
                if off <= 1e-3 {
                    continue; // negligible influence at the rim
                }
                n_inside_sphere += 1;

                // Distance-weighted blend of the nearest-surface planes over
                // the 3×3×3 neighbourhood. Each cell carries only ITS nearest
                // seed's plane, so across a seed-territory boundary adjacent
                // cells carry different planes whose values disagree — a
                // discontinuous field that shattered the dome top (max
                // extrapolation = max disagreement). Averaging the plane
                // evaluations AT this cell, weighted by each contributing
                // seed's proximity, gives a C0-smooth reconstruction that
                // still tracks the true surface.
                let (sd0, n_field) = {
                    let mut w_sum = 0.0f32;
                    let mut sd_acc = 0.0f32;
                    let mut n_acc = Vec3::ZERO;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                let nz = z as i32 + dz;
                                if nx < lo.x as i32
                                    || nx >= hi.x as i32
                                    || ny < lo.y as i32
                                    || ny >= hi.y as i32
                                    || nz < lo.z as i32
                                    || nz >= hi.z as i32
                                {
                                    continue;
                                }
                                let j = idx(UVec3::new(nx as u32, ny as u32, nz as u32));
                                if !best_d2[j].is_finite() {
                                    continue;
                                }
                                let pj = p_surf_grid[j];
                                let nj = normal_grid[j];
                                let d2 = (cc - pj).length_squared();
                                let w = 1.0 / (0.25 + d2);
                                sd_acc += w * (cc - pj).dot(nj);
                                n_acc += w * nj;
                                w_sum += w;
                            }
                        }
                    }
                    if w_sum > 0.0 {
                        // `n_field` = the weighted-AVERAGE normal (NOT
                        // renormalised). To first order it IS the gradient of
                        // the blended `sd0` field: where the contributing seed
                        // normals agree (flat / uniform slope) its length is 1;
                        // where they disagree (curvature / seed-territory
                        // boundary) its length drops below 1, exactly encoding
                        // `|∇sd0| < 1` of the smoothed field. Dividing the field
                        // value by this length below recovers the true Euclidean
                        // distance — renormalising it to a unit vector (as an
                        // earlier version did) would over-state `|∇sdp|` on
                        // slopes and shrink the stored distance, mis-placing the
                        // dual vertex.
                        (sd_acc / w_sum, n_acc / w_sum)
                    } else {
                        ((cc - p_surf_grid[i]).dot(normal_grid[i]), normal_grid[i])
                    }
                };
                let sdp = sd0 + sgn * off;

                // The offset field is NOT unit-gradient: moving outward both
                // raises `sd` (toward empty) and lowers `offset`, so the two
                // gradients ADD and `|∇sdp|` reaches ~2 at a near-flat dome top.
                // Left raw, the `[−band, 0]` band would span < 1 real voxel
                // (leaving the layer just under the surface as data-less
                // interior → Manifold-DC's both-sides-sentinel fallback, which
                // spikes the near-tangent top) and the stored distance would be
                // ~2× too large (mis-placing the dual vertex). Normalising by
                // the gradient length recovers a true Euclidean voxel distance
                // — the same `d_vox = d / |∇d|` the terrain bake applies — and
                // the normalised gradient IS the surface normal.
                //
                // ∇sdp = ∇sd + sgn·∇offset = n_field + sgn·grad_off, where
                // ∇offset is brush-radial with magnitude
                // strength·falloff'(t)·(−1/radius) (offset falls as axis_d grows).
                let radial = {
                    let v = cc - closest_on_axis(cc, op);
                    let l2 = v.length_squared();
                    if l2 > 1e-8 {
                        v * l2.sqrt().recip()
                    } else {
                        Vec3::ZERO
                    }
                };
                let grad_off =
                    radial * (op.strength * op.falloff_curve.evaluate_deriv(t) * (-inv_r));
                let grad = n_field + grad_off * sgn;
                let glen = grad.length();
                let (surf_n, dist_vox) = if glen > 1e-4 {
                    // surf_n is the UNIT surface normal (stored); dist_vox is the
                    // Euclidean voxel distance (field value ÷ gradient length).
                    (grad / glen, sdp / glen)
                } else {
                    // Degenerate gradient (opposing seed normals cancel): fall
                    // back to a stable unit direction, leave the field value raw.
                    (normal_grid[i], sdp)
                };

                let kind = state_kind[i];
                // Current accumulated distance for an existing surface leaf
                // (the running field from earlier stamps this stroke; defensive
                // `.get` for the no-distances fallback).
                let prev_dist = |slot: u32| {
                    dists
                        .get(slot as usize)
                        .copied()
                        .map(crate::LeafAttrPool::dequantize_dist)
                        .unwrap_or(0.0)
                };
                match sign {
                    OffsetSign::Inflate => {
                        // Monotonic GROW: across a stroke the surface only moves
                        // outward — the union of the stamp domes (deepest
                        // inflation wins). A re-touched surface leaf's distance
                        // only ever goes MORE negative; a later far stamp never
                        // raises (lowers the surface of) an earlier near stamp's
                        // inflation, which would scallop a drag ridge.
                        match kind {
                            2 => {
                                let slot = slot_grid[i];
                                let prev = prev_dist(slot);
                                let nd = dist_vox.min(prev);
                                if nd < prev - 1e-4 {
                                    edits.push(LeafEdit {
                                        coord: c,
                                        op: LeafEditOp::SetDist { slot, normal: surf_n, dist: nd },
                                    });
                                }
                            }
                            0 if dist_vox <= 0.0 => {
                                if dist_vox >= -band {
                                    edits.push(LeafEdit {
                                        coord: c,
                                        op: LeafEditOp::Add {
                                            material: op.material,
                                            normal: surf_n,
                                            dist: dist_vox,
                                        },
                                    });
                                } else {
                                    edits.push(LeafEdit { coord: c, op: LeafEditOp::SetInterior });
                                }
                            }
                            // Empty cell this dome doesn't reach, or interior
                            // bulk (already solid) — nothing to do.
                            _ => {}
                        }
                    }
                    OffsetSign::Deflate => {
                        // Monotonic ERODE: the surface only moves inward — the
                        // union of erosions (most-eroded wins). A re-touched
                        // cell's distance only ever goes MORE positive.
                        match kind {
                            2 => {
                                let slot = slot_grid[i];
                                let prev = prev_dist(slot);
                                let nd = dist_vox.max(prev);
                                if nd > band {
                                    edits.push(LeafEdit { coord: c, op: LeafEditOp::Remove });
                                } else if nd > prev + 1e-4 {
                                    edits.push(LeafEdit {
                                        coord: c,
                                        op: LeafEditOp::SetDist { slot, normal: surf_n, dist: nd },
                                    });
                                }
                            }
                            1 => {
                                // Interior bulk first exposed by this erosion.
                                if dist_vox > band {
                                    edits.push(LeafEdit { coord: c, op: LeafEditOp::Empty });
                                } else if dist_vox >= -band {
                                    edits.push(LeafEdit {
                                        coord: c,
                                        op: LeafEditOp::Add {
                                            material: op.material,
                                            normal: surf_n,
                                            dist: dist_vox,
                                        },
                                    });
                                }
                                // Deep interior beyond the erosion stays interior.
                            }
                            // Empty stays empty — Deflate never adds material back.
                            _ => {}
                        }
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
            n_neighbor_calls: 0,
            t_total_ns: t_start.elapsed().as_nanos() as u64,
        },
    }
}

/// Flat-top cross-section profile for clay strips. Returns the target
/// thickness in cells at perpendicular distance `d` from the capsule axis.
/// Full `strength` across the entire radius — no shoulder taper for now,
/// because per-stamp shoulder falloff creates visible circular artifacts
/// along drag strokes. Shoulder taper will be reintroduced once the
/// kernel evaluates against the full stroke polyline instead of
/// individual stamp capsules.
#[inline]
pub fn clay_strip_profile(d: f32, radius: f32, strength: f32, _falloff: FalloffCurve) -> f32 {
    if d <= radius {
        strength
    } else {
        0.0
    }
}

/// Analytical normal for a clay strip cell. All cells get (0,1,0)
/// since the profile is flat across the entire radius (no shoulder).
#[inline]
fn clay_strip_normal(_horiz_dist: f32, _diff: Vec3, _op: &BrushOp) -> Vec3 {
    Vec3::Y
}

/// Clay strip deposit. A brushfire-shell kernel (Inflate/Deflate were
/// migrated to the SDF-offset [`compute_offset_edits`] path; ClayStrip still
/// uses brushfire) with a flat-top thickness profile: full `strength` cells
/// across the inner 75% of the brush radius, tapering to zero over a 25%
/// shoulder at the lateral edges.
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
                        op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(c, op) },
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
                            op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(c, op) },
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
                    op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(c, op) },
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

/// Depth (voxels, inside the brush surface) of the surface band the Raise
/// emitter stores analytic distance+normal on (when [`sculpt_band_on`]).
/// The legacy code collapsed the WHOLE brush interior to `SetInterior`
/// (no Hermite data), leaving only the one-voxel OUTSIDE rim with stored
/// distances — so a straddling cube on the dome flank had a single sparse
/// plane and the Manifold-DC mesher extrapolated + clamped to integer rows
/// (= terracing). Storing analytic dist+normal on the inside-surface band
/// gives the cube INSIDE-side Hermite data too; paired with the unchanged
/// outside rim, DC interpolates a true two-sided crossing. Cells deeper
/// than this stay `SetInterior` bulk. Within the brush footprint, so no
/// extra walk padding is needed.
const SCULPT_BAND_HALF_WIDTH: f32 = 1.5;

/// Sculpt RAISE surface representation. On (default): the inside-surface
/// band ([`SCULPT_BAND_HALF_WIDTH`]) stores the analytic brush
/// distance+normal as `Add` surface leaves (de-terraces the dome + feeds
/// dense analytic normals to the re-extract), pairing with the existing
/// outside rim for two-sided DC data. Off (`ARVX_SCULPT_BAND=0`): the
/// legacy path where the interior collapses to `SetInterior` with no
/// Hermite data → integer-row terracing. Read once.
pub fn sculpt_band_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ARVX_SCULPT_BAND").map(|v| v != "0").unwrap_or(true))
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
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(coord, op) } });
    }
}

/// Raise, inside-brush cell (`d = brush_sdf(center) ≤ 0`). `Empty` cells
/// become solid: within the near-surface band (`d ≥ −band`) they are
/// `Add` surface leaves carrying the analytic brush distance (negative,
/// inside) + normal, so the dome flank has inside-side Hermite data;
/// deeper cells collapse to `SetInterior` bulk. Solid/Interior cells stay
/// (Raise unions onto existing mass — it isn't paint). With the band off,
/// every inside `Empty` cell is `SetInterior` (legacy).
#[inline]
fn emit_raise_inside(
    edits: &mut Vec<LeafEdit>,
    coord: UVec3,
    state: CellState,
    d: f32,
    op: &BrushOp,
) {
    if !matches!(state, CellState::Empty) {
        return;
    }
    if sculpt_band_on() && d >= -SCULPT_BAND_HALF_WIDTH {
        let normal = brush_add_normal(coord, op);
        edits.push(LeafEdit {
            coord,
            op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(coord, op) },
        });
    } else {
        edits.push(LeafEdit { coord, op: LeafEditOp::SetInterior });
    }
}

/// Raise, outside-brush cell. Only `Empty` cells with an inside-brush
/// 6-neighbour that will flip to `Interior` (Empty pre-stamp) become
/// dome-surface `Add` cells — the OUTSIDE-side Hermite data (positive
/// dist + analytic normal) that pairs with [`emit_raise_inside`]'s
/// inside-band so a straddling cube has a two-sided crossing. The
/// neighbour gate (vs a plain distance band) is what keeps a Raise over
/// existing solid a no-op — no new mass, no spurious surface shell.
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
        edits.push(LeafEdit { coord, op: LeafEditOp::Add { material: op.material, normal, dist: brush_add_dist(coord, op) } });
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
/// Brush-SDF signed distance from the cell center to the new surface, in
/// voxel units, sign-matched to [`brush_add_normal`] (negative inside the
/// new solid). The QEF-Hermite companion to the Add normal.
///
/// `brush_sdf = axis_distance − radius` is negative inside the brush and
/// `|∇| = 1` (Euclidean), so for **Raise/Inflate/ClayStrip** (new solid =
/// inside the brush) it is already the new-surface SDF. For **Carve/Deflate**
/// the new solid is OUTSIDE the brush (the wall around the cavity), so the
/// SDF flips sign — exactly as the normal flips to point into the cavity.
/// `p_surf = center − dist·normal` then lands on the brush surface in both
/// cases.
fn brush_add_dist(coord: UVec3, op: &BrushOp) -> f32 {
    let cell_center = Vec3::new(
        coord.x as f32 + 0.5,
        coord.y as f32 + 0.5,
        coord.z as f32 + 0.5,
    );
    let sdf = brush_sdf(cell_center, op);
    match op.mode {
        BrushMode::Raise | BrushMode::Inflate | BrushMode::ClayStrip => sdf,
        BrushMode::Carve | BrushMode::Deflate => -sdf,
        // Smooth computes its own Add normals/positions; the brush SDF is
        // not the governing surface there, so leave it cell-centered.
        BrushMode::Smooth => 0.0,
    }
}

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
            if let LeafEditOp::Add { material, normal, dist } = edit.op {
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
                allocated_slots.push((slot, LeafEditAttrs { material, normal, dist }));
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

    // SDF-offset Inflate/Deflate's `SetDist` edits land here: the cell stays
    // Solid (no octree mutation, no slot alloc) — only its stored distance
    // (and refreshed normal) change. The scene-manager consumer drains
    // `redist_slots` into the LeafAttrPool via `set_dist` + `normal_oct`.
    let redist_slots: Vec<(u32, Vec3, f32)> = delta
        .edits
        .iter()
        .filter_map(|edit| {
            if let LeafEditOp::SetDist { slot, normal, dist } = edit.op {
                Some((slot, normal, dist))
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
        redist_slots,
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

    /// **Stage 6 — sculpt distance sign/magnitude.** `brush_add_dist` must
    /// be negative inside the NEW solid (the QEF/SDF convention) for BOTH
    /// brush polarities, with magnitude = the perpendicular distance from the
    /// cell center to the brush (sphere) surface. Raise's new solid is INSIDE
    /// the brush; Carve's is OUTSIDE (the cavity wall) — so the sign flips
    /// with the brush mode exactly as the Add normal does.
    #[test]
    fn brush_add_dist_sign_and_magnitude() {
        let mk = |mode| BrushOp {
            center: Vec3::new(8.0, 8.0, 8.0),
            segment_start: Vec3::new(8.0, 8.0, 8.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Constant,
            strength: 1.0,
            mode,
            material: 1,
        };
        // Raise dome cell INSIDE the sphere → dist = brush_sdf < 0.
        let d_raise = brush_add_dist(UVec3::new(8, 10, 8), &mk(BrushMode::Raise));
        let p_in = Vec3::new(8.5, 10.5, 8.5);
        let sdf_in = p_in.distance(Vec3::splat(8.0)) - 4.0; // < 0 inside
        assert!(sdf_in < 0.0);
        assert!(d_raise < 0.0, "raise dome cell must be inside the new solid");
        assert!((d_raise - sdf_in).abs() < 1e-4, "raise dist = brush_sdf");

        // Carve cavity-wall cell OUTSIDE the sphere → dist = −brush_sdf < 0.
        let d_carve = brush_add_dist(UVec3::new(8, 13, 8), &mk(BrushMode::Carve));
        let p_out = Vec3::new(8.5, 13.5, 8.5);
        let sdf_out = p_out.distance(Vec3::splat(8.0)) - 4.0; // > 0 outside
        assert!(sdf_out > 0.0);
        assert!(d_carve < 0.0, "carve wall cell must be inside the new solid");
        assert!((d_carve + sdf_out).abs() < 1e-4, "carve dist = −brush_sdf");
    }

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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        // Band rule (ARVX_SCULPT_BAND, default on): inside-brush EMPTY
        // within `SCULPT_BAND_HALF_WIDTH` of the surface → `Add` surface
        // leaf (analytic dist+normal); deeper inside-brush EMPTY →
        // SetInterior; outside-brush EMPTY with an inside-brush EMPTY
        // neighbour → `Add` (outside rim). A radius-0.4 brush at
        // (4.5,4.5,4.5) catches one inside-brush cell (4,4,4) at d=−0.4
        // (within the band → `Add`, no bulk) and its six face-neighbours
        // become outside-rim `Add`s — seven `Add`s, zero SetInterior.
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
        assert_eq!(delta.count_interior(), 0, "inside cell is within the band → Add, no SetInterior bulk");
        assert_eq!(delta.count_added(), 7, "one inside-band Add + six outside-rim Adds");
        assert_eq!(delta.count_removed(), 0);

        // Allocator hands out monotonically increasing ids.
        let mut next = 100u32;
        let applied = apply_delta(&mut t, &mut pool, &delta, || {
            let s = next;
            next += 1;
            s
        });
        assert_eq!(applied.allocated_slots.len(), 7);
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
        // Cell (4,4,4) is within the band → now a SURFACE leaf (Solid),
        // carrying the analytic brush dist/normal, not collapsed to bulk.
        let center_state = t.cell_state(UVec3::new(4, 4, 4), &pool);
        assert!(matches!(center_state, CellState::Solid(_)), "centre cell should be a surface leaf, got {:?}", center_state);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
        assert!(delta.is_empty());

        // And on the far side.
        let op2 = BrushOp { center: Vec3::splat(100.0), ..op };
        let delta2 = compute_brush_edits(&t, &pool, &[], &[], op2);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
            LeafEdit { coord: UVec3::new(0, 1, 0), op: LeafEditOp::Add { material: 5, normal: Vec3::Y, dist: 0.0 } },
            LeafEdit { coord: UVec3::new(1, 1, 0), op: LeafEditOp::Add { material: 5, normal: Vec3::Y, dist: 0.0 } },
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
            LeafEdit { coord: UVec3::new(1, 0, 0), op: LeafEditOp::Add { material: 7, normal: Vec3::Y, dist: 0.0 } },
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
            LeafEdit { coord: UVec3::new(2, 2, 2), op: LeafEditOp::Add { material: 1, normal: Vec3::Y, dist: 0.0 } },
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
        let mut next = 0u32;
        let _applied = apply_delta(&mut t, &mut pool, &delta, || { let s = next; next += 1; s });

        // No Add cell may float in empty space. Each must touch solid
        // mass — an INTERIOR (deep bulk) OR SOLID (a band/rim surface
        // leaf) face-neighbour. (Under the band rule the inside-surface
        // Adds border other band Adds rather than the Interior bulk, so
        // the old "INTERIOR neighbour" proxy is too strict; "connected to
        // mass" is the real anti-pinhole invariant.)
        let mut adds = 0;
        for edit in &delta.edits {
            if !matches!(edit.op, LeafEditOp::Add { .. }) {
                continue;
            }
            adds += 1;
            let c = edit.coord;
            let mut touches_mass = false;
            for d in [IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z] {
                let n = IVec3::new(c.x as i32, c.y as i32, c.z as i32) + d;
                if n.x < 0 || n.y < 0 || n.z < 0 || n.x >= 16 || n.y >= 16 || n.z >= 16 { continue; }
                let nu = UVec3::new(n.x as u32, n.y as u32, n.z as u32);
                if matches!(t.cell_state(nu, &pool), CellState::Interior | CellState::Solid(_)) {
                    touches_mass = true;
                    break;
                }
            }
            assert!(touches_mass, "Add cell {c} has no solid neighbour — would be a pinhole ghost");
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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

    // ── P1 Inflate / Deflate tests (SDF-offset kernel) ───────────────
    //
    // The retired brushfire kernel was exercised on bare-occupancy octrees
    // (no per-leaf distances). The SDF-offset kernel reconstructs the field
    // from surface LEAVES that carry a signed distance + outward normal — the
    // shape every real asset (terrain bake, imported `.arvx`) actually has —
    // so these tests build that: a flat solid ground whose top layer is
    // surface leaves (normal +Y, dist ≈ 0) over interior bulk.

    /// Flat solid ground at octree `depth`, filled `y ∈ [0, top]`: the top
    /// layer (`y == top`) is surface leaves with outward normal +Y and a
    /// near-zero signed distance; everything below is interior bulk. Returns
    /// the octree plus the per-slot leaf-attr + quantized-distance vectors
    /// (slot `i` ↔ `attrs[i]` / `dists[i]`), ready for `compute_brush_edits`.
    fn flat_ground(depth: u8, top: u32) -> (SparseOctree, Vec<LeafAttr>, Vec<i16>) {
        let n = 1u32 << depth;
        let mut t = SparseOctree::new(depth, 1.0);
        let mut attrs: Vec<LeafAttr> = Vec::new();
        let mut dists: Vec<i16> = Vec::new();
        let mut slot = 0u32;
        for z in 0..n {
            for x in 0..n {
                for y in 0..top {
                    t.insert_interior(UVec3::new(x, y, z));
                }
                t.insert(UVec3::new(x, top, z), slot);
                attrs.push(LeafAttr {
                    material_primary: 1,
                    material_secondary_blend: 0,
                    normal_oct: pack_oct(Vec3::Y),
                });
                dists.push(crate::LeafAttrPool::quantize_dist(0.0));
                slot += 1;
            }
        }
        (t, attrs, dists)
    }

    /// Max solid extent (highest `y` of any Add / SetInterior / surviving
    /// surface leaf) in column `(cx, cz)` after a delta — the dome height.
    fn dome_top_in_column(delta: &SculptDelta, cx: u32, cz: u32) -> Option<u32> {
        delta
            .edits
            .iter()
            .filter(|e| {
                e.coord.x == cx
                    && e.coord.z == cz
                    && matches!(e.op, LeafEditOp::Add { .. } | LeafEditOp::SetInterior)
            })
            .map(|e| e.coord.y)
            .max()
    }

    #[test]
    fn inflate_zero_strength_is_noop() {
        // strength = 0 → max_k = 0 → early bail, empty delta.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 0.0,
            mode: BrushMode::Inflate,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        assert!(delta.is_empty(), "zero-strength Inflate must be a no-op");
    }

    #[test]
    fn gentle_inflate_emits_setdist_only() {
        // A small Inflate that only re-distances existing surface leaves (the
        // offset is too small to flip any empty cell to solid) produces a delta
        // of PURELY SetDist edits. This is a real mutation (it moves the stored
        // surface), so any "did anything change?" guard MUST count SetDist — the
        // render consumer's early-return previously dropped it and the brush
        // silently did nothing. This pins the case so that regression can't
        // return unnoticed.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 6.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 0.5,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        assert!(!delta.is_empty(), "gentle Inflate must still produce edits");
        assert!(delta.count_set_dist() > 0, "expected SetDist re-distancing");
        assert_eq!(delta.count_added(), 0, "offset too small to add cells");
        assert_eq!(delta.count_interior(), 0);
        assert_eq!(delta.count_removed(), 0, "Inflate never removes");
        assert_eq!(delta.count_set_normal(), 0);
    }

    #[test]
    fn inflate_grows_dome_taller_at_center() {
        // Inflate on flat ground: adds material, removes nothing, and the
        // dome is taller under the brush center than at the rim.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 6.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        assert!(delta.count_added() > 0, "Inflate must add material");
        assert_eq!(delta.count_removed(), 0, "Inflate must not remove anything");
        // Both columns must be MEASURED (Add/SetInterior present), not defaulted
        // to the original surface height — x=36 (dx=4) still lifts empty cells
        // to solid, so it's a genuine rim, unlike x=39 which only re-distances.
        let center = dome_top_in_column(&delta, 32, 32)
            .expect("center column must grow a dome");
        let rim = dome_top_in_column(&delta, 36, 32)
            .expect("rim column must grow some dome");
        assert!(
            center > rim,
            "dome must be taller at center than rim: center={center} rim={rim}",
        );
        // Height is bounded by the surface (y=8) + the brush strength: the
        // offset field never lifts the surface by more than `strength` voxels.
        assert!(
            center <= 8 + 6 + 1,
            "dome top {center} must not exceed surface + strength (+1 slack)",
        );
        // Every Add carries the brush material and an inside-band distance.
        for e in &delta.edits {
            if let LeafEditOp::Add { material, dist, .. } = e.op {
                assert_eq!(material, 9);
                assert!(
                    dist <= 1e-4 && dist >= -SCULPT_BAND_HALF_WIDTH - 1e-3,
                    "Add band distance {dist} out of [-band, 0]",
                );
            }
        }
    }

    #[test]
    fn inflate_pinhole_free() {
        // Every Inflate-added surface cell must attach to solid material —
        // no floating ghost cells across a void. Walk Adds in row-major
        // order; each must have a pre-existing-solid or already-emitted
        // (Add / SetInterior) 6-neighbour.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 5.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        // Guard against a vacuous pass: the anchoring loop below is a no-op if
        // the kernel emits zero Adds, so assert a dense shell first (this setup
        // emits ~100+). Catches a zero-Add regression directly here.
        assert!(
            delta.count_added() >= 30,
            "expected a dense Inflate shell, got only {} Adds",
            delta.count_added(),
        );
        let solid_at = |c: UVec3| -> bool {
            !matches!(
                t.cell_state(c, &pool),
                CellState::Empty | CellState::OutOfBounds,
            )
        };
        let mut emitted_solid: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        for e in &delta.edits {
            if matches!(e.op, LeafEditOp::Add { .. } | LeafEditOp::SetInterior) {
                emitted_solid.insert(e.coord);
            }
        }
        for e in &delta.edits {
            if !matches!(e.op, LeafEditOp::Add { .. }) {
                continue;
            }
            let c = e.coord;
            let ci = IVec3::new(c.x as i32, c.y as i32, c.z as i32);
            let anchored = [IVec3::X, -IVec3::X, IVec3::Y, -IVec3::Y, IVec3::Z, -IVec3::Z]
                .iter()
                .any(|&d| {
                    let n = ci + d;
                    if n.x < 0 || n.y < 0 || n.z < 0 {
                        return false;
                    }
                    let nu = UVec3::new(n.x as u32, n.y as u32, n.z as u32);
                    solid_at(nu) || emitted_solid.contains(&nu)
                });
            assert!(anchored, "floating Inflate cell at {c}");
        }
    }

    #[test]
    fn deflate_zero_strength_is_noop() {
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 4.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 0.0,
            mode: BrushMode::Deflate,
            material: 1,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        assert!(delta.is_empty(), "zero-strength Deflate must be a no-op");
    }

    #[test]
    fn deflate_carves_pit_deeper_at_center() {
        // Deflate on flat ground: removes the original surface, exposes a
        // receded surface that dips deepest under the brush center.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 5.0,
            mode: BrushMode::Deflate,
            material: 7,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        assert!(delta.count_removed() > 0, "Deflate must erode the surface");
        // New receded surface = lowest Add per column; deeper (lower y) at center.
        let new_surface_y = |cx: u32, cz: u32| -> Option<u32> {
            delta
                .edits
                .iter()
                .filter(|e| {
                    e.coord.x == cx && e.coord.z == cz && matches!(e.op, LeafEditOp::Add { .. })
                })
                .map(|e| e.coord.y)
                .min()
        };
        let center = new_surface_y(32, 32);
        let rim = new_surface_y(39, 32);
        if let (Some(c), Some(r)) = (center, rim) {
            assert!(c < r, "pit must be deeper at center than rim: center_y={c} rim_y={r}");
        } else {
            assert!(center.is_some(), "Deflate must expose a new surface under the center");
        }
    }

    #[test]
    fn deflate_exposed_surface_normal_points_up() {
        // The receded Deflate surface faces the same way the original did
        // (up, +Y) — a uniform-ish lowering, not a sideways pit wall.
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 5.0,
            mode: BrushMode::Deflate,
            material: 0,
        };
        let delta = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        // Find an Add directly under the brush center (the pit floor).
        let floor = delta
            .edits
            .iter()
            .filter(|e| e.coord.x == 32 && e.coord.z == 32 && matches!(e.op, LeafEditOp::Add { .. }))
            .min_by_key(|e| e.coord.y)
            .expect("Deflate must expose a pit-floor surface under the center");
        if let LeafEditOp::Add { normal, .. } = floor.op {
            assert!(
                normal.y > 0.85,
                "pit-floor normal should point up (+Y), got {normal:?}",
            );
        }
    }

    #[test]
    fn inflate_far_from_surface_emits_nothing() {
        // A brush hovering in empty space far from any surface adds nothing:
        // no surface leaves seed the reconstruction, so every cell stays
        // unreached and the emit pass skips them.
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
        let delta = compute_brush_edits(&t, &pool, &[], &[], op);
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

    /// Highest solid `y` in column `(cx, cz)` of the octree (the surface
    /// height), or `None` if the column is empty up to `ymax`.
    fn column_solid_top(
        t: &SparseOctree,
        pool: &BrickPool,
        cx: u32,
        cz: u32,
        ymax: u32,
    ) -> Option<u32> {
        (0..=ymax).rev().find(|&y| {
            !matches!(
                t.cell_state(UVec3::new(cx, y, cz), pool),
                CellState::Empty | CellState::OutOfBounds,
            )
        })
    }

    /// First stamp of a stroke (empty touched set) must match the plain
    /// `compute_brush_edits` byte-for-byte — the stroke-capping seed
    /// exclusion only kicks in for already-edited cells.
    #[test]
    fn inflate_first_stamp_matches_no_stroke() {
        let (t, attrs, dists) = flat_ground(6, 8);
        let pool = fresh_pool();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 6.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let plain = compute_brush_edits(&t, &pool, &attrs, &dists, op);
        let in_stroke =
            compute_brush_edits_in_stroke(&t, &pool, &attrs, &dists, op, |_| false);
        assert_eq!(plain.edits.len(), in_stroke.edits.len());
        for (a, b) in plain.edits.iter().zip(in_stroke.edits.iter()) {
            assert_eq!(a.coord, b.coord);
            assert_eq!(
                std::mem::discriminant(&a.op),
                std::mem::discriminant(&b.op),
            );
        }
    }

    /// Holding the brush at one spot must converge to ONE offset layer, not
    /// compound. Stamp twice at the same position with the touched set; the
    /// surface height after stamp 2 must equal the height after stamp 1.
    #[test]
    fn inflate_hold_does_not_compound() {
        let (mut t, attrs, dists) = flat_ground(6, 8);
        let mut pool = fresh_pool();
        // Live mutable copies of the per-slot data so re-stamps read the
        // accumulated field (mirrors the scene-manager pool).
        let mut leaf = attrs.clone();
        let mut dvec = dists.clone();
        let op = BrushOp {
            center: Vec3::new(32.0, 8.5, 32.0),
            segment_start: Vec3::new(32.0, 8.5, 32.0),
            radius: 8.0,
            falloff_curve: FalloffCurve::Smooth,
            strength: 6.0,
            mode: BrushMode::Inflate,
            material: 9,
        };
        let mut touched: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        let mut next_slot = leaf.len() as u32;

        let apply_stamp = |t: &mut SparseOctree,
                               pool: &mut BrickPool,
                               leaf: &mut Vec<LeafAttr>,
                               dvec: &mut Vec<i16>,
                               touched: &mut std::collections::HashSet<UVec3>,
                               next_slot: &mut u32| {
            let delta = compute_brush_edits_in_stroke(
                t, pool, leaf, dvec, op, |c| touched.contains(&c),
            );
            for e in &delta.edits {
                touched.insert(e.coord);
            }
            let applied = apply_delta(t, pool, &delta, || {
                let s = *next_slot;
                *next_slot += 1;
                if s as usize >= leaf.len() {
                    leaf.resize(s as usize + 1, LeafAttr::default());
                    dvec.resize(s as usize + 1, 0);
                }
                s
            });
            for (slot, a) in &applied.allocated_slots {
                if *slot as usize >= leaf.len() {
                    leaf.resize(*slot as usize + 1, LeafAttr::default());
                    dvec.resize(*slot as usize + 1, 0);
                }
                leaf[*slot as usize] = a.to_leaf_attr();
                dvec[*slot as usize] = crate::LeafAttrPool::quantize_dist(a.dist);
            }
            for (slot, _n, d) in &applied.redist_slots {
                dvec[*slot as usize] = crate::LeafAttrPool::quantize_dist(*d);
            }
        };

        apply_stamp(&mut t, &mut pool, &mut leaf, &mut dvec, &mut touched, &mut next_slot);
        let top_after_1 = column_solid_top(&t, &pool, 32, 32, 20)
            .expect("stamp 1 must raise a dome");
        apply_stamp(&mut t, &mut pool, &mut leaf, &mut dvec, &mut touched, &mut next_slot);
        let top_after_2 = column_solid_top(&t, &pool, 32, 32, 20).unwrap();
        assert_eq!(
            top_after_1, top_after_2,
            "held Inflate compounded: dome grew from y={top_after_1} to y={top_after_2}",
        );
    }

    /// A drag (offset stamps) is a monotonic GROW: it never removes material,
    /// and both stamp centers end up raised — the union of stamp domes, with
    /// no scalloping between them.
    #[test]
    fn inflate_drag_is_monotonic_grow() {
        let (mut t, attrs, dists) = flat_ground(6, 8);
        let mut pool = fresh_pool();
        let mut leaf = attrs.clone();
        let mut dvec = dists.clone();
        let mut touched: std::collections::HashSet<UVec3> = std::collections::HashSet::new();
        let mut next_slot = leaf.len() as u32;
        let mut prev = Vec3::new(26.0, 8.5, 32.0);
        let centers = [
            Vec3::new(26.0, 8.5, 32.0),
            Vec3::new(32.0, 8.5, 32.0),
            Vec3::new(38.0, 8.5, 32.0),
        ];
        let mut total_removed = 0usize;
        for &ctr in &centers {
            let op = BrushOp {
                center: ctr,
                segment_start: prev,
                radius: 8.0,
                falloff_curve: FalloffCurve::Smooth,
                strength: 6.0,
                mode: BrushMode::Inflate,
                material: 9,
            };
            prev = ctr;
            let delta = compute_brush_edits_in_stroke(
                &t, &pool, &leaf, &dvec, op, |c| touched.contains(&c),
            );
            total_removed += delta.count_removed();
            for e in &delta.edits {
                touched.insert(e.coord);
            }
            let applied = apply_delta(&mut t, &mut pool, &delta, || {
                let s = next_slot;
                next_slot += 1;
                if s as usize >= leaf.len() {
                    leaf.resize(s as usize + 1, LeafAttr::default());
                    dvec.resize(s as usize + 1, 0);
                }
                s
            });
            for (slot, a) in &applied.allocated_slots {
                if *slot as usize >= leaf.len() {
                    leaf.resize(*slot as usize + 1, LeafAttr::default());
                    dvec.resize(*slot as usize + 1, 0);
                }
                leaf[*slot as usize] = a.to_leaf_attr();
                dvec[*slot as usize] = crate::LeafAttrPool::quantize_dist(a.dist);
            }
            for (slot, _n, d) in &applied.redist_slots {
                dvec[*slot as usize] = crate::LeafAttrPool::quantize_dist(*d);
            }
        }
        assert_eq!(total_removed, 0, "Inflate drag must never remove material");
        // Both ends of the drag are raised above the original surface (y=8).
        let left = column_solid_top(&t, &pool, 26, 32, 20).unwrap_or(8);
        let right = column_solid_top(&t, &pool, 38, 32, 20).unwrap_or(8);
        assert!(left > 8, "left stamp center must be raised, got y={left}");
        assert!(right > 8, "right stamp center must be raised, got y={right}");
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
        let fill = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::Add { material, normal, .. } = e.op {
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);

        let bump_removed = delta.edits.iter().any(|e| {
            e.coord == UVec3::new(8, 8, 9) && matches!(e.op, LeafEditOp::Remove)
        });
        assert!(bump_removed, "isolated bump must morph-shave");

        let wall = delta.edits.iter().find_map(|e| {
            if e.coord == UVec3::new(8, 8, 8) {
                if let LeafEditOp::Add { normal, material, .. } = e.op {
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
        let delta = compute_brush_edits(&t, &pool, &pool_data, &[], op);
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
