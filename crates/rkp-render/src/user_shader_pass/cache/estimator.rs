//! Per-region pool size estimator: tile-geometry + paint-density → pool extents.

use super::{
    BRICK_BUCKET_MAX, BRICK_BUCKET_MIN, BRICK_CELLS, FILL_TASK_BUCKET_MAX,
    FILL_TASK_BUCKET_MIN, LEAF_ATTR_BUCKET_MAX, LEAF_ATTR_BUCKET_MIN, OCTREE_BUCKET_MAX,
    OCTREE_BUCKET_MIN, ShaderRegionRequest,
};

// ============================================================
// Pool-size estimator
// ============================================================

/// Per-region pool size estimate driving extent allocation. The
/// bucket allocator rounds up to the next bucket, so over-estimation
/// is cheap; under-estimation drops bricks and leaves visual holes.
#[derive(Debug, Clone, Copy)]
pub struct PoolEstimate {
    pub octree: u32,
    pub bricks: u32,
    pub leaf_attrs: u32,
    pub fill_tasks: u32,
}

/// Estimate per-region pool needs from TILE GEOMETRY.
///
/// The BFS classifier descends every cell within `region_thickness`
/// of the host surface, regardless of paint material. So `fill_tasks`
/// and `octree` counts scale with **tile dimensions × proximity-band
/// fraction**, NOT painted-cell count.
///
/// `bricks` and `leaf_attrs` ARE paint-driven: V12 deferred allocation
/// only consumes a brick slot when the user shader emits at least one
/// occupied cell. We use `painted_leaf_count` as a paint-density
/// proxy and clamp at the geometric upper bound.
///
/// Inputs all come from `ShaderRegionRequest` — `aabb_min/max`,
/// `cell_size`, `max_depth`, `region_thickness`, `painted_leaf_count`.
pub fn estimate_region_pool(request: &ShaderRegionRequest) -> PoolEstimate {
    // Brick-parent cells per axis at depth `max_depth`. Each
    // brick-parent spans `cell_size * BRICK_DIM = cell_size * 4`.
    let extent = (request.aabb_max[0] - request.aabb_min[0]).max(1e-6);
    let bp_cell = (request.cell_size * 4.0).max(1e-6);
    let bp_per_axis = ((extent / bp_cell).ceil() as u32).max(1);
    let bp_total = bp_per_axis.saturating_mul(bp_per_axis).saturating_mul(bp_per_axis);

    // Proximity-band fraction. With band B + half-cell-diag headroom
    // the gate keeps cells whose Lipschitz lower bound puts them
    // within ±(B + diag) of the surface. For a roughly-flat host
    // surface this is approximately
    //   fraction ≈ min(1, 2 * (B + diag) / extent)
    // Round generously up — over-estimating fill tasks is cheap.
    let band = request.region_thickness;
    let bp_diag_half = bp_cell * 0.866_025_4; // sqrt(3)/2
    let band_thickness = band + bp_diag_half;
    let band_fraction = if band > 0.0 {
        ((2.0 * band_thickness / extent).min(1.0)).max(0.5)
    } else {
        // No proximity gate → every cell is MIXED.
        1.0
    };
    // Estimates are clamped at the corresponding bucket-max so the
    // allocator can always satisfy the request. A region that
    // legitimately needs more than the max bucket falls back to the
    // GPU-side overflow counters (graceful degradation: the relevant
    // pool's overflow counter increments and individual bricks /
    // cells / branches drop to OCTREE_EMPTY).
    let fill_tasks = ((bp_total as f32 * band_fraction).ceil() as u32)
        .max(FILL_TASK_BUCKET_MIN)
        .min(FILL_TASK_BUCKET_MAX);

    // Octree allocations: sum across levels of (MIXED-cells × 8).
    // Conservative: 1.2 × fill_tasks (deepest level dominates) +
    // a constant for the spine.
    let depth_overhead = (request.max_depth.max(1) + 1) * 8;
    let octree = ((fill_tasks as u64 * 12 / 10) as u32)
        .saturating_add(depth_overhead)
        .max(OCTREE_BUCKET_MIN)
        .min(OCTREE_BUCKET_MAX);

    // Bricks: paint-driven via V12 deferred allocation. For grass-style
    // shaders each painted host cell projects to several brick-parent
    // cells (vertical extent of blade × thinness × cluster density).
    // Multiplier 12 sizes per-region brick blocks at bucket 8192 for
    // typical 1 m grass tiles, leaving 3 M / 8192 = 366 typical
    // regions fitting globally. Higher multiplier reduces per-region
    // overflow but eats more of the global pool — at 1 GB binding
    // limit, that means whole-tile drops once paint coverage grows.
    // Per-region overflow degrades gracefully (a few missing bricks
    // per tile, visible as small holes); whole-tile drops are much
    // worse visually.
    let painted = request.painted_leaf_count.max(BRICK_BUCKET_MIN);
    let bricks = painted
        .saturating_mul(12)
        .min(fill_tasks)
        .max(BRICK_BUCKET_MIN)
        .min(BRICK_BUCKET_MAX);

    // Leaf-attrs: each emitting brick has up to ~BRICK_CELLS / 2 = 32
    // occupied cells for grass-density shaders. Higher-density shaders
    // (full solids) overflow gracefully via the overflow counter.
    let leaf_attrs = bricks
        .saturating_mul(BRICK_CELLS / 2)
        .max(LEAF_ATTR_BUCKET_MIN)
        .min(LEAF_ATTR_BUCKET_MAX);

    PoolEstimate {
        octree,
        bricks,
        leaf_attrs,
        fill_tasks,
    }
}
