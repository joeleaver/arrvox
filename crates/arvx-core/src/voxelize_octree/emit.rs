//! Terminal-level emission: surface cells get gradient taps, baked into
//! `LeafAttr` slots, and wired into the octree as bricks (deep trees) or
//! single leaves (shallow trees with bricks disabled).

use glam::Vec3;

use crate::brick_pool::{BRICK_DIM, BrickPool};
use crate::leaf_attr::LeafAttr;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::SparseOctree;

use super::{BakeStats, BrickJob, LeafJob};

/// Gradient-normalized signed distance in voxel units for QEF-Hermite
/// meshing. `grad = (d_xp−d_xm, d_yp−d_ym, d_zp−d_zm)` is the voxelizer's
/// `eps = cell_size/2` 6-tap central difference, so
/// `grad.length() = 2·eps·|∇f| = cell_size·|∇f|` and `d_center /
/// grad.length()` is the perpendicular distance from the cell center to
/// the surface in voxel (grid) units — exactly the Hermite offset the
/// mesher consumes (`p_surf = center − d_vox·n`, same `n = grad.normalize()`
/// stored in the leaf). Guarded so a degenerate (locally-flat) gradient
/// yields 0; the leaf then meshes through its cell center (harmless).
#[inline]
pub(crate) fn grad_normalized_distance(d_center: f32, grad: Vec3) -> f32 {
    let len = grad.length();
    if len > 1e-6 {
        d_center / len
    } else {
        0.0
    }
}

/// Two-phase brick emission.
///
/// **Phase 1**: sample only `d_center` per cell (one sample per cell,
/// BRICK_DIM³ per brick). Classify each cell via 1-Lipschitz bounds:
///
/// * `d_center >  cell_size * sqrt(3)/2` → **EMPTY**. No corner of the
///   cell can be inside; leave as `BRICK_EMPTY`.
/// * `d_center < -cell_size * sqrt(3)/2` → **INTERIOR**. No corner of
///   the cell can be outside; set `BRICK_INTERIOR` (no leaf_attr, no
///   gradient, same render cost as EMPTY).
/// * otherwise → **SURFACE**. Queue a 6-tap gradient fetch for phase
///   2 and store the center sample's material for later.
///
/// **Phase 2**: dispatch 6 axis-aligned taps at `±eps` per surface
/// cell. Build the gradient normal, allocate a `LeafAttr`, write the
/// cell.
///
/// Previously this was a single 7-sample dispatch per cell that
/// fetched the gradient taps even for clearly-EMPTY or clearly-
/// INTERIOR cells. For solid objects that's 6-8× wasted GPU work —
/// the vast majority of cells sit well away from the surface and the
/// gradient would never be read. The 2-phase rework cuts a 20 m ramp
/// bake from ~200 M brick samples to ~25 M.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_bricks_batched<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    voxel_count: &mut u32,
    brick_ids: &mut Vec<u32>,
    brick_queue: &[BrickJob],
    brick_depth: u8,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    let t_start = std::time::Instant::now();
    let cell_size = base_voxel_size;
    let eps = cell_size * 0.5;
    // 1-Lipschitz threshold for "no point in the cell crosses zero"
    // using the cell-center sample alone. The cell's far corner sits
    // at `cell_size * sqrt(3)/2 ≈ 0.866 * cell_size` from the center,
    // so if `|d_center|` exceeds that, the surface is definitively
    // outside the cell.
    let lipschitz_threshold = cell_size * (3.0_f32.sqrt() * 0.5);
    let cells_per_brick = (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize;

    // ── Allocate brick IDs up-front so phase 2's surface queue can
    //    reference them by (brick_id, cx, cy, cz).
    let mut brick_slots: Vec<u32> = Vec::with_capacity(brick_queue.len());
    for _ in brick_queue {
        brick_slots.push(brick_pool.allocate()?);
    }
    // Mirror into the caller's `brick_ids` so deallocate knows what
    // we held. Done here so an early-return on phase 2 alloc failure
    // still leaves state deallocatable.
    brick_ids.extend_from_slice(&brick_slots);

    // ── Phase 1: d_center per cell. ──────────────────────────────
    let phase1_count = brick_queue.len() * cells_per_brick;
    let mut phase1_samples: Vec<Vec3> = Vec::with_capacity(phase1_count);
    for job in brick_queue {
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let cell_min = job.world_min
                        + Vec3::new(
                            cx as f32 * cell_size,
                            cy as f32 * cell_size,
                            cz as f32 * cell_size,
                        );
                    let cell_center = cell_min + Vec3::splat(cell_size * 0.5);
                    phase1_samples.push(cell_center);
                }
            }
        }
    }

    let t_phase1_prep = t_start.elapsed();
    let t_phase1_sdf = std::time::Instant::now();
    let phase1_results = sdf_fn(&phase1_samples);
    stats.sdf_bricks += t_phase1_sdf.elapsed();
    stats.brick_sample_total += phase1_count;
    debug_assert_eq!(phase1_results.len(), phase1_count);

    // ── Classify + queue surface cells. ──────────────────────────
    let t_classify = std::time::Instant::now();
    // Each surface entry records everything needed to populate its
    // cell after phase 2 reads back: the brick slot, the 3D cell
    // coord, and the phase-1 sample that carries material/color/blend.
    struct SurfaceCell {
        brick_slot: u32,
        cx: u32,
        cy: u32,
        cz: u32,
        d_center: f32,
        primary: u16,
        secondary: u16,
        blend: u8,
        color: u32,
    }
    let mut surface_cells: Vec<SurfaceCell> = Vec::new();
    for (brick_idx, _job) in brick_queue.iter().enumerate() {
        let brick_slot = brick_slots[brick_idx];
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                    let flat = brick_idx * cells_per_brick + cell_idx;
                    let (d_center, primary, secondary, blend, color) = phase1_results[flat];
                    if d_center > lipschitz_threshold {
                        // Fully outside. Default brick cell is already
                        // BRICK_EMPTY, nothing to write.
                        continue;
                    }
                    if d_center < -lipschitz_threshold {
                        // Fully inside. Sentinel → no leaf_attr, no
                        // gradient fetch, same cost as EMPTY at march
                        // time.
                        brick_pool.set_cell(
                            brick_slot, cx, cy, cz,
                            crate::brick_pool::BRICK_INTERIOR,
                        );
                        continue;
                    }
                    // Surface cell — defer to phase 2.
                    surface_cells.push(SurfaceCell {
                        brick_slot, cx, cy, cz,
                        d_center, primary, secondary, blend, color,
                    });
                }
            }
        }
    }
    stats.brick_cpu += t_classify.elapsed();

    // ── Phase 2: 6 gradient taps per surface cell. ───────────────
    if !surface_cells.is_empty() {
        let t_phase2_prep = std::time::Instant::now();
        let phase2_count = surface_cells.len() * 6;
        let mut phase2_samples: Vec<Vec3> = Vec::with_capacity(phase2_count);
        for sc in &surface_cells {
            let cell_min = Vec3::new(
                sc.cx as f32 * cell_size,
                sc.cy as f32 * cell_size,
                sc.cz as f32 * cell_size,
            );
            // Reconstruct brick world_min from the brick's first cell
            // in phase1. Easier: keep world_min on the SurfaceCell.
            // Avoid by iterating brick_queue too. Simpler: we stored
            // the cell-local offset; grab world_min from brick_queue.
            let _ = cell_min;
        }
        // Build phase 2 sample list. Re-derive world positions from
        // brick_queue; cheaper than threading world_min through
        // SurfaceCell for each of the millions of surface cells.
        phase2_samples.clear();
        // Index surface cells by brick so we can reuse each brick's
        // world_min. SurfaceCell already carries (brick_slot, cx..cz)
        // — brick index is `brick_slot`'s position in `brick_slots`.
        // We stored insertion order so `brick_slot` is unique per
        // brick_idx. Build reverse lookup once.
        let mut slot_to_idx = std::collections::HashMap::with_capacity(brick_slots.len());
        for (i, &id) in brick_slots.iter().enumerate() {
            slot_to_idx.insert(id, i);
        }
        for sc in &surface_cells {
            let brick_idx = slot_to_idx[&sc.brick_slot];
            let job = &brick_queue[brick_idx];
            let cell_min = job.world_min
                + Vec3::new(
                    sc.cx as f32 * cell_size,
                    sc.cy as f32 * cell_size,
                    sc.cz as f32 * cell_size,
                );
            let cell_center = cell_min + Vec3::splat(cell_size * 0.5);
            phase2_samples.push(cell_center + Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center - Vec3::new(eps, 0.0, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center - Vec3::new(0.0, eps, 0.0));
            phase2_samples.push(cell_center + Vec3::new(0.0, 0.0, eps));
            phase2_samples.push(cell_center - Vec3::new(0.0, 0.0, eps));
        }
        let _ = t_phase2_prep;

        let t_phase2_sdf = std::time::Instant::now();
        let phase2_results = sdf_fn(&phase2_samples);
        stats.sdf_bricks += t_phase2_sdf.elapsed();
        stats.brick_sample_total += phase2_count;
        debug_assert_eq!(phase2_results.len(), phase2_count);

        // ── Populate surface cells from phase 2 readback. ────────
        let t_populate = std::time::Instant::now();
        for (i, sc) in surface_cells.iter().enumerate() {
            let base = i * 6;
            let d_xp = phase2_results[base    ].0;
            let d_xm = phase2_results[base + 1].0;
            let d_yp = phase2_results[base + 2].0;
            let d_ym = phase2_results[base + 3].0;
            let d_zp = phase2_results[base + 4].0;
            let d_zm = phase2_results[base + 5].0;
            // Second-chance INTERIOR / EMPTY checks with the tighter
            // sample set. Occasionally the center falls inside the
            // Lipschitz band but all 6 face-taps land on one side —
            // cheaper to reclassify than emit a leaf_attr that never
            // contributes a visible surface.
            let max_tap = d_xp.max(d_xm).max(d_yp).max(d_ym).max(d_zp).max(d_zm);
            let min_tap = d_xp.min(d_xm).min(d_yp).min(d_ym).min(d_zp).min(d_zm);
            if min_tap > 0.0 && sc.d_center > 0.0 {
                // Stays EMPTY.
                continue;
            }
            if max_tap < 0.0 && sc.d_center < 0.0 {
                brick_pool.set_cell(
                    sc.brick_slot, sc.cx, sc.cy, sc.cz,
                    crate::brick_pool::BRICK_INTERIOR,
                );
                continue;
            }

            let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
            let normal = if grad.length_squared() > 1e-12 {
                grad.normalize()
            } else {
                Vec3::Y
            };
            let attr = LeafAttr::new_blended(normal, sc.primary, sc.secondary, sc.blend);
            // No dedup — every cell gets its own slot so paint and
            // cursor have per-cell identity to work with. Bump-only
            // allocate keeps the asset's pool range contiguous, which
            // `release_asset` relies on for a single deallocate_range
            // to free everything.
            let leaf_attr_id = leaf_attr_pool.allocate_contiguous_bump(1)?;
            *leaf_attr_pool.get_mut(leaf_attr_id) = attr;
            // Per-leaf signed distance for QEF-Hermite meshing (voxel units;
            // same `grad` whose normalization is the stored normal).
            leaf_attr_pool.set_dist(leaf_attr_id, grad_normalized_distance(sc.d_center, grad));
            if sc.color != 0 {
                leaf_attr_pool.set_color(leaf_attr_id, sc.color);
            }
            brick_pool.set_cell(sc.brick_slot, sc.cx, sc.cy, sc.cz, leaf_attr_id);
            *voxel_count += 1;
        }
        stats.brick_cpu += t_populate.elapsed();
    }

    // ── Wire each allocated brick into the octree. ───────────────
    for (brick_idx, job) in brick_queue.iter().enumerate() {
        octree.set_at_level(
            job.coord,
            brick_depth,
            crate::sparse_octree::make_brick(brick_slots[brick_idx]),
        );
    }

    stats.brick_cpu += t_phase1_prep;
    Some(())
}

/// Emit finest-level single-cell leaves for shallow trees (depth ≤
/// BRICK_LEVELS, where bricks are disabled). Same 7-sample layout
/// per cell as the brick path.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_leaves_batched<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    voxel_count: &mut u32,
    leaf_queue: &[LeafJob],
    max_depth: u8,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
{
    let t_start = std::time::Instant::now();
    let eps = base_voxel_size * 0.5;
    let samples_per_leaf = 7usize;
    let total_samples = leaf_queue.len() * samples_per_leaf;
    stats.brick_sample_total += total_samples;

    let mut samples: Vec<Vec3> = Vec::with_capacity(total_samples);
    for job in leaf_queue {
        let voxel_center = job.world_min + Vec3::splat(base_voxel_size * 0.5);
        super::push_cell_samples(&mut samples, voxel_center, eps);
    }
    let t_prep = t_start.elapsed();

    let t_sdf_start = std::time::Instant::now();
    let results = sdf_fn(&samples);
    stats.sdf_bricks += t_sdf_start.elapsed();
    debug_assert_eq!(results.len(), total_samples);

    let t_cpu_start = std::time::Instant::now();
    for (leaf_idx, job) in leaf_queue.iter().enumerate() {
        let cell_base = leaf_idx * samples_per_leaf;
        let (d_center, primary, secondary, blend, color) = results[cell_base];
        if d_center > 0.0 {
            // Center is outside — this corner of a Mixed region is
            // not itself solid. Leave it EMPTY.
            continue;
        }
        let d_xp = results[cell_base + 1].0;
        let d_xm = results[cell_base + 2].0;
        let d_yp = results[cell_base + 3].0;
        let d_ym = results[cell_base + 4].0;
        let d_zp = results[cell_base + 5].0;
        let d_zm = results[cell_base + 6].0;
        let grad = Vec3::new(d_xp - d_xm, d_yp - d_ym, d_zp - d_zm);
        let normal = if grad.length_squared() > 1e-12 {
            grad.normalize()
        } else {
            Vec3::Y
        };
        let attr = LeafAttr::new_blended(normal, primary, secondary, blend);
        // No dedup — see emit_bricks_batched for the reasoning. Plain
        // LEAF emission only fires for shallow trees (depth ≤ BRICK_LEVELS),
        // so this path is rare in practice.
        let leaf_attr_id = leaf_attr_pool.allocate()?;
        *leaf_attr_pool.get_mut(leaf_attr_id) = attr;
        // Per-leaf signed distance for QEF-Hermite meshing (voxel units).
        leaf_attr_pool.set_dist(leaf_attr_id, grad_normalized_distance(d_center, grad));
        if color != 0 {
            leaf_attr_pool.set_color(leaf_attr_id, color);
        }
        octree.set_at_level(
            job.coord,
            max_depth,
            crate::sparse_octree::make_leaf(leaf_attr_id),
        );
        *voxel_count += 1;
    }

    stats.brick_cpu += t_prep + t_cpu_start.elapsed();
    Some(())
}

#[cfg(test)]
mod tests {
    use super::grad_normalized_distance;
    use glam::Vec3;

    /// The stored distance is the true PERPENDICULAR distance from the
    /// cell center to the surface, in VOXEL units — even for a non-unit
    /// gradient field (the terrain vertical-gap SDF `f = y − m·x`, whose
    /// `|∇f| = √(m²+1) ≠ 1`). Pins both the gradient normalization and the
    /// voxel-unit conversion against the 6-tap central-difference the
    /// voxelizer uses (`grad.length() = 2·eps·|∇f| = cell_size·|∇f|`).
    #[test]
    fn distance_is_perpendicular_in_voxel_units() {
        let m = 0.5f32;
        let eps = 0.5f32; // cell_size = 2·eps = 1
        let cell_size = 2.0 * eps;
        let f = |p: Vec3| p.y - m * p.x;
        for &center in &[
            Vec3::new(1.0, 0.7, 0.3),
            Vec3::new(-2.0, 0.1, 4.0),
            Vec3::new(0.0, -0.4, 0.0),
        ] {
            let d_center = f(center);
            let grad = Vec3::new(
                f(center + Vec3::new(eps, 0.0, 0.0)) - f(center - Vec3::new(eps, 0.0, 0.0)),
                f(center + Vec3::new(0.0, eps, 0.0)) - f(center - Vec3::new(0.0, eps, 0.0)),
                f(center + Vec3::new(0.0, 0.0, eps)) - f(center - Vec3::new(0.0, 0.0, eps)),
            );
            let d_vox = grad_normalized_distance(d_center, grad);
            // Analytic perpendicular distance to the plane, in voxel units.
            let perp_vox = (d_center / (m * m + 1.0).sqrt()) / cell_size;
            assert!(
                (d_vox - perp_vox).abs() < 1e-5,
                "center={center:?}: d_vox {d_vox} vs analytic {perp_vox}"
            );
        }
    }

    /// 45° plane (unit-gradient SDF): at cell_size = 1 the voxel-unit
    /// perpendicular distance equals `d_center` exactly.
    #[test]
    fn distance_unit_gradient_equals_d_center() {
        let eps = 0.5f32;
        let f = |p: Vec3| (p.y - p.x) / 2.0_f32.sqrt(); // |∇f| = 1
        let center = Vec3::new(0.3, 0.9, 0.0);
        let d_center = f(center);
        let grad = Vec3::new(
            f(center + Vec3::new(eps, 0.0, 0.0)) - f(center - Vec3::new(eps, 0.0, 0.0)),
            f(center + Vec3::new(0.0, eps, 0.0)) - f(center - Vec3::new(0.0, eps, 0.0)),
            f(center + Vec3::new(0.0, 0.0, eps)) - f(center - Vec3::new(0.0, 0.0, eps)),
        );
        assert!((grad_normalized_distance(d_center, grad) - d_center).abs() < 1e-5);
    }

    /// Degenerate (locally-flat) gradient → 0, not NaN/inf.
    #[test]
    fn distance_degenerate_gradient_is_zero() {
        assert_eq!(grad_normalized_distance(0.5, Vec3::ZERO), 0.0);
        assert_eq!(grad_normalized_distance(-3.0, Vec3::ZERO), 0.0);
    }
}
