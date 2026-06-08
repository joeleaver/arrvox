//! Level-by-level BFS octree classification driver.
//!
//! Per level: build a sample list across every active node, dispatch
//! one `sdf_fn` call, classify each node, queue Mixed terminal nodes
//! for emission, recurse the rest. Final two batched dispatches emit
//! bricks and finest-level leaves.

use glam::{UVec3, Vec3};

use crate::brick_pool::BrickPool;
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::SparseOctree;

use super::emit::{emit_bricks_batched, emit_leaves_batched};
use super::{
    BakeStats, BrickJob, LeafJob, RegionClass, classify_from_samples, push_classify_positions,
};

/// BFS octree classification + terminal-level emission.
///
/// Iterates from level 0 down to the terminal level (brick_depth if
/// bricks are enabled, else max_depth). At each level, batches the
/// classify samples for every active node into one `sdf_fn` call.
/// Mixed nodes feed the next level (as 8 children) or are queued for
/// terminal-level geometry emission. After the BFS loop, two more
/// batched calls emit bricks (all in one dispatch) and finest-level
/// leaves (all in one dispatch).
#[allow(clippy::too_many_arguments)]
pub(super) fn subdivide_bfs<F>(
    sdf_fn: &mut F,
    octree: &mut SparseOctree,
    leaf_attr_pool: &mut LeafAttrPool,
    brick_pool: &mut BrickPool,
    voxel_count: &mut u32,
    brick_ids: &mut Vec<u32>,
    grid_origin: Vec3,
    max_depth: u8,
    brick_depth: Option<u8>,
    base_voxel_size: f32,
    stats: &mut BakeStats,
) -> Option<()>
where
    F: FnMut(&[Vec3]) -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)>,
{
    // The octree's root spans `2^max_depth * base_voxel_size` on a side.
    let root_extent = (1u64 << max_depth) as f32 * base_voxel_size;

    // Active set for the current level. Holds coord in finest-level
    // units — multiplying by `base_voxel_size` and adding
    // `grid_origin` gives the node's world_min. Size: octree integer
    // coords fit in u32 because `depth <= 30` in practice.
    let mut active: Vec<UVec3> = vec![UVec3::ZERO];

    // Queues populated at the terminal level during BFS, drained in a
    // single batched pass after classification finishes.
    let mut brick_queue: Vec<BrickJob> = Vec::new();
    let mut leaf_queue: Vec<LeafJob> = Vec::new();

    // Classify level-by-level. We always visit level 0 (root) even if
    // the tree is trivial; beyond that, the loop body early-exits
    // when `active` becomes empty.
    let terminal_level = brick_depth.unwrap_or(max_depth);
    for level in 0..=terminal_level {
        if active.is_empty() {
            break;
        }

        // Extent of a node at this level, in world units.
        let level_extent = (1u64 << (max_depth - level)) as f32 * base_voxel_size;

        // Generate 9 classify samples per node. The layout assumes
        // `classify_from_samples`'s corner + center order.
        let t_level_cpu_start = std::time::Instant::now();
        let mut samples: Vec<Vec3> = Vec::with_capacity(active.len() * 9);
        for &coord in &active {
            let world_min =
                grid_origin + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                    * base_voxel_size;
            push_classify_positions(&mut samples, world_min, level_extent);
        }
        stats.classify_cpu += t_level_cpu_start.elapsed();

        let t_sdf_start = std::time::Instant::now();
        let results = sdf_fn(&samples);
        stats.sdf_classify_total += t_sdf_start.elapsed();
        stats.classify_dispatches += 1;
        stats.classify_samples += samples.len() as u64;
        debug_assert_eq!(results.len(), samples.len());

        let t_process_start = std::time::Instant::now();
        // Process each node's classification. Mixed nodes either
        // recurse (schedule 8 children for the next level) or are
        // queued for terminal-level geometry emission.
        let mut next_active: Vec<UVec3> = Vec::new();
        let child_voxels = if level < max_depth {
            1u32 << (max_depth - level - 1)
        } else {
            0
        };

        for (i, &coord) in active.iter().enumerate() {
            let slice = &results[i * 9..i * 9 + 9];
            let class = classify_from_samples(slice, level_extent);

            match class {
                RegionClass::Empty => {
                    // Default octree state is EMPTY — no write needed.
                }
                RegionClass::Interior => {
                    octree.set_at_level(coord, level, crate::sparse_octree::INTERIOR_NODE);
                }
                RegionClass::Mixed => {
                    if brick_depth == Some(level) {
                        // Terminal: emit a brick for this node after the
                        // BFS classification loop.
                        let world_min = grid_origin
                            + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                                * base_voxel_size;
                        brick_queue.push(BrickJob { coord, world_min });
                    } else if level == max_depth {
                        // Terminal: emit a finest-level leaf after the
                        // BFS classification loop. Only fires for
                        // shallow trees (depth ≤ BRICK_LEVELS) where
                        // bricking is disabled.
                        let world_min = grid_origin
                            + Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
                                * base_voxel_size;
                        leaf_queue.push(LeafJob { coord, world_min });
                    } else {
                        // Descend: schedule 8 children for the next
                        // level. octant xyz-minor ordering matches the
                        // old recursive path.
                        for octant in 0u32..8 {
                            let dx = octant & 1;
                            let dy = (octant >> 1) & 1;
                            let dz = (octant >> 2) & 1;
                            next_active.push(UVec3::new(
                                coord.x + dx * child_voxels,
                                coord.y + dy * child_voxels,
                                coord.z + dz * child_voxels,
                            ));
                        }
                    }
                }
            }
        }

        active = next_active;
        stats.classify_cpu += t_process_start.elapsed();
    }

    // ── Terminal-level geometry: bricks ──
    if !brick_queue.is_empty() {
        emit_bricks_batched(
            &mut *sdf_fn,
            octree,
            leaf_attr_pool,
            brick_pool,
            voxel_count,
            brick_ids,
            &brick_queue,
            brick_depth.expect("brick_queue non-empty ⇒ brick_depth set"),
            base_voxel_size,
            stats,
        )?;
    }

    // ── Terminal-level geometry: finest-level leaves (shallow tree) ──
    if !leaf_queue.is_empty() {
        emit_leaves_batched(
            &mut *sdf_fn,
            octree,
            leaf_attr_pool,
            voxel_count,
            &leaf_queue,
            max_depth,
            base_voxel_size,
            stats,
        )?;
    }

    let _ = root_extent; // kept for future diagnostics / overflow guard
    Some(())
}
