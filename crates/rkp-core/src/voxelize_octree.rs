//! Octree-based opacity voxelization with adaptive subdivision.
//!
//! Top-down recursive: at each octree node, sample the opacity function to decide
//! whether to subdivide (mixed content), mark as EMPTY (all below threshold),
//! mark as INTERIOR (all above threshold), or allocate a voxel at max depth.
//!
//! Produces a [`SparseOctree`] with variable-depth leaves — uniform regions
//! terminate early, detail concentrates where opacity varies. Each leaf is a
//! single voxel (no bricks).

use glam::{UVec3, Vec3};
use rkf_core::Aabb;

use crate::sparse_octree::SparseOctree;
use crate::voxel_pool::VoxelPool;
use crate::SplatVoxel;

/// Threshold below which a sample is considered empty.
const EMPTY_THRESHOLD: f32 = 0.001;

/// Threshold above which a sample is considered fully opaque.
const OPAQUE_THRESHOLD: f32 = 0.999;

/// Voxelize an opacity function into a sparse octree with adaptive subdivision.
///
/// `opacity_fn`: returns `(opacity, material_id)` at a world-space position.
/// `aabb`: world-space bounding box of the object.
/// `base_voxel_size`: voxel size at the finest level.
///
/// The octree depth is computed automatically from the AABB and voxel size.
/// Each leaf is a single voxel in the pool.
///
/// Returns `Some((octree, voxel_count, grid_origin))` on success, `None` if pool
/// allocation fails.
pub fn voxelize_opacity_octree<F>(
    opacity_fn: F,
    aabb: &Aabb,
    base_voxel_size: f32,
    pool: &mut VoxelPool,
) -> Option<(SparseOctree, u32, Vec3)>
where
    F: Fn(Vec3) -> (f32, u16),
{
    let aabb_size = aabb.max - aabb.min;
    let max_dim = aabb_size.x.max(aabb_size.y).max(aabb_size.z);

    // Compute depth: smallest power of 2 that covers the AABB in voxels.
    let voxels_needed = (max_dim / base_voxel_size).ceil().max(1.0) as u32;
    let depth = if voxels_needed <= 1 {
        1
    } else {
        (32 - (voxels_needed - 1).leading_zeros()) as u8
    };

    let mut octree = SparseOctree::new(depth, base_voxel_size);
    let mut voxel_count = 0u32;

    // Center the octree on the AABB.
    let extent = octree.extent_world();
    let aabb_center = (aabb.min + aabb.max) * 0.5;
    let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

    subdivide_node(
        &opacity_fn,
        &mut octree,
        pool,
        &mut voxel_count,
        UVec3::ZERO,
        0,
        depth,
        grid_origin,
        extent,
        base_voxel_size,
    )?;

    Some((octree, voxel_count, grid_origin))
}

/// Recursive subdivision.
///
/// `coord`: lower-corner voxel coordinate of this node's region.
/// `level`: current depth (0 = root).
/// `max_depth`: finest level.
/// `world_min`: world-space minimum corner of this node's region.
/// `node_extent`: world-space extent of this node (one axis).
fn subdivide_node<F>(
    opacity_fn: &F,
    octree: &mut SparseOctree,
    pool: &mut VoxelPool,
    voxel_count: &mut u32,
    coord: UVec3,
    level: u8,
    max_depth: u8,
    world_min: Vec3,
    node_extent: f32,
    base_voxel_size: f32,
) -> Option<()>
where
    F: Fn(Vec3) -> (f32, u16),
{
    // Sample the opacity function at 9 points (8 corners + center) to classify.
    let classification = classify_region(opacity_fn, world_min, node_extent);

    match classification {
        RegionClass::Empty => {
            // Already EMPTY by default — nothing to do.
        }
        RegionClass::Interior => {
            // Set the entire subtree at this level to INTERIOR.
            octree.set_at_level(coord, level, crate::sparse_octree::INTERIOR_NODE);
        }
        RegionClass::Mixed => {
            if level == max_depth {
                // At finest level — this IS a single voxel. Sample at its center.
                let voxel_center = world_min + Vec3::splat(base_voxel_size * 0.5);
                let (opacity, material_id) = opacity_fn(voxel_center);
                let sv = SplatVoxel::new(opacity.clamp(0.0, 1.0), material_id);
                let slot = pool.allocate()?;
                *pool.get_mut(slot) = sv;
                octree.set_at_level(coord, level, crate::sparse_octree::make_leaf(slot));
                *voxel_count += 1;
            } else {
                // Subdivide into 8 children.
                let half = node_extent * 0.5;
                let child_voxels = 1u32 << (max_depth - level - 1);

                for octant in 0u32..8 {
                    let dx = octant & 1;
                    let dy = (octant >> 1) & 1;
                    let dz = (octant >> 2) & 1;

                    let child_min = world_min
                        + Vec3::new(dx as f32 * half, dy as f32 * half, dz as f32 * half);
                    let child_coord = UVec3::new(
                        coord.x + dx * child_voxels,
                        coord.y + dy * child_voxels,
                        coord.z + dz * child_voxels,
                    );

                    subdivide_node(
                        opacity_fn,
                        octree,
                        pool,
                        voxel_count,
                        child_coord,
                        level + 1,
                        max_depth,
                        child_min,
                        half,
                        base_voxel_size,
                    )?;
                }
            }
        }
    }

    Some(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RegionClass {
    Empty,
    Interior,
    Mixed,
}

/// Classify a cubic region by sampling opacity at 9 points (8 corners + center).
fn classify_region<F>(opacity_fn: &F, world_min: Vec3, extent: f32) -> RegionClass
where
    F: Fn(Vec3) -> (f32, u16),
{
    let mut all_empty = true;
    let mut all_opaque = true;

    // Sample 8 corners.
    for cz in 0..2u32 {
        for cy in 0..2u32 {
            for cx in 0..2u32 {
                let pos = world_min
                    + Vec3::new(
                        cx as f32 * extent,
                        cy as f32 * extent,
                        cz as f32 * extent,
                    );
                let (opacity, _) = opacity_fn(pos);
                if opacity > EMPTY_THRESHOLD {
                    all_empty = false;
                }
                if opacity < OPAQUE_THRESHOLD {
                    all_opaque = false;
                }
            }
        }
    }

    // Sample center.
    let center = world_min + Vec3::splat(extent * 0.5);
    let (center_opacity, _) = opacity_fn(center);
    if center_opacity > EMPTY_THRESHOLD {
        all_empty = false;
    }
    if center_opacity < OPAQUE_THRESHOLD {
        all_opaque = false;
    }

    if all_empty {
        RegionClass::Empty
    } else if all_opaque {
        RegionClass::Interior
    } else {
        RegionClass::Mixed
    }
}

/// Convenience: voxelize a sphere into a sparse octree.
pub fn voxelize_opacity_sphere_octree(
    center: Vec3,
    radius: f32,
    material_id: u16,
    voxel_size: f32,
    pool: &mut VoxelPool,
) -> Option<(SparseOctree, u32, Vec3)> {
    let padding = voxel_size * 2.0;
    let aabb = Aabb {
        min: center - Vec3::splat(radius + padding),
        max: center + Vec3::splat(radius + padding),
    };

    let opacity_fn = |pos: Vec3| -> (f32, u16) {
        let dist = (pos - center).length() - radius;
        let opacity = (1.0 - dist / voxel_size).clamp(0.0, 1.0);
        (opacity, material_id)
    };

    voxelize_opacity_octree(opacity_fn, &aabb, voxel_size, pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::INTERIOR_NODE;

    #[test]
    fn sphere_produces_leaves() {
        let mut pool = VoxelPool::new(1_000_000);
        let (octree, voxel_count, _) =
            voxelize_opacity_sphere_octree(Vec3::ZERO, 0.5, 0, 0.1, &mut pool).unwrap();

        assert!(voxel_count > 0, "should allocate voxels for sphere surface");
        assert_eq!(octree.leaf_count(), voxel_count as usize);
    }

    #[test]
    fn sphere_has_interior_nodes() {
        let mut pool = VoxelPool::new(1_000_000);
        let (octree, _, _) =
            voxelize_opacity_sphere_octree(Vec3::ZERO, 3.0, 0, 0.1, &mut pool).unwrap();

        // The interior of a sphere should produce INTERIOR nodes at coarse levels.
        let mut found_interior = false;
        let ext = octree.extent();
        // Don't iterate all voxels (could be millions) — sample a known-interior coord.
        let mid = ext / 2;
        if let Some(val) = octree.lookup(glam::UVec3::new(mid, mid, mid)) {
            if val == INTERIOR_NODE {
                found_interior = true;
            }
        }
        assert!(found_interior, "large sphere should have interior at center");
    }

    #[test]
    fn empty_region_produces_no_voxels() {
        let mut pool = VoxelPool::new(256);
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };
        let (octree, voxel_count, _) =
            voxelize_opacity_octree(|_| (0.0, 0), &aabb, 0.1, &mut pool).unwrap();

        assert_eq!(voxel_count, 0);
        assert_eq!(octree.leaf_count(), 0);
    }

    #[test]
    fn fully_opaque_region_is_interior() {
        let mut pool = VoxelPool::new(256);
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(0.05),
        };
        let (octree, voxel_count, _) =
            voxelize_opacity_octree(|_| (1.0, 0), &aabb, 0.1, &mut pool).unwrap();

        assert_eq!(voxel_count, 0, "fully opaque should be INTERIOR, not voxels");
        assert_eq!(octree.as_slice()[0], INTERIOR_NODE);
    }

    #[test]
    fn leaf_voxels_have_correct_opacity() {
        let mut pool = VoxelPool::new(1_000_000);
        let (octree, _, _) =
            voxelize_opacity_sphere_octree(Vec3::ZERO, 0.3, 42, 0.1, &mut pool).unwrap();

        let mut found_opaque = false;
        for (_, slot, _) in octree.iter_leaves() {
            let sv = pool.get(slot);
            if sv.opacity_f32() > 0.5 {
                found_opaque = true;
                assert_eq!(sv.material_id(), 42);
            }
        }
        assert!(found_opaque, "should have opaque voxels in sphere");
    }
}
