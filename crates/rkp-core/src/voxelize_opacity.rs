//! Opacity-field voxelization — converts an opacity function into splat bricks.
//!
//! Mirrors rkf-core's `voxelize_sdf` but writes opacity values instead of SDF
//! distances. The closure returns `(opacity, material_id)` where opacity is 0.0
//! (empty) to 1.0 (fully opaque).

use glam::{UVec3, Vec3};
use rkf_core::brick::Brick;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT};
use rkf_core::scene_node::BrickMapHandle;
use rkf_core::brick_pool::BrickPool;
use rkf_core::constants::BRICK_DIM;
use rkf_core::Aabb;

use crate::SplatVoxel;

/// Voxelize an opacity function into splat bricks.
///
/// `opacity_fn`: returns `(opacity, material_id)` at a world-space position.
///   - `opacity`: 0.0 = empty, 1.0 = fully opaque.
///   - `material_id`: 16-bit material palette index.
///
/// `aabb`: world-space bounding box of the object.
/// `voxel_size`: size of a single voxel in world units.
///
/// Returns `Some((handle, brick_count))` on success, `None` if pool allocation fails.
pub fn voxelize_opacity<F>(
    opacity_fn: F,
    aabb: &Aabb,
    voxel_size: f32,
    pool: &mut BrickPool,
    map_alloc: &mut BrickMapAllocator,
) -> Option<(BrickMapHandle, u32)>
where
    F: Fn(Vec3) -> (f32, u16),
{
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    let aabb_size = aabb.max - aabb.min;
    let dims = UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    let grid_origin = -Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );

    // Pass 1: determine which bricks need allocation by sampling center opacity.
    let mut brick_map = BrickMap::new(dims);
    let mut needed_count = 0u32;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);

                let (center_opacity, _) = opacity_fn(brick_center);

                // Sample corners to detect bricks near the surface.
                // A brick needs allocation if any sample is non-zero or the
                // center is non-zero (the opacity field may have a surface edge).
                let mut any_nonzero = center_opacity > 0.001;
                if !any_nonzero {
                    // Check all 8 corners
                    for cz in 0..2u32 {
                        for cy in 0..2u32 {
                            for cx in 0..2u32 {
                                let corner = brick_min
                                    + Vec3::new(
                                        cx as f32 * brick_world_size,
                                        cy as f32 * brick_world_size,
                                        cz as f32 * brick_world_size,
                                    );
                                let (corner_opacity, _) = opacity_fn(corner);
                                if corner_opacity > 0.001 {
                                    any_nonzero = true;
                                }
                            }
                        }
                    }
                }

                if any_nonzero {
                    brick_map.set(bx, by, bz, 0); // placeholder — replaced in pass 2
                    needed_count += 1;
                }
            }
        }
    }

    if needed_count == 0 {
        let handle = map_alloc.allocate(&brick_map);
        return Some((handle, 0));
    }

    let slots = pool.allocate_range(needed_count)?;
    let mut slot_idx = 0;

    // Pass 2: populate bricks with opacity data.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                if brick_map.get(bx, by, bz) == Some(EMPTY_SLOT) {
                    continue;
                }

                let slot = slots[slot_idx];
                slot_idx += 1;
                brick_map.set(bx, by, bz, slot);

                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );

                let brick = pool.get_mut(slot);
                populate_opacity_brick(brick, &opacity_fn, brick_min, voxel_size);
            }
        }
    }

    let handle = map_alloc.allocate(&brick_map);
    Some((handle, needed_count))
}

/// Populate a single brick with opacity-field voxels.
fn populate_opacity_brick<F>(
    brick: &mut Brick,
    opacity_fn: &F,
    brick_min: Vec3,
    voxel_size: f32,
) where
    F: Fn(Vec3) -> (f32, u16),
{
    let half_voxel = voxel_size * 0.5;

    for vz in 0..BRICK_DIM {
        for vy in 0..BRICK_DIM {
            for vx in 0..BRICK_DIM {
                let pos = brick_min
                    + Vec3::new(
                        vx as f32 * voxel_size + half_voxel,
                        vy as f32 * voxel_size + half_voxel,
                        vz as f32 * voxel_size + half_voxel,
                    );

                let (opacity, material_id) = opacity_fn(pos);
                let sample: rkf_core::voxel::VoxelSample =
                    SplatVoxel::new(opacity.clamp(0.0, 1.0), material_id).into();
                brick.set(vx, vy, vz, sample);
            }
        }
    }
}

/// Convenience: voxelize a sphere into the opacity field.
///
/// Returns `(handle, brick_count, aabb, dims)`.
pub fn voxelize_opacity_sphere(
    center: Vec3,
    radius: f32,
    material_id: u16,
    voxel_size: f32,
    pool: &mut BrickPool,
    map_alloc: &mut BrickMapAllocator,
) -> Option<(BrickMapHandle, u32, Aabb, UVec3)> {
    let padding = voxel_size * 2.0;
    let aabb = Aabb {
        min: center - Vec3::splat(radius + padding),
        max: center + Vec3::splat(radius + padding),
    };

    let opacity_fn = |pos: Vec3| -> (f32, u16) {
        let dist = (pos - center).length() - radius;
        // Smooth falloff over one voxel width
        let opacity = (1.0 - dist / voxel_size).clamp(0.0, 1.0);
        (opacity, material_id)
    };

    let (handle, brick_count) = voxelize_opacity(opacity_fn, &aabb, voxel_size, pool, map_alloc)?;

    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let aabb_size = aabb.max - aabb.min;
    let dims = UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    Some((handle, brick_count, aabb, dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxelize_sphere_produces_bricks() {
        let mut pool = BrickPool::new(256);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_opacity_sphere(
            Vec3::ZERO,
            0.5,
            0,
            0.1,
            &mut pool,
            &mut alloc,
        );

        let (handle, brick_count, _aabb, dims) = result.expect("voxelization should succeed");
        assert!(brick_count > 0, "should allocate at least one brick");
        assert!(dims.x > 0 && dims.y > 0 && dims.z > 0);
        let _ = handle; // just verify it was created
    }

    #[test]
    fn voxelize_sphere_has_surface_voxels() {
        let mut pool = BrickPool::new(256);
        let mut alloc = BrickMapAllocator::new();

        let (handle, _count, _aabb, dims) = voxelize_opacity_sphere(
            Vec3::ZERO,
            0.5,
            42,
            0.1,
            &mut pool,
            &mut alloc,
        )
        .unwrap();

        // Check that at least one voxel has non-zero opacity
        let mut found_opaque = false;
        let mut found_empty = false;

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(slot) = alloc.get_entry(&handle, bx, by, bz) {
                        if slot == EMPTY_SLOT {
                            continue;
                        }
                        let brick = pool.get(slot);
                        for vz in 0..BRICK_DIM {
                            for vy in 0..BRICK_DIM {
                                for vx in 0..BRICK_DIM {
                                    let sample = brick.sample(vx, vy, vz);
                                    let sv = SplatVoxel::from(sample);
                                    if sv.opacity_f32() > 0.5 {
                                        found_opaque = true;
                                        assert_eq!(sv.material_id(), 42);
                                    }
                                    if sv.opacity_f32() < 0.01 {
                                        found_empty = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(found_opaque, "should have opaque voxels inside the sphere");
        assert!(found_empty, "should have empty voxels outside the sphere");
    }

    #[test]
    fn empty_region_produces_no_bricks() {
        let mut pool = BrickPool::new(256);
        let mut alloc = BrickMapAllocator::new();

        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(1.0),
        };

        let (_, brick_count) =
            voxelize_opacity(|_| (0.0, 0), &aabb, 0.1, &mut pool, &mut alloc).unwrap();

        assert_eq!(brick_count, 0, "fully empty region should produce no bricks");
    }
}
