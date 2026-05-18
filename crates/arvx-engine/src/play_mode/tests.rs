use super::*;
use glam::IVec3;
use arvx_core::sparse_octree::{INTERIOR_NODE, make_brick, make_leaf};
use arvx_core::{BRICK_DIM, BRICK_INTERIOR, BrickPool};

// base_voxel_size = 1.0 throughout these tests so coarse-cell math is
// visible by eye: ratio = collider_cell_size / 1.0 = collider_cell_size.

#[test]
fn interior_node_root_fills_full_extent() {
    // Depth-2 tree (4³ fine voxels), root collapsed to INTERIOR.
    // Coarse cell = 2 fine voxels, so 2³ = 8 coarse cells expected.
    let nodes = vec![INTERIOR_NODE];
    let pool = BrickPool::new(0);
    let (coords, _) = build_coarse_collider(&nodes, pool.as_slice(), 0, 2, 1, 1.0, 2.0);
    assert_eq!(coords.len(), 8);
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                assert!(
                    coords.contains(&IVec3::new(x, y, z)),
                    "missing coarse cell ({x},{y},{z})"
                );
            }
        }
    }
}

#[test]
fn mip_leaf_root_fills_full_extent() {
    // Same shape as the INTERIOR test but with a coarse LEAF at the root.
    let nodes = vec![make_leaf(0)];
    let pool = BrickPool::new(0);
    let (coords, _) = build_coarse_collider(&nodes, pool.as_slice(), 0, 2, 1, 1.0, 2.0);
    assert_eq!(coords.len(), 8);
}

#[test]
fn brick_cells_are_walked_not_dropped() {
    // Depth-2 tree (4³ = brick depth), root = single brick.
    // Populate a sparse pattern in the brick: a leaf at (0,0,0),
    // BRICK_INTERIOR at (3,3,3), and a leaf at (1,2,3). Everything
    // else stays BRICK_EMPTY.
    //
    // With coarse cell = 2 fine voxels, we expect three coarse buckets:
    //   (0,0,0) ← from fine (0,0,0)
    //   (1,1,1) ← from fine (3,3,3)
    //   (0,1,1) ← from fine (1,2,3)
    let mut pool = BrickPool::new(4);
    let bid = pool.allocate().expect("brick alloc");
    assert_eq!(bid, 0);
    pool.set_cell(bid, 0, 0, 0, 0); // leaf_attr 0 (occupied)
    pool.set_cell(bid, 3, 3, 3, BRICK_INTERIOR);
    pool.set_cell(bid, 1, 2, 3, 7);

    let nodes = vec![make_brick(bid)];
    let (coords, _) = build_coarse_collider(&nodes, pool.as_slice(), 0, 2, 1, 1.0, 2.0);
    assert_eq!(
        coords.len(), 3,
        "got {coords:?}; pre-fix this returned 0 because iter_leaves skipped bricks",
    );
    assert!(coords.contains(&IVec3::new(0, 0, 0)));
    assert!(coords.contains(&IVec3::new(1, 1, 1)));
    assert!(coords.contains(&IVec3::new(0, 1, 1)));
}

#[test]
fn empty_brick_contributes_no_coords() {
    // All-empty brick (default state after allocate) must emit nothing.
    let mut pool = BrickPool::new(4);
    let bid = pool.allocate().unwrap();
    let nodes = vec![make_brick(bid)];
    let (coords, _) = build_coarse_collider(&nodes, pool.as_slice(), 0, 2, 1, 1.0, 2.0);
    assert!(coords.is_empty());
}

#[test]
fn root_offset_is_subtracted_from_branch_children() {
    // Branch at root with a non-zero root_offset. We hand-pack a tiny
    // depth-3 tree (8³ fine voxels): 1 branch + 8 children. Bricks live
    // at depth 3 - BRICK_LEVELS(=2) = 1, so each branch child either
    // collapses to a sentinel or is a brick covering BRICK_DIM=4 voxels
    // per axis. We splice it into a larger buffer to give root_offset a
    // non-zero value — the engine always passes the full scene-wide
    // buffer with the entity's subtree at some interior offset.
    //
    // Without the `node - root_offset` correction, the branch's child
    // offset would dereference the wrong slot.
    use arvx_core::sparse_octree::EMPTY_NODE;
    let prefix = 17usize;
    let branch_target = (prefix + 1) as u32; // children start right after branch

    let mut nodes = vec![EMPTY_NODE; prefix];
    nodes.push(branch_target); // root branch at buffer index `prefix`
    // 8 children, only octant 0 (origin (0,0,0)) carries a brick.
    let mut pool = BrickPool::new(4);
    let bid = pool.allocate().unwrap();
    pool.set_cell(bid, 0, 0, 0, 0); // single occupied cell at (0,0,0)
    nodes.push(make_brick(bid));
    for _ in 1..8 {
        nodes.push(EMPTY_NODE);
    }

    let len = 1 + 8;
    // coarse cell = 2 fine voxels. Brick cell (0,0,0) → coarse (0,0,0).
    let (coords, _) = build_coarse_collider(&nodes, pool.as_slice(), prefix, 3, len, 1.0, 2.0);
    let _ = BRICK_DIM;
    assert_eq!(coords, vec![IVec3::new(0, 0, 0)]);
}

// ── tight AABB walker ────────────────────────────────────────────────

#[test]
fn tight_aabb_skips_padding() {
    // Depth-3 tree (8³ voxels). Only the (0,0,0) octant carries
    // geometry — a brick whose only occupied cell is (1,1,1) inside it.
    // Total occupied region in finest-voxel units: a single 1³ voxel
    // at fine coord (1,1,1).
    //
    // base_voxel_size = 0.5 → that voxel spans [0.5, 1.0]³ in
    // grid-local space. With grid_origin = ZERO, that's the result.
    //
    // Crucially: this proves the tight AABB ignores the rest of the
    // tree's empty volume. The padded SpatialData.aabb would cover
    // the whole 4 units of extent.
    use arvx_core::sparse_octree::EMPTY_NODE;
    let mut pool = BrickPool::new(4);
    let bid = pool.allocate().unwrap();
    pool.set_cell(bid, 1, 1, 1, 0); // single occupied cell

    let mut nodes = vec![1u32]; // branch points at children starting at index 1
    nodes.push(make_brick(bid));
    for _ in 1..8 {
        nodes.push(EMPTY_NODE);
    }

    let aabb = compute_tight_local_aabb(
        &nodes, pool.as_slice(), 0, 3, nodes.len() as u32, 0.5, glam::Vec3::ZERO,
    )
    .expect("should find occupied region");
    assert_eq!(aabb.min, glam::Vec3::new(0.5, 0.5, 0.5));
    assert_eq!(aabb.max, glam::Vec3::new(1.0, 1.0, 1.0));
}

#[test]
fn tight_aabb_offsets_by_grid_origin() {
    // INTERIOR_NODE root at depth=2 → fully solid 4³ region in fine
    // voxels. base_voxel_size = 0.25 → 1m total extent. With
    // grid_origin = (-0.5, -0.5, -0.5), the AABB is centered on origin.
    use arvx_core::sparse_octree::INTERIOR_NODE;
    let nodes = vec![INTERIOR_NODE];
    let pool = BrickPool::new(0);
    let aabb = compute_tight_local_aabb(
        &nodes, pool.as_slice(), 0, 2, 1, 0.25, glam::Vec3::splat(-0.5),
    )
    .unwrap();
    assert_eq!(aabb.min, glam::Vec3::splat(-0.5));
    assert_eq!(aabb.max, glam::Vec3::splat(0.5));
}

#[test]
fn tight_aabb_returns_none_for_empty() {
    use arvx_core::sparse_octree::EMPTY_NODE;
    let nodes = vec![EMPTY_NODE];
    let pool = BrickPool::new(0);
    let aabb = compute_tight_local_aabb(
        &nodes, pool.as_slice(), 0, 2, 1, 1.0, glam::Vec3::ZERO,
    );
    assert!(aabb.is_none());
}
