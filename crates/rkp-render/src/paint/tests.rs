use super::*;
use glam::{UVec3, Vec3};
use rkp_core::brick_face_links::FACE_EMPTY;
use rkp_core::brick_pool::{BrickPool, BRICK_DIM};
use rkp_core::leaf_attr_pool::LeafAttrPool;
use rkp_core::sparse_octree::SparseOctree;
use rkp_core::OctreeAllocator;

const EPS: f32 = 1e-4;

// -------- brush_weight --------

#[test]
fn brush_weight_center_is_strength() {
    assert!((brush_weight(0.0, 1.0, 0.8, 0.5) - 0.8).abs() < EPS);
}

#[test]
fn brush_weight_edge_is_zero() {
    assert!(brush_weight(1.0, 1.0, 0.8, 0.5) < EPS);
}

#[test]
fn brush_weight_beyond_radius_is_zero() {
    assert_eq!(brush_weight(2.0, 1.0, 1.0, 0.5), 0.0);
}

#[test]
fn brush_weight_hard_cutoff_is_flat() {
    assert!((brush_weight(0.0, 1.0, 0.7, 0.0) - 0.7).abs() < EPS);
    assert!((brush_weight(0.99, 1.0, 0.7, 0.0) - 0.7).abs() < EPS);
    assert_eq!(brush_weight(1.0, 1.0, 0.7, 0.0), 0.0);
}

#[test]
fn brush_weight_monotonically_decreases() {
    let (r, s, fo) = (2.0, 0.9, 0.6);
    let mut prev = brush_weight(0.0, r, s, fo);
    for i in 1..20 {
        let d = (i as f32) * r / 20.0;
        let cur = brush_weight(d, r, s, fo);
        assert!(cur <= prev + EPS, "d={d} cur={cur} prev={prev}");
        prev = cur;
    }
}

// -------- pack/unpack roundtrip --------

#[test]
fn pack_unpack_roundtrip() {
    let (rgb, i) = unpack_color(pack_color([0.5, 0.25, 0.75], 0.8));
    assert!((rgb[0] - 0.5).abs() < 0.01);
    assert!((rgb[1] - 0.25).abs() < 0.01);
    assert!((rgb[2] - 0.75).abs() < 0.01);
    assert!((i - 0.8).abs() < 0.01);
}

#[test]
fn pack_intensity_zero_means_no_override() {
    // A voxel with no color override reads back as zeros in every channel.
    let packed = pack_color([0.0, 0.0, 0.0], 0.0);
    assert_eq!(packed, 0);
}

// -------- paint_leaf_material --------

#[test]
fn paint_material_full_weight_overwrites() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(2).unwrap();
    *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 1);
    paint_leaf_material(&mut pool, 0, 42, 1.0);
    assert_eq!(pool.get(0).material_primary, 42);
    assert_eq!(pool.get(0).blend_weight(), 0);
}

#[test]
fn paint_material_partial_weight_blends_into_secondary() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 3);
    paint_leaf_material(&mut pool, 0, 7, 0.5);
    let a = pool.get(0);
    assert_eq!(a.material_primary, 3);
    assert_eq!(a.material_secondary(), 7);
    // 0.5 * 15 = 7.5 → rounds to 8
    assert_eq!(a.blend_weight(), 8);
}

#[test]
fn paint_material_noop_when_already_matching_primary() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 5);
    paint_leaf_material(&mut pool, 0, 5, 0.5);
    // Half-weight paint onto an already-matching primary: no secondary blend.
    let a = pool.get(0);
    assert_eq!(a.material_primary, 5);
    assert_eq!(a.material_secondary(), 0);
    assert_eq!(a.blend_weight(), 0);
}

#[test]
fn paint_material_zero_weight_is_noop() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 9);
    paint_leaf_material(&mut pool, 0, 42, 0.0);
    assert_eq!(pool.get(0).material_primary, 9);
}

// -------- paint_leaf_color --------

#[test]
fn paint_color_from_unpainted_creates_override() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    paint_leaf_color(&mut pool, 0, [1.0, 0.0, 0.0], 1.0);
    assert_ne!(pool.color(0), 0, "color override should be set");
    let (rgb, i) = unpack_color(pool.color(0));
    assert!((rgb[0] - 1.0).abs() < 0.01);
    assert!((i - 1.0).abs() < 0.01);
}

#[test]
fn paint_color_partial_weight_ramps_intensity() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    paint_leaf_color(&mut pool, 0, [1.0, 0.0, 0.0], 0.5);
    let (_, i) = unpack_color(pool.color(0));
    // 0.0 + (1.0-0.0)*0.5 = 0.5
    assert!((i - 0.5).abs() < 0.01, "partial weight should give ~0.5 intensity, got {i}");
}

#[test]
fn paint_color_zero_weight_noop() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    paint_leaf_color(&mut pool, 0, [1.0, 1.0, 1.0], 0.0);
    assert_eq!(pool.color(0), 0);
}

// -------- erase_leaf_color --------

#[test]
fn erase_full_weight_clears_override() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    pool.set_color(0, pack_color([1.0, 0.5, 0.0], 1.0));
    erase_leaf_color(&mut pool, 0, 1.0);
    assert_eq!(pool.color(0), 0);
}

#[test]
fn erase_partial_weight_dims_intensity() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    pool.set_color(0, pack_color([1.0, 0.0, 0.0], 1.0));
    erase_leaf_color(&mut pool, 0, 0.5);
    let (rgb, i) = unpack_color(pool.color(0));
    assert!((rgb[0] - 1.0).abs() < 0.01); // RGB preserved
    assert!((i - 0.5).abs() < 0.01); // intensity halved
}

#[test]
fn erase_unpainted_is_noop() {
    let mut pool = LeafAttrPool::new(8);
    pool.allocate_range(1).unwrap();
    erase_leaf_color(&mut pool, 0, 1.0);
    assert_eq!(pool.color(0), 0);
}

// -------- leaves_in_sphere --------

/// Build a packed-buffer octree that contains a single LEAF at
/// `coord` with `slot`. Returns the buffer + its root_offset, depth,
/// and base_voxel_size so tests can call `leaves_in_sphere`
/// straight against the packed form (same shape that
/// `OctreeGpu::data()` hands to the shader).
fn packed_octree_with_leaf(coord: UVec3, slot: u32, depth: u8, vs: f32)
    -> (Vec<u32>, u32, u8, f32)
{
    let mut tree = SparseOctree::new(depth, vs);
    tree.insert(coord, slot);
    let mut alloc = OctreeAllocator::new();
    let handle = alloc.allocate(&tree);
    (alloc.as_slice().to_vec(), handle.root_offset, handle.depth, handle.base_voxel_size)
}

#[test]
fn leaves_in_sphere_hits_leaf_at_center() {
    // 4³ octree (depth=2), voxel_size=1.0, leaf at (1,1,1) → center (1.5, 1.5, 1.5).
    let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(1, 1, 1), 42, 2, 1.0);
    let pool = BrickPool::new(1);
    let hits = leaves_in_sphere(
        &buf, root, depth, vs,
        &pool,
        Vec3::ZERO,
        Vec3::new(1.5, 1.5, 1.5),
        0.9,
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].leaf_slot, 42);
    assert!(hits[0].distance < EPS);
}

#[test]
fn leaves_in_sphere_misses_leaf_outside_radius() {
    let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(0, 0, 0), 5, 2, 1.0);
    let pool = BrickPool::new(1);
    // Leaf center is (0.5, 0.5, 0.5) — brush at (3, 3, 3) with radius 1.0 can't reach it.
    let hits = leaves_in_sphere(
        &buf, root, depth, vs,
        &pool,
        Vec3::ZERO,
        Vec3::new(3.0, 3.0, 3.0),
        1.0,
    );
    assert!(hits.is_empty(), "brush far from leaf should yield no hits, got {}", hits.len());
}

#[test]
fn leaves_in_sphere_zero_radius_no_hits() {
    let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(0, 0, 0), 1, 2, 1.0);
    let pool = BrickPool::new(1);
    let hits = leaves_in_sphere(
        &buf, root, depth, vs,
        &pool,
        Vec3::ZERO,
        Vec3::ZERO,
        0.0,
    );
    assert!(hits.is_empty());
}

#[test]
fn leaves_in_sphere_multiple_hits() {
    // Two leaves at adjacent cells, large brush reaches both.
    let mut tree = SparseOctree::new(2, 1.0);
    tree.insert(UVec3::new(0, 0, 0), 11); // center (0.5, 0.5, 0.5)
    tree.insert(UVec3::new(1, 0, 0), 22); // center (1.5, 0.5, 0.5)
    let mut alloc = OctreeAllocator::new();
    let handle = alloc.allocate(&tree);
    let pool = BrickPool::new(1);
    // Brush at the midpoint (1.0, 0.5, 0.5) with radius 0.8 catches both.
    let hits = leaves_in_sphere(
        alloc.as_slice(),
        handle.root_offset,
        handle.depth,
        handle.base_voxel_size,
        &pool,
        Vec3::ZERO,
        Vec3::new(1.0, 0.5, 0.5),
        0.8,
    );
    assert_eq!(hits.len(), 2);
    let slots: Vec<u32> = hits.iter().map(|h| h.leaf_slot).collect();
    assert!(slots.contains(&11));
    assert!(slots.contains(&22));
}

// -------- leaf_at_local_pos --------

/// Build a small octree with a single 4³ brick allocated at the
/// origin corner and populated with leaf slots for a flat XY
/// "slab" (z=0 layer of the brick). Helper for the flood-fill
/// tests — deterministic layout where we know exactly which
/// (brick, cell) → slot mapping exists.
fn slab_octree(
    attrs: &mut LeafAttrPool,
    bricks: &mut BrickPool,
) -> (SparseOctree, u32) {
    // depth 2 → 4³ voxels, one brick terminates the root directly
    // since BRICK_LEVELS = 2 == depth. Voxel size 1.0 → brick
    // covers [0,4) per axis.
    let depth = 2u8;
    let voxel_size = 1.0_f32;
    let mut tree = SparseOctree::new(depth, voxel_size);
    // Allocate one brick and populate its z=0 layer with unique
    // leaf slots (0..16).
    let brick_id = bricks.allocate().unwrap();
    // The octree needs to point AT this brick. SparseOctree::new
    // gives an all-EMPTY root; set the root directly to a brick
    // terminator.
    // `set_at_level` with target_level 0 puts a value at the root.
    let brick_node = rkp_core::sparse_octree::make_brick(brick_id);
    tree.set_at_level(UVec3::ZERO, 0, brick_node);

    // Allocate 16 leaf slots and write them into the z=0 layer.
    let slots = attrs.allocate_range(16).unwrap();
    for y in 0..BRICK_DIM {
        for x in 0..BRICK_DIM {
            let slot = slots[(x + y * BRICK_DIM) as usize];
            bricks.set_cell(brick_id, x, y, 0, slot);
        }
    }
    (tree, brick_id)
}

#[test]
fn leaf_at_local_pos_hits_populated_cell() {
    let mut attrs = LeafAttrPool::new(64);
    let mut bricks = BrickPool::new(4);
    let (tree, brick_id) = slab_octree(&mut attrs, &mut bricks);
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // Query the center of cell (2, 1, 0) → (2.5, 1.5, 0.5).
    let hit = leaf_at_local_pos(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, Vec3::ZERO, Vec3::new(2.5, 1.5, 0.5),
    ).expect("hit should resolve to a brick cell");

    assert_eq!(hit.brick_id, brick_id);
    assert_eq!(hit.cell, UVec3::new(2, 1, 0));
    // center should be exactly (2.5, 1.5, 0.5) for voxel_size 1.0.
    assert!((hit.center_local - Vec3::new(2.5, 1.5, 0.5)).length() < EPS);
    // cell_size = brick extent / BRICK_DIM = 4.0 / 4 = 1.0.
    assert!((hit.cell_size - 1.0).abs() < EPS);
}

#[test]
fn leaf_at_local_pos_returns_none_in_empty_cell() {
    let mut attrs = LeafAttrPool::new(64);
    let mut bricks = BrickPool::new(4);
    let (tree, _) = slab_octree(&mut attrs, &mut bricks);
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // z=2 layer is empty — slab only populated z=0.
    let hit = leaf_at_local_pos(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, Vec3::ZERO, Vec3::new(2.5, 1.5, 2.5),
    );
    assert!(hit.is_none());
}

#[test]
fn leaf_at_local_pos_returns_none_for_empty_octree() {
    let bricks = BrickPool::new(4);
    let tree = SparseOctree::new(2, 1.0);
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);
    let hit = leaf_at_local_pos(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0),
    );
    assert!(hit.is_none());
}

// -------- surface_flood_fill --------

#[test]
fn flood_fill_starts_from_hit_voxel() {
    let mut attrs = LeafAttrPool::new(64);
    let mut bricks = BrickPool::new(4);
    let (tree, _) = slab_octree(&mut attrs, &mut bricks);
    // Force the face links to all-empty — this test uses a single
    // brick so no cross-brick walks are needed.
    let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // Radius 0.0-ish: only the hit cell should come back.
    let hits = surface_flood_fill(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, &face_links, Vec3::ZERO,
        Vec3::new(2.5, 1.5, 0.5), 0.0001,
    );
    // init_dist from center is 0, so even an epsilon radius captures the hit.
    assert_eq!(hits.len(), 1, "got {hits:?}");
    assert!(hits[0].distance < EPS);
}

#[test]
fn flood_fill_expands_face_neighbors_within_radius() {
    let mut attrs = LeafAttrPool::new(64);
    let mut bricks = BrickPool::new(4);
    let (tree, _) = slab_octree(&mut attrs, &mut bricks);
    let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // Start at (2.5, 1.5, 0.5). Radius 1.5 emits every z=0 voxel
    // whose world-local center sits within 1.5 world units of the
    // brush origin — BFS uses face adjacency for reachability but
    // the emitted distance is straight-line, so diagonals enter
    // once the diagonal neighbor is within radius.
    let hits = surface_flood_fill(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, &face_links, Vec3::ZERO,
        Vec3::new(2.5, 1.5, 0.5), 1.5,
    );
    // Expected: 1 center + 4 face neighbors (dist 1.0) + 4
    // diagonals (dist sqrt(2) ≈ 1.414 < 1.5). Missing voxels
    // beyond the 3×3 centered patch are at dist >= 2.
    assert_eq!(hits.len(), 9, "got {hits:?}");

    let near_zero = hits.iter().filter(|h| h.distance < 0.01).count();
    let near_one = hits.iter().filter(|h| (h.distance - 1.0).abs() < 0.01).count();
    let near_diag = hits.iter()
        .filter(|h| (h.distance - std::f32::consts::SQRT_2).abs() < 0.01)
        .count();
    assert_eq!(near_zero, 1);
    assert_eq!(near_one, 4);
    assert_eq!(near_diag, 4);
}

#[test]
fn flood_fill_crosses_brick_boundary_via_face_link() {
    // Two bricks side-by-side on +X. Start in brick A, expand into
    // brick B through the face link.
    let mut attrs = LeafAttrPool::new(128);
    let mut bricks = BrickPool::new(4);

    let brick_a = bricks.allocate().unwrap();
    let brick_b = bricks.allocate().unwrap();

    // Populate brick A's +X edge cell (3, 0, 0) and brick B's
    // −X edge cell (0, 0, 0). Leave everything else empty.
    let slots = attrs.allocate_range(2).unwrap();
    bricks.set_cell(brick_a, 3, 0, 0, slots[0]);
    bricks.set_cell(brick_b, 0, 0, 0, slots[1]);

    // Build an octree that has both bricks. Depth 3 gives us 8³
    // voxels split into a 2×2×2 brick grid (brick edge = 4 voxels).
    let mut tree = SparseOctree::new(3, 1.0);
    // Put brick A at brick-coord (0,0,0) (voxel 0), brick B at
    // (1,0,0) (voxel 4). `set_at_level` with target_level equal to
    // the brick depth (1, for depth 3 with BRICK_LEVELS=2) puts a
    // brick terminator.
    let brick_depth = 3 - 2; // 1
    let a_node = rkp_core::sparse_octree::make_brick(brick_a);
    let b_node = rkp_core::sparse_octree::make_brick(brick_b);
    tree.set_at_level(UVec3::new(0, 0, 0), brick_depth, a_node);
    tree.set_at_level(UVec3::new(4, 0, 0), brick_depth, b_node);

    // Face-link table: A's +X → B; B's −X → A; everything else empty.
    let max_brick_id = brick_a.max(brick_b);
    let mut face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]; (max_brick_id + 1) as usize];
    face_links[brick_a as usize][1] = brick_b; // FACE_PX
    face_links[brick_b as usize][0] = brick_a; // FACE_NX

    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // Start at brick A's edge cell center (3.5, 0.5, 0.5). Radius
    // 1.5 should reach the cell at brick B's (0, 0, 0) which is at
    // world (4.5, 0.5, 0.5) — distance 1.0 away.
    let hits = surface_flood_fill(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, &face_links, Vec3::ZERO,
        Vec3::new(3.5, 0.5, 0.5), 1.5,
    );
    assert_eq!(hits.len(), 2, "expected A + B neighbor, got {hits:?}");
    let slots_hit: Vec<u32> = hits.iter().map(|h| h.leaf_slot).collect();
    assert!(slots_hit.contains(&slots[0]));
    assert!(slots_hit.contains(&slots[1]));
}

#[test]
fn flood_fill_respects_radius_cutoff() {
    let mut attrs = LeafAttrPool::new(64);
    let mut bricks = BrickPool::new(4);
    let (tree, _) = slab_octree(&mut attrs, &mut bricks);
    let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
    let mut alloc = OctreeAllocator::new();
    let h = alloc.allocate(&tree);

    // Radius 0.7 is less than one cell hop (1.0), so only the hit
    // cell should be returned.
    let hits = surface_flood_fill(
        alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
        &bricks, &face_links, Vec3::ZERO,
        Vec3::new(2.5, 1.5, 0.5), 0.7,
    );
    assert_eq!(hits.len(), 1, "got {hits:?}");
}

#[test]
fn leaves_in_sphere_respects_grid_origin() {
    // Same leaf setup, but the octree's (0,0,0) sits at (-2, -2, -2).
    // Leaf at (1,1,1) is now at world (-2 + 1.5, -2 + 1.5, -2 + 1.5).
    let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(1, 1, 1), 99, 2, 1.0);
    let pool = BrickPool::new(1);
    let origin = Vec3::new(-2.0, -2.0, -2.0);
    let hits = leaves_in_sphere(
        &buf, root, depth, vs,
        &pool, origin,
        origin + Vec3::new(1.5, 1.5, 1.5),
        0.6,
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].leaf_slot, 99);
}
