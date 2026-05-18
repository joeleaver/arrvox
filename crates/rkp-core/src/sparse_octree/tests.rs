use super::*;
use glam::UVec3;
use crate::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};


#[test]
fn new_octree_is_empty() {
    let tree = SparseOctree::new(3, 0.1);
    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.nodes[0], EMPTY_NODE);
    assert_eq!(tree.depth(), 3);
    assert_eq!(tree.extent(), 8); // 2^3
    assert_eq!(tree.leaf_count(), 0);
}

#[test]
fn insert_single_leaf() {
    let mut tree = SparseOctree::new(2, 0.1); // 4x4x4 bricks
    tree.insert(UVec3::new(1, 2, 3), 42);

    let result = tree.lookup(UVec3::new(1, 2, 3));
    assert_eq!(result, Some(make_leaf(42)));
    assert_eq!(tree.leaf_count(), 1);

    // Other coords should be EMPTY.
    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(EMPTY_NODE));
}

#[test]
fn insert_multiple_leaves() {
    let mut tree = SparseOctree::new(3, 0.1); // 8x8x8 bricks
    tree.insert(UVec3::new(0, 0, 0), 10);
    tree.insert(UVec3::new(7, 7, 7), 20);
    tree.insert(UVec3::new(3, 4, 5), 30);

    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(10)));
    assert_eq!(tree.lookup(UVec3::new(7, 7, 7)), Some(make_leaf(20)));
    assert_eq!(tree.lookup(UVec3::new(3, 4, 5)), Some(make_leaf(30)));
    assert_eq!(tree.leaf_count(), 3);
}

#[test]
fn insert_interior() {
    let mut tree = SparseOctree::new(2, 0.1);
    tree.insert_interior(UVec3::new(1, 1, 1));

    assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), Some(INTERIOR_NODE));
    // Interior nodes aren't counted as leaves (no brick pool slot).
    assert_eq!(tree.leaf_count(), 0);
}

#[test]
fn lookup_out_of_bounds() {
    let tree = SparseOctree::new(2, 0.1); // 4x4x4
    assert_eq!(tree.lookup(UVec3::new(4, 0, 0)), None);
    assert_eq!(tree.lookup(UVec3::new(0, 4, 0)), None);
    assert_eq!(tree.lookup(UVec3::new(0, 0, 4)), None);
}

#[test]
fn collapse_uniform_children() {
    let mut tree = SparseOctree::new(1, 0.1); // 2x2x2 = 8 leaves at depth 1
    // Fill all 8 positions with the same slot.
    for z in 0..2u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                tree.insert(UVec3::new(x, y, z), 99);
            }
        }
    }
    // All children identical — root should collapse to a single leaf.
    assert_eq!(tree.nodes[0], make_leaf(99));
    assert_eq!(tree.leaf_count(), 1);
}

#[test]
fn compact_drops_orphan_slots_after_collapse() {
    // Build a tree that will have orphaned slots post-collapse, then
    // compact and verify the buffer shrinks but lookups still work.
    let mut tree = SparseOctree::new(2, 0.1); // 4x4x4
    for z in 0..4u32 {
        for y in 0..4u32 {
            for x in 0..4u32 {
                tree.insert(UVec3::new(x, y, z), 42);
            }
        }
    }
    // Fully uniform — should have collapsed to a single LEAF at the root,
    // but the intermediate branch allocations are still in `nodes`.
    assert_eq!(tree.nodes[0], make_leaf(42));
    assert!(tree.node_count() > 1, "should have orphaned slots before compact");

    tree.compact();
    // Only the root remains.
    assert_eq!(tree.node_count(), 1);
    assert_eq!(tree.nodes[0], make_leaf(42));

    // Lookups still work.
    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(42)));
    assert_eq!(tree.lookup(UVec3::new(3, 3, 3)), Some(make_leaf(42)));
}

#[test]
fn compact_preserves_tree_with_no_orphans() {
    // A tree with distinct children per octant has nothing to collapse.
    // compact() should produce a buffer of the same shape.
    let mut tree = SparseOctree::new(1, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 10);
    tree.insert(UVec3::new(1, 1, 1), 20);

    let before_count = tree.node_count();
    let before_lookup_000 = tree.lookup(UVec3::new(0, 0, 0));
    let before_lookup_111 = tree.lookup(UVec3::new(1, 1, 1));

    tree.compact();

    // Same number of reachable nodes (nothing to reclaim).
    assert_eq!(tree.node_count(), before_count);
    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), before_lookup_000);
    assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), before_lookup_111);
}

#[test]
fn deduplicate_shares_identical_subtrees() {
    // Build a depth-2 tree where each of the root's 8 children is an
    // identical branch: a branch whose 8 leaves all point to slot 99.
    // After dedup, those 8 parent-branches all reference the same 8-leaf
    // block, AND that block gets collapsed into a single LEAF by
    // try_collapse (so the tree is actually just a single LEAF at the
    // root after `insert` fires collapse).
    //
    // To specifically exercise DAG sharing (subtrees that don't themselves
    // collapse), build a non-uniform child and place it at the same
    // octant in every root-child.
    let mut tree = SparseOctree::new(2, 0.1); // 4x4x4

    // Fill octant 0 of each of the root's 8 quadrants with slot 7,
    // others with slot 11. So each of the 8 root-child branches has the
    // same internal structure — but because it's non-uniform, the branch
    // itself can't collapse into a single leaf.
    for root_oct in 0..8u32 {
        let dx = root_oct & 1;
        let dy = (root_oct >> 1) & 1;
        let dz = (root_oct >> 2) & 1;
        let base = UVec3::new(dx * 2, dy * 2, dz * 2);
        for inner_oct in 0..8u32 {
            let ix = inner_oct & 1;
            let iy = (inner_oct >> 1) & 1;
            let iz = (inner_oct >> 2) & 1;
            let coord = UVec3::new(base.x + ix, base.y + iy, base.z + iz);
            let slot = if inner_oct == 0 { 7 } else { 11 };
            tree.insert(coord, slot);
        }
    }

    let before = tree.node_count();
    tree.deduplicate_subtrees();
    let after = tree.node_count();

    // All 8 root-children are structurally identical; they should all
    // reference a single shared 8-child block after dedup.
    assert!(
        after < before,
        "dedup should shrink: {} -> {}", before, after,
    );

    // Sanity: every lookup returns the correct slot.
    for root_oct in 0..8u32 {
        let dx = root_oct & 1;
        let dy = (root_oct >> 1) & 1;
        let dz = (root_oct >> 2) & 1;
        let base = UVec3::new(dx * 2, dy * 2, dz * 2);
        for inner_oct in 0..8u32 {
            let ix = inner_oct & 1;
            let iy = (inner_oct >> 1) & 1;
            let iz = (inner_oct >> 2) & 1;
            let coord = UVec3::new(base.x + ix, base.y + iy, base.z + iz);
            let expected = if inner_oct == 0 {
                make_leaf(7)
            } else {
                make_leaf(11)
            };
            assert_eq!(
                tree.lookup(coord),
                Some(expected),
                "wrong lookup at {:?}", coord,
            );
        }
    }
}

#[test]
fn deduplicate_preserves_unique_subtrees() {
    // A tree whose 8 root-children are all structurally different should
    // not shrink (nothing to share).
    let mut tree = SparseOctree::new(1, 0.1);
    for i in 0..8u32 {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        // Each position gets a unique slot.
        tree.insert(UVec3::new(x, y, z), 100 + i);
    }

    // Verify each lookup is distinct and correct BEFORE dedup.
    for i in 0..8u32 {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        assert_eq!(
            tree.lookup(UVec3::new(x, y, z)),
            Some(make_leaf(100 + i)),
        );
    }

    tree.deduplicate_subtrees();

    // Lookups still correct.
    for i in 0..8u32 {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        assert_eq!(
            tree.lookup(UVec3::new(x, y, z)),
            Some(make_leaf(100 + i)),
            "post-dedup lookup wrong at i={i}",
        );
    }
}

#[test]
fn deduplicate_handles_trivial_root() {
    // A single-leaf tree: no branches, nothing to dedup, but shouldn't
    // crash and should leave the tree valid.
    let mut tree = SparseOctree::new(3, 0.1);
    // The default root is EMPTY_NODE. Dedup should be a no-op.
    tree.deduplicate_subtrees();
    assert_eq!(tree.nodes[0], EMPTY_NODE);
    assert_eq!(tree.node_count(), 1);
}

#[test]
fn deduplicate_recursive_self_similar_pattern() {
    // Build a "corner" pattern: at every level of subdivision, octant 0
    // gets subdivided the same way. This creates nested self-similar
    // structure — dedup should collapse it dramatically.
    let mut tree = SparseOctree::new(4, 0.1); // 16x16x16

    // Insert a single voxel at (0,0,0) and another at (15,15,15).
    // This forces subdivision along two diagonal chains. The empty
    // octants at each level of the chain share structure (all EMPTY).
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(15, 15, 15), 2);

    let before = tree.node_count();
    tree.deduplicate_subtrees();
    let after = tree.node_count();

    // Lookups preserved.
    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(1)));
    assert_eq!(tree.lookup(UVec3::new(15, 15, 15)), Some(make_leaf(2)));
    assert_eq!(tree.lookup(UVec3::new(5, 5, 5)), Some(EMPTY_NODE));

    // Even without obvious symmetry, there's enough shared sentinel
    // structure that dedup shouldn't grow the tree.
    assert!(after <= before, "dedup should not grow: {} -> {}", before, after);
}

#[test]
fn compact_handles_mixed_orphans_and_reachable() {
    // Insert enough to create nested branches, then insert more causing
    // some subtrees to collapse — producing orphans — while leaving other
    // subtrees intact. Compact should drop the orphans but preserve the
    // rest.
    let mut tree = SparseOctree::new(2, 0.1);
    // Half of the tree gets uniform data (will collapse); the other half
    // gets two distinct values (can't collapse).
    for z in 0..2u32 {
        for y in 0..4u32 {
            for x in 0..4u32 {
                tree.insert(UVec3::new(x, y, z), 7);
            }
        }
    }
    tree.insert(UVec3::new(0, 0, 3), 100);
    tree.insert(UVec3::new(1, 1, 3), 200);

    let before_count = tree.node_count();
    tree.compact();
    let after_count = tree.node_count();

    assert!(after_count < before_count, "compact should shrink when orphans exist ({} -> {})", before_count, after_count);

    // All original lookups must still succeed with the same values.
    assert_eq!(tree.lookup(UVec3::new(2, 2, 0)), Some(make_leaf(7)));
    assert_eq!(tree.lookup(UVec3::new(3, 3, 1)), Some(make_leaf(7)));
    assert_eq!(tree.lookup(UVec3::new(0, 0, 3)), Some(make_leaf(100)));
    assert_eq!(tree.lookup(UVec3::new(1, 1, 3)), Some(make_leaf(200)));
}

#[test]
fn no_collapse_with_different_children() {
    let mut tree = SparseOctree::new(1, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 10);
    tree.insert(UVec3::new(1, 0, 0), 20);

    // Root should be a branch, not collapsed.
    assert!(is_branch(tree.nodes[0]));
    assert_eq!(tree.leaf_count(), 2);
}

#[test]
fn overwrite_leaf() {
    let mut tree = SparseOctree::new(2, 0.1);
    tree.insert(UVec3::new(1, 1, 1), 42);
    tree.insert(UVec3::new(1, 1, 1), 99);

    assert_eq!(tree.lookup(UVec3::new(1, 1, 1)), Some(make_leaf(99)));
    assert_eq!(tree.leaf_count(), 1);
}

#[test]
fn lookup_with_depth_finest() {
    let mut tree = SparseOctree::new(3, 0.1);
    tree.insert(UVec3::new(2, 3, 4), 50);

    let (node, depth) = tree.lookup_with_depth(UVec3::new(2, 3, 4)).unwrap();
    assert_eq!(node, make_leaf(50));
    assert_eq!(depth, 3); // at finest level
}

#[test]
fn lookup_with_depth_coarse() {
    // A tree where a leaf exists at a non-max depth (uniform subtree).
    let tree = SparseOctree::new(3, 0.1);
    // The entire tree is EMPTY — lookup should return EMPTY at depth 0 (root).
    let (node, depth) = tree.lookup_with_depth(UVec3::new(2, 3, 4)).unwrap();
    assert_eq!(node, EMPTY_NODE);
    assert_eq!(depth, 0);
}

/// GPU-style position-based lookup (mirrors octree_lookup in WGSL).
/// Uses floating-point comparisons instead of integer bit tests.
fn gpu_style_lookup(tree: &SparseOctree, pos: glam::Vec3) -> (u32, u8) {
    let extent = tree.extent() as f32 * tree.base_voxel_size();
    let mut offset = 0usize;
    let mut half = extent * 0.5;
    let mut center = glam::Vec3::splat(half);

    for level in 0..tree.depth() {
        let node = tree.as_slice()[offset];
        if node == EMPTY_NODE { return (EMPTY_NODE, level); }
        if node == INTERIOR_NODE { return (INTERIOR_NODE, level); }
        if is_leaf(node) { return (leaf_slot(node), level); }

        // Branch — same logic as GPU shader
        let gx = if pos.x >= center.x { 1u32 } else { 0 };
        let gy = if pos.y >= center.y { 1u32 } else { 0 };
        let gz = if pos.z >= center.z { 1u32 } else { 0 };
        let child = (gx + gy * 2 + gz * 4) as usize;
        offset = node as usize + child;

        half *= 0.5;
        center.x += if pos.x >= center.x { half } else { -half };
        center.y += if pos.y >= center.y { half } else { -half };
        center.z += if pos.z >= center.z { half } else { -half };
    }

    let node = tree.as_slice()[offset];
    if node == EMPTY_NODE { return (EMPTY_NODE, tree.depth()); }
    if node == INTERIOR_NODE { return (INTERIOR_NODE, tree.depth()); }
    if is_leaf(node) { return (leaf_slot(node), tree.depth()); }
    (EMPTY_NODE, tree.depth())
}

#[test]
fn gpu_lookup_matches_coord_lookup() {
    // Build a small sphere octree (depth low enough that bricks don't
    // activate) and verify every leaf is reachable by position. Brick
    // path is exercised by tests in voxelize_octree.
    let mut attrs = crate::LeafAttrPool::new(100_000);
    let mut bricks = crate::BrickPool::new(64);
    let r = crate::voxelize_octree::voxelize_sphere_octree(
        glam::Vec3::ZERO, 0.4, 0, 0.4, &mut attrs, &mut bricks,
    ).unwrap();
    let tree = &r.octree;
    let _voxel_count = r.voxel_count;

    let vs = tree.base_voxel_size();
    let _extent = tree.extent() as f32 * vs;
    let mut mismatches = 0u32;
    let mut total = 0u32;

    for (coord, slot, leaf_depth) in tree.iter_leaves() {
        total += 1;
        let depth_diff = tree.depth() - leaf_depth;
        let leaf_vs = vs * (1u32 << depth_diff) as f32;
        // Position at center of the leaf voxel
        let pos = glam::Vec3::new(
            coord.x as f32 * vs + leaf_vs * 0.5,
            coord.y as f32 * vs + leaf_vs * 0.5,
            coord.z as f32 * vs + leaf_vs * 0.5,
        );

        let (gpu_slot, _gpu_depth) = gpu_style_lookup(&tree, pos);
        let (coord_node, _) = tree.lookup_with_depth(coord).unwrap();
        let coord_slot = if is_leaf(coord_node) { leaf_slot(coord_node) } else { coord_node };

        if gpu_slot != slot {
            if mismatches < 5 {
                eprintln!(
                    "MISMATCH at coord={:?} pos={:?}: coord_lookup_slot={} gpu_slot={} (expected {})",
                    coord, pos, coord_slot, gpu_slot, slot
                );
            }
            mismatches += 1;
        }
    }

    eprintln!("GPU lookup test: {total} leaves, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "{mismatches}/{total} leaves unreachable by GPU-style position lookup");
}

#[test]
fn gpu_lookup_matches_rkp_file() {
    // Test with an actual .rkp file if available.
    let path = "/home/joe/dev/rkifield_game/splat5/assets/models/bunny_pbr/scene.rkp";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping .rkp test — file not found: {path}");
        return;
    }

    let mut file = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(&mut file);
    let header = match crate::asset_file::read_rkp_header(&mut reader) {
        Ok(h) => h,
        Err(e) => { eprintln!("Skipping .rkp test — header error: {e}"); return; }
    };
    let octree_nodes = crate::asset_file::read_rkp_octree(&mut reader, &header).unwrap();

    let depth = header.octree_depth as u8;
    let vs = header.base_voxel_size;
    let tree = SparseOctree::from_raw(&octree_nodes, depth, vs);

    let _voxel_data = crate::asset_file::read_rkp_voxels(&mut reader, &header).unwrap();

    let extent = tree.extent() as f32 * vs;
    let mut mismatches = 0u32;
    let mut total = 0u32;

    for (coord, slot, leaf_depth) in tree.iter_leaves() {
        total += 1;
        let depth_diff = tree.depth() - leaf_depth;
        let leaf_vs = vs * (1u32 << depth_diff) as f32;
        let pos = glam::Vec3::new(
            coord.x as f32 * vs + leaf_vs * 0.5,
            coord.y as f32 * vs + leaf_vs * 0.5,
            coord.z as f32 * vs + leaf_vs * 0.5,
        );

        let (gpu_slot, _) = gpu_style_lookup(&tree, pos);
        if gpu_slot != slot {
            if mismatches < 10 {
                eprintln!(
                    "MISMATCH coord={:?} pos={:?}: expected slot={} got gpu_slot={}",
                    coord, pos, slot, gpu_slot
                );
            }
            mismatches += 1;
        }
    }

    eprintln!("GPU lookup .rkp test: {total} leaves, {mismatches} mismatches (extent={extent}, depth={depth}, vs={vs})");
    assert_eq!(mismatches, 0, "{mismatches}/{total} leaves unreachable by GPU-style lookup");
}

#[test]
fn iter_leaves_empty() {
    let tree = SparseOctree::new(3, 0.1);
    assert_eq!(tree.iter_leaves().count(), 0);
}

#[test]
fn iter_leaves_collects_all() {
    let mut tree = SparseOctree::new(2, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 10);
    tree.insert(UVec3::new(3, 3, 3), 20);
    tree.insert(UVec3::new(1, 2, 0), 30);

    let mut leaves: Vec<_> = tree.iter_leaves().collect();
    leaves.sort_by_key(|&(coord, slot, _)| (coord.z, coord.y, coord.x, slot));

    assert_eq!(leaves.len(), 3);
    assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(0, 0, 0) && s == 10));
    assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(3, 3, 3) && s == 20));
    assert!(leaves.iter().any(|&(c, s, _)| c == UVec3::new(1, 2, 0) && s == 30));
}

#[test]
fn from_brick_map_roundtrip() {
    let mut map = BrickMap::new(UVec3::new(4, 4, 4));
    map.set(0, 0, 0, 10);
    map.set(3, 3, 3, 20);
    map.set(1, 2, 3, 30);
    map.set(2, 2, 2, INTERIOR_SLOT);

    let tree = SparseOctree::from_brick_map(&map, 0.1);

    // Verify all lookups match the original map.
    for bz in 0..4 {
        for by in 0..4 {
            for bx in 0..4 {
                let map_val = map.get(bx, by, bz).unwrap();
                let tree_val = tree.lookup(UVec3::new(bx, by, bz)).unwrap();
                match map_val {
                    EMPTY_SLOT => assert_eq!(tree_val, EMPTY_NODE,
                        "mismatch at ({bx},{by},{bz}): map=EMPTY, tree={tree_val:#x}"),
                    INTERIOR_SLOT => assert_eq!(tree_val, INTERIOR_NODE,
                        "mismatch at ({bx},{by},{bz}): map=INTERIOR, tree={tree_val:#x}"),
                    slot => assert_eq!(tree_val, make_leaf(slot),
                        "mismatch at ({bx},{by},{bz}): map={slot}, tree={tree_val:#x}"),
                }
            }
        }
    }
}

#[test]
fn from_brick_map_non_power_of_two() {
    // BrickMap dims that aren't a power of 2 — octree rounds up.
    let mut map = BrickMap::new(UVec3::new(3, 5, 2));
    map.set(2, 4, 1, 42);

    let tree = SparseOctree::from_brick_map(&map, 0.1);
    assert!(tree.extent() >= 5); // must cover the largest dim
    assert_eq!(tree.lookup(UVec3::new(2, 4, 1)), Some(make_leaf(42)));
}

#[test]
fn extent_world() {
    let tree = SparseOctree::new(3, 0.1);
    // 2^3 = 8 voxels per axis, each voxel 0.1 → 8 * 0.1 = 0.8
    assert!((tree.extent_world() - 0.8).abs() < 1e-6);
}

#[test]
fn leaf_and_branch_encoding() {
    assert!(is_leaf(make_leaf(0)));
    assert!(is_leaf(make_leaf(42)));
    assert!(is_leaf(make_leaf(0x3FFF_FFFD))); // max leaf_attr_id (30 bits - 2 reserved)
    assert!(!is_leaf(EMPTY_NODE));
    assert!(!is_leaf(INTERIOR_NODE));
    assert!(!is_leaf(make_brick(0)));
    assert!(!is_leaf(make_brick(42)));

    assert!(is_branch(0)); // offset 0 is a valid branch
    assert!(is_branch(100));
    assert!(!is_branch(EMPTY_NODE));
    assert!(!is_branch(INTERIOR_NODE));
    assert!(!is_branch(make_leaf(0)));
    assert!(!is_branch(make_brick(0)));
}

#[test]
fn brick_encoding() {
    assert!(is_brick(make_brick(0)));
    assert!(is_brick(make_brick(42)));
    assert!(is_brick(make_brick(0x3FFF_FFFD)));
    assert!(!is_brick(EMPTY_NODE));
    assert!(!is_brick(INTERIOR_NODE));
    assert!(!is_brick(make_leaf(0)));
    assert!(!is_brick(0)); // branch
}

#[test]
fn leaf_slot_roundtrip() {
    for slot in [0u32, 1, 42, 1000, 0x3FFF_FFFD] {
        assert_eq!(leaf_slot(make_leaf(slot)), slot);
    }
}

#[test]
fn brick_id_roundtrip() {
    for id in [0u32, 1, 42, 1000, 0x3FFF_FFFD] {
        assert_eq!(brick_id(make_brick(id)), id);
    }
}

#[test]
#[should_panic]
fn insert_out_of_bounds_panics() {
    let mut tree = SparseOctree::new(2, 0.1); // 4x4x4
    tree.insert(UVec3::new(4, 0, 0), 1);
}

#[test]
fn depth_zero_single_node() {
    // A depth-0 tree can't store any brick coordinates (extent = 1).
    // Actually extent is 2^0 = 1, so coord (0,0,0) is valid.
    let mut tree = SparseOctree::new(1, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 5);
    assert_eq!(tree.lookup(UVec3::new(0, 0, 0)), Some(make_leaf(5)));
}

#[test]
fn many_inserts_no_panic() {
    let mut tree = SparseOctree::new(4, 0.1); // 16x16x16
    let mut count = 0;
    for z in 0..16u32 {
        for y in 0..16u32 {
            for x in 0..16u32 {
                // Sparse: only insert ~10% of positions.
                if (x + y * 3 + z * 7) % 10 == 0 {
                    tree.insert(UVec3::new(x, y, z), count);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(tree.leaf_count(), count as usize);
}

#[test]
fn live_node_count_excludes_dead_space() {
    let mut tree = SparseOctree::new(1, 0.1); // 2x2x2
    // Fill all 8 positions with same slot to trigger collapse.
    for z in 0..2u32 {
        for y in 0..2u32 {
            for x in 0..2u32 {
                tree.insert(UVec3::new(x, y, z), 99);
            }
        }
    }
    // Root collapsed to a single leaf — node_count includes dead children,
    // but live_node_count should be 1.
    assert_eq!(tree.nodes[0], make_leaf(99));
    assert_eq!(tree.live_node_count(), 1);
    assert!(tree.node_count() >= 1); // may have dead space
}

#[test]
fn from_brick_map_all_interior() {
    let mut map = BrickMap::new(UVec3::new(2, 2, 2));
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                map.set(x, y, z, INTERIOR_SLOT);
            }
        }
    }
    let tree = SparseOctree::from_brick_map(&map, 0.1);
    // Should collapse to a single INTERIOR root.
    assert_eq!(tree.nodes[0], INTERIOR_NODE);
    // live_node_count excludes dead children from collapsed branches.
    assert_eq!(tree.live_node_count(), 1);
}

#[test]
fn from_brick_map_all_empty() {
    let map = BrickMap::new(UVec3::new(4, 4, 4));
    let tree = SparseOctree::from_brick_map(&map, 0.1);
    assert_eq!(tree.nodes[0], EMPTY_NODE);
    assert_eq!(tree.node_count(), 1);
}

// ── internal_attr_index (prefiltered LOD) scaffolding tests ───────────
//
// These tests exercise only the parallel-buffer maintenance. No real
// prefilter pass populates these ids yet (Phase 1). The property
// we verify here is: *whatever* we write into `internal_attr_index`
// at a branch slot survives the rewriting passes (compact, dedup,
// morton) and ends up at the corresponding branch slot in the new
// buffer. The prefilter pass in Phase 1 will rely on this invariant
// (it'll seed values then run the passes).

/// Seed every branch slot in the tree with a cookie value; leave
/// non-branch slots untouched.
fn seed_branch_prefilters(tree: &mut SparseOctree, cookie: u32) {
    for i in 0..tree.node_count() {
        let node = tree.as_slice()[i];
        if is_branch(node) {
            tree.set_internal_attr(i as u32, cookie);
        }
    }
}

/// Assert that every branch slot in the tree carries the given cookie.
fn assert_branch_prefilters_match(tree: &SparseOctree, cookie: u32) {
    let mut checked = 0usize;
    for i in 0..tree.node_count() {
        let node = tree.as_slice()[i];
        if is_branch(node) {
            assert_eq!(
                tree.internal_attr(i as u32),
                cookie,
                "branch at slot {i} lost prefilter-id (got {:#x}, expected {cookie:#x})",
                tree.internal_attr(i as u32),
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "test is vacuous — tree has no branches");
}

#[test]
fn new_tree_has_sentinel_filled_internal_attr() {
    let tree = SparseOctree::new(3, 0.1);
    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_eq!(tree.internal_attr(0), INTERNAL_ATTR_NONE);
}

#[test]
fn from_raw_fills_internal_attr_sentinels() {
    let raw = vec![make_leaf(1), EMPTY_NODE, INTERIOR_NODE];
    let tree = SparseOctree::from_raw(&raw, 2, 0.1);
    assert_eq!(tree.internal_attr_slice().len(), 3);
    for &a in tree.internal_attr_slice() {
        assert_eq!(a, INTERNAL_ATTR_NONE);
    }
}

#[test]
fn insert_grows_internal_attr_in_lockstep() {
    let mut tree = SparseOctree::new(3, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 10);
    tree.insert(UVec3::new(7, 7, 7), 20);
    tree.insert(UVec3::new(3, 4, 5), 30);

    // Lockstep length invariant.
    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    // Freshly-allocated slots default to sentinel.
    for &a in tree.internal_attr_slice() {
        assert_eq!(a, INTERNAL_ATTR_NONE);
    }
}

#[test]
fn internal_attr_set_get_roundtrip() {
    let mut tree = SparseOctree::new(2, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(3, 3, 3), 2);

    // Find one branch slot and set/get.
    let branch_idx = (0..tree.node_count())
        .find(|&i| is_branch(tree.as_slice()[i]))
        .expect("should have at least one branch");
    tree.set_internal_attr(branch_idx as u32, 0xDEAD_BEEF);
    assert_eq!(tree.internal_attr(branch_idx as u32), 0xDEAD_BEEF);
}

#[test]
fn compact_preserves_internal_attr_at_branches() {
    let mut tree = SparseOctree::new(3, 0.1);
    // Distinct-leaf inserts so the tree has real branches at multiple
    // levels (not a uniform-collapse case).
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(7, 0, 0), 2);
    tree.insert(UVec3::new(0, 7, 0), 3);
    tree.insert(UVec3::new(0, 0, 7), 4);
    tree.insert(UVec3::new(7, 7, 7), 5);

    seed_branch_prefilters(&mut tree, 0xCAFEBABE);
    tree.compact();

    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_branch_prefilters_match(&tree, 0xCAFEBABE);
}

#[test]
fn dedup_preserves_internal_attr_at_branches() {
    // Same shape as compact test — plus dedup pass.
    let mut tree = SparseOctree::new(3, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(7, 0, 0), 2);
    tree.insert(UVec3::new(0, 7, 0), 3);
    tree.insert(UVec3::new(0, 0, 7), 4);
    tree.insert(UVec3::new(7, 7, 7), 5);

    tree.compact();
    seed_branch_prefilters(&mut tree, 0xABCD_0001);
    tree.deduplicate_subtrees();

    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_branch_prefilters_match(&tree, 0xABCD_0001);
}

#[test]
fn morton_preserves_internal_attr_at_branches() {
    let mut tree = SparseOctree::new(3, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(7, 7, 7), 2);
    tree.insert(UVec3::new(3, 4, 5), 3);

    tree.compact();
    tree.deduplicate_subtrees();
    seed_branch_prefilters(&mut tree, 0x12345678);
    tree.morton_reorder();

    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_branch_prefilters_match(&tree, 0x12345678);
}

#[test]
fn full_pipeline_preserves_internal_attr() {
    let mut tree = SparseOctree::new(4, 0.1); // 16³
    // Several widely-separated leaves to force branches at multiple depths.
    for (x, y, z) in [
        (0, 0, 0), (15, 15, 15), (0, 15, 0), (15, 0, 15),
        (7, 7, 7), (8, 8, 8), (3, 4, 5), (12, 11, 10),
    ] {
        tree.insert(UVec3::new(x, y, z), (x * 100 + y * 10 + z + 1) as u32);
    }

    // Seed *after* insert (when buffer is stable) and *before* the
    // rewriting passes — this is the exact order the prefilter pass
    // will use in Phase 1.
    seed_branch_prefilters(&mut tree, 0xF00D_F00D);

    tree.compact();
    tree.deduplicate_subtrees();
    tree.morton_reorder();

    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_branch_prefilters_match(&tree, 0xF00D_F00D);
}

#[test]
fn dag_shared_subtrees_share_internal_attr() {
    // After dedup, two parent branches can reference the same 8-child
    // block. Verify that block's prefilter-ids survive the share.
    // Uniform-subtree pattern: every root octant contains the same
    // 8-leaf block → dedup collapses them to one.
    let mut tree = SparseOctree::new(2, 0.1); // 4³
    // Leaf=99 at every (x,y,z) where x+y+z is even; empty otherwise.
    // This gives each root octant an identical 8-child sub-block.
    for z in 0..4u32 {
        for y in 0..4u32 {
            for x in 0..4u32 {
                if (x + y + z) % 2 == 0 {
                    tree.insert(UVec3::new(x, y, z), 99);
                }
            }
        }
    }

    tree.compact();
    seed_branch_prefilters(&mut tree, 0xAAAA_5555);
    tree.deduplicate_subtrees();
    tree.morton_reorder();

    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    // Every surviving branch in the DAG carries the seeded cookie.
    // (If a branch got dropped by DAG collapse, the remaining branches
    // still hold a valid cookie — which is what the shader needs.)
    assert_branch_prefilters_match(&tree, 0xAAAA_5555);
}

#[test]
fn trivial_root_rewrites_keep_parallel_buffer_consistent() {
    // Fully uniform insert → try_collapse reduces to single leaf at
    // root. compact/dedup/morton all take the trivial-root fast path;
    // internal_attr_index should also truncate to 1.
    let mut tree = SparseOctree::new(2, 0.1);
    for z in 0..4u32 {
        for y in 0..4u32 {
            for x in 0..4u32 {
                tree.insert(UVec3::new(x, y, z), 42);
            }
        }
    }
    assert!(tree.node_count() > 1, "precondition: has orphan tail");

    tree.compact();
    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_eq!(tree.node_count(), 1);

    tree.deduplicate_subtrees();
    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_eq!(tree.node_count(), 1);

    tree.morton_reorder();
    assert_eq!(tree.internal_attr_slice().len(), tree.node_count());
    assert_eq!(tree.node_count(), 1);
}

#[test]
#[should_panic(expected = "internal_attr_index length must match nodes length")]
fn set_internal_attr_index_length_mismatch_panics() {
    let mut tree = SparseOctree::new(2, 0.1);
    tree.insert(UVec3::new(0, 0, 0), 1);
    tree.insert(UVec3::new(3, 3, 3), 2);
    // Deliberately wrong length.
    tree.set_internal_attr_index(vec![INTERNAL_ATTR_NONE; 1]);
}
