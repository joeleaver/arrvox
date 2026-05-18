use super::*;
use super::cpu_reference::morton_30;
use crate::rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance};

fn entry(min: [f32; 3], max: [f32; 3], asset: u32, live: u32) -> InstanceTileCullEntry {
    InstanceTileCullEntry {
        aabb_min: min,
        asset_id: asset,
        aabb_max: max,
        instance_state_offset: 0,
        material_id: 0,
        live,
        _pad0: 0,
        _pad1: 0,
    }
}

#[test]
fn tlas_prim_size_is_48() {
    assert_eq!(std::mem::size_of::<TlasPrim>(), 48);
}

#[test]
fn user_shader_filters_dead_entries() {
    let scratch = vec![
        entry([0.0; 3], [1.0; 3], 1, 1),
        entry([2.0; 3], [3.0; 3], 2, 0), // dead — skipped
        entry([4.0; 3], [5.0; 3], 3, 1),
    ];
    let (out, count) = cpu_reference_assemble_user_shader(&scratch, 16);
    assert_eq!(count, 2);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].asset_id, 1);
    assert_eq!(out[1].asset_id, 3);
    for p in &out {
        assert_eq!(p.instance_index, TLAS_LEAF_USER_SHADER);
    }
}

#[test]
fn user_shader_filters_degenerate_aabb() {
    // live=1 but zero-volume → filtered.
    let scratch = vec![
        entry([0.0; 3], [0.0; 3], 1, 1),  // zero extent
        entry([0.0; 3], [1.0; 3], 2, 1),  // valid
    ];
    let (out, count) = cpu_reference_assemble_user_shader(&scratch, 16);
    assert_eq!(count, 1);
    assert_eq!(out[0].asset_id, 2);
}

#[test]
fn user_shader_capacity_overflow_drops_excess() {
    let scratch = vec![
        entry([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1, 1),
        entry([2.0, 2.0, 2.0], [3.0, 3.0, 3.0], 2, 1),
        entry([4.0, 4.0, 4.0], [5.0, 5.0, 5.0], 3, 1),
    ];
    let (out, count) = cpu_reference_assemble_user_shader(&scratch, 2);
    // count reflects ALL writes attempted (including overflow);
    // out only carries those that fit.
    assert_eq!(count, 3);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].asset_id, 1);
    assert_eq!(out[1].asset_id, 2);
}

fn make_asset(min: [f32; 3], max: [f32; 3], shader_id: u32) -> RkpGpuAsset {
    RkpGpuAsset {
        aabb_min: min,
        octree_root: 0,
        aabb_max: max,
        octree_depth: 0,
        octree_extent_bits: 0,
        voxel_size: 0.0,
        geom_type: 0,
        bone_count: 0,
        grid_origin: [0.0; 3],
        rest_octree_root: 0,
        rest_octree_depth: 0,
        rest_octree_extent_bits: 0,
        shader_id,
        _pad: 0,
    }
}

fn make_instance(asset_id: u32, world: [[f32; 4]; 4], material: u32) -> RkpGpuInstance {
    RkpGpuInstance {
        world,
        asset_id,
        material_id: material,
        object_id: 0,
        layer_mask: 0xFFFF_FFFF,
        is_skinned: 0,
        bone_buffer_offset: 0,
        overlay_offset: 0,
        overlay_count: 0,
        sculpt_offset: 0,
        sculpt_count: 0,
        _pad: [0; 2],
    }
}

#[test]
fn host_skips_user_shader_assets() {
    let assets = vec![
        make_asset([0.0; 3], [1.0; 3], 0), // host
        make_asset([0.0; 3], [1.0; 3], 7), // user-shader proto
    ];
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let instances = vec![make_instance(0, identity, 1), make_instance(1, identity, 2)];
    let (out, count) = cpu_reference_assemble_host(&instances, &assets, 16);
    assert_eq!(count, 1);
    assert_eq!(out[0].asset_id, 0);
    assert_eq!(out[0].material_id, 1);
    assert_eq!(out[0].instance_index, 0);
}

#[test]
fn host_transforms_aabb_with_translation() {
    let assets = vec![make_asset([0.0; 3], [1.0; 3], 0)];
    let mut t = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [10.0, 20.0, 30.0, 1.0],
    ];
    // (matches column-major layout for `world[col][row]`)
    let _ = t;
    t = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [10.0, 20.0, 30.0, 1.0],
    ];
    let instances = vec![make_instance(0, t, 0)];
    let (out, _) = cpu_reference_assemble_host(&instances, &assets, 16);
    assert_eq!(out[0].aabb_min, [10.0, 20.0, 30.0]);
    assert_eq!(out[0].aabb_max, [11.0, 21.0, 31.0]);
    assert_eq!(out[0].instance_index, 0);
}

#[test]
fn ensure_prims_capacity_doubles_until_fit() {
    // CPU-only sanity check on the doubling logic. No GPU
    // device — just exercise the field math by mirroring it.
    let mut cap: u32 = 1;
    let target = 17u32;
    while cap < target {
        cap = cap.saturating_mul(2);
    }
    assert_eq!(cap, 32);
}

fn make_prim(min: [f32; 3], max: [f32; 3]) -> TlasPrim {
    TlasPrim {
        aabb_min: min,
        asset_id: 0,
        aabb_max: max,
        instance_state_offset: 0,
        material_id: 0,
        instance_index: 0,
        _pad0: 0,
        _pad1: 0,
    }
}

#[test]
fn scene_aabb_handles_empty_input() {
    let (mn, mx) = scene_aabb_from_prims(&[]);
    assert_eq!(mn, [0.0; 3]);
    assert_eq!(mx, [1.0; 3]);
}

#[test]
fn scene_aabb_unions_multiple_prims() {
    let prims = vec![
        make_prim([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        make_prim([-2.0, 5.0, -3.0], [-1.0, 6.0, -2.0]),
        make_prim([3.0, 0.5, 0.5], [3.5, 1.5, 1.5]),
    ];
    let (mn, mx) = scene_aabb_from_prims(&prims);
    assert_eq!(mn, [-2.0, 0.0, -3.0]);
    assert_eq!(mx, [3.5, 6.0, 1.5]);
}

#[test]
fn morton_30_interleaves_correctly() {
    // x=1 (001), y=2 (010), z=3 (011). Morton interleaves with z
    // at bits 0,3,6; y at 1,4,7; x at 2,5,8 (the WGSL writes
    // `(expand(x) << 2) | (expand(y) << 1) | expand(z)`).
    //   bit 0 (z0)=1, bit 1 (y0)=0, bit 2 (x0)=1,
    //   bit 3 (z1)=1, bit 4 (y1)=1, bit 5 (x1)=0,
    //   bit 6+ all zero.
    // = 1 + 4 + 8 + 16 = 29.
    let m = morton_30(1, 2, 3);
    assert_eq!(m, 29);
}

#[test]
fn morton_preserves_locality_along_x() {
    // Centroids at (0,0,0), (1,0,0), (2,0,0) should produce
    // strictly increasing Mortons (since z and y stay 0; x
    // increments → bit 2, 5, 8 etc. flip).
    let prims: Vec<TlasPrim> = (0..4)
        .map(|i| {
            let f = i as f32;
            make_prim([f, 0.0, 0.0], [f + 0.1, 0.1, 0.1])
        })
        .collect();
    let scene_min = [0.0, 0.0, 0.0];
    let scene_max = [10.0, 1.0, 1.0];
    let pairs = cpu_reference_morton(&prims, scene_min, scene_max);
    for w in pairs.windows(2) {
        assert!(
            w[0].0 < w[1].0,
            "Morton not strictly increasing along x: {} -> {}",
            w[0].0,
            w[1].0,
        );
    }
}

#[test]
fn cpu_reference_radix_sort_sorts_pairs() {
    let unsorted: Vec<(u32, u32)> = vec![(7, 0), (3, 1), (5, 2), (3, 3), (9, 4)];
    let sorted = cpu_reference_radix_sort(&unsorted);
    assert_eq!(sorted, vec![(3, 1), (3, 3), (5, 2), (7, 0), (9, 4)]);
}

#[test]
fn karras_two_leaves_root_has_two_leaf_children() {
    // N=2: one internal node at idx 0, two leaf-markers at 1, 2.
    // Both children of root must be leaf-markers.
    let keys = [0b01u32, 0b10u32];
    let (l, r) = cpu_reference_karras_node(&keys, 0);
    assert_eq!(l, 1, "left child = leaf-marker for leaf 0");
    assert_eq!(r, 2, "right child = leaf-marker for leaf 1");
}

#[test]
fn karras_three_leaves_balanced_tree() {
    // Mortons [1, 2, 4]: distinct, ascending. Expected topology
    // (matches the trace I worked out by hand):
    //     [0] internal: left=internal[1], right=leaf-marker[4]
    //     [1] internal: left=leaf-marker[2], right=leaf-marker[3]
    let keys = [1u32, 2, 4];
    let (l0, r0) = cpu_reference_karras_node(&keys, 0);
    let (l1, r1) = cpu_reference_karras_node(&keys, 1);
    // Internal node 0
    assert_eq!(l0, 1, "node 0 left = internal 1");
    assert_eq!(r0, 4, "node 0 right = leaf-marker 2 (= idx 4)");
    // Internal node 1
    assert_eq!(l1, 2, "node 1 left = leaf-marker 0 (= idx 2)");
    assert_eq!(r1, 3, "node 1 right = leaf-marker 1 (= idx 3)");
}

#[test]
fn karras_four_leaves_two_subtrees() {
    // Mortons [0b00, 0b01, 0b10, 0b11]: full binary partition
    // expected by the algorithm — root splits 2-2.
    let keys = [0u32, 1, 2, 3];
    // Internal indices 0..2. Verify each.
    let (l0, r0) = cpu_reference_karras_node(&keys, 0);
    let (l1, r1) = cpu_reference_karras_node(&keys, 1);
    let (l2, r2) = cpu_reference_karras_node(&keys, 2);
    // Topology from a balanced 4-leaf Karras tree:
    //     [0] internal:  left=internal[1], right=internal[2]
    //     [1] internal:  left=leaf[3], right=leaf[4]   (= leaves 0, 1)
    //     [2] internal:  left=leaf[5], right=leaf[6]   (= leaves 2, 3)
    // Leaf-marker offset = N-1 = 3.
    assert_eq!((l0, r0), (1, 2));
    assert_eq!((l1, r1), (3, 4));
    assert_eq!((l2, r2), (5, 6));
}

#[test]
fn karras_handles_duplicate_mortons() {
    // Duplicate-Morton case — algorithm relies on the index
    // tiebreak in `delta()`. Algorithm should still produce a
    // valid (topology-wise) tree.
    let keys = [5u32, 5, 5, 5];
    // For 4 identical Mortons, every internal node's delta_min
    // is decided purely by the index tiebreak. Tree should
    // still reach all 4 leaves exactly once.
    let mut leaf_visits = [0u32; 4];
    let mut visit = vec![false; 7]; // 4 leaves + 3 internal = 7 nodes
    // Walk from node 0 (root).
    let mut stack = vec![0u32];
    let n = 4u32;
    while let Some(idx) = stack.pop() {
        assert!(!visit[idx as usize], "node {idx} visited twice — cycle");
        visit[idx as usize] = true;
        if idx >= n - 1 {
            let leaf_idx = idx - (n - 1);
            leaf_visits[leaf_idx as usize] += 1;
            continue;
        }
        let (l, r) = cpu_reference_karras_node(&keys, idx as i32);
        assert!(l < 2 * n - 1, "child {l} out of range for n={n}");
        assert!(r < 2 * n - 1, "child {r} out of range");
        stack.push(l);
        stack.push(r);
    }
    for (i, &count) in leaf_visits.iter().enumerate() {
        assert_eq!(count, 1, "leaf {i} visited {count} times (expected once)");
    }
}

#[test]
fn karras_random_eight_leaves_visits_all() {
    // 8 distinct ascending Mortons → full balanced tree, all
    // leaves reachable from root.
    let keys: Vec<u32> = (0..8u32).map(|i| i * 0x100 + 0x42).collect();
    let n = keys.len() as u32;
    let mut leaf_visits = vec![0u32; n as usize];
    let mut visit = vec![false; (2 * n - 1) as usize];
    let mut stack = vec![0u32];
    while let Some(idx) = stack.pop() {
        assert!(!visit[idx as usize]);
        visit[idx as usize] = true;
        if idx >= n - 1 {
            leaf_visits[(idx - (n - 1)) as usize] += 1;
            continue;
        }
        let (l, r) = cpu_reference_karras_node(&keys, idx as i32);
        stack.push(l);
        stack.push(r);
    }
    assert!(leaf_visits.iter().all(|&c| c == 1));
}
