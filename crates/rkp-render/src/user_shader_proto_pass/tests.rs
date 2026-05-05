use super::*;

fn assert_wgsl_valid(source: &str, label: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!("[{label}] parse error:\n{}", e.emit_to_string(source))
    });
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
}

#[test]
fn octree_node_count_matches_geometric_series() {
    // Sum 1 + 8 + 64 + ... + 8^d = (8^(d+1) - 1) / 7
    assert_eq!(octree_node_count_for_depth(0), 1);
    assert_eq!(octree_node_count_for_depth(1), 9);
    assert_eq!(octree_node_count_for_depth(2), 73);
    assert_eq!(octree_node_count_for_depth(3), 585);
    assert_eq!(octree_node_count_for_depth(4), 4681);
}

#[test]
fn level_starts_are_cumulative_sizes() {
    // For max_depth=2: [0, 1, 9, 73, 585]
    // (level 0 starts at 0, level 1 at 1, level 2 at 9, leaf-level
    // ends at 73 which is the total).
    let lv = level_starts_inclusive(2);
    assert_eq!(lv, vec![0, 1, 9, 73]);
    let lv = level_starts_inclusive(3);
    assert_eq!(lv, vec![0, 1, 9, 73, 585]);
}

#[test]
fn max_bricks_and_leaf_attrs_at_depth() {
    // 8^max_depth bricks, 64 cells each.
    assert_eq!(max_bricks_for_depth(0), 1);
    assert_eq!(max_bricks_for_depth(2), 64);
    assert_eq!(max_bricks_for_depth(4), 4096);
    assert_eq!(max_leaf_attrs_for_depth(2), 64 * 64);
    assert_eq!(max_leaf_attrs_for_depth(4), 64 * 4096);
}

#[test]
fn octree_alloc_is_exact_size() {
    // No bucket clamping — the octree extent equals the dense
    // spine count for the requested depth, byte-for-byte.
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let (entry, _) = cache.lookup_or_allocate(1, 0, 2).unwrap();
    assert_eq!(entry.octree_extent.1, octree_node_count_for_depth(2));
}

#[test]
fn build_internal_levels_layout_for_depth_2() {
    // pool_octree_base = 1000, block_offset = 50.
    // Block root = 1050.
    // levels for depth 2: [0, 1, 9, 73].
    // Total nodes: 73.
    // Level 0 (1 node at slot 0): value = 1050 + 1 = 1051
    // Level 1 (8 nodes at slots 1..9): values = 1050 + 9 + i*8 for i in 0..8
    //   → 1059, 1067, 1075, 1083, 1091, 1099, 1107, 1115
    // Level 2 (64 nodes at slots 9..73): all OCTREE_EMPTY
    let nodes = build_internal_levels(1000, 50, 2);
    assert_eq!(nodes.len(), 73);
    assert_eq!(nodes[0], [1051, INTERNAL_ATTR_NONE, 0, 0]);
    for i in 0..8u32 {
        assert_eq!(
            nodes[1 + i as usize],
            [1050 + 9 + i * 8, INTERNAL_ATTR_NONE, 0, 0],
            "level-1 node {i} mismatch",
        );
    }
    for (idx, node) in nodes.iter().enumerate().skip(9) {
        assert_eq!(
            *node,
            [OCTREE_EMPTY, INTERNAL_ATTR_NONE, 0, 0],
            "leaf-level slot {idx} should start empty",
        );
    }
}

#[test]
fn build_internal_levels_root_only_for_depth_0() {
    let nodes = build_internal_levels(0, 0, 0);
    // depth 0: only the leaf level exists, 1 node, EMPTY.
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0], [OCTREE_EMPTY, INTERNAL_ATTR_NONE, 0, 0]);
}

#[test]
fn cache_first_lookup_is_dirty() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let (entry, dirty) = cache.lookup_or_allocate(1, 0xDEAD_BEEFu64, 2).unwrap();
    assert!(dirty);
    assert_eq!(entry.shader_id, 1);
    assert_eq!(entry.source_hash, 0xDEAD_BEEFu64);
    assert_eq!(entry.max_depth, 2);
}

#[test]
fn cache_repeat_lookup_with_same_hash_is_clean() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let _ = cache.lookup_or_allocate(1, 0xDEAD, 2).unwrap();
    let (_, dirty) = cache.lookup_or_allocate(1, 0xDEAD, 2).unwrap();
    assert!(!dirty);
}

#[test]
fn cache_source_change_re_dirties_without_re_allocating() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    let oct_hw_after_first = cache.octree_high_water();
    let (e2, dirty) = cache.lookup_or_allocate(1, 0xBBBB, 2).unwrap();
    assert!(dirty);
    // Same depth → same octree extent reused, no fresh bump.
    assert_eq!(e1.octree_extent, e2.octree_extent);
    assert_eq!(cache.octree_high_water(), oct_hw_after_first);
}

#[test]
fn cache_distinct_shader_ids_get_distinct_extents() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    let (e2, _) = cache.lookup_or_allocate(2, 0xBBBB, 2).unwrap();
    assert_ne!(e1.octree_extent.0, e2.octree_extent.0);
}

#[test]
fn cache_evicts_untouched_entries() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    let _ = cache.lookup_or_allocate(2, 0xBBBB, 2).unwrap();
    assert_eq!(cache.entry_count(), 2);
    cache.begin_frame();
    // Touch only shader 1 this frame.
    let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    cache.evict_untouched();
    assert_eq!(cache.entry_count(), 1);
    assert!(cache.get(1).is_some());
    assert!(cache.get(2).is_none());
}

#[test]
fn cache_depth_change_reallocs_octree() {
    // Depth-4 octree spine is 4681 nodes — bigger than depth 2's
    // 73, so the new extent occupies a fresh range and the old
    // depth-2 extent goes onto the free list.
    let mut cache = PrototypeCache::with_capacities(20_000, 8192, 200_000);
    cache.set_pool_bases(0, 0, 0);
    let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    let (e2, dirty) = cache.lookup_or_allocate(1, 0xAAAA, 4).unwrap();
    assert!(dirty);
    assert_eq!(e2.max_depth, 4);
    assert_eq!(e2.octree_extent.1, octree_node_count_for_depth(4));
    assert!(e2.octree_extent.1 > e1.octree_extent.1);
}

#[test]
fn cache_pool_base_change_flushes() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    cache.set_pool_bases(100, 0, 0);
    // Flush dropped the entry.
    assert_eq!(cache.entry_count(), 0);
}

#[test]
fn pool_exhaustion_returns_none() {
    // Pool sized for exactly one depth-2 spine (73 nodes); second
    // request can't fit and returns None.
    let mut cache = PrototypeCache::with_capacities(73, 64, 4096);
    cache.set_pool_bases(0, 0, 0);
    let _ = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    assert!(cache.lookup_or_allocate(2, 0xBBBB, 2).is_none());
}

#[test]
fn evicted_octree_extent_is_reused_on_realloc() {
    // shader 1 depth 2 → consumes an extent at offset 0.
    // begin_frame + evict (no touch) returns it to the free-list.
    // shader 2 depth 2 → same size → reuses offset 0; high-water
    // doesn't advance.
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(0, 0, 0);
    let (e1, _) = cache.lookup_or_allocate(1, 0xAAAA, 2).unwrap();
    let hw_after_first = cache.octree_high_water();
    cache.begin_frame();
    cache.evict_untouched();
    let (e2, _) = cache.lookup_or_allocate(2, 0xBBBB, 2).unwrap();
    assert_eq!(e2.octree_extent, e1.octree_extent);
    assert_eq!(cache.octree_high_water(), hw_after_first);
}

#[test]
fn proto_uniform_size_is_32() {
    assert_eq!(std::mem::size_of::<PrototypeUniform>(), 32);
}

#[test]
fn proto_uniform_carries_capacity_and_octree_offset() {
    let mut cache = PrototypeCache::with_capacities(10_000, 1024, 32_768);
    cache.set_pool_bases(1000, 2000, 3000);
    let (entry, _) = cache.lookup_or_allocate(7, 0xCAFE, 2).unwrap();
    let u = PrototypeUniform::from_entry(&entry, &cache);
    assert_eq!(u.shader_id, 7);
    assert_eq!(u.max_depth, 2);
    // Phase 4 — capacities are ABSOLUTE upper bounds (base +
    // reservation_size), so the bake's `id >= capacity` check
    // works against the cursor that starts at `base`.
    assert_eq!(u.brick_capacity, 2000 + 1024);
    assert_eq!(u.leaf_attr_capacity, 3000 + 32_768);
    // octree_leaf_offset = pool_octree_base + extent.0 + level_starts[max_depth]
    let level_starts = level_starts_inclusive(2);
    let expected = 1000 + entry.octree_extent.0 + level_starts[2];
    assert_eq!(u.octree_leaf_offset, expected);
}

#[test]
fn proto_shader_validates_with_empty_chunk() {
    // Empty proto chunk should still produce valid WGSL — the
    // identity stub `dispatch_user_proto` is the default.
    let source = compose_proto_source("");
    assert_wgsl_valid(&source, "user_shader_proto");
    assert!(source.contains("proto_bake_main"));
}

#[test]
fn proto_shader_validates_with_nonempty_chunk() {
    // Splice in a minimal user dispatch chunk and confirm the
    // composed source is valid WGSL. The chunk has to provide its
    // own dispatch_user_proto definition (the splice removes the
    // identity stub between the markers).
    let chunk = r#"
fn rkp_user_1_proto(uvw: vec3<f32>) -> VoxelEmit {
var v: VoxelEmit;
v.occupancy = 1u;
v.normal = vec3<f32>(0.0, 1.0, 0.0);
return v;
}
fn dispatch_user_proto(shader_id: u32, uvw: vec3<f32>) -> VoxelEmit {
switch shader_id {
    case 1u: { return rkp_user_1_proto(uvw); }
    default: { return voxel_emit_skip(); }
}
}
"#;
    let source = compose_proto_source(chunk);
    assert_wgsl_valid(&source, "user_shader_proto.spliced");
    assert!(source.contains("rkp_user_1_proto"));
}
