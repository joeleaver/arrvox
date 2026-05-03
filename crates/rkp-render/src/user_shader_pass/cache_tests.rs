use super::*;

fn req(host: u32, mat: u32) -> ShaderRegionRequest {
    ShaderRegionRequest {
        host_object_id: host,
        material_id: mat,
        shader_name: "x".to_string(),
        params: vec![],
        aabb_min: [0.0; 3],
        aabb_max: [1.0; 3],
        cell_size: 0.25,
        input_hash: 0,
        animated: false,
        region_thickness: 0.0,
        max_depth: 4,
        painted_leaf_count: 8,
        host_octree_root: HOST_NO_HOST_SENTINEL,
        host_octree_depth: 0,
        host_octree_extent: 0.0,
        host_grid_origin: [0.0; 3],
        host_inverse_world: [[0.0; 4]; 4],
        tile_index: NO_TILE,
        is_band_region: false,
        host_surface_y: 0.0,
        painted_world_min: [0.0; 3],
        painted_world_max: [0.0; 3],
    }
}

fn small_cache() -> UserShaderObjectCache {
    // Tight test pool: 1024 octree, 256 bricks, 4096 leaf-attrs,
    // 256 fill tasks. Big enough for a handful of small regions.
    UserShaderObjectCache::with_capacities(1024, 256, 4096, 256)
}

fn small_estimate() -> PoolEstimate {
    // Small enough to fit several entries in `small_cache`.
    PoolEstimate {
        octree: 64,
        bricks: 16,
        leaf_attrs: 512,
        fill_tasks: 16,
    }
}

#[test]
fn cache_first_lookup_is_topology_and_fill_dirty() {
    let mut c = small_cache();
    let s = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
    assert!(s.topology_dirty);
    assert!(s.fill_dirty);
}

#[test]
fn cache_second_lookup_with_same_hashes_is_clean() {
    let mut c = small_cache();
    let s1 = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
    let s2 = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
    // Cache hit, both hashes match → both flags clean.
    assert!(!s2.topology_dirty);
    assert!(!s2.fill_dirty);
    // Same physical extents reused.
    assert_eq!(s1.octree_root, s2.octree_root);
    assert_eq!(s1.brick_block_offset, s2.brick_block_offset);
    assert_eq!(s1.object_id, s2.object_id);
}

#[test]
fn cache_topology_unchanged_fill_changed_yields_fill_only() {
    let mut c = small_cache();
    c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
    // Different fill hash, same topology hash.
    let s = c.lookup_or_allocate(&req(1, 1), 0xAA, 0xCC, &small_estimate()).unwrap();
    assert!(!s.topology_dirty);
    assert!(s.fill_dirty);
}

#[test]
fn cache_topology_changed_yields_full_rebake() {
    let mut c = small_cache();
    c.lookup_or_allocate(&req(1, 1), 0xAA, 0xBB, &small_estimate()).unwrap();
    let s = c.lookup_or_allocate(&req(1, 1), 0xCC, 0xBB, &small_estimate()).unwrap();
    assert!(s.topology_dirty);
    assert!(s.fill_dirty);
}

#[test]
fn cache_distinguishes_keys() {
    let mut c = small_cache();
    let s1 = c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    let s2 = c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
    // Different (object, material) → different extents.
    assert_ne!(s1.octree_root, s2.octree_root);
    assert_ne!(s1.brick_block_offset, s2.brick_block_offset);
}

#[test]
fn evict_untouched_returns_extents_to_free_list() {
    let mut c = small_cache();
    c.begin_frame();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
    let pre_brick_high = c.brick_high_water();
    // Frame 2 — only touch one of the two entries.
    c.begin_frame();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    c.evict_untouched();
    // Untouched entry's extents are now in free lists; brick
    // high-water shouldn't have advanced.
    assert_eq!(c.brick_high_water(), pre_brick_high);
    assert_eq!(c.entry_count(), 1);
    // Frame 3 — request a NEW key; should reuse a freed bucket
    // before bumping high-water.
    c.begin_frame();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    c.lookup_or_allocate(&req(1, 9), 0, 0, &small_estimate()).unwrap();
    assert_eq!(c.brick_high_water(), pre_brick_high);
}

#[test]
fn pool_exhaustion_returns_none() {
    // Pool sized for exactly one tiny region.
    let mut c = UserShaderObjectCache::with_capacities(64, 16, 512, 16);
    assert!(c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).is_some());
    // Second allocation has nothing left.
    assert!(c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).is_none());
}

#[test]
fn build_transient_objects_includes_only_touched() {
    let mut c = small_cache();
    c.begin_frame();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    c.lookup_or_allocate(&req(1, 2), 0, 0, &small_estimate()).unwrap();
    // Frame 2 — touch only one.
    c.begin_frame();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    let (assets, objs) = c.build_transient_assets_and_instances(0);
    // Only the touched entry shows up; the untouched one is
    // pending eviction at end-of-frame and shouldn't render.
    assert_eq!(objs.len(), 1);
    assert_eq!(assets.len(), 1);
}

#[test]
fn flush_on_geometry_epoch_bump() {
    let mut c = small_cache();
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    assert_eq!(c.entry_count(), 1);
    assert!(c.reconcile_epoch(1));
    assert_eq!(c.entry_count(), 0);
    // Subsequent lookup is a fresh allocation.
    let s = c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    assert!(s.topology_dirty);
}

#[test]
fn flush_on_pool_base_change() {
    let mut c = small_cache();
    c.set_pool_bases(0, 0, 0);
    c.lookup_or_allocate(&req(1, 1), 0, 0, &small_estimate()).unwrap();
    assert_eq!(c.entry_count(), 1);
    // Different bases → flush.
    c.set_pool_bases(100, 200, 300);
    assert_eq!(c.entry_count(), 0);
}

#[test]
fn allocator_rounds_up_to_next_bucket() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    // Request 17 → bucket 32.
    let (o, s) = a.alloc(17).unwrap();
    assert_eq!(o, 0);
    assert_eq!(s, 32);
    // Request exactly 32 → bucket 32.
    let (o2, s2) = a.alloc(32).unwrap();
    assert_eq!(o2, 32);
    assert_eq!(s2, 32);
    // Request 200 → bucket 256.
    let (o3, s3) = a.alloc(200).unwrap();
    assert_eq!(o3, 64);
    assert_eq!(s3, 256);
}

#[test]
fn allocator_clamps_below_min_bucket() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    // Request 1 → still get bucket 16.
    let (_, s) = a.alloc(1).unwrap();
    assert_eq!(s, 16);
}

#[test]
fn allocator_rejects_above_max_bucket() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    // Request 257 → exceeds max bucket → reject.
    assert!(a.alloc(257).is_none());
}

#[test]
fn allocator_reuses_freed_extents_per_bucket() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    let (o1, s1) = a.alloc(20).unwrap();
    assert_eq!(s1, 32);
    let pre_high = a.high_water();
    a.free(o1, s1);
    assert_eq!(a.free_count(), 1);
    // Re-alloc same bucket → reuses freed offset, doesn't bump high-water.
    let (o2, s2) = a.alloc(20).unwrap();
    assert_eq!(o2, o1);
    assert_eq!(s2, 32);
    assert_eq!(a.high_water(), pre_high);
    assert_eq!(a.free_count(), 0);
}

#[test]
fn allocator_separate_free_lists_per_bucket() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    let (o16, s16) = a.alloc(16).unwrap();
    let (_o32, s32) = a.alloc(32).unwrap();
    a.free(o16, s16);
    // Asking for 32 should NOT pop the 16-bucket free list.
    let (o32b, s32b) = a.alloc(32).unwrap();
    assert_eq!(s32b, 32);
    assert_ne!(o32b, o16);
    // Asking for 16 picks up the freed 16.
    let (o16b, s16b) = a.alloc(16).unwrap();
    assert_eq!(o16b, o16);
    assert_eq!(s16b, 16);
    let _ = (s16, s32);
}

#[test]
fn allocator_exhaustion_returns_none() {
    let mut a = BucketPoolAllocator::new(64, 16, 64);
    assert!(a.alloc(64).is_some()); // claim 0..64
    assert!(a.alloc(16).is_none());  // no room
}

#[test]
fn allocator_high_water_only_advances_on_fresh_alloc() {
    let mut a = BucketPoolAllocator::new(1024, 16, 256);
    let (_, s1) = a.alloc(20).unwrap();
    let after_first = a.high_water();
    a.free(0, s1);
    let _ = a.alloc(20).unwrap(); // should reuse, not advance
    assert_eq!(a.high_water(), after_first);
}
