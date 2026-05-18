//! Phase 3 guard — paint mutations land in per-instance overlays, not
//! the asset's shared `LeafAttrPool`.
//!
//! Bug fixed: load bunny.rkp twice, paint one of the instances, both
//! visually paint because they share `octree_root` → share leaf slots
//! → share leaf-attr writes. The fix is `apply_paint_sphere` now
//! mutates a caller-owned `LeafAttrOverlay`; the asset pool is
//! immutable post-load. This test bakes a sphere, captures the pool
//! baseline, then paints into two separate overlays and asserts:
//!
//! 1. Overlay #1 receives entries, overlay #2 stays empty.
//! 2. The asset's `leaf_attr_pool` byte-content is unchanged.
//! 3. The same code path with a brush that misses every leaf writes
//!    nothing into either overlay.

use glam::{Affine3A, Vec3};
use rkp_core::scene_node::SdfPrimitive;
use rkp_core::LeafAttrOverlay;
use rkp_render::paint::PaintStamp;
use rkp_render::rkp_scene_manager::{AssetInfo, RkpSceneManager};

fn build_scene_with_sphere() -> (RkpSceneManager, AssetInfo) {
    let mut sm = RkpSceneManager::new(256_000);
    let voxel_size = 0.05; // small enough to land >0 leaves under a 0.5 m brush
    let result = sm
        .voxelize_primitive(
            &SdfPrimitive::Sphere { radius: 0.4 },
            7, // primary material
            voxel_size,
            Vec3::ONE,
            42, // object_id
        )
        .expect("voxelize_primitive failed");
    let asset_info = AssetInfo {
        spatial: result.spatial,
        voxel_size: result.voxel_size,
        aabb: result.aabb,
        grid_origin: result.grid_origin,
        voxel_count: result.voxel_count,
        leaf_attr_slot_start: result.leaf_attr_slot_start,
        leaf_attr_slot_count: result.leaf_attr_slot_count,
        has_skinning: false,
    };
    (sm, asset_info)
}

#[test]
fn paint_writes_to_overlay_not_pool() {
    let (mut sm, asset_info) = build_scene_with_sphere();
    let pool_baseline = sm.leaf_attr_pool.as_slice().to_vec();

    let mut overlay_a = LeafAttrOverlay::new();
    let overlay_b = LeafAttrOverlay::new();
    let written = sm.apply_paint_sphere(
        &asset_info,
        Affine3A::IDENTITY,
        Vec3::ZERO,        // brush center at world origin (sphere center)
        0.5,               // radius — covers the full 0.4 m sphere
        1.0,               // strength
        0.5,               // falloff
        PaintStamp::Material { material_id: 13 },
        &mut overlay_a,
    );

    assert!(written > 0, "expected the brush to hit at least one leaf");
    assert_eq!(
        overlay_a.len() as usize,
        written,
        "overlay should record one entry per painted leaf",
    );
    assert!(overlay_b.is_empty(), "the un-painted overlay must stay empty");
    assert_eq!(
        sm.leaf_attr_pool.as_slice(),
        pool_baseline.as_slice(),
        "the shared asset pool must be untouched by paint",
    );
}

#[test]
fn separate_overlays_paint_independently() {
    let (mut sm, asset_info) = build_scene_with_sphere();

    // Paint material 13 into overlay A; material 31 into overlay B —
    // simulating two instances of the same asset receiving different
    // paint strokes. Same brush footprint so the entry sets line up.
    let mut overlay_a = LeafAttrOverlay::new();
    let written_a = sm.apply_paint_sphere(
        &asset_info, Affine3A::IDENTITY, Vec3::ZERO, 0.5, 1.0, 0.5,
        PaintStamp::Material { material_id: 13 }, &mut overlay_a,
    );

    let mut overlay_b = LeafAttrOverlay::new();
    let written_b = sm.apply_paint_sphere(
        &asset_info, Affine3A::IDENTITY, Vec3::ZERO, 0.5, 1.0, 0.5,
        PaintStamp::Material { material_id: 31 }, &mut overlay_b,
    );

    assert_eq!(written_a, written_b, "same brush + same asset → same hit count");
    assert_eq!(overlay_a.len(), overlay_b.len());
    // For every painted leaf, the two overlays should disagree somewhere
    // — proving they're truly independent storage. The brush sits on
    // the sphere shell (weight is in the falloff band), so the result
    // is a partial blend: primary stays the baked material, secondary
    // gets the painted id, and `material_secondary_blend` therefore
    // holds the painted id (low 12 bits). Comparing the full attr
    // avoids assuming which slot the painted id lands in.
    let identical_count = overlay_a
        .entries()
        .iter()
        .zip(overlay_b.entries())
        .filter(|(a, b)| a.attr() == b.attr())
        .count();
    assert_eq!(
        identical_count, 0,
        "overlays painting different materials should never produce identical attrs",
    );
}

#[test]
fn brush_miss_writes_nothing() {
    let (mut sm, asset_info) = build_scene_with_sphere();
    let mut overlay = LeafAttrOverlay::new();
    // Brush far outside the sphere — no leaves should be inside.
    let written = sm.apply_paint_sphere(
        &asset_info,
        Affine3A::IDENTITY,
        Vec3::new(100.0, 100.0, 100.0),
        0.05, // tiny radius, so even floating-point slop misses
        1.0,
        0.5,
        PaintStamp::Material { material_id: 1 },
        &mut overlay,
    );
    assert_eq!(written, 0);
    assert!(overlay.is_empty());
}
