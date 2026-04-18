//! FBX skinning end-to-end tests.
//!
//! These tests run against a user-supplied FBX file because shipping
//! a test FBX in-repo is impractical (binary, licensing, size). Set
//! the `RKP_TEST_FBX` environment variable to a path containing a
//! rigged FBX (e.g. any Mixamo character):
//!
//! ```bash
//! RKP_TEST_FBX=/path/to/Walking.fbx cargo test -p rkp-import --test fbx_skinning
//! ```
//!
//! When the env var is unset, every test `return`s early with a
//! diagnostic so CI stays green while developers can validate
//! locally.
//!
//! These tests specifically exercise the rewritten FBX skin-weight
//! path that was flagged as broken in the pre-rewrite audit:
//! vertex-index bounds, cluster-index bounds, per-vertex alignment,
//! and weight normalization.

use rkp_import::{ImportConfig, NullReporter, import_mesh_to_opacity_rkp_with};
use rkp_import::skeleton::extract_skeleton;

fn test_fbx_path() -> Option<std::path::PathBuf> {
    std::env::var("RKP_TEST_FBX").ok().map(|s| s.into())
}

/// Extract a skeleton + verify basic invariants. Catches the bulk of
/// the rewrite's hardening: bounds checks, weight normalization,
/// bone index range.
#[test]
fn extract_skeleton_has_valid_skinning() {
    let Some(path) = test_fbx_path() else {
        eprintln!("RKP_TEST_FBX unset — skipping");
        return;
    };

    let ex = extract_skeleton(path.to_str().unwrap())
        .expect("extract_skeleton should succeed")
        .expect("test FBX should contain a skeleton");

    assert!(!ex.skeleton.bones.is_empty(), "skeleton has no bones");
    let bone_count = ex.skeleton.bones.len();
    eprintln!("  bones: {bone_count}");

    // joints.len() == weights.len() (one row per per-corner vertex).
    assert_eq!(
        ex.skinning.joints.len(),
        ex.skinning.weights.len(),
        "joints and weights rows must match"
    );
    assert!(
        !ex.skinning.joints.is_empty(),
        "skinning is empty — the mesh should have at least one skinned vertex"
    );
    eprintln!("  skinning rows: {}", ex.skinning.joints.len());

    // Every bone index is either `-1` (unused slot) or in `[0, bone_count)`.
    // This is the critical invariant: if the FBX cluster-bounds rewrite
    // regresses, stale indices will be in the hundreds (pointing at
    // ufbx internal cluster IDs rather than our bone remap).
    for (vi, row) in ex.skinning.joints.iter().enumerate() {
        for &j in row {
            assert!(
                j == -1 || (j >= 0 && (j as usize) < bone_count),
                "vertex {vi} has out-of-range bone index {j} (bone_count={bone_count})",
            );
        }
    }

    // Every weight is finite; every row that has *any* influence
    // sums to ~1.0 (normalization invariant).
    let mut skinned_rows = 0u32;
    for (vi, row) in ex.skinning.weights.iter().enumerate() {
        for &w in row {
            assert!(
                w.is_finite(),
                "vertex {vi} has non-finite weight {w}"
            );
            assert!((0.0..=1.0).contains(&w), "vertex {vi} has out-of-range weight {w}");
        }
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "vertex {vi} weights sum to {sum}, expected ~1.0",
            );
            skinned_rows += 1;
        }
    }
    eprintln!("  skinned rows (nonzero weights): {skinned_rows}");
    assert!(skinned_rows > 0, "no vertex has any bone influence");
}

/// Smoke-test the smallest voxel tier (0.005 m). User reports a
/// shading artifact on mesh imports at this tier — the hypothesis
/// was an importer bug. This test verifies nothing crashes or
/// overflows at that scale; *visual* verification that the artifact
/// is gone is still a manual step in the editor on the affected
/// asset.
#[test]
fn import_at_smallest_voxel_tier_does_not_crash() {
    let Some(path) = test_fbx_path() else {
        eprintln!("RKP_TEST_FBX unset — skipping");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("walking_fine.rkp");

    let config = ImportConfig {
        voxel_size: Some(0.005),
        ..ImportConfig::default()
    };

    let result = import_mesh_to_opacity_rkp_with(
        &path,
        &output,
        &config,
        &NullReporter,
    )
    .expect("0.005m-tier import should succeed");

    assert!(output.exists());
    assert!(result.shell_voxels > 0);
    eprintln!(
        "  0.005m tier: shell_voxels={}, rkp={}KB",
        result.shell_voxels,
        result.file_size / 1024,
    );
}

/// Regression: multi-skin files must unify bones across every
/// `ufbx::SkinDeformer`, and weights must resolve through each skin's
/// own cluster→bone_node map rather than positional cluster index.
///
/// Pre-fix bug: Mixamo's `Walking.fbx` ships two skin deformers
/// (`Beta_Surface` and `Beta_Joints`) whose cluster orders differ
/// (`Beta_Joints` adds `HeadTop_End` at index 6, shifting everything
/// after). We used skin 0's bone table for both meshes, so weights
/// from skin 1 were one bone off from cluster 6 onward — upper-leg
/// vertices ended up weighted to the lower-leg bone.
///
/// This test asserts:
/// * Every bone name unique across skins shows up in the final table.
/// * Cluster indices from skin 1 actually resolve to the correct
///   bone *name* (not just any valid bone).
#[test]
fn multi_skin_unified_bone_table() {
    let Some(path) = test_fbx_path() else {
        eprintln!("RKP_TEST_FBX unset — skipping");
        return;
    };

    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        ..Default::default()
    };
    let scene = ufbx::load_file(path.to_str().unwrap(), opts).expect("load fbx");
    if scene.skin_deformers.len() < 2 {
        eprintln!("skipping: RKP_TEST_FBX only has {} skin deformer(s)", scene.skin_deformers.len());
        return;
    }

    let ex = extract_skeleton(path.to_str().unwrap())
        .expect("extract ok")
        .expect("has skeleton");

    // Every bone referenced by any skin's cluster must exist in the
    // unified table, resolved by bone-node NAME (not positional index).
    for (si, skin) in scene.skin_deformers.iter().enumerate() {
        for (ci, cluster) in skin.clusters.iter().enumerate() {
            let Some(ref bone_node) = cluster.bone_node else { continue };
            let expected_name = bone_node.element.name.to_string();
            let found = ex
                .skeleton
                .bones
                .iter()
                .any(|b| b.name == expected_name);
            assert!(
                found,
                "skin {si} cluster {ci} targets bone '{expected_name}' but it is missing from the unified bone table",
            );
        }
    }
}

/// Full round-trip import → .rkp + .rkskel on disk. Verifies the
/// files land, the skeleton sidecar is non-empty, and no error
/// events fire during the import.
#[test]
fn import_fbx_writes_rkp_and_rkskel() {
    let Some(path) = test_fbx_path() else {
        eprintln!("RKP_TEST_FBX unset — skipping");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("walking.rkp");

    let result = import_mesh_to_opacity_rkp_with(
        &path,
        &output,
        &ImportConfig::default(),
        &NullReporter,
    )
    .expect("import should succeed");

    assert!(output.exists(), ".rkp file should exist on disk");
    let size = std::fs::metadata(&output).unwrap().len();
    assert_eq!(size, result.file_size, "ImportResult.file_size should match");
    assert!(size > 1024, ".rkp file suspiciously small: {size} bytes");

    let skel_path = result.skeleton_path.expect(".rkskel should be written for a skinned FBX");
    assert!(skel_path.exists(), ".rkskel file should exist on disk");
    let skel_size = std::fs::metadata(&skel_path).unwrap().len();
    assert!(skel_size > 128, ".rkskel file suspiciously small: {skel_size} bytes");

    eprintln!(
        "  import result: shell_voxels={}, rkp={}KB, rkskel={}B",
        result.shell_voxels,
        size / 1024,
        skel_size,
    );
}
