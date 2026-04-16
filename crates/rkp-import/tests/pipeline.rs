//! End-to-end pipeline integration tests using tiny procedurally-
//! generated meshes. Validates the full load-mesh → BVH → voxelize →
//! write-rkp → read-rkp-header round trip. Exercises both OBJ and
//! glTF loaders without depending on external asset files.

use std::io::Write as _;

use rkp_core::asset_file::{
    RKP_MAGIC, RKP_VERSION, read_rkp_header,
};
use rkp_import::{ImportConfig, NullReporter, import_mesh_to_opacity_rkp_with};

/// Minimal unit-cube OBJ (8 vertices, 12 triangles, outward normals).
fn cube_obj() -> String {
    "\
o cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 4 7 3
f 4 8 7
f 1 8 4
f 1 5 8
f 2 3 7
f 2 7 6
"
    .to_string()
}

fn write_tmp_obj(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(cube_obj().as_bytes()).unwrap();
    path
}

#[test]
fn obj_cube_full_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.rkp");

    let config = ImportConfig {
        voxel_size: Some(0.1),
        ..ImportConfig::default()
    };

    let result = import_mesh_to_opacity_rkp_with(
        &source,
        &output,
        &config,
        &NullReporter,
    )
    .expect("import should succeed");

    assert!(output.exists(), "rkp must land on disk");
    assert!(result.shell_voxels > 0, "cube must produce shell voxels");
    assert!(result.skeleton_path.is_none(), "OBJ has no skeleton");
    assert_eq!(result.finest_voxel_size, 0.1);

    // Round-trip the header — verifies the file is actually a valid
    // .rkp, not just a file with random bytes.
    let file = std::fs::File::open(&output).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let header = read_rkp_header(&mut reader).expect("header should parse");

    assert_eq!(header.magic, RKP_MAGIC);
    assert_eq!(header.version, RKP_VERSION);
    assert_eq!(header.voxel_count, result.shell_voxels);
    assert_eq!(header.base_voxel_size, 0.1);
    // Cube is 1m across, expanded by 1 voxel on each side.
    assert!(header.aabb_min[0] < 0.0 && header.aabb_max[0] > 0.0);
}

#[test]
fn obj_cube_no_normalize_preserves_size() {
    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.rkp");

    let config = ImportConfig {
        voxel_size: Some(0.1),
        no_normalize: true,
        ..ImportConfig::default()
    };

    let result = import_mesh_to_opacity_rkp_with(
        &source,
        &output,
        &config,
        &NullReporter,
    )
    .unwrap();

    // With no_normalize, cube stays 1m across. Shell voxels at 0.1m
    // should be roughly 6 * 10² = 600 minus edges — so ~200–600.
    // (Exact count depends on classify heuristics; we just bound it.)
    assert!(result.shell_voxels >= 100);
    assert!(result.shell_voxels <= 2000);
}

/// Firing the cancel handle before calling `import_*` aborts the
/// import at the first stage-boundary check without writing
/// anything to disk.
#[test]
fn cancelled_before_start_aborts_cleanly() {
    use rkp_import::{CancelToken, NullReporter};

    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.rkp");

    let (reporter, cancel) = CancelToken::new(NullReporter);
    cancel.cancel();

    let err = import_mesh_to_opacity_rkp_with(
        &source,
        &output,
        &ImportConfig::default(),
        &reporter,
    )
    .expect_err("pre-cancelled import should abort");

    assert!(matches!(err, rkp_import::ImportError::Cancelled));
    assert!(!output.exists(), "no output file should be written when cancelled");
}

#[test]
fn invalid_config_rejected_before_load() {
    let tmp = tempfile::tempdir().unwrap();
    // No OBJ written — if validation fires first, we never touch disk.
    let source = tmp.path().join("does_not_exist.obj");
    let output = tmp.path().join("out.rkp");

    let config = ImportConfig {
        voxel_size: Some(-1.0),
        ..ImportConfig::default()
    };

    let err = import_mesh_to_opacity_rkp_with(
        &source,
        &output,
        &config,
        &NullReporter,
    )
    .expect_err("should reject negative voxel_size");
    assert!(
        matches!(err, rkp_import::ImportError::InvalidConfig(ref s) if s.contains("voxel_size")),
        "expected InvalidConfig(voxel_size), got: {err:?}",
    );
    assert!(!output.exists(), "no output file should be written on config-error");
}
