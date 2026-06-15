//! End-to-end pipeline integration tests using tiny procedurally-
//! generated meshes. Validates the full load-mesh → BVH → voxelize →
//! write-rkp → read-arvx-header round trip. Exercises both OBJ and
//! glTF loaders without depending on external asset files.

use std::io::Write as _;

use arvx_core::asset_file::{
    ARVX_MAGIC, ARVX_VERSION, FLAG_HAS_DISTANCE, read_rkp_bricks, read_rkp_color,
    read_rkp_dag_consumed, read_rkp_dag_groups, read_rkp_dag_produced, read_rkp_distance,
    read_rkp_header, read_rkp_mesh_indices, read_rkp_mesh_vertices, read_rkp_meshlet_clusters,
    read_rkp_normals, read_rkp_octree, read_rkp_skin_meta, read_rkp_voxels,
};
use arvx_import::{ImportConfig, NullReporter, import_mesh_to_opacity_rkp_with};

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
    let output = tmp.path().join("cube.arvx");

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
    // .arvx, not just a file with random bytes.
    let file = std::fs::File::open(&output).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let header = read_rkp_header(&mut reader).expect("header should parse");

    assert_eq!(header.magic, ARVX_MAGIC);
    assert_eq!(header.version, ARVX_VERSION);
    assert_eq!(header.voxel_count, result.shell_voxels);
    assert_eq!(header.base_voxel_size, 0.1);
    // Cube is 1m across, expanded by 1 voxel on each side.
    assert!(header.aabb_min[0] < 0.0 && header.aabb_max[0] > 0.0);
}

/// The full import pipeline must persist the v7 per-leaf distance
/// section so a loaded import re-extracts / sculpts with Manifold-DC
/// instead of the blur fallback. Distances are 1:1 with the leaves
/// (== header voxel_count) and POSITIVE (shell leaves are outside
/// voxels). Before the wiring, the import passed None and the section
/// never existed.
#[test]
fn obj_cube_persists_distance_section() {
    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.arvx");

    let config = ImportConfig {
        voxel_size: Some(0.1),
        ..ImportConfig::default()
    };
    let result = import_mesh_to_opacity_rkp_with(&source, &output, &config, &NullReporter)
        .expect("import should succeed");

    let file = std::fs::File::open(&output).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let header = read_rkp_header(&mut reader).expect("header");
    assert_eq!(header.version, ARVX_VERSION);
    assert!(
        header.flags & FLAG_HAS_DISTANCE != 0,
        "import must set FLAG_HAS_DISTANCE",
    );
    assert!(header.distance_compressed_size > 0, "distance section must be non-empty");

    // Distance is written LAST — drain every prior section in order.
    let _ = read_rkp_octree(&mut reader, &header).expect("octree");
    let _ = read_rkp_voxels(&mut reader, &header).expect("voxels");
    let _ = read_rkp_normals(&mut reader, &header).expect("normals");
    let _ = read_rkp_bricks(&mut reader, &header).expect("bricks");
    let _ = read_rkp_color(&mut reader, &header).expect("color");
    let _ = read_rkp_skin_meta(&mut reader, &header).expect("skin");
    let _ = read_rkp_mesh_vertices(&mut reader, &header).expect("mesh verts");
    let _ = read_rkp_mesh_indices(&mut reader, &header).expect("mesh indices");
    let _ = read_rkp_meshlet_clusters(&mut reader, &header).expect("clusters");
    let _ = read_rkp_dag_groups(&mut reader, &header).expect("dag groups");
    let _ = read_rkp_dag_consumed(&mut reader, &header).expect("dag consumed");
    let _ = read_rkp_dag_produced(&mut reader, &header).expect("dag produced");
    let dist_bytes = read_rkp_distance(&mut reader, &header).expect("distance");
    let dists: &[i16] = bytemuck::cast_slice(&dist_bytes);

    assert_eq!(
        dists.len(),
        result.shell_voxels as usize,
        "distances must be 1:1 with shell leaves (== voxel_count)",
    );
    assert!(
        dists.iter().all(|&q| q >= 0),
        "every shell leaf is an outside voxel → non-negative signed distance",
    );
    assert!(
        dists.iter().any(|&q| q > 0),
        "a real bake must carry nonzero distances",
    );
}

#[test]
fn obj_cube_no_normalize_preserves_size() {
    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.arvx");

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
    use arvx_import::{CancelToken, NullReporter};

    let tmp = tempfile::tempdir().unwrap();
    let source = write_tmp_obj(tmp.path(), "cube.obj");
    let output = tmp.path().join("cube.arvx");

    let (reporter, cancel) = CancelToken::new(NullReporter);
    cancel.cancel();

    let err = import_mesh_to_opacity_rkp_with(
        &source,
        &output,
        &ImportConfig::default(),
        &reporter,
    )
    .expect_err("pre-cancelled import should abort");

    assert!(matches!(err, arvx_import::ImportError::Cancelled));
    assert!(!output.exists(), "no output file should be written when cancelled");
}

#[test]
fn invalid_config_rejected_before_load() {
    let tmp = tempfile::tempdir().unwrap();
    // No OBJ written — if validation fires first, we never touch disk.
    let source = tmp.path().join("does_not_exist.obj");
    let output = tmp.path().join("out.arvx");

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
        matches!(err, arvx_import::ImportError::InvalidConfig(ref s) if s.contains("voxel_size")),
        "expected InvalidConfig(voxel_size), got: {err:?}",
    );
    assert!(!output.exists(), "no output file should be written on config-error");
}
