//! Loads a real `.rkp` file and runs the splat-extract walk against
//! it, printing the resulting splat count + bytes. Verifies that the
//! Phase A extractor doesn't crash, doesn't drop voxels, and produces
//! a vertex buffer that fits in a sensible memory budget.
//!
//! Skipped by default — set `RKP_SPLAT_TEST_ASSET=/path/to/scene.rkp`
//! to run. Suggested asset for the prototype: the elephant at
//! `rkifield_game/splat5/assets/models/elephant/scene.rkp`.

use std::io::BufReader;

use glam::{Mat4, Vec3};
use rkp_render::splat_pass::extract_splats;

#[test]
fn elephant_extracts_to_a_sensible_splat_count() {
    let Ok(path) = std::env::var("RKP_SPLAT_TEST_ASSET") else {
        eprintln!("[splat_extract] skipping — set RKP_SPLAT_TEST_ASSET to a .rkp path");
        return;
    };
    let path = std::path::PathBuf::from(path);
    assert!(path.exists(), "asset path does not exist: {}", path.display());

    let mut file =
        std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut reader = BufReader::new(&mut file);

    let header = rkp_core::asset_file::read_rkp_header(&mut reader)
        .expect("read .rkp header");
    let octree_nodes =
        rkp_core::asset_file::read_rkp_octree(&mut reader, &header).expect("read octree");
    // Voxels section sits between octree and the optional sections;
    // we don't use the data but need to advance the reader past it.
    let _voxels =
        rkp_core::asset_file::read_rkp_voxels(&mut reader, &header).expect("read voxels");
    if header.flags & rkp_core::asset_file::FLAG_HAS_NORMALS != 0 {
        let _ = rkp_core::asset_file::read_rkp_normals(&mut reader, &header)
            .expect("read normals");
    }
    let bricks_bytes = if header.flags & rkp_core::asset_file::FLAG_HAS_BRICKS != 0 {
        rkp_core::asset_file::read_rkp_bricks(&mut reader, &header)
            .expect("read bricks")
    } else {
        Vec::new()
    };
    let bricks: &[u32] = if !bricks_bytes.is_empty() {
        bytemuck::cast_slice(&bricks_bytes)
    } else {
        &[]
    };

    // Reconstruct grid_origin from the AABB the same way `AssetEntry::info`
    // does in the runtime — `aabb_center - extent/2`.
    let aabb_min = Vec3::from(header.aabb_min);
    let aabb_max = Vec3::from(header.aabb_max);
    let extent = (1u32 << header.octree_depth as u8) as f32 * header.base_voxel_size;
    let grid_origin = (aabb_min + aabb_max) * 0.5 - Vec3::splat(extent * 0.5);

    assert!(header.octree_depth <= u8::MAX as u32, "depth overflow");
    let depth_u8 = header.octree_depth as u8;
    let started = std::time::Instant::now();
    let splats = extract_splats(
        &octree_nodes,
        depth_u8,
        header.base_voxel_size,
        grid_origin,
        Mat4::IDENTITY,
        bricks,
    );
    let elapsed = started.elapsed();

    let bytes = splats.len() * std::mem::size_of::<rkp_render::splat_pass::SplatVertex>();
    eprintln!(
        "[splat_extract] {}: splats={} ({:.1} MB) in {:.1} ms; depth={} vs={}",
        path.display(),
        splats.len(),
        bytes as f32 / (1024.0 * 1024.0),
        elapsed.as_secs_f32() * 1000.0,
        header.octree_depth,
        header.base_voxel_size,
    );

    // Sanity floor: ANY real asset has at least a thousand surface
    // voxels. If we get fewer, the walk is broken.
    assert!(splats.len() > 1000, "suspiciously few splats: {}", splats.len());

    // Sanity ceiling: 30 MB asset → couple hundred million splats would
    // be absurd. 50M cap is a soft "is this scaling right" alarm.
    assert!(splats.len() < 50_000_000, "splat count blew past 50M");

    // Per-splat sanity: every position should sit inside the asset's
    // AABB plus a small margin (half a voxel for cell-center).
    let pad = header.base_voxel_size;
    for s in splats.iter().take(1000) {
        let p = s.world_pos;
        assert!(
            p[0] >= aabb_min.x - pad && p[0] <= aabb_max.x + pad,
            "splat x={} out of aabb [{},{}]",
            p[0],
            aabb_min.x,
            aabb_max.x,
        );
        assert!(p[1] >= aabb_min.y - pad && p[1] <= aabb_max.y + pad);
        assert!(p[2] >= aabb_min.z - pad && p[2] <= aabb_max.z + pad);
        assert!(s.radius > 0.0);
    }
}
