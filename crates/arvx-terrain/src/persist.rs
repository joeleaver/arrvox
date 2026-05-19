//! `.arvxtile` save / path helpers.
//!
//! Phase 4 of the V1 terrain plan persists touched tiles to disk
//! alongside the scene. A `.arvxtile` is literally an `.arvx` v6 with
//! a different file extension — the on-disk format is shared so the
//! engine's load path can bring tiles back without a parallel codec
//! (handled in Phase 4.4).
//!
//! Untouched tiles never hit disk; they regenerate deterministically
//! from `TerrainFn(tile_key, …)`.

use std::path::{Path, PathBuf};

use arvx_core::asset_file::write_artifact_rkp;
use arvx_core::BakeArtifact;

use crate::tile_key::TileKey;

/// Standard subdirectory name under the scene root that holds saved
/// tile files. Mirrored on read in Phase 4.4.
pub const TILES_SUBDIR: &str = "tiles";

/// Resolve the on-disk `.arvxtile` path for a tile inside a scene
/// directory.
///
/// `scene_dir` should be the directory the scene file lives in —
/// usually `scene_path.parent()`. The result is
/// `<scene_dir>/tiles/{level}_{x}_{y}_{z}.arvxtile`.
pub fn tile_path(scene_dir: &Path, key: TileKey) -> PathBuf {
    scene_dir.join(TILES_SUBDIR).join(format!(
        "{}_{}_{}_{}.arvxtile",
        key.level, key.x, key.y, key.z
    ))
}

/// Persist one terrain tile to disk.
///
/// `artifact` carries file-local IDs (built by the scene manager's
/// `extract_artifact_from_handle`). `voxel_size_m` is the tile's
/// voxel size — derived from `Terrain::voxel_size_for_level(level)`
/// on the engine side. The tile's world-space AABB is reconstructed
/// from the artifact's `grid_origin` + octree extent.
///
/// Writes atomically (`.inprogress` + rename) via the underlying
/// `write_artifact_rkp`. Creates the `<scene_dir>/tiles/` directory
/// on first save.
pub fn save_tile(
    scene_dir: &Path,
    key: TileKey,
    artifact: &BakeArtifact,
    voxel_size_m: f32,
) -> Result<PathBuf, String> {
    let path = tile_path(scene_dir, key);
    let extent_m = artifact.octree.extent() as f32 * voxel_size_m;
    let aabb_min: [f32; 3] = artifact.grid_origin.into();
    let aabb_max: [f32; 3] = [
        artifact.grid_origin.x + extent_m,
        artifact.grid_origin.y + extent_m,
        artifact.grid_origin.z + extent_m,
    ];
    write_artifact_rkp(&path, artifact, aabb_min, aabb_max, voxel_size_m)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_key::TileKey;

    #[test]
    fn tile_path_uses_tiles_subdir_and_key_filename() {
        let dir = PathBuf::from("/tmp/scene");
        let p = tile_path(&dir, TileKey::level0(3, -2, 5));
        assert_eq!(p, PathBuf::from("/tmp/scene/tiles/0_3_-2_5.arvxtile"));
    }

    #[test]
    fn tile_path_respects_level() {
        let dir = PathBuf::from("/tmp/scene");
        let p = tile_path(&dir, TileKey { level: 2, x: 0, y: 0, z: 0 });
        assert_eq!(p, PathBuf::from("/tmp/scene/tiles/2_0_0_0.arvxtile"));
    }
}
