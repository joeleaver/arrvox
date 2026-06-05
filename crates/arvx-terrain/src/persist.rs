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

use arvx_core::asset_file::{
    write_artifact_rkp, ArvxFileError, MeshSectionsBlob,
};
use arvx_core::brick_pool::BRICK_CELLS;
use arvx_core::sparse_octree::SparseOctree;
use arvx_core::{BakeArtifact, LeafAttr};

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

/// Path of the per-scene bake-signature sidecar:
/// `<scene_dir>/tiles/.bake_signature`. Holds the
/// [`crate::Terrain::bake_signature`] in effect when the cached tiles
/// were written, so a load can tell whether the cache still matches the
/// live terrain.
pub fn signature_path(scene_dir: &Path) -> PathBuf {
    scene_dir.join(TILES_SUBDIR).join(".bake_signature")
}

/// Write the bake signature for the currently-cached tiles. Creates the
/// `tiles/` directory if needed.
pub fn write_signature(scene_dir: &Path, signature: u64) -> Result<(), String> {
    let path = signature_path(scene_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(&path, signature.to_string())
        .map_err(|e| format!("write {}: {e}", path.display()))
}

/// Read the cached bake signature, or `None` if absent / unparsable
/// (treated as "no valid cache" → tiles re-bake).
pub fn read_signature(scene_dir: &Path) -> Option<u64> {
    let path = signature_path(scene_dir);
    std::fs::read_to_string(&path).ok()?.trim().parse::<u64>().ok()
}

/// Read a `.arvxtile` back into a `BakeArtifact + MeshSectionsBlob`
/// pair that `ArvxSceneManager::integrate_baked_tile` accepts.
///
/// The on-disk format is a v6 `.arvx` (shared with non-terrain
/// assets) — this is a thin composer over the per-section readers in
/// `arvx_core::asset_file`. Phase 4.4 V1 limitation: the loaded
/// artifact's `halo_cells` field is empty — halo data isn't
/// serialised by Phase 4.3 and isn't recomputed here. Loaded tiles
/// fall back to "no halo," leaving boundary seam quads slightly
/// regressed until Phase 4.2b runs neighbour-aware halo refresh.
///
/// Returns the artifact + mesh blob along with the tile's voxel size
/// from the header. The caller is responsible for matching the
/// tile's `TileKey` against the AABB on integrate.
pub fn read_baked_tile(
    path: &Path,
) -> Result<(BakeArtifact, MeshSectionsBlob, f32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let header = arvx_core::asset_file::read_rkp_header(&mut reader)
        .map_err(|e: ArvxFileError| format!("read header: {e}"))?;

    let voxel_size = header.base_voxel_size;
    let octree_depth = header.octree_depth as u8;

    // ── Octree nodes ─────────────────────────────────────────────
    let nodes = arvx_core::asset_file::read_rkp_octree(&mut reader, &header)
        .map_err(|e| format!("read octree: {e}"))?;
    let octree = SparseOctree::from_raw(&nodes, octree_depth, voxel_size);

    // ── Voxels + normals + colors → LeafAttrs ────────────────────
    let voxel_data = arvx_core::asset_file::read_rkp_voxels(&mut reader, &header)
        .map_err(|e| format!("read voxels: {e}"))?;
    let has_normals = (header.flags & arvx_core::asset_file::FLAG_HAS_NORMALS) != 0;
    let normals_bytes = arvx_core::asset_file::read_rkp_normals(&mut reader, &header)
        .map_err(|e| format!("read normals: {e}"))?;
    let normals_u32s: Vec<u32> = if has_normals {
        bytemuck::cast_slice(&normals_bytes).to_vec()
    } else {
        Vec::new()
    };
    let color_bytes = arvx_core::asset_file::read_rkp_color(&mut reader, &header)
        .map_err(|e| format!("read color: {e}"))?;
    let color_u32s: Vec<u32> = if !color_bytes.is_empty() {
        bytemuck::cast_slice(&color_bytes).to_vec()
    } else {
        Vec::new()
    };

    // Skin meta — never produced for terrain; read + discard.
    let _ = arvx_core::asset_file::read_rkp_skin_meta(&mut reader, &header)
        .map_err(|e| format!("read skin meta: {e}"))?;

    // ── Bricks ──────────────────────────────────────────────────
    let bricks_bytes = arvx_core::asset_file::read_rkp_bricks(&mut reader, &header)
        .map_err(|e| format!("read bricks: {e}"))?;
    let bricks_flat: Vec<u32> = if !bricks_bytes.is_empty() {
        bytemuck::cast_slice(&bricks_bytes).to_vec()
    } else {
        Vec::new()
    };
    let brick_cells_per = BRICK_CELLS as usize;
    let n_bricks = if brick_cells_per == 0 {
        0
    } else {
        bricks_flat.len() / brick_cells_per
    };
    let mut brick_cells: Vec<[u32; BRICK_CELLS as usize]> =
        Vec::with_capacity(n_bricks);
    for b in 0..n_bricks {
        let mut arr = [arvx_core::brick_pool::BRICK_EMPTY; BRICK_CELLS as usize];
        let src = &bricks_flat[b * brick_cells_per..(b + 1) * brick_cells_per];
        arr.copy_from_slice(src);
        brick_cells.push(arr);
    }

    // ── Brick face links ────────────────────────────────────────
    // Not serialised — recompute from the loaded octree. Same path
    // `voxelize_to_artifact` runs, so the result matches what the
    // original bake produced.
    let max_brick_id =
        if n_bricks == 0 { 0u32 } else { (n_bricks - 1) as u32 };
    let brick_face_links = arvx_core::brick_face_links::compute_brick_face_links(
        &octree,
        max_brick_id,
    );

    // ── LeafAttrs assembled from voxel + normal + color ─────────
    let voxel_count = header.voxel_count;
    let bytes_per_voxel = std::mem::size_of::<arvx_core::voxel::VoxelSample>();
    let mut leaf_attrs: Vec<LeafAttr> =
        Vec::with_capacity(voxel_count as usize);
    let mut leaf_attr_colors: Vec<u32> =
        Vec::with_capacity(voxel_count as usize);
    for i in 0..voxel_count as usize {
        let off = i * bytes_per_voxel;
        if off + bytes_per_voxel > voxel_data.len() {
            break;
        }
        let vs: &arvx_core::voxel::VoxelSample =
            bytemuck::from_bytes(&voxel_data[off..off + bytes_per_voxel]);
        let mut attr = LeafAttr::new_blended(
            glam::Vec3::Y,
            vs.material_id(),
            vs.secondary_material_id(),
            vs.blend_weight(),
        );
        if has_normals {
            if let Some(&n) = normals_u32s.get(i) {
                attr.normal_oct = n;
            }
        }
        leaf_attrs.push(attr);
        leaf_attr_colors.push(color_u32s.get(i).copied().unwrap_or(0));
    }

    // ── Mesh sections ───────────────────────────────────────────
    let mesh_vertices_bytes =
        arvx_core::asset_file::read_rkp_mesh_vertices(&mut reader, &header)
            .map_err(|e| format!("read mesh vertices: {e}"))?;
    let mesh_indices_bytes =
        arvx_core::asset_file::read_rkp_mesh_indices(&mut reader, &header)
            .map_err(|e| format!("read mesh indices: {e}"))?;
    let meshlet_clusters_bytes =
        arvx_core::asset_file::read_rkp_meshlet_clusters(&mut reader, &header)
            .map_err(|e| format!("read meshlet clusters: {e}"))?;
    let dag_groups_bytes =
        arvx_core::asset_file::read_rkp_dag_groups(&mut reader, &header)
            .map_err(|e| format!("read dag groups: {e}"))?;
    let dag_consumed_bytes =
        arvx_core::asset_file::read_rkp_dag_consumed(&mut reader, &header)
            .map_err(|e| format!("read dag consumed: {e}"))?;
    let dag_produced_bytes =
        arvx_core::asset_file::read_rkp_dag_produced(&mut reader, &header)
            .map_err(|e| format!("read dag produced: {e}"))?;

    let mesh = MeshSectionsBlob {
        vertices: mesh_vertices_bytes,
        indices: mesh_indices_bytes,
        clusters: meshlet_clusters_bytes,
        lod0_index_count: header.mesh_lod0_index_count,
        dag_groups: dag_groups_bytes,
        dag_consumed: dag_consumed_bytes,
        dag_produced: dag_produced_bytes,
    };

    let artifact = BakeArtifact {
        octree,
        voxel_count,
        grid_origin: glam::Vec3::from(header.aabb_min),
        leaf_attrs,
        leaf_attr_colors,
        brick_cells,
        brick_face_links,
        halo_cells: Vec::new(),
    };

    Ok((artifact, mesh, voxel_size))
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
    use crate::bake::bake_tile;
    use crate::fbm::FbmTerrainFn;
    use crate::tile_key::TileKey;

    #[test]
    fn signature_sidecar_roundtrips_and_is_absent_by_default() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // No cache yet → no signature.
        assert_eq!(read_signature(tmp.path()), None);
        // Write then read back the exact value (creates tiles/ on demand).
        write_signature(tmp.path(), 0xDEAD_BEEF_1234_5678).expect("write_signature");
        assert_eq!(read_signature(tmp.path()), Some(0xDEAD_BEEF_1234_5678));
        assert!(signature_path(tmp.path()).exists());
    }

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

    /// Round-trip: bake → save → read → confirm core fields match.
    /// Halo cells aren't persisted by V1 (see `read_baked_tile`
    /// docs), so we just assert the loaded artifact's interior data
    /// matches the baked one.
    #[test]
    fn bake_save_read_roundtrip_preserves_interior() {
        // Coarse voxel size keeps the test fast (~milliseconds).
        let voxel_size_m = 1.0_f32;
        let key = TileKey::level0(0, 0, 0);
        let baked = bake_tile(
            key,
            voxel_size_m,
            &FbmTerrainFn::default().resolve(&arvx_core::NullMaterialLookup),
            &[],
            &crate::TerrainRegionSnapshot::new(),
        )
        .expect("bake_tile should succeed on default FBM at origin");
        assert!(
            baked.artifact.voxel_count > 0,
            "test relies on FBM at origin producing a non-empty tile",
        );

        let tmp = tempfile::tempdir().expect("tempdir");
        let saved_path = save_tile(
            tmp.path(),
            key,
            &baked.artifact,
            baked.voxel_size_m,
        )
        .expect("save_tile");
        assert!(saved_path.exists());

        let (loaded_artifact, loaded_mesh, loaded_voxel_size) =
            read_baked_tile(&saved_path).expect("read_baked_tile");

        assert_eq!(loaded_voxel_size, voxel_size_m);
        // The on-disk file's `voxel_count` header is the LeafAttrs
        // length (the writer sets it to `artifact.leaf_attrs.len()`),
        // which for terrain bakes includes both the interior shell
        // AND halo cells. The `BakeArtifact.voxel_count` field on the
        // in-memory bake counts only interior leaves; the two
        // diverge by the halo. Compare against leaf_attrs.len()
        // instead — that's what the file truly carries.
        assert_eq!(
            loaded_artifact.voxel_count as usize,
            baked.artifact.leaf_attrs.len(),
            "header voxel_count must match leaf_attrs count",
        );
        assert_eq!(
            loaded_artifact.leaf_attrs.len(),
            baked.artifact.leaf_attrs.len(),
            "leaf_attrs count must round-trip"
        );
        assert_eq!(
            loaded_artifact.brick_cells.len(),
            baked.artifact.brick_cells.len(),
            "brick_cells count must round-trip"
        );
        assert_eq!(
            loaded_artifact.octree.as_slice().len(),
            baked.artifact.octree.as_slice().len(),
            "octree node count must round-trip"
        );
        assert_eq!(
            loaded_mesh.vertices.len(),
            baked.mesh.vertices.len(),
            "mesh vertex bytes must round-trip"
        );
        assert_eq!(
            loaded_mesh.indices.len(),
            baked.mesh.indices.len(),
            "mesh index bytes must round-trip"
        );
        assert_eq!(
            loaded_mesh.lod0_index_count, baked.mesh.lod0_index_count,
            "LOD-0 prefix length must round-trip"
        );
        // Halo deliberately empty in V1 — see read_baked_tile docs.
        assert!(loaded_artifact.halo_cells.is_empty());
    }
}
