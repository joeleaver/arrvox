//! Compose `arvx-core`'s voxelization + mesh-extraction + cluster-DAG
//! passes into a single `bake_tile` call that turns a `TerrainFn` into
//! a self-contained `BakedTile`.
//!
//! Designed to be invoked from a worker thread (Phase 2's streamer
//! will). `voxelize_to_artifact` allocates its own pools internally,
//! so the caller doesn't need to share `LeafAttrPool` / `BrickPool`
//! ownership across threads.

use crate::baked_tile::BakedTile;
use crate::terrain_fn::TerrainFn;
use crate::tile_key::TileKey;
use arvx_core::asset_file::build_mesh_sections_blob;
use arvx_core::voxelize_octree::voxelize_to_artifact;
use arvx_core::Aabb;
use glam::Vec3;

/// Bake one terrain tile end-to-end: voxelize the `TerrainFn` across
/// the tile's footprint, extract the surface mesh, and build the
/// cluster DAG.
///
/// Returns `None` if voxelization fails (e.g., empty tile result the
/// downstream `voxelize_to_artifact` rejects). For terrain this is
/// rare — even an "all sky" tile produces a valid empty octree.
///
/// * `key` — which tile to bake.
/// * `voxel_size_m` — the voxel size to use. Caller derives this from
///   the `Terrain` (typically `Terrain::voxel_size_for_level(key.level)`).
/// * `terrain_fn` — the procedural source.
pub fn bake_tile(
    key: TileKey,
    voxel_size_m: f32,
    terrain_fn: &dyn TerrainFn,
) -> Option<BakedTile> {
    let t0 = std::time::Instant::now();

    // Tile origin in absolute world coords (f32). Phase 1 only bakes
    // tiles near origin so f32 precision is fine; Phase 2 will switch
    // the world→local translation to an integer-anchored path.
    let tile_origin_world: Vec3 = key.origin_world().to_vec3();
    let extent = key.extent_m();
    let aabb = Aabb {
        min: tile_origin_world,
        max: tile_origin_world + Vec3::splat(extent),
    };

    // SDF callback: receives a batch of absolute-world positions from
    // the voxelizer, translates each to tile-local, asks the TerrainFn.
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|&world_pos| {
                let local = world_pos - tile_origin_world;
                let s = terrain_fn.sample(key, local, voxel_size_m);
                let blend_u4 = (s.blend.clamp(0.0, 1.0) * 15.0).round() as u8;
                (s.sd, s.primary_mat, s.secondary_mat, blend_u4, 0)
            })
            .collect()
    };

    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size_m)?;

    // Flatten the per-brick cell payloads (Vec<[u32; BRICK_CELLS]>) into
    // a single Vec<u32> so `build_mesh_sections_blob` can index it as
    // `brick_id * BRICK_CELLS + flat`.
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();

    let mesh = build_mesh_sections_blob(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size_m,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[], // terrain is never skinned.
    );

    Some(BakedTile {
        key,
        voxel_size_m,
        artifact,
        mesh,
        bake_time_ms: t0.elapsed().as_secs_f32() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbm::FbmTerrainFn;
    use crate::terrain::Terrain;
    use crate::terrain_fn::TerrainSample;

    /// All-empty terrain (everywhere above surface): the bake should
    /// still return a valid artifact, with an empty surface mesh.
    struct AllSky;
    impl TerrainFn for AllSky {
        fn sample(&self, _t: TileKey, _l: Vec3, _v: f32) -> TerrainSample {
            TerrainSample {
                sd: 100.0,
                primary_mat: 1,
                secondary_mat: 1,
                blend: 0.0,
            }
        }
    }

    /// All-solid terrain. Surface mesh is empty (no boundary between
    /// solid and air inside the tile).
    struct AllSolid;
    impl TerrainFn for AllSolid {
        fn sample(&self, _t: TileKey, _l: Vec3, _v: f32) -> TerrainSample {
            TerrainSample {
                sd: -100.0,
                primary_mat: 1,
                secondary_mat: 1,
                blend: 0.0,
            }
        }
    }

    /// A flat plane at y=32m cutting through the middle of the tile.
    /// Surface mesh MUST contain triangles — this is the smoke test
    /// for the whole pipeline.
    struct FlatHalf;
    impl TerrainFn for FlatHalf {
        fn sample(&self, _t: TileKey, l: Vec3, _v: f32) -> TerrainSample {
            TerrainSample {
                sd: l.y - 32.0,
                primary_mat: 1,
                secondary_mat: 1,
                blend: 0.0,
            }
        }
    }

    #[test]
    fn bake_all_sky_returns_empty_mesh() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &AllSky).expect("bake");
        assert_eq!(
            baked.vertex_count(),
            0,
            "all-sky tile should have no surface"
        );
        assert_eq!(baked.index_count(), 0);
    }

    #[test]
    fn bake_all_solid_returns_empty_mesh() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &AllSolid).expect("bake");
        // All-solid means no air-solid boundary inside the tile.
        // Surface extraction emits zero triangles.
        assert_eq!(baked.vertex_count(), 0);
    }

    #[test]
    fn bake_flat_half_produces_surface_mesh() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf).expect("bake");
        assert!(
            baked.vertex_count() > 0,
            "flat plane bisecting tile must produce vertices; got {}",
            baked.vertex_count()
        );
        assert!(baked.index_count() >= 3, "must produce at least one tri");
        // Index count is a multiple of 3 (triangle list).
        assert_eq!(baked.index_count() % 3, 0);
    }

    /// FBM at origin produces a non-trivial surface. This is the closest
    /// thing to a Phase-1 deliverable smoke test — the whole pipeline
    /// runs end to end with the real procedural source.
    #[test]
    fn bake_fbm_at_origin_produces_surface() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let fbm = FbmTerrainFn::default();
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &fbm).expect("bake");
        assert!(
            baked.vertex_count() > 100,
            "FBM in a 64m tile should produce many surface vertices; got {}",
            baked.vertex_count()
        );
        assert!(baked.cluster_count() > 0, "should produce at least one cluster");
        assert!(baked.bake_time_ms > 0.0);
    }
}
