//! Compose `arvx-core`'s voxelization + mesh-extraction + cluster-DAG
//! passes into a single `bake_tile` call that turns a `TerrainFn` into
//! a self-contained `BakedTile`.
//!
//! Designed to be invoked from a worker thread (Phase 2's streamer
//! will). `voxelize_to_artifact` allocates its own pools internally,
//! so the caller doesn't need to share `LeafAttrPool` / `BrickPool`
//! ownership across threads.

use crate::baked_tile::BakedTile;
use crate::region_snapshot::TerrainRegionSnapshot;
use crate::stamp::{combine_heights, Stamp};
use crate::terrain_fn::TerrainFn;
use crate::tile_key::TileKey;
use arvx_core::asset_file::build_mesh_sections_blob_haloed;
use arvx_core::voxelize_octree::voxelize_to_artifact;
use arvx_core::Aabb;
use glam::Vec3;

/// Halo width in finest-grid voxels that `bake_tile` requests from
/// `voxelize_to_artifact`. With `2`, each tile samples two voxels past
/// every face/edge/corner of its nominal AABB.
///
/// **Why 2, not 1.** The surface-mesh extractor needs symmetric
/// emission at every tile boundary to avoid see-through cracks where
/// the surface slopes across the seam. With a 1-voxel halo only
/// interior cells iterate — but the surface row that fires `+Y` for
/// tile A's interior may sit one cell above or below the row that
/// fires for tile B's interior, leaving the boundary cube referenced
/// from only one side. Extending the halo to 2 voxels lets the
/// extractor iterate one halo layer (cells with one axis in the band
/// `[-1, 0) ∪ [N, N+1)`); the 2nd halo layer remains pure corner
/// data. With both tiles iterating their shared boundary cells, every
/// boundary cube is referenced by at least one quad in each tile and
/// the meshes produce identical (overdrawn) triangles at the seam.
const TILE_HALO_VOXELS: u32 = 2;

/// Bake one terrain tile end-to-end: voxelize the `TerrainFn` across
/// the tile's footprint, compose Layer-2 stamps, extract the surface
/// mesh, and build the cluster DAG.
///
/// Returns `None` if voxelization fails (e.g., empty tile result the
/// downstream `voxelize_to_artifact` rejects). For terrain this is
/// rare — even an "all sky" tile produces a valid empty octree.
///
/// * `key` — which tile to bake.
/// * `voxel_size_m` — the voxel size to use. Caller derives this from
///   the `Terrain` (typically `Terrain::voxel_size_for_level(key.level)`).
/// * `terrain_fn` — the procedural source.
/// * `stamps` — Layer-2 stamps overlapping this tile, in composition
///   order. The streamer pre-filters the global `StampIndex` to just
///   the tile-relevant subset before submitting the bake job, so this
///   slice is typically small (zero in scenes with no stamps).
/// * `regions` — Phase 7 snapshot of biome regions. Bake queries this
///   per voxel for material overrides; the BVH inside
///   `regions.index` is the spatial accelerator. Empty snapshot is a
///   no-op (the common case in scenes without biomes).
///
/// ## Heightmap composition contract
///
/// Each TerrainFn sample produces an `sd`. For heightmap-style
/// TerrainFns this is `sd = wy - base_h(wx, wz)`. We recover
/// `base_h = wy - sd`, fold stamps via `combine_heights`, and repack
/// `sd = wy - composed_h`. V1 stamps are all heightmap kinds — this
/// is the right shape. V2 volumetric stamps will skip this and apply
/// their SD contribution directly (a future overload).
///
/// ## Material override precedence (Phase 7)
///
/// Per voxel the final material is decided in this order:
///
/// 1. Biome region material override (highest-priority overlapping
///    `BiomeRegion` with a `Some(material)` — see
///    [`TerrainRegionSnapshot::material_override_at`]).
/// 2. Stamp material override (last stamp whose footprint covers the
///    voxel and that carries `material_override`).
/// 3. Base [`TerrainFn`] material from `sample`.
///
/// Biome wins over stamp because biomes are large-scale
/// "this whole forest is moss" intent; stamp material is per-feature
/// (Mountain → rock above the slope threshold) and should defer to
/// the broader biome wash. Authors who want a stamp to punch through
/// a biome do so by stacking a higher-priority biome on the stamp.
pub fn bake_tile(
    key: TileKey,
    voxel_size_m: f32,
    terrain_fn: &dyn TerrainFn,
    stamps: &[Stamp],
    regions: &TerrainRegionSnapshot,
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
    // the voxelizer, translates each to tile-local, asks the TerrainFn,
    // and folds Layer-2 stamps over the heightmap before repacking.
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|&world_pos| {
                let local = world_pos - tile_origin_world;
                let mut s = terrain_fn.sample(key, local, voxel_size_m);

                // Layer 2 — heightmap-style stamps.
                if !stamps.is_empty() {
                    let wy = world_pos.y;
                    let mut h = wy - s.sd;
                    let mut mat_override: Option<u16> = None;
                    for stamp in stamps {
                        if let Some(target_h) =
                            stamp.sample_height(world_pos.x, world_pos.z)
                        {
                            h = combine_heights(
                                h,
                                target_h,
                                stamp.combine_op,
                                stamp.position.y,
                            );
                            if let Some(m) = stamp.material_override {
                                mat_override = Some(m);
                            }
                        }
                    }
                    s.sd = wy - h;
                    if let Some(m) = mat_override {
                        s.primary_mat = m;
                        s.secondary_mat = m;
                        s.blend = 0.0;
                    }
                }

                // Phase 7 — biome regions. Highest-priority overlapping
                // BiomeRegion with a `material_override` wins, replacing
                // any stamp or base-TerrainFn material assignment.
                if !regions.is_empty() {
                    if let Some(m) = regions.material_override_at(world_pos) {
                        s.primary_mat = m;
                        s.secondary_mat = m;
                        s.blend = 0.0;
                    }
                }

                let blend_u4 = (s.blend.clamp(0.0, 1.0) * 15.0).round() as u8;
                (s.sd, s.primary_mat, s.secondary_mat, blend_u4, 0)
            })
            .collect()
    };

    let artifact = voxelize_to_artifact(sdf_fn, &aabb, voxel_size_m, TILE_HALO_VOXELS)?;

    // Flatten the per-brick cell payloads (Vec<[u32; BRICK_CELLS]>) into
    // a single Vec<u32> so `build_mesh_sections_blob_haloed` can index
    // it as `brick_id * BRICK_CELLS + flat`.
    let brick_pool_flat: Vec<u32> = artifact.brick_cells.iter().flatten().copied().collect();

    let mesh = build_mesh_sections_blob_haloed(
        artifact.octree.as_slice(),
        artifact.octree.depth(),
        voxel_size_m,
        artifact.grid_origin,
        &brick_pool_flat,
        &artifact.leaf_attrs,
        &[], // terrain is never skinned.
        &artifact.halo_cells,
        TILE_HALO_VOXELS,
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
    use crate::stamp::{FalloffCurve, StampKind};
    use crate::terrain::Terrain;
    use crate::terrain_fn::TerrainSample;

    /// Empty region snapshot — the "no biomes in this scene" baseline
    /// every existing test wants. Phase 7 adds the regions parameter
    /// to `bake_tile`; tests that don't care about regions pass this.
    fn empty_regions() -> TerrainRegionSnapshot {
        TerrainRegionSnapshot::new()
    }

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
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &AllSky, &[], &empty_regions()).expect("bake");
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
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &AllSolid, &[], &empty_regions()).expect("bake");
        // All-solid means no air-solid boundary inside the tile.
        // Surface extraction emits zero triangles.
        assert_eq!(baked.vertex_count(), 0);
    }

    #[test]
    fn bake_flat_half_produces_surface_mesh() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions()).expect("bake");
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
        let baked = bake_tile(TileKey::level0(0, 0, 0), vs, &fbm, &[], &empty_regions()).expect("bake");
        assert!(
            baked.vertex_count() > 100,
            "FBM in a 64m tile should produce many surface vertices; got {}",
            baked.vertex_count()
        );
        assert!(baked.cluster_count() > 0, "should produce at least one cluster");
        assert!(baked.bake_time_ms > 0.0);
    }

    /// Phase 3 watertightness — bake two adjacent tiles whose
    /// procedural source is a perfectly flat plane that cuts through
    /// both tiles' interiors. Verify that boundary SN cubes produce
    /// coincident vertices on both sides of the shared `x = 64 m`
    /// face.
    ///
    /// The flat-plane case is the cleanest target — no slope means
    /// no X-edge crossings, so every boundary cube's vertex lands
    /// exactly at the centre of the boundary plane in the X
    /// direction. Tile A's interior cube (lo coord `N-1` in X) and
    /// tile B's halo cube (lo coord `-1` in X) live at the same
    /// world cube position and see the same 8-corner solidity
    /// pattern, so the centroid is bit-identical.
    ///
    /// FBM with non-trivial slope is a separate (looser) test —
    /// noise-driven centroid shifts make vertex positions drift by
    /// up to a slope-dependent fraction of a voxel, so we'd need a
    /// noise-aware tolerance there. The flat-plane test verifies
    /// the core watertightness invariant; the FBM test that follows
    /// verifies it still holds under realistic terrain.
    #[test]
    fn adjacent_flat_tiles_meet_at_shared_face() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let surface = FlatHalf;

        // Tile A: x ∈ [0, 64). Tile B: x ∈ [64, 128). FlatHalf
        // surface at local y = 32 m → world y = 32 (both tiles see
        // identical SDF since `sd = l.y - 32` is a pure local
        // formula).
        let baked_a = bake_tile(TileKey::level0(0, 0, 0), vs, &surface, &[], &empty_regions()).expect("bake A");
        let baked_b = bake_tile(TileKey::level0(1, 0, 0), vs, &surface, &[], &empty_regions()).expect("bake B");

        // For a horizontal surface (no X-edge crossings), each cube's
        // vertex sits at x = (cube_lo.x + 1) · voxel_size. The
        // boundary cube has cube_lo.x = N-1 in tile A (= -1 in tile B),
        // so its vertex world x is exactly N · voxel_size = 64 m.
        // Filter to a very tight band: 5 % of a voxel.
        let band_centre = 64.0;
        let band = vs * 0.05;
        let boundary_a = collect_boundary_verts(&baked_a, band, band_centre);
        let boundary_b = collect_boundary_verts(&baked_b, band, band_centre);
        assert!(
            !boundary_a.is_empty() && !boundary_b.is_empty(),
            "both tiles must produce boundary vertices crossing the \
             shared x = 64 m face (A={}, B={})",
            boundary_a.len(),
            boundary_b.len(),
        );

        // Every boundary vertex in A must have a matching B vertex
        // within `tol`. Symmetric. 1 mm tol — well below voxel
        // resolution and far below any plausible f32-round-off drift.
        let tol = 1e-3_f32;
        let max_a_to_b = max_nearest_distance(&boundary_a, &boundary_b);
        let max_b_to_a = max_nearest_distance(&boundary_b, &boundary_a);
        assert!(
            max_a_to_b <= tol,
            "tile A boundary vertex farthest from any B vertex: {max_a_to_b:.5} m \
             (tol={tol:.5}, |A|={}, |B|={}); seams are not watertight",
            boundary_a.len(),
            boundary_b.len(),
        );
        assert!(
            max_b_to_a <= tol,
            "tile B boundary vertex farthest from any A vertex: {max_b_to_a:.5} m \
             (tol={tol:.5}, |A|={}, |B|={}); seams are not watertight",
            boundary_a.len(),
            boundary_b.len(),
        );

        assert_eq!(
            boundary_a.len(),
            boundary_b.len(),
            "boundary vertex counts must match across the seam"
        );
    }

    /// FBM seams under realistic terrain — the FBM is continuous
    /// across the boundary (`world_origin.x + local.x` keeps every
    /// SDF lookup keyed on the absolute world position), so the
    /// boundary cube's 8-corner solidity pattern matches on both
    /// sides regardless of slope. The vertex centroid is therefore
    /// computed bit-identically.
    ///
    /// Where the FBM case differs from `FlatHalf` is the centroid's
    /// X coordinate: any non-zero `∂surface/∂x` introduces X-edge
    /// crossings that shift the centroid away from the exact
    /// boundary plane. We can't band on `x = 64 m` with sub-mm
    /// tolerance like the flat case; instead we verify the set of
    /// "near-boundary" vertices in each tile match 1:1 within 1 mm
    /// by allowing a slope-derived band around the boundary plane,
    /// then doing a strict nearest-neighbour distance check on the
    /// captured sets.
    #[test]
    fn adjacent_fbm_tiles_meet_at_shared_face() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let fbm = FbmTerrainFn::default();

        let baked_a = bake_tile(TileKey::level0(0, 0, 0), vs, &fbm, &[], &empty_regions()).expect("bake A");
        let baked_b = bake_tile(TileKey::level0(1, 0, 0), vs, &fbm, &[], &empty_regions()).expect("bake B");

        // For watertightness we only need to verify that vertices
        // landing exactly at the shared `x = 64 m` plane match
        // across tiles. A 1 mm band around the seam captures only
        // boundary-cube vertices whose centroid X coincides with
        // the plane (no significant X-edge slope contribution at
        // that point) — both tiles' boundary cubes produce these
        // from the same 8-corner solidity pattern, so they must
        // line up bit-for-bit. Vertices off the seam (slope-shifted
        // centroids inside one tile's interior) are fully owned by
        // that tile and need no counterpart in the neighbour.
        let band_centre = 64.0;
        let band = 1e-3_f32;
        let boundary_a = collect_boundary_verts(&baked_a, band, band_centre);
        let boundary_b = collect_boundary_verts(&baked_b, band, band_centre);
        assert!(
            !boundary_a.is_empty() && !boundary_b.is_empty(),
            "both tiles must produce boundary vertices for the FBM \
             surface (A={}, B={})",
            boundary_a.len(),
            boundary_b.len(),
        );

        let tol = 1e-3_f32;
        let max_a_to_b = max_nearest_distance(&boundary_a, &boundary_b);
        let max_b_to_a = max_nearest_distance(&boundary_b, &boundary_a);
        assert!(
            max_a_to_b <= tol,
            "tile A boundary vertex farthest from any B vertex: {max_a_to_b:.5} m \
             (tol={tol:.5}, |A|={}, |B|={}); FBM seams not watertight",
            boundary_a.len(),
            boundary_b.len(),
        );
        assert!(
            max_b_to_a <= tol,
            "tile B boundary vertex farthest from any A vertex: {max_b_to_a:.5} m \
             (tol={tol:.5}, |A|={}, |B|={}); FBM seams not watertight",
            boundary_a.len(),
            boundary_b.len(),
        );
        assert_eq!(
            boundary_a.len(),
            boundary_b.len(),
            "boundary vertex counts must match across the FBM seam"
        );
    }

    /// Return the world positions of LOD-0 vertices that fall within
    /// `band` metres of the plane `x = target_world_x` in the tile's
    /// baked mesh.
    ///
    /// The mesh blob's vertex buffer is shared across all DAG LOD
    /// levels. Higher LODs are simplified, so their vertices drift
    /// from the SN cube centroids — restricting to LOD-0 (the
    /// indices in `[0, lod0_index_count)`) keeps the watertight
    /// check honest.
    fn collect_boundary_verts(
        baked: &BakedTile,
        band: f32,
        target_world_x: f32,
    ) -> Vec<glam::Vec3> {
        let v_stride = 32usize;
        let i_stride = 4usize;
        let lod0_index_count = baked.mesh.lod0_index_count as usize;

        // Collect the LOD-0 vertex-id set.
        let mut lod0_vids: std::collections::HashSet<u32> =
            std::collections::HashSet::with_capacity(lod0_index_count);
        for i in 0..lod0_index_count {
            let base = i * i_stride;
            let id = u32::from_le_bytes(
                baked.mesh.indices[base..base + i_stride].try_into().unwrap(),
            );
            lod0_vids.insert(id);
        }

        let mut out = Vec::new();
        for &vid in &lod0_vids {
            let base = vid as usize * v_stride;
            let px = f32::from_le_bytes(
                baked.mesh.vertices[base..base + 4].try_into().unwrap(),
            );
            let py = f32::from_le_bytes(
                baked.mesh.vertices[base + 4..base + 8].try_into().unwrap(),
            );
            let pz = f32::from_le_bytes(
                baked.mesh.vertices[base + 8..base + 12].try_into().unwrap(),
            );
            if (px - target_world_x).abs() <= band {
                out.push(glam::Vec3::new(px, py, pz));
            }
        }
        out
    }

    fn max_nearest_distance(from: &[glam::Vec3], to: &[glam::Vec3]) -> f32 {
        from.iter()
            .map(|&a| {
                to.iter()
                    .map(|&b| (a - b).length())
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0_f32, f32::max)
    }

    /// Diagnostic probe: bake two adjacent FBM tiles, capture all LOD-0
    /// vertices in a half-voxel band around the shared seam plane, and
    /// report A↔B nearest-neighbour distances + count any that exceed
    /// 1 mm. Surfaces visible cracks the strict seam-plane test would
    /// otherwise miss.
    #[test]
    #[ignore = "diagnostic probe — run manually with --ignored"]
    fn probe_fbm_seam_distances() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let fbm = FbmTerrainFn::default();
        let baked_a = bake_tile(TileKey::level0(0, 0, 0), vs, &fbm, &[], &empty_regions()).expect("bake A");
        let baked_b = bake_tile(TileKey::level0(1, 0, 0), vs, &fbm, &[], &empty_regions()).expect("bake B");

        let stride = 32usize;
        let i_stride = 4usize;
        eprintln!(
            "Tile A: {} verts in vbo, LOD-0 indices: {}",
            baked_a.mesh.vertices.len() / stride,
            baked_a.mesh.lod0_index_count
        );
        eprintln!(
            "Tile B: {} verts in vbo, LOD-0 indices: {}",
            baked_b.mesh.vertices.len() / stride,
            baked_b.mesh.lod0_index_count
        );

        let read_pos = |bytes: &[u8], i: usize| -> glam::Vec3 {
            let base = i * stride;
            glam::Vec3::new(
                f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap()),
                f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap()),
                f32::from_le_bytes(bytes[base + 8..base + 12].try_into().unwrap()),
            )
        };

        let collect_lod0 = |baked: &BakedTile| -> std::collections::HashSet<u32> {
            let lod0_n = baked.mesh.lod0_index_count as usize;
            let mut s = std::collections::HashSet::new();
            for i in 0..lod0_n {
                let base = i * i_stride;
                let id = u32::from_le_bytes(
                    baked.mesh.indices[base..base + i_stride].try_into().unwrap(),
                );
                s.insert(id);
            }
            s
        };
        let lod0_a = collect_lod0(&baked_a);
        let lod0_b = collect_lod0(&baked_b);

        let band = vs * 0.5;
        let target_x = 64.0_f32;
        let mut bx_a: Vec<glam::Vec3> = Vec::new();
        for &vid in &lod0_a {
            let p = read_pos(&baked_a.mesh.vertices, vid as usize);
            if (p.x - target_x).abs() <= band {
                bx_a.push(p);
            }
        }
        let mut bx_b: Vec<glam::Vec3> = Vec::new();
        for &vid in &lod0_b {
            let p = read_pos(&baked_b.mesh.vertices, vid as usize);
            if (p.x - target_x).abs() <= band {
                bx_b.push(p);
            }
        }
        eprintln!(
            "Band ±{band}m around x=64 (LOD-0 only): A={} verts, B={} verts",
            bx_a.len(),
            bx_b.len()
        );

        let mut max_d = 0.0_f32;
        let mut count_over_1mm = 0;
        let mut worst: Option<(glam::Vec3, glam::Vec3, f32)> = None;
        for &a in &bx_a {
            let mut best = f32::INFINITY;
            let mut best_b = glam::Vec3::ZERO;
            for &b in &bx_b {
                let d = (a - b).length();
                if d < best {
                    best = d;
                    best_b = b;
                }
            }
            if best > 0.001 {
                count_over_1mm += 1;
            }
            if best > max_d {
                max_d = best;
                worst = Some((a, best_b, best));
            }
        }
        eprintln!("A→B: max distance = {max_d:.5}m, count over 1 mm = {count_over_1mm}");
        if let Some((a, b, d)) = worst {
            eprintln!("Worst A vertex: ({:.4}, {:.4}, {:.4}) → B ({:.4}, {:.4}, {:.4}) dist={d:.4}", a.x, a.y, a.z, b.x, b.y, b.z);
        }

        let mut max_d2 = 0.0_f32;
        let mut count_over_1mm2 = 0;
        for &b in &bx_b {
            let mut best = f32::INFINITY;
            for &a in &bx_a {
                let d = (a - b).length();
                if d < best {
                    best = d;
                }
            }
            if best > 0.001 {
                count_over_1mm2 += 1;
            }
            if best > max_d2 {
                max_d2 = best;
            }
        }
        eprintln!("B→A: max distance = {max_d2:.5}m, count over 1 mm = {count_over_1mm2}");
    }

    // ── Phase 5.3 — stamp composition through bake_tile ───────────────────

    /// Return (min_y, max_y) of LOD-0 vertices in a baked tile.
    fn lod0_y_range(baked: &BakedTile) -> (f32, f32) {
        let v_stride = 32usize;
        let i_stride = 4usize;
        let lod0_n = baked.mesh.lod0_index_count as usize;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for i in 0..lod0_n {
            let base = i * i_stride;
            let id = u32::from_le_bytes(
                baked.mesh.indices[base..base + i_stride].try_into().unwrap(),
            );
            if !seen.insert(id) {
                continue;
            }
            let vbase = id as usize * v_stride;
            let y = f32::from_le_bytes(
                baked.mesh.vertices[vbase + 4..vbase + 8].try_into().unwrap(),
            );
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        (min_y, max_y)
    }

    /// Bake with no stamps vs the same base + a mountain stamp — the
    /// surface MUST rise inside the stamp's footprint. Verifies the
    /// per-voxel stamp composition path in `bake_tile`.
    #[test]
    fn mountain_stamp_raises_surface() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        let baseline =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions()).expect("baseline");
        let (_, base_max_y) = lod0_y_range(&baseline);

        // Mountain at tile centre, lifting 15 m above the flat plane.
        let stamp = crate::stamp::Stamp::new(
            StampKind::Mountain {
                h_max: 15.0,
                radius: 16.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(32.0, 32.0, 32.0),
        );
        let stamped = bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[stamp], &empty_regions())
            .expect("stamped");
        let (_, stamped_max_y) = lod0_y_range(&stamped);

        assert!(
            stamped_max_y > base_max_y + 5.0,
            "mountain stamp should raise the surface by ~h_max; baseline max y = {base_max_y}, stamped max y = {stamped_max_y}",
        );
        // FlatHalf surface is at y=32, mountain peak target = 32+15=47.
        // Allow some discretisation slack ±1 voxel.
        assert!(
            (stamped_max_y - 47.0).abs() < vs * 2.0,
            "stamped max y {stamped_max_y} should be near peak target 47.0",
        );
    }

    /// Bake with no stamps vs the same base + a lake stamp — the
    /// surface MUST drop inside the stamp's footprint.
    #[test]
    fn lake_stamp_lowers_surface() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        let baseline =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions()).expect("baseline");
        let (base_min_y, _) = lod0_y_range(&baseline);

        let stamp = crate::stamp::Stamp::new(
            StampKind::Lake {
                depth: 10.0,
                radius: 16.0,
                falloff: FalloffCurve::Smoothstep,
            },
            // FlatHalf surface is at world y = 32. Place the lake surface there
            // so the basin floor lands at y = 22.
            Vec3::new(32.0, 32.0, 32.0),
        );
        let stamped = bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[stamp], &empty_regions())
            .expect("stamped");
        let (stamped_min_y, _) = lod0_y_range(&stamped);

        assert!(
            stamped_min_y < base_min_y - 3.0,
            "lake stamp should lower the surface; baseline min y = {base_min_y}, stamped min y = {stamped_min_y}",
        );
        // Basin floor target = 32 - 10 = 22. Allow ±1 voxel slack.
        assert!(
            (stamped_min_y - 22.0).abs() < vs * 2.0,
            "stamped min y {stamped_min_y} should be near basin floor 22.0",
        );
    }

    /// Flatten stamp + Replace op forces a target Y regardless of base.
    /// Bake with a wildly varying FBM and a Flatten over the whole tile
    /// XZ extent — the resulting surface should land near `position.y`
    /// everywhere LOD-0 vertices fall inside the rectangle.
    #[test]
    fn flatten_stamp_forces_target_y() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let fbm = FbmTerrainFn::default();

        let target_y = 16.0;
        // Half-extents 40 m covers the full tile (64 m) + slack so every
        // sample lands inside the rect.
        let stamp = crate::stamp::Stamp::new(
            StampKind::Flatten {
                half_extents: glam::Vec2::new(40.0, 40.0),
            },
            Vec3::new(32.0, target_y, 32.0),
        );
        let stamped = bake_tile(TileKey::level0(0, 0, 0), vs, &fbm, &[stamp], &empty_regions())
            .expect("stamped");

        let (min_y, max_y) = lod0_y_range(&stamped);
        // All LOD-0 vertices must land within ±1 voxel of target_y.
        // The Replace op snaps the heightmap to target_y everywhere
        // inside the rect; surface-nets puts the surface at that Y.
        assert!(
            (min_y - target_y).abs() < vs * 2.0 && (max_y - target_y).abs() < vs * 2.0,
            "flatten should force y ≈ {target_y}; got min={min_y}, max={max_y}",
        );
    }

    /// Multiple stamps compose in priority order. A Lake with priority -1
    /// applies first; a Mountain with priority +1 applies on top. The
    /// combined surface should show BOTH effects in their respective
    /// XZ regions.
    #[test]
    fn multiple_stamps_compose_in_order() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        // Lake in the western half of the tile.
        let mut lake = crate::stamp::Stamp::new(
            StampKind::Lake {
                depth: 8.0,
                radius: 14.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(16.0, 32.0, 32.0),
        );
        lake.priority = -1;
        // Mountain in the eastern half.
        let mut mountain = crate::stamp::Stamp::new(
            StampKind::Mountain {
                h_max: 12.0,
                radius: 14.0,
                falloff: FalloffCurve::Smoothstep,
            },
            Vec3::new(48.0, 32.0, 32.0),
        );
        mountain.priority = 1;

        let baked = bake_tile(
            TileKey::level0(0, 0, 0),
            vs,
            &FlatHalf,
            &[lake, mountain],
            &empty_regions(),
        )
        .expect("bake");
        let (min_y, max_y) = lod0_y_range(&baked);

        // Both stamps must show through: min_y around basin floor,
        // max_y around mountain peak.
        assert!(min_y < 30.0, "lake should pull min y below 30; got {min_y}");
        assert!(max_y > 38.0, "mountain should push max y above 38; got {max_y}");
    }

    // ── Phase 7 — biome region material overrides ─────────────────

    /// Collect the set of primary material ids appearing on LOD-0
    /// surface voxels in the baked tile via the artifact's leaf attr
    /// array. (The mesh blob's vertex format doesn't carry material
    /// directly — material is per-leaf, resolved at shade time — so
    /// we read it from the source artifact, not the mesh.)
    fn lod0_primary_materials(baked: &BakedTile) -> std::collections::HashSet<u16> {
        baked
            .artifact
            .leaf_attrs
            .iter()
            .map(|a| a.material_primary)
            .collect()
    }

    /// A biome that covers the whole tile and forces a known material.
    /// FlatHalf gives a known base material; the biome must replace it.
    #[test]
    fn biome_region_material_override_replaces_base() {
        use crate::biome_region::BiomeRegion;
        use crate::region_snapshot::TerrainRegionSnapshot;
        use arvx_regions::{Falloff, Region, RegionEntry, RegionShape};
        use std::sync::Arc;

        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        // Baseline — FlatHalf hard-codes material id 1.
        let baseline =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions())
                .expect("baseline");
        let base_mats = lod0_primary_materials(&baseline);
        assert!(
            base_mats.contains(&1),
            "FlatHalf baseline should write material 1; got {base_mats:?}"
        );

        // Single biome covering the entire tile, material override = 42.
        let region = Region {
            shape: RegionShape::Sphere { radius: 200.0 },
            falloff: Falloff::Smoothstep { transition_m: 50.0 },
            priority: 0,
        };
        let mut w = hecs::World::new();
        let e = w.spawn((region,));
        let snapshot = TerrainRegionSnapshot {
            index: Arc::new(arvx_regions::RegionIndex::from_entries(vec![
                RegionEntry::new(e, region, glam::Vec3::new(32.0, 32.0, 32.0)),
            ])),
            biomes: Arc::new(vec![Some(BiomeRegion {
                material_override: Some(42),
                ..Default::default()
            })]),
        };

        let stamped =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &snapshot).expect("stamped");
        let mats = lod0_primary_materials(&stamped);
        // Every surface voxel inside the region must carry material 42.
        assert!(
            mats.contains(&42),
            "biome override material 42 should appear in baked materials; got {mats:?}"
        );
        // Base material 1 should NOT appear anywhere the biome reaches;
        // the biome covers the entire tile so 1 should be gone.
        assert!(
            !mats.contains(&1),
            "biome should fully override base material 1; got {mats:?}"
        );
    }

    /// A biome with `material_override: None` doesn't touch the material
    /// — it could still influence terrain_fn (V2) but Phase 7 only acts
    /// on the override.
    #[test]
    fn biome_region_without_override_leaves_material_alone() {
        use crate::biome_region::BiomeRegion;
        use crate::region_snapshot::TerrainRegionSnapshot;
        use arvx_regions::{Falloff, Region, RegionEntry, RegionShape};
        use std::sync::Arc;

        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        let region = Region {
            shape: RegionShape::Sphere { radius: 200.0 },
            falloff: Falloff::Smoothstep { transition_m: 50.0 },
            priority: 0,
        };
        let mut w = hecs::World::new();
        let e = w.spawn((region,));
        let snapshot = TerrainRegionSnapshot {
            index: Arc::new(arvx_regions::RegionIndex::from_entries(vec![
                RegionEntry::new(e, region, glam::Vec3::new(32.0, 32.0, 32.0)),
            ])),
            biomes: Arc::new(vec![Some(BiomeRegion::default())]),
        };

        let baked =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &snapshot).expect("baked");
        let mats = lod0_primary_materials(&baked);
        // Base material 1 should still be present — no override.
        assert!(
            mats.contains(&1),
            "empty BiomeRegion should leave base material intact; got {mats:?}"
        );
    }
}
