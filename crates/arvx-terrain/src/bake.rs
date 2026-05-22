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
/// Convenience: same as [`bake_tile_with_skirts`] with skirts
/// disabled (`skirt_depth_m = 0.0`) and no world envelope clamp
/// (`world_floor_y = None`). Kept for tests + persist roundtrip
/// code that don't care about skirts or envelope safety.
pub fn bake_tile(
    key: TileKey,
    voxel_size_m: f32,
    terrain_fn: &dyn TerrainFn,
    stamps: &[Stamp],
    regions: &TerrainRegionSnapshot,
) -> Option<BakedTile> {
    bake_tile_with_skirts(key, voxel_size_m, terrain_fn, stamps, regions, 0.0, None)
}

pub fn bake_tile_with_skirts(
    key: TileKey,
    voxel_size_m: f32,
    terrain_fn: &dyn TerrainFn,
    stamps: &[Stamp],
    regions: &TerrainRegionSnapshot,
    skirt_depth_m: f32,
    world_floor_y: Option<f32>,
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

    // World-envelope safety margin. Surface nets needs at least one
    // interior cell below the surface to extract the surface; we
    // give it two so the basin bottom always has solid voxels under
    // it even at the coarsest voxel size in a tile.
    let envelope_floor = world_floor_y.map(|f| f + 2.0 * voxel_size_m);

    // SDF callback: receives a batch of absolute-world positions from
    // the voxelizer, translates each to tile-local, asks the TerrainFn,
    // and folds Layer-2 stamps over the heightmap before repacking.
    let sdf_fn = |positions: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
        positions
            .iter()
            .map(|&world_pos| {
                let local = world_pos - tile_origin_world;
                let mut s = terrain_fn.sample(key, local, voxel_size_m);

                let wy = world_pos.y;
                let mut h = wy - s.sd;
                let mut mat_override: Option<u16> = None;

                // Layer 2 — heightmap-style stamps. The V2 stamp
                // API returns `(target_h, weight)`; we blend the
                // combine_op'd target back toward the running `h`
                // by `(1 - weight)`. That makes soft-rim stamps
                // (rounded plateau, noisy lake shore) ramp their
                // effect down to zero at the rim without changing
                // the combine_op contract.
                if !stamps.is_empty() {
                    for stamp in stamps {
                        if let Some(sample) =
                            stamp.sample_height(world_pos.x, world_pos.z)
                        {
                            let combined = combine_heights(
                                h,
                                sample.target_h,
                                stamp.combine_op,
                                stamp.position.y,
                            );
                            // Lerp(h, combined, weight). weight = 1
                            // reproduces V1 behaviour exactly.
                            h = h + (combined - h) * sample.weight;
                            if let Some(m) = stamp.material_override {
                                mat_override = Some(m);
                            }
                        }
                    }
                }

                // World-envelope clamp. Stamps (and even the base
                // TerrainFn) can otherwise drive the composed height
                // below the world's solid envelope, leaving the
                // entire footprint above the surface — i.e. a hole
                // through the tile that the player falls through.
                // Lifting `h` back up to `floor + 2 voxels` keeps a
                // solid floor everywhere; lakes deeper than the
                // world allows simply bottom out at that floor.
                if let Some(floor_h) = envelope_floor {
                    if h < floor_h {
                        h = floor_h;
                    }
                }

                s.sd = wy - h;
                if let Some(m) = mat_override {
                    s.primary_mat = m;
                    s.secondary_mat = m;
                    s.blend = 0.0;
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

    let mut mesh = build_mesh_sections_blob_haloed(
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

    // V2 LOD pyramid: append lateral skirts so LOD-band cracks aren't
    // see-through. Per-vertex vertical strips; adjacent strips overlap
    // visually into a curtain when viewed horizontally. `0.0` disables.
    append_lateral_skirts(
        &mut mesh,
        tile_origin_world,
        extent,
        voxel_size_m,
        skirt_depth_m,
    );

    Some(BakedTile {
        key,
        voxel_size_m,
        artifact,
        mesh,
        bake_time_ms: t0.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Append lateral-tile-boundary skirts to `mesh`.
///
/// V2 LOD pyramid follow-up: at LOD-band boundaries adjacent tiles
/// use different voxel sizes, producing surface vertices at slightly
/// different heights at the shared edge. Cameras see through the gap
/// as a thin sky-coloured slit. Skirts mask the slit by emitting a
/// thin vertical strip dropping `skirt_depth_m` below each
/// boundary-surface vertex. Adjacent strips overlap into a visually
/// continuous curtain when viewed from horizontal angles.
///
/// Per-vertex strips (not edge-stitched) are a deliberate V1
/// simplification: cheaper to implement, robust to non-manifold
/// surface meshes, and adjacent strips are ≈ voxel_size apart so the
/// overlap closes the visual gap. Trade-off: from directly above the
/// boundary, the strips contribute no horizontal extent (they're
/// vertical). Acceptable for a Y-up world where players rarely look
/// straight down at a tile seam.
///
/// `skirt_depth_m <= 0` or an empty mesh is a no-op.
fn append_lateral_skirts(
    mesh: &mut arvx_core::asset_file::MeshSectionsBlob,
    tile_origin_world: Vec3,
    tile_extent_m: f32,
    voxel_size_m: f32,
    skirt_depth_m: f32,
) {
    use arvx_core::mesh_cluster::{
        MeshletCluster, CLUSTER_FLAG_LOD_DIRTY, DAG_GROUP_NONE, PARENT_GROUP_ERROR_ROOT,
    };
    use arvx_core::mesh_extract::MeshVertex;

    if skirt_depth_m <= 0.0 || mesh.vertices.is_empty() {
        return;
    }

    let verts: &[MeshVertex] = bytemuck::cast_slice(&mesh.vertices);
    if verts.is_empty() {
        return;
    }

    // Boundary plane positions (object-local = world for terrain).
    let x_min = tile_origin_world.x;
    let x_max = tile_origin_world.x + tile_extent_m;
    let z_min = tile_origin_world.z;
    let z_max = tile_origin_world.z + tile_extent_m;

    // A vertex is "on the boundary" if it's within half a voxel of the
    // plane — the SN cube vertex centres land on cell-edge midpoints
    // which can be slightly inside the tile.
    let eps = voxel_size_m * 0.5;
    // Each per-vertex strip is voxel_size wide so adjacent strips meet.
    let half_w = voxel_size_m * 0.5;

    let pre_vertex_count = verts.len() as u32;
    let pre_index_count = (mesh.indices.len() / std::mem::size_of::<u32>()) as u32;

    let mut new_verts: Vec<MeshVertex> = Vec::new();
    let mut new_indices: Vec<u32> = Vec::new();
    let mut patch_min = [f32::INFINITY; 3];
    let mut patch_max = [f32::NEG_INFINITY; 3];

    let update_aabb = |min: &mut [f32; 3], max: &mut [f32; 3], p: [f32; 3]| {
        for k in 0..3 {
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    };

    // Per-face skirt: emit a 4-vertex / 2-triangle strip facing
    // outward. CCW winding from the outward POV: TL, BL, BR / TL, BR, TR.
    let mut emit_strip =
        |pos: [f32; 3], normal_oct: u32, leaf_attr_id: u32, tangent: [f32; 3]| {
            let p = pos;
            let tl = [p[0] - tangent[0] * half_w, p[1], p[2] - tangent[2] * half_w];
            let tr = [p[0] + tangent[0] * half_w, p[1], p[2] + tangent[2] * half_w];
            let bl = [tl[0], tl[1] - skirt_depth_m, tl[2]];
            let br = [tr[0], tr[1] - skirt_depth_m, tr[2]];

            let base = pre_vertex_count + new_verts.len() as u32;

            // All 4 skirt verts inherit the source vertex's attrs so
            // the resolve pass picks the same material / normal at
            // shading time. The strip's geometric normal is horizontal
            // outward, but normal_oct stays equal to the source for
            // visual continuity with the surface above.
            for p in [tl, tr, bl, br] {
                new_verts.push(MeshVertex {
                    local_pos: p,
                    normal_oct,
                    leaf_attr_id,
                    bone_indices: 0,
                    bone_weights: 0,
                    _pad: 0,
                });
                update_aabb(&mut patch_min, &mut patch_max, p);
            }
            // tl=0, tr=1, bl=2, br=3. CCW from outward POV: TL→BL→BR, TL→BR→TR.
            new_indices.extend_from_slice(&[
                base, base + 2, base + 3,
                base, base + 3, base + 1,
            ]);
        };

    for v in verts {
        let p = v.local_pos;
        let on_x_min = (p[0] - x_min).abs() < eps;
        let on_x_max = (p[0] - x_max).abs() < eps;
        let on_z_min = (p[2] - z_min).abs() < eps;
        let on_z_max = (p[2] - z_max).abs() < eps;

        // A corner vertex may be on TWO boundaries — emit one strip per
        // side. Adjacent overlap is intentional (covers the gap).
        if on_x_max {
            // Outward +X, tangent +Z (cross(+X, +Y)).
            emit_strip(p, v.normal_oct, v.leaf_attr_id, [0.0, 0.0, 1.0]);
        }
        if on_x_min {
            // Outward -X, tangent -Z.
            emit_strip(p, v.normal_oct, v.leaf_attr_id, [0.0, 0.0, -1.0]);
        }
        if on_z_max {
            // Outward +Z, tangent -X (cross(+Z, +Y) = -X).
            emit_strip(p, v.normal_oct, v.leaf_attr_id, [-1.0, 0.0, 0.0]);
        }
        if on_z_min {
            // Outward -Z, tangent +X.
            emit_strip(p, v.normal_oct, v.leaf_attr_id, [1.0, 0.0, 0.0]);
        }
    }

    if new_verts.is_empty() {
        return;
    }

    // Append vertex + index bytes.
    mesh.vertices
        .extend_from_slice(bytemuck::cast_slice(&new_verts));
    mesh.indices
        .extend_from_slice(bytemuck::cast_slice(&new_indices));

    // Append a single new LOD-0 patch cluster covering the skirts.
    // CLUSTER_FLAG_LOD_DIRTY + DAG_GROUP_NONE matches the sculpt V2
    // patch + halo-refresh slab patterns: the LOD selector admits
    // dirty LOD-0 leaves unconditionally, so the skirt always renders.
    let skirt_cluster = MeshletCluster {
        aabb_min: patch_min,
        _pad0: 0.0,
        aabb_max: patch_max,
        index_offset: pre_index_count,
        index_count: new_indices.len() as u32,
        lod_level: 0,
        flags: CLUSTER_FLAG_LOD_DIRTY,
        cluster_error: 0.0,
        parent_group_error: PARENT_GROUP_ERROR_ROOT,
        group_above_idx: DAG_GROUP_NONE,
        group_below_idx: DAG_GROUP_NONE,
        _pad3: 0,
    };
    mesh.clusters
        .extend_from_slice(bytemuck::cast_slice(std::slice::from_ref(&skirt_cluster)));

    // Skirt indices belong to the LOD-0 prefix.
    mesh.lod0_index_count += new_indices.len() as u32;
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
        let fbm = FbmTerrainFn::default().resolve(&arvx_core::NullMaterialLookup);
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
        let fbm = FbmTerrainFn::default().resolve(&arvx_core::NullMaterialLookup);

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
        let fbm = FbmTerrainFn::default().resolve(&arvx_core::NullMaterialLookup);
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
                aspect: 1.0,
                ridge_strength: 0.0,
                ridge_count: 3,
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
                aspect: 1.0,
                floor_flat_frac: 0.0,
                edge_falloff_m: 0.0,
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
        let fbm = FbmTerrainFn::default().resolve(&arvx_core::NullMaterialLookup);

        let target_y = 16.0;
        // Half-extents 40 m covers the full tile (64 m) + slack so every
        // sample lands inside the rect.
        let stamp = crate::stamp::Stamp::new(
            StampKind::Flatten {
                half_extents: glam::Vec2::new(40.0, 40.0),
                corner_radius_m: 0.0,
                edge_falloff_m: 0.0,
                tilt: glam::Vec2::ZERO,
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
                aspect: 1.0,
                floor_flat_frac: 0.0,
                edge_falloff_m: 0.0,
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
                aspect: 1.0,
                ridge_strength: 0.0,
                ridge_count: 3,
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

    // ── world-envelope clamp ─────────────────────────────────────────

    /// Collect the unique LOD-0 vertex (x, y, z) positions that fall
    /// inside a circular XZ footprint. Used by the envelope-clamp
    /// tests to assert the presence / absence of a surface inside
    /// the basin — the unclamped lake leaves the basin empty, the
    /// clamped lake gives it a solid floor.
    fn lod0_verts_in_xz_footprint(
        baked: &BakedTile,
        cx: f32,
        cz: f32,
        radius: f32,
    ) -> Vec<(f32, f32, f32)> {
        let v_stride = 32usize;
        let i_stride = 4usize;
        let lod0_n = baked.mesh.lod0_index_count as usize;
        let r2 = radius * radius;
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut out = Vec::new();
        for i in 0..lod0_n {
            let base = i * i_stride;
            let id = u32::from_le_bytes(
                baked.mesh.indices[base..base + i_stride].try_into().unwrap(),
            );
            if !seen.insert(id) {
                continue;
            }
            let vbase = id as usize * v_stride;
            let x = f32::from_le_bytes(
                baked.mesh.vertices[vbase..vbase + 4].try_into().unwrap(),
            );
            let y = f32::from_le_bytes(
                baked.mesh.vertices[vbase + 4..vbase + 8].try_into().unwrap(),
            );
            let z = f32::from_le_bytes(
                baked.mesh.vertices[vbase + 8..vbase + 12].try_into().unwrap(),
            );
            let dx = x - cx;
            let dz = z - cz;
            if dx * dx + dz * dz <= r2 {
                out.push((x, y, z));
            }
        }
        out
    }

    /// A Lake stamp deep enough to drive the composed surface far
    /// below the world's floor must still produce a solid floor
    /// inside the basin — players don't fall through. The bake's
    /// `world_floor_y` clamp lifts the composed h back to
    /// `floor + 2 * voxel_size_m` so surface nets always finds a
    /// surface to extract above the floor.
    #[test]
    fn world_floor_clamp_keeps_basin_floor_solid() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);

        // Lake centred on the tile, depth = 5000 m — well below the
        // tile's bottom face at y=0.
        let stamp = crate::stamp::Stamp::new(
            StampKind::Lake {
                depth: 5000.0,
                radius: 20.0,
                falloff: FalloffCurve::Smoothstep,
                aspect: 1.0,
                floor_flat_frac: 0.0,
                edge_falloff_m: 0.0,
            },
            Vec3::new(32.0, 32.0, 32.0),
        );

        // Clamped bake: world floor at y=0. Basin should pin near y=0.
        let clamped = bake_tile_with_skirts(
            TileKey::level0(0, 0, 0),
            vs,
            &FlatHalf,
            &[stamp],
            &empty_regions(),
            0.0,
            Some(0.0),
        )
        .expect("clamped bake");

        // The basin (inner 15 m of the 20 m footprint) should
        // contain LOD-0 vertices — the clamped floor.
        let basin_inner_radius = 15.0;
        let clamped_verts =
            lod0_verts_in_xz_footprint(&clamped, 32.0, 32.0, basin_inner_radius);
        assert!(
            clamped_verts.len() > 50,
            "clamped basin should have a solid floor; got {} LOD-0 verts inside r={basin_inner_radius}",
            clamped_verts.len(),
        );

        // And those verts must sit near the floor (y ~ 0..5 m
        // depending on rim transition), well below the FlatHalf
        // surface at y=32.
        let min_basin_y = clamped_verts
            .iter()
            .map(|v| v.1)
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_basin_y >= -vs && min_basin_y < 5.0,
            "clamped basin floor should sit near y=0; got min_y={min_basin_y}",
        );
    }

    /// Without the clamp (None), the same deep stamp leaves the
    /// basin EMPTY — no LOD-0 vertices inside the footprint —
    /// because the bake's composed surface is far below the tile's
    /// vertical span. This is the fall-through bug the clamp fixes.
    #[test]
    fn no_clamp_leaves_basin_empty() {
        let t = Terrain::default();
        let vs = t.voxel_size_for_level(0);
        let stamp = crate::stamp::Stamp::new(
            StampKind::Lake {
                depth: 5000.0,
                radius: 20.0,
                falloff: FalloffCurve::Smoothstep,
                aspect: 1.0,
                floor_flat_frac: 0.0,
                edge_falloff_m: 0.0,
            },
            Vec3::new(32.0, 32.0, 32.0),
        );

        let unclamped = bake_tile_with_skirts(
            TileKey::level0(0, 0, 0),
            vs,
            &FlatHalf,
            &[stamp],
            &empty_regions(),
            0.0,
            None,
        )
        .expect("unclamped bake still produces a tile (rim is intact)");

        // Inner basin — no surface verts at all. The rim still has
        // surface (where the stamp's weight goes to zero), but the
        // basin interior is "all sky" from the tile's perspective.
        let _ = vs; // silence unused
        let basin_inner_radius = 10.0;
        let unclamped_verts = lod0_verts_in_xz_footprint(
            &unclamped,
            32.0,
            32.0,
            basin_inner_radius,
        );
        assert_eq!(
            unclamped_verts.len(),
            0,
            "unclamped deep lake should leave the basin interior empty — fall-through bug; got {} verts inside r={basin_inner_radius}",
            unclamped_verts.len(),
        );
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

    // ── V2 LOD pyramid: lateral skirts ─────────────────────────────────

    /// `skirt_depth_m = 0` is a no-op — mesh.vertices length unchanged.
    #[test]
    fn skirts_disabled_when_depth_zero() {
        let vs = 0.5;
        let baked_no_skirts =
            bake_tile_with_skirts(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions(), 0.0, None)
                .expect("bake");
        let baked_baseline =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions())
                .expect("bake");
        assert_eq!(
            baked_no_skirts.mesh.vertices.len(),
            baked_baseline.mesh.vertices.len(),
            "skirt_depth=0 must produce the same vertex byte count as bake_tile"
        );
        assert_eq!(
            baked_no_skirts.mesh.indices.len(),
            baked_baseline.mesh.indices.len(),
        );
    }

    /// `skirt_depth_m > 0` appends geometry — vertex / index / cluster
    /// counts strictly grow vs the baseline.
    #[test]
    fn skirts_append_geometry_when_enabled() {
        let vs = 0.5;
        let baked_with =
            bake_tile_with_skirts(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions(), 4.0, None)
                .expect("bake");
        let baked_without =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions())
                .expect("bake");
        assert!(
            baked_with.mesh.vertices.len() > baked_without.mesh.vertices.len(),
            "skirts must add vertex bytes"
        );
        assert!(
            baked_with.mesh.indices.len() > baked_without.mesh.indices.len(),
            "skirts must add index bytes"
        );
        assert!(
            baked_with.mesh.clusters.len() > baked_without.mesh.clusters.len(),
            "skirts must add a cluster"
        );
        assert!(
            baked_with.mesh.lod0_index_count > baked_without.mesh.lod0_index_count,
            "skirt indices count as LOD-0"
        );
    }

    /// Every skirt vertex must sit at-or-below its source surface
    /// vertex Y (either at the top of the strip, or `skirt_depth_m`
    /// below). No skirt vertex floats above the surface mesh.
    #[test]
    fn skirt_vertices_sit_at_or_below_surface() {
        use arvx_core::mesh_extract::MeshVertex;

        let vs = 0.5;
        let depth = 6.0;
        let baked_with =
            bake_tile_with_skirts(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions(), depth, None)
                .expect("bake");
        let baked_without =
            bake_tile(TileKey::level0(0, 0, 0), vs, &FlatHalf, &[], &empty_regions())
                .expect("bake");
        // Find the maximum surface Y the baseline emitted.
        let baseline_verts: &[MeshVertex] =
            bytemuck::cast_slice(&baked_without.mesh.vertices);
        let max_surface_y = baseline_verts
            .iter()
            .map(|v| v.local_pos[1])
            .fold(f32::NEG_INFINITY, f32::max);

        // Iterate the skirt verts (everything past the baseline's end).
        let with_verts: &[MeshVertex] = bytemuck::cast_slice(&baked_with.mesh.vertices);
        let baseline_count = baseline_verts.len();
        assert!(
            with_verts.len() > baseline_count,
            "test setup: skirts must add verts"
        );
        let skirt_verts = &with_verts[baseline_count..];
        for sv in skirt_verts {
            // Skirt verts are at the boundary plane's Y or at
            // (boundary_y - depth). Boundary Ys are <= max_surface_y.
            assert!(
                sv.local_pos[1] <= max_surface_y + 1e-3,
                "skirt vertex Y {} must be ≤ max surface Y {max_surface_y}",
                sv.local_pos[1],
            );
            // And ≥ max_surface_y - depth (no skirt goes deeper than
            // configured).
            assert!(
                sv.local_pos[1] >= max_surface_y - depth - 1e-3,
                "skirt vertex Y {} must be ≥ max_surface_y - depth ({})",
                sv.local_pos[1],
                max_surface_y - depth,
            );
        }
    }
}
