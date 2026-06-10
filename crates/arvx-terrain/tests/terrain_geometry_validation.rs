//! Headless GPU-hostility check for baked terrain geometry.
//!
//! The editor's terrain-gen freezes the whole desktop (a GPU TDR) at a
//! deterministic point. The render thread blocks forever waiting for GPU
//! work that never completes — i.e. the GPU *hangs*. A whole-GPU hang with
//! no validation error is almost always **NaN/Inf or wildly out-of-bounds
//! geometry** fed to the rasterizer, or a cluster index range that runs
//! past the index buffer (→ out-of-bounds vertex fetch).
//!
//! Terrain baking is 100% CPU (`voxelize_to_artifact` + surface-nets +
//! cluster-DAG — no GPU), so we can reproduce the *exact* geometry the
//! editor uploads and validate it here, with no GPU and no freeze. This
//! bakes the default terrain's resident tile grid through the real
//! `bake_tile_with_skirts` path and asserts the mesh is renderable:
//!   * every vertex position is finite and within a sane magnitude,
//!   * every index is < vertex count,
//!   * every cluster's `[index_offset, index_offset+index_count)` fits
//!     the index buffer and is a whole number of triangles.

use arvx_core::NullMaterialLookup;
use arvx_terrain::bake::bake_tile_with_skirts;
use arvx_terrain::region_snapshot::TerrainRegionSnapshot;
use arvx_terrain::tile_key::TileKey;
use arvx_terrain::FbmTerrainFn;

fn le_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[test]
fn terrain_bake_produces_renderable_geometry() {
    // The default FbmTerrainFn is what `Spawn → Terrain` creates.
    let fbm = FbmTerrainFn::default().resolve(&NullMaterialLookup);
    let regions = TerrainRegionSnapshot::new();
    let skirt_depth_m = 4.0_f32; // Terrain::default().skirt_depth_m

    // LOD voxel sizes the streamer uses (finest first). 0.25 m is the
    // depth-8 level-0 bake that dominates the editor's heavy tiles.
    let voxel_sizes = [0.25_f32, 0.5, 1.0, 2.0];

    // Tile grid: 64 m tiles. A 192 m terrain is ~3×3 columns; bake a
    // generous 5×5 column footprint over a vertical span that brackets
    // the FBM surface. Empty sky/underground tiles bake fast.
    let (tx_range, tz_range, ty_range) = (-2..=2, -2..=2, -3..=6);

    let mut problems: Vec<String> = Vec::new();
    let mut surface_tiles = 0usize;

    'scan: for &vs in &voxel_sizes {
        for ty in ty_range.clone() {
            for tx in tx_range.clone() {
                for tz in tz_range.clone() {
                    let key = TileKey::level0(tx, ty, tz);
                    let Some(baked) = bake_tile_with_skirts(
                        key, vs, &fbm, &[], &regions, skirt_depth_m, None,
                    ) else {
                        continue;
                    };

                    let verts = &baked.mesh.vertices;
                    let idx = &baked.mesh.indices;
                    let clusters = &baked.mesh.clusters;
                    let vcount = verts.len() / 32;
                    let icount = idx.len() / 4;
                    if vcount == 0 {
                        continue;
                    }
                    surface_tiles += 1;
                    let tag = format!("tile=({tx},{ty},{tz}) vs={vs}");

                    // 1. Vertex positions (object-local, bytes 0..12).
                    //    Within a tile they must be ~[0, 64]+skirt; flag
                    //    anything non-finite or absurdly far.
                    for vi in 0..vcount {
                        let o = vi * 32;
                        let (x, y, z) = (le_f32(verts, o), le_f32(verts, o + 4), le_f32(verts, o + 8));
                        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                            problems.push(format!("{tag}: NON-FINITE vertex {vi} = ({x}, {y}, {z})"));
                        } else if x.abs() > 5000.0 || y.abs() > 5000.0 || z.abs() > 5000.0 {
                            problems.push(format!("{tag}: HUGE vertex {vi} = ({x}, {y}, {z})"));
                        }
                        if problems.len() >= 50 {
                            break 'scan;
                        }
                    }

                    // 2. Indices must reference existing vertices.
                    for ii in 0..icount {
                        let v = le_u32(idx, ii * 4) as usize;
                        if v >= vcount {
                            problems.push(format!(
                                "{tag}: OOB index[{ii}] = {v} >= vcount {vcount}"
                            ));
                            if problems.len() >= 50 {
                                break 'scan;
                            }
                        }
                    }

                    // 3. Cluster index ranges (MeshletCluster: 64 B;
                    //    index_offset@28, index_count@32) must fit the IBO
                    //    and be whole triangles. An over-range cluster is
                    //    drawn via `multi_draw_indexed_indirect` → the GPU
                    //    fetches indices past the IBO → OOB vertex fetch.
                    let ccount = clusters.len() / 64;
                    for ci in 0..ccount {
                        let c = ci * 64;
                        let off = le_u32(clusters, c + 28) as usize;
                        let cnt = le_u32(clusters, c + 32) as usize;
                        if off + cnt > icount {
                            problems.push(format!(
                                "{tag}: cluster {ci} range [{off}, {}) past IBO (icount {icount})",
                                off + cnt
                            ));
                        } else if cnt % 3 != 0 {
                            problems.push(format!(
                                "{tag}: cluster {ci} index_count {cnt} not a triangle multiple"
                            ));
                        }
                        if problems.len() >= 50 {
                            break 'scan;
                        }
                    }
                }
            }
        }
    }

    eprintln!(
        "[terrain-geo-validation] baked grid, {surface_tiles} non-empty tiles, {} problems",
        problems.len()
    );
    assert!(
        problems.is_empty(),
        "Found {} GPU-hostile geometry problem(s) in baked terrain:\n{}",
        problems.len(),
        problems.join("\n")
    );
    assert!(surface_tiles > 0, "no surface tiles baked — grid missed the terrain surface");
}
