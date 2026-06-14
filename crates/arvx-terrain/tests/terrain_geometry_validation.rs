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
use arvx_terrain::{FbmTerrainFn, TerrainFn};
use glam::Vec3;

fn le_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Octahedral unpack matching `LeafAttr::normal_oct` (2× snorm16 in a u32).
fn unpack_oct(p: u32) -> Vec3 {
    let x = ((p & 0xffff) as u16 as i16) as f32 / 32767.0;
    let y = ((p >> 16) as u16 as i16) as f32 / 32767.0;
    let mut n = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
    if n.z < 0.0 {
        let nx = (1.0 - n.y.abs()) * if n.x >= 0.0 { 1.0 } else { -1.0 };
        let ny = (1.0 - n.x.abs()) * if n.y >= 0.0 { 1.0 } else { -1.0 };
        n.x = nx;
        n.y = ny;
    }
    n.normalize_or_zero()
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

/// **Default new-scene terrain bakes SMOOTH shading normals.** Regression
/// for the "default terrain looks rough" bug: a fresh `Spawn → Terrain` is
/// a *Bounded* terrain, so `world_floor_y()` is `Some(0.0)` and the bake
/// receives a non-`None` floor — the path the streamer actually drives. The
/// old whole-tile gate disabled the analytic shading normal for ANY tile of
/// a floored terrain (every peak included), falling back to the `∇D` normal
/// whose ~5° mean / ~22° max deviation reads as fine rippled/contour noise.
///
/// The sibling `terrain_bake_produces_renderable_geometry` bakes with
/// `world_floor_y = None` (the analytic path) and so never caught this — it
/// did not exercise what the editor runs. Here we bake the FLOORED path and
/// assert each vertex's stored shading normal matches the true surface
/// normal of the COMPOSED height field (raw FBM above the floor; flat +Y on
/// any clamped sub-floor column).
#[test]
fn floored_default_terrain_has_smooth_shading_normals() {
    let fbm = FbmTerrainFn::default().resolve(&NullMaterialLookup);
    let regions = TerrainRegionSnapshot::new();
    let vs = 0.25_f32; // Terrain::default() level-0 voxel size.
    let skirt = 4.0_f32;
    // The editor passes `terrain.world_floor_y()` = Some(0.0) for the
    // default Bounded terrain; the envelope floor is then `0.0 + 2·vs`.
    let floor_y = 0.0_f32;
    let envelope_floor = floor_y + 2.0 * vs;

    // Find the column-(0,*,0) tile with the most geometry — the heavy
    // surface tile the editor renders up close.
    let mut best: Option<(i32, usize)> = None;
    for ty in -2..=6 {
        let key = TileKey::level0(0, ty, 0);
        if let Some(b) =
            bake_tile_with_skirts(key, vs, &fbm, &[], &regions, skirt, Some(floor_y))
        {
            let vc = b.mesh.vertices.len() / 32;
            if best.map(|(_, c)| vc > c).unwrap_or(true) {
                best = Some((ty, vc));
            }
        }
    }
    let (ty, _) = best.expect("a surface tile in column (0,*,0)");
    let key = TileKey::level0(0, ty, 0);
    let tile_origin = key.origin_world().to_vec3();

    let baked = bake_tile_with_skirts(key, vs, &fbm, &[], &regions, skirt, Some(floor_y))
        .expect("floored bake");
    let verts = &baked.mesh.vertices;
    let vcount = verts.len() / 32;
    assert!(vcount > 1000, "heavy tile should have plenty of vertices, got {vcount}");

    // Per-vertex deviation of the stored shading normal from the true
    // composed-field normal. Skirt apron verts (the vertical drop below the
    // tile floor) are excluded — they are intentionally near-horizontal.
    let mut devs: Vec<f32> = Vec::with_capacity(vcount);
    for vi in 0..vcount {
        let o = vi * 32;
        let local = Vec3::new(le_f32(verts, o), le_f32(verts, o + 4), le_f32(verts, o + 8));
        let world = tile_origin + local;
        // Skip skirt apron verts: their stored normal is the zero-vector
        // ∇D fallback (length ~0), filtered just below.
        let n_stored = unpack_oct(le_u32(verts, o + 12));
        if n_stored.length_squared() < 0.5 {
            continue;
        }
        // Raw FBM height at this column: sd = query_y − h_fbm.
        let sd = fbm.sample(key, local, vs).sd;
        let h_fbm = world.y - sd;
        // True normal of the baked (clamped) surface.
        let truth = if h_fbm < envelope_floor {
            Vec3::Y // clamped flat floor
        } else {
            match fbm.sample_grad(key, local, vs) {
                Some(g) => g.normalize_or_zero(),
                None => continue,
            }
        };
        if truth.length_squared() < 0.5 {
            continue;
        }
        let dot = n_stored.dot(truth).clamp(-1.0, 1.0);
        devs.push(dot.acos().to_degrees());
    }
    assert!(devs.len() > 1000, "need enough surface verts, got {}", devs.len());
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = devs.iter().sum::<f32>() / devs.len() as f32;
    let p95 = devs[((devs.len() as f32 * 0.95) as usize).min(devs.len() - 1)];
    let max = *devs.last().unwrap();
    eprintln!(
        "[floored-normals] tile=(0,{ty},0) {} verts | shading-normal dev vs composed field: \
         mean={mean:.3}° p95={p95:.3}° max={max:.3}°",
        devs.len(),
    );

    // Pre-fix the FLOORED path measured mean≈5.1° / p95≈12.2° / max≈21.8°
    // (the rippled ∇D look). The analytic path is mean≈0.01°. Assert we are
    // firmly on the analytic side — generously, to leave headroom for the
    // octahedral pack floor and any genuinely-clamped crease verts.
    assert!(
        mean < 0.5,
        "default floored terrain shading normals are rough (mean {mean:.3}°) — \
         the per-sample analytic-normal gate has regressed",
    );
    assert!(p95 < 1.5, "p95 shading-normal deviation too high: {p95:.3}°");
}
