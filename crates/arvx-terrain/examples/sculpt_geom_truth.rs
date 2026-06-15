//! Ground-truth for SCULPT geometry: render the ACTUAL mesh a Raise brush
//! produces, so terracing (geometry) and speckle (shading) can be judged
//! HEADLESSLY — no editor round-trip.
//!
//! Pipeline = the real sculpt path on a flat voxelized ground:
//!   voxelize_octree (flat plane, with per-leaf distances)
//!   → compute_brush_edits + apply_delta  (the brush kernel)
//!   → extract_surface_mesh_density_haloed (the distance-aware DC mesher)
//!   → render FACE normals + vertex normals + side wireframe to PNG.
//!
//! FACE-normal render is the terracing-truth view (a smooth vertex normal
//! HIDES geometric stair-steps; the per-triangle face normal exposes them).
//!
//!   cargo run -p arvx-terrain --example sculpt_geom_truth --release
//! Flags (A/B the fixes): ARVX_SCULPT_BAND=0, ARVX_SCULPT_ANALYTIC_NORMAL=0,
//! ARVX_MESH_DC=0 (legacy QEF), ARVX_QEF_HERMITE=0 (blur fallback).

use arvx_core::mesh_extract::{
    collect_cell_map_in_region, extract_mesh_region_from_cells_pooled_haloed,
    extract_surface_mesh_density_haloed, MeshVertex, SculptExtractScratch,
};
use arvx_core::mesh_test_bench::{render, voxelize as bench_voxelize, Camera, Image, RenderOpts, Shape};
use arvx_core::sculpt::{apply_delta, BrushMode, BrushOp, FalloffCurve};
use arvx_core::voxelize_octree::voxelize_octree;
use arvx_core::{Aabb, BrickPool, LeafAttrPool, NullMaterialLookup};
use arvx_terrain::bake::bake_tile_with_skirts;
use arvx_terrain::region_snapshot::TerrainRegionSnapshot;
use arvx_terrain::tile_key::TileKey;
use arvx_terrain::FbmTerrainFn;
use glam::{IVec3, Vec3};
use std::path::{Path, PathBuf};

fn le_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

const SIZE: u32 = 900;
const VS: f32 = 0.25;
const N: u32 = 64; // grid cells per axis (2^6) → 16 m tile

fn save(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.rgb.clone())
        .expect("rgb")
        .save(path)
        .unwrap();
}

/// Octahedral pack (matches `arvx_core::leaf_attr::pack_oct`; local copy so
/// the example doesn't depend on a possibly-private helper).
fn pack_oct(n: Vec3) -> u32 {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-6);
    let (mut x, mut y) = (n.x, n.y);
    if n.z < 0.0 {
        let ox = (1.0 - y.abs()) * if x >= 0.0 { 1.0 } else { -1.0 };
        let oy = (1.0 - x.abs()) * if y >= 0.0 { 1.0 } else { -1.0 };
        x = ox;
        y = oy;
    }
    let xi = (x.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16 as u16 as u32;
    let yi = (y.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16 as u16 as u32;
    xi | (yi << 16)
}

/// Replace each vertex normal with its triangle's geometric FACE normal.
fn with_face_normals(verts: &[MeshVertex], idx: &[u32]) -> Vec<MeshVertex> {
    let mut v = verts.to_vec();
    for tri in idx.chunks_exact(3) {
        let p0 = Vec3::from(v[tri[0] as usize].local_pos);
        let p1 = Vec3::from(v[tri[1] as usize].local_pos);
        let p2 = Vec3::from(v[tri[2] as usize].local_pos);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let packed = pack_oct(n);
        for &t in tri {
            v[t as usize].normal_oct = packed;
        }
    }
    v
}

fn main() {
    let out = PathBuf::from("target/sculpt_geom_truth");
    std::fs::create_dir_all(&out).unwrap();
    let band = std::env::var("ARVX_SCULPT_BAND").as_deref() != Ok("0");
    let analytic = std::env::var("ARVX_SCULPT_ANALYTIC_NORMAL").as_deref() != Ok("0");
    let tag = format!(
        "band{}_norm{}",
        if band { 1 } else { 0 },
        if analytic { 1 } else { 0 }
    );

    // ── 1. Build the surface. FBM (default) bakes REAL editor terrain and
    // reconstructs live pools from the artifact; ARVX_SCULPT_TRUTH_FBM=0
    // falls back to a flat/sloped voxelized plane.
    let use_fbm = std::env::var("ARVX_SCULPT_TRUTH_FBM").as_deref() != Ok("0");
    let (mut octree, mut leaf, mut bricks, grid_origin, n_grid, surf_center) = if use_fbm {
        let fbm = FbmTerrainFn::default().resolve(&NullMaterialLookup);
        let regions = TerrainRegionSnapshot::new();
        // Heaviest surface tile in column (0,*,0) — like terrain_geom_truth.
        let mut best: Option<(i32, usize)> = None;
        for ty in -2..=6 {
            if let Some(b) =
                bake_tile_with_skirts(TileKey::level0(0, ty, 0), VS, &fbm, &[], &regions, 4.0, Some(0.0))
            {
                let vc = b.mesh.vertices.len() / 32;
                if best.map(|(_, c)| vc > c).unwrap_or(true) {
                    best = Some((ty, vc));
                }
            }
        }
        let ty = best.expect("a surface tile exists").0;
        let baked = bake_tile_with_skirts(TileKey::level0(0, ty, 0), VS, &fbm, &[], &regions, 4.0, Some(0.0)).unwrap();
        // Surface centroid (object-local) from the baked mesh — skip skirts
        // (normal_oct == 0). Used to place the brush on the surface.
        let mb = &baked.mesh.vertices;
        let (mut slo, mut shi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for vi in 0..(mb.len() / 32) {
            let o = vi * 32;
            if le_u32(mb, o + 12) == 0 {
                continue;
            }
            let p = Vec3::new(le_f32(mb, o), le_f32(mb, o + 4), le_f32(mb, o + 8));
            slo = slo.min(p);
            shi = shi.max(p);
        }
        // The brush must land ON the surface at the centre XZ — not at the
        // bbox-centre y (mid-height of the whole rolling surface, which is
        // usually below the local surface → a buried brush adds nothing).
        // Average the surface verts within 2 m of the centre XZ column.
        let (ccx, ccz) = ((slo.x + shi.x) * 0.5, (slo.z + shi.z) * 0.5);
        let (mut acc, mut nacc) = (Vec3::ZERO, 0u32);
        for vi in 0..(mb.len() / 32) {
            let o = vi * 32;
            if le_u32(mb, o + 12) == 0 {
                continue;
            }
            let p = Vec3::new(le_f32(mb, o), le_f32(mb, o + 4), le_f32(mb, o + 8));
            let (dx, dz) = (p.x - ccx, p.z - ccz);
            if dx * dx + dz * dz < 4.0 {
                acc += p;
                nacc += 1;
            }
        }
        let surf_center = if nacc > 0 { acc / nacc as f32 } else { (slo + shi) * 0.5 };
        let art = baked.artifact;
        let n_attrs = art.leaf_attrs.len();
        let mut leaf = LeafAttrPool::new((n_attrs as u32).max(1) + 16384);
        leaf.allocate_contiguous_bump(n_attrs as u32);
        for (i, a) in art.leaf_attrs.iter().enumerate() {
            *leaf.get_mut(i as u32) = *a;
        }
        for (i, d) in art.leaf_attr_dists.iter().enumerate() {
            leaf.set_dist_quantized(i as u32, *d);
        }
        let n_bricks = art.brick_cells.len();
        let mut bricks = BrickPool::new((n_bricks as u32).max(1));
        bricks.allocate_contiguous_bump(n_bricks as u32);
        for (b, cells) in art.brick_cells.iter().enumerate() {
            bricks.brick_cells_mut(b as u32).copy_from_slice(cells);
        }
        let n_grid = 1u32 << art.octree.depth();
        eprintln!("[sculpt-truth] FBM tile (0,{ty},0): {n_attrs} leaves, surf_center={surf_center:?}");
        (art.octree, leaf, bricks, art.grid_origin, n_grid, surf_center)
    } else {
        let extent = N as f32 * VS;
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(extent));
        let g = extent * 0.5;
        let slope: f32 = std::env::var("ARVX_SCULPT_TRUTH_SLOPE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.35);
        let cxf = extent * 0.5;
        let mut sdf = |ps: &[Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<Vec3>)> {
            ps.iter().map(|p| (p.y - (g + slope * (p.x - cxf)), 0u16, 0u16, 0u8, 0u32, None)).collect()
        };
        let mut leaf = LeafAttrPool::new(8192);
        let mut bricks = BrickPool::new(1024);
        let res = voxelize_octree(&mut sdf, &aabb, VS, &mut leaf, &mut bricks, 0).expect("voxelize");
        let sc = res.grid_origin + Vec3::splat(g);
        (res.octree, leaf, bricks, res.grid_origin, N, sc)
    };

    // ── 2. Raise at the surface centre. Single stamp by default;
    // ARVX_SCULPT_TRUTH_STROKE=1 sweeps a multi-stamp drag.
    let ground = surf_center.y;
    // Brush centre in GRID units (the kernel works in finest-voxel coords).
    let cg = (surf_center - grid_origin) / VS;
    let stroke = std::env::var("ARVX_SCULPT_TRUTH_STROKE").as_deref() == Ok("1");
    let radius = 8.0_f32;
    // Brush MODE under test. The editor's soft brushes (Inflate/Deflate/
    // ClayStrip) use a DIFFERENT brushfire kernel that the Raise/Carve band
    // fix does not touch — test them here.
    let mode_str = std::env::var("ARVX_SCULPT_TRUTH_MODE").unwrap_or_else(|_| "raise".into());
    let (mode, strength) = match mode_str.as_str() {
        "carve" => (BrushMode::Carve, 0.0),
        "inflate" => (BrushMode::Inflate, 6.0),
        "deflate" => (BrushMode::Deflate, 6.0),
        "claystrip" => (BrushMode::ClayStrip, 6.0),
        "smooth" => (BrushMode::Smooth, 16.0),
        _ => (BrushMode::Raise, 0.0),
    };
    eprintln!("[sculpt-truth] mode={mode_str} strength={strength}");
    // STROKE=1 sweeps a drag; HOLD=1 repeats the SAME center N times (stress
    // the stroke-capping: a held brush must converge to one offset layer, not
    // compound). Both route through the stroke `touched` set below.
    let hold = std::env::var("ARVX_SCULPT_TRUTH_HOLD").as_deref() == Ok("1");
    let centres: Vec<Vec3> = if hold {
        vec![cg; 6]
    } else if stroke {
        (0..7).map(|i| {
            let t = (i as f32 / 6.0 - 0.5) * 2.0; // -1..1
            cg + Vec3::new(t * (n_grid as f32 * 0.20), 0.0, 0.0)
        }).collect()
    } else {
        vec![cg]
    };
    let _ = n_grid;
    // Per-stroke "already edited" set — only populated for stroke/hold so a
    // single stamp keeps the every-leaf-seeds path. Mirrors the scene
    // manager's `sculpt_stroke_touched`.
    let multi = stroke || hold;
    let mut touched: std::collections::HashSet<glam::UVec3> = std::collections::HashSet::new();

    let mut prev = centres[0];
    let mut total_added = 0usize;
    for &centre in &centres {
        let op = BrushOp {
            center: centre,
            segment_start: prev, // capsule sweep from the previous stamp
            radius,
            falloff_curve: FalloffCurve::Smooth,
            strength,
            mode,
            material: 0,
        };
        prev = centre;
        let delta = arvx_core::sculpt::compute_brush_edits_in_stroke(
            &octree,
            &bricks,
            leaf.as_slice(),
            leaf.dists_as_slice(),
            op,
            |c| multi && touched.contains(&c),
        );
        if multi {
            for e in &delta.edits {
                touched.insert(e.coord);
            }
        }
        total_added += delta.count_added();
        let n_added = delta.count_added() as u32;
        if delta.is_empty() {
            continue;
        }
        let base = if n_added > 0 {
            leaf.allocate_contiguous_bump(n_added).expect("alloc slots")
        } else {
            0
        };
        let mut next = base;
        let applied = apply_delta(&mut octree, &mut bricks, &delta, || {
            let s = next;
            next += 1;
            s
        });
        for (slot, attrs) in &applied.allocated_slots {
            *leaf.get_mut(*slot) = attrs.to_leaf_attr();
            leaf.set_dist(*slot, attrs.dist);
        }
        for (slot, normal) in &applied.renormalized_slots {
            leaf.get_mut(*slot).normal_oct = pack_oct(*normal);
        }
        // SDF-offset Inflate/Deflate rewrite existing leaves' stored distance
        // in place — the harness must mirror the editor's `redist_slots` drain
        // or the re-extract reads stale near-zero distances and shatters.
        for (slot, normal, dist) in &applied.redist_slots {
            leaf.get_mut(*slot).normal_oct = pack_oct(*normal);
            leaf.set_dist(*slot, *dist);
        }
    }
    eprintln!(
        "[sculpt-truth] {} stamp(s), {} cells added total",
        centres.len(),
        total_added,
    );

    // ── DEBUG: vertical X-Y slice of occupancy + dist through the brush
    // center (ARVX_SCULPT_TRUTH_DUMP=1). For each cell: ' '=empty, '#'=deep
    // interior, else a digit/sign of the stored dist (inside ≤0). Lets the
    // kernel's re-discretised field be eyeballed without the mesher.
    if std::env::var("ARVX_SCULPT_TRUTH_DUMP").as_deref() == Ok("1") {
        use arvx_core::sparse_octree::CellState;
        let cz = cg.z.round() as u32;
        let cx0 = cg.x.round() as i32;
        let cy0 = cg.y.round() as i32;
        eprintln!("[dump] vertical slice z={cz}, x∈[{}..{}], y high→low", cx0 - 12, cx0 + 12);
        for yy in (cy0 - 2..=cy0 + 10).rev() {
            let mut row = String::new();
            for xx in (cx0 - 12)..=(cx0 + 12) {
                if xx < 0 || yy < 0 {
                    row.push(' ');
                    continue;
                }
                let c = glam::UVec3::new(xx as u32, yy as u32, cz);
                let ch = match octree.cell_state(c, &bricks) {
                    CellState::Empty | CellState::OutOfBounds => '.',
                    CellState::Interior => '#',
                    CellState::Solid(slot) => {
                        let d = leaf.dist(slot);
                        if d <= 0.0 {
                            // inside surface band: '-' shades by depth
                            if d > -0.5 { '0' } else if d > -1.0 { '1' } else if d > -1.5 { '2' } else { '3' }
                        } else {
                            '+' // surface leaf whose center is just outside
                        }
                    }
                };
                row.push(ch);
            }
            eprintln!("[dump] y={yy:>3} |{row}|");
        }
    }

    // ── 3. Distance-aware extract (the sculpt re-extract mesher). FULL
    // extract: the geometry (Manifold-DC placement on the stored distances)
    // is identical to the editor's region re-extract — both call
    // manifold_dc_placement on the same octree+dists, so a FACE-normal
    // render of this reproduces the editor's terracing exactly.
    let (verts, idx) = extract_surface_mesh_density_haloed(
        octree.as_slice(),
        octree.depth(),
        VS,
        grid_origin,
        bricks.as_slice(),
        leaf.as_slice(),
        leaf.bones_as_slice(),
        &[],
        0,
        None,
        leaf.dists_as_slice(),
        None,
    );
    eprintln!("[sculpt-truth] full mesh: {} verts, {} tris", verts.len(), idx.len() / 3);

    // ── 4. ALSO extract via the REGION path — the EDITOR's sculpt
    // re-extract. Shares manifold_dc_placement with the full extract, so a
    // geometry difference between the two is region-path-specific.
    let extent_i = n_grid as i32;
    let (mut rmin, mut rmax_b) = (IVec3::splat(extent_i), IVec3::ZERO);
    for &ctr in &centres {
        let lo = (ctr - Vec3::splat(radius + 2.0)).floor();
        let hi = (ctr + Vec3::splat(radius + 2.0)).ceil();
        rmin = rmin.min(IVec3::new(lo.x as i32, lo.y as i32, lo.z as i32));
        rmax_b = rmax_b.max(IVec3::new(hi.x as i32, hi.y as i32, hi.z as i32));
    }
    rmin = rmin.max(IVec3::ZERO);
    rmax_b = rmax_b.min(IVec3::splat(extent_i));
    let cells = collect_cell_map_in_region(
        octree.as_slice(),
        octree.depth(),
        bricks.as_slice(),
        rmin - IVec3::splat(3),
        rmax_b + IVec3::splat(3),
    );
    let mut scratch = SculptExtractScratch::new();
    let (rverts, ridx) = extract_mesh_region_from_cells_pooled_haloed(
        &mut scratch,
        &cells,
        rmin,
        rmax_b,
        octree.as_slice(),
        octree.depth(),
        VS,
        grid_origin,
        bricks.as_slice(),
        leaf.as_slice(),
        leaf.bones_as_slice(),
        &[],
        None,
        None::<&fn(Vec3) -> f32>,
        leaf.dists_as_slice(),
    );
    eprintln!(
        "[sculpt-truth] grid_origin={grid_origin:?} cg={cg:?} rmin={rmin:?} rmax={rmax_b:?} cells={}",
        cells.len()
    );
    eprintln!("[sculpt-truth] region mesh: {} verts, {} tris", rverts.len(), ridx.len() / 3);

    // ── 5. Render both extracts (face + vertex + side) and report metrics.
    render_set(&out, &format!("{tag}_full"), &verts, &idx, ground, surf_center, radius * VS);
    render_set(&out, &format!("{tag}_region"), &rverts, &ridx, ground, surf_center, radius * VS);
    eprintln!("[sculpt-truth] PNGs in {}", out.canonicalize().unwrap_or(out).display());
}

#[allow(clippy::too_many_arguments)]
fn render_set(
    out: &Path,
    tag: &str,
    verts: &[MeshVertex],
    idx: &[u32],
    ground: f32,
    bc: Vec3,
    br: f32,
) {
    if verts.is_empty() {
        eprintln!("[sculpt-truth/{tag}] EMPTY MESH");
        return;
    }
    // Frame on the raised cap (above the surface).
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for v in verts {
        let p = Vec3::from(v.local_pos);
        if p.y > ground - 0.5 {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let center = (lo + hi) * 0.5;
    let cam_r = ((hi - lo).length() * 0.5).max(1.0);
    let dummy = bench_voxelize(Shape::all()[0], Shape::all()[0].bounds(), 1.0);
    let cam_g = Camera::three_quarter_framing(SIZE, SIZE, center, cam_r * 1.15);
    let cam_s = Camera::side_ortho_framing(SIZE, SIZE, center, cam_r * 1.1);
    let vf = with_face_normals(verts, idx);
    save(
        &render(&cam_g, &dummy, &vf, idx,
            RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false }, SIZE),
        &out.join(format!("{tag}_face.png")),
    );
    save(
        &render(&cam_g, &dummy, verts, idx,
            RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false }, SIZE),
        &out.join(format!("{tag}_vertex.png")),
    );
    save(
        &render(&cam_s, &dummy, verts, idx,
            RenderOpts { shaded: true, wireframe: true, voxels: false, dim_shading: true }, SIZE),
        &out.join(format!("{tag}_side.png")),
    );
    let (mut rss, mut rmax, mut n) = (0.0f32, 0.0f32, 0u32);
    for v in verts {
        let p = Vec3::from(v.local_pos);
        if p.y <= ground + 0.2 {
            continue;
        }
        let r = (p - bc).length() - br;
        rss += r * r;
        rmax = rmax.max(r.abs());
        n += 1;
    }
    let rms = if n > 0 { (rss / n as f32).sqrt() } else { 0.0 };
    eprintln!(
        "[sculpt-truth/{tag}] cap {n} verts | sphere-fit residual rms={:.3} vox max={:.3} vox",
        rms / VS, rmax / VS,
    );
}
