//! Ground-truth: is the ACTUAL baked default-terrain geometry stepped?
//!
//! Renders the real `bake_tile_with_skirts` output (editor path) three ways
//! so we can separate GEOMETRY stepping from SHADING:
//!   * side_profile_wire   — side ortho, wireframe. A staircase silhouette
//!                           here = the geometry itself is stepped.
//!   * grazing_face        — grazing 3/4, shaded by per-triangle FACE normal
//!                           (reveals faceting the smooth normal hides).
//!   * grazing_vertex      — grazing 3/4, shaded by the stored vertex normal
//!                           (what the editor approximates).
//! For QEF (editor default) and the blur path (ARVX_QEF_HERMITE=0).
//!
//!   cargo run -p arvx-terrain --example terrain_geom_truth --release
//!   ARVX_QEF_HERMITE=0 cargo run -p arvx-terrain --example terrain_geom_truth --release

use arvx_core::mesh_extract::MeshVertex;
use arvx_core::mesh_test_bench::{render, voxelize, Camera, Image, RenderOpts, Shape};
use arvx_core::NullMaterialLookup;
use arvx_terrain::bake::bake_tile_with_skirts;
use arvx_terrain::region_snapshot::TerrainRegionSnapshot;
use arvx_terrain::tile_key::TileKey;
use arvx_terrain::FbmTerrainFn;
use glam::Vec3;
use std::path::{Path, PathBuf};

const SIZE: u32 = 900;
const VS: f32 = 0.25;

fn le_f32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn save(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.rgb.clone())
        .expect("rgb")
        .save(path)
        .unwrap();
}

/// Decode verts (pos + stored normal). Optionally REPLACE each vertex normal
/// with the per-triangle FACE normal (flat) to expose geometric faceting.
fn decode(verts_b: &[u8], idx_b: &[u8], face_normals: bool) -> (Vec<MeshVertex>, Vec<u32>) {
    let vcount = verts_b.len() / 32;
    let mut verts: Vec<MeshVertex> = (0..vcount)
        .map(|vi| {
            let o = vi * 32;
            MeshVertex {
                local_pos: [le_f32(verts_b, o), le_f32(verts_b, o + 4), le_f32(verts_b, o + 8)],
                normal_oct: le_u32(verts_b, o + 12),
                leaf_attr_id: 0,
                bone_indices: 0,
                bone_weights: 0,
                _pad: 0,
            }
        })
        .collect();
    let icount = idx_b.len() / 4;
    let idx: Vec<u32> = (0..icount).map(|i| le_u32(idx_b, i * 4)).collect();

    if face_normals {
        // Pack each face's geometric normal into all 3 of its verts. Last
        // write wins per shared vertex — fine, we only want to see facets.
        for tri in idx.chunks_exact(3) {
            let p0 = Vec3::from(verts[tri[0] as usize].local_pos);
            let p1 = Vec3::from(verts[tri[1] as usize].local_pos);
            let p2 = Vec3::from(verts[tri[2] as usize].local_pos);
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            let packed = pack_oct(n);
            for &v in tri {
                verts[v as usize].normal_oct = packed;
            }
        }
    }
    (verts, idx)
}

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

fn main() {
    let out = PathBuf::from("target/terrain_geom_truth");
    std::fs::create_dir_all(&out).unwrap();
    let qef_off = std::env::var("ARVX_QEF_HERMITE").as_deref() == Ok("0");
    let tag = if qef_off { "blur" } else { "qef" };

    let fbm = FbmTerrainFn::default().resolve(&NullMaterialLookup);
    let regions = TerrainRegionSnapshot::new();

    // Heaviest surface tile in column (0,*,0).
    let mut best: Option<(i32, usize)> = None;
    for ty in -2..=6 {
        if let Some(b) = bake_tile_with_skirts(TileKey::level0(0, ty, 0), VS, &fbm, &[], &regions, 4.0, Some(0.0)) {
            let vc = b.mesh.vertices.len() / 32;
            if best.map(|(_, c)| vc > c).unwrap_or(true) {
                best = Some((ty, vc));
            }
        }
    }
    let (ty, _) = best.unwrap();
    let key = TileKey::level0(0, ty, 0);
    let baked = bake_tile_with_skirts(key, VS, &fbm, &[], &regions, 4.0, Some(0.0)).unwrap();

    let (verts_v, idx) = decode(&baked.mesh.vertices, &baked.mesh.indices, false);
    let (verts_f, _) = decode(&baked.mesh.vertices, &baked.mesh.indices, true);

    // Frame on a SLICE near the tile centre so the side profile reads (full
    // 64 m tile side-on is too busy). Use the vertex bbox.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in &verts_v {
        let p = Vec3::from(v.local_pos);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    let dummy = voxelize(Shape::all()[0], Shape::all()[0].bounds(), 1.0);

    let cam_side = Camera::side_ortho_framing(SIZE, SIZE, center, radius);
    let cam_graze = Camera::three_quarter_framing(SIZE, SIZE, center, radius * 1.1);

    let side = render(&cam_side, &dummy, &verts_v, &idx,
        RenderOpts { shaded: true, wireframe: true, voxels: false, dim_shading: true }, SIZE);
    save(&side, &out.join(format!("{tag}_side_profile_wire.png")));

    let gf = render(&cam_graze, &dummy, &verts_f, &idx,
        RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false }, SIZE);
    save(&gf, &out.join(format!("{tag}_grazing_face.png")));

    let gv = render(&cam_graze, &dummy, &verts_v, &idx,
        RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false }, SIZE);
    save(&gv, &out.join(format!("{tag}_grazing_vertex.png")));

    // Numeric stepping metric: residual of each surface vertex's height from a
    // local least-squares plane fit over its neighbours in a small XZ window —
    // the "smooth-stairs" amplitude. (Skip skirts: normal_oct==0 or boundary.)
    let mut pts: Vec<Vec3> = Vec::new();
    for v in &verts_v {
        let p = Vec3::from(v.local_pos);
        if v.normal_oct == 0 || p.x < 1.5 || p.x > 62.5 || p.z < 1.5 || p.z > 62.5 {
            continue;
        }
        pts.push(p);
    }
    // Global plane fit (cheap proxy): y ≈ a*x + b*z + c; report RMS + max
    // residual in voxels. A smooth gentle slope ⇒ small; a staircase ⇒ ~0.5 vox.
    let n = pts.len() as f32;
    let (sx, sz, sy) = pts.iter().fold((0.0, 0.0, 0.0), |(ax, az, ay), p| (ax + p.x, az + p.z, ay + p.y));
    let (mx, mz, my) = (sx / n, sz / n, sy / n);
    let (mut sxx, mut szz, mut sxz, mut sxy, mut szy) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
    for p in &pts {
        let (dx, dz, dy) = (p.x - mx, p.z - mz, p.y - my);
        sxx += dx * dx; szz += dz * dz; sxz += dx * dz; sxy += dx * dy; szy += dz * dy;
    }
    let det = sxx * szz - sxz * sxz;
    let (a, b) = if det.abs() > 1e-6 {
        ((sxy * szz - szy * sxz) / det, (szy * sxx - sxy * sxz) / det)
    } else {
        (0.0, 0.0)
    };
    let mut rss = 0.0f32;
    let mut rmax = 0.0f32;
    for p in &pts {
        let pred = a * (p.x - mx) + b * (p.z - mz) + my;
        let r = (p.y - pred).abs();
        rss += r * r;
        rmax = rmax.max(r);
    }
    let rms = (rss / n).sqrt();
    eprintln!(
        "[geom-truth/{tag}] tile=(0,{ty},0) {} surf verts | plane-fit residual: rms={:.4} m ({:.2} vox) max={:.4} m ({:.2} vox)",
        pts.len(), rms, rms / VS, rmax, rmax / VS,
    );
    eprintln!("[geom-truth/{tag}] PNGs in {}", out.canonicalize().unwrap_or(out).display());
}
