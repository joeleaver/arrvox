//! Smooth-stairs reproduction — bakes the REAL terrain through the EXACT
//! bake path at the EXACT on-screen LOD voxel sizes and renders +
//! measures the low-frequency ripple.
//!
//!   cargo run -p arvx-terrain --example smooth_stairs_repro
//!
//! Writes PNGs under `target/smooth_stairs/<case>/` and prints a ripple
//! table. The COARSER LODs (0.5–2.0 m) are the on-screen ones where the
//! "smooth stairs" are worst (treads wide relative to the R=2 blur).
//!
//! Cases:
//!   slope010 / slope020 — GENTLE planar slopes (dh/dx 0.10 / 0.20), the
//!                         canonical wide-tread probe, through the bake path.
//!   fbm                 — FbmTerrainFn::default(), realistic rolling terrain.
//! Each at vs ∈ {0.25, 0.5, 1.0, 2.0}. Variants: baseline (R=2), the
//! WIDER-WINDOW vertex-placement FIX, and wider-blur R=3/R=4 for contrast.
//! Plus a bake-vs-region comparison on the identical slope occupancy.

use arvx_core::mesh_test_bench::{
    measure_ripple_raw, render, BlurParams, Camera, Image, Occupancy, RenderOpts,
};
use arvx_terrain::repro::{
    bake_terrain_window, bake_terrain_window_baseline, bake_terrain_window_blur,
    default_terrain_fn, SlopeTerrainFn, TerrainWindowMesh,
};
use arvx_terrain::{TerrainFn, TileKey};
use glam::Vec3;
use std::path::{Path, PathBuf};

const SIZE: u32 = 768;
/// On-screen LOD voxel sizes (m). 0.25 is the fine reference; 0.5/1.0/2.0
/// are the coarse on-screen LODs where the ripple is worst.
const LODS: &[f32] = &[0.25, 0.5, 1.0, 2.0];
/// Window footprint in metres (legible at 768²; faithful sub-tile).
const WINDOW_M: f32 = 24.0;

fn save_png(img: &Image, path: &Path) {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.rgb.clone())
        .expect("rgb buffer size matches dimensions");
    buf.save(path).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Render the standard PNGs for a terrain window into `dir`. The side
/// profile uses a Y-magnified ortho so low-amplitude stairs are visible:
/// the surface band's mid-height is `surface_mid_y`, and `band_half_y`
/// brackets it tightly.
fn render_case(dir: &Path, mesh: &TerrainWindowMesh, surface_mid_y: f32, band_half_y: f32) {
    std::fs::create_dir_all(dir).expect("create case dir");
    let occ: Occupancy = mesh.as_occupancy();
    let center = (mesh.aabb.min + mesh.aabb.max) * 0.5;
    let half_x = (mesh.aabb.max.x - mesh.aabb.min.x) * 0.5;

    // 3/4 perspective framed on the surface band (center Y at the surface).
    let surf_center = Vec3::new(center.x, surface_mid_y, center.z);
    let cam34 = Camera::three_quarter_framing(SIZE, SIZE, surf_center, half_x * 0.85);
    let shaded = render(
        &cam34, &occ, &mesh.verts, &mesh.indices,
        RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false },
        SIZE,
    );
    save_png(&shaded, &dir.join("shaded.png"));

    let money = render(
        &cam34, &occ, &mesh.verts, &mesh.indices,
        RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
        SIZE,
    );
    save_png(&money, &dir.join("voxels_plus_wireframe.png"));

    // Side profile: full X span, Y magnified onto the surface band so a
    // 0.1-voxel ripple on a gentle slope reads clearly.
    let cam_side = Camera::side_ortho_xy(
        SIZE, SIZE,
        Vec3::new(center.x, surface_mid_y, center.z),
        half_x * 1.05,
        band_half_y,
    );
    let side = render(
        &cam_side, &occ, &mesh.verts, &mesh.indices,
        RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
        SIZE,
    );
    save_png(&side, &dir.join("side_profile.png"));
}

/// Measure ripple of a terrain window against the real surface height.
fn ripple(
    mesh: &TerrainWindowMesh,
    terrain_fn: &dyn TerrainFn,
    key: TileKey,
    label: &str,
) {
    let tile_origin = key.origin_world().to_vec3();
    let vs = mesh.voxel_size;
    let h = |x: f32, z: f32| -> f32 {
        let local = Vec3::new(x - tile_origin.x, 0.0, z - tile_origin.z);
        tile_origin.y - terrain_fn.sample(key, local, vs).sd
    };
    let r = measure_ripple_raw(&mesh.verts, &mesh.indices, vs, &h, Vec3::X);
    println!(
        "    {label:<22} vs={:<4} verts={:<6} tris={:<6} | λ={:>5.2}v amp={:>6.3}v resid_rms={:>5.3}v rough={:>5.3}v (n={})",
        vs,
        mesh.verts.len(),
        mesh.indices.len() / 3,
        r.wavelength_vox,
        r.amplitude_vox,
        r.residual_rms_vox,
        r.roughness_vox,
        r.n_samples,
    );
}

/// Surface mid-height + a tight Y half-extent for the side-profile band.
/// `slope_total_rise` is the surface's vertical span across the window.
fn band(mesh: &TerrainWindowMesh, mid_y: f32, slope_total_rise: f32) -> (f32, f32) {
    // Bracket the surface band: half the slope rise + a few voxels margin.
    let half = (slope_total_rise * 0.5 + 4.0 * mesh.voxel_size).max(2.0 * mesh.voxel_size);
    (mid_y, half)
}

fn main() {
    let out = PathBuf::from("target/smooth_stairs");
    std::fs::create_dir_all(&out).expect("create target/smooth_stairs");
    let key = TileKey::level0(0, 0, 0);

    // ════════════════════════════════════════════════════════════════
    // GENTLE SLOPES — the canonical wide-tread probe.
    // ════════════════════════════════════════════════════════════════
    let slopes: &[(&str, f32)] = &[("slope010", 0.10), ("slope020", 0.20)];
    let centre_xz = Vec3::new(0.0, 0.0, 0.0);
    for &(name, mx) in slopes {
        let sf = SlopeTerrainFn { mx, mz: 0.035, base: 24.0 };
        println!("\n=== {name}: GENTLE SLOPE dh/dx={mx} (real bake path) ===");
        for &vs in LODS {
            // BASELINE: production path with the fix FORCED OFF (raw ripple).
            let mesh = bake_terrain_window_baseline(&sf, key, centre_xz, WINDOW_M, vs);
            let mid_y = surface_mid(&sf, key, &mesh);
            let (cy, hy) = band(&mesh, mid_y, WINDOW_M * mx);
            render_case(&out.join(format!("{name}_baseline_vs{}", fmt(vs))), &mesh, cy, hy);
            ripple(&mesh, &sf, key, "BAKE baseline (fix OFF)");

            // PRODUCTION: the shipping bake path (fix DEFAULT-ON, r=5,
            // seam-ring pinned).
            let fmesh = bake_terrain_window(&sf, key, centre_xz, WINDOW_M, vs);
            render_case(&out.join(format!("{name}_FIXwide_vs{}", fmt(vs))), &fmesh, cy, hy);
            ripple(&fmesh, &sf, key, "BAKE PRODUCTION (fix ON)");

            // CONTRAST: wider blur R=3 alone (fix isolated off).
            let b3 = bake_terrain_window_blur(
                &sf, key, centre_xz, WINDOW_M, vs, BlurParams::for_radius(3),
            );
            render_case(&out.join(format!("{name}_blurR3_vs{}", fmt(vs))), &b3, cy, hy);
            ripple(&b3, &sf, key, "BAKE blur R3 (fix OFF)");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // FBM — realistic rolling terrain.
    // ════════════════════════════════════════════════════════════════
    let fbm = default_terrain_fn();
    let fbm_centre = Vec3::new(20.0, 0.0, 20.0);
    println!("\n=== fbm: FbmTerrainFn::default() (real bake path) ===");
    for &vs in LODS {
        let mesh = bake_terrain_window_baseline(&fbm, key, fbm_centre, WINDOW_M, vs);
        let mid_y = surface_mid(&fbm, key, &mesh);
        let (cy, hy) = band(&mesh, mid_y, 6.0);
        render_case(&out.join(format!("fbm_baseline_vs{}", fmt(vs))), &mesh, cy, hy);
        ripple(&mesh, &fbm, key, "BAKE baseline (fix OFF)");

        let fmesh = bake_terrain_window(&fbm, key, fbm_centre, WINDOW_M, vs);
        render_case(&out.join(format!("fbm_FIXwide_vs{}", fmt(vs))), &fmesh, cy, hy);
        ripple(&fmesh, &fbm, key, "BAKE PRODUCTION (fix ON)");

        let b3 = bake_terrain_window_blur(&fbm, key, fbm_centre, WINDOW_M, vs, BlurParams::for_radius(3));
        render_case(&out.join(format!("fbm_blurR3_vs{}", fmt(vs))), &b3, cy, hy);
        ripple(&b3, &fbm, key, "BAKE blur R3 (fix OFF)");
    }

    println!("\nPNGs under: {}", out.canonicalize().unwrap_or(out).display());
}

/// Surface mid-height (world Y) at the window centre.
fn surface_mid(tf: &dyn TerrainFn, key: TileKey, mesh: &TerrainWindowMesh) -> f32 {
    let center = (mesh.aabb.min + mesh.aabb.max) * 0.5;
    let tile_origin = key.origin_world().to_vec3();
    let local = Vec3::new(center.x - tile_origin.x, 0.0, center.z - tile_origin.z);
    tile_origin.y - tf.sample(key, local, mesh.voxel_size).sd
}

fn fmt(vs: f32) -> String {
    format!("{vs}").replace('.', "p")
}
