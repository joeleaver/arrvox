//! Headless meshing test-bench driver.
//!
//! Runs every analytic shape at voxel {0.5, 0.25} AND the Goal-2
//! confound scenarios (coarse / region-truncation / irregular) through
//! the engine's blurred-occupancy surface-net extract, writes validation
//! PNGs to `target/mesh_bench/<name>/*.png`, and prints the validation
//! tables to stdout.
//!
//!   cargo run -p arvx-core --example mesh_bench
//!
//! PNGs per case:
//!   shaded.png                 — Lambert-shaded by extracted normals (3/4 view)
//!   wireframe_over_shaded.png  — bright triangle wireframe over dim shading
//!   voxels_plus_wireframe.png  — faint source voxels + wireframe + dim shading
//!   side_profile.png           — side ortho, wireframe + voxel edges (terracing)

use arvx_core::mesh_test_bench::{
    confound_scenarios, evaluate, format_table, mesh_occupancy, mesh_occupancy_blur, render,
    voxelize, voxelize_irregular, BlurParams, Camera, Image, Metrics, Occupancy, RenderOpts, Shape,
};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const SIZE: u32 = 512;

fn save_png(img: &Image, path: &Path) {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.rgb.clone())
        .expect("rgb buffer size matches dimensions");
    buf.save(path).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Render the 4 standard layer PNGs for one occupancy into `dir`.
fn render_case(dir: &Path, occ: &Occupancy, verts: &[arvx_core::mesh_extract::MeshVertex], indices: &[u32]) {
    std::fs::create_dir_all(dir).expect("create case dir");
    let cam34 = Camera::three_quarter(SIZE, SIZE);
    let cam_side = Camera::side_ortho(SIZE, SIZE);

    let shaded = render(
        &cam34, occ, verts, indices,
        RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false },
        SIZE,
    );
    save_png(&shaded, &dir.join("shaded.png"));

    let wire = render(
        &cam34, occ, verts, indices,
        RenderOpts { shaded: true, wireframe: true, voxels: false, dim_shading: true },
        SIZE,
    );
    save_png(&wire, &dir.join("wireframe_over_shaded.png"));

    let money = render(
        &cam34, occ, verts, indices,
        RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
        SIZE,
    );
    save_png(&money, &dir.join("voxels_plus_wireframe.png"));

    let side = render(
        &cam_side, occ, verts, indices,
        RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
        SIZE,
    );
    save_png(&side, &dir.join("side_profile.png"));
}

fn main() {
    let out_root = PathBuf::from("target/mesh_bench");
    std::fs::create_dir_all(&out_root).expect("create target/mesh_bench");

    // ── Analytic shapes at vs {0.5, 0.25}. ──
    let voxels = [0.5f32, 0.25];
    let mut jobs: Vec<(Shape, f32)> = Vec::new();
    for &shape in Shape::all() {
        for &vs in &voxels {
            jobs.push((shape, vs));
        }
    }
    let mut rows: Vec<(usize, Metrics)> = jobs
        .par_iter()
        .enumerate()
        .map(|(ji, &(shape, vs))| {
            let occ = voxelize(shape, shape.bounds(), vs);
            let (verts, indices) = mesh_occupancy(&occ);
            let metrics = evaluate(shape, &occ, &verts, &indices);
            let dir = out_root.join(format!("{}_{}", shape.name(), fmt_vs(vs)));
            render_case(&dir, &occ, &verts, &indices);
            (ji, metrics)
        })
        .collect();
    rows.sort_by_key(|(ji, _)| *ji);
    let shape_rows: Vec<Metrics> = rows.into_iter().map(|(_, m)| m).collect();

    // ── Goal-2 confound scenarios. ──
    let scenarios = confound_scenarios();
    let mut sc_rows: Vec<(usize, Metrics)> = scenarios
        .par_iter()
        .enumerate()
        .map(|(si, sc)| {
            let (verts, indices) = mesh_occupancy(&sc.occ);
            let mut metrics = evaluate(sc.shape, &sc.occ, &verts, &indices);
            // Label the row with the scenario name (table reads &'static str).
            metrics.shape = Box::leak(sc.name.clone().into_boxed_str());
            let dir = out_root.join(&sc.name);
            render_case(&dir, &sc.occ, &verts, &indices);
            (si, metrics)
        })
        .collect();
    sc_rows.sort_by_key(|(si, _)| *si);
    let scenario_rows: Vec<Metrics> = sc_rows.into_iter().map(|(_, m)| m).collect();

    // ── BLUR-RADIUS SWEEP (R ∈ {2,3,4}) on the terrain-matching shapes. ──
    // steep_slope + mound, both CLEAN and IRREGULAR (sculpt-brush-like),
    // at vs=0.25. For each we emit side_profile.png + voxels_plus_
    // wireframe.png under `rsweep_<case>_R<r>/` and a metrics row.
    let vs_sweep = 0.25f32;
    let sweep_cases: Vec<(String, Shape, bool)> = vec![
        ("steep_slope".into(), Shape::SteepSlope, false),
        ("mound".into(), Shape::Mound, false),
        ("irr_steep_slope".into(), Shape::SteepSlope, true),
        ("irr_mound".into(), Shape::Mound, true),
    ];
    let mut sweep_jobs: Vec<(usize, String, Shape, bool, i32)> = Vec::new();
    for (label, shape, irr) in &sweep_cases {
        for r in [2i32, 3, 4] {
            let idx = sweep_jobs.len();
            sweep_jobs.push((idx, label.clone(), *shape, *irr, r));
        }
    }
    let mut sweep_rows: Vec<(usize, Metrics)> = sweep_jobs
        .par_iter()
        .map(|(idx, label, shape, irr, r)| {
            let occ = if *irr {
                voxelize_irregular(*shape, shape.bounds(), vs_sweep, 1.5)
            } else {
                voxelize(*shape, shape.bounds(), vs_sweep)
            };
            let bp = BlurParams::for_radius(*r);
            let (verts, indices) = mesh_occupancy_blur(&occ, bp);
            let mut metrics = evaluate(*shape, &occ, &verts, &indices);
            metrics.shape = Box::leak(format!("{label}_R{r}").into_boxed_str());
            let dir = out_root.join(format!("rsweep_{label}_R{r}"));
            render_case(&dir, &occ, &verts, &indices);
            (*idx, metrics)
        })
        .collect();
    sweep_rows.sort_by_key(|(i, _)| *i);
    let sweep_metrics: Vec<Metrics> = sweep_rows.into_iter().map(|(_, m)| m).collect();

    println!("\n=== ANALYTIC SHAPES (clean) ===\n{}", format_table(&shape_rows));
    println!("\n=== GOAL-2 CONFOUND SCENARIOS ===\n{}", format_table(&scenario_rows));
    println!(
        "\n=== BLUR-RADIUS SWEEP — steep_slope + mound, clean & irregular, vs=0.25 ===\n{}",
        format_table(&sweep_metrics)
    );
    println!(
        "PNGs written under: {}",
        out_root.canonicalize().unwrap_or(out_root).display()
    );
}

fn fmt_vs(vs: f32) -> String {
    format!("{vs}").replace('.', "p")
}
