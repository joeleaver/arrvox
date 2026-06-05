//! Headless meshing test-bench driver.
//!
//! Runs every analytic shape at voxel {0.5, 0.25} through the engine's
//! blurred-occupancy surface-net extract, writes validation PNGs to
//! `target/mesh_bench/<shape>_<voxel>/*.png`, and prints the validation
//! table to stdout.
//!
//!   cargo run -p arvx-core --example mesh_bench
//!
//! PNGs per shape×voxel:
//!   shaded.png                 — Lambert-shaded by extracted normals (3/4 view)
//!   wireframe_over_shaded.png  — bright triangle wireframe over dim shading
//!   voxels_plus_wireframe.png  — faint source voxels + wireframe + dim shading
//!   side_profile.png           — side ortho, wireframe + voxel edges (terracing)

use arvx_core::mesh_test_bench::{
    evaluate, format_table, mesh_occupancy, render, voxelize, Camera, Image, Metrics, RenderOpts,
    Shape,
};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const SIZE: u32 = 512;

fn save_png(img: &Image, path: &Path) {
    let buf = image::RgbImage::from_raw(img.width, img.height, img.rgb.clone())
        .expect("rgb buffer size matches dimensions");
    buf.save(path).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let out_root = PathBuf::from("target/mesh_bench");
    std::fs::create_dir_all(&out_root).expect("create target/mesh_bench");

    let voxels = [0.5f32, 0.25];

    // One job per (shape, voxel); rendered in parallel (the software
    // rasterizer is CPU-bound and the jobs are independent).
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
            std::fs::create_dir_all(&dir).expect("create shape dir");

            let cam34 = Camera::three_quarter(SIZE, SIZE);
            let cam_side = Camera::side_ortho(SIZE, SIZE);

            // shaded.png — shaded only (clean, not dimmed).
            let shaded = render(
                &cam34, &occ, &verts, &indices,
                RenderOpts { shaded: true, wireframe: false, voxels: false, dim_shading: false },
                SIZE,
            );
            save_png(&shaded, &dir.join("shaded.png"));

            // wireframe_over_shaded.png — bright wireframe over dim shading.
            let wire = render(
                &cam34, &occ, &verts, &indices,
                RenderOpts { shaded: true, wireframe: true, voxels: false, dim_shading: true },
                SIZE,
            );
            save_png(&wire, &dir.join("wireframe_over_shaded.png"));

            // voxels_plus_wireframe.png — the money shot.
            let money = render(
                &cam34, &occ, &verts, &indices,
                RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
                SIZE,
            );
            save_png(&money, &dir.join("voxels_plus_wireframe.png"));

            // side_profile.png — side ortho, wireframe + voxel edges.
            let side = render(
                &cam_side, &occ, &verts, &indices,
                RenderOpts { shaded: true, wireframe: true, voxels: true, dim_shading: true },
                SIZE,
            );
            save_png(&side, &dir.join("side_profile.png"));

            (ji, metrics)
        })
        .collect();

    // Restore table order (par_iter collect order is nondeterministic).
    rows.sort_by_key(|(ji, _)| *ji);
    let table_rows: Vec<Metrics> = rows.into_iter().map(|(_, m)| m).collect();

    println!("\n{}", format_table(&table_rows));
    println!(
        "PNGs written under: {}",
        out_root.canonicalize().unwrap_or(out_root).display()
    );
}

fn fmt_vs(vs: f32) -> String {
    // 0.5 -> "0p5", 0.25 -> "0p25"
    format!("{vs}").replace('.', "p")
}
