//! Wide-window plane-fit FIX radius sweep — measurement only (no PNGs).
//! Finds the plane-fit neighbourhood radius that cleanly removes the
//! smooth-stairs ripple on gentle slopes without rounding FBM curvature.
//!
//!   cargo run -p arvx-terrain --example fix_radius_sweep

use arvx_core::mesh_test_bench::measure_ripple_raw;
use arvx_terrain::repro::{
    bake_terrain_window_baseline, bake_terrain_window_fix, default_terrain_fn, SlopeTerrainFn,
    TerrainWindowMesh,
};
use arvx_terrain::{TerrainFn, TileKey};
use glam::Vec3;

const LODS: &[f32] = &[0.5, 1.0, 2.0];
const WINDOW_M: f32 = 24.0;
const RADII: &[f32] = &[0.0, 3.0, 5.0, 8.0, 12.0];

fn resid_rms(mesh: &TerrainWindowMesh, tf: &dyn TerrainFn, key: TileKey) -> (f32, f32) {
    let tile_origin = key.origin_world().to_vec3();
    let vs = mesh.voxel_size;
    let h = |x: f32, z: f32| {
        let local = Vec3::new(x - tile_origin.x, 0.0, z - tile_origin.z);
        tile_origin.y - tf.sample(key, local, vs).sd
    };
    let r = measure_ripple_raw(&mesh.verts, &mesh.indices, vs, &h, Vec3::X);
    (r.residual_rms_vox, r.roughness_vox)
}

fn main() {
    let key = TileKey::level0(0, 0, 0);
    println!("radius=0 is BASELINE (no fix). residual_rms / roughness in voxels.\n");

    for (name, mx) in [("slope010", 0.10f32), ("slope020", 0.20)] {
        let sf = SlopeTerrainFn { mx, mz: 0.035, base: 24.0 };
        println!("=== {name} (dh/dx={mx}) ===");
        for &vs in LODS {
            print!("  vs={vs:<4}");
            for &rad in RADII {
                let mesh = if rad == 0.0 {
                    bake_terrain_window_baseline(&sf, key, Vec3::ZERO, WINDOW_M, vs)
                } else {
                    bake_terrain_window_fix(&sf, key, Vec3::ZERO, WINDOW_M, vs, rad)
                };
                let (rr, ro) = resid_rms(&mesh, &sf, key);
                print!("  r{rad:>4.1}:{rr:.3}/{ro:.3}");
            }
            println!();
        }
    }

    // FBM curvature-preservation check: the fix must NOT inflate
    // residual_rms on real rolling terrain (that would mean it's rounding
    // genuine curvature, not just ripple).
    let fbm = default_terrain_fn();
    let c = Vec3::new(20.0, 0.0, 20.0);
    println!("\n=== fbm (curvature-preservation check) ===");
    for &vs in LODS {
        print!("  vs={vs:<4}");
        for &rad in RADII {
            let mesh = if rad == 0.0 {
                bake_terrain_window_baseline(&fbm, key, c, WINDOW_M, vs)
            } else {
                bake_terrain_window_fix(&fbm, key, c, WINDOW_M, vs, rad)
            };
            let (rr, ro) = resid_rms(&mesh, &fbm, key);
            print!("  r{rad:>4.1}:{rr:.3}/{ro:.3}");
        }
        println!();
    }
}
