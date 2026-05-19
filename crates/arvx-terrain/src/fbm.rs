//! Minimal FBM heightmap `TerrainFn` impl for Phase 1 validation.
//!
//! Pure value-noise FBM, no external deps. Surface defined as
//! `y = base_height + amplitude * fbm(x, z)`. Below sea level the
//! material switches to `sand_material`; above, `ground_material`.

use crate::terrain_fn::{TerrainFn, TerrainSample};
use crate::tile_key::TileKey;
use glam::Vec3;

/// Simple FBM heightmap terrain.
///
/// Deterministic given the seed. Self-contained — uses an internal
/// xorshift-style hash for the noise lattice, no `rand`/`noise` crate.
#[derive(Debug, Clone, Copy)]
pub struct FbmTerrainFn {
    /// Determinism seed.
    pub seed: u32,
    /// Octave count. Higher = more detail, slower.
    pub octaves: u8,
    /// Horizontal scale in metres. 1 noise period covers `scale_m * 2π`
    /// of world distance roughly.
    pub scale_m: f32,
    /// Vertical amplitude in metres. Peak-to-trough is `~2 * amplitude_m`.
    pub amplitude_m: f32,
    /// Base height around which the surface oscillates.
    pub base_height_m: f32,
    /// Below this Y value, the material switches to `sand_material`.
    pub sea_level_y: f32,
    /// Material id used above sea level.
    pub ground_material: u16,
    /// Material id used at or below sea level.
    pub sand_material: u16,
}

impl Default for FbmTerrainFn {
    fn default() -> Self {
        Self {
            seed: 42,
            octaves: 5,
            scale_m: 120.0,
            amplitude_m: 24.0,
            base_height_m: 16.0,
            sea_level_y: 8.0,
            ground_material: 1,
            sand_material: 2,
        }
    }
}

impl FbmTerrainFn {
    /// Evaluate the height field at a world-space (x, z). Returns the
    /// terrain surface Y in metres.
    pub fn height_at(&self, x_m: f32, z_m: f32) -> f32 {
        let n = fbm_2d(
            x_m / self.scale_m,
            z_m / self.scale_m,
            self.octaves,
            self.seed,
        );
        self.base_height_m + n * self.amplitude_m
    }
}

impl TerrainFn for FbmTerrainFn {
    fn sample(&self, tile: TileKey, local: Vec3, _voxel_size_m: f32) -> TerrainSample {
        // Tile-local → world-space x/z for the noise lookup.
        // `origin_world().to_vec3()` is f32-precision in absolute world
        // coords; fine for tiles near the origin. Phase 2 will switch
        // to integer-anchored noise inputs for drift safety at large
        // distances (TileKey already carries the integer key — noise
        // impls can adopt it without a TerrainFn API change).
        let world_origin = tile.origin_world().to_vec3();
        let wx = world_origin.x + local.x;
        let wy = world_origin.y + local.y;
        let wz = world_origin.z + local.z;

        let surface_y = self.height_at(wx, wz);
        // SDF: positive above the surface (empty), negative below (solid).
        let sd = wy - surface_y;

        let mat = if wy < self.sea_level_y {
            self.sand_material
        } else {
            self.ground_material
        };

        TerrainSample {
            sd,
            primary_mat: mat,
            secondary_mat: mat,
            blend: 0.0,
        }
    }
}

// ── value-noise FBM ────────────────────────────────────────────────────────

/// Hash a 2D integer lattice point to a float in `[-1, 1]`.
fn hash2(x: i32, z: i32, seed: u32) -> f32 {
    let mut n = seed
        .wrapping_add((x as u32).wrapping_mul(73_856_093))
        .wrapping_add((z as u32).wrapping_mul(19_349_663));
    n ^= n >> 13;
    n = n.wrapping_mul(0x5bd1_e995);
    n ^= n >> 15;
    // Map u32 → [-1, 1].
    (n as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Bilinear-interpolated 2D value noise with smoothstep weights.
fn value_noise_2d(x: f32, z: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - xi as f32;
    let zf = z - zi as f32;

    let v00 = hash2(xi, zi, seed);
    let v10 = hash2(xi + 1, zi, seed);
    let v01 = hash2(xi, zi + 1, seed);
    let v11 = hash2(xi + 1, zi + 1, seed);

    let smx = xf * xf * (3.0 - 2.0 * xf);
    let smz = zf * zf * (3.0 - 2.0 * zf);

    let a = v00 * (1.0 - smx) + v10 * smx;
    let b = v01 * (1.0 - smx) + v11 * smx;
    a * (1.0 - smz) + b * smz
}

/// Octave-summed FBM. Output is approximately in `[-1, 1]`.
fn fbm_2d(x: f32, z: f32, octaves: u8, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut total = 0.0;
    for o in 0..octaves {
        let s = seed.wrapping_add((o as u32).wrapping_mul(0x9E37_79B9));
        sum += amplitude * value_noise_2d(x * frequency, z * frequency, s);
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    if total > 0.0 {
        sum / total
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_finite_at_origin() {
        let f = FbmTerrainFn::default();
        let h = f.height_at(0.0, 0.0);
        assert!(h.is_finite(), "height at origin = {h}");
    }

    #[test]
    fn deterministic_across_calls() {
        let f = FbmTerrainFn::default();
        let s1 = f.sample(TileKey::level0(0, 0, 0), Vec3::new(1.0, 2.0, 3.0), 0.08);
        let s2 = f.sample(TileKey::level0(0, 0, 0), Vec3::new(1.0, 2.0, 3.0), 0.08);
        assert_eq!(s1, s2);
    }

    #[test]
    fn different_seeds_produce_different_heights() {
        let mut a = FbmTerrainFn::default();
        let mut b = FbmTerrainFn::default();
        a.seed = 1;
        b.seed = 2;
        // 100 samples on a coarse grid; almost certainly at least one differs.
        let mut differ = 0usize;
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f32 * 20.0;
                let z = j as f32 * 20.0;
                if (a.height_at(x, z) - b.height_at(x, z)).abs() > 1e-3 {
                    differ += 1;
                }
            }
        }
        assert!(differ > 50, "seed didn't perturb noise enough ({differ}/100 differed)");
    }

    /// SDF sign convention: above surface positive, below negative.
    #[test]
    fn sign_convention_above_and_below() {
        let f = FbmTerrainFn::default();
        let k = TileKey::level0(0, 0, 0);
        // Well above the base height should be positive.
        let high = f.sample(k, Vec3::new(0.0, 1000.0, 0.0), 0.08);
        assert!(high.sd > 0.0, "above surface should be positive; got {}", high.sd);
        // Well below should be negative.
        let low = f.sample(k, Vec3::new(0.0, -1000.0, 0.0), 0.08);
        assert!(low.sd < 0.0, "below surface should be negative; got {}", low.sd);
    }

    /// Far-from-origin sample is still finite and follows the sign rule.
    /// At large world coords the f32-coalesced noise input will visibly
    /// repeat (the Phase-1 limitation called out in `sample`'s docstring);
    /// the value remains well-defined.
    #[test]
    fn far_tile_sample_is_finite() {
        let f = FbmTerrainFn::default();
        let k = TileKey::level0(1000, 0, 1000);
        let s = f.sample(k, Vec3::new(32.0, 100.0, 32.0), 0.08);
        assert!(s.sd.is_finite());
    }

    #[test]
    fn material_switches_below_sea_level() {
        let f = FbmTerrainFn::default();
        let k = TileKey::level0(0, 0, 0);
        let above = f.sample(k, Vec3::new(0.0, 100.0, 0.0), 0.08);
        let below = f.sample(k, Vec3::new(0.0, -100.0, 0.0), 0.08);
        assert_eq!(above.primary_mat, f.ground_material);
        assert_eq!(below.primary_mat, f.sand_material);
    }
}
