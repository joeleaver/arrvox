//! FBM heightmap `TerrainFn` impl with slope+height material rules.
//!
//! Pure value-noise FBM, no external deps. Surface defined as
//! `y = base_height + amplitude * fbm(x, z)`. Phase 7 added slope-
//! and-height material assignment: sand below sea level, snow above
//! the snow line, rock on steep slopes, grass everywhere else.
//!
//! ## Material priority
//!
//! Per voxel the rule applied is the first that matches:
//!
//! 1. `wy < sea_level_y` → `sand_material`
//! 2. `wy > snow_level_y` → `snow_material`
//! 3. `slope > slope_rock_threshold_deg` → `rock_material`
//! 4. otherwise → `grass_material`
//!
//! Slope is the angle between the surface normal and world-up,
//! computed analytically from `∂h/∂x` and `∂h/∂z` via a small
//! central-difference probe (cheap because `height_at` is already
//! cached-free; we just call it three times per voxel).

use crate::terrain_fn::{TerrainFn, TerrainSample};
use crate::tile_key::TileKey;
use glam::Vec3;

/// FBM heightmap terrain with slope+height material rules.
///
/// Deterministic given the seed. Self-contained — uses an internal
/// xorshift-style hash for the noise lattice, no `rand`/`noise` crate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Below this Y value (in world coords), the material is `sand_material`
    /// regardless of slope. Set very low to disable.
    pub sea_level_y: f32,
    /// Above this Y value, the material is `snow_material` regardless
    /// of slope. Set very high to disable.
    pub snow_level_y: f32,
    /// Slope threshold in degrees. Above this slope (measured as the
    /// angle between the surface normal and world-up), the material
    /// switches to `rock_material`. Set to 90+ to disable.
    pub slope_rock_threshold_deg: f32,
    /// World-space epsilon (metres) used for the analytic-slope
    /// finite-difference probe. Smaller = more accurate, but at sub-
    /// voxel ε you start picking up noise wiggle as "slope". 0.5 m
    /// is a sensible default for terrain at the V1 default tier
    /// (0.25 m voxels).
    pub slope_probe_m: f32,
    /// Material id used on near-flat ground above sea level and
    /// below the snow line.
    pub grass_material: u16,
    /// Material id used on slopes steeper than `slope_rock_threshold_deg`.
    pub rock_material: u16,
    /// Material id used above `snow_level_y`.
    pub snow_material: u16,
    /// Material id used at or below `sea_level_y`.
    pub sand_material: u16,
}

impl Default for FbmTerrainFn {
    fn default() -> Self {
        // Defaults paired with `Terrain::default()` (16 × 16 × 4 grid,
        // 64 m tiles, 0.25 m voxels). With base_height_m = 16 and
        // amplitude = 24, the surface oscillates in [-8, 40] m —
        // sea_level at 0 covers troughs, snow_level at 32 covers
        // upper peaks, rock kicks in on steep flanks.
        Self {
            seed: 42,
            octaves: 5,
            scale_m: 120.0,
            amplitude_m: 24.0,
            base_height_m: 16.0,
            sea_level_y: 0.0,
            snow_level_y: 32.0,
            slope_rock_threshold_deg: 35.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 3,
            snow_material: 4,
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

    /// Slope at world-space `(x, z)` in degrees from horizontal,
    /// measured as the angle between the surface normal and world-up.
    ///
    /// Computed analytically by central-difference probes of
    /// [`Self::height_at`] at `±slope_probe_m / 2` on each axis. For
    /// the surface `y = h(x, z)` the upward normal is
    /// `(-∂h/∂x, 1, -∂h/∂z)`, so the slope angle is
    /// `atan2(sqrt(∂h/∂x² + ∂h/∂z²), 1)`.
    pub fn slope_degrees_at(&self, x_m: f32, z_m: f32) -> f32 {
        let half = self.slope_probe_m.max(1e-3) * 0.5;
        let dh_dx = (self.height_at(x_m + half, z_m) - self.height_at(x_m - half, z_m))
            / (2.0 * half);
        let dh_dz = (self.height_at(x_m, z_m + half) - self.height_at(x_m, z_m - half))
            / (2.0 * half);
        let grad_mag = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();
        grad_mag.atan().to_degrees()
    }

    /// Pick the material id at a world-space `(x, y, z)` using the
    /// configured slope/height rules. Pure data — no SDF involvement.
    pub fn material_at(&self, x_m: f32, y_m: f32, z_m: f32) -> u16 {
        if y_m < self.sea_level_y {
            return self.sand_material;
        }
        if y_m > self.snow_level_y {
            return self.snow_material;
        }
        if self.slope_degrees_at(x_m, z_m) > self.slope_rock_threshold_deg {
            return self.rock_material;
        }
        self.grass_material
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

        // Material assignment uses the SURFACE Y, not the sample's Y.
        // A column of voxels straddling the surface should all carry
        // the same material — the SDF picks which side of the surface
        // the voxel falls on, not what it's made of.
        let mat = self.material_at(wx, surface_y, wz);

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

    // ── slope+height material rules ────────────────────────────────

    fn flat_fbm() -> FbmTerrainFn {
        // Flat heightmap (zero amplitude) at y = 5 — every slope probe
        // returns 0°. Lets us isolate the height-band rules from the
        // slope rule.
        FbmTerrainFn {
            amplitude_m: 0.0,
            base_height_m: 5.0,
            sea_level_y: 0.0,
            snow_level_y: 100.0,
            slope_rock_threshold_deg: 35.0,
            ..FbmTerrainFn::default()
        }
    }

    #[test]
    fn flat_default_band_is_grass() {
        let f = flat_fbm();
        // Surface at y=5; well above sea, below snow line, zero slope.
        let m = f.material_at(0.0, 5.0, 0.0);
        assert_eq!(m, f.grass_material);
    }

    #[test]
    fn below_sea_level_is_sand() {
        let mut f = flat_fbm();
        f.base_height_m = -5.0; // surface below sea
        let m = f.material_at(0.0, -5.0, 0.0);
        assert_eq!(m, f.sand_material);
    }

    #[test]
    fn above_snow_line_is_snow() {
        let mut f = flat_fbm();
        f.snow_level_y = 10.0;
        f.base_height_m = 50.0; // surface well above snow line
        let m = f.material_at(0.0, 50.0, 0.0);
        assert_eq!(m, f.snow_material);
    }

    #[test]
    fn snow_takes_priority_over_rock_at_altitude() {
        // High-altitude steep slope: snow_level_y check fires first,
        // so the material is snow, not rock.
        let mut f = FbmTerrainFn::default();
        f.snow_level_y = 10.0;
        f.slope_rock_threshold_deg = 0.1;
        let m = f.material_at(0.0, 100.0, 0.0);
        assert_eq!(m, f.snow_material);
    }

    #[test]
    fn steep_slope_becomes_rock() {
        // Construct a steep heightmap. amplitude_m = 50 over scale_m
        // = 10 produces gradients on the order of several units per
        // metre, well past the default 35° threshold.
        let f = FbmTerrainFn {
            seed: 42,
            octaves: 5,
            scale_m: 10.0,
            amplitude_m: 50.0,
            base_height_m: 5.0,
            sea_level_y: -1000.0, // disable sea rule
            snow_level_y: 1000.0, // disable snow rule
            slope_rock_threshold_deg: 35.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 3,
            snow_material: 4,
            sand_material: 2,
        };
        // Sample at the midpoint of a noise cell where the gradient
        // is non-zero (any point off-lattice).
        let slope = f.slope_degrees_at(2.3, 7.9);
        assert!(
            slope > 35.0,
            "expected steep slope for stress-test FBM, got {slope}°"
        );
        let m = f.material_at(2.3, f.height_at(2.3, 7.9), 7.9);
        assert_eq!(m, f.rock_material);
    }

    #[test]
    fn zero_amplitude_has_zero_slope() {
        let f = flat_fbm();
        // ∂h/∂x = ∂h/∂z = 0 → slope is 0°.
        let slope = f.slope_degrees_at(13.7, -42.1);
        assert!(slope.abs() < 1e-3, "expected 0°, got {slope}°");
    }

    #[test]
    fn sample_picks_material_from_surface_not_sample_y() {
        // FlatHalf-style: surface at y = 5, no amplitude. Sample a
        // voxel high in the air — it must still pick GRASS (the
        // surface's material), not snow (which would only apply if
        // the snow rule used the sample's Y).
        let f = flat_fbm();
        let k = TileKey::level0(0, 0, 0);
        let s = f.sample(k, Vec3::new(0.0, 200.0, 0.0), 0.25);
        assert_eq!(s.primary_mat, f.grass_material);
    }

    #[test]
    fn sample_below_sea_picks_sand() {
        // Surface below sea_level → sand even though the sample is
        // above the surface.
        let mut f = flat_fbm();
        f.base_height_m = -3.0;
        f.sea_level_y = 0.0;
        let k = TileKey::level0(0, 0, 0);
        let s = f.sample(k, Vec3::new(0.0, 50.0, 0.0), 0.25);
        assert_eq!(s.primary_mat, f.sand_material);
    }
}
