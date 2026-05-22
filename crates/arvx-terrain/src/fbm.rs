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
use arvx_core::{MaterialLibraryLookup, MaterialRef};
use glam::Vec3;

/// FBM heightmap terrain with slope+height material rules.
///
/// Deterministic given the seed. Self-contained — uses an internal
/// xorshift-style hash for the noise lattice, no `rand`/`noise` crate.
///
/// **Spec form** — this struct is what's authored on disk (in the
/// Inspector / scene file). It does NOT impl [`TerrainFn`] because
/// the material fields are [`MaterialRef`]s that need a
/// [`MaterialLibraryLookup`] to become concrete slot ids. Call
/// [`FbmTerrainFn::resolve`] to produce the runtime
/// [`FbmTerrainFnResolved`] that does impl `TerrainFn`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Material reference used on near-flat ground above sea level and
    /// below the snow line. Default: `"assets/materials/grass.arvxmat"`.
    #[serde(default = "default_grass_material")]
    pub grass_material: MaterialRef,
    /// Material reference used on slopes steeper than
    /// `slope_rock_threshold_deg`. Default:
    /// `"assets/materials/rock.arvxmat"`.
    #[serde(default = "default_rock_material")]
    pub rock_material: MaterialRef,
    /// Material reference used above `snow_level_y`. Default:
    /// `"assets/materials/snow.arvxmat"`.
    #[serde(default = "default_snow_material")]
    pub snow_material: MaterialRef,
    /// Material reference used at or below `sea_level_y`. Default:
    /// `"assets/materials/sand.arvxmat"`.
    #[serde(default = "default_sand_material")]
    pub sand_material: MaterialRef,
}

fn default_grass_material() -> MaterialRef {
    MaterialRef::path("assets/materials/grass.arvxmat")
}
fn default_rock_material() -> MaterialRef {
    MaterialRef::path("assets/materials/rock.arvxmat")
}
fn default_snow_material() -> MaterialRef {
    MaterialRef::path("assets/materials/snow.arvxmat")
}
fn default_sand_material() -> MaterialRef {
    MaterialRef::path("assets/materials/sand.arvxmat")
}

impl Default for FbmTerrainFn {
    fn default() -> Self {
        // Defaults paired with `Terrain::default()` (16 × 16 × 4 grid,
        // 64 m tiles, 0.25 m voxels). With base_height_m = 16 and
        // amplitude = 24, the surface oscillates in [-8, 40] m —
        // sea_level at 0 covers troughs, snow_level at 32 covers
        // upper peaks, rock kicks in on steep flanks.
        //
        // Materials use PATH refs by default so a fresh project's
        // starter palette resolves cleanly regardless of palette
        // sort order. The starter pack ships grass/rock/snow/sand
        // for exactly this reason — without those files the refs
        // fall back to slot 0 (default opaque) with a console
        // warning, which is much better than silently colliding
        // with whatever happens to land at slots 1..4.
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
            grass_material: default_grass_material(),
            rock_material: default_rock_material(),
            snow_material: default_snow_material(),
            sand_material: default_sand_material(),
        }
    }
}

impl FbmTerrainFn {
    /// Nyquist-clamped octave count for a given voxel size.
    ///
    /// Octave `o` produces detail at wavelength `scale_m / 2^o`. Any
    /// octave whose wavelength is shorter than `2 * voxel_size_m`
    /// aliases when point-sampled at that voxel size — the
    /// classic Nyquist-Shannon cutoff. Solving for the highest safe
    /// octave: `o ≤ log2(scale_m / (2 * voxel_size_m))`.
    ///
    /// At the V1 default tier (vs = 0.25 m, scale_m = 120 m) the
    /// cutoff is `log2(240) ≈ 7.9`, so all of the default 5 octaves
    /// pass through unchanged. At a coarse LOD level (vs = 1 m for
    /// level 2 on the default tier) the cutoff drops to `log2(60)
    /// ≈ 5.9` → 6 octaves; at vs = 4 m → `log2(15) ≈ 3.9` → 4.
    ///
    /// `voxel_size_m <= 0` or `scale_m <= 0` short-circuit to
    /// `self.octaves` (fall back to V1 behavior — no filtering).
    pub fn octaves_for_voxel(&self, voxel_size_m: f32) -> u8 {
        if voxel_size_m <= 0.0 || self.scale_m <= 0.0 {
            return self.octaves;
        }
        let max_safe = (self.scale_m / (2.0 * voxel_size_m)).log2().floor() as i32 + 1;
        let clamped = max_safe.max(1).min(self.octaves as i32) as u8;
        clamped
    }

    /// Evaluate the height field at a world-space (x, z). Returns the
    /// terrain surface Y in metres.
    ///
    /// Uses the full octave count from `self.octaves` — for
    /// LOD-aware sampling that drops aliasing octaves, use
    /// [`Self::height_at_with_octaves`].
    pub fn height_at(&self, x_m: f32, z_m: f32) -> f32 {
        self.height_at_with_octaves(x_m, z_m, self.octaves)
    }

    /// Same as [`Self::height_at`] but with an explicit octave count.
    /// Used by the LOD-pyramid bake to drop octaves that would alias
    /// at the tile's voxel size (see [`Self::octaves_for_voxel`]).
    pub fn height_at_with_octaves(&self, x_m: f32, z_m: f32, octaves: u8) -> f32 {
        let n = fbm_2d(
            x_m / self.scale_m,
            z_m / self.scale_m,
            octaves,
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
    ///
    /// Uses the full octave count from `self.octaves`. For LOD-aware
    /// slope use [`Self::slope_degrees_at_with_octaves`].
    pub fn slope_degrees_at(&self, x_m: f32, z_m: f32) -> f32 {
        self.slope_degrees_at_with_octaves(x_m, z_m, self.octaves)
    }

    /// Same as [`Self::slope_degrees_at`] but with an explicit octave
    /// count. Threaded through by `sample` to keep slope-based
    /// material assignment consistent with the LOD-filtered height
    /// at the same voxel size.
    pub fn slope_degrees_at_with_octaves(
        &self,
        x_m: f32,
        z_m: f32,
        octaves: u8,
    ) -> f32 {
        let half = self.slope_probe_m.max(1e-3) * 0.5;
        let dh_dx = (self.height_at_with_octaves(x_m + half, z_m, octaves)
            - self.height_at_with_octaves(x_m - half, z_m, octaves))
            / (2.0 * half);
        let dh_dz = (self.height_at_with_octaves(x_m, z_m + half, octaves)
            - self.height_at_with_octaves(x_m, z_m - half, octaves))
            / (2.0 * half);
        let grad_mag = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();
        grad_mag.atan().to_degrees()
    }

    /// Resolve [`MaterialRef`]s against `lookup`, producing a runtime
    /// [`FbmTerrainFnResolved`] that impls [`TerrainFn`]. Resolved
    /// once per bake-job submission (cheap — 4 hash-map lookups), not
    /// once per voxel.
    pub fn resolve(&self, lookup: &dyn MaterialLibraryLookup) -> FbmTerrainFnResolved {
        FbmTerrainFnResolved {
            seed: self.seed,
            octaves: self.octaves,
            scale_m: self.scale_m,
            amplitude_m: self.amplitude_m,
            base_height_m: self.base_height_m,
            sea_level_y: self.sea_level_y,
            snow_level_y: self.snow_level_y,
            slope_rock_threshold_deg: self.slope_rock_threshold_deg,
            slope_probe_m: self.slope_probe_m,
            grass_material: self.grass_material.resolve(lookup),
            rock_material: self.rock_material.resolve(lookup),
            snow_material: self.snow_material.resolve(lookup),
            sand_material: self.sand_material.resolve(lookup),
        }
    }
}

/// Resolved runtime form of [`FbmTerrainFn`]. Material refs have been
/// collapsed to concrete `u16` slot ids — this struct is what the bake
/// worker calls `sample` on, once per voxel position. Built by
/// [`FbmTerrainFn::resolve`] at bake-job submission time so the hot
/// path stays a single field load per material lookup.
///
/// Never serialized: the on-disk authoritative form is `FbmTerrainFn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbmTerrainFnResolved {
    pub seed: u32,
    pub octaves: u8,
    pub scale_m: f32,
    pub amplitude_m: f32,
    pub base_height_m: f32,
    pub sea_level_y: f32,
    pub snow_level_y: f32,
    pub slope_rock_threshold_deg: f32,
    pub slope_probe_m: f32,
    pub grass_material: u16,
    pub rock_material: u16,
    pub snow_material: u16,
    pub sand_material: u16,
}

impl FbmTerrainFnResolved {
    /// Nyquist-clamped octave count for a given voxel size. See
    /// [`FbmTerrainFn::octaves_for_voxel`] for the derivation; this
    /// mirror exists so the resolved form is self-contained at the
    /// bake hot path.
    pub fn octaves_for_voxel(&self, voxel_size_m: f32) -> u8 {
        if voxel_size_m <= 0.0 || self.scale_m <= 0.0 {
            return self.octaves;
        }
        let max_safe = (self.scale_m / (2.0 * voxel_size_m)).log2().floor() as i32 + 1;
        max_safe.max(1).min(self.octaves as i32) as u8
    }

    pub fn height_at_with_octaves(&self, x_m: f32, z_m: f32, octaves: u8) -> f32 {
        let n = fbm_2d(
            x_m / self.scale_m,
            z_m / self.scale_m,
            octaves,
            self.seed,
        );
        self.base_height_m + n * self.amplitude_m
    }

    pub fn height_at(&self, x_m: f32, z_m: f32) -> f32 {
        self.height_at_with_octaves(x_m, z_m, self.octaves)
    }

    pub fn slope_degrees_at_with_octaves(
        &self,
        x_m: f32,
        z_m: f32,
        octaves: u8,
    ) -> f32 {
        let half = self.slope_probe_m.max(1e-3) * 0.5;
        let dh_dx = (self.height_at_with_octaves(x_m + half, z_m, octaves)
            - self.height_at_with_octaves(x_m - half, z_m, octaves))
            / (2.0 * half);
        let dh_dz = (self.height_at_with_octaves(x_m, z_m + half, octaves)
            - self.height_at_with_octaves(x_m, z_m - half, octaves))
            / (2.0 * half);
        let grad_mag = (dh_dx * dh_dx + dh_dz * dh_dz).sqrt();
        grad_mag.atan().to_degrees()
    }

    pub fn slope_degrees_at(&self, x_m: f32, z_m: f32) -> f32 {
        self.slope_degrees_at_with_octaves(x_m, z_m, self.octaves)
    }

    /// Pick the material id at a world-space `(x, y, z)` using the
    /// configured slope/height rules. Pure data — no SDF involvement.
    pub fn material_at(&self, x_m: f32, y_m: f32, z_m: f32) -> u16 {
        self.material_at_with_octaves(x_m, y_m, z_m, self.octaves)
    }

    /// LOD-aware material lookup. The slope test uses
    /// `slope_degrees_at_with_octaves(octaves)` so material assignment
    /// stays consistent with the height field a coarse-LOD bake
    /// actually emits.
    pub fn material_at_with_octaves(
        &self,
        x_m: f32,
        y_m: f32,
        z_m: f32,
        octaves: u8,
    ) -> u16 {
        if y_m < self.sea_level_y {
            return self.sand_material;
        }
        if y_m > self.snow_level_y {
            return self.snow_material;
        }
        if self.slope_degrees_at_with_octaves(x_m, z_m, octaves)
            > self.slope_rock_threshold_deg
        {
            return self.rock_material;
        }
        self.grass_material
    }
}

impl TerrainFn for FbmTerrainFnResolved {
    fn sample(&self, tile: TileKey, local: Vec3, voxel_size_m: f32) -> TerrainSample {
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

        // V2 LOD pyramid: drop aliasing octaves at coarse voxel sizes.
        // Level 0 (fine) gets all octaves; level N gets fewer per the
        // Nyquist clamp. Kills the "frequency pop" at LOD transitions.
        let octaves = self.octaves_for_voxel(voxel_size_m);

        let surface_y = self.height_at_with_octaves(wx, wz, octaves);
        // SDF: positive above the surface (empty), negative below (solid).
        let sd = wy - surface_y;

        // Material assignment uses the SURFACE Y, not the sample's Y.
        // A column of voxels straddling the surface should all carry
        // the same material — the SDF picks which side of the surface
        // the voxel falls on, not what it's made of.
        //
        // Same octave count for the slope probe so the material rule
        // matches the height field this bake will actually emit.
        let mat = self.material_at_with_octaves(wx, surface_y, wz, octaves);

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
///
/// Normalises by the GEOMETRIC-SERIES LIMIT `2.0` rather than the
/// per-call partial sum. This is load-bearing for the V2 LOD-pyramid
/// pre-filter: dropping high-frequency octaves (to avoid Nyquist
/// aliasing at coarse voxel sizes) leaves the same low-frequency
/// amplitude as the fine bake. The partial-sum normalisation would
/// instead AMPLIFY the surviving low-freq octaves by the missing
/// amplitude — turning the pre-filter into a brightness shift
/// (coarse terrain ends up taller than fine).
///
/// Cost: outputs are at most ~3% smaller in absolute amplitude than
/// the pre-fix normalisation (5 octaves: partial sum = 1.9375, vs
/// limit 2.0 → 0.97× factor). Visually identical for V1 / V2 defaults.
fn fbm_2d(x: f32, z: f32, octaves: u8, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    for o in 0..octaves {
        let s = seed.wrapping_add((o as u32).wrapping_mul(0x9E37_79B9));
        sum += amplitude * value_noise_2d(x * frequency, z * frequency, s);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    // Infinite-series limit of 1 + 0.5 + 0.25 + ... = 2.0. Dividing by
    // the limit makes low-freq amplitude invariant under octave-count
    // changes (the LOD pre-filter is then a pure low-pass, not a
    // re-amplifier).
    sum / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_core::NullMaterialLookup;

    /// All tests that need to call `TerrainFn::sample` or
    /// `material_at` operate on the resolved form. We resolve the
    /// authored defaults via the null lookup (every `Path` → 0) when
    /// the test doesn't care about which specific material id comes
    /// back; tests that DO care construct the resolved struct
    /// directly with explicit slot ids.
    fn resolved_default() -> FbmTerrainFnResolved {
        FbmTerrainFn::default().resolve(&NullMaterialLookup)
    }

    #[test]
    fn default_is_finite_at_origin() {
        let f = resolved_default();
        let h = f.height_at(0.0, 0.0);
        assert!(h.is_finite(), "height at origin = {h}");
    }

    #[test]
    fn deterministic_across_calls() {
        let f = resolved_default();
        let s1 = f.sample(TileKey::level0(0, 0, 0), Vec3::new(1.0, 2.0, 3.0), 0.08);
        let s2 = f.sample(TileKey::level0(0, 0, 0), Vec3::new(1.0, 2.0, 3.0), 0.08);
        assert_eq!(s1, s2);
    }

    #[test]
    fn different_seeds_produce_different_heights() {
        let mut a = resolved_default();
        let mut b = resolved_default();
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
        let f = resolved_default();
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
        let f = resolved_default();
        let k = TileKey::level0(1000, 0, 1000);
        let s = f.sample(k, Vec3::new(32.0, 100.0, 32.0), 0.08);
        assert!(s.sd.is_finite());
    }

    // ── slope+height material rules ────────────────────────────────

    /// Flat heightmap (zero amplitude) at y = 5 — every slope probe
    /// returns 0°. Lets us isolate the height-band rules from the
    /// slope rule. Explicit slot ids so material assertions can
    /// distinguish grass/rock/snow/sand.
    fn flat_fbm() -> FbmTerrainFnResolved {
        FbmTerrainFnResolved {
            seed: 42,
            octaves: 5,
            scale_m: 120.0,
            amplitude_m: 0.0,
            base_height_m: 5.0,
            sea_level_y: 0.0,
            snow_level_y: 100.0,
            slope_rock_threshold_deg: 35.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 3,
            snow_material: 4,
            sand_material: 2,
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
        let mut f = flat_fbm();
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
        let f = FbmTerrainFnResolved {
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

    // ── V2 LOD pyramid: Nyquist octave clamp ──────────────────────────

    /// `octaves_for_voxel` returns at most the octave count whose
    /// shortest-wavelength octave is still ≥ 2× the voxel size.
    #[test]
    fn octaves_for_voxel_clamps_at_nyquist() {
        let mut f = resolved_default();
        f.scale_m = 120.0;
        f.octaves = 12;
        // vs = 1.0 m → log2(60) ≈ 5.9 → 6 octaves.
        assert_eq!(f.octaves_for_voxel(1.0), 6);
        // vs = 0.25 m → log2(240) ≈ 7.9 → 8 octaves.
        assert_eq!(f.octaves_for_voxel(0.25), 8);
        // vs = 4.0 m → log2(15) ≈ 3.9 → 4 octaves.
        assert_eq!(f.octaves_for_voxel(4.0), 4);
        // Coarse beyond the highest octave saturates at 1 (always at
        // least one octave so the noise isn't fully silenced).
        assert_eq!(f.octaves_for_voxel(1024.0), 1);
    }

    /// The clamp never RAISES the octave count above `self.octaves` —
    /// it's a low-pass, not a re-amplification.
    #[test]
    fn octaves_for_voxel_never_exceeds_configured_octaves() {
        let mut f = resolved_default();
        f.scale_m = 120.0;
        f.octaves = 3;
        // Even at sub-voxel resolution where Nyquist would allow 10+
        // octaves, we cap at `self.octaves`.
        assert_eq!(f.octaves_for_voxel(0.01), 3);
    }

    /// Coarse sampling should produce LESS HIGH-FREQUENCY CONTENT than
    /// fine sampling. We measure this by the mean-squared first
    /// difference between adjacent samples at a tight (sub-voxel-fine)
    /// step — the wiggle the highest octaves contribute.
    ///
    /// Overall variance is a bad metric here because it's dominated by
    /// the low-frequency octaves (amplitudes 1, 1/2, 1/4 vastly bigger
    /// than 1/64, 1/128). The high-pass content is what changes when
    /// we drop high octaves.
    #[test]
    fn coarse_voxel_height_drops_high_frequency_content() {
        let f = FbmTerrainFnResolved {
            seed: 42,
            octaves: 8,
            scale_m: 64.0,
            amplitude_m: 16.0,
            base_height_m: 0.0,
            sea_level_y: -1000.0,
            snow_level_y: 1000.0,
            slope_rock_threshold_deg: 90.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 2,
            snow_material: 3,
            sand_material: 4,
        };

        let octs_fine = f.octaves_for_voxel(0.25);
        let octs_coarse = f.octaves_for_voxel(4.0);
        assert!(octs_coarse < octs_fine, "coarse must drop at least one octave");

        // Tight step (1/8 m) is well below the Nyquist limit at vs=0.25
        // (= 0.5 m), so we capture every octave the fine bake would.
        let n = 64;
        let step = 0.125_f32;

        let mut mean_sq_first_diff = |octaves: u8| -> f64 {
            let mut energy = 0.0;
            let mut count = 0;
            // Walk a 64-cell strip in x. The squared difference between
            // adjacent samples is the discrete 1st-derivative energy
            // — proportional to high-freq content.
            for j in 0..n {
                let mut prev = f.height_at_with_octaves(0.0, j as f32 * step, octaves);
                for i in 1..n {
                    let h = f.height_at_with_octaves(i as f32 * step, j as f32 * step, octaves);
                    let d = (h - prev) as f64;
                    energy += d * d;
                    prev = h;
                    count += 1;
                }
            }
            energy / count as f64
        };

        let hf_fine = mean_sq_first_diff(octs_fine);
        let hf_coarse = mean_sq_first_diff(octs_coarse);
        // Coarse should have LESS first-difference energy because the
        // wiggle from the dropped octaves is gone.
        assert!(
            hf_coarse < hf_fine * 0.9,
            "coarse (octaves={octs_coarse}, hf={hf_coarse:.6}) should drop high-freq energy vs fine (octaves={octs_fine}, hf={hf_fine:.6})"
        );
    }

    /// `sample` at a coarse voxel size produces the same height as
    /// calling `height_at_with_octaves` with the clamped octave count.
    #[test]
    fn sample_uses_octaves_for_voxel() {
        let f = resolved_default();
        let k = TileKey::level0(0, 0, 0);
        let local = Vec3::new(13.7, 0.0, 42.1);
        let world_origin = k.origin_world().to_vec3();
        let wx = world_origin.x + local.x;
        let wz = world_origin.z + local.z;

        let vs_coarse = 4.0;
        let octs = f.octaves_for_voxel(vs_coarse);
        let s = f.sample(k, local, vs_coarse);
        let expected_surface_y = f.height_at_with_octaves(wx, wz, octs);
        // sample's `sd` is `wy - surface_y`; we sampled at wy = local.y
        // (= 0), so surface_y = -sd.
        let surface_y_from_sample = local.y + world_origin.y - s.sd;
        assert!(
            (surface_y_from_sample - expected_surface_y).abs() < 1e-4,
            "sample(vs={vs_coarse}) must use octaves_for_voxel={octs}"
        );
    }

    /// `FbmTerrainFn::resolve` swaps every `MaterialRef::Path` for the
    /// looked-up slot id; missing paths fall back to 0.
    #[test]
    fn resolve_swaps_paths_for_slot_ids() {
        use std::collections::HashMap;
        use std::path::{Path, PathBuf};

        struct MapLookup(HashMap<PathBuf, u16>);
        impl arvx_core::MaterialLibraryLookup for MapLookup {
            fn resolve_path(&self, p: &Path) -> Option<u16> {
                self.0.get(p).copied()
            }
        }

        let mut m = HashMap::new();
        m.insert(PathBuf::from("assets/materials/grass.arvxmat"), 7);
        m.insert(PathBuf::from("assets/materials/rock.arvxmat"), 9);
        // snow + sand intentionally missing → resolve to 0.
        let lookup = MapLookup(m);

        let resolved = FbmTerrainFn::default().resolve(&lookup);
        assert_eq!(resolved.grass_material, 7);
        assert_eq!(resolved.rock_material, 9);
        assert_eq!(resolved.snow_material, 0);
        assert_eq!(resolved.sand_material, 0);
    }

    /// `MaterialRef::Slot` variants resolve unchanged (back-compat
    /// with bare-u16 scenes).
    #[test]
    fn resolve_slot_variants_pass_through() {
        use arvx_core::MaterialRef;
        let f = FbmTerrainFn {
            grass_material: MaterialRef::Slot(11),
            rock_material: MaterialRef::Slot(22),
            snow_material: MaterialRef::Slot(33),
            sand_material: MaterialRef::Slot(44),
            ..FbmTerrainFn::default()
        };
        let resolved = f.resolve(&NullMaterialLookup);
        assert_eq!(resolved.grass_material, 11);
        assert_eq!(resolved.rock_material, 22);
        assert_eq!(resolved.snow_material, 33);
        assert_eq!(resolved.sand_material, 44);
    }
}
