//! 2D value-noise FBM helpers shared by [`crate::fbm`] (the procedural
//! height field source) and [`crate::stamp`] (per-stamp shape
//! perturbation).
//!
//! Self-contained — same xorshift-style hash and smoothstep-bilerp
//! lattice as the FBM terrain source. Pulled into its own module so
//! stamps can perturb their footprint with the same noise primitives
//! without duplicating the implementation.

/// Hash a 2D integer lattice point to a float in `[-1, 1]`.
pub fn hash2(x: i32, z: i32, seed: u32) -> f32 {
    let mut n = seed
        .wrapping_add((x as u32).wrapping_mul(73_856_093))
        .wrapping_add((z as u32).wrapping_mul(19_349_663));
    n ^= n >> 13;
    n = n.wrapping_mul(0x5bd1_e995);
    n ^= n >> 15;
    (n as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Bilinear-interpolated 2D value noise with smoothstep weights.
/// Output in `[-1, 1]`.
pub fn value_noise_2d(x: f32, z: f32, seed: u32) -> f32 {
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

/// Octave-summed FBM in `[-1, 1]`. Normalises by the geometric-series
/// limit `2.0` (see [`crate::fbm`]'s docstring for why). Stamps
/// call this with `octaves = 1..4` and a small `scale` to perturb
/// their radial / local coordinates.
pub fn fbm_2d(x: f32, z: f32, octaves: u8, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    for o in 0..octaves {
        let s = seed.wrapping_add((o as u32).wrapping_mul(0x9E37_79B9));
        sum += amplitude * value_noise_2d(x * frequency, z * frequency, s);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum / 2.0
}

/// [`value_noise_2d`] with its analytic partial derivatives.
///
/// Returns `(value, ∂/∂x, ∂/∂z)`. The lattice values `v00..v11` are
/// constant w.r.t. the fractional position, so the only `x`/`z`
/// dependence flows through the smoothstep weights `smx`/`smz`. With
/// `smx = xf²(3 − 2xf)` and `xf = x − ⌊x⌋` (so `dxf/dx = 1`), the
/// smoothstep derivative is `dsmx = 6·xf(1 − xf)`; the chain rule on
/// the bilinear blend gives the gradient below. Differentiating the
/// closed form (not finite differences) keeps this exact and cheap —
/// the lattice hashes are shared with the value.
pub fn value_noise_2d_grad(x: f32, z: f32, seed: u32) -> (f32, f32, f32) {
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
    // d(smx)/dx = 6·xf·(1 − xf); likewise for z.
    let dsmx = 6.0 * xf * (1.0 - xf);
    let dsmz = 6.0 * zf * (1.0 - zf);

    let a = v00 * (1.0 - smx) + v10 * smx;
    let b = v01 * (1.0 - smx) + v11 * smx;
    let value = a * (1.0 - smz) + b * smz;

    // a, b depend only on x; the final blend depends on both. Hence
    // ∂/∂x flows through a,b via dsmx, and ∂/∂z is (b − a)·dsmz.
    let dvdx = dsmx * ((v10 - v00) * (1.0 - smz) + (v11 - v01) * smz);
    let dvdz = (b - a) * dsmz;
    (value, dvdx, dvdz)
}

/// [`fbm_2d`] with its analytic partial derivatives, `(value, ∂/∂x,
/// ∂/∂z)`. Each octave samples the lattice at `(x·freq, z·freq)`, so
/// the chain rule scales that octave's derivative by `freq`; the sum
/// and the `/ 2.0` normaliser carry through linearly. Used by the
/// terrain bake to hand the voxelizer an exact surface gradient
/// instead of a 6-tap finite difference.
pub fn fbm_2d_grad(x: f32, z: f32, octaves: u8, seed: u32) -> (f32, f32, f32) {
    let mut sum = 0.0;
    let mut dsum_dx = 0.0;
    let mut dsum_dz = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    for o in 0..octaves {
        let s = seed.wrapping_add((o as u32).wrapping_mul(0x9E37_79B9));
        let (v, dvdx, dvdz) = value_noise_2d_grad(x * frequency, z * frequency, s);
        sum += amplitude * v;
        dsum_dx += amplitude * frequency * dvdx;
        dsum_dz += amplitude * frequency * dvdz;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    (sum / 2.0, dsum_dx / 2.0, dsum_dz / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_in_range() {
        for x in -50..50 {
            for z in -50..50 {
                let v = hash2(x, z, 42);
                assert!((-1.0..=1.0).contains(&v), "hash {x},{z} = {v}");
            }
        }
    }

    #[test]
    fn fbm_is_deterministic() {
        let a = fbm_2d(1.7, -3.4, 5, 99);
        let b = fbm_2d(1.7, -3.4, 5, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_responds_to_seed_change() {
        // Sample many points; the seed should perturb noise enough
        // that the two outputs diverge at most coordinates.
        let mut differ = 0usize;
        for i in 0..50 {
            let p = i as f32 * 0.37;
            if (fbm_2d(p, p, 4, 1) - fbm_2d(p, p, 4, 2)).abs() > 1e-3 {
                differ += 1;
            }
        }
        assert!(differ > 25, "{differ}/50 differed across seeds");
    }

    // ── analytic gradients ─────────────────────────────────────────

    /// The value returned by the `_grad` variant must equal the plain
    /// value function bit-for-bit (same arithmetic) — the gradient is a
    /// free by-product, not a re-derivation.
    #[test]
    fn value_noise_grad_value_matches_plain() {
        for &(x, z) in &[(1.3, -2.7), (0.0, 0.0), (5.5, 5.5), (-4.2, 8.1)] {
            let (v, _, _) = value_noise_2d_grad(x, z, 7);
            assert_eq!(v, value_noise_2d(x, z, 7), "value mismatch at {x},{z}");
        }
        for &(x, z) in &[(1.3, -2.7), (0.31, 9.9), (-4.2, 8.1)] {
            let (v, _, _) = fbm_2d_grad(x, z, 5, 7);
            assert_eq!(v, fbm_2d(x, z, 5, 7), "fbm value mismatch at {x},{z}");
        }
    }

    /// Analytic ∂/∂x, ∂/∂z match a tight central difference. Sampled at
    /// many off-lattice points (the lattice corners have kinks where the
    /// FD is a poor reference) and several seeds. Tolerance is loose
    /// enough for the O(h²) FD truncation but tight enough to catch a
    /// wrong factor.
    #[test]
    fn value_noise_grad_matches_central_difference() {
        let h = 1e-3_f32;
        for seed in [1u32, 42, 9999] {
            for i in 0..23 {
                for j in 0..23 {
                    // Off-lattice fractional offsets (avoid integer xf/zf).
                    let x = i as f32 * 0.41 + 0.17;
                    let z = j as f32 * 0.37 - 0.23;
                    let (_, dvdx, dvdz) = value_noise_2d_grad(x, z, seed);
                    let fd_x = (value_noise_2d(x + h, z, seed)
                        - value_noise_2d(x - h, z, seed))
                        / (2.0 * h);
                    let fd_z = (value_noise_2d(x, z + h, seed)
                        - value_noise_2d(x, z - h, seed))
                        / (2.0 * h);
                    assert!(
                        (dvdx - fd_x).abs() < 2e-3 && (dvdz - fd_z).abs() < 2e-3,
                        "seed {seed} at ({x:.3},{z:.3}): analytic ({dvdx:.5},{dvdz:.5}) \
                         vs FD ({fd_x:.5},{fd_z:.5})"
                    );
                }
            }
        }
    }

    /// FBM gradient matches a central difference across octave counts.
    ///
    /// Smoothstep value-noise is only C1: its 2nd derivative jumps
    /// across lattice lines, so a central difference straddling a line
    /// degrades from O(h²) to O(h) accuracy. That's a flaw in the FD
    /// *reference*, not the analytic gradient (which is exact and 0
    /// there). For higher octaves the input is `coord·2^o`, so a point
    /// can sit on a lattice line for one octave while off it for others.
    /// We skip any point within a small margin of a lattice line for any
    /// active octave; the off-lattice samples that remain are where the
    /// FD is a faithful reference.
    #[test]
    fn fbm_grad_matches_central_difference() {
        let h = 1e-3_f32;
        // True iff `coord·2^o` is within `margin` of an integer for any
        // octave `o < octaves` — i.e. near a C1 curvature kink.
        let near_lattice = |coord: f32, octaves: u8| -> bool {
            let mut freq = 1.0f32;
            for _ in 0..octaves {
                let f = (coord * freq).rem_euclid(1.0);
                if f < 0.1 || f > 0.9 {
                    return true;
                }
                freq *= 2.0;
            }
            false
        };
        for octaves in [1u8, 3, 5] {
            let mut tested = 0usize;
            for i in 0..40 {
                for j in 0..40 {
                    let x = i as f32 * 0.53 + 0.11;
                    let z = j as f32 * 0.29 - 0.31;
                    if near_lattice(x, octaves) || near_lattice(z, octaves) {
                        continue;
                    }
                    tested += 1;
                    let (_, dvdx, dvdz) = fbm_2d_grad(x, z, octaves, 5);
                    let fd_x =
                        (fbm_2d(x + h, z, octaves, 5) - fbm_2d(x - h, z, octaves, 5)) / (2.0 * h);
                    let fd_z =
                        (fbm_2d(x, z + h, octaves, 5) - fbm_2d(x, z - h, octaves, 5)) / (2.0 * h);
                    // Higher octaves pack higher frequencies → larger FD
                    // truncation even off-lattice, so scale tolerance.
                    let tol = 4e-3 * octaves as f32;
                    assert!(
                        (dvdx - fd_x).abs() < tol && (dvdz - fd_z).abs() < tol,
                        "octaves {octaves} at ({x:.3},{z:.3}): analytic \
                         ({dvdx:.5},{dvdz:.5}) vs FD ({fd_x:.5},{fd_z:.5})"
                    );
                }
            }
            assert!(tested > 50, "octaves {octaves}: only {tested} off-lattice samples");
        }
    }

    #[test]
    fn grad_is_deterministic() {
        assert_eq!(value_noise_2d_grad(1.7, -3.4, 9), value_noise_2d_grad(1.7, -3.4, 9));
        assert_eq!(fbm_2d_grad(1.7, -3.4, 5, 9), fbm_2d_grad(1.7, -3.4, 5, 9));
    }
}
