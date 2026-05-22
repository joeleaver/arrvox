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
}
