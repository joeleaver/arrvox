//! Tiny deterministic 3D noise used by `NodeKind::NoiseDisplace`.
//!
//! Not simplex in the academic sense — it's value noise with a 3D
//! hash, wrapped in a cubic-hermite filter for C1 continuity. That's
//! plenty for domain-warping an SDF: the eye doesn't distinguish
//! between simplex and hashed-value noise at the frequencies we
//! care about, and hashed-value is ~30 lines and ports verbatim to
//! WGSL for an eventual GPU path (no permutation tables).
//!
//! The noise is seeded by `seed` + deterministic for a given position,
//! so a rebake reproduces the same displaced geometry.
//!
//! `noise_3d_vec` returns three independent noise components (via seed
//! offsets) — call it once per sample point, multiply by amplitude,
//! add to `pos`. Output magnitude is bounded: each component in
//! `[-1, 1]`, so `|output| <= sqrt(3)` worst case.

use glam::Vec3;

/// Bit-mixing hash → f32 in [-1, 1]. Matches what a matching WGSL
/// port would do so the CPU and future GPU paths produce the same
/// noise values on identical inputs.
#[inline(always)]
fn hash_f32(x: u32) -> f32 {
    // Variant of xorshift — chosen for reasonable spectral properties
    // at low cost, not cryptographic.
    let mut n = x;
    n = (n ^ 61) ^ (n >> 16);
    n = n.wrapping_mul(9);
    n ^= n >> 4;
    n = n.wrapping_mul(0x27d4_eb2d);
    n ^= n >> 15;
    // Map the u32 to [-1, 1] via 0..2^24 → [0, 1] → [-1, 1]. Keeping
    // the range to 24 bits sidesteps the f32-mantissa precision loss.
    let unit = (n & 0x00ff_ffff) as f32 * (1.0 / 16_777_216.0);
    unit * 2.0 - 1.0
}

#[inline(always)]
fn hash_3i(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    // Ronchetti-style spatial hash: each axis shifted by a prime and
    // xor'd into `seed`. Works because we take `hash_f32` after, which
    // does the actual mixing.
    let k = (ix as u32)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add((iy as u32).wrapping_mul(0x7ed5_5d16))
        .wrapping_add((iz as u32).wrapping_mul(0xa3a5_2d49))
        .wrapping_add(seed);
    hash_f32(k)
}

/// Smoothstep-like cubic hermite — zero first derivative at 0 and 1.
#[inline(always)]
fn smootherstep(t: f32) -> f32 {
    // Ken Perlin's quintic: 6t^5 - 15t^4 + 10t^3. Gives C2
    // continuity at the cost of five muls per axis; cheap.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Single-octave 3D value noise. Returns `[-1, 1]`.
pub fn noise_3d(pos: Vec3, seed: u32) -> f32 {
    let xf = pos.x.floor();
    let yf = pos.y.floor();
    let zf = pos.z.floor();
    let ix = xf as i32;
    let iy = yf as i32;
    let iz = zf as i32;

    let tx = smootherstep(pos.x - xf);
    let ty = smootherstep(pos.y - yf);
    let tz = smootherstep(pos.z - zf);

    // 8 corner samples of the unit cube at (ix..ix+1, iy..iy+1, iz..iz+1).
    let c000 = hash_3i(ix, iy, iz, seed);
    let c100 = hash_3i(ix + 1, iy, iz, seed);
    let c010 = hash_3i(ix, iy + 1, iz, seed);
    let c110 = hash_3i(ix + 1, iy + 1, iz, seed);
    let c001 = hash_3i(ix, iy, iz + 1, seed);
    let c101 = hash_3i(ix + 1, iy, iz + 1, seed);
    let c011 = hash_3i(ix, iy + 1, iz + 1, seed);
    let c111 = hash_3i(ix + 1, iy + 1, iz + 1, seed);

    // Trilerp.
    let x00 = c000 + (c100 - c000) * tx;
    let x10 = c010 + (c110 - c010) * tx;
    let x01 = c001 + (c101 - c001) * tx;
    let x11 = c011 + (c111 - c011) * tx;
    let y0 = x00 + (x10 - x00) * ty;
    let y1 = x01 + (x11 - x01) * ty;
    y0 + (y1 - y0) * tz
}

/// 3D vector noise — three independent scalar-noise values with
/// separated seeds. Useful as a domain-warp: `pos + noise_3d_vec(pos)
/// * amplitude` distorts the space the SDF is evaluated in.
pub fn noise_3d_vec(pos: Vec3, seed: u32) -> Vec3 {
    // Seed offsets chosen to be large relative to the spatial hash
    // primes so the three components don't alias each other.
    Vec3::new(
        noise_3d(pos, seed),
        noise_3d(pos, seed.wrapping_add(0x9e37_79b1)),
        noise_3d(pos, seed.wrapping_add(0xb746_84ab)),
    )
}

/// FBM — sum `octaves` layers of noise at doubling frequencies and
/// halving amplitudes. Output bounded to `[-1, 1]` via the standard
/// `1/(2 - 2^(1-octaves))` normalizer.
///
/// `octaves` is clamped to `[1, 8]` so a bad param value can't stall
/// the voxelizer on a per-sample basis.
pub fn fbm_3d_vec(pos: Vec3, frequency: f32, seed: u32, octaves: u32) -> Vec3 {
    let octaves = octaves.clamp(1, 8) as usize;
    let mut sum = Vec3::ZERO;
    let mut amp = 1.0f32;
    let mut freq = frequency.max(1e-6);
    let mut total_amp = 0.0f32;
    for k in 0..octaves {
        sum += noise_3d_vec(pos * freq, seed.wrapping_add(k as u32 * 131)) * amp;
        total_amp += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    // Normalize so the output stays roughly in `[-1, 1]` regardless of
    // octave count.
    sum / total_amp.max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scalar noise is deterministic for a fixed seed — same input
    /// twice must give the same output bit-for-bit.
    #[test]
    fn noise_is_deterministic() {
        let p = Vec3::new(1.234, -0.7, 3.14);
        let a = noise_3d(p, 42);
        let b = noise_3d(p, 42);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    /// Changing the seed changes the value at the same point (most of
    /// the time — coin-flip matches are vanishingly unlikely with
    /// reasonable hashes, but check a handful of points).
    #[test]
    fn different_seeds_disagree() {
        let mut disagreements = 0;
        for i in 0..20 {
            let p = Vec3::new(i as f32 * 0.13, i as f32 * -0.27, i as f32 * 0.71);
            if noise_3d(p, 0) != noise_3d(p, 1) {
                disagreements += 1;
            }
        }
        // Out of 20, essentially all should differ.
        assert!(disagreements > 15, "seeds too correlated: {disagreements}/20");
    }

    /// Scalar noise stays in `[-1, 1]` for a wide spread of inputs.
    #[test]
    fn noise_bounded() {
        for ix in -50..=50 {
            for iy in -50..=50 {
                for iz in -50..=50 {
                    let p = Vec3::new(
                        ix as f32 * 0.37,
                        iy as f32 * -0.21,
                        iz as f32 * 0.53,
                    );
                    let v = noise_3d(p, 7);
                    assert!(
                        (-1.0..=1.0).contains(&v),
                        "noise out of range at {p:?}: {v}",
                    );
                }
            }
        }
    }

    /// FBM respects the `[-1, 1]` envelope too.
    #[test]
    fn fbm_bounded() {
        for octaves in 1..=8 {
            for i in 0..200 {
                let p = Vec3::new(i as f32 * 0.3, i as f32 * 0.07, i as f32 * -0.11);
                let v = fbm_3d_vec(p, 1.5, 99, octaves);
                assert!(
                    v.x.abs() <= 1.05 && v.y.abs() <= 1.05 && v.z.abs() <= 1.05,
                    "fbm out of range at oct={octaves} p={p:?}: {v:?}",
                );
            }
        }
    }
}
