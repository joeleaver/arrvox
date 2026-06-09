//! Procedural source for terrain — Layer 1 of the three-layer model.
//!
//! `TerrainFn::sample` takes a `(TileKey, local, voxel_size)` triple
//! and returns a `TerrainSample`. The integer tile key seeds any noise
//! lookups deterministically across the whole world; the local f32
//! stays bounded inside one tile so noise inputs never collapse
//! precision at large world coords.

use crate::tile_key::TileKey;
use glam::Vec3;

/// One sample of the procedural terrain source at a leaf cell.
///
/// `sd` follows the engine SDF convention: negative inside the
/// surface, positive outside, zero on the surface. The 1-Lipschitz
/// property of a true SDF is what makes the voxelizer's coarse-level
/// Empty/Interior classifier provably correct — implementations must
/// return a real signed distance, not just a sign-correct field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainSample {
    /// Signed distance to the terrain surface.
    pub sd: f32,
    /// Primary material id (16-bit; matches `arvx-core`'s palette).
    pub primary_mat: u16,
    /// Secondary material id for dual-material blending. Same value as
    /// `primary_mat` when no blend is wanted.
    pub secondary_mat: u16,
    /// Blend weight in `[0, 1]` between primary (0.0) and secondary
    /// (1.0). Quantised to 4 bits inside the voxelizer; pass `0.0` for
    /// single-material output.
    pub blend: f32,
}

/// Procedural terrain source. Implementations are user-supplied.
///
/// Sampled in tile-local coordinates so noise impls can seed on the
/// integer `TileKey` and never collapse f32 precision at large world
/// coordinates — see `docs/TERRAIN.md`'s "Floating-point handling"
/// section.
pub trait TerrainFn: Send + Sync {
    /// Evaluate the source at one leaf cell.
    ///
    /// * `tile` — the key for the tile this sample is inside. Stable
    ///   integer across the world; seed noise lookups on this.
    /// * `local` — the sample's position relative to the tile's lo
    ///   corner, in metres. Always in `[0, tile.extent_m())` plus the
    ///   gradient-tap epsilon, regardless of how far the tile is from
    ///   the world origin.
    /// * `voxel_size_m` — the voxel size used for this tile's
    ///   voxelization, in metres. Implementations can use it to pick
    ///   an appropriate noise frequency for the resolution.
    fn sample(&self, tile: TileKey, local: Vec3, voxel_size_m: f32) -> TerrainSample;

    /// Optional analytic gradient of the signed-distance field —
    /// `∇sd = (∂sd/∂x, ∂sd/∂y, ∂sd/∂z)` in world units — at the same
    /// point [`Self::sample`] evaluates.
    ///
    /// When `Some`, the voxelizer derives the per-leaf surface normal
    /// (`∇sd.normalize()`) and the exact perpendicular distance directly
    /// from this, skipping its 6-tap finite difference: exact normals,
    /// faster bakes. When `None` (the default), the voxelizer falls back
    /// to the finite difference. Implementations whose field is not
    /// cheaply differentiable should leave this `None`.
    ///
    /// MUST be consistent with `sample` at the same `(tile, local,
    /// voxel_size_m)` — same coordinate transform, same LOD octave
    /// clamp. The bake path only consults it where the height is the
    /// raw procedural value (no stamp / region / world-envelope warp),
    /// since those post-modifications invalidate the analytic gradient.
    fn sample_grad(&self, _tile: TileKey, _local: Vec3, _voxel_size_m: f32) -> Option<Vec3> {
        None
    }

    /// Combined [`Self::sample`] + [`Self::sample_grad`] in one call.
    ///
    /// The default just forwards to both, but heightfield impls can
    /// override it to compute height, gradient, slope, AND material in a
    /// single noise walk — the gradient *is* the slope (`atan(|∇h|)`), so
    /// a separate finite-difference slope probe for material assignment
    /// is redundant. The bake path calls this once per sample instead of
    /// `sample` + `sample_grad` separately.
    fn sample_with_grad(
        &self,
        tile: TileKey,
        local: Vec3,
        voxel_size_m: f32,
    ) -> (TerrainSample, Option<Vec3>) {
        (
            self.sample(tile, local, voxel_size_m),
            self.sample_grad(tile, local, voxel_size_m),
        )
    }

    /// Conservative world-space `(min_y, max_y)` the surface can occupy,
    /// if cheaply known. The bake skips the halo-sampling pass for tiles
    /// whose whole Y-span (expanded by the halo reach) lies entirely
    /// outside this band — they're provably all-sky or all-solid. MUST
    /// over-approximate (never clip the real surface). `None` (the
    /// default) disables the skip. The bake only consults it on tiles
    /// with no stamps/regions/envelope, which can push the surface beyond
    /// the base field's range.
    fn surface_y_bounds(&self) -> Option<(f32, f32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlatPlane;

    impl TerrainFn for FlatPlane {
        fn sample(&self, _tile: TileKey, local: Vec3, _voxel_size_m: f32) -> TerrainSample {
            TerrainSample {
                sd: local.y - 5.0,
                primary_mat: 1,
                secondary_mat: 1,
                blend: 0.0,
            }
        }
    }

    #[test]
    fn flat_plane_returns_expected_sd() {
        let f = FlatPlane;
        let s_below = f.sample(TileKey::level0(0, 0, 0), Vec3::new(0.0, 0.0, 0.0), 0.08);
        let s_above = f.sample(TileKey::level0(0, 0, 0), Vec3::new(0.0, 10.0, 0.0), 0.08);
        assert!(s_below.sd < 0.0, "below surface should be negative; got {}", s_below.sd);
        assert!(s_above.sd > 0.0, "above surface should be positive; got {}", s_above.sd);
        assert_eq!(s_below.primary_mat, 1);
    }

    /// Returning the same value for the same input is the determinism
    /// contract for noise impls.
    #[test]
    fn flat_plane_is_deterministic() {
        let f = FlatPlane;
        let p = Vec3::new(3.0, 7.0, 11.0);
        let k = TileKey::level0(42, -7, 13);
        let a = f.sample(k, p, 0.08);
        let b = f.sample(k, p, 0.08);
        assert_eq!(a, b);
    }
}
