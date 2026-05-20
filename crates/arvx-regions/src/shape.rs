//! Region shape primitives + their signed-distance functions.
//!
//! Every variant produces a signed distance to its surface:
//! `sd < 0` inside, `sd == 0` on the surface, `sd > 0` outside. The
//! distance is in metres and exact (not just a sign) so the falloff
//! curve's metric-distance contract holds.
//!
//! Shapes are expressed in world-space already: each variant carries
//! its own transform (`Sphere` stores its centre, `Box` stores its
//! centre + half-extents, `Obb` stores centre + rotation +
//! half-extents). Decoupling shape transforms from the consuming
//! `Transform` component keeps the math local and lets a region's
//! Transform stay at the entity's pivot (which the gizmo and
//! scene-tree expect to be the centre of mass — matches the rest of
//! the editor's transform convention).
//!
//! Volumetric (voxelised) regions are V2 territory and will land as
//! an additive enum variant.

use arvx_core::Aabb;
use glam::{Quat, Vec3};

/// Region shape. V1 ships three analytical primitives; voxelised is V2.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RegionShape {
    /// Centred sphere of `radius`.
    Sphere {
        /// Sphere radius in metres. Negative values clamp to `0`.
        radius: f32,
    },
    /// Axis-aligned (world-axes) box defined by half-extents around
    /// the entity's pivot.
    Box {
        /// Per-axis half-extents in metres.
        half_extents: Vec3,
    },
    /// Oriented box. The rotation is applied around the entity's
    /// pivot; half-extents are along the rotated local axes.
    Obb {
        /// Per-axis half-extents in the rotated local frame, metres.
        half_extents: Vec3,
        /// Rotation from world axes to the box's local axes.
        rotation: Quat,
    },
}

impl Default for RegionShape {
    fn default() -> Self {
        Self::Sphere { radius: 25.0 }
    }
}

impl RegionShape {
    /// Signed distance from `point` to the shape's surface, with the
    /// shape positioned at `center`.
    ///
    /// Returns `sd < 0` inside, `sd > 0` outside, `sd == 0` on the
    /// surface. Distance is in metres and exact (not a lower bound),
    /// so [`crate::Falloff::apply`]'s metric contract holds.
    pub fn signed_distance(&self, center: Vec3, point: Vec3) -> f32 {
        match *self {
            Self::Sphere { radius } => {
                let r = radius.max(0.0);
                (point - center).length() - r
            }
            Self::Box { half_extents } => sd_box(point - center, half_extents),
            Self::Obb { half_extents, rotation } => {
                // Transform `point` into the box's local frame, then
                // call the axis-aligned SDF — distance is invariant
                // under rotation.
                let local = rotation.inverse() * (point - center);
                sd_box(local, half_extents)
            }
        }
    }

    /// World-space AABB tightly bounding the shape positioned at
    /// `center`, expanded by `transition_m` to cover the falloff band.
    ///
    /// Index lookups use this to prune candidates: a point further
    /// from `center` than this AABB allows can't possibly have
    /// non-zero membership weight.
    pub fn world_aabb_with_falloff(&self, center: Vec3, transition_m: f32) -> Aabb {
        let pad = transition_m.max(0.0);
        match *self {
            Self::Sphere { radius } => {
                let r = radius.max(0.0) + pad;
                Aabb::from_center_half_extents(center, Vec3::splat(r))
            }
            Self::Box { half_extents } => {
                let he = half_extents.max(Vec3::ZERO) + Vec3::splat(pad);
                Aabb::from_center_half_extents(center, he)
            }
            Self::Obb { half_extents, rotation } => {
                // Axis-aligned bound of an OBB: project each rotated
                // half-extent onto the world axes and sum the absolute
                // contributions per axis.
                let he = half_extents.max(Vec3::ZERO);
                let x_axis = (rotation * Vec3::X) * he.x;
                let y_axis = (rotation * Vec3::Y) * he.y;
                let z_axis = (rotation * Vec3::Z) * he.z;
                let aabb_he = Vec3::new(
                    x_axis.x.abs() + y_axis.x.abs() + z_axis.x.abs(),
                    x_axis.y.abs() + y_axis.y.abs() + z_axis.y.abs(),
                    x_axis.z.abs() + y_axis.z.abs() + z_axis.z.abs(),
                ) + Vec3::splat(pad);
                Aabb::from_center_half_extents(center, aabb_he)
            }
        }
    }
}

/// Signed distance from `p` to an axis-aligned box centred at the
/// origin with `half_extents`. Negative inside, zero on the surface,
/// positive outside. Standard IQ-style formulation.
fn sd_box(p: Vec3, half_extents: Vec3) -> f32 {
    let he = half_extents.max(Vec3::ZERO);
    let q = p.abs() - he;
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sphere ─────────────────────────────────────────────────────

    #[test]
    fn sphere_sd_zero_on_surface() {
        let s = RegionShape::Sphere { radius: 10.0 };
        let sd = s.signed_distance(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        assert!(sd.abs() < 1e-5, "sd={sd}");
    }

    #[test]
    fn sphere_sd_negative_inside() {
        let s = RegionShape::Sphere { radius: 10.0 };
        let sd = s.signed_distance(Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0));
        // Exact distance to surface = radius - 3 = 7, sign negative.
        assert!((sd + 7.0).abs() < 1e-5, "sd={sd}");
    }

    #[test]
    fn sphere_sd_positive_outside() {
        let s = RegionShape::Sphere { radius: 5.0 };
        let sd = s.signed_distance(Vec3::ZERO, Vec3::new(0.0, 0.0, 12.0));
        // Exact distance = 12 - 5 = 7.
        assert!((sd - 7.0).abs() < 1e-5);
    }

    #[test]
    fn sphere_off_origin_centre() {
        let s = RegionShape::Sphere { radius: 4.0 };
        let sd = s.signed_distance(Vec3::new(100.0, 50.0, 200.0), Vec3::new(104.0, 50.0, 200.0));
        assert!(sd.abs() < 1e-5);
    }

    #[test]
    fn sphere_negative_radius_collapses_to_point() {
        let s = RegionShape::Sphere { radius: -3.0 };
        let sd = s.signed_distance(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        // Negative-radius sphere is treated as a degenerate point at the
        // centre; sd is the distance to the point.
        assert!((sd - 2.0).abs() < 1e-5);
    }

    // ── Box ────────────────────────────────────────────────────────

    #[test]
    fn box_sd_zero_on_face() {
        let b = RegionShape::Box {
            half_extents: Vec3::new(2.0, 3.0, 4.0),
        };
        let sd = b.signed_distance(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        assert!(sd.abs() < 1e-5);
    }

    #[test]
    fn box_sd_negative_inside_at_centre_equals_neg_min_half_extent() {
        // The deepest point inside is at the centre; sd equals the
        // negative minimum half-extent.
        let b = RegionShape::Box {
            half_extents: Vec3::new(2.0, 3.0, 4.0),
        };
        let sd = b.signed_distance(Vec3::ZERO, Vec3::ZERO);
        assert!((sd + 2.0).abs() < 1e-5, "sd={sd}");
    }

    #[test]
    fn box_sd_positive_outside_corner() {
        let b = RegionShape::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        };
        // Outside a corner: queryPos = (2, 2, 0), corner = (1, 1, 0),
        // dist = sqrt(1^2 + 1^2 + 0) = sqrt(2).
        let sd = b.signed_distance(Vec3::ZERO, Vec3::new(2.0, 2.0, 0.0));
        assert!((sd - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    // ── OBB ────────────────────────────────────────────────────────

    #[test]
    fn obb_with_identity_rotation_matches_box() {
        let q = Quat::IDENTITY;
        let b = RegionShape::Box {
            half_extents: Vec3::new(2.0, 3.0, 4.0),
        };
        let o = RegionShape::Obb {
            half_extents: Vec3::new(2.0, 3.0, 4.0),
            rotation: q,
        };
        for p in [
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(-3.0, 4.0, -1.0),
        ] {
            let sd_b = b.signed_distance(Vec3::ZERO, p);
            let sd_o = o.signed_distance(Vec3::ZERO, p);
            assert!((sd_b - sd_o).abs() < 1e-4, "p={p:?} box={sd_b} obb={sd_o}");
        }
    }

    #[test]
    fn obb_rotation_actually_rotates_extents() {
        // Half-extents (5, 0.5, 0.5), rotated 90° around Y. After
        // rotation the long axis runs along world Z.
        let o = RegionShape::Obb {
            half_extents: Vec3::new(5.0, 0.5, 0.5),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        };
        // Point along world Z at distance 4 from origin should be
        // INSIDE the rotated box (because the long axis is now Z).
        let sd_inside = o.signed_distance(Vec3::ZERO, Vec3::new(0.0, 0.0, 4.0));
        assert!(sd_inside < 0.0, "expected inside, sd={sd_inside}");
        // Same point along world X is OUTSIDE the rotated box.
        let sd_outside = o.signed_distance(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0));
        assert!(sd_outside > 0.0, "expected outside, sd={sd_outside}");
    }

    // ── world_aabb_with_falloff ────────────────────────────────────

    #[test]
    fn sphere_aabb_radius_plus_falloff() {
        let s = RegionShape::Sphere { radius: 10.0 };
        let aabb = s.world_aabb_with_falloff(Vec3::new(0.0, 0.0, 0.0), 5.0);
        assert!((aabb.max.x - 15.0).abs() < 1e-4);
        assert!((aabb.min.x + 15.0).abs() < 1e-4);
    }

    #[test]
    fn box_aabb_expanded_by_falloff() {
        let b = RegionShape::Box {
            half_extents: Vec3::new(2.0, 3.0, 4.0),
        };
        let aabb = b.world_aabb_with_falloff(Vec3::new(10.0, 20.0, 30.0), 1.0);
        assert!((aabb.min.x - 7.0).abs() < 1e-4);
        assert!((aabb.max.x - 13.0).abs() < 1e-4);
        assert!((aabb.min.y - 16.0).abs() < 1e-4);
        assert!((aabb.max.y - 24.0).abs() < 1e-4);
    }

    #[test]
    fn obb_aabb_grows_under_rotation() {
        // 45° rotation of a (10, 0.5, 0.5) box around Y. The X-axis
        // projection of the rotated x-axis half-extent = 10 * cos45
        // = 10/sqrt(2). The Z-axis projection of the rotated x-axis
        // = 10 * sin45 = 10/sqrt(2). So aabb_he.x ≈ 10/sqrt(2) +
        // small contribution from y/z.
        let o = RegionShape::Obb {
            half_extents: Vec3::new(10.0, 0.5, 0.5),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        };
        let aabb = o.world_aabb_with_falloff(Vec3::ZERO, 0.0);
        let expected_x = 10.0 * (std::f32::consts::FRAC_1_SQRT_2);
        // (Plus a tiny contribution from the y/z axes' projections — the
        // y-axis after Y-rotation is still Y, so it doesn't add to X.
        // The z-axis after Y-rotation lands in the XZ plane and adds
        // 0.5 * sin45 to X.)
        let extra = 0.5 * std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (aabb.max.x - (expected_x + extra)).abs() < 1e-4,
            "x = {} expected {}",
            aabb.max.x,
            expected_x + extra,
        );
    }

    // ── serde ──────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrips_sphere() {
        let s = RegionShape::Sphere { radius: 12.5 };
        let json = serde_json::to_string(&s).unwrap();
        let back: RegionShape = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn serde_roundtrips_obb() {
        let o = RegionShape::Obb {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(0.5),
        };
        let json = serde_json::to_string(&o).unwrap();
        let back: RegionShape = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }
}
