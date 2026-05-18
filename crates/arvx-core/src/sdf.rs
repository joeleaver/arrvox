//! SDF generation utilities — analytic SDF primitives for testing and ray marching.
//!
//! Provides analytic SDF primitives ([`sphere_sdf`], [`box_sdf`], [`capsule_sdf`])
//! and a smooth-min blending function ([`smin`]).

use glam::Vec3;

/// Signed distance from `point` to the surface of a sphere.
///
/// Negative inside, positive outside.
#[inline]
pub fn sphere_sdf(center: Vec3, radius: f32, point: Vec3) -> f32 {
    (point - center).length() - radius
}

/// Signed distance from `point` to a capsule defined by line segment `a`–`b`
/// with the given `radius`.
///
/// Negative inside, positive outside.
#[inline]
pub fn capsule_sdf(a: Vec3, b: Vec3, radius: f32, point: Vec3) -> f32 {
    let pa = point - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - radius
}

/// Signed distance from `point` to an axis-aligned box centered at the origin.
///
/// Negative inside, positive outside.
#[inline]
pub fn box_sdf(half_extents: Vec3, point: Vec3) -> f32 {
    let q = point.abs() - half_extents;
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// Polynomial smooth-min (smooth union) of two SDF distances.
///
/// Produces a C1-continuous blend between `a` and `b` within a radius of `k`.
/// When `|a - b| >= k`, returns `min(a, b)` exactly.
/// When `|a - b| < k`, returns a smoothly interpolated value below `min(a, b)`.
///
/// Use this instead of `f32::min` to avoid gradient discontinuities at SDF
/// intersections, which cause black shading artifacts from bad normals.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ Analytic SDF primitives ------

    #[test]
    fn sphere_sdf_at_center_is_negative_radius() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::ZERO);
        assert!((d - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_on_surface_is_zero() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_outside_is_positive() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_inside_is_negative() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(0.5, 0.0, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_offset_center() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let d = sphere_sdf(center, 0.5, center);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    // ------ Capsule SDF ------

    #[test]
    fn capsule_sdf_at_endpoint_a_center_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::ZERO);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_at_endpoint_b_center_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::new(0.0, 1.0, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_at_midpoint_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::new(0.0, 0.5, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_on_surface_is_zero() {
        let d = capsule_sdf(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            Vec3::new(0.5, 0.5, 0.0),
        );
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_outside_is_positive() {
        let d = capsule_sdf(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            Vec3::new(1.5, 0.5, 0.0),
        );
        assert!((d - 1.0).abs() < 1e-6);
    }

    // ------ Box SDF ------

    #[test]
    fn box_sdf_at_origin_is_negative() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::ZERO);
        assert!((d - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn box_sdf_on_face_is_zero() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn box_sdf_outside_face_is_positive() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_sdf_outside_corner() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(2.0, 2.0, 2.0));
        assert!((d - 3.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn box_sdf_inside_is_negative() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(0.5, 0.0, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }
}
