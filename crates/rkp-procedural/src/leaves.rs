//! Analytical shape evaluation — each leaf computes a signed distance, then
//! converts to opacity via a smooth falloff.
//!
//! Every shape function takes a position in **local space** (already transformed
//! by the node's inverse transform) and returns a [`Sample`].

use glam::Vec3;

use crate::node_kind::*;
use crate::sample::Sample;

/// Convert a signed distance to opacity using a linear falloff.
///
/// - `dist < 0` (inside): opacity = 1.0
/// - `0 < dist < falloff`: opacity transitions linearly from 1.0 to 0.0
/// - `dist > falloff`: opacity = 0.0
///
/// Falloff is clamped to a minimum to avoid division by zero.
fn distance_to_opacity(dist: f32, falloff: f32) -> f32 {
    let falloff = falloff.max(1e-6);
    (1.0 - dist / falloff).clamp(0.0, 1.0)
}

/// Evaluate a sphere.
pub fn eval_sphere(pos: Vec3, params: &SphereParams) -> Sample {
    let dist = pos.length() - params.radius;
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate an axis-aligned box with optional rounding.
pub fn eval_box(pos: Vec3, params: &BoxParams) -> Sample {
    // Signed distance to a rounded box.
    let q = pos.abs() - params.half_extents + Vec3::splat(params.rounding);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    let dist = outside + inside - params.rounding;
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate a capsule along the Y axis.
pub fn eval_capsule(pos: Vec3, params: &CapsuleParams) -> Sample {
    // Closest point on the line segment from (0, -h, 0) to (0, h, 0).
    let half_h = params.half_height;
    let t = pos.y.clamp(-half_h, half_h);
    let closest = Vec3::new(0.0, t, 0.0);
    let dist = (pos - closest).length() - params.radius;
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate a cylinder along the Y axis.
pub fn eval_cylinder(pos: Vec3, params: &CylinderParams) -> Sample {
    let radial = Vec3::new(pos.x, 0.0, pos.z).length() - params.radius;
    let axial = pos.y.abs() - params.half_height;
    // SDF for a capped cylinder: max of radial and axial distances.
    let dist = if radial > 0.0 && axial > 0.0 {
        // Outside both the barrel and cap — distance to the edge ring.
        (radial * radial + axial * axial).sqrt()
    } else {
        // Inside at least one — take the larger (less negative) distance.
        radial.max(axial)
    };
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate a torus in the XZ plane.
pub fn eval_torus(pos: Vec3, params: &TorusParams) -> Sample {
    let xz_len = Vec3::new(pos.x, 0.0, pos.z).length();
    let q = Vec3::new(xz_len - params.major_radius, pos.y, 0.0);
    let dist = q.length() - params.minor_radius;
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate an infinite plane with Y-up normal at y=0.
pub fn eval_plane(pos: Vec3, params: &PlaneParams) -> Sample {
    // Signed distance: positive above the plane, negative below.
    let dist = pos.y;
    let opacity = distance_to_opacity(dist, params.falloff);
    Sample::with_color(opacity, params.material_id, params.color)
}

/// Evaluate any leaf node kind. Dispatches to the appropriate shape function.
///
/// `pos` must be in the node's **local space**.
///
/// Panics if called with a combinator kind.
pub fn eval_leaf(pos: Vec3, kind: &NodeKind) -> Sample {
    match kind {
        NodeKind::Sphere(p) => eval_sphere(pos, p),
        NodeKind::Box(p) => eval_box(pos, p),
        NodeKind::Capsule(p) => eval_capsule(pos, p),
        NodeKind::Cylinder(p) => eval_cylinder(pos, p),
        NodeKind::Torus(p) => eval_torus(pos, p),
        NodeKind::Plane(p) => eval_plane(pos, p),
        _ => panic!("eval_leaf called with a non-leaf node kind"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn sphere_center_is_opaque() {
        let s = eval_sphere(Vec3::ZERO, &SphereParams::default());
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn sphere_far_away_is_empty() {
        let s = eval_sphere(Vec3::new(100.0, 0.0, 0.0), &SphereParams::default());
        assert!(s.opacity < EPS);
    }

    #[test]
    fn sphere_surface_is_partial() {
        let params = SphereParams {
            radius: 1.0,
            falloff: 0.2,
            ..Default::default()
        };
        // Exactly at surface: dist = 0, opacity should be 1.0.
        let s = eval_sphere(Vec3::new(1.0, 0.0, 0.0), &params);
        assert!((s.opacity - 1.0).abs() < EPS);

        // Half-way through falloff.
        let s = eval_sphere(Vec3::new(1.1, 0.0, 0.0), &params);
        assert!((s.opacity - 0.5).abs() < EPS);
    }

    #[test]
    fn box_center_is_opaque() {
        let s = eval_box(Vec3::ZERO, &BoxParams::default());
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn box_outside_is_empty() {
        let s = eval_box(Vec3::new(10.0, 0.0, 0.0), &BoxParams::default());
        assert!(s.opacity < EPS);
    }

    #[test]
    fn capsule_on_axis_is_opaque() {
        let s = eval_capsule(Vec3::new(0.0, 0.3, 0.0), &CapsuleParams::default());
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn cylinder_center_is_opaque() {
        let s = eval_cylinder(Vec3::ZERO, &CylinderParams::default());
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn torus_on_ring_is_opaque() {
        let params = TorusParams::default();
        // Point on the ring at (major_radius, 0, 0).
        let s = eval_torus(Vec3::new(params.major_radius, 0.0, 0.0), &params);
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn torus_center_is_empty() {
        let params = TorusParams::default();
        let s = eval_torus(Vec3::ZERO, &params);
        // Center of the torus hole should be empty (major_radius >> minor_radius).
        assert!(s.opacity < 0.5);
    }

    #[test]
    fn plane_below_is_opaque() {
        let s = eval_plane(Vec3::new(0.0, -1.0, 0.0), &PlaneParams::default());
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn plane_above_is_empty() {
        let s = eval_plane(Vec3::new(0.0, 1.0, 0.0), &PlaneParams::default());
        assert!(s.opacity < EPS);
    }

    #[test]
    fn eval_leaf_dispatches() {
        let kind = NodeKind::Sphere(SphereParams::default());
        let s = eval_leaf(Vec3::ZERO, &kind);
        assert!((s.opacity - 1.0).abs() < EPS);
    }

    #[test]
    fn material_and_color_propagate() {
        let params = SphereParams {
            material_id: 42,
            color: Vec3::new(0.5, 0.3, 0.1),
            ..Default::default()
        };
        let s = eval_sphere(Vec3::ZERO, &params);
        assert_eq!(s.material_id, 42);
        assert_eq!(s.color, Vec3::new(0.5, 0.3, 0.1));
    }
}
