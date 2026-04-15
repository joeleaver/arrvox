//! Analytical shape evaluation — each leaf computes a signed distance from
//! the surface at a world-space query point. Combinators merge these with
//! SDF algebra; the voxelizer samples the result and classifies regions.
//!
//! Every shape function takes a position in **local space** (already
//! transformed by the node's inverse transform) plus the current voxel size
//! (unused today, reserved for shape variants that want voxel-aware fading)
//! and returns a [`Sample`].

use glam::Vec3;

use crate::node_kind::*;
use crate::sample::Sample;

/// Evaluate a sphere.
pub fn eval_sphere(pos: Vec3, params: &SphereParams, _voxel_size: f32) -> Sample {
    let dist = pos.length() - params.radius;
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate an axis-aligned box with optional rounding.
pub fn eval_box(pos: Vec3, params: &BoxParams, _voxel_size: f32) -> Sample {
    let q = pos.abs() - params.half_extents + Vec3::splat(params.rounding);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    let dist = outside + inside - params.rounding;
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate a capsule along the Y axis.
pub fn eval_capsule(pos: Vec3, params: &CapsuleParams, _voxel_size: f32) -> Sample {
    let half_h = params.half_height;
    let t = pos.y.clamp(-half_h, half_h);
    let closest = Vec3::new(0.0, t, 0.0);
    let dist = (pos - closest).length() - params.radius;
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate a cylinder along the Y axis.
pub fn eval_cylinder(pos: Vec3, params: &CylinderParams, _voxel_size: f32) -> Sample {
    let radial = Vec3::new(pos.x, 0.0, pos.z).length() - params.radius;
    let axial = pos.y.abs() - params.half_height;
    let dist = if radial > 0.0 && axial > 0.0 {
        (radial * radial + axial * axial).sqrt()
    } else {
        radial.max(axial)
    };
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate a torus in the XZ plane.
pub fn eval_torus(pos: Vec3, params: &TorusParams, _voxel_size: f32) -> Sample {
    let xz_len = Vec3::new(pos.x, 0.0, pos.z).length();
    let q = Vec3::new(xz_len - params.major_radius, pos.y, 0.0);
    let dist = q.length() - params.minor_radius;
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate an infinite plane with Y-up normal at y=0. Inside = below the plane.
pub fn eval_plane(pos: Vec3, params: &PlaneParams, _voxel_size: f32) -> Sample {
    let dist = pos.y;
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate a ramp (triangular prism). Intersection of a bounding box with the
/// half-space below the diagonal plane from (-L, -H) to (+L, +H).
pub fn eval_ramp(pos: Vec3, params: &RampParams, _voxel_size: f32) -> Sample {
    let l = params.half_length;
    let h = params.half_height;
    let w = params.half_width;

    // Box SDF (sharp).
    let q = pos.abs() - Vec3::new(l, h, w);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    let box_dist = outside + inside;

    // Signed distance to the diagonal cut plane. Points above the slope are
    // outside the ramp.
    let hyp = (l * l + h * h).sqrt().max(1e-6);
    let plane_dist = (l * pos.y - h * pos.x) / hyp;

    // Intersection in SDF space: max.
    let dist = box_dist.max(plane_dist);
    Sample::with_color(dist, params.material_id, params.color)
}

/// Evaluate any leaf node kind. Dispatches to the appropriate shape function.
///
/// `pos` must be in the node's **local space**.
///
/// Panics if called with a combinator kind.
pub fn eval_leaf(pos: Vec3, kind: &NodeKind, voxel_size: f32) -> Sample {
    match kind {
        NodeKind::Sphere(p) => eval_sphere(pos, p, voxel_size),
        NodeKind::Box(p) => eval_box(pos, p, voxel_size),
        NodeKind::Capsule(p) => eval_capsule(pos, p, voxel_size),
        NodeKind::Cylinder(p) => eval_cylinder(pos, p, voxel_size),
        NodeKind::Torus(p) => eval_torus(pos, p, voxel_size),
        NodeKind::Plane(p) => eval_plane(pos, p, voxel_size),
        NodeKind::Ramp(p) => eval_ramp(pos, p, voxel_size),
        _ => panic!("eval_leaf called with a non-leaf node kind"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS: f32 = 0.02;

    #[test]
    fn sphere_center_is_inside() {
        let s = eval_sphere(Vec3::ZERO, &SphereParams::default(), VS);
        assert!(s.is_inside(), "expected inside, dist={}", s.distance);
    }

    #[test]
    fn sphere_far_away_is_outside() {
        let s = eval_sphere(Vec3::new(100.0, 0.0, 0.0), &SphereParams::default(), VS);
        assert!(!s.is_inside());
        assert!(s.distance > 10.0);
    }

    #[test]
    fn sphere_at_surface_has_zero_distance() {
        let params = SphereParams { radius: 1.0, ..Default::default() };
        let s = eval_sphere(Vec3::new(1.0, 0.0, 0.0), &params, VS);
        assert!(s.distance.abs() < 1e-4, "expected ~0 at surface, got {}", s.distance);
    }

    #[test]
    fn box_center_is_inside() {
        let s = eval_box(Vec3::ZERO, &BoxParams::default(), VS);
        assert!(s.is_inside());
    }

    #[test]
    fn box_outside_is_outside() {
        let s = eval_box(Vec3::new(10.0, 0.0, 0.0), &BoxParams::default(), VS);
        assert!(!s.is_inside());
    }

    #[test]
    fn capsule_on_axis_is_inside() {
        let s = eval_capsule(Vec3::new(0.0, 0.3, 0.0), &CapsuleParams::default(), VS);
        assert!(s.is_inside());
    }

    #[test]
    fn cylinder_center_is_inside() {
        let s = eval_cylinder(Vec3::ZERO, &CylinderParams::default(), VS);
        assert!(s.is_inside());
    }

    #[test]
    fn torus_on_ring_is_inside() {
        let params = TorusParams::default();
        let s = eval_torus(Vec3::new(params.major_radius, 0.0, 0.0), &params, VS);
        assert!(s.is_inside());
    }

    #[test]
    fn torus_center_is_outside() {
        let params = TorusParams::default();
        let s = eval_torus(Vec3::ZERO, &params, VS);
        assert!(!s.is_inside());
    }

    #[test]
    fn plane_below_is_inside() {
        let s = eval_plane(Vec3::new(0.0, -1.0, 0.0), &PlaneParams::default(), VS);
        assert!(s.is_inside());
    }

    #[test]
    fn plane_above_is_outside() {
        let s = eval_plane(Vec3::new(0.0, 1.0, 0.0), &PlaneParams::default(), VS);
        assert!(!s.is_inside());
    }

    #[test]
    fn eval_leaf_dispatches() {
        let kind = NodeKind::Sphere(SphereParams::default());
        let s = eval_leaf(Vec3::ZERO, &kind, VS);
        assert!(s.is_inside());
    }

    #[test]
    fn material_and_color_propagate() {
        let params = SphereParams {
            material_id: 42,
            color: Vec3::new(0.5, 0.3, 0.1),
            ..Default::default()
        };
        let s = eval_sphere(Vec3::ZERO, &params, VS);
        assert_eq!(s.material_id, 42);
        assert_eq!(s.color, Vec3::new(0.5, 0.3, 0.1));
    }
}
