//! Analytical-primitive SDFs for CPU picking.
//!
//! After Phase 4 of the procedural-math consolidation, the renderer
//! and voxel bake evaluate procedural trees entirely on the GPU
//! (`proc_eval.wgsl` / `proc_sample.wgsl`). The only CPU path that
//! remains is the BUILD-viewport ghost-pick, which sphere-traces one
//! primitive at a time to resolve "which cutter did the user click?"
//! when the surface is fully subtracted away.
//!
//! These SDFs mirror the WGSL versions in `proc_eval.wgsl` but return
//! plain `f32` distance — the click-path doesn't need material /
//! color / blend, and removing them lets the Sample struct go away
//! entirely.

use glam::Vec3;

use crate::node_kind::*;

pub fn sdf_sphere(pos: Vec3, params: &SphereParams) -> f32 {
    pos.length() - params.radius
}

pub fn sdf_box(pos: Vec3, params: &BoxParams) -> f32 {
    let q = pos.abs() - params.half_extents + Vec3::splat(params.rounding);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside - params.rounding
}

pub fn sdf_capsule(pos: Vec3, params: &CapsuleParams) -> f32 {
    let t = pos.y.clamp(-params.half_height, params.half_height);
    let closest = Vec3::new(0.0, t, 0.0);
    (pos - closest).length() - params.radius
}

pub fn sdf_cylinder(pos: Vec3, params: &CylinderParams) -> f32 {
    let radial = Vec3::new(pos.x, 0.0, pos.z).length() - params.radius;
    let axial = pos.y.abs() - params.half_height;
    if radial > 0.0 && axial > 0.0 {
        (radial * radial + axial * axial).sqrt()
    } else {
        radial.max(axial)
    }
}

pub fn sdf_torus(pos: Vec3, params: &TorusParams) -> f32 {
    let xz_len = Vec3::new(pos.x, 0.0, pos.z).length();
    let q = Vec3::new(xz_len - params.major_radius, pos.y, 0.0);
    q.length() - params.minor_radius
}

pub fn sdf_plane(pos: Vec3, _params: &PlaneParams) -> f32 {
    pos.y
}

pub fn sdf_ramp(pos: Vec3, params: &RampParams) -> f32 {
    let l = params.half_length;
    let h = params.half_height;
    let w = params.half_width;

    let q = pos.abs() - Vec3::new(l, h, w);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    let box_dist = outside + inside;

    let hyp = (l * l + h * h).sqrt().max(1e-6);
    let plane_dist = (l * pos.y - h * pos.x) / hyp;

    box_dist.max(plane_dist)
}

/// Signed distance from `pos` to the surface of a leaf primitive, in
/// the node's local space. Non-leaf kinds return `f32::INFINITY` —
/// callers are expected to pre-filter via `NodeKind::is_leaf`.
pub fn eval_leaf_distance(pos: Vec3, kind: &NodeKind) -> f32 {
    match kind {
        NodeKind::Sphere(p) => sdf_sphere(pos, p),
        NodeKind::Box(p) => sdf_box(pos, p),
        NodeKind::Capsule(p) => sdf_capsule(pos, p),
        NodeKind::Cylinder(p) => sdf_cylinder(pos, p),
        NodeKind::Torus(p) => sdf_torus(pos, p),
        NodeKind::Plane(p) => sdf_plane(pos, p),
        NodeKind::Ramp(p) => sdf_ramp(pos, p),
        _ => f32::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_center_is_inside() {
        assert!(sdf_sphere(Vec3::ZERO, &SphereParams::default()) < 0.0);
    }

    #[test]
    fn sphere_at_surface_is_zero() {
        let params = SphereParams { radius: 1.0, ..Default::default() };
        assert!(sdf_sphere(Vec3::new(1.0, 0.0, 0.0), &params).abs() < 1e-4);
    }

    #[test]
    fn box_center_is_inside() {
        assert!(sdf_box(Vec3::ZERO, &BoxParams::default()) < 0.0);
    }

    #[test]
    fn cylinder_center_is_inside() {
        assert!(sdf_cylinder(Vec3::ZERO, &CylinderParams::default()) < 0.0);
    }

    #[test]
    fn torus_on_ring_is_inside() {
        let params = TorusParams::default();
        assert!(sdf_torus(Vec3::new(params.major_radius, 0.0, 0.0), &params) < 0.0);
    }

    #[test]
    fn torus_center_is_outside() {
        assert!(sdf_torus(Vec3::ZERO, &TorusParams::default()) > 0.0);
    }

    #[test]
    fn plane_below_is_inside() {
        assert!(sdf_plane(Vec3::new(0.0, -1.0, 0.0), &PlaneParams::default()) < 0.0);
    }

    #[test]
    fn eval_leaf_dispatches() {
        let kind = NodeKind::Sphere(SphereParams::default());
        assert!(eval_leaf_distance(Vec3::ZERO, &kind) < 0.0);
    }
}
