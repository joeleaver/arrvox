//! `evaluate_primitive` — dispatch an `SdfPrimitive` to the matching
//! analytic SDF in [`crate::sdf`]. Split out from rkf-core's
//! `voxelize_object.rs` during the arrvox/rkifield code split so we
//! don't drag along rkf-core's SDF voxelization helpers (which
//! arrvox doesn't use).

use glam::Vec3;

use crate::scene_node::SdfPrimitive;

/// Evaluate an [`SdfPrimitive`] at a local-space position. Returns
/// signed distance: negative inside, positive outside.
pub fn evaluate_primitive(primitive: &SdfPrimitive, pos: Vec3) -> f32 {
    match *primitive {
        SdfPrimitive::Sphere { radius } => pos.length() - radius,
        SdfPrimitive::Box { half_extents } => crate::sdf::box_sdf(half_extents, pos),
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => {
            let a = Vec3::new(0.0, -half_height, 0.0);
            let b = Vec3::new(0.0, half_height, 0.0);
            crate::sdf::capsule_sdf(a, b, radius, pos)
        }
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => {
            let q_x = Vec3::new(pos.x, 0.0, pos.z).length() - major_radius;
            let q = glam::Vec2::new(q_x, pos.y);
            q.length() - minor_radius
        }
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => {
            let d_radial = Vec3::new(pos.x, 0.0, pos.z).length() - radius;
            let d_height = pos.y.abs() - half_height;
            let outside = glam::Vec2::new(d_radial.max(0.0), d_height.max(0.0)).length();
            let inside = d_radial.max(d_height).min(0.0);
            outside + inside
        }
        SdfPrimitive::Plane { normal, distance } => pos.dot(normal) - distance,
    }
}
