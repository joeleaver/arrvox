//! Barill et al. 2018 BVH-accelerated generalized winding number.
//!
//! For each BVH subtree the construction pre-computes a dipole
//! approximation (see [`crate::bvh::build::WindingData`]). At query
//! time, subtrees that are far from the point relative to their
//! radius are approximated by the dipole formula
//! `dot(dipole, r) / |r|³`; closer subtrees recurse. At leaf level we
//! fall back to exact triangle solid-angle evaluation via the
//! Van Oosterom & Strackee formula.
//!
//! Returns values near ±1 inside a closed surface and near 0
//! outside. Handles triangle soups (non-watertight meshes) gracefully.

use glam::Vec3;

use super::build::BvhNode;

/// Opening-angle threshold for the Barnes–Hut criterion. Lower =
/// more accurate, slower. `0.3` is high-accuracy for inside/outside
/// classification at moderate cost.
pub(super) const BETA: f32 = 0.3;

pub(super) fn winding_recursive(
    node: &BvhNode,
    positions: &[[Vec3; 3]],
    tri_order: &[usize],
    point: Vec3,
) -> f32 {
    let winding_data = node.winding();

    let r = point - winding_data.centroid;
    let r_len = r.length();

    if r_len > 1e-10 && winding_data.radius / r_len < BETA {
        let r3 = r_len * r_len * r_len;
        return winding_data.dipole.dot(r) / r3;
    }

    match node {
        BvhNode::Leaf { start, count, .. } => {
            let mut sum = 0.0f32;
            for &idx in &tri_order[*start..*start + *count] {
                let [a, b, c] = positions[idx];
                sum += triangle_solid_angle(point, a, b, c);
            }
            sum
        }
        BvhNode::Interior { left, right, .. } => {
            winding_recursive(left, positions, tri_order, point)
                + winding_recursive(right, positions, tri_order, point)
        }
    }
}

/// Signed solid angle subtended by triangle `(a, b, c)` at point `p`.
///
/// Uses the Van Oosterom & Strackee formula:
/// `Ω = 2·atan2(a'·(b'×c'), |a'||b'||c'| + |a'|(b'·c') + |b'|(a'·c') + |c'|(a'·b'))`
/// where `a' = a - p`, etc.
fn triangle_solid_angle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let a = a - p;
    let b = b - p;
    let c = c - p;
    let la = a.length();
    let lb = b.length();
    let lc = c.length();
    if la < 1e-10 || lb < 1e-10 || lc < 1e-10 {
        return 0.0;
    }
    let na = a / la;
    let nb = b / lb;
    let nc = c / lc;
    let num = na.dot(nb.cross(nc));
    let den = 1.0 + na.dot(nb) + nb.dot(nc) + na.dot(nc);
    2.0 * num.atan2(den)
}
