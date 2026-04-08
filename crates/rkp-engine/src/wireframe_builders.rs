//! Gizmo wireframe geometry builders.
//!
//! Produces `Vec<LineVertex>` for translate/rotate/scale gizmos.
//! Pure data — no GPU, no state.

use glam::Vec3;
use rkf_render::wireframe::{aabb_wireframe, circle_wireframe, LineVertex};

use crate::gizmo::GizmoAxis;

// Axis colors: R = X, G = Y, B = Z.
const GIZMO_X_COLOR: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
const GIZMO_Y_COLOR: [f32; 4] = [0.2, 1.0, 0.2, 1.0];
const GIZMO_Z_COLOR: [f32; 4] = [0.3, 0.3, 1.0, 1.0];

const GIZMO_X_HOVER: [f32; 4] = [1.0, 0.4, 0.4, 1.0];
const GIZMO_Y_HOVER: [f32; 4] = [0.4, 1.0, 0.4, 1.0];
const GIZMO_Z_HOVER: [f32; 4] = [0.4, 0.4, 1.0, 1.0];

const GIZMO_X_DIM: [f32; 4] = [0.4, 0.08, 0.08, 0.4];
const GIZMO_Y_DIM: [f32; 4] = [0.08, 0.4, 0.08, 0.4];
const GIZMO_Z_DIM: [f32; 4] = [0.12, 0.12, 0.4, 0.4];

fn gizmo_axis_color(axis_idx: usize, hovered: GizmoAxis) -> [f32; 4] {
    let normal = [GIZMO_X_COLOR, GIZMO_Y_COLOR, GIZMO_Z_COLOR];
    let bright = [GIZMO_X_HOVER, GIZMO_Y_HOVER, GIZMO_Z_HOVER];
    let dim = [GIZMO_X_DIM, GIZMO_Y_DIM, GIZMO_Z_DIM];

    let hovered_idx = match hovered {
        GizmoAxis::X => Some(0),
        GizmoAxis::Y => Some(1),
        GizmoAxis::Z => Some(2),
        GizmoAxis::None => None,
        _ => None,
    };

    match hovered_idx {
        None => normal[axis_idx],
        Some(hi) if hi == axis_idx => bright[axis_idx],
        Some(_) => dim[axis_idx],
    }
}

/// Build a translate gizmo: 3 axis arrows from `center` with length `size`.
pub fn translate_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let mut verts = Vec::new();
    let head_len = size * 0.2;
    let head_radius = size * 0.06;
    let to_cam = (cam_pos - center).normalize_or_zero();

    for (axis_idx, axis_dir) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        let tip = center + axis_dir * size;
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        if is_hovered {
            let perp = axis_dir.cross(to_cam).normalize_or_zero();
            let offset = size * 0.004;
            for sign in [-1.0f32, 1.0] {
                let off = perp * (offset * sign);
                verts.push(LineVertex { position: (center + off).to_array(), color });
                verts.push(LineVertex { position: (tip + off).to_array(), color });
            }
        }

        let tangent = if axis_dir.dot(Vec3::Y).abs() < 0.99 {
            axis_dir.cross(Vec3::Y).normalize()
        } else {
            axis_dir.cross(Vec3::X).normalize()
        };
        let bitangent = axis_dir.cross(tangent);
        let base = tip - axis_dir * head_len;

        let step = std::f32::consts::TAU / 4.0;
        for i in 0..4 {
            let a = step * i as f32;
            let p = base + (tangent * a.cos() + bitangent * a.sin()) * head_radius;
            verts.push(LineVertex { position: tip.to_array(), color });
            verts.push(LineVertex { position: p.to_array(), color });
        }
    }
    verts
}

/// Build a rotate gizmo: 3 axis rings at `center` with radius `size`.
pub fn rotate_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let segs = 48;
    let to_cam = (cam_pos - center).normalize_or_zero();
    let offset_mag = size * 0.004;

    let mut verts = Vec::new();
    for (axis_idx, normal) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        verts.extend(circle_wireframe(center, normal, size, color, segs));

        if is_hovered {
            let perp = normal.cross(to_cam).normalize_or_zero();
            for sign in [-1.0f32, 1.0] {
                let off_center = center + perp * (offset_mag * sign);
                verts.extend(circle_wireframe(off_center, normal, size, color, segs));
            }
        }
    }
    verts
}

/// Build a scale gizmo: 3 axis lines with small cubes at the ends.
pub fn scale_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let cube_half = size * 0.06;
    let to_cam = (cam_pos - center).normalize_or_zero();
    let mut verts = Vec::new();

    for (axis_idx, axis_dir) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        let tip = center + axis_dir * size;
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        if is_hovered {
            let perp = axis_dir.cross(to_cam).normalize_or_zero();
            let offset = size * 0.004;
            for sign in [-1.0f32, 1.0] {
                let off = perp * (offset * sign);
                verts.push(LineVertex { position: (center + off).to_array(), color });
                verts.push(LineVertex { position: (tip + off).to_array(), color });
            }
        }

        let min = tip - Vec3::splat(cube_half);
        let max = tip + Vec3::splat(cube_half);
        verts.extend(aabb_wireframe(min, max, color));
    }

    let center_color = if hovered == GizmoAxis::View {
        [1.0, 1.0, 1.0, 1.0]
    } else if hovered != GizmoAxis::None {
        [0.4, 0.4, 0.4, 0.4]
    } else {
        [0.9, 0.9, 0.9, 1.0]
    };
    let cc = Vec3::splat(cube_half * 1.2);
    verts.extend(aabb_wireframe(center - cc, center + cc, center_color));
    verts
}
