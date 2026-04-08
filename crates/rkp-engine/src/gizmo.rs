//! Transform gizmo math for the RKIPatch editor.
//!
//! Provides the geometric calculations for translate/rotate/scale gizmos:
//! ray-axis closest point, ray-plane intersection, axis picking, and
//! constrained delta computation. This is pure math — no rendering or input handling.

#![allow(dead_code)]

use glam::{Quat, Vec3};

/// Which transform operation the gizmo performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which axis or plane the gizmo is constrained to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    XY,
    XZ,
    YZ,
    /// Screen-space (view-aligned) operation.
    View,
    None,
}

impl GizmoAxis {
    /// Return the unit direction vector for single-axis constraints.
    /// Returns `Vec3::ZERO` for plane/view/none axes.
    pub fn direction(&self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            _ => Vec3::ZERO,
        }
    }

    /// Return the plane normal for plane constraints.
    /// For single axes, returns the axis direction (used as rotation plane normal).
    /// Returns `Vec3::ZERO` for View/None.
    pub fn plane_normal(&self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            GizmoAxis::XY => Vec3::Z,
            GizmoAxis::XZ => Vec3::Y,
            GizmoAxis::YZ => Vec3::X,
            _ => Vec3::ZERO,
        }
    }
}

/// State of an in-progress gizmo drag operation.
#[derive(Debug, Clone)]
pub struct GizmoState {
    /// Current gizmo mode.
    pub mode: GizmoMode,
    /// Which axis/plane is active.
    pub active_axis: GizmoAxis,
    /// Whether a drag is currently in progress.
    pub dragging: bool,
    /// World-space point where the drag started.
    pub drag_start: Vec3,
    /// Current world-space drag point.
    pub drag_current: Vec3,
    /// World-space position of the object when the drag began.
    pub initial_position: Vec3,
    /// World-space rotation of the object when the drag began.
    pub initial_rotation: Quat,
    /// World-space scale of the object when the drag began.
    pub initial_scale: Vec3,
    /// Local-space position of the object when the drag began (for undo).
    pub initial_local_position: Vec3,
    /// Local-space rotation of the object when the drag began (for undo).
    pub initial_local_rotation: Quat,
    /// Local-space scale of the object when the drag began (for undo).
    pub initial_local_scale: Vec3,
    /// Parent's world-space position (identity if no parent).
    pub parent_world_position: Vec3,
    /// Parent's world-space rotation (identity if no parent).
    pub parent_world_rotation: Quat,
    /// Parent's world-space scale (identity if no parent).
    pub parent_world_scale: Vec3,
    /// View-plane normal captured at drag start (for uniform scale center handle).
    pub drag_view_normal: Vec3,
    /// Which axis the mouse is currently hovering over (for visual feedback).
    pub hovered_axis: GizmoAxis,
    /// World-space pivot (gizmo center) captured at drag start.
    /// Used for rotation (rotate around this point) and scale (scale relative to this point).
    pub pivot: Vec3,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self::new()
    }
}

impl GizmoState {
    /// Create a new idle gizmo state.
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            active_axis: GizmoAxis::None,
            dragging: false,
            drag_start: Vec3::ZERO,
            drag_current: Vec3::ZERO,
            initial_position: Vec3::ZERO,
            initial_rotation: Quat::IDENTITY,
            initial_scale: Vec3::ONE,
            initial_local_position: Vec3::ZERO,
            initial_local_rotation: Quat::IDENTITY,
            initial_local_scale: Vec3::ONE,
            parent_world_position: Vec3::ZERO,
            parent_world_rotation: Quat::IDENTITY,
            parent_world_scale: Vec3::ONE,
            drag_view_normal: Vec3::Z,
            hovered_axis: GizmoAxis::None,
            pivot: Vec3::ZERO,
        }
    }

    /// Begin a drag operation.
    pub fn begin_drag(
        &mut self,
        axis: GizmoAxis,
        start_point: Vec3,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        view_normal: Vec3,
    ) {
        self.active_axis = axis;
        self.dragging = true;
        self.drag_start = start_point;
        self.drag_current = start_point;
        self.initial_position = position;
        self.initial_rotation = rotation;
        self.initial_scale = scale;
        self.drag_view_normal = view_normal;
    }

    /// End the current drag operation.
    pub fn end_drag(&mut self) {
        self.dragging = false;
        self.active_axis = GizmoAxis::None;
    }
}

/// Find the parameter `t` along an axis line that is closest to a ray.
///
/// Given a ray (origin + direction) and an axis line (origin + direction),
/// returns the parameter `t` such that `axis_origin + axis_dir * t` is the
/// closest point on the axis to the ray.
///
/// Returns 0.0 if the ray and axis are parallel (degenerate case).
pub fn ray_axis_closest_point(
    ray_origin: Vec3,
    ray_dir: Vec3,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> f32 {
    // Closest approach between two lines:
    // Line 1: P = ray_origin + ray_dir * s
    // Line 2: Q = axis_origin + axis_dir * t
    // We want to find t that minimizes |P - Q|.
    //
    // Using the standard formula for closest points on two lines:
    let w0 = ray_origin - axis_origin;
    let a = ray_dir.dot(ray_dir); // always >= 0
    let b = ray_dir.dot(axis_dir);
    let c = axis_dir.dot(axis_dir); // always >= 0
    let d = ray_dir.dot(w0);
    let e = axis_dir.dot(w0);

    let denom = a * c - b * b;

    // If denom is ~0, the lines are parallel.
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    (a * e - b * d) / denom
}

/// Intersect a ray with a plane.
///
/// Returns the intersection point, or `None` if the ray is parallel to the plane
/// or the intersection is behind the ray origin.
pub fn project_to_plane(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = ray_dir.dot(plane_normal);

    // Ray is parallel to or nearly parallel to the plane.
    if denom.abs() < 1e-7 {
        return None;
    }

    let t = (plane_origin - ray_origin).dot(plane_normal) / denom;

    // Intersection is behind the ray origin.
    if t < 0.0 {
        return None;
    }

    Some(ray_origin + ray_dir * t)
}

/// Hit-test the gizmo axes and return the closest axis within a threshold.
///
/// Tests the three primary axes (X, Y, Z) emanating from `gizmo_center`,
/// each with length `gizmo_size`. Returns the axis whose line is closest
/// to the pick ray, provided the distance is within a screen-space-like threshold.
///
/// In `Rotate` mode, tests against rings (circles) instead of lines.
/// The threshold scales with camera distance for consistent screen-space feel.
pub fn pick_gizmo_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_center: Vec3,
    gizmo_size: f32,
) -> GizmoAxis {
    pick_gizmo_axis_for_mode(ray_origin, ray_dir, gizmo_center, gizmo_size, GizmoMode::Translate)
}

/// Mode-aware gizmo axis picking.
///
/// For `Translate`/`Scale`, tests against axis lines.
/// For `Rotate`, tests against rings (circles) in the plane perpendicular to each axis.
pub fn pick_gizmo_axis_for_mode(
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_center: Vec3,
    gizmo_size: f32,
    mode: GizmoMode,
) -> GizmoAxis {
    let ray_dir = ray_dir.normalize();

    // Camera-distance-proportional threshold for consistent screen-space feel.
    let cam_dist = (gizmo_center - ray_origin).length().max(0.1);
    let threshold = cam_dist * 0.04; // ~4% of camera distance ≈ generous screen-space band

    let axes = [
        (GizmoAxis::X, Vec3::X),
        (GizmoAxis::Y, Vec3::Y),
        (GizmoAxis::Z, Vec3::Z),
    ];

    let mut best_axis = GizmoAxis::None;
    let mut best_dist = f32::MAX;

    // Scale mode: test center box first (uniform scaling)
    if mode == GizmoMode::Scale {
        let center_half = gizmo_size * 0.12; // slightly larger than visual for easy picking
        let min = gizmo_center - Vec3::splat(center_half);
        let max = gizmo_center + Vec3::splat(center_half);
        if ray_aabb_hit(ray_origin, ray_dir, min, max) {
            // Center box takes priority at distance 0
            best_axis = GizmoAxis::View; // View = uniform/center
            best_dist = 0.0;
        }
    }

    for (axis_id, axis_dir) in &axes {
        let dist = match mode {
            GizmoMode::Rotate => {
                // Test against a ring: intersect ray with the plane, check distance to circle
                ring_pick_distance(ray_origin, ray_dir, gizmo_center, *axis_dir, gizmo_size)
            }
            _ => {
                // Test against axis line segment [0, gizmo_size]
                line_pick_distance(ray_origin, ray_dir, gizmo_center, *axis_dir, gizmo_size)
            }
        };

        if let Some(d) = dist {
            if d < threshold && d < best_dist {
                best_dist = d;
                best_axis = *axis_id;
            }
        }
    }

    best_axis
}

/// Simple ray-AABB intersection test (slab method). Returns true if the ray hits the box.
fn ray_aabb_hit(ray_origin: Vec3, ray_dir: Vec3, min: Vec3, max: Vec3) -> bool {
    let inv_dir = Vec3::new(
        if ray_dir.x.abs() > 1e-8 { 1.0 / ray_dir.x } else { f32::MAX.copysign(ray_dir.x) },
        if ray_dir.y.abs() > 1e-8 { 1.0 / ray_dir.y } else { f32::MAX.copysign(ray_dir.y) },
        if ray_dir.z.abs() > 1e-8 { 1.0 / ray_dir.z } else { f32::MAX.copysign(ray_dir.z) },
    );
    let t1 = (min - ray_origin) * inv_dir;
    let t2 = (max - ray_origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    t_exit >= t_enter.max(0.0)
}

/// Distance from a ray to an axis line segment [0, gizmo_size].
/// Returns `None` if the closest point is behind the camera.
fn line_pick_distance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    center: Vec3,
    axis_dir: Vec3,
    gizmo_size: f32,
) -> Option<f32> {
    let t = ray_axis_closest_point(ray_origin, ray_dir, center, axis_dir);
    let t_clamped = t.clamp(0.0, gizmo_size);
    let axis_point = center + axis_dir * t_clamped;

    let ray_t = (axis_point - ray_origin).dot(ray_dir);
    if ray_t < 0.0 {
        return None; // Behind camera
    }
    let ray_point = ray_origin + ray_dir * ray_t;
    Some((ray_point - axis_point).length())
}

/// Distance from a ray to a ring (circle) of radius `gizmo_size` in the plane
/// perpendicular to `axis_dir` at `center`.
/// Returns `None` if the ray doesn't intersect near the plane or is behind camera.
fn ring_pick_distance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    center: Vec3,
    axis_dir: Vec3,
    radius: f32,
) -> Option<f32> {
    // Intersect ray with the ring's plane
    let denom = ray_dir.dot(axis_dir);

    // If ray is nearly parallel to the plane, use closest-approach fallback
    if denom.abs() < 1e-4 {
        // Project ray onto plane via closest point, then check ring distance
        let t = ray_axis_closest_point(ray_origin, ray_dir, center, axis_dir);
        let closest_on_axis = center + axis_dir * t;
        let ray_t = (closest_on_axis - ray_origin).dot(ray_dir);
        if ray_t < 0.0 {
            return None;
        }
        let ray_point = ray_origin + ray_dir * ray_t;
        let to_point = ray_point - center;
        let in_plane = to_point - axis_dir * to_point.dot(axis_dir);
        let dist_from_ring = (in_plane.length() - radius).abs();
        return Some(dist_from_ring);
    }

    let t = (center - ray_origin).dot(axis_dir) / denom;
    if t < 0.0 {
        return None; // Behind camera
    }

    let hit = ray_origin + ray_dir * t;
    let offset = hit - center;
    // Project onto the plane (should already be in-plane, but be safe)
    let in_plane = offset - axis_dir * offset.dot(axis_dir);
    let dist_from_center = in_plane.length();

    // Distance from hit point to the ring circumference
    Some((dist_from_center - radius).abs())
}

/// Compute the translation delta constrained to the active axis.
///
/// Projects the current ray onto the active axis or plane and returns the
/// world-space translation offset from the drag start.
pub fn compute_translate_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Vec3 {
    let ray_dir = ray_dir.normalize();

    match state.active_axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let axis_dir = state.active_axis.direction();
            let t_current =
                ray_axis_closest_point(ray_origin, ray_dir, state.initial_position, axis_dir);
            let t_start = ray_axis_closest_point(
                ray_origin,
                ray_dir,
                state.initial_position,
                axis_dir,
            );
            // We need to compute the delta relative to drag_start.
            // Project drag_start onto the axis to get reference t.
            let start_t = (state.drag_start - state.initial_position).dot(axis_dir);
            let _ = t_start; // The current ray projection gives us the new t.
            axis_dir * (t_current - start_t)
        }
        GizmoAxis::XY | GizmoAxis::XZ | GizmoAxis::YZ => {
            let normal = state.active_axis.plane_normal();
            let current = project_to_plane(ray_origin, ray_dir, state.initial_position, normal);
            match current {
                Some(point) => point - state.drag_start,
                None => Vec3::ZERO,
            }
        }
        GizmoAxis::View => {
            // View-plane translation: project onto a plane facing the camera.
            let view_normal = -ray_dir;
            let current =
                project_to_plane(ray_origin, ray_dir, state.initial_position, view_normal);
            match current {
                Some(point) => point - state.drag_start,
                None => Vec3::ZERO,
            }
        }
        GizmoAxis::None => Vec3::ZERO,
    }
}

/// Compute the rotation delta around the active axis.
///
/// Projects the ray onto the plane perpendicular to the active axis at the
/// gizmo center, then computes the angle between drag_start and current point.
pub fn compute_rotate_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_center: Vec3,
) -> Quat {
    let axis = match state.active_axis {
        GizmoAxis::X => Vec3::X,
        GizmoAxis::Y => Vec3::Y,
        GizmoAxis::Z => Vec3::Z,
        _ => return Quat::IDENTITY,
    };

    let ray_dir = ray_dir.normalize();

    // Project both drag_start and current ray onto the rotation plane.
    let Some(current_point) = project_to_plane(ray_origin, ray_dir, gizmo_center, axis) else {
        return Quat::IDENTITY;
    };

    // Vectors from center to start and current, projected onto the plane.
    let start_vec = (state.drag_start - gizmo_center).normalize();
    let current_vec = (current_point - gizmo_center).normalize();

    if start_vec.length_squared() < 1e-6 || current_vec.length_squared() < 1e-6 {
        return Quat::IDENTITY;
    }

    // Compute angle using atan2 for proper sign.
    let dot = start_vec.dot(current_vec).clamp(-1.0, 1.0);
    let cross = start_vec.cross(current_vec);
    let angle = cross.dot(axis).atan2(dot);

    Quat::from_axis_angle(axis, angle)
}

/// The result of applying a completed gizmo drag to an object's transform.
///
/// Returned by helpers like `compute_translate_delta`/`compute_rotate_delta`/
/// `compute_scale_delta` and consumed by `apply_to_v2_object`.
#[derive(Debug, Clone)]
pub struct GizmoResult {
    /// New world-space position of the object.
    pub position: Vec3,
    /// New world-space rotation of the object.
    pub rotation: Quat,
    /// New per-axis scale of the object.
    pub scale: Vec3,
}

/// Compute the scale delta from a drag operation.
///
/// Returns a `Vec3` scale factor. Per-axis handles (X/Y/Z) scale only
/// the corresponding axis. The center handle (`View`) scales uniformly.
pub fn compute_scale_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Vec3 {
    if state.active_axis == GizmoAxis::None {
        return Vec3::ONE;
    }

    let ray_dir = ray_dir.normalize();

    // Determine drag axis direction and which components to scale.
    let per_axis = match state.active_axis {
        GizmoAxis::X => Some(0),
        GizmoAxis::Y => Some(1),
        GizmoAxis::Z => Some(2),
        _ => None, // View or fallback = uniform
    };

    // Unity-style ratio-based scaling:
    // factor = current_distance_from_center / initial_distance_from_center
    // This gives the same visual drag rate regardless of current scale.
    let factor = if per_axis.is_some() {
        // Per-axis: project onto the world axis line
        let drag_axis = match state.active_axis {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            _ => unreachable!(),
        };
        let initial_dist = (state.drag_start - state.initial_position).dot(drag_axis);
        let current_dist =
            ray_axis_closest_point(ray_origin, ray_dir, state.initial_position, drag_axis);

        if initial_dist.abs() < 0.001 {
            let delta = current_dist - initial_dist;
            (1.0 + delta * 0.5).clamp(0.01, 100.0)
        } else {
            (current_dist / initial_dist).clamp(0.01, 100.0)
        }
    } else {
        // Uniform scale: intersect ray with the view plane through center,
        // measure distance from center in that plane.
        let normal = state.drag_view_normal;
        let initial_dist = (state.drag_start - state.initial_position).length();
        let current_point =
            project_to_plane(ray_origin, ray_dir, state.initial_position, normal)
                .unwrap_or(state.drag_start);
        let current_dist = (current_point - state.initial_position).length();

        if initial_dist < 0.001 {
            let delta = current_dist - initial_dist;
            (1.0 + delta * 0.5).clamp(0.01, 100.0)
        } else {
            (current_dist / initial_dist).clamp(0.01, 100.0)
        }
    };

    match per_axis {
        Some(axis_idx) => {
            // Per-axis scale: only affect the dragged axis
            let mut scale = Vec3::ONE;
            scale[axis_idx] = factor;
            scale
        }
        None => {
            // Uniform scale
            Vec3::splat(factor)
        }
    }
}

