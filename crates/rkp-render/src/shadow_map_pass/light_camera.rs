//! Light camera derivation — pure CPU math that produces the
//! `LightCameraUniform` from the scene AABB or visible-caster bounds.
//!
//! Two flavors:
//! - [`compute_light_camera`] — V1 scene-AABB-fit derivation
//!   (look_to + scene-AABB-projected ortho).
//! - [`compute_light_camera_frustum_fit`] — V3 frustum-fit
//!   derivation that intersects the camera frustum (capped at
//!   `SHADOW_FAR_DISTANCE`) with the scene to produce a tighter
//!   ortho.

use glam::{Mat4, Vec3};

use super::types::LightCameraUniform;

/// Derive an orthographic light camera. See V1 commit 3d862b0 for
/// the derivation rationale (look_to + scene-AABB-fit).
pub fn compute_light_camera(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    light_dir: [f32; 3],
    shadow_map_size: u32,
    depth_bias: f32,
) -> LightCameraUniform {
    let l = Vec3::from_array(light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0)
    } else {
        l
    };
    let world_up = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();
    let scene_center = Vec3::new(
        0.5 * (scene_min[0] + scene_max[0]),
        0.5 * (scene_min[1] + scene_max[1]),
        0.5 * (scene_min[2] + scene_max[2]),
    );
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lz = l.dot(corner);
        if lz < min_z { min_z = lz; }
        if lz > max_z { max_z = lz; }
    }
    let z_extent = (max_z - min_z).max(1e-3);
    let eye = scene_center - l * (z_extent * 1.5);
    let view = Mat4::look_to_rh(eye, l, up);
    let mut vmin = Vec3::splat(f32::INFINITY);
    let mut vmax = Vec3::splat(f32::NEG_INFINITY);
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let v = view.transform_point3(corner);
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }
    let near = -vmax.z;
    let far = -vmin.z;
    let proj = Mat4::orthographic_rh(vmin.x, vmax.x, vmin.y, vmax.y, near, far);
    let view_proj = proj * view;
    let view_proj_inv = view_proj.inverse();
    LightCameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        view_proj_inv: view_proj_inv.to_cols_array_2d(),
        light_dir: l.to_array(),
        depth_bias,
        inv_shadow_map_size: [
            1.0 / shadow_map_size as f32,
            1.0 / shadow_map_size as f32,
        ],
        shadow_map_size: [shadow_map_size, shadow_map_size],
    }
}

/// Per-VR frustum-fit light camera. Same shape as
/// `compute_light_camera` but the orthographic xy bounds are
/// fitted to the camera's *visible* frustum rather than the whole
/// scene AABB. The z range is extended to encompass the whole
/// scene's light-space depth so casters outside the visible
/// frustum (e.g., a tower above the camera's forward cone) still
/// reach the shadow map.
///
/// The camera's far plane is clamped at `shadow_far_dist` from
/// the camera position — the camera's actual far plane can be
/// kilometers, which would dilute texel density in the foreground.
/// CSM is the proper fix for variable depth ranges; this single-
/// cascade approach trades far-field shadow quality for near-field
/// sharpness.
///
/// Inputs:
/// * `scene_min` / `scene_max` — world-space bounds of all shadow
///   casters. Used only for z-range extension; not for xy.
/// * `camera_view_proj_inv` — inverse of the camera's view-proj
///   matrix. Used to unproject the 8 NDC frustum corners into
///   world space.
/// * `camera_position` — world-space camera origin. Used to clamp
///   the far corners' distance to `shadow_far_dist`.
/// * `light_dir`, `shadow_map_size`, `depth_bias` — same as the
///   scene-fit variant.
/// * `shadow_far_dist` — camera-relative far cap for the fit.
pub fn compute_light_camera_frustum_fit(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    camera_view_proj_inv: Mat4,
    camera_position: Vec3,
    light_dir: [f32; 3],
    shadow_map_size: u32,
    depth_bias: f32,
    shadow_far_dist: f32,
) -> LightCameraUniform {
    let l = Vec3::from_array(light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0)
    } else {
        l
    };
    let world_up = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();

    // 8 frustum corners in NDC: (±1, ±1, {0, 1}). z=0 = near
    // plane, z=1 = far plane (wgpu convention).
    let mut frustum_world: [Vec3; 8] = [Vec3::ZERO; 8];
    for c in 0..8u32 {
        let ndc = Vec3::new(
            if (c & 1) != 0 { 1.0 } else { -1.0 },
            if (c & 2) != 0 { 1.0 } else { -1.0 },
            if (c & 4) != 0 { 1.0 } else { 0.0 },
        );
        let world = camera_view_proj_inv * ndc.extend(1.0);
        let world_pos = world.truncate() / world.w;
        // Far corners: clamp distance from camera. The camera's
        // far plane can be 10 km+; clamp keeps per-meter density
        // high in the foreground. Near corners pass through.
        if (c & 4) != 0 {
            let dir = world_pos - camera_position;
            let dist = dir.length();
            if dist > shadow_far_dist {
                frustum_world[c as usize] =
                    camera_position + dir / dist * shadow_far_dist;
            } else {
                frustum_world[c as usize] = world_pos;
            }
        } else {
            frustum_world[c as usize] = world_pos;
        }
    }

    // Set the eye well behind the scene along -L. Distance is
    // chosen so every potential caster sits in front of the
    // ortho's near plane.
    let scene_center = Vec3::new(
        0.5 * (scene_min[0] + scene_max[0]),
        0.5 * (scene_min[1] + scene_max[1]),
        0.5 * (scene_min[2] + scene_max[2]),
    );
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lz = l.dot(corner);
        if lz < min_z { min_z = lz; }
        if lz > max_z { max_z = lz; }
    }
    let z_extent = (max_z - min_z).max(1e-3);
    let eye = scene_center - l * (z_extent * 1.5);
    let view = Mat4::look_to_rh(eye, l, up);

    // Project camera frustum AND scene AABB into light view-space.
    let mut frustum_vmin = Vec3::splat(f32::INFINITY);
    let mut frustum_vmax = Vec3::splat(f32::NEG_INFINITY);
    for &corner in &frustum_world {
        let v = view.transform_point3(corner);
        frustum_vmin = frustum_vmin.min(v);
        frustum_vmax = frustum_vmax.max(v);
    }
    let mut scene_vmin = Vec3::splat(f32::INFINITY);
    let mut scene_vmax = Vec3::splat(f32::NEG_INFINITY);
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let v = view.transform_point3(corner);
        scene_vmin = scene_vmin.min(v);
        scene_vmax = scene_vmax.max(v);
    }

    // INTERSECT xy: shadow map should only cover the visible
    // region that contains scene geometry. The camera frustum's
    // far plane can project to a huge area (200 m far cap × ~90°
    // FOV ~= 400 m × 200 m in light xy), but if the scene AABB
    // is small (e.g., 10 m × 10 m), the frustum bounds dilute
    // texel density.
    //
    // Z bounds: full scene span (any caster between the visible
    // surfaces and the light belongs in the shadow map).
    let xy_min = Vec3::new(
        frustum_vmin.x.max(scene_vmin.x),
        frustum_vmin.y.max(scene_vmin.y),
        0.0,
    );
    let xy_max = Vec3::new(
        frustum_vmax.x.min(scene_vmax.x),
        frustum_vmax.y.min(scene_vmax.y),
        0.0,
    );
    // Empty intersection (camera looking away from scene): fall
    // back to scene-fit so the shadow map still has valid bounds.
    let (final_x_min, final_x_max, final_y_min, final_y_max) =
        if xy_min.x >= xy_max.x || xy_min.y >= xy_max.y {
            (scene_vmin.x, scene_vmax.x, scene_vmin.y, scene_vmax.y)
        } else {
            (xy_min.x, xy_max.x, xy_min.y, xy_max.y)
        };

    let near = -scene_vmax.z;
    let far = -scene_vmin.z;
    let proj = Mat4::orthographic_rh(
        final_x_min, final_x_max, final_y_min, final_y_max, near, far,
    );
    let view_proj = proj * view;
    let view_proj_inv = view_proj.inverse();

    LightCameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        view_proj_inv: view_proj_inv.to_cols_array_2d(),
        light_dir: l.to_array(),
        depth_bias,
        inv_shadow_map_size: [
            1.0 / shadow_map_size as f32,
            1.0 / shadow_map_size as f32,
        ],
        shadow_map_size: [shadow_map_size, shadow_map_size],
    }
}
