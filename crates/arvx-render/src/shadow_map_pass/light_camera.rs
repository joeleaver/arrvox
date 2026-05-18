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

use glam::{Mat4, Vec3, Vec4};

use super::types::{LightCameraCsm, LightCameraUniform, CSM_CASCADE_COUNT};

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

/// CSM input parameters. Bundled into a struct to keep the call site
/// readable — `frame_helpers::prepare_shadow_maps` already juggles a
/// dozen values.
#[derive(Debug, Clone, Copy)]
pub struct CsmInputs {
    pub scene_min: [f32; 3],
    pub scene_max: [f32; 3],
    /// Inverse of the camera's `view_proj`. Used to unproject the four
    /// far-plane NDC corners into world-space rays from the eye.
    pub camera_view_proj_inv: Mat4,
    /// World-space camera origin.
    pub camera_position: Vec3,
    /// World-space camera forward (the direction the eye looks toward).
    /// Must already be normalized.
    pub camera_forward: Vec3,
    pub light_dir: [f32; 3],
    pub shadow_map_size: u32,
    pub depth_bias: f32,
    /// CSM near distance in view-space (positive). Frame's camera near
    /// is fine; if it's < 1mm we clamp.
    pub csm_near: f32,
    /// CSM far cap in view-space. Anything beyond this falls back to
    /// fully lit at shade time — matches today's "out-of-bounds → 1.0"
    /// behavior at the single-cascade edge.
    pub csm_max_distance: f32,
    /// PSSM hybrid factor: 0 = uniform (linear) splits, 1 = log splits.
    /// 0.5 is the practical default that keeps the near cascade tight
    /// while still giving the far cascade enough breathing room.
    pub csm_lambda: f32,
    /// User-facing override for cascade-0 extent (m). When valid
    /// (`csm_near < sharp_distance < csm_max_distance`), cascade 0
    /// is hard-set to cover `[csm_near, sharp_distance]` and PSSM
    /// distributes cascades 1..N over `[sharp_distance,
    /// csm_max_distance]`. Pass 0.0 (or any value out of range) to
    /// disable and let PSSM run over the full range as before.
    pub sharp_distance: f32,
}

/// Per-cascade light camera derivation for the CSM path.
///
/// Builds N light cameras (one per cascade) using:
/// 1. PSSM hybrid splits over `[csm_near, csm_max_distance]`.
/// 2. Sphere fit of each cascade's frustum-slice corners (rotation-
///    invariant — the sphere doesn't shift under camera yaw, which is
///    the standard CSM trick to kill texel "swimming" along edges).
/// 3. Texel snap of the sphere center in light-space xy so the ortho
///    bounds quantize to whole-texel multiples (idempotent at fixed
///    yaw → static-camera shadows are bit-stable).
/// 4. Z bounds extended to the scene-AABB's light-space depth so casters
///    above/behind the visible frustum still write into the cascade.
///
/// `cascade_far_view_z[i]` is the view-space far Z of cascade `i`. The
/// shade pass picks the smallest `i` such that the fragment's view_z
/// is `<= cascade_far_view_z[i]`.
pub fn compute_csm_cascades(inputs: CsmInputs) -> LightCameraCsm {
    let n = CSM_CASCADE_COUNT as usize;
    let near = inputs.csm_near.max(1e-3);
    let far = inputs.csm_max_distance.max(near + 1e-3);
    let lambda = inputs.csm_lambda.clamp(0.0, 1.0);

    // ── PSSM splits ───────────────────────────────────────────
    // splits[0] = near, splits[N] = far, intermediates blend log +
    // uniform. When `sharp_distance` is in `(near, far)`, cascade 0
    // is hard-set to `[near, sharp_distance]` and PSSM splits the
    // remaining `[sharp_distance, far]` range across cascades 1..N
    // (so the user-facing "Sharp Distance" knob directly controls
    // where the highest-detail tier ends, not the abstract λ).
    let mut splits = [0.0_f32; CSM_CASCADE_COUNT as usize + 1];
    splits[0] = near;
    splits[n] = far;
    let use_sharp = inputs.sharp_distance > near && inputs.sharp_distance < far;
    let pssm_near = if use_sharp {
        splits[1] = inputs.sharp_distance;
        inputs.sharp_distance
    } else {
        near
    };
    let pssm_start = if use_sharp { 2 } else { 1 };
    let pssm_n = (n - pssm_start + 1) as f32; // sub-intervals over [pssm_near, far]
    for i in pssm_start..n {
        let f = (i - pssm_start + 1) as f32 / pssm_n;
        let z_log = pssm_near * (far / pssm_near).powf(f);
        let z_uniform = pssm_near + (far - pssm_near) * f;
        splits[i] = lambda * z_log + (1.0 - lambda) * z_uniform;
    }

    // ── 4 world-space rays from camera through NDC far corners ──
    // Each ray r is the vector from camera_position to the
    // unprojected NDC (±1, ±1, 1) point. Scaling r by t produces the
    // world-space corner at view-space depth t * dot(r, fwd).
    let fwd = inputs.camera_forward.normalize_or_zero();
    let mut rays: [Vec3; 4] = [Vec3::ZERO; 4];
    for c in 0..4 {
        let nx = if c & 1 != 0 { 1.0 } else { -1.0 };
        let ny = if c & 2 != 0 { 1.0 } else { -1.0 };
        let world = inputs.camera_view_proj_inv * Vec4::new(nx, ny, 1.0, 1.0);
        let world_pos = world.truncate() / world.w;
        rays[c] = world_pos - inputs.camera_position;
    }

    // ── Light basis ───────────────────────────────────────────
    let l = Vec3::from_array(inputs.light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0)
    } else {
        l
    };
    let world_up = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();

    // ── Scene-AABB light-space Z extent ───────────────────────
    // Used to set per-cascade ortho near/far so a tower above the
    // visible frustum still writes its shadow.
    let scene_min = inputs.scene_min;
    let scene_max = inputs.scene_max;
    let mut scene_min_lz = f32::INFINITY;
    let mut scene_max_lz = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lz = l.dot(corner);
        if lz < scene_min_lz { scene_min_lz = lz; }
        if lz > scene_max_lz { scene_max_lz = lz; }
    }
    let scene_z_extent = (scene_max_lz - scene_min_lz).max(1.0);

    // ── Per-cascade fit ───────────────────────────────────────
    let zero_slice = LightCameraUniform {
        view_proj: [[0.0; 4]; 4],
        view_proj_inv: [[0.0; 4]; 4],
        light_dir: [0.0; 3],
        depth_bias: 0.0,
        inv_shadow_map_size: [0.0; 2],
        shadow_map_size: [0; 2],
    };
    let mut cascades = [zero_slice; CSM_CASCADE_COUNT as usize];
    let mut cascade_far_view_z = [0.0_f32; CSM_CASCADE_COUNT as usize];

    for i in 0..n {
        let z_near_i = splits[i];
        let z_far_i = splits[i + 1];
        cascade_far_view_z[i] = z_far_i;

        // 8 world-space corners of slice [z_near_i, z_far_i].
        let mut corners = [Vec3::ZERO; 8];
        for c in 0..4 {
            let r = rays[c];
            let camera_far = r.dot(fwd).max(1e-3);
            let t_near = z_near_i / camera_far;
            let t_far = z_far_i / camera_far;
            corners[c] = inputs.camera_position + r * t_near;
            corners[c + 4] = inputs.camera_position + r * t_far;
        }

        // Sphere fit: centroid + max-distance radius. Centroid + max
        // distance is rotation-invariant when *all* 8 corners
        // contribute, which is what we want — yawing the camera
        // permutes the corner labels but the bounding sphere stays
        // the same.
        let mut centroid = Vec3::ZERO;
        for &c in &corners { centroid += c; }
        centroid /= 8.0;
        let mut radius: f32 = 0.0;
        for &c in &corners {
            radius = radius.max((c - centroid).length());
        }
        // Tiny pad to avoid edge clipping when the sphere fits exactly.
        radius = (radius * 1.005).max(1e-3);

        // Project centroid into light-space (right, up, l) basis.
        let cx = centroid.dot(right);
        let cy = centroid.dot(up);
        let cl = centroid.dot(l);

        // Snap centroid xy to a whole-texel multiple in light-space.
        // texel_size = 2*radius / shadow_map_size — the world-space
        // size of one shadow-map texel at this cascade.
        let texel = (2.0 * radius) / (inputs.shadow_map_size as f32);
        let snap_cx = (cx / texel).floor() * texel;
        let snap_cy = (cy / texel).floor() * texel;
        let snapped_center = right * snap_cx + up * snap_cy + l * cl;

        // Eye sits far behind the snapped center along -l, with enough
        // margin to clear the scene's full light-space z extent.
        let eye_back = (scene_z_extent * 1.5).max(2.0 * radius);
        let eye = snapped_center - l * eye_back;

        // Light view: look_to_rh with forward = l.
        let view = Mat4::look_to_rh(eye, l, up);

        // Compute view-space z range enclosing both the slice's 8
        // corners AND the scene AABB. orthographic_rh near/far take
        // positive distance along the forward direction; for a
        // look_to_rh view that's `-v.z`.
        let mut zmin: f32 = f32::INFINITY;
        let mut zmax: f32 = f32::NEG_INFINITY;
        for &corner in &corners {
            let v = view.transform_point3(corner);
            let d = -v.z;
            if d < zmin { zmin = d; }
            if d > zmax { zmax = d; }
        }
        for c in 0..8u32 {
            let corner = Vec3::new(
                if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
                if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
                if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
            );
            let v = view.transform_point3(corner);
            let d = -v.z;
            if d < zmin { zmin = d; }
            if d > zmax { zmax = d; }
        }
        let near_p = zmin.max(0.01);
        let far_p = zmax.max(near_p + 0.01);

        // Ortho extents = ±radius around the snapped center (which
        // by construction sits at view-space (0, 0, -eye_back)).
        let proj = Mat4::orthographic_rh(
            -radius, radius,
            -radius, radius,
            near_p, far_p,
        );
        let view_proj = proj * view;
        let view_proj_inv = view_proj.inverse();

        cascades[i] = LightCameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            view_proj_inv: view_proj_inv.to_cols_array_2d(),
            light_dir: l.to_array(),
            depth_bias: inputs.depth_bias,
            inv_shadow_map_size: [
                1.0 / inputs.shadow_map_size as f32,
                1.0 / inputs.shadow_map_size as f32,
            ],
            shadow_map_size: [inputs.shadow_map_size, inputs.shadow_map_size],
        };
    }

    LightCameraCsm {
        cascades,
        cascade_far_view_z,
        cascade_count: CSM_CASCADE_COUNT,
        _pad: [0; 3],
    }
}
