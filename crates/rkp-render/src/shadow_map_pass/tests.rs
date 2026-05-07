use super::*;
use glam::{Mat4, Vec3, Vec4};

#[test]
fn light_camera_uniform_size_is_160() {
    assert_eq!(std::mem::size_of::<LightCameraUniform>(), 160);
}

#[test]
fn light_camera_csm_size_is_672() {
    assert_eq!(std::mem::size_of::<LightCameraCsm>(), 672);
}

#[test]
fn light_camera_csm_offsets_are_correct() {
    use std::mem::offset_of;
    assert_eq!(offset_of!(LightCameraCsm, cascades), 0);
    assert_eq!(offset_of!(LightCameraCsm, cascade_far_view_z), 640);
    assert_eq!(offset_of!(LightCameraCsm, cascade_count), 656);
}

#[test]
fn single_cascade_helper_returns_count_one_with_inf_threshold() {
    let cam = compute_light_camera(
        [-2.0, 0.0, -3.0], [4.0, 5.0, 1.0],
        Vec3::new(-0.3, -0.7, 0.5).normalize().to_array(),
        1024, 0.001,
    );
    let csm = LightCameraCsm::single_cascade(cam);
    assert_eq!(csm.cascade_count, 1);
    assert!(csm.cascade_far_view_z[0].is_infinite());
    // Cascade 0 view_proj round-trips through cascade 0 view_proj_inv.
    let vp = Mat4::from_cols_array_2d(&csm.cascades[0].view_proj);
    let vpi = Mat4::from_cols_array_2d(&csm.cascades[0].view_proj_inv);
    let p = vp * Vec4::new(0.5, 1.0, 0.3, 1.0);
    let p = vpi * p;
    assert!((p.x / p.w - 0.5).abs() < 1e-3);
    assert!((p.y / p.w - 1.0).abs() < 1e-3);
    assert!((p.z / p.w - 0.3).abs() < 1e-3);
}

fn assert_wgsl_valid(label: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("[{label}] parse error:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module).unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
}

#[test]
fn shadow_clear_shader_is_valid_wgsl() {
    assert_wgsl_valid("shadow_clear", wesl::include_wesl!("shadow_clear"));
}

#[test]
fn shadow_scatter_setup_shader_is_valid_wgsl() {
    assert_wgsl_valid("shadow_scatter_setup", wesl::include_wesl!("shadow_scatter_setup"));
}

#[test]
fn shadow_scatter_emit_shader_is_valid_wgsl() {
    assert_wgsl_valid("shadow_scatter_emit", wesl::include_wesl!("shadow_scatter_emit"));
}

#[test]
fn shadow_scatter_finalize_shader_is_valid_wgsl() {
    assert_wgsl_valid("shadow_scatter_finalize", wesl::include_wesl!("shadow_scatter_finalize"));
}

#[test]
fn shadow_scatter_shader_is_valid_wgsl() {
    assert_wgsl_valid("shadow_scatter", wesl::include_wesl!("shadow_scatter"));
}

// ── compute_csm_cascades ─────────────────────────────────────

fn default_csm_inputs() -> CsmInputs {
    // Standard editor camera: at origin, looking down -Z, FOV 60°,
    // aspect 16:9, near 0.1, far 1000.
    let proj = Mat4::perspective_rh(60_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
    let view = Mat4::look_to_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
    let view_proj = proj * view;
    CsmInputs {
        scene_min: [-50.0, -1.0, -50.0],
        scene_max: [50.0, 20.0, 50.0],
        camera_view_proj_inv: view_proj.inverse(),
        camera_position: Vec3::ZERO,
        camera_forward: -Vec3::Z,
        light_dir: Vec3::new(0.3, -0.8, 0.5).normalize().to_array(),
        shadow_map_size: 1024,
        depth_bias: 0.001,
        csm_near: 0.1,
        csm_max_distance: 100.0,
        csm_lambda: 0.5,
    }
}

#[test]
fn csm_far_view_z_last_equals_max_distance() {
    let csm = compute_csm_cascades(default_csm_inputs());
    let last = csm.cascade_far_view_z[CSM_CASCADE_COUNT as usize - 1];
    assert!((last - 100.0).abs() < 1e-3, "expected last far_view_z = 100, got {last}");
}

#[test]
fn csm_splits_monotonic_increasing() {
    let csm = compute_csm_cascades(default_csm_inputs());
    for i in 1..CSM_CASCADE_COUNT as usize {
        assert!(
            csm.cascade_far_view_z[i] > csm.cascade_far_view_z[i - 1],
            "splits not monotonic: [{i}] = {}, [{}] = {}",
            csm.cascade_far_view_z[i],
            i - 1,
            csm.cascade_far_view_z[i - 1],
        );
    }
}

#[test]
fn csm_lambda_zero_is_uniform_split() {
    // λ=0 → splits should be evenly spaced over [near, max_distance].
    let mut inp = default_csm_inputs();
    inp.csm_lambda = 0.0;
    let csm = compute_csm_cascades(inp);
    let expected = [25.075, 50.05, 75.025, 100.0];
    for (i, &e) in expected.iter().enumerate() {
        let got = csm.cascade_far_view_z[i];
        assert!((got - e).abs() < 0.5, "uniform split [{i}]: expected {e}, got {got}");
    }
}

#[test]
fn csm_lambda_one_is_log_split() {
    // λ=1 → splits geometric: near * (far/near)^(i/N). With near=0.1
    // and far=100 over 4 cascades, the i=1..4 splits are
    // 0.1 * 1000^(i/4) = 0.5623, 3.162, 17.78, 100.
    let mut inp = default_csm_inputs();
    inp.csm_lambda = 1.0;
    let csm = compute_csm_cascades(inp);
    let near: f32 = 0.1;
    let far: f32 = 100.0;
    let n = CSM_CASCADE_COUNT as usize;
    for i in 0..n {
        let f = (i + 1) as f32 / n as f32;
        let expected = near * (far / near).powf(f);
        let got = csm.cascade_far_view_z[i];
        assert!(
            (got / expected - 1.0).abs() < 0.01,
            "log split [{i}]: expected {expected}, got {got}",
        );
    }
}

#[test]
fn csm_cascade_count_is_filled() {
    let csm = compute_csm_cascades(default_csm_inputs());
    assert_eq!(csm.cascade_count, CSM_CASCADE_COUNT);
    // All slices should have a non-zero light_dir (every cascade is real).
    for i in 0..CSM_CASCADE_COUNT as usize {
        let l = csm.cascades[i].light_dir;
        let len2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
        assert!(len2 > 0.5, "cascade {i} light_dir is zero");
    }
}

/// Recover the world-space half-width of cascade `i`'s ortho fit by
/// unprojecting NDC ±1 along the X axis and taking half the world-space
/// distance between them. Robust against light-view rotation: doesn't
/// require knowing the light view matrix.
fn cascade_half_width(csm: &LightCameraCsm, i: usize) -> f32 {
    let vpi = Mat4::from_cols_array_2d(&csm.cascades[i].view_proj_inv);
    let p_pos = vpi * Vec4::new(1.0, 0.0, 0.5, 1.0);
    let p_neg = vpi * Vec4::new(-1.0, 0.0, 0.5, 1.0);
    let world_pos = p_pos.truncate() / p_pos.w;
    let world_neg = p_neg.truncate() / p_neg.w;
    (world_pos - world_neg).length() * 0.5
}

#[test]
fn csm_sphere_fit_rotation_invariant_under_yaw() {
    // Yawing the camera around world Y permutes the slice's frustum
    // corners but keeps the bounding sphere's radius constant, so the
    // ortho half-width (which becomes shadow-map texel size) should
    // match across a 360° yaw sweep. Texel snap can shift the centre
    // by < 1 texel = 2*radius / 1024, so we allow a small tolerance.
    let base = compute_csm_cascades(default_csm_inputs());
    let base_half_width = cascade_half_width(&base, 0);

    for yaw_deg in [30.0, 90.0, 145.0, 200.0, 280.0] {
        let yaw: f32 = (yaw_deg as f32).to_radians();
        let fwd = Vec3::new(yaw.sin(), 0.0, -yaw.cos()).normalize();
        let view_y = Mat4::look_to_rh(Vec3::ZERO, fwd, Vec3::Y);
        let proj_y = Mat4::perspective_rh(60_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let mut inp = default_csm_inputs();
        inp.camera_view_proj_inv = (proj_y * view_y).inverse();
        inp.camera_forward = fwd;
        let csm = compute_csm_cascades(inp);
        let half_width = cascade_half_width(&csm, 0);

        let rel = (half_width - base_half_width).abs() / base_half_width;
        assert!(
            rel < 0.02,
            "sphere fit not rotation-invariant under yaw {yaw_deg}°: \
             base half-width = {base_half_width}, yaw half-width = {half_width}",
        );
    }
}

#[test]
fn csm_texel_snap_is_idempotent() {
    // Calling compute_csm_cascades twice with identical inputs must
    // produce bit-identical output. (The snap-to-texel-grid logic is
    // deterministic; this test guards against any nondeterminism
    // sneaking in.)
    let a = compute_csm_cascades(default_csm_inputs());
    let b = compute_csm_cascades(default_csm_inputs());
    for i in 0..CSM_CASCADE_COUNT as usize {
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    a.cascades[i].view_proj[r][c],
                    b.cascades[i].view_proj[r][c],
                    "cascade {i} view_proj[{r}][{c}] differs",
                );
            }
        }
    }
}

#[test]
fn csm_view_proj_inv_round_trips_per_cascade() {
    let csm = compute_csm_cascades(default_csm_inputs());
    for i in 0..CSM_CASCADE_COUNT as usize {
        let vp = Mat4::from_cols_array_2d(&csm.cascades[i].view_proj);
        let vpi = Mat4::from_cols_array_2d(&csm.cascades[i].view_proj_inv);
        for &ndc in &[
            [-0.5_f32, -0.5, 0.1],
            [0.0, 0.0, 0.5],
            [0.7, 0.3, 0.9],
        ] {
            let world = vpi * Vec3::new(ndc[0], ndc[1], ndc[2]).extend(1.0);
            let world = world.truncate() / world.w;
            let clip = vp * world.extend(1.0);
            let recovered = clip.truncate() / clip.w;
            assert!((recovered.x - ndc[0]).abs() < 1e-2);
            assert!((recovered.y - ndc[1]).abs() < 1e-2);
            assert!((recovered.z - ndc[2]).abs() < 1e-2);
        }
    }
}

#[test]
fn compute_light_camera_view_proj_inv_round_trips() {
    let cam = compute_light_camera(
        [-2.0, 0.0, -3.0], [4.0, 5.0, 1.0],
        Vec3::new(-0.3, -0.7, 0.5).normalize().to_array(),
        2048, 0.005,
    );
    let vp = Mat4::from_cols_array_2d(&cam.view_proj);
    let vpi = Mat4::from_cols_array_2d(&cam.view_proj_inv);
    for &ndc in &[
        [-0.9_f32, -0.9, 0.0],
        [0.9, -0.9, 0.5],
        [0.0, 0.0, 0.7],
        [0.7, 0.3, 1.0],
    ] {
        let world = vpi * Vec3::new(ndc[0], ndc[1], ndc[2]).extend(1.0);
        let world = world.truncate() / world.w;
        let clip = vp * world.extend(1.0);
        let recovered = clip.truncate() / clip.w;
        assert!((recovered.x - ndc[0]).abs() < 1e-3);
        assert!((recovered.y - ndc[1]).abs() < 1e-3);
        assert!((recovered.z - ndc[2]).abs() < 1e-3);
    }
}
