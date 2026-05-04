use super::*;
use glam::{Mat4, Vec3};

#[test]
fn light_camera_uniform_size_is_160() {
    assert_eq!(std::mem::size_of::<LightCameraUniform>(), 160);
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
