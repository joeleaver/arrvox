//! CPU mirror of the WGSL `mat4_affine_inverse` helper (in
//! `lib/math.wesl`, currently consumed by `user_shader_mesh_compute`).
//! The Rust side computes the same formula and compares against
//! glam's general inverse for several non-trivial transforms.
//!
//! An earlier version shipped a column-vs-row-vector confusion that
//! gave R instead of R^T for rotations — invisible to syntax
//! validation, but every shaded normal was wrong. This test fails
//! fast instead.

use glam::{Mat4, Quat, Vec3};

/// Pure-Rust mirror of the WGSL helper. Must stay in lock-step with the
/// shader copies — if you edit one, edit all three.
fn mat4_affine_inverse_oracle(m: Mat4) -> Mat4 {
    // glam Mat4 is column-major; `m.col(i).truncate()` gives the i-th
    // column of the upper-left 3x3.
    let a = m.col(0).truncate();
    let b = m.col(1).truncate();
    let c = m.col(2).truncate();
    let t = m.col(3).truncate();
    let inv_det = 1.0 / a.dot(b.cross(c));
    let row0 = b.cross(c) * inv_det;
    let row1 = c.cross(a) * inv_det;
    let row2 = a.cross(b) * inv_det;
    let new_t = -Vec3::new(row0.dot(t), row1.dot(t), row2.dot(t));
    Mat4::from_cols_array_2d(&[
        [row0.x, row1.x, row2.x, 0.0],
        [row0.y, row1.y, row2.y, 0.0],
        [row0.z, row1.z, row2.z, 0.0],
        [new_t.x, new_t.y, new_t.z, 1.0],
    ])
}

fn approx_eq(a: Mat4, b: Mat4, tol: f32) -> bool {
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array().iter())
        .all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn identity_inverts_to_identity() {
    let m = Mat4::IDENTITY;
    assert!(approx_eq(mat4_affine_inverse_oracle(m), Mat4::IDENTITY, 1e-6));
}

#[test]
fn translation_only() {
    let m = Mat4::from_translation(Vec3::new(3.0, -7.0, 11.0));
    let want = m.inverse();
    assert!(approx_eq(mat4_affine_inverse_oracle(m), want, 1e-5));
}

#[test]
fn rotation_z_90deg_matches_transpose() {
    // The exact bug from the Phase 2 regression: a 90° rotation that
    // returned R instead of R^T. R(90°z)^-1 sends +x → -y (not +y).
    let m = Mat4::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));
    let inv = mat4_affine_inverse_oracle(m);
    let want = m.inverse();
    assert!(approx_eq(inv, want, 1e-5),
        "rotation inverse wrong:\nexpected:\n{:?}\ngot:\n{:?}",
        want.to_cols_array_2d(), inv.to_cols_array_2d());
    // Concrete spot-check: applied to (+x, 0, 0, 1) we should get (0, -1, 0, 1).
    let v = inv * glam::Vec4::new(1.0, 0.0, 0.0, 1.0);
    assert!((v.x - 0.0).abs() < 1e-5 && (v.y - (-1.0)).abs() < 1e-5);
}

#[test]
fn rotation_arbitrary_axis() {
    let q = Quat::from_axis_angle(Vec3::new(0.4, 0.7, -0.6).normalize(), 1.23);
    let m = Mat4::from_quat(q);
    let want = m.inverse();
    assert!(approx_eq(mat4_affine_inverse_oracle(m), want, 1e-5));
}

#[test]
fn trs_with_uniform_scale() {
    let m = Mat4::from_scale_rotation_translation(
        Vec3::splat(2.5),
        Quat::from_rotation_y(0.7),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let want = m.inverse();
    assert!(approx_eq(mat4_affine_inverse_oracle(m), want, 1e-5));
}

#[test]
fn trs_with_non_uniform_scale() {
    // Non-uniform scale is the hard case for normal-transform code that
    // confuses M, M^T, and M^-T. The march hands the inverse through to
    // the local-space lookup; correctness here means the local march
    // sees the same world.
    let m = Mat4::from_scale_rotation_translation(
        Vec3::new(1.5, 0.5, 3.0),
        Quat::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 0.9),
        Vec3::new(-4.0, 5.0, 6.0),
    );
    let want = m.inverse();
    assert!(approx_eq(mat4_affine_inverse_oracle(m), want, 1e-4));
}

#[test]
fn round_trip_to_identity() {
    // Composing M and our M^-1 should give I — strongest end-to-end check.
    let m = Mat4::from_scale_rotation_translation(
        Vec3::new(1.5, 2.0, 0.7),
        Quat::from_axis_angle(Vec3::new(0.2, -0.8, 0.5).normalize(), 1.5),
        Vec3::new(7.0, -3.0, 2.0),
    );
    let inv = mat4_affine_inverse_oracle(m);
    let composed = m * inv;
    assert!(approx_eq(composed, Mat4::IDENTITY, 1e-4));
}
