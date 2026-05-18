//! Phase A: `MaterialDef.shader: Option<String>` resolves to
//! `GpuMaterial.shader_id` via a `UserShaderRegistry` resolver.
//!
//! Phase B's shade pass dispatches on `shader_id`; this test only
//! checks the resolution step (no GPU required).

use arvx_engine::material_library::MaterialDef;
use arvx_render::shader_composer::UserShaderRegistry;

#[test]
fn empty_registry_resolves_all_to_zero() {
    let reg = UserShaderRegistry::empty();
    let mut def = MaterialDef::default();
    def.shader = Some("hologram".to_string());
    let gpu = def.to_gpu(&|n| reg.resolve(n), &|n| reg.resolve(n));
    assert_eq!(
        gpu.shader_id, 0,
        "unregistered shader name must resolve to 0 (identity dispatch)",
    );
}

#[test]
fn no_shader_field_resolves_to_zero() {
    let reg = UserShaderRegistry::empty();
    let def = MaterialDef::default();
    assert!(def.shader.is_none());
    let gpu = def.to_gpu(&|n| reg.resolve(n), &|n| reg.resolve(n));
    assert_eq!(gpu.shader_id, 0);
}

#[test]
fn registered_shader_resolves_to_id() {
    use std::io::Write;

    // Stand up a real registry with a single shader so `resolve`
    // returns 1.
    let dir = std::env::temp_dir().join(format!(
        "rkpatch_mat_resolve_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("hologram.wgsl");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(
        b"fn user_hologram_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }",
    )
    .unwrap();

    let reg = arvx_render::shader_composer::scan_dir(&dir).unwrap();
    assert_eq!(reg.resolve("hologram"), Some(1));

    let mut def = MaterialDef::default();
    def.shader = Some("hologram".to_string());
    let gpu = def.to_gpu(&|n| reg.resolve(n), &|n| reg.resolve(n));
    assert_eq!(gpu.shader_id, 1);

    // Different name on the same registry → still 0.
    def.shader = Some("nope".to_string());
    let gpu = def.to_gpu(&|n| reg.resolve(n), &|n| reg.resolve(n));
    assert_eq!(gpu.shader_id, 0);
}

#[test]
fn empty_string_shader_name_resolves_to_zero() {
    // Materials that have a `shader: Some("")` (e.g. user cleared the
    // dropdown) must resolve to 0, not be misinterpreted as the
    // first-registered shader.
    let reg = UserShaderRegistry::empty();
    let mut def = MaterialDef::default();
    def.shader = Some(String::new());
    let gpu = def.to_gpu(&|n| reg.resolve(n), &|n| reg.resolve(n));
    assert_eq!(gpu.shader_id, 0);
}
