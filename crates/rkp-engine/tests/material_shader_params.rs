//! Phase A: per-material shader-param buffer construction.
//!
//! The buffer is one [f32; 8] slot per material slot, parallel to
//! `build_palette`. For each material with a shader assigned, slot
//! values come from `MaterialDef.shader_params` packed in the order
//! the shader's metadata declared them. Missing param values use the
//! shader's declared default. Materials with no shader (or an
//! unregistered shader name) get an all-zeros slot.
//!
//! Tests stand up `.rkmat` files on disk and load them via
//! `MaterialLibrary::scan()` — the same path the editor uses at
//! project load.

use rkp_engine::material_library::MaterialLibrary;
use rkp_render::shader_composer::UserShaderRegistry;
use std::io::Write;

fn tempdir(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "rkpatch_mat_params_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_file(dir: &std::path::Path, name: &str, contents: &str) {
    let mut f = std::fs::File::create(dir.join(name)).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
}

#[test]
fn empty_registry_yields_all_zero_slots() {
    let dir = tempdir("empty_reg");
    write_file(
        &dir,
        "test.rkmat",
        r#"{
  "name": "Test",
  "shader": "hologram",
  "shader_params": { "strength": 2.0 }
}"#,
    );
    let mut lib = MaterialLibrary::new();
    lib.scan(&dir);

    let reg = UserShaderRegistry::empty();
    let buf = lib.build_shader_params(&reg);
    // Slot 0 = default; slot 1 = the loaded Test material.
    assert!(buf.len() >= 2);
    assert_eq!(buf[0], [0.0; 8]);
    assert_eq!(buf[1], [0.0; 8], "unregistered shader → all zeros");
}

#[test]
fn registered_shader_packs_named_params_in_declaration_order() {
    let dir = tempdir("packed");
    let shaders_dir = dir.join("shaders");
    let materials_dir = dir.join("materials");
    std::fs::create_dir_all(&shaders_dir).unwrap();
    std::fs::create_dir_all(&materials_dir).unwrap();

    write_file(
        &shaders_dir,
        "grass.wgsl",
        r#"
// @param density: f32 = 4.0, range = [0.1, 100.0]
// @param height: f32 = 0.5, range = [0.05, 2.0]
// @param wind_amp: f32 = 0.0
fn user_grass_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#,
    );
    write_file(
        &materials_dir,
        "grass.rkmat",
        r#"{
  "name": "Grass",
  "shader": "grass",
  "shader_params": { "density": 12.0, "wind_amp": 0.7 }
}"#,
    );

    let reg = rkp_render::shader_composer::scan_dir(&shaders_dir).unwrap();
    let mut lib = MaterialLibrary::new();
    lib.scan(&materials_dir);

    let buf = lib.build_shader_params(&reg);
    let slot = buf[1];
    assert!((slot[0] - 12.0).abs() < 1e-6, "param 0 (density) overridden");
    assert!((slot[1] - 0.5).abs() < 1e-6, "param 1 (height) defaults");
    assert!((slot[2] - 0.7).abs() < 1e-6, "param 2 (wind_amp) overridden");
    for i in 3..8 {
        assert_eq!(slot[i], 0.0, "slot {i}");
    }
}

#[test]
fn material_without_shader_yields_zero_slot() {
    let dir = tempdir("no_shader");
    let shaders_dir = dir.join("shaders");
    let materials_dir = dir.join("materials");
    std::fs::create_dir_all(&shaders_dir).unwrap();
    std::fs::create_dir_all(&materials_dir).unwrap();

    write_file(
        &shaders_dir,
        "x.wgsl",
        "// @param p: f32 = 9.0
fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }",
    );
    write_file(
        &materials_dir,
        "no_shader.rkmat",
        r#"{ "name": "NoShader" }"#,
    );

    let reg = rkp_render::shader_composer::scan_dir(&shaders_dir).unwrap();
    let mut lib = MaterialLibrary::new();
    lib.scan(&materials_dir);

    let buf = lib.build_shader_params(&reg);
    assert_eq!(buf[1], [0.0; 8]);
}

#[test]
fn nine_params_are_truncated_to_eight() {
    // Defensive — until the per-material slot grows, anything past
    // index 7 is silently dropped. The packing must not write OOB.
    let dir = tempdir("nine");
    let shaders_dir = dir.join("shaders");
    let materials_dir = dir.join("materials");
    std::fs::create_dir_all(&shaders_dir).unwrap();
    std::fs::create_dir_all(&materials_dir).unwrap();

    let mut header = String::new();
    for i in 0..9 {
        header.push_str(&format!("// @param p{i}: f32 = {i}\n"));
    }
    let body = format!(
        "{header}fn user_z_shade(ctx: ShadeCtx) -> ShadeResult {{ var r: ShadeResult; return r; }}"
    );
    write_file(&shaders_dir, "z.wgsl", &body);
    write_file(
        &materials_dir,
        "z.rkmat",
        r#"{ "name": "ZMat", "shader": "z" }"#,
    );

    let reg = rkp_render::shader_composer::scan_dir(&shaders_dir).unwrap();
    let mut lib = MaterialLibrary::new();
    lib.scan(&materials_dir);

    let buf = lib.build_shader_params(&reg);
    for i in 0..8 {
        assert!((buf[1][i] - i as f32).abs() < 1e-6, "slot {i}");
    }
}
