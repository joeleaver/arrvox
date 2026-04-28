//! Phase C — geom-build composition + WGSL validation.
//!
//! Verifies the user-shader composer's `generate` chunk splices into
//! the geom-build pipeline source cleanly: types resolve, the
//! dispatcher routes by `shader_id`, and the resulting WGSL passes
//! naga's full validator (parser + capability check).

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_pass::compose_geom_source;

fn write(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
    use std::io::Write;
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    p
}

fn tmpdir(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "rkp_user_shader_geom_compose_{label}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn validate(src: &str, label: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        panic!("[{label}] parse error:\n{}", e.emit_to_string(src))
    });
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module).unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
}

#[test]
fn empty_chunk_validates() {
    let src = compose_geom_source("");
    validate(&src, "empty_chunk");
}

#[test]
fn ball_shader_composes_and_validates() {
    // A "ball-on-cell" shader: every cell within `radius` of the
    // region center is occupied. Demonstrates the full compose path —
    // header parser, body capture, dispatcher emission, splice into
    // geom-build pipeline, naga validation.
    let src = r#"
// @param radius: f32 = 1.0, range = [0.1, 5.0]
fn user_ball_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
    var v: VoxelEmit;
    let center = vec3<f32>(0.0, 0.0, 0.0);
    let d = length(cell_world_pos - center);
    if (d < ctx.params[0]) {
        v.occupancy = 1u;
        v.normal = normalize(cell_world_pos - center);
        v.material_primary = 1u;
    } else {
        v.occupancy = 0u;
    }
    return v;
}
"#;
    let dir = tmpdir("ball");
    write(&dir, "ball.wgsl", src);
    let reg = scan_dir(&dir).unwrap();
    let chunks = compose(&reg);
    assert!(!chunks.generate.is_empty());
    assert!(chunks.generate.contains("rkp_user_1_generate"));
    assert!(chunks.generate.contains("dispatch_user_generate"));

    let composed = compose_geom_source(&chunks.generate);
    // Confirm the in-tree default body got replaced.
    assert!(!composed.contains("Default identity stub"));
    validate(&composed, "ball");
}

#[test]
fn animated_grass_with_params_validates() {
    // Verifies that:
    //   * `@animated` parses without rejecting the file
    //   * `@cell_size` is accepted on the file
    //   * Multi-param packing into vec4 storage round-trips through
    //     `ctx.params[0..N]` access in the user code without naga
    //     errors
    let src = r#"
// @param density: f32 = 4.0, range = [0.5, 20.0]
// @param wind_amp: f32 = 0.3, range = [0.0, 1.0]
// @param blade_height: f32 = 0.5, range = [0.05, 2.0]
// @cell_size 0.05
// @animated

fn user_grass_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
    var v: VoxelEmit;
    let dx = sin(cell_world_pos.x * ctx.params[0]) * ctx.params[1] * sin(ctx.time);
    let blade_top = host.distance + ctx.params[2];
    if (cell_world_pos.y < blade_top + dx) {
        v.occupancy = 1u;
        v.normal = host.normal;
        v.material_primary = 2u;
    } else {
        v.occupancy = 0u;
    }
    return v;
}
"#;
    let dir = tmpdir("grass");
    write(&dir, "grass.wgsl", src);
    let reg = scan_dir(&dir).unwrap();
    let entry = &reg.entries()[0];
    assert!(entry.metadata.animated);
    assert_eq!(entry.metadata.cell_size, Some(0.05));
    assert_eq!(entry.metadata.params.len(), 3);

    let chunks = compose(&reg);
    let composed = compose_geom_source(&chunks.generate);
    validate(&composed, "grass");
}

#[test]
fn shader_with_no_generate_hook_leaves_default_in_place() {
    // A shade-only shader (Phase B) should produce a generate chunk
    // that contains the dispatcher but no `case` arms beyond the
    // default. The geom-build pipeline still validates.
    let src = r#"
fn user_holo_shade(ctx: ShadeCtx) -> ShadeResult {
    var r: ShadeResult;
    return r;
}
"#;
    let dir = tmpdir("holo");
    write(&dir, "holo.wgsl", src);
    let reg = scan_dir(&dir).unwrap();
    let chunks = compose(&reg);
    // Generate chunk has the dispatcher with no case arms — only the
    // default identity arm.
    assert!(chunks.generate.contains("dispatch_user_generate"));
    assert!(!chunks.generate.contains("case "));
    let composed = compose_geom_source(&chunks.generate);
    validate(&composed, "holo");
}
