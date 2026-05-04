//! Phase B: `compose_shade_source` produces naga-valid WGSL with both
//! the empty-user-chunk path (in-tree identity stub stays put) and
//! a real registry-composed chunk (replaces the stub).

use rkp_render::shader_composer;

fn naga_validates(source: &str) -> Result<(), String> {
    match naga::front::wgsl::parse_str(source) {
        Ok(module) => {
            let mut v = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            v.validate(&module)
                .map_err(|e| format!("validation error: {e}"))
                .map(|_| ())
        }
        Err(e) => Err(format!("parse error:\n{}", e.emit_to_string(source))),
    }
}

#[test]
fn empty_user_chunk_validates() {
    let src = rkp_render::rkp_shade::compose_shade_source("");
    naga_validates(&src).expect("empty-chunk shade WGSL should validate");
}

#[test]
fn registry_chunk_replaces_marker_block_and_validates() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!(
        "rkpatch_shade_compose_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = std::fs::File::create(dir.join("hologram.wgsl")).unwrap();
    f.write_all(
        br#"
// @param strength: f32 = 1.0, range = [0.0, 4.0]
fn user_hologram_shade(ctx: ShadeCtx) -> ShadeResult {
    let pulse = 0.5 + 0.5 * sin(ctx.time * 4.0);
    let fres = pow(1.0 - ctx.n_dot_v, 3.0);
    let strength = ctx.params[0].x;
    let rgb = vec3<f32>(0.1, 0.7, 1.0) * (fres * strength) + vec3<f32>(0.0, 0.05, 0.1) * pulse;
    return ShadeResult(rgb);
}
"#,
    )
    .unwrap();

    let reg = shader_composer::scan_dir(&dir).unwrap();
    let chunks = shader_composer::compose(&reg);
    assert!(chunks.shade.contains("rkp_user_1_shade"));
    assert!(chunks.shade.contains("dispatch_user_shade"));

    let src = rkp_render::rkp_shade::compose_shade_source(&chunks.shade);
    // The composed source MUST NOT still contain the in-tree identity
    // dispatch — the marker block should have been swapped out.
    assert!(
        !src.contains("return shade_result_passthrough(ctx);\n}\nconst USER_SHADE_DISPATCH_END"),
        "marker block was not replaced"
    );
    assert!(src.contains("rkp_user_1_shade"));
    naga_validates(&src).expect("composed shade WGSL should validate");
}
