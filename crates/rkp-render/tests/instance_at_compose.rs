//! Composer's `instance_at` chunk is now a stub (empty string) since
//! the band-cell descent path it fed has been deleted. The user shader
//! API still parses `instance_at` / `inst_aabb` / `inst_to_local`
//! hooks — those will be consumed by the new emit pass (rebuild
//! Phase 8/9) which writes `RkpInstance` records with forward affine
//! `world` matrices. Until that lands, the chunk stays empty and any
//! splice into a host template is a no-op.

use rkp_render::shader_composer::{compose, scan_dir};

fn assert_wgsl_valid(source: &str, label: &str) {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!("[{label}] parse error:\n{}", e.emit_to_string(source))
    });
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
}

fn write_shader(dir: &std::path::Path, name: &str, body: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(name), body).unwrap();
}

#[test]
fn instance_at_chunk_is_empty_when_hook_present() {
    // Even when a shader registers all four instance hooks (proto,
    // inst_aabb, inst_to_local, instance_at), the composer's
    // `instance_at` chunk is empty. The new emit pass owns the work
    // these hooks feed; the host march no longer splices in any
    // per-shader descent body.
    let src = r#"
// @instance_proto Pebble
struct Pebble {
    pos: vec3<f32>,
    radius: f32,
}

fn user_pebble_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;
    return v;
}

fn user_pebble_inst_aabb(inst: Pebble) -> Aabb {
    var a: Aabb;
    a.min = inst.pos - vec3<f32>(inst.radius);
    a.max = inst.pos + vec3<f32>(inst.radius);
    return a;
}

fn user_pebble_inst_to_local(world_pos: vec3<f32>, inst: Pebble) -> vec3<f32> {
    let inv = 1.0 / max(inst.radius * 2.0, 1e-6);
    return (world_pos - inst.pos) * inv + vec3<f32>(0.5);
}

fn user_pebble_inst_world_matrix(inst: Pebble) -> mat4x4<f32> {
    let d = inst.radius * 2.0;
    let p = inst.pos;
    return mat4x4<f32>(
        vec4<f32>(d, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, d, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, d, 0.0),
        vec4<f32>(p.x - inst.radius, p.y - inst.radius, p.z - inst.radius, 1.0),
    );
}

fn user_pebble_instance_at(
    host_pos: vec3<f32>,
    host: HostSample,
    ctx: UserCtx,
    k: u32,
    out_instance: ptr<function, Pebble>,
) -> bool {
    if (k > 0u) { return false; }
    if (host.normal.y < 0.5) { return false; }
    var p: Pebble;
    p.pos = host_pos;
    p.radius = 0.05;
    *out_instance = p;
    return true;
}
"#;

    let tmp = std::env::temp_dir().join(format!(
        "rkp_instance_at_compose_{}",
        std::process::id(),
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    write_shader(&tmp, "pebble.wgsl", src);
    let reg = scan_dir(&tmp).unwrap_or_else(|e| {
        let _ = std::fs::remove_dir_all(&tmp);
        panic!("scan_dir failed: {e}");
    });
    let chunks = compose(&reg);
    let _ = std::fs::remove_dir_all(&tmp);

    assert!(
        chunks.instance_at.is_empty(),
        "instance_at chunk should be empty stub. Got:\n{}",
        chunks.instance_at,
    );
    let _ = assert_wgsl_valid;
}

#[test]
fn instance_at_chunk_empty_for_no_instance_at_hook_validates() {
    // Registry with shaders that don't define instance_at → chunk is
    // just the header comment. Confirm the trivially-empty chunk
    // still produces valid WGSL when wrapped (no surprise unbalanced
    // markers or syntax fragments).
    let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32> }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;

    let tmp = std::env::temp_dir().join(format!(
        "rkp_instance_at_empty_{}",
        std::process::id(),
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    write_shader(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    let _ = std::fs::remove_dir_all(&tmp);

    let test_source = format!(
        r#"
{instance_at}

@compute @workgroup_size(1)
fn _test_entry() {{ }}
"#,
        instance_at = chunks.instance_at,
    );
    assert_wgsl_valid(&test_source, "instance_at empty");
}
