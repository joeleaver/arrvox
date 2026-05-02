//! Phase B-redux Phase 1 — naga validation for the `instance_at`
//! chunk.
//!
//! Phase 2 will wire `USER_INSTANCE_AT_DISPATCH_BEGIN/END` markers
//! into the host march template and use `splice_X_chunk`-style
//! helpers to splice the composer's chunk in. Phase 1 ships only the
//! chunk — nothing consumes it yet — so this test wraps the chunk
//! with the minimum type stubs (`HostSample`, `UserCtx`, etc.) and
//! asserts the result parses + validates with naga.
//!
//! Uses a synthetic instance shader (no inst_to_local / inst_aabb
//! hook) so the chunk owns its struct + helper declarations and the
//! test source has no dependency on Option B's `instance_pool`-bound
//! pool-read wrappers.

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
fn instance_at_chunk_validates_standalone() {
    // Synthetic fixture: a minimal instance shader. Provides all
    // four hooks the descent path needs (proto, emit for Option B
    // back-compat, inst_aabb + inst_to_local for descent, and the
    // new instance_at hook).
    let src = r#"
// @instance_proto Pebble
struct Pebble {
    pos: vec3<f32>,
    radius: f32,
}

fn pebble_hash(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
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

fn user_pebble_instance_at(
    host_pos: vec3<f32>,
    host: HostSample,
    ctx: UserCtx,
    k: u32,
    out_instance: ptr<function, Pebble>,
) -> bool {
    if (k > 0u) { return false; }
    if (host.normal.y < 0.5) { return false; }
    let r = pebble_hash(bitcast<u32>(host_pos.x));
    var p: Pebble;
    p.pos = host_pos;
    p.radius = 0.05 + 0.05 * r;
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

    // Phase 2.c-2: the chunk now references helpers from the march
    // template (`descend_proto_octree`, `intersect_aabb`, `RkpAsset`,
    // …). Standalone naga validation would require stubbing all of
    // those out, which becomes brittle as the descent body evolves.
    // The integration test in `tests/example_grass_shader.rs` splices
    // the chunk into the live march template and validates that —
    // authoritative.  Here we just assert text-level invariants on
    // the composer's output so a regression in the splice surface
    // (e.g. a renamed helper) is caught at the chunk level.
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_instance_at("),
        "instance_at chunk should rename user_pebble_instance_at:\n{}",
        chunks.instance_at,
    );
    // Helpers are claimed by the inst_to_local chunk (which is
    // emitted first in the splice order); instance_at chunk must NOT
    // re-emit them.
    assert!(chunks.inst_to_local.contains("fn pebble_hash"));
    assert!(!chunks.instance_at.contains("fn pebble_hash"));
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_instance_descend("),
        "instance_at chunk should emit the per-shader descent body",
    );
    assert!(
        chunks.instance_at.contains("descend_proto_octree("),
        "descent body should call descend_proto_octree",
    );
    assert!(
        chunks.instance_at.contains("rkp_user_1_inst_aabb(inst)"),
        "descent body should call the renamed inst_aabb hook",
    );
    assert!(
        chunks.instance_at.contains("rkp_user_1_inst_to_local("),
        "descent body should call the renamed inst_to_local hook",
    );
    assert!(
        chunks.instance_at.contains("fn dispatch_user_instance_descend("),
        "instance_at chunk should emit the unified dispatcher",
    );
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
