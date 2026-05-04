//! Regression guard for the user-shader instance_at splice surface
//! across every pipeline that calls `dispatch_user_instance_descend`.
//!
//! When a new pipeline adopts band-cell descent (Phase 4 wired march
//! + shadow_trace + shadow_scatter), the composer's `instance_at`
//! chunk gets spliced into its template. The chunk references
//! `descend_proto_octree`, `Aabb`, `unpack_oct_normal`, `RkpAsset`,
//! and a handful of constants. A new template missing any of these
//! produces a "ComputePipeline ... is invalid" GPU validation error
//! at runtime — too late to catch in CI.
//!
//! This test composes a representative `instance_at` shader, splices
//! the chunk into each consuming template, and runs naga validation
//! up front. Failure prints the exact missing identifier so the
//! template can be patched before shipping.

use rkp_render::shader_composer::{compose, scan_dir, splice_inst_chunks};

fn write_shader(dir: &std::path::Path, name: &str, contents: &str) {
    use std::io::Write;
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
}

const FIXTURE: &str = r#"
// @instance_proto Pebble
// @region_thickness 0.5

struct Pebble {
    pos: vec3<f32>,
    radius: f32,
}

fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;
    return v;
}

fn user_x_inst_aabb(inst: Pebble) -> Aabb {
    var a: Aabb;
    a.min = inst.pos - vec3<f32>(inst.radius);
    a.max = inst.pos + vec3<f32>(inst.radius);
    return a;
}

fn user_x_inst_to_local(world_pos: vec3<f32>, inst: Pebble) -> vec3<f32> {
    let inv = 1.0 / max(inst.radius * 2.0, 1e-6);
    return (world_pos - inst.pos) * inv + vec3<f32>(0.5);
}

fn user_x_instance_at(
    host_pos: vec3<f32>,
    host: HostSample,
    ctx: UserCtx,
    k: u32,
    out_instance: ptr<function, Pebble>,
) -> bool {
    if (k > 0u) { return false; }
    var p: Pebble;
    p.pos = host_pos;
    p.radius = 0.05;
    *out_instance = p;
    return true;
}
"#;

#[test]
fn instance_at_chunk_splices_cleanly_into_all_consumers() {
    let tmp = std::env::temp_dir().join(format!(
        "rkp_inst_at_splice_{}",
        std::process::id(),
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    write_shader(&tmp, "x.wgsl", FIXTURE);
    let reg = scan_dir(&tmp).expect("scan_dir");
    let chunks = compose(&reg);
    let _ = std::fs::remove_dir_all(&tmp);

    // Proto chunk is consumed by user_shader_proto.wgsl through a
    // separate marker pair (USER_PROTO_DISPATCH_BEGIN/END). Verify it
    // splices+validates separately.
    let proto_source = rkp_render::user_shader_proto_pass::compose_proto_source(&chunks.proto);
    let proto_module = naga::front::wgsl::parse_str(&proto_source).unwrap_or_else(|e| {
        panic!(
            "[user_shader_proto] proto chunk failed to PARSE:\n{}",
            e.emit_to_string(&proto_source)
        );
    });
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&proto_module).unwrap_or_else(|e| {
        panic!("[user_shader_proto] proto chunk failed naga VALIDATION: {e:?}");
    });

    for (label, template) in [
        ("octree_march", include_str!("../src/shaders/octree_march.wgsl")),
        ("rkp_shadow_trace", include_str!("../src/shaders/rkp_shadow_trace.wgsl")),
        ("shadow_scatter", include_str!("../src/shaders/shadow_scatter.wgsl")),
    ] {
        let source = splice_inst_chunks(template, &chunks.instance_at);
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            let msg = e.emit_to_string(&source);
            panic!(
                "[{label}] splice with instance_at chunk failed to PARSE.\n\
                 If you just added a new pipeline that splices the\n\
                 instance_at chunk, the template is missing one of:\n  \
                 `descend_proto_octree`, `ProtoHit`, `Aabb`,\n  \
                 `unpack_oct_normal`, or related constants. Mirror\n  \
                 the body from `octree_march.wgsl` (adapting the\n  \
                 inner `octree_lookup` call to the local signature).\n\n\
                 naga error:\n{msg}"
            );
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| {
            panic!("[{label}] splice with instance_at chunk failed naga VALIDATION: {e:?}");
        });
    }
}
