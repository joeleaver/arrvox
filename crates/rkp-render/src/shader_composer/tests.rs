use super::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    p
}

#[test]
fn empty_dir_yields_empty_registry() {
    let tmp = tempfile_dir("empty_dir");
    let reg = scan_dir(&tmp).unwrap();
    assert!(reg.entries().is_empty());
    assert_eq!(reg.source_hash(), fnv1a_64(&[]));
}

#[test]
fn missing_dir_yields_empty_registry() {
    let tmp = tempfile_dir("missing_root");
    let nonexistent = tmp.join("does-not-exist");
    let reg = scan_dir(&nonexistent).unwrap();
    assert!(reg.entries().is_empty());
}

#[test]
fn parses_both_hooks() {
    let src = r#"
fn user_grass_shade(ctx: ShadeCtx) -> ShadeResult {
var r: ShadeResult;
r.rgb = vec3<f32>(0.2, 0.6, 0.1);
return r;
}

fn user_grass_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
var v: VoxelEmit;
v.occupancy = host.distance < 0.5;
return v;
}
"#;
    let tmp = tempfile_dir("both_hooks");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert_eq!(e.name, "grass");
    assert_eq!(e.id, 1);
    assert!(e.shade_text.is_some());
    assert!(e.generate_text.is_some());
    assert_eq!(reg.resolve("grass"), Some(1));
    assert_eq!(reg.resolve("missing"), None);
    assert_eq!(reg.resolve(""), None);
}

#[test]
fn parses_shade_only() {
    // A pure shade-pass shader (hologram, toon, custom PBR) only
    // needs the shade hook; the geometry dispatcher's identity
    // arm covers it.
    let src = r#"
fn user_holo_shade(ctx: ShadeCtx) -> ShadeResult {
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("shade_only");
    write(&tmp, "holo.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert!(e.shade_text.is_some());
    assert!(e.generate_text.is_none());
}

#[test]
fn parses_generate_only() {
    let src = r#"
fn user_dust_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
var v: VoxelEmit;
return v;
}
"#;
    let tmp = tempfile_dir("gen_only");
    write(&tmp, "dust.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert!(e.shade_text.is_none());
    assert!(e.generate_text.is_some());
}

#[test]
fn parses_nested_braces_in_body() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult {
if (ctx.distance < 0.0) {
    var s: ShadeResult;
    if (ctx.world_pos.y > 0.0) {
        s.rgb = vec3<f32>(0.5);
    } else {
        s.rgb = vec3<f32>(0.0);
    }
    return s;
}
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("nested");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let shade = reg.entries()[0].shade_text.as_ref().unwrap();
    assert!(shade.contains("else"));
    assert!(shade.trim_end().ends_with('}'));
}

#[test]
fn skips_braces_inside_comments() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult {
// before { brace
/* commented { unbalanced } { } */
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("comments");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let shade = reg.entries()[0].shade_text.as_ref().unwrap();
    assert!(shade.contains("return r;"));
    assert!(shade.trim_end().ends_with('}'));
}

#[test]
fn rejects_unknown_hook() {
    let src = r#"
fn user_test_garble(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("bad_hook");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("unknown hook"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_hook() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("dup_hook");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    assert!(matches!(err, ShaderComposerError::Parse { .. }));
}

#[test]
fn deterministic_ids_in_alphabetical_order() {
    let tmp = tempfile_dir("ordering");
    let body = "fn user_X_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }";
    write(&tmp, "zeta.wgsl", &body.replace('X', "zeta"));
    write(&tmp, "alpha.wgsl", &body.replace('X', "alpha"));
    write(&tmp, "mu.wgsl", &body.replace('X', "mu"));
    let reg = scan_dir(&tmp).unwrap();
    assert_eq!(reg.entries()[0].name, "alpha");
    assert_eq!(reg.entries()[0].id, 1);
    assert_eq!(reg.entries()[1].name, "mu");
    assert_eq!(reg.entries()[1].id, 2);
    assert_eq!(reg.entries()[2].name, "zeta");
    assert_eq!(reg.entries()[2].id, 3);
}

#[test]
fn source_hash_changes_with_edits() {
    let tmp = tempfile_dir("hash_change");
    write(
        &tmp,
        "x.wgsl",
        "fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }",
    );
    let h1 = scan_dir(&tmp).unwrap().source_hash();
    write(
        &tmp,
        "x.wgsl",
        "fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; r.rgb = vec3<f32>(1.0); return r; }",
    );
    let h2 = scan_dir(&tmp).unwrap().source_hash();
    assert_ne!(h1, h2);
}

#[test]
fn parses_param_with_range() {
    let src = r#"
// @param density: f32 = 4.0, range = [0.1, 100.0]
// @param height: f32 = 0.5, range = [0.05, 2.0]
fn user_grass_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("params");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let md = &reg.entries()[0].metadata;
    assert_eq!(md.params.len(), 2);
    assert_eq!(md.params[0].name, "density");
    assert!((md.params[0].default - 4.0).abs() < 1e-6);
    assert_eq!(md.params[0].range, Some((0.1, 100.0)));
    assert_eq!(md.params[1].name, "height");
    assert_eq!(md.params[1].range, Some((0.05, 2.0)));
}

#[test]
fn parses_param_without_range() {
    let src = r#"
// @param wind_amp: f32 = 0.0
fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("noparams");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let md = &reg.entries()[0].metadata;
    assert_eq!(md.params.len(), 1);
    assert_eq!(md.params[0].name, "wind_amp");
    assert_eq!(md.params[0].range, None);
}

#[test]
fn parses_animated_and_region_thickness() {
    let src = r#"
// @region_thickness 0.6
// @animated
// @cell_size 0.05
fn user_grass_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
var v: VoxelEmit;
return v;
}
"#;
    let tmp = tempfile_dir("flags");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let md = &reg.entries()[0].metadata;
    assert!((md.region_thickness - 0.6).abs() < 1e-6);
    assert!(md.animated);
    assert_eq!(md.cell_size, Some(0.05));
}

#[test]
fn metadata_defaults_when_no_directives() {
    let src = r#"
fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("defaults");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let md = &reg.entries()[0].metadata;
    assert!(md.params.is_empty());
    assert_eq!(md.region_thickness, 0.0);
    assert!(!md.animated);
    assert_eq!(md.cell_size, None);
}

#[test]
fn rejects_unknown_directive() {
    let src = r#"
// @whatever 42
fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("bad_directive");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("unknown directive"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_malformed_param() {
    // Missing `=` between type and default — must reject rather
    // than silently dropping the param so users see the typo.
    let src = r#"
// @param density: f32 4.0
fn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("bad_param");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    assert!(matches!(err, ShaderComposerError::Parse { .. }));
}

#[test]
fn metadata_changes_invalidate_source_hash() {
    // The cache key for generated voxels folds in metadata so
    // toggling @animated or shifting a param default re-bakes.
    let tmp = tempfile_dir("md_hash");
    write(
        &tmp,
        "x.wgsl",
        "// @param density: f32 = 4.0\nfn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }",
    );
    let h1 = scan_dir(&tmp).unwrap().source_hash();
    write(
        &tmp,
        "x.wgsl",
        "// @param density: f32 = 5.0\nfn user_x_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }",
    );
    let h2 = scan_dir(&tmp).unwrap().source_hash();
    assert_ne!(h1, h2);
}

#[test]
fn compose_emits_both_chunks() {
    let src = r#"
// @param density: f32 = 4.0, range = [0.1, 10.0]
fn user_grass_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
fn user_grass_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
var v: VoxelEmit;
return v;
}
"#;
    let tmp = tempfile_dir("compose");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    assert!(chunks.shade.contains("dispatch_user_shade"));
    assert!(chunks.shade.contains("rkp_user_1_shade"));
    assert!(chunks.generate.contains("dispatch_user_generate"));
    assert!(chunks.generate.contains("rkp_user_1_generate"));
}

#[test]
fn compose_empty_registry_emits_identity_only() {
    let reg = UserShaderRegistry::empty();
    let chunks = compose(&reg);
    // No `case` arms — only the default identity arm.
    assert!(!chunks.shade.contains("case "));
    assert!(chunks.shade.contains("default:"));
    assert!(!chunks.generate.contains("case "));
    assert!(chunks.generate.contains("default:"));
    assert!(!chunks.proto.contains("case "));
    assert!(chunks.proto.contains("default:"));
}

#[test]
fn compose_emits_proto_chunk_for_instance_shaders() {
    let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32> }
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("compose_proto");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    assert!(chunks.proto.contains("dispatch_user_proto"));
    assert!(chunks.proto.contains("rkp_user_1_proto"));
    // The instance-struct decl must be spliced so the proto body
    // (and any helper fns) can name it.
    assert!(chunks.proto.contains("struct Blade"));
    // Non-instance shaders contribute nothing to the proto chunk.
    assert!(!chunks.proto.contains("rkp_user_2_proto"));
}

#[test]
fn compose_proto_chunk_skips_classic_shaders() {
    // A shader without `@instance_proto` must not get a proto arm.
    let src = r#"
fn user_grass_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("compose_proto_skip");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    assert!(!chunks.proto.contains("rkp_user_1_proto"));
    assert!(chunks.proto.contains("default:"));
}

// ── Phase B-redux: compose_instance_at_chunk ────────────────────

/// Composed chunk renames `user_<name>_instance_at` →
/// `rkp_user_<id>_instance_at` and emits the body verbatim under
/// the new name. The instance_at chunk is the SOLE emitter of
/// the instance struct + helpers + the bare per-shader
/// `inst_aabb` / `inst_to_local` bodies that the descent body
/// calls.
#[test]
fn compose_instance_at_chunk_renames_and_emits_struct() {
    // `instance_at` requires `inst_aabb` + `inst_to_local`
    // (descent calls both).
    let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32>, scale: f32 }

fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_x_inst_aabb(inst: Pt) -> Aabb {
var a: Aabb;
a.min = inst.pos - vec3<f32>(0.5 * inst.scale);
a.max = inst.pos + vec3<f32>(0.5 * inst.scale);
return a;
}
fn user_x_inst_to_local(world_pos: vec3<f32>, inst: Pt) -> vec3<f32> {
return (world_pos - inst.pos) / max(inst.scale, 1e-6) + vec3<f32>(0.5);
}
fn user_x_instance_at(
host_pos: vec3<f32>, host: HostSample, ctx: UserCtx, k: u32,
out_instance: ptr<function, Pt>,
) -> bool {
if (k > 0u) { return false; }
var p: Pt;
p.pos = host_pos;
p.scale = 1.0;
*out_instance = p;
return true;
}
"#;
    let tmp = tempfile_dir("instance_at_renames");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);

    // instance_at chunk now ALWAYS emits the struct decl (sole emitter).
    assert!(
        chunks.instance_at.contains("struct Pt"),
        "instance_at chunk should emit struct decl. Got:\n{}",
        chunks.instance_at,
    );
    // Bare per-shader functions called by the descent body.
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_inst_aabb("),
        "instance_at chunk should emit bare rkp_user_<id>_inst_aabb. Got:\n{}",
        chunks.instance_at,
    );
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_inst_to_local("),
        "instance_at chunk should emit bare rkp_user_<id>_inst_to_local. Got:\n{}",
        chunks.instance_at,
    );
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_instance_at("),
        "instance_at chunk should rename user_x_instance_at to \
         per-id form. Got:\n{}",
        chunks.instance_at,
    );
    // The user's body is emitted verbatim under the new name.
    assert!(chunks.instance_at.contains("ptr<function, Pt>"));
    assert!(chunks.instance_at.contains("*out_instance = p;"));
    // Phase 2.c-2 — per-shader descent body + dispatcher.
    assert!(
        chunks.instance_at.contains("fn rkp_user_1_instance_descend("),
        "instance_at chunk should emit the per-shader descent body",
    );
    assert!(
        chunks.instance_at.contains("descend_proto_octree("),
        "descent body should call descend_proto_octree",
    );
    assert!(
        chunks.instance_at.contains("fn dispatch_user_instance_descend("),
        "instance_at chunk should emit the unified dispatcher",
    );
}

/// The instance_at chunk is the SOLE emitter of the instance
/// struct + helpers + bare `inst_aabb` / `inst_to_local`. Helpers
/// from a shader that also defines those hooks come through
/// exactly once.
#[test]
fn compose_instance_at_chunk_emits_helpers_once() {
    let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32>, scale: f32 }

fn helper_noop(p: vec3<f32>) -> vec3<f32> { return p; }

fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_x_inst_to_local(world_pos: vec3<f32>, inst: Pt) -> vec3<f32> {
return helper_noop(world_pos - inst.pos);
}
fn user_x_inst_aabb(inst: Pt) -> Aabb {
var a: Aabb;
a.min = inst.pos - vec3<f32>(0.5);
a.max = inst.pos + vec3<f32>(0.5);
return a;
}
fn user_x_instance_at(
host_pos: vec3<f32>, host: HostSample, ctx: UserCtx, k: u32,
out_instance: ptr<function, Pt>,
) -> bool {
return false;
}
"#;
    let tmp = tempfile_dir("instance_at_dedupe");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);

    // instance_at chunk emits struct + helpers exactly once.
    assert_eq!(
        chunks.instance_at.matches("struct Pt").count(), 1,
        "instance_at should emit struct Pt exactly once. Got:\n{}",
        chunks.instance_at,
    );
    assert_eq!(
        chunks.instance_at.matches("fn helper_noop").count(), 1,
        "instance_at should emit helper_noop exactly once. Got:\n{}",
        chunks.instance_at,
    );
    assert!(chunks.instance_at.contains("fn rkp_user_1_instance_at("));
    assert!(chunks.instance_at.contains("fn rkp_user_1_inst_aabb("));
    assert!(chunks.instance_at.contains("fn rkp_user_1_inst_to_local("));
}

/// Empty registry → empty chunk (no `instance_at` hook
/// registered). Downstream pipelines splicing the chunk see
/// only a header comment.
#[test]
fn compose_instance_at_chunk_empty_when_no_instance_at_hook() {
    let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32>, scale: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("instance_at_empty");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    // Header comment only — no struct, no fn.
    assert!(!chunks.instance_at.contains("struct Pt"));
    assert!(!chunks.instance_at.contains("rkp_user_"));
}

/// `user_<name>_instance_at` declared without `@instance_proto`
/// directive must be rejected with a clear error.
#[test]
fn rejects_instance_at_hook_without_directive() {
    let src = r#"
fn user_x_instance_at(
host_pos: vec3<f32>, host: HostSample, ctx: UserCtx, k: u32,
out_instance: ptr<function, vec3<f32>>,
) -> bool { return false; }
"#;
    let tmp = tempfile_dir("instance_at_no_directive");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("user_x_instance_at"),
        "error message should name the offending hook; got: {msg}",
    );
    assert!(
        msg.contains("@instance_proto"),
        "error message should reference the missing directive; got: {msg}",
    );
}

// ── @instance_proto pipeline ──────────────────────────────────────

/// Canonical happy-path instance shader. Has the directive, struct,
/// proto hook + the Phase B-redux helper hooks. Should parse and
/// populate `instance_layout`.
#[test]
fn parses_full_instance_shader() {
    let src = r#"
// @instance_proto Blade
// @region_thickness 0.5
// @animated

struct Blade {
pos: vec3<f32>,
yaw: f32,
sway_phase: f32,
height_scale: f32,
tint: u32,
}

fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit {
var v: VoxelEmit;
return v;
}

fn user_grass_inst_to_local(world_pos: vec3<f32>, inst: Blade) -> vec3<f32> {
return world_pos - inst.pos;
}
"#;
    let tmp = tempfile_dir("instance_full");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert_eq!(e.metadata.instance_proto_struct.as_deref(), Some("Blade"));
    assert!(e.proto_text.is_some());
    assert!(e.inst_to_local_text.is_some());
    assert!(e.inst_aabb_text.is_none());
    let layout = e.instance_layout.as_ref().unwrap();
    assert_eq!(layout.struct_name, "Blade");
    assert_eq!(layout.total_size, 32);
    assert_eq!(layout.fields.len(), 5);
}

/// Plain shade-only shaders shouldn't pick up any instance state.
#[test]
fn classic_shade_shader_has_no_instance_layout() {
    let src = r#"
fn user_holo_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("classic_shade");
    write(&tmp, "holo.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert!(e.instance_layout.is_none());
}

/// Declaring `@instance_proto` without the matching struct decl is
/// a clear authoring error — reject so the user sees the typo
/// immediately.
#[test]
fn rejects_instance_proto_without_struct() {
    let src = r#"
// @instance_proto Missing
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("instance_no_struct");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("no matching `struct"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_instance_proto_without_proto_hook() {
    let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32> }
"#;
    let tmp = tempfile_dir("instance_no_proto");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("`user_x_proto` hook is missing"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_proto_hook_without_directive() {
    // `proto` is reserved for instance mode. Defining it without
    // `@instance_proto` is almost certainly a typo — fail loudly.
    let src = r#"
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("proto_no_directive");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("@instance_proto"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_instance_struct_missing_pos() {
    let src = r#"
// @instance_proto Bad
struct Bad { foo: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("instance_no_pos");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("missing required field"), "got: {msg}");
            assert!(msg.contains("pos"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_oversize_instance_struct() {
    let src = r#"
// @instance_proto Big
struct Big {
pos: vec3<f32>,
a: vec4<f32>,
b: vec4<f32>,
c: vec4<f32>,
d: vec4<f32>,
e: vec4<f32>,
f: vec4<f32>,
g: vec4<f32>,
}
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("instance_oversize");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("hard cap"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_duplicate_instance_proto_directive() {
    let src = r#"
// @instance_proto Blade
// @instance_proto Other
struct Blade { pos: vec3<f32> }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("dup_instance_proto");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("declared twice"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_invalid_instance_proto_identifier() {
    let src = r#"
// @instance_proto 123Bad
struct X { pos: vec3<f32> }
"#;
    let tmp = tempfile_dir("bad_proto_ident");
    write(&tmp, "x.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("not a valid identifier"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn helper_struct_alongside_instance_struct_is_legal() {
    // Users may want helper structs (e.g. an internal sampling
    // result) alongside the instance struct. They should be
    // captured in struct_decls without affecting the layout.
    let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32>, yaw: f32 }
struct LocalSample { density: f32, color: vec3<f32> }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("helper_struct");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert!(e.proto_text.is_some());
    assert!(e.instance_layout.is_some());
    assert_eq!(e.struct_decls.len(), 2);
    assert_eq!(
        e.instance_layout.as_ref().unwrap().struct_name,
        "Blade"
    );
}

#[test]
fn instance_shader_changes_invalidate_source_hash() {
    let tmp = tempfile_dir("inst_hash");
    write(
        &tmp,
        "x.wgsl",
        r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32>, yaw: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#,
    );
    let h1 = scan_dir(&tmp).unwrap().source_hash();
    // Change just the struct field — should invalidate cache.
    write(
        &tmp,
        "x.wgsl",
        r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32>, yaw: f32, scale: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#,
    );
    let h2 = scan_dir(&tmp).unwrap().source_hash();
    assert_ne!(h1, h2);
}

#[test]
fn shader_info_surfaces_instance_metadata() {
    let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32>, yaw: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("info");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let info = &reg.shader_infos()[0];
    assert_eq!(info.instance_struct_name.as_deref(), Some("Blade"));
    assert_eq!(info.instance_struct_size, Some(16));
}

fn tempfile_dir(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "rkpatch_shader_composer_{label}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}
