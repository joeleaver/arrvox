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

// A line comment containing `/*` must not trip the block-comment
// scanner — the `//` claims everything to end-of-line.
#[test]
fn line_comment_swallows_block_comment_opener() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult {
// look ma /* not a block comment } { }
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("linecmt_blockopen");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let shade = reg.entries()[0].shade_text.as_ref().unwrap();
    assert!(shade.contains("return r;"));
    assert!(shade.trim_end().ends_with('}'));
}

// A block comment containing `//` must not terminate early — only
// `*/` ends a block comment.
#[test]
fn block_comment_swallows_line_comment_marker() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult {
/* this is // not a line comment } { } } */
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("blockcmt_lineopen");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let shade = reg.entries()[0].shade_text.as_ref().unwrap();
    assert!(shade.contains("return r;"));
    assert!(shade.trim_end().ends_with('}'));
}

// An unterminated block comment must fail — earlier behaviour silently
// walked past EOF, which made downstream errors hard to attribute.
#[test]
fn rejects_unterminated_block_comment() {
    let src = r#"
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult {
/* never closed
var r: ShadeResult;
return r;
}
"#;
    let tmp = tempfile_dir("unterminated_block");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    assert!(matches!(err, ShaderComposerError::Parse { .. }));
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

// ── compose_instance_at_chunk: stub since band-cell strip ───────
//
// The composer's `instance_at` chunk used to splice per-shader
// `rkp_user_<id>_instance_descend` bodies into the host march and
// shadow shaders for the band-cell descent path. After the strip
// the band-cell branches are gone, and the new emit pass (Phase 2
// of the rebuild) consumes the parsed `instance_at` / `inst_aabb` /
// `inst_to_local` / `inst_world_matrix` hooks directly. The chunk
// is now always empty so the splice is a no-op for every consumer
// template (`splice_const_marker` returns the template unchanged
// on empty input).

/// Even when shaders register an `instance_at` hook (with all the
/// required companion hooks), the composer emits an empty chunk —
/// the new emit pass owns this responsibility.
#[test]
fn compose_instance_at_chunk_is_empty_with_instance_at_hook() {
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
fn user_x_inst_world_matrix(inst: Pt) -> mat4x4<f32> {
let s = inst.scale;
let p = inst.pos;
return mat4x4<f32>(
vec4<f32>(s, 0.0, 0.0, 0.0),
vec4<f32>(0.0, s, 0.0, 0.0),
vec4<f32>(0.0, 0.0, s, 0.0),
vec4<f32>(p.x - 0.5 * s, p.y - 0.5 * s, p.z - 0.5 * s, 1.0),
);
}
fn user_x_instance_at(
host_pos: vec3<f32>, host: HostSample, ctx: UserCtx, k: u32,
out_instance: ptr<function, Pt>,
) -> bool {
return false;
}
"#;
    let tmp = tempfile_dir("instance_at_stub_with_hook");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    assert!(
        chunks.instance_at.is_empty(),
        "instance_at chunk should be empty after band-cell strip. Got:\n{}",
        chunks.instance_at,
    );
}

/// Empty registry → empty chunk. Same outcome as the with-hook
/// case above.
#[test]
fn compose_instance_at_chunk_is_empty_without_instance_at_hook() {
    let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32>, scale: f32 }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
    let tmp = tempfile_dir("instance_at_stub_without_hook");
    write(&tmp, "x.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let chunks = compose(&reg);
    assert!(chunks.instance_at.is_empty());
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

#[test]
fn rejects_helper_fn_colliding_with_lib_symbol() {
    // `mat_albedo` lives in lib/types.wesl. A user helper of the same
    // name would produce a duplicate root-level fn under
    // ManglerKind::None.
    let src = r#"
fn mat_albedo(m: u32) -> u32 { return m; }
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("collide_helper");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("mat_albedo"), "got: {msg}");
            assert!(msg.contains("collides with a lib symbol"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_struct_colliding_with_lib_symbol() {
    // `Aabb` is declared in lib/types.wesl; user-shader struct of
    // the same name would clash.
    let src = r#"
struct Aabb { lo: vec3<f32>, hi: vec3<f32> }
fn user_test_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
    let tmp = tempfile_dir("collide_struct");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("Aabb"), "got: {msg}");
            assert!(msg.contains("collides with a lib symbol"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn splice_marker_skips_in_comment_occurrence() {
    let template = "\
// @author wrote about USER_TEST_DISPATCH_BEGIN: u32 = 0u; somewhere
const USER_TEST_DISPATCH_BEGIN: u32 = 0u;
fn old_body() {}
const USER_TEST_DISPATCH_END: u32 = 0u;
";
    let out = splice_const_marker(template, "USER_TEST_DISPATCH", "fn new_body() {}\n");
    assert!(out.contains("fn new_body() {}"));
    assert!(!out.contains("fn old_body()"));
    // The literal mention in the leading comment must survive,
    // because the splice replaced the syntactic anchor pair only.
    assert!(out.contains("@author"));
}

#[test]
#[should_panic(expected = "missing top-level anchor")]
fn splice_marker_panics_when_only_comment_match_exists() {
    let template = "\
// const USER_TEST_DISPATCH_BEGIN: u32 = 0u;
// const USER_TEST_DISPATCH_END: u32 = 0u;
";
    let _ = splice_const_marker(template, "USER_TEST_DISPATCH", "// chunk\n");
}

#[test]
#[should_panic(expected = "contains 2 top-level anchors")]
fn splice_marker_panics_on_duplicate_top_level_anchor() {
    let template = "\
const USER_TEST_DISPATCH_BEGIN: u32 = 0u;
const USER_TEST_DISPATCH_END: u32 = 0u;
const USER_TEST_DISPATCH_BEGIN: u32 = 0u;
const USER_TEST_DISPATCH_END: u32 = 0u;
";
    let _ = splice_const_marker(template, "USER_TEST_DISPATCH", "// chunk\n");
}

#[test]
fn splice_marker_empty_chunk_is_identity() {
    let template = "\
const USER_TEST_DISPATCH_BEGIN: u32 = 0u;
fn old_body() {}
const USER_TEST_DISPATCH_END: u32 = 0u;
";
    assert_eq!(
        splice_const_marker(template, "USER_TEST_DISPATCH", ""),
        template,
    );
}

// ── V1 mesh-path tests ─────────────────────────────────────────────

#[test]
fn parses_mesh_path_shader_minimal() {
    let src = r#"
// @geometry procedural { vertex_count: 7 }
// @param density: f32 = 1.0, range = [0.0, 4.0]

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return u32(ctx_param(0) * anchor.surface_area);
}

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
}
"#;
    let tmp = tempfile_dir("mesh_path_minimal");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    assert_eq!(reg.entries().len(), 1);
    let e = &reg.entries()[0];
    assert!(e.is_mesh_path());
    assert!(matches!(
        e.metadata.mesh_geometry,
        Some(GeometryDecl::Procedural { vertex_count: 7 })
    ));
    assert_eq!(e.metadata.spawn_count_cache, SpawnCountCache::Static);
    assert!(e.spawn_count_text.is_some());
    assert!(e.vs_text.is_some());
    assert!(e.spawn_alive_text.is_none());
    assert!(e.fs_text.is_none());
}

#[test]
fn parses_mesh_path_with_per_frame_cache_and_spawn_alive() {
    let src = r#"
// @geometry procedural { vertex_count: 12 }
// @spawn_count_cache per_frame
// @animated

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return u32(10.0 + frame.time);
}

fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    return spawn_idx < 5u;
}

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0);
    return out;
}
"#;
    let tmp = tempfile_dir("mesh_path_per_frame");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    assert_eq!(e.metadata.spawn_count_cache, SpawnCountCache::PerFrame);
    assert!(e.spawn_alive_text.is_some());
}

#[test]
fn parses_mesh_geometry_asset_form() {
    let src = r#"
// @geometry mesh { asset: "rock.glb" }

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 { return 1u; }
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut; out.clip_pos = vec4<f32>(0.0); return out;
}
"#;
    let tmp = tempfile_dir("mesh_path_asset");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let e = &reg.entries()[0];
    match &e.metadata.mesh_geometry {
        Some(GeometryDecl::Mesh { asset }) => assert_eq!(asset, "rock.glb"),
        other => panic!("expected mesh, got {other:?}"),
    }
}

#[test]
fn rejects_mesh_path_missing_spawn_count() {
    let src = r#"
// @geometry procedural { vertex_count: 1 }
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut; out.clip_pos = vec4<f32>(0.0); return out;
}
"#;
    let tmp = tempfile_dir("mesh_path_no_spawn_count");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("spawn_count"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_mesh_path_missing_vs() {
    let src = r#"
// @geometry procedural { vertex_count: 1 }
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 { return 1u; }
"#;
    let tmp = tempfile_dir("mesh_path_no_vs");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains(" vs"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_mesh_path_hook_without_geometry() {
    let src = r#"
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut; out.clip_pos = vec4<f32>(0.0); return out;
}
"#;
    let tmp = tempfile_dir("mesh_path_no_geometry");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("@geometry"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn rejects_static_cache_with_frame_reference() {
    // Default cache is static; spawn_count reads frame.time → reject.
    let src = r#"
// @geometry procedural { vertex_count: 1 }
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return u32(frame.time);
}
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut; out.clip_pos = vec4<f32>(0.0); return out;
}
"#;
    let tmp = tempfile_dir("static_cache_frame_ref");
    write(&tmp, "test.wgsl", src);
    let err = scan_dir(&tmp).unwrap_err();
    match err {
        ShaderComposerError::Parse { msg, .. } => {
            assert!(msg.contains("per_frame"), "got: {msg}");
        }
        other => panic!("expected Parse, got {other:?}"),
    }
}

#[test]
fn compose_mesh_path_splices_user_body_into_both_templates() {
    // Build a registry by parsing a minimal mesh-path shader, then
    // compose against fake templates. Verify the user's vs body
    // lands in the raster output and spawn_count lands in the
    // compute output; helpers + default fs land where expected.
    let src = r#"
// @geometry procedural { vertex_count: 3 }

fn helper_double(x: f32) -> f32 { return x * 2.0; }

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return u32(helper_double(anchor.surface_area));
}

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
}
"#;
    let tmp = tempfile_dir("compose_mesh_splice");
    write(&tmp, "test.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let entry = &reg.entries()[0];

    let raster_template = "\
struct VsOut { @builtin(position) clip_pos: vec4<f32> }
const USER_BODY_BEGIN: u32 = 0u;
fn stub_placeholder() {}
const USER_BODY_END: u32 = 0u;
";
    let compute_template = "\
const USER_BODY_BEGIN: u32 = 0u;
fn stub_placeholder() {}
const USER_BODY_END: u32 = 0u;
";
    let (raster_wgsl, compute_wgsl) =
        compose_mesh_path_pipeline_sources(entry, raster_template, compute_template);

    // Raster splice received vs body + helper.
    assert!(raster_wgsl.contains("fn helper_double"));
    assert!(raster_wgsl.contains("fn vs("));
    // Default fs was emitted because user didn't provide one.
    assert!(raster_wgsl.contains("fn fs(in: VsOut)"));
    // spawn_count is compute-only; should NOT appear in raster.
    assert!(!raster_wgsl.contains("fn spawn_count"));
    // Stub got replaced.
    assert!(!raster_wgsl.contains("stub_placeholder"));

    // Compute splice received spawn_count body + helper + default
    // spawn_alive (user omitted it).
    assert!(compute_wgsl.contains("fn helper_double"));
    assert!(compute_wgsl.contains("fn spawn_count"));
    assert!(compute_wgsl.contains("fn spawn_alive"));
    // vs is raster-only; should NOT appear in compute.
    assert!(!compute_wgsl.contains("fn vs("));
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
