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
    return u32(ctx_param(0) * (anchor.tile_max.x - anchor.tile_min.x) * (anchor.tile_max.z - anchor.tile_min.z));
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
fn compose_mesh_path_against_real_templates_produces_valid_wgsl() {
    // Regression for the "removed splice anchors leave dangling
    // identifier refs in entry points" bug — composes a real
    // mesh-path grass shader against the live skeleton WGSL
    // templates and runs the result through naga's parser +
    // validator. Catches at `cargo test` what otherwise only
    // surfaces as "RenderPipeline ... is invalid" at runtime in
    // the editor.
    let src = r#"
// @geometry procedural { vertex_count: 6 }
// @param density: f32 = 80.0, range = [1.0, 400.0]

fn hash01(s: u32) -> f32 {
    var x = s;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    return f32(x) / 4294967295.0;
}

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return u32(ctx_param(0) * (anchor.tile_max.x - anchor.tile_min.x) * (anchor.tile_max.z - anchor.tile_min.z));
}

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    let r = hash01(anchor.seed ^ spawn_idx);
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.world_pos = anchor.tile_min;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = r;
    out.intensity = 0u;
    return out;
}
"#;
    let tmp = tempfile_dir("compose_mesh_real_templates");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let entry = &reg.entries()[0];
    assert!(entry.is_mesh_path());

    let (raster_template, compute_template) =
        crate::user_shader_mesh_pass::UserShaderMeshPass::template_sources();
    let (raster_wgsl, compute_wgsl) =
        compose_mesh_path_pipeline_sources(entry, raster_template, compute_template);

    let validate = |label: &str, src: &str| {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
            panic!(
                "{label} composed WGSL failed to parse:\n{}\n\n--- source ---\n{src}",
                e.emit_to_string(src),
            )
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| {
            panic!("{label} composed WGSL failed to validate: {e:?}\n\n--- source ---\n{src}")
        });
    };
    validate("raster", &raster_wgsl);
    validate("compute", &compute_wgsl);
}

#[test]
fn compose_mesh_path_paint_probe_splices_against_real_templates() {
    // Regression for paint_probe: a mesh-path shader whose
    // spawn_alive calls the engine-provided paint_probe builtin
    // must compose + validate against the real compute template
    // (which declares the paint_probe fn + its scene bindings).
    // Catches at cargo test what would otherwise only surface as
    // "ComputePipeline ... is invalid" at runtime.
    let src = r#"
// @geometry procedural { vertex_count: 3 }
// @tile_size 0.5
// @param density: f32 = 100.0, range = [1.0, 1000.0]

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return 4u;
}

fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    let mid = mix(anchor.tile_min, anchor.tile_max, vec3<f32>(0.5));
    return paint_probe(mid, anchor);
}

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.world_pos = anchor.tile_min;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}
"#;
    let tmp = tempfile_dir("compose_mesh_paint_probe");
    write(&tmp, "grass.wgsl", src);
    let reg = scan_dir(&tmp).unwrap();
    let entry = &reg.entries()[0];
    assert!(entry.is_mesh_path());

    let (raster_template, compute_template) =
        crate::user_shader_mesh_pass::UserShaderMeshPass::template_sources();
    let (raster_wgsl, compute_wgsl) =
        compose_mesh_path_pipeline_sources(entry, raster_template, compute_template);

    let validate = |label: &str, src: &str| {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
            panic!(
                "{label} composed WGSL failed to parse:\n{}\n\n--- source ---\n{src}",
                e.emit_to_string(src),
            )
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| {
            panic!("{label} composed WGSL failed to validate: {e:?}\n\n--- source ---\n{src}")
        });
    };
    validate("raster", &raster_wgsl);
    validate("compute", &compute_wgsl);
    // Sanity: the compose actually pulled paint_probe + the user
    // spawn_alive into the compute splice.
    assert!(compute_wgsl.contains("fn paint_probe"));
    assert!(compute_wgsl.contains("paint_probe(mid, anchor)"));
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
    return u32(helper_double((anchor.tile_max.x - anchor.tile_min.x) * (anchor.tile_max.z - anchor.tile_min.z)));
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
