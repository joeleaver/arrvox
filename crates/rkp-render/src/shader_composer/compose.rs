//! Registry → per-pipeline WGSL chunks.
//!
//! Four chunks, one per pipeline that consumes user shaders:
//!   - `shade` — spliced into `rkp_shade.wgsl`; defines
//!     `dispatch_user_shade(shader_id, ctx) -> ShadeResult`.
//!   - `generate` — spliced into `user_shader_geom.wgsl`'s BFS fill;
//!     defines `dispatch_user_generate(shader_id, ...) -> VoxelEmit`.
//!   - `proto` — spliced into `user_shader_proto.wgsl`'s prototype
//!     bake; defines `dispatch_user_proto(shader_id, uvw) -> VoxelEmit`.
//!   - `instance_at` — spliced into the host march + shadow trace;
//!     defines per-shader `rkp_user_<id>_instance_descend(...)` plus
//!     the unified `dispatch_user_instance_descend(...)` switch.
//!
//! Each chunk emits the user's hook bodies verbatim under a stable name
//! (`rkp_user_<id>_<hook>`), then a switch-statement dispatcher that
//! routes by `shader_id`. `shader_id == 0` (no shader registered) hits
//! the identity default arm in every dispatcher.
//!
//! [`splice_inst_chunks`] is a small splice helper consumers call to
//! drop the `instance_at` chunk into a host-side WGSL template
//! between the `USER_INSTANCE_AT_DISPATCH_BEGIN/END` const-decl
//! anchors. The four splice consumers (this one plus the shade,
//! generate, and proto pipelines) all funnel through
//! [`splice_const_marker`].
//!
//! Anchor form: `const USER_<NAME>_BEGIN: u32 = 0u;` — declarations
//! survive WESL's parse-and-emit roundtrip (line comments do not),
//! so the same templates can be either text-spliced raw `.wgsl` or
//! WESL-emitted `.wesl` and the splicer keeps working.

use super::types::{ComposedChunks, UserShaderEntry, UserShaderRegistry};

/// Compose the per-pipeline dispatch chunks. Returns identity-default
/// chunks when the registry is empty. Phase A: emits structurally
/// valid switch statements but the surrounding WGSL types
/// (`ShadeCtx`, `ShadeResult`, `HostSample`, `VoxelEmit`, `UserCtx`)
/// will land with their consuming pipelines in Phases B and C.
pub fn compose(reg: &UserShaderRegistry) -> ComposedChunks {
    ComposedChunks {
        shade: compose_shade_chunk(reg),
        generate: compose_generate_chunk(reg),
        proto: compose_proto_chunk(reg),
        instance_at: compose_instance_at_chunk(reg),
        emit: compose_emit_chunk(reg),
    }
}

/// Splice the composer's `emit` chunk into the user-shader emit
/// shader between the `USER_EMIT_DISPATCH_BEGIN/END` markers. Empty
/// chunk leaves the template's no-op stub in place (no-shader-
/// registered case).
pub fn splice_emit_chunks(template: &str, emit_chunk: &str) -> String {
    splice_const_marker(template, "USER_EMIT_DISPATCH", emit_chunk)
}

/// V1 mesh-path — compose the per-shader raster + compute WGSL
/// sources for a single user shader. The orchestration layer caches
/// the output keyed on `(entry.id, source_hash)` and builds
/// pipelines via `UserShaderMeshPass::build_pipelines`.
///
/// Two outputs (one per shader module):
///   · `raster` — `user_shader_mesh.wgsl` with the user's body
///     (helpers + structs + `vs` + optional `fs`) spliced between
///     `USER_BODY_BEGIN/END`.
///   · `compute` — `user_shader_mesh_compute.wgsl` with helpers +
///     structs + `spawn_count` + optional `spawn_alive` spliced.
///
/// Helpers and struct decls land in BOTH outputs since the user
/// might call a helper from `vs` (raster) AND from `spawn_count`
/// (compute). Per-spawn determinism relies on identical helper
/// behaviour across the two compilations.
pub fn compose_mesh_path_pipeline_sources(
    entry: &UserShaderEntry,
    raster_template: &str,
    compute_template: &str,
) -> (String, String) {
    let raster_body = build_mesh_path_raster_body(entry);
    let compute_body = build_mesh_path_compute_body(entry);
    let raster = splice_const_marker(raster_template, "USER_BODY", &raster_body);
    let compute = splice_const_marker(compute_template, "USER_BODY", &compute_body);
    (raster, compute)
}

fn build_mesh_path_raster_body(entry: &UserShaderEntry) -> String {
    let mut out = String::new();
    out.push_str("// ── user shader (mesh-path raster): ");
    out.push_str(&entry.name);
    out.push_str(" ──\n");
    for sd in &entry.struct_decls {
        out.push_str(sd);
        out.push('\n');
    }
    for helper in &entry.helpers {
        out.push_str(helper);
        out.push('\n');
    }
    if let Some(text) = &entry.vs_text {
        out.push_str(text);
        out.push('\n');
    }
    // Only emit user fs if defined. When absent, the template's
    // default fs body stays in place (the splicer leaves the section
    // unchanged when the chunk is empty, but here we always splice,
    // so we must include the default).
    if let Some(text) = &entry.fs_text {
        out.push_str(text);
        out.push('\n');
    } else {
        out.push_str(DEFAULT_FS_BODY);
        out.push('\n');
    }
    out
}

fn build_mesh_path_compute_body(entry: &UserShaderEntry) -> String {
    let mut out = String::new();
    out.push_str("// ── user shader (mesh-path compute): ");
    out.push_str(&entry.name);
    out.push_str(" ──\n");
    for sd in &entry.struct_decls {
        out.push_str(sd);
        out.push('\n');
    }
    for helper in &entry.helpers {
        out.push_str(helper);
        out.push('\n');
    }
    if let Some(text) = &entry.spawn_count_text {
        out.push_str(text);
        out.push('\n');
    }
    if let Some(text) = &entry.spawn_alive_text {
        out.push_str(text);
        out.push('\n');
    } else {
        out.push_str(DEFAULT_SPAWN_ALIVE_BODY);
        out.push('\n');
    }
    out
}

/// Default `fs` body matching the engine skeleton's stub. The
/// composer always splices the USER_BODY region (it can't selectively
/// leave parts untouched), so when the user omits `fs` we re-emit
/// the engine default here.
const DEFAULT_FS_BODY: &str = r#"fn fs(in: VsOut) -> FsOut {
    let n_world = normalize(in.world_normal);
    let primary   = in.material_packed & 0xFFFFu;
    let secondary = (in.material_packed >> 16u) & 0xFFFu;
    let blend_clamped = clamp(in.blend_f, 0.0, 1.0);
    let blend4 = u32(blend_clamped * 15.0 + 0.5);
    let blend8 = (blend4 << 4u) | blend4;
    let cr8 = u32(clamp(in.color_rgb.r, 0.0, 1.0) * 255.0 + 0.5);
    let cg8 = u32(clamp(in.color_rgb.g, 0.0, 1.0) * 255.0 + 0.5);
    let cb8 = u32(clamp(in.color_rgb.b, 0.0, 1.0) * 255.0 + 0.5);
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);
    let packed_r = primary | (secondary << 16u);
    let packed_g = (blend8 & 0xFFu) | ((in.intensity & 0xFFu) << 8u) | (color_rgb565 << 16u);
    let hit_distance = length(in.world_pos - camera.position.xyz);
    var out: FsOut;
    out.position = vec4<f32>(in.world_pos, hit_distance);
    out.pick     = 0xFFFFFFFEu;
    out.normal   = vec4<f32>(n_world, 1.0);
    out.material = vec2<u32>(packed_r, packed_g);
    out.glass    = vec2<u32>(0u, 0u);
    return out;
}"#;

/// Default `spawn_alive` body — always true. Same rationale as
/// `DEFAULT_FS_BODY`.
const DEFAULT_SPAWN_ALIVE_BODY: &str =
    "fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool { return true; }";

/// Splice the composer's `instance_at` chunk into a host-side WGSL
/// template (`octree_march.wgsl` / `rkp_shadow_trace.wgsl`) between
/// the `USER_INSTANCE_AT_DISPATCH_BEGIN/END` marker pair. Empty chunk
/// leaves the template's identity-arm stub in place — that's the
/// no-user-shader-registered case. Pipelines call this whenever the
/// registry's `source_hash` changes.
pub fn splice_inst_chunks(
    template: &str,
    instance_at_chunk: &str,
) -> String {
    splice_const_marker(
        template,
        concat!("USER_INSTANCE_AT_DISPATCH"),
        instance_at_chunk,
    )
}

/// Replace everything from the `const <NAME>_BEGIN: u32 = 0u;`
/// declaration through the matching `const <NAME>_END: u32 = 0u;`
/// declaration (inclusive, both anchors consumed) with `chunk`. Empty
/// chunk returns the template unchanged — anchors stay in place and
/// behave as harmless unused const declarations.
///
/// Const-decl anchors (versus comment markers) survive WESL's
/// parse-emit roundtrip; this is the splice contract every
/// user-shader pipeline depends on.
///
/// Anchors are matched only when they appear at top level — the
/// marker text must occupy its own line modulo leading whitespace.
/// A literal mention of the marker in a comment, docstring, or
/// string body therefore does not match. Templates must contain
/// exactly one BEGIN and one END anchor; both 0 and >1 matches
/// panic with a diagnostic.
pub fn splice_const_marker(template: &str, marker_name: &str, chunk: &str) -> String {
    if chunk.is_empty() {
        return template.to_string();
    }
    let begin = format!("const {marker_name}_BEGIN: u32 = 0u;");
    let end = format!("const {marker_name}_END: u32 = 0u;");
    let begin_idx = find_unique_top_level_anchor(template, &begin);
    // Search for END strictly after BEGIN — a paired anchor never
    // appears before its opener.
    let end_rel = find_unique_top_level_anchor(&template[begin_idx..], &end);
    let end_idx = begin_idx + end_rel + end.len();
    let mut out = String::with_capacity(template.len() + chunk.len());
    out.push_str(&template[..begin_idx]);
    out.push_str(chunk);
    out.push_str(&template[end_idx..]);
    out
}

/// Locate `needle` at top-level position in `haystack` — i.e. the
/// match's line contains only whitespace before the needle. Panics
/// with a diagnostic when there are zero or multiple top-level
/// matches; in-comment occurrences are skipped silently.
fn find_unique_top_level_anchor(haystack: &str, needle: &str) -> usize {
    let mut hits: Vec<usize> = Vec::new();
    let mut search_from = 0usize;
    while let Some(off) = haystack[search_from..].find(needle) {
        let absolute = search_from + off;
        if anchor_is_top_level(haystack, absolute) {
            hits.push(absolute);
        }
        search_from = absolute + needle.len();
    }
    match hits.len() {
        0 => panic!("template missing top-level anchor `{needle}`"),
        1 => hits[0],
        n => panic!(
            "template contains {n} top-level anchors `{needle}` — \
             anchors must be unique (at offsets {:?})",
            hits
        ),
    }
}

/// True iff the bytes from the start of `byte_offset`'s line up to
/// the offset are all ASCII whitespace. Skips in-comment matches
/// (which have `//` before the marker on the same line).
fn anchor_is_top_level(source: &str, byte_offset: usize) -> bool {
    let bytes = source.as_bytes();
    let mut i = byte_offset;
    while i > 0 {
        let prev = bytes[i - 1];
        if prev == b'\n' {
            break;
        }
        if !prev.is_ascii_whitespace() {
            return false;
        }
        i -= 1;
    }
    true
}

fn compose_shade_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader helpers + bodies: shade ───────────────\n");
    for entry in &reg.entries {
        if entry.shade_text.is_some() {
            for helper in &entry.helpers {
                out.push_str(helper);
                out.push('\n');
            }
        }
    }
    for entry in &reg.entries {
        if let Some(text) = &entry.shade_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_shade", entry.name),
                &format!("rkp_user_{}_shade", entry.id),
            ));
            out.push('\n');
        }
    }
    out.push_str("\n// ── dispatch_user_shade ────────────────────────────────\n");
    out.push_str(
        "fn dispatch_user_shade(shader_id: u32, ctx: ShadeCtx) -> ShadeResult {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.shade_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ return rkp_user_{}_shade(ctx); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str("        default: { return shade_result_passthrough(ctx); }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

fn compose_proto_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader helpers + bodies: proto ───────────────\n");
    // Splice every helper struct from instance shaders so the user's
    // `proto` body and any helper fns can reference them. Non-instance
    // shaders contribute nothing to the proto chunk.
    for entry in &reg.entries {
        if entry.proto_text.is_some() {
            for sd in &entry.struct_decls {
                out.push_str(sd);
                out.push('\n');
            }
            for helper in &entry.helpers {
                out.push_str(helper);
                out.push('\n');
            }
        }
    }
    for entry in &reg.entries {
        if let Some(text) = &entry.proto_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_proto", entry.name),
                &format!("rkp_user_{}_proto", entry.id),
            ));
            out.push('\n');
        }
    }
    out.push_str("\n// ── dispatch_user_proto ────────────────────────────────\n");
    out.push_str(
        "fn dispatch_user_proto(shader_id: u32, uvw: vec3<f32>) -> VoxelEmit {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.proto_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ return rkp_user_{}_proto(uvw); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str("        default: { return voxel_emit_skip(); }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

fn compose_generate_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader helpers + bodies: generate ────────────\n");
    for entry in &reg.entries {
        if entry.generate_text.is_some() {
            for helper in &entry.helpers {
                out.push_str(helper);
                out.push('\n');
            }
        }
    }
    for entry in &reg.entries {
        if let Some(text) = &entry.generate_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_generate", entry.name),
                &format!("rkp_user_{}_generate", entry.id),
            ));
            out.push('\n');
        }
    }
    out.push_str(
        "\n// ── dispatch_user_generate ─────────────────────────────\n",
    );
    out.push_str(
        "fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.generate_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ return rkp_user_{}_generate(cell_world_pos, host, ctx); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str("        default: { return voxel_emit_skip(); }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

/// `instance_at` chunk composer. Stub since the band-cell descent
/// path is gone — there are no consumers of an `instance_at` chunk
/// in the host march / shadow shaders any more. Returns an empty
/// chunk so `splice_inst_chunks` is a no-op for every consumer
/// template (templates never had to remove their markers, since the
/// splice helper is a no-op on empty input).
///
/// The user-shader API still parses `instance_at` / `inst_aabb` /
/// `inst_to_local` / `inst_world_matrix` — those bodies feed the
/// new emit pass (Phase 2 of the rebuild). The composer will grow a
/// new `emit` chunk that consumes them, replacing this empty stub.
fn compose_instance_at_chunk(_reg: &UserShaderRegistry) -> String {
    String::new()
}

/// Compose the `emit` chunk. The user-shader emit pass splices this
/// between its `USER_EMIT_DISPATCH_BEGIN/END` markers; the chunk
/// defines:
///
///   - The per-shader instance struct (e.g. `struct Blade { ... }`),
///     helper fns, and renamed bodies for `instance_at` +
///     `inst_world_matrix` (named `rkp_user_<id>_instance_at` and
///     `rkp_user_<id>_inst_world_matrix`).
///   - The unified `dispatch_user_emit(shader_id, ...)` switch. On
///     success, sets `*out_world_matrix` to the forward affine and
///     returns `true`. On instance_at returning `false` (no instance
///     at this k), or shader_id missing, returns `false`.
///
/// Empty when no shader registers an `instance_at` hook (the empty-
/// registry stub in the template returns `false` for all inputs).
fn compose_emit_chunk(reg: &UserShaderRegistry) -> String {
    if !reg.entries.iter().any(|e| e.instance_at_text.is_some()) {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("// ── user-shader bodies: emit ─────────────────────────\n");

    // Splice instance struct decls + helpers for every emit-eligible
    // shader. This chunk is the SOLE emitter of the per-shader struct
    // + helpers when the emit chunk is non-empty (the shade / proto /
    // generate chunks own their own copies for their respective
    // pipelines; the emit pass lives in a separate compute shader).
    for entry in &reg.entries {
        if entry.instance_at_text.is_some() {
            for sd in &entry.struct_decls {
                out.push_str(sd);
                out.push('\n');
            }
            for helper in &entry.helpers {
                out.push_str(helper);
                out.push('\n');
            }
        }
    }

    // Per-shader bodies. Validation upstream guarantees `instance_at`
    // implies `inst_world_matrix` + `inst_aabb` + `inst_to_local`.
    for entry in &reg.entries {
        if entry.instance_at_text.is_none() {
            continue;
        }
        if let Some(text) = &entry.instance_at_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_instance_at", entry.name),
                &format!("rkp_user_{}_instance_at", entry.id),
            ));
            out.push('\n');
        }
        if let Some(text) = &entry.inst_world_matrix_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_inst_world_matrix", entry.name),
                &format!("rkp_user_{}_inst_world_matrix", entry.id),
            ));
            out.push('\n');
        }
        // `inst_aabb` and `inst_to_local` aren't called from the emit
        // pass, but their bodies sometimes share helper types with
        // `inst_world_matrix`. Splicing them costs nothing (naga
        // DCE-eliminates) and keeps the per-shader splice consistent
        // across pipelines.
        if let Some(text) = &entry.inst_aabb_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_inst_aabb", entry.name),
                &format!("rkp_user_{}_inst_aabb", entry.id),
            ));
            out.push('\n');
        }
        if let Some(text) = &entry.inst_to_local_text {
            out.push_str(&rewrite_fn_name(
                text,
                &format!("user_{}_inst_to_local", entry.name),
                &format!("rkp_user_{}_inst_to_local", entry.id),
            ));
            out.push('\n');
        }
    }

    // Unified dispatcher.
    out.push_str("// ── dispatch_user_emit ───────────────────────────────\n");
    out.push_str(
        "fn dispatch_user_emit(\n\
         \x20   shader_id: u32,\n\
         \x20   host_pos: vec3<f32>,\n\
         \x20   host: HostSample,\n\
         \x20   ctx: UserCtx,\n\
         \x20   k: u32,\n\
         \x20   out_world_matrix: ptr<function, mat4x4<f32>>,\n\
         \x20   out_aabb: ptr<function, Aabb>,\n\
         ) -> bool {\n\
         \x20   switch shader_id {\n",
    );
    for entry in &reg.entries {
        if entry.instance_at_text.is_none() {
            continue;
        }
        let layout = entry.instance_layout.as_ref().expect(
            "instance_at hook implies @instance_proto layout parsed",
        );
        out.push_str(&format!(
            "        case {id}u: {{\n\
             \x20           var inst: {struct_name};\n\
             \x20           if (!rkp_user_{id}_instance_at(host_pos, host, ctx, k, &inst)) {{\n\
             \x20               return false;\n\
             \x20           }}\n\
             \x20           *out_world_matrix = rkp_user_{id}_inst_world_matrix(inst);\n\
             \x20           *out_aabb = rkp_user_{id}_inst_aabb(inst);\n\
             \x20           return true;\n\
             \x20       }}\n",
            id = entry.id,
            struct_name = layout.struct_name,
        ));
    }
    out.push_str(
        "        default: { return false; }\n\
         \x20   }\n\
         }\n",
    );
    out
}

fn rewrite_fn_name(fn_text: &str, from: &str, to: &str) -> String {
    // Replace only the first occurrence — the body may legitimately
    // mention the original name in a comment (e.g. self-referential
    // documentation) and we don't want to break that.
    fn_text.replacen(from, to, 1)
}
