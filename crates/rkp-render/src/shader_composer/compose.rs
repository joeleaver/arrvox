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

use super::types::{ComposedChunks, UserShaderRegistry};

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
    }
}

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
pub fn splice_const_marker(template: &str, marker_name: &str, chunk: &str) -> String {
    if chunk.is_empty() {
        return template.to_string();
    }
    let begin = format!("const {marker_name}_BEGIN: u32 = 0u;");
    let end = format!("const {marker_name}_END: u32 = 0u;");
    let begin_idx = template
        .find(&begin)
        .unwrap_or_else(|| panic!("template missing `{begin}` anchor"));
    let end_idx = template[begin_idx..]
        .find(&end)
        .map(|off| begin_idx + off + end.len())
        .unwrap_or_else(|| panic!("template missing `{end}` anchor"));
    let mut out = String::with_capacity(template.len() + chunk.len());
    out.push_str(&template[..begin_idx]);
    out.push_str(chunk);
    out.push_str(&template[end_idx..]);
    out
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

/// Phase B-redux. Compose the per-shader `instance_at` chunk for
/// splice into the host march / shadow templates between the
/// `USER_INSTANCE_AT_DISPATCH_BEGIN/END` markers.
///
/// Wire shape (per registered shader with an `instance_at` hook):
///
///   - The shader's instance struct decl (e.g. `struct Blade { ... }`)
///     plus its helper fns, captured verbatim.
///   - The user's `user_<name>_inst_aabb` and `user_<name>_inst_to_local`
///     bare functions, fn-names rewritten to `rkp_user_<id>_inst_aabb`
///     and `rkp_user_<id>_inst_to_local`. These are CALLED BY the
///     per-shader `rkp_user_<id>_instance_descend` body below — they
///     take the in-register `inst` struct directly (no pool read).
///   - The user's `user_<name>_instance_at` function, fn-name
///     rewritten to `rkp_user_<id>_instance_at`. Signature contract:
///
///     ```text
///     fn rkp_user_<id>_instance_at(
///         host_pos: vec3<f32>,
///         host: HostSample,
///         ctx: UserCtx,
///         k: u32,
///         out_instance: ptr<function, <Struct>>,
///     ) -> bool;
///     ```
///
///   - A per-shader `rkp_user_<id>_instance_descend(...)` function
///     wrapping the actual prototype-octree descent. Calls
///     `rkp_user_<id>_inst_aabb(inst)` for the world AABB cull and
///     `rkp_user_<id>_inst_to_local(world_pos, inst)` for the
///     world↔canonical map and Jacobian.
///
///   - A unified `dispatch_user_instance_descend(shader_id, ...)`
///     switch routing into the per-shader descend fns. Replaces the
///     identity-stub dispatcher in the template.
fn compose_instance_at_chunk(reg: &UserShaderRegistry) -> String {
    // Bail when no shader registers an `instance_at` hook — the
    // template's identity stub stays in place and the splice is a
    // no-op (empty chunk).
    if !reg.entries.iter().any(|e| e.instance_at_text.is_some()) {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("// ── user-shader bodies: instance_at ───────────────────\n");
    // Splice instance struct decls + helpers for every shader that has
    // an `instance_at` hook. This chunk is the SOLE emitter of the
    // per-shader bare functions (`rkp_user_<id>_inst_aabb`,
    // `rkp_user_<id>_inst_to_local`) and their type / helper context;
    // the dead `_at(base_u32, ...)` pool-read wrappers were dropped
    // along with the per-pixel Option B pipeline.
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
    // Bare `inst_aabb` / `inst_to_local` bodies — called directly by
    // the per-shader `instance_descend` with the in-register `inst`
    // struct. Validation upstream guarantees both are present alongside
    // any `instance_at`.
    for entry in &reg.entries {
        if entry.instance_at_text.is_some() {
            if let Some(text) = &entry.inst_aabb_text {
                let renamed = rewrite_fn_name(
                    text,
                    &format!("user_{}_inst_aabb", entry.name),
                    &format!("rkp_user_{}_inst_aabb", entry.id),
                );
                out.push_str(&renamed);
                out.push('\n');
            }
            if let Some(text) = &entry.inst_to_local_text {
                let renamed = rewrite_fn_name(
                    text,
                    &format!("user_{}_inst_to_local", entry.name),
                    &format!("rkp_user_{}_inst_to_local", entry.id),
                );
                out.push_str(&renamed);
                out.push('\n');
            }
        }
    }
    for entry in &reg.entries {
        if let Some(text) = &entry.instance_at_text {
            let renamed = rewrite_fn_name(
                text,
                &format!("user_{}_instance_at", entry.name),
                &format!("rkp_user_{}_instance_at", entry.id),
            );
            out.push_str(&renamed);
            out.push('\n');
        }
    }

    // Phase 2.c-2 — per-shader descent body. For each derived
    // instance (k = 0..max_emits):
    //
    //   1. `rkp_user_<id>_instance_at` derives the k-th instance
    //      struct.
    //   2. `rkp_user_<id>_inst_aabb(inst)` returns the world AABB.
    //      Ray-AABB cull rejects rays missing the bound.
    //   3. `rkp_user_<id>_inst_to_local` maps `world_entry` and
    //      `world_entry + ray_dir` into proto-canonical space; the
    //      difference is `local_dir_unnorm` whose length is the
    //      world↔local scale factor.
    //   4. `descend_proto_octree` walks the prototype octree from
    //      that local origin/direction, returns first opaque hit.
    //   5. World-normal via Jacobian: 4 more `inst_to_local` calls
    //      around the hit position, normal transforms by `Jᵀ`.
    //
    // `world_t` accumulates as `world_t_entry + hit.local_t *
    // local_to_world` (since the proto descent walks oc-space along
    // `local_dir_unnorm`-normalized direction, the local-distance is
    // already aligned with world-distance scaled by 1/|local_dir_unnorm|).
    // Closest hit across all k wins.
    out.push_str(
        "\n// ── per-shader instance-descent bodies (Phase 2.c-2) ─\n",
    );
    for entry in &reg.entries {
        if entry.instance_at_text.is_some() {
            let layout = entry.instance_layout.as_ref().expect(
                "instance_at hook implies @instance_proto layout parsed",
            );
            let max_emits = entry.metadata.max_emits_per_thread.unwrap_or(1);
            out.push_str(&format!(
                "fn rkp_user_{id}_instance_descend(\n\
                \x20   host_pos: vec3<f32>,\n\
                \x20   host: HostSample,\n\
                \x20   ctx: UserCtx,\n\
                \x20   leaf_slot: u32,\n\
                \x20   ray_origin: vec3<f32>,\n\
                \x20   ray_dir: vec3<f32>,\n\
                \x20   world_max_t: f32,\n\
                \x20   asset: RkpAsset,\n\
                ) -> InstanceHit {{\n\
                \x20   var best: InstanceHit;\n\
                \x20   best.valid = false;\n\
                \x20   best.world_t = world_max_t;\n\
                \x20   best.world_pos = vec3<f32>(0.0);\n\
                \x20   best.world_normal = vec3<f32>(0.0, 1.0, 0.0);\n\
                \x20   for (var k: u32 = 0u; k < {max_emits}u; k = k + 1u) {{\n\
                \x20       var inst: {struct_name};\n\
                \x20       if (!rkp_user_{id}_instance_at(host_pos, host, ctx, k, &inst)) {{ continue; }}\n\
                \x20       let aabb = rkp_user_{id}_inst_aabb(inst);\n\
                \x20       let safe_world_dir = vec3<f32>(\n\
                \x20           select(ray_dir.x, select(-1e-10, 1e-10, ray_dir.x >= 0.0), abs(ray_dir.x) < 1e-10),\n\
                \x20           select(ray_dir.y, select(-1e-10, 1e-10, ray_dir.y >= 0.0), abs(ray_dir.y) < 1e-10),\n\
                \x20           select(ray_dir.z, select(-1e-10, 1e-10, ray_dir.z >= 0.0), abs(ray_dir.z) < 1e-10),\n\
                \x20       );\n\
                \x20       let aabb_t = intersect_aabb(ray_origin, 1.0 / safe_world_dir, aabb.min, aabb.max);\n\
                \x20       if (aabb_t.x > aabb_t.y) {{ continue; }}\n\
                \x20       let world_t_entry = max(aabb_t.x, 0.0);\n\
                \x20       if (world_t_entry >= best.world_t) {{ continue; }}\n\
                \x20       let world_entry = ray_origin + ray_dir * world_t_entry;\n\
                \x20       let local_entry = rkp_user_{id}_inst_to_local(world_entry, inst);\n\
                \x20       let local_endpoint = rkp_user_{id}_inst_to_local(world_entry + ray_dir, inst);\n\
                \x20       let local_dir_unnorm = local_endpoint - local_entry;\n\
                \x20       let local_dir_len = max(length(local_dir_unnorm), 1.0e-8);\n\
                \x20       let local_dir = local_dir_unnorm / local_dir_len;\n\
                \x20       let local_to_world = 1.0 / local_dir_len;\n\
                \x20       let oc_origin = local_entry - asset.grid_origin;\n\
                \x20       let safe_dir = vec3<f32>(\n\
                \x20           select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),\n\
                \x20           select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),\n\
                \x20           select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),\n\
                \x20       );\n\
                \x20       let inv_dir = 1.0 / safe_dir;\n\
                \x20       let extent = bitcast<f32>(asset.octree_extent_bits);\n\
                \x20       let t_range = intersect_aabb(oc_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));\n\
                \x20       if (t_range.x > t_range.y) {{ continue; }}\n\
                \x20       // Cap descent in oc-space at the world-distance budget remaining vs. best hit.\n\
                \x20       let world_remaining = best.world_t - world_t_entry;\n\
                \x20       let local_t_cap = world_remaining / max(local_to_world, 1e-10);\n\
                \x20       let local_t_end = min(t_range.y, local_t_cap);\n\
                \x20       let hit = descend_proto_octree(\n\
                \x20           asset, oc_origin, safe_dir, inv_dir,\n\
                \x20           max(t_range.x, 0.0), local_t_end, local_to_world,\n\
                \x20           PROTO_PHASE,\n\
                \x20       );\n\
                \x20       if (!hit.valid) {{ continue; }}\n\
                \x20       let world_t = world_t_entry + hit.local_t * local_to_world;\n\
                \x20       if (world_t >= best.world_t) {{ continue; }}\n\
                \x20       let world_pos = ray_origin + ray_dir * world_t;\n\
                \x20       // Jacobian normal — see octree_march.wgsl::march_object for derivation.\n\
                \x20       let eps: f32 = 1.0e-3;\n\
                \x20       let l0 = rkp_user_{id}_inst_to_local(world_pos, inst);\n\
                \x20       let lx = rkp_user_{id}_inst_to_local(world_pos + vec3<f32>(eps, 0.0, 0.0), inst);\n\
                \x20       let ly = rkp_user_{id}_inst_to_local(world_pos + vec3<f32>(0.0, eps, 0.0), inst);\n\
                \x20       let lz = rkp_user_{id}_inst_to_local(world_pos + vec3<f32>(0.0, 0.0, eps), inst);\n\
                \x20       let jx = (lx - l0) / eps;\n\
                \x20       let jy = (ly - l0) / eps;\n\
                \x20       let jz = (lz - l0) / eps;\n\
                \x20       let n_local = hit.local_normal;\n\
                \x20       let n_world_unnorm = vec3<f32>(\n\
                \x20           dot(jx, n_local),\n\
                \x20           dot(jy, n_local),\n\
                \x20           dot(jz, n_local),\n\
                \x20       );\n\
                \x20       let n_world_len = length(n_world_unnorm);\n\
                \x20       var world_normal = n_local;\n\
                \x20       if (n_world_len >= 1e-6) {{\n\
                \x20           world_normal = n_world_unnorm / n_world_len;\n\
                \x20       }}\n\
                \x20       best.valid = true;\n\
                \x20       best.world_t = world_t;\n\
                \x20       best.world_pos = world_pos;\n\
                \x20       best.world_normal = world_normal;\n\
                \x20   }}\n\
                \x20   return best;\n\
                }}\n\n",
                id = entry.id,
                struct_name = layout.struct_name,
                max_emits = max_emits,
            ));
        }
    }

    // Unified dispatcher — replaces the in-template identity stub.
    // Per-shader cases route into rkp_user_<id>_instance_descend. The
    // `asset` arg threads the prototype's asset record (octree root,
    // depth, voxel_size, grid_origin) through to the per-shader
    // descent so it can run `descend_proto_octree`. The march call
    // site looks it up from the host hit's material → asset_id.
    out.push_str("// ── dispatch_user_instance_descend ───────────────────\n");
    out.push_str(
        "fn dispatch_user_instance_descend(\n\
         \x20   shader_id: u32,\n\
         \x20   host_pos: vec3<f32>,\n\
         \x20   host: HostSample,\n\
         \x20   leaf_slot: u32,\n\
         \x20   ray_origin: vec3<f32>,\n\
         \x20   ray_dir: vec3<f32>,\n\
         \x20   world_max_t: f32,\n\
         \x20   ctx: UserCtx,\n\
         \x20   asset: RkpAsset,\n\
         ) -> InstanceHit {\n\
         \x20   switch shader_id {\n",
    );
    for entry in &reg.entries {
        if entry.instance_at_text.is_some() {
            out.push_str(&format!(
                "        case {id}u: {{ return rkp_user_{id}_instance_descend(\n\
                 \x20           host_pos, host, ctx, leaf_slot, ray_origin, ray_dir, world_max_t, asset,\n\
                 \x20       ); }}\n",
                id = entry.id,
            ));
        }
    }
    out.push_str(
        "        default: {\n\
         \x20           var r: InstanceHit;\n\
         \x20           r.valid = false;\n\
         \x20           r.world_t = world_max_t;\n\
         \x20           r.world_pos = vec3<f32>(0.0);\n\
         \x20           r.world_normal = vec3<f32>(0.0, 1.0, 0.0);\n\
         \x20           return r;\n\
         \x20       }\n\
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
