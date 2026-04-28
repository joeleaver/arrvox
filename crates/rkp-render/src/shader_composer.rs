//! User-shader composition for the deferred shade pass + GPU geometry pass.
//!
//! Scans `<project_root>/assets/shaders/*.wgsl`, parses each shader's
//! optional hook functions, and emits dispatch chunks that get spliced
//! into `rkp_shade.wgsl` (Phase B) and the geometry-build pipeline
//! (Phase C). Both pipelines use the same registry and same
//! `compose()` output structure.
//!
//! ## Authoring contract
//!
//! Each `*.wgsl` file is one shader, named by its file stem
//! (`assets/shaders/grass.wgsl` → "grass"). A shader provides up to
//! four hooks; the function name signals which hook:
//!
//! ```ignore
//! fn user_grass_pre(world_pos: vec3<f32>, ctx: UserCtx) -> vec3<f32>
//! fn user_grass_generate(world_pos: vec3<f32>, ctx: UserCtx) -> TreeSample
//! fn user_grass_post(child: TreeSample, world_pos: vec3<f32>, ctx: UserCtx) -> TreeSample
//! fn user_grass_envelope(ctx: UserCtx) -> f32
//! ```
//!
//! Hooks not present default to identity (`pre` returns `world_pos`,
//! `post` returns `child`, `envelope` returns `0`, `generate` returns
//! a miss). Files that declare no hooks are still legal — they're
//! registered but contribute no behavior.
//!
//! ## Composition strategy
//!
//! 1. Each user function is captured verbatim from the source file
//!    (full `fn ... { ... }` text, brace-matched).
//! 2. The function name `user_<name>_<hook>` is rewritten to
//!    `rkp_user_<id>_<hook>` so dispatch can call it by a stable name
//!    independent of the user's choice of `<name>`.
//! 3. Four `dispatch_user_*` switches are emitted, one per hook. Each
//!    switch routes by `shader_id` to the matching `rkp_user_<id>_<hook>`
//!    function; shaders that don't provide that hook fall through to
//!    the switch's default (identity).
//!
//! `shader_id` 0 is reserved for "no shader" — the default arms
//! return the identity behavior. Registered shaders get ids 1..=N in
//! filesystem-walk order.

use std::path::{Path, PathBuf};

/// One user-declared parameter: name, default, optional UI range. Built
/// from `// @param <name>: <type> = <default>, range = [<lo>, <hi>]`
/// header comments in the shader source.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParamDef {
    pub name: String,
    pub default: f32,
    pub range: Option<(f32, f32)>,
}

/// Per-shader metadata extracted from `// @<key> ...` comments at the top
/// of the source file (anything before the first `fn`). All fields are
/// optional with sensible defaults; missing headers don't error.
///
/// Editor controls are built from `params`; the `animated`,
/// `region_thickness`, and `cell_size` flags steer the geometry pass's
/// region-collection and cache logic in Phase C.
#[derive(Debug, Clone, Default)]
pub struct ShaderMetadata {
    /// Named parameter schema — order is the order they appear in the
    /// source. Materials store values keyed by name; the GPU param
    /// buffer packs them in this order.
    pub params: Vec<ParamDef>,
    /// How far from the host surface the geometry hook may emit voxels,
    /// in world units. Drives the bounding region used to size the
    /// per-(object, material) generation dispatch in Phase C.
    /// Default 0.0 — pure shade-pass shaders don't need it.
    pub region_thickness: f32,
    /// Opt-in: regenerate every frame instead of caching. For waving
    /// grass / fluttering hair / etc. Default false (cache by source +
    /// param hash).
    pub animated: bool,
    /// Preferred voxel resolution for the geometry pass. `None` falls
    /// back to a per-object default (e.g. host's voxel size).
    pub cell_size: Option<f32>,
    /// Preferred octree depth N for the geometry pass — the region's
    /// brick grid is `(4 * 2^N)` cells per axis. `None` falls back
    /// to the engine's default (V2: 2 → 16 cells/axis). Capped to
    /// 6 by the dispatcher; deeper requires sparse BFS.
    pub octree_depth: Option<u32>,
}

/// One user shader's parsed hook bodies + header metadata. Each `*_text`
/// field, when `Some`, is the full `fn ... { ... }` declaration as it
/// appeared in the source file (the function name is rewritten to the
/// dispatch form at emit time, not at capture).
///
/// Two hooks are recognized:
///   * `shade(ctx)` — Phase B; called per-pixel from the deferred shade
///     pass to override or augment PBR.
///   * `generate(cell_world_pos, host, ctx)` — Phase C; called from the
///     GPU geometry pipeline to emit voxels into the sidecar pool.
///
/// A shader may declare either or both hooks (or neither — empty is
/// legal, just contributes nothing). Helper functions (anything not
/// matching `user_<stem>_<hook>`) are captured into `helpers` and
/// emitted alongside the hook bodies so user code can call them.
#[derive(Debug, Clone)]
pub struct UserShaderEntry {
    /// File stem, used both as the on-disk shader name (what materials
    /// reference via `MaterialDef.shader`) and as the prefix the
    /// parser scans for in the source.
    pub name: String,
    /// Path the entry was loaded from. Stored so the editor can offer
    /// "open in external editor" and error messages include source
    /// locations.
    pub file_path: PathBuf,
    /// Numeric dispatch id, 1-based. Registry assigns these in scan
    /// order; resolved by `MaterialDef::to_gpu` into
    /// `GpuMaterial.shader_id`.
    pub id: u32,
    /// Header-comment metadata.
    pub metadata: ShaderMetadata,
    /// Captured fn declarations.
    pub shade_text: Option<String>,
    pub generate_text: Option<String>,
    /// User-defined helper functions (not hooks). Captured verbatim
    /// so hook bodies can call them. Identifier collisions across
    /// shaders are user-managed — pick unique helper names if
    /// loading multiple shaders together.
    pub helpers: Vec<String>,
}

impl UserShaderEntry {
    /// Whether this shader contributes any dispatchable hook. Shaders
    /// with neither hook are legal (the file might just be header-only
    /// for now) but the dispatcher won't call into them.
    fn has_any_hook(&self) -> bool {
        self.shade_text.is_some() || self.generate_text.is_some()
    }
}

/// Registry of all user shaders discovered in the project's
/// `assets/shaders/` directory. Built once per scan; consumers hold
/// references for the lifetime of the bake worker's current shader
/// generation, then a new registry replaces it on filesystem change.
#[derive(Debug, Clone, Default)]
pub struct UserShaderRegistry {
    entries: Vec<UserShaderEntry>,
    /// Stable hash of the concatenation of every entry's source text
    /// in deterministic (alphabetical) order. Bake outputs use this
    /// in their cache key so editing a `.wgsl` invalidates only
    /// dependent caches; callers compare hashes to skip no-op reloads.
    source_hash: u64,
}

impl UserShaderRegistry {
    /// An empty registry — equivalent to "no user shaders." Bake/dispatch
    /// behave as identity for every `shader_id`.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[UserShaderEntry] {
        &self.entries
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Names of all registered shaders, in id order.
    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// Build a snapshot-friendly view: just the metadata the editor
    /// needs (shader name, file path, param schema, flags). Excludes
    /// the captured fn bodies, which the editor never needs and would
    /// just inflate every snapshot.
    pub fn shader_infos(&self) -> Vec<UserShaderInfo> {
        self.entries
            .iter()
            .map(|e| UserShaderInfo {
                name: e.name.clone(),
                file_path: e.file_path.clone(),
                params: e.metadata.params.clone(),
                region_thickness: e.metadata.region_thickness,
                animated: e.metadata.animated,
                cell_size: e.metadata.cell_size,
                octree_depth: e.metadata.octree_depth,
                has_shade: e.shade_text.is_some(),
                has_generate: e.generate_text.is_some(),
            })
            .collect()
    }

    /// Resolve a `shader_name` (as stored on `MaterialDef.shader`) to
    /// the numeric dispatch id. `None` means "not registered" — material
    /// falls back to id=0 (identity), which is the default-arm of every
    /// dispatch switch.
    pub fn resolve(&self, name: &str) -> Option<u32> {
        if name.is_empty() {
            return None;
        }
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.id)
    }
}

/// Editor-facing snapshot of one registered shader. The shader_infos
/// endpoint produces these so the editor's material panel can build a
/// shader dropdown + dynamic param controls without hauling around the
/// captured WGSL bodies.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UserShaderInfo {
    pub name: String,
    pub file_path: PathBuf,
    pub params: Vec<ParamDef>,
    pub region_thickness: f32,
    pub animated: bool,
    pub cell_size: Option<f32>,
    pub octree_depth: Option<u32>,
    pub has_shade: bool,
    pub has_generate: bool,
}

/// Errors that can arise while scanning / parsing user shaders.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShaderComposerError {
    #[error("io error reading {path}: {msg}")]
    Io { path: PathBuf, msg: String },

    #[error("parse error in {path}:{line}: {msg}")]
    Parse {
        path: PathBuf,
        line: usize,
        msg: String,
    },
}

/// Scan a directory for `*.wgsl` files and build a registry. Files are
/// processed in alphabetical order (deterministic ids across runs;
/// stable cache keys). Subdirectories are not recursed into. Missing
/// directory yields an empty registry — projects without user shaders
/// are valid.
pub fn scan_dir(dir: &Path) -> Result<UserShaderRegistry, ShaderComposerError> {
    let mut reg = UserShaderRegistry::default();
    if !dir.exists() {
        return Ok(reg);
    }
    let read_dir = std::fs::read_dir(dir).map_err(|e| ShaderComposerError::Io {
        path: dir.to_path_buf(),
        msg: e.to_string(),
    })?;
    let mut wgsl_files: Vec<PathBuf> = Vec::new();
    for entry in read_dir {
        let entry = entry.map_err(|e| ShaderComposerError::Io {
            path: dir.to_path_buf(),
            msg: e.to_string(),
        })?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("wgsl"))
        {
            wgsl_files.push(path);
        }
    }
    wgsl_files.sort();

    for (idx, path) in wgsl_files.iter().enumerate() {
        let source = std::fs::read_to_string(path).map_err(|e| ShaderComposerError::Io {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let mut entry = parse_file(path, &source)?;
        entry.id = (idx as u32) + 1;
        reg.entries.push(entry);
    }
    reg.source_hash = compute_registry_hash(&reg.entries);
    Ok(reg)
}

/// Parse a single user-shader source file. Extracts:
///   * `// @<key> ...` header comments (anything before the first `fn`)
///     into [`ShaderMetadata`]
///   * `fn user_<stem>_<hook>` declarations matching one of the
///     recognized hooks (`shade`, `generate`)
///
/// The file stem is the shader name. Functions whose name doesn't
/// match the `user_<stem>_` prefix are tolerated as helpers (silently
/// dropped from the dispatch chunk — Phase B adds explicit "helper"
/// capture if shared utilities turn out to be needed). Functions
/// matching the prefix but with an unknown hook suffix reject with a
/// clear error so typos don't disappear.
pub fn parse_file(path: &Path, source: &str) -> Result<UserShaderEntry, ShaderComposerError> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: 0,
            msg: "filename has no UTF-8 stem".to_string(),
        })?
        .to_string();
    let prefix = format!("user_{name}_");

    // Headers live above the first `fn`. We scan that prefix for
    // `// @<key>` directives. Anything else (regular comments, blank
    // lines, struct decls — though shaders shouldn't have those at
    // file scope) is ignored.
    let metadata_end = find_top_level_keyword("fn", source, 0).unwrap_or(source.len());
    let metadata = parse_metadata(path, &source[..metadata_end])?;

    let mut entry = UserShaderEntry {
        name: name.clone(),
        file_path: path.to_path_buf(),
        id: 0,
        metadata,
        shade_text: None,
        generate_text: None,
        helpers: Vec::new(),
    };

    let mut cursor = 0usize;
    while cursor < source.len() {
        let Some(fn_start) = find_top_level_keyword("fn", source, cursor) else {
            break;
        };
        let after_fn = fn_start + 2;
        let name_start = skip_ws(source, after_fn);
        if name_start >= source.len() {
            break;
        }
        let name_end = source[name_start..]
            .find(|c: char| !is_ident(c))
            .map(|off| name_start + off)
            .unwrap_or(source.len());
        let fn_name = &source[name_start..name_end];

        let Some(body_open) = find_open_brace(source, name_end) else {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_of(source, name_start),
                msg: format!("function `{fn_name}` has no body"),
            });
        };
        let body_close = match_brace(source, body_open).ok_or_else(|| {
            ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_of(source, body_open),
                msg: format!("unmatched `{{` in body of `{fn_name}`"),
            }
        })?;
        let fn_text = source[fn_start..=body_close].to_string();

        if let Some(hook) = fn_name.strip_prefix(&prefix) {
            let slot = match hook {
                "shade" => &mut entry.shade_text,
                "generate" => &mut entry.generate_text,
                other => {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_of(source, name_start),
                        msg: format!(
                            "unknown hook `{other}` — expected `shade` or `generate`"
                        ),
                    });
                }
            };
            if slot.is_some() {
                return Err(ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_of(source, name_start),
                    msg: format!("hook `{hook}` defined twice in this file"),
                });
            }
            *slot = Some(fn_text);
        } else {
            // Non-hook function — user-defined helper. Captured
            // verbatim so the hook body can call it.
            entry.helpers.push(fn_text);
        }

        cursor = body_close + 1;
    }
    let _ = entry.has_any_hook(); // entries with zero hooks are legal
    Ok(entry)
}

/// Parse the prefix-of-source where `@`-directives live. Recognized:
///
/// ```text
/// // @param <name>: <type> = <default>, range = [<lo>, <hi>]
/// // @region_thickness <f32>
/// // @cell_size <f32>
/// // @animated
/// ```
///
/// Lines that aren't comments or aren't `@`-prefixed are skipped. Lines
/// that ARE `@`-prefixed but don't match a known directive reject with
/// a parse error — silent typo absorption is the failure mode this
/// plan calls out.
fn parse_metadata(
    path: &Path,
    source_prefix: &str,
) -> Result<ShaderMetadata, ShaderComposerError> {
    let mut md = ShaderMetadata::default();
    for (line_idx, raw_line) in source_prefix.lines().enumerate() {
        let line = raw_line.trim();
        // Only `// @...` lines carry directives. Plain `//` comments
        // and blank lines are fine. WGSL block comments `/* @ */` are
        // not parsed as directives — keep things one-line for clarity.
        let Some(rest) = line.strip_prefix("//") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('@') else {
            continue;
        };
        let line_no = line_idx + 1;
        // Split on the first whitespace to get the directive name.
        let (key, args) = match rest.find(char::is_whitespace) {
            Some(idx) => (&rest[..idx], rest[idx..].trim()),
            None => (rest, ""),
        };
        match key {
            "param" => md.params.push(parse_param_line(path, line_no, args)?),
            "region_thickness" => {
                md.region_thickness = parse_f32(path, line_no, "region_thickness", args)?;
            }
            "cell_size" => {
                md.cell_size = Some(parse_f32(path, line_no, "cell_size", args)?);
            }
            "octree_depth" => {
                let v: u32 = args.trim().parse().map_err(|_| ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("@octree_depth expects a u32, got `{args}`"),
                })?;
                if v > 6 {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!(
                            "@octree_depth {v} too deep — V4 dense bricks cap at 6 (sparse BFS needed for deeper)"
                        ),
                    });
                }
                md.octree_depth = Some(v);
            }
            "animated" => {
                if !args.is_empty() {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: "@animated takes no argument".to_string(),
                    });
                }
                md.animated = true;
            }
            other => {
                return Err(ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("unknown directive @{other}"),
                });
            }
        }
    }
    Ok(md)
}

/// Parse a `@param <name>: <type> = <default>, range = [<lo>, <hi>]`
/// line. The `: <type>` part is read but only `f32` is accepted today
/// (other scalar types may follow when the GPU param buffer grows).
/// `range` is optional.
fn parse_param_line(
    path: &Path,
    line_no: usize,
    args: &str,
) -> Result<ParamDef, ShaderComposerError> {
    let err = |msg: &str| ShaderComposerError::Parse {
        path: path.to_path_buf(),
        line: line_no,
        msg: msg.to_string(),
    };
    // Pull "name : type" out, before the first `=`.
    let (head, after_eq) = args.split_once('=').ok_or_else(|| {
        err("expected `<name>: <type> = <default>` after @param")
    })?;
    let (name, ty) = head.split_once(':').ok_or_else(|| {
        err("expected `:` between param name and type")
    })?;
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(err("@param name is empty"));
    }
    let ty = ty.trim();
    if ty != "f32" {
        return Err(ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: line_no,
            msg: format!("@param type `{ty}` not supported (only `f32` for now)"),
        });
    }
    // After the `=` we have "<default>" or "<default>, range = [lo, hi]".
    let (default_str, range_str) = match after_eq.find(',') {
        Some(idx) => (after_eq[..idx].trim(), Some(after_eq[idx + 1..].trim())),
        None => (after_eq.trim(), None),
    };
    let default: f32 = default_str.parse().map_err(|_| {
        ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: line_no,
            msg: format!("could not parse default `{default_str}` as f32"),
        }
    })?;
    let range = match range_str {
        Some(r) => {
            let r = r.trim();
            let r = r.strip_prefix("range").ok_or_else(|| {
                err("expected `, range = [lo, hi]` after default")
            })?;
            let r = r.trim_start().strip_prefix('=').ok_or_else(|| {
                err("expected `=` after `range`")
            })?;
            let r = r.trim().trim_start_matches('[').trim_end_matches(']');
            let (lo, hi) = r.split_once(',').ok_or_else(|| {
                err("range must be `[lo, hi]`")
            })?;
            let lo: f32 = lo.trim().parse().map_err(|_| {
                ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("range lo `{}` not f32", lo.trim()),
                }
            })?;
            let hi: f32 = hi.trim().parse().map_err(|_| {
                ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("range hi `{}` not f32", hi.trim()),
                }
            })?;
            Some((lo, hi))
        }
        None => None,
    };
    Ok(ParamDef { name, default, range })
}

fn parse_f32(
    path: &Path,
    line_no: usize,
    key: &str,
    args: &str,
) -> Result<f32, ShaderComposerError> {
    args.trim().parse().map_err(|_| ShaderComposerError::Parse {
        path: path.to_path_buf(),
        line: line_no,
        msg: format!("@{key} expects a single f32, got `{args}`"),
    })
}

/// Output of [`compose`] — one chunk per pipeline that consumes user
/// shaders. Each chunk is self-contained: rewritten user fn bodies for
/// that pipeline's hook, followed by a `dispatch_user_<hook>` switch
/// statement with an identity default arm. Pipelines splice the
/// matching chunk into their own WGSL between begin/end markers.
///
/// Both chunks share the same user-shader names + ids, so a single
/// material's `shader_id` correctly routes through both pipelines.
#[derive(Debug, Clone, Default)]
pub struct ComposedChunks {
    /// Spliced into `rkp_shade.wgsl` by the deferred shade pass.
    /// Defines `dispatch_user_shade(shader_id, ctx) -> ShadeResult`.
    pub shade: String,
    /// Spliced into the geometry-build compute shader. Defines
    /// `dispatch_user_generate(shader_id, cell_world_pos, host, ctx)
    /// -> VoxelEmit`.
    pub generate: String,
}

/// Compose the per-pipeline dispatch chunks. Returns identity-default
/// chunks when the registry is empty. Phase A: emits structurally
/// valid switch statements but the surrounding WGSL types
/// (`ShadeCtx`, `ShadeResult`, `HostSample`, `VoxelEmit`, `UserCtx`)
/// will land with their consuming pipelines in Phases B and C.
pub fn compose(reg: &UserShaderRegistry) -> ComposedChunks {
    ComposedChunks {
        shade: compose_shade_chunk(reg),
        generate: compose_generate_chunk(reg),
    }
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

// ── Parser helpers ──────────────────────────────────────────────────────

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn skip_ws(source: &str, mut i: usize) -> usize {
    while i < source.len() {
        let b = source.as_bytes()[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Skip line comment.
        if b == b'/' && i + 1 < source.len() && source.as_bytes()[i + 1] == b'/' {
            while i < source.len() && source.as_bytes()[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Skip block comment.
        if b == b'/' && i + 1 < source.len() && source.as_bytes()[i + 1] == b'*' {
            i += 2;
            while i + 1 < source.len() {
                if source.as_bytes()[i] == b'*' && source.as_bytes()[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        break;
    }
    i
}

/// Locate the next top-level occurrence of `keyword` at or after
/// `from`, skipping comments. Returns the byte offset of the keyword's
/// first character, or `None` if not found. Top-level here means "outside
/// any `{...}` block" — every input we feed is the file body, so we
/// don't need to track depth, but we DO need to skip comments to avoid
/// false matches inside `// fn faux()` and the like.
fn find_top_level_keyword(keyword: &str, source: &str, from: usize) -> Option<usize> {
    let mut i = from;
    while i < source.len() {
        let j = skip_ws(source, i);
        if j >= source.len() {
            return None;
        }
        if source[j..].starts_with(keyword) {
            // Must not be a substring of a longer identifier
            // (e.g. `function` shouldn't match `fn`).
            let after = j + keyword.len();
            let prev_is_ident = j > 0 && is_ident(source.as_bytes()[j - 1] as char);
            let next_is_ident =
                after < source.len() && is_ident(source.as_bytes()[after] as char);
            if !prev_is_ident && !next_is_ident {
                return Some(j);
            }
            i = after;
            continue;
        }
        // No match here — skip past this token (or single char).
        i = j + 1;
    }
    None
}

fn find_open_brace(source: &str, from: usize) -> Option<usize> {
    let mut i = from;
    while i < source.len() {
        let j = skip_ws(source, i);
        if j >= source.len() {
            return None;
        }
        if source.as_bytes()[j] == b'{' {
            return Some(j);
        }
        i = j + 1;
    }
    None
}

fn match_brace(source: &str, open: usize) -> Option<usize> {
    debug_assert_eq!(source.as_bytes()[open], b'{');
    let mut depth: i32 = 1;
    let mut i = open + 1;
    while i < source.len() {
        let b = source.as_bytes()[i];
        // Skip comments inside the body so braces in `// {` don't
        // throw the depth count off.
        if b == b'/' && i + 1 < source.len() && source.as_bytes()[i + 1] == b'/' {
            while i < source.len() && source.as_bytes()[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < source.len() && source.as_bytes()[i + 1] == b'*' {
            i += 2;
            while i + 1 < source.len() {
                if source.as_bytes()[i] == b'*' && source.as_bytes()[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn line_of(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset.min(source.len())]
        .bytes()
        .filter(|&b| b == b'\n')
        .count()
        + 1
}

fn rewrite_fn_name(fn_text: &str, from: &str, to: &str) -> String {
    // Replace only the first occurrence — the body may legitimately
    // mention the original name in a comment (e.g. self-referential
    // documentation) and we don't want to break that.
    fn_text.replacen(from, to, 1)
}

// ── Hashing ────────────────────────────────────────────────────────────

/// Deterministic FNV-1a 64. std's DefaultHasher uses a per-process
/// random seed, which would invalidate every cache on every restart;
/// FNV is keyless and stable across runs.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

fn compute_registry_hash(entries: &[UserShaderEntry]) -> u64 {
    let mut sorted: Vec<&UserShaderEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut buf = Vec::new();
    for e in sorted {
        buf.extend_from_slice(e.name.as_bytes());
        buf.push(0);
        for hook in [&e.shade_text, &e.generate_text] {
            if let Some(t) = hook {
                buf.extend_from_slice(t.as_bytes());
            }
            buf.push(0);
        }
        for helper in &e.helpers {
            buf.extend_from_slice(helper.as_bytes());
            buf.push(0);
        }
        // Metadata also contributes to the hash so a change to default
        // values / range / @animated invalidates dependent caches.
        for p in &e.metadata.params {
            buf.extend_from_slice(p.name.as_bytes());
            buf.push(0);
            buf.extend_from_slice(&p.default.to_le_bytes());
            if let Some((lo, hi)) = p.range {
                buf.extend_from_slice(&lo.to_le_bytes());
                buf.extend_from_slice(&hi.to_le_bytes());
            }
            buf.push(0);
        }
        buf.extend_from_slice(&e.metadata.region_thickness.to_le_bytes());
        buf.push(if e.metadata.animated { 1 } else { 0 });
        if let Some(s) = e.metadata.cell_size {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.push(0);
        if let Some(d) = e.metadata.octree_depth {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(0);
    }
    fnv1a_64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
}
