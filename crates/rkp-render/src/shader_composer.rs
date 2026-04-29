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

use crate::instance_proto::{parse_instance_layout, InstanceLayout, WgslType};

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
    /// back to a per-object default (e.g. host's voxel size). The
    /// region's max octree depth is derived from this — sim solves
    /// `extent / (cell_size * 4) = 2^depth` and clamps against
    /// `max_depth` below.
    pub cell_size: Option<f32>,
    /// Cap on the V9 sparse-BFS octree depth. `None` falls back to
    /// the engine default. Hard ceiling is `MAX_DEPTH = 8` (mirrors
    /// the WGSL constant + queue buffer sizing); requests above that
    /// silently clamp.
    pub max_depth: Option<u32>,
    /// V10 multi-region tiling. When set, the engine's auto-scan
    /// splits the painted area into `tile_size`-edge cubes (in
    /// host-local space) and emits one region per tile that contains
    /// any painted leaves. Each tile's cube is fixed-extent, so cell
    /// size is independent of paint area — `cell_size = tile_size /
    /// (4 × 2^max_depth)` regardless of how big a patch the user
    /// paints. None falls back to V9 single-region behaviour
    /// (one region per (object, material) covering the painted-leaf
    /// AABB; cell size grows with paint extent).
    pub tile_size: Option<f32>,
    /// `@instance_proto <StructName>` — opt-in for the Option B voxel
    /// sprite instancing pipeline. When `Some`, the file MUST also
    /// contain the named struct declaration plus the `user_<stem>_proto`
    /// and `user_<stem>_emit` hooks; the parsed struct layout lives on
    /// the [`UserShaderEntry`]. None means the shader uses the existing
    /// per-cell `generate` pipeline.
    pub instance_proto_struct: Option<String>,
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
    /// Option B — `fn user_<stem>_proto(uvw: vec3<f32>) -> VoxelEmit`,
    /// the prototype shape used by every instance the shader emits.
    /// Required when `metadata.instance_proto_struct` is `Some`.
    pub proto_text: Option<String>,
    /// Option B — `fn user_<stem>_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx)`,
    /// the per-host-sample instance scatter. Required when
    /// `metadata.instance_proto_struct` is `Some`.
    pub emit_text: Option<String>,
    /// Option B — optional override for non-affine deformation.
    /// `fn user_<stem>_inst_aabb(inst: <Struct>) -> Aabb`. Falls back
    /// to engine-derived AABB (`pos + rotated/scaled prototype AABB`)
    /// when absent.
    pub inst_aabb_text: Option<String>,
    /// Option B — optional override for non-affine deformation.
    /// `fn user_<stem>_inst_to_local(world_pos: vec3<f32>, inst: <Struct>) -> vec3<f32>`.
    /// Falls back to TRS inverse when absent.
    pub inst_to_local_text: Option<String>,
    /// Verbatim `struct ... { ... }` declarations captured from the
    /// file's top level, in source order. Shader code can declare its
    /// own helper structs; the engine splices them all back into the
    /// generated WGSL alongside the hooks. The instance struct (if any)
    /// is one of these.
    pub struct_decls: Vec<String>,
    /// Parsed layout of the per-instance state struct named by
    /// `metadata.instance_proto_struct`. Populated alongside the entry
    /// when the shader opts into Option B.
    pub instance_layout: Option<InstanceLayout>,
}

impl UserShaderEntry {
    /// Whether this shader contributes any dispatchable hook. Shaders
    /// with neither hook are legal (the file might just be header-only
    /// for now) but the dispatcher won't call into them.
    fn has_any_hook(&self) -> bool {
        self.shade_text.is_some()
            || self.generate_text.is_some()
            || self.proto_text.is_some()
            || self.emit_text.is_some()
    }

    /// Routes a fully-formed instance shader (Option B): has a parsed
    /// per-instance struct layout AND both required hooks. Used by the
    /// engine to dispatch this shader through the instance pipeline
    /// instead of the per-cell `generate` path.
    pub fn is_instance_pipeline(&self) -> bool {
        self.instance_layout.is_some()
            && self.proto_text.is_some()
            && self.emit_text.is_some()
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
                max_depth: e.metadata.max_depth,
                tile_size: e.metadata.tile_size,
                has_shade: e.shade_text.is_some(),
                has_generate: e.generate_text.is_some(),
                is_instance_pipeline: e.is_instance_pipeline(),
                instance_struct_name: e
                    .metadata
                    .instance_proto_struct
                    .clone(),
                instance_struct_size: e
                    .instance_layout
                    .as_ref()
                    .map(|l| l.total_size),
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
    pub max_depth: Option<u32>,
    pub tile_size: Option<f32>,
    pub has_shade: bool,
    pub has_generate: bool,
    /// True if the shader opts into Option B (voxel sprite instancing).
    /// Mutually exclusive with the per-cell `generate` path at dispatch
    /// time; the editor surfaces this so users can see at a glance which
    /// pipeline a shader belongs to.
    pub is_instance_pipeline: bool,
    /// Name of the per-instance struct (from `@instance_proto`) if any.
    pub instance_struct_name: Option<String>,
    /// Byte size of the per-instance struct, if parsed. Helpful for
    /// editor visibility into "am I close to the soft/hard limit?"
    pub instance_struct_size: Option<u32>,
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
        proto_text: None,
        emit_text: None,
        inst_aabb_text: None,
        inst_to_local_text: None,
        struct_decls: Vec::new(),
        instance_layout: None,
    };

    // Walk the file linearly, dispatching on whichever keyword (`fn` or
    // `struct`) comes next at top level. Comments are skipped by
    // `find_top_level_keyword`, so `// fn faux()` and `// struct Faux`
    // never produce false positives.
    let mut cursor = 0usize;
    loop {
        let next_fn = find_top_level_keyword("fn", source, cursor);
        let next_struct = find_top_level_keyword("struct", source, cursor);
        let (kind, item_start) = match (next_fn, next_struct) {
            (None, None) => break,
            (Some(f), None) => ("fn", f),
            (None, Some(s)) => ("struct", s),
            (Some(f), Some(s)) => {
                if f < s {
                    ("fn", f)
                } else {
                    ("struct", s)
                }
            }
        };

        if kind == "fn" {
            let after_kw = item_start + 2;
            let name_start = skip_ws(source, after_kw);
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
            let fn_text = source[item_start..=body_close].to_string();

            if let Some(hook) = fn_name.strip_prefix(&prefix) {
                let slot = match hook {
                    "shade" => &mut entry.shade_text,
                    "generate" => &mut entry.generate_text,
                    "proto" => &mut entry.proto_text,
                    "emit" => &mut entry.emit_text,
                    "inst_aabb" => &mut entry.inst_aabb_text,
                    "inst_to_local" => &mut entry.inst_to_local_text,
                    other => {
                        return Err(ShaderComposerError::Parse {
                            path: path.to_path_buf(),
                            line: line_of(source, name_start),
                            msg: format!(
                                "unknown hook `{other}` — expected `shade`, `generate`, `proto`, `emit`, `inst_aabb`, or `inst_to_local`"
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
        } else {
            // `struct` declaration: capture verbatim from `struct` keyword
            // through the matching `}`. Not validated here — user may
            // declare helper structs unrelated to @instance_proto.
            let after_kw = item_start + "struct".len();
            let name_start = skip_ws(source, after_kw);
            if name_start >= source.len() {
                break;
            }
            let name_end = source[name_start..]
                .find(|c: char| !is_ident(c))
                .map(|off| name_start + off)
                .unwrap_or(source.len());
            let struct_name = &source[name_start..name_end];
            let Some(body_open) = find_open_brace(source, name_end) else {
                return Err(ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_of(source, name_start),
                    msg: format!("struct `{struct_name}` has no body"),
                });
            };
            let body_close = match_brace(source, body_open).ok_or_else(|| {
                ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_of(source, body_open),
                    msg: format!("unmatched `{{` in body of struct `{struct_name}`"),
                }
            })?;
            let struct_text = source[item_start..=body_close].to_string();
            entry.struct_decls.push(struct_text);
            cursor = body_close + 1;
        }
    }

    // Validate / parse the @instance_proto target now that all fns +
    // structs are captured. Errors here are user-facing — they wrote
    // `@instance_proto Blade` but skipped one of the required pieces.
    if let Some(target) = entry.metadata.instance_proto_struct.clone() {
        if entry.proto_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "@instance_proto declared but `user_{name}_proto` hook is missing"
                ),
            });
        }
        if entry.emit_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "@instance_proto declared but `user_{name}_emit` hook is missing"
                ),
            });
        }
        let Some(struct_text) = entry
            .struct_decls
            .iter()
            .find(|t| struct_decl_name(t) == target)
            .cloned()
        else {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "@instance_proto target `{target}` — no matching `struct {target} {{ ... }}` in this file"
                ),
            });
        };
        let layout = parse_instance_layout(path, &target, &struct_text).map_err(|e| {
            ShaderComposerError::Parse {
                path: e.path,
                line: 0,
                msg: e.msg,
            }
        })?;
        entry.instance_layout = Some(layout);
    } else {
        // No @instance_proto directive — the instance hooks are reserved
        // names that don't make sense outside Option B. Reject so the
        // user gets a clear error instead of silent no-op.
        if entry.proto_text.is_some() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "`user_{name}_proto` defined without `// @instance_proto <StructName>` directive"
                ),
            });
        }
        if entry.emit_text.is_some() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "`user_{name}_emit` defined without `// @instance_proto <StructName>` directive"
                ),
            });
        }
        if entry.inst_aabb_text.is_some() || entry.inst_to_local_text.is_some() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: "instance helper hook defined without `// @instance_proto <StructName>` directive"
                    .to_string(),
            });
        }
    }

    let _ = entry.has_any_hook(); // entries with zero hooks are legal
    Ok(entry)
}

/// Pull the struct's name out of a captured `struct <Name> { ... }` block.
/// Returns "" if the text doesn't match that shape — callers are expected
/// to feed only text we just captured, so the empty-string fallback only
/// fires when the directive's target genuinely doesn't match any struct.
fn struct_decl_name(struct_text: &str) -> &str {
    let trimmed = struct_text.trim_start();
    let after = match trimmed.strip_prefix("struct") {
        Some(s) => s,
        None => return "",
    };
    let rest = after.trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    &rest[..end]
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
            "max_depth" => {
                let v: u32 = args.trim().parse().map_err(|_| ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("@max_depth expects a u32, got `{args}`"),
                })?;
                if v > 8 {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!(
                            "@max_depth {v} exceeds V9 ceiling of 8 (queue buffer is sized for MAX_DEPTH=8)"
                        ),
                    });
                }
                md.max_depth = Some(v);
            }
            "tile_size" => {
                md.tile_size = Some(parse_f32(path, line_no, "tile_size", args)?);
            }
            "instance_proto" => {
                let target = args.trim();
                if target.is_empty() {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: "@instance_proto requires a struct name (e.g. `// @instance_proto Blade`)".to_string(),
                    });
                }
                if !target.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    || !target.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!("@instance_proto target `{target}` is not a valid identifier"),
                    });
                }
                if md.instance_proto_struct.is_some() {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: "@instance_proto declared twice in this file".to_string(),
                    });
                }
                md.instance_proto_struct = Some(target.to_string());
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
    /// Spliced into the prototype-bake compute shader (Option B).
    /// Defines `dispatch_user_proto(shader_id, uvw) -> VoxelEmit`.
    /// Routes only shaders with `@instance_proto` directives; identity
    /// default arm returns a skip emit.
    pub proto: String,
    /// Spliced into the per-region instance-emit compute shader
    /// (Option B). Defines per-shader `rkp_user_<id>_emit_instance`
    /// fns (with bitcast writes derived from the parsed
    /// [`InstanceLayout`]) plus `dispatch_user_emit(shader_id,
    /// host_pos, host, ctx)`. The user's `emit` body has its
    /// `emit_instance(` calls textually rewritten to the per-shader
    /// generated form.
    pub emit: String,
    /// Spliced into `user_shader_instance_march_main.wgsl` between
    /// the `USER_INST_TO_LOCAL_DISPATCH_*` markers. Defines per-shader
    /// pool-read wrappers + `dispatch_user_inst_to_local(shader_id,
    /// base_u32, world_pos, fallback_pos, fallback_scale)`. Identity
    /// default arm calls `inst_world_to_local` (translate + uniform
    /// scale).
    pub inst_to_local: String,
    /// Spliced into `user_shader_instance_march_main.wgsl` between
    /// the `USER_INST_AABB_DISPATCH_*` markers. Defines per-shader
    /// pool-read wrappers + `dispatch_user_inst_aabb(shader_id,
    /// base_u32, fallback_pos, fallback_scale) -> Aabb`. Identity
    /// default arm returns `pos ± 0.5 × scale × √3` (covers any
    /// rotation of the canonical [0, 1]³ cube).
    pub inst_aabb: String,
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
        proto: compose_proto_chunk(reg),
        emit: compose_emit_chunk(reg),
        inst_to_local: compose_inst_to_local_chunk(reg),
        inst_aabb: compose_inst_aabb_chunk(reg),
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

fn compose_emit_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader helpers + bodies: emit ───────────────\n");
    // Splice instance struct decls + helper structs + helper fns from
    // each instance shader. Helper fn bodies get `emit_instance(` calls
    // rewritten to the per-shader generated form so user helpers can
    // also place instances.
    for entry in &reg.entries {
        if entry.emit_text.is_some() {
            for sd in &entry.struct_decls {
                out.push_str(sd);
                out.push('\n');
            }
            for helper in &entry.helpers {
                out.push_str(&rewrite_emit_instance_calls(helper, entry.id));
                out.push('\n');
            }
        }
    }
    // Generate `rkp_user_<id>_emit_instance(<Struct>)` per shader and
    // splice the user's emit body (renamed + with rewritten calls).
    for entry in &reg.entries {
        if let Some(emit_text) = &entry.emit_text {
            if let Some(layout) = &entry.instance_layout {
                out.push_str(&generate_emit_instance(entry.id, layout));
                out.push('\n');
            }
            let renamed = rewrite_fn_name(
                emit_text,
                &format!("user_{}_emit", entry.name),
                &format!("rkp_user_{}_emit", entry.id),
            );
            let rewritten = rewrite_emit_instance_calls(&renamed, entry.id);
            out.push_str(&rewritten);
            out.push('\n');
        }
    }
    out.push_str("\n// ── dispatch_user_emit ─────────────────────────────────\n");
    out.push_str(
        "fn dispatch_user_emit(shader_id: u32, host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.emit_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ rkp_user_{}_emit(host_pos, host, ctx); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str("        default: { }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

/// Generate the `rkp_user_<id>_emit_instance(<Struct>)` function for
/// one instance shader. Body: atomic-add a slot in the per-region
/// instance counter, then write the struct's fields into the global
/// `instance_pool` (u32-array) via per-field bitcasts at the offsets
/// the parsed [`InstanceLayout`] computed.
///
/// Reads workgroup-shared `region: EmitRegionUniform` and
/// `emit_region_index: u32` (set by `emit_main` before its barrier).
/// Overflow (atomic-add past `instance_block_size`) bumps a counter
/// in the `overflow` buffer at slot `OVERFLOW_INSTANCE = 0`.
fn generate_emit_instance(shader_id: u32, layout: &InstanceLayout) -> String {
    let mut out = String::new();
    // Stride in u32s, rounded up so byte alignment of the struct end
    // is preserved when consecutive instances pack tightly.
    let stride_u32 = layout.total_size.div_ceil(4);
    out.push_str(&format!(
        "fn rkp_user_{shader_id}_emit_instance(inst: {struct_name}) {{\n",
        struct_name = layout.struct_name,
    ));
    out.push_str("    let slot = atomicAdd(&instance_alloc[emit_region_index], 1u);\n");
    out.push_str("    if (slot >= region.instance_block_size) {\n");
    out.push_str("        atomicAdd(&overflow[OVERFLOW_INSTANCE], 1u);\n");
    out.push_str("        return;\n");
    out.push_str("    }\n");
    out.push_str(&format!(
        "    let base = region.instance_block_offset + slot * {stride_u32}u;\n"
    ));
    for field in &layout.fields {
        let u32_offset = field.byte_offset / 4;
        match field.ty {
            WgslType::F32 => out.push_str(&format!(
                "    instance_pool[base + {u32_offset}u] = bitcast<u32>(inst.{name});\n",
                name = field.name,
            )),
            WgslType::U32 => out.push_str(&format!(
                "    instance_pool[base + {u32_offset}u] = inst.{name};\n",
                name = field.name,
            )),
            WgslType::I32 => out.push_str(&format!(
                "    instance_pool[base + {u32_offset}u] = bitcast<u32>(inst.{name});\n",
                name = field.name,
            )),
            WgslType::Vec2F32 => {
                for (i, comp) in ["x", "y"].iter().enumerate() {
                    out.push_str(&format!(
                        "    instance_pool[base + {}u] = bitcast<u32>(inst.{}.{});\n",
                        u32_offset + i as u32,
                        field.name,
                        comp,
                    ));
                }
            }
            WgslType::Vec3F32 => {
                for (i, comp) in ["x", "y", "z"].iter().enumerate() {
                    out.push_str(&format!(
                        "    instance_pool[base + {}u] = bitcast<u32>(inst.{}.{});\n",
                        u32_offset + i as u32,
                        field.name,
                        comp,
                    ));
                }
            }
            WgslType::Vec4F32 => {
                for (i, comp) in ["x", "y", "z", "w"].iter().enumerate() {
                    out.push_str(&format!(
                        "    instance_pool[base + {}u] = bitcast<u32>(inst.{}.{});\n",
                        u32_offset + i as u32,
                        field.name,
                        comp,
                    ));
                }
            }
        }
    }
    out.push_str("}\n");
    out
}

/// Rewrite `emit_instance(` call sites in `text` to the per-shader
/// generated form. Conservative regex-style match: only call-syntax
/// (open paren attached) is rewritten — bare uses of `emit_instance`
/// as an identifier are left alone, so a user calling another helper
/// (e.g. `emit_instance_helper(...)`) doesn't get accidentally
/// rewritten.
fn rewrite_emit_instance_calls(text: &str, shader_id: u32) -> String {
    let target = "emit_instance(";
    let replacement = format!("rkp_user_{shader_id}_emit_instance(");
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i..].starts_with(target.as_bytes()) {
            // Make sure this isn't a substring of a longer identifier
            // (e.g. `not_emit_instance(...)`). Look at the byte
            // immediately preceding `i`: if it's an ident character,
            // skip the rewrite.
            let prev_is_ident = i > 0
                && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if !prev_is_ident {
                out.push_str(&replacement);
                i += target.len();
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
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

/// Generate a `var inst: <Struct>; inst.field = ...;` block that reads
/// the per-field bytes out of `instance_pool[base_u32 + offset]` into
/// a local. Mirrors [`generate_emit_instance`] but as READS, used by
/// the inst_to_local + inst_aabb wrappers to reconstruct the user's
/// struct from the global pool at march time.
fn generate_read_instance(layout: &InstanceLayout) -> String {
    let mut out = String::new();
    out.push_str(&format!("    var inst: {};\n", layout.struct_name));
    for field in &layout.fields {
        let u32_offset = field.byte_offset / 4;
        match field.ty {
            WgslType::F32 => out.push_str(&format!(
                "    inst.{name} = bitcast<f32>(instance_pool[base_u32 + {u32_offset}u]);\n",
                name = field.name,
            )),
            WgslType::U32 => out.push_str(&format!(
                "    inst.{name} = instance_pool[base_u32 + {u32_offset}u];\n",
                name = field.name,
            )),
            WgslType::I32 => out.push_str(&format!(
                "    inst.{name} = bitcast<i32>(instance_pool[base_u32 + {u32_offset}u]);\n",
                name = field.name,
            )),
            WgslType::Vec2F32 => {
                out.push_str(&format!(
                    "    inst.{name} = vec2<f32>(\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n    );\n",
                    u32_offset, u32_offset + 1, name = field.name,
                ));
            }
            WgslType::Vec3F32 => {
                out.push_str(&format!(
                    "    inst.{name} = vec3<f32>(\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n    );\n",
                    u32_offset, u32_offset + 1, u32_offset + 2, name = field.name,
                ));
            }
            WgslType::Vec4F32 => {
                out.push_str(&format!(
                    "    inst.{name} = vec4<f32>(\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n        bitcast<f32>(instance_pool[base_u32 + {}u]),\n    );\n",
                    u32_offset, u32_offset + 1, u32_offset + 2, u32_offset + 3, name = field.name,
                ));
            }
        }
    }
    out
}

fn compose_inst_to_local_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader bodies: inst_to_local ─────────────────\n");
    // Splice instance struct decls + helper structs from each shader
    // that has the inst_to_local hook (helpers + decls already came
    // through the proto/emit chunks, but the march template doesn't
    // include those — the chunk has to be self-contained).
    for entry in &reg.entries {
        if entry.inst_to_local_text.is_some() {
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
    // Per-shader user fn body (renamed) + pool-read wrapper.
    for entry in &reg.entries {
        if let Some(text) = &entry.inst_to_local_text {
            let layout = entry
                .instance_layout
                .as_ref()
                .expect("inst_to_local hook implies @instance_proto layout parsed");
            let renamed = rewrite_fn_name(
                text,
                &format!("user_{}_inst_to_local", entry.name),
                &format!("rkp_user_{}_inst_to_local", entry.id),
            );
            out.push_str(&renamed);
            out.push('\n');
            // Wrapper: read instance from pool at base_u32, call user fn.
            out.push_str(&format!(
                "fn rkp_user_{}_inst_to_local_at(base_u32: u32, world_pos: vec3<f32>) -> vec3<f32> {{\n",
                entry.id,
            ));
            out.push_str(&generate_read_instance(layout));
            out.push_str(&format!(
                "    return rkp_user_{}_inst_to_local(world_pos, inst);\n}}\n",
                entry.id,
            ));
        }
    }
    out.push_str("\n// ── dispatch_user_inst_to_local ───────────────────────\n");
    out.push_str(
        "fn dispatch_user_inst_to_local(shader_id: u32, base_u32: u32, world_pos: vec3<f32>, fallback_pos: vec3<f32>, fallback_scale: f32) -> vec3<f32> {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.inst_to_local_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ return rkp_user_{}_inst_to_local_at(base_u32, world_pos); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str(
        "        default: { return inst_world_to_local(world_pos, fallback_pos, fallback_scale); }\n",
    );
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

fn compose_inst_aabb_chunk(reg: &UserShaderRegistry) -> String {
    let mut out = String::new();
    out.push_str("// ── user-shader bodies: inst_aabb ─────────────────────\n");
    // Same self-contained-chunk pattern as inst_to_local. Note the
    // structs + helpers spliced here may DUPLICATE those spliced by
    // inst_to_local at the same template scope — naga rejects double
    // declaration. Skip them when inst_to_local already emitted.
    for entry in &reg.entries {
        if entry.inst_aabb_text.is_some() && entry.inst_to_local_text.is_none() {
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
        if let Some(text) = &entry.inst_aabb_text {
            let layout = entry
                .instance_layout
                .as_ref()
                .expect("inst_aabb hook implies @instance_proto layout parsed");
            let renamed = rewrite_fn_name(
                text,
                &format!("user_{}_inst_aabb", entry.name),
                &format!("rkp_user_{}_inst_aabb", entry.id),
            );
            out.push_str(&renamed);
            out.push('\n');
            out.push_str(&format!(
                "fn rkp_user_{}_inst_aabb_at(base_u32: u32) -> Aabb {{\n",
                entry.id,
            ));
            out.push_str(&generate_read_instance(layout));
            out.push_str(&format!(
                "    return rkp_user_{}_inst_aabb(inst);\n}}\n",
                entry.id,
            ));
        }
    }
    out.push_str("\n// ── dispatch_user_inst_aabb ───────────────────────────\n");
    out.push_str(
        "fn dispatch_user_inst_aabb(shader_id: u32, base_u32: u32, fallback_pos: vec3<f32>, fallback_scale: f32) -> Aabb {\n",
    );
    out.push_str("    switch shader_id {\n");
    for entry in &reg.entries {
        if entry.inst_aabb_text.is_some() {
            out.push_str(&format!(
                "        case {}u: {{ return rkp_user_{}_inst_aabb_at(base_u32); }}\n",
                entry.id, entry.id,
            ));
        }
    }
    out.push_str("        default: {\n");
    out.push_str("            let half = fallback_scale * 0.5 * 1.7320508;\n");
    out.push_str("            var a: Aabb;\n");
    out.push_str("            a.min = fallback_pos - vec3<f32>(half);\n");
    out.push_str("            a.max = fallback_pos + vec3<f32>(half);\n");
    out.push_str("            return a;\n");
    out.push_str("        }\n");
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
        for hook in [
            &e.shade_text,
            &e.generate_text,
            &e.proto_text,
            &e.emit_text,
            &e.inst_aabb_text,
            &e.inst_to_local_text,
        ] {
            if let Some(t) = hook {
                buf.extend_from_slice(t.as_bytes());
            }
            buf.push(0);
        }
        for helper in &e.helpers {
            buf.extend_from_slice(helper.as_bytes());
            buf.push(0);
        }
        for sd in &e.struct_decls {
            buf.extend_from_slice(sd.as_bytes());
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
        if let Some(d) = e.metadata.max_depth {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.push(0);
        if let Some(s) = e.metadata.tile_size {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.push(0);
        if let Some(name) = &e.metadata.instance_proto_struct {
            buf.extend_from_slice(name.as_bytes());
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
        assert!(!chunks.proto.contains("case "));
        assert!(chunks.proto.contains("default:"));
    }

    #[test]
    fn compose_emits_proto_chunk_for_instance_shaders() {
        let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32> }
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_grass_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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

    #[test]
    fn compose_emit_chunk_generates_emit_instance_writes() {
        let src = r#"
// @instance_proto Blade
struct Blade {
    pos: vec3<f32>,
    yaw: f32,
    sway_phase: f32,
    height_scale: f32,
    tint: u32,
}
fn user_grass_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_grass_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var b: Blade;
    b.pos = host_pos;
    b.yaw = 0.0;
    b.sway_phase = ctx.time;
    b.height_scale = 1.0;
    b.tint = 0u;
    emit_instance(b);
}
"#;
        let tmp = tempfile_dir("emit_codegen");
        write(&tmp, "grass.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let chunks = compose(&reg);
        // The struct decl is spliced.
        assert!(chunks.emit.contains("struct Blade"));
        // Per-shader emit_instance fn is generated and writes each field
        // via bitcast at the right u32 offset.
        assert!(chunks.emit.contains("fn rkp_user_1_emit_instance(inst: Blade)"));
        // Stride = ceil(32 / 4) = 8.
        assert!(chunks.emit.contains("slot * 8u"));
        // pos.x at u32 offset 0, .y at 1, .z at 2.
        assert!(chunks.emit.contains("instance_pool[base + 0u] = bitcast<u32>(inst.pos.x);"));
        assert!(chunks.emit.contains("instance_pool[base + 1u] = bitcast<u32>(inst.pos.y);"));
        assert!(chunks.emit.contains("instance_pool[base + 2u] = bitcast<u32>(inst.pos.z);"));
        // yaw f32 at byte 12 → u32 index 3.
        assert!(chunks.emit.contains("instance_pool[base + 3u] = bitcast<u32>(inst.yaw);"));
        // tint u32 at byte 24 → u32 index 6, no bitcast.
        assert!(chunks.emit.contains("instance_pool[base + 6u] = inst.tint;"));
        // The user's emit body has its `emit_instance(b)` call rewritten.
        assert!(chunks.emit.contains("rkp_user_1_emit_instance(b)"));
        assert!(!chunks.emit.contains(" emit_instance(b)"));
        // dispatch switch routes by shader_id.
        assert!(chunks.emit.contains("dispatch_user_emit"));
        assert!(chunks.emit.contains("rkp_user_1_emit(host_pos, host, ctx)"));
    }

    #[test]
    fn compose_emit_chunk_handles_optional_tagged_fields() {
        // pos required, rot + scale-as-f32 optional. Layout: pos at 0,
        // rot vec4 at 16, scale at 32. Total = 48 (rounded to vec4 align
        // 16). Stride u32 = 12.
        let src = r#"
// @instance_proto Tag
struct Tag {
    pos: vec3<f32>,
    rot: vec4<f32>,
    scale: f32,
}
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
"#;
        let tmp = tempfile_dir("emit_tagged");
        write(&tmp, "x.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let chunks = compose(&reg);
        assert!(chunks.emit.contains("slot * 12u"));
        // pos at u32 offsets 0..3
        assert!(chunks.emit.contains("instance_pool[base + 0u] = bitcast<u32>(inst.pos.x);"));
        // rot at byte 16 → u32 offsets 4..8
        assert!(chunks.emit.contains("instance_pool[base + 4u] = bitcast<u32>(inst.rot.x);"));
        assert!(chunks.emit.contains("instance_pool[base + 7u] = bitcast<u32>(inst.rot.w);"));
        // scale f32 at byte 32 → u32 offset 8
        assert!(chunks.emit.contains("instance_pool[base + 8u] = bitcast<u32>(inst.scale);"));
    }

    #[test]
    fn compose_emit_chunk_rewrites_helper_fn_calls() {
        // emit_instance() called from a USER HELPER must also be
        // rewritten so the helper places instances correctly.
        let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32> }

fn place_at(p: vec3<f32>) {
    var pt: Pt;
    pt.pos = p;
    emit_instance(pt);
}

fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    place_at(host_pos);
}
"#;
        let tmp = tempfile_dir("emit_helper_rewrite");
        write(&tmp, "x.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let chunks = compose(&reg);
        // The helper fn `place_at` is captured AND rewritten.
        assert!(chunks.emit.contains("fn place_at(p: vec3<f32>)"));
        assert!(chunks.emit.contains("rkp_user_1_emit_instance(pt)"));
    }

    #[test]
    fn compose_emit_chunk_does_not_rewrite_substrings() {
        // A user fn or comment containing `emit_instance` as a
        // substring (e.g. `not_emit_instance(`, `emit_instance_count`)
        // must NOT be rewritten — the rewriter only matches calls.
        let src = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32> }

// note: this comment mentions emit_instance(b) but should not be touched
fn not_emit_instance(p: vec3<f32>) -> f32 { return p.x; }

fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var ignored = not_emit_instance(host_pos);
}
"#;
        let tmp = tempfile_dir("emit_no_substring");
        write(&tmp, "x.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let chunks = compose(&reg);
        assert!(chunks.emit.contains("not_emit_instance"));
        // The rewriter mustn't have changed the helper fn name.
        assert!(!chunks.emit.contains("rkp_user_1_emit_instance(p)"));
    }

    #[test]
    fn compose_emit_chunk_empty_for_no_instance_shaders() {
        // A registry with no instance shaders must still emit a valid
        // dispatch_user_emit (with empty switch body).
        let src = r#"
fn user_holo_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
        let tmp = tempfile_dir("emit_empty");
        write(&tmp, "holo.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let chunks = compose(&reg);
        assert!(chunks.emit.contains("dispatch_user_emit"));
        assert!(!chunks.emit.contains("emit_instance"));
        assert!(!chunks.emit.contains("rkp_user_"));
    }

    // ── Option B: @instance_proto pipeline ────────────────────────────

    /// Canonical happy-path instance shader. Has the directive, struct,
    /// proto + emit hooks, plus an optional `inst_to_local` deformation
    /// helper. Should parse, populate `instance_layout`, and report
    /// `is_instance_pipeline()` = true.
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

fn user_grass_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var b: Blade;
    emit_instance(b);
}

fn user_grass_inst_to_local(world_pos: vec3<f32>, inst: Blade) -> vec3<f32> {
    return world_pos - inst.pos;
}
"#;
        let tmp = tempfile_dir("instance_full");
        write(&tmp, "grass.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let e = &reg.entries()[0];
        assert!(e.is_instance_pipeline());
        assert_eq!(e.metadata.instance_proto_struct.as_deref(), Some("Blade"));
        assert!(e.proto_text.is_some());
        assert!(e.emit_text.is_some());
        assert!(e.inst_to_local_text.is_some());
        assert!(e.inst_aabb_text.is_none());
        let layout = e.instance_layout.as_ref().unwrap();
        assert_eq!(layout.struct_name, "Blade");
        assert_eq!(layout.total_size, 32);
        assert_eq!(layout.fields.len(), 5);
    }

    /// `is_instance_pipeline()` must be false for plain shade-only shaders
    /// — they take the existing dispatch path and shouldn't be routed
    /// through the new pipeline.
    #[test]
    fn classic_shade_shader_is_not_instance_pipeline() {
        let src = r#"
fn user_holo_shade(ctx: ShadeCtx) -> ShadeResult { var r: ShadeResult; return r; }
"#;
        let tmp = tempfile_dir("classic_shade");
        write(&tmp, "holo.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let e = &reg.entries()[0];
        assert!(!e.is_instance_pipeline());
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
    fn rejects_instance_proto_without_emit_hook() {
        let src = r#"
// @instance_proto Blade
struct Blade { pos: vec3<f32> }
fn user_x_proto(uvw: vec3<f32>) -> VoxelEmit { var v: VoxelEmit; return v; }
"#;
        let tmp = tempfile_dir("instance_no_emit");
        write(&tmp, "x.wgsl", src);
        let err = scan_dir(&tmp).unwrap_err();
        match err {
            ShaderComposerError::Parse { msg, .. } => {
                assert!(msg.contains("`user_x_emit` hook is missing"), "got: {msg}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn rejects_proto_hook_without_directive() {
        // `proto`/`emit` are reserved for instance mode. Defining one
        // without `@instance_proto` is almost certainly a typo or
        // misunderstanding — fail loudly.
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
"#;
        let tmp = tempfile_dir("helper_struct");
        write(&tmp, "x.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let e = &reg.entries()[0];
        assert!(e.is_instance_pipeline());
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
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
fn user_x_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) { }
"#;
        let tmp = tempfile_dir("info");
        write(&tmp, "x.wgsl", src);
        let reg = scan_dir(&tmp).unwrap();
        let info = &reg.shader_infos()[0];
        assert!(info.is_instance_pipeline);
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
}
