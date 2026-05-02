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

use crate::instance_proto::{parse_instance_layout, InstanceLayout};

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
    /// `@instance_proto <StructName>` — opt-in for the per-instance
    /// pipeline (Phase B-redux band-cell descent). When `Some`, the
    /// file MUST also contain the named struct declaration plus the
    /// `user_<stem>_proto` hook; the parsed struct layout lives on the
    /// [`UserShaderEntry`]. None means the shader uses the existing
    /// per-cell `generate` pipeline.
    pub instance_proto_struct: Option<String>,
    /// `@max_emits_per_thread <u32>` — per-host-position cap on how
    /// many instances `instance_at` may return for a single host hit
    /// before the dispatcher gives up. Uses 1 when absent. Hard
    /// ceiling: 16.
    pub max_emits_per_thread: Option<u32>,
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
    /// `fn user_<stem>_proto(uvw: vec3<f32>) -> VoxelEmit` — the
    /// prototype shape descended at march time from band-cell hits.
    /// Required when `metadata.instance_proto_struct` is `Some`.
    pub proto_text: Option<String>,
    /// `fn user_<stem>_inst_aabb(inst: <Struct>) -> Aabb` — instance
    /// world-space AABB. Required alongside `instance_at`.
    pub inst_aabb_text: Option<String>,
    /// `fn user_<stem>_inst_to_local(world_pos: vec3<f32>, inst: <Struct>) -> vec3<f32>`
    /// — world→prototype-local mapping. Required alongside `instance_at`.
    pub inst_to_local_text: Option<String>,
    /// Phase B-redux march-time derivation hook. Signature:
    /// `fn user_<stem>_instance_at(host_pos: vec3<f32>, host: HostSample,
    /// ctx: UserCtx, k: u32, out_instance: ptr<function, <Struct>>) -> bool`.
    /// Returns the k-th instance for this host position, or `false` to
    /// signal "no instance at index k." Called from the host march on
    /// band-cell hits; allows zero per-frame state writes.
    pub instance_at_text: Option<String>,
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
            || self.instance_at_text.is_some()
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
                has_instance_at: e.instance_at_text.is_some(),
                instance_struct_name: e
                    .metadata
                    .instance_proto_struct
                    .clone(),
                instance_struct_size: e
                    .instance_layout
                    .as_ref()
                    .map(|l| l.total_size),
                max_emits_per_thread: e.metadata.max_emits_per_thread,
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
    /// Phase B-redux — true if the shader provides a `instance_at`
    /// hook. Such shaders take the march-time derivation path
    /// (Phase 3a host-leaf dispatch + Phase 3b band-cell dispatch)
    /// instead of Option B's emit-into-instance-pool flow.
    pub has_instance_at: bool,
    /// Name of the per-instance struct (from `@instance_proto`) if any.
    pub instance_struct_name: Option<String>,
    /// Byte size of the per-instance struct, if parsed. Helpful for
    /// editor visibility into "am I close to the soft/hard limit?"
    pub instance_struct_size: Option<u32>,
    /// Phase 7b — per-thread emit cap. `None` falls back to 1.
    pub max_emits_per_thread: Option<u32>,
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
        inst_aabb_text: None,
        inst_to_local_text: None,
        instance_at_text: None,
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
                    "inst_aabb" => &mut entry.inst_aabb_text,
                    "inst_to_local" => &mut entry.inst_to_local_text,
                    "instance_at" => &mut entry.instance_at_text,
                    other => {
                        return Err(ShaderComposerError::Parse {
                            path: path.to_path_buf(),
                            line: line_of(source, name_start),
                            msg: format!(
                                "unknown hook `{other}` — expected `shade`, `generate`, `proto`, `inst_aabb`, `inst_to_local`, or `instance_at`"
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
        if entry.inst_aabb_text.is_some() || entry.inst_to_local_text.is_some() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: "instance helper hook defined without `// @instance_proto <StructName>` directive"
                    .to_string(),
            });
        }
        if entry.instance_at_text.is_some() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "`user_{name}_instance_at` defined without `// @instance_proto <StructName>` directive"
                ),
            });
        }
    }

    // Phase B-redux precondition: `instance_at` shaders must also
    // provide `inst_aabb` and `inst_to_local`. The march-time descent
    // calls all three on each derived instance — `inst_aabb` for the
    // ray-AABB cull, `inst_to_local` for the world↔canonical map and
    // the world-normal Jacobian. Reject early so the user gets a
    // clear error instead of a spurious WGSL link error at splice
    // time when the composer references a missing
    // `rkp_user_<id>_inst_aabb` symbol.
    if entry.instance_at_text.is_some() {
        if entry.inst_aabb_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "`user_{name}_instance_at` requires `user_{name}_inst_aabb` (the per-pixel descent calls it for ray-AABB cull)"
                ),
            });
        }
        if entry.inst_to_local_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: format!(
                    "`user_{name}_instance_at` requires `user_{name}_inst_to_local` (the per-pixel descent calls it for the world↔canonical map and Jacobian)"
                ),
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
/// // @max_emits_per_thread <u32>   // Option B; default 1
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
            "max_emits_per_thread" => {
                let v: u32 = args.trim().parse().map_err(|_| ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_no,
                    msg: format!("@max_emits_per_thread expects a u32, got `{args}`"),
                })?;
                if v == 0 || v > 16 {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!(
                            "@max_emits_per_thread must be in 1..=16 (got {v}); higher values bloat per-region pool reservations"
                        ),
                    });
                }
                md.max_emits_per_thread = Some(v);
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
    /// Spliced into the prototype-bake compute shader. Defines
    /// `dispatch_user_proto(shader_id, uvw) -> VoxelEmit`. Routes
    /// only shaders with `@instance_proto` directives; identity
    /// default arm returns a skip emit.
    pub proto: String,
    /// Phase B-redux — march-time derivation chunk. Spliced into the
    /// host march / shadow-trace templates between the
    /// `USER_INSTANCE_AT_DISPATCH_*` markers. Defines per-shader
    /// `rkp_user_<id>_instance_at(host_pos, host, ctx, k, &instance)
    /// -> bool` (verbatim user body, fn name rewritten). The march
    /// splices per-shader switch cases that call into these directly
    /// with the user's instance-struct-typed local var; no unified
    /// dispatcher because each shader's instance struct differs.
    /// Empty when no instance shaders register an `instance_at` hook.
    pub instance_at: String,
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
    splice_user_marker(
        template,
        concat!("USER_INSTANCE_AT_DISPATCH", "_BEGIN"),
        concat!("USER_INSTANCE_AT_DISPATCH", "_END"),
        instance_at_chunk,
    )
}

fn splice_user_marker(template: &str, begin: &str, end: &str, chunk: &str) -> String {
    if chunk.is_empty() {
        return template.to_string();
    }
    let begin_idx = template
        .find(begin)
        .unwrap_or_else(|| panic!("template missing {begin} marker"));
    let end_idx = template[begin_idx..]
        .find(end)
        .map(|off| begin_idx + off + end.len())
        .unwrap_or_else(|| panic!("template missing {end} marker"));
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
            &e.inst_aabb_text,
            &e.inst_to_local_text,
            &e.instance_at_text,
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
}
