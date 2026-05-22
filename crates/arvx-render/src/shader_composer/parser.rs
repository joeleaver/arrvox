//! WGSL source → [`UserShaderEntry`] / [`UserShaderRegistry`].
//!
//! Top-level entry points:
//! - [`scan_dir`] — scan a directory of `*.wgsl` files in alphabetical
//!   order, parse each, assign 1-based ids, hash the result.
//! - [`parse_file`] — parse a single shader source.
//!
//! The parser is hand-rolled (no naga / WGSL grammar dependency) — it
//! only needs to:
//!   - locate top-level `fn` and `struct` declarations
//!   - brace-match their bodies to capture verbatim text
//!   - skip line + block comments so `// fn faux` doesn't false-positive
//!   - recognize a few `// @<key>` header directives
//!
//! Hook routing: a function whose name matches `user_<file_stem>_<hook>`
//! is captured under the matching slot on [`UserShaderEntry`]. Other
//! top-level fns become helpers; structs become struct decls. Header
//! directives populate [`super::types::ShaderMetadata`].

use std::path::{Path, PathBuf};

use super::hash::compute_registry_hash;
use super::lib_symbols::is_lib_symbol;
use super::types::{
    GeometryDecl, ParamDef, ShaderComposerError, ShaderMetadata, SpawnCountCache,
    UserShaderEntry, UserShaderRegistry,
};

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

    // Build into a local Vec, then wrap in Arc once at the end —
    // avoids paying `Arc::make_mut`'s capacity check on every push.
    let mut entries: Vec<UserShaderEntry> = Vec::with_capacity(wgsl_files.len());
    for (idx, path) in wgsl_files.iter().enumerate() {
        let source = std::fs::read_to_string(path).map_err(|e| ShaderComposerError::Io {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        let mut entry = parse_file(path, &source)?;
        entry.id = (idx as u32) + 1;
        entries.push(entry);
    }
    reg.source_hash = compute_registry_hash(&entries);
    reg.entries = std::sync::Arc::new(entries);
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
        struct_decls: Vec::new(),
        spawn_count_text: None,
        spawn_alive_text: None,
        vs_text: None,
        fs_text: None,
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
            } else if matches!(fn_name, "vs" | "fs" | "spawn_count" | "spawn_alive") {
                // V1 mesh-path hooks — bare names (no `user_<stem>_`
                // prefix). Each shader gets its own pipeline so no
                // dispatch-switch renaming is needed.
                let slot = match fn_name {
                    "vs" => &mut entry.vs_text,
                    "fs" => &mut entry.fs_text,
                    "spawn_count" => &mut entry.spawn_count_text,
                    "spawn_alive" => &mut entry.spawn_alive_text,
                    _ => unreachable!(),
                };
                if slot.is_some() {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_of(source, name_start),
                        msg: format!("mesh-path hook `{fn_name}` defined twice in this file"),
                    });
                }
                *slot = Some(fn_text);
            } else {
                // Non-hook function — user-defined helper. Captured
                // verbatim so the hook body can call it. Reject if
                // it would collide with a lib symbol (post-splice
                // duplicate-decl in naga).
                if is_lib_symbol(fn_name) {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_of(source, name_start),
                        msg: format!(
                            "helper fn `{fn_name}` collides with a lib symbol — \
                             rename it (every root-level identifier in the emitted \
                             artifact must be unique under ManglerKind::None)"
                        ),
                    });
                }
                entry.helpers.push(fn_text);
            }

            cursor = body_close + 1;
        } else {
            // `struct` declaration: capture verbatim from `struct` keyword
            // through the matching `}`.
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
            if is_lib_symbol(struct_name) {
                return Err(ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: line_of(source, name_start),
                    msg: format!(
                        "struct `{struct_name}` collides with a lib symbol — \
                         rename it (every root-level identifier in the emitted \
                         artifact must be unique under ManglerKind::None)"
                    ),
                });
            }
            let struct_text = source[item_start..=body_close].to_string();
            entry.struct_decls.push(struct_text);
            cursor = body_close + 1;
        }
    }

    // Mesh-path completeness check. `@geometry` opts in; once opted
    // in, both `spawn_count` and `vs` are required.
    if entry.metadata.mesh_geometry.is_some() {
        if entry.spawn_count_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: "@geometry declared but `fn spawn_count(anchor, frame) -> u32` is missing".to_string(),
            });
        }
        if entry.vs_text.is_none() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: 0,
                msg: "@geometry declared but `fn vs(anchor, spawn_idx, vid, frame) -> VsOut` is missing".to_string(),
            });
        }
    } else if entry.spawn_count_text.is_some()
        || entry.spawn_alive_text.is_some()
        || entry.vs_text.is_some()
        || entry.fs_text.is_some()
    {
        return Err(ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: 0,
            msg: "mesh-path hook (vs / fs / spawn_count / spawn_alive) defined without `// @geometry` directive".to_string(),
        });
    }

    // Static-cache + per-frame reference is contradictory. Reject
    // shaders whose spawn_count body references `frame.` when the
    // cache is declared static — they'd silently use stale values.
    if entry.metadata.spawn_count_cache == SpawnCountCache::Static {
        if let Some(text) = &entry.spawn_count_text {
            if references_frame_context(text) {
                return Err(ShaderComposerError::Parse {
                    path: path.to_path_buf(),
                    line: 0,
                    msg: "`fn spawn_count` references `frame.` but @spawn_count_cache is `static` — \
                          declare `@spawn_count_cache per_frame` to read time-varying frame fields"
                        .to_string(),
                });
            }
        }
    }

    let _ = entry.has_any_hook(); // entries with zero hooks are legal
    Ok(entry)
}

/// Parse a `@geometry procedural { vertex_count: N }` or
/// `@geometry mesh { asset: "path" }` directive body.
fn parse_geometry_decl(
    path: &Path,
    line_no: usize,
    args: &str,
) -> Result<GeometryDecl, ShaderComposerError> {
    let trimmed = args.trim();
    if let Some(body) = trimmed
        .strip_prefix("procedural")
        .map(str::trim_start)
        .and_then(|s| s.strip_prefix('{').and_then(|b| b.strip_suffix('}')))
    {
        // body: "vertex_count: N" (V1 — index_count deferred)
        let mut vertex_count: Option<u32> = None;
        for kv in body.split(',') {
            let kv = kv.trim();
            if kv.is_empty() {
                continue;
            }
            let (k, v) = kv.split_once(':').ok_or_else(|| ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_no,
                msg: format!("@geometry procedural body: expected `key: value`, got `{kv}`"),
            })?;
            let k = k.trim();
            let v = v.trim();
            match k {
                "vertex_count" => {
                    let n: u32 = v.parse().map_err(|_| ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!("@geometry vertex_count expects a u32, got `{v}`"),
                    })?;
                    if n == 0 || n > 4096 {
                        return Err(ShaderComposerError::Parse {
                            path: path.to_path_buf(),
                            line: line_no,
                            msg: format!(
                                "@geometry vertex_count must be in 1..=4096 (got {n}); larger \
                                 procedural meshes should use `@geometry mesh {{ asset: ... }}` instead"
                            ),
                        });
                    }
                    vertex_count = Some(n);
                }
                other => {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: format!("unknown @geometry procedural key `{other}`"),
                    });
                }
            }
        }
        let vertex_count = vertex_count.ok_or_else(|| ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: line_no,
            msg: "@geometry procedural requires `vertex_count: N`".to_string(),
        })?;
        return Ok(GeometryDecl::Procedural { vertex_count });
    }
    if let Some(body) = trimmed
        .strip_prefix("mesh")
        .map(str::trim_start)
        .and_then(|s| s.strip_prefix('{').and_then(|b| b.strip_suffix('}')))
    {
        // body: `asset: "path"`
        let body = body.trim();
        let (k, v) = body.split_once(':').ok_or_else(|| ShaderComposerError::Parse {
            path: path.to_path_buf(),
            line: line_no,
            msg: "@geometry mesh body: expected `asset: \"path\"`".to_string(),
        })?;
        if k.trim() != "asset" {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_no,
                msg: format!("@geometry mesh: unknown key `{}`", k.trim()),
            });
        }
        let v = v.trim();
        let asset = v
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_no,
                msg: format!("@geometry mesh asset must be a quoted string, got `{v}`"),
            })?
            .to_string();
        if asset.is_empty() {
            return Err(ShaderComposerError::Parse {
                path: path.to_path_buf(),
                line: line_no,
                msg: "@geometry mesh asset path is empty".to_string(),
            });
        }
        return Ok(GeometryDecl::Mesh { asset });
    }
    Err(ShaderComposerError::Parse {
        path: path.to_path_buf(),
        line: line_no,
        msg: format!(
            "@geometry expects `procedural {{ vertex_count: N }}` or `mesh {{ asset: \"path\" }}`, got `{args}`"
        ),
    })
}

/// Heuristic check: does `text` reference the `frame` uniform? Used to
/// reject `@spawn_count_cache static` shaders whose spawn_count reads
/// frame-dependent state (which the static cache would freeze stale).
fn references_frame_context(text: &str) -> bool {
    // The `frame` parameter is bound in the engine prelude as an
    // identifier in scope; user code references `frame.time`,
    // `frame.wind_strength`, etc. A token-prefix check covers all
    // realistic cases without an AST.
    let mut search = text;
    while let Some(idx) = search.find("frame") {
        let before_ok = idx == 0
            || !search.as_bytes()[idx - 1].is_ascii_alphanumeric()
                && search.as_bytes()[idx - 1] != b'_';
        let after = idx + "frame".len();
        let after_ok = after < search.len() && search.as_bytes()[after] == b'.';
        if before_ok && after_ok {
            return true;
        }
        search = &search[idx + "frame".len()..];
    }
    false
}

/// Parse the prefix-of-source where `@`-directives live. Recognized:
///
/// ```text
/// // @param <name>: <type> = <default>, range = [<lo>, <hi>]
/// // @region_thickness <f32>
/// // @cell_size <f32>
/// // @animated
/// // @geometry procedural { vertex_count: N }   // V1 mesh-path
/// // @tile_size <f32>
/// // @max_distance <f32>      // world units, anchors past this are not uploaded
/// // @spawn_count_cache static | per_frame
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
            "max_distance" => {
                md.max_distance =
                    Some(parse_f32(path, line_no, "max_distance", args)?);
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
            "geometry" => {
                if md.mesh_geometry.is_some() {
                    return Err(ShaderComposerError::Parse {
                        path: path.to_path_buf(),
                        line: line_no,
                        msg: "@geometry declared twice in this file".to_string(),
                    });
                }
                md.mesh_geometry = Some(parse_geometry_decl(path, line_no, args)?);
            }
            "spawn_count_cache" => {
                md.spawn_count_cache = match args.trim() {
                    "static" => SpawnCountCache::Static,
                    "per_frame" => SpawnCountCache::PerFrame,
                    other => {
                        return Err(ShaderComposerError::Parse {
                            path: path.to_path_buf(),
                            line: line_no,
                            msg: format!(
                                "@spawn_count_cache expects `static` or `per_frame`, got `{other}`"
                            ),
                        });
                    }
                };
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

// ── Low-level scanner helpers ────────────────────────────────────────

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

/// Brace-balance scan from an opening `{` to its matching `}`.
///
/// The input is assumed to be syntactically valid WGSL inside the
/// function body. WGSL has no string or character literals and no
/// nested block comments, so the only constructs that can contain a
/// stray `{` or `}` are line (`//`) and block (`/* ... */`)
/// comments. Both forms are skipped here.
///
/// Multi-byte UTF-8 is safe to byte-scan: continuation bytes
/// (0x80..=0xBF) and lead bytes (0xC2+) never collide with the
/// 0x7B/0x7D bytes for `{`/`}` or with `/`/`*`.
///
/// Returns the index of the matching `}`, or `None` for unbalanced
/// input or an unterminated block comment.
fn match_brace(source: &str, open: usize) -> Option<usize> {
    debug_assert_eq!(source.as_bytes()[open], b'{');
    let bytes = source.as_bytes();
    let mut depth: i32 = 1;
    let mut i = open + 1;
    while i < bytes.len() {
        let b = bytes[i];
        // Line comment — skip to the newline. Anything inside is
        // inert (including a literal `/*` that would otherwise look
        // like a block-comment start).
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment — skip to the next `*/`. WGSL forbids nesting
        // so we only consume the first terminator. Unterminated
        // comment (no `*/` before EOF) is a malformed input — fail
        // closed by returning `None`.
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            let mut closed = false;
            while i + 1 < bytes.len() {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return None;
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
