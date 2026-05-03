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
//!
//! Validation: `@instance_proto Foo` requires a matching `struct Foo`
//! AND a `user_<stem>_proto` hook; `instance_at` requires its
//! companion `inst_aabb` + `inst_to_local`. Reject fast — the user
//! sees a clear error instead of a downstream WGSL link failure.

use std::path::{Path, PathBuf};

use crate::instance_proto::parse_instance_layout;

use super::hash::compute_registry_hash;
use super::types::{
    ParamDef, ShaderComposerError, ShaderMetadata, UserShaderEntry, UserShaderRegistry,
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
