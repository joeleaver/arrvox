//! Reserved-name table for user-shader collision detection.
//!
//! Under `ManglerKind::None`, every root-level identifier in the
//! emitted artifact must be unique. A user-shader helper / struct /
//! const that shares a name with a lib symbol triggers a duplicate-
//! declaration error from naga at pipeline-create time, with no
//! attribution back to the offending user shader.
//!
//! We pre-empt that by scanning the embedded lib sources at composer
//! init and rejecting collisions during `parse_file`. Lib sources are
//! embedded at compile time with `include_str!`, so the reserved
//! set is always exactly in sync with the on-disk lib state — no
//! manual list to keep updated.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Every lib `*.wesl` file embedded by name. Update this list when
/// lib gains or loses a module; the names parameter only needs to be
/// distinct (used for diagnostics).
const LIB_SOURCES: &[(&str, &str)] = &[
    ("atmosphere", include_str!("../shaders/lib/atmosphere.wesl")),
    ("brick", include_str!("../shaders/lib/brick.wesl")),
    ("leaf_attr", include_str!("../shaders/lib/leaf_attr.wesl")),
    ("math", include_str!("../shaders/lib/math.wesl")),
    ("oct_normal", include_str!("../shaders/lib/oct_normal.wesl")),
    ("octree", include_str!("../shaders/lib/octree.wesl")),
    ("octree_slot", include_str!("../shaders/lib/octree_slot.wesl")),
    ("pbr", include_str!("../shaders/lib/pbr.wesl")),
    ("sdf", include_str!("../shaders/lib/sdf.wesl")),
    ("types", include_str!("../shaders/lib/types.wesl")),
];

/// True iff `name` is the name of a top-level declaration in any
/// lib `*.wesl` module. Used by `parse_file` to reject user-shader
/// helpers / structs that would collide post-splice.
pub fn is_lib_symbol(name: &str) -> bool {
    lib_symbols().contains(name)
}

fn lib_symbols() -> &'static HashSet<String> {
    static SYMBOLS: OnceLock<HashSet<String>> = OnceLock::new();
    SYMBOLS.get_or_init(|| {
        let mut set = HashSet::new();
        for (_, src) in LIB_SOURCES {
            for name in collect_lib_decls(src) {
                set.insert(name);
            }
        }
        set
    })
}

/// Extract every top-level `fn`, `struct`, and `const` declaration
/// name from a single lib source. Comments are skipped; declarations
/// are only recognized when they start at column 0 (matches the
/// existing lib coding style).
fn collect_lib_decls(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        // Skip comment-only / blank lines fast.
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        // Top-level decls in lib files start at column 0 — anything
        // indented is inside a struct or function body.
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("fn ") {
            if let Some(name) = ident_prefix(rest) {
                out.push(name.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("struct ") {
            if let Some(name) = ident_prefix(rest) {
                out.push(name.to_string());
            }
        } else if let Some(rest) = trimmed.strip_prefix("const ") {
            if let Some(name) = ident_prefix(rest) {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn ident_prefix(s: &str) -> Option<&str> {
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(s.len());
    if end == 0 {
        None
    } else {
        Some(&s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_up_known_lib_names() {
        // A canary for each lib module so the symbol set is exercised
        // against every embedded source.
        for name in [
            "PI",
            "SQRT3",
            "SKY_DEPTH_SENTINEL",
            "intersect_aabb",
            "GpuMaterial",
            "ArvxInstance",
            "OctreeResult",
            "BRICK_DIM",
            "OCTREE_EMPTY",
            "octree_lookup",
            "fetch_leaf_attr_for",
            "fresnel_dielectric",
            "beer_absorption",
            "sdf_sphere",
            "PROC_MISS_DISTANCE",
            "lookup_transmittance",
            "unpack_oct_normal",
        ] {
            assert!(
                is_lib_symbol(name),
                "expected `{name}` to be a registered lib symbol",
            );
        }
    }

    #[test]
    fn rejects_non_lib_names() {
        for name in ["my_helper", "MyStruct", "USER_PARAM", "fn"] {
            assert!(
                !is_lib_symbol(name),
                "did not expect `{name}` to be a lib symbol",
            );
        }
    }
}
