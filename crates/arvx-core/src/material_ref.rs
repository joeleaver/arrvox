//! Stable, palette-reorder-safe references to materials.
//!
//! Procedural sources (e.g. `arvx-terrain`'s `FbmTerrainFn`) and other
//! authoritative on-disk descriptions need to refer to a material in a
//! way that survives palette reorders, renames-at-the-slot-level, and
//! cross-project copies. A bare `u16` slot id ties the data to a
//! specific palette layout — `FbmTerrainFn::rock_material = 3` only
//! means "rock" in the palette that happens to put rock at slot 3.
//!
//! [`MaterialRef`] solves this by separating the *identity* of the
//! material (a project-relative path) from the *runtime slot id* it
//! resolves to. The path variant is what's authored and serialized;
//! the slot variant is for explicit slot binding (tests, generated
//! data, and back-compat with old scene files written before this
//! refactor — bare `u16`s round-trip as `MaterialRef::Slot`).
//!
//! Resolution goes through a [`MaterialLibraryLookup`] trait so the
//! `arvx-core` crate doesn't have to know about
//! `arvx-engine::MaterialLibrary`. The engine's library implements
//! the trait; tests use [`NullMaterialLookup`] (always `None`).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// How something refers to a material.
///
/// `Path` is the authored form — a project-root-relative path like
/// `"assets/materials/rock.arvxmat"`. `Slot` is an explicit numeric
/// slot binding; used by tests, generated content, and to read back
/// existing scenes that were authored before path-based refs existed.
///
/// Serde shape uses `#[serde(untagged)]` so JSON stays terse:
/// * `"assets/materials/rock.arvxmat"` → `Path(...)`
/// * `7` → `Slot(7)`
///
/// Untagged serde picks the FIRST variant that matches; we put `Slot`
/// first so a JSON integer is unambiguously a slot id (a bare integer
/// can't be coerced to a string path, but reversing the order would
/// silently turn slot ids into invalid string paths).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaterialRef {
    /// Pre-resolved slot id; primarily for tests, generated content,
    /// and back-compat with scene files that serialized bare `u16`s
    /// before the `Path` variant existed.
    Slot(u16),
    /// Project-root-relative path to a `.arvxmat` file (e.g.
    /// `"assets/materials/rock.arvxmat"`). Stable across palette
    /// reorders and across projects that share the same material
    /// filenames.
    Path(PathBuf),
}

impl MaterialRef {
    /// Convenience: build a `Path` variant from anything that's a
    /// `Path`-like type.
    pub fn path(p: impl Into<PathBuf>) -> Self {
        Self::Path(p.into())
    }

    /// Resolve to a concrete slot id. `Slot(n)` returns `n` unchanged.
    /// `Path(p)` calls `lookup.resolve_path(p)`; missing paths fall
    /// back to slot 0 (the built-in default opaque material) and
    /// trigger a warn-once console message per unique missing path —
    /// resolving the same missing path many times during a tile bake
    /// won't spam.
    pub fn resolve(&self, lookup: &dyn MaterialLibraryLookup) -> u16 {
        match self {
            Self::Slot(id) => *id,
            Self::Path(p) => match lookup.resolve_path(p) {
                Some(id) => id,
                None => {
                    warn_missing_path_once(p);
                    0
                }
            },
        }
    }
}

impl Default for MaterialRef {
    /// Defaults to the built-in default slot (0). Callers building
    /// procedural sources should override with a meaningful `Path`.
    fn default() -> Self {
        Self::Slot(0)
    }
}

impl From<u16> for MaterialRef {
    fn from(id: u16) -> Self {
        Self::Slot(id)
    }
}

/// Resolve a [`MaterialRef::Path`] to a runtime slot id.
///
/// Implementors: `arvx-engine::MaterialLibrary` provides the live
/// path-to-slot map. The path passed in is the project-root-relative
/// form authored in `MaterialRef::Path` — the implementor is
/// responsible for joining it against the project's root (or
/// equivalent base) before looking it up.
pub trait MaterialLibraryLookup {
    /// Return the slot id for `path` (project-root-relative), or
    /// `None` if the path doesn't correspond to any loaded material.
    fn resolve_path(&self, path: &Path) -> Option<u16>;
}

/// Lookup that always returns `None`. Used by `Terrain::default()` and
/// tests that don't care about real material resolution. Every
/// `MaterialRef::Path` resolves to slot 0 via the warn-once fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullMaterialLookup;

impl MaterialLibraryLookup for NullMaterialLookup {
    fn resolve_path(&self, _path: &Path) -> Option<u16> {
        None
    }
}

// ── warn-once tracking ─────────────────────────────────────────────

/// Tracks which `MaterialRef::Path` lookups have already failed this
/// session so the warning fires once per unique missing path, not
/// once per voxel sample. The fast path is a single relaxed
/// atomic-bool load; the slow path takes a mutex only the first time
/// a particular missing path is observed.
static SEEN_MISSING: AtomicBool = AtomicBool::new(false);
static SEEN_MISSING_PATHS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

fn warn_missing_path_once(path: &Path) {
    // Fast path: a per-path dedup would be expensive on every voxel.
    // We accept that the first miss across the whole session may race
    // and produce two log lines; subsequent misses for the SAME path
    // are deduped under the mutex.
    if !SEEN_MISSING.load(Ordering::Relaxed) {
        SEEN_MISSING.store(true, Ordering::Relaxed);
    }
    if let Ok(mut seen) = SEEN_MISSING_PATHS.lock() {
        if seen.iter().any(|p| p == path) {
            return;
        }
        seen.push(path.to_owned());
        eprintln!(
            "[MaterialRef] '{}' not found in material library; \
             falling back to slot 0 (default opaque). Add the file \
             to <project>/assets/materials/ to fix.",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapLookup(HashMap<PathBuf, u16>);

    impl MaterialLibraryLookup for MapLookup {
        fn resolve_path(&self, path: &Path) -> Option<u16> {
            self.0.get(path).copied()
        }
    }

    #[test]
    fn slot_resolves_to_itself() {
        let lookup = NullMaterialLookup;
        assert_eq!(MaterialRef::Slot(7).resolve(&lookup), 7);
    }

    #[test]
    fn known_path_resolves_to_slot() {
        let mut m = HashMap::new();
        m.insert(PathBuf::from("assets/materials/rock.arvxmat"), 5);
        let lookup = MapLookup(m);
        let r = MaterialRef::path("assets/materials/rock.arvxmat");
        assert_eq!(r.resolve(&lookup), 5);
    }

    #[test]
    fn missing_path_falls_back_to_slot_zero() {
        let lookup = NullMaterialLookup;
        let r = MaterialRef::path("nope.arvxmat");
        assert_eq!(r.resolve(&lookup), 0);
    }

    /// Untagged serde: a JSON string is `Path`, a JSON number is `Slot`.
    #[test]
    fn serde_string_is_path() {
        let r: MaterialRef = serde_json::from_str(r#""rock.arvxmat""#).unwrap();
        assert_eq!(r, MaterialRef::Path(PathBuf::from("rock.arvxmat")));
        // And round-trip:
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, r#""rock.arvxmat""#);
    }

    #[test]
    fn serde_number_is_slot() {
        let r: MaterialRef = serde_json::from_str("3").unwrap();
        assert_eq!(r, MaterialRef::Slot(3));
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(s, "3");
    }

    /// Back-compat: scene files authored before this refactor have
    /// bare `u16` material slots in their FBM JSON. Those must
    /// deserialize as `MaterialRef::Slot(..)` so old scenes keep
    /// loading.
    #[test]
    fn back_compat_bare_u16_is_slot() {
        let json = r#"{"grass_material": 1, "rock_material": 3}"#;
        #[derive(Deserialize)]
        struct Wrap {
            grass_material: MaterialRef,
            rock_material: MaterialRef,
        }
        let w: Wrap = serde_json::from_str(json).unwrap();
        assert_eq!(w.grass_material, MaterialRef::Slot(1));
        assert_eq!(w.rock_material, MaterialRef::Slot(3));
    }

    #[test]
    fn from_u16_yields_slot() {
        let r: MaterialRef = 12u16.into();
        assert_eq!(r, MaterialRef::Slot(12));
    }
}
