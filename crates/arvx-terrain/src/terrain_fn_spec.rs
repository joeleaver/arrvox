//! Serializable selector for the active [`TerrainFn`].
//!
//! `Terrain.terrain_fn` is `Arc<dyn TerrainFn>` — a trait object that
//! can't be deserialized directly. The editor / scene-file pipeline
//! needs a concrete representation that round-trips: that's
//! `TerrainFnSpec`. V1 ships one variant (`Fbm`); future
//! TerrainFn impls (heightmap import, erosion-baked, node-graph)
//! will gain their own variants without breaking existing scenes.
//!
//! Convention: a `TerrainFnSpec` is the *authoritative* on-disk
//! description. `Arc<dyn TerrainFn>` is the runtime form built from
//! the spec at load time and edited via `From<&FbmTerrainFn>` after
//! Inspector edits.

use std::sync::Arc;

use crate::fbm::FbmTerrainFn;
use crate::terrain_fn::TerrainFn;

/// Concrete TerrainFn variants the scene file can carry. Add variants
/// for future TerrainFn impls; the V1 default is `Fbm`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerrainFnSpec {
    /// Built-in FBM heightmap source.
    Fbm(FbmTerrainFn),
}

impl Default for TerrainFnSpec {
    fn default() -> Self {
        Self::Fbm(FbmTerrainFn::default())
    }
}

impl TerrainFnSpec {
    /// Build the runtime trait object handed to the bake worker.
    pub fn to_dyn(&self) -> Arc<dyn TerrainFn> {
        match self {
            Self::Fbm(f) => Arc::new(*f),
        }
    }

    /// Best-effort capture of a runtime TerrainFn back into a spec.
    /// V1 only ships one impl; if the caller hands us something
    /// other than `FbmTerrainFn` we return `None` rather than guess.
    /// The Inspector path always knows the concrete type and uses
    /// the specific `from_*` constructors instead — this fallback
    /// is for safety in code paths that have only the trait object
    /// (none today).
    pub fn try_from_dyn(_terrain_fn: &dyn TerrainFn) -> Option<Self> {
        // Trait objects can't be downcast without `Any`; we don't
        // require `Any` on the trait because that pollutes every
        // future impl. Concrete-type-aware call sites are expected
        // to use `Fbm(*f)` directly. Returning None here makes the
        // caller fall back to the Terrain's own cached `spec` (set
        // at Inspector-write time).
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_fbm() {
        let s = TerrainFnSpec::default();
        match s {
            TerrainFnSpec::Fbm(f) => {
                assert_eq!(f, FbmTerrainFn::default());
            }
        }
    }

    #[test]
    fn serde_roundtrip_fbm() {
        let original = TerrainFnSpec::Fbm(FbmTerrainFn {
            seed: 1234,
            octaves: 3,
            scale_m: 75.0,
            amplitude_m: 12.5,
            base_height_m: 8.0,
            sea_level_y: -2.0,
            snow_level_y: 100.0,
            slope_rock_threshold_deg: 40.0,
            slope_probe_m: 0.5,
            grass_material: 1,
            rock_material: 3,
            snow_material: 4,
            sand_material: 2,
        });
        let json = serde_json::to_string(&original).unwrap();
        let back: TerrainFnSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn to_dyn_produces_callable_sampler() {
        let spec = TerrainFnSpec::default();
        let dyn_fn = spec.to_dyn();
        let s = dyn_fn.sample(
            crate::tile_key::TileKey::level0(0, 0, 0),
            glam::Vec3::ZERO,
            0.25,
        );
        // Trivial sanity — the sample doesn't panic and produces a
        // finite SD value.
        assert!(s.sd.is_finite());
    }

    /// New variants must extend the enum without breaking deserialise
    /// of old "kind: fbm" scenes. The tag-based serde format makes
    /// this trivial; this test pins the expected JSON shape so a
    /// future refactor doesn't silently change the wire format.
    #[test]
    fn json_shape_is_tag_named_kind() {
        let spec = TerrainFnSpec::default();
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""kind":"fbm""#), "got {json}");
    }
}
