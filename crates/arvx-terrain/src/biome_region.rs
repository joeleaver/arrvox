//! `BiomeRegion` — terrain's first region-data component.
//!
//! A [`BiomeRegion`] is attached to an entity that already carries the
//! cross-cutting [`arvx_regions::Region`] component. Where the
//! `Region` answers "is this point inside, and how much?", the
//! `BiomeRegion` answers "and if so, what does that mean for the
//! terrain at this point?" — namely an optional `TerrainFn` override
//! and an optional primary-material override, blended across
//! overlapping regions by membership weight.
//!
//! Phase 6 ships the **struct** only. Terrain bake doesn't consume it
//! yet — that's Phase 7, which extends `bake_tile` to query
//! `RegionIndex` per voxel and blend the overrides into the base
//! `TerrainFn`. Putting the struct in place now means the rest of the
//! editor integration (Inspector entry, gizmo, save/load) can land
//! without churn at the Phase 7 boundary.
//!
//! ## Why this lives in `arvx-terrain`, not `arvx-regions`
//!
//! `terrain_fn_override` is an `Option<Arc<dyn TerrainFn>>`. `TerrainFn`
//! is defined in `arvx-terrain` because it's terrain-specific. Putting
//! `BiomeRegion` in `arvx-regions` would create a circular dependency
//! (regions → terrain for `TerrainFn`; terrain → regions for `Region`).
//! Future consumer components (`AmbientAudio`, `FogVolume`,
//! `GameplayTrigger`) follow the same rule: define them in whichever
//! crate owns the trait/data they reference.

use std::sync::Arc;

use crate::terrain_fn::TerrainFn;

/// Data attached to a region entity that the terrain bake pipeline
/// consumes (in Phase 7).
///
/// Both fields are optional:
///
/// * `terrain_fn_override` — replace the base [`TerrainFn`] within
///   this region. Multiple overlapping biomes blend their TerrainFn
///   outputs weighted by [`arvx_regions::membership`]; ties between
///   single-valued properties resolve by `Region.priority`.
/// * `material_override` — force the primary material within this
///   region. Single-valued (a leaf has exactly one primary material),
///   so `Region.priority` arbitrates overlap.
///
/// Material IDs are the same `u16` index used everywhere else in
/// arvx (see `Stamp.material_override`); `None` means "fall through
/// to whatever the base TerrainFn / stamps decided."
#[derive(Clone, Default)]
pub struct BiomeRegion {
    /// Optional override of the base procedural source within this
    /// region. `None` keeps the global TerrainFn. `Arc` so the bake
    /// worker can clone the handle cheaply for off-thread sampling.
    pub terrain_fn_override: Option<Arc<dyn TerrainFn>>,
    /// Optional override of the primary material id within this
    /// region. Phase 7 will blend with overlapping biomes using
    /// [`arvx_regions::Region::priority`].
    pub material_override: Option<u16>,
}

impl std::fmt::Debug for BiomeRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BiomeRegion")
            .field("has_terrain_fn_override", &self.terrain_fn_override.is_some())
            .field("material_override", &self.material_override)
            .finish()
    }
}

impl BiomeRegion {
    /// True if this biome overrides nothing — equivalent to having no
    /// `BiomeRegion` at all. Inspector adds default-empty instances;
    /// authors fill them in field-by-field.
    pub fn is_empty(&self) -> bool {
        self.terrain_fn_override.is_none() && self.material_override.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let b = BiomeRegion::default();
        assert!(b.is_empty());
        assert!(b.material_override.is_none());
        assert!(b.terrain_fn_override.is_none());
    }

    #[test]
    fn material_override_makes_non_empty() {
        let b = BiomeRegion {
            material_override: Some(7),
            ..Default::default()
        };
        assert!(!b.is_empty());
        assert_eq!(b.material_override, Some(7));
    }
}
