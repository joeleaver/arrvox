//! `Terrain` ECS component — singleton per scene.
//!
//! Phase 1: just the struct, not yet wired into any scene/inspector.
//! Phase 2: gains `terrain_fn` + `render_radius_m`, consumed by the
//! `TileStreamer` to materialise tiles around the camera.
//! Phase 9 (editor integration) registers it with the editor and wires
//! the Inspector / viewport toolbar / save-scene path.

use crate::bounds::TerrainBounds;
use crate::region_snapshot::{TerrainRegionSnapshot, TerrainRegionSnapshotHandle};
use crate::stamp_index::{StampIndex, StampIndexHandle};
use crate::terrain_fn::TerrainFn;
use crate::terrain_fn_spec::TerrainFnSpec;
use arvx_core::{MaterialLibraryLookup, NullMaterialLookup};
use std::sync::Arc;

/// Per-scene terrain feature. Singleton enforced by the editor.
///
/// Carries the configuration that determines tile materialisation:
/// world bounds, base voxel-size tier, the procedural `TerrainFn`
/// source, and (in later phases) stamps and streaming policies.
///
/// Phase 2 dropped `Copy` in favour of `Clone`; `Arc<dyn TerrainFn>`
/// makes clones cheap.
#[derive(Clone)]
pub struct Terrain {
    /// World extent. Defaults to a 16 × 16 × 4 bounded grid; toggle to
    /// `Unbounded` for true infinite procedural worlds.
    pub bounds: TerrainBounds,
    /// `arvx_core::constants::RESOLUTION_TIERS` index for level-0
    /// tiles. Defaults to `DEFAULT_TERRAIN_TIER` (Tier 2 = 0.25 m in
    /// the unified pow2 table). The V2 LOD pyramid walks one tier
    /// coarser per LOD level (`base_tier + level`).
    pub base_tier: usize,
    /// Procedural source as a serializable spec — authoritative on
    /// disk. Phase 9 Inspector edits write here; `terrain_fn`
    /// stays in lockstep via `spec.to_dyn()`. On scene load we
    /// reconstruct `terrain_fn` from the loaded `spec`.
    pub spec: TerrainFnSpec,
    /// Procedural source (runtime form). Always equals
    /// `spec.to_dyn()`; the two fields move together on every
    /// write. Wrapped in `Arc` so bake jobs clone the pointer
    /// cheaply.
    pub terrain_fn: Arc<dyn TerrainFn>,
    /// Layer-2 stamp index — cached mirror of the world's `Stamp` ECS
    /// entities owned by this Terrain. The engine rebuilds this
    /// snapshot on stamp add/move/delete and shares it across bake
    /// jobs via `Arc`. Tiles compose stamps over the Layer-1
    /// `terrain_fn` output during bake.
    pub stamps: StampIndexHandle,
    /// Phase 7 biome regions snapshot — cached mirror of every
    /// `(Region, Transform)` entity in the scene (paired with optional
    /// `BiomeRegion` data). The engine rebuilds this on Region or
    /// BiomeRegion add/move/edit/delete; tiles read it during bake to
    /// apply per-voxel material overrides.
    pub regions: TerrainRegionSnapshotHandle,
    /// Camera-centric residency radius in metres. Tiles whose centre
    /// is within this distance from the camera (and inside bounds)
    /// are materialised; tiles beyond are evicted.
    ///
    /// Default 192 m ≈ a 3-tile radius around the camera. Small enough
    /// to make tile streaming visible in a normal editor session,
    /// large enough that motion doesn't constantly trip eviction.
    pub render_radius_m: f32,
    /// V2 LOD pyramid: number of LOD levels to materialise.
    /// `1` = level 0 only (V1 behavior, the default). `N` enables
    /// levels `0..N` with each level using a 2× coarser voxel size
    /// (one tier coarser per level via [`Self::voxel_size_for_level`]).
    /// The streamer assigns levels via geometric distance-banding so
    /// the inner-most band uses level 0 and the outer-most uses
    /// level `N - 1`; bands divide `[0, render_radius_m)`.
    ///
    /// Clamped to `[1, 8]` by the Inspector — higher values would
    /// saturate at Tier 0 (1 m voxels) via `voxel_size_for_level`'s
    /// floor.
    pub lod_levels: u8,
    /// V2 LOD pyramid: lateral skirt depth in metres. Each boundary
    /// surface vertex emits a thin vertical strip dropping by this
    /// many metres, masking the height-mismatch cracks between
    /// LOD-band neighbours. `0.0` disables skirts entirely.
    ///
    /// Default 4.0 m. Clamped to `[0.0, 64.0]` by the Inspector. The
    /// upper bound is one tile's worth — past that the skirt sticks
    /// out below the world, visible only to debug cameras.
    pub skirt_depth_m: f32,
}

impl std::fmt::Debug for Terrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Terrain")
            .field("bounds", &self.bounds)
            .field("base_tier", &self.base_tier)
            .field("render_radius_m", &self.render_radius_m)
            .field("lod_levels", &self.lod_levels)
            .field("skirt_depth_m", &self.skirt_depth_m)
            .field("stamps_count", &self.stamps.len())
            .field("regions_count", &self.regions.len())
            .field("terrain_fn", &"<Arc<dyn TerrainFn>>")
            .finish()
    }
}

impl Default for Terrain {
    fn default() -> Self {
        // No material library available here — `Path` refs in the
        // default FBM fall back to slot 0. The engine refreshes the
        // runtime trait object via `refresh_terrain_fn` as soon as
        // the library is wired (see `arvx-engine` tick).
        let spec = TerrainFnSpec::default();
        let terrain_fn = spec.to_dyn(&NullMaterialLookup);
        Self {
            bounds: TerrainBounds::default(),
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
            spec,
            terrain_fn,
            stamps: Arc::new(StampIndex::new()),
            regions: Arc::new(TerrainRegionSnapshot::new()),
            render_radius_m: 192.0,
            lod_levels: 1,
            skirt_depth_m: 4.0,
        }
    }
}

impl Terrain {
    /// Replace the procedural source. Updates both `spec` and the
    /// cached runtime `terrain_fn` in lockstep so subsequent bake
    /// jobs use the new source.
    ///
    /// `lookup` resolves any [`arvx_core::MaterialRef::Path`]s in the
    /// new spec to concrete slot ids at trait-object build time. The
    /// engine passes its `MaterialLibrary`; tests pass
    /// [`arvx_core::NullMaterialLookup`].
    pub fn set_spec(&mut self, spec: TerrainFnSpec, lookup: &dyn MaterialLibraryLookup) {
        self.terrain_fn = spec.to_dyn(lookup);
        self.spec = spec;
    }

    /// Mutate the spec in place (Inspector edits hit one field at a
    /// time) and refresh the runtime trait object. The closure
    /// returns whether it actually changed anything — `false`
    /// skips the trait-object rebuild.
    pub fn mutate_spec<F>(&mut self, f: F, lookup: &dyn MaterialLibraryLookup) -> bool
    where
        F: FnOnce(&mut TerrainFnSpec) -> bool,
    {
        let changed = f(&mut self.spec);
        if changed {
            self.terrain_fn = self.spec.to_dyn(lookup);
        }
        changed
    }

    /// Rebuild the runtime `terrain_fn` from the existing `spec` using
    /// `lookup`. Call this when the [`MaterialLibraryLookup`] changes
    /// (e.g. the engine's `MaterialLibrary` re-scanned a project) but
    /// the spec itself is unchanged — the resolved slot ids may now
    /// differ.
    pub fn refresh_terrain_fn(&mut self, lookup: &dyn MaterialLibraryLookup) {
        self.terrain_fn = self.spec.to_dyn(lookup);
    }

    /// World-Y of the bottom face of the terrain's solid envelope.
    /// Used by the bake to clamp the composed surface height so
    /// stamps (or a misbehaving TerrainFn) can't drive the entire
    /// footprint above the world's solid envelope and produce a
    /// fall-through hole. Returns `None` for `Unbounded` terrains
    /// (no floor — the user owns this concern manually).
    pub fn world_floor_y(&self) -> Option<f32> {
        match self.bounds {
            TerrainBounds::Bounded { origin, .. } => {
                Some(origin.origin_world().to_vec3().y)
            }
            TerrainBounds::Unbounded => None,
        }
    }

    /// Voxel size in metres for a tile at the given LOD level.
    ///
    /// V2 LOD-pyramid semantics: each LOD level *doubles* the voxel
    /// size (so the cell count per tile stays constant). In the
    /// unified pow2 table, doubling voxel size = stepping ONE tier
    /// COARSER (lower index). So `level=N` uses tier
    /// `base_tier - N`, saturated at 0 (Tier 0 = 1 m, the table's
    /// coarsest entry).
    pub fn voxel_size_for_level(&self, level: u8) -> f32 {
        use arvx_core::constants::RESOLUTION_TIERS;
        let idx = self.base_tier.saturating_sub(level as usize);
        RESOLUTION_TIERS[idx].voxel_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_voxel_size_at_level0_is_quarter_meter() {
        let t = Terrain::default();
        // DEFAULT_TERRAIN_TIER (Tier 2) in the unified pow2 table = 0.25 m.
        assert!((t.voxel_size_for_level(0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn level1_doubles_voxel_size_to_half_meter() {
        let t = Terrain::default();
        // Level 1 LOD: one tier coarser = Tier 1 = 0.5 m.
        assert!((t.voxel_size_for_level(1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn level2_doubles_again_to_one_meter() {
        let t = Terrain::default();
        // Level 2 LOD: two tiers coarser = Tier 0 = 1.0 m.
        assert!((t.voxel_size_for_level(2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deep_levels_saturate_at_coarsest_tier() {
        let t = Terrain::default();
        // Beyond Tier 0 we can't go coarser — saturate.
        assert!((t.voxel_size_for_level(99) - 1.0).abs() < 1e-6);
    }

    /// V1 of the L-pyramid opt-in: default `lod_levels = 1` preserves
    /// pre-pyramid behavior bit-identically (level 0 only).
    #[test]
    fn default_lod_levels_is_one() {
        let t = Terrain::default();
        assert_eq!(t.lod_levels, 1);
    }
}
