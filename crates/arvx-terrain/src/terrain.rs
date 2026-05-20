//! `Terrain` ECS component — singleton per scene.
//!
//! Phase 1: just the struct, not yet wired into any scene/inspector.
//! Phase 2: gains `terrain_fn` + `render_radius_m`, consumed by the
//! `TileStreamer` to materialise tiles around the camera.
//! Phase 9 (editor integration) registers it with the editor and wires
//! the Inspector / viewport toolbar / save-scene path.

use crate::bounds::TerrainBounds;
use crate::fbm::FbmTerrainFn;
use crate::stamp_index::{StampIndex, StampIndexHandle};
use crate::terrain_fn::TerrainFn;
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
    /// Procedural source. Defaults to a vanilla [`FbmTerrainFn`].
    /// Phase 9 will swap this from the Inspector. `Arc` makes job
    /// submission cheap.
    pub terrain_fn: Arc<dyn TerrainFn>,
    /// Layer-2 stamp index — cached mirror of the world's `Stamp` ECS
    /// entities owned by this Terrain. The engine rebuilds this
    /// snapshot on stamp add/move/delete and shares it across bake
    /// jobs via `Arc`. Tiles compose stamps over the Layer-1
    /// `terrain_fn` output during bake.
    pub stamps: StampIndexHandle,
    /// Camera-centric residency radius in metres. Tiles whose centre
    /// is within this distance from the camera (and inside bounds)
    /// are materialised; tiles beyond are evicted.
    ///
    /// Default 192 m ≈ a 3-tile radius around the camera. Small enough
    /// to make tile streaming visible in a normal editor session,
    /// large enough that motion doesn't constantly trip eviction.
    pub render_radius_m: f32,
}

impl std::fmt::Debug for Terrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Terrain")
            .field("bounds", &self.bounds)
            .field("base_tier", &self.base_tier)
            .field("render_radius_m", &self.render_radius_m)
            .field("stamps_count", &self.stamps.len())
            .field("terrain_fn", &"<Arc<dyn TerrainFn>>")
            .finish()
    }
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            bounds: TerrainBounds::default(),
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
            terrain_fn: Arc::new(FbmTerrainFn::default()),
            stamps: Arc::new(StampIndex::new()),
            render_radius_m: 192.0,
        }
    }
}

impl Terrain {
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
}
