//! `Terrain` ECS component — singleton per scene.
//!
//! Phase 1: just the struct, not yet wired into any scene/inspector.
//! Phase 9 (editor integration) registers it with the editor and wires
//! the Inspector / viewport toolbar / save-scene path.

use crate::bounds::TerrainBounds;

/// Per-scene terrain feature. Singleton enforced by the editor.
///
/// Carries the configuration that determines tile materialisation:
/// world bounds, base voxel-size tier, and (in later phases) the
/// `TerrainFn`, stamps, and streaming policies.
#[derive(Debug, Clone, Copy)]
pub struct Terrain {
    /// World extent. Defaults to a 16 × 16 × 4 bounded grid; toggle to
    /// `Unbounded` for true infinite procedural worlds.
    pub bounds: TerrainBounds,
    /// `arvx_core::constants::RESOLUTION_TIERS` index for level-0
    /// tiles. Defaults to `DEFAULT_TERRAIN_TIER` (Tier 2 = 0.25 m in
    /// the unified pow2 table). The V2 LOD pyramid walks one tier
    /// coarser per LOD level (`base_tier + level`).
    pub base_tier: usize,
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            bounds: TerrainBounds::default(),
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
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
