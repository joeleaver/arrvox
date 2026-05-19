//! Engine-wide constants and configuration.
//!
//! Absorbed from `rkf-core::constants` as part of the arrvox / rkifield
//! split. The ex-`BRICK_DIM` constant here is renamed `MESH_BRICK_DIM`
//! to disambiguate from `crate::brick_pool::BRICK_DIM` (4) — this one
//! is 8, the **mesh-voxelization brick size** used by the importer's
//! narrow-band classifier, NOT the octree brick size.

/// World chunk size in meters (8m × 8m × 8m chunks)
pub const CHUNK_SIZE: f32 = 8.0;

/// Mesh-voxelization brick dimension — each mesh brick is 8×8×8 = 512
/// voxels. Distinct from `crate::brick_pool::BRICK_DIM` (4), which is
/// the octree brick size.
pub const MESH_BRICK_DIM: u32 = 8;

/// Total voxels per brick
pub const VOXELS_PER_BRICK: u32 = MESH_BRICK_DIM * MESH_BRICK_DIM * MESH_BRICK_DIM; // 512

/// Maximum number of materials (16-bit material IDs, 0–65535).
pub const MAX_MATERIALS: u32 = 65536;

/// Maximum secondary material ID (16-bit, same as primary).
pub const MAX_SECONDARY_MATERIALS: u32 = 65536;

/// A resolution tier defining voxel size and brick spatial extent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolutionTier {
    /// Voxel edge length in meters
    pub voxel_size: f32,
    /// Brick spatial extent in meters (voxel_size * MESH_BRICK_DIM)
    pub brick_extent: f32,
}

/// Unified engine-wide resolution tier table.
///
/// **All voxel sizes are negative powers of 2 in meters**, so any
/// terrain-tile or asset extent that's a power-of-2 multiple of the
/// finest tier (1/128 m) cleanly divides at every tier. This is what
/// makes voxel-aligned grid-snap, terrain-tile alignment, and any
/// future global-voxel-grid feature possible without per-asset
/// padding fudge.
///
/// **Ratios are 2× per step** (not 4×, as the pre-unification table
/// was). 2× gives a gentler LOD pyramid (Nanite-standard), making
/// per-cluster admit transitions smoother across LOD boundaries.
///
/// Tier sizing is chosen so that a 64 m terrain tile divides cleanly
/// at every tier (`64 / voxel_size` is always a power of 2: 64, 128,
/// 256, 512, 1024, 2048, 4096, 8192).
///
/// - Tier 0: 1 m — geoscape, very distant terrain
/// - Tier 1: 0.5 m — coarse terrain, large structures
/// - Tier 2: 0.25 m — terrain ground, voxel-game props (recommended terrain default)
/// - Tier 3: 0.125 m — fine props, detailed terrain
/// - Tier 4: 0.0625 m — character body, detailed props (~6 cm)
/// - Tier 5: 0.03125 m — character face, fine detail (~3 cm; recommended mesh-import default)
/// - Tier 6: 0.015625 m — very fine detail (~1.5 cm)
/// - Tier 7: 0.0078125 m — micro-detail (~8 mm)
pub const RESOLUTION_TIERS: [ResolutionTier; 8] = [
    ResolutionTier { voxel_size: 1.0,        brick_extent: 8.0 },        // Tier 0
    ResolutionTier { voxel_size: 0.5,        brick_extent: 4.0 },        // Tier 1
    ResolutionTier { voxel_size: 0.25,       brick_extent: 2.0 },        // Tier 2
    ResolutionTier { voxel_size: 0.125,      brick_extent: 1.0 },        // Tier 3
    ResolutionTier { voxel_size: 0.0625,     brick_extent: 0.5 },        // Tier 4
    ResolutionTier { voxel_size: 0.03125,    brick_extent: 0.25 },       // Tier 5
    ResolutionTier { voxel_size: 0.015625,   brick_extent: 0.125 },      // Tier 6
    ResolutionTier { voxel_size: 0.0078125,  brick_extent: 0.0625 },     // Tier 7
];

/// Number of resolution tiers.
pub const NUM_TIERS: usize = 8;

/// Default tier for mesh imports — Tier 5 (≈ 3 cm). Closest power-of-2
/// match to the pre-unification 2 cm default. Author can override per-
/// import via `ImportConfig.voxel_size`.
pub const DEFAULT_MESH_IMPORT_TIER: usize = 5;

/// Default tier for terrain level-0 tiles — Tier 2 (0.25 m). Picked
/// for the balance between paint resolution (25 cm cells), bake cost,
/// and visible-tile triangle count. Author can override per Terrain
/// via `Terrain::base_tier`. The V2 LOD pyramid walks +1 tier per
/// level (coarser).
pub const DEFAULT_TERRAIN_TIER: usize = 2;

/// Default brick pool capacity for core geometry (~512MB at 4KB/brick)
pub const DEFAULT_CORE_POOL_CAPACITY: u32 = 131_072; // ~131K bricks

/// Default brick pool capacity for bone data (~64MB at 4KB/brick)
pub const DEFAULT_BONE_POOL_CAPACITY: u32 = 16_384; // ~16K bricks

/// Default brick pool capacity for volumetric data (~64MB at 2KB/brick)
pub const DEFAULT_VOLUMETRIC_POOL_CAPACITY: u32 = 32_768; // ~32K bricks

/// Default brick pool capacity for color data (~32MB at 2KB/brick)
pub const DEFAULT_COLOR_POOL_CAPACITY: u32 = 16_384; // ~16K bricks

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxels_per_brick_is_512() {
        assert_eq!(VOXELS_PER_BRICK, 512);
    }

    #[test]
    fn tier_brick_extent_equals_voxel_size_times_dim() {
        for (i, tier) in RESOLUTION_TIERS.iter().enumerate() {
            let expected = tier.voxel_size * MESH_BRICK_DIM as f32;
            assert!(
                (tier.brick_extent - expected).abs() < 1e-6,
                "Tier {i}: brick_extent {} != voxel_size {} * MESH_BRICK_DIM {}  (expected {})",
                tier.brick_extent,
                tier.voxel_size,
                MESH_BRICK_DIM,
                expected
            );
        }
    }

    #[test]
    fn tier_voxel_sizes_step_by_2x() {
        for i in 1..RESOLUTION_TIERS.len() {
            let ratio = RESOLUTION_TIERS[i - 1].voxel_size / RESOLUTION_TIERS[i].voxel_size;
            assert!(
                (ratio - 2.0).abs() < 1e-6,
                "Tier {i} voxel_size is not 1/2 of tier {}: ratio = {}",
                i - 1,
                ratio
            );
        }
    }

    /// Every voxel_size is a negative power of 2 in meters — the
    /// load-bearing property for grid alignment.
    #[test]
    fn every_voxel_size_is_negative_pow2() {
        for (i, tier) in RESOLUTION_TIERS.iter().enumerate() {
            let recip = 1.0 / tier.voxel_size;
            let rounded = recip.round();
            assert!(
                (recip - rounded).abs() < 1e-4,
                "Tier {i}: 1 / {} = {recip} is not an integer",
                tier.voxel_size
            );
            // round() should give an integer power of 2.
            let n = rounded as u32;
            assert!(
                n.is_power_of_two() || n == 1,
                "Tier {i}: 1 / voxel_size = {n} is not a power of 2"
            );
        }
    }

    /// A 64 m terrain tile divides cleanly at every tier.
    #[test]
    fn sixty_four_meter_tile_divides_at_every_tier() {
        for (i, tier) in RESOLUTION_TIERS.iter().enumerate() {
            let cells = 64.0 / tier.voxel_size;
            let cells_int = cells.round() as u32;
            assert!(
                (cells - cells_int as f32).abs() < 1e-4,
                "Tier {i}: 64 / {} = {cells} is not an integer",
                tier.voxel_size
            );
            assert!(
                cells_int.is_power_of_two(),
                "Tier {i}: 64 / {} = {cells_int} is not a power of 2",
                tier.voxel_size
            );
        }
    }

    #[test]
    fn num_tiers_matches_array_length() {
        assert_eq!(NUM_TIERS, RESOLUTION_TIERS.len());
    }

    #[test]
    fn default_tiers_are_in_range() {
        assert!(DEFAULT_MESH_IMPORT_TIER < NUM_TIERS);
        assert!(DEFAULT_TERRAIN_TIER < NUM_TIERS);
    }

    #[test]
    fn default_mesh_import_tier_is_around_3cm() {
        assert!((RESOLUTION_TIERS[DEFAULT_MESH_IMPORT_TIER].voxel_size - 0.03125).abs() < 1e-6);
    }

    #[test]
    fn default_terrain_tier_is_quarter_meter() {
        assert!((RESOLUTION_TIERS[DEFAULT_TERRAIN_TIER].voxel_size - 0.25).abs() < 1e-6);
    }
}
