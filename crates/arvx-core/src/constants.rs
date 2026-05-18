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

/// Resolution tier table.
///
/// - Tier 0: 0.5cm voxels, 4cm brick extent — fine detail (faces, small props)
/// - Tier 1: 2cm voxels, 16cm brick extent — standard geometry (default)
/// - Tier 2: 8cm voxels, 64cm brick extent — large structures, terrain
/// - Tier 3: 32cm voxels, 256cm brick extent — distant terrain, horizon
pub const RESOLUTION_TIERS: [ResolutionTier; 4] = [
    ResolutionTier { voxel_size: 0.005, brick_extent: 0.04 },  // Tier 0: 0.5cm
    ResolutionTier { voxel_size: 0.02, brick_extent: 0.16 },   // Tier 1: 2cm
    ResolutionTier { voxel_size: 0.08, brick_extent: 0.64 },   // Tier 2: 8cm
    ResolutionTier { voxel_size: 0.32, brick_extent: 2.56 },   // Tier 3: 32cm
];

/// Number of resolution tiers
pub const NUM_TIERS: usize = 4;

/// Default resolution tier index (Tier 1 = 2cm voxels)
pub const DEFAULT_TIER: usize = 1;

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
    fn tier_voxel_sizes_increase_by_4x() {
        for i in 1..RESOLUTION_TIERS.len() {
            let ratio = RESOLUTION_TIERS[i].voxel_size / RESOLUTION_TIERS[i - 1].voxel_size;
            assert!(
                (ratio - 4.0).abs() < 1e-6,
                "Tier {i} voxel_size is not 4× tier {}: ratio = {}",
                i - 1,
                ratio
            );
        }
    }

    #[test]
    fn num_tiers_matches_array_length() {
        assert_eq!(NUM_TIERS, RESOLUTION_TIERS.len());
    }

    #[test]
    fn default_tier_is_in_range() {
        assert!(DEFAULT_TIER < NUM_TIERS);
    }

    #[test]
    fn default_tier_is_2cm() {
        assert!((RESOLUTION_TIERS[DEFAULT_TIER].voxel_size - 0.02).abs() < 1e-6);
    }
}
