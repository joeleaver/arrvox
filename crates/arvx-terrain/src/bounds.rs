//! World extent for a Terrain — bounded grid by default, unbounded opt-in.
//!
//! Bounded matches the Unity/Unreal/Godot mental model: the author sees
//! a fixed N×N×K grid of tiles and the world ends cleanly past the
//! boundary. Unbounded is opt-in for infinite procedural projects.

use crate::tile_key::TileKey;
use glam::{UVec3, Vec3};

/// World extent for a `Terrain`.
///
/// Bounded is the default with a 16 × 16 × 4 grid (1024 × 1024 × 256 m
/// at level 0). The streamer respects bounds by not materialising tiles
/// outside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TerrainBounds {
    /// Fixed grid of tiles. All tiles share `origin.level`.
    Bounded {
        /// Inclusive lo-corner tile. `origin.level` is the level all
        /// tiles in the bound are addressed at.
        origin: TileKey,
        /// Size of the grid in tiles along (x, y, z).
        extent: UVec3,
    },
    /// Infinite world; streamer materialises tiles around the camera
    /// indefinitely. Opt-in only.
    Unbounded,
}

impl Default for TerrainBounds {
    /// Default: 16 × 16 × 4 grid of level-0 tiles at the world origin.
    /// Spans 1024 × 1024 × 256 m.
    fn default() -> Self {
        Self::Bounded {
            origin: TileKey::level0(0, 0, 0),
            extent: UVec3::new(16, 16, 4),
        }
    }
}

impl TerrainBounds {
    /// Test whether `key` falls inside the bounded extent. `Unbounded`
    /// returns `true` for any key.
    ///
    /// For `Bounded`, this is a **world-AABB intersection** between the
    /// candidate tile and the bounded region. Tiles at *any* LOD level
    /// can pass — a coarse level-1 tile partially overlapping a level-0
    /// bounded region materialises and bakes its full extent; the few
    /// cells past the authored edge are harmless for V1. This is what
    /// unlocks the V2 LOD pyramid inside `Bounded` worlds.
    pub fn contains(&self, key: TileKey) -> bool {
        match *self {
            Self::Unbounded => true,
            Self::Bounded { origin, extent } => {
                // Bounded region's world AABB. `extent` counts tiles at
                // `origin.level`, so the per-axis world span is
                // `extent.X * origin.extent_m()`.
                let region_min = origin.origin_world().to_vec3();
                let origin_tile_m = origin.extent_m();
                let region_max = region_min
                    + Vec3::new(
                        extent.x as f32 * origin_tile_m,
                        extent.y as f32 * origin_tile_m,
                        extent.z as f32 * origin_tile_m,
                    );

                // Candidate tile's world AABB.
                let tile_min = key.origin_world().to_vec3();
                let tile_max = tile_min + Vec3::splat(key.extent_m());

                // Half-open intersection: a tile whose `max` exactly
                // equals the region's `min` doesn't overlap.
                tile_min.x < region_max.x
                    && tile_max.x > region_min.x
                    && tile_min.y < region_max.y
                    && tile_max.y > region_min.y
                    && tile_min.z < region_max.z
                    && tile_max.z > region_min.z
            }
        }
    }

    /// Total tile count inside the bound. `Unbounded` returns `None`.
    pub fn total_tiles(&self) -> Option<u64> {
        match *self {
            Self::Unbounded => None,
            Self::Bounded { extent, .. } => {
                Some(extent.x as u64 * extent.y as u64 * extent.z as u64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_16x16x4_at_origin() {
        let b = TerrainBounds::default();
        let TerrainBounds::Bounded { origin, extent } = b else {
            panic!("default should be Bounded");
        };
        assert_eq!(origin, TileKey::level0(0, 0, 0));
        assert_eq!(extent, UVec3::new(16, 16, 4));
        assert_eq!(b.total_tiles(), Some(16 * 16 * 4));
    }

    #[test]
    fn default_contains_origin_tile() {
        let b = TerrainBounds::default();
        assert!(b.contains(TileKey::level0(0, 0, 0)));
        assert!(b.contains(TileKey::level0(15, 15, 3)));
    }

    #[test]
    fn default_excludes_outside_tiles() {
        let b = TerrainBounds::default();
        assert!(!b.contains(TileKey::level0(-1, 0, 0)));
        assert!(!b.contains(TileKey::level0(16, 0, 0)));
        assert!(!b.contains(TileKey::level0(0, 16, 0)));
        assert!(!b.contains(TileKey::level0(0, 0, 4)));
    }

    #[test]
    fn unbounded_contains_anything() {
        let b = TerrainBounds::Unbounded;
        assert!(b.contains(TileKey::level0(0, 0, 0)));
        assert!(b.contains(TileKey::level0(1_000_000, -999_999, 0)));
        assert!(b.total_tiles().is_none());
    }

    /// V2 LOD pyramid: a level-1 tile that overlaps a level-0 bounded
    /// region is "contained" — its world AABB intersects the region's.
    /// The level-equality V1 check was a sanity assertion and is
    /// replaced by the world-AABB test.
    #[test]
    fn bounded_accepts_coarse_level_overlapping_region() {
        let b = TerrainBounds::default(); // origin level 0, 16x16x4 tiles at 64m
        // Level-1 tile at (0, 0, 0) — world AABB [0, 128)³. Bounded
        // region world AABB is [0, 1024)×[0, 1024)×[0, 256). Overlaps.
        assert!(b.contains(TileKey { level: 1, x: 0, y: 0, z: 0 }));
        // Level-2 tile at (0, 0, 0) — world AABB [0, 256)³. Still overlaps.
        assert!(b.contains(TileKey { level: 2, x: 0, y: 0, z: 0 }));
    }

    /// A coarse tile FAR outside the region is rejected regardless of
    /// level (no AABB overlap).
    #[test]
    fn bounded_rejects_coarse_level_outside_region() {
        let b = TerrainBounds::default();
        // Level-1 tile at (100, 0, 0) — world x in [12800, 12928).
        // Region world x in [0, 1024). No overlap.
        assert!(!b.contains(TileKey { level: 1, x: 100, y: 0, z: 0 }));
        // Level-2 tile far below world: world z in [-512, -256).
        assert!(!b.contains(TileKey { level: 2, x: 0, y: 0, z: -2 }));
    }

    /// A coarse tile straddling the region edge (partial overlap) is
    /// accepted. Bake will compute terrain past the authored edge for
    /// the few cells in the overflow; harmless for V1 of L1.
    #[test]
    fn bounded_accepts_coarse_tile_straddling_edge() {
        let b = TerrainBounds::default();
        // Level-1 tile at x=7 — world x in [896, 1024) — flush with the
        // region's +x edge from inside.
        assert!(b.contains(TileKey { level: 1, x: 7, y: 0, z: 0 }));
        // Level-1 tile at x=8 — world x in [1024, 1152) — exactly
        // outside (half-open). Should reject.
        assert!(!b.contains(TileKey { level: 1, x: 8, y: 0, z: 0 }));
        // Level-2 tile at x=3 — world x in [768, 1024) — flush with the
        // +x edge.
        assert!(b.contains(TileKey { level: 2, x: 3, y: 0, z: 0 }));
    }
}
