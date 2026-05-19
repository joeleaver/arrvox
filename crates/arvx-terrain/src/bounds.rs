//! World extent for a Terrain — bounded grid by default, unbounded opt-in.
//!
//! Bounded matches the Unity/Unreal/Godot mental model: the author sees
//! a fixed N×N×K grid of tiles and the world ends cleanly past the
//! boundary. Unbounded is opt-in for infinite procedural projects.

use crate::tile_key::TileKey;
use glam::UVec3;

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
    pub fn contains(&self, key: TileKey) -> bool {
        match *self {
            Self::Unbounded => true,
            Self::Bounded { origin, extent } => {
                key.level == origin.level
                    && key.x >= origin.x
                    && key.x < origin.x + extent.x as i32
                    && key.y >= origin.y
                    && key.y < origin.y + extent.y as i32
                    && key.z >= origin.z
                    && key.z < origin.z + extent.z as i32
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

    #[test]
    fn bounded_rejects_wrong_level() {
        let b = TerrainBounds::default();
        // origin is level 0; a level-1 key with the same xyz is NOT in bounds.
        assert!(!b.contains(TileKey { level: 1, x: 0, y: 0, z: 0 }));
    }
}
