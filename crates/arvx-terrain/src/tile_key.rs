//! Tile-octree spatial keys.
//!
//! A `TileKey` identifies one 64 m cubic terrain tile in the world-tile
//! octree. The `level` field is reserved for the V2 LOD pyramid; V1 only
//! allocates `level = 0` tiles, but the field exists in the API and on
//! disk so V2 is purely additive.
//!
//! ## Integer-keyed addressing
//!
//! Tile keys are i32 along each axis, scaled by `TILE_SIZE_M` to give
//! world meters. This is the integer side of the FP-drift boundary
//! described in `docs/TERRAIN.md` — keys cross tile boundaries; f32
//! stays inside a single tile (relative to the tile origin) so a tile
//! at any world distance has the same internal precision.

use arvx_core::WorldPosition;
use glam::{IVec3, Vec3};

/// Side length of one tile at level 0, in metres. Locked to 64 m for V1.
pub const TILE_SIZE_M: f32 = 64.0;

/// Chunks per tile axis at level 0. `WorldPosition` chunks are 8 m,
/// so a 64 m tile = exactly 8 chunks; integer math has no remainders.
pub const CHUNKS_PER_TILE_AXIS_L0: i32 = 8;

/// Identifies one tile in the world tile-octree.
///
/// V1 only uses `level = 0`. Higher levels double the tile footprint
/// per increment (coarser LODs in V2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TileKey {
    /// LOD level. 0 = finest. V1 only allocates level 0; higher levels
    /// are reserved for V2's coarse-tile pyramid.
    pub level: u8,
    /// X coordinate in tile space (scaled by `extent_m` to get world m).
    pub x: i32,
    /// Y coordinate in tile space.
    pub y: i32,
    /// Z coordinate in tile space.
    pub z: i32,
}

impl TileKey {
    /// Construct a level-0 tile at the given tile-space coordinate.
    pub const fn level0(x: i32, y: i32, z: i32) -> Self {
        Self { level: 0, x, y, z }
    }

    /// Side length of this tile in metres. Each LOD level doubles.
    pub fn extent_m(self) -> f32 {
        TILE_SIZE_M * (1u32 << self.level) as f32
    }

    /// World-space origin (lo corner) of this tile as a `WorldPosition`.
    ///
    /// Integer math throughout: `chunks_per_axis = 8 * 2^level`. No
    /// floating-point rounding regardless of how far the tile is from
    /// the world origin.
    pub fn origin_world(self) -> WorldPosition {
        let chunks_per_axis = CHUNKS_PER_TILE_AXIS_L0 * (1i32 << self.level);
        let chunks = IVec3::new(
            self.x * chunks_per_axis,
            self.y * chunks_per_axis,
            self.z * chunks_per_axis,
        );
        WorldPosition::new(chunks, Vec3::ZERO)
    }

    /// World-space centre of this tile.
    pub fn centre_world(self) -> WorldPosition {
        self.origin_world() + Vec3::splat(self.extent_m() * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level0_origin_is_at_world_origin() {
        let k = TileKey::level0(0, 0, 0);
        let o = k.origin_world();
        assert_eq!(o.chunk, IVec3::ZERO);
        assert_eq!(o.local, Vec3::ZERO);
    }

    #[test]
    fn level0_extent_is_64m() {
        let k = TileKey::level0(0, 0, 0);
        assert!((k.extent_m() - 64.0).abs() < 1e-6);
    }

    #[test]
    fn level1_extent_doubles() {
        let k = TileKey { level: 1, x: 0, y: 0, z: 0 };
        assert!((k.extent_m() - 128.0).abs() < 1e-6);
    }

    #[test]
    fn unit_tile_step_at_level0_is_8_chunks() {
        let k = TileKey::level0(1, 0, 0);
        // One tile over on x = 8 chunks east of origin (CHUNK_SIZE=8m, TILE_SIZE=64m).
        assert_eq!(k.origin_world().chunk, IVec3::new(8, 0, 0));
    }

    #[test]
    fn negative_tile_coords_produce_negative_chunks() {
        let k = TileKey::level0(-1, -2, -3);
        assert_eq!(k.origin_world().chunk, IVec3::new(-8, -16, -24));
    }

    #[test]
    fn origin_far_from_world_origin_uses_integer_chunks() {
        // 1 million tiles east = 8 million chunks. Integer math => exact.
        let k = TileKey::level0(1_000_000, 0, 0);
        let o = k.origin_world();
        assert_eq!(o.chunk.x, 8_000_000);
        assert_eq!(o.local, Vec3::ZERO);
    }

    #[test]
    fn centre_is_midway_between_origin_corners() {
        let k = TileKey::level0(2, 0, 0);
        let centre = k.centre_world();
        // Tile (2, 0, 0) origin: chunk (16, 0, 0). Centre: chunk (16, 0, 0) + (32, 32, 32).
        // 32m = 4 chunks, so centre normalises to chunk (20, 4, 4), local (0, 0, 0).
        assert_eq!(centre.chunk, IVec3::new(20, 4, 4));
        assert!(centre.local.length() < 1e-4);
    }

    #[test]
    fn level1_tile_step_uses_16_chunks() {
        let k = TileKey { level: 1, x: 1, y: 0, z: 0 };
        // Level-1 tile = 128m = 16 chunks per axis.
        assert_eq!(k.origin_world().chunk, IVec3::new(16, 0, 0));
    }
}
