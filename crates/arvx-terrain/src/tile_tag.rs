//! ECS marker component for streamed terrain tile entities.
//!
//! Carrying `TerrainTile` distinguishes a runtime-spawned tile from
//! author-created scene objects. Two consumers use the marker:
//!
//! - `scene_io_ops::build_scene_file` skips tile entities during scene
//!   save (Phase 2 doesn't persist materialised tiles — they
//!   regenerate from `TerrainFn`).
//! - `TileStreamer` recognises its own tile entities during the
//!   eviction sweep, in case a Terrain is destroyed without going
//!   through the streamer's despawn path.

use crate::tile_key::TileKey;

/// Marks an ECS entity as a terrain tile. Carries the originating
/// `TileKey` so the streamer can match entity → slot during recovery.
#[derive(Debug, Clone, Copy)]
pub struct TerrainTile {
    /// The tile key this entity materialised.
    pub key: TileKey,
}
