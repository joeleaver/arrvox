//! # arvx-terrain
//!
//! Streamed editable voxel terrain for the Arrvox engine.
//!
//! See `docs/TERRAIN.md` for the full design. V1 Phase 1 ships the bake
//! pipeline: a `TerrainFn` defines a procedural source, a `TileKey`
//! identifies a 64 m cubic tile, and `bake_tile` composes the existing
//! `arvx-core` voxelization + surface-nets + cluster-DAG passes into a
//! self-contained `BakedTile` artifact. Streaming, halo seams, sculpt
//! integration, stamps, regions, materials, physics, and editor
//! integration land in subsequent phases.

#![warn(missing_docs)]

pub mod bake;
pub mod baked_tile;
pub mod biome_region;
pub mod bounds;
pub mod fbm;
pub mod persist;
pub mod stamp;
pub mod stamp_index;
pub mod streamer;
pub mod terrain;
pub mod terrain_fn;
pub mod tile_key;
pub mod tile_tag;
pub mod worker;

pub use bake::bake_tile;
pub use baked_tile::BakedTile;
pub use biome_region::BiomeRegion;
pub use bounds::TerrainBounds;
pub use fbm::FbmTerrainFn;
pub use persist::{save_tile, tile_path, TILES_SUBDIR};
pub use stamp::{combine_heights, CombineOp, FalloffCurve, Stamp, StampKind};
pub use stamp_index::{StampIndex, StampIndexHandle};
pub use streamer::{StreamerStats, TileSlot, TileState, TileStreamer};
pub use terrain::Terrain;
pub use terrain_fn::{TerrainFn, TerrainSample};
pub use tile_key::{tile_keys_intersecting_aabb, TileKey, TILE_SIZE_M};
pub use tile_tag::TerrainTile;
pub use worker::{BakeJob, BakeJobResult, BakeWorker};
