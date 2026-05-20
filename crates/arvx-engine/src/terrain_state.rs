//! Engine-side runtime state for the active Terrain.
//!
//! `arvx-terrain`'s [`TileStreamer`] is engine-agnostic — it doesn't
//! know about `arvx_render::AssetHandle` or `hecs::Entity`. This
//! module bridges the two: the streamer hands each baked tile to an
//! `integrate` closure that does the scene_mgr lock + ECS spawn, and
//! gets a u64 token back. [`TerrainRuntime`] maps tokens to live
//! `(Entity, AssetHandle)` pairs so eviction can release both.

use std::collections::{HashMap, HashSet};

use arvx_render::AssetHandle;
use arvx_terrain::{TileKey, TileStreamer};

/// Engine-side runtime state for the active Terrain.
///
/// Owned via `Option<Box<TerrainRuntime>>` on [`EngineState`]; created
/// in `SpawnTerrain` and dropped when the Terrain entity is removed.
pub struct TerrainRuntime {
    /// The Terrain ECS entity itself — carries the `Terrain`
    /// component the streamer reads each tick.
    pub terrain_entity: hecs::Entity,
    /// The streamer instance.
    pub streamer: TileStreamer,
    /// Tile-integration bookkeeping. The streamer hands the engine an
    /// opaque u64 token when it integrates a tile and gives the same
    /// token back when it evicts. We map tokens to live
    /// `(Entity, AssetHandle)` pairs so the eviction handler can
    /// despawn the entity and release the asset.
    pub live_tiles: HashMap<u64, (hecs::Entity, AssetHandle)>,
    /// Reverse map for Phase 4 brush dispatch: a world-space brush
    /// AABB enumerates intersecting `TileKey`s, then looks up each
    /// live tile's `(Entity, AssetHandle)` here. Mirrors `live_tiles`
    /// — populated on integrate, depopulated on evict — so all
    /// reads are O(1).
    pub tile_keys: HashMap<TileKey, (hecs::Entity, AssetHandle)>,
    /// Phase 4.3: tiles that have been edited (sculpt or paint) since
    /// the last save flush. On `File → Save scene` the engine
    /// serialises each entry to `<scene>/tiles/<key>.arvxtile` and
    /// clears the set. Eviction does NOT clear an entry — when a
    /// dirty tile leaves the residency radius before a save, we keep
    /// the bit so the next save still writes it (the in-memory
    /// state is reconstructed from disk on the next residency pass,
    /// at which point the eviction-time `release_asset` may already
    /// have dropped the live state — Phase 4.4 wires that side).
    /// For Phase 4.3 V1 we accept that an evict-without-save loses
    /// the edits; the editor-driven workflow saves frequently
    /// enough that this is rare.
    pub dirty_tiles: HashSet<TileKey>,
    /// Phase 9b: tiles divergent from the procedural baseline —
    /// what the heatmap visualises. Superset of `dirty_tiles`:
    /// includes any tile that has ever been persisted as
    /// `.arvxtile` on disk (so cross-session edits stay visible) or
    /// has been edited this session. Cleared on
    /// `revert_terrain_in_aabb`.
    pub divergent_tiles: HashSet<TileKey>,
    /// Monotonic token counter — the streamer doesn't generate these
    /// itself.
    pub next_token: u64,
}

impl TerrainRuntime {
    /// Construct a fresh runtime with default worker pool sizing
    /// (2 workers, 2 in-flight).
    pub fn new(terrain_entity: hecs::Entity) -> Self {
        Self {
            terrain_entity,
            streamer: TileStreamer::new(2, 2),
            live_tiles: HashMap::new(),
            tile_keys: HashMap::new(),
            dirty_tiles: HashSet::new(),
            divergent_tiles: HashSet::new(),
            next_token: 1,
        }
    }

    /// Phase 9b: mark a tile as divergent from baseline (sculpt /
    /// paint edited it this session). Both `dirty_tiles` and
    /// `divergent_tiles` get the key — the former tracks "needs to
    /// be flushed on save," the latter "shows up in the heatmap."
    pub fn mark_dirty(&mut self, key: TileKey) {
        self.dirty_tiles.insert(key);
        self.divergent_tiles.insert(key);
    }
}
