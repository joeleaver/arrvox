//! Engine-side runtime state for the active Terrain.
//!
//! `arvx-terrain`'s [`TileStreamer`] is engine-agnostic — it doesn't
//! know about `arvx_render::AssetHandle` or `hecs::Entity`. This
//! module bridges the two: the streamer hands each baked tile to an
//! `integrate` closure that does the scene_mgr lock + ECS spawn, and
//! gets a u64 token back. [`TerrainRuntime`] maps tokens to live
//! `(Entity, AssetHandle)` pairs so eviction can release both.

use std::collections::{HashMap, HashSet, VecDeque};

use arvx_render::AssetHandle;
use arvx_terrain::{BakedTile, SculptDiff, TileKey, TileStreamer};

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
    /// V2 LOD pyramid follow-up: persistent per-tile sculpt diffs,
    /// captured at brush time and replayed by the bake pipeline so:
    ///
    /// 1. **Re-bakes preserve sculpt.** When a tile evicts and later
    ///    re-loads (or invalidation re-runs the bake), the diff is
    ///    replayed against the fresh procedural octree.
    /// 2. **Coarse LOD shows sculpts too.** Level-N≥1 ancestor tiles
    ///    enumerate every level-0 descendant key present in this map,
    ///    downsample each into the coarse grid via
    ///    [`SculptDiff::downsampled_to`], and replay post-integrate.
    /// 3. **Scene reload preserves sculpts.** The save path writes a
    ///    `.arvxsculpt` sidecar per non-empty entry; the load path
    ///    populates this map before the streamer ticks.
    ///
    /// Keyed by **fine (level-0) `TileKey`** — that's the diff's
    /// authoritative coordinate system. Empty entries are pruned to
    /// keep the map size proportional to actually-sculpted regions.
    pub diffs: HashMap<TileKey, SculptDiff>,
    /// Monotonic token counter — the streamer doesn't generate these
    /// itself.
    pub next_token: u64,
    /// P3-A integrate backlog. Freshly-drained bakes land here and are
    /// integrated at most `ARVX_TERRAIN_INTEGRATE_BUDGET` per sim tick, so a
    /// warm-cache burst (a whole footprint ready at once, bake_time≈0) spreads
    /// across ticks instead of stalling the sim — and therefore presentation —
    /// for seconds in a single tick. Every queued tile is still integrated and
    /// still flows through `record_integrated` / hot-swap eviction in order.
    pub pending_integrations: VecDeque<(TileKey, BakedTile)>,
}

impl TerrainRuntime {
    /// Bake-worker pool size for the terrain streamer, chosen to leave
    /// the editor's main/present thread, the render worker, and the
    /// engine sim with CPU headroom during a cold generation — when all
    /// of them plus the bake workers compete, the present thread gets
    /// starved, the Wayland surface goes Outdated, and (with rinch #42
    /// unpatched) the viewport dies ("surface lost").
    ///
    /// Reserve 3 logical cores (present + render + sim) and cap at 2 so
    /// this only ever *reduces* the pool on constrained machines (≤4
    /// cores → 1); typical dev machines keep the historical 2. Bigger
    /// pools aren't worth it now that bakes are cheap (analytic-slope +
    /// O(1) face-links) — extra workers would only add per-tile integrate
    /// pressure on the sim thread. Falls back to 1 if the core count is
    /// unavailable.
    pub fn bake_worker_count() -> usize {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        cores.saturating_sub(3).clamp(1, 2)
    }

    /// Construct a fresh runtime. Worker pool sized by
    /// [`Self::bake_worker_count`]; 2 in-flight bakes.
    pub fn new(terrain_entity: hecs::Entity) -> Self {
        Self {
            terrain_entity,
            streamer: TileStreamer::new(Self::bake_worker_count(), 2),
            live_tiles: HashMap::new(),
            tile_keys: HashMap::new(),
            dirty_tiles: HashSet::new(),
            divergent_tiles: HashSet::new(),
            diffs: HashMap::new(),
            next_token: 1,
            pending_integrations: VecDeque::new(),
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

    /// Append a sculpt-stamp's captured `LeafEdit`s into the per-tile
    /// diff. SetNormal edits should already be filtered by the
    /// caller (`SculptApplyResult::captured_edits` does this), but
    /// `SculptDiff::append_delta` re-filters defensively.
    ///
    /// `key` must be the level-0 tile that owned the brush stamp;
    /// downsampling to coarser ancestors happens at bake time.
    pub fn append_sculpt_edits(
        &mut self,
        key: TileKey,
        edits: &[arvx_core::sculpt::LeafEdit],
    ) {
        if edits.is_empty() {
            return;
        }
        // Wrap in a transient SculptDelta so we go through the same
        // filter SculptDiff::append_delta applies to bare deltas —
        // single source of truth for the SetNormal-drop rule.
        let delta = arvx_core::sculpt::SculptDelta {
            edits: edits.to_vec(),
            ..Default::default()
        };
        self.diffs.entry(key).or_default().append_delta(&delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arvx_core::sculpt::{LeafEdit, LeafEditOp};
    use glam::{UVec3, Vec3};

    fn make_runtime() -> TerrainRuntime {
        // Entity::DANGLING is fine for runtime state we don't tick;
        // these tests cover diff bookkeeping only.
        TerrainRuntime::new(hecs::Entity::DANGLING)
    }

    /// The adaptive bake-worker pool stays in `[1, 2]` regardless of core
    /// count: it reserves cores for present/render/sim (so it can only
    /// *reduce* on constrained machines) and never exceeds the historical
    /// 2. A fresh runtime's streamer reflects it.
    #[test]
    fn bake_worker_count_is_clamped() {
        let n = TerrainRuntime::bake_worker_count();
        assert!((1..=2).contains(&n), "worker count {n} out of [1, 2]");
        assert_eq!(make_runtime().streamer.worker_count(), n);
    }

    fn add(coord: UVec3) -> LeafEdit {
        LeafEdit {
            coord,
            op: LeafEditOp::Add {
                material: 1,
                normal: Vec3::Y,
                dist: 0.0,
            },
        }
    }

    #[test]
    fn append_sculpt_edits_creates_per_tile_diff_entry() {
        let mut rt = make_runtime();
        let key = TileKey::level0(2, 0, 1);
        rt.append_sculpt_edits(key, &[add(UVec3::new(0, 0, 0))]);
        assert_eq!(rt.diffs.len(), 1);
        assert_eq!(rt.diffs[&key].len(), 1);
    }

    #[test]
    fn append_sculpt_edits_accumulates_across_calls() {
        let mut rt = make_runtime();
        let key = TileKey::level0(0, 0, 0);
        rt.append_sculpt_edits(key, &[add(UVec3::new(0, 0, 0))]);
        rt.append_sculpt_edits(key, &[add(UVec3::new(1, 0, 0))]);
        assert_eq!(rt.diffs[&key].len(), 2);
    }

    #[test]
    fn append_sculpt_edits_empty_is_no_op() {
        let mut rt = make_runtime();
        rt.append_sculpt_edits(TileKey::level0(0, 0, 0), &[]);
        assert!(rt.diffs.is_empty());
    }

    /// SetNormal carries a per-octree slot id and can't be replayed.
    /// `SculptDiff::append_delta` filters it; `append_sculpt_edits`
    /// delegates so the same filter applies.
    #[test]
    fn append_sculpt_edits_filters_set_normal() {
        let mut rt = make_runtime();
        let key = TileKey::level0(0, 0, 0);
        let edits = vec![
            add(UVec3::new(0, 0, 0)),
            LeafEdit {
                coord: UVec3::new(1, 0, 0),
                op: LeafEditOp::SetNormal {
                    slot: 42,
                    normal: Vec3::Y,
                },
            },
        ];
        rt.append_sculpt_edits(key, &edits);
        assert_eq!(rt.diffs[&key].len(), 1);
    }
}
