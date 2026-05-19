//! Per-Terrain tile streamer.
//!
//! Owns a sparse map of [`TileKey`] → [`TileSlot`] plus a
//! [`BakeWorker`] pool. Each call to [`TileStreamer::tick`] advances
//! the state machine:
//!
//! 1. Drain worker results and hand finished `BakedTile`s to the
//!    caller's `integrate` closure.
//! 2. Compute the *desired set* from the camera radius intersected
//!    with the terrain's bounds.
//! 3. Mark tiles in the loaded set but outside the desired set as
//!    evicted, hand them to the caller's `evict` closure, drop the
//!    slot.
//! 4. Submit fresh bake jobs for desired-but-not-loaded tiles, up to
//!    `max_in_flight`, sorted nearest-camera-first.
//!
//! The streamer is intentionally crate-pure: it knows nothing about
//! `arvx_render::AssetHandle`, ECS entities, or GPU buffers. The
//! `integrate` closure returns an opaque `u64` token the engine uses
//! to look up its own bookkeeping when an eviction lands.
//!
//! ### Data structure choice — HashMap, not octree
//!
//! `docs/TERRAIN.md` calls for "a sparse 3D tile-octree backing
//! store". With V1 limited to `level = 0` and the default bounds of
//! 16 × 16 × 4 = 256 tiles, a HashMap is the appropriate data
//! structure — same big-O, lower constants, simpler invariants. The
//! sparse-octree variant lands when V2 introduces `level > 0` and
//! the per-tile mix justifies the indirection.

use std::collections::HashMap;
use std::sync::Arc;

use arvx_core::WorldPosition;

use crate::baked_tile::BakedTile;
use crate::terrain::Terrain;
use crate::tile_key::{TileKey, TILE_SIZE_M};
use crate::worker::{BakeJob, BakeWorker};

/// Per-tile state in the streamer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileState {
    /// In the desired set; not yet submitted to the worker.
    Queued,
    /// Bake submitted; awaiting worker result.
    Submitted,
    /// Integrated and live in the scene.
    Live,
    /// Last bake failed (panic or pool exhaustion). The streamer
    /// holds onto the slot so it doesn't immediately re-submit;
    /// Phase 9 / V2 can wire a retry timer. For V1 the slot stays
    /// in this state until the camera moves out of the desired set.
    Failed,
}

/// One tile slot inside the streamer.
#[derive(Debug)]
pub struct TileSlot {
    /// Current lifecycle state of this tile.
    pub state: TileState,
    /// Set once integrated. Opaque to the streamer; the engine maps
    /// this to its own ECS entity + asset handle.
    pub integrated_token: Option<u64>,
    /// Cached camera distance² for nearest-first sort. Updated each
    /// tick.
    pub camera_dist_sq: f32,
    /// Per-tile request generation. Bumped whenever the streamer
    /// re-submits the same tile; results carrying an older
    /// generation are discarded.
    pub requested_generation: u64,
}

impl TileSlot {
    fn new() -> Self {
        Self {
            state: TileState::Queued,
            integrated_token: None,
            camera_dist_sq: f32::MAX,
            requested_generation: 0,
        }
    }
}

/// Telemetry snapshot, returned by [`TileStreamer::stats`] for the
/// editor / debug overlay.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamerStats {
    /// Tiles in [`TileState::Queued`].
    pub queued: u32,
    /// Tiles in [`TileState::Submitted`] (worker has the job).
    pub submitted: u32,
    /// Tiles in [`TileState::Live`] (integrated into the scene).
    pub live: u32,
    /// Tiles in [`TileState::Failed`] (bake or integrate failed).
    pub failed: u32,
    /// Worker pool's in-flight count (`submitted - completed`).
    pub in_flight: u32,
    /// Number of worker threads in the pool.
    pub worker_count: u32,
}

/// The tile streamer itself.
pub struct TileStreamer {
    worker: BakeWorker,
    tiles: HashMap<TileKey, TileSlot>,
    max_in_flight: usize,
    /// Workspace buffers reused across ticks to keep `tick` allocation-free
    /// in steady state.
    submit_scratch: Vec<(TileKey, f32)>,
    evict_scratch: Vec<TileKey>,
    /// Phase 4.4: scene directory used to resolve per-tile `.arvxtile`
    /// paths at submit time. `None` for unsaved scratch scenes — the
    /// worker then runs `TerrainFn` baking exclusively. The engine
    /// updates this whenever `scene_path` changes (load / save-as).
    scene_dir: Option<std::path::PathBuf>,
}

impl TileStreamer {
    /// Spawn a fresh streamer with `worker_count` worker threads and
    /// `max_in_flight` concurrent bakes. V1 default: 2 / 2.
    pub fn new(worker_count: usize, max_in_flight: usize) -> Self {
        let worker = BakeWorker::spawn(worker_count);
        Self {
            worker,
            tiles: HashMap::new(),
            max_in_flight: max_in_flight.max(1),
            submit_scratch: Vec::new(),
            evict_scratch: Vec::new(),
            scene_dir: None,
        }
    }

    /// Phase 4.4: tell the streamer which scene directory holds the
    /// `.arvxtile` files for this scene's terrain. Submitted bake
    /// jobs will resolve `<scene_dir>/tiles/<key>.arvxtile` and the
    /// worker will load from disk when the file exists.
    ///
    /// Pass `None` for unsaved scratch scenes — bakes then always
    /// run `TerrainFn` voxelization. The engine sets this on
    /// scene-load + save-as.
    pub fn set_scene_dir(&mut self, dir: Option<std::path::PathBuf>) {
        self.scene_dir = dir;
    }

    /// Worker pool size this streamer was constructed with.
    pub fn worker_count(&self) -> usize {
        self.worker.worker_count()
    }

    /// Number of tiles currently tracked, by state.
    pub fn stats(&self) -> StreamerStats {
        let mut s = StreamerStats {
            worker_count: self.worker.worker_count() as u32,
            in_flight: self.worker.in_flight() as u32,
            ..Default::default()
        };
        for slot in self.tiles.values() {
            match slot.state {
                TileState::Queued => s.queued += 1,
                TileState::Submitted => s.submitted += 1,
                TileState::Live => s.live += 1,
                TileState::Failed => s.failed += 1,
            }
        }
        s
    }

    /// Iterate live tile keys + tokens. Used by the engine to release
    /// every tile when a Terrain is destroyed.
    pub fn iter_live(&self) -> impl Iterator<Item = (TileKey, u64)> + '_ {
        self.tiles.iter().filter_map(|(k, s)| match s.state {
            TileState::Live => s.integrated_token.map(|t| (*k, t)),
            _ => None,
        })
    }

    /// Drain every live tile's `(key, token)` and clear the slot
    /// map. The caller despawns + releases each pair. After this the
    /// streamer is back to its initial empty state.
    pub fn drain_all_live(&mut self) -> Vec<(TileKey, u64)> {
        let mut out = Vec::new();
        for (key, slot) in self.tiles.drain() {
            if let (TileState::Live, Some(token)) =
                (slot.state, slot.integrated_token)
            {
                out.push((key, token));
            }
        }
        out
    }

    /// Phase 1 of the streamer tick: drain every completed bake from
    /// the worker outbox. Returns the per-tile `(key, BakedTile)`
    /// pairs the caller should integrate. The slot stays in
    /// `Submitted` state until the caller reports success/failure
    /// via [`record_integrated`] or [`record_failed`].
    ///
    /// Stale results (slot evicted, generation mismatched, state
    /// diverged) are silently dropped here so the caller never has
    /// to see them.
    pub fn drain_completed(&mut self) -> Vec<(TileKey, BakedTile)> {
        let mut out = Vec::new();
        while let Some(result) = self.worker.try_recv() {
            let Some(slot) = self.tiles.get_mut(&result.key) else {
                continue;
            };
            if result.generation != slot.requested_generation {
                continue;
            }
            if slot.state != TileState::Submitted {
                continue;
            }
            match result.baked {
                Some(baked) => out.push((result.key, baked)),
                None => {
                    slot.state = TileState::Failed;
                    eprintln!(
                        "[arvx-terrain-streamer] bake failed for tile ({}, {}, {}, lvl {})",
                        result.key.x, result.key.y, result.key.z, result.key.level,
                    );
                }
            }
        }
        out
    }

    /// Caller reports a successful integrate. Transitions the slot
    /// `Submitted → Live` and stores `token` for later eviction
    /// callback.
    pub fn record_integrated(&mut self, key: TileKey, token: u64) {
        if let Some(slot) = self.tiles.get_mut(&key) {
            slot.state = TileState::Live;
            slot.integrated_token = Some(token);
        }
    }

    /// Caller reports an integrate failure (e.g. pool exhaustion).
    /// Transitions the slot to `Failed`.
    pub fn record_failed(&mut self, key: TileKey) {
        if let Some(slot) = self.tiles.get_mut(&key) {
            slot.state = TileState::Failed;
            slot.integrated_token = None;
            eprintln!(
                "[arvx-terrain-streamer] integrate failed for tile ({}, {}, {}, lvl {})",
                key.x, key.y, key.z, key.level,
            );
        }
    }

    /// Phase 2 of the streamer tick: compute the desired tile set from
    /// `(camera, terrain.bounds, terrain.render_radius_m)`, refresh
    /// camera distances on every slot, return the list of tiles to
    /// evict (those that were Live but no longer in the desired set).
    ///
    /// Caller despawns + releases each returned `(key, token)`. The
    /// slot is already removed from the streamer by the time this
    /// returns — no `record_evicted` callback needed.
    pub fn update_residency(
        &mut self,
        terrain: &Terrain,
        camera_world: WorldPosition,
    ) -> Vec<(TileKey, u64)> {
        let camera_vec = camera_world.to_vec3();
        let radius = terrain.render_radius_m.max(0.0);
        let radius_sq = radius * radius;
        let r_tiles = (radius / TILE_SIZE_M).ceil() as i32 + 1;
        let cam_tile_x = (camera_vec.x / TILE_SIZE_M).floor() as i32;
        let cam_tile_y = (camera_vec.y / TILE_SIZE_M).floor() as i32;
        let cam_tile_z = (camera_vec.z / TILE_SIZE_M).floor() as i32;

        // Mark every loaded tile as "not yet seen this tick"; the
        // candidate sweep below restores camera_dist_sq for tiles
        // still in the desired set.
        for slot in self.tiles.values_mut() {
            slot.camera_dist_sq = f32::MAX;
        }

        self.submit_scratch.clear();
        for dx in -r_tiles..=r_tiles {
            for dy in -r_tiles..=r_tiles {
                for dz in -r_tiles..=r_tiles {
                    let key = TileKey::level0(
                        cam_tile_x + dx,
                        cam_tile_y + dy,
                        cam_tile_z + dz,
                    );
                    if !terrain.bounds.contains(key) {
                        continue;
                    }
                    let centre = key.centre_world().to_vec3();
                    let d_sq = (centre - camera_vec).length_squared();
                    if d_sq > radius_sq {
                        continue;
                    }
                    let slot = self
                        .tiles
                        .entry(key)
                        .or_insert_with(TileSlot::new);
                    slot.camera_dist_sq = d_sq;
                    if slot.state == TileState::Queued {
                        self.submit_scratch.push((key, d_sq));
                    }
                }
            }
        }

        // Sweep slots whose distance wasn't refreshed → evict.
        self.evict_scratch.clear();
        for (key, slot) in &self.tiles {
            if slot.camera_dist_sq == f32::MAX {
                self.evict_scratch.push(*key);
            }
        }
        let mut evicted: Vec<(TileKey, u64)> = Vec::new();
        for key in self.evict_scratch.drain(..) {
            if let Some(slot) = self.tiles.remove(&key) {
                if let (TileState::Live, Some(token)) =
                    (slot.state, slot.integrated_token)
                {
                    evicted.push((key, token));
                }
            }
        }
        evicted
    }

    /// Phase 3 of the streamer tick: submit fresh bake jobs to the
    /// worker pool, nearest-camera-first, up to the in-flight budget.
    /// Must be called after [`update_residency`] populated
    /// `submit_scratch`.
    pub fn submit_pending(&mut self, terrain: &Terrain) {
        self.submit_scratch.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut budget = self
            .max_in_flight
            .saturating_sub(self.worker.in_flight());
        for (key, _) in self.submit_scratch.drain(..) {
            if budget == 0 {
                break;
            }
            let Some(slot) = self.tiles.get_mut(&key) else { continue };
            if slot.state != TileState::Queued {
                continue;
            }
            slot.requested_generation = slot.requested_generation.wrapping_add(1);
            let disk_path = self
                .scene_dir
                .as_ref()
                .map(|d| crate::persist::tile_path(d, key));
            let job = BakeJob {
                key,
                voxel_size_m: terrain.voxel_size_for_level(0),
                terrain_fn: Arc::clone(&terrain.terrain_fn),
                generation: slot.requested_generation,
                disk_path,
            };
            if self.worker.submit(job) {
                slot.state = TileState::Submitted;
                budget -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::TerrainBounds;
    use crate::terrain::Terrain;
    use crate::terrain_fn::{TerrainFn, TerrainSample};
    use glam::{IVec3, Vec3};

    /// Always-empty terrain: bake returns a valid (zero-tri) tile so
    /// the worker pipeline runs end-to-end without integrating any
    /// non-trivial geometry.
    struct AllSky;
    impl TerrainFn for AllSky {
        fn sample(&self, _t: TileKey, _l: Vec3, _v: f32) -> TerrainSample {
            TerrainSample { sd: 100.0, primary_mat: 1, secondary_mat: 1, blend: 0.0 }
        }
    }

    fn small_terrain() -> Terrain {
        Terrain {
            bounds: TerrainBounds::Bounded {
                origin: TileKey { level: 0, x: 0, y: 0, z: 0 },
                extent: glam::UVec3::new(2, 1, 2),
            },
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
            terrain_fn: Arc::new(AllSky),
            render_radius_m: 200.0,
        }
    }

    #[test]
    fn empty_streamer_has_no_tiles() {
        let s = TileStreamer::new(1, 1);
        assert_eq!(s.stats().live, 0);
        assert_eq!(s.stats().queued, 0);
    }

    #[test]
    fn tick_queues_tiles_inside_bounds_and_radius() {
        let mut s = TileStreamer::new(0, 1); // worker_count 0 → clamps to 1
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        // Phase 2 of tick — populates the tile map.
        let _evicted = s.update_residency(&terrain, camera);
        assert!(s.tiles.len() > 0, "expected tiles to be queued/submitted");
    }

    #[test]
    fn tiles_outside_bounds_are_skipped() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        // Camera way outside the bounded grid → nothing materialises.
        let camera = WorldPosition::new(IVec3::new(800, 0, 0), Vec3::ZERO);
        let _ = s.update_residency(&terrain, camera);
        assert_eq!(
            s.tiles.len(),
            0,
            "tiles outside bounds should never be queued; got {:?}",
            s.tiles.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn moving_camera_evicts_old_tiles() {
        let mut s = TileStreamer::new(1, 1);
        let large = Terrain {
            bounds: TerrainBounds::Unbounded,
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
            terrain_fn: Arc::new(AllSky),
            render_radius_m: 80.0, // ~1 tile radius
        };

        // First residency pass — populate slots near origin.
        let cam_a = WorldPosition::new(IVec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        let _ = s.update_residency(&large, cam_a);
        // Force any queued/submitted tile to Live for the test.
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(42);
        }

        // Move camera 500 m away — every tile from cam_a must evict.
        let cam_b = WorldPosition::new(IVec3::new(500 / 8, 0, 0), Vec3::ZERO);
        let evicted = s.update_residency(&large, cam_b);
        assert!(!evicted.is_empty(), "expected at least one eviction");
    }
}
