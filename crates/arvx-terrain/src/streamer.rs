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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arvx_core::{Aabb, WorldPosition};
use glam::Vec3;

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

/// LOD band `[band_inner, band_outer)` for `level` in a pyramid of
/// `lod_levels` total levels. The bands divide `[0, render_radius)`
/// geometrically — each level reaches out to twice the previous
/// level's far edge, so the band widths double per level (level 0 is
/// the smallest, the outer-most level is the largest).
///
/// Closed form: `band_outer(k) = render_radius * (2^(k+1) - 1) /
/// (2^N - 1)` where `N = lod_levels`. The geometric-series total
/// `1 + 2 + 4 + ... + 2^(N-1) = 2^N - 1` ensures `band_outer(N-1) ==
/// render_radius` regardless of N.
///
/// For `lod_levels == 1` this returns `(0, render_radius)`, matching
/// the V1 single-level sweep bit-identically.
#[inline]
fn lod_band(render_radius: f32, lod_levels: u8, level: u8) -> (f32, f32) {
    debug_assert!(level < lod_levels.max(1));
    let n = lod_levels.max(1) as u32;
    let denom = ((1u32 << n) - 1) as f32; // 2^N - 1
    let lo_num = if level == 0 {
        0u32
    } else {
        (1u32 << level) - 1
    } as f32;
    let hi_num = ((1u32 << (level + 1)) - 1) as f32;
    let band_inner = render_radius * lo_num / denom;
    let band_outer = if level + 1 == lod_levels {
        render_radius
    } else {
        render_radius * hi_num / denom
    };
    (band_inner, band_outer)
}

/// True if a level-0 tile's full 3D cube intersects the query AABB.
#[inline]
fn tile_intersects_aabb(key: TileKey, q: &Aabb) -> bool {
    let origin = key.origin_world().to_vec3();
    let extent = key.extent_m();
    let max = origin + Vec3::splat(extent);
    q.max.x >= origin.x
        && q.min.x <= max.x
        && q.max.y >= origin.y
        && q.min.y <= max.y
        && q.max.z >= origin.z
        && q.min.z <= max.z
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
    ///
    /// Reports any slot with `integrated_token == Some(_)` — including
    /// mid-hot-swap slots that have already invalidated but still
    /// carry resident geometry from the previous bake.
    pub fn iter_live(&self) -> impl Iterator<Item = (TileKey, u64)> + '_ {
        self.tiles
            .iter()
            .filter_map(|(k, s)| s.integrated_token.map(|t| (*k, t)))
    }

    /// Drain every live tile's `(key, token)` and clear the slot
    /// map. The caller despawns + releases each pair. After this the
    /// streamer is back to its initial empty state.
    ///
    /// Emits eviction for any slot with `integrated_token == Some(_)`
    /// regardless of its state — a slot mid-hot-swap (Queued/Submitted
    /// with the previous token still set) still has live geometry in
    /// the scene that needs releasing.
    pub fn drain_all_live(&mut self) -> Vec<(TileKey, u64)> {
        let mut out = Vec::new();
        for (key, slot) in self.tiles.drain() {
            if let Some(token) = slot.integrated_token {
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
    ///
    /// Returns `Some(prev_token)` when this integration is a **hot-swap**
    /// — i.e. the slot was already Live (with `prev_token`) when an
    /// invalidation queued a re-bake. The caller despawns / releases
    /// the previous (entity, asset_handle) pair now that the fresh
    /// geometry is in place. Returns `None` for first-time integrations.
    ///
    /// The collider system is keyed by `TileKey` and already handles
    /// re-bake during the caller's `on_terrain_tile_added` step, so the
    /// deferred eviction here MUST NOT also call `on_terrain_tile_evicted`
    /// — it would drop the just-installed collider.
    #[must_use = "the returned previous token must be evicted to complete the hot-swap"]
    pub fn record_integrated(&mut self, key: TileKey, token: u64) -> Option<u64> {
        let slot = self.tiles.get_mut(&key)?;
        let prev = slot.integrated_token;
        slot.state = TileState::Live;
        slot.integrated_token = Some(token);
        prev
    }

    /// Caller reports an integrate failure (e.g. pool exhaustion).
    /// Transitions the slot to `Failed`.
    ///
    /// Returns the previous `integrated_token` (if any) so the caller
    /// can evict any predecessor that was kept alive for hot-swap.
    /// Without this the predecessor would orphan: residency eviction
    /// uses `integrated_token` and we just cleared it.
    #[must_use = "the returned previous token must be evicted"]
    pub fn record_failed(&mut self, key: TileKey) -> Option<u64> {
        let slot = self.tiles.get_mut(&key)?;
        let prev = slot.integrated_token;
        slot.state = TileState::Failed;
        slot.integrated_token = None;
        eprintln!(
            "[arvx-terrain-streamer] integrate failed for tile ({}, {}, {}, lvl {})",
            key.x, key.y, key.z, key.level,
        );
        prev
    }

    /// Phase 2 of the streamer tick: compute the desired tile set from
    /// `(camera, terrain.bounds, terrain.render_radius_m)`, refresh
    /// camera distances on every slot, return the list of tiles to
    /// evict (those that were Live but no longer in the desired set).
    ///
    /// Caller despawns + releases each returned `(key, token)`. The
    /// slot is already removed from the streamer by the time this
    /// returns — no `record_evicted` callback needed.
    /// Convenience: like [`Self::update_residency_with_pinned`] but
    /// passes an empty pinned-tile set. Kept for tests + callers that
    /// don't have a sculpt-dirty set to thread through.
    pub fn update_residency(
        &mut self,
        terrain: &Terrain,
        camera_world: WorldPosition,
    ) -> Vec<(TileKey, u64)> {
        self.update_residency_with_pinned(terrain, camera_world, &HashSet::new())
    }

    /// Recompute the desired-tile set against the current camera +
    /// terrain config, with `dirty_pinned` extending the always-load
    /// set so sculpted level-0 tiles stay resident at fine LOD even
    /// past `render_radius_m`. Coarse-LOD ancestors are NOT
    /// suppressed in V2 LOD pyramid + diff-propagation builds —
    /// they coexist with the pinned fine tile and render the
    /// downsampled sculpt via the engine's post-integrate replay.
    pub fn update_residency_with_pinned(
        &mut self,
        terrain: &Terrain,
        camera_world: WorldPosition,
        dirty_pinned: &HashSet<TileKey>,
    ) -> Vec<(TileKey, u64)> {
        let camera_vec = camera_world.to_vec3();
        let radius = terrain.render_radius_m.max(0.0);
        let radius_sq = radius * radius;
        let lod_levels = terrain.lod_levels.max(1);

        // Mark every loaded tile as "not yet seen this tick"; the
        // candidate sweep below restores camera_dist_sq for tiles
        // still in the desired set.
        for slot in self.tiles.values_mut() {
            slot.camera_dist_sq = f32::MAX;
        }

        self.submit_scratch.clear();

        // V2 L-pyramid: dirty-tile pin. Sculpted level-0 tiles stay
        // loaded at fine LOD regardless of distance so the user keeps
        // seeing their sculpt at brush resolution near the camera even
        // if the camera briefly drifts past `render_radius_m`. The
        // bake-replay path (engine's `gather_replay_edits` after
        // `integrate_baked_tile`) makes eviction itself safe — diffs
        // live in `TerrainRuntime::diffs`, persistent across the
        // streamer's tile lifecycle — so the pin is purely a
        // "keep fine detail visible without a re-bake hiccup" comfort
        // rule, no longer a correctness requirement.
        for key in dirty_pinned {
            let centre = key.centre_world().to_vec3();
            let d_sq = (centre - camera_vec).length_squared();
            let slot = self.tiles.entry(*key).or_insert_with(TileSlot::new);
            slot.camera_dist_sq = d_sq;
            if slot.state == TileState::Queued {
                self.submit_scratch.push((*key, d_sq));
            }
        }

        // V2 L-pyramid: distance-banded multi-level residency. Each
        // level covers `[band_inner, band_outer)` chosen by geometric
        // split of `[0, render_radius)`. A tile is canonical owner of
        // its footprint iff its CENTRE distance lies in its level's
        // band — non-overlapping, no z-fight at boundaries. Visible
        // cracks at the boundary are V1-of-L1 trade-off; later
        // sessions add fine-ring overlap + Transvoxel.
        //
        // For `lod_levels = 1` this collapses to the V1 single-level
        // sweep over `[0, render_radius_m)`.
        for level in 0..lod_levels {
            let (band_inner, band_outer) = lod_band(radius, lod_levels, level);
            let band_inner_sq = band_inner * band_inner;
            let band_outer_sq = if band_outer.is_finite() {
                band_outer * band_outer
            } else {
                radius_sq // outer-most level capped at the residency radius
            };

            let tile_size_at_level = TILE_SIZE_M * (1u32 << level) as f32;
            let r_tiles = (band_outer / tile_size_at_level).ceil() as i32 + 1;
            let cam_tile_x = (camera_vec.x / tile_size_at_level).floor() as i32;
            let cam_tile_y = (camera_vec.y / tile_size_at_level).floor() as i32;
            let cam_tile_z = (camera_vec.z / tile_size_at_level).floor() as i32;

            for dx in -r_tiles..=r_tiles {
                for dy in -r_tiles..=r_tiles {
                    for dz in -r_tiles..=r_tiles {
                        let key = TileKey {
                            level,
                            x: cam_tile_x + dx,
                            y: cam_tile_y + dy,
                            z: cam_tile_z + dz,
                        };
                        if !terrain.bounds.contains(key) {
                            continue;
                        }
                        let centre = key.centre_world().to_vec3();
                        let d_sq = (centre - camera_vec).length_squared();
                        // Half-open `[band_inner, band_outer)` against
                        // tile centre — assigns each spatial position
                        // to exactly one level. Compare with the
                        // total residency radius too, so the outer-
                        // most level still respects `render_radius_m`.
                        if d_sq < band_inner_sq || d_sq >= band_outer_sq {
                            continue;
                        }
                        if d_sq > radius_sq {
                            continue;
                        }
                        // V2 sculpt-pin coarse-LOD suppression was
                        // removed when `SculptDiff` propagation landed:
                        // the engine's `gather_replay_edits` post-
                        // integrate hook now downsamples every level-0
                        // descendant diff into the coarse tile's grid,
                        // so a coarse ancestor carries the sculpt at
                        // its resolution. Skipping it would hide the
                        // sculpt at coarse LOD, which was the prior
                        // V1 limitation this propagation phase exists
                        // to close. Where the dirty-tile pin keeps a
                        // level-0 tile resident in the coarse band,
                        // both render in overlap — accepted tradeoff
                        // for the visible coarse-LOD sculpt.
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
                // Emit eviction for any slot that has live geometry
                // attached — including mid-hot-swap slots whose state
                // is Queued/Submitted but still carry the previous
                // `integrated_token`. Without this the old entity +
                // asset would orphan if the camera moved out of range
                // during a re-bake window.
                if let Some(token) = slot.integrated_token {
                    evicted.push((key, token));
                }
            }
        }
        evicted
    }

    /// Phase 5 stamp invalidation: mark every tile whose AABB
    /// intersects `aabb` for re-bake under the current `Terrain` state
    /// (including the latest `terrain.stamps`).
    ///
    /// **Hot-swap semantics.** Live tiles keep their `integrated_token`
    /// across invalidation — their geometry stays resident in the
    /// renderer until the re-bake completes and `record_integrated`
    /// returns the previous token for deferred eviction. This kills
    /// the 1–2 s "tile flickers out" window that the synchronous
    /// eviction model produced when stamps moved.
    ///
    /// State transitions:
    /// - **Live** → `Queued`. `integrated_token` is preserved; the old
    ///   ECS entity + asset handle stay resident. Generation bumped so
    ///   any in-flight stale result is discarded.
    /// - **Submitted** → `Queued`. Generation bumped so the in-flight
    ///   stale result is discarded by `drain_completed`.
    ///   `integrated_token` (if any, carried over from a prior hot-swap)
    ///   stays put so the swap chain doesn't drop intermediate frames.
    /// - **Queued** → unchanged. The pending bake hasn't started; it
    ///   will read the latest stamp set when submitted.
    /// - **Failed** → `Queued`. Stamps may have changed the failure
    ///   condition; give it another shot.
    ///
    /// Caller is responsible for updating `terrain.stamps` BEFORE
    /// invoking this so the next `submit_pending` baked job sees the
    /// post-change stamp set.
    pub fn invalidate_aabb(&mut self, aabb: Aabb) {
        self.invalidate_aabb_excluding(aabb, &HashSet::new());
    }

    /// Same as [`Self::invalidate_aabb`] but tiles whose `TileKey` is
    /// in `exclude` are left alone.
    ///
    /// **Why exclude.** Sculpt edits live in the in-RAM `AssetEntry`'s
    /// octree, not on disk (until the user saves the scene). Re-baking
    /// a sculpted tile in response to a stamp / region / Inspector
    /// change would drop the sculpt — the new bake reads from
    /// `TerrainFn + stamps` (or the saved `.arvxtile` from the LAST
    /// save), neither of which carry the in-RAM sculpt diff. The
    /// engine passes its `dirty_tiles` set here so sculpted tiles
    /// stay frozen at their authored state until the user explicitly
    /// Reverts them. Matches the `docs/TERRAIN.md` Phase 4.3 edit-
    /// persistence design ("Full baked tile when edited").
    pub fn invalidate_aabb_excluding(
        &mut self,
        aabb: Aabb,
        exclude: &HashSet<TileKey>,
    ) {
        for (key, slot) in self.tiles.iter_mut() {
            if !tile_intersects_aabb(*key, &aabb) {
                continue;
            }
            if exclude.contains(key) {
                continue;
            }
            match slot.state {
                TileState::Live => {
                    // Hot-swap: keep `integrated_token`, just queue
                    // the re-bake. Old geometry stays visible until
                    // `record_integrated` surfaces the prev token.
                    slot.state = TileState::Queued;
                    slot.requested_generation =
                        slot.requested_generation.wrapping_add(1);
                }
                TileState::Submitted => {
                    slot.requested_generation =
                        slot.requested_generation.wrapping_add(1);
                    slot.state = TileState::Queued;
                }
                TileState::Queued => {
                    // No-op; pending bake will use the new stamps.
                }
                TileState::Failed => {
                    slot.state = TileState::Queued;
                }
            }
        }

        // Re-populate submit_scratch so the next `submit_pending`
        // schedules these tiles. `submit_pending` skips slots whose
        // state isn't `Queued`, so duplicate entries (e.g. one from
        // `update_residency`, another from here) are self-cleaning —
        // the first instance transitions the slot to `Submitted` and
        // the second sees the new state and falls through.
        for (key, slot) in &self.tiles {
            if slot.state == TileState::Queued {
                self.submit_scratch.push((*key, slot.camera_dist_sq));
            }
        }
    }

    /// Invalidate every loaded tile. Used by editor flows that change
    /// the global terrain source (TerrainFn parameters, base tier) —
    /// every loaded tile is now stale and the streamer must rebuild
    /// from scratch. Same hot-swap semantics as
    /// [`Self::invalidate_aabb`].
    pub fn invalidate_all(&mut self) {
        self.invalidate_all_excluding(&HashSet::new());
    }

    /// Same as [`Self::invalidate_all`] but tiles in `exclude` are
    /// left alone. See [`Self::invalidate_aabb_excluding`] for the
    /// rationale.
    pub fn invalidate_all_excluding(&mut self, exclude: &HashSet<TileKey>) {
        // f32::MAX would overflow into NaN arithmetic for slot AABB
        // intersection; use a finite but enormous box that comfortably
        // exceeds any plausible world extent (one billion metres on
        // each side).
        let huge = Aabb {
            min: glam::Vec3::splat(-1.0e9),
            max: glam::Vec3::splat(1.0e9),
        };
        self.invalidate_aabb_excluding(huge, exclude);
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
            // Pre-filter Layer-2 stamps down to those whose AABB
            // overlaps the tile's XZ footprint. Skipped entirely
            // when no stamps exist — avoids the empty Vec allocation
            // in the common case.
            let stamps = if terrain.stamps.is_empty() {
                Arc::new(Vec::new())
            } else {
                Arc::new(terrain.stamps.relevant_for_tile(key))
            };
            let job = BakeJob {
                key,
                // V2 L-pyramid: each level uses one tier coarser than
                // the previous. Was hardcoded to level 0 in V1.
                voxel_size_m: terrain.voxel_size_for_level(key.level),
                terrain_fn: Arc::clone(&terrain.terrain_fn),
                generation: slot.requested_generation,
                disk_path,
                stamps,
                regions: Arc::clone(&terrain.regions),
                skirt_depth_m: terrain.skirt_depth_m,
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
            spec: crate::TerrainFnSpec::default(),
            terrain_fn: Arc::new(AllSky),
            stamps: Arc::new(crate::stamp_index::StampIndex::new()),
            regions: Arc::new(crate::TerrainRegionSnapshot::new()),
            render_radius_m: 200.0,
            lod_levels: 1,
            skirt_depth_m: 0.0,
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
            spec: crate::TerrainFnSpec::default(),
            terrain_fn: Arc::new(AllSky),
            stamps: Arc::new(crate::stamp_index::StampIndex::new()),
            regions: Arc::new(crate::TerrainRegionSnapshot::new()),
            render_radius_m: 80.0, // ~1 tile radius
            lod_levels: 1,
            skirt_depth_m: 0.0,
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

    // ── Hot-swap invalidation ────────────────────────────────────────────

    /// `invalidate_aabb` on a Live slot transitions it to Queued but
    /// keeps `integrated_token` set so the old geometry remains
    /// resident in the scene until the new bake lands. This is what
    /// kills the stamp-move flicker.
    #[test]
    fn invalidate_aabb_keeps_token_for_hot_swap() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);
        let mut next_tok: u64 = 100;
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(next_tok);
            next_tok += 1;
        }
        let live_before = s.stats().live;
        assert!(live_before > 0, "test setup: at least one live tile");

        // AABB centred on tile (0, 0, 0).
        let aabb = Aabb {
            min: Vec3::new(20.0, 0.0, 20.0),
            max: Vec3::new(40.0, 64.0, 40.0),
        };
        s.invalidate_aabb(aabb);

        // Slots intersecting the AABB are now Queued but still carry
        // their `integrated_token` — the engine treats them as still
        // resident in the renderer.
        let mut queued_with_token = 0u32;
        for (key, slot) in &s.tiles {
            if tile_intersects_aabb(*key, &aabb) {
                assert_eq!(slot.state, TileState::Queued);
                assert!(
                    slot.integrated_token.is_some(),
                    "hot-swap must keep the previous integrated_token"
                );
                queued_with_token += 1;
            }
        }
        assert!(queued_with_token > 0, "expected at least one hot-swap slot");
    }

    /// `record_integrated` returns the previous live token when a
    /// hot-swap completes — the engine uses this for deferred eviction
    /// of the prior (entity, asset_handle) pair.
    #[test]
    fn record_integrated_returns_prev_token_on_hot_swap() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        // Force one slot Live with a known token, then invalidate + submit + record.
        let key = *s.tiles.keys().next().unwrap();
        {
            let slot = s.tiles.get_mut(&key).unwrap();
            slot.state = TileState::Live;
            slot.integrated_token = Some(7);
        }
        let aabb = Aabb { min: Vec3::splat(-1000.0), max: Vec3::splat(1000.0) };
        s.invalidate_aabb(aabb);
        // Simulate the worker picking up the queued tile.
        s.tiles.get_mut(&key).unwrap().state = TileState::Submitted;

        // New bake completes — record_integrated must surface the old token.
        let prev = s.record_integrated(key, 99);
        assert_eq!(prev, Some(7), "previous token must surface for deferred eviction");
        let slot = &s.tiles[&key];
        assert_eq!(slot.state, TileState::Live);
        assert_eq!(slot.integrated_token, Some(99));
    }

    /// First-time `record_integrated` (no prior live token) returns `None`.
    #[test]
    fn record_integrated_returns_none_on_first_integrate() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);
        let key = *s.tiles.keys().next().unwrap();
        s.tiles.get_mut(&key).unwrap().state = TileState::Submitted;
        assert_eq!(s.record_integrated(key, 1), None);
    }

    /// `invalidate_all` re-queues every loaded tile while preserving
    /// each slot's `integrated_token` for hot-swap.
    #[test]
    fn invalidate_all_keeps_tokens_for_hot_swap() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        let mut next_tok: u64 = 100;
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(next_tok);
            next_tok += 1;
        }
        let n = s.stats().live;
        assert!(n > 0);

        s.invalidate_all();
        assert_eq!(s.stats().live, 0);
        assert!(s.stats().queued >= n);
        // Every slot still has its token.
        for slot in s.tiles.values() {
            assert!(slot.integrated_token.is_some());
        }
    }

    /// `invalidate_aabb` bumps generation on a Submitted slot so the
    /// in-flight result is later discarded by `drain_completed`.
    #[test]
    fn invalidate_aabb_bumps_submitted_generation() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        let key = *s.tiles.keys().next().unwrap();
        let g_before;
        {
            let slot = s.tiles.get_mut(&key).unwrap();
            slot.state = TileState::Submitted;
            slot.requested_generation = 7;
            g_before = slot.requested_generation;
        }

        let aabb = Aabb { min: Vec3::splat(-1000.0), max: Vec3::splat(1000.0) };
        s.invalidate_aabb(aabb);

        let slot = s.tiles.get(&key).unwrap();
        assert_eq!(slot.state, TileState::Queued);
        assert_ne!(slot.requested_generation, g_before,
            "generation should bump so the in-flight result is discarded");
    }

    /// `invalidate_aabb` on a Queued slot is a no-op — the pending
    /// bake hasn't started, it'll naturally use the post-change stamps.
    #[test]
    fn invalidate_aabb_leaves_queued_slots_alone() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        let key = *s.tiles.keys().next().unwrap();
        let g_before = s.tiles[&key].requested_generation;

        let aabb = Aabb { min: Vec3::splat(-1000.0), max: Vec3::splat(1000.0) };
        s.invalidate_aabb(aabb);

        let slot = s.tiles.get(&key).unwrap();
        assert_eq!(slot.state, TileState::Queued);
        assert_eq!(slot.requested_generation, g_before, "Queued generation unchanged");
    }

    /// `invalidate_aabb` doesn't touch tiles outside the AABB.
    #[test]
    fn invalidate_aabb_ignores_tiles_outside_box() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);
        let mut next_tok = 100u64;
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(next_tok);
            next_tok += 1;
        }
        let live_before = s.stats().live;

        let aabb = Aabb {
            min: Vec3::new(10000.0, 0.0, 10000.0),
            max: Vec3::new(10010.0, 64.0, 10010.0),
        };
        s.invalidate_aabb(aabb);
        // Slots outside the AABB stay Live with their token.
        assert_eq!(s.stats().live, live_before);
    }

    /// A residency eviction during a hot-swap window must NOT orphan
    /// the previous token — even if the slot is Queued/Submitted, the
    /// engine still has live geometry attached.
    #[test]
    fn residency_eviction_during_hot_swap_emits_prev_token() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = Terrain {
            bounds: TerrainBounds::Unbounded,
            base_tier: arvx_core::constants::DEFAULT_TERRAIN_TIER,
            spec: crate::TerrainFnSpec::default(),
            terrain_fn: Arc::new(AllSky),
            stamps: Arc::new(crate::stamp_index::StampIndex::new()),
            regions: Arc::new(crate::TerrainRegionSnapshot::new()),
            render_radius_m: 80.0,
            lod_levels: 1,
            skirt_depth_m: 0.0,
        };

        let cam_a = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        let _ = s.update_residency(&terrain, cam_a);
        // Force every slot mid-hot-swap: Queued state, token still set.
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Queued;
            slot.integrated_token = Some(42);
        }

        // Camera moves far away — every previous slot must evict and
        // surface its token so the engine releases the old geometry.
        let cam_b = WorldPosition::new(IVec3::new(500 / 8, 0, 0), Vec3::ZERO);
        let evicted = s.update_residency(&terrain, cam_b);
        assert!(
            evicted.iter().any(|(_, t)| *t == 42),
            "residency eviction during hot-swap must surface the previous token"
        );
    }

    /// `invalidate_aabb_excluding` must NOT change state on excluded
    /// keys. Sculpt-dirty tiles are excluded so the engine's
    /// stamp/region/Inspector invalidations don't drop in-RAM sculpt
    /// edits.
    #[test]
    fn invalidate_aabb_excluding_skips_listed_keys() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        // Force every slot Live with a known token.
        let mut tok: u64 = 100;
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(tok);
            tok += 1;
        }
        // Pick one key as the "sculpt-dirty" tile.
        let excluded_key = *s.tiles.keys().next().unwrap();
        let excluded_token = s.tiles[&excluded_key].integrated_token;
        let mut exclude = HashSet::new();
        exclude.insert(excluded_key);

        // Invalidate a giant AABB covering every slot.
        let aabb = Aabb { min: Vec3::splat(-1e6), max: Vec3::splat(1e6) };
        s.invalidate_aabb_excluding(aabb, &exclude);

        // The excluded slot must remain Live with its token intact.
        let excluded_slot = &s.tiles[&excluded_key];
        assert_eq!(
            excluded_slot.state,
            TileState::Live,
            "excluded key must stay Live"
        );
        assert_eq!(
            excluded_slot.integrated_token, excluded_token,
            "excluded key must keep its integrated_token unchanged"
        );

        // Every other slot must transition to Queued (hot-swap with
        // token preserved).
        for (key, slot) in &s.tiles {
            if *key == excluded_key {
                continue;
            }
            assert_eq!(slot.state, TileState::Queued, "non-excluded key {key:?} should be Queued");
            assert!(slot.integrated_token.is_some(), "non-excluded key keeps token for hot-swap");
        }
    }

    /// `invalidate_all_excluding` mirrors `invalidate_aabb_excluding`
    /// but covers every tile in the streamer regardless of AABB.
    #[test]
    fn invalidate_all_excluding_skips_listed_keys() {
        let mut s = TileStreamer::new(1, 1);
        let terrain = small_terrain();
        let camera = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));
        let _ = s.update_residency(&terrain, camera);

        let mut tok: u64 = 200;
        for slot in s.tiles.values_mut() {
            slot.state = TileState::Live;
            slot.integrated_token = Some(tok);
            tok += 1;
        }
        let excluded_keys: Vec<TileKey> = s.tiles.keys().take(2).copied().collect();
        let exclude: HashSet<TileKey> = excluded_keys.iter().copied().collect();

        s.invalidate_all_excluding(&exclude);

        for key in &excluded_keys {
            assert_eq!(
                s.tiles[key].state,
                TileState::Live,
                "excluded key {key:?} must stay Live under invalidate_all_excluding"
            );
        }
        for (key, slot) in &s.tiles {
            if exclude.contains(key) {
                continue;
            }
            assert_eq!(slot.state, TileState::Queued);
        }
    }

    // ── V2 L-pyramid (Session 1 / L1) ───────────────────────────────────

    /// Sanity: at `lod_levels = 1` the streamer behaves V1-style.
    /// Every materialised tile is at level 0.
    #[test]
    fn residency_emits_only_level0_when_lod_levels_is_1() {
        let mut s = TileStreamer::new(1, 1);
        let mut terrain = small_terrain();
        terrain.bounds = TerrainBounds::Unbounded;
        terrain.render_radius_m = 256.0;
        terrain.lod_levels = 1;
        let cam = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        let _ = s.update_residency(&terrain, cam);
        assert!(s.tiles.len() > 0, "expected some level-0 tiles");
        for key in s.tiles.keys() {
            assert_eq!(key.level, 0, "lod_levels=1 must produce level-0 keys only");
        }
    }

    /// With `lod_levels = 2`, both level-0 (near) and level-1 (far) keys
    /// must materialise. Inner band uses level 0; outer band uses level 1.
    #[test]
    fn residency_emits_multi_level_when_lod_levels_is_2() {
        let mut s = TileStreamer::new(1, 1);
        let mut terrain = small_terrain();
        terrain.bounds = TerrainBounds::Unbounded;
        // Pyramid radius reaches well past the level-0 band so level-1
        // tiles exist around the residency edge.
        terrain.render_radius_m = 384.0;
        terrain.lod_levels = 2;
        let cam = WorldPosition::new(IVec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        let _ = s.update_residency(&terrain, cam);

        let mut level0 = 0;
        let mut level1 = 0;
        for key in s.tiles.keys() {
            match key.level {
                0 => level0 += 1,
                1 => level1 += 1,
                _ => panic!("lod_levels=2 should not emit level {}", key.level),
            }
        }
        assert!(level0 > 0, "expected level-0 tiles in the inner band");
        assert!(level1 > 0, "expected level-1 tiles in the outer band");
    }

    /// The streamer's submit path must pass the LEVEL-aware voxel size
    /// to `BakeJob` — V1 hardcoded `voxel_size_for_level(0)` for every
    /// key, which would mis-bake any level > 0 tile.
    ///
    /// We can't easily intercept `BakeJob`s emitted by `submit_pending`
    /// (the channel is internal to the worker pool). Instead test via
    /// the formula that the level-aware lookup produces a coarser voxel
    /// size at level 1 than level 0 — confirming the lookup is exposed
    /// per-level. The wiring of `voxel_size_for_level(key.level)` into
    /// `submit_pending` is verified by reading the source.
    #[test]
    fn voxel_size_for_level_is_monotone_coarsening() {
        let terrain = small_terrain();
        let vs0 = terrain.voxel_size_for_level(0);
        let vs1 = terrain.voxel_size_for_level(1);
        let vs2 = terrain.voxel_size_for_level(2);
        assert!(vs1 > vs0, "level 1 must be coarser than level 0");
        assert!(vs2 > vs1, "level 2 must be coarser than level 1");
        assert!((vs1 / vs0 - 2.0).abs() < 1e-5, "each level should double voxel size");
    }

    /// `lod_band` closed form: at `lod_levels = N`, level `N - 1`'s outer
    /// edge must equal `render_radius_m` and level 0's inner edge must
    /// be 0. Total band coverage = `[0, render_radius)`.
    #[test]
    fn lod_band_covers_full_render_radius_geometrically() {
        let r = 256.0;
        for n in 1u8..=4 {
            let (lo0, _) = lod_band(r, n, 0);
            let (_, hi_last) = lod_band(r, n, n - 1);
            assert_eq!(lo0, 0.0, "level 0 inner must start at 0 (N={n})");
            assert!(
                (hi_last - r).abs() < 1e-3,
                "level N-1 outer must equal render_radius (N={n}, hi={hi_last})",
            );
        }

        // Geometric-doubling property: each level's width is 2× the
        // previous level's width. Check for N=3.
        let (lo0, hi0) = lod_band(r, 3, 0);
        let (lo1, hi1) = lod_band(r, 3, 1);
        let (lo2, hi2) = lod_band(r, 3, 2);
        let w0 = hi0 - lo0;
        let w1 = hi1 - lo1;
        let w2 = hi2 - lo2;
        assert!((w1 / w0 - 2.0).abs() < 1e-3, "w1 must be 2× w0");
        assert!((w2 / w1 - 2.0).abs() < 1e-3, "w2 must be 2× w1");
        // Bands are contiguous, non-overlapping.
        assert!((hi0 - lo1).abs() < 1e-3, "hi0 must meet lo1");
        assert!((hi1 - lo2).abs() < 1e-3, "hi1 must meet lo2");
    }

    // ── V2 sculpt-pinned residency ─────────────────────────────────────

    /// A dirty-pinned level-0 tile materialises even when it sits far
    /// beyond the level-0 band — sculpted regions can't drop to a
    /// coarser LOD without losing the sculpt diff.
    #[test]
    fn sculpt_pinned_dirty_tile_loads_beyond_band() {
        let mut s = TileStreamer::new(1, 1);
        let mut terrain = small_terrain();
        terrain.bounds = TerrainBounds::Unbounded;
        terrain.render_radius_m = 384.0;
        terrain.lod_levels = 2;
        // Tile at (12, 0, 0) sits at world centre (800, 32, 32) —
        // distance ~800 m from origin, well past `render_radius_m = 384`.
        let far_pinned = TileKey::level0(12, 0, 0);
        let mut pinned = HashSet::new();
        pinned.insert(far_pinned);
        let cam = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);

        let _ = s.update_residency_with_pinned(&terrain, cam, &pinned);

        assert!(
            s.tiles.contains_key(&far_pinned),
            "dirty-pinned tile must materialise regardless of distance"
        );
        let slot = &s.tiles[&far_pinned];
        assert!(
            slot.camera_dist_sq > terrain.render_radius_m * terrain.render_radius_m,
            "pinned tile's distance should be past render_radius — confirms pin bypass"
        );
    }

    /// A level-1 tile whose footprint overlaps a pinned level-0 tile
    /// **coexists** with the pin under V2 diff propagation. Before
    /// `SculptDiff` propagation we suppressed the coarse ancestor to
    /// avoid z-fight; now the engine's post-integrate replay puts the
    /// downsampled sculpt into the coarse tile too, so we keep it
    /// loaded and accept the overlap.
    #[test]
    fn sculpt_pinned_does_not_suppress_overlapping_coarse() {
        let mut s = TileStreamer::new(1, 1);
        let mut terrain = small_terrain();
        terrain.bounds = TerrainBounds::Unbounded;
        terrain.render_radius_m = 512.0;
        terrain.lod_levels = 2;
        // Pin a level-0 tile that falls inside what would otherwise be
        // a level-1 band region. Pick a level-0 tile at the +X side
        // beyond the level-0 band's outer edge.
        // band_outer(0) at lod_levels=2 = render_radius * 1/3 ≈ 170 m
        // Tile (4, 0, 0) has centre (288, 32, 32) → ~290 m: in level-1 band.
        let pinned_l0 = TileKey::level0(4, 0, 0);
        let mut pinned = HashSet::new();
        pinned.insert(pinned_l0);
        let cam = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);

        let _ = s.update_residency_with_pinned(&terrain, cam, &pinned);

        // The pinned level-0 must be present.
        assert!(
            s.tiles.contains_key(&pinned_l0),
            "pinned level-0 tile must be present"
        );

        // Level-1 tile (2, 0, 0) covers world [256, 384) × [0, 128)² —
        // overlaps pinned_l0's footprint [256, 320) × [0, 64)² × [0, 64).
        // It must ALSO be present so the engine bakes it and replays
        // the descendant diff into the coarse grid.
        let overlap_l1 = TileKey { level: 1, x: 2, y: 0, z: 0 };
        assert!(
            s.tiles.contains_key(&overlap_l1),
            "level-1 tile overlapping a pinned level-0 must coexist \
             (no suppression — the diff-propagation path renders the \
             sculpt at coarse LOD too)"
        );

        // A level-1 tile elsewhere (no overlap) is still emitted normally.
        // Tile (-3, 0, 0) covers [-384, -256) on x — no overlap with pinned.
        let unaffected_l1 = TileKey { level: 1, x: -3, y: 0, z: 0 };
        assert!(
            s.tiles.contains_key(&unaffected_l1),
            "non-overlapping level-1 tile should still emit"
        );
    }

    /// When the pinned set is empty, residency behaves bit-identically
    /// to the no-pin version (regression guard).
    #[test]
    fn empty_pinned_set_matches_unpinned_behavior() {
        let mut s_pinned = TileStreamer::new(1, 1);
        let mut s_plain = TileStreamer::new(1, 1);
        let mut terrain = small_terrain();
        terrain.bounds = TerrainBounds::Unbounded;
        terrain.render_radius_m = 256.0;
        terrain.lod_levels = 2;
        let cam = WorldPosition::new(IVec3::ZERO, Vec3::new(32.0, 32.0, 32.0));

        let _ = s_pinned.update_residency_with_pinned(&terrain, cam, &HashSet::new());
        let _ = s_plain.update_residency(&terrain, cam);

        let keys_pinned: std::collections::HashSet<TileKey> =
            s_pinned.tiles.keys().copied().collect();
        let keys_plain: std::collections::HashSet<TileKey> =
            s_plain.tiles.keys().copied().collect();
        assert_eq!(keys_pinned, keys_plain, "empty pinned set must be a no-op");
    }
}
