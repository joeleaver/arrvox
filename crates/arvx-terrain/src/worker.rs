//! Dedicated worker pool for terrain tile bakes.
//!
//! Phase 2 stands up a fixed-size pool of background threads dedicated
//! to `bake_tile()` calls. Each `BakeJob` consists of a `TileKey`, a
//! voxel size, and an `Arc<dyn TerrainFn>`; each thread pulls jobs
//! from a shared `crossbeam::channel` and pushes results onto a
//! single result channel that the streamer drains on the main thread.
//!
//! ## Why a dedicated pool, not the existing `bake_worker`?
//!
//! The engine's `bake_worker` is a single-threaded queue that
//! multiplexes procedural edits and generator emissions and serialises
//! every result against `scene_mgr.lock()`. Procedural edits need
//! sub-100 ms turnaround for sculpt drag-feel; terrain bakes are
//! seconds-long CPU jobs and would block the queue. Splitting them
//! gives terrain its own thread pool and keeps the procedural path
//! responsive.
//!
//! ## Backpressure
//!
//! The streamer tracks `(submitted, completed)` counts (`AtomicUsize`)
//! and only submits new jobs when `submitted - completed <
//! max_in_flight`. Jobs themselves go through an *unbounded* inbox —
//! the streamer caps concurrency, not the channel.
//!
//! ## Panic safety
//!
//! `bake_tile` invokes user-supplied `TerrainFn::sample` once per
//! voxel position. A panicking implementation could otherwise tear
//! the worker thread down silently and stall streaming forever; the
//! worker wraps the bake body in [`std::panic::catch_unwind`] and
//! reports a failed result instead.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam::channel::{unbounded, Receiver, Sender};

// `bake_tile_with_skirts` is the V2 LOD-pyramid path; the legacy
// `bake_tile` wrapper (no skirts) still lives in `crate::bake` for
// tests / persist roundtrip.
#[allow(unused_imports)]
use crate::bake::bake_tile;
use crate::baked_tile::BakedTile;
use crate::persist::read_baked_tile;
use crate::region_snapshot::TerrainRegionSnapshot;
use crate::stamp::Stamp;
use crate::terrain_fn::TerrainFn;
use crate::tile_key::TileKey;

/// A single bake job submitted to the worker pool. Cloning the
/// `Arc<dyn TerrainFn>` is cheap; the worker runs against the cloned
/// pointer so a Terrain swap on the main thread doesn't race a
/// bake in flight (Phase 5 will use `generation` to discard stale
/// results).
pub struct BakeJob {
    /// Which tile to bake.
    pub key: TileKey,
    /// Voxel size in metres (derived from the terrain's tier table).
    pub voxel_size_m: f32,
    /// Procedural source; cloned cheaply from `Terrain::terrain_fn`.
    pub terrain_fn: Arc<dyn TerrainFn>,
    /// Per-tile request generation. The streamer bumps this whenever
    /// it (re-)submits the same tile (evict-then-reload during camera
    /// oscillation, stamp invalidation, sculpt-edit re-bake). The
    /// streamer drops results whose generation no longer matches
    /// `slot.requested_generation`.
    pub generation: u64,
    /// Phase 4.4: optional `.arvxtile` path. When `Some` and the file
    /// exists on disk, the worker loads the tile from disk instead of
    /// running `TerrainFn`-driven voxelization. Caller resolves this
    /// via `persist::tile_path(scene_dir, key)`.
    pub disk_path: Option<std::path::PathBuf>,
    /// Phase 5: Layer-2 stamps overlapping this tile, in composition
    /// order. Pre-filtered by the streamer's `submit_pending` from
    /// the global `StampIndex` so the worker never iterates stamps
    /// outside the tile's XZ footprint. Empty for scenes with no
    /// stamps or tiles outside every stamp's reach. Wrapped in `Arc`
    /// so the streamer can re-submit the same tile under the same
    /// stamp set without re-allocating.
    pub stamps: Arc<Vec<Stamp>>,
    /// Phase 7: biome region snapshot. Shared by `Arc` across every
    /// in-flight bake job — the streamer hands every worker the same
    /// snapshot, and rebuilds it (allocating a fresh `Arc`) whenever
    /// the region set changes. Always present, even for scenes
    /// without regions (then it's the empty default).
    pub regions: Arc<TerrainRegionSnapshot>,
    /// V2 LOD pyramid: lateral skirt depth (m). `0.0` disables.
    /// Threaded from `Terrain::skirt_depth_m`.
    pub skirt_depth_m: f32,
    /// World-envelope floor Y, in absolute world coords. The bake
    /// clamps the composed surface height to `world_floor_y + 2 *
    /// voxel_size_m` so stamps (or a misbehaving TerrainFn) can't
    /// drive the entire tile's surface below the world's solid
    /// envelope and create a fall-through hole. `None` for
    /// `Unbounded` terrains, which have no floor.
    pub world_floor_y: Option<f32>,
}

/// Outcome of one worker job. `baked = None` means the bake failed —
/// either a panic in `TerrainFn::sample` or `voxelize_to_artifact`
/// returning `None` (pool exhaustion / degenerate AABB).
pub struct BakeJobResult {
    /// Originating tile key. The streamer's slot lookup is keyed on
    /// this; an orphan key (slot was evicted while bake ran) is
    /// discarded.
    pub key: TileKey,
    /// Echo of the originating `BakeJob.generation`. The streamer
    /// drops results whose generation no longer matches the slot's
    /// `requested_generation`.
    pub generation: u64,
    /// `Some(baked)` on success; `None` on panic or `voxelize_to_artifact`
    /// failure.
    pub baked: Option<BakedTile>,
}

/// Multi-threaded worker pool driving `bake_tile`. Owns its threads;
/// dropping the pool waits for in-flight bakes to complete (via the
/// `JoinHandle::join` in `Drop`).
pub struct BakeWorker {
    inbox_tx: Option<Sender<BakeJob>>,
    outbox_rx: Receiver<BakeJobResult>,
    handles: Vec<JoinHandle<()>>,
    submitted: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    worker_count: usize,
}

impl BakeWorker {
    /// Spawn `worker_count` worker threads. Pick `2` for V1; the value
    /// is tunable from the streamer's constructor (Phase 9 / V2 may
    /// derive it from `rayon::current_num_threads`).
    pub fn spawn(worker_count: usize) -> Self {
        let worker_count = worker_count.max(1);
        let (inbox_tx, inbox_rx) = unbounded::<BakeJob>();
        let (outbox_tx, outbox_rx) = unbounded::<BakeJobResult>();
        let submitted = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::with_capacity(worker_count);
        for wid in 0..worker_count {
            let rx = inbox_rx.clone();
            let tx = outbox_tx.clone();
            let completed = completed.clone();
            let handle = std::thread::Builder::new()
                .name(format!("arvx-terrain-bake-{wid}"))
                .spawn(move || worker_loop(rx, tx, completed))
                .expect("spawn arvx-terrain-bake thread");
            handles.push(handle);
        }
        // `outbox_tx` was cloned once per worker above. The original
        // is dropped here so the channel closes naturally when all
        // workers exit (each worker drops its clone on exit).
        drop(outbox_tx);

        Self {
            inbox_tx: Some(inbox_tx),
            outbox_rx,
            handles,
            submitted,
            completed,
            worker_count,
        }
    }

    /// Worker count this pool was spawned with.
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Number of bake jobs currently in flight (submitted but not yet
    /// returned via `try_recv`). The streamer uses this to enforce
    /// `in_flight < max_in_flight`.
    pub fn in_flight(&self) -> usize {
        let submitted = self.submitted.load(Ordering::Acquire);
        let completed = self.completed.load(Ordering::Acquire);
        submitted.saturating_sub(completed)
    }

    /// Submit a bake job. Returns `false` if the worker pool has been
    /// shut down (Drop in progress); otherwise increments the in-flight
    /// counter and sends.
    pub fn submit(&self, job: BakeJob) -> bool {
        let Some(tx) = self.inbox_tx.as_ref() else {
            eprintln!("[arvx-terrain-bake] submit after shutdown");
            return false;
        };
        // Bump submitted BEFORE send so a try_recv racing with this
        // submit can't observe completed > submitted. The worker
        // bumps completed AFTER send-result.
        self.submitted.fetch_add(1, Ordering::AcqRel);
        if let Err(e) = tx.send(job) {
            eprintln!("[arvx-terrain-bake] send failed: {e}");
            // Roll the counter back so in_flight() reflects reality.
            self.submitted.fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        true
    }

    /// Drain any completed result. Non-blocking; returns `None` when
    /// no result is queued.
    pub fn try_recv(&self) -> Option<BakeJobResult> {
        self.outbox_rx.try_recv().ok()
    }
}

impl Drop for BakeWorker {
    fn drop(&mut self) {
        // Dropping the sender closes the inbox; each worker exits its
        // `while recv()` loop and drops its outbox sender clone.
        drop(self.inbox_tx.take());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    inbox: Receiver<BakeJob>,
    outbox: Sender<BakeJobResult>,
    completed: Arc<AtomicUsize>,
) {
    while let Ok(job) = inbox.recv() {
        let BakeJob {
            key,
            voxel_size_m,
            terrain_fn,
            generation,
            disk_path,
            stamps,
            regions,
            skirt_depth_m,
            world_floor_y,
        } = job;

        // Phase 4.4: if a `.arvxtile` is on disk for this key, load
        // it instead of re-running `TerrainFn`. Falls back to the
        // procedural bake on any read error so a corrupted file
        // doesn't permanently block streaming the tile.
        //
        // Phase 5 note: disk-loaded tiles already contain the stamp
        // contribution baked-in (the persisted artifact is post-
        // composition), so we deliberately do NOT re-apply stamps
        // here. The slot bumps its generation when stamps change,
        // which invalidates the on-disk cache for that tile via the
        // editor's "Edits → Revert stamps" pathway in Phase 9.
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            if let Some(ref p) = disk_path {
                if p.exists() {
                    match read_baked_tile(p) {
                        Ok((artifact, mesh, vs)) => {
                            // Loaded from cache — not freshly baked, so no
                            // write-through.
                            return (
                                Some(BakedTile {
                                    key,
                                    artifact,
                                    mesh,
                                    voxel_size_m: vs,
                                    bake_time_ms: 0.0,
                                }),
                                false,
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[arvx-terrain-bake] read {} failed: {e}; \
                                 falling back to TerrainFn",
                                p.display(),
                            );
                        }
                    }
                }
            }
            (
                crate::bake::bake_tile_with_skirts(
                    key,
                    voxel_size_m,
                    &*terrain_fn,
                    stamps.as_slice(),
                    regions.as_ref(),
                    skirt_depth_m,
                    world_floor_y,
                ),
                true,
            )
        }));
        drop(terrain_fn);
        drop(stamps);
        drop(regions);

        let (baked, baked_fresh) = match result {
            Ok(t) => t,
            Err(payload) => {
                let msg = panic_payload_str(&payload);
                eprintln!(
                    "[arvx-terrain-bake] tile ({}, {}, {}, lvl {}) panic in bake_tile: {msg}",
                    key.x, key.y, key.z, key.level
                );
                (None, false)
            }
        };

        // Write-through cache: persist freshly-baked tiles to their
        // `.arvxtile`. `disk_path` is only `Some` when the streamer
        // judged the on-disk cache valid (signature matches the live
        // terrain), so every written tile stays consistent with the
        // cache signature — exploring an area bakes its tiles once, then
        // every later load reads them from disk instead of re-running
        // TerrainFn. Runs on the worker thread, off the render path.
        if baked_fresh {
            if let (Some(p), Some(bt)) = (&disk_path, &baked) {
                if let Err(e) =
                    crate::persist::write_tile_to_path(p, &bt.artifact, bt.voxel_size_m)
                {
                    eprintln!(
                        "[arvx-terrain-bake] write-through {} failed: {e}",
                        p.display(),
                    );
                }
            }
        }

        // Bump completed BEFORE send so `in_flight()` reads
        // monotonically — sim-side try_recv that wins the race still
        // sees an accurate count.
        completed.fetch_add(1, Ordering::AcqRel);
        if let Err(e) = outbox.send(BakeJobResult { key, generation, baked }) {
            // Outbox closed → streamer is shutting down. Stop.
            eprintln!("[arvx-terrain-bake] outbox send failed: {e}; worker exiting");
            return;
        }
    }
}

fn panic_payload_str(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}
