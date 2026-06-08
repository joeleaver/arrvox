//! Off-thread `.arvx` asset loader (task #8 — smooth the per-big-asset
//! integrate hitch on scene load).
//!
//! Scene load can reference assets of several million voxels each. The
//! expensive part — read + octree compact/dedup/morton + prefilter +
//! mesh deserialize — used to run inline on the sim thread under the
//! `scene_mgr` lock, freezing presentation for ~1 s per big asset. This
//! worker moves the *build* off the sim thread: it produces a fully
//! private, file-local [`LoadedAsset`] (touching no shared state), which
//! the sim then SPLICES into the scene pools in a bounded main-thread
//! pass via `ArvxSceneManager::integrate_loaded_asset`.
//!
//! ```text
//!   sim tick N:    pop (entity, path) → submit AssetLoadJob
//!   worker:        recv job → build_loaded_asset(path) → send result
//!   sim tick N+k:  try_recv result → integrate_loaded_asset (splice) →
//!                  wire the waiting entities' Renderable
//! ```
//!
//! Mirrors the bounded(1) inbox + unbounded outbox + `in_flight` atomic
//! shape of [`super::paint_walk::PaintWalkWorker`] (Phase E1) and
//! [`super::collider_worker`] (Phase E2).

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError};

use arvx_render::{ArvxSceneManager, LoadedAsset};

/// One queued `.arvx` load. Built on the sim thread, consumed on the worker.
pub(crate) struct AssetLoadJob {
    pub entity: hecs::Entity,
    pub path: std::path::PathBuf,
}

/// Outcome of one build. The sim splices `Ok` results into the scene
/// pools on a later tick; `Err` is logged and the entity stays
/// geometry-less (recoverable on re-import). `entity` is the job's
/// originating entity; the drain maps `path` back to the full set of
/// entities waiting on this load (instances share one build).
pub(crate) struct AssetLoadResult {
    #[allow(dead_code)]
    pub entity: hecs::Entity,
    pub path: std::path::PathBuf,
    pub result: Result<LoadedAsset, String>,
}

/// Dedicated worker thread for off-thread `.arvx` asset builds.
///
/// Backpressure: `inbox` is bounded(1). The sim checks [`Self::is_idle`]
/// before submitting; a build already in flight means the sim keeps the
/// job queued and tries next tick. One build at a time keeps the worker
/// simple — concurrent builds are possible (each is fully private) but
/// unnecessary while the splice still serializes on the `scene_mgr` lock.
pub(crate) struct AssetLoadWorker {
    /// `None` only during shutdown — dropping the Sender disconnects the
    /// worker's recv loop.
    inbox_tx: Option<Sender<AssetLoadJob>>,
    outbox_rx: Receiver<AssetLoadResult>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// Set on submit, cleared when the matching result is drained.
    in_flight: std::sync::atomic::AtomicBool,
}

impl AssetLoadWorker {
    pub fn spawn() -> Self {
        let (inbox_tx, inbox_rx) = bounded::<AssetLoadJob>(1);
        let (outbox_tx, outbox_rx) = unbounded::<AssetLoadResult>();
        let handle = std::thread::Builder::new()
            .name("arvx-asset-load".to_string())
            .spawn(move || worker_loop(inbox_rx, outbox_tx))
            .expect("spawn arvx-asset-load thread");
        Self {
            inbox_tx: Some(inbox_tx),
            outbox_rx,
            handle: Some(handle),
            in_flight: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// True when no build is in flight. The sim submits the next queued
    /// job only when this holds — the bounded(1) inbox would otherwise
    /// drop work on the floor.
    pub fn is_idle(&self) -> bool {
        !self.in_flight.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Submit a build job. Caller must ensure [`Self::is_idle`] was true.
    pub fn submit(&self, job: AssetLoadJob) {
        self.in_flight
            .store(true, std::sync::atomic::Ordering::Release);
        let Some(tx) = self.inbox_tx.as_ref() else {
            eprintln!("[asset-load-worker] submit after shutdown");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        };
        if let Err(e) = tx.send(job) {
            eprintln!("[asset-load-worker] send failed: {e}");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    /// Drain a completed build, if any. Returns `None` while the worker
    /// is still building or idle.
    pub fn try_recv(&self) -> Option<AssetLoadResult> {
        match self.outbox_rx.try_recv() {
            Ok(r) => {
                self.in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                Some(r)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                None
            }
        }
    }
}

impl Drop for AssetLoadWorker {
    fn drop(&mut self) {
        // Dropping the sender disconnects the worker's recv loop; it
        // returns Err from inbox.recv() and exits naturally. Then join so
        // the thread tears down before EngineState releases.
        drop(self.inbox_tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(inbox: Receiver<AssetLoadJob>, outbox: Sender<AssetLoadResult>) {
    while let Ok(job) = inbox.recv() {
        let AssetLoadJob { entity, path } = job;
        let result = ArvxSceneManager::build_loaded_asset(&path);
        if outbox
            .send(AssetLoadResult { entity, path, result })
            .is_err()
        {
            return; // sim is gone — exit cleanly
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a small sphere `.arvx` to `dir` (pure CPU, no GPU).
    fn write_sphere_arvx(dir: &std::path::Path) -> std::path::PathBuf {
        let voxel_size = 0.1_f32;
        let radius = 1.0_f32;
        let natural = arvx_core::Aabb::new(glam::Vec3::splat(-1.6), glam::Vec3::splat(1.6));
        let aabb = arvx_core::pad_to_pow2_cubic(&natural, voxel_size);
        let mut sdf = |ps: &[glam::Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<glam::Vec3>)> {
            ps.iter()
                .map(|p| (p.length() - radius, 0u16, 0u16, 0u8, 0u32, None))
                .collect()
        };
        let artifact = arvx_core::voxelize_to_artifact(&mut sdf, &aabb, voxel_size, 0)
            .expect("sphere voxelizes");
        let path = dir.join("sphere.arvx");
        arvx_core::asset_file::write_artifact_rkp(
            &path,
            &artifact,
            aabb.min.to_array(),
            aabb.max.to_array(),
            voxel_size,
        )
        .expect("write_artifact_rkp");
        path
    }

    /// End-to-end worker round-trip: submit a job, poll until the build
    /// lands, splice it into a fresh scene manager. The spliced asset
    /// must carry voxels — proving the off-thread build → main-thread
    /// integrate path produces real geometry.
    #[test]
    fn worker_builds_and_integrates() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = write_sphere_arvx(tmp.path());

        let worker = AssetLoadWorker::spawn();
        assert!(worker.is_idle());
        let entity = hecs::Entity::DANGLING;
        worker.submit(AssetLoadJob { entity, path: path.clone() });
        assert!(!worker.is_idle(), "in_flight set after submit");

        // Poll for the result (build is fast for a small sphere).
        let mut result = None;
        for _ in 0..2000 {
            if let Some(r) = worker.try_recv() {
                result = Some(r);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let result = result.expect("worker should produce a result");
        assert!(worker.is_idle(), "in_flight cleared after recv");
        assert_eq!(result.path, path);
        let loaded = result.result.expect("build should succeed");

        // Splice into a fresh manager — the integrate path the drain runs.
        let mut sm = arvx_render::ArvxSceneManager::new(1_000_000);
        let (_handle, info) = sm.integrate_loaded_asset(loaded);
        assert!(info.voxel_count > 0, "spliced asset must carry voxels");
        assert!(info.leaf_attr_slot_count > 0);
    }
}
