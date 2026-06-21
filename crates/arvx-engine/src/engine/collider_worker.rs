//! Collider-cache rebuild on a worker thread. Phase E2 of
//! `docs/PERF_DEBT.md`.
//!
//! The collider rebuild walks an entity's octree to compute the tight
//! occupied AABB + (for Auto-shape rigidbodies) a coarse-grid bucket
//! list. Pre-E2 it ran inline on the sim tick — `rebuild_collider_caches`
//! (world-walk) or `rebuild_collider_cache_for(entity)` (narrow).
//!
//! E2 moves the heavy compute to a dedicated worker, mirroring the
//! shape of [`crate::engine::paint_walk::PaintWalkWorker`]:
//!
//! ```text
//!   sim tick N:    drain collider_caches_dirty → build batch → submit
//!   worker:        recv batch → compute caches per entity → send result
//!   sim tick N+k:  try_recv result → insert ColliderCache per entity
//! ```
//!
//! Today `collider_caches_dirty` only fires on play-mode entry / project
//! load (cold paths), so the worker mostly serves as architectural
//! scaffolding — but it lets a future per-stamp `geometry_dirty` setter
//! land without re-introducing the sim-blocking pattern.

use std::sync::Arc;
use std::time::Duration;

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use arvx_physics::rigid_body::ColliderShape;
use arvx_render::arvx_scene_manager::WalkSnapshot;

use crate::components::{ColliderCache, RigidBody};

/// Per-entity inputs captured by the sim. Send-able so the worker can
/// own them across the channel.
#[derive(Clone)]
pub(crate) struct ColliderJob {
    pub entity: hecs::Entity,
    pub rb: RigidBody,
    /// `None` when the entity has no octree spatial (Box/Sphere
    /// fallback path on rigidbodies without a Renderable).
    pub spatial: Option<JobSpatial>,
    pub scale: glam::Vec3,
    /// Diagnostic — used by the `[ColliderCache]` log line so we can
    /// see which entity the worker is processing.
    pub name: String,
    pub pos: glam::Vec3,
}

#[derive(Clone)]
pub(crate) struct JobSpatial {
    pub root_offset: u32,
    pub depth: u8,
    pub len: u32,
    pub base_voxel_size: f32,
    pub grid_origin: glam::Vec3,
    pub aabb_min: glam::Vec3,
    pub aabb_max: glam::Vec3,
}

pub(crate) struct ColliderBatch {
    pub snapshot: WalkSnapshot,
    pub jobs: Vec<ColliderJob>,
}

pub(crate) struct ColliderResult {
    pub entries: Vec<(hecs::Entity, ColliderCache)>,
    pub worker_duration: Duration,
}

pub(crate) struct ColliderWorker {
    inbox_tx: Option<Sender<ColliderBatch>>,
    outbox_rx: Receiver<ColliderResult>,
    handle: Option<std::thread::JoinHandle<()>>,
    in_flight: std::sync::atomic::AtomicBool,
}

impl ColliderWorker {
    pub fn spawn() -> Self {
        let (inbox_tx, inbox_rx) = bounded::<ColliderBatch>(1);
        let (outbox_tx, outbox_rx) = unbounded::<ColliderResult>();
        let handle = std::thread::Builder::new()
            .name("arvx-collider-build".to_string())
            .spawn(move || worker_loop(inbox_rx, outbox_tx))
            .expect("spawn arvx-collider-build thread");
        Self {
            inbox_tx: Some(inbox_tx),
            outbox_rx,
            handle: Some(handle),
            in_flight: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn is_idle(&self) -> bool {
        !self.in_flight.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn submit(&self, batch: ColliderBatch) {
        self.in_flight
            .store(true, std::sync::atomic::Ordering::Release);
        let Some(tx) = self.inbox_tx.as_ref() else {
            eprintln!("[collider-worker] submit after shutdown");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        };
        if let Err(e) = tx.send(batch) {
            eprintln!("[collider-worker] send failed: {e}");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn try_recv(&self) -> Option<ColliderResult> {
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

impl Drop for ColliderWorker {
    fn drop(&mut self) {
        drop(self.inbox_tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(inbox: Receiver<ColliderBatch>, outbox: Sender<ColliderResult>) {
    while let Ok(batch) = inbox.recv() {
        let t0 = std::time::Instant::now();
        let ColliderBatch { snapshot, jobs } = batch;
        let octree = snapshot.octree_data;
        let bricks = snapshot.brick_pool_data;
        let mut entries = Vec::with_capacity(jobs.len());
        for job in jobs {
            let cache = build_one(&octree, &bricks, &job);
            entries.push((job.entity, cache));
        }
        let result = ColliderResult {
            entries,
            worker_duration: t0.elapsed(),
        };
        if outbox.send(result).is_err() {
            return;
        }
    }
}

fn build_one(
    octree: &[u32],
    bricks: &[u32],
    job: &ColliderJob,
) -> ColliderCache {
    let all_nodes: &[u32] = octree;
    let brick_data: &[u32] = bricks;

    // Tight-AABB from actual occupied voxels.
    let tight_local = job.spatial.as_ref().and_then(|sp| {
        crate::play_mode::compute_tight_local_aabb(
            all_nodes,
            brick_data,
            sp.root_offset as usize,
            sp.depth,
            sp.len,
            sp.base_voxel_size,
            sp.grid_origin,
        )
    });

    let (aabb_half, local_center) = match tight_local {
        Some(t) => (
            t.half_extents() * job.scale,
            (t.min + t.max) * 0.5 * job.scale,
        ),
        None => (glam::Vec3::splat(0.5), glam::Vec3::ZERO),
    };

    if let Some(ref sp) = job.spatial {
        eprintln!(
            "[ColliderCache] '{}' pos={:?} scale={:?} \
             padded_aabb={:?}..{:?} tight_local={:?} \
             aabb_half={:?} local_center={:?}",
            job.name,
            job.pos,
            job.scale,
            sp.aabb_min,
            sp.aabb_max,
            tight_local,
            aabb_half,
            local_center,
        );
    }

    let (resolved_shape, voxel_coords, voxel_size) = match &job.rb.collider_shape {
        ColliderShape::Auto => {
            if let Some(ref sp) = job.spatial {
                let (coords, cell_size) = crate::play_mode::build_coarse_collider(
                    all_nodes,
                    brick_data,
                    sp.root_offset as usize,
                    sp.depth,
                    sp.len,
                    sp.base_voxel_size,
                    job.rb.collider_cell_size,
                );
                if coords.is_empty() {
                    (ColliderShape::Box, Vec::new(), 0.0)
                } else {
                    (ColliderShape::Auto, coords, cell_size)
                }
            } else {
                (ColliderShape::Box, Vec::new(), 0.0)
            }
        }
        other => (other.clone(), Vec::new(), 0.0),
    };

    let (grid_origin, tree_depth) = match job.spatial.as_ref() {
        Some(sp) => (sp.grid_origin, sp.depth),
        None => (glam::Vec3::ZERO, 0),
    };

    ColliderCache {
        shape: resolved_shape,
        voxel_coords,
        collider_cell_size: voxel_size,
        aabb_half,
        local_center,
        grid_origin,
        tree_depth,
    }
}
