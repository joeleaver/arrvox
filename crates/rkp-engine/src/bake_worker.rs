//! Async bake worker.
//!
//! Spawns a single background thread that owns its own GPU evaluator
//! and chews through [`BakeRequest`]s one at a time. For each job the
//! worker:
//!
//! 1. Runs `voxelize_to_artifact` against its own private pools (no
//!    shared state, no lock).
//! 2. Acquires the shared `RkpSceneManager` lock, frees the entity's
//!    previous geometry allocation if any, then calls
//!    `integrate_artifact` to splice the new one into the scene
//!    pools. This is the hot part — ~60-90 ms on a 20 m procedural
//!    for the 28 M u32 brick-cell remap + octree node/face-link
//!    remap.
//! 3. Releases the lock and sends the engine a slim `BakeResult`
//!    carrying just the freshly allocated `SpatialData` + summary.
//!
//! The engine's `drain_bake_results` tick now does O(1) ECS updates
//! rather than touching scene pools — the main thread's bake-related
//! stall drops from "full integrate" to ~microseconds. Lock
//! contention with `render_frame`'s `geometry_upload` is rare (that
//! lock is held for ~1 ms per dirty frame) and naturally serialized
//! by the bake-in-flight flag on the procedural.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam::channel::{Receiver, Sender};

use rkp_render::rkp_scene_manager::RkpSceneManager;
use rkp_render::proc_sample::GpuEvaluator;

use crate::components::SpatialData;

/// One bake job. `generation` is compared against the entity's current
/// generation at result-receive time; stale results get dropped.
pub struct BakeRequest {
    pub entity: hecs::Entity,
    pub generation: u64,
    pub scene_id: u32,
    /// Flattened procedural opcode stream. Cheap to clone and tree-free,
    /// so we don't ship the full `ProceduralObject` across threads.
    pub instructions: Vec<rkp_procedural::flatten::ProcInstruction>,
    pub aabb: rkp_core::Aabb,
    pub voxel_size: f32,
    /// Root.transform.scale at the moment this request was captured.
    /// Worker echoes it back in the result so integrate can set
    /// `last_evaluated_root_scale` to what the voxels *actually*
    /// represent — not whatever the user has since dragged the tree
    /// to. Otherwise the preview-multiplier math drifts every time
    /// the user edits during a bake.
    pub root_scale: glam::Vec3,
    /// Previous geometry allocation to free under the same lock as
    /// the new integrate. The engine reads it off `Renderable` at
    /// enqueue time; we don't want the worker to round-trip through
    /// the ECS. `None` on an initial bake.
    pub prev_spatial: Option<SpatialData>,
}

/// Worker → engine handoff. Integrate has already run on the worker
/// by the time the engine sees this.
pub struct BakeResult {
    pub entity: hecs::Entity,
    pub generation: u64,
    pub scene_id: u32,
    pub aabb: rkp_core::Aabb,
    pub voxel_size: f32,
    pub root_scale: glam::Vec3,
    pub outcome: BakeOutcome,
}

pub enum BakeOutcome {
    /// Voxelize + integrate both succeeded. `spatial` points at the
    /// freshly allocated scene-pool region.
    Ok {
        spatial: SpatialData,
        voxel_count: u32,
    },
    /// Either voxelize_to_artifact returned None (pool exhaustion at
    /// unreasonable depths) or integrate failed (contiguous range
    /// allocation failed). Engine logs + clears in-flight flag.
    Failed,
}

pub struct BakeWorker {
    /// Send bake jobs here.
    pub tx_request: Sender<BakeRequest>,
    /// Engine drains this each tick.
    pub rx_result: Receiver<BakeResult>,
    _handle: JoinHandle<()>,
}

impl BakeWorker {
    pub fn spawn(
        device: wgpu::Device,
        queue: wgpu::Queue,
        scene_mgr: Arc<Mutex<RkpSceneManager>>,
    ) -> Self {
        let (tx_request, rx_request) = crossbeam::channel::unbounded::<BakeRequest>();
        let (tx_result, rx_result) = crossbeam::channel::unbounded::<BakeResult>();

        let handle = std::thread::Builder::new()
            .name("rkp-bake-worker".to_string())
            .spawn(move || {
                // GpuEvaluator is created on the worker thread so its
                // buffers + pipelines are owned there. Device/Queue
                // handles are cheap clones (internally Arc'd).
                let mut evaluator = GpuEvaluator::new(&device);

                while let Ok(req) = rx_request.recv() {
                    let t_start = std::time::Instant::now();
                    let instructions = req.instructions.clone();
                    let sdf_fn =
                        |positions: &[glam::Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
                            evaluator
                                .evaluate(&device, &queue, positions, &instructions)
                                .into_iter()
                                .map(|r| r.into_tuple())
                                .collect()
                        };

                    let artifact = rkp_core::voxelize_to_artifact(
                        sdf_fn,
                        &req.aabb,
                        req.voxel_size,
                    );
                    let t_after_voxelize = t_start.elapsed();
                    let outcome = match artifact {
                        Some(a) => {
                            let voxel_count = a.voxel_count;
                            // Integrate under the scene_mgr lock. Lock is
                            // held briefly (tens of ms on big bakes) —
                            // render frames that need to upload geometry
                            // will wait, but that's at most a one-frame
                            // stall per bake and we explicitly want the
                            // new voxels committed before anything reads.
                            let t_lock_start = std::time::Instant::now();
                            let mut sm = scene_mgr.lock().unwrap();
                            let t_lock_acquired = t_lock_start.elapsed();

                            // Free the entity's previous allocation so the
                            // pools don't leak. Happens under the same
                            // lock so the brick IDs the new integrate
                            // picks up from the free list are exactly the
                            // ones we just released.
                            if let Some(prev) = &req.prev_spatial {
                                let handle = rkp_core::OctreeHandle {
                                    root_offset: prev.root_offset,
                                    len: prev.len,
                                    depth: prev.depth,
                                    base_voxel_size: prev.base_voxel_size,
                                };
                                sm.deallocate_geometry(
                                    &handle,
                                    prev.voxel_slot_start,
                                    prev.voxel_slot_count,
                                    &prev.brick_ids,
                                );
                            }
                            let result = sm.integrate_artifact(
                                a, &req.aabb, req.voxel_size, req.scene_id,
                            );
                            drop(sm);
                            let t_total = t_start.elapsed();
                            match result {
                                Some(result) => {
                                    let spatial = spatial_from_voxelize_result(&result);
                                    eprintln!(
                                        "[bake_worker] gen={} scene_id={} voxels={} \
                                         voxelize={:.1}ms lock_wait={:.2}ms \
                                         integrate+dealloc={:.1}ms total={:.1}ms",
                                        req.generation, req.scene_id, voxel_count,
                                        t_after_voxelize.as_secs_f32() * 1000.0,
                                        t_lock_acquired.as_secs_f32() * 1000.0,
                                        (t_total - t_after_voxelize).as_secs_f32() * 1000.0,
                                        t_total.as_secs_f32() * 1000.0,
                                    );
                                    BakeOutcome::Ok {
                                        spatial,
                                        voxel_count,
                                    }
                                }
                                None => BakeOutcome::Failed,
                            }
                        }
                        None => {
                            eprintln!(
                                "[bake_worker] gen={} scene_id={} voxelize FAILED wall={:.1}ms",
                                req.generation,
                                req.scene_id,
                                t_after_voxelize.as_secs_f32() * 1000.0,
                            );
                            BakeOutcome::Failed
                        }
                    };

                    let result = BakeResult {
                        entity: req.entity,
                        generation: req.generation,
                        scene_id: req.scene_id,
                        aabb: req.aabb,
                        voxel_size: req.voxel_size,
                        root_scale: req.root_scale,
                        outcome,
                    };
                    // Send-failed means the engine has shut down — exit.
                    if tx_result.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn rkp-bake-worker");

        BakeWorker {
            tx_request,
            rx_result,
            _handle: handle,
        }
    }
}

/// Build a `SpatialData` from a `VoxelizeResult`. The worker calls
/// this directly after integrate so the engine doesn't need to know
/// the `SpatialHandle` layout.
fn spatial_from_voxelize_result(r: &rkp_render::rkp_scene_manager::VoxelizeResult) -> SpatialData {
    if let rkp_core::scene_node::SpatialHandle::Octree {
        root_offset, len, depth, base_voxel_size,
    } = r.spatial
    {
        SpatialData {
            root_offset,
            len,
            depth,
            base_voxel_size,
            aabb: r.aabb,
            voxel_size: r.voxel_size,
            grid_origin: r.grid_origin,
            voxel_slot_start: r.leaf_attr_slot_start,
            voxel_slot_count: r.leaf_attr_slot_count,
            brick_ids: r.brick_ids.clone(),
        }
    } else {
        SpatialData {
            root_offset: 0, len: 0, depth: 0, base_voxel_size: r.voxel_size,
            aabb: r.aabb,
            voxel_size: r.voxel_size,
            grid_origin: r.grid_origin,
            voxel_slot_start: r.leaf_attr_slot_start,
            voxel_slot_count: r.leaf_attr_slot_count,
            brick_ids: r.brick_ids.clone(),
        }
    }
}
