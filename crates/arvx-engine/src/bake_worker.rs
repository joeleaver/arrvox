//! Async bake + generator worker.
//!
//! Spawns a single background thread that owns a `GpuEvaluator` and
//! handles two kinds of work from the engine:
//!
//! **`BakeRequest`** — voxelize (optionally) and integrate into the
//! scene. Two input modes:
//!   * `BakeInput::Procedural(tree)` — the worker flattens the tree
//!     into opcodes, runs `voxelize_to_artifact` against its GPU
//!     evaluator, then integrates.
//!   * `BakeInput::Artifact(a)` — the caller already produced voxels
//!     (e.g. a CPU-sampled SDF or a mesh voxelization). The worker
//!     skips straight to integrate.
//!
//! Regular procedural edits send `Procedural` with `generator_child =
//! None`; the result updates the owning entity's `Renderable`.
//! Generator-emitted children send either variant with a
//! `GeneratorChildSpec` set, and the engine's `drain_bake_results`
//! spawns a new child entity for each result.
//!
//! **`GeneratorRequest`** — run a user-authored generator function on
//! the worker thread. The generator body calls `ctx.emit_child(...)`
//! which enqueues `BakeRequest`s back onto this same worker's queue.
//! Because the worker is single-threaded, those bakes run after the
//! generator itself returns — no deadlock possible between the
//! generator's voxelization and its own emissions.
//!
//! Both request kinds share the same `GpuEvaluator`. The worker
//! multiplexes the two input channels with `crossbeam::select!`.

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam::channel::{Receiver, Sender};

use arvx_render::arvx_scene_manager::ArvxSceneManager;
use arvx_render::proc_sample::GpuEvaluator;
use arvx_render::proc_surface_nets::{GpuSurfaceNets, SurfaceMesh};
use arvx_core::mesh_cluster::MeshletCluster;

use crate::components::{SpatialData, Transform};
use crate::generator::{GenerateFn, GeneratorError};

/// What the worker will voxelize into a `BakeArtifact`, or skip past if
/// already done.
pub enum BakeInput {
    /// Flattened procedural opcode stream — the worker runs surface-
    /// nets-from-SDF and ships a triangle proxy mesh back. This is the
    /// live editing path: cheap, fast, but the result isn't paintable
    /// or sculptable (no octree, no leaf attrs).
    Procedural(Vec<arvx_procedural::flatten::ProcInstruction>),
    /// Flattened procedural opcode stream — the worker GPU-evaluates
    /// the tree at every octree voxel cell and emits a true
    /// `BakeArtifact`. Used by Copy/Convert when the user wants a
    /// paintable / sculptable static voxel object from a procedural.
    /// Heavier than `Procedural` (full voxelization + integrate
    /// instead of surface-nets), so reserved for user-triggered
    /// one-shots.
    ProceduralVoxelize(Vec<arvx_procedural::flatten::ProcInstruction>),
    /// Pre-voxelized by the caller. Worker skips straight to integrate.
    /// Used for "other means" generators (CPU-sampled SDFs, mesh
    /// voxelization, etc.) that don't want to route through the
    /// procedural evaluator.
    Artifact(arvx_core::BakeArtifact),
}

/// If a BakeRequest originated from a generator's `emit_child` call,
/// the request carries this spec. On successful integrate the engine
/// spawns a new child entity rather than updating the existing
/// `request.entity`.
pub struct GeneratorChildSpec {
    /// The generator entity that emitted this child. Child's
    /// `GeneratorOwned.parent` points here.
    pub parent_entity: hecs::Entity,
    /// Child-local transform. The engine composes
    /// `parent.transform × local_transform` at spawn time to get the
    /// absolute world transform stored on the child.
    pub local_transform: Transform,
    /// Optional name hint for `EditorMetadata.name`. Defaults to
    /// `"<parent>.child"` if None.
    pub name_hint: Option<String>,
    /// Generator's current generation counter. Stale results (from a
    /// regen whose generation has been bumped) can be detected and
    /// dropped by checking this against the generator's tracker.
    pub generation: u64,
    /// Stable identity assigned by the generator at emit time. The
    /// engine matches new emits against existing children by
    /// (parent, slot_key) and reuses the existing entity in place
    /// (preserving any user-attached components like lights or
    /// scripts). Children whose key disappears in a later generation
    /// are despawned. Also keys the on-disk bake cache.
    pub slot_key: String,
}

/// One unit of bake work.
pub struct BakeRequest {
    /// For a standard procedural bake: the entity whose `Renderable` to
    /// update. For a generator-emitted child: the generator entity (the
    /// actual child is a new entity spawned in `drain_bake_results`).
    pub entity: hecs::Entity,
    /// For regular bakes: the procedural's `bake_generation` counter.
    /// For generator-child bakes: unused — `generator_child.generation`
    /// is the authoritative counter.
    pub generation: u64,
    pub input: BakeInput,
    pub aabb: arvx_core::Aabb,
    pub voxel_size: f32,
    /// Root.transform.scale at the moment this request was captured.
    /// Worker echoes it back so integrate can set
    /// `last_evaluated_root_scale` to what the voxels actually
    /// represent. Only meaningful for procedural-entity bakes; zero
    /// for generator children.
    pub root_scale: glam::Vec3,
    /// Previous geometry allocation to free. Always `None` for
    /// generator children (each is a new allocation).
    pub prev_spatial: Option<SpatialData>,
    /// Optional `.arvx` sidecar path to write the artifact to before
    /// integrating. `None` skips the cache write.
    pub cache_output_path: Option<std::path::PathBuf>,
    /// Set → this is a generator-emitted child bake. Drives main-thread
    /// post-processing (spawn new entity vs. update existing).
    pub generator_child: Option<GeneratorChildSpec>,
}

/// Worker → engine handoff. Integrate has already run on the worker
/// by the time the engine sees this.
pub struct BakeResult {
    pub entity: hecs::Entity,
    pub generation: u64,
    pub aabb: arvx_core::Aabb,
    pub voxel_size: f32,
    pub root_scale: glam::Vec3,
    pub outcome: BakeOutcome,
    /// Echoed back from the request so main-thread processing knows
    /// whether to spawn a child or update an existing entity.
    pub generator_child: Option<GeneratorChildSpec>,
}

pub enum BakeOutcome {
    /// Voxelize + integrate both succeeded. `spatial` points at the
    /// freshly allocated scene-pool region.
    Ok {
        spatial: SpatialData,
        voxel_count: u32,
    },
    /// Surface-nets extraction succeeded; engine should allocate a
    /// procedural asset handle, upload `surface_mesh` + `cluster`
    /// to the renderer, and set the entity's `Renderable.spatial`
    /// to `RenderGeometry::ProxyMesh`. `aabb` is the world-space
    /// extent the mesh was sampled over (caller passes it in
    /// `BakeRequest`; echoed back so the integrate path doesn't
    /// have to re-derive it).
    ProxyMeshOk {
        surface_mesh: SurfaceMesh,
        cluster: MeshletCluster,
    },
    /// Either voxelize_to_artifact returned None (pool exhaustion at
    /// unreasonable depths) or integrate failed (contiguous range
    /// allocation failed). Engine logs + clears in-flight flag.
    Failed,
}

/// A generator run request. Params are type-erased; the generator's
/// erased wrapper (emitted by `#[arvx_generator]`) downcasts inside the
/// worker.
pub struct GeneratorRequest {
    pub entity: hecs::Entity,
    /// Bumped by the system each submission. Generator children inherit
    /// this generation into their `GeneratorChildSpec`.
    pub generation: u64,
    pub generator_name: String,
    /// Hash of params at submission time; used to drop stale results.
    pub param_hash: u64,
    pub params: Box<dyn Any + Send>,
    pub cancel: crate::generator::CancelToken,
    pub progress: crate::generator::ProgressHandle,
    pub transform: crate::components::Transform,
    pub world_position: arvx_core::WorldPosition,
    pub generate_fn: GenerateFn,
    /// UUID of the generator entity. Used (in concert with each
    /// emit's slot_key) to compute deterministic disk paths for
    /// persistent-child bake caches. `None` for entities that
    /// haven't been UUID-stamped yet (shouldn't happen — every
    /// generator entity gets one at spawn).
    pub generator_entity_uuid: Option<uuid::Uuid>,
    /// Directory under which persistent-child bake caches are
    /// written by the worker (typically `{scene}.bakes/`). `None`
    /// when the scene has no on-disk path yet (unsaved scratch
    /// session) — children will still bake into pool memory but
    /// won't have a persistent cache, so a save+reload will trigger
    /// a regen.
    pub child_cache_dir: Option<std::path::PathBuf>,
}

/// Generator lifecycle events. Child emissions do NOT flow through
/// here — they go out via `BakeResult` instead (unified path).
pub enum GeneratorWorkerEvent {
    Completed {
        generator_entity: hecs::Entity,
        generation: u64,
        generator_name: String,
        param_hash: u64,
    },
    Failed {
        generator_entity: hecs::Entity,
        generation: u64,
        generator_name: String,
        param_hash: u64,
        error: GeneratorError,
    },
}

pub struct BakeWorker {
    /// Send bake jobs here. Both procedural edits and generator-child
    /// emissions flow through this channel.
    pub tx_request: Sender<BakeRequest>,
    /// Engine drains bake results each tick.
    pub rx_result: Receiver<BakeResult>,
    /// Send generator jobs here.
    pub tx_generator: Sender<GeneratorRequest>,
    /// Engine drains generator lifecycle events here.
    pub rx_generator: Receiver<GeneratorWorkerEvent>,
    _handle: JoinHandle<()>,
}

impl BakeWorker {
    pub fn spawn(
        device: wgpu::Device,
        queue: wgpu::Queue,
        scene_mgr: Arc<Mutex<ArvxSceneManager>>,
    ) -> Self {
        let (tx_request, rx_request) = crossbeam::channel::unbounded::<BakeRequest>();
        let (tx_result, rx_result) = crossbeam::channel::unbounded::<BakeResult>();
        let (tx_generator, rx_gen_request) =
            crossbeam::channel::unbounded::<GeneratorRequest>();
        let (tx_gen_event, rx_generator) =
            crossbeam::channel::unbounded::<GeneratorWorkerEvent>();

        // Cloned into the generator context so it can enqueue child
        // bakes through the same channel the engine uses for ordinary
        // procedural edits.
        let tx_request_for_ctx = tx_request.clone();

        let handle = std::thread::Builder::new()
            .name("arvx-bake-worker".to_string())
            .spawn(move || {
                let mut evaluator = GpuEvaluator::new(&device);
                // Lazy: most procedurals voxelize, so don't pay the
                // pipeline-creation cost until the first ProxyMesh
                // bake actually arrives.
                let mut surface_nets: Option<GpuSurfaceNets> = None;

                loop {
                    let req_opt: Option<WorkerJob> = crossbeam::select! {
                        recv(rx_request) -> msg => msg.ok().map(WorkerJob::Bake),
                        recv(rx_gen_request) -> msg => msg.ok().map(WorkerJob::Generator),
                    };
                    let req = match req_opt {
                        Some(j) => j,
                        None => break,
                    };

                    match req {
                        WorkerJob::Generator(g) => {
                            run_generator(
                                g,
                                &device,
                                &queue,
                                &mut evaluator,
                                &scene_mgr,
                                &tx_request_for_ctx,
                                &tx_gen_event,
                            );
                        }
                        WorkerJob::Bake(req) => {
                            let result = run_bake(
                                req,
                                &device,
                                &queue,
                                &mut evaluator,
                                &mut surface_nets,
                                &scene_mgr,
                            );
                            if tx_result.send(result).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
            .expect("spawn arvx-bake-worker");

        BakeWorker {
            tx_request,
            rx_result,
            tx_generator,
            rx_generator,
            _handle: handle,
        }
    }
}

/// Internal multiplexer discriminant.
enum WorkerJob {
    Bake(BakeRequest),
    Generator(GeneratorRequest),
}

/// Diagnostic probe used only on the ProceduralVoxelize collapse path.
/// Evaluates the flattened SDF on a dense grid spanning `aabb` and
/// returns `(neg, pos, zero, dmin, dmax)` — the sign distribution and
/// distance range. If both `neg` and `pos` are non-zero a surface
/// genuinely exists in the AABB, so a collapse-to-empty means the
/// octree classify *missed* it (a GPU-result problem) rather than the
/// SDF being empty (an upstream input problem).
fn probe_sdf_grid(
    evaluator: &mut GpuEvaluator,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    instructions: &[arvx_procedural::flatten::ProcInstruction],
    aabb: &arvx_core::Aabb,
) -> (u32, u32, u32, f32, f32) {
    const N: u32 = 8;
    let span = aabb.max - aabb.min;
    let denom = (N - 1) as f32;
    let mut pts: Vec<glam::Vec3> = Vec::with_capacity((N * N * N) as usize);
    for z in 0..N {
        for y in 0..N {
            for x in 0..N {
                let t = glam::Vec3::new(
                    x as f32 / denom,
                    y as f32 / denom,
                    z as f32 / denom,
                );
                pts.push(aabb.min + span * t);
            }
        }
    }
    let results = evaluator.evaluate(device, queue, &pts, instructions);
    let (mut neg, mut pos, mut zero) = (0u32, 0u32, 0u32);
    let (mut dmin, mut dmax) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in results {
        let d = s.into_tuple().0;
        if d < 0.0 {
            neg += 1;
        } else if d > 0.0 {
            pos += 1;
        } else {
            zero += 1;
        }
        dmin = dmin.min(d);
        dmax = dmax.max(d);
    }
    (neg, pos, zero, dmin, dmax)
}

/// Run one bake job to completion. Accepts both procedural-tree input
/// and pre-voxelized artifact input. Integrates under the `scene_mgr`
/// lock and returns a `BakeResult` describing the outcome.
fn run_bake(
    req: BakeRequest,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    evaluator: &mut GpuEvaluator,
    surface_nets: &mut Option<GpuSurfaceNets>,
    scene_mgr: &Arc<Mutex<ArvxSceneManager>>,
) -> BakeResult {
    let t_start = std::time::Instant::now();

    // Procedural inputs split by intent: live-edit `Procedural`
    // produces a proxy mesh via surface-nets (no octree). One-shot
    // `ProceduralVoxelize` runs the SDF through the GPU evaluator and
    // emits a full `BakeArtifact` — same shape Artifact inputs take —
    // so it falls through to the integrate/write path below.
    if let BakeInput::Procedural(instructions) = req.input {
        let sn = surface_nets.get_or_insert_with(|| GpuSurfaceNets::new(device));
        // Cell size matches the procedural's chosen voxel size, so
        // proxy mesh detail tracks the existing voxel-size tier the
        // user picks. Caps follow the spike's `O(N²) × const` rule.
        let extent = (req.aabb.max - req.aabb.min).max_element();
        let grid_n = ((extent / req.voxel_size).ceil() as u32).max(8);
        let n2 = (grid_n as u64).pow(2);
        let vertex_cap = (n2 * 16).min(u32::MAX as u64) as u32;
        let index_cap = (n2 * 96).min(u32::MAX as u64) as u32;
        let (mesh, _stats) = sn.extract(
            device,
            queue,
            &instructions,
            req.aabb.min,
            req.aabb.max,
            grid_n,
            vertex_cap,
            index_cap,
            /* read_geometry = */ true,
        );
        let outcome = match mesh {
            Some(m) => {
                // Persist the proxy mesh to a sidecar cache if the
                // caller asked for one (generator-child bakes do; live
                // procedural edits don't). `cache_output_path` is
                // already keyed `gen_<parent>_<slot>.arvxproxy` by the
                // generator context — see `context::child_cache_path`.
                // Cache failures are non-fatal: log + continue so the
                // in-memory bake still installs on the entity.
                if let Some(ref out_path) = req.cache_output_path {
                    let cache = arvx_core::asset_file::ProxyCache {
                        aabb_min: m.aabb_min.to_array(),
                        aabb_max: m.aabb_max.to_array(),
                        vertices: m.vertices.clone(),
                        indices: m.indices.clone(),
                    };
                    let t_write_start = std::time::Instant::now();
                    match arvx_core::asset_file::write_arvxproxy(out_path, &cache) {
                        Ok(()) => {
                            eprintln!(
                                "[bake_worker] wrote proxy cache {} in {:.1}ms",
                                out_path.display(),
                                t_write_start.elapsed().as_secs_f32() * 1000.0,
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[bake_worker] proxy cache write failed ({}): {e}",
                                out_path.display(),
                            );
                        }
                    }
                }
                let cluster = m.single_cluster();
                BakeOutcome::ProxyMeshOk {
                    surface_mesh: m,
                    cluster,
                }
            }
            None => BakeOutcome::Failed,
        };
        return BakeResult {
            entity: req.entity,
            generation: req.generation,
            aabb: req.aabb,
            voxel_size: req.voxel_size,
            root_scale: req.root_scale,
            outcome,
            generator_child: req.generator_child,
        };
    }

    // Artifact-input voxelize/integrate path — used by imported
    // `.arvx` loads (which carry pre-voxelized data the file format
    // ships with) AND by `ProceduralVoxelize` (one-shot user-
    // triggered escape hatch that turns a procedural SDF into a
    // paintable/sculptable octree).
    let artifact = match req.input {
        BakeInput::Procedural(_) => unreachable!("handled above"),
        BakeInput::ProceduralVoxelize(instructions) => {
            // Pre-dispatch guard. A degenerate AABB or an empty opcode
            // stream can only ever voxelize to nothing — but a doomed
            // bake still pays a full GPU classify dispatch + blocking
            // `device.poll` that contends with frame presentation on
            // the (currently shared) queue. Fail fast with the reason
            // instead of submitting work the GPU can't usefully do.
            let extent = req.aabb.max - req.aabb.min;
            let degenerate =
                instructions.is_empty() || !extent.is_finite() || extent.min_element() <= 0.0;
            if degenerate {
                eprintln!(
                    "[bake_worker] ProceduralVoxelize SKIPPED (degenerate input) \
                     entity={:?} gen={} instructions={} aabb_min={:?} aabb_max={:?} \
                     extent={:?} voxel_size={} root_scale={:?}",
                    req.entity, req.generation, instructions.len(),
                    req.aabb.min.to_array(), req.aabb.max.to_array(),
                    extent.to_array(), req.voxel_size, req.root_scale.to_array(),
                );
                None
            } else {
                // GPU-evaluate the SDF at every cell the octree
                // classifier asks about. `voxelize_to_artifact` is
                // pool-private (fresh `LeafAttrPool` / `BrickPool`), so
                // we don't need the scene_mgr lock yet — that's only
                // taken below for the integrate step. Returns `None` on
                // an empty tree or zero-voxel result, which propagates
                // to `BakeOutcome::Failed`.
                let artifact = {
                    let mut closure =
                        |positions: &[glam::Vec3]| -> Vec<(f32, u16, u16, u8, u32, Option<glam::Vec3>)> {
                            evaluator
                                .evaluate(device, queue, positions, &instructions)
                                .into_iter()
                                .map(|s| s.into_tuple())
                                .collect()
                        };
                    arvx_core::voxelize_to_artifact(&mut closure, &req.aabb, req.voxel_size, 0)
                };
                if artifact.is_none() {
                    // Non-degenerate input, yet the octree classifier
                    // saw a uniform-sign SDF and emitted nothing. This
                    // is the nondeterministic "voxels=0 ~1/3 of loads"
                    // collapse. Re-probe the SDF on a dense grid across
                    // the AABB to tell the two root causes apart:
                    //   * sign varies here → a surface genuinely exists
                    //     in this AABB, so the classify *missed* it →
                    //     GPU-result corruption (the shared-queue race
                    //     is the prime suspect), NOT bad input.
                    //   * sign uniform here too → the flattened SDF is
                    //     genuinely empty over its own bounds → an
                    //     upstream flatten / bounds / transform bug.
                    let (neg, pos, zero, dmin, dmax) =
                        probe_sdf_grid(evaluator, device, queue, &instructions, &req.aabb);
                    let verdict = if neg > 0 && pos > 0 {
                        "SURFACE-PRESENT (classify missed it → GPU-result race?)"
                    } else {
                        "GENUINELY-EMPTY (upstream flatten/bounds/transform bug?)"
                    };
                    eprintln!(
                        "[bake_worker] ProceduralVoxelize COLLAPSED to empty — {verdict} \
                         entity={:?} gen={} instructions={} aabb_min={:?} aabb_max={:?} \
                         extent={:?} voxel_size={} root_scale={:?} \
                         probe8x8x8[neg={} pos={} zero={} dmin={:.4} dmax={:.4}]",
                        req.entity, req.generation, instructions.len(),
                        req.aabb.min.to_array(), req.aabb.max.to_array(),
                        extent.to_array(), req.voxel_size, req.root_scale.to_array(),
                        neg, pos, zero, dmin, dmax,
                    );
                }
                artifact
            }
        }
        BakeInput::Artifact(a) => Some(a),
    };
    let t_after_voxelize = t_start.elapsed();

    let outcome = match artifact {
        Some(a) => {
            let voxel_count = a.voxel_count;

            // Cache to sidecar outside the scene_mgr lock.
            if let Some(ref out_path) = req.cache_output_path {
                let t_write_start = std::time::Instant::now();
                match arvx_core::asset_file::write_artifact_rkp(
                    out_path,
                    &a,
                    req.aabb.min.to_array(),
                    req.aabb.max.to_array(),
                    req.voxel_size,
                ) {
                    Ok(()) => {
                        eprintln!(
                            "[bake_worker] wrote cache {} in {:.1}ms",
                            out_path.display(),
                            t_write_start.elapsed().as_secs_f32() * 1000.0,
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[bake_worker] cache write failed ({}): {e}",
                            out_path.display(),
                        );
                    }
                }
            }

            // Integrate under the scene_mgr lock. `prev_spatial` is
            // always None for generator children — each is a fresh
            // allocation.
            let t_lock_start = std::time::Instant::now();
            let mut sm = scene_mgr.lock().unwrap();
            let t_lock_acquired = t_lock_start.elapsed();

            if let Some(prev) = &req.prev_spatial {
                let handle = arvx_core::OctreeHandle {
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
                a, &req.aabb, req.voxel_size,
            );
            drop(sm);
            let t_total = t_start.elapsed();
            match result {
                Some(result) => {
                    let spatial = spatial_from_voxelize_result(&result);
                    eprintln!(
                        "[bake_worker] gen={} entity={:?} voxels={} \
                         voxelize={:.1}ms lock_wait={:.2}ms \
                         integrate+dealloc={:.1}ms total={:.1}ms",
                        req.generation, req.entity, voxel_count,
                        t_after_voxelize.as_secs_f32() * 1000.0,
                        t_lock_acquired.as_secs_f32() * 1000.0,
                        (t_total - t_after_voxelize).as_secs_f32() * 1000.0,
                        t_total.as_secs_f32() * 1000.0,
                    );
                    BakeOutcome::Ok { spatial, voxel_count }
                }
                None => BakeOutcome::Failed,
            }
        }
        None => {
            eprintln!(
                "[bake_worker] gen={} entity={:?} voxelize FAILED wall={:.1}ms",
                req.generation,
                req.entity,
                t_after_voxelize.as_secs_f32() * 1000.0,
            );
            BakeOutcome::Failed
        }
    };

    BakeResult {
        entity: req.entity,
        generation: req.generation,
        aabb: req.aabb,
        voxel_size: req.voxel_size,
        root_scale: req.root_scale,
        outcome,
        generator_child: req.generator_child,
    }
}

/// Run one generator function to completion. The generator body calls
/// `ctx.emit_child(...)` which enqueues further `BakeRequest`s onto
/// this same worker's queue — those get processed after the generator
/// returns, since the worker is single-threaded.
fn run_generator(
    g: GeneratorRequest,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    evaluator: &mut GpuEvaluator,
    scene_mgr: &Arc<Mutex<ArvxSceneManager>>,
    tx_request: &Sender<BakeRequest>,
    tx_gen_event: &Sender<GeneratorWorkerEvent>,
) {
    use crate::generator::GeneratorContext;

    let mut ctx = GeneratorContext::new_worker(
        g.transform.clone(),
        g.world_position,
        g.generation,
        g.cancel.clone(),
        g.progress.clone(),
        g.entity,
        g.generator_entity_uuid,
        g.child_cache_dir.clone(),
        g.generator_name.clone(),
        g.param_hash,
        device,
        queue,
        evaluator,
        scene_mgr,
        tx_request.clone(),
    );

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (g.generate_fn)(&*g.params, &mut ctx)
    }))
    .unwrap_or_else(|panic_payload| {
        let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "generator panicked".to_string()
        };
        Err(GeneratorError::Failed(format!("panic: {msg}")))
    });

    let event = match outcome {
        Ok(()) => GeneratorWorkerEvent::Completed {
            generator_entity: g.entity,
            generation: g.generation,
            generator_name: g.generator_name,
            param_hash: g.param_hash,
        },
        Err(error) => GeneratorWorkerEvent::Failed {
            generator_entity: g.entity,
            generation: g.generation,
            generator_name: g.generator_name,
            param_hash: g.param_hash,
            error,
        },
    };
    let _ = tx_gen_event.send(event);
}

fn spatial_from_voxelize_result(r: &arvx_render::arvx_scene_manager::VoxelizeResult) -> SpatialData {
    if let arvx_core::scene_node::SpatialHandle::Octree {
        root_offset, len, depth, base_voxel_size,
    } = r.spatial
    {
        SpatialData {
            root_offset, len, depth, base_voxel_size,
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
