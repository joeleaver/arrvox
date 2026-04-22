//! Generator runtime context — the handle a generator function uses to
//! talk to the engine while it's running on the worker thread.
//!
//! The context's job is narrower than it used to be: it no longer
//! voxelizes or integrates itself. Instead, `emit_child` / `emit_child_artifact`
//! synthesize `BakeRequest`s and send them onto the same channel that
//! ordinary procedural edits use. The bake worker processes them with
//! the exact same pipeline — voxelize (if tree) → integrate →
//! `BakeResult` — and the engine's `drain_bake_results` turns each
//! result into a new child entity by detecting the attached
//! `GeneratorChildSpec`.
//!
//! The upshot: any bug the bake path doesn't have, the generator path
//! can't have either. There's only one voxelize+integrate path to
//! maintain.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam::channel::Sender;

use rkp_core::{Aabb, BakeArtifact, WorldPosition};
use rkp_render::proc_sample::GpuEvaluator;
use rkp_render::rkp_scene_manager::RkpSceneManager;

use crate::bake_worker::{BakeInput, BakeRequest, GeneratorChildSpec};
use crate::components::Transform;

use super::error::GeneratorError;

/// Cooperative cancellation handle.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
}

/// Shared progress state (u32 scaled 0..=10_000 for lock-free atomics).
#[derive(Clone, Default)]
pub struct ProgressHandle {
    value: Arc<AtomicU32>,
}

impl ProgressHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> f32 {
        self.value.load(Ordering::Relaxed) as f32 / 10_000.0
    }

    fn set(&self, fraction: f32) {
        let clamped = fraction.clamp(0.0, 1.0);
        self.value.store((clamped * 10_000.0) as u32, Ordering::Relaxed);
    }
}

/// The handle passed to every generator invocation.
///
/// `'w` ties the context to worker-local borrows. Generators cannot
/// stash a `GeneratorContext` across calls because those borrows
/// expire when the function returns.
pub struct GeneratorContext<'w> {
    pub transform: Transform,
    pub world_position: WorldPosition,
    pub generation: u64,

    cancel: CancelToken,
    progress: ProgressHandle,

    // ── Worker-local — populated only by `new_worker` ──────────────
    generator_entity: Option<hecs::Entity>,
    /// UUID of the generator entity. Combined with each emit's
    /// `slot_key` to compute deterministic disk paths for
    /// persistent-child bake caches.
    generator_entity_uuid: Option<uuid::Uuid>,
    /// Directory for persistent-child bake caches (typically
    /// `{scene}.bakes/`). `None` = no scene path yet (unsaved
    /// session) → caches not written → save+reload triggers regen.
    child_cache_dir: Option<std::path::PathBuf>,
    param_hash: u64,
    next_scene_id: Option<&'w Arc<AtomicU32>>,
    tx_request: Option<Sender<BakeRequest>>,
    /// Running counter per run used for `name_hint` defaults and for
    /// the synthetic scene_id allocated to each emitted child bake.
    emission_counter: u32,
    // Reserved for future in-worker use (not exposed to generators).
    #[allow(dead_code)]
    device: Option<&'w wgpu::Device>,
    #[allow(dead_code)]
    queue: Option<&'w wgpu::Queue>,
    #[allow(dead_code)]
    evaluator: Option<&'w mut GpuEvaluator>,
    #[allow(dead_code)]
    scene_mgr: Option<&'w Arc<Mutex<RkpSceneManager>>>,
}

impl<'w> GeneratorContext<'w> {
    /// Lightweight constructor for tests. Emit methods error cleanly
    /// without worker context.
    pub fn new(
        transform: Transform,
        world_position: WorldPosition,
        generation: u64,
        cancel: CancelToken,
        progress: ProgressHandle,
    ) -> Self {
        Self {
            transform,
            world_position,
            generation,
            cancel,
            progress,
            generator_entity: None,
            generator_entity_uuid: None,
            child_cache_dir: None,
            param_hash: 0,
            next_scene_id: None,
            tx_request: None,
            emission_counter: 0,
            device: None,
            queue: None,
            evaluator: None,
            scene_mgr: None,
        }
    }

    /// Build a context wired to worker-local state. Called by the
    /// bake+generator worker right before invoking `generate_fn`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_worker(
        transform: Transform,
        world_position: WorldPosition,
        generation: u64,
        cancel: CancelToken,
        progress: ProgressHandle,
        generator_entity: hecs::Entity,
        generator_entity_uuid: Option<uuid::Uuid>,
        child_cache_dir: Option<std::path::PathBuf>,
        _generator_name: String,
        param_hash: u64,
        device: &'w wgpu::Device,
        queue: &'w wgpu::Queue,
        evaluator: &'w mut GpuEvaluator,
        scene_mgr: &'w Arc<Mutex<RkpSceneManager>>,
        next_scene_id: &'w Arc<AtomicU32>,
        tx_request: Sender<BakeRequest>,
    ) -> Self {
        Self {
            transform,
            world_position,
            generation,
            cancel,
            progress,
            generator_entity: Some(generator_entity),
            generator_entity_uuid,
            child_cache_dir,
            param_hash,
            next_scene_id: Some(next_scene_id),
            tx_request: Some(tx_request),
            emission_counter: 0,
            device: Some(device),
            queue: Some(queue),
            evaluator: Some(evaluator),
            scene_mgr: Some(scene_mgr),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn check_cancelled(&self) -> Result<(), GeneratorError> {
        if self.is_cancelled() {
            Err(GeneratorError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn report_progress(&self, fraction: f32) {
        self.progress.set(fraction);
    }

    pub fn progress_handle(&self) -> &ProgressHandle {
        &self.progress
    }

    // ─── Child emission ───────────────────────────────────────────────

    /// Emit a child whose geometry is a procedural tree.
    ///
    /// `slot_key` is the child's stable identity. On regen the engine
    /// matches new emits against existing children by
    /// `(parent, slot_key)`: matched children get reused (Transform +
    /// geometry are swapped in place; every other component — Light,
    /// physics colliders, gameplay scripts the user attached — is
    /// preserved). Children whose key disappears in a later generation
    /// are despawned. Slot keys also key the on-disk bake cache, so
    /// they're what makes save+reload work without re-running the
    /// generator.
    ///
    /// For loops emitting many of the same kind, use a positional key
    /// like `format!("wall_{i}")` — stable as long as the iteration
    /// order is stable.
    ///
    /// `local_transform` is local to the generator entity. The engine
    /// composes with the generator's snapshot transform at spawn time
    /// so the child ends up at the right world position.
    pub fn emit_child(
        &mut self,
        slot_key: impl Into<String>,
        tree: &rkp_procedural::ProceduralObject,
        voxel_size: f32,
        local_transform: Transform,
        name_hint: Option<String>,
    ) -> Result<(), GeneratorError> {
        self.check_cancelled()?;
        if voxel_size <= 0.0 {
            return Err(GeneratorError::InvalidParams(format!(
                "voxel_size must be > 0, got {voxel_size}"
            )));
        }
        let aabb = rkp_procedural::compute_bounds(tree);
        let instructions = rkp_procedural::flatten_tree(tree);
        self.enqueue_child_bake(
            BakeInput::Procedural(instructions),
            aabb,
            voxel_size,
            local_transform,
            name_hint,
            slot_key.into(),
        )
    }

    /// Emit a child whose geometry is already voxelized. Use this for
    /// "other means" generators — CPU-sampled SDFs, mesh voxelization,
    /// anything that produces a `BakeArtifact` outside the procedural
    /// evaluator. The worker skips voxelization and goes straight to
    /// integrate.
    ///
    /// See [`emit_child`](Self::emit_child) for `slot_key` semantics.
    pub fn emit_child_artifact(
        &mut self,
        slot_key: impl Into<String>,
        artifact: BakeArtifact,
        aabb: Aabb,
        voxel_size: f32,
        local_transform: Transform,
        name_hint: Option<String>,
    ) -> Result<(), GeneratorError> {
        self.check_cancelled()?;
        if voxel_size <= 0.0 {
            return Err(GeneratorError::InvalidParams(format!(
                "voxel_size must be > 0, got {voxel_size}"
            )));
        }
        self.enqueue_child_bake(
            BakeInput::Artifact(artifact),
            aabb,
            voxel_size,
            local_transform,
            name_hint,
            slot_key.into(),
        )
    }

    fn enqueue_child_bake(
        &mut self,
        input: BakeInput,
        aabb: Aabb,
        voxel_size: f32,
        local_transform: Transform,
        name_hint: Option<String>,
        slot_key: String,
    ) -> Result<(), GeneratorError> {
        let tx_request = self.tx_request.as_ref().ok_or_else(|| {
            GeneratorError::Failed("emit_child called without worker context".into())
        })?;
        let next_scene_id = self.next_scene_id.ok_or_else(|| {
            GeneratorError::Failed("missing scene_id allocator".into())
        })?;
        let generator_entity = self.generator_entity.ok_or_else(|| {
            GeneratorError::Failed("missing generator entity".into())
        })?;

        let scene_id = next_scene_id.fetch_add(1, Ordering::Relaxed);
        self.emission_counter = self.emission_counter.wrapping_add(1);

        // Deterministic on-disk bake-cache path keyed by
        // (parent_uuid, slot_key). Same emit from the same generator
        // always lands on the same file across sessions. `None` only
        // when there's no scene path yet (unsaved scratch session) or
        // the generator entity has no UUID — both edge cases that
        // suppress caching gracefully.
        let cache_output_path = match (&self.child_cache_dir, self.generator_entity_uuid) {
            (Some(dir), Some(parent_uuid)) => {
                Some(child_cache_path(dir, parent_uuid, &slot_key))
            }
            _ => None,
        };

        let spec = GeneratorChildSpec {
            parent_entity: generator_entity,
            local_transform,
            name_hint,
            generation: self.generation,
            slot_key,
        };

        let req = BakeRequest {
            entity: generator_entity,
            generation: self.generation,
            scene_id,
            input,
            aabb,
            voxel_size,
            // Not used for generator children — integrate doesn't
            // consult root_scale for this path.
            root_scale: glam::Vec3::ONE,
            prev_spatial: None,
            cache_output_path,
            generator_child: Some(spec),
        };
        if tx_request.send(req).is_err() {
            // Worker gone → engine shutting down. Surface as cancel.
            return Err(GeneratorError::Cancelled);
        }
        // Used in param_hash — silence unused-field warning.
        let _ = self.param_hash;
        Ok(())
    }
}

/// Compute the on-disk bake-cache path for a persistent generator
/// child: `{child_cache_dir}/gen_{parent_uuid_short}_{slot_slug}.rkp`.
///
/// Same path is used at write time (worker bakes the artifact here)
/// and at save time (engine references it from the SceneObject's
/// `procedural_cache` field). Slot keys with non-filename-safe
/// characters get sanitized to `_`.
pub fn child_cache_path(
    child_cache_dir: &std::path::Path,
    parent_uuid: uuid::Uuid,
    slot_key: &str,
) -> std::path::PathBuf {
    let slot_slug: String = slot_key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // First 8 hex chars of the parent UUID is enough to disambiguate
    // children across generators within the same scene without making
    // filenames unwieldy.
    let parent_prefix = format!("{}", parent_uuid.simple());
    let parent_prefix = &parent_prefix[..parent_prefix.len().min(8)];
    child_cache_dir.join(format!("gen_{parent_prefix}_{slot_slug}.rkp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> GeneratorContext<'static> {
        GeneratorContext::new(
            Transform::default(),
            WorldPosition::default(),
            0,
            CancelToken::new(),
            ProgressHandle::new(),
        )
    }

    #[test]
    fn cancel_flag_propagates_through_clone() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!clone.is_cancelled());
        token.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn check_cancelled_returns_err_when_set() {
        let cancel = CancelToken::new();
        let c = GeneratorContext::new(
            Transform::default(),
            WorldPosition::default(),
            0,
            cancel.clone(),
            ProgressHandle::new(),
        );
        assert!(c.check_cancelled().is_ok());
        cancel.cancel();
        assert!(matches!(c.check_cancelled(), Err(GeneratorError::Cancelled)));
    }

    #[test]
    fn progress_roundtrips_via_shared_handle() {
        let c = ctx();
        let handle = c.progress_handle().clone();
        c.report_progress(0.37);
        assert!((handle.get() - 0.37).abs() < 1e-3);
    }

    #[test]
    fn progress_clamps() {
        let c = ctx();
        c.report_progress(-1.0);
        assert_eq!(c.progress_handle().get(), 0.0);
        c.report_progress(2.0);
        assert_eq!(c.progress_handle().get(), 1.0);
    }

    #[test]
    fn emit_child_without_worker_context_errors() {
        let mut c = ctx();
        let obj = rkp_procedural::ProceduralObject::new(
            rkp_procedural::NodeKind::Root,
        );
        let result = c.emit_child(&obj, 0.1, Transform::default(), None);
        assert!(matches!(result, Err(GeneratorError::Failed(_))));
    }
}
