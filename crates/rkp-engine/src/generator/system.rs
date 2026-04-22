//! Generator tick driver — the main-thread half of the generator system.
//!
//! Responsibilities each frame:
//! 1. Drain completed results from the bake-worker generator channel.
//!    Update `GeneratorState` on the owning entity.
//! 2. Scan entities with a `GeneratorState`. For each, hash its current
//!    params and compare against the hash we last submitted. If they
//!    differ (or status is `Pending`), mark stale and submit a new job
//!    to the worker, cancelling any in-flight run for that entity first.
//! 3. Keep a per-entity `progress_handle` so the UI can read live
//!    progress without touching the ECS every frame.
//!
//! No output integration yet — that's M3.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crossbeam::channel::{Receiver, Sender};

use crate::bake_worker::{GeneratorRequest, GeneratorWorkerEvent};
use crate::component_registry::ComponentRegistry;
use crate::components::Transform;

use super::context::{CancelToken, ProgressHandle};
use super::error::GeneratorStatus;
use super::registry::GeneratorRegistry;
use super::state::GeneratorState;

/// Per-entity generator tracking state. Lives in the system, not the ECS,
/// because it holds `Sync`-unfriendly things like cancel tokens.
struct Tracked {
    /// Last hash we submitted to the worker. Used both to detect param
    /// edits and to drop stale results that arrive after a newer
    /// submission.
    last_submitted_hash: u64,
    /// Bumped on every submission. Matches `GeneratorRequest.generation`.
    submitted_generation: u64,
    /// Cancel token for the currently in-flight run, if any.
    active_cancel: Option<CancelToken>,
    /// Progress handle for the currently in-flight run, if any. The UI
    /// reads this through `GeneratorSystem::progress`.
    active_progress: Option<ProgressHandle>,
}

impl Default for Tracked {
    fn default() -> Self {
        Self {
            last_submitted_hash: 0,
            submitted_generation: 0,
            active_cancel: None,
            active_progress: None,
        }
    }
}

/// Generator lifecycle events surfaced to the engine each tick. Child
/// emissions flow through the bake pipeline (`BakeResult`) rather than
/// through here — see `drain_bake_results` for the child-spawn path.
#[derive(Debug)]
pub enum GeneratorEvent {
    /// Fired right before `Submitted` so the engine can blow away the
    /// previous generation's anonymous children + reset slot-key
    /// tracking. The engine performs the actual despawns (it owns the
    /// scene_mgr deallocation path) — the system can't do that itself
    /// without coupling to GPU pool management.
    WillResubmit { entity: hecs::Entity, name: String },
    Submitted { entity: hecs::Entity, name: String },
    Completed { entity: hecs::Entity, name: String },
    Failed { entity: hecs::Entity, name: String, error: String },
    Cancelled { entity: hecs::Entity, name: String },
}

pub struct GeneratorSystem {
    registry: GeneratorRegistry,
    tx: Sender<GeneratorRequest>,
    rx: Receiver<GeneratorWorkerEvent>,
    tracked: HashMap<hecs::Entity, Tracked>,
}

impl GeneratorSystem {
    pub fn new(
        registry: GeneratorRegistry,
        tx: Sender<GeneratorRequest>,
        rx: Receiver<GeneratorWorkerEvent>,
    ) -> Self {
        Self {
            registry,
            tx,
            rx,
            tracked: HashMap::new(),
        }
    }

    /// Access the registry (for UI spawn menus, MCP listing, etc.).
    pub fn registry(&self) -> &GeneratorRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut GeneratorRegistry {
        &mut self.registry
    }

    /// Read live progress for an in-flight run. `None` if idle.
    pub fn progress(&self, entity: hecs::Entity) -> Option<f32> {
        self.tracked
            .get(&entity)
            .and_then(|t| t.active_progress.as_ref())
            .map(|p| p.get())
    }

    /// Force the entity's next tick to re-submit (e.g. user hit "Regenerate").
    pub fn force_regenerate(
        &mut self,
        entity: hecs::Entity,
        world: &mut hecs::World,
    ) {
        self.cancel(entity);
        if let Ok(mut state) = world.get::<&mut GeneratorState>(entity) {
            state.status = GeneratorStatus::Stale;
        }
        // Clear the hash so the next tick treats the current params as fresh.
        if let Some(t) = self.tracked.get_mut(&entity) {
            t.last_submitted_hash = 0;
        }
    }

    /// Cancel any in-flight run for this entity. Safe to call if none.
    pub fn cancel(&mut self, entity: hecs::Entity) {
        if let Some(t) = self.tracked.get_mut(&entity) {
            if let Some(token) = t.active_cancel.take() {
                token.cancel();
            }
            t.active_progress = None;
        }
    }

    /// Drop all tracking for an entity. Call on despawn.
    pub fn forget(&mut self, entity: hecs::Entity) {
        self.cancel(entity);
        self.tracked.remove(&entity);
    }

    /// Drop all gameplay-dylib-provided generators. Call before hot-reload.
    pub fn clear_gameplay_generators(&mut self) {
        self.registry.clear_gameplay();
    }

    /// Register generator entries discovered in the gameplay dylib.
    pub fn register_gameplay(&mut self, entries: &[&'static super::registry::GeneratorEntry]) {
        for e in entries {
            self.registry.register_gameplay(*e);
        }
    }

    /// Main tick — poll results, then scan for stale entities and submit.
    ///
    /// `entity_uuids` lets emitted persistent children compute
    /// deterministic disk-cache paths keyed by the generator's UUID.
    /// `child_cache_dir` is the directory under which those caches
    /// land (typically `{scene}.bakes/`); `None` skips caching, which
    /// means a save+reload of the scene will trigger a regen instead
    /// of restoring from disk.
    pub fn tick(
        &mut self,
        world: &mut hecs::World,
        components: &ComponentRegistry,
        entity_uuids: &std::collections::HashMap<hecs::Entity, uuid::Uuid>,
        child_cache_dir: Option<&std::path::Path>,
    ) -> Vec<GeneratorEvent> {
        let mut events = Vec::new();
        self.poll_results(world, &mut events);
        self.scan_and_submit(world, components, entity_uuids, child_cache_dir, &mut events);
        events
    }

    fn poll_results(
        &mut self,
        world: &mut hecs::World,
        events: &mut Vec<GeneratorEvent>,
    ) {
        use crate::generator::error::GeneratorError;

        while let Ok(evt) = self.rx.try_recv() {
            match evt {
                GeneratorWorkerEvent::Completed {
                    generator_entity,
                    generation,
                    generator_name,
                    param_hash,
                } => {
                    let tracked = self.tracked.entry(generator_entity).or_default();
                    if generation != tracked.submitted_generation {
                        continue;
                    }
                    tracked.active_cancel = None;
                    tracked.active_progress = None;
                    if let Ok(mut state) = world.get::<&mut GeneratorState>(generator_entity) {
                        state.status = GeneratorStatus::Ready;
                        state.generation = state.generation.wrapping_add(1);
                        state.param_hash = param_hash;
                    }
                    events.push(GeneratorEvent::Completed {
                        entity: generator_entity,
                        name: generator_name,
                    });
                }
                GeneratorWorkerEvent::Failed {
                    generator_entity,
                    generation,
                    generator_name,
                    param_hash: _,
                    error,
                } => {
                    let tracked = self.tracked.entry(generator_entity).or_default();
                    if generation != tracked.submitted_generation {
                        continue;
                    }
                    tracked.active_cancel = None;
                    tracked.active_progress = None;
                    match error {
                        GeneratorError::Cancelled => {
                            events.push(GeneratorEvent::Cancelled {
                                entity: generator_entity,
                                name: generator_name,
                            });
                        }
                        err => {
                            let msg = err.to_string();
                            if let Ok(mut state) =
                                world.get::<&mut GeneratorState>(generator_entity)
                            {
                                state.status = GeneratorStatus::Error(msg.clone());
                            }
                            events.push(GeneratorEvent::Failed {
                                entity: generator_entity,
                                name: generator_name,
                                error: msg,
                            });
                        }
                    }
                }
            }
        }
    }

    fn scan_and_submit(
        &mut self,
        world: &mut hecs::World,
        components: &ComponentRegistry,
        entity_uuids: &std::collections::HashMap<hecs::Entity, uuid::Uuid>,
        child_cache_dir: Option<&std::path::Path>,
        events: &mut Vec<GeneratorEvent>,
    ) {
        // Collect entities that need submission. Deferred because we need
        // to mutate the world (set status → Generating) while we still have
        // the reference borrowed for the query.
        struct Pending {
            entity: hecs::Entity,
            generator_name: String,
            transform: Transform,
            world_position: rkp_core::WorldPosition,
            param_hash: u64,
        }
        let mut pending: Vec<Pending> = Vec::new();
        // Entities whose saved `param_hash != 0` but loaded `status` is
        // the default Pending. Flip to Ready after the read borrow ends
        // so the inspector + UI see the correct lifecycle state.
        let mut pending_status_writes: Vec<hecs::Entity> = Vec::new();

        // For each known generator entity (by UUID), does any
        // GeneratorOwned child reference it as parent? Used below to
        // force regen when an entity's saved param_hash claims it
        // already ran but no children survived save+reload — the
        // common case being a generator that uses `emit_child`
        // (anonymous), which we deliberately never save (they're
        // disposable). Without this force, those generators would
        // load as a parent with zero children and silently render
        // nothing.
        let parents_with_children: std::collections::HashSet<uuid::Uuid> = world
            .query::<&super::owned::GeneratorOwned>()
            .iter()
            .map(|(_, owned)| owned.parent_uuid)
            .collect();

        for (entity, state) in world.query::<&GeneratorState>().iter() {
            if state.generator_name.is_empty() {
                continue;
            }
            let Some(entry) = self.registry.get(&state.generator_name) else {
                continue;
            };
            // Hash current params.
            let param_hash = match hash_component(
                world,
                entity,
                entry.param_component_name,
                components,
            ) {
                Some(h) => h,
                None => continue, // params missing — nothing to run against
            };

            // Does this generator have any persistent children that
            // survived load? If `state.param_hash != 0` claims a prior
            // run completed but we see zero children, force regen —
            // either the generator emits only anonymous children
            // (which we never save) or persistent caches were lost.
            // Without this, anonymous-only generators would silently
            // load their parent with no geometry.
            let has_children = entity_uuids
                .get(&entity)
                .map(|u| parents_with_children.contains(u))
                .unwrap_or(false);

            // First-time-this-session seed for entities loaded from
            // disk: if we've never seen this entity AND it carries a
            // non-zero saved `param_hash` AND its persistent children
            // are present, treat it as already-Ready and seed
            // `last_submitted_hash` from the saved value. Without
            // this, every freshly-loaded generator entity arrives
            // with status=Pending (transient field, defaults on serde)
            // and last_submitted_hash=0, so every load would force a
            // regen even when the on-disk bake cache is valid.
            let was_already_tracked = self.tracked.contains_key(&entity);
            let tracked = self.tracked.entry(entity).or_default();
            if !was_already_tracked && state.param_hash != 0 && has_children {
                tracked.last_submitted_hash = state.param_hash;
                if matches!(state.status, GeneratorStatus::Pending) {
                    pending_status_writes.push(entity);
                }
            }
            let needs_submit = match state.status {
                GeneratorStatus::Pending if state.param_hash == 0 => true,
                GeneratorStatus::Pending => tracked.last_submitted_hash != param_hash,
                GeneratorStatus::Stale => true,
                _ => tracked.last_submitted_hash != param_hash,
            } || (state.param_hash != 0 && !has_children);
            if !needs_submit {
                continue;
            }

            // Snapshot the context inputs now while we hold the query borrow.
            let transform = world
                .get::<&Transform>(entity)
                .map(|t| (*t).clone())
                .unwrap_or_default();
            // WorldPosition is split across chunk + local in the full engine;
            // for M2 the default (origin) is fine — generators don't use it yet.
            let world_position = rkp_core::WorldPosition::default();

            pending.push(Pending {
                entity,
                generator_name: state.generator_name.clone(),
                transform,
                world_position,
                param_hash,
            });
        }

        // Apply the post-load Pending → Ready flips now that the
        // query's read-borrow has been dropped.
        for entity in pending_status_writes {
            if let Ok(mut state) = world.get::<&mut GeneratorState>(entity) {
                state.status = GeneratorStatus::Ready;
            }
        }

        for p in pending {
            let Some(entry) = self.registry.get(&p.generator_name) else {
                continue;
            };

            // Clone params off the ECS via the macro-generated helper.
            let Some(params) = (entry.clone_params)(world, p.entity) else {
                continue;
            };

            // Tell the engine to blow away anonymous children (and
            // reset the slot-key tracker) before any new emits land.
            // Engine handles the actual despawn — it owns the
            // scene_mgr pool dealloc path.
            events.push(GeneratorEvent::WillResubmit {
                entity: p.entity,
                name: p.generator_name.clone(),
            });

            // Set status → Generating on the ECS.
            if let Ok(mut state) = world.get::<&mut GeneratorState>(p.entity) {
                state.status = GeneratorStatus::Generating;
            }

            // Replace any prior cancel/progress — the previous run's token
            // gets flipped so the worker bails early. Stale results from
            // before the cancel are dropped by the generation-counter check.
            let tracked = self.tracked.entry(p.entity).or_default();
            if let Some(old) = tracked.active_cancel.take() {
                old.cancel();
            }
            let cancel = CancelToken::new();
            let progress = ProgressHandle::new();
            tracked.active_cancel = Some(cancel.clone());
            tracked.active_progress = Some(progress.clone());
            tracked.submitted_generation = tracked.submitted_generation.wrapping_add(1);
            tracked.last_submitted_hash = p.param_hash;
            let submitted_gen = tracked.submitted_generation;

            let req = GeneratorRequest {
                entity: p.entity,
                generation: submitted_gen,
                generator_name: p.generator_name.clone(),
                param_hash: p.param_hash,
                params,
                cancel,
                progress,
                transform: p.transform,
                world_position: p.world_position,
                generate_fn: entry.generate_fn,
                generator_entity_uuid: entity_uuids.get(&p.entity).copied(),
                child_cache_dir: child_cache_dir.map(|p| p.to_path_buf()),
            };
            if self.tx.send(req).is_err() {
                return;
            }
            events.push(GeneratorEvent::Submitted {
                entity: p.entity,
                name: p.generator_name,
            });
        }
    }
}

/// Hash a component's serialized form. Matches rkifield's approach —
/// serialize via the registry's `serialize` fn, hash the resulting JSON
/// bytes. Stable as long as serialization is stable, which for `serde_json`
/// over plain `#[derive(Serialize)]` structs it is.
fn hash_component(
    world: &hecs::World,
    entity: hecs::Entity,
    component_name: &str,
    components: &ComponentRegistry,
) -> Option<u64> {
    let entry = components.get(component_name)?;
    let json = (entry.serialize)(world, entity)?;
    let mut h = DefaultHasher::new();
    json.hash(&mut h);
    Some(h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_registry::ComponentRegistry;

    fn worker_channels() -> (
        Sender<GeneratorRequest>,
        Receiver<GeneratorRequest>,
        Sender<GeneratorWorkerEvent>,
        Receiver<GeneratorWorkerEvent>,
    ) {
        let (tx_req, rx_req) = crossbeam::channel::unbounded();
        let (tx_res, rx_res) = crossbeam::channel::unbounded();
        (tx_req, rx_req, tx_res, rx_res)
    }

    fn empty_registry() -> GeneratorRegistry {
        // Bypass inventory scan — tests don't have real generators linked.
        GeneratorRegistry::default()
    }

    #[test]
    fn tick_without_generators_is_noop() {
        let (tx_req, _rx_req, _tx_res, rx_res) = worker_channels();
        let mut sys = GeneratorSystem::new(empty_registry(), tx_req, rx_res);
        let mut world = hecs::World::new();
        let components = ComponentRegistry::new();

        let entity_uuids = std::collections::HashMap::new();
        let events = sys.tick(&mut world, &components, &entity_uuids, None);
        assert!(events.is_empty());
    }

    #[test]
    fn forget_drops_entity() {
        let (tx_req, _rx_req, _tx_res, rx_res) = worker_channels();
        let mut sys = GeneratorSystem::new(empty_registry(), tx_req, rx_res);
        let e = hecs::Entity::from_bits(0x0000_0001_0000_0001).unwrap();
        sys.tracked.insert(e, Tracked::default());
        assert!(sys.tracked.contains_key(&e));
        sys.forget(e);
        assert!(!sys.tracked.contains_key(&e));
    }

    #[test]
    fn cancel_sets_token() {
        let (tx_req, _rx_req, _tx_res, rx_res) = worker_channels();
        let mut sys = GeneratorSystem::new(empty_registry(), tx_req, rx_res);
        let e = hecs::Entity::from_bits(0x0000_0001_0000_0001).unwrap();
        let token = CancelToken::new();
        sys.tracked.insert(
            e,
            Tracked {
                active_cancel: Some(token.clone()),
                active_progress: Some(ProgressHandle::new()),
                ..Default::default()
            },
        );
        sys.cancel(e);
        assert!(token.is_cancelled());
        assert!(sys.tracked[&e].active_cancel.is_none());
        assert!(sys.tracked[&e].active_progress.is_none());
    }
}
