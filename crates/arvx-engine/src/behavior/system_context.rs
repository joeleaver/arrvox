//! System context — the API surface available to gameplay systems.
//!
//! Every `#[arvx_system]` function receives `&mut SystemContext`. It provides:
//! - Read-only world queries (query, get, has)
//! - Deferred mutations via CommandQueue (spawn, despawn, insert, remove)
//! - Engine component reads via EngineAccess (cross-dylib safe)
//! - Buffered transform updates (set_position, set_rotation, set_transform)
//! - Game store access (key-value state + events)
//! - Time (delta_time, total_time, frame)

use glam::Vec3;

use super::command_queue::{CommandQueue, TempEntity, ViewportRequest};
use super::engine_access::{EngineAccess, TransformUpdate};
use super::game_store::GameStore;

/// The context passed to every system function.
pub struct SystemContext<'a> {
    world: &'a mut hecs::World,
    commands: &'a mut CommandQueue,
    store: &'a mut GameStore,
    engine: &'a dyn EngineAccess,
    transform_updates: Vec<TransformUpdate>,
    delta_time: f32,
    total_time: f64,
    frame: u64,
}

impl<'a> SystemContext<'a> {
    pub fn new(
        world: &'a mut hecs::World,
        commands: &'a mut CommandQueue,
        store: &'a mut GameStore,
        engine: &'a dyn EngineAccess,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) -> Self {
        Self {
            world,
            commands,
            store,
            engine,
            transform_updates: Vec::new(),
            delta_time,
            total_time,
            frame,
        }
    }

    // ── World queries (read-only) ───────────────────────────────────

    /// Query the world for entities matching a component tuple.
    ///
    /// The returned `QueryBorrow` holds an immutable borrow on the world for
    /// its lifetime. If you need to call mutable methods on `ctx` (like
    /// `set_rotation`, `despawn`, etc.) inside the loop, collect first:
    ///
    /// ```ignore
    /// // Won't compile — query borrows ctx immutably, set_rotation borrows mutably
    /// for (entity, spin) in ctx.query::<&Spin>().iter() {
    ///     ctx.set_rotation(entity, new_rot);  // ERROR: mutable borrow
    /// }
    ///
    /// // Collect first to release the borrow, then mutate
    /// let spins: Vec<_> = ctx.query::<&Spin>()
    ///     .iter()
    ///     .map(|(entity, spin)| (entity, spin.speed, spin.axis))
    ///     .collect();
    /// for (entity, speed, axis) in spins {
    ///     ctx.set_rotation(entity, new_rot);  // OK
    /// }
    /// ```
    pub fn query<Q: hecs::Query>(&self) -> hecs::QueryBorrow<'_, Q> {
        self.world.query::<Q>()
    }

    /// Get a single component from an entity.
    pub fn get<C: hecs::ComponentRef<'a>>(&'a self, entity: hecs::Entity) -> Result<C::Ref, hecs::ComponentError> {
        self.world.get::<C>(entity)
    }

    /// Check if an entity has a component.
    pub fn has<C: hecs::Component>(&self, entity: hecs::Entity) -> bool {
        self.world.get::<&C>(entity).is_ok()
    }

    /// Check if an entity exists.
    pub fn entity_exists(&self, entity: hecs::Entity) -> bool {
        self.world.contains(entity)
    }

    // ── Deferred mutations ──────────────────────────────────────────

    /// Access the command queue directly.
    pub fn commands(&mut self) -> &mut CommandQueue {
        self.commands
    }

    /// Spawn a new entity (deferred). Returns a handle for queuing components.
    pub fn spawn(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        self.commands.spawn(builder)
    }

    /// Despawn an entity (deferred, cascading to children).
    pub fn despawn(&mut self, entity: hecs::Entity) {
        self.commands.despawn(entity);
    }

    /// Insert a component on an existing entity (deferred).
    pub fn insert<C: hecs::Component>(&mut self, entity: hecs::Entity, component: C) {
        self.commands.insert(entity, component);
    }

    // ── Game store ──────────────────────────────────────────────────

    /// Mutable access to the game store.
    pub fn store(&mut self) -> &mut GameStore {
        self.store
    }

    /// Immutable access to the game store.
    pub fn store_ref(&self) -> &GameStore {
        self.store
    }

    // ── Engine component access (cross-dylib safe) ──────────────────

    /// Access the engine component bridge.
    pub fn engine(&self) -> &dyn EngineAccess {
        self.engine
    }

    /// Read an entity's position.
    pub fn position(&self, entity: hecs::Entity) -> Option<Vec3> {
        self.engine.position(entity)
    }

    /// Read an entity's full transform: (position, rotation_euler, scale).
    pub fn get_transform(&self, entity: hecs::Entity) -> Option<(Vec3, Vec3, Vec3)> {
        self.engine.transform(entity)
    }

    /// Find an entity by its EditorMetadata name.
    pub fn find_entity_by_name(&self, name: &str) -> Option<hecs::Entity> {
        self.engine.find_entity_by_name(name)
    }

    // ── Buffered transform updates ──────────────────────────────────

    /// Set an entity's position (buffered, applied after system returns).
    pub fn set_position(&mut self, entity: hecs::Entity, position: Vec3) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: Some(position),
            rotation: None,
            scale: None,
        });
    }

    /// Set an entity's rotation in Euler degrees (buffered).
    pub fn set_rotation(&mut self, entity: hecs::Entity, rotation: Vec3) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: None,
            rotation: Some(rotation),
            scale: None,
        });
    }

    /// Set an entity's full transform (buffered).
    pub fn set_transform(&mut self, entity: hecs::Entity, position: Vec3, rotation: Vec3, scale: Vec3) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: Some(position),
            rotation: Some(rotation),
            scale: Some(scale),
        });
    }

    /// Drain and return buffered transform updates. Called by the executor.
    pub fn take_transform_updates(&mut self) -> Vec<TransformUpdate> {
        std::mem::take(&mut self.transform_updates)
    }

    // ── Viewport control ────────────────────────────────────────────

    /// Hand the MAIN viewport over to `entity`'s Camera + Transform.
    /// Takes effect after this system returns, when the engine drains
    /// pending viewport requests. The entity must carry both components
    /// for the swap to have a visible effect — otherwise rendering falls
    /// back to the editor camera as usual.
    pub fn set_active_camera(&mut self, entity: hecs::Entity) {
        self.commands.push_viewport_request(ViewportRequest::SetActiveCamera(entity));
    }

    /// Clear the MAIN viewport's runtime-camera override. Rendering
    /// returns to the editor camera on the next frame.
    pub fn clear_active_camera(&mut self) {
        self.commands.push_viewport_request(ViewportRequest::ClearActiveCamera);
    }

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Entities that were spawned during the most recent command flush.
    pub fn spawned_this_frame(&self) -> &[hecs::Entity] {
        self.commands.spawned_this_frame()
    }

    // ── Time ────────────────────────────────────────────────────────

    /// Seconds since last frame (variable for Update/LateUpdate, fixed for FixedUpdate).
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Total elapsed time in seconds (high precision).
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Monotonic frame counter.
    pub fn frame(&self) -> u64 {
        self.frame
    }
}
