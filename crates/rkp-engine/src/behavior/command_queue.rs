//! Command queue — deferred ECS mutations for systems.
//!
//! Systems cannot directly mutate the world (it's borrowed). Instead they
//! queue spawn/despawn/insert/remove commands which are flushed between phases.

use std::any::TypeId;
use std::collections::HashMap;

/// Handle to a not-yet-spawned entity. Use with `commands.insert_temp()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TempEntity(usize);

/// Type-erased component insertion.
trait ComponentBox: Send + 'static {
    fn insert_into(self: Box<Self>, world: &mut hecs::World, entity: hecs::Entity);
    fn type_id(&self) -> TypeId;
}

struct TypedComponent<C: hecs::Component> {
    component: C,
}

impl<C: hecs::Component> ComponentBox for TypedComponent<C> {
    fn insert_into(self: Box<Self>, world: &mut hecs::World, entity: hecs::Entity) {
        let _ = world.insert_one(entity, self.component);
    }
    fn type_id(&self) -> TypeId {
        TypeId::of::<C>()
    }
}

struct PendingInsert {
    data: Box<dyn ComponentBox>,
}

struct PendingRemove {
    remove_fn: fn(&mut hecs::World, hecs::Entity),
}

struct PendingSpawn {
    builder: hecs::EntityBuilder,
}

/// A request from a gameplay system targeting viewport state rather than
/// the ECS world — e.g., swapping the play-mode camera mid-session. The
/// engine drains these after systems tick (the executor itself never
/// touches viewports).
#[derive(Debug, Clone, Copy)]
pub enum ViewportRequest {
    /// Set MAIN's runtime_override to this entity. Expects the entity to
    /// carry a `Camera` + `Transform` component; if either is missing, the
    /// override resolves to the editor camera as usual.
    SetActiveCamera(hecs::Entity),
    /// Clear MAIN's runtime_override so rendering falls back to the
    /// editor camera (or the behavior's next `set_active_camera` call).
    ClearActiveCamera,
}

/// Deferred ECS mutation queue.
///
/// Systems push commands; the executor flushes between phases.
pub struct CommandQueue {
    spawns: Vec<PendingSpawn>,
    temp_inserts: HashMap<usize, Vec<PendingInsert>>,
    entity_inserts: Vec<(hecs::Entity, PendingInsert)>,
    despawns: Vec<hecs::Entity>,
    removes: Vec<(hecs::Entity, PendingRemove)>,
    /// Viewport-scoped requests. Drained by the engine after systems tick
    /// — not by `flush()`, since flushing only reaches the ECS world.
    viewport_requests: Vec<ViewportRequest>,
    /// Entities that were spawned during the most recent flush.
    spawned: Vec<hecs::Entity>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self {
            spawns: Vec::new(),
            temp_inserts: HashMap::new(),
            entity_inserts: Vec::new(),
            despawns: Vec::new(),
            removes: Vec::new(),
            viewport_requests: Vec::new(),
            spawned: Vec::new(),
        }
    }

    /// Push a viewport-level request (e.g. camera swap). Drained by the
    /// engine after the system tick finishes — `flush()` ignores these.
    pub fn push_viewport_request(&mut self, request: ViewportRequest) {
        self.viewport_requests.push(request);
    }

    /// Drain pending viewport requests. The engine calls this after
    /// systems tick; within gameplay code use the
    /// `SystemContext::set_active_camera` / `clear_active_camera` helpers.
    pub fn take_viewport_requests(&mut self) -> Vec<ViewportRequest> {
        std::mem::take(&mut self.viewport_requests)
    }

    /// Queue a new entity spawn. Returns a `TempEntity` handle for queuing
    /// additional components via `insert_temp`.
    pub fn spawn(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn { builder });
        TempEntity(idx)
    }

    /// Queue a component insertion on a not-yet-spawned entity.
    pub fn insert_temp<C: hecs::Component>(&mut self, temp: TempEntity, component: C) {
        self.temp_inserts
            .entry(temp.0)
            .or_default()
            .push(PendingInsert {
                data: Box::new(TypedComponent { component }),
            });
    }

    /// Queue a component insertion on an existing entity.
    pub fn insert<C: hecs::Component>(&mut self, entity: hecs::Entity, component: C) {
        self.entity_inserts.push((
            entity,
            PendingInsert {
                data: Box::new(TypedComponent { component }),
            },
        ));
    }

    /// Queue a component removal from an entity.
    pub fn remove<C: hecs::Component>(&mut self, entity: hecs::Entity) {
        self.removes.push((
            entity,
            PendingRemove {
                remove_fn: |world, entity| {
                    let _ = world.remove_one::<C>(entity);
                },
            },
        ));
    }

    /// Queue an entity despawn (cascading to children).
    pub fn despawn(&mut self, entity: hecs::Entity) {
        self.despawns.push(entity);
    }

    /// Entities that were spawned during the most recent `flush()`.
    pub fn spawned_this_frame(&self) -> &[hecs::Entity] {
        &self.spawned
    }

    /// Flush all queued commands into the world.
    ///
    /// Order: spawns → inserts on existing → removes → despawns.
    /// Auto-injects Transform + EditorMetadata on spawned entities if missing.
    pub fn flush(&mut self, world: &mut hecs::World) {
        self.spawned.clear();

        // Phase 1: Spawns + temp inserts
        let spawns = std::mem::take(&mut self.spawns);
        let mut temp_inserts = std::mem::take(&mut self.temp_inserts);

        for (idx, mut pending) in spawns.into_iter().enumerate() {
            let entity = world.spawn(pending.builder.build());

            // Apply temp inserts for this spawn.
            if let Some(inserts) = temp_inserts.remove(&idx) {
                for insert in inserts {
                    insert.data.insert_into(world, entity);
                }
            }

            // Auto-inject defaults if missing.
            if world.get::<&crate::components::Transform>(entity).is_err() {
                let _ = world.insert_one(entity, crate::components::Transform::default());
            }
            if world.get::<&crate::components::EditorMetadata>(entity).is_err() {
                let _ = world.insert_one(entity, crate::components::EditorMetadata::default());
            }

            self.spawned.push(entity);
        }

        // Phase 2: Inserts on existing entities
        let inserts = std::mem::take(&mut self.entity_inserts);
        for (entity, insert) in inserts {
            if world.contains(entity) {
                insert.data.insert_into(world, entity);
            }
        }

        // Phase 3: Removes
        let removes = std::mem::take(&mut self.removes);
        for (entity, remove) in removes {
            if world.contains(entity) {
                (remove.remove_fn)(world, entity);
            }
        }

        // Phase 4: Despawns (cascading)
        let despawns = std::mem::take(&mut self.despawns);
        for entity in despawns {
            despawn_cascading(world, entity);
        }
    }

    /// Whether all queues are empty.
    pub fn is_empty(&self) -> bool {
        self.spawns.is_empty()
            && self.temp_inserts.is_empty()
            && self.entity_inserts.is_empty()
            && self.despawns.is_empty()
            && self.removes.is_empty()
            && self.viewport_requests.is_empty()
    }
}

/// Despawn an entity and all its children (entities with Parent pointing to it).
fn despawn_cascading(world: &mut hecs::World, entity: hecs::Entity) {
    // Collect children first to avoid borrow conflicts.
    let children: Vec<hecs::Entity> = world
        .query::<&crate::components::Parent>()
        .iter()
        .filter(|(_, p)| {
            // Check if this entity's parent_id matches.
            // Parent stores a UUID; we'd need to resolve. For now, use direct entity refs
            // if available. This is simplified — full implementation needs UUID resolution.
            false // TODO: implement parent-child cascading when Parent component is resolved
        })
        .map(|(e, _)| e)
        .collect();

    for child in children {
        despawn_cascading(world, child);
    }

    let _ = world.despawn(entity);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_and_flush() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();

        let mut builder = hecs::EntityBuilder::new();
        builder.add(42u32);
        queue.spawn(builder);

        assert_eq!(world.len(), 0);
        queue.flush(&mut world);
        assert_eq!(world.len(), 1);
        assert_eq!(queue.spawned_this_frame().len(), 1);

        // Verify auto-injected components.
        let entity = queue.spawned_this_frame()[0];
        assert!(world.get::<&crate::components::Transform>(entity).is_ok());
        assert!(world.get::<&crate::components::EditorMetadata>(entity).is_ok());
    }

    #[test]
    fn despawn_and_flush() {
        let mut world = hecs::World::new();
        let entity = world.spawn((42u32,));
        assert_eq!(world.len(), 1);

        let mut queue = CommandQueue::new();
        queue.despawn(entity);
        queue.flush(&mut world);
        assert_eq!(world.len(), 0);
    }

    #[test]
    fn insert_on_existing() {
        let mut world = hecs::World::new();
        let entity = world.spawn((42u32,));

        let mut queue = CommandQueue::new();
        queue.insert(entity, 3.14f32);
        queue.flush(&mut world);

        assert_eq!(*world.get::<&f32>(entity).unwrap(), 3.14);
    }
}
