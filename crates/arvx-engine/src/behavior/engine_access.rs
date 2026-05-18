//! Engine access bridge — cross-dylib safe reads of host-side components.
//!
//! Systems in the gameplay dylib cannot directly query host-side components
//! (Transform, EditorMetadata) because TypeIds differ across dylib boundaries.
//! The `EngineAccess` trait provides a safe bridge: the host implements it
//! using its own TypeIds, and systems receive `&dyn EngineAccess`.

use glam::Vec3;

/// Cross-dylib safe interface for reading engine components.
///
/// Implemented by `WorldEngineAccess` in the host. Systems receive this
/// as `ctx.engine()` and use it to read Transform, find entities by name, etc.
pub trait EngineAccess {
    /// Read an entity's position (Vec3).
    fn position(&self, entity: hecs::Entity) -> Option<Vec3>;
    /// Read an entity's rotation (Euler degrees, Vec3).
    fn rotation_euler(&self, entity: hecs::Entity) -> Option<Vec3>;
    /// Read an entity's scale (Vec3).
    fn scale(&self, entity: hecs::Entity) -> Option<Vec3>;
    /// Read an entity's full transform: (position, rotation_euler, scale).
    fn transform(&self, entity: hecs::Entity) -> Option<(Vec3, Vec3, Vec3)>;
    /// Find an entity by its EditorMetadata name.
    fn find_entity_by_name(&self, name: &str) -> Option<hecs::Entity>;
}

/// A buffered transform update from a system.
///
/// Systems call `ctx.set_position()` / `ctx.set_rotation()` / `ctx.set_transform()`
/// which queue these updates. The executor applies them after each system returns.
/// Optional fields allow partial updates (e.g., rotate without moving).
pub struct TransformUpdate {
    pub entity: hecs::Entity,
    pub position: Option<Vec3>,
    /// Euler degrees.
    pub rotation: Option<Vec3>,
    pub scale: Option<Vec3>,
}

/// Host-side implementation of `EngineAccess` using a raw world pointer.
///
/// # Safety
///
/// The raw pointer must remain valid and not be mutably aliased for the
/// lifetime of this struct. The executor creates it from `&raw const *world`
/// before constructing `SystemContext` with `&mut world`, ensuring no
/// mutable aliasing occurs through the trait methods (which only read).
pub struct WorldEngineAccess {
    world: *const hecs::World,
}

impl WorldEngineAccess {
    /// # Safety
    /// Caller guarantees: pointer is valid, no mutable aliasing during method calls.
    pub unsafe fn new(world: *const hecs::World) -> Self {
        Self { world }
    }
}

impl EngineAccess for WorldEngineAccess {
    fn position(&self, entity: hecs::Entity) -> Option<Vec3> {
        let world = unsafe { &*self.world };
        world.get::<&crate::components::Transform>(entity).ok().map(|t| t.position)
    }

    fn rotation_euler(&self, entity: hecs::Entity) -> Option<Vec3> {
        let world = unsafe { &*self.world };
        world.get::<&crate::components::Transform>(entity).ok().map(|t| t.rotation)
    }

    fn scale(&self, entity: hecs::Entity) -> Option<Vec3> {
        let world = unsafe { &*self.world };
        world.get::<&crate::components::Transform>(entity).ok().map(|t| t.scale)
    }

    fn transform(&self, entity: hecs::Entity) -> Option<(Vec3, Vec3, Vec3)> {
        let world = unsafe { &*self.world };
        world.get::<&crate::components::Transform>(entity)
            .ok()
            .map(|t| (t.position, t.rotation, t.scale))
    }

    fn find_entity_by_name(&self, name: &str) -> Option<hecs::Entity> {
        let world = unsafe { &*self.world };
        for (entity, meta) in world.query::<&crate::components::EditorMetadata>().iter() {
            if meta.name == name {
                return Some(entity);
            }
        }
        None
    }
}
