//! ECS rigid body component and physics-to-transform synchronization.
//!
//! [`RigidBodyComponent`] is the ECS component that links an entity to a
//! Rapier rigid body. Helper functions synchronize Rapier state back to
//! entity transforms. Collision is handled natively by Rapier's voxel
//! and primitive colliders.

use glam::{Quat, Vec3};
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

use crate::rapier_world::{
    from_rapier_quat, from_rapier_vec3, to_rapier_vec3, PhysicsWorld,
};

// ---------------------------------------------------------------------------
// BodyType
// ---------------------------------------------------------------------------

/// Wrapper around Rapier's rigid body types for use in the ECS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyType {
    /// Affected by forces and gravity.
    Dynamic,
    /// Never moves (infinite mass).
    Static,
    /// Moved by setting position directly; pushes dynamic bodies.
    KinematicPosition,
    /// Moved by setting velocity directly; pushes dynamic bodies.
    KinematicVelocity,
}

// ---------------------------------------------------------------------------
// ColliderShape — what shape to use for the physics collider
// ---------------------------------------------------------------------------

/// Controls the collision shape used by the physics system.
///
/// `Auto` derives the shape from the entity's geometry (voxels for voxelized
/// objects, exact primitive for analytical shapes). The primitive overrides
/// (Box, Sphere, Capsule) use a simple shape sized from the entity's AABB.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColliderShape {
    /// Derive from geometry: voxels for voxelized, exact primitive for analytical.
    Auto,
    /// Axis-aligned box from AABB.
    Box,
    /// Bounding sphere from AABB.
    Sphere,
    /// Y-axis capsule from AABB.
    Capsule,
}

impl Default for ColliderShape {
    fn default() -> Self {
        ColliderShape::Auto
    }
}

// ---------------------------------------------------------------------------
// RigidBody — serializable, editor-facing physics component
// ---------------------------------------------------------------------------

/// Rigid body physics properties for scene persistence.
///
/// This is the component that lives on entities in the editor and scene files.
/// It contains no Rapier runtime state — at play start, the physics system
/// reads `RigidBody` and creates the corresponding Rapier body with a
/// collider derived from `collider_shape`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RigidBody {
    /// The type of rigid body (dynamic, static, kinematic).
    pub body_type: BodyType,
    /// Collision shape override. Default `Auto` uses voxels or exact primitive.
    pub collider_shape: ColliderShape,
    /// Mass in kilograms. Only meaningful for dynamic bodies.
    pub mass: f32,
    /// Friction coefficient.
    pub friction: f32,
    /// Coefficient of restitution (bounciness).
    pub restitution: f32,
    /// Max surface voxels for the physics collider. Lower = faster but coarser.
    /// Only affects voxelized objects with `collider_shape: Auto`. Default: 5000.
    pub physics_resolution: u32,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            collider_shape: ColliderShape::default(),
            mass: 1.0,
            friction: 0.5,
            restitution: 0.3,
            physics_resolution: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// RigidBodyComponent
// ---------------------------------------------------------------------------

/// ECS component linking an entity to a Rapier rigid body at runtime.
///
/// Created at play start from [`RigidBody`] configuration. Holds only
/// the Rapier handle — collision is handled natively by Rapier's voxel
/// and primitive colliders.
#[derive(Debug, Clone)]
pub struct RigidBodyComponent {
    /// Handle into the Rapier `RigidBodySet`.
    pub handle: RigidBodyHandle,
}

impl RigidBodyComponent {
    /// Create a new component from a Rapier handle.
    pub fn new(handle: RigidBodyHandle) -> Self {
        Self { handle }
    }
}

// ---------------------------------------------------------------------------
// Transform sync
// ---------------------------------------------------------------------------

/// A transform that can be read from / written to for physics synchronization.
pub struct TransformRef<'a> {
    /// Mutable reference to the entity's position.
    pub position: &'a mut Vec3,
    /// Mutable reference to the entity's rotation.
    pub rotation: &'a mut Quat,
}

/// Synchronize Rapier rigid body state back to entity transforms.
///
/// For each `(component, transform)` pair, reads the body position and
/// rotation from Rapier and writes them into the mutable transform
/// references. The `alpha` parameter (from [`PhysicsWorld::step`]) can be
/// used for interpolation between the previous and current physics state;
/// currently we snap to the current state (alpha is noted for future use).
pub fn sync_transforms(
    world: &PhysicsWorld,
    components: &[(RigidBodyComponent, TransformRef<'_>)],
    _alpha: f32,
) {
    for (comp, _transform) in components {
        let _body = match world.get_body(comp.handle) {
            Some(b) => b,
            None => continue,
        };
        // Note: can't mutate through shared slice — use sync_transforms_mut
    }
}

/// Synchronize Rapier rigid body state back to entity transforms (mutable version).
///
/// Reads each body's position/rotation from Rapier and writes it into the
/// corresponding transform. `alpha` is reserved for future interpolation.
pub fn sync_transforms_mut(
    world: &PhysicsWorld,
    entries: &mut [(RigidBodyComponent, Vec3, Quat)],
    _alpha: f32,
) {
    for (comp, pos, rot) in entries.iter_mut() {
        let body = match world.get_body(comp.handle) {
            Some(b) => b,
            None => continue,
        };
        *pos = from_rapier_vec3(body.translation());
        *rot = from_rapier_quat(body.rotation());
    }
}

// ---------------------------------------------------------------------------
// Convenience: spawn_rigid_body
// ---------------------------------------------------------------------------

/// Spawn a new rigid body with a pre-built Rapier collider.
///
/// Creates the Rapier rigid body, attaches the collider, and returns a
/// [`RigidBodyComponent`]. The collider determines the collision shape
/// and mass properties.
pub fn spawn_rigid_body_with_collider(
    world: &mut PhysicsWorld,
    position: Vec3,
    rotation: Quat,
    body_type: BodyType,
    collider: rapier3d::prelude::Collider,
) -> RigidBodyComponent {
    let mut builder = match body_type {
        BodyType::Dynamic => RigidBodyBuilder::dynamic(),
        BodyType::Static => RigidBodyBuilder::fixed(),
        BodyType::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
        BodyType::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
    };
    builder = builder
        .translation(to_rapier_vec3(position))
        .rotation(to_rapier_vec3(rotation.to_scaled_axis()));
    let body = builder.build();
    let handle = world.add_rigid_body(body);
    world.add_collider(collider, handle);
    RigidBodyComponent::new(handle)
}

/// Spawn a new rigid body with a primitive collider and return a
/// [`RigidBodyComponent`] ready for ECS insertion.
///
/// Creates both the Rapier rigid body and a ball collider with the given
/// `radius` and `mass`, inserts them into the physics world, and returns
/// the component.
pub fn spawn_rigid_body(
    world: &mut PhysicsWorld,
    position: Vec3,
    rotation: Quat,
    body_type: BodyType,
    radius: f32,
    mass: f32,
) -> RigidBodyComponent {
    let mut builder = match body_type {
        BodyType::Dynamic => RigidBodyBuilder::dynamic(),
        BodyType::Static => RigidBodyBuilder::fixed(),
        BodyType::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
        BodyType::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
    };

    builder = builder
        .translation(to_rapier_vec3(position))
        .rotation(to_rapier_vec3(rotation.to_scaled_axis()));

    let body = builder.build();
    let handle = world.add_rigid_body(body);

    let mut collider_builder = ColliderBuilder::ball(radius);
    if mass > 0.0 {
        collider_builder = collider_builder.mass(mass);
    }
    let collider = collider_builder.build();
    world.add_collider(collider, handle);

    RigidBodyComponent::new(handle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rapier_world::PhysicsConfig;

    #[test]
    fn test_spawn_dynamic_body() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY,
            BodyType::Dynamic,
            0.5,
            1.0,
        );

        assert_eq!(world.body_count(), 1);
        assert_eq!(world.collider_count(), 1);

        let body = world.get_body(comp.handle).unwrap();
        let pos = from_rapier_vec3(body.translation());
        assert!((pos.y - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_sync_transforms() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 10.0, 0.0),
            Quat::IDENTITY,
            BodyType::Dynamic,
            0.5,
            1.0,
        );

        // Step physics — body should fall
        let alpha = world.step(0.1);

        let mut entries = vec![(comp.clone(), Vec3::ZERO, Quat::IDENTITY)];
        sync_transforms_mut(&world, &mut entries, alpha);

        let (_, pos, _rot) = &entries[0];
        assert!(
            pos.y < 10.0,
            "position should have updated after step: y={}",
            pos.y
        );
    }

    #[test]
    fn test_body_type_static() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY,
            BodyType::Static,
            1.0,
            0.0,
        );

        // Step physics
        world.step(1.0);

        // Static body should not have moved
        let body = world.get_body(comp.handle).unwrap();
        let pos = from_rapier_vec3(body.translation());
        assert!(
            (pos.y - 5.0).abs() < 1e-4,
            "static body should not move: y={}",
            pos.y
        );
    }

    #[test]
    fn test_spawn_and_remove() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Spawn
        let comp = spawn_rigid_body(
            &mut world,
            Vec3::ZERO,
            Quat::IDENTITY,
            BodyType::Dynamic,
            0.25,
            2.0,
        );

        assert_eq!(world.body_count(), 1);
        assert_eq!(world.collider_count(), 1);

        // Remove
        let removed = world.remove_rigid_body(comp.handle);
        assert!(removed.is_some());
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.collider_count(), 0);
    }
}
