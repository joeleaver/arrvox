//! Rapier physics world setup and stepping.
//!
//! [`PhysicsWorld`] wraps all Rapier state and provides a fixed-timestep
//! accumulator for deterministic simulation.
//!
//! ## glam version bridging
//!
//! The workspace uses glam 0.29 while rapier3d 0.32 depends on glam 0.30
//! internally. The Rapier API surfaces glam 0.30 types (`rapier3d::math::*`).
//! Thin conversion helpers bridge the two versions via component extraction.

use glam::{Quat, Vec3};
use rapier3d::prelude::*;

// ---------------------------------------------------------------------------
// glam 0.29 <-> rapier (glam 0.30) conversions
// ---------------------------------------------------------------------------

/// Convert a workspace `glam::Vec3` (0.29) to Rapier's `Vector` (glam 0.30).
#[inline]
pub fn to_rapier_vec3(v: Vec3) -> rapier3d::math::Vector {
    rapier3d::math::Vector::new(v.x, v.y, v.z)
}

/// Convert Rapier's `Vector` (glam 0.30) to workspace `glam::Vec3` (0.29).
#[inline]
pub fn from_rapier_vec3(v: rapier3d::math::Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Convert a workspace `glam::Quat` (0.29) to Rapier's `Rotation` (glam 0.30 Quat).
#[inline]
pub fn to_rapier_quat(q: Quat) -> rapier3d::math::Rotation {
    rapier3d::math::Rotation::from_xyzw(q.x, q.y, q.z, q.w)
}

/// Convert Rapier's `Rotation` (glam 0.30 Quat) to workspace `glam::Quat` (0.29).
#[inline]
pub fn from_rapier_quat(r: &rapier3d::math::Rotation) -> Quat {
    Quat::from_xyzw(r.x, r.y, r.z, r.w)
}

// ---------------------------------------------------------------------------
// PhysicsConfig
// ---------------------------------------------------------------------------

/// Configuration for the physics simulation.
#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    /// Gravity vector in world space. Default: `(0, -9.81, 0)`.
    pub gravity: Vec3,
    /// Fixed timestep for each physics substep. Default: `1/60`.
    pub timestep: f32,
    /// Maximum number of substeps per frame to prevent spiral of death.
    /// Default: `4`.
    pub max_substeps: u32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            timestep: 1.0 / 60.0,
            max_substeps: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// PhysicsWorld
// ---------------------------------------------------------------------------

/// Wrapper around all Rapier physics state with fixed-timestep accumulator.
pub struct PhysicsWorld {
    /// Set of all rigid bodies.
    pub rigid_body_set: RigidBodySet,
    /// Set of all colliders.
    pub collider_set: ColliderSet,
    /// Integration parameters (timestep, solver iterations, etc.).
    pub integration_params: IntegrationParameters,
    /// The main physics pipeline that drives simulation.
    pub physics_pipeline: PhysicsPipeline,
    /// Tracks active/sleeping islands for bodies.
    pub island_manager: IslandManager,
    /// Broad-phase collision detection (BVH-based).
    pub broad_phase: DefaultBroadPhase,
    /// Narrow-phase collision detection (exact contacts).
    pub narrow_phase: NarrowPhase,
    /// Impulse-based joints (ball, revolute, prismatic, etc.).
    pub impulse_joint_set: ImpulseJointSet,
    /// Multibody joints (for reduced-coordinate articulations).
    pub multibody_joint_set: MultibodyJointSet,
    /// Continuous collision detection solver.
    pub ccd_solver: CCDSolver,
    /// Physics configuration (gravity, timestep, substep limit).
    pub config: PhysicsConfig,
    /// Time accumulator for fixed-timestep stepping.
    accumulator: f32,
    /// Substeps performed in the most recent `step()` call. Surfaced to
    /// the editor's profiling readout so users can tell whether physics
    /// is stepping at the target rate.
    pub last_step_substeps: u32,
}

impl PhysicsWorld {
    /// Create a new physics world with the given configuration.
    pub fn new(config: PhysicsConfig) -> Self {
        let integration_params = IntegrationParameters {
            dt: config.timestep,
            ..Default::default()
        };

        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_params,
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            config,
            accumulator: 0.0,
            last_step_substeps: 0,
        }
    }

    /// Advance the simulation by `dt` seconds using a fixed timestep accumulator.
    ///
    /// Returns the interpolation factor `alpha` in `[0, 1)` for smooth rendering.
    /// If `alpha == 0.0`, the simulation is exactly at a timestep boundary.
    pub fn step(&mut self, dt: f32) -> f32 {
        self.accumulator += dt;

        let gravity = to_rapier_vec3(self.config.gravity);
        let mut substeps = 0u32;

        while self.accumulator >= self.config.timestep && substeps < self.config.max_substeps {
            self.physics_pipeline.step(
                gravity,
                &self.integration_params,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                &mut self.ccd_solver,
                &(),
                &(),
            );

            self.accumulator -= self.config.timestep;
            substeps += 1;
        }

        // Clamp accumulator if we hit the substep limit (prevent spiral of death)
        if substeps >= self.config.max_substeps {
            self.accumulator = 0.0;
        }

        self.last_step_substeps = substeps;

        // Return interpolation factor
        if self.config.timestep > 0.0 {
            self.accumulator / self.config.timestep
        } else {
            0.0
        }
    }

    /// Insert a rigid body into the simulation. Returns its handle.
    pub fn add_rigid_body(&mut self, body: RigidBody) -> RigidBodyHandle {
        self.rigid_body_set.insert(body)
    }

    /// Insert a collider attached to a parent rigid body. Returns its handle.
    pub fn add_collider(
        &mut self,
        collider: Collider,
        parent: RigidBodyHandle,
    ) -> ColliderHandle {
        self.collider_set
            .insert_with_parent(collider, parent, &mut self.rigid_body_set)
    }

    /// Remove a rigid body and all its attached colliders from the simulation.
    ///
    /// Returns the removed `RigidBody` if it existed.
    pub fn remove_rigid_body(&mut self, handle: RigidBodyHandle) -> Option<RigidBody> {
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true, // remove attached colliders
        )
    }

    /// Get a reference to a rigid body by handle.
    pub fn get_body(&self, handle: RigidBodyHandle) -> Option<&RigidBody> {
        self.rigid_body_set.get(handle)
    }

    /// Get a mutable reference to a rigid body by handle.
    pub fn get_body_mut(&mut self, handle: RigidBodyHandle) -> Option<&mut RigidBody> {
        self.rigid_body_set.get_mut(handle)
    }

    /// Number of rigid bodies in the simulation.
    pub fn body_count(&self) -> usize {
        self.rigid_body_set.len()
    }

    /// Number of colliders in the simulation.
    pub fn collider_count(&self) -> usize {
        self.collider_set.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation() {
        let world = PhysicsWorld::new(PhysicsConfig::default());
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.collider_count(), 0);
        assert_eq!(world.config.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert!((world.config.timestep - 1.0 / 60.0).abs() < 1e-6);
        assert_eq!(world.config.max_substeps, 4);
    }

    #[test]
    fn test_step_fixed_timestep() {
        let config = PhysicsConfig {
            timestep: 1.0 / 60.0,
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        // dt = 0.1s at 60Hz timestep (1/60 ~= 0.01667s) → 0.1 / 0.01667 = ~6 substeps
        // Add a body so we can observe the world actually stepping
        let body = RigidBodyBuilder::dynamic()
            .translation(to_rapier_vec3(Vec3::new(0.0, 10.0, 0.0)))
            .build();
        let handle = world.add_rigid_body(body);
        let collider = ColliderBuilder::ball(0.5).build();
        world.add_collider(collider, handle);

        let alpha = world.step(0.1);

        // With timestep=1/60, dt=0.1 → 6 substeps, remainder ~= 0.1 - 6*(1/60) = 0.0
        // Alpha should be small (close to 0)
        assert!(alpha >= 0.0 && alpha < 1.0, "alpha={alpha}");

        // Body should have moved down due to gravity
        let body = world.get_body(handle).unwrap();
        let pos = from_rapier_vec3(body.translation());
        assert!(
            pos.y < 10.0,
            "body should have fallen: y={}",
            pos.y
        );
    }

    #[test]
    fn test_add_remove_body() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Add a body
        let body = RigidBodyBuilder::dynamic()
            .translation(to_rapier_vec3(Vec3::new(1.0, 2.0, 3.0)))
            .build();
        let handle = world.add_rigid_body(body);
        assert_eq!(world.body_count(), 1);
        assert!(world.get_body(handle).is_some());

        // Add a collider to it
        let collider = ColliderBuilder::ball(1.0).build();
        let _col_handle = world.add_collider(collider, handle);
        assert_eq!(world.collider_count(), 1);

        // Remove the body (should also remove collider)
        let removed = world.remove_rigid_body(handle);
        assert!(removed.is_some());
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.collider_count(), 0);
        assert!(world.get_body(handle).is_none());
    }

    #[test]
    fn test_gravity_applies() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let body = RigidBodyBuilder::dynamic()
            .translation(to_rapier_vec3(Vec3::new(0.0, 100.0, 0.0)))
            .build();
        let handle = world.add_rigid_body(body);
        let collider = ColliderBuilder::ball(0.5).build();
        world.add_collider(collider, handle);

        let initial_y = from_rapier_vec3(world.get_body(handle).unwrap().translation()).y;

        // Step for a significant amount of time
        world.step(0.1);

        let final_y = from_rapier_vec3(world.get_body(handle).unwrap().translation()).y;
        assert!(
            final_y < initial_y,
            "body should fall: initial_y={initial_y}, final_y={final_y}"
        );
    }

    #[test]
    fn test_interpolation_factor() {
        let config = PhysicsConfig {
            timestep: 1.0 / 60.0, // ~0.01667s
            max_substeps: 100,     // high limit so we don't cap
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        // Step with exactly one timestep → alpha should be ~0
        let alpha = world.step(1.0 / 60.0);
        assert!(
            alpha.abs() < 0.01,
            "exact timestep should give alpha ~0, got {alpha}"
        );

        // Step with 1.5 timesteps → 1 substep consumed, 0.5 remaining → alpha ~0.5
        let alpha = world.step(1.5 / 60.0);
        assert!(
            (alpha - 0.5).abs() < 0.1,
            "1.5 timesteps should give alpha ~0.5, got {alpha}"
        );
    }

    // ------ Conversion helpers ------

    #[test]
    fn test_vec3_roundtrip() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rv = to_rapier_vec3(v);
        let back = from_rapier_vec3(rv);
        assert!((v - back).length() < 1e-6);
    }

    #[test]
    fn test_quat_roundtrip() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let rq = to_rapier_quat(q);
        let back = from_rapier_quat(&rq);
        // Compare quaternion components (may differ by sign)
        let dot = q.x * back.x + q.y * back.y + q.z * back.z + q.w * back.w;
        assert!(
            dot.abs() > 0.999,
            "quaternion roundtrip failed: original={q:?}, back={back:?}"
        );
    }
}
