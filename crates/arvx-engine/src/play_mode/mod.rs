//! Play mode — physics stepping and transform synchronization.
//!
//! When play mode starts, a PhysicsWorld is created and rigid bodies are spawned
//! for every entity with a RigidBody component. Each tick, Rapier is stepped and
//! entity transforms are synced back. When play stops, transforms are restored.

use std::collections::HashMap;

use glam::Vec3;
use rapier3d::prelude::*;
use arvx_physics::rapier_world::{PhysicsConfig, PhysicsWorld, to_rapier_vec3, from_rapier_vec3, from_rapier_quat};
use arvx_physics::rigid_body::BodyType;

use crate::components::{RigidBody, RigidBodyRuntime, Transform};

mod collider;

#[cfg(test)]
mod tests;

use collider::build_collider_from_cache;
pub use collider::{build_coarse_collider, compute_tight_local_aabb};

/// Saved transform for restoring on play stop.
struct SavedTransform {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
}

/// Active play mode state.
pub struct PlayModeState {
    physics: PhysicsWorld,
    body_map: HashMap<hecs::Entity, RigidBodyRuntime>,
    saved_transforms: HashMap<hecs::Entity, SavedTransform>,
    frame_count: u32,
}

impl PlayModeState {
    /// Start play mode: create physics world, spawn bodies from cached colliders.
    pub fn start(world: &mut hecs::World) -> Self {
        let mut physics = PhysicsWorld::new(PhysicsConfig::default());

        let mut body_map = HashMap::new();
        let mut saved_transforms = HashMap::new();

        // Collect entities with RigidBody + Transform + ColliderCache.
        let entities: Vec<_> = world
            .query::<(&RigidBody, &Transform, &crate::components::ColliderCache)>()
            .iter()
            .map(|(e, (rb, t, cache))| (e, rb.clone(), t.clone(), cache.clone()))
            .collect();

        for (entity, rb, transform, cache) in &entities {
            saved_transforms.insert(*entity, SavedTransform {
                position: transform.position,
                rotation: transform.rotation,
                scale: transform.scale,
            });

            // Build collider from cache.
            let collider = build_collider_from_cache(&rb, cache, transform.scale);

            // Spawn rigid body.
            let builder = match rb.body_type {
                BodyType::Dynamic => RigidBodyBuilder::dynamic(),
                BodyType::Static => RigidBodyBuilder::fixed(),
                BodyType::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
                BodyType::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
            };

            let body = builder
                .translation(to_rapier_vec3(transform.position))
                .rotation(to_rapier_vec3(euler_to_axis_angle(transform.rotation)))
                .build();

            let handle = physics.add_rigid_body(body);
            let collider_handle = physics.add_collider(collider, handle);

            // Debug: print collider AABB as Rapier sees it.
            if let Some(c) = physics.collider_set.get(collider_handle) {
                let aabb = c.compute_aabb();
                eprintln!("[PlayMode] body at {:?}, collider AABB: min={:?} max={:?}",
                    transform.position, aabb.mins, aabb.maxs);
            }

            body_map.insert(*entity, RigidBodyRuntime { handle });
        }

        eprintln!("[PlayMode] started with {} bodies + ground plane, gravity={:?}, timestep={}",
            body_map.len(), physics.config.gravity, physics.config.timestep);

        Self { physics, body_map, saved_transforms, frame_count: 0 }
    }

    /// Substeps performed in the most recent `step()`. Surfaced so the
    /// engine's profiling readout can show "physics Hz" alongside FPS.
    pub fn last_step_substeps(&self) -> u32 {
        self.physics.last_step_substeps
    }

    /// Step physics and sync transforms back to the ECS world.
    pub fn step(&mut self, dt: f32, world: &mut hecs::World) -> bool {
        self.physics.step(dt);
        self.frame_count += 1;

        // After 2 seconds, dump all body positions and collider AABBs.
        if self.frame_count == 120 || self.frame_count == 10 {
            eprintln!("[Physics] === State at frame {} ===", self.frame_count);
            for (_entity, runtime) in &self.body_map {
                if let Some(body) = self.physics.get_body(runtime.handle) {
                    let pos = from_rapier_vec3(body.translation().clone());
                    let vel = from_rapier_vec3(body.linvel().clone());
                    let sleeping = body.is_sleeping();
                    let is_dynamic = body.is_dynamic();
                    eprintln!("[Physics] body pos={pos:?} vel={vel:?} dynamic={is_dynamic} sleeping={sleeping}");
                }
                // Also dump all colliders for this body
                for (_, collider) in self.physics.collider_set.iter() {
                    if collider.parent() == Some(runtime.handle) {
                        let aabb = collider.compute_aabb();
                        let local_pos = collider.position_wrt_parent().map(|p| from_rapier_vec3(p.translation.clone()));
                        eprintln!("[Physics] collider aabb={:?}..{:?} local_offset={:?}",
                            aabb.mins, aabb.maxs, local_pos);
                    }
                }
            }
        }

        let mut changed = false;
        for (entity, runtime) in &self.body_map {
            if let Some(body) = self.physics.get_body(runtime.handle) {
                if let Ok(mut t) = world.get::<&mut Transform>(*entity) {
                    let pos = from_rapier_vec3(body.translation().clone());
                    let rot = from_rapier_quat(body.rotation());
                    let euler = quat_to_euler_deg(rot);
                    if (t.position - pos).length_squared() > 1e-8 {
                        t.position = pos;
                        t.rotation = euler;
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// Stop play mode: restore saved transforms, drop physics.
    pub fn stop(self, world: &mut hecs::World) {
        for (entity, saved) in &self.saved_transforms {
            if let Ok(mut t) = world.get::<&mut Transform>(*entity) {
                t.position = saved.position;
                t.rotation = saved.rotation;
                t.scale = saved.scale;
            }
        }
        eprintln!("[PlayMode] stopped, restored {} transforms", self.saved_transforms.len());
    }
}

// ── Collider building from cache ──────────────────────────────────────────


fn euler_to_axis_angle(euler_deg: Vec3) -> Vec3 {
    let q = glam::Quat::from_euler(
        glam::EulerRot::XYZ,
        euler_deg.x.to_radians(),
        euler_deg.y.to_radians(),
        euler_deg.z.to_radians(),
    );
    let (axis, angle) = q.to_axis_angle();
    axis * angle
}

fn quat_to_euler_deg(q: glam::Quat) -> Vec3 {
    let (x, y, z) = q.to_euler(glam::EulerRot::XYZ);
    Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
}

