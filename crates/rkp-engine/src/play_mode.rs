//! Play mode — physics stepping and transform synchronization.
//!
//! When play mode starts, a PhysicsWorld is created and rigid bodies are spawned
//! for every entity with a RigidBody component. Each tick, Rapier is stepped and
//! entity transforms are synced back. When play stops, transforms are restored.

use std::collections::HashMap;

use glam::{IVec3, Vec3};
use rapier3d::prelude::*;
use rkp_physics::rapier_world::{PhysicsConfig, PhysicsWorld, to_rapier_vec3, from_rapier_vec3, from_rapier_quat};
use rkp_physics::rigid_body::BodyType;

use crate::components::{RigidBody, RigidBodyRuntime, Transform};

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

    /// Step physics and sync transforms back to the ECS world.
    pub fn step(&mut self, dt: f32, world: &mut hecs::World) -> bool {
        self.physics.step(dt);
        self.frame_count += 1;

        // After 2 seconds, dump all body positions and collider AABBs.
        if self.frame_count == 120 || self.frame_count == 10 {
            eprintln!("[Physics] === State at frame {} ===", self.frame_count);
            for (entity, runtime) in &self.body_map {
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

fn build_collider_from_cache(
    rb: &RigidBody,
    cache: &crate::components::ColliderCache,
    scale: Vec3,
) -> Collider {
    use rkp_physics::rapier_world::to_rapier_vec3 as to_rv;
    use rkp_physics::rigid_body::ColliderShape;

    eprintln!("[build_collider] cache.shape={:?} aabb_half={:?} grid_origin={:?} voxel_coords={} scale={:?}",
        cache.shape, cache.aabb_half, cache.grid_origin, cache.voxel_coords.len(), scale);

    match cache.shape {
        ColliderShape::Box => {
            ColliderBuilder::cuboid(cache.aabb_half.x, cache.aabb_half.y, cache.aabb_half.z)
                .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
        }
        ColliderShape::Sphere => {
            let r = cache.aabb_half.max_element();
            ColliderBuilder::ball(r)
                .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
        }
        ColliderShape::Capsule => {
            let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
            let hh = (cache.aabb_half.y - r).max(0.01);
            ColliderBuilder::capsule_y(hh, r)
                .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
        }
        ColliderShape::Auto => {
            if !cache.voxel_coords.is_empty() {
                let cs = cache.collider_cell_size;
                let cell_size = to_rapier_vec3(Vec3::splat(cs) * scale);
                let rapier_coords: Vec<_> = cache.voxel_coords.iter()
                    .map(|c| rapier3d::math::IVector::new(c.x, c.y, c.z))
                    .collect();
                let density = if cs > 0.0 {
                    rb.mass / (cache.voxel_coords.len() as f32 * cs * cs * cs)
                } else {
                    1.0
                };
                // Position offset: grid_origin maps coarse grid space to the
                // octree's local space, which the renderer uses for placement.
                let offset = to_rapier_vec3(cache.grid_origin * scale);
                ColliderBuilder::voxels(cell_size, &rapier_coords)
                    .position(rapier3d::math::Pose::new(offset, Default::default()))
                    .friction(rb.friction).restitution(rb.restitution).density(density).build()
            } else {
                ColliderBuilder::cuboid(cache.aabb_half.x, cache.aabb_half.y, cache.aabb_half.z)
                    .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
            }
        }
    }
}

/// Build a coarse voxel collider grid from an entity's fine octree.
///
/// Reconstructs the octree, iterates all occupied fine voxels, and buckets
/// them into a coarse grid with cell size `collider_cell_size`. Returns
/// the occupied coarse cell coordinates (in collider-grid units) and the
/// actual cell size used.
///
/// The returned coordinates are in coarse grid units — to get local-space
/// position: `grid_origin + coord * collider_cell_size`.
pub fn build_coarse_collider(
    octree_data: &[u32],
    root_offset: usize,
    tree_depth: u8,
    len: u32,
    base_voxel_size: f32,
    collider_cell_size: f32,
) -> (Vec<IVec3>, f32) {
    let end = root_offset + len as usize;
    if end > octree_data.len() || collider_cell_size <= 0.0 {
        return (Vec::new(), collider_cell_size);
    }

    // Reconstruct the 0-based octree from the packed buffer.
    let slice = &octree_data[root_offset..end];
    let mut local_nodes = slice.to_vec();
    for node in &mut local_nodes {
        let v = *node;
        if v == rkp_core::sparse_octree::EMPTY_NODE
            || v == rkp_core::sparse_octree::INTERIOR_NODE
        { continue; }
        if !rkp_core::sparse_octree::is_leaf(v) {
            *node = v - root_offset as u32;
        }
    }
    let octree = rkp_core::SparseOctree::from_raw(&local_nodes, tree_depth, base_voxel_size);

    // How many fine voxels fit in one coarse cell?
    let ratio = (collider_cell_size / base_voxel_size).max(1.0);

    // Every leaf in the tree is surface now — no opacity threshold to check.
    let mut occupied = std::collections::HashSet::new();
    for (coord, _leaf_id, _depth) in octree.iter_leaves() {
        let coarse = IVec3::new(
            (coord.x as f32 / ratio).floor() as i32,
            (coord.y as f32 / ratio).floor() as i32,
            (coord.z as f32 / ratio).floor() as i32,
        );
        occupied.insert(coarse);
    }

    let mut coords: Vec<IVec3> = occupied.into_iter().collect();
    coords.sort_by_key(|c| (c.z, c.y, c.x));
    // Coords are in coarse grid units. The local-space position of voxel (c) is:
    //   c * collider_cell_size
    // The grid_origin offset (stored in ColliderCache) maps these to the octree's
    // local space. The collider is attached to the Rapier body with a position offset
    // of grid_origin so the voxels align with the rendered geometry.
    (coords, collider_cell_size)
}

// ── Helpers ──────────────────────────────────────────────────────────────

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
