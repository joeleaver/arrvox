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
    use rkp_physics::rigid_body::ColliderShape;

    eprintln!("[build_collider] cache.shape={:?} aabb_half={:?} local_center={:?} grid_origin={:?} voxel_coords={} scale={:?}",
        cache.shape, cache.aabb_half, cache.local_center, cache.grid_origin, cache.voxel_coords.len(), scale);

    // Box / Sphere / Capsule live at `local_center` in entity-local space —
    // the tight occupied-AABB midpoint, not the entity origin.
    let center_offset = to_rapier_vec3(cache.local_center);
    let pose = rapier3d::math::Pose::new(center_offset, Default::default());

    match cache.shape {
        ColliderShape::Box => {
            ColliderBuilder::cuboid(cache.aabb_half.x, cache.aabb_half.y, cache.aabb_half.z)
                .position(pose)
                .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
        }
        ColliderShape::Sphere => {
            let r = cache.aabb_half.max_element();
            ColliderBuilder::ball(r)
                .position(pose)
                .friction(rb.friction).restitution(rb.restitution).mass(rb.mass).build()
        }
        ColliderShape::Capsule => {
            let r = cache.aabb_half.x.max(cache.aabb_half.z).max(0.01);
            let hh = (cache.aabb_half.y - r).max(0.01);
            ColliderBuilder::capsule_y(hh, r)
                .position(pose)
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
/// Walks the octree directly so it sees all four occupied terminator
/// kinds — fine LEAF nodes, mip LEAF nodes (at coarser depths),
/// `INTERIOR_NODE` uniform-solid branches, and BRICK nodes (whose 4³
/// cells need [`BrickPool`] access to enumerate). Each occupied fine
/// voxel buckets into a coarse grid with cell size `collider_cell_size`.
///
/// Returns occupied coarse cell coordinates and the actual cell size
/// used. The local-space position of coord `c` is `c * collider_cell_size`,
/// offset by the `ColliderCache.grid_origin` the caller computes.
pub fn build_coarse_collider(
    octree_data: &[u32],
    brick_pool: &rkp_core::BrickPool,
    root_offset: usize,
    tree_depth: u8,
    len: u32,
    base_voxel_size: f32,
    collider_cell_size: f32,
) -> (Vec<IVec3>, f32) {
    let end = root_offset + len as usize;
    if end > octree_data.len() || collider_cell_size <= 0.0 || len == 0 {
        return (Vec::new(), collider_cell_size);
    }

    // Coarse-cell width measured in finest-voxel units. Stays as f32 because
    // collider_cell_size doesn't have to be a multiple of base_voxel_size.
    let ratio = (collider_cell_size / base_voxel_size).max(1e-6);

    let local_slice = &octree_data[root_offset..end];
    let mut occupied = std::collections::HashSet::new();

    fn coarse_of(fine: u32, ratio: f32) -> i32 {
        (fine as f32 / ratio).floor() as i32
    }

    // Insert every coarse bucket touched by a solid fine-voxel cube of side
    // `extent` rooted at `origin` (units: finest voxels).
    fn insert_solid(
        origin: glam::UVec3,
        extent: u32,
        ratio: f32,
        out: &mut std::collections::HashSet<IVec3>,
    ) {
        let lo = IVec3::new(
            coarse_of(origin.x, ratio),
            coarse_of(origin.y, ratio),
            coarse_of(origin.z, ratio),
        );
        let hi = IVec3::new(
            coarse_of(origin.x + extent - 1, ratio),
            coarse_of(origin.y + extent - 1, ratio),
            coarse_of(origin.z + extent - 1, ratio),
        );
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    out.insert(IVec3::new(x, y, z));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        nodes: &[u32],
        root_offset: u32,
        depth: u8,
        brick_pool: &rkp_core::BrickPool,
        node_idx: usize,
        origin: glam::UVec3,
        level: u8,
        ratio: f32,
        out: &mut std::collections::HashSet<IVec3>,
    ) {
        use rkp_core::sparse_octree::{
            brick_id, is_brick, is_leaf, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::{BRICK_DIM, BRICK_EMPTY};

        let node = nodes[node_idx];
        if node == EMPTY_NODE {
            return;
        }
        // Both INTERIOR_NODE branches and any LEAF are fully-solid terminators
        // covering 2^(depth-level) fine voxels per axis.
        if node == INTERIOR_NODE || is_leaf(node) {
            let extent = 1u32 << (depth - level);
            insert_solid(origin, extent, ratio, out);
            return;
        }
        if is_brick(node) {
            // Bricks always live at the fixed brick depth — they cover
            // exactly BRICK_DIM fine voxels per axis. We have to look at
            // every cell because BRICK_EMPTY can be interleaved with
            // surface leaf_attr_ids and BRICK_INTERIOR sentinels.
            let bid = brick_id(node);
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell = brick_pool.get_cell(bid, cx, cy, cz);
                        if cell == BRICK_EMPTY {
                            continue;
                        }
                        let fx = origin.x + cx;
                        let fy = origin.y + cy;
                        let fz = origin.z + cz;
                        out.insert(IVec3::new(
                            coarse_of(fx, ratio),
                            coarse_of(fy, ratio),
                            coarse_of(fz, ratio),
                        ));
                    }
                }
            }
            return;
        }
        // Branch — node value is the absolute children offset into the
        // scene-wide buffer. Subtract root_offset to reach local indexing.
        let children_local = (node - root_offset) as usize;
        let half = 1u32 << (depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = glam::UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            walk(
                nodes,
                root_offset,
                depth,
                brick_pool,
                children_local + octant as usize,
                child_origin,
                level + 1,
                ratio,
                out,
            );
        }
    }

    walk(
        local_slice,
        root_offset as u32,
        tree_depth,
        brick_pool,
        0,
        glam::UVec3::ZERO,
        0,
        ratio,
        &mut occupied,
    );

    let mut coords: Vec<IVec3> = occupied.into_iter().collect();
    coords.sort_by_key(|c| (c.z, c.y, c.x));
    (coords, collider_cell_size)
}

/// Walk the same octree+bricks `build_coarse_collider` walks and compute the
/// AABB (in entity-local space) of the **actually occupied** fine voxels.
///
/// This is what Box/Sphere/Capsule colliders should size against. The
/// `SpatialData.aabb` field is the input geometry's AABB padded by ~14 voxels
/// for boundary-sampling purposes — fine for the renderer, but it bloats
/// fitted shapes by 30-60% on small procedural primitives.
///
/// Returns `None` if no voxels are occupied. `grid_origin` is the
/// entity-local corner of the octree (`SpatialData.grid_origin`); the
/// returned AABB is in the same frame.
pub fn compute_tight_local_aabb(
    octree_data: &[u32],
    brick_pool: &rkp_core::BrickPool,
    root_offset: usize,
    tree_depth: u8,
    len: u32,
    base_voxel_size: f32,
    grid_origin: glam::Vec3,
) -> Option<rkp_core::Aabb> {
    let end = root_offset + len as usize;
    if end > octree_data.len() || len == 0 {
        return None;
    }

    let local_slice = &octree_data[root_offset..end];
    let mut min_v = glam::UVec3::new(u32::MAX, u32::MAX, u32::MAX);
    let mut max_v = glam::UVec3::ZERO; // exclusive upper (one past the last voxel)
    let mut any = false;

    fn extend(
        min_v: &mut glam::UVec3,
        max_v: &mut glam::UVec3,
        any: &mut bool,
        origin: glam::UVec3,
        extent: u32,
    ) {
        let hi = glam::UVec3::new(
            origin.x + extent,
            origin.y + extent,
            origin.z + extent,
        );
        if !*any {
            *min_v = origin;
            *max_v = hi;
            *any = true;
        } else {
            min_v.x = min_v.x.min(origin.x);
            min_v.y = min_v.y.min(origin.y);
            min_v.z = min_v.z.min(origin.z);
            max_v.x = max_v.x.max(hi.x);
            max_v.y = max_v.y.max(hi.y);
            max_v.z = max_v.z.max(hi.z);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        nodes: &[u32],
        root_offset: u32,
        depth: u8,
        brick_pool: &rkp_core::BrickPool,
        node_idx: usize,
        origin: glam::UVec3,
        level: u8,
        min_v: &mut glam::UVec3,
        max_v: &mut glam::UVec3,
        any: &mut bool,
    ) {
        use rkp_core::sparse_octree::{
            brick_id, is_brick, is_leaf, EMPTY_NODE, INTERIOR_NODE,
        };
        use rkp_core::{BRICK_DIM, BRICK_EMPTY};

        let node = nodes[node_idx];
        if node == EMPTY_NODE {
            return;
        }
        if node == INTERIOR_NODE || is_leaf(node) {
            let extent = 1u32 << (depth - level);
            extend(min_v, max_v, any, origin, extent);
            return;
        }
        if is_brick(node) {
            // Tightest possible bound: scan the cells, only count occupied.
            let bid = brick_id(node);
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell = brick_pool.get_cell(bid, cx, cy, cz);
                        if cell == BRICK_EMPTY {
                            continue;
                        }
                        extend(
                            min_v, max_v, any,
                            glam::UVec3::new(origin.x + cx, origin.y + cy, origin.z + cz),
                            1,
                        );
                    }
                }
            }
            return;
        }
        let children_local = (node - root_offset) as usize;
        let half = 1u32 << (depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = glam::UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            walk(
                nodes, root_offset, depth, brick_pool,
                children_local + octant as usize,
                child_origin, level + 1, min_v, max_v, any,
            );
        }
    }

    walk(
        local_slice, root_offset as u32, tree_depth, brick_pool,
        0, glam::UVec3::ZERO, 0, &mut min_v, &mut max_v, &mut any,
    );

    if !any {
        return None;
    }

    // fine-voxel grid -> entity-local space.
    let lo = grid_origin + glam::Vec3::new(
        min_v.x as f32 * base_voxel_size,
        min_v.y as f32 * base_voxel_size,
        min_v.z as f32 * base_voxel_size,
    );
    let hi = grid_origin + glam::Vec3::new(
        max_v.x as f32 * base_voxel_size,
        max_v.y as f32 * base_voxel_size,
        max_v.z as f32 * base_voxel_size,
    );
    Some(rkp_core::Aabb { min: lo, max: hi })
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

#[cfg(test)]
mod collider_tests {
    use super::*;
    use rkp_core::sparse_octree::{make_brick, make_leaf, INTERIOR_NODE};
    use rkp_core::{BrickPool, BRICK_DIM, BRICK_INTERIOR};

    // base_voxel_size = 1.0 throughout these tests so coarse-cell math is
    // visible by eye: ratio = collider_cell_size / 1.0 = collider_cell_size.

    #[test]
    fn interior_node_root_fills_full_extent() {
        // Depth-2 tree (4³ fine voxels), root collapsed to INTERIOR.
        // Coarse cell = 2 fine voxels, so 2³ = 8 coarse cells expected.
        let nodes = vec![INTERIOR_NODE];
        let pool = BrickPool::new(0);
        let (coords, _) = build_coarse_collider(&nodes, &pool, 0, 2, 1, 1.0, 2.0);
        assert_eq!(coords.len(), 8);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    assert!(
                        coords.contains(&IVec3::new(x, y, z)),
                        "missing coarse cell ({x},{y},{z})"
                    );
                }
            }
        }
    }

    #[test]
    fn mip_leaf_root_fills_full_extent() {
        // Same shape as the INTERIOR test but with a coarse LEAF at the root.
        let nodes = vec![make_leaf(0)];
        let pool = BrickPool::new(0);
        let (coords, _) = build_coarse_collider(&nodes, &pool, 0, 2, 1, 1.0, 2.0);
        assert_eq!(coords.len(), 8);
    }

    #[test]
    fn brick_cells_are_walked_not_dropped() {
        // Depth-2 tree (4³ = brick depth), root = single brick.
        // Populate a sparse pattern in the brick: a leaf at (0,0,0),
        // BRICK_INTERIOR at (3,3,3), and a leaf at (1,2,3). Everything
        // else stays BRICK_EMPTY.
        //
        // With coarse cell = 2 fine voxels, we expect three coarse buckets:
        //   (0,0,0) ← from fine (0,0,0)
        //   (1,1,1) ← from fine (3,3,3)
        //   (0,1,1) ← from fine (1,2,3)
        let mut pool = BrickPool::new(4);
        let bid = pool.allocate().expect("brick alloc");
        assert_eq!(bid, 0);
        pool.set_cell(bid, 0, 0, 0, 0); // leaf_attr 0 (occupied)
        pool.set_cell(bid, 3, 3, 3, BRICK_INTERIOR);
        pool.set_cell(bid, 1, 2, 3, 7);

        let nodes = vec![make_brick(bid)];
        let (coords, _) = build_coarse_collider(&nodes, &pool, 0, 2, 1, 1.0, 2.0);
        assert_eq!(
            coords.len(), 3,
            "got {coords:?}; pre-fix this returned 0 because iter_leaves skipped bricks",
        );
        assert!(coords.contains(&IVec3::new(0, 0, 0)));
        assert!(coords.contains(&IVec3::new(1, 1, 1)));
        assert!(coords.contains(&IVec3::new(0, 1, 1)));
    }

    #[test]
    fn empty_brick_contributes_no_coords() {
        // All-empty brick (default state after allocate) must emit nothing.
        let mut pool = BrickPool::new(4);
        let bid = pool.allocate().unwrap();
        let nodes = vec![make_brick(bid)];
        let (coords, _) = build_coarse_collider(&nodes, &pool, 0, 2, 1, 1.0, 2.0);
        assert!(coords.is_empty());
    }

    #[test]
    fn root_offset_is_subtracted_from_branch_children() {
        // Branch at root with a non-zero root_offset. We hand-pack a tiny
        // depth-3 tree (8³ fine voxels): 1 branch + 8 children. Bricks live
        // at depth 3 - BRICK_LEVELS(=2) = 1, so each branch child either
        // collapses to a sentinel or is a brick covering BRICK_DIM=4 voxels
        // per axis. We splice it into a larger buffer to give root_offset a
        // non-zero value — the engine always passes the full scene-wide
        // buffer with the entity's subtree at some interior offset.
        //
        // Without the `node - root_offset` correction, the branch's child
        // offset would dereference the wrong slot.
        use rkp_core::sparse_octree::EMPTY_NODE;
        let prefix = 17usize;
        let branch_target = (prefix + 1) as u32; // children start right after branch

        let mut nodes = vec![EMPTY_NODE; prefix];
        nodes.push(branch_target); // root branch at buffer index `prefix`
        // 8 children, only octant 0 (origin (0,0,0)) carries a brick.
        let mut pool = BrickPool::new(4);
        let bid = pool.allocate().unwrap();
        pool.set_cell(bid, 0, 0, 0, 0); // single occupied cell at (0,0,0)
        nodes.push(make_brick(bid));
        for _ in 1..8 {
            nodes.push(EMPTY_NODE);
        }

        let len = 1 + 8;
        // coarse cell = 2 fine voxels. Brick cell (0,0,0) → coarse (0,0,0).
        let (coords, _) = build_coarse_collider(&nodes, &pool, prefix, 3, len, 1.0, 2.0);
        let _ = BRICK_DIM;
        assert_eq!(coords, vec![IVec3::new(0, 0, 0)]);
    }

    // ── tight AABB walker ────────────────────────────────────────────────

    #[test]
    fn tight_aabb_skips_padding() {
        // Depth-3 tree (8³ voxels). Only the (0,0,0) octant carries
        // geometry — a brick whose only occupied cell is (1,1,1) inside it.
        // Total occupied region in finest-voxel units: a single 1³ voxel
        // at fine coord (1,1,1).
        //
        // base_voxel_size = 0.5 → that voxel spans [0.5, 1.0]³ in
        // grid-local space. With grid_origin = ZERO, that's the result.
        //
        // Crucially: this proves the tight AABB ignores the rest of the
        // tree's empty volume. The padded SpatialData.aabb would cover
        // the whole 4 units of extent.
        use rkp_core::sparse_octree::EMPTY_NODE;
        let mut pool = BrickPool::new(4);
        let bid = pool.allocate().unwrap();
        pool.set_cell(bid, 1, 1, 1, 0); // single occupied cell

        let mut nodes = vec![1u32]; // branch points at children starting at index 1
        nodes.push(make_brick(bid));
        for _ in 1..8 {
            nodes.push(EMPTY_NODE);
        }

        let aabb = compute_tight_local_aabb(
            &nodes, &pool, 0, 3, nodes.len() as u32, 0.5, glam::Vec3::ZERO,
        )
        .expect("should find occupied region");
        assert_eq!(aabb.min, glam::Vec3::new(0.5, 0.5, 0.5));
        assert_eq!(aabb.max, glam::Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn tight_aabb_offsets_by_grid_origin() {
        // INTERIOR_NODE root at depth=2 → fully solid 4³ region in fine
        // voxels. base_voxel_size = 0.25 → 1m total extent. With
        // grid_origin = (-0.5, -0.5, -0.5), the AABB is centered on origin.
        use rkp_core::sparse_octree::INTERIOR_NODE;
        let nodes = vec![INTERIOR_NODE];
        let pool = BrickPool::new(0);
        let aabb = compute_tight_local_aabb(
            &nodes, &pool, 0, 2, 1, 0.25, glam::Vec3::splat(-0.5),
        )
        .unwrap();
        assert_eq!(aabb.min, glam::Vec3::splat(-0.5));
        assert_eq!(aabb.max, glam::Vec3::splat(0.5));
    }

    #[test]
    fn tight_aabb_returns_none_for_empty() {
        use rkp_core::sparse_octree::EMPTY_NODE;
        let nodes = vec![EMPTY_NODE];
        let pool = BrickPool::new(0);
        let aabb = compute_tight_local_aabb(
            &nodes, &pool, 0, 2, 1, 1.0, glam::Vec3::ZERO,
        );
        assert!(aabb.is_none());
    }
}
