//! Coarse-voxel collider construction for rigid bodies, plus the
//! tight-AABB computation that drives sphere/box/capsule auto-fit.

use glam::{IVec3, Vec3};
use rapier3d::prelude::*;
use rkp_physics::rapier_world::to_rapier_vec3;

use crate::components::RigidBody;

pub(super) fn build_collider_from_cache(
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
