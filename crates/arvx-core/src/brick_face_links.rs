//! Face-adjacency links between bricks — an acceleration structure
//! that lets shaders read a brick's neighbors in one indirect lookup
//! instead of a full octree descent.
//!
//! # Layout
//!
//! `Vec<[u32; 6]>` indexed by `brick_id`. Each entry is the 6 face-
//! neighbors' brick ids (or sentinels) in the order:
//!
//! | Index | Direction |
//! |-------|-----------|
//! | 0     | −X        |
//! | 1     | +X        |
//! | 2     | −Y        |
//! | 3     | +Y        |
//! | 4     | −Z        |
//! | 5     | +Z        |
//!
//! Values:
//! - `0..FACE_INTERIOR`: the neighbor in that direction is a brick with
//!   that `brick_id`.
//! - [`FACE_INTERIOR`]: the neighbor is a fully-solid bulk region
//!   (collapsed to `INTERIOR_NODE` somewhere up-tree — callers that
//!   want to know "is neighbor occupied?" should treat this as yes).
//! - [`FACE_EMPTY`]: the neighbor is empty air, out of bounds, or
//!   stored in a way the link table doesn't represent (e.g. a LEAF
//!   at brick_depth from a subtree collapse). Callers treat as no
//!   occupancy and fall back to their normal handling.
//!
//! # Usage
//!
//! Chain multiple face-link reads to reach edge- or corner-adjacent
//! bricks. E.g., the `(−X, −Y)` brick diagonal is two face hops:
//! `face_links[links[B][FACE_NX]][FACE_NY]`. This lets 6 face links
//! per brick cover all 26 spatial neighbors without storing an
//! edge/corner table — at the cost of 2-3 indirect reads for
//! corners/edges. The storage (6 × u32 × brick_count) is typically a
//! few MiB on real assets.
//!
//! # Computation
//!
//! [`compute_brick_face_links`] walks the octree, collects each brick's
//! brick-level coord (derived from its position during traversal),
//! then looks up each of 6 neighbor coords via
//! [`SparseOctree::lookup_with_depth`]. The resulting link is
//! determined by the returned node type.

use glam::UVec3;

use crate::brick_pool::BRICK_LEVELS;
use crate::sparse_octree::{
    brick_id as brick_id_of, is_brick, SparseOctree, EMPTY_NODE, INTERIOR_NODE,
};

/// Sentinel: neighbor is a fully-opaque bulk region.
pub const FACE_INTERIOR: u32 = 0xFFFF_FFFE;

/// Sentinel: neighbor is empty space / out of bounds / unrepresented.
pub const FACE_EMPTY: u32 = 0xFFFF_FFFF;

/// Face index constants. Match the shader's FACE_* constants.
pub const FACE_NX: usize = 0;
pub const FACE_PX: usize = 1;
pub const FACE_NY: usize = 2;
pub const FACE_PY: usize = 3;
pub const FACE_NZ: usize = 4;
pub const FACE_PZ: usize = 5;

/// 6 face deltas in the conventional order `(−X, +X, −Y, +Y, −Z, +Z)`.
const FACE_DELTAS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Flat byte view of the link table, for GPU upload.
pub fn as_bytes(links: &[[u32; 6]]) -> &[u8] {
    bytemuck::cast_slice(links)
}

/// Build the face-link table for every brick referenced by `octree`.
///
/// `max_brick_id` is the highest brick id in use (one past the last
/// allocated brick id in the `BrickPool`). The returned vector has
/// length `max_brick_id + 1`; entries at unreferenced ids are filled
/// with all-sentinel rows (safe but uninformative — a brick whose id
/// isn't referenced by the octree will never be shaded, so nothing
/// reads those rows).
///
/// If `octree.depth() < BRICK_LEVELS` the tree is too shallow to
/// contain bricks; returns an empty vector.
pub fn compute_brick_face_links(octree: &SparseOctree, max_brick_id: u32) -> Vec<[u32; 6]> {
    let depth = octree.depth();
    if depth < BRICK_LEVELS {
        return Vec::new();
    }
    let brick_depth = depth - BRICK_LEVELS;
    let cells_per_brick: u32 = 1u32 << BRICK_LEVELS;
    let bricks_per_axis: u32 = 1u32 << brick_depth;

    // Pass 1: gather each brick's minimum-corner brick-level coord by
    // walking the tree. BRICK nodes at arbitrary levels (not just
    // brick_depth — the octree doesn't promote brick_depth to a hard
    // invariant, but in practice voxelize places bricks exactly there)
    // are all recorded; their brick_coord is derived from the
    // voxel_coord they sit at.
    let cap = (max_brick_id as usize) + 1;
    let mut brick_coord: Vec<UVec3> = vec![UVec3::ZERO; cap];
    let mut found: Vec<bool> = vec![false; cap];
    collect_brick_positions(
        octree,
        0,
        UVec3::ZERO,
        0,
        depth,
        &mut brick_coord,
        &mut found,
    );

    // Pass 2: for each found brick, look up each of 6 neighbor bricks.
    let mut links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]; cap];
    for brick_id in 0..cap {
        if !found[brick_id] {
            continue;
        }
        let my_coord = brick_coord[brick_id];
        for (face, delta) in FACE_DELTAS.iter().enumerate() {
            let nx = my_coord.x as i32 + delta.0;
            let ny = my_coord.y as i32 + delta.1;
            let nz = my_coord.z as i32 + delta.2;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= bricks_per_axis as i32
                || ny >= bricks_per_axis as i32
                || nz >= bricks_per_axis as i32
            {
                // Out of octree bounds — empty.
                links[brick_id][face] = FACE_EMPTY;
                continue;
            }
            // Neighbor's minimum-corner voxel coord. lookup_with_depth
            // returns whatever node covers that point, at whatever
            // level the tree actually terminates.
            let neighbor_voxel = UVec3::new(
                nx as u32 * cells_per_brick,
                ny as u32 * cells_per_brick,
                nz as u32 * cells_per_brick,
            );
            let (value, _d) = match octree.lookup_with_depth(neighbor_voxel) {
                Some(v) => v,
                None => {
                    links[brick_id][face] = FACE_EMPTY;
                    continue;
                }
            };
            links[brick_id][face] = classify_neighbor(value);
        }
    }
    links
}

/// Decode an octree node value into a face-link entry. LEAF and
/// BRANCH at brick-level (rare but possible via subtree collapses)
/// aren't representable as a single brick reference, so they resolve
/// to [`FACE_EMPTY`] — the shader's cross-brick path then falls back
/// to the baked normal (as if we had no neighbor information).
fn classify_neighbor(node: u32) -> u32 {
    if node == EMPTY_NODE {
        FACE_EMPTY
    } else if node == INTERIOR_NODE {
        FACE_INTERIOR
    } else if is_brick(node) {
        brick_id_of(node)
    } else {
        FACE_EMPTY
    }
}

/// Walk the octree, record the (brick_id → brick_coord) mapping for
/// every BRICK node. Tracks the caller's voxel_coord for the minimum
/// corner of the current node's spatial extent.
fn collect_brick_positions(
    octree: &SparseOctree,
    node_idx: usize,
    voxel_coord: UVec3,
    level: u8,
    depth: u8,
    brick_coord: &mut [UVec3],
    found: &mut [bool],
) {
    let node = octree.as_slice()[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }
    if is_brick(node) {
        // brick_coord in brick-level units = voxel_coord / BRICK_DIM.
        // At brick_depth, voxel_coord is an exact multiple of BRICK_DIM;
        // shallower BRICK nodes (from collapses) would misalign, but
        // voxelize doesn't produce those.
        let bx = voxel_coord.x >> BRICK_LEVELS;
        let by = voxel_coord.y >> BRICK_LEVELS;
        let bz = voxel_coord.z >> BRICK_LEVELS;
        let id = brick_id_of(node) as usize;
        if id < brick_coord.len() {
            brick_coord[id] = UVec3::new(bx, by, bz);
            found[id] = true;
        }
        return;
    }
    if crate::sparse_octree::is_leaf(node) {
        return;
    }
    // Branch — recurse into children.
    let children_offset = node as usize;
    let half_voxels: u32 = 1u32 << (depth - level - 1);
    for octant in 0u32..8 {
        let dx = octant & 1;
        let dy = (octant >> 1) & 1;
        let dz = (octant >> 2) & 1;
        let child_coord = UVec3::new(
            voxel_coord.x + dx * half_voxels,
            voxel_coord.y + dy * half_voxels,
            voxel_coord.z + dz * half_voxels,
        );
        collect_brick_positions(
            octree,
            children_offset + octant as usize,
            child_coord,
            level + 1,
            depth,
            brick_coord,
            found,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick_pool::BrickPool;
    use crate::leaf_attr_pool::LeafAttrPool;
    use crate::voxelize_octree::voxelize_sphere_octree;
    use glam::Vec3;

    #[test]
    fn sphere_face_links_point_at_real_bricks() {
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.4, 0, 0.05, &mut attrs, &mut bricks).unwrap();
        let max_brick_id = *r.brick_ids.iter().max().unwrap();
        let links = compute_brick_face_links(&r.octree, max_brick_id);

        assert_eq!(links.len(), (max_brick_id + 1) as usize);

        // Every brick that was allocated should have at least one face
        // link that ISN'T FACE_EMPTY — a surface voxel next to interior
        // or another surface brick. Fully-isolated bricks shouldn't exist
        // for a connected surface.
        for &bid in &r.brick_ids {
            let row = &links[bid as usize];
            let non_empty = row.iter().filter(|&&v| v != FACE_EMPTY).count();
            assert!(
                non_empty > 0,
                "brick {} has no non-empty face links, row = {:?}",
                bid,
                row,
            );
        }
    }

    #[test]
    fn face_links_are_symmetric_for_brick_pairs() {
        // If A says its −X neighbor is B, then B should say its +X
        // neighbor is A. Verifies the walker tags bricks with consistent
        // coordinates and the lookup path is correct.
        let mut attrs = LeafAttrPool::new(1_000_000);
        let mut bricks = BrickPool::new(10_000);
        let r = voxelize_sphere_octree(Vec3::ZERO, 0.3, 0, 0.05, &mut attrs, &mut bricks).unwrap();
        let max_brick_id = *r.brick_ids.iter().max().unwrap();
        let links = compute_brick_face_links(&r.octree, max_brick_id);

        const OPPOSITE: [usize; 6] = [
            FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ,
        ];

        let mut checked = 0usize;
        for &bid in &r.brick_ids {
            let row = &links[bid as usize];
            for face in 0..6 {
                let other = row[face];
                if other >= FACE_INTERIOR {
                    continue;
                }
                assert!(
                    (other as usize) < links.len(),
                    "face link {other} out of range",
                );
                let back = links[other as usize][OPPOSITE[face]];
                assert_eq!(
                    back, bid,
                    "brick {} face {} → brick {} but reverse says {} (expected {})",
                    bid, face, other, back, bid,
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "test must check at least one brick-brick link");
    }
}
