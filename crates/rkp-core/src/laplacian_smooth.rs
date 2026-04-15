//! Bake-time Laplacian smoothing of shell-voxel normals.
//!
//! For each shell voxel (an occupied brick cell carrying a `LeafAttr`),
//! replace its stored normal with the normalized average of its own
//! normal plus the normals of its 6 face-neighbor shell voxels. Iterate
//! N times. Non-shell neighbors (empty, interior bulk, out of bounds)
//! simply don't contribute.
//!
//! Converts render-time binary-centroid reconstruction noise into
//! bake-time smoothing. Produces the per-voxel normal as a few
//! iterations of Laplacian relaxation — equivalent to the limit
//! behavior of naive-surface-nets vertex smoothing, applied to the
//! normal field rather than the position field.
//!
//! # Requirements
//!
//! Each shell voxel must have its own `leaf_attr_id` (1:1, no dedup
//! across voxels) — otherwise smoothing one voxel affects every
//! voxel that shares the id. The mesh import path
//! (`voxelize_opacity`) already allocates 1:1 because per-voxel
//! textures vary. The procedural path dedups; don't call this on
//! procedural trees without first un-deduplicating.
//!
//! # Cost
//!
//! O(shell_count × iterations × 6) lookups and writes. On an asset
//! of ~100k shell voxels with 3 iterations → ~1.8M lookups, runs in
//! milliseconds at bake time.
//!
//! # Non-goals
//!
//! - Doesn't touch `BRICK_INTERIOR` cells (no stored normal).
//! - Doesn't move vertex positions (full NSN would; this is normals
//!   only).

use std::collections::HashMap;

use glam::{UVec3, Vec3};

use crate::brick_pool::{BrickPool, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR, BRICK_LEVELS};
use crate::leaf_attr::{pack_oct, unpack_oct};
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::{brick_id, is_brick, SparseOctree};

/// Run `iterations` passes of Laplacian smoothing over every shell
/// voxel's stored normal. Returns the number of shell voxels smoothed.
pub fn smooth_shell_normals(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &mut LeafAttrPool,
    iterations: u32,
) -> usize {
    if iterations == 0 {
        return 0;
    }

    // Pass 1: enumerate shell voxels.
    let mut shell: Vec<(UVec3, u32)> = Vec::new();
    let depth = octree.depth();
    collect_shell_voxels(octree, brick_pool, 0, UVec3::ZERO, 0, depth, &mut shell);

    if shell.is_empty() {
        return 0;
    }

    // Pass 2: leaf_attr_id → shell-index map for O(1) neighbor lookup.
    // Requires each id to map to exactly one shell voxel (1:1 layout).
    // If we ever call this on a deduped tree we'll silently lose
    // per-voxel fidelity — debug_assert catches the bug.
    let mut id_to_index: HashMap<u32, usize> = HashMap::with_capacity(shell.len());
    for (i, (_, id)) in shell.iter().enumerate() {
        let prev = id_to_index.insert(*id, i);
        debug_assert!(
            prev.is_none(),
            "smooth_shell_normals requires 1:1 leaf_attr ids across shell voxels (id {id} shared)",
        );
    }

    // Pass 3: double-buffered iteration.
    let mut current: Vec<Vec3> = shell
        .iter()
        .map(|(_, id)| unpack_oct(leaf_attr_pool.get(*id).normal_oct))
        .collect();
    let mut next: Vec<Vec3> = current.clone();

    const FACE_OFFSETS: [(i32, i32, i32); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    let extent = octree.extent() as i64;

    for _ in 0..iterations {
        for (i, (coord, _)) in shell.iter().enumerate() {
            let mut sum = current[i]; // self-weight = 1
            for &(dx, dy, dz) in FACE_OFFSETS.iter() {
                let nx = coord.x as i64 + dx as i64;
                let ny = coord.y as i64 + dy as i64;
                let nz = coord.z as i64 + dz as i64;
                if nx < 0 || ny < 0 || nz < 0 || nx >= extent || ny >= extent || nz >= extent {
                    continue;
                }
                let nc = UVec3::new(nx as u32, ny as u32, nz as u32);
                if let Some(n) = neighbor_shell_normal(octree, brick_pool, nc, &id_to_index, &current) {
                    sum += n;
                }
            }
            let len_sq = sum.length_squared();
            next[i] = if len_sq > 1e-12 {
                sum / len_sq.sqrt()
            } else {
                current[i]
            };
        }
        std::mem::swap(&mut current, &mut next);
    }

    // Pass 4: write back smoothed normals.
    for (i, (_, id)) in shell.iter().enumerate() {
        let attr = leaf_attr_pool.get_mut(*id);
        attr.normal_oct = pack_oct(current[i]);
    }

    shell.len()
}

/// Look up the current-iteration normal for the shell voxel at
/// `coord`, if one exists. O(1) via the id_to_index map.
fn neighbor_shell_normal(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    coord: UVec3,
    id_to_index: &HashMap<u32, usize>,
    current: &[Vec3],
) -> Option<Vec3> {
    let node = octree.lookup(coord)?;
    if node == crate::sparse_octree::EMPTY_NODE
        || node == crate::sparse_octree::INTERIOR_NODE
    {
        return None;
    }
    let id = if crate::sparse_octree::is_leaf(node) {
        crate::sparse_octree::leaf_slot(node)
    } else if is_brick(node) {
        let bid = brick_id(node);
        let brick_voxels: u32 = 1 << BRICK_LEVELS;
        let cx = coord.x & (brick_voxels - 1);
        let cy = coord.y & (brick_voxels - 1);
        let cz = coord.z & (brick_voxels - 1);
        let cell = brick_pool.get_cell(bid, cx % BRICK_DIM, cy % BRICK_DIM, cz % BRICK_DIM);
        if cell == BRICK_EMPTY || cell == BRICK_INTERIOR {
            return None;
        }
        cell
    } else {
        return None;
    };
    let idx = *id_to_index.get(&id)?;
    Some(current[idx])
}

/// Walk the octree, emit every (voxel_coord, leaf_attr_id) for shell
/// cells. Shell cells have real leaf_attr_ids (not EMPTY / INTERIOR
/// sentinels).
fn collect_shell_voxels(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    node_idx: usize,
    voxel_coord: UVec3,
    level: u8,
    depth: u8,
    out: &mut Vec<(UVec3, u32)>,
) {
    let node = octree.as_slice()[node_idx];
    if node == crate::sparse_octree::EMPTY_NODE
        || node == crate::sparse_octree::INTERIOR_NODE
    {
        return;
    }
    if crate::sparse_octree::is_leaf(node) {
        let slot = crate::sparse_octree::leaf_slot(node);
        out.push((voxel_coord, slot));
        return;
    }
    if is_brick(node) {
        let bid = brick_id(node);
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let cell = brick_pool.get_cell(bid, x, y, z);
                    if cell == BRICK_EMPTY || cell == BRICK_INTERIOR {
                        continue;
                    }
                    out.push((
                        UVec3::new(voxel_coord.x + x, voxel_coord.y + y, voxel_coord.z + z),
                        cell,
                    ));
                }
            }
        }
        return;
    }
    let children_offset = node as usize;
    let half_voxels: u32 = 1 << (depth - level - 1);
    for octant in 0u32..8 {
        let dx = octant & 1;
        let dy = (octant >> 1) & 1;
        let dz = (octant >> 2) & 1;
        let child_coord = UVec3::new(
            voxel_coord.x + dx * half_voxels,
            voxel_coord.y + dy * half_voxels,
            voxel_coord.z + dz * half_voxels,
        );
        collect_shell_voxels(
            octree,
            brick_pool,
            children_offset + octant as usize,
            child_coord,
            level + 1,
            depth,
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick_pool::BrickPool;
    use crate::leaf_attr::LeafAttr;
    use crate::sparse_octree::{make_brick, SparseOctree};

    /// Build a tiny hand-constructed tree with a 4³ brick whose
    /// cells carry distinct leaf_attr_ids (1:1 mesh-style). All shell
    /// voxels start with a `(0, 0, 1)` normal; after smoothing they
    /// should still average to that (self-consistent).
    #[test]
    fn uniform_field_is_a_fixed_point() {
        let mut pool = LeafAttrPool::new(512);
        let mut bricks = BrickPool::new(4);
        let bid = bricks.allocate().unwrap();

        // Fill the brick with 1:1 leaf_attr ids, same normal everywhere.
        let normal = Vec3::Z;
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let id = pool.allocate_contiguous_bump(1).unwrap();
                    *pool.get_mut(id) = LeafAttr::new(normal, 0);
                    bricks.set_cell(bid, x, y, z, id);
                }
            }
        }

        // Tree with brick_depth=0 (tree depth = BRICK_LEVELS), so the
        // root node IS the brick. Start from_raw with that single node.
        let mut octree = SparseOctree::from_raw(&[make_brick(bid)], BRICK_LEVELS, 0.05);
        let _ = &mut octree;
        let count = smooth_shell_normals(&octree, &bricks, &mut pool, 3);

        assert_eq!(count, (BRICK_DIM * BRICK_DIM * BRICK_DIM) as usize);
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let cell = bricks.get_cell(bid, x, y, z);
                    let back = unpack_oct(pool.get(cell).normal_oct);
                    assert!(back.dot(normal) > 0.999, "got {back:?}");
                }
            }
        }
    }
}
