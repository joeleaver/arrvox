//! Laplacian smoothing of shell-voxel normals.
//!
//! For each shell voxel (an occupied brick cell carrying a `LeafAttr`),
//! replace its stored normal with the normalized average of its own
//! normal plus the normals of its 6 face-neighbor shell voxels. Iterate
//! N times. Non-shell neighbors (empty, interior bulk, out of bounds)
//! simply don't contribute.
//!
//! Converts render-time binary-centroid reconstruction noise into
//! pre-smoothed normals. Equivalent to the limit behavior of
//! naive-surface-nets vertex smoothing, applied to the normal field
//! rather than the position field.
//!
//! # Entry points
//!
//! * [`smooth_shell_normals_raw`] — operates on the flat buffers
//!   produced at import time (`file_bricks: &[u32]`,
//!   `normals_packed: &mut [u32]`). Called by `rkp-import` so the
//!   `.rkp` file carries pre-smoothed normals.
//! * [`smooth_shell_normals`] — operates on the runtime
//!   [`BrickPool`] / [`LeafAttrPool`] pair. Kept for completeness; the
//!   load-time smoothing pass in the scene manager has been retired in
//!   favour of the import-time variant above.
//!
//! Both entry points share a single generic implementation so the
//! algorithm lives in one place.
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
//! O(shell_count × iterations × 6) lookups and writes.
//!
//! # Non-goals
//!
//! - Doesn't touch `BRICK_INTERIOR` cells (no stored normal).
//! - Doesn't move vertex positions (full NSN would; this is normals
//!   only).

use std::collections::HashMap;

use glam::{UVec3, Vec3};

use crate::brick_pool::{BrickPool, BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR, BRICK_LEVELS};
use crate::leaf_attr::{pack_oct, unpack_oct};
use crate::leaf_attr_pool::LeafAttrPool;
use crate::sparse_octree::{brick_id, is_brick, SparseOctree};

/// Smooth shell-voxel normals using the import-time flat buffers.
///
/// `file_bricks` is the brick-cell array as it's written into the
/// `.rkp` file (`BRICK_CELLS` u32 cells per brick, contiguous).
/// `normals_packed` is the octahedrally-packed normals array — one
/// entry per shell voxel, indexed by the `slot` values stored inside
/// shell cells.
///
/// Mutates `normals_packed` in place and returns the number of shell
/// voxels smoothed.
pub fn smooth_shell_normals_raw(
    octree: &SparseOctree,
    file_bricks: &[u32],
    normals_packed: &mut [u32],
    iterations: u32,
) -> usize {
    let cells_per_brick = BRICK_CELLS as usize;
    let smoothed = {
        let get_cell = |bid: u32, x: u32, y: u32, z: u32| -> u32 {
            let base = bid as usize * cells_per_brick;
            let idx = (x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM) as usize;
            file_bricks[base + idx]
        };
        let read_initial = |slot: u32| -> u32 { normals_packed[slot as usize] };
        smooth_shell_normals_inner(octree, iterations, get_cell, read_initial)
    };
    let count = smoothed.len();
    for (slot, packed) in smoothed {
        normals_packed[slot as usize] = packed;
    }
    count
}

/// Smooth shell-voxel normals using the runtime pool types.
/// Provided for API completeness; prefer [`smooth_shell_normals_raw`]
/// for import-time work so the cost doesn't repeat each asset load.
pub fn smooth_shell_normals(
    octree: &SparseOctree,
    brick_pool: &BrickPool,
    leaf_attr_pool: &mut LeafAttrPool,
    iterations: u32,
) -> usize {
    let smoothed = {
        let get_cell = |bid, x, y, z| brick_pool.get_cell(bid, x, y, z);
        let read_initial = |slot: u32| leaf_attr_pool.get(slot).normal_oct;
        smooth_shell_normals_inner(octree, iterations, get_cell, read_initial)
    };
    let count = smoothed.len();
    for (slot, packed) in smoothed {
        leaf_attr_pool.get_mut(slot).normal_oct = packed;
    }
    count
}

/// Storage-agnostic smoother. Enumerates the shell, runs the Laplacian
/// iterations, and returns `Vec<(slot_id, packed_normal)>` for the
/// caller to write back however it owns the target buffer. Returning
/// the pairs (rather than writing via a closure) avoids an `&`/`&mut`
/// overlap when the read and write closures would both capture the
/// same buffer.
fn smooth_shell_normals_inner<GetCell, ReadNormal>(
    octree: &SparseOctree,
    iterations: u32,
    get_cell: GetCell,
    read_initial: ReadNormal,
) -> Vec<(u32, u32)>
where
    GetCell: Fn(u32, u32, u32, u32) -> u32,
    ReadNormal: Fn(u32) -> u32,
{
    if iterations == 0 {
        return Vec::new();
    }

    // Pass 1: enumerate shell voxels.
    let mut shell: Vec<(UVec3, u32)> = Vec::new();
    let depth = octree.depth();
    collect_shell_voxels(octree, &get_cell, 0, UVec3::ZERO, 0, depth, &mut shell);
    if shell.is_empty() {
        return Vec::new();
    }

    // Pass 2: leaf_attr_id → shell-index map for O(1) neighbor lookup.
    let mut id_to_index: HashMap<u32, usize> = HashMap::with_capacity(shell.len());
    for (i, (_, id)) in shell.iter().enumerate() {
        let prev = id_to_index.insert(*id, i);
        debug_assert!(
            prev.is_none(),
            "smooth_shell_normals requires 1:1 leaf_attr ids across shell voxels (id {id} shared)",
        );
    }

    // Pass 3: double-buffered Laplacian iterations.
    let mut current: Vec<Vec3> = shell
        .iter()
        .map(|(_, id)| unpack_oct(read_initial(*id)))
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
                if let Some(n) = neighbor_shell_normal(octree, &get_cell, nc, &id_to_index, &current) {
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

    shell
        .iter()
        .enumerate()
        .map(|(i, (_, id))| (*id, pack_oct(current[i])))
        .collect()
}

fn neighbor_shell_normal(
    octree: &SparseOctree,
    get_cell: &impl Fn(u32, u32, u32, u32) -> u32,
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
        let cell = get_cell(bid, cx % BRICK_DIM, cy % BRICK_DIM, cz % BRICK_DIM);
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

fn collect_shell_voxels(
    octree: &SparseOctree,
    get_cell: &impl Fn(u32, u32, u32, u32) -> u32,
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
                    let cell = get_cell(bid, x, y, z);
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
            get_cell,
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

        let octree = SparseOctree::from_raw(&[make_brick(bid)], BRICK_LEVELS, 0.05);
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

    /// Same mesh, but now call the raw-buffer entry point directly.
    /// Verifies the import-time variant produces identical results.
    #[test]
    fn raw_entry_point_matches_pool_based() {
        let normal = Vec3::new(0.6, 0.0, 0.8).normalize();

        // Build parallel representations: pool (runtime) + raw buffers (import).
        let mut pool = LeafAttrPool::new(512);
        let mut bricks = BrickPool::new(4);
        let bid = bricks.allocate().unwrap();
        let mut file_bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        let mut normals_packed: Vec<u32> = Vec::new();

        let mut slot_counter: u32 = 0;
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let id = pool.allocate_contiguous_bump(1).unwrap();
                    *pool.get_mut(id) = LeafAttr::new(normal, 0);
                    bricks.set_cell(bid, x, y, z, id);

                    // Raw buffers use their own slot numbering — start
                    // from 0 and assign sequentially.
                    let raw_slot = slot_counter;
                    slot_counter += 1;
                    normals_packed.push(pack_oct(normal));
                    let cell_flat = (x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM) as usize;
                    file_bricks[cell_flat] = raw_slot;
                }
            }
        }

        // Pool variant.
        let octree = SparseOctree::from_raw(&[make_brick(bid)], BRICK_LEVELS, 0.05);
        smooth_shell_normals(&octree, &bricks, &mut pool, 3);

        // Raw variant (note: needs an octree whose bricks reference the
        // raw-brick's id, which happens to be 0 in both cases).
        let mut normals_after = normals_packed.clone();
        smooth_shell_normals_raw(&octree, &file_bricks, &mut normals_after, 3);

        // Results should agree cell-by-cell (compare unpacked normals to
        // tolerate octahedral rounding).
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let cell_flat = (x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM) as usize;
                    let raw_slot = file_bricks[cell_flat] as usize;
                    let pool_cell = bricks.get_cell(bid, x, y, z);
                    let pool_n = unpack_oct(pool.get(pool_cell).normal_oct);
                    let raw_n = unpack_oct(normals_after[raw_slot]);
                    assert!(pool_n.dot(raw_n) > 0.999, "mismatch at ({x},{y},{z}): pool={pool_n:?} raw={raw_n:?}");
                }
            }
        }
    }
}
