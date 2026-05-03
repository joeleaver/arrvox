//! Spatial paint selection — sphere brush, single-cell cursor pick, and
//! geodesic surface flood fill. Every fn here is a pure read of the
//! octree + brick pool + face-link table; no `LeafAttrPool` writes.
//!
//! Output types ([`PaintedLeaf`], [`LeafHit`], [`FloodedLeaf`]) live
//! here too — they're the products of selection that [`super::write`]
//! consumes.
//!
//! The 6-face delta table [`FACE_DELTAS`] is `pub(super)` so write-side
//! helpers can match on the same ordering as `brick_face_links`.

use glam::{UVec3, Vec3};
use rkp_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
use rkp_core::brick_pool::{brick_flat_index, BrickPool, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use rkp_core::sparse_octree::{
    brick_id as node_brick_id, is_branch, is_brick, is_leaf, leaf_slot as node_leaf_slot,
    EMPTY_NODE, INTERIOR_NODE,
};

/// 6-face deltas matching `brick_face_links` ordering:
/// `(−X, +X, −Y, +Y, −Z, +Z)`. Used by the geodesic flood fill so cell
/// neighbors pair correctly with the brick face the hop crosses.
const FACE_DELTAS: [(i32, i32, i32); 6] = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
];

/// One leaf inside a brush's influence, ready to receive a paint edit.
#[derive(Debug, Clone, Copy)]
pub struct PaintedLeaf {
    /// Scene-global leaf_attr slot (already offset by `leaf_attr_slot_start`).
    pub leaf_slot: u32,
    /// Voxel center in object-local space.
    pub center_local: Vec3,
    /// Euclidean distance from the brush center, in object-local units.
    pub distance: f32,
}

/// Enumerate every surface voxel inside a spherical brush footprint.
///
/// `octree_buffer` is the packed node buffer (from `OctreeGpu::data()`);
/// branch values are already absolute offsets into this buffer, leaves hold
/// scene-global `leaf_attr_slot` values, and bricks hold scene-global
/// `brick_id`s. `root_offset` + `depth` + `base_voxel_size` come from the
/// target asset's `SpatialHandle`. `grid_origin_local` is the object-local
/// position of the octree's (0,0,0) corner (typically
/// `aabb_center - extent/2`).
///
/// The implementation descends the packed buffer recursively, AABB-culling
/// each subtree against the sphere. Leaves inside the sphere are emitted as
/// one [`PaintedLeaf`]; bricks expand to their non-empty cells. Nothing
/// allocates or reads from the pool — this is a pure spatial query.
pub fn leaves_in_sphere(
    octree_buffer: &[u32],
    root_offset: u32,
    depth: u8,
    base_voxel_size: f32,
    brick_pool: &BrickPool,
    grid_origin_local: Vec3,
    center_local: Vec3,
    radius: f32,
) -> Vec<PaintedLeaf> {
    if radius <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let root_extent = (1u32 << depth) as f32 * base_voxel_size;
    descend(
        octree_buffer,
        brick_pool,
        root_offset as usize,
        grid_origin_local,
        root_extent,
        base_voxel_size,
        center_local,
        radius,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn descend(
    octree_buffer: &[u32],
    brick_pool: &BrickPool,
    node_idx: usize,
    node_origin_local: Vec3,
    node_extent: f32,
    base_voxel_size: f32,
    center_local: Vec3,
    radius: f32,
    out: &mut Vec<PaintedLeaf>,
) {
    // AABB-vs-sphere cull. Compare the sphere's squared distance to the
    // node's AABB against `radius²`.
    let node_max = node_origin_local + Vec3::splat(node_extent);
    let clamped = center_local.clamp(node_origin_local, node_max);
    let delta = center_local - clamped;
    if delta.length_squared() > radius * radius {
        return;
    }

    let node = octree_buffer[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }

    if is_leaf(node) {
        // A leaf at non-finest depth covers a whole subtree; we treat it as
        // one voxel centered in its AABB. Rare in baked scenes — bricks
        // dominate.
        let center = node_origin_local + Vec3::splat(node_extent * 0.5);
        let d = (center - center_local).length();
        if d <= radius {
            out.push(PaintedLeaf {
                leaf_slot: node_leaf_slot(node),
                center_local: center,
                distance: d,
            });
        }
        return;
    }

    if is_brick(node) {
        let brick_id = node_brick_id(node);
        emit_brick_leaves(
            brick_pool,
            brick_id,
            node_origin_local,
            node_extent,
            center_local,
            radius,
            out,
        );
        return;
    }

    if is_branch(node) {
        let children_offset = node as usize;
        let half = node_extent * 0.5;
        for octant in 0u32..8 {
            let dx = (octant & 1) as f32;
            let dy = ((octant >> 1) & 1) as f32;
            let dz = ((octant >> 2) & 1) as f32;
            let child_origin = node_origin_local + Vec3::new(dx * half, dy * half, dz * half);
            descend(
                octree_buffer,
                brick_pool,
                children_offset + octant as usize,
                child_origin,
                half,
                base_voxel_size,
                center_local,
                radius,
                out,
            );
        }
    }
}

fn emit_brick_leaves(
    brick_pool: &BrickPool,
    brick_id: u32,
    brick_origin_local: Vec3,
    brick_extent: f32,
    center_local: Vec3,
    radius: f32,
    out: &mut Vec<PaintedLeaf>,
) {
    let cells = brick_pool.brick_cells(brick_id);
    let cell_size = brick_extent / BRICK_DIM as f32;
    let r2 = radius * radius;
    for dz in 0..BRICK_DIM {
        for dy in 0..BRICK_DIM {
            for dx in 0..BRICK_DIM {
                let idx = brick_flat_index(dx, dy, dz) as usize;
                let cell = cells[idx];
                if cell == BRICK_EMPTY || cell == BRICK_INTERIOR {
                    continue;
                }
                let cell_center = brick_origin_local
                    + Vec3::new(
                        (dx as f32 + 0.5) * cell_size,
                        (dy as f32 + 0.5) * cell_size,
                        (dz as f32 + 0.5) * cell_size,
                    );
                let delta = cell_center - center_local;
                let d2 = delta.length_squared();
                if d2 <= r2 {
                    out.push(PaintedLeaf {
                        leaf_slot: cell,
                        center_local: cell_center,
                        distance: d2.sqrt(),
                    });
                }
            }
        }
    }
}

/// Result of descending the octree at a single world-local position.
/// Identifies the surface voxel the position sits inside so the flood
/// fill knows where to start. Plain LEAF octree terminators (not
/// BRICK) aren't currently handled — they're rare in baked scenes, and
/// `BrickFaceLinks` (the adjacency structure the flood uses) only
/// covers brick-level neighbors. For a scene with plain LEAFs mixed
/// in, the flood simply returns empty and the cursor falls back to
/// the Phase-2 wireframe sphere.
#[derive(Debug, Clone, Copy)]
pub struct LeafHit {
    /// Brick the hit cell belongs to.
    pub brick_id: u32,
    /// Cell coordinates within the brick, 0..[`BRICK_DIM`].
    pub cell: UVec3,
    /// Scene-global leaf_attr slot (the cell value).
    pub leaf_slot: u32,
    /// Cell center in object-local space.
    pub center_local: Vec3,
    /// Length of one cell along any axis in world units — equal to
    /// the octree's `base_voxel_size` for bricks at the finest level.
    /// Cached here so the caller doesn't need to re-derive it.
    pub cell_size: f32,
}

/// Descend the packed octree buffer to the leaf containing `pos_local`.
///
/// Returns `None` when the traversal lands in an empty subtree, an
/// INTERIOR region, a plain LEAF (not handled — see [`LeafHit`] doc),
/// or a BRICK cell that reads as empty / interior. Most positions
/// that the paint pick's world-space hit lands on will resolve to a
/// BRICK surface cell; the other outcomes are the "nothing to flood
/// from" fallbacks.
pub fn leaf_at_local_pos(
    octree_buffer: &[u32],
    root_offset: u32,
    depth: u8,
    base_voxel_size: f32,
    brick_pool: &BrickPool,
    grid_origin_local: Vec3,
    pos_local: Vec3,
) -> Option<LeafHit> {
    let mut node_idx = root_offset as usize;
    let mut node_origin = grid_origin_local;
    let mut node_extent = (1u32 << depth) as f32 * base_voxel_size;

    // Safety cap — the octree depth is the worst case. Each iteration
    // either descends one level (branch) or terminates.
    for _ in 0..=depth {
        if node_idx >= octree_buffer.len() {
            return None;
        }
        let node = octree_buffer[node_idx];
        if node == EMPTY_NODE || node == INTERIOR_NODE {
            return None;
        }
        if is_leaf(node) {
            // Plain-LEAF terminator — skip for Phase 3 v1. See `LeafHit`
            // doc for why.
            return None;
        }
        if is_brick(node) {
            let brick_id = node_brick_id(node);
            let rel = pos_local - node_origin;
            let cell_size = node_extent / BRICK_DIM as f32;
            // Clamp to brick bounds — the caller might pass a point
            // slightly outside due to floating-point drift.
            let cx = ((rel.x / cell_size).floor().clamp(0.0, (BRICK_DIM - 1) as f32)) as u32;
            let cy = ((rel.y / cell_size).floor().clamp(0.0, (BRICK_DIM - 1) as f32)) as u32;
            let cz = ((rel.z / cell_size).floor().clamp(0.0, (BRICK_DIM - 1) as f32)) as u32;
            let cell_value = brick_pool.brick_cells(brick_id)[brick_flat_index(cx, cy, cz) as usize];
            if cell_value == BRICK_EMPTY || cell_value == BRICK_INTERIOR {
                return None;
            }
            let center_local = node_origin + Vec3::new(
                (cx as f32 + 0.5) * cell_size,
                (cy as f32 + 0.5) * cell_size,
                (cz as f32 + 0.5) * cell_size,
            );
            return Some(LeafHit {
                brick_id,
                cell: UVec3::new(cx, cy, cz),
                leaf_slot: cell_value,
                center_local,
                cell_size,
            });
        }
        if is_branch(node) {
            let half = node_extent * 0.5;
            let rel = pos_local - node_origin;
            let ox = if rel.x >= half { 1u32 } else { 0 };
            let oy = if rel.y >= half { 1u32 } else { 0 };
            let oz = if rel.z >= half { 1u32 } else { 0 };
            let octant = (ox + oy * 2 + oz * 4) as usize;
            let children_offset = node as usize;
            node_idx = children_offset + octant;
            node_origin = node_origin + Vec3::new(
                if ox == 1 { half } else { 0.0 },
                if oy == 1 { half } else { 0.0 },
                if oz == 1 { half } else { 0.0 },
            );
            node_extent = half;
            continue;
        }
        // Unreachable for well-formed nodes.
        return None;
    }
    None
}

/// One voxel touched by the geodesic flood fill. `distance` is the
/// world-space geodesic distance from the brush origin (the pick
/// position, not the voxel center) — surface-walking distance, not
/// straight-line — which is why the cursor wraps around corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloodedLeaf {
    pub leaf_slot: u32,
    pub distance: f32,
}

/// Walk face-adjacent surface voxels from the brush origin, collecting
/// leaves whose **euclidean distance** to the brush origin is within
/// `radius`. The walk itself uses face adjacency (so the cursor can't
/// jump across gaps / through air), but the emitted distance is
/// straight-line — BFS hop count × cell_size gives a rhombic
/// (Manhattan-ball) cursor shape, which looks hexagonal on flat
/// surfaces. Euclidean distance on the face-connected set gives a
/// proper circle on flat regions, cleanly cut off by surface
/// topology around edges and corners.
///
/// Neighbor lookup: cell-local within a brick (4³), cross-brick via
/// `brick_face_links` (wrap the stepped-out-of-bounds cell coord into
/// the neighbor brick's face).
pub fn surface_flood_fill(
    octree_buffer: &[u32],
    root_offset: u32,
    depth: u8,
    base_voxel_size: f32,
    brick_pool: &BrickPool,
    brick_face_links: &[[u32; 6]],
    grid_origin_local: Vec3,
    start_pos_local: Vec3,
    radius: f32,
) -> Vec<FloodedLeaf> {
    if radius <= 0.0 {
        return Vec::new();
    }
    let Some(start) = leaf_at_local_pos(
        octree_buffer, root_offset, depth, base_voxel_size,
        brick_pool, grid_origin_local, start_pos_local,
    ) else {
        return Vec::new();
    };

    let init_dist = (start.center_local - start_pos_local).length();
    if init_dist > radius {
        return Vec::new();
    }

    let cell_size = start.cell_size;

    // BFS state. `visited` de-dupes leaf_slots; each queue entry
    // carries the cell's world-local center so neighbors can compute
    // their own world positions by adding `cell_size * face_delta`.
    //
    // Pruning: we keep exploring as long as a neighbor COULD land
    // within the radius. A face-neighbor is at most `cell_size`
    // closer to the brush origin than the current cell, so we can
    // safely cut the search off once a cell is farther than
    // `radius + cell_size` (any neighbor would still be > radius).
    use std::collections::{HashSet, VecDeque};
    let mut visited: HashSet<u32> = HashSet::new();
    let mut queue: VecDeque<(u32, UVec3, u32, Vec3)> = VecDeque::new();
    let mut out: Vec<FloodedLeaf> = Vec::new();

    visited.insert(start.leaf_slot);
    queue.push_back((start.brick_id, start.cell, start.leaf_slot, start.center_local));
    out.push(FloodedLeaf { leaf_slot: start.leaf_slot, distance: init_dist });

    let brick_dim = BRICK_DIM as i32;
    let prune_dist = radius + cell_size;

    while let Some((brick_id, cell, _slot, center)) = queue.pop_front() {
        let cur_dist = (center - start_pos_local).length();
        if cur_dist > prune_dist {
            // No neighbor can cross back under the radius from here.
            continue;
        }

        for face_idx in 0..6 {
            let (dx, dy, dz) = FACE_DELTAS[face_idx];
            let nx = cell.x as i32 + dx;
            let ny = cell.y as i32 + dy;
            let nz = cell.z as i32 + dz;

            let (nbr_brick_id, ncx, ncy, ncz) =
                if (0..brick_dim).contains(&nx)
                    && (0..brick_dim).contains(&ny)
                    && (0..brick_dim).contains(&nz)
                {
                    // Within the same brick.
                    (brick_id, nx as u32, ny as u32, nz as u32)
                } else {
                    // Stepped off a face — follow the brick face link.
                    let row = brick_face_links.get(brick_id as usize).copied();
                    let link = match row {
                        Some(r) => r[face_idx],
                        None => continue,
                    };
                    if link == FACE_EMPTY || link == FACE_INTERIOR {
                        continue;
                    }
                    // Wrap the out-of-bounds coord into the neighbor's
                    // facing edge: +X out of cell.x=3 lands on the
                    // neighbor's x=0, and so on.
                    let wx = ((nx + brick_dim) % brick_dim) as u32;
                    let wy = ((ny + brick_dim) % brick_dim) as u32;
                    let wz = ((nz + brick_dim) % brick_dim) as u32;
                    (link, wx, wy, wz)
                };

            let cells = brick_pool.brick_cells(nbr_brick_id);
            let nbr_slot = cells[brick_flat_index(ncx, ncy, ncz) as usize];
            if nbr_slot == BRICK_EMPTY || nbr_slot == BRICK_INTERIOR {
                continue;
            }
            if !visited.insert(nbr_slot) {
                continue;
            }
            // Neighbor's world-local center: face hops move exactly
            // `cell_size` along the face-delta axis. Works identically
            // for within-brick and across-brick-face moves.
            let nbr_center = center
                + Vec3::new(
                    dx as f32 * cell_size,
                    dy as f32 * cell_size,
                    dz as f32 * cell_size,
                );
            let nbr_dist = (nbr_center - start_pos_local).length();
            if nbr_dist <= radius {
                out.push(FloodedLeaf { leaf_slot: nbr_slot, distance: nbr_dist });
            }
            // Still enqueue even when outside the radius so BFS can
            // reach voxels that are topologically connected through a
            // just-outside-radius voxel (rare but possible on curved
            // surfaces). `prune_dist` bounds the search at
            // `radius + cell_size`.
            queue.push_back((nbr_brick_id, UVec3::new(ncx, ncy, ncz), nbr_slot, nbr_center));
        }
    }

    out
}
