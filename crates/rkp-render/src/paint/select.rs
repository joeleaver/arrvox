//! Spatial paint selection — sphere brush + single-cell cursor pick.
//! Every fn here is a pure read of the octree + brick pool; no
//! `LeafAttrPool` writes.
//!
//! Output types ([`PaintedLeaf`], [`LeafHit`]) live here too —
//! they're the products of selection that [`super::write`] consumes.

use glam::{UVec3, Vec3};
use rkp_core::brick_pool::{brick_flat_index, BrickPool, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use rkp_core::sparse_octree::{
    brick_id as node_brick_id, is_branch, is_brick, is_leaf, leaf_slot as node_leaf_slot,
    EMPTY_NODE, INTERIOR_NODE,
};

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

