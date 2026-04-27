//! Paint-write operations against the scene's leaf_attr pool.
//!
//! Given a brush sphere in object-local space, enumerate the leaf_attr slots
//! inside the sphere and mutate their material or color. The existing
//! `LeafAttrPool` already owns both the per-leaf material entries and the
//! parallel `colors` array, and GPU upload is driven by the scene manager's
//! `geometry_epoch` — so paint is a CPU-side edit plus one epoch bump.
//!
//! Phase 1: euclidean sphere with smoothstep falloff. Phase 3 will swap this
//! for a geodesic surface flood fill that wraps around corners.
//!
//! This module is agnostic to commands / input / UI — it operates on raw
//! octree + brick data and a `LeafAttrPool`. Call sites (the engine's paint
//! command handler) are responsible for looking up the target entity's
//! `AssetInfo` and resolving the brush world position into object-local
//! space.

use glam::{UVec3, Vec3};
use rkp_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
use rkp_core::brick_pool::{brick_flat_index, BrickPool, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use rkp_core::leaf_attr_pool::LeafAttrPool;
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

/// What a single brush stamp writes. The scene-manager orchestrator
/// takes this + a brush sphere and applies the matching per-leaf write
/// to every voxel under the brush. Rkp-engine converts its command's
/// `PaintMode` + fields into one of these variants at call time.
#[derive(Debug, Clone, Copy)]
pub enum PaintStamp {
    /// Flip the target leaf's primary material (with soft blend on the
    /// sphere shoulder via `LeafAttr.material_secondary`).
    Material { material_id: u16 },
    /// Write `rgb` into the companion color_pool, lerping from the
    /// existing color by the per-leaf weight.
    Color { rgb: [f32; 3] },
    /// Fade the companion color override toward the material base
    /// color (color_pool=0 sentinel).
    Erase,
}

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

/// Weight-curve used by paint writes: returns a value in `[0, strength]` that
/// falls from the brush center to zero at the radius edge. Matches the
/// rkifield curve (linear core + smoothstep shoulder) so Phase 3's geodesic
/// upgrade can swap the distance metric without touching the weight shape.
///
/// `falloff` is the fraction of the radius occupied by the smoothstep
/// shoulder: 0.0 = hard edge, 1.0 = smoothstep from the center outward.
#[inline]
pub fn brush_weight(distance: f32, radius: f32, strength: f32, falloff: f32) -> f32 {
    if radius <= 0.0 || distance < 0.0 || distance >= radius {
        return 0.0;
    }
    if falloff <= 0.0 {
        return strength;
    }
    let t = distance / radius;
    let edge_start = 1.0 - falloff;
    if t <= edge_start {
        strength
    } else {
        let s = (t - edge_start) / falloff;
        strength * (1.0 - s * s * (3.0 - 2.0 * s))
    }
}

/// Pack a linear RGB triple plus a write intensity (0-255) into the
/// `LeafAttrPool::colors` layout: `R | G<<8 | B<<16 | intensity<<24`. An
/// intensity of 0 is the "no override, fall back to material base_color"
/// sentinel — paint writes that want to zero a voxel's color should use
/// `erase_leaf_color`, not pack intensity=0 here, so partial erasures still
/// blend correctly.
#[inline]
pub fn pack_color(rgb: [f32; 3], intensity: f32) -> u32 {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u32;
    let i = (intensity.clamp(0.0, 1.0) * 255.0).round() as u32;
    r | (g << 8) | (b << 16) | (i << 24)
}

/// Unpack a packed color u32 into linear RGB + intensity. Inverse of
/// [`pack_color`].
#[inline]
pub fn unpack_color(packed: u32) -> ([f32; 3], f32) {
    let r = (packed & 0xFF) as f32 / 255.0;
    let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
    let b = ((packed >> 16) & 0xFF) as f32 / 255.0;
    let i = ((packed >> 24) & 0xFF) as f32 / 255.0;
    ([r, g, b], i)
}

/// Write a new primary material to a leaf, with weighted dual-material
/// blending.
///
/// `weight` is in `[0, 1]` — typically [`brush_weight`] output for this
/// leaf. At weight=1 the leaf becomes pure `material_id` (blend weight 0).
/// At intermediate weights the new material rides into `material_secondary`
/// with a blend weight proportional to `weight`, so dragging the brush
/// softly paints a gradient from old to new material at the sphere edge.
///
/// The `LeafAttr` layout only allows 4 bits of blend weight (0-15), so
/// fractional painting quantizes to 16 levels. That's enough for visually
/// smooth transitions at typical voxel sizes; fine-grain material
/// gradients would need the blend-weight field widened in `LeafAttr`.
pub fn paint_leaf_material(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    material_id: u16,
    weight: f32,
) {
    let w = weight.clamp(0.0, 1.0);
    let leaf = pool.get_mut(leaf_slot);
    let cur_primary = leaf.material_primary;
    if cur_primary == material_id || w <= 0.0 {
        // Either already painted with this material, or weight is zero.
        // Full-weight case still falls here when primary already matches.
        if w >= 0.999 {
            // Clear any leftover secondary blend toward a different material.
            *leaf = rkp_core::LeafAttr {
                normal_oct: leaf.normal_oct,
                material_primary: material_id,
                material_secondary_blend: 0,
            };
        }
        return;
    }

    if w >= 0.999 {
        // Hard overwrite — primary flips to the new material, blend cleared.
        *leaf = rkp_core::LeafAttr {
            normal_oct: leaf.normal_oct,
            material_primary: material_id,
            material_secondary_blend: 0,
        };
        return;
    }

    // Partial blend. Quantize weight to the 4-bit blend field.
    let blend_weight = (w * 15.0).round().clamp(0.0, 15.0) as u8;
    *leaf = rkp_core::LeafAttr::new_blended(
        leaf.normal(),
        cur_primary,
        material_id,
        blend_weight,
    );
}

/// Write a new color onto a leaf, lerping from the existing color by
/// `weight`. Unpainted leaves (intensity=0 in the colors array) start from
/// the target RGB at reduced intensity so a single dab gives visible color
/// immediately.
pub fn paint_leaf_color(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    rgb: [f32; 3],
    weight: f32,
) {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return;
    }
    let cur = pool.color(leaf_slot);
    let (cur_rgb, cur_i) = if cur == 0 {
        // No existing override — treat the leaf as "target RGB, zero
        // intensity" so the first dab lerps from transparent to opaque
        // at the target color instead of going through material-albedo
        // grey.
        (rgb, 0.0)
    } else {
        unpack_color(cur)
    };
    let new_rgb = [
        cur_rgb[0] + (rgb[0] - cur_rgb[0]) * w,
        cur_rgb[1] + (rgb[1] - cur_rgb[1]) * w,
        cur_rgb[2] + (rgb[2] - cur_rgb[2]) * w,
    ];
    let new_i = cur_i + (1.0 - cur_i) * w;
    pool.set_color(leaf_slot, pack_color(new_rgb, new_i));
}

/// Erase a leaf's color by lerping the intensity channel toward zero.
/// Full strength wipes the override entirely (clears `color_pool[slot]`
/// to the 0 sentinel), returning the leaf to its material's base
/// albedo. Partial strength fades toward the material over multiple
/// strokes — same feel as Photoshop's eraser. The shade pass routes
/// `color_pool[slot] == 0` to `mat_albedo(material)` via a 0 in the
/// gbuffer's RGB565 channel; see `octree_march.wgsl`.
pub fn erase_leaf_color(
    pool: &mut LeafAttrPool,
    leaf_slot: u32,
    weight: f32,
) {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return;
    }
    let cur = pool.color(leaf_slot);
    if cur == 0 {
        return; // already material-albedo, nothing to erase.
    }
    let (cur_rgb, cur_i) = unpack_color(cur);
    let new_i = cur_i * (1.0 - w);
    if new_i <= 1.0 / 255.0 {
        // Intensity quantized to zero — clear the whole override so the
        // shader takes the material base color fast-path.
        pool.set_color(leaf_slot, 0);
    } else {
        pool.set_color(leaf_slot, pack_color(cur_rgb, new_i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec3;
    use rkp_core::sparse_octree::SparseOctree;
    use rkp_core::OctreeAllocator;

    const EPS: f32 = 1e-4;

    // -------- brush_weight --------

    #[test]
    fn brush_weight_center_is_strength() {
        assert!((brush_weight(0.0, 1.0, 0.8, 0.5) - 0.8).abs() < EPS);
    }

    #[test]
    fn brush_weight_edge_is_zero() {
        assert!(brush_weight(1.0, 1.0, 0.8, 0.5) < EPS);
    }

    #[test]
    fn brush_weight_beyond_radius_is_zero() {
        assert_eq!(brush_weight(2.0, 1.0, 1.0, 0.5), 0.0);
    }

    #[test]
    fn brush_weight_hard_cutoff_is_flat() {
        assert!((brush_weight(0.0, 1.0, 0.7, 0.0) - 0.7).abs() < EPS);
        assert!((brush_weight(0.99, 1.0, 0.7, 0.0) - 0.7).abs() < EPS);
        assert_eq!(brush_weight(1.0, 1.0, 0.7, 0.0), 0.0);
    }

    #[test]
    fn brush_weight_monotonically_decreases() {
        let (r, s, fo) = (2.0, 0.9, 0.6);
        let mut prev = brush_weight(0.0, r, s, fo);
        for i in 1..20 {
            let d = (i as f32) * r / 20.0;
            let cur = brush_weight(d, r, s, fo);
            assert!(cur <= prev + EPS, "d={d} cur={cur} prev={prev}");
            prev = cur;
        }
    }

    // -------- pack/unpack roundtrip --------

    #[test]
    fn pack_unpack_roundtrip() {
        let (rgb, i) = unpack_color(pack_color([0.5, 0.25, 0.75], 0.8));
        assert!((rgb[0] - 0.5).abs() < 0.01);
        assert!((rgb[1] - 0.25).abs() < 0.01);
        assert!((rgb[2] - 0.75).abs() < 0.01);
        assert!((i - 0.8).abs() < 0.01);
    }

    #[test]
    fn pack_intensity_zero_means_no_override() {
        // A voxel with no color override reads back as zeros in every channel.
        let packed = pack_color([0.0, 0.0, 0.0], 0.0);
        assert_eq!(packed, 0);
    }

    // -------- paint_leaf_material --------

    #[test]
    fn paint_material_full_weight_overwrites() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(2).unwrap();
        *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 1);
        paint_leaf_material(&mut pool, 0, 42, 1.0);
        assert_eq!(pool.get(0).material_primary, 42);
        assert_eq!(pool.get(0).blend_weight(), 0);
    }

    #[test]
    fn paint_material_partial_weight_blends_into_secondary() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 3);
        paint_leaf_material(&mut pool, 0, 7, 0.5);
        let a = pool.get(0);
        assert_eq!(a.material_primary, 3);
        assert_eq!(a.material_secondary(), 7);
        // 0.5 * 15 = 7.5 → rounds to 8
        assert_eq!(a.blend_weight(), 8);
    }

    #[test]
    fn paint_material_noop_when_already_matching_primary() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 5);
        paint_leaf_material(&mut pool, 0, 5, 0.5);
        // Half-weight paint onto an already-matching primary: no secondary blend.
        let a = pool.get(0);
        assert_eq!(a.material_primary, 5);
        assert_eq!(a.material_secondary(), 0);
        assert_eq!(a.blend_weight(), 0);
    }

    #[test]
    fn paint_material_zero_weight_is_noop() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        *pool.get_mut(0) = rkp_core::LeafAttr::new(Vec3::Y, 9);
        paint_leaf_material(&mut pool, 0, 42, 0.0);
        assert_eq!(pool.get(0).material_primary, 9);
    }

    // -------- paint_leaf_color --------

    #[test]
    fn paint_color_from_unpainted_creates_override() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        paint_leaf_color(&mut pool, 0, [1.0, 0.0, 0.0], 1.0);
        assert_ne!(pool.color(0), 0, "color override should be set");
        let (rgb, i) = unpack_color(pool.color(0));
        assert!((rgb[0] - 1.0).abs() < 0.01);
        assert!((i - 1.0).abs() < 0.01);
    }

    #[test]
    fn paint_color_partial_weight_ramps_intensity() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        paint_leaf_color(&mut pool, 0, [1.0, 0.0, 0.0], 0.5);
        let (_, i) = unpack_color(pool.color(0));
        // 0.0 + (1.0-0.0)*0.5 = 0.5
        assert!((i - 0.5).abs() < 0.01, "partial weight should give ~0.5 intensity, got {i}");
    }

    #[test]
    fn paint_color_zero_weight_noop() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        paint_leaf_color(&mut pool, 0, [1.0, 1.0, 1.0], 0.0);
        assert_eq!(pool.color(0), 0);
    }

    // -------- erase_leaf_color --------

    #[test]
    fn erase_full_weight_clears_override() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        pool.set_color(0, pack_color([1.0, 0.5, 0.0], 1.0));
        erase_leaf_color(&mut pool, 0, 1.0);
        assert_eq!(pool.color(0), 0);
    }

    #[test]
    fn erase_partial_weight_dims_intensity() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        pool.set_color(0, pack_color([1.0, 0.0, 0.0], 1.0));
        erase_leaf_color(&mut pool, 0, 0.5);
        let (rgb, i) = unpack_color(pool.color(0));
        assert!((rgb[0] - 1.0).abs() < 0.01); // RGB preserved
        assert!((i - 0.5).abs() < 0.01); // intensity halved
    }

    #[test]
    fn erase_unpainted_is_noop() {
        let mut pool = LeafAttrPool::new(8);
        pool.allocate_range(1).unwrap();
        erase_leaf_color(&mut pool, 0, 1.0);
        assert_eq!(pool.color(0), 0);
    }

    // -------- leaves_in_sphere --------

    /// Build a packed-buffer octree that contains a single LEAF at
    /// `coord` with `slot`. Returns the buffer + its root_offset, depth,
    /// and base_voxel_size so tests can call `leaves_in_sphere`
    /// straight against the packed form (same shape that
    /// `OctreeGpu::data()` hands to the shader).
    fn packed_octree_with_leaf(coord: UVec3, slot: u32, depth: u8, vs: f32)
        -> (Vec<u32>, u32, u8, f32)
    {
        let mut tree = SparseOctree::new(depth, vs);
        tree.insert(coord, slot);
        let mut alloc = OctreeAllocator::new();
        let handle = alloc.allocate(&tree);
        (alloc.as_slice().to_vec(), handle.root_offset, handle.depth, handle.base_voxel_size)
    }

    #[test]
    fn leaves_in_sphere_hits_leaf_at_center() {
        // 4³ octree (depth=2), voxel_size=1.0, leaf at (1,1,1) → center (1.5, 1.5, 1.5).
        let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(1, 1, 1), 42, 2, 1.0);
        let pool = BrickPool::new(1);
        let hits = leaves_in_sphere(
            &buf, root, depth, vs,
            &pool,
            Vec3::ZERO,
            Vec3::new(1.5, 1.5, 1.5),
            0.9,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].leaf_slot, 42);
        assert!(hits[0].distance < EPS);
    }

    #[test]
    fn leaves_in_sphere_misses_leaf_outside_radius() {
        let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(0, 0, 0), 5, 2, 1.0);
        let pool = BrickPool::new(1);
        // Leaf center is (0.5, 0.5, 0.5) — brush at (3, 3, 3) with radius 1.0 can't reach it.
        let hits = leaves_in_sphere(
            &buf, root, depth, vs,
            &pool,
            Vec3::ZERO,
            Vec3::new(3.0, 3.0, 3.0),
            1.0,
        );
        assert!(hits.is_empty(), "brush far from leaf should yield no hits, got {}", hits.len());
    }

    #[test]
    fn leaves_in_sphere_zero_radius_no_hits() {
        let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(0, 0, 0), 1, 2, 1.0);
        let pool = BrickPool::new(1);
        let hits = leaves_in_sphere(
            &buf, root, depth, vs,
            &pool,
            Vec3::ZERO,
            Vec3::ZERO,
            0.0,
        );
        assert!(hits.is_empty());
    }

    #[test]
    fn leaves_in_sphere_multiple_hits() {
        // Two leaves at adjacent cells, large brush reaches both.
        let mut tree = SparseOctree::new(2, 1.0);
        tree.insert(UVec3::new(0, 0, 0), 11); // center (0.5, 0.5, 0.5)
        tree.insert(UVec3::new(1, 0, 0), 22); // center (1.5, 0.5, 0.5)
        let mut alloc = OctreeAllocator::new();
        let handle = alloc.allocate(&tree);
        let pool = BrickPool::new(1);
        // Brush at the midpoint (1.0, 0.5, 0.5) with radius 0.8 catches both.
        let hits = leaves_in_sphere(
            alloc.as_slice(),
            handle.root_offset,
            handle.depth,
            handle.base_voxel_size,
            &pool,
            Vec3::ZERO,
            Vec3::new(1.0, 0.5, 0.5),
            0.8,
        );
        assert_eq!(hits.len(), 2);
        let slots: Vec<u32> = hits.iter().map(|h| h.leaf_slot).collect();
        assert!(slots.contains(&11));
        assert!(slots.contains(&22));
    }

    // -------- leaf_at_local_pos --------

    /// Build a small octree with a single 4³ brick allocated at the
    /// origin corner and populated with leaf slots for a flat XY
    /// "slab" (z=0 layer of the brick). Helper for the flood-fill
    /// tests — deterministic layout where we know exactly which
    /// (brick, cell) → slot mapping exists.
    fn slab_octree(
        attrs: &mut LeafAttrPool,
        bricks: &mut BrickPool,
    ) -> (SparseOctree, u32) {
        // depth 2 → 4³ voxels, one brick terminates the root directly
        // since BRICK_LEVELS = 2 == depth. Voxel size 1.0 → brick
        // covers [0,4) per axis.
        let depth = 2u8;
        let voxel_size = 1.0_f32;
        let mut tree = SparseOctree::new(depth, voxel_size);
        // Allocate one brick and populate its z=0 layer with unique
        // leaf slots (0..16).
        let brick_id = bricks.allocate().unwrap();
        // The octree needs to point AT this brick. SparseOctree::new
        // gives an all-EMPTY root; set the root directly to a brick
        // terminator.
        // `set_at_level` with target_level 0 puts a value at the root.
        let brick_node = rkp_core::sparse_octree::make_brick(brick_id);
        tree.set_at_level(UVec3::ZERO, 0, brick_node);

        // Allocate 16 leaf slots and write them into the z=0 layer.
        let slots = attrs.allocate_range(16).unwrap();
        for y in 0..BRICK_DIM {
            for x in 0..BRICK_DIM {
                let slot = slots[(x + y * BRICK_DIM) as usize];
                bricks.set_cell(brick_id, x, y, 0, slot);
            }
        }
        (tree, brick_id)
    }

    #[test]
    fn leaf_at_local_pos_hits_populated_cell() {
        let mut attrs = LeafAttrPool::new(64);
        let mut bricks = BrickPool::new(4);
        let (tree, brick_id) = slab_octree(&mut attrs, &mut bricks);
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // Query the center of cell (2, 1, 0) → (2.5, 1.5, 0.5).
        let hit = leaf_at_local_pos(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, Vec3::ZERO, Vec3::new(2.5, 1.5, 0.5),
        ).expect("hit should resolve to a brick cell");

        assert_eq!(hit.brick_id, brick_id);
        assert_eq!(hit.cell, UVec3::new(2, 1, 0));
        // center should be exactly (2.5, 1.5, 0.5) for voxel_size 1.0.
        assert!((hit.center_local - Vec3::new(2.5, 1.5, 0.5)).length() < EPS);
        // cell_size = brick extent / BRICK_DIM = 4.0 / 4 = 1.0.
        assert!((hit.cell_size - 1.0).abs() < EPS);
    }

    #[test]
    fn leaf_at_local_pos_returns_none_in_empty_cell() {
        let mut attrs = LeafAttrPool::new(64);
        let mut bricks = BrickPool::new(4);
        let (tree, _) = slab_octree(&mut attrs, &mut bricks);
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // z=2 layer is empty — slab only populated z=0.
        let hit = leaf_at_local_pos(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, Vec3::ZERO, Vec3::new(2.5, 1.5, 2.5),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn leaf_at_local_pos_returns_none_for_empty_octree() {
        let bricks = BrickPool::new(4);
        let tree = SparseOctree::new(2, 1.0);
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);
        let hit = leaf_at_local_pos(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(hit.is_none());
    }

    // -------- surface_flood_fill --------

    #[test]
    fn flood_fill_starts_from_hit_voxel() {
        let mut attrs = LeafAttrPool::new(64);
        let mut bricks = BrickPool::new(4);
        let (tree, _) = slab_octree(&mut attrs, &mut bricks);
        // Force the face links to all-empty — this test uses a single
        // brick so no cross-brick walks are needed.
        let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // Radius 0.0-ish: only the hit cell should come back.
        let hits = surface_flood_fill(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, &face_links, Vec3::ZERO,
            Vec3::new(2.5, 1.5, 0.5), 0.0001,
        );
        // init_dist from center is 0, so even an epsilon radius captures the hit.
        assert_eq!(hits.len(), 1, "got {hits:?}");
        assert!(hits[0].distance < EPS);
    }

    #[test]
    fn flood_fill_expands_face_neighbors_within_radius() {
        let mut attrs = LeafAttrPool::new(64);
        let mut bricks = BrickPool::new(4);
        let (tree, _) = slab_octree(&mut attrs, &mut bricks);
        let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // Start at (2.5, 1.5, 0.5). Radius 1.5 emits every z=0 voxel
        // whose world-local center sits within 1.5 world units of the
        // brush origin — BFS uses face adjacency for reachability but
        // the emitted distance is straight-line, so diagonals enter
        // once the diagonal neighbor is within radius.
        let hits = surface_flood_fill(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, &face_links, Vec3::ZERO,
            Vec3::new(2.5, 1.5, 0.5), 1.5,
        );
        // Expected: 1 center + 4 face neighbors (dist 1.0) + 4
        // diagonals (dist sqrt(2) ≈ 1.414 < 1.5). Missing voxels
        // beyond the 3×3 centered patch are at dist >= 2.
        assert_eq!(hits.len(), 9, "got {hits:?}");

        let near_zero = hits.iter().filter(|h| h.distance < 0.01).count();
        let near_one = hits.iter().filter(|h| (h.distance - 1.0).abs() < 0.01).count();
        let near_diag = hits.iter()
            .filter(|h| (h.distance - std::f32::consts::SQRT_2).abs() < 0.01)
            .count();
        assert_eq!(near_zero, 1);
        assert_eq!(near_one, 4);
        assert_eq!(near_diag, 4);
    }

    #[test]
    fn flood_fill_crosses_brick_boundary_via_face_link() {
        // Two bricks side-by-side on +X. Start in brick A, expand into
        // brick B through the face link.
        let mut attrs = LeafAttrPool::new(128);
        let mut bricks = BrickPool::new(4);

        let brick_a = bricks.allocate().unwrap();
        let brick_b = bricks.allocate().unwrap();

        // Populate brick A's +X edge cell (3, 0, 0) and brick B's
        // −X edge cell (0, 0, 0). Leave everything else empty.
        let slots = attrs.allocate_range(2).unwrap();
        bricks.set_cell(brick_a, 3, 0, 0, slots[0]);
        bricks.set_cell(brick_b, 0, 0, 0, slots[1]);

        // Build an octree that has both bricks. Depth 3 gives us 8³
        // voxels split into a 2×2×2 brick grid (brick edge = 4 voxels).
        let mut tree = SparseOctree::new(3, 1.0);
        // Put brick A at brick-coord (0,0,0) (voxel 0), brick B at
        // (1,0,0) (voxel 4). `set_at_level` with target_level equal to
        // the brick depth (1, for depth 3 with BRICK_LEVELS=2) puts a
        // brick terminator.
        let brick_depth = 3 - 2; // 1
        let a_node = rkp_core::sparse_octree::make_brick(brick_a);
        let b_node = rkp_core::sparse_octree::make_brick(brick_b);
        tree.set_at_level(UVec3::new(0, 0, 0), brick_depth, a_node);
        tree.set_at_level(UVec3::new(4, 0, 0), brick_depth, b_node);

        // Face-link table: A's +X → B; B's −X → A; everything else empty.
        let max_brick_id = brick_a.max(brick_b);
        let mut face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]; (max_brick_id + 1) as usize];
        face_links[brick_a as usize][1] = brick_b; // FACE_PX
        face_links[brick_b as usize][0] = brick_a; // FACE_NX

        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // Start at brick A's edge cell center (3.5, 0.5, 0.5). Radius
        // 1.5 should reach the cell at brick B's (0, 0, 0) which is at
        // world (4.5, 0.5, 0.5) — distance 1.0 away.
        let hits = surface_flood_fill(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, &face_links, Vec3::ZERO,
            Vec3::new(3.5, 0.5, 0.5), 1.5,
        );
        assert_eq!(hits.len(), 2, "expected A + B neighbor, got {hits:?}");
        let slots_hit: Vec<u32> = hits.iter().map(|h| h.leaf_slot).collect();
        assert!(slots_hit.contains(&slots[0]));
        assert!(slots_hit.contains(&slots[1]));
    }

    #[test]
    fn flood_fill_respects_radius_cutoff() {
        let mut attrs = LeafAttrPool::new(64);
        let mut bricks = BrickPool::new(4);
        let (tree, _) = slab_octree(&mut attrs, &mut bricks);
        let face_links: Vec<[u32; 6]> = vec![[FACE_EMPTY; 6]];
        let mut alloc = OctreeAllocator::new();
        let h = alloc.allocate(&tree);

        // Radius 0.7 is less than one cell hop (1.0), so only the hit
        // cell should be returned.
        let hits = surface_flood_fill(
            alloc.as_slice(), h.root_offset, h.depth, h.base_voxel_size,
            &bricks, &face_links, Vec3::ZERO,
            Vec3::new(2.5, 1.5, 0.5), 0.7,
        );
        assert_eq!(hits.len(), 1, "got {hits:?}");
    }

    #[test]
    fn leaves_in_sphere_respects_grid_origin() {
        // Same leaf setup, but the octree's (0,0,0) sits at (-2, -2, -2).
        // Leaf at (1,1,1) is now at world (-2 + 1.5, -2 + 1.5, -2 + 1.5).
        let (buf, root, depth, vs) = packed_octree_with_leaf(UVec3::new(1, 1, 1), 99, 2, 1.0);
        let pool = BrickPool::new(1);
        let origin = Vec3::new(-2.0, -2.0, -2.0);
        let hits = leaves_in_sphere(
            &buf, root, depth, vs,
            &pool, origin,
            origin + Vec3::new(1.5, 1.5, 1.5),
            0.6,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].leaf_slot, 99);
    }
}
