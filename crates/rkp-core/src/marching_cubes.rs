//! Marching cubes triangle extraction from a sparse octree's opacity field.
//!
//! Produces a standard indexed triangle mesh from the trilinearly-interpolated
//! opacity field. The surface is where opacity crosses [`THRESHOLD`].
//!
//! This is the Phase 1-5 implementation per `notes/marching-cubes-migration.md`:
//! - Vertex sharing by edge identity — every MC vertex sits on a cube edge
//!   uniquely identified by (lower_voxel_coord, axis). Cells sharing the
//!   same edge point at the same vertex index, so the vertex buffer shrinks
//!   ~3-5× vs the naïve "3 unique verts per triangle" approach (Phase 5).
//! - Normals derived from a 6-tap central difference of the trilinearly-
//!   interpolated opacity field at each MC vertex (Phase 2).
//! - Colors lerped along the MC edge with the same `t` used to place the
//!   vertex — `color = lerp(color_A, color_B, t)` in linear RGB (Phase 3).
//! - Dual-material blending (Phase 3.5): each vertex carries (primary_mat,
//!   secondary_mat, blend_weight). Primary/secondary ids pick from the
//!   inside (above-threshold) corner; the per-voxel blend_weight is lerped
//!   along the edge so seams between distinctly-blended voxels get a smooth
//!   material transition. Falls back cleanly to single-material rendering
//!   when secondary == 0 and blend_weight == 0.
//! - Active cells found by expanding each leaf's lower corner into its 8 adjacent
//!   cells — correct for depth-`max_depth` leaves. Coarse (LOD) leaves miss their
//!   far-face boundaries (future LOD expansion).
//!
//! Positions are in **object-local world-units**, centered on the origin —
//! the same convention the GPU octree march uses. An octree voxel coord
//! `c` maps to local position `c * base_voxel_size − extent_world/2`. The
//! vertex shader then applies the object's `world` matrix to get world
//! space, no extra scale or offset needed.

use glam::{UVec3, Vec3};
use std::collections::{HashMap, HashSet};

use crate::sparse_octree::{is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE};
use crate::{SparseOctree, VoxelPool};

/// Isosurface threshold for MC extraction. The surface sits where the
/// trilinear opacity field crosses this value. Standard industry default;
/// yields clean half-voxel surfaces.
pub const THRESHOLD: f32 = 0.5;

/// Extracted indexed triangle mesh, ready for upload to a GPU vertex/index
/// buffer pool.
///
/// All attributes are per-vertex, parallel arrays. Positions are in
/// octree-local space (voxel units, `[0, extent]`). Normals are unit vectors
/// pointing away from the opaque side of the isosurface (Phase 2; Phase 1
/// stores placeholder `Vec3::Y`). Indices are 3-per-triangle, CCW when viewed
/// from outside.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtractedMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Packed R8G8B8 | intensity, same format as [`VoxelPool::color`].
    pub colors: Vec<u32>,
    /// Primary material id per vertex.
    pub material_ids: Vec<u16>,
    /// Secondary material id per vertex (0 when unused).
    pub secondary_material_ids: Vec<u16>,
    /// Per-vertex blend weight: 0 = pure primary, 255 = pure secondary.
    pub blend_weights: Vec<u8>,
    pub indices: Vec<u32>,
}

impl ExtractedMesh {
    /// Number of triangles (indices / 3).
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Corner offsets for a cell at (cx,cy,cz). Index is the Paul-Bourke corner id.
///
/// | idx | (dx, dy, dz) |
/// |-----|--------------|
/// |  0  | (0, 0, 0)    |
/// |  1  | (1, 0, 0)    |
/// |  2  | (1, 1, 0)    |
/// |  3  | (0, 1, 0)    |
/// |  4  | (0, 0, 1)    |
/// |  5  | (1, 0, 1)    |
/// |  6  | (1, 1, 1)    |
/// |  7  | (0, 1, 1)    |
const CORNER_OFFSETS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// Vertex-index endpoints for each of the 12 cube edges.
const EDGE_ENDPOINTS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Extract a triangle mesh from the octree's opacity field.
///
/// Iterates every leaf in `octree`, collects the set of cells that touch a
/// non-empty region, and runs marching cubes on each. Corner opacity comes
/// from `octree.lookup(coord)` — INTERIOR sentinels resolve to 1.0, out-of-
/// bounds / EMPTY to 0.0, leaves to `pool.get(slot).opacity_f32()`.
///
/// Returns an empty mesh for an empty octree.
pub fn extract_mesh(octree: &SparseOctree, pool: &VoxelPool) -> ExtractedMesh {
    let mut mesh = ExtractedMesh::default();

    // 1. Collect opacity samples at every leaf coord. Only fine leaves; coarse
    //    leaves are uniform and don't generate triangles except at their
    //    boundaries, which are covered via `octree.lookup()` during corner
    //    sampling.
    let opacity_grid: HashMap<UVec3, f32> = octree
        .iter_leaves()
        .filter(|(_, _, depth)| *depth == octree.depth())
        .map(|(coord, slot, _)| (coord, pool.get(slot).opacity_f32()))
        .collect();

    if opacity_grid.is_empty() {
        return mesh;
    }

    // Octree-coord → object-local conversion: positions end up centered on
    // the origin in world units (matching the octree march's `oc_origin =
    // local_origin + extent/2` convention).
    let base_vs = octree.base_voxel_size();
    let half_extent = octree.extent_world() * 0.5;

    // 2. Seed active cells: every voxel coord v is a corner of up to 8 cells
    //    (v-dx, v-dy, v-dz) for d in {0,1}^3. Only include non-negative cell
    //    coords (the grid has no cells "below zero").
    let mut active_cells: HashSet<UVec3> = HashSet::with_capacity(opacity_grid.len() * 8);
    for &coord in opacity_grid.keys() {
        for &dx in &[0u32, 1] {
            if coord.x < dx {
                continue;
            }
            for &dy in &[0u32, 1] {
                if coord.y < dy {
                    continue;
                }
                for &dz in &[0u32, 1] {
                    if coord.z < dz {
                        continue;
                    }
                    active_cells.insert(UVec3::new(coord.x - dx, coord.y - dy, coord.z - dz));
                }
            }
        }
    }

    // 3. Process each active cell through MC. A shared `edge_cache` keeps
    //    one vertex per unique octree edge — cells that touch the same edge
    //    reuse the same vertex index, cutting the vertex buffer by the
    //    typical 3-5× seen in MC.
    let mut edge_cache: HashMap<EdgeKey, u32> = HashMap::new();
    for cell in active_cells {
        let sample = sample_cell(octree, pool, cell);
        emit_cell_triangles(
            cell,
            &sample,
            base_vs,
            half_extent,
            octree,
            pool,
            &mut edge_cache,
            &mut mesh,
        );
    }

    mesh
}

/// Canonical identifier for a cube edge in octree voxel-index space.
/// Edges along +X at (x,y,z) → (x+1,y,z) key as `(UVec3(x,y,z), 0)`;
/// +Y is axis 1, +Z is axis 2. Any two MC cells sharing that edge produce
/// the same key and thus a single shared vertex.
type EdgeKey = (UVec3, u8);

/// Build the canonical edge key for edge `e` of a cell at `cell`. The
/// 12 cube edges each align with one of the 3 axes; we take the smaller
/// endpoint voxel coord and tag it with the axis direction.
#[inline]
fn edge_key(cell: UVec3, e: usize) -> EdgeKey {
    let [a, b] = EDGE_ENDPOINTS[e];
    let ca = UVec3::new(
        cell.x + CORNER_OFFSETS[a][0],
        cell.y + CORNER_OFFSETS[a][1],
        cell.z + CORNER_OFFSETS[a][2],
    );
    let cb = UVec3::new(
        cell.x + CORNER_OFFSETS[b][0],
        cell.y + CORNER_OFFSETS[b][1],
        cell.z + CORNER_OFFSETS[b][2],
    );
    let lower = ca.min(cb);
    // Exactly one axis differs between endpoints of a cube edge.
    let axis = if ca.x != cb.x {
        0
    } else if ca.y != cb.y {
        1
    } else {
        2
    };
    (lower, axis)
}

/// Per-cell corner samples. Each array is indexed 0..8 by Paul-Bourke corner.
struct CellSample {
    opacity: [f32; 8],
    primary_mat: [u16; 8],
    secondary_mat: [u16; 8],
    /// 0..=255, meaning 0.0..=1.0 blend toward secondary.
    blend_weight: [u8; 8],
    color: [u32; 8],
}

/// Sample the 8 corners of a cell.
fn sample_cell(octree: &SparseOctree, pool: &VoxelPool, cell: UVec3) -> CellSample {
    let mut s = CellSample {
        opacity: [0.0; 8],
        primary_mat: [0; 8],
        secondary_mat: [0; 8],
        blend_weight: [0; 8],
        color: [0; 8],
    };
    for i in 0..8 {
        let off = CORNER_OFFSETS[i];
        let c = UVec3::new(cell.x + off[0], cell.y + off[1], cell.z + off[2]);
        let (op, pm, sm, bw, col) = sample_corner(octree, pool, c);
        s.opacity[i] = op;
        s.primary_mat[i] = pm;
        s.secondary_mat[i] = sm;
        s.blend_weight[i] = bw;
        s.color[i] = col;
    }
    s
}

/// Resolve all material/color data at a voxel coordinate. OOB / EMPTY / INTERIOR
/// have no associated voxel data — they return zero for everything except
/// opacity (INTERIOR is 1.0).
#[inline]
fn sample_corner(
    octree: &SparseOctree,
    pool: &VoxelPool,
    coord: UVec3,
) -> (f32, u16, u16, u8, u32) {
    match octree.lookup(coord) {
        None => (0.0, 0, 0, 0, 0),
        Some(EMPTY_NODE) => (0.0, 0, 0, 0, 0),
        Some(INTERIOR_NODE) => (1.0, 0, 0, 0, 0),
        Some(node) if is_leaf(node) => {
            let slot = leaf_slot(node);
            let v = pool.get(slot);
            (
                v.opacity_f32(),
                v.material_id(),
                v.secondary_material_id(),
                v.blend_weight(),
                pool.color(slot),
            )
        }
        _ => (0.0, 0, 0, 0, 0),
    }
}

/// Run MC on one cell, append triangles to `mesh`.
///
/// `base_vs` and `half_extent` convert octree voxel-coord space to object-
/// local world units: `local = coord * base_vs − half_extent`. `octree` and
/// `pool` are passed through so we can compute per-vertex gradient normals.
/// `edge_cache` holds one vertex index per unique octree edge so cells
/// sharing an edge reference the same vertex.
#[allow(clippy::too_many_arguments)]
fn emit_cell_triangles(
    cell: UVec3,
    s: &CellSample,
    base_vs: f32,
    half_extent: f32,
    octree: &SparseOctree,
    pool: &VoxelPool,
    edge_cache: &mut HashMap<EdgeKey, u32>,
    mesh: &mut ExtractedMesh,
) {
    // Build the 8-bit cube index: bit i set iff corner i is "inside"
    // (opacity >= THRESHOLD).
    let mut cube_index = 0u8;
    for i in 0..8 {
        if s.opacity[i] >= THRESHOLD {
            cube_index |= 1 << i;
        }
    }

    let edge_mask = MC_EDGE_TABLE[cube_index as usize];
    if edge_mask == 0 {
        return; // cell fully inside or fully outside — no surface
    }

    let offset = Vec3::splat(half_extent);
    let cell_vox = Vec3::new(cell.x as f32, cell.y as f32, cell.z as f32);

    // Resolve each active edge to a (possibly shared) vertex index. Missing
    // entries are emitted into the mesh on demand.
    let mut edge_vertex_idx = [u32::MAX; 12];
    for e in 0..12 {
        if edge_mask & (1 << e) == 0 {
            continue;
        }
        let key = edge_key(cell, e);
        edge_vertex_idx[e] = if let Some(&idx) = edge_cache.get(&key) {
            idx
        } else {
            let [a, b] = EDGE_ENDPOINTS[e];
            let t = interp_t(s.opacity[a], s.opacity[b]);
            let pa = cell_vox
                + Vec3::new(
                    CORNER_OFFSETS[a][0] as f32,
                    CORNER_OFFSETS[a][1] as f32,
                    CORNER_OFFSETS[a][2] as f32,
                );
            let pb = cell_vox
                + Vec3::new(
                    CORNER_OFFSETS[b][0] as f32,
                    CORNER_OFFSETS[b][1] as f32,
                    CORNER_OFFSETS[b][2] as f32,
                );
            let vert_vox = pa + (pb - pa) * t;
            let normal = gradient_normal(vert_vox, octree, pool);
            // Material ids: pick the pair from the inside (above-threshold)
            // corner. Phase 3.5 treats the pair as fixed per voxel.
            let inside = if s.opacity[a] >= THRESHOLD { a } else { b };
            let primary = s.primary_mat[inside];
            let secondary = s.secondary_mat[inside];
            // Blend weight: lerp the two per-voxel blends so a surface
            // between voxels with authored-different blends transitions
            // smoothly.
            let bw_a = s.blend_weight[a] as f32;
            let bw_b = s.blend_weight[b] as f32;
            let blend = (bw_a * (1.0 - t) + bw_b * t).round().clamp(0.0, 255.0) as u8;
            let color = lerp_packed_color(s.color[a], s.color[b], t);

            let idx = mesh.positions.len() as u32;
            mesh.positions.push(vert_vox * base_vs - offset);
            mesh.normals.push(normal);
            mesh.colors.push(color);
            mesh.material_ids.push(primary);
            mesh.secondary_material_ids.push(secondary);
            mesh.blend_weights.push(blend);
            edge_cache.insert(key, idx);
            idx
        };
    }

    // Emit triangles from MC_TRI_TABLE using the shared vertex indices.
    let tris = &MC_TRI_TABLE[cube_index as usize];
    let mut i = 0;
    while i < tris.len() && tris[i] != -1 {
        mesh.indices.push(edge_vertex_idx[tris[i] as usize]);
        mesh.indices.push(edge_vertex_idx[tris[i + 1] as usize]);
        mesh.indices.push(edge_vertex_idx[tris[i + 2] as usize]);
        i += 3;
    }
}

/// Trilinear interpolation of the opacity field at a voxel-index-space
/// position. Samples the 8 surrounding integer voxel corners via
/// [`sample_corner`] and blends by fractional offset.
///
/// Negative coordinates are treated as opacity 0 (outside the octree grid).
fn sample_opacity_at(pos: Vec3, octree: &SparseOctree, pool: &VoxelPool) -> f32 {
    let fx = pos.x.floor();
    let fy = pos.y.floor();
    let fz = pos.z.floor();
    let tx = pos.x - fx;
    let ty = pos.y - fy;
    let tz = pos.z - fz;
    let bx = fx as i64;
    let by = fy as i64;
    let bz = fz as i64;

    let mut sum = 0.0f32;
    for dz in 0..2i64 {
        let wz = if dz == 0 { 1.0 - tz } else { tz };
        for dy in 0..2i64 {
            let wy = if dy == 0 { 1.0 - ty } else { ty };
            for dx in 0..2i64 {
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let cx = bx + dx;
                let cy = by + dy;
                let cz = bz + dz;
                let opacity = if cx < 0 || cy < 0 || cz < 0 {
                    0.0
                } else {
                    sample_corner(octree, pool, UVec3::new(cx as u32, cy as u32, cz as u32)).0
                };
                sum += opacity * wx * wy * wz;
            }
        }
    }
    sum
}

/// Compute the outward surface normal at `pos` (voxel-index space) as the
/// negated unit gradient of the trilinear opacity field, using a 6-tap
/// central difference with step = half a voxel.
///
/// Returns `Vec3::Y` for degenerate cases where the gradient magnitude is
/// effectively zero — rare, only hits on perfectly uniform neighborhoods
/// where the MC vertex shouldn't have been emitted anyway.
fn gradient_normal(pos: Vec3, octree: &SparseOctree, pool: &VoxelPool) -> Vec3 {
    let h = 0.5;
    let gx = sample_opacity_at(pos + Vec3::new(h, 0.0, 0.0), octree, pool)
        - sample_opacity_at(pos - Vec3::new(h, 0.0, 0.0), octree, pool);
    let gy = sample_opacity_at(pos + Vec3::new(0.0, h, 0.0), octree, pool)
        - sample_opacity_at(pos - Vec3::new(0.0, h, 0.0), octree, pool);
    let gz = sample_opacity_at(pos + Vec3::new(0.0, 0.0, h), octree, pool)
        - sample_opacity_at(pos - Vec3::new(0.0, 0.0, h), octree, pool);
    let grad = Vec3::new(gx, gy, gz);
    let len2 = grad.length_squared();
    if len2 < 1e-16 {
        return Vec3::Y;
    }
    // Gradient points from low to high opacity (outside → inside).
    // Outward surface normal is the opposite.
    -grad / len2.sqrt()
}

/// Linear interpolation parameter on an edge. Given corner opacities `oa` and
/// `ob` (with opposite sign relative to `THRESHOLD`), returns `t ∈ [0, 1]`
/// such that `lerp(a, b, t)` hits the isosurface.
#[inline]
fn interp_t(oa: f32, ob: f32) -> f32 {
    let denom = ob - oa;
    if denom.abs() < 1e-6 {
        0.5
    } else {
        ((THRESHOLD - oa) / denom).clamp(0.0, 1.0)
    }
}

/// Linear interpolation of two packed colors (`R8G8B8 | intensity8`) at
/// parameter `t ∈ [0, 1]`. Interpolates each channel independently in linear
/// space (matches what the shade pass expects when it unpacks to RGB565).
///
/// Missing-color handling: if exactly one endpoint is zero (no color
/// assigned) we pass the other through verbatim — lerping toward black
/// would wrongly darken surfaces that straddle an author-set color and a
/// default-color-less neighbor. If both endpoints are zero, result is zero.
#[inline]
fn lerp_packed_color(a: u32, b: u32, t: f32) -> u32 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let u = 1.0 - t;
    let lerp_u8 = |ca: u32, cb: u32| -> u32 {
        let f = (ca as f32) * u + (cb as f32) * t;
        f.round().clamp(0.0, 255.0) as u32
    };
    let r = lerp_u8(a & 0xFF, b & 0xFF);
    let g = lerp_u8((a >> 8) & 0xFF, (b >> 8) & 0xFF);
    let blue = lerp_u8((a >> 16) & 0xFF, (b >> 16) & 0xFF);
    let intensity = lerp_u8((a >> 24) & 0xFF, (b >> 24) & 0xFF);
    r | (g << 8) | (blue << 16) | (intensity << 24)
}

// ===========================================================================
// Standard Paul-Bourke Marching Cubes tables.
// https://paulbourke.net/geometry/polygonise/
// 256 cube configurations × 12 edges, then 256 × (up to 15) triangle indices.
// ===========================================================================

/// For each cube configuration (0..256), a 12-bit mask indicating which edges
/// are crossed by the isosurface.
#[rustfmt::skip]
const MC_EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// For each cube configuration, up to 5 triangles (15 edge indices + terminator).
/// The list is terminated by `-1`. Each triangle is 3 consecutive edge indices.
#[rustfmt::skip]
const MC_TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [ 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [ 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [ 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [ 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [ 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [ 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [ 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [ 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [ 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [ 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [ 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [ 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [ 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [ 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [ 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [ 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [ 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [ 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [ 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [ 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [ 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [ 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1],
    [ 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1],
    [ 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [ 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [ 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [ 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [ 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [ 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [ 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [ 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [ 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [ 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [ 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [ 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [ 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [ 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [ 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [ 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [ 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [ 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [ 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [ 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [ 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [ 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [ 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [ 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [ 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [ 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [ 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [ 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [ 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [ 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [ 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [ 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [ 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [ 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [ 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [ 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [ 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [ 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [ 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [ 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [ 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [ 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SplatVoxel;

    /// Build an octree of the given depth and fill one voxel at `coord` with
    /// full opacity.
    fn single_voxel(depth: u8, coord: UVec3) -> (SparseOctree, VoxelPool) {
        let mut octree = SparseOctree::new(depth, 1.0);
        let mut pool = VoxelPool::new(16);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = SplatVoxel::new(1.0, 0);
        octree.insert(coord, slot);
        (octree, pool)
    }

    #[test]
    fn empty_octree_yields_empty_mesh() {
        let octree = SparseOctree::new(3, 1.0);
        let pool = VoxelPool::new(8);
        let mesh = extract_mesh(&octree, &pool);
        assert!(mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 0);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn single_voxel_produces_surface() {
        let (octree, pool) = single_voxel(3, UVec3::new(2, 2, 2));
        let mesh = extract_mesh(&octree, &pool);

        // A lone opacity=1 voxel surrounded by empty (opacity=0) creates
        // an isosurface on all 6 of its faces.
        assert!(!mesh.is_empty(), "expected some geometry");
        assert_eq!(mesh.indices.len() % 3, 0, "indices must be multiples of 3");
        // With vertex sharing, a closed manifold has roughly `tris * 0.5`
        // vertices (each vert touches ~6 tris by Euler). Upper bound:
        // vertex_count <= index_count (pre-sharing parity).
        assert!(
            mesh.vertex_count() <= mesh.indices.len(),
            "sharing shouldn't grow the vertex buffer beyond the pre-sharing upper bound",
        );
        assert!(
            mesh.vertex_count() < mesh.indices.len(),
            "at least some edges should be shared across cells (sharing is working)",
        );

        // Object-local coords: extent=8 vs vs=1.0 → half_extent=4. The cells
        // touching voxel (2,2,2) span voxel indices [1,4] in each axis,
        // which maps to local [1-4, 4-4] = [-3, 0].
        for p in &mesh.positions {
            assert!(
                p.x >= -3.0 && p.x <= 0.0 && p.y >= -3.0 && p.y <= 0.0 && p.z >= -3.0 && p.z <= 0.0,
                "vertex {p:?} outside expected range",
            );
            assert!(p.is_finite(), "vertex has NaN/inf: {p:?}");
        }
    }

    #[test]
    fn vertex_sharing_reduces_count() {
        // For a solid 4x4x4 block of voxels, MC produces a closed shell.
        // Without sharing we'd get ~3 verts per triangle; with sharing each
        // vertex touches multiple triangles, so vertex_count << indices.
        let depth = 3u8;
        let mut octree = SparseOctree::new(depth, 1.0);
        let mut pool = VoxelPool::new(128);
        for z in 2..6u32 {
            for y in 2..6u32 {
                for x in 2..6u32 {
                    let slot = pool.allocate().unwrap();
                    *pool.get_mut(slot) = SplatVoxel::new(1.0, 0);
                    octree.insert(UVec3::new(x, y, z), slot);
                }
            }
        }
        let mesh = extract_mesh(&octree, &pool);
        assert!(!mesh.is_empty());
        // Should see at least ~2x compression (tris / verts >= 2).
        let ratio = mesh.indices.len() as f32 / mesh.vertex_count() as f32;
        assert!(
            ratio >= 2.0,
            "expected index/vertex ratio >= 2 (sharing), got {ratio:.2} ({} idx / {} vert)",
            mesh.indices.len(),
            mesh.vertex_count(),
        );
    }

    #[test]
    fn single_voxel_has_no_degenerate_triangles() {
        let (octree, pool) = single_voxel(3, UVec3::new(2, 2, 2));
        let mesh = extract_mesh(&octree, &pool);
        for tri in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let area2 = (b - a).cross(c - a).length_squared();
            assert!(area2 > 1e-8, "degenerate triangle: {a:?} {b:?} {c:?}");
        }
    }

    #[test]
    fn sphere_produces_closed_manifold() {
        // Fill a ball of radius ~3 at the center of a depth-3 (8³) grid with
        // opacity=1. The MC isosurface should be a roughly spherical shell.
        let depth = 3u8;
        let extent = 1u32 << depth;
        let center = Vec3::splat((extent as f32) / 2.0 - 0.5);
        let radius = 3.0f32;

        let mut octree = SparseOctree::new(depth, 1.0);
        let mut pool = VoxelPool::new(extent.pow(3));
        for z in 0..extent {
            for y in 0..extent {
                for x in 0..extent {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    if p.distance(center) <= radius {
                        let slot = pool.allocate().unwrap();
                        *pool.get_mut(slot) = SplatVoxel::new(1.0, 0);
                        octree.insert(UVec3::new(x, y, z), slot);
                    }
                }
            }
        }

        let mesh = extract_mesh(&octree, &pool);
        assert!(mesh.triangle_count() >= 50, "expected many triangles, got {}", mesh.triangle_count());

        // Every triangle non-degenerate and finite.
        for tri in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let area2 = (b - a).cross(c - a).length_squared();
            assert!(area2 > 1e-8, "degenerate triangle");
            assert!(a.is_finite() && b.is_finite() && c.is_finite());
        }

        // Euler-ish sanity: for a closed manifold with T triangles and no
        // vertex sharing, expect 3T verts. Closed surface implies each
        // triangle edge is shared with exactly one other triangle, so the
        // number of triangles should be even.
        assert_eq!(mesh.triangle_count() % 2, 0, "closed mesh has even tri count");
    }

    #[test]
    fn interior_sentinel_treated_as_opaque() {
        // Mark an entire subtree as INTERIOR at the root. The MC march should
        // treat its corners as opacity=1 and therefore see no surface inside.
        let mut octree = SparseOctree::new(3, 1.0);
        octree.set_at_level(UVec3::ZERO, 0, INTERIOR_NODE);
        let pool = VoxelPool::new(1);
        let mesh = extract_mesh(&octree, &pool);
        // No leaves → opacity_grid empty → no active cells → empty mesh.
        // (A boundary is only generated where a leaf meets non-interior space,
        // which doesn't happen in a pure-INTERIOR tree.)
        assert!(mesh.is_empty());
    }

    #[test]
    fn triangle_count_matches_indices_div_3() {
        let (octree, pool) = single_voxel(3, UVec3::new(1, 1, 1));
        let mesh = extract_mesh(&octree, &pool);
        assert_eq!(mesh.indices.len() % 3, 0);
        assert_eq!(mesh.triangle_count(), mesh.indices.len() / 3);
    }

    #[test]
    fn single_voxel_normals_point_outward() {
        // Octree voxels are POINT SAMPLES at integer coords, not unit cubes.
        // The sample at (2,2,2) with base_vs=1 sits at local (2,2,2)·1 −
        // half_extent(4) = (−2,−2,−2). A lone opacity=1 sample creates a
        // 6-faced isosurface whose vertices are at sample ± 0.5 along each
        // axis. Every normal should point outward from the sample point.
        let (octree, pool) = single_voxel(3, UVec3::new(2, 2, 2));
        let mesh = extract_mesh(&octree, &pool);
        let sample_local = Vec3::new(-2.0, -2.0, -2.0);

        assert!(!mesh.is_empty());
        for (i, (&pos, &nrm)) in mesh.positions.iter().zip(mesh.normals.iter()).enumerate() {
            let len = nrm.length();
            assert!(
                (len - 1.0).abs() < 1e-3,
                "normal[{i}] = {nrm:?} is not unit length (len={len})",
            );
            let radial = pos - sample_local;
            if radial.length_squared() < 1e-8 {
                continue; // shouldn't happen, but guard against div-by-zero.
            }
            let out = radial.normalize();
            let dot = nrm.dot(out);
            assert!(
                dot > 0.1,
                "normal[{i}] at {pos:?} points inward: nrm={nrm:?}, out={out:?}, dot={dot}",
            );
        }
    }

    #[test]
    fn sphere_normals_point_outward_from_center() {
        let depth = 3u8;
        let extent = 1u32 << depth;
        let center_vox = Vec3::splat((extent as f32) / 2.0 - 0.5);
        let radius = 3.0f32;
        let base_vs = 1.0f32;
        let half_extent = (extent as f32) * base_vs * 0.5;
        let center_local = center_vox * base_vs - Vec3::splat(half_extent);

        let mut octree = SparseOctree::new(depth, base_vs);
        let mut pool = VoxelPool::new(extent.pow(3));
        for z in 0..extent {
            for y in 0..extent {
                for x in 0..extent {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    if p.distance(center_vox) <= radius {
                        let slot = pool.allocate().unwrap();
                        *pool.get_mut(slot) = SplatVoxel::new(1.0, 0);
                        octree.insert(UVec3::new(x, y, z), slot);
                    }
                }
            }
        }

        let mesh = extract_mesh(&octree, &pool);
        assert!(!mesh.is_empty());

        // Majority of normals should point roughly outward from the sphere
        // center — MC produces discrete-step surface faces, so individual
        // vertices can deviate (e.g., on a facet edge), but the average dot
        // product against the radial direction should be strongly positive.
        let mut dot_sum = 0.0f32;
        let mut n = 0usize;
        for (&pos, &nrm) in mesh.positions.iter().zip(mesh.normals.iter()) {
            assert!(nrm.is_finite());
            let len = nrm.length();
            assert!((len - 1.0).abs() < 1e-3 || len == 0.0);
            let radial = pos - center_local;
            if radial.length_squared() > 1e-8 {
                dot_sum += nrm.dot(radial.normalize());
                n += 1;
            }
        }
        let mean = dot_sum / n as f32;
        assert!(mean > 0.7, "mean outward dot {mean} too low — normals inverted?");
    }

    #[test]
    fn lerp_packed_color_basic() {
        let red = 0x0000_00FF; // R=255
        let blue = 0x00FF_0000; // B=255
        let mid = lerp_packed_color(red, blue, 0.5);
        // Each channel ~127
        assert_eq!(mid & 0xFF, 128); // R lerps from 255 to 0 at t=0.5 → 128 (round)
        assert_eq!((mid >> 16) & 0xFF, 128); // B lerps from 0 to 255 → 128
    }

    #[test]
    fn lerp_packed_color_missing_color_passes_through() {
        // When one endpoint has no color assigned (0), we take the other
        // endpoint rather than lerping toward black.
        let red = 0x0000_00FF;
        let zero = 0u32;
        assert_eq!(lerp_packed_color(red, zero, 0.3), red);
        assert_eq!(lerp_packed_color(zero, red, 0.3), red);
        assert_eq!(lerp_packed_color(zero, zero, 0.5), 0);
    }

    #[test]
    fn lerp_packed_color_endpoints() {
        let a = 0xAA_BB_CC_DD;
        let b = 0x11_22_33_44;
        assert_eq!(lerp_packed_color(a, b, 0.0), a);
        assert_eq!(lerp_packed_color(a, b, 1.0), b);
    }

    #[test]
    fn mc_vertex_between_colored_voxels_gets_blended_color() {
        // Two adjacent voxels, red and blue, with opposite opacities so the
        // isosurface sits on the edge between them. The MC vertex there
        // should have a color that mixes red and blue.
        let mut octree = SparseOctree::new(3, 1.0);
        let mut pool = VoxelPool::new(16);

        let red_slot = pool.allocate().unwrap();
        *pool.get_mut(red_slot) = SplatVoxel::new(1.0, 0);
        pool.set_color(red_slot, 0x0000_00FF); // R=255
        octree.insert(UVec3::new(3, 3, 3), red_slot);

        // Neighbor at (4,3,3) is absent (opacity=0). For the edge between
        // (3,3,3) and (4,3,3), the MC vertex should be near the red side
        // since opacity goes 1 → 0. But t = (0.5-1)/(0-1) = 0.5, so the
        // vertex sits at the midpoint and picks up 50% red + 50% of
        // (4,3,3)'s color (which is 0/empty → lerp rule passes red through).
        let mesh = extract_mesh(&octree, &pool);

        // Every vertex should have the red color passed through (since the
        // "other side" of every edge is empty/zero).
        assert!(!mesh.is_empty());
        for &c in &mesh.colors {
            assert_eq!(c & 0xFF, 0xFF, "expected R=255 on every vertex, got {c:#x}");
        }
    }

    #[test]
    fn mc_vertex_carries_secondary_and_blend() {
        // A voxel with dual-material blending should carry primary, secondary,
        // and blend_weight through extraction onto every MC vertex it
        // generates (material pair is taken from the inside corner).
        let mut octree = SparseOctree::new(3, 1.0);
        let mut pool = VoxelPool::new(16);
        let slot = pool.allocate().unwrap();
        *pool.get_mut(slot) = SplatVoxel::new_blended(1.0, 10, 20, 128);
        octree.insert(UVec3::new(2, 2, 2), slot);

        let mesh = extract_mesh(&octree, &pool);
        assert!(!mesh.is_empty());
        assert_eq!(mesh.material_ids.len(), mesh.positions.len());
        assert_eq!(mesh.secondary_material_ids.len(), mesh.positions.len());
        assert_eq!(mesh.blend_weights.len(), mesh.positions.len());

        // Every vertex should carry material pair (10, 20) since the inside
        // corner for every active cell is voxel (2,2,2). Blend weights lerp
        // between that voxel's 128 and neighbor 0, so they'll be 0..128.
        for (i, (&p, &s)) in mesh
            .material_ids
            .iter()
            .zip(mesh.secondary_material_ids.iter())
            .enumerate()
        {
            assert_eq!(p, 10, "primary mismatch at vertex {i}");
            assert_eq!(s, 20, "secondary mismatch at vertex {i}");
        }
        for (i, &bw) in mesh.blend_weights.iter().enumerate() {
            assert!(bw <= 128, "blend_weight[{i}] = {bw} above max seed 128");
        }
    }

    #[test]
    fn edge_table_and_tri_table_agree() {
        // For every cube configuration, the edges used in the tri table must
        // be a subset of the edges flagged in the edge table. Catches table
        // transcription bugs.
        for cfg in 0..256 {
            let mask = MC_EDGE_TABLE[cfg];
            let tris = &MC_TRI_TABLE[cfg];
            for &e in tris {
                if e == -1 {
                    break;
                }
                let bit = 1u16 << e;
                assert!(
                    mask & bit != 0,
                    "cfg {cfg}: tri table uses edge {e} not in edge mask {mask:#x}",
                );
            }
        }
    }
}
