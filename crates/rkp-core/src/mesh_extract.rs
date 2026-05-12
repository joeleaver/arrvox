//! CPU surface-mesh extraction (naive surface nets) at asset load.
//!
//! Walks an asset's brick-terminated octree and emits a triangle mesh
//! that follows the surface defined by the cell-occupancy field. One
//! [`MeshVertex`] per active SN-cube (a `2×2×2` grouping of cells whose
//! corner cells contain a mix of solid and void). Two triangles per
//! active sample-edge (an axis edge between a solid cell and an EMPTY
//! cell). Vertices carry an octahedral-packed average normal and a
//! `leaf_attr_id` slot for the resolve / shade pass to look up
//! prefiltered surface attributes — the same indirection
//! [`SplatVertex`](crate::splat_pass::SplatVertex) uses.
//!
//! No GPU work in Phase 1 — this just produces `(vertices, indices)`
//! that the per-asset cache stores alongside the splat buffer.

use std::collections::HashMap;

use glam::{IVec3, UVec3, Vec3};

use crate::brick_pool::{BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use crate::companion::BoneVoxel;
use crate::leaf_attr::{pack_oct, unpack_oct, LeafAttr};
use crate::sparse_octree::{
    brick_id, is_branch, is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE,
};

/// One surface-mesh vertex.
///
/// 32 B, `repr(C)`, `bytemuck`-castable straight into a vertex buffer.
/// Positions are **object-local**; the per-instance world matrix is
/// applied in the vertex shader. Layout matches
/// [`SplatVertex`](crate::splat_pass::SplatVertex)'s 32 B stride so the
/// per-asset GPU cache can use the same allocation logic.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Cube center in object-local coords. Lands on a grid corner of
    /// the cell lattice (between cells, not on a cell center).
    pub local_pos: [f32; 3],
    /// Octahedral-packed average of the surface-cell normals at the
    /// vertex's 8 corner cells. Falls back to +Y for cubes with no
    /// surface cells (only INTERIOR + EMPTY contributors), which on a
    /// well-baked 1-thick shell shouldn't happen but keeps the
    /// extractor total. Encoding matches `LeafAttr::normal_oct`.
    pub normal_oct: u32,
    /// Absolute slot into the global `leaf_attr_pool`. Picked from the
    /// surface cell with the smallest `(z, y, x)` coord among the
    /// cube's 8 corners — deterministic and stable across reruns.
    /// Falls back to 0 when no corner is a surface cell.
    pub leaf_attr_id: u32,
    /// 4 × u8 bone indices packed little-endian (matches `BoneVoxel.indices`).
    /// Sourced from the same cell that contributed `leaf_attr_id` so the
    /// per-vertex attribution is internally consistent. Zero for
    /// unskinned assets — the matching `bone_weights` is then also zero,
    /// which the vertex shader treats as "skip skinning, rest pose".
    pub bone_indices: u32,
    /// 4 × u8 bone weights packed little-endian (sum to 255 in
    /// well-formed skinning data; 0 for unskinned cells).
    pub bone_weights: u32,
    /// Reserved for future per-vertex attributes (LOD bias, blend
    /// shapes, etc). Keeps the stride at 32 B and the layout
    /// 16-byte-aligned for GPU access.
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<MeshVertex>() == 32);
// Hand-checked field offsets — vertex layout in `mesh_pass/pass.rs`
// pulls position from offset 0, normal_oct from 12, leaf_attr_id from 16.
// Bone fields live in what was `_pad[0..1]`; the GPU-side decl picks
// them up in commit 4 when the VS starts skinning.
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(MeshVertex, local_pos) == 0);
    assert!(offset_of!(MeshVertex, normal_oct) == 12);
    assert!(offset_of!(MeshVertex, leaf_attr_id) == 16);
    assert!(offset_of!(MeshVertex, bone_indices) == 20);
    assert!(offset_of!(MeshVertex, bone_weights) == 24);
    assert!(offset_of!(MeshVertex, _pad) == 28);
};

/// Vertex format for the procedural proxy-mesh pipeline (GPU surface-
/// nets-from-SDF). Distinct from [`MeshVertex`]: proxy meshes have no
/// octree, no LeafAttr pool slots, no skinning. Instead the SDF
/// evaluator's full `TreeSample` (material + secondary + blend + color)
/// is baked per-vertex at extraction time; the proxy raster pipeline
/// reads these directly and writes the G-buffer without going through
/// the LeafAttr indirection used by octree-backed meshes.
///
/// 32 B, `repr(C)`, `bytemuck`-castable. Same stride as [`MeshVertex`]
/// so the surface-nets extractor's buffer allocation logic carries over.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProxyVertex {
    /// SN-cube vertex position in object-local space.
    pub local_pos: [f32; 3],
    /// SDF-gradient normal, octahedral-packed. Encoding matches
    /// [`LeafAttr::normal_oct`].
    pub normal_oct: u32,
    /// Packed material identifiers + blend weight. Same layout as
    /// [`LeafAttr::material_packed`]:
    ///   bits  0-15: primary material_id (u16)
    ///   bits 16-27: secondary material_id (u12)
    ///   bits 28-31: blend_weight (u4)
    pub material_packed: u32,
    /// Per-vertex RGBA8 color from the procedural's color nodes
    /// (`ColorByHeight`, `ColorByNoise`, leaf `color` params).
    /// Low byte = R, next = G, then B, then alpha/intensity.
    /// 0 = "no procedural override, use material base_color".
    pub color_packed: u32,
    /// Reserved for future per-vertex attributes (LOD bias, emission,
    /// node_id for picking, etc.). Keeps the stride at 32 B.
    pub _reserved: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<ProxyVertex>() == 32);
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(ProxyVertex, local_pos) == 0);
    assert!(offset_of!(ProxyVertex, normal_oct) == 12);
    assert!(offset_of!(ProxyVertex, material_packed) == 16);
    assert!(offset_of!(ProxyVertex, color_packed) == 20);
    assert!(offset_of!(ProxyVertex, _reserved) == 24);
};

/// Sentinel marking INTERIOR cells in the dense cell map. INTERIOR
/// cells count as "solid" for SN sign purposes but carry no per-cell
/// `leaf_attr_id`, so we can't store a real slot here.
const CELL_INTERIOR: u32 = u32::MAX;

/// Walk a brick-terminated octree and emit the surface mesh as
/// `(vertices, indices)`.
///
/// * `octree_nodes` — `tree.as_slice()` from the asset's `SparseOctree`.
///   Must already have its brick ids and per-cell `leaf_attr_id` slots
///   remapped to scene-global values (matches the splat extractor's
///   contract).
/// * `octree_depth` — the asset's `depth` field (matches
///   `SparseOctree::depth()`).
/// * `base_voxel_size` — finest cell edge length in object-local units.
/// * `grid_origin` — object-local position of the octree extent's lo
///   corner, same value used by `extract_splats`.
/// * `brick_cells` — flat brick storage; `brick_id * BRICK_CELLS + flat`
///   indexes into it.
/// * `leaf_attr_pool` — the scene-global LeafAttr pool. Indexed by
///   per-cell `leaf_attr_id` to read the prefiltered normal that gets
///   averaged into vertex normals. Pass `&[]` to skip vertex-normal
///   averaging entirely (vertices fall back to +Y); useful for tests.
/// * `bone_voxel_pool` — parallel `BoneVoxel` pool indexed by the same
///   `leaf_attr_id` slots. Vertex shader skinning reads from
///   `bone_indices/weights` baked here. Pass `&[]` for unskinned
///   assets (or tests) — vertices then carry zero weights, which the
///   VS treats as "rest pose".
pub fn extract_surface_mesh(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    if octree_nodes.is_empty() {
        return (vertices, indices);
    }

    // Pass 1: collect every non-empty cell into a dense lookup map.
    // Surface cells store their `leaf_attr_id`; brick-internal INTERIOR
    // cells store `CELL_INTERIOR`. INTERIOR_NODE-region cells are NOT
    // expanded — `is_solid_lookup` resolves them on demand. That keeps
    // the map size proportional to the surface shell, not the asset's
    // solid volume.
    let mut cells: HashMap<IVec3, u32> = HashMap::new();
    walk_collect_cells(
        octree_nodes,
        brick_cells,
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        &mut cells,
    );
    if cells.is_empty() {
        return (vertices, indices);
    }

    // Pass 2: iterate every cell-pair across the 6 face directions.
    // For each (solid → void) edge, the 4 SN cubes around that edge
    // form a quad. Iterating cells in `cells` (rather than scanning
    // every grid edge) keeps us proportional to surface area.
    let mut cube_vertex: HashMap<IVec3, u32> = HashMap::new();
    let extent = 1i32 << octree_depth;

    for &cell in cells.keys() {
        for face in 0..6 {
            let dir = FACE_DIRS[face];
            let neighbor = cell + dir;
            if cells.contains_key(&neighbor) {
                continue;
            }
            // Neighbor isn't in the cell map — could still be inside an
            // INTERIOR_NODE region (which we deliberately didn't expand
            // into the map). Hit the octree to disambiguate.
            if is_solid_lookup(
                octree_nodes,
                brick_cells,
                octree_depth,
                neighbor,
                extent,
            ) {
                continue;
            }
            // Active edge: emit a quad of 4 SN-cube vertices, wound
            // CCW around the outward normal (`dir`, pointing from solid
            // into void).
            let cube_offsets = CUBE_OFFSETS_PER_FACE[face];
            let mut quad = [0u32; 4];
            for i in 0..4 {
                let cube = cell + cube_offsets[i];
                quad[i] = match cube_vertex.get(&cube) {
                    Some(&v) => v,
                    None => {
                        let vertex = build_cube_vertex(
                            cube,
                            &cells,
                            base_voxel_size,
                            grid_origin,
                            leaf_attr_pool,
                            bone_voxel_pool,
                        );
                        let vid = vertices.len() as u32;
                        vertices.push(vertex);
                        cube_vertex.insert(cube, vid);
                        vid
                    }
                };
            }
            indices.extend([quad[0], quad[1], quad[2]]);
            indices.extend([quad[0], quad[2], quad[3]]);
        }
    }

    (vertices, indices)
}

/// Pass 1 of mesh extraction — walk the octree and produce the dense
/// non-empty cell map.
///
/// Exposed separately from [`extract_surface_mesh`] so callers that
/// re-extract **multiple regions per stamp** (the sculpt per-cluster
/// re-extract path in Phase B R4c) can build the map once and run
/// [`extract_mesh_region_from_cells`] against it per region. Each
/// rebuild of the map is O(surface area); per-region pass 2 is
/// proportional to the region's cell count, so amortization is
/// load-bearing for drag-paint perf.
///
/// Returns an empty map for empty octrees.
pub fn collect_cell_map(
    octree_nodes: &[u32],
    octree_depth: u8,
    brick_cells: &[u32],
) -> HashMap<IVec3, u32> {
    let mut cells = HashMap::new();
    if octree_nodes.is_empty() {
        return cells;
    }
    walk_collect_cells(
        octree_nodes,
        brick_cells,
        0,
        UVec3::ZERO,
        0,
        octree_depth,
        &mut cells,
    );
    cells
}

/// Pass 2 of mesh extraction, scoped to a region — produce the surface
/// mesh for cells in `[region_min, region_max)` (half-open).
///
/// **What gets emitted:**
/// * For each solid cell C inside the region (or one cell outside, see
///   pad below): for each of the 6 face directions, if C's neighbor in
///   that direction is empty (or out-of-bounds), emit the quad of 4
///   SN-cube vertices around the face's shared edge.
///
/// **Region boundary handling.** Iteration runs over cells in
/// `[region_min - 1, region_max + 1)` — a 1-cell pad on each side. The
/// pad catches two crack-causing cases:
/// 1. A solid cell *outside* the region whose face-neighbor inside
///    the region is empty: without the pad, the boundary quad on
///    that face would be missing on the region's side.
/// 2. An SN-cube whose vertex sits at the region's edge, with one
///    contributing corner cell just past `region_max`: without the
///    pad, that cube's vertex would be built from incomplete corner
///    data.
///
/// The pad means some output triangles' vertex positions land slightly
/// past `region_max` (up to 1 voxel). Callers that union region outputs
/// (R4c) accept this overlap — duplicate boundary verts across adjacent
/// regions are intentional under the per-cluster-owned model.
///
/// Output indices are *local* to the returned vertex buffer (0-based,
/// referencing positions in the returned `Vec<MeshVertex>`). Caller
/// can drop them straight into a [`crate::cluster_mesh_data::ClusterMesh`]
/// without further remapping.
pub fn extract_mesh_region_from_cells(
    cells: &HashMap<IVec3, u32>,
    region_min: IVec3,
    region_max: IVec3,
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    if cells.is_empty() {
        return (vertices, indices);
    }
    // Empty-region guard (no cells to iterate).
    if region_min.x >= region_max.x
        || region_min.y >= region_max.y
        || region_min.z >= region_max.z
    {
        return (vertices, indices);
    }

    let pad_min = region_min - IVec3::ONE;
    let pad_max = region_max + IVec3::ONE;
    let extent = 1i32 << octree_depth;
    let mut cube_vertex: HashMap<IVec3, u32> = HashMap::new();

    // Iterate the region's bounding box with map lookups. Iterating
    // `cells.iter()` + filtering by region was O(total cells in the
    // asset) per region, which on a 6.5 M-cell asset and 81 dirty
    // clusters per stamp was ~500 M filter checks (12 s wall on
    // splat5). For a typical 10×10×10 region the new path is 1000
    // lookups; for the union-with-brush region (first dirty cluster)
    // it scales with the brush volume, not the asset.
    for z in pad_min.z..pad_max.z {
        for y in pad_min.y..pad_max.y {
            for x in pad_min.x..pad_max.x {
                let cell = IVec3::new(x, y, z);
                if !cells.contains_key(&cell) {
                    continue;
                }
                for face in 0..6 {
                    let dir = FACE_DIRS[face];
                    let neighbor = cell + dir;
                    if cells.contains_key(&neighbor) {
                        continue;
                    }
                    if is_solid_lookup(
                        octree_nodes,
                        brick_cells,
                        octree_depth,
                        neighbor,
                        extent,
                    ) {
                        continue;
                    }
                    let cube_offsets = CUBE_OFFSETS_PER_FACE[face];
                    let mut quad = [0u32; 4];
                    for i in 0..4 {
                        let cube = cell + cube_offsets[i];
                        quad[i] = match cube_vertex.get(&cube) {
                            Some(&v) => v,
                            None => {
                                let vertex = build_cube_vertex(
                                    cube,
                                    cells,
                                    base_voxel_size,
                                    grid_origin,
                                    leaf_attr_pool,
                                    bone_voxel_pool,
                                );
                                let vid = vertices.len() as u32;
                                vertices.push(vertex);
                                cube_vertex.insert(cube, vid);
                                vid
                            }
                        };
                    }
                    indices.extend([quad[0], quad[1], quad[2]]);
                    indices.extend([quad[0], quad[2], quad[3]]);
                }
            }
        }
    }

    (vertices, indices)
}

/// Convenience wrapper: full octree walk + single-region extract in one
/// call. Equivalent to
/// `extract_mesh_region_from_cells(collect_cell_map(..), region, ..)`.
///
/// Use this for one-shot region extraction (R4b unit tests, ad-hoc
/// diagnostics); use the two-step form for sculpt's per-stamp loop
/// across many regions ([`extract_mesh_region_from_cells`] reuses one
/// cell map across all regions).
#[allow(clippy::too_many_arguments)]
pub fn extract_surface_mesh_region(
    octree_nodes: &[u32],
    octree_depth: u8,
    base_voxel_size: f32,
    grid_origin: Vec3,
    brick_cells: &[u32],
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
    region_min: IVec3,
    region_max: IVec3,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let cells = collect_cell_map(octree_nodes, octree_depth, brick_cells);
    extract_mesh_region_from_cells(
        &cells,
        region_min,
        region_max,
        octree_nodes,
        octree_depth,
        base_voxel_size,
        grid_origin,
        brick_cells,
        leaf_attr_pool,
        bone_voxel_pool,
    )
}

/// Build the [`MeshVertex`] for an SN cube whose lo corner is `cube`.
/// The cube spans cells `cube..cube+1` along each axis (8 corner cells
/// total).
///
/// **Position** is the centroid of edge crossings — classical naive
/// surface nets. For each of the cube's 12 axis edges, if the two
/// corner cells have different solidity, the surface "crosses" that
/// edge, and the crossing point is taken at the midpoint of the two
/// cell centers. The vertex sits at the average of all crossings.
///
/// This is the smoothing that takes the mesh from blocky-grid-corner
/// to a recognizable surface — for an isolated single solid cell, the
/// 8 vertices form a smaller cube inscribed at offsets (1/3, 2/3) of
/// the original; for larger features the result tends toward the
/// underlying surface.
///
/// **Solidity** test is `cells.contains_key`. Surface and brick-INTERIOR
/// cells are both in the map; INTERIOR_NODE-region cells aren't, but
/// the typical 1-thick-shell guarantee means any cube near the surface
/// is bounded by cells we already track.
///
/// Falls back to the SN cube's grid corner (`cube + (1, 1, 1)`) when
/// no edge crossings are detected — defensive only.
fn build_cube_vertex(
    cube: IVec3,
    cells: &HashMap<IVec3, u32>,
    voxel_size: f32,
    grid_origin: Vec3,
    leaf_attr_pool: &[LeafAttr],
    bone_voxel_pool: &[BoneVoxel],
) -> MeshVertex {
    // Pre-classify the 8 corner cells once; the edge loop reuses these.
    // Bit layout: index = bit0(+X) | bit1(+Y) | bit2(+Z).
    let mut corner_solid = [false; 8];
    let mut normal_sum = Vec3::ZERO;
    let mut leaf_attr_id: u32 = 0;
    let mut chosen: Option<IVec3> = None;
    for i in 0u32..8 {
        let oa = corner_offset(i);
        let c = cube + oa;
        if let Some(&slot) = cells.get(&c) {
            corner_solid[i as usize] = true;
            if slot != CELL_INTERIOR {
                if let Some(attr) = leaf_attr_pool.get(slot as usize) {
                    normal_sum += unpack_oct(attr.normal_oct);
                }
                let take = match chosen {
                    None => true,
                    Some(prev) => coord_less(c, prev),
                };
                if take {
                    chosen = Some(c);
                    leaf_attr_id = slot;
                }
            }
        }
    }

    // Walk the 12 edges; accumulate crossing midpoints. Midpoint of
    // cells A and B (cube + oa, cube + ob) is at
    // `cube + (oa + ob) * 0.5 + 0.5` in cell-coord units.
    let mut crossing_sum = Vec3::ZERO;
    let mut crossing_count: u32 = 0;
    for &(a, b) in &CUBE_EDGES {
        if corner_solid[a as usize] != corner_solid[b as usize] {
            let oa = corner_offset(a);
            let ob = corner_offset(b);
            let mid = Vec3::new(
                cube.x as f32 + (oa.x + ob.x) as f32 * 0.5 + 0.5,
                cube.y as f32 + (oa.y + ob.y) as f32 * 0.5 + 0.5,
                cube.z as f32 + (oa.z + ob.z) as f32 * 0.5 + 0.5,
            );
            crossing_sum += mid;
            crossing_count += 1;
        }
    }

    let normal_oct = if normal_sum.length_squared() > 1e-12 {
        pack_oct(normal_sum)
    } else {
        pack_oct(Vec3::Y)
    };

    let local_centroid = if crossing_count > 0 {
        crossing_sum / crossing_count as f32
    } else {
        // Cube has no edge crossings (all-solid or all-void) — should
        // never happen for a cube we're emitting a vertex for, but
        // pin to the grid corner if it does.
        Vec3::new(
            cube.x as f32 + 1.0,
            cube.y as f32 + 1.0,
            cube.z as f32 + 1.0,
        )
    };
    let local_pos = grid_origin + local_centroid * voxel_size;

    // Bone weights come from the same chosen surface cell that
    // contributed `leaf_attr_id` — keeps the per-vertex attribution
    // consistent across normal / material / skinning. SN cubes that
    // straddle a bone boundary will pick whichever side won the
    // (z, y, x) tie-break; a smarter blend (max-weighted bone across
    // the 8 corners) is possible but unnecessary at finest voxel size,
    // where each cube already spans a sub-millimeter neighborhood.
    let bone_voxel = bone_voxel_pool
        .get(leaf_attr_id as usize)
        .copied()
        .unwrap_or_default();

    MeshVertex {
        local_pos: local_pos.to_array(),
        normal_oct,
        leaf_attr_id,
        bone_indices: bone_voxel.indices,
        bone_weights: bone_voxel.weights,
        _pad: 0,
    }
}

/// Cube corner offset for index `i` — bit 0 = +X, bit 1 = +Y, bit 2 = +Z.
#[inline]
fn corner_offset(i: u32) -> IVec3 {
    IVec3::new(
        (i & 1) as i32,
        ((i >> 1) & 1) as i32,
        ((i >> 2) & 1) as i32,
    )
}

/// The 12 axis-aligned edges of a cube, as (corner_a, corner_b) index
/// pairs. Order: 4 X-edges, 4 Y-edges, 4 Z-edges.
const CUBE_EDGES: [(u32, u32); 12] = [
    // +X axis (offsets differ in bit 0)
    (0, 1), (2, 3), (4, 5), (6, 7),
    // +Y axis (bit 1)
    (0, 2), (1, 3), (4, 6), (5, 7),
    // +Z axis (bit 2)
    (0, 4), (1, 5), (2, 6), (3, 7),
];

#[inline]
fn coord_less(a: IVec3, b: IVec3) -> bool {
    (a.z, a.y, a.x) < (b.z, b.y, b.x)
}

/// Outward normals for the 6 cell faces, in this order:
/// +X, -X, +Y, -Y, +Z, -Z. Used to walk neighbor cells.
const FACE_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// For each face direction (matching [`FACE_DIRS`]), the 4 SN-cube
/// offsets relative to the solid cell that form the active sample-edge
/// between the solid cell and its empty neighbor — listed in CCW order
/// about the outward normal so triangles `(0, 1, 2)` and `(0, 2, 3)`
/// face outward.
///
/// Derivation: the axis edge between cell A and cell A+dir is shared by
/// 4 SN cubes whose corner cells include both A and A+dir; rotating the
/// 2×2 group of cubes about `dir` (right-hand rule) gives CCW order.
const CUBE_OFFSETS_PER_FACE: [[IVec3; 4]; 6] = [
    // +X — CCW about +X is +Y → +Z.
    [
        IVec3::new(0, -1, -1),
        IVec3::new(0, 0, -1),
        IVec3::new(0, 0, 0),
        IVec3::new(0, -1, 0),
    ],
    // -X — CCW about -X (reverse of +X traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(-1, -1, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(-1, 0, -1),
    ],
    // +Y — CCW about +Y is +Z → +X.
    [
        IVec3::new(-1, 0, -1),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 0, 0),
        IVec3::new(0, 0, -1),
    ],
    // -Y — CCW about -Y (reverse of +Y traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(0, -1, -1),
        IVec3::new(0, -1, 0),
        IVec3::new(-1, -1, 0),
    ],
    // +Z — CCW about +Z is +X → +Y.
    [
        IVec3::new(-1, -1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 0),
        IVec3::new(-1, 0, 0),
    ],
    // -Z — CCW about -Z (reverse of +Z traversal).
    [
        IVec3::new(-1, -1, -1),
        IVec3::new(-1, 0, -1),
        IVec3::new(0, 0, -1),
        IVec3::new(0, -1, -1),
    ],
];

/// Walk the octree and populate `cells` with one entry per non-empty
/// cell, at finest resolution. INTERIOR_NODE-region cells are NOT
/// expanded — `is_solid_lookup` resolves them on demand.
fn walk_collect_cells(
    nodes: &[u32],
    brick_cells: &[u32],
    node_idx: usize,
    origin: UVec3,
    level: u8,
    max_depth: u8,
    cells: &mut HashMap<IVec3, u32>,
) {
    let node = nodes[node_idx];
    if node == EMPTY_NODE || node == INTERIOR_NODE {
        return;
    }
    if is_leaf(node) {
        // Variable-depth LEAF: covers `2^(max_depth - level)` cells per
        // axis. For typical assets these are at finest depth (1 cell);
        // for procedural primitives they may be coarser. Expand to all
        // finest cells so SN sees one uniform lattice.
        let cell_voxels = 1u32 << (max_depth - level);
        debug_assert!(
            cell_voxels <= 64,
            "LEAF too coarse for naive SN extraction (covers {}^3 finest cells)",
            cell_voxels,
        );
        let slot = leaf_slot(node);
        for dz in 0..cell_voxels {
            for dy in 0..cell_voxels {
                for dx in 0..cell_voxels {
                    let c = IVec3::new(
                        origin.x as i32 + dx as i32,
                        origin.y as i32 + dy as i32,
                        origin.z as i32 + dz as i32,
                    );
                    cells.insert(c, slot);
                }
            }
        }
        return;
    }
    if is_brick(node) {
        let bid = brick_id(node);
        let base = (bid * BRICK_CELLS) as usize;
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                for cx in 0..BRICK_DIM {
                    let flat =
                        (cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                    let v = brick_cells[base + flat];
                    if v == BRICK_EMPTY {
                        continue;
                    }
                    let c = IVec3::new(
                        origin.x as i32 + cx as i32,
                        origin.y as i32 + cy as i32,
                        origin.z as i32 + cz as i32,
                    );
                    let stored = if v == BRICK_INTERIOR { CELL_INTERIOR } else { v };
                    cells.insert(c, stored);
                }
            }
        }
        return;
    }
    if is_branch(node) {
        let children_offset = node as usize;
        let half = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_origin = UVec3::new(
                origin.x + dx * half,
                origin.y + dy * half,
                origin.z + dz * half,
            );
            walk_collect_cells(
                nodes,
                brick_cells,
                children_offset + octant as usize,
                child_origin,
                level + 1,
                max_depth,
                cells,
            );
        }
    }
}

/// Resolve "is the cell at this coord solid?" by descending the octree.
/// Used for cells outside the dense cell map — primarily INTERIOR_NODE
/// regions, which we don't expand to keep memory bounded. Returns false
/// for out-of-bounds coords (the asset extent's exterior is empty).
///
/// O(depth) per call — within a few-cell-thick surface shell this fires
/// only for the small number of EMPTY-side neighbor lookups per surface
/// cell, so the total cost stays proportional to surface area.
fn is_solid_lookup(
    nodes: &[u32],
    brick_cells: &[u32],
    depth: u8,
    coord: IVec3,
    extent: i32,
) -> bool {
    if coord.x < 0
        || coord.y < 0
        || coord.z < 0
        || coord.x >= extent
        || coord.y >= extent
        || coord.z >= extent
    {
        return false;
    }
    let coord_u = UVec3::new(coord.x as u32, coord.y as u32, coord.z as u32);
    let mut idx = 0usize;
    for level in 0..depth {
        let node = nodes[idx];
        if node == EMPTY_NODE {
            return false;
        }
        if node == INTERIOR_NODE {
            return true;
        }
        if is_leaf(node) {
            return true;
        }
        if is_brick(node) {
            // BRICK lives at this level; its cells span `1 << (depth -
            // level)` finest voxels per axis. The flat brick index is
            // the low bits of `coord` modulo that span.
            let span = 1u32 << (depth - level);
            let mask = span - 1;
            let lx = coord_u.x & mask;
            let ly = coord_u.y & mask;
            let lz = coord_u.z & mask;
            let flat = (lx + ly * span + lz * span * span) as usize;
            let v = brick_cells[(brick_id(node) * BRICK_CELLS) as usize + flat];
            return v != BRICK_EMPTY;
        }
        // Branch: descend.
        let half = 1u32 << (depth - level - 1);
        let ox = if coord_u.x & half != 0 { 1u32 } else { 0 };
        let oy = if coord_u.y & half != 0 { 1u32 } else { 0 };
        let oz = if coord_u.z & half != 0 { 1u32 } else { 0 };
        let octant = (ox + oy * 2 + oz * 4) as usize;
        idx = node as usize + octant;
    }
    let node = nodes[idx];
    match node {
        EMPTY_NODE => false,
        INTERIOR_NODE => true,
        n if is_leaf(n) => true,
        _ => false,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_octree::{make_brick, make_leaf};

    #[test]
    fn mesh_vertex_size_is_32() {
        assert_eq!(std::mem::size_of::<MeshVertex>(), 32);
    }

    #[test]
    fn empty_octree_yields_nothing() {
        let nodes = vec![EMPTY_NODE];
        let (verts, indices) = extract_surface_mesh(&nodes, 4, 0.001, Vec3::ZERO, &[], &[], &[]);
        assert!(verts.is_empty());
        assert!(indices.is_empty());
    }

    /// A single LEAF at the root with depth=0 covers exactly one cell.
    /// All 6 neighbors are out-of-bounds (= EMPTY for SN sign), so we
    /// expect a fully-closed unit cube: 8 unique vertices, 6 faces ×
    /// 2 triangles = 12 triangles = 36 indices.
    ///
    /// With naive-SN smoothing each vertex lands at the centroid of
    /// its SN cube's edge crossings — for a single isolated cell the
    /// 8 cubes around it each have exactly 3 sign-change edges
    /// meeting at the cell, and the centroid of those 3 crossings is
    /// at offset (1/3, 1/3, 1/3) from the cube's "near" corner. So
    /// the vertices form an inscribed cube at offsets (1/3·vs,
    /// 2/3·vs) along each axis.
    #[test]
    fn single_leaf_at_root_emits_a_closed_cube() {
        let nodes = vec![make_leaf(7)];
        let vs = 0.5;
        let origin = Vec3::new(1.0, 2.0, 3.0);
        let (verts, indices) = extract_surface_mesh(&nodes, 0, vs, origin, &[], &[], &[]);

        assert_eq!(verts.len(), 8, "8 SN-cube vertices around the unit cell");
        assert_eq!(indices.len(), 36, "6 faces × 2 triangles × 3 indices");

        // Inscribed-cube corners at (1/3 or 2/3) × vs offset on each
        // axis. Order doesn't matter — we sort both lists.
        let third = vs / 3.0;
        let two_thirds = 2.0 * vs / 3.0;
        let mut got: Vec<[f32; 3]> = verts.iter().map(|v| v.local_pos).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected: Vec<[f32; 3]> = (0..8)
            .map(|i| {
                let dx = if (i & 1) != 0 { two_thirds } else { third };
                let dy = if ((i >> 1) & 1) != 0 { two_thirds } else { third };
                let dz = if ((i >> 2) & 1) != 0 { two_thirds } else { third };
                [origin.x + dx, origin.y + dy, origin.z + dz]
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, e) in got.iter().zip(expected.iter()) {
            for k in 0..3 {
                assert!((g[k] - e[k]).abs() < 1e-5, "{:?} != {:?}", g, e);
            }
        }

        // Every vertex should carry the leaf's `leaf_attr_id`.
        for v in &verts {
            assert_eq!(v.leaf_attr_id, 7);
        }
    }

    /// Six face triangles must wind so their cross product points along
    /// the outward axis (+X, -X, +Y, -Y, +Z, -Z). For a single root
    /// leaf the 12 triangles split exactly 2 per axis-direction, with
    /// no inward-facing triangles.
    #[test]
    fn closed_cube_winds_outward() {
        let nodes = vec![make_leaf(0)];
        let (verts, indices) = extract_surface_mesh(&nodes, 0, 1.0, Vec3::ZERO, &[], &[], &[]);
        let mut counts = [0i32; 6]; // +X -X +Y -Y +Z -Z

        for tri in indices.chunks(3) {
            let a = Vec3::from_array(verts[tri[0] as usize].local_pos);
            let b = Vec3::from_array(verts[tri[1] as usize].local_pos);
            let c = Vec3::from_array(verts[tri[2] as usize].local_pos);
            let n = (b - a).cross(c - a).normalize_or_zero();
            // Each cube-face triangle is axis-aligned. Find which axis +/- it points.
            let bucket = if n.x > 0.5 {
                0
            } else if n.x < -0.5 {
                1
            } else if n.y > 0.5 {
                2
            } else if n.y < -0.5 {
                3
            } else if n.z > 0.5 {
                4
            } else if n.z < -0.5 {
                5
            } else {
                panic!("triangle normal not axis-aligned: {:?}", n);
            };
            counts[bucket] += 1;
        }
        // 2 triangles per face × 6 faces — perfectly balanced.
        assert_eq!(counts, [2, 2, 2, 2, 2, 2]);
    }

    /// A brick with one filled cell at (0,0,0) should produce the same
    /// closed-cube mesh as a single root leaf, just at a finer
    /// resolution. Verifies brick traversal and per-cell exposure logic
    /// agree with the leaf path.
    #[test]
    fn single_filled_brick_cell_is_a_unit_cube() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 99;
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[]);
        assert_eq!(verts.len(), 8);
        assert_eq!(indices.len(), 36);
        for v in &verts {
            assert_eq!(v.leaf_attr_id, 99);
        }
    }

    /// Two horizontally-adjacent filled cells share an interior face;
    /// the mesh should *not* emit that face. Total faces = 10 (5 per
    /// cell — 6 cube faces minus the 1 shared face), so we expect
    /// 12 grid-corner vertices (the 12 unique corners of a 2×1×1 box)
    /// and 10 × 2 = 20 triangles = 60 indices.
    #[test]
    fn shared_face_between_adjacent_cells_is_skipped() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1; // (0,0,0)
        bricks[1] = 2; // (1,0,0) — face-adjacent in +X
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[]);
        assert_eq!(verts.len(), 12, "12 unique corners of a 2×1×1 box");
        assert_eq!(indices.len(), 60, "10 exposed faces × 2 triangles × 3 indices");
    }

    /// INTERIOR cells (sentinel inside a brick) must not emit faces
    /// toward each other or toward INTERIOR_NODE-region cells, but must
    /// hide adjacent surface-cell faces. With one surface cell at
    /// (0,0,0) and an INTERIOR neighbor at (1,0,0), the shared +X face
    /// of the surface cell is hidden; we expect 5 exposed surface
    /// faces, no faces from the INTERIOR cell itself.
    #[test]
    fn interior_cells_hide_adjacent_surface_faces() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 5; // (0,0,0) surface
        bricks[1] = BRICK_INTERIOR; // (1,0,0) interior
        let (verts, indices) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[]);
        // Surface cell exposes 5 of 6 faces (+X is hidden by INTERIOR).
        // INTERIOR cell exposes 5 of 6 faces toward EMPTY (-X is hidden
        // by the surface cell, but +X, +Y, -Y, +Z, -Z are exposed to
        // EMPTY since we're in a 2-cell box at the corner of the brick
        // and out-of-brick cells are EMPTY).
        // Total exposed faces = 10. Vertices = 12 (the 2×1×1 box's
        // corners). Indices = 10 × 6 = 60.
        assert_eq!(verts.len(), 12);
        assert_eq!(indices.len(), 60);
    }

    /// INTERIOR_NODE regions must be treated as solid by the on-demand
    /// solidity check. Build a tree where one octant of the root is a
    /// surface BRICK and another is INTERIOR_NODE, sharing a face. The
    /// shared face must be hidden — surface BRICK cells don't emit
    /// faces toward INTERIOR_NODE-region cells.
    #[test]
    fn interior_node_region_is_solid_for_sn_sign() {
        // depth=2 root tree, one branch level. 8 octants.
        // Octant 0 (-X-Y-Z): surface BRICK.
        // Octant 1 (+X-Y-Z): INTERIOR_NODE.
        // Other octants: EMPTY_NODE.
        // With BRICK_DIM=4 and depth=2, each octant covers 1<<(2-1)=2
        // finest voxels per axis — but a BRICK lives at depth-2=0 i.e.
        // at the root. Conflict: BRICK requires being at level 0 with
        // depth-level = BRICK_LEVELS = 2, so root depth 2 with BRICK at
        // root works. But we have a branch at root, so the BRICK lives
        // at level 1 with depth-level=1 — wrong (BRICK needs span 4).
        //
        // Instead: depth=3 root with branch at root. Each octant is at
        // level 1, span 1 << (3-1) = 4 cells per axis = BRICK span. So
        // place a BRICK in octant 0 and INTERIOR_NODE in octant 1.
        let mut nodes = vec![0u32; 9]; // root + 8 children
        nodes[0] = 1; // branch: children at offset 1
        nodes[1] = make_brick(0); // octant 0 (-X-Y-Z)
        nodes[2] = INTERIOR_NODE; // octant 1 (+X-Y-Z)
        nodes[3] = EMPTY_NODE;
        nodes[4] = EMPTY_NODE;
        nodes[5] = EMPTY_NODE;
        nodes[6] = EMPTY_NODE;
        nodes[7] = EMPTY_NODE;
        nodes[8] = EMPTY_NODE;

        // Brick: fill the +X face cells (x=3) with surface, leave rest
        // EMPTY. With INTERIOR_NODE in octant 1 (touching x=4..7), the
        // surface cells at x=3 abut INTERIOR_NODE on their +X side —
        // those +X faces must be hidden.
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        for cz in 0..BRICK_DIM {
            for cy in 0..BRICK_DIM {
                let flat = (3 + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM) as usize;
                bricks[flat] = 11;
            }
        }

        let (verts, indices) = extract_surface_mesh(&nodes, 3, 1.0, Vec3::ZERO, &bricks, &[], &[]);

        // Every triangle must point along an outward axis. Check that
        // *no* triangle points in +X (those would be surface→INTERIOR
        // faces that should have been hidden).
        let mut plus_x_triangles = 0;
        for tri in indices.chunks(3) {
            let a = Vec3::from_array(verts[tri[0] as usize].local_pos);
            let b = Vec3::from_array(verts[tri[1] as usize].local_pos);
            let c = Vec3::from_array(verts[tri[2] as usize].local_pos);
            let n = (b - a).cross(c - a).normalize_or_zero();
            if n.x > 0.5 {
                plus_x_triangles += 1;
            }
        }
        assert_eq!(
            plus_x_triangles, 0,
            "no triangles should face +X — those are surface→INTERIOR_NODE faces"
        );
        // Sanity: we did emit *something* (the other 5 faces of each
        // surface cell are exposed).
        assert!(!indices.is_empty());
    }

    /// Bone weights baked at extract time should match the BoneVoxel
    /// of the surface cell that contributed `leaf_attr_id`. With both
    /// surface slots sharing a single bone (idx 7, weight 255), every
    /// emitted vertex should carry that exact pair — no zeros, no
    /// averaging artifacts. Confirms the extractor reads the parallel
    /// pool by `leaf_attr_id` and the layout matches the VS contract.
    #[test]
    fn vertex_carries_bone_weights_from_chosen_cell() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        let leaf_attrs = vec![LeafAttr::EMPTY; 2];
        let bone_pool = vec![
            BoneVoxel::new([7, 0, 0, 0], [255, 0, 0, 0]),
            BoneVoxel::new([7, 0, 0, 0], [255, 0, 0, 0]),
        ];
        let (verts, _) = extract_surface_mesh(
            &nodes, 2, 1.0, Vec3::ZERO, &bricks, &leaf_attrs, &bone_pool,
        );
        assert!(!verts.is_empty(), "extractor produced no vertices");
        for v in &verts {
            assert_eq!(v.bone_indices, u32::from_le_bytes([7, 0, 0, 0]));
            assert_eq!(v.bone_weights, u32::from_le_bytes([255, 0, 0, 0]));
        }
    }

    /// Empty bone pool → vertices carry zero bone fields. The VS
    /// treats this as "skip skinning, rest pose" (weights sum to 0).
    #[test]
    fn vertex_bone_fields_zero_for_unskinned_assets() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        let (verts, _) = extract_surface_mesh(
            &nodes, 2, 1.0, Vec3::ZERO, &bricks, &[], &[],
        );
        assert!(!verts.is_empty());
        for v in &verts {
            assert_eq!(v.bone_indices, 0);
            assert_eq!(v.bone_weights, 0);
        }
    }

    /// Vertex normal averaging: with two surface cells sharing a
    /// vertex, both contributing identical +Y normals, the vertex
    /// should pack to +Y after averaging.
    #[test]
    fn vertex_normal_averaging_uses_leaf_attr_pool() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 0;
        bricks[1] = 1;
        // LeafAttr pool with two slots, both pointing +Y.
        let pool = vec![
            LeafAttr {
                normal_oct: pack_oct(Vec3::Y),
                material_primary: 0,
                material_secondary_blend: 0,
            },
            LeafAttr {
                normal_oct: pack_oct(Vec3::Y),
                material_primary: 0,
                material_secondary_blend: 0,
            },
        ];
        let (verts, _) = extract_surface_mesh(&nodes, 2, 1.0, Vec3::ZERO, &bricks, &pool, &[]);
        for v in &verts {
            let n = unpack_oct(v.normal_oct);
            assert!((n - Vec3::Y).length() < 1e-3, "expected +Y, got {:?}", n);
        }
    }

    // ── Phase B R4b — region-scoped extract ─────────────────────────

    /// Triangle multiset keyed by sorted vertex-position triple. Used to
    /// compare triangle sets across different VBO orderings.
    fn triangle_position_set(
        indices: &[u32],
        verts: &[MeshVertex],
    ) -> std::collections::HashMap<[[i32; 3]; 3], usize> {
        let mut m = std::collections::HashMap::new();
        for tri in indices.chunks_exact(3) {
            let mut p = [
                pos_key(verts[tri[0] as usize].local_pos),
                pos_key(verts[tri[1] as usize].local_pos),
                pos_key(verts[tri[2] as usize].local_pos),
            ];
            p.sort();
            *m.entry(p).or_insert(0) += 1;
        }
        m
    }

    fn pos_key(p: [f32; 3]) -> [i32; 3] {
        [
            (p[0] * 1000.0) as i32,
            (p[1] * 1000.0) as i32,
            (p[2] * 1000.0) as i32,
        ]
    }

    fn region_extent(extent: i32) -> (IVec3, IVec3) {
        (IVec3::ZERO, IVec3::splat(extent))
    }

    /// Region covering the whole asset should produce the same triangle
    /// set as a full-asset extract.
    #[test]
    fn region_extract_matches_full_extract_on_full_region() {
        // 4×4×4 brick: two adjacent surface cells + a couple of others.
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1; // (0,0,0)
        bricks[1] = 2; // (1,0,0)
        bricks[BRICK_DIM as usize * BRICK_DIM as usize] = 3; // (0,0,1)

        let depth = 2u8;
        let extent = 1i32 << depth;
        let (full_v, full_i) =
            extract_surface_mesh(&nodes, depth, 1.0, Vec3::ZERO, &bricks, &[], &[]);
        let (region_v, region_i) = extract_surface_mesh_region(
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::ZERO,
            IVec3::splat(extent),
        );
        let _ = region_extent(extent);
        assert_eq!(
            triangle_position_set(&full_i, &full_v),
            triangle_position_set(&region_i, &region_v),
            "region covering full extent must match full-extract triangle set",
        );
    }

    /// Region far from any solid cell yields nothing (or a degenerate
    /// empty mesh).
    #[test]
    fn region_extract_in_empty_space_yields_nothing() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        let (v, i) = extract_surface_mesh_region(
            &nodes,
            2,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::splat(20),
            IVec3::splat(25),
        );
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    /// Region scoped to the cell containing the single solid voxel
    /// emits the closed-cube mesh. Pad ensures the cell's 6 face quads
    /// are all produced.
    #[test]
    fn region_extract_capturing_one_cell_emits_full_cube() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 7; // (0,0,0)
        let (v, i) = extract_surface_mesh_region(
            &nodes,
            2,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::ZERO,
            IVec3::ONE, // half-open [0, 1) — covers only cell (0,0,0)
        );
        assert_eq!(v.len(), 8, "single cell → 8 cube vertices");
        assert_eq!(i.len(), 36, "6 faces × 2 triangles × 3 indices");
    }

    /// Region restricted to a subset is exactly the subset of triangles
    /// that the full extract would emit for cells in the padded subset.
    /// Build a 2×1×1 box (cells (0,0,0) and (1,0,0)). Region [0..1)
    /// (padded [-1..2)) covers both cells, since cell (1,0,0) is one
    /// past region but inside the pad. Region [3..4) misses entirely.
    #[test]
    fn region_extract_subset_includes_padded_neighbors() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1; // (0,0,0)
        bricks[1] = 2; // (1,0,0)
        let depth = 2u8;
        // Full extract for reference (10 faces × 2 = 20 tris).
        let (full_v, full_i) =
            extract_surface_mesh(&nodes, depth, 1.0, Vec3::ZERO, &bricks, &[], &[]);
        let full_tris = full_i.len() / 3;
        assert_eq!(full_tris, 20, "2-cell box has 10 exposed faces × 2 tris");

        // Region [0..1) padded to [-1..2). Includes both cells (0,0,0)
        // and (1,0,0) (since (1,0,0) is at the +X edge of pad).
        let (v_a, i_a) = extract_surface_mesh_region(
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::ZERO,
            IVec3::ONE,
        );
        assert_eq!(
            triangle_position_set(&i_a, &v_a),
            triangle_position_set(&full_i, &full_v),
            "pad expansion of region [0..1) must reach cell (1,0,0)",
        );

        // Region [3..4) padded to [2..5) — neither solid cell is in pad.
        let (v_b, i_b) = extract_surface_mesh_region(
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::splat(3),
            IVec3::splat(4),
        );
        assert!(
            v_b.is_empty() && i_b.is_empty(),
            "region far from solids must emit nothing"
        );
    }

    /// `collect_cell_map` + `extract_mesh_region_from_cells` produce the
    /// same result as `extract_surface_mesh_region` — the convenience
    /// wrapper is just sugar over the two-step form.
    #[test]
    fn two_step_form_matches_convenience_wrapper() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        bricks[1] = 2;
        bricks[BRICK_DIM as usize * BRICK_DIM as usize] = 3;
        let depth = 2u8;
        let region_min = IVec3::ZERO;
        let region_max = IVec3::splat(2);

        let (v1, i1) = extract_surface_mesh_region(
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            region_min,
            region_max,
        );
        let cells = collect_cell_map(&nodes, depth, &bricks);
        let (v2, i2) = extract_mesh_region_from_cells(
            &cells,
            region_min,
            region_max,
            &nodes,
            depth,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
        );
        assert_eq!(
            triangle_position_set(&i1, &v1),
            triangle_position_set(&i2, &v2),
        );
    }

    /// Empty region (min == max on any axis) returns nothing.
    #[test]
    fn region_extract_empty_region_returns_nothing() {
        let nodes = vec![make_brick(0)];
        let mut bricks = vec![BRICK_EMPTY; BRICK_CELLS as usize];
        bricks[0] = 1;
        let (v, i) = extract_surface_mesh_region(
            &nodes,
            2,
            1.0,
            Vec3::ZERO,
            &bricks,
            &[],
            &[],
            IVec3::splat(2),
            IVec3::splat(2), // empty
        );
        assert!(v.is_empty());
        assert!(i.is_empty());
    }
}
