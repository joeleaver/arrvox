// Procedural → triangle-mesh on GPU via naive surface nets.
//
// Spike: skip the octree + brick + DAG + meshlet stack entirely. Take
// a flattened procedural tree's bytecode (same `instructions` stream
// that the BFS voxel bake feeds to `proc_sample`), evaluate it on a
// dense `N³` cell grid, and emit a triangle mesh that follows the
// surface defined by `eval_tree_distance(p) <= 0`.
//
// Three compute entry points share the bind group:
//   1. `classify`    — sample SDF at every cell center, write u32 (1=solid, 0=empty).
//   2. `vertex_emit` — per SN-cube corner, if active emit one ProxyVertex
//                      with edge-crossing centroid + 6-tap gradient normal
//                      + per-vertex shading payload from `eval_tree`.
//   3. `index_emit`  — per cell × 3 +-axis edges, if sign-change emit a
//                      6-index quad referencing the 4 SN-cube vertices around it.
//
// Conventions match `arvx_core::mesh_extract`:
//   - SN cube at integer corner `q ∈ [0..N+1)` owns the 8 cells at
//     `q-1..q` along each axis. Out-of-bounds cells = EMPTY.
//   - Vertex world position: average of cell-center midpoints over the
//     cube's active edges. For an isolated solid cell that's the
//     classic inscribed cube at offsets (1/3, 2/3) of `cell_size`.
//   - Triangle winding CCW about the outward normal (solid → empty).

import package::lib::types::ProcInstruction;
import package::lib::oct_normal::pack_oct_normal;
import package::proc_eval::eval_tree;
import package::proc_eval::eval_tree_distance;
import package::proc_eval::gradient_normal;

// Pipeline-override constant — declared per-consumer (see comment in
// `proc_eval_types.wesl`). `proc_eval`'s `eval_tree_distance` references
// this as a free name; the Rust side names it at pipeline creation.
override HAS_POS_WARPS: bool = true;

struct SnParams {
    // World-space lo corner of the procedural's AABB. Cell `(i,j,k)`
    // center sits at `aabb_min + (vec3<f32>(i,j,k) + 0.5) * cell_size`.
    aabb_min: vec3<f32>,
    cell_size: f32,

    // Cells per axis (`grid_n`); cube grid is `(grid_n + 1)³`.
    grid_n: u32,
    instruction_count: u32,
    vertex_cap: u32,
    index_cap: u32,
}

// 32 bytes — matches `arvx_core::mesh_extract::ProxyVertex` layout.
// Proxy meshes have no octree and no LeafAttr indirection: the full
// shading payload (material primary/secondary/blend + color) is
// baked per vertex at extraction time and consumed directly by the
// proxy raster's FS.
struct ProxyVertex {
    local_pos_x: f32,
    local_pos_y: f32,
    local_pos_z: f32,
    normal_oct: u32,
    // bits  0-15: primary material_id (u16)
    // bits 16-27: secondary material_id (u12)
    // bits 28-31: blend_weight (u4)
    material_packed: u32,
    // R | G<<8 | B<<16 | 0xFF<<24 — high byte forced to 0xFF so a
    // black procedural color doesn't collide with the
    // "no override" sentinel used by the voxel color path.
    color_packed: u32,
    _reserved0: u32,
    _reserved1: u32,
}

@group(0) @binding(0) var<uniform> params: SnParams;
@group(0) @binding(1) var<storage, read> instructions: array<ProcInstruction>;
// Cell occupancy: `1u` = solid, `0u` = empty. Sized `grid_n³`.
@group(0) @binding(2) var<storage, read_write> cell_solid: array<u32>;
// Maps cube-corner index → vertex slot. `0xFFFFFFFFu` if cube is inactive.
// Sized `(grid_n + 1)³`.
@group(0) @binding(3) var<storage, read_write> cube_vertex_map: array<u32>;
// `counters[0]` = vertex_count, `counters[1]` = index_count. One
// binding with two atomics lets us avoid the
// `min_storage_buffer_offset_alignment` (typically 256 B) constraint
// that two separate bindings into adjacent u32 offsets would hit.
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>, 2>;
@group(0) @binding(5) var<storage, read_write> vertices: array<ProxyVertex>;
@group(0) @binding(6) var<storage, read_write> indices: array<u32>;

const SOLID: u32 = 1u;
const EMPTY: u32 = 0u;
const NO_VERT: u32 = 0xFFFFFFFFu;

// ── Cell + cube indexing helpers ─────────────────────────────────────

fn cell_storage_index(c: vec3<i32>) -> u32 {
    let n = i32(params.grid_n);
    return u32(c.x + c.y * n + c.z * n * n);
}

fn cube_storage_index(q: vec3<i32>) -> u32 {
    let n = i32(params.grid_n) + 1;
    return u32(q.x + q.y * n + q.z * n * n);
}

fn read_cell_solid(c: vec3<i32>) -> u32 {
    let n = i32(params.grid_n);
    if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= n || c.y >= n || c.z >= n) {
        return EMPTY;
    }
    return cell_solid[cell_storage_index(c)];
}

// Cube-corner → cell offset. Bit 0 = +X, bit 1 = +Y, bit 2 = +Z.
// SN-cube at corner `q` owns cells `q + corner_offset(i) + (-1, -1, -1)`
// (so corner 7 = (1,1,1) maps to cell `q`, corner 0 = (0,0,0) maps to
// cell `q - 1`). This matches the CPU mesh_extract convention where
// `cube` is the cube's lo-corner cell and `corner_offset(i)` selects
// the cell from there.
fn corner_offset(i: u32) -> vec3<i32> {
    return vec3<i32>(
        i32(i & 1u),
        i32((i >> 1u) & 1u),
        i32((i >> 2u) & 1u),
    );
}

// ── Pass 1: classify cells ───────────────────────────────────────────

@compute @workgroup_size(4, 4, 4)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.grid_n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let center = params.aabb_min
               + (vec3<f32>(gid) + vec3<f32>(0.5)) * params.cell_size;
    let d = eval_tree_distance(center, params.instruction_count);
    let solid = select(EMPTY, SOLID, d <= 0.0);
    cell_solid[cell_storage_index(vec3<i32>(gid))] = solid;
}

// ── Pass 2: emit vertex per active SN cube ──────────────────────────

@compute @workgroup_size(4, 4, 4)
fn vertex_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cube_n = params.grid_n + 1u;
    if (gid.x >= cube_n || gid.y >= cube_n || gid.z >= cube_n) { return; }
    // Cube q's 8 corner cells are at `q + corner_offset(i) - (1,1,1)`.
    // (Equivalent to cells `q-1..q` per axis.)
    let q = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    let cell_base = q - vec3<i32>(1, 1, 1);

    var corner_solid: array<u32, 8>;
    var any_solid = false;
    var any_empty = false;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let s = read_cell_solid(cell_base + corner_offset(k));
        corner_solid[k] = s;
        if (s == SOLID) { any_solid = true; }
        else            { any_empty = true; }
    }
    if (!(any_solid && any_empty)) {
        cube_vertex_map[cube_storage_index(q)] = NO_VERT;
        return;
    }

    // Walk the 12 cube edges. Each edge is a pair of corner cells
    // adjacent along one axis. Active edges contribute their
    // cell-center midpoint to the centroid sum.
    //
    // Cell center (in cell-coord units relative to the AABB) for cell
    // `c` is `c + 0.5`. Cube `q`'s corner i maps to cell
    // `cell_base + corner_offset(i)`. Edge midpoint:
    //   `0.5 * ((cell_a + 0.5) + (cell_b + 0.5))`
    // = `0.5 * (cell_a + cell_b) + 0.5`
    // = `cell_base + 0.5*(corner_offset(a) + corner_offset(b)) + 0.5`.

    var crossing_sum = vec3<f32>(0.0);
    var crossing_count: u32 = 0u;

    // 4 X-edges (corners differ in bit 0)
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let a_idx = (b & 1u) | ((b & 2u) << 1u); // 0,1,4,5 → corner pairs (0,1),(2,3),(4,5),(6,7)
        let b_idx = a_idx | 1u;
        if (corner_solid[a_idx] != corner_solid[b_idx]) {
            let oa = corner_offset(a_idx);
            let ob = corner_offset(b_idx);
            crossing_sum = crossing_sum + vec3<f32>(cell_base) + (vec3<f32>(oa) + vec3<f32>(ob)) * 0.5 + vec3<f32>(0.5);
            crossing_count = crossing_count + 1u;
        }
    }
    // 4 Y-edges (bit 1)
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let a_idx = (b & 1u) | ((b & 2u) << 1u); // 0,1,4,5 → (0,2),(1,3),(4,6),(5,7)
        let b_idx = a_idx | 2u;
        if (corner_solid[a_idx] != corner_solid[b_idx]) {
            let oa = corner_offset(a_idx);
            let ob = corner_offset(b_idx);
            crossing_sum = crossing_sum + vec3<f32>(cell_base) + (vec3<f32>(oa) + vec3<f32>(ob)) * 0.5 + vec3<f32>(0.5);
            crossing_count = crossing_count + 1u;
        }
    }
    // 4 Z-edges (bit 2)
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        let a_idx = b; // 0,1,2,3 → (0,4),(1,5),(2,6),(3,7)
        let b_idx = a_idx | 4u;
        if (corner_solid[a_idx] != corner_solid[b_idx]) {
            let oa = corner_offset(a_idx);
            let ob = corner_offset(b_idx);
            crossing_sum = crossing_sum + vec3<f32>(cell_base) + (vec3<f32>(oa) + vec3<f32>(ob)) * 0.5 + vec3<f32>(0.5);
            crossing_count = crossing_count + 1u;
        }
    }

    // Defensive: an SN cube with mixed solidity always has at least
    // one active edge. Fall back to the cube's nominal hub if not.
    let local_centroid = select(
        vec3<f32>(cell_base) + vec3<f32>(1.0),
        crossing_sum / max(f32(crossing_count), 1.0),
        crossing_count > 0u,
    );
    let pos_world = params.aabb_min + local_centroid * params.cell_size;

    // Gradient normal — `cell_size` is wide enough that the 6-tap
    // central differences land in well-defined SDF territory rather
    // than chasing finite-precision noise at the hit point.
    let n = gradient_normal(pos_world, params.instruction_count, params.cell_size);
    let normal_oct = pack_oct_normal(n);

    // Full TreeSample at the vertex — primary/secondary material, blend
    // weight, and procedural color. Pack into the vertex so the proxy
    // raster's FS reads shading data directly with no LeafAttr lookup.
    let s = eval_tree(pos_world, params.instruction_count);

    // Match `proc_sample.wesl`'s blend quantization (round-to-nearest u4).
    let blend_u4 = u32(trunc(clamp(s.blend_weight, 0.0, 1.0) * 15.0 + 0.5));

    // Material packing layout — must match `LeafAttr.material_packed`
    // as decoded by `mesh_resolve.wesl` (primary lo16, secondary
    // mid12, blend hi4) so downstream packing matches the existing
    // G-buffer material consumer.
    let material_packed =
        (s.material_id & 0xFFFFu)
        | ((s.secondary_material_id & 0xFFFu) << 16u)
        | ((blend_u4 & 0xFu) << 28u);

    // RGBA8 color packing — high byte 0xFF avoids the "no override"
    // sentinel (=0) the voxel color path uses.
    let cr = u32(clamp(s.color.r, 0.0, 1.0) * 255.0 + 0.5);
    let cg = u32(clamp(s.color.g, 0.0, 1.0) * 255.0 + 0.5);
    let cb = u32(clamp(s.color.b, 0.0, 1.0) * 255.0 + 0.5);
    let color_packed = cr | (cg << 8u) | (cb << 16u) | (0xFFu << 24u);

    let slot = atomicAdd(&counters[0], 1u);
    if (slot >= params.vertex_cap) {
        // Out of vertex space — leave map empty so index_emit skips
        // any quad referencing this cube.
        cube_vertex_map[cube_storage_index(q)] = NO_VERT;
        return;
    }

    var v: ProxyVertex;
    v.local_pos_x = pos_world.x;
    v.local_pos_y = pos_world.y;
    v.local_pos_z = pos_world.z;
    v.normal_oct = normal_oct;
    v.material_packed = material_packed;
    v.color_packed = color_packed;
    v._reserved0 = 0u;
    v._reserved1 = 0u;
    vertices[slot] = v;
    cube_vertex_map[cube_storage_index(q)] = slot;
}

// ── Pass 3: emit triangle quads ──────────────────────────────────────
// One thread per cell, walks its three +axis neighbors. Each sign-change
// edge contributes a quad of 4 SN-cube vertex indices around it. We only
// look in +axis directions so each axis edge is visited by exactly one
// thread (the lower-coord side).

// CCW-around-outward-normal cube offsets for each face direction:
//   face 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.
// Mirror of `CUBE_OFFSETS_PER_FACE` in `arvx_core::mesh_extract`. Indices
// resolve to SN-cube grid coords relative to the *solid* cell of the edge.
// Since cube q's 8 corner cells are `q-1..q`, a SN-cube at offset
// `(0,-1,-1)` from cell `c` means cube q = `c + (1, 0, 0)` (its lo-cell
// corner is the cell at offset `(0,-1,-1)`).
fn face_cube_offset(face: u32, k: u32) -> vec3<i32> {
    // Layout: face × 4 entries. Hand-transcribed from the CPU table.
    if (face == 0u) { // +X — CCW about +X is +Y → +Z
        if (k == 0u) { return vec3<i32>(1, 0, 0); }
        if (k == 1u) { return vec3<i32>(1, 1, 0); }
        if (k == 2u) { return vec3<i32>(1, 1, 1); }
        return vec3<i32>(1, 0, 1);
    }
    if (face == 1u) { // -X
        if (k == 0u) { return vec3<i32>(0, 0, 0); }
        if (k == 1u) { return vec3<i32>(0, 0, 1); }
        if (k == 2u) { return vec3<i32>(0, 1, 1); }
        return vec3<i32>(0, 1, 0);
    }
    if (face == 2u) { // +Y — CCW about +Y is +Z → +X
        if (k == 0u) { return vec3<i32>(0, 1, 0); }
        if (k == 1u) { return vec3<i32>(0, 1, 1); }
        if (k == 2u) { return vec3<i32>(1, 1, 1); }
        return vec3<i32>(1, 1, 0);
    }
    if (face == 3u) { // -Y
        if (k == 0u) { return vec3<i32>(0, 0, 0); }
        if (k == 1u) { return vec3<i32>(1, 0, 0); }
        if (k == 2u) { return vec3<i32>(1, 0, 1); }
        return vec3<i32>(0, 0, 1);
    }
    if (face == 4u) { // +Z — CCW about +Z is +X → +Y
        if (k == 0u) { return vec3<i32>(0, 0, 1); }
        if (k == 1u) { return vec3<i32>(1, 0, 1); }
        if (k == 2u) { return vec3<i32>(1, 1, 1); }
        return vec3<i32>(0, 1, 1);
    }
    // face == 5u: -Z
    if (k == 0u) { return vec3<i32>(0, 0, 0); }
    if (k == 1u) { return vec3<i32>(0, 1, 0); }
    if (k == 2u) { return vec3<i32>(1, 1, 0); }
    return vec3<i32>(1, 0, 0);
}

fn emit_quad_for_edge(self_cell: vec3<i32>, axis: u32, self_is_solid: bool) {
    // Translate to "solid cell + +axis face" canonical form.
    var solid_cell = self_cell;
    var face: u32;
    if (self_is_solid) {
        face = 2u * axis; // +X / +Y / +Z
    } else {
        // Self is empty, neighbor is solid. Walk from the neighbor; the
        // empty side from its perspective is -axis.
        var step: vec3<i32>;
        if (axis == 0u) { step = vec3<i32>(1, 0, 0); }
        else if (axis == 1u) { step = vec3<i32>(0, 1, 0); }
        else                 { step = vec3<i32>(0, 0, 1); }
        solid_cell = self_cell + step;
        face = 2u * axis + 1u;
    }

    let cube_n = i32(params.grid_n) + 1;
    var quad: array<u32, 4>;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let off = face_cube_offset(face, k);
        let cube_pos = solid_cell + off;
        if (cube_pos.x < 0 || cube_pos.y < 0 || cube_pos.z < 0
         || cube_pos.x >= cube_n || cube_pos.y >= cube_n || cube_pos.z >= cube_n) {
            return;
        }
        let v = cube_vertex_map[cube_storage_index(cube_pos)];
        if (v == NO_VERT) { return; }
        quad[k] = v;
    }

    let base = atomicAdd(&counters[1], 6u);
    if (base + 6u > params.index_cap) { return; }
    indices[base + 0u] = quad[0];
    indices[base + 1u] = quad[1];
    indices[base + 2u] = quad[2];
    indices[base + 3u] = quad[0];
    indices[base + 4u] = quad[2];
    indices[base + 5u] = quad[3];
}

@compute @workgroup_size(4, 4, 4)
fn index_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.grid_n;
    if (gid.x >= n || gid.y >= n || gid.z >= n) { return; }
    let cell = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    let s_self = read_cell_solid(cell);

    // +X
    if (gid.x + 1u < n) {
        let s_nx = read_cell_solid(cell + vec3<i32>(1, 0, 0));
        if (s_self != s_nx) {
            emit_quad_for_edge(cell, 0u, s_self == SOLID);
        }
    }
    // +Y
    if (gid.y + 1u < n) {
        let s_ny = read_cell_solid(cell + vec3<i32>(0, 1, 0));
        if (s_self != s_ny) {
            emit_quad_for_edge(cell, 1u, s_self == SOLID);
        }
    }
    // +Z
    if (gid.z + 1u < n) {
        let s_nz = read_cell_solid(cell + vec3<i32>(0, 0, 1));
        if (s_self != s_nz) {
            emit_quad_for_edge(cell, 2u, s_self == SOLID);
        }
    }
}
