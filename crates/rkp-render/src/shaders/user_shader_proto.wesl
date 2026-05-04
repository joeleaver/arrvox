// user_shader_proto.wgsl — Option B prototype voxelizer.
//
// One-time bake of a user shader's static prototype shape into a small
// dedicated octree+brick+leaf-attr triple. Produces the canonical
// per-shader voxelization that every instance the shader emits will
// reuse at march time.
//
// ## Pipeline
//
// CPU pre-builds the prototype's INTERNAL octree levels (levels
// 0..max_depth-1) at allocation time — they're a fixed dense
// structure determined entirely by `max_depth`, no per-shader
// classification is needed for the spine. Each internal node points
// to its 8 children at the next level. The bake fills only the
// LEAF level (level == max_depth), which corresponds to brick-parent
// nodes.
//
// One workgroup per leaf-level octree slot, dispatched as
// `dispatch_workgroups(2^max_depth, 2^max_depth, 2^max_depth)`. The
// 3D workgroup id IS the brick coord in [0, 2^max_depth)³ and the
// leaf-level slot index is its Morton encoding (matches the
// octant-recursive layout the CPU pre-builder uses).
//
// Each workgroup runs `workgroup_size(4, 4, 4)` so its 64 threads
// cover one (4-cell)³ brick in lock-step. uvw ∈ [0, 1]³ in canonical
// prototype-local space; `dispatch_user_proto(shader_id, uvw)`
// returns occupancy + normal + material.
//
// Workgroup-cooperative deferred brick allocation mirrors
// `brick_fill_main` in `user_shader_geom.wgsl`: every thread votes
// "any cell emitted?", and thread 0 only claims a brick slot from
// the prototype's brick block when the vote is positive. All-empty
// bricks consume zero brick / leaf-attr slots.
//
// ## Compose contract
//
// The Rust composer splices each user shader's `proto` body and the
// `dispatch_user_proto` switch between the BEGIN/END markers below.

import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_LEAF_BIT;
import package::lib::octree_slot::OCTREE_BRICK_BIT;
import package::lib::octree_slot::OCTREE_BAND_BIT;
import package::lib::octree_slot::OCTREE_PAYLOAD_MASK;
import package::lib::types::LeafAttr;
import package::lib::types::VoxelEmit;
import package::lib::brick::BRICK_DIM;
import package::lib::brick::BRICK_CELLS;
import package::lib::brick::BRICK_CELL_EMPTY;
import package::lib::brick::INTERNAL_ATTR_NONE;


// Overflow counter slots — must match the `OVERFLOW_*` indices the
// Rust pass writes into the same buffer.
const OVERFLOW_BRICK: u32 = 1u;
const OVERFLOW_LEAF_ATTR: u32 = 2u;


// One prototype to bake this dispatch. The Rust pass writes one of
// these per dirty shader and dispatches with it bound at group(1).
//
// `octree_leaf_offset` is the absolute pool index of the FIRST
// leaf-level octree slot for this prototype (CPU pre-builds the
// internal levels at `octree_block_offset + level_starts[max_depth]`).
// The bake's workgroups land at `octree_leaf_offset + morton(wid)`.
//
// Bricks and leaf-attrs are allocated from GLOBAL atomic cursors in
// `GlobalCursors` — they're persistent across bake dispatches so
// successive prototype bakes accumulate slots. `brick_capacity` and
// `leaf_attr_capacity` are the proto-pool's total slot budgets, used
// as overflow gates so an over-budget bake degrades to OCTREE_EMPTY
// at the offending cells rather than corrupting downstream pools.
struct PrototypeUniform {
    shader_id: u32,
    max_depth: u32,
    octree_leaf_offset: u32,
    brick_capacity: u32,
    leaf_attr_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// GLOBAL bricks + leaf-attrs cursors. Single-pair atomic counters bumped
// once per emitted brick / leaf-attr across the entire proto pool.
// Different prototypes interleave; the march follows octree → brick_id
// → leaf_attr_id pointers, which are absolute, so layout doesn't matter.
struct GlobalCursors {
    brick: atomic<u32>,
    leaf_attr: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> octree_nodes: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read_write> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(3) var<storage, read_write> cursors: GlobalCursors;
@group(0) @binding(4) var<storage, read_write> overflow: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> proto_u: PrototypeUniform;

// Workgroup-shared deferred-allocation state — same pattern as
// `brick_fill_main` in `user_shader_geom.wgsl`.
var<workgroup> wg_any_emit: atomic<u32>;
var<workgroup> wg_brick_offset: u32;
var<workgroup> wg_alloc_failed: atomic<u32>;

fn pack_oct(n: vec3<f32>) -> u32 {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    var nx = n.x / max(l1, 1e-8);
    var ny = n.y / max(l1, 1e-8);
    if (n.z < 0.0) {
        let ax = (1.0 - abs(ny)) * select(-1.0, 1.0, nx >= 0.0);
        let ay = (1.0 - abs(nx)) * select(-1.0, 1.0, ny >= 0.0);
        nx = ax;
        ny = ay;
    }
    let ix = u32(i32(round(clamp(nx, -1.0, 1.0) * 32767.0)) & 0xFFFF);
    let iy = u32(i32(round(clamp(ny, -1.0, 1.0) * 32767.0)) & 0xFFFF);
    return ix | (iy << 16u);
}

fn voxel_emit_skip() -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;
    v.normal = vec3<f32>(0.0, 1.0, 0.0);
    v.material_primary = 0u;
    v.material_secondary = 0u;
    v.blend_weight = 0u;
    return v;
}

// Morton-encode a 3D brick coord into a leaf-level octree slot index.
// At level k from the root the (cx, cy, cz) of the descent is bit
// `max_depth-1-k` of (bx, by, bz). The pre-built internal levels lay
// out children as `octant = cx | (cy << 1) | (cz << 2)`, so the leaf's
// linear position is `morton(bx, by, bz)`.
fn morton_3d(b: vec3<u32>, max_depth: u32) -> u32 {
    var result: u32 = 0u;
    for (var k: u32 = 0u; k < max_depth; k = k + 1u) {
        let bit = max_depth - 1u - k;
        let cx = (b.x >> bit) & 1u;
        let cy = (b.y >> bit) & 1u;
        let cz = (b.z >> bit) & 1u;
        let octant = cx | (cy << 1u) | (cz << 2u);
        result = (result << 3u) | octant;
    }
    return result;
}

const USER_PROTO_DISPATCH_BEGIN: u32 = 0u;
// Default identity stub — the Rust composer replaces this block with
// the concatenated user-shader bodies + `dispatch_user_proto` switch
// when any registered shader provides a `proto` hook. Empty-registry
// path keeps this stub so the pipeline always validates.
fn dispatch_user_proto(shader_id: u32, uvw: vec3<f32>) -> VoxelEmit {
    return voxel_emit_skip();
}
const USER_PROTO_DISPATCH_END: u32 = 0u;

@compute @workgroup_size(4, 4, 4)
fn proto_bake_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let max_depth = proto_u.max_depth;
    let bricks_per_axis = 1u << max_depth;
    let cells_per_axis = bricks_per_axis * BRICK_DIM;

    // uvw — sample at cell center in [0, 1]³ canonical prototype space.
    let cell_3d = wid * BRICK_DIM + lid;
    let uvw = (vec3<f32>(cell_3d) + vec3<f32>(0.5)) / f32(cells_per_axis);

    let cell_idx = lid.z * BRICK_DIM * BRICK_DIM + lid.y * BRICK_DIM + lid.x;
    let leaf_slot_idx = morton_3d(wid, max_depth);
    let leaf_octree_offset = proto_u.octree_leaf_offset + leaf_slot_idx;

    if (cell_idx == 0u) {
        atomicStore(&wg_any_emit, 0u);
        atomicStore(&wg_alloc_failed, 0u);
        wg_brick_offset = 0u;
        // Pre-write OCTREE_EMPTY at the leaf slot. Thread 0 below
        // overwrites with a brick reference iff any cell emitted.
        // Mirrors the brick_fill_main pattern — required for
        // deterministic re-bakes (free old extents, the leaf level
        // would carry stale brick references otherwise).
        octree_nodes[leaf_octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
    }
    workgroupBarrier();

    let emit = dispatch_user_proto(proto_u.shader_id, uvw);
    if (emit.occupancy != 0u) {
        atomicStore(&wg_any_emit, 1u);
    }
    workgroupBarrier();

    // Cooperative deferred brick allocation. Thread 0 claims a brick
    // slot iff at least one cell voted yes; otherwise leave the leaf
    // octree slot at OCTREE_EMPTY. The brick id is GLOBAL — bumped from
    // the cross-bake `cursors.brick` counter.
    if (cell_idx == 0u) {
        if (atomicLoad(&wg_any_emit) != 0u) {
            let brick_id = atomicAdd(&cursors.brick, 1u);
            if (brick_id >= proto_u.brick_capacity) {
                atomicAdd(&overflow[OVERFLOW_BRICK], 1u);
                atomicStore(&wg_alloc_failed, 1u);
            } else {
                wg_brick_offset = brick_id * BRICK_CELLS;
                octree_nodes[leaf_octree_offset] = vec2<u32>(
                    OCTREE_LEAF_BIT | OCTREE_BRICK_BIT | brick_id,
                    INTERNAL_ATTR_NONE,
                );
            }
        } else {
            atomicStore(&wg_alloc_failed, 1u);
        }
    }
    workgroupBarrier();

    if (atomicLoad(&wg_alloc_failed) != 0u) {
        return;
    }

    let brick_slot_idx = wg_brick_offset + cell_idx;
    if (emit.occupancy == 0u) {
        brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
        return;
    }
    let global_slot = atomicAdd(&cursors.leaf_attr, 1u);
    if (global_slot >= proto_u.leaf_attr_capacity) {
        atomicAdd(&overflow[OVERFLOW_LEAF_ATTR], 1u);
        brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
        return;
    }
    var attr: LeafAttr;
    attr.normal_oct = pack_oct(emit.normal);
    let pri = emit.material_primary & 0xFFFFu;
    let sec = (emit.material_secondary & 0x0FFFu) << 16u;
    let bw = (emit.blend_weight & 0x0Fu) << 28u;
    attr.material_packed = pri | sec | bw;
    leaf_attr_pool[global_slot] = attr;
    brick_pool[brick_slot_idx] = global_slot;
}
