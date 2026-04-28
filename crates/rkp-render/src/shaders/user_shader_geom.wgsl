// user_shader_geom.wgsl — GPU runtime octree builder for user shaders.
//
// Phase C V2: hierarchical dense-brick octree at configurable depth N.
// Each region materializes a perfect octree of depth N with bricks at
// the deepest level. Total cells per region = (4 * 2^N)^3:
//   depth=0  →  4³ = 64 cells (V1)
//   depth=1  →  8³ = 512 cells
//   depth=2  → 16³ = 4 096 cells (V2 default)
//   depth=3  → 32³ = 32 768 cells
//   depth=4  → 64³ = 262 144 cells
//
// Internal octree nodes (levels 0..N-1) are pre-written by the CPU
// when allocating the cache slot — they're a deterministic perfect
// tree, no GPU work needed. The GPU compute pass writes:
//   * One octree LEAF node per brick (level-N pointer to the brick).
//   * 64 cells per brick (BRICK_CELL_EMPTY / a leaf_attr_id).
//   * One LeafAttr per occupied cell, atomically dispensed from the
//     region's reserved range.
//
// Workgroups dispatched per region: (2^N, 2^N, 2^N). Each workgroup is
// 4×4×4 threads, one thread per brick cell. Workgroup id (wid) selects
// which brick within the region. The brick's octree leaf offset is
// computed via Morton-order traversal so it lines up with the perfect
// tree's level-N layout.

const BRICK_DIM: u32 = 4u;
const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu;

struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

// HOST_SAMPLE_BEGIN — V3 implementation. Descends the host's octree
// (read-only, sharing the same pool the march reads) at world_pos and
// returns:
//   .distance — Lipschitz-bounded signed distance to the host
//               surface. Negative inside, positive outside, ~0 on
//               surface. Magnitude is conservative.
//   .normal   — leaf normal at the queried point if the cell is
//               occupied; +Y default otherwise.
//
// `host_octree_root == HOST_NO_HOST_SENTINEL` (0xFFFFFFFF) means the
// region has no host (free-standing). Returns (+inf, +Y) so user code
// that gates on `host.distance < threshold` won't spuriously fill cells.
const HOST_NO_HOST_SENTINEL: u32 = 0xFFFFFFFFu;
struct HostSample {
    distance: f32,
    normal: vec3<f32>,
    /// V7 — primary material id of the host's leaf at the queried
    /// point, when the descent terminates at a LEAF / BRICK cell.
    /// 0 for EMPTY / INTERIOR terminations and out-of-bounds queries.
    material: u32,
    /// V7b — secondary material id and blend weight from the leaf's
    /// material_packed. Non-zero `blend_weight` means the surface is
    /// partially painted with `material_secondary` over the
    /// primary. Shaders that gate on "painted with my id" should
    /// check `material == X || (material_secondary == X && blend_weight > 0)`.
    material_secondary: u32,
    blend_weight: u32,
}
fn unpack_oct(packed: u32) -> vec3<f32> {
    let ix = i32(packed & 0xFFFFu);
    let iy = i32((packed >> 16u) & 0xFFFFu);
    // Sign-extend from 16-bit to 32-bit.
    let sx = select(ix - 0x10000, ix, ix < 0x8000);
    let sy = select(iy - 0x10000, iy, iy < 0x8000);
    var x = clamp(f32(sx) / 32767.0, -1.0, 1.0);
    var y = clamp(f32(sy) / 32767.0, -1.0, 1.0);
    let z = 1.0 - abs(x) - abs(y);
    if (z < 0.0) {
        let ax = (1.0 - abs(y)) * select(-1.0, 1.0, x >= 0.0);
        let ay = (1.0 - abs(x)) * select(-1.0, 1.0, y >= 0.0);
        x = ax;
        y = ay;
    }
    return normalize(vec3<f32>(x, y, z));
}
fn host_sample_at(world_pos: vec3<f32>) -> HostSample {
    var s: HostSample;
    s.distance = 1e30;
    s.normal = vec3<f32>(0.0, 1.0, 0.0);
    s.material = 0u;
    s.material_secondary = 0u;
    s.blend_weight = 0u;
    if (region.host_octree_root == HOST_NO_HOST_SENTINEL) {
        return s;
    }
    // World → object-local via inverse_world.
    let local4 = region.host_inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz / max(local4.w, 1e-12);
    // Octree-local position: subtract grid origin; octree spans
    // [0, host_octree_extent] on each axis.
    let oc = local - region.host_grid_origin;
    let extent = region.host_octree_extent;
    if (oc.x < 0.0 || oc.y < 0.0 || oc.z < 0.0
        || oc.x > extent || oc.y > extent || oc.z > extent) {
        // Outside the octree's bounds — Lipschitz lower bound on
        // distance is the nearest face of the octree's AABB.
        let to_box = max(max(-oc, oc - vec3<f32>(extent)), vec3<f32>(0.0));
        s.distance = length(to_box);
        return s;
    }
    // Descend the octree. At each branch, pick the child octant
    // containing `pos`. Track the level's spatial extent so when we
    // bottom out at EMPTY/INTERIOR we can bound the distance.
    var offset = region.host_octree_root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    let max_levels = region.host_octree_depth + 8u; // bricks dive deeper
    for (var i: u32 = 0u; i < max_levels; i = i + 1u) {
        let pair = octree_nodes[offset];
        let value = pair.x;
        if (value == OCTREE_EMPTY) {
            // EMPTY at level L means the entire cell of half-extent
            // `half` is empty. Surface is somewhere OUTSIDE this
            // cell, so signed distance to surface ≥ pos's inset
            // distance from the cell's nearest face. That inset =
            // -distance_to_local_box (positive when pos is inside).
            s.distance = max(0.0, -distance_to_local_box(oc, center, half));
            return s;
        }
        if (value == 0xFFFFFFFEu) { // OCTREE_INTERIOR
            // INTERIOR: pos is inside a fully-solid region. Distance
            // is NEGATIVE; magnitude bounded by the inset distance.
            s.distance = min(0.0, distance_to_local_box(oc, center, half));
            return s;
        }
        let is_leaf = (value & OCTREE_LEAF_BIT) != 0u;
        let is_brick = is_leaf && ((value & OCTREE_BRICK_BIT) != 0u);
        if (is_brick) {
            // Descend into the brick to find the specific cell.
            let brick_id = value & 0x3FFFFFFFu;
            let cell_size_at = (half * 2.0) / f32(BRICK_DIM);
            let brick_min = center - vec3<f32>(half);
            let pos_in_brick = oc - brick_min;
            let cx = u32(clamp(floor(pos_in_brick.x / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cy = u32(clamp(floor(pos_in_brick.y / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cz = u32(clamp(floor(pos_in_brick.z / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cell_idx = cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx;
            let cell = brick_pool[brick_id * BRICK_CELLS + cell_idx];
            // Cell's own AABB for the Lipschitz bound — same EMPTY /
            // INTERIOR derivation as the octree-level case, just at
            // brick-cell granularity.
            let cell_center = brick_min
                + vec3<f32>(f32(cx), f32(cy), f32(cz)) * cell_size_at
                + vec3<f32>(cell_size_at * 0.5);
            let cell_half = cell_size_at * 0.5;
            if (cell == BRICK_CELL_EMPTY) {
                s.distance = max(0.0, -distance_to_local_box(oc, cell_center, cell_half));
                return s;
            }
            if (cell == BRICK_CELL_INTERIOR) {
                s.distance = min(0.0, distance_to_local_box(oc, cell_center, cell_half));
                // INTERIOR cells have no leaf_attr to read material
                // from. Walk this brick's other cells looking for a
                // LEAF — its material is a representative for the
                // painted region (paint typically covers contiguous
                // surface cells, so any LEAF in the same brick
                // shares the painted material). Lets grass shaders
                // gate on `host.material == ctx.material_id` for
                // cells whose down-probe lands inside the host body.
                var rep_primary: u32 = 0u;
                var rep_secondary: u32 = 0u;
                var rep_blend: u32 = 0u;
                for (var i: u32 = 0u; i < BRICK_CELLS; i = i + 1u) {
                    let other = brick_pool[brick_id * BRICK_CELLS + i];
                    if (other != BRICK_CELL_EMPTY && other != BRICK_CELL_INTERIOR) {
                        let other_attr = leaf_attr_pool[other];
                        rep_primary = other_attr.material_packed & 0xFFFFu;
                        rep_secondary = (other_attr.material_packed >> 16u) & 0x0FFFu;
                        rep_blend = (other_attr.material_packed >> 28u) & 0x0Fu;
                        break;
                    }
                }
                s.material = rep_primary;
                s.material_secondary = rep_secondary;
                s.blend_weight = rep_blend;
                return s;
            }
            // Occupied surface cell — read leaf normal + materials.
            let attr = leaf_attr_pool[cell];
            s.distance = 0.0;
            s.normal = unpack_oct(attr.normal_oct);
            s.material = attr.material_packed & 0xFFFFu;
            s.material_secondary = (attr.material_packed >> 16u) & 0x0FFFu;
            s.blend_weight = (attr.material_packed >> 28u) & 0x0Fu;
            return s;
        }
        if (is_leaf) {
            let attr = leaf_attr_pool[value & 0x3FFFFFFFu];
            s.distance = 0.0;
            s.normal = unpack_oct(attr.normal_oct);
            s.material = attr.material_packed & 0xFFFFu;
            s.material_secondary = (attr.material_packed >> 16u) & 0x0FFFu;
            s.blend_weight = (attr.material_packed >> 28u) & 0x0Fu;
            return s;
        }
        // Branch — descend.
        let cx = select(0u, 1u, oc.x >= center.x);
        let cy = select(0u, 1u, oc.y >= center.y);
        let cz = select(0u, 1u, oc.z >= center.z);
        let octant = cx + cy * 2u + cz * 4u;
        offset = value + octant;
        half = half * 0.5;
        center = vec3<f32>(
            center.x + select(-half, half, cx == 1u),
            center.y + select(-half, half, cy == 1u),
            center.z + select(-half, half, cz == 1u),
        );
    }
    // Hit max iterations without termination — return on-surface as
    // the safest fallback (ensures the user shader sees a finite,
    // close-to-zero distance rather than a stale +inf).
    s.distance = 0.0;
    return s;
}

// L∞ distance from `pos` to the cube centered at `box_center` with
// half-extent `box_half`. 0 if inside.
fn distance_to_local_box(pos: vec3<f32>, box_center: vec3<f32>, box_half: f32) -> f32 {
    let d = abs(pos - box_center) - vec3<f32>(box_half);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
// HOST_SAMPLE_END

struct UserCtx {
    time: f32,
    cell_size: f32,
    material_id: u32,
    aabb_min: vec3<f32>,
    params: array<f32, 8>,
}

struct VoxelEmit {
    occupancy: u32,
    normal: vec3<f32>,
    material_primary: u32,
    material_secondary: u32,
    blend_weight: u32,
}

struct RegionUniform {
    aabb_min: vec3<f32>,
    cell_size: f32,
    aabb_max: vec3<f32>,
    shader_id: u32,
    octree_offset: u32,
    brick_offset: u32,
    leaf_attr_offset: u32,
    leaf_attr_capacity: u32,
    brick_capacity: u32,
    _pad_brick_cap: u32,
    time: f32,
    material_id: u32,
    region_index: u32,
    // Octree depth N. 0 = single-brick root (V1), 2 = 16 cells/axis,
    // etc. Determines workgroup dispatch shape.
    depth: u32,
    // Pre-computed at uniform-build time so the shader avoids
    // pow(8, depth) and `(pow(8, depth) - 1) / 7` per thread.
    // bricks_per_axis = 2^depth.
    bricks_per_axis: u32,
    // level_n_start = (8^N - 1) / 7. Octree leaf nodes (brick pointers)
    // sit at indices [region.octree_offset + level_n_start ..
    //                 region.octree_offset + level_n_start + 8^N).
    level_n_start: u32,
    // V3 — host octree info for `host_sample_at` queries. The geom
    // pipeline shares the read-write `octree_nodes`/`brick_pool`/
    // `leaf_attr_pool` bindings with the march, so reads against the
    // host's slice come for free as long as we know the host's
    // root/depth/extent. `host_octree_root == 0xFFFFFFFF` = no host.
    host_octree_root: u32,
    host_octree_depth: u32,
    host_octree_extent: f32,
    region_thickness: f32,
    _pad_thickness: f32,
    _pad_thickness2: f32,
    _pad_thickness3: f32,
    _pad_thickness4: f32,
    host_grid_origin: vec3<f32>,
    _pad_grid: f32,
    params: array<vec4<f32>, 2>,
    host_inverse_world: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read_write> octree_nodes: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read_write> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(3) var<storage, read_write> leaf_attr_alloc: array<atomic<u32>>;
// V5 — atomic counter per region for sparse brick allocation. The
// per-region brick reservation is smaller than `8^depth` (the dense
// worst case); workgroups whose brick survives the proximity gate
// `atomicAdd` here to claim a slot. Overflow falls back to skip
// (OCTREE_EMPTY at that octree leaf).
@group(0) @binding(4) var<storage, read_write> brick_alloc: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> region: RegionUniform;

// Workgroup-shared early-out flag: thread 0 queries the host at the
// brick center, decides whether the entire brick is far enough from
// the host surface to skip user shader calls. All other threads read
// this after a barrier and either early-out (writing OCTREE_EMPTY for
// the brick) or proceed with normal cell evaluation.
var<workgroup> wg_brick_skip: u32;
// V5 — atomically-claimed brick slot index within this region's
// brick reservation. Set by thread 0 and read by all other threads
// after the barrier. Only valid when `wg_brick_skip == 0u`.
var<workgroup> wg_brick_slot: u32;

fn voxel_emit_skip() -> VoxelEmit {
    var v: VoxelEmit;
    v.occupancy = 0u;
    v.normal = vec3<f32>(0.0, 1.0, 0.0);
    v.material_primary = 0u;
    v.material_secondary = 0u;
    v.blend_weight = 0u;
    return v;
}

// Octahedral normal pack — same convention as `rkp_core::leaf_attr::pack_oct`.
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

// Morton-encode a brick's (wx, wy, wz) coordinate (each in [0, 2^N)) to
// its index in the octree's level-N flat array. Each level's child
// uses (cx + cy*2 + cz*4) ordering — same as the march descent —
// processing the highest bit first.
fn morton_brick_idx(w: vec3<u32>, depth: u32) -> u32 {
    var idx: u32 = 0u;
    var d: u32 = 0u;
    loop {
        if (d >= depth) { break; }
        let shift = depth - 1u - d;
        let cx = (w.x >> shift) & 1u;
        let cy = (w.y >> shift) & 1u;
        let cz = (w.z >> shift) & 1u;
        let octant = cx + cy * 2u + cz * 4u;
        idx = idx * 8u + octant;
        d = d + 1u;
    }
    return idx;
}

// USER_GENERATE_DISPATCH_BEGIN
// Default identity stub — the Rust composer replaces this block with the
// concatenated user-shader bodies + `dispatch_user_generate` switch when
// any registered shader provides a `generate` hook. The empty-registry
// path keeps this stub so the pipeline always validates.
fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
    return voxel_emit_skip();
}
// USER_GENERATE_DISPATCH_END

// 4×4×4 = 64 threads per workgroup → one brick. The Rust pass
// dispatches (2^depth)³ workgroups per region.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let cell_idx = lid.z * BRICK_DIM * BRICK_DIM + lid.y * BRICK_DIM + lid.x;
    let brick_world_size = region.cell_size * f32(BRICK_DIM);
    let brick_origin = region.aabb_min + vec3<f32>(wid) * brick_world_size;
    let cell_world_pos =
        brick_origin
        + vec3<f32>(lid) * region.cell_size
        + vec3<f32>(region.cell_size * 0.5);

    let brick_idx_in_region = morton_brick_idx(wid, region.depth);
    let leaf_node_offset =
        region.octree_offset + region.level_n_start + brick_idx_in_region;

    // V4 + V5 — thread 0 runs the brick proximity gate AND, on pass,
    // atomically claims a brick slot from the region's sparse pool.
    // Skip / overflow → OCTREE_EMPTY at the leaf. Otherwise the
    // claimed slot's brick_id is encoded into the leaf node and shared
    // with the rest of the workgroup via `wg_brick_slot`.
    if (cell_idx == 0u) {
        var skip = 0u;
        if (region.region_thickness > 0.0) {
            let brick_center = brick_origin + vec3<f32>(brick_world_size * 0.5);
            let host = host_sample_at(brick_center);
            // sqrt(3)/2 ≈ 0.866 — half the brick's space diagonal in
            // L2, the worst case for L2 distance from center to corner.
            let lipschitz = region.region_thickness + brick_world_size * 0.866;
            if (host.distance > lipschitz) {
                skip = 1u;
            }
            // Don't gate on material at brick granularity — the brick
            // center typically sits in EMPTY space above the host
            // surface, where `host_sample` returns material=0 even
            // when the surface below is painted. Per-cell material
            // checks handle the actual emission gate.
        }
        if (skip == 0u) {
            // Claim a brick slot from the region's reserved range.
            let claimed = atomicAdd(&brick_alloc[region.region_index], 1u);
            if (claimed >= region.brick_capacity) {
                // Reserve overflow — fall back to skip. User can
                // shrink octree_depth or tighten region_thickness.
                skip = 1u;
            } else {
                wg_brick_slot = claimed;
                let brick_id = (region.brick_offset / BRICK_CELLS) + claimed;
                octree_nodes[leaf_node_offset] = vec2<u32>(
                    OCTREE_LEAF_BIT | OCTREE_BRICK_BIT | brick_id,
                    INTERNAL_ATTR_NONE,
                );
            }
        }
        if (skip == 1u) {
            octree_nodes[leaf_node_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
        }
        wg_brick_skip = skip;
    }
    workgroupBarrier();

    if (wg_brick_skip == 1u) {
        return;
    }

    var ctx: UserCtx;
    ctx.time = region.time;
    ctx.cell_size = region.cell_size;
    ctx.material_id = region.material_id;
    ctx.aabb_min = region.aabb_min;
    ctx.params[0] = region.params[0].x;
    ctx.params[1] = region.params[0].y;
    ctx.params[2] = region.params[0].z;
    ctx.params[3] = region.params[0].w;
    ctx.params[4] = region.params[1].x;
    ctx.params[5] = region.params[1].y;
    ctx.params[6] = region.params[1].z;
    ctx.params[7] = region.params[1].w;

    let host = host_sample_at(cell_world_pos);
    let emit = dispatch_user_generate(region.shader_id, cell_world_pos, host, ctx);

    // Sparse — brick offset comes from the atomically-claimed slot,
    // not the Morton-encoded position. Claimed slots are tightly
    // packed at the start of the region's reservation regardless of
    // their spatial brick_idx_in_region.
    let brick_offset_global = region.brick_offset + wg_brick_slot * BRICK_CELLS;
    let brick_slot_idx = brick_offset_global + cell_idx;

    if (emit.occupancy == 0u) {
        brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
    } else {
        // Allocate a leaf_attr slot from this region's reserved range
        // via the per-region atomic counter (indexed by region_index).
        // V1 used wid.x — broken when a region dispatches more than
        // one workgroup; V2 uses region_index from the uniform so
        // the counter is shared correctly across all bricks of one
        // region but not across regions.
        let local_id = atomicAdd(&leaf_attr_alloc[region.region_index], 1u);
        if (local_id >= region.leaf_attr_capacity) {
            // Pool exhaustion — shader produced more occupied cells
            // than the slot's reserved capacity holds. Drop the cell.
            // A future tuning pass can refuse the slot earlier or
            // grow capacity dynamically; for V2 we just lose detail.
            brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
        } else {
            let global_slot = region.leaf_attr_offset + local_id;
            var attr: LeafAttr;
            attr.normal_oct = pack_oct(emit.normal);
            let pri = emit.material_primary & 0xFFFFu;
            let sec = (emit.material_secondary & 0x0FFFu) << 16u;
            let bw = (emit.blend_weight & 0x0Fu) << 28u;
            attr.material_packed = pri | sec | bw;
            leaf_attr_pool[global_slot] = attr;
            brick_pool[brick_slot_idx] = global_slot;
        }
    }

    // Octree leaf node was already written by thread 0 in the gate
    // block above — sparse allocation needs the BRICK pointer to
    // reflect the atomically-claimed slot, not the Morton-encoded
    // position, so it can't be deferred to a post-fill barrier.
}
