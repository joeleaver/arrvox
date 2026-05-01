// user_shader_geom.wgsl — sparse BFS GPU octree builder.
//
// Builds the whole tree top-down by atomically allocating nodes / bricks /
// leaf-attrs / fill-tasks from PER-REGION blocks in the global pools.
// Each cached region owns variable-size extents (allocated CPU-side from
// a free-list bucketed allocator) and bumps within them via per-region
// atomic counters. Memory and compute scale with painted surface area;
// per-region brick locality is preserved (cache-friendly march).
//
// Allocator model: four `array<atomic<u32>>` counters indexed by
// `region_index` — one each for `octree_alloc`, `brick_alloc`,
// `leaf_attr_alloc`, `fill_task_alloc`. Every classify / fill thread
// that needs a slot does `atomicAdd(&pool_alloc[region_index], n)` and
// composes the global pool offset as `region.X_block_offset + slot`.
// CPU pre-seeds counters before dispatch (octree=1, others=0) and
// pre-fills topology-dirty regions' fill-task extents with sentinels
// so the fill pass can early-out on unused slots.
//
// Pipeline (per frame):
//   1. CPU walks the persistent cache, gathers DIRTY regions
//      (topology_dirty | fill_dirty), assigns each a 0-based
//      `region_index_this_frame`, builds the regions storage array.
//   2. For TOPOLOGY-DIRTY regions only: CPU writes one root ActiveCell
//      per region into active_queue[level=0] with
//      `octree_offset = region.octree_block_offset` and sets
//      `active_count[0] = topology_dirty_count`.
//   3. CPU resets per-region counters: octree=1, brick=0, leaf_attr=0,
//      fill_task=0 (per topology-dirty region; fill-only regions
//      preserve their cached fill-task extent and reset only
//      brick/leaf_attr).
//   4. For L in 0..=max_depth: dispatch `classify_main` over
//      active_queue[L]. Each thread samples host_sample_at, classifies
//      the cell, and:
//        - EMPTY → write OCTREE_EMPTY, done.
//        - INTERIOR → write OCTREE_EMPTY (transient regions don't carry
//          host material; rendering INTERIOR as a solid block is wrong).
//        - L < max_depth and MIXED → atomicAdd 8 from the region's
//          octree block, write self as a branch, push children to L+1.
//        - L == max_depth and MIXED → atomicAdd 1 from the region's
//          fill-task block, write a BrickFillTask there. Pre-write
//          OCTREE_EMPTY at the cell (fill decides whether to allocate
//          a brick via V12 deferred allocation).
//   5. Dispatch `brick_fill_main` per fill-dirty region. Workgroup
//      `(task_idx, region_idx)` reads
//      `fill_task_pool[region.fill_task_block_offset + task_idx]`.
//      Sentinel slots early-out. Workgroup-cooperative deferred
//      allocation: one brick from the region's brick block iff at
//      least one cell emits.
//
// Overflow handling: when an allocator atomicAdd would exceed the
// region's block capacity, the offending node degrades to OCTREE_EMPTY
// and increments a global counter in the `overflow` binding (CPU reads
// asynchronously and logs).
//
// Compose contract preserved: the Rust composer splices the user
// shader's `dispatch_user_generate` body between the BEGIN/END
// markers below (their literal names live further down the file
// to avoid polluting `find`).

const BRICK_DIM: u32 = 4u;
const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
// Phase B-redux 3b — band-cell sentinel. Set on a leaf at max_depth
// when an `instance_at` shader baked this region. The leaf-attr slot
// reinterprets as `{region_id, anchor_leaf_slot}`. See
// octree_march.wgsl for the canonical encoding contract.
const OCTREE_BAND_BIT: u32 = 0x20000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x1FFFFFFFu;
const INTERNAL_ATTR_NONE: u32 = 0xFFFFFFFFu;

// Hard ceiling on octree depth — limits per-frame queue capacity.
// `MAX_DEPTH = 8` gives 4 × 2^8 = 1024 cells/axis at the deepest level
// per region — fine enough that most paint workflows never hit it.
const MAX_DEPTH: u32 = 8u;
// sqrt(3) — half the L2 length of a unit cube's diagonal. Used to
// inflate `half_extent` into the conservative L2 distance from cell
// center to its furthest corner, for the Lipschitz proximity classifier.
const SQRT3: f32 = 1.7320508075688772f;

const HOST_NO_HOST_SENTINEL: u32 = 0xFFFFFFFFu;

struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

struct HostSample {
    distance: f32,
    normal: vec3<f32>,
    material: u32,
    material_secondary: u32,
    blend_weight: u32,
}

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

// One "active" octree node awaiting classification at level L. The Rust
// side seeds active_queue[level=0] with one root cell per region; each
// classify dispatch reads its level's slice and pushes children to
// level L+1.
struct ActiveCell {
    octree_offset: u32,
    region_index: u32,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    half_extent: f32,
    _pad0: u32,
    _pad1: u32,
}

// One "fill this brick" task pushed by classify_main when a leaf-level
// cell survives the proximity gate. brick_fill_main consumes these.
struct BrickFillTask {
    octree_offset: u32,
    region_index: u32,
    // u32-element index into `brick_pool` of this brick's first cell.
    brick_offset: u32,
    cell_size: f32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    _pad: u32,
}

struct RegionUniform {
    aabb_min: vec3<f32>,
    cell_size: f32,
    aabb_max: vec3<f32>,
    shader_id: u32,
    max_depth: u32,
    time: f32,
    material_id: u32,
    region_thickness: f32,
    host_octree_root: u32,
    host_octree_depth: u32,
    host_octree_extent: f32,
    // Per-region pool block offsets and sizes. *Offsets* are absolute
    // GPU-buffer indices in pool-native units (octree nodes /
    // BRICKS / LeafAttr slots / FillTask slots). *Sizes* are counts
    // in those same units — the bucket allocator ensures they are
    // powers of 2.
    octree_block_offset: u32,
    octree_block_size: u32,
    brick_block_offset: u32,
    brick_block_size: u32,
    leaf_attr_block_offset: u32,
    leaf_attr_block_size: u32,
    fill_task_block_offset: u32,
    fill_task_block_size: u32,
    // Pad so host_grid_origin (vec3, 16-byte aligned) lands at the
    // next 16-aligned offset.
    _pad_host: u32,
    host_grid_origin: vec3<f32>,
    // Pad so params (vec4) lands at next 16-aligned offset.
    _pad_grid: f32,
    params: array<vec4<f32>, 2>,
    host_inverse_world: mat4x4<f32>,
}

// Per-dispatch state — written by the host once per classify call.
struct LevelUniform {
    current_level: u32,
    per_level_cap: u32,
    max_active_per_level: u32,
    _pad: u32,
}

// Overflow counter slots. Must match `OverflowSlot` on the Rust side.
const OVERFLOW_OCTREE: u32 = 0u;
const OVERFLOW_BRICK: u32 = 1u;
const OVERFLOW_LEAF_ATTR: u32 = 2u;
const OVERFLOW_FILL_QUEUE: u32 = 3u;
// Per-level active-queue overflow lives at OVERFLOW_ACTIVE_QUEUE_BASE + L
// for L in [0, MAX_DEPTH].
const OVERFLOW_ACTIVE_QUEUE_BASE: u32 = 4u;

// Sentinel value stored in unused fill_task_pool slots so fill_main
// can early-out without needing to know the per-region count. CPU
// pre-fills topology-dirty regions' extents with this before classify.
const FILL_TASK_SENTINEL: u32 = 0xFFFFFFFEu;

@group(0) @binding(0) var<storage, read_write> octree_nodes: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read_write> leaf_attr_pool: array<LeafAttr>;
// Per-region atomic counters — array length = MAX_REGIONS on Rust side.
@group(0) @binding(3) var<storage, read_write> octree_alloc: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> brick_alloc: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> leaf_attr_alloc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> active_queue: array<ActiveCell>;
@group(0) @binding(7) var<storage, read_write> active_count: array<atomic<u32>>;
// Persistent fill-task pool — owned entirely by the user-shader pass.
// Each region writes/reads from its own block within this buffer.
@group(0) @binding(8) var<storage, read_write> fill_task_pool: array<BrickFillTask>;
// Per-region count of fill tasks emitted by classify into the region's
// fill_task block. Used by classify (atomicAdd to claim a slot) and
// by fill (read to bound iteration; sentinel-checking is the primary
// guard).
@group(0) @binding(9) var<storage, read_write> fill_task_alloc: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> overflow: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> regions: array<RegionUniform>;

@group(2) @binding(0) var<uniform> level_u: LevelUniform;

// Workgroup-shared region uniform. brick_fill_main's thread 0 copies
// this from `regions[task.region_index]` before the barrier so the
// rest of the workgroup AND any user code called from
// `dispatch_user_generate` can reference fields like
// `region.host_octree_extent` / `region.cell_size` directly — keeping
// the user-facing API identical to V4-V8 (where `region` was a
// dynamic-offset uniform binding).
//
// classify_main does NOT use this — its threads each process a
// potentially different region, so they read into per-thread locals
// (named `cur_region` to avoid shadowing this workgroup global).
var<workgroup> region: RegionUniform;

// V12 — workgroup-shared state for deferred brick allocation in
// brick_fill_main. Cells evaluate the user shader, then the
// workgroup decides whether to allocate a brick at all (skip if no
// cell emitted). Eliminates wasted brick slots for all-empty bricks.
var<workgroup> wg_any_emit: atomic<u32>;
var<workgroup> wg_brick_offset: u32;
var<workgroup> wg_alloc_failed: atomic<u32>;

fn unpack_oct(packed: u32) -> vec3<f32> {
    let ix = i32(packed & 0xFFFFu);
    let iy = i32((packed >> 16u) & 0xFFFFu);
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

// Signed L∞-as-L2 distance from `pos` to the cube centered at `c` with
// half-extent `h`. Negative inside.
fn distance_to_local_box(pos: vec3<f32>, c: vec3<f32>, h: f32) -> f32 {
    let d = abs(pos - c) - vec3<f32>(h);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// host_sample variant that takes the region by index — used by classify
// threads (each may process a different region in the same workgroup).
fn host_sample_in_region(world_pos: vec3<f32>, region_index: u32) -> HostSample {
    var s: HostSample;
    s.distance = 1e30;
    s.normal = vec3<f32>(0.0, 1.0, 0.0);
    s.material = 0u;
    s.material_secondary = 0u;
    s.blend_weight = 0u;
    let region = regions[region_index];
    if (region.host_octree_root == HOST_NO_HOST_SENTINEL) {
        return s;
    }
    let local4 = region.host_inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz / max(local4.w, 1e-12);
    let oc = local - region.host_grid_origin;
    let extent = region.host_octree_extent;
    if (oc.x < 0.0 || oc.y < 0.0 || oc.z < 0.0
        || oc.x > extent || oc.y > extent || oc.z > extent) {
        let to_box = max(max(-oc, oc - vec3<f32>(extent)), vec3<f32>(0.0));
        s.distance = length(to_box);
        return s;
    }
    var offset = region.host_octree_root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    let max_levels = region.host_octree_depth + 8u;
    for (var i: u32 = 0u; i < max_levels; i = i + 1u) {
        let pair = octree_nodes[offset];
        let value = pair.x;
        if (value == OCTREE_EMPTY) {
            s.distance = max(0.0, -distance_to_local_box(oc, center, half));
            return s;
        }
        if (value == OCTREE_INTERIOR) {
            s.distance = min(0.0, distance_to_local_box(oc, center, half));
            return s;
        }
        let is_leaf = (value & OCTREE_LEAF_BIT) != 0u;
        let is_brick = is_leaf && ((value & OCTREE_BRICK_BIT) != 0u);
        if (is_brick) {
            let brick_id = value & 0x3FFFFFFFu;
            let cell_size_at = (half * 2.0) / f32(BRICK_DIM);
            let brick_min = center - vec3<f32>(half);
            let pos_in_brick = oc - brick_min;
            let cx = u32(clamp(floor(pos_in_brick.x / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cy = u32(clamp(floor(pos_in_brick.y / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cz = u32(clamp(floor(pos_in_brick.z / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cell_idx = cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx;
            let cell = brick_pool[brick_id * BRICK_CELLS + cell_idx];
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
                var rep_primary: u32 = 0u;
                var rep_secondary: u32 = 0u;
                var rep_blend: u32 = 0u;
                for (var j: u32 = 0u; j < BRICK_CELLS; j = j + 1u) {
                    let other = brick_pool[brick_id * BRICK_CELLS + j];
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
    s.distance = 0.0;
    return s;
}

// Wrapper used by user code from inside `dispatch_user_generate`.
// Reads the workgroup-shared `region` that brick_fill_main's thread 0
// initialised before the barrier. The body inlines the descent
// (rather than calling `host_sample_in_region` with a region_index)
// because brick_fill threads share one region — copying from
// workgroup memory each call would be cheap, but reading
// `regions[task.region_index]` per thread is wasteful when we
// already have it in workgroup storage.
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
    let local4 = region.host_inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz / max(local4.w, 1e-12);
    let oc = local - region.host_grid_origin;
    let extent = region.host_octree_extent;
    if (oc.x < 0.0 || oc.y < 0.0 || oc.z < 0.0
        || oc.x > extent || oc.y > extent || oc.z > extent) {
        let to_box = max(max(-oc, oc - vec3<f32>(extent)), vec3<f32>(0.0));
        s.distance = length(to_box);
        return s;
    }
    var offset = region.host_octree_root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    let max_levels = region.host_octree_depth + 8u;
    for (var i: u32 = 0u; i < max_levels; i = i + 1u) {
        let pair = octree_nodes[offset];
        let value = pair.x;
        if (value == OCTREE_EMPTY) {
            s.distance = max(0.0, -distance_to_local_box(oc, center, half));
            return s;
        }
        if (value == OCTREE_INTERIOR) {
            s.distance = min(0.0, distance_to_local_box(oc, center, half));
            return s;
        }
        let is_leaf = (value & OCTREE_LEAF_BIT) != 0u;
        let is_brick = is_leaf && ((value & OCTREE_BRICK_BIT) != 0u);
        if (is_brick) {
            let brick_id = value & 0x3FFFFFFFu;
            let cell_size_at = (half * 2.0) / f32(BRICK_DIM);
            let brick_min = center - vec3<f32>(half);
            let pos_in_brick = oc - brick_min;
            let cx = u32(clamp(floor(pos_in_brick.x / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cy = u32(clamp(floor(pos_in_brick.y / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cz = u32(clamp(floor(pos_in_brick.z / cell_size_at), 0.0, f32(BRICK_DIM - 1u)));
            let cell_idx = cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx;
            let cell = brick_pool[brick_id * BRICK_CELLS + cell_idx];
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
                var rep_primary: u32 = 0u;
                var rep_secondary: u32 = 0u;
                var rep_blend: u32 = 0u;
                for (var j: u32 = 0u; j < BRICK_CELLS; j = j + 1u) {
                    let other = brick_pool[brick_id * BRICK_CELLS + j];
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
    s.distance = 0.0;
    return s;
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

// USER_GENERATE_DISPATCH_BEGIN
// Default identity stub — the Rust composer replaces this block with the
// concatenated user-shader bodies + `dispatch_user_generate` switch when
// any registered shader provides a `generate` hook. The empty-registry
// path keeps this stub so the pipeline always validates.
fn dispatch_user_generate(shader_id: u32, cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
    return voxel_emit_skip();
}
// USER_GENERATE_DISPATCH_END

// Classify pass — workgroup_size 64. One thread per active cell at level
// `level_u.current_level`. Threads with gid.x past the level's count
// early-out (saves us a separate "build indirect args" dispatch pass at
// the cost of always launching MAX_QUEUE_CAP/64 workgroups per level).
@compute @workgroup_size(64)
fn classify_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let L = level_u.current_level;
    let cap = level_u.per_level_cap;
    let count = atomicLoad(&active_count[L]);
    if (gid.x >= count) {
        return;
    }
    let queue_idx = L * cap + gid.x;
    let cell = active_queue[queue_idx];
    // Per-thread local copy. Named `cur_region` so it doesn't shadow
    // the module-scope workgroup global `region` (used by user code
    // in brick_fill_main).
    let cur_region = regions[cell.region_index];
    let center = vec3<f32>(cell.center_x, cell.center_y, cell.center_z);
    let half = cell.half_extent;
    let host = host_sample_in_region(center, cell.region_index);

    // Free-standing regions (no host) skip the proximity gate — the
    // shader is expected to emit cells based on its own logic, not on
    // proximity to a host surface. Otherwise:
    //   Lipschitz-conservative proximity classifier. The cell is a cube
    //   of half-extent `half` (L∞); furthest point from center within
    //   the cell is at L2 distance `half * sqrt(3)`. If host's distance
    //   lower bound exceeds this plus the shader's region_thickness,
    //   no point in the cell can be within the band the shader cares
    //   about → EMPTY. Conversely, a cell entirely inside the host
    //   body and beyond the band is INTERIOR.
    if (cur_region.host_octree_root != HOST_NO_HOST_SENTINEL) {
        let cell_diag_half = half * SQRT3;
        let band = cur_region.region_thickness;
        // Cell is too far OUTSIDE the host body to be in the band.
        if (host.distance > cell_diag_half + band) {
            octree_nodes[cell.octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
            return;
        }
        // Cell is INSIDE the host body — surface-effect shaders
        // (grass, fur, moss) emit nothing here. The proximity gate's
        // older `< -cell_diag_half - band` threshold left a thick
        // sub-surface band that produced bricks the user shader
        // returned occupancy=0 for; using `< -cell_diag_half`
        // (any cell whose Lipschitz bound says it's fully inside)
        // halves the band volume without losing surface-straddling
        // cells (those have distance in [-half_diag, +half_diag]
        // and pass through to MIXED).
        //
        // Mark EMPTY (not INTERIOR): writing OCTREE_INTERIOR for
        // transient regions renders as a solid voxel block in the
        // march pass (transient regions don't carry the entity-
        // level material slot needed for INTERIOR shading).
        if (host.distance < -cell_diag_half) {
            octree_nodes[cell.octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
            return;
        }
    }

    if (L == cur_region.max_depth) {
        // V11 — paint-targeted brick allocation. Skip bricks whose
        // projected surface point isn't painted with this region's
        // material. This is what makes paint-driven shaders (grass,
        // moss, fur) actually scale: instead of allocating bricks
        // for every cell within `region_thickness` of the surface,
        // we allocate only for bricks above painted surface area.
        //
        // Skipped only when region.material_id != 0 AND there's a
        // host. Free-standing or material-agnostic regions go
        // through the existing alloc path. The projection here
        // mirrors what the user shader's body does — sphere-trace
        // down -Y from the brick center, find the surface, read the
        // material — but cheaply, with one host_sample at an
        // estimated surface point rather than an iterative trace.
        // No early material-match check. An earlier revision tried
        // to skip bricks whose projected surface point wasn't
        // painted with the region's material — but the projection
        // relied on `host.distance` (a Lipschitz LOWER BOUND on
        // distance to the surface, not the actual height-above).
        // For cells in coalesced empty octree regions the distance
        // can be tiny (distance to the nearest empty-cell face,
        // typically a side face) while the actual height to the
        // surface is much larger. The projected probe then landed
        // in air above the surface, read material=0, and rejected
        // every blade brick.
        //
        // Sphere-tracing to recover the actual surface point would
        // cost 5-15 host_samples per brick-parent cell. For
        // hundreds of tiled regions × thousands of cells per tile
        // that's millions of extra samples per frame — too
        // expensive for the gain.
        //
        // V12 deferred allocation makes this check unnecessary for
        // correctness: bricks whose 64 cells all return
        // occupancy=0 from the user shader never consume a slot,
        // and the user shader's own logic (e.g. grass's
        // sphere-trace down to find painted surface) does the
        // material match cell-by-cell in the fill pass. The cost
        // we're paying is dispatching `brick_fill_main` for
        // bricks that would have been pre-rejected — bounded by
        // the proximity gate at all higher levels, so it's not
        // unbounded.

        // V12 — DEFERRED BRICK ALLOCATION. Don't allocate a brick
        // here. Just queue the fill task and pre-write OCTREE_EMPTY;
        // the fill pass evaluates all 64 cells and only allocates a
        // brick (overwriting the octree node) when at least one
        // cell actually emits. Bricks with all-empty cells never
        // consume a slot.
        //
        // For grass: classify lets through cells that *might* emit
        // (proximity gate, material match) but most of them are
        // still all-empty (blade gaps, cluster gating, height
        // beyond blade_height_max). Pre-V12 these allocated bricks
        // anyway, eating ~5-10× more brick slots than necessary.
        //
        // brick_offset is set to U32_MAX as a sentinel: the fill
        // pass computes the real offset from its atomically-claimed
        // slot, not from this stored value.
        octree_nodes[cell.octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
        let fill_slot = atomicAdd(&fill_task_alloc[cell.region_index], 1u);
        if (fill_slot < cur_region.fill_task_block_size) {
            var task: BrickFillTask;
            task.octree_offset = cell.octree_offset;
            task.region_index = cell.region_index;
            task.brick_offset = 0xFFFFFFFFu; // sentinel: alloc on demand in fill
            task.cell_size = cur_region.cell_size;
            task.min_x = center.x - half;
            task.min_y = center.y - half;
            task.min_z = center.z - half;
            task._pad = 0u;
            fill_task_pool[cur_region.fill_task_block_offset + fill_slot] = task;
        } else {
            atomicAdd(&overflow[OVERFLOW_FILL_QUEUE], 1u);
        }
        return;
    }

    // Internal level — alloc 8 children from THIS REGION's octree
    // block. Per-region atomic preserves locality (region's children
    // are contiguous in the pool) and eliminates global atomic
    // contention.
    let child_slot = atomicAdd(&octree_alloc[cell.region_index], 8u);
    if (child_slot + 8u > cur_region.octree_block_size) {
        atomicAdd(&overflow[OVERFLOW_OCTREE], 1u);
        octree_nodes[cell.octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
        return;
    }
    // Compose the absolute pool offset.
    let first_child = cur_region.octree_block_offset + child_slot;
    octree_nodes[cell.octree_offset] = vec2<u32>(first_child, INTERNAL_ATTR_NONE);
    let child_half = half * 0.5;

    // V13 — when the next level IS the brick-parent level, process
    // those 8 children INLINE rather than queuing them. The deepest
    // BFS level is otherwise the dominant queue consumer (tiles ×
    // band_layers × surface_bricks_per_axis² cells); by collapsing
    // it into the second-to-last level we cut active-queue memory
    // by ~8× without needing a 4×-larger queue cap. Each child
    // gets its own proximity check + (deferred) fill task; no
    // `active_queue[L+1]` write happens.
    if (L + 1u == cur_region.max_depth) {
        let child_diag_half = child_half * SQRT3;
        let band = cur_region.region_thickness;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let cx = (k & 1u);
            let cy = (k >> 1u) & 1u;
            let cz = (k >> 2u) & 1u;
            let off_x = select(-child_half, child_half, cx == 1u);
            let off_y = select(-child_half, child_half, cy == 1u);
            let off_z = select(-child_half, child_half, cz == 1u);
            let child_center = vec3<f32>(
                center.x + off_x,
                center.y + off_y,
                center.z + off_z,
            );
            let child_offset = first_child + k;

            // Brick-parent classification, inline:
            if (cur_region.host_octree_root != HOST_NO_HOST_SENTINEL) {
                let child_host = host_sample_in_region(child_center, cell.region_index);
                if (child_host.distance > child_diag_half + band) {
                    octree_nodes[child_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
                    continue;
                }
                if (child_host.distance < -child_diag_half) {
                    octree_nodes[child_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
                    continue;
                }
            }
            // V12 deferred allocation — pre-write EMPTY, queue fill
            // task. Fill decides whether to allocate based on
            // workgroup-cooperative occupancy vote.
            octree_nodes[child_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
            let fill_slot = atomicAdd(&fill_task_alloc[cell.region_index], 1u);
            if (fill_slot < cur_region.fill_task_block_size) {
                var task: BrickFillTask;
                task.octree_offset = child_offset;
                task.region_index = cell.region_index;
                task.brick_offset = 0xFFFFFFFFu;
                task.cell_size = cur_region.cell_size;
                task.min_x = child_center.x - child_half;
                task.min_y = child_center.y - child_half;
                task.min_z = child_center.z - child_half;
                task._pad = 0u;
                fill_task_pool[cur_region.fill_task_block_offset + fill_slot] = task;
            } else {
                atomicAdd(&overflow[OVERFLOW_FILL_QUEUE], 1u);
            }
        }
        return;
    }

    // Normal internal level — push children to L+1 queue.
    let next_count = atomicAdd(&active_count[L + 1u], 8u);
    if (next_count + 8u > cap) {
        // Next-level queue overflow. Children are allocated in
        // octree_nodes but we won't classify them; pre-stamp them as
        // EMPTY so the march doesn't read uninitialised pointers.
        atomicAdd(&overflow[OVERFLOW_ACTIVE_QUEUE_BASE + L + 1u], 1u);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            octree_nodes[first_child + k] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
        }
        return;
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let cx = (k & 1u);
        let cy = (k >> 1u) & 1u;
        let cz = (k >> 2u) & 1u;
        let off_x = select(-child_half, child_half, cx == 1u);
        let off_y = select(-child_half, child_half, cy == 1u);
        let off_z = select(-child_half, child_half, cz == 1u);
        var ch: ActiveCell;
        ch.octree_offset = first_child + k;
        ch.region_index = cell.region_index;
        ch.center_x = center.x + off_x;
        ch.center_y = center.y + off_y;
        ch.center_z = center.z + off_z;
        ch.half_extent = child_half;
        ch._pad0 = 0u;
        ch._pad1 = 0u;
        active_queue[(L + 1u) * cap + next_count + k] = ch;
    }
}

// Brick fill — workgroup_size 4³, one workgroup per fill task. Host
// dispatches with `dispatch_workgroups(max_block_size, fill_dirty_count, 1)`:
//   wid.x = task index within the region's fill_task_block.
//   wid.y = region index in this frame's regions array (the
//           fill-dirty subset is at indices [0, fill_dirty_count)).
//
// Workgroups past the region's actual emitted task count (or that
// hit a sentinel slot, or whose task_idx exceeds fill_task_block_size)
// early-out before doing any brick allocation.
@compute @workgroup_size(4, 4, 4)
fn brick_fill_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wid: vec3<u32>) {
    let region_idx = wid.y;
    let task_idx_in_region = wid.x;

    // Per-thread copy of region (cheap; common subexpression).
    let cur_region = regions[region_idx];
    if (task_idx_in_region >= cur_region.fill_task_block_size) {
        return;
    }

    let task_global_offset = cur_region.fill_task_block_offset + task_idx_in_region;
    let task = fill_task_pool[task_global_offset];
    if (task.octree_offset == FILL_TASK_SENTINEL) {
        // Unused slot in the region's fill_task_block.
        return;
    }

    let cell_idx = lid.z * BRICK_DIM * BRICK_DIM + lid.y * BRICK_DIM + lid.x;

    if (cell_idx == 0u) {
        region = regions[region_idx];
        atomicStore(&wg_any_emit, 0u);
        atomicStore(&wg_alloc_failed, 0u);
        wg_brick_offset = 0u;
        // Pre-write OCTREE_EMPTY. If a cell emits, thread 0 below
        // overwrites with a brick reference; otherwise the slot
        // stays EMPTY. Required for fill-only re-bakes (cached
        // topology, dirty fill — e.g. shader-param slider moves):
        // classify did NOT run this frame, so the octree node
        // still holds the prior bake's brick reference. Without
        // this pre-write, a region whose new shader output emits
        // fewer cells leaves stale brick references behind, and
        // the march reads old grass.
        octree_nodes[task.octree_offset] = vec2<u32>(OCTREE_EMPTY, INTERNAL_ATTR_NONE);
    }
    workgroupBarrier();

    let brick_min = vec3<f32>(task.min_x, task.min_y, task.min_z);
    let cell_world_pos =
        brick_min
        + vec3<f32>(lid) * task.cell_size
        + vec3<f32>(task.cell_size * 0.5);

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

    // V12 — workgroup-cooperative deferred brick allocation. Each
    // thread votes "any cell emit?". After the barrier, thread 0
    // allocates a brick from the per-region atomic pool ONLY if at
    // least one cell voted yes. All-empty bricks consume zero
    // brick-pool slots and zero leaf-attr slots; the octree node
    // stays at OCTREE_EMPTY (pre-written by classify) and the
    // march reads no transient geometry there.
    if (emit.occupancy != 0u) {
        atomicStore(&wg_any_emit, 1u);
    }
    workgroupBarrier();

    if (cell_idx == 0u) {
        if (atomicLoad(&wg_any_emit) != 0u) {
            // At least one cell wants to emit — claim a brick slot
            // from THIS REGION's brick block.
            let brick_slot = atomicAdd(&brick_alloc[region_idx], 1u);
            if (brick_slot >= region.brick_block_size) {
                atomicAdd(&overflow[OVERFLOW_BRICK], 1u);
                atomicStore(&wg_alloc_failed, 1u);
            } else {
                let brick_id = region.brick_block_offset + brick_slot;
                wg_brick_offset = brick_id * BRICK_CELLS;
                octree_nodes[task.octree_offset] = vec2<u32>(
                    OCTREE_LEAF_BIT | OCTREE_BRICK_BIT | brick_id,
                    INTERNAL_ATTR_NONE,
                );
            }
        } else {
            // No cells want to emit — leave octree EMPTY (already
            // pre-written by classify), don't allocate a brick.
            atomicStore(&wg_alloc_failed, 1u);
        }
    }
    workgroupBarrier();

    if (atomicLoad(&wg_alloc_failed) != 0u) {
        // Either no emits or brick capacity exceeded. Either way,
        // no brick to write into. Octree stays EMPTY.
        return;
    }

    let brick_slot_idx = wg_brick_offset + cell_idx;
    if (emit.occupancy == 0u) {
        brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
        return;
    }
    let local_id = atomicAdd(&leaf_attr_alloc[region_idx], 1u);
    if (local_id >= region.leaf_attr_block_size) {
        atomicAdd(&overflow[OVERFLOW_LEAF_ATTR], 1u);
        brick_pool[brick_slot_idx] = BRICK_CELL_EMPTY;
        return;
    }
    let global_slot = region.leaf_attr_block_offset + local_id;
    var attr: LeafAttr;
    attr.normal_oct = pack_oct(emit.normal);
    let pri = emit.material_primary & 0xFFFFu;
    let sec = (emit.material_secondary & 0x0FFFu) << 16u;
    let bw = (emit.blend_weight & 0x0Fu) << 28u;
    attr.material_packed = pri | sec | bw;
    leaf_attr_pool[global_slot] = attr;
    brick_pool[brick_slot_idx] = global_slot;
}
