// user_shader_emit.wgsl — Option B per-region instance scatter.
//
// One pass per dirty instance region. Walks a 3D sample grid at
// brick-parent granularity (4 × cell_size) over the region's AABB,
// invokes `host_sample_at(host_pos)` for proximity / material info,
// then calls the user's `emit` hook for each sample. The user's body
// calls `emit_instance(<MyStruct>)` zero or more times per call; the
// composer-generated `rkp_user_<id>_emit_instance` does the actual
// atomic-append into the per-region slice of the global instance pool.
//
// ## Dispatch shape
//
// Each region gets its own dispatch:
//   `dispatch_workgroups(wgs_per_axis, wgs_per_axis, wgs_per_axis)`
// where `wgs_per_axis = ceil(samples_per_axis / 4)` and
// `samples_per_axis = ceil(extent / (cell_size * 4))`.
//
// Workgroup_size is (4, 4, 4) so each workgroup covers a 4³ sub-grid
// of brick-parent cells (= 16 cm cube at cell_size 1 cm). Threads past
// the region's actual sample count early-out cheaply.
//
// ## Compose contract
//
// The Rust composer emits a chunk that:
//   1. Defines each user shader's instance struct + helper structs
//      + helper fns (with `emit_instance(` calls rewritten to the
//      per-shader `rkp_user_<id>_emit_instance(` form).
//   2. Defines `rkp_user_<id>_emit_instance(<Struct>)` per shader,
//      with bitcast writes into `instance_pool` driven by the
//      parsed layout's field byte offsets.
//   3. Defines the user's `emit` body, renamed to `rkp_user_<id>_emit`.
//   4. Defines `dispatch_user_emit(shader_id, host_pos, host, ctx)`
//      that switches to the right `rkp_user_<id>_emit`.
// All four pieces splice into the BEGIN/END markers below.

const BRICK_DIM: u32 = 4u;

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;

const HOST_NO_HOST_SENTINEL: u32 = 0xFFFFFFFFu;

// Overflow counter slots — must match the `OVERFLOW_*` indices the
// Rust pass uses in this buffer.
const OVERFLOW_INSTANCE: u32 = 0u;

const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;

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

// Per-region uniform — bound at group(1) via dynamic offset, one
// region per dispatch. Shape mirrors `EmitRegionUniform` in the
// matching Rust file.
struct EmitRegionUniform {
    aabb_min: vec3<f32>,
    cell_size: f32,
    aabb_max: vec3<f32>,
    shader_id: u32,
    time: f32,
    material_id: u32,
    region_thickness: f32,
    // Instance pool block in u32 units — `instance_block_offset` is
    // the absolute u32 index of the first instance slot, NOT the byte
    // offset. `instance_block_size` is in INSTANCE SLOTS (use
    // `instance_stride_u32` to convert to u32 indices when needed).
    instance_block_offset: u32,
    instance_block_size: u32,
    // Stride between consecutive instance records, in u32s. Equal to
    // `roundUpDiv(sizeof(<Struct>), 4)` for this shader.
    instance_stride_u32: u32,
    host_octree_root: u32,
    host_octree_depth: u32,
    host_octree_extent: f32,
    _pad_host: u32,
    host_grid_origin: vec3<f32>,
    _pad_grid: f32,
    params: array<vec4<f32>, 2>,
    host_inverse_world: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read_write> instance_pool: array<u32>;
// Per-region atomic counter — array length = MAX_REGIONS on Rust side.
// Indexed by `region_index` (carried by the dispatch's group(2) uniform).
@group(0) @binding(1) var<storage, read_write> instance_alloc: array<atomic<u32>>;
// Host octree data so `host_sample_at` can read the painted host.
@group(0) @binding(2) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(4) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(5) var<storage, read_write> overflow: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> regions: array<EmitRegionUniform>;

// Per-dispatch state — single u32 telling the shader which region in
// the regions array to process this dispatch. CPU writes a different
// value before each region's dispatch.
struct EmitDispatchUniform {
    region_index: u32,
    samples_per_axis: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(2) @binding(0) var<uniform> dispatch_u: EmitDispatchUniform;

// Workgroup-shared region uniform copy. Thread 0 fills, all threads
// read after the barrier. Same pattern as brick_fill_main in
// user_shader_geom.wgsl — `region` and `emit_region_index` are the
// names the composer-generated `rkp_user_<id>_emit_instance` body
// references.
var<workgroup> region: EmitRegionUniform;
var<workgroup> emit_region_index: u32;

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

fn distance_to_local_box(pos: vec3<f32>, c: vec3<f32>, h: f32) -> f32 {
    let d = abs(pos - c) - vec3<f32>(h);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

// Walk the host's painted octree to sample distance + material at
// `world_pos`. Identical structure to the geom shader's host_sample_at,
// adapted to `EmitRegionUniform`. Reads the workgroup-shared `region`
// so all threads in this dispatch use the same region data.
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

// USER_EMIT_DISPATCH_BEGIN
// Default identity stub — the Rust composer replaces this whole block
// (markers + body) with the per-shader emit_instance fns + dispatch
// switch when any registered shader provides an `emit` hook. The
// empty-registry path keeps this stub so the pipeline always validates.
fn dispatch_user_emit(shader_id: u32, host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    return;
}
// USER_EMIT_DISPATCH_END

@compute @workgroup_size(4, 4, 4)
fn emit_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    if (tid == 0u) {
        region = regions[dispatch_u.region_index];
        emit_region_index = dispatch_u.region_index;
    }
    workgroupBarrier();

    let sample_3d = wid * BRICK_DIM + lid;
    let samples_per_axis = dispatch_u.samples_per_axis;
    if (sample_3d.x >= samples_per_axis
        || sample_3d.y >= samples_per_axis
        || sample_3d.z >= samples_per_axis) {
        return;
    }

    // Brick-parent cell width = cell_size * 4. Sample the cell center.
    let bp_cell = region.cell_size * f32(BRICK_DIM);
    let host_pos =
        region.aabb_min
        + vec3<f32>(sample_3d) * bp_cell
        + vec3<f32>(bp_cell * 0.5);

    let host = host_sample_at(host_pos);

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

    dispatch_user_emit(region.shader_id, host_pos, host, ctx);
}
