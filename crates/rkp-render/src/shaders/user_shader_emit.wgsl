// user_shader_emit.wgsl — Option B per-region instance scatter,
// leaf-driven dispatch.
//
// One pass per dirty instance region. The engine collects every
// painted leaf during its painted-material scan and concatenates
// them into the global `leaves` buffer; the dispatch fires one
// thread per leaf, builds a `HostSample` directly from the leaf's
// (already known) pos / normal / material, and hands it to the
// user shader's `emit` hook. No sample grid, no host_sample_at
// descent — placement is exactly the painted-leaf set.
//
// ## Dispatch shape
//
// `dispatch_workgroups(ceil(leaf_count / 64), 1, 1)` per region;
// `@workgroup_size(64, 1, 1)`. Threads with `gid >= leaf_count`
// early-out cheaply.
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

// Overflow counter slots — must match the `OVERFLOW_*` indices the
// Rust pass uses in this buffer.
const OVERFLOW_INSTANCE: u32 = 0u;

struct PaintedLeaf {
    world_pos: vec3<f32>,
    material_packed: u32,
    world_normal: vec3<f32>,
    _pad: f32,
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

// Per-region uniform — bound at group(1), one region per dispatch.
// Shape mirrors `EmitRegionUniform` in the matching Rust file.
struct EmitRegionUniform {
    aabb_min: vec3<f32>,
    cell_size: f32,
    aabb_max: vec3<f32>,
    shader_id: u32,
    time: f32,
    material_id: u32,
    region_thickness: f32,
    instance_block_offset: u32,
    instance_block_size: u32,
    instance_stride_u32: u32,
    leaf_offset: u32,
    leaf_count: u32,
    host_grid_origin: vec3<f32>,
    _pad_grid: f32,
    params: array<vec4<f32>, 2>,
    host_inverse_world: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read_write> instance_pool: array<u32>;
// Per-region "live count" array — length = MAX_REGIONS on Rust side.
// Indexed by `region_index` (carried by the dispatch's group(2) uniform).
//
// Phase 7b: this used to be an `atomic<u32>` that the per-shader
// `rkp_user_<id>_emit_instance` body atomically incremented to claim a
// slot. Slots are now deterministically `thread_id × max_emits_per_thread
// + local_count`, so the emit path no longer touches this buffer. The
// engine writes `leaves.len() × max_emits_per_thread` here CPU-side
// before the AABB pass so the downstream tile cull's `i < alloc` skip
// still works.
//
// Bound here as `read_write` purely so the wgpu binding layout matches
// the previous version's resource flags — the shader doesn't write to
// it anymore.
@group(0) @binding(1) var<storage, read_write> instance_alloc: array<u32>;
// All regions' painted leaves concatenated. The current dispatch's
// region uniform carries `leaf_offset` / `leaf_count` to slice.
@group(0) @binding(2) var<storage, read> leaves: array<PaintedLeaf>;
@group(0) @binding(3) var<storage, read_write> overflow: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> regions: array<EmitRegionUniform>;

struct EmitDispatchUniform {
    region_index: u32,
    leaf_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(2) @binding(0) var<uniform> dispatch_u: EmitDispatchUniform;

// Workgroup-shared region uniform copy. Thread 0 fills, all threads
// read after the barrier. `region` and `emit_region_index` are the
// names the composer-generated `rkp_user_<id>_emit_instance` body
// references.
var<workgroup> region: EmitRegionUniform;
var<workgroup> emit_region_index: u32;

// Phase 7b — per-thread deterministic slot allocation. `emit_main`
// initializes both at thread entry; `rkp_user_<id>_emit_instance`
// computes its slot as `thread_id × MAX × local_count` and bumps
// `rkp_emit_local_count` on each successful emit. Both are
// `var<private>`, which in WGSL means per-invocation storage — exactly
// what we want here (separate from `var<workgroup>` which would be
// shared by every thread in the workgroup).
var<private> rkp_emit_thread_id: u32;
var<private> rkp_emit_local_count: u32;

// USER_EMIT_DISPATCH_BEGIN
// Default identity stub — the Rust composer replaces this whole block
// (markers + body) with the per-shader emit_instance fns + dispatch
// switch when any registered shader provides an `emit` hook. The
// empty-registry path keeps this stub so the pipeline always validates.
fn dispatch_user_emit(shader_id: u32, host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    return;
}
// USER_EMIT_DISPATCH_END

@compute @workgroup_size(64, 1, 1)
fn emit_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    if (tid == 0u) {
        region = regions[dispatch_u.region_index];
        emit_region_index = dispatch_u.region_index;
    }
    workgroupBarrier();

    let leaf_idx = gid.x;
    if (leaf_idx >= dispatch_u.leaf_count) {
        return;
    }

    // Phase 7b — bind the deterministic slot allocator's per-thread
    // state for any emit_instance() call this thread makes.
    rkp_emit_thread_id = leaf_idx;
    rkp_emit_local_count = 0u;

    let leaf = leaves[region.leaf_offset + leaf_idx];

    var host: HostSample;
    host.distance = 0.0;
    host.normal = leaf.world_normal;
    host.material = leaf.material_packed & 0xFFFFu;
    host.material_secondary = (leaf.material_packed >> 16u) & 0x0FFFu;
    host.blend_weight = (leaf.material_packed >> 28u) & 0x0Fu;

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

    dispatch_user_emit(region.shader_id, leaf.world_pos, host, ctx);
}
