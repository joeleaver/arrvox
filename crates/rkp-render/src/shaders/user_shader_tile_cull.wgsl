// Phase 6 Session 2 — user-shader tile-cull AABB compute pass.
//
// Per filled instance slot (one thread per slot, 64 threads/workgroup),
// dispatches the user shader's `inst_aabb` hook to compute the
// instance's world-space axis-aligned bounding box, then writes one
// `InstanceTileCullEntry` into the scratch buffer that Session 3's
// tile-cull (count + prefix + scatter) consumes.
//
// One dispatch per `InstanceRegionRequest`. Each thread reads
// `instance_alloc[region.region_index]` to know how many slots the
// emit pass actually populated this frame. Threads with `gid >= alloc`
// emit a `live = 0` placeholder so Session 3 can skip them with one
// branch and no per-thread atomics. Threads with `gid >= block_size`
// (past the region's reservation) early-return entirely — those slots
// don't exist in the scratch buffer.
//
// World-space AABBs are VP-independent on purpose: Session 3 will run
// per viewport against this single scratch buffer rather than baking
// per-VP screen AABBs into the bake (that would tie the scratch's
// lifetime to a single frame's specific viewport set).

const TILE_CULL_LIVE: u32 = 1u;
const TILE_CULL_DEAD: u32 = 0u;

// World-space AABB returned by `dispatch_user_inst_aabb`. Mirror of the
// type in `octree_march.wgsl` and `rkp_shadow_trace.wgsl`. The composer
// expects this exact name for the spliced per-shader bodies.
struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
}

// One scratch entry per reserved instance slot. 48 bytes (vec3<f32> in
// WGSL aligns to 16 but packs a trailing u32 into the same 16-byte
// stride, so the {vec3, u32} pairs hold without padding). Phase 6
// design memo: caching AABBs between count and scatter is simpler than
// recomputing — at 500K instances 48 B/entry ≈ 24 MB, well within
// budget for V1 caps.
struct InstanceTileCullEntry {
    aabb_min: vec3<f32>,
    asset_id: u32,
    aabb_max: vec3<f32>,
    instance_state_offset: u32,
    material_id: u32,
    live: u32,
    _pad0: u32,
    _pad1: u32,
}

// Per-region uniform — must match `TileCullRegionUniform` in
// `user_shader_tile_cull_pass.rs`. 32 bytes; uploaded once per region
// per frame and bound with a 256-byte dynamic-offset stride.
struct TileCullRegionUniform {
    region_index: u32,
    asset_id: u32,
    material_id: u32,
    shader_id: u32,
    instance_block_offset: u32,  // u32 offset into instance_pool
    instance_block_size: u32,    // capacity (max instances reserved)
    instance_stride_u32: u32,    // u32 stride between per-instance records
    scratch_offset: u32,         // entry index in tile_cull_scratch
}

@group(0) @binding(0) var<storage, read> instance_pool: array<u32>;
@group(0) @binding(1) var<storage, read> instance_alloc: array<u32>;
@group(0) @binding(2) var<storage, read_write> tile_cull_scratch: array<InstanceTileCullEntry>;

@group(1) @binding(0) var<uniform> region: TileCullRegionUniform;

// Default canonical → world map. Used only by the identity stub of
// `dispatch_user_inst_to_local` below — the AABB compute itself never
// calls it. The marker pair lets `splice_inst_chunks` succeed without
// a separate code path. (The composer always emits both chunks; if
// only `inst_aabb` were spliced, naga would still see the spliced
// pool-read helpers reference structs declared in the `inst_to_local`
// chunk, so we keep both wired.)
fn inst_world_to_local(
    world_pos: vec3<f32>, instance_pos: vec3<f32>, instance_scale: f32,
) -> vec3<f32> {
    let inv_s = 1.0 / max(instance_scale, 1e-10);
    return (world_pos - instance_pos) * inv_s + vec3<f32>(0.5);
}

// USER_INST_TO_LOCAL_DISPATCH_BEGIN
fn dispatch_user_inst_to_local(
    shader_id: u32,
    base_u32: u32,
    world_pos: vec3<f32>,
    fallback_pos: vec3<f32>,
    fallback_scale: f32,
) -> vec3<f32> {
    return inst_world_to_local(world_pos, fallback_pos, fallback_scale);
}
// USER_INST_TO_LOCAL_DISPATCH_END

// USER_INST_AABB_DISPATCH_BEGIN
fn dispatch_user_inst_aabb(
    shader_id: u32,
    base_u32: u32,
    fallback_pos: vec3<f32>,
    fallback_scale: f32,
) -> Aabb {
    let half = fallback_scale * 0.5 * 1.7320508; // √3 ≈ 1.732
    var a: Aabb;
    a.min = fallback_pos - vec3<f32>(half);
    a.max = fallback_pos + vec3<f32>(half);
    return a;
}
// USER_INST_AABB_DISPATCH_END

@compute @workgroup_size(64, 1, 1)
fn tile_cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= region.instance_block_size) {
        return;
    }

    let scratch_idx = region.scratch_offset + i;
    let alloc = instance_alloc[region.region_index];

    if (i >= alloc) {
        // Emit-pass didn't populate this slot — write a dead placeholder
        // so Session 3 can skip with one branch (no per-thread atomics).
        var dead: InstanceTileCullEntry;
        dead.aabb_min = vec3<f32>(0.0);
        dead.aabb_max = vec3<f32>(0.0);
        dead.asset_id = 0u;
        dead.instance_state_offset = 0u;
        dead.material_id = 0u;
        dead.live = TILE_CULL_DEAD;
        dead._pad0 = 0u;
        dead._pad1 = 0u;
        tile_cull_scratch[scratch_idx] = dead;
        return;
    }

    let base_u32 = region.instance_block_offset + i * region.instance_stride_u32;

    // Properly-written shaders override `inst_aabb` and read their
    // per-instance struct from `instance_pool[base_u32 + N]`. Fallback
    // pos/scale are only consulted by the identity stub (which assumes
    // `instance_pool[0..3]` is a position; meaningless for arbitrary
    // user structs). Pass zeros — the stub's degenerate AABB is
    // harmless since real shaders never reach it.
    let aabb = dispatch_user_inst_aabb(
        region.shader_id,
        base_u32,
        vec3<f32>(0.0),
        1.0,
    );

    var e: InstanceTileCullEntry;
    e.aabb_min = aabb.min;
    e.asset_id = region.asset_id;
    e.aabb_max = aabb.max;
    e.instance_state_offset = base_u32;
    e.material_id = region.material_id;
    e.live = TILE_CULL_LIVE;
    e._pad0 = 0u;
    e._pad1 = 0u;
    tile_cull_scratch[scratch_idx] = e;
}
