// Phase 7c Session 1 — TLAS primitive assembly (user-shader path).
//
// Walks the per-region tile-cull scratch buffer (populated by
// `user_shader_tile_cull.wgsl::tile_cull_main`), filters out dead
// slots and degenerate AABBs, and atomic-appends a `TlasPrim`
// carrying the shadow trace's per-instance metadata
// (asset_id, instance_state_offset, material_id) plus the
// `TLAS_LEAF_USER_SHADER` sentinel in `instance_index` so Session 5
// can route shadow descent through the user-shader hooks.
//
// The companion host-instance assembly pass
// (`tlas_assemble_host.wgsl`) writes into the same `tlas_prims` +
// `tlas_prim_count` pair, so the engine fires both dispatches
// before Session 2's Morton sort consumes the unified buffer.
//
// ## Wire format
//
// `TlasPrim` is 48 bytes — same field layout as
// `InstanceTileCullEntry` from `user_shader_tile_cull.wgsl` minus
// the `live` flag (filtered here) plus an `instance_index` field
// that distinguishes host (real `instances[]` index) from
// user-shader (`TLAS_LEAF_USER_SHADER`) leaves.

const TLAS_LEAF_USER_SHADER: u32 = 0xFFFFFFFEu;
const TILE_CULL_LIVE: u32 = 1u;

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

struct TlasPrim {
    aabb_min: vec3<f32>,
    asset_id: u32,
    aabb_max: vec3<f32>,
    instance_state_offset: u32,
    material_id: u32,
    instance_index: u32,
    _pad0: u32,
    _pad1: u32,
}

struct AssembleUserShaderUniform {
    // Number of scratch entries to walk. Equal to
    // `tile_cull_scratch_count` from `tick_instance_pipeline`.
    scratch_count: u32,
    // Capacity of `tlas_prims` in entries; threads that would
    // overflow it bump the atomic but skip the write.
    prims_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> tile_cull_scratch: array<InstanceTileCullEntry>;
@group(0) @binding(1) var<storage, read_write> tlas_prims: array<TlasPrim>;
@group(0) @binding(2) var<storage, read_write> tlas_prim_count: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> u: AssembleUserShaderUniform;

@compute @workgroup_size(64, 1, 1)
fn assemble_user_shader_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.scratch_count) {
        return;
    }
    let entry = tile_cull_scratch[i];
    if (entry.live != TILE_CULL_LIVE) {
        return;
    }
    // Reject a degenerate AABB (zero-volume from an unowned
    // deterministic slot whose state stayed pre-cleared, or from a
    // shader that emitted a degenerate instance). Cheaper to filter
    // here than to carry through Morton + sort + tree build.
    let extent = entry.aabb_max - entry.aabb_min;
    if (extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0) {
        return;
    }
    let slot = atomicAdd(&tlas_prim_count[0], 1u);
    if (slot >= u.prims_capacity) {
        // Pool full — bail. Atomic still incremented so the engine
        // can detect overflow on readback.
        return;
    }
    var p: TlasPrim;
    p.aabb_min = entry.aabb_min;
    p.asset_id = entry.asset_id;
    p.aabb_max = entry.aabb_max;
    p.instance_state_offset = entry.instance_state_offset;
    p.material_id = entry.material_id;
    p.instance_index = TLAS_LEAF_USER_SHADER;
    p._pad0 = 0u;
    p._pad1 = 0u;
    tlas_prims[slot] = p;
}
