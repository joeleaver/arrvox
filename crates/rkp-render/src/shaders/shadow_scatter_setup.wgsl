// Phase 8 V2 — shadow scatter setup pass (work-list architecture).
//
// One thread per `tlas_prim`. Reads the prim's world AABB, projects
// the 8 corners through the light camera's `view_proj` to find the
// prim's footprint in light-NDC space, converts to texel coordinates,
// and writes:
//
// * `scatter_instances[i]` — per-instance descent metadata + the
//   work-list offset where this instance's tiles live.
// * `total_work` (atomic) — atomicAdd `tile_count` to allocate the
//   instance's slot range; capture pre-add value as `work_offset`.
//
// The downstream emit pass fills `work_list[work_offset..]` with
// `tile_count` packed entries; the scatter pass dispatches once
// over the entire `work_list`.
//
// "Tile" = an 8×8 texel block in shadow-map space. `tile_w * tile_h`
// is the number of 8×8 blocks needed to cover the instance's
// projected rect. Each tile becomes one workgroup in the scatter.

struct TlasPrim {
    aabb_min: vec3<f32>, asset_id: u32,
    aabb_max: vec3<f32>, instance_state_offset: u32,
    material_id: u32,
    instance_index: u32,
    _pad0: u32, _pad1: u32,
}

struct LightCameraShadow {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    light_dir: vec3<f32>,
    depth_bias: f32,
    inv_shadow_map_size: vec2<f32>,
    shadow_map_size: vec2<u32>,
}

struct ScatterInstance {
    tx0: u32, ty0: u32,
    tile_w: u32, tile_h: u32, // tile counts (8×8 each)
    asset_id: u32,
    instance_state_offset: u32,
    instance_index: u32,
    work_offset: u32,         // index into work_list[]
}

struct SetupParams {
    /// Live `tlas_prim_count` from the assembly pass. Threads
    /// beyond this index zero their slots so stale data from
    /// previous frames doesn't trigger ghost dispatches.
    prim_count: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(0) @binding(0) var<storage, read> tlas_prims: array<TlasPrim>;
@group(0) @binding(1) var<storage, read_write> scatter_instances: array<ScatterInstance>;
@group(0) @binding(2) var<storage, read_write> total_work: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> light_camera: LightCameraShadow;
@group(1) @binding(1) var<uniform> setup_params: SetupParams;

fn write_skip(i: u32) {
    scatter_instances[i] = ScatterInstance(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

@compute @workgroup_size(64, 1, 1)
fn setup_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&scatter_instances) { return; }
    if i >= setup_params.prim_count {
        write_skip(i);
        return;
    }
    let prim = tlas_prims[i];

    // Project the 8 AABB corners through the light view_proj.
    var ndc_min = vec3<f32>(1e10);
    var ndc_max = vec3<f32>(-1e10);
    for (var c = 0u; c < 8u; c++) {
        let corner = vec3<f32>(
            select(prim.aabb_min.x, prim.aabb_max.x, (c & 1u) != 0u),
            select(prim.aabb_min.y, prim.aabb_max.y, (c & 2u) != 0u),
            select(prim.aabb_min.z, prim.aabb_max.z, (c & 4u) != 0u),
        );
        let clip = light_camera.view_proj * vec4<f32>(corner, 1.0);
        let ndc = clip.xyz / max(clip.w, 1e-6);
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }
    if ndc_max.z < 0.0 || ndc_min.z > 1.0 {
        write_skip(i);
        return;
    }
    if ndc_max.x < -1.0 || ndc_min.x > 1.0
        || ndc_max.y < -1.0 || ndc_min.y > 1.0 {
        write_skip(i);
        return;
    }

    // NDC → texel; inflate by 1 on each side to absorb FP rounding.
    let ndc_lo = max(ndc_min.xy, vec2<f32>(-1.0));
    let ndc_hi = min(ndc_max.xy, vec2<f32>(1.0));
    let size_f = vec2<f32>(light_camera.shadow_map_size);
    let size_u = light_camera.shadow_map_size;
    let tx0_f = (ndc_lo.x * 0.5 + 0.5) * size_f.x - 1.0;
    let tx1_f = (ndc_hi.x * 0.5 + 0.5) * size_f.x + 1.0;
    let ty0_f = (1.0 - (ndc_hi.y * 0.5 + 0.5)) * size_f.y - 1.0;
    let ty1_f = (1.0 - (ndc_lo.y * 0.5 + 0.5)) * size_f.y + 1.0;

    let tx0 = u32(max(floor(tx0_f), 0.0));
    let tx1 = min(u32(max(ceil(tx1_f), 0.0)), size_u.x);
    let ty0 = u32(max(floor(ty0_f), 0.0));
    let ty1 = min(u32(max(ceil(ty1_f), 0.0)), size_u.y);

    if tx0 >= tx1 || ty0 >= ty1 {
        write_skip(i);
        return;
    }

    // Snap to 8-pixel tile boundaries — each tile = one workgroup.
    let tile_x0 = tx0 / 8u;
    let tile_x1 = (tx1 + 7u) / 8u;
    let tile_y0 = ty0 / 8u;
    let tile_y1 = (ty1 + 7u) / 8u;
    let tile_w = tile_x1 - tile_x0;
    let tile_h = tile_y1 - tile_y0;
    let tile_count = tile_w * tile_h;
    if tile_count == 0u {
        write_skip(i);
        return;
    }

    // Allocate this instance's slot range in the global work list.
    let work_offset = atomicAdd(&total_work[0], tile_count);

    scatter_instances[i] = ScatterInstance(
        tile_x0 * 8u, tile_y0 * 8u,
        tile_w, tile_h,
        prim.asset_id,
        prim.instance_state_offset,
        prim.instance_index,
        work_offset,
    );
}
