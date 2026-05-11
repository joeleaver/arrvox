// V1 mesh-path user-shader compute passes — spawn_count, prefix_sum,
// fill.
//
// Engine-authored skeleton; the composer replaces the
// `// USER_BODY_BEGIN` / `// USER_BODY_END` block below with the
// user's parsed module (matching the raster-side file's body). The
// stub bodies make the file valid WGSL standalone.
//
// Pipeline contract:
//   · Five dispatches per active user-shader material per frame:
//       1. entry_spawn_count
//       2. entry_prefix_local
//       3. entry_prefix_scan_sums
//       4. entry_prefix_add_back
//       5. entry_fill
//     (Future work gates 1–4 by paint_epoch + geometry_epoch +
//     param_epoch for static-cache shaders.)
//   · `entry_spawn_count`: 1 thread per anchor; reads `anchors[i]`,
//     calls `spawn_count(anchor, frame)`, writes `out_counts[i]`.
//   · `entry_prefix_local` + `entry_prefix_scan_sums` +
//     `entry_prefix_add_back`: 3-pass Blelloch-style scan. Each
//     workgroup scans 256 elements locally and emits its total to
//     `wg_sums[wg_id]`; one scanning workgroup turns wg_sums into
//     cumulative offsets; final pass adds the workgroup-offset back
//     into every per-thread offset. V1 ceiling: 256 × 32 = 8192
//     anchors per material (= `MAX_ANCHORS_PER_SHADER_V1`). The
//     scan-of-sums pass also writes the indirect-draw args at slot 0.
//   · `entry_fill`: 1 thread per anchor; reads `offsets[i]` +
//     `counts[i]`, runs `spawn_alive` per spawn (defaults true),
//     materializes `instance_records[offset+k] = (anchor_idx, k)` or
//     `(ANCHOR_IDX_DEAD, 0)` for dead spawns.
//
// Bindings (single group, all passes share the same data):
//   @group(0) @binding(0)  anchors              (storage, read)
//   @group(0) @binding(1)  out_counts           (storage, read_write)
//   @group(0) @binding(2)  out_offsets          (storage, read_write)
//   @group(0) @binding(3)  out_records          (storage, read_write)
//   @group(0) @binding(4)  out_indirect_args    (storage, read_write)
//   @group(0) @binding(5)  frame                (uniform)
//   @group(0) @binding(6)  params               (uniform)
//   @group(0) @binding(7)  dispatch             (uniform)
//   @group(0) @binding(8)  wg_sums              (storage, read_write)

const ANCHOR_IDX_DEAD: u32 = 0xFFFFFFFFu;
const PREFIX_SUM_WG_SIZE: u32 = 256u;
const PREFIX_SUM_MAX_WG_COUNT: u32 = 32u;

struct AnchorContext {
    world_pos:       vec3<f32>,
    leaf_extent:     f32,
    surface_normal:  vec3<f32>,
    surface_area:    f32,
    host_color:      vec4<f32>,
    material_id:     u32,
    material_blend:  u32,
    leaf_slot:       u32,
    seed:            u32,
}

struct FrameContext {
    time:            f32,
    delta_time:      f32,
    wind_dir:        vec3<f32>,
    wind_strength:   f32,
    camera_pos:      vec3<f32>,
    _pad:            f32,
}

struct InstanceRecord {
    anchor_idx:      u32,
    spawn_idx:       u32,
}

struct UserParams {
    p: array<vec4<f32>, 2>,
}

// Standard indexed-indirect args layout: (vertex_count, instance_count,
// first_index, base_vertex, first_instance). We use non-indexed draws
// in V1; layout shifts to (vertex_count, instance_count, first_vertex,
// first_instance). Slot 0 is the only entry — one indirect dispatch
// per user-shader material.
struct DrawIndirectArgs {
    vertex_count:    u32,
    instance_count:  u32,
    first_vertex:    u32,
    first_instance:  u32,
}

struct DispatchInfo {
    num_anchors:     u32,
    verts_per_spawn: u32,
    _pad0:           u32,
    _pad1:           u32,
}

@group(0) @binding(0) var<storage, read>       anchors:        array<AnchorContext>;
@group(0) @binding(1) var<storage, read_write> out_counts:     array<u32>;
@group(0) @binding(2) var<storage, read_write> out_offsets:    array<u32>;
@group(0) @binding(3) var<storage, read_write> out_records:    array<InstanceRecord>;
@group(0) @binding(4) var<storage, read_write> out_indirect:   DrawIndirectArgs;
@group(0) @binding(5) var<uniform>             frame:          FrameContext;
@group(0) @binding(6) var<uniform>             params:         UserParams;
@group(0) @binding(7) var<uniform>             dispatch:       DispatchInfo;
@group(0) @binding(8) var<storage, read_write> wg_sums:        array<u32>;

fn ctx_param(i: u32) -> f32 {
    let v = params.p[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// USER_BODY anchors — `splice_const_marker` replaces everything
// between these two top-level const decls with the composed user
// body. Required: `fn spawn_count(anchor, frame) -> u32`. Optional:
// `fn spawn_alive(anchor, spawn_idx, frame) -> bool` (default true).
//
// Stubs: spawn_count returns 0 → no spawns → entry_fill no-ops →
// indirect draw with instance_count=0.
const USER_BODY_BEGIN: u32 = 0u;

fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    return 0u;
}

fn spawn_alive(anchor: AnchorContext, spawn_idx: u32, frame: FrameContext) -> bool {
    return true;
}

const USER_BODY_END: u32 = 0u;

// ── spawn_count entry ─────────────────────────────────────────────

@compute @workgroup_size(64)
fn entry_spawn_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dispatch.num_anchors) { return; }
    out_counts[i] = spawn_count(anchors[i], frame);
}

// ── Multi-pass prefix sum (Blelloch-style, 3 passes) ──────────────
//
// Goal: exclusive scan of `out_counts[0..num_anchors]` into
// `out_offsets[]`, plus write the total spawn count into the
// indirect-draw args. Supports up to PREFIX_SUM_WG_SIZE ×
// PREFIX_SUM_MAX_WG_COUNT anchors per material.

var<workgroup> scan_data: array<u32, PREFIX_SUM_WG_SIZE>;

// Pass 1: each workgroup scans 256 elements locally, writes its
// total to wg_sums[workgroup_id].
@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tid = lid.x;
    let idx = gid.x;

    var val: u32 = 0u;
    if (idx < dispatch.num_anchors) { val = out_counts[idx]; }
    scan_data[tid] = val;
    workgroupBarrier();

    // Hillis-Steele inclusive scan within the workgroup.
    var stride: u32 = 1u;
    while (stride < PREFIX_SUM_WG_SIZE) {
        var t: u32 = 0u;
        if (tid >= stride) { t = scan_data[tid - stride]; }
        workgroupBarrier();
        if (tid >= stride) { scan_data[tid] = scan_data[tid] + t; }
        workgroupBarrier();
        stride = stride << 1u;
    }

    // Convert inclusive → exclusive and store the workgroup-local
    // offset (will get the workgroup base added back in pass 3).
    if (idx < dispatch.num_anchors) {
        out_offsets[idx] = scan_data[tid] - val;
    }

    // Last thread in each workgroup stashes the local total.
    if (tid == PREFIX_SUM_WG_SIZE - 1u) {
        wg_sums[wgid.x] = scan_data[tid];
    }
}

// Pass 2: single workgroup scans wg_sums[] in place. Each lane
// reads its own initial value into scan_data, runs the same
// Hillis-Steele scan, then writes its EXCLUSIVE result back to
// wg_sums (overwriting the per-workgroup total with the cumulative
// offset). Thread 0 also writes the indirect-draw args.
@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_scan_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    var val: u32 = 0u;
    if (tid < PREFIX_SUM_MAX_WG_COUNT) { val = wg_sums[tid]; }
    scan_data[tid] = val;
    workgroupBarrier();

    var stride: u32 = 1u;
    while (stride < PREFIX_SUM_WG_SIZE) {
        var t: u32 = 0u;
        if (tid >= stride) { t = scan_data[tid - stride]; }
        workgroupBarrier();
        if (tid >= stride) { scan_data[tid] = scan_data[tid] + t; }
        workgroupBarrier();
        stride = stride << 1u;
    }

    if (tid < PREFIX_SUM_MAX_WG_COUNT) {
        wg_sums[tid] = scan_data[tid] - val;
    }

    if (tid == 0u) {
        let n = dispatch.num_anchors;
        var total: u32 = 0u;
        if (n > 0u) {
            let last_wg = (n + PREFIX_SUM_WG_SIZE - 1u) / PREFIX_SUM_WG_SIZE;
            if (last_wg > 0u && last_wg <= PREFIX_SUM_MAX_WG_COUNT) {
                total = scan_data[last_wg - 1u];
            }
        }
        out_indirect.vertex_count   = dispatch.verts_per_spawn;
        out_indirect.instance_count = total;
        out_indirect.first_vertex   = 0u;
        out_indirect.first_instance = 0u;
    }
}

// Pass 3: add the per-workgroup base back into every per-thread
// local offset, yielding the global exclusive scan in out_offsets.
@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_add_back(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= dispatch.num_anchors) { return; }
    out_offsets[idx] = out_offsets[idx] + wg_sums[wgid.x];
}

// ── fill entry — one thread per anchor; fans out per-spawn records ─

@compute @workgroup_size(64)
fn entry_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dispatch.num_anchors) { return; }
    let off = out_offsets[i];
    let cnt = out_counts[i];
    let anchor = anchors[i];
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
        let slot = off + k;
        if (spawn_alive(anchor, k, frame)) {
            var r: InstanceRecord;
            r.anchor_idx = i;
            r.spawn_idx  = k;
            out_records[slot] = r;
        } else {
            var r: InstanceRecord;
            r.anchor_idx = ANCHOR_IDX_DEAD;
            r.spawn_idx  = 0u;
            out_records[slot] = r;
        }
    }
}
