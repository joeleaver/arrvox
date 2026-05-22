// V1 mesh-path user-shader compute passes — spawn_count, prefix_sum,
// fill.
//
// Engine-authored skeleton; the composer replaces the
// `// USER_BODY_BEGIN` / `// USER_BODY_END` block below with the
// user's parsed module (matching the raster-side file's body). The
// stub bodies make the file valid WGSL standalone.
//
// Pipeline contract:
//   · Seven dispatches per active user-shader material per frame:
//       1. entry_spawn_count
//       2. entry_prefix_local
//       3. entry_prefix_local2
//       4. entry_prefix_scan_sums
//       5. entry_prefix_add_back2
//       6. entry_prefix_add_back
//       7. entry_fill
//     (Future work gates 1–6 by paint_epoch + geometry_epoch +
//     param_epoch for static-cache shaders.)
//   · `entry_spawn_count`: 1 thread per anchor; reads `anchors[i]`,
//     calls `spawn_count(anchor, frame)`, writes `out_counts[i]`.
//   · entry_prefix_local + _local2 + _scan_sums + _add_back2 +
//     _add_back form a 4-level Blelloch-style exclusive scan:
//       - tier 1 (_local): 256-wide WGs scan anchor counts in place;
//         each WG emits its inclusive total to `wg_sums[wgid]`.
//       - tier 2 (_local2): 256-wide WGs scan wg_sums in 256-element
//         chunks; each emits its inclusive total to `wg_sums2[wgid2]`
//         and overwrites its chunk of wg_sums with exclusive offsets.
//       - tier 3 (_scan_sums): single 256-wide WG scans wg_sums2 in
//         place (≤ 4 entries used at the V1 max), writes the
//         indirect-draw args.
//       - tier 4 (_add_back2): 256-wide WGs add wg_sums2[wgid2] back
//         into each entry of wg_sums.
//       - tier 5 (_add_back): 256-wide WGs add wg_sums[wgid] back
//         into each per-anchor offset.
//     V1 ceiling: 256 × 1024 = 262 144 anchors per material
//     (= `MAX_ANCHORS_PER_SHADER_V1`). Adding a 5th tier scales to
//     256³ ≈ 16 M structurally.
//   · `entry_fill`: 1 thread per anchor; reads `offsets[i]` +
//     `counts[i]`, runs `spawn_alive` per spawn (defaults true),
//     materializes `instance_records[offset+k] = (anchor_idx, k)` or
//     `(ANCHOR_IDX_DEAD, 0)` for dead spawns.
//
// Bindings:
//   @group(0) @binding(0)  anchors              (storage, read)
//   @group(0) @binding(1)  out_counts           (storage, read_write)
//   @group(0) @binding(2)  out_offsets          (storage, read_write)
//   @group(0) @binding(3)  out_records          (storage, read_write)
//   @group(0) @binding(4)  out_indirect_args    (storage, read_write)
//   @group(0) @binding(5)  frame                (uniform)
//   @group(0) @binding(6)  params               (uniform)
//   @group(0) @binding(7)  dispatch             (uniform)
//   @group(0) @binding(8)  wg_sums              (storage, read_write)
//   @group(0) @binding(9)  wg_sums2             (storage, read_write)
//
// Group 1 = scene (ArvxScene::bind_group_layout). Lets `paint_probe`
// from `spawn_alive` descend the host octree at the spawn position
// to cull blades whose tile-cube position lands on unpainted host
// voxels (tile-fringe culling).

import package::lib::types::ArvxInstance;
import package::lib::types::ArvxAsset;
import package::lib::types::OverlayEntry;
import package::lib::types::LeafAttr;
import package::lib::octree::octree_lookup;
import package::lib::octree::PHASE_SHADOW;
import package::lib::octree_slot::slot_is_brick;
import package::lib::octree_slot::slot_brick_id;
import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_INTERIOR;
import package::lib::brick::BRICK_DIM;
import package::lib::brick::BRICK_DIM_F;
import package::lib::brick::BRICK_CELLS;
import package::lib::brick::BRICK_CELL_EMPTY;
import package::lib::brick::BRICK_CELL_INTERIOR;
import package::lib::math::mat4_affine_inverse;
import package::lib::leaf_attr::fetch_leaf_attr_for;

const ANCHOR_IDX_DEAD: u32 = 0xFFFFFFFFu;
const PREFIX_SUM_WG_SIZE: u32 = 256u;
// Tier-1 wg count: at WG_SIZE = 256, 1024 WGs cover the V1 ceiling
// of 262 144 anchors. Each WG emits its inclusive total to
// `wg_sums[wgid]`.
const PREFIX_SUM_MAX_WG_COUNT: u32 = 1024u;
// Tier-2 wg count: each tier-2 WG (256 threads) scans 256 entries
// of `wg_sums`, so 1024 / 256 = 4 tier-2 WGs cover the full
// `wg_sums` array. Each emits its inclusive total to
// `wg_sums2[wgid2]`. The tier-3 scan reads only the first
// `PREFIX_SUM_MAX_WG_COUNT_2` entries of `wg_sums2`.
const PREFIX_SUM_MAX_WG_COUNT_2: u32 = 4u;

// Hard cap on per-anchor spawn count, mirroring
// `MAX_SPAWNS_PER_ANCHOR_V1` in the engine. `entry_spawn_count` clamps
// the user shader's `spawn_count` return to this value so the records
// buffer (sized `MAX_ANCHORS × MAX_SPAWNS_PER_ANCHOR_V1`) cannot
// overflow if a user pushes a density slider past the buffer cap.
const MAX_SPAWNS_PER_ANCHOR_V1: u32 = 32u;

// Layout mirrors `arvx_render::user_shader_mesh_pass::AnchorRecord`.
// 80 B std430. `surface_normal` (vec3) sits at offset 64 with `seed`
// packing into the trailing 4 B at offset 76 — see
// `feedback_wgsl_std430_vec3_no_padding`.
struct AnchorContext {
    tile_min:        vec3<f32>,
    material_id:     u32,
    tile_max:        vec3<f32>,
    leaf_count:      u32,
    paint_min:       vec3<f32>,
    object_id:       u32,
    paint_max:       vec3<f32>,
    surface_y:       f32,
    surface_normal:  vec3<f32>,
    seed:            u32,
    paint_mask:      u32,
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
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
@group(0) @binding(9) var<storage, read_write> wg_sums2:       array<u32>;

// Scene group — same layout ArvxScene::create_layout exposes.
// Names locked to what `lib::octree` / `lib::leaf_attr` reference as
// free symbols; binding numbers locked to ArvxScene's slots.
@group(1) @binding(0)  var<storage, read> brick_pool:       array<u32>;
@group(1) @binding(1)  var<storage, read> octree_nodes:     array<vec4<u32>>;
@group(1) @binding(2)  var<storage, read> instances:        array<ArvxInstance>;
// WESL splices an entire module on any import from it, so unused
// helpers (descend_proto_octree, fetch_leaf_color_for) reference
// these bindings as free names even though paint_probe never
// calls them. Declare so naga can resolve.
@group(1) @binding(4)  var<storage, read> color_pool_data:  array<u32>;
@group(1) @binding(7)  var<storage, read> brick_face_links: array<u32>;
@group(1) @binding(8)  var<storage, read> leaf_attr_pool:   array<LeafAttr>;
@group(1) @binding(10) var<storage, read> assets:           array<ArvxAsset>;
@group(1) @binding(11) var<storage, read> instance_overlay: array<OverlayEntry>;
// Per-instance sculpt-carve overlay (Phase A). Free-name reference
// pulled in by `lib::leaf_attr::is_leaf_removed`. paint_probe doesn't
// itself call `is_leaf_removed`; the binding has to exist because
// `use_stripping(false)` keeps the helper body in the splice.
@group(1) @binding(12) var<storage, read> instance_sculpt:  array<u32>;

// `octree_lookup` reads `bucket_depth` + `bucket_lod_exit` as free
// names. We have no `stats` binding, so they're no-ops.
fn bucket_depth(phase: u32, level: u32) {}
fn bucket_lod_exit(level: u32) {}

// Walk the host octree at `world_pos` for `anchor.object_id` and
// return true iff the resolved leaf carries `anchor.material_id`.
// Used by `spawn_alive` to cull blades whose tile-cube position
// lands on unpainted host voxels (the "tile fringe" V1 limit).
//
// Cost per call: 1 instance read, 1 asset read, 1 affine inverse
// (~30 ops), 1 octree descent (4-7 levels typical), 1 brick cell
// read on brick leaves, overlay binary search if the leaf is
// painted. Net cost is paid per spawn, not per fragment, so
// fewer-but-correct blades are cheaper downstream than the old
// "fill the cube" behavior.
fn paint_probe(world_pos: vec3<f32>, anchor: AnchorContext) -> bool {
    let inst = instances[anchor.object_id];
    let asset = assets[inst.asset_id];

    let inv_world = mat4_affine_inverse(inst.world);
    let local_pos = (inv_world * vec4<f32>(world_pos, 1.0)).xyz;
    // Octree coords ARE object-local-world-units centered on
    // grid_origin (NOT divided by voxel_size). `extent` and the
    // per-level `half` walked by octree_lookup are world-scale.
    // `oc_origin = local_origin - asset.grid_origin`.
    let oc_pos = local_pos - asset.grid_origin;

    let extent = bitcast<f32>(asset.octree_extent_bits);
    if (oc_pos.x < 0.0 || oc_pos.y < 0.0 || oc_pos.z < 0.0
        || oc_pos.x >= extent || oc_pos.y >= extent || oc_pos.z >= extent) {
        return false;
    }

    // Static lookup. Per the lib comment in octree.wesl, callers
    // without a meaningful ray pass `(vec3(0), vec3(1))` to disable
    // the per-node tight-AABB cull — otherwise oc_pos is treated as
    // a ray origin and the cull rejects when the cell's tight
    // content sits "behind" the ray. lod_enabled = 0 skips the LOD
    // branch (irrelevant for a probe).
    let lookup = octree_lookup(
        asset.octree_root,
        asset.octree_depth,
        extent,
        oc_pos,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
        PHASE_SHADOW,
        0u,
        0.0,
        1.0,
        1.0,
    );

    if (lookup.slot == OCTREE_EMPTY) { return false; }
    // Interior bulk regions don't carry per-cell leaf data and can't
    // be paint targets — treat as not painted.
    if (lookup.slot == OCTREE_INTERIOR) { return false; }

    var leaf_slot: u32;
    if (slot_is_brick(lookup.slot)) {
        let brick_id = slot_brick_id(lookup.slot);
        let cell_size = (lookup.cell_half * 2.0) / BRICK_DIM_F;
        let brick_origin = lookup.cell_center - vec3<f32>(lookup.cell_half);
        let local = (oc_pos - brick_origin) / cell_size;
        let cell = clamp(
            vec3<i32>(floor(local)),
            vec3<i32>(0),
            vec3<i32>(3),
        );
        let flat = u32(cell.x)
            + u32(cell.y) * BRICK_DIM
            + u32(cell.z) * BRICK_DIM * BRICK_DIM;
        let c = brick_pool[brick_id * BRICK_CELLS + flat];
        if (c == BRICK_CELL_EMPTY || c == BRICK_CELL_INTERIOR) {
            return false;
        }
        leaf_slot = c;
    } else {
        leaf_slot = lookup.slot;
    }

    let attr = fetch_leaf_attr_for(inst, leaf_slot);
    let primary = attr.material_packed & 0xFFFFu;
    let secondary = (attr.material_packed >> 16u) & 0x0FFFu;
    let blend = (attr.material_packed >> 28u) & 0xFu;
    if (primary == anchor.material_id) { return true; }
    // Secondary only counts when blend dial is at the high end —
    // matches the editor's "secondary dominates if blend ≥ 12/15"
    // visual cue.
    if (blend >= 12u && secondary == anchor.material_id) { return true; }
    return false;
}

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
    // Hard-clamp the user shader's return to the records-buffer cap.
    // A runaway density slider can't overflow `out_records` because
    // every anchor contributes at most `MAX_SPAWNS_PER_ANCHOR_V1` to
    // the global total `instance_count` produced by the scan.
    out_counts[i] = min(spawn_count(anchors[i], frame), MAX_SPAWNS_PER_ANCHOR_V1);
}

// ── Multi-pass Blelloch prefix sum ────────────────────────────────
//
// Four-level exclusive scan covering up to
// `PREFIX_SUM_WG_SIZE * PREFIX_SUM_MAX_WG_COUNT = 256 * 1024 = 262 144`
// anchors per material (= `MAX_ANCHORS_PER_SHADER_V1`).
//
//   1. `entry_prefix_local` — each tier-1 WG of 256 threads does a
//      Hillis-Steele scan over its slice; last thread stashes the
//      workgroup total into `wg_sums[wgid]`. Per-thread offsets are
//      written into `out_offsets` *without* the per-workgroup base
//      added yet. Up to 1024 tier-1 WGs.
//   2. `entry_prefix_local2` — each tier-2 WG of 256 threads scans
//      a 256-entry chunk of `wg_sums` in place. The local exclusive
//      offsets are written back to `wg_sums`; the chunk total goes
//      to `wg_sums2[wgid2]`. Up to 4 tier-2 WGs.
//   3. `entry_prefix_scan_sums` — single workgroup of 256 threads
//      scans `wg_sums2[0..MAX_WG_COUNT_2]` in place, converting per-
//      tier-2-block totals into exclusive offsets. Thread 0 also
//      writes the indirect-draw args (vertex_count + total
//      instance_count).
//   4. `entry_prefix_add_back2` — adds each tier-2 base
//      (`wg_sums2[wgid2]`) onto every entry of its 256-entry block
//      in `wg_sums`. After this `wg_sums` holds global exclusive
//      offsets per tier-1 WG.
//   5. `entry_prefix_add_back` — adds each tier-1 base
//      (`wg_sums[wgid]`) onto every `out_offsets[idx]` so the final
//      per-thread offset is the global exclusive scan.

var<workgroup> scan_data: array<u32, PREFIX_SUM_WG_SIZE>;

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

    var stride: u32 = 1u;
    while (stride < PREFIX_SUM_WG_SIZE) {
        var t: u32 = 0u;
        if (tid >= stride) { t = scan_data[tid - stride]; }
        workgroupBarrier();
        if (tid >= stride) { scan_data[tid] = scan_data[tid] + t; }
        workgroupBarrier();
        stride = stride << 1u;
    }

    if (idx < dispatch.num_anchors) {
        out_offsets[idx] = scan_data[tid] - val;
    }

    if (tid == PREFIX_SUM_WG_SIZE - 1u) {
        wg_sums[wgid.x] = scan_data[tid];
    }
}

@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_local2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let tid = lid.x;
    let idx = gid.x;

    // Each tier-2 WG scans 256 entries of wg_sums. Only the entries
    // for live tier-1 WGs were written this frame — past that index
    // wg_sums holds stale values from a previous frame's run, so
    // clamp reads to the live range and feed 0 into the scan.
    let live_tier1_wgs =
        (dispatch.num_anchors + PREFIX_SUM_WG_SIZE - 1u) / PREFIX_SUM_WG_SIZE;
    var val: u32 = 0u;
    if (idx < live_tier1_wgs) { val = wg_sums[idx]; }
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

    // Exclusive offsets back to wg_sums; chunk total to wg_sums2.
    if (idx < PREFIX_SUM_MAX_WG_COUNT) {
        wg_sums[idx] = scan_data[tid] - val;
    }
    if (tid == PREFIX_SUM_WG_SIZE - 1u) {
        wg_sums2[wgid.x] = scan_data[tid];
    }
}

@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_scan_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // wg_sums2[i] is stale past the live tier-2 range (the host
    // dispatches only as many tier-2 WGs as needed). Same clamp
    // pattern as entry_prefix_local2.
    let live_tier1_wgs =
        (dispatch.num_anchors + PREFIX_SUM_WG_SIZE - 1u) / PREFIX_SUM_WG_SIZE;
    let live_tier2_wgs =
        (live_tier1_wgs + PREFIX_SUM_WG_SIZE - 1u) / PREFIX_SUM_WG_SIZE;
    var val: u32 = 0u;
    if (tid < live_tier2_wgs) { val = wg_sums2[tid]; }
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

    if (tid < PREFIX_SUM_MAX_WG_COUNT_2) {
        wg_sums2[tid] = scan_data[tid] - val;
    }

    if (tid == 0u) {
        let n = dispatch.num_anchors;
        var total: u32 = 0u;
        if (n > 0u) {
            // anchors-per-tier-2-block = WG_SIZE * WG_SIZE = 65536.
            let per_block = PREFIX_SUM_WG_SIZE * PREFIX_SUM_WG_SIZE;
            let last_wg2 = (n + per_block - 1u) / per_block;
            if (last_wg2 > 0u && last_wg2 <= PREFIX_SUM_MAX_WG_COUNT_2) {
                total = scan_data[last_wg2 - 1u];
            }
        }
        out_indirect.vertex_count   = dispatch.verts_per_spawn;
        out_indirect.instance_count = total;
        out_indirect.first_vertex   = 0u;
        out_indirect.first_instance = 0u;
    }
}

@compute @workgroup_size(PREFIX_SUM_WG_SIZE)
fn entry_prefix_add_back2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= PREFIX_SUM_MAX_WG_COUNT) { return; }
    wg_sums[idx] = wg_sums[idx] + wg_sums2[wgid.x];
}

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
