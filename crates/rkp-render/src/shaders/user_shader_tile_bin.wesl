// User-shader tile-bin pass.
//
// One thread per emitted user-shader instance. Projects the
// instance's world AABB to screen space, computes the range of
// 8×8-pixel tiles it overlaps, and atomically appends the instance
// index to each overlapped tile's list.
//
// Output per viewport:
//   - `tile_counts[tile_id]` — number of instances bound to this tile.
//   - `tile_lists[tile_id * MAX_PER_TILE + slot]` — instance index.
//
// Capped at `MAX_PER_TILE` instances per tile; overflow drops silently
// (artifact: blades disappear in dense regions). The march's
// user-shader scan only iterates the current pixel's tile list, so
// per-pixel cost drops from O(N_total) to O(N_in_my_tile).

import package::lib::types::CameraUniforms;

const MAX_INSTANCES_PER_TILE: u32 = 256u;

struct BinParams {
    instance_count_upper_bound: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> instance_aabbs: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tile_lists: array<u32>;
@group(0) @binding(4) var<uniform> bin_params: BinParams;
// Actual instance count, written by the emit pass on the GPU. CPU
// only knows an upper bound (`painted_leaves × MAX_EMITS_PER_LEAF`),
// so the dispatch is sized to that and threads past the actual
// count early-return.
@group(0) @binding(5) var<storage, read> instance_count: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn bin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inst_idx = gid.x;
    if inst_idx >= bin_params.instance_count_upper_bound { return; }
    if inst_idx >= instance_count[0] { return; }

    let aabb_min = instance_aabbs[inst_idx * 2u + 0u].xyz;
    let aabb_max = instance_aabbs[inst_idx * 2u + 1u].xyz;

    // Project all 8 AABB corners to clip space and find the screen-
    // space AABB. Three correctness gotchas to handle:
    //
    //   1. Corners with w <= 0 are behind the camera; their projected
    //      NDC is meaningless. If ALL corners are behind, the instance
    //      is fully behind the camera — skip.
    //   2. If SOME corners are behind, the projected AABB is incomplete.
    //      Conservatively, an instance straddling the near plane may
    //      cover any part of the visible screen. We return early to
    //      avoid binning to wrong tiles; it'd be visually pop-out as
    //      the camera approaches, but keeps perf sane (the alternative
    //      is binning every straddler to the entire screen).
    //   3. Off-screen corners (NDC outside [-1, 1]) are projected past
    //      the screen edges. The pixel-AABB clamp below handles
    //      partial off-screen, but if the entire AABB is off-screen
    //      the clamp would still bin to the edge — explicitly reject.
    var screen_min = vec2<f32>(1.0e30);
    var screen_max = vec2<f32>(-1.0e30);
    var corners_in_front: u32 = 0u;
    for (var ci: u32 = 0u; ci < 8u; ci = ci + 1u) {
        let cx = select(aabb_min.x, aabb_max.x, (ci & 1u) != 0u);
        let cy = select(aabb_min.y, aabb_max.y, (ci & 2u) != 0u);
        let cz = select(aabb_min.z, aabb_max.z, (ci & 4u) != 0u);
        let clip = camera.view_proj * vec4<f32>(cx, cy, cz, 1.0);
        if clip.w <= 1.0e-4 { continue; }
        corners_in_front = corners_in_front + 1u;
        let ndc = clip.xy / clip.w;
        screen_min = min(screen_min, ndc);
        screen_max = max(screen_max, ndc);
    }
    // Fully behind camera, or straddler we conservatively skip.
    if corners_in_front == 0u { return; }
    if corners_in_front < 8u { return; }
    // Fully off-screen — the visible NDC range is [-1, 1] on each axis.
    if screen_max.x < -1.0 || screen_min.x > 1.0
        || screen_max.y < -1.0 || screen_min.y > 1.0 {
        return;
    }
    // Clamp to the visible NDC range so the pixel conversion below
    // doesn't generate out-of-screen tile coords (which `clamp()`
    // would silently fold to the edge tile, over-binning).
    screen_min = max(screen_min, vec2<f32>(-1.0));
    screen_max = min(screen_max, vec2<f32>(1.0));

    // NDC → pixel coords. Y flips so the "min NDC y" corner becomes
    // the "max pixel y"; sort with min/max.
    let res = camera.resolution.xy;
    let pix_min = (screen_min * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * res;
    let pix_max = (screen_max * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * res;
    let pix_lo = min(pix_min, pix_max);
    let pix_hi = max(pix_min, pix_max);

    // Convert to 8×8 tile coords.
    let total_tiles_x = bin_params.tile_count_x;
    let total_tiles_y = bin_params.tile_count_y;
    let tx_min = u32(clamp(floor(pix_lo.x / 8.0), 0.0, f32(total_tiles_x - 1u)));
    let ty_min = u32(clamp(floor(pix_lo.y / 8.0), 0.0, f32(total_tiles_y - 1u)));
    let tx_max = u32(clamp(floor(pix_hi.x / 8.0), 0.0, f32(total_tiles_x - 1u)));
    let ty_max = u32(clamp(floor(pix_hi.y / 8.0), 0.0, f32(total_tiles_y - 1u)));

    for (var ty: u32 = ty_min; ty <= ty_max; ty = ty + 1u) {
        for (var tx: u32 = tx_min; tx <= tx_max; tx = tx + 1u) {
            let tile_id = ty * total_tiles_x + tx;
            let slot = atomicAdd(&tile_counts[tile_id], 1u);
            if slot < MAX_INSTANCES_PER_TILE {
                tile_lists[tile_id * MAX_INSTANCES_PER_TILE + slot] = inst_idx;
            }
        }
    }
}
