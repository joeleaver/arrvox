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

const MAX_INSTANCES_PER_TILE: u32 = 1024u;

struct BinParams {
    instance_count_upper_bound: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    // Threads per Y-stripe — `dispatch_x * 64`. The CPU side writes
    // this so the shader can rebuild the linear instance index when
    // the dispatch is split across X+Y to dodge wgpu's 65535 per-dim
    // workgroup-count limit.
    dispatch_x_threads: u32,
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

// Project a 2D axis-aligned rectangle onto a 1D axis. Returns
// `vec2(min, max)` of the projection over the rect's 4 corners.
fn project_rect_on_axis(
    axis: vec2<f32>,
    lo: vec2<f32>,
    hi: vec2<f32>,
) -> vec2<f32> {
    // For an axis-aligned rect, the extreme projections sit at opposite
    // corners chosen by the sign of each axis component — no need to
    // iterate the 4 corners.
    let xL = min(axis.x * lo.x, axis.x * hi.x);
    let xH = max(axis.x * lo.x, axis.x * hi.x);
    let yL = min(axis.y * lo.y, axis.y * hi.y);
    let yH = max(axis.y * lo.y, axis.y * hi.y);
    return vec2<f32>(xL + yL, xH + yH);
}

@compute @workgroup_size(64, 1, 1)
fn bin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let inst_idx = gid.y * bin_params.dispatch_x_threads + gid.x;
    if inst_idx >= bin_params.instance_count_upper_bound { return; }
    if inst_idx >= instance_count[0] { return; }

    let aabb_min = instance_aabbs[inst_idx * 2u + 0u].xyz;
    let aabb_max = instance_aabbs[inst_idx * 2u + 1u].xyz;

    // Project all 8 AABB corners to clip space, then to pixel space.
    // Three correctness gotchas:
    //
    //   1. Corners with w <= 0 are behind the camera; their projected
    //      NDC is meaningless. If ALL corners are behind, the instance
    //      is fully behind the camera — skip.
    //   2. If SOME corners are behind, the projected polygon is
    //      incomplete. Conservatively skip; visible pop-in as the
    //      camera approaches, but keeps perf sane.
    //   3. Off-screen corners (NDC outside [-1, 1]) — explicit reject
    //      below if the bounding rect is fully off-screen.
    let res = camera.resolution.xy;
    var corners_pix: array<vec2<f32>, 8>;
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
        corners_pix[ci] = (ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * res;
    }
    if corners_in_front == 0u { return; }
    if corners_in_front < 8u { return; }
    if screen_max.x < -1.0 || screen_min.x > 1.0
        || screen_max.y < -1.0 || screen_min.y > 1.0 {
        return;
    }
    let clamped_min = max(screen_min, vec2<f32>(-1.0));
    let clamped_max = min(screen_max, vec2<f32>(1.0));
    let pix_a = (clamped_min * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * res;
    let pix_b = (clamped_max * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * res;
    let pix_lo = min(pix_a, pix_b);
    let pix_hi = max(pix_a, pix_b);

    let total_tiles_x = bin_params.tile_count_x;
    let total_tiles_y = bin_params.tile_count_y;
    let tx_min = u32(clamp(floor(pix_lo.x / 8.0), 0.0, f32(total_tiles_x - 1u)));
    let ty_min = u32(clamp(floor(pix_lo.y / 8.0), 0.0, f32(total_tiles_y - 1u)));
    let tx_max = u32(clamp(floor(pix_hi.x / 8.0), 0.0, f32(total_tiles_x - 1u)));
    let ty_max = u32(clamp(floor(pix_hi.y / 8.0), 0.0, f32(total_tiles_y - 1u)));

    // SAT (Separating Axis Theorem) tightening. The bounding rect we
    // just computed already covers the two screen-axis-aligned SAT
    // candidates (X and Y of the screen). What's missing is the
    // polygon's own edge directions: a slender blade rotated in 3D
    // projects to a thin diagonal polygon whose bounding rect is far
    // wider than the polygon itself. Adding three more separating
    // axes — perpendiculars to the three projected world-axis edges
    // (corner 0→1 = world X, 0→2 = world Y, 0→4 = world Z) — catches
    // most of that slack on slender geometry without the cost of
    // computing a true convex hull on GPU. The 4 parallel cube edges
    // along each world axis aren't exactly parallel under perspective,
    // so picking corner 0's representative edge is approximate; in
    // practice the dominant silhouette directions land within a few
    // degrees of these picks for typical viewing angles.
    let edge_x = corners_pix[1u] - corners_pix[0u];
    let edge_y = corners_pix[2u] - corners_pix[0u];
    let edge_z = corners_pix[4u] - corners_pix[0u];
    let n_x = vec2<f32>(-edge_x.y, edge_x.x);
    let n_y = vec2<f32>(-edge_y.y, edge_y.x);
    let n_z = vec2<f32>(-edge_z.y, edge_z.x);

    // Polygon's [min, max] projection on each separating axis (over
    // all 8 corners — the polygon's vertices are a subset).
    var poly_min = vec3<f32>(1.0e30);
    var poly_max = vec3<f32>(-1.0e30);
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let p = corners_pix[i];
        let dx = dot(p, n_x);
        let dy = dot(p, n_y);
        let dz = dot(p, n_z);
        poly_min = min(poly_min, vec3<f32>(dx, dy, dz));
        poly_max = max(poly_max, vec3<f32>(dx, dy, dz));
    }

    for (var ty: u32 = ty_min; ty <= ty_max; ty = ty + 1u) {
        for (var tx: u32 = tx_min; tx <= tx_max; tx = tx + 1u) {
            let tlo = vec2<f32>(f32(tx) * 8.0, f32(ty) * 8.0);
            let thi = vec2<f32>(f32(tx + 1u) * 8.0, f32(ty + 1u) * 8.0);
            let r_x = project_rect_on_axis(n_x, tlo, thi);
            if r_x.y < poly_min.x || r_x.x > poly_max.x { continue; }
            let r_y = project_rect_on_axis(n_y, tlo, thi);
            if r_y.y < poly_min.y || r_y.x > poly_max.y { continue; }
            let r_z = project_rect_on_axis(n_z, tlo, thi);
            if r_z.y < poly_min.z || r_z.x > poly_max.z { continue; }

            let tile_id = ty * total_tiles_x + tx;
            let slot = atomicAdd(&tile_counts[tile_id], 1u);
            if slot < MAX_INSTANCES_PER_TILE {
                tile_lists[tile_id * MAX_INSTANCES_PER_TILE + slot] = inst_idx;
            }
        }
    }
}
