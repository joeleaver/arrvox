// Phase 6 Session 3a — user-shader tile-cull count pass.
//
// Per `InstanceTileCullEntry` produced by Session 2's AABB pass:
//
// 1. If `live == 0`, return.
// 2. Project the 8 world-AABB corners through `view_proj` and compute
//    the tight screen-space [tile_min, tile_max] rectangle the AABB
//    covers in this viewport's tile grid.
// 3. For each tile in that rectangle, `atomicAdd(us_tile_counts[t], 1)`.
//
// The downstream prefix-sum + scatter passes turn the resulting
// per-tile counts into a flat `us_tile_entries` array partitioned by
// tile.
//
// Per-VP: tile_count + view_proj + resolution all live in
// `TileCullViewportUniform`, uploaded once per dispatch by the engine.

const TILE_PX: u32 = 8u;
// Cap per-entry tile-span to avoid runaway atomic loops if the world
// AABB projects degenerate (e.g. a corner straddles the near plane and
// the screen AABB blows up). Real instance AABBs cover <= 32×32 tiles
// at typical viewing distances; 256 is a safety net.
const MAX_TILE_SPAN: u32 = 256u;

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

struct TileCullViewportUniform {
    view_proj: mat4x4<f32>,
    resolution_x: f32,
    resolution_y: f32,
    tile_count_x: u32,
    tile_count_y: u32,
    tile_count: u32,
    scratch_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> tile_cull_scratch: array<InstanceTileCullEntry>;
@group(0) @binding(1) var<storage, read_write> us_tile_counts: array<atomic<u32>>;

@group(1) @binding(0) var<uniform> vp: TileCullViewportUniform;

// Project a world-space AABB through `view_proj` into screen tile
// coordinates. Returns `(tile_min, tile_max)` in [0, tile_count_xy)
// inclusive of `tile_max`. When any corner falls behind the camera
// (clip.w <= 0), returns the full visible tile rectangle — a
// conservative upper bound that keeps correctness without the bookkeep
// of plane-clipping the AABB.
//
// Returns `(vec2<u32>(MAX, MAX), vec2<u32>(MAX, MAX))` on a fully-off-
// screen AABB so the caller can early-skip. Real off-screen AABBs are
// rare for user-shader instances (they only exist where paint exists);
// this is here for correctness, not perf.
struct TileRect {
    tile_min_x: u32,
    tile_min_y: u32,
    tile_max_x: u32,
    tile_max_y: u32,
    valid: u32,
}

// Near-plane epsilon. Corners with `clip.w <= NEAR_EPS` are treated
// as "behind camera"; AABB edges crossing this boundary are clipped
// at `clip.w = NEAR_EPS` so the projection stays well-defined.
const NEAR_EPS: f32 = 1.0e-3;

fn project_world_aabb_to_tiles(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> TileRect {
    var rect: TileRect;
    rect.valid = 0u;

    // Project all 8 corners to clip space; track which are in front.
    var clips: array<vec4<f32>, 8>;
    var in_front: array<u32, 8>;
    var n_in_front: u32 = 0u;

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let cx = select(aabb_min.x, aabb_max.x, (i & 1u) != 0u);
        let cy = select(aabb_min.y, aabb_max.y, (i & 2u) != 0u);
        let cz = select(aabb_min.z, aabb_max.z, (i & 4u) != 0u);
        let clip = vp.view_proj * vec4<f32>(cx, cy, cz, 1.0);
        clips[i] = clip;
        if (clip.w > NEAR_EPS) {
            in_front[i] = 1u;
            n_in_front = n_in_front + 1u;
        } else {
            in_front[i] = 0u;
        }
    }

    // No corners in front → AABB fully behind camera, invisible.
    if (n_in_front == 0u) { return rect; }

    var min_ndc = vec2<f32>( 1.0e30,  1.0e30);
    var max_ndc = vec2<f32>(-1.0e30, -1.0e30);

    // Project in-front corners normally.
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (in_front[i] == 1u) {
            let ndc = clips[i].xyz / clips[i].w;
            min_ndc = min(min_ndc, ndc.xy);
            max_ndc = max(max_ndc, ndc.xy);
        }
    }

    // Mixed visibility: clip the 12 AABB edges against the near plane
    // at w = NEAR_EPS, project the intersection points. Iterate edges
    // by enumerating low corner + flip axis; visit each pair once via
    // `c < other`. Replaces the previous "any_behind → full screen"
    // broadcast that caused us_tile_entries overflow when the camera
    // got close to a blade.
    if (n_in_front < 8u) {
        for (var c: u32 = 0u; c < 8u; c = c + 1u) {
            for (var ax: u32 = 0u; ax < 3u; ax = ax + 1u) {
                let mask = 1u << ax;
                let other = c ^ mask;
                if (c >= other) { continue; }
                if (in_front[c] == in_front[other]) { continue; }
                // One in, one out — lerp at clip.w == NEAR_EPS.
                let ca = clips[c];
                let cb = clips[other];
                let denom = cb.w - ca.w;
                // Robust against zero-length denominator (parallel to
                // near plane) — both endpoints are near eps, take the
                // midpoint.
                let t = select((NEAR_EPS - ca.w) / denom, 0.5, abs(denom) < 1e-12);
                let clip_at = mix(ca, cb, t);
                let ndc_xy = clip_at.xy / max(clip_at.w, NEAR_EPS);
                min_ndc = min(min_ndc, ndc_xy);
                max_ndc = max(max_ndc, ndc_xy);
            }
        }
    }

    // Convert NDC → pixel → tile.
    let px_min_x = (min_ndc.x * 0.5 + 0.5) * vp.resolution_x;
    let px_min_y = (-max_ndc.y * 0.5 + 0.5) * vp.resolution_y;  // flip y
    let px_max_x = (max_ndc.x * 0.5 + 0.5) * vp.resolution_x;
    let px_max_y = (-min_ndc.y * 0.5 + 0.5) * vp.resolution_y;

    let tile_min_x_f = floor(px_min_x / f32(TILE_PX));
    let tile_min_y_f = floor(px_min_y / f32(TILE_PX));
    let tile_max_x_f = floor(px_max_x / f32(TILE_PX));
    let tile_max_y_f = floor(px_max_y / f32(TILE_PX));

    // Off-screen entirely if any axis lies wholly outside [0, tc) range.
    if (tile_max_x_f < 0.0 || tile_max_y_f < 0.0) { return rect; }
    if (tile_min_x_f >= f32(vp.tile_count_x)) { return rect; }
    if (tile_min_y_f >= f32(vp.tile_count_y)) { return rect; }

    rect.tile_min_x = u32(max(tile_min_x_f, 0.0));
    rect.tile_min_y = u32(max(tile_min_y_f, 0.0));
    rect.tile_max_x = u32(min(tile_max_x_f, f32(vp.tile_count_x - 1u)));
    rect.tile_max_y = u32(min(tile_max_y_f, f32(vp.tile_count_y - 1u)));
    rect.valid = 1u;
    return rect;
}

@compute @workgroup_size(64, 1, 1)
fn tile_count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vp.scratch_count) { return; }

    let entry = tile_cull_scratch[i];
    if (entry.live == 0u) { return; }

    let rect = project_world_aabb_to_tiles(entry.aabb_min, entry.aabb_max);
    if (rect.valid == 0u) { return; }

    // Cap span — see MAX_TILE_SPAN comment.
    let span_x = min(rect.tile_max_x - rect.tile_min_x + 1u, MAX_TILE_SPAN);
    let span_y = min(rect.tile_max_y - rect.tile_min_y + 1u, MAX_TILE_SPAN);

    for (var ty: u32 = 0u; ty < span_y; ty = ty + 1u) {
        let abs_y = rect.tile_min_y + ty;
        for (var tx: u32 = 0u; tx < span_x; tx = tx + 1u) {
            let abs_x = rect.tile_min_x + tx;
            let tile_idx = abs_y * vp.tile_count_x + abs_x;
            atomicAdd(&us_tile_counts[tile_idx], 1u);
        }
    }
}
