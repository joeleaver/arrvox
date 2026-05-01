// Phase 6 Session 3c — user-shader tile-cull scatter pass.
//
// Per `InstanceTileCullEntry`, project the world AABB to its screen
// tile rectangle (same math as the count pass) and write a 16-byte
// `UserShaderTileEntry` into `us_tile_entries` for each covered tile.
// Slot allocation uses an atomic cursor per tile, initialized to the
// tile's prefix-sum start by the engine (a `queue.copy_buffer_to_buffer`
// from `us_tile_offsets[..tile_count]` into `us_tile_scatter_cursor[]`).
//
// `us_tile_offsets` is left untouched — the host march reads from it.
//
// ## Bindings
//
// * group(0) binding(0): `tile_cull_scratch`           (read)
// * group(0) binding(1): `us_tile_scatter_cursor`      (atomic rw)
// * group(0) binding(2): `us_tile_entries`             (rw)
// * group(1) binding(0): `TileCullViewportUniform`     (uniform)
//
// `TileCullViewportUniform` mirrors the count pass exactly so engine
// can reuse one upload per VR.

const TILE_PX: u32 = 8u;
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

// 48 B per entry — matches `UserShaderTileEntry` in
// `octree_march.rs` / `octree_march.wgsl`. The host march iterates
// these alongside `tile_object_ids`. World AABB + screen-pixel rect
// are cached here so the per-pixel march path doesn't re-call the
// user shader's `inst_aabb` hook and can pre-reject pixels outside
// the blade's screen footprint with a 4-compare. WGSL `vec3<f32>` +
// trailing `u32` packs into the vec3's 16-byte alignment slot
// (offsets 0..12, 12..16) — same trick as `InstanceTileCullEntry`.
struct UserShaderTileEntry {
    aabb_min: vec3<f32>,
    asset_id: u32,
    aabb_max: vec3<f32>,
    instance_state_offset: u32,
    material_id: u32,
    // (screen_min_x as u16) | ((screen_min_y as u16) << 16). Pixel
    // coords clamped to viewport bounds.
    screen_min_packed: u32,
    screen_max_packed: u32,
    _pad: u32,
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
@group(0) @binding(1) var<storage, read_write> us_tile_scatter_cursor: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> us_tile_entries: array<UserShaderTileEntry>;

@group(1) @binding(0) var<uniform> vp: TileCullViewportUniform;

struct TileRect {
    tile_min_x: u32,
    tile_min_y: u32,
    tile_max_x: u32,
    tile_max_y: u32,
    valid: u32,
    // Per-pixel screen bounds (clamped to viewport, inclusive). Cached
    // into the entry so the per-pixel march loop can reject pixels
    // outside the blade footprint without calling the user shader.
    pixel_min_x: u32,
    pixel_min_y: u32,
    pixel_max_x: u32,
    pixel_max_y: u32,
}

// Near-plane epsilon. Must match `user_shader_tile_count.wgsl`.
const NEAR_EPS: f32 = 1.0e-3;

fn project_world_aabb_to_tiles(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> TileRect {
    var rect: TileRect;
    rect.valid = 0u;

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

    if (n_in_front == 0u) { return rect; }

    var min_ndc = vec2<f32>( 1.0e30,  1.0e30);
    var max_ndc = vec2<f32>(-1.0e30, -1.0e30);

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (in_front[i] == 1u) {
            let ndc = clips[i].xyz / clips[i].w;
            min_ndc = min(min_ndc, ndc.xy);
            max_ndc = max(max_ndc, ndc.xy);
        }
    }

    // Edges crossing the near plane: lerp at clip.w == NEAR_EPS,
    // project. Same algorithm as count pass — must produce the
    // identical TileRect or count and scatter disagree on tile sets.
    if (n_in_front < 8u) {
        for (var c: u32 = 0u; c < 8u; c = c + 1u) {
            for (var ax: u32 = 0u; ax < 3u; ax = ax + 1u) {
                let mask = 1u << ax;
                let other = c ^ mask;
                if (c >= other) { continue; }
                if (in_front[c] == in_front[other]) { continue; }
                let ca = clips[c];
                let cb = clips[other];
                let denom = cb.w - ca.w;
                let t = select((NEAR_EPS - ca.w) / denom, 0.5, abs(denom) < 1e-12);
                let clip_at = mix(ca, cb, t);
                let ndc_xy = clip_at.xy / max(clip_at.w, NEAR_EPS);
                min_ndc = min(min_ndc, ndc_xy);
                max_ndc = max(max_ndc, ndc_xy);
            }
        }
    }

    let px_min_x = (min_ndc.x * 0.5 + 0.5) * vp.resolution_x;
    let px_min_y = (-max_ndc.y * 0.5 + 0.5) * vp.resolution_y;
    let px_max_x = (max_ndc.x * 0.5 + 0.5) * vp.resolution_x;
    let px_max_y = (-min_ndc.y * 0.5 + 0.5) * vp.resolution_y;

    let tile_min_x_f = floor(px_min_x / f32(TILE_PX));
    let tile_min_y_f = floor(px_min_y / f32(TILE_PX));
    let tile_max_x_f = floor(px_max_x / f32(TILE_PX));
    let tile_max_y_f = floor(px_max_y / f32(TILE_PX));

    if (tile_max_x_f < 0.0 || tile_max_y_f < 0.0) { return rect; }
    if (tile_min_x_f >= f32(vp.tile_count_x)) { return rect; }
    if (tile_min_y_f >= f32(vp.tile_count_y)) { return rect; }

    rect.tile_min_x = u32(max(tile_min_x_f, 0.0));
    rect.tile_min_y = u32(max(tile_min_y_f, 0.0));
    rect.tile_max_x = u32(min(tile_max_x_f, f32(vp.tile_count_x - 1u)));
    rect.tile_max_y = u32(min(tile_max_y_f, f32(vp.tile_count_y - 1u)));
    // Per-pixel bounds — clamp to [0, resolution-1] so the cached
    // values in the entry never sit outside the viewport. The
    // per-pixel reject in march does an inclusive [min, max] test.
    let max_x = max(vp.resolution_x - 1.0, 0.0);
    let max_y = max(vp.resolution_y - 1.0, 0.0);
    rect.pixel_min_x = u32(clamp(floor(px_min_x), 0.0, max_x));
    rect.pixel_min_y = u32(clamp(floor(px_min_y), 0.0, max_y));
    rect.pixel_max_x = u32(clamp(floor(px_max_x), 0.0, max_x));
    rect.pixel_max_y = u32(clamp(floor(px_max_y), 0.0, max_y));
    rect.valid = 1u;
    return rect;
}

@compute @workgroup_size(64, 1, 1)
fn tile_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vp.scratch_count) { return; }

    let entry = tile_cull_scratch[i];
    if (entry.live == 0u) { return; }

    let rect = project_world_aabb_to_tiles(entry.aabb_min, entry.aabb_max);
    if (rect.valid == 0u) { return; }

    let span_x = min(rect.tile_max_x - rect.tile_min_x + 1u, MAX_TILE_SPAN);
    let span_y = min(rect.tile_max_y - rect.tile_min_y + 1u, MAX_TILE_SPAN);

    let screen_min_packed =
        (rect.pixel_min_x & 0xFFFFu) | ((rect.pixel_min_y & 0xFFFFu) << 16u);
    let screen_max_packed =
        (rect.pixel_max_x & 0xFFFFu) | ((rect.pixel_max_y & 0xFFFFu) << 16u);

    var out_entry: UserShaderTileEntry;
    out_entry.aabb_min = entry.aabb_min;
    out_entry.asset_id = entry.asset_id;
    out_entry.aabb_max = entry.aabb_max;
    out_entry.instance_state_offset = entry.instance_state_offset;
    out_entry.material_id = entry.material_id;
    out_entry.screen_min_packed = screen_min_packed;
    out_entry.screen_max_packed = screen_max_packed;
    out_entry._pad = 0u;

    for (var ty: u32 = 0u; ty < span_y; ty = ty + 1u) {
        let abs_y = rect.tile_min_y + ty;
        for (var tx: u32 = 0u; tx < span_x; tx = tx + 1u) {
            let abs_x = rect.tile_min_x + tx;
            let tile_idx = abs_y * vp.tile_count_x + abs_x;
            // atomicAdd returns the OLD cursor value — that's our slot
            // index in us_tile_entries (= prefix-offset[tile] +
            // local_count_so_far).
            let slot = atomicAdd(&us_tile_scatter_cursor[tile_idx], 1u);
            us_tile_entries[slot] = out_entry;
        }
    }
}
