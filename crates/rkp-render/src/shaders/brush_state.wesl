// Brush-state probe — single-thread compute that reads the G-buffer
// at the cursor pixel and writes a 32-byte `BrushState` buffer
// consumed by `rkp_shade.wesl` to draw the screen-space paint cursor.
//
// Why this exists: the prior cursor used a sim-thread geodesic flood
// fill over the asset's octree, gated on a pick readback round-trip
// — cursor-to-photon latency was 80 ms+. This pass replaces the
// round-trip with one texture read at the live mouse pixel, so the
// cursor follows the pointer at frame rate with zero CPU work per
// hover.
//
// The `gbuf_position` texture carries `(world_pos.xyz, hit_distance)`
// at every pixel; `gbuf_pick` carries the hit entity's `gpu_idx`
// (`u32::MAX` for sky). Reading both at the cursor coord gives us
// everything `rkp_shade` needs — the world center for the brush
// sphere intersect and the entity id for the selection-lock check.

import package::lib::types::BrushState;

struct BrushParams {
    /// Cursor pixel in viewport coords. Must satisfy
    /// `cursor_xy < (width, height)` — the dispatcher clamps before
    /// upload, so the shader doesn't bounds-check.
    cursor_xy: vec2<u32>,
    /// 1 = paint mode active (write the live cursor cell);
    /// 0 = paint inactive (write the miss sentinel so the shader
    /// hides the cursor without any CPU coordination).
    enabled: u32,
    _pad0: u32,
}

@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_pick:     texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> brush_state: BrushState;
@group(0) @binding(3) var<uniform> params: BrushParams;

@compute @workgroup_size(1)
fn cs_main() {
    if (params.enabled == 0u) {
        // Sentinel matches what the gbuf clears to on a sky pixel —
        // shade reads `hit_distance >= 1e9` as "no cursor."
        brush_state.center_world = vec3<f32>(0.0);
        brush_state.hit_distance = 1.0e10;
        brush_state.hit_object_id = 0xFFFFFFFFu;
        return;
    }
    let coord = vec2<i32>(i32(params.cursor_xy.x), i32(params.cursor_xy.y));
    let p = textureLoad(gbuf_position, coord, 0);
    let pick = textureLoad(gbuf_pick, coord, 0).r;
    brush_state.center_world = p.xyz;
    brush_state.hit_distance = p.w;
    brush_state.hit_object_id = pick;
}
