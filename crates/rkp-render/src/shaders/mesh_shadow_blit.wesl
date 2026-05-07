// Mesh-shadow depth → shadow_buffer blit (compute, per-cascade).
//
// One thread per shadow-map texel. Reads the depth texture
// `MeshShadowMapPass` filled (depth-only rasterization at cascade
// `params.cascade_index`), bitcasts f32 → u32, and writes the result
// into the matching slice of `shadow_buffer` (cascade-major layout:
// `[cascade * SIZE² + ty * SIZE + tx]`). No atomic ops — each thread
// writes a unique texel, so there's no race.
//
// Texels not covered by any triangle keep the depth attachment's
// clear value (1.0); after bitcast that's `0x3F800000` =
// `SHADOW_MAP_FAR_DEPTH_BITS`, exactly the "no occluder" sentinel
// shade's `sample_shadow_map` expects.

const SHADOW_MAP_SIZE: u32 = 1024u;

struct BlitParams {
    cascade_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> shadow_buffer: array<u32>;
@group(0) @binding(2) var<uniform> params: BlitParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= SHADOW_MAP_SIZE || gid.y >= SHADOW_MAP_SIZE {
        return;
    }
    let depth = textureLoad(depth_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let slice = SHADOW_MAP_SIZE * SHADOW_MAP_SIZE;
    let idx = params.cascade_index * slice + gid.y * SHADOW_MAP_SIZE + gid.x;
    shadow_buffer[idx] = bitcast<u32>(depth);
}
