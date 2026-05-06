// Mesh-shadow depth → shadow_buffer blit (compute).
//
// One thread per shadow-map texel. Reads the depth texture
// `MeshShadowMapPass` filled (via depth-only rasterization), bitcasts
// f32 → u32, and writes the result into the existing `shadow_buffer`
// storage that shade already samples. No atomic ops — each thread
// writes a unique texel, so there's no race.
//
// Texels not covered by any triangle keep the depth attachment's
// clear value (1.0); after bitcast that's `0x3F800000` =
// `SHADOW_MAP_FAR_DEPTH_BITS`, exactly the "no occluder" sentinel
// shade's `sample_shadow_map` expects.

const SHADOW_MAP_SIZE: u32 = 1024u;

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> shadow_buffer: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= SHADOW_MAP_SIZE || gid.y >= SHADOW_MAP_SIZE {
        return;
    }
    let depth = textureLoad(depth_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let idx = gid.y * SHADOW_MAP_SIZE + gid.x;
    shadow_buffer[idx] = bitcast<u32>(depth);
}
