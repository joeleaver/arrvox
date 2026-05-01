// Phase 8 V2 — clear the shadow buffer to FAR_DEPTH bits.
//
// Run once per frame before any scatter dispatches. Each thread
// stores `bitcast<u32>(1.0) = 0x3F800000` into one entry of the
// `shadow_buffer`. Scatter dispatches then atomicMin the actual
// nearest-occluder depth on top.
//
// The buffer holds f32 depths bit-cast to u32 so atomicMin works
// directly: f32 in [0, 1] is monotonic in its u32 bit pattern.

const FAR_DEPTH_BITS: u32 = 0x3F800000u; // bitcast<u32>(1.0)

@group(0) @binding(0) var<storage, read_write> shadow_buffer: array<atomic<u32>>;

// Dispatched as a 2D grid of 8×8 workgroups so a 2K×2K shadow
// buffer fits within wgpu's per-dimension dispatch limit (65535).
// `gid.xy` directly indexes a (tx, ty) within the W×H shadow map;
// the engine sets dispatch dims to `(W/8, H/8, 1)`.
@compute @workgroup_size(8, 8, 1)
fn clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = gid.x;
    let ty = gid.y;
    // Pack the 2D coord into a 1D index. `arrayLength` gives the
    // total entry count; the engine sized the buffer to W*H so
    // `ty * W + tx` is the natural mapping. Threads beyond the
    // buffer (rare — only when W or H isn't a multiple of 8) bail.
    let len = arrayLength(&shadow_buffer);
    // The engine knows W and H but the shader doesn't have direct
    // access; use `len`'s implicit square assumption — V1 shadow
    // maps are square, so `W = sqrt(len)` and `idx = ty * W + tx`.
    // For a 2K square: len = 4_194_304, W = 2048.
    let w = u32(sqrt(f32(len)));
    let idx = ty * w + tx;
    if idx >= len { return; }
    atomicStore(&shadow_buffer[idx], FAR_DEPTH_BITS);
}
