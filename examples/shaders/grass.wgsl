// grass.wgsl — V1 mesh-path reference user shader.
//
// Replaces the previous Option B (proto-bake + emit-pass) demo.
// This shader runs through the V1 mesh-path: a per-painted-leaf
// dynamic spawn count, then a hardware-instanced vertex shader that
// emits one triangle-quad per blade directly into the G-buffer.
//
// Drop this file (or a copy named after your material's shader name)
// into your project's `assets/shaders/`. Then create a material whose
// `shader` field is `"grass"` and paint that material onto a surface.
//
// V1 mesh-path API (see notes/user-shaders-mesh.md):
//   · `@geometry procedural { vertex_count: N }` — N verts per blade,
//     VS reads `@builtin(vertex_index)` 0..N-1.
//   · `fn spawn_count(anchor, frame) -> u32` — required. Returns how
//     many blades this painted-leaf anchor should emit.
//   · `fn vs(anchor, spawn_idx, vid, frame) -> VsOut` — required.
//     Computes world position + shading payload for one vertex.
//   · `fn spawn_alive(anchor, spawn_idx, frame) -> bool` — optional
//     last-mile filter. Default true.
//   · `fn fs(in: VsOut) -> FsOut` — optional. Default packs anchor
//     material + interpolated color into the G-buffer.

// ── Geometry: 6 verts (2 triangles, 1 quad) per blade ─────────────
// @geometry procedural { vertex_count: 6 }
// `@animated` is informational (signals wind sway in `vs`); does not
// gate caching since spawn_count is `f(anchor)` only.
// @animated

// ── Per-material params ───────────────────────────────────────────
// @param blade_height: f32 = 0.35, range = [0.05, 1.5]
// @param blade_width:  f32 = 0.04, range = [0.01, 0.2]
// @param density:      f32 = 80.0, range = [1.0, 400.0]
// @param wind_amp:     f32 = 0.08, range = [0.0, 0.3]
// @param wind_freq:    f32 = 1.5,  range = [0.0, 6.0]

// ── Helpers ───────────────────────────────────────────────────────
fn grass_hash_u01(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / 4294967295.0;
}

// ── spawn_count ───────────────────────────────────────────────────
// Density-based: `blades = density × surface_area`. anchor.surface_area
// is the painted-leaf face's area (V1 def: `leaf_size²`). Hard cap at
// 64 per V1's per-anchor spawn ceiling (`MAX_SPAWNS_PER_ANCHOR_V1`).
fn spawn_count(anchor: AnchorContext, frame: FrameContext) -> u32 {
    let density = ctx_param(2);
    let raw = density * anchor.surface_area;
    return u32(clamp(raw, 0.0, 64.0));
}

// ── vs — one blade quad per (anchor, spawn_idx) ───────────────────
// vid 0..5 maps to a 2-triangle quad standing upright at the anchor:
//   tri 1: (0,0), (1,0), (1,1)
//   tri 2: (0,0), (1,1), (0,1)
// where (u, v) ∈ [0,1]² is local blade space (u = width axis, v = height).
fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    let blade_height = ctx_param(0);
    let blade_width  = ctx_param(1);
    let wind_amp     = ctx_param(3);
    let wind_freq    = ctx_param(4);

    // Per-spawn deterministic seeds (stable across frames so blades
    // don't shimmer).
    let s0 = anchor.seed ^ (spawn_idx * 0x9E3779B9u);
    let r_jx     = grass_hash_u01(s0 ^ 0xBF58476Du);
    let r_jz     = grass_hash_u01(s0 ^ 0x94D049BBu);
    let r_yaw    = grass_hash_u01(s0 ^ 0xA2B5C7D9u);
    let r_height = grass_hash_u01(s0 ^ 0xCBF29CE4u);
    let r_phase  = grass_hash_u01(s0 ^ 0xFEEDFACEu);

    // Anchor's leaf footprint — half-size. Jitter blade position
    // inside the leaf so a 1024-leaf paint patch doesn't show a grid.
    let jx = (r_jx - 0.5) * 2.0 * anchor.leaf_extent;
    let jz = (r_jz - 0.5) * 2.0 * anchor.leaf_extent;

    // Per-blade height jitter.
    let h = blade_height * (0.7 + r_height * 0.6);

    // Per-blade yaw rotation around Y.
    let yaw = r_yaw * 6.28318530718;
    let c = cos(yaw);
    let s = sin(yaw);

    // Map vid → (u, v).
    var u: f32 = 0.0;
    var v: f32 = 0.0;
    if (vid == 0u)      { u = 0.0; v = 0.0; }
    else if (vid == 1u) { u = 1.0; v = 0.0; }
    else if (vid == 2u) { u = 1.0; v = 1.0; }
    else if (vid == 3u) { u = 0.0; v = 0.0; }
    else if (vid == 4u) { u = 1.0; v = 1.0; }
    else                { u = 0.0; v = 1.0; }

    // Local blade-space coords. Tapered toward tip; tip width = 20%
    // of base width.
    let local_x = (u - 0.5) * blade_width * (1.0 - v * 0.8);
    let local_y = v * h;
    let local_z = 0.0;

    // Wind sway — top of blade displaces in world XZ as `v`-weighted
    // sinusoid. Per-blade phase keeps neighbors out of phase.
    let wind_phase = r_phase * 6.28318530718;
    let wind_x = sin(frame.time * wind_freq + wind_phase) * wind_amp * v;
    let wind_z = cos(frame.time * wind_freq + wind_phase * 0.73) * wind_amp * v;

    // Rotate local XZ by yaw, then translate to anchor + jitter.
    let rotated_x = local_x * c - local_z * s + wind_x;
    let rotated_z = local_x * s + local_z * c + wind_z;

    let world_pos = vec3<f32>(
        anchor.world_pos.x + jx + rotated_x,
        anchor.world_pos.y + local_y,
        anchor.world_pos.z + jz + rotated_z,
    );

    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    // V1 normal: anchor surface normal. Could fancy this with a
    // blade-tangent + curvature in V1.1; this keeps the grass lit
    // correctly relative to the painted surface.
    out.world_normal = anchor.surface_normal;
    out.material_packed = anchor.material_id;
    out.color_rgb = anchor.host_color.rgb;
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}
