// V1 mesh-path user-shader raster — vertex + fragment entry points.
//
// This file is the engine-authored skeleton. The shader composer
// replaces the `// USER_BODY_BEGIN` / `// USER_BODY_END` block below
// with the user's parsed module (helper fns + spawn_count / vs /
// spawn_alive / fs). On its own this file is valid WGSL — the stub
// bodies make the pipeline compile but emit no geometry, which is
// useful for validation tests.
//
// Pipeline contract:
//   · No vertex buffer. Geometry is procedural — VS reads
//     `@builtin(vertex_index)` (0..verts_per_spawn) and
//     `@builtin(instance_index)` (the global spawn slot).
//   · Per-spawn `InstanceRecord` resolves a spawn slot → (anchor_idx,
//     spawn_idx). `anchors[anchor_idx]` is the painted leaf this
//     spawn lives on; `spawn_idx` is the per-anchor index in
//     0..count[anchor_idx]. Dead spawns (filtered by spawn_alive)
//     carry anchor_idx = 0xFFFFFFFF and the VS emits a degenerate
//     vertex.
//   · One indirect draw per active user-shader material. Indirect
//     args are written by the prefix_sum pass.
//   · Picking writes USER_SHADER_PICK_SENTINEL unconditionally
//     (load-bearing — user-shader geometry is intentionally not
//     pickable).
//   · Output attachments match mesh_proxy.wesl exactly so downstream
//     PBR / shadow / GI passes are agnostic to source path.
//
// Bindings:
//   @group(0) @binding(0)  camera                  (uniform)
//   @group(1) @binding(0)  anchors                 (storage, read)
//   @group(1) @binding(1)  instance_records        (storage, read)
//   @group(1) @binding(2)  frame                   (uniform)
//   @group(1) @binding(3)  params                  (uniform)
//
// Composer splice surface — see notes/user-shaders-mesh.md §
// "Authoring surface" for the full contract.

import package::lib::types::CameraUniforms;

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// ── Engine-authored prelude (composer does not touch) ─────────────

// Sentinel value written into the pick G-buffer for every fragment
// that came from a user-shader VS. Picking lookups for this value
// fall through to the host entity underneath. Load-bearing per
// `feedback_no_user_shader_picking`.
const USER_SHADER_PICK_SENTINEL: u32 = 0xFFFFFFFEu;

// Dead-instance sentinel written into instance_records[slot] by the
// fill compute pass when spawn_alive returned false. The VS short-
// circuits with a NaN clip position so the rasterizer culls the
// triangle.
const ANCHOR_IDX_DEAD: u32 = 0xFFFFFFFFu;

// One painted *tile* — the world-space AABB of every painted leaf
// inside this tile (tight bounds, not the tile cube) plus enough
// shading context to drive `vs`. Layout must match
// `rkp_render::user_shader_mesh_pass::AnchorRecord` (CPU side).
// 64 B; std430 packing.
// Layout mirrors `rkp_render::user_shader_mesh_pass::AnchorRecord`.
// The trailing `_pad` makes the WGSL struct size match Rust's 64 B
// — without it, naga rounds the natural size (44 B → 48 B) and the
// `array<AnchorContext>` stride diverges from the CPU upload.
struct AnchorContext {
    tile_min:        vec3<f32>,
    material_id:     u32,
    tile_max:        vec3<f32>,
    leaf_count:      u32,
    object_id:       u32,
    surface_y:       f32,
    seed:            u32,
    _pad:            array<u32, 5>,
}

// Per-frame engine uniforms. User shaders subscribe to these via
// the @animated manifest directive (V1 fills all fields unconditionally).
struct FrameContext {
    time:            f32,
    delta_time:      f32,
    wind_dir:        vec3<f32>,
    wind_strength:   f32,
    camera_pos:      vec3<f32>,
    _pad:            f32,
}

// One per allocated spawn — output of the fill compute pass, input
// to the VS via `@builtin(instance_index)`.
struct InstanceRecord {
    anchor_idx:      u32,
    spawn_idx:       u32,
}

// User-param uniform. V1 caps at 8 f32 params per shader. Composer
// emits this struct with the user's @param names; the stub body
// below just references positional access.
struct UserParams {
    p: array<vec4<f32>, 2>,  // 8 × f32 packed as 2 × vec4
}

@group(1) @binding(0) var<storage, read> anchors:          array<AnchorContext>;
@group(1) @binding(1) var<storage, read> instance_records: array<InstanceRecord>;
@group(1) @binding(2) var<uniform>       frame:            FrameContext;
@group(1) @binding(3) var<uniform>       params:           UserParams;

// VS output — must stay compatible with the FS / G-buffer pack below.
// Composer-driven user fns receive an AnchorContext + spawn_idx +
// vid + frame and fill this struct.
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos:    vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) material_packed: u32,
    @location(3) color_rgb:    vec3<f32>,
    @location(4) blend_f:      f32,
    @location(5) @interpolate(flat) intensity: u32,
}

struct FsOut {
    @location(0) position: vec4<f32>,
    @location(1) pick:     u32,
    @location(2) normal:   vec4<f32>,
    @location(3) material: vec2<u32>,
    @location(4) glass:    vec2<u32>,
}

// Helper accessors so user code reads params by stable indices
// regardless of how the composer reorders them.
fn ctx_param(i: u32) -> f32 {
    let v = params.p[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// USER_BODY anchors — `splice_const_marker` replaces everything
// between these two top-level const decls (inclusive) with the
// composed user body. The decls themselves survive WESL compilation
// because `build.rs` runs with `use_stripping(false)`; at runtime
// the splicer removes them so no other code may reference these
// identifiers.
const USER_BODY_BEGIN: u32 = 0u;

// The composer's splice keeps:
//   · the user's helper fns (any free-form fn decls);
//   · `fn spawn_count(anchor, frame) -> u32` (required, compute-only
//     copy lives in user_shader_mesh_compute.wesl);
//   · `fn vs(anchor, spawn_idx, vid, frame) -> VsOut` (required);
//   · `fn spawn_alive(anchor, spawn_idx, frame) -> bool` (optional;
//     compute-only);
//   · `fn fs(in: VsOut) -> FsOut` (optional; default body below).
//
// Stub bodies below: vs returns a clipped vertex (z < 0 in homogeneous
// clip). The stub shader compiles and binds but emits nothing —
// useful for pipeline-construction tests and as a "no-op fallback"
// while the composer wires real user shaders.

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    // z = -1 with w = 1 places the vertex behind the near plane —
    // reliably clipped without using NaN (which naga rejects as a
    // constant-folded literal).
    out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    out.world_pos = anchor.tile_min;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}

// Default fs body — user can override. Repackages VsOut interpolants
// into the standard G-buffer format. Picks USER_SHADER_PICK_SENTINEL
// unconditionally.
fn fs(in: VsOut) -> FsOut {
    let n_world = normalize(in.world_normal);

    let primary   = in.material_packed & 0xFFFFu;
    let secondary = (in.material_packed >> 16u) & 0xFFFu;

    let blend_clamped = clamp(in.blend_f, 0.0, 1.0);
    let blend4 = u32(blend_clamped * 15.0 + 0.5);
    let blend8 = (blend4 << 4u) | blend4;

    let cr8 = u32(clamp(in.color_rgb.r, 0.0, 1.0) * 255.0 + 0.5);
    let cg8 = u32(clamp(in.color_rgb.g, 0.0, 1.0) * 255.0 + 0.5);
    let cb8 = u32(clamp(in.color_rgb.b, 0.0, 1.0) * 255.0 + 0.5);
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);

    let packed_r = primary | (secondary << 16u);
    let packed_g =
        (blend8 & 0xFFu)
        | ((in.intensity & 0xFFu) << 8u)
        | (color_rgb565 << 16u);

    let hit_distance = length(in.world_pos - camera.position.xyz);

    var out: FsOut;
    out.position = vec4<f32>(in.world_pos, hit_distance);
    out.pick     = USER_SHADER_PICK_SENTINEL;
    out.normal   = vec4<f32>(n_world, 1.0);
    out.material = vec2<u32>(packed_r, packed_g);
    // V1: no glass output for user-shader materials. Glass-material
    // user shaders render opaque-uniform; full glass routing is a
    // follow-up (see notes/user-shaders-mesh.md §"After V1").
    out.glass    = vec2<u32>(0u, 0u);
    return out;
}

const USER_BODY_END: u32 = 0u;

// ── Engine entry points (composer does not touch) ─────────────────

@vertex
fn entry_vert(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let rec = instance_records[iid];
    if (rec.anchor_idx == ANCHOR_IDX_DEAD) {
        var dead: VsOut;
        dead.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        dead.world_pos = vec3<f32>(0.0);
        dead.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        dead.material_packed = 0u;
        dead.color_rgb = vec3<f32>(0.0);
        dead.blend_f = 0.0;
        dead.intensity = 0u;
        return dead;
    }
    let anchor = anchors[rec.anchor_idx];
    return vs(anchor, rec.spawn_idx, vid, frame);
}

@fragment
fn entry_frag(in: VsOut) -> FsOut {
    return fs(in);
}
