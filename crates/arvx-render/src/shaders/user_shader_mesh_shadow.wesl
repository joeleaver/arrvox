// V1 mesh-path user-shader shadow VS — depth-only render that writes
// blade silhouettes into the directional cascade depth maps. Mirrors
// the structure of `user_shader_mesh.wesl` (same composer USER_BODY
// splice, same group(1) layout, same per-spawn record consumption)
// but:
//   · adds `light_camera` + `shadow_params` at group(0) bindings 1+2
//     so the engine entry can project through the cascade VP;
//   · drops the FS entirely — depth-only render keeps the GPU's
//     early-z at full strength;
//   · ignores the user vs's `clip_pos` — only `world_pos` is consumed
//     (re-projected through the cascade), so user shaders that haven't
//     been authored with shadow casting in mind work unchanged.
//
// Bindings:
//   @group(0) @binding(0)  camera          (uniform — shared with the
//                                            primary raster so the user's
//                                            vs body, which references
//                                            `camera.view_proj`, compiles
//                                            even though we don't read
//                                            its output here)
//   @group(0) @binding(1)  light_camera    (uniform — LightCameraCsm)
//   @group(0) @binding(2)  shadow_params   (uniform — cascade_index)
//   @group(1) @binding(0)  anchors         (storage, read)
//   @group(1) @binding(1)  instance_records (storage, read)
//   @group(1) @binding(2)  frame           (uniform)
//   @group(1) @binding(3)  params          (uniform)

import package::lib::types::CameraUniforms;
import package::lib::types::LightCameraCsm;

@group(0) @binding(0) var<uniform> camera:        CameraUniforms;
@group(0) @binding(1) var<uniform> light_camera:  LightCameraCsm;
@group(0) @binding(2) var<uniform> shadow_params: ShadowParams;

struct ShadowParams {
    cascade_index: u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}

const USER_SHADER_PICK_SENTINEL: u32 = 0xFFFFFFFEu;
const ANCHOR_IDX_DEAD: u32 = 0xFFFFFFFFu;

// AnchorContext / FrameContext / InstanceRecord / UserParams / VsOut
// / FsOut — same struct decls as the primary raster template
// (user_shader_mesh.wesl). Kept verbatim so the user's vs body
// splices identically. FsOut is unused here (no FS) but declared so
// a user that defined `fn fs(...)` still parses cleanly under naga
// before DCE drops it.

struct AnchorContext {
    tile_min:        vec3<f32>,
    material_id:     u32,
    tile_max:        vec3<f32>,
    leaf_count:      u32,
    paint_min:       vec3<f32>,
    object_id:       u32,
    paint_max:       vec3<f32>,
    surface_y:       f32,
    surface_normal:  vec3<f32>,
    seed:            u32,
    paint_mask:      u32,
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
}

struct FrameContext {
    time:            f32,
    delta_time:      f32,
    wind_dir:        vec3<f32>,
    wind_strength:   f32,
    camera_pos:      vec3<f32>,
    _pad:            f32,
}

struct InstanceRecord {
    anchor_idx:      u32,
    spawn_idx:       u32,
}

struct UserParams {
    p: array<vec4<f32>, 2>,
}

@group(1) @binding(0) var<storage, read> anchors:          array<AnchorContext>;
@group(1) @binding(1) var<storage, read> instance_records: array<InstanceRecord>;
@group(1) @binding(2) var<uniform>       frame:            FrameContext;
@group(1) @binding(3) var<uniform>       params:           UserParams;

// VsOut + FsOut copied from user_shader_mesh.wesl so the spliced user
// body compiles. Only `world_pos` is consumed downstream; the rest
// are written-but-discarded.
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

fn ctx_param(i: u32) -> f32 {
    let v = params.p[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// USER_BODY anchors — same splice contract as the primary raster
// template. Composer drops the user's vs + helpers + structs (and
// optional fs, which we don't consume here) between these markers.
const USER_BODY_BEGIN: u32 = 0u;

fn vs(anchor: AnchorContext, spawn_idx: u32, vid: u32, frame: FrameContext) -> VsOut {
    var out: VsOut;
    out.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    out.world_pos = anchor.tile_min;
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_packed = anchor.material_id;
    out.color_rgb = vec3<f32>(1.0);
    out.blend_f = 0.0;
    out.intensity = 0u;
    return out;
}

fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    out.position = vec4<f32>(0.0);
    out.pick     = USER_SHADER_PICK_SENTINEL;
    out.normal   = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    out.material = vec2<u32>(0u, 0u);
    out.glass    = vec2<u32>(0u, 0u);
    return out;
}

const USER_BODY_END: u32 = 0u;

// Engine entry — depth-only. Output is just `clip_pos`; the
// rasterizer fills the depth attachment, no FS runs.
struct ShadowVsOut {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn entry_vert(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> ShadowVsOut {
    let rec = instance_records[iid];
    if (rec.anchor_idx == ANCHOR_IDX_DEAD) {
        var dead: ShadowVsOut;
        dead.clip_pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return dead;
    }
    let anchor = anchors[rec.anchor_idx];
    // Run the user's vs to get world_pos. The user's clip_pos is
    // discarded — we re-project through the cascade VP.
    let user_out = vs(anchor, rec.spawn_idx, vid, frame);
    let cam = light_camera.cascades[shadow_params.cascade_index];
    var out: ShadowVsOut;
    out.clip_pos = cam.view_proj * vec4<f32>(user_out.world_pos, 1.0);
    return out;
}
