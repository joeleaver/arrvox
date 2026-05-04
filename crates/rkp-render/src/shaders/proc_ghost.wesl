// Ghost-cutter overlay for the procedural raymarch preview.
//
// Renders a subset of primitives as a translucent silhouette over the
// composite texture, independent of the main CSG tree. Used to show
// the user where a Subtract cutter (or Intersect operand) lives in
// space even when it's fully carved away by its parent combinator —
// no surface from the cutter exists in the main G-buffer, so a
// separate pass is the only way to visualize it.
//
// Runs after the outline pass in the composite chain, so ghosts sit
// on top of both scene and outline; no depth test — the ghost is
// visible through solid geometry by design (you want to see your
// cutter through the minuend when editing).
//
// Fragment pass, full-screen triangle. Primitives arrive as a
// `ProcInstruction` array in the same byte layout as the main
// raymarch's; we just ignore the combinator opcodes and assume a
// simple Union over the whole set (arity-based flattening is done
// CPU-side in `engine::collect_ghost_primitives`).

const OP_SPHERE:    u32 = 0u;
const OP_BOX:       u32 = 1u;
const OP_CAPSULE:   u32 = 2u;
const OP_CYLINDER:  u32 = 3u;
const OP_TORUS:     u32 = 4u;
const OP_PLANE:     u32 = 5u;
const OP_RAMP:      u32 = 6u;

const MAX_STEPS:    u32 = 96u;
const MAX_DIST:     f32 = 500.0;
const SURFACE_EPS:  f32 = 0.002;

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    layer_mask: u32, focus_object_id: u32,
    _cam_pad0: u32, _cam_pad1: u32,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

/// Must match the byte layout of `rkp_procedural::flatten::ProcInstruction`.
struct ProcInstruction {
    op: u32,
    arity: u32,
    material_combine: u32,
    material_id: u32,
    node_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    params_lo: vec4<f32>,
    params_hi: vec4<f32>,
    color: vec4<f32>,
    inverse_world: mat4x4<f32>,
}

struct GhostParams {
    /// How many entries in `instructions` to evaluate. Shorter than
    /// buffer capacity when the current ghost set is smaller than
    /// previous frames.
    instruction_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// Ghost overlay color, premultiplied-alpha expected by the
    /// pipeline's `src = One, dst = OneMinusSrcAlpha` blend.
    color_rgba: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> params: GhostParams;
@group(1) @binding(1) var<storage, read> instructions: array<ProcInstruction>;

// ── Primitive SDFs ─────────────────────────────────────────────────────
// Duplicates of `proc_raymarch.wgsl`. WGSL has no include mechanism; the
// alternative (CPU-side string concat at shader-module creation) buys
// little given the CPU-side SDF is also duplicated into `leaves.rs`.
// If we ever move to a shader preprocessor, factor these out.

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}
fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>, rounding: f32) -> f32 {
    let q = abs(p) - half_extents + vec3<f32>(rounding);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside - rounding;
}
fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let t = clamp(p.y, -half_height, half_height);
    return length(p - vec3<f32>(0.0, t, 0.0)) - radius;
}
fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let radial = length(vec3<f32>(p.x, 0.0, p.z)) - radius;
    let axial = abs(p.y) - half_height;
    if (radial > 0.0 && axial > 0.0) { return sqrt(radial * radial + axial * axial); }
    return max(radial, axial);
}
fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let xz_len = length(vec3<f32>(p.x, 0.0, p.z));
    let q = vec3<f32>(xz_len - major_radius, p.y, 0.0);
    return length(q) - minor_radius;
}
fn sdf_plane(p: vec3<f32>) -> f32 { return p.y; }
fn sdf_ramp(p: vec3<f32>, half_length: f32, half_height: f32, half_width: f32) -> f32 {
    let q = abs(p) - vec3<f32>(half_length, half_height, half_width);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    let box_dist = outside + inside;
    let hyp = max(sqrt(half_length * half_length + half_height * half_height), 1e-6);
    let plane_dist = (half_length * p.y - half_height * p.x) / hyp;
    return max(box_dist, plane_dist);
}

fn eval_primitive_dist(ins: ProcInstruction, world_pos: vec3<f32>) -> f32 {
    let local = (ins.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    switch ins.op {
        case 0u: { return sdf_sphere(local, ins.params_lo.x); }
        case 1u: { return sdf_box(local, ins.params_lo.xyz, ins.params_lo.w); }
        case 2u: { return sdf_capsule(local, ins.params_lo.x, ins.params_lo.y); }
        case 3u: { return sdf_cylinder(local, ins.params_lo.x, ins.params_lo.y); }
        case 4u: { return sdf_torus(local, ins.params_lo.x, ins.params_lo.y); }
        case 5u: { return sdf_plane(local); }
        case 6u: { return sdf_ramp(local, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z); }
        default: { return 1e30; }
    }
}

/// Union of every instruction's SDF. Combinators in the stream are
/// skipped (op >= 100); the caller (engine's ghost flattening)
/// pre-filters these out but the skip keeps this robust if a
/// combinator ever sneaks in.
fn ghost_sdf(world_pos: vec3<f32>) -> f32 {
    var min_d: f32 = 1e30;
    let count = params.instruction_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let ins = instructions[i];
        if (ins.op >= 100u) { continue; }
        let d = eval_primitive_dist(ins, world_pos);
        if (d < min_d) { min_d = d; }
    }
    return min_d;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(vi & 2u) * 2.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if (params.instruction_count == 0u) { discard; }

    // Reconstruct ray in the same basis the main raymarch uses — the
    // camera's `right` / `up` are pre-scaled by aspect + tan(fov/2),
    // so this matches what the user sees in the main preview pass
    // without us having to unproject through view_proj.
    let uv = (frag.xy + vec2<f32>(0.5) + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz,
    );

    var t: f32 = 0.0;
    var hit: bool = false;
    for (var step: u32 = 0u; step < MAX_STEPS; step = step + 1u) {
        let p = ray_origin + ray_dir * t;
        let d = ghost_sdf(p);
        if (d < SURFACE_EPS) { hit = true; break; }
        t = t + max(d, SURFACE_EPS);
        if (t > MAX_DIST) { break; }
    }

    if (!hit) { discard; }

    // Shader-side premultiply so the pipeline's pre-mul blend gives a
    // clean alpha-over composite. Ghost is intentionally translucent
    // so the real geometry beneath stays readable; the user's job is
    // to reconcile "where is my cutter" with "what's the boolean
    // result," and seeing both at partial opacity works better than
    // fully opaque.
    let a = params.color_rgba.a;
    return vec4<f32>(params.color_rgba.rgb * a, a);
}
