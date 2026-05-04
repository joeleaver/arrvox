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
import package::lib::sdf::sdf_sphere;
import package::lib::sdf::sdf_box;
import package::lib::sdf::sdf_capsule;
import package::lib::sdf::sdf_cylinder;
import package::lib::sdf::sdf_torus;
import package::lib::sdf::sdf_plane;
import package::lib::sdf::sdf_ramp;
import package::lib::types::CameraUniforms;
import package::lib::types::ProcInstruction;

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

/// Must match the byte layout of `rkp_procedural::flatten::ProcInstruction`.

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

// Primitive SDFs are imported from `lib::sdf` at the top of this file.

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
