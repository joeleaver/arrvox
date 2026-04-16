// Procedural CSG raymarcher — live preview path for procedural objects
// in the build viewport. One compute dispatch per frame, one thread per
// pixel, sphere-traces the flattened RPN tree against the camera ray
// and writes the same G-buffer layout as `octree_march.wgsl` so the
// downstream shadow / SSAO / shade / post passes see identical inputs.
//
// This is NOT a replacement for voxel rendering — it exists so that
// interactive tree edits (slider drags, transform moves) can update at
// 60 Hz without paying the ~700 ms voxelization cost per change. The
// baked voxels remain the truth everywhere else (main viewport, play
// mode, final shading quality).

// ── Opcodes ────────────────────────────────────────────────────────────
// Match `flatten::OpKind` — values are u32-compared inline below, keep
// in sync.
const OP_SPHERE:    u32 = 0u;
const OP_BOX:       u32 = 1u;
const OP_CAPSULE:   u32 = 2u;
const OP_CYLINDER:  u32 = 3u;
const OP_TORUS:     u32 = 4u;
const OP_PLANE:     u32 = 5u;
const OP_RAMP:      u32 = 6u;
const OP_UNION:     u32 = 100u;
const OP_INTERSECT: u32 = 101u;
const OP_SUBTRACT:  u32 = 102u;

const MAT_COMBINE_WINNER:  u32 = 0u;
// `Layered` is represented but the shader treats it as Winner for now —
// the dual-material G-buffer isn't wired through the raymarch path, so
// there's no place to land the secondary material even if we computed
// it here. See `combine_*`.
const MAT_COMBINE_LAYERED: u32 = 1u;
// `Blend { radius }` — smooth color interpolation inside a narrow band
// where both surfaces are equally close. Geometry distance is still the
// sharp min/max; we only lerp color between the two samples so the seam
// looks soft instead of a hard material edge. The radius rides along
// in each combinator instruction's `params_lo.x`.
const MAT_COMBINE_BLEND:   u32 = 2u;

// Max RPN stack depth. Tree depth is bounded by however many nested
// combinators the user builds; 16 accommodates pathological cases while
// keeping on-register use small. If the stream overflows we cap at the
// top (behavior degrades to "drop extra children") rather than wedging
// the shader.
const STACK_CAP: u32 = 16u;

// March parameters. Kept conservative to converge around typical
// procedural geometry (primitives are 1-Lipschitz so classical sphere
// tracing applies). Tuned by eye — feel free to adjust if you hit
// over-/under-march artifacts on new shape combinations.
const MAX_STEPS:    u32 = 128u;
const MAX_DIST:     f32 = 500.0;
const SURFACE_EPS:  f32 = 0.001;

// ── Bindings ───────────────────────────────────────────────────────────

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    layer_mask: u32, focus_object_id: u32,
    _cam_pad0: u32, _cam_pad1: u32,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

struct RaymarchParams {
    // How many `ProcInstruction` entries in `instructions` to execute.
    // Shorter than the buffer length when we've grown the buffer but
    // the current tree is smaller.
    instruction_count: u32,
    // 1-based object id for the owning entity. Packed into the material
    // G-channel's pick byte so the G-buffer hit looks like any other
    // object from the shader's POV.
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
}

// One flattened instruction. Layout MUST match
// `rkp_procedural::flatten::ProcInstruction` byte-for-byte.
struct ProcInstruction {
    op: u32,
    arity: u32,
    material_combine: u32,
    material_id: u32,
    // `node_id` is the source NodeId the primitive came from, used for
    // per-primitive picking; `u32::MAX` on combinators. The three
    // `_pad` fields keep the CPU struct 16-byte aligned for vec4 loads.
    node_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // vec4s chosen for WGSL alignment (8 floats → two vec4s).
    params_lo: vec4<f32>,
    params_hi: vec4<f32>,
    color: vec4<f32>,
    inverse_world: mat4x4<f32>,
}

// A sample at one point: distance + material payload. Mirrors the CPU
// `Sample` loosely — we drop fields (secondary/blend/color) that the
// current shade pass pipeline doesn't need from this preview path.
// Winner-mode CSG is enough for the preview; `Layered` falls through
// to Winner here.
struct TreeSample {
    distance: f32,
    material_id: u32,
    color: vec3<f32>,
    // Source primitive's NodeId — used so the G-buffer can tell the
    // pick path "which primitive was hit." Propagated through
    // combinators by picking the winner's id (same rule as material).
    node_id: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg32uint, write>;

@group(2) @binding(0) var<uniform> params: RaymarchParams;
@group(2) @binding(1) var<storage, read> instructions: array<ProcInstruction>;

// ── Primitive SDFs ─────────────────────────────────────────────────────
// Each mirrors its CPU counterpart in `rkp_procedural::leaves`; keep
// these in sync with that file if the Rust side ever changes.

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
    let closest = vec3<f32>(0.0, t, 0.0);
    return length(p - closest) - radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let radial = length(vec3<f32>(p.x, 0.0, p.z)) - radius;
    let axial = abs(p.y) - half_height;
    // Keep the CPU branching exactly (`leaves::eval_cylinder`): when the
    // point is outside both cylinder and caps, distance is the diagonal
    // length; otherwise it's `max` of the signed axes (standard SDF
    // boolean). Matching semantics avoids classifier disagreement.
    if (radial > 0.0 && axial > 0.0) {
        return sqrt(radial * radial + axial * axial);
    }
    return max(radial, axial);
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let xz_len = length(vec3<f32>(p.x, 0.0, p.z));
    let q = vec3<f32>(xz_len - major_radius, p.y, 0.0);
    return length(q) - minor_radius;
}

fn sdf_plane(p: vec3<f32>) -> f32 {
    return p.y;
}

fn sdf_ramp(p: vec3<f32>, half_length: f32, half_height: f32, half_width: f32) -> f32 {
    let q = abs(p) - vec3<f32>(half_length, half_height, half_width);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    let box_dist = outside + inside;
    let hyp = max(sqrt(half_length * half_length + half_height * half_height), 1e-6);
    let plane_dist = (half_length * p.y - half_height * p.x) / hyp;
    return max(box_dist, plane_dist);
}

fn eval_primitive(ins: ProcInstruction, world_pos: vec3<f32>) -> TreeSample {
    let local4 = ins.inverse_world * vec4<f32>(world_pos, 1.0);
    let local = local4.xyz;
    var d: f32 = 1e30;
    switch ins.op {
        case 0u: { d = sdf_sphere(local, ins.params_lo.x); }
        case 1u: { d = sdf_box(local, ins.params_lo.xyz, ins.params_lo.w); }
        case 2u: { d = sdf_capsule(local, ins.params_lo.x, ins.params_lo.y); }
        case 3u: { d = sdf_cylinder(local, ins.params_lo.x, ins.params_lo.y); }
        case 4u: { d = sdf_torus(local, ins.params_lo.x, ins.params_lo.y); }
        case 5u: { d = sdf_plane(local); }
        case 6u: { d = sdf_ramp(local, ins.params_lo.x, ins.params_lo.y, ins.params_lo.z); }
        default: { d = 1e30; }
    }
    var s: TreeSample;
    s.distance = d;
    s.material_id = ins.material_id;
    s.color = ins.color.xyz;
    s.node_id = ins.node_id;
    return s;
}

// ── Combinators ────────────────────────────────────────────────────────
// Winner-mode material selection: whichever sample has the smaller
// signed distance "wins" the material + color. `Layered` collapses to
// Winner for now (see MAT_COMBINE_LAYERED comment).

// In Blend mode, smooth the material/color transition across a band of
// width `radius` centered where the two samples have equal distance.
// Outside the band we fall back to Winner. Keeps the geometry's sharp
// min/max (matching the CPU path) but gives a visually soft seam.
fn blended_union_sample(a: TreeSample, b: TreeSample, radius: f32) -> TreeSample {
    let distance = min(a.distance, b.distance);
    let diff = abs(a.distance - b.distance);
    let r = max(radius, 1e-6);
    if (diff >= r) {
        if (a.distance <= b.distance) { return a; }
        return b;
    }
    // t=0 → fully b, t=1 → fully a (matches combine.rs's convention).
    let t = 0.5 + 0.5 * (b.distance - a.distance) / r;
    let winner_is_a = a.distance <= b.distance;
    var s: TreeSample;
    s.distance = distance;
    s.material_id = select(b.material_id, a.material_id, winner_is_a);
    s.color = mix(b.color, a.color, t);
    s.node_id = select(b.node_id, a.node_id, winner_is_a);
    return s;
}

fn combine_union(a: TreeSample, b: TreeSample, mat_mode: u32, radius: f32) -> TreeSample {
    if (mat_mode == MAT_COMBINE_BLEND) {
        return blended_union_sample(a, b, radius);
    }
    if (a.distance <= b.distance) { return a; }
    return b;
}

fn combine_intersect(a: TreeSample, b: TreeSample) -> TreeSample {
    // Max of distances; material from the loser (the one with the
    // larger, i.e. more-outside, distance) — that's the boundary that
    // defines the intersect surface. Matches `combine::combine_intersect`
    // winner semantics. Blend radius intentionally unused here —
    // intersect blends are rare and the visual seam is already soft
    // by virtue of being the max-of-two surface.
    if (a.distance >= b.distance) { return a; }
    return b;
}

fn combine_subtract(a: TreeSample, b: TreeSample) -> TreeSample {
    // Subtract: max(a, -b). Material (and picking source) always from
    // `a` — cutters don't contribute geometry you can click on.
    let neg_b = -b.distance;
    if (a.distance >= neg_b) {
        return a;
    }
    var r: TreeSample;
    r.distance = neg_b;
    r.material_id = a.material_id;
    r.color = a.color;
    r.node_id = a.node_id;
    return r;
}

// ── RPN execution ──────────────────────────────────────────────────────

fn eval_tree(world_pos: vec3<f32>) -> TreeSample {
    var stack: array<TreeSample, STACK_CAP>;
    var sp: u32 = 0u;

    let count = params.instruction_count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let ins = instructions[i];
        if (ins.op < 100u) {
            let s = eval_primitive(ins, world_pos);
            if (sp < STACK_CAP) {
                stack[sp] = s;
                sp = sp + 1u;
            }
        } else {
            // Combinator. Pop `arity`, combine, push one.
            let arity = ins.arity;
            if (arity == 0u || arity > sp) {
                // Malformed stream: treat as no-op. Keeps the shader
                // safe when the CPU emits a combinator with nothing to
                // consume (e.g. empty Subtract).
                continue;
            }
            let base = sp - arity;
            var acc = stack[base];
            let blend_radius = ins.params_lo.x;
            for (var k: u32 = 1u; k < arity; k = k + 1u) {
                let rhs = stack[base + k];
                switch ins.op {
                    case 100u: { acc = combine_union(acc, rhs, ins.material_combine, blend_radius); }
                    case 101u: { acc = combine_intersect(acc, rhs); }
                    case 102u: { acc = combine_subtract(acc, rhs); }
                    default: {}
                }
            }
            stack[base] = acc;
            sp = base + 1u;
        }
    }

    if (sp == 0u) {
        var miss: TreeSample;
        miss.distance = 1e30;
        miss.material_id = 0u;
        miss.color = vec3<f32>(0.0);
        miss.node_id = 0xFFFFFFFFu;
        return miss;
    }
    return stack[sp - 1u];
}

// ── Gradient normal ────────────────────────────────────────────────────
// Standard 6-tap central-difference gradient. Step size tuned against
// SURFACE_EPS so we're sampling a neighborhood wider than the hit
// epsilon — keeps the normal stable on slightly-over-marched hits.

fn gradient_normal(p: vec3<f32>) -> vec3<f32> {
    let h = SURFACE_EPS * 4.0;
    let dx = eval_tree(p + vec3<f32>(h, 0.0, 0.0)).distance
           - eval_tree(p - vec3<f32>(h, 0.0, 0.0)).distance;
    let dy = eval_tree(p + vec3<f32>(0.0, h, 0.0)).distance
           - eval_tree(p - vec3<f32>(0.0, h, 0.0)).distance;
    let dz = eval_tree(p + vec3<f32>(0.0, 0.0, h)).distance
           - eval_tree(p - vec3<f32>(0.0, 0.0, h)).distance;
    let g = vec3<f32>(dx, dy, dz);
    // Degenerate fallback (shouldn't fire in practice — every primitive
    // has a well-defined gradient everywhere on its surface).
    let len = length(g);
    if (len < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
    return g / len;
}

// ── Ray construction ───────────────────────────────────────────────────

struct Ray { origin: vec3<f32>, dir: vec3<f32> };

fn make_ray(coord: vec2<i32>) -> Ray {
    // Match `octree_march.wgsl:main`: the camera's `right` and `up`
    // basis vectors are already pre-scaled by aspect and tan(fov/2), so
    // the ray is just `forward + ndc.x * right + ndc.y * up`. We do not
    // touch view_proj here — the forward/right/up triplet is the
    // caller-side source of truth and keeps both marchers consistent.
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5) + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    var r: Ray;
    r.origin = camera.position.xyz;
    r.dir = normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz,
    );
    return r;
}

// ── Main ───────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(vec2<u32>(u32(camera.resolution.x), u32(camera.resolution.y)));
    let coord = vec2<i32>(gid.xy);
    if (coord.x >= res.x || coord.y >= res.y) {
        return;
    }

    let ray = make_ray(coord);

    var t: f32 = 0.0;
    var hit: bool = false;
    var hit_pos: vec3<f32> = vec3<f32>(0.0);
    var hit_sample: TreeSample;
    for (var step: u32 = 0u; step < MAX_STEPS; step = step + 1u) {
        let p = ray.origin + ray.dir * t;
        let s = eval_tree(p);
        if (s.distance < SURFACE_EPS) {
            hit = true;
            hit_pos = p;
            hit_sample = s;
            break;
        }
        // Sphere trace: step by the signed distance (which is, for a
        // 1-Lipschitz SDF, the max safe step). Negative distances can
        // happen after over-march; clamp to `SURFACE_EPS` so we don't
        // walk backwards.
        t = t + max(s.distance, SURFACE_EPS);
        if (t > MAX_DIST) { break; }
    }

    if (!hit) {
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let normal = gradient_normal(hit_pos);

    // Pack material/color into the same G-buffer format octree_march
    // uses — see that shader for the layout. Secondary material is
    // repurposed in this shader to carry the hit primitive's NodeId
    // (capped at 16 bits, same field width secondary_material normally
    // occupies): shade doesn't read the secondary slot, and the pick
    // readback wants a stable place to land. Combinators keep
    // `u32::MAX`, which fits in 16 bits as `0xFFFFu` — the pick path
    // treats that sentinel as "no primitive."
    //
    // Blend is always 0 here (dual-material output isn't wired through
    // the raymarch path); object_id byte is the owning entity's
    // scene_id so it still resolves via the same pick-readback table
    // on MAIN if we ever enable this on that viewport.
    let primary = hit_sample.material_id & 0xFFFFu;
    let primitive_node_id = hit_sample.node_id & 0xFFFFu;
    let blend = 0u;
    let cr = u32(clamp(hit_sample.color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(hit_sample.color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(hit_sample.color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    let packed_r = primary | (primitive_node_id << 16u);
    let packed_g = (blend & 0xFFu)
                 | ((params.object_id & 0xFFu) << 8u)
                 | (color_rgb565 << 16u);

    textureStore(gbuf_position, coord, vec4<f32>(hit_pos, t));
    textureStore(gbuf_normal,   coord, vec4<f32>(normal, 1.0));
    textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
}
