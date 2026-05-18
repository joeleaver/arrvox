// Procedural CSG raymarcher — live preview path for procedural objects
// in the build viewport. One compute dispatch per frame, one thread per
// pixel, sphere-traces the flattened RPN tree against the camera ray
// and writes the same G-buffer the mesh raster + `mesh_resolve` fill
// so downstream shadow / SSAO / shade / post passes see identical
// inputs.
//
// Shared with the (Phase 2+) GPU bake compute: all math — primitive
// SDFs, combinators, effects, noise, RPN interpretation — lives in
// `proc_eval.wgsl`. This shader owns only the bindings and the
// sphere-trace loop.
//
// Imports the RPN interpreter from `proc_eval` and types/opcodes
// from `proc_eval_types`. The shared `eval_tree` reads the
// `instructions` binding declared below.

import package::lib::types::ProcInstruction;
import package::proc_eval_types::TreeSample;
import package::proc_eval::eval_tree;
import package::proc_eval::eval_tree_distance;
import package::proc_eval::gradient_normal;
import package::lib::types::CameraUniforms;

// Pipeline-override constant — kept unmangled so the Rust side
// (`proc_raymarch.rs`) can override it at pipeline creation by name.
// proc_eval's body references this as a free name resolved at emit.
override HAS_POS_WARPS: bool = true;

// March parameters. Tight by design — the build viewport is an
// editor preview, not a final renderer, and the user's procedural
// objects sit in the 0.1 m – 10 m range. `MAX_DIST` is in entity-
// local units; it scales with the entity's transform so a 10× larger
// object still fits. `MAX_STEPS` = 64 converges on every 1-Lipschitz
// primitive we ship and even leaves headroom for chains of combinators.
const MAX_STEPS:    u32 = 64u;
const MAX_DIST:     f32 = 30.0;
const SURFACE_EPS:  f32 = 0.001;

// ── Bindings ───────────────────────────────────────────────────────────

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
    // Owning entity's world transform and its inverse. The shader
    // marches in entity-local space so the preview stays pinned
    // wherever the entity is in the world.
    entity_world: mat4x4<f32>,
    entity_inverse_world: mat4x4<f32>,
    // Tree bounding box in entity-local space. Used to reject rays
    // that can't possibly hit before we pay the sphere-trace loop,
    // and to fast-forward hit-pixel rays to the AABB entry point.
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg32uint, write>;
// rkp-side pick texture — R32Uint. Holds the hit primitive's NodeId
// for per-primitive selection / outline.
@group(1) @binding(3) var gbuf_pick:     texture_storage_2d<r32uint, write>;
// Glass info — proc_raymarch has no glass support; write zero here
// so the shared buffer stays coherent with the rest of the G-buffer.
// Layout: R = oct-packed normal, G = (thickness_mm << 16) | material_id.
@group(1) @binding(4) var gbuf_glass:    texture_storage_2d<rg32uint, write>;
// Leaf-slot target — procedurals write 0 since they have no stable
// leaf_attr_slot; the shade pass treats 0 as "no geodesic data for
// this pixel" and skips the paint cursor there.
@group(1) @binding(5) var gbuf_leaf_slot: texture_storage_2d<r32uint, write>;

@group(2) @binding(0) var<uniform> params: RaymarchParams;
@group(2) @binding(1) var<storage, read> instructions: array<ProcInstruction>;

// ── Ray construction ───────────────────────────────────────────────────

struct Ray { origin: vec3<f32>, dir: vec3<f32> };

fn make_ray(coord: vec2<i32>) -> Ray {
    // The camera's `right` and `up` basis vectors are pre-scaled by
    // aspect and tan(fov/2), so the ray is just
    // `forward + ndc.x * right + ndc.y * up`.
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

    // Transform the camera ray into the entity's local frame so we can
    // march the procedural tree (authored in local space) as if the
    // entity were at origin.
    let local_origin_h = params.entity_inverse_world * vec4<f32>(ray.origin, 1.0);
    let local_dir_h = params.entity_inverse_world * vec4<f32>(ray.dir, 0.0);
    let local_origin = local_origin_h.xyz;
    let local_dir_unnorm = local_dir_h.xyz;
    let local_dir = normalize(local_dir_unnorm);
    let local_scale = length(local_dir_unnorm);
    let local_max_dist = MAX_DIST * local_scale;

    // ── AABB cull ─────────────────────────────────────────────────────
    // Ray/AABB slab test in entity-local space. Rays that miss the
    // tree's bounding box skip the sphere trace entirely (the vast
    // majority of pixels on a small object in a big viewport). For
    // rays that hit, `t_enter` fast-forwards the trace to the AABB
    // entry so we don't waste steps marching through empty space
    // outside the bounds.
    //
    // Division-by-zero on axis-aligned rays gives `inf` which the
    // min/max chain collapses to ignore that axis — standard slab
    // trick, no explicit epsilon needed.
    let inv_dir = vec3<f32>(1.0, 1.0, 1.0) / local_dir;
    let t0 = (params.aabb_min.xyz - local_origin) * inv_dir;
    let t1 = (params.aabb_max.xyz - local_origin) * inv_dir;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let t_enter = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit  = min(min(t_max.x, t_max.y), t_max.z);

    // Miss: the ray never enters the AABB (or exits behind the
    // camera). Write the miss G-buffer and bail — the sphere trace
    // would return the same answer at a much higher cost.
    if (t_exit < 0.0 || t_enter > t_exit || t_enter > local_max_dist) {
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_pick,     coord, vec4<u32>(0xFFFFu, 0u, 0u, 0u));
        textureStore(gbuf_glass,    coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_leaf_slot, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let count = params.instruction_count;

    // Sphere trace runs the distance-only fast path — a 64-byte
    // `array<f32, STACK_CAP>` fits in registers where the 512-byte
    // `array<TreeSample, STACK_CAP>` would spill to global memory.
    // Full `eval_tree` runs once below at the hit point so the
    // G-buffer sample still carries material + color + node_id.
    //
    // Start at the AABB entry (or origin if camera is inside the
    // box). The cap at `t_exit` lets trace-ending run short when
    // the ray leaves the box without hitting the surface.
    var t: f32 = max(t_enter, 0.0);
    let t_cap = min(t_exit, local_max_dist);
    var hit: bool = false;
    var hit_local: vec3<f32> = vec3<f32>(0.0);
    for (var step: u32 = 0u; step < MAX_STEPS; step = step + 1u) {
        let p = local_origin + local_dir * t;
        let d = eval_tree_distance(p, count);
        if (d < SURFACE_EPS) {
            hit = true;
            hit_local = p;
            break;
        }
        // Sphere trace: step by the signed distance (max safe step
        // for a 1-Lipschitz SDF). Clamp to `SURFACE_EPS` on negative
        // distances so we don't walk backwards after over-march.
        t = t + max(d, SURFACE_EPS);
        if (t > t_cap) { break; }
    }

    if (!hit) {
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        // 0xFFFF = "no primitive" sentinel that outline + pick treat
        // as a miss.
        textureStore(gbuf_pick,     coord, vec4<u32>(0xFFFFu, 0u, 0u, 0u));
        textureStore(gbuf_glass,    coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_leaf_slot, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    // Back to world space for the G-buffer. Downstream shade / shadow
    // / post passes all read world-space positions. For uniform or
    // rigid transforms the naive matrix-mul for the normal is correct;
    // with non-uniform scale the strictly-right thing is inverse-
    // transpose, but the build-viewport preview is an editing tool so
    // we live with the mild lighting error until it actually bites.
    // One full-sample call at the hit to pull material + color +
    // node_id for the G-buffer. Gradient uses the distance-only path.
    let hit_sample = eval_tree(hit_local, count);
    let hit_pos = (params.entity_world * vec4<f32>(hit_local, 1.0)).xyz;
    let local_normal = gradient_normal(hit_local, count, SURFACE_EPS * 4.0);
    let normal = normalize((params.entity_world * vec4<f32>(local_normal, 0.0)).xyz);

    // Pack material/color into the same G-buffer format
    // `mesh_resolve.wesl` writes. Dual-material:
    //
    //   packed_r = primary (low 16) | secondary (high 16)
    //   packed_g = blend_u8 | object_id_u8 << 8 | color_rgb565 << 16
    //
    // `rkp_shade` lerps PBR properties between primary/secondary using
    // blend_weight, so by-height / by-noise transition zones render
    // with smooth seams both here and in the voxel path.
    //
    // The hit primitive's NodeId lives in `gbuf_pick` instead of the
    // old packed_r high-16 slot. `0xFFFF` is the "no primitive"
    // sentinel consumed by proc_outline + the engine's pick readback.
    let primary = hit_sample.material_id & 0xFFFFu;
    let secondary = hit_sample.secondary_material_id & 0xFFFFu;
    let blend_u8 = u32(clamp(hit_sample.blend_weight, 0.0, 1.0) * 255.0);
    let primitive_node_id = hit_sample.node_id & 0xFFFFu;
    let cr = u32(clamp(hit_sample.color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(hit_sample.color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(hit_sample.color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    let packed_r = primary | (secondary << 16u);
    let packed_g = (blend_u8 & 0xFFu)
                 | ((params.object_id & 0xFFu) << 8u)
                 | (color_rgb565 << 16u);

    // `t` is in local-space units; the G-buffer's depth slot expects a
    // world-space distance along the ray so downstream passes (fog,
    // SSAO, etc.) stay comparable with voxel hits. Dividing by
    // `local_scale` undoes the inverse_world's scale contribution.
    let world_t = select(t / local_scale, t, local_scale < 1e-8);
    textureStore(gbuf_position, coord, vec4<f32>(hit_pos, world_t));
    textureStore(gbuf_normal,   coord, vec4<f32>(normal, 1.0));
    textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
    textureStore(gbuf_pick,     coord, vec4<u32>(primitive_node_id, 0u, 0u, 0u));
    textureStore(gbuf_glass,    coord, vec4<u32>(0u, 0u, 0u, 0u));
    textureStore(gbuf_leaf_slot, coord, vec4<u32>(0u, 0u, 0u, 0u));
}
