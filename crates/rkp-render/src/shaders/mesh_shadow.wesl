// Mesh-rendered shadow map — depth-only render.
//
// Phase 3 of the splat-to-mesh pivot. Vertex stage projects mesh
// triangles through the directional light's view-proj into a 1024²
// `Depth32Float` attachment. There is NO fragment shader — the
// rasterizer fills depth on its own. This lets the GPU run early-z
// culling at full speed; the prior atomic-min-from-fragment approach
// disabled early-z because the fragment had a side effect, costing
// ~3-5× the fragment work.
//
// A separate compute pass (`mesh_shadow_blit.wesl`) reads this depth
// texture and writes `bitcast<u32>(depth)` into the existing
// `shadow_buffer` storage that `shadow_map_pass` owns and shade
// already samples — so the shade-side sampling code is reused
// unchanged.
//
// **Phase 6.6:** the VS now does the same LBS / DQS skinning
// `mesh.wesl` does, so shadows follow skeletal animation. The math
// is duplicated (rather than imported from a shared lib) because the
// bone bindings live on this pipeline's `g0`, not the shared scene
// `g0` that `skin_deform.wesl` reads from. A shared `lib::skin`
// helper that takes pre-fetched bone palettes is a candidate
// follow-on, but inline keeps each render-pass self-contained.

import package::lib::types::LightCameraShadow;

struct MeshShadowInstance {
    world: mat4x4<f32>,
    grid_origin: vec3<f32>,
    object_id: u32,
    bone_offset_lbs: u32,
    bone_offset_dqs: u32,
    skinning_mode: u32,
}

const SKINNING_MODE_NONE: u32 = 0xFFFFFFFFu;
const SKINNING_MODE_LBS: u32 = 0u;
const SKINNING_MODE_DQS: u32 = 1u;

struct DualQuat {
    r: vec4<f32>,
    d: vec4<f32>,
}

// `bone_matrices` is a 64 B `mat4x4<f32>` per bone (matches
// `skin_deform.wesl`). Declaring it as `mat3x4` would shave 16 B per
// bone but mismatches the on-GPU stride that the writer uses, which
// produced "stretched all over" geometry in an earlier draft.
@group(0) @binding(0) var<uniform> light_camera: LightCameraShadow;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshShadowInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
}

fn skin_lbs(p: vec3<f32>, indices: u32, weights: u32, offset: u32) -> vec3<f32> {
    let inv255 = 1.0 / 255.0;
    let p4 = vec4<f32>(p, 1.0);
    var acc = vec3<f32>(0.0);
    var total_w = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bi = (indices >> (i * 8u)) & 0xFFu;
        let wu = (weights >> (i * 8u)) & 0xFFu;
        if wu == 0u { continue; }
        let w = f32(wu) * inv255;
        let m = bone_matrices[offset + bi];
        acc = acc + w * (m * p4).xyz;
        total_w = total_w + w;
    }
    if total_w > 0.0 { return acc / total_w; }
    return p;
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn dual_quat_transform_pos(dq: DualQuat, p: vec3<f32>) -> vec3<f32> {
    let rotated = quat_rotate(dq.r, p);
    let translation = 2.0 * (dq.r.w * dq.d.xyz - dq.d.w * dq.r.xyz + cross(dq.r.xyz, dq.d.xyz));
    return rotated + translation;
}

fn skin_dqs(p: vec3<f32>, indices: u32, weights: u32, offset: u32) -> vec3<f32> {
    var acc_r = vec4<f32>(0.0);
    var acc_d = vec4<f32>(0.0);
    var total_w = 0.0;
    var base_r = vec4<f32>(0.0);
    var have_base = false;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bi = (indices >> (i * 8u)) & 0xFFu;
        let wu = (weights >> (i * 8u)) & 0xFFu;
        if wu == 0u { continue; }
        let w = f32(wu);
        let dq = bone_dual_quats[offset + bi];
        var r = dq.r;
        var d = dq.d;
        if !have_base {
            base_r = r;
            have_base = true;
        } else if dot(base_r, r) < 0.0 {
            r = -r;
            d = -d;
        }
        acc_r = acc_r + w * r;
        acc_d = acc_d + w * d;
        total_w = total_w + w;
    }
    if total_w <= 0.0 { return p; }
    let len_r = length(acc_r);
    if len_r < 1e-6 { return p; }
    let inv_len = 1.0 / len_r;
    return dual_quat_transform_pos(DualQuat(acc_r * inv_len, acc_d * inv_len), p);
}

@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
    @location(3) bone_indices: u32,
    @location(4) bone_weights: u32,
) -> VsOut {
    var skinned_pos = local_pos;
    if instance.skinning_mode != SKINNING_MODE_NONE && bone_weights != 0u {
        let grid_pos = local_pos - instance.grid_origin;
        var deformed: vec3<f32>;
        if instance.skinning_mode == SKINNING_MODE_LBS {
            deformed = skin_lbs(grid_pos, bone_indices, bone_weights, instance.bone_offset_lbs);
        } else {
            deformed = skin_dqs(grid_pos, bone_indices, bone_weights, instance.bone_offset_dqs);
        }
        skinned_pos = deformed + instance.grid_origin;
    }

    let world_pos = (instance.world * vec4<f32>(skinned_pos, 1.0)).xyz;
    let clip = light_camera.view_proj * vec4<f32>(world_pos, 1.0);
    var out: VsOut;
    out.clip_pos = clip;
    return out;
}
