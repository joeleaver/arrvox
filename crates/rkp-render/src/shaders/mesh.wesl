// Surface-mesh rasterizer — vertex + fragment entry points.
//
// Phase 2 of the splat-to-mesh pivot. Mirrors `splat.wesl`'s
// visibility-buffer contract: the raster writes (position, pick,
// leaf_slot); the existing `splat_resolve` compute pass fills in the
// rest of the G-buffer (normal / material / glass) per-pixel via the
// `leaf_attr_pool` indirection.
//
// Pipeline contract:
//   · Vertex buffer: one `MeshVertex` per surface-net SN-cube vertex.
//     · @location(0) local_pos:     vec3<f32>   (grid corner, OBJECT-LOCAL)
//     · @location(1) normal_oct:    u32         (oct-packed; unused here,
//                                                 resolve reads from pool)
//     · @location(2) leaf_attr_id:  u32         (absolute leaf_attr_pool slot)
//     · @location(3) bone_indices:  u32         (4×u8, Phase 6.6)
//     · @location(4) bone_weights:  u32         (4×u8, Phase 6.6)
//   · Index buffer: u32, two triangles per exposed cell face, CCW-wound
//     around the outward normal.
//   · One indexed draw per scene-instance per asset; the per-instance
//     world matrix + bone offsets come from `instance` at @group(1)
//     @binding(0).
//
// Bind-group shape is identical to `splat.wesl`'s so the same
// `splat_g0` / `splat_instance_*` bind groups can be reused for the
// mesh pipeline. `bone_matrices` / `bone_dual_quats` were added in
// Phase 6.6 (mesh-VS skinning) — the splat path declares them in the
// shared layout but doesn't read them.

import package::lib::types::CameraUniforms;

// Per-instance uniform — shape-compatible with `splat.wesl`'s
// `SplatInstance` (80 B, mat4x4 + 4×u32). Skinning fields drive the
// VS branch — see `SKINNING_MODE_NONE` below for the rest-pose
// sentinel.
struct MeshInstance {
    world: mat4x4<f32>,
    object_id: u32,
    bone_offset_lbs: u32,
    bone_offset_dqs: u32,
    skinning_mode: u32,
}

// Sentinel matching `splat_pass::SKINNING_MODE_NONE`. Skips skinning
// regardless of per-vertex weights — used for rigid instances and for
// skinned assets without a live `Skeleton` (no bone matrices uploaded).
const SKINNING_MODE_NONE: u32 = 0xFFFFFFFFu;
const SKINNING_MODE_LBS: u32 = 0u;
const SKINNING_MODE_DQS: u32 = 1u;

// Dual-quaternion form for DQS — 32 B per bone. `r` is the rotation
// (real) part, `d` is the translation-encoded (dual) part. Matches
// `skin_deform.wesl`'s `DualQuat` so the same per-frame buffer feeds
// both compute and raster paths.
struct DualQuat {
    r: vec4<f32>,
    d: vec4<f32>,
}

// `bone_matrices` matches the buffer `BoneMatrixAllocator` writes —
// one full `mat4x4<f32>` (64 B) per bone. The trailing row is always
// `(0,0,0,1)` for affine bones; reading the bottom row as part of
// the matrix is harmless and keeps the WGSL layout in lockstep with
// `skin_deform.wesl`'s declaration. (An earlier draft of this shader
// declared `BoneMatrix { row0, row1, row2 }` at 48 B and produced
// "stretched all over" geometry — every other bone read landed mid-
// matrix because the actual on-GPU stride is 64.)
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) leaf_attr_id: u32,
}

// Linear blend of 4 bone influences in matrix space. Weights and
// indices are unpacked from the per-vertex u32 quads on the call
// site; offset shifts the bone index into this instance's palette.
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
    // Normalise — well-formed weights sum to 255 (= 1.0), but the
    // unskinned-vertex sentinel is `weights == 0`, which is caught
    // by the early-out in the VS before this is called. A guard
    // here keeps the math sane if a stray vertex carries weights
    // that don't sum to 1.0 due to quantisation.
    if total_w > 0.0 {
        return acc / total_w;
    }
    return p;
}

// Standard quaternion vector rotation: v + 2 * w * (q × v) + 2 * (q × (q × v)).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn dual_quat_transform_pos(dq: DualQuat, p: vec3<f32>) -> vec3<f32> {
    let rotated = quat_rotate(dq.r, p);
    let translation = 2.0 * (dq.r.w * dq.d.xyz - dq.d.w * dq.r.xyz + cross(dq.r.xyz, dq.d.xyz));
    return rotated + translation;
}

// Antipodality-correct DQS blend over 4 influences. Lifted from
// `skin_deform.wesl`'s `blend_dual_quat` — flipping each successive
// quat into the same hemisphere as the first prevents the rotation
// from cancelling at bones whose quaternions straddle the double
// cover.
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

// `normal_oct` (vertex location 1) is declared in the pipeline's
// vertex-buffer layout but isn't consumed here — the resolve compute
// reads `LeafAttr.normal_oct` from the pool slot for each hit pixel.
// wgpu tolerates unused locations; the buffer reader still strides
// past them.
@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
    @location(2) leaf_attr_id: u32,
    @location(3) bone_indices: u32,
    @location(4) bone_weights: u32,
) -> VsOut {
    // Skinning early-out: a per-vertex `bone_weights == 0` means the
    // vertex sits in an unskinned region of the asset (or the asset
    // is fully unskinned), and a per-instance `SKINNING_MODE_NONE`
    // means the entity has no live bone matrices regardless. Both
    // collapse to "render at rest pose" — the same path unskinned
    // assets always took.
    var skinned_pos = local_pos;
    if instance.skinning_mode != SKINNING_MODE_NONE && bone_weights != 0u {
        if instance.skinning_mode == SKINNING_MODE_LBS {
            skinned_pos = skin_lbs(local_pos, bone_indices, bone_weights, instance.bone_offset_lbs);
        } else if instance.skinning_mode == SKINNING_MODE_DQS {
            skinned_pos = skin_dqs(local_pos, bone_indices, bone_weights, instance.bone_offset_dqs);
        }
    }

    let world_pos = (instance.world * vec4<f32>(skinned_pos, 1.0)).xyz;
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

// Three-attachment fragment output — same visibility-buffer triplet
// `splat.wesl` writes:
//   position   (Rgba32Float, 16 B): world.xyz + hit_distance
//   pick       (R32Uint,      4 B): instance.object_id
//   leaf_slot  (R32Uint,      4 B): leaf_attr_id
struct FsOut {
    @location(0) position:  vec4<f32>,
    @location(1) pick:      u32,
    @location(2) leaf_slot: u32,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    let hit_distance = length(in.world_pos - camera.position.xyz);
    var out: FsOut;
    out.position  = vec4<f32>(in.world_pos, hit_distance);
    out.pick      = instance.object_id;
    out.leaf_slot = in.leaf_attr_id;
    return out;
}
