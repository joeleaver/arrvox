// Glass shadow render — captures glass front-face / back-face depth
// from each cascade's light POV so the shade pass can apply Beer
// attenuation through glass meshes.
//
// Two pipelines share this VS + FS:
//   · front: cull = Back,  depth_compare = Less    (closest entry)
//   · back:  cull = Front, depth_compare = Greater (farthest exit)
//
// The FS discards non-glass fragments (opacity ≥ 0.99) so the depth
// targets only carry glass surfaces. Opaque casters land in the
// existing `mesh_shadow_render` map; the shade pass multiplies the
// opaque shadow factor by `exp(-sigma * thickness)` where thickness =
// `back_depth - front_depth` projected to world units.
//
// VS is byte-for-byte identical to `mesh_shadow.wesl::vert_main` —
// duplicated rather than imported so each render pass stays
// self-contained. Bindings match `mesh_shadow.wesl` for `g0`/`g1`/
// `g2` so the same per-cascade `g0` and per-instance `g1` bind groups
// drive both pipelines.

import package::lib::types::LightCameraCsm;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::ArvxInstance;
import package::lib::types::OverlayEntry;
import package::lib::types::is_glass;
import package::lib::leaf_attr::fetch_leaf_attr_for;
import package::lib::leaf_attr::is_leaf_removed;

struct MeshShadowInstance {
    world: mat4x4<f32>,
    grid_origin: vec3<f32>,
    object_id: u32,
    bone_offset_lbs: u32,
    bone_offset_dqs: u32,
    skinning_mode: u32,
}

struct MeshShadowParams {
    cascade_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SKINNING_MODE_NONE: u32 = 0xFFFFFFFFu;
const SKINNING_MODE_LBS: u32 = 0u;
const SKINNING_MODE_DQS: u32 = 1u;

struct DualQuat {
    r: vec4<f32>,
    d: vec4<f32>,
}

@group(0) @binding(0) var<uniform> light_camera: LightCameraCsm;
@group(0) @binding(1) var<uniform> shadow_params: MeshShadowParams;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshShadowInstance;

@group(2) @binding(0) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read> instances: array<ArvxInstance>;
@group(2) @binding(3) var<storage, read> instance_overlay: array<OverlayEntry>;
@group(2) @binding(4) var<storage, read> color_pool_data: array<u32>;
// Per-instance sculpt-carve overlay (Phase A). Glass-shadow FS
// discards on hit so carved leaves stop contributing to per-cascade
// glass shadow depth.
@group(2) @binding(10) var<storage, read> instance_sculpt: array<u32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) leaf_attr_id: u32,
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
    @location(2) leaf_attr_id: u32,
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
    let cam = light_camera.cascades[shadow_params.cascade_index];
    let clip = cam.view_proj * vec4<f32>(world_pos, 1.0);
    var out: VsOut;
    out.clip_pos = clip;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

// Single FS shared by both `frag_front` and `frag_back` pipelines —
// they differ only in the Rust-side cull mode + depth compare. The
// classify is the inverse of `mesh.wesl` / `mesh_shadow.wesl`:
// discard *opaque* fragments so glass-only depth lands in the target.
@fragment
fn frag_main(in: VsOut) {
    let inst = instances[instance.object_id];
    // Sculpt-carve discard — carved leaves contribute no glass
    // shadow.
    if is_leaf_removed(inst, in.leaf_attr_id) { discard; }
    let attr = fetch_leaf_attr_for(inst, in.leaf_attr_id);
    let mat_id = attr.material_packed & 0xFFFFu;
    // OOB-safe lookup — same rationale as `mesh.wesl`. OOB reads
    // produce a zero `GpuMaterial.opacity` (below the glass
    // threshold), so without the clamp those fragments would
    // *survive* the glass-pass discard and pollute the glass shadow
    // depth. This pass keeps only glass, so it discards the opaque
    // (non-`is_glass`) complement.
    let mat_id_safe = select(
        mat_id, 0u, mat_id >= arrayLength(&materials),
    );
    if !is_glass(materials[mat_id_safe].opacity) {
        discard;
    }
}
