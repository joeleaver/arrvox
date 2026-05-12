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

import package::lib::types::LightCameraCsm;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::RkpInstance;
import package::lib::types::OverlayEntry;
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

/// Per-render-pass uniform selecting which cascade to render this
/// pass. Engine writes `cascade_index = i` and binds before each
/// `dispatch_mesh_shadow` cascade iteration. 16 B (vec4 alignment).
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

// `bone_matrices` is a 64 B `mat4x4<f32>` per bone (matches
// `skin_deform.wesl`). Declaring it as `mat3x4` would shave 16 B per
// bone but mismatches the on-GPU stride that the writer uses, which
// produced "stretched all over" geometry in an earlier draft.
@group(0) @binding(0) var<uniform> light_camera: LightCameraCsm;
@group(0) @binding(1) var<uniform> shadow_params: MeshShadowParams;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshShadowInstance;

// Glass-classify bindings — same shape as `mesh.wesl::group(2)`. The
// FS reads the cell's material and discards glass fragments so the
// opaque shadow map only contains opaque casters; glass casters are
// rendered separately by `mesh_glass_shadow.wesl` and the shade pass
// applies Beer attenuation on top of the opaque shadow factor.
//
// Adding this FS does forfeit early-z on instances that go through
// it; the per-instance `has_glass` filter sends all-opaque instances
// to a NoGlass pipeline that has no FS (preserving early-z), and
// only mixed / glass-only instances pay the FS cost.
@group(2) @binding(0) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read> instances: array<RkpInstance>;
@group(2) @binding(3) var<storage, read> instance_overlay: array<OverlayEntry>;
@group(2) @binding(4) var<storage, read> color_pool_data: array<u32>;
// Per-instance sculpt-carve overlay (Phase A). Same layout slot as
// `mesh.wesl`; shadow FS discards on hit so carved leaves stop
// casting shadows.
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

@fragment
fn frag_main(in: VsOut) {
    let inst = instances[instance.object_id];
    // Sculpt-carve discard — carved leaves cast no shadow.
    if is_leaf_removed(inst, in.leaf_attr_id) {
        discard;
    }
    let attr = fetch_leaf_attr_for(inst, in.leaf_attr_id);
    let mat_id = attr.material_packed & 0xFFFFu;
    // OOB-safe lookup — see `mesh.wesl` for the rationale. Without
    // this, virgin pool slots that point at material ids beyond the
    // loaded palette would discard as glass and miss casting shadow,
    // creating dark/light fringes that follow the brush boundary.
    let mat_id_safe = select(
        mat_id, 0u, mat_id >= arrayLength(&materials),
    );
    if materials[mat_id_safe].opacity < 0.99 {
        discard;
    }
}
