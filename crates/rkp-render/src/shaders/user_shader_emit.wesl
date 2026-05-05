// User-shader instance emit pass.
//
// One thread per painted leaf. Per leaf, runs the registered
// shader's `instance_at` hook for k = 0..MAX_EMITS_PER_LEAF and writes
// one `RkpInstance` per accepted instance into the scene-global
// `user_shader_instance_buffer`, atomic-bumping
// `user_shader_instance_count_buffer` for slot allocation.
//
// The composer's `emit` chunk replaces the empty-stub
// `dispatch_user_emit` switch below with one case per registered
// shader that has an `instance_at` hook. Each case decodes the
// shader's per-instance struct, runs `rkp_user_<id>_instance_at` to
// derive the k-th instance, and runs `rkp_user_<id>_inst_world_matrix`
// to build the forward affine matrix written into `RkpInstance.world`.
//
// The host march descends each emitted blade through its standard
// `inv_world × ray` flow — the same code path used for every other
// scene object. No band cells, no per-pixel descend body, no
// 4-tap Jacobian.

import package::lib::types::RkpInstance;
import package::lib::types::HostSample;
import package::lib::types::UserCtx;
import package::lib::types::Aabb;
import package::lib::oct_normal::unpack_oct_normal;

// Per-painted-leaf record. CPU packs one of these per leaf cell whose
// material has a shader-bearing `instance_at` hook. Layout mirrors
// `crate::user_shader_emit_pass::EmitLeaf` (Rust-side struct).
struct EmitLeaf {
    world_pos: vec3<f32>,
    material_id: u32,
    normal_oct: u32,
    object_id: u32,
    leaf_slot: u32,
    cell_size: f32,
}

// Per-material → (shader_id, proto_asset_id) lookup. Indexed by
// `EmitLeaf.material_id`. Materials with no instance shader carry
// `shader_id = 0` and the per-leaf thread early-returns.
struct MatToProto {
    shader_id: u32,
    proto_asset_id: u32,
}

struct EmitParams {
    leaf_count: u32,
    instance_capacity: u32,
    time: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> leaves: array<EmitLeaf>;
@group(0) @binding(1) var<storage, read_write> instances: array<RkpInstance>;
@group(0) @binding(2) var<storage, read_write> instance_count: atomic<u32>;
@group(0) @binding(3) var<storage, read> shader_params: array<f32>;
@group(0) @binding(4) var<storage, read> mat_to_proto: array<MatToProto>;
@group(0) @binding(5) var<uniform> emit_params: EmitParams;

// Hard cap on the per-leaf k-loop. Per-shader `@max_emits_per_thread`
// directives may declare lower caps via early-return from
// `instance_at`; this value bounds the loop unrolling so the kernel
// stays compact and avoids pathological per-leaf cost from a buggy
// `instance_at` that always returns true.
const MAX_EMITS_PER_LEAF: u32 = 8u;

const USER_EMIT_DISPATCH_BEGIN: u32 = 0u;
// Default empty-registry stub. Returns `false` (no instance) for every
// (shader_id, k) pair so the emit dispatch produces no instances when
// no shader is registered.
fn dispatch_user_emit(
    shader_id: u32,
    host_pos: vec3<f32>,
    host: HostSample,
    ctx: UserCtx,
    k: u32,
    out_world_matrix: ptr<function, mat4x4<f32>>,
) -> bool {
    return false;
}
const USER_EMIT_DISPATCH_END: u32 = 0u;

@compute @workgroup_size(64, 1, 1)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x;
    if leaf_idx >= emit_params.leaf_count { return; }
    let leaf = leaves[leaf_idx];

    let mtp = mat_to_proto[leaf.material_id];
    let shader_id = mtp.shader_id;
    let proto_asset_id = mtp.proto_asset_id;
    if shader_id == 0u { return; }

    var host: HostSample;
    host.distance = 0.0;
    host.normal = unpack_oct_normal(leaf.normal_oct);
    host.material = leaf.material_id;
    host.material_secondary = 0u;
    host.blend_weight = 0u;

    var ctx: UserCtx;
    ctx.time = emit_params.time;
    ctx.cell_size = leaf.cell_size;
    ctx.material_id = leaf.material_id;
    ctx.aabb_min = vec3<f32>(0.0);
    let pbase = leaf.material_id * 8u;
    ctx.params[0] = shader_params[pbase + 0u];
    ctx.params[1] = shader_params[pbase + 1u];
    ctx.params[2] = shader_params[pbase + 2u];
    ctx.params[3] = shader_params[pbase + 3u];
    ctx.params[4] = shader_params[pbase + 4u];
    ctx.params[5] = shader_params[pbase + 5u];
    ctx.params[6] = shader_params[pbase + 6u];
    ctx.params[7] = shader_params[pbase + 7u];

    for (var k: u32 = 0u; k < MAX_EMITS_PER_LEAF; k = k + 1u) {
        var world_matrix: mat4x4<f32>;
        if !dispatch_user_emit(shader_id, leaf.world_pos, host, ctx, k, &world_matrix) {
            break;
        }
        let slot = atomicAdd(&instance_count, 1u);
        if slot >= emit_params.instance_capacity {
            // Capacity exhausted; un-bump and bail. Subsequent k's
            // would also fail; one wasted increment per overflow.
            atomicSub(&instance_count, 1u);
            return;
        }
        // Build the RkpInstance record. Field order + offsets mirror
        // `RkpGpuInstance` exactly; the host march reads this through
        // its standard `instances` binding.
        var inst: RkpInstance;
        inst.world = world_matrix;
        inst.asset_id = proto_asset_id;
        inst.material_id = leaf.material_id;
        inst.object_id = leaf.object_id;
        inst.layer_mask = 0xFFFFFFFFu;
        inst.is_skinned = 0u;
        inst.bone_buffer_offset = 0u;
        inst.bone_field_offset = 0u;
        inst.bone_field_occ_offset = 0u;
        inst.bone_field_dim_x = 0u;
        inst.bone_field_dim_y = 0u;
        inst.bone_field_dim_z = 0u;
        inst.bone_field_origin_x = 0.0;
        inst.bone_field_origin_y = 0.0;
        inst.bone_field_origin_z = 0.0;
        inst.overlay_offset = 0u;
        inst.overlay_count = 0u;
        instances[slot] = inst;
    }
}
