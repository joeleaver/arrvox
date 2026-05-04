// Phase 8 V2 — geometry-driven shadow rendering (work-list scatter).
//
// Single indirect dispatch over the global work list. Each work
// list entry corresponds to one 8×8 tile of one instance's
// projected light-space rect. Workgroups (8×8) walk the tile in
// parallel, descend the instance, atomic-min the depth.
//
// Per workgroup:
//   flat_work_idx = wid.y * DISPATCH_X + wid.x
//   if flat_work_idx >= total_work: return
//   entry = work_list[flat_work_idx]
//   inst_idx, tile_x_local, tile_y_local = unpack(entry)
//   inst = scatter_instances[inst_idx]
//   tx = inst.tx0 + tile_x_local * 8 + lid.x
//   ty = inst.ty0 + tile_y_local * 8 + lid.y
//   ... descend, atomicMin shadow_buffer[ty * W + tx]
//
// The dispatch dimensions (`DISPATCH_X` × ceil(total/DISPATCH_X) × 1)
// come from the finalize pass via `dispatch_workgroups_indirect`,
// so the CPU never reads back the work count.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::math::mat4_affine_inverse;
import package::lib::math::intersect_aabb;
import package::lib::brick::BRICK_DIM;
import package::lib::brick::BRICK_DIM_F;
import package::lib::brick::BRICK_CELLS;
import package::lib::brick::BRICK_CELL_EMPTY;
import package::lib::brick::BRICK_CELL_INTERIOR;
import package::lib::brick::BRICK_MAX_STEPS;
import package::lib::brick::FACE_INTERIOR;
import package::lib::brick::FACE_EMPTY_LINK;
import package::lib::brick::FACE_NX;
import package::lib::brick::FACE_PX;
import package::lib::brick::FACE_NY;
import package::lib::brick::FACE_PY;
import package::lib::brick::FACE_NZ;
import package::lib::brick::FACE_PZ;
import package::lib::types::unpack_band_cell;
import package::lib::octree::octree_lookup;
import package::lib::octree::descend_proto_octree;
import package::lib::octree::PHASE_SHADOW;
import package::lib::octree::skip_node;
import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_INTERIOR;
import package::lib::octree_slot::OCTREE_LEAF_BIT;
import package::lib::octree_slot::OCTREE_BRICK_BIT;
import package::lib::octree_slot::OCTREE_BAND_BIT;
import package::lib::octree_slot::OCTREE_PAYLOAD_MASK;
import package::lib::octree_slot::slot_is_brick;
import package::lib::octree_slot::slot_is_band;
import package::lib::octree_slot::slot_brick_id;
import package::lib::octree_slot::slot_band_offset;
import package::lib::types::OctreeResult;
import package::lib::types::LeafAttr;
import package::lib::types::BandCell;
import package::lib::types::RkpInstance;
import package::lib::types::GpuMaterial;
import package::lib::types::CameraUniforms;
import package::lib::types::OverlayEntry;
import package::lib::types::ScatterInstance;
import package::lib::types::ProtoHit;
import package::lib::types::LightCameraShadow;
import package::lib::types::Aabb;
import package::lib::types::UserCtx;
import package::lib::types::HostSample;
import package::lib::types::RkpAsset;
import package::lib::types::InstanceHit;


const SHADOW_MAP_MAX_STEPS: u32 = 256u;
const SKINNED_MAX_STEPS: u32 = 512u;
const NO_HIT_T: f32 = 1.0e20;

const TLAS_LEAF_USER_SHADER: u32 = 0xFFFFFFFEu;
const DISPATCH_X: u32 = 256u;



// Phase B-redux 3b — band-cell wire format. Mirrors
// octree_march.wgsl::BandCell.
// Phase 4 — user-shader-facing types for the band-cell shadow
// dispatch. Same shapes as octree_march.wgsl so the composer's
// `dispatch_user_instance_descend` body splices in unchanged.



// vec3 fields flattened to f32 — see rkp_shade.wgsl for the
// rationale (96 B vs 128 B padded).
// Subset of the primary `MarchParams` — only the fields the band-
// cell dispatch needs (time + asset_count). Layout is the leading
// portion of the full uniform; we declare a tight wrapper here to
// keep this pass independent of unused fields.
struct MarchParamsLite {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    lod_enabled: u32,
    surfacenet_enabled: u32,
    tile_count_x: u32,
    tlas_node_count: u32,
    shadow_map_enabled: u32,
    time: f32,
    asset_count: u32,
    _pad0: u32,
}


// Group 0 — scene resources.
@group(0) @binding(0) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> instances: array<RkpInstance>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;
@group(0) @binding(7) var<storage, read> brick_face_links: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(9) var<storage, read> bone_field: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read> bone_field_occ: array<u32>;
@group(0) @binding(12) var<storage, read> assets: array<RkpAsset>;
@group(0) @binding(13) var<storage, read> instance_overlay: array<OverlayEntry>;

// Group 1 — pass-private resources.
@group(1) @binding(0) var<uniform> light_camera: LightCameraShadow;
@group(1) @binding(1) var<storage, read_write> shadow_buffer: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> scatter_instances: array<ScatterInstance>;
@group(1) @binding(3) var<storage, read> work_list: array<u32>;
// dispatch_args[3] holds total_work (set by finalize). Used to
// bounds-check workgroups dispatched past the live work count.
@group(1) @binding(4) var<storage, read> dispatch_args: array<u32>;
// Phase 4 — band-cell shadow dispatch reads:
//  - materials: to look up the painted host material's `shader_id`
//    when a band cell is hit.
//  - shader_params: per-material 8 × f32 array shared with the
//    primary march, used to populate `ctx.params` for the user-
//    shader instance descent.
//  - march_params: only `time` and `asset_count` are read.
@group(1) @binding(5) var<storage, read> materials: array<GpuMaterial>;
@group(1) @binding(6) var<storage, read> shader_params: array<f32>;
@group(1) @binding(7) var<uniform> march_params: MarchParamsLite;

// ── Helpers (mirror of the host march; see shadow_scatter docs
//   for context on the find_hit_in_instance contract). ─────────

// `mat4_affine_inverse` is imported from `lib::math` at the top.
//
// `intersect_aabb` stays local — it gets referenced from the
// user-shader composer's spliced `instance_at` chunk, which uses
// the unmangled name. Lifting it to lib would break the splice
// (WESL renames imported symbols).

fn safe_dir3(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(d.x, select(-1e-10, 1e-10, d.x >= 0.0), abs(d.x) < 1e-10),
        select(d.y, select(-1e-10, 1e-10, d.y >= 0.0), abs(d.y) < 1e-10),
        select(d.z, select(-1e-10, 1e-10, d.z >= 0.0), abs(d.z) < 1e-10),
    );
}

// Stats no-op stubs. shadow_scatter doesn't bind a `stats` buffer;
// the unified `octree_lookup` calls these by free name, so they have
// to exist. naga DCE-eliminates the empty bodies. If we ever want
// scatter-phase profiling, replace these with real bucket writes
// (and add the stats binding to the bind group).
fn bucket_depth(phase: u32, level: u32) {}
fn bucket_lod_exit(level: u32) {}

// `slot_is_brick`, `slot_brick_id`, `slot_is_band`, `slot_band_offset`
// are imported from `lib::octree_slot` at the top of this file.

fn read_band_cell(band_off: u32) -> BandCell {
    return unpack_band_cell(leaf_attr_pool[band_off], leaf_attr_pool[band_off + 1u]);
}

// Stats phase for `descend_proto_octree`'s internal `octree_lookup`.
// scatter has no stats binding — its `bucket_depth` is a no-op stub —
// so the value is moot, but we keep the const for symmetry with the
// other consumer files.
const PROTO_PHASE: u32 = PHASE_SHADOW;


const USER_INSTANCE_AT_DISPATCH_BEGIN: u32 = 0u;
// Default identity stub — Rust composer replaces this whole block
// (markers + body) with per-shader descent helpers + the dispatch
// switch when any registered shader provides an `instance_at` hook.
// Empty-registry path returns "no instance hit" so the band-cell
// shadow branch falls through to the no-hit case.
fn dispatch_user_instance_descend(
    shader_id: u32,
    host_pos: vec3<f32>,
    host: HostSample,
    leaf_slot: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    world_max_t: f32,
    ctx: UserCtx,
    asset: RkpAsset,
) -> InstanceHit {
    var r: InstanceHit;
    r.valid = false;
    r.world_t = world_max_t;
    r.world_pos = vec3<f32>(0.0);
    r.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    return r;
}
const USER_INSTANCE_AT_DISPATCH_END: u32 = 0u;

// `skip_node` (formerly `skip_node_t` here) is imported from
// `lib::octree` at the top. The `_t` suffix was historical and
// served no purpose — same body as `octree_march::skip_node`.


const OCC_BRICK_DIM: i32 = 4;

fn skinned_brick_populated(
    cell: vec3<i32>, cell_dims: vec3<i32>, occ_offset: u32,
) -> bool {
    if any(cell < vec3<i32>(0)) || any(cell >= cell_dims) { return false; }
    let brick = cell / vec3<i32>(OCC_BRICK_DIM);
    let bx_dim = u32((cell_dims.x + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM);
    let by_dim = u32((cell_dims.y + OCC_BRICK_DIM - 1) / OCC_BRICK_DIM);
    let brick_idx = u32(brick.x)
        + u32(brick.y) * bx_dim
        + u32(brick.z) * bx_dim * by_dim;
    let word = bone_field_occ[occ_offset + (brick_idx >> 5u)];
    return (word & (1u << (brick_idx & 31u))) != 0u;
}

fn skinned_brick_exit_t(
    origin: vec3<f32>, inv_dir: vec3<f32>,
    cell: vec3<i32>, grid_origin: vec3<f32>, vs: f32,
) -> f32 {
    let brick = cell / vec3<i32>(OCC_BRICK_DIM);
    let brick_min = grid_origin + vec3<f32>(brick * OCC_BRICK_DIM) * vs;
    let brick_max = brick_min + vec3<f32>(f32(OCC_BRICK_DIM) * vs);
    let t0 = (brick_min - origin) * inv_dir;
    let t1 = (brick_max - origin) * inv_dir;
    let t_far = max(t0, t1);
    return min(t_far.x, min(t_far.y, t_far.z));
}

fn skinned_sample(
    cell: vec3<i32>, dims: vec3<i32>, offset: u32,
) -> u32 {
    if any(cell < vec3<i32>(0)) || any(cell >= dims) { return 0u; }
    let idx = u32(cell.x)
        + u32(cell.y) * u32(dims.x)
        + u32(cell.z) * u32(dims.x) * u32(dims.y);
    return bone_field[offset + idx].x;
}

fn find_hit_skinned(
    world_origin: vec3<f32>, world_dir: vec3<f32>,
    inst: RkpInstance, asset: RkpAsset,
) -> f32 {
    let inv_world = mat4_affine_inverse(inst.world);
    let local_origin_mesh = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    let local_scale = length(local_dir_unnorm);
    let vs = asset.voxel_size;

    let rest_extent = bitcast<f32>(asset.rest_octree_extent_bits);
    let local_origin = local_origin_mesh + vec3<f32>(rest_extent * 0.5);

    let grid_dim = vec3<i32>(
        i32(inst.bone_field_dim_x),
        i32(inst.bone_field_dim_y),
        i32(inst.bone_field_dim_z),
    );
    if grid_dim.x <= 0 || grid_dim.y <= 0 || grid_dim.z <= 0 {
        return NO_HIT_T;
    }
    let grid_origin = vec3<f32>(
        inst.bone_field_origin_x,
        inst.bone_field_origin_y,
        inst.bone_field_origin_z,
    );
    let grid_max = grid_origin + vec3<f32>(grid_dim) * vs;

    let inv_dir = 1.0 / safe_dir3(local_dir);
    let t_range = intersect_aabb(local_origin, inv_dir, grid_origin, grid_max);
    if t_range.x > t_range.y { return NO_HIT_T; }

    var t = max(t_range.x, 0.0) + vs * 0.001;
    let t_limit = t_range.y;

    for (var step = 0u; step < SKINNED_MAX_STEPS; step++) {
        if t > t_limit { break; }
        let p_local = local_origin + local_dir * t;
        let cell_f = (p_local - grid_origin) / vs;
        let cell_i = vec3<i32>(floor(cell_f));

        if !skinned_brick_populated(cell_i, grid_dim, inst.bone_field_occ_offset) {
            let t_exit = skinned_brick_exit_t(local_origin, inv_dir, cell_i, grid_origin, vs);
            t = max(t + vs * 0.01, t_exit + vs * 0.001);
            continue;
        }

        let leaf_slot = skinned_sample(cell_i, grid_dim, inst.bone_field_offset);
        if leaf_slot != 0u {
            return t / max(local_scale, 1e-10);
        }
        t += vs;
    }
    return NO_HIT_T;
}

fn find_hit_in_instance(
    inst: RkpInstance, asset: RkpAsset,
    world_origin: vec3<f32>, world_dir: vec3<f32>,
) -> f32 {
    if asset.geom_type == 0u { return NO_HIT_T; }

    if inst.is_skinned != 0u && inst.bone_field_dim_x > 0u {
        return find_hit_skinned(world_origin, world_dir, inst, asset);
    }

    let inv_world = mat4_affine_inverse(inst.world);
    let local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    let local_scale = length(local_dir_unnorm);
    if local_scale < 1e-10 { return NO_HIT_T; }

    let root = asset.octree_root;
    let max_depth = asset.octree_depth;
    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    let min_step = vs * 2.0;

    let oc_origin = local_origin - asset.grid_origin;
    let safe = safe_dir3(local_dir);
    let inv_dir = 1.0 / safe;

    let t_range = intersect_aabb(oc_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));
    if t_range.x > t_range.y { return NO_HIT_T; }

    var t = max(t_range.x, 0.0);
    let t_limit = t_range.y;
    let lookup_bias = vs * 1.0e-3;

    for (var step = 0u; step < SHADOW_MAP_MAX_STEPS; step++) {
        if t > t_limit { break; }

        let pos = clamp(oc_origin + safe * (t + lookup_bias), vec3<f32>(vs * 0.01), vec3<f32>(extent - vs * 0.01));
        let r = octree_lookup(root, max_depth, extent, pos, PHASE_SHADOW, 0u, 0.0, 1.0, 0.0);

        if r.slot == OCTREE_EMPTY {
            t += max(skip_node(pos, safe, inv_dir, r.depth, extent, vs), min_step);
            continue;
        }

        // Phase 4 — band-cell shadow dispatch. Mirror of the
        // primary march's band-cell branch (octree_march.wgsl). On
        // hit return the world-space `t` directly (the dispatcher
        // works in world space; the caller multiplies by world_dir
        // to recover the world hit). On miss skip past the cell.
        if slot_is_band(r.slot) {
            let band_off = slot_band_offset(r.slot);
            let band = read_band_cell(band_off);
            let band_mat_id = band.material_id;
            let mat = materials[band_mat_id];
            let shader_id = mat.instance_shader_id;
            var proto_idx: u32 = 0xFFFFFFFFu;
            if shader_id != 0u {
                let acount = march_params.asset_count;
                for (var ai: u32 = 0u; ai < acount; ai = ai + 1u) {
                    if assets[ai].shader_id == shader_id {
                        proto_idx = ai;
                        break;
                    }
                }
            }
            if proto_idx == 0xFFFFFFFFu {
                t += max(skip_node(pos, safe, inv_dir, r.depth, extent, vs), min_step);
                continue;
            }
            let proto_asset = assets[proto_idx];

            var host_sample: HostSample;
            host_sample.distance = 0.0;
            host_sample.normal = vec3<f32>(0.0, 1.0, 0.0);
            host_sample.material = band_mat_id;
            host_sample.material_secondary = 0u;
            host_sample.blend_weight = 0u;

            var ctx: UserCtx;
            ctx.time = march_params.time;
            ctx.cell_size = r.cell_half * 2.0;
            ctx.material_id = band_mat_id;
            ctx.aabb_min = vec3<f32>(0.0);
            let pbase = band_mat_id * 8u;
            ctx.params[0] = shader_params[pbase + 0u];
            ctx.params[1] = shader_params[pbase + 1u];
            ctx.params[2] = shader_params[pbase + 2u];
            ctx.params[3] = shader_params[pbase + 3u];
            ctx.params[4] = shader_params[pbase + 4u];
            ctx.params[5] = shader_params[pbase + 5u];
            ctx.params[6] = shader_params[pbase + 6u];
            ctx.params[7] = shader_params[pbase + 7u];

            // Dispatcher takes WORLD-space inputs; pass the
            // caller's `world_origin` / `world_dir` directly.
            let inst_hit = dispatch_user_instance_descend(
                shader_id,
                band.anchor_world_pos,
                host_sample,
                0u,
                world_origin,
                world_dir,
                1.0e30,
                ctx,
                proto_asset,
            );
            if inst_hit.valid {
                // The function contract returns world-space `t` —
                // dispatcher already returns world distance.
                return inst_hit.world_t;
            }
            t += max(skip_node(pos, safe, inv_dir, r.depth, extent, vs), min_step);
            continue;
        }

        if r.slot == OCTREE_INTERIOR {
            return t / local_scale;
        }

        if slot_is_brick(r.slot) {
            var brick_id = slot_brick_id(r.slot);
            let cell_size = (r.cell_half * 2.0) / BRICK_DIM_F;
            let inv_cell_size = 1.0 / cell_size;
            var brick_origin = r.cell_center - vec3<f32>(r.cell_half);
            var brick_base = brick_id * BRICK_CELLS;

            let p0 = oc_origin + safe * t;
            let local0 = (p0 - brick_origin) * inv_cell_size;
            var cell = clamp(
                vec3<i32>(floor(local0)),
                vec3<i32>(0),
                vec3<i32>(3),
            );
            let step_i = vec3<i32>(
                select(-1, 1, safe.x >= 0.0),
                select(-1, 1, safe.y >= 0.0),
                select(-1, 1, safe.z >= 0.0),
            );
            let step_gt = vec3<f32>(
                select(0.0, 1.0, safe.x >= 0.0),
                select(0.0, 1.0, safe.y >= 0.0),
                select(0.0, 1.0, safe.z >= 0.0),
            );
            let next_b = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
            var t_max = t + (next_b - p0) * inv_dir;
            let t_delta = abs(vec3<f32>(cell_size) * inv_dir);
            let dda_eps = cell_size * 1.0e-3;

            for (var bs = 0u; bs < BRICK_MAX_STEPS; bs++) {
                if t > t_limit { break; }

                if cell.x < 0 || cell.x >= 4
                    || cell.y < 0 || cell.y >= 4
                    || cell.z < 0 || cell.z >= 4 {
                    var face_idx: u32;
                    if cell.x < 0 { face_idx = FACE_NX; }
                    else if cell.x >= 4 { face_idx = FACE_PX; }
                    else if cell.y < 0 { face_idx = FACE_NY; }
                    else if cell.y >= 4 { face_idx = FACE_PY; }
                    else if cell.z < 0 { face_idx = FACE_NZ; }
                    else { face_idx = FACE_PZ; }
                    let link = brick_face_links[brick_id * 6u + face_idx];
                    if link == FACE_INTERIOR {
                        return t / local_scale;
                    }
                    if link == FACE_EMPTY_LINK {
                        break;
                    }
                    brick_id = link;
                    brick_base = link * BRICK_CELLS;
                    let brick_extent = BRICK_DIM_F * cell_size;
                    if face_idx == FACE_NX { cell.x = 3; brick_origin.x -= brick_extent; }
                    else if face_idx == FACE_PX { cell.x = 0; brick_origin.x += brick_extent; }
                    else if face_idx == FACE_NY { cell.y = 3; brick_origin.y -= brick_extent; }
                    else if face_idx == FACE_PY { cell.y = 0; brick_origin.y += brick_extent; }
                    else if face_idx == FACE_NZ { cell.z = 3; brick_origin.z -= brick_extent; }
                    else { cell.z = 0; brick_origin.z += brick_extent; }
                    let p_now = oc_origin + safe * t;
                    let next_b2 = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                    t_max = t + (next_b2 - p_now) * inv_dir;
                }

                let flat = u32(cell.x) + u32(cell.y) * BRICK_DIM + u32(cell.z) * BRICK_DIM * BRICK_DIM;
                let c = brick_pool[brick_base + flat];
                if c == BRICK_CELL_INTERIOR {
                    return t / local_scale;
                }
                if c != BRICK_CELL_EMPTY {
                    return t / local_scale;
                }

                if t_max.x < t_max.y && t_max.x < t_max.z {
                    t = t_max.x + dda_eps;
                    cell.x += step_i.x;
                    t_max.x += t_delta.x;
                } else if t_max.y < t_max.z {
                    t = t_max.y + dda_eps;
                    cell.y += step_i.y;
                    t_max.y += t_delta.y;
                } else {
                    t = t_max.z + dda_eps;
                    cell.z += step_i.z;
                    t_max.z += t_delta.z;
                }
            }
            continue;
        }

        return t / local_scale;
    }

    return NO_HIT_T;
}

fn synth_inst_from_scatter(s: ScatterInstance) -> RkpInstance {
    var inst: RkpInstance;
    inst.world = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    inst.asset_id = s.asset_id;
    inst.material_id = 0u;
    inst.object_id = TLAS_LEAF_USER_SHADER;
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
    return inst;
}

@compute @workgroup_size(8, 8, 1)
fn scatter_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let total_work = dispatch_args[3];
    let flat_work_idx = wid.y * DISPATCH_X + wid.x;
    if flat_work_idx >= total_work { return; }

    let entry = work_list[flat_work_idx];
    let inst_idx = entry & 0xFFFFu;
    let tile_x_local = (entry >> 16u) & 0xFFu;
    let tile_y_local = (entry >> 24u) & 0xFFu;

    let s = scatter_instances[inst_idx];

    let tx = s.tx0 + tile_x_local * 8u + lid.x;
    let ty = s.ty0 + tile_y_local * 8u + lid.y;
    if tx >= light_camera.shadow_map_size.x || ty >= light_camera.shadow_map_size.y {
        return;
    }

    let inv_size = light_camera.inv_shadow_map_size;
    let ndc = vec2<f32>(
        (f32(tx) + 0.5) * inv_size.x * 2.0 - 1.0,
        1.0 - (f32(ty) + 0.5) * inv_size.y * 2.0,
    );
    let near_clip = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    let near_world = light_camera.view_proj_inv * near_clip;
    let ray_origin = near_world.xyz / near_world.w;
    let ray_dir = light_camera.light_dir;

    var inst: RkpInstance;
    var asset: RkpAsset;
    if s.instance_index != TLAS_LEAF_USER_SHADER {
        inst = instances[s.instance_index];
        asset = assets[inst.asset_id];
    } else {
        inst = synth_inst_from_scatter(s);
        asset = assets[s.asset_id];
    }

    let t = find_hit_in_instance(inst, asset, ray_origin, ray_dir);
    if t >= NO_HIT_T * 0.5 { return; }

    let world_hit = ray_origin + ray_dir * t;
    let clip = light_camera.view_proj * vec4<f32>(world_hit, 1.0);
    let depth = clamp(clip.z / clip.w, 0.0, 0.999999);
    let depth_bits = bitcast<u32>(depth);

    let buffer_idx = ty * light_camera.shadow_map_size.x + tx;
    atomicMin(&shadow_buffer[buffer_idx], depth_bits);
}
