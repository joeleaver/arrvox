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

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
// Phase B-redux 3b — band-cell sentinel. Shadow scatter (the
// shadow-map setup pass) treats band cells as not-occluding; Phase 4
// will revisit if band shadow casting is added.
const OCTREE_BAND_BIT: u32 = 0x20000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x1FFFFFFFu;
const BRICK_DIM: u32 = 4u;
const BRICK_DIM_F: f32 = 4.0;
const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;
const BRICK_MAX_STEPS: u32 = 4096u;

const FACE_INTERIOR: u32 = 0xFFFFFFFEu;
const FACE_EMPTY_LINK: u32 = 0xFFFFFFFFu;
const FACE_NX: u32 = 0u;
const FACE_PX: u32 = 1u;
const FACE_NY: u32 = 2u;
const FACE_PY: u32 = 3u;
const FACE_NZ: u32 = 4u;
const FACE_PZ: u32 = 5u;

const SHADOW_MAP_MAX_STEPS: u32 = 256u;
const SKINNED_MAX_STEPS: u32 = 512u;
const NO_HIT_T: f32 = 1.0e20;

const TLAS_LEAF_USER_SHADER: u32 = 0xFFFFFFFEu;
const DISPATCH_X: u32 = 256u;

struct RkpInstance {
    world: mat4x4<f32>,
    asset_id: u32,
    material_id: u32,
    object_id: u32,
    layer_mask: u32,
    is_skinned: u32,
    bone_buffer_offset: u32,
    bone_field_offset: u32,
    bone_field_occ_offset: u32,
    bone_field_dim_x: u32, bone_field_dim_y: u32,
    bone_field_dim_z: u32,
    bone_field_origin_x: f32,
    bone_field_origin_y: f32,
    bone_field_origin_z: f32,
    overlay_offset: u32,
    overlay_count: u32,
    instance_state_offset: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}

struct RkpAsset {
    aabb_min: vec3<f32>, octree_root: u32,
    aabb_max: vec3<f32>, octree_depth: u32,
    octree_extent_bits: u32, voxel_size: f32,
    geom_type: u32, bone_count: u32,
    grid_origin: vec3<f32>, rest_octree_root: u32,
    rest_octree_depth: u32, rest_octree_extent_bits: u32,
    shader_id: u32, _pad: u32,
}

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    layer_mask: u32, focus_object_id: u32,
    _cam_pad0: u32, _cam_pad1: u32,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

struct OverlayEntry {
    leaf_slot: u32,
    normal_oct: u32,
    material_packed: u32,
    color_packed: u32,
}

struct LightCameraShadow {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    light_dir: vec3<f32>,
    depth_bias: f32,
    inv_shadow_map_size: vec2<f32>,
    shadow_map_size: vec2<u32>,
}

struct ScatterInstance {
    tx0: u32, ty0: u32,
    tile_w: u32, tile_h: u32,
    asset_id: u32,
    instance_state_offset: u32,
    instance_index: u32,
    work_offset: u32,
}

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
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
@group(0) @binding(14) var<storage, read> instance_pool: array<u32>;

// Group 1 — pass-private resources.
@group(1) @binding(0) var<uniform> light_camera: LightCameraShadow;
@group(1) @binding(1) var<storage, read_write> shadow_buffer: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read> scatter_instances: array<ScatterInstance>;
@group(1) @binding(3) var<storage, read> work_list: array<u32>;
// dispatch_args[3] holds total_work (set by finalize). Used to
// bounds-check workgroups dispatched past the live work count.
@group(1) @binding(4) var<storage, read> dispatch_args: array<u32>;

// ── Helpers (mirror of the host march; see shadow_scatter docs
//   for context on the find_hit_in_instance contract). ─────────

fn mat4_affine_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let t = m[3].xyz;
    let inv_det = 1.0 / dot(a, cross(b, c));
    let row0 = cross(b, c) * inv_det;
    let row1 = cross(c, a) * inv_det;
    let row2 = cross(a, b) * inv_det;
    let new_t = -vec3<f32>(dot(row0, t), dot(row1, t), dot(row2, t));
    return mat4x4<f32>(
        vec4<f32>(row0.x, row1.x, row2.x, 0.0),
        vec4<f32>(row0.y, row1.y, row2.y, 0.0),
        vec4<f32>(row0.z, row1.z, row2.z, 0.0),
        vec4<f32>(new_t, 1.0),
    );
}

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(max(tmin.x, tmin.y), tmin.z), 0.0),
                     min(min(tmax.x, tmax.y), tmax.z));
}

fn safe_dir3(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(d.x, select(-1e-10, 1e-10, d.x >= 0.0), abs(d.x) < 1e-10),
        select(d.y, select(-1e-10, 1e-10, d.y >= 0.0), abs(d.y) < 1e-10),
        select(d.z, select(-1e-10, 1e-10, d.z >= 0.0), abs(d.z) < 1e-10),
    );
}

struct OctreeResult {
    slot: u32,
    depth: u32,
    cell_center: vec3<f32>,
    cell_half: f32,
}

fn octree_lookup_no_stats(root: u32, max_depth: u32, extent: f32, pos: vec3<f32>) -> OctreeResult {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset].x;
        if node == OCTREE_EMPTY {
            return OctreeResult(OCTREE_EMPTY, level, center, half);
        }
        if node == OCTREE_INTERIOR {
            return OctreeResult(OCTREE_INTERIOR, level, center, half);
        }
        if (node & OCTREE_LEAF_BIT) != 0u {
            // Phase B-redux 3b — band cells skipped (don't cast
            // shadow). Phase 4 will wire descent here.
            if (node & OCTREE_BAND_BIT) != 0u {
                return OctreeResult(OCTREE_EMPTY, level, center, half);
            }
            return OctreeResult(node & OCTREE_PAYLOAD_MASK | (node & OCTREE_BRICK_BIT), level, center, half);
        }
        let gt = vec3<u32>(pos >= center);
        offset = node + gt.x + gt.y * 2u + gt.z * 4u;
        half *= 0.5;
        center += vec3<f32>(
            select(-half, half, pos.x >= center.x),
            select(-half, half, pos.y >= center.y),
            select(-half, half, pos.z >= center.z),
        );
    }
    let node = octree_nodes[offset].x;
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth, center, half); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth, center, half); }
    if (node & OCTREE_LEAF_BIT) != 0u {
        if (node & OCTREE_BAND_BIT) != 0u {
            return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
        }
        return OctreeResult(node & OCTREE_PAYLOAD_MASK | (node & OCTREE_BRICK_BIT), max_depth, center, half);
    }
    return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
}

fn slot_is_brick(slot: u32) -> bool {
    return (slot & OCTREE_BRICK_BIT) != 0u
        && slot != OCTREE_EMPTY
        && slot != OCTREE_INTERIOR;
}

fn slot_brick_id(slot: u32) -> u32 {
    return slot & OCTREE_PAYLOAD_MASK;
}

fn skip_node_t(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    let t_pos = max(t_exit, vec3<f32>(1e-6));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * 0.01;
}


fn inst_world_to_local(
    world_pos: vec3<f32>, instance_pos: vec3<f32>, instance_scale: f32,
) -> vec3<f32> {
    let inv_s = 1.0 / max(instance_scale, 1e-10);
    return (world_pos - instance_pos) * inv_s + vec3<f32>(0.5);
}

// USER_INST_TO_LOCAL_DISPATCH_BEGIN
fn dispatch_user_inst_to_local(
    shader_id: u32,
    base_u32: u32,
    world_pos: vec3<f32>,
    fallback_pos: vec3<f32>,
    fallback_scale: f32,
) -> vec3<f32> {
    return inst_world_to_local(world_pos, fallback_pos, fallback_scale);
}
// USER_INST_TO_LOCAL_DISPATCH_END

// USER_INST_AABB_DISPATCH_BEGIN
fn dispatch_user_inst_aabb(
    shader_id: u32,
    base_u32: u32,
    fallback_pos: vec3<f32>,
    fallback_scale: f32,
) -> Aabb {
    let half = fallback_scale * 0.5 * 1.7320508;
    var a: Aabb;
    a.min = fallback_pos - vec3<f32>(half);
    a.max = fallback_pos + vec3<f32>(half);
    return a;
}
// USER_INST_AABB_DISPATCH_END

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

    var local_origin: vec3<f32>;
    var local_dir_unnorm: vec3<f32>;
    if asset.shader_id != 0u {
        let inst_pos = inst.world[3].xyz;
        let inst_scale = length(inst.world[0].xyz);
        let aabb = dispatch_user_inst_aabb(
            asset.shader_id, inst.instance_state_offset,
            inst_pos, inst_scale,
        );
        let inv_world_dir = 1.0 / safe_dir3(world_dir);
        let aabb_t = intersect_aabb(world_origin, inv_world_dir, aabb.min, aabb.max);
        if aabb_t.x > aabb_t.y { return NO_HIT_T; }
        let world_t_entry = max(aabb_t.x, 0.0);
        let world_entry = world_origin + world_dir * world_t_entry;
        let local_entry = dispatch_user_inst_to_local(
            asset.shader_id, inst.instance_state_offset,
            world_entry, inst_pos, inst_scale,
        );
        let local_endpoint = dispatch_user_inst_to_local(
            asset.shader_id, inst.instance_state_offset,
            world_entry + world_dir, inst_pos, inst_scale,
        );
        local_origin = local_entry;
        local_dir_unnorm = local_endpoint - local_entry;
    } else {
        let inv_world = mat4_affine_inverse(inst.world);
        local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
        local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    }
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
        let r = octree_lookup_no_stats(root, max_depth, extent, pos);

        if r.slot == OCTREE_EMPTY {
            t += max(skip_node_t(pos, safe, inv_dir, r.depth, extent, vs), min_step);
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
    inst.instance_state_offset = s.instance_state_offset;
    inst._pad0 = 0u;
    inst._pad1 = 0u;
    inst._pad2 = 0u;
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
