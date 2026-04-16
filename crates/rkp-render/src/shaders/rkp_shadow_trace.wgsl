// Half-resolution shadow trace.
//
// Dispatched at (width/2, height/2). Reads the full-res G-buffer at the
// corresponding upper-left pixel of each 2×2 block, traces shadow rays
// through the scene octree for each shadow-casting light, and writes the
// per-light transmittance (up to 4 lights) into a half-res rgba8unorm
// texture. The shade pass upsamples this with a position/normal-weighted
// bilateral filter.
//
// Shared helpers (types, bindings, octree_lookup, trace_shadow_ray) are
// duplicated from octree_march.wgsl for now — refactor into a common
// file once the half-res shadow path has stuck.

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OCTREE_BRICK_BIT: u32 = 0x40000000u;
const OCTREE_PAYLOAD_MASK: u32 = 0x3FFFFFFFu;
const MAX_OBJECTS: u32 = 32u;
const BRICK_DIM: u32 = 4u;
const BRICK_DIM_F: f32 = 4.0;
const BRICK_CELLS: u32 = 64u;
const BRICK_CELL_EMPTY: u32 = 0xFFFFFFFFu;
const BRICK_CELL_INTERIOR: u32 = 0xFFFFFFFDu;
// Raised from 16 to 128: the inner DDA now chains across adjacent
// bricks via brick_face_links, so a single inner loop can traverse
// many bricks before the outer loop needs to intervene.
const BRICK_MAX_STEPS: u32 = 4096u;

// Face-link sentinels — must match rkp_core::brick_face_links.
const FACE_INTERIOR: u32 = 0xFFFFFFFEu;
const FACE_EMPTY_LINK: u32 = 0xFFFFFFFFu;
const FACE_NX: u32 = 0u;
const FACE_PX: u32 = 1u;
const FACE_NY: u32 = 2u;
const FACE_PY: u32 = 3u;
const FACE_NZ: u32 = 4u;
const FACE_PZ: u32 = 5u;

struct RkpObject {
    world: mat4x4<f32>,
    aabb_min: vec3<f32>, octree_root: u32,
    aabb_max: vec3<f32>, octree_depth: u32,
    octree_extent_bits: u32, voxel_size: f32,
    material_id: u32, object_id: u32,
    geom_type: u32, is_skinned: u32,
    bone_count: u32, bone_buffer_offset: u32,
    rest_octree_root: u32, rest_octree_depth: u32,
    rest_octree_extent_bits: u32, deformed_pool_offset: u32,
    layer_mask: u32,
    _pre_grid0: u32, _pre_grid1: u32, _pre_grid2: u32,
    grid_origin: vec3<f32>,
    _post_grid: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
    inverse_world: mat4x4<f32>,
}

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    layer_mask: u32, focus_object_id: u32,
    _cam_pad0: u32, _cam_pad1: u32,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

struct MarchParams {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    // Must match octree_march.wgsl: same uniform buffer binding.
    lod_enabled: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct GpuLight {
    position: vec4<f32>,
    color: vec4<f32>,
    direction: vec4<f32>,
    params: vec4<f32>,
}

struct GpuMaterial {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emission_strength: f32,
    opacity: f32,
}

struct LeafAttr {
    normal_oct: u32,
    material_packed: u32,
}

struct OctreeResult {
    slot: u32,
    depth: u32,
    cell_center: vec3<f32>,
    cell_half: f32,
}

// Group 0: scene data (shared with march).
@group(0) @binding(0) var<storage, read> brick_pool: array<u32>;
// Interleaved (node_value, prefilter_attr_id) — see octree_march.wgsl.
@group(0) @binding(1) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> objects: array<RkpObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;
// brick_face_links[brick_id * 6 + face] → neighbor brick_id, or one of
// FACE_EMPTY_LINK / FACE_INTERIOR. See rkp_core::brick_face_links.
@group(0) @binding(7) var<storage, read> brick_face_links: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_attr_pool: array<LeafAttr>;

// Group 1: gbuf inputs (full-res, read) + half-res shadow output (write).
@group(1) @binding(0) var gbuf_position: texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(2) var shadow_lo_res: texture_storage_2d<rgba8unorm, write>;

// Group 2: march params + materials + stats + lights (shared with march).
@group(2) @binding(0) var<uniform> march_params: MarchParams;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 64>;
@group(2) @binding(3) var<storage, read> screen_aabbs: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read> lights: array<GpuLight>;

const PHASE_SHADOW: u32 = 2u;

fn leaf_attr_material_primary(a: LeafAttr) -> u32 { return a.material_packed & 0xFFFFu; }

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(max(tmin.x, tmin.y), tmin.z), 0.0),
                     min(min(tmax.x, tmax.y), tmax.z));
}

fn bucket_depth(phase: u32, level: u32) {
    let base = 4u + phase * 12u;
    atomicAdd(&stats[base + min(level, 11u)], 1u);
}

fn octree_lookup(root: u32, max_depth: u32, extent: f32, pos: vec3<f32>, phase: u32) -> OctreeResult {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset].x;
        if node == OCTREE_EMPTY {
            bucket_depth(phase, level);
            return OctreeResult(OCTREE_EMPTY, level, center, half);
        }
        if node == OCTREE_INTERIOR {
            bucket_depth(phase, level);
            return OctreeResult(OCTREE_INTERIOR, level, center, half);
        }
        if (node & OCTREE_LEAF_BIT) != 0u {
            bucket_depth(phase, level);
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
    bucket_depth(phase, max_depth);
    let node = octree_nodes[offset].x;
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth, center, half); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth, center, half); }
    if (node & OCTREE_LEAF_BIT) != 0u {
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

fn skip_node(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    let t_pos = max(t_exit, vec3<f32>(1e-6));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * 0.01;
}

// Shadow ray — returns transmittance in [0, 1]. 0 = fully occluded, 1 = lit.
fn trace_shadow_ray(
    world_origin: vec3<f32>,
    world_dir: vec3<f32>,
    num_objects: u32,
    max_steps: u32,
    max_world_dist: f32,
) -> f32 {
    var transmittance = 1.0;

    for (var oi = 0u; oi < num_objects && oi < MAX_OBJECTS; oi++) {
        let obj = objects[oi];
        if obj.geom_type == 0u { continue; }
        // Same gate as primary visibility (Phase 2). SHADOW_ONLY semantics
        // come later — they need a separate camera shadow_layer_mask.
        if (obj.layer_mask & camera.layer_mask) == 0u
            && obj.object_id != camera.focus_object_id { continue; }

        let inv_world = obj.inverse_world;
        let local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
        let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
        let local_dir = normalize(local_dir_unnorm);
        let local_scale = length(local_dir_unnorm);
        let local_max_t = max_world_dist * local_scale;

        let root = obj.octree_root;
        let max_depth = obj.octree_depth;
        let extent = bitcast<f32>(obj.octree_extent_bits);
        let vs = obj.voxel_size;
        let min_step = vs * 2.0;

        let oc_origin = local_origin - obj.grid_origin;
        let safe_dir = vec3<f32>(
            select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
            select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
            select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
        );
        let inv_dir = 1.0 / safe_dir;

        let shadow_origin = oc_origin + safe_dir * vs * 4.0;
        let t_range = intersect_aabb(shadow_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));
        if t_range.x > t_range.y { continue; }

        let t_limit = min(t_range.y, local_max_t);
        var t = max(t_range.x, 0.0);
        // Tiny forward-bias used only for octree_lookup / skip_node. At
        // brick-split boundaries `pos.x == center.x` is FP-ambiguous in
        // `pos >= center`; biasing forward disambiguates toward the cell
        // the ray is actually entering, eliminating the dashed-seam
        // pattern caused by rounding into an EMPTY sibling subtree.
        let lookup_bias = vs * 1.0e-3;

        for (var step = 0u; step < max_steps; step++) {
            if t > t_limit { break; }

            let pos = clamp(shadow_origin + safe_dir * (t + lookup_bias), vec3<f32>(vs * 0.01), vec3<f32>(extent - vs * 0.01));
            let r = octree_lookup(root, max_depth, extent, pos, PHASE_SHADOW);

            if r.slot == OCTREE_EMPTY {
                t += max(skip_node(pos, safe_dir, inv_dir, r.depth, extent, vs), min_step);
                continue;
            }

            if slot_is_brick(r.slot) {
                var brick_id = slot_brick_id(r.slot);
                let cell_size = (r.cell_half * 2.0) / BRICK_DIM_F;
                let inv_cell_size = 1.0 / cell_size;
                var brick_origin = r.cell_center - vec3<f32>(r.cell_half);
                var brick_base = brick_id * BRICK_CELLS;

                // Amanatides-Woo 3D DDA with brick_face_links chaining.
                // On exit through a face, consult the face-link table
                // instead of re-querying the octree — bypasses the
                // FP-ambiguity at brick boundaries that produces seams.
                let p0 = shadow_origin + safe_dir * t;
                let local0 = (p0 - brick_origin) * inv_cell_size;
                var cell = clamp(
                    vec3<i32>(floor(local0)),
                    vec3<i32>(0),
                    vec3<i32>(3),
                );
                let step_i = vec3<i32>(
                    select(-1, 1, safe_dir.x >= 0.0),
                    select(-1, 1, safe_dir.y >= 0.0),
                    select(-1, 1, safe_dir.z >= 0.0),
                );
                let step_gt = vec3<f32>(
                    select(0.0, 1.0, safe_dir.x >= 0.0),
                    select(0.0, 1.0, safe_dir.y >= 0.0),
                    select(0.0, 1.0, safe_dir.z >= 0.0),
                );
                let next_b = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                var t_max = t + (next_b - p0) * inv_dir;
                let t_delta = abs(vec3<f32>(cell_size) * inv_dir);
                // Nudge past cell boundaries for FP robustness when we
                // do fall through to the outer loop (FACE_EMPTY_LINK case).
                let dda_eps = cell_size * 1.0e-3;

                var blocked = false;
                for (var bs = 0u; bs < BRICK_MAX_STEPS; bs++) {
                    if t > t_limit { break; }

                    // Out-of-brick → follow face link to neighbor brick.
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
                            blocked = true;
                            break;
                        }
                        if link == FACE_EMPTY_LINK {
                            // Fall back to outer loop's skip_node.
                            break;
                        }
                        // Neighbor brick — swap brick state, re-enter at
                        // the opposite face's cell column. `t_max` and
                        // `t_delta` are mathematically ray-invariant, but
                        // FP error from incremental `t_max += t_delta`
                        // accumulates across long chains. Re-anchor
                        // `t_max` from the current ray position at every
                        // face-link crossing — see the matching comment
                        // in octree_march.wgsl for the full rationale.
                        brick_id = link;
                        brick_base = link * BRICK_CELLS;
                        let brick_extent = BRICK_DIM_F * cell_size;
                        if face_idx == FACE_NX { cell.x = 3; brick_origin.x -= brick_extent; }
                        else if face_idx == FACE_PX { cell.x = 0; brick_origin.x += brick_extent; }
                        else if face_idx == FACE_NY { cell.y = 3; brick_origin.y -= brick_extent; }
                        else if face_idx == FACE_PY { cell.y = 0; brick_origin.y += brick_extent; }
                        else if face_idx == FACE_NZ { cell.z = 3; brick_origin.z -= brick_extent; }
                        else { cell.z = 0; brick_origin.z += brick_extent; }
                        let p_now = shadow_origin + safe_dir * t;
                        let next_b = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                        t_max = t + (next_b - p_now) * inv_dir;
                    }

                    let flat = u32(cell.x) + u32(cell.y) * BRICK_DIM + u32(cell.z) * BRICK_DIM * BRICK_DIM;
                    let c = brick_pool[brick_base + flat];
                    if c != BRICK_CELL_EMPTY && c != BRICK_CELL_INTERIOR {
                        let attr = leaf_attr_pool[c];
                        let mid = leaf_attr_material_primary(attr);
                        let m_op = materials[mid].opacity;
                        if m_op >= 0.99 { blocked = true; break; }
                        transmittance *= (1.0 - m_op);
                        if transmittance < 0.01 { blocked = true; break; }
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
                if blocked { return 0.0; }
                continue;
            }

            var mat_opacity = 1.0;
            if r.slot != OCTREE_INTERIOR {
                atomicAdd(&stats[44], 1u);
                let attr = leaf_attr_pool[r.slot];
                let mid = leaf_attr_material_primary(attr);
                atomicAdd(&stats[47], 1u);
                mat_opacity = materials[mid].opacity;
            }

            if mat_opacity >= 0.99 {
                return 0.0;
            }
            transmittance *= (1.0 - mat_opacity);
            if transmittance < 0.01 {
                return 0.0;
            }

            t += min_step;
        }
    }

    return transmittance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Half-res pixel: each thread owns one output pixel in the half-res
    // shadow texture. Read primary surface data from the full-res G-buffer
    // at (gid.xy * 2) — we use the block's top-left pixel as the
    // representative surface for the 2×2 block. Bilateral upsample in the
    // shade pass compensates for the undersampling at silhouettes.
    let half_coord = vec2<i32>(gid.xy);
    let full_coord = half_coord * 2;

    let dims = textureDimensions(gbuf_position);
    if u32(full_coord.x) >= dims.x || u32(full_coord.y) >= dims.y {
        textureStore(shadow_lo_res, half_coord, vec4<f32>(1.0));
        return;
    }

    let pos_sample = textureLoad(gbuf_position, full_coord, 0);
    let normal_sample = textureLoad(gbuf_normal, full_coord, 0);
    let depth_marker = pos_sample.w;

    // Sky / miss pixels carry 1e10 in depth slot — no shadow needed.
    if depth_marker >= 1e9 {
        textureStore(shadow_lo_res, half_coord, vec4<f32>(1.0));
        return;
    }

    let surface_pos = pos_sample.xyz;
    let surface_normal = normalize(normal_sample.xyz);
    let num_objects = march_params.object_count;

    var shadow_values = vec4<f32>(1.0);
    if march_params.shadow_max_steps > 0u {
        var shadow_idx = 0u;
        for (var li = 0u; li < march_params.num_lights && shadow_idx < 4u; li++) {
            let light = lights[li];
            let cast_shadow = light.params.w;
            if cast_shadow < 0.5 { continue; }

            let light_type = u32(light.position.w);
            var shadow_dir: vec3<f32>;
            var shadow_max_dist = 1e20;

            if light_type == 0u {
                shadow_dir = normalize(-light.direction.xyz);
            } else {
                let to_light = light.position.xyz - surface_pos;
                let dist_to_light = length(to_light);
                let range = light.params.x;
                if range > 0.0 && dist_to_light > range {
                    shadow_values[shadow_idx] = 1.0;
                    shadow_idx++;
                    continue;
                }
                shadow_dir = to_light / max(dist_to_light, 0.001);
                shadow_max_dist = dist_to_light;

                if light_type == 2u {
                    let spot_cos = dot(-shadow_dir, normalize(light.direction.xyz));
                    let spot_angle_cos = cos(light.params.y);
                    if spot_cos < spot_angle_cos {
                        shadow_values[shadow_idx] = 1.0;
                        shadow_idx++;
                        continue;
                    }
                }
            }

            let n_dot_l = dot(surface_normal, shadow_dir);
            if n_dot_l <= 0.0 {
                shadow_values[shadow_idx] = 0.0;
                shadow_idx++;
                continue;
            }

            shadow_values[shadow_idx] = trace_shadow_ray(
                surface_pos,
                shadow_dir,
                num_objects,
                march_params.shadow_max_steps,
                shadow_max_dist,
            );
            shadow_idx++;
        }
    }

    textureStore(shadow_lo_res, half_coord, shadow_values);
}
