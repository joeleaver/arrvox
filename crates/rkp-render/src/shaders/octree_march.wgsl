// Octree-accelerated compute ray marcher.
//
// Step-and-query: advance along the ray, query the octree at each position.
// EMPTY nodes at coarse depth levels let us skip large regions in one step.
// Surface detected at first occupied voxel (opacity > threshold).

const OCTREE_EMPTY: u32 = 0xFFFFFFFFu;
const OCTREE_INTERIOR: u32 = 0xFFFFFFFEu;
const OCTREE_LEAF_BIT: u32 = 0x80000000u;
const OPACITY_THRESHOLD: f32 = 0.05;
const MAX_STEPS: u32 = 256u;
const MAX_OBJECTS: u32 = 64u;

struct VoxelSample { word0: u32, word1: u32, }

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
    _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
    _pad4: u32, _pad5: u32, _pad6: u32, _pad7: u32,
    _pad8: u32, _pad9: u32, _pad10: u32, _pad11: u32,
}

struct CameraUniforms {
    position: vec4<f32>, forward: vec4<f32>,
    right: vec4<f32>, up: vec4<f32>,
    resolution: vec2<f32>, jitter: vec2<f32>,
    prev_vp: mat4x4<f32>, view_proj: mat4x4<f32>,
}

struct MarchParams { object_count: u32, _pad0: u32, _pad1: u32, _pad2: u32, }

struct GpuMaterial {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emission_strength: f32,
    opacity: f32,
}

struct OctreeResult { slot: u32, depth: u32, }

// --- Bindings ---

@group(0) @binding(0) var<storage, read> voxel_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> octree_nodes: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<RkpObject>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;

@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg32uint, write>;

@group(2) @binding(0) var<uniform> march_params: MarchParams;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;

// --- Helpers ---

fn extract_opacity(word0: u32) -> f32 {
    return clamp(unpack2x16float(word0 & 0xFFFFu).x, 0.0, 1.0);
}
fn extract_material_id(word1: u32) -> u32 { return word1 & 0xFFFFu; }
fn extract_secondary_material_id(word1: u32) -> u32 { return (word1 >> 16u) & 0xFFFFu; }
fn extract_blend_weight(word0: u32) -> u32 { return (word0 >> 16u) & 0xFFu; }

fn invert_rigid(m: mat4x4<f32>) -> mat4x4<f32> {
    let s2 = dot(m[0].xyz, m[0].xyz);
    let inv_s2 = 1.0 / s2;
    let c0 = vec3<f32>(m[0].x, m[1].x, m[2].x) * inv_s2;
    let c1 = vec3<f32>(m[0].y, m[1].y, m[2].y) * inv_s2;
    let c2 = vec3<f32>(m[0].z, m[1].z, m[2].z) * inv_s2;
    let t = m[3].xyz;
    let inv_t = -vec3<f32>(dot(c0, t), dot(c1, t), dot(c2, t));
    return mat4x4<f32>(
        vec4<f32>(c0, 0.0), vec4<f32>(c1, 0.0),
        vec4<f32>(c2, 0.0), vec4<f32>(inv_t, 1.0),
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

fn octree_lookup(root: u32, max_depth: u32, extent: f32, pos: vec3<f32>) -> OctreeResult {
    var offset = root;
    var half = extent * 0.5;
    var center = vec3<f32>(half);
    for (var level = 0u; level < max_depth; level++) {
        let node = octree_nodes[offset];
        if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, level); }
        if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, level); }
        if (node & OCTREE_LEAF_BIT) != 0u {
            return OctreeResult(node & ~OCTREE_LEAF_BIT, level);
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
    let node = octree_nodes[offset];
    if node == OCTREE_EMPTY { return OctreeResult(OCTREE_EMPTY, max_depth); }
    if node == OCTREE_INTERIOR { return OctreeResult(OCTREE_INTERIOR, max_depth); }
    if (node & OCTREE_LEAF_BIT) != 0u {
        return OctreeResult(node & ~OCTREE_LEAF_BIT, max_depth);
    }
    return OctreeResult(OCTREE_EMPTY, max_depth);
}

// Skip past an empty/interior node's region along the ray.
// Uses DDA exit: find the nearest axis-aligned plane the ray crosses to leave this node.
fn skip_node(pos: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, node_depth: u32, extent: f32, vs: f32) -> f32 {
    let node_size = extent / f32(1u << node_depth);
    let node_min = floor(pos / node_size) * node_size;
    let node_max = node_min + node_size;
    // Exit planes: for positive dir, exit through max; for negative, through min.
    let t_exit = select((node_min - pos) * inv_dir, (node_max - pos) * inv_dir, dir > vec3<f32>(0.0));
    // Smallest positive exit = nearest boundary crossing.
    let t_pos = max(t_exit, vec3<f32>(1e-6));
    return min(min(t_pos.x, t_pos.y), t_pos.z) + vs * 0.01;
}

fn sample_opacity(pos: vec3<f32>, root: u32, depth: u32, extent: f32) -> f32 {
    let r = octree_lookup(root, depth, extent, pos);
    if r.slot == OCTREE_EMPTY { return 0.0; }
    if r.slot == OCTREE_INTERIOR { return 1.0; }
    return extract_opacity(voxel_pool[r.slot].word0);
}

fn sample_trilinear(pos: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32) -> f32 {
    let h = vs * 0.5;
    let s000 = sample_opacity(pos + vec3(-h,-h,-h), root, depth, extent);
    let s100 = sample_opacity(pos + vec3( h,-h,-h), root, depth, extent);
    let s010 = sample_opacity(pos + vec3(-h, h,-h), root, depth, extent);
    let s110 = sample_opacity(pos + vec3( h, h,-h), root, depth, extent);
    let s001 = sample_opacity(pos + vec3(-h,-h, h), root, depth, extent);
    let s101 = sample_opacity(pos + vec3( h,-h, h), root, depth, extent);
    let s011 = sample_opacity(pos + vec3(-h, h, h), root, depth, extent);
    let s111 = sample_opacity(pos + vec3( h, h, h), root, depth, extent);
    let f = fract(pos / vs + 0.5);
    let x0 = mix(s000, s100, f.x); let x1 = mix(s010, s110, f.x);
    let x2 = mix(s001, s101, f.x); let x3 = mix(s011, s111, f.x);
    return mix(mix(x0, x1, f.y), mix(x2, x3, f.y), f.z);
}

fn compute_normal(pos: vec3<f32>, root: u32, depth: u32, extent: f32, vs: f32) -> vec3<f32> {
    let eps = vs * 0.5;
    let gx = sample_trilinear(pos + vec3(eps,0.0,0.0), root, depth, extent, vs)
           - sample_trilinear(pos - vec3(eps,0.0,0.0), root, depth, extent, vs);
    let gy = sample_trilinear(pos + vec3(0.0,eps,0.0), root, depth, extent, vs)
           - sample_trilinear(pos - vec3(0.0,eps,0.0), root, depth, extent, vs);
    let gz = sample_trilinear(pos + vec3(0.0,0.0,eps), root, depth, extent, vs)
           - sample_trilinear(pos - vec3(0.0,0.0,eps), root, depth, extent, vs);
    let grad = vec3<f32>(gx, gy, gz);
    let len = length(grad);
    if len < 1e-8 { return vec3<f32>(0.0, 1.0, 0.0); }
    return -grad / len;
}

// --- Accumulating march (per object) ---
//
// Front-to-back opacity accumulation within a single object. Accumulates
// position and color (cheap). Normal computed ONCE at the end (expensive).

struct MarchResult {
    oc_pos: vec3<f32>,       // weighted average octree-space position
    color: vec3<f32>,        // weighted average color
    alpha: f32,              // total accumulated opacity
    t: f32,                  // parameter of first contribution
    first_slot: u32,         // voxel slot of first hit (for material)
    valid: bool,
}

fn march_object(
    world_origin: vec3<f32>, world_dir: vec3<f32>, obj: RkpObject, max_t: f32,
) -> MarchResult {
    var result = MarchResult(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, 0u, false);

    let inv_world = invert_rigid(obj.world);
    let local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir = normalize((inv_world * vec4<f32>(world_dir, 0.0)).xyz);

    let root = obj.octree_root;
    let max_depth = obj.octree_depth;
    let extent = bitcast<f32>(obj.octree_extent_bits);
    let vs = obj.voxel_size;
    let half_ext = extent * 0.5;

    let oc_origin = local_origin + vec3<f32>(half_ext);
    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let t_range = intersect_aabb(oc_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));
    if t_range.x > t_range.y || t_range.x > max_t {
        return result;
    }

    var t = t_range.x;

    for (var step = 0u; step < MAX_STEPS; step++) {
        if t > t_range.y || t > max_t { break; }
        if result.alpha > 0.99 { break; }

        let pos = clamp(oc_origin + safe_dir * t, vec3<f32>(vs * 0.01), vec3<f32>(extent - vs * 0.01));
        let r = octree_lookup(root, max_depth, extent, pos);

        if r.slot == OCTREE_EMPTY {
            t += skip_node(pos, safe_dir, inv_dir, r.depth, extent, vs);
            continue;
        }

        var voxel_opacity = 0.0;
        var slot = 0u;
        var mat_opacity = 1.0;
        if r.slot == OCTREE_INTERIOR {
            voxel_opacity = 1.0;
        } else {
            slot = r.slot;
            voxel_opacity = extract_opacity(voxel_pool[slot].word0);
            let mid = extract_material_id(voxel_pool[slot].word1);
            mat_opacity = materials[mid].opacity;
        }

        // Threshold on raw voxel opacity (is this voxel occupied?).
        // Material opacity only affects accumulation weight, not occupancy.
        if voxel_opacity < OPACITY_THRESHOLD {
            t += vs * 0.5;
            continue;
        }

        let sample_opacity = voxel_opacity * mat_opacity;

        // Front-to-back compositing.
        let remaining = 1.0 - result.alpha;
        let weight = sample_opacity * remaining;

        // Accumulate position (octree space — cheap, no transform).
        result.oc_pos += pos * weight;

        // Accumulate per-voxel color.
        var color = vec3<f32>(0.5);
        if slot != 0u {
            let cp = color_pool_data[slot];
            if cp != 0u {
                color = vec3<f32>(
                    f32(cp & 0xFFu) / 255.0,
                    f32((cp >> 8u) & 0xFFu) / 255.0,
                    f32((cp >> 16u) & 0xFFu) / 255.0,
                );
            }
        }
        result.color += color * weight;
        result.alpha += weight;

        // First hit — record for depth and material.
        if !result.valid {
            result.t = t;
            result.first_slot = slot;
            result.valid = true;
        }

        t += vs * 0.5;
    }

    return result;
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = textureDimensions(gbuf_position);
    if pixel.x >= dims.x || pixel.y >= dims.y { return; }

    let coord = vec2<i32>(pixel.xy);
    let uv = (vec2<f32>(pixel.xy) + 0.5 + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz);

    // March all objects, collect results.
    var results: array<MarchResult, MAX_OBJECTS>;
    var obj_indices: array<u32, MAX_OBJECTS>;
    var result_count = 0u;

    let num_objects = march_params.object_count;
    for (var i = 0u; i < num_objects && i < MAX_OBJECTS; i++) {
        let obj = objects[i];
        if obj.geom_type == 0u { continue; }
        let r = march_object(ray_origin, ray_dir, obj, 1e20);
        if r.valid {
            results[result_count] = r;
            obj_indices[result_count] = i;
            result_count += 1u;
        }
    }

    if result_count == 0u {
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, 1e10));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    // Sort results front-to-back by entry t (insertion sort — few objects).
    for (var i = 1u; i < result_count; i++) {
        var j = i;
        while j > 0u && results[j - 1u].t > results[j].t {
            let tmp_r = results[j - 1u]; results[j - 1u] = results[j]; results[j] = tmp_r;
            let tmp_i = obj_indices[j - 1u]; obj_indices[j - 1u] = obj_indices[j]; obj_indices[j] = tmp_i;
            j -= 1u;
        }
    }

    // Front-to-back composite across objects.
    var accum_pos = vec3<f32>(0.0);
    var accum_normal = vec3<f32>(0.0);
    var accum_color = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    var first_dist = 0.0;
    var first_mat_id = 0u;
    var first_sec_mat = 0u;
    var first_blend = 0u;
    var first_obj_id = 0u;
    var have_first = false;

    for (var i = 0u; i < result_count; i++) {
        if accum_alpha > 0.99 { break; }

        let r = results[i];
        let oi = obj_indices[i];
        let obj = objects[oi];
        let extent = bitcast<f32>(obj.octree_extent_bits);
        let vs = obj.voxel_size;

        // Normalize this object's accumulated values.
        let inv_a = 1.0 / max(r.alpha, 0.001);
        let oc_pos = r.oc_pos * inv_a;
        let color = r.color * inv_a;

        // World position + normal for this object's contribution.
        let local_hit = oc_pos - vec3<f32>(extent * 0.5);
        let world_pos = (obj.world * vec4<f32>(local_hit, 1.0)).xyz;
        let local_normal = compute_normal(oc_pos, obj.octree_root, obj.octree_depth, extent, vs);
        let world_normal = normalize((obj.world * vec4<f32>(local_normal, 0.0)).xyz);

        // Composite this object's contribution.
        let remaining = 1.0 - accum_alpha;
        let weight = r.alpha * remaining;
        accum_pos += world_pos * weight;
        accum_normal += world_normal * weight;
        accum_color += color * weight;
        accum_alpha += weight;

        if !have_first {
            first_dist = length(world_pos - ray_origin);
            first_obj_id = obj.object_id;
            if r.first_slot != 0u {
                let voxel = voxel_pool[r.first_slot];
                first_mat_id = extract_material_id(voxel.word1);
                first_sec_mat = extract_secondary_material_id(voxel.word1);
                first_blend = extract_blend_weight(voxel.word0);
            } else {
                first_mat_id = obj.material_id;
            }
            have_first = true;
        }
    }

    let inv_alpha = 1.0 / max(accum_alpha, 0.001);
    let final_pos = accum_pos * inv_alpha;
    let final_normal = normalize(accum_normal);
    let final_color = accum_color * inv_alpha;

    let cr = u32(clamp(final_color.r, 0.0, 1.0) * 31.0);
    let cg = u32(clamp(final_color.g, 0.0, 1.0) * 63.0);
    let cb = u32(clamp(final_color.b, 0.0, 1.0) * 31.0);
    let color_rgb565 = cr | (cg << 5u) | (cb << 11u);

    let packed_r = (first_mat_id & 0xFFFFu) | ((first_sec_mat & 0xFFFFu) << 16u);
    let packed_g = (first_blend & 0xFFu)
                 | (((first_obj_id + 1u) & 0xFFu) << 8u)
                 | (color_rgb565 << 16u);

    textureStore(gbuf_position, coord, vec4<f32>(final_pos, first_dist));
    textureStore(gbuf_normal, coord, vec4<f32>(final_normal, accum_alpha));
    textureStore(gbuf_material, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
}
