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

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::math::mat4_affine_inverse;
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
import package::lib::types::leaf_attr_material_primary;
import package::lib::types::BandCell;
import package::lib::types::RkpInstance;
import package::lib::types::GpuMaterial;
import package::lib::types::mat_albedo;
import package::lib::types::CameraUniforms;
// Shadow rays scan every object every ray — AABB misses short-circuit
// cheaply, and shadow rays go in arbitrary directions so the march
// pass's screen-space per-tile lists don't apply.
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

// Per-instance + per-asset records — see octree_march.wgsl for the
// authoritative layouts. Shadow trace marches the same scene the primary
// march does, so the structs and bindings mirror.

struct RkpAsset {
    aabb_min: vec3<f32>, octree_root: u32,
    aabb_max: vec3<f32>, octree_depth: u32,
    octree_extent_bits: u32, voxel_size: f32,
    geom_type: u32, bone_count: u32,
    grid_origin: vec3<f32>, rest_octree_root: u32,
    rest_octree_depth: u32, rest_octree_extent_bits: u32,
    shader_id: u32, _pad: u32,
}

struct MarchParams {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    // Must match octree_march.wgsl: same uniform buffer binding.
    lod_enabled: u32,
    surfacenet_enabled: u32,
    tile_count_x: u32,
    // Phase 7 Session 4b — TLAS node count. Zero gates the BVH
    // traversal off; the helper function falls back to "no shadow
    // caster found" (transmittance unchanged).
    tlas_node_count: u32,
    // Phase 8 — `1` ⇒ a shadow-map march wrote a directional
    // depth texture this frame; the per-pixel directional rays in
    // this pass are skipped (the shade pass reads
    // `sample_shadow_map` for those lights instead).
    shadow_map_enabled: u32,
    // Phase B-redux Phase 3a — wire-format sync with octree_march's
    // uniform; shadow trace doesn't read either field yet (Phase 4
    // will when shadow goes through instance_at descent), but the
    // shared uniform binding requires a layout match.
    time: f32,
    asset_count: u32,
    _pad0: u32,
}

struct GpuLight {
    position: vec4<f32>,
    color: vec4<f32>,
    direction: vec4<f32>,
    params: vec4<f32>,
}

// vec3 fields flattened to f32 components — see rkp_shade.wgsl for the
// full rationale (WGSL vec3 alignment would pad this to 128 bytes, but
// the Rust/GpuMaterial is tightly packed at 96).
// Phase B-redux 3b — band-cell wire format. Mirrors
// `octree_march.wgsl::BandCell` (and the Rust `GpuBandCell` it
// targets). Each band cell's payload reinterprets two consecutive
// `leaf_attr_pool` slots; see `read_band_cell` below.
// Phase 4 — user-shader-facing types for the band-cell shadow
// dispatch. Same shapes as octree_march.wgsl so the composer's
// shared `dispatch_user_instance_descend` body splices in
// unchanged. Wire-format with the corresponding Rust structs.
struct HostSample {
    distance: f32,
    normal: vec3<f32>,
    material: u32,
    material_secondary: u32,
    blend_weight: u32,
}

struct UserCtx {
    time: f32,
    cell_size: f32,
    material_id: u32,
    aabb_min: vec3<f32>,
    params: array<f32, 8>,
}

struct InstanceHit {
    valid: bool,
    world_t: f32,
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
}

// World-space AABB returned by per-shader `inst_aabb` hooks. Mirrors
// `octree_march.wgsl::Aabb`; needed here because the composer-emitted
// `rkp_user_<id>_inst_aabb` body returns this type.
struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
}

// MAX_STEPS for `descend_proto_octree`'s outer DDA loop. Protos are
// small (max_depth ≤ 8) so the cap rarely matters in practice.
const MAX_STEPS_PROTO: u32 = 4096u;

// `unpack_oct_normal` is imported from `lib::oct_normal` at the top
// of this file.

// Prototype-octree descent, mirror of `octree_march.wgsl::descend_proto_octree`.
// Signature differs only in the inner `octree_lookup` call — shadow_trace's
// 5-arg variant skips the LOD machinery the primary march uses (LOD is
// pointless for shadow rays + small proto octrees). Called from the
// composer-emitted per-shader `rkp_user_<id>_instance_descend` body.
struct ProtoHit {
    valid: bool,
    local_t: f32,
    oc_pos: vec3<f32>,
    local_normal: vec3<f32>,
    leaf_slot: u32,
}

fn descend_proto_octree(
    asset: RkpAsset,
    oc_origin: vec3<f32>,
    safe_dir: vec3<f32>,
    inv_dir: vec3<f32>,
    t_start: f32,
    t_end: f32,
    local_to_world: f32,
) -> ProtoHit {
    var r: ProtoHit;
    r.valid = false;
    r.local_t = t_end;
    r.oc_pos = vec3<f32>(0.0);
    r.local_normal = vec3<f32>(0.0, 1.0, 0.0);
    r.leaf_slot = 0u;

    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    let root = asset.octree_root;
    let max_depth = asset.octree_depth;
    let lookup_bias = vs * 1.0e-3;

    var t = t_start;
    for (var step = 0u; step < MAX_STEPS_PROTO; step++) {
        if t > t_end { break; }

        let pos = clamp(
            oc_origin + safe_dir * (t + lookup_bias),
            vec3<f32>(vs * 0.01),
            vec3<f32>(extent - vs * 0.01),
        );
        let lookup = octree_lookup(root, max_depth, extent, pos, PHASE_SHADOW);

        if lookup.slot == OCTREE_EMPTY {
            t += skip_node(pos, safe_dir, inv_dir, lookup.depth, extent, vs);
            continue;
        }
        if lookup.slot == OCTREE_INTERIOR {
            r.valid = true;
            r.local_t = t;
            r.oc_pos = oc_origin + safe_dir * t;
            r.local_normal = -safe_dir;
            r.leaf_slot = 0u;
            return r;
        }

        if slot_is_brick(lookup.slot) {
            var brick_id = slot_brick_id(lookup.slot);
            let cell_size = (lookup.cell_half * 2.0) / BRICK_DIM_F;
            let inv_cell_size = 1.0 / cell_size;
            var brick_origin = lookup.cell_center - vec3<f32>(lookup.cell_half);
            var brick_base = brick_id * BRICK_CELLS;

            let p0 = oc_origin + safe_dir * t;
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
            let dda_eps = cell_size * 1.0e-3;

            for (var bs = 0u; bs < BRICK_MAX_STEPS; bs++) {
                if t > t_end { break; }

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
                        r.valid = true;
                        r.local_t = t;
                        r.oc_pos = oc_origin + safe_dir * t;
                        r.local_normal = -safe_dir;
                        r.leaf_slot = 0u;
                        return r;
                    }
                    if link == FACE_EMPTY_LINK { break; }
                    brick_id = link;
                    brick_base = link * BRICK_CELLS;
                    let brick_extent = BRICK_DIM_F * cell_size;
                    if face_idx == FACE_NX { cell.x = 3; brick_origin.x -= brick_extent; }
                    else if face_idx == FACE_PX { cell.x = 0; brick_origin.x += brick_extent; }
                    else if face_idx == FACE_NY { cell.y = 3; brick_origin.y -= brick_extent; }
                    else if face_idx == FACE_PY { cell.y = 0; brick_origin.y += brick_extent; }
                    else if face_idx == FACE_NZ { cell.z = 3; brick_origin.z -= brick_extent; }
                    else { cell.z = 0; brick_origin.z += brick_extent; }
                    let p_now = oc_origin + safe_dir * t;
                    let next_b2 = brick_origin + (vec3<f32>(cell) + step_gt) * cell_size;
                    t_max = t + (next_b2 - p_now) * inv_dir;
                }

                let cx = u32(cell.x);
                let cy = u32(cell.y);
                let cz = u32(cell.z);
                let flat = cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM;
                let c = brick_pool[brick_base + flat];

                if c != BRICK_CELL_EMPTY && c != BRICK_CELL_INTERIOR {
                    let attr = leaf_attr_pool[c];
                    r.valid = true;
                    r.local_t = t;
                    r.oc_pos = oc_origin + safe_dir * t;
                    r.local_normal = unpack_oct_normal(attr.normal_oct);
                    r.leaf_slot = c;
                    return r;
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
            t += vs;
            continue;
        }

        // Plain leaf — single-voxel slot.
        let leaf_slot = lookup.slot & OCTREE_PAYLOAD_MASK;
        let attr = leaf_attr_pool[leaf_slot];
        r.valid = true;
        r.local_t = t;
        r.oc_pos = oc_origin + safe_dir * t;
        r.local_normal = unpack_oct_normal(attr.normal_oct);
        r.leaf_slot = leaf_slot;
        return r;
    }
    return r;
}


// Opacity extinction reference: opacity=X over glass this thick gives
// transmittance `1 - X` at full density, matching the primary pass's
// one-shot `(1 - opacity)` gate. Glass thinner than this brightens
// the shadow, thicker darkens it.
const OPACITY_REFERENCE_M: f32 = 0.3;

// Overall glass-shadow darkness multiplier. Scales the total
// extinction sigma down: 1.0 = full physical match to the primary
// transmission pass; <1.0 lets more light through the shadow ray
// regardless of glass thickness. Chosen by eye after user feedback
// that a physically-matched pass read as "too dark" — 0.25 gives
// roughly 1/4 the darkness across the board (more visually, since
// the mapping is `T → T^0.25` in the exponential).
const GLASS_SHADOW_DENSITY: f32 = 0.25;

// Scalar Beer transmittance for shadows — combines albedo absorption
// (mirroring rkp_shade's `beer_absorption`) with a per-meter opacity
// extinction so cell-by-cell shadow-ray accumulation stays aligned
// with the primary pass's `beer(albedo, total_t) * (1 - opacity)`
// behavior, then softens the whole thing by `GLASS_SHADOW_DENSITY`.
// Returns a scalar (shadow rays care about visibility, not color) —
// per-channel albedo absorption is averaged, which is fine for the
// typical mildly-tinted glass case.
fn shadow_beer(glass_color: vec3<f32>, opacity: f32, thickness: f32) -> f32 {
    let sigma_albedo_v = max(-log(max(glass_color, vec3<f32>(0.01))), vec3<f32>(0.0));
    let sigma_albedo = (sigma_albedo_v.x + sigma_albedo_v.y + sigma_albedo_v.z) * (1.0 / 3.0);
    let sigma_opacity = -log(max(1.0 - opacity, 0.01)) / OPACITY_REFERENCE_M;
    let sigma = (sigma_albedo * 5.0 + sigma_opacity) * GLASS_SHADOW_DENSITY;
    return exp(-sigma * thickness);
}

// Group 0: scene data (shared with march).
@group(0) @binding(0) var<storage, read> brick_pool: array<u32>;
// Interleaved (node_value, prefilter_attr_id) — see octree_march.wgsl.
@group(0) @binding(1) var<storage, read> octree_nodes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> instances: array<RkpInstance>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> color_pool_data: array<u32>;
// brick_face_links[brick_id * 6 + face] → neighbor brick_id, or one of
// FACE_EMPTY_LINK / FACE_INTERIOR. See rkp_core::brick_face_links.
@group(0) @binding(7) var<storage, read> brick_face_links: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_attr_pool: array<LeafAttr>;
// Deformed-space bone field + 4³ brick occupancy bitmap — same
// buffers the skin-deform scatter writes and the primary march reads.
// Shadow only needs to know whether a deformed cell is populated, so
// it ignores the payload's normal half and just tests `leaf_slot`.
@group(0) @binding(9) var<storage, read> bone_field: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read> bone_field_occ: array<u32>;
@group(0) @binding(12) var<storage, read> assets: array<RkpAsset>;

// Per-instance paint overlay (Phase 3) — see octree_march.wgsl for the
// authoritative description. Shadow trace uses the same fetch helpers.
struct OverlayEntry {
    leaf_slot: u32,
    normal_oct: u32,
    material_packed: u32,
    color_packed: u32,
}
@group(0) @binding(13) var<storage, read> instance_overlay: array<OverlayEntry>;

fn fetch_overlay_index(inst: RkpInstance, leaf_slot: u32) -> u32 {
    if (inst.overlay_count == 0u) {
        return 0xFFFFFFFFu;
    }
    var lo: u32 = 0u;
    var hi: u32 = inst.overlay_count;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        let e = instance_overlay[inst.overlay_offset + mid];
        if (e.leaf_slot < leaf_slot) {
            lo = mid + 1u;
        } else if (e.leaf_slot > leaf_slot) {
            hi = mid;
        } else {
            return inst.overlay_offset + mid;
        }
    }
    return 0xFFFFFFFFu;
}

fn fetch_leaf_attr_for(inst: RkpInstance, leaf_slot: u32) -> LeafAttr {
    let idx = fetch_overlay_index(inst, leaf_slot);
    if (idx != 0xFFFFFFFFu) {
        let e = instance_overlay[idx];
        return LeafAttr(e.normal_oct, e.material_packed);
    }
    return leaf_attr_pool[leaf_slot];
}

fn fetch_leaf_color_for(inst: RkpInstance, leaf_slot: u32) -> u32 {
    let idx = fetch_overlay_index(inst, leaf_slot);
    if (idx != 0xFFFFFFFFu) {
        return instance_overlay[idx].color_packed;
    }
    return color_pool_data[leaf_slot];
}

// Inverse of an affine 4x4 matrix. See `mat4_affine_inverse` in
// octree_march.wgsl for the full derivation. Duplicated here because
// shadow trace and primary march don't share a WGSL source file.
// `mat4_affine_inverse` is imported from `lib::math` at the top.

// Group 1: gbuf inputs (full-res, read) + half-res shadow output (write).
@group(1) @binding(0) var gbuf_position: texture_2d<f32>;
@group(1) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(1) @binding(2) var shadow_lo_res: texture_storage_2d<rgba8unorm, write>;

// Group 2: march params + materials + stats + lights (shared with march).
// Shadow trace doesn't need the tile-list buffers (shadow rays don't
// share a screen tile), so bindings 4-7 stay undeclared here — naga
// drops them from the shadow pipeline. Phase 7 Session 4b adds the
// TLAS bindings 8/9 which shadow trace DOES read.
@group(2) @binding(0) var<uniform> march_params: MarchParams;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 64>;
@group(2) @binding(3) var<storage, read> lights: array<GpuLight>;
// Phase 4 — per-material 8 × f32 array shared with the primary
// march. Used to populate `ctx.params` for the band-cell instance
// descent dispatcher when a band cell is hit by a shadow ray.
@group(2) @binding(10) var<storage, read> shader_params: array<f32>;

// Phase 7 — TLAS over instance AABBs. Built each frame on CPU by
// `TlasPass::build_tlas` and uploaded; shadow trace traverses
// stack-based to find candidate occluders.
struct TlasNode {
    aabb_min: vec3<f32>,
    left_or_leaf: u32,    // high bit = leaf marker (0x80000000)
    aabb_max: vec3<f32>,
    right_or_count: u32,
}
struct TlasInstanceLeaf {
    asset_id: u32,
    instance_state_offset: u32,  // legacy field kept for TLAS wire-format compatibility
    material_id: u32,
    instance_index: u32,         // host: index into instances[]; user-shader: TLAS_LEAF_USER_SHADER
}
@group(2) @binding(8) var<storage, read> tlas_nodes: array<TlasNode>;
@group(2) @binding(9) var<storage, read> tlas_leaves: array<TlasInstanceLeaf>;

const TLAS_NODE_LEAF_BIT: u32 = 0x80000000u;
const TLAS_LEAF_USER_SHADER: u32 = 0xFFFFFFFEu;
// BVH traversal stack depth. Binary tree with N leaves is ≤ ⌈log2 N⌉
// deep; 32 covers up to 4 billion leaves.
const TLAS_STACK_DEPTH: u32 = 32u;

const PHASE_SHADOW: u32 = 2u;


// `intersect_aabb` is duplicated across splice-consumer files
// (octree_march, rkp_shadow_trace, shadow_scatter) — it gets
// referenced from the user-shader composer's spliced
// `instance_at` chunk, which uses the unmangled name. Lifting it
// to lib would break the splice (WESL renames imported symbols).
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
            // Phase 4 — preserve BRICK_BIT and BAND_BIT in the
            // returned slot so the caller can distinguish a leaf-
            // attr leaf from a brick leaf or a band cell. Band
            // cells trigger `dispatch_user_instance_descend` for
            // proper shadow casting through painted instance
            // shaders (e.g. grass blades).
            return OctreeResult(
                (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT) | (node & OCTREE_BAND_BIT),
                level, center, half,
            );
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
        return OctreeResult(
            (node & OCTREE_PAYLOAD_MASK) | (node & OCTREE_BRICK_BIT) | (node & OCTREE_BAND_BIT),
            max_depth, center, half,
        );
    }
    return OctreeResult(OCTREE_EMPTY, max_depth, center, half);
}

// `slot_is_brick`, `slot_brick_id`, `slot_is_band`, `slot_band_offset`
// are imported from `lib::octree_slot` at the top of this file.

/// Phase B-redux 3b — read a `BandCell` payload by unpacking two
/// consecutive `leaf_attr_pool` slots. Same byte layout the BFS
/// bake (user_shader_geom.wgsl) writes.
fn read_band_cell(band_off: u32) -> BandCell {
    let s0 = leaf_attr_pool[band_off];
    let s1 = leaf_attr_pool[band_off + 1u];
    var c: BandCell;
    c.anchor_world_pos = vec3<f32>(
        bitcast<f32>(s0.normal_oct),
        bitcast<f32>(s0.material_packed),
        bitcast<f32>(s1.normal_oct),
    );
    c.material_id = s1.material_packed;
    return c;
}

const USER_INSTANCE_AT_DISPATCH_BEGIN: u32 = 0u;
// Default identity stub — Rust composer replaces this whole block
// (markers + body) with per-shader descent helpers + the dispatch
// switch when any registered shader provides an `instance_at` hook.
// Empty-registry path returns "no instance hit" so the band-cell
// shadow branch falls through to the no-hit case (light passes
// straight through).
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

// `skip_node` is imported from `lib::octree` at the top.

// ── Skinned-object shadow branch ────────────────────────────────────
//
// Walks the scatter pass's deformed bone field (binding 9) the same
// way the primary `march_object_skinned` does in `octree_march.wgsl`,
// with the same 4³-brick occupancy skip (binding 10). On the first
// populated cell it looks up the leaf's material opacity and either
// blocks the ray or attenuates transmittance — same semantics as the
// rigid path below.

const OCC_BRICK_DIM: i32 = 4;
const SKINNED_SHADOW_MAX_STEPS: u32 = 512u;

fn skinned_shadow_brick_populated(
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

fn skinned_shadow_brick_exit_t(
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

fn skinned_shadow_sample(
    cell: vec3<i32>, dims: vec3<i32>, offset: u32,
) -> u32 {
    if any(cell < vec3<i32>(0)) || any(cell >= dims) { return 0u; }
    let idx = u32(cell.x)
        + u32(cell.y) * u32(dims.x)
        + u32(cell.z) * u32(dims.x) * u32(dims.y);
    return bone_field[offset + idx].x;
}

/// Trace a shadow ray through one skinned object. Returns the new
/// transmittance; caller folds this into its running total.
fn trace_shadow_skinned(
    world_origin: vec3<f32>, world_dir: vec3<f32>,
    inst: RkpInstance, asset: RkpAsset,
    max_world_dist: f32, transmittance_in: f32,
) -> f32 {
    let inv_world = mat4_affine_inverse(inst.world);
    let local_origin_mesh = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    let local_scale = length(local_dir_unnorm);
    let local_max_t = max_world_dist * local_scale;
    let vs = asset.voxel_size;

    let rest_extent = bitcast<f32>(asset.rest_octree_extent_bits);
    let local_origin = local_origin_mesh + vec3<f32>(rest_extent * 0.5);

    let grid_dim = vec3<i32>(
        i32(inst.bone_field_dim_x),
        i32(inst.bone_field_dim_y),
        i32(inst.bone_field_dim_z),
    );
    if grid_dim.x <= 0 || grid_dim.y <= 0 || grid_dim.z <= 0 {
        return transmittance_in;
    }
    let grid_origin = vec3<f32>(
        inst.bone_field_origin_x,
        inst.bone_field_origin_y,
        inst.bone_field_origin_z,
    );
    let grid_max = grid_origin + vec3<f32>(grid_dim) * vs;

    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    // Push the start point forward by a few voxels so self-shadow
    // from the surface we're shading doesn't block its own light.
    // Primary shadow path uses a similar `vs * 4` bias.
    let shadow_origin = local_origin + safe_dir * vs * 4.0;
    let t_range = intersect_aabb(shadow_origin, inv_dir, grid_origin, grid_max);
    if t_range.x > t_range.y { return transmittance_in; }
    let t_limit = min(t_range.y, local_max_t);

    var t = max(t_range.x, 0.0) + vs * 0.001;
    var transmittance = transmittance_in;

    for (var step = 0u; step < SKINNED_SHADOW_MAX_STEPS; step++) {
        if t > t_limit { break; }
        let p_local = shadow_origin + safe_dir * t;
        let cell_f = (p_local - grid_origin) / vs;
        let cell_i = vec3<i32>(floor(cell_f));

        if !skinned_shadow_brick_populated(cell_i, grid_dim, inst.bone_field_occ_offset) {
            let t_exit = skinned_shadow_brick_exit_t(shadow_origin, inv_dir, cell_i, grid_origin, vs);
            t = max(t + vs * 0.01, t_exit + vs * 0.001);
            continue;
        }

        let leaf_slot = skinned_shadow_sample(cell_i, grid_dim, inst.bone_field_offset);
        if leaf_slot == 0u {
            t += vs;
            continue;
        }

        let attr = fetch_leaf_attr_for(inst, leaf_slot);
        let mid = leaf_attr_material_primary(attr);
        let m_op = materials[mid].opacity;
        if m_op >= 0.99 { return 0.0; }
        // Route through `shadow_beer` so opacity extinction is
        // thickness-proportional rather than per-iteration linear
        // (which compounded into near-black on deep glass bodies).
        let glass_albedo = mat_albedo(materials[mid]);
        let step_world = vs / local_scale;
        transmittance *= shadow_beer(glass_albedo, m_op, step_world);
        if transmittance < 0.01 { return 0.0; }
        t += vs;
    }
    return transmittance;
}

// Shadow ray — returns transmittance in [0, 1]. 0 = fully occluded, 1 = lit.
// Per-instance shadow descent — one ray, one instance. Returns the
// updated transmittance, or `0.0` on full occlusion. Phase 7 Session
// 4a extracted this from `trace_shadow_ray`'s for-loop body so the
// BVH traversal in 4b can dispatch the same code path from leaf hits.
//
// `transmittance_in` is the running transmittance accumulator; the fn
// multiplies into it and returns the new value. Caller short-circuits
// on `< 0.01`.
fn shadow_step_one_instance(
    inst: RkpInstance,
    asset: RkpAsset,
    world_origin: vec3<f32>,
    world_dir: vec3<f32>,
    max_steps: u32,
    max_world_dist: f32,
    transmittance_in: f32,
) -> f32 {
    var transmittance = transmittance_in;

    if asset.geom_type == 0u { return transmittance; }
    // Same gate as primary visibility (Phase 2). SHADOW_ONLY semantics
    // come later — they need a separate camera shadow_layer_mask.
    if (inst.layer_mask & camera.layer_mask) == 0u
        && inst.object_id != camera.focus_object_id { return transmittance; }

    // Skinned objects walk the per-frame deformed bone field
    // instead of the rest-pose octree — otherwise the shadow would
    // track a stale (bind-pose) silhouette.
    if inst.is_skinned != 0u && inst.bone_field_dim_x > 0u {
        return trace_shadow_skinned(
            world_origin, world_dir, inst, asset, max_world_dist, transmittance,
        );
    }

    // Standard host-asset path — affine inv_world × world.
    let inv_world = mat4_affine_inverse(inst.world);
    let local_origin = (inv_world * vec4<f32>(world_origin, 1.0)).xyz;
    let local_dir_unnorm = (inv_world * vec4<f32>(world_dir, 0.0)).xyz;
    let local_dir = normalize(local_dir_unnorm);
    let local_scale = length(local_dir_unnorm);
    let local_max_t = max_world_dist * local_scale;

    let root = asset.octree_root;
    let max_depth = asset.octree_depth;
    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    let min_step = vs * 2.0;

    let oc_origin = local_origin - asset.grid_origin;
    let safe_dir = vec3<f32>(
        select(local_dir.x, select(-1e-10, 1e-10, local_dir.x >= 0.0), abs(local_dir.x) < 1e-10),
        select(local_dir.y, select(-1e-10, 1e-10, local_dir.y >= 0.0), abs(local_dir.y) < 1e-10),
        select(local_dir.z, select(-1e-10, 1e-10, local_dir.z >= 0.0), abs(local_dir.z) < 1e-10),
    );
    let inv_dir = 1.0 / safe_dir;

    let shadow_origin = oc_origin + safe_dir * vs * 4.0;
    let t_range = intersect_aabb(shadow_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(extent));
    if t_range.x > t_range.y { return transmittance; }

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

        // Phase 4 — band-cell shadow dispatch with partial opacity.
        // V1's "any hit blocks fully" produced black grass via dense
        // self-shadow. Now: each band cell that contains a blade hit
        // attenuates transmittance by `BAND_BLADE_TRANSMITTANCE`. The
        // blade-hit bookkeeping is per-cell rather than per-blade
        // (no recursive descent past the first hit), which models
        // the cell's average blade-projected coverage well enough for
        // soft self-shadowing and avoids the O(blades-per-cell)
        // descent cost. After processing, advance past the cell so
        // subsequent cells along the ray accumulate independently.
        if slot_is_band(r.slot) {
            atomicAdd(&stats[62], 1u); // band-cell shadow invocation
            let band_off = slot_band_offset(r.slot);
            let band = read_band_cell(band_off);
            let band_mat_id = band.material_id;
            let band_mat = materials[band_mat_id];
            let shader_id = band_mat.instance_shader_id;
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
            if proto_idx != 0xFFFFFFFFu {
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
                // Dispatcher takes WORLD-space ray; inputs to this
                // function are already world-frame.
                let inst_hit = dispatch_user_instance_descend(
                    shader_id,
                    band.anchor_world_pos,
                    host_sample,
                    0u,
                    world_origin,
                    world_dir,
                    max_world_dist,
                    ctx,
                    proto_asset,
                );
                if inst_hit.valid {
                    atomicAdd(&stats[63], 1u); // band-cell shadow hit
                    let BAND_BLADE_TRANSMITTANCE: f32 = 0.65;
                    transmittance *= BAND_BLADE_TRANSMITTANCE;
                    if transmittance < 0.01 { return 0.0; }
                }
            }
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
                        // Ray exits brick into a solid-bulk region.
                        // For glass objects, the bulk is part of
                        // the glass body — bail out of the brick
                        // DDA and let the outer loop attenuate
                        // the ray across the INTERIOR_NODE span
                        // via Beer. Opaque objects still block.
                        let obj_opacity = materials[inst.material_id].opacity;
                        if obj_opacity < 0.99 {
                            break;
                        }
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
                if c == BRICK_CELL_INTERIOR {
                    // Solid-bulk cell inside a brick. Opaque
                    // objects keep the old "skip like air"
                    // behavior (shadow passes straight through
                    // mesh interiors). Glass objects attenuate
                    // by one cell's worth of Beer + opacity.
                    let obj_opacity = materials[inst.material_id].opacity;
                    if obj_opacity < 0.99 {
                        let glass_albedo = mat_albedo(materials[inst.material_id]);
                        let cell_world = cell_size / local_scale;
                        transmittance *= shadow_beer(glass_albedo, obj_opacity, cell_world);
                        if transmittance < 0.01 { blocked = true; break; }
                    }
                } else if c != BRICK_CELL_EMPTY {
                    let attr = fetch_leaf_attr_for(inst, c);
                    let mid = leaf_attr_material_primary(attr);
                    let m_op = materials[mid].opacity;
                    if m_op >= 0.99 { blocked = true; break; }
                    // Full Beer + opacity over one cell width.
                    // Earlier code used a per-cell linear
                    // `(1 - m_op)` multiplier which compounded
                    // across many cells into near-black shadows;
                    // routing through `shadow_beer` keeps the
                    // extinction thickness-proportional and
                    // matches the primary pass's one-shot gate.
                    let glass_albedo = mat_albedo(materials[mid]);
                    let cell_world = cell_size / local_scale;
                    transmittance *= shadow_beer(glass_albedo, m_op, cell_world);
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

        // INTERIOR handling mirrors the primary march: a solid-
        // bulk region inside a glass object attenuates the light
        // across the full span (Beer over the node's ray length)
        // and the shadow ray continues. Opaque INTERIOR still
        // blocks the light, matching the existing mesh-interior
        // shadow behavior.
        if r.slot == OCTREE_INTERIOR {
            let obj_opacity = materials[inst.material_id].opacity;
            if obj_opacity < 0.99 {
                let span = max(skip_node(pos, safe_dir, inv_dir, r.depth, extent, vs), min_step);
                let glass_albedo = mat_albedo(materials[inst.material_id]);
                let span_world = span / local_scale;
                transmittance *= shadow_beer(glass_albedo, obj_opacity, span_world);
                if transmittance < 0.01 { return 0.0; }
                t += span;
                continue;
            }
            return 0.0;
        }

        atomicAdd(&stats[44], 1u);
        let attr = fetch_leaf_attr_for(inst, r.slot);
        let mid = leaf_attr_material_primary(attr);
        atomicAdd(&stats[47], 1u);
        let mat_opacity = materials[mid].opacity;

        if mat_opacity >= 0.99 {
            return 0.0;
        }
        // Route through `shadow_beer` so albedo + opacity both
        // accumulate as thickness-proportional extinction.
        // Earlier per-iteration `(1 - opacity)` multiplier
        // compounded across many cells into over-dark shadows.
        let glass_albedo = mat_albedo(materials[mid]);
        let step_world = min_step / local_scale;
        transmittance *= shadow_beer(glass_albedo, mat_opacity, step_world);
        if transmittance < 0.01 {
            return 0.0;
        }

        t += min_step;
    }

    return transmittance;
}

// Phase 7 Session 4b — synthesize a minimal `RkpInstance` from a
// user-shader TLAS leaf. The user-shader Option B descent path that
// once consumed `instance_state_offset` is gone; this synthesis is
// kept so user-shader TLAS leaves still produce a valid instance for
// `shadow_step_one_instance`'s host-affine path. The leaf's
// `instance_state_offset` is silently dropped.
fn synth_inst_from_tlas_leaf(leaf: TlasInstanceLeaf) -> RkpInstance {
    var inst: RkpInstance;
    inst.world = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    inst.asset_id = leaf.asset_id;
    inst.material_id = leaf.material_id;
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

// Phase 7 Session 4b — stack-based TLAS traversal. Replaces the
// previous `for oi in 0..num_objects { instances[oi] }` iteration.
// Empty TLAS (`tlas_node_count == 0`) returns transmittance unchanged
// (no shadow casters this frame — sun shine-through is correct).
//
// `num_objects` is retained in the signature for caller compatibility
// but unused here; the BVH knows its own size.
fn trace_shadow_ray(
    world_origin: vec3<f32>,
    world_dir: vec3<f32>,
    num_objects: u32,
    max_steps: u32,
    max_world_dist: f32,
) -> f32 {
    var transmittance = 1.0;
    _ = num_objects;
    if march_params.tlas_node_count == 0u { return transmittance; }

    let safe_world_dir = vec3<f32>(
        select(world_dir.x, select(-1e-10, 1e-10, world_dir.x >= 0.0), abs(world_dir.x) < 1e-10),
        select(world_dir.y, select(-1e-10, 1e-10, world_dir.y >= 0.0), abs(world_dir.y) < 1e-10),
        select(world_dir.z, select(-1e-10, 1e-10, world_dir.z >= 0.0), abs(world_dir.z) < 1e-10),
    );
    let inv_world_dir = 1.0 / safe_world_dir;

    var stack: array<u32, TLAS_STACK_DEPTH>;
    stack[0] = 0u;
    var top: u32 = 1u;

    loop {
        if top == 0u { break; }
        top = top - 1u;
        let node = tlas_nodes[stack[top]];

        // World AABB cull. Skip subtrees whose AABB the ray misses
        // OR that lie entirely past the shadow ray's max distance.
        let t_range = intersect_aabb(world_origin, inv_world_dir, node.aabb_min, node.aabb_max);
        if t_range.x > t_range.y { continue; }
        if t_range.x > max_world_dist { continue; }

        if (node.left_or_leaf & TLAS_NODE_LEAF_BIT) != 0u {
            // Leaf — dispatch the per-instance descent. Host vs user-
            // shader distinguishes by `instance_index` sentinel.
            let leaf_idx = node.left_or_leaf & 0x7FFFFFFFu;
            let leaf = tlas_leaves[leaf_idx];
            var inst: RkpInstance;
            var asset: RkpAsset;
            if leaf.instance_index != TLAS_LEAF_USER_SHADER {
                inst = instances[leaf.instance_index];
                asset = assets[inst.asset_id];
            } else {
                inst = synth_inst_from_tlas_leaf(leaf);
                asset = assets[leaf.asset_id];
            }
            transmittance = shadow_step_one_instance(
                inst, asset, world_origin, world_dir, max_steps, max_world_dist, transmittance,
            );
            if transmittance < 0.01 { return 0.0; }
        } else {
            // Internal — push children. Stack-overflow guard: a
            // perfectly-balanced binary tree with 2^32 leaves is 32
            // deep, which is the stack capacity. Real TLASes are far
            // shallower; the guard is for pathological inputs.
            if top + 1u < TLAS_STACK_DEPTH {
                stack[top] = node.left_or_leaf;
                top = top + 1u;
                stack[top] = node.right_or_count;
                top = top + 1u;
            }
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

            // Phase 8 S5 — directional lights are served by the
            // shadow-map sample in the shade pass. Skip the
            // expensive per-pixel BVH descent here. Shade reads
            // `sample_shadow_map(world_pos)` for those lights;
            // `shadow_values[shadow_idx]` is written to 1.0 as an
            // unused-but-defined sentinel and `shadow_idx`
            // advances so the per-light slot mapping stays in
            // sync with shade's reader (which advances the same
            // way for both branches).
            if light_type == 0u && march_params.shadow_map_enabled != 0u {
                shadow_values[shadow_idx] = 1.0;
                shadow_idx++;
                continue;
            }

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
