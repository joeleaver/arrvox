// Mesh-mode glass entry/exit raster.
//
// Two pipelines share this VS:
//   · `frag_front` — cull = Back, depth_compare = Less. Writes the
//     entry (front-face) packed data: oct-packed world normal, glass
//     material id, and bitcast f32 entry-distance into a single
//     Rgba32Uint target.
//   · `frag_back`  — cull = Front, depth_compare = Greater. Writes
//     the exit (back-face) world distance into a single R32Float
//     target.
//
// A `mesh_glass_combine` compute pass reads both, plus
// `gbuf_position.w` (opaque hit distance), and packs the final
// `gbuf_glass` (Rg32Uint) with the same layout as the march writes.
//
// Both FSs read `leaf_attr_pool[leaf_attr_id]` for the cell's normal +
// material, then `materials[material_id]` to gate `discard` on
// `opacity >= 0.99`. This mirrors the march's classification and lets
// the same shader run over every mesh instance — no per-instance
// glass filter is required for correctness, only for perf.

import package::lib::types::CameraUniforms;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::RkpInstance;
import package::lib::types::RkpAsset;
import package::lib::types::OverlayEntry;
import package::lib::types::OctreeResult;
import package::lib::oct_normal::pack_oct_normal;
import package::lib::oct_normal::unpack_oct_normal;
import package::lib::leaf_attr::fetch_leaf_attr_for;
import package::lib::leaf_attr::is_leaf_removed;
import package::lib::octree::octree_lookup;
import package::lib::octree::PHASE_MARCH;
import package::lib::octree_slot::OCTREE_EMPTY;
import package::lib::octree_slot::OCTREE_INTERIOR;
import package::lib::octree_slot::OCTREE_PAYLOAD_MASK;
import package::lib::octree_slot::slot_is_brick;
import package::lib::octree_slot::slot_brick_id;
import package::lib::brick::BRICK_DIM;
import package::lib::brick::BRICK_DIM_F;
import package::lib::brick::BRICK_CELLS;
import package::lib::brick::BRICK_CELL_EMPTY;
import package::lib::brick::BRICK_CELL_INTERIOR;

// Same shape as `mesh.wesl::MeshInstance` (96 B). Skinning fields are
// declared so the same bind-group layout drives this pipeline; the VS
// branch on `skinning_mode` mirrors `mesh.wesl`.
struct MeshInstance {
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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshInstance;

// Scene-global glass lookup tables. Same buffers `splat_resolve` and
// the march bind, just exposed as a separate group here so the mesh
// pipeline's existing g0/g1 layouts stay unchanged.
//
// `instances` + `instance_overlay` are required for overlay-aware
// material classification — when the user paints a material onto a
// specific instance (Phase 3 paint), the new material is recorded in
// `instance_overlay`, NOT written back to the shared
// `leaf_attr_pool`. The march and shadow trace use
// `fetch_leaf_attr_for(inst, slot)` which checks the overlay slice
// first; the glass FS does the same so painted-glass classifies
// correctly.
//
// `color_pool_data` is unused at fragment time but the imported
// `lib::leaf_attr` module's `fetch_leaf_color_for` references the
// name; the WESL composer requires every name pulled in by the
// import graph to resolve, even when DCE will drop the read in the
// final SPIR-V (`use_stripping(false)` in `build.rs`).
@group(2) @binding(0) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read> instances: array<RkpInstance>;
@group(2) @binding(3) var<storage, read> instance_overlay: array<OverlayEntry>;
@group(2) @binding(4) var<storage, read> color_pool_data: array<u32>;
// Octree-lookup bindings — feed `lookup_cell_at` so the front FS can
// resolve the actual surface cell at the fragment's mesh-frame
// rest_pos (instead of barycentric-blending per-vertex normals).
// Matches the splat_resolve binding set so we get the same
// per-pixel quality the primary mesh path achieves via
// `splat_resolve`.
@group(2) @binding(6) var<storage, read> assets: array<RkpAsset>;
@group(2) @binding(7) var<storage, read> octree_nodes: array<vec4<u32>>;
@group(2) @binding(8) var<storage, read> brick_pool: array<u32>;
// `brick_face_links` referenced by `lib::octree::descend_proto_octree`
// as a free name; we don't call that function but the composer
// (`use_stripping(false)`) pulls it in, so the binding has to be
// declared. naga DCE drops the unused reads from compiled SPIR-V.
@group(2) @binding(9) var<storage, read> brick_face_links: array<u32>;
// Per-instance sculpt-carve overlay (Phase A). `is_leaf_removed`
// discards glass fragments belonging to carved leaves so they
// neither block primary visibility nor write into `gbuf_glass`.
@group(2) @binding(10) var<storage, read> instance_sculpt: array<u32>;

// `octree_lookup` calls `bucket_depth` / `bucket_lod_exit` by free
// name; this pass has no stats binding so the stubs stay no-ops. The
// LOD branch is gated on `lod_enabled != 0` AND `phase == PHASE_MARCH`
// — we pass `lod_enabled = 0` so it never fires either.
fn bucket_depth(phase: u32, level: u32) {}
fn bucket_lod_exit(level: u32) {}

struct GlassFsParams {
    /// Opacity gate. A material with `opacity >= opacity_threshold`
    /// is classified as opaque (FS discards). The march uses 0.99;
    /// `RKP_MESH_GLASS_DEBUG_FORCE=2` raises this above 1.0 so every
    /// fragment classifies as glass — bisects "leaf lookup wrong"
    /// vs "discard threshold wrong" vs "FS never runs".
    opacity_threshold: f32,
    _pad0: u32, _pad1: u32, _pad2: u32,
}
@group(2) @binding(5) var<uniform> fs_params: GlassFsParams;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) leaf_attr_id: u32,
    /// Per-vertex world-space normal, decoded from the VBO's
    /// `normal_oct` attribute. Used as a HINT for the per-pixel
    /// resolve — the FS snaps `local_rest_pos` ½ voxel along
    /// `-local_normal` to land safely inside a surface cell, then
    /// reads that cell's prefiltered `LeafAttr.normal_oct` for the
    /// final normal. This matches the per-pixel quality the primary
    /// mesh path gets from `splat_resolve`.
    @location(2) world_normal: vec3<f32>,
    /// Pre-skin, mesh-frame local position. Same value the surface-
    /// nets extractor wrote per vertex; barycentric-interpolated to
    /// the fragment surface point. Fed into `lookup_cell_at`.
    @location(3) local_rest_pos: vec3<f32>,
    /// Pre-skin per-vertex local normal (pre-world-rotation). Used as
    /// the `hint_normal_oc` for `lookup_cell_at`'s ½-voxel snap.
    @location(4) local_normal: vec3<f32>,
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
    @location(1) normal_oct: u32,
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
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Per-vertex normal — decoded from the VBO and rotated into
    // world space. For unskinned / rigid-affine instances this is
    // the upper-3x3 of `instance.world` applied to a unit vector,
    // matching the approximation `octree_march` uses on line 1361.
    // Skinned normals would require applying the bone-blended
    // rotation to the normal too; deferred — glass meshes are
    // typically rigid, and skinned glass would show as flat-
    // interpolated rest-pose normals (still correct on average,
    // wrong at extreme bends).
    let local_normal = unpack_oct_normal(normal_oct);
    let world_normal_v = (instance.world * vec4<f32>(local_normal, 0.0)).xyz;

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.leaf_attr_id = leaf_attr_id;
    out.world_normal = world_normal_v;
    out.local_rest_pos = local_pos;
    out.local_normal = local_normal;
    return out;
}

// Lifted byte-for-byte from `splat_resolve.wesl::lookup_cell_at`.
// Walks the asset's octree at a known oc-space surface point and
// returns the `leaf_attr_pool` slot of the surface cell visible at
// this fragment, or `0` ("no slot") if no occupied cell is found in
// the search neighborhood. The caller falls back to the
// flat-interpolated `leaf_attr_id` (the provoking vertex's cell) on
// a `0` return.
//
// See the original for the full rationale on the ½-voxel solid snap
// and the 3×3×3 brick-cell search.
fn lookup_cell_at(asset: RkpAsset, oc_pos: vec3<f32>, hint_normal_oc: vec3<f32>) -> u32 {
    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    let snapped = oc_pos - hint_normal_oc * (vs * 0.5);
    let pos = clamp(
        snapped,
        vec3<f32>(vs * 0.01),
        vec3<f32>(extent - vs * 0.01),
    );

    let r: OctreeResult = octree_lookup(
        asset.octree_root, asset.octree_depth, extent, pos,
        vec3<f32>(0.0), vec3<f32>(1.0),
        PHASE_MARCH, 0u, 0.0, 1.0, 0.0,
    );

    if r.slot == OCTREE_EMPTY { return 0u; }
    if r.slot == OCTREE_INTERIOR { return 0u; }

    if slot_is_brick(r.slot) {
        let brick_id = slot_brick_id(r.slot);
        let cell_size = (r.cell_half * 2.0) / BRICK_DIM_F;
        let brick_origin = r.cell_center - vec3<f32>(r.cell_half);
        let local = (pos - brick_origin) / cell_size;
        let center_cell = clamp(
            vec3<i32>(floor(local)),
            vec3<i32>(0),
            vec3<i32>(i32(BRICK_DIM) - 1),
        );

        var best_slot: u32 = 0u;
        var best_dist_sq: f32 = 1.0e30;
        for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    let cell = center_cell + vec3<i32>(dx, dy, dz);
                    if any(cell < vec3<i32>(0)) || any(cell > vec3<i32>(i32(BRICK_DIM) - 1)) {
                        continue;
                    }
                    let flat = u32(cell.x)
                        + u32(cell.y) * BRICK_DIM
                        + u32(cell.z) * BRICK_DIM * BRICK_DIM;
                    let c = brick_pool[brick_id * BRICK_CELLS + flat];
                    if c == BRICK_CELL_EMPTY || c == BRICK_CELL_INTERIOR {
                        continue;
                    }
                    let cell_center_oc = brick_origin
                        + (vec3<f32>(cell) + vec3<f32>(0.5)) * cell_size;
                    let d = pos - cell_center_oc;
                    let d_sq = dot(d, d);
                    if d_sq < best_dist_sq {
                        best_dist_sq = d_sq;
                        best_slot = c;
                    }
                }
            }
        }
        return best_slot;
    }

    return r.slot & OCTREE_PAYLOAD_MASK;
}

// Resolved glass-classification result. `material_id == 0xFFFFFFFFu`
// means "not glass; caller should `discard`". Otherwise the cell's
// resolved attr (overlay-aware) and material id are the inputs the
// front FS packs into `glass_entry_packed`.
struct GlassClass {
    material_id: u32,
    attr: LeafAttr,
}

// Returns the resolved glass classification for the cell at
// `leaf_attr_id` on the current draw's instance. Two-step lookup:
//   1. `fetch_leaf_attr_for(inst, slot)` checks the per-instance paint
//      overlay first, falling back to `leaf_attr_pool[slot]` — the
//      same path the march and shadow trace take, so painted-glass
//      and asset-pool-glass both classify identically.
//   2. `materials[material_primary].opacity < 0.99` ⇒ glass;
//      otherwise return the discard sentinel.
// `instance.object_id` doubles as the instance-array index (mesh +
// splat paths both write the slot index into `object_id`, and
// `splat_resolve` indexes `instances[]` with the same value).
fn classify_glass(leaf_attr_id: u32) -> GlassClass {
    let inst = instances[instance.object_id];
    let attr = fetch_leaf_attr_for(inst, leaf_attr_id);
    let mat_id = attr.material_packed & 0xFFFFu;
    // OOB-safe lookup — same rationale as `mesh.wesl`. Without this,
    // virgin pool slots whose material ids are outside the loaded
    // palette would classify as glass (zero `GpuMaterial.opacity`)
    // and capture as glass entries here even though the underlying
    // material is implicitly the project default.
    let mat_id_safe = select(
        mat_id, 0u, mat_id >= arrayLength(&materials),
    );
    let mat = materials[mat_id_safe];
    var out: GlassClass;
    out.attr = attr;
    if mat.opacity >= fs_params.opacity_threshold {
        out.material_id = 0xFFFFFFFFu;
    } else {
        out.material_id = mat_id_safe;
    }
    return out;
}

// `glass_entry_packed` channel layout — kept in lockstep with
// `mesh_glass_combine.wesl` and the `glass_entry_packed_view` binding
// in `viewport_renderer.rs`.
struct FsFront {
    @location(0) packed: vec4<u32>,
}

@fragment
fn frag_front(in: VsOut) -> FsFront {
    let cls = classify_glass(in.leaf_attr_id);
    if cls.material_id == 0xFFFFFFFFu { discard; }

    // Per-pixel resolved normal — same path `splat_resolve` takes
    // for the primary mesh G-buffer. Look up the actual surface
    // cell at this fragment's mesh-frame rest position, fall back
    // to the per-vertex interpolated normal if the descent fails
    // (cell outside the brick neighborhood, etc.).
    let inst = instances[instance.object_id];
    if is_leaf_removed(inst, in.leaf_attr_id) { discard; }
    let asset = assets[inst.asset_id];
    let oc_pos = in.local_rest_pos - asset.grid_origin;
    let resolved_slot = lookup_cell_at(asset, oc_pos, in.local_normal);
    var local_n: vec3<f32>;
    if resolved_slot != 0u {
        let resolved_attr = leaf_attr_pool[resolved_slot];
        local_n = unpack_oct_normal(resolved_attr.normal_oct);
    } else {
        local_n = normalize(in.local_normal);
    }
    // Rotate into world space — same approximation the march uses on
    // line 1361. Non-uniform scale would need the inverse-transpose;
    // deferred until a non-uniform glass instance shows up.
    let world_normal = normalize((instance.world * vec4<f32>(local_n, 0.0)).xyz);

    let entry_dist = length(in.world_pos - camera.position.xyz);

    var out: FsFront;
    // R: oct-packed world normal.
    // G: glass material_id (lower 16 bits; upper 16 are filled with
    //    thickness_mm by `mesh_glass_combine`).
    // B: bitcast<u32>(entry_dist) — the combine pass reverses this
    //    with `bitcast<f32>` to recover the world-space distance.
    // A: 0 (reserved).
    out.packed = vec4<u32>(
        pack_oct_normal(world_normal),
        cls.material_id & 0xFFFFu,
        bitcast<u32>(entry_dist),
        0u,
    );
    return out;
}

struct FsBack {
    @location(0) exit_dist: f32,
}

@fragment
fn frag_back(in: VsOut) -> FsBack {
    let cls = classify_glass(in.leaf_attr_id);
    if cls.material_id == 0xFFFFFFFFu { discard; }

    let inst = instances[instance.object_id];
    if is_leaf_removed(inst, in.leaf_attr_id) { discard; }

    var out: FsBack;
    out.exit_dist = length(in.world_pos - camera.position.xyz);
    return out;
}
