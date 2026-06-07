// Surface-mesh rasterizer — vertex + fragment entry points.
//
// Visibility-buffer contract: the raster writes (position, pick,
// leaf_slot) plus a rest-pos target; the `mesh_resolve` compute pass
// fills in the rest of the G-buffer (normal / material / glass)
// per-pixel via the `leaf_attr_pool` indirection.
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
// Bind-group shape is shared with the other mesh paths (mesh shadow,
// mesh LOD select, mesh glass, user-shader mesh) — the `mesh g0` /
// `mesh g1` bind groups built from `MeshInstanceLayouts` drive every
// per-instance raster. `bone_matrices` / `bone_dual_quats` are read
// only when the per-instance `skinning_mode` selects LBS or DQS.

import package::lib::types::CameraUniforms;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::ArvxInstance;
import package::lib::types::ArvxAsset;
import package::lib::types::OverlayEntry;
import package::lib::types::OctreeResult;
import package::lib::types::is_glass;
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

// Per-instance uniform — 96 B, mirror of `MeshInstanceUniform` on
// the CPU side. Skinning fields drive the VS branch — see
// `SKINNING_MODE_NONE` below for the rest-pose sentinel.
//
// `grid_origin` is the asset's voxel-grid origin in mesh-frame —
// the same value the surface-mesh extractor added to every cell's
// centroid when baking `local_pos`. Bone matrices operate on
// **grid-frame** positions (origin at octree corner) per
// `animation::tick`'s `N · pose · N⁻¹` conjugation. The VS subtracts
// grid_origin from `local_pos` before bones and adds it back after,
// so bones see the same frame they were trained against.
struct MeshInstance {
    world: mat4x4<f32>,
    grid_origin: vec3<f32>,
    object_id: u32,
    bone_offset_lbs: u32,
    bone_offset_dqs: u32,
    skinning_mode: u32,
}

// Sentinel matching `mesh_instance::SKINNING_MODE_NONE`. Skips
// skinning regardless of per-vertex weights — used for rigid
// instances and for skinned assets without a live `Skeleton` (no
// bone matrices uploaded).
const SKINNING_MODE_NONE: u32 = 0xFFFFFFFFu;
const SKINNING_MODE_LBS: u32 = 0u;
const SKINNING_MODE_DQS: u32 = 1u;

// Dual-quaternion form for DQS — 32 B per bone. `r` is the rotation
// (real) part, `d` is the translation-encoded (dual) part. Matches
// the CPU-side `DualQuat` in `scene_sync.rs`.
struct DualQuat {
    r: vec4<f32>,
    d: vec4<f32>,
}

// `bone_matrices` matches the buffer `BoneMatrixAllocator` writes —
// one full `mat4x4<f32>` (64 B) per bone. The trailing row is always
// `(0,0,0,1)` for affine bones; reading the bottom row as part of
// the matrix is harmless. (An earlier draft of this shader declared
// `BoneMatrix { row0, row1, row2 }` at 48 B and produced "stretched
// all over" geometry — every other bone read landed mid-matrix
// because the actual on-GPU stride is 64.)
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> bone_dual_quats: array<DualQuat>;

@group(1) @binding(0) var<uniform> instance: MeshInstance;

// Glass-classification bindings — see `mesh_glass.wesl` for the
// matching FS-side group. The primary FS reads the cell's material
// so it can `discard` glass fragments, matching the march's
// behavior of stepping past glass to find the opaque hit behind.
// Without this discard, glass meshes write opaque visibility into
// the G-buffer, the glass composite pass then gates them out
// (entry_dist == opaque_dist), and they render fully opaque.
//
// `color_pool_data` is unused at FS time but the imported
// `lib::leaf_attr` module's `fetch_leaf_color_for` references it
// at parse time (`use_stripping(false)` in build.rs).
@group(2) @binding(0) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(2) @binding(1) var<storage, read> materials: array<GpuMaterial>;
@group(2) @binding(2) var<storage, read> instances: array<ArvxInstance>;
@group(2) @binding(3) var<storage, read> instance_overlay: array<OverlayEntry>;
@group(2) @binding(4) var<storage, read> color_pool_data: array<u32>;
// Octree-lookup bindings — feed `lookup_cell_at` so the FS can
// per-pixel-resolve the surface cell at the fragment's mesh-frame
// rest_pos. Without per-pixel resolve, the discard tests the flat-
// interpolated `leaf_attr_id` (one cell per SN-cube vertex shared
// across many triangles), and carving one slot removes whole
// meshlet-scale regions. With per-pixel resolve, the carve hole
// follows voxel resolution. Matches the binding set mesh_resolve +
// mesh_glass use.
@group(2) @binding(6) var<storage, read> assets: array<ArvxAsset>;
@group(2) @binding(7) var<storage, read> octree_nodes: array<vec4<u32>>;
@group(2) @binding(8) var<storage, read> brick_pool: array<u32>;
// `brick_face_links` is referenced by `lib::octree::descend_proto_octree`
// as a free name even though we don't call that function — composer
// keeps the body (`use_stripping(false)`), so the binding has to be
// declared. naga DCE drops the unused reads.
@group(2) @binding(9) var<storage, read> brick_face_links: array<u32>;
// Per-instance sculpt-carve overlay (Phase A). Sorted ascending u32
// `leaf_attr_id`s removed from this instance; sliced by `ArvxInstance.
// sculpt_offset` + `sculpt_count`. `is_leaf_removed` (lib/leaf_attr)
// binary-searches this; the FS discards on hit so carved leaves
// don't appear in primary visibility. Bound at the next free slot
// after the octree-lookup bindings (6..9).
@group(2) @binding(10) var<storage, read> instance_sculpt: array<u32>;

// `octree_lookup` calls `bucket_depth` / `bucket_lod_exit` by free
// name. No stats binding here — declare no-op stubs. The LOD branch
// is gated behind `lod_enabled != 0` AND `phase == PHASE_MARCH`,
// and we pass `lod_enabled = 0` to lookup_cell_at so it never fires.
fn bucket_depth(phase: u32, level: u32) {}
fn bucket_lod_exit(level: u32) {}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) leaf_attr_id: u32,
    // Pre-skin, mesh-frame local position. Interpolates barycentrically
    // across the triangle so the FS gets the per-pixel rest-pose
    // surface point. `mesh_resolve` uses this to descend the asset's
    // octree per pixel and recover the actual cell — fixes the
    // chunky-per-triangle look caused by `@interpolate(flat) leaf_attr_id`.
    // Same value for skinned and unskinned (skinning happens AFTER this
    // is captured, so the rest pose is preserved across the wire).
    @location(2) local_rest_pos: vec3<f32>,
    // Pre-skin per-vertex local normal (unpacked from `normal_oct`).
    // Used by the FS as `hint_normal_oc` for `lookup_cell_at`'s
    // ½-voxel-into-solid snap — same trick mesh_glass / mesh_resolve
    // use to land the descent on the correct surface cell (SN-vertices
    // sit between cells; the snap nudges back inside).
    @location(3) local_normal: vec3<f32>,
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

// Antipodality-correct DQS blend over 4 influences. Flipping each
// successive quat into the same hemisphere as the first prevents the
// rotation from cancelling at bones whose quaternions straddle the
// double cover.
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

// `normal_oct` (vertex location 1) is unpacked and forwarded as
// `local_normal` for the FS's per-pixel resolve (the ½-voxel snap
// in `lookup_cell_at`). `mesh_resolve` reads `LeafAttr.normal_oct`
// from the pool slot for the final shading normal; this per-vertex
// value is purely a hint to land the cell descent on the right side
// of the surface.
@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
    @location(1) normal_oct: u32,
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
        // Bridge mesh-frame → grid-frame, skin, bridge back.
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

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.leaf_attr_id = leaf_attr_id;
    // Pre-skin local mesh-frame position — same value the surface-net
    // extractor wrote for this vertex. Interpolated per-fragment, this
    // recovers the rest-pose surface point even for skinned instances.
    out.local_rest_pos = local_pos;
    // Per-vertex local-frame normal as the FS's `hint_normal_oc`.
    // Interpolated as a vec3 (NOT flat) — slight smoothing across the
    // triangle is fine since the snap is half-voxel and the normal
    // only needs to point roughly outward.
    out.local_normal = unpack_oct_normal(normal_oct);
    return out;
}

// Per-pixel surface-cell lookup. Lifted byte-for-byte from
// `mesh_glass.wesl::lookup_cell_at`, which lifted it from
// `mesh_resolve.wesl`. Returns the `leaf_attr_pool` slot of the
// surface cell at `oc_pos` (descended through the octree + brick
// pool), or `0` ("no slot") if the descent lands on EMPTY /
// INTERIOR or misses the brick neighborhood. Caller falls back to
// the flat-interpolated `leaf_attr_id` on a `0` return.
//
// See `mesh_resolve.wesl` for the full rationale on the ½-voxel
// solid snap and the 3×3×3 brick-cell search — both compensate for
// SN-vertices living between cells (not on them) and for
// FP/skinning slop in the rasterized rest_pos.
fn lookup_cell_at(asset: ArvxAsset, oc_pos: vec3<f32>, hint_normal_oc: vec3<f32>) -> u32 {
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
        let bid = slot_brick_id(r.slot);
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
                    let c = brick_pool[bid * BRICK_CELLS + flat];
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

// Five-attachment fragment output — visibility-buffer triplet plus
// rest_pos for per-pixel cell descent in `mesh_resolve`, plus
// vis_normal for the smooth shading normal:
//   position   (Rgba32Float, 16 B): world.xyz + hit_distance
//   pick       (R32Uint,      4 B): instance.object_id
//   leaf_slot  (R32Uint,      4 B): leaf_attr_id (provoking vertex's
//                                   cell — used as a fallback /
//                                   miss-path hint)
//   rest_pos   (Rgba32Float, 16 B): local mesh-frame rest position
//                                   (xyz) + valid-bit (.w = 1)
//   vis_normal (Rgba16Float,  8 B): object-local INTERPOLATED per-vertex
//                                   normal (xyz, normalized) + valid-bit
//                                   (.w = 1). `mesh_resolve` rotates this
//                                   into world space and uses it as the
//                                   shading normal — smooth across the
//                                   triangle, unlike the per-cell
//                                   `LeafAttr.normal_oct` which facets.
struct FsOut {
    @location(0) position:   vec4<f32>,
    @location(1) pick:       u32,
    @location(2) leaf_slot:  u32,
    @location(3) rest_pos:   vec4<f32>,
    @location(4) vis_normal: vec4<f32>,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    // Glass discard: step past transparent fragments so the opaque
    // hit captured in `gbuf_position` is the surface BEHIND the
    // glass (matching the march). The glass passes
    // (`mesh_glass.wesl`) capture entry/exit on the same triangles
    // separately and pack into `gbuf_glass` for the composite.
    let inst = instances[instance.object_id];
    // Sculpt-carve discard (Phase A) — per-pixel resolved.
    //
    // The flat-interpolated `in.leaf_attr_id` is shared across many
    // triangles (each SN-vertex picks ONE cell's slot, but multiple
    // triangles whose provoking vertex sits at that SN-vertex all
    // see the same id). Testing the discard against it removes
    // whole meshlet-scale regions per carved cell — the user's
    // first stamp produced a sharp angular bird-shape, not a divot.
    //
    // Per-pixel resolve via `lookup_cell_at` descends the octree at
    // the fragment's rest_pos to recover the actual surface cell.
    // Test that against the sculpt set; fall back to the flat id
    // when the descent misses (rare — only the brick neighborhood
    // boundary, where it doesn't matter for a sphere-shaped brush).
    //
    // Short-circuit: `is_leaf_removed` returns false in O(1) when
    // the instance has nothing carved, so the resolve is only paid
    // by instances under active sculpt.
    if inst.sculpt_count != 0u {
        let asset = assets[inst.asset_id];
        let oc_pos = in.local_rest_pos - asset.grid_origin;
        let resolved_slot = lookup_cell_at(asset, oc_pos, in.local_normal);
        let test_slot = select(in.leaf_attr_id, resolved_slot, resolved_slot != 0u);
        if is_leaf_removed(inst, test_slot) {
            discard;
        }
    }
    let attr = fetch_leaf_attr_for(inst, in.leaf_attr_id);
    let mat_id = attr.material_packed & 0xFFFFu;
    // Bounds-check the material lookup. WGSL OOB array reads return
    // a zero `GpuMaterial` whose `opacity` is below the glass
    // threshold, which would cause the fragment to discard as if it
    // were glass — the pixel falls through to the dark background and
    // shows up as a thin black ring at painted-region boundaries on
    // assets whose leaves point at material slots outside the
    // project's loaded palette. Clamp OOB ids to slot 0 (default
    // material) so those fragments stay in the opaque path.
    let mat_id_safe = select(
        mat_id, 0u, mat_id >= arrayLength(&materials),
    );
    if is_glass(materials[mat_id_safe].opacity) {
        discard;
    }

    let hit_distance = length(in.world_pos - camera.position.xyz);
    var out: FsOut;
    out.position  = vec4<f32>(in.world_pos, hit_distance);
    out.pick      = instance.object_id;
    out.leaf_slot = in.leaf_attr_id;
    // .w = 1 marks this pixel as "mesh wrote rest_pos" — mesh_resolve
    // gates the per-pixel descent on this.
    out.rest_pos  = vec4<f32>(in.local_rest_pos, 1.0);
    // Interpolated per-vertex object-local normal — the smooth shading
    // normal. `mesh_resolve` reads this, rotates it into world space,
    // and writes it to the shading normal target. Re-normalize because
    // barycentric interpolation across the triangle shortens the vector.
    // .w = 1 ⇒ "raster wrote a valid normal" (resolve falls back to the
    // per-leaf normal on a miss / zero pixel).
    out.vis_normal = vec4<f32>(normalize(in.local_normal), 1.0);
    return out;
}
