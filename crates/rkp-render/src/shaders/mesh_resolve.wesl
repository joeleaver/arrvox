// Mesh-resolve compute pass — derives the rest of the G-buffer from
// the visibility-buffer attachments the mesh raster writes.
//
// The mesh raster writes (position, pick, leaf_slot, rest_pos). The
// `rest_pos` target carries the pre-skin mesh-frame position
// interpolated barycentrically across each triangle, so the FS-
// fragment value is the per-pixel rest-pose surface point. Resolve
// walks the asset's octree at that point to recover the actual
// per-pixel cell — fixes the chunky-per-triangle look caused by
// `@interpolate(flat) leaf_attr_id` in `mesh.wesl`.
//
// Per pixel:
//   leaf_slot_in == 0  →  miss. Write zeros to normal/material/glass.
//   rest_pos.w == 1    →  Descend `assets[inst.asset_id]`'s octree at
//                         `rest_pos.xyz - asset.grid_origin` to find
//                         the per-pixel slot. If the descent lands on
//                         EMPTY/INTERIOR or out-of-bounds (e.g. mesh
//                         interpolation drift past the surface), fall
//                         back to the visibility-buffer `leaf_slot_in`
//                         so we never display a garbage cell.
//   else               →  fallback. Use `leaf_slot_in` directly.
//
// Workgroup size matches the gbuffer's tile granularity (8×8). One
// dispatch covers the whole viewport.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::LeafAttr;
import package::lib::types::RkpInstance;
import package::lib::types::RkpAsset;
import package::lib::types::OctreeResult;
import package::lib::types::OverlayEntry;
import package::lib::leaf_attr::fetch_leaf_attr_for;
import package::lib::leaf_attr::fetch_leaf_color_for;
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

// Group 0 — per-VR textures: visibility-buffer reads + storage writes
// for the rest of the G-buffer. `rest_pos_in` is the new mesh-path
// per-pixel rest-pose position; .w gates the per-pixel descent.
@group(0) @binding(0) var leaf_slot_in:  texture_2d<u32>;
@group(0) @binding(1) var pick_in:       texture_2d<u32>;
@group(0) @binding(2) var normal_out:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var material_out:  texture_storage_2d<rg32uint,    write>;
@group(0) @binding(4) var glass_out:     texture_storage_2d<rg32uint,    write>;
@group(0) @binding(5) var rest_pos_in:   texture_2d<f32>;

// Group 1 — scene-wide buffers driving the lookup. The first three
// entries match the existing layout (`leaf_attr_pool`, `color_pool`,
// `instances`); `assets`/`octree_nodes`/`brick_pool` are added for the
// per-pixel descent — same buffers `octree_march` binds, so the lookup
// resolves identically to a degenerate march (a single descent at a
// known surface point, no ray walk).
// `lib::leaf_attr` references `leaf_attr_pool`, `color_pool_data`, and
// `instance_overlay` by free name (see its module docstring). The
// names + binding shapes below match those expectations so the
// inlined `fetch_leaf_attr_for` / `fetch_leaf_color_for` resolve
// under `ManglerKind::None`.
@group(1) @binding(0) var<storage, read> leaf_attr_pool:  array<LeafAttr>;
@group(1) @binding(1) var<storage, read> color_pool_data: array<u32>;
@group(1) @binding(2) var<storage, read> instances:       array<RkpInstance>;
@group(1) @binding(3) var<storage, read> assets:          array<RkpAsset>;
@group(1) @binding(4) var<storage, read> octree_nodes:    array<vec4<u32>>;
@group(1) @binding(5) var<storage, read> brick_pool:      array<u32>;
// `brick_face_links` is referenced by `lib::octree::descend_proto_octree`
// as a free name. We don't call that function — `lookup_cell_at` only
// uses `octree_lookup` — but the WESL composer pulls the whole
// `lib::octree` module into the emitted WGSL, so the symbol has to be
// in scope to parse. Bind it to the same scene buffer the march uses;
// naga's DCE drops the unused reads from the resolve's compiled SPIR-V.
@group(1) @binding(6) var<storage, read> brick_face_links: array<u32>;
// Per-instance paint overlay — sorted-by-leaf_slot slice carrying the
// painted attr + color overrides. `fetch_leaf_attr_for` /
// `fetch_leaf_color_for` consult it via binary search before falling
// through to the asset's pool. Without this binding mesh-mode paint
// stamps are invisible (the resolve reads the unpainted base).
@group(1) @binding(7) var<storage, read> instance_overlay: array<OverlayEntry>;
// Per-instance sculpt-carve overlay (Phase A). Free-name reference
// pulled in by `lib::leaf_attr::is_leaf_removed` even though the
// resolve doesn't itself call it — `use_stripping(false)` requires
// every imported free name to resolve.
@group(1) @binding(8) var<storage, read> instance_sculpt: array<u32>;

// `octree_lookup` (in `lib::octree`) calls `bucket_depth` and
// `bucket_lod_exit` by free name. We have no stats binding here, so
// declare no-op stubs. The lib's LOD branch is gated behind
// `lod_enabled != 0` AND `phase == PHASE_MARCH` — we pass
// `lod_enabled = 0` so the LOD branch never fires either.
fn bucket_depth(phase: u32, level: u32) {}
fn bucket_lod_exit(level: u32) {}

// Walk the per-instance octree at a known oc-space surface point and
// return the leaf_attr slot for the surface cell visible at this
// fragment. Returns 0 ("no slot") if no occupied cell is found in the
// search neighborhood — caller falls back to the visibility-buffer
// hint in that case.
//
// Why pass `(vec3(0), vec3(1))` for ray origin/inv_dir: `octree_lookup`
// uses these for a per-node tight-AABB ray cull. We aren't tracing a
// ray, so disable that test by handing it a zero-extent fallback —
// every node's tight AABB passes, and we descend purely on `pos`.
//
// **Why we snap into the solid first:** two effects can push the raw
// `oc_pos` off the rest-surface cells, both fixed by stepping ½ voxel
// along `-hint_normal_oc` (i.e., into the solid):
//
//   * Surface-nets vertices live at edge-crossing centroids of 2×2×2
//     SN-cubes (`mesh_extract::build_cube_vertex`) — they're between
//     cells, not on cells. Linear interpolation across a triangle can
//     land `rest_pos` on the empty side of the surface.
//   * Skinned meshes blend bone weights per-vertex; the per-fragment
//     barycentric blend of rest vertex positions is an *approximate*
//     inverse-LBS, not exact. The error can shift `rest_pos` by 1+
//     voxels for triangles that span a bone-weight boundary, which
//     pushes the descent into an unrelated adjacent brick (the
//     wrong-color triangle patches you saw on CesiumMan's face).
//
// Stepping ½ voxel into the solid using the hint cell's outward
// normal puts us safely inside a surface cell of the rest octree
// (the shell is 1-thick, so going farther would risk hitting INTERIOR
// bulk and losing the per-cell color).
//
// **Why a 3×3×3 search after snapping:** even after the snap, FP
// rounding + cell-boundary alignment can put `pos` in an EMPTY corner
// of an otherwise-populated brick. Surface cells are always within
// ±1 voxel of the snapped position (otherwise they wouldn't have
// contributed to the SN-cube's vertex), so a 3×3×3 search finds the
// closest occupied cell. Worst-case 27 brick_pool reads per pixel;
// cheap relative to the rest of the frame.
fn lookup_cell_at(asset: RkpAsset, oc_pos: vec3<f32>, hint_normal_oc: vec3<f32>) -> u32 {
    let extent = bitcast<f32>(asset.octree_extent_bits);
    let vs = asset.voxel_size;
    // Snap into the solid by ½ voxel along `-hint_normal_oc`. See
    // fn-doc for the two effects this compensates for.
    let snapped = oc_pos - hint_normal_oc * (vs * 0.5);
    // Clamp to the octree's bounded interior so the descent's
    // `pos >= center` comparisons can't walk off the cell boundary
    // when the rasterizer's perspective interpolation lands a fragment
    // exactly on the asset's outer face. Same eps the march uses.
    let pos = clamp(
        snapped,
        vec3<f32>(vs * 0.01),
        vec3<f32>(extent - vs * 0.01),
    );

    let r: OctreeResult = octree_lookup(
        asset.octree_root, asset.octree_depth, extent, pos,
        // Disabled ray-AABB cull (zero ray, large inv_dir) — we are
        // doing a point lookup, not a ray walk.
        vec3<f32>(0.0), vec3<f32>(1.0),
        PHASE_MARCH, 0u, 0.0, 1.0, 0.0,
    );

    if r.slot == OCTREE_EMPTY { return 0u; }
    if r.slot == OCTREE_INTERIOR { return 0u; }

    if slot_is_brick(r.slot) {
        // Locate the cell within the brick using the same oc-space
        // math as the march. `r.cell_center` and `r.cell_half` describe
        // the brick's bounds.
        let brick_id = slot_brick_id(r.slot);
        let cell_size = (r.cell_half * 2.0) / BRICK_DIM_F;
        let brick_origin = r.cell_center - vec3<f32>(r.cell_half);
        let local = (pos - brick_origin) / cell_size;
        let center_cell = clamp(
            vec3<i32>(floor(local)),
            vec3<i32>(0),
            vec3<i32>(i32(BRICK_DIM) - 1),
        );

        // 3×3×3 search for the closest occupied cell — see fn-doc
        // "Why a 3×3×3 search inside the brick" for why a direct
        // floor-cell read isn't enough. Cells outside this brick are
        // skipped (no face-link chase) — fragments near a brick face
        // fall back to `hint_slot`, which is rare enough at typical
        // viewing distances that it's not worth chasing.
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
                    // Squared distance from pos to the cell's center —
                    // sufficient for ranking (no sqrt needed).
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

    // Plain leaf — single-voxel slot.
    return r.slot & OCTREE_PAYLOAD_MASK;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(leaf_slot_in);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let hint_slot = textureLoad(leaf_slot_in, coord, 0).x;
    if (hint_slot == 0u) {
        textureStore(normal_out,   coord, vec4<f32>(0.0));
        textureStore(material_out, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(glass_out,    coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let pick = textureLoad(pick_in, coord, 0).x;
    let inst = instances[pick];

    // Decode the hint cell's outward normal — used both as the
    // snap direction during per-pixel descent (fn-doc on
    // `lookup_cell_at` for the why) and as the world-space normal
    // for the G-buffer write below if the descent doesn't find a
    // better slot.
    let hint_attr = leaf_attr_pool[hint_slot];
    let hint_normal_oc = unpack_oct_normal(hint_attr.normal_oct);

    // Per-pixel cell selection. The mesh raster writes rest_pos.w = 1.
    // Walk the asset's octree at rest_pos to recover the cell that
    // actually contains this fragment. When the descent fails (mesh
    // interpolation drift past the surface), fall back to the
    // visibility-buffer hint slot.
    let rest = textureLoad(rest_pos_in, coord, 0);
    var leaf_slot = hint_slot;
    if (rest.w > 0.5) {
        let asset = assets[inst.asset_id];
        // Mesh-frame → oc-space: subtract grid_origin. The mesh raster
        // VS captures local_pos in mesh-frame (the surface-net
        // extractor's coordinate); the asset's octree lives in
        // oc-space (origin at the octree corner).
        let oc_pos = rest.xyz - asset.grid_origin;
        let resolved = lookup_cell_at(asset, oc_pos, hint_normal_oc);
        if (resolved != 0u) {
            leaf_slot = resolved;
        }
        // else: descent failed (mesh fragment landed past the surface
        // in EMPTY/INTERIOR space, or out of bounds) — keep `hint_slot`
        // so we never display a garbage cell.
    }

    // Decode the prefiltered normal and rotate it into the instance's
    // world frame. `instances[pick].world`'s upper-left 3×3 is the
    // rotation+scale; for affine transforms with non-uniform scale the
    // normalize() at the end handles the resulting vector.
    //
    // Overlay-aware fetch: paint stamps land in the per-instance
    // `instance_overlay` slice — the helpers binary-search it and
    // fall through to the asset's pool when the slot isn't painted.
    // Without this the mesh path renders the unpainted base color.
    //
    // Empty-slot guard: when the per-pixel descent picks a leaf whose
    // pool entry is all-zeros (`normal_oct == 0 && material_packed
    // == 0` — an uninitialized slot in the asset's allocated leaf-
    // attr range, can happen at brick boundaries / interior-cell
    // resolution edge cases), substitute the hint cell's data. The
    // hint slot comes from the triangle's provoking vertex, which the
    // mesh extractor binds to a known-populated surface cell, so its
    // pool entry is reliable. Without this guard, the resolved-bad
    // leaf renders with `unpack_oct_normal(0) == +Z`, giving sun
    // direct=0 and SSAO≈0 — manifesting as a black ring at the
    // brush boundary specifically because painted leaves bypass the
    // pool via the overlay path and unpainted ones don't.
    var attr = fetch_leaf_attr_for(inst, leaf_slot);
    if (attr.normal_oct == 0u && attr.material_packed == 0u) {
        attr = hint_attr;
    }
    let n_local = unpack_oct_normal(attr.normal_oct);
    let world = inst.world;
    let world_basis = mat3x3<f32>(world[0].xyz, world[1].xyz, world[2].xyz);
    let n_world = normalize(world_basis * n_local);

    // Material packing — must match `octree_march`'s opaque-hit path
    // (search for `let packed_r =` / `let packed_g =`):
    //   r: primary (lo16) | secondary (hi16)
    //   g: blend_8 (lo8)  | color_rgb565 (hi16)
    let primary   = attr.material_packed & 0xFFFFu;
    let secondary = (attr.material_packed >> 16u) & 0xFFFu;
    let blend4    = (attr.material_packed >> 28u) & 0xFu;
    let blend8    = (blend4 << 4u) | blend4;
    let packed_r  = (primary & 0xFFFFu) | ((secondary & 0xFFFFu) << 16u);

    // Per-voxel colour packed as RGBA8 in `color_pool_data`, or in
    // the per-instance overlay if the slot's been painted. The high
    // byte is the brush "intensity" (0 = "no override, use material
    // base_color"; 255 = "fully overrides base_color"; intermediate
    // values lerp the two so a soft-falloff brush edge fades into
    // the underlying material instead of producing a hard boundary
    // disc of painted RGB).
    let c_packed = fetch_leaf_color_for(inst, leaf_slot);
    let cr8 = c_packed & 0xFFu;
    let cg8 = (c_packed >> 8u) & 0xFFu;
    let cb8 = (c_packed >> 16u) & 0xFFu;
    let ci8 = (c_packed >> 24u) & 0xFFu;
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);
    // packed_g layout — must match `octree_march.wesl`'s opaque-hit
    // path and `rkp_shade.wesl`'s decode:
    //   bits  0- 7: blend8 (dual-material weight)
    //   bits  8-15: paint intensity (0..255)
    //   bits 16-31: color_rgb565
    let packed_g = (blend8 & 0xFFu)
                 | ((ci8 & 0xFFu) << 8u)
                 | (color_rgb565 << 16u);

    // gbuf_normal alpha = opaque hit ⇒ 1.0.
    textureStore(normal_out,   coord, vec4<f32>(n_world, 1.0));
    textureStore(material_out, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
    // Glass is handled by the dedicated front/back glass pass; the
    // primary mesh raster discards glass fragments so resolve never
    // sees them.
    textureStore(glass_out,    coord, vec4<u32>(0u, 0u, 0u, 0u));
}
