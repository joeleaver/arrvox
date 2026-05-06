// Splat-resolve compute pass — derives the rest of the G-buffer from
// the visibility-buffer triplet (position, pick, leaf_slot) the splat
// raster pass writes.
//
// Per pixel:
//   leaf_slot == 0  →  miss. Write zeros to normal/material/glass —
//                      matching what `octree_march` writes for non-hits
//                      so downstream passes (shade, SSAO, …) see a
//                      consistent format regardless of which primary
//                      path ran.
//   leaf_slot != 0  →  hit. Look up leaf_attr_pool[leaf_slot] (normal +
//                      material packing), color_pool[leaf_slot]
//                      (per-voxel colour, 0 ⇒ "use material base"),
//                      and instances[pick].world (for the world-space
//                      normal). Build the same packed values the
//                      compute march would write for an opaque hit on
//                      the same leaf.
//
// Workgroup size matches the gbuffer's tile granularity (8×8) used
// elsewhere in the codebase. One dispatch covers the whole viewport.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::LeafAttr;
import package::lib::types::RkpInstance;

// Group 0 — per-VR textures: visibility-buffer reads + storage writes
// for the rest of the G-buffer.
@group(0) @binding(0) var leaf_slot_in:  texture_2d<u32>;
@group(0) @binding(1) var pick_in:       texture_2d<u32>;
@group(0) @binding(2) var normal_out:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var material_out:  texture_storage_2d<rg32uint,    write>;
@group(0) @binding(4) var glass_out:     texture_storage_2d<rg32uint,    write>;

// Group 1 — scene-wide buffers that drive the lookup. `instances` is
// the same `objects_buffer` the rest of the renderer reads; `pick` is
// the per-instance object_id, which equals the instance's index here
// (set in `engine::scene_gpu::update_scene_gpu`).
@group(1) @binding(0) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(1) @binding(1) var<storage, read> color_pool:     array<u32>;
@group(1) @binding(2) var<storage, read> instances:      array<RkpInstance>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(leaf_slot_in);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let leaf_slot = textureLoad(leaf_slot_in, coord, 0).x;
    if (leaf_slot == 0u) {
        textureStore(normal_out,   coord, vec4<f32>(0.0));
        textureStore(material_out, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(glass_out,    coord, vec4<u32>(0u, 0u, 0u, 0u));
        return;
    }

    let pick = textureLoad(pick_in, coord, 0).x;

    // Decode the prefiltered normal and rotate it into the instance's
    // world frame. `instances[pick].world`'s upper-left 3×3 is the
    // rotation+scale; for affine transforms with non-uniform scale the
    // normalize() at the end handles the resulting vector.
    let attr = leaf_attr_pool[leaf_slot];
    let n_local = unpack_oct_normal(attr.normal_oct);
    let world = instances[pick].world;
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

    // Per-voxel colour packed as RGBA8 in `color_pool`. 0 ⇒ "no
    // override; use material base_color" — the same convention the
    // shade pass already handles.
    let c_packed = color_pool[leaf_slot];
    let cr8 = c_packed & 0xFFu;
    let cg8 = (c_packed >> 8u) & 0xFFu;
    let cb8 = (c_packed >> 16u) & 0xFFu;
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);
    let packed_g = (blend8 & 0xFFu) | (color_rgb565 << 16u);

    // gbuf_normal alpha = `accum_alpha` in the march; opaque splat hit
    // ⇒ 1.0.
    textureStore(normal_out,   coord, vec4<f32>(n_world, 1.0));
    textureStore(material_out, coord, vec4<u32>(packed_r, packed_g, 0u, 0u));
    // Splats don't write glass — transparency wasn't in scope for the
    // Phase B-2 prototype. Always zero.
    textureStore(glass_out,    coord, vec4<u32>(0u, 0u, 0u, 0u));
}
