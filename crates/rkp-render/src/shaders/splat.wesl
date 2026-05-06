// Surface-splat rasterizer — vertex + fragment entry points.
//
// Rasterizes one quad per voxel. The quad is oriented to the prefiltered
// surface normal stored in `LeafAttr.normal_oct`, so adjacent voxel discs
// produce a smooth surface at any voxel resolution — silhouettes follow
// the actual surface, not the grid axes.
//
// Pipeline contract:
//   · Vertex buffer: one `SplatVertex` per voxel (Rust splat_pass::extract).
//     · @location(0) local_pos:    vec3<f32>   (cell center, OBJECT-LOCAL)
//     · @location(1) radius:       f32         (object-local disc radius)
//     · @location(2) leaf_attr_id: u32         (absolute leaf_attr_pool slot)
//   · Step mode: Instance. Draw 4 verts × N instances; @builtin(vertex_index)
//     selects the quad corner (triangle-strip).
//   · One draw per scene-instance per asset; the per-instance world matrix
//     comes from `instance` at @group(1) @binding(0). A separate draw +
//     `set_bind_group(1, ...)` is issued for each scene-instance.
//
// Fragment writes the SAME G-buffer the compute octree_march produces
// (position / normal / material / pick / glass / leaf_slot). Glass is
// always written zero — splats are alpha-1 with hard discard. Motion is
// not bound; TAA isn't dispatched in render_to today, so motion staleness
// has no observable effect.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::CameraUniforms;

// Per-instance uniform — one mat4x4 world transform plus the entity's
// `object_id` (written into the pick texture). 80 B, multiple of 16 so
// the WGSL uniform alignment rules are happy.
struct SplatInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(2) var<storage, read> materials: array<GpuMaterial>;
// Per-voxel colour overrides (packed RGBA8, u32 per leaf slot). Zero
// means "use material base_color". Same encoding as `color_pool` in
// `RkpScene` and the same downstream consumer (`rkp_shade` decodes
// `color_rgb565` from `gbuf_material.g` bits 16..32).
@group(0) @binding(3) var<storage, read> color_pool: array<u32>;

@group(1) @binding(0) var<uniform> instance: SplatInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) corner_uv: vec2<f32>,        // -1..1 across the disc
    @location(1) world_normal: vec3<f32>,     // prefiltered, flat across splat
    @location(2) world_pos: vec3<f32>,        // interpolated; for depth + dist
    @location(3) @interpolate(flat) leaf_attr_id: u32,
};

// Branch-free tangent-basis construction (Frisvad 2012). For unit `n`,
// returns a mat3 whose columns are (u, v, n) — orthonormal. Stable
// everywhere except n.z = -1 (handled with the explicit fallback).
fn frisvad_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    if (n.z < -0.999999) {
        return mat3x3<f32>(
            vec3<f32>(0.0, -1.0, 0.0),
            vec3<f32>(-1.0, 0.0, 0.0),
            n,
        );
    }
    let a = 1.0 / (1.0 + n.z);
    let b = -n.x * n.y * a;
    let u = vec3<f32>(1.0 - n.x * n.x * a, b, -n.x);
    let v = vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y);
    return mat3x3<f32>(u, v, n);
}

@vertex
fn vert_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) local_pos: vec3<f32>,
    @location(1) radius: f32,
    @location(2) leaf_attr_id: u32,
) -> VsOut {
    // Decode the prefiltered normal once per splat (vertex-shader work
    // amortizes across the 4 corners — same value for all of them).
    let attr = leaf_attr_pool[leaf_attr_id];
    let n_local = unpack_oct_normal(attr.normal_oct);

    // Build the disc in OBJECT-LOCAL space, then transform corners
    // through `instance.world`. This makes the disc inherit any scale
    // / non-uniform scale / rotation in the instance transform without
    // special-casing — a rotated parent rotates the disc, a scaled
    // parent scales the disc. The alternative ("transform the basis to
    // world, then size in world units") loses this property under
    // non-uniform scale.
    let basis_local = frisvad_tangent_basis(n_local);
    let local_u = basis_local[0] * radius;
    let local_v = basis_local[1] * radius;

    // World-space disc anchors.
    let world_pos = (instance.world * vec4<f32>(local_pos, 1.0)).xyz;
    let world_basis = mat3x3<f32>(
        instance.world[0].xyz,
        instance.world[1].xyz,
        instance.world[2].xyz,
    );
    let world_u = world_basis * local_u;
    let world_v = world_basis * local_v;
    let n_world = normalize(world_basis * n_local);

    // Triangle-strip corner layout: bits of vertex_index pick ±u, ±v.
    let uv = vec2<f32>(
        select(-1.0, 1.0, (vertex_index & 1u) != 0u),
        select(-1.0, 1.0, (vertex_index & 2u) != 0u),
    );

    // Edge-on fall-through fix. When the projected screen-space length
    // of either tangent axis drops below ~half a pixel, the splat is
    // nearly edge-on to the view; rasterized as a thin sliver, adjacent
    // splats leave visible black gaps along the back of the silhouette.
    // Detect that case and switch this splat to a small camera-facing
    // billboard sized to ~1 px in screen space.
    let clip_center = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let proj_u = camera.view_proj * vec4<f32>(world_u, 0.0);
    let proj_v = camera.view_proj * vec4<f32>(world_v, 0.0);
    let inv_w = 1.0 / max(abs(clip_center.w), 1e-6);
    let half_res = 0.5 * camera.resolution;
    let px_u = length(proj_u.xy * inv_w * half_res);
    let px_v = length(proj_v.xy * inv_w * half_res);
    let min_px = min(px_u, px_v);

    // Camera-facing basis for the billboard fallback. world_up = +Y;
    // breaks at the zenith/nadir singularity, fine for prototype scenes.
    let view_dir = normalize(world_pos - camera.position.xyz);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let cam_right = normalize(cross(view_dir, world_up));
    let cam_up = cross(cam_right, view_dir);
    // World-space radius that yields ~1 pixel at the disc center's
    // clip-w. ndc_per_pixel = 2/resolution.x, so world_per_pixel ≈
    // clip_w * 2 / resolution.x.
    let billboard_world_radius =
        max(abs(clip_center.w), 1e-6) * 2.0 / camera.resolution.x;

    let billboard_threshold: f32 = 0.5;
    let use_billboard = min_px < billboard_threshold;
    let final_u = select(world_u, cam_right * billboard_world_radius, use_billboard);
    let final_v = select(world_v, cam_up * billboard_world_radius, use_billboard);

    let corner_world = world_pos + final_u * uv.x + final_v * uv.y;
    let clip = camera.view_proj * vec4<f32>(corner_world, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.corner_uv = uv;
    out.world_normal = n_world;
    // Interpolate the disc CENTER's world position into the fragment.
    // Using `corner_world` here would give per-corner positions and the
    // fragment-side `length(world_pos - camera.position)` would compute
    // distance to the rim, not the center. The center is what matches
    // the compute-march's `first_dist`.
    out.world_pos = world_pos;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

// Six-attachment fragment output — matches the texture layout written
// by `octree_march` (storage textures there, render attachments here).
//
// Motion (`gbuf_motion`) is intentionally NOT in this list. The march
// doesn't write it either, and TAA isn't dispatched in `render_to`
// today; if TAA ever lands here, this is the place to add the velocity
// output (prev_vp * prev_world * local_pos vs current).
struct FsOut {
    @location(0) position:  vec4<f32>,    // gbuf_position (Rgba32Float)
    @location(1) normal:    vec4<f32>,    // gbuf_normal   (Rgba16Float)
    @location(2) material:  vec2<u32>,    // gbuf_material (Rg32Uint)
    @location(3) pick:      u32,          // gbuf_pick     (R32Uint)
    @location(4) glass:     vec2<u32>,    // gbuf_glass    (Rg32Uint)
    @location(5) leaf_slot: u32,          // gbuf_leaf_slot (R32Uint)
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    // Round-disc mask. Anything outside the unit circle is hole.
    if (length(in.corner_uv) > 1.0) {
        discard;
    }

    let attr = leaf_attr_pool[in.leaf_attr_id];

    // LeafAttr packing:
    //   bits  0..16  primary material id (u16)
    //   bits 16..28  secondary material id (u12)
    //   bits 28..32  blend weight (u4)
    let primary   = attr.material_packed & 0xFFFFu;
    let secondary = (attr.material_packed >> 16u) & 0xFFFu;
    let blend4    = (attr.material_packed >> 28u) & 0xFu;

    // gbuf_material packing — must match `octree_march`:
    //   r: primary (lo16) | secondary (hi16)
    //   g: blend_8 (lo8)  | color_rgb565 (hi16)
    let blend8 = (blend4 << 4u) | blend4;
    let packed_r = (primary & 0xFFFFu) | ((secondary & 0xFFFFu) << 16u);

    // Per-voxel colour. 0 means "use material base_color"; the shade
    // pass already handles that as the no-override path.
    let c_packed = color_pool[in.leaf_attr_id];
    let cr8 = c_packed & 0xFFu;
    let cg8 = (c_packed >> 8u) & 0xFFu;
    let cb8 = (c_packed >> 16u) & 0xFFu;
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);
    let packed_g = (blend8 & 0xFFu) | (color_rgb565 << 16u);

    let hit_distance = length(in.world_pos - camera.position.xyz);

    var out: FsOut;
    out.position  = vec4<f32>(in.world_pos, hit_distance);
    // Match march: alpha channel carries `accum_alpha` (=1 for opaque).
    out.normal    = vec4<f32>(in.world_normal, 1.0);
    out.material  = vec2<u32>(packed_r, packed_g);
    out.pick      = instance.object_id;
    out.glass     = vec2<u32>(0u, 0u);
    out.leaf_slot = in.leaf_attr_id;
    return out;
}
