// Surface-splat rasterizer — vertex + fragment entry points.
//
// Visibility-buffer architecture: the raster pass writes only the data
// that can't be reconstructed downstream — world position, the per-leaf
// `leaf_attr_id`, and the per-instance `object_id`. A follow-up
// `splat_resolve` compute pass reads (leaf_slot, pick) per pixel and
// derives the rest of the G-buffer (normal, material, glass) via the
// existing `leaf_attr_pool` / `color_pool` / `instances` indirection.
//
// This split keeps the raster pass under wgpu's 32 B/sample
// `max_color_attachment_bytes_per_sample` limit (16 + 4 + 4 = 24 B) so
// the pipeline validates with the default device limits. The compute
// fixup runs once per pixel — trivial cost compared to the raster
// itself.
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

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::LeafAttr;
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

@group(1) @binding(0) var<uniform> instance: SplatInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) corner_uv: vec2<f32>,        // -1..1 across the disc
    @location(1) world_pos: vec3<f32>,        // disc CENTER world pos
    @location(2) @interpolate(flat) leaf_attr_id: u32,
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

    let world_pos = (instance.world * vec4<f32>(local_pos, 1.0)).xyz;
    let world_basis = mat3x3<f32>(
        instance.world[0].xyz,
        instance.world[1].xyz,
        instance.world[2].xyz,
    );
    let n_world_unscaled = world_basis * n_local;

    // Backface cull. The compute march implicitly does this by stopping
    // at the first hit; the splat path has to do it explicitly or every
    // back-facing splat costs vertex+fragment work for nothing. Roughly
    // halves emitted fragments on a closed mesh.
    //
    // Strict `< 0` keeps silhouette splats (dot ≈ 0) — they're picked
    // up by the billboard-fallback below. Going more aggressive
    // (e.g. `< 0.1`) reintroduces silhouette holes.
    let to_camera = camera.position.xyz - world_pos;
    if (dot(n_world_unscaled, to_camera) < 0.0) {
        var out: VsOut;
        // Degenerate the quad: all four corners at the same outside-
        // clip-volume point ⇒ both triangles in the strip have zero
        // area, rasterizer emits zero fragments.
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.corner_uv = vec2<f32>(0.0);
        out.world_pos = vec3<f32>(0.0);
        out.leaf_attr_id = 0u;
        return out;
    }

    // Build the disc in OBJECT-LOCAL space, then transform corners
    // through `instance.world`. This makes the disc inherit any scale
    // / non-uniform scale / rotation in the instance transform without
    // special-casing.
    let basis_local = frisvad_tangent_basis(n_local);
    let local_u = basis_local[0] * radius;
    let local_v = basis_local[1] * radius;

    let world_u = world_basis * local_u;
    let world_v = world_basis * local_v;

    // Triangle-strip corner layout: bits of vertex_index pick ±u, ±v.
    let uv = vec2<f32>(
        select(-1.0, 1.0, (vertex_index & 1u) != 0u),
        select(-1.0, 1.0, (vertex_index & 2u) != 0u),
    );

    // Edge-on fall-through fix. When the projected screen-space length
    // of either tangent axis drops below ~half a pixel, swap to a
    // small camera-facing billboard sized to ~1 px to keep the
    // silhouette tight. Same fix as the prototype, retained because
    // it's a vertex-stage cheap test that prevents visible holes on
    // glancing surfaces.
    let clip_center = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let proj_u = camera.view_proj * vec4<f32>(world_u, 0.0);
    let proj_v = camera.view_proj * vec4<f32>(world_v, 0.0);
    let inv_w = 1.0 / max(abs(clip_center.w), 1e-6);
    let half_res = 0.5 * camera.resolution;
    let px_u = length(proj_u.xy * inv_w * half_res);
    let px_v = length(proj_v.xy * inv_w * half_res);
    let min_px = min(px_u, px_v);

    let view_dir = normalize(world_pos - camera.position.xyz);
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let cam_right = normalize(cross(view_dir, world_up));
    let cam_up = cross(cam_right, view_dir);
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
    // Per-pixel world position. Interpolating `corner_world` (with
    // perspective correction) gives each fragment the actual surface
    // point under that pixel — equivalent to the camera ray's
    // intersection with the disc plane, since the disc is planar.
    //
    // Earlier we passed `world_pos` (the disc center, identical for
    // all 4 corners). That made `gbuf_position` piecewise-flat across
    // each disc, which broke shade's screen-space neighbor reads
    // (SSAO, bilateral) and pushed the shade pass to ~13 ms. With the
    // interpolation, neighbor reads see smoothly varying positions.
    //
    // For the billboard-fallback case, `corner_world` is on the
    // billboard plane rather than the surface plane — a small offset
    // from "true" surface position, but the billboard is an
    // approximation anyway.
    out.world_pos = corner_world;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

// Three-attachment fragment output — visibility-buffer triplet:
//   position   (Rgba32Float, 16 B): world.xyz + hit_distance
//   pick       (R32Uint,      4 B): instance.object_id
//   leaf_slot  (R32Uint,      4 B): leaf_attr_id
// Total 24 B, well under the default 32 B/sample limit. The remaining
// G-buffer entries (normal/material/glass) are filled by the
// `splat_resolve` compute pass.
struct FsOut {
    @location(0) position:  vec4<f32>,
    @location(1) pick:      u32,
    @location(2) leaf_slot: u32,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    // Round-disc mask. Anything outside the unit circle is hole.
    if (length(in.corner_uv) > 1.0) {
        discard;
    }
    let hit_distance = length(in.world_pos - camera.position.xyz);
    var out: FsOut;
    out.position  = vec4<f32>(in.world_pos, hit_distance);
    out.pick      = instance.object_id;
    out.leaf_slot = in.leaf_attr_id;
    return out;
}
