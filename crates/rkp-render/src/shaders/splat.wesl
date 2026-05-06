// Surface-splat rasterizer prototype — vertex + fragment entry points.
//
// Rasterizes one quad per voxel. The quad is oriented to the prefiltered
// surface normal stored in `LeafAttr.normal_oct`, so adjacent voxel discs
// produce a smooth surface at any voxel resolution — the silhouette
// follows the actual surface, not the grid axes.
//
// Pipeline contract:
//   · Vertex buffer: one `SplatVertex` per voxel (Rust splat_pass::extract).
//     · @location(0) world_pos: vec3<f32>   (cell center)
//     · @location(1) radius:    f32         (world-space disc radius)
//     · @location(2) leaf_attr_id: u32      (absolute leaf_attr_pool slot)
//   · Step mode: Instance. Draw 4 verts × N instances; @builtin(vertex_index)
//     selects the quad corner (triangle-strip).
//   · `corner_uv` ∈ [-1,1]² interpolated to fragment for the round mask.
//
// Color path: writes both the diffuse albedo (from material) and the
// world-space normal as MRT outputs, alongside writing depth from
// the world-space corner. Designed for the prototype test harness, not
// for the production G-buffer (which is currently compute-storage).
// Phase B-2 will extend this to write into the existing G-buffer once
// the textures are reconfigured with RENDER_ATTACHMENT usage.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::LeafAttr;
import package::lib::types::GpuMaterial;
import package::lib::types::mat_albedo;

struct SplatCamera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad0: f32,
    resolution: vec2<f32>,
    _pad1: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: SplatCamera;
@group(0) @binding(1) var<storage, read> leaf_attr_pool: array<LeafAttr>;
@group(0) @binding(2) var<storage, read> materials: array<GpuMaterial>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) corner_uv: vec2<f32>,        // -1..1 across the disc
    @location(1) world_normal: vec3<f32>,     // prefiltered, flat across splat
    @location(2) world_pos: vec3<f32>,        // interpolated; for depth + lighting
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
    @location(0) world_pos: vec3<f32>,
    @location(1) radius: f32,
    @location(2) leaf_attr_id: u32,
) -> VsOut {
    // Decode the prefiltered normal once per splat (vertex-shader work
    // amortizes across the 4 corners — same value for all of them).
    let attr = leaf_attr_pool[leaf_attr_id];
    let n_world = unpack_oct_normal(attr.normal_oct);
    let basis = frisvad_tangent_basis(n_world);
    let tangent_u = basis[0];
    let tangent_v = basis[1];

    // Triangle-strip corner layout: bits of vertex_index pick ±u, ±v.
    let uv = vec2<f32>(
        select(-1.0, 1.0, (vertex_index & 1u) != 0u),
        select(-1.0, 1.0, (vertex_index & 2u) != 0u),
    );

    // Edge-on fall-through fix. When the projected screen-space
    // length of either tangent axis drops below ~half a pixel, the
    // splat is nearly edge-on to the view; rasterized as a thin
    // sliver, adjacent splats leave visible black gaps along the
    // back of the silhouette. Detect that case and switch this
    // splat to a small camera-facing billboard sized to 1 px in
    // screen space — covers the gap without distorting splats that
    // were rasterizing cleanly.
    //
    // We check the WORLD-space tangent disc's projected size by
    // running its u/v deltas through view_proj (with w=0 input so
    // we pick up just the linear projection), then dividing by the
    // disc CENTER's clip-w to convert to NDC.
    let clip_center = camera.view_proj * vec4<f32>(world_pos, 1.0);
    let proj_u = camera.view_proj * vec4<f32>(tangent_u * radius, 0.0);
    let proj_v = camera.view_proj * vec4<f32>(tangent_v * radius, 0.0);
    let inv_w = 1.0 / max(abs(clip_center.w), 1e-6);
    let half_res = 0.5 * camera.resolution;
    let px_u = length(proj_u.xy * inv_w * half_res);
    let px_v = length(proj_v.xy * inv_w * half_res);
    let min_px = min(px_u, px_v);

    // Camera-facing basis for the billboard fallback. world_up = +Y;
    // breaks at the zenith/nadir singularity, fine for prototype scenes.
    let view_dir = normalize(world_pos - camera.position);
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
    let final_u = select(tangent_u * radius, cam_right * billboard_world_radius, use_billboard);
    let final_v = select(tangent_v * radius, cam_up * billboard_world_radius, use_billboard);

    let corner_world = world_pos + final_u * uv.x + final_v * uv.y;
    let clip = camera.view_proj * vec4<f32>(corner_world, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.corner_uv = uv;
    out.world_normal = n_world;
    out.world_pos = corner_world;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

struct FsOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    // Round-disc mask. Anything outside the unit circle is hole.
    if (length(in.corner_uv) > 1.0) {
        discard;
    }
    // Recover material albedo via the prefiltered slot.
    let attr = leaf_attr_pool[in.leaf_attr_id];
    let mat_id = attr.material_packed & 0xFFFFu;
    let mat = materials[mat_id];

    var out: FsOut;
    out.albedo = vec4<f32>(mat_albedo(mat), 1.0);
    out.normal = vec4<f32>(in.world_normal * 0.5 + vec3<f32>(0.5), 1.0);
    return out;
}
