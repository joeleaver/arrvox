// Procedural proxy-mesh rasterizer — vertex + fragment entry points.
//
// First-class rendering path for procedurals baked to a triangle mesh
// via GPU surface-nets-from-SDF (`proc_surface_nets`). Distinct from
// `mesh.wesl` (the octree-mesh raster): proxy meshes have no octree,
// no `LeafAttr` indirection, and no `mesh_resolve` participation.
// Every vertex carries its own shading payload (normal + dual-material
// + color) and the FS writes the full G-buffer directly.
//
// Pipeline contract:
//   · Vertex buffer: one `ProxyVertex` (32 B) per surface-net SN-cube
//     vertex.
//     · @location(0) local_pos:        vec3<f32>  (object-local)
//     · @location(1) normal_oct:       u32        (SDF gradient, oct-packed)
//     · @location(2) material_packed:  u32        (primary lo16 | secondary mid12 | blend hi4)
//     · @location(3) color_packed:     u32        (RGBA8, alpha=0xFF)
//   · Index buffer: u32, two triangles per active sample-edge, CCW
//     about the outward normal.
//   · One indexed draw per proxy-mesh instance — no LOD select, no
//     cluster table. The proxy mesh is a single root cluster.
//   · Per-instance uniform at @group(1) @binding(0) carries the
//     world matrix + object_id for picking.
//
// Output attachments — must match the G-buffer layout consumed by the
// downstream PBR / shadow / GI passes. Same target set + packing
// `mesh_resolve.wesl` writes for the mesh path, so downstream code
// is agnostic to which path filled a pixel.
//   @location(0) position: Rgba32Float  (world.xyz + hit_distance)
//   @location(1) pick:     R32Uint      (instance object_id)
//   @location(2) normal:   Rgba16Float  (world_normal.xyz + alpha=1)
//   @location(3) material: Rg32Uint     (packed_r, packed_g — see below)
//   @location(4) glass:    Rg32Uint     (zeros — proxy meshes don't write glass)
//
// Phase 1 scope: opaque only. If a procedural's primary material is
// glass, it still renders opaque here — full glass requires the voxel
// path (see notes/proxy-mesh-first-class.md).

import package::lib::types::CameraUniforms;
import package::lib::oct_normal::unpack_oct_normal;

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Per-instance uniform. Shape kept minimal: a world matrix and an
// object_id. No skinning, no grid_origin (proxy meshes write
// world-space-equivalent positions via the world matrix).
struct ProxyInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> instance: ProxyInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    // material IDs can't be interpolated — flat, taken from the
    // provoking vertex of the triangle. The matching `blend` channel
    // below DOES interpolate, so smooth dual-material gradients still
    // work as long as adjacent triangles share material slots.
    @location(2) @interpolate(flat) material_packed: u32,
    // RGB color, linearly interpolated across the triangle in
    // 0..1 space. Repacked back into RGB565 at FS time to match the
    // G-buffer format `mesh_resolve` writes.
    @location(3) color_rgb: vec3<f32>,
    // Per-vertex blend weight from the SDF tree, interpolated.
    // Matches the dual-material blend the voxel path stores per cell.
    @location(4) blend_f: f32,
    // Per-vertex paint intensity (high byte of color_packed). 0xFF
    // for procedural-set colors so they fully override the material
    // base_color in the PBR shader; 0 if the procedural left color
    // unset (RGB will be zero, intensity=0 makes the FS path treat
    // it as "no override"). Flat — no interpolation makes sense here.
    @location(5) @interpolate(flat) intensity: u32,
}

@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
    @location(1) normal_oct: u32,
    @location(2) material_packed: u32,
    @location(3) color_packed: u32,
) -> VsOut {
    let world_pos = (instance.world * vec4<f32>(local_pos, 1.0)).xyz;
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    // Object-space normal → world space via the world matrix's
    // upper-left 3x3. For uniform scale this is exact; for non-
    // uniform scale the renormalize() in the FS handles the bias.
    let n_local = unpack_oct_normal(normal_oct);
    let basis = mat3x3<f32>(
        instance.world[0].xyz,
        instance.world[1].xyz,
        instance.world[2].xyz,
    );
    let n_world = basis * n_local;

    // Extract blend_4 from the packed material word, expand to f32 in
    // 0..1 so per-fragment interpolation gives a smooth dual-material
    // gradient across the triangle.
    let blend4 = (material_packed >> 28u) & 0xFu;
    let blend_f = f32(blend4) / 15.0;

    // Unpack RGBA8 color. Low 3 bytes = RGB, high byte = intensity.
    let cr8 = color_packed & 0xFFu;
    let cg8 = (color_packed >> 8u) & 0xFFu;
    let cb8 = (color_packed >> 16u) & 0xFFu;
    let ci8 = (color_packed >> 24u) & 0xFFu;
    let color_rgb = vec3<f32>(
        f32(cr8) * (1.0 / 255.0),
        f32(cg8) * (1.0 / 255.0),
        f32(cb8) * (1.0 / 255.0),
    );

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.world_normal = n_world;
    out.material_packed = material_packed;
    out.color_rgb = color_rgb;
    out.blend_f = blend_f;
    out.intensity = ci8;
    return out;
}

struct FsOut {
    @location(0) position: vec4<f32>,
    @location(1) pick:     u32,
    @location(2) normal:   vec4<f32>,
    @location(3) material: vec2<u32>,
    @location(4) glass:    vec2<u32>,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    // Renormalize — barycentric interp shrinks vector length.
    let n_world = normalize(in.world_normal);

    // Decode the flat material IDs.
    let primary   = in.material_packed & 0xFFFFu;
    let secondary = (in.material_packed >> 16u) & 0xFFFu;

    // Re-quantize the interpolated blend back to an 8-bit weight
    // matching `mesh_resolve`'s packing (blend4 << 4 | blend4).
    let blend_clamped = clamp(in.blend_f, 0.0, 1.0);
    let blend4 = u32(blend_clamped * 15.0 + 0.5);
    let blend8 = (blend4 << 4u) | blend4;

    // Repack interpolated RGB into RGB565 — same encoding the
    // G-buffer's gbuf_material consumes downstream.
    let cr8 = u32(clamp(in.color_rgb.r, 0.0, 1.0) * 255.0 + 0.5);
    let cg8 = u32(clamp(in.color_rgb.g, 0.0, 1.0) * 255.0 + 0.5);
    let cb8 = u32(clamp(in.color_rgb.b, 0.0, 1.0) * 255.0 + 0.5);
    let cr5 = (cr8 * 31u) / 255u;
    let cg6 = (cg8 * 63u) / 255u;
    let cb5 = (cb8 * 31u) / 255u;
    let color_rgb565 = cr5 | (cg6 << 5u) | (cb5 << 11u);

    // Material packing — must match `mesh_resolve.wesl` so the PBR
    // shading pass reads identical fields regardless of source.
    //   r: primary (lo16) | secondary (hi16)
    //   g: blend_8 (lo8) | intensity (mid8) | color_rgb565 (hi16)
    let packed_r = primary | (secondary << 16u);
    let packed_g =
        (blend8 & 0xFFu)
        | ((in.intensity & 0xFFu) << 8u)
        | (color_rgb565 << 16u);

    let hit_distance = length(in.world_pos - camera.position.xyz);

    var out: FsOut;
    out.position = vec4<f32>(in.world_pos, hit_distance);
    out.pick     = instance.object_id;
    out.normal   = vec4<f32>(n_world, 1.0);
    out.material = vec2<u32>(packed_r, packed_g);
    // No glass writes for proxy meshes — see Phase 1 scope.
    out.glass    = vec2<u32>(0u, 0u);
    return out;
}
