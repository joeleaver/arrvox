// Surface-mesh rasterizer — vertex + fragment entry points.
//
// Phase 2 of the splat-to-mesh pivot. Mirrors `splat.wesl`'s
// visibility-buffer contract: the raster writes (position, pick,
// leaf_slot); the existing `splat_resolve` compute pass fills in the
// rest of the G-buffer (normal / material / glass) per-pixel via the
// `leaf_attr_pool` indirection.
//
// Pipeline contract:
//   · Vertex buffer: one `MeshVertex` per surface-net SN-cube vertex.
//     · @location(0) local_pos:    vec3<f32>   (grid corner, OBJECT-LOCAL)
//     · @location(1) normal_oct:   u32         (oct-packed; unused here,
//                                                resolve reads from pool)
//     · @location(2) leaf_attr_id: u32         (absolute leaf_attr_pool slot)
//   · Index buffer: u32, two triangles per exposed cell face, CCW-wound
//     around the outward normal.
//   · One indexed draw per scene-instance per asset; the per-instance
//     world matrix comes from `instance` at @group(1) @binding(0).
//
// Bind-group shape is identical to `splat.wesl`'s so the same
// `splat_g0` / `splat_instance_*` bind groups can be reused for the
// mesh pipeline. Binding 1 of g0 (leaf_attr_pool) is declared in the
// shared layout but isn't referenced from this shader — the splat
// vertex stage needs it for disc-basis construction; the mesh vertex
// stage doesn't.

import package::lib::types::CameraUniforms;

// Per-instance uniform — shape-compatible with `splat.wesl`'s
// `SplatInstance` (80 B, mat4x4 + u32 + 12 B pad). The same g1 bind
// group is reused across both pipelines.
struct MeshInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<uniform> instance: MeshInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) @interpolate(flat) leaf_attr_id: u32,
}

// `normal_oct` (vertex location 1) is declared in the pipeline's
// vertex-buffer layout but isn't consumed here — the resolve compute
// reads `LeafAttr.normal_oct` from the pool slot for each hit pixel.
// wgpu tolerates unused locations; the buffer reader still strides
// past them.
@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
    @location(2) leaf_attr_id: u32,
) -> VsOut {
    let world_pos = (instance.world * vec4<f32>(local_pos, 1.0)).xyz;
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world_pos;
    out.leaf_attr_id = leaf_attr_id;
    return out;
}

// Three-attachment fragment output — same visibility-buffer triplet
// `splat.wesl` writes:
//   position   (Rgba32Float, 16 B): world.xyz + hit_distance
//   pick       (R32Uint,      4 B): instance.object_id
//   leaf_slot  (R32Uint,      4 B): leaf_attr_id
struct FsOut {
    @location(0) position:  vec4<f32>,
    @location(1) pick:      u32,
    @location(2) leaf_slot: u32,
}

@fragment
fn frag_main(in: VsOut) -> FsOut {
    let hit_distance = length(in.world_pos - camera.position.xyz);
    var out: FsOut;
    out.position  = vec4<f32>(in.world_pos, hit_distance);
    out.pick      = instance.object_id;
    out.leaf_slot = in.leaf_attr_id;
    return out;
}
