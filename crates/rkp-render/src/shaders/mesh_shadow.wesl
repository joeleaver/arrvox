// Mesh-rendered shadow map — depth-only render.
//
// Phase 3 of the splat-to-mesh pivot. Vertex stage projects mesh
// triangles through the directional light's view-proj into a 1024²
// `Depth32Float` attachment. There is NO fragment shader — the
// rasterizer fills depth on its own. This lets the GPU run early-z
// culling at full speed; the prior atomic-min-from-fragment approach
// disabled early-z because the fragment had a side effect, costing
// ~3-5× the fragment work.
//
// A separate compute pass (`mesh_shadow_blit.wesl`) reads this depth
// texture and writes `bitcast<u32>(depth)` into the existing
// `shadow_buffer` storage that `shadow_map_pass` owns and shade
// already samples — so the shade-side sampling code is reused
// unchanged.

import package::lib::types::LightCameraShadow;

struct MeshShadowInstance {
    world: mat4x4<f32>,
    object_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> light_camera: LightCameraShadow;

@group(1) @binding(0) var<uniform> instance: MeshShadowInstance;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vert_main(
    @location(0) local_pos: vec3<f32>,
) -> VsOut {
    let world_pos = (instance.world * vec4<f32>(local_pos, 1.0)).xyz;
    let clip = light_camera.view_proj * vec4<f32>(world_pos, 1.0);
    var out: VsOut;
    out.clip_pos = clip;
    return out;
}
