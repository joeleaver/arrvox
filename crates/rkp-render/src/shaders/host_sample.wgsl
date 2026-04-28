// host_sample.wgsl — distance/normal sampler at a host octree's surface.
//
// V1 stub: the geometry pipeline doesn't query a real host yet —
// regions are free-standing AABBs (e.g. a hillside Plane's bounding box).
// `HostSample` ships through `dispatch_user_generate` so the contract is
// stable; future versions will fill the field by descending the host's
// octree (mirrors the CPU `voxelize_octree` 9-tap classifier) and
// reading per-leaf normal_oct.
//
// Returning `distance = 0` tells the shader "you're on the host's
// surface"; users who want grass *only* near the surface should gate on
// the future `distance` field rather than relying on the V1 stub.

struct HostSample {
    /// Signed distance to the nearest host surface. Negative inside.
    /// V1: always 0 (on-surface).
    distance: f32,
    /// Surface normal at the nearest host surface point.
    /// V1: always +Y (worldspace up).
    normal: vec3<f32>,
}

fn host_sample_at(world_pos: vec3<f32>) -> HostSample {
    var s: HostSample;
    s.distance = 0.0;
    s.normal = vec3<f32>(0.0, 1.0, 0.0);
    return s;
}
