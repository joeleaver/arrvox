// General-purpose math helpers shared across shaders. Anything here
// must be pure: no bindings, no struct dependencies, no side effects.
//
// As of Wave V the build runs WESL with `ManglerKind::None`, so any
// helper here is reachable both from native imports and from
// shader-composer text splices — names round-trip through imports
// unmodified.

// Float32 has 24 bits of mantissa precision, so any digits past
// `3.14159265` are below FP rounding noise — keep the canonical
// constant short and identical everywhere.
const PI: f32 = 3.14159265;

// Conservative L2 distance from a unit cube's center to its corner.
// Used by procedural noise displacement to shrink a child's
// distance estimate so sphere tracing stays safe across the warp.
const SQRT3: f32 = 1.7320508075688772;

// Ray-sphere intersection (origin assumed at sphere centre's frame).
// Returns (t_enter, t_exit) in `vec2<f32>(-1.0, -1.0)` on miss.
fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(d);
    return vec2<f32>(-b - s, -b + s);
}

// Convenience for callers that only want the exit. Returns -1.0 on miss.
fn ray_sphere_exit(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return -1.0; }
    return -b + sqrt(d);
}

// Jimenez 2014 interleaved-gradient noise — temporal blue-noise dither
// for fog/cloud/shadow stippling. `frame % 64` cycles the offset so a
// 64-frame TAA history can converge.
fn interleaved_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let offset = 5.588238 * f32(frame % 64u);
    let p = pixel + vec2<f32>(offset, offset);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

// Slab-test ray-AABB intersection. Returns
// `vec2<f32>(t_enter, t_exit)` clamped to `t_enter >= 0` (so rays
// starting inside the box clip the entry to the origin). Miss is
// indicated by `t_enter > t_exit` — callers must check.
fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(max(tmin.x, tmin.y), tmin.z), 0.0),
                     min(min(tmax.x, tmax.y), tmax.z));
}

// Hash 3 floats into one float in [0, 1). Hugues Hoppe's "hash13" —
// the canonical building block for value/gradient noise. Cheap and
// good enough for cloud / fog / fresnel stippling; not for crypto.
fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D value noise in [0, 1], trilinearly interpolated from a hashed
// lattice. Smoothstep on the fractional coords kills the lattice-
// derivative discontinuity. Multiple octaves layer cleanly via fbm
// (caller's responsibility).
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let ip = floor(p);
    let fp = fract(p);
    let w = fp * fp * (3.0 - 2.0 * fp);
    let c000 = hash13(ip);
    let c100 = hash13(ip + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash13(ip + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash13(ip + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash13(ip + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash13(ip + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash13(ip + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash13(ip + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, w.x);
    let x10 = mix(c010, c110, w.x);
    let x01 = mix(c001, c101, w.x);
    let x11 = mix(c011, c111, w.x);
    let y0 = mix(x00, x10, w.y);
    let y1 = mix(x01, x11, w.y);
    return mix(y0, y1, w.z);
}

// Affine 4x4 inverse — the common case for object↔world transforms.
// Cheaper than a full 4x4 inverse; assumes the bottom row is `(0,0,0,1)`
// and the top-left 3x3 is invertible (any rotation+scale satisfies that).
fn mat4_affine_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let t = m[3].xyz;
    let inv_det = 1.0 / dot(a, cross(b, c));
    let row0 = cross(b, c) * inv_det;
    let row1 = cross(c, a) * inv_det;
    let row2 = cross(a, b) * inv_det;
    // R^-1 * t — each component is a row · t.
    let new_t = -vec3<f32>(dot(row0, t), dot(row1, t), dot(row2, t));
    // Pack rows as the columns of a WGSL column-major matrix → effectively
    // transposes (row0, row1, row2) into the inverse rotation.
    return mat4x4<f32>(
        vec4<f32>(row0.x, row1.x, row2.x, 0.0),
        vec4<f32>(row0.y, row1.y, row2.y, 0.0),
        vec4<f32>(row0.z, row1.z, row2.z, 0.0),
        vec4<f32>(new_t, 1.0),
    );
}
