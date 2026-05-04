// General-purpose math helpers shared across shaders. Anything here
// must be pure: no bindings, no struct dependencies, no side effects.
// Type-bound helpers (mat_albedo, leaf_attr_material_primary, etc.)
// belong in the Phase 3 type-unification lib once the structs they
// reference move there too.
//
// CAVEAT — splice-consumer compatibility: any helper called from a
// `shader_composer` spliced chunk must NOT live here. The chunk is
// emitted as text with unmangled names, but WESL renames imported
// symbols at emit time, so the names diverge and the splice fails
// at parse. `intersect_aabb` is the canonical example; it stays
// duplicated across `octree_march`, `rkp_shadow_trace`, and
// `shadow_scatter` until the composer migrates to WESL-native
// composition (deferred Phase 4 work).

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
