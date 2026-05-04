// Primitive signed-distance functions. Each mirrors its CPU
// counterpart in `rkp_procedural::leaves`. Pure functions — no
// bindings, no struct dependencies.

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>, rounding: f32) -> f32 {
    let q = abs(p) - half_extents + vec3<f32>(rounding);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside - rounding;
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let t = clamp(p.y, -half_height, half_height);
    let closest = vec3<f32>(0.0, t, 0.0);
    return length(p - closest) - radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let radial = length(vec3<f32>(p.x, 0.0, p.z)) - radius;
    let axial = abs(p.y) - half_height;
    // Keep the CPU branching exactly (`leaves::eval_cylinder`): when
    // the point is outside both cylinder and caps, distance is the
    // diagonal length; otherwise it's `max` of the signed axes.
    if (radial > 0.0 && axial > 0.0) {
        return sqrt(radial * radial + axial * axial);
    }
    return max(radial, axial);
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let xz_len = length(vec3<f32>(p.x, 0.0, p.z));
    let q = vec3<f32>(xz_len - major_radius, p.y, 0.0);
    return length(q) - minor_radius;
}

fn sdf_plane(p: vec3<f32>) -> f32 {
    return p.y;
}

fn sdf_ramp(p: vec3<f32>, half_length: f32, half_height: f32, half_width: f32) -> f32 {
    let q = abs(p) - vec3<f32>(half_length, half_height, half_width);
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    let box_dist = outside + inside;
    let hyp = max(sqrt(half_length * half_length + half_height * half_height), 1e-6);
    let plane_dist = (half_length * p.y - half_height * p.x) / hyp;
    return max(box_dist, plane_dist);
}
