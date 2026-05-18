// PBR helpers shared across shade / glass / shadow passes. Pure: no
// bindings, no struct dependencies.

// Schlick reflectance at normal incidence for a dielectric surface.
fn dielectric_f0(ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

// Schlick Fresnel approximation, vec3 form (per-channel f0 for metals
// + tinted dielectrics).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Scalar variant — saves a vec3 broadcast for monochrome dielectrics.
fn fresnel_schlick_f0(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Convenience: Fresnel for a dielectric given its IOR.
fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    return fresnel_schlick_f0(cos_theta, dielectric_f0(ior));
}

// Beer-Lambert absorption through a tinted dielectric. A colorless
// glass passes everything (glass_color near (1,1,1) → absorption →
// (1,1,1)); a saturated color absorbs its complement, deepening with
// thickness. The `* 5.0` scales `thickness` into a visually-useful
// range — matches the rkifield reference implementation. Callers
// pass world-space thickness in metres.
fn beer_absorption(glass_color: vec3<f32>, thickness: f32) -> vec3<f32> {
    let sigma = max(-log(max(glass_color, vec3<f32>(0.01))), vec3<f32>(0.0));
    return exp(-sigma * thickness * 5.0);
}
