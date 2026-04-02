// Shading functions — PBR, shadows, AO, SSS, noise, material blending, GI cone tracing, atmosphere, brush overlay.
//
// Opacity-field version for RKIPatch. This file is concatenated AFTER
// opacity_shade_common.wgsl by the CPU-side ShaderComposer. All structs, bindings,
// constants, opacity sampling functions, and compatibility shims are defined in
// opacity_shade_common.wgsl and available here.

// ---------- Opacity Transmittance Shadow ----------

/// Hard shadow via fixed-step transmittance march through the opacity field.
/// Accumulates opacity along the ray; returns 0.0 (fully shadowed) when
/// transmittance drops below threshold, 1.0 (fully lit) otherwise.
fn hard_shadow(origin: vec3<f32>, light_dir: vec3<f32>, max_dist: f32) -> f32 {
    var t = SHADOW_EPSILON;
    var transmittance = 1.0;
    let step_base = OPACITY_TRANSMITTANCE_STEP;

    for (var i = 0u; i < 48u; i++) {
        if t > max_dist || transmittance < 0.01 {
            break;
        }
        let pos = origin + light_dir * t;

        // Coarse field early-out: skip regions far from any object.
        let cam_rel = pos - shade_uniforms.camera_pos.xyz;
        let coarse_dist = sample_coarse_field(cam_rel);
        if coarse_dist > COARSE_NEAR_THRESHOLD {
            t += coarse_dist * 0.5;
            continue;
        }

        let opacity = sample_opacity_bvh(pos);
        // Attenuate transmittance by local opacity.
        transmittance *= (1.0 - opacity);

        // Adaptive step size: step faster through empty space, slower near surfaces.
        let step_size = mix(step_base * 4.0, step_base, opacity);
        t += max(step_size, SHADOW_EPSILON);
    }

    return transmittance;
}

// ---------- Opacity Soft Shadow ----------
//
// Transmittance march with softness modulation. For opacity fields, hard and soft
// shadows both use transmittance accumulation. The `k` parameter controls how
// aggressively transmittance is accumulated — higher k means sharper shadows.

fn soft_shadow(origin: vec3<f32>, light_dir: vec3<f32>, max_dist: f32, k: f32) -> f32 {
    var t = SHADOW_EPSILON;
    var transmittance = 1.0;
    let step_base = OPACITY_TRANSMITTANCE_STEP;
    // k modulates sharpness: higher k = steeper falloff = sharper shadow.
    // Normalize so k=128 gives roughly the same result as hard_shadow.
    let sharpness = k / 128.0;

    for (var i = 0u; i < 48u; i++) {
        if t > max_dist || transmittance < 0.01 {
            break;
        }
        let pos = origin + light_dir * t;

        let cam_rel = pos - shade_uniforms.camera_pos.xyz;
        let coarse_dist = sample_coarse_field(cam_rel);
        if coarse_dist > COARSE_NEAR_THRESHOLD {
            t += coarse_dist * 0.5;
            continue;
        }

        let opacity = sample_opacity_bvh(pos);
        // Weight opacity by sharpness and distance — nearby occluders are sharper.
        let weighted_opacity = opacity * sharpness * (1.0 / max(t * 0.5, 0.1));
        transmittance *= clamp(1.0 - weighted_opacity * step_base, 0.0, 1.0);

        let step_size = mix(step_base * 4.0, step_base, opacity);
        t += max(step_size / max(sharpness * 0.5, 0.5), SHADOW_EPSILON);
    }

    return transmittance;
}

/// Dispatch to hard or soft shadow based on shadow_softness parameter.
fn compute_shadow(origin: vec3<f32>, light_dir: vec3<f32>, max_dist: f32) -> f32 {
    let softness = shade_uniforms.shadow_params.x;
    if softness > 0.0 {
        let k = 128.0 / softness;
        return soft_shadow(origin, light_dir, max_dist, k);
    }
    return hard_shadow(origin, light_dir, max_dist);
}

// ---------- Opacity Ambient Occlusion ----------

/// 5-layer opacity probe along the surface normal.
/// Samples opacity at increasing distances and accumulates occlusion.
fn sdf_ao(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var weight = 1.0;
    for (var i = 1u; i <= 5u; i++) {
        let dist = AO_STEP_SIZE * f32(i);
        let opacity = sample_opacity_bvh(pos + normal * dist);
        occlusion += weight * opacity;
        weight *= 0.5;
    }
    return clamp(1.0 - 1.5 * occlusion, 0.0, 1.0);
}

// ---------- BRDF Evaluation Helper ----------

struct BrdfResult {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

/// Evaluate Cook-Torrance GGX BRDF for a given light direction.
/// Returns separate diffuse and specular components (not yet multiplied by radiance or n_dot_l).
fn evaluate_brdf(light_dir: vec3<f32>, ctx: ShadingContext) -> BrdfResult {
    let half_vec = normalize(ctx.view_dir + light_dir);
    let n_dot_h = max(dot(ctx.normal, half_vec), 0.0);
    let h_dot_v = max(dot(half_vec, ctx.view_dir), 0.0);
    let n_dot_l = max(dot(ctx.normal, light_dir), 0.0);

    let d = distribution_ggx(n_dot_h, ctx.roughness);
    let v = visibility_smith_ggx(ctx.n_dot_v, n_dot_l, ctx.roughness);
    let f = fresnel_schlick(h_dot_v, ctx.f0);

    let specular_brdf = d * v * f;
    let ks = f;
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - ctx.metallic);
    let diffuse_brdf = kd * ctx.albedo / PI;

    return BrdfResult(diffuse_brdf, specular_brdf);
}

// ---------- Subsurface Scattering ----------

/// Compute SSS thickness from the opacity field. Call once before the light loop.
/// Probes opacity along -normal to estimate material thickness.
fn sss_thickness(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let interior_pos = pos - normal * SSS_MAX_THICKNESS;
    let opacity = sample_opacity_bvh(interior_pos);
    return opacity * SSS_MAX_THICKNESS;
}

/// Compute SSS contribution for a single light given pre-computed thickness.
fn sss_from_thickness(thickness: f32, normal: vec3<f32>, light_dir: vec3<f32>,
                      subsurface: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
    let attenuation = exp(-thickness * SSS_SIGMA);
    let wrap = max(0.0, dot(normal, light_dir) + SSS_WRAP) / (1.0 + SSS_WRAP);
    return subsurface_color * attenuation * wrap * subsurface;
}

/// Legacy wrapper — computes thickness inline. Prefer sss_thickness + sss_from_thickness.
fn sss_contribution(pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>,
                    subsurface: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
    if subsurface <= 0.0 {
        return vec3<f32>(0.0);
    }
    let thickness = sss_thickness(pos, normal);
    return sss_from_thickness(thickness, normal, light_dir, subsurface, subsurface_color);
}

// ---------- Light Attenuation ----------

fn distance_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = dist * dist;
    let r2 = range * range;
    let factor = d2 / r2;
    let window = clamp(1.0 - factor, 0.0, 1.0);
    return (window * window) / max(d2, 0.0001);
}

// ---------- PBR Functions ----------

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn visibility_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(ggxv + ggxl, 0.0001);
}

// ---------- 3D Simplex Noise (Ashima Arts webgl-noise) ----------

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 10.0) * x);
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    var i = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x));

    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + vec3<f32>(C.x);
    let x2 = x0 - i2 + vec3<f32>(C.y);
    let x3 = x0 - D.yyy;

    i = mod289_3(i);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + vec4<f32>(ns.y);
    let y = y_ * ns.x + vec4<f32>(ns.y);
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    var m = max(vec4<f32>(0.5) - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 105.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// ---------- Material Blending ----------

struct ResolvedMaterial {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    emission_strength: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
}

fn resolve_material_from(m: Material) -> ResolvedMaterial {
    return ResolvedMaterial(
        vec3<f32>(m.albedo_r, m.albedo_g, m.albedo_b),
        m.roughness,
        m.metallic,
        vec3<f32>(m.emission_r, m.emission_g, m.emission_b),
        m.emission_strength,
        m.subsurface,
        vec3<f32>(m.subsurface_r, m.subsurface_g, m.subsurface_b),
        m.opacity,
        m.ior,
        m.noise_scale,
        m.noise_strength,
        m.noise_channels,
    );
}

/// Lerp all PBR properties between two resolved materials.
fn blend_resolved_materials(a: ResolvedMaterial, b: ResolvedMaterial, t: f32) -> ResolvedMaterial {
    return ResolvedMaterial(
        mix(a.albedo, b.albedo, t),
        mix(a.roughness, b.roughness, t),
        mix(a.metallic, b.metallic, t),
        mix(a.emission, b.emission, t),
        mix(a.emission_strength, b.emission_strength, t),
        mix(a.subsurface, b.subsurface, t),
        mix(a.subsurface_color, b.subsurface_color, t),
        mix(a.opacity, b.opacity, t),
        mix(a.ior, b.ior, t),
        select(a.noise_scale, b.noise_scale, t > 0.5),
        select(a.noise_strength, b.noise_strength, t > 0.5),
        select(a.noise_channels, b.noise_channels, t > 0.5),
    );
}

// ---------- Radiance Volume Cone Tracing ----------

/// Sample a single level of the radiance volume.
/// `cam_rel_pos` is camera-relative (since volume center = camera world pos,
/// this is the same as volume-center-relative).
fn sample_radiance_level(cam_rel_pos: vec3<f32>, level: u32) -> vec4<f32> {
    let inv_ext = radiance_vol.inv_extents[level];
    let uvw = cam_rel_pos * inv_ext + 0.5;
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    switch level {
        case 0u: { return textureSampleLevel(radiance_L0, radiance_sampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(radiance_L1, radiance_sampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(radiance_L2, radiance_sampler, uvw, 0.0); }
        default: { return textureSampleLevel(radiance_L3, radiance_sampler, uvw, 0.0); }
    }
}

/// Sample the radiance volume with continuous mip level (interpolate between levels).
fn sample_radiance(cam_rel_pos: vec3<f32>, mip_f: f32) -> vec4<f32> {
    let lo = u32(floor(mip_f));
    let hi = u32(ceil(mip_f));
    let lo_clamped = min(lo, 3u);
    let hi_clamped = min(hi, 3u);
    let s_lo = sample_radiance_level(cam_rel_pos, lo_clamped);
    if lo_clamped == hi_clamped {
        return s_lo;
    }
    let s_hi = sample_radiance_level(cam_rel_pos, hi_clamped);
    return mix(s_lo, s_hi, fract(mip_f));
}

/// Trace a cone through the radiance volume using front-to-back compositing.
/// `origin` is in world-space; positions are converted to camera-relative
/// before sampling the radiance volume (which is centered at the camera).
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, tan_half_angle: f32, max_dist: f32, jitter: f32) -> vec4<f32> {
    var color = vec3<f32>(0.0);
    var opacity = 0.0;
    // Start past L0 voxel to avoid self-illumination, with jitter to break banding.
    var t = radiance_vol.voxel_sizes.x * (2.0 + jitter);

    for (var i = 0u; i < GI_CONE_STEPS; i++) {
        if opacity > 0.95 || t > max_dist {
            break;
        }
        let pos = origin + dir * t;
        let cone_radius = t * tan_half_angle;
        // Mip selection: *0.5 accounts for 4x (not 2x) clipmap ratio between levels.
        let mip_f = log2(max(cone_radius / radiance_vol.voxel_sizes.x, 1.0)) * 0.5;

        // Convert world-space position to camera-relative for radiance volume sampling.
        let cam_rel_pos = pos - shade_uniforms.camera_pos.xyz;
        let s = sample_radiance(cam_rel_pos, mip_f);
        let step_opacity = s.a;

        // Front-to-back compositing.
        let w = (1.0 - opacity) * step_opacity;
        color += s.rgb * w;
        opacity += w;

        // Step size increases with cone radius.
        let step = max(cone_radius * 0.5, radiance_vol.voxel_sizes.x);
        t += min(step, GI_MAX_STEP);
    }

    return vec4<f32>(color, opacity);
}

/// Trace 6 diffuse cones in a hemisphere around the surface normal.
fn cone_trace_diffuse(pos: vec3<f32>, normal: vec3<f32>, jitter: f32) -> vec3<f32> {
    // Build tangent frame.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    let tan_half = 0.7; // tan(~35deg) for wider cones to compensate for fewer samples
    var gi = vec3<f32>(0.0);

    // 4 cones tilted 30deg from normal, evenly spaced azimuthally (90deg apart).
    let cos30 = 0.866;
    let sin30 = 0.5;

    for (var i = 0u; i < 4u; i++) {
        let angle = f32(i) * PI * 0.5 + jitter * 0.5;
        let dir = normalize(
            normal * cos30
            + (tangent * cos(angle) + bitangent * sin(angle)) * sin30
        );
        let result = trace_cone(pos, dir, tan_half, GI_DIFFUSE_MAX_DIST, jitter);
        gi += result.rgb;
    }

    return gi / 4.0;
}

/// Trace 1 specular cone along the reflection direction.
fn cone_trace_specular(pos: vec3<f32>, reflect_dir: vec3<f32>, roughness: f32, jitter: f32) -> vec3<f32> {
    // Narrower cone for smoother surfaces.
    let tan_half = max(roughness * 0.5, 0.02);
    let result = trace_cone(pos, reflect_dir, tan_half, GI_SPECULAR_MAX_DIST, jitter);
    return result.rgb;
}

// ---------- Analytic Atmosphere ----------

/// Henyey-Greenstein phase function for sky Mie scattering.
fn henyey_greenstein_sky(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-6), 1.5));
}

/// Compute sky color for a given view ray using analytic Rayleigh + Mie scattering.
/// Returns linear HDR RGB.
fn atmosphere_sky(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cos_theta = ray_dir.y;  // elevation: dot(ray_dir, up)
    let cos_sun = dot(ray_dir, sun_dir);

    let rayleigh_scale = shade_uniforms.sky_params.x;
    let mie_scale = shade_uniforms.sky_params.y;
    let sun_intensity = shade_uniforms.sun_dir.w;
    let sun_col = shade_uniforms.sun_color.xyz;

    // Height-integrated optical depth coefficients (beta x scale_height).
    // Rayleigh: beta_R(lambda) x H_R(8400m), Mie: beta_M x H_M(1200m).
    let tau_r = vec3<f32>(0.032, 0.114, 0.278) * rayleigh_scale;
    let tau_m = 0.025 * mie_scale;

    // Path length through atmosphere (longer at horizon, ~1 at zenith).
    let path = 1.0 / max(cos_theta + 0.025, 0.01);

    // Extinction along the view ray.
    let total_tau = tau_r + vec3<f32>(tau_m);
    let extinction = exp(-total_tau * path);

    // Phase functions.
    let phase_r = 0.75 * (1.0 + cos_sun * cos_sun);
    let g = 0.76;
    let phase_m = henyey_greenstein_sky(cos_sun, g);

    // In-scattered radiance (single scattering).
    // inscatter_i = (beta_i * phase_i / sum_beta) * (1 - exp(-sum_beta * path)) * L_sun
    let scatter_r = tau_r * phase_r;
    let scatter_m = vec3<f32>(tau_m * phase_m);
    let safe_total = max(total_tau, vec3<f32>(1e-6));
    let inscatter = (scatter_r + scatter_m) / safe_total * (vec3<f32>(1.0) - extinction);
    let sky = inscatter * sun_col * sun_intensity;

    // Sun disk + bloom.
    let sun_angular_radius = 0.00465;  // ~0.267 degrees
    let sun_disk = smoothstep(cos(sun_angular_radius * 3.0), cos(sun_angular_radius), cos_sun);
    let sun_bloom = pow(max(cos_sun, 0.0), 256.0) * 2.0;
    let sun_contribution = (sun_disk * 50.0 + sun_bloom) * sun_col * sun_intensity * extinction;

    return sky + sun_contribution;
}

/// Compute view ray direction from pixel coordinates using camera basis.
fn compute_view_ray(pixel: vec2<u32>, dims: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;
    // cam_right and cam_up are pre-scaled by tan(fov/2)*aspect and tan(fov/2).
    return normalize(
        shade_uniforms.cam_forward.xyz
        + ndc.x * shade_uniforms.cam_right.xyz
        - ndc.y * shade_uniforms.cam_up.xyz  // -y: screen y is top-down
    );
}

// ---------- Glass / Transparency ----------

fn dielectric_f0(ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let f0 = dielectric_f0(ior);
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn beer_absorption(glass_color: vec3<f32>, thickness: f32) -> vec3<f32> {
    let sigma = max(-log(max(glass_color, vec3<f32>(0.01))), vec3<f32>(0.0));
    return exp(-sigma * thickness * 5.0);
}

fn march_glass_interior(
    entry_pos: vec3<f32>,
    refract_dir: vec3<f32>,
    obj_idx: u32,
    max_thickness: f32,
) -> vec2<f32> {
    var t = 0.005;
    for (var i = 0u; i < 64u; i++) {
        if t > max_thickness { break; }
        let pos = entry_pos + refract_dir * t;
        // Shadow SDF: interior bricks return positive (vs*8), not negative.
        // Standard SDF traps the march in interior (d always negative → never exits).
        let d = evaluate_object_dist_shadow(pos, obj_idx).x;
        if d > 0.001 {
            return vec2<f32>(t, 1.0);
        }
        t += max(abs(d), 0.01);
    }
    return vec2<f32>(t, 0.0);
}

struct TransmitResult {
    color: vec3<f32>,
    hit: bool,
}

/// Maximum number of glass layers to trace through before giving up.
const MAX_GLASS_LAYERS: u32 = 4u;

fn trace_and_shade_behind_glass(
    exit_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    max_dist: f32,
    view_dir: vec3<f32>,
    jitter: f32,
) -> TransmitResult {
    var result: TransmitResult;
    result.color = vec3<f32>(0.0);
    result.hit = false;

    if v2_scene.num_objects == 0u {
        return result;
    }

    // Track cumulative absorption across multiple glass layers.
    var total_absorption = vec3<f32>(1.0);
    var current_origin = exit_pos;
    var current_dir = ray_dir;

    // Outer loop: each iteration either finds an opaque surface (done)
    // or passes through another glass layer and continues.
    for (var layer = 0u; layer < MAX_GLASS_LAYERS; layer++) {

        // --- BVH ray march to find next surface ---
        var t = 0.005;
        var found_surface = false;
        var hit_pos = vec3<f32>(0.0);
        var best_obj_idx = 0u;

        for (var step = 0u; step < 128u; step++) {
            if t > max_dist { break; }
            let pos = current_origin + current_dir * t;

            var min_dist = MAX_FLOAT;
            var step_best = 0u;

            var stack: array<u32, 32>;
            var sp = 0u;
            stack[0] = 0u;
            sp = 1u;

            while sp > 0u {
                sp -= 1u;
                let ni = stack[sp];
                let node = bvh_nodes[ni];
                let nmin = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
                let nmax = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);
                let closest = clamp(pos, nmin, nmax);
                if length(closest - pos) > min_dist { continue; }

                if node.left == BVH_INVALID {
                    let oi = node.right_or_object;
                    if oi < v2_scene.num_objects {
                        // Shadow SDF: INTERIOR_SLOT → passthrough (vs*8).
                        // This avoids false hits when the exit position is near
                        // or inside adjacent objects (window frames, walls).
                        // Wall interior faces have allocated surface bricks with
                        // proper SDF, so the shadow variant still detects them.
                        let d = evaluate_object_dist_shadow(pos, oi).x;
                        if d < min_dist {
                            min_dist = d;
                            step_best = oi;
                        }
                    }
                } else {
                    if sp < BVH_STACK_SIZE - 1u {
                        stack[sp] = node.left; sp += 1u;
                        stack[sp] = node.right_or_object; sp += 1u;
                    }
                }
            }

            if min_dist < 0.005 {
                hit_pos = current_origin + current_dir * t;
                best_obj_idx = step_best;
                found_surface = true;
                break;
            }

            t += max(min_dist, 0.005);
        }

        if !found_surface {
            // Miss — sky
            if shade_uniforms.sky_params.z > 0.5 {
                result.color = atmosphere_sky(current_dir, normalize(shade_uniforms.sun_dir.xyz)) * total_absorption;
            } else {
                let uv_y = clamp(current_dir.y * 0.5 + 0.5, 0.0, 1.0);
                result.color = mix(SKY_HORIZON, SKY_ZENITH, uv_y) * total_absorption;
            }
            result.hit = true;
            return result;
        }

        // --- Surface hit: compute normal and resolve material ---
        let obj = objects[best_obj_idx];

        let e = 0.01;
        let nx = evaluate_object_dist(hit_pos + vec3<f32>(e, 0.0, 0.0), best_obj_idx)
               - evaluate_object_dist(hit_pos - vec3<f32>(e, 0.0, 0.0), best_obj_idx);
        let ny = evaluate_object_dist(hit_pos + vec3<f32>(0.0, e, 0.0), best_obj_idx)
               - evaluate_object_dist(hit_pos - vec3<f32>(0.0, e, 0.0), best_obj_idx);
        let nz = evaluate_object_dist(hit_pos + vec3<f32>(0.0, 0.0, e), best_obj_idx)
               - evaluate_object_dist(hit_pos - vec3<f32>(0.0, 0.0, e), best_obj_idx);
        var normal = normalize(vec3<f32>(nx, ny, nz));
        if length(vec3<f32>(nx, ny, nz)) < 1e-10 {
            normal = vec3<f32>(0.0, 1.0, 0.0);
        }

        var mat_data = vec3<f32>(f32(obj.material_id), 0.0, 0.0);
        if obj.sdf_type == SDF_TYPE_VOXELIZED || obj.sdf_type == SDF_TYPE_PROCEDURAL {
            let local_hit = (obj.inverse_world * vec4<f32>(hit_pos, 1.0)).xyz;
            mat_data = sample_voxelized_material_full(local_hit, obj);
        }
        let primary_id = u32(mat_data.x);
        let secondary_id = u32(mat_data.y);
        let blend_w = mat_data.z;

        var resolved = resolve_material_from(materials[primary_id]);
        if blend_w > 0.0 && secondary_id != primary_id {
            let secondary_resolved = resolve_material_from(materials[secondary_id]);
            resolved = blend_resolved_materials(resolved, secondary_resolved, blend_w);
        }

        if obj.sdf_type == SDF_TYPE_VOXELIZED || obj.sdf_type == SDF_TYPE_PROCEDURAL {
            let local_hit = (obj.inverse_world * vec4<f32>(hit_pos, 1.0)).xyz;
            let paint = sample_voxelized_color(local_hit, obj);
            if paint.a > 0.0 {
                resolved.albedo = mix(resolved.albedo, paint.rgb, paint.a);
            }
        }

        // --- Check if this surface is glass ---
        if resolved.opacity < 1.0 {
            // Another glass layer: refract through it, accumulate absorption, continue
            let hit_n_dot_v = max(dot(normal, -current_dir), 0.001);
            let glass_F = fresnel_dielectric(hit_n_dot_v, resolved.ior);
            let glass_transmission = (1.0 - glass_F) * (1.0 - resolved.opacity);

            if glass_transmission > 0.01 {
                // Step through transparent material by sampling voxel materials.
                let glass_obj = objects[best_obj_idx];
                let glass_vs = max(glass_obj.voxel_size, 0.01);
                var glass_path = glass_vs;
                for (var gsi = 0u; gsi < 32u; gsi++) {
                    glass_path += glass_vs * 0.5;
                    let gp = hit_pos + current_dir * glass_path;
                    let glp = (glass_obj.inverse_world * vec4<f32>(gp, 1.0)).xyz;
                    let gmat = sample_voxelized_material_full(glp, glass_obj);
                    let gmat_opacity = materials[u32(gmat.x)].opacity;
                    if gmat_opacity >= 1.0 {
                        break;
                    }
                    // Check if we've exited the object by probing opacity.
                    let obj_opacity = sample_opacity_trilinear(glp, glass_obj);
                    if obj_opacity < 0.1 {
                        break;
                    }
                }
                let glass_exit = hit_pos + current_dir * glass_path;

                total_absorption *= beer_absorption(resolved.albedo, glass_path) * glass_transmission;

                current_origin = glass_exit;
                // current_dir unchanged — thin glass preserves ray direction
                continue;
            }
            // Total internal reflection or negligible transmission — treat as opaque
        }

        // --- Opaque surface: full per-voxel material shading ---
        var albedo = resolved.albedo;
        var roughness = clamp(resolved.roughness, 0.04, 1.0);
        let metallic = resolved.metallic;

        if resolved.noise_channels != 0u && resolved.noise_scale > 0.0 && resolved.noise_strength > 0.0 {
            let noise_pos = hit_pos * resolved.noise_scale;
            let n = simplex3d(noise_pos);
            let ns = n * resolved.noise_strength;
            if (resolved.noise_channels & NOISE_CHANNEL_ALBEDO) != 0u {
                albedo = clamp(albedo + albedo * ns, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            if (resolved.noise_channels & NOISE_CHANNEL_ROUGHNESS) != 0u {
                roughness = clamp(roughness + ns * 0.5, 0.04, 1.0);
            }
            if (resolved.noise_channels & NOISE_CHANNEL_NORMAL) != 0u {
                let eps = 0.01 / resolved.noise_scale;
                let dnx = simplex3d(noise_pos + vec3<f32>(eps, 0.0, 0.0)) - simplex3d(noise_pos - vec3<f32>(eps, 0.0, 0.0));
                let dny = simplex3d(noise_pos + vec3<f32>(0.0, eps, 0.0)) - simplex3d(noise_pos - vec3<f32>(0.0, eps, 0.0));
                let dnz = simplex3d(noise_pos + vec3<f32>(0.0, 0.0, eps)) - simplex3d(noise_pos - vec3<f32>(0.0, 0.0, eps));
                let bump = vec3<f32>(dnx, dny, dnz) * resolved.noise_strength * 0.5;
                normal = normalize(normal + bump);
            }
        }

        let f0 = mix(vec3<f32>(0.04), albedo, metallic);
        let behind_view = view_dir;
        let n_dot_v = max(dot(normal, behind_view), 0.001);
        var lit = vec3<f32>(0.0);

        for (var li = 0u; li < shade_uniforms.num_lights; li++) {
            let light = lights[li];
            if light.light_type != LIGHT_TYPE_DIRECTIONAL { continue; }

            let light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let radiance = vec3<f32>(light.color_r, light.color_g, light.color_b) * light.intensity;
            let n_dot_l = max(dot(normal, light_dir), 0.0);
            let half_vec = normalize(behind_view + light_dir);
            let n_dot_h = max(dot(normal, half_vec), 0.0);
            let h_dot_v = max(dot(half_vec, behind_view), 0.0);

            let d = distribution_ggx(n_dot_h, roughness);
            let v = visibility_smith_ggx(n_dot_v, n_dot_l, roughness);
            let f = fresnel_schlick(h_dot_v, f0);
            let ks = f;
            let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

            var shadow = 1.0;
            if light.shadow_caster == 1u {
                let so = hit_pos + normal * SHADOW_BIAS;
                shadow = hard_shadow(so, light_dir, SHADOW_MAX_DIST * 0.5);
            }

            lit += (kd * albedo / PI + d * v * f) * radiance * n_dot_l * shadow;
        }

        let ao = sdf_ao(hit_pos + normal * SHADOW_BIAS, normal);

        var ambient_col: vec3<f32>;
        if shade_uniforms.sky_params.z > 0.5 {
            let sun_d = normalize(shade_uniforms.sun_dir.xyz);
            ambient_col = atmosphere_sky(vec3<f32>(0.0, 1.0, 0.0), sun_d) * 0.1;
        } else {
            ambient_col = AMBIENT_COLOR;
        }
        let kd_ambient = 1.0 - metallic;
        lit += ambient_col * albedo * ao * 0.15 * kd_ambient;

        lit += resolved.emission * resolved.emission_strength;

        result.color = lit * total_absorption;
        result.hit = true;
        return result;

    } // end layer loop

    // Exhausted all glass layers — return sky through accumulated absorption
    if shade_uniforms.sky_params.z > 0.5 {
        result.color = atmosphere_sky(current_dir, normalize(shade_uniforms.sun_dir.xyz)) * total_absorption;
    } else {
        let uv_y = clamp(current_dir.y * 0.5 + 0.5, 0.0, 1.0);
        result.color = mix(SKY_HORIZON, SKY_ZENITH, uv_y) * total_absorption;
    }
    result.hit = true;
    return result;
}

// ---------- Brush Overlay Sampling ----------

/// Read a single brush overlay voxel. Returns geodesic distance or -1.0 if unmapped.
fn read_brush_overlay_voxel(vc: vec3<i32>, dims: vec3<u32>, bm_offset: u32) -> f32 {
    let max_v = vec3<i32>(dims) * 8 - vec3<i32>(1);
    if any(vc < vec3<i32>(0)) || any(vc > max_v) {
        return -1.0;
    }
    let brick = vec3<u32>(vc / vec3<i32>(8));
    let lv = vec3<u32>(vc % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[bm_offset + flat_brick];
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return -1.0;
    }
    let overlay_slot = brush_overlay_map[slot];
    if overlay_slot == EMPTY_SLOT {
        return -1.0;
    }
    let vi = lv.x + lv.y * 8u + lv.z * 64u;
    return brush_overlay_data[overlay_slot * 512u + vi];
}

/// Sample the brush overlay geodesic distance at a local-space position on a voxelized object.
/// Uses trilinear interpolation for smooth sub-voxel results.
/// Returns the geodesic distance from the brush center, or -1.0 if not in the overlay.
fn sample_brush_overlay(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    if brush_overlay.brush_active == 0u || obj.object_id != brush_overlay.brush_object_id {
        return -1.0;
    }
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;
    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return -1.0;
    }
    // Continuous voxel coordinate, shifted to voxel centers.
    let voxel_coord = grid_pos / vs - vec3<f32>(0.5);
    let vc0 = vec3<i32>(floor(voxel_coord));
    let frac = voxel_coord - vec3<f32>(vc0); // 0..1 within the cell

    let bm_offset = obj.brick_map_offset;

    // Sample 8 corners of the trilinear cell.
    let d000 = read_brush_overlay_voxel(vc0 + vec3<i32>(0, 0, 0), dims, bm_offset);
    let d100 = read_brush_overlay_voxel(vc0 + vec3<i32>(1, 0, 0), dims, bm_offset);
    let d010 = read_brush_overlay_voxel(vc0 + vec3<i32>(0, 1, 0), dims, bm_offset);
    let d110 = read_brush_overlay_voxel(vc0 + vec3<i32>(1, 1, 0), dims, bm_offset);
    let d001 = read_brush_overlay_voxel(vc0 + vec3<i32>(0, 0, 1), dims, bm_offset);
    let d101 = read_brush_overlay_voxel(vc0 + vec3<i32>(1, 0, 1), dims, bm_offset);
    let d011 = read_brush_overlay_voxel(vc0 + vec3<i32>(0, 1, 1), dims, bm_offset);
    let d111 = read_brush_overlay_voxel(vc0 + vec3<i32>(1, 1, 1), dims, bm_offset);

    // Count valid samples. If fewer than 2, not enough data for interpolation.
    var valid_count = 0u;
    var valid_sum = 0.0;
    if d000 >= 0.0 { valid_count += 1u; valid_sum += d000; }
    if d100 >= 0.0 { valid_count += 1u; valid_sum += d100; }
    if d010 >= 0.0 { valid_count += 1u; valid_sum += d010; }
    if d110 >= 0.0 { valid_count += 1u; valid_sum += d110; }
    if d001 >= 0.0 { valid_count += 1u; valid_sum += d001; }
    if d101 >= 0.0 { valid_count += 1u; valid_sum += d101; }
    if d011 >= 0.0 { valid_count += 1u; valid_sum += d011; }
    if d111 >= 0.0 { valid_count += 1u; valid_sum += d111; }

    if valid_count == 0u {
        return -1.0;
    }

    // Replace missing samples with the average of valid ones (extrapolate at edges).
    let fallback = valid_sum / f32(valid_count);
    let s000 = select(fallback, d000, d000 >= 0.0);
    let s100 = select(fallback, d100, d100 >= 0.0);
    let s010 = select(fallback, d010, d010 >= 0.0);
    let s110 = select(fallback, d110, d110 >= 0.0);
    let s001 = select(fallback, d001, d001 >= 0.0);
    let s101 = select(fallback, d101, d101 >= 0.0);
    let s011 = select(fallback, d011, d011 >= 0.0);
    let s111 = select(fallback, d111, d111 >= 0.0);

    // Trilinear interpolation.
    let fx = frac.x;
    let fy = frac.y;
    let fz = frac.z;
    let c00 = mix(s000, s100, fx);
    let c10 = mix(s010, s110, fx);
    let c01 = mix(s001, s101, fx);
    let c11 = mix(s011, s111, fx);
    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);
    return mix(c0, c1, fz);
}
