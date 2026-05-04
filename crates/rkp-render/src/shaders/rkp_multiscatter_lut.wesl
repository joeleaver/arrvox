// Multi-Scattering LUT — Hillaire 2020 approximation.
//
// 32×32 rgba16float texture. Each texel stores the isotropic multi-scattered
// luminance per unit sun illuminance, as a function of (height, sunZenithCos).
//
// Algorithm (Hillaire 2020, Equations 5-10):
// 1. For 64 uniformly distributed directions on the sphere:
//    - Ray march with sun_illuminance=1 and isotropic phase (1/4π)
//    - Accumulate L_2nd (2nd-order scattered luminance) and f_ms (transfer fraction)
// 2. Sum via geometric series: Psi_ms = L_2nd / (1 - f_ms)
//
// This captures ALL orders of scattering in one pass, fixing the orange horizon
// that single-scattering produces at oblique viewing angles.

import package::lib::atmosphere::PI;
import package::lib::atmosphere::EARTH_RADIUS;
import package::lib::atmosphere::ATMO_RADIUS;
import package::lib::atmosphere::RAYLEIGH_SCALE_H;
import package::lib::atmosphere::MIE_SCALE_H;
import package::lib::atmosphere::BETA_R;
import package::lib::atmosphere::BETA_M_SCAT;
import package::lib::atmosphere::sample_extinction;
import package::lib::atmosphere::transmittance_params_to_uv;

const GROUND_ALBEDO: vec3<f32> = vec3<f32>(0.3, 0.3, 0.3);

const NUM_MARCH_STEPS: u32 = 20u;
const SQRT_SAMPLES: u32 = 8u; // 8×8 = 64 sphere directions

// --- Bindings ---

@group(0) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(1) var lut_sampler: sampler;
@group(0) @binding(2) var multiscatter_out: texture_storage_2d<rgba16float, write>;

// --- Shared memory for parallel reduction ---

var<workgroup> shared_l: array<vec3<f32>, 64>;
var<workgroup> shared_fms: array<vec3<f32>, 64>;

// --- Helpers ---

fn sample_scattering(altitude: f32) -> vec3<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);
    return density_r * BETA_R + density_m * BETA_M_SCAT;
}

fn ray_sphere_exit(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return -1.0; }
    return -b + sqrt(d);
}

fn ray_sphere_entry(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return -1.0; }
    return -b - sqrt(d);
}

fn lookup_transmittance(view_height: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_params_to_uv(view_height, cos_zenith);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

// --- Main ---

// Workgroup size: 1×1×64 — each thread handles one sphere direction,
// parallel reduction sums all 64 contributions.
@compute @workgroup_size(1, 1, 64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let dims = textureDimensions(multiscatter_out);
    let pixel = vec2<u32>(gid.x, gid.y);
    if pixel.x >= dims.x || pixel.y >= dims.y { return; }

    // Map pixel to (height, sunZenithCos).
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let view_height = EARTH_RADIUS + uv.y * (ATMO_RADIUS - EARTH_RADIUS);
    let sun_cos_zenith = uv.x * 2.0 - 1.0;

    // Sun direction from zenith cosine.
    let sun_sin_zenith = sqrt(max(1.0 - sun_cos_zenith * sun_cos_zenith, 0.0));
    let sun_dir = vec3<f32>(sun_sin_zenith, sun_cos_zenith, 0.0);

    // Generate uniformly distributed direction on the sphere for this thread.
    let sample_idx = local_idx;
    let ix = sample_idx % SQRT_SAMPLES;
    let iy = sample_idx / SQRT_SAMPLES;
    // Stratified uniform sphere sampling.
    let rand_a = (f32(ix) + 0.5) / f32(SQRT_SAMPLES);
    let rand_b = (f32(iy) + 0.5) / f32(SQRT_SAMPLES);
    let cos_theta = 1.0 - 2.0 * rand_b;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * rand_a;
    let ray_dir = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));

    let origin = vec3<f32>(0.0, view_height, 0.0);

    // Find atmosphere exit.
    let t_max_atmo = ray_sphere_exit(origin, ray_dir, ATMO_RADIUS);
    // Check ground intersection.
    let t_ground = ray_sphere_entry(origin, ray_dir, EARTH_RADIUS);
    var t_max = t_max_atmo;
    var hit_ground = false;
    if t_ground > 0.0 {
        t_max = t_ground;
        hit_ground = true;
    }

    if t_max <= 0.0 {
        shared_l[local_idx] = vec3<f32>(0.0);
        shared_fms[local_idx] = vec3<f32>(0.0);
        workgroupBarrier();
    } else {
        // Ray march with sun_illuminance = 1.0 and isotropic phase = 1/(4π).
        let dt = t_max / f32(NUM_MARCH_STEPS);
        var throughput = vec3<f32>(1.0);
        var l_accum = vec3<f32>(0.0);
        var fms_accum = vec3<f32>(0.0);

        for (var i = 0u; i < NUM_MARCH_STEPS; i++) {
            let t = (f32(i) + 0.5) * dt;
            let pos = origin + ray_dir * t;
            let altitude = length(pos) - EARTH_RADIUS;
            if altitude < 0.0 { break; }

            let extinction = sample_extinction(altitude);
            let scattering = sample_scattering(altitude);
            let sample_transmittance = exp(-extinction * dt);

            // Transmittance from this point to the sun.
            let up = pos / length(pos);
            let sun_cos = dot(up, sun_dir);
            let sun_trans = lookup_transmittance(length(pos), sun_cos);

            // Check earth shadow.
            let earth_shadow = select(1.0, 0.0, ray_sphere_entry(pos, sun_dir, EARTH_RADIUS) > 0.0);

            // Multi-scattering from previous LUT (for iterative refinement, use 0 for first pass).
            // For single-pass approximation, we set this to 0.
            let ms_prev = vec3<f32>(0.0);

            // In-scattered luminance (isotropic phase = 1/4π, sun_illuminance = 1).
            let s = earth_shadow * sun_trans * scattering / (4.0 * PI)
                  + ms_prev * scattering;

            // Integrate analytically within step (Hillaire Eq. 3).
            let s_int = (s - s * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
            l_accum += throughput * s_int;

            // Transfer fraction: how much scattering occurs per bounce.
            let ms = scattering;
            let ms_int = (ms - ms * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
            fms_accum += throughput * ms_int;

            throughput *= sample_transmittance;
        }

        // Ground contribution (if ray hits ground).
        if hit_ground {
            let ground_pos = origin + ray_dir * t_max;
            let up = ground_pos / length(ground_pos);
            let sun_cos = dot(up, sun_dir);
            let earth_shadow = select(1.0, 0.0, sun_cos < 0.0);
            let sun_trans = lookup_transmittance(length(ground_pos), sun_cos);
            // Ground reflects light diffusely (Lambertian).
            l_accum += throughput * GROUND_ALBEDO * earth_shadow * sun_trans * max(sun_cos, 0.0) / PI;
        }

        shared_l[local_idx] = l_accum;
        shared_fms[local_idx] = fms_accum;
        workgroupBarrier();
    }

    // Parallel reduction — sum all 64 thread contributions.
    // Only power-of-2 reductions.
    var stride = 32u;
    while stride > 0u {
        if local_idx < stride {
            shared_l[local_idx] += shared_l[local_idx + stride];
            shared_fms[local_idx] += shared_fms[local_idx + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    // Thread 0 writes the final result.
    if local_idx == 0u {
        // Monte Carlo sphere average: integral of f(ω)×(1/4π) dω ≈ (1/N)×Σf(ωi).
        // Each thread contributes one direction; the mean gives the sphere average.
        let inv_n = 1.0 / f32(SQRT_SAMPLES * SQRT_SAMPLES); // 1/64
        let l_2nd = shared_l[0] * inv_n;
        let f_ms = shared_fms[0] * inv_n;

        // Geometric series: Psi_ms = L_2nd / (1 - f_ms)
        let psi_ms = l_2nd / max(vec3<f32>(1.0) - f_ms, vec3<f32>(0.001));

        textureStore(multiscatter_out, vec2<i32>(pixel), vec4<f32>(psi_ms, 1.0));
    }
}
