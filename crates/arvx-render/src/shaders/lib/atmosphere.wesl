// Shared atmosphere constants + pure-math helpers (Bruneton 2017 /
// Hillaire 2020 references). Includes the LUT-sampling lookups,
// which take their texture + sampler as parameters so consumers
// can wire their own bind-group-bound LUT and sampler in (the
// binding numbers vary per pass).

import package::lib::math::PI;

// --- Earth + atmosphere geometry ---
const EARTH_RADIUS: f32 = 6360000.0;       // meters
const ATMO_RADIUS:  f32 = 6460000.0;       // Earth + 100 km

// --- Scattering coefficients (per-meter, sea-level) ---
const BETA_R:      vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6); // Rayleigh scattering
const BETA_M_SCAT: vec3<f32> = vec3<f32>(3.996e-6, 3.996e-6, 3.996e-6); // Mie scattering
const BETA_M_EXT:  vec3<f32> = vec3<f32>(4.44e-6,  4.44e-6,  4.44e-6);  // Mie extinction (scat + abs)
const BETA_OZONE:  vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6); // Ozone absorption

// --- Density scale heights ---
const RAYLEIGH_SCALE_H: f32 = 8000.0;
const MIE_SCALE_H:      f32 = 1200.0;

// LUT lookups. The transmittance LUT stores ⟨view-height, sun-cos-zenith⟩ →
// extinction towards space; the multiscatter LUT stores ⟨view-height,
// sun-cos-zenith⟩ → second-order in-scatter (Hillaire 2020). Both are
// 2D, sampled with a linear sampler at level 0 — pass the texture +
// sampler in from the caller's bind group.

fn lookup_transmittance(
    transmittance_lut: texture_2d<f32>,
    s: sampler,
    view_height: f32,
    cos_zenith: f32,
) -> vec3<f32> {
    let uv = transmittance_params_to_uv(view_height, cos_zenith);
    return textureSampleLevel(transmittance_lut, s, uv, 0.0).rgb;
}

fn lookup_multiscatter(
    multiscatter_lut: texture_2d<f32>,
    s: sampler,
    view_height: f32,
    sun_cos_zenith: f32,
) -> vec3<f32> {
    let v = clamp((view_height - EARTH_RADIUS) / (ATMO_RADIUS - EARTH_RADIUS), 0.0, 1.0);
    let u = clamp(sun_cos_zenith * 0.5 + 0.5, 0.0, 1.0);
    return textureSampleLevel(multiscatter_lut, s, vec2<f32>(u, v), 0.0).rgb;
}

// Sample atmospheric extinction at a given altitude (Rayleigh + Mie + Ozone).
// `altitude` is meters above the planet centre? No — meters above sea level.
// (Density profiles use sea level as origin; ozone tent peaks at 25 km.)
fn sample_extinction(altitude: f32) -> vec3<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);

    // Ozone: tent profile peaking at 25 km.
    let h_km = altitude / 1000.0;
    var density_o = 0.0;
    if h_km < 25.0 {
        density_o = max(h_km / 15.0 - 2.0 / 3.0, 0.0);
    } else {
        density_o = max(-h_km / 15.0 + 8.0 / 3.0, 0.0);
    }

    return density_r * BETA_R       // Rayleigh scattering (= extinction, no absorption)
         + density_m * BETA_M_EXT   // Mie extinction (scattering + absorption)
         + density_o * BETA_OZONE;  // Ozone absorption
}

// --- UV ↔ physical parameter mapping (Bruneton 2017) for the
//     transmittance LUT. UV.x encodes ground-relative path through
//     the atmosphere; UV.y encodes view height above earth surface.

fn transmittance_params_to_uv(view_height: f32, cos_zenith: f32) -> vec2<f32> {
    let H = sqrt(ATMO_RADIUS * ATMO_RADIUS - EARTH_RADIUS * EARTH_RADIUS);
    let rho = sqrt(max(view_height * view_height - EARTH_RADIUS * EARTH_RADIUS, 0.0));
    let d_min = ATMO_RADIUS - view_height;
    let d_max = rho + H;
    let disc = view_height * view_height * (cos_zenith * cos_zenith - 1.0)
             + ATMO_RADIUS * ATMO_RADIUS;
    let d = max(-view_height * cos_zenith + sqrt(max(disc, 0.0)), 0.0);
    let u = clamp((d - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0);
    let v = clamp(rho / max(H, 1e-6), 0.0, 1.0);
    return vec2<f32>(u, v);
}

fn uv_to_transmittance_params(uv: vec2<f32>) -> vec2<f32> {
    let H = sqrt(ATMO_RADIUS * ATMO_RADIUS - EARTH_RADIUS * EARTH_RADIUS);
    let rho = H * uv.y;
    let view_height = sqrt(rho * rho + EARTH_RADIUS * EARTH_RADIUS);

    let d_min = ATMO_RADIUS - view_height;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);
    let cos_zenith = select(
        (H * H - rho * rho - d * d) / (2.0 * view_height * d),
        1.0,
        d == 0.0
    );

    return vec2<f32>(view_height, clamp(cos_zenith, -1.0, 1.0));
}

// --- Phase functions ---

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Cornette-Shanks phase function (Hillaire 2020 reference).
// More accurate than Henyey-Greenstein for Mie scattering.
fn cornette_shanks(cos_theta: f32, g: f32) -> f32 {
    let k = (3.0 / (8.0 * PI)) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-6), 1.5);
}
