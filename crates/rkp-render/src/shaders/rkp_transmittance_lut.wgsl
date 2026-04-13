// Transmittance LUT — precomputed atmospheric transmittance.
//
// 256×64 rgba16float texture. Each texel stores exp(-optical_depth) for a ray
// from a given height in a given direction through the atmosphere.
//
// Parameterization (Bruneton 2017):
//   U axis: view zenith cosine (non-linear mapping for horizon precision)
//   V axis: height above ground (non-linear mapping)
//
// Includes Rayleigh scattering, Mie extinction, and ozone absorption.

// --- Atmosphere constants (Hillaire 2020 reference values) ---

const EARTH_RADIUS: f32 = 6360000.0;       // meters
const ATMO_RADIUS: f32 = 6460000.0;        // Earth + 100km
const RAYLEIGH_SCALE_H: f32 = 8000.0;
const MIE_SCALE_H: f32 = 1200.0;
const BETA_R: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);  // Rayleigh scattering
const BETA_M_SCAT: vec3<f32> = vec3<f32>(3.996e-6, 3.996e-6, 3.996e-6);  // Mie scattering
const BETA_M_EXT: vec3<f32> = vec3<f32>(4.44e-6, 4.44e-6, 4.44e-6);     // Mie extinction (scat + abs)
const BETA_OZONE: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);  // Ozone absorption

const NUM_STEPS: u32 = 40u;

// --- Bindings ---

@group(0) @binding(0) var transmittance_out: texture_storage_2d<rgba16float, write>;

// --- Helpers ---

fn ray_sphere_exit(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return -1.0; }
    return -b + sqrt(d);
}

/// Sample atmospheric extinction (Rayleigh + Mie + Ozone) at a given altitude.
fn sample_extinction(altitude: f32) -> vec3<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);

    // Ozone: tent profile peaking at 25km.
    let h_km = altitude / 1000.0;
    var density_o = 0.0;
    if h_km < 25.0 {
        density_o = max(h_km / 15.0 - 2.0 / 3.0, 0.0);
    } else {
        density_o = max(-h_km / 15.0 + 8.0 / 3.0, 0.0);
    }

    return density_r * BETA_R    // Rayleigh scattering (= extinction, no absorption)
         + density_m * BETA_M_EXT  // Mie extinction (scattering + absorption)
         + density_o * BETA_OZONE; // Ozone absorption
}

// --- UV ↔ Physical parameter mapping (Bruneton 2017) ---

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

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(transmittance_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    // Map pixel to UV [0, 1].
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // UV → (view_height, cos_zenith).
    let params = uv_to_transmittance_params(uv);
    let view_height = params.x;
    let cos_zenith = params.y;

    // Ray origin at (0, view_height, 0), direction from zenith cosine.
    let sin_zenith = sqrt(max(1.0 - cos_zenith * cos_zenith, 0.0));
    let origin = vec3<f32>(0.0, view_height, 0.0);
    let dir = vec3<f32>(sin_zenith, cos_zenith, 0.0);

    // Find atmosphere exit distance.
    let t_max = ray_sphere_exit(origin, dir, ATMO_RADIUS);
    if t_max < 0.0 {
        textureStore(transmittance_out, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, 1.0));
        return;
    }

    // Ray march, accumulating optical depth.
    let dt = t_max / f32(NUM_STEPS);
    var optical_depth = vec3<f32>(0.0);

    for (var i = 0u; i < NUM_STEPS; i++) {
        let t = (f32(i) + 0.5) * dt;
        let pos = origin + dir * t;
        let altitude = length(pos) - EARTH_RADIUS;

        if altitude < 0.0 { break; } // Hit ground

        optical_depth += sample_extinction(altitude) * dt;
    }

    let transmittance = exp(-optical_depth);
    textureStore(transmittance_out, vec2<i32>(gid.xy), vec4<f32>(transmittance, 1.0));
}
