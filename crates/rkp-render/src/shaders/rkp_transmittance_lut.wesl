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

import package::lib::atmosphere::EARTH_RADIUS;
import package::lib::atmosphere::ATMO_RADIUS;
import package::lib::atmosphere::sample_extinction;
import package::lib::atmosphere::uv_to_transmittance_params;

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
