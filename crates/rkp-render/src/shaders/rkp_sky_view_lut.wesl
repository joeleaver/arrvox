// Sky View LUT — precomputed sky radiance per frame.
//
// 192×108 rgba16float. Parameterized by (view zenith angle, view-sun azimuth).
// Non-linear mapping concentrates resolution near the horizon.
// Shade pass samples this instead of per-pixel 32-step atmosphere march.

import package::lib::atmosphere::PI;
import package::lib::atmosphere::EARTH_RADIUS;
import package::lib::atmosphere::ATMO_RADIUS;
import package::lib::atmosphere::RAYLEIGH_SCALE_H;
import package::lib::atmosphere::MIE_SCALE_H;
import package::lib::atmosphere::BETA_R;
import package::lib::atmosphere::BETA_M_SCAT;
import package::lib::atmosphere::sample_extinction;
import package::lib::atmosphere::transmittance_params_to_uv;
import package::lib::atmosphere::rayleigh_phase;
import package::lib::atmosphere::cornette_shanks;
import package::lib::math::ray_sphere;

const MIE_G: f32 = 0.8;

// Layout shared with the CPU-side AtmosphereFrameParams. The three scalars
// after camera_altitude sit at unaligned offset 20; WGSL can't put a vec3 there
// (align 16), so keep the albedo as three f32s and recombine in code.
struct SkyViewParams {
    sun_dir: vec3<f32>,
    sun_intensity: f32,
    camera_altitude: f32,
    ground_albedo_r: f32,
    ground_albedo_g: f32,
    ground_albedo_b: f32,
}

// --- Bindings ---

@group(0) @binding(0) var<uniform> params: SkyViewParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var sky_view_out: texture_storage_2d<rgba16float, write>;

// --- File-local helpers (atmosphere math + ray_sphere come from lib) ---

fn lookup_transmittance(view_height: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_params_to_uv(view_height, cos_zenith);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

fn lookup_multiscatter(view_height: f32, sun_cos_zenith: f32) -> vec3<f32> {
    let v = clamp((view_height - EARTH_RADIUS) / (ATMO_RADIUS - EARTH_RADIUS), 0.0, 1.0);
    let u = clamp(sun_cos_zenith * 0.5 + 0.5, 0.0, 1.0);
    return textureSampleLevel(multiscatter_lut, lut_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// --- Sky View UV parameterization (Hillaire 2020) ---

/// Map UV [0,1]² to (view_zenith_cos, view_sun_cos) with non-linear horizon focus.
fn sky_view_uv_to_params(uv: vec2<f32>, cam_height: f32) -> vec2<f32> {
    // Horizon angle at camera height.
    let horizon_cos = -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (cam_height * cam_height), 0.0));
    let horizon_angle = acos(horizon_cos);

    var view_zenith_cos: f32;
    if uv.y >= 0.5 {
        // Above horizon (v=0.5 → horizon, v=1.0 → zenith).
        let coord = pow(2.0 * uv.y - 1.0, 2.0); // 0 at horizon, 1 at zenith
        view_zenith_cos = cos(horizon_angle * (1.0 - coord));
    } else {
        // Below horizon (v=0.5 → horizon, v=0.0 → nadir).
        let coord = pow(1.0 - 2.0 * uv.y, 2.0); // 0 at horizon, 1 at nadir
        let beta = PI - horizon_angle;
        view_zenith_cos = cos(horizon_angle + beta * coord);
    }

    // U axis: view-sun azimuth (sqrt mapping for sun-focused resolution).
    let light_view_cos = uv.x * uv.x * 2.0 - 1.0;

    return vec2<f32>(view_zenith_cos, light_view_cos);
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(sky_view_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let cam_height = EARTH_RADIUS + params.camera_altitude;
    let sky_params = sky_view_uv_to_params(uv, cam_height);
    let view_zenith_cos = sky_params.x;
    let light_view_cos = sky_params.y;

    // Reconstruct view direction from zenith cosine and light-view cosine.
    let sun_dir = params.sun_dir;
    let view_zenith_sin = sqrt(max(1.0 - view_zenith_cos * view_zenith_cos, 0.0));

    // Build a coordinate frame where up = (0,1,0) at camera position.
    // View direction in the plane containing up and sun:
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sun_zenith_cos = sun_dir.y;
    let sun_zenith_sin = sqrt(max(1.0 - sun_zenith_cos * sun_zenith_cos, 0.0));

    // Compute the azimuth angle from light_view_cos.
    let view_sun_azimuth = acos(clamp(light_view_cos, -1.0, 1.0));

    // View direction: rotate from up by zenith angle, then rotate azimuth relative to sun.
    let ray_dir = vec3<f32>(
        view_zenith_sin * sin(view_sun_azimuth),
        view_zenith_cos,
        view_zenith_sin * cos(view_sun_azimuth),
    );

    // Transform to world: align the frame so that sun is in the xz plane.
    // The sun direction projected to horizontal = (sun_dir.x, 0, sun_dir.z).
    let sun_horiz = vec2<f32>(sun_dir.x, sun_dir.z);
    let sun_horiz_len = length(sun_horiz);
    var world_ray = ray_dir;
    if sun_horiz_len > 0.001 {
        let sun_h = sun_horiz / sun_horiz_len;
        // Rotate ray_dir.xz by the sun's horizontal angle.
        let rx = ray_dir.x * sun_h.y + ray_dir.z * sun_h.x;
        let rz = -ray_dir.x * sun_h.x + ray_dir.z * sun_h.y;
        world_ray = vec3<f32>(rx, ray_dir.y, rz);
    }

    // --- Atmosphere march (same as rkp_shade.wgsl atmosphere()) ---
    let origin = vec3<f32>(0.0, cam_height, 0.0);
    let atmo_hit = ray_sphere(origin, world_ray, ATMO_RADIUS);
    if atmo_hit.y < 0.0 {
        textureStore(sky_view_out, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let t_start = max(atmo_hit.x, 0.0);
    var t_end = atmo_hit.y;
    let earth_hit = ray_sphere(origin, world_ray, EARTH_RADIUS);
    if earth_hit.x > 0.0 { t_end = min(t_end, earth_hit.x); }

    let cos_sun = dot(world_ray, sun_dir);
    let phase_r = rayleigh_phase(cos_sun);
    let phase_m = cornette_shanks(cos_sun, MIE_G);

    let steps = 32u;
    let step_len = (t_end - t_start) / f32(steps);
    var throughput = vec3<f32>(1.0);
    var scatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = t_start + (f32(i) + 0.5) * step_len;
        let pos = origin + world_ray * t;
        let altitude = length(pos) - EARTH_RADIUS;
        if altitude < 0.0 { break; }

        let extinction = sample_extinction(altitude);
        let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
        let density_m = exp(-altitude / MIE_SCALE_H);
        let scattering = density_r * BETA_R + density_m * BETA_M_SCAT;
        let sample_transmittance = exp(-extinction * step_len);

        let pos_up = pos / length(pos);
        let sun_cos_at_pos = dot(pos_up, sun_dir);
        let sun_trans = lookup_transmittance(length(pos), sun_cos_at_pos);
        let earth_shadow = select(0.0, 1.0, sun_cos_at_pos > -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (length(pos) * length(pos)), 0.0)));

        let ss = (density_r * BETA_R * phase_r + density_m * BETA_M_SCAT * phase_m)
               * earth_shadow * sun_trans * params.sun_intensity;
        let ms = lookup_multiscatter(length(pos), sun_cos_at_pos) * scattering * params.sun_intensity;

        let s_total = ss + ms;
        let s_int = (s_total - s_total * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
        scatter += throughput * s_int;

        throughput *= sample_transmittance;
        if all(throughput < vec3<f32>(0.001)) { break; }
    }

    // Ground radiance — below-horizon rays that intersect the Earth sphere get
    // the light reflected at the hit point (direct sun + multi-scattered sky),
    // attenuated by the view-ray transmittance. Without this, rays that look
    // at the Earth surface return pure atmospheric in-scatter and appear as a
    // dark band past the geometric horizon. Hillaire 2020 §5.5.
    if earth_hit.x > 0.0 {
        let p_ground = origin + world_ray * earth_hit.x;
        let r_ground = length(p_ground);
        let normal_ground = p_ground / r_ground;
        let sun_cos_ground = dot(normal_ground, sun_dir);
        let sun_trans_ground = lookup_transmittance(r_ground, sun_cos_ground);
        let direct = sun_trans_ground * max(sun_cos_ground, 0.0) * params.sun_intensity;
        let ambient = lookup_multiscatter(r_ground, sun_cos_ground) * params.sun_intensity;
        let ground_albedo = vec3<f32>(
            params.ground_albedo_r, params.ground_albedo_g, params.ground_albedo_b,
        );
        let ground_L = (ground_albedo / PI) * (direct + ambient);
        scatter += throughput * ground_L;
    }

    textureStore(sky_view_out, vec2<i32>(gid.xy), vec4<f32>(scatter, 1.0));
}
