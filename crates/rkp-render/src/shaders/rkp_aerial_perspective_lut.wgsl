// Aerial Perspective LUT — atmospheric haze for distant geometry.
//
// 32×32×32 rgba16float 3D texture. XY = screen UV, Z = distance along view ray
// under an exponential mapping: slice z stores the camera-to-d(z) integral where
//   d(z) = AP_NEAR * (AP_FAR/AP_NEAR)^((z+0.5)/N).
// This packs near-field precision (sub-metre to tens of metres) into the first
// ~half of the slices and sweeps the remainder out to ~100 km to meet the sky
// at the horizon. A linear mapping wasted 99% of the LUT on distances beyond
// the scene; this one actually serves voxel-scale geometry.
//
// RGB = inscattered luminance, A = averaged transmittance.
// Applied to geometry pixels: final = inscattered + surface_color * transmittance.

const PI: f32 = 3.14159265;
const EARTH_RADIUS: f32 = 6360000.0;
const ATMO_RADIUS: f32 = 6460000.0;
const RAYLEIGH_SCALE_H: f32 = 8000.0;
const MIE_SCALE_H: f32 = 1200.0;
const BETA_R: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
const BETA_M_SCAT: vec3<f32> = vec3<f32>(3.996e-6, 3.996e-6, 3.996e-6);
const BETA_M_EXT: vec3<f32> = vec3<f32>(4.44e-6, 4.44e-6, 4.44e-6);
const BETA_OZONE: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);
const MIE_G: f32 = 0.8;
// Exponential slice parameterization — keep in sync with rkp_shade.wgsl.
// With (1 m, 128 km) the 32 slices fall at roughly
// 1.2 m, 1.7 m, 2.5 m, …, 500 m, 10 km, 109 km — one slice per ~1.45× distance.
const AP_NEAR_DISTANCE: f32 = 1.0;
const AP_FAR_DISTANCE: f32 = 128000.0;
const AP_SLICE_COUNT: f32 = 32.0;

struct AerialParams {
    sun_dir: vec3<f32>,
    sun_intensity: f32,
    camera_altitude: f32,
    // Ground albedo — stored here for layout parity with the sky-view LUT's
    // uniform buffer; unused by this shader since AP samples geometry pixels.
    ground_albedo_r: f32,
    ground_albedo_g: f32,
    ground_albedo_b: f32,
    cam_forward: vec3<f32>,
    _pad3: f32,
    cam_right: vec3<f32>,
    _pad4: f32,
    cam_up: vec3<f32>,
    _pad5: f32,
}

// --- Bindings ---

@group(0) @binding(0) var<uniform> params: AerialParams;
@group(0) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(2) var multiscatter_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var ap_out: texture_storage_3d<rgba16float, write>;

// --- Shared functions (same as sky view) ---

fn sample_extinction(altitude: f32) -> vec3<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);
    let h_km = altitude / 1000.0;
    var density_o = 0.0;
    if h_km < 25.0 { density_o = max(h_km / 15.0 - 2.0 / 3.0, 0.0); }
    else { density_o = max(-h_km / 15.0 + 8.0 / 3.0, 0.0); }
    return density_r * BETA_R + density_m * BETA_M_EXT + density_o * BETA_OZONE;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn cornette_shanks(cos_theta: f32, g: f32) -> f32 {
    let k = (3.0 / (8.0 * PI)) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-6), 1.5);
}

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

fn lookup_transmittance(view_height: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_params_to_uv(view_height, cos_zenith);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

fn lookup_multiscatter(view_height: f32, sun_cos_zenith: f32) -> vec3<f32> {
    let v = clamp((view_height - EARTH_RADIUS) / (ATMO_RADIUS - EARTH_RADIUS), 0.0, 1.0);
    let u = clamp(sun_cos_zenith * 0.5 + 0.5, 0.0, 1.0);
    return textureSampleLevel(multiscatter_lut, lut_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(ap_out);
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) || gid.z >= u32(dims.z) { return; }

    // Screen UV from XY.
    let screen_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(vec2<u32>(dims.x, dims.y));
    let ndc = vec2<f32>(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0);

    // Reconstruct view ray.
    let ray_dir = normalize(params.cam_forward + ndc.x * params.cam_right + ndc.y * params.cam_up);

    // Distance for this slice — exponential mapping centered on the slice so
    // that linear trilinear sampling in the shade pass lands on slice centers.
    let slice_u = (f32(gid.z) + 0.5) / AP_SLICE_COUNT;
    let max_dist = AP_NEAR_DISTANCE * pow(AP_FAR_DISTANCE / AP_NEAR_DISTANCE, slice_u);

    let origin = vec3<f32>(0.0, EARTH_RADIUS + params.camera_altitude, 0.0);
    let cos_sun = dot(ray_dir, params.sun_dir);
    let phase_r = rayleigh_phase(cos_sun);
    let phase_m = cornette_shanks(cos_sun, MIE_G);

    // Step count scales with slice index so far slices (tens of km) still
    // resolve the Mie scale height, while near slices stay cheap.
    let steps = clamp(u32(4.0 + f32(gid.z)), 4u, 24u);
    let step_len = max_dist / f32(steps);

    var throughput = vec3<f32>(1.0);
    var scatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = (f32(i) + 0.5) * step_len;
        let pos = origin + ray_dir * t;
        let altitude = length(pos) - EARTH_RADIUS;
        if altitude < 0.0 { break; }

        let extinction = sample_extinction(altitude);
        let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
        let density_m = exp(-altitude / MIE_SCALE_H);
        let scattering = density_r * BETA_R + density_m * BETA_M_SCAT;
        let sample_transmittance = exp(-extinction * step_len);

        let pos_up = pos / length(pos);
        let sun_cos_at_pos = dot(pos_up, params.sun_dir);
        let sun_trans = lookup_transmittance(length(pos), sun_cos_at_pos);
        let earth_shadow = select(0.0, 1.0, sun_cos_at_pos > -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (length(pos) * length(pos)), 0.0)));

        let ss = (density_r * BETA_R * phase_r + density_m * BETA_M_SCAT * phase_m)
               * earth_shadow * sun_trans * params.sun_intensity;
        let ms = lookup_multiscatter(length(pos), sun_cos_at_pos) * scattering * params.sun_intensity;

        let s_total = ss + ms;
        let s_int = (s_total - s_total * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
        scatter += throughput * s_int;

        throughput *= sample_transmittance;
    }

    // Average transmittance (single channel for simpler blending).
    let avg_trans = (throughput.r + throughput.g + throughput.b) / 3.0;

    textureStore(ap_out, vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)), vec4<f32>(scatter, avg_trans));
}
