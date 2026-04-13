// RKIPatch deferred PBR shading compute shader.
//
// Reads G-buffer (position, normal+shadow, material) + SSAO texture.
// Evaluates PBR Cook-Torrance BRDF with direct lighting.
// Writes final HDR color to output texture.

const PI: f32 = 3.14159265;

// --- Structs ---

struct CameraUniforms {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec4<f32>,   // xyz = position/direction, w = type (0=dir, 1=point, 2=spot)
    color: vec4<f32>,      // rgb = color, w = intensity
    direction: vec4<f32>,  // xyz = spot direction, w = spot angle
    params: vec4<f32>,     // x = range, y = inner_angle, z = shadow_softness, w = cast_shadow
}

struct ShadeParams {
    num_lights: u32,
    ambient_intensity: f32,
    camera_altitude: f32,
    sun_intensity: f32,
    sky_color_top: vec3<f32>,
    _pad0: f32,
    sky_color_horizon: vec3<f32>,
    _pad1: f32,
    sun_dir: vec3<f32>,
    _pad2: f32,
    ambient_color: vec3<f32>,
    _pad3: f32,
}

struct Material {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emission_strength: f32,
    opacity: f32,
}

// --- Bindings ---

// Group 0: G-buffer (read)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;

// Group 1: shadow texture (read, full-res) + SSAO texture (read, half-res)
@group(1) @binding(0) var shadow_tex: texture_2d<f32>;
@group(1) @binding(1) var ssao_tex: texture_2d<f32>;

// Group 2: output HDR color (write, full-res)
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Group 3: shading params + lights + materials
@group(3) @binding(0) var<uniform> shade_params: ShadeParams;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
@group(3) @binding(2) var<storage, read> materials: array<Material>;

// Group 4: camera
@group(4) @binding(0) var<uniform> camera: CameraUniforms;

// Group 5: atmosphere LUTs
@group(5) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(5) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(5) @binding(2) var atmo_sampler: sampler;

// --- Atmospheric scattering ---

// Atmosphere constants — Bruneton 2017 / Hillaire 2020 reference values.
const EARTH_RADIUS: f32 = 6360000.0;
const ATMO_RADIUS: f32 = 6460000.0;      // Earth + 100km
const RAYLEIGH_SCALE_H: f32 = 8000.0;
const MIE_SCALE_H: f32 = 1200.0;
const BETA_R: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
const BETA_M_SCAT: vec3<f32> = vec3<f32>(3.996e-6, 3.996e-6, 3.996e-6);
const BETA_M_EXT: vec3<f32> = vec3<f32>(4.44e-6, 4.44e-6, 4.44e-6);
const BETA_OZONE: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);
const MIE_G: f32 = 0.8;
const SUN_ANGULAR_RADIUS: f32 = 0.004675;

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(d);
    return vec2<f32>(-b - s, -b + s);
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

/// Cornette-Shanks phase function (Hillaire 2020 reference).
/// More accurate than Henyey-Greenstein for Mie scattering.
fn cornette_shanks(cos_theta: f32, g: f32) -> f32 {
    let k = (3.0 / (8.0 * PI)) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-6), 1.5);
}

/// Sample atmospheric extinction at a given altitude (Rayleigh + Mie + Ozone).
fn sample_extinction(altitude: f32) -> vec3<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);
    let h_km = altitude / 1000.0;
    var density_o = 0.0;
    if h_km < 25.0 {
        density_o = max(h_km / 15.0 - 2.0 / 3.0, 0.0);
    } else {
        density_o = max(-h_km / 15.0 + 8.0 / 3.0, 0.0);
    }
    return density_r * BETA_R + density_m * BETA_M_EXT + density_o * BETA_OZONE;
}

fn sample_scattering(altitude: f32) -> vec2<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);
    return vec2<f32>(density_r, density_m);
}

/// UV mapping for transmittance LUT lookup (Bruneton 2017).
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

/// Look up transmittance from the precomputed LUT.
fn lookup_transmittance(view_height: f32, cos_zenith: f32) -> vec3<f32> {
    let uv = transmittance_params_to_uv(view_height, cos_zenith);
    return textureSampleLevel(transmittance_lut, atmo_sampler, uv, 0.0).rgb;
}

/// Look up multi-scattered luminance from the precomputed LUT.
fn lookup_multiscatter(view_height: f32, sun_cos_zenith: f32) -> vec3<f32> {
    let v = clamp((view_height - EARTH_RADIUS) / (ATMO_RADIUS - EARTH_RADIUS), 0.0, 1.0);
    let u = clamp(sun_cos_zenith * 0.5 + 0.5, 0.0, 1.0);
    return textureSampleLevel(multiscatter_lut, atmo_sampler, vec2<f32>(u, v), 0.0).rgb;
}

/// Atmosphere sky radiance using precomputed LUTs.
fn atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let origin = vec3<f32>(0.0, EARTH_RADIUS + camera_alt, 0.0);
    let atmo_hit = ray_sphere(origin, ray_dir, ATMO_RADIUS);
    if atmo_hit.y < 0.0 { return vec3<f32>(0.0); }

    let t_start = max(atmo_hit.x, 0.0);
    var t_end = atmo_hit.y;

    // Clip to earth surface.
    let earth_hit = ray_sphere(origin, ray_dir, EARTH_RADIUS);
    if earth_hit.x > 0.0 { t_end = min(t_end, earth_hit.x); }

    let cos_sun = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_sun);
    let phase_m = cornette_shanks(cos_sun, MIE_G);

    let steps = 32u;
    let step_len = (t_end - t_start) / f32(steps);
    var throughput = vec3<f32>(1.0);
    var scatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = t_start + (f32(i) + 0.5) * step_len;
        let pos = origin + ray_dir * t;
        let altitude = length(pos) - EARTH_RADIUS;
        if altitude < 0.0 { break; }

        let extinction = sample_extinction(altitude);
        let densities = sample_scattering(altitude);
        let density_r = densities.x;
        let density_m = densities.y;
        let scattering = density_r * BETA_R + density_m * BETA_M_SCAT;

        let sample_transmittance = exp(-extinction * step_len);

        // Sun transmittance from LUT (replaces per-pixel secondary march).
        let up = pos / length(pos);
        let sun_cos = dot(up, sun_dir);
        let sun_trans = lookup_transmittance(length(pos), sun_cos);

        // Earth shadow check.
        let earth_shadow = select(0.0, 1.0, sun_cos > -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (length(pos) * length(pos)), 0.0)));

        // Single-scattering: phase-weighted.
        let ss = (density_r * BETA_R * phase_r + density_m * BETA_M_SCAT * phase_m)
               * earth_shadow * sun_trans * sun_intensity;

        // Multi-scattering from LUT (replaces inline fms hack).
        let ms = lookup_multiscatter(length(pos), sun_cos) * scattering * sun_intensity;

        // Integrate analytically within step (more accurate than Euler).
        let s_total = ss + ms;
        let s_int = (s_total - s_total * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
        scatter += throughput * s_int;

        throughput *= sample_transmittance;
        if all(throughput < vec3<f32>(0.001)) { break; }
    }

    return scatter;
}

/// Sun disc with atmospheric extinction from LUT.
fn sun_disc(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let cos_angle = dot(ray_dir, sun_dir);
    if cos_angle < cos(SUN_ANGULAR_RADIUS * 3.0) { return vec3<f32>(0.0); }

    // Sun transmittance from LUT (single texture sample).
    let view_height = EARTH_RADIUS + camera_alt;
    let sun_cos = dot(vec3<f32>(0.0, 1.0, 0.0), sun_dir);
    let sun_transmittance = lookup_transmittance(view_height, sun_cos);

    // Sun disc with soft edge.
    let sun_cos_r = cos(SUN_ANGULAR_RADIUS);
    let glow_cos = cos(SUN_ANGULAR_RADIUS * 3.0);
    if cos_angle > sun_cos_r {
        let limb = 1.0 - 0.3 * (1.0 - (cos_angle - sun_cos_r) / (1.0 - sun_cos_r));
        return sun_transmittance * sun_intensity * limb;
    } else {
        let t = (cos_angle - glow_cos) / (sun_cos_r - glow_cos);
        return sun_transmittance * sun_intensity * t * t * 0.15;
    }
}

// --- PBR helpers ---

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gbuf_position);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let pos_data = textureLoad(gbuf_position, coord, 0);
    let world_pos = pos_data.xyz;
    let hit_t = pos_data.w;

    // No geometry → physically-based sky via atmospheric scattering.
    if hit_t >= 9999.0 || hit_t <= 0.0 {
        let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let ray_dir = normalize(camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz);

        let s_dir = normalize(shade_params.sun_dir);
        var sky = atmosphere(ray_dir, s_dir, shade_params.sun_intensity, shade_params.camera_altitude);
        sky += sun_disc(ray_dir, s_dir, shade_params.sun_intensity, shade_params.camera_altitude);

        textureStore(output, coord, vec4<f32>(sky, 1.0));
        return;
    }

    let normal = normalize(textureLoad(gbuf_normal, coord, 0).xyz);
    let mat_data = textureLoad(gbuf_material, coord, 0);
    let packed_r = mat_data.r;
    let packed_g = mat_data.g;

    // Unpack material IDs.
    let primary_mat_id = packed_r & 0xFFFFu;
    let blend_weight = f32(packed_g & 0xFFu) / 255.0;

    // Unpack RGB565 color from bits 16-31 of packed_g.
    let color_rgb565 = (packed_g >> 16u) & 0xFFFFu;
    var voxel_color = vec3<f32>(1.0); // default white (material color only)
    if color_rgb565 != 0u {
        let cr5 = color_rgb565 & 0x1Fu;
        let cg6 = (color_rgb565 >> 5u) & 0x3Fu;
        let cb5 = (color_rgb565 >> 11u) & 0x1Fu;
        voxel_color = vec3<f32>(f32(cr5) / 31.0, f32(cg6) / 63.0, f32(cb5) / 31.0);
    }

    // Resolve material.
    let mat = materials[primary_mat_id];
    let albedo = mat.base_color.rgb * voxel_color;
    let metallic = mat.metallic;
    let roughness = max(mat.roughness, 0.04);

    // View direction.
    let V = normalize(camera.position.xyz - world_pos);
    let N = normal;
    let n_dot_v = max(dot(N, V), 0.001);

    // F0 for dielectrics vs metals.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Per-light shadow from shadow texture (written by march pass).
    let shadow_data = textureLoad(shadow_tex, coord, 0);

    // AO from half-res SSAO texture.
    let half_coord = vec2<i32>(gid.xy) / 2;
    let ao = textureLoad(ssao_tex, half_coord, 0).r;

    // Accumulate direct lighting.
    var lo = vec3<f32>(0.0);
    var shadow_idx = 0u; // tracks which shadow channel to read

    for (var li = 0u; li < shade_params.num_lights; li++) {
        let light = lights[li];
        let light_type = u32(light.position.w);

        var L: vec3<f32>;
        var attenuation = 1.0;

        if light_type == 0u {
            // Directional light.
            L = normalize(-light.direction.xyz);
        } else {
            // Point/spot light.
            let to_light = light.position.xyz - world_pos;
            let dist = length(to_light);
            L = to_light / dist;
            let range = light.params.x;
            if range > 0.0 {
                attenuation = max(1.0 - (dist / range), 0.0);
                attenuation *= attenuation;
            }
        }

        // Read per-light shadow BEFORE the n_dot_l skip — must stay in sync
        // with the march pass which always writes shadow for every shadow-casting light.
        var light_shadow = 1.0;
        let cast_shadow = light.params.w;
        if cast_shadow >= 0.5 && shadow_idx < 4u {
            light_shadow = shadow_data[shadow_idx];
            shadow_idx++;
        }

        let n_dot_l = max(dot(N, L), 0.0);
        if n_dot_l <= 0.0 { continue; }

        let H = normalize(V + L);
        let n_dot_h = max(dot(N, H), 0.0);
        let h_dot_v = max(dot(H, V), 0.0);

        // Cook-Torrance BRDF.
        let D = distribution_ggx(n_dot_h, roughness);
        let G = geometry_smith(n_dot_v, n_dot_l, roughness);
        let F = fresnel_schlick(h_dot_v, f0);

        let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);
        let kd = (1.0 - F) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;

        let radiance = light.color.rgb * light.color.w * attenuation;

        lo += (diffuse + specular) * radiance * n_dot_l * light_shadow;
    }

    // Ambient from multi-scattering LUT — replaces crude CPU 6-direction sampling.
    // The LUT gives isotropic scattered luminance per unit sun illuminance at
    // (camera_height, sun_zenith). This captures all scattering orders.
    let cam_height = EARTH_RADIUS + shade_params.camera_altitude;
    let sun_cos_z = shade_params.sun_dir.y; // sun_dir is toward-sun, .y = cos(zenith)
    let ms_irradiance = lookup_multiscatter(cam_height, sun_cos_z)
                      * shade_params.sun_intensity
                      * shade_params.ambient_intensity;

    let ambient_diffuse = ms_irradiance * albedo * (1.0 - metallic) * ao;

    // Ambient specular: approximate sky reflection for energy conservation.
    let F_env = fresnel_schlick(n_dot_v, f0);
    let ambient_specular = ms_irradiance * F_env * ao;
    let ambient = ambient_diffuse + ambient_specular;

    // Emission.
    let emission = mat.base_color.rgb * mat.emission_strength;

    let final_color = lo + ambient + emission;
    textureStore(output, coord, vec4<f32>(final_color, 1.0));
}
