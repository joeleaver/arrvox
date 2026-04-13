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

// --- Atmospheric scattering ---

const EARTH_RADIUS: f32 = 6371000.0;
const ATMO_RADIUS: f32 = 6471000.0;
const RAYLEIGH_SCALE_H: f32 = 8000.0;
const MIE_SCALE_H: f32 = 1200.0;
const BETA_R: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);
const BETA_M: vec3<f32> = vec3<f32>(21e-6, 21e-6, 21e-6);
const MIE_G: f32 = 0.76;
const SUN_ANGULAR_RADIUS: f32 = 0.00465; // ~0.267 degrees

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

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-6), 1.5));
}

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
    let phase_m = henyey_greenstein(cos_sun, MIE_G);

    let steps = 16u;
    let step_len = (t_end - t_start) / f32(steps);
    var od_r = vec3<f32>(0.0); // optical depth Rayleigh (camera → sample)
    var od_m = vec3<f32>(0.0); // optical depth Mie
    var scatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = t_start + (f32(i) + 0.5) * step_len;
        let pos = origin + ray_dir * t;
        let alt = length(pos) - EARTH_RADIUS;

        let density_r = exp(-alt / RAYLEIGH_SCALE_H) * step_len;
        let density_m = exp(-alt / MIE_SCALE_H) * step_len;
        od_r += BETA_R * density_r;
        od_m += BETA_M * density_m;

        // Secondary ray: optical depth from sample to sun (atmosphere exit).
        let sun_hit = ray_sphere(pos, sun_dir, ATMO_RADIUS);
        let sun_steps = 4u;
        let sun_step_len = sun_hit.y / f32(sun_steps);
        var od_sun_r = vec3<f32>(0.0);
        var od_sun_m = vec3<f32>(0.0);
        var in_shadow = false;
        for (var j = 0u; j < sun_steps; j++) {
            let st = (f32(j) + 0.5) * sun_step_len;
            let sun_pos = pos + sun_dir * st;
            let sun_alt = length(sun_pos) - EARTH_RADIUS;
            if sun_alt < 0.0 { in_shadow = true; break; }
            let sd_r = exp(-sun_alt / RAYLEIGH_SCALE_H) * sun_step_len;
            let sd_m = exp(-sun_alt / MIE_SCALE_H) * sun_step_len;
            od_sun_r += BETA_R * sd_r;
            od_sun_m += BETA_M * sd_m;
        }
        if in_shadow { continue; }

        let transmittance = exp(-(od_r + od_m + od_sun_r + od_sun_m));
        scatter += (density_r * BETA_R * phase_r + density_m * BETA_M * phase_m)
                 * transmittance * sun_intensity;
    }

    return scatter;
}

fn sun_disc(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let cos_angle = dot(ray_dir, sun_dir);
    if cos_angle < cos(SUN_ANGULAR_RADIUS * 3.0) { return vec3<f32>(0.0); }

    // Compute sun transmittance (atmospheric extinction along sun direction).
    let origin = vec3<f32>(0.0, EARTH_RADIUS + camera_alt, 0.0);
    let sun_hit = ray_sphere(origin, sun_dir, ATMO_RADIUS);
    if sun_hit.y < 0.0 { return vec3<f32>(0.0); }
    let sun_steps = 8u;
    let sun_step_len = sun_hit.y / f32(sun_steps);
    var od_r = vec3<f32>(0.0);
    var od_m = vec3<f32>(0.0);
    for (var j = 0u; j < sun_steps; j++) {
        let st = (f32(j) + 0.5) * sun_step_len;
        let pos = origin + sun_dir * st;
        let alt = length(pos) - EARTH_RADIUS;
        if alt < 0.0 { return vec3<f32>(0.0); }
        od_r += BETA_R * exp(-alt / RAYLEIGH_SCALE_H) * sun_step_len;
        od_m += BETA_M * exp(-alt / MIE_SCALE_H) * sun_step_len;
    }
    let sun_transmittance = exp(-(od_r + od_m));

    // Sun disc with soft edge.
    let sun_cos = cos(SUN_ANGULAR_RADIUS);
    let glow_cos = cos(SUN_ANGULAR_RADIUS * 3.0);
    if cos_angle > sun_cos {
        // Hard disc with limb darkening.
        let limb = 1.0 - 0.3 * (1.0 - (cos_angle - sun_cos) / (1.0 - sun_cos));
        return sun_transmittance * sun_intensity * limb;
    } else {
        // Soft glow.
        let t = (cos_angle - glow_cos) / (sun_cos - glow_cos);
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

        // Per-light shadow from shadow texture. Shadow channels are written
        // in light order for shadow-casting lights (matching march pass).
        var light_shadow = 1.0;
        let cast_shadow = light.params.w;
        if cast_shadow >= 0.5 && shadow_idx < 4u {
            light_shadow = shadow_data[shadow_idx];
            shadow_idx++;
        }

        lo += (diffuse + specular) * radiance * n_dot_l * light_shadow;
    }

    // Ambient: atmosphere-derived hemisphere irradiance + AO.
    let hemisphere = shade_params.ambient_color
                   * mix(0.5, 1.0, dot(N, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5);
    let ambient = hemisphere * albedo * (1.0 - metallic) * ao;

    // Emission.
    let emission = mat.base_color.rgb * mat.emission_strength;

    let final_color = lo + ambient + emission;
    textureStore(output, coord, vec4<f32>(final_color, 1.0));
}
