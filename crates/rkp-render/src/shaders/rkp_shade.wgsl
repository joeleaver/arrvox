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
    layer_mask: u32,
    focus_object_id: u32,
    _cam_pad0: u32,
    _cam_pad1: u32,
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
    isolation: u32,
}

// Neutral gray for isolation-mode sky + ambient — chosen to match a
// typical 18% middle-gray studio backdrop in linear light.
const ISOLATION_BG: vec3<f32> = vec3<f32>(0.18, 0.18, 0.18);
const ISOLATION_AMBIENT: vec3<f32> = vec3<f32>(0.4, 0.4, 0.4);

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

// Group 1: shadow texture (read, half-res) + SSAO texture (read, half-res)
// shadow_tex is written by rkp_shadow_trace at half-res; we upsample it
// per-pixel with a bilateral gather guided by position + normal deltas.
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
@group(5) @binding(3) var sky_view_lut: texture_2d<f32>;
@group(5) @binding(4) var aerial_perspective_lut: texture_3d<f32>;

// Aerial-perspective LUT slice parameterization — must match rkp_aerial_perspective_lut.wgsl.
// Slice z stores the camera-to-d(z) atmospheric integral under an exponential
// map d(z) = AP_NEAR * (AP_FAR/AP_NEAR)^((z+0.5)/N); the shade pass inverts this.
const AP_NEAR_DISTANCE: f32 = 1.0;
const AP_FAR_DISTANCE: f32 = 128000.0;

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

/// Map a view ray direction to Sky View LUT UV coordinates.
/// Inverse of the parameterization in rkp_sky_view_lut.wgsl.
fn ray_dir_to_sky_view_uv(ray_dir: vec3<f32>, sun_dir: vec3<f32>, cam_height: f32) -> vec2<f32> {
    let view_zenith_cos = ray_dir.y;

    let horizon_cos = -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (cam_height * cam_height), 0.0));
    let horizon_angle = acos(horizon_cos);

    let view_angle = acos(clamp(view_zenith_cos, -1.0, 1.0));

    // V: non-linear mapping with horizon at v=0.5.
    var v: f32;
    if view_angle <= horizon_angle {
        // Above horizon: angle goes from horizon_angle (v=0.5) to 0 (v=1.0).
        // coord = 1 - angle/horizon_angle. Invert: coord = (2v-1)² → v = 0.5 + 0.5*sqrt(coord)
        let coord = 1.0 - view_angle / max(horizon_angle, 1e-6);
        v = 0.5 + 0.5 * sqrt(max(coord, 0.0));
    } else {
        // Below horizon: angle goes from horizon_angle (v=0.5) to PI (v=0.0).
        // coord = (angle - horizon) / (PI - horizon). Invert: coord = (1-2v)² → v = 0.5 - 0.5*sqrt(coord)
        let beta = PI - horizon_angle;
        let coord = (view_angle - horizon_angle) / max(beta, 1e-6);
        v = 0.5 - 0.5 * sqrt(max(coord, 0.0));
    }

    // U: view-sun azimuth via horizontal projection.
    let sun_horiz_len = length(vec2<f32>(sun_dir.x, sun_dir.z));
    let view_horiz_len = length(vec2<f32>(ray_dir.x, ray_dir.z));
    var light_view_cos = 0.0;
    if sun_horiz_len > 0.001 && view_horiz_len > 0.001 {
        let sun_h = vec2<f32>(sun_dir.x, sun_dir.z) / sun_horiz_len;
        let view_h = vec2<f32>(ray_dir.x, ray_dir.z) / view_horiz_len;
        light_view_cos = dot(sun_h, view_h);
    }
    let u = sqrt(max((light_view_cos + 1.0) * 0.5, 0.0));

    return clamp(vec2<f32>(u, v), vec2<f32>(0.001), vec2<f32>(0.999));
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

/// Sun disc with atmospheric extinction from LUT + aureole glow.
fn sun_disc(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let cos_angle = dot(ray_dir, sun_dir);
    let glow_radius = SUN_ANGULAR_RADIUS * 10.0;
    if cos_angle < cos(glow_radius) { return vec3<f32>(0.0); }

    // Clip sun disc/glow below the horizon — the view ray must be above the
    // geometric horizon for the sun to be visible at that pixel.
    let view_height = EARTH_RADIUS + camera_alt;
    let horizon_cos = -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (view_height * view_height), 0.0));
    if ray_dir.y < horizon_cos { return vec3<f32>(0.0); }
    let sun_cos = dot(vec3<f32>(0.0, 1.0, 0.0), sun_dir);
    let sun_transmittance = lookup_transmittance(view_height, sun_cos);

    // Sun disc luminance = illuminance / solid_angle (Filament reference).
    let sun_solid_angle = 2.0 * PI * (1.0 - cos(SUN_ANGULAR_RADIUS));
    let sun_luminance = sun_intensity / sun_solid_angle;

    let sun_cos_r = cos(SUN_ANGULAR_RADIUS);
    let glow_cos = cos(glow_radius);

    var result = vec3<f32>(0.0);
    if cos_angle > sun_cos_r {
        // Hard disc — clips to white after tone mapping.
        let center_dist = (cos_angle - sun_cos_r) / (1.0 - sun_cos_r);
        let limb = 1.0 - 0.3 * (1.0 - center_dist);
        result = sun_transmittance * sun_luminance * limb;
    }

    // Aureole glow — bright near the disc, fading outward.
    // Uses sun_luminance scaled down so the inner glow is bright enough
    // to show transmittance color (orange at sunset) after tone mapping.
    let t = (cos_angle - glow_cos) / (sun_cos_r - glow_cos);
    if t > 0.0 {
        let glow_luminance = sun_luminance * 0.002; // ~0.2% of disc brightness
        result += sun_transmittance * glow_luminance * pow(t, 2.0);
    }

    return result;
}

/// Sample the aerial-perspective LUT for a shaded geometry pixel.
/// Returns (inscatter_rgb, transmittance). Apply as:
///   final = inscatter + surface_color * transmittance.
fn sample_aerial_perspective(screen_uv: vec2<f32>, surface_distance: f32) -> vec4<f32> {
    // Invert the exponential slice map to recover the texture-space Z that
    // encodes this distance, then let trilinear filtering interpolate between
    // the two bracketing slices in the LUT.
    let log_ratio = log(AP_FAR_DISTANCE / AP_NEAR_DISTANCE);
    let d = max(surface_distance, AP_NEAR_DISTANCE);
    let slice_u = clamp(log(d / AP_NEAR_DISTANCE) / log_ratio, 0.0, 1.0);
    return textureSampleLevel(
        aerial_perspective_lut, atmo_sampler,
        vec3<f32>(screen_uv, slice_u), 0.0,
    );
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

    // No geometry → sky.
    if hit_t >= 9999.0 || hit_t <= 0.0 {
        if shade_params.isolation != 0u {
            // Isolation: flat neutral background, no sun disc, no sky LUT.
            // The grid pass paints over this for floor pixels.
            textureStore(output, coord, vec4<f32>(ISOLATION_BG, 1.0));
            return;
        }
        let uv_screen = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
        let ndc = vec2<f32>(uv_screen.x * 2.0 - 1.0, 1.0 - uv_screen.y * 2.0);
        let ray_dir = normalize(camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz);
        let s_dir = normalize(shade_params.sun_dir);

        // Look up precomputed sky radiance from Sky View LUT.
        let sky_uv = ray_dir_to_sky_view_uv(ray_dir, s_dir, EARTH_RADIUS + shade_params.camera_altitude);
        var sky = textureSampleLevel(sky_view_lut, atmo_sampler, sky_uv, 0.0).rgb;
        sky += sun_disc(ray_dir, s_dir, shade_params.sun_intensity, shade_params.camera_altitude);

        textureStore(output, coord, vec4<f32>(sky, 1.0));
        return;
    }

    let normal = normalize(textureLoad(gbuf_normal, coord, 0).xyz);
    let mat_data = textureLoad(gbuf_material, coord, 0);
    let packed_r = mat_data.r;
    let packed_g = mat_data.g;

    // Unpack material IDs + blend.
    let primary_mat_id = packed_r & 0xFFFFu;
    let secondary_mat_id = (packed_r >> 16u) & 0xFFFFu;
    let blend_weight = f32(packed_g & 0xFFu) / 255.0;

    // Resolve material. Metallic/roughness/emission come from the
    // palette; base_color is used for the albedo unless a per-voxel
    // color override (RGB565 in packed_g high 16) is present.
    //
    // Dual-material path: octree_march (voxel) and proc_raymarch
    // (procedural, eventually) can both write a secondary material +
    // non-zero blend_weight for smooth seams across MaterialByHeight
    // / MaterialByNoise transition zones. We lerp every PBR property
    // in linear space. The secondary palette lookup is guarded behind
    // `blend_weight > 0` — some paths currently reuse the secondary
    // slot for pick IDs (see proc_raymarch) and those pixels always
    // have blend=0, so the guard prevents an out-of-bounds material
    // read on stale bits. Once pick IDs move to a dedicated texture
    // the guard is still desirable (free early-out on single-material
    // pixels, which is the common case).
    let mat_a = materials[primary_mat_id];
    var base_color = mat_a.base_color.rgb;
    var metallic_raw = mat_a.metallic;
    var roughness_raw = mat_a.roughness;
    var emission_strength = mat_a.emission_strength;
    if blend_weight > 0.0 {
        let mat_b = materials[secondary_mat_id];
        base_color = mix(base_color, mat_b.base_color.rgb, blend_weight);
        metallic_raw = mix(metallic_raw, mat_b.metallic, blend_weight);
        roughness_raw = mix(roughness_raw, mat_b.roughness, blend_weight);
        emission_strength = mix(emission_strength, mat_b.emission_strength, blend_weight);
    }

    let color_rgb565 = (packed_g >> 16u) & 0xFFFFu;
    var albedo = base_color;
    if color_rgb565 != 0u {
        let cr5 = color_rgb565 & 0x1Fu;
        let cg6 = (color_rgb565 >> 5u) & 0x3Fu;
        let cb5 = (color_rgb565 >> 11u) & 0x1Fu;
        albedo = vec3<f32>(f32(cr5) / 31.0, f32(cg6) / 63.0, f32(cb5) / 31.0);
    }
    let metallic = metallic_raw;
    let roughness = max(roughness_raw, 0.04);

    // View direction.
    let V = normalize(camera.position.xyz - world_pos);
    let N = normal;
    let n_dot_v = max(dot(N, V), 0.001);

    // F0 for dielectrics vs metals.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Per-light shadow: bilateral upsample from the half-res shadow texture
    // written by rkp_shadow_trace. Each half-res sample's "reference
    // surface" is the full-res gbuf pixel at (half_coord * 2); compare
    // that against our pixel's surface to weight the 4 nearest samples
    // and reject neighbors on different surfaces.
    let shadow_dims = textureDimensions(shadow_tex);
    let gbuf_dims = textureDimensions(gbuf_position);
    // Continuous half-res coord: our full-res pixel center in half-res UVs.
    // Full-res pixel (x, y) maps to half-res sample at (x/2, y/2); the
    // 4 nearest integer half-res coords straddle that point.
    let half_coord_f = (vec2<f32>(coord) + 0.5) * 0.5 - 0.5;
    let base_half = vec2<i32>(floor(half_coord_f));
    let frac = half_coord_f - vec2<f32>(base_half);
    let spatial_w = vec4<f32>(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y,
    );
    var shadow_data = vec4<f32>(0.0);
    var w_sum = 0.0;
    var bilinear_data = vec4<f32>(0.0);
    // Isolation: shadow_trace pass is skipped, so shadow_tex contains
    // stale data. Force fully-lit and bypass the bilateral gather.
    if shade_params.isolation != 0u {
        shadow_data = vec4<f32>(1.0);
        w_sum = 1.0;
    } else {
    // Position sigma — in world units. The shadow trace samples one full
    // surface per 2-pixel block, so neighbors straddling a depth
    // discontinuity should get ~zero weight. 5 cm is tight enough to
    // prevent bleed across surface edges at the elephant's scale.
    let sigma_pos = 0.05;
    let inv_sigma2 = 1.0 / (sigma_pos * sigma_pos);
    for (var k = 0u; k < 4u; k++) {
        let dx = i32(k & 1u);
        let dy = i32((k >> 1u) & 1u);
        let half_c = base_half + vec2<i32>(dx, dy);
        if half_c.x < 0 || half_c.y < 0
            || u32(half_c.x) >= shadow_dims.x || u32(half_c.y) >= shadow_dims.y {
            continue;
        }
        // Reference surface for this half-res sample = gbuf at half_c * 2.
        let ref_full = half_c * 2;
        if ref_full.x < 0 || ref_full.y < 0
            || u32(ref_full.x) >= gbuf_dims.x || u32(ref_full.y) >= gbuf_dims.y {
            continue;
        }
        let ref_pos = textureLoad(gbuf_position, ref_full, 0);
        let ref_normal = textureLoad(gbuf_normal, ref_full, 0).xyz;
        // Sky/miss neighbor: skip — no surface to compare against.
        if ref_pos.w >= 1e9 { continue; }

        let s = textureLoad(shadow_tex, half_c, 0);
        bilinear_data += s * spatial_w[k];

        let d_pos = ref_pos.xyz - world_pos;
        let pos_term = exp(-dot(d_pos, d_pos) * inv_sigma2);
        let n_dot = clamp(dot(ref_normal, N), 0.0, 1.0);
        let normal_term = n_dot * n_dot * n_dot * n_dot * n_dot * n_dot * n_dot * n_dot; // ^8
        let w = spatial_w[k] * pos_term * normal_term;
        shadow_data += s * w;
        w_sum += w;
    }
    // Fallback: if bilateral weights all rejected, fall back to plain
    // bilinear so we never return zero from this gather on valid surfaces.
    if w_sum < 1e-5 {
        shadow_data = bilinear_data;
    } else {
        shadow_data /= w_sum;
    }
    } // end !isolation block

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

    // Ambient: in-situ uses multi-scattering LUT for sky irradiance; in
    // isolation we substitute a flat neutral irradiance so the result is
    // independent of sun direction / atmosphere state.
    var ms_irradiance: vec3<f32>;
    if shade_params.isolation != 0u {
        ms_irradiance = ISOLATION_AMBIENT;
    } else {
        let cam_height = EARTH_RADIUS + shade_params.camera_altitude;
        let sun_cos_z = shade_params.sun_dir.y;
        ms_irradiance = lookup_multiscatter(cam_height, sun_cos_z)
                      * shade_params.sun_intensity
                      * shade_params.ambient_intensity;
    }

    let ambient_diffuse = ms_irradiance * albedo * (1.0 - metallic) * ao;

    // Ambient specular: approximate sky reflection for energy conservation.
    let F_env = fresnel_schlick(n_dot_v, f0);
    let ambient_specular = ms_irradiance * F_env * ao;
    let ambient = ambient_diffuse + ambient_specular;

    // Emission — drive from the (possibly blended) base_color and
    // emission_strength, so a secondary material emitting in the
    // transition zone properly shows through.
    let emission = base_color * emission_strength;

    var final_color = lo + ambient + emission;

    // Aerial perspective — atmospheric inscatter + extinction between camera
    // and the shaded surface. Under an exponential slice map, near-field
    // precision is fine (slice 0 ≈ 1.2 m) so close voxels receive ≈ 1.0
    // transmittance and ≈ 0 scatter, no discoloration. Far geometry meets
    // the sky-view LUT at the horizon because both integrate the same
    // atmosphere through most of its depth.
    let dims_f = vec2<f32>(dims);
    let screen_uv = (vec2<f32>(coord) + 0.5) / dims_f;
    let surface_dist = length(world_pos - camera.position.xyz);
    let ap = sample_aerial_perspective(screen_uv, surface_dist);
    final_color = final_color * ap.a + ap.rgb;

    textureStore(output, coord, vec4<f32>(final_color, 1.0));
}
