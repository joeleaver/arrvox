// RKIPatch volumetric march — fog, dust, and procedural clouds.
//
// Half-resolution compute shader. Marches view rays through the atmosphere,
// evaluating fog + dust + cloud density at each step. Accumulates in-scattered
// light and transmittance. Output: Rgba16Float (rgb=scatter, a=transmittance).

const PI: f32 = 3.14159265;

// --- Structs ---

struct VolumetricParams {
    cam_pos:      vec4<f32>,
    cam_forward:  vec4<f32>,
    cam_right:    vec4<f32>,
    cam_up:       vec4<f32>,
    sun_dir:      vec4<f32>,   // xyz = toward sun
    sun_color:    vec4<f32>,   // xyz = color * intensity
    width:        u32,
    height:       u32,
    full_width:   u32,
    full_height:  u32,
    max_steps:    u32,
    step_size:    f32,
    near:         f32,
    far:          f32,
    fog_color:    vec4<f32>,   // xyz = scatter albedo, w = height_fog_enable
    fog_height:   vec4<f32>,   // x = base_density, y = base_height, z = falloff, w = dist_fog_enable
    fog_distance: vec4<f32>,   // x = dist_density, y = dist_falloff, z = dust_density, w = dust_g
    frame_index:  u32,
    vol_ambient_r: f32,
    vol_ambient_g: f32,
    vol_ambient_b: f32,
}

struct CloudParams {
    altitude: vec4<f32>,   // x=cloud_min, y=cloud_max, z=threshold, w=density_scale
    noise:    vec4<f32>,   // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    wind:     vec4<f32>,   // x=wind_dir.x, y=wind_dir.y, z=wind_speed, w=time
    flags:    vec4<f32>,   // x=enable (0/1)
}

// --- Bindings ---

@group(0) @binding(0) var<uniform> params: VolumetricParams;
@group(0) @binding(1) var depth_buffer: texture_2d<f32>;
@group(0) @binding(2) var output_scatter: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> cloud_params: CloudParams;

// --- Helpers ---

fn interleaved_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let offset = 5.588238 * f32(frame % 64u);
    let p = pixel + vec2<f32>(offset, offset);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-6), 1.5));
}

// --- Fog density ---

fn height_fog_density(pos: vec3<f32>) -> f32 {
    let base_density = params.fog_height.x;
    let base_height = params.fog_height.y;
    let falloff = params.fog_height.z;
    return base_density * exp(-falloff * max(pos.y - base_height, 0.0));
}

fn distance_fog_density(t: f32) -> f32 {
    let density = params.fog_distance.x;
    let falloff = params.fog_distance.y;
    return density * (1.0 - exp(-falloff * t));
}

// --- Noise (for clouds) ---

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i);
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x0 = mix(n000, n100, u.x);
    let x1 = mix(n010, n110, u.x);
    let x2 = mix(n001, n101, u.x);
    let x3 = mix(n011, n111, u.x);
    return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

fn fbm_3d(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        sum += amp * value_noise_3d(pos);
        amp *= 0.5;
        pos *= 2.0;
    }
    return sum;
}

// --- Cloud density ---

fn cloud_density(pos: vec3<f32>) -> f32 {
    if cloud_params.flags.x < 0.5 { return 0.0; }

    let cloud_min = cloud_params.altitude.x;
    let cloud_max = cloud_params.altitude.y;
    let threshold = cloud_params.altitude.z;
    let density_scale = cloud_params.altitude.w;

    let height = pos.y;
    if height < cloud_min || height > cloud_max { return 0.0; }

    // Height gradient.
    let height_above_base = height - cloud_min;
    let height_below_top = cloud_max - height;
    let height_grad = smoothstep(0.0, 50.0, height_above_base)
                    * smoothstep(0.0, 200.0, height_below_top);

    // Wind scrolling.
    let wind_offset = vec2<f32>(cloud_params.wind.x, cloud_params.wind.y)
                    * cloud_params.wind.z * cloud_params.wind.w;
    let noise_pos = vec3<f32>(pos.x + wind_offset.x, pos.y, pos.z + wind_offset.y)
                  + vec3<f32>(173.5, 247.3, 391.7);

    // Shape FBM (4 octaves).
    let shape = fbm_3d(noise_pos * cloud_params.noise.x, 4u);

    // Weather modulation (2 octaves, coarse scale).
    // At high coverage, blend toward 1.0 to suppress large-scale gaps.
    let raw_weather = fbm_3d(noise_pos * (1.0 / max(cloud_params.noise.w, 1.0)), 2u);
    let coverage = cloud_params.flags.y;
    let weather = mix(raw_weather, 1.0, coverage * coverage);

    var base = shape * weather * height_grad;
    base = max(base - threshold, 0.0);

    // Detail erosion (3 octaves).
    let detail = fbm_3d(noise_pos * cloud_params.noise.y, 3u);
    base = max(base - detail * cloud_params.noise.z, 0.0);

    return base * density_scale;
}

// --- Combined density ---

fn sample_density(pos: vec3<f32>, t: f32) -> vec2<f32> {
    var fog = params.fog_distance.z; // ambient dust
    if params.fog_color.w > 0.5 {
        fog += height_fog_density(pos);
    }
    if params.fog_height.w > 0.5 {
        fog += distance_fog_density(t);
    }
    let cloud = cloud_density(pos);
    return vec2<f32>(fog, cloud);
}

// --- Main march ---

@compute @workgroup_size(8, 8, 1)
fn march_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(params.cam_forward.xyz + ndc.x * params.cam_right.xyz + ndc.y * params.cam_up.xyz);

    // Scene depth from G-buffer (sample center of 2x2 block).
    let full_coord = vec2<i32>(gid.xy) * 2 + vec2<i32>(1, 1);
    let depth_data = textureLoad(depth_buffer, full_coord, 0);
    var max_t = min(depth_data.w, params.far);
    if depth_data.w >= 9999.0 || depth_data.w <= 0.0 {
        max_t = params.far;
    }

    let jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), params.frame_index);
    let cos_sun = dot(ray_dir, params.sun_dir.xyz);
    let dust_g = params.fog_distance.w;
    let scatter_albedo = params.fog_color.xyz;
    let sky_ambient = vec3<f32>(params.vol_ambient_r, params.vol_ambient_g, params.vol_ambient_b);

    let cloud_g_forward = 0.6;
    let cloud_g_back = -0.2;
    let cloud_forward_weight = 0.3;
    let cloud_albedo = vec3<f32>(1.0);

    var scatter = vec3<f32>(0.0);
    var transmittance = 1.0;

    // Near-field march (fog + dust + low clouds).
    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if t >= max_t { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;
        let densities = sample_density(pos, t);
        let fog_dens = densities.x;
        let cloud_dens = densities.y;
        let total = fog_dens + cloud_dens;
        if total <= 0.001 { continue; }

        let step_trans = exp(-total * params.step_size);

        // Fog/dust in-scattering.
        let fog_sun = fog_dens * henyey_greenstein(cos_sun, dust_g)
                    * params.sun_color.xyz * scatter_albedo;
        let fog_sky = fog_dens * sky_ambient * scatter_albedo;

        // Cloud in-scattering.
        let cloud_phase = mix(
            henyey_greenstein(cos_sun, cloud_g_back),
            henyey_greenstein(cos_sun, cloud_g_forward),
            cloud_forward_weight,
        );
        let cloud_sun = cloud_dens * cloud_phase * params.sun_color.xyz * cloud_albedo;
        let cloud_sky = cloud_dens * sky_ambient * cloud_albedo;

        scatter += (fog_sun + fog_sky + cloud_sun + cloud_sky) * transmittance * params.step_size;
        transmittance *= step_trans;
        if transmittance < 0.01 { break; }
    }

    // High-altitude cloud march (ray-slab intersection).
    // Only for sky pixels — clouds behind opaque geometry are occluded.
    let is_sky = depth_data.w >= 9999.0 || depth_data.w <= 0.0;
    if cloud_params.flags.x > 0.5 && transmittance > 0.01 && is_sky {
        let cloud_min = cloud_params.altitude.x;
        let cloud_max = cloud_params.altitude.y;
        let cam_y = params.cam_pos.y;

        var hit_slab = false;
        var slab_near = 0.0;
        var slab_far = 0.0;

        if abs(ray_dir.y) > 0.001 {
            let t_min = (cloud_min - cam_y) / ray_dir.y;
            let t_max = (cloud_max - cam_y) / ray_dir.y;
            slab_near = max(min(t_min, t_max), max_t);
            slab_far = min(max(t_min, t_max), 6000.0);
            hit_slab = slab_far > slab_near && slab_far > 0.0;
        }

        if hit_slab {
            let cloud_steps = 48u;
            let cloud_step_size = (slab_far - slab_near) / f32(cloud_steps);
            let cloud_jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), 0u);

            for (var i = 0u; i < cloud_steps; i++) {
                let t = slab_near + (f32(i) + cloud_jitter) * cloud_step_size;
                let pos = params.cam_pos.xyz + ray_dir * t;
                let cd = cloud_density(pos);
                if cd <= 0.001 { continue; }

                let step_trans = exp(-cd * cloud_step_size);
                let cloud_phase = mix(
                    henyey_greenstein(cos_sun, cloud_g_back),
                    henyey_greenstein(cos_sun, cloud_g_forward),
                    cloud_forward_weight,
                );
                let cloud_sun = cd * cloud_phase * params.sun_color.xyz * cloud_albedo;
                let cloud_sky = cd * sky_ambient * cloud_albedo;

                scatter += (cloud_sun + cloud_sky) * transmittance * cloud_step_size;
                transmittance *= step_trans;
                if transmittance < 0.01 { break; }
            }
        }
    }

    textureStore(output_scatter, coord, vec4<f32>(scatter, transmittance));
}
