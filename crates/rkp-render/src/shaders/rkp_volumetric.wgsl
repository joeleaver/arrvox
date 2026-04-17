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
    prev_view_proj: mat4x4<f32>,
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
@group(0) @binding(4) var history_scatter: texture_2d<f32>;
@group(0) @binding(5) var history_samp: sampler;

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

// FBM with footprint-aware octave LOD. Octaves whose wavelength falls below the
// sampling footprint are smoothly faded out — prefiltering kills the binary
// edge aliasing that appears when step size exceeds noise detail scale.
fn fbm_3d_lod(p: vec3<f32>, max_octaves: u32, base_freq: f32, footprint: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0u; i < max_octaves; i++) {
        let freq = base_freq * pow(2.0, f32(i));
        let wavelength = 1.0 / max(freq, 1e-8);
        // Wider fade (0.25×–2×) so octave transitions don't show up as visible
        // bands at the distances where a given wavelength crosses Nyquist.
        let lod = 1.0 - smoothstep(0.25 * wavelength, 2.0 * wavelength, footprint);
        sum += amp * value_noise_3d(pos) * lod;
        amp *= 0.5;
        pos *= 2.0;
    }
    return sum;
}

// --- Cloud density ---

fn cloud_density(pos: vec3<f32>, footprint: f32) -> f32 {
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

    // Shape FBM (4 octaves) — fades high octaves once footprint exceeds their wavelength.
    let shape_freq = cloud_params.noise.x;
    let shape = fbm_3d_lod(noise_pos * shape_freq, 4u, shape_freq, footprint);

    // Weather modulation (2 octaves, coarse scale).
    // At high coverage, blend toward 1.0 to suppress large-scale gaps.
    let weather_freq = 1.0 / max(cloud_params.noise.w, 1.0);
    let raw_weather = fbm_3d_lod(noise_pos * weather_freq, 2u, weather_freq, footprint);
    let coverage = cloud_params.flags.y;
    let weather = mix(raw_weather, 1.0, coverage * coverage);

    var base = shape * weather * height_grad;
    base = max(base - threshold, 0.0);

    // Detail erosion (4 octaves for finer wispy features; the extra octave
    // fades naturally at distance via the LOD term).
    let detail_freq = cloud_params.noise.y;
    let detail = fbm_3d_lod(noise_pos * detail_freq, 4u, detail_freq, footprint);
    base = max(base - detail * cloud_params.noise.z, 0.0);

    return base * density_scale;
}

// --- Cloud phase + scatter constants ---
const CLOUD_G_FORWARD: f32 = 0.6;
const CLOUD_G_BACK: f32 = -0.2;
const CLOUD_FORWARD_WEIGHT: f32 = 0.3;
const CLOUD_ALBEDO: vec3<f32> = vec3<f32>(1.0);

// Multi-scatter octave parameters (Wrenninge / Hillaire). Each successive octave
// attenuates extinction (b), phase anisotropy (c), and overall contribution (a).
const CLOUD_MS_OCTAVES: u32 = 3u;
const CLOUD_MS_ATTEN: f32 = 0.4;        // b — how much less sunlight is attenuated
const CLOUD_MS_CONTRIB: f32 = 0.3;      // a — weight of each successive octave (lower = deeper shadows, more visible cloud form)
const CLOUD_MS_PHASE_ATTEN: f32 = 0.5;  // c — pushes phase toward isotropic

// Atmospheric extinction along a camera→cloud path, used to blend distant cloud
// scatter toward sky color (aerial perspective). Without this, horizon clouds
// look too dark because single-scatter never gets the atmospheric wash-out.
const CLOUD_AP_SIGMA: f32 = 1.0e-4;

fn cloud_phase_at(cos_sun: f32, phase_g_scale: f32) -> f32 {
    return mix(
        henyey_greenstein(cos_sun, CLOUD_G_BACK * phase_g_scale),
        henyey_greenstein(cos_sun, CLOUD_G_FORWARD * phase_g_scale),
        CLOUD_FORWARD_WEIGHT,
    );
}

// --- Cloud self-shadow ---
// 4 exponentially-spaced samples toward the sun (40 m → 320 m range, ~600 m total).
// Reduced from 5 — TAA smooths the residual noise, and the last 480 m step was
// sampling at such a coarse LOD that the detail FBM was already faded out.
fn cloud_sun_optical_depth(pos: vec3<f32>, jitter: f32) -> f32 {
    let num_steps = 4u;
    let base_step = 40.0;
    var tau = 0.0;
    var p = pos + params.sun_dir.xyz * (jitter * base_step);
    var step = base_step;
    for (var i = 0u; i < num_steps; i++) {
        let d = cloud_density(p, step);
        tau += d * step;
        p += params.sun_dir.xyz * step;
        step *= 2.0;
    }
    return tau;
}

// Multi-scatter approximation: sum several (phase, Beer) octaves with progressively
// attenuated extinction and anisotropy. First octave is direct single-scatter
// (bright edge), later octaves brighten the core where Beer saturates to zero.
fn cloud_sun_inscatter(tau_sun: f32, cos_sun: f32, sun_col: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var a = 1.0;
    var b = 1.0;
    var c = 1.0;
    for (var n = 0u; n < CLOUD_MS_OCTAVES; n++) {
        sum += a * cloud_phase_at(cos_sun, c) * exp(-tau_sun * b) * sun_col;
        a *= CLOUD_MS_CONTRIB;
        b *= CLOUD_MS_ATTEN;
        c *= CLOUD_MS_PHASE_ATTEN;
    }
    return sum;
}

// --- Combined density ---

fn sample_density(pos: vec3<f32>, t: f32, footprint: f32) -> vec2<f32> {
    var fog = params.fog_distance.z; // ambient dust
    if params.fog_color.w > 0.5 {
        fog += height_fog_density(pos);
    }
    if params.fog_height.w > 0.5 {
        fog += distance_fog_density(t);
    }
    let cloud = cloud_density(pos, footprint);
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

    var scatter = vec3<f32>(0.0);
    var transmittance = 1.0;

    // Near-field march (fog + dust + low clouds).
    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if t >= max_t { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;
        // Fade out fog near the camera to avoid visible density boundary.
        let near_fade = smoothstep(0.0, 20.0, t);
        let densities = sample_density(pos, t, params.step_size);
        let fog_dens = densities.x * near_fade;
        let cloud_dens = densities.y;
        let total = fog_dens + cloud_dens;
        if total <= 0.001 { continue; }

        let sun_vis = 1.0;

        // Fog/dust in-scattering.
        let fog_sun = fog_dens * sun_vis * henyey_greenstein(cos_sun, dust_g)
                    * params.sun_color.xyz * scatter_albedo;
        let fog_sky = fog_dens * sky_ambient * scatter_albedo;

        // Cloud in-scattering — self-shadow + multi-scatter octaves when cloud is present.
        var cloud_sun_L = vec3<f32>(0.0);
        if cloud_dens > 0.001 {
            let tau_sun = cloud_sun_optical_depth(pos, jitter);
            cloud_sun_L = cloud_sun_inscatter(tau_sun, cos_sun, params.sun_color.xyz) * CLOUD_ALBEDO;
        }
        let cloud_sun = cloud_dens * cloud_sun_L;
        let cloud_sky = cloud_dens * sky_ambient * CLOUD_ALBEDO;

        // Analytic per-step integration assuming constant density over the step:
        // ∫₀^dt σ_s·L·exp(-σ_t·s) ds = (σ_s·L / σ_t)·(1-exp(-σ_t·dt)).
        // σ_s terms (fog_*, cloud_*) already carry their densities, so factor out 1/total.
        let absorbed = 1.0 - exp(-total * params.step_size);
        scatter += (fog_sun + fog_sky + cloud_sun + cloud_sky) * transmittance * (absorbed / total);
        transmittance *= 1.0 - absorbed;
        if transmittance < 0.03 { break; }
    }

    // High-altitude cloud march (ray-slab intersection).
    // Only for sky pixels — clouds behind opaque geometry are occluded.
    let is_sky = depth_data.w >= 9999.0 || depth_data.w <= 0.0;
    if cloud_params.flags.x > 0.5 && transmittance > 0.01 && is_sky {
        let cloud_min = cloud_params.altitude.x;
        let cloud_max = cloud_params.altitude.y;
        let cam_y = params.cam_pos.y;

        // Max march distance — large enough to reach the flat-slab horizon at
        // grazing angles for any reasonable cloud altitude.
        let MAX_CLOUD_DIST = 100000.0;

        var hit_slab = false;
        var slab_near = 0.0;
        var slab_far = 0.0;

        if abs(ray_dir.y) > 0.0001 {
            // Non-horizontal ray: standard ray-slab intersection.
            let t_min = (cloud_min - cam_y) / ray_dir.y;
            let t_max = (cloud_max - cam_y) / ray_dir.y;
            slab_near = max(min(t_min, t_max), max_t);
            slab_far = min(max(t_min, t_max), MAX_CLOUD_DIST);
            hit_slab = slab_far > slab_near && slab_far > 0.0;
        } else if cam_y >= cloud_min && cam_y <= cloud_max {
            // Near-horizontal ray while camera sits inside the cloud layer —
            // the ray stays in the slab for its entire length.
            slab_near = max_t;
            slab_far = MAX_CLOUD_DIST;
            hit_slab = slab_far > slab_near;
        }

        if hit_slab {
            // Quadratic step distribution: dense sampling near the camera,
            // progressively coarser toward the horizon. This keeps detail for
            // close clouds while letting the march reach tens of kilometers.
            let cloud_steps = 32u;
            // Per-frame jitter — combined with temporal reprojection this averages out
            // as dither rather than locking a static noise pattern into screen space.
            let cloud_jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), params.frame_index);
            let slab_len = slab_far - slab_near;

            // Atmospheric in-scatter radiance along the view ray — the color distant
            // clouds blend toward. Sky shader handles empty-sky aerial perspective
            // on its own, so we only apply this to cloud samples (not empty steps).
            let atm_L = henyey_greenstein(cos_sun, dust_g) * params.sun_color.xyz * scatter_albedo
                      + sky_ambient * scatter_albedo;

            for (var i = 0u; i < cloud_steps; i++) {
                let u_a = f32(i) / f32(cloud_steps);
                let u_b = f32(i + 1u) / f32(cloud_steps);
                let t_a = slab_near + u_a * u_a * slab_len;
                let t_b = slab_near + u_b * u_b * slab_len;
                let cloud_step_size = t_b - t_a;
                let t = mix(t_a, t_b, cloud_jitter);
                let pos = params.cam_pos.xyz + ray_dir * t;
                let cd = cloud_density(pos, cloud_step_size);
                if cd <= 0.001 { continue; }

                let tau_sun = cloud_sun_optical_depth(pos, cloud_jitter);
                let sun_L = cloud_sun_inscatter(tau_sun, cos_sun, params.sun_color.xyz) * CLOUD_ALBEDO;
                let cloud_L = sun_L + sky_ambient * CLOUD_ALBEDO;

                // Aerial perspective per cloud sample: the intrinsic cloud radiance
                // reaches the camera attenuated by exp(-σ_air·t), with the missing
                // fraction replaced by atmospheric in-scatter along that same path.
                let ap_T = exp(-CLOUD_AP_SIGMA * t);
                let displayed_L = cloud_L * ap_T + atm_L * (1.0 - ap_T);

                // Analytic per-step integration (albedo=1: σ_s = σ_t = cd, so cd cancels).
                let absorbed = 1.0 - exp(-cd * cloud_step_size);
                scatter += displayed_L * absorbed * transmittance;
                transmittance *= 1.0 - absorbed;
                if transmittance < 0.03 { break; }
            }
        }
    }

    // --- Temporal reprojection ---
    // Rotation-only reprojection: multiply the world-space ray direction by prev_vp
    // with w=0. This ignores camera translation entirely (valid for distant sky/cloud
    // content) and captures camera rotation exactly, which is the main source of
    // between-frame pixel-to-pixel variation for a sky pass.
    var final_scatter = scatter;
    var final_trans = transmittance;

    let prev_clip = params.prev_view_proj * vec4<f32>(ray_dir, 0.0);
    if prev_clip.w > 0.0 && params.frame_index > 0u {
        let prev_ndc = prev_clip.xyz / prev_clip.w;
        let prev_uv = prev_ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
        if all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
            let hist = textureSampleLevel(history_scatter, history_samp, prev_uv, 0.0);
            // Balance between detail (high alpha) and flicker (low alpha). 0.25
            // keeps visible cloud structure while damping high-freq sample jitter
            // that the 4th detail octave can produce at high density.
            let alpha = 0.25;
            final_scatter = mix(hist.rgb, scatter, alpha);
            final_trans = mix(hist.a, transmittance, alpha);
        }
    }

    textureStore(output_scatter, coord, vec4<f32>(final_scatter, final_trans));
}
