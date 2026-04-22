// RKIPatch cloud march — procedural-cloud participating medium, sky pixels only.
//
// Half-resolution compute shader. For tiles whose 4 full-res covered pixels
// are ALL sky, marches the view ray through the cloud slab (and any near-
// field cloud volume the camera sits inside), accumulating single-scatter
// plus a multi-scatter octave approximation, then temporally reprojects
// with the previous frame's cloud output.
//
// For tiles with any non-sky coverage we write an identity (0,0,0,1) to
// cloud_out so the composite becomes a no-op, and the history-update pass
// separately marks those history texels with an invalidity sentinel. This
// lets the TAA reprojection's per-texel validity check reject stale cloud
// data when a previously-sky pixel had been voxel-occluded last frame.
//
// Distance fog / participating-medium haze live in rkp_fog_march.wgsl.

const PI: f32 = 3.14159265;

// Matches the CPU-side VolumetricParams layout. Must stay in sync with the
// fog shader's declaration and with rkp_volumetric.rs.
struct VolParams {
    cam_pos:      vec4<f32>,
    cam_forward:  vec4<f32>,
    cam_right:    vec4<f32>,
    cam_up:       vec4<f32>,
    sun_dir:      vec4<f32>,
    sun_color:    vec4<f32>,
    width:        u32,
    height:       u32,
    full_width:   u32,
    full_height:  u32,
    max_steps:    u32,
    step_size:    f32,
    near:         f32,
    far:          f32,
    fog_color:    vec4<f32>,
    fog_height:   vec4<f32>,
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
    flags:    vec4<f32>,   // x=enable, y=coverage
    quality:  vec4<f32>,   // x=slab_steps, y=shadow_steps, z=detail_octaves, w=ms_octaves
    quality2: vec4<f32>,   // x=taa_alpha
}

@group(0) @binding(0) var<uniform> params: VolParams;
@group(0) @binding(1) var depth_buffer: texture_2d<f32>;
@group(0) @binding(2) var cloud_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> cloud_params: CloudParams;
// History is sampled via manual 4-tap bilateral (textureLoad × 4 with
// per-texel rejection based on the validity marker in alpha) rather than a
// hardware sampler — the sampler would blend across the boundary and bleed
// the marker into valid samples, producing ghost outlines around voxel
// silhouettes in motion.
@group(0) @binding(4) var history_scatter: texture_2d<f32>;

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

// Forward-biased HG for the atmospheric-perspective blend colour used on
// cloud samples (matches fog shader's FOG_ASYMMETRY).
const FOG_ASYMMETRY: f32 = 0.3;

// --- Noise ---

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

// FBM with footprint-aware octave LOD. Octaves whose wavelength falls below
// the sampling footprint are smoothly faded out — prefiltering kills the
// binary edge aliasing that appears when step size exceeds noise detail scale.
fn fbm_3d_lod(p: vec3<f32>, max_octaves: u32, base_freq: f32, footprint: f32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0u; i < max_octaves; i++) {
        let freq = base_freq * pow(2.0, f32(i));
        let wavelength = 1.0 / max(freq, 1e-8);
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

    let height_above_base = height - cloud_min;
    let height_below_top = cloud_max - height;
    let height_grad = smoothstep(0.0, 50.0, height_above_base)
                    * smoothstep(0.0, 200.0, height_below_top);

    let wind_offset = vec2<f32>(cloud_params.wind.x, cloud_params.wind.y)
                    * cloud_params.wind.z * cloud_params.wind.w;
    let noise_pos = vec3<f32>(pos.x + wind_offset.x, pos.y, pos.z + wind_offset.y)
                  + vec3<f32>(173.5, 247.3, 391.7);

    let shape_freq = cloud_params.noise.x;
    let shape = fbm_3d_lod(noise_pos * shape_freq, 4u, shape_freq, footprint);

    let weather_freq = 1.0 / max(cloud_params.noise.w, 1.0);
    let raw_weather = fbm_3d_lod(noise_pos * weather_freq, 2u, weather_freq, footprint);
    let coverage = cloud_params.flags.y;
    let weather = mix(raw_weather, 1.0, coverage * coverage);

    var base = shape * weather * height_grad;
    base = max(base - threshold, 0.0);

    // Detail erosion — uses the finest N octaves of a standard 4-octave FBM.
    let detail_freq = cloud_params.noise.y;
    let detail_octaves = u32(clamp(cloud_params.quality.z, 1.0, 4.0));
    let detail_skip = 4u - detail_octaves;
    var detail_sum = 0.0;
    var detail_amp = pow(0.5, f32(detail_skip + 1u));
    var detail_pos = noise_pos * detail_freq * pow(2.0, f32(detail_skip));
    for (var i = 0u; i < detail_octaves; i++) {
        let freq = detail_freq * pow(2.0, f32(detail_skip + i));
        let wavelength = 1.0 / max(freq, 1e-8);
        let lod = 1.0 - smoothstep(0.25 * wavelength, 2.0 * wavelength, footprint);
        detail_sum += detail_amp * value_noise_3d(detail_pos) * lod;
        detail_amp *= 0.5;
        detail_pos *= 2.0;
    }
    base = max(base - detail_sum * cloud_params.noise.z, 0.0);

    return base * density_scale;
}

// --- Cloud phase + scatter constants ---
const CLOUD_G_FORWARD: f32 = 0.6;
const CLOUD_G_BACK: f32 = -0.2;
const CLOUD_FORWARD_WEIGHT: f32 = 0.3;
const CLOUD_ALBEDO: vec3<f32> = vec3<f32>(1.0);

const CLOUD_MS_ATTEN: f32 = 0.4;
const CLOUD_MS_CONTRIB: f32 = 0.3;
const CLOUD_MS_PHASE_ATTEN: f32 = 0.5;

const CLOUD_AP_SIGMA: f32 = 1.0e-4;

fn cloud_phase_at(cos_sun: f32, phase_g_scale: f32) -> f32 {
    return mix(
        henyey_greenstein(cos_sun, CLOUD_G_BACK * phase_g_scale),
        henyey_greenstein(cos_sun, CLOUD_G_FORWARD * phase_g_scale),
        CLOUD_FORWARD_WEIGHT,
    );
}

fn cloud_sun_optical_depth(pos: vec3<f32>, jitter: f32) -> f32 {
    let num_steps = u32(max(cloud_params.quality.y, 1.0));
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

fn cloud_sun_inscatter(tau_sun: f32, cos_sun: f32, sun_col: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var a = 1.0;
    var b = 1.0;
    var c = 1.0;
    let num_octaves = u32(max(cloud_params.quality.w, 1.0));
    for (var n = 0u; n < num_octaves; n++) {
        sum += a * cloud_phase_at(cos_sun, c) * exp(-tau_sun * b) * sun_col;
        a *= CLOUD_MS_CONTRIB;
        b *= CLOUD_MS_ATTEN;
        c *= CLOUD_MS_PHASE_ATTEN;
    }
    return sum;
}

// --- Main march ---

@compute @workgroup_size(8, 8, 1)
fn cloud_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }

    let coord = vec2<i32>(gid.xy);

    // The half-res tile is sky only if ALL four full-res pixels it covers are
    // sky — otherwise partial-object edges would leak TAA-blended cloud onto
    // object pixels when the composite upsamples.
    let full_base = vec2<i32>(gid.xy) * 2;
    let d0 = textureLoad(depth_buffer, full_base, 0).w;
    let d1 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 0), 0).w;
    let d2 = textureLoad(depth_buffer, full_base + vec2<i32>(0, 1), 0).w;
    let d3 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 1), 0).w;
    let is_sky = (d0 >= 9999.0 || d0 <= 0.0)
              && (d1 >= 9999.0 || d1 <= 0.0)
              && (d2 >= 9999.0 || d2 <= 0.0)
              && (d3 >= 9999.0 || d3 <= 0.0);

    // Non-sky → write identity so the composite's cloud layer is a no-op
    // there, and bail out before touching any cloud bindings.
    if !is_sky {
        textureStore(cloud_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(params.cam_forward.xyz + ndc.x * params.cam_right.xyz + ndc.y * params.cam_up.xyz);

    // max_t guards near-field cloud samples against stepping past geometry.
    // For the all-sky case this is just `far`.
    let max_t = params.far;

    let jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), params.frame_index);
    let cos_sun = dot(ray_dir, params.sun_dir.xyz);
    let scatter_albedo = params.fog_color.xyz;
    let sky_ambient = vec3<f32>(params.vol_ambient_r, params.vol_ambient_g, params.vol_ambient_b);

    var cloud_scatter = vec3<f32>(0.0);
    var cloud_trans = 1.0;

    // Near-field march: relevant only when the camera sits inside the cloud
    // layer. Reuses the same step cadence as the fog march so precision near
    // the camera matches.
    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if t >= max_t { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;
        let cloud_dens = cloud_density(pos, params.step_size);

        if cloud_dens > 0.001 {
            let tau_sun = cloud_sun_optical_depth(pos, jitter);
            let sun_L = cloud_sun_inscatter(tau_sun, cos_sun, params.sun_color.xyz) * CLOUD_ALBEDO;
            let cloud_L = sun_L + sky_ambient * CLOUD_ALBEDO;
            let cloud_absorbed = 1.0 - exp(-cloud_dens * params.step_size);
            cloud_scatter += cloud_L * cloud_absorbed * cloud_trans;
            cloud_trans *= 1.0 - cloud_absorbed;
        }

        if cloud_trans < 0.03 { break; }
    }

    // High-altitude cloud march (ray-slab intersection).
    if cloud_params.flags.x > 0.5 && cloud_trans > 0.01 {
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
            let t_min = (cloud_min - cam_y) / ray_dir.y;
            let t_max = (cloud_max - cam_y) / ray_dir.y;
            slab_near = max(min(t_min, t_max), max_t);
            slab_far = min(max(t_min, t_max), MAX_CLOUD_DIST);
            hit_slab = slab_far > slab_near && slab_far > 0.0;
        } else if cam_y >= cloud_min && cam_y <= cloud_max {
            slab_near = max_t;
            slab_far = MAX_CLOUD_DIST;
            hit_slab = slab_far > slab_near;
        }

        if hit_slab {
            let cloud_steps = u32(max(cloud_params.quality.x, 8.0));
            let cloud_jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), params.frame_index);
            let slab_len = slab_far - slab_near;

            // Atmospheric colour distant cloud samples blend toward (AP).
            let atm_L = henyey_greenstein(cos_sun, FOG_ASYMMETRY) * params.sun_color.xyz * scatter_albedo
                      + sky_ambient * scatter_albedo;

            for (var i = 0u; i < cloud_steps; i++) {
                let u_a = f32(i) / f32(cloud_steps);
                let u_b = f32(i + 1u) / f32(cloud_steps);
                // Quadratic step distribution — dense near camera, coarse at horizon.
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

                let ap_T = exp(-CLOUD_AP_SIGMA * t);
                let displayed_L = cloud_L * ap_T + atm_L * (1.0 - ap_T);

                let absorbed = 1.0 - exp(-cd * cloud_step_size);
                cloud_scatter += displayed_L * absorbed * cloud_trans;
                cloud_trans *= 1.0 - absorbed;
                if cloud_trans < 0.03 { break; }
            }
        }
    }

    // --- Temporal reprojection (cloud only) ---
    // Rotation-only reprojection (w=0) — valid for sky content at altitude.
    // Per-texel validity gate on history alpha keeps voxel-occluded texels
    // from bleeding into valid sky samples at silhouette boundaries.
    var final_cloud_scatter = cloud_scatter;
    var final_cloud_trans = cloud_trans;

    let prev_clip = params.prev_view_proj * vec4<f32>(ray_dir, 0.0);
    if prev_clip.w > 0.0 && params.frame_index > 0u {
        let prev_ndc = prev_clip.xyz / prev_clip.w;
        let prev_uv = prev_ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
        if all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv <= vec2<f32>(1.0)) {
            let hist_dims = textureDimensions(history_scatter);
            let hist_dims_f = vec2<f32>(hist_dims);
            let cf = prev_uv * hist_dims_f - 0.5;
            let base = vec2<i32>(floor(cf));
            let fr = cf - vec2<f32>(base);
            let spatial = vec4<f32>(
                (1.0 - fr.x) * (1.0 - fr.y),
                fr.x * (1.0 - fr.y),
                (1.0 - fr.x) * fr.y,
                fr.x * fr.y,
            );
            var h_rgb = vec3<f32>(0.0);
            var h_a = 0.0;
            var w_sum = 0.0;
            for (var k = 0u; k < 4u; k++) {
                let dx = i32(k & 1u);
                let dy = i32((k >> 1u) & 1u);
                let c = base + vec2<i32>(dx, dy);
                if c.x < 0 || c.y < 0
                    || u32(c.x) >= hist_dims.x || u32(c.y) >= hist_dims.y {
                    continue;
                }
                let t = textureLoad(history_scatter, c, 0);
                // alpha < 0 is the "was voxel-occluded" marker — reject.
                if t.a < 0.0 { continue; }
                let w = spatial[k];
                h_rgb += t.rgb * w;
                h_a += t.a * w;
                w_sum += w;
            }
            if w_sum >= 0.25 {
                h_rgb /= w_sum;
                h_a /= w_sum;
                let current_uv = (vec2<f32>(gid.xy) + 0.5)
                               / vec2<f32>(f32(params.width), f32(params.height));
                let motion = length(prev_uv - current_uv);
                let validity = 1.0 - smoothstep(0.04, 0.15, motion);
                let alpha = mix(1.0, cloud_params.quality2.x, validity);
                final_cloud_scatter = mix(h_rgb, cloud_scatter, alpha);
                final_cloud_trans = mix(h_a, cloud_trans, alpha);
            }
        }
    }

    textureStore(cloud_out, coord, vec4<f32>(final_cloud_scatter, final_cloud_trans));
}
