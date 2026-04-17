// RKIPatch cloud sun attenuation — one ray from camera toward the sun through
// the cloud slab. Writes exp(-τ) to a single-scalar storage buffer for CPU
// readback. Cloud density replicates `cloud_density` in rkp_volumetric.wgsl —
// if that shader changes, update this one too.

struct VolumetricParams {
    cam_pos:      vec4<f32>,
    cam_forward:  vec4<f32>,
    cam_right:    vec4<f32>,
    cam_up:       vec4<f32>,
    sun_dir:      vec4<f32>,   // xyz = toward sun
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
    fog_distance: vec4<f32>,
    frame_index:  u32,
    vol_ambient_r: f32,
    vol_ambient_g: f32,
    vol_ambient_b: f32,
    prev_view_proj: mat4x4<f32>,
}

struct CloudParams {
    altitude: vec4<f32>,   // x=min, y=max, z=threshold, w=density_scale
    noise:    vec4<f32>,   // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    wind:     vec4<f32>,
    flags:    vec4<f32>,   // x=enable, y=coverage
}

@group(0) @binding(0) var<uniform> params: VolumetricParams;
@group(0) @binding(1) var<uniform> cloud_params: CloudParams;
@group(0) @binding(2) var<storage, read_write> sun_atten_out: vec4<f32>;

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

fn cloud_density(pos: vec3<f32>) -> f32 {
    if cloud_params.flags.x < 0.5 { return 0.0; }

    let cloud_min = cloud_params.altitude.x;
    let cloud_max = cloud_params.altitude.y;
    let threshold = cloud_params.altitude.z;
    let density_scale = cloud_params.altitude.w;

    let height = pos.y;
    if height < cloud_min || height > cloud_max { return 0.0; }

    let height_above = height - cloud_min;
    let height_below = cloud_max - height;
    let height_grad = smoothstep(0.0, 50.0, height_above)
                    * smoothstep(0.0, 200.0, height_below);

    let wind_offset = vec2<f32>(cloud_params.wind.x, cloud_params.wind.y)
                    * cloud_params.wind.z * cloud_params.wind.w;
    let noise_pos = vec3<f32>(pos.x + wind_offset.x, pos.y, pos.z + wind_offset.y)
                  + vec3<f32>(173.5, 247.3, 391.7);

    let shape = fbm_3d(noise_pos * cloud_params.noise.x, 4u);

    let raw_weather = fbm_3d(noise_pos * (1.0 / max(cloud_params.noise.w, 1.0)), 2u);
    let coverage = cloud_params.flags.y;
    let weather = mix(raw_weather, 1.0, coverage * coverage);

    var base = shape * weather * height_grad;
    base = max(base - threshold, 0.0);

    let detail = fbm_3d(noise_pos * cloud_params.noise.y, 3u);
    base = max(base - detail * cloud_params.noise.z, 0.0);

    return base * density_scale;
}

@compute @workgroup_size(1, 1, 1)
fn sun_atten_main() {
    // Ray from camera toward the sun. Intersect with the cloud slab and
    // integrate density along the slab segment.
    let cam = params.cam_pos.xyz;
    let sun_dir = params.sun_dir.xyz;

    if sun_dir.y <= 0.001 || cloud_params.flags.x < 0.5 {
        sun_atten_out = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        return;
    }

    let cloud_min = cloud_params.altitude.x;
    let cloud_max = cloud_params.altitude.y;
    let t_lo = (cloud_min - cam.y) / sun_dir.y;
    let t_hi = (cloud_max - cam.y) / sun_dir.y;
    let t_enter = max(min(t_lo, t_hi), 0.0);
    let t_exit = max(t_lo, t_hi);

    if t_exit <= 0.0 || t_exit <= t_enter {
        sun_atten_out = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        return;
    }

    // 24 samples with a jitter based on frame_index — combined with the engine-side
    // temporal lerp, different frames probe slightly different positions, so even if
    // a cloud feature is thinner than the step spacing, it registers statistically.
    let num_steps = 24u;
    let step = (t_exit - t_enter) / f32(num_steps);
    let jitter = fract(f32(params.frame_index) * 0.61803398);
    var tau = 0.0;
    for (var i = 0u; i < num_steps; i++) {
        let t = t_enter + (f32(i) + jitter) * step;
        let pos = cam + sun_dir * t;
        tau += cloud_density(pos) * step;
    }

    sun_atten_out = vec4<f32>(exp(-tau), tau, 0.0, 0.0);
}
