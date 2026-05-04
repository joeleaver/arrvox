// RKIPatch fog march — height-fog participating medium, all pixels.
//
// Half-resolution compute shader. Marches every view ray from the camera to
// the scene depth (or to `far` for sky pixels), accumulating scattering and
// transmittance through an exponential-in-altitude fog density. Output:
// Rgba16Float (rgb=scatter, a=transmittance).
//
// Clouds are NOT handled here — they live in rkp_cloud_march.wgsl and run
// over sky pixels only, so sky and voxel work is never mixed inside one
// shader. This eliminates the is_sky branching and the (0,0,0,1)-neutral
// marker dance that the old combined pass needed.

const PI: f32 = 3.14159265;

// Matches the CPU-side VolumetricParams layout. Kept in sync with the cloud
// shader's declaration — any layout change in rkp_volumetric.rs must update
// both. Prev-view-proj lives at the tail and is only read by the cloud
// shader, so omitting it here is safe (WGSL reads only up to the declared
// struct size).
struct VolParams {
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
    fog_color:    vec4<f32>,   // xyz = scatter albedo
    fog_height:   vec4<f32>,   // x = base_density, y = base_height, z = falloff
    frame_index:  u32,
    vol_ambient_r: f32,
    vol_ambient_g: f32,
    vol_ambient_b: f32,
}

@group(0) @binding(0) var<uniform> params: VolParams;
@group(0) @binding(1) var depth_buffer: texture_2d<f32>;
@group(0) @binding(2) var fog_out: texture_storage_2d<rgba16float, write>;

// Forward-biased Henyey-Greenstein g for water/mist droplets. 0.3 is a broadly
// accepted default; exposing it as a knob never proved useful in practice.
const FOG_ASYMMETRY: f32 = 0.3;

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

fn height_fog_density(pos: vec3<f32>) -> f32 {
    let base_density = params.fog_height.x;
    let base_height = params.fog_height.y;
    let falloff = params.fog_height.z;
    return base_density * exp(-falloff * max(pos.y - base_height, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn fog_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.width), f32(params.height));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(params.cam_forward.xyz + ndc.x * params.cam_right.xyz + ndc.y * params.cam_up.xyz);

    // Scene depth: sample all 4 full-res pixels covered by this half-res tile,
    // take the closest non-sky depth as max_t so the fog march doesn't step
    // past any geometry inside the block.
    let full_base = vec2<i32>(gid.xy) * 2;
    let d0 = textureLoad(depth_buffer, full_base, 0).w;
    let d1 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 0), 0).w;
    let d2 = textureLoad(depth_buffer, full_base + vec2<i32>(0, 1), 0).w;
    let d3 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 1), 0).w;
    let is_sky0 = d0 >= 9999.0 || d0 <= 0.0;
    let is_sky1 = d1 >= 9999.0 || d1 <= 0.0;
    let is_sky2 = d2 >= 9999.0 || d2 <= 0.0;
    let is_sky3 = d3 >= 9999.0 || d3 <= 0.0;
    var min_depth = params.far;
    if !is_sky0 { min_depth = min(min_depth, d0); }
    if !is_sky1 { min_depth = min(min_depth, d1); }
    if !is_sky2 { min_depth = min(min_depth, d2); }
    if !is_sky3 { min_depth = min(min_depth, d3); }
    let max_t = min(min_depth, params.far);

    let jitter = interleaved_gradient_noise(vec2<f32>(gid.xy), params.frame_index);
    let cos_sun = dot(ray_dir, params.sun_dir.xyz);
    let scatter_albedo = params.fog_color.xyz;
    let sky_ambient = vec3<f32>(params.vol_ambient_r, params.vol_ambient_g, params.vol_ambient_b);

    var fog_scatter = vec3<f32>(0.0);
    var fog_trans = 1.0;

    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if t >= max_t { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;
        let near_fade = smoothstep(0.0, 20.0, t);
        let fog_dens = height_fog_density(pos) * near_fade;

        if fog_dens > 0.001 {
            let fog_L = henyey_greenstein(cos_sun, FOG_ASYMMETRY) * params.sun_color.xyz * scatter_albedo
                      + sky_ambient * scatter_albedo;
            let fog_absorbed = 1.0 - exp(-fog_dens * params.step_size);
            fog_scatter += fog_L * fog_absorbed * fog_trans;
            fog_trans *= 1.0 - fog_absorbed;
        }

        if fog_trans < 0.03 { break; }
    }

    textureStore(fog_out, coord, vec4<f32>(fog_scatter, fog_trans));
}
