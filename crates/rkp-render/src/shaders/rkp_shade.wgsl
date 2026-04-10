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
    _pad_align0: f32,
    _pad_align1: f32,
    sky_color_top: vec3<f32>,
    _pad0: f32,
    sky_color_horizon: vec3<f32>,
    _pad1: f32,
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

// Group 1: SSAO texture (read, half-res)
@group(1) @binding(0) var ssao_tex: texture_2d<f32>;

// Group 2: output HDR color (write, full-res)
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Group 3: shading params + lights + materials
@group(3) @binding(0) var<uniform> shade_params: ShadeParams;
@group(3) @binding(1) var<storage, read> lights: array<Light>;
@group(3) @binding(2) var<storage, read> materials: array<Material>;

// Group 4: camera
@group(4) @binding(0) var<uniform> camera: CameraUniforms;

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

    // No geometry → sky color.
    if hit_t >= 9999.0 || hit_t <= 0.0 {
        let uv_y = f32(gid.y) / f32(dims.y);
        let sky = mix(shade_params.sky_color_horizon, shade_params.sky_color_top, uv_y);
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

    // Shadow from G-buffer normal.w (written by march pass).
    let shadow = textureLoad(gbuf_normal, coord, 0).w;

    // AO from half-res SSAO texture.
    let half_coord = vec2<i32>(gid.xy) / 2;
    let ao = textureLoad(ssao_tex, half_coord, 0).r;

    // Accumulate direct lighting.
    var lo = vec3<f32>(0.0);

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

        // Apply shadow to first light only (matches rkf-render convention).
        var light_shadow = 1.0;
        if li == 0u {
            light_shadow = shadow;
        }

        lo += (diffuse + specular) * radiance * n_dot_l * light_shadow;
    }

    // Ambient: hemisphere sky + AO.
    let sky_up = shade_params.sky_color_top * shade_params.ambient_intensity;
    let sky_down = shade_params.sky_color_horizon * shade_params.ambient_intensity * 0.3;
    let hemisphere = mix(sky_down, sky_up, dot(N, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5);
    let ambient = hemisphere * albedo * (1.0 - metallic) * ao;

    // Emission.
    let emission = mat.base_color.rgb * mat.emission_strength;

    let final_color = lo + ambient + emission;
    textureStore(output, coord, vec4<f32>(final_color, 1.0));
}
