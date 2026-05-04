// Glass composite post-pass.
//
// Reads the fully-shaded HDR (opaque behind already includes PBR +
// shadows + SSAO + sky for miss pixels) and, for every pixel whose
// primary ray passed through a transparent voxel, composites a glass
// look:
//  - Screen-space refraction of the behind, Beer-tinted by the
//    material's albedo and glass thickness.
//  - Opacity directly attenuates the transmitted leg — higher opacity
//    = less see-through, the reflection/highlight side takes over.
//  - Fresnel-weighted screen-space reflection, blurred by roughness as
//    a cheap stand-in for a rough-glass env lobe.
//  - Direct specular highlight from every directional light via a
//    GGX microfacet BRDF driven by roughness.
//  - Metallic tints reflection by albedo and kills transmission
//    (metallic=1 ≈ colored mirror).
//
// Non-glass pixels pass through unchanged.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::types::GpuMaterial;

const PI: f32 = 3.14159265358979323846;

// Camera uniform — matches rkp_shade.wgsl's layout.
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

// vec3 fields flattened to f32 channels — see rkp_shade.wgsl for why.

// Mirrors rkp_shade.wgsl's Light layout (position.w encodes type).
struct Light {
    position: vec4<f32>,
    color: vec4<f32>,
    direction: vec4<f32>,
    params: vec4<f32>,
}

// Only the handful of fields we actually use — layout must match
// rkp_shade.wgsl::ShadeParams so the same uniform buffer binds cleanly.
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

@group(0) @binding(0) var hdr_in: texture_2d<f32>;
@group(0) @binding(1) var gbuf_glass: texture_2d<u32>;
@group(0) @binding(2) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(5) var<uniform> shade_params: ShadeParams;
@group(0) @binding(6) var<storage, read> lights: array<Light>;

fn mat_albedo(m: GpuMaterial) -> vec3<f32> {
    return vec3<f32>(m.albedo_r, m.albedo_g, m.albedo_b);
}

// `unpack_oct_normal` is imported from `lib::oct_normal` at the top.
// Wave A2: this site previously used a Variant-B copy without the
// zero-length guard. The lib version (Variant A) returns +Y for
// degenerate inputs instead of NaN.

fn dielectric_f0(ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

fn fresnel_schlick_f0(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    return fresnel_schlick_f0(cos_theta, dielectric_f0(ior));
}

fn beer_absorption(glass_color: vec3<f32>, thickness: f32) -> vec3<f32> {
    let sigma = max(-log(max(glass_color, vec3<f32>(0.01))), vec3<f32>(0.0));
    return exp(-sigma * thickness * 5.0);
}

// Standard GGX / Trowbridge-Reitz microfacet distribution.
fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith masking-shadowing with GGX.
fn smith_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let gv = n_dot_v + sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v);
    let gl = n_dot_l + sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l);
    return (2.0 * n_dot_v) * (2.0 * n_dot_l) / max(gv * gl, 1e-7);
}

fn sample_hdr_clamped(c: vec2<f32>, dims: vec2<i32>) -> vec3<f32> {
    let i = vec2<i32>(
        clamp(i32(c.x), 0, dims.x - 1),
        clamp(i32(c.y), 0, dims.y - 1),
    );
    return textureLoad(hdr_in, i, 0).rgb;
}

// 5-tap cross blur whose radius scales with pixel-space spread. Cheap
// stand-in for a rough-surface env lobe: we don't have mip-filtered
// radiance, so we approximate the rough-reflection convolution with a
// screen-space box.
fn sample_hdr_spread(c: vec2<f32>, dims: vec2<i32>, radius_px: f32) -> vec3<f32> {
    if radius_px < 0.75 {
        return sample_hdr_clamped(c, dims);
    }
    let c0 = sample_hdr_clamped(c,                                      dims);
    let c1 = sample_hdr_clamped(c + vec2<f32>( radius_px,  0.0),        dims);
    let c2 = sample_hdr_clamped(c + vec2<f32>(-radius_px,  0.0),        dims);
    let c3 = sample_hdr_clamped(c + vec2<f32>( 0.0,        radius_px),  dims);
    let c4 = sample_hdr_clamped(c + vec2<f32>( 0.0,       -radius_px),  dims);
    return (c0 + c1 + c2 + c3 + c4) * 0.2;
}

fn world_to_screen(p_world: vec3<f32>, dims: vec2<f32>) -> vec2<f32> {
    let clip = camera.view_proj * vec4<f32>(p_world, 1.0);
    if clip.w <= 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    let ndc = clip.xy / clip.w;
    return vec2<f32>(
        (ndc.x * 0.5 + 0.5) * dims.x,
        (-ndc.y * 0.5 + 0.5) * dims.y,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims_u = textureDimensions(hdr_in);
    if gid.x >= dims_u.x || gid.y >= dims_u.y { return; }
    let coord = vec2<i32>(gid.xy);
    let dims_f = vec2<f32>(dims_u);

    let hdr_here = textureLoad(hdr_in, coord, 0).rgb;
    let glass_raw = textureLoad(gbuf_glass, coord, 0).rg;
    let thickness_mm = (glass_raw.y >> 16u) & 0xFFFFu;

    // No glass → pass through.
    if thickness_mm == 0u {
        textureStore(hdr_out, coord, vec4<f32>(hdr_here, 1.0));
        return;
    }

    let glass_mat_id = glass_raw.y & 0xFFFFu;
    let glass_N = normalize(unpack_oct_normal(glass_raw.x));
    let thickness = f32(thickness_mm) * 0.001;
    let gm = materials[glass_mat_id];
    let glass_albedo = mat_albedo(gm);
    let glass_ior = gm.ior;
    // GGX blows up analytically at roughness=0; clamp a floor in so the
    // specular highlight is still a finite point rather than a divide
    // by zero.
    let glass_roughness = clamp(gm.roughness, 0.04, 1.0);
    let glass_metallic = clamp(gm.metallic, 0.0, 1.0);
    // The march classifies anything with opacity < 0.99 as glass, so
    // within the glass regime `opacity` lives on [0, 0.99). Renormalise
    // to [0, 1] so the slider covers its full range visually.
    let glass_opacity = clamp(gm.opacity / 0.99, 0.0, 1.0);

    // Reconstruct this pixel's world-space ray direction.
    let uv = (vec2<f32>(gid.xy) + 0.5) / dims_f;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz,
    );
    let V = -ray_dir;
    let n_dot_v = max(dot(V, glass_N), 1e-4);

    // --- Transmission leg --------------------------------------------
    // Entry-only screen-space refraction: bend the view ray at the
    // glass front face, trace `thickness` worth of world-space into
    // the bent direction, subtract the straight and refracted
    // projections to get a pixel offset, sample HDR at `coord +
    // offset`. Over-bends vs. full entry+exit Snell on flat panes
    // but reads as real refraction on curves/tilts.
    let dims_i = vec2<i32>(i32(dims_u.x), i32(dims_u.y));
    var refract_dir = refract(ray_dir, glass_N, 1.0 / max(glass_ior, 1.0001));
    if dot(refract_dir, refract_dir) < 1e-6 {
        refract_dir = ray_dir; // total internal reflection fallback
    }
    let anchor_world = camera.position.xyz + ray_dir * thickness;
    let refract_world = camera.position.xyz + refract_dir * thickness;
    let anchor_px = world_to_screen(anchor_world, dims_f);
    let refract_px = world_to_screen(refract_world, dims_f);
    var offset_px = refract_px - anchor_px;

    // If the refracted sample would land off-screen, drop refraction
    // for this pixel. Without this, `sample_hdr_clamped` snaps the
    // out-of-range coord to the nearest frame edge, producing
    // horizontal / vertical streaks of whatever's at that edge —
    // most visible when looking near-parallel to a flat pane, where
    // the tangent path through the glass inflates `thickness` enough
    // to send the offset off frame.
    let sample_test = vec2<f32>(coord) + offset_px;
    if sample_test.x < 0.0 || sample_test.x >= dims_f.x
        || sample_test.y < 0.0 || sample_test.y >= dims_f.y {
        offset_px = vec2<f32>(0.0);
    }

    let sample_f = vec2<f32>(coord) + offset_px;
    let behind = sample_hdr_clamped(sample_f, dims_i);

    let absorption = beer_absorption(glass_albedo, thickness);

    // Metallic kills transmission. Opacity scales what survives Beer
    // tinting — an opacity=0 glass is crystal clear, opacity→1 is
    // almost fully blocked on the transmission side (what fills the
    // screen is the reflection + specular, weighted by Fresnel).
    let transmission_gate = (1.0 - glass_metallic) * (1.0 - glass_opacity);
    let transmitted = behind * absorption * transmission_gate;

    // --- Reflection leg (environment probe) --------------------------
    // Screen-space reflection: project the reflected ray's endpoint
    // onto screen and sample HDR there, blurred by roughness as a
    // stand-in for a prefiltered env map. Off-screen reflections fall
    // back to `behind` so silhouettes don't pin to a random edge.
    let reflect_dir = reflect(-V, glass_N);
    let reflect_world = camera.position.xyz + reflect_dir * 50.0;
    let reflect_px = world_to_screen(reflect_world, dims_f);
    let reflect_blur_px = glass_roughness * 16.0;
    var reflect_env = vec3<f32>(0.0);
    if reflect_px.x >= 0.0 && reflect_px.y >= 0.0
        && reflect_px.x < dims_f.x && reflect_px.y < dims_f.y {
        reflect_env = sample_hdr_spread(reflect_px, dims_i, reflect_blur_px);
    } else {
        reflect_env = behind;
    }

    // --- Direct specular highlights from punctual lights -------------
    // A proper GGX microfacet BRDF for every directional light in the
    // scene. Roughness drives both the D (lobe width) and G (masking)
    // terms, so the highlight shrinks/sharpens naturally as roughness
    // drops. We skip point/spot lights: the glass surface's world
    // position isn't in gbuf_glass today, so we can't evaluate their
    // attenuation terms correctly. The dominant light source (the sun)
    // is directional, so glass gets the iconic "hot specular pip" knob.
    // Shadows are intentionally skipped here — the existing shadow
    // texture is aligned to the opaque behind, not the glass front
    // face. An always-lit spec highlight is better than a plausible-
    // but-misaligned one for a sub-ms post-pass.
    var spec_direct = vec3<f32>(0.0);
    let num_lights = shade_params.num_lights;
    for (var li = 0u; li < num_lights; li = li + 1u) {
        let light = lights[li];
        let light_type = u32(light.position.w);
        if light_type != 0u { continue; }
        let L = normalize(-light.direction.xyz);
        let n_dot_l = dot(glass_N, L);
        if n_dot_l <= 0.0 { continue; }
        let H = normalize(L + V);
        let n_dot_h = max(dot(glass_N, H), 0.0);
        let v_dot_h = max(dot(V, H), 0.0);

        let f0 = dielectric_f0(glass_ior);
        let f = fresnel_schlick_f0(v_dot_h, f0);
        let d = ggx_d(n_dot_h, glass_roughness);
        let g = smith_g(n_dot_v, max(n_dot_l, 1e-4), glass_roughness);
        let spec_bsdf = f * d * g / (4.0 * n_dot_v * max(n_dot_l, 1e-4));
        let radiance = light.color.rgb * light.color.w;
        spec_direct = spec_direct + spec_bsdf * radiance * n_dot_l;
    }

    // --- Combine ------------------------------------------------------
    // Fresnel mixes the env reflection vs. the transmitted view. The
    // direct GGX specular is added on top — it's the light-source
    // contribution from the reflection side of the BRDF, separate
    // from the env probe (which is a screen-space approximation of
    // incoming radiance from every other direction).
    let fresnel_vn = fresnel_dielectric(n_dot_v, glass_ior);
    let reflect_tint = mix(vec3<f32>(1.0), glass_albedo, glass_metallic);
    let mixed = mix(transmitted, reflect_env * reflect_tint, fresnel_vn);
    let result = mixed + spec_direct * reflect_tint;
    textureStore(hdr_out, coord, vec4<f32>(result, 1.0));
}
