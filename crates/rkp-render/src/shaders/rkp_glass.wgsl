// Glass composite post-pass.
//
// Reads the fully-shaded HDR (opaque behind already includes PBR +
// shadows + SSAO + sky for miss pixels) and, for every pixel whose
// primary ray passed through a transparent voxel, composites a
// glass look: screen-space refraction of the behind, Beer tint,
// Fresnel-weighted sky reflection off the glass surface.
//
// Non-glass pixels pass through unchanged.

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
struct Material {
    albedo_r: f32, albedo_g: f32, albedo_b: f32,
    roughness: f32,
    metallic: f32,
    emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32,
    subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    shader_id: u32,
    _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
}

@group(0) @binding(0) var hdr_in: texture_2d<f32>;
@group(0) @binding(1) var gbuf_glass: texture_2d<u32>;
@group(0) @binding(2) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<storage, read> materials: array<Material>;

fn mat_albedo(m: Material) -> vec3<f32> {
    return vec3<f32>(m.albedo_r, m.albedo_g, m.albedo_b);
}

// Mirror of octree_march's unpack_oct_normal (and rkp_shade's).
fn unpack_oct_normal(packed: u32) -> vec3<f32> {
    let ul = packed & 0xFFFFu;
    let vl = (packed >> 16u) & 0xFFFFu;
    let ux = f32(i32(ul << 16u) >> 16) / 32767.0;
    let vx = f32(i32(vl << 16u) >> 16) / 32767.0;
    var n = vec3<f32>(ux, vx, 1.0 - abs(ux) - abs(vx));
    if n.z < 0.0 {
        let nx = (1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0);
        let ny = (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0);
        n.x = nx;
        n.y = ny;
    }
    return normalize(n);
}

fn dielectric_f0(ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    return r * r;
}

fn fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let f0 = dielectric_f0(ior);
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn beer_absorption(glass_color: vec3<f32>, thickness: f32) -> vec3<f32> {
    let sigma = max(-log(max(glass_color, vec3<f32>(0.01))), vec3<f32>(0.0));
    return exp(-sigma * thickness * 5.0);
}

// Project a world-space point onto screen pixel coords. Returns
// vec2<i32> in the HDR texture's coord space, or a sentinel out-of-
// range value via clamp in the caller.
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

    // Reconstruct this pixel's world-space ray direction.
    let uv = (vec2<f32>(gid.xy) + 0.5) / dims_f;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz,
    );
    let V = -ray_dir;

    // Screen-space refraction — bend the ray at the glass front face,
    // trace it `thickness` worth of world-space into the bent
    // direction, project both the straight and bent endpoints onto
    // screen, sample HDR at the offset. This approximates "the
    // behind pixel at the position refraction would have routed the
    // ray to," using the already-shaded HDR as the source. Entry-
    // only refraction — over-bends vs. full entry+exit Snell on
    // thick flat glass, but reads as real refraction.
    var refract_dir = refract(ray_dir, glass_N, 1.0 / max(glass_ior, 1.0001));
    if dot(refract_dir, refract_dir) < 1e-6 {
        refract_dir = ray_dir; // total internal reflection fallback
    }

    let anchor_world = camera.position.xyz + ray_dir * thickness;
    let refract_world = camera.position.xyz + refract_dir * thickness;
    let anchor_px = world_to_screen(anchor_world, dims_f);
    let refract_px = world_to_screen(refract_world, dims_f);
    let offset_px = refract_px - anchor_px;

    // Clamp the sample coord to texture bounds — off-screen samples
    // would otherwise return black and punch holes at grazing glass
    // silhouettes. Reading the edge pixel is a cheap approximation
    // that stays well-behaved.
    let sample_f = vec2<f32>(coord) + offset_px;
    let sample_i = vec2<i32>(
        clamp(i32(sample_f.x), 0, i32(dims_u.x) - 1),
        clamp(i32(sample_f.y), 0, i32(dims_u.y) - 1),
    );
    let behind = textureLoad(hdr_in, sample_i, 0).rgb;

    // Fresnel + reflection sample. Project the reflected ray onto
    // screen and sample HDR there — cheap env-map approximation.
    // Off-screen reflections (common: glass aimed at the sky above
    // the frame) clamp to the edge and read whatever's at the image
    // top / bottom; for a sky-heavy upper edge this reads as "sky
    // reflection," which is the common case. If the reflection
    // should probe outside the camera frustum (e.g. looking down at
    // glass reflecting sky) we fall back to `behind` — keeps the
    // result plausibly lit rather than pinned to a random edge.
    let reflect_dir = reflect(-V, glass_N);
    let reflect_world = camera.position.xyz + reflect_dir * 50.0;
    let reflect_px = world_to_screen(reflect_world, dims_f);
    var reflect_sample = vec3<f32>(0.0);
    let rx = i32(reflect_px.x);
    let ry = i32(reflect_px.y);
    if rx >= 0 && ry >= 0 && rx < i32(dims_u.x) && ry < i32(dims_u.y) {
        reflect_sample = textureLoad(hdr_in, vec2<i32>(rx, ry), 0).rgb;
    } else {
        reflect_sample = behind;
    }

    let cos_vn = max(dot(V, glass_N), 0.0);
    let fresnel = fresnel_dielectric(cos_vn, glass_ior);
    let absorption = beer_absorption(glass_albedo, thickness);
    let transmitted = behind * absorption;
    let result = mix(transmitted, reflect_sample, fresnel);
    textureStore(hdr_out, coord, vec4<f32>(result, 1.0));
}
