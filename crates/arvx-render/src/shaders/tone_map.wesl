// Tone mapping compute shader — Phase 10 upgrade.
//
// Reads HDR Rgba16Float, applies exposure, selectable tone mapping curve
// (ACES or AgX), and sRGB gamma correction. Writes LDR Rgba8Unorm.

// ---------- Bindings ----------

@group(0) @binding(0) var hdr_input: texture_2d<f32>;
@group(1) @binding(0) var ldr_output: texture_storage_2d<rgba8unorm, write>;

struct ToneMapParams {
    mode: u32,       // 0 = ACES, 1 = AgX
    exposure: f32,   // manual exposure multiplier (default 1.0)
    _pad0: u32,
    _pad1: u32,
}
@group(2) @binding(0) var<uniform> params: ToneMapParams;
@group(2) @binding(1) var<storage, read> exposure_data: array<f32, 2>;  // [0]=current_ev, [1]=target_ev

// ---------- ACES Tone Mapping ----------

// ACES filmic tone mapping curve (scalar, for luminance-based mapping).
fn aces_curve(v: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
}

// ACES filmic tone mapping — luminance-based to preserve color saturation.
// Applying the curve per-channel (R,G,B independently) desaturates because
// bright channels are compressed more, pulling all channels toward each other.
// Instead: map luminance through ACES, then scale the color to match.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let luma = dot(x, vec3<f32>(0.2126, 0.7152, 0.0722));
    let mapped_luma = aces_curve(luma);
    let scale = mapped_luma / max(luma, 1e-6);
    let mapped = x * scale;
    // Desaturate toward mapped luminance where channels would clip,
    // preventing out-of-gamut artifacts from highly saturated HDR inputs.
    let max_c = max(mapped.r, max(mapped.g, mapped.b));
    if max_c > 1.0 {
        return mix(vec3<f32>(mapped_luma), mapped, vec3<f32>(1.0 / max_c));
    }
    return mapped;
}

// ---------- AgX Tone Mapping ----------

// AgX tone mapping (Troy Sobotka, Blender 3.5+).
// Better highlight and shadow preservation than ACES.
// Uses a log-domain S-curve with configurable look.

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    // 6th order polynomial fit of the AgX default contrast curve
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
         - 40.14 * x4 * x
         + 31.96 * x4
         - 6.868 * x2 * x
         + 0.4298 * x2
         + 0.1191 * x
         - 0.00232;
}

fn agx_tonemap(color: vec3<f32>) -> vec3<f32> {
    // AgX log2 encoding: compress HDR range to [0, 1]
    let min_ev = -12.47393;
    let max_ev = 4.026069;

    // Convert to AgX log space
    var c = max(color, vec3(1e-10));
    c = log2(c);
    c = (c - min_ev) / (max_ev - min_ev);
    c = clamp(c, vec3(0.0), vec3(1.0));

    // Apply contrast curve
    c = agx_default_contrast_approx(c);

    return c;
}

// AgX "punchy" look — optional saturation boost for more vivid output
fn agx_look(color: vec3<f32>) -> vec3<f32> {
    let luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    let offset = pow(luma, 0.1);
    var c = color * offset / max(luma, 1e-6);
    // Mild saturation boost
    c = mix(vec3(luma), c, vec3(1.1));
    return c;
}

// ---------- sRGB ----------

/// Linear to sRGB gamma correction.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, linear <= cutoff);
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(ldr_output));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);
    var hdr = textureLoad(hdr_input, coord, 0).rgb;

    // Apply auto-exposure (adapted EV → multiplier) and manual exposure override
    let auto_ev = exposure_data[0];
    let auto_multiplier = pow(2.0, auto_ev);
    hdr = hdr * params.exposure * auto_multiplier;

    // Tone map based on selected mode
    var mapped: vec3<f32>;
    if (params.mode == 1u) {
        mapped = agx_tonemap(hdr);
    } else {
        mapped = aces_tonemap(hdr);
    }

    // NOTE: No sRGB gamma here — the blit pass renders to an sRGB swapchain
    // surface, which applies the linear→sRGB conversion in hardware.
    // Applying linear_to_srgb() here would cause double gamma (washed out,
    // flat, desaturated output).
    textureStore(ldr_output, coord, vec4<f32>(clamp(mapped, vec3(0.0), vec3(1.0)), 1.0));
}
