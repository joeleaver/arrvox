// Screen-space god rays — radial blur of the composited scene, with geometry
// masked out. Classic Mittring/Crytek formulation.
//
// For each pixel, marches toward the sun's screen position and accumulates
// the per-sample luminance of the already-composited HDR scene. The sun disc
// saturates fp16 (~65504) so it dominates the blur and naturally concentrates
// the rays outward from the sun — no synthetic radial falloff required.
// Two masks shape the accumulation:
//   * is_sky — voxel-hit pixels contribute 0, so voxel silhouettes cut
//     clean dark shafts into the rays.
//   * cloud_trans — thick clouds covering the sun dim their march samples,
//     so clouds soften the beam; gaps sharpen it into fingers.

struct GodRayParams {
    sun_screen_pos: vec2<f32>, // sun UV [0,1]
    sun_on_screen:  f32,       // 1.0 if in front of camera
    density:        f32,       // march-length scale (≤ 1 reaches the sun)
    weight:         f32,       // per-sample mask weight
    decay:          f32,       // per-step exponential falloff
    exposure:       f32,       // overall brightness multiplier
    num_samples:    u32,
    sun_color:      vec3<f32>, // linear-RGB, atmospherically tinted
    _pad:           f32,
}

@group(0) @binding(0) var<uniform> params: GodRayParams;
// Full-res HDR scene the rays get added onto.
@group(0) @binding(1) var composite:     texture_2d<f32>;
// Full-res G-buffer position; .w is linear-eye-space depth (≥ 9999 = sky miss).
@group(0) @binding(2) var gbuf_position: texture_2d<f32>;
// Half-res cloud buffer; .a is cloud transmittance.
@group(0) @binding(3) var cloud_tex:     texture_2d<f32>;
@group(0) @binding(4) var output:        texture_storage_2d<rgba16float, write>;

// Interleaved-gradient noise in [0,1) — hashes screen coord to a smooth
// per-pixel value. Used to jitter the march start by a fraction of a step
// so the "geometry → sky" transition along the ray averages across a few
// pixels instead of snapping on a pixel boundary (otherwise high decay
// produces visible blocky artefacts).
fn igradient_noise(pixel: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pixel, magic.xy)));
}

// Bright-pass threshold — only samples whose HDR luminance exceeds this
// contribute to the march. Kills the "whole sky lifts with the sun"
// halo: the sun disc saturates fp16 (~65 500) and dominates hard, so rays
// now radiate only from genuinely sun-bright sources. Calibrated for the
// engine's default sun_intensity ≈ 110 000 lux; bump higher if the rays
// feel too diffuse, lower if they feel too tight on the disc.
const BRIGHT_THRESHOLD: f32 = 2000.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(composite);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let scene_color = textureLoad(composite, coord, 0);

    if params.sun_on_screen < 0.5 || params.exposure <= 0.0 {
        textureStore(output, coord, scene_color);
        return;
    }

    let delta = (params.sun_screen_pos - uv) * params.density / f32(params.num_samples);
    let dims_f = vec2<f32>(dims);
    let cloud_dims = textureDimensions(cloud_tex);
    let cloud_dims_f = vec2<f32>(cloud_dims);

    // Start the march one jittered fraction of a step in so adjacent
    // pixels sample at different positions along the ray. Prevents the
    // block-stepping that otherwise appears at high decay.
    let jitter = igradient_noise(vec2<f32>(gid.xy));
    var sample_uv = uv + delta * jitter;
    var illumination = 0.0;
    var current_decay = 1.0;

    for (var i = 0u; i < params.num_samples; i++) {
        sample_uv += delta;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let full_coord = vec2<i32>(clamped * dims_f);
        let half_coord = vec2<i32>(clamped * cloud_dims_f);

        let depth = textureLoad(gbuf_position, full_coord, 0).w;
        let is_sky = depth >= 9999.0 || depth <= 0.0;

        if is_sky {
            // Perceptual luminance of the composite, thresholded so only
            // genuinely bright samples (sun disc + sunlit cloud edges)
            // propagate. Everyday sky contributes 0 → no generic halo.
            let s = textureLoad(composite, full_coord, 0).rgb;
            let lum = dot(s, vec3<f32>(0.2126, 0.7152, 0.0722));
            let bright = max(lum - BRIGHT_THRESHOLD, 0.0);
            let cloud_trans = textureLoad(cloud_tex, half_coord, 0).a;
            illumination += bright * cloud_trans * params.weight * current_decay;
        }
        // Geometry samples contribute 0 — voxel silhouettes cut dark shafts.

        current_decay *= params.decay;
    }

    // sun_color is a plain 0-1 tint (not intensity-scaled) — we already
    // have HDR magnitude from the scene-luminance accumulation. Exposure is
    // a perceptual scale that meets the engine's physical-lux tone-map.
    let rays = params.sun_color * illumination * params.exposure;
    textureStore(output, coord, vec4<f32>(scene_color.rgb + rays, scene_color.a));
}
