// Screen-space god rays — radial blur from sun position.
//
// For each pixel, samples along the line from the pixel toward the sun's
// screen position. Bright pixels (sky gaps between clouds/objects) accumulate
// into visible light shafts. The result is additively blended onto the HDR image.

struct GodRayParams {
    sun_screen_pos: vec2<f32>,  // sun position in UV [0,1] space
    sun_on_screen: f32,         // 1.0 if sun is in front of camera, 0.0 if behind
    density: f32,               // ray density / overall strength (default 1.0)
    weight: f32,                // per-sample weight (default 0.01)
    decay: f32,                 // falloff per sample (default 0.97)
    exposure: f32,              // final brightness multiplier (default 0.3)
    num_samples: u32,           // number of samples along ray (default 64)
}

@group(0) @binding(0) var<uniform> params: GodRayParams;
@group(0) @binding(1) var source: texture_2d<f32>;       // HDR input (volumetric composite output)
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>; // HDR output with god rays

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(source);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // Original scene color.
    let scene_color = textureLoad(source, coord, 0);

    // If sun is behind camera, no god rays.
    if params.sun_on_screen < 0.5 {
        textureStore(output, coord, scene_color);
        return;
    }

    // Direction from pixel toward sun in UV space.
    let delta = (params.sun_screen_pos - uv) * params.density / f32(params.num_samples);

    // March from pixel toward sun, accumulating brightness.
    var sample_uv = uv;
    var illumination = vec3<f32>(0.0);
    var current_decay = 1.0;

    for (var i = 0u; i < params.num_samples; i++) {
        sample_uv += delta;

        // Clamp to screen bounds.
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_coord = vec2<i32>(clamped * vec2<f32>(dims));
        let sample_color = textureLoad(source, sample_coord, 0).rgb;

        // Accumulate with decay.
        illumination += sample_color * params.weight * current_decay;
        current_decay *= params.decay;
    }

    // Add god rays to original scene.
    let result = scene_color.rgb + illumination * params.exposure;
    textureStore(output, coord, vec4<f32>(result, scene_color.a));
}
