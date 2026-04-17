// Bloom compute shader — three entry points:
//   1. extract   — threshold HDR color into mip0
//   2. downsample — 13-tap tent filter halving resolution
//   3. blur      — separable 9-tap Gaussian (horizontal or vertical)

struct BloomParams {
    threshold: f32,   // extract: luminance threshold; blur: 0.0=H, 1.0=V
    knee: f32,        // soft knee width (extract only)
    width: u32,       // destination width in pixels
    height: u32,      // destination height in pixels
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BloomParams;

// ---------------------------------------------------------------------------
// Entry point 1: extract
// Extracts bright pixels from the HDR shading output using a soft-knee curve.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let color = textureLoad(input_tex, vec2<i32>(gid.xy), 0).rgb;
    let brightness = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Soft knee: smooth contribution curve around threshold
    let soft = brightness - params.threshold + params.knee;
    let contribution = clamp(soft / (2.0 * params.knee + 0.0001), 0.0, 1.0);
    let weight = contribution * contribution;
    let result = color * weight;

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}

// ---------------------------------------------------------------------------
// Entry point 2: downsample
// 13-tap tent filter that halves resolution (CoD Advanced Warfare style).
// params.width/height are the DESTINATION (half-res) dimensions.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_w = params.width;
    let dst_h = params.height;
    if (gid.x >= dst_w || gid.y >= dst_h) { return; }

    // Map destination pixel to source 2×2 block origin
    let src_coord = vec2<i32>(gid.xy) * 2;
    let src_dims = vec2<i32>(textureDimensions(input_tex));

    // Helper: clamped load
    // Center 4 samples — weight 0.5 total (0.125 each)
    var color = textureLoad(input_tex, clamp(src_coord + vec2<i32>( 0,  0), vec2<i32>(0), src_dims - 1), 0).rgb * 0.125;
    color    += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 1,  0), vec2<i32>(0), src_dims - 1), 0).rgb * 0.125;
    color    += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 0,  1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.125;
    color    += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 1,  1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.125;

    // Corner samples — weight 0.125 total (0.03125 each)
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>(-1, -1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.03125;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 2, -1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.03125;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>(-1,  2), vec2<i32>(0), src_dims - 1), 0).rgb * 0.03125;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 2,  2), vec2<i32>(0), src_dims - 1), 0).rgb * 0.03125;

    // Edge samples — weight 0.375 total (0.0625 each)
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 0, -1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 1, -1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>(-1,  0), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 2,  0), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>(-1,  1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;
    color += textureLoad(input_tex, clamp(src_coord + vec2<i32>( 2,  1), vec2<i32>(0), src_dims - 1), 0).rgb * 0.0625;

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}

// ---------------------------------------------------------------------------
// Entry point 3: blur
// Separable 9-tap Gaussian (sigma ~= 2.0).
// params.threshold is repurposed as direction: 0.0 = horizontal, 1.0 = vertical.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let is_vertical = params.threshold > 0.5;
    let dir = select(vec2<i32>(1, 0), vec2<i32>(0, 1), is_vertical);
    let dims = vec2<i32>(textureDimensions(input_tex));
    let center = vec2<i32>(gid.xy);

    // 9-tap Gaussian weights (sigma ~= 2.0): w[0] center, w[1..4] offsets 1-4
    let w0 = 0.2270270270;
    let w1 = 0.1945945946;
    let w2 = 0.1216216216;
    let w3 = 0.0540540541;
    let w4 = 0.0162162162;

    var color = textureLoad(input_tex, center, 0).rgb * w0;

    color += textureLoad(input_tex, clamp(center + dir * 1, vec2<i32>(0), dims - 1), 0).rgb * w1;
    color += textureLoad(input_tex, clamp(center - dir * 1, vec2<i32>(0), dims - 1), 0).rgb * w1;

    color += textureLoad(input_tex, clamp(center + dir * 2, vec2<i32>(0), dims - 1), 0).rgb * w2;
    color += textureLoad(input_tex, clamp(center - dir * 2, vec2<i32>(0), dims - 1), 0).rgb * w2;

    color += textureLoad(input_tex, clamp(center + dir * 3, vec2<i32>(0), dims - 1), 0).rgb * w3;
    color += textureLoad(input_tex, clamp(center - dir * 3, vec2<i32>(0), dims - 1), 0).rgb * w3;

    color += textureLoad(input_tex, clamp(center + dir * 4, vec2<i32>(0), dims - 1), 0).rgb * w4;
    color += textureLoad(input_tex, clamp(center - dir * 4, vec2<i32>(0), dims - 1), 0).rgb * w4;

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
