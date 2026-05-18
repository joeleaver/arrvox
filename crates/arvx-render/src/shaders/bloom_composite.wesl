struct BloomCompositeParams {
    display_width: u32,
    display_height: u32,
    bloom_intensity: f32,  // overall bloom strength (default 0.3)
    _pad: u32,
}

@group(0) @binding(0) var input_hdr: texture_2d<f32>;     // upscaled HDR at display resolution
@group(0) @binding(1) var bloom_mip0: texture_2d<f32>;     // bloom mip 0 (internal res)
@group(0) @binding(2) var bloom_mip1: texture_2d<f32>;     // bloom mip 1 (1/2 internal)
@group(0) @binding(3) var bloom_mip2: texture_2d<f32>;     // bloom mip 2 (1/4 internal)
@group(0) @binding(4) var bloom_mip3: texture_2d<f32>;     // bloom mip 3 (1/8 internal)
@group(0) @binding(5) var bloom_sampler: sampler;           // bilinear sampler
@group(0) @binding(6) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> params: BloomCompositeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.display_width || gid.y >= params.display_height) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.display_width), f32(params.display_height));

    // Read upscaled HDR color
    let hdr_color = textureLoad(input_hdr, vec2<i32>(gid.xy), 0).rgb;

    // Sample all bloom mip levels with bilinear filtering at display UV
    // Weight: smaller mips contribute more for wider bloom
    let b0 = textureSampleLevel(bloom_mip0, bloom_sampler, uv, 0.0).rgb * 0.5;
    let b1 = textureSampleLevel(bloom_mip1, bloom_sampler, uv, 0.0).rgb * 0.3;
    let b2 = textureSampleLevel(bloom_mip2, bloom_sampler, uv, 0.0).rgb * 0.15;
    let b3 = textureSampleLevel(bloom_mip3, bloom_sampler, uv, 0.0).rgb * 0.05;

    let bloom = (b0 + b1 + b2 + b3) * params.bloom_intensity;

    // Additive blend
    let result = hdr_color + bloom;
    textureStore(output_hdr, vec2<i32>(gid.xy), vec4(result, 1.0));
}
