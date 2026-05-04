// RKIPatch cloud history update — copies the current half-res cloud_out
// into the cloud history texture for sky pixels, and writes an invalidation
// marker (alpha = -1) for non-sky pixels. The marker tells next frame's TAA
// sampler that history at that pixel is stale (the pixel was occluded and
// now sky rays reprojecting onto it must not blend in pre-occlusion data).

@group(0) @binding(0) var cloud_in: texture_2d<f32>;
@group(0) @binding(1) var depth_buffer: texture_2d<f32>;
@group(0) @binding(2) var history_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn update_history(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(history_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    // Depth lookup mirrors the march pass: a half-res pixel is sky only if all
    // four full-res pixels it represents are sky.
    let full_base = coord * 2;
    let d0 = textureLoad(depth_buffer, full_base, 0).w;
    let d1 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 0), 0).w;
    let d2 = textureLoad(depth_buffer, full_base + vec2<i32>(0, 1), 0).w;
    let d3 = textureLoad(depth_buffer, full_base + vec2<i32>(1, 1), 0).w;
    let all_sky = (d0 >= 9999.0 || d0 <= 0.0)
               && (d1 >= 9999.0 || d1 <= 0.0)
               && (d2 >= 9999.0 || d2 <= 0.0)
               && (d3 >= 9999.0 || d3 <= 0.0);
    if all_sky {
        // Real sky — copy current cloud scatter into history.
        let value = textureLoad(cloud_in, coord, 0);
        textureStore(history_out, coord, value);
    } else {
        // Occluded — write an "invalid" marker (alpha = -1, outside the valid
        // [0,1] transmittance range). Sky rays that reproject to this pixel
        // next frame detect the marker and use current-only instead of blending
        // with stale sky values from N frames ago before the occlusion began.
        textureStore(history_out, coord, vec4<f32>(0.0, 0.0, 0.0, -1.0));
    }
}
