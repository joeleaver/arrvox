// RKIPatch screen-space ambient occlusion (SSAO) compute shader.
//
// Half-resolution: each invocation processes one half-res pixel.
// Reads G-buffer position + normal, samples hemisphere in screen space,
// compares depths to estimate occlusion. Writes AO factor to Rgba8Unorm (.r).

const PI: f32 = 3.14159265;
const NUM_SAMPLES: u32 = 16u;

// --- Structs ---

struct SsaoParams {
    radius: f32,
    bias: f32,
    intensity: f32,
    _pad: u32,
    kernel: array<vec4<f32>, 16>,  // hemisphere samples (xyz = direction, w = unused)
}

struct CameraUniforms {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
}

// --- Bindings ---

// Group 0: G-buffer (depth + normal, read, full-res). World position is
// reconstructed from depth so the triangle pass can skip a position target.
@group(0) @binding(0) var gbuf_depth: texture_depth_2d;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;

// Group 1: output AO texture (write, half-res)
@group(1) @binding(0) var ao_out: texture_storage_2d<rgba8unorm, write>;

// Group 2: SSAO params + noise
@group(2) @binding(0) var<uniform> params: SsaoParams;
@group(2) @binding(1) var noise_tex: texture_2d<f32>;

// Group 3: camera (for inverse_view_proj).
@group(3) @binding(0) var<uniform> camera: CameraUniforms;

fn reconstruct_world_pos(coord: vec2<i32>, depth: f32, resolution: vec2<f32>, inv_vp: mat4x4<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(coord) + 0.5) / resolution;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = inv_vp * ndc;
    return world_h.xyz / world_h.w;
}

// --- Helpers ---

// Build a TBN matrix to orient hemisphere samples along the surface normal.
fn build_tbn(normal: vec3<f32>, random_vec: vec2<f32>) -> mat3x3<f32> {
    let rv = vec3<f32>(random_vec, 0.0);
    let tangent = normalize(rv - normal * dot(rv, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(ao_out);
    if gid.x >= out_dims.x || gid.y >= out_dims.y {
        return;
    }

    // Half-res → full-res pixel (sample center of 2x2 block).
    let full_coord = vec2<i32>(gid.xy) * 2 + vec2<i32>(1, 1);
    let full_dims = vec2<i32>(textureDimensions(gbuf_depth));
    let full_dims_f = vec2<f32>(full_dims);
    let depth = textureLoad(gbuf_depth, full_coord, 0);

    // No geometry hit → no occlusion.
    if depth >= 1.0 {
        textureStore(ao_out, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let world_pos = reconstruct_world_pos(full_coord, depth, full_dims_f, camera.inverse_view_proj);

    let normal = normalize(textureLoad(gbuf_normal, full_coord, 0).xyz);

    // Random rotation from noise texture (4x4, tiled).
    let noise_dims = textureDimensions(noise_tex);
    let noise_coord = vec2<i32>(gid.xy) % vec2<i32>(noise_dims);
    let noise = textureLoad(noise_tex, noise_coord, 0).xy * 2.0 - 1.0;

    let tbn = build_tbn(normal, noise);

    var occlusion = 0.0;
    let radius = params.radius;

    for (var i = 0u; i < NUM_SAMPLES; i++) {
        // Orient sample to hemisphere around normal.
        let sample_dir = tbn * params.kernel[i].xyz;
        let sample_world = world_pos + sample_dir * radius;

        // Find nearest G-buffer pixel in the sample direction.
        // Use a simple pixel-radius approach: offset the full-res coordinate
        // by a fixed pixel radius scaled by the kernel sample magnitude.
        let pixel_radius = 16.0; // sample within 16 pixels
        let sample_mag = length(params.kernel[i].xyz);
        let offset_2d = vec2<f32>(sample_dir.x, -sample_dir.y) * pixel_radius * sample_mag;
        let sample_coord = clamp(
            full_coord + vec2<i32>(offset_2d),
            vec2<i32>(0),
            full_dims - vec2<i32>(1),
        );

        let sample_depth = textureLoad(gbuf_depth, sample_coord, 0);

        // No geometry at sample → not occluded by this sample.
        if sample_depth >= 1.0 {
            continue;
        }
        let sample_pos = reconstruct_world_pos(sample_coord, sample_depth, full_dims_f, camera.inverse_view_proj);

        // Check if the geometry at the sample pixel is closer to the original
        // surface than our hemisphere sample point — if so, it occludes.
        let to_sample = sample_pos - world_pos;
        let dist = length(to_sample);

        // Range check: ignore samples too far away.
        if dist > radius { continue; }

        // Depth check: does the sampled geometry sit inside our hemisphere?
        // Positive dot with normal = above surface = potential occluder.
        let depth_diff = dot(to_sample, normal);
        if depth_diff > params.bias && depth_diff < radius {
            let range_factor = smoothstep(0.0, 1.0, radius / max(dist, 0.001));
            occlusion += range_factor;
        }
    }

    let ao = 1.0 - clamp((occlusion / f32(NUM_SAMPLES)) * params.intensity, 0.0, 1.0);
    textureStore(ao_out, vec2<i32>(gid.xy), vec4<f32>(ao, 0.0, 0.0, 0.0));
}
