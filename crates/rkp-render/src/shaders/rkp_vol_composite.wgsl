// RKIPatch volumetric composite — blends volumetric scatter over scene HDR.
//
// Full-resolution compute shader. Reads scene HDR + half-res scatter/transmittance,
// writes composited HDR: result = scene * transmittance + scatter.

@group(0) @binding(0) var scene_hdr: texture_2d<f32>;
@group(0) @binding(1) var vol_scatter: texture_2d<f32>;
@group(0) @binding(2) var composited: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(composited);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let scene = textureLoad(scene_hdr, coord, 0);

    // Nearest-neighbor sample from half-res scatter+transmittance.
    let half_coord = coord / 2;
    let vol = textureLoad(vol_scatter, half_coord, 0);

    let result = scene.rgb * vol.a + vol.rgb;
    textureStore(composited, coord, vec4<f32>(result, 1.0));
}
