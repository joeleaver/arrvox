// Arrvox volumetric composite — layers fog, then clouds, over scene HDR.
//
// Fog is integrated along the ray to scene depth (or to far for sky pixels)
// for every pixel. Clouds are only relevant when the ray reaches the sky, so
// the march writes identity (0,0,0,1) into cloud_in for non-sky pixels — the
// cloud layer composite becomes a no-op there.
//
//   after_fog = scene.rgb * fog_trans + fog_scatter
//   result    = after_fog * cloud_trans + cloud_scatter

@group(0) @binding(0) var scene_hdr: texture_2d<f32>;
@group(0) @binding(1) var cloud_in: texture_2d<f32>;
@group(0) @binding(2) var composited: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var fog_in: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(composited);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);
    let scene = textureLoad(scene_hdr, coord, 0);

    // Nearest-neighbor sample from the half-res buffers.
    let half_coord = coord / 2;
    let fog = textureLoad(fog_in, half_coord, 0);
    let cloud = textureLoad(cloud_in, half_coord, 0);

    let after_fog = scene.rgb * fog.a + fog.rgb;
    let result = after_fog * cloud.a + cloud.rgb;
    textureStore(composited, coord, vec4<f32>(result, 1.0));
}
