// user_shader_instance_composite.wgsl — Stage 6b composite pass.
//
// Reads:
//   * `output_hits[]` — per-pixel `InstanceMarchHit` from
//     `instance_march_main` (Stage 5b/6a). 48 B per pixel.
//   * Host G-buffer textures — position, normal, material, leaf_slot
//     written by `octree_march.wgsl::main`.
//
// Writes:
//   * Merged G-buffer textures — same four formats. For each pixel,
//     one of two paths runs:
//       - Instance wins (`hit==1u && t_world < host.position.w`):
//         instance data with world position re-derived from the camera
//         ray + `t_world`.
//       - Otherwise: host data passes through unmodified.
//
// Why a separate output set instead of in-place RMW:
//   WebGPU doesn't support a writable storage view of a texture
//   concurrently bound as a sampled view in the same dispatch. The
//   two-set design keeps the read/write resources disjoint, so binding
//   validation passes cleanly and there are no intra-dispatch
//   synchronisation hazards. Stage 6c can wire it up so downstream
//   passes (shade) read from the merged set; the host G-buffer becomes
//   intermediate scratch.
//
// V1 instance pixels emit material_packed exactly the way octree_march
// packs `gbuf_material`'s R channel: `(pri & 0xFFFF) | (sec << 16)`.
// Blend remaps from 4 bits → 8 bits via the same `(b<<4)|b` formula
// `octree_march` uses. Per-voxel color override (the R channel's
// rgb565 in bits 16..31 of `gbuf_material`'s G channel) is left at 0
// for V1 — the shade pass falls back to the material's base colour.

struct InstanceMarchHit {
    hit: u32,
    region_index: u32,
    instance_index: u32,
    leaf_attr_slot: u32,
    t_world: f32,
    material_packed: u32,
    _pad0: u32,
    _pad1: u32,
    normal: vec3<f32>,
    _pad2: f32,
}

/// Subset of the Stage 6a `MarchUniforms` — only `screen_width` is read
/// here, but the full struct is bound so the same uniform buffer can be
/// shared across march + composite without re-uploading.
struct MarchUniforms {
    tile_index_count: u32,
    proto_lookup_count: u32,
    screen_width: u32,
    screen_height: u32,
    march_max_steps_outer: u32,
    march_max_steps_brick: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Same MarchCameraUniform layout as the march pass — same 80-byte
/// prefix-binary-compatibility with `rkp_scene::CameraUniforms`. The
/// Stage 6c renderer integration binds the renderer's existing camera
/// buffer to both the march and the composite without translation.
struct MarchCameraUniform {
    position: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> output_hits: array<InstanceMarchHit>;
@group(0) @binding(1) var<uniform> march_uniforms: MarchUniforms;
@group(0) @binding(2) var<uniform> camera: MarchCameraUniform;

@group(1) @binding(0) var gbuf_host_position: texture_2d<f32>;
@group(1) @binding(1) var gbuf_host_normal: texture_2d<f32>;
@group(1) @binding(2) var gbuf_host_material: texture_2d<u32>;
@group(1) @binding(3) var gbuf_host_leaf_slot: texture_2d<u32>;

@group(2) @binding(0) var gbuf_out_position: texture_storage_2d<rgba32float, write>;
@group(2) @binding(1) var gbuf_out_normal: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var gbuf_out_material: texture_storage_2d<rg32uint, write>;
@group(2) @binding(3) var gbuf_out_leaf_slot: texture_storage_2d<r32uint, write>;

/// Construct the per-pixel ray. Mirrors `camera_pixel_ray` in
/// `user_shader_instance_march_main.wgsl` exactly so the world-position
/// the composite re-derives matches what `instance_march_main` traversed.
fn camera_pixel_ray_origin() -> vec3<f32> {
    return camera.position.xyz;
}

fn camera_pixel_ray_direction(pixel_xy: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_xy) + 0.5 + camera.jitter) / camera.resolution;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(
        camera.forward.xyz
        + ndc.x * camera.right.xyz
        + ndc.y * camera.up.xyz
    );
}

/// Translate the instance's leaf-attr `material_packed`
/// (`pri | sec << 16 | bw << 28`) into the (R, G) pair `octree_march`
/// writes into `gbuf_material`. Mirrors `octree_march.wgsl` lines
/// 1576-1588 exactly so downstream passes (PBR shade etc.) can't tell
/// instance vs host pixels from the material payload alone.
fn pack_instance_material(material_packed: u32) -> vec2<u32> {
    let packed_r = material_packed & 0x0FFFFFFFu;
    let blend_4 = (material_packed >> 28u) & 0x0Fu;
    let blend_8 = (blend_4 << 4u) | blend_4;
    // V1 leaves bits 16..31 of the G channel (rgb565 colour override)
    // at 0 — the shade pass falls back to the material's base colour.
    let packed_g = blend_8 & 0xFFu;
    return vec2<u32>(packed_r, packed_g);
}

@compute @workgroup_size(8, 8, 1)
fn instance_composite_main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let w = march_uniforms.screen_width;
    let h = march_uniforms.screen_height;
    if pixel.x >= w || pixel.y >= h { return; }

    let coord = vec2<i32>(pixel.xy);
    let host_pos = textureLoad(gbuf_host_position, coord, 0);
    let host_normal = textureLoad(gbuf_host_normal, coord, 0);
    let host_material = textureLoad(gbuf_host_material, coord, 0);
    let host_leaf_slot = textureLoad(gbuf_host_leaf_slot, coord, 0);

    let idx = pixel.x + pixel.y * w;
    let hit = output_hits[idx];
    let host_t = host_pos.w;
    let instance_wins = hit.hit == 1u && hit.t_world < host_t;

    if instance_wins {
        // Re-derive the world position using the same camera ray the
        // march traversed; the hit's `t_world` is along that ray.
        let ray_origin = camera_pixel_ray_origin();
        let ray_dir = camera_pixel_ray_direction(pixel.xy);
        let world_pos = ray_origin + ray_dir * hit.t_world;

        let mat = pack_instance_material(hit.material_packed);

        textureStore(gbuf_out_position, coord, vec4<f32>(world_pos, hit.t_world));
        // alpha = 1.0 to match the host's "solid hit" convention; the
        // shade pass uses W as a coverage gate.
        textureStore(gbuf_out_normal, coord, vec4<f32>(hit.normal, 1.0));
        textureStore(gbuf_out_material, coord, vec4<u32>(mat.x, mat.y, 0u, 0u));
        textureStore(gbuf_out_leaf_slot, coord, vec4<u32>(hit.leaf_attr_slot, 0u, 0u, 0u));
    } else {
        // Pass-through. Texture loads / stores work on the host's
        // formats since the merged G-buffer textures were created with
        // the same formats as the host G-buffer.
        textureStore(gbuf_out_position, coord, host_pos);
        textureStore(gbuf_out_normal, coord, host_normal);
        textureStore(gbuf_out_material, coord, host_material);
        textureStore(gbuf_out_leaf_slot, coord, host_leaf_slot);
    }
}
