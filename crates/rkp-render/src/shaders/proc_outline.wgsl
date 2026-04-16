// Selected-primitive outline for the procedural raymarch preview.
//
// Draws a 1-pixel color band along the silhouette of the currently-
// selected primitive in the build viewport. Reads the per-pixel
// NodeId that `proc_raymarch.wgsl` packs into bits 16-31 of the
// material G-buffer's `.r` channel; writes an alpha-blended outline
// color to the composite texture.
//
// Runs after tone-map / grid so the outline is in LDR and unaffected
// by bloom / exposure. Fragment shader only — no per-vertex work,
// full-screen triangle via `vertex_index`. Intended to be cheap (one
// material fetch + 4 neighbor fetches + discard on miss).

struct OutlineParams {
    // `u32::MAX` disables the pass (shader discards everything). The
    // low 16 bits are compared against the packed NodeId — matches
    // the width the raymarch packs into the G-buffer.
    selected_node_id: u32,
    /// Outline color, premultiplied alpha. Shader emits this as-is;
    /// the pipeline's `src = One, dst = OneMinusSrcAlpha` blend does
    /// the compositing. Example: `(1,0.5,0, 1)` → opaque orange.
    color_rgba: vec4<f32>,
}

@group(0) @binding(0) var gbuf_material: texture_2d<u32>;
@group(0) @binding(1) var<uniform> params: OutlineParams;

const INVALID_NODE: u32 = 0xFFFFu;

fn node_at(coord: vec2<i32>) -> u32 {
    let dims = textureDimensions(gbuf_material);
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let packed = textureLoad(gbuf_material, clamped, 0).r;
    return (packed >> 16u) & 0xFFFFu;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Standard full-screen triangle trick: three vertices in NDC cover
    // the whole viewport with no vertex buffer.
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = 1.0 - f32(vi & 2u) * 2.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // Sentinel = no selection → nothing to outline.
    let sel = params.selected_node_id & 0xFFFFu;
    if (params.selected_node_id == 0xFFFFFFFFu) { discard; }

    let coord = vec2<i32>(frag.xy);
    let self_id = node_at(coord);

    // Outline rule: a pixel is an outline pixel if
    //   - it is NOT part of the selected primitive, AND
    //   - at least one of its 4 axis-aligned neighbors IS.
    // This gives a 1-pixel-thick band drawn on the OUTSIDE of the
    // selected shape. Running on the outside keeps the selected
    // geometry's color unchanged underneath, so highlights and
    // material look are preserved.
    //
    // Misses (pixel's node_id == INVALID_NODE = 0xFFFF) count as
    // "not selected" — if the selected primitive's silhouette ends
    // at empty space we still want the outline there.
    if (self_id == sel && self_id != INVALID_NODE) { discard; }

    let n_px = node_at(coord + vec2<i32>( 1,  0));
    let n_nx = node_at(coord + vec2<i32>(-1,  0));
    let n_py = node_at(coord + vec2<i32>( 0,  1));
    let n_ny = node_at(coord + vec2<i32>( 0, -1));

    let hit = (n_px == sel) || (n_nx == sel) || (n_py == sel) || (n_ny == sel);
    if (!hit) { discard; }

    // Premultiply color by alpha so the blend-state math
    // (src=One, dst=OneMinusSrcAlpha) gives a clean over.
    let a = params.color_rgba.a;
    return vec4<f32>(params.color_rgba.rgb * a, a);
}
