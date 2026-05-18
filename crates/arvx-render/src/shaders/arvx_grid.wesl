// Infinite world-space grid overlay.
//
// Fullscreen-triangle render pass: reconstructs a camera ray per pixel,
// intersects with y = 0, writes anti-aliased grid lines as RGBA fragments
// blended over the composite. Two grid scales (1 m primary, 0.1 m
// secondary) with distance fade. X/Z origin axes painted distinctly.

import package::lib::math::SKY_DEPTH_SENTINEL;
import package::lib::types::CameraUniforms;

struct GridParams {
    /// Distance at which the grid fades fully out, in world units.
    fade_distance: f32,
    enabled: u32,
    _pad0: vec2<u32>,
    /// World-space origin for the grid. Plane sits at `y = plane_origin.y`;
    /// X/Z grid lines and origin axes are measured relative to
    /// `plane_origin.xz`. Default `(0, 0, 0)` = classic world-axis grid.
    /// The build viewport sets this to the previewed entity's position.
    plane_origin: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> params: GridParams;
@group(0) @binding(2) var gbuf_position: texture_2d<f32>;

const MAJOR_SPACING: f32 = 1.0;
const MINOR_SPACING: f32 = 0.1;
const MAJOR_COLOR: vec3<f32> = vec3<f32>(0.55, 0.55, 0.55);
const MINOR_COLOR: vec3<f32> = vec3<f32>(0.30, 0.30, 0.30);
const AXIS_X_COLOR: vec3<f32> = vec3<f32>(0.95, 0.32, 0.32);
const AXIS_Z_COLOR: vec3<f32> = vec3<f32>(0.32, 0.55, 0.95);

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle covering NDC [-1, 3] × [-3, 1] — three vertex
// indices generate it without any vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    let pos = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn grid_line(coord: f32, spacing: f32, pixel_size: f32, width_px: f32) -> f32 {
    let w = max(pixel_size * width_px, 1e-6);
    let d = abs(coord - round(coord / spacing) * spacing);
    return 1.0 - smoothstep(0.5 * w, 1.5 * w, d);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if params.enabled == 0u {
        return vec4<f32>(0.0);
    }

    let dims = textureDimensions(gbuf_position);
    let pix = vec2<i32>(in.uv * vec2<f32>(dims));
    let coord = clamp(pix, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));

    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray_dir = normalize(camera.forward.xyz
                          + ndc.x * camera.right.xyz
                          + ndc.y * camera.up.xyz);
    let ray_origin = camera.position.xyz;

    let denom = ray_dir.y;
    if abs(denom) < 1e-4 {
        return vec4<f32>(0.0);
    }
    let plane_y = params.plane_origin.y;
    let t = (plane_y - ray_origin.y) / denom;
    if t <= 0.0 {
        return vec4<f32>(0.0);
    }

    let hit = ray_origin + ray_dir * t;
    // Grid coords relative to `plane_origin` so the lines (and the
    // red/blue origin axes) always run under the origin, no matter
    // where the build viewport has placed it.
    let rel_x = hit.x - params.plane_origin.x;
    let rel_z = hit.z - params.plane_origin.z;

    // Z-test: only paint where the floor-plane hit is closer than the
    // gbuffer surface distance. `pos.w` is the surface t; >= 9999 = miss.
    let pos = textureLoad(gbuf_position, coord, 0);
    let scene_t = pos.w;
    if scene_t > 0.0 && scene_t < SKY_DEPTH_SENTINEL && t > scene_t {
        return vec4<f32>(0.0);
    }

    // Pixel footprint at the hit — keeps line widths visually constant.
    let view_dist = length(hit - ray_origin);
    let pixel_size = view_dist * 2.0 * length(camera.up.xyz) / camera.resolution.y;

    let major_x = grid_line(rel_x, MAJOR_SPACING, pixel_size, 1.5);
    let major_z = grid_line(rel_z, MAJOR_SPACING, pixel_size, 1.5);
    let minor_x = grid_line(rel_x, MINOR_SPACING, pixel_size, 1.0);
    let minor_z = grid_line(rel_z, MINOR_SPACING, pixel_size, 1.0);

    let major = max(major_x, major_z);
    let minor = max(minor_x, minor_z);
    let minor_only = minor * (1.0 - major);

    // Origin axes — wider, colored. `1e9` spacing collapses to "any
    // distance from 0", giving a single line at rel_z=0 (axis_x) and
    // rel_x=0 — relative to `plane_origin`.
    let axis_x = grid_line(rel_z, 1e9, pixel_size, 2.0);
    let axis_z = grid_line(rel_x, 1e9, pixel_size, 2.0);

    let fade = 1.0 - smoothstep(params.fade_distance * 0.6,
                                params.fade_distance, view_dist);
    if fade <= 0.0 {
        return vec4<f32>(0.0);
    }

    var grid_rgb = vec3<f32>(0.0);
    var grid_alpha = 0.0;

    if minor_only > 0.0 {
        grid_rgb = MINOR_COLOR;
        grid_alpha = minor_only * 0.6;
    }
    if major > 0.0 {
        grid_rgb = MAJOR_COLOR;
        grid_alpha = major * 0.85;
    }
    if axis_x > 0.0 {
        grid_rgb = AXIS_X_COLOR;
        grid_alpha = max(grid_alpha, axis_x);
    }
    if axis_z > 0.0 {
        grid_rgb = AXIS_Z_COLOR;
        grid_alpha = max(grid_alpha, axis_z);
    }

    grid_alpha *= fade;
    // Pre-multiply alpha so the standard SrcAlpha/OneMinusSrcAlpha blend
    // gives a clean lerp.
    return vec4<f32>(grid_rgb, grid_alpha);
}
