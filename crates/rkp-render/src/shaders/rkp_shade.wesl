// RKIPatch deferred PBR shading compute shader.
//
// Reads G-buffer (position, normal+shadow, material) + SSAO texture.
// Evaluates PBR Cook-Torrance BRDF with direct lighting.
// Writes final HDR color to output texture.

import package::lib::oct_normal::unpack_oct_normal;
import package::lib::atmosphere::EARTH_RADIUS;
import package::lib::atmosphere::ATMO_RADIUS;
import package::lib::atmosphere::RAYLEIGH_SCALE_H;
import package::lib::atmosphere::MIE_SCALE_H;
import package::lib::atmosphere::BETA_R;
import package::lib::atmosphere::BETA_M_SCAT;
import package::lib::atmosphere::sample_extinction;
import package::lib::atmosphere::transmittance_params_to_uv;
import package::lib::atmosphere::rayleigh_phase;
import package::lib::atmosphere::cornette_shanks;
import package::lib::atmosphere::lookup_transmittance;
import package::lib::atmosphere::lookup_multiscatter;
import package::lib::math::PI;
import package::lib::math::SKY_DEPTH_SENTINEL;
import package::lib::math::value_noise_3d;
import package::lib::math::ray_sphere;
import package::lib::pbr::dielectric_f0;
import package::lib::pbr::fresnel_dielectric;
import package::lib::pbr::fresnel_schlick;
import package::lib::pbr::beer_absorption;
import package::lib::types::GpuMaterial;
import package::lib::types::mat_albedo;
import package::lib::types::CameraUniforms;
import package::lib::types::GpuLight;
import package::lib::types::ShadeParams;


// --- Structs ---



// 8 user-named scalar params per material, packed as two vec4s (32
// bytes — matches `Vec<[f32; 8]>` produced by the engine's
// `MaterialLibrary::build_shader_params`). Indexed by material_id
// parallel to the `materials` array.
struct ShaderParamsSlot {
    p0: vec4<f32>,
    p1: vec4<f32>,
}

// Inputs to the user `shade` hook. The dispatcher fills this from
// G-buffer + resolved material data so the user fn doesn't have to
// re-do material lookup. Layout is stable; new fields go at the end.
struct ShadeCtx {
    world_pos: vec3<f32>,
    distance: f32,
    normal: vec3<f32>,
    n_dot_v: f32,
    view_dir: vec3<f32>,
    time: f32,
    base_color: vec3<f32>,
    metallic: f32,
    emission_color: vec3<f32>,
    roughness: f32,
    // 8 named scalar parameters from this material's
    // `ShaderParamsSlot`. Index 0..3 = params[0].xyzw, 4..7 = params[1].xyzw.
    params: array<vec4<f32>, 2>,
    primary_material_id: u32,
    secondary_material_id: u32,
    blend_weight: f32,
    // Reserved for future per-pixel data; pads the struct to a stable
    // 16-byte-aligned size and reserves the slot so adding a new
    // input later doesn't shuffle existing field offsets.
    _reserved0: f32,
}

// Output of the user `shade` hook is a luminance value in nits (cd/m²).
// The deferred shade pass writes physically-scaled values into the HDR
// target — auto-exposure adapts so the brightest typical pixel maps
// near 1.0 after tone-map. Reference points:
//
//   * sun-lit white surface (PBR direct lighting): ~90 000 nits
//   * bright office light:                          ~3 000 nits
//   * laptop / phone display at full brightness:   ~500–1 000 nits
//   * dim emissive UI element:                      ~100 nits
//
// A hologram emitting 5 000 nits looks vivid in a dim room and faint
// in sunlight — same as a real holographic display would. Pick the
// value that matches the look you want, not "1.0 = display white",
// since auto-exposure will rescale.

// Output of the user `shade` hook. V1 is replace-only: the dispatcher
// uses `rgb` as the final pixel color and skips the rest of the
// deferred shade pass. Augment-mode (modify ctx fields, fall through
// to PBR) is reserved for a future phase.
struct ShadeResult {
    rgb: vec3<f32>,
}

fn shade_result_passthrough(ctx: ShadeCtx) -> ShadeResult {
    // The identity arm — never reached when `shader_id == 0` (the
    // dispatcher branch is gated on it). Returning the resolved base
    // color keeps the output semantically valid if a user shader
    // somehow forgets a `case` arm and the default fires.
    return ShadeResult(ctx.base_color);
}

// Neutral gray for isolation-mode sky + ambient — chosen to match a
// typical 18% middle-gray studio backdrop in linear light.
const ISOLATION_BG: vec3<f32> = vec3<f32>(0.18, 0.18, 0.18);
const ISOLATION_AMBIENT: vec3<f32> = vec3<f32>(0.4, 0.4, 0.4);

// vec3 fields on the Rust side are [f32; 3] (12 bytes, tightly packed).
// WGSL's vec3<f32> has 16-byte alignment which would balloon the struct
// to 128 bytes — so on the shader side every 3-vector is spelled out as
// three f32 components to keep the WGSL and Rust layouts in lockstep at
// 96 bytes.

fn mat_emission(m: GpuMaterial) -> vec3<f32> {
    return vec3<f32>(m.emission_r, m.emission_g, m.emission_b);
}
fn mat_subsurface_color(m: GpuMaterial) -> vec3<f32> {
    return vec3<f32>(m.subsurface_r, m.subsurface_g, m.subsurface_b);
}

// --- Bindings ---

// Group 0: G-buffer (read)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;
// R = oct-packed normal of the glass surface, G = (thickness_mm <<
// 16) | material_id. `thickness_mm == 0` means "no glass at this
// pixel" — use as the gate for the glass composite path.
@group(0) @binding(3) var gbuf_glass: texture_2d<u32>;
// Primary hit's scene-global leaf_attr_slot. `0` = sky / no-hit /
// procedural (no stable slot). Used by the paint cursor to look up
// per-voxel geodesic distance in `brush_overlay_distances`.
@group(0) @binding(4) var gbuf_leaf_slot: texture_2d<u32>;

// Group 1: shadow texture (read, half-res) + SSAO texture (read, half-res)
// shadow_tex is written by rkp_shadow_trace at half-res; we upsample it
// per-pixel with a bilateral gather guided by position + normal deltas.
@group(1) @binding(0) var shadow_tex: texture_2d<f32>;
@group(1) @binding(1) var ssao_tex: texture_2d<f32>;

// CSM (Cascaded Shadow Maps) — directional shadow uniform + buffer.
// Wire-format mirror of `rkp_render::shadow_map_pass::LightCameraCsm`
// (672 B). Each cascade is a 160 B `LightCameraShade` slice; the
// shade pass picks the slice per-pixel based on view-space depth and
// indexes into `shadow_buffer` at `cascade * SIZE² + ty * SIZE + tx`.
//
// In march-mode `cascade_count = 1` and only `cascades[0]` is filled,
// so cascade selection collapses to "always slice 0" and the math
// reduces to today's single-cascade flow.
struct LightCameraShade {
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    light_dir: vec3<f32>,
    depth_bias: f32,
    inv_shadow_map_size: vec2<f32>,
    shadow_map_size: vec2<u32>,
}
struct LightCameraShadeCsm {
    cascades: array<LightCameraShade, 4>,
    cascade_far_view_z: vec4<f32>,
    cascade_count: u32,
    _csm_pad0: u32,
    _csm_pad1: u32,
    _csm_pad2: u32,
}
@group(1) @binding(2) var<uniform> light_camera: LightCameraShadeCsm;
// Phase 8 V2 — shadow_buffer is `array<u32>` holding bit-cast
// f32 depths, written by the mesh blit (mesh-mode) or scatter pass
// (march-mode). For CSM the buffer is `CSM_CASCADE_COUNT * SIZE²`
// u32s laid out cascade-major: `[cascade * SIZE² + ty * SIZE + tx]`.
@group(1) @binding(3) var<storage, read> shadow_buffer: array<u32>;
// Glass-shadow per-cascade depth maps from the light POV — captured
// by `mesh_glass_shadow.wesl`'s front/back pipelines. Cleared to 1.0
// on the front (closest entry wins) and 0.0 on the back (farthest
// exit wins). The shade pass samples both at the fragment's
// light-space coords and applies Beer attenuation when glass sits
// between the light and the fragment. Mesh-mode only — march mode
// leaves these at clear values, which `glass_shadow_attenuation`
// reads as "no glass" (front >= back) and skips.
@group(1) @binding(4) var glass_shadow_front: texture_depth_2d_array;
@group(1) @binding(5) var glass_shadow_back:  texture_depth_2d_array;

// Group 2: output HDR color (write, full-res)
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Group 3: shading params + lights + materials + user-shader params
@group(3) @binding(0) var<uniform> shade_params: ShadeParams;
@group(3) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(3) @binding(2) var<storage, read> materials: array<GpuMaterial>;
@group(3) @binding(3) var<storage, read> shader_params: array<ShaderParamsSlot>;

// Group 4: camera
@group(4) @binding(0) var<uniform> camera: CameraUniforms;

// Group 5: atmosphere LUTs
@group(5) @binding(0) var transmittance_lut: texture_2d<f32>;
@group(5) @binding(1) var multiscatter_lut: texture_2d<f32>;
@group(5) @binding(2) var atmo_sampler: sampler;
@group(5) @binding(3) var sky_view_lut: texture_2d<f32>;
@group(5) @binding(4) var aerial_perspective_lut: texture_3d<f32>;

// Paint cursor's per-leaf geodesic distance. Parallel to
// `leaf_attr_pool`; sentinel (INFINITY) means "not under the brush".
// Indexed by the slot written to `gbuf_leaf_slot`; 0 is the no-hit
// sentinel so the lookup must guard `slot != 0u`.
@group(6) @binding(0) var<storage, read> brush_overlay_distances: array<f32>;

// ── CSM directional shadow-map sample ──────────────────────────
//
// Project `world_pos` through cascade `i`'s light camera, look up
// the matching slot in `shadow_buffer`, depth-compare against the
// surface's projected NDC z. Returns:
//   shadow ∈ [0, 1]  — 1.0 lit, 0.0 shadowed
//   in_bounds        — false if the projection landed outside
//                      cascade `i`'s clip volume (caller falls
//                      through to cascade `i+1`).
//
// Storage layout: `shadow_buffer` holds CSM_CASCADE_COUNT slices
// laid out cascade-major. Index = `cascade * (W*H) + ty*W + tx`.
//
// `depth_bias` (per-cascade, set CPU-side by `compute_csm_cascades`)
// pushes the surface toward the light pre-compare to eat up
// slope-scale + acne. Positive reduces self-shadowing at the cost of
// peter-panning.
struct CascadeSample {
    shadow: f32,
    in_bounds: bool,
}

// Poisson disc taps in unit-disc space — hand-tuned 16-sample
// distribution that doubles as the source for the 4 / 9 sub-counts
// (just take the prefix). The first tap is `(0,0)` so a 1-sample
// fast-path matches the hard-shadow behavior of the pre-PCF code
// exactly.
const PCF_DISC: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>( 0.000000,  0.000000),
    vec2<f32>(-0.942016, -0.399062),
    vec2<f32>( 0.945586, -0.768907),
    vec2<f32>(-0.094184, -0.929389),
    vec2<f32>( 0.344959,  0.293878),
    vec2<f32>(-0.915886,  0.457714),
    vec2<f32>(-0.815442, -0.879125),
    vec2<f32>(-0.382775,  0.276768),
    vec2<f32>( 0.974844,  0.756484),
    vec2<f32>( 0.443233, -0.975116),
    vec2<f32>( 0.537430, -0.473734),
    vec2<f32>(-0.264969, -0.418930),
    vec2<f32>( 0.791975,  0.190902),
    vec2<f32>(-0.241888,  0.997065),
    vec2<f32>(-0.814100,  0.914376),
    vec2<f32>( 0.199841,  0.786414),
);

// Radius (in shadow-map texels) the PCF disc is scaled to. 1.5 keeps
// the penumbra narrow enough that detail isn't washed out at low tap
// counts but wide enough to mask single-texel staircasing. The disc
// is in [-1,1] so a tap at `(±1, ±1)` lands ~`PCF_RADIUS_TEXELS`
// texels from the projected center.
const PCF_RADIUS_TEXELS: f32 = 1.5;

fn sample_one_cascade(world_pos: vec3<f32>, i: u32) -> CascadeSample {
    let cam = light_camera.cascades[i];
    let clip = cam.view_proj * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 { return CascadeSample(1.0, false); }
    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0
        || ndc.y < -1.0 || ndc.y > 1.0
        || ndc.z < 0.0 || ndc.z > 1.0 {
        return CascadeSample(1.0, false);
    }
    let size_u = cam.shadow_map_size;
    let size_f = vec2<f32>(size_u);
    let tx_f = (ndc.x * 0.5 + 0.5) * size_f.x;
    let ty_f = (1.0 - (ndc.y * 0.5 + 0.5)) * size_f.y;
    let surface_z = ndc.z - cam.depth_bias;
    let slice = size_u.x * size_u.y;
    let base_idx = i * slice;
    // PCF tap count comes from the Quality preset via ShadeParams.
    // Clamp 1..16; 1 keeps the hard-shadow look (single tap at the
    // projected texel), higher counts soften the edge by averaging
    // a Poisson disc neighborhood.
    let taps = clamp(shade_params.pcf_taps, 1u, 16u);
    var lit: f32 = 0.0;
    for (var k: u32 = 0u; k < taps; k = k + 1u) {
        let offs = PCF_DISC[k] * PCF_RADIUS_TEXELS;
        let stx = clamp(u32(tx_f + offs.x), 0u, size_u.x - 1u);
        let sty = clamp(u32(ty_f + offs.y), 0u, size_u.y - 1u);
        let map_z = bitcast<f32>(shadow_buffer[base_idx + sty * size_u.x + stx]);
        lit = lit + select(0.0, 1.0, surface_z <= map_z);
    }
    return CascadeSample(lit / f32(taps), true);
}

// View-space depth of `world_pos` along the camera forward axis.
// `camera.forward.xyz` is unit (per `build_camera_uniforms`), so
// dot-product with the offset is exactly the camera-space z (positive
// into the scene).
fn view_z_of(world_pos: vec3<f32>) -> f32 {
    return dot(world_pos - camera.position.xyz, camera.forward.xyz);
}

// Cascade-aware shadow lookup. Picks the smallest cascade `i` such
// that `view_z <= cascade_far_view_z[i]`, samples it, and falls
// through to `i+1` if the projection landed outside its clip volume
// (covers fragments near a slice boundary that get nudged out by
// the texel snap). If the last cascade is also out-of-bounds, return
// 1.0 (fully lit) — same edge behavior as today's single-cascade
// path.
//
// In march-mode `cascade_count == 1` and `cascade_far_view_z[0]` is
// `+INF`, so this collapses to a single sample of slice 0 (today's
// behavior bit-for-bit).
fn sample_shadow_map(world_pos: vec3<f32>) -> f32 {
    let view_z = view_z_of(world_pos);
    let n = light_camera.cascade_count;

    var i: u32 = 0u;
    loop {
        if i + 1u >= n { break; }
        if view_z <= light_camera.cascade_far_view_z[i] { break; }
        i = i + 1u;
    }

    let r0 = sample_one_cascade(world_pos, i);
    if r0.in_bounds { return r0.shadow; }
    if i + 1u < n {
        let r1 = sample_one_cascade(world_pos, i + 1u);
        if r1.in_bounds { return r1.shadow; }
    }
    return 1.0;
}

// Per-meter optical density used for glass shadow attenuation —
// matches the march's `GLASS_SHADOW_DENSITY * sigma_albedo` averaged
// over a neutral glass material (albedo ~0.7, opacity ~0.5). Tunable
// constant; raise for darker glass shadows, lower for almost-clear.
const GLASS_SHADOW_SIGMA: f32 = 1.5;

// Beer-attenuation factor for glass between the directional light
// and `world_pos`. Returns 1.0 when no glass sits in the path
// (`back_z <= front_z` at the projected light-space coords, or the
// fragment is in front of the back-face — i.e., the glass is BEHIND
// the fragment relative to the light). Otherwise multiplies into the
// opaque CSM shadow factor.
//
// Picks the cascade the same way `sample_shadow_map` does (smallest
// `i` whose far Z covers `view_z`); falls through to the next on a
// frustum miss.
fn glass_shadow_attenuation(world_pos: vec3<f32>) -> f32 {
    let view_z = view_z_of(world_pos);
    let n = light_camera.cascade_count;

    var i: u32 = 0u;
    loop {
        if i + 1u >= n { break; }
        if view_z <= light_camera.cascade_far_view_z[i] { break; }
        i = i + 1u;
    }

    var atten = 1.0;
    for (var step: u32 = 0u; step < 2u; step = step + 1u) {
        let c = i + step;
        if c >= n { break; }
        let cam = light_camera.cascades[c];
        let clip = cam.view_proj * vec4<f32>(world_pos, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc = clip.xyz / clip.w;
        if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
            continue;
        }
        // Light-space tex coords. y flipped to match wgpu's
        // [0,0]-top-left convention.
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
        let map_size = vec2<f32>(cam.shadow_map_size);
        let icoord = vec2<i32>(clamp(uv * map_size, vec2<f32>(0.0), map_size - vec2<f32>(1.0)));
        let front_z = textureLoad(glass_shadow_front, icoord, c, 0);
        let back_z  = textureLoad(glass_shadow_back,  icoord, c, 0);
        // No glass at this texel — front cleared to 1.0, back to 0.0,
        // so back <= front means uncovered.
        if back_z <= front_z { return atten; }
        // If the fragment is closer to the light than the glass back
        // face, the glass sits BEHIND the fragment along the light
        // ray — it can't attenuate light reaching this fragment.
        if ndc.z <= back_z { return atten; }
        // Back-project both depths to world space. For wgpu's NDC
        // convention, the front/back textures store the same NDC z
        // values that `view_proj` produces — so the inverse maps
        // directly.
        let p_front_h = cam.view_proj_inv * vec4<f32>(ndc.x, ndc.y, front_z, 1.0);
        let p_back_h  = cam.view_proj_inv * vec4<f32>(ndc.x, ndc.y, back_z,  1.0);
        let p_front = p_front_h.xyz / p_front_h.w;
        let p_back  = p_back_h.xyz  / p_back_h.w;
        let thickness = length(p_back - p_front);
        atten = exp(-GLASS_SHADOW_SIGMA * thickness);
        return atten;
    }
    return atten;
}

// Aerial-perspective LUT slice parameterization — must match rkp_aerial_perspective_lut.wgsl.
// Slice z stores the camera-to-d(z) atmospheric integral under an exponential
// map d(z) = AP_NEAR * (AP_FAR/AP_NEAR)^((z+0.5)/N); the shade pass inverts this.
const AP_NEAR_DISTANCE: f32 = 1.0;
const AP_FAR_DISTANCE: f32 = 128000.0;

// --- Atmospheric scattering ---
// Constants + pure-math helpers come from `lib::atmosphere` (imported
// at the top of this file). Texture-sampling lookups stay local
// because they reference this shader's own bind-group bindings.

const MIE_G: f32 = 0.8;
const SUN_ANGULAR_RADIUS: f32 = 0.004675;

// `ray_sphere` is imported from `lib::math` at the top.

fn sample_scattering(altitude: f32) -> vec2<f32> {
    let density_r = exp(-altitude / RAYLEIGH_SCALE_H);
    let density_m = exp(-altitude / MIE_SCALE_H);
    return vec2<f32>(density_r, density_m);
}

/// Look up transmittance from the precomputed LUT.

/// Look up multi-scattered luminance from the precomputed LUT.

/// Map a view ray direction to Sky View LUT UV coordinates.
/// Inverse of the parameterization in rkp_sky_view_lut.wgsl.
fn ray_dir_to_sky_view_uv(ray_dir: vec3<f32>, sun_dir: vec3<f32>, cam_height: f32) -> vec2<f32> {
    let view_zenith_cos = ray_dir.y;

    let horizon_cos = -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (cam_height * cam_height), 0.0));
    let horizon_angle = acos(horizon_cos);

    let view_angle = acos(clamp(view_zenith_cos, -1.0, 1.0));

    // V: non-linear mapping with horizon at v=0.5.
    var v: f32;
    if view_angle <= horizon_angle {
        // Above horizon: angle goes from horizon_angle (v=0.5) to 0 (v=1.0).
        // coord = 1 - angle/horizon_angle. Invert: coord = (2v-1)² → v = 0.5 + 0.5*sqrt(coord)
        let coord = 1.0 - view_angle / max(horizon_angle, 1e-6);
        v = 0.5 + 0.5 * sqrt(max(coord, 0.0));
    } else {
        // Below horizon: angle goes from horizon_angle (v=0.5) to PI (v=0.0).
        // coord = (angle - horizon) / (PI - horizon). Invert: coord = (1-2v)² → v = 0.5 - 0.5*sqrt(coord)
        let beta = PI - horizon_angle;
        let coord = (view_angle - horizon_angle) / max(beta, 1e-6);
        v = 0.5 - 0.5 * sqrt(max(coord, 0.0));
    }

    // U: view-sun azimuth via horizontal projection.
    let sun_horiz_len = length(vec2<f32>(sun_dir.x, sun_dir.z));
    let view_horiz_len = length(vec2<f32>(ray_dir.x, ray_dir.z));
    var light_view_cos = 0.0;
    if sun_horiz_len > 0.001 && view_horiz_len > 0.001 {
        let sun_h = vec2<f32>(sun_dir.x, sun_dir.z) / sun_horiz_len;
        let view_h = vec2<f32>(ray_dir.x, ray_dir.z) / view_horiz_len;
        light_view_cos = dot(sun_h, view_h);
    }
    let u = sqrt(max((light_view_cos + 1.0) * 0.5, 0.0));

    return clamp(vec2<f32>(u, v), vec2<f32>(0.001), vec2<f32>(0.999));
}

/// Atmosphere sky radiance using precomputed LUTs.
fn atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let origin = vec3<f32>(0.0, EARTH_RADIUS + camera_alt, 0.0);
    let atmo_hit = ray_sphere(origin, ray_dir, ATMO_RADIUS);
    if atmo_hit.y < 0.0 { return vec3<f32>(0.0); }

    let t_start = max(atmo_hit.x, 0.0);
    var t_end = atmo_hit.y;

    // Clip to earth surface.
    let earth_hit = ray_sphere(origin, ray_dir, EARTH_RADIUS);
    if earth_hit.x > 0.0 { t_end = min(t_end, earth_hit.x); }

    let cos_sun = dot(ray_dir, sun_dir);
    let phase_r = rayleigh_phase(cos_sun);
    let phase_m = cornette_shanks(cos_sun, MIE_G);

    let steps = 32u;
    let step_len = (t_end - t_start) / f32(steps);
    var throughput = vec3<f32>(1.0);
    var scatter = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = t_start + (f32(i) + 0.5) * step_len;
        let pos = origin + ray_dir * t;
        let altitude = length(pos) - EARTH_RADIUS;
        if altitude < 0.0 { break; }

        let extinction = sample_extinction(altitude);
        let densities = sample_scattering(altitude);
        let density_r = densities.x;
        let density_m = densities.y;
        let scattering = density_r * BETA_R + density_m * BETA_M_SCAT;

        let sample_transmittance = exp(-extinction * step_len);

        // Sun transmittance from LUT (replaces per-pixel secondary march).
        let up = pos / length(pos);
        let sun_cos = dot(up, sun_dir);
        let sun_trans = lookup_transmittance(transmittance_lut, atmo_sampler, length(pos), sun_cos);

        // Earth shadow check.
        let earth_shadow = select(0.0, 1.0, sun_cos > -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (length(pos) * length(pos)), 0.0)));

        // Single-scattering: phase-weighted.
        let ss = (density_r * BETA_R * phase_r + density_m * BETA_M_SCAT * phase_m)
               * earth_shadow * sun_trans * sun_intensity;

        // Multi-scattering from LUT (replaces inline fms hack).
        let ms = lookup_multiscatter(multiscatter_lut, atmo_sampler, length(pos), sun_cos) * scattering * sun_intensity;

        // Integrate analytically within step (more accurate than Euler).
        let s_total = ss + ms;
        let s_int = (s_total - s_total * sample_transmittance) / max(extinction, vec3<f32>(1e-10));
        scatter += throughput * s_int;

        throughput *= sample_transmittance;
        if all(throughput < vec3<f32>(0.001)) { break; }
    }

    return scatter;
}

/// Sun disc with atmospheric extinction from LUT + aureole glow.
fn sun_disc(ray_dir: vec3<f32>, sun_dir: vec3<f32>, sun_intensity: f32, camera_alt: f32) -> vec3<f32> {
    let cos_angle = dot(ray_dir, sun_dir);
    let glow_radius = SUN_ANGULAR_RADIUS * 10.0;
    if cos_angle < cos(glow_radius) { return vec3<f32>(0.0); }

    // Clip sun disc/glow below the horizon — the view ray must be above the
    // geometric horizon for the sun to be visible at that pixel.
    let view_height = EARTH_RADIUS + camera_alt;
    let horizon_cos = -sqrt(max(1.0 - (EARTH_RADIUS * EARTH_RADIUS) / (view_height * view_height), 0.0));
    if ray_dir.y < horizon_cos { return vec3<f32>(0.0); }
    let sun_cos = dot(vec3<f32>(0.0, 1.0, 0.0), sun_dir);
    let sun_transmittance = lookup_transmittance(transmittance_lut, atmo_sampler, view_height, sun_cos);

    // Sun disc luminance = illuminance / solid_angle (Filament reference).
    let sun_solid_angle = 2.0 * PI * (1.0 - cos(SUN_ANGULAR_RADIUS));
    let sun_luminance = sun_intensity / sun_solid_angle;

    let sun_cos_r = cos(SUN_ANGULAR_RADIUS);
    let glow_cos = cos(glow_radius);

    var result = vec3<f32>(0.0);
    if cos_angle > sun_cos_r {
        // Hard disc — clips to white after tone mapping.
        let center_dist = (cos_angle - sun_cos_r) / (1.0 - sun_cos_r);
        let limb = 1.0 - 0.3 * (1.0 - center_dist);
        result = sun_transmittance * sun_luminance * limb;
    }

    // Aureole glow — bright near the disc, fading outward.
    // Uses sun_luminance scaled down so the inner glow is bright enough
    // to show transmittance color (orange at sunset) after tone mapping.
    let t = (cos_angle - glow_cos) / (sun_cos_r - glow_cos);
    if t > 0.0 {
        let glow_luminance = sun_luminance * 0.002; // ~0.2% of disc brightness
        result += sun_transmittance * glow_luminance * pow(t, 2.0);
    }

    return result;
}

/// Sample the aerial-perspective LUT for a shaded geometry pixel.
/// Returns (inscatter_rgb, transmittance). Apply as:
///   final = inscatter + surface_color * transmittance.
fn sample_aerial_perspective(screen_uv: vec2<f32>, surface_distance: f32) -> vec4<f32> {
    // Invert the exponential slice map to recover the texture-space Z that
    // encodes this distance, then let trilinear filtering interpolate between
    // the two bracketing slices in the LUT.
    let log_ratio = log(AP_FAR_DISTANCE / AP_NEAR_DISTANCE);
    let d = max(surface_distance, AP_NEAR_DISTANCE);
    let slice_u = clamp(log(d / AP_NEAR_DISTANCE) / log_ratio, 0.0, 1.0);
    return textureSampleLevel(
        aerial_perspective_lut, atmo_sampler,
        vec3<f32>(screen_uv, slice_u), 0.0,
    );
}

// --- PBR helpers ---

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// --- Glass / Transparency ---
//
// Dielectric Fresnel driven by IOR, and Beer-Lambert absorption for
// glass tinting. The full rkifield-style glass pipeline (continue
// the ray past the glass to find scene objects behind, composite
// with Beer over the behind color, handle multi-layer panes) needs
// a secondary march pass that rkp_shade can't run without scene
// bindings, and is deferred. For now, "transmission" samples the
// sky view LUT directly — correct when the camera sees sky through
// the glass, not-quite-correct when there are opaque objects on
// the far side (they'll read as sky instead of themselves).

// `unpack_oct_normal` is imported from `lib::oct_normal` at the top.
// Wave A2: this site previously used a Variant-B copy without the
// zero-length guard. The lib version (Variant A) returns +Y for
// degenerate inputs instead of NaN.

// dielectric_f0, fresnel_dielectric, beer_absorption are imported
// from `lib::pbr` at the top of this file (Wave AF unification with
// rkp_glass).

// --- Noise (procedural material variation) ---

const NOISE_CHANNEL_ALBEDO: u32    = 1u;
const NOISE_CHANNEL_ROUGHNESS: u32 = 2u;
const NOISE_CHANNEL_NORMAL: u32    = 4u;

// Forward-difference gradient of the noise field. Reuses the caller's
// already-computed center sample `n0` so this path costs 3 extra
// value_noise calls (was 6 under central differences). Slightly more
// directional bias at the lattice scale; invisible for bump use.
fn noise_gradient_3d(p: vec3<f32>, n0: f32) -> vec3<f32> {
    let e = 0.1;
    let inv_e = 1.0 / e;
    let nx = value_noise_3d(p + vec3<f32>(e, 0.0, 0.0)) - n0;
    let ny = value_noise_3d(p + vec3<f32>(0.0, e, 0.0)) - n0;
    let nz = value_noise_3d(p + vec3<f32>(0.0, 0.0, e)) - n0;
    return vec3<f32>(nx, ny, nz) * inv_e;
}

// fresnel_schlick is imported from `lib::pbr` at the top of this file.

// --- User-shader dispatch (composer-injectable) ---
//
// `rkp_render::shader_composer::compose()` produces a chunk that
// replaces this whole block via the begin/end markers. The chunk
// defines `dispatch_user_shade(shader_id, ctx)` as a switch over
// every registered shader's `user_<name>_shade` body. Phase B keeps
// the in-tree identity stub here so the file compiles standalone.

const USER_SHADE_DISPATCH_BEGIN: u32 = 0u;
fn dispatch_user_shade(shader_id: u32, ctx: ShadeCtx) -> ShadeResult {
    return shade_result_passthrough(ctx);
}
const USER_SHADE_DISPATCH_END: u32 = 0u;

// --- Main ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(gbuf_position);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let pos_data = textureLoad(gbuf_position, coord, 0);
    let world_pos = pos_data.xyz;
    let hit_t = pos_data.w;

    // No geometry → sky.
    if hit_t >= SKY_DEPTH_SENTINEL || hit_t <= 0.0 {
        if shade_params.isolation != 0u {
            // Isolation: flat neutral background, no sun disc, no sky LUT.
            // The grid pass paints over this for floor pixels.
            textureStore(output, coord, vec4<f32>(ISOLATION_BG, 1.0));
            return;
        }
        let uv_screen = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
        let ndc = vec2<f32>(uv_screen.x * 2.0 - 1.0, 1.0 - uv_screen.y * 2.0);
        let ray_dir = normalize(camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz);
        let s_dir = normalize(shade_params.sun_dir);

        // Look up precomputed sky radiance from Sky View LUT.
        // Glass compositing over sky is now handled by rkp_glass —
        // this path just writes the raw sky; rkp_glass reads it back
        // and does Fresnel / Beer / screen-space refraction on top.
        let sky_uv = ray_dir_to_sky_view_uv(ray_dir, s_dir, EARTH_RADIUS + shade_params.camera_altitude);
        var sky = textureSampleLevel(sky_view_lut, atmo_sampler, sky_uv, 0.0).rgb;
        sky += sun_disc(ray_dir, s_dir, shade_params.sun_intensity, shade_params.camera_altitude);

        textureStore(output, coord, vec4<f32>(sky, 1.0));
        return;
    }

    let normal = normalize(textureLoad(gbuf_normal, coord, 0).xyz);
    let mat_data = textureLoad(gbuf_material, coord, 0);
    let packed_r = mat_data.r;
    let packed_g = mat_data.g;

    // Unpack material IDs + blend.
    let primary_mat_id = packed_r & 0xFFFFu;
    let secondary_mat_id = (packed_r >> 16u) & 0xFFFFu;
    let blend_weight = f32(packed_g & 0xFFu) / 255.0;

    // Resolve material. Metallic/roughness/emission come from the
    // palette; base_color is used for the albedo unless a per-voxel
    // color override (RGB565 in packed_g high 16) is present.
    //
    // Dual-material path: octree_march (voxel) and proc_raymarch
    // (procedural, eventually) can both write a secondary material +
    // non-zero blend_weight for smooth seams across MaterialByHeight
    // / MaterialByNoise transition zones. We lerp every PBR property
    // in linear space. The secondary palette lookup is guarded behind
    // `blend_weight > 0` — some paths currently reuse the secondary
    // slot for pick IDs (see proc_raymarch) and those pixels always
    // have blend=0, so the guard prevents an out-of-bounds material
    // read on stale bits. Once pick IDs move to a dedicated texture
    // the guard is still desirable (free early-out on single-material
    // pixels, which is the common case).
    let mat_a = materials[primary_mat_id];
    var base_color = mat_albedo(mat_a);
    var metallic_raw = mat_a.metallic;
    var roughness_raw = mat_a.roughness;
    var emission_color = mat_emission(mat_a);
    var emission_strength = mat_a.emission_strength;
    var subsurface = mat_a.subsurface;
    var subsurface_color = mat_subsurface_color(mat_a);
    var noise_scale = mat_a.noise_scale;
    var noise_strength = mat_a.noise_strength;
    var opacity = mat_a.opacity;
    var ior = mat_a.ior;
    // Bitmask channels don't blend meaningfully — keep primary's mask.
    let noise_channels = mat_a.noise_channels;
    if blend_weight > 0.0 {
        let mat_b = materials[secondary_mat_id];
        base_color = mix(base_color, mat_albedo(mat_b), blend_weight);
        metallic_raw = mix(metallic_raw, mat_b.metallic, blend_weight);
        roughness_raw = mix(roughness_raw, mat_b.roughness, blend_weight);
        emission_color = mix(emission_color, mat_emission(mat_b), blend_weight);
        emission_strength = mix(emission_strength, mat_b.emission_strength, blend_weight);
        subsurface = mix(subsurface, mat_b.subsurface, blend_weight);
        subsurface_color = mix(subsurface_color, mat_subsurface_color(mat_b), blend_weight);
        noise_scale = mix(noise_scale, mat_b.noise_scale, blend_weight);
        noise_strength = mix(noise_strength, mat_b.noise_strength, blend_weight);
        opacity = mix(opacity, mat_b.opacity, blend_weight);
        ior = mix(ior, mat_b.ior, blend_weight);
    }

    // User-shader dispatch. When the resolved material has a
    // shader_id != 0, hand off to the user's `shade` hook for the
    // final color and skip the rest of the deferred shade pass.
    // Picks the dominant material's shader_id under blending — the
    // primary unless the secondary's blend weight passes 50 %.
    let primary_shader_id = mat_a.shader_id;
    let secondary_shader_id = select(0u, materials[secondary_mat_id].shader_id, blend_weight > 0.0);
    var active_shader_id = primary_shader_id;
    var active_mat_id = primary_mat_id;
    if blend_weight > 0.5 && secondary_shader_id != 0u {
        active_shader_id = secondary_shader_id;
        active_mat_id = secondary_mat_id;
    }
    if active_shader_id != 0u {
        let V_pre = normalize(camera.position.xyz - world_pos);
        var ctx: ShadeCtx;
        ctx.world_pos = world_pos;
        ctx.distance = pos_data.w;
        ctx.normal = normal;
        ctx.n_dot_v = max(dot(normal, V_pre), 0.001);
        ctx.view_dir = V_pre;
        ctx.time = shade_params.time;
        ctx.base_color = base_color;
        ctx.metallic = metallic_raw;
        ctx.emission_color = emission_color;
        ctx.roughness = roughness_raw;
        ctx._reserved0 = 0.0;
        // Read this material's 8 named params. The buffer is parallel
        // to `materials` — guarded against a stale registry by the
        // arrayLength check, since a freshly-loaded scene can hit a
        // frame between materials buffer upload and shader_params buffer
        // upload (we upload them together, but be defensive).
        if active_mat_id < arrayLength(&shader_params) {
            ctx.params[0] = shader_params[active_mat_id].p0;
            ctx.params[1] = shader_params[active_mat_id].p1;
        } else {
            ctx.params[0] = vec4<f32>(0.0);
            ctx.params[1] = vec4<f32>(0.0);
        }
        ctx.primary_material_id = primary_mat_id;
        ctx.secondary_material_id = secondary_mat_id;
        ctx.blend_weight = blend_weight;
        let result = dispatch_user_shade(active_shader_id, ctx);
        textureStore(output, coord, vec4<f32>(result.rgb, 1.0));
        return;
    }

    let color_rgb565 = (packed_g >> 16u) & 0xFFFFu;
    var albedo = base_color;
    if color_rgb565 != 0u {
        let cr5 = color_rgb565 & 0x1Fu;
        let cg6 = (color_rgb565 >> 5u) & 0x3Fu;
        let cb5 = (color_rgb565 >> 11u) & 0x1Fu;
        albedo = vec3<f32>(f32(cr5) / 31.0, f32(cg6) / 63.0, f32(cb5) / 31.0);
    }
    let metallic = metallic_raw;
    var roughness = max(roughness_raw, 0.04);
    var N = normal;

    // Procedural noise modulation. Single value_noise sample drives albedo
    // / roughness per the noise_channels bitmask; the normal-perturbation
    // bit additionally pays for a 6-sample finite-difference gradient.
    if noise_strength > 0.0 && noise_scale > 0.0 && noise_channels != 0u {
        let np = world_pos * noise_scale;
        let n = value_noise_3d(np) * 2.0 - 1.0; // -1..1
        if (noise_channels & NOISE_CHANNEL_ALBEDO) != 0u {
            albedo = clamp(albedo * (1.0 + n * noise_strength), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        if (noise_channels & NOISE_CHANNEL_ROUGHNESS) != 0u {
            roughness = clamp(roughness + n * noise_strength * 0.5, 0.04, 1.0);
        }
        if (noise_channels & NOISE_CHANNEL_NORMAL) != 0u {
            // Reuse the center sample from above (n in [-1,1]), remapped
            // back to the raw [0,1] range the gradient helper expects.
            let g = noise_gradient_3d(np, n * 0.5 + 0.5);
            let g_tan = g - N * dot(g, N);
            N = normalize(N - g_tan * noise_strength);
        }
    }

    // View direction.
    let V = normalize(camera.position.xyz - world_pos);
    let n_dot_v = max(dot(N, V), 0.001);

    // F0 for dielectrics vs metals.
    // IOR-derived F0 for dielectrics (glass IOR 1.5 → F0 ≈ 0.04,
    // matching the old hardcoded constant; higher IOR materials like
    // diamond (~2.4) now get their correct F0 ≈ 0.17). Metals still
    // tint the spectrum via albedo.
    let f0_dielectric = vec3<f32>(dielectric_f0(ior));
    let f0 = mix(f0_dielectric, albedo, metallic);

    // Per-light shadow: bilateral upsample from the half-res shadow texture
    // written by rkp_shadow_trace. Each half-res sample's "reference
    // surface" is the full-res gbuf pixel at (half_coord * 2); compare
    // that against our pixel's surface to weight the 4 nearest samples
    // and reject neighbors on different surfaces.
    let shadow_dims = textureDimensions(shadow_tex);
    let gbuf_dims = textureDimensions(gbuf_position);
    // Continuous half-res coord: our full-res pixel center in half-res UVs.
    // Full-res pixel (x, y) maps to half-res sample at (x/2, y/2); the
    // 4 nearest integer half-res coords straddle that point.
    let half_coord_f = (vec2<f32>(coord) + 0.5) * 0.5 - 0.5;
    let base_half = vec2<i32>(floor(half_coord_f));
    let frac = half_coord_f - vec2<f32>(base_half);
    let spatial_w = vec4<f32>(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y,
    );
    var shadow_data = vec4<f32>(0.0);
    var w_sum = 0.0;
    var bilinear_data = vec4<f32>(0.0);
    // Isolation: shadow_trace pass is skipped, so shadow_tex contains
    // stale data. Force fully-lit and bypass the bilateral gather.
    // Splat path takes the same branch via `shadow_disabled` — see
    // ShadeParams docs.
    if shade_params.isolation != 0u || shade_params.shadow_disabled != 0u {
        shadow_data = vec4<f32>(1.0);
        w_sum = 1.0;
    } else {
    // Position sigma — in world units. The shadow trace samples one full
    // surface per 2-pixel block, so neighbors straddling a depth
    // discontinuity should get ~zero weight. 5 cm is tight enough to
    // prevent bleed across surface edges at the elephant's scale.
    let sigma_pos = 0.05;
    let inv_sigma2 = 1.0 / (sigma_pos * sigma_pos);
    for (var k = 0u; k < 4u; k++) {
        let dx = i32(k & 1u);
        let dy = i32((k >> 1u) & 1u);
        let half_c = base_half + vec2<i32>(dx, dy);
        if half_c.x < 0 || half_c.y < 0
            || u32(half_c.x) >= shadow_dims.x || u32(half_c.y) >= shadow_dims.y {
            continue;
        }
        // Reference surface for this half-res sample = gbuf at half_c * 2.
        let ref_full = half_c * 2;
        if ref_full.x < 0 || ref_full.y < 0
            || u32(ref_full.x) >= gbuf_dims.x || u32(ref_full.y) >= gbuf_dims.y {
            continue;
        }
        let ref_pos = textureLoad(gbuf_position, ref_full, 0);
        let ref_normal = textureLoad(gbuf_normal, ref_full, 0).xyz;
        // Sky/miss neighbor: skip — no surface to compare against.
        if ref_pos.w >= 1e9 { continue; }

        let s = textureLoad(shadow_tex, half_c, 0);
        bilinear_data += s * spatial_w[k];

        let d_pos = ref_pos.xyz - world_pos;
        let pos_term = exp(-dot(d_pos, d_pos) * inv_sigma2);
        let n_dot = clamp(dot(ref_normal, N), 0.0, 1.0);
        // BILATERAL_NORMAL_POW = 8 — sharper exponent than the cosine
        // weight under-rejects neighbors at >~25° of normal difference;
        // 8 keeps shadow detail from bleeding across surface facets.
        // Open-coded as repeated multiplies to avoid pow()'s log/exp.
        let nd2 = n_dot * n_dot;
        let nd4 = nd2 * nd2;
        let normal_term = nd4 * nd4;
        let w = spatial_w[k] * pos_term * normal_term;
        shadow_data += s * w;
        w_sum += w;
    }
    // Fallback: if bilateral weights all rejected, fall back to plain
    // bilinear so we never return zero from this gather on valid surfaces.
    if w_sum < 1e-5 {
        shadow_data = bilinear_data;
    } else {
        shadow_data /= w_sum;
    }
    } // end !isolation block

    // AO from half-res SSAO texture.
    let half_coord = vec2<i32>(gid.xy) / 2;
    let ao = textureLoad(ssao_tex, half_coord, 0).r;

    // Accumulate direct lighting.
    var lo = vec3<f32>(0.0);
    var shadow_idx = 0u; // tracks which shadow channel to read

    for (var li = 0u; li < shade_params.num_lights; li++) {
        let light = lights[li];
        let light_type = u32(light.position.w);

        var L: vec3<f32>;
        var attenuation = 1.0;

        if light_type == 0u {
            // Directional light.
            L = normalize(-light.direction.xyz);
        } else {
            // Point/spot light.
            let to_light = light.position.xyz - world_pos;
            let dist = length(to_light);
            L = to_light / dist;
            let range = light.params.x;
            if range > 0.0 {
                attenuation = max(1.0 - (dist / range), 0.0);
                attenuation *= attenuation;
            }
        }

        // Read per-light shadow BEFORE the n_dot_l skip — must stay in sync
        // with the march pass which always writes shadow for every shadow-casting light.
        var light_shadow = 1.0;
        let cast_shadow = light.params.w;
        if cast_shadow >= 0.5 && shadow_idx < 4u {
            // Phase 8 S3 — directional lights pull from the shadow
            // map when enabled; everything else still reads the
            // half-res ray-traced shadow_data. We always advance
            // `shadow_idx` so the per-pixel slot mapping (set up
            // by rkp_shadow_trace) stays in sync regardless of
            // which path provided the visibility.
            if light_type == 0u && shade_params.shadow_map_enabled != 0u {
                light_shadow = sample_shadow_map(world_pos)
                    * glass_shadow_attenuation(world_pos);
            } else {
                light_shadow = shadow_data[shadow_idx];
            }
            shadow_idx++;
        }

        let n_dot_l_raw = dot(N, L);
        let n_dot_l_front = max(n_dot_l_raw, 0.0);

        // Wrap Lambert for diffuse: subsurface scalar widens the lit
        // hemisphere so light "leaks" around the terminator.
        // wrap=0 → standard Lambert; wrap=0.5 → full wrap-around.
        let wrap = subsurface * 0.5;
        let n_dot_l_wrapped = max(0.0, (n_dot_l_raw + wrap) / (1.0 + wrap));

        // Back-light transmission (UE4/SpeedTree style). Light arriving
        // from behind the surface scatters forward toward the camera.
        var transmission = vec3<f32>(0.0);
        if subsurface > 0.0 {
            let distortion = 0.2;
            let H_t = normalize(L + N * distortion);
            let t = pow(max(dot(-V, H_t), 0.0), 4.0) * subsurface;
            transmission = t * subsurface_color * albedo;
        }

        // Skip entirely when fully back-facing and no subsurface contribution.
        let has_trans = transmission.r + transmission.g + transmission.b > 0.001;
        if n_dot_l_wrapped <= 0.0 && !has_trans { continue; }

        let H = normalize(V + L);
        let n_dot_h = max(dot(N, H), 0.0);
        let h_dot_v = max(dot(H, V), 0.0);

        // Cook-Torrance BRDF. Specular uses the un-wrapped cosine — the
        // wrap term is a diffuse-only hack; it doesn't model mirror-like
        // light bending through the surface.
        let D = distribution_ggx(n_dot_h, roughness);
        let G = geometry_smith(n_dot_v, n_dot_l_front, roughness);
        let F = fresnel_schlick(h_dot_v, f0);

        let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l_front + 0.0001);
        let kd = (1.0 - F) * (1.0 - metallic);
        let diffuse = kd * albedo / PI;

        let radiance = light.color.rgb * light.color.w * attenuation;

        lo += (diffuse * n_dot_l_wrapped
             + specular * n_dot_l_front
             + transmission) * radiance * light_shadow;
    }

    // Ambient: in-situ uses multi-scattering LUT for sky irradiance; in
    // isolation we substitute a flat neutral irradiance so the result is
    // independent of sun direction / atmosphere state.
    var ms_irradiance: vec3<f32>;
    if shade_params.isolation != 0u {
        ms_irradiance = ISOLATION_AMBIENT;
    } else {
        let cam_height = EARTH_RADIUS + shade_params.camera_altitude;
        let sun_cos_z = shade_params.sun_dir.y;
        ms_irradiance = lookup_multiscatter(multiscatter_lut, atmo_sampler, cam_height, sun_cos_z)
                      * shade_params.sun_intensity
                      * shade_params.ambient_intensity;
    }

    let ambient_diffuse = ms_irradiance * albedo * (1.0 - metallic) * ao;

    // Ambient specular: approximate sky reflection for energy conservation.
    let F_env = fresnel_schlick(n_dot_v, f0);
    let ambient_specular = ms_irradiance * F_env * ao;
    let ambient = ambient_diffuse + ambient_specular;

    // Emission — dedicated emission_color × emission_strength so an
    // emissive material can glow a different color than its albedo.
    // Blended across the dual-material transition band.
    let emission = emission_color * emission_strength;

    var final_color = lo + ambient + emission;

    // Glass composite has moved to the dedicated `rkp_glass` post-
    // pass so it can do screen-space refraction by sampling the
    // already-shaded HDR at an offset pixel. This pass now writes
    // the raw "behind" color (opaque PBR) — rkp_glass overlays
    // Fresnel / Beer / refraction on top if gbuf_glass says this
    // pixel has glass in front of it.

    // Aerial perspective — atmospheric inscatter + extinction between camera
    // and the shaded surface. Under an exponential slice map, near-field
    // precision is fine (slice 0 ≈ 1.2 m) so close voxels receive ≈ 1.0
    // transmittance and ≈ 0 scatter, no discoloration. Far geometry meets
    // the sky-view LUT at the horizon because both integrate the same
    // atmosphere through most of its depth.
    let dims_f = vec2<f32>(dims);
    let screen_uv = (vec2<f32>(coord) + 0.5) / dims_f;
    let surface_dist = length(world_pos - camera.position.xyz);
    let ap = sample_aerial_perspective(screen_uv, surface_dist);
    final_color = final_color * ap.a + ap.rgb;

    // Paint cursor overlay (Phase 3b) — geodesic surface-walking
    // distance from the brush origin, stored per-leaf in
    // `brush_overlay_distances` by the CPU flood fill. This version
    // wraps around corners cleanly: a voxel two hops around an edge
    // is >2*cell_size away even if it's closer in world units.
    //
    // Skip when:
    //   * brush is inactive (cursor off);
    //   * slot == 0 (sky / procedural surface / no hit — the flood
    //     fill has no data for these);
    //   * brush_overlay_distances[slot] is the INFINITY sentinel
    //     (naturally filtered by the `d < r` comparison).
    if shade_params.brush_active != 0u {
        let slot = textureLoad(gbuf_leaf_slot, coord, 0).r;
        if slot != 0u && slot < arrayLength(&brush_overlay_distances) {
            let d = brush_overlay_distances[slot];
            let r = max(shade_params.brush_radius, 1e-4);
            if d < r {
                let t = d / r;
                // Bright rim near the radius edge + a soft fill.
                let rim = smoothstep(0.85, 0.99, t) * (1.0 - smoothstep(0.99, 1.0, t));
                let fill = (1.0 - smoothstep(0.0, 1.0, t)) * 0.25;
                let alpha = clamp(rim + fill, 0.0, 1.0);
                // Local-luminance boost keeps the brush visible against
                // both bright-sunlit and shadowed surfaces.
                let lum = dot(final_color, vec3<f32>(0.299, 0.587, 0.114));
                let scale = max(shade_params.ambient_intensity, lum) * 3.0;
                let overlay = shade_params.brush_color.rgb * scale;
                final_color = mix(final_color, overlay, alpha);
            }
        }
    }

    textureStore(output, coord, vec4<f32>(final_color, 1.0));
}
