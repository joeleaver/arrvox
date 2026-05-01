//! Phase 8 — directional shadow maps.
//!
//! Replaces the per-pixel ray-traced shadow path
//! (`rkp_shadow_trace::trace_shadow_ray`) with a single light-POV
//! march that writes nearest-hit depth into a shadow-map texture.
//! Per-pixel shadow query becomes a single texture sample +
//! depth compare — O(1) per shadow query regardless of how many
//! shadow casters the scene contains.
//!
//! ## Why
//!
//! Per-pixel ray tracing is O(pixels × blades_per_ray). With
//! grass at scale (10k+ instances) the per-blade descent through
//! each instance's prototype octree dominates. Even with the
//! TLAS (Phase 7c) and the directional tile cull (Phase 7d),
//! shadow trace ran ~5 ms for ~100 grass blades — wouldn't fly
//! at 10k+. Shadow maps decouple the per-pixel cost from caster
//! count: the per-pixel path is a single texture sample.
//!
//! ## What this session (S1) ships
//!
//! * `LightCameraUniform` wire format mirroring the WGSL counter-
//!   part the shadow-map march pass (S2) will read.
//! * `compute_light_camera` — CPU helper that derives an
//!   orthographic light-camera view+proj from the sun direction
//!   and the scene's world-space AABB.
//! * `ShadowMapPass` skeleton — owns the depth texture (default
//!   2K × 2K, `R32Float` storage) and the uniform buffer. The
//!   compute pipeline lands in S2.
//!
//! ## V1 limitations
//!
//! * **Directional only.** Point/spot lights still use the
//!   ray-traced shadow path until a follow-up session adds
//!   cube/2D shadow maps for them.
//! * **Single shadow map.** No CSM (cascaded shadow maps); the
//!   one map covers the whole scene's projected light-space
//!   AABB. Quality suffers at large scenes; CSM is a follow-up.
//! * **Hard shadows.** No PCF / VSM / contact-hardening; just a
//!   depth compare. Soft shadows are a polish session later.

use glam::{Mat4, Vec3};

/// Default shadow-map resolution. 2 K square at `R32Float` =
/// 16 MB. Reasonable for V1 outdoor scenes; CSM-style multi-cascade
/// is the answer for very large scenes.
pub const SHADOW_MAP_DEFAULT_SIZE: u32 = 2048;

/// Shadow-map texture format. `R32Float` keeps the storage write
/// path simple for the compute march; depth comparison happens
/// in the shade-side query (S3) by manual texel sample + scalar
/// compare. We could move to `Depth32Float` later if we ever
/// switch the shadow-map render to graphics-pipeline rasterization.
pub const SHADOW_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// "Sky" depth marker stored at shadow-map texels the light-march
/// reaches without hitting any caster. Per-pixel shadow query
/// treats `sample == FAR_DEPTH` as "no occluder along the ray" →
/// returns full transmittance. Chosen as 1.0 (the far-plane NDC z
/// after our orthographic projection's [0,1] z range).
pub const SHADOW_MAP_FAR_DEPTH: f32 = 1.0;

/// Per-frame uniform shared between the shadow-map march (writes
/// the texture) and the shade-side query (samples it). 128 B.
///
/// `view_proj` transforms a world-space point into light clip
/// space; `view` alone can be useful for debug visualizations or
/// for deriving a 1D depth without going through the full
/// projection. Both stored to keep the WGSL math symmetric with
/// the camera path.
///
/// `light_dir` is normalized (the direction of light propagation —
/// e.g., the sun's outgoing rays). The shadow ray from any
/// surface to the light goes in `-light_dir`.
///
/// `inv_shadow_map_size` is `(1.0 / W, 1.0 / H)`; the shade pass
/// uses it for PCF filtering and for converting clip-space xy to
/// integer texel coords.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCameraUniform {
    pub view_proj: [[f32; 4]; 4], // 0..64
    pub view: [[f32; 4]; 4],       // 64..128
    pub light_dir: [f32; 3],        // 128..140
    pub depth_bias: f32,             // 140..144
    pub inv_shadow_map_size: [f32; 2], // 144..152
    pub shadow_map_size: [u32; 2],     // 152..160
}

const _: () = assert!(std::mem::size_of::<LightCameraUniform>() == 160);

/// Derive an orthographic light camera (view + projection) for a
/// directional light shining in `light_dir`, fitting the scene
/// AABB tightly into the projection's view volume.
///
/// Returns:
/// * `view_proj` — world → light NDC ([-1, 1]² × [0, 1] z, wgpu
///   convention).
/// * `view` — world → light view-space (basis transform).
///
/// `light_dir` must be normalized; passing the unnormalized sun
/// direction is fine if its magnitude is close to 1 (the helper
/// re-normalizes internally to be safe).
///
/// The resulting projection is sized to JUST contain the scene
/// AABB. For very wide scenes this means coarse shadow-map
/// resolution per world-meter; CSM is the follow-up that fixes
/// resolution for distant geometry.
pub fn compute_light_camera(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    light_dir: [f32; 3],
    shadow_map_size: u32,
    depth_bias: f32,
) -> LightCameraUniform {
    let l = Vec3::from_array(light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0) // safe default if input is degenerate
    } else {
        l
    };

    // World up — fall back to +Z if light is too close to ±Y.
    let world_up = if l.y.abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();

    // Project all 8 scene-AABB corners into the light's basis to
    // find the orthographic frustum bounds.
    let scene_center = Vec3::new(
        0.5 * (scene_min[0] + scene_max[0]),
        0.5 * (scene_min[1] + scene_max[1]),
        0.5 * (scene_min[2] + scene_max[2]),
    );
    let mut min_xy = [f32::INFINITY; 2];
    let mut max_xy = [f32::NEG_INFINITY; 2];
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lx = right.dot(corner);
        let ly = up.dot(corner);
        let lz = l.dot(corner);
        if lx < min_xy[0] { min_xy[0] = lx; }
        if ly < min_xy[1] { min_xy[1] = ly; }
        if lx > max_xy[0] { max_xy[0] = lx; }
        if ly > max_xy[1] { max_xy[1] = ly; }
        if lz < min_z { min_z = lz; }
        if lz > max_z { max_z = lz; }
    }

    // Place the light "camera" at scene_center pulled back along
    // -L by enough distance to fit the depth range. Using
    // `look_to_rh` so we can supply the direction directly. The
    // eye position only matters for ortho up to a constant
    // translation along L, so any choice that puts every scene
    // point in front of the camera works.
    let z_extent = (max_z - min_z).max(1e-3);
    let eye = scene_center - l * (z_extent * 1.5);
    let view = Mat4::look_to_rh(eye, l, up);

    // Compute ortho bounds in view-space (after `view`). The view
    // matrix's z-axis is -L (right-handed look-to), so view-space
    // z is mostly negative. Find scene corners in view-space to
    // get exact ortho bounds.
    let mut vmin = Vec3::splat(f32::INFINITY);
    let mut vmax = Vec3::splat(f32::NEG_INFINITY);
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let v = view.transform_point3(corner);
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }
    // For RH view, near = -vmax.z, far = -vmin.z (camera looks
    // toward -view.z; smaller view.z = farther).
    let near = -vmax.z;
    let far = -vmin.z;
    // glam's `orthographic_rh` uses [0,1] z range — matches wgpu
    // convention. Resulting view_proj maps world → wgpu-style NDC.
    let proj = Mat4::orthographic_rh(vmin.x, vmax.x, vmin.y, vmax.y, near, far);
    let view_proj = proj * view;

    LightCameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        light_dir: l.to_array(),
        depth_bias,
        inv_shadow_map_size: [
            1.0 / shadow_map_size as f32,
            1.0 / shadow_map_size as f32,
        ],
        shadow_map_size: [shadow_map_size, shadow_map_size],
    }
}

/// Pipeline holder for the shadow-map render. S1 only carries the
/// depth texture + uniform buffer; the compute pipeline lands in
/// S2. The texture's view is exposed so the engine can bind it
/// into both the march pass (storage write) and the shade pass
/// (sample).
pub struct ShadowMapPass {
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub uniform_buffer: wgpu::Buffer,
    pub size: u32,
}

impl ShadowMapPass {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_MAP_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map light_camera_uniform"),
            size: std::mem::size_of::<LightCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            texture,
            texture_view,
            uniform_buffer,
            size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_camera_uniform_size_is_160() {
        assert_eq!(std::mem::size_of::<LightCameraUniform>(), 160);
    }

    #[test]
    fn compute_light_camera_projects_scene_into_clip_space() {
        // Sun straight down: light_dir = (0, -1, 0). Scene = unit
        // cube at origin. Every scene corner should project into
        // [-1, 1]² × [0, 1] clip space.
        let scene_min = [-0.5, -0.5, -0.5];
        let scene_max = [0.5, 0.5, 0.5];
        let light_dir = [0.0, -1.0, 0.0];
        let cam = compute_light_camera(scene_min, scene_max, light_dir, 2048, 0.005);
        let view_proj = Mat4::from_cols_array_2d(&cam.view_proj);

        for c in 0..8u32 {
            let corner = Vec3::new(
                if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
                if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
                if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
            );
            let clip = view_proj * corner.extend(1.0);
            // Ortho ⇒ w = 1.
            assert!((clip.w - 1.0).abs() < 1e-4, "ortho should give w=1");
            assert!(
                clip.x >= -1.0 - 1e-4 && clip.x <= 1.0 + 1e-4,
                "corner {c} clip.x = {} out of [-1, 1]",
                clip.x,
            );
            assert!(
                clip.y >= -1.0 - 1e-4 && clip.y <= 1.0 + 1e-4,
                "corner {c} clip.y = {} out of [-1, 1]",
                clip.y,
            );
            assert!(
                clip.z >= 0.0 - 1e-4 && clip.z <= 1.0 + 1e-4,
                "corner {c} clip.z = {} out of [0, 1]",
                clip.z,
            );
        }
    }

    #[test]
    fn compute_light_camera_handles_y_axis_aligned_light() {
        // Light pointing straight up shouldn't divide by zero on
        // the basis derivation (the world_up fallback kicks in).
        let cam = compute_light_camera(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            1024,
            0.005,
        );
        // view_proj entries should be finite.
        for row in &cam.view_proj {
            for &v in row {
                assert!(v.is_finite(), "view_proj has non-finite value: {v}");
            }
        }
    }

    #[test]
    fn compute_light_camera_depth_increases_along_light_direction() {
        // For sun pointing down, points HIGHER in world Y should
        // map to SMALLER light-space depth (closer to light).
        // Equivalently, projected NDC z increases as we move
        // farther FROM the light.
        let scene_min = [-1.0, -2.0, -1.0];
        let scene_max = [1.0, 2.0, 1.0];
        let cam = compute_light_camera(scene_min, scene_max, [0.0, -1.0, 0.0], 1024, 0.005);
        let vp = Mat4::from_cols_array_2d(&cam.view_proj);
        let high = vp * Vec3::new(0.0, 2.0, 0.0).extend(1.0);
        let low = vp * Vec3::new(0.0, -2.0, 0.0).extend(1.0);
        // Sun shines downward; "high" point (y=+2) is closer to
        // the light (smaller depth) than "low" (y=-2).
        assert!(
            high.z < low.z,
            "expected high point's NDC z ({}) < low ({})",
            high.z,
            low.z,
        );
    }

    #[test]
    fn compute_light_camera_oblique_sun_projects_every_corner_in_clip() {
        // 30° elevation, 45° azimuth — a typical sun direction.
        let l = Vec3::new(-0.612, -0.5, 0.612).normalize();
        let scene_min = [-5.0, 0.0, -5.0];
        let scene_max = [5.0, 3.0, 5.0];
        let cam = compute_light_camera(scene_min, scene_max, l.to_array(), 2048, 0.005);
        let vp = Mat4::from_cols_array_2d(&cam.view_proj);
        for c in 0..8u32 {
            let corner = Vec3::new(
                if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
                if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
                if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
            );
            let clip = vp * corner.extend(1.0);
            assert!(
                clip.x >= -1.0 - 1e-3 && clip.x <= 1.0 + 1e-3,
                "oblique corner {c} clip.x = {} out of bounds",
                clip.x,
            );
            assert!(
                clip.y >= -1.0 - 1e-3 && clip.y <= 1.0 + 1e-3,
                "oblique corner {c} clip.y = {}",
                clip.y,
            );
            assert!(
                clip.z >= 0.0 - 1e-3 && clip.z <= 1.0 + 1e-3,
                "oblique corner {c} clip.z = {}",
                clip.z,
            );
        }
    }
}
