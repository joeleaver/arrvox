//! Phase 8 — directional shadow maps (V2: work-list scatter).
//!
//! Replaces the per-pixel ray-traced shadow path with a four-pass
//! scatter pipeline that lays the geometry down into a shared
//! depth buffer instead of marching rays from the light.
//!
//! ## Why work-list scatter
//!
//! V1 of this pass marched rays from the light's POV (one per
//! shadow-map texel). V1.5 of the scatter approach indirectly
//! dispatched ONCE per TLAS leaf — fast in theory, but the CPU
//! loop's `set_bind_group + dispatch_workgroups_indirect` per
//! instance hit ~5–10 µs of driver overhead each, and dense-grass
//! scenes (1000+ instances) burned multiple ms in dispatch
//! overhead alone.
//!
//! V2 collapses every per-instance dispatch into ONE indirect
//! scatter dispatch over a global work list. Setup pass projects
//! each TLAS prim's AABB to a tile rect, atomic-adds its tile
//! count to a global counter (capturing the per-instance offset).
//! Emit pass parallel-fills `work_list` with packed (instance,
//! tile_x_local, tile_y_local) tuples — workgroups parallelize
//! per instance, threads parallelize across that instance's tiles.
//! Finalize pass converts the total work count into 2D dispatch
//! args. Scatter pass dispatches ONCE indirectly; each workgroup
//! reads its work-list entry, descends the indicated instance for
//! its 8×8 tile, atomic-mins depth into `shadow_buffer`.
//!
//! Per-frame dispatch count: 5 (clear / setup / emit / finalize /
//! scatter), regardless of scene complexity.
//!
//! ## V1 limitations (carry from V1)
//!
//! * **Directional only.** Spot/point lights still use the
//!   ray-traced shadow path.
//! * **Single shadow map.** No CSM yet; one map covers the whole
//!   scene's projected light-space AABB.
//! * **Hard shadows.** No PCF / VSM; just a depth compare.
//! * **Opacity ignored.** Every voxel counts as opaque.

use glam::{Mat4, Vec3};

use crate::validate_wgsl;

/// Default shadow-map resolution. 1 K square at 4 bytes / texel
/// = 4 MB. With frustum-fit + scene clip, a 1 K map at typical
/// view bounds gives ~3 cm/texel — sharper than scene-fit 2 K
/// would. CSM is the long-term lever for far-field detail.
pub const SHADOW_MAP_DEFAULT_SIZE: u32 = 1024;

/// Cap on the distance (world units, from camera) that the
/// frustum-fit shadow camera covers. The camera's actual far
/// plane can be 10 km+; capping keeps per-meter texel density
/// high in the visible region. The proper fix (visible-caster
/// AABB fit) makes this less load-bearing — until then, 30 m
/// keeps shadow-map texels in the ~1.5 cm range for typical
/// scenes.
pub const SHADOW_FAR_DISTANCE: f32 = 30.0;

/// "Sky" depth marker. Per-pixel shadow query treats
/// `sample == FAR_DEPTH` as "no occluder" → returns full
/// transmittance.
pub const SHADOW_MAP_FAR_DEPTH: f32 = 1.0;

/// `bitcast::<u32>(1.0)` — what the clear pass writes into every
/// entry of `shadow_buffer`. atomic-min on the u32 representation
/// works because f32 in [0, 1] is monotonic in IEEE-754.
pub const SHADOW_MAP_FAR_DEPTH_BITS: u32 = 0x3F800000;

/// Initial capacity for the per-frame TLAS-prim → ScatterInstance
/// scratch arrays. Grows on demand if the prim count exceeds it.
pub const SHADOW_MAP_MAX_CASTERS_INITIAL: u32 = 2048;

/// Initial work-list capacity (one entry per 8×8 tile). 256 K
/// entries ≈ 1 MB at 4 bytes / entry. Covers ~4 instances each
/// fully covering a 2K shadow map (65 536 tiles each), or ~10 k
/// grass blades (~25 tiles each). Grows on demand.
pub const SHADOW_MAP_WORK_LIST_INITIAL: u32 = 262144;

/// Scatter pass dispatch X dimension — must match the constant in
/// `shadow_scatter.wgsl` and `shadow_scatter_finalize.wgsl`. The
/// finalize pass writes `(DISPATCH_X, ceil(total / DISPATCH_X), 1)`
/// into the indirect dispatch args.
pub const SHADOW_SCATTER_DISPATCH_X: u32 = 256;

/// Stride of a `ScatterInstance` slot — see WGSL definition. 8 ×
/// u32 = 32 bytes.
pub const SCATTER_INSTANCE_STRIDE: u64 = 32;

/// Per-frame uniform shared between the shadow-map setup +
/// scatter passes (writes the depth) and the shade-side query
/// (reads it). 160 B.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_proj_inv: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub depth_bias: f32,
    pub inv_shadow_map_size: [f32; 2],
    pub shadow_map_size: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<LightCameraUniform>() == 160);

/// Setup-pass per-frame uniform. Layout matches WGSL struct in
/// `shadow_scatter_setup.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SetupParams {
    pub prim_count: u32,
    /// Maximum distance a shadow can travel through the scene.
    /// Setup uses this to extrude per-prim AABBs along
    /// `light_dir` for the shadow-frustum cull.
    pub scene_extent: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    /// Camera view-proj matrix (world → camera NDC). The cull
    /// projects each prim's swept AABB through this and tests
    /// the resulting NDC bounds against `[-1,1]² × [0,1]`.
    pub camera_view_proj: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<SetupParams>() == 80);

/// Derive an orthographic light camera. See V1 commit 3d862b0 for
/// the derivation rationale (look_to + scene-AABB-fit).
pub fn compute_light_camera(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    light_dir: [f32; 3],
    shadow_map_size: u32,
    depth_bias: f32,
) -> LightCameraUniform {
    let l = Vec3::from_array(light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0)
    } else {
        l
    };
    let world_up = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();
    let scene_center = Vec3::new(
        0.5 * (scene_min[0] + scene_max[0]),
        0.5 * (scene_min[1] + scene_max[1]),
        0.5 * (scene_min[2] + scene_max[2]),
    );
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lz = l.dot(corner);
        if lz < min_z { min_z = lz; }
        if lz > max_z { max_z = lz; }
    }
    let z_extent = (max_z - min_z).max(1e-3);
    let eye = scene_center - l * (z_extent * 1.5);
    let view = Mat4::look_to_rh(eye, l, up);
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
    let near = -vmax.z;
    let far = -vmin.z;
    let proj = Mat4::orthographic_rh(vmin.x, vmax.x, vmin.y, vmax.y, near, far);
    let view_proj = proj * view;
    let view_proj_inv = view_proj.inverse();
    LightCameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        view_proj_inv: view_proj_inv.to_cols_array_2d(),
        light_dir: l.to_array(),
        depth_bias,
        inv_shadow_map_size: [
            1.0 / shadow_map_size as f32,
            1.0 / shadow_map_size as f32,
        ],
        shadow_map_size: [shadow_map_size, shadow_map_size],
    }
}

/// Per-VR frustum-fit light camera. Same shape as
/// `compute_light_camera` but the orthographic xy bounds are
/// fitted to the camera's *visible* frustum rather than the whole
/// scene AABB. The z range is extended to encompass the whole
/// scene's light-space depth so casters outside the visible
/// frustum (e.g., a tower above the camera's forward cone) still
/// reach the shadow map.
///
/// The camera's far plane is clamped at `shadow_far_dist` from
/// the camera position — the camera's actual far plane can be
/// kilometers, which would dilute texel density in the foreground.
/// CSM is the proper fix for variable depth ranges; this single-
/// cascade approach trades far-field shadow quality for near-field
/// sharpness.
///
/// Inputs:
/// * `scene_min` / `scene_max` — world-space bounds of all shadow
///   casters. Used only for z-range extension; not for xy.
/// * `camera_view_proj_inv` — inverse of the camera's view-proj
///   matrix. Used to unproject the 8 NDC frustum corners into
///   world space.
/// * `camera_position` — world-space camera origin. Used to clamp
///   the far corners' distance to `shadow_far_dist`.
/// * `light_dir`, `shadow_map_size`, `depth_bias` — same as the
///   scene-fit variant.
/// * `shadow_far_dist` — camera-relative far cap for the fit.
pub fn compute_light_camera_frustum_fit(
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    camera_view_proj_inv: Mat4,
    camera_position: Vec3,
    light_dir: [f32; 3],
    shadow_map_size: u32,
    depth_bias: f32,
    shadow_far_dist: f32,
) -> LightCameraUniform {
    let l = Vec3::from_array(light_dir).normalize_or_zero();
    let l = if l.length_squared() < 0.5 {
        Vec3::new(0.0, -1.0, 0.0)
    } else {
        l
    };
    let world_up = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
    let right = world_up.cross(l).normalize_or_zero();
    let up = l.cross(right).normalize_or_zero();

    // 8 frustum corners in NDC: (±1, ±1, {0, 1}). z=0 = near
    // plane, z=1 = far plane (wgpu convention).
    let mut frustum_world: [Vec3; 8] = [Vec3::ZERO; 8];
    for c in 0..8u32 {
        let ndc = Vec3::new(
            if (c & 1) != 0 { 1.0 } else { -1.0 },
            if (c & 2) != 0 { 1.0 } else { -1.0 },
            if (c & 4) != 0 { 1.0 } else { 0.0 },
        );
        let world = camera_view_proj_inv * ndc.extend(1.0);
        let world_pos = world.truncate() / world.w;
        // Far corners: clamp distance from camera. The camera's
        // far plane can be 10 km+; clamp keeps per-meter density
        // high in the foreground. Near corners pass through.
        if (c & 4) != 0 {
            let dir = world_pos - camera_position;
            let dist = dir.length();
            if dist > shadow_far_dist {
                frustum_world[c as usize] =
                    camera_position + dir / dist * shadow_far_dist;
            } else {
                frustum_world[c as usize] = world_pos;
            }
        } else {
            frustum_world[c as usize] = world_pos;
        }
    }

    // Set the eye well behind the scene along -L. Distance is
    // chosen so every potential caster sits in front of the
    // ortho's near plane.
    let scene_center = Vec3::new(
        0.5 * (scene_min[0] + scene_max[0]),
        0.5 * (scene_min[1] + scene_max[1]),
        0.5 * (scene_min[2] + scene_max[2]),
    );
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let lz = l.dot(corner);
        if lz < min_z { min_z = lz; }
        if lz > max_z { max_z = lz; }
    }
    let z_extent = (max_z - min_z).max(1e-3);
    let eye = scene_center - l * (z_extent * 1.5);
    let view = Mat4::look_to_rh(eye, l, up);

    // Project camera frustum AND scene AABB into light view-space.
    let mut frustum_vmin = Vec3::splat(f32::INFINITY);
    let mut frustum_vmax = Vec3::splat(f32::NEG_INFINITY);
    for &corner in &frustum_world {
        let v = view.transform_point3(corner);
        frustum_vmin = frustum_vmin.min(v);
        frustum_vmax = frustum_vmax.max(v);
    }
    let mut scene_vmin = Vec3::splat(f32::INFINITY);
    let mut scene_vmax = Vec3::splat(f32::NEG_INFINITY);
    for c in 0..8u32 {
        let corner = Vec3::new(
            if (c & 1) != 0 { scene_max[0] } else { scene_min[0] },
            if (c & 2) != 0 { scene_max[1] } else { scene_min[1] },
            if (c & 4) != 0 { scene_max[2] } else { scene_min[2] },
        );
        let v = view.transform_point3(corner);
        scene_vmin = scene_vmin.min(v);
        scene_vmax = scene_vmax.max(v);
    }

    // INTERSECT xy: shadow map should only cover the visible
    // region that contains scene geometry. The camera frustum's
    // far plane can project to a huge area (200 m far cap × ~90°
    // FOV ~= 400 m × 200 m in light xy), but if the scene AABB
    // is small (e.g., 10 m × 10 m), the frustum bounds dilute
    // texel density.
    //
    // Z bounds: full scene span (any caster between the visible
    // surfaces and the light belongs in the shadow map).
    let xy_min = Vec3::new(
        frustum_vmin.x.max(scene_vmin.x),
        frustum_vmin.y.max(scene_vmin.y),
        0.0,
    );
    let xy_max = Vec3::new(
        frustum_vmax.x.min(scene_vmax.x),
        frustum_vmax.y.min(scene_vmax.y),
        0.0,
    );
    // Empty intersection (camera looking away from scene): fall
    // back to scene-fit so the shadow map still has valid bounds.
    let (final_x_min, final_x_max, final_y_min, final_y_max) =
        if xy_min.x >= xy_max.x || xy_min.y >= xy_max.y {
            (scene_vmin.x, scene_vmax.x, scene_vmin.y, scene_vmax.y)
        } else {
            (xy_min.x, xy_max.x, xy_min.y, xy_max.y)
        };

    let near = -scene_vmax.z;
    let far = -scene_vmin.z;
    let proj = Mat4::orthographic_rh(
        final_x_min, final_x_max, final_y_min, final_y_max, near, far,
    );
    let view_proj = proj * view;
    let view_proj_inv = view_proj.inverse();

    LightCameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        view_proj_inv: view_proj_inv.to_cols_array_2d(),
        light_dir: l.to_array(),
        depth_bias,
        inv_shadow_map_size: [
            1.0 / shadow_map_size as f32,
            1.0 / shadow_map_size as f32,
        ],
        shadow_map_size: [shadow_map_size, shadow_map_size],
    }
}

/// Pipeline holder for the work-list scatter shadow render. Owns
/// five compute pipelines (clear / setup / emit / finalize /
/// scatter), the depth-target storage buffer, the per-frame
/// uniforms, and per-frame scatter scratch (instances + work
/// list + counter + indirect args).
pub struct ShadowMapPass {
    pub size: u32,
    pub uniform_buffer: wgpu::Buffer,

    /// `array<atomic<u32>>` of length `size * size` — bit-cast
    /// f32 depths.
    pub shadow_buffer: wgpu::Buffer,

    setup_params_buffer: wgpu::Buffer,

    /// `atomic<u32>` global counter. Setup atomic-adds tile counts
    /// here; finalize reads it; engine zeros it before setup
    /// (`encoder.clear_buffer`).
    total_work_buffer: wgpu::Buffer,

    /// `array<u32>` — packed `(instance_idx:16, tile_x:8, tile_y:8)`
    /// per 8×8 tile. Filled by the emit pass; read by the scatter.
    pub work_list_buffer: wgpu::Buffer,

    /// `array<u32>` of length 4: `(x, y, z, total_work)`. Finalize
    /// writes; scatter reads `[3]` for bounds-check.
    pub dispatch_args_buffer: wgpu::Buffer,

    /// `array<ScatterInstance>` — written by setup, read by emit
    /// + scatter.
    pub scatter_instances_buffer: wgpu::Buffer,
    pub scatter_capacity: u32,

    // ── Pipelines + bind groups ────────────────────────────────
    clear_pipeline: wgpu::ComputePipeline,
    clear_g0_bg: wgpu::BindGroup,

    setup_pipeline: wgpu::ComputePipeline,
    setup_g0_layout: wgpu::BindGroupLayout,
    setup_g0_bg: Option<wgpu::BindGroup>,
    setup_g1_bg: wgpu::BindGroup,

    emit_pipeline: wgpu::ComputePipeline,
    emit_g0_layout: wgpu::BindGroupLayout,
    emit_g0_bg: wgpu::BindGroup,

    finalize_pipeline: wgpu::ComputePipeline,
    finalize_g0_bg: wgpu::BindGroup,

    scatter_pipeline: wgpu::ComputePipeline,
    scatter_pipeline_layout: wgpu::PipelineLayout,
    scatter_pass_layout: wgpu::BindGroupLayout,
    scatter_pass_bg: wgpu::BindGroup,
    user_shader_source_hash: u64,
}

impl ShadowMapPass {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        size: u32,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // ── Buffers ────────────────────────────────────────────
        let shadow_buffer_bytes = (size as u64) * (size as u64) * 4;
        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map shadow_buffer"),
            size: shadow_buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map light_camera_uniform"),
            size: std::mem::size_of::<LightCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let setup_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map setup_params"),
            size: std::mem::size_of::<SetupParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let total_work_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map total_work"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map dispatch_args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scatter_capacity = SHADOW_MAP_MAX_CASTERS_INITIAL;
        let scatter_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map scatter_instances"),
            size: (scatter_capacity as u64) * SCATTER_INSTANCE_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let work_list_capacity = SHADOW_MAP_WORK_LIST_INITIAL;
        let work_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map work_list"),
            size: (work_list_capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Layouts ────────────────────────────────────────────
        let clear_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_clear g0"),
            entries: &[rw_storage_layout_entry(0)],
        });
        let setup_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_setup g0"),
            entries: &[
                ro_storage_layout_entry(0), // tlas_prims
                rw_storage_layout_entry(1), // scatter_instances
                rw_storage_layout_entry(2), // total_work
            ],
        });
        let setup_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_setup g1"),
            entries: &[uniform_layout_entry(0), uniform_layout_entry(1)],
        });
        let emit_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_emit g0"),
            entries: &[
                ro_storage_layout_entry(0), // scatter_instances
                rw_storage_layout_entry(1), // work_list
            ],
        });
        let finalize_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_finalize g0"),
            entries: &[
                rw_storage_layout_entry(0), // total_work
                rw_storage_layout_entry(1), // dispatch_args
            ],
        });
        let scatter_pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_scatter g1"),
            entries: &[
                uniform_layout_entry(0),    // light_camera
                rw_storage_layout_entry(1), // shadow_buffer (atomic)
                ro_storage_layout_entry(2), // scatter_instances
                ro_storage_layout_entry(3), // work_list
                ro_storage_layout_entry(4), // dispatch_args (read-only here)
            ],
        });

        // ── Pipelines ──────────────────────────────────────────
        let clear_pipeline = build_pipeline(
            device, "shadow_clear",
            include_str!("shaders/shadow_clear.wgsl"),
            "clear_main",
            &[Some(&clear_g0_layout)],
        );
        let setup_pipeline = build_pipeline(
            device, "shadow_scatter_setup",
            include_str!("shaders/shadow_scatter_setup.wgsl"),
            "setup_main",
            &[Some(&setup_g0_layout), Some(&setup_g1_layout)],
        );
        let emit_pipeline = build_pipeline(
            device, "shadow_scatter_emit",
            include_str!("shaders/shadow_scatter_emit.wgsl"),
            "emit_main",
            &[Some(&emit_g0_layout)],
        );
        let finalize_pipeline = build_pipeline(
            device, "shadow_scatter_finalize",
            include_str!("shaders/shadow_scatter_finalize.wgsl"),
            "finalize_main",
            &[Some(&finalize_g0_layout)],
        );
        let scatter_src = include_str!("shaders/shadow_scatter.wgsl");
        validate_wgsl(scatter_src, "shadow_scatter");
        let scatter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_scatter"),
            source: wgpu::ShaderSource::Wgsl(scatter_src.into()),
        });
        let scatter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_scatter pipeline layout"),
            bind_group_layouts: &[
                Some(scene_bind_group_layout),
                Some(&scatter_pass_layout),
            ],
            immediate_size: 0,
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Bind groups (resources we own) ─────────────────────
        let clear_g0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_clear g0 bg"),
            layout: &clear_g0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_buffer.as_entire_binding(),
            }],
        });
        let setup_g1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_setup g1 bg"),
            layout: &setup_g1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: setup_params_buffer.as_entire_binding() },
            ],
        });
        let emit_g0_bg = build_emit_g0_bg(
            device, &emit_g0_layout,
            &scatter_instances_buffer, &work_list_buffer,
        );
        let finalize_g0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_finalize g0 bg"),
            layout: &finalize_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: total_work_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dispatch_args_buffer.as_entire_binding() },
            ],
        });
        let scatter_pass_bg = build_scatter_pass_bg(
            device, &scatter_pass_layout,
            &uniform_buffer, &shadow_buffer, &scatter_instances_buffer,
            &work_list_buffer, &dispatch_args_buffer,
        );

        Self {
            size,
            uniform_buffer,
            shadow_buffer,
            setup_params_buffer,
            total_work_buffer,
            work_list_buffer,
            dispatch_args_buffer,
            scatter_instances_buffer,
            scatter_capacity,
            clear_pipeline,
            clear_g0_bg,
            setup_pipeline,
            setup_g0_layout,
            setup_g0_bg: None,
            setup_g1_bg,
            emit_pipeline,
            emit_g0_layout,
            emit_g0_bg,
            finalize_pipeline,
            finalize_g0_bg,
            scatter_pipeline,
            scatter_pipeline_layout,
            scatter_pass_layout,
            scatter_pass_bg,
            user_shader_source_hash: 0,
        }
    }

    /// Rebuild the scatter pipeline against spliced user-shader chunks.
    /// Shadow-map scatter doesn't run `instance_at` (Phase 4 will add
    /// per-leaf grass descent into the half-res shadow path, but the
    /// scatter pass itself just rasterizes screen-space AABBs).
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.user_shader_source_hash {
            return false;
        }
        let template = include_str!("shaders/shadow_scatter.wgsl");
        // Pass empty for instance_at — the splice helper short-circuits
        // when the chunk is empty (shadow_scatter has no markers in V1).
        let source = crate::shader_composer::splice_inst_chunks(
            template, "",
        );
        validate_wgsl(&source, "shadow_scatter");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_scatter"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_scatter"),
            layout: Some(&self.scatter_pipeline_layout),
            module: &module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.user_shader_source_hash = source_hash;
        true
    }

    /// Bind the TLAS prims buffer the setup pass reads.
    pub fn set_tlas_prims_buffer(
        &mut self,
        device: &wgpu::Device,
        tlas_prims: &wgpu::Buffer,
    ) {
        self.setup_g0_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_setup g0 bg"),
            layout: &self.setup_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tlas_prims.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scatter_instances_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.total_work_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Grow the scatter scratch + work-list buffers as needed.
    /// Engine calls this each frame before `dispatch_setup`.
    pub fn ensure_scatter_capacity(
        &mut self,
        device: &wgpu::Device,
        prim_count: u32,
    ) -> bool {
        let mut grew = false;
        if prim_count > self.scatter_capacity {
            let mut new_cap = self.scatter_capacity.max(1);
            while new_cap < prim_count {
                new_cap = new_cap.saturating_mul(2);
            }
            self.scatter_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_map scatter_instances"),
                size: (new_cap as u64) * SCATTER_INSTANCE_STRIDE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter_capacity = new_cap;
            self.setup_g0_bg = None; // engine rebinds via set_tlas_prims_buffer
            self.emit_g0_bg = build_emit_g0_bg(
                device, &self.emit_g0_layout,
                &self.scatter_instances_buffer, &self.work_list_buffer,
            );
            self.scatter_pass_bg = build_scatter_pass_bg(
                device, &self.scatter_pass_layout,
                &self.uniform_buffer, &self.shadow_buffer,
                &self.scatter_instances_buffer,
                &self.work_list_buffer, &self.dispatch_args_buffer,
            );
            grew = true;
        }
        grew
    }

    /// Record the clear pass — fills `shadow_buffer` with FAR_DEPTH
    /// bits AND zeros `total_work` for the upcoming setup pass.
    pub fn dispatch_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        // total_work counter must start at 0 each frame.
        encoder.clear_buffer(&self.total_work_buffer, 0, Some(4));
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_clear"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.clear_pipeline);
        cpass.set_bind_group(0, &self.clear_g0_bg, &[]);
        let groups = self.size.div_ceil(8);
        cpass.dispatch_workgroups(groups, groups, 1);
    }

    /// Record the setup pass — projects TLAS prims, fills
    /// `scatter_instances`, atomic-adds tile counts to `total_work`.
    /// `camera_view_proj` and `scene_extent` drive the shadow-
    /// frustum cull (skip prims whose swept volume can't reach
    /// the camera view).
    pub fn dispatch_setup(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        prim_count: u32,
        camera_view_proj: [[f32; 4]; 4],
        scene_extent: f32,
    ) {
        let Some(ref g0) = self.setup_g0_bg else { return; };
        queue.write_buffer(
            &self.setup_params_buffer,
            0,
            bytemuck::bytes_of(&SetupParams {
                prim_count,
                scene_extent,
                _pad0: 0, _pad1: 0,
                camera_view_proj,
            }),
        );
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_setup"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.setup_pipeline);
        cpass.set_bind_group(0, g0, &[]);
        cpass.set_bind_group(1, &self.setup_g1_bg, &[]);
        let workgroups = self.scatter_capacity.div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Record the emit pass — fills `work_list[scatter_instances[i]
    /// .work_offset + 0..tile_count]` with packed work tuples.
    /// One workgroup of 64 threads per instance.
    pub fn dispatch_emit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prim_count: u32,
    ) {
        if prim_count == 0 { return; }
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_emit"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.emit_pipeline);
        cpass.set_bind_group(0, &self.emit_g0_bg, &[]);
        cpass.dispatch_workgroups(prim_count, 1, 1);
    }

    /// Record the finalize pass — packs `total_work` into the
    /// scatter pass's indirect-dispatch args.
    pub fn dispatch_finalize(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_finalize"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.finalize_pipeline);
        cpass.set_bind_group(0, &self.finalize_g0_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    /// Record the scatter pass — single indirect dispatch over
    /// the work list. Each workgroup descends one instance for
    /// one 8×8 tile, atomic-mins depth into `shadow_buffer`.
    pub fn dispatch_scatter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bind_group: &wgpu::BindGroup,
        prim_count: u32,
    ) {
        if prim_count == 0 { return; }
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.scatter_pipeline);
        cpass.set_bind_group(0, scene_bind_group, &[]);
        cpass.set_bind_group(1, &self.scatter_pass_bg, &[]);
        cpass.dispatch_workgroups_indirect(&self.dispatch_args_buffer, 0);
    }
}

fn ro_storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    entry_point: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::ComputePipeline {
    validate_wgsl(src, label);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} pipeline layout")),
        bind_group_layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn build_emit_g0_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scatter_instances_buffer: &wgpu::Buffer,
    work_list_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_emit g0 bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scatter_instances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: work_list_buffer.as_entire_binding() },
        ],
    })
}

fn build_scatter_pass_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    shadow_buffer: &wgpu::Buffer,
    scatter_instances_buffer: &wgpu::Buffer,
    work_list_buffer: &wgpu::Buffer,
    dispatch_args_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_scatter pass bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: shadow_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scatter_instances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: work_list_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dispatch_args_buffer.as_entire_binding() },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_camera_uniform_size_is_160() {
        assert_eq!(std::mem::size_of::<LightCameraUniform>(), 160);
    }

    fn assert_wgsl_valid(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("[{label}] parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("[{label}] validation error: {e:?}"));
    }

    #[test]
    fn shadow_clear_shader_is_valid_wgsl() {
        assert_wgsl_valid("shadow_clear", include_str!("shaders/shadow_clear.wgsl"));
    }

    #[test]
    fn shadow_scatter_setup_shader_is_valid_wgsl() {
        assert_wgsl_valid("shadow_scatter_setup", include_str!("shaders/shadow_scatter_setup.wgsl"));
    }

    #[test]
    fn shadow_scatter_emit_shader_is_valid_wgsl() {
        assert_wgsl_valid("shadow_scatter_emit", include_str!("shaders/shadow_scatter_emit.wgsl"));
    }

    #[test]
    fn shadow_scatter_finalize_shader_is_valid_wgsl() {
        assert_wgsl_valid("shadow_scatter_finalize", include_str!("shaders/shadow_scatter_finalize.wgsl"));
    }

    #[test]
    fn shadow_scatter_shader_is_valid_wgsl() {
        assert_wgsl_valid("shadow_scatter", include_str!("shaders/shadow_scatter.wgsl"));
    }

    #[test]
    fn compute_light_camera_view_proj_inv_round_trips() {
        let cam = compute_light_camera(
            [-2.0, 0.0, -3.0], [4.0, 5.0, 1.0],
            Vec3::new(-0.3, -0.7, 0.5).normalize().to_array(),
            2048, 0.005,
        );
        let vp = Mat4::from_cols_array_2d(&cam.view_proj);
        let vpi = Mat4::from_cols_array_2d(&cam.view_proj_inv);
        for &ndc in &[
            [-0.9_f32, -0.9, 0.0],
            [0.9, -0.9, 0.5],
            [0.0, 0.0, 0.7],
            [0.7, 0.3, 1.0],
        ] {
            let world = vpi * Vec3::new(ndc[0], ndc[1], ndc[2]).extend(1.0);
            let world = world.truncate() / world.w;
            let clip = vp * world.extend(1.0);
            let recovered = clip.truncate() / clip.w;
            assert!((recovered.x - ndc[0]).abs() < 1e-3);
            assert!((recovered.y - ndc[1]).abs() < 1e-3);
            assert!((recovered.z - ndc[2]).abs() < 1e-3);
        }
    }
}
