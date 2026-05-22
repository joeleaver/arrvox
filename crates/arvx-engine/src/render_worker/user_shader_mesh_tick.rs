//! V1 mesh-path user-shader per-frame orchestration.
//!
//! For each painted material whose user shader opts into the mesh path
//! (via `@geometry` + `vs` + `spawn_count`), this module:
//!
//!   1. Builds the per-shader pipelines on first use or source-hash
//!      change (composes WGSL via `compose_mesh_path_pipeline_sources`,
//!      then `UserShaderMeshPass::build_pipelines`).
//!   2. Allocates fixed-capacity GPU buffers (per material) for
//!      anchors / counts / offsets / records / indirect args /
//!      uniforms. V1 caps at 262 144 anchors × 32 spawns/anchor.
//!   3. Uploads the per-frame data: anchor records, FrameUniforms,
//!      UserShaderParams, DispatchInfo.
//!   4. Dispatches the three compute passes: spawn_count → prefix_sum
//!      → fill. The fill output drives an indirect draw the renderer
//!      consumes in a later pass (currently a stub — task #7).
//!
//! Static-cache gating is wired but not yet honored — V1 always
//! re-runs the compute trio. Caching by (paint_epoch + geometry_epoch
//! + params hash) is a follow-up.

use arvx_render::user_shader_mesh_pass::{
    AnchorRecord, DispatchInfo, DrawIndirectArgs, FrameUniforms, InstanceRecord,
    UserShaderMeshDraw, UserShaderMeshPipelines, UserShaderParams,
    MAX_ANCHORS_PER_SHADER_V1, PREFIX_SUM_MAX_WG_COUNT, PREFIX_SUM_MAX_WG_COUNT_2,
};

use crate::render_frame::RenderFrame;
use crate::viewport::ViewportId;

use super::state::RenderState;

/// Six frustum planes extracted from a view-projection matrix. Each
/// plane is stored as `(nx, ny, nz, d)` where a point `p` is inside
/// the plane when `dot(n, p) + d >= 0`. Convention: glam's
/// `perspective_rh` produces clip space with `z ∈ [0, 1]` (Vulkan /
/// D3D / wgpu), so the near plane is `row(2)` (no `row(3)` offset).
fn extract_frustum_planes(vp: glam::Mat4) -> [glam::Vec4; 6] {
    // glam stores Mat4 column-major; row(i) is the i-th element of
    // each column.
    let r0 = glam::Vec4::new(vp.col(0).x, vp.col(1).x, vp.col(2).x, vp.col(3).x);
    let r1 = glam::Vec4::new(vp.col(0).y, vp.col(1).y, vp.col(2).y, vp.col(3).y);
    let r2 = glam::Vec4::new(vp.col(0).z, vp.col(1).z, vp.col(2).z, vp.col(3).z);
    let r3 = glam::Vec4::new(vp.col(0).w, vp.col(1).w, vp.col(2).w, vp.col(3).w);
    [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near (clip z = 0)
        r3 - r2, // far
    ]
}

/// Returns true if the AABB sits entirely outside at least one
/// frustum plane (p-vertex test — conservative, no false rejects).
fn aabb_outside_frustum(
    min: glam::Vec3,
    max: glam::Vec3,
    planes: &[glam::Vec4; 6],
) -> bool {
    for &plane in planes {
        let n = plane.truncate();
        let d = plane.w;
        // p-vertex: corner of the AABB most in the plane normal's
        // direction — if it's behind, every corner is behind.
        let p_vertex = glam::Vec3::new(
            if n.x >= 0.0 { max.x } else { min.x },
            if n.y >= 0.0 { max.y } else { min.y },
            if n.z >= 0.0 { max.z } else { min.z },
        );
        if n.dot(p_vertex) + d < 0.0 {
            return true;
        }
    }
    false
}

/// Filter anchors against the camera frustum + per-shader
/// `max_distance` (tile center vs camera position). Drops anchors
/// outside the frustum OR beyond the radius. Pass `None` for
/// `planes` / `max_distance` to disable the respective filter.
///
/// Always returns a fresh `Vec<AnchorRecord>`: even with neither
/// filter active we still take a copy because the upload happens
/// from a slice that the caller owns. Cost is `O(input_len)` —
/// at the V1 cap (262 144 anchors × 80 B ≈ 20 MB) this is a few
/// ms; well below the compute-trio budget.
fn filter_anchors(
    anchors: &[AnchorRecord],
    cam_pos: glam::Vec3,
    planes: Option<&[glam::Vec4; 6]>,
    max_distance: Option<f32>,
) -> Vec<AnchorRecord> {
    let max_dist_sq = max_distance.map(|d| d * d);
    let mut out = Vec::with_capacity(anchors.len());
    for a in anchors {
        let center = glam::Vec3::new(
            (a.tile_min[0] + a.tile_max[0]) * 0.5,
            (a.tile_min[1] + a.tile_max[1]) * 0.5,
            (a.tile_min[2] + a.tile_max[2]) * 0.5,
        );
        if let Some(d_sq) = max_dist_sq {
            let delta = center - cam_pos;
            if delta.length_squared() > d_sq {
                continue;
            }
        }
        if let Some(p) = planes {
            let mn = glam::Vec3::from(a.tile_min);
            let mx = glam::Vec3::from(a.tile_max);
            if aabb_outside_frustum(mn, mx, p) {
                continue;
            }
        }
        out.push(*a);
    }
    out
}

/// Fixed V1 cap on per-anchor spawn count. With
/// `MAX_ANCHORS_PER_SHADER_V1 = 262 144` and a 32-spawn/anchor cap, the
/// records buffer is `262 144 × 32 × 8 B ≈ 64 MB` per material — half
/// the size of the V1 64-cap. The WESL `entry_spawn_count` clamps the
/// user shader's return to this constant so a runaway density slider
/// cannot overflow the records buffer; the visible blade cap at
/// `@tile_size 0.5` becomes `32 / 0.25 m² = 128 blades/m²`, which is
/// well above any realistic grass density.
pub const MAX_SPAWNS_PER_ANCHOR_V1: u32 = 32;
pub const MAX_SPAWNS_PER_SHADER_V1: u32 =
    MAX_ANCHORS_PER_SHADER_V1 * MAX_SPAWNS_PER_ANCHOR_V1;

/// Per-material runtime state. Created on first use of a mesh-path
/// material; rebuilt when the shader's `source_hash` changes.
pub(super) struct MeshUserShaderMaterialState {
    pub source_hash: u64,
    pub pipelines: UserShaderMeshPipelines,
    pub anchors_buffer: wgpu::Buffer,
    pub counts_buffer: wgpu::Buffer,
    pub offsets_buffer: wgpu::Buffer,
    pub records_buffer: wgpu::Buffer,
    pub indirect_buffer: wgpu::Buffer,
    pub frame_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub dispatch_buffer: wgpu::Buffer,
    pub wg_sums_buffer: wgpu::Buffer,
    pub wg_sums2_buffer: wgpu::Buffer,
    /// Last per-material params snapshot we uploaded to
    /// `params_buffer`. If the next frame's params for this material
    /// differ, we force re-upload + compute trio rerun even when
    /// `anchors_unchanged` would otherwise let us skip — otherwise
    /// `@param` slider drags don't take effect on static painted
    /// scenes. `None` until the first upload.
    pub last_uploaded_params: Option<[f32; 8]>,
    pub compute_g0: wgpu::BindGroup,
    pub raster_g1: wgpu::BindGroup,
    /// V1 stats; updated each frame.
    pub last_anchor_count: u32,
}

/// Per-frame mesh-path user-shader orchestration. Mutates state
/// (per-material cache, draw descriptors). The renderer consumes the
/// pending draw set in a later phase.
pub(super) fn tick_user_shader_mesh(state: &mut RenderState, frame: &RenderFrame) {
    // Reset the per-frame draw set before doing anything else — even
    // an empty frame should clear last frame's draws.
    state.user_shader_mesh_draws.clear();

    if frame.painted_anchors.is_empty() {
        state.last_uploaded_painted_anchors = None;
        return;
    }

    let source_hash = frame.user_shader_source_hash;

    // MAIN camera state for the per-anchor frustum + distance cull.
    // Fall back to the first viewport if MAIN isn't around (e.g. a
    // headless config that drives a non-MAIN VR only).
    let cam_vp = frame
        .viewports
        .iter()
        .find(|v| v.id == ViewportId::MAIN)
        .or_else(|| frame.viewports.first());
    let (cam_pos, frustum_planes, cur_vp_mat): (
        glam::Vec3,
        Option<[glam::Vec4; 6]>,
        Option<[[f32; 4]; 4]>,
    ) = match cam_vp {
        Some(vp) => {
            let p = vp.camera.position;
            let cam_pos = glam::Vec3::new(p[0], p[1], p[2]);
            let vp_mat = glam::Mat4::from_cols_array_2d(&vp.camera.view_proj);
            (cam_pos, Some(extract_frustum_planes(vp_mat)), Some(vp.camera.view_proj))
        }
        None => (glam::Vec3::ZERO, None, None),
    };

    // Fast-path detection: when the MAIN camera's view-projection
    // matches the previous frame's AND the painted-anchors Arc is the
    // same instance, the frustum+distance-filtered output is byte-
    // identical to last frame's upload. We can skip per-material
    // refilter + upload + 7 compute dispatches and still emit the
    // draw descriptors with cached pipelines. Params still gate
    // per-material via `params_changed` below — slider drags bypass
    // the fast-path for the affected material(s).
    let camera_unchanged = match (state.last_main_camera_vp, cur_vp_mat) {
        (Some(prev), Some(cur)) => prev == cur,
        _ => false,
    };
    let anchors_unchanged = state
        .last_uploaded_painted_anchors
        .as_ref()
        .map(|prev| std::sync::Arc::ptr_eq(prev, &frame.painted_anchors))
        .unwrap_or(false);
    let fast_path_eligible = camera_unchanged && anchors_unchanged;

    // Track the bookkeeping for next frame. Storing now is safe
    // because we only USE the comparison above this point.
    state.last_uploaded_painted_anchors = Some(std::sync::Arc::clone(&frame.painted_anchors));
    state.last_main_camera_vp = cur_vp_mat;

    // Collect (material_id, anchors) pairs up front so the per-material
    // loop can mutate state without aliasing frame.painted_anchors.
    let materials: Vec<(u16, &Vec<AnchorRecord>)> = frame
        .painted_anchors
        .iter()
        .map(|(k, v)| (*k, v))
        .collect();

    for (material_id, anchors) in materials {
        if anchors.is_empty() {
            continue;
        }

        // Look up which user shader this material routes to. Mesh-path
        // shaders use `instance_shader_id`; shade-only shaders set it
        // to 0 and we skip them here.
        let Some(material) = frame.materials.get(material_id as usize) else {
            continue;
        };
        let shader_id = material.instance_shader_id;
        if shader_id == 0 {
            continue;
        }

        // Find the parsed user-shader entry. Skip if the shader hasn't
        // opted into the mesh path (`@geometry` + `vs` + `spawn_count`).
        let Some(entry) = frame
            .user_shader_entries
            .iter()
            .find(|e| e.id == shader_id)
        else {
            continue;
        };
        if !entry.is_mesh_path() {
            continue;
        }

        let vertex_count_per_spawn = match &entry.metadata.mesh_geometry {
            Some(arvx_render::shader_composer::GeometryDecl::Procedural { vertex_count }) => {
                *vertex_count
            }
            Some(arvx_render::shader_composer::GeometryDecl::Mesh { .. }) => {
                // V1: mesh-source geometry not yet wired.
                eprintln!(
                    "[user-shader mesh] material {material_id}: @geometry mesh not implemented in V1; skipping"
                );
                continue;
            }
            None => continue,
        };

        // Get-or-build material state. On source-hash change, rebuild.
        let existing_hash = state
            .mesh_user_shader_cache
            .get(&material_id)
            .map(|s| s.source_hash);
        let just_built = existing_hash != Some(source_hash);
        if just_built {
            let new_state = build_material_state(state, entry, source_hash);
            state
                .mesh_user_shader_cache
                .insert(material_id, new_state);
        }

        let mat_state = state
            .mesh_user_shader_cache
            .get_mut(&material_id)
            .expect("just-inserted material state");

        // Always upload FrameUniforms — `frame.time` advances each
        // frame and the raster VS uses it for wind animation. Cheap
        // 48-byte write.
        let frame_uniforms = FrameUniforms {
            time: frame.shade_params_base.time,
            delta_time: 0.0,
            _pad0: [0.0; 2],
            wind_dir: [0.0, 0.0, 1.0],
            wind_strength: 1.0,
            camera_pos: [cam_pos.x, cam_pos.y, cam_pos.z],
            _pad1: 0.0,
        };
        state.queue.write_buffer(
            &mat_state.frame_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Per-material params change check — slider drags don't bump
        // paint_epoch but DO change spawn_count behavior. Forces the
        // refilter + recompute path for the affected material.
        let params = frame
            .shader_params_slots
            .get(material_id as usize)
            .copied()
            .unwrap_or([0.0; 8]);
        let params_changed = mat_state
            .last_uploaded_params
            .map(|p| p != params)
            .unwrap_or(true);

        // Fast-path: when MAIN camera VP, painted_anchors Arc, params,
        // and shader source are all unchanged, last frame's compute
        // outputs (records, indirect args, offsets) are still valid on
        // the GPU — skip the per-material refilter + 7 dispatches and
        // re-emit the same draw descriptor. Saves ~5-10 ms/frame at
        // the V1 cap for a steady-camera grass scene; the regression
        // was that the previous "anchors_unchanged" gate didn't hold
        // when the filter became per-frame.
        let needs_work =
            !fast_path_eligible || params_changed || just_built || mat_state.last_anchor_count == 0;

        if !needs_work {
            // Skip refilter + upload + compute. Still emit the draw
            // descriptor below using the cached indirect_buffer +
            // pipelines (its instance_count was written last frame).
        } else {
            // Look up the per-shader max_distance from the
            // user_shader_infos snapshot. None ⇒ no distance cull
            // (frustum cull still applies).
            let max_distance: Option<f32> = entry.metadata.max_distance;
            let filtered = filter_anchors(
                anchors,
                cam_pos,
                frustum_planes.as_ref(),
                max_distance,
            );
            if filtered.is_empty() {
                mat_state.last_anchor_count = 0;
                continue;
            }
            if filtered.len() > MAX_ANCHORS_PER_SHADER_V1 as usize {
                eprintln!(
                    "[user-shader mesh] material {material_id}: {} anchors (post-cull) \
                     exceeds V1 cap ({MAX_ANCHORS_PER_SHADER_V1}); clamping to cap",
                    filtered.len(),
                );
            }
            let anchor_count = (filtered.len() as u32).min(MAX_ANCHORS_PER_SHADER_V1);
            let anchors_ref: &[AnchorRecord] = &filtered;

            let upload_slice = &anchors_ref[..anchor_count as usize];
            state.queue.write_buffer(
                &mat_state.anchors_buffer,
                0,
                bytemuck::cast_slice(upload_slice),
            );

            if params_changed {
                let params_uniform = UserShaderParams { p: params };
                state.queue.write_buffer(
                    &mat_state.params_buffer,
                    0,
                    bytemuck::bytes_of(&params_uniform),
                );
                mat_state.last_uploaded_params = Some(params);
            }

            // DispatchInfo — current anchor count + verts-per-spawn.
            let dispatch_info = DispatchInfo {
                num_anchors: anchor_count,
                verts_per_spawn: vertex_count_per_spawn,
                _pad0: 0,
                _pad1: 0,
            };
            state.queue.write_buffer(
                &mat_state.dispatch_buffer,
                0,
                bytemuck::bytes_of(&dispatch_info),
            );

            mat_state.last_anchor_count = anchor_count;

            // Encode + submit the compute pipeline. Each pipeline
            // stage runs in its own compute pass so wgpu's
            // begin/end-pass implicit barriers strictly serialize
            // the chain (spawn_count → prefix_local → prefix_scan_sums
            // → prefix_add_back → fill). A single multi-dispatch
            // pass *should* auto-barrier on the read-after-write
            // hazards between these stages, but explicit
            // pass-per-stage ordering avoids any implementation
            // ambiguity.
            let mut encoder = state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!(
                        "user_shader_mesh material {material_id} compute"
                    )),
                });
            // Scene bind group lands at group(1) so `paint_probe` from
            // `spawn_alive` can descend the host octree. The bind group
            // is identical across VRs except for the camera at binding
            // 3 (which compute doesn't read); MAIN's is canonical.
            let scene_bg = state
                .viewport_renderers
                .get(&crate::viewport::ViewportId::MAIN)
                .or_else(|| state.viewport_renderers.values().next())
                .map(|vr| vr.scene_bind_group.clone());
            let Some(scene_bg) = scene_bg else {
                // No VR yet — skip the compute trio this frame; the
                // anchors haven't moved, so the indirect draw will
                // re-use stale results (one-frame visual flicker is
                // acceptable here since this path only fires at startup
                // before the first VR exists).
                continue;
            };
            let wg_x_64 = anchor_count.div_ceil(64).max(1);
            let wg_x_256 = anchor_count.div_ceil(256).max(1);
            // Tier-2 covers wg_sums in 256-wide chunks. With at most
            // `PREFIX_SUM_MAX_WG_COUNT` tier-1 WGs alive, only the
            // chunks that contain a live tier-1 WG need to fire.
            let live_tier1_wgs = wg_x_256.min(PREFIX_SUM_MAX_WG_COUNT);
            let wg_x_tier2 = live_tier1_wgs.div_ceil(256).max(1).min(PREFIX_SUM_MAX_WG_COUNT_2);
            for (label, pipeline, wgs) in [
                ("spawn_count",      &mat_state.pipelines.spawn_count,      wg_x_64),
                ("prefix_local",     &mat_state.pipelines.prefix_local,     wg_x_256),
                ("prefix_local2",    &mat_state.pipelines.prefix_local2,    wg_x_tier2),
                ("prefix_scan_sums", &mat_state.pipelines.prefix_scan_sums, 1),
                ("prefix_add_back2", &mat_state.pipelines.prefix_add_back2, wg_x_tier2),
                ("prefix_add_back",  &mat_state.pipelines.prefix_add_back,  wg_x_256),
                ("fill",             &mat_state.pipelines.fill,             wg_x_64),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &mat_state.compute_g0, &[]);
                pass.set_bind_group(1, &scene_bg, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            state.queue.submit(Some(encoder.finish()));
        }

        // Enqueue draw descriptor. The shadow pipeline shares
        // raster_g1 (anchors + records + frame + params) — the
        // engine binds a per-cascade shadow_g0 separately.
        state.user_shader_mesh_draws.push(UserShaderMeshDraw {
            material_id,
            shader_id,
            vertex_count_per_spawn,
            raster_pipeline: mat_state.pipelines.raster.clone(),
            shadow_pipeline: mat_state.pipelines.shadow.clone(),
            raster_g1: mat_state.raster_g1.clone(),
            indirect_buffer: mat_state.indirect_buffer.clone(),
        });
    }
}

/// Build per-material state from scratch: compose WGSL, build
/// pipelines, allocate buffers, build bind groups.
fn build_material_state(
    state: &RenderState,
    entry: &arvx_render::shader_composer::UserShaderEntry,
    source_hash: u64,
) -> MeshUserShaderMaterialState {
    let (raster_template, compute_template, shadow_template) =
        arvx_render::user_shader_mesh_pass::UserShaderMeshPass::template_sources();
    let (raster_wgsl, compute_wgsl, shadow_wgsl) =
        arvx_render::shader_composer::compose_mesh_path_pipeline_sources(
            entry,
            raster_template,
            compute_template,
            shadow_template,
        );

    let label = format!("user_shader_mesh:{}", entry.name);
    let pipelines = state.renderer.user_shader_mesh.build_pipelines(
        &state.device,
        &raster_wgsl,
        &compute_wgsl,
        &shadow_wgsl,
        &label,
    );

    let anchors_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} anchors")),
        size: (MAX_ANCHORS_PER_SHADER_V1 as u64) * std::mem::size_of::<AnchorRecord>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let counts_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} counts")),
        size: (MAX_ANCHORS_PER_SHADER_V1 as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let offsets_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} offsets")),
        size: (MAX_ANCHORS_PER_SHADER_V1 as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let records_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} records")),
        size: (MAX_SPAWNS_PER_SHADER_V1 as u64) * std::mem::size_of::<InstanceRecord>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let indirect_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} indirect")),
        size: std::mem::size_of::<DrawIndirectArgs>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let frame_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} frame")),
        size: std::mem::size_of::<FrameUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} params")),
        size: std::mem::size_of::<UserShaderParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let dispatch_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} dispatch")),
        size: std::mem::size_of::<DispatchInfo>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Tier-1 prefix-sum scratch: one slot per tier-1 workgroup
    // (256 anchors each, 1024 WGs at the V1 cap).
    let wg_sums_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} wg_sums")),
        size: (PREFIX_SUM_MAX_WG_COUNT as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // Tier-2 prefix-sum scratch: one slot per tier-2 workgroup (each
    // scans a 256-entry chunk of `wg_sums`, so 4 slots at the V1 cap).
    // Sized to 16 B minimum to clear `min_storage_buffer_offset_alignment`.
    let wg_sums2_size = ((PREFIX_SUM_MAX_WG_COUNT_2 as u64) * 4).max(16);
    let wg_sums2_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} wg_sums2")),
        size: wg_sums2_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let compute_g0 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label} compute g0")),
        layout: &state.renderer.user_shader_mesh.compute_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: anchors_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: counts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: offsets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: records_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: indirect_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: frame_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: dispatch_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: wg_sums_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: wg_sums2_buffer.as_entire_binding() },
        ],
    });

    let raster_g1 = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label} raster g1")),
        layout: &state.renderer.user_shader_mesh.raster_g1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: anchors_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: records_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: frame_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
        ],
    });

    MeshUserShaderMaterialState {
        source_hash,
        pipelines,
        anchors_buffer,
        counts_buffer,
        offsets_buffer,
        records_buffer,
        indirect_buffer,
        frame_buffer,
        params_buffer,
        dispatch_buffer,
        wg_sums_buffer,
        wg_sums2_buffer,
        last_uploaded_params: None,
        compute_g0,
        raster_g1,
        last_anchor_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor_at(tile_min: [f32; 3], tile_max: [f32; 3]) -> AnchorRecord {
        AnchorRecord {
            tile_min,
            material_id: 0,
            tile_max,
            leaf_count: 1,
            paint_min: tile_min,
            object_id: 0,
            paint_max: tile_max,
            surface_y: 0.0,
            surface_normal: [0.0, 1.0, 0.0],
            seed: 0,
            paint_mask: 0xFFFF,
            _pad: [0; 3],
        }
    }

    #[test]
    fn distance_cull_drops_far_anchors() {
        let near = anchor_at([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mid  = anchor_at([99.5, 0.0, -0.5], [100.5, 1.0, 0.5]);
        let far  = anchor_at([199.5, 0.0, -0.5], [200.5, 1.0, 0.5]);
        let anchors = vec![near, mid, far];
        let out = filter_anchors(
            &anchors,
            glam::Vec3::ZERO,
            None,           // no frustum
            Some(150.0),    // max_distance
        );
        // near (≈0.87 m) + mid (≈100 m) survive; far (≈200 m) is dropped.
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|a| a.tile_min[0] == 0.0));
        assert!(out.iter().any(|a| a.tile_min[0] == 99.5));
    }

    #[test]
    fn frustum_cull_drops_anchors_behind_camera() {
        // Camera at origin looking down -Z, fov 90°, aspect 1, near
        // 0.1, far 100. RH+z=[0,1].
        let view = glam::Mat4::look_to_rh(
            glam::Vec3::ZERO,
            -glam::Vec3::Z,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(
            90.0_f32.to_radians(),
            1.0,
            0.1,
            100.0,
        );
        let planes = extract_frustum_planes(proj * view);

        let in_front = anchor_at([-0.5, -0.5, -10.5], [0.5, 0.5, -9.5]);
        let behind   = anchor_at([-0.5, -0.5,   9.5], [0.5, 0.5, 10.5]);
        let anchors = vec![in_front, behind];
        let out = filter_anchors(
            &anchors,
            glam::Vec3::ZERO,
            Some(&planes),
            None,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tile_min[2], -10.5);
    }

    #[test]
    fn no_filters_returns_input_copy() {
        let a = anchor_at([0.0; 3], [1.0; 3]);
        let out = filter_anchors(&[a], glam::Vec3::ZERO, None, None);
        assert_eq!(out.len(), 1);
    }

    /// Pin the V1 per-anchor spawn cap at 32. The records buffer is
    /// sized `MAX_ANCHORS × MAX_SPAWNS_PER_ANCHOR_V1`, so silently
    /// raising this without resizing the buffer would walk off the
    /// end during `entry_fill`. The WESL also hardcodes the matching
    /// `MAX_SPAWNS_PER_ANCHOR_V1 = 32u`; if you change one, the
    /// composer's naga-validated test will fail before runtime.
    #[test]
    fn max_spawns_per_anchor_v1_is_32() {
        assert_eq!(MAX_SPAWNS_PER_ANCHOR_V1, 32);
        // Sanity: records buffer total = MAX_ANCHORS × MAX_SPAWNS,
        // and the prefix sum's `instance_count` cannot exceed it.
        assert_eq!(
            MAX_SPAWNS_PER_SHADER_V1 as u64,
            MAX_ANCHORS_PER_SHADER_V1 as u64 * 32,
        );
    }
}
