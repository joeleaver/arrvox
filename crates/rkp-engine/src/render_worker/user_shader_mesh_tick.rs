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
//!      uniforms. V1 caps at 1024 anchors × 64 spawns/anchor.
//!   3. Uploads the per-frame data: anchor records, FrameUniforms,
//!      UserShaderParams, DispatchInfo.
//!   4. Dispatches the three compute passes: spawn_count → prefix_sum
//!      → fill. The fill output drives an indirect draw the renderer
//!      consumes in a later pass (currently a stub — task #7).
//!
//! Static-cache gating is wired but not yet honored — V1 always
//! re-runs the compute trio. Caching by (paint_epoch + geometry_epoch
//! + params hash) is a follow-up.

use rkp_render::user_shader_mesh_pass::{
    AnchorRecord, DispatchInfo, DrawIndirectArgs, FrameUniforms, InstanceRecord,
    UserShaderMeshDraw, UserShaderMeshPipelines, UserShaderParams,
    MAX_ANCHORS_PER_SHADER_V1, PREFIX_SUM_MAX_WG_COUNT,
};

use crate::render_frame::RenderFrame;

use super::state::RenderState;

/// Fixed V1 cap on per-anchor spawn count × anchor count. With
/// `MAX_ANCHORS_PER_SHADER_V1 = 1024` and a 64-spawn/anchor cap, the
/// records buffer is `1024 × 64 × 8 B = 512 KB` per material.
pub const MAX_SPAWNS_PER_ANCHOR_V1: u32 = 64;
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

    // Skip per-material anchor upload + compute trio when the painted
    // anchor set is byte-identical to last frame (`Arc::ptr_eq` —
    // sim swaps the inner `Arc` only on paint/geometry/param epoch
    // rebuild). Steady-state idle frames pay only the per-frame
    // uniform write + draw-descriptor emit, not 5 compute dispatches
    // per material. Big CPU win for static painted scenes.
    let anchors_unchanged = state
        .last_uploaded_painted_anchors
        .as_ref()
        .map(|prev| std::sync::Arc::ptr_eq(prev, &frame.painted_anchors))
        .unwrap_or(false);

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
        if anchors.len() > MAX_ANCHORS_PER_SHADER_V1 as usize {
            eprintln!(
                "[user-shader mesh] material {material_id}: {} anchors exceeds V1 cap \
                 ({MAX_ANCHORS_PER_SHADER_V1}); clamping to cap (excess will be skipped)",
                anchors.len(),
            );
        }
        let anchor_count = (anchors.len() as u32).min(MAX_ANCHORS_PER_SHADER_V1);

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
            Some(rkp_render::shader_composer::GeometryDecl::Procedural { vertex_count }) => {
                *vertex_count
            }
            Some(rkp_render::shader_composer::GeometryDecl::Mesh { .. }) => {
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
        if existing_hash != Some(source_hash) {
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
            camera_pos: [0.0; 3],
            _pad1: 0.0,
        };
        state.queue.write_buffer(
            &mat_state.frame_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        // Per-material params change check — slider drags don't bump
        // paint_epoch but DO change spawn_count behavior, so we have
        // to re-run the compute trio on params change too. Without
        // this, `anchors_unchanged` would freeze the cached spawn
        // counts and the slider would be a no-op on a static painted
        // scene.
        let params = frame
            .shader_params_slots
            .get(material_id as usize)
            .copied()
            .unwrap_or([0.0; 8]);
        let params_changed = mat_state
            .last_uploaded_params
            .map(|p| p != params)
            .unwrap_or(true);

        // Anchor data + params + dispatch + compute trio only run
        // when the painted-anchor set OR the per-material params
        // changed. Steady-state idle → only the FrameUniforms write
        // above + the draw emit below run.
        if !anchors_unchanged || params_changed {
            if !anchors_unchanged {
                let upload_slice = &anchors[..anchor_count as usize];
                state.queue.write_buffer(
                    &mat_state.anchors_buffer,
                    0,
                    bytemuck::cast_slice(upload_slice),
                );
            }

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
            for (label, pipeline, wgs) in [
                ("spawn_count", &mat_state.pipelines.spawn_count, wg_x_64),
                ("prefix_local", &mat_state.pipelines.prefix_local, wg_x_256),
                ("prefix_scan_sums", &mat_state.pipelines.prefix_scan_sums, 1),
                ("prefix_add_back", &mat_state.pipelines.prefix_add_back, wg_x_256),
                ("fill", &mat_state.pipelines.fill, wg_x_64),
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

    if !anchors_unchanged {
        state.last_uploaded_painted_anchors =
            Some(std::sync::Arc::clone(&frame.painted_anchors));
    }
}

/// Build per-material state from scratch: compose WGSL, build
/// pipelines, allocate buffers, build bind groups.
fn build_material_state(
    state: &RenderState,
    entry: &rkp_render::shader_composer::UserShaderEntry,
    source_hash: u64,
) -> MeshUserShaderMaterialState {
    let (raster_template, compute_template, shadow_template) =
        rkp_render::user_shader_mesh_pass::UserShaderMeshPass::template_sources();
    let (raster_wgsl, compute_wgsl, shadow_wgsl) =
        rkp_render::shader_composer::compose_mesh_path_pipeline_sources(
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
    // Prefix-sum scratch: one slot per workgroup (256 anchors each).
    let wg_sums_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} wg_sums")),
        size: (PREFIX_SUM_MAX_WG_COUNT as u64) * 4,
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
        last_uploaded_params: None,
        compute_g0,
        raster_g1,
        last_anchor_count: 0,
    }
}
