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
    MAX_ANCHORS_PER_SHADER_V1,
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
        return;
    }

    let source_hash = frame.user_shader_source_hash;

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

        // Upload anchor records (truncated to V1 cap).
        let upload_slice = &anchors[..anchor_count as usize];
        state.queue.write_buffer(
            &mat_state.anchors_buffer,
            0,
            bytemuck::cast_slice(upload_slice),
        );

        // FrameUniforms — time/wind/camera. V1 fills time from the
        // shade params; camera_pos + delta_time + wind are zeroed for
        // V1 (subscribed-uniforms wiring is a follow-up).
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

        // Params — copy from the per-material shader_params_slots entry.
        let params = frame
            .shader_params_slots
            .get(material_id as usize)
            .copied()
            .unwrap_or([0.0; 8]);
        let params_uniform = UserShaderParams { p: params };
        state.queue.write_buffer(
            &mat_state.params_buffer,
            0,
            bytemuck::bytes_of(&params_uniform),
        );

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

        // Encode + submit the three compute passes.
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!(
                    "user_shader_mesh material {material_id} compute"
                )),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("user_shader_mesh compute"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &mat_state.compute_g0, &[]);

            // 1. spawn_count — 1 thread per anchor.
            pass.set_pipeline(&mat_state.pipelines.spawn_count);
            let wg_x = anchor_count.div_ceil(64).max(1);
            pass.dispatch_workgroups(wg_x, 1, 1);

            // 2. prefix_sum — single workgroup of 1024 threads.
            pass.set_pipeline(&mat_state.pipelines.prefix_sum);
            pass.dispatch_workgroups(1, 1, 1);

            // 3. fill — 1 thread per anchor.
            pass.set_pipeline(&mat_state.pipelines.fill);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }
        state.queue.submit(Some(encoder.finish()));

        // Enqueue draw descriptor.
        state.user_shader_mesh_draws.push(UserShaderMeshDraw {
            material_id,
            shader_id,
            vertex_count_per_spawn,
            raster_pipeline: mat_state.pipelines.raster.clone(),
            raster_g1: mat_state.raster_g1.clone(),
            indirect_buffer: mat_state.indirect_buffer.clone(),
        });
    }
}

/// Build per-material state from scratch: compose WGSL, build
/// pipelines, allocate buffers, build bind groups.
fn build_material_state(
    state: &RenderState,
    entry: &rkp_render::shader_composer::UserShaderEntry,
    source_hash: u64,
) -> MeshUserShaderMaterialState {
    let (raster_template, compute_template) =
        rkp_render::user_shader_mesh_pass::UserShaderMeshPass::template_sources();
    let (raster_wgsl, compute_wgsl) =
        rkp_render::shader_composer::compose_mesh_path_pipeline_sources(
            entry,
            raster_template,
            compute_template,
        );

    let label = format!("user_shader_mesh:{}", entry.name);
    let pipelines = state.renderer.user_shader_mesh.build_pipelines(
        &state.device,
        &raster_wgsl,
        &compute_wgsl,
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
        compute_g0,
        raster_g1,
        last_anchor_count: 0,
    }
}
