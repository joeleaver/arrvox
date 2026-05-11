//! Pre-frame setup: phases 0 (uploads + user-shader registry resync),
//! 1 (geometry/instance/TLAS upload + shadow setup), 2 (skin scatter).
//!
//! Returns [`super::PreFrameOutput`] carrying the cross-phase state
//! [`super::encode::encode_viewports`] needs (transient indices, scene
//! AABB, shadow-map-enabled flag, combined asset count).

use rkp_render::rkp_scene::FrameUpload;

use crate::render_frame::RenderFrame;
use crate::viewport::ViewportId;

use super::super::frame_helpers::{compute_tlas_scene_aabb, prepare_shadow_maps};
use super::super::state::RenderState;
use super::super::user_shader_mesh_tick::tick_user_shader_mesh;
use super::super::user_shader_tick::{tick_emit_pass, tick_instance_pipeline};

use super::PreFrameOutput;

pub(super) fn run_pre_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
) -> PreFrameOutput {
    // Sub-phase timing inside pre, gated on `RKP_RENDER_PROFILE=1`.
    // Splits the `pre` bucket of `[render.frame]` into the major
    // upload/dispatch groups so we can attribute the cost.
    let pre_profile = std::env::var("RKP_RENDER_PROFILE").is_ok();
    let pre_start = std::time::Instant::now();

    // 0. Drive the wgpu async runtime so any in-flight async maps can
    //    complete (volumetric sun-atten readbacks, frame readbacks,
    //    pick readbacks).
    let _ = state.device.poll(wgpu::PollType::Poll);

    // 0a. Material palette upload — every frame (cheap; ~1 KB).
    state
        .renderer
        .update_materials(&state.queue, &frame.materials);

    // 0b. Lights upload — sim hands us the full list each tick
    //     (entry 0 = sun, 1..N = scene point/spot lights).
    state.renderer.update_lights(&state.queue, &frame.lights);

    // 0c. Environment-driven bloom + tonemap settings — every frame.
    //     Walk every viewport renderer (each VR owns its own bloom +
    //     tonemap pass; no per-VR override today). Each set_* is one
    //     small queue.write_buffer.
    let env = frame.env_update;
    let vr_ids: Vec<_> = state.viewport_renderers.keys().copied().collect();
    for vr_id in &vr_ids {
        let vr = state
            .viewport_renderers
            .get_mut(vr_id)
            .expect("viewport renderer must exist");
        vr.tone_map.set_exposure(&state.queue, env.exposure);
        vr.bloom
            .set_threshold(&state.queue, env.bloom_threshold, env.bloom_knee);
        vr.bloom_composite
            .set_intensity(&state.queue, env.bloom_intensity);
    }

    // 0d. User-shader integration. Each viewport's shade pass owns
    //     its own pipeline + per-material params buffer. Recompile
    //     when the registry's source hash changes (idempotent). Upload
    //     params alongside the materials buffer; if the buffer grew,
    //     the bind group gets cleared and we rebuild it via
    //     set_shade_data. Uploading on every viewport repeats the
    //     queue.write_buffer (cost: 32 B × num_materials) but keeps
    //     bind-group lifetimes simple.
    for vr_id in &vr_ids {
        let vr = state
            .viewport_renderers
            .get_mut(vr_id)
            .expect("viewport renderer must exist");
        vr.shade.reload_user_shaders(
            &state.device,
            &frame.user_shader_shade_chunk,
            frame.user_shader_source_hash,
        );
        // Host march + shadow shaders no longer splice an `instance_at`
        // descent chunk (the band-cell path is gone). Pass an empty
        // chunk so the splice is a no-op for unchanged source. Once
        // the new emit pass lands, reloading happens on the emit pass
        // itself, not on the consumer templates.
        vr.march.reload_user_shaders(
            &state.device,
            "",
            frame.user_shader_source_hash,
        );
        vr.shadow_trace.reload_user_shaders(
            &state.device,
            "",
            frame.user_shader_source_hash,
        );
        vr.shadow_map.reload_user_shaders(
            &state.device,
            "",
            frame.user_shader_source_hash,
        );
        vr.shade.upload_shader_params(
            &state.device,
            &state.queue,
            &frame.shader_params_slots,
        );
        // Phase B-redux Phase 3a — refresh march's binding too.
        // upload_shader_params may have reallocated the buffer; the
        // existing per-frame `set_shade_data` below picks up shade's
        // side, march mirrors that here.
        vr.march.set_shader_params(
            &state.device,
            vr.shade.shader_params_buffer(),
        );
        // Phase 4 — shadow trace shares the params bind group that
        // OctreeMarchPass owns (binding 10 already lives there), but
        // we still record the buffer handle on the shadow pass for
        // API symmetry with the rest of the chain.
        vr.shadow_trace.set_shader_params(
            &state.device,
            vr.shade.shader_params_buffer(),
        );
        // Phase 4 — shadow-map scatter has its own scatter_pass_bg
        // (group 1) that needs materials + shader_params bound for
        // the band-cell dispatch. Re-wiring is cheap; the inner
        // try-rebuild short-circuits when both handles + the
        // scatter_instances buffer are stable.
        vr.shadow_map.set_materials(
            &state.device,
            &state.renderer.materials_buffer,
        );
        vr.shadow_map.set_shader_params(
            &state.device,
            vr.shade.shader_params_buffer(),
        );
        // Re-bind unconditionally — set_shade_data is one bind-group
        // create; cheaper than threading state for "did the buffer
        // grow" through callers, and the materials/lights buffers may
        // have just been swapped above too.
        vr.shade.set_shade_data(
            &state.device,
            &state.renderer.shade_params_buffer,
            &state.renderer.lights_buffer,
            &state.renderer.materials_buffer,
        );
    }

    let p_t_setup = pre_start.elapsed();

    // 1. Geometry upload — epoch-driven. Robust to snapshot drops:
    //    sim ships scene_mgr's current epoch every frame, so we'll
    //    catch up on the next snapshot if an intermediate one was
    //    dropped by the newest-wins inbox.
    if frame.geometry_epoch > state.last_uploaded_geometry_epoch {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let geo = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &geo);
        // Phase B-2 — keep the splat-raster per-asset vertex-buffer
        // cache in step with the loaded asset set. Re-upload all of
        // them on every geometry-epoch bump rather than tracking which
        // assets actually changed; matches the upload_geometry "build
        // everything from scratch" pattern, and the cost is bounded by
        // total splat count (rare, off-the-hot-path).
        for (handle, splats) in sm.iter_loaded_asset_splats() {
            state.renderer.upload_splats_for_asset(handle.raw(), splats);
        }
        // Phase 2 (splat-to-mesh pivot) — same logic for the mesh
        // path's per-asset (vbo, ibo) cache. iter_loaded_asset_meshes
        // skips empty mesh extractions (procedurals etc.) so this only
        // touches assets that produced a non-empty surface mesh at
        // load time. Phase 6.1: indices is the full DAG IBO (LOD-0
        // first, then LOD-1, ...); dispatch draws only the LOD-0
        // prefix until Phase 6.2 wires the indirect path.
        for (handle, vertices, indices, lod0_index_count) in sm.iter_loaded_asset_meshes() {
            state.renderer.upload_mesh_for_asset(
                handle.raw(),
                vertices,
                indices,
                lod0_index_count,
            );
        }
        // Phase 5 — per-asset meshlet cluster table. Storage buffer
        // for the Phase 6 LOD-selection compute pass; uploaded here
        // but unused by current dispatch (validates the upload path
        // without rewiring the hot draw call).
        for (handle, clusters) in sm.iter_loaded_asset_clusters() {
            state.renderer.upload_mesh_clusters_for_asset(handle.raw(), clusters);
        }
        // Read-back the epoch *under the same lock* so concurrent
        // mutations (bake worker integrating an artifact mid-frame)
        // don't trick us into thinking we're caught up when we're
        // not. Worst case: we re-upload next frame, which is fine.
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
    }

    // 1.5. Phase 3 — paint mutations land in per-instance overlays
    //      (`paint_overlays` on EngineState), shipped each tick as
    //      `frame.gpu_instance_overlays`. The upload happens
    //      unconditionally inside `upload_frame` below; no slot-range
    //      slice-upload of `leaf_attr_pool`/`color_pool` is needed.
    //      `frame.paint_epoch` is informational only.

    // (The geodesic paint-cursor overlay was retired in favor of the
    // screen-space cursor — its per-frame upload of
    // `brush_overlay_distances` is gone. The per-VR `BrushState`
    // buffer the new cursor reads is written by the GPU-side
    // brush-state probe pass each frame, so there's nothing to
    // upload from sim here.)

    let p_t_uploads = pre_start.elapsed();

    // 1.7. Per-frame proto bake. Bakes each registered instance
    //      shader's prototype octree (canonical [0,1]³) into the
    //      shared host pool tail; emit pass + march descend it as a
    //      regular asset. Returns one `RkpGpuAsset` per shader_id so
    //      emitted blade instances can reference it.
    let inst_result = tick_instance_pipeline(state, frame);

    // 1.7a. Upload `instance_overlay_buffer` for the per-instance
    //       paint cursor (Phase 3). Independent of the proto bake;
    //       just needs to land before any consumer reads.
    if !frame.gpu_instance_overlays.is_empty() {
        let overlay_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_overlays);
        state.renderer.scene.upload_instance_overlay(
            &state.device, &state.queue, overlay_bytes,
        );
    }

    let mut combined_assets: Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>;
    // Splice proto assets onto the host's persistent set. Emitted
    // blade instances (written by the user-shader emit pass below)
    // reference these by `asset_id` — the proto's absolute index in
    // the combined assets buffer is `frame.gpu_assets.len() + idx`.
    let proto_asset_id_base = frame.gpu_assets.len() as u32;
    let (assets_for_upload, instances_for_upload): (
        &[rkp_render::rkp_gpu_object::RkpGpuAsset],
        &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    ) = if inst_result.is_empty() {
        (frame.gpu_assets.as_slice(), gpu_instances)
    } else {
        combined_assets = Vec::with_capacity(
            frame.gpu_assets.len() + inst_result.len(),
        );
        combined_assets.extend_from_slice(&frame.gpu_assets);
        combined_assets.extend_from_slice(&inst_result);
        (combined_assets.as_slice(), gpu_instances)
    };

    let p_t_proto = pre_start.elapsed();

    // 1.7c. User-shader emit pass. Reads `frame.painted_leaves`,
    //       runs each shader's `instance_at` / `inst_world_matrix`
    //       hooks, writes `RkpInstance` records into the scene's
    //       user_shader_instance_buffer. The host march reads those
    //       records in Task #10 (currently the buffer is written but
    //       nothing consumes it; engine logs the count behind
    //       `RKP_MARCH_STATS=1` for verification).
    tick_emit_pass(state, frame, &inst_result, proto_asset_id_base);

    // 1.7d. V1 mesh-path user-shader orchestration. Reads
    //       `frame.painted_anchors`, dispatches per-material
    //       compute trio (spawn_count → prefix_sum → fill) for
    //       mesh-path shaders, and stages draw descriptors on
    //       `state.user_shader_mesh_draws`. The renderer consumes
    //       the draw set in the per-VR encode phase (task #7 — wired
    //       to nothing yet, so the compute trio runs but no raster
    //       draws happen).
    tick_user_shader_mesh(state, frame);

    let p_t_emit = pre_start.elapsed();

    // 1b. Per-frame upload. `gpu_instances` here may be interpolated
    //     between the last two sim snapshots (see `interpolate_instances`),
    //     so at high render rates physics-driven motion is smooth
    //     instead of stuttering at the sim rate. Assets are pose-static
    //     within a frame.
    // 1b'. Phase 7c — GPU TLAS build. The CPU build_tlas (median-
    //      split BVH) used `pos ± region_thickness` per-leaf AABBs
    //      for user-shader instances because it had no way to call
    //      the shader's `inst_aabb` hook. With grass-style shaders
    //      that's a 3 m cube around each painted leaf — 5000 leaves'
    //      AABBs all overlap, BVH degenerates, shadow trace tanks
    //      (30-40 ms for one .5 m grass splat).
    //
    //      Phase 7c reads the tight per-instance AABBs the Phase 6
    //      tile-cull AABB pass already wrote into
    //      `instance_tile_cull_scratch_buffer`. Pipeline:
    //        S1 — assemble (filter scratch.live + transform host
    //             AABBs) → packed `tlas_prims[]`.
    //        S2 — 30-bit Morton compute + 4×8-bit radix sort.
    //        S3 — Karras parallel binary tree topology.
    //        S4 — bottom-up AABB propagation via atomic visit
    //             counter.
    //      Output writes directly into `state.tlas_pass.{nodes,leaves}_buffer`,
    //      which the shadow trace already binds.
    //
    //      Note: must run AFTER `upload_frame` because the host
    //      assembly pass reads `state.renderer.scene.objects_buffer`
    //      / `assets_buffer`. (Tile-cull scratch was populated
    //      earlier by `tick_instance_pipeline`.)
    let scene_aabb = compute_tlas_scene_aabb(
        instances_for_upload,
        assets_for_upload,
    );
    let overlay_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_overlays);
    state.renderer.upload_frame(
        &state.queue,
        &FrameUpload {
            assets: assets_for_upload,
            instances: instances_for_upload,
            bone_matrices: &frame.bone_matrix_lbs,
            bone_dual_quats: &frame.bone_matrix_dqs,
            instance_overlays: overlay_bytes,
        },
    );

    let p_t_upload_frame = pre_start.elapsed();

    // 1b''. Phase 7c — fire the GPU TLAS build. Inputs:
    //   * tile-cull scratch buffer (populated earlier by
    //     `tick_instance_pipeline`)
    //   * scene's host-instance and host-asset buffers (just
    //     populated by `upload_frame`)
    //   * scene AABB (CPU-derived above)
    //   The pipeline writes the final `tlas_nodes` + `tlas_leaves`
    //   into `state.tlas_pass`'s buffers, which the shadow trace
    //   already binds. Returns the actual prim count after the
    //   live-filter; sets `tlas_pass.last_*_count` accordingly so
    //   the empty-scene skip in shadow trace works.
    let host_count = instances_for_upload.len() as u32;
    let asset_count = assets_for_upload.len() as u32;
    let tlas_inputs = rkp_render::tlas_build_pass::GpuTlasBuildInputs {
        instances_buffer: &state.renderer.scene.objects_buffer,
        instance_count: host_count,
        assets_buffer: &state.renderer.scene.assets_buffer,
        asset_count,
        scene_min: scene_aabb.0,
        scene_max: scene_aabb.1,
    };
    let tlas_prim_count = state.tlas_build_pass.build_gpu_tlas(
        &state.device,
        &state.queue,
        &tlas_inputs,
        &mut state.tlas_pass,
        Some(&state.renderer.profiler),
    );
    // Refresh per-VR shadow-trace bind groups so they pick up any
    // capacity-doubling reallocation of `tlas_pass.{nodes,leaves}_buffer`.
    // Cheap when handles match.
    for vr in state.viewport_renderers.values_mut() {
        vr.march.set_tlas_buffers(
            &state.device,
            &state.tlas_pass.nodes_buffer,
            &state.tlas_pass.leaves_buffer,
        );
        // Phase 8 — shadow-map setup pass reads `tlas_prims`
        // directly (no BVH walk needed; per-prim AABB → texel rect
        // is a flat scan). Grow scatter scratch if the prim count
        // jumps, then rebind the prims buffer.
        vr.shadow_map.ensure_scatter_capacity(&state.device, tlas_prim_count);
        vr.shadow_map.set_tlas_prims_buffer(
            &state.device,
            &state.tlas_build_pass.tlas_prims_buffer,
        );
        // Phase 4 — band-cell shadow dispatch reads `time` and
        // `asset_count` from a lite march_params uniform.
        // `asset_count` matches what the primary march sees
        // because both originate from the same `assets_for_upload`.
        vr.shadow_map.update_march_params(
            &state.queue,
            frame.shade_params_base.time,
            asset_count,
        );
    }

    let p_t_tlas = pre_start.elapsed();

    // Phase 8 — directional shadow map. Picks the first
    // directional light, derives the light camera covering the
    // scene AABB, writes the uniform into every VR. Returns
    // whether the shadow map will be live this frame; the shade
    // pass gates its sample on that. Texture dispatch happens
    // later in `render_to`.
    let shadow_map_enabled = prepare_shadow_maps(state, frame, scene_aabb, tlas_prim_count);

    let p_t_shadow_prepare = pre_start.elapsed();

    // 2. Skin scatter (one batched compute dispatch). Sim folded every
    //    skinned entity into `frame.skin.batch`; we just fire it.
    //
    // **Phase 6.6 gate:** the scatter writes the voxel-march bone
    // field that only the legacy march path samples. Mesh primary
    // does its skinning in the vertex shader against the same per-
    // frame `bone_matrices` / `bone_dual_quats` buffers (which the
    // sim still uploads via `RenderFrame::bone_matrix_lbs/dqs`), so
    // the bone-field scatter is dead weight in mesh mode — measured
    // at ~1.6 ms p50 on splat5 elephant pre-gate. The check stays
    // local to this dispatch so the legacy buffers + the matrix
    // upload upstream remain wired for any viewport that's still on
    // march (`RKP_PRIMARY=march` or default).
    let skip_skin_deform = matches!(
        state.renderer.primary_mode,
        rkp_render::rkp_renderer::PrimaryMode::Mesh,
    );
    if let Some(skin) = &frame.skin {
        if !skip_skin_deform {
            let mut skin_encoder = state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rkp_skin_deform_encoder"),
                });
            let q = state
                .renderer
                .profiler
                .begin_query("skin_deform", &mut skin_encoder);
            state.renderer.prepare_bone_field(
                &state.queue,
                &mut skin_encoder,
                skin.bone_field_bytes,
                skin.bone_field_occ_bytes,
            );
            state
                .renderer
                .scatter_skin_batch(&state.queue, &mut skin_encoder, &skin.batch);
            state.renderer.profiler.end_query(&mut skin_encoder, q);
            state.queue.submit(std::iter::once(skin_encoder.finish()));
        }
    }

    // The user-shader BFS path (Phase C) is gone — only persistent
    // host instances live in the buffer right now. Phase 9's emit
    // pass will append blade `RkpInstance`s here too; the layout
    // shape (persistent + transient) stays.
    let transient_indices: Vec<u32> = Vec::new();
    let object_count = gpu_instances.len() as u32;

    // Upper bound on emitted instances. The shader threshold-checks
    // against the actual count, so over-shooting is safe (just wastes
    // a few workgroups).
    const MAX_EMITS_PER_LEAF: u32 = 8;
    let user_shader_instance_count = (frame.painted_leaves.len() as u32)
        .saturating_mul(MAX_EMITS_PER_LEAF)
        .min(rkp_render::rkp_scene::USER_SHADER_INSTANCE_CAPACITY);

    let p_t_skin = pre_start.elapsed();
    if pre_profile {
        let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[render.pre] setup={:.2} uploads={:.2} proto={:.2} emit={:.2} upload_frame={:.2} tlas={:.2} shadow={:.2} skin={:.2} | total={:.2}",
            to_ms(p_t_setup),
            to_ms(p_t_uploads) - to_ms(p_t_setup),
            to_ms(p_t_proto) - to_ms(p_t_uploads),
            to_ms(p_t_emit) - to_ms(p_t_proto),
            to_ms(p_t_upload_frame) - to_ms(p_t_emit),
            to_ms(p_t_tlas) - to_ms(p_t_upload_frame),
            to_ms(p_t_shadow_prepare) - to_ms(p_t_tlas),
            to_ms(p_t_skin) - to_ms(p_t_shadow_prepare),
            to_ms(p_t_skin),
        );
    }
    PreFrameOutput {
        transient_indices,
        object_count,
        asset_count,
        scene_aabb,
        shadow_map_enabled,
        user_shader_instance_count,
    }
}
