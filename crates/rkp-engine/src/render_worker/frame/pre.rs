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
use super::super::user_shader_tick::{run_user_shader_geom, tick_instance_pipeline};

use super::PreFrameOutput;

pub(super) fn run_pre_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
) -> PreFrameOutput {
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
        // Host march + shadow trace splice the user-shader instance_at
        // chunk (the band-cell descent hook). Hash gate inside each
        // `reload` makes the no-change frame a no-op.
        vr.march.reload_user_shaders(
            &state.device,
            &frame.user_shader_instance_at_chunk,
            frame.user_shader_source_hash,
        );
        // Phase 4 — shadow trace splices the same instance_at chunk
        // the primary march does so band cells dispatch the same
        // user-shader prototype descent into the shadow path.
        vr.shadow_trace.reload_user_shaders(
            &state.device,
            &frame.user_shader_instance_at_chunk,
            frame.user_shader_source_hash,
        );
        // Phase 4 — shadow-map scatter splices the same instance_at
        // chunk so band cells fire the user-shader instance descent
        // for directional-light shadows too.
        vr.shadow_map.reload_user_shaders(
            &state.device,
            &frame.user_shader_instance_at_chunk,
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

    // 1. Geometry upload — epoch-driven. Robust to snapshot drops:
    //    sim ships scene_mgr's current epoch every frame, so we'll
    //    catch up on the next snapshot if an intermediate one was
    //    dropped by the newest-wins inbox.
    if frame.geometry_epoch > state.last_uploaded_geometry_epoch {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let geo = sm.geometry_upload();
        state.renderer.upload_geometry(&state.queue, &geo);
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

    // 1.6. Brush-overlay upload — paint cursor geodesic distances.
    //      MAIN-only (BUILD viewport doesn't show the paint cursor).
    //      queue.write_buffer is cheap (staging-buffer enqueue), so
    //      we do it inside the scene_mgr lock — cheaper than cloning
    //      the full ~4 MB overlay buffer out of the critical section.
    if frame.brush_overlay_epoch > state.last_uploaded_brush_overlay_epoch {
        let sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let bytes = sm.brush_overlay_bytes();
        if let Some(main_vr) = state.viewport_renderers.get_mut(&ViewportId::MAIN) {
            main_vr.shade.upload_brush_overlay(&state.device, &state.queue, bytes);
        }
        state.last_uploaded_brush_overlay_epoch = sm.brush_overlay_epoch();
        drop(sm);
    }

    // 1.7. Option B (instance pipeline) per-frame tick. Stage 6c-3.5c
    //      reserves the proto sub-pool past the user-shader-cache tail,
    //      walks `instance_region_requests`, dispatches bake/scatter,
    //      and uploads TileIndex + ProtoLookup. **Runs BEFORE
    //      run_user_shader_geom** so the proto tail's buffer reservation
    //      stacks correctly: tick reserves the full
    //      `cpu + user_shader_max + proto_max` envelope when there's
    //      any instance work, then run_user_shader_geom's own reservation
    //      is a no-op (buffer already big enough).
    let inst_result = tick_instance_pipeline(state, frame);

    // 1.7b. User-shader geometry pass (Phase C). Reserve transient
    //       pool tail, reload pipeline if the shader source changed,
    //       walk regions, dispatch the geom-build pipeline for any
    //       that need a re-bake, and concatenate the resulting
    //       transient (asset, instance) pairs onto the persistent
    //       lists so the per-frame upload below ships them alongside.
    //       Each transient region is its own asset slot — assigned
    //       via `asset_id_base` so it points into the correct slot in
    //       the combined assets vec. The march/shade passes treat
    //       transients identically to bake-built objects — same
    //       octree node encoding, same leaf attr layout.
    //
    //       Phase 4 — user-shader instance assets/instances from
    //       `tick_instance_pipeline` go BETWEEN the host's persistent
    //       set and Phase C's transient set. Phase C's `asset_id_base`
    //       shifts up to account for them so its per-instance
    //       `asset_id` references stay correct after splicing.
    let asset_id_base =
        frame.gpu_assets.len() as u32 + inst_result.len() as u32;
    let (transient_assets, transient_instances) =
        run_user_shader_geom(state, frame, asset_id_base);

    let mut combined_assets: Vec<rkp_render::rkp_gpu_object::RkpGpuAsset>;
    let mut combined_instances: Vec<rkp_render::rkp_gpu_object::RkpGpuInstance>;
    // Phase B-redux — `inst_result` is the per-shader instance-prototype
    // asset list. No host instances reference these directly; the band-
    // cell descent path resolves them by `shader_id` linear scan when a
    // band hit fires `descend_proto_octree`. Splice into the assets vec
    // so that scan finds them.
    let need_combine = !inst_result.is_empty() || !transient_instances.is_empty();
    let (assets_for_upload, instances_for_upload): (
        &[rkp_render::rkp_gpu_object::RkpGpuAsset],
        &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    ) = if !need_combine {
        (frame.gpu_assets.as_slice(), gpu_instances)
    } else {
        combined_assets = Vec::with_capacity(
            frame.gpu_assets.len()
                + inst_result.len()
                + transient_assets.len(),
        );
        combined_assets.extend_from_slice(&frame.gpu_assets);
        combined_assets.extend_from_slice(&inst_result);
        combined_assets.extend_from_slice(&transient_assets);
        combined_instances = Vec::with_capacity(
            gpu_instances.len() + transient_instances.len(),
        );
        combined_instances.extend_from_slice(gpu_instances);
        combined_instances.extend_from_slice(&transient_instances);
        (combined_assets.as_slice(), combined_instances.as_slice())
    };

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

    // Phase 8 — directional shadow map. Picks the first
    // directional light, derives the light camera covering the
    // scene AABB, writes the uniform into every VR. Returns
    // whether the shadow map will be live this frame; the shade
    // pass gates its sample on that. Texture dispatch happens
    // later in `render_to`.
    let shadow_map_enabled = prepare_shadow_maps(state, frame, scene_aabb, tlas_prim_count);

    // 2. Skin scatter (one batched compute dispatch). Sim folded every
    //    skinned entity into `frame.skin.batch`; we just fire it.
    if let Some(skin) = &frame.skin {
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

    // Phase 6 Session 4d — user-shader instances no longer appear in
    // the host instances buffer. Layout is now:
    //   [persistent | transient]
    // and the host march iterates `us_tile_entries[]` for user-shader
    // work (Sessions 1–3 + 4b). The per-VR `compute_screen_aabbs` /
    // `build_tile_lists` for user-shader instances + the
    // `user_shader_tile_lists_per_vp` Vec are gone.
    let transient_count = transient_instances.len() as u32;
    let persistent_count = gpu_instances.len() as u32;
    let object_count = persistent_count + transient_count;
    let transient_indices: Vec<u32> =
        (persistent_count..persistent_count + transient_count).collect();

    PreFrameOutput {
        transient_indices,
        object_count,
        asset_count,
        scene_aabb,
        shadow_map_enabled,
    }
}
