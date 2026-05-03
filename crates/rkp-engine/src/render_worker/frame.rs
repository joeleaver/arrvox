//! Per-frame orchestration — `render_one_frame` runs the full
//! encode/submit/readback cycle for one snapshot.
//!
//! Sized at ~800 lines (`render_one_frame` itself is the bulk) — over
//! the file budget but it's structurally one big sequencing function.
//! The small helpers it calls live in [`super::frame_helpers`]; the
//! user-shader-pipeline tick lives in [`super::user_shader_tick`].
//!
//! [`RenderOutcome`] is the per-frame result — what `render_one_frame`
//! hands back to the loop so the loop can ship `RenderResult` to sim.

use rkp_render::rkp_scene::FrameUpload;

use crate::render_frame::RenderFrame;
use crate::viewport::ViewportId;

use super::frame_helpers::{compute_tlas_scene_aabb, merge_tile_lists, prepare_shadow_maps};
use super::state::{FrameCallback, RenderState, MIN_FRAME_CALLBACK_INTERVAL};
use super::user_shader_tick::{run_user_shader_geom, tick_instance_pipeline};

/// Render a single snapshot.
///
/// `frame` is the canonical sim snapshot (lights, environment,
/// cameras, proc raymarch state, etc.). `gpu_objects` is the
/// possibly-interpolated object list to upload — at α=1 or when
/// there's no prev snapshot, it's `frame.gpu_objects.clone()`;
/// otherwise it's the TRS-blended version from
/// [`interpolate_gpu_objects`].
///
/// `new_snapshot_consumed` is true on the iteration that just
/// took a fresh snapshot from the inbox — gates the editor pixel
/// callback. When false (we're re-rendering the same snapshot for
/// interpolation), GPU work still runs but pixels are not shipped
/// to the editor surface (the content didn't change, so shipping
/// would just thrash rinch's `Mutex<RenderSurfaceBuffer>` with no
/// visible benefit).
pub(super) struct RenderOutcome {
    /// Latest cloud-sun attenuation read from MAIN's volumetric
    /// pass (NaN if MAIN isn't visible).
    pub(super) cloud_sun_atten_raw: f32,
    /// Wall-clock ms since the previous iteration that successfully
    /// shipped pixels to the editor. `None` when this iteration did
    /// not ship (skipped via `ship_pixels` gate) — sim uses `None`
    /// to hold the previous delivered-FPS EMA sample unchanged.
    pub(super) delivered_dt_ms: Option<f32>,
}

pub(super) fn render_one_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    new_snapshot_consumed: bool,
    frame_callback: &FrameCallback,
) -> RenderOutcome {
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

    // 3. Per-viewport encode + submit + readback. One submit per VR so
    //    `queue.write_buffer` writes for that VR's per-frame params
    //    (vol/cloud/atmo/god-ray/shade) are correctly paired with the
    //    encoded dispatches reading them.
    let mut pick_issued = false;

    // Drop a freshly-arrived pick request if a previous pick is
    // still in flight on the readback buffer. Encoding a second
    // copy_texture_to_buffer + map_async into a still-mapped buffer
    // causes a validation error at submit and a panic in map_async.
    //
    // This race was rare at 60 Hz (picks resolve in 1-2 sim frames)
    // but very common with `render_pacing: Uncapped`: at 200 Hz a
    // pick takes ~10 render iterations to complete, plenty of time
    // for the user to click again. Dropping the new request is the
    // simplest correct behavior — the user can re-click; a second
    // click 50 ms later is invisibly close to the first as far as
    // pick UX goes.
    let active_pending_pick = if state.pick_in_flight.is_some() {
        None
    } else {
        frame.pending_pick
    };

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

    for (vp_idx, vp) in frame.viewports.iter().enumerate() {
        // Override `prev_vp` (and the parallel `prev_view_proj` field
        // on the volumetric params) with the view_proj we actually
        // rendered last for THIS viewport. Sim bakes its previous
        // tick's view_proj into the snapshot, but with the GPU-
        // backpressure backoff we may have skipped several sim ticks
        // between renders — TAA reprojection (cloud march, octree
        // march, shade) would then sample history with a `prev_vp`
        // that doesn't describe what's actually in the history
        // texture, producing the streak/blur seen on the sky.
        //
        // Both the camera uniform and the volumetric params carry
        // their own copy of the matrix; patch them in lock-step so
        // the cloud-TAA reprojection and the rest of the pipeline
        // agree on the same previous frame.
        let prev_vp_override = state
            .last_rendered_vp
            .get(&vp.id)
            .copied()
            .unwrap_or(vp.camera.view_proj);
        let mut camera = vp.camera;
        camera.prev_vp = prev_vp_override;
        let mut vol_params = vp.vol_params;
        vol_params.prev_view_proj = prev_vp_override;

        let vr = state
            .viewport_renderers
            .get_mut(&vp.id)
            .expect("snapshot referenced an unknown viewport");

        // 3a. Per-VR camera + scene/lights bind group refresh.
        vr.upload_camera(&state.queue, &camera);
        vr.refresh_bindings(&state.device, &state.renderer);

        // 3b. Per-VR per-frame param uploads (vol/cloud/god-ray).
        vr.volumetric.update_params(&state.queue, &vol_params);
        vr.volumetric.update_cloud_params(&state.queue, &vp.cloud_params);
        vr.god_rays.update_params(&state.queue, &vp.god_ray_params);

        // 3c. Per-VR shade params (isolation-aware). Phase 8 S4 —
        // flip shadow_map_enabled in lockstep with the shadow-map
        // dispatch gate so the shade pass samples the fresh map
        // when one was rendered. Isolation/raymarch leave it 0 so
        // the existing forced-1.0 / shadow_data path stays in
        // charge.
        let mut shade_params = vp.shade_params;
        let in_situ = matches!(vp.mode, rkp_render::RenderMode::InSitu);
        let raymarch = matches!(vp.preview_mode, rkp_render::BuildPreviewMode::Raymarch);
        let vr_shadow_map_live = shadow_map_enabled && in_situ && !raymarch;
        shade_params.shadow_map_enabled = u32::from(vr_shadow_map_live);
        state
            .renderer
            .update_shade_params(&state.queue, &shade_params);

        // 3d. Bloom-composite intensity (zero in isolation mode).
        vr.bloom_composite
            .set_intensity(&state.queue, vp.bloom_composite_intensity);

        // 3e. BUILD viewport: optionally pin the studio floor under the
        //     previewed entity instead of world origin.
        if let Some(grid) = vp.grid_override {
            vr.grid.update_params(&state.queue, &grid);
        }

        // 3f. Per-viewport encoder.
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rkp viewport"),
            });

        // 3g. Procedural raymarch upload (instructions + outline + ghosts)
        //     when this VR is in raymarch preview mode.
        if let Some(proc) = &vp.proc_raymarch {
            vr.proc_raymarch.upload_instructions(
                &state.device,
                &state.queue,
                &proc.instructions,
            );
            vr.proc_raymarch.set_params(
                &state.queue,
                proc.instructions.len() as u32,
                proc.object_id + 1,
                proc.entity_world,
                proc.aabb_min,
                proc.aabb_max,
            );
            let outline_params = match proc.selected_node {
                Some(n) => rkp_render::proc_outline::OutlineParams::new(
                    n,
                    [1.0, 0.55, 0.15, 1.0],
                ),
                None => rkp_render::proc_outline::OutlineParams::NONE,
            };
            vr.proc_outline.update_params(&state.queue, &outline_params);
            vr.proc_ghost.upload_instructions(
                &state.device,
                &state.queue,
                &proc.ghost_instructions,
            );
            vr.proc_ghost.update_params(
                &state.queue,
                &rkp_render::proc_ghost::GhostParams::new(
                    proc.ghost_instructions.len() as u32,
                    [0.25, 0.7, 1.0, 0.35],
                ),
            );
        }

        // 3h. The big one — full per-VR dispatch chain (atmo, march or
        //     proc_raymarch, shadow, ssao, shade, vol, god_rays, bloom,
        //     bloom_composite, tone_map, composite, grid).
        //
        // Merge per-tile object lists across two sources:
        //   - sim's persistent objects (`vp.tile_*_bytes`, already culled)
        //   - Phase C transient indices (broadcast to every tile;
        //     small N, mostly used for whole-entity user-shader regions)
        // No-op pass-through when transients are empty. User-shader
        // instances flow through the GPU tile-cull pipeline now and
        // don't go through this CPU merge (Phase 6 Session 4d).
        let (effective_tile_offsets, effective_tile_object_ids);
        let need_merge = !transient_indices.is_empty();
        let (tile_offsets_ref, tile_object_ids_ref): (&[u8], &[u8]) = if !need_merge {
            (&vp.tile_offsets_bytes, &vp.tile_object_ids_bytes)
        } else {
            let (offsets, ids) = merge_tile_lists(
                &vp.tile_offsets_bytes,
                &vp.tile_object_ids_bytes,
                &transient_indices,
            );
            effective_tile_offsets = offsets;
            effective_tile_object_ids = ids;
            (
                bytemuck::cast_slice(&effective_tile_offsets),
                bytemuck::cast_slice(&effective_tile_object_ids),
            )
        };
        state.renderer.render_to(
            &mut encoder,
            &state.queue,
            vr,
            object_count,
            frame.shadow_steps,
            vp.shade_params.num_lights,
            frame.lod_enabled,
            frame.surfacenet_enabled,
            tile_offsets_ref,
            tile_object_ids_ref,
            vp.tile_count_x,
            state.tlas_pass.last_node_count,
            // Phase B-redux Phase 3a — frame time + asset count for
            // user-shader instance_at derivation in march.
            frame.shade_params_base.time,
            asset_count,
            state.tlas_pass.last_leaf_count,
            // Conservative scene extent for shadow-frustum cull —
            // the longest axis of the scene AABB.
            (scene_aabb.1[0] - scene_aabb.0[0])
                .max(scene_aabb.1[1] - scene_aabb.0[1])
                .max(scene_aabb.1[2] - scene_aabb.0[2]),
            vp.camera.view_proj,
            shadow_map_enabled,
            &vp.atmo_frame,
            vp.mode,
            vp.preview_mode,
        );

        // 3i. Pick encode — if there's a pending pick targeted at this
        //     viewport AND no previous pick is still in flight (see
        //     `active_pending_pick`), copy the relevant 1×1 G-buffer
        //     pixels into the readback buffer slots.
        if let Some(pp) = &active_pending_pick {
            if pp.viewport == vp.id && pp.x < vr.width && pp.y < vr.height {
                pick_issued = true;
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.gbuffer.material_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.pick_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 256,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                // Position slot (Rgba32Float, 16 B per texel). The sim
                // reads xyz + hit_distance; drag-drop uses the xyz as
                // the surface snap point and the hit_distance (>1e9 →
                // sky miss) as the "did it hit anything" bit.
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &vr.gbuffer.position_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: pp.x, y: pp.y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &state.pick_readback_buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 512,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }

        // 3j. Wireframe overlays — gizmo on MAIN (when editor overlays
        //     are enabled) and procedural-node gizmo on BUILD. Sim
        //     pre-built the verts; render just submits.
        if vp.show_editor_overlays && !vp.wireframe_verts.is_empty() {
            let composite_view = &vr.composite_view;
            let vw = vr.width as f32;
            let vh = vr.height as f32;
            vr.wireframe_pass.draw(
                &state.device,
                &state.queue,
                &mut encoder,
                composite_view,
                vp.vp_matrix,
                (0.0, 0.0, vw, vh),
                &vp.wireframe_verts,
            );
        }

        // 3k. Composite readback (frame pixels back to the editor).
        let readback_idx = vr.encode_composite_readback(&mut encoder);
        state.renderer.resolve_profiler_queries(&mut encoder);
        state.queue.submit(std::iter::once(encoder.finish()));

        if let Some(idx) = readback_idx {
            vr.readback.issue_map_async(idx);
        }
    }

    // Stash this frame's un-interpolated view_proj per viewport for
    // next render's `prev_vp` override. See `last_rendered_vp` doc
    // comment on `RenderState` for why this lives render-side now
    // instead of being trusted from the snapshot.
    for vp in &frame.viewports {
        state.last_rendered_vp.insert(vp.id, vp.camera.view_proj);
    }

    // 4. Kick off MAIN's cloud-sun-atten readback (used by sim's
    //    smoothed sun-color attenuation next frame).
    let cloud_sun_atten_raw = if let Some(main_vr) =
        state.viewport_renderers.get(&ViewportId::MAIN)
    {
        main_vr.volumetric.issue_sun_atten_map();
        main_vr.volumetric.sun_atten_value()
    } else {
        f32::NAN
    };

    // 5. Drive async runtime so map_async callbacks can fire.
    let _ = state.device.poll(wgpu::PollType::Poll);

    // 6. If we issued a pick this frame, wire it up so next frame's
    //    `drain_pick` can return the result to sim. The
    //    `active_pending_pick` filter above guarantees `pick_in_flight`
    //    is `None` here whenever `pick_issued` is true, so the
    //    `map_async` can't double-map.
    if pick_issued {
        if let Some(pp) = active_pending_pick {
            let (tx, rx) = std::sync::mpsc::channel();
            state
                .pick_readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            state.pick_in_flight = Some((pp, rx));
        }
    }

    // 7. Drain composite readbacks for each visible viewport. The
    //    readback drain itself runs every iteration so the rings
    //    don't back up. Whether to fire the editor pixel callback
    //    is gated on TWO things:
    //
    //    a) `new_snapshot_consumed` — there's no point shipping
    //       pixels for an iteration that just re-rendered the same
    //       sim state. The visual content is identical to whatever
    //       we shipped last time. With Uncapped render at 200 Hz
    //       and 60 Hz sim, this alone drops pixel ships from 200
    //       /sec to 60 /sec — matching display refresh.
    //
    //    b) `MIN_FRAME_CALLBACK_INTERVAL` — soft cap that handles
    //       the edge case where sim itself runs faster than
    //       display refresh (Uncapped sim, very fast scenes).
    //       Without this an Uncapped sim at 600 Hz would still try
    //       to ship 600 frames/sec to the editor and saturate
    //       rinch's surface buffer Mutex.
    //
    //    Together: pixel ship rate = min(sim_rate, display_rate),
    //    which is exactly what the editor surface can usefully
    //    consume.
    let now = std::time::Instant::now();
    let time_ok = now.duration_since(state.last_frame_callback)
        >= MIN_FRAME_CALLBACK_INTERVAL;
    let ship_pixels = new_snapshot_consumed && time_ok;
    // Interval since the previous successful pixel ship. Sampled
    // BEFORE we update `last_frame_callback` below so we get the
    // gap between ship N-1 and ship N. Only populated when at least
    // one viewport actually handed fresh pixels to the callback —
    // `ship_pixels` gates the try, but `cached_pixels()` may still
    // return None (readback not ready). Delivered FPS should only
    // count real pixel deliveries; a skipped ship leaves the sim
    // EMA unchanged rather than double-counting.
    let mut delivered_dt_ms: Option<f32> = None;
    let mut shipped_any = false;
    for vp in &frame.viewports {
        let vr = state
            .viewport_renderers
            .get_mut(&vp.id)
            .expect("viewport renderer must exist");
        let w = vr.width;
        let h = vr.height;
        let padded_row = vr.readback_padded_row();
        vr.readback.drain_completed(w, h, padded_row);
        if ship_pixels {
            if let Some((pixels, cw, ch)) = vr.readback.cached_pixels() {
                frame_callback(vp.id, pixels, cw, ch);
                shipped_any = true;
            }
        }
    }
    if shipped_any {
        delivered_dt_ms = Some(
            now.duration_since(state.last_frame_callback).as_secs_f32() * 1000.0,
        );
        state.last_frame_callback = now;
    }

    RenderOutcome { cloud_sun_atten_raw, delivered_dt_ms }
}

