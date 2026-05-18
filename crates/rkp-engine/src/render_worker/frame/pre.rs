//! Pre-frame setup: phases 0 (uploads + user-shader registry resync),
//! 1 (geometry/instance/TLAS upload + shadow setup), 2 (skin scatter).
//!
//! Returns [`super::PreFrameOutput`] carrying the cross-phase state
//! [`super::encode::encode_viewports`] needs (transient indices, scene
//! AABB, shadow-map-enabled flag, combined asset count).

use rkp_render::rkp_scene::FrameUpload;

use crate::render_frame::RenderFrame;

use super::super::frame_helpers::{compute_tlas_scene_aabb, prepare_shadow_maps};
use super::super::state::RenderState;
use super::super::user_shader_mesh_tick::tick_user_shader_mesh;

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
        vr.shade.upload_shader_params(
            &state.device,
            &state.queue,
            &frame.shader_params_slots,
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
        let geo_epoch_t0 = std::time::Instant::now();
        let mut sm = state.scene_mgr.lock().expect("scene_mgr poisoned");
        let t_lock = geo_epoch_t0.elapsed();
        // Diagnostic: decompose the wall-clock latency between sim's
        // `bump_geometry_epoch` and the render worker noticing the new
        // epoch into two phases:
        //   • bump→submit — sim work post-bump in the same submit_render_
        //     frame tick (snapshot building, palette, gpu_objects rebuild,
        //     painted-material scan, etc.). Stays inside one sim tick by
        //     design.
        //   • submit→pickup — wall time from sim handing the snapshot to
        //     the inbox until render's iteration picked it up. Captures
        //     render-thread cadence + GPU backpressure (readback ring
        //     full → 500 µs sleep + retry).
        let bump_ns = sm.last_geometry_bump_ns();
        let submit_ns = sm.last_geometry_submit_ns();
        if let Some(sim_to_render_ms) =
            rkp_render::rkp_scene_manager::ms_since_process_ns(bump_ns)
        {
            let bump_to_submit_ms = if submit_ns > bump_ns {
                (submit_ns - bump_ns) as f64 / 1.0e6
            } else {
                f64::NAN
            };
            let submit_to_pickup_ms = if submit_ns >= bump_ns {
                rkp_render::rkp_scene_manager::ms_since_process_ns(submit_ns)
                    .unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            eprintln!(
                "[sculpt-pipeline] sim→render={:.2}ms = bump→submit={:.2}ms + submit→pickup={:.2}ms (geo_epoch={})",
                sim_to_render_ms, bump_to_submit_ms, submit_to_pickup_ms, frame.geometry_epoch,
            );
        }
        let t1 = std::time::Instant::now();
        let geo = sm.geometry_upload();
        let t_snapshot = t1.elapsed();
        let t2 = std::time::Instant::now();
        state.renderer.upload_geometry(&state.queue, &geo);
        let t_pool_upload = t2.elapsed();
        let t3 = std::time::Instant::now();
        // Delta-upload: the snapshot in `geo` cloned the per-pool dirty
        // trackers; now that the writes have been queued, clear the
        // manager-side trackers so the next epoch ships only fresh
        // mutations rather than the same bytes again. Must run AFTER
        // upload_geometry so a write_buffer failure (in tests, panics)
        // leaves the trackers populated for retry.
        sm.clear_geometry_dirty_ranges();
        let t_clear = t3.elapsed();
        // Mesh path — keep the per-asset (vbo, ibo) cache in step with the
        // path's per-asset (vbo, ibo) cache. iter_loaded_asset_meshes
        // skips empty mesh extractions (procedurals etc.) so this only
        // touches assets that produced a non-empty surface mesh at
        // load time. Phase 6.1: indices is the full DAG IBO (LOD-0
        // first, then LOD-1, ...); dispatch draws only the LOD-0
        // prefix until Phase 6.2 wires the indirect path.
        let t_mesh_start = std::time::Instant::now();
        let mut mesh_bytes_total: u64 = 0;
        let mut mesh_asset_count: usize = 0;
        for (handle, vertices, indices, indices_dirty, lod0_index_count)
            in sm.iter_loaded_asset_meshes()
        {
            let bytes = state.renderer.upload_mesh_for_asset(
                &state.queue,
                handle.raw(),
                vertices,
                indices,
                indices_dirty,
                lod0_index_count,
            );
            mesh_bytes_total += bytes;
            mesh_asset_count += 1;
        }
        let t_mesh_upload = t_mesh_start.elapsed();
        if mesh_asset_count > 0 {
            let mib = mesh_bytes_total as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[delta upload] mesh: {mesh_asset_count} asset(s) · {mib:.3} MiB total \
                 (VBO+IBO tail writes) in {:.2} ms",
                t_mesh_upload.as_secs_f64() * 1000.0,
            );
        }
        let t_cluster_start = std::time::Instant::now();
        // Phase 5 — per-asset meshlet cluster table. Storage buffer
        // for the Phase 6 LOD-selection compute pass; uploaded here
        // but unused by current dispatch (validates the upload path
        // without rewiring the hot draw call).
        let mut cluster_asset_count = 0;
        let mut any_cluster_buffer_replaced = false;
        for (handle, clusters) in sm.iter_loaded_asset_clusters() {
            let replaced = state.renderer.upload_mesh_clusters_for_asset(
                &state.queue, handle.raw(), clusters,
            );
            any_cluster_buffer_replaced |= replaced;
            cluster_asset_count += 1;
        }
        let t_cluster_upload = t_cluster_start.elapsed();
        if cluster_asset_count > 0 {
            eprintln!(
                "[delta upload] cluster table: {cluster_asset_count} asset(s) in {:.2} ms",
                t_cluster_upload.as_secs_f64() * 1000.0,
            );
        }
        // Per-asset dirty-flag clean-up: every iter above only yielded
        // assets whose `mesh_dirty / splats_dirty / clusters_dirty`
        // flag was set; clear them now so the next epoch bump only
        // re-uploads assets that mutated in the interim. Cut the
        // sculpt-stamp upload cost from "every loaded asset × full
        // realloc" to "just the one sculpted asset".
        sm.mark_loaded_asset_uploads_clean();
        // Read-back the epoch *under the same lock* so concurrent
        // mutations (bake worker integrating an artifact mid-frame)
        // don't trick us into thinking we're caught up when we're
        // not. Worst case: we re-upload next frame, which is fine.
        state.last_uploaded_geometry_epoch = sm.geometry_epoch();
        drop(sm);
        let t_total = geo_epoch_t0.elapsed();
        eprintln!(
            "[geo-epoch] lock={:.2}ms snap={:.2}ms pool={:.2}ms clear={:.2}ms \
             mesh={:.2}ms clusters={:.2}ms total={:.2}ms",
            t_lock.as_secs_f64() * 1000.0,
            t_snapshot.as_secs_f64() * 1000.0,
            t_pool_upload.as_secs_f64() * 1000.0,
            t_clear.as_secs_f64() * 1000.0,
            t_mesh_upload.as_secs_f64() * 1000.0,
            t_cluster_upload.as_secs_f64() * 1000.0,
            t_total.as_secs_f64() * 1000.0,
        );

        // Invalidate cached `mesh_lod_select_g2_bgs` (and shadow
        // counterparts) across every viewport, but only when at least
        // one cluster buffer was actually replaced this epoch. The g2
        // bind groups hold references to the per-asset cluster table
        // buffer; if the cluster table was replaced (initial alloc,
        // grow, or empty-clear), the cached BG still points at the
        // dropped buffer and the compute pass reads stale cluster
        // data — admit fails on every cluster and the asset's
        // geometry vanishes from the frame. The freshness key in
        // viewport_renderer (asset_handle_raw, args_capacity) doesn't
        // catch this case.
        //
        // When the cluster buffer is reused in place
        // (`queue.write_buffer`, no allocation), the wgpu::Buffer
        // object is unchanged and the cached BG still points at the
        // correct buffer — skip the invalidation. This is the steady-
        // state sculpt path; eliding the cascade avoids re-creating
        // every BG per stamp.
        if any_cluster_buffer_replaced {
            for vr in state.viewport_renderers.values_mut() {
                for slot in vr.mesh_lod_select_g2_bgs.iter_mut() {
                    *slot = None;
                }
                for per_slot in vr.mesh_lod_shadow_g2_bgs.iter_mut() {
                    for per_cascade in per_slot.iter_mut() {
                        *per_cascade = None;
                    }
                }
            }
        }
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

    // 1.7a. Upload `instance_overlay_buffer` for the per-instance
    //       paint cursor (Phase 3).
    if !frame.gpu_instance_overlays.is_empty() {
        let overlay_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_overlays);
        state.renderer.scene.upload_instance_overlay(
            &state.device, &state.queue, overlay_bytes,
        );
    }
    // 1.7b. Upload `instance_sculpt_buffer` for the per-instance sculpt
    //        carve overlay (Phase A). Same out-of-band path as paint —
    //        needs to be visible to compute passes that run before the
    //        main `upload_frame` below (e.g. tile-cull, user-shader BFS).
    if !frame.gpu_instance_sculpts.is_empty() {
        let sculpt_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_sculpts);
        state.renderer.scene.upload_instance_sculpt(
            &state.device, &state.queue, sculpt_bytes,
        );
    }

    let assets_for_upload: &[rkp_render::rkp_gpu_object::RkpGpuAsset] = frame.gpu_assets.as_slice();
    let instances_for_upload: &[rkp_render::rkp_gpu_object::RkpGpuInstance] = gpu_instances;

    let p_t_mesh = pre_start.elapsed();

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
    //      / `assets_buffer`.
    let scene_aabb = compute_tlas_scene_aabb(
        instances_for_upload,
        assets_for_upload,
    );
    let overlay_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_overlays);
    let sculpt_bytes: &[u8] = bytemuck::cast_slice(&frame.gpu_instance_sculpts);
    state.renderer.upload_frame(
        &state.queue,
        &FrameUpload {
            assets: assets_for_upload,
            instances: instances_for_upload,
            bone_matrices: &frame.bone_matrix_lbs,
            bone_matrices_dirty: &frame.bone_matrix_lbs_dirty,
            bone_dual_quats: &frame.bone_matrix_dqs,
            bone_dual_quats_dirty: &frame.bone_matrix_dqs_dirty,
            instance_overlays: overlay_bytes,
            instance_overlays_dirty: &frame.gpu_instance_overlays_dirty,
            instance_sculpts: sculpt_bytes,
            instance_sculpts_dirty: &frame.gpu_instance_sculpts_dirty,
        },
    );

    let p_t_upload_frame = pre_start.elapsed();

    // V1 mesh-path user-shader orchestration. MUST run AFTER
    // `upload_frame` — the compute trio's `spawn_alive` calls
    // `paint_probe` which reads `instances[anchor.object_id]` and
    // `assets[inst.asset_id]` from the scene bind group. Running
    // before upload reads stale (or zero-init) data and rejects
    // every spawn.
    tick_user_shader_mesh(state, frame);

    // 1b''. Phase 7c — fire the GPU TLAS build. Inputs:
    //   * tile-cull scratch buffer (populated by Phase 6 tile-cull
    //     AABB pass)
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
    // Mesh path consumes TLAS via the mesh raster's own bindings;
    // no per-viewport setup is needed here after the march/shadow
    // scatter retirement.
    let _ = tlas_prim_count;
    let _ = asset_count;

    let p_t_tlas = pre_start.elapsed();

    // Phase 8 — directional shadow map. Picks the first
    // directional light, derives the light camera covering the
    // scene AABB, writes the uniform into every VR. Returns
    // whether the shadow map will be live this frame; the shade
    // pass gates its sample on that. Texture dispatch happens
    // later in `render_to`.
    let shadow_map_enabled = prepare_shadow_maps(state, frame, scene_aabb, tlas_prim_count);

    let p_t_shadow_prepare = pre_start.elapsed();

    // Mesh path skins in the vertex shader against the per-frame
    // `bone_matrices` / `bone_dual_quats` buffers — no scatter pass.

    let transient_indices: Vec<u32> = Vec::new();
    let object_count = gpu_instances.len() as u32;

    let p_t_skin = pre_start.elapsed();
    if pre_profile {
        let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[render.pre] setup={:.2} uploads={:.2} mesh_user_shader={:.2} upload_frame={:.2} tlas={:.2} shadow={:.2} skin={:.2} | total={:.2}",
            to_ms(p_t_setup),
            to_ms(p_t_uploads) - to_ms(p_t_setup),
            to_ms(p_t_mesh) - to_ms(p_t_uploads),
            to_ms(p_t_upload_frame) - to_ms(p_t_mesh),
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
    }
}
