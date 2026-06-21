//! Pre-frame setup: phases 0 (uploads + user-shader registry resync),
//! 1 (geometry/instance/TLAS upload + shadow setup), 2 (skin scatter).
//!
//! Returns [`super::PreFrameOutput`] carrying the cross-phase state
//! [`super::encode::encode_viewports`] needs (transient indices, scene
//! AABB, shadow-map-enabled flag, combined asset count).

use arvx_render::arvx_scene::FrameUpload;

use crate::render_frame::RenderFrame;

use super::super::frame_helpers::{compute_tlas_scene_aabb, prepare_shadow_maps};
use super::super::state::RenderState;
use super::super::user_shader_mesh_tick::tick_user_shader_mesh;

use super::PreFrameOutput;

pub(super) fn run_pre_frame(
    state: &mut RenderState,
    frame: &RenderFrame,
    gpu_instances: &[arvx_render::arvx_gpu_object::ArvxGpuInstance],
) -> PreFrameOutput {
    // Sub-phase timing inside pre, gated on `ARVX_RENDER_PROFILE=1`.
    // Splits the `pre` bucket of `[render.frame]` into the major
    // upload/dispatch groups so we can attribute the cost.
    let pre_profile = std::env::var("ARVX_RENDER_PROFILE").is_ok();
    let pre_start = std::time::Instant::now();

    // 0. (P2) No `device.poll` here. The dedicated readback-poll thread is the
    //    SOLE device poller; it drives every in-flight async map (volumetric
    //    sun-atten, composite readbacks, pick readbacks) and every
    //    `on_submitted_work_done` callback. A second poller on the render
    //    thread would defeat the decoupling and re-introduce render-side polls.

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
    if frame.geometry_epoch > state.last_uploaded_geometry_epoch { 'geo: {
        let geo_epoch_t0 = std::time::Instant::now();
        // Mark a load "active" for ~600 ms whenever geometry is PENDING —
        // set BEFORE the try_lock so it stays set even while a big-asset
        // splice holds scene_mgr (try_lock fails, the upload is skipped, but
        // the load is still in progress). The loop holds the lower inflight
        // cap while active so a deep render queue can't build up behind the
        // splice and stall the upload that follows it (#8). 600 ms spans a
        // big splice's lock hold; refreshed every frame there's pending geo.
        state.geo_active_until =
            std::time::Instant::now() + std::time::Duration::from_millis(600);
        // Never block the render thread on a sim-side asset splice. A
        // big-asset splice can hold `scene_mgr` for 100-400ms; WAITING on
        // it here is exactly the `lock=367ms` stall that produced the
        // ~573ms frame. Instead `try_lock` and, if the sim holds it, skip
        // this frame's geometry upload and retry next frame: the epoch
        // gate stays unsatisfied so we self-heal, the pool dirty ranges +
        // per-asset mesh flags persist for the retry, and an asset whose
        // mesh hasn't uploaded yet returns `None` from `mesh_buffer()` —
        // simply undrawn for a frame or two, never a panic.
        let Ok(mut sm) = state.scene_mgr.try_lock() else {
            break 'geo;
        };
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
            arvx_render::arvx_scene_manager::ms_since_process_ns(bump_ns)
        {
            let bump_to_submit_ms = if submit_ns > bump_ns {
                (submit_ns - bump_ns) as f64 / 1.0e6
            } else {
                f64::NAN
            };
            let submit_to_pickup_ms = if submit_ns >= bump_ns {
                arvx_render::arvx_scene_manager::ms_since_process_ns(submit_ns)
                    .unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            eprintln!(
                "[sculpt-pipeline] sim→render={:.2}ms = bump→submit={:.2}ms + submit→pickup={:.2}ms (geo_epoch={})",
                sim_to_render_ms, bump_to_submit_ms, submit_to_pickup_ms, frame.geometry_epoch,
            );
        }
        // ── #3 byte-budgeted geometry upload ──────────────────────────
        // Spread a big (coalesced multi-asset) upload across frames under
        // a per-frame byte budget so it never freezes the present. Reset
        // the cross-frame cursor when a new epoch arrives mid-drain: fresh
        // appended geometry means the references must re-upload and a
        // previously-"done" mesh may be dirty again. The per-pool GPU
        // high-water marks (on ArvxScene) are NOT reset — their bytes stay
        // valid; they just chase the now-larger pool length.
        if state.geo_budget.in_progress_epoch != frame.geometry_epoch {
            state.geo_budget.restart(frame.geometry_epoch);
        }
        let budget = state.geo_upload_budget_bytes;

        let t1 = std::time::Instant::now();
        let geo = sm.geometry_upload();
        let t_snapshot = t1.elapsed();

        // Phase A (data pools, budget-split over frames) + Phase B
        // (octree + brick_face_links, atomic, only once the data pools are
        // fully resident). Until Phase B runs the appended geometry is
        // invisible — no octree node references the new leaf/brick slots —
        // so a half-filled pool tail never shows as garbage.
        let t2 = std::time::Instant::now();
        let progress = state.renderer.upload_geometry_budgeted(
            &state.queue,
            &geo,
            budget,
            &mut state.geo_budget.refs_uploaded,
        );
        // Drop the append-dirty the cursor just consumed so a pool that
        // drained early stops re-uploading its resident prefix every frame
        // (which would starve the budget and stall the whole drain). The
        // remaining dirty is the un-shipped tail + any genuine in-place
        // edit, both still handled. (`geo` borrowed the dirty snapshot;
        // its borrow has ended, so taking `sm` mutably here is fine.)
        let (lv, cv, bv, brv) = state.renderer.pool_valid_marks();
        sm.clear_geometry_dirty_below(lv, cv, bv, brv);
        let t_pool_upload = t2.elapsed();

        // Phase C — per-asset mesh + cluster upload, budgeted. A fresh
        // asset fills its VBO/IBO across frames (undrawn until complete —
        // `mesh_buffer()` returns `None`, never garbage); a small asset
        // finishes in one call. Each asset's own `mesh_dirty` flag is the
        // cursor: cleared the moment its mesh is fully resident so it's
        // not re-yielded (and never re-uploaded in full on the next epoch
        // bump). Runs only after the references are up, so an asset's mesh
        // is never drawable before its octree/leaf data is resident.
        let t_mesh_start = std::time::Instant::now();
        let mut mesh_bytes_total: u64 = 0;
        let mut mesh_asset_count: usize = 0;
        let mut any_cluster_buffer_replaced = false;
        let mut mesh_drained = false;
        if progress.refs_uploaded {
            // Collect this frame's work under immutable borrows of `sm`, so
            // the per-asset dirty-clear afterwards can take `sm` mutably
            // without fighting the live mesh iterator.
            let cluster_map: std::collections::HashMap<u32, _> = sm
                .iter_loaded_asset_clusters()
                .map(|(h, c)| (h.raw(), c))
                .collect();
            let pending: Vec<_> = sm
                .iter_loaded_asset_meshes()
                .map(|(h, v, i, vd, id, lod0)| (h.raw(), v, i, vd, id, lod0))
                .collect();
            let mut done_handles: Vec<u32> = Vec::new();
            let mut stalled = false; // ran out of budget before draining all
            let mut remaining = budget;
            for (raw, vertices, indices, vertices_dirty, indices_dirty, lod0_index_count)
                in pending
            {
                if remaining == 0 {
                    stalled = true;
                    break;
                }
                let (bytes, done) = state.renderer.upload_mesh_for_asset_budgeted(
                    &state.queue,
                    raw,
                    vertices,
                    indices,
                    vertices_dirty,
                    indices_dirty,
                    lod0_index_count,
                    remaining,
                );
                remaining = remaining.saturating_sub(bytes);
                mesh_bytes_total += bytes;
                if done {
                    // Clusters upload only when the mesh is fully resident,
                    // SAME frame, AFTER the mesh (the table indexes the IBO).
                    if let Some(&clusters) = cluster_map.get(&raw) {
                        let replaced = state.renderer.upload_mesh_clusters_for_asset(
                            &state.queue, raw, clusters,
                        );
                        any_cluster_buffer_replaced |= replaced;
                    }
                    done_handles.push(raw);
                    mesh_asset_count += 1;
                } else {
                    // Still mid-upload — it consumed this frame's budget.
                    stalled = true;
                    break;
                }
            }
            // The iterator + cluster_map borrows have ended; now clear each
            // completed asset's dirty flag (the mesh cursor).
            for raw in &done_handles {
                sm.mark_asset_upload_clean(*raw);
            }
            // Every pending (dirty) asset was processed unless we stalled on
            // the budget, so `!stalled` ⇒ no dirty meshes remain.
            mesh_drained = !stalled;
        }
        let t_mesh_upload = t_mesh_start.elapsed();

        // Completion — only when Phase A + B + C are ALL drained do we
        // clear the dirty trackers, mark assets clean, and advance the
        // epoch. A partial frame leaves them intact so the gate
        // (frame.geometry_epoch > last_uploaded) re-enters next frame and
        // the cursors continue. This clear-and-advance-on-complete rule is
        // the load-bearing invariant: clearing early would drop the
        // un-uploaded remainder permanently (the epoch already advanced).
        let complete = progress.data_drained && progress.refs_uploaded && mesh_drained;
        if complete {
            sm.clear_geometry_dirty_ranges();
            sm.mark_loaded_asset_uploads_clean();
            // Read the epoch under the SAME lock so a mid-frame mutation
            // (bake worker integrating an artifact) doesn't trick us into
            // thinking we're caught up. Worst case: re-upload next frame.
            state.last_uploaded_geometry_epoch = sm.geometry_epoch();
            state.geo_budget.restart(0); // idle
        }
        drop(sm);

        let t_total = geo_epoch_t0.elapsed();
        eprintln!(
            "[geo-epoch] lock={:.2}ms snap={:.2}ms pools+refs={:.2}ms mesh={:.2}ms total={:.2}ms \
             | epoch={} dataDrained={} refs={} meshThisFrame={} meshDrained={} consumed={:.2}MiB complete={}",
            t_lock.as_secs_f64() * 1000.0,
            t_snapshot.as_secs_f64() * 1000.0,
            t_pool_upload.as_secs_f64() * 1000.0,
            t_mesh_upload.as_secs_f64() * 1000.0,
            t_total.as_secs_f64() * 1000.0,
            frame.geometry_epoch,
            progress.data_drained,
            progress.refs_uploaded,
            mesh_asset_count,
            mesh_drained,
            (progress.bytes_consumed + mesh_bytes_total) as f64 / (1024.0 * 1024.0),
            complete,
        );

        // Invalidate cached `mesh_lod_select_g2_bgs` (and shadow
        // counterparts) across every viewport whenever a cluster buffer
        // was actually replaced THIS frame. The g2 bind groups hold
        // references to the per-asset cluster table buffer; if the cluster
        // table was replaced (initial alloc, grow, or empty-clear), the
        // cached BG still points at the dropped buffer and the compute
        // pass reads stale cluster data — admit fails on every cluster and
        // the asset's geometry vanishes. The freshness key in
        // viewport_renderer (asset_handle_raw, args_capacity) doesn't
        // catch this, and under budgeting a cluster replace can land on a
        // later frame than the pool-drain — so this MUST run on whatever
        // frame the replace happens, not gated on the epoch "completing".
        // In-place cluster reuse (no realloc) leaves the buffer object
        // unchanged → skip the cascade (the steady-state sculpt path).
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
    } } // close `'geo` (try_lock skip-on-contention) + the epoch gate

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

    let assets_for_upload: &[arvx_render::arvx_gpu_object::ArvxGpuAsset] = frame.gpu_assets.as_slice();
    let instances_for_upload: &[arvx_render::arvx_gpu_object::ArvxGpuInstance] = gpu_instances;

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
    let tlas_inputs = arvx_render::tlas_build_pass::GpuTlasBuildInputs {
        instances_buffer: &state.renderer.scene.objects_buffer,
        instance_count: host_count,
        assets_buffer: &state.renderer.scene.assets_buffer,
        asset_count,
        scene_min: scene_aabb.0,
        scene_max: scene_aabb.1,
    };
    state.tlas_build_pass.build_gpu_tlas(
        &state.device,
        &state.queue,
        &tlas_inputs,
        &mut state.tlas_pass,
        Some(&state.renderer.profiler),
    );
    let _ = asset_count;

    let p_t_tlas = pre_start.elapsed();

    // Directional CSM setup — picks the dominant directional
    // shadow caster, fits per-cascade projections to the camera
    // frustum + scene AABB, writes per-VR `LightCameraCsm` uniforms.
    // Returns whether the shadow render should dispatch this frame;
    // the shade pass gates its sample on the matching
    // `ShadeParams.shadow_map_enabled`.
    let shadow_map_enabled = prepare_shadow_maps(state, frame, scene_aabb);

    let p_t_shadow_prepare = pre_start.elapsed();

    if pre_profile {
        let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[render.pre] setup={:.2} uploads={:.2} mesh_user_shader={:.2} upload_frame={:.2} tlas={:.2} shadow={:.2} | total={:.2}",
            to_ms(p_t_setup),
            to_ms(p_t_uploads) - to_ms(p_t_setup),
            to_ms(p_t_mesh) - to_ms(p_t_uploads),
            to_ms(p_t_upload_frame) - to_ms(p_t_mesh),
            to_ms(p_t_tlas) - to_ms(p_t_upload_frame),
            to_ms(p_t_shadow_prepare) - to_ms(p_t_tlas),
            to_ms(p_t_shadow_prepare),
        );
    }
    PreFrameOutput {
        scene_aabb,
        shadow_map_enabled,
    }
}
