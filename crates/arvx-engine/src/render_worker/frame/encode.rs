//! Per-viewport encode: walk every viewport in the snapshot, build
//! a command encoder, dispatch the full per-VR chain (atmo / march /
//! shadow / SSAO / shade / vol / god rays / bloom / tone map /
//! composite / grid + overlays + pick), submit, and queue the
//! composite readback.
//!
//! Phase 3 of [`super::render_one_frame`]. Reads cross-phase state
//! from [`super::PreFrameOutput`] and emits [`super::EncodeOutput`]
//! carrying pick wiring data into [`super::post::finalize_frame`].

use std::sync::atomic::Ordering;

use crate::render_frame::RenderFrame;

use super::super::readback_poll::ReadbackJob;
use super::super::state::RenderState;

use super::{EncodeOutput, PreFrameOutput};

pub(super) fn encode_viewports(
    state: &mut RenderState,
    frame: &RenderFrame,
    pre: &PreFrameOutput,
) -> EncodeOutput {
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

    for vp in frame.viewports.iter() {
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
        let in_situ = matches!(vp.mode, arvx_render::RenderMode::InSitu);
        // Raymarch is the BUILD viewport's only primary-visibility
        // pass — every other viewport (MAIN, play, future) uses the
        // standard mesh raster. Single-viewport check is cheaper than
        // a per-frame enum field.
        let raymarch = vp.id == crate::viewport::ViewportId::BUILD;
        // Mesh path renders its directional shadow map into
        // `shadow_buffer` — gated by `pre.shadow_map_enabled`,
        // which is true whenever a directional caster is live.
        let vr_shadow_map_live =
            pre.shadow_map_enabled && in_situ && !raymarch;
        shade_params.shadow_map_enabled = u32::from(vr_shadow_map_live);
        shade_params.shadow_disabled = 0;
        // PCF tap count comes from the Shadow Quality preset; the
        // shade shader clamps 1..16 internally.
        shade_params.pcf_taps = frame.shadow_csm_pcf_taps;
        // Diagnostic G-buffer debug view (ARVX_DEBUG_VIEW); 0 in normal use.
        shade_params.debug_view = arvx_render::arvx_shade::debug_view_from_env();
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
        //
        // Drain the previous frame's stats readback BEFORE building this
        // frame's encoder, so the next `copy_stats` (encoded inside
        // `render_to`) sees state IDLE and isn't skipped. Doing this
        // after submit (the old order) deadlocked the readback into a
        // stuck-snapshot loop: state went IDLE → PENDING → READY and
        // then every subsequent frame's copy was skipped because state
        // stayed READY until the next drain.
        // March stats readback retired with the march pass. Mesh path
        // uses ARVX_MESH_PIPESTATS for similar diagnostics.
        let drained_stats: Option<()> = None;

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
                Some(n) => arvx_render::proc_outline::OutlineParams::new(
                    n,
                    [1.0, 0.55, 0.15, 1.0],
                ),
                None => arvx_render::proc_outline::OutlineParams::NONE,
            };
            vr.proc_outline.update_params(&state.queue, &outline_params);
            vr.proc_ghost.upload_instructions(
                &state.device,
                &state.queue,
                &proc.ghost_instructions,
            );
            vr.proc_ghost.update_params(
                &state.queue,
                &arvx_render::proc_ghost::GhostParams::new(
                    proc.ghost_instructions.len() as u32,
                    [0.25, 0.7, 1.0, 0.35],
                ),
            );
        }

        // 3h. The big one — full per-VR dispatch chain (atmo, march or
        //     proc_raymarch, shadow, ssao, shade, vol, god_rays, bloom,
        //     bloom_composite, tone_map, composite, grid).
        //
        // Snapshot the user-shader mesh draws before the
        // `state.renderer.render_to` mutable borrow so the
        // immutable read above doesn't conflict. The draws are
        // regenerated each frame by `tick_user_shader_mesh`, so a
        // clone is structurally correct (the slices are small
        // — ≤ N_materials entries; wgpu types inside are cheap-
        // clone refcounted handles).
        let user_shader_mesh_draws = state.user_shader_mesh_draws.clone();
        state.renderer.render_to(
            &mut encoder,
            &state.queue,
            vr,
            pre.shadow_map_enabled,
            &vp.atmo_frame,
            vp.mode,
            raymarch,
            // Mesh-raster instance list built in `update_scene_gpu`
            // from `Renderable.asset_handle`. Procedural entities
            // without an `AssetHandle` ride `proxy_draws` instead.
            &frame.mesh_draws,
            // Procedural proxy-mesh draws. Rendered after primary
            // visibility regardless of `primary_mode` — proxy meshes
            // are first-class scene geometry.
            &frame.proxy_draws,
            // V1 mesh-path user-shader draws. The engine's
            // `tick_user_shader_mesh` already ran the per-material
            // compute trio earlier in `pre.rs`; this slice carries
            // the per-material raster descriptors the renderer
            // consumes after the proxy raster.
            &user_shader_mesh_draws,
            // Cursor pixel for the screen-space paint cursor's
            // brush-state probe pass. `None` = paint mode off (or
            // mouse outside the framebuffer); the probe will write
            // the miss sentinel and shade hides the cursor.
            vp.brush_pixel,
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

        // 3k. Composite readback (frame pixels back to the editor). Claims a
        //     free ring slot + encodes the texture→buffer copy; `None` when
        //     all slots are still in flight on the poll thread (we then render
        //     + submit normally but ship no pixels this frame — newest-wins
        //     tolerates the drop, and we NEVER stall waiting for a slot).
        let readback_idx = vr.encode_composite_readback(&mut encoder);
        state.renderer.resolve_profiler_queries(&mut encoder);

        // P2 pacing: bound GPU-queue depth by in-flight submissions, NOT by
        // readback slots. Increment before submit; decrement when the GPU
        // signals the work is done (fired by the poll thread's `device.poll`).
        state.inflight_submits.fetch_add(1, Ordering::Relaxed);
        state.queue.submit(std::iter::once(encoder.finish()));
        let inflight_done = state.inflight_submits.clone();
        state
            .queue
            .on_submitted_work_done(move || {
                inflight_done.fetch_sub(1, Ordering::Relaxed);
            });

        // Hand the just-written composite slot to the dedicated readback-poll
        // thread (it owns map_async → read → ship → unmap → recycle). The
        // generation tags both newest-wins ship ordering and the slot-free
        // match (so a resize race can't free the wrong slot).
        if let Some(idx) = readback_idx {
            let generation = state.readback_generation.fetch_add(1, Ordering::Relaxed) + 1;
            let buffer = vr.readback.build_in_flight(idx, generation);
            let _ = state.readback_job_tx.send(ReadbackJob {
                vp_id: vp.id,
                slot: idx,
                buffer,
                generation,
                width: vr.width,
                height: vr.height,
                padded_row: vr.readback_padded_row(),
            });
        }
        // Pair the LOD-stats map_async with the matching submit
        // (validation requires map_async after submit, not before).
        // No-op when `ARVX_MESH_LOD_STATS=1` is unset since
        // `dispatch_mesh*` skipped the encoder copy.
        vr.lod_stats_post_submit();

        // March stats — async readback. Gated behind ARVX_MARCH_STATS=1
        // so it doesn't spam by default; when enabled, drains any
        // previously-resolved snapshot and eprintln's the descend-body
        // breakdown counters. Single staging buffer per source,
        // skip-if-busy — never blocks the render thread.
        let _ = drained_stats;
    }

    // Stash this frame's un-interpolated view_proj per viewport for
    // next render's `prev_vp` override. See `last_rendered_vp` doc
    // comment on `RenderState` for why this lives render-side now
    // instead of being trusted from the snapshot.
    for vp in &frame.viewports {
        state.last_rendered_vp.insert(vp.id, vp.camera.view_proj);
    }

    EncodeOutput { pick_issued, active_pending_pick }
}

