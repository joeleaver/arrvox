//! Per-viewport encode: walk every viewport in the snapshot, build
//! a command encoder, dispatch the full per-VR chain (atmo / march /
//! shadow / SSAO / shade / vol / god rays / bloom / tone map /
//! composite / grid + overlays + pick), submit, and queue the
//! composite readback.
//!
//! Phase 3 of [`super::render_one_frame`]. Reads cross-phase state
//! from [`super::PreFrameOutput`] and emits [`super::EncodeOutput`]
//! carrying pick wiring data into [`super::post::finalize_frame`].

use crate::render_frame::RenderFrame;

use super::super::frame_helpers::merge_tile_lists;
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
        let in_situ = matches!(vp.mode, rkp_render::RenderMode::InSitu);
        let raymarch = matches!(vp.preview_mode, rkp_render::BuildPreviewMode::Raymarch);
        let vr_shadow_map_live = pre.shadow_map_enabled && in_situ && !raymarch;
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
        let need_merge = !pre.transient_indices.is_empty();
        let (tile_offsets_ref, tile_object_ids_ref): (&[u8], &[u8]) = if !need_merge {
            (&vp.tile_offsets_bytes, &vp.tile_object_ids_bytes)
        } else {
            let (offsets, ids) = merge_tile_lists(
                &vp.tile_offsets_bytes,
                &vp.tile_object_ids_bytes,
                &pre.transient_indices,
            );
            effective_tile_offsets = offsets;
            effective_tile_object_ids = ids;
            (
                bytemuck::cast_slice(&effective_tile_offsets),
                bytemuck::cast_slice(&effective_tile_object_ids),
            )
        };

        // 3h-pre. User-shader tile-bin pass. Reset per-tile counts and
        // dispatch the binner into the same encoder so its writes are
        // visible to the march that follows. No-op when the emit pass
        // produced 0 instances; the march still reads the (zero-
        // initialized) counts and skips the inner loop.
        vr.clear_user_shader_tile_counts(&mut encoder);
        let bin_params = rkp_render::user_shader_tile_bin_pass::BinParams {
            instance_count_upper_bound: pre.user_shader_instance_count,
            tile_count_x: vr.user_shader_tile_count_x,
            tile_count_y: vr.user_shader_tile_count_y,
            _pad0: 0,
        };
        vr.user_shader_tile_bin.update_params(&state.queue, &bin_params);
        if let Some(bg) = &vr.user_shader_tile_bin_bg {
            let q = state
                .renderer
                .profiler
                .begin_query("user_shader_tile_bin", &mut encoder);
            vr.user_shader_tile_bin.dispatch(
                &mut encoder,
                bg,
                pre.user_shader_instance_count,
            );
            state.renderer.profiler.end_query(&mut encoder, q);
        }

        state.renderer.render_to(
            &mut encoder,
            &state.queue,
            vr,
            pre.object_count,
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
            pre.asset_count,
            state.tlas_pass.last_leaf_count,
            // Conservative scene extent for shadow-frustum cull —
            // the longest axis of the scene AABB.
            (pre.scene_aabb.1[0] - pre.scene_aabb.0[0])
                .max(pre.scene_aabb.1[1] - pre.scene_aabb.0[1])
                .max(pre.scene_aabb.1[2] - pre.scene_aabb.0[2]),
            vp.camera.view_proj,
            pre.shadow_map_enabled,
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

        // March stats — async readback. Gated behind RKP_MARCH_STATS=1
        // so it doesn't spam by default; when enabled, drains any
        // previously-resolved snapshot and eprintln's the descend-body
        // breakdown counters + the user-shader emit instance count.
        // Single staging buffer per source, skip-if-busy — never
        // blocks the render thread.
        if std::env::var("RKP_MARCH_STATS").is_ok() {
            if let Some(stats) = vr.march.try_drain_stats() {
                eprint_march_stats(vp.id, &stats);
            }
            vr.march.submit_stats_readback();
            if let Some(count) = state.user_shader_emit_pass.try_drain_count() {
                eprintln!("[user_shader_emit] emitted instances={count}");
            }
        }
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

/// Format the user-shader descend breakdown from a stats snapshot.
/// See `shaders/octree_march.wgsl` for the slot layout. Only meaningful
/// when there are non-zero band-cell or Phase-3a invocations; otherwise
/// the band-cell descent path didn't fire this frame and the line is
/// dominated by zeros.
fn eprint_march_stats(vp_id: crate::viewport::ViewportId, stats: &[u32]) {
    let band_invoke = stats.get(60).copied().unwrap_or(0);
    let band_hit = stats.get(61).copied().unwrap_or(0);
    let p3a_invoke = stats.get(64).copied().unwrap_or(0);
    let p3a_hit = stats.get(65).copied().unwrap_or(0);
    if band_invoke == 0 && p3a_invoke == 0 {
        return;
    }
    let k_test = stats.get(66).copied().unwrap_or(0);
    let aabb_rej = stats.get(67).copied().unwrap_or(0);
    let descended = stats.get(68).copied().unwrap_or(0);
    let miss = stats.get(69).copied().unwrap_or(0);
    let hit = stats.get(70).copied().unwrap_or(0);
    eprintln!(
        "[march_stats vp={vp_id:?}] dispatch: band={band_invoke}/hits={band_hit} \
         phase3a={p3a_invoke}/hits={p3a_hit} | candidates: k_test={k_test} \
         aabb_rej={aabb_rej} descended={descended} miss={miss} hit={hit}",
    );
}
