//! Small frame-rendering helpers — tile-list splice, AABB transforms,
//! shadow-map setup hook.
//!
//! Pulled out of the 800-line `render_one_frame` so the orchestration
//! body in [`super::frame`] stays focused on the per-frame sequencing.
//! All of these are called only from `render_one_frame` today.

use crate::render_frame::RenderFrame;

use super::state::RenderState;

/// Phase C V1.5 — append `transient_indices` to every tile's object
/// list, returning rebuilt `(tile_offsets, tile_object_ids)` arrays.
///
/// Sim's tile_object_ids only enumerate persistent objects; transient
/// ones (built render-thread-side after the snapshot arrives) need to
/// be visible from every tile so the march visits them. With T tiles
/// and N transient objects the cost is O(T × N) per frame; for V1's
/// few-region demos that's negligible (~MB/frame at most).
///
/// Layout: `tile_offsets` is a prefix-sum (length `T + 1`), so each
/// tile `t` has range `[offsets[t]..offsets[t+1])` in `tile_object_ids`.
/// We splice `transient_indices` after each tile's existing range,
/// shifting downstream offsets accordingly.
pub(super) fn splice_transient_into_tile_lists(
    tile_offsets_bytes: &[u8],
    tile_object_ids_bytes: &[u8],
    transient_indices: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n_tile = if tile_offsets_bytes.is_empty() {
        0
    } else {
        (tile_offsets_bytes.len() / 4).saturating_sub(1)
    };
    if n_tile == 0 || transient_indices.is_empty() {
        // Empty input — return whatever was passed in as u32 vecs.
        let offsets = bytemuck::cast_slice::<u8, u32>(tile_offsets_bytes).to_vec();
        let ids = bytemuck::cast_slice::<u8, u32>(tile_object_ids_bytes).to_vec();
        return (offsets, ids);
    }
    let orig_offsets: &[u32] = bytemuck::cast_slice(tile_offsets_bytes);
    let orig_ids: &[u32] = bytemuck::cast_slice(tile_object_ids_bytes);
    let n_transient = transient_indices.len();
    let mut new_offsets: Vec<u32> = Vec::with_capacity(n_tile + 1);
    let mut new_ids: Vec<u32> = Vec::with_capacity(orig_ids.len() + n_tile * n_transient);

    new_offsets.push(0);
    for t in 0..n_tile {
        let a = orig_offsets[t] as usize;
        let b = orig_offsets[t + 1] as usize;
        new_ids.extend_from_slice(&orig_ids[a..b]);
        new_ids.extend_from_slice(transient_indices);
        new_offsets.push(new_ids.len() as u32);
    }
    (new_offsets, new_ids)
}

/// Derive a conservative scene AABB for the GPU TLAS
/// build's Morton normalization. Walks the host instances' transformed
/// asset AABBs (Arvo). Returns `(min, max)`; falls back to
/// `[0,0,0] → [1,1,1]` for empty input — the Morton dispatch's
/// `extent.max(1e-6)` clamp prevents divide-by-zero, and the empty-
/// TLAS skip gates the downstream chain anyway.
pub(super) fn compute_tlas_scene_aabb(
    instances: &[rkp_render::rkp_gpu_object::RkpGpuInstance],
    assets: &[rkp_render::rkp_gpu_object::RkpGpuAsset],
) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;

    for inst in instances {
        let asset_id = inst.asset_id as usize;
        if asset_id >= assets.len() { continue; }
        let asset = &assets[asset_id];
        let (wmin, wmax) = transform_aabb_world(asset.aabb_min, asset.aabb_max, &inst.world);
        for ax in 0..3 {
            if wmin[ax] < min[ax] { min[ax] = wmin[ax]; }
            if wmax[ax] > max[ax] { max[ax] = wmax[ax]; }
        }
        any = true;
    }

    if !any {
        return ([0.0; 3], [1.0; 3]);
    }
    (min, max)
}

/// Arvo's transform-AABB. Mirrors `tlas_pass.rs::transform_aabb`. Used
/// by the TLAS scene-AABB helper above.
pub(super) fn transform_aabb_world(
    local_min: [f32; 3],
    local_max: [f32; 3],
    world: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let mut new_min = [world[3][0], world[3][1], world[3][2]];
    let mut new_max = [world[3][0], world[3][1], world[3][2]];
    for i in 0..3 {
        for j in 0..3 {
            let a = world[j][i] * local_min[j];
            let b = world[j][i] * local_max[j];
            new_min[i] += a.min(b);
            new_max[i] += a.max(b);
        }
    }
    (new_min, new_max)
}

/// Per-frame CSM setup for the mesh-mode directional shadow.
///
/// Picks the dominant directional shadow caster, fits per-cascade
/// projections to the camera frustum + scene AABB, writes the
/// `LightCameraCsm` uniform on every visible viewport, and writes
/// the per-cascade LOD-select cameras. Returns `true` when a caster
/// was found and the shadow render should dispatch — the shade pass
/// reads `ShadeParams.shadow_map_enabled` (set in lockstep) to gate
/// its sample.
pub(super) fn prepare_shadow_maps(
    state: &mut RenderState,
    frame: &RenderFrame,
    scene_aabb: ([f32; 3], [f32; 3]),
) -> bool {
    // Stash the per-cascade LOD pixel-threshold falloff on the
    // renderer so the next `dispatch_mesh_shadow` picks it up. The
    // env-var override (`RKP_CSM_THRESHOLD_FALLOFF`) takes precedence
    // inside the renderer for one-shot CI / headless tuning.
    state.renderer.set_shadow_csm_threshold_falloff(
        frame.shadow_csm_threshold_falloff,
    );
    use rkp_render::shadow_map_pass::{
        compute_csm_cascades, CsmInputs, CSM_CASCADE_COUNT,
    };
    // Pick the first directional caster — `position[3] == 0` flags
    // a directional light in the GpuLight wire format, and
    // `params[3] >= 0.5` is the shadow-caster bit. CSM applies to
    // the dominant directional light only.
    let Some(light) = frame
        .lights
        .iter()
        .find(|l| (l.position[3] as u32) == 0 && l.params[3] >= 0.5)
    else {
        return false;
    };
    let light_dir = [
        light.direction[0],
        light.direction[1],
        light.direction[2],
    ];

    // CSM knobs from the scene environment (runtime-configurable
    // via the editor's Environment panel). Env-var overrides
    // (RKP_CSM_*) take precedence for headless / CI testing.
    let csm_max_distance = std::env::var("RKP_CSM_MAX_DISTANCE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(frame.shadow_csm_max_distance)
        .clamp(10.0, 1000.0);
    let csm_lambda = std::env::var("RKP_CSM_LAMBDA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(frame.shadow_csm_lambda)
        .clamp(0.0, 1.0);
    let csm_near = std::env::var("RKP_CSM_NEAR")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(frame.shadow_csm_near)
        .clamp(0.05, 10.0);
    let depth_bias = frame.shadow_csm_depth_bias.max(0.0);
    let sharp_distance = std::env::var("RKP_CSM_SHARP_DISTANCE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(frame.shadow_csm_sharp_distance)
        .clamp(0.0, 100.0);
    let map_size = std::env::var("RKP_CSM_MAP_SIZE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(frame.shadow_csm_map_size)
        .clamp(256, 4096);

    // Optional one-time fit log — extends the existing
    // RKP_SHADOW_FIT_LOG with the scene-AABB context.
    let fit_log = std::env::var("RKP_CSM_FIT_LOG").is_ok()
        || std::env::var("RKP_SHADOW_FIT_LOG").is_ok();
    if fit_log {
        let dx = scene_aabb.1[0] - scene_aabb.0[0];
        let dy = scene_aabb.1[1] - scene_aabb.0[1];
        let dz = scene_aabb.1[2] - scene_aabb.0[2];
        eprintln!(
            "[csm] scene_aabb extent = {:.2} × {:.2} × {:.2} m, \
             near = {:.2} m, max_distance = {:.1} m, λ = {:.2}, cascades = {}",
            dx, dy, dz, csm_near, csm_max_distance, csm_lambda, CSM_CASCADE_COUNT,
        );
    }

    let mut wrote_any = false;
    for vp in &frame.viewports {
        let Some(vr) = state.viewport_renderers.get_mut(&vp.id) else {
            continue;
        };

        // Resize the per-viewport shadow_buffer + depth texture +
        // shade bg + blit params (which carry shadow_map_size for
        // the per-cascade blit's stride math) if the user picked a
        // new Shadow Quality tier. No-op if already at `map_size`.
        // Must run before the bind-group refresh inside
        // dispatch_mesh_shadow.
        vr.set_shadow_map_size(&state.device, &state.queue, map_size);

        // Per-VR cascade fit: each viewport's camera frustum drives
        // its own CSM. The light direction + scene AABB are scene-
        // wide.
        let view_proj = glam::Mat4::from_cols_array_2d(&vp.camera.view_proj);
        let camera_position = glam::Vec3::new(
            vp.camera.position[0],
            vp.camera.position[1],
            vp.camera.position[2],
        );
        let camera_forward = glam::Vec3::new(
            vp.camera.forward[0],
            vp.camera.forward[1],
            vp.camera.forward[2],
        );
        let csm = compute_csm_cascades(CsmInputs {
            scene_min: scene_aabb.0,
            scene_max: scene_aabb.1,
            camera_view_proj_inv: view_proj.inverse(),
            camera_position,
            camera_forward,
            light_dir,
            shadow_map_size: map_size,
            depth_bias,
            csm_near,
            csm_max_distance,
            csm_lambda,
            sharp_distance,
        });

        if fit_log {
            for i in 0..CSM_CASCADE_COUNT as usize {
                let m: glam::Mat4 = glam::Mat4::from_cols_array_2d(
                    &csm.cascades[i].view_proj_inv,
                );
                let p_pos = m * glam::Vec4::new(1.0, 0.0, 0.5, 1.0);
                let p_neg = m * glam::Vec4::new(-1.0, 0.0, 0.5, 1.0);
                let world_pos = p_pos.truncate() / p_pos.w;
                let world_neg = p_neg.truncate() / p_neg.w;
                let half_width = (world_pos - world_neg).length() * 0.5;
                let texel_world = (2.0 * half_width) / map_size as f32;
                eprintln!(
                    "[csm]   cascade {i}: far_view_z = {:.2} m, \
                     half_width = {:.3} m → ~{:.2} mm/texel",
                    csm.cascade_far_view_z[i],
                    half_width,
                    texel_world * 1000.0,
                );
            }
        }

        // Single write of the consolidated 672-byte CSM uniform.
        state.queue.write_buffer(
            &vr.shadow_map.uniform_buffer,
            0,
            bytemuck::bytes_of(&csm),
        );

        // Per-cascade synthetic LOD-select camera. The shader picks
        // the ortho admit path from view_proj[3][3], so each cascade
        // gets its own CameraUniforms-shaped buffer.
        for i in 0..CSM_CASCADE_COUNT {
            vr.write_mesh_lod_shadow_camera(
                &state.queue,
                i,
                &csm.cascades[i as usize],
            );
        }

        wrote_any = true;
    }
    wrote_any
}

#[cfg(test)]
#[path = "frame_helpers_tests.rs"]
mod tests;
