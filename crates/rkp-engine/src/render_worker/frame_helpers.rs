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

/// Phase 4 — merge sim's per-tile object lists with two render-side
/// sources: a properly-culled per-tile user-shader-instance list (one
/// entry per (tile, instance) where the instance's screen-AABB
/// overlaps the tile), and a Phase C broadcast list (every tile gets
/// every transient).
///
/// For each tile, output is `[sim_persistent | user_shader_in_tile |
/// transient_broadcast]`. All three index spaces are disjoint
/// (persistent < persistent_count ≤ user-shader < persistent+
/// user_shader ≤ transient < object_count), so the march can dispatch
/// any of them without aliasing.
pub(super) fn merge_tile_lists(
    sim_offsets_bytes: &[u8],
    sim_ids_bytes: &[u8],
    transient_broadcast: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n_tile = if sim_offsets_bytes.is_empty() {
        0
    } else {
        (sim_offsets_bytes.len() / 4).saturating_sub(1)
    };
    if n_tile == 0 {
        return (
            bytemuck::cast_slice::<u8, u32>(sim_offsets_bytes).to_vec(),
            bytemuck::cast_slice::<u8, u32>(sim_ids_bytes).to_vec(),
        );
    }
    let sim_offsets: &[u32] = bytemuck::cast_slice(sim_offsets_bytes);
    let sim_ids: &[u32] = bytemuck::cast_slice(sim_ids_bytes);

    let mut new_offsets: Vec<u32> = Vec::with_capacity(n_tile + 1);
    let mut new_ids: Vec<u32> =
        Vec::with_capacity(sim_ids.len() + n_tile * transient_broadcast.len());
    new_offsets.push(0);
    for t in 0..n_tile {
        let sa = sim_offsets[t] as usize;
        let sb = sim_offsets[t + 1] as usize;
        new_ids.extend_from_slice(&sim_ids[sa..sb]);
        new_ids.extend_from_slice(transient_broadcast);
        new_offsets.push(new_ids.len() as u32);
    }
    (new_offsets, new_ids)
}

/// Phase 5.2 — derive a conservative scene AABB for the GPU TLAS
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

/// Phase 8 V3 disabled — voxel-stepped shadow-map silhouettes were
/// worse quality than `rkp_shadow_trace`'s per-pixel ray-traced path,
/// and the floor's per-texel descent cost made the pipeline slower
/// than the path it was meant to replace. Always returning `false`
/// here skips the shadow_map dispatch chain, leaves
/// `ShadeParams.shadow_map_enabled = 0`, and routes directional
/// lights through `shadow_data[]` — the same path point/spot lights
/// already use.
///
/// All shadow_map_pass.rs + scatter shader code is kept in tree in
/// case a follow-up revisits with a fundamentally different shadow
/// representation. To re-enable: replace this with the frustum-fit
/// walk from commit 0a6aeed.
pub(super) fn prepare_shadow_maps(
    state: &mut RenderState,
    frame: &RenderFrame,
    scene_aabb: ([f32; 3], [f32; 3]),
    _tlas_prim_count: u32,
) -> bool {
    // The voxel-march scatter shadow_map remains disabled (the
    // commit-comment context above still applies — it produced
    // worse silhouettes than rkp_shadow_trace at higher cost).
    //
    // The mesh path needs an actual depth-rendered shadow map
    // though, and it has the right input data — real triangle
    // geometry. Re-enable the per-VR LightCameraUniform write
    // for mesh mode; `MeshShadowMapPass` is what consumes it.
    let mesh_mode = matches!(
        state.renderer.primary_mode,
        rkp_render::rkp_renderer::PrimaryMode::Mesh,
    );
    if !mesh_mode {
        return false;
    }
    use rkp_render::shadow_map_pass::{
        compute_csm_cascades, CsmInputs, CSM_CASCADE_COUNT, SHADOW_MAP_DEFAULT_SIZE,
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
            shadow_map_size: SHADOW_MAP_DEFAULT_SIZE,
            depth_bias,
            csm_near,
            csm_max_distance,
            csm_lambda,
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
                let texel_world = (2.0 * half_width)
                    / SHADOW_MAP_DEFAULT_SIZE as f32;
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
