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
    use rkp_render::shadow_map_pass::{compute_light_camera, SHADOW_MAP_DEFAULT_SIZE};
    // Pick the first directional caster — `position[3] == 0` flags
    // a directional light in the GpuLight wire format, and
    // `params[3] >= 0.5` is the shadow-caster bit. Multi-caster /
    // CSM is future work.
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
    // Small bias to avoid classic shadow-map self-shadow acne. Front-
    // face cull (in `MeshShadowMapPass`) does most of the heavy
    // lifting; this just covers the residual when the lit surface
    // sits very close to its own back-face shadow depth.
    let depth_bias = 0.001;

    // `scene_aabb` here is `compute_tlas_scene_aabb`'s output — the
    // tightest world AABB of the visible casters. Passing it to
    // `compute_light_camera` gives a caster-fit orthographic light
    // camera, which is dramatically tighter than the camera-frustum
    // fit that V1 used. For a 1 m elephant 5 m from camera the xy
    // bounds collapse from ~30 m → ~2 m, taking shadow texels from
    // ~3 cm → ~2 mm and erasing the stair-stepping. Trade-off:
    // shadow casters that don't visually overlap the camera view
    // will still cast (the AABB is whole-scene), but casters
    // entirely outside the AABB won't reach the map. Acceptable
    // when the AABB is the union of all live casters.
    let uniform = compute_light_camera(
        scene_aabb.0,
        scene_aabb.1,
        light_dir,
        SHADOW_MAP_DEFAULT_SIZE,
        depth_bias,
    );
    // Per-frame shadow-fit diagnostic. Helps diagnose "blocky
    // shadows" reports — if the scene AABB is large the 1024²
    // shadow map's texel coverage is intrinsically chunky
    // regardless of the rasterizer.
    if std::env::var("RKP_SHADOW_FIT_LOG").is_ok() {
        let dx = scene_aabb.1[0] - scene_aabb.0[0];
        let dy = scene_aabb.1[1] - scene_aabb.0[1];
        let dz = scene_aabb.1[2] - scene_aabb.0[2];
        let max_extent = dx.max(dy).max(dz);
        let texel_world = max_extent / SHADOW_MAP_DEFAULT_SIZE as f32;
        eprintln!(
            "[shadow] scene_aabb extent = {:.2} × {:.2} × {:.2} m → max ~{:.1} m / {} = ~{:.0} mm per shadow-map texel",
            dx, dy, dz, max_extent, SHADOW_MAP_DEFAULT_SIZE, texel_world * 1000.0,
        );
    }

    let mut wrote_any = false;
    for vp in &frame.viewports {
        let Some(vr) = state.viewport_renderers.get_mut(&vp.id) else {
            continue;
        };
        // Per-VR write — same uniform across all viewports because
        // the caster fit is camera-independent. Multi-viewport CSM
        // would diverge here per-VR.
        state.queue.write_buffer(
            &vr.shadow_map.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniform),
        );
        // Phase 6.4: also populate the per-VR shadow LOD-select
        // camera buffer (CameraUniforms-shaped) from the same
        // LightCameraUniform so the LOD compute pass picks
        // shadow-side admit decisions one LOD coarser than primary.
        vr.write_mesh_lod_shadow_camera(&state.queue, &uniform);
        wrote_any = true;
    }
    wrote_any
}

#[cfg(test)]
#[path = "frame_helpers_tests.rs"]
mod tests;
