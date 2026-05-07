//! Phase 8 — directional shadow maps (V2: work-list scatter).
//!
//! Replaces the per-pixel ray-traced shadow path with a four-pass
//! scatter pipeline that lays the geometry down into a shared
//! depth buffer instead of marching rays from the light.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — constants + `LightCameraUniform` + `SetupParams`
//!   wire-format types.
//! - [`light_camera`] — pure CPU math: `compute_light_camera` +
//!   `compute_light_camera_frustum_fit`.
//! - [`pass`] — `ShadowMapPass` GPU runtime: pipelines, buffers, and
//!   the per-frame dispatch chain (clear → setup → emit → finalize →
//!   scatter). Plus internal `ShadowScatterMarchParams` + bind-group
//!   layout helpers.
//!
//! ## Why work-list scatter
//!
//! V1 of this pass marched rays from the light's POV (one per
//! shadow-map texel). V1.5 of the scatter approach indirectly
//! dispatched ONCE per TLAS leaf — fast in theory, but the CPU
//! loop's `set_bind_group + dispatch_workgroups_indirect` per
//! instance hit ~5–10 µs of driver overhead each, and dense-grass
//! scenes (1000+ instances) burned multiple ms in dispatch
//! overhead alone.
//!
//! V2 collapses every per-instance dispatch into ONE indirect
//! scatter dispatch over a global work list. Setup pass projects
//! each TLAS prim's AABB to a tile rect, atomic-adds its tile
//! count to a global counter (capturing the per-instance offset).
//! Emit pass parallel-fills `work_list` with packed (instance,
//! tile_x_local, tile_y_local) tuples — workgroups parallelize
//! per instance, threads parallelize across that instance's tiles.
//! Finalize pass converts the total work count into 2D dispatch
//! args. Scatter pass dispatches ONCE indirectly; each workgroup
//! reads its work-list entry, descends the indicated instance for
//! its 8×8 tile, atomic-mins depth into `shadow_buffer`.
//!
//! Per-frame dispatch count: 5 (clear / setup / emit / finalize /
//! scatter), regardless of scene complexity.
//!
//! ## V1 limitations (carry from V1)
//!
//! * **Directional only.** Spot/point lights still use the
//!   ray-traced shadow path.
//! * **Single shadow map.** No CSM yet; one map covers the whole
//!   scene's projected light-space AABB.
//! * **Hard shadows.** No PCF / VSM; just a depth compare.
//! * **Opacity ignored.** Every voxel counts as opaque.

pub mod light_camera;
pub mod pass;
pub mod types;

// Public re-exports — keep `rkp_render::shadow_map_pass::Foo` stable.
pub use light_camera::{
    compute_csm_cascades, compute_light_camera, compute_light_camera_frustum_fit, CsmInputs,
};
pub use pass::ShadowMapPass;
pub use types::{
    CSM_CASCADE_COUNT, LightCameraCsm, LightCameraUniform, SetupParams,
    SCATTER_INSTANCE_STRIDE, SHADOW_FAR_DISTANCE, SHADOW_MAP_DEFAULT_SIZE,
    SHADOW_MAP_FAR_DEPTH, SHADOW_MAP_FAR_DEPTH_BITS, SHADOW_MAP_MAX_CASTERS_INITIAL,
    SHADOW_MAP_WORK_LIST_INITIAL, SHADOW_SCATTER_DISPATCH_X,
};

#[cfg(test)]
#[path = "shadow_map_pass/tests.rs"]
mod tests;
