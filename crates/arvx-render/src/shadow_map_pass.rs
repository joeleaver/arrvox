//! Directional shadow-map storage.
//!
//! Owns the atomic-u32-backed `shadow_buffer` (where `arvx_shade`
//! samples shadow depth) and the per-frame `LightCameraCsm` uniform
//! that drives the per-pixel CSM cascade selection.
//!
//! Historical: this module used to also drive a work-list scatter
//! compute chain (clear / setup / emit / finalize / scatter) for the
//! march path. The mesh path replaces all that with
//! `mesh_shadow_map_pass`: depth raster from the light POV plus a
//! blit compute that copies per-cascade depth into `shadow_buffer`.
//! After the march retirement only the buffer + uniform live here.
//!
//! ## Module layout
//!
//! - [`types`] — `LightCameraUniform` / `LightCameraCsm` wire format,
//!   `CSM_CASCADE_COUNT`, default map size and depth sentinels.
//! - [`light_camera`] — pure CPU math: `compute_csm_cascades`,
//!   `compute_light_camera`, `compute_light_camera_frustum_fit`.
//! - [`pass`] — `ShadowMapPass`: `shadow_buffer` + `uniform_buffer`
//!   storage + resize.

pub mod light_camera;
pub mod pass;
pub mod types;

// Public re-exports — `arvx_render::shadow_map_pass::Foo` stays stable.
pub use light_camera::{
    compute_csm_cascades, compute_light_camera, compute_light_camera_frustum_fit, CsmInputs,
};
pub use pass::ShadowMapPass;
pub use types::{
    CSM_CASCADE_COUNT, LightCameraCsm, LightCameraUniform, SHADOW_FAR_DISTANCE,
    SHADOW_MAP_DEFAULT_SIZE, SHADOW_MAP_FAR_DEPTH, SHADOW_MAP_FAR_DEPTH_BITS,
};

#[cfg(test)]
#[path = "shadow_map_pass/tests.rs"]
mod tests;
