//! RKP-Render: Splat rendering pipeline.
//!
//! Replaces rkf-render's ray march with a fixed-step march through the opacity
//! field. All other passes (shading, shadows, GI, post-process) are reused from
//! rkf-render via direct dependency.
//!
//! The only new pass is [`SplatMarchPass`] — everything else is orchestration.

/// Splat march compute pass — surface-finding through opacity field, G-buffer output.
pub mod splat_march;
/// Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
pub mod voxelize_opacity;

pub use splat_march::SplatMarchPass;
pub use voxelize_opacity::import_mesh_to_opacity_rkf;
