//! Splat-rasterizer prototype path. Walks an asset's octree once at load
//! time, emits one [`extract::SplatVertex`] per occupied surface voxel,
//! and (Phase B) rasterizes the resulting vertex buffer as oriented
//! disc splats — one per voxel, sized to the cell, oriented to the
//! prefiltered surface normal in `LeafAttr.normal_oct`.
//!
//! Phase A: extract module + CPU tests, validates the leaf walk.
//! Phase B: GPU pipeline + integration with the editor's render path.
//!
//! This is a measurement prototype — the goal is empirical perf data
//! against the existing per-pixel `octree_march` path. See the session
//! memory `project_splat_prototype` for findings.

pub mod extract;
pub mod pass;

pub use extract::{
    extract_splats, extract_splats_with_radius, SplatVertex, DISC_RADIUS_FACTOR,
};
pub use pass::{SplatInstanceUniform, SplatPass, SPLAT_INSTANCE_BYTES};
