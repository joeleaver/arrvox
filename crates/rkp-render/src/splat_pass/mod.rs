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

/// One scene-instance to render in this frame's splat dispatch. The
/// engine populates a `Vec<SplatDraw>` per visible viewport when the
/// primary path is `Splat`, and passes it through to `RkpRenderer::render_to`.
///
/// `asset_handle_raw` is the `AssetHandle::raw()` of the asset to draw —
/// used to look up the per-asset vertex buffer in
/// `RkpRenderer::splat_buffer`. `world` is the instance's world
/// transform; `object_id` lands in the pick texture so picking works
/// the same as the march path.
#[derive(Debug, Clone, Copy)]
pub struct SplatDraw {
    pub asset_handle_raw: u32,
    pub world: [[f32; 4]; 4],
    pub object_id: u32,
}
