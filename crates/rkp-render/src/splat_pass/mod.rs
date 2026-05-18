//! Shared per-instance render infrastructure for the surface-mesh path.
//!
//! Historical note: this module owned the splat-rasterizer prototype.
//! After the splat→mesh pivot the prototype was retired; the per-instance
//! uniform layout and bind-group layouts proved general-purpose and are
//! still used by the mesh raster, mesh shadow render, mesh LOD select,
//! and user-shader mesh paths. The `splat_*` names will be renamed to
//! `mesh_*` in the cleanup pass.

pub mod pass;

pub use pass::{SplatInstanceUniform, SplatPass, SKINNING_MODE_NONE, SPLAT_INSTANCE_BYTES};

/// One scene-instance to render in this frame's primary-visibility
/// dispatch. The engine populates a `Vec<SplatDraw>` per visible viewport
/// and passes it through to `RkpRenderer::render_to`.
///
/// `asset_handle_raw` is the `AssetHandle::raw()` of the asset to draw —
/// used to look up the per-asset mesh buffers in
/// `RkpRenderer::mesh_buffer`. `world` is the instance's world transform;
/// `object_id` lands in the pick texture so picking works as expected.
///
/// **Skinning fields:** copy of the per-instance state the mesh VS reads
/// via the per-instance uniform. `skinning_mode` is `SKINNING_MODE_NONE`
/// for unskinned instances and most rigid passes; the engine sets it to
/// `0` (LBS) or `1` (DQS) only when this entity has both a live
/// `Skeleton` component and a baked skin-meta payload on the asset.
#[derive(Debug, Clone, Copy)]
pub struct SplatDraw {
    pub asset_handle_raw: u32,
    pub world: [[f32; 4]; 4],
    pub object_id: u32,
    /// Asset's voxel-grid origin in mesh-frame. The mesh VS uses this
    /// to bridge from the vertex's mesh-frame local_pos to the
    /// grid-frame bone matrices want. Zero for assets where grid and
    /// mesh frames coincide (e.g. anything voxelized with origin at
    /// the octree corner).
    pub grid_origin: [f32; 3],
    /// First mat4 in `bone_matrices` for this instance's bones.
    /// Ignored when `skinning_mode != 0`.
    pub bone_offset_lbs: u32,
    /// First DualQuat in `bone_dual_quats` for this instance's bones.
    /// Ignored when `skinning_mode != 1`.
    pub bone_offset_dqs: u32,
    /// `0` = LBS, `1` = DQS, [`SKINNING_MODE_NONE`] = no skinning.
    pub skinning_mode: u32,
    /// Whether this instance contains any transparent (`opacity < 0.99`)
    /// material — either in the asset's `leaf_attr_pool` slice or via
    /// a paint-overlay remap. The mesh-mode primary path uses this
    /// flag to skip the front/back glass raster passes entirely on
    /// instances that can't contribute glass — saves the
    /// triangulation cost on opaque-only assets, which is most of
    /// them. Glass meshes still pay full cost; the FS-side
    /// per-fragment classify catches per-cell glass within those.
    /// Conservative default is `true` (preserves correctness if the
    /// caller hasn't computed it).
    pub has_glass: bool,
}
