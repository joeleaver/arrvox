//! `Mesher` — the surface re-extract, freed from `ArvxSceneManager`.
//!
//! The [`super::types::VoxelModel`] / [`super::types::MeshView`] split made
//! the voxel/mesh boundary a type; this module makes the *meshing* a near-pure
//! operation over that boundary. Every re-extract path (the two
//! `remesh_region` executors, the three incremental orchestrators, and the two
//! full-asset rebuilds) runs through `Mesher` on explicit inputs —
//! `(&VoxelModel, &mut MeshView)` plus the scene-global pools meshing must read
//! (`&BrickPool` for occupancy) and read/write (`&mut LeafAttrPool` for the ∇D
//! per-leaf normals). The `Mesher` never sees an [`super::types::AssetHandle`],
//! the asset cache, or the geometry epoch — the scene manager fetches the entry,
//! lends the pools, and bumps the epoch around the call.
//!
//! The `impl Mesher` blocks live in the domain files (`remesh_region.rs` for the
//! executors, `sculpt.rs` for the sculpt orchestrators, `terrain_halo_refresh.rs`
//! for the terrain ones) so the meshing code stays next to its domain context.
//!
//! Concrete CPU surface-nets mesher for now. A `trait` with a GPU sibling impl
//! (`proc_surface_nets.wesl` minus its classify body) and a bundled
//! `ClusterDelta` return are the GPU-endgame follow-on — premature with a single
//! impl and no delta consumer.

use arvx_core::mesh_extract::SculptExtractScratch;

/// Owns the reusable surface-nets extraction scratch buffer and hosts the
/// re-extract methods. Held by [`super::ArvxSceneManager`] (replacing the old
/// `sculpt_extract_scratch` field); the manager's thin wrappers field-destructure
/// `self` into disjoint borrows and delegate here.
pub(super) struct Mesher {
    /// Reusable working buffers for the region-scoped incremental re-extracts
    /// (`rebuild_dirty_clusters` / `rebuild_stroke_clusters` /
    /// `rebuild_face_band_clusters`). The full-asset rebuilds and the executors
    /// don't use it.
    pub(super) scratch: SculptExtractScratch,
}

impl Mesher {
    pub(super) fn new() -> Self {
        Self {
            scratch: SculptExtractScratch::new(),
        }
    }
}
