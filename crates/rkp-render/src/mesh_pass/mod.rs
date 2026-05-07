//! Surface-mesh path — Phase 1 of the splat-to-mesh pivot.
//!
//! At asset load, walk the octree once and emit a triangle mesh
//! `(vertices, indices)` that the visibility-buffer raster (Phase 2)
//! will draw. Vertices carry a `leaf_attr_id` so the existing
//! splat-resolve compute pass can unpack the prefiltered normal +
//! material straight from `leaf_attr_pool` — same indirection the
//! splat path uses, no new shade-side machinery needed.
//!
//! Phase 1 is CPU-only: no GPU pipeline, no per-asset upload. The
//! `(vertices, indices)` buffer just lives on `AssetEntry` next to
//! `splats` until the Phase 2 forward pipeline is wired in.

pub mod pass;

// Mesh extraction + Karis-Nanite cluster DAG construction now live
// in rkp-core (`mesh_extract`, `mesh_cluster`, `mesh_lod`) so
// `rkp-import` (write-time bake) and the procedural bake path
// (`rkp_core::asset_file::write_artifact_rkp`) can both serialise
// the result into `.rkp` and skip the rebuild at editor load. The
// re-exports here keep call sites in `rkp-render` from churning.
pub use rkp_core::mesh_cluster::{
    cluster_mesh, MeshletCluster, MAX_TRIS_PER_CLUSTER, MAX_VERTS_PER_CLUSTER,
    PARENT_GROUP_ERROR_ROOT,
};
pub use rkp_core::mesh_extract::{extract_surface_mesh, MeshVertex};
pub use rkp_core::mesh_lod::{build_cluster_dag, ClusterDag, LOD_LEVELS};
pub use pass::{MeshDraw, MeshPass};
