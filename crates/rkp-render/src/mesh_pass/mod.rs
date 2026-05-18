//! Surface-mesh path.
//!
//! At asset load, walk the octree once and emit a triangle mesh
//! `(vertices, indices)` that the visibility-buffer raster will draw.
//! Vertices carry a `leaf_attr_id` so the `mesh_resolve` compute pass
//! unpacks the prefiltered normal + material straight from
//! `leaf_attr_pool` — same indirection at shade time, no per-pixel
//! gradient reconstruction.

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
pub use pass::MeshPass;

pub use crate::mesh_instance::MeshDraw;
