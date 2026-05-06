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

pub mod cluster;
pub mod extract;
pub mod lod;
pub mod pass;

pub use cluster::{
    cluster_mesh, MeshletCluster, MAX_TRIS_PER_CLUSTER, MAX_VERTS_PER_CLUSTER,
    PARENT_GROUP_ERROR_ROOT,
};
pub use extract::{extract_surface_mesh, MeshVertex};
pub use lod::{build_cluster_dag, ClusterDag, LOD_LEVELS};
pub use pass::{MeshDraw, MeshPass};
