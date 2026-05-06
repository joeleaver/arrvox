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
pub use extract::{extract_shadow_mesh_lod, extract_surface_mesh, MeshVertex};
pub use lod::{build_cluster_dag, ClusterDag, LOD_LEVELS};
pub use pass::{MeshDraw, MeshPass};

/// LOD level used for the shadow-mesh path.
///
/// `lod_levels = 1` puts shadow cells at `2 × finest_voxel_size`. On
/// a 5 mm-finest asset that's ~10 mm shadow cells — fine enough that
/// the LOD cube tessellation isn't visually obvious in the cast
/// shadow but coarse enough to drop triangle count by ~8× vs. finest.
///
/// `walk_collect_cells_lod` subdivides bricks at this level so the
/// LOD mesh tracks each brick's actual occupancy instead of inflating
/// to a full cube — without that step the shadow shows axis-aligned
/// brick-cube silhouettes wherever the surface passes through a
/// brick. Coarser values (2, 3, …) skip the subdivision and produce
/// progressively blockier shadow silhouettes.
pub const SHADOW_LOD_LEVELS: u8 = 1;
