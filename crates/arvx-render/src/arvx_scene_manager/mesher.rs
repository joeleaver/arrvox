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
//! ## The extraction seam ([`SurfaceMesher`], GPU-mesher layer A)
//!
//! The one operation that genuinely differs between a CPU and a GPU mesher is
//! the *extraction* itself: octree/brick occupancy in a region → triangle
//! verts and indices. Everything else (filtering stale clusters, appending
//! patches, the LOD-dirty CC walk) is mesh bookkeeping that stays CPU-side.
//!
//! [`SurfaceMesher`] isolates that step behind a trait. `Mesher` holds a
//! `Box<dyn SurfaceMesher>` and the orchestrators call `extract_region` /
//! `extract_full` through it. The only impl today is [`CpuSurfaceNets`] (wraps
//! the `arvx_core::mesh_extract` free fns + owns the reusable scratch). A GPU
//! sibling (`proc_surface_nets.wesl` minus its classify body, device/queue as
//! its own fields) is layer B — it satisfies the same trait without leaking any
//! wgpu type across the boundary, because the args are plain POD slices.

use glam::{IVec3, Vec3};

use arvx_core::mesh_extract::{CellMap, SculptExtractScratch};
use arvx_core::{BoneVoxel, LeafAttr};

use crate::mesh_pass::MeshVertex;

/// Inputs an *incremental region* extract needs beyond the swappable mesher's
/// own state. A borrowed POD bundle — no wgpu, no scratch, no `MeshView` — so
/// the same call satisfies the CPU impl (walks the slices) and a future GPU
/// impl (uploads them). Adding a field here doesn't churn the trait or its
/// call sites.
pub(super) struct RegionExtractArgs<'a> {
    /// Region cell-occupancy map, built by `collect_cell_map_in_region`. This
    /// is CPU-impl input — a GPU impl builds occupancy on-device and ignores
    /// it. Kept here (rather than behind the trait) so layer A is a pure
    /// mechanical move; layer B can drop the field.
    pub(super) cells: &'a CellMap,
    /// Half-open grid-cell extract span the surface walk iterates.
    pub(super) region_min: IVec3,
    pub(super) region_max: IVec3,
    pub(super) octree_nodes: &'a [u32],
    pub(super) octree_depth: u8,
    pub(super) base_voxel_size: f32,
    pub(super) grid_origin: Vec3,
    pub(super) brick_cells: &'a [u32],
    pub(super) leaf_attr_pool: &'a [LeafAttr],
    pub(super) bone_voxel_pool: &'a [BoneVoxel],
    /// Per-slot signed distances (voxel units), parallel to `leaf_attr_pool`
    /// (`LeafAttrPool::dists_as_slice()`). Non-empty selects the QEF-Hermite
    /// re-extract so a sculpted/refreshed region matches the QEF base; `&[]`
    /// keeps the blur path.
    pub(super) dists: &'a [i16],
    pub(super) halo_cells: &'a [(IVec3, u32)],
    pub(super) sculpt_slots: Option<&'a rustc_hash::FxHashSet<u32>>,
}

/// Inputs a *whole-asset* extract needs. Distinct from [`RegionExtractArgs`]:
/// no region bounds (it covers the full grid) and a halo width (`0` for the
/// sculpt full rebuild, the tile halo for terrain).
pub(super) struct FullExtractArgs<'a> {
    pub(super) octree_nodes: &'a [u32],
    pub(super) octree_depth: u8,
    pub(super) base_voxel_size: f32,
    pub(super) grid_origin: Vec3,
    pub(super) brick_cells: &'a [u32],
    pub(super) leaf_attr_pool: &'a [LeafAttr],
    pub(super) bone_voxel_pool: &'a [BoneVoxel],
    pub(super) halo_cells: &'a [(IVec3, u32)],
    pub(super) halo: u32,
    pub(super) sculpt_slots: Option<&'a rustc_hash::FxHashSet<u32>>,
}

/// The CPU-vs-GPU-swappable surface extraction step. Holds no `MeshView` and
/// performs no view mutation — that is [`super::cluster_delta::ClusterDelta`]'s
/// job. `&mut self` so an impl can own reusable scratch / GPU buffers.
///
/// `Send + Sync` so `Mesher` (and thus `ArvxSceneManager`) keeps the auto
/// traits the old `scratch`-field struct had — `arvx-engine` moves the scene
/// manager across threads on the off-thread build/load path. `CpuSurfaceNets`
/// satisfies it (its scratch is `Send + Sync`); a future GPU impl holds
/// `Arc<Device>`/`Arc<Queue>`, which are too.
pub(super) trait SurfaceMesher: Send + Sync {
    /// Incremental region extract — the 3 orchestrators' inner-region call.
    /// Returns object-local verts + indices LOCAL to those verts (index 0 =
    /// first returned vert).
    fn extract_region(&mut self, args: &RegionExtractArgs<'_>) -> (Vec<MeshVertex>, Vec<u32>);

    /// Whole-asset extract — the 2 full-rebuild orchestrators' call.
    fn extract_full(&mut self, args: &FullExtractArgs<'_>) -> (Vec<MeshVertex>, Vec<u32>);
}

/// The CPU surface-nets mesher. Wraps the `arvx_core::mesh_extract` free fns
/// and owns the reusable region-extract scratch.
pub(super) struct CpuSurfaceNets {
    /// Reusable working buffers for the region-scoped incremental re-extracts.
    /// The full-asset extracts allocate their own and don't touch it.
    scratch: SculptExtractScratch,
}

impl CpuSurfaceNets {
    pub(super) fn new() -> Self {
        Self {
            scratch: SculptExtractScratch::new(),
        }
    }
}

impl SurfaceMesher for CpuSurfaceNets {
    fn extract_region(&mut self, a: &RegionExtractArgs<'_>) -> (Vec<MeshVertex>, Vec<u32>) {
        arvx_core::mesh_extract::extract_mesh_region_from_cells_pooled_haloed(
            &mut self.scratch,
            a.cells,
            a.region_min,
            a.region_max,
            a.octree_nodes,
            a.octree_depth,
            a.base_voxel_size,
            a.grid_origin,
            a.brick_cells,
            a.leaf_attr_pool,
            a.bone_voxel_pool,
            a.halo_cells,
            a.sculpt_slots,
            // The brush-projection `sdf_fn` path was deleted in A4; always
            // `None` now.
            None::<&fn(Vec3) -> f32>,
            a.dists,
        )
    }

    fn extract_full(&mut self, a: &FullExtractArgs<'_>) -> (Vec<MeshVertex>, Vec<u32>) {
        // `extract_surface_mesh_haloed` with `halo = 0` is bit-identical to
        // the old `extract_surface_mesh`, so both full rebuilds share it.
        arvx_core::mesh_extract::extract_surface_mesh_haloed(
            a.octree_nodes,
            a.octree_depth,
            a.base_voxel_size,
            a.grid_origin,
            a.brick_cells,
            a.leaf_attr_pool,
            a.bone_voxel_pool,
            a.halo_cells,
            a.halo,
            a.sculpt_slots,
        )
    }
}

/// Owns the swappable [`SurfaceMesher`] and hosts the re-extract methods (in
/// the domain files). Held by [`super::ArvxSceneManager`]; the manager's thin
/// wrappers field-destructure `self` into disjoint borrows and delegate here.
pub(super) struct Mesher {
    pub(super) surface_mesher: Box<dyn SurfaceMesher>,
}

impl Mesher {
    pub(super) fn new() -> Self {
        Self {
            surface_mesher: Box::new(CpuSurfaceNets::new()),
        }
    }
}
