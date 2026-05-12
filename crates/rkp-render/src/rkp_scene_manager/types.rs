//! Wire types + asset cache for the scene manager.
//!
//! All public types that callers reference (`FaceInstance`, `AssetHandle`,
//! `AssetInfo`, `SkinBrick`, `SkinningAssetData`, `ReloadResult`,
//! `VoxelizeResult`) live here. Private cache machinery (`AssetEntry`,
//! `AssetCache`) is `pub(super)` so the asset-load impl in
//! [`super::asset_load`] can manipulate it.

use std::collections::HashMap;
use std::path::PathBuf;

use rkp_core::cluster_mesh_data::ClusterMesh;
use rkp_core::{OctreeHandle, SparseOctree};

/// Face instance for CPU-side face emission (legacy — kept for scene loading
/// compatibility; the splat raster pipeline it fed is not dispatched).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FaceInstance {
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    pub voxel_size: f32,
    pub voxel_slot: u32,
    pub packed: u32,
}

/// Opaque handle into the scene's asset cache. Obtained via
/// [`RkpSceneManager::acquire_asset`] and released with
/// [`RkpSceneManager::release_asset`]. Callers must pair acquires with
/// releases — when the last instance drops, the cache deallocates the
/// shared leaf_attr / brick / octree ranges. Not persistable (an index
/// into an in-memory cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle(u32);

impl AssetHandle {
    pub fn raw(self) -> u32 { self.0 }
    /// Build an `AssetHandle` from its raw u32 representation.
    /// Internal-only — used by `iter_loaded_asset_splats` to surface
    /// handles without exposing the private constructor.
    pub(super) fn from_raw(raw: u32) -> Self { AssetHandle(raw) }
}

/// Everything a scene instance needs to render an asset. Returned from
/// both `acquire_asset` (.rkp) and the procedural voxelize_* paths so
/// instance spawning can share one code path downstream.
#[derive(Debug, Clone, Copy)]
pub struct AssetInfo {
    pub spatial: rkp_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkp_core::Aabb,
    /// Entity-local grid origin (`aabb_center - extent/2`). Derived at
    /// load time — .rkp files voxelized before this field existed used
    /// the same formula, so re-deriving reproduces the exact bake.
    pub grid_origin: glam::Vec3,
    pub voxel_count: u32,
    pub leaf_attr_slot_start: u32,
    pub leaf_attr_slot_count: u32,
    /// `true` if this asset has skinning data (bone weights + SkinBricks
    /// + rest bone AABBs) baked in. Caller fetches the full data via
    /// [`RkpSceneManager::skinning_data`].
    pub has_skinning: bool,
}

/// One populated octree brick, with its scene-global id and its origin
/// in finest-voxel grid units. Produced at load time by shifting each
/// baked file-local origin's id by the asset's `scene_brick_offset`.
#[derive(Debug, Clone, Copy)]
pub struct SkinBrick {
    /// Scene-global brick id (matches the ids stored in octree nodes).
    pub brick_id: u32,
    /// Brick corner in finest-voxel grid units.
    pub origin: [u32; 3],
}

/// Per-asset skinning metadata read from the `.rkp`'s skin-meta
/// section. Phase-3 scatter pass consumes this to size + populate the
/// deformed bone field each frame.
#[derive(Debug, Clone, Default)]
pub struct SkinningAssetData {
    /// One entry per populated brick in the asset's octree.
    pub bricks: Vec<SkinBrick>,
    /// Per-bone rest-pose AABB, in object-local voxel space. Index is
    /// the bone id (as stored in per-leaf `BoneVoxel.bone_index`).
    /// Empty AABBs (zero-extent) are sentinels for unused bone slots.
    pub rest_bone_aabbs: Vec<[f32; 6]>,
}

/// One entry in the asset cache: the shared geometry allocations plus
/// a refcount. When `refcount` hits zero, `release_asset` frees the
/// octree / leaf_attr / brick ranges.
pub(super) struct AssetEntry {
    pub(super) path: PathBuf,
    pub(super) refcount: u32,
    pub(super) spatial_handle: OctreeHandle,
    pub(super) voxel_size: f32,
    pub(super) aabb: rkp_core::Aabb,
    pub(super) voxel_count: u32,
    pub(super) leaf_attr_slot_start: u32,
    pub(super) leaf_attr_slot_count: u32,
    pub(super) brick_start: u32,
    pub(super) brick_count: u32,
    /// Populated only when the asset has a `FLAG_HAS_BONES` skin-meta
    /// section. Phase-3 scatter pass reads this to drive the per-frame
    /// bone-field write.
    pub(super) skinning: Option<SkinningAssetData>,
    /// Flattened surface-voxel data for the splat-rasterizer path. One
    /// entry per non-empty, non-interior cell, in **object-local**
    /// coordinates (per-instance world is applied in the splat vertex
    /// shader). Shared across every scene-instance of this asset; the
    /// render side uploads it to a GPU vertex buffer once per geometry
    /// epoch via `RkpRenderer::upload_splats_for_asset`.
    ///
    /// ~32 B per cell. A 2.5 M-cell asset (elephant) carries ~80 MB
    /// resident on the CPU; future optimization may release the Vec
    /// after the GPU buffer is built, but for now we keep it so re-
    /// extraction isn't needed when the GPU side reallocates.
    pub(super) splats: Vec<crate::splat_pass::SplatVertex>,
    /// Surface-mesh vertices from naive surface-nets extraction
    /// (Phase 1 of the splat-to-mesh pivot). Object-local positions on
    /// grid corners; carries oct-packed normal + `leaf_attr_id`. Sized
    /// proportional to surface area, not voxel count — typically a
    /// large fraction of `splats.len()` on assets with thin shells.
    /// Stored alongside `splats` until the Phase 2 forward triangle
    /// pipeline replaces the splat path.
    pub(super) mesh_vertices: Vec<crate::mesh_pass::MeshVertex>,
    /// Triangle indices into `mesh_vertices`.
    ///
    /// **Phase 5** stored only the LOD-0 (finest) cluster-reordered
    /// IBO. **Phase 6.1** grew this to the full DAG: LOD-0 indices
    /// first (in `[0 .. mesh_lod0_index_count)`), then LOD-1, then
    /// LOD-2, … Each [`MeshletCluster`] entry's `index_offset` is
    /// global into this concatenated buffer. The Phase 6.1 dispatch
    /// path renders only the LOD-0 prefix (visuals unchanged); the
    /// upcoming Phase 6.2 indirect dispatch will reference per-LOD
    /// offsets via the cluster table.
    pub(super) mesh_indices: Vec<u32>,
    /// Index count of the LOD-0 prefix in `mesh_indices`. Equal to
    /// `mesh_indices.len()` for empty-DAG assets (single-triangle,
    /// pre-Phase-6 behaviour); otherwise strictly less. Phase 6.1's
    /// `dispatch_mesh` draws `0 .. mesh_lod0_index_count`.
    pub(super) mesh_lod0_index_count: u32,
    /// Per-asset meshlet cluster table — **the full DAG** as of
    /// Phase 6.1, spanning every LOD level the builder reached. Each
    /// cluster carries `lod_level`, `cluster_error`, and
    /// `parent_group_error` so the Phase 6.2 GPU LOD-select compute
    /// pass can apply the Karis selection rule. Phase 6.4: the
    /// shadow path consumes the same DAG with a doubled pixel
    /// threshold (~lod + 1), retiring the previously-dormant voxel-
    /// LOD shadow mesh.
    pub(super) meshlet_clusters: Vec<crate::mesh_pass::MeshletCluster>,
    /// Per-cluster owned mesh data — `cluster_meshes[i]` is the local
    /// vertex + index list for `meshlet_clusters[i]`. **Phase B
    /// R4-proper** source of truth for the asset's mesh: the flat
    /// `mesh_vertices` / `mesh_indices` above are derived by
    /// [`flatten_cluster_meshes`] at load / re-extract time.
    ///
    /// Why per-cluster ownership: sculpt's per-cluster re-extract (R4c)
    /// regenerates one cluster's geometry in isolation by replacing its
    /// entry here, then flattening to rebuild the flat buffers for GPU
    /// upload. The legacy shared-IBO model required shifting every
    /// downstream cluster's `index_offset` on any re-extract — R4-proper
    /// drops that whole class of bookkeeping.
    pub(super) cluster_meshes: Vec<ClusterMesh>,
    /// CPU-side mirror of the asset's octree, retained after upload so
    /// runtime sculpt can mutate it without round-tripping the GPU. Same
    /// node buffer the load path built and uploaded; not memory-cheap on
    /// big assets (~4 B per node + parallel prefilter index), but mesh-
    /// mode sculpt edits can't reconstruct it from the cluster DAG.
    pub(super) cpu_octree: SparseOctree,
}

impl AssetEntry {
    pub(super) fn info(&self) -> AssetInfo {
        // Reconstruct grid origin the same way voxelize_octree does:
        // `aabb_center - extent/2`. Matches the bake-time geometry, so
        // existing .rkp files render identically.
        let extent = (1u32 << self.spatial_handle.depth) as f32
            * self.spatial_handle.base_voxel_size;
        let aabb_center = (self.aabb.min + self.aabb.max) * 0.5;
        let grid_origin = aabb_center - glam::Vec3::splat(extent * 0.5);
        AssetInfo {
            spatial: rkp_core::scene_node::SpatialHandle::Octree {
                root_offset: self.spatial_handle.root_offset,
                len: self.spatial_handle.len,
                depth: self.spatial_handle.depth,
                base_voxel_size: self.spatial_handle.base_voxel_size,
            },
            voxel_size: self.voxel_size,
            aabb: self.aabb,
            grid_origin,
            voxel_count: self.voxel_count,
            leaf_attr_slot_start: self.leaf_attr_slot_start,
            leaf_attr_slot_count: self.leaf_attr_slot_count,
            has_skinning: self.skinning.is_some(),
        }
    }
}

/// Maps file paths to cached asset entries. Keyed on the canonical path
/// that was resolved against the `.rkp` extension, so two different
/// inputs that normalize to the same file share a handle.
#[derive(Default)]
pub(super) struct AssetCache {
    pub(super) entries: Vec<Option<AssetEntry>>,
    pub(super) path_to_handle: HashMap<PathBuf, AssetHandle>,
    pub(super) free_slots: Vec<u32>,
}

impl AssetCache {
    pub(super) fn insert(&mut self, entry: AssetEntry) -> AssetHandle {
        let handle = if let Some(slot) = self.free_slots.pop() {
            self.entries[slot as usize] = Some(entry);
            AssetHandle(slot)
        } else {
            let idx = self.entries.len() as u32;
            self.entries.push(Some(entry));
            AssetHandle(idx)
        };
        self.path_to_handle
            .insert(self.entries[handle.0 as usize].as_ref().unwrap().path.clone(), handle);
        handle
    }

    /// Reserve a handle without backing an `AssetEntry`. Used for
    /// procedural proxy-mesh entities that have GPU mesh buffers
    /// (in `RkpRenderer::mesh_buffers` / `mesh_cluster_buffers`)
    /// but no octree / leaf_attr / brick allocations to refcount —
    /// they aren't shared via path lookup either, since each
    /// procedural owns its own proxy mesh.
    pub(super) fn reserve_handle(&mut self) -> AssetHandle {
        if let Some(slot) = self.free_slots.pop() {
            self.entries[slot as usize] = None;
            AssetHandle(slot)
        } else {
            let idx = self.entries.len() as u32;
            self.entries.push(None);
            AssetHandle(idx)
        }
    }

    /// Release a handle previously reserved with `reserve_handle`.
    /// Pushes the slot onto the free list so the next `insert` /
    /// `reserve_handle` reuses it.
    pub(super) fn release_reserved(&mut self, handle: AssetHandle) {
        if (handle.0 as usize) < self.entries.len() {
            self.entries[handle.0 as usize] = None;
            self.free_slots.push(handle.0);
        }
    }

    pub(super) fn lookup_path(&self, path: &std::path::Path) -> Option<AssetHandle> {
        self.path_to_handle.get(path).copied()
    }

    pub(super) fn get(&self, handle: AssetHandle) -> Option<&AssetEntry> {
        self.entries.get(handle.0 as usize).and_then(|e| e.as_ref())
    }

    pub(super) fn get_mut(&mut self, handle: AssetHandle) -> Option<&mut AssetEntry> {
        self.entries.get_mut(handle.0 as usize).and_then(|e| e.as_mut())
    }

    pub(super) fn remove(&mut self, handle: AssetHandle) -> Option<AssetEntry> {
        let slot = handle.0 as usize;
        let taken = self.entries.get_mut(slot)?.take()?;
        self.path_to_handle.remove(&taken.path);
        self.free_slots.push(handle.0);
        Some(taken)
    }
}

/// Result of [`RkpSceneManager::reload_asset`]. `old_handle` is the handle
/// that was invalidated (so callers can find entities still holding it);
/// `new_handle` points at the freshly-loaded entry. They may be equal when
/// the cache reuses the vacated slot, but callers must not rely on that.
#[derive(Debug, Clone, Copy)]
pub struct ReloadResult {
    pub old_handle: AssetHandle,
    pub new_handle: AssetHandle,
    pub info: AssetInfo,
}

/// Result of voxelizing a primitive.
pub struct VoxelizeResult {
    pub spatial: rkp_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkp_core::Aabb,
    /// Entity-local position where the octree grid starts (the
    /// `aabb_center - extent/2` corner). The shader uses this to
    /// convert world→octree coords, so it must be stored and
    /// propagated all the way to the GPU object.
    pub grid_origin: glam::Vec3,
    /// Logical voxel count (octree leaves).
    pub voxel_count: u32,
    /// First leaf_attr pool slot used by this allocation.
    pub leaf_attr_slot_start: u32,
    /// Number of leaf_attr slots allocated.
    pub leaf_attr_slot_count: u32,
    /// Brick ids owned by this allocation — `deallocate_geometry` frees
    /// them one at a time so procedurals don't leak bricks on
    /// re-voxelize / delete.
    pub brick_ids: Vec<u32>,
}

/// Emit face instances from an octree into the given buffer. Legacy —
/// splat raster is not dispatched in the active pipeline. Kept for
/// scene-loading compatibility: every leaf is a surface voxel now, so the
/// output just enumerates leaf centers with exposed-face flags.
pub(super) fn emit_faces(
    octree: &SparseOctree,
    obj_idx: u32,
    faces: &mut Vec<FaceInstance>,
) {
    let base_vs = octree.base_voxel_size();

    for (coord, leaf_id, leaf_depth) in octree.iter_leaves() {
        let depth_diff = octree.depth() - leaf_depth;
        let leaf_vs = base_vs * (1u32 << depth_diff) as f32;

        let center = glam::Vec3::new(
            coord.x as f32 * base_vs + leaf_vs * 0.5,
            coord.y as f32 * base_vs + leaf_vs * 0.5,
            coord.z as f32 * base_vs + leaf_vs * 0.5,
        );

        let offsets: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        for (face, &(dx, dy, dz)) in offsets.iter().enumerate() {
            let nx = coord.x as i64 + dx as i64;
            let ny = coord.y as i64 + dy as i64;
            let nz = coord.z as i64 + dz as i64;

            let exposed = if nx < 0 || ny < 0 || nz < 0 {
                true
            } else {
                let nc = glam::UVec3::new(nx as u32, ny as u32, nz as u32);
                match octree.lookup(nc) {
                    None => true,
                    Some(node) if node == rkp_core::sparse_octree::EMPTY_NODE => true,
                    Some(node) if node == rkp_core::sparse_octree::INTERIOR_NODE => false,
                    Some(node) if rkp_core::sparse_octree::is_leaf(node) => false,
                    _ => true,
                }
            };

            if exposed {
                let face = face as u32;
                faces.push(FaceInstance {
                    pos_x: center.x,
                    pos_y: center.y,
                    pos_z: center.z,
                    voxel_size: leaf_vs,
                    voxel_slot: leaf_id,
                    packed: (face & 0x7) | ((obj_idx & 0xFFFFF) << 3),
                });
            }
        }
    }
}
