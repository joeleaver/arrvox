//! Wire types + asset cache for the scene manager.
//!
//! All public types that callers reference (`FaceInstance`, `AssetHandle`,
//! `AssetInfo`, `SkinningAssetData`, `ReloadResult`,
//! `VoxelizeResult`) live here. Private cache machinery (`AssetEntry`,
//! `AssetCache`) is `pub(super)` so the asset-load impl in
//! [`super::asset_load`] can manipulate it.

use std::collections::HashMap;
use std::path::PathBuf;

use arvx_core::{DirtyRanges, OctreeHandle, SparseOctree};

/// Byte stride of one [`u32`] index inside `mesh_indices`. The slab
/// allocator works in element units; the [`DirtyRanges`] tracker
/// records byte offsets so it can drive `queue.write_buffer` directly.
pub(super) const MESH_INDEX_STRIDE: u32 = 4;

/// Face instance for CPU-side face emission (legacy — kept for scene
/// loading compatibility; nothing dispatches against this anymore).
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
/// [`ArvxSceneManager::acquire_asset`] and released with
/// [`ArvxSceneManager::release_asset`]. Callers must pair acquires with
/// releases — when the last instance drops, the cache deallocates the
/// shared leaf_attr / brick / octree ranges. Not persistable (an index
/// into an in-memory cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle(u32);

impl AssetHandle {
    pub fn raw(self) -> u32 { self.0 }
    /// Build an `AssetHandle` from its raw u32 representation.
    /// Internal-only.
    #[allow(dead_code)]
    pub(super) fn from_raw(raw: u32) -> Self { AssetHandle(raw) }
}

/// Everything a scene instance needs to render an asset. Returned from
/// both `acquire_asset` (.arvx) and the procedural voxelize_* paths so
/// instance spawning can share one code path downstream.
#[derive(Debug, Clone, Copy)]
pub struct AssetInfo {
    pub spatial: arvx_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: arvx_core::Aabb,
    /// Entity-local grid origin (`aabb_center - extent/2`). Derived at
    /// load time — .arvx files voxelized before this field existed used
    /// the same formula, so re-deriving reproduces the exact bake.
    pub grid_origin: glam::Vec3,
    pub voxel_count: u32,
    pub leaf_attr_slot_start: u32,
    pub leaf_attr_slot_count: u32,
    /// `true` if this asset has skinning data (bone weights + rest
    /// bone AABBs) baked in. Caller fetches the full data via
    /// [`ArvxSceneManager::skinning_data`].
    pub has_skinning: bool,
}

/// Per-asset skinning metadata read from the `.arvx`'s skin-meta
/// section. The runtime only uses `rest_bone_aabbs.len()` as the
/// asset's bone count today — the cluster-AABB expansion at bake
/// time also uses the AABBs themselves, but that lives in `arvx-core`.
#[derive(Debug, Clone, Default)]
pub struct SkinningAssetData {
    /// Per-bone rest-pose AABB, in object-local voxel space. Index is
    /// the bone id (as stored in per-leaf `BoneVoxel.bone_index`).
    /// Empty AABBs (zero-extent) are sentinels for unused bone slots.
    pub rest_bone_aabbs: Vec<[f32; 6]>,
}

/// One entry in the asset cache: the [`VoxelModel`] source-of-truth, its
/// derived [`MeshView`], plus a refcount. When `refcount` hits zero,
/// `release_asset` frees the octree / leaf_attr / brick ranges.
///
/// The split is the boundary between voxel ops and meshing
/// ([`VoxelModel`] = truth, [`MeshView`] = disposable derived view): the
/// re-extract paths read `&model` and rebuild `&mut view`, the shape the
/// future `Mesher::remesh(&VoxelModel, &[RemeshRegion], &mut MeshView)`
/// consumes. `path` / `refcount` are cache bookkeeping — neither truth
/// nor derived mesh.
pub(super) struct AssetEntry {
    pub(super) path: PathBuf,
    pub(super) refcount: u32,
    /// Voxel source-of-truth: octree, pool allocations, halo, material
    /// set. The only half a voxel edit mutates; never names a triangle.
    pub(super) model: VoxelModel,
    /// Derived surface-mesh view: vertices, indices + slab allocator,
    /// meshlet clusters, DAG, dirty trackers, spatial index. Rebuilt
    /// from [`VoxelModel`] by the re-extract paths; never edited by
    /// voxel code.
    pub(super) view: MeshView,
}

/// The voxel source-of-truth for one cached asset: the CPU octree
/// mirror, its leaf_attr / brick pool allocations, the terrain halo, and
/// the material set. The only state a voxel edit (sculpt / halo refresh)
/// mutates directly; the mesher reads it shared (`&VoxelModel`) and never
/// writes it. Carries no triangle, index, or cluster — those are derived
/// into the companion [`MeshView`].
pub(super) struct VoxelModel {
    pub(super) spatial_handle: OctreeHandle,
    pub(super) voxel_size: f32,
    pub(super) aabb: arvx_core::Aabb,
    pub(super) voxel_count: u32,
    pub(super) leaf_attr_slot_start: u32,
    pub(super) leaf_attr_slot_count: u32,
    pub(super) brick_start: u32,
    pub(super) brick_count: u32,
    /// Populated only when the asset has a `FLAG_HAS_BONES` skin-meta
    /// section. Phase-3 scatter pass reads this to drive the per-frame
    /// bone-field write.
    pub(super) skinning: Option<SkinningAssetData>,
    /// CPU-side mirror of the asset's octree, retained after upload so
    /// runtime sculpt can mutate it without round-tripping the GPU. Same
    /// node buffer the load path built and uploaded; not memory-cheap on
    /// big assets (~4 B per node + parallel prefilter index), but mesh-
    /// mode sculpt edits can't reconstruct it from the cluster DAG.
    pub(super) cpu_octree: SparseOctree,
    /// Leaf-attr slots allocated for this asset by sculpt **after**
    /// integration — i.e., not part of the contiguous bump range at
    /// `[leaf_attr_slot_start, leaf_attr_slot_start + leaf_attr_slot_count)`.
    ///
    /// Sculpt's Add edits (Raise / Inflate / Deflate cavity walls /
    /// material-replace) request fresh slots via the pool's general
    /// `allocate()`, which can land anywhere in the global pool. The
    /// asset's static range can't grow, so we track these here. Two
    /// consumers care:
    ///
    /// 1. `apply_paint_sphere`'s slot validation — paint after sculpt
    ///    on a tile (or any asset) was silently dropping every hit
    ///    whose `leaf_slot` was outside the base range. Phase 4
    ///    flushed this out on terrain ("paint stops working after
    ///    sculpt") but the underlying bug pre-dates terrain.
    /// 2. `release_asset` — bake-range slots get freed by
    ///    `deallocate_range(slot_start, slot_count)`; these need
    ///    individual `deallocate_range(slot, 1)` calls.
    ///
    /// Stored as a `HashSet<u32>` so freed-then-reallocated slots
    /// don't accumulate duplicates that would double-free on release.
    pub(super) sculpt_extra_slots: std::collections::HashSet<u32>,
    /// Every leaf-attr slot the sculpt brush has allocated for this
    /// asset (superset of `sculpt_extra_slots` — includes both
    /// out-of-bake-range AND reused in-bake-range slots). Used by
    /// `build_cube_vertex`'s sculpt-bias tie-break: when an SN cube
    /// has corner cells from both sculpt and pre-existing surface,
    /// the per-vertex `leaf_attr_id` (which drives material + color)
    /// prefers the sculpt slot. Without this bias the lowest-coord
    /// corner wins purely by position, so sculpt cells sometimes
    /// inherit a procedural neighbour's material and the brush
    /// material disappears in a position-dependent pattern.
    ///
    /// Populated on every sculpt-allocated slot in `apply_delta`'s
    /// post-write loop; entries removed when slots are freed (so the
    /// set tracks only currently-live sculpt cells). Empty for
    /// assets that have never been sculpted — those paths see the
    /// original `coord_less`-only behaviour.
    pub(super) sculpt_owned_slots: rustc_hash::FxHashSet<u32>,
    /// Phase 4.2b: leaf-attr slots allocated specifically for new
    /// halo cells discovered during cross-tile halo refresh. The
    /// bake-time halo's slots live inside the asset's contiguous
    /// `[slot_start, +slot_count)` range; cells that flipped
    /// empty→solid on the neighbour AFTER bake need fresh slots
    /// allocated from the pool's general `allocate()`, which can
    /// land anywhere. Tracked here so `release_asset` frees them
    /// individually (the contiguous deallocate only covers the
    /// bake range). HashSet because the same slot can be freed and
    /// re-allocated within a session.
    pub(super) halo_extra_slots: std::collections::HashSet<u32>,
    /// Terrain Phase 4: per-cell halo data carried from the original
    /// bake.
    ///
    /// Each entry maps an octree-frame coord OUTSIDE the nominal
    /// `[0, S)³` cube to a `leaf_attr_id` (or [`CELL_INTERIOR`] for
    /// halo cells classified bulk-solid). For terrain tiles this is
    /// populated by `integrate_baked_tile` (the slot ids are scene-
    /// pool-relocated to match the asset's leaf-attr range). For
    /// disk-loaded non-terrain assets the vec is empty — they have
    /// no halo by construction.
    ///
    /// **Used by sculpt:** the per-cluster re-extract folds these
    /// cells into the local cell grid so SN-cubes at a tile boundary
    /// see valid 8-corner classification. Without this, sculpting a
    /// boundary cluster regresses its seam quads — the original bake
    /// had halo data, the re-extract didn't, the new cluster's
    /// boundary cubes diverge from the neighbour's.
    pub(super) halo_cells: Vec<(glam::IVec3, u32)>,
    /// The complete deduped set of project `material_primary` IDs across
    /// this asset's leaves + prefilter attrs, when known — the runtime
    /// material authority (same IDs the per-leaf `LeafAttr` carries). Lets
    /// the engine answer `has_glass` in O(distinct) without a leaf walk.
    ///
    /// `None` means "not computed for this integration path" — the engine
    /// falls back to the per-leaf walk (correct, just not the fast path).
    /// Only the off-thread `.arvx` load populates `Some`; terrain-tile and
    /// halo-refresh entries leave it `None` and walk.
    pub(super) distinct_materials: Option<Vec<u16>>,
}

/// The derived surface-mesh view for one cached asset: the vertex / index
/// buffers, their slab allocator + byte-range dirty trackers, the meshlet
/// cluster DAG, and the cluster spatial index. Rebuilt from the companion
/// [`VoxelModel`] by the re-extract paths and uploaded to the GPU; it is
/// disposable derived state, never the source of truth.
pub(super) struct MeshView {
    /// Surface-mesh vertices from naive surface-nets extraction.
    /// Object-local positions on grid corners; carries oct-packed
    /// normal + `leaf_attr_id`. Sized proportional to surface area,
    /// not voxel count.
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
    /// Slab allocator over `mesh_indices` — list of `(start_elem,
    /// len_elem)` free ranges, sorted by `start_elem`. Coalesces with
    /// adjacent ranges on insert (see [`Self::free_index_range`]). The
    /// filter and patch paths in [`super::sculpt::rebuild_dirty_clusters`]
    /// reclaim slots via this list instead of orphaning them at the tail
    /// of `mesh_indices` — without it, every sculpt stamp grew the IBO
    /// monotonically until `max_buffer_size` was hit and the IBO became
    /// invalid mid-render.
    pub(super) mesh_indices_free_list: Vec<(u32, u32)>,
    /// Bump pointer for the slab allocator — first element index not yet
    /// reachable through any cluster or any free-list entry. Always
    /// `<= mesh_indices.len() as u32`.
    pub(super) mesh_indices_next_free: u32,
    /// Byte-range dirty tracker over `mesh_indices`. The slab allocator
    /// writes into INTERIOR offsets when it reuses a freed slot, so the
    /// renderer's old tail-only upload path silently dropped those
    /// writes. Sculpt writes through `mesh_indices_write_at` (which
    /// marks the range here); the next upload iterates this and issues
    /// one `queue.write_buffer` per range. Cleared on upload via
    /// [`super::ArvxSceneManager::mark_loaded_asset_uploads_clean`].
    pub(super) mesh_indices_dirty: DirtyRanges,
    /// Byte-range dirty tracker over `mesh_vertices`. Mirrors
    /// `mesh_indices_dirty` for the VBO side.
    ///
    /// **Why this exists:** the renderer's earlier VBO upload path
    /// assumed the CPU-side vertex buffer was append-only (sculpt's
    /// `mesh_vertices.extend_from_slice` pattern) and only re-uploaded
    /// the tail. That assumption breaks when an `AssetHandle` is
    /// recycled by a fresh integrate — terrain stamp move evicts a
    /// tile and re-bakes it; the new asset takes the freed slot with
    /// a completely different vertex set, but `vbo_uploaded_bytes`
    /// from the previous asset still pointed past the new bytes the
    /// tail-only upload would write, leaving stale prefix vertices on
    /// the GPU that the new IBO then indexed into → fully-replaced
    /// triangles with stale vertex positions = the visible
    /// "spider-leg" mesh-shard corruption on stamp drag (Phase 5.6
    /// 2026-05-19). Switching to explicit dirty-range tracking — full
    /// re-mark on integrate, append-only mark on sculpt — keeps the
    /// upload behaviour correct for every writer.
    pub(super) mesh_vertices_dirty: DirtyRanges,
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
    /// Number of cluster entries that came from the bake-time DAG —
    /// i.e. `meshlet_clusters[0 .. bake_time_cluster_count]`. Patch
    /// clusters appended during sculpt occupy `[bake_time_cluster_count
    /// .. len)`. The split matters for compaction: bake-time clusters
    /// are referenced by `dag_consumed` / `dag_produced` so their IDs
    /// can't move (we tombstone them with `index_count = 0`), while
    /// patch clusters have no DAG references and can be `swap_remove`d
    /// freely. Set at load + reset on full mesh re-extract; the empty-
    /// mesh reset path drops it to 0.
    pub(super) bake_time_cluster_count: u32,
    /// DAG group spans for sculpt's per-chain LOD-0 clamp. Each entry
    /// describes one simplification group: the consumed prev-level
    /// cluster IDs and produced this-level cluster IDs. Combined with
    /// [`MeshletCluster::group_above_idx`] / `group_below_idx`, this
    /// gives sculpt a CC walk over the DAG to mark every cluster in a
    /// brush-touched chain as `LOD_DIRTY` — narrower than R4d V1's
    /// asset-wide clamp.
    ///
    /// Empty for v5 assets without a baked DAG (the load-path
    /// fallback rebuilds the DAG from the unclustered LOD-0 indices,
    /// which populates this).
    pub(super) dag_groups: Vec<arvx_core::mesh_lod::DagGroup>,
    /// Flat per-group consumed cluster IDs, indexed by each
    /// `DagGroup::consumed_first..consumed_first+consumed_count`.
    pub(super) dag_consumed: Vec<u32>,
    /// Flat per-group produced cluster IDs.
    pub(super) dag_produced: Vec<u32>,
    /// Per-asset "needs GPU re-upload" flags. The render thread checks
    /// these on every geometry-epoch bump and skips assets whose data
    /// hasn't changed — cuts the ~25-asset × ~175 MB re-upload cost
    /// (the dominant 2-4 s/stamp bottleneck on splat5) down to just
    /// the one asset the sculpt mutated.
    ///
    /// Set to `true` at load. Sculpt sets `mesh_dirty` and
    /// `clusters_dirty` true. The render thread clears both after
    /// upload via
    /// [`ArvxSceneManager::mark_loaded_asset_uploads_clean`].
    pub(super) mesh_dirty: bool,
    pub(super) clusters_dirty: bool,
    /// D7 — bucket-grid spatial index over LOD-0 `meshlet_clusters`.
    /// Queried by `clusters_in_brush_grid_aabb` to skip the ~105 k-
    /// entry linear scan. Built at asset load + rebuilt on full
    /// mesh re-extract + incrementally updated on patch-cluster
    /// append. See `cluster_spatial_index.rs`.
    pub(super) cluster_spatial_index: super::cluster_spatial_index::ClusterSpatialIndex,
}

impl MeshView {
    /// Reset the slab allocator state to "everything beyond
    /// `mesh_indices.len()` is unallocated, the whole prefix is in use
    /// by callers". Use after a full mesh rebuild — the new
    /// `meshlet_clusters` reference the freshly-built indices and there
    /// are no orphaned slots yet. Marks the full prefix dirty so the
    /// next GPU upload re-uploads the entire IBO.
    pub(super) fn reset_mesh_indices_slab(&mut self) {
        self.mesh_indices_free_list.clear();
        self.mesh_indices_next_free = self.mesh_indices.len() as u32;
        self.mesh_indices_dirty.clear();
        let total_bytes = self
            .mesh_indices
            .len()
            .checked_mul(MESH_INDEX_STRIDE as usize)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0);
        if total_bytes > 0 {
            self.mesh_indices_dirty.mark_full(total_bytes);
        }
    }

    /// Allocate a contiguous run of `count` u32 slots inside
    /// `mesh_indices`, returning the start element index. First-fit
    /// search on the free list; falls back to bumping
    /// `mesh_indices_next_free` (resizing `mesh_indices` if needed).
    ///
    /// Pre-condition: `count > 0`. Returns the start element index;
    /// caller is responsible for writing the actual indices through
    /// [`Self::mesh_indices_write_at`] (which marks the range dirty).
    pub(super) fn alloc_index_range(&mut self, count: u32) -> u32 {
        debug_assert!(count > 0, "alloc_index_range with count=0");
        if let Some(idx) = self
            .mesh_indices_free_list
            .iter()
            .position(|(_, c)| *c >= count)
        {
            let (start, free_count) = self.mesh_indices_free_list[idx];
            if free_count == count {
                self.mesh_indices_free_list.remove(idx);
            } else {
                // Shrink the free range from the front so the remaining
                // tail stays sorted.
                self.mesh_indices_free_list[idx] = (start + count, free_count - count);
            }
            return start;
        }

        let start = self.mesh_indices_next_free;
        let end = start + count;
        if (end as usize) > self.mesh_indices.len() {
            // Mirrors the LeafAttrPool growth policy — double, but at
            // least enough to fit the request.
            let new_cap = ((self.mesh_indices.len() as u32).saturating_mul(2)).max(end);
            self.mesh_indices.resize(new_cap as usize, 0);
        }
        self.mesh_indices_next_free = end;
        start
    }

    /// Return `[start, start + count)` to the free list, coalescing
    /// with adjacent free ranges on either side. The contained indices
    /// are not zeroed — the GPU never reads bytes outside an active
    /// cluster's `(index_offset, index_count)` span, so stale data is
    /// inert.
    ///
    /// When the freed range touches `mesh_indices_next_free`, the bump
    /// pointer is pulled back instead of appending to the free list.
    /// No-op when `count == 0`.
    pub(super) fn free_index_range(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let end = start + count;
        debug_assert!(
            end as usize <= self.mesh_indices.len(),
            "free_index_range out of bounds: {}..{} vs len {}",
            start,
            end,
            self.mesh_indices.len(),
        );

        // Tail-touch: pull the bump pointer back instead of recording a
        // free range. Then absorb any free-list entries that now abut
        // the new bump pointer (left-coalesce).
        if end == self.mesh_indices_next_free {
            self.mesh_indices_next_free = start;
            loop {
                let pos = self
                    .mesh_indices_free_list
                    .iter()
                    .position(|(s, c)| s + c == self.mesh_indices_next_free);
                match pos {
                    Some(i) => {
                        let (s, _) = self.mesh_indices_free_list.remove(i);
                        self.mesh_indices_next_free = s;
                    }
                    None => break,
                }
            }
            return;
        }

        // Interior free — insert into the sorted free list and coalesce
        // with the immediate predecessor / successor if they abut.
        let mut insert = (start, count);
        let mut i = 0;
        while i < self.mesh_indices_free_list.len() {
            let (s, c) = self.mesh_indices_free_list[i];
            if s + c == insert.0 {
                // Previous range abuts ours on the left — absorb it.
                insert = (s, c + insert.1);
                self.mesh_indices_free_list.remove(i);
                continue;
            }
            if insert.0 + insert.1 == s {
                // Next range abuts ours on the right — absorb it.
                insert = (insert.0, insert.1 + c);
                self.mesh_indices_free_list.remove(i);
                continue;
            }
            i += 1;
        }
        // Keep the list sorted by start so first-fit search is
        // predictable and `cluster.index_offset` can be reasoned about.
        let pos = self
            .mesh_indices_free_list
            .binary_search_by_key(&insert.0, |(s, _)| *s)
            .unwrap_or_else(|p| p);
        self.mesh_indices_free_list.insert(pos, insert);
    }

    /// Write `src` into `mesh_indices` starting at `offset` (in element
    /// units), marking the corresponding byte range dirty so the next
    /// GPU upload picks it up. `offset + src.len()` must fit within
    /// `mesh_indices.len()` — callers obtain the offset via
    /// [`Self::alloc_index_range`] which sizes the underlying Vec.
    pub(super) fn mesh_indices_write_at(&mut self, offset: u32, src: &[u32]) {
        if src.is_empty() {
            return;
        }
        let end = offset as usize + src.len();
        debug_assert!(
            end <= self.mesh_indices.len(),
            "mesh_indices_write_at out of bounds: {}..{} vs len {}",
            offset,
            end,
            self.mesh_indices.len(),
        );
        self.mesh_indices[offset as usize..end].copy_from_slice(src);
        self.mesh_indices_dirty
            .mark(offset * MESH_INDEX_STRIDE, (src.len() as u32) * MESH_INDEX_STRIDE);
    }

    /// Heuristic: does `mesh_indices` carry enough free-list fragments
    /// to make a `compact_mesh_indices` pass worth its O(N) cost?
    /// Returns true when free-list bytes exceed 30 % of the in-use
    /// region (`next_free`), and that region is large enough that
    /// compaction saves meaningful memory (≥ 64k indices = 256 KiB).
    /// Below the size floor a fragmented buffer fits comfortably in
    /// one wgpu allocation either way.
    pub(super) fn should_compact_mesh_indices(&self) -> bool {
        const MIN_NEXT_FREE_ELEMS: u32 = 64 * 1024;
        if self.mesh_indices_next_free < MIN_NEXT_FREE_ELEMS {
            return false;
        }
        let free: u64 = self
            .mesh_indices_free_list
            .iter()
            .map(|(_, len)| *len as u64)
            .sum();
        // free / next_free >= 0.30 → free * 10 >= next_free * 3
        free.saturating_mul(10) >= (self.mesh_indices_next_free as u64).saturating_mul(3)
    }

    /// Defragment `mesh_indices` by walking every cluster in table
    /// order and copying its `[index_offset, index_offset + index_count)`
    /// range into a fresh dense `Vec`. Each cluster's `index_offset` is
    /// rewritten to its new position. After the pass the free list is
    /// empty, `next_free` sits at the new dense length, and the whole
    /// IBO is marked dirty — every byte in the buffer may have moved,
    /// so the next upload re-pushes the full prefix to the GPU. The
    /// `meshlet_clusters` table itself isn't reordered (cluster IDs
    /// stay stable), so DAG references and the spatial index remain
    /// valid.
    ///
    /// Tombstone clusters (`index_count == 0`) keep their slot but
    /// take a defensive in-range offset (the current new-buffer end)
    /// so any later CPU code doing `index_offset + index_count`
    /// arithmetic stays inside the Vec.
    ///
    /// Returns the number of bytes reclaimed (old `next_free * 4` minus
    /// new). Callers typically gate on
    /// [`Self::should_compact_mesh_indices`].
    pub(super) fn compact_mesh_indices(&mut self) -> u64 {
        let old_next_free = self.mesh_indices_next_free;

        // Empty asset — drop the buffer entirely. Hit by the
        // empty-mesh reset path; we mostly handle it here for
        // completeness so direct-call tests don't have to special-case.
        if self.meshlet_clusters.is_empty() {
            let reclaimed = (old_next_free as u64) * (MESH_INDEX_STRIDE as u64);
            self.mesh_indices.clear();
            self.mesh_indices_next_free = 0;
            self.mesh_indices_free_list.clear();
            self.mesh_indices_dirty.clear();
            return reclaimed;
        }

        let total_live: usize = self
            .meshlet_clusters
            .iter()
            .map(|c| c.index_count as usize)
            .sum();
        let mut new_indices: Vec<u32> = Vec::with_capacity(total_live);

        // Split-borrow so the cluster loop can read the old `mesh_indices`
        // while rewriting `meshlet_clusters[i].index_offset` in place.
        let old_indices = &self.mesh_indices;
        for cluster in self.meshlet_clusters.iter_mut() {
            let count = cluster.index_count as usize;
            if count == 0 {
                // Tombstone: take a valid in-range offset. The shader
                // won't dereference (count=0 → no draw); CPU arithmetic
                // on (offset+count) stays inside the Vec.
                cluster.index_offset = new_indices.len() as u32;
                continue;
            }
            let old_offset = cluster.index_offset as usize;
            let new_offset = new_indices.len() as u32;
            new_indices.extend_from_slice(&old_indices[old_offset..old_offset + count]);
            cluster.index_offset = new_offset;
        }

        let new_len = new_indices.len() as u32;
        self.mesh_indices = new_indices;
        self.mesh_indices_next_free = new_len;
        self.mesh_indices_free_list.clear();

        // Every offset may have moved → full re-upload. Clear first to
        // drop the previous per-stamp dirty entries (now stale relative
        // to the new content).
        self.mesh_indices_dirty.clear();
        let total_bytes = new_len.saturating_mul(MESH_INDEX_STRIDE);
        if total_bytes > 0 {
            self.mesh_indices_dirty.mark_full(total_bytes);
        }

        let reclaimed = (old_next_free.saturating_sub(new_len) as u64)
            * (MESH_INDEX_STRIDE as u64);
        reclaimed
    }

    /// Walk the patch range `[bake_time_cluster_count .. len)` forward,
    /// `swap_remove`ing every entry with `index_count == 0`. After each
    /// removal the previous last entry slides into the current slot, so
    /// we recheck the same slot before advancing — that catches a chain
    /// of consecutive empties without a second pass.
    ///
    /// Empty bake-time clusters (originals carved away by the filter)
    /// stay as zero-count tombstones. Their IDs are referenced by
    /// `dag_consumed` / `dag_produced` so we can't move them; a
    /// zero-count cluster contributes nothing to `mesh_lod0_index_count`
    /// and emits a no-op draw on the GPU.
    ///
    /// When at least one patch was removed, the cluster IDs that
    /// `cluster_spatial_index` records under the moved entries are
    /// now stale — full rebuild from the compacted table. The caller
    /// supplies `grid_origin` / `base_vs` (from the companion
    /// [`VoxelModel`]) since the view no longer holds the model.
    /// Returns the number of patches dropped.
    pub(super) fn compact_empty_patches(&mut self, grid_origin: glam::Vec3, base_vs: f32) -> u32 {
        let bake = self.bake_time_cluster_count as usize;
        if self.meshlet_clusters.len() <= bake {
            return 0;
        }
        let mut removed = 0u32;
        let mut i = bake;
        while i < self.meshlet_clusters.len() {
            if self.meshlet_clusters[i].index_count == 0 {
                self.meshlet_clusters.swap_remove(i);
                removed += 1;
                // Don't advance `i` — the just-swapped-in entry hasn't
                // been checked yet. Loop bound (`len`) shrinks by 1.
            } else {
                i += 1;
            }
        }
        if removed > 0 {
            self.cluster_spatial_index
                .rebuild(&self.meshlet_clusters, grid_origin, base_vs);
        }
        removed
    }
}

impl VoxelModel {
    /// Object-local grid origin used by the spatial index and brush
    /// math. Derived from `(aabb_center - extent/2)` — same formula
    /// every caller uses (asset_load, sculpt, `info`); centralised here
    /// so compaction can rebuild the spatial index without callers
    /// having to thread the value through.
    pub(super) fn grid_origin(&self) -> glam::Vec3 {
        let extent = (1u32 << self.spatial_handle.depth) as f32
            * self.spatial_handle.base_voxel_size;
        let aabb_center = (self.aabb.min + self.aabb.max) * 0.5;
        aabb_center - glam::Vec3::splat(extent * 0.5)
    }

    pub(super) fn info(&self) -> AssetInfo {
        // Reconstruct grid origin the same way voxelize_octree does:
        // `aabb_center - extent/2`. Matches the bake-time geometry, so
        // existing .arvx files render identically.
        let extent = (1u32 << self.spatial_handle.depth) as f32
            * self.spatial_handle.base_voxel_size;
        let aabb_center = (self.aabb.min + self.aabb.max) * 0.5;
        let grid_origin = aabb_center - glam::Vec3::splat(extent * 0.5);
        AssetInfo {
            spatial: arvx_core::scene_node::SpatialHandle::Octree {
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
/// that was resolved against the `.arvx` extension, so two different
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
    /// (in `ArvxRenderer::mesh_buffers` / `mesh_cluster_buffers`)
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

    /// Iterate every populated `(handle, &entry)` pair. Used by
    /// paint's slot-range relaxation (Phase 4 fix): paint needs to
    /// consult the entry's `sculpt_extra_slots` HashSet, but is
    /// called with `&AssetInfo` rather than `&AssetEntry`, so we
    /// look up the entry via its `grid_origin` / slot range.
    pub(super) fn iter(&self) -> impl Iterator<Item = (AssetHandle, &AssetEntry)> {
        self.entries.iter().enumerate().filter_map(|(i, e)| {
            e.as_ref().map(|entry| (AssetHandle(i as u32), entry))
        })
    }

    pub(super) fn remove(&mut self, handle: AssetHandle) -> Option<AssetEntry> {
        let slot = handle.0 as usize;
        let taken = self.entries.get_mut(slot)?.take()?;
        self.path_to_handle.remove(&taken.path);
        self.free_slots.push(handle.0);
        Some(taken)
    }
}

/// Result of [`ArvxSceneManager::reload_asset`]. `old_handle` is the handle
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
    pub spatial: arvx_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: arvx_core::Aabb,
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
/// nothing in the active pipeline reads these. Kept for scene-
/// loading compatibility: every leaf is a surface voxel now, so the
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
                    Some(node) if node == arvx_core::sparse_octree::EMPTY_NODE => true,
                    Some(node) if node == arvx_core::sparse_octree::INTERIOR_NODE => false,
                    Some(node) if arvx_core::sparse_octree::is_leaf(node) => false,
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

#[cfg(test)]
mod slab_tests {
    use super::*;
    use arvx_core::{Aabb, DirtyRanges, OctreeHandle, SparseOctree};

    /// Build an `AssetEntry` with `mesh_indices` pre-populated as if the
    /// asset had just been loaded with `count` baked indices. `next_free`
    /// sits at the tail (every slot is "owned" by a hypothetical cluster
    /// table). Dirty ranges are empty so test assertions don't have to
    /// account for the load-time mark_full.
    fn entry_with_indices(initial: &[u32]) -> AssetEntry {
        AssetEntry {
            path: std::path::PathBuf::from("test:slab"),
            refcount: 1,
            model: VoxelModel {
                spatial_handle: OctreeHandle {
                    root_offset: 0,
                    len: 0,
                    depth: 8,
                    base_voxel_size: 1.0,
                },
                voxel_size: 1.0,
                aabb: Aabb {
                    min: glam::Vec3::ZERO,
                    max: glam::Vec3::splat(256.0),
                },
                voxel_count: 0,
                leaf_attr_slot_start: 0,
                leaf_attr_slot_count: 0,
                brick_start: 0,
                brick_count: 0,
                skinning: None,
                cpu_octree: SparseOctree::new(8, 1.0),
                sculpt_extra_slots: std::collections::HashSet::new(),
                sculpt_owned_slots: rustc_hash::FxHashSet::default(),
                halo_extra_slots: std::collections::HashSet::new(),
                halo_cells: Vec::new(),
                distinct_materials: None,
            },
            view: MeshView {
                mesh_vertices: Vec::new(),
                mesh_indices: initial.to_vec(),
                mesh_indices_free_list: Vec::new(),
                mesh_indices_next_free: initial.len() as u32,
                mesh_indices_dirty: DirtyRanges::new(),
                mesh_vertices_dirty: DirtyRanges::new(),
                mesh_lod0_index_count: 0,
                bake_time_cluster_count: 0,
                meshlet_clusters: Vec::new(),
                dag_groups: Vec::new(),
                dag_consumed: Vec::new(),
                dag_produced: Vec::new(),
                mesh_dirty: false,
                clusters_dirty: false,
                cluster_spatial_index:
                    crate::arvx_scene_manager::cluster_spatial_index::ClusterSpatialIndex::new(),
            },
        }
    }

    /// Mirrors the filter path's "kept indices stay at the cluster's
    /// existing offset" rule: write a shrunk subset back to the same
    /// slot, free the tail, verify the writes land in-place and the
    /// slab doesn't grow.
    #[test]
    fn mesh_indices_slab_filter_reuses_slot() {
        // Imagine one cluster occupying [0..9): 3 tris.
        let mut entry = entry_with_indices(&[10, 11, 12, 20, 21, 22, 30, 31, 32]);
        let before_len = entry.view.mesh_indices.len();
        // The filter drops the middle tri; kept is [10,11,12,30,31,32].
        let kept: [u32; 6] = [10, 11, 12, 30, 31, 32];
        entry.view.mesh_indices_write_at(0, &kept);
        entry.view.free_index_range(6, 3);

        assert_eq!(
            entry.view.mesh_indices.len(),
            before_len,
            "filter must not grow the slab — kept tris fit in the existing slot",
        );
        assert_eq!(&entry.view.mesh_indices[..6], &kept);
        assert_eq!(entry.view.mesh_indices_free_list, vec![]);
        assert_eq!(
            entry.view.mesh_indices_next_free, 6,
            "freed tail abuts next_free → pulled back, no interior fragment",
        );
        let dirty: Vec<_> = entry.view.mesh_indices_dirty.iter().collect();
        assert_eq!(
            dirty,
            vec![(0, 24)],
            "in-place write marks exactly the 6×4 = 24 bytes that changed",
        );
    }

    /// Mirrors the patch path's "alloc reuses a freed slot when one
    /// fits": apply patch, free it, apply same-size patch, verify the
    /// second patch lands at the first's offset and the slab doesn't
    /// grow.
    #[test]
    fn mesh_indices_slab_patch_reuses_freed_slot() {
        // Empty asset, no baked indices. First patch grows the slab.
        let mut entry = entry_with_indices(&[]);
        let first_offset = entry.view.alloc_index_range(6);
        entry.view.mesh_indices_write_at(first_offset, &[100, 101, 102, 103, 104, 105]);
        assert_eq!(first_offset, 0);
        let after_first_len = entry.view.mesh_indices.len();
        assert!(after_first_len >= 6);

        // Free the first patch (simulating cluster-table compaction or
        // a re-stamp that overwrites the prior patch's region).
        entry.view.free_index_range(first_offset, 6);
        // Free range abuts next_free → next_free pulls back to 0.
        assert_eq!(entry.view.mesh_indices_next_free, 0);
        assert_eq!(entry.view.mesh_indices_free_list, vec![]);

        // Second same-size patch must reuse the slot, no growth.
        let second_offset = entry.view.alloc_index_range(6);
        assert_eq!(
            second_offset, first_offset,
            "second patch reuses the first's offset",
        );
        entry.view.mesh_indices_write_at(second_offset, &[200, 201, 202, 203, 204, 205]);
        assert_eq!(
            entry.view.mesh_indices.len(),
            after_first_len,
            "slab does not grow when a freed slot fits the new alloc",
        );
        assert_eq!(&entry.view.mesh_indices[..6], &[200, 201, 202, 203, 204, 205]);
    }

    /// Mirrors the long-session worst case: 100 successive patches at
    /// the same location with intervening frees. Old behaviour grew
    /// `mesh_indices` by `100 × patch_size`; with the slab the underlying
    /// vec stays bounded by the maximum-ever patch size.
    #[test]
    fn mesh_indices_slab_size_bounded() {
        let mut entry = entry_with_indices(&[]);
        const PATCH_LEN: u32 = 32;
        const STAMPS: u32 = 100;

        for stamp in 0..STAMPS {
            let off = entry.view.alloc_index_range(PATCH_LEN);
            // Same offset every time — proves we never grow past the
            // first stamp's allocation.
            assert_eq!(off, 0, "stamp {stamp} expected offset 0");
            let payload: Vec<u32> = (0..PATCH_LEN).map(|i| stamp * 1000 + i).collect();
            entry.view.mesh_indices_write_at(off, &payload);
            // Simulate the next stamp's "filter" path freeing the prior
            // patch's range before reallocating.
            entry.view.free_index_range(off, PATCH_LEN);
        }

        assert!(
            entry.view.mesh_indices.len() <= (PATCH_LEN * 2) as usize,
            "100 stamps must not grow the slab linearly — got len {}",
            entry.view.mesh_indices.len(),
        );
        assert_eq!(
            entry.view.mesh_indices_next_free, 0,
            "every stamp freed before next alloc → bump pointer at 0",
        );
        assert!(
            entry.view.mesh_indices_free_list.is_empty(),
            "tail-pullback collapses the free list",
        );
    }

    /// Interior frees can't collapse to `next_free`; they accumulate as
    /// free-list entries and must coalesce when neighbours abut.
    #[test]
    fn mesh_indices_slab_interior_free_coalesces() {
        // Pre-allocate `[A][B][C]` of 4 indices each. Free B then C; B+C
        // should coalesce into one (4, 8) entry, then tail-pullback
        // because (4, 8) abuts next_free=12.
        let mut entry = entry_with_indices(&[0; 12]);
        // Slab thinks all 12 are "in use" (entry_with_indices sets
        // next_free to len). Free B = [4..8).
        entry.view.free_index_range(4, 4);
        assert_eq!(entry.view.mesh_indices_free_list, vec![(4, 4)]);
        assert_eq!(entry.view.mesh_indices_next_free, 12);

        // Free C = [8..12). Touches next_free → pull back; then absorbs
        // the abutting (4, 4) free-list entry on the left.
        entry.view.free_index_range(8, 4);
        assert_eq!(entry.view.mesh_indices_free_list, vec![]);
        assert_eq!(entry.view.mesh_indices_next_free, 4);
    }

    /// Cluster fixture for compaction tests. LOD-0, given AABB + index
    /// span. Anything that the spatial index would re-bucket flows
    /// through `aabb_min` / `aabb_max`.
    fn cluster(
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        index_offset: u32,
        index_count: u32,
    ) -> crate::mesh_pass::MeshletCluster {
        crate::mesh_pass::MeshletCluster {
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            index_offset,
            index_count,
            lod_level: 0,
            flags: 0,
            cluster_error: 0.0,
            parent_group_error: arvx_core::mesh_cluster::PARENT_GROUP_ERROR_ROOT,
            group_above_idx: arvx_core::mesh_cluster::DAG_GROUP_NONE,
            group_below_idx: arvx_core::mesh_cluster::DAG_GROUP_NONE,
            _pad3: 0,
        }
    }

    /// Build an entry pre-populated with `clusters`, splitting the
    /// table into `bake` originals + the remainder as patches. Spatial
    /// index is rebuilt to reflect the supplied geometry so queries
    /// return real candidate sets. Same grid math as the real load
    /// path (aabb_center − extent/2 = origin → grid coords map 1:1).
    fn entry_with_clusters(
        clusters: Vec<crate::mesh_pass::MeshletCluster>,
        bake: u32,
    ) -> AssetEntry {
        let mut entry = AssetEntry {
            path: std::path::PathBuf::from("test:compact"),
            refcount: 1,
            model: VoxelModel {
                spatial_handle: OctreeHandle {
                    root_offset: 0,
                    len: 0,
                    depth: 8,
                    base_voxel_size: 1.0,
                },
                voxel_size: 1.0,
                aabb: Aabb {
                    min: glam::Vec3::ZERO,
                    max: glam::Vec3::splat(256.0),
                },
                voxel_count: 0,
                leaf_attr_slot_start: 0,
                leaf_attr_slot_count: 0,
                brick_start: 0,
                brick_count: 0,
                skinning: None,
                cpu_octree: SparseOctree::new(8, 1.0),
                sculpt_extra_slots: std::collections::HashSet::new(),
                sculpt_owned_slots: rustc_hash::FxHashSet::default(),
                halo_extra_slots: std::collections::HashSet::new(),
                halo_cells: Vec::new(),
                distinct_materials: None,
            },
            view: MeshView {
                mesh_vertices: Vec::new(),
                mesh_indices: Vec::new(),
                mesh_indices_free_list: Vec::new(),
                mesh_indices_next_free: 0,
                mesh_indices_dirty: DirtyRanges::new(),
                mesh_vertices_dirty: DirtyRanges::new(),
                mesh_lod0_index_count: 0,
                bake_time_cluster_count: bake,
                meshlet_clusters: clusters,
                dag_groups: Vec::new(),
                dag_consumed: Vec::new(),
                dag_produced: Vec::new(),
                mesh_dirty: false,
                clusters_dirty: false,
                cluster_spatial_index:
                    crate::arvx_scene_manager::cluster_spatial_index::ClusterSpatialIndex::new(),
            },
        };
        let grid_origin = entry.model.grid_origin();
        let base_vs = entry.model.spatial_handle.base_voxel_size;
        entry.view.cluster_spatial_index.rebuild(
            &entry.view.meshlet_clusters,
            grid_origin,
            base_vs,
        );
        entry
    }

    /// Call `MeshView::compact_empty_patches` with the `grid_origin` /
    /// `base_vs` the companion `VoxelModel` would supply at runtime.
    fn compact_empty(entry: &mut AssetEntry) -> u32 {
        let grid_origin = entry.model.grid_origin();
        let base_vs = entry.model.spatial_handle.base_voxel_size;
        entry.view.compact_empty_patches(grid_origin, base_vs)
    }

    /// Empty patch clusters are removed via swap_remove; the cluster
    /// table shrinks accordingly. Bake-time originals stay put.
    #[test]
    fn compact_empty_patch_swap_removed() {
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 12), // bake-time
                cluster([5.0; 3], [6.0; 3], 12, 0), // empty patch
            ],
            1,
        );
        let removed = compact_empty(&mut entry);
        assert_eq!(removed, 1);
        assert_eq!(entry.view.meshlet_clusters.len(), 1);
        // Bake-time cluster preserved at its original id.
        assert_eq!(entry.view.meshlet_clusters[0].index_count, 12);
    }

    /// Empty bake-time clusters MUST NOT be swap_removed — DAG
    /// topology references their IDs and reshuffling would corrupt
    /// `dag_consumed` / `dag_produced`. They stay as zero-count
    /// tombstones; the GPU still walks them but emits no draw work.
    #[test]
    fn compact_preserves_empty_original_as_tombstone() {
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 0),  // empty original — KEEP
                cluster([2.0; 3], [3.0; 3], 0, 6),  // original — keep
            ],
            2,
        );
        let removed = compact_empty(&mut entry);
        assert_eq!(removed, 0, "originals are tombstoned, not removed");
        assert_eq!(entry.view.meshlet_clusters.len(), 2);
        assert_eq!(entry.view.meshlet_clusters[0].index_count, 0);
    }

    /// Interleaved empty/non-empty patches all get compacted in one
    /// pass. The forward + same-slot-recheck walk handles consecutive
    /// empties that arise after swap_remove brings in another empty.
    #[test]
    fn compact_multiple_interleaved_empty_patches() {
        // 1 bake + 4 patches: empty, non-empty, empty, non-empty.
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 9),    // bake-time
                cluster([10.0; 3], [11.0; 3], 9, 0),  // patch A — empty
                cluster([20.0; 3], [21.0; 3], 9, 6),  // patch B — keep
                cluster([30.0; 3], [31.0; 3], 15, 0), // patch C — empty
                cluster([40.0; 3], [41.0; 3], 15, 3), // patch D — keep
            ],
            1,
        );
        let removed = compact_empty(&mut entry);
        assert_eq!(removed, 2);
        assert_eq!(entry.view.meshlet_clusters.len(), 3);
        // Surviving patches keep their content (we can't predict their
        // post-compaction slot because swap_remove rearranges, but both
        // their AABBs should be present in the table).
        let aabbs: Vec<_> = entry.view
            .meshlet_clusters
            .iter()
            .map(|c| c.aabb_min)
            .collect();
        assert!(aabbs.contains(&[0.0; 3]));  // original
        assert!(aabbs.contains(&[20.0; 3])); // patch B
        assert!(aabbs.contains(&[40.0; 3])); // patch D
    }

    /// After compaction the spatial index reflects the new cluster IDs.
    /// We construct two patches in distinct grid regions, drop the
    /// middle one, and verify the remaining patch is queryable under
    /// its (possibly relocated) id.
    #[test]
    fn compact_spatial_index_reflects_relocations() {
        use glam::IVec3;
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 6),    // bake-time @ [0,1]
                cluster([100.0; 3], [101.0; 3], 6, 0), // empty patch @ [100,101]
                cluster([200.0; 3], [201.0; 3], 6, 6), // patch @ [200,201]
            ],
            1,
        );

        // Sanity: pre-compaction, patch ids 1 + 2 are both in the
        // spatial index.
        let pre = entry.view.cluster_spatial_index.query(
            IVec3::new(50, 50, 50),
            IVec3::new(250, 250, 250),
        );
        assert_eq!(pre, vec![1, 2]);

        let removed = compact_empty(&mut entry);
        assert_eq!(removed, 1);
        assert_eq!(entry.view.meshlet_clusters.len(), 2);

        // The empty patch is gone. The other patch slid into slot 1
        // via swap_remove. Query its bucket — must yield the new id.
        let post = entry.view.cluster_spatial_index.query(
            IVec3::new(195, 195, 195),
            IVec3::new(210, 210, 210),
        );
        assert_eq!(post, vec![1]);
        // The vacated region must NOT yield a hit any more.
        let vacated = entry.view.cluster_spatial_index.query(
            IVec3::new(95, 95, 95),
            IVec3::new(110, 110, 110),
        );
        assert!(vacated.is_empty(), "removed patch must vacate its buckets");
    }

    /// Long-session simulation — N rounds of "append empty patch +
    /// compact". The cluster table must stay at the bake-time count
    /// regardless of how many cycles we run.
    #[test]
    fn compact_bounded_over_many_stamps() {
        let mut entry = entry_with_clusters(
            vec![cluster([0.0; 3], [1.0; 3], 0, 9)], // one bake-time cluster
            1,
        );

        for stamp in 0..200 {
            // Simulate sculpt: emit an empty patch, then compact.
            let f = stamp as f32;
            entry.view
                .meshlet_clusters
                .push(cluster([f; 3], [f + 1.0; 3], 9, 0));
            compact_empty(&mut entry);
        }

        assert_eq!(
            entry.view.meshlet_clusters.len(),
            1,
            "table grows by patch + immediately shrinks via compaction \
             — bake-time count is the steady state",
        );
    }

    /// `compact_mesh_indices` rewrites every cluster's offset to a
    /// dense layout, clears the free list, and shrinks `next_free` to
    /// the new packed length. The bytes reclaimed equal the sum of the
    /// previous free-list ranges.
    #[test]
    fn compact_mesh_indices_densifies_buffer() {
        // Three clusters: [0..3), [10..13), [20..22). Gaps at [3..10)
        // and [13..20) are in the free list (10 elems = 40 bytes free).
        // Cluster contents recorded so we can verify them post-compact.
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 3),
                cluster([10.0; 3], [11.0; 3], 10, 3),
                cluster([20.0; 3], [21.0; 3], 20, 2),
            ],
            1, // 1 bake-time, 2 patches
        );
        entry.view.mesh_indices = vec![
            100, 101, 102,           // cluster 0
            0, 0, 0, 0, 0, 0, 0,     // free
            200, 201, 202,           // cluster 1
            0, 0, 0, 0, 0, 0, 0,     // free
            300, 301,                // cluster 2
        ];
        entry.view.mesh_indices_next_free = 22;
        entry.view.mesh_indices_free_list = vec![(3, 7), (13, 7)];

        let reclaimed = entry.view.compact_mesh_indices();

        // Densely packed at 3 + 3 + 2 = 8 elements.
        assert_eq!(entry.view.mesh_indices_next_free, 8);
        assert!(entry.view.mesh_indices_free_list.is_empty());
        assert_eq!(reclaimed, 14 * 4, "reclaimed = 14 freed elems * 4 B");

        // Cluster offsets rewritten in table order.
        assert_eq!(entry.view.meshlet_clusters[0].index_offset, 0);
        assert_eq!(entry.view.meshlet_clusters[1].index_offset, 3);
        assert_eq!(entry.view.meshlet_clusters[2].index_offset, 6);
    }

    /// Per-cluster index content is preserved by the compaction copy.
    /// Easy to break with an off-by-one in the slice indexing — the
    /// test reads back each cluster through its NEW offset and asserts
    /// the bytes match what was written at the OLD offset.
    #[test]
    fn compact_mesh_indices_preserves_cluster_content() {
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 5, 4),
                cluster([10.0; 3], [11.0; 3], 20, 6),
            ],
            1,
        );
        // 32-element scratch buffer with cluster content at the
        // expected offsets; everything else is garbage that compaction
        // should drop.
        entry.view.mesh_indices = vec![999; 32];
        entry.view.mesh_indices[5..9].copy_from_slice(&[10, 20, 30, 40]);
        entry.view.mesh_indices[20..26].copy_from_slice(&[50, 60, 70, 80, 90, 100]);
        entry.view.mesh_indices_next_free = 32;
        entry.view.mesh_indices_free_list = vec![(0, 5), (9, 11), (26, 6)];

        entry.view.compact_mesh_indices();

        let c0 = &entry.view.meshlet_clusters[0];
        let c1 = &entry.view.meshlet_clusters[1];
        assert_eq!(
            &entry.view.mesh_indices
                [c0.index_offset as usize..(c0.index_offset + c0.index_count) as usize],
            &[10, 20, 30, 40],
        );
        assert_eq!(
            &entry.view.mesh_indices
                [c1.index_offset as usize..(c1.index_offset + c1.index_count) as usize],
            &[50, 60, 70, 80, 90, 100],
        );
    }

    /// Tombstone clusters (index_count = 0) keep their slot but get an
    /// in-range offset post-compact so future `offset + count`
    /// arithmetic on the cluster never lands past `next_free`.
    #[test]
    fn compact_mesh_indices_tombstones_get_in_range_offset() {
        let mut entry = entry_with_clusters(
            vec![
                cluster([0.0; 3], [1.0; 3], 0, 4),  // bake, live
                cluster([2.0; 3], [3.0; 3], 100, 0), // bake, tombstoned
                cluster([4.0; 3], [5.0; 3], 10, 3), // bake, live
            ],
            3,
        );
        entry.view.mesh_indices = vec![0; 20];
        entry.view.mesh_indices[0..4].copy_from_slice(&[1, 2, 3, 4]);
        entry.view.mesh_indices[10..13].copy_from_slice(&[5, 6, 7]);
        entry.view.mesh_indices_next_free = 20;
        entry.view.mesh_indices_free_list = vec![(4, 6), (13, 7)];

        entry.view.compact_mesh_indices();

        let final_len = entry.view.mesh_indices_next_free;
        // Every offset must satisfy `offset + count <= next_free`.
        for c in &entry.view.meshlet_clusters {
            assert!(
                c.index_offset + c.index_count <= final_len,
                "tombstone or live cluster offset {} + count {} exceeded next_free {}",
                c.index_offset,
                c.index_count,
                final_len,
            );
        }
        // Live clusters survived with their content intact.
        let live0 = &entry.view.meshlet_clusters[0];
        let live2 = &entry.view.meshlet_clusters[2];
        assert_eq!(
            &entry.view.mesh_indices
                [live0.index_offset as usize..(live0.index_offset + 4) as usize],
            &[1, 2, 3, 4],
        );
        assert_eq!(
            &entry.view.mesh_indices
                [live2.index_offset as usize..(live2.index_offset + 3) as usize],
            &[5, 6, 7],
        );
    }

    /// After compaction, the IBO dirty tracker carries one full-range
    /// `(0, new_len * 4)` entry — every byte may have moved, so the
    /// renderer must re-upload the whole prefix.
    #[test]
    fn compact_mesh_indices_marks_full_buffer_dirty() {
        let mut entry = entry_with_clusters(
            vec![cluster([0.0; 3], [1.0; 3], 0, 3)],
            1,
        );
        entry.view.mesh_indices = vec![100, 101, 102, 0, 0, 0, 0];
        entry.view.mesh_indices_next_free = 7;
        entry.view.mesh_indices_free_list = vec![(3, 4)];
        // Leave a residual per-stamp dirty entry that compaction must
        // clear (it's stale relative to the new dense layout).
        entry.view.mesh_indices_dirty.mark(0, 12);

        entry.view.compact_mesh_indices();

        // Exactly one entry, covering the new dense buffer.
        let ranges: Vec<_> = entry.view.mesh_indices_dirty.iter().collect();
        assert_eq!(ranges, vec![(0, 3 * MESH_INDEX_STRIDE)]);
        assert!(entry.view.mesh_indices_dirty.is_full_pool(3 * MESH_INDEX_STRIDE));
    }

    /// Threshold gating: small buffers don't trigger compaction even
    /// when 100 % fragmented (compaction not worth the O(N) cost); a
    /// large buffer with ≥ 30 % free does.
    #[test]
    fn compact_mesh_indices_threshold_gates_on_size_and_fraction() {
        let mut entry = entry_with_clusters(
            vec![cluster([0.0; 3], [1.0; 3], 0, 0)],
            1,
        );

        // Small buffer, fully free → still below threshold (size floor
        // is 64k elems).
        entry.view.mesh_indices = vec![0; 1024];
        entry.view.mesh_indices_next_free = 1024;
        entry.view.mesh_indices_free_list = vec![(0, 1024)];
        assert!(
            !entry.view.should_compact_mesh_indices(),
            "below size floor → false even at 100 % fragmentation",
        );

        // Past size floor but only 20 % free.
        let big = 200 * 1024;
        let twenty_pct = big / 5;
        entry.view.mesh_indices = vec![0; big as usize];
        entry.view.mesh_indices_next_free = big;
        entry.view.mesh_indices_free_list = vec![(0, twenty_pct)];
        assert!(
            !entry.view.should_compact_mesh_indices(),
            "20 % fragmentation is below the 30 % cutoff",
        );

        // Past size floor and 35 % free → trip.
        let thirty_five_pct = (big as u64 * 35 / 100) as u32;
        entry.view.mesh_indices_free_list = vec![(0, thirty_five_pct)];
        assert!(
            entry.view.should_compact_mesh_indices(),
            "35 % fragmentation past size floor → compact",
        );
    }

    /// Frees a hole that doesn't touch existing free ranges (no
    /// coalesce); a subsequent alloc smaller than the hole shrinks it
    /// from the front, keeping the free list sorted.
    #[test]
    fn mesh_indices_slab_first_fit_shrinks_from_front() {
        let mut entry = entry_with_indices(&[0; 32]);
        // Punch holes at [4..8) and [20..28).
        entry.view.free_index_range(4, 4);
        entry.view.free_index_range(20, 8);
        assert_eq!(entry.view.mesh_indices_free_list, vec![(4, 4), (20, 8)]);

        // Alloc 4 → first-fit picks the (4, 4) hole; free list drops it.
        let off1 = entry.view.alloc_index_range(4);
        assert_eq!(off1, 4);
        assert_eq!(entry.view.mesh_indices_free_list, vec![(20, 8)]);

        // Alloc 4 → first-fit picks the (20, 8) hole; it shrinks from
        // the front to (24, 4).
        let off2 = entry.view.alloc_index_range(4);
        assert_eq!(off2, 20);
        assert_eq!(entry.view.mesh_indices_free_list, vec![(24, 4)]);
    }
}
