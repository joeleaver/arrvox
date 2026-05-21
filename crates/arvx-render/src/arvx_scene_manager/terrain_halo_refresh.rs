//! Phase 4.2b — cross-tile halo refresh on boundary edits.
//!
//! When tile A is sculpted at its boundary face F, the neighbour
//! tile B's halo cells at the matching -F face are stale (they were
//! sampled from `TerrainFn` at bake time and don't reflect A's
//! post-sculpt state). This module owns the refresh op + the
//! full-asset haloed re-extract that bakes the refreshed halo
//! into B's mesh.
//!
//! The cost model: O(S² · halo) cell lookups per face refresh (for
//! S = 256 voxels, halo = 2 → ~131 k lookups, sub-millisecond) plus
//! a full mesh re-extract + DAG rebuild on B (hundreds of ms on a
//! Tier-2 tile). Acceptable for a V1 boundary-edit stutter; per-
//! cluster boundary re-extract is a follow-up.

use std::collections::HashSet;

use glam::{IVec3, UVec3, Vec3};

use arvx_core::brick_pool::{BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};
use arvx_core::mesh_cluster::{
    MeshletCluster, CLUSTER_FLAG_LOD_DIRTY, DAG_GROUP_NONE, PARENT_GROUP_ERROR_ROOT,
};
use arvx_core::mesh_extract::{
    collect_cell_map_in_region, extract_mesh_region_from_cells_pooled_haloed,
    extract_surface_mesh_haloed, CELL_INTERIOR,
};
use arvx_core::mesh_lod::build_cluster_dag_with_levels;
use arvx_core::sparse_octree::{brick_id, is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE};
use arvx_core::LeafAttr;

use super::manager::ArvxSceneManager;
use super::types::AssetHandle;

/// Face index in `arvx_core::mesh_extract::FACE_DIRS` order.
/// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
pub const FACE_PX: u8 = 0;
pub const FACE_NX: u8 = 1;
pub const FACE_PY: u8 = 2;
pub const FACE_NY: u8 = 3;
pub const FACE_PZ: u8 = 4;
pub const FACE_NZ: u8 = 5;

/// Halo width (in finest-grid voxels) baked into terrain tiles. Must
/// match `arvx_terrain::bake::TILE_HALO_VOXELS`.
const TILE_HALO_VOXELS: i32 = 2;

/// One terrain neighbour's view of the shared face. Refresh op
/// updates `target`'s halo cells on `target_face` using `source`'s
/// interior boundary cells on the opposite face.
#[derive(Debug, Clone, Copy)]
pub struct HaloRefresh {
    /// Asset handle of the tile whose halo we're updating (the
    /// stale side — neighbour of the sculpted tile).
    pub target: AssetHandle,
    /// Face index on `target` (which of its 6 faces needs refreshing).
    /// `target_face = NX` means the halo cells at `coord.x in [-halo, 0)`
    /// in `target`'s tile-local frame.
    pub target_face: u8,
    /// Asset handle of the tile whose interior provides the new data
    /// (the sculpted tile).
    pub source: AssetHandle,
}

/// Resolved cell state for a single source coord. `Interior` means
/// the cell is bulk-solid inside an `INTERIOR_NODE` branch (no
/// explicit LeafAttr); `Surface` means a real leaf with attached
/// attributes.
#[derive(Debug, Clone, Copy)]
enum SourceCellState {
    Empty,
    Interior,
    Surface(LeafAttr, u32),
}

impl ArvxSceneManager {
    /// Apply one halo-refresh op + re-mesh the target tile.
    ///
    /// Returns the number of halo cells that changed. `None` if
    /// either handle is unknown or the target asset has no halo
    /// (non-terrain asset).
    pub fn apply_halo_refresh(&mut self, op: HaloRefresh) -> Option<usize> {
        // Sanity: both must exist.
        self.asset_cache.get(op.target)?;
        self.asset_cache.get(op.source)?;

        let changed = self.refresh_halo_face(op.target, op.target_face, op.source)?;
        if changed > 0 {
            // Phase 4.2b — slab path by default. Env-gated fallback
            // routes through the full-tile re-extract for sanity
            // comparison and as a safety valve if the slab path
            // produces a visual regression in the field.
            if std::env::var("ARVX_TERRAIN_HALO_FULL_REEXTRACT").is_ok() {
                self.rebuild_asset_mesh_haloed(op.target);
            } else {
                self.rebuild_face_band_clusters(op.target, op.target_face);
            }
            // Mesh changed → bump the geometry epoch so render-side
            // caches (skinning_data cache, asset_has_glass cache,
            // painted-walk snapshot) invalidate on the next tick.
            self.bump_geometry_epoch();
        }
        Some(changed)
    }

    /// Update `target.halo_cells` on `target_face` using `source`'s
    /// interior boundary cells. Allocates new pool slots for cells
    /// that didn't have a halo entry before; overwrites LeafAttrs in
    /// existing slots; drops entries for cells whose source side is
    /// now empty.
    ///
    /// Returns the number of cells whose state changed.
    fn refresh_halo_face(
        &mut self,
        target: AssetHandle,
        target_face: u8,
        source: AssetHandle,
    ) -> Option<usize> {
        // Read the source's geometry into immutable snapshots so the
        // borrow checker accepts the mutable target update below.
        let (source_octree, source_brick_pool_offset, source_attr_pool_offset) = {
            let entry = self.asset_cache.get(source)?;
            // We hold the octree node slice by clone for the lookup loop;
            // brick / leaf-attr pools are shared so we keep refs via
            // self below.
            let nodes = entry.cpu_octree.as_slice().to_vec();
            let depth = entry.cpu_octree.depth();
            let voxel_size = entry.cpu_octree.base_voxel_size();
            ((nodes, depth, voxel_size), entry.brick_start, entry.leaf_attr_slot_start)
        };
        let (source_nodes, source_depth, _source_voxel_size) = source_octree;
        let _ = source_brick_pool_offset; // brick refs in nodes are already scene-global

        let target_depth = self.asset_cache.get(target)?.cpu_octree.depth();
        debug_assert_eq!(
            source_depth, target_depth,
            "halo refresh assumes tiles share LOD level",
        );
        let s: i32 = 1 << target_depth;

        // For each cell on the shared boundary slab, resolve source's
        // current state. Build a snapshot first so we can mutate
        // target without holding a source borrow.
        let (axis, source_sign, target_lo_in_target_frame) = axis_for_face(target_face);
        let mut updates: Vec<(IVec3, SourceCellState)> =
            Vec::with_capacity((s as usize * s as usize * TILE_HALO_VOXELS as usize).max(1));

        for d in 0..TILE_HALO_VOXELS {
            for u in 0..s {
                for v in 0..s {
                    // Map (u, v, d) into the source's frame:
                    let source_coord = match axis {
                        0 => match source_sign {
                            // Target_face = NX → source coord at +X side: (S-1-d, u, v)
                            -1 => IVec3::new(s - 1 - d, u, v),
                            // Target_face = PX → source coord at -X side: (d, u, v)
                            _ => IVec3::new(d, u, v),
                        },
                        1 => match source_sign {
                            -1 => IVec3::new(u, s - 1 - d, v),
                            _ => IVec3::new(u, d, v),
                        },
                        _ => match source_sign {
                            -1 => IVec3::new(u, v, s - 1 - d),
                            _ => IVec3::new(u, v, d),
                        },
                    };
                    // Target coord — flip the d sign into the halo band:
                    let target_coord = match axis {
                        0 => match target_lo_in_target_frame {
                            // -X face: coord.x in [-halo, 0); d=0 → -1, d=1 → -2.
                            -1 => IVec3::new(-1 - d, u, v),
                            // +X face: coord.x in [S, S+halo); d=0 → S, d=1 → S+1.
                            _ => IVec3::new(s + d, u, v),
                        },
                        1 => match target_lo_in_target_frame {
                            -1 => IVec3::new(u, -1 - d, v),
                            _ => IVec3::new(u, s + d, v),
                        },
                        _ => match target_lo_in_target_frame {
                            -1 => IVec3::new(u, v, -1 - d),
                            _ => IVec3::new(u, v, s + d),
                        },
                    };
                    let state = resolve_source_cell(
                        &source_nodes,
                        source_depth,
                        &self.brick_pool,
                        &self.leaf_attr_pool,
                        source_coord,
                    );
                    updates.push((target_coord, state));
                }
            }
        }

        // Apply updates to target.halo_cells + the leaf_attr_pool. We
        // need a few field-level splits to avoid double-borrowing
        // self.asset_cache + self.leaf_attr_pool concurrently.
        let mut changed: usize = 0;
        let target_attr_base = self
            .asset_cache
            .get(target)
            .map(|e| (e.leaf_attr_slot_start, e.leaf_attr_slot_count))?;
        let target_lo = target_attr_base.0;
        let target_hi = target_lo + target_attr_base.1;

        // Build a lookup from coord → index into halo_cells for the
        // target's existing entries, then iterate updates against it.
        let existing: std::collections::HashMap<IVec3, (usize, u32)> = {
            let entry = self.asset_cache.get(target)?;
            entry
                .halo_cells
                .iter()
                .enumerate()
                .map(|(i, &(c, s))| (c, (i, s)))
                .collect()
        };

        // Pending mutations to target's halo_cells.
        let mut new_entries: Vec<(IVec3, u32)> = Vec::new();
        let mut overwrite_at: Vec<(usize, u32)> = Vec::new();
        let mut drop_indices: HashSet<usize> = HashSet::new();
        let mut new_extra_slots: Vec<u32> = Vec::new();

        for (coord, state) in updates {
            let existing_entry = existing.get(&coord).copied();
            match (state, existing_entry) {
                (SourceCellState::Empty, None) => {
                    // No-op; neither side has a cell here.
                }
                (SourceCellState::Empty, Some((idx, _slot))) => {
                    // Source went empty; drop the halo entry. We
                    // intentionally don't deallocate the slot — if
                    // it's in the bake range, release_asset frees
                    // it; if it's an extra slot, it stays in
                    // `halo_extra_slots` (a small over-count is OK).
                    drop_indices.insert(idx);
                    changed += 1;
                }
                (SourceCellState::Interior, Some((idx, slot))) if slot == CELL_INTERIOR => {
                    // Already INTERIOR; nothing to update. Suppress
                    // unused-binding warning.
                    let _ = idx;
                }
                (SourceCellState::Interior, Some((idx, _slot))) => {
                    // Was surface, now interior. Drop the explicit
                    // entry and add a CELL_INTERIOR sentinel entry.
                    drop_indices.insert(idx);
                    new_entries.push((coord, CELL_INTERIOR));
                    changed += 1;
                }
                (SourceCellState::Interior, None) => {
                    new_entries.push((coord, CELL_INTERIOR));
                    changed += 1;
                }
                (SourceCellState::Surface(_attr, source_slot), Some((idx, target_slot))) => {
                    if target_slot == CELL_INTERIOR {
                        // Was bulk-solid, now surface. Allocate a
                        // fresh slot for the explicit LeafAttr.
                        drop_indices.insert(idx);
                        let new_slot = self.allocate_halo_slot_for(
                            target,
                            source_slot,
                            &mut new_extra_slots,
                        )?;
                        new_entries.push((coord, new_slot));
                        changed += 1;
                    } else {
                        // Existing real slot. Overwrite its LeafAttr
                        // in place (only changed if the LeafAttr is
                        // actually different — but cheap to write
                        // unconditionally).
                        overwrite_at.push((idx, source_slot));
                        // Conservative: count as changed; the mesh
                        // re-extract will pick up any normal/material
                        // diff.
                        changed += 1;
                    }
                }
                (SourceCellState::Surface(_attr, source_slot), None) => {
                    let new_slot = self.allocate_halo_slot_for(
                        target,
                        source_slot,
                        &mut new_extra_slots,
                    )?;
                    new_entries.push((coord, new_slot));
                    changed += 1;
                }
            }
        }

        // Apply overwrites by copying the source slot's LeafAttr
        // into the target slot. Both slots live in `self.leaf_attr_pool`.
        {
            // Snapshot the target halo_cells we'll need slot ids for.
            let target_halo: Vec<(IVec3, u32)> = self
                .asset_cache
                .get(target)
                .map(|e| e.halo_cells.clone())
                .unwrap_or_default();
            for (idx, source_slot) in &overwrite_at {
                let Some((_coord, target_slot)) = target_halo.get(*idx) else { continue };
                if *target_slot >= target_lo && *target_slot < target_hi {
                    // Slot is in target's bake range — overwrite in place.
                    let attr = *self.leaf_attr_pool.get(*source_slot);
                    *self.leaf_attr_pool.get_mut(*target_slot) = attr;
                } else if self
                    .asset_cache
                    .get(target)
                    .map(|e| e.halo_extra_slots.contains(target_slot))
                    .unwrap_or(false)
                {
                    let attr = *self.leaf_attr_pool.get(*source_slot);
                    *self.leaf_attr_pool.get_mut(*target_slot) = attr;
                } else {
                    // Slot reference doesn't belong to target — skip
                    // defensively. This shouldn't happen but a stale
                    // halo entry from a previous bug could appear.
                }
                let _ = _coord;
            }
        }

        // Commit halo_cells mutations.
        if let Some(entry) = self.asset_cache.get_mut(target) {
            if !drop_indices.is_empty() {
                let mut keep: Vec<(IVec3, u32)> = Vec::with_capacity(
                    entry.halo_cells.len().saturating_sub(drop_indices.len()),
                );
                for (i, e) in entry.halo_cells.iter().enumerate() {
                    if !drop_indices.contains(&i) {
                        keep.push(*e);
                    }
                }
                entry.halo_cells = keep;
            }
            entry.halo_cells.extend(new_entries);
            entry.halo_extra_slots.extend(new_extra_slots);
        }

        Some(changed)
    }

    /// Allocate a fresh leaf-attr pool slot for a new halo entry and
    /// copy the source's LeafAttr value into it. The returned slot
    /// is also appended to `new_extra_slots` so the caller can fold
    /// it into the target's `halo_extra_slots` HashSet after the
    /// borrow loop ends.
    fn allocate_halo_slot_for(
        &mut self,
        _target: AssetHandle,
        source_slot: u32,
        new_extra_slots: &mut Vec<u32>,
    ) -> Option<u32> {
        let attr = *self.leaf_attr_pool.get(source_slot);
        let color = self.leaf_attr_pool.color(source_slot);
        let new_slot = self.leaf_attr_pool.allocate()?;
        *self.leaf_attr_pool.get_mut(new_slot) = attr;
        if color != 0 {
            self.leaf_attr_pool.set_color(new_slot, color);
        }
        new_extra_slots.push(new_slot);
        Some(new_slot)
    }

    /// Phase 4.2b — slab-only re-extract on halo refresh.
    ///
    /// Replaces `rebuild_asset_mesh_haloed`'s full-tile re-extract
    /// (hundreds of ms on a Tier-2 tile) with a face-band slab
    /// re-extract that mirrors the sculpt V2 per-cluster patch pattern
    /// in [`super::sculpt::ArvxSceneManager::rebuild_dirty_clusters`].
    ///
    /// Steps:
    /// 1. Query the cluster spatial index for LOD-0 clusters whose grid
    ///    AABB overlaps the face-band slab (cube-coord AABB).
    /// 2. Rayon-parallel filter each cluster's tris — drop any tri with
    ///    a vertex inside the slab obj-local AABB. In-place rewrite the
    ///    kept indices at the cluster's existing `index_offset`; return
    ///    the unused tail to the slab allocator via `free_index_range`.
    /// 3. Re-extract the slab region via [`collect_cell_map_in_region`]
    ///    + [`extract_mesh_region_from_cells_pooled_haloed`] — the same
    ///    pipeline the sculpt V2 patch uses, scoped to the slab's
    ///    cell range.
    /// 4. Append the extracted verts + indices as a single LOD-0 patch
    ///    cluster (`CLUSTER_FLAG_LOD_DIRTY`, `DAG_GROUP_NONE`) and
    ///    insert it into the cluster spatial index.
    /// 5. CC-walk LOD_DIRTY marking via
    ///    [`super::sculpt::mark_lod_dirty_chains`] over the slab AABB
    ///    so the shader's LOD selector drops LOD>0 ancestors at the
    ///    face band and admits the new patch unconditionally.
    ///
    /// **What the slab covers.** The halo refresh updates cells at the
    /// face band (e.g. `coord.x ∈ [S, S+halo)` for `target_face = PX`).
    /// SN cubes whose corners include any of these cells have changed
    /// vertex positions: `cube.x = S-1` (one interior corner + one halo
    /// corner) and `cube.x = S` (both halo corners — only emitted in
    /// the initial bake via halo-cell iteration, not in our slab
    /// re-extract).
    ///
    /// **Filter slab** (cube-coord vertex bounds, used for the per-tri
    /// drop test): `cube.x ∈ [S-3, S+1)` for `PX`, accounting for
    /// (a) all cube positions the slab extract emits (cube.x ∈ {S-3,
    /// S-2, S-1} from interior cells in the pad range), plus (b) halo
    /// cubes (cube.x = S) that the initial bake produced but the slab
    /// won't re-emit. Kept-cluster tris with all 3 verts strictly
    /// outside the AABB stay; the rest are dropped and the slab
    /// extract's emission fills in.
    ///
    /// **Extract region** (cell coords, narrower than the filter):
    /// `cell.x ∈ [S-1, S+1)` for `PX`. The extractor's automatic +1
    /// pad expands to `[S-2, S+2)`, so interior solid cells at
    /// `C.x ∈ {S-2, S-1}` iterate and contribute the changed cube
    /// vertices.
    ///
    /// **Seam coverage at cube.x = S.** Initial-bake halo emissions
    /// (cube.x = S, generated from `halo_cells` iterating in
    /// `extract_surface_mesh_haloed`) are filtered out and NOT
    /// re-emitted here. Sufficient seam coverage remains from the
    /// neighbour (source) tile's own emissions at the shared boundary.
    /// Trade-off accepted for V1 — see
    /// `project_terrain_phase4_session_endpoint` for the 4.2b followup
    /// this implements.
    fn rebuild_face_band_clusters(&mut self, handle: AssetHandle, target_face: u8) {
        let t0 = std::time::Instant::now();

        // Resolve asset geometry config + slab corners.
        let (
            depth,
            base_vs,
            grid_origin,
            extract_lo,
            extract_hi,
            filter_lo,
            filter_hi,
            slab_aabb_min_obj,
            slab_aabb_max_obj,
        ) = {
            let Some(entry) = self.asset_cache.get(handle) else { return };
            if entry.meshlet_clusters.is_empty() {
                return;
            }
            let depth = entry.spatial_handle.depth;
            let base_vs = entry.spatial_handle.base_voxel_size;
            let extent_f = (1u32 << depth) as f32 * base_vs;
            let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
            let grid_origin = aabb_center - Vec3::splat(extent_f * 0.5);

            let s = 1i32 << depth;
            let (extract_lo, extract_hi, filter_lo, filter_hi) =
                slab_grid_for_face(target_face, s);

            // Obj-local AABB for the per-vertex filter test. filter_lo /
            // filter_hi are in cube coords; a vertex from SN cube `C`
            // has position in `[C, C+1) * vs + grid_origin`, so the
            // inclusive AABB `[filter_lo * vs + grid_origin,
            // filter_hi * vs + grid_origin]` catches every vertex
            // whose source cube is in the cube range `[filter_lo,
            // filter_hi)`.
            let slab_min = grid_origin + filter_lo.as_vec3() * base_vs;
            let slab_max = grid_origin + filter_hi.as_vec3() * base_vs;

            (
                depth, base_vs, grid_origin, extract_lo, extract_hi,
                filter_lo, filter_hi, slab_min, slab_max,
            )
        };

        // Phase 1: query LOD-0 clusters overlapping the slab.
        let dirty = self.clusters_in_brush_grid_aabb(handle, filter_lo, filter_hi);

        // Phase 2: rayon-parallel per-tri filter on dirty clusters.
        // Drops tris with any vertex inside the slab AABB; kept tris
        // get rewritten in-place at the cluster's existing slot.
        let results: Vec<(u32, Vec<u32>)> = if dirty.is_empty() {
            Vec::new()
        } else {
            use rayon::prelude::*;
            let Some(entry) = self.asset_cache.get(handle) else { return };
            let clusters = &entry.meshlet_clusters;
            let indices = &entry.mesh_indices;
            let verts = &entry.mesh_vertices;
            dirty
                .par_iter()
                .filter_map(|&cid| {
                    let c = &clusters[cid as usize];
                    let start = c.index_offset as usize;
                    let count = c.index_count as usize;
                    // Cluster-AABB rejection: skip clusters whose AABB
                    // is fully outside the slab AABB. Mirrors the
                    // sphere-AABB rejection in sculpt's V2 patch.
                    let cluster_aabb_min = Vec3::from(c.aabb_min);
                    let cluster_aabb_max = Vec3::from(c.aabb_max);
                    if cluster_aabb_max.x < slab_aabb_min_obj.x
                        || cluster_aabb_min.x > slab_aabb_max_obj.x
                        || cluster_aabb_max.y < slab_aabb_min_obj.y
                        || cluster_aabb_min.y > slab_aabb_max_obj.y
                        || cluster_aabb_max.z < slab_aabb_min_obj.z
                        || cluster_aabb_min.z > slab_aabb_max_obj.z
                    {
                        return None;
                    }

                    let inside = |p: Vec3| -> bool {
                        p.x >= slab_aabb_min_obj.x
                            && p.x <= slab_aabb_max_obj.x
                            && p.y >= slab_aabb_min_obj.y
                            && p.y <= slab_aabb_max_obj.y
                            && p.z >= slab_aabb_min_obj.z
                            && p.z <= slab_aabb_max_obj.z
                    };

                    let mut out = Vec::with_capacity(count);
                    for tri_start in (start..start + count).step_by(3) {
                        let i0 = indices[tri_start];
                        let i1 = indices[tri_start + 1];
                        let i2 = indices[tri_start + 2];
                        let p0 = Vec3::from(verts[i0 as usize].local_pos);
                        let p1 = Vec3::from(verts[i1 as usize].local_pos);
                        let p2 = Vec3::from(verts[i2 as usize].local_pos);
                        if !inside(p0) && !inside(p1) && !inside(p2) {
                            out.push(i0);
                            out.push(i1);
                            out.push(i2);
                        }
                    }
                    Some((cid, out))
                })
                .collect()
        };

        // Phase 3: sequential merge — write each cluster's kept indices
        // back in-place at its original slot, return the freed tail to
        // the slab allocator.
        let mut total_dropped_tris = 0usize;
        let mut total_kept_tris = 0usize;
        {
            let Some(entry) = self.asset_cache.get_mut(handle) else { return };
            for (cid, kept) in &results {
                let cluster = &entry.meshlet_clusters[*cid as usize];
                let old_offset = cluster.index_offset;
                let old_count = cluster.index_count;
                let new_count = kept.len() as u32;
                total_kept_tris += (new_count as usize) / 3;
                total_dropped_tris += ((old_count - new_count) as usize) / 3;
                if new_count > 0 {
                    entry.mesh_indices_write_at(old_offset, kept);
                }
                if new_count < old_count {
                    entry.free_index_range(old_offset + new_count, old_count - new_count);
                }
                entry.meshlet_clusters[*cid as usize].index_count = new_count;
            }
        }

        // Phase 4: re-extract slab region.
        let (slab_verts, slab_indices, cells_count) = {
            let Some(entry) = self.asset_cache.get(handle) else { return };
            // Pad collect by +3 each side so the extractor's pad gets
            // boundary cells for 8-corner classification (mirrors
            // sculpt.rs's `cells_min = brush_lo - IVec3::splat(3)`).
            let cells_lo = extract_lo - IVec3::splat(3);
            let cells_hi = extract_hi + IVec3::splat(3);
            let cells = collect_cell_map_in_region(
                entry.cpu_octree.as_slice(),
                depth,
                self.brick_pool.as_slice(),
                cells_lo,
                cells_hi,
            );
            let cells_count = cells.len();
            let (verts, indices) = extract_mesh_region_from_cells_pooled_haloed(
                &mut self.sculpt_extract_scratch,
                &cells,
                extract_lo,
                extract_hi,
                entry.cpu_octree.as_slice(),
                depth,
                base_vs,
                grid_origin,
                self.brick_pool.as_slice(),
                self.leaf_attr_pool.as_slice(),
                self.leaf_attr_pool.bones_as_slice(),
                &entry.halo_cells,
                Some(&entry.sculpt_owned_slots),
            );
            (verts, indices, cells_count)
        };

        // Phase 5: append slab as a fresh LOD-0 patch cluster.
        let patch_indices_count = slab_indices.len();
        let patch_verts_count = slab_verts.len();
        if !slab_verts.is_empty() {
            let mut patch_min = [f32::INFINITY; 3];
            let mut patch_max = [f32::NEG_INFINITY; 3];
            for v in &slab_verts {
                for k in 0..3 {
                    if v.local_pos[k] < patch_min[k] {
                        patch_min[k] = v.local_pos[k];
                    }
                    if v.local_pos[k] > patch_max[k] {
                        patch_max[k] = v.local_pos[k];
                    }
                }
            }

            let Some(entry) = self.asset_cache.get_mut(handle) else { return };
            let vertex_offset = entry.mesh_vertices.len() as u32;
            let append_start_bytes = (vertex_offset as usize
                * std::mem::size_of::<crate::mesh_pass::MeshVertex>())
                as u32;
            entry.mesh_vertices.extend_from_slice(&slab_verts);
            let append_len_bytes = (slab_verts.len()
                * std::mem::size_of::<crate::mesh_pass::MeshVertex>())
                as u32;
            if append_len_bytes > 0 {
                entry
                    .mesh_vertices_dirty
                    .mark(append_start_bytes, append_len_bytes);
            }

            let patch_index_count = slab_indices.len() as u32;
            let new_index_offset = if patch_index_count > 0 {
                let offset = entry.alloc_index_range(patch_index_count);
                let rebased: Vec<u32> = slab_indices
                    .iter()
                    .map(|&i| i + vertex_offset)
                    .collect();
                entry.mesh_indices_write_at(offset, &rebased);
                offset
            } else {
                0
            };

            let patch_cluster = MeshletCluster {
                aabb_min: patch_min,
                _pad0: 0.0,
                aabb_max: patch_max,
                index_offset: new_index_offset,
                index_count: patch_index_count,
                lod_level: 0,
                // Born dirty — the patch is outside the bake-time DAG
                // so the LOD selector's Karis admit has nothing to
                // project against. Force unconditional admit.
                flags: CLUSTER_FLAG_LOD_DIRTY,
                cluster_error: 0.0,
                parent_group_error: PARENT_GROUP_ERROR_ROOT,
                // Standalone — no DAG chain. CC walks from
                // brush-touched LOD-0 clusters don't traverse through
                // it, which is correct.
                group_above_idx: DAG_GROUP_NONE,
                group_below_idx: DAG_GROUP_NONE,
                _pad3: 0,
            };
            let patch_cluster_id = entry.meshlet_clusters.len() as u32;
            entry.meshlet_clusters.push(patch_cluster);
            entry.cluster_spatial_index.insert(
                patch_cluster_id,
                &patch_cluster,
                grid_origin,
                base_vs,
            );
        }

        // Phase 6: CC-walk LOD_DIRTY marking over slab AABB. Forces
        // the LOD selector to drop dirty ancestors and admit dirty
        // LOD-0 leaves in the chains the slab touches; without it,
        // coarse LOD>0 clusters at the boundary would render with
        // stale (pre-refresh) vertex positions.
        let _walk_visited = if !dirty.is_empty() {
            let Some(entry) = self.asset_cache.get_mut(handle) else { return };
            super::sculpt::mark_lod_dirty_chains(
                entry,
                &dirty,
                slab_aabb_min_obj,
                slab_aabb_max_obj,
            )
        } else {
            0
        };

        // Phase 7: bookkeeping flags. `bump_geometry_epoch` happens
        // at the `apply_halo_refresh` call site, mirroring the
        // sculpt path.
        if let Some(entry) = self.asset_cache.get_mut(handle) {
            entry.mesh_dirty = true;
            entry.clusters_dirty = true;
        }

        if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
            eprintln!(
                "[halo-refresh] band re-extract handle={:?} face={} \
                 dirty_clusters={} cells={} kept_tris={} dropped_tris={} \
                 slab_verts={} slab_indices={} ({:.2}ms)",
                handle,
                target_face,
                dirty.len(),
                cells_count,
                total_kept_tris,
                total_dropped_tris,
                patch_verts_count,
                patch_indices_count,
                t0.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    /// Full-asset re-extract using the asset's `halo_cells` for the
    /// haloed mesh-extract path. Mirrors `rebuild_asset_mesh` but
    /// uses `extract_surface_mesh_haloed` with halo width 2.
    /// Called from `apply_halo_refresh` after the halo cells are
    /// updated.
    fn rebuild_asset_mesh_haloed(&mut self, handle: AssetHandle) {
        let t0 = std::time::Instant::now();

        let (depth, voxel_size, grid_origin, halo_cells) = {
            let Some(entry) = self.asset_cache.get(handle) else { return; };
            let depth = entry.spatial_handle.depth;
            let voxel_size = entry.spatial_handle.base_voxel_size;
            let extent = (1u32 << depth) as f32 * voxel_size;
            let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
            let grid_origin = aabb_center - Vec3::splat(extent * 0.5);
            (depth, voxel_size, grid_origin, entry.halo_cells.clone())
        };

        let (vertices, indices_unc) = {
            let entry = self.asset_cache.get(handle).expect("just confirmed above");
            let nodes = entry.cpu_octree.as_slice();
            extract_surface_mesh_haloed(
                nodes,
                depth,
                voxel_size,
                grid_origin,
                self.brick_pool.as_slice(),
                self.leaf_attr_pool.as_slice(),
                self.leaf_attr_pool.bones_as_slice(),
                &halo_cells,
                TILE_HALO_VOXELS as u32,
                // Halo refresh re-extracts the whole target tile —
                // bias the SN tie-break toward sculpt-allocated
                // slots so brush-added cells along the boundary
                // keep their material/colour against neighbour
                // halo cells (which always carry the procedural
                // material).
                Some(&entry.sculpt_owned_slots),
            )
        };

        if vertices.is_empty() {
            if let Some(entry) = self.asset_cache.get_mut(handle) {
                entry.mesh_vertices.clear();
                entry.mesh_indices.clear();
                entry.meshlet_clusters.clear();
                entry.bake_time_cluster_count = 0;
                entry.mesh_lod0_index_count = 0;
                entry.reset_mesh_indices_slab();
                entry.mesh_dirty = true;
                entry.clusters_dirty = true;
                entry.cluster_spatial_index =
                    super::cluster_spatial_index::ClusterSpatialIndex::new();
            }
            return;
        }

        // Use the same LOD_LEVELS=1 fallback the in-line sculpt
        // rebuild_asset_mesh uses — full multi-level Karis-Nanite
        // simplification at refresh time isn't worth the cost; the
        // boundary clusters get correct LOD-0 detail and the rest of
        // the DAG retains the asset's pre-sculpt structure for now.
        let dag = build_cluster_dag_with_levels(&vertices, &indices_unc, 1);
        let mesh_lod0_index_count = dag.lod0_index_range.1 - dag.lod0_index_range.0;

        let Some(entry) = self.asset_cache.get_mut(handle) else { return; };
        entry.mesh_vertices = vertices;
        entry.mesh_indices = dag.indices;
        entry.meshlet_clusters = dag.clusters;
        entry.bake_time_cluster_count = entry.meshlet_clusters.len() as u32;
        entry.mesh_lod0_index_count = mesh_lod0_index_count;
        entry.reset_mesh_indices_slab();
        // Full re-extract — mirror the IBO reset on the VBO side so the
        // upload doesn't carry stale prefix bytes.
        entry.mesh_vertices_dirty.clear();
        let vbo_bytes = (entry.mesh_vertices.len()
            * std::mem::size_of::<crate::mesh_pass::MeshVertex>())
            as u32;
        if vbo_bytes > 0 {
            entry.mesh_vertices_dirty.mark_full(vbo_bytes);
        }
        entry.mesh_dirty = true;
        entry.clusters_dirty = true;
        entry
            .cluster_spatial_index
            .rebuild(&entry.meshlet_clusters, grid_origin, voxel_size);

        if std::env::var("ARVX_TERRAIN_DEBUG").is_ok() {
            eprintln!(
                "[halo-refresh] mesh re-extract handle={:?} verts={} indices={} clusters={} ({:.2}ms)",
                handle,
                entry.mesh_vertices.len(),
                entry.mesh_indices.len(),
                entry.meshlet_clusters.len(),
                t0.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
}

/// Compute the slab corners for a face-band re-extract.
///
/// Returns `(extract_lo, extract_hi, filter_lo, filter_hi)`:
///
/// - `extract_lo, extract_hi` — half-open cell-coord region for
///   [`extract_mesh_region_from_cells_pooled_haloed`]'s `region_min /
///   region_max` args. Two cells deep along the face-axis (e.g.
///   `cell.x ∈ [S-1, S+1)` for `PX`), full `[0, S)` perpendicular.
///   The extractor's automatic +1 pad expands to 4 cells deep, but
///   only the interior cells (`{S-2, S-1}` for PX, `{0, 1}` for NX)
///   actually contribute since halo cells aren't in the cell map
///   that `collect_cell_map_in_region` produces.
///
/// - `filter_lo, filter_hi` — cube-coord AABB for the per-vertex
///   filter test. Wider than the extract region by one cube on each
///   side along the face-axis: it covers (a) every cube the slab
///   extract may emit and (b) the halo cubes that the initial bake
///   emitted from halo-cell iteration but the slab re-extract won't
///   re-emit. For PX, this is `cube.x ∈ [S-3, S+1)`; for NX,
///   `cube.x ∈ [-2, 2)`. Perpendicular axes span `[-1, S+1)` to catch
///   Y/Z cubes at the boundary.
fn slab_grid_for_face(face: u8, s: i32) -> (IVec3, IVec3, IVec3, IVec3) {
    // Perpendicular-axis range — same regardless of which face. SN
    // cubes at the perpendicular boundaries have one corner at the
    // tile interior and one in the inner halo ring, so the filter
    // extends one cube beyond `[0, S)` on each side.
    let perp_lo = -1;
    let perp_hi = s + 1;
    match face {
        FACE_PX => (
            IVec3::new(s - 1, 0, 0),
            IVec3::new(s + 1, s, s),
            IVec3::new(s - 3, perp_lo, perp_lo),
            IVec3::new(s + 1, perp_hi, perp_hi),
        ),
        FACE_NX => (
            IVec3::new(-1, 0, 0),
            IVec3::new(1, s, s),
            IVec3::new(-2, perp_lo, perp_lo),
            IVec3::new(2, perp_hi, perp_hi),
        ),
        FACE_PY => (
            IVec3::new(0, s - 1, 0),
            IVec3::new(s, s + 1, s),
            IVec3::new(perp_lo, s - 3, perp_lo),
            IVec3::new(perp_hi, s + 1, perp_hi),
        ),
        FACE_NY => (
            IVec3::new(0, -1, 0),
            IVec3::new(s, 1, s),
            IVec3::new(perp_lo, -2, perp_lo),
            IVec3::new(perp_hi, 2, perp_hi),
        ),
        FACE_PZ => (
            IVec3::new(0, 0, s - 1),
            IVec3::new(s, s, s + 1),
            IVec3::new(perp_lo, perp_lo, s - 3),
            IVec3::new(perp_hi, perp_hi, s + 1),
        ),
        FACE_NZ => (
            IVec3::new(0, 0, -1),
            IVec3::new(s, s, 1),
            IVec3::new(perp_lo, perp_lo, -2),
            IVec3::new(perp_hi, perp_hi, 2),
        ),
        _ => (IVec3::ZERO, IVec3::ZERO, IVec3::ZERO, IVec3::ZERO),
    }
}

/// Map a face index to `(axis, source_sign, target_sign)`.
///
/// - `axis` is 0, 1, or 2 (x, y, z).
/// - `source_sign` is `-1` when the source's interior boundary is on
///   the -axis side (target_face is the +axis face) and `+1` when
///   it's on the +axis side.
/// - `target_sign` is the band location on the target side, flipped
///   from `source_sign`.
///
/// For target_face = NX (1): source's +X is the data side
/// (`source_sign = -1`, meaning we read from coords near `S-1`); the
/// halo band on target lives in `[-halo, 0)`
/// (`target_lo_in_target_frame = -1`).
fn axis_for_face(target_face: u8) -> (u8, i8, i32) {
    match target_face {
        FACE_PX => (0, 1, 1),  // target halo on +X (coord.x = S, S+1); source's -X interior
        FACE_NX => (0, -1, -1), // target halo on -X (coord.x = -1, -2); source's +X interior
        FACE_PY => (1, 1, 1),
        FACE_NY => (1, -1, -1),
        FACE_PZ => (2, 1, 1),
        FACE_NZ => (2, -1, -1),
        _ => (0, 1, 1),
    }
}

/// Look up a single source-cell coord and classify its state.
fn resolve_source_cell(
    nodes: &[u32],
    depth: u8,
    brick_pool: &arvx_core::brick_pool::BrickPool,
    leaf_attr_pool: &arvx_core::leaf_attr_pool::LeafAttrPool,
    coord: IVec3,
) -> SourceCellState {
    let s = 1i32 << depth;
    if coord.x < 0 || coord.x >= s
        || coord.y < 0 || coord.y >= s
        || coord.z < 0 || coord.z >= s
    {
        return SourceCellState::Empty;
    }
    // Walk the packed octree to find the node at this coord.
    let mut idx = 0usize;
    for level in 0..depth {
        let node = nodes[idx];
        if node == EMPTY_NODE {
            return SourceCellState::Empty;
        }
        if node == INTERIOR_NODE {
            return SourceCellState::Interior;
        }
        if is_brick(node) {
            let bid = brick_id(node);
            let cx = coord.x & (BRICK_DIM - 1) as i32;
            let cy = coord.y & (BRICK_DIM - 1) as i32;
            let cz = coord.z & (BRICK_DIM - 1) as i32;
            let cell = brick_pool.get_cell(bid, cx as u32, cy as u32, cz as u32);
            return match cell {
                BRICK_EMPTY => SourceCellState::Empty,
                BRICK_INTERIOR => SourceCellState::Interior,
                slot => SourceCellState::Surface(*leaf_attr_pool.get(slot), slot),
            };
        }
        if is_leaf(node) {
            // Coarse leaf — covers a sub-tree. Treat the cell as
            // having the leaf's attrs.
            let slot = leaf_slot(node);
            return SourceCellState::Surface(*leaf_attr_pool.get(slot), slot);
        }
        // Branch — descend.
        let octant = octant_for_coord(coord, level, depth) as usize;
        idx = node as usize + octant;
    }
    let node = nodes[idx];
    if node == EMPTY_NODE {
        return SourceCellState::Empty;
    }
    if node == INTERIOR_NODE {
        return SourceCellState::Interior;
    }
    if is_brick(node) {
        let bid = brick_id(node);
        let cx = coord.x & (BRICK_DIM - 1) as i32;
        let cy = coord.y & (BRICK_DIM - 1) as i32;
        let cz = coord.z & (BRICK_DIM - 1) as i32;
        let cell = brick_pool.get_cell(bid, cx as u32, cy as u32, cz as u32);
        return match cell {
            BRICK_EMPTY => SourceCellState::Empty,
            BRICK_INTERIOR => SourceCellState::Interior,
            slot => SourceCellState::Surface(*leaf_attr_pool.get(slot), slot),
        };
    }
    if is_leaf(node) {
        let slot = leaf_slot(node);
        return SourceCellState::Surface(*leaf_attr_pool.get(slot), slot);
    }
    SourceCellState::Empty
}

/// Mirror of `arvx_core::sparse_octree::octant_for_coord` for
/// `IVec3` input (the public API takes `UVec3`). Inputs are already
/// known to be in `[0, 2^depth)`.
fn octant_for_coord(coord: IVec3, level: u8, depth: u8) -> u32 {
    let shift = depth - 1 - level;
    let bx = ((coord.x as u32 >> shift) & 1) as u32;
    let by = ((coord.y as u32 >> shift) & 1) as u32;
    let bz = ((coord.z as u32 >> shift) & 1) as u32;
    bx | (by << 1) | (bz << 2)
}

// Silence unused-import linter — UVec3 is used transitively by the
// octree's lookup API but we walk the buffer ourselves here.
#[allow(dead_code)]
fn _suppress_uvec3_unused() {
    let _ = UVec3::ZERO;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a tile of side `s = 16`, each face's slab corners must
    /// match the spec in the rustdoc on [`slab_grid_for_face`].
    #[test]
    fn slab_grid_for_face_each_axis() {
        let s: i32 = 16;
        let perp_lo = -1;
        let perp_hi = s + 1;

        // PX
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_PX, s);
        assert_eq!(ex_lo, IVec3::new(s - 1, 0, 0));
        assert_eq!(ex_hi, IVec3::new(s + 1, s, s));
        assert_eq!(fl_lo, IVec3::new(s - 3, perp_lo, perp_lo));
        assert_eq!(fl_hi, IVec3::new(s + 1, perp_hi, perp_hi));

        // NX
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_NX, s);
        assert_eq!(ex_lo, IVec3::new(-1, 0, 0));
        assert_eq!(ex_hi, IVec3::new(1, s, s));
        assert_eq!(fl_lo, IVec3::new(-2, perp_lo, perp_lo));
        assert_eq!(fl_hi, IVec3::new(2, perp_hi, perp_hi));

        // PY
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_PY, s);
        assert_eq!(ex_lo, IVec3::new(0, s - 1, 0));
        assert_eq!(ex_hi, IVec3::new(s, s + 1, s));
        assert_eq!(fl_lo, IVec3::new(perp_lo, s - 3, perp_lo));
        assert_eq!(fl_hi, IVec3::new(perp_hi, s + 1, perp_hi));

        // NY
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_NY, s);
        assert_eq!(ex_lo, IVec3::new(0, -1, 0));
        assert_eq!(ex_hi, IVec3::new(s, 1, s));
        assert_eq!(fl_lo, IVec3::new(perp_lo, -2, perp_lo));
        assert_eq!(fl_hi, IVec3::new(perp_hi, 2, perp_hi));

        // PZ
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_PZ, s);
        assert_eq!(ex_lo, IVec3::new(0, 0, s - 1));
        assert_eq!(ex_hi, IVec3::new(s, s, s + 1));
        assert_eq!(fl_lo, IVec3::new(perp_lo, perp_lo, s - 3));
        assert_eq!(fl_hi, IVec3::new(perp_hi, perp_hi, s + 1));

        // NZ
        let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_NZ, s);
        assert_eq!(ex_lo, IVec3::new(0, 0, -1));
        assert_eq!(ex_hi, IVec3::new(s, s, 1));
        assert_eq!(fl_lo, IVec3::new(perp_lo, perp_lo, -2));
        assert_eq!(fl_hi, IVec3::new(perp_hi, perp_hi, 2));
    }

    /// Filter slab strictly contains the extract region along the
    /// face axis on both sides, and is exactly 4 cubes wide along
    /// that axis — covering every cube position the slab extract may
    /// emit (3 cubes wide, from the pad-expanded solid cells) plus
    /// one initial-bake halo cube that the slab won't re-emit.
    #[test]
    fn slab_filter_strictly_contains_extract_on_face_axis() {
        let s: i32 = 16;
        for face in [
            FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ,
        ] {
            let (ex_lo, ex_hi, fl_lo, fl_hi) = slab_grid_for_face(face, s);
            let axis = match face {
                FACE_PX | FACE_NX => 0usize,
                FACE_PY | FACE_NY => 1,
                _ => 2,
            };
            assert!(
                fl_lo[axis] <= ex_lo[axis],
                "filter lo must extend at-or-below extract lo on face axis (face {face})"
            );
            assert!(
                fl_hi[axis] >= ex_hi[axis],
                "filter hi must extend at-or-above extract hi on face axis (face {face})"
            );
            // Filter is strictly wider on the face axis (at least one
            // direction must extend further than extract).
            let lo_extend = ex_lo[axis] - fl_lo[axis];
            let hi_extend = fl_hi[axis] - ex_hi[axis];
            assert!(
                lo_extend > 0 || hi_extend > 0,
                "filter must extend beyond extract on at least one side (face {face})"
            );
            // Exactly 4 cubes wide.
            assert_eq!(
                fl_hi[axis] - fl_lo[axis],
                4,
                "filter slab must be 4 cubes wide on the face axis (face {face})"
            );
        }
    }

    /// A vertex from SN cube at `cube.x = C` has obj-local position in
    /// `[(C * vs + grid_origin.x), ((C + 1) * vs + grid_origin.x)]`.
    /// The filter slab AABB built from `(filter_lo, filter_hi)` in
    /// cube coords (translated to obj-local floats via
    /// `* vs + grid_origin`) must contain every vertex whose source
    /// cube falls in `[filter_lo, filter_hi)` and exclude every
    /// vertex outside that range.
    #[test]
    fn filter_slab_aabb_covers_intended_cubes() {
        let s: i32 = 16;
        let vs: f32 = 1.0;
        let grid_origin = Vec3::ZERO;

        let (_ex_lo, _ex_hi, fl_lo, fl_hi) = slab_grid_for_face(FACE_PX, s);
        let slab_min = grid_origin + fl_lo.as_vec3() * vs;
        let slab_max = grid_origin + fl_hi.as_vec3() * vs;

        // A vertex from cube.x = S-3 with position at the cube's lower
        // edge (= cube.x * vs) is inside the filter slab.
        let v = Vec3::new((s - 3) as f32 * vs, 0.0, 0.0);
        assert!(
            v.x >= slab_min.x && v.x <= slab_max.x,
            "vertex from cube.x = S-3 must be inside filter slab"
        );

        // A vertex from cube.x = S-4 (outside the filter range) sits
        // at vertex.x ∈ [(S-4) * vs, (S-3) * vs]. The upper end is the
        // boundary; the lower end is strictly outside.
        let v = Vec3::new((s - 4) as f32 * vs, 0.0, 0.0);
        assert!(
            v.x < slab_min.x,
            "vertex from cube.x = S-4 at cube lo must be strictly outside filter slab"
        );

        // A vertex from cube.x = S has position in [S * vs, (S+1) * vs].
        // Both ends are inside the filter slab.
        let v = Vec3::new(s as f32 * vs, 0.0, 0.0);
        assert!(v.x >= slab_min.x && v.x <= slab_max.x);
        let v = Vec3::new((s + 1) as f32 * vs, 0.0, 0.0);
        assert!(
            v.x <= slab_max.x,
            "vertex from cube.x = S at cube hi must be at the filter slab's upper boundary"
        );

        // A vertex from cube.x = S+1 (halo cube outside iteration range)
        // is strictly outside the filter slab.
        let v = Vec3::new((s + 2) as f32 * vs, 0.0, 0.0);
        assert!(v.x > slab_max.x);
    }

    /// Bail-out paths must not panic on empty asset state.
    #[test]
    fn rebuild_face_band_clusters_bails_on_empty_clusters() {
        use crate::arvx_scene_manager::manager::ArvxSceneManager;
        use crate::arvx_scene_manager::types::AssetEntry;
        use arvx_core::sparse_octree::SparseOctree;
        use arvx_core::{Aabb, OctreeHandle};

        let mut sm = ArvxSceneManager::new(16);
        let depth: u8 = 4;
        let base_vs: f32 = 1.0;
        let extent = (1u32 << depth) as f32 * base_vs;
        let entry = AssetEntry {
            path: std::path::PathBuf::from("test:empty-tile"),
            refcount: 1,
            spatial_handle: OctreeHandle {
                root_offset: 0,
                len: 0,
                depth,
                base_voxel_size: base_vs,
            },
            voxel_size: base_vs,
            aabb: Aabb {
                min: Vec3::ZERO,
                max: Vec3::splat(extent),
            },
            voxel_count: 0,
            leaf_attr_slot_start: 0,
            leaf_attr_slot_count: 0,
            brick_start: 0,
            brick_count: 0,
            skinning: None,
            mesh_vertices: Vec::new(),
            mesh_indices: Vec::new(),
            mesh_indices_free_list: Vec::new(),
            mesh_indices_next_free: 0,
            mesh_indices_dirty: arvx_core::DirtyRanges::new(),
            mesh_vertices_dirty: arvx_core::DirtyRanges::new(),
            mesh_lod0_index_count: 0,
            bake_time_cluster_count: 0,
            meshlet_clusters: Vec::new(),
            dag_groups: Vec::new(),
            dag_consumed: Vec::new(),
            dag_produced: Vec::new(),
            cpu_octree: SparseOctree::new(depth, base_vs),
            sculpt_extra_slots: std::collections::HashSet::new(),
            sculpt_owned_slots: rustc_hash::FxHashSet::default(),
            halo_extra_slots: std::collections::HashSet::new(),
            halo_cells: Vec::new(),
            mesh_dirty: false,
            clusters_dirty: false,
            cluster_spatial_index:
                crate::arvx_scene_manager::cluster_spatial_index::ClusterSpatialIndex::new(),
        };
        let handle = sm.asset_cache.insert(entry);

        // Should be a no-op: no clusters to filter, no panic.
        sm.rebuild_face_band_clusters(handle, FACE_PX);
        let entry = sm.asset_cache.get(handle).expect("entry should still exist");
        assert!(entry.meshlet_clusters.is_empty());
        assert!(!entry.mesh_dirty, "no-op refresh should leave mesh_dirty unset");
    }
}
