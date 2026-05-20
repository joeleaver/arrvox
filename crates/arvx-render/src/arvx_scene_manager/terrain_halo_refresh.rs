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
use arvx_core::mesh_extract::{extract_surface_mesh_haloed, CELL_INTERIOR};
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
            self.rebuild_asset_mesh_haloed(op.target);
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
