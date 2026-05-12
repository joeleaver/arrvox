//! Sculpt-brush resolve — computes per-stamp edit lists without
//! mutating the asset's octree.
//!
//! Phase A overlay path (see memory `project_sculpt_phase_a_overlay_plan`):
//! each call hands the engine a list of `leaf_attr_id`s to insert into
//! the per-instance [`SculptOverlay`]. No mesh re-extract, no cluster
//! DAG rebuild, no `geometry_epoch` bump — the overlay rides through
//! the existing per-frame upload at [`crate::rkp_scene::FrameUpload::
//! instance_sculpts`].
//!
//! Why an overlay, not the Phase 2 octree mutation:
//!
//! * Brick-everywhere assets (the dominant case for `.rkp` mesh imports)
//!   silently no-oped under Phase 2's Carve because the kernel only saw
//!   LEAF nodes. The Phase A kernel extension emits `Remove` for BRICK
//!   cells too; the caller resolves grid_coord → leaf_attr_id here so
//!   the engine can drop slot ids into the overlay uniformly across LEAF
//!   and BRICK cells.
//! * Drag perf — no per-stamp re-bake. Cost is bounded by the brush
//!   AABB walk + binary-search insert into the overlay.
//! * Save path applies the accumulated overlay back into the octree
//!   in one shot (Phase A task #7). The octree stays the source of
//!   truth at rest; the overlay carries the in-session edits.
//!
//! Raise (Add) is deferred to Phase B and skipped here with a log line.

use glam::{Affine3A, IVec3, Vec3};

use rkp_core::mesh_cluster::{cluster_grid_aabb, cluster_overlaps_brush_grid_aabb};
use rkp_core::mesh_extract::{
    collect_cell_map, extract_mesh_region_from_cells, extract_surface_mesh,
};
use rkp_core::mesh_lod::build_cluster_dag_with_levels;
use rkp_core::sculpt::{
    apply_delta, brush_cell_range, compute_brush_edits, BrushMode, BrushOp, LeafEditOp,
};
use rkp_core::sparse_octree::{is_brick, is_leaf, leaf_slot, brick_id};
use rkp_core::brick_pool::{BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR};

use super::manager::RkpSceneManager;
use super::types::AssetHandle;

/// Outcome of [`RkpSceneManager::apply_sculpt_brush`]. The engine
/// merges `removed_leaf_attr_ids` into the per-entity
/// [`rkp_core::SculptOverlay`] and re-uploads on the next frame.
#[derive(Debug, Clone, Default)]
pub struct SculptApplyResult {
    /// `leaf_attr_id`s the brush carved away. Already de-duplicated
    /// and sorted ascending so the engine can `insert_batch` directly.
    /// May be empty when the brush footprint hit only empty / interior
    /// cells.
    pub removed_leaf_attr_ids: Vec<u32>,
    /// How many cells the kernel emitted as Remove (pre-filter).
    pub leaves_removed: usize,
    /// How many Add edits were skipped (Phase B). Logged so the user
    /// gets feedback if they switch to Raise while it's disabled.
    pub leaves_add_skipped: usize,
}

impl RkpSceneManager {
    /// Apply one sculpt brush stamp against an asset's geometry.
    ///
    /// Returns `None` when:
    /// * The handle is unknown.
    /// * The brush footprint produces no edits (outside the asset or
    ///   over empty / interior cells only).
    ///
    /// `brush_radius` is in world-space units. Object-local scale is
    /// applied via `entity_world.to_scale_rotation_translation()` (mean
    /// of the three scale axes, matching paint's convention).
    ///
    /// **Phase A:** does not mutate the octree, does not bump the
    /// geometry epoch. Carve only — Raise edits are skipped and counted.
    /// The caller is responsible for inserting `removed_leaf_attr_ids`
    /// into the per-entity [`rkp_core::SculptOverlay`].
    pub fn apply_sculpt_brush(
        &mut self,
        handle: AssetHandle,
        world_pos: Vec3,
        entity_world: Affine3A,
        brush_radius: f32,
        brush_falloff: f32,
        mode: BrushMode,
        material: u16,
    ) -> Option<SculptApplyResult> {
        if brush_radius <= 0.0 {
            return None;
        }

        // ── 1. Resolve grid coords ──────────────────────────────────
        let (op, depth, base_vs) = {
            let entry = self.asset_cache.get(handle)?;
            let depth = entry.spatial_handle.depth;
            let base_vs = entry.spatial_handle.base_voxel_size;
            let extent = (1u32 << depth) as f32 * base_vs;
            let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
            let asset_grid_origin = aabb_center - Vec3::splat(extent * 0.5);

            let inv_world = entity_world.inverse();
            let center_local = inv_world.transform_point3(world_pos);
            // Mean-of-axes scale, same as paint. Accurate enough; the
            // user can compensate via the radius slider.
            let (scale, _, _) = entity_world.to_scale_rotation_translation();
            let mean_scale = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
            let local_radius = brush_radius / mean_scale.max(1e-6);

            // Object-local → grid coords. `base_vs` is the finest-voxel
            // size, which matches the kernel's unit convention.
            let center_grid = (center_local - asset_grid_origin) / base_vs;
            let radius_grid = local_radius / base_vs;

            let op = BrushOp {
                center: center_grid,
                radius: radius_grid,
                falloff: brush_falloff,
                mode,
                material,
            };
            (op, depth, base_vs)
        };

        // ── 2. Compute edit list against current octree + brick pool. ─
        let delta = {
            let entry = self.asset_cache.get(handle)?;
            compute_brush_edits(&entry.cpu_octree, &self.brick_pool, op)
        };
        if delta.is_empty() {
            return None;
        }

        // ── 3. Resolve every Remove edit's grid coord → leaf_attr_id.
        //
        // Octree lookup returns the raw node value at the finest grid
        // coord. For LEAF nodes we just unpack the slot id. For BRICK
        // nodes we follow through into the brick pool — the cell's
        // value is either a slot id, `BRICK_EMPTY`, or `BRICK_INTERIOR`
        // (mesh-import bulk marker). Empty / interior cells get
        // filtered out here so the overlay only carries real
        // surface-leaf slots.
        let mut removed: Vec<u32> = Vec::new();
        let mut leaves_add_skipped: usize = 0;
        for edit in &delta.edits {
            match edit.op {
                LeafEditOp::Remove => {
                    let entry = self.asset_cache.get(handle)?;
                    let Some(node) = entry.cpu_octree.lookup(edit.coord) else {
                        continue;
                    };
                    if is_leaf(node) {
                        removed.push(leaf_slot(node));
                    } else if is_brick(node) {
                        let bid = brick_id(node);
                        let cx = edit.coord.x & (BRICK_DIM - 1);
                        let cy = edit.coord.y & (BRICK_DIM - 1);
                        let cz = edit.coord.z & (BRICK_DIM - 1);
                        let cell = self.brick_pool.get_cell(bid, cx, cy, cz);
                        if cell == BRICK_EMPTY || cell == BRICK_INTERIOR {
                            // Brick covers this finest cell, but the
                            // cell itself isn't a surface — nothing to
                            // carve.
                            continue;
                        }
                        removed.push(cell);
                    }
                    // EMPTY / INTERIOR / branch — no leaf_attr_id to
                    // remove. The kernel shouldn't emit Remove for
                    // those anyway, but defensive.
                }
                LeafEditOp::Add { .. } => {
                    // Phase B. Counted, not applied. The editor
                    // disables the Raise button so this path is only
                    // reachable from tests / scripted commands.
                    leaves_add_skipped += 1;
                }
                LeafEditOp::Empty | LeafEditOp::SetInterior => {
                    // R2b kernel variants — overlay path doesn't carry
                    // ADD info or INTERIOR bulk semantics, so these
                    // collapse to "no-op" for the legacy overlay. The
                    // real-geometry mutation path (R2c → apply_delta)
                    // will handle them properly.
                }
            }
        }

        let leaves_removed = removed.len();
        if removed.is_empty() && delta.count_added() == 0 && delta.count_interior() == 0 {
            return None;
        }

        // Sort + dedupe so the engine-side `insert_batch` walks the
        // smallest set possible. The kernel emits coords in row-major
        // order so adjacent finest-voxel cells inside one brick share
        // the brick's slot ids for at-most a handful of entries —
        // sorting collapses the obvious duplicates.
        removed.sort_unstable();
        removed.dedup();

        // ── 4. Real-geometry mutation (Phase B R2c). ────────────────
        //
        // Mutate the asset's octree + the scene's brick / leaf_attr
        // pools to reflect the delta. The overlay still rides through
        // `removed_leaf_attr_ids` for fragment-discard parity until
        // R4 (per-cluster re-extract) makes the mutation directly
        // visible by regenerating the mesh.
        //
        // Borrows: we split `self` field-by-field (`asset_cache`,
        // `brick_pool`, `leaf_attr_pool`) so `alloc_slot` can call
        // back into the leaf_attr_pool while apply_delta holds
        // mutable borrows of the octree + brick pool.
        let applied = {
            let Self {
                asset_cache,
                brick_pool,
                leaf_attr_pool,
                ..
            } = self;
            let entry = asset_cache.get_mut(handle)?;
            let octree = &mut entry.cpu_octree;
            apply_delta(
                octree,
                brick_pool,
                &delta,
                || {
                    leaf_attr_pool
                        .allocate()
                        .expect("leaf_attr_pool exhausted during sculpt apply")
                },
            )
        };

        // Write LeafAttrs for newly-allocated slots. The brush picks
        // the material; the normal is whatever the kernel emitted
        // (outward-from-brush-center today, R7 may refine).
        for (slot, attrs) in &applied.allocated_slots {
            *self.leaf_attr_pool.get_mut(*slot) = attrs.to_leaf_attr();
            // Default color (0) — sculpt-added cells fall back to the
            // material's base_color, same convention as paint's "no
            // override".
            self.leaf_attr_pool.set_color(*slot, 0);
        }
        // Release slots vacated by Remove / displaced-by-Add edits.
        // Done one-at-a-time since the slots aren't contiguous; the
        // pool's free-list absorbs them.
        for slot in &applied.freed_slots {
            self.leaf_attr_pool.deallocate_range(*slot, 1);
        }

        // ── 5. Mesh re-extract (Phase B R4-proper). ─────────────────
        //
        // Per-cluster region re-extract: only clusters whose grid AABB
        // intersects the brush get re-extracted; the rest keep their
        // existing `ClusterMesh` entries. Final flatten rebuilds the
        // flat VBO/IBO for upload (R4e will move this into the
        // renderer to drop the cached flat copies).
        //
        // Fallback to full re-extract when no clusters intersect AND
        // the delta actually mutated geometry — the brush landed in a
        // region no cluster currently owns, but new cells DID get
        // created in the octree (Raise into far empty space). The
        // full re-extract picks them up; cluster set then grows by
        // whatever meshopt produces. R5 will handle this case more
        // gracefully via re-clustering at idle.
        let used_per_cluster = self.rebuild_dirty_clusters(handle, &op);
        if !used_per_cluster {
            // Per-cluster path didn't fire (empty dirty set or asset
            // was empty before); fall back to full re-extract.
            self.rebuild_asset_mesh(handle);
        }

        // Bump the geometry epoch so the renderer re-uploads the
        // mutated octree / brick / leaf_attr buffers AND the new
        // mesh data on the next pre-frame pass.
        self.bump_geometry_epoch();

        eprintln!(
            "[sculpt] stamp handle={:?} mode={:?} edits={} removed={} \
             applied(adds={} freed={} interior={}) (depth={}, base_vs={:.5})",
            handle, mode, delta.len(), removed.len(),
            applied.allocated_slots.len(), applied.freed_slots.len(),
            delta.count_interior(), depth, base_vs,
        );

        Some(SculptApplyResult {
            removed_leaf_attr_ids: removed,
            leaves_removed,
            leaves_add_skipped,
        })
    }

    /// Find every LOD-0 cluster on an asset whose grid-coord AABB
    /// overlaps the brush's grid-coord AABB.
    ///
    /// **Phase B R3** — the per-cluster re-extract path's dirty-cluster
    /// query. Inputs are integer grid coords in the same convention as
    /// [`rkp_core::sculpt::compute_brush_edits`]: `brush_lo .. brush_hi`
    /// is half-open, the brush walks cells in `lo.x..hi.x` etc. Cluster
    /// AABBs are derived on the fly from each cluster's object-local
    /// float AABB via [`cluster_grid_aabb`] (1-voxel pad on each side
    /// so SN-cube neighbor cells are conservatively included).
    ///
    /// Returns LOD-0 (`lod_level == 0`) cluster ids only. Coarser
    /// levels regenerate via the R5 lazy-ancestor path when their
    /// children change. Order is ascending cluster id.
    ///
    /// Returns an empty vec for unknown handles, an empty cluster
    /// table, or zero overlap. ~50 µs on a 46 k-cluster asset; no
    /// allocation reuse — caller owns the returned Vec.
    pub fn clusters_in_brush_grid_aabb(
        &self,
        handle: AssetHandle,
        brush_lo: IVec3,
        brush_hi: IVec3,
    ) -> Vec<u32> {
        let Some(entry) = self.asset_cache.get(handle) else {
            return Vec::new();
        };
        if entry.meshlet_clusters.is_empty() {
            return Vec::new();
        }
        // Empty brush AABB → no clusters can intersect.
        if brush_lo.x >= brush_hi.x || brush_lo.y >= brush_hi.y || brush_lo.z >= brush_hi.z {
            return Vec::new();
        }
        let depth = entry.spatial_handle.depth;
        let base_vs = entry.spatial_handle.base_voxel_size;
        let extent = (1u32 << depth) as f32 * base_vs;
        let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
        let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

        let mut dirty = Vec::new();
        for (idx, c) in entry.meshlet_clusters.iter().enumerate() {
            if c.lod_level != 0 {
                continue;
            }
            let (cmin, cmax) = cluster_grid_aabb(c, grid_origin, base_vs);
            if cluster_overlaps_brush_grid_aabb(cmin, cmax, brush_lo, brush_hi) {
                dirty.push(idx as u32);
            }
        }
        dirty
    }

    /// Per-cluster region re-extract (Phase B R4c) — **append-only**.
    ///
    /// For each LOD-0 cluster whose grid AABB intersects the brush
    /// (via R3's `clusters_in_brush_grid_aabb`):
    /// 1. Re-extracts the cluster's grid AABB (1-cell pad) via
    ///    [`extract_mesh_region_from_cells`] against a single cell
    ///    map built once per stamp.
    /// 2. Appends the new verts to the tail of `mesh_vertices` and the
    ///    new (rebased) indices to the tail of `mesh_indices`.
    /// 3. Redirects the cluster's `index_offset` / `index_count` to
    ///    the appended range — the cluster's *old* indices become
    ///    orphaned (still in the buffer but unreferenced by any
    ///    cluster table entry, harmless besides a small per-stamp
    ///    leak that R5 will compact away).
    ///
    /// Sets `CLUSTER_FLAG_LOD_DIRTY` on every cluster of the asset so
    /// the shader admits only LOD-0 (force-fresh) and drops every
    /// LOD>0 (otherwise stale-LOD ancestors render their pre-sculpt
    /// geometry on top of the new LOD-0).
    ///
    /// **Why append-only.** An earlier per-cluster-owned design
    /// stored each cluster's `(local_vertices, local_indices)` in
    /// `cluster_meshes`, then re-flattened on every stamp. That
    /// duplicated boundary verts per-cluster — on a ~100 k-cluster
    /// multi-LOD asset that 2-3 ×'d the VBO size (~6.5 M verts vs
    /// ~2.5 M original), OOM'd "mesh asset vbo" on a 4-6 GB GPU on
    /// the user's first stamp. Append-only keeps the load-time flat
    /// VBO/IBO verbatim; each stamp grows it by ~the dirty clusters'
    /// post-extract size only.
    ///
    /// Returns `true` when the per-cluster path fired (at least one
    /// LOD-0 cluster was dirty); `false` when the dirty set is empty
    /// or the asset has no mesh — caller falls back to full re-extract.
    fn rebuild_dirty_clusters(&mut self, handle: AssetHandle, op: &BrushOp) -> bool {
        let t0 = std::time::Instant::now();

        let (depth, base_vs, grid_origin, brush_lo, brush_hi) = {
            let Some(entry) = self.asset_cache.get(handle) else { return false; };
            if entry.meshlet_clusters.is_empty() {
                return false;
            }
            let depth = entry.spatial_handle.depth;
            let base_vs = entry.spatial_handle.base_voxel_size;
            let extent_f = (1u32 << depth) as f32 * base_vs;
            let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
            let grid_origin = aabb_center - Vec3::splat(extent_f * 0.5);

            let extent = 1u32 << depth;
            let (lo, hi) = brush_cell_range(op, extent);
            let brush_lo = IVec3::new(lo.x as i32, lo.y as i32, lo.z as i32);
            let brush_hi = IVec3::new(hi.x as i32, hi.y as i32, hi.z as i32);
            (depth, base_vs, grid_origin, brush_lo, brush_hi)
        };

        if brush_lo.x >= brush_hi.x || brush_lo.y >= brush_hi.y || brush_lo.z >= brush_hi.z {
            return false;
        }

        let dirty = self.clusters_in_brush_grid_aabb(handle, brush_lo, brush_hi);
        if dirty.is_empty() {
            return false;
        }

        // Build the dense cell map ONCE for the stamp.
        let cells = {
            let Some(entry) = self.asset_cache.get(handle) else { return false; };
            collect_cell_map(
                entry.cpu_octree.as_slice(),
                depth,
                self.brick_pool.as_slice(),
            )
        };

        // V1 limitation (also true for the previous design): each
        // dirty cluster's region is bounded by its own pre-stamp grid
        // AABB only. Raise past existing cluster bounds creates cells
        // outside any cluster's region; those are mutated in the
        // octree but not captured in the mesh. R5 fixes via
        // re-clustering at idle.
        let mut total_appended_verts = 0usize;
        let mut total_appended_indices = 0usize;
        let mut total_freed_indices = 0usize;
        for &cid in &dirty {
            let (region_min, region_max) = {
                let Some(entry) = self.asset_cache.get(handle) else { return false; };
                let cluster = entry.meshlet_clusters[cid as usize];
                let (cmin, mut cmax) = cluster_grid_aabb(&cluster, grid_origin, base_vs);
                cmax += IVec3::ONE; // inclusive → half-open
                (cmin, cmax)
            };

            let (new_verts, new_indices) = {
                let Some(entry) = self.asset_cache.get(handle) else { return false; };
                extract_mesh_region_from_cells(
                    &cells,
                    region_min,
                    region_max,
                    entry.cpu_octree.as_slice(),
                    depth,
                    base_vs,
                    grid_origin,
                    self.brick_pool.as_slice(),
                    self.leaf_attr_pool.as_slice(),
                    self.leaf_attr_pool.bones_as_slice(),
                )
            };

            // New float AABB from re-extracted verts (collapses to
            // origin if the cluster's region is now empty).
            let mut new_aabb_min = [f32::INFINITY; 3];
            let mut new_aabb_max = [f32::NEG_INFINITY; 3];
            for v in &new_verts {
                for k in 0..3 {
                    if v.local_pos[k] < new_aabb_min[k] {
                        new_aabb_min[k] = v.local_pos[k];
                    }
                    if v.local_pos[k] > new_aabb_max[k] {
                        new_aabb_max[k] = v.local_pos[k];
                    }
                }
            }
            if new_verts.is_empty() {
                new_aabb_min = [0.0; 3];
                new_aabb_max = [0.0; 3];
            }

            // Append-only update: push the new verts/indices onto the
            // tail of the flat buffers, redirect the cluster to the
            // new range. Old indices stay in the buffer but no cluster
            // references them — orphaned, harmless leak.
            let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
            let vertex_offset = entry.mesh_vertices.len() as u32;
            let new_index_offset = entry.mesh_indices.len() as u32;
            entry.mesh_vertices.extend_from_slice(&new_verts);
            entry.mesh_indices.extend(
                new_indices.iter().map(|&i| i + vertex_offset),
            );

            total_appended_verts += new_verts.len();
            total_appended_indices += new_indices.len();

            let cluster = &mut entry.meshlet_clusters[cid as usize];
            total_freed_indices += cluster.index_count as usize;
            cluster.aabb_min = new_aabb_min;
            cluster.aabb_max = new_aabb_max;
            cluster.index_offset = new_index_offset;
            cluster.index_count = new_indices.len() as u32;
        }

        // Asset-wide LOD-0 clamp via CLUSTER_FLAG_LOD_DIRTY (see the
        // shader's admit-rule comment for the why).
        {
            let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
            for c in entry.meshlet_clusters.iter_mut() {
                c.flags |= rkp_core::mesh_cluster::CLUSTER_FLAG_LOD_DIRTY;
            }
        }

        // Re-sum the LOD-0 prefix length. With append-only, dirty
        // LOD-0 clusters' indices are now in the tail of mesh_indices
        // (past their original LOD-0-prefix slots), but the legacy
        // `direct_mode` debug-only dispatch still walks
        // `0 .. mesh_lod0_index_count`. Sum is approximate since the
        // appended tail can include both LOD-0 and (in future)
        // LOD>0 entries; keeping the formula identical to the
        // re-extract / build path so the field stays interpretable.
        let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
        entry.mesh_lod0_index_count = entry
            .meshlet_clusters
            .iter()
            .filter(|c| c.lod_level == 0)
            .map(|c| c.index_count)
            .sum();

        eprintln!(
            "[sculpt] per-cluster re-extract: handle={:?} dirty={} appended verts={} \
             indices={} freed indices={} total flat verts={} indices={} ({:.2}ms)",
            handle,
            dirty.len(),
            total_appended_verts,
            total_appended_indices,
            total_freed_indices,
            entry.mesh_vertices.len(),
            entry.mesh_indices.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
        );

        true
    }

    /// Re-extract the surface mesh + LOD-0 cluster table for one
    /// asset, replacing the cached `mesh_vertices` / `mesh_indices` /
    /// `meshlet_clusters` / `mesh_lod0_index_count`. The geometry
    /// upload path picks up the new buffers on the next geometry
    /// epoch.
    ///
    /// V1 (Phase B R4-minimal) re-extracts the **entire** asset on
    /// every stamp. Per-cluster re-extract is the perf path the full
    /// R4 covers; for now the cost is bounded by surface area + DAG
    /// build at `LOD_LEVELS=1` (no multi-level simplify).
    fn rebuild_asset_mesh(&mut self, handle: AssetHandle) {
        let t0 = std::time::Instant::now();

        // Snapshot the per-asset parameters we need to pass into
        // `extract_surface_mesh`. We need them as owned values (not
        // borrows of the asset entry) because the entry will be
        // re-borrowed mutably later to write back the new mesh.
        let Some(entry) = self.asset_cache.get(handle) else { return; };
        let depth = entry.spatial_handle.depth;
        let voxel_size = entry.spatial_handle.base_voxel_size;
        let extent = (1u32 << depth) as f32 * voxel_size;
        let aabb_center = (entry.aabb.min + entry.aabb.max) * 0.5;
        let grid_origin = aabb_center - Vec3::splat(extent * 0.5);

        let (vertices, indices_unc) = extract_surface_mesh(
            entry.cpu_octree.as_slice(),
            depth,
            voxel_size,
            grid_origin,
            self.brick_pool.as_slice(),
            self.leaf_attr_pool.as_slice(),
            self.leaf_attr_pool.bones_as_slice(),
        );

        if vertices.is_empty() {
            // Asset carved away to nothing — clear mesh state. The
            // upload path drops the GPU buffers on empty input.
            if let Some(entry) = self.asset_cache.get_mut(handle) {
                entry.mesh_vertices.clear();
                entry.mesh_indices.clear();
                entry.meshlet_clusters.clear();
                entry.mesh_lod0_index_count = 0;
            }
            return;
        }

        // LOD_LEVELS=1: pure LOD-0 clustering, no multi-level
        // simplification. The Karis admit rule treats every cluster
        // as "can't go coarser" (parent_group_error = ∞), so the
        // mesh raster pass always picks LOD-0 — full detail, no
        // pop-in.
        let dag = build_cluster_dag_with_levels(&vertices, &indices_unc, 1);
        let mesh_lod0_index_count = dag.lod0_index_range.1 - dag.lod0_index_range.0;

        let Some(entry) = self.asset_cache.get_mut(handle) else { return; };
        entry.mesh_vertices = vertices;
        entry.mesh_indices = dag.indices;
        entry.meshlet_clusters = dag.clusters;
        entry.mesh_lod0_index_count = mesh_lod0_index_count;

        eprintln!(
            "[sculpt] mesh re-extract: handle={:?} verts={} indices={} clusters={} ({:.2}ms)",
            handle,
            entry.mesh_vertices.len(),
            entry.mesh_indices.len(),
            entry.meshlet_clusters.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rkp_scene_manager::types::AssetEntry;
    use rkp_core::mesh_cluster::{MeshletCluster, PARENT_GROUP_ERROR_ROOT};
    use rkp_core::sparse_octree::SparseOctree;
    use rkp_core::{Aabb, OctreeHandle};

    fn cluster(
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        lod_level: u32,
    ) -> MeshletCluster {
        MeshletCluster {
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            index_offset: 0,
            index_count: 0,
            lod_level,
            flags: 0,
            cluster_error: 0.0,
            parent_group_error: PARENT_GROUP_ERROR_ROOT,
            _pad3: [0; 3],
        }
    }

    /// Build a minimal AssetEntry with caller-supplied clusters and
    /// asset bounds, sized so `grid_origin == aabb.min` and the cluster
    /// AABB coords map 1:1 to grid coords at `base_voxel_size = 1.0`.
    fn make_entry(
        clusters: Vec<MeshletCluster>,
        depth: u8,
    ) -> AssetEntry {
        let base_vs = 1.0_f32;
        let extent = (1u32 << depth) as f32 * base_vs;
        // aabb_center - extent/2 must equal Vec3::ZERO so grid coords
        // are read directly from the cluster's float AABB.
        let aabb = Aabb {
            min: Vec3::ZERO,
            max: Vec3::splat(extent),
        };
        AssetEntry {
            path: std::path::PathBuf::from("test:in-memory"),
            refcount: 1,
            spatial_handle: OctreeHandle {
                root_offset: 0,
                len: 0,
                depth,
                base_voxel_size: base_vs,
            },
            voxel_size: base_vs,
            aabb,
            voxel_count: 0,
            leaf_attr_slot_start: 0,
            leaf_attr_slot_count: 0,
            brick_start: 0,
            brick_count: 0,
            skinning: None,
            splats: Vec::new(),
            mesh_vertices: Vec::new(),
            mesh_indices: Vec::new(),
            mesh_lod0_index_count: 0,
            meshlet_clusters: clusters,
            cpu_octree: SparseOctree::new(depth, base_vs),
        }
    }

    #[test]
    fn r3b_brush_overlap_returns_only_intersecting_lod0_clusters() {
        // Three LOD-0 clusters: A near origin, B in the middle, C far.
        // One LOD-1 cluster D that *would* overlap if it weren't filtered.
        let clusters = vec![
            cluster([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], 0),   // A — id 0
            cluster([10.0, 10.0, 10.0], [14.0, 14.0, 14.0], 0), // B — id 1
            cluster([30.0, 30.0, 30.0], [40.0, 40.0, 40.0], 0), // C — id 2
            cluster([10.0, 10.0, 10.0], [14.0, 14.0, 14.0], 1), // D — id 3 (LOD-1)
        ];
        let mut sm = RkpSceneManager::new(16);
        let handle = sm.asset_cache.insert(make_entry(clusters, 8));

        // Brush centered on B's volume → only B is dirty. D matches the
        // same AABB but is LOD-1, so it must be filtered out.
        let lo = IVec3::splat(11);
        let hi = IVec3::splat(13);
        let dirty = sm.clusters_in_brush_grid_aabb(handle, lo, hi);
        assert_eq!(dirty, vec![1], "only LOD-0 cluster B should be dirty");

        // Brush straddling A and B → both LOD-0 hit. D still filtered.
        let lo = IVec3::new(3, 3, 3);
        let hi = IVec3::new(12, 12, 12);
        let dirty = sm.clusters_in_brush_grid_aabb(handle, lo, hi);
        assert_eq!(dirty, vec![0, 1]);

        // Brush in empty space → no clusters.
        let lo = IVec3::splat(50);
        let hi = IVec3::splat(60);
        assert!(sm.clusters_in_brush_grid_aabb(handle, lo, hi).is_empty());
    }

    #[test]
    fn r3b_empty_brush_aabb_returns_empty() {
        let clusters = vec![cluster([0.0, 0.0, 0.0], [4.0, 4.0, 4.0], 0)];
        let mut sm = RkpSceneManager::new(16);
        let handle = sm.asset_cache.insert(make_entry(clusters, 8));
        // hi <= lo on any axis → empty range, return empty regardless.
        assert!(sm
            .clusters_in_brush_grid_aabb(handle, IVec3::splat(5), IVec3::splat(5))
            .is_empty());
        assert!(sm
            .clusters_in_brush_grid_aabb(handle, IVec3::new(0, 5, 0), IVec3::new(10, 5, 10))
            .is_empty());
    }

    #[test]
    fn r3b_unknown_handle_returns_empty() {
        let sm = RkpSceneManager::new(16);
        // No assets inserted — any handle is bogus.
        let bogus = AssetHandle::from_raw(99);
        assert!(sm
            .clusters_in_brush_grid_aabb(bogus, IVec3::ZERO, IVec3::splat(10))
            .is_empty());
    }

    #[test]
    fn r3b_empty_cluster_table_returns_empty() {
        let mut sm = RkpSceneManager::new(16);
        let handle = sm.asset_cache.insert(make_entry(vec![], 8));
        assert!(sm
            .clusters_in_brush_grid_aabb(handle, IVec3::ZERO, IVec3::splat(10))
            .is_empty());
    }

    #[test]
    fn r3b_brush_at_cluster_edge_inclusive_pad_overlap() {
        // Cluster at float AABB [4.0, 5.0] in each axis → with 1-cell
        // pad, grid AABB is [3..6] inclusive. A brush at exactly cell 6
        // (half-open [6, 7)) overlaps because cluster_max = 6 inclusive.
        // A brush at cell 7 (half-open [7, 8)) does NOT overlap.
        let clusters = vec![cluster([4.0, 4.0, 4.0], [5.0, 5.0, 5.0], 0)];
        let mut sm = RkpSceneManager::new(16);
        let handle = sm.asset_cache.insert(make_entry(clusters, 8));

        assert_eq!(
            sm.clusters_in_brush_grid_aabb(handle, IVec3::splat(6), IVec3::splat(7)),
            vec![0],
            "brush at cluster_max should overlap (inclusive bound)"
        );
        assert!(
            sm.clusters_in_brush_grid_aabb(handle, IVec3::splat(7), IVec3::splat(8))
                .is_empty(),
            "brush past cluster_max should miss"
        );
    }
}
