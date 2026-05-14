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
use rkp_core::mesh_cluster::{MeshletCluster, PARENT_GROUP_ERROR_ROOT};
use rkp_core::mesh_extract::{
    collect_cell_map_in_region, extract_mesh_region_from_cells, extract_surface_mesh,
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

/// Float-AABB intersection test in object-local coordinates.
fn local_aabb_intersects(
    a_min: [f32; 3], a_max: [f32; 3],
    b_min: Vec3, b_max: Vec3,
) -> bool {
    a_min[0] <= b_max.x && a_max[0] >= b_min.x
        && a_min[1] <= b_max.y && a_max[1] >= b_min.y
        && a_min[2] <= b_max.z && a_max[2] >= b_min.z
}

/// Mark meshlet clusters with `CLUSTER_FLAG_LOD_DIRTY` using AABB
/// localization on LOD>0 ancestors plus DAG descent to force-admit
/// LOD-0 leaves whose Karis-admittable ancestor was dropped.
///
/// Algorithm:
/// 1. **Seeds** — brush-touched LOD-0 clusters are force-admitted.
/// 2. **AABB-dirty LOD>0** — any LOD>0 cluster whose object-local
///    AABB intersects the brush AABB is marked dirty (dropped at
///    render time). These are the would-be Karis admitters whose
///    territory now contains stale geometry.
/// 3. **DAG descent** — for each dirty LOD>0 cluster, walk its DAG
///    descendants via `group_below → consumed[]` recursively. Mark
///    every reached cluster dirty:
///      - **LOD-0 descendants** → force-admit (correct, replaces the
///        dropped ancestor's would-be geometry).
///      - **LOD>0 descendants** → also drop, because at the camera
///        distance where the original ancestor was Karis-picked, no
///        intermediate non-dirty descendant could admit either (their
///        `parent_err_proj` falls below threshold once the original
///        ancestor is the Karis pick). Marking them explicitly is
///        defensive and idempotent.
///
/// Why descent is required (and the earlier coverage check was wrong):
/// when Karis would have admitted ancestor `A` at the current camera
/// and `A` is dropped, no DAG descendant admits via Karis: the
/// next-finer level says `parent_err_proj < threshold → parent will
/// admit` (it would have been `A`, now dropped), and `A` itself says
/// `drop_dirty_ancestor`. The chain unrolls down to LOD-0; only an
/// explicit force-admit there saves us from a void. So we must
/// force-admit every LOD-0 descendant of every dropped ancestor.
///
/// Locality on connected meshes: step 2 only marks LOD>0 clusters
/// whose AABB *intersects* the brush AABB. A small brush touches a
/// handful of clusters at each LOD; each cluster's DAG descendants
/// are bounded by its territory (~`4^lod_level` leaves). So the
/// total marked set scales with brush footprint, not asset size.
///
/// Falls back to asset-wide marking when the entry has no DAG
/// topology (`dag_groups.is_empty()`), preserving v5-load behavior.
///
/// Returns the number of clusters newly flagged dirty.
fn mark_lod_dirty_chains(
    entry: &mut super::types::AssetEntry,
    seeds: &[u32],
    brush_aabb_min: Vec3,
    brush_aabb_max: Vec3,
) -> usize {
    use rkp_core::mesh_cluster::{CLUSTER_FLAG_LOD_DIRTY, DAG_GROUP_NONE};
    let cluster_count = entry.meshlet_clusters.len();

    if entry.dag_groups.is_empty() {
        // Pre-v6 asset — no DAG topology baked. Asset-wide marking
        // preserves current behavior. Re-bake via
        // `rkp-convert --rebuild-mesh` to migrate to v6.
        for c in entry.meshlet_clusters.iter_mut() {
            c.flags |= CLUSTER_FLAG_LOD_DIRTY;
        }
        return cluster_count;
    }

    let mut dirty = vec![false; cluster_count];

    // Step 1: seed brush-touched LOD-0 clusters.
    for &cidx in seeds {
        if (cidx as usize) < cluster_count {
            dirty[cidx as usize] = true;
        }
    }

    // Step 2: LOD>0 clusters whose AABB intersects the brush AABB
    // become dirty. Push them onto a descent stack for step 3.
    let mut descent_stack: Vec<u32> = Vec::new();
    for (idx, c) in entry.meshlet_clusters.iter().enumerate() {
        if c.lod_level == 0 || dirty[idx] {
            continue;
        }
        if local_aabb_intersects(c.aabb_min, c.aabb_max, brush_aabb_min, brush_aabb_max) {
            dirty[idx] = true;
            descent_stack.push(idx as u32);
        }
    }

    // Step 3: descend each dirty LOD>0 via group_below.consumed[]
    // recursively. Mark every reached cluster dirty; only LOD>0
    // descendants need to be re-pushed for further descent (LOD-0 is
    // a leaf in the DAG). The shared `dirty` array de-dupes work
    // when multiple ancestors share descendants.
    while let Some(cidx) = descent_stack.pop() {
        let c = &entry.meshlet_clusters[cidx as usize];
        if c.group_below_idx == DAG_GROUP_NONE {
            continue;
        }
        let g = entry.dag_groups[c.group_below_idx as usize];
        let lo = g.consumed_first as usize;
        let hi = lo + g.consumed_count as usize;
        for &child in &entry.dag_consumed[lo..hi] {
            let n = child as usize;
            if n >= cluster_count || dirty[n] {
                continue;
            }
            dirty[n] = true;
            if entry.meshlet_clusters[n].lod_level > 0 {
                descent_stack.push(child);
            }
        }
    }

    // Step 4: apply dirty flags to clusters.
    let mut marked = 0usize;
    for (idx, c) in entry.meshlet_clusters.iter_mut().enumerate() {
        if dirty[idx] {
            c.flags |= CLUSTER_FLAG_LOD_DIRTY;
            marked += 1;
        }
    }
    marked
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
        let _sculpt_t0 = std::time::Instant::now();
        // D0 per-phase timings. Each `_p*` records the wall-clock cost
        // of one internal step of `apply_sculpt_brush`. Emitted in the
        // existing `[sculpt] stamp …` log line at the end so users
        // (and the perf-debt drain plan) can see where the budget
        // goes without enabling a separate flag. Cost: ~9 Instant::now
        // calls per stamp — sub-microsecond noise.
        let mut _p_resolve_ms = 0.0;
        let mut _p_compute_edits_ms = 0.0;
        let mut _p_resolve_removes_ms = 0.0;
        let mut _p_apply_delta_ms = 0.0;
        let mut _p_octree_sync_ms = 0.0;
        let mut _p_leaf_attr_ms = 0.0;
        let mut _p_rebuild_clusters_ms = 0.0;
        let _ph_t0 = std::time::Instant::now();

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
        _p_resolve_ms = _ph_t0.elapsed().as_secs_f64() * 1000.0;
        let _ph_t1 = std::time::Instant::now();

        // ── 2. Compute edit list against current octree + brick pool. ─
        let delta = {
            let entry = self.asset_cache.get(handle)?;
            compute_brush_edits(&entry.cpu_octree, &self.brick_pool, op)
        };
        _p_compute_edits_ms = _ph_t1.elapsed().as_secs_f64() * 1000.0;
        if delta.is_empty() {
            return None;
        }
        let _ph_t2 = std::time::Instant::now();

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
        _p_resolve_removes_ms = _ph_t2.elapsed().as_secs_f64() * 1000.0;
        let _ph_t3 = std::time::Instant::now();

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
        _p_apply_delta_ms = _ph_t3.elapsed().as_secs_f64() * 1000.0;
        let _ph_t4 = std::time::Instant::now();

        // ── 4b. Sync cpu_octree mutations into OctreeGpu's packed buffer.
        //
        // Closes the latent CPU↔GPU sync bug: `apply_delta` mutated
        // `entry.cpu_octree.nodes`, but without this step those changes
        // never reach the packed buffer that `upload_geometry` ships.
        //
        // Growth handling: every drag stamp typically subdivides a few
        // empty/interior terminators (+200-400 nodes on splat5). D8's
        // slot-slack lets us extend `handle.len` in place without a
        // re-allocation — the mutation log already records the
        // appended slots, so applying it after `try_extend_in_slack`
        // populates the slack region. We only fall back to dealloc +
        // allocate_with_slack when the slack itself is exhausted; the
        // re-allocation then reserves more headroom for future stamps.
        {
            let entry = self.asset_cache.get_mut(handle)?;
            let spatial = entry.spatial_handle;
            let new_node_count = entry.cpu_octree.node_count() as u32;
            if applied.octree_log.grew(new_node_count) {
                let extended = self.octree.try_extend_in_slack(&spatial, new_node_count);
                match extended {
                    Some(new_handle) => {
                        let entry = self.asset_cache.get_mut(handle)?;
                        entry.spatial_handle = new_handle;
                        self.octree.apply_mutation_log(&new_handle, &applied.octree_log);
                    }
                    None => {
                        eprintln!(
                            "[sculpt] octree slack exhausted ({} → {} nodes) — re-allocating slot",
                            applied.octree_log.initial_node_count, new_node_count,
                        );
                        self.octree.deallocate(spatial);
                        let entry = self.asset_cache.get_mut(handle)?;
                        let new_handle = self.octree.allocate_with_slack(&entry.cpu_octree, 1.5);
                        let entry = self.asset_cache.get_mut(handle)?;
                        entry.spatial_handle = new_handle;
                    }
                }
            } else if !applied.octree_log.is_empty() {
                self.octree.apply_mutation_log(&spatial, &applied.octree_log);
            }
        }
        _p_octree_sync_ms = _ph_t4.elapsed().as_secs_f64() * 1000.0;
        let _ph_t5 = std::time::Instant::now();

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
        _p_leaf_attr_ms = _ph_t5.elapsed().as_secs_f64() * 1000.0;
        let _ph_t6 = std::time::Instant::now();

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
        _p_rebuild_clusters_ms = _ph_t6.elapsed().as_secs_f64() * 1000.0;

        // Bump the geometry epoch so the renderer re-uploads the
        // mutated octree / brick / leaf_attr buffers AND the new
        // mesh data on the next pre-frame pass.
        self.bump_geometry_epoch();

        // D0 — per-phase breakdown appended to the existing log line.
        // The seven phases cover every step from grid-coord resolve
        // through the per-cluster mesh patch; their sum should match
        // `total` to within a few µs (Instant::now overhead). The
        // dominant phase identifies the next drain-optimization target.
        eprintln!(
            "[sculpt] stamp handle={:?} mode={:?} edits={} removed={} \
             applied(adds={} freed={} interior={}) (depth={}, base_vs={:.5}) total={:.2}ms \
             [phases: resolve={:.2} edits={:.2} resolve_rm={:.2} apply_delta={:.2} \
             octree_sync={:.2} leaf_attr={:.2} rebuild_clusters={:.2}]",
            handle, mode, delta.len(), removed.len(),
            applied.allocated_slots.len(), applied.freed_slots.len(),
            delta.count_interior(), depth, base_vs,
            _sculpt_t0.elapsed().as_secs_f64() * 1000.0,
            _p_resolve_ms, _p_compute_edits_ms, _p_resolve_removes_ms,
            _p_apply_delta_ms, _p_octree_sync_ms, _p_leaf_attr_ms,
            _p_rebuild_clusters_ms,
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

    /// Per-stamp mesh patch — **filter dirty clusters + extract brush
    /// region once** (Phase B R4c V2).
    ///
    /// The earlier "re-extract each dirty cluster's full grid AABB"
    /// path was correct but produced ~3 000 tris per cluster on
    /// splat5-style ground-plane assets (cluster AABBs are far bigger
    /// than the brush sphere). With 80+ dirty clusters per stamp the
    /// mesh grew ~25 ×/stamp unboundedly, and the per-cluster extract
    /// loop ran 5 s/stamp.
    ///
    /// V2 flips the model: each dirty cluster's existing triangles get
    /// *filtered* (any tri whose vertex sits inside the brush sphere
    /// is dropped, the rest are kept); the brush region's new surface
    /// is then re-extracted **once** and appended as a fresh LOD-0
    /// cluster. Per-stamp growth is bounded by the brush volume, not
    /// the dirty-cluster footprint.
    ///
    /// **Boundary seam.** Pre-brush SN-cube vertex positions (centroid
    /// of edge crossings against the OLD occupancy field) differ
    /// sub-voxel from post-brush positions at the same cube. The
    /// filter/keep boundary on the *dirty cluster* side uses pre-brush
    /// verts; the new brush-region patch uses post-brush verts. A
    /// thin band at the brush-sphere boundary may show a tiny seam.
    /// Acceptable for V1; proper stitching is an R5 follow-up.
    ///
    /// Returns `true` when the V2 path fired; `false` when the dirty
    /// set is empty or the asset has no mesh — caller falls back to
    /// the full re-extract path.
    fn rebuild_dirty_clusters(&mut self, handle: AssetHandle, op: &BrushOp) -> bool {
        let t0 = std::time::Instant::now();
        // D0 per-phase timings inside the cluster-patch path. Mirrors
        // the apply_sculpt_brush phase timings; written into the
        // `[sculpt] V2 patch` log line at the end.
        let mut _p_setup_ms = 0.0;
        let mut _p_dirty_query_ms = 0.0;
        let mut _p_filter_ms = 0.0;
        let mut _p_extract_ms = 0.0;
        let mut _p_append_patch_ms = 0.0;
        let mut _p_cc_walk_ms = 0.0;
        let _ph_t0 = t0;

        let (depth, base_vs, grid_origin, brush_lo, brush_hi,
             brush_center_local, brush_radius_local, brush_radius_sq) = {
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

            // Brush sphere in **object-local** space (same coords as
            // `MeshVertex::local_pos`). `op.center` is in finest-grid
            // units; convert to object-local via grid_origin + grid * vs.
            let brush_center_local = grid_origin + op.center * base_vs;
            let brush_radius_local = op.radius * base_vs;
            let brush_radius_sq = brush_radius_local * brush_radius_local;

            (depth, base_vs, grid_origin, brush_lo, brush_hi,
             brush_center_local, brush_radius_local, brush_radius_sq)
        };

        if brush_lo.x >= brush_hi.x || brush_lo.y >= brush_hi.y || brush_lo.z >= brush_hi.z {
            return false;
        }
        _p_setup_ms = _ph_t0.elapsed().as_secs_f64() * 1000.0;
        let _ph_t1 = std::time::Instant::now();

        let dirty = self.clusters_in_brush_grid_aabb(handle, brush_lo, brush_hi);
        _p_dirty_query_ms = _ph_t1.elapsed().as_secs_f64() * 1000.0;
        if dirty.is_empty() {
            return false;
        }
        let _ph_t2 = std::time::Instant::now();

        // ── Phase 1: filter each dirty cluster's tris ─────────────────
        //
        // Walk each cluster's existing tris; keep only those with ALL
        // three verts strictly outside the brush sphere. Append the
        // kept indices to the tail of `mesh_indices`; redirect the
        // cluster's `index_offset` / `index_count`.
        //
        // Reading `mesh_vertices[idx]` is safe alongside `mesh_indices.
        // extend(...)` because we read in a separate scope per cluster
        // and the extend goes to the TAIL — the original cluster slot
        // (which we read) is not touched.
        //
        // **D1 — cluster-AABB → brush-sphere rejection.** `clusters_in_
        // brush_grid_aabb` returns every LOD-0 cluster whose grid AABB
        // overlaps the brush AABB. A box-vs-box test admits clusters
        // whose AABB corners touch the brush box but whose closest
        // point to the brush *sphere* center is still outside the
        // sphere — every triangle in those clusters would survive the
        // per-tri test below. Pre-D1 the loop ran the per-tri test
        // anyway: 80+ clusters × ~3000 tris × 3 length_squared = ~700k
        // float ops/stamp on splat5 elephant. D1 short-circuits each
        // sphere-outside cluster to a single `extend_from_slice` of
        // its original indices — saves the per-tri loop entirely
        // for those clusters. Sphere-inside clusters still run the
        // per-tri test (some of their tris may be inside the sphere).
        let mut total_kept_tris = 0usize;
        let mut total_dropped_tris = 0usize;
        // D1 telemetry — count clusters that took the sphere-outside
        // fast path. High ratios prove the AABB → sphere filter is
        // doing its job; low ratios mean the brush sphere genuinely
        // touches most candidate clusters and the per-tri test is
        // unavoidable.
        let mut d1_clusters_sphere_outside = 0usize;
        for &cid in &dirty {
            let (start, count, cluster_aabb_min, cluster_aabb_max) = {
                let Some(entry) = self.asset_cache.get(handle) else { return false; };
                let c = &entry.meshlet_clusters[cid as usize];
                (
                    c.index_offset as usize,
                    c.index_count as usize,
                    Vec3::from(c.aabb_min),
                    Vec3::from(c.aabb_max),
                )
            };

            // Sphere-AABB rejection: closest point on the cluster's
            // AABB to the brush center. If it's outside the brush
            // sphere, no tri in this cluster can have a vertex inside.
            let closest = brush_center_local.clamp(cluster_aabb_min, cluster_aabb_max);
            let aabb_dist_sq = (closest - brush_center_local).length_squared();
            let cluster_fully_outside = aabb_dist_sq > brush_radius_sq;
            if cluster_fully_outside {
                d1_clusters_sphere_outside += 1;
            }

            // Snapshot tri-by-tri "keep" decisions into a small vec to
            // dodge borrow-checker fights between immutable reads of
            // mesh_vertices/mesh_indices and the later mutable extend.
            let kept: Vec<u32> = {
                let Some(entry) = self.asset_cache.get(handle) else { return false; };
                let indices = &entry.mesh_indices;
                if cluster_fully_outside {
                    // Every tri is "keep" — just copy the original
                    // index range. Saves the per-tri sphere test.
                    indices[start..start + count].to_vec()
                } else {
                    let verts = &entry.mesh_vertices;
                    let mut out = Vec::with_capacity(count);
                    for tri_start in (start..start + count).step_by(3) {
                        let i0 = indices[tri_start];
                        let i1 = indices[tri_start + 1];
                        let i2 = indices[tri_start + 2];
                        let p0 = Vec3::from(verts[i0 as usize].local_pos);
                        let p1 = Vec3::from(verts[i1 as usize].local_pos);
                        let p2 = Vec3::from(verts[i2 as usize].local_pos);
                        let d0 = (p0 - brush_center_local).length_squared();
                        let d1 = (p1 - brush_center_local).length_squared();
                        let d2 = (p2 - brush_center_local).length_squared();
                        if d0 > brush_radius_sq && d1 > brush_radius_sq && d2 > brush_radius_sq {
                            out.push(i0);
                            out.push(i1);
                            out.push(i2);
                        }
                    }
                    out
                }
            };

            total_kept_tris += kept.len() / 3;
            total_dropped_tris += count / 3 - kept.len() / 3;

            let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
            let new_offset = entry.mesh_indices.len() as u32;
            entry.mesh_indices.extend_from_slice(&kept);
            let cluster = &mut entry.meshlet_clusters[cid as usize];
            cluster.index_offset = new_offset;
            cluster.index_count = kept.len() as u32;
            // AABB stays at the cluster's pre-stamp bounds — the kept
            // tris are a subset of the original, so they fit inside
            // the original AABB. Shrinking is left to R5.
        }

        _p_filter_ms = _ph_t2.elapsed().as_secs_f64() * 1000.0;
        let _ph_t3 = std::time::Instant::now();

        // ── Phase 2: extract brush region once ────────────────────────
        //
        // Use the region-scoped cell-map walker so we don't pay
        // O(asset surface area) per stamp. Pad the map by +3 cells
        // each side so SN-cube vertex builds at the boundary see all
        // 8 corner cells in the map.
        let cells_min = brush_lo - IVec3::splat(3);
        let cells_max = brush_hi + IVec3::splat(3);
        let cells = {
            let Some(entry) = self.asset_cache.get(handle) else { return false; };
            collect_cell_map_in_region(
                entry.cpu_octree.as_slice(),
                depth,
                self.brick_pool.as_slice(),
                cells_min,
                cells_max,
            )
        };

        let (brush_verts, brush_indices) = {
            let Some(entry) = self.asset_cache.get(handle) else { return false; };
            extract_mesh_region_from_cells(
                &cells,
                brush_lo,
                brush_hi,
                entry.cpu_octree.as_slice(),
                depth,
                base_vs,
                grid_origin,
                self.brick_pool.as_slice(),
                self.leaf_attr_pool.as_slice(),
                self.leaf_attr_pool.bones_as_slice(),
            )
        };

        _p_extract_ms = _ph_t3.elapsed().as_secs_f64() * 1000.0;
        let _ph_t4 = std::time::Instant::now();

        // ── Phase 3: append brush region as a new LOD-0 patch cluster ─
        if !brush_verts.is_empty() {
            let mut patch_aabb_min = [f32::INFINITY; 3];
            let mut patch_aabb_max = [f32::NEG_INFINITY; 3];
            for v in &brush_verts {
                for k in 0..3 {
                    if v.local_pos[k] < patch_aabb_min[k] {
                        patch_aabb_min[k] = v.local_pos[k];
                    }
                    if v.local_pos[k] > patch_aabb_max[k] {
                        patch_aabb_max[k] = v.local_pos[k];
                    }
                }
            }

            let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
            let vertex_offset = entry.mesh_vertices.len() as u32;
            let new_index_offset = entry.mesh_indices.len() as u32;
            entry.mesh_vertices.extend_from_slice(&brush_verts);
            entry.mesh_indices
                .extend(brush_indices.iter().map(|&i| i + vertex_offset));

            entry.meshlet_clusters.push(MeshletCluster {
                aabb_min: patch_aabb_min,
                _pad0: 0.0,
                aabb_max: patch_aabb_max,
                index_offset: new_index_offset,
                index_count: brush_indices.len() as u32,
                lod_level: 0,
                // Patch cluster is born already-dirty so it admits
                // unconditionally on the next render (its position
                // wasn't in the original DAG so Karis would otherwise
                // have nothing to project against).
                flags: rkp_core::mesh_cluster::CLUSTER_FLAG_LOD_DIRTY,
                cluster_error: 0.0,
                parent_group_error: PARENT_GROUP_ERROR_ROOT,
                // Patch cluster is appended after the bake-time DAG;
                // it has no group on either side. CC walks from
                // brush-touched LOD-0 clusters don't traverse through
                // it, which is correct — the patch is standalone-dirty
                // and force-admits unconditionally.
                group_above_idx: rkp_core::mesh_cluster::DAG_GROUP_NONE,
                group_below_idx: rkp_core::mesh_cluster::DAG_GROUP_NONE,
                _pad3: 0,
            });
        }

        _p_append_patch_ms = _ph_t4.elapsed().as_secs_f64() * 1000.0;
        let _ph_t5 = std::time::Instant::now();

        // ── Phase 4: per-chain LOD_DIRTY via DAG CC walk ──────────────
        //
        // R4d V2 (Option A from project_sculpt_drag_pick_roundtrip):
        // walk the bipartite cluster ↔ DAG-group graph starting from
        // every brush-touched LOD-0 cluster, and mark every cluster
        // it reaches with `LOD_DIRTY`. The mesh_lod_select shader's
        // admit rule force-admits dirty LOD-0 leaves and drops dirty
        // LOD>0 ancestors — so by marking exactly the chains in the
        // brush footprint, we keep the un-touched portion of the
        // asset on its normal Karis admit path (renders at coarse
        // LOD, ~188 fps) instead of the asset-wide LOD-0 clamp that
        // R4d V1 used (~44 fps on the elephant).
        //
        // Correctness of the walk: the bake-time DAG records, per
        // cluster, `group_above_idx` (the group that consumed it as
        // a child) and `group_below_idx` (the group that produced it
        // as a parent). For each visited cluster we enqueue both
        // groups' full membership — `consumed` (siblings + further
        // descendants via shared parents) and `produced` (parents /
        // ancestors). Iterating to fixpoint gives the connected
        // component containing the brush seeds. Every chain a
        // brush-touched leaf belongs to is fully marked, so the
        // shader's drop-ancestor + force-admit-leaf logic stays
        // consistent within each chain (no cluster-shaped voids,
        // which is what killed the earlier per-brush-AABB attempt
        // in commit c577d85d).
        //
        // Fallback: assets baked before v6 carry an empty
        // `dag_groups` — we can't run the CC walk without topology,
        // so keep the asset-wide loop. Re-bake via
        // `cargo run -p rkp-convert -- --rebuild-mesh <asset>.rkp`
        // to migrate to v6 and unlock the per-chain narrowing.
        // Brush AABB in object-local floats. Inflated by `base_vs` on
        // each side so the AABB intersection check stays consistent
        // with `clusters_in_brush_grid_aabb`'s one-cell padding on
        // the cluster side — a brush exactly at a cluster edge still
        // counts as overlapping.
        let brush_aabb_min = brush_center_local - Vec3::splat(brush_radius_local + base_vs);
        let brush_aabb_max = brush_center_local + Vec3::splat(brush_radius_local + base_vs);
        let walk_visited_count = {
            let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
            mark_lod_dirty_chains(entry, &dirty, brush_aabb_min, brush_aabb_max)
        };
        _p_cc_walk_ms = _ph_t5.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 5: refresh `mesh_lod0_index_count` + dirty flags ────
        // The legacy direct-draw dispatch (debug-only) reads
        // `[0..mesh_lod0_index_count)`. The active indirect-draw path
        // ignores this and walks per-cluster offsets/counts, so the
        // value is informational for the legacy path only. Sum across
        // all LOD-0 cluster entries (including the new patch cluster).
        //
        // Mark mesh + clusters dirty so the next geometry-epoch upload
        // loop picks this asset up and skips all the others.
        let Some(entry) = self.asset_cache.get_mut(handle) else { return false; };
        entry.mesh_lod0_index_count = entry
            .meshlet_clusters
            .iter()
            .filter(|c| c.lod_level == 0)
            .map(|c| c.index_count)
            .sum();
        entry.mesh_dirty = true;
        entry.clusters_dirty = true;

        let total_clusters = entry.meshlet_clusters.len();
        // D0 — per-phase breakdown of rebuild_dirty_clusters. The
        // dominant sub-phase identifies which step of the cluster
        // patch path to target next (filter vs extract vs CC walk).
        eprintln!(
            "[sculpt] V2 patch: handle={:?} dirty={} (sphere_outside={}) kept_tris={} dropped_tris={} \
             brush_patch verts={} tris={} total flat verts={} indices={} \
             lod_dirty={}/{} ({:.0}%) ({:.2}ms) \
             [phases: setup={:.2} dirty_q={:.2} filter={:.2} extract={:.2} append={:.2} cc_walk={:.2}]",
            handle,
            dirty.len(),
            d1_clusters_sphere_outside,
            total_kept_tris,
            total_dropped_tris,
            brush_verts.len(),
            brush_indices.len() / 3,
            entry.mesh_vertices.len(),
            entry.mesh_indices.len(),
            walk_visited_count,
            total_clusters,
            if total_clusters > 0 {
                100.0 * walk_visited_count as f64 / total_clusters as f64
            } else {
                0.0
            },
            t0.elapsed().as_secs_f64() * 1000.0,
            _p_setup_ms, _p_dirty_query_ms, _p_filter_ms,
            _p_extract_ms, _p_append_patch_ms, _p_cc_walk_ms,
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
                entry.mesh_dirty = true;
                entry.clusters_dirty = true;
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
        entry.mesh_dirty = true;
        entry.clusters_dirty = true;

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
    use rkp_core::mesh_cluster::{DAG_GROUP_NONE, MeshletCluster, PARENT_GROUP_ERROR_ROOT};
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
            group_above_idx: DAG_GROUP_NONE,
            group_below_idx: DAG_GROUP_NONE,
            _pad3: 0,
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
            dag_groups: Vec::new(),
            dag_consumed: Vec::new(),
            dag_produced: Vec::new(),
            cpu_octree: SparseOctree::new(depth, base_vs),
            mesh_dirty: false,
            splats_dirty: false,
            clusters_dirty: false,
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

    /// Hand-build a small two-chain DAG and stick it onto an
    /// `AssetEntry`, so [`mark_lod_dirty_chains`] tests can run
    /// without invoking the full bake.
    ///
    /// Layout — two independent chains, each LOD-0 + LOD-1:
    ///   chain X: c0_a, c0_b (LOD-0) →  g0  → c1_x (LOD-1)
    ///   chain Y: c0_c, c0_d (LOD-0) →  g1  → c1_y (LOD-1)
    /// IDs in `meshlet_clusters`: 0 = c0_a, 1 = c0_b, 2 = c0_c,
    /// 3 = c0_d, 4 = c1_x, 5 = c1_y.
    fn two_chain_entry() -> AssetEntry {
        use rkp_core::mesh_lod::DagGroup;
        let clusters = vec![
            cluster([0.0; 3], [1.0; 3], 0),  // 0 c0_a (chain X)
            cluster([1.0; 3], [2.0; 3], 0),  // 1 c0_b (chain X)
            cluster([5.0; 3], [6.0; 3], 0),  // 2 c0_c (chain Y)
            cluster([6.0; 3], [7.0; 3], 0),  // 3 c0_d (chain Y)
            cluster([0.0; 3], [2.0; 3], 1),  // 4 c1_x (chain X)
            cluster([5.0; 3], [7.0; 3], 1),  // 5 c1_y (chain Y)
        ];
        let mut entry = make_entry(clusters, 8);
        entry.dag_groups = vec![
            DagGroup { consumed_first: 0, consumed_count: 2, produced_first: 0, produced_count: 1 },
            DagGroup { consumed_first: 2, consumed_count: 2, produced_first: 1, produced_count: 1 },
        ];
        entry.dag_consumed = vec![0, 1, 2, 3];
        entry.dag_produced = vec![4, 5];
        // Set per-cluster topology pointers consistent with groups.
        entry.meshlet_clusters[0].group_above_idx = 0;
        entry.meshlet_clusters[1].group_above_idx = 0;
        entry.meshlet_clusters[4].group_below_idx = 0;
        entry.meshlet_clusters[2].group_above_idx = 1;
        entry.meshlet_clusters[3].group_above_idx = 1;
        entry.meshlet_clusters[5].group_below_idx = 1;
        entry
    }

    fn dirty_set(entry: &AssetEntry) -> Vec<u32> {
        use rkp_core::mesh_cluster::CLUSTER_FLAG_LOD_DIRTY;
        entry
            .meshlet_clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.flags & CLUSTER_FLAG_LOD_DIRTY != 0)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Brush AABB covering only chain X (clusters with AABB in [0, 2]).
    fn brush_chain_x() -> (Vec3, Vec3) {
        (Vec3::splat(0.0), Vec3::splat(1.5))
    }

    /// Brush AABB covering both chains (spans [0, 7]).
    fn brush_both_chains() -> (Vec3, Vec3) {
        (Vec3::splat(0.0), Vec3::splat(7.0))
    }

    /// Brush AABB far away from every cluster (no LOD>0 will match).
    fn brush_far_away() -> (Vec3, Vec3) {
        (Vec3::splat(100.0), Vec3::splat(101.0))
    }

    #[test]
    fn mark_lod_dirty_chains_narrows_to_seed_chain() {
        // Brush covers chain X (cluster AABBs [0, 2]). Seed c0_a.
        // Expected dirty: {c0_a (seed), c0_b (ancestor c1_x dirty →
        // not covered), c1_x (AABB intersects brush)}. Chain Y must
        // stay clean — its only ancestor c1_y has AABB [5,7] which
        // doesn't intersect the brush, so c0_c / c0_d are covered.
        let mut entry = two_chain_entry();
        let (bmin, bmax) = brush_chain_x();
        let marked = mark_lod_dirty_chains(&mut entry, &[0], bmin, bmax);
        assert_eq!(marked, 3);
        assert_eq!(dirty_set(&entry), vec![0, 1, 4]);
    }

    #[test]
    fn mark_lod_dirty_chains_seeds_from_both_chains_mark_both() {
        // Brush spans both chains [0, 7]. Seeds c0_a and c0_c. Both
        // LOD-1 ancestors (c1_x, c1_y) intersect brush → dirty. Then
        // c0_b, c0_d's only ancestor is dirty → force-admit. All 6.
        let mut entry = two_chain_entry();
        let (bmin, bmax) = brush_both_chains();
        let marked = mark_lod_dirty_chains(&mut entry, &[0, 2], bmin, bmax);
        assert_eq!(marked, 6);
        assert_eq!(dirty_set(&entry), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn mark_lod_dirty_chains_fallback_when_dag_empty() {
        // V5 asset with no baked DAG topology — fallback marks
        // every cluster (preserving the R4d V1 behavior).
        let mut entry = two_chain_entry();
        entry.dag_groups.clear();
        entry.dag_consumed.clear();
        entry.dag_produced.clear();
        for c in entry.meshlet_clusters.iter_mut() {
            c.group_above_idx = rkp_core::mesh_cluster::DAG_GROUP_NONE;
            c.group_below_idx = rkp_core::mesh_cluster::DAG_GROUP_NONE;
        }
        let (bmin, bmax) = brush_chain_x();
        let marked = mark_lod_dirty_chains(&mut entry, &[0], bmin, bmax);
        assert_eq!(marked, 6);
        assert_eq!(dirty_set(&entry), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn mark_lod_dirty_chains_brush_misses_everything() {
        // Brush far from any cluster + seeds empty. No clusters
        // should be marked; AABB-step finds nothing, coverage check
        // finds non-dirty ancestors for every LOD-0 leaf.
        let mut entry = two_chain_entry();
        let (bmin, bmax) = brush_far_away();
        let marked = mark_lod_dirty_chains(&mut entry, &[], bmin, bmax);
        assert_eq!(marked, 0);
        assert!(dirty_set(&entry).is_empty());
    }

    #[test]
    fn mark_lod_dirty_chains_handles_out_of_range_seeds() {
        // A stale seed past the cluster table must be silently
        // skipped — patch-cluster appends shift cluster ids and we
        // don't want to panic.
        let mut entry = two_chain_entry();
        let (bmin, bmax) = brush_chain_x();
        let marked = mark_lod_dirty_chains(&mut entry, &[0, 999], bmin, bmax);
        assert_eq!(marked, 3);
        assert_eq!(dirty_set(&entry), vec![0, 1, 4]);
    }

    /// Connected-DAG fixture — the worst case for the previous
    /// symmetric CC walk. Six LOD-0 + 2 LOD-1 + 1 LOD-2 root that
    /// covers the whole asset. Chain X: c0_a, c0_b → c1_x.
    /// Chain Y: c0_c, c0_d → c1_y. Root: c1_x, c1_y → c2_z.
    /// Cluster IDs: 0=c0_a, 1=c0_b, 2=c0_c, 3=c0_d, 4=c1_x, 5=c1_y, 6=c2_z.
    fn three_lod_connected_entry() -> AssetEntry {
        use rkp_core::mesh_lod::DagGroup;
        let clusters = vec![
            cluster([0.0; 3], [1.0; 3], 0),  // 0 c0_a
            cluster([1.0; 3], [2.0; 3], 0),  // 1 c0_b
            cluster([2.0; 3], [3.0; 3], 0),  // 2 c0_c
            cluster([3.0; 3], [4.0; 3], 0),  // 3 c0_d
            cluster([0.0; 3], [2.0; 3], 1),  // 4 c1_x
            cluster([2.0; 3], [4.0; 3], 1),  // 5 c1_y
            cluster([0.0; 3], [4.0; 3], 2),  // 6 c2_z (root, full asset)
        ];
        let mut entry = make_entry(clusters, 8);
        entry.dag_groups = vec![
            // g0: consumed [c0_a, c0_b] produced [c1_x]
            DagGroup { consumed_first: 0, consumed_count: 2, produced_first: 0, produced_count: 1 },
            // g1: consumed [c0_c, c0_d] produced [c1_y]
            DagGroup { consumed_first: 2, consumed_count: 2, produced_first: 1, produced_count: 1 },
            // g2: consumed [c1_x, c1_y] produced [c2_z]
            DagGroup { consumed_first: 4, consumed_count: 2, produced_first: 2, produced_count: 1 },
        ];
        entry.dag_consumed = vec![0, 1, 2, 3, 4, 5];
        entry.dag_produced = vec![4, 5, 6];
        // Pointer wiring.
        entry.meshlet_clusters[0].group_above_idx = 0;
        entry.meshlet_clusters[1].group_above_idx = 0;
        entry.meshlet_clusters[4].group_below_idx = 0;
        entry.meshlet_clusters[2].group_above_idx = 1;
        entry.meshlet_clusters[3].group_above_idx = 1;
        entry.meshlet_clusters[5].group_below_idx = 1;
        entry.meshlet_clusters[4].group_above_idx = 2;
        entry.meshlet_clusters[5].group_above_idx = 2;
        entry.meshlet_clusters[6].group_below_idx = 2;
        entry
    }

    #[test]
    fn mark_lod_dirty_chains_single_root_descent_marks_full_asset() {
        // A connected DAG with a SINGLE root covering the whole asset
        // is the worst case for descent: any brush intersecting the
        // root triggers descent down to every LOD-0 leaf. This is
        // correct — when the root is the Karis pick at some camera
        // distance and we drop it, every chain needs force-admit at
        // LOD-0 (no Karis-admittable intermediate level survives).
        //
        // Real assets (the procedural box has 9.6k LOD-3 roots, each
        // localized) don't hit this case — they have many roots and
        // the brush only intersects a few of them, so the descent
        // stays local. See `multi_root_dag_narrows_to_brush_region`
        // for the realistic narrowing test.
        let mut entry = three_lod_connected_entry();
        let bmin = Vec3::splat(0.0);
        let bmax = Vec3::splat(1.5);
        let marked = mark_lod_dirty_chains(&mut entry, &[0], bmin, bmax);
        assert_eq!(marked, 7, "single-root DAG → descent marks everything");
        assert_eq!(dirty_set(&entry), vec![0, 1, 2, 3, 4, 5, 6]);
    }

    /// Multi-root DAG fixture matching the procedural-box structure:
    /// two SEPARATE LOD-1 → LOD-2 chains with no shared root. Brush
    /// in one chain's region must not reach the other.
    ///
    /// Chain X: c0_a [0,1], c0_b [1,2] → c1_x [0,2] → c2_x [0,2]
    /// Chain Y: c0_c [5,6], c0_d [6,7] → c1_y [5,7] → c2_y [5,7]
    /// IDs: 0=c0_a, 1=c0_b, 2=c0_c, 3=c0_d, 4=c1_x, 5=c1_y, 6=c2_x, 7=c2_y.
    fn multi_root_entry() -> AssetEntry {
        use rkp_core::mesh_lod::DagGroup;
        let clusters = vec![
            cluster([0.0; 3], [1.0; 3], 0),  // 0 c0_a
            cluster([1.0; 3], [2.0; 3], 0),  // 1 c0_b
            cluster([5.0; 3], [6.0; 3], 0),  // 2 c0_c
            cluster([6.0; 3], [7.0; 3], 0),  // 3 c0_d
            cluster([0.0; 3], [2.0; 3], 1),  // 4 c1_x
            cluster([5.0; 3], [7.0; 3], 1),  // 5 c1_y
            cluster([0.0; 3], [2.0; 3], 2),  // 6 c2_x (root X)
            cluster([5.0; 3], [7.0; 3], 2),  // 7 c2_y (root Y)
        ];
        let mut entry = make_entry(clusters, 8);
        entry.dag_groups = vec![
            // g0: consumed [c0_a, c0_b] produced [c1_x]
            DagGroup { consumed_first: 0, consumed_count: 2, produced_first: 0, produced_count: 1 },
            // g1: consumed [c0_c, c0_d] produced [c1_y]
            DagGroup { consumed_first: 2, consumed_count: 2, produced_first: 1, produced_count: 1 },
            // g2: consumed [c1_x] produced [c2_x]
            DagGroup { consumed_first: 4, consumed_count: 1, produced_first: 2, produced_count: 1 },
            // g3: consumed [c1_y] produced [c2_y]
            DagGroup { consumed_first: 5, consumed_count: 1, produced_first: 3, produced_count: 1 },
        ];
        entry.dag_consumed = vec![0, 1, 2, 3, 4, 5];
        entry.dag_produced = vec![4, 5, 6, 7];
        entry.meshlet_clusters[0].group_above_idx = 0;
        entry.meshlet_clusters[1].group_above_idx = 0;
        entry.meshlet_clusters[4].group_below_idx = 0;
        entry.meshlet_clusters[2].group_above_idx = 1;
        entry.meshlet_clusters[3].group_above_idx = 1;
        entry.meshlet_clusters[5].group_below_idx = 1;
        entry.meshlet_clusters[4].group_above_idx = 2;
        entry.meshlet_clusters[6].group_below_idx = 2;
        entry.meshlet_clusters[5].group_above_idx = 3;
        entry.meshlet_clusters[7].group_below_idx = 3;
        entry
    }

    #[test]
    fn mark_lod_dirty_chains_multi_root_narrows_to_brush_region() {
        // **The load-bearing test for connected-mesh assets.** Two
        // separate DAG-root chains (no shared root, like the real
        // procedural box with 9.6k LOD-3 roots). Brush touches
        // chain X only; chain Y must stay clean.
        //
        // Expected dirty: {c0_a (seed), c0_b (descended from c1_x),
        // c1_x (AABB), c2_x (AABB, descent doesn't cross into chain Y)}.
        let mut entry = multi_root_entry();
        let bmin = Vec3::splat(0.0);
        let bmax = Vec3::splat(1.5);
        let marked = mark_lod_dirty_chains(&mut entry, &[0], bmin, bmax);
        assert_eq!(
            marked, 4,
            "narrow to chain X: c0_a + c0_b + c1_x + c2_x"
        );
        assert_eq!(dirty_set(&entry), vec![0, 1, 4, 6]);
    }

    #[test]
    fn mark_lod_dirty_chains_descent_force_admits_under_dropped_ancestor() {
        // When a LOD>0 ancestor is dropped by AABB intersection, ALL
        // of its DAG descendants down to LOD-0 must be force-admitted
        // — even non-brush-touched siblings whose own AABBs don't
        // intersect the brush. This is the load-bearing correctness
        // property; the earlier "non-dirty ancestor covers" version
        // missed this and produced cluster-shaped voids in-editor.
        //
        // Setup: brush hits c1_x's AABB but does NOT touch c0_b
        // (brush only covers c0_a's spatial region). c0_b's only
        // ancestor c1_x is dropped → c0_b must be force-admitted via
        // descent so the chain doesn't void at the Karis-picked LOD.
        let mut entry = multi_root_entry();
        let bmin = Vec3::splat(0.0);
        let bmax = Vec3::splat(0.5);  // Only touches c0_a, c1_x, c2_x.
        let marked = mark_lod_dirty_chains(&mut entry, &[0], bmin, bmax);
        // Chain X: c0_a seed, c1_x + c2_x AABB-dirty, c0_b force-admitted
        // by descent through c1_x. Chain Y untouched.
        assert_eq!(marked, 4);
        assert_eq!(dirty_set(&entry), vec![0, 1, 4, 6]);
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
