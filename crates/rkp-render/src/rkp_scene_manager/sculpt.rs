//! Runtime sculpt — mutates an asset's CPU octree, re-extracts its
//! surface mesh, rebuilds the cluster DAG, and re-allocates the GPU
//! octree handle so the next frame renders the modified geometry.
//!
//! ## What this is, what it isn't
//!
//! This is Phase 2's "make sculpt visible" path. Each call does a
//! **full asset re-bake**:
//!
//! * Apply the Phase 1 [`compute_brush_edits`] / [`apply_delta`] kernel
//!   to the entry's [`SparseOctree`]
//! * Allocate fresh [`LeafAttrPool`] slots for `Add` edits and write
//!   their attrs (brush material + outward-sphere normal)
//! * Re-extract surface mesh via [`extract_surface_mesh`] and rebuild
//!   the full cluster DAG via [`build_cluster_dag`]
//! * Deallocate the asset's old GPU octree handle and allocate a fresh
//!   one from the mutated tree
//! * Bump `geometry_epoch` so render re-uploads on the next frame
//!
//! Per-stamp cost scales with asset size (~tens of ms on small assets,
//! seconds on a 2.5 M-vert elephant). That's tolerable for "verify the
//! tool works" and gets the visible-feedback loop closed. The Phase 5
//! perf pass replaces this with per-cluster incremental re-bake.
//!
//! ## Known limitations (Phase 4+ work)
//!
//! * **Asset sharing not handled** — sculpting one entity of a shared
//!   `.rkp` mutates the shared `AssetEntry` and therefore *also*
//!   deforms every sibling instance. Clone-on-write is Phase 4.
//! * **Brick regions are silently skipped** — the Phase 1 kernel only
//!   sees LEAF nodes (`is_leaf == true`); `BRICK` nodes (compressed
//!   sub-trees) and `INTERIOR_NODE` regions are no-ops. For typical
//!   surface-shell assets (LEAF on the shell, BRICK / INTERIOR_NODE
//!   inside) Carve on the visible surface still works, but carving
//!   deep into a solid interior won't expose anything.
//! * **Newly-allocated slots leak on `release_asset`** — the entry's
//!   `leaf_attr_slot_start / _count` describes the asset's original
//!   contiguous allocation; sculpt-added slots come from the pool's
//!   tail allocator and aren't tracked for free-on-release.
//! * **Paint on sculpt-added leaves silently no-ops** — paint clamps
//!   to the entry's slot range; sculpt-added slots fall outside it.

use std::collections::HashSet;

use glam::{Affine3A, Vec3};

use rkp_core::sculpt::{
    apply_delta, compute_brush_edits, BrushMode, BrushOp, LeafEditOp,
};
use rkp_core::sparse_octree::{is_leaf, leaf_slot};
use rkp_core::LeafAttr;

use super::manager::RkpSceneManager;
use super::types::{AssetHandle, AssetInfo};

/// Outcome of a successful [`RkpSceneManager::apply_sculpt_brush`].
/// Carries the refreshed [`AssetInfo`] so the caller can update its
/// per-entity `Renderable.spatial` cache (the asset's GPU octree
/// handle, leaf-attr range, and AABB may all have changed).
#[derive(Debug, Clone)]
pub struct SculptApplyResult {
    pub new_info: AssetInfo,
    pub leaves_added: usize,
    pub leaves_removed: usize,
}

impl RkpSceneManager {
    /// Apply one sculpt brush stamp against an asset's geometry.
    ///
    /// Returns `None` when:
    /// * The handle is unknown.
    /// * The asset isn't an octree-backed spatial.
    /// * The brush footprint produces no edits (outside the asset or
    ///   over regions that don't qualify under Phase 1 semantics — see
    ///   [`rkp_core::sculpt`] for the rules).
    ///
    /// `brush_radius` is in world-space units. Object-local scale is
    /// applied via `entity_world.to_scale_rotation_translation()` (mean
    /// of the three scale axes, matching paint's convention).
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
        let (op, depth, base_vs, asset_grid_origin) = {
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
            // size, which matches the Phase 1 kernel's unit convention.
            let center_grid = (center_local - asset_grid_origin) / base_vs;
            let radius_grid = local_radius / base_vs;

            let op = BrushOp {
                center: center_grid,
                radius: radius_grid,
                falloff: brush_falloff,
                mode,
                material,
            };
            eprintln!(
                "[sculpt-debug] world_pos=({:.3}, {:.3}, {:.3}) local=({:.3}, {:.3}, {:.3}) \
                 grid=({:.2}, {:.2}, {:.2}) r_world={:.4} r_grid={:.2} \
                 base_vs={:.5} mean_scale={:.3} mode={:?} extent_voxels={}",
                world_pos.x, world_pos.y, world_pos.z,
                center_local.x, center_local.y, center_local.z,
                center_grid.x, center_grid.y, center_grid.z,
                brush_radius, radius_grid, base_vs, mean_scale,
                mode, 1u32 << depth,
            );
            (op, depth, base_vs, asset_grid_origin)
        };

        // ── 2. Compute edit list against current octree (read-only). ─
        let delta = {
            let entry = self.asset_cache.get(handle)?;
            compute_brush_edits(&entry.cpu_octree, op)
        };
        if delta.is_empty() {
            return None;
        }

        let leaves_added = delta.count_added();
        let leaves_removed = delta.count_removed();

        // ── 3. Allocate slots for Add edits up-front. ─────────────────
        //
        // We hand the kernel a closure that pulls from a pre-allocated
        // vec — this way slot allocation happens here (mutating the
        // shared pool) instead of inside `apply_delta` (which only
        // wants `FnMut() -> u32`).
        let new_slots: Vec<u32> = if leaves_added > 0 {
            self.leaf_attr_pool
                .allocate_range(leaves_added as u32)
                .expect("leaf_attr_pool exhausted during sculpt — bump pool capacity")
        } else {
            Vec::new()
        };
        debug_assert_eq!(new_slots.len(), leaves_added);

        // ── 4. Apply delta to the entry's CPU octree. ─────────────────
        let mut slot_cursor = 0usize;
        let applied = {
            let entry = self.asset_cache.get_mut(handle)?;
            apply_delta(&mut entry.cpu_octree, &delta, || {
                let s = new_slots[slot_cursor];
                slot_cursor += 1;
                s
            })
        };
        debug_assert_eq!(slot_cursor, leaves_added);

        // ── 5. Write LeafAttr entries for the freshly-allocated slots. ─
        for (edit, slot) in delta.edits.iter()
            .filter_map(|e| match e.op {
                LeafEditOp::Add { material, normal } => Some((material, normal)),
                _ => None,
            })
            .zip(applied.allocated_slots.iter().map(|(s, _)| *s))
        {
            let (material, normal) = edit;
            let attr = LeafAttr::new_blended(normal, material, 0, 0);
            *self.leaf_attr_pool.get_mut(slot) = attr;
        }

        // ── 6. Re-pack and re-allocate the GPU octree handle. ────────
        //
        // The mutated `cpu_octree` may have grown its node buffer (each
        // `set_at_level` that hits an `EMPTY_NODE` / `INTERIOR_NODE`
        // subdivides). `collapse_all` reclaims uniformity that surfaced
        // post-mutation; `compact` shrinks the dead slots from the
        // subdivisions back out. Without these, repeated stamps would
        // accumulate orphaned children in `nodes`.
        let (_new_handle, new_min_slot, new_max_slot_plus_one) = {
            let entry = self.asset_cache.get_mut(handle)?;
            entry.cpu_octree.collapse_all();
            entry.cpu_octree.compact();

            // Find the min/max slot id referenced by any LEAF node so
            // we can extend the asset's slot range to cover them.
            // Bricks aren't traversed by sculpt; their slot ids stay
            // within the original range.
            let (mut min_slot, mut max_slot_plus_one) = (
                entry.leaf_attr_slot_start,
                entry.leaf_attr_slot_start + entry.leaf_attr_slot_count,
            );
            for n in entry.cpu_octree.as_slice() {
                if is_leaf(*n) {
                    let s = leaf_slot(*n);
                    if s < min_slot {
                        min_slot = s;
                    }
                    if s + 1 > max_slot_plus_one {
                        max_slot_plus_one = s + 1;
                    }
                }
            }

            let old_handle = entry.spatial_handle;
            let new_handle = self.octree.allocate(&entry.cpu_octree);
            self.octree.deallocate(old_handle);

            let entry = self.asset_cache.get_mut(handle)?;
            entry.spatial_handle = new_handle;
            (new_handle, min_slot, max_slot_plus_one)
        };

        // ── 7. Re-extract surface mesh + rebuild cluster DAG. ────────
        let (new_vertices, new_indices, new_clusters, new_lod0_index_count) = {
            // Hold the read borrow on the entry briefly to pull the
            // tree slice; release before extracting so the pools can
            // be re-borrowed.
            let entry = self.asset_cache.get(handle)?;
            let tree_slice: Vec<u32> = entry.cpu_octree.as_slice().to_vec();
            // The leaf_attr_pool / brick_pool slices are owned by self;
            // we can read them without conflicting with `entry` since
            // they're sibling fields.
            let brick_slice: Vec<u32> = self.brick_pool.as_slice().to_vec();
            let leaf_attr_slice: Vec<LeafAttr> =
                self.leaf_attr_pool.as_slice().to_vec();
            let bone_slice: Vec<rkp_core::companion::BoneVoxel> =
                self.leaf_attr_pool.bones_as_slice().to_vec();

            let (verts, indices) = rkp_core::mesh_extract::extract_surface_mesh(
                &tree_slice,
                depth,
                base_vs,
                asset_grid_origin,
                &brick_slice,
                &leaf_attr_slice,
                &bone_slice,
            );
            // Conversion to the rkp-render-local MeshVertex type the
            // pipeline expects. The two share the same `#[repr(C)]`
            // layout (verified via bytemuck cast).
            let v_render: Vec<crate::mesh_pass::MeshVertex> =
                bytemuck::cast_slice(&verts).to_vec();
            // Phase 2 POC: build LOD-0 only. The full DAG simplification
            // takes 6 s+ per stamp on big assets, which freezes drag.
            // LOD=1 produces a valid renderable DAG where every cluster
            // is admitted at finest level; per-cluster incremental
            // re-bake (Phase 5 perf) supersedes both.
            let dag = rkp_core::mesh_lod::build_cluster_dag_with_levels(
                &v_render, &indices, 1,
            );
            let lod0 = dag.lod0_index_range.1 - dag.lod0_index_range.0;
            (v_render, dag.indices, dag.clusters, lod0)
        };

        // ── 8. Update entry mesh fields + slot range. ────────────────
        let entry = self.asset_cache.get_mut(handle)?;
        entry.mesh_vertices = new_vertices;
        entry.mesh_indices = new_indices;
        entry.meshlet_clusters = new_clusters;
        entry.mesh_lod0_index_count = new_lod0_index_count;
        entry.leaf_attr_slot_start = new_min_slot;
        entry.leaf_attr_slot_count = new_max_slot_plus_one - new_min_slot;
        // `SparseOctree::leaf_count` doesn't handle BRICK nodes
        // (treats them as branches and OOBs the node array). The
        // delta-derived increment is correct anyway: sculpt only
        // mutates LEAF nodes, brick cells are untouched.
        let delta_count = leaves_added as i64 - leaves_removed as i64;
        entry.voxel_count = (entry.voxel_count as i64 + delta_count).max(0) as u32;

        let new_info = entry.info();

        // ── 9. Drop the snapshot cache and bump epoch. ───────────────
        self.walk_snapshot_cache = None;
        self.bump_geometry_epoch();

        // Defensive: assert all freed slots in `applied.freed_slots`
        // are inside the asset's leaf-attr range we tracked. Without
        // this, a Phase-1 bug could free a slot that was never the
        // asset's. (Free-list deallocation is intentionally skipped
        // for now — see "known limitations" in the module doc.)
        if cfg!(debug_assertions) && !applied.freed_slots.is_empty() {
            let lo = new_info.leaf_attr_slot_start;
            let hi = lo + new_info.leaf_attr_slot_count;
            let freed: HashSet<u32> = applied.freed_slots.iter().copied().collect();
            for f in &freed {
                debug_assert!(
                    *f >= lo && *f < hi,
                    "freed slot {} outside asset range [{}, {})",
                    f, lo, hi,
                );
            }
        }

        Some(SculptApplyResult {
            new_info,
            leaves_added,
            leaves_removed,
        })
    }
}
