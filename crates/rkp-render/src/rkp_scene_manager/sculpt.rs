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

use glam::{Affine3A, Vec3};

use rkp_core::sculpt::{
    compute_brush_edits, BrushMode, BrushOp, LeafEditOp,
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

        // ── 2. Compute edit list against current octree (read-only). ─
        let delta = {
            let entry = self.asset_cache.get(handle)?;
            compute_brush_edits(&entry.cpu_octree, op)
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
        if removed.is_empty() && leaves_add_skipped == 0 {
            return None;
        }

        // Sort + dedupe so the engine-side `insert_batch` walks the
        // smallest set possible. The kernel emits coords in row-major
        // order so adjacent finest-voxel cells inside one brick share
        // the brick's slot ids for at-most a handful of entries —
        // sorting collapses the obvious duplicates.
        removed.sort_unstable();
        removed.dedup();

        eprintln!(
            "[sculpt] stamp handle={:?} mode={:?} edits={} removed={} add_skipped={} (depth={}, base_vs={:.5})",
            handle, mode, delta.len(), removed.len(), leaves_add_skipped, depth, base_vs,
        );

        Some(SculptApplyResult {
            removed_leaf_attr_ids: removed,
            leaves_removed,
            leaves_add_skipped,
        })
    }
}
