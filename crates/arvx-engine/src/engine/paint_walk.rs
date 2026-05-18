//! Painted-material walk + worker thread. Phase E1 of `docs/PERF_DEBT.md`.
//!
//! The walk visits each dirty entity's octree to find which shader-bearing
//! materials it carries and at which tile coordinates. The result feeds
//! the per-tile anchor stream the grass / user-shader pipeline reads.
//!
//! Pre-Phase E this ran on the sim thread inline with `submit_render_frame`.
//! Phase E1 moves the per-entity scan to a dedicated worker thread:
//!
//! ```text
//!   sim tick N:    drain dirty entities → build PaintWalkBatch → submit
//!   worker:        recv batch → scan each entity → send PaintWalkResult
//!   sim tick N+k:  try_recv result → merge into painted_per_entity →
//!                  rebuild flat painted_materials / painted_anchors views
//! ```
//!
//! Where `k ≥ 1` depending on how long the walk takes vs sim pacing.
//! In steady state (region-bounded walks are <1 ms post-C1) `k = 1`,
//! so paint state visible on screen lags geometry by one frame.

use std::collections::HashMap;
use std::sync::Arc;

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use glam::Vec3;
use arvx_core::brick_pool::{BRICK_CELLS, BRICK_DIM, BRICK_INTERIOR};
use arvx_core::sparse_octree::{brick_id, is_brick, is_leaf, leaf_slot, EMPTY_NODE, INTERIOR_NODE};
use arvx_core::{Aabb, LeafAttr, LeafAttrOverlay};
use arvx_render::arvx_scene_manager::WalkSnapshot;
use arvx_render::shader_composer::UserShaderInfo;

use super::state::{EntityPaintedCache, PaintedTileEntry};

const BRICK_CELL_EMPTY: u32 = 0xFFFF_FFFFu32;

/// Sentinel tile coord for shader materials without `@tile_size`. Picked
/// far enough into i32::MIN territory that real coords can't collide
/// (would require ±2 billion-tile entities — well past usable scale).
pub(crate) const NO_TILE_COORD: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Per-entity job description. Built on sim thread; consumed on worker.
#[derive(Clone)]
pub(crate) struct PaintWalkJob {
    pub entity: hecs::Entity,
    pub root_offset: usize,
    pub depth: u8,
    pub grid_origin: Vec3,
    pub base_voxel_size: f32,
    /// Snapshot of the entity's paint overlay at submit time. Cloned
    /// rather than Arc'd — overlays are typically small (hundreds of
    /// entries) and this lets the worker scan without contending on
    /// the sim-side map.
    pub overlay: Option<LeafAttrOverlay>,
    /// Object-local inverse transform for projecting world-space dirty
    /// regions into the entity's grid frame. Identity when no Transform.
    pub entity_inverse: glam::Affine3A,
    /// World-space brush AABBs accumulated since the last walk. `None`
    /// (or empty) → full-walk fallback (asset load / geometry-epoch
    /// invalidation).
    pub regions: Option<Vec<Aabb>>,
    /// Existing per-material tile entries to seed the region-bounded
    /// rebuild. Taken out of `painted_per_entity` on sim before submit;
    /// the worker mutates this in place and ships it back in the result.
    /// Empty for the full-walk path.
    pub existing_mat_tiles: HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
}

/// A bundle of per-entity walk jobs. Shares one pool snapshot + shader
/// material table across all jobs in the batch.
pub(crate) struct PaintWalkBatch {
    pub snapshot: WalkSnapshot,
    pub jobs: Vec<PaintWalkJob>,
    pub shader_materials: Arc<HashMap<u16, UserShaderInfo>>,
    pub max_tile_size: Option<f32>,
    pub any_unsized: bool,
    /// Paint epoch at submit time. The sim uses this to attribute the
    /// result back to a consistent `painted_materials_paint_epoch`.
    pub submitted_paint_epoch: u64,
    pub submitted_geom_epoch: u64,
    /// Submission instant. Used to time the worker phase in the same
    /// `[sculpt-pipeline-sim]` telemetry that the synchronous path
    /// used to drive — sim measures wall-clock from submit→result.
    pub submitted_at: std::time::Instant,
}

/// Outcome of one walk batch. Sim merges this into `painted_per_entity`
/// on the next tick after recv.
pub(crate) struct PaintWalkResult {
    pub completed_paint_epoch: u64,
    pub completed_geom_epoch: u64,
    /// One entry per job in the originating batch. `cache.mat_tiles`
    /// may be empty → sim removes the entity from `painted_per_entity`.
    pub entries: Vec<(hecs::Entity, EntityPaintedCache)>,
    /// Wall-clock walk time on the worker thread. Excludes
    /// submit-side and merge-side overhead.
    pub worker_walk_duration: std::time::Duration,
}

/// Dedicated worker thread for painted-material walks.
///
/// Backpressure: `inbox` is bounded(1). Sim queries `is_idle()` before
/// submitting; if a batch is in flight, sim retains its dirty entities
/// and tries again next tick. This preserves the "at most one walk in
/// flight" invariant without sim having to track an `in_flight` flag —
/// `inbox.is_empty() && outbox.is_empty()` is the single source of
/// truth, but for clarity we expose [`PaintWalkWorker::is_idle`].
pub(crate) struct PaintWalkWorker {
    /// `None` only during shutdown — dropping the Sender disconnects
    /// the worker's recv loop.
    inbox_tx: Option<Sender<PaintWalkBatch>>,
    outbox_rx: Receiver<PaintWalkResult>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// Set when sim submits a batch; cleared when sim drains the
    /// matching result. Spells the invariant out explicitly so callers
    /// don't have to reason about channel emptiness.
    in_flight: std::sync::atomic::AtomicBool,
}

impl PaintWalkWorker {
    pub fn spawn() -> Self {
        let (inbox_tx, inbox_rx) = bounded::<PaintWalkBatch>(1);
        let (outbox_tx, outbox_rx) = unbounded::<PaintWalkResult>();
        let handle = std::thread::Builder::new()
            .name("arvx-paint-walk".to_string())
            .spawn(move || worker_loop(inbox_rx, outbox_tx))
            .expect("spawn arvx-paint-walk thread");
        Self {
            inbox_tx: Some(inbox_tx),
            outbox_rx,
            handle: Some(handle),
            in_flight: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// True when no batch is in flight. Sim checks this before draining
    /// dirty entities into a fresh batch — submitting while a batch is
    /// in flight would drop work on the bounded(1) inbox floor.
    pub fn is_idle(&self) -> bool {
        !self.in_flight.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Submit a batch. Caller must ensure [`Self::is_idle`] was true.
    pub fn submit(&self, batch: PaintWalkBatch) {
        self.in_flight
            .store(true, std::sync::atomic::Ordering::Release);
        let Some(tx) = self.inbox_tx.as_ref() else {
            eprintln!("[paint-walk-worker] submit after shutdown");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
            return;
        };
        if let Err(e) = tx.send(batch) {
            eprintln!("[paint-walk-worker] send failed: {e}");
            self.in_flight
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    /// Drain any completed result. Returns `None` when no batch has
    /// finished since the last call.
    pub fn try_recv(&self) -> Option<PaintWalkResult> {
        match self.outbox_rx.try_recv() {
            Ok(r) => {
                self.in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                Some(r)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.in_flight
                    .store(false, std::sync::atomic::Ordering::Release);
                None
            }
        }
    }
}

impl Drop for PaintWalkWorker {
    fn drop(&mut self) {
        // Dropping the sender disconnects the worker's recv loop; it
        // returns Err from inbox.recv() and exits naturally. Then we
        // join so the thread tears down before EngineState releases.
        drop(self.inbox_tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(inbox: Receiver<PaintWalkBatch>, outbox: Sender<PaintWalkResult>) {
    while let Ok(batch) = inbox.recv() {
        let t0 = std::time::Instant::now();
        let _ = batch.submitted_at; // diagnostic only; sim records the submit instant separately.
        let PaintWalkBatch {
            snapshot,
            jobs,
            shader_materials,
            max_tile_size,
            any_unsized,
            submitted_paint_epoch,
            submitted_geom_epoch,
            submitted_at: _,
        } = batch;

        let mut entries: Vec<(hecs::Entity, EntityPaintedCache)> = Vec::with_capacity(jobs.len());
        for job in jobs {
            let cache = walk_one(
                &snapshot,
                &shader_materials,
                max_tile_size,
                any_unsized,
                job,
            );
            entries.push(cache);
        }

        let result = PaintWalkResult {
            completed_paint_epoch: submitted_paint_epoch,
            completed_geom_epoch: submitted_geom_epoch,
            entries,
            worker_walk_duration: t0.elapsed(),
        };
        if outbox.send(result).is_err() {
            // Sim is gone — exit cleanly.
            return;
        }
    }
}

/// Run one entity's walk. Returns the (entity, cache) pair the sim
/// merges into `painted_per_entity`. An empty cache (no mat_tiles)
/// signals "remove this entity from the map."
fn walk_one(
    snapshot: &WalkSnapshot,
    shader_materials: &HashMap<u16, UserShaderInfo>,
    max_tile_size: Option<f32>,
    any_unsized: bool,
    job: PaintWalkJob,
) -> (hecs::Entity, EntityPaintedCache) {
    let PaintWalkJob {
        entity,
        root_offset,
        depth,
        grid_origin,
        base_voxel_size,
        overlay,
        entity_inverse,
        regions,
        mut existing_mat_tiles,
    } = job;

    let can_region_bound = regions
        .as_ref()
        .is_some_and(|v| !v.is_empty())
        && max_tile_size.is_some()
        && !any_unsized;

    let cache = if can_region_bound {
        // Project world-space brush AABBs into object-local space and
        // union into a single dirty box. Iterating 8 corners per region
        // keeps the bound tight even under rotation.
        let regions = regions.expect("guarded above");
        let max_ts = max_tile_size.expect("guarded above");
        let mut lmin = Vec3::splat(f32::INFINITY);
        let mut lmax = Vec3::splat(f32::NEG_INFINITY);
        for r in &regions {
            for c in r.corners() {
                let lp = entity_inverse.transform_point3(c);
                lmin = lmin.min(lp);
                lmax = lmax.max(lp);
            }
        }
        let local_dirty = Aabb { min: lmin, max: lmax };
        let walk_clip = Aabb {
            min: lmin - Vec3::splat(max_ts),
            max: lmax + Vec3::splat(max_ts),
        };

        // Clear tile entries overlapping `local_dirty` for every
        // material whose tile_size we know. Without this the rescan
        // would double-count leaves the previous walk already merged.
        for (mat_id, tile_map) in existing_mat_tiles.iter_mut() {
            let Some(info) = shader_materials.get(mat_id) else {
                tile_map.clear();
                continue;
            };
            let Some(ts) = info.tile_size else { continue };
            if ts <= 0.0 {
                continue;
            }
            let inv_ts = 1.0 / ts;
            let lo = (local_dirty.min * inv_ts).floor();
            let hi = (local_dirty.max * inv_ts).floor();
            for ix in (lo.x as i32)..=(hi.x as i32) {
                for iy in (lo.y as i32)..=(hi.y as i32) {
                    for iz in (lo.z as i32)..=(hi.z as i32) {
                        tile_map.remove(&[ix, iy, iz]);
                    }
                }
            }
        }

        scan_painted_aabbs_clipped(
            &snapshot.octree_data,
            &snapshot.brick_pool_data,
            &snapshot.leaf_attr_data,
            overlay.as_ref(),
            root_offset,
            depth,
            grid_origin,
            base_voxel_size,
            shader_materials,
            &mut existing_mat_tiles,
            walk_clip,
            local_dirty,
        );

        existing_mat_tiles.retain(|_, m| !m.is_empty());
        EntityPaintedCache {
            mat_tiles: existing_mat_tiles,
        }
    } else {
        // Full-walk path — runs on asset load / geometry-epoch
        // invalidation (no regions recorded) or whenever any shader
        // material declares no `@tile_size`.
        let mut entry = EntityPaintedCache::default();
        scan_painted_aabbs(
            &snapshot.octree_data,
            &snapshot.brick_pool_data,
            &snapshot.leaf_attr_data,
            overlay.as_ref(),
            root_offset,
            depth,
            grid_origin,
            base_voxel_size,
            shader_materials,
            &mut entry.mat_tiles,
        );
        entry
    };

    (entity, cache)
}

#[inline]
fn resolve_leaf_attr(
    overlay: Option<&LeafAttrOverlay>,
    leaf_attrs: &[LeafAttr],
    slot: u32,
) -> LeafAttr {
    if let Some(o) = overlay {
        if let Some(e) = o.get(slot) {
            return e.attr();
        }
    }
    leaf_attrs[slot as usize]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_painted_aabbs(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &[LeafAttr],
    overlay: Option<&LeafAttrOverlay>,
    root_offset: usize,
    depth: u8,
    grid_origin: Vec3,
    base_voxel_size: f32,
    shader_materials: &HashMap<u16, UserShaderInfo>,
    out: &mut HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
) {
    #[allow(clippy::too_many_arguments)]
    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &[LeafAttr],
        overlay: Option<&LeafAttrOverlay>,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: Vec3,
        base_vs: f32,
        shader_materials: &HashMap<u16, UserShaderInfo>,
        out: &mut HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
    ) {
        if offset >= octree_data.len() {
            return;
        }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE {
            return;
        }
        if is_brick(node) {
            let brick_id = brick_id(node);
            let base_idx = (brick_id * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() {
                            continue;
                        }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR {
                            continue;
                        }
                        let attr = resolve_leaf_attr(overlay, leaf_attrs, cell);
                        let primary = attr.material_primary;
                        let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
                        let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
                        let painted_mat = if shader_materials.contains_key(&primary) {
                            Some(primary)
                        } else if blend > 0 && shader_materials.contains_key(&secondary) {
                            Some(secondary)
                        } else {
                            None
                        };
                        if let Some(mat) = painted_mat {
                            let cell_voxel = glam::UVec3::new(
                                coord_voxels.x + cx,
                                coord_voxels.y + cy,
                                coord_voxels.z + cz,
                            );
                            let cell_local = grid_origin
                                + Vec3::new(
                                    cell_voxel.x as f32,
                                    cell_voxel.y as f32,
                                    cell_voxel.z as f32,
                                ) * base_vs;
                            let cell_max = cell_local + Vec3::splat(base_vs);
                            let tile_size =
                                shader_materials.get(&mat).and_then(|i| i.tile_size);
                            expand_aabb(
                                out,
                                mat,
                                cell_local,
                                cell_max,
                                attr.normal(),
                                tile_size,
                                None,
                            );
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            let slot = leaf_slot(node);
            let attr = resolve_leaf_attr(overlay, leaf_attrs, slot);
            let primary = attr.material_primary;
            let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
            let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
            let painted_mat = if shader_materials.contains_key(&primary) {
                Some(primary)
            } else if blend > 0 && shader_materials.contains_key(&secondary) {
                Some(secondary)
            } else {
                None
            };
            if let Some(mat) = painted_mat {
                let voxels_per_side = 1u32 << (max_depth - level);
                let leaf_size = voxels_per_side as f32 * base_vs;
                let leaf_min = grid_origin
                    + Vec3::new(
                        coord_voxels.x as f32,
                        coord_voxels.y as f32,
                        coord_voxels.z as f32,
                    ) * base_vs;
                let leaf_max = leaf_min + Vec3::splat(leaf_size);
                let tile_size = shader_materials.get(&mat).and_then(|i| i.tile_size);
                expand_aabb(
                    out,
                    mat,
                    leaf_min,
                    leaf_max,
                    attr.normal(),
                    tile_size,
                    None,
                );
            }
            return;
        }
        if level >= max_depth {
            return;
        }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            let child_offset = node as usize + octant as usize;
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                overlay,
                child_offset,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                shader_materials,
                out,
            );
        }
    }

    walk(
        octree_data,
        brick_pool,
        leaf_attrs,
        overlay,
        root_offset,
        0,
        depth,
        glam::UVec3::ZERO,
        grid_origin,
        base_voxel_size,
        shader_materials,
        out,
    );
}

/// Region-bounded scan. Skips octree nodes whose cell AABB doesn't
/// intersect `walk_clip`, and forwards `dirty_local` to [`expand_aabb`]
/// so per-tile inserts are confined to the cleared region.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_painted_aabbs_clipped(
    octree_data: &[u32],
    brick_pool: &[u32],
    leaf_attrs: &[LeafAttr],
    overlay: Option<&LeafAttrOverlay>,
    root_offset: usize,
    depth: u8,
    grid_origin: Vec3,
    base_voxel_size: f32,
    shader_materials: &HashMap<u16, UserShaderInfo>,
    out: &mut HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
    walk_clip: Aabb,
    dirty_local: Aabb,
) {
    #[allow(clippy::too_many_arguments)]
    fn walk(
        octree_data: &[u32],
        brick_pool: &[u32],
        leaf_attrs: &[LeafAttr],
        overlay: Option<&LeafAttrOverlay>,
        offset: usize,
        level: u8,
        max_depth: u8,
        coord_voxels: glam::UVec3,
        grid_origin: Vec3,
        base_vs: f32,
        shader_materials: &HashMap<u16, UserShaderInfo>,
        out: &mut HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
        walk_clip: Aabb,
        dirty_local: Aabb,
    ) {
        if offset >= octree_data.len() {
            return;
        }
        // Node AABB in object-local space.
        let node_side_voxels = 1u32 << (max_depth - level);
        let node_min = grid_origin
            + Vec3::new(
                coord_voxels.x as f32,
                coord_voxels.y as f32,
                coord_voxels.z as f32,
            ) * base_vs;
        let node_max = node_min + Vec3::splat(node_side_voxels as f32 * base_vs);
        let node_aabb = Aabb { min: node_min, max: node_max };
        if !walk_clip.intersects(&node_aabb) {
            return;
        }
        let node = octree_data[offset];
        if node == EMPTY_NODE || node == INTERIOR_NODE {
            return;
        }
        if is_brick(node) {
            let brick_id = brick_id(node);
            let base_idx = (brick_id * BRICK_CELLS) as usize;
            for cz in 0..BRICK_DIM {
                for cy in 0..BRICK_DIM {
                    for cx in 0..BRICK_DIM {
                        let cell_idx = (cz * BRICK_DIM * BRICK_DIM + cy * BRICK_DIM + cx) as usize;
                        let pool_idx = base_idx + cell_idx;
                        if pool_idx >= brick_pool.len() {
                            continue;
                        }
                        let cell = brick_pool[pool_idx];
                        if cell == BRICK_CELL_EMPTY || cell == BRICK_INTERIOR {
                            continue;
                        }
                        let attr = resolve_leaf_attr(overlay, leaf_attrs, cell);
                        let primary = attr.material_primary;
                        let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
                        let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
                        let painted_mat = if shader_materials.contains_key(&primary) {
                            Some(primary)
                        } else if blend > 0 && shader_materials.contains_key(&secondary) {
                            Some(secondary)
                        } else {
                            None
                        };
                        if let Some(mat) = painted_mat {
                            let cell_voxel = glam::UVec3::new(
                                coord_voxels.x + cx,
                                coord_voxels.y + cy,
                                coord_voxels.z + cz,
                            );
                            let cell_local = grid_origin
                                + Vec3::new(
                                    cell_voxel.x as f32,
                                    cell_voxel.y as f32,
                                    cell_voxel.z as f32,
                                ) * base_vs;
                            let cell_max = cell_local + Vec3::splat(base_vs);
                            let cell_aabb = Aabb { min: cell_local, max: cell_max };
                            if !walk_clip.intersects(&cell_aabb) {
                                continue;
                            }
                            let tile_size =
                                shader_materials.get(&mat).and_then(|i| i.tile_size);
                            expand_aabb(
                                out,
                                mat,
                                cell_local,
                                cell_max,
                                attr.normal(),
                                tile_size,
                                Some(dirty_local),
                            );
                        }
                    }
                }
            }
            return;
        }
        if is_leaf(node) {
            let slot = leaf_slot(node);
            let attr = resolve_leaf_attr(overlay, leaf_attrs, slot);
            let primary = attr.material_primary;
            let secondary: u16 = attr.material_secondary_blend & 0x0FFF;
            let blend: u16 = (attr.material_secondary_blend >> 12) & 0xF;
            let painted_mat = if shader_materials.contains_key(&primary) {
                Some(primary)
            } else if blend > 0 && shader_materials.contains_key(&secondary) {
                Some(secondary)
            } else {
                None
            };
            if let Some(mat) = painted_mat {
                let voxels_per_side = 1u32 << (max_depth - level);
                let leaf_size = voxels_per_side as f32 * base_vs;
                let leaf_min = grid_origin
                    + Vec3::new(
                        coord_voxels.x as f32,
                        coord_voxels.y as f32,
                        coord_voxels.z as f32,
                    ) * base_vs;
                let leaf_max = leaf_min + Vec3::splat(leaf_size);
                let tile_size = shader_materials.get(&mat).and_then(|i| i.tile_size);
                expand_aabb(
                    out,
                    mat,
                    leaf_min,
                    leaf_max,
                    attr.normal(),
                    tile_size,
                    Some(dirty_local),
                );
            }
            return;
        }
        if level >= max_depth {
            return;
        }
        let child_voxels = 1u32 << (max_depth - level - 1);
        for octant in 0u32..8 {
            let dx = octant & 1;
            let dy = (octant >> 1) & 1;
            let dz = (octant >> 2) & 1;
            let child_coord = glam::UVec3::new(
                coord_voxels.x + dx * child_voxels,
                coord_voxels.y + dy * child_voxels,
                coord_voxels.z + dz * child_voxels,
            );
            let child_offset = node as usize + octant as usize;
            walk(
                octree_data,
                brick_pool,
                leaf_attrs,
                overlay,
                child_offset,
                level + 1,
                max_depth,
                child_coord,
                grid_origin,
                base_vs,
                shader_materials,
                out,
                walk_clip,
                dirty_local,
            );
        }
    }

    walk(
        octree_data,
        brick_pool,
        leaf_attrs,
        overlay,
        root_offset,
        0,
        depth,
        glam::UVec3::ZERO,
        grid_origin,
        base_voxel_size,
        shader_materials,
        out,
        walk_clip,
        dirty_local,
    );
}

pub(crate) fn expand_aabb(
    out: &mut HashMap<u16, HashMap<[i32; 3], PaintedTileEntry>>,
    mat: u16,
    mn: Vec3,
    mx: Vec3,
    normal: Vec3,
    tile_size: Option<f32>,
    dirty_local: Option<Aabb>,
) {
    let mat_map = out.entry(mat).or_default();

    fn merge(
        mat_map: &mut HashMap<[i32; 3], PaintedTileEntry>,
        key: [i32; 3],
        mn: Vec3,
        mx: Vec3,
        normal: Vec3,
    ) {
        let entry = mat_map.entry(key).or_insert_with(PaintedTileEntry::empty);
        entry.aabb.min = entry.aabb.min.min(mn);
        entry.aabb.max = entry.aabb.max.max(mx);
        entry.leaf_count = entry.leaf_count.saturating_add(1);
        entry.normal_sum += normal;
    }

    match tile_size {
        None => {
            merge(mat_map, NO_TILE_COORD, mn, mx, normal);
        }
        Some(s) if s > 0.0 => {
            let inv = 1.0 / s;
            let lo = (mn * inv).floor();
            let hi = ((mx - Vec3::splat(1e-6)) * inv).floor();
            for ix in (lo.x as i32)..=(hi.x as i32) {
                for iy in (lo.y as i32)..=(hi.y as i32) {
                    for iz in (lo.z as i32)..=(hi.z as i32) {
                        if let Some(d) = dirty_local {
                            let tile_min =
                                Vec3::new(ix as f32, iy as f32, iz as f32) * s;
                            let tile_max = tile_min + Vec3::splat(s);
                            let tile_aabb = Aabb { min: tile_min, max: tile_max };
                            if !d.intersects(&tile_aabb) {
                                continue;
                            }
                        }
                        merge(mat_map, [ix, iy, iz], mn, mx, normal);
                    }
                }
            }
        }
        Some(_) => merge(mat_map, NO_TILE_COORD, mn, mx, normal),
    }
}

