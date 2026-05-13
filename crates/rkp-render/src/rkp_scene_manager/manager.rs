//! `RkpSceneManager` — the central CPU-side scene type. Holds the
//! leaf_attr / brick / octree pools + asset cache + brush overlay
//! state + the geometry/paint/brush epoch counters that drive the
//! render thread's re-upload decisions.
//!
//! Core methods (construction, faces, geometry epoch, slices,
//! deallocation) live here. Asset lifecycle methods live in
//! [`super::asset_load`]; paint methods in [`super::paint`]; voxelize
//! methods in [`super::voxelize`]. All operate on this struct via
//! per-file `impl RkpSceneManager` blocks; private fields are
//! `pub(super)` so sibling impls can drive them directly.

use rkp_core::{BrickPool, LeafAttrPool, OctreeHandle, SparseOctree};

use crate::octree_gpu::OctreeGpu;
use crate::rkp_scene::GeometryUpload;

use super::types::{emit_faces, AssetCache, FaceInstance};

/// Process-start anchor used by [`process_elapsed_ns`]. Lazy-initialised
/// on first read; nanos-since-this-anchor fit in u64 for ~584 years.
fn process_start() -> std::time::Instant {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    *START.get_or_init(std::time::Instant::now)
}

/// Nanoseconds elapsed since [`process_start`]. Cross-thread-readable
/// (writable as `AtomicU64`) for cheap latency diagnostics.
fn process_elapsed_ns() -> u64 {
    process_start().elapsed().as_nanos() as u64
}

/// Compute the wall-clock millisecond gap between `nanos_then`
/// (typically `last_geometry_bump_ns()`) and now. Returns `None` if
/// `nanos_then` is zero (never set) or "in the future" (race
/// condition between read and now-sample).
pub fn ms_since_process_ns(nanos_then: u64) -> Option<f64> {
    if nanos_then == 0 {
        return None;
    }
    let now = process_elapsed_ns();
    if now < nanos_then {
        return None;
    }
    Some((now - nanos_then) as f64 / 1.0e6)
}

/// Lock-free snapshot of the three pools the painted-material walk
/// reads. Built under a brief `scene_mgr` lock as constant-time
/// `Arc::clone`s of the pool storage; the walk traverses the snapshot
/// without holding any lock, so sim and render don't serialize on
/// the duration of the walk (~80 ms on big asset trees).
///
/// **No memcpy on construction** — each pool stores its data as
/// `Arc<Vec<…>>` (see PERF_DEBT.md A2). The walk shares the buffer
/// with the pool until either the walk drops its `Arc` (refcount
/// returns to 1, future writes stay in place) or the pool's next
/// mutation calls `Arc::make_mut` (one-time clone-on-write).
///
/// `epoch` is the `geometry_epoch` snapshot was taken at — purely
/// diagnostic; lets a long-running walk verify it operated against
/// the geometry generation it expected.
#[derive(Clone)]
pub struct WalkSnapshot {
    pub octree_data: std::sync::Arc<Vec<u32>>,
    pub brick_pool_data: std::sync::Arc<Vec<u32>>,
    pub leaf_attr_data: std::sync::Arc<Vec<rkp_core::LeafAttr>>,
    /// `geometry_epoch` value at the time this snapshot was taken.
    pub epoch: u64,
}

/// CPU-side scene manager — leaf_attr data, bricks, octrees, face instances.
pub struct RkpSceneManager {
    /// Per-leaf attributes: {material_primary, material_secondary+blend,
    /// normal} + parallel per-leaf color. The sole per-voxel payload now
    /// that opacity has been removed.
    pub leaf_attr_pool: LeafAttrPool,
    /// Pool of fixed-size bricks (4³ flat cells each). The octree's deepest
    /// branches point at bricks; the shader does flat brick lookups instead
    /// of descending the final two octree levels per step.
    pub brick_pool: BrickPool,
    /// Face-adjacency links for every allocated brick — indexed by
    /// `brick_id`, 6 u32 per entry (−X, +X, −Y, +Y, −Z, +Z). Each entry
    /// is either a neighboring brick_id or a FACE_EMPTY / FACE_INTERIOR
    /// sentinel (see `rkp_core::brick_face_links`). Sized to cover
    /// `brick_pool.allocated_count()`; newly-allocated bricks append
    /// sentinel rows until voxelize / load_asset fills them in.
    pub brick_face_links: Vec<[u32; 6]>,
    /// GPU octree allocator (packs all octrees into one buffer).
    pub octree: OctreeGpu,
    /// Cache of loaded .rkp assets keyed by canonical file path. Instances
    /// of the same asset share one octree + one leaf_attr range + one brick
    /// range via refcounting — release_asset frees them when the last
    /// instance goes away.
    pub(super) asset_cache: AssetCache,
    /// Face instances for rasterization (surface shell).
    pub(super) pending_faces: Vec<FaceInstance>,
    /// Whether face data needs re-upload to GPU.
    pub(super) faces_dirty: bool,
    /// Monotonic counter incremented every time geometry data changes
    /// (asset load/release/reload, voxelize, integrate_artifact). The
    /// render thread compares this against its own last-uploaded
    /// epoch to decide whether to call `geometry_upload` + re-upload
    /// to the GPU. Survives lost snapshots: if a snapshot carrying
    /// an epoch bump is dropped by the newest-wins inbox, the next
    /// snapshot still carries the same (or higher) epoch and render
    /// catches up.
    ///
    /// **Wrapped in `Arc<AtomicU64>`** so sim and render can read the
    /// epoch lock-free via [`Self::epoch_handle`]. The previous
    /// design had sim taking the `scene_mgr` Mutex every tick just
    /// to read this counter — fine when nothing else held the lock,
    /// but a 50 ms bake_worker integrate would block sim's tick for
    /// 50 ms, dropping sim from 60 Hz to ~20 Hz with every bake.
    /// Now sim clones the Arc once at startup and reads the counter
    /// directly; only the actual geometry-mutation methods need the
    /// Mutex (which they already hold via `&mut self`).
    pub(super) geometry_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Wall-clock nanoseconds (relative to the lazy-initialised
    /// process-start anchor inside `bump_geometry_epoch`) of the most
    /// recent epoch bump. Sim writes; render reads on geo-epoch start
    /// to log the `[sculpt-pipeline] sim→render` wait — the gap that
    /// the existing component-level timings don't capture (sim mutex
    /// hold, render-thread cadence, vsync alignment).
    ///
    /// `0` until the first bump. Loosely synchronized — Relaxed
    /// ordering is fine because the timestamp is for diagnostics, not
    /// for correctness.
    pub(super) last_geometry_bump_ns: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Wall-clock nanoseconds when the sim last submitted a
    /// `RenderFrame` carrying a fresh geometry epoch into the render
    /// inbox. Companion to `last_geometry_bump_ns`: the `bump→submit`
    /// delta is sim post-bump work (snapshot building, palette,
    /// gpu_objects rebuild); the `submit→render-pickup` delta is the
    /// render thread cadence + GPU backpressure.
    pub(super) last_geometry_submit_ns: std::sync::Arc<std::sync::atomic::AtomicU64>,

    // ── Paint data writes (Phase 3b perf) ───────────────────────────
    /// Separate epoch for paint mutations. Pre-perf: paint bumped
    /// Bumped by `apply_paint_sphere` whenever a stamp writes into a
    /// per-instance overlay. Sim reads the value via the shared atomic
    /// handle to drive UI (cursor refresh, save indicator). The actual
    /// overlay data is shipped through `RenderFrame.gpu_instance_overlays`
    /// and uploaded by the render thread inside `RkpScene::upload_frame`,
    /// so this epoch is informational — there's no longer a paint-only
    /// fast path on the render side that gates on it.
    pub(super) paint_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl RkpSceneManager {
    /// Create with default capacity.
    pub fn new(capacity: u32) -> Self {
        Self {
            leaf_attr_pool: LeafAttrPool::new(capacity),
            brick_pool: BrickPool::new((capacity / 16).max(64)),
            brick_face_links: Vec::new(),
            octree: OctreeGpu::new(),
            asset_cache: AssetCache::default(),
            pending_faces: Vec::new(),
            faces_dirty: false,
            geometry_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_geometry_bump_ns: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_geometry_submit_ns: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            paint_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Reset every pool / cache to empty without breaking the
    /// shared-epoch handle. Use this for "wipe the scene" scenarios
    /// (project close, project switch) — replacing the entire
    /// `RkpSceneManager` instance would create a fresh epoch atomic
    /// orphaning any handles sim/render are holding (visible bug:
    /// render stops uploading geometry → everything renders as the
    /// raw bounding-box cubes). The epoch bumps after the reset so
    /// any consumer holding the handle sees the change and re-uploads
    /// the (now-empty) geometry.
    pub fn clear(&mut self, capacity: u32) {
        self.leaf_attr_pool = LeafAttrPool::new(capacity);
        self.brick_pool = BrickPool::new((capacity / 16).max(64));
        self.brick_face_links.clear();
        self.octree = OctreeGpu::new();
        self.asset_cache = AssetCache::default();
        self.pending_faces.clear();
        self.faces_dirty = false;
        // Preserve the Arc identity, but bump the value so the
        // shared handle observes the wipe.
        self.bump_geometry_epoch();
    }

    /// Splice one asset's computed face-link rows into the scene-wide
    /// table. The rows are indexed by global brick_id (the asset's
    /// voxelize/load produced them that way), so we copy in place.
    pub(super) fn merge_face_links(&mut self, rows: &[[u32; 6]]) {
        if rows.is_empty() {
            return;
        }
        if self.brick_face_links.len() < rows.len() {
            self.brick_face_links.resize(
                rows.len(),
                [rkp_core::brick_face_links::FACE_EMPTY; 6],
            );
        }
        // Copy only rows that correspond to bricks the asset actually owns
        // (identified by a non-all-empty row — unused slots stay at the
        // default FACE_EMPTY sentinel). This is equivalent to looping
        // over the asset's brick_ids, but avoids threading that list
        // through every call site.
        for (i, row) in rows.iter().enumerate() {
            if row.iter().any(|&v| v != rkp_core::brick_face_links::FACE_EMPTY) {
                self.brick_face_links[i] = *row;
            }
        }
    }

    // ── Face emission ────────────────────────────────────────────────

    pub fn emit_faces_from_octree(
        &mut self,
        octree: &SparseOctree,
        obj_idx: u32,
    ) {
        emit_faces(octree, obj_idx, &mut self.pending_faces);
        self.faces_dirty = true;
    }

    pub fn emit_faces_from_raw_octree(
        &mut self,
        nodes: &[u32],
        depth: u8,
        base_vs: f32,
        obj_idx: u32,
    ) {
        let octree = SparseOctree::from_raw(nodes, depth, base_vs);
        emit_faces(&octree, obj_idx, &mut self.pending_faces);
        self.faces_dirty = true;
    }

    pub fn pending_faces(&self) -> &[FaceInstance] { &self.pending_faces }
    pub fn faces_dirty(&self) -> bool { self.faces_dirty }
    pub fn mark_faces_clean(&mut self) { self.faces_dirty = false; }

    /// Monotonic counter that ticks every time geometry data
    /// (octree, leaf attrs, brick pool, brick face links) changes.
    /// Render compares this to its own last-uploaded epoch each
    /// frame and re-uploads when behind. Robust to snapshot drops:
    /// since the next snapshot still carries the latest epoch, a
    /// dropped intermediate snapshot doesn't lose the upload.
    ///
    /// Lock-free read — but the caller still has to dereference
    /// through the `Arc<Mutex<RkpSceneManager>>`, which means *they
    /// already hold the Mutex*. For per-tick lock-free reads from
    /// sim or render, use [`Self::epoch_handle`] to clone the
    /// underlying `Arc<AtomicU64>` once at startup, then load on it
    /// directly without ever touching the Mutex.
    pub fn geometry_epoch(&self) -> u64 {
        self.geometry_epoch
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Clone the geometry-epoch atomic for lock-free reads outside
    /// the `scene_mgr` Mutex. Hold the returned `Arc` in sim and
    /// render; load via `handle.load(Ordering::Acquire)` to get the
    /// current epoch without contending with bake_worker for the
    /// scene_mgr Mutex.
    pub fn epoch_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.geometry_epoch.clone()
    }

    /// Bump the geometry epoch. Called by every method that mutates
    /// the GPU-uploaded geometry buffers (asset acquire/release,
    /// voxelize, integrate_artifact, deallocate_geometry, …). External
    /// callers that mutate scene_mgr through other paths can also
    /// invoke this manually to force a render-side re-upload.
    ///
    /// Takes `&mut self` to keep the API symmetric with the
    /// mutation methods that wrap it (and to require the caller
    /// already holds the scene_mgr Mutex), but the counter itself
    /// is atomic — Release ordering pairs with the Acquire load in
    /// `geometry_epoch()` so render observes the bump after any
    /// preceding writes to the geometry data are visible.
    pub fn bump_geometry_epoch(&mut self) {
        self.geometry_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        let ns = process_elapsed_ns();
        self.last_geometry_bump_ns
            .store(ns, std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the nanoseconds-since-process-start when
    /// [`Self::bump_geometry_epoch`] was last called. Returns `0` if
    /// the epoch has never been bumped. Used by the render worker to
    /// log the sim→render wait gap (`[sculpt-pipeline]`).
    pub fn last_geometry_bump_ns(&self) -> u64 {
        self.last_geometry_bump_ns
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record the sim-side submit timestamp for the latest bumped
    /// geometry epoch. Called by `submit_render_frame` just before
    /// `inbox.submit`. Lets render decompose the sim→render delta
    /// into `bump→submit` (sim post-bump work) and `submit→pickup`
    /// (render-thread cadence). No-op when no bump has happened.
    pub fn record_geometry_submit_now(&self) {
        // Only record when there's actually a bump to attribute the
        // submit to — else the field reads as the wall-time of the
        // last submit of a non-sculpt-mutating frame, which isn't
        // useful for the [sculpt-pipeline] decomposition.
        let bump = self.last_geometry_bump_ns();
        if bump == 0 {
            return;
        }
        let prev_submit = self
            .last_geometry_submit_ns
            .load(std::sync::atomic::Ordering::Relaxed);
        // Don't overwrite — keep the FIRST submit timestamp for a
        // given bump so we measure sim post-bump work, not the
        // sim-tick frequency.
        if prev_submit >= bump {
            return;
        }
        self.last_geometry_submit_ns
            .store(process_elapsed_ns(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Read the submit timestamp recorded by
    /// [`Self::record_geometry_submit_now`]. Returns `0` if no submit
    /// has been recorded for the current bump.
    pub fn last_geometry_submit_ns(&self) -> u64 {
        self.last_geometry_submit_ns
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn clear_faces(&mut self) {
        self.pending_faces.clear();
        self.faces_dirty = true;
    }

    // ── Geometry upload snapshot ─────────────────────────────────────

    pub fn geometry_upload(&self) -> GeometryUpload<'_> {
        // Coalesce + gap-merge each pool's dirty ranges before handing
        // them to the upload code. Plain coalesce was the D9 win
        // (collapse duplicate brick marks); the gap-merge added here
        // is the D10 win — modern wgpu's queue.write_buffer pays
        // ~1 ms of staging-buffer + command-record overhead per call,
        // so a stamp with ~2 000 tiny disjoint writes serialized
        // ~2 s of upload time. Merging ranges within `GAP_BYTES` of
        // each other trades a small over-upload (the gap bytes) for
        // ~10-50× fewer write_buffer calls. 16 KiB gap covers the
        // typical leaf_attr / brick stride density without ballooning
        // total bytes.
        const GAP_BYTES: u32 = 16 * 1024;
        let mut octree_dirty = self.octree.dirty_ranges().clone();
        octree_dirty.coalesce_with_gap(GAP_BYTES);
        let mut leaf_attr_dirty = self.leaf_attr_pool.dirty_attrs().clone();
        leaf_attr_dirty.coalesce_with_gap(GAP_BYTES);
        let mut color_dirty = self.leaf_attr_pool.dirty_colors().clone();
        color_dirty.coalesce_with_gap(GAP_BYTES);
        let mut bone_dirty = self.leaf_attr_pool.dirty_bones().clone();
        bone_dirty.coalesce_with_gap(GAP_BYTES);
        let mut brick_dirty = self.brick_pool.dirty_ranges().clone();
        brick_dirty.coalesce_with_gap(GAP_BYTES);
        GeometryUpload {
            octree_nodes: self.octree.data(),
            octree_internal_attrs: self.octree.internal_attrs_data(),
            leaf_attr_pool: self.leaf_attr_pool.as_bytes(),
            color_pool: self.leaf_attr_pool.color_bytes(),
            bone_weights: self.leaf_attr_pool.bone_bytes(),
            brick_pool: self.brick_pool.as_bytes(),
            brick_face_links: rkp_core::brick_face_links::as_bytes(&self.brick_face_links),
            octree_dirty,
            leaf_attr_dirty,
            color_dirty,
            bone_dirty,
            brick_dirty,
        }
    }

    /// Clear every per-pool dirty range tracker. Called by the render
    /// worker after `RkpScene::upload_geometry` succeeds — the upload
    /// writes only marked bytes, so the trackers can be drained for
    /// the next stamp. Failing to call this would either re-upload the
    /// same bytes on every subsequent frame (waste) or, in the
    /// `should_coalesce_to_full` case, force every frame to the full-
    /// pool fallback.
    pub fn clear_geometry_dirty_ranges(&mut self) {
        self.octree.dirty_ranges_mut().clear();
        self.brick_pool.dirty_ranges_mut().clear();
        self.leaf_attr_pool.dirty_attrs_mut().clear();
        self.leaf_attr_pool.dirty_colors_mut().clear();
        self.leaf_attr_pool.dirty_bones_mut().clear();
    }

    /// Returns a lock-free snapshot of the three pool buffers the
    /// painted-material walk reads. Hold the returned [`WalkSnapshot`]
    /// outside the `scene_mgr` lock to walk the octree without
    /// blocking sim/render.
    ///
    /// **O(1)** — each pool's storage is `Arc<Vec<…>>` so the snapshot
    /// is three `Arc::clone`s. No memcpy, regardless of pool size.
    ///
    /// Snapshot lifetime interacts with future pool mutations via
    /// `Arc::make_mut`: while a snapshot is held, the next mutation
    /// pays a one-time clone-on-write of the affected pool. Drop the
    /// snapshot promptly after the walk so subsequent writes stay
    /// in place.
    ///
    /// Note: the `LeafAttr` Arc shares the pool's *full* backing
    /// `Vec` (length = `pool.capacity()`), not the `[..next_free]`
    /// prefix that `as_slice()` exposes. Walks should therefore
    /// continue to bound their indexing by `next_free` — but since
    /// only `[..next_free]` is reachable from the octree, that
    /// happens naturally; the surplus bytes are unread.
    pub fn walk_snapshot(&self) -> WalkSnapshot {
        WalkSnapshot {
            octree_data: self.octree.data_arc(),
            brick_pool_data: self.brick_pool.data_arc(),
            leaf_attr_data: self.leaf_attr_pool.data_arc(),
            epoch: self.geometry_epoch(),
        }
    }

    // ── Spatial deallocation ─────────────────────────────────────────

    pub fn deallocate_spatial(&mut self, handle: &rkp_core::scene_node::SpatialHandle) {
        self.bump_geometry_epoch();
        if let rkp_core::scene_node::SpatialHandle::Octree {
            root_offset, len, depth, base_voxel_size,
        } = handle
        {
            self.octree.deallocate(OctreeHandle {
                root_offset: *root_offset,
                len: *len,
                depth: *depth,
                base_voxel_size: *base_voxel_size,
            });
        }
    }
}
