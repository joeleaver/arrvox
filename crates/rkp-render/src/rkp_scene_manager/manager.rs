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

    // ── Paint cursor overlay (Phase 3b) ─────────────────────────────
    /// Per-leaf geodesic distance from the paint cursor's world hit,
    /// parallel to [`LeafAttrPool`] slots. `f32::INFINITY` means "not
    /// currently under the brush"; finite values are surface-walking
    /// distances produced by [`crate::paint::surface_flood_fill`]. The
    /// shade pass reads this array to draw the cursor ring — indexing
    /// by the leaf_slot written to `gbuf_leaf_slot`.
    pub(super) brush_overlay_distances: Vec<f32>,
    /// Leaf slots written by the most recent flood fill. Next update
    /// resets each back to `f32::INFINITY` before writing the new fill
    /// — cheap O(previous_fill_size) vs. clearing the whole array.
    pub(super) brush_overlay_flooded_slots: Vec<u32>,
    /// Bumped on every brush-overlay mutation. Separate from
    /// `geometry_epoch` so the render thread can re-upload the small
    /// overlay buffer every time the cursor moves without triggering
    /// a full re-upload of octree / leaf_attr / color buffers.
    pub(super) brush_overlay_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
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
            paint_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            brush_overlay_distances: vec![f32::INFINITY; capacity as usize],
            brush_overlay_flooded_slots: Vec::new(),
            brush_overlay_epoch: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
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
        self.brush_overlay_distances = vec![f32::INFINITY; capacity as usize];
        self.brush_overlay_flooded_slots.clear();
        self.bump_brush_overlay_epoch();
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
    }
    pub fn clear_faces(&mut self) {
        self.pending_faces.clear();
        self.faces_dirty = true;
    }

    // ── Geometry upload snapshot ─────────────────────────────────────

    pub fn geometry_upload(&self) -> GeometryUpload<'_> {
        GeometryUpload {
            octree_nodes: self.octree.data(),
            octree_internal_attrs: self.octree.internal_attrs_data(),
            leaf_attr_pool: self.leaf_attr_pool.as_bytes(),
            color_pool: self.leaf_attr_pool.color_bytes(),
            bone_weights: self.leaf_attr_pool.bone_bytes(),
            brick_pool: self.brick_pool.as_bytes(),
            brick_face_links: rkp_core::brick_face_links::as_bytes(&self.brick_face_links),
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
