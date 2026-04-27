//! Scene management for RKIPatch — owns the leaf_attr pool, octrees, and
//! face instances.
//!
//! This is the CPU-side scene representation. It manages the LeafAttrPool
//! (material + normal + color per leaf), the OctreeGpu allocator, and the
//! face instance list (legacy, unused by the active pipeline).
//!
//! No wgpu types, no GPU buffers here — RkpRenderer consumes the snapshot.

use std::collections::HashMap;
use std::path::PathBuf;

use rkp_core::{BrickPool, LeafAttr, LeafAttrPool, OctreeHandle, SparseOctree};

use crate::octree_gpu::OctreeGpu;
use crate::rkp_scene::GeometryUpload;

/// Face instance for CPU-side face emission (legacy — kept for scene loading
/// compatibility; the splat raster pipeline it fed is not dispatched).
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
/// [`RkpSceneManager::acquire_asset`] and released with
/// [`RkpSceneManager::release_asset`]. Callers must pair acquires with
/// releases — when the last instance drops, the cache deallocates the
/// shared leaf_attr / brick / octree ranges. Not persistable (an index
/// into an in-memory cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle(u32);

impl AssetHandle {
    pub fn raw(self) -> u32 { self.0 }
}

/// Everything a scene instance needs to render an asset. Returned from
/// both `acquire_asset` (.rkp) and the procedural voxelize_* paths so
/// instance spawning can share one code path downstream.
#[derive(Debug, Clone, Copy)]
pub struct AssetInfo {
    pub spatial: rkp_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkp_core::Aabb,
    /// Entity-local grid origin (`aabb_center - extent/2`). Derived at
    /// load time — .rkp files voxelized before this field existed used
    /// the same formula, so re-deriving reproduces the exact bake.
    pub grid_origin: glam::Vec3,
    pub voxel_count: u32,
    pub leaf_attr_slot_start: u32,
    pub leaf_attr_slot_count: u32,
    /// `true` if this asset has skinning data (bone weights + SkinBricks
    /// + rest bone AABBs) baked in. Caller fetches the full data via
    /// [`RkpSceneManager::skinning_data`].
    pub has_skinning: bool,
}

/// One populated octree brick, with its scene-global id and its origin
/// in finest-voxel grid units. Produced at load time by shifting each
/// baked file-local origin's id by the asset's `scene_brick_offset`.
#[derive(Debug, Clone, Copy)]
pub struct SkinBrick {
    /// Scene-global brick id (matches the ids stored in octree nodes).
    pub brick_id: u32,
    /// Brick corner in finest-voxel grid units.
    pub origin: [u32; 3],
}

/// Per-asset skinning metadata read from the `.rkp`'s skin-meta
/// section. Phase-3 scatter pass consumes this to size + populate the
/// deformed bone field each frame.
#[derive(Debug, Clone, Default)]
pub struct SkinningAssetData {
    /// One entry per populated brick in the asset's octree.
    pub bricks: Vec<SkinBrick>,
    /// Per-bone rest-pose AABB, in object-local voxel space. Index is
    /// the bone id (as stored in per-leaf `BoneVoxel.bone_index`).
    /// Empty AABBs (zero-extent) are sentinels for unused bone slots.
    pub rest_bone_aabbs: Vec<[f32; 6]>,
}

/// One entry in the asset cache: the shared geometry allocations plus
/// a refcount. When `refcount` hits zero, `release_asset` frees the
/// octree / leaf_attr / brick ranges.
struct AssetEntry {
    path: PathBuf,
    refcount: u32,
    spatial_handle: OctreeHandle,
    voxel_size: f32,
    aabb: rkp_core::Aabb,
    voxel_count: u32,
    leaf_attr_slot_start: u32,
    leaf_attr_slot_count: u32,
    brick_start: u32,
    brick_count: u32,
    /// Populated only when the asset has a `FLAG_HAS_BONES` skin-meta
    /// section. Phase-3 scatter pass reads this to drive the per-frame
    /// bone-field write.
    skinning: Option<SkinningAssetData>,
}

impl AssetEntry {
    fn info(&self) -> AssetInfo {
        // Reconstruct grid origin the same way voxelize_octree does:
        // `aabb_center - extent/2`. Matches the bake-time geometry, so
        // existing .rkp files render identically.
        let extent = (1u32 << self.spatial_handle.depth) as f32
            * self.spatial_handle.base_voxel_size;
        let aabb_center = (self.aabb.min + self.aabb.max) * 0.5;
        let grid_origin = aabb_center - glam::Vec3::splat(extent * 0.5);
        AssetInfo {
            spatial: rkp_core::scene_node::SpatialHandle::Octree {
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
/// that was resolved against the `.rkp` extension, so two different
/// inputs that normalize to the same file share a handle.
#[derive(Default)]
struct AssetCache {
    entries: Vec<Option<AssetEntry>>,
    path_to_handle: HashMap<PathBuf, AssetHandle>,
    free_slots: Vec<u32>,
}

impl AssetCache {
    fn insert(&mut self, entry: AssetEntry) -> AssetHandle {
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

    fn lookup_path(&self, path: &std::path::Path) -> Option<AssetHandle> {
        self.path_to_handle.get(path).copied()
    }

    fn get(&self, handle: AssetHandle) -> Option<&AssetEntry> {
        self.entries.get(handle.0 as usize).and_then(|e| e.as_ref())
    }

    fn get_mut(&mut self, handle: AssetHandle) -> Option<&mut AssetEntry> {
        self.entries.get_mut(handle.0 as usize).and_then(|e| e.as_mut())
    }

    fn remove(&mut self, handle: AssetHandle) -> Option<AssetEntry> {
        let slot = handle.0 as usize;
        let taken = self.entries.get_mut(slot)?.take()?;
        self.path_to_handle.remove(&taken.path);
        self.free_slots.push(handle.0);
        Some(taken)
    }
}

/// Result of [`RkpSceneManager::reload_asset`]. `old_handle` is the handle
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
    pub spatial: rkp_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkp_core::Aabb,
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
/// splat raster is not dispatched in the active pipeline. Kept for
/// scene-loading compatibility: every leaf is a surface voxel now, so the
/// output just enumerates leaf centers with exposed-face flags.
fn emit_faces(
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
                    Some(node) if node == rkp_core::sparse_octree::EMPTY_NODE => true,
                    Some(node) if node == rkp_core::sparse_octree::INTERIOR_NODE => false,
                    Some(node) if rkp_core::sparse_octree::is_leaf(node) => false,
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
    asset_cache: AssetCache,
    /// Face instances for rasterization (surface shell).
    pending_faces: Vec<FaceInstance>,
    /// Whether face data needs re-upload to GPU.
    faces_dirty: bool,
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
    geometry_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,

    // ── Paint data writes (Phase 3b perf) ───────────────────────────
    /// Separate epoch for paint mutations. Pre-perf: paint bumped
    /// `geometry_epoch`, which made the render thread re-upload every
    /// scene buffer (octree + leaf_attr + color + bricks + face
    /// links) — ~45 MB per stroke stamp on a 1M-leaf scene, at 60 Hz
    /// that's ~2.7 GB/s for a few bytes of actual change. Paint now
    /// bumps this epoch instead, and the render thread only uploads
    /// the dirty slot range of `leaf_attr_pool` + `color_pool`.
    paint_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Inclusive `[min, max]` slot range modified by paint since the
    /// last upload. Accumulates across stamps; reset on upload.
    paint_dirty_range: Option<(u32, u32)>,

    // ── Paint cursor overlay (Phase 3b) ─────────────────────────────
    /// Per-leaf geodesic distance from the paint cursor's world hit,
    /// parallel to [`LeafAttrPool`] slots. `f32::INFINITY` means "not
    /// currently under the brush"; finite values are surface-walking
    /// distances produced by [`crate::paint::surface_flood_fill`]. The
    /// shade pass reads this array to draw the cursor ring — indexing
    /// by the leaf_slot written to `gbuf_leaf_slot`.
    brush_overlay_distances: Vec<f32>,
    /// Leaf slots written by the most recent flood fill. Next update
    /// resets each back to `f32::INFINITY` before writing the new fill
    /// — cheap O(previous_fill_size) vs. clearing the whole array.
    brush_overlay_flooded_slots: Vec<u32>,
    /// Bumped on every brush-overlay mutation. Separate from
    /// `geometry_epoch` so the render thread can re-upload the small
    /// overlay buffer every time the cursor moves without triggering
    /// a full re-upload of octree / leaf_attr / color buffers.
    brush_overlay_epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
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
            paint_dirty_range: None,
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
        self.paint_dirty_range = None;
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
    fn merge_face_links(&mut self, rows: &[[u32; 6]]) {
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

    // ── Asset loading (.rkp files) ───────────────────────────────────

    /// Resolve a user-supplied path (with or without `.rkp` extension)
    /// into a canonical file path we can use as a cache key.
    fn resolve_rkp_path(path: &str) -> Result<PathBuf, String> {
        let rkp_path = if path.ends_with(".rkp") {
            PathBuf::from(path)
        } else {
            let p = std::path::Path::new(path);
            let appended = p.with_file_name(format!(
                "{}.rkp",
                p.file_name().map(|f| f.to_string_lossy()).unwrap_or_default()
            ));
            if appended.exists() {
                appended
            } else {
                let replaced = p.with_extension("rkp");
                if replaced.exists() {
                    replaced
                } else {
                    return Err(format!("no .rkp file found for {path}"));
                }
            }
        };
        if !rkp_path.exists() {
            return Err(format!("{} does not exist", rkp_path.display()));
        }
        rkp_path.canonicalize().map_err(|e| format!("canonicalize {}: {e}", rkp_path.display()))
    }

    /// Acquire a shared asset. First call for a given path allocates the
    /// octree / leaf_attr / brick ranges and caches them. Subsequent calls
    /// return the cached handle and bump its refcount. Every successful
    /// `acquire_asset` must be paired with a `release_asset` when the
    /// instance goes away.
    pub fn acquire_asset(
        &mut self,
        path: &str,
    ) -> Result<(AssetHandle, AssetInfo), String> {
        self.bump_geometry_epoch();
        let canonical = Self::resolve_rkp_path(path)?;

        if let Some(handle) = self.asset_cache.lookup_path(&canonical) {
            let entry = self.asset_cache.get_mut(handle).expect("cache/handle mismatch");
            entry.refcount += 1;
            return Ok((handle, entry.info()));
        }

        let entry = self.load_asset_from_disk(&canonical)?;
        let info = entry.info();
        let handle = self.asset_cache.insert(entry);
        Ok((handle, info))
    }

    /// Force a reload of a cached asset from disk. Used after re-import
    /// rewrites the `.rkp` file so existing scene instances pick up the
    /// new geometry. Frees the previous pool allocations, loads the fresh
    /// file, and preserves the refcount so outstanding instances remain
    /// valid once they've been updated to the returned handle.
    ///
    /// Returns `Ok(None)` when the asset isn't currently cached (nothing
    /// to refresh — the next `acquire_asset` will read the new file).
    pub fn reload_asset(&mut self, path: &str) -> Result<Option<ReloadResult>, String> {
        self.bump_geometry_epoch();
        let canonical = Self::resolve_rkp_path(path)?;
        let Some(old_handle) = self.asset_cache.lookup_path(&canonical) else {
            return Ok(None);
        };

        let old_refcount = self.asset_cache.get(old_handle)
            .map(|e| e.refcount).unwrap_or(0);

        let entry = self.asset_cache.remove(old_handle).expect("just looked up");
        self.octree.deallocate(entry.spatial_handle);
        self.leaf_attr_pool.deallocate_range(entry.leaf_attr_slot_start, entry.leaf_attr_slot_count);
        for id in entry.brick_start..(entry.brick_start + entry.brick_count) {
            self.brick_pool.deallocate(id);
        }

        let mut fresh = self.load_asset_from_disk(&canonical)?;
        fresh.refcount = old_refcount;
        let info = fresh.info();
        let new_handle = self.asset_cache.insert(fresh);
        Ok(Some(ReloadResult { old_handle, new_handle, info }))
    }

    /// Release an instance's claim on a cached asset. When the last
    /// outstanding reference drops, we deallocate the shared ranges from
    /// the scene pools.
    pub fn release_asset(&mut self, handle: AssetHandle) {
        self.bump_geometry_epoch();
        let Some(entry) = self.asset_cache.get_mut(handle) else { return; };
        if entry.refcount == 0 { return; }
        entry.refcount -= 1;
        if entry.refcount > 0 { return; }

        // Last reference — free the pool ranges and drop the cache slot.
        let entry = self.asset_cache.remove(handle).expect("just looked up");
        self.octree.deallocate(entry.spatial_handle);
        self.leaf_attr_pool.deallocate_range(entry.leaf_attr_slot_start, entry.leaf_attr_slot_count);
        for id in entry.brick_start..(entry.brick_start + entry.brick_count) {
            self.brick_pool.deallocate(id);
        }
    }

    /// Disk read + pool allocation for one .rkp file. Called exactly once
    /// per unique path — repeated acquisitions share the returned entry
    /// via the cache.
    fn load_asset_from_disk(&mut self, rkp_path: &std::path::Path) -> Result<AssetEntry, String> {
        use rkp_core::voxel::VoxelSample;

        let mut file = std::fs::File::open(rkp_path)
            .map_err(|e| format!("open {}: {e}", rkp_path.display()))?;
        let mut reader = std::io::BufReader::new(&mut file);

        let header = rkp_core::asset_file::read_rkp_header(&mut reader)
            .map_err(|e| format!("read .rkp header: {e}"))?;

        let octree_nodes = rkp_core::asset_file::read_rkp_octree(&mut reader, &header)
            .map_err(|e| format!("read octree: {e}"))?;

        let voxel_data = rkp_core::asset_file::read_rkp_voxels(&mut reader, &header)
            .map_err(|e| format!("read voxels: {e}"))?;

        let voxel_size = header.base_voxel_size;
        let voxel_count = header.voxel_count;
        let aabb = rkp_core::Aabb::new(
            glam::Vec3::from(header.aabb_min),
            glam::Vec3::from(header.aabb_max),
        );

        // Pre-baked octahedrally-packed normals per slot. One u32 per shell
        // voxel, written at import time from the mesh SDF gradient — the
        // runtime never sees an SDF.
        let has_normals = header.flags & rkp_core::asset_file::FLAG_HAS_NORMALS != 0;
        let normals_bytes = if has_normals {
            rkp_core::asset_file::read_rkp_normals(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let normals_u32s: &[u32] = if normals_bytes.len() >= 4 {
            bytemuck::cast_slice(&normals_bytes)
        } else {
            &[]
        };

        // Brick-terminated octree (v4). Each brick is a flat run of
        // BRICK_CELLS u32s; cell value is either BRICK_EMPTY or a slot
        // index into the parallel voxel arrays.
        let has_bricks = header.flags & rkp_core::asset_file::FLAG_HAS_BRICKS != 0;
        let bricks_bytes = if has_bricks {
            rkp_core::asset_file::read_rkp_bricks(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let file_brick_cells: &[u32] = if !bricks_bytes.is_empty() {
            bytemuck::cast_slice(&bricks_bytes)
        } else {
            &[]
        };

        let has_color = header.flags & rkp_core::asset_file::FLAG_HAS_COLOR != 0;
        let color_bytes = if has_color {
            rkp_core::asset_file::read_rkp_color(&mut reader, &header).unwrap_or_default()
        } else {
            Vec::new()
        };
        let color_u32s: &[u32] = if color_bytes.len() >= 4 {
            bytemuck::cast_slice(&color_bytes)
        } else {
            &[]
        };

        // Skin-meta section — structured payload carrying per-leaf bone
        // weights, per-brick origins, and per-bone rest AABBs. Only
        // present when rkp-import resolved a skinned skeleton.
        let has_bones = header.flags & rkp_core::asset_file::FLAG_HAS_BONES != 0;
        let skin_meta = if has_bones {
            match rkp_core::asset_file::read_rkp_skin_meta(&mut reader, &header) {
                Ok(m) => {
                    eprintln!(
                        "[RkpSceneManager] {}: skin-meta loaded ({} bone voxels, {} bricks, {} bone AABBs)",
                        rkp_path.display(),
                        m.bone_voxels.len() / 8,
                        m.brick_origins.len(),
                        m.rest_bone_aabbs.len(),
                    );
                    m
                }
                Err(e) => {
                    // Old Phase-2 file format wrote the bones section
                    // as a raw `BoneVoxel` array; the new structured
                    // blob fails to decode that. Warn loudly so a
                    // stale `.rkp` on disk doesn't silently mask the
                    // whole skinning pipeline as "nothing broken, no
                    // deformation".
                    eprintln!(
                        "[RkpSceneManager] {}: FLAG_HAS_BONES set but skin-meta decode failed ({e}). \
                         Re-import the asset to write the new wire format.",
                        rkp_path.display(),
                    );
                    rkp_core::asset_file::SkinMetaOut::default()
                }
            }
        } else {
            rkp_core::asset_file::SkinMetaOut::default()
        };
        let file_bones: &[rkp_core::companion::BoneVoxel] = if skin_meta.bone_voxels.len() >= std::mem::size_of::<rkp_core::companion::BoneVoxel>() {
            bytemuck::cast_slice(&skin_meta.bone_voxels)
        } else {
            &[]
        };

        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        // `Option<u32>` for normal so we distinguish "file has no normals"
        // (stays None → leaf_attr keeps its default) from "file has a
        // normal that happens to oct-pack to 0" (which is the legitimate
        // +Z direction; previously the load path skipped that override
        // because it used `if normal_oct != 0`, corrupting every voxel
        // whose baked normal pointed +Z — manifested as one face of a
        // cube rendering with wrong refraction after save/reload, fixed
        // only by re-baking).
        let mut file_voxel_mat: Vec<(u16, u16, u8, u32, Option<u32>)> = Vec::with_capacity(voxel_count as usize);
        for i in 0..voxel_count as usize {
            let src_offset = i * bytes_per_voxel;
            if src_offset + bytes_per_voxel > voxel_data.len() {
                break;
            }
            let vs: &VoxelSample =
                bytemuck::from_bytes(&voxel_data[src_offset..src_offset + bytes_per_voxel]);
            let color = color_u32s.get(i).copied().unwrap_or(0);
            let normal_oct = if has_normals {
                normals_u32s.get(i).copied()
            } else {
                None
            };
            file_voxel_mat.push((
                vs.material_id(), vs.secondary_material_id(), vs.blend_weight(), color, normal_oct,
            ));
        }

        let octree_depth = header.octree_depth as u8;
        let mut tree = SparseOctree::from_raw(&octree_nodes, octree_depth, voxel_size);

        // 1:1 leaf_attr allocation. We don't dedup file slots → leaf_attrs
        // because texture-sampled colors vary per voxel (measured dedup
        // ratio ≈1.0× on mesh imports — HashMap overhead costs more than
        // the trivial savings). Each file slot maps directly to
        // `leaf_attr_slot_start + file_slot`.
        let leaf_attr_slot_count = voxel_count;
        let leaf_attr_slot_start = self.leaf_attr_pool
            .allocate_contiguous_bump(leaf_attr_slot_count)
            .expect("leaf_attr_pool.allocate_contiguous_bump failed");

        for (i, &(mat_p, mat_s, blend, color, normal_oct)) in file_voxel_mat.iter().enumerate() {
            let mut attr = LeafAttr::new_blended(glam::Vec3::Y, mat_p, mat_s, blend);
            if let Some(n) = normal_oct {
                attr.normal_oct = n;
            }
            let slot = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(slot) = attr;
            if color != 0 {
                self.leaf_attr_pool.set_color(slot, color);
            }
            // File-local bone slot i → scene-global leaf_attr slot. The
            // `file_bones` slice is empty for unskinned assets, in which
            // case the pool's zero-default BoneVoxel stands.
            if let Some(&bv) = file_bones.get(i) {
                self.leaf_attr_pool.set_bone(slot, bv);
            }
        }

        // v4: copy file brick pool into the scene brick pool. Each file
        // cell holds a file-local slot index; we shift both brick-ids
        // (in the octree nodes) and slot indices (in the cells) by our
        // contiguous allocation offsets.
        let file_brick_count = (file_brick_cells.len() / rkp_core::brick_pool::BRICK_CELLS as usize) as u32;
        let scene_brick_offset = self.brick_pool
            .allocate_contiguous_bump(file_brick_count)
            .expect("brick_pool.allocate_contiguous_bump failed");

        // Remap BRICK node ids in the flat nodes array.
        let nodes = tree.as_slice_mut();
        for n in nodes.iter_mut() {
            if rkp_core::sparse_octree::is_brick(*n) {
                let file_id = rkp_core::sparse_octree::brick_id(*n);
                *n = rkp_core::sparse_octree::make_brick(scene_brick_offset + file_id);
            }
        }

        // Actual surface-cell count across this asset. `header.voxel_count`
        // only counts unique LeafAttr slots (one per unique normal +
        // material + blend + color tuple after bake-time dedup), which
        // badly understates the painted surface on flat-faced primitives
        // — a 20×1×20 procedural cube has ~2.3M cells but ~100 unique
        // attrs, so the header number reads as "126 voxels" after
        // Convert-to-Voxel even though the geometry is fully intact.
        // Count non-sentinel brick cells here (+ LEAF octree nodes
        // below) and report that instead.
        let mut actual_cell_count: u32 = 0;
        let brick_cells = rkp_core::brick_pool::BRICK_CELLS as usize;
        for file_id in 0..file_brick_count {
            let scene_id = scene_brick_offset + file_id;
            let src = &file_brick_cells[
                file_id as usize * brick_cells..(file_id as usize + 1) * brick_cells
            ];
            for (i, &cell) in src.iter().enumerate() {
                if cell == rkp_core::brick_pool::BRICK_EMPTY {
                    continue;
                }
                // BRICK_INTERIOR is a scene-global sentinel (0xFFFFFFFD),
                // not a file-local slot index — pass it through without
                // the leaf_attr_slot_start offset, which would overflow
                // and corrupt the slot into a bogus leaf_attr_id. Also
                // skip it from the user-facing voxel count: interior
                // sentinels mark "inside the solid" and never render /
                // paint as voxels.
                let remapped = if cell == rkp_core::brick_pool::BRICK_INTERIOR {
                    rkp_core::brick_pool::BRICK_INTERIOR
                } else {
                    // Real leaf: cell is a file-local slot index; shift
                    // by our leaf_attr allocation offset to get the
                    // scene-global leaf_attr_id.
                    actual_cell_count += 1;
                    leaf_attr_slot_start + cell
                };
                let x = (i as u32) % rkp_core::brick_pool::BRICK_DIM;
                let y = ((i as u32) / rkp_core::brick_pool::BRICK_DIM) % rkp_core::brick_pool::BRICK_DIM;
                let z = (i as u32) / (rkp_core::brick_pool::BRICK_DIM * rkp_core::brick_pool::BRICK_DIM);
                self.brick_pool.set_cell(scene_id, x, y, z, remapped);
            }
        }
        // Shallow trees (depth ≤ BRICK_LEVELS) skip the brick path and
        // emit LEAF nodes at `max_depth` instead — count those too.
        for &n in &octree_nodes {
            if rkp_core::sparse_octree::is_leaf(n) {
                actual_cell_count += 1;
            }
        }

        if !has_bricks {
            return Err(format!(
                "{}: v4 format requires a bricks section (FLAG_HAS_BRICKS); older files are not supported",
                rkp_path.display(),
            ));
        }

        let raw_count = tree.node_count();
        tree.collapse_all();
        tree.compact();
        let compact_count = tree.node_count();
        tree.deduplicate_subtrees();
        let dedup_count = tree.node_count();
        tree.morton_reorder();

        // Note: Laplacian shell-normal smoothing used to run here.
        // It's now performed at asset-bake time inside `rkp-import`'s
        // `smooth_normals` stage so each asset pays the cost once
        // instead of on every load. Older `.rkp` files written before
        // that change will have un-smoothed SDF-gradient normals
        // (noisier but still correct); re-import to pick up the
        // pre-smoothed variant.

        // Run the prefilter pass on-load so v4 assets (no baked internal
        // attrs) still benefit from the GPU's LOD early-exit. Phase 4
        // bumps the .rkp format to v5 which bakes these at conversion
        // time — this is the fallback until then.
        //
        // The prefilter appends new attrs at the tail of the asset's
        // contiguous leaf_attr range via allocate_contiguous_bump(1), so
        // the `leaf_attr_slot_count` grows to cover them and the
        // existing deallocate_range releases everything on asset drop.
        rkp_core::prefilter::prefilter_octree_internals(
            &mut tree,
            &mut self.leaf_attr_pool,
            &self.brick_pool,
        );
        let final_leaf_attr_slot_count =
            self.leaf_attr_pool.allocated_count() - leaf_attr_slot_start;

        // Compute brick face-links for this asset. The tree's brick ids
        // have already been remapped to global ids above, so the rows
        // produced are scene-global and ready to merge. When the file
        // had zero bricks there's nothing to compute.
        if file_brick_count > 0 {
            let max_brick = scene_brick_offset + file_brick_count - 1;
            let face_links = rkp_core::brick_face_links::compute_brick_face_links(&tree, max_brick);
            self.merge_face_links(&face_links);
        }

        // Allocate the octree with its now-populated internal_attr_index
        // intact. `allocate(&tree)` preserves both buffers; the legacy
        // `allocate_raw(nodes, …)` would have dropped the prefilter ids
        // by round-tripping through `SparseOctree::from_raw`.
        let handle = self.octree.allocate(&tree);

        eprintln!(
            "[RkpSceneManager] loaded {}: {} voxels ({} unique attrs), {} bricks, octree {} → compact {} → dedup {} ({:.1}× total), +{} prefilter attrs",
            rkp_path.display(),
            actual_cell_count,
            voxel_count,
            file_brick_count,
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
            final_leaf_attr_slot_count - leaf_attr_slot_count,
        );

        // Promote the baked skin-meta (file-local brick ids) into
        // scene-global SkinBrick entries. Rest bone AABBs are already
        // in object-local voxel space — no transform needed.
        let skinning = if has_bones {
            let bricks: Vec<SkinBrick> = skin_meta.brick_origins.iter().enumerate()
                .map(|(file_id, &origin)| SkinBrick {
                    brick_id: scene_brick_offset + file_id as u32,
                    origin,
                })
                .collect();
            Some(SkinningAssetData {
                bricks,
                rest_bone_aabbs: skin_meta.rest_bone_aabbs,
            })
        } else {
            None
        };

        Ok(AssetEntry {
            path: rkp_path.to_path_buf(),
            refcount: 1,
            spatial_handle: handle,
            voxel_size,
            aabb,
            voxel_count: actual_cell_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: final_leaf_attr_slot_count,
            brick_start: scene_brick_offset,
            brick_count: file_brick_count,
            skinning,
        })
    }

    /// Peek at an asset's skinning metadata. Returns `None` when the
    /// asset was imported without bone weights.
    pub fn skinning_data(&self, handle: AssetHandle) -> Option<&SkinningAssetData> {
        self.asset_cache.get(handle)?.skinning.as_ref()
    }

    // ── Paint data epoch + dirty range ─────────────────────────────

    /// Current paint-data epoch (lock-free read). Bumped by
    /// `apply_paint_sphere` when leaf_attr / color writes happen.
    /// Render polls this to decide whether to slice-upload the
    /// dirty range.
    pub fn paint_epoch(&self) -> u64 {
        self.paint_epoch.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Clone of the paint-epoch atomic for lock-free sim-side reads.
    pub fn paint_epoch_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.paint_epoch.clone()
    }

    fn bump_paint_epoch(&mut self) {
        self.paint_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    fn mark_paint_range_dirty(&mut self, slot: u32) {
        self.paint_dirty_range = Some(match self.paint_dirty_range {
            Some((a, b)) => (a.min(slot), b.max(slot)),
            None => (slot, slot),
        });
    }

    /// Take the current accumulated dirty range (if any) — render-side
    /// callers use this to decide what byte ranges of `leaf_attr_pool`
    /// and `color_pool` to slice-upload. Clears the range; the next
    /// paint stamp re-accumulates from empty.
    pub fn take_paint_dirty_range(&mut self) -> Option<(u32, u32)> {
        self.paint_dirty_range.take()
    }

    /// Return a byte slice of `leaf_attr_pool` covering slots
    /// `[slot_start, slot_start + slot_count)`. Used by the render
    /// thread to `queue.write_buffer` only the dirty range instead of
    /// the whole buffer.
    pub fn leaf_attr_slice_bytes(&self, slot_start: u32, slot_count: u32) -> &[u8] {
        let bytes_per = std::mem::size_of::<LeafAttr>();
        let start = slot_start as usize * bytes_per;
        let end = (slot_start as usize + slot_count as usize) * bytes_per;
        let full = self.leaf_attr_pool.as_bytes();
        if end > full.len() {
            return &[];
        }
        &full[start..end]
    }

    /// Byte slice of the color pool covering slots
    /// `[slot_start, slot_start + slot_count)`. Sibling of
    /// `leaf_attr_slice_bytes` for the color companion buffer.
    pub fn color_slice_bytes(&self, slot_start: u32, slot_count: u32) -> &[u8] {
        let bytes_per = 4; // u32 per slot
        let start = slot_start as usize * bytes_per;
        let end = (slot_start as usize + slot_count as usize) * bytes_per;
        let full = self.leaf_attr_pool.color_bytes();
        if end > full.len() {
            return &[];
        }
        &full[start..end]
    }


    // ── Paint cursor overlay ────────────────────────────────────────

    /// Current brush-overlay epoch (lock-free read).
    pub fn brush_overlay_epoch(&self) -> u64 {
        self.brush_overlay_epoch
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Clone the brush-overlay epoch atomic so render / sim can poll
    /// it without taking the scene_mgr Mutex.
    pub fn brush_overlay_epoch_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.brush_overlay_epoch.clone()
    }

    fn bump_brush_overlay_epoch(&mut self) {
        self.brush_overlay_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Brush-overlay distance array bytes for GPU upload. Parallel to
    /// `leaf_attr_pool` — grown to match the pool's allocated size.
    /// Sentinel: `f32::INFINITY` means "not under the current brush".
    pub fn brush_overlay_bytes(&self) -> &[u8] {
        let n = self.leaf_attr_pool.allocated_count() as usize;
        if n == 0 {
            return &[];
        }
        let len = n.min(self.brush_overlay_distances.len());
        bytemuck::cast_slice(&self.brush_overlay_distances[..len])
    }

    /// Drop any currently-active brush overlay. Cheap — only touches
    /// the slots flooded by the last `update_brush_overlay` call.
    pub fn clear_brush_overlay(&mut self) {
        if self.brush_overlay_flooded_slots.is_empty() {
            return;
        }
        for &slot in &self.brush_overlay_flooded_slots {
            if let Some(d) = self.brush_overlay_distances.get_mut(slot as usize) {
                *d = f32::INFINITY;
            }
        }
        self.brush_overlay_flooded_slots.clear();
        self.bump_brush_overlay_epoch();
    }

    /// Run a geodesic surface flood fill from `brush_center_world` on
    /// the target asset and write per-leaf distances into the overlay
    /// array. Clears the previous fill first (so the brush doesn't
    /// "smear" across frames). No-op when the entity's spatial handle
    /// isn't an octree (procedurals — they don't own leaf slots and
    /// thus have no per-voxel cursor).
    pub fn update_brush_overlay(
        &mut self,
        asset: &AssetInfo,
        entity_world: glam::Affine3A,
        brush_center_world: glam::Vec3,
        radius: f32,
    ) {
        use rkp_core::scene_node::SpatialHandle;
        // Start by dropping the previous fill — even if this new call
        // produces no hits we want the cursor to vanish, not linger.
        self.clear_brush_overlay();

        if radius <= 0.0 {
            return;
        }
        let SpatialHandle::Octree { root_offset, depth, base_voxel_size, .. } = asset.spatial
        else {
            return;
        };

        let inv_world = entity_world.inverse();
        let center_local = inv_world.transform_point3(brush_center_world);
        let (scale, _, _) = entity_world.to_scale_rotation_translation();
        let mean_scale = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
        let local_radius = radius / mean_scale.max(1e-6);

        let hits = crate::paint::surface_flood_fill(
            self.octree.data(),
            root_offset,
            depth,
            base_voxel_size,
            &self.brick_pool,
            &self.brick_face_links,
            asset.grid_origin,
            center_local,
            local_radius,
        );

        if hits.is_empty() {
            return;
        }

        // Grow the distance array if the leaf_attr_pool has outgrown it.
        let pool_len = self.leaf_attr_pool.capacity() as usize;
        if self.brush_overlay_distances.len() < pool_len {
            self.brush_overlay_distances.resize(pool_len, f32::INFINITY);
        }

        let slot_lo = asset.leaf_attr_slot_start;
        let slot_hi = slot_lo + asset.leaf_attr_slot_count;
        self.brush_overlay_flooded_slots.reserve(hits.len());
        for hit in &hits {
            if hit.leaf_slot < slot_lo || hit.leaf_slot >= slot_hi {
                continue;
            }
            let idx = hit.leaf_slot as usize;
            if idx < self.brush_overlay_distances.len() {
                // In world-scale units — the flood fill converts
                // object-local cell_size back up into world distance
                // implicitly because the mean-scale adjustment above
                // gave it a local_radius in local units. To keep the
                // shader's radius comparison in world units, remap
                // back up by the mean scale.
                self.brush_overlay_distances[idx] = hit.distance * mean_scale;
                self.brush_overlay_flooded_slots.push(hit.leaf_slot);
            }
        }
        self.bump_brush_overlay_epoch();
    }

    // ── Paint orchestrator ───────────────────────────────────────────

    /// Apply one brush stamp to every leaf whose voxel center sits inside a
    /// world-space sphere centered at `brush_center_world`.
    ///
    /// `asset` describes the entity's geometry allocation (octree +
    /// leaf_attr range). `entity_world` is the entity's full world
    /// transform (from its Transform component); paint transforms the
    /// brush center into the entity's local frame before querying the
    /// octree, which keeps the brush aligned with rotated / scaled
    /// objects.
    ///
    /// Returns the number of leaves that received a write. The caller
    /// (editor command handler) is responsible for:
    ///
    /// * Skipping procedural / generator-owned entities (paint would be
    ///   wiped on the next rebake).
    /// * Converting its editor-side `PaintMode` into the right
    ///   [`PaintStamp`] variant.
    /// * Ensuring this is called on the sim / engine thread that owns
    ///   the scene_mgr Mutex — paint is a geometry mutation.
    pub fn apply_paint_sphere(
        &mut self,
        asset: &AssetInfo,
        entity_world: glam::Affine3A,
        brush_center_world: glam::Vec3,
        radius: f32,
        strength: f32,
        falloff: f32,
        stamp: crate::paint::PaintStamp,
    ) -> usize {
        use rkp_core::scene_node::SpatialHandle;
        if radius <= 0.0 || strength <= 0.0 {
            return 0;
        }

        let SpatialHandle::Octree { root_offset, depth, base_voxel_size, .. } =
            asset.spatial
        else {
            // Non-octree spatial handles don't have leaf_attr data to paint.
            return 0;
        };

        // World → object-local. Use the affine inverse so non-uniform
        // scale still works correctly. The brush radius must be scaled
        // inversely too — otherwise scaling an object up would shrink
        // the effective paint footprint.
        let inv_world = entity_world.inverse();
        let center_local = inv_world.transform_point3(brush_center_world);
        // Radius in object-local units — average the three scale axes.
        // Exact local radius is the world radius divided by the
        // directional scale; painting is approximate enough that the
        // mean is a good default.
        let (scale, _, _) = entity_world.to_scale_rotation_translation();
        let mean_scale = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
        let local_radius = radius / mean_scale.max(1e-6);

        let hits = crate::paint::leaves_in_sphere(
            self.octree.data(),
            root_offset,
            depth,
            base_voxel_size,
            &self.brick_pool,
            asset.grid_origin,
            center_local,
            local_radius,
        );

        if hits.is_empty() {
            return 0;
        }

        // Validate slots against this asset's allocated range — the
        // packed buffer's leaves carry scene-global slot ids, so a
        // corrupted octree could in theory produce ids outside the
        // asset's range. Clamp defensively.
        let slot_lo = asset.leaf_attr_slot_start;
        let slot_hi = slot_lo + asset.leaf_attr_slot_count;

        let mut written: usize = 0;
        let mut dirty_min = u32::MAX;
        let mut dirty_max = 0u32;
        for hit in &hits {
            if hit.leaf_slot < slot_lo || hit.leaf_slot >= slot_hi {
                continue;
            }
            let weight = crate::paint::brush_weight(
                hit.distance, local_radius, strength, falloff,
            );
            if weight <= 0.0 {
                continue;
            }
            let paint_slot = hit.leaf_slot;
            match stamp {
                crate::paint::PaintStamp::Material { material_id } => {
                    crate::paint::paint_leaf_material(
                        &mut self.leaf_attr_pool, paint_slot, material_id, weight,
                    );
                }
                crate::paint::PaintStamp::Color { rgb } => {
                    crate::paint::paint_leaf_color(
                        &mut self.leaf_attr_pool, paint_slot, rgb, weight,
                    );
                }
                crate::paint::PaintStamp::Erase => {
                    crate::paint::erase_leaf_color(
                        &mut self.leaf_attr_pool, paint_slot, weight,
                    );
                }
            }
            dirty_min = dirty_min.min(paint_slot);
            dirty_max = dirty_max.max(paint_slot);
            written += 1;
        }

        if written > 0 {
            // Paint mutations are small byte-level edits — don't
            // bump `geometry_epoch` (that would trigger a full
            // re-upload of octree + bricks + face_links too).
            // Record the dirty slot range and bump the paint-only
            // epoch instead; the render thread slice-uploads just
            // the changed leaf_attr + color bytes.
            self.mark_paint_range_dirty(dirty_min);
            self.mark_paint_range_dirty(dirty_max);
            self.bump_paint_epoch();
        }
        written
    }

    // ── Primitive voxelization ───────────────────────────────────────

    /// Voxelize an SDF primitive into the octree.
    pub fn voxelize_primitive(
        &mut self,
        primitive: &rkp_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        object_id: u32,
    ) -> Option<VoxelizeResult> {
        self.bump_geometry_epoch();
        use rkp_core::scene_node::SdfPrimitive;

        fn primitive_half_extents(prim: &SdfPrimitive) -> glam::Vec3 {
            match *prim {
                SdfPrimitive::Sphere { radius } => glam::Vec3::splat(radius),
                SdfPrimitive::Box { half_extents } => half_extents,
                SdfPrimitive::Capsule { radius, half_height } => {
                    glam::Vec3::new(radius, half_height + radius, radius)
                }
                SdfPrimitive::Torus { major_radius, minor_radius } => {
                    let r = major_radius + minor_radius;
                    glam::Vec3::new(r, minor_radius, r)
                }
                SdfPrimitive::Cylinder { radius, half_height } => {
                    glam::Vec3::new(radius, half_height, radius)
                }
                SdfPrimitive::Plane { .. } => glam::Vec3::splat(1.0),
            }
        }

        let half_extents = primitive_half_extents(primitive) * bake_scale;
        let margin = voxel_size * 8.0 * 1.8 + voxel_size;
        let aabb = rkp_core::Aabb::new(
            -half_extents - glam::Vec3::splat(margin),
            half_extents + glam::Vec3::splat(margin),
        );

        // SDF closure passed directly to the voxelizer. Negative = inside.
        let sdf_fn: Box<dyn Fn(glam::Vec3) -> f32> = match primitive {
            SdfPrimitive::Box { half_extents: he } => {
                let scaled = SdfPrimitive::Box { half_extents: *he * bake_scale };
                Box::new(move |pos| rkp_core::evaluate_primitive(&scaled, pos))
            }
            _ => {
                let prim = primitive.clone();
                let min_scale = bake_scale.x.min(bake_scale.y).min(bake_scale.z).max(1e-6);
                let inv_scale = glam::Vec3::new(
                    1.0 / bake_scale.x.max(1e-6),
                    1.0 / bake_scale.y.max(1e-6),
                    1.0 / bake_scale.z.max(1e-6),
                );
                Box::new(move |pos| rkp_core::evaluate_primitive(&prim, pos * inv_scale) * min_scale)
            }
        };

        // Batched callback: primitive SDF is CPU-only, so just loop.
        // `voxelize_octree`'s BFS hands us one call per level plus one
        // per terminal-geometry phase — the extra Vec allocations are
        // negligible next to the primitive evaluation cost.
        let sdf_batch = |positions: &[glam::Vec3]| -> Vec<(f32, u16, u16, u8, u32)> {
            // Single-material import path — secondary/blend left at 0,
            // so the shader's dual-material guard short-circuits.
            // Color = 0 = "no override, use material base color".
            positions
                .iter()
                .map(|p| (sdf_fn(*p), material_id, 0u16, 0u8, 0u32))
                .collect()
        };

        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_batch, &aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        let geometry_aabb = rkp_core::Aabb::new(-half_extents, half_extents);
        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: geometry_aabb,
            grid_origin: r.grid_origin,
            voxel_count: r.voxel_count,
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
            brick_ids: r.brick_ids,
        })
    }

    /// Voxelize an arbitrary SDF function into the octree.
    ///
    /// The closure takes a batch of positions and returns a parallel
    /// vec of `(signed_distance, primary_material, secondary_material,
    /// blend_weight_u4)` — one entry per input. Negative distance =
    /// inside. Pass `(secondary = 0, blend = 0)` for single-material
    /// voxelization; the shader's dual-material lerp is guarded behind
    /// `blend_weight > 0` so zero-blend voxels render identically to
    /// the old single-material path. The batched shape lets GPU-
    /// backed evaluators dispatch one compute pass per octree level.
    pub fn voxelize_sdf_fn<F>(
        &mut self,
        sdf_fn: F,
        aabb: &rkp_core::Aabb,
        voxel_size: f32,
        object_id: u32,
    ) -> Option<VoxelizeResult>
    where
        F: FnMut(&[glam::Vec3]) -> Vec<(f32, u16, u16, u8, u32)>,
    {
        self.bump_geometry_epoch();
        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_fn, aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: *aabb,
            grid_origin: r.grid_origin,
            voxel_count: r.voxel_count,
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
            brick_ids: r.brick_ids,
        })
    }

    /// Integrate a self-contained [`rkp_core::BakeArtifact`] (produced
    /// by `voxelize_to_artifact` on a worker thread against fresh
    /// private pools) into the shared scene pools. Remaps all
    /// worker-local leaf_attr IDs and brick IDs into the scene's global
    /// IDs, then runs the same tail that `voxelize_sdf_fn` does:
    /// `emit_faces`, `merge_face_links`, `octree.allocate`.
    pub fn integrate_artifact(
        &mut self,
        mut artifact: rkp_core::BakeArtifact,
        aabb: &rkp_core::Aabb,
        voxel_size: f32,
    ) -> Option<VoxelizeResult> {
        self.bump_geometry_epoch();
        use rkp_core::brick_face_links::{FACE_EMPTY, FACE_INTERIOR};
        use rkp_core::brick_pool::{BRICK_EMPTY, BRICK_INTERIOR};
        use rkp_core::sparse_octree::{
            brick_id as node_brick_id, is_brick, is_leaf, leaf_slot as node_leaf_slot,
            make_brick, make_leaf, INTERNAL_ATTR_NONE,
        };
        let t_start = std::time::Instant::now();

        // ── Leaf-attr pool: allocate a contiguous range, copy ──
        let n_attrs = artifact.leaf_attrs.len() as u32;
        let leaf_attr_slot_start = self
            .leaf_attr_pool
            .allocate_contiguous_bump(n_attrs)?;
        let t_attr_alloc = t_start.elapsed();
        for (i, attr) in artifact.leaf_attrs.iter().enumerate() {
            let scene_id = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(scene_id) = *attr;
            let color = artifact.leaf_attr_colors[i];
            if color != 0 {
                self.leaf_attr_pool.set_color(scene_id, color);
            }
        }

        let t_attr_copy = t_start.elapsed();
        // ── Brick pool: allocate scene IDs, copy cells with leaf remap ──
        let n_bricks = artifact.brick_cells.len();
        let mut worker_to_scene_brick: Vec<u32> = Vec::with_capacity(n_bricks);
        let mut brick_ids_scene: Vec<u32> = Vec::with_capacity(n_bricks);
        let mut max_scene_brick: u32 = 0;
        for cells in &artifact.brick_cells {
            let scene_id = self.brick_pool.allocate()?;
            worker_to_scene_brick.push(scene_id);
            brick_ids_scene.push(scene_id);
            if scene_id > max_scene_brick {
                max_scene_brick = scene_id;
            }
            // Bulk-copy the cell slice, adding `leaf_attr_slot_start`
            // to every non-sentinel entry. A flat slice walk beats
            // 64 `set_cell` calls per brick — at millions of bricks
            // the overhead per cell dominates.
            let dst = self.brick_pool.brick_cells_mut(scene_id);
            debug_assert_eq!(dst.len(), cells.len());
            for (d, &c) in dst.iter_mut().zip(cells.iter()) {
                *d = if c == BRICK_EMPTY || c == BRICK_INTERIOR {
                    c
                } else {
                    leaf_attr_slot_start + c
                };
            }
        }

        let t_brick_copy = t_start.elapsed();
        // ── Octree node slice: remap leaf slots + brick IDs ──
        {
            let nodes = artifact.octree.as_slice_mut();
            for node in nodes.iter_mut() {
                let v = *node;
                if is_leaf(v) {
                    let worker_slot = node_leaf_slot(v);
                    *node = make_leaf(leaf_attr_slot_start + worker_slot);
                } else if is_brick(v) {
                    let worker_id = node_brick_id(v);
                    *node = make_brick(worker_to_scene_brick[worker_id as usize]);
                }
                // EMPTY_NODE / INTERIOR_NODE / branch pointers pass through.
            }
        }

        // ── Prefiltered internal attrs: remap parallel to nodes ──
        {
            let old = artifact.octree.internal_attr_slice().to_vec();
            let new: Vec<u32> = old
                .into_iter()
                .map(|v| if v == INTERNAL_ATTR_NONE { v } else { leaf_attr_slot_start + v })
                .collect();
            artifact.octree.set_internal_attr_index(new);
        }

        let t_octree_remap = t_start.elapsed();
        // ── Face links: remap indices + values into scene brick space ──
        // The scene-wide table is indexed by scene brick_id, so we
        // place each worker row at its remapped slot and pad the rest.
        if n_bricks > 0 {
            let mut scene_rows: Vec<[u32; 6]> =
                vec![[FACE_EMPTY; 6]; (max_scene_brick + 1) as usize];
            for (worker_id, row) in artifact.brick_face_links.iter().enumerate() {
                if worker_id >= n_bricks {
                    // `brick_face_links` is sized to max_worker_brick + 1
                    // which equals n_bricks since worker IDs are a dense
                    // 0..n range. Defensive against future changes.
                    break;
                }
                let scene_id = worker_to_scene_brick[worker_id];
                let mut remapped = [FACE_EMPTY; 6];
                for (i, &neighbor) in row.iter().enumerate() {
                    remapped[i] = if neighbor == FACE_EMPTY || neighbor == FACE_INTERIOR {
                        neighbor
                    } else {
                        worker_to_scene_brick[neighbor as usize]
                    };
                }
                scene_rows[scene_id as usize] = remapped;
            }
            self.merge_face_links(&scene_rows);
        }

        let t_face_links = t_start.elapsed();
        // NOTE: previously called `emit_faces` here to populate
        // `pending_faces`, but that Vec has no consumer anywhere in
        // the engine today (splat raster pipeline is retired). At
        // 5-10 M voxels the per-leaf 6-neighbor-lookup pass is
        // multi-second on the main thread for zero benefit. If a
        // face rasterizer comes back, resurrect this + re-wire the
        // consumer rather than routing unused work through every
        // bake.
        let handle = self.octree.allocate(&artifact.octree);
        let t_octree_alloc = t_start.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f32() * 1000.0;
        eprintln!(
            "[integrate_artifact] voxels={} bricks={} attrs={}  \
             attr_alloc={:.2}ms attr_copy={:.2}ms brick_copy={:.2}ms \
             octree_remap={:.2}ms face_links={:.2}ms \
             octree_alloc={:.2}ms total={:.2}ms",
            artifact.voxel_count,
            n_bricks,
            n_attrs,
            ms(t_attr_alloc),
            ms(t_attr_copy - t_attr_alloc),
            ms(t_brick_copy - t_attr_copy),
            ms(t_octree_remap - t_brick_copy),
            ms(t_face_links - t_octree_remap),
            ms(t_octree_alloc - t_face_links),
            ms(t_octree_alloc),
        );
        let spatial = rkp_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        Some(VoxelizeResult {
            spatial,
            voxel_size,
            aabb: *aabb,
            grid_origin: artifact.grid_origin,
            voxel_count: artifact.voxel_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: n_attrs,
            brick_ids: brick_ids_scene,
        })
    }

    /// Deallocate geometry previously produced by voxelize_*. Frees the
    /// octree, the leaf_attr range, and every brick that voxelization
    /// allocated. Bricks go through the bulk-batch path — the per-
    /// brick `BrickPool::deallocate` has a tail-coalesce loop that is
    /// O(n²) when the batch is a contiguous range (the common case
    /// when re-baking a procedural). Async bakes of 10M+ voxels had
    /// apply times of 5 s+ sitting in that loop; the batch path
    /// collapses it to milliseconds.
    pub fn deallocate_geometry(
        &mut self,
        spatial: &rkp_core::OctreeHandle,
        leaf_attr_slot_start: u32,
        leaf_attr_slot_count: u32,
        brick_ids: &[u32],
    ) {
        self.bump_geometry_epoch();
        self.octree.deallocate(*spatial);
        self.leaf_attr_pool.deallocate_range(leaf_attr_slot_start, leaf_attr_slot_count);
        self.brick_pool.deallocate_batch(brick_ids);
    }
}
