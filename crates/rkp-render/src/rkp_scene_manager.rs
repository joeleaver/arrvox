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
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    /// Entity-local grid origin (`aabb_center - extent/2`). Derived at
    /// load time — .rkp files voxelized before this field existed used
    /// the same formula, so re-deriving reproduces the exact bake.
    pub grid_origin: glam::Vec3,
    pub voxel_count: u32,
    pub leaf_attr_slot_start: u32,
    pub leaf_attr_slot_count: u32,
}

/// One entry in the asset cache: the shared geometry allocations plus
/// a refcount. When `refcount` hits zero, `release_asset` frees the
/// octree / leaf_attr / brick ranges.
struct AssetEntry {
    path: PathBuf,
    refcount: u32,
    spatial_handle: OctreeHandle,
    voxel_size: f32,
    aabb: rkf_core::Aabb,
    voxel_count: u32,
    leaf_attr_slot_start: u32,
    leaf_attr_slot_count: u32,
    brick_start: u32,
    brick_count: u32,
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
            spatial: rkf_core::scene_node::SpatialHandle::Octree {
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
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
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
        }
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
            brick_pool: self.brick_pool.as_bytes(),
            brick_face_links: rkp_core::brick_face_links::as_bytes(&self.brick_face_links),
        }
    }

    // ── Spatial deallocation ─────────────────────────────────────────

    pub fn deallocate_spatial(&mut self, handle: &rkf_core::scene_node::SpatialHandle) {
        if let rkf_core::scene_node::SpatialHandle::Octree {
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
        use rkf_core::voxel::VoxelSample;

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
        let aabb = rkf_core::Aabb::new(
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

        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        let mut file_voxel_mat: Vec<(u16, u16, u8, u32, u32)> = Vec::with_capacity(voxel_count as usize);
        for i in 0..voxel_count as usize {
            let src_offset = i * bytes_per_voxel;
            if src_offset + bytes_per_voxel > voxel_data.len() {
                break;
            }
            let vs: &VoxelSample =
                bytemuck::from_bytes(&voxel_data[src_offset..src_offset + bytes_per_voxel]);
            let color = color_u32s.get(i).copied().unwrap_or(0);
            let normal_oct = normals_u32s.get(i).copied().unwrap_or(0);
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
            if normal_oct != 0 {
                attr.normal_oct = normal_oct;
            }
            let slot = leaf_attr_slot_start + i as u32;
            *self.leaf_attr_pool.get_mut(slot) = attr;
            if color != 0 {
                self.leaf_attr_pool.set_color(slot, color);
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
                // and corrupt the slot into a bogus leaf_attr_id.
                let remapped = if cell == rkp_core::brick_pool::BRICK_INTERIOR {
                    rkp_core::brick_pool::BRICK_INTERIOR
                } else {
                    // Real leaf: cell is a file-local slot index; shift
                    // by our leaf_attr allocation offset to get the
                    // scene-global leaf_attr_id.
                    leaf_attr_slot_start + cell
                };
                let x = (i as u32) % rkp_core::brick_pool::BRICK_DIM;
                let y = ((i as u32) / rkp_core::brick_pool::BRICK_DIM) % rkp_core::brick_pool::BRICK_DIM;
                let z = (i as u32) / (rkp_core::brick_pool::BRICK_DIM * rkp_core::brick_pool::BRICK_DIM);
                self.brick_pool.set_cell(scene_id, x, y, z, remapped);
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

        // Bake-time Laplacian relaxation of shell-voxel normals.
        // Converts per-voxel SDF-gradient samples (which alias into
        // discrete directions at voxel scale) into locally-averaged
        // smooth normals. Runs once per asset load; each asset's
        // leaf_attrs are 1:1 with voxels (no dedup), which is the
        // invariant the smoother requires.
        let smoothed_count = rkp_core::laplacian_smooth::smooth_shell_normals(
            &tree, &self.brick_pool, &mut self.leaf_attr_pool, 3,
        );
        eprintln!(
            "[RkpSceneManager]   smoothed {} shell normals (3 Laplacian iterations)",
            smoothed_count,
        );

        // Run the prefilter pass on-load so v4 assets (no baked internal
        // attrs) still benefit from the GPU's LOD early-exit. Phase 4
        // bumps the .rkp format to v5 which bakes these at conversion
        // time — this is the fallback until then.
        //
        // The prefilter appends new attrs at the tail of the asset's
        // contiguous leaf_attr range via allocate_contiguous_bump(1), so
        // the `leaf_attr_slot_count` grows to cover them and the
        // existing deallocate_range releases everything on asset drop.
        let mut attr_dedup: HashMap<LeafAttr, u32> = HashMap::new();
        for i in 0..leaf_attr_slot_count {
            let slot = leaf_attr_slot_start + i;
            attr_dedup.insert(*self.leaf_attr_pool.get(slot), slot);
        }
        rkp_core::prefilter::prefilter_octree_internals(
            &mut tree,
            &mut self.leaf_attr_pool,
            &self.brick_pool,
            &mut attr_dedup,
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
            "[RkpSceneManager] loaded {}: {} voxels, {} bricks, octree {} → compact {} → dedup {} ({:.1}× total), +{} prefilter attrs",
            rkp_path.display(),
            voxel_count,
            file_brick_count,
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
            final_leaf_attr_slot_count - leaf_attr_slot_count,
        );

        Ok(AssetEntry {
            path: rkp_path.to_path_buf(),
            refcount: 1,
            spatial_handle: handle,
            voxel_size,
            aabb,
            voxel_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count: final_leaf_attr_slot_count,
            brick_start: scene_brick_offset,
            brick_count: file_brick_count,
        })
    }

    // ── Primitive voxelization ───────────────────────────────────────

    /// Voxelize an SDF primitive into the octree.
    pub fn voxelize_primitive(
        &mut self,
        primitive: &rkf_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        object_id: u32,
    ) -> Option<VoxelizeResult> {
        use rkf_core::scene_node::SdfPrimitive;

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
        let aabb = rkf_core::Aabb::new(
            -half_extents - glam::Vec3::splat(margin),
            half_extents + glam::Vec3::splat(margin),
        );

        // SDF closure passed directly to the voxelizer. Negative = inside.
        let sdf_fn: Box<dyn Fn(glam::Vec3) -> f32> = match primitive {
            SdfPrimitive::Box { half_extents: he } => {
                let scaled = SdfPrimitive::Box { half_extents: *he * bake_scale };
                Box::new(move |pos| rkf_core::evaluate_primitive(&scaled, pos))
            }
            _ => {
                let prim = primitive.clone();
                let min_scale = bake_scale.x.min(bake_scale.y).min(bake_scale.z).max(1e-6);
                let inv_scale = glam::Vec3::new(
                    1.0 / bake_scale.x.max(1e-6),
                    1.0 / bake_scale.y.max(1e-6),
                    1.0 / bake_scale.z.max(1e-6),
                );
                Box::new(move |pos| rkf_core::evaluate_primitive(&prim, pos * inv_scale) * min_scale)
            }
        };

        let sdf_with_material = |pos: glam::Vec3| -> (f32, u16) {
            (sdf_fn(pos), material_id)
        };

        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_with_material, &aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        let geometry_aabb = rkf_core::Aabb::new(-half_extents, half_extents);
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
    /// The closure returns `(signed_distance, material_id)`. Negative = inside.
    pub fn voxelize_sdf_fn<F>(
        &mut self,
        sdf_fn: F,
        aabb: &rkf_core::Aabb,
        voxel_size: f32,
        object_id: u32,
    ) -> Option<VoxelizeResult>
    where
        F: Fn(glam::Vec3) -> (f32, u16),
    {
        let r = rkp_core::voxelize_octree::voxelize_octree(
            sdf_fn, aabb, voxel_size, &mut self.leaf_attr_pool, &mut self.brick_pool,
        )?;

        emit_faces(&r.octree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        self.merge_face_links(&r.brick_face_links);
        let handle = self.octree.allocate(&r.octree);
        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
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

    /// Deallocate geometry previously produced by voxelize_*. Frees the
    /// octree, the leaf_attr range, and every brick that voxelization
    /// allocated. Bricks are freed one at a time via `BrickPool::deallocate`
    /// (no contiguous range guarantee — `voxelize_octree` may reuse freed
    /// brick ids from the pool's free list on the way down the tree).
    pub fn deallocate_geometry(
        &mut self,
        spatial: &rkp_core::OctreeHandle,
        leaf_attr_slot_start: u32,
        leaf_attr_slot_count: u32,
        brick_ids: &[u32],
    ) {
        self.octree.deallocate(*spatial);
        self.leaf_attr_pool.deallocate_range(leaf_attr_slot_start, leaf_attr_slot_count);
        for &id in brick_ids {
            self.brick_pool.deallocate(id);
        }
    }
}

// `compute_leaf_normal_neighborhood26` used to live here for the legacy
// v2/v3 LEAF-path fallback; removed with the switch to v4-only loading.
