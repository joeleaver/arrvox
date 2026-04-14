//! Scene management for RKIPatch — owns voxel data, octrees, and face instances.
//!
//! This is the CPU-side scene representation. It manages the VoxelPool (opacity +
//! color per voxel), the OctreeGpu allocator (spatial structure), and the face
//! instance list (surface shell for rasterization).
//!
//! The RkpSceneManager is GPU-agnostic: it produces data that RkpRenderer uploads.
//! No wgpu types, no GPU buffers, no rendering logic.

use std::collections::HashMap;

use rkp_core::{ExtractedMesh, OctreeHandle, SparseOctree, SplatVoxel, VoxelPool};

use crate::octree_gpu::OctreeGpu;
use crate::rkp_scene::GeometryUpload;

/// Face instance for CPU-side face emission (legacy — kept for scene loading compatibility).
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

/// Result of loading an .rkp asset.
pub struct AssetLoadResult {
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    /// Logical voxel count (octree leaves) — for display / stats.
    pub voxel_count: u32,
    /// First voxel pool slot used by this allocation.
    pub voxel_slot_start: u32,
    /// Number of slots actually allocated. May be less than `voxel_count` when
    /// identical (voxel, color) pairs share a slot via dedup.
    pub voxel_slot_count: u32,
    /// Triangle mesh extracted from the opacity field via marching cubes.
    /// The caller uploads this to [`MeshPool`](crate::MeshPool) keyed by
    /// its `object_id` so the triangle raster pass can draw it.
    pub extracted_mesh: ExtractedMesh,
}

/// Result of voxelizing a primitive.
pub struct VoxelizeResult {
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    /// Logical voxel count (octree leaves).
    pub voxel_count: u32,
    /// First voxel pool slot used by this allocation.
    pub voxel_slot_start: u32,
    /// Number of slots actually allocated — use this (not `voxel_count`) for
    /// deallocation, since the two can diverge when slots are shared.
    pub voxel_slot_count: u32,
    /// Triangle mesh extracted from the opacity field via marching cubes.
    /// The caller uploads this to [`MeshPool`](crate::MeshPool) keyed by
    /// its `object_id` so the triangle raster pass can draw it.
    pub extracted_mesh: ExtractedMesh,
}

/// Emit face instances from an octree into the given buffer.
///
/// Free function to avoid borrow-checker issues when the octree and pool
/// are both owned by the same struct as the face buffer.
fn emit_faces(
    octree: &SparseOctree,
    pool: &VoxelPool,
    obj_idx: u32,
    faces: &mut Vec<FaceInstance>,
) {
    let base_vs = octree.base_voxel_size();

    for (coord, slot, leaf_depth) in octree.iter_leaves() {
        let sv = pool.get(slot);
        if sv.opacity_f32() <= 0.01 {
            continue;
        }

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
                    Some(node) if rkp_core::sparse_octree::is_leaf(node) => {
                        let nb_slot = rkp_core::sparse_octree::leaf_slot(node);
                        pool.get(nb_slot).opacity_f32() <= 0.01
                    }
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
                    voxel_slot: slot,
                    packed: (face & 0x7) | ((obj_idx & 0xFFFFF) << 3),
                });
            }
        }
    }
}

/// CPU-side scene manager — voxel data, octrees, face instances.
pub struct RkpSceneManager {
    /// Per-voxel opacity + material + color storage.
    pub voxel_pool: VoxelPool,
    /// GPU octree allocator (packs all octrees into one buffer).
    pub octree: OctreeGpu,
    /// Face instances for rasterization (surface shell).
    pending_faces: Vec<FaceInstance>,
    /// Whether face data needs re-upload to GPU.
    faces_dirty: bool,
}

impl RkpSceneManager {
    /// Create with default capacity.
    pub fn new(voxel_capacity: u32) -> Self {
        Self {
            voxel_pool: VoxelPool::new(voxel_capacity),
            octree: OctreeGpu::new(),
            pending_faces: Vec::new(),
            faces_dirty: false,
        }
    }

    // ── Face emission ────────────────────────────────────────────────

    /// Emit face instances from a per-voxel octree.
    ///
    /// Traverses leaves. For each non-empty leaf, checks 6 neighbors and emits
    /// exposed faces. Each leaf IS a single voxel (no bricks).
    pub fn emit_faces_from_octree(
        &mut self,
        octree: &SparseOctree,
        pool: &VoxelPool,
        obj_idx: u32,
    ) {
        emit_faces(octree, pool, obj_idx, &mut self.pending_faces);
        self.faces_dirty = true;
    }

    /// Emit faces from raw octree node data (for file loading).
    pub fn emit_faces_from_raw_octree(
        &mut self,
        nodes: &[u32],
        depth: u8,
        base_vs: f32,
        pool: &VoxelPool,
        obj_idx: u32,
    ) {
        let octree = SparseOctree::from_raw(nodes, depth, base_vs);
        emit_faces(&octree, pool, obj_idx, &mut self.pending_faces);
        self.faces_dirty = true;
    }

    /// Access pending face instances.
    pub fn pending_faces(&self) -> &[FaceInstance] {
        &self.pending_faces
    }

    /// Whether face data has changed since last upload.
    pub fn faces_dirty(&self) -> bool {
        self.faces_dirty
    }

    /// Mark faces as clean (after GPU upload).
    pub fn mark_faces_clean(&mut self) {
        self.faces_dirty = false;
    }

    /// Clear all face instances.
    pub fn clear_faces(&mut self) {
        self.pending_faces.clear();
        self.faces_dirty = true;
    }

    // ── Geometry upload snapshot ──────────────────────────────────────

    /// Build a GeometryUpload snapshot for RkpRenderer.
    pub fn geometry_upload(&self) -> GeometryUpload<'_> {
        let octree_data = self.octree.data();
        GeometryUpload {
            voxel_pool: self.voxel_pool.as_bytes(),
            octree_nodes: bytemuck::cast_slice(octree_data),
            color_pool: self.voxel_pool.color_bytes(),
        }
    }

    // ── Spatial deallocation ─────────────────────────────────────────

    /// Deallocate an octree spatial handle.
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

    /// Load an .rkp asset file into the scene.
    ///
    /// Allocates voxels in the pool, remaps octree leaf slots, emits faces,
    /// loads per-voxel color data. Returns the spatial handle and metadata
    /// needed to create a scene object.
    pub fn load_rkp(&mut self, path: &str, object_id: u32) -> Result<AssetLoadResult, String> {
        use rkf_core::voxel::VoxelSample;

        // Find the .rkp file.
        let rkp_path = if path.ends_with(".rkp") {
            std::path::PathBuf::from(path)
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

        let mut file = std::fs::File::open(&rkp_path)
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

        // Read per-voxel color data up front — needed so dedup can key on
        // (voxel, color) pairs, not just voxel contents.
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

        // Deduplicate (voxel, color) pairs. Meshes typically have huge
        // redundancy — a textured surface often has only thousands of unique
        // (opacity, material, color) combinations across millions of voxels.
        // Sharing slots cuts voxel_pool + color_pool upload size dramatically
        // and keeps the working set inside the GPU cache.
        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        let mut dedup: std::collections::HashMap<(SplatVoxel, u32), u32> =
            std::collections::HashMap::new();
        let mut unique: Vec<(SplatVoxel, u32)> = Vec::new();
        let mut file_to_unique: Vec<u32> = Vec::with_capacity(voxel_count as usize);
        for i in 0..voxel_count as usize {
            let src_offset = i * bytes_per_voxel;
            if src_offset + bytes_per_voxel > voxel_data.len() {
                break;
            }
            let vs: &VoxelSample =
                bytemuck::from_bytes(&voxel_data[src_offset..src_offset + bytes_per_voxel]);
            let splat = SplatVoxel::from(*vs);
            let color = color_u32s.get(i).copied().unwrap_or(0);
            let key = (splat, color);
            let idx = *dedup.entry(key).or_insert_with(|| {
                let new_idx = unique.len() as u32;
                unique.push(key);
                new_idx
            });
            file_to_unique.push(idx);
        }
        let unique_count = unique.len() as u32;

        // Allocate one pool slot per UNIQUE (voxel, color) — not per file
        // voxel. A typical mesh shrinks by 100×+ here.
        if self.voxel_pool.free_count() < unique_count {
            let new_cap = (self.voxel_pool.capacity() * 2)
                .max(self.voxel_pool.capacity() + unique_count);
            self.voxel_pool.grow(new_cap);
        }
        let slots = self.voxel_pool.allocate_range(unique_count)
            .ok_or_else(|| "failed to allocate voxel pool slots".to_string())?;
        let voxel_slot_start = slots.first().copied().unwrap_or(0);

        // Write unique voxels + colors into their pool slots.
        for (i, &(splat, color)) in unique.iter().enumerate() {
            let slot = slots[i];
            *self.voxel_pool.get_mut(slot) = splat;
            if has_color {
                self.voxel_pool.set_color(slot, color);
            }
        }

        // Remap octree leaf slots: file voxel index → dedup index → pool slot.
        let mut remapped_nodes = octree_nodes.clone();
        for node in &mut remapped_nodes {
            if *node != rkp_core::sparse_octree::EMPTY_NODE
                && *node != rkp_core::sparse_octree::INTERIOR_NODE
                && (*node & rkp_core::sparse_octree::LEAF_BIT) != 0
            {
                let file_idx = (*node & !rkp_core::sparse_octree::LEAF_BIT) as usize;
                if file_idx < file_to_unique.len() {
                    let dedup_idx = file_to_unique[file_idx] as usize;
                    *node = rkp_core::sparse_octree::make_leaf(slots[dedup_idx]);
                }
            }
        }

        // After dedup-remap, many leaves that used to have distinct slot
        // indices now share the same slot. Three passes in order:
        //   collapse_all()         — merge uniform 8-child subtrees into a
        //                            single leaf (try_collapse didn't fire
        //                            during the post-load remap).
        //   compact()              — reclaim orphan storage from the collapse.
        //   deduplicate_subtrees() — share identical 8-child blocks as DAG
        //                            references. Massive reduction for
        //                            geometry with any repetition.
        let octree_depth = header.octree_depth as u8;
        let mut tree = SparseOctree::from_raw(&remapped_nodes, octree_depth, voxel_size);
        let raw_count = tree.node_count();
        tree.collapse_all();
        tree.compact();
        let compact_count = tree.node_count();
        tree.deduplicate_subtrees();
        let dedup_count = tree.node_count();
        let compact_nodes = tree.as_slice().to_vec();

        let handle = self.octree.allocate_raw(&compact_nodes, octree_depth, voxel_size);

        // Emit faces via octree-based neighbor lookup.
        eprintln!("[RkpSceneManager] emitting faces with object_id={}", object_id);
        {
            emit_faces(&tree, &self.voxel_pool, object_id, &mut self.pending_faces);
            self.faces_dirty = true;
        }

        // Extract a marching-cubes mesh for the triangle raster pass.
        let extracted_mesh = rkp_core::extract_mesh(&tree, &self.voxel_pool);
        let nonzero_colors = extracted_mesh.colors.iter().filter(|&&c| c != 0).count();
        let nonzero_mats = extracted_mesh.material_ids.iter().filter(|&&m| m != 0).count();
        eprintln!(
            "[RkpSceneManager] load_rkp extracted mesh: {} tris, {} verts, \
             {} nonzero colors, {} nonzero materials",
            extracted_mesh.triangle_count(),
            extracted_mesh.vertex_count(),
            nonzero_colors,
            nonzero_mats,
        );

        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        eprintln!(
            "[RkpSceneManager] loaded .rkp: {} voxels → {} unique pool slots ({:.1}×), {} faces, octree {} → compact {} → dedup {} ({:.1}× total)",
            voxel_count,
            unique_count,
            if unique_count > 0 { voxel_count as f64 / unique_count as f64 } else { 0.0 },
            self.pending_faces.len(),
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
        );

        Ok(AssetLoadResult {
            spatial, voxel_size, aabb, voxel_count, voxel_slot_start,
            voxel_slot_count: unique_count,
            extracted_mesh,
        })
    }

    // ── Primitive voxelization ───────────────────────────────────────

    /// Voxelize an SDF primitive into the octree as an opacity field.
    pub fn voxelize_primitive(
        &mut self,
        primitive: &rkf_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        object_id: u32,
    ) -> Option<VoxelizeResult> {
        use rkf_core::scene_node::SdfPrimitive;

        fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }

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

        let fade_inner = voxel_size * 1.0;
        let fade_outer = voxel_size * 3.0;

        let half_extents = primitive_half_extents(primitive) * bake_scale;
        let margin = voxel_size * 8.0 * 1.8 + voxel_size;
        let aabb = rkf_core::Aabb::new(
            -half_extents - glam::Vec3::splat(margin),
            half_extents + glam::Vec3::splat(margin),
        );

        // Build SDF closure.
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

        // Convert SDF to opacity closure.
        let opacity_fn = |pos: glam::Vec3| -> (f32, u16) {
            let d = sdf_fn(pos);
            let opacity = 1.0 - smoothstep(-fade_inner, fade_outer, d);
            (opacity.clamp(0.0, 1.0), material_id)
        };

        // Per-voxel octree voxelization.
        let r = rkp_core::voxelize_octree::voxelize_opacity_octree(
            opacity_fn, &aabb, voxel_size, &mut self.voxel_pool,
        )?;

        // Emit face instances.
        emit_faces(&r.octree, &self.voxel_pool, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        // Extract MC mesh for the triangle raster pass.
        let extracted_mesh = rkp_core::extract_mesh(&r.octree, &self.voxel_pool);

        // Allocate octree into the GPU allocator.
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
            voxel_count: r.voxel_count,
            voxel_slot_start: r.slot_start,
            voxel_slot_count: r.unique_count,
            extracted_mesh,
        })
    }

    /// Voxelize an arbitrary opacity function into the octree.
    ///
    /// Generic entry point — the caller provides the opacity closure and bounding
    /// box. Used by the procedural object system and any other source of opacity
    /// fields.
    pub fn voxelize_opacity_fn<F>(
        &mut self,
        opacity_fn: F,
        aabb: &rkf_core::Aabb,
        voxel_size: f32,
        object_id: u32,
    ) -> Option<VoxelizeResult>
    where
        F: Fn(glam::Vec3) -> (f32, u16),
    {
        let r = rkp_core::voxelize_octree::voxelize_opacity_octree(
            opacity_fn, aabb, voxel_size, &mut self.voxel_pool,
        )?;

        emit_faces(&r.octree, &self.voxel_pool, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        // Extract MC mesh for the triangle raster pass.
        let extracted_mesh = rkp_core::extract_mesh(&r.octree, &self.voxel_pool);

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
            voxel_count: r.voxel_count,
            voxel_slot_start: r.slot_start,
            voxel_slot_count: r.unique_count,
            extracted_mesh,
        })
    }

    /// Deallocate geometry previously produced by voxelize_*.
    ///
    /// Frees both the octree region (returned to the octree allocator's free
    /// list) and the voxel pool slots (returned to the voxel pool's free list
    /// or bump-shrunk if contiguous).
    pub fn deallocate_geometry(&mut self, spatial: &rkp_core::OctreeHandle, voxel_slot_start: u32, voxel_slot_count: u32) {
        self.octree.deallocate(*spatial);
        self.voxel_pool.deallocate_range(voxel_slot_start, voxel_slot_count);
    }
}
