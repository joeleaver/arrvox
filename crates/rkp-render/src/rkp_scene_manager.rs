//! Scene management for RKIPatch — owns the leaf_attr pool, octrees, and
//! face instances.
//!
//! This is the CPU-side scene representation. It manages the LeafAttrPool
//! (material + normal + color per leaf), the OctreeGpu allocator, and the
//! face instance list (legacy, unused by the active pipeline).
//!
//! No wgpu types, no GPU buffers here — RkpRenderer consumes the snapshot.

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

/// Result of loading an .rkp asset.
pub struct AssetLoadResult {
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    /// Logical voxel count (octree leaves) — for display / stats.
    pub voxel_count: u32,
    /// First leaf_attr pool slot used by this allocation.
    pub leaf_attr_slot_start: u32,
    /// Number of leaf_attr slots allocated (distinct (material, normal) tuples).
    pub leaf_attr_slot_count: u32,
}

/// Result of voxelizing a primitive.
pub struct VoxelizeResult {
    pub spatial: rkf_core::scene_node::SpatialHandle,
    pub voxel_size: f32,
    pub aabb: rkf_core::Aabb,
    /// Logical voxel count (octree leaves).
    pub voxel_count: u32,
    /// First leaf_attr pool slot used by this allocation.
    pub leaf_attr_slot_start: u32,
    /// Number of leaf_attr slots allocated.
    pub leaf_attr_slot_count: u32,
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
    /// GPU octree allocator (packs all octrees into one buffer).
    pub octree: OctreeGpu,
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
            octree: OctreeGpu::new(),
            pending_faces: Vec::new(),
            faces_dirty: false,
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
        let octree_data = self.octree.data();
        GeometryUpload {
            octree_nodes: bytemuck::cast_slice(octree_data),
            leaf_attr_pool: self.leaf_attr_pool.as_bytes(),
            color_pool: self.leaf_attr_pool.color_bytes(),
            brick_pool: self.brick_pool.as_bytes(),
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

    /// Load an .rkp asset file into the scene.
    ///
    /// Reads the legacy per-voxel format (opacity + material + color), then
    /// collapses each leaf's material + computed normal into a single
    /// LeafAttr entry. Opacity values from the file are discarded — the
    /// file-format version is unchanged for now, migration at load time is
    /// cheap enough that a full format bump can wait.
    pub fn load_rkp(&mut self, path: &str, object_id: u32) -> Result<AssetLoadResult, String> {
        use rkf_core::voxel::VoxelSample;

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
        let leaf_attr_slot_start = self.leaf_attr_pool.allocated_count();
        let mut tree = SparseOctree::from_raw(&octree_nodes, octree_depth, voxel_size);

        let mut attr_dedup: std::collections::HashMap<(LeafAttr, u32), u32> =
            std::collections::HashMap::new();

        // Closure: resolve a file slot to a leaf_attr_pool id, deduping on
        // (LeafAttr, color). Used by both the BRICK path (v4) and the
        // legacy LEAF path (v2/v3).
        let resolve_slot = |slot: u32,
                            attr_dedup: &mut std::collections::HashMap<(LeafAttr, u32), u32>,
                            pool: &mut LeafAttrPool|
         -> u32 {
            let (mat_p, mat_s, blend, color, normal_oct) = file_voxel_mat
                .get(slot as usize)
                .copied()
                .unwrap_or((0, 0, 0, 0, 0));
            let mut attr = LeafAttr::new_blended(glam::Vec3::Y, mat_p, mat_s, blend);
            if normal_oct != 0 {
                attr.normal_oct = normal_oct;
            }
            let key = (attr, color);
            *attr_dedup.entry(key).or_insert_with(|| {
                let id = pool.allocate().expect("leaf_attr_pool.allocate failed");
                *pool.get_mut(id) = attr;
                if color != 0 {
                    pool.set_color(id, color);
                }
                id
            })
        };

        // v4: walk BRICK nodes, allocate a scene brick for each, fill its
        // cells from the file brick after dedup'ing slot indices into
        // leaf_attr_pool ids.
        let file_brick_count = file_brick_cells.len() / rkp_core::brick_pool::BRICK_CELLS as usize;
        let scene_brick_offset = self.brick_pool.allocated_count();
        for _ in 0..file_brick_count {
            self.brick_pool.allocate().expect("brick_pool.allocate failed");
        }
        // Remap every BRICK node in the flat nodes array: shift its brick_id
        // by the scene offset. This rewrites nodes in place — no tree walk
        // needed because BRICK encoding is distinguishable from every other
        // node type by is_brick.
        {
            let nodes = tree.as_slice_mut();
            for n in nodes.iter_mut() {
                if rkp_core::sparse_octree::is_brick(*n) {
                    let file_id = rkp_core::sparse_octree::brick_id(*n);
                    *n = rkp_core::sparse_octree::make_brick(scene_brick_offset + file_id);
                }
            }
        }
        let brick_cells = rkp_core::brick_pool::BRICK_CELLS as usize;
        for file_id in 0..file_brick_count as u32 {
            let scene_id = scene_brick_offset + file_id;
            let src = &file_brick_cells[file_id as usize * brick_cells..(file_id as usize + 1) * brick_cells];
            for (i, &slot_or_empty) in src.iter().enumerate() {
                if slot_or_empty == rkp_core::brick_pool::BRICK_EMPTY {
                    continue;
                }
                let leaf_attr_id = resolve_slot(slot_or_empty, &mut attr_dedup, &mut self.leaf_attr_pool);
                let x = (i as u32) % rkp_core::brick_pool::BRICK_DIM;
                let y = ((i as u32) / rkp_core::brick_pool::BRICK_DIM) % rkp_core::brick_pool::BRICK_DIM;
                let z = (i as u32) / (rkp_core::brick_pool::BRICK_DIM * rkp_core::brick_pool::BRICK_DIM);
                self.brick_pool.set_cell(scene_id, x, y, z, leaf_attr_id);
            }
        }

        // Legacy LEAF path (v2/v3 files, no bricks): walk leaves, dedup,
        // rewrite node. 26-neighborhood kernel falls back when no baked
        // normal is present.
        if !has_bricks {
            let leaves: Vec<(glam::UVec3, u32, u8)> = tree.iter_leaves().collect();
            let mut rewrites: Vec<(glam::UVec3, u8, u32)> = Vec::with_capacity(leaves.len());
            for (coord, file_idx, leaf_depth) in &leaves {
                let (mat_p, mat_s, blend, color, normal_oct) = file_voxel_mat
                    .get(*file_idx as usize)
                    .copied()
                    .unwrap_or((0, 0, 0, 0, 0));
                let attr = if has_normals && normal_oct != 0 {
                    let mut a = LeafAttr::new_blended(glam::Vec3::Y, mat_p, mat_s, blend);
                    a.normal_oct = normal_oct;
                    a
                } else {
                    let normal = compute_leaf_normal_neighborhood26(&tree, *coord);
                    LeafAttr::new_blended(normal, mat_p, mat_s, blend)
                };
                let key = (attr, color);
                let leaf_attr_id = *attr_dedup.entry(key).or_insert_with(|| {
                    let id = self.leaf_attr_pool.allocate()
                        .expect("leaf_attr_pool.allocate failed");
                    *self.leaf_attr_pool.get_mut(id) = attr;
                    if color != 0 {
                        self.leaf_attr_pool.set_color(id, color);
                    }
                    id
                });
                rewrites.push((*coord, *leaf_depth, rkp_core::sparse_octree::make_leaf(leaf_attr_id)));
            }
            for (coord, leaf_depth, new_value) in rewrites {
                tree.set_at_level(coord, leaf_depth, new_value);
            }
        }
        let leaf_attr_slot_count = self.leaf_attr_pool.allocated_count() - leaf_attr_slot_start;

        let raw_count = tree.node_count();
        tree.collapse_all();
        tree.compact();
        let compact_count = tree.node_count();
        tree.deduplicate_subtrees();
        let dedup_count = tree.node_count();
        tree.morton_reorder();
        let compact_nodes = tree.as_slice().to_vec();

        let handle = self.octree.allocate_raw(&compact_nodes, octree_depth, voxel_size);

        emit_faces(&tree, object_id, &mut self.pending_faces);
        self.faces_dirty = true;

        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        eprintln!(
            "[RkpSceneManager] loaded .rkp: {} voxels → {} unique leaf_attrs ({:.1}×), {} faces, octree {} → compact {} → dedup {} ({:.1}× total)",
            voxel_count,
            leaf_attr_slot_count,
            if leaf_attr_slot_count > 0 { voxel_count as f64 / leaf_attr_slot_count as f64 } else { 0.0 },
            self.pending_faces.len(),
            raw_count,
            compact_count,
            dedup_count,
            if dedup_count > 0 { raw_count as f64 / dedup_count as f64 } else { 0.0 },
        );

        Ok(AssetLoadResult {
            spatial, voxel_size, aabb, voxel_count,
            leaf_attr_slot_start,
            leaf_attr_slot_count,
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
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
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
            leaf_attr_slot_start: r.leaf_attr_slot_start,
            leaf_attr_slot_count: r.leaf_attr_unique_count,
        })
    }

    /// Deallocate geometry previously produced by voxelize_*.
    pub fn deallocate_geometry(&mut self, spatial: &rkp_core::OctreeHandle, leaf_attr_slot_start: u32, leaf_attr_slot_count: u32) {
        self.octree.deallocate(*spatial);
        self.leaf_attr_pool.deallocate_range(leaf_attr_slot_start, leaf_attr_slot_count);
    }
}

/// 26-neighborhood centroid kernel. For each of the 26 surrounding cells,
/// if the neighbor is occupied (leaf or interior), accumulate its offset
/// vector; the mean points toward the centroid of occupied mass, so the
/// outward normal is its negation. Uses all 26 neighbors instead of just
/// the 6 axis-aligned ones, yielding ~direction-quantized-to-sphere output
/// instead of being pinned to ~26 discrete axial directions.
fn compute_leaf_normal_neighborhood26(
    tree: &SparseOctree,
    coord: glam::UVec3,
) -> glam::Vec3 {
    let occupied_at = |c: glam::UVec3| -> bool {
        match tree.lookup(c) {
            Some(n) if n == rkp_core::sparse_octree::INTERIOR_NODE => true,
            Some(n) if rkp_core::sparse_octree::is_leaf(n) => true,
            _ => false,
        }
    };
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0.0f32;
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let x = coord.x as i64 + dx as i64;
                let y = coord.y as i64 + dy as i64;
                let z = coord.z as i64 + dz as i64;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let nb = glam::UVec3::new(x as u32, y as u32, z as u32);
                if occupied_at(nb) {
                    sum += glam::Vec3::new(dx as f32, dy as f32, dz as f32);
                    count += 1.0;
                }
            }
        }
    }
    if count == 0.0 {
        return glam::Vec3::Y;
    }
    let centroid = sum / count;
    if centroid.length_squared() > 1e-12 {
        -centroid.normalize()
    } else {
        glam::Vec3::Y
    }
}
