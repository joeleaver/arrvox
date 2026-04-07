//! SplatRasterPass — MarchPass implementation using forward rasterization.
//!
//! Replaces `SplatMarchPass` (compute ray march) with:
//! 1. Emit compute pass: traverses octrees, emits transition face quads
//! 2. Raster render pass: vertex/fragment pipeline writes G-buffer via MRT
//!
//! Implements the `rkf_render::MarchPass` trait so the editor engine can use it
//! as a drop-in replacement. The `dispatch()` method begins a render pass
//! (not a compute pass) — this is allowed because the trait takes
//! `&mut CommandEncoder` which supports both.

use crate::octree_gpu::OctreeGpu;
use crate::splat_emit::SplatEmitPass;
use crate::splat_raster::SplatRasterPipeline;

/// Forward rasterization march pass — replaces the compute ray march.
pub struct SplatRasterPass {
    emit: SplatEmitPass,
    raster: SplatRasterPipeline,
    octree: OctreeGpu,
    /// Per-voxel pool — each leaf is a single voxel, no bricks.
    voxel_pool: rkp_core::VoxelPool,
    /// Standalone renderer (shadow/AO + shading). Created by new_standalone().
    rkp_renderer: Option<std::cell::RefCell<crate::rkp_renderer::RkpRenderer>>,
    /// Face instances from CPU-side emit. The packed field stores sdf_object_id
    /// (not GPU index) — corrected during upload using object_gpu_mapping.
    pending_faces: std::cell::RefCell<Vec<crate::splat_emit::FaceInstance>>,
    /// The device, needed for creating staging buffers in dispatch.
    device: wgpu::Device,
    /// The queue, needed for buffer writes.
    queue: wgpu::Queue,
    /// Whether face data needs re-upload.
    faces_dirty: std::cell::Cell<bool>,
    /// Mapping from SDF object_id → GPU object index. Updated by the engine.
    object_gpu_mapping: std::cell::RefCell<std::collections::HashMap<u32, u32>>,
}

impl SplatRasterPass {
    /// Create the raster pass.
    ///
    /// Matches the `MarchFactory` signature: receives device, scene, gbuffer,
    /// tile_cull (ignored), material_buffer, shader_params, opacity_code (ignored).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &rkf_render::GpuScene,
        _gbuffer: &rkf_render::GBuffer,
        _tile_cull: &rkf_render::TileObjectCullPass,
        _material_buffer: &wgpu::Buffer,
        _shader_params: &rkf_render::ShaderParamsBuffer,
        _opacity_code: &str,
    ) -> Self {
        let emit = SplatEmitPass::new(device, &scene.bind_group_layout);
        let raster = SplatRasterPipeline::new(
            device,
            &scene.bind_group_layout,
            &emit,
        );
        let octree = OctreeGpu::new();

        Self {
            emit,
            raster,
            octree,
            voxel_pool: rkp_core::VoxelPool::new(1_000_000),
            rkp_renderer: None,
            pending_faces: std::cell::RefCell::new(Vec::new()),
            device: device.clone(),
            queue: queue.clone(),
            faces_dirty: std::cell::Cell::new(false),
            object_gpu_mapping: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Create the raster pass with the full standalone pipeline (shadow/AO + shading).
    /// Uses RkpScene's bind group layout. No rkf-render dependency for scene data.
    pub fn new_standalone(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let renderer = crate::rkp_renderer::RkpRenderer::new(device, width, height);
        let emit = SplatEmitPass::new(device, &renderer.scene.bind_group_layout);
        let raster = SplatRasterPipeline::new(device, &renderer.scene.bind_group_layout, &emit);
        let octree = OctreeGpu::new();

        Self {
            emit,
            raster,
            octree,
            voxel_pool: rkp_core::VoxelPool::new(1_000_000),
            rkp_renderer: Some(std::cell::RefCell::new(renderer)),
            pending_faces: std::cell::RefCell::new(Vec::new()),
            device: device.clone(),
            queue: queue.clone(),
            faces_dirty: std::cell::Cell::new(false),
            object_gpu_mapping: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// Access the octree GPU manager (for external octree allocation/upload).
    pub fn octree_gpu(&self) -> &OctreeGpu {
        &self.octree
    }

    /// Mutable access to the octree GPU manager.
    pub fn octree_gpu_mut(&mut self) -> &mut OctreeGpu {
        &mut self.octree
    }

}

impl SplatRasterPass {
    /// Emit face instances from a per-voxel octree.
    ///
    /// Traverses the octree nodes. Each leaf IS a single voxel. For each
    /// non-empty leaf, checks 6 neighbors via octree traversal and emits
    /// exposed faces. No bricks, no within-brick indexing.
    fn emit_faces_from_octree(
        &self,
        octree: &rkp_core::SparseOctree,
        pool: &rkp_core::VoxelPool,
        obj_idx: u32,
    ) {
        let mut faces = self.pending_faces.borrow_mut();
        let base_vs = octree.base_voxel_size();

        for (coord, slot, leaf_depth) in octree.iter_leaves() {
            let sv = pool.get(slot);
            if sv.opacity_f32() <= 0.01 {
                continue;
            }

            let depth_diff = octree.depth() - leaf_depth;
            let leaf_vs = base_vs * (1u32 << depth_diff) as f32;

            // Voxel center in octree space (0-based).
            let center = glam::Vec3::new(
                coord.x as f32 * base_vs + leaf_vs * 0.5,
                coord.y as f32 * base_vs + leaf_vs * 0.5,
                coord.z as f32 * base_vs + leaf_vs * 0.5,
            );

            // Check 6 neighbors via octree lookup.
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
                        None => true, // out of bounds
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
                    faces.push(crate::splat_emit::FaceInstance {
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

    /// Dispatch the raster pass using RkpScene's bind group. No rkf-render GpuScene dependency.
    pub fn dispatch_standalone(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &crate::rkp_scene::RkpScene,
        gbuffer: &rkf_render::GBuffer,
    ) {
        // Upload face instances.
        {
            let faces = self.pending_faces.borrow();
            if self.faces_dirty.get() && !faces.is_empty() {
                let mapping = self.object_gpu_mapping.borrow();
                let mut upload_faces = faces.clone();
                for face in upload_faces.iter_mut() {
                    let sdf_obj_id = (face.packed >> 3) & 0xFFFFF;
                    let gpu_idx = mapping.get(&sdf_obj_id).copied().unwrap_or(0);
                    face.packed = (face.packed & 0x7) | ((gpu_idx & 0xFFFFF) << 3);
                }
                drop(mapping);

                let face_count = upload_faces.len() as u32;
                let face_bytes: &[u8] = bytemuck::cast_slice(&upload_faces);

                {
                    let buf = self.emit.face_buffer.borrow();
                    if face_bytes.len() as u64 > buf.size() {
                        let new_size = (face_bytes.len() as u64).max(buf.size() * 2);
                        drop(buf);
                        let new_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("emit face instances"),
                            size: new_size,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        *self.raster.face_bind_group.borrow_mut() = self.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some("raster face bind group"),
                                layout: &self.raster.face_bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: new_buf.as_entire_binding(),
                                }],
                            },
                        );
                        *self.emit.face_buffer.borrow_mut() = new_buf;
                    }
                }

                let face_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("face staging"),
                    size: face_bytes.len() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                face_staging.slice(..).get_mapped_range_mut()
                    .copy_from_slice(face_bytes);
                face_staging.unmap();
                encoder.copy_buffer_to_buffer(
                    &face_staging, 0,
                    &self.emit.face_buffer.borrow(), 0,
                    face_bytes.len() as u64,
                );

                let draw_args: [u32; 4] = [6, face_count, 0, 0];
                let args_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("indirect args staging"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                args_staging.slice(..).get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&draw_args));
                args_staging.unmap();
                encoder.copy_buffer_to_buffer(
                    &args_staging, 0,
                    &self.emit.indirect_buffer, 0,
                    16,
                );

                self.faces_dirty.set(false);
            }
        }

        // Raster pass with RkpScene bind group.
        {
            let mut render_pass = SplatRasterPipeline::begin_render_pass(encoder, gbuffer);
            self.raster.draw(
                &mut render_pass,
                &scene.bind_group,
                &self.emit.indirect_buffer,
            );
        }
    }

    /// Emit face instances from raw octree node data + voxel pool (for file loading).
    fn emit_faces_from_raw_octree(
        &self,
        nodes: &[u32],
        depth: u8,
        base_vs: f32,
        pool: &rkp_core::VoxelPool,
        obj_idx: u32,
    ) {
        let octree = rkp_core::SparseOctree::from_raw(nodes, depth, base_vs);
        self.emit_faces_from_octree(&octree, pool, obj_idx);
    }
}

impl rkf_render::MarchPass for SplatRasterPass {
    fn spatial_data(&self) -> &[u32] {
        self.octree.data()
    }

    fn frame_data(&self) -> Option<rkf_render::MarchFrameData<'_>> {
        if self.rkp_renderer.is_some() {
            // Standalone pipeline: data uploaded to RkpScene in prepare(), not GpuScene.
            None
        } else {
            Some(rkf_render::MarchFrameData {
                pool_data: self.voxel_pool.as_bytes(),
                color_data: self.voxel_pool.color_bytes(),
            })
        }
    }

    fn write_spatial_fields(
        &self,
        handle: &rkf_core::scene_node::SpatialHandle,
        gpu_obj: &mut rkf_render::gpu_object::GpuObject,
    ) {
        if let rkf_core::scene_node::SpatialHandle::Octree {
            root_offset, depth, base_voxel_size, ..
        } = handle
        {
            let extent = (1u32 << depth) as f32 * base_voxel_size;
            gpu_obj.brick_map_offset = *root_offset;
            gpu_obj.brick_map_dims = [*depth as u32, extent.to_bits(), 0];
        }

    }

    #[allow(clippy::type_complexity)]
    fn load_asset(
        &mut self,
        path: &str,
        _pool: &mut rkf_core::brick_pool::BrickPool,
        _object_id: Option<u32>,
    ) -> Result<
        Option<(
            rkf_core::scene_node::SpatialHandle,
            f32,
            rkf_core::Aabb,
            u32,
            Vec<(u32, rkf_core::companion::ColorBrick)>,
            Vec<(u32, rkf_core::companion::BoneBrick)>,
        )>,
        String,
    > {
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
                    return Ok(None);
                }
            }
        };

        if !rkp_path.exists() {
            return Ok(None);
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

        // Allocate voxels in pool
        if self.voxel_pool.free_count() < voxel_count {
            let new_cap = (self.voxel_pool.capacity() * 2).max(self.voxel_pool.capacity() + voxel_count);
            self.voxel_pool.grow(new_cap);
        }
        let slots = self.voxel_pool.allocate_range(voxel_count)
            .ok_or_else(|| "failed to allocate voxel pool slots".to_string())?;

        // Copy voxel data into pool (1 VoxelSample per slot)
        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        for (i, &slot) in slots.iter().enumerate() {
            let src_offset = i * bytes_per_voxel;
            if src_offset + bytes_per_voxel > voxel_data.len() {
                break;
            }
            let vs: &VoxelSample =
                bytemuck::from_bytes(&voxel_data[src_offset..src_offset + bytes_per_voxel]);
            *self.voxel_pool.get_mut(slot) = rkp_core::SplatVoxel::from(*vs);
        }

        // Remap octree leaf slots to newly allocated pool slots.
        let mut remapped_nodes = octree_nodes.clone();
        for node in &mut remapped_nodes {
            if *node != rkp_core::sparse_octree::EMPTY_NODE
                && *node != rkp_core::sparse_octree::INTERIOR_NODE
                && (*node & rkp_core::sparse_octree::LEAF_BIT) != 0
            {
                let old_slot = *node & !rkp_core::sparse_octree::LEAF_BIT;
                if (old_slot as usize) < slots.len() {
                    *node = rkp_core::sparse_octree::make_leaf(slots[old_slot as usize]);
                }
            }
        }

        let octree_depth = header.octree_depth as u8;
        let handle = self.octree.allocate_raw(&remapped_nodes, octree_depth, voxel_size);

        // Emit faces via octree-based neighbor lookup.
        let obj_id_for_faces = _object_id.unwrap_or(0);
        eprintln!("[SplatRasterPass] emitting faces with sdf_object_id={}", obj_id_for_faces);
        self.emit_faces_from_raw_octree(
            &remapped_nodes,
            octree_depth,
            voxel_size,
            &self.voxel_pool,
            obj_id_for_faces,
        );
        self.faces_dirty.set(true);

        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        // Load per-voxel color data into VoxelPool's parallel color array.
        let has_color = header.flags & rkp_core::asset_file::FLAG_HAS_COLOR != 0;
        if has_color {
            let color_bytes = rkp_core::asset_file::read_rkp_color(&mut reader, &header)
                .unwrap_or_default();
            let color_u32s: &[u32] = if color_bytes.len() >= 4 {
                bytemuck::cast_slice(&color_bytes)
            } else {
                &[]
            };
            let mut nonzero = 0u32;
            let mut samples = Vec::new();
            for (i, &slot) in slots.iter().enumerate() {
                if i < color_u32s.len() {
                    let packed = color_u32s[i];
                    self.voxel_pool.set_color(slot, packed);
                    if packed != 0 {
                        nonzero += 1;
                        if samples.len() < 3 {
                            let r = packed & 0xFF;
                            let g = (packed >> 8) & 0xFF;
                            let b = (packed >> 16) & 0xFF;
                            let a = (packed >> 24) & 0xFF;
                            samples.push(format!("slot{}=({},{},{},{})", slot, r, g, b, a));
                        }
                    }
                }
            }
            eprintln!(
                "[SplatRasterPass] loaded color: {} u32s, {} nonzero, samples: [{}]",
                color_u32s.len(), nonzero, samples.join(", "),
            );
        } else {
            eprintln!(
                "[SplatRasterPass] no color data in .rkp (flags={:#x}, color_compressed_size={})",
                header.flags, header.color_compressed_size,
            );
        }

        // Return empty companion pairs — color is managed by VoxelPool directly.
        let color_pairs = Vec::new();
        let bone_pairs = Vec::new();

        {
            let faces = self.pending_faces.borrow();
            // Debug: compare face range (octree space) vs returned AABB (local space).
            if !faces.is_empty() {
                let mut min_pos = glam::Vec3::splat(f32::MAX);
                let mut max_pos = glam::Vec3::splat(f32::MIN);
                for f in faces.iter() {
                    let p = glam::Vec3::new(f.pos_x, f.pos_y, f.pos_z);
                    min_pos = min_pos.min(p);
                    max_pos = max_pos.max(p);
                }
                let extent = (1u32 << octree_depth) as f32 * voxel_size;
                eprintln!(
                    "[SplatRasterPass] face octree range: [{:.4},{:.4},{:.4}]→[{:.4},{:.4},{:.4}]",
                    min_pos.x, min_pos.y, min_pos.z, max_pos.x, max_pos.y, max_pos.z,
                );
                eprintln!(
                    "[SplatRasterPass] extent={:.4}, local range (face - extent/2): [{:.4},{:.4},{:.4}]→[{:.4},{:.4},{:.4}]",
                    extent,
                    min_pos.x - extent * 0.5, min_pos.y - extent * 0.5, min_pos.z - extent * 0.5,
                    max_pos.x - extent * 0.5, max_pos.y - extent * 0.5, max_pos.z - extent * 0.5,
                );
                eprintln!(
                    "[SplatRasterPass] returned AABB: [{:.4},{:.4},{:.4}]→[{:.4},{:.4},{:.4}]",
                    aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z,
                );
            }
            eprintln!(
                "[SplatRasterPass] loaded .rkp: {} voxels, {} faces, {} octree nodes",
                voxel_count, faces.len(), remapped_nodes.len(),
            );
        }

        Ok(Some((spatial, voxel_size, aabb, voxel_count, color_pairs, bone_pairs)))
    }

    fn deallocate_spatial(&mut self, handle: &rkf_core::scene_node::SpatialHandle) {
        if let rkf_core::scene_node::SpatialHandle::Octree {
            root_offset, len, depth, base_voxel_size,
        } = handle
        {
            self.octree.deallocate(rkp_core::OctreeHandle {
                root_offset: *root_offset,
                len: *len,
                depth: *depth,
                base_voxel_size: *base_voxel_size,
            });
        }
    }

    fn set_object_gpu_index(&self, object_id: u32, gpu_index: u32) {
        eprintln!("[SplatRasterPass] set_object_gpu_index: sdf_obj_id={} → gpu_idx={}", object_id, gpu_index);
        let mut mapping = self.object_gpu_mapping.borrow_mut();
        let old = mapping.insert(object_id, gpu_index);
        if old != Some(gpu_index) {
            self.faces_dirty.set(true);
        }
    }

    fn prepare(&self, _queue: &wgpu::Queue) {
        // Data upload to RkpScene happens in dispatch() via staging buffers,
        // since prepare() doesn't have device access for buffer growth.
        // Shadow/AO params are also set in dispatch().
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, ctx: &rkf_render::MarchContext) {
        // Upload face instances from CPU emit (replaces GPU emit pass).
        {
            let faces = self.pending_faces.borrow();
            if self.faces_dirty.get() && !faces.is_empty() {
                // Build upload buffer with GPU object indices patched in.
                // Don't modify pending_faces — the sdf_object_id must be preserved
                // for future re-uploads when the mapping changes.
                let mapping = self.object_gpu_mapping.borrow();
                let mut upload_faces = faces.clone();
                let mut remap_counts: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
                for face in upload_faces.iter_mut() {
                    let sdf_obj_id = (face.packed >> 3) & 0xFFFFF;
                    let gpu_idx = mapping.get(&sdf_obj_id).copied().unwrap_or(0);
                    *remap_counts.entry((sdf_obj_id, gpu_idx)).or_default() += 1;
                    face.packed = (face.packed & 0x7) | ((gpu_idx & 0xFFFFF) << 3);
                }
                let num_objects = ctx.scene.num_objects();
                for ((sdf_id, gpu_id), count) in &remap_counts {
                    eprintln!("[SplatRasterPass] dispatch remap: sdf_obj_id={} → gpu_idx={} ({} faces), scene has {} GPU objects", sdf_id, gpu_id, count, num_objects);
                }
                drop(mapping);

                let face_count = upload_faces.len() as u32;
                let face_bytes: &[u8] = bytemuck::cast_slice(&upload_faces);

                // Grow face buffer if needed.
                {
                    let buf = self.emit.face_buffer.borrow();
                    if face_bytes.len() as u64 > buf.size() {
                        let new_size = (face_bytes.len() as u64).max(buf.size() * 2);
                        drop(buf);
                        let new_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("emit face instances"),
                            size: new_size,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        *self.raster.face_bind_group.borrow_mut() = self.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some("raster face bind group"),
                                layout: &self.raster.face_bind_group_layout,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: new_buf.as_entire_binding(),
                                }],
                            },
                        );
                        *self.emit.face_buffer.borrow_mut() = new_buf;
                    }
                }

                // Upload face data via staging buffer.
                let face_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("face staging"),
                    size: face_bytes.len() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                face_staging.slice(..).get_mapped_range_mut()
                    .copy_from_slice(face_bytes);
                face_staging.unmap();
                encoder.copy_buffer_to_buffer(
                    &face_staging, 0,
                    &self.emit.face_buffer.borrow(), 0,
                    face_bytes.len() as u64,
                );

                // Set indirect draw args: vertex_count=6, instance_count=face_count.
                let draw_args: [u32; 4] = [6, face_count, 0, 0];
                let args_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("indirect args staging"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                args_staging.slice(..).get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&draw_args));
                args_staging.unmap();
                encoder.copy_buffer_to_buffer(
                    &args_staging, 0,
                    &self.emit.indirect_buffer, 0,
                    16,
                );

                self.faces_dirty.set(false);
                // Keep faces around (don't clear) in case we need them later.
            }
        }


        // 2. Raster + (optionally) shadow/AO + shading.
        if let Some(ref rkp_cell) = self.rkp_renderer {
            let mut rkp = rkp_cell.borrow_mut();

            // Upload geometry data (voxels, octree, color) — only what changed.
            {
                use crate::rkp_scene::GeometryUpload;
                let octree_data = self.octree.data();
                let geo = GeometryUpload {
                    voxel_pool: self.voxel_pool.as_bytes(),
                    octree_nodes: bytemuck::cast_slice(octree_data),
                    color_pool: self.voxel_pool.color_bytes(),
                };
                rkp.scene.upload_geometry(&self.device, &self.queue, &geo);
            }

            // Use the engine's object buffer directly — it already has correct
            // world transforms, octree params, and animation data.
            rkp.scene.set_external_objects_buffer(&self.device, &ctx.scene.object_buffer);

            // Copy camera from engine (GPU→GPU).
            rkp.scene.copy_camera_from(encoder, &ctx.scene.camera_buffer);

            // Wire G-buffer + HDR output.
            rkp.set_gbuffer(
                &ctx.gbuffer.position_view,
                &ctx.gbuffer.normal_view,
                &ctx.gbuffer.material_view,
            );
            if let Some(hdr_view) = ctx.hdr_output_view {
                rkp.shade.set_output_view(&self.device, hdr_view);
            }

            // Render: raster → shadow/AO → shade.
            {
                let mut render_pass =
                    SplatRasterPipeline::begin_render_pass(encoder, ctx.gbuffer);
                self.raster.draw(
                    &mut render_pass,
                    &rkp.scene.bind_group,
                    &self.emit.indirect_buffer,
                );
            }
            let shadow_params = crate::rkp_shadow_ao::ShadowAoParams::default();
            rkp.shadow_ao.update_params(&self.queue, &shadow_params);
            rkp.shadow_ao.dispatch(encoder, &rkp.scene);
            rkp.shade.dispatch(encoder);
        } else {
            // Legacy path: raster with engine's GpuScene bind group.
            let mut render_pass =
                SplatRasterPipeline::begin_render_pass(encoder, ctx.gbuffer);
            self.raster.draw(
                &mut render_pass,
                &ctx.scene.bind_group,
                &self.emit.indirect_buffer,
            );
        }
    }

    fn recreate_pipeline(&mut self, _device: &wgpu::Device, _module: &wgpu::ShaderModule) {
        // The raster pipeline doesn't use the composed shader module from the
        // engine (that's for the march shader). For hot-reload, we'd recreate
        // the raster pipeline from its own source. For now, no-op.
    }

    fn handles_full_pipeline(&self) -> bool {
        self.rkp_renderer.is_some()
    }

    fn needs_sdf_recompile(&self) -> bool {
        false
    }

    fn needs_skin_deform(&self) -> bool {
        true // Skinned objects need the SkinDeformPass for bone weight scatter.
    }

    fn transform_brick(&self, brick: &mut rkf_core::brick::Brick, voxel_size: f32) {
        // Same transform as SplatMarchPass: convert SDF distance to opacity.
        // The brick data is loaded from .rkf files which store SDF distances.
        use rkf_core::constants::BRICK_DIM;

        fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }

        let fade_inner = voxel_size;
        let fade_outer = voxel_size * 3.0;

        for vz in 0..BRICK_DIM {
            for vy in 0..BRICK_DIM {
                for vx in 0..BRICK_DIM {
                    let sample = brick.sample(vx, vy, vz);
                    let sv = rkp_core::SplatVoxel::from(sample);
                    let d = sv.opacity_f32(); // In .rkf files, word0 stores SDF distance
                    let opacity = 1.0 - smoothstep(-fade_inner, fade_outer, d);
                    let mut new_sv = sv;
                    new_sv.set_opacity(opacity.clamp(0.0, 1.0));
                    brick.set(vx, vy, vz, new_sv.into());
                }
            }
        }
    }

    fn voxelize_primitive(
        &mut self,
        primitive: &rkf_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        _pool: &mut rkf_core::brick_pool::BrickPool,
        object_id: u32,
    ) -> Option<(rkf_core::scene_node::SpatialHandle, f32, rkf_core::Aabb, u32)> {
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

        let fade_inner = voxel_size;
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
        let (octree, voxel_count, _grid_origin) =
            rkp_core::voxelize_octree::voxelize_opacity_octree(
                opacity_fn, &aabb, voxel_size, &mut self.voxel_pool,
            )?;

        // Emit face instances via octree neighbor lookup.
        self.emit_faces_from_octree(&octree, &self.voxel_pool, object_id);
        self.faces_dirty.set(true);

        // Allocate octree into the GPU allocator.
        let handle = self.octree.allocate(&octree);
        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        let geometry_aabb = rkf_core::Aabb::new(-half_extents, half_extents);
        Some((spatial, voxel_size, geometry_aabb, voxel_count))
    }

    fn shader_overrides(&self) -> rkf_render::ShaderOverrides {
        // Same overrides as SplatMarchPass — the shadow/AO/GI/shading passes
        // still use opacity-field sampling. These will be updated with octree
        // traversal in Phase 7b (shader override updates).
        rkf_render::ShaderOverrides {
            shadow_ao: Some(include_str!("shaders/opacity_shadow_ao.wgsl").to_string()),
            radiance_inject: Some(include_str!("shaders/opacity_radiance_inject.wgsl").to_string()),
            shade_common: Some(include_str!("shaders/opacity_shade_common.wgsl").to_string()),
            shade_common_shading: Some(
                include_str!("shaders/opacity_shade_common_shading.wgsl").to_string(),
            ),
            shade_main: Some(include_str!("shaders/opacity_shade_main.wgsl").to_string()),
            shade_models: vec![(
                "pbr".into(),
                include_str!("shaders/opacity_shade_pbr.wgsl").to_string(),
            )],
        }
    }

    fn march_source(&self) -> Option<&str> {
        // No march shader — rasterization doesn't use composed shader injection.
        None
    }

    fn handles_opacity_volumes(&self) -> bool {
        true
    }

    fn volume_gpu_objects(
        &self,
    ) -> Vec<(rkf_render::gpu_object::GpuObject, glam::Vec3, glam::Vec3)> {
        // No procedural volumes — all volumes are voxelized into octree bricks.
        Vec::new()
    }
}
