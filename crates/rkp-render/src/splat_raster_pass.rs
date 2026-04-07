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
use crate::surface_shell_gpu::SurfaceShellGpu;

/// Forward rasterization march pass — replaces the compute ray march.
pub struct SplatRasterPass {
    emit: SplatEmitPass,
    raster: SplatRasterPipeline,
    shell: SurfaceShellGpu,
    octree: OctreeGpu,
    /// Pending surface shell uploads (slot, occupancy) — flushed in dispatch via staging.
    pending_shell_uploads: std::cell::RefCell<Vec<(u32, [u64; 8])>>,
    /// Face instances from CPU-side emit. The packed field stores sdf_object_id
    /// (not GPU index) — corrected during upload using object_gpu_mapping.
    pending_faces: std::cell::RefCell<Vec<crate::splat_emit::FaceInstance>>,
    /// The device, needed for creating staging buffers in dispatch.
    device: wgpu::Device,
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
        scene: &rkf_render::GpuScene,
        _gbuffer: &rkf_render::GBuffer,
        _tile_cull: &rkf_render::TileObjectCullPass,
        _material_buffer: &wgpu::Buffer,
        _shader_params: &rkf_render::ShaderParamsBuffer,
        _opacity_code: &str,
    ) -> Self {
        let shell = SurfaceShellGpu::new(device, 1024);
        let emit = SplatEmitPass::new(device, &scene.bind_group_layout, &shell);
        let raster = SplatRasterPipeline::new(
            device,
            &scene.bind_group_layout,
            &shell,
            &emit,
        );
        let octree = OctreeGpu::new();

        Self {
            emit,
            raster,
            shell,
            octree,
            pending_shell_uploads: std::cell::RefCell::new(Vec::new()),
            pending_faces: std::cell::RefCell::new(Vec::new()),
            device: device.clone(),
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

    /// Access the surface shell GPU buffer.
    pub fn surface_shell(&self) -> &SurfaceShellGpu {
        &self.shell
    }

    /// Mutable access to the surface shell.
    pub fn surface_shell_mut(&mut self) -> &mut SurfaceShellGpu {
        &mut self.shell
    }
}

impl SplatRasterPass {
    /// Emit face instances from raw octree data (for file loading).
    ///
    /// Traverses the octree nodes, and for each leaf, reads occupancy from
    /// the brick pool and emits exposed face instances.
    fn emit_faces_from_octree_data(
        &self,
        nodes: &[u32],
        depth: u8,
        extent: f32,
        base_vs: f32,
        pool: &rkf_core::brick_pool::BrickPool,
        obj_idx: u32,
    ) {
        let mut faces = self.pending_faces.borrow_mut();

        // Stack-based traversal (mirrors the GPU emit but on CPU).
        struct Entry {
            node_idx: usize,
            center: glam::Vec3,
            half_extent: f32,
            level: u8,
        }

        let half = extent * 0.5;
        let mut stack = vec![Entry {
            node_idx: 0,
            center: glam::Vec3::splat(half),
            half_extent: half,
            level: 0,
        }];

        while let Some(entry) = stack.pop() {
            if entry.node_idx >= nodes.len() {
                continue;
            }
            let node = nodes[entry.node_idx];

            if node == rkp_core::sparse_octree::EMPTY_NODE
                || node == rkp_core::sparse_octree::INTERIOR_NODE
            {
                continue;
            }

            if rkp_core::sparse_octree::is_leaf(node) {
                let slot = rkp_core::sparse_octree::leaf_slot(node);
                let depth_diff = depth - entry.level;
                let leaf_vs = base_vs * (1u32 << depth_diff) as f32;

                let brick_origin = glam::Vec3::new(
                    entry.center.x - entry.half_extent,
                    entry.center.y - entry.half_extent,
                    entry.center.z - entry.half_extent,
                );

                // Build occupancy from brick pool data
                let brick = pool.get(slot);
                let mut geo = rkf_core::BrickGeometry::new();
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            let sample = brick.sample(vx as u32, vy as u32, vz as u32);
                            let sv = rkp_core::SplatVoxel::from(sample);
                            geo.set_solid(vx, vy, vz, sv.opacity_f32() > 0.5);
                        }
                    }
                }

                // Emit faces
                for vz in 0..8u32 {
                    for vy in 0..8u32 {
                        for vx in 0..8u32 {
                            if !geo.is_solid(vx as u8, vy as u8, vz as u8) {
                                continue;
                            }
                            let voxel_idx = vx + vy * 8 + vz * 64;
                            let center = brick_origin
                                + (glam::Vec3::new(vx as f32, vy as f32, vz as f32) + 0.5)
                                    * leaf_vs;

                            for face in 0..6u32 {
                                let (nx, ny, nz) = match face {
                                    0 => (-1i32, 0, 0),
                                    1 => (1, 0, 0),
                                    2 => (0, -1, 0),
                                    3 => (0, 1, 0),
                                    4 => (0, 0, -1),
                                    5 => (0, 0, 1),
                                    _ => unreachable!(),
                                };
                                let nbx = vx as i32 + nx;
                                let nby = vy as i32 + ny;
                                let nbz = vz as i32 + nz;

                                let exposed = if nbx < 0 || nbx >= 8 || nby < 0 || nby >= 8 || nbz < 0 || nbz >= 8 {
                                    true
                                } else {
                                    !geo.is_solid(nbx as u8, nby as u8, nbz as u8)
                                };

                                if exposed {
                                    faces.push(crate::splat_emit::FaceInstance {
                                        pos_x: center.x,
                                        pos_y: center.y,
                                        pos_z: center.z,
                                        voxel_size: leaf_vs,
                                        brick_slot: slot,
                                        packed: (voxel_idx & 0x1FF)
                                            | ((face & 0x7) << 9)
                                            | ((obj_idx & 0xFFFF) << 12),
                                    });
                                }
                            }
                        }
                    }
                }
                continue;
            }

            if rkp_core::sparse_octree::is_branch(node) {
                let children_offset = node as usize;
                let child_half = entry.half_extent * 0.5;
                for octant in 0..8u32 {
                    let dx = (octant & 1) as f32;
                    let dy = ((octant >> 1) & 1) as f32;
                    let dz = ((octant >> 2) & 1) as f32;
                    let child_center = glam::Vec3::new(
                        entry.center.x + (dx * 2.0 - 1.0) * child_half,
                        entry.center.y + (dy * 2.0 - 1.0) * child_half,
                        entry.center.z + (dz * 2.0 - 1.0) * child_half,
                    );
                    stack.push(Entry {
                        node_idx: children_offset + octant as usize,
                        center: child_center,
                        half_extent: child_half,
                        level: entry.level + 1,
                    });
                }
            }
        }
    }
}

impl rkf_render::MarchPass for SplatRasterPass {
    fn spatial_data(&self) -> &[u32] {
        let data = self.octree.data();
        if !data.is_empty() {
            eprintln!("[SplatRasterPass] spatial_data: {} u32s", data.len());
        }
        data
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
            let extent = (1u32 << depth) as f32 * 8.0 * base_voxel_size;
            gpu_obj.brick_map_offset = *root_offset;
            gpu_obj.brick_map_dims = [*depth as u32, extent.to_bits(), 0];
        }
    }

    #[allow(clippy::type_complexity)]
    fn load_asset(
        &mut self,
        path: &str,
        pool: &mut rkf_core::brick_pool::BrickPool,
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
        use rkf_core::companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel};
        use rkf_core::voxel::VoxelSample;

        // Check for .rkp extension
        if !path.ends_with(".rkp") {
            return Ok(None); // Not our format — fall back to default loading
        }

        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("open {path}: {e}"))?;
        let mut reader = std::io::BufReader::new(&mut file);

        let header = rkp_core::asset_file::read_rkp_header(&mut reader)
            .map_err(|e| format!("read .rkp header: {e}"))?;

        let octree_nodes = rkp_core::asset_file::read_rkp_octree(&mut reader, &header)
            .map_err(|e| format!("read octree: {e}"))?;

        let brick_data = rkp_core::asset_file::read_rkp_bricks(&mut reader, &header)
            .map_err(|e| format!("read bricks: {e}"))?;

        let geometry_data = rkp_core::asset_file::read_rkp_geometry(&mut reader, &header)
            .map_err(|e| format!("read geometry: {e}"))?;

        let voxel_size = header.base_voxel_size;
        let brick_count = header.brick_count;
        let aabb = rkf_core::Aabb::new(
            glam::Vec3::from(header.aabb_min),
            glam::Vec3::from(header.aabb_max),
        );

        // Allocate bricks in pool
        if pool.free_count() < brick_count {
            let new_cap = (pool.capacity() * 2).max(pool.capacity() + brick_count);
            pool.grow(new_cap);
        }
        let slots = pool.allocate_range(brick_count)
            .ok_or_else(|| "failed to allocate brick pool slots".to_string())?;

        // Copy brick voxel data into pool
        let voxels_per_brick = 512usize;
        let bytes_per_voxel = std::mem::size_of::<VoxelSample>();
        let bytes_per_brick = voxels_per_brick * bytes_per_voxel;

        for (i, &slot) in slots.iter().enumerate() {
            let src_offset = i * bytes_per_brick;
            if src_offset + bytes_per_brick > brick_data.len() {
                break;
            }
            let src_voxels: &[VoxelSample] =
                bytemuck::cast_slice(&brick_data[src_offset..src_offset + bytes_per_brick]);
            let brick = pool.get_mut(slot);
            brick.voxels.copy_from_slice(src_voxels);
        }

        // Build SparseOctree from the raw nodes, remapping leaf slots to the
        // newly allocated pool slots.
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

        // Allocate the octree into our OctreeAllocator
        // (We need to create a SparseOctree from the remapped nodes)
        let mut octree = rkp_core::SparseOctree::new(header.octree_depth as u8, voxel_size);
        // Replace the octree's internal nodes with our remapped data.
        // Since SparseOctree::nodes is private, we rebuild via from_raw.
        // For now, allocate directly into the allocator from raw data.
        let handle = self.octree.allocate_raw(&remapped_nodes, header.octree_depth as u8, voxel_size);

        // Build surface shell + face instances for each brick
        let occupancy_stride = 8usize; // 8 u64s per brick = 64 bytes
        let base_vs = voxel_size;
        let octree_depth = header.octree_depth as u8;
        let octree_extent = (1u32 << octree_depth) as f32 * 8.0 * base_vs;

        {
            let mut shells = self.pending_shell_uploads.borrow_mut();
            let mut faces = self.pending_faces.borrow_mut();

            for (i, &slot) in slots.iter().enumerate() {
                // Read occupancy from geometry data
                let geo_offset = i * occupancy_stride * 8; // 8 u64s * 8 bytes each = 64 bytes
                let mut occupancy = [0u64; 8];
                if geo_offset + 64 <= geometry_data.len() {
                    let occ_bytes = &geometry_data[geo_offset..geo_offset + 64];
                    occupancy.copy_from_slice(bytemuck::cast_slice(occ_bytes));
                }

                shells.push((slot, occupancy));
                self.shell.ensure_capacity(&self.device, slot + 1);

                // Build face instances (same logic as voxelize_primitive)
                // We need the brick's position in octree space, but for file-loaded
                // assets we'd need to reconstruct it from the octree. For now, iterate
                // the octree leaves to find this slot's position.
                // TODO: store leaf positions alongside brick data for faster loading.
            }
        }

        // To emit faces, we need each brick's octree-space position. Iterate the
        // octree to find leaf positions. Build a slot→position map.
        // For now, use a simpler approach: reconstruct from the allocator's data.
        // Actually, we need to iterate the raw octree to find leaf positions.
        self.emit_faces_from_octree_data(
            &remapped_nodes,
            octree_depth,
            octree_extent,
            base_vs,
            pool,
            _object_id.unwrap_or(0),
        );

        self.faces_dirty.set(true);

        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        // Parse color bricks
        let mut color_pairs = Vec::new();
        if header.flags & rkp_core::asset_file::FLAG_HAS_COLOR != 0 {
            // Read color section
            let color_bytes = rkp_core::asset_file::read_rkp_color(&mut reader, &header)
                .unwrap_or_default();
            let cb_size = std::mem::size_of::<ColorBrick>();
            for (i, &slot) in slots.iter().enumerate() {
                let offset = i * cb_size;
                if offset + cb_size <= color_bytes.len() {
                    let cb: ColorBrick = *bytemuck::from_bytes(&color_bytes[offset..offset + cb_size]);
                    color_pairs.push((slot, cb));
                }
            }
        }

        // Parse bone bricks
        let mut bone_pairs = Vec::new();
        if header.flags & rkp_core::asset_file::FLAG_HAS_BONES != 0 {
            let bone_bytes = rkp_core::asset_file::read_rkp_bones(&mut reader, &header)
                .unwrap_or_default();
            let bb_size = std::mem::size_of::<BoneBrick>();
            for (i, &slot) in slots.iter().enumerate() {
                let offset = i * bb_size;
                if offset + bb_size <= bone_bytes.len() {
                    let bb: BoneBrick = *bytemuck::from_bytes(&bone_bytes[offset..offset + bb_size]);
                    bone_pairs.push((slot, bb));
                }
            }
        }

        {
            let faces = self.pending_faces.borrow();
            let face_count = faces.len();
            if face_count > 0 {
                let mut min_pos = glam::Vec3::splat(f32::MAX);
                let mut max_pos = glam::Vec3::splat(f32::MIN);
                for f in faces.iter() {
                    min_pos = min_pos.min(glam::Vec3::new(f.pos_x, f.pos_y, f.pos_z));
                    max_pos = max_pos.max(glam::Vec3::new(f.pos_x, f.pos_y, f.pos_z));
                }
                eprintln!(
                    "[SplatRasterPass] loaded .rkp: {} bricks, {} faces, octree_pos range [{:.3},{:.3},{:.3}]→[{:.3},{:.3},{:.3}]",
                    brick_count, face_count, min_pos.x, min_pos.y, min_pos.z, max_pos.x, max_pos.y, max_pos.z,
                );
            }
        }
        eprintln!(
            "[SplatRasterPass] loaded .rkp: {} octree nodes, {} colors, {} bones, extent={}",
            remapped_nodes.len(), color_pairs.len(), bone_pairs.len(),
            (1u32 << octree_depth) as f32 * 8.0 * base_vs,
        );

        Ok(Some((spatial, voxel_size, aabb, brick_count, color_pairs, bone_pairs)))
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
        let mut mapping = self.object_gpu_mapping.borrow_mut();
        let old = mapping.insert(object_id, gpu_index);
        if old != Some(gpu_index) {
            // Mapping changed — faces need re-upload with corrected obj_idx.
            self.faces_dirty.set(true);
        }
    }

    fn prepare(&self, queue: &wgpu::Queue) {
        // Flush pending surface shell uploads.
        let mut pending = self.pending_shell_uploads.borrow_mut();
        eprintln!("[SplatRasterPass] prepare: pending={}", pending.len());
        if !pending.is_empty() {
            eprintln!("[SplatRasterPass] prepare: flushing {} shell uploads", pending.len());
            for &(slot, ref occupancy) in pending.iter() {
                self.shell.upload_slot(queue, slot, occupancy);
            }
            pending.clear();
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, ctx: &rkf_render::MarchContext) {
        let object_count = ctx.scene.num_objects() as u32;

        // Flush pending surface shell uploads via mapped staging buffer + copy.
        {
            let mut pending = self.pending_shell_uploads.borrow_mut();
            if !pending.is_empty() {
                // Each entry is (slot, [u64; 8]) = 64 bytes of occupancy per slot.
                let entry_bytes = 64usize; // 8 * size_of::<u64>()
                let total_bytes = pending.len() * entry_bytes;

                let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shell staging"),
                    size: total_bytes as u64,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                {
                    let mut view = staging.slice(..).get_mapped_range_mut();
                    for (i, &(slot, ref occupancy)) in pending.iter().enumerate() {
                        // Write each occupancy to the correct offset for its slot
                        // in the staging buffer. We'll copy per-entry to the shell buffer.
                        let src_offset = i * entry_bytes;
                        view[src_offset..src_offset + entry_bytes]
                            .copy_from_slice(bytemuck::cast_slice(occupancy));
                    }
                }
                staging.unmap();

                // Copy each entry from staging to the correct slot in the shell buffer.
                let slot_stride = 16u64 * 4; // 16 u32s * 4 bytes = 64 bytes per slot
                for (i, &(slot, _)) in pending.iter().enumerate() {
                    encoder.copy_buffer_to_buffer(
                        &staging,
                        (i * entry_bytes) as u64,
                        &self.shell.buffer,
                        slot as u64 * slot_stride,
                        entry_bytes as u64,
                    );
                }
                pending.clear();
            }
        }

        // Upload face instances from CPU emit (replaces GPU emit pass).
        {
            let mut faces = self.pending_faces.borrow_mut();
            if self.faces_dirty.get() && !faces.is_empty() {
                // Patch obj_idx in packed field using the GPU object mapping.
                // Faces store sdf_object_id in bits 12-27. Replace with GPU index.
                let mapping = self.object_gpu_mapping.borrow();
                for face in faces.iter_mut() {
                    let sdf_obj_id = (face.packed >> 12) & 0xFFFF;
                    let gpu_idx = mapping.get(&sdf_obj_id).copied().unwrap_or(0);
                    // Clear old obj_idx bits and write new gpu_idx.
                    face.packed = (face.packed & 0xFFF) | ((gpu_idx & 0xFFFF) << 12);
                }
                drop(mapping);

                let face_count = faces.len() as u32;

                // Upload face data via staging buffer.
                let face_bytes: &[u8] = bytemuck::cast_slice(&faces);
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
                    &self.emit.face_buffer, 0,
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


        // 2. Raster: begin render pass with MRT, draw indirect.
        {
            let mut render_pass =
                SplatRasterPipeline::begin_render_pass(encoder, ctx.gbuffer);
            self.raster.draw(
                &mut render_pass,
                &ctx.scene.bind_group,
                &self.shell.bind_group,
                &self.emit.indirect_buffer,
            );
        }
    }

    fn recreate_pipeline(&mut self, _device: &wgpu::Device, _module: &wgpu::ShaderModule) {
        // The raster pipeline doesn't use the composed shader module from the
        // engine (that's for the march shader). For hot-reload, we'd recreate
        // the raster pipeline from its own source. For now, no-op.
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
        pool: &mut rkf_core::brick_pool::BrickPool,
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

        // Octree-native voxelization with adaptive subdivision.
        let (octree, brick_count, grid_origin) =
            rkp_core::voxelize_octree::voxelize_opacity_octree(
                opacity_fn, &aabb, voxel_size, pool,
            )?;


        // Build surface shell + face instances on CPU.
        // This replaces the GPU emit pass — computed once per voxelization,
        // drawn every frame by the raster pass.
        let base_vs = octree.base_voxel_size();
        let obj_idx = object_id; // Stored as sdf_object_id, remapped to GPU index during upload.

        {
            let mut shells = self.pending_shell_uploads.borrow_mut();
            let mut faces = self.pending_faces.borrow_mut();

            for (coord, slot, leaf_depth) in octree.iter_leaves() {
                let depth_diff = octree.depth() - leaf_depth;
                let leaf_vs = base_vs * (1u32 << depth_diff) as f32;

                // Brick's lower corner in octree space (0-based).
                // The vertex shader offsets by grid_origin for world transform.
                // The fragment shader uses these directly for octree sampling.
                let brick_origin = glam::Vec3::new(
                    coord.x as f32 * base_vs * 8.0,
                    coord.y as f32 * base_vs * 8.0,
                    coord.z as f32 * base_vs * 8.0,
                );

                // Build occupancy from brick data.
                let brick = pool.get(slot);
                let mut geo = rkf_core::BrickGeometry::new();
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            let sample = brick.sample(vx as u32, vy as u32, vz as u32);
                            let sv = rkp_core::SplatVoxel::from(sample);
                            geo.set_solid(vx, vy, vz, sv.opacity_f32() > 0.5);
                        }
                    }
                }

                // Store shell for GPU upload.
                shells.push((slot, geo.occupancy));
                self.shell.ensure_capacity(&self.device, slot + 1);

                // Emit exposed faces for this brick.
                for vz in 0..8u32 {
                    for vy in 0..8u32 {
                        for vx in 0..8u32 {
                            if !geo.is_solid(vx as u8, vy as u8, vz as u8) {
                                continue;
                            }
                            let voxel_idx = vx + vy * 8 + vz * 64;
                            let center = brick_origin
                                + (glam::Vec3::new(vx as f32, vy as f32, vz as f32) + 0.5)
                                    * leaf_vs;

                            // Check 6 faces: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
                            for face in 0..6u32 {
                                let (nx, ny, nz) = match face {
                                    0 => (-1i32, 0i32, 0i32),
                                    1 => (1, 0, 0),
                                    2 => (0, -1, 0),
                                    3 => (0, 1, 0),
                                    4 => (0, 0, -1),
                                    5 => (0, 0, 1),
                                    _ => unreachable!(),
                                };
                                let nbx = vx as i32 + nx;
                                let nby = vy as i32 + ny;
                                let nbz = vz as i32 + nz;

                                // Exposed if neighbor is out of brick or empty.
                                let exposed = if nbx < 0 || nbx >= 8 || nby < 0 || nby >= 8 || nbz < 0 || nbz >= 8 {
                                    true
                                } else {
                                    !geo.is_solid(nbx as u8, nby as u8, nbz as u8)
                                };

                                if exposed {
                                    faces.push(crate::splat_emit::FaceInstance {
                                        pos_x: center.x,
                                        pos_y: center.y,
                                        pos_z: center.z,
                                        voxel_size: leaf_vs,
                                        brick_slot: slot,
                                        packed: (voxel_idx & 0x1FF)
                                            | ((face & 0x7) << 9)
                                            | ((obj_idx & 0xFFFF) << 12),
                                    });
                                }
                            }
                        }
                    }
                }
            }

        }
        self.faces_dirty.set(true);

        // Allocate octree into the GPU allocator.
        let handle = self.octree.allocate(&octree);
        let spatial = rkf_core::scene_node::SpatialHandle::Octree {
            root_offset: handle.root_offset,
            len: handle.len,
            depth: handle.depth,
            base_voxel_size: handle.base_voxel_size,
        };

        self.faces_dirty.set(true);

        let geometry_aabb = rkf_core::Aabb::new(-half_extents, half_extents);
        Some((spatial, voxel_size, geometry_aabb, brick_count))
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
