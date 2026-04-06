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
    /// Pending face instances from CPU-side emit — uploaded in dispatch.
    pending_faces: std::cell::RefCell<Vec<crate::splat_emit::FaceInstance>>,
    /// The device, needed for creating staging buffers in dispatch.
    device: wgpu::Device,
    /// Whether face data needs re-upload.
    faces_dirty: std::cell::Cell<bool>,
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
        let obj_idx = 0u32; // Will be corrected when we support multiple objects.

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
