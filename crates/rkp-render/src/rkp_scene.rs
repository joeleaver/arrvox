//! RKP scene GPU buffer management.
//!
//! Two upload paths, both explicit:
//! - [`RkpScene::upload_geometry`]: voxel pool, octree, color. Called on geometry change only.
//! - [`RkpScene::upload_frame`]: objects, camera. Called every frame (cheap — ~200 KB).
//!
//! No incremental updates, no caching, no callbacks. The caller builds the full
//! data each time and passes it in.

use crate::rkp_gpu_object::RkpGpuObject;

/// Camera uniforms matching the WGSL CameraUniforms struct.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub position: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub resolution: [f32; 2],
    pub jitter: [f32; 2],
    pub prev_vp: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
}

/// Geometry data — uploaded once when geometry changes (load, sculpt, voxelize).
pub struct GeometryUpload<'a> {
    /// Octree node values (packed u32s), one per node slot.
    pub octree_nodes: &'a [u32],
    /// Parallel prefiltered-LOD attr ids (u32s), one per node slot. Same
    /// length as `octree_nodes`. Entry is `INTERNAL_ATTR_NONE` for non-
    /// branches and for branches without a prefilter. The scene buffer
    /// interleaves these with `octree_nodes` into a single
    /// `array<vec2<u32>>` binding so we stay under the 12-storage-buffer
    /// per-stage limit.
    pub octree_internal_attrs: &'a [u32],
    /// Per-leaf attributes: `LeafAttr { normal_oct, material_primary,
    /// material_secondary_blend }`, 8 B each. Indexed by the leaf_attr_id
    /// stored in octree leaf nodes.
    pub leaf_attr_pool: &'a [u8],
    /// Per-leaf color — parallel to `leaf_attr_pool`, 4 B packed RGBA per slot.
    /// 0 means "no override; use material base_color".
    pub color_pool: &'a [u8],
    /// Brick storage: each brick is a contiguous run of 64 u32 cells (256 B).
    /// Indexed by `brick_id * 64 + flat_cell_index`. A cell's value is either
    /// 0xFFFFFFFF (empty) or a leaf_attr_id.
    pub brick_pool: &'a [u8],
}

/// Per-frame data — uploaded every frame (cheap: objects + camera).
pub struct FrameUpload<'a> {
    /// Per-object metadata, built from scene/ECS state.
    pub objects: &'a [RkpGpuObject],
    /// Camera uniforms.
    pub camera: &'a CameraUniforms,
}

/// GPU scene buffer manager for RKIPatch.
///
/// Bind group layout (group 0):
///   0: brick_pool (storage, read) — flat array of u32 cells, `brick_id * 64 + idx` indexes into it.
///       (Was a dummy voxel_pool slot pre-bricks; repurposed because we
///       were one storage-buffer over the per-stage limit.)
///   1: octree_nodes (storage, read) — `array<vec2<u32>>`: `.x` = node
///       value (EMPTY / INTERIOR / BRANCH offset / LEAF id / BRICK id),
///       `.y` = prefiltered-LOD attr id (INTERNAL_ATTR_NONE when absent).
///       Interleaved to stay under the 12-storage-buffer-per-stage limit
///       — a separate buffer would have pushed us over.
///   2: objects (storage, read)
///   3: camera (uniform)
///   4: color_pool (storage, read) — parallel to leaf_attr_pool
///   5: bone_matrices (storage, read)
///   6: bone_weights (storage, read)
///   7: deformed_pool (storage, read)
///   8: leaf_attr_pool (storage, read) — `LeafAttr { normal_oct, material_primary, material_secondary_blend }`
///
/// 8 storage buffers + 1 uniform in group 0; group 2 holds 4 more storage
/// buffers + 1 uniform — total 12 storage buffers per stage, exactly at
/// the rkf-render device limit.
pub struct RkpScene {
    pub brick_pool_buffer: wgpu::Buffer,
    pub octree_nodes_buffer: wgpu::Buffer,
    pub objects_buffer: wgpu::Buffer,
    pub camera_buffer: wgpu::Buffer,
    pub color_pool_buffer: wgpu::Buffer,
    pub bone_matrices_buffer: wgpu::Buffer,
    pub bone_weights_buffer: wgpu::Buffer,
    pub deformed_pool_buffer: wgpu::Buffer,
    pub leaf_attr_pool_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl RkpScene {
    pub fn new(device: &wgpu::Device) -> Self {
        let brick_pool_buffer = Self::create_storage(device, "rkp_brick_pool", 256);
        // 8-byte stride: each slot is `vec2<u32>` (value, prefilter-id).
        let octree_nodes_buffer = Self::create_storage(device, "rkp_octree_nodes", 8);
        let objects_buffer = Self::create_storage(device, "rkp_objects", std::mem::size_of::<RkpGpuObject>() as u64);
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_camera"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color_pool_buffer = Self::create_storage(device, "rkp_color_pool", 4);
        let bone_matrices_buffer = Self::create_storage(device, "rkp_bone_matrices", 64);
        let bone_weights_buffer = Self::create_storage(device, "rkp_bone_weights", 4);
        let deformed_pool_buffer = Self::create_storage(device, "rkp_deformed_pool", 8);
        let leaf_attr_pool_buffer = Self::create_storage(device, "rkp_leaf_attr_pool", 8);

        let bind_group_layout = Self::create_layout(device);
        let bind_group = Self::create_bind_group(device, &bind_group_layout,
            &brick_pool_buffer, &octree_nodes_buffer, &objects_buffer,
            &camera_buffer, &color_pool_buffer, &bone_matrices_buffer,
            &bone_weights_buffer, &deformed_pool_buffer, &leaf_attr_pool_buffer,
        );

        Self {
            brick_pool_buffer, octree_nodes_buffer, objects_buffer,
            camera_buffer, color_pool_buffer, bone_matrices_buffer,
            bone_weights_buffer, deformed_pool_buffer, leaf_attr_pool_buffer,
            bind_group_layout, bind_group,
        }
    }

    /// Upload geometry data. Call only when geometry changes (load, sculpt, voxelize).
    /// Grows buffers and rebuilds bind group as needed.
    pub fn upload_geometry(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &GeometryUpload) {
        assert_eq!(
            data.octree_nodes.len(),
            data.octree_internal_attrs.len(),
            "octree_nodes and octree_internal_attrs must have matching length",
        );

        // Interleave (value, prefilter_id) into a `vec2<u32>`-layout buffer
        // so a single binding slot carries both. Two separate bindings
        // would have pushed us over the 12-storage-buffer-per-stage limit.
        // One allocation per upload; octree_nodes uploads are rare
        // (voxelize, load) so the cost is amortized.
        let interleaved_u32_count = data.octree_nodes.len() * 2;
        let mut interleaved: Vec<u32> = Vec::with_capacity(interleaved_u32_count);
        for (i, &node) in data.octree_nodes.iter().enumerate() {
            interleaved.push(node);
            interleaved.push(data.octree_internal_attrs[i]);
        }
        let interleaved_bytes: &[u8] = bytemuck::cast_slice(&interleaved);

        let mut needs_rebuild = false;
        needs_rebuild |= Self::ensure_and_write(device, queue, &mut self.brick_pool_buffer, "rkp_brick_pool", data.brick_pool);
        needs_rebuild |= Self::ensure_and_write(device, queue, &mut self.octree_nodes_buffer, "rkp_octree_nodes", interleaved_bytes);
        needs_rebuild |= Self::ensure_and_write(device, queue, &mut self.leaf_attr_pool_buffer, "rkp_leaf_attr_pool", data.leaf_attr_pool);
        needs_rebuild |= Self::ensure_and_write(device, queue, &mut self.color_pool_buffer, "rkp_color_pool", data.color_pool);

        let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[rkp_scene] upload_geometry: octree_nodes={:.2} MiB (incl. prefilter ids)  leaf_attr={:.2} MiB  color_pool={:.2} MiB  bricks={:.2} MiB  total={:.2} MiB",
            mib(interleaved_bytes.len()),
            mib(data.leaf_attr_pool.len()),
            mib(data.color_pool.len()),
            mib(data.brick_pool.len()),
            mib(interleaved_bytes.len() + data.leaf_attr_pool.len() + data.color_pool.len() + data.brick_pool.len()),
        );

        if needs_rebuild {
            self.rebuild_bind_group(device);
        }
    }

    /// Upload per-frame data. Call every frame. Cheap (~200 KB for 1000 objects).
    /// Grows the objects buffer and rebuilds bind group if needed.
    pub fn upload_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &FrameUpload) {
        let obj_bytes: &[u8] = bytemuck::cast_slice(data.objects);
        let needs_rebuild = Self::ensure_and_write(device, queue, &mut self.objects_buffer, "rkp_objects", obj_bytes);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(data.camera));

        if needs_rebuild {
            self.rebuild_bind_group(device);
        }
    }

    /// Use an external objects buffer (e.g., the engine's GpuObject buffer).
    /// Rebuilds the bind group to reference it. Call each frame if the external
    /// buffer may have been replaced.
    pub fn set_external_objects_buffer(&mut self, device: &wgpu::Device, buffer: &wgpu::Buffer) {
        self.objects_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_objects_proxy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bind_group = Self::create_bind_group(device, &self.bind_group_layout,
            &self.brick_pool_buffer, &self.octree_nodes_buffer, buffer,
            &self.camera_buffer, &self.color_pool_buffer, &self.bone_matrices_buffer,
            &self.bone_weights_buffer, &self.deformed_pool_buffer, &self.leaf_attr_pool_buffer,
        );
    }

    /// Copy camera data from an external buffer (GPU→GPU) into our camera buffer.
    pub fn copy_camera_from(&self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Buffer) {
        let size = self.camera_buffer.size().min(src.size());
        encoder.copy_buffer_to_buffer(src, 0, &self.camera_buffer, 0, size);
    }

    fn ensure_and_write(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        label: &str,
        data: &[u8],
    ) -> bool {
        if data.is_empty() {
            return false;
        }
        if data.len() as u64 > buffer.size() {
            *buffer = Self::create_storage(device, label, data.len() as u64);
            queue.write_buffer(buffer, 0, data);
            true
        } else {
            queue.write_buffer(buffer, 0, data);
            false
        }
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = Self::create_bind_group(device, &self.bind_group_layout,
            &self.brick_pool_buffer, &self.octree_nodes_buffer, &self.objects_buffer,
            &self.camera_buffer, &self.color_pool_buffer, &self.bone_matrices_buffer,
            &self.bone_weights_buffer, &self.deformed_pool_buffer, &self.leaf_attr_pool_buffer,
        );
    }

    fn create_storage(device: &wgpu::Device, label: &str, min_size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: min_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rkp_scene_layout"),
            entries: &[
                storage_ro(0), // brick_pool
                storage_ro(1), // octree_nodes
                storage_ro(2), // objects
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(4), // color_pool
                storage_ro(5), // bone_matrices
                storage_ro(6), // bone_weights
                storage_ro(7), // deformed_pool
                storage_ro(8), // leaf_attr_pool
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        brick_pool: &wgpu::Buffer,
        octree_nodes: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        color_pool: &wgpu::Buffer,
        bone_matrices: &wgpu::Buffer,
        bone_weights: &wgpu::Buffer,
        deformed_pool: &wgpu::Buffer,
        leaf_attr_pool: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_scene_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: brick_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: octree_nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: objects.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: color_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bone_matrices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: bone_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: deformed_pool.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: leaf_attr_pool.as_entire_binding() },
            ],
        })
    }
}
