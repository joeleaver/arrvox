//! RKP scene GPU buffer management.
//!
//! Phase 4 (post-raymarch) minimal layout: the triangle raster pass only
//! needs `objects` (per-object world matrix + metadata) and `camera`. The
//! voxel_pool / octree / color_pool / bone_* buffers that the old compute
//! march fed from have been dropped — voxel data still lives on CPU in
//! `RkpSceneManager` as authoring state and drives mesh extraction, but
//! the GPU has no use for it once triangle geometry is in the mesh pool.
//!
//! Future skeletal animation for mesh-backed objects will reintroduce
//! bone/skinning buffers; they're intentionally absent now.

use crate::rkp_gpu_object::RkpGpuObject;

/// Camera uniforms matching the WGSL CameraUniforms struct.
///
/// `inverse_view_proj` is used by shade/SSAO/volumetric to reconstruct
/// world-space position from the depth buffer — this lets us skip writing
/// a dedicated position G-buffer target, saving 16 B/fragment of write
/// bandwidth.
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
    pub inverse_view_proj: [[f32; 4]; 4],
}

/// Per-frame data uploaded every frame. Cheap — a few hundred KB for a
/// scene with ~1000 objects.
pub struct FrameUpload<'a> {
    /// Per-object metadata, built from scene/ECS state.
    pub objects: &'a [RkpGpuObject],
    /// Camera uniforms.
    pub camera: &'a CameraUniforms,
}

/// GPU scene buffer manager for RKIPatch.
///
/// Bind group layout (group 0):
///   0: objects (storage, read)
///   1: camera  (uniform)
pub struct RkpScene {
    pub objects_buffer: wgpu::Buffer,
    pub camera_buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl RkpScene {
    pub fn new(device: &wgpu::Device) -> Self {
        let objects_buffer = Self::create_storage(
            device,
            "rkp_objects",
            std::mem::size_of::<RkpGpuObject>() as u64,
        );
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_camera"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::create_layout(device);
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &objects_buffer,
            &camera_buffer,
        );

        Self {
            objects_buffer,
            camera_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Upload per-frame data (objects + camera). Called every frame.
    /// Grows the objects buffer and rebuilds the bind group if needed.
    pub fn upload_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &FrameUpload) {
        let obj_bytes: &[u8] = bytemuck::cast_slice(data.objects);
        let needs_rebuild = Self::ensure_and_write(
            device,
            queue,
            &mut self.objects_buffer,
            "rkp_objects",
            obj_bytes,
        );
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(data.camera));

        if needs_rebuild {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.objects_buffer,
                &self.camera_buffer,
            );
        }
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

    fn create_storage(device: &wgpu::Device, label: &str, min_size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: min_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rkp_scene_layout"),
            entries: &[
                // 0: objects (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        objects: &wgpu::Buffer,
        camera: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_scene_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: objects.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera.as_entire_binding(),
                },
            ],
        })
    }
}
