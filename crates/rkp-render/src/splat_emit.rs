//! Emit compute pass — traverses octrees and emits transition face quad instances.
//!
//! One workgroup per visible object. Each workgroup traverses the object's octree,
//! reads occupancy bitmasks from the surface shell buffer, and atomically appends
//! face instances to an output buffer. The output feeds directly into an indirect
//! draw call for the rasterization pass.

use crate::surface_shell_gpu::SurfaceShellGpu;

/// Per-face instance data written by the emit shader, read by the raster vertex shader.
///
/// 24 bytes per instance. Must match the WGSL `FaceInstance` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FaceInstance {
    /// Local-space voxel center X.
    pub pos_x: f32,
    /// Local-space voxel center Y.
    pub pos_y: f32,
    /// Local-space voxel center Z.
    pub pos_z: f32,
    /// Voxel size in world units (varies with octree depth).
    pub voxel_size: f32,
    /// Brick pool slot.
    pub brick_slot: u32,
    /// Packed: voxel_index(9) | face_id(3) | obj_idx(16) | unused(4)
    pub packed: u32,
}

/// Uniform parameters for the emit shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitParams {
    pub max_faces: u32,
    pub object_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// The emit compute pass.
pub struct SplatEmitPass {
    pipeline: wgpu::ComputePipeline,
    /// Output: face instance buffer (storage, read by raster pass). Growable.
    pub face_buffer: std::cell::RefCell<wgpu::Buffer>,
    /// Output: indirect draw args (vertex_count=6, instance_count=atomic).
    pub indirect_buffer: wgpu::Buffer,
    /// Staging buffer with reset values for indirect args (vertex_count=6, rest=0).
    indirect_reset_buffer: wgpu::Buffer,
    /// Bind group layout for the output buffers (group 2).
    pub output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the output buffers.
    pub output_bind_group: wgpu::BindGroup,
    /// Bind group layout for emit params (group 3).
    params_bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for emit params.
    pub params_buffer: wgpu::Buffer,
    /// Bind group for emit params.
    params_bind_group: wgpu::BindGroup,
    /// Maximum face instances the buffer can hold.
    max_faces: u32,
}

/// Default initial capacity for face instances.
const DEFAULT_MAX_FACES: u32 = 1_000_000;

/// Size of one FaceInstance in bytes.
const FACE_INSTANCE_SIZE: u64 = std::mem::size_of::<FaceInstance>() as u64;

/// Size of DrawIndirectArgs in bytes (4 u32s = 16 bytes).
const DRAW_INDIRECT_SIZE: u64 = 16;

impl SplatEmitPass {
    /// Create the emit pass.
    ///
    /// `scene_bind_group_layout`: group 0 layout (octree_nodes + objects).
    /// `shell`: the surface shell GPU buffer (group 1).
    pub fn new(
        device: &wgpu::Device,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        shell: &SurfaceShellGpu,
    ) -> Self {
        let max_faces = DEFAULT_MAX_FACES;

        // Face instance buffer.
        let face_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emit face instances"),
            size: max_faces as u64 * FACE_INSTANCE_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Indirect draw args buffer: vertex_count(6), instance_count(0), first_vertex(0), first_instance(0).
        let indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emit indirect args"),
            size: DRAW_INDIRECT_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffer with reset values for indirect args.
        let indirect_reset_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emit indirect reset"),
            size: DRAW_INDIRECT_SIZE,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let reset_data: [u32; 4] = [6, 0, 0, 0]; // vertex_count=6, instance_count=0
            indirect_reset_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&reset_data));
            indirect_reset_buffer.unmap();
        }

        // Output bind group (group 2): face_instances + draw_args.
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("emit output layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emit output bind group"),
            layout: &output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: face_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_buffer.as_entire_binding(),
                },
            ],
        });

        // Params uniform (group 3).
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("emit params layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emit params"),
            size: std::mem::size_of::<EmitParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emit params bind group"),
            layout: &params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Pipeline.
        let shader_src = include_str!("shaders/splat_emit.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_emit"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_emit pipeline layout"),
            bind_group_layouts: &[
                scene_bind_group_layout,     // group 0: octree_nodes + objects
                &shell.bind_group_layout,    // group 1: surface_shell
                &output_bind_group_layout,   // group 2: face_instances + draw_args
                &params_bind_group_layout,   // group 3: emit_params
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_emit pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            face_buffer: std::cell::RefCell::new(face_buffer),
            indirect_buffer,
            indirect_reset_buffer,
            output_bind_group_layout,
            output_bind_group,
            params_bind_group_layout,
            params_buffer,
            params_bind_group,
            max_faces,
        }
    }

    /// Update the object count uniform. Call when the scene changes.
    pub fn update_params(&self, queue: &wgpu::Queue, object_count: u32) {
        let params = EmitParams {
            max_faces: self.max_faces,
            object_count,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Record the emit compute dispatch into the command encoder.
    ///
    /// Resets the indirect draw args (via staging copy) then dispatches the emit
    /// compute shader. `update_params()` should be called when object count changes.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bind_group: &wgpu::BindGroup,
        shell_bind_group: &wgpu::BindGroup,
        object_count: u32,
    ) {
        // Reset indirect args: copy from staging buffer (vertex_count=6, instance_count=0).
        encoder.copy_buffer_to_buffer(
            &self.indirect_reset_buffer,
            0,
            &self.indirect_buffer,
            0,
            DRAW_INDIRECT_SIZE,
        );

        // Dispatch emit.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("splat_emit"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.set_bind_group(1, shell_bind_group, &[]);
        pass.set_bind_group(2, &self.output_bind_group, &[]);
        pass.set_bind_group(3, &self.params_bind_group, &[]);
        pass.dispatch_workgroups(object_count, 1, 1);
    }

    /// Maximum face instances the buffer can hold.
    pub fn max_faces(&self) -> u32 {
        self.max_faces
    }
}
