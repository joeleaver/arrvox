//! Procedural CSG raymarcher — live preview pass for the build viewport.
//!
//! Runs as a compute dispatch (one thread per pixel), sphere-traces the
//! flattened RPN tree against each camera ray, and writes into the same
//! G-buffer layout (`position`, `normal`, `material`) as
//! [`crate::octree_march::OctreeMarchPass`]. Downstream passes (shadow,
//! SSAO, shade, post) don't distinguish its output from the voxel march.
//!
//! The shader input is an array of `ProcInstruction` (see
//! `rkp_procedural::flatten`) — a post-order RPN encoding of the tree.
//! The CPU flattens once per edit; re-uploads to the GPU are cheap
//! (O(tree size), tens of bytes per node). This replaces the voxel
//! march in the build viewport when the user toggles live preview —
//! the voxel path remains the source of truth elsewhere.
//!
//! This pass owns no shared state with `RkpScene`: it binds only the
//! viewport's camera buffer (group 0), the G-buffer textures (group 1),
//! and its own params + instruction buffer (group 2). Keeping it
//! decoupled means adding or removing the preview in the pipeline is a
//! simple toggle at the frame-driver level.

use crate::validate_wgsl;
use bytemuck::{Pod, Zeroable};
use rkp_procedural::flatten::ProcInstruction;

/// Params uniform mirroring the shader struct of the same name.
///
/// `instruction_count` is the number of entries in the instructions
/// buffer the shader should execute — *not* the buffer's capacity. We
/// grow the buffer to the high-water mark and reuse it across bakes;
/// trees that get smaller just ignore the tail.
///
/// `object_id` is packed into the G-buffer material channel's pick
/// byte so this procedural's hits show up like any other entity when
/// the pick shader reads back.
///
/// `entity_world` / `entity_inverse_world` carry the owning entity's
/// world transform (and its inverse) so the shader can march the ray
/// in the entity's local frame and then convert the hit back to world
/// for the G-buffer. Without this, moving the entity in world space
/// shifts the procedural preview out from under the camera — the tree
/// primitives' own `inverse_world` only encodes intra-tree hierarchy.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ProcRaymarchParams {
    pub instruction_count: u32,
    pub object_id: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub entity_world: [[f32; 4]; 4],
    pub entity_inverse_world: [[f32; 4]; 4],
}

/// Procedural CSG raymarch compute pass.
pub struct ProcRaymarchPass {
    pipeline: wgpu::ComputePipeline,

    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: Option<wgpu::BindGroup>,

    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: Option<wgpu::BindGroup>,

    params_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,

    /// Storage buffer holding the current flattened instruction stream.
    /// Grows on demand; never shrinks (shader only reads the first
    /// `instruction_count` entries).
    instructions_buffer: wgpu::Buffer,
    instructions_capacity: usize,
}

impl ProcRaymarchPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Group 0: camera uniform. Mirrors `RkpScene`'s camera binding
        // shape (`CameraUniforms` 224 B). We declare our own layout
        // rather than reuse the scene layout because we don't need the
        // scene buffers — keeps the pass self-contained.
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch camera layout"),
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

        // Group 1: G-buffer (same triple as OctreeMarch — position,
        // normal, material).
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch gbuffer layout"),
                entries: &[
                    bgl_storage_tex(0, wgpu::TextureFormat::Rgba32Float),
                    bgl_storage_tex(1, wgpu::TextureFormat::Rgba16Float),
                    bgl_storage_tex(2, wgpu::TextureFormat::Rg32Uint),
                ],
            });

        // Group 2: params + instructions.
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch params layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_raymarch params"),
            size: std::mem::size_of::<ProcRaymarchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Minimal seed size — we'll grow this on first upload. The
        // shader reads up to `instruction_count` entries, never past
        // the buffer, so even a zero-instruction tree renders fine
        // against this stub.
        let initial_cap = 4usize;
        let instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_raymarch instructions"),
            size: (initial_cap * std::mem::size_of::<ProcInstruction>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_src = include_str!("shaders/proc_raymarch.wgsl");
        validate_wgsl(shader_src, "proc_raymarch");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("proc_raymarch"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("proc_raymarch pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&gbuffer_bind_group_layout),
                Some(&params_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("proc_raymarch"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            camera_bind_group_layout,
            camera_bind_group: None,
            gbuffer_bind_group_layout,
            gbuffer_bind_group: None,
            params_bind_group_layout,
            params_bind_group: None,
            params_buffer,
            instructions_buffer,
            instructions_capacity: initial_cap,
        }
    }

    /// Wire the viewport's camera uniform into this pass.
    pub fn set_camera(&mut self, device: &wgpu::Device, camera_buffer: &wgpu::Buffer) {
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch camera"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Wire the viewport's G-buffer views into this pass. Re-call after
    /// a G-buffer resize.
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch gbuffer"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
            ],
        }));
        self.rebuild_params_bind_group(device);
    }

    /// Upload (or replace) the flattened instruction stream. If the
    /// new stream exceeds the current buffer capacity we grow the
    /// buffer (doubling past-capacity; keeps the amortized cost
    /// proportional to a single upload) and rebuild the bind group.
    /// Zero-length streams are valid — the shader writes "miss" to
    /// every pixel.
    pub fn upload_instructions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instructions: &[ProcInstruction],
    ) {
        if instructions.len() > self.instructions_capacity {
            let new_cap = instructions.len().next_power_of_two().max(4);
            self.instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_raymarch instructions"),
                size: (new_cap * std::mem::size_of::<ProcInstruction>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instructions_capacity = new_cap;
            self.rebuild_params_bind_group(device);
        }
        if !instructions.is_empty() {
            queue.write_buffer(&self.instructions_buffer, 0, bytemuck::cast_slice(instructions));
        }
    }

    /// Update the params uniform.
    ///
    /// `entity_world` is the owning entity's world transform; the
    /// shader uses its inverse to pull the camera ray into the entity's
    /// local frame before marching, and the forward transform to push
    /// the hit position + normal back to world space for the G-buffer.
    /// Pass [`glam::Affine3A::IDENTITY`] for entities that live at the
    /// world origin — the math still works, just with no shift.
    pub fn set_params(
        &self,
        queue: &wgpu::Queue,
        instruction_count: u32,
        object_id: u32,
        entity_world: glam::Affine3A,
    ) {
        let world = glam::Mat4::from(entity_world);
        let inverse = world.inverse();
        let p = ProcRaymarchParams {
            instruction_count,
            object_id,
            _pad0: 0,
            _pad1: 0,
            entity_world: world.to_cols_array_2d(),
            entity_inverse_world: inverse.to_cols_array_2d(),
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&p));
    }

    fn rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        self.params_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch params"),
            layout: &self.params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instructions_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Record the compute dispatch. Caller is responsible for having
    /// called `set_camera` / `set_gbuffer` / `upload_instructions` /
    /// `set_params` first — we skip silently if any bind group is
    /// unset rather than panic inside an encoder.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let (Some(cam), Some(gbuf), Some(params)) = (
            &self.camera_bind_group,
            &self.gbuffer_bind_group,
            &self.params_bind_group,
        ) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proc_raymarch"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, cam, &[]);
        pass.set_bind_group(1, gbuf, &[]);
        pass.set_bind_group(2, params, &[]);
        let gx = width.div_ceil(8);
        let gy = height.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

fn bgl_storage_tex(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
