//! Ghost-cutter overlay pass.
//!
//! Alongside `ProcRaymarchPass` + `ProcOutlinePass`, this pass lets the
//! build viewport show Subtract / Intersect operands that would
//! otherwise be invisible — a cutter fully buried inside its minuend
//! has no surface in the main G-buffer at all, so a separate
//! raymarch is the only way to render its silhouette.
//!
//! The caller (engine) decides which primitives to ghost based on the
//! current selection (see `engine::collect_ghost_primitives`), flattens
//! them as a subset of the existing `ProcInstruction` stream, and
//! uploads via `upload_instructions`. Zero-length uploads are valid —
//! the shader early-outs on `instruction_count == 0`, so this is
//! effectively free when no CSG op is selected.

use crate::compile_pass_shader;
use bytemuck::{Pod, Zeroable};
use arvx_procedural::flatten::ProcInstruction;

/// Uniform mirroring `GhostParams` in the shader. Color is stored in
/// non-premultiplied form; the shader multiplies by alpha so the
/// pipeline's `src=One, dst=OneMinusSrcAlpha` blend is a straight
/// pre-mul over.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GhostParams {
    pub instruction_count: u32,
    pub _pad0: [u32; 3],
    pub color_rgba: [f32; 4],
}

impl GhostParams {
    pub const NONE: Self = Self {
        instruction_count: 0,
        _pad0: [0; 3],
        color_rgba: [0.0; 4],
    };

    pub fn new(instruction_count: u32, color: [f32; 4]) -> Self {
        Self {
            instruction_count,
            _pad0: [0; 3],
            color_rgba: color,
        }
    }
}

pub struct ProcGhostPass {
    pipeline: wgpu::RenderPipeline,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: Option<wgpu::BindGroup>,
    params_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
    instructions_buffer: wgpu::Buffer,
    instructions_capacity: usize,
}

impl ProcGhostPass {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_ghost camera layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_ghost params layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("proc_ghost params"),
            size: std::mem::size_of::<GhostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_cap = 4usize;
        let instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_ghost instructions"),
            size: (initial_cap * std::mem::size_of::<ProcInstruction>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = compile_pass_shader(device, wesl::include_wesl!("proc_ghost"), "proc_ghost");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("proc_ghost pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&params_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("proc_ghost"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_bind_group_layout,
            camera_bind_group: None,
            params_bind_group_layout,
            params_bind_group: None,
            params_buffer,
            instructions_buffer,
            instructions_capacity: initial_cap,
        }
    }

    pub fn set_camera(&mut self, device: &wgpu::Device, camera_buffer: &wgpu::Buffer) {
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_ghost camera"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        }));
    }

    pub fn upload_instructions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instructions: &[ProcInstruction],
    ) {
        if instructions.len() > self.instructions_capacity {
            let new_cap = instructions.len().next_power_of_two().max(4);
            self.instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_ghost instructions"),
                size: (new_cap * std::mem::size_of::<ProcInstruction>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instructions_capacity = new_cap;
            self.params_bind_group = None; // force rebuild below
        }
        if !instructions.is_empty() {
            queue.write_buffer(&self.instructions_buffer, 0, bytemuck::cast_slice(instructions));
        }
        if self.params_bind_group.is_none() {
            self.rebuild_params_bind_group(device);
        }
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GhostParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    fn rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        self.params_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_ghost params"),
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

    /// Draw the ghost overlay into `target`. `LoadOp::Load` preserves
    /// whatever's already on the composite (scene + grid + outline).
    /// Silently skipped if bind groups aren't wired — keeps the call
    /// site in `render_to` straightforward.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let (Some(cam), Some(params)) = (&self.camera_bind_group, &self.params_bind_group) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("proc_ghost"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, cam, &[]);
        pass.set_bind_group(1, params, &[]);
        pass.draw(0..3, 0..1);
    }
}
