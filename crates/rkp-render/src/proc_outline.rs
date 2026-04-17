//! Selected-primitive outline overlay for the procedural raymarch
//! preview.
//!
//! A thin render pass that draws an outline around whichever primitive
//! the user currently has selected in the build panel. Reads the
//! per-pixel NodeId that `proc_raymarch.wgsl` packs into the material
//! G-buffer and emits a 1-pixel band along the silhouette.
//!
//! Render-pipeline position: after tone-map / grid, so it sits in LDR
//! and isn't smeared by bloom. Full-screen triangle, alpha-blended
//! over the composite texture. Sentinel `u32::MAX` disables the pass
//! cheaply (shader discards every pixel). Only dispatched in raymarch
//! preview mode — the voxel-march G-buffer doesn't carry NodeId in
//! the same slot (it carries secondary_material_id there), so reading
//! it in voxel mode would outline noise.

use crate::validate_wgsl;
use bytemuck::{Pod, Zeroable};

/// Uniform matching `OutlineParams` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OutlineParams {
    pub selected_node_id: u32,
    pub _pad: [u32; 3],
    pub color_rgba: [f32; 4],
}

impl OutlineParams {
    /// Sentinel used by the shader for "no selection, discard
    /// everything." Keeps the pass cheap when nothing is selected.
    pub const NONE: Self = Self {
        selected_node_id: u32::MAX,
        _pad: [0; 3],
        color_rgba: [0.0; 4],
    };

    pub fn new(selected_node_id: u32, color: [f32; 4]) -> Self {
        Self {
            selected_node_id,
            _pad: [0; 3],
            color_rgba: color,
        }
    }
}

pub struct ProcOutlinePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
}

impl ProcOutlinePass {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("proc_outline layout"),
            entries: &[
                // Material G-buffer — packed u32s, read as sampled
                // uint texture via `textureLoad`.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_outline params"),
            size: std::mem::size_of::<OutlineParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_src = include_str!("shaders/proc_outline.wgsl");
        validate_wgsl(shader_src, "proc_outline");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("proc_outline"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("proc_outline pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("proc_outline"),
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
                    // Shader premultiplies the color by alpha; blend
                    // is standard pre-mul over: out = src + dst*(1-a).
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
            bind_group_layout,
            bind_group: None,
            params_buffer,
        }
    }

    /// Rewire the pick G-buffer view. Call after VR construction and
    /// after every resize. The outline reads `primitive_node_id`
    /// directly from the rkp-side `R32Uint` pick texture — moving it
    /// out of the shared material G-buffer freed the secondary-material
    /// slot for dual-material shading.
    pub fn set_gbuffer(&mut self, device: &wgpu::Device, pick_view: &wgpu::TextureView) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_outline bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(pick_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &OutlineParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Draw the outline into `target`. Uses `LoadOp::Load` so the
    /// composite's existing contents survive — we're overlaying, not
    /// replacing. No-op if the bind group isn't yet wired.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let Some(bg) = &self.bind_group else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("proc_outline"),
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
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
