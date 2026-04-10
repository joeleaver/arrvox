//! Splat rasterization — camera-facing billboards with gradient normals.
//!
//! Single pass: closest billboard wins (depth test). Gradient normal from
//! trilinear opacity field. Circular discard for round silhouettes.

use crate::splat_emit::SplatEmitPass;

/// The splat rasterization pipeline.
pub struct SplatRasterPipeline {
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout for splat instance buffer (group 1).
    pub face_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for splat instance buffer. Recreated when buffer grows.
    pub face_bind_group: std::cell::RefCell<wgpu::BindGroup>,
}

impl SplatRasterPipeline {
    pub fn new(
        device: &wgpu::Device,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        emit: &SplatEmitPass,
    ) -> Self {
        let face_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("raster face layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let face_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster face bind group"),
            layout: &face_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: emit.face_buffer.borrow().as_entire_binding(),
            }],
        });

        let shader_src = include_str!("shaders/splat_raster.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_raster"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_raster pipeline layout"),
            bind_group_layouts: &[
                Some(scene_bind_group_layout),
                Some(&face_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_raster pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: rkf_render::gbuffer::GBUFFER_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_POSITION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            face_bind_group_layout,
            face_bind_group: std::cell::RefCell::new(face_bind_group),
        }
    }

    /// Record the raster draw call.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        scene_bind_group: &'a wgpu::BindGroup,
        indirect_buffer: &'a wgpu::Buffer,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, scene_bind_group, &[]);
        render_pass.set_bind_group(1, &*self.face_bind_group.borrow(), &[]);
        render_pass.draw_indirect(indirect_buffer, 0);
    }

    /// Begin a render pass for G-buffer MRT + depth.
    pub fn begin_render_pass<'a>(
        encoder: &'a mut wgpu::CommandEncoder,
        gbuffer: &'a rkf_render::gbuffer::GBuffer,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat_raster pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.position_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: f64::from(f32::MAX),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.material_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0, g: 0.0, b: 0.0, a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}
