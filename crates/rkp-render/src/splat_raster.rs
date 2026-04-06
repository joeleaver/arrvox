//! Rasterization render pipeline — draws emitted face quads into the G-buffer.
//!
//! Uses a `wgpu::RenderPipeline` with vertex + fragment stages. The vertex shader
//! reads `FaceInstance` data from a storage buffer and expands each instance into a
//! quad (6 vertices, 2 triangles). The fragment shader does trilinear opacity
//! refinement, gradient normal computation, and material reads, writing the final
//! G-buffer via MRT.

use crate::splat_emit::SplatEmitPass;
use crate::surface_shell_gpu::SurfaceShellGpu;

/// The rasterization render pipeline for surface voxel faces.
pub struct SplatRasterPipeline {
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout for face instance buffer (group 1).
    pub face_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for face instance buffer.
    pub face_bind_group: wgpu::BindGroup,
}

impl SplatRasterPipeline {
    /// Create the render pipeline.
    ///
    /// `scene_bind_group_layout`: group 0 (brick_pool, octree_nodes, objects, camera, etc.)
    /// `shell`: group 2 (surface shell occupancy)
    /// `emit`: provides the face instance buffer for group 1
    pub fn new(
        device: &wgpu::Device,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        shell: &SurfaceShellGpu,
        emit: &SplatEmitPass,
    ) -> Self {
        // Group 1: face instances (read-only storage).
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
                resource: emit.face_buffer.as_entire_binding(),
            }],
        });

        // Shader module.
        let shader_src = include_str!("shaders/splat_raster.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_raster"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Pipeline layout: group 0 = scene, group 1 = faces, group 2 = shell
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_raster pipeline layout"),
            bind_group_layouts: &[
                scene_bind_group_layout,     // group 0: scene data
                &face_bind_group_layout,     // group 1: face instances
                &shell.bind_group_layout,    // group 2: surface shell
            ],
            push_constant_ranges: &[],
        });

        // Render pipeline with MRT matching G-buffer formats.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat_raster pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers — all data from storage buffer
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
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
                // MRT is limited to 32 bytes/sample on many GPUs.
                // Position(16) + Normal(8) + Material(8) = 32 bytes exactly.
                // Motion vectors are written in a post-pass or set to zero.
                targets: &[
                    // Target 0: position (Rgba32Float) = 16 bytes
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_POSITION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 1: normal (Rgba16Float) = 8 bytes
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    // Target 2: material (Rg32Uint) = 8 bytes
                    Some(wgpu::ColorTargetState {
                        format: rkf_render::gbuffer::GBUFFER_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            face_bind_group_layout,
            face_bind_group,
        }
    }

    /// Record the raster draw call using indirect args from the emit pass.
    ///
    /// The render pass must already be begun with the correct MRT attachments.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        scene_bind_group: &'a wgpu::BindGroup,
        shell_bind_group: &'a wgpu::BindGroup,
        indirect_buffer: &'a wgpu::Buffer,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, scene_bind_group, &[]);
        render_pass.set_bind_group(1, &self.face_bind_group, &[]);
        render_pass.set_bind_group(2, shell_bind_group, &[]);
        render_pass.draw_indirect(indirect_buffer, 0);
    }

    /// Create the render pass descriptor for G-buffer MRT + depth.
    ///
    /// Returns owned descriptors — caller begins the render pass from these.
    pub fn begin_render_pass<'a>(
        encoder: &'a mut wgpu::CommandEncoder,
        gbuffer: &'a rkf_render::gbuffer::GBuffer,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat_raster pass"),
            color_attachments: &[
                // Target 0: position
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
                // Target 1: normal
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
                // Target 2: material
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
                // Motion vectors (target 3) omitted — 32 byte/sample MRT limit.
                // Motion is zeroed via a separate clear or post-pass.
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
        })
    }
}
