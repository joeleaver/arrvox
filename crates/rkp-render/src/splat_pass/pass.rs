//! `SplatPass` — render pipeline owner for the surface-splat prototype.
//!
//! Holds the pipeline + camera/material bind layout. Caller supplies the
//! per-frame vertex buffer (built once from [`super::extract_splats`])
//! and the bind group entries (camera uniform + leaf_attr_pool + materials).
//!
//! The pass is intentionally minimal — no depth-pre-pass, no instancing
//! across multiple assets, no LOD cut. The prototype runs on a single
//! asset and writes albedo + normal to MRT outputs supplied by the
//! caller. Phase B-2 is where this gets folded into the editor's
//! G-buffer + shade pipeline.

use super::extract::SplatVertex;

/// Pipeline-level configuration. Caller creates the textures/bind groups
/// and tells `SplatPass` what colour formats and depth format the
/// pipeline should target.
pub struct SplatPassConfig {
    pub albedo_format: wgpu::TextureFormat,
    pub normal_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
    /// Multi-sample count. Use 1 for the perf prototype.
    pub sample_count: u32,
}

/// Splat-rasterizer GPU pass.
///
/// `g0_layout` is the bind-group layout the caller fills with
/// (camera_uniform, leaf_attr_pool, materials). The shader expects
/// these exact bindings; see `splat.wesl` for the WGSL contract.
pub struct SplatPass {
    pub pipeline: wgpu::RenderPipeline,
    pub g0_layout: wgpu::BindGroupLayout,
}

impl SplatPass {
    pub fn new(device: &wgpu::Device, config: &SplatPassConfig) -> Self {
        // Group 0: camera (uniform) + leaf_attr_pool (storage<read>)
        //          + materials (storage<read>). Same bindings as in
        //          `splat.wesl`. Vertex stage needs all three (it
        //          reads LeafAttr for the basis); fragment also needs
        //          all three.
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat g0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(SPLAT_CAMERA_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat pipeline layout"),
            bind_group_layouts: &[Some(&g0_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("splat"),
            "splat",
        );

        // SplatVertex: 32 B per instance.
        //   @location(0) world_pos:    vec3<f32>  @ offset 0
        //   @location(1) radius:       f32        @ offset 12
        //   @location(2) leaf_attr_id: u32        @ offset 16
        // The trailing 12 bytes of padding aren't bound to any input.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SplatVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    shader_location: 2,
                    offset: 16,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("splat"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[vertex_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Disc splats are double-sided
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: config.depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: config.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("frag_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: config.albedo_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: config.normal_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline, g0_layout }
    }

    /// Encode one splat-render pass.
    ///
    /// `vertex_buffer` is the per-asset vertex buffer (built from
    /// [`super::extract_splats`]). `vertex_count` is the splat count
    /// (= one instance per splat; 4 verts per instance for the quad).
    /// `bind_group` must match `g0_layout`.
    ///
    /// Caller supplies the colour + depth attachments — we don't manage
    /// textures here so the same pass can write into a test target or
    /// the editor's G-buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn render<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        vertex_buffer: &'a wgpu::Buffer,
        vertex_count: u32,
        bind_group: &'a wgpu::BindGroup,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        clear_color: wgpu::Color,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: albedo_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        // 4 verts per instance (triangle strip), one instance per splat.
        pass.draw(0..4, 0..vertex_count);
    }
}

/// Camera uniform for the splat shader. Mirrors `SplatCamera` in
/// `splat.wesl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatCamera {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 3],
    pub _pad0: f32,
    pub resolution: [f32; 2],
    pub _pad1: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<SplatCamera>() == 96);
pub const SPLAT_CAMERA_BYTES: u64 = std::mem::size_of::<SplatCamera>() as u64;
