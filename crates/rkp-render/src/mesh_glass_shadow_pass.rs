//! Mesh-mode glass shadow pipelines.
//!
//! Per cascade, two depth-only raster passes capture glass front-face
//! and back-face depth from the light's POV:
//!
//!   · `front_pipeline` — `cull = Back`, `depth_compare = Less`,
//!     writes the closest glass entry depth. Cleared to 1.0 (far)
//!     before each pass.
//!   · `back_pipeline`  — `cull = Front`, `depth_compare = Greater`,
//!     writes the farthest glass exit depth. Cleared to 0.0 (near)
//!     so the first hit passes.
//!
//! Both pipelines share the same VS + FS in `mesh_glass_shadow.wesl`;
//! the FS discards opaque fragments so only glass surfaces land in
//! the depth target. The shade pass reads the two depth maps and
//! multiplies the existing CSM shadow factor by `exp(-sigma *
//! thickness)` where `thickness = back_depth - front_depth` projected
//! to world units via the cascade's view-proj scale.
//!
//! Bind-group shape mirrors `mesh_shadow_map_pass`:
//!   · `g0` — `LightCameraCsm` + `MeshShadowParams` + bone palettes.
//!     Reused from `MeshShadowMapPass::render_g0_layout`.
//!   · `g1` — per-instance `MeshShadowInstance` (shared with splat /
//!     primary mesh raster).
//!   · `g2` — glass-classify (leaf_attr_pool + materials + instances
//!     + overlay + color_pool). Shared with `MeshGlassPass`.

use rkp_core::mesh_extract::MeshVertex;

pub const GLASS_SHADOW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct MeshGlassShadowPass {
    pub front_pipeline: wgpu::RenderPipeline,
    pub back_pipeline: wgpu::RenderPipeline,
}

impl MeshGlassShadowPass {
    pub fn new(
        device: &wgpu::Device,
        render_g0_layout: &wgpu::BindGroupLayout,
        splat_g1_layout: &wgpu::BindGroupLayout,
        g2_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_glass_shadow pipeline layout"),
            bind_group_layouts: &[
                Some(render_g0_layout),
                Some(splat_g1_layout),
                Some(g2_layout),
            ],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_glass_shadow"),
            "mesh_glass_shadow",
        );

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    shader_location: 2,
                    offset: 16,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    shader_location: 3,
                    offset: 20,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    shader_location: 4,
                    offset: 24,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        let front_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_glass_shadow front"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[vertex_layout.clone()],
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
                format: GLASS_SHADOW_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("frag_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });

        let back_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_glass_shadow back"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[vertex_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GLASS_SHADOW_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("frag_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            front_pipeline,
            back_pipeline,
        }
    }

    pub fn begin_front_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_glass_shadow front"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
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

    pub fn begin_back_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_glass_shadow back"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
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

#[cfg(test)]
mod tests {
    

    #[test]
    fn mesh_glass_shadow_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("mesh_glass_shadow");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}
