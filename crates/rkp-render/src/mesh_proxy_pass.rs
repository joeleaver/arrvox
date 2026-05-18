//! `MeshProxyPass` — render-pipeline owner for the procedural proxy-
//! mesh path (GPU surface-nets-from-SDF).
//!
//! Distinct from `MeshPass` (the octree-backed mesh raster):
//!   · Vertex layout: `ProxyVertex` (32 B, material + color payload),
//!     not `MeshVertex`.
//!   · Bindings: minimal — camera (g0) + per-instance world matrix
//!     and object_id (g1). No bones, no scene buffers, no LeafAttr
//!     pool. Proxy meshes carry their full shading data per-vertex.
//!   · FS writes the **full** G-buffer directly: position, pick,
//!     normal, material, glass=0. No `mesh_resolve` participation.
//!   · Render-pass attachments load (not clear) so the proxy
//!     composites on top of whatever the mesh raster + `mesh_resolve`
//!     already wrote, with depth-test against the shared depth
//!     attachment.
//!
//! Scheduling order each frame:
//!   1. mesh raster              → visibility buffer + depth
//!   2. mesh_resolve compute     → gbuf_normal/material/glass
//!   3. **mesh_proxy raster**    → gbuf_position/pick/normal/material/glass + depth
//!
//! See `notes/proxy-mesh-first-class.md` for the full architecture.

use crate::gbuffer::{
    GBUFFER_DEPTH_FORMAT, GBUFFER_GLASS_FORMAT, GBUFFER_LEAF_SLOT_FORMAT,
    GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT, GBUFFER_POSITION_FORMAT,
};

const GBUFFER_PICK_FORMAT: wgpu::TextureFormat = GBUFFER_LEAF_SLOT_FORMAT;

/// Per-instance uniform consumed by `mesh_proxy.wesl::ProxyInstance`.
/// 80 B, 16 B aligned. Single per-instance draw call binds one of
/// these via `g1`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProxyInstanceUniform {
    pub world: [[f32; 4]; 4],
    pub object_id: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<ProxyInstanceUniform>() == 80);
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(ProxyInstanceUniform, world) == 0);
    assert!(offset_of!(ProxyInstanceUniform, object_id) == 64);
};

pub const PROXY_INSTANCE_BYTES: u64 = std::mem::size_of::<ProxyInstanceUniform>() as u64;

/// One proxy mesh to draw this frame. The renderer's `proxy_mesh_buffer`
/// lookup resolves `handle_raw` to `(vbo, ibo, index_count)`.
#[derive(Clone, Copy, Debug)]
pub struct ProxyDraw {
    pub handle_raw: u32,
    pub world: [[f32; 4]; 4],
    pub object_id: u32,
}

pub struct MeshProxyPass {
    pub pipeline: wgpu::RenderPipeline,
    pub g0_layout: wgpu::BindGroupLayout,
    pub g1_layout: wgpu::BindGroupLayout,
}

impl MeshProxyPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_proxy g0 (camera)"),
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

        let g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_proxy g1 (instance)"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_proxy pipeline layout"),
            bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_proxy"),
            "mesh_proxy",
        );

        // ProxyVertex (32 B): local_pos, normal_oct, material_packed,
        // color_packed, _reserved[2]. Locations match `mesh_proxy.wesl`.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<rkp_core::mesh_extract::ProxyVertex>() as u64,
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
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_proxy"),
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
                // Surface-nets extractor emits outward CCW winding —
                // back-face cull is safe and roughly halves frag work.
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GBUFFER_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
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
                // Five attachments — order locked by `mesh_proxy.wesl`'s
                // `FsOut`: position, pick, normal, material, glass.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_POSITION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_GLASS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            g0_layout,
            g1_layout,
        }
    }

    /// Build the camera bind group for this pass. Camera uniform buffer
    /// is the same one driving the rest of the renderer.
    pub fn create_g0_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_proxy g0"),
            layout: &self.g0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        })
    }

    pub fn create_g1_bind_group(
        &self,
        device: &wgpu::Device,
        instance_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_proxy g1"),
            layout: &self.g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.as_entire_binding(),
            }],
        })
    }

    /// Begin the proxy raster render pass. **Loads** all attachments —
    /// the mesh raster + mesh_resolve already wrote the G-buffer for
    /// their pixels, and proxy meshes composite on top via depth-test
    /// against the shared depth buffer.
    pub fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        position_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) -> wgpu::RenderPass<'a> {
        let load_op_color = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_proxy render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: position_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: pick_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: material_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: glass_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    #[cfg(test)]
    fn shader_source() -> &'static str {
        wesl::include_wesl!("mesh_proxy")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_is_valid_wgsl() {
        let src = MeshProxyPass::shader_source();
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
