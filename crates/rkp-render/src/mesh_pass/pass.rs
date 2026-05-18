//! `MeshPass` — render-pipeline owner for the surface-mesh path.
//!
//! Writes the visibility-buffer triplet (position, pick, leaf_slot) per
//! fragment plus the rest-pos target. The `mesh_resolve` compute pass
//! reads those and fills in the rest of the G-buffer per-pixel.
//!
//! Bind-group layouts come from `MeshInstanceLayouts` (`mesh_instance`
//! module) and are shared with every other raster path — mesh shadow,
//! mesh LOD select, mesh glass, user-shader mesh — so `ViewportRenderer`
//! builds one set of g0 / g1 bind groups and feeds all consumers.
//!
//! Pipeline shape:
//!   · `TriangleList` topology with an index buffer.
//!   · `cull_mode: Some(Back)` — the surface-nets extractor emits
//!     CCW-outward winding so back-face cull is safe and ~halves the
//!     fragment work.
//!   · Vertex layout: per-vertex — `MeshVertex` at a 32 B stride,
//!     locations matching `extract::MeshVertex`.

use crate::gbuffer::{GBUFFER_DEPTH_FORMAT, GBUFFER_POSITION_FORMAT, GBUFFER_REST_POS_FORMAT};

use rkp_core::mesh_extract::MeshVertex;

// Visibility-buffer formats. `mesh_resolve` is the sole consumer.
const GBUFFER_PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const GBUFFER_LEAF_SLOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Surface-mesh GPU pass.
pub struct MeshPass {
    pub pipeline: wgpu::RenderPipeline,
}

impl MeshPass {
    /// Build the mesh raster pipeline. Reuses `g0_layout` / `g1_layout`
    /// from `MeshInstanceLayouts` so the same bind groups drive every
    /// raster pipeline. `g2_layout` carries the glass-classification
    /// bindings (`leaf_attr_pool` + `materials` + `instances` +
    /// `instance_overlay` + `color_pool_data`); the FS reads them to
    /// `discard` glass fragments so `gbuf_position` only carries
    /// opaque-hit depth — glass is rasterised separately in
    /// `mesh_glass`. Reuse `MeshGlassPass::g2_layout` so a single bind
    /// group drives both passes.
    pub fn new(
        device: &wgpu::Device,
        g0_layout: &wgpu::BindGroupLayout,
        g1_layout: &wgpu::BindGroupLayout,
        g2_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh pipeline layout"),
            bind_group_layouts: &[Some(g0_layout), Some(g1_layout), Some(g2_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh"),
            "mesh",
        );

        // MeshVertex: 32 B per vertex. Layout matches
        // `extract::MeshVertex` (local_pos, normal_oct, leaf_attr_id,
        // bone_indices, bone_weights, _pad). `normal_oct` is currently
        // unused by the shader (the resolve pass reads
        // `LeafAttr.normal_oct` from the pool); it's declared in the
        // layout to keep the buffer reader aligned. `bone_indices` /
        // `bone_weights` are the Phase 6.6 skinning attributes — the
        // VS reads them only when the per-instance `skinning_mode`
        // selects LBS or DQS.
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
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
                // Visibility-buffer triplet + rest_pos. Order locked
                // by `mesh.wesl`'s `FsOut`. The 4th target carries the
                // pre-skin mesh-frame rest position so `mesh_resolve`
                // can descend the asset's octree per pixel — fixes the
                // chunky-per-triangle look of `@interpolate(flat)
                // leaf_attr_id`.
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
                        format: GBUFFER_LEAF_SLOT_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: GBUFFER_REST_POS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    #[cfg(test)]
    fn shader_source() -> &'static str {
        wesl::include_wesl!("mesh")
    }

    /// Begin a mesh render pass on the supplied G-buffer attachments.
    /// Clear values:
    ///
    /// | target      | clear value     |
    /// |-------------|-----------------|
    /// | position    | (0, 0, 0, 1e10) |
    /// | pick        | 0xFFFFFFFF      |
    /// | leaf_slot   | 0               |
    /// | rest_pos    | (0, 0, 0, 0)    |
    /// | depth       | 1.0             |
    ///
    /// `rest_pos` clears to .w = 0 (the "no rest_pos written" sentinel)
    /// so any miss pixel reads as "fall back to leaf_slot" in
    /// `mesh_resolve`.
    pub fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        position_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        leaf_slot_view: &wgpu::TextureView,
        rest_pos_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh render"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: position_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0e10,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: pick_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 4_294_967_295.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: leaf_slot_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: rest_pos_view,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_shader_is_valid_wgsl() {
        let src = MeshPass::shader_source();
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
