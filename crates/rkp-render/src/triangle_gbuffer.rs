//! Triangle G-buffer pass — rasterizes marching-cubes meshes into the
//! deferred G-buffer.
//!
//! Runs *after* the compute march during the Phase 1 A/B transition. Only
//! objects that own a [`MeshAllocation`] in [`MeshPool`] are drawn here; their
//! pixels overwrite whatever the march wrote via the depth test. Shading,
//! SSAO, and every downstream pass are unchanged.
//!
//! One draw call per object, each with instance_count=1. We use
//! `first_instance = gpu_object_index` so the vertex shader can read
//! `objects[instance_index]` for the world matrix and object_id.

use crate::mesh_pool::{MeshAllocation, MeshPool, MeshVertex};
use crate::validate_wgsl;

#[cfg(test)]
mod tests {
    #[test]
    fn shader_parses_and_validates() {
        let src = include_str!("shaders/triangle_gbuffer.wgsl");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("triangle_gbuffer.wgsl parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("triangle_gbuffer.wgsl validation error: {e:?}"));
    }
}
use rkf_render::gbuffer::{
    GBUFFER_DEPTH_FORMAT, GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT,
};

/// A single triangle-mesh draw call for this frame. Built by the engine each
/// frame from `gpu_objects ↔ object_id ↔ MeshPool.get(...)`.
#[derive(Debug, Copy, Clone)]
pub struct MeshDraw {
    /// Index into the bound objects storage array — drives `instance_index`.
    pub gpu_object_index: u32,
    /// Allocation in the [`MeshPool`] holding this object's geometry.
    pub allocation: MeshAllocation,
}

/// Triangle rasterization pass writing to the deferred G-buffer.
pub struct TriangleGBufferPass {
    pipeline: wgpu::RenderPipeline,
}

impl TriangleGBufferPass {
    /// Build the pipeline against the scene (group 0) layout — the shader
    /// reads `objects` and `camera` from it.
    pub fn new(device: &wgpu::Device, scene_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let shader_src = include_str!("shaders/triangle_gbuffer.wgsl");
        validate_wgsl(shader_src, "triangle_gbuffer");
        // Catch validation errors at pipeline creation — without this the
        // device error is deferred and surfaces only as "pipeline invalid"
        // at submit time with no diagnostic.
        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_gbuffer"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle_gbuffer pipeline layout"),
            bind_group_layouts: &[Some(scene_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle_gbuffer"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[MeshVertex::vertex_layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Paul-Bourke MC tables give vertex order such that the
                // triangle's right-hand-rule normal points INWARD — to cull
                // back faces correctly we treat Cw winding as the front face.
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GBUFFER_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    // Position target dropped — world position reconstructed
                    // in downstream passes from depth + inverse_view_proj.
                    // This saves 16 B/fragment of ROP write bandwidth.
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
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        if let Some(err) = pollster::block_on(err_scope.pop()) {
            panic!("triangle_gbuffer pipeline creation failed: {err}");
        }

        Self { pipeline }
    }

    /// Record a render pass writing the G-buffer for all supplied draws.
    ///
    /// The depth attachment is always cleared to 1.0 at the start — depth
    /// doesn't persist across frames.
    ///
    /// `clear_gbuffer` controls the color attachment load op: `true` clears
    /// (used when no other pass writes the G-buffer — e.g. `mesh_only_mode`
    /// skipping the compute march), `false` loads (the default during the
    /// A/B transition when the march writes first and triangles overwrite).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bind_group: &wgpu::BindGroup,
        gbuffer: &rkf_render::GBuffer,
        mesh_pool: &MeshPool,
        draws: &[MeshDraw],
        clear_gbuffer: bool,
    ) {
        if draws.is_empty() && !clear_gbuffer {
            return;
        }

        let color_load = if clear_gbuffer {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("triangle_gbuffer"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.normal_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.material_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Material is Rg32Uint — wgpu's Color::TRANSPARENT
                        // (zeroes) maps to (0,0) uint, which is correct for
                        // "no hit" (object_id offset by +1 in shading).
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
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
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.set_vertex_buffer(0, mesh_pool.vertex_buffer().slice(..));
        pass.set_index_buffer(
            mesh_pool.index_buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );

        for d in draws {
            if d.allocation.is_empty() {
                continue;
            }
            let first_index = d.allocation.index_start;
            let last_index = first_index + d.allocation.index_count;
            pass.draw_indexed(
                first_index..last_index,
                0, // base_vertex — our indices are already pool-absolute
                d.gpu_object_index..d.gpu_object_index + 1,
            );
        }
    }
}
