//! `MeshShadowMapPass` — directional shadow map rendered from the
//! mesh path's triangle geometry.
//!
//! Phase 3 of the splat-to-mesh pivot. Two-pipeline pass:
//!
//!  1. **Render** — depth-only rasterization. Vertex stage projects
//!     the mesh through the light camera's view-proj; the rasterizer
//!     fills a `Depth32Float` attachment. NO fragment shader, so the
//!     GPU's early-z culling is at full strength.
//!  2. **Blit** — single-thread-per-texel compute pass that reads the
//!     depth texture and writes `bitcast<u32>(depth)` into the
//!     existing `shadow_buffer` storage that `shadow_map_pass` owns
//!     and shade already samples. No atomics needed — each thread
//!     writes a unique texel.
//!
//! Splitting the work this way means the per-fragment cost is only
//! the GPU's fixed depth-write, with no shader side-effect that would
//! disable early-z. The blit adds ~0.3 ms but lets the render itself
//! drop to a fraction of the previous fragment-atomic approach.
//!
//! Bind groups:
//!  · render `g0` — light_camera uniform.
//!  · render `g1` — per-instance uniform (SHARED with `SplatPass.g1_layout`).
//!  · blit `g0` — depth texture (read) + shadow_buffer (write).

use crate::shadow_map_pass::types::SHADOW_MAP_DEFAULT_SIZE;

use rkp_core::mesh_extract::MeshVertex;

/// Render-pipeline owner for the mesh path's directional shadow map.
/// One pipeline shared across viewports — per-VR state (depth texture,
/// g0 bind group) lives in `ViewportRenderer`.
pub struct MeshShadowMapPass {
    /// Depth-only render pipeline. No fragment shader, no color
    /// attachments — only writes the depth attachment.
    pub render_pipeline: wgpu::RenderPipeline,
    /// Render-pass `g0` layout — just the light_camera uniform.
    pub render_g0_layout: wgpu::BindGroupLayout,
    /// Compute pipeline that copies the depth attachment into the
    /// shadow_buffer (with f32→u32 bitcast).
    pub blit_pipeline: wgpu::ComputePipeline,
    /// Blit-pass `g0` layout — depth texture (read) + shadow_buffer
    /// (write).
    pub blit_g0_layout: wgpu::BindGroupLayout,
    /// Shadow-map side length in texels. Matches `SHADOW_MAP_DEFAULT_SIZE`.
    pub size: u32,
}

impl MeshShadowMapPass {
    pub fn new(
        device: &wgpu::Device,
        splat_g1_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // ── Render pipeline (depth-only, no fragment shader) ───────
        let render_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_shadow render g0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_shadow render layout"),
            bind_group_layouts: &[Some(&render_g0_layout), Some(splat_g1_layout)],
            immediate_size: 0,
        });

        let render_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_shadow"),
            "mesh_shadow",
        );

        // Vertex layout matches `MeshVertex`; only `local_pos` is read.
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
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_shadow render"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_module,
                entry_point: Some("vert_main"),
                compilation_options: Default::default(),
                buffers: &[vertex_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Front-face cull (back faces write depth) — standard
                // shadow-map trick to mitigate self-shadow acne on
                // lit surfaces.
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
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
            // No fragment stage — depth-only render. The rasterizer
            // fills the depth attachment; the GPU's early-z culling
            // runs at full speed because there's no fragment side
            // effect that would force late-z.
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        // ── Blit pipeline (compute: depth tex → shadow_buffer) ─────
        let blit_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_shadow blit g0"),
            entries: &[
                // depth_tex — texture_depth_2d, sampled untyped.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // shadow_buffer — `array<u32>` storage, written
                // single-thread-per-texel so non-atomic is fine.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_shadow blit layout"),
            bind_group_layouts: &[Some(&blit_g0_layout)],
            immediate_size: 0,
        });

        let blit_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_shadow_blit"),
            "mesh_shadow_blit",
        );

        let blit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh_shadow blit"),
            layout: Some(&blit_pipeline_layout),
            module: &blit_module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            render_pipeline,
            render_g0_layout,
            blit_pipeline,
            blit_g0_layout,
            size: SHADOW_MAP_DEFAULT_SIZE,
        }
    }

    /// Build the per-VR render `g0` bind group — the light_camera
    /// uniform.
    pub fn create_render_g0_bind_group(
        &self,
        device: &wgpu::Device,
        light_camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_shadow render g0 bg"),
            layout: &self.render_g0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_camera_buffer.as_entire_binding(),
            }],
        })
    }

    /// Build the per-VR blit `g0` bind group — depth texture view +
    /// shadow_buffer.
    pub fn create_blit_g0_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        shadow_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_shadow blit g0 bg"),
            layout: &self.blit_g0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shadow_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Begin the depth-only render pass. No color attachments — only
    /// a depth attachment that the rasterizer fills directly. Depth
    /// clears to 1.0 each frame so uncovered texels stay at FAR_DEPTH
    /// (the blit copies that through to the shadow_buffer's
    /// `SHADOW_MAP_FAR_DEPTH_BITS`).
    pub fn begin_render_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_shadow render"),
            color_attachments: &[],
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

    /// Dispatch the depth → shadow_buffer copy. One workgroup per
    /// 8×8 texel tile, writes one u32 per texel.
    pub fn dispatch_blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        blit_g0_bg: &wgpu::BindGroup,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh_shadow blit"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.blit_pipeline);
        cpass.set_bind_group(0, blit_g0_bg, &[]);
        let groups = self.size.div_ceil(8);
        cpass.dispatch_workgroups(groups, groups, 1);
    }

    #[cfg(test)]
    fn render_shader_source() -> &'static str {
        wesl::include_wesl!("mesh_shadow")
    }

    #[cfg(test)]
    fn blit_shader_source() -> &'static str {
        wesl::include_wesl!("mesh_shadow_blit")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }

    #[test]
    fn mesh_shadow_render_shader_is_valid_wgsl() {
        validate(MeshShadowMapPass::render_shader_source());
    }

    #[test]
    fn mesh_shadow_blit_shader_is_valid_wgsl() {
        validate(MeshShadowMapPass::blit_shader_source());
    }
}
