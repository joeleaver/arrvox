//! Infinite world-space grid overlay.
//!
//! Render-pipeline pass that draws an anti-aliased XZ grid into the
//! composite texture (LDR, post-tonemap) using the gbuffer position
//! buffer as a depth-occlusion source. Used by the build viewport in
//! `RenderMode::Isolation` to give a clean studio backdrop.

use crate::validate_wgsl;

/// Uniform parameters for the grid pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    /// Distance at which the grid fades fully out, world units. Picked
    /// so the grid is legible across the typical build-viewport zoom
    /// range without aliasing into a flat plate at the horizon.
    pub fade_distance: f32,
    /// 0 = pass renders nothing (early-out in fragment shader). Used so
    /// the host can share one bind group across both modes if needed.
    pub enabled: u32,
    pub _pad0: [u32; 2],
    /// World-space origin for the grid plane. `xyz` = (x, y, z) of
    /// the grid's "zero" point; lines are drawn at `hit.xz - plane.xz`
    /// on the plane `y = plane.y`. Default `(0, 0, 0)` keeps the
    /// classic world-origin grid; the build viewport overrides this
    /// with the previewed entity's world position so the grid always
    /// sits under the object and the red/blue origin axes cross
    /// through it, no matter where the object has been moved in the
    /// scene. `w` is padding (vec4 alignment).
    pub plane_origin: [f32; 4],
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            fade_distance: 50.0,
            enabled: 1,
            _pad0: [0; 2],
            plane_origin: [0.0; 4],
        }
    }
}

pub struct RkpGridPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
}

impl RkpGridPass {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rkp_grid layout"),
            entries: &[
                // 0: camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: grid params (uniform)
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
                // 2: gbuf_position (sampled, world-space + t in .w)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_grid params"),
            size: std::mem::size_of::<GridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader_src = wesl::include_wesl!("rkp_grid");
        validate_wgsl(shader_src, "rkp_grid");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rkp_grid"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_grid pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rkp_grid"),
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
                        // Pre-multiplied-alpha blend: src.rgb already
                        // includes the alpha factor in the shader.
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

    /// Wire the camera + gbuffer bindings. Call after VR construction
    /// and after every resize that recreates the gbuffer position view.
    pub fn set_bindings(
        &mut self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        gbuf_position_view: &wgpu::TextureView,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_grid bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(gbuf_position_view),
                },
            ],
        }));
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &GridParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Draw the grid into `target`. Uses LoadOp::Load so existing
    /// composite contents are preserved and the grid blends on top.
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let bg = match &self.bind_group {
            Some(bg) => bg,
            None => return,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rkp_grid"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
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
