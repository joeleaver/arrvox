//! `SplatPass` — render pipeline owner for the surface-splat path.
//!
//! Visibility-buffer architecture: this pass writes only what the raster
//! can produce uniquely (world position, per-leaf id, per-instance id).
//! A separate compute pass (`splat_resolve_pass`) reads those and fills
//! in the rest of the G-buffer (normal / material / glass). The split
//! exists so the raster pass fits under wgpu's default
//! `max_color_attachment_bytes_per_sample` limit (32 B; we use 24 B
//! across three attachments).
//!
//! Two bind groups:
//!  * `g0` — scene-wide: camera (`CameraUniforms`, 224 B) + `leaf_attr_pool`
//!    (storage<read>, the vertex shader reads `normal_oct` to build the
//!    disc tangent basis). Bind once per pass.
//!  * `g1` — per-instance: a 80 B `SplatInstanceUniform` carrying the
//!    instance's world matrix + `object_id`. One bind group per scene
//!    instance; rebound between draws.

use crate::gbuffer::{GBUFFER_DEPTH_FORMAT, GBUFFER_POSITION_FORMAT};
use crate::rkp_scene::CameraUniforms;

use super::extract::SplatVertex;

const GBUFFER_PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const GBUFFER_LEAF_SLOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Per-instance uniform — one mat4 world transform, the entity's
/// `object_id` (written into the pick texture), and the per-instance
/// bone-skinning state (Phase 6.6). 80 B, multiple of 16. CPU mirror
/// of the `SplatInstance` struct in `splat.wesl` and the matching
/// declaration in the mesh raster + shadow shaders.
///
/// **Skinning semantics:**
/// * `skinning_mode == SKINNING_MODE_NONE` → instance is not skinned
///   (no live bone matrices); the mesh VS skips skinning entirely
///   regardless of per-vertex `bone_weights` and emits the rest-pose
///   transform.
/// * `skinning_mode == 0` → linear blend skinning; the VS reads
///   `bone_matrices[bone_offset_lbs + bone_idx]` for the four
///   referenced bones, weighted-sums, and applies.
/// * `skinning_mode == 1` → dual-quaternion skinning; the VS reads
///   `bone_dual_quats[bone_offset_dqs + bone_idx]`, blends, and
///   normalises before applying.
///
/// The two offsets are independent — LBS and DQS palettes are sized
/// and packed separately by `BoneMatrixAllocator`. The unused offset
/// is harmless filler.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplatInstanceUniform {
    pub world: [[f32; 4]; 4],
    pub object_id: u32,
    /// First index in `bone_matrices` for this instance's LBS palette.
    pub bone_offset_lbs: u32,
    /// First index in `bone_dual_quats` for this instance's DQS palette.
    pub bone_offset_dqs: u32,
    /// `0` = LBS, `1` = DQS, `SKINNING_MODE_NONE` = not skinned.
    pub skinning_mode: u32,
}

/// Sentinel `skinning_mode` value meaning "this instance carries no
/// live bone matrices; render rest pose." Lives in the value space of
/// `u32` outside the LBS / DQS enum so the VS can branch on it without
/// an extra "is_skinned" flag.
pub const SKINNING_MODE_NONE: u32 = u32::MAX;

const _: () = assert!(std::mem::size_of::<SplatInstanceUniform>() == 80);
pub const SPLAT_INSTANCE_BYTES: u64 = std::mem::size_of::<SplatInstanceUniform>() as u64;

/// Splat-rasterizer GPU pass.
pub struct SplatPass {
    pub pipeline: wgpu::RenderPipeline,
    pub g0_layout: wgpu::BindGroupLayout,
    pub g1_layout: wgpu::BindGroupLayout,
}

impl SplatPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // ── g0: scene-wide bindings ────────────────────────────────
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat g0"),
            entries: &[
                // camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<CameraUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // leaf_attr_pool (storage<read>) — vertex stage uses
                // `normal_oct` to build the disc basis. Materials and
                // colours are read by the resolve compute pass, not
                // here.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── g1: per-instance uniform ───────────────────────────────
        // Visibility includes COMPUTE so the same per-VR
        // `splat_instance_bind_groups` drive both the splat / mesh
        // render pipelines (vertex+fragment) AND the Phase 6.2
        // `mesh_lod_select` compute pass — one bind group, one layout.
        let g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat g1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(SPLAT_INSTANCE_BYTES),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat pipeline layout"),
            bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("splat"),
            "splat",
        );

        // SplatVertex: 32 B per instance. (See `extract::SplatVertex`.)
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
                // Three-attachment visibility-buffer output:
                //   position (16) + pick (4) + leaf_slot (4) = 24 B.
                // Order is locked by `splat.wesl`'s `FsOut`.
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

    /// Build the scene-wide `g0` bind group. Bound once per pass, before
    /// the per-instance draws.
    pub fn create_g0_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        leaf_attr_pool_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat g0 bg"),
            layout: &self.g0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: leaf_attr_pool_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Build a per-instance `g1` bind group. One per scene-instance.
    pub fn create_g1_bind_group(
        &self,
        device: &wgpu::Device,
        instance_uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat g1 bg"),
            layout: &self.g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_uniform_buffer.as_entire_binding(),
            }],
        })
    }

    /// Begin a splat render pass on the supplied G-buffer attachments.
    /// All three color targets (and the depth buffer) are cleared to
    /// the same miss sentinels `octree_march` writes for non-hit
    /// pixels:
    ///
    /// | target      | clear value                  |
    /// |-------------|------------------------------|
    /// | position    | (0, 0, 0, 1e10)              |
    /// | pick        | 0xFFFFFFFF                   |
    /// | leaf_slot   | 0                            |
    /// | depth       | 1.0                          |
    ///
    /// The remaining G-buffer entries (normal / material / glass) are
    /// **not** cleared by this pass — the resolve compute pass writes
    /// them per-pixel based on the leaf_slot value at the same coord.
    /// For miss pixels the resolve pass writes zeros, matching what
    /// the compute march writes for non-hits.
    pub fn begin_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        position_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        leaf_slot_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat render"),
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
                    // miss_pick_id = 0xFFFFFFFF (see octree_march.wesl
                    // line 1245). wgpu casts the .r component of the
                    // f64 Color struct to u32 for Uint targets.
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
