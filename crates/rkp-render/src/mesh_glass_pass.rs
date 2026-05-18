//! Mesh-mode glass pipeline.
//!
//! Three GPU stages run between the primary mesh raster (which writes
//! the opaque visibility buffer) and the existing `rkp_glass`
//! composite (which reads `gbuf_glass` to do screen-space refraction
//! + Beer + Fresnel). The composite is reused unchanged — this module
//! produces the exact same `gbuf_glass` packing the march used to
//! emit, so all downstream PBR / shadow / volumetric passes are
//! oblivious to which primary path produced the glass.
//!
//! 1. **Front raster** (`MeshGlassPass::front_pipeline`) — `cull = Back`,
//!    `depth_compare = Less`. One indexed draw per glass-bearing
//!    instance. The FS reads `leaf_attr_pool[leaf_attr_id].material`
//!    and `discard`s on `opacity ≥ 0.99`. Otherwise writes
//!    `(oct_normal, material_id, bitcast<u32>(entry_dist), 0)` to
//!    `glass_entry_packed` (Rgba32Uint).
//!
//! 2. **Back raster** (`MeshGlassPass::back_pipeline`) — `cull = Front`,
//!    `depth_compare = Greater`, depth target initialised to 0.0 so
//!    the *farthest* back-face wins. Same per-cell glass `discard`,
//!    writes the world-space exit distance to `glass_exit_dist`
//!    (R32Float).
//!
//! 3. **Combine compute** (`MeshGlassPass::combine_pipeline`) — reads
//!    the front + back targets plus `gbuf_position.w` (opaque hit
//!    distance), gates glass behind the closest opaque, and packs the
//!    final Rg32Uint `gbuf_glass`. Output layout matches
//!    `octree_march`'s glass write byte-for-byte so `rkp_glass.wesl`
//!    runs unchanged.
//!
//! Bind groups (raster passes):
//!   · `g0` — camera + bones (reuses splat path's `g0_layout`).
//!   · `g1` — per-instance `MeshInstance` (reuses splat path's
//!            `g1_layout` and the existing `splat_instance_bind_groups`
//!            in `ViewportRenderer`).
//!   · `g2` — `leaf_attr_pool` (binding 0) + `materials` (binding 1).
//!            Scene-global; rebuilt by the caller when either backing
//!            buffer reallocates.
//!
//! Bind groups (combine compute):
//!   · `g0` — `glass_entry_packed` (sampled uint) + `glass_exit_dist`
//!            (sampled f32) + `gbuf_position` (sampled f32) +
//!            `gbuf_glass_out` (storage Rg32Uint write).

/// Render-target format for the entry pass — packs world entry normal
/// (R), material id (G, low 16), bitcast f32 entry distance (B), 0
/// (A). Read in `mesh_glass_combine.wesl`.
pub const GLASS_ENTRY_PACKED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

/// Render-target format for the exit pass — world-space exit distance
/// from the camera, in metres. R32Float so the combine compute can
/// read it directly without bitcasting.
pub const GLASS_EXIT_DIST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Depth target format for both glass raster passes. Separate from
/// the primary G-buffer depth so neither pass clobbers the opaque
/// hit's depth buffer.
pub const GLASS_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sentinel value the entry FS encodes when no glass fragment writes
/// the pixel. Read by `mesh_glass_combine.wesl`; values stored as
/// `bitcast<u32>(ENTRY_SENTINEL)` are the cleared state of
/// `glass_entry_packed.b`. Anything `≥ 1e30` is treated as "no entry".
pub const GLASS_ENTRY_SENTINEL: f32 = 1.0e30;

/// CPU mirror of `mesh_glass_combine.wesl`'s `CombineParams` uniform.
/// 16 B, std140-aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CombineParams {
    pub debug_force: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<CombineParams>() == 16);

/// CPU mirror of `mesh_glass.wesl`'s `GlassFsParams` uniform — the
/// FS discard threshold. Production = 0.99 (matches the march).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlassFsParams {
    pub opacity_threshold: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<GlassFsParams>() == 16);

/// Default FS opacity threshold — same as the march's
/// `m_opacity >= 0.99` glass classification.
pub const DEFAULT_OPACITY_THRESHOLD: f32 = 0.99;

pub struct MeshGlassPass {
    /// Front-face raster (entry packing).
    pub front_pipeline: wgpu::RenderPipeline,
    /// Back-face raster (exit distance).
    pub back_pipeline: wgpu::RenderPipeline,
    /// Combine compute (writes final `gbuf_glass`).
    pub combine_pipeline: wgpu::ComputePipeline,
    /// `g2` for raster passes: leaf_attr_pool + materials.
    pub g2_layout: wgpu::BindGroupLayout,
    /// `g0` for the combine compute pass.
    pub combine_layout: wgpu::BindGroupLayout,
}

impl MeshGlassPass {
    pub fn new(
        device: &wgpu::Device,
        g0_layout: &wgpu::BindGroupLayout,
        g1_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Five storage-buffer bindings: leaf_attr_pool, materials,
        // instances (RkpInstance array), instance_overlay, color_pool
        // — last is unused at FS time but the imported
        // `lib::leaf_attr` module's `fetch_leaf_color_for` references
        // the symbol at parse time (the build.rs runs with
        // `use_stripping(false)`, so unused-by-DCE bindings still
        // need to be declared).
        let storage_ro_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // g2 layout — 0..4 are the glass-classify essentials shared
        // with `mesh.wesl`/`mesh_shadow.wesl`. 5 is the FS-side debug
        // params uniform. 6..9 are the octree-lookup bindings (used
        // only by `mesh_glass.wesl`'s per-pixel resolved normal). The
        // other consumers ignore 6..9 — wgpu accepts a layout with
        // bindings the shader doesn't reference.
        let g2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_glass g2 (leaf_attr_pool + materials + instances + overlay + color_pool + fs_params + octree-lookup + sculpt)"),
            entries: &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_ro_entry(3),
                storage_ro_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro_entry(6),   // assets
                storage_ro_entry(7),   // octree_nodes
                storage_ro_entry(8),   // brick_pool
                storage_ro_entry(9),   // brick_face_links
                storage_ro_entry(10),  // instance_sculpt (Phase A sculpt overlay)
            ],
        });

        let raster_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_glass raster pipeline layout"),
            bind_group_layouts: &[Some(g0_layout), Some(g1_layout), Some(&g2_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_glass"),
            "mesh_glass",
        );

        // Vertex layout matches `MeshPass` exactly — same MeshVertex
        // buffer feeds primary + glass passes.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<rkp_core::mesh_extract::MeshVertex>() as u64,
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
            label: Some("mesh_glass front"),
            layout: Some(&raster_pipeline_layout),
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
                // Cull back-faces — front pass collects entry surface.
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GLASS_DEPTH_FORMAT,
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
                entry_point: Some("frag_front"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GLASS_ENTRY_PACKED_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let back_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_glass back"),
            layout: Some(&raster_pipeline_layout),
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
                // Cull front-faces — back pass collects exit surface.
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GLASS_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Greater so the *farthest* back-face wins — matches
                // the march's `glass_exit_t = max(glass_exit_t, t)`.
                // Caller clears the depth target to 0.0 (near) so any
                // valid hit passes the initial test.
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
                entry_point: Some("frag_back"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GLASS_EXIT_DIST_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let combine_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_glass_combine g0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: crate::gbuffer::GBUFFER_GLASS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let combine_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_glass_combine pipeline layout"),
            bind_group_layouts: &[Some(&combine_layout)],
            immediate_size: 0,
        });

        let combine_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_glass_combine"),
            "mesh_glass_combine",
        );

        let combine_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh_glass_combine"),
            layout: Some(&combine_pipeline_layout),
            module: &combine_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            front_pipeline,
            back_pipeline,
            combine_pipeline,
            g2_layout,
            combine_layout,
        }
    }

    /// Begin the front-face glass pass. Caller binds the pipeline,
    /// g0/g1/g2, and issues per-instance indexed draws (or indirect
    /// LOD draws). Clear values:
    ///
    /// | target            | clear                            |
    /// |-------------------|----------------------------------|
    /// | entry_packed      | (0, 0, bitcast(GLASS_ENTRY_SENTINEL), 0) |
    /// | depth (front)     | 1.0                              |
    pub fn begin_front_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        entry_packed_view: &wgpu::TextureView,
        depth_front_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_glass front"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: entry_packed_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        // bitcast<u32>(GLASS_ENTRY_SENTINEL) — clear
                        // value is interpreted as four f64 channels by
                        // wgpu. For uint render targets, the value
                        // gets reinterpreted bit-wise as the channel
                        // type. Encoding 1e30 as u32 ≈ 0x7149f2ca, so
                        // we set b to that magnitude as f64.
                        b: f32::to_bits(GLASS_ENTRY_SENTINEL) as f64,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_front_view,
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

    /// Begin the back-face glass pass. Depth target clears to 0.0
    /// (near) so the Greater compare admits the first fragment, then
    /// each subsequent farther fragment overwrites.
    ///
    /// | target          | clear |
    /// |-----------------|-------|
    /// | exit_dist       | 0.0   |
    /// | depth (back)    | 0.0   |
    pub fn begin_back_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        exit_dist_view: &wgpu::TextureView,
        depth_back_view: &wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh_glass back"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: exit_dist_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_back_view,
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

    /// Run the combine compute. Writes the final `gbuf_glass`.
    pub fn dispatch_combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh_glass_combine"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.combine_pipeline);
        pass.set_bind_group(0, bg, &[]);
        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    

    #[test]
    fn mesh_glass_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("mesh_glass");
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
    fn mesh_glass_combine_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("mesh_glass_combine");
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
