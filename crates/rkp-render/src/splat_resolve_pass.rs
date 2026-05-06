//! `SplatResolvePass` — compute fixup for the splat path's G-buffer.
//!
//! The splat raster pass writes only the visibility-buffer triplet
//! (position, pick, leaf_slot) — staying within wgpu's default 32 B/sample
//! color-attachment limit. This pass reads that triplet and fills in
//! the rest of the G-buffer (normal, material, glass) by chasing the
//! same `leaf_attr_pool` / `color_pool` / `instances` indirection
//! `octree_march` would. See `shaders/splat_resolve.wesl` for the
//! WGSL contract.

/// Compute pipeline + bind group layouts for the splat resolve pass.
/// One instance shared across all viewports — the per-VR resources
/// (texture views + scene buffers) live on `ViewportRenderer`.
pub struct SplatResolvePass {
    pub pipeline: wgpu::ComputePipeline,
    pub g0_layout: wgpu::BindGroupLayout,
    pub g1_layout: wgpu::BindGroupLayout,
}

impl SplatResolvePass {
    pub fn new(device: &wgpu::Device) -> Self {
        // ── g0: per-VR textures ─────────────────────────────────────
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_resolve g0"),
            entries: &[
                // leaf_slot_in (R32Uint, sampled)
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
                // pick_in (R32Uint, sampled)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // normal_out (Rgba16Float, storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: crate::gbuffer::GBUFFER_NORMAL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // material_out (Rg32Uint, storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: crate::gbuffer::GBUFFER_MATERIAL_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // glass_out (Rg32Uint, storage write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: crate::gbuffer::GBUFFER_GLASS_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ── g1: scene-wide buffers driving the lookup ──────────────
        let storage_ro = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_resolve g1"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, ..storage_ro },
                wgpu::BindGroupLayoutEntry { binding: 1, ..storage_ro },
                wgpu::BindGroupLayoutEntry { binding: 2, ..storage_ro },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_resolve pipeline layout"),
            bind_group_layouts: &[Some(&g0_layout), Some(&g1_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("splat_resolve"),
            "splat_resolve",
        );

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_resolve"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, g0_layout, g1_layout }
    }

    /// Build the per-VR `g0` bind group. Rebuild after a viewport
    /// resize — every gbuffer view changes.
    pub fn create_g0_bind_group(
        &self,
        device: &wgpu::Device,
        leaf_slot_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_resolve g0 bg"),
            layout: &self.g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(leaf_slot_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(pick_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(material_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(glass_view) },
            ],
        })
    }

    /// Build the scene-wide `g1` bind group. Rebuild after a scene-
    /// buffers epoch bump — `leaf_attr_pool` / `color_pool` / `instances`
    /// can move underneath us when the scene resizes its pools.
    pub fn create_g1_bind_group(
        &self,
        device: &wgpu::Device,
        leaf_attr_pool_buffer: &wgpu::Buffer,
        color_pool_buffer: &wgpu::Buffer,
        instances_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_resolve g1 bg"),
            layout: &self.g1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: leaf_attr_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: color_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: instances_buffer.as_entire_binding() },
            ],
        })
    }

    /// One dispatch covers the full viewport at 8×8 tile granularity
    /// (matching the workgroup size in `splat_resolve.wesl`).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        g0_bg: &wgpu::BindGroup,
        g1_bg: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("splat_resolve"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, g0_bg, &[]);
        cpass.set_bind_group(1, g1_bg, &[]);
        cpass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}
