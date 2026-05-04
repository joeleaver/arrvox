//! Half-resolution shadow trace compute pass.
//!
//! Reads the full-res G-buffer (position + normal), traces shadow rays for
//! each shadow-casting light through the scene octree, writes a half-res
//! `rgba8unorm` shadow texture. The shade pass upsamples this with a
//! position/normal-weighted bilateral gather; quality approaches full-res
//! at ~25% of the shadow-trace cost.

use crate::compile_pass_shader;

pub struct ShadowTracePass {
    pipeline: wgpu::ComputePipeline,
    /// Kept around so `reload_user_shaders` can rebuild the pipeline
    /// against the same bind-group layouts when user-shader chunks
    /// change. Phase 4c.
    pipeline_layout: wgpu::PipelineLayout,
    /// Hash of the user-shader source mix this pipeline was last built
    /// against. Same semantics as `OctreeMarchPass`.
    user_shader_source_hash: u64,
    /// Group 1 layout: full-res gbuf reads + half-res shadow write.
    io_bind_group_layout: wgpu::BindGroupLayout,
    io_bind_group: Option<wgpu::BindGroup>,
    /// Phase 4 — kept for future binding rebuilds. The shadow trace
    /// shares `OctreeMarchPass`'s `params_bind_group_layout` (which
    /// already includes binding 10 for `shader_params`); rebuilding
    /// the actual bind group lives on `OctreeMarchPass`. Storing the
    /// buffer handle here keeps the API symmetric with the primary
    /// march for future-proofing if shadow grows its own params bind
    /// group.
    shader_params_buffer: Option<wgpu::Buffer>,
    /// Half-res shadow output texture.
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    half_w: u32,
    half_h: u32,
}

impl ShadowTracePass {
    pub fn new(
        device: &wgpu::Device,
        full_w: u32,
        full_h: u32,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
        params_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let (half_w, half_h) = half_res_dims(full_w, full_h);

        // Group 1: gbuf_position (read), gbuf_normal (read), shadow_lo_res (write).
        let io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_trace io layout"),
                entries: &[
                    // gbuf_position: sampled texture, rgba32float
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // gbuf_normal: sampled texture, rgba16float
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
                    // shadow_lo_res: storage texture (write, rgba8unorm)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let module = compile_pass_shader(device, wesl::include_wesl!("rkp_shadow_trace"), "rkp_shadow_trace");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_shadow_trace pipeline layout"),
            bind_group_layouts: &[
                Some(scene_bind_group_layout),   // group 0
                Some(&io_bind_group_layout),     // group 1
                Some(params_bind_group_layout),  // group 2
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_shadow_trace"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (output_texture, output_view) = create_output_texture(device, half_w, half_h);

        Self {
            pipeline,
            pipeline_layout,
            user_shader_source_hash: 0,
            io_bind_group_layout,
            io_bind_group: None,
            shader_params_buffer: None,
            output_texture,
            output_view,
            half_w, half_h,
        }
    }

    /// Phase 4 — record the per-material `shader_params` buffer.
    /// The actual binding is on the shared params bind group built
    /// by `OctreeMarchPass`; this is symmetry for when the engine
    /// calls `set_shader_params` on every pipeline that consumes the
    /// buffer.
    pub fn set_shader_params(
        &mut self,
        _device: &wgpu::Device,
        shader_params_buffer: &wgpu::Buffer,
    ) {
        self.shader_params_buffer = Some(shader_params_buffer.clone());
    }

    /// Rebuild the compute pipeline against spliced user-shader chunks.
    /// Mirrors `OctreeMarchPass::reload_user_shaders`.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        instance_at_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.user_shader_source_hash {
            return false;
        }
        let template = wesl::include_wesl!("rkp_shadow_trace");
        let source = crate::shader_composer::splice_inst_chunks(
            template, instance_at_chunk,
        );
        let module = compile_pass_shader(device, &source, "rkp_shadow_trace");
        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rkp_shadow_trace"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.user_shader_source_hash = source_hash;
        true
    }

    pub fn resize(&mut self, device: &wgpu::Device, full_w: u32, full_h: u32) {
        let (half_w, half_h) = half_res_dims(full_w, full_h);
        if half_w == self.half_w && half_h == self.half_h { return; }
        let (t, v) = create_output_texture(device, half_w, half_h);
        self.output_texture = t;
        self.output_view = v;
        self.half_w = half_w;
        self.half_h = half_h;
        self.io_bind_group = None;
    }

    /// Rebuild the I/O bind group from current G-buffer views + own output.
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        gbuf_position: &wgpu::TextureView,
        gbuf_normal: &wgpu::TextureView,
    ) {
        self.io_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_trace io bind group"),
            layout: &self.io_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gbuf_position) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuf_normal) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.output_view) },
            ],
        }));
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
    ) {
        let Some(ref io_bg) = self.io_bind_group else { return; };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_shadow_trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.set_bind_group(1, io_bg, &[]);
        pass.set_bind_group(2, params_bind_group, &[]);
        let gx = self.half_w.div_ceil(8);
        let gy = self.half_h.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

fn half_res_dims(full_w: u32, full_h: u32) -> (u32, u32) {
    ((full_w + 1) / 2, (full_h + 1) / 2)
}

fn create_output_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rkp_shadow_lo_res"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rkp_shadow_trace_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("rkp_shadow_trace");
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("parse error:\n{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}
