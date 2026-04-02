//! Splat march compute pass — surface-finding through trilinear opacity field.
//!
//! [`SplatMarchPass`] is the splat engine's replacement for rkf-render's
//! [`RayMarchPass`]. It marches through the opacity field with fixed steps,
//! finds the surface where opacity crosses a threshold, computes the gradient
//! normal, and writes to the same G-buffer format.
//!
//! # Bind Groups
//!
//! | Group | Content |
//! |-------|---------|
//! | 0 | GpuScene (brick pool, brick maps, objects, camera, scene, BVH) |
//! | 1 | G-buffer write targets (position, normal, material, motion) |
//! | 2 | Per-tile object lists from [`TileObjectCullPass`] (indices + counts) |
//! | 3 | Materials + shader params (for opacity shader evaluation) |

use rkf_render::gbuffer::GBuffer;
use rkf_render::gpu_scene::GpuScene;
use rkf_render::shader_params::ShaderParamsBuffer;
use rkf_render::tile_object_cull::TileObjectCullPass;

/// Splat march compute pass — fixed-step march through opacity field.
pub struct SplatMarchPass {
    pipeline: wgpu::ComputePipeline,
    pipeline_layout: wgpu::PipelineLayout,
    /// Bind group layout for group 3 (materials + shader params).
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for group 3.
    material_bind_group: wgpu::BindGroup,
    /// Shader params buffer reference (for rebuilding bind group on material update).
    shader_params_buffer: wgpu::Buffer,
}

impl SplatMarchPass {
    /// Raw WGSL source for the splat march shader (with injection placeholders).
    pub const SOURCE: &'static str = include_str!("shaders/splat_march.wgsl");

    /// Create the splat march pass.
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuScene,
        gbuffer: &GBuffer,
        tile_cull: &TileObjectCullPass,
        material_buffer: &wgpu::Buffer,
        shader_params: &ShaderParamsBuffer,
        opacity_shader_code: &str,
    ) -> Self {
        // Inject opacity shader functions into the march source
        let source = Self::SOURCE.replace("// OPACITY_SHADER_FUNCTIONS", opacity_shader_code);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Group 3: materials + shader params
        let material_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat_march_material_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let material_bind_group = Self::create_material_bind_group(
            device, &material_bind_group_layout, material_buffer, &shader_params.buffer,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_march_pipeline_layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,           // group 0
                &gbuffer.write_bind_group_layout,   // group 1
                &tile_cull.read_bind_group_layout,  // group 2
                &material_bind_group_layout,        // group 3
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_march_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_layout,
            material_bind_group_layout,
            material_bind_group,
            shader_params_buffer: shader_params.buffer.clone(),
        }
    }

    fn create_material_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        material_buffer: &wgpu::Buffer,
        shader_params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("splat_march_materials"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shader_params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Recreate the compute pipeline with a new shader module (hot-reload).
    pub fn recreate_pipeline(&mut self, device: &wgpu::Device, module: &wgpu::ShaderModule) {
        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("splat_march_pipeline"),
            layout: Some(&self.pipeline_layout),
            module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    }

    /// Record the splat march dispatch into a command encoder.
    ///
    /// Dispatches one thread per pixel at internal resolution using 8x8 workgroups.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        gbuffer: &GBuffer,
        tile_cull: &TileObjectCullPass,
    ) {
        let workgroups_x = (gbuffer.width + 7) / 8;
        let workgroups_y = (gbuffer.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("splat_march"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &gbuffer.write_bind_group, &[]);
        pass.set_bind_group(2, &tile_cull.read_bind_group, &[]);
        pass.set_bind_group(3, &self.material_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

impl rkf_render::MarchPass for SplatMarchPass {
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, ctx: &rkf_render::MarchContext) {
        self.dispatch(encoder, ctx.scene, ctx.gbuffer, ctx.tile_cull);
    }

    fn recreate_pipeline(&mut self, device: &wgpu::Device, module: &wgpu::ShaderModule) {
        self.recreate_pipeline(device, module);
    }

    fn update_materials(&mut self, device: &wgpu::Device, material_buffer: &wgpu::Buffer) {
        self.material_bind_group = Self::create_material_bind_group(
            device, &self.material_bind_group_layout, material_buffer, &self.shader_params_buffer,
        );
    }

    // needs_skin_deform: default true — use the SkinDeformPass to scatter
    // bone weights into posed space. The march shader reads them and inverse-skins
    // back to rest-pose for opacity sampling.

    fn shader_overrides(&self) -> rkf_render::ShaderOverrides {
        rkf_render::ShaderOverrides {
            shadow_ao: Some(include_str!("shaders/opacity_shadow_ao.wgsl").to_string()),
            radiance_inject: Some(include_str!("shaders/opacity_radiance_inject.wgsl").to_string()),
            shade_common: Some(include_str!("shaders/opacity_shade_common.wgsl").to_string()),
            shade_common_shading: Some(include_str!("shaders/opacity_shade_common_shading.wgsl").to_string()),
            shade_main: Some(include_str!("shaders/opacity_shade_main.wgsl").to_string()),
            shade_models: vec![
                ("pbr".into(), include_str!("shaders/opacity_shade_pbr.wgsl").to_string()),
            ],
        }
    }

}
