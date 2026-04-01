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

use rkf_render::gbuffer::GBuffer;
use rkf_render::gpu_scene::GpuScene;
use rkf_render::tile_object_cull::TileObjectCullPass;

/// Splat march compute pass — fixed-step march through opacity field.
pub struct SplatMarchPass {
    pipeline: wgpu::ComputePipeline,
    pipeline_layout: wgpu::PipelineLayout,
}

impl SplatMarchPass {
    /// Raw WGSL source for the splat march shader.
    pub const SOURCE: &'static str = include_str!("shaders/splat_march.wgsl");

    /// Create the splat march pass.
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuScene,
        gbuffer: &GBuffer,
        tile_cull: &TileObjectCullPass,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Self::SOURCE.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat_march_pipeline_layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,           // group 0
                &gbuffer.write_bind_group_layout,   // group 1
                &tile_cull.read_bind_group_layout,  // group 2
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

        Self { pipeline, pipeline_layout }
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

    fn transform_brick(&self, brick: &mut rkf_core::brick::Brick, voxel_size: f32) {
        use half::f16;

        for voxel in brick.voxels.iter_mut() {
            // Extract f16 distance from word0 bits 0-15
            let dist_bits = (voxel.word0 & 0xFFFF) as u16;
            let distance = f16::from_bits(dist_bits).to_f32();

            // Convert: opacity = clamp(1.0 - distance / voxel_size, 0.0, 1.0)
            let opacity = (1.0 - distance / voxel_size).clamp(0.0, 1.0);

            // Write opacity back to word0 bits 0-15, preserving bits 16-31
            let opacity_bits = f16::from_f32(opacity).to_bits() as u32;
            voxel.word0 = (voxel.word0 & 0xFFFF_0000) | opacity_bits;
        }
    }
}
