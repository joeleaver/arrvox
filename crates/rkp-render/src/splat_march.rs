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

/// Compute axis-aligned half extents for an analytical primitive.
fn primitive_half_extents(prim: &rkf_core::scene_node::SdfPrimitive) -> glam::Vec3 {
    use rkf_core::scene_node::SdfPrimitive;
    match *prim {
        SdfPrimitive::Sphere { radius } => glam::Vec3::splat(radius),
        SdfPrimitive::Box { half_extents } => half_extents,
        SdfPrimitive::Capsule { radius, half_height } => {
            glam::Vec3::new(radius, half_height + radius, radius)
        }
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            let r = major_radius + minor_radius;
            glam::Vec3::new(r, minor_radius, r)
        }
        SdfPrimitive::Cylinder { radius, half_height } => {
            glam::Vec3::new(radius, half_height, radius)
        }
        SdfPrimitive::Plane { .. } => glam::Vec3::splat(1.0),
    }
}

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

    fn voxelize_primitive(
        &self,
        primitive: &rkf_core::scene_node::SdfPrimitive,
        material_id: u16,
        voxel_size: f32,
        bake_scale: glam::Vec3,
        pool: &mut rkf_core::brick_pool::BrickPool,
        map_alloc: &mut rkf_core::BrickMapAllocator,
    ) -> Option<(rkf_core::scene_node::BrickMapHandle, f32, rkf_core::Aabb, u32)> {
        use rkf_core::scene_node::SdfPrimitive;

        let half_extents = primitive_half_extents(primitive) * bake_scale;
        let margin = voxel_size * 2.0;
        let aabb = rkf_core::Aabb::new(
            -half_extents - glam::Vec3::splat(margin),
            half_extents + glam::Vec3::splat(margin),
        );

        // Build an opacity function from the analytical SDF primitive.
        // For primitives that support exact scaling (Box), evaluate the scaled
        // primitive directly. For others, use the inv_scale approximation.
        let opacity_fn: Box<dyn Fn(glam::Vec3) -> (f32, u16)> = match primitive {
            SdfPrimitive::Box { half_extents: he } => {
                let scaled = SdfPrimitive::Box { half_extents: *he * bake_scale };
                Box::new(move |pos: glam::Vec3| {
                    let d = rkf_core::evaluate_primitive(&scaled, pos);
                    let opacity = (0.5 - d / voxel_size).clamp(0.0, 1.0);
                    (opacity, material_id)
                })
            }
            _ => {
                let prim = primitive.clone();
                let min_scale = bake_scale.x.min(bake_scale.y).min(bake_scale.z).max(1e-6);
                let inv_scale = glam::Vec3::new(
                    1.0 / bake_scale.x.max(1e-6),
                    1.0 / bake_scale.y.max(1e-6),
                    1.0 / bake_scale.z.max(1e-6),
                );
                Box::new(move |pos: glam::Vec3| {
                    let d = rkf_core::evaluate_primitive(&prim, pos * inv_scale) * min_scale;
                    let opacity = (0.5 - d / voxel_size).clamp(0.0, 1.0);
                    (opacity, material_id)
                })
            }
        };

        let (handle, brick_count) =
            rkp_core::voxelize_opacity::voxelize_opacity(opacity_fn, &aabb, voxel_size, pool, map_alloc)?;

        // Use the geometry AABB (half_extents + margin) for culling, not the
        // full grid AABB. The shader's march volume (-half_grid to half_grid)
        // may be larger due to brick quantization, but the extra region is
        // empty — no need to include it in the BVH/wireframe AABB.
        let geometry_aabb = aabb;

        Some((handle, voxel_size, geometry_aabb, brick_count))
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

    fn march_source(&self) -> Option<&str> {
        Some(Self::SOURCE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::brick_map::BrickMapAllocator;
    use rkf_core::brick_pool::BrickPool;
    use rkf_core::scene_node::SdfPrimitive;

    #[test]
    fn voxelize_sphere_produces_opacity() {
        let mut pool = BrickPool::new(256);
        let mut alloc = BrickMapAllocator::new();

        let prim = SdfPrimitive::Sphere { radius: 0.5 };
        let voxel_size = 0.05;
        let scale = glam::Vec3::ONE;

        let half_ext = primitive_half_extents(&prim) * scale;
        let margin = voxel_size * 2.0;
        let aabb = rkf_core::Aabb::new(
            -half_ext - glam::Vec3::splat(margin),
            half_ext + glam::Vec3::splat(margin),
        );

        let prim_clone = prim.clone();
        let opacity_fn = move |pos: glam::Vec3| -> (f32, u16) {
            let d = rkf_core::evaluate_primitive(&prim_clone, pos);
            let opacity = (0.5 - d / voxel_size).clamp(0.0, 1.0);
            (opacity, 1)
        };

        let result = rkp_core::voxelize_opacity::voxelize_opacity(
            opacity_fn, &aabb, voxel_size, &mut pool, &mut alloc,
        );
        assert!(result.is_some(), "voxelization should succeed");

        let (handle, brick_count) = result.unwrap();
        assert!(brick_count > 0, "should allocate bricks for a sphere");

        // Verify all stored voxels have opacity in [0, 1] — no SDF distances.
        for slot_idx in 0..brick_count {
            let brick = pool.get(slot_idx);
            for v in &brick.voxels {
                let bits = (v.word0 & 0xFFFF) as u16;
                let o = half::f16::from_bits(bits).to_f32();
                assert!(o >= 0.0 && o <= 1.0, "opacity should be in [0,1], got {o}");
            }
        }
    }

    #[test]
    fn voxelize_box_with_scale() {
        let mut pool = BrickPool::new(256);
        let mut alloc = BrickMapAllocator::new();

        let prim = SdfPrimitive::Box { half_extents: glam::Vec3::splat(0.3) };
        let voxel_size = 0.05;
        let scale = glam::Vec3::new(2.0, 1.0, 1.0);

        let half_ext = primitive_half_extents(&prim) * scale;
        let margin = voxel_size * 2.0;
        let aabb = rkf_core::Aabb::new(
            -half_ext - glam::Vec3::splat(margin),
            half_ext + glam::Vec3::splat(margin),
        );

        let prim_clone = prim.clone();
        let min_scale = scale.x.min(scale.y).min(scale.z).max(1e-6);
        let inv_scale = glam::Vec3::new(
            1.0 / scale.x.max(1e-6),
            1.0 / scale.y.max(1e-6),
            1.0 / scale.z.max(1e-6),
        );
        let opacity_fn = move |pos: glam::Vec3| -> (f32, u16) {
            let d = rkf_core::evaluate_primitive(&prim_clone, pos * inv_scale) * min_scale;
            let opacity = (0.5 - d / voxel_size).clamp(0.0, 1.0);
            (opacity, 5)
        };

        let result = rkp_core::voxelize_opacity::voxelize_opacity(
            opacity_fn, &aabb, voxel_size, &mut pool, &mut alloc,
        );
        assert!(result.is_some());

        let (handle, _) = result.unwrap();
        // Scaled box: X extent is 2× larger. Grid should be wider on X.
        assert!(
            handle.dims.x > handle.dims.y,
            "X dims ({}) should be larger than Y dims ({}) due to scale",
            handle.dims.x, handle.dims.y,
        );
    }

    #[test]
    fn aabb_is_tight_around_geometry() {
        let prim = SdfPrimitive::Box { half_extents: glam::Vec3::splat(0.5) };
        let voxel_size = 0.05;
        let scale = glam::Vec3::new(4.0, 0.1, 4.0);

        let half_ext = primitive_half_extents(&prim) * scale;
        let margin = voxel_size * 2.0;

        // The returned AABB should be the geometry extent + margin,
        // NOT the full grid extent (which includes brick quantization padding).
        let expected_half = half_ext + glam::Vec3::splat(margin);

        // The geometry Y half is 0.05. With margin 0.1, expected Y half = 0.15.
        // A full grid AABB would round up to whole bricks (e.g., 0.2 or 0.4).
        assert!(
            expected_half.y < 0.2,
            "tight AABB Y should be small, got {}",
            expected_half.y
        );
        assert!(
            expected_half.y > half_ext.y,
            "tight AABB should include margin"
        );
    }
}
