//! Stage 5b — instance-march compute pipeline.
//!
//! The march that consumes everything Stages 1-5a produced:
//!
//!   * **Group 0** — pool buffers (`octree_nodes`, `brick_pool`,
//!     `leaf_attr_pool`). Same layout as the Stage 5a test pipeline so
//!     the shared helpers in
//!     `shaders/user_shader_instance_march_helpers.wgsl` work against
//!     either one.
//!   * **Group 1** — per-frame instance state: `regions_buffer`,
//!     `instance_pool`, `tile_index_buffer`, `instance_alloc`. The first
//!     three are produced by Stage 4 (`InstanceRegionCache` +
//!     `flatten_tile_index`); `instance_alloc` is the per-region atomic
//!     counter the emit pass writes.
//!   * **Group 2** — `proto_lookup_buffer` from
//!     [`crate::instance_proto_lookup::flatten_prototype_lookup`].
//!   * **Group 3** — march uniforms, ray buffer, output hit buffer.
//!     Output is `array<InstanceMarchHit>`; one slot per ray.
//!
//! V1 is single-ray-per-dispatch — `dispatch_workgroups(num_rays, 1, 1)`
//! with workgroup_size(1). Stage 6 will batch by screen tile.
//!
//! ## Source composition
//!
//! WGSL source is the concatenation of
//! [`crate::user_shader_instance_march::instance_march_helpers_source`]
//! and `shaders/user_shader_instance_march_main.wgsl`. The helpers
//! file declares the @group(0) pool bindings + helper fns; the main
//! file declares the @group(1/2/3) bindings + the `@compute` entry.

use std::num::NonZeroU64;

use crate::instance_proto_lookup::GpuPrototypeEntry;
use crate::instance_tile_index_gpu::GpuTileIndexEntry;
use crate::user_shader_emit_pass::EmitRegionUniform;
use crate::user_shader_instance_march::instance_march_helpers_source;

/// Per-frame uniform — must match `MarchUniforms` in
/// `user_shader_instance_march_main.wgsl`. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchUniforms {
    pub tile_index_count: u32,
    pub proto_lookup_count: u32,
    pub ray_count: u32,
    pub _pad0: u32,
}

const _: () = assert!(std::mem::size_of::<MarchUniforms>() == 16);

/// Per-ray input. Mirror of `MarchRay` in the WGSL. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchRay {
    pub origin: [f32; 3],
    pub instance_count_per_region: u32,
    pub direction: [f32; 3],
    pub max_steps_outer: u32,
    pub max_steps_brick: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<MarchRay>() == 48);

/// Per-ray output — the closest instance hit found for this ray. Mirror
/// of `InstanceMarchHit` in the WGSL. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceMarchHit {
    pub hit: u32,
    pub region_index: u32,
    pub instance_index: u32,
    pub leaf_attr_slot: u32,
    pub t_world: f32,
    pub material_packed: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}

const _: () = assert!(std::mem::size_of::<InstanceMarchHit>() == 48);

/// Source-text composition: helpers + main entry. Exposed so tests can
/// validate the WGSL with naga without going through pipeline creation.
pub fn instance_march_main_source() -> String {
    format!(
        "{}\n{}",
        instance_march_helpers_source(),
        include_str!("shaders/user_shader_instance_march_main.wgsl"),
    )
}

/// Pipeline owner. Construction validates the composed WGSL with naga
/// (panics on any regression) and creates four bind-group layouts +
/// the compute pipeline.
pub struct InstanceMarchPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub group2_layout: wgpu::BindGroupLayout,
    pub group3_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl InstanceMarchPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group0 (pools)"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2)],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group1 (per-frame instance state)"),
            entries: &[
                // regions_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<EmitRegionUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                // instance_pool
                ro_storage(1),
                // tile_index_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuTileIndexEntry>() as u64,
                        ),
                    },
                    count: None,
                },
                // instance_alloc (read-only here — written by emit pass)
                ro_storage(3),
            ],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group2 (proto lookup)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        std::mem::size_of::<GpuPrototypeEntry>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group3_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group3 (uniforms + IO)"),
            entries: &[
                // march_uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<MarchUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // rays_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<MarchRay>() as u64,
                        ),
                    },
                    count: None,
                },
                // output_hits
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<InstanceMarchHit>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instance_march pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
                Some(&group3_layout),
            ],
            immediate_size: 0,
        });

        let source = instance_march_main_source();
        crate::validate_wgsl(&source, "instance_march_main");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance_march_main"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_march_main"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("instance_march_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            group3_layout,
            pipeline_layout,
            pipeline,
        }
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn march_uniforms_layout() {
        let u = MarchUniforms {
            tile_index_count: 1,
            proto_lookup_count: 2,
            ray_count: 3,
            _pad0: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&u);
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(words, &[1, 2, 3, 0]);
    }

    #[test]
    fn march_ray_layout_origin_and_dir_match_offsets() {
        let r = MarchRay {
            origin: [1.0, 2.0, 3.0],
            instance_count_per_region: 7,
            direction: [4.0, 5.0, 6.0],
            max_steps_outer: 256,
            max_steps_brick: 64,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(floats[0..3], [1.0, 2.0, 3.0]);
        // Slot 3 is `instance_count_per_region` (a u32) — bitcast 7u → tiny float.
        assert_eq!(floats[4..7], [4.0, 5.0, 6.0]);
    }

    #[test]
    fn instance_march_hit_layout_normal_at_offset_32() {
        // Verify the InstanceMarchHit struct lays out so `normal` lands
        // at byte offset 32 (matching the WGSL struct's vec3<f32>
        // alignment-driven slot). Catches future drift.
        let hit = InstanceMarchHit {
            hit: 1,
            region_index: 2,
            instance_index: 3,
            leaf_attr_slot: 4,
            t_world: 5.0,
            material_packed: 6,
            _pad0: 0,
            _pad1: 0,
            normal: [7.0, 8.0, 9.0],
            _pad2: 0.0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&hit);
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(words[0..6], [1, 2, 3, 4, f32::to_bits(5.0), 6]);
        let normal_floats: [f32; 3] = [
            f32::from_bits(words[8]),
            f32::from_bits(words[9]),
            f32::from_bits(words[10]),
        ];
        assert_eq!(normal_floats, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn march_main_wgsl_validates() {
        let source = instance_march_main_source();
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            panic!("parse error:\n{}", e.emit_to_string(&source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}
