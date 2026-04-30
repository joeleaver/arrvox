//! Phase 6 Session 3b — user-shader tile-cull prefix sum pass.
//!
//! Single-workgroup blocked scan that turns `us_tile_counts` into the
//! tile-offset prefix sum:
//!
//! ```text
//! us_tile_offsets[0]   = 0
//! us_tile_offsets[t+1] = Σ counts[0..=t]
//! us_tile_offsets[T]   = total entry count (= length of us_tile_entries)
//! ```
//!
//! ## V1 cap
//!
//! Supports up to [`PREFIX_MAX_TILES`] = 65536 tiles per dispatch,
//! which covers ~2300×800 px at 8 px tiles. Above this — e.g. 4K at
//! 8 px tiles is ~129K tiles — the engine should clamp by viewport
//! width or run a multi-block scan (TODO if 4K becomes a target).
//!
//! ## Bindings
//!
//! * group(0): `us_tile_counts` (read), `us_tile_offsets` (rw).
//! * group(1): `PrefixUniform { tile_count }` (uniform, 16 B).

use crate::validate_wgsl;

/// Maximum tiles supportable by one prefix-sum dispatch. See module docs.
pub const PREFIX_MAX_TILES: u32 = 65536;
const _: () = assert!(PREFIX_MAX_TILES == 256 * 256, "wgsl uses PREFIX_BLOCK×PREFIX_THREADS=256×256");

/// Uniform for the prefix-sum dispatch. 16 bytes (mat-aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrefixUniform {
    pub tile_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<PrefixUniform>() == 16);

/// GPU pipeline owner for the prefix-sum compute shader.
pub struct TilePrefixPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl TilePrefixPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_prefix group0"),
            entries: &[
                ro_storage(0), // us_tile_counts
                rw_storage(1), // us_tile_offsets
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_prefix group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<PrefixUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_tile_prefix pipeline layout"),
            bind_group_layouts: &[Some(&group0_layout), Some(&group1_layout)],
            immediate_size: 0,
        });
        let pipeline = build_pipeline(device, &pipeline_layout);

        Self {
            group0_layout,
            group1_layout,
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

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let source = include_str!("shaders/user_shader_tile_prefix.wgsl");
    validate_wgsl(source, "user_shader_tile_prefix");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_tile_prefix"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_tile_prefix"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("tile_prefix_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_uniform_size_is_16() {
        assert_eq!(std::mem::size_of::<PrefixUniform>(), 16);
    }

    #[test]
    fn prefix_max_tiles_covers_1080p() {
        // 1920×1080 at 8 px tiles = 240×135 = 32400; well under cap.
        assert!(PREFIX_MAX_TILES >= 240 * 135);
    }

    #[test]
    fn template_validates_with_naga() {
        let source = include_str!("shaders/user_shader_tile_prefix.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("[user_shader_tile_prefix] parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("[user_shader_tile_prefix] validation error: {e:?}"));
        assert!(source.contains("tile_prefix_main"));
    }
}
