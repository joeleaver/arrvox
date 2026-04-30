//! Phase 6 Session 3a — user-shader tile-cull count pass.
//!
//! Per `InstanceTileCullEntry` produced by Session 2's AABB pass,
//! projects the world AABB through the viewport's `view_proj` to a
//! tile rectangle and `atomicAdd`s a count into `us_tile_counts[tile]`
//! for each tile in that rectangle. The downstream prefix-sum + scatter
//! passes turn the per-tile counts into a flat
//! `us_tile_entries` array partitioned by tile.
//!
//! ## Bindings
//!
//! * group(0): `tile_cull_scratch` (read), `us_tile_counts` (atomic rw).
//! * group(1): `TileCullViewportUniform` (uniform, no dynamic offset —
//!   one viewport per dispatch is fine; the engine builds a fresh bind
//!   group per VR per frame).
//!
//! ## Dispatch
//!
//! 1-D, `@workgroup_size(64, 1, 1)`, one thread per scratch entry.
//! `workgroups = scratch_count.div_ceil(64).max(1)`.

use crate::validate_wgsl;

/// Per-VP uniform — must match `TileCullViewportUniform` in
/// `user_shader_tile_count.wgsl` (and the matching scatter shader).
/// 96 bytes; the struct align is 16 (mat4x4 alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileCullViewportUniform {
    pub view_proj: [[f32; 4]; 4],
    pub resolution_x: f32,
    pub resolution_y: f32,
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub tile_count: u32,
    /// Number of valid entries in `tile_cull_scratch` for this dispatch
    /// (== Σ region.instance_block_size across all regions). Threads
    /// past this count early-return.
    pub scratch_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<TileCullViewportUniform>() == 96);

/// Workgroup count for the tile-count dispatch. Workgroup size = 64,
/// one thread per scratch entry.
pub fn workgroups_for_scratch(scratch_count: u32) -> u32 {
    scratch_count.div_ceil(64).max(1)
}

/// Pixel size of one tile. Must match `TILE_PX` in the WGSL and the
/// host march's `(width + 7) / 8` tile-grid math.
pub const TILE_PX: u32 = 8;

/// Number of tiles required to cover an `(width × height)` viewport.
pub fn tile_count_for_viewport(width: u32, height: u32) -> (u32, u32, u32) {
    let tx = width.div_ceil(TILE_PX);
    let ty = height.div_ceil(TILE_PX);
    (tx, ty, tx * ty)
}

/// GPU pipeline owner for the count compute shader.
pub struct TileCountPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl TileCountPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_count group0"),
            entries: &[
                ro_storage(0), // tile_cull_scratch
                rw_storage(1), // us_tile_counts (atomic)
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_count group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<TileCullViewportUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_tile_count pipeline layout"),
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
    let source = include_str!("shaders/user_shader_tile_count.wgsl");
    validate_wgsl(source, "user_shader_tile_count");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_tile_count"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_tile_count"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("tile_count_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_uniform_size_is_96() {
        assert_eq!(std::mem::size_of::<TileCullViewportUniform>(), 96);
    }

    #[test]
    fn workgroups_for_scratch_at_least_one() {
        assert_eq!(workgroups_for_scratch(0), 1);
        assert_eq!(workgroups_for_scratch(1), 1);
        assert_eq!(workgroups_for_scratch(64), 1);
        assert_eq!(workgroups_for_scratch(65), 2);
        assert_eq!(workgroups_for_scratch(640), 10);
    }

    #[test]
    fn tile_count_matches_host_march_math() {
        // Host march dispatches `(w+7)/8 × (h+7)/8` workgroups; the tile
        // grid here uses the same formula.
        assert_eq!(tile_count_for_viewport(1920, 1080), (240, 135, 240 * 135));
        assert_eq!(tile_count_for_viewport(8, 8), (1, 1, 1));
        assert_eq!(tile_count_for_viewport(9, 9), (2, 2, 4));
    }

    #[test]
    fn template_validates_with_naga() {
        let source = include_str!("shaders/user_shader_tile_count.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("[user_shader_tile_count] parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("[user_shader_tile_count] validation error: {e:?}"));
        assert!(source.contains("tile_count_main"));
    }
}
