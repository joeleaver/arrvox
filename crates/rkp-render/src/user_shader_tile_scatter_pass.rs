//! Phase 6 Session 3c — user-shader tile-cull scatter pass.
//!
//! Per `InstanceTileCullEntry`, projects the world AABB to its screen
//! tile rectangle and writes a 16-byte
//! [`crate::octree_march::UserShaderTileEntry`] into `us_tile_entries`
//! for each covered tile. Slot allocation uses an atomic cursor per
//! tile, initialized by the engine to a copy of `us_tile_offsets[..]`
//! (so atomicAdd returns slots in the [offset[t], offset[t+1]) range).
//!
//! `us_tile_offsets` itself is left untouched — the host march reads
//! from it to discover each tile's slice.
//!
//! ## Bindings
//!
//! * group(0) binding(0): `tile_cull_scratch`         (read)
//! * group(0) binding(1): `us_tile_scatter_cursor`    (atomic rw)
//! * group(0) binding(2): `us_tile_entries`           (rw)
//! * group(1) binding(0): `TileCullViewportUniform`   (uniform — same
//!   layout as the count pass)

use crate::user_shader_tile_count_pass::TileCullViewportUniform;
use crate::validate_wgsl;

/// GPU pipeline owner for the scatter compute shader.
pub struct TileScatterPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl TileScatterPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_scatter group0"),
            entries: &[
                ro_storage(0), // tile_cull_scratch
                rw_storage(1), // us_tile_scatter_cursor (atomic)
                rw_storage(2), // us_tile_entries
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_scatter group1"),
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
            label: Some("user_shader_tile_scatter pipeline layout"),
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
    let source = include_str!("shaders/user_shader_tile_scatter.wgsl");
    validate_wgsl(source, "user_shader_tile_scatter");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("user_shader_tile_scatter"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_tile_scatter"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("tile_scatter_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn template_validates_with_naga() {
        let source = include_str!("shaders/user_shader_tile_scatter.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("[user_shader_tile_scatter] parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("[user_shader_tile_scatter] validation error: {e:?}"));
        assert!(source.contains("tile_scatter_main"));
    }

    #[test]
    fn scatter_uses_same_uniform_as_count() {
        // Ensure the shared `TileCullViewportUniform` Rust struct has
        // the same size as the WGSL struct here. (Sanity — both
        // pipelines bind the same buffer.)
        use crate::user_shader_tile_count_pass::TileCullViewportUniform;
        assert_eq!(std::mem::size_of::<TileCullViewportUniform>(), 96);
    }
}
