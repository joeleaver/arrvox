//! BFS dispatcher pipeline / buffer construction.
//!
//! Pulled out of `dispatch.rs` so the per-frame work
//! (`dispatch_regions`) doesn't share a file with the one-time
//! constructor and the WGSL splice helpers. Two `impl UserShaderPass`
//! blocks across the two files; the per-frame impl is in
//! `dispatch.rs`.

use crate::compile_pass_shader;

use super::cache::MAX_GLOBAL_FILL_TASKS;
use super::dispatch::{
    ActiveCell, UserShaderPass, LEVEL_UNIFORM_STRIDE, MAX_DEPTH, MAX_REGIONS,
    PER_LEVEL_QUEUE_CAP,
};
use super::overflow::{OverflowReadback, OVERFLOW_BUFFER_BYTES};
use super::region::RegionUniform;

/// Compose the geom-build WGSL with a user `generate` chunk. Empty
/// chunk leaves the in-tree identity stub in place.
pub fn compose_geom_source(user_chunk: &str) -> String {
    let geom_src = wesl::include_wesl!("user_shader_geom");
    crate::shader_composer::splice_const_marker(
        geom_src,
        "USER_GENERATE_DISPATCH",
        user_chunk,
    )
}

pub(super) fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn build_pipelines(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    user_chunk: &str,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let source = compose_geom_source(user_chunk);
    let module = compile_pass_shader(device, &source, "user_shader_geom");
    let classify = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom classify"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("classify_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let fill = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_geom fill"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("brick_fill_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (classify, fill)
}

impl UserShaderPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Catch the (unlikely) case of a backend that demands greater
        // uniform-buffer-offset alignment than our compile-time stride
        // can satisfy. Today every wgpu backend caps the requirement
        // at 256.
        assert!(
            (device.limits().min_uniform_buffer_offset_alignment as u64)
                <= LEVEL_UNIFORM_STRIDE,
            "device requires min_uniform_buffer_offset_alignment of {} \
             which exceeds the LEVEL_UNIFORM_STRIDE constant of {}",
            device.limits().min_uniform_buffer_offset_alignment,
            LEVEL_UNIFORM_STRIDE,
        );
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group0"),
            entries: &[
                rw_storage(0),  // octree_nodes
                rw_storage(1),  // brick_pool
                rw_storage(2),  // leaf_attr_pool
                rw_storage(3),  // octree_alloc (per-region atomic array)
                rw_storage(4),  // brick_alloc  (per-region atomic array)
                rw_storage(5),  // leaf_attr_alloc (per-region atomic array)
                rw_storage(6),  // active_queue
                rw_storage(7),  // active_count
                rw_storage(8),  // fill_task_pool
                rw_storage(9),  // fill_task_alloc (per-region atomic array)
                rw_storage(10), // overflow counters
                ro_storage(11), // instance_overlay (per-instance paint)
            ],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<RegionUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_geom group2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<super::dispatch::LevelUniform>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_geom pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
            ],
            immediate_size: 0,
        });
        let (classify_pipeline, fill_pipeline) = build_pipelines(device, &pipeline_layout, "");

        // Per-region atomic counters: array<atomic<u32>, MAX_REGIONS>.
        // Sized at MAX_REGIONS u32s. Reset per frame for dirty regions.
        let alloc_buf_size = (MAX_REGIONS as u64) * 4;
        let make_alloc_buf = |label| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: alloc_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let octree_alloc_buffer = make_alloc_buf("user_shader_geom octree_alloc");
        let brick_alloc_buffer = make_alloc_buf("user_shader_geom brick_alloc");
        let leaf_attr_alloc_buffer = make_alloc_buf("user_shader_geom leaf_attr_alloc");
        let fill_task_alloc_buffer = make_alloc_buf("user_shader_geom fill_task_alloc");

        let overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom overflow"),
            size: OVERFLOW_BUFFER_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let overflow_readback = OverflowReadback::new(device);

        let queue_size_bytes =
            (MAX_DEPTH + 1) as u64 * PER_LEVEL_QUEUE_CAP as u64 * std::mem::size_of::<ActiveCell>() as u64;
        let active_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_queue"),
            size: queue_size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom active_count"),
            size: (MAX_DEPTH + 1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Persistent fill-task pool. ~48 MB at MAX_GLOBAL_FILL_TASKS.
        let fill_task_pool_size = MAX_GLOBAL_FILL_TASKS as u64 * 32;
        let fill_task_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom fill_task_pool"),
            size: fill_task_pool_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let regions_capacity: u64 =
            std::mem::size_of::<RegionUniform>() as u64 * MAX_REGIONS as u64;
        let regions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom regions"),
            size: regions_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let level_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_geom level_uniforms"),
            size: LEVEL_UNIFORM_STRIDE * (MAX_DEPTH + 1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            pipeline_layout,
            classify_pipeline,
            fill_pipeline,
            octree_alloc_buffer,
            brick_alloc_buffer,
            leaf_attr_alloc_buffer,
            fill_task_alloc_buffer,
            active_queue_buffer,
            active_count_buffer,
            fill_task_pool_buffer,
            overflow_buffer,
            overflow_readback,
            regions_buffer,
            level_uniforms_buffer,
            source_hash: 0,
            group0_bind_group: None,
            group0_buffers_epoch: 0,
        }
    }

    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        user_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        let (classify, fill) = build_pipelines(device, &self.pipeline_layout, user_chunk);
        self.classify_pipeline = classify;
        self.fill_pipeline = fill;
        self.source_hash = source_hash;
        true
    }
}
