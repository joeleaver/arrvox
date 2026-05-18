//! Brush-state probe pass — single-thread compute that reads
//! `gbuf_position` + `gbuf_pick` at the cursor pixel and writes a
//! 32-byte [`BrushState`] buffer. Consumed by `rkp_shade` to draw the
//! screen-space paint cursor.
//!
//! Replaces the earlier sim-thread geodesic flood fill that ran on
//! every cursor move and round-tripped a pick readback. With this
//! pass the cursor follows the pointer at frame rate — zero CPU per
//! hover, zero round-trip — because the gbuffer has already been
//! populated by the primary visibility pass(es) by the time we read
//! it on the same frame.
//!
//! See `shaders/brush_state.wesl` for the (very small) shader.
//!
//! ## Dispatch contract
//!
//! - One dispatch per VR per frame, after the primary raster +
//!   `mesh_resolve` (so `gbuf_position` / `gbuf_pick` are
//!   populated) and before `rkp_shade`.
//! - `params` carries the cursor pixel in viewport coords + an
//!   `active` flag. When `active = 0`, the shader writes the miss
//!   sentinel (`hit_distance = 1e10, hit_object_id = u32::MAX`),
//!   so the shade pass hides the cursor without any CPU
//!   coordination — useful when paint mode is off, the mouse has
//!   left the viewport, or the click was outside the framebuffer.

use crate::rkp_shade::BrushState;

/// 16-byte uniform driving [`BrushStatePass::dispatch`]. Mirrors the
/// WGSL `BrushParams` struct in `shaders/brush_state.wesl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushParams {
    pub cursor_x: u32,
    pub cursor_y: u32,
    pub enabled: u32,
    pub _pad0: u32,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self { cursor_x: 0, cursor_y: 0, enabled: 0, _pad0: 0 }
    }
}

const _: () = assert!(std::mem::size_of::<BrushParams>() == 16);

pub struct BrushStatePass {
    pub pipeline: wgpu::ComputePipeline,
    pub bg_layout: wgpu::BindGroupLayout,
}

impl BrushStatePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush_state bg"),
            entries: &[
                // gbuf_position (Rgba32Float, sampled)
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
                // gbuf_pick (R32Uint, sampled)
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
                // brush_state (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush_state pipeline layout"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size: 0,
        });
        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("brush_state"),
            "brush_state",
        );
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brush_state"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bg_layout }
    }

    /// Allocate a `BrushState` storage buffer pre-initialised to the
    /// miss sentinel — safe to read before the first dispatch.
    pub fn create_state_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let initial = BrushState::default();
        wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("brush_state buffer"),
                contents: bytemuck::bytes_of(&initial),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )
    }

    /// Allocate the `BrushParams` uniform buffer.
    pub fn create_params_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let initial = BrushParams::default();
        wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("brush_state params"),
                contents: bytemuck::bytes_of(&initial),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        gbuf_position_view: &wgpu::TextureView,
        gbuf_pick_view: &wgpu::TextureView,
        state_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush_state bg"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gbuf_position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuf_pick_view) },
                wgpu::BindGroupEntry { binding: 2, resource: state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brush_state"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
}
