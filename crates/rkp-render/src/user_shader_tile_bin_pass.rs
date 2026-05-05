//! GPU tile-binning pass for user-shader emitted instances.
//!
//! Per viewport: walks `user_shader_instance_aabbs`, projects each
//! instance's world AABB to screen space, and atomically appends the
//! instance index to the per-tile list of every overlapped 8×8-pixel
//! tile. The march's user-shader scan only iterates the current
//! pixel's tile list, dropping per-pixel cost from
//! `O(N_total_emitted)` to `O(N_in_my_tile)`.
//!
//! Per-viewport because tile count depends on screen resolution. The
//! pass is dispatched once per viewport per frame, between
//! `tick_emit_pass` and the march dispatch.
//!
//! Capacity:
//!   - `MAX_INSTANCES_PER_TILE = 1024` matches the WGSL constant.
//!   - `tile_lists` size = `num_tiles × 256 × 4 bytes`. At 1920×1080
//!     (240×135 = 32400 tiles) that's 32 MB.
//!   - Overflow drops silently — visible artifact: blades disappear
//!     in dense-paint regions. Bump the cap or build a variable-
//!     length linked-list version when the V1 cap proves a problem.
//!
//! Reset semantics: `tile_counts` must be cleared to 0 each frame
//! (engine handles via `encoder.clear_buffer`).

use crate::compile_pass_shader;

/// Mirror of the WGSL constant. Tile-list buffer sizing depends on
/// it; engine multiplies tile count by this when sizing the buffer.
pub const MAX_INSTANCES_PER_TILE: u32 = 1024;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BinParams {
    /// CPU-side upper bound (`painted_leaves × MAX_EMITS_PER_LEAF`).
    /// Threads past the GPU-side actual count (read from
    /// `instance_count_buffer`) early-return.
    pub instance_count_upper_bound: u32,
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    /// Threads per Y-stripe for the 2D-split dispatch — see
    /// `UserShaderEmitPass` for the same pattern. Lets us spread
    /// `instance_count > 65535 * 64` across X+Y dispatch dims.
    pub dispatch_x_threads: u32,
}

pub struct UserShaderTileBinPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
}

impl UserShaderTileBinPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_tile_bin bgl"),
            entries: &[
                bgl_storage_ro(0),  // instance_aabbs
                bgl_uniform(1),     // camera
                bgl_storage_rw(2),  // tile_counts (atomic)
                bgl_storage_rw(3),  // tile_lists
                bgl_uniform(4),     // bin_params
                bgl_storage_ro(5),  // instance_count (single u32)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_tile_bin layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let source = wesl::include_wesl!("user_shader_tile_bin");
        let module = compile_pass_shader(device, source, "user_shader_tile_bin");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("user_shader_tile_bin"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bin_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_tile_bin params"),
            size: std::mem::size_of::<BinParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group_layout, params_buffer }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn update_params(&self, queue: &wgpu::Queue, params: &BinParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn build_bind_group(
        &self,
        device: &wgpu::Device,
        instance_aabbs: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        tile_counts: &wgpu::Buffer,
        tile_lists: &wgpu::Buffer,
        instance_count: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_tile_bin bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: instance_aabbs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tile_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tile_lists.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: instance_count.as_entire_binding() },
            ],
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("user_shader_tile_bin"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        // Same X+Y split logic as `UserShaderEmitPass::dispatch`. The
        // shader rebuilds inst_idx as `gid.y * dispatch_x_threads + gid.x`,
        // where dispatch_x_threads is written into BinParams by the
        // caller (must match `dispatch_x` here).
        const MAX_DIM: u32 = 65535;
        let workgroups = instance_count.div_ceil(64);
        let x = workgroups.min(MAX_DIM);
        let y = workgroups.div_ceil(x);
        cpass.dispatch_workgroups(x, y, 1);
    }

    /// Compute the `dispatch_x_threads` value the shader needs, given
    /// the same `instance_count_upper_bound` the caller will pass to
    /// [`Self::dispatch`]. Caller writes this into [`BinParams`] before
    /// `update_params`.
    pub fn dispatch_x_threads_for(instance_count_upper_bound: u32) -> u32 {
        const MAX_DIM: u32 = 65535;
        let workgroups = instance_count_upper_bound.div_ceil(64);
        let x = workgroups.min(MAX_DIM);
        x * 64
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn bin_params_size_is_16_bytes() {
        assert_eq!(mem::size_of::<BinParams>(), 16);
    }
}
