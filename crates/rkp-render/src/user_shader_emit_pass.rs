//! User-shader instance emit pass.
//!
//! Per painted-leaf compute pass that runs each registered shader's
//! `instance_at` hook and writes one [`RkpGpuInstance`] per accepted
//! instance into the scene-global `user_shader_instance_buffer`. The
//! host march reads those instances through its standard flow — no
//! band-cell branch, no per-pixel descend body.
//!
//! ## Wire format
//!
//! - [`EmitLeaf`] — one per painted leaf cell whose material has an
//!   `instance_at` hook. CPU collects from the existing octree walk.
//! - [`MatToProto`] — per-material `(shader_id, proto_asset_id)`. CPU
//!   builds at frame start by walking the material library against
//!   the current proto-bake registration list.
//! - [`EmitParams`] — uniform: leaf count + instance capacity + time.
//!
//! ## Dispatch shape
//!
//! `workgroup_size(64, 1, 1)`, dispatch
//! `(leaves_count.div_ceil(64), 1, 1)`. One thread per leaf; the
//! per-thread loop runs k = 0..MAX_EMITS_PER_LEAF (= 8) calls into
//! `dispatch_user_emit(shader_id, ...)`. Per-shader `instance_at`
//! bodies short-circuit by returning `false` once their density-
//! driven count is exhausted.

use crate::compile_pass_shader;
use crate::shader_composer::splice_const_marker;

const COUNT_MAP_IDLE: u8 = 0;
const COUNT_MAP_PENDING: u8 = 1;
const COUNT_MAP_READY: u8 = 2;
const COUNT_MAP_FAILED: u8 = 3;

/// One painted leaf record. CPU packs one per (object × material ×
/// painted-leaf-cell) tuple where the material has an `instance_at`
/// hook. Layout mirrors `EmitLeaf` in `user_shader_emit.wgsl`.
///
/// 32 bytes — round multiple of 16, no trailing pad needed.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitLeaf {
    pub world_pos: [f32; 3],
    pub material_id: u32,
    pub normal_oct: u32,
    pub object_id: u32,
    pub leaf_slot: u32,
    pub cell_size: f32,
}

/// Per-material `(shader_id, proto_asset_id)` lookup. `shader_id == 0`
/// means "no instance shader" → emit-pass thread early-returns.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatToProto {
    pub shader_id: u32,
    pub proto_asset_id: u32,
}

/// Uniform parameters for the emit dispatch.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitParams {
    pub leaf_count: u32,
    pub instance_capacity: u32,
    pub time: f32,
    pub _pad0: u32,
}

/// Compute pass that consumes painted-leaf records and emits
/// `RkpGpuInstance`s into the scene's user-shader instance buffer.
pub struct UserShaderEmitPass {
    pipeline: wgpu::ComputePipeline,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Per-material lookup buffer. Resized on material count change.
    mat_to_proto_buffer: wgpu::Buffer,
    mat_to_proto_capacity: u64,
    /// Per-frame leaf list. Resized when needed.
    leaves_buffer: wgpu::Buffer,
    leaves_capacity: u64,
    /// Uniform with leaf_count / capacity / time.
    params_buffer: wgpu::Buffer,
    /// Hash of the user-shader source mix this pipeline was last
    /// built against. Comparing to the registry's `source_hash`
    /// decides whether a rebuild is needed.
    user_shader_source_hash: u64,
    /// Cached bind group, rebuilt when any input buffer reallocates.
    bind_group: Option<wgpu::BindGroup>,
    bind_group_inputs_epoch: u64,
    /// Single 4-byte staging buffer for `instance_count_buffer` readback.
    count_readback_buffer: wgpu::Buffer,
    /// Map state for the async count readback (skip-if-busy pattern,
    /// same as `OctreeMarchPass::stats_map_state`).
    count_map_state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl UserShaderEmitPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_emit bgl"),
            entries: &[
                bgl_storage_ro(0),  // leaves
                bgl_storage_rw(1),  // instances
                bgl_storage_rw(2),  // instance_count atomic
                bgl_storage_ro(3),  // shader_params
                bgl_storage_ro(4),  // mat_to_proto
                bgl_uniform(5),     // emit_params
                bgl_storage_rw(6),  // instance_aabbs (parallel to instances)
                bgl_storage_rw(7),  // instance_inv_world (parallel)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_emit layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = build_pipeline(device, &pipeline_layout, "");

        // Tiny placeholder buffers — engine resizes on first non-empty
        // frame and rebuilds the bind group via the buffers_epoch
        // comparison in `ensure_bind_group`.
        let mat_to_proto_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit mat_to_proto"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit leaves"),
            size: 32, // one EmitLeaf placeholder
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit params"),
            size: std::mem::size_of::<EmitParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user_shader_emit count readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            pipeline_layout,
            bind_group_layout,
            mat_to_proto_buffer,
            mat_to_proto_capacity: 16,
            leaves_buffer,
            leaves_capacity: 32,
            params_buffer,
            user_shader_source_hash: 0,
            bind_group: None,
            bind_group_inputs_epoch: 0,
            count_readback_buffer,
            count_map_state: std::sync::Arc::new(
                std::sync::atomic::AtomicU8::new(COUNT_MAP_IDLE),
            ),
        }
    }

    /// Encode a copy from `instance_count_buffer` into the local
    /// readback buffer. Skip-if-busy: if a previous map_async is
    /// pending, returns `false` and the caller should not call
    /// `submit_count_readback`.
    pub fn copy_count_for_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        instance_count_buffer: &wgpu::Buffer,
    ) -> bool {
        use std::sync::atomic::Ordering;
        let state = self.count_map_state.load(Ordering::Acquire);
        if state == COUNT_MAP_PENDING || state == COUNT_MAP_READY {
            return false;
        }
        encoder.copy_buffer_to_buffer(instance_count_buffer, 0, &self.count_readback_buffer, 0, 4);
        true
    }

    /// Schedule the async map. Call AFTER `queue.submit` of the
    /// encoder containing `copy_count_for_readback`.
    pub fn submit_count_readback(&self) {
        use std::sync::atomic::Ordering;
        let state = self.count_map_state.load(Ordering::Acquire);
        if state == COUNT_MAP_PENDING || state == COUNT_MAP_READY {
            return;
        }
        self.count_map_state.store(COUNT_MAP_PENDING, Ordering::Release);
        let state_arc = std::sync::Arc::clone(&self.count_map_state);
        let slice = self.count_readback_buffer.slice(0..4);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let next = if result.is_ok() { COUNT_MAP_READY } else { COUNT_MAP_FAILED };
            state_arc.store(next, Ordering::Release);
        });
    }

    /// Drain the count readback. Returns the latest emitted instance
    /// count if the previous `submit_count_readback` resolved, else
    /// `None`.
    pub fn try_drain_count(&self) -> Option<u32> {
        use std::sync::atomic::Ordering;
        let state = self.count_map_state.load(Ordering::Acquire);
        if state == COUNT_MAP_FAILED {
            self.count_map_state.store(COUNT_MAP_IDLE, Ordering::Release);
            return None;
        }
        if state != COUNT_MAP_READY {
            return None;
        }
        let slice = self.count_readback_buffer.slice(0..4);
        let count = {
            let view = slice.get_mapped_range();
            u32::from_le_bytes([view[0], view[1], view[2], view[3]])
        };
        self.count_readback_buffer.unmap();
        self.count_map_state.store(COUNT_MAP_IDLE, Ordering::Release);
        Some(count)
    }

    /// Recompile the pipeline when the user-shader source hash
    /// changes. No-op when the hash matches the last build.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        emit_chunk: &str,
        source_hash: u64,
    ) {
        if source_hash == self.user_shader_source_hash {
            return;
        }
        self.pipeline = build_pipeline(device, &self.pipeline_layout, emit_chunk);
        self.user_shader_source_hash = source_hash;
    }

    /// Upload painted-leaf records. Grows the GPU buffer if needed.
    pub fn upload_leaves(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, leaves: &[EmitLeaf]) {
        let bytes: &[u8] = bytemuck::cast_slice(leaves);
        let needed = bytes.len().max(32) as u64;
        if needed > self.leaves_capacity {
            self.leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_emit leaves"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.leaves_capacity = needed;
            self.bind_group = None;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.leaves_buffer, 0, bytes);
        }
    }

    /// Upload per-material lookup. Grows the buffer if needed.
    pub fn upload_mat_to_proto(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        table: &[MatToProto],
    ) {
        let bytes: &[u8] = bytemuck::cast_slice(table);
        let needed = bytes.len().max(16) as u64;
        if needed > self.mat_to_proto_capacity {
            self.mat_to_proto_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("user_shader_emit mat_to_proto"),
                size: needed,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.mat_to_proto_capacity = needed;
            self.bind_group = None;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.mat_to_proto_buffer, 0, bytes);
        }
    }

    /// Reset the instance count buffer to 0. Call BEFORE the dispatch.
    pub fn reset_instance_count(
        &self,
        queue: &wgpu::Queue,
        instance_count_buffer: &wgpu::Buffer,
    ) {
        queue.write_buffer(instance_count_buffer, 0, &[0u8; 4]);
    }

    /// Update the per-frame uniform.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &EmitParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Build / refresh the bind group. Caller passes the scene-side
    /// instance buffer + count + shader_params buffer (which the
    /// emit pass shares with shade / march).
    pub fn ensure_bind_group(
        &mut self,
        device: &wgpu::Device,
        instance_buffer: &wgpu::Buffer,
        instance_count_buffer: &wgpu::Buffer,
        instance_aabbs_buffer: &wgpu::Buffer,
        instance_inv_world_buffer: &wgpu::Buffer,
        shader_params_buffer: &wgpu::Buffer,
        inputs_epoch: u64,
    ) {
        if self.bind_group.is_some() && inputs_epoch == self.bind_group_inputs_epoch {
            return;
        }
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("user_shader_emit bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.leaves_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: instance_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: instance_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: shader_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.mat_to_proto_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: instance_aabbs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: instance_inv_world_buffer.as_entire_binding() },
            ],
        }));
        self.bind_group_inputs_epoch = inputs_epoch;
    }

    /// Dispatch the emit pass for the given leaf count. Caller must
    /// have called `upload_leaves`, `upload_mat_to_proto`,
    /// `update_params`, `reset_instance_count`, and `ensure_bind_group`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        leaf_count: u32,
    ) {
        if leaf_count == 0 {
            return;
        }
        let bg = self.bind_group.as_ref().expect("ensure_bind_group not called");
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("user_shader_emit"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        let workgroups = leaf_count.div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    emit_chunk: &str,
) -> wgpu::ComputePipeline {
    let template = wesl::include_wesl!("user_shader_emit");
    let source = splice_const_marker(template, "USER_EMIT_DISPATCH", emit_chunk);
    let module = compile_pass_shader(device, &source, "user_shader_emit");
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("user_shader_emit"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("emit_main"),
        compilation_options: Default::default(),
        cache: None,
    })
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
    fn emit_leaf_size_is_32_bytes() {
        assert_eq!(mem::size_of::<EmitLeaf>(), 32);
    }

    #[test]
    fn mat_to_proto_size_is_8_bytes() {
        assert_eq!(mem::size_of::<MatToProto>(), 8);
    }

    #[test]
    fn emit_params_size_is_16_bytes() {
        // Uniform buffers must be at least 16 B and 16-aligned. The
        // `_pad0` field rounds the struct up to satisfy that.
        assert_eq!(mem::size_of::<EmitParams>(), 16);
    }
}
