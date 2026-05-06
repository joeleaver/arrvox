//! Per-cluster LOD-select compute pass — Phase 6.2.
//!
//! For each draw, runs one workgroup per ~64 clusters of the asset's
//! [`MeshletCluster`] table. Each thread applies the Karis-Nanite
//! admit rule (`parent_group_error_proj ≥ thresh` AND
//! `cluster_error_proj < thresh`) and writes a
//! [`DrawIndexedIndirectArgs`] entry into a per-draw `args` buffer:
//! admitted clusters get real draw args, non-admitted slots get
//! zeroed args (`index_count = 0` → no-op draw at
//! `multi_draw_indexed_indirect` time without a CPU count read).
//!
//! Bind-group shape:
//!   · `g0` = camera + per-draw `MeshLodSelectParams` uniforms
//!     (`pixel_threshold`, `focal_pixels`, `cluster_count`)
//!   · `g1` = per-instance world matrix uniform (reuses splat's
//!     `g1_layout` — same `MeshInstance` shape as the render path)
//!   · `g2` = per-asset cluster table (storage, read) + per-draw
//!     args buffer (storage, read-write)
//!
//! Phase 6.4 will run this pipeline twice per frame (primary +
//! shadow) with different `pixel_threshold` so the shadow chain
//! picks ~`lod + 1`.

use bytemuck::{Pod, Zeroable};

/// `wgpu::DrawIndexedIndirectArgs` shape — what
/// `multi_draw_indexed_indirect` reads, 5 × `u32`. Matches the
/// `DrawIndexedIndirectArgs` struct in `mesh_lod_select.wesl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

const _: () = assert!(std::mem::size_of::<DrawIndexedIndirectArgs>() == 20);

/// Per-draw uniform driving the LOD-select admit rule. 16 B,
/// std140-aligned. Mirrors `MeshLodSelectParams` in WGSL. The
/// shader derives `focal_pixels` from `camera.view_proj[1][1]`
/// directly so the engine doesn't have to plumb a FOV value
/// alongside the camera buffer.
///
/// `force_admit` is a debug knob: when non-zero, every cluster is
/// admitted regardless of the Karis selection rule. Used to bisect
/// "no geometry" issues in the indirect dispatch path — if
/// geometry shows up with `force_admit=1` but not with `=0`, the
/// bug is in the admit-rule math, not in the
/// `multi_draw_indexed_indirect` plumbing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshLodSelectParams {
    pub pixel_threshold: f32,
    pub cluster_count: u32,
    pub force_admit: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<MeshLodSelectParams>() == 16);

/// Compute pipeline + reusable bind-group layouts.
pub struct MeshLodSelectPass {
    pub pipeline: wgpu::ComputePipeline,
    /// Layout for `g0`: camera (binding 0) + LOD params (binding 1).
    pub g0_layout: wgpu::BindGroupLayout,
    /// Layout for `g2`: cluster table + args buffer. (`g1` is the
    /// caller-passed splat g1 layout for the per-instance uniform.)
    pub g2_layout: wgpu::BindGroupLayout,
}

impl MeshLodSelectPass {
    /// Build the compute pipeline. `splat_g1_layout` is reused for
    /// `g1` directly so the existing per-VR
    /// `splat_instance_bind_groups` drive both the render path (vertex
    /// stage) and this compute pass without duplication. The splat
    /// g1 layout's visibility includes COMPUTE for this purpose.
    pub fn new(device: &wgpu::Device, splat_g1_layout: &wgpu::BindGroupLayout) -> Self {
        let g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_lod_select g0 (camera + params)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        let g2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_lod_select g2 (clusters + args)"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_lod_select layout"),
            bind_group_layouts: &[Some(&g0_layout), Some(splat_g1_layout), Some(&g2_layout)],
            immediate_size: 0,
        });

        let module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("mesh_lod_select"),
            "mesh_lod_select",
        );

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh_lod_select"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("lod_select"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, g0_layout, g2_layout }
    }

    /// Build the per-draw `g0` bind group. `params_buffer` is a 16 B
    /// uniform that the caller must keep alive for the duration of
    /// the bind group; one allocation per draw-slot is the standard
    /// pattern. Camera binding is the per-VR camera buffer that
    /// already exists.
    pub fn create_g0_bind_group(
        &self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_lod_select g0 bg"),
            layout: &self.g0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Build the per-draw `g2` bind group — asset cluster table + the
    /// per-draw args buffer. The args buffer must have STORAGE +
    /// INDIRECT usage so the same buffer can be written here and
    /// read by `multi_draw_indexed_indirect` in the render pass.
    pub fn create_g2_bind_group(
        &self,
        device: &wgpu::Device,
        cluster_buffer: &wgpu::Buffer,
        args_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_lod_select g2 bg"),
            layout: &self.g2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: args_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Convenience for callers that already hold all three bind
    /// groups: dispatch one workgroup per 64 clusters.
    pub fn dispatch<'pass>(
        &'pass self,
        cpass: &mut wgpu::ComputePass<'pass>,
        g0: &'pass wgpu::BindGroup,
        g1: &'pass wgpu::BindGroup,
        g2: &'pass wgpu::BindGroup,
        cluster_count: u32,
    ) {
        if cluster_count == 0 {
            return;
        }
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, g0, &[]);
        cpass.set_bind_group(1, g1, &[]);
        cpass.set_bind_group(2, g2, &[]);
        cpass.dispatch_workgroups(cluster_count.div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_layout_matches_wgpu_indirect_size() {
        // wgpu::util::DrawIndexedIndirectArgs is 20 bytes (5 × u32).
        // Our type must match for `multi_draw_indexed_indirect` to
        // read the buffer correctly.
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
        assert_eq!(std::mem::align_of::<DrawIndexedIndirectArgs>(), 4);
    }

    #[test]
    fn lod_select_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("mesh_lod_select");
        crate::validate_wgsl(src, "mesh_lod_select");
    }
}
