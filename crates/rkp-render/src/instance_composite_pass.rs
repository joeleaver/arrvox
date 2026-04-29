//! Stage 6b — instance-march composite pass.
//!
//! The compute pipeline that consumes the per-pixel
//! [`crate::instance_march_pass::InstanceMarchHit`] array produced by
//! Stage 6a's `instance_march_main` and overlays winning instance hits
//! onto the host G-buffer.
//!
//! ## Bind groups
//!
//! * **Group 0 — instance inputs.**
//!   - `binding 0`: storage RO `output_hits[]` (the
//!     [`crate::instance_march_pass::InstanceMarchHit`] buffer).
//!   - `binding 1`: uniform `MarchUniforms` (provides screen dims).
//!     Shared with the march pass — bind the same buffer.
//!   - `binding 2`: uniform `MarchCameraUniform`. Shared with the march
//!     pass — bind the same buffer.
//! * **Group 1 — host G-buffer reads.** Four sampled textures matching
//!   the formats in [`crate::gbuffer`]:
//!     0: `Rgba32Float` position, 1: `Rgba16Float` normal,
//!     2: `Rg32Uint` material, 3: `R32Uint` leaf_slot.
//! * **Group 2 — merged G-buffer writes.** Four `write`-only storage
//!   textures with the same formats as group 1.
//!
//! ## Why a separate output set
//!
//! WebGPU disallows binding the same physical texture as a writable
//! storage view alongside any other view in the same dispatch. Reading
//! the host G-buffer + writing it back in-place would require either
//! `read_write` storage access (format-restricted, not portable) or
//! two-pass copy. The two-set design is the cleanest single-pass
//! solution; Stage 6c can wire downstream passes to read from the
//! merged set, treating the host G-buffer as intermediate scratch.
//!
//! ## Source composition
//!
//! Standalone WGSL — no helper concatenation needed. The composite
//! re-declares the small `MarchUniforms` / `MarchCameraUniform` /
//! `InstanceMarchHit` shapes used here; they're documented as
//! mirrors of the Stage 6a structs and validated by the inline naga
//! test in this module.

use std::num::NonZeroU64;

use crate::gbuffer::{
    GBUFFER_LEAF_SLOT_FORMAT, GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT,
    GBUFFER_POSITION_FORMAT,
};
use crate::instance_march_pass::{InstanceMarchHit, MarchCameraUniform, MarchUniforms};

/// Source text — single self-contained WGSL file. Exposed so tests can
/// validate it through naga without going through pipeline creation.
pub fn instance_composite_source() -> &'static str {
    include_str!("shaders/user_shader_instance_composite.wgsl")
}

/// Pipeline owner. Construction validates the composed WGSL with naga
/// and creates three bind-group layouts + the compute pipeline.
pub struct InstanceCompositePass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub group2_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl InstanceCompositePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_composite group0 (hits + uniforms)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<InstanceMarchHit>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<MarchCameraUniform>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_composite group1 (host gbuf reads)"),
            entries: &[
                texture_2d_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                texture_2d_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_2d_entry(2, wgpu::TextureSampleType::Uint),
                texture_2d_entry(3, wgpu::TextureSampleType::Uint),
            ],
        });

        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_composite group2 (merged gbuf writes)"),
            entries: &[
                storage_tex_write_entry(0, GBUFFER_POSITION_FORMAT),
                storage_tex_write_entry(1, GBUFFER_NORMAL_FORMAT),
                storage_tex_write_entry(2, GBUFFER_MATERIAL_FORMAT),
                storage_tex_write_entry(3, GBUFFER_LEAF_SLOT_FORMAT),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instance_composite pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
            ],
            immediate_size: 0,
        });

        let source = instance_composite_source();
        crate::validate_wgsl(source, "instance_composite");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance_composite"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_composite_main"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("instance_composite_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            pipeline_layout,
            pipeline,
        }
    }

    /// Encode the per-pixel composite dispatch into an open compute
    /// pass. Caller binds the three groups before calling. Same dispatch
    /// shape as the march pass — `(ceil(W/8), ceil(H/8), 1)` —
    /// workgroup_size in WGSL is (8, 8, 1).
    pub fn dispatch_per_pixel(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        screen_width: u32,
        screen_height: u32,
    ) {
        cpass.set_pipeline(&self.pipeline);
        cpass.dispatch_workgroups(
            crate::instance_march_pass::InstanceMarchPass::workgroup_count_for_pixels(screen_width),
            crate::instance_march_pass::InstanceMarchPass::workgroup_count_for_pixels(screen_height),
            1,
        );
    }
}

fn texture_2d_entry(binding: u32, sample: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: sample,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_tex_write_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_wgsl_validates() {
        let source = instance_composite_source();
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!("parse error:\n{}", e.emit_to_string(source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }

    #[test]
    fn composite_source_uses_helpers_consistent_with_march() {
        // The composite re-declares MarchUniforms / MarchCameraUniform /
        // InstanceMarchHit. If the march's struct shapes drift, the
        // composite's pack code will silently misinterpret the
        // packed material — guard against that by asserting the
        // composite source still mentions the shared field names the
        // march writes.
        let src = instance_composite_source();
        assert!(src.contains("struct InstanceMarchHit"));
        assert!(src.contains("material_packed"));
        assert!(src.contains("t_world"));
        assert!(src.contains("struct MarchCameraUniform"));
        assert!(src.contains("instance_composite_main"));
    }
}
