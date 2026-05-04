//! Bloom composite compute pass (post-upscale).
//!
//! Bilinear-samples the bloom mip chain from internal resolution,
//! combines all 4 mip levels with decreasing weights, and additively
//! blends onto the upscaled HDR color at display resolution.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default bloom composite intensity (overall bloom strength).
pub const DEFAULT_BLOOM_INTENSITY: f32 = 0.1;

/// GPU-uploadable bloom composite parameters (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BloomCompositeParams {
    /// Display resolution width in pixels.
    pub display_width: u32,
    /// Display resolution height in pixels.
    pub display_height: u32,
    /// Overall bloom strength multiplier (default 0.3).
    pub bloom_intensity: f32,
    /// Padding to 16 bytes.
    pub _pad: u32,
}

/// Bloom composite compute pass (post-upscale).
///
/// Reads the upscaled HDR buffer at display resolution, bilinear-samples
/// the 4-level bloom mip chain (at internal resolution) and additively
/// blends the weighted sum onto the HDR output.
///
/// Mip weights: mip0=0.5, mip1=0.3, mip2=0.15, mip3=0.05 (sums to 1.0),
/// then multiplied by `bloom_intensity`.
///
/// After [`BloomCompositePass::dispatch`], the result is in
/// [`BloomCompositePass::output_view`].
#[allow(dead_code)]
pub struct BloomCompositePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Composited HDR output texture (Rgba16Float, display resolution).
    pub output_texture: wgpu::Texture,
    /// View of the composited HDR output texture.
    pub output_view: wgpu::TextureView,
    display_width: u32,
    display_height: u32,
}

impl BloomCompositePass {
    /// Create the bloom composite pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `upscaled_hdr_view`: upscaled HDR texture view at display resolution (non-filterable).
    /// - `bloom_mip_views`: the 4 blurred bloom mip views from [`crate::BloomPass::mip_views`].
    /// - `display_width`: display resolution width (e.g. 1920).
    /// - `display_height`: display resolution height (e.g. 1080).
    pub fn new(
        device: &wgpu::Device,
        upscaled_hdr_view: &wgpu::TextureView,
        bloom_mip_views: &[wgpu::TextureView; 4],
        display_width: u32,
        display_height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = crate::compile_pass_shader(
            device, wesl::include_wesl!("bloom_composite"), "bloom_composite",
        );

        // --- Bilinear sampler for bloom texture upsampling ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom composite bilinear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // --- Output texture (Rgba16Float, display resolution) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom composite output texture"),
            size: wgpu::Extent3d {
                width: display_width,
                height: display_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- Bind group layout ---
        // binding 0: texture_2d<f32> (upscaled HDR, non-filterable — textureLoad)
        // binding 1: texture_2d<f32> (bloom mip0, filterable — textureSampleLevel)
        // binding 2: texture_2d<f32> (bloom mip1, filterable)
        // binding 3: texture_2d<f32> (bloom mip2, filterable)
        // binding 4: texture_2d<f32> (bloom mip3, filterable)
        // binding 5: sampler (filtering)
        // binding 6: texture_storage_2d<rgba16float, write>
        // binding 7: uniform buffer (BloomCompositeParams)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom composite bind group layout"),
                entries: &[
                    // 0: upscaled HDR input (non-filterable, loaded by pixel coord)
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
                    // 1-4: bloom mip textures (filterable — bilinear via textureSampleLevel)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 5: bilinear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 6: storage texture output
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 7: uniform params
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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

        // --- Params buffer ---
        let default_params = BloomCompositeParams {
            display_width,
            display_height,
            bloom_intensity: DEFAULT_BLOOM_INTENSITY,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom composite params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom composite bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(upscaled_hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&bloom_mip_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bloom composite pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloom composite pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            sampler,
            output_texture,
            output_view,
            display_width,
            display_height,
        }
    }

    /// Dispatch the bloom composite pass into the command encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bloom composite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.display_width.div_ceil(8),
            self.display_height.div_ceil(8),
            1,
        );
    }

    /// Update the bloom intensity at runtime.
    ///
    /// Writes only the `bloom_intensity` field (byte offset 8).
    pub fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 8, bytemuck::bytes_of(&intensity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_composite_params_size_is_16() {
        assert_eq!(std::mem::size_of::<BloomCompositeParams>(), 16);
    }

    #[test]
    fn bloom_composite_params_pod_roundtrip() {
        let p = BloomCompositeParams {
            display_width: 1920,
            display_height: 1080,
            bloom_intensity: 0.3,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &BloomCompositeParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.bloom_intensity, p2.bloom_intensity);
    }

    #[test]
    fn default_bloom_intensity() {
        assert!((DEFAULT_BLOOM_INTENSITY - 0.1).abs() < f32::EPSILON);
    }
}
