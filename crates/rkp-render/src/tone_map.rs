//! Tone mapping compute pass — HDR to LDR conversion.
//!
//! Reads the HDR `Rgba16Float` output from the shading pass, applies exposure,
//! tone mapping (ACES or AgX), and sRGB gamma correction, and writes to an
//! `Rgba8Unorm` texture ready for display via the blit pass.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// LDR output format.
pub const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Tone mapping curve selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapMode {
    /// ACES filmic (Narkowicz approximation). Good default, slightly warm.
    Aces = 0,
    /// AgX (Troy Sobotka). Better highlight/shadow preservation.
    AgX = 1,
}

/// GPU-uploadable tone map parameters (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ToneMapParams {
    /// Tone mapping mode (0 = ACES, 1 = AgX).
    pub mode: u32,
    /// Exposure multiplier (typically 2^EV from auto-exposure).
    pub exposure: f32,
    #[doc(hidden)]
    pub _pad0: u32,
    #[doc(hidden)]
    pub _pad1: u32,
}

/// Default exposure multiplier (1.0 = no exposure adjustment).
pub const DEFAULT_EXPOSURE: f32 = 1.0;

/// Tone mapping compute pass.
#[allow(dead_code)]
pub struct ToneMapPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// LDR output texture.
    pub ldr_texture: wgpu::Texture,
    /// View for the LDR output (used by blit pass).
    pub ldr_view: wgpu::TextureView,
    /// Bind group layout for HDR input (sampled texture).
    hdr_input_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for HDR input.
    hdr_input_bind_group: wgpu::BindGroup,
    /// Bind group layout for LDR output (storage texture).
    ldr_output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for LDR output.
    ldr_output_bind_group: wgpu::BindGroup,
    /// Tone map params buffer.
    params_buffer: wgpu::Buffer,
    /// Bind group layout for params (group 2).
    params_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for params.
    params_bind_group: wgpu::BindGroup,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl ToneMapPass {
    /// Create the tone mapping pass.
    ///
    /// `hdr_view` is the HDR texture view from the shading pass.
    /// `exposure_buffer` is the auto-exposure GPU buffer ([current_ev, target_ev]).
    /// If provided, the shader reads adapted EV and applies `2^EV` as exposure.
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        Self::new_with_exposure(device, hdr_view, width, height, None)
    }

    /// Create the tone mapping pass with an auto-exposure buffer binding.
    pub fn new_with_exposure(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        exposure_buffer: Option<&wgpu::Buffer>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tone_map.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tone_map.wgsl").into()),
        });

        // LDR output texture
        let ldr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ldr output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ldr_view = ldr_texture.create_view(&Default::default());

        // Group 0: HDR input (sampled texture)
        let hdr_input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tone map hdr input layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let hdr_input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone map hdr input bind group"),
            layout: &hdr_input_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(hdr_view),
            }],
        });

        // Group 1: LDR output (storage texture)
        let ldr_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tone map ldr output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: LDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let ldr_output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone map ldr output bind group"),
            layout: &ldr_output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ldr_view),
            }],
        });

        // Group 2: Tone map params (mode + exposure) + auto-exposure buffer
        let tone_params = ToneMapParams {
            mode: ToneMapMode::Aces as u32,
            exposure: DEFAULT_EXPOSURE,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tone map params"),
            contents: bytemuck::bytes_of(&tone_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // If no auto-exposure buffer provided, create a dummy with EV=0 (multiplier=1.0).
        let dummy_exposure_buf;
        let exposure_buf = match exposure_buffer {
            Some(buf) => buf,
            None => {
                dummy_exposure_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tone map dummy exposure"),
                    contents: bytemuck::cast_slice(&[0.0f32, 0.0f32]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                &dummy_exposure_buf
            }
        };

        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tone map params layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone map params bind group"),
            layout: &params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exposure_buf.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tone map pipeline layout"),
            bind_group_layouts: &[
                Some(&hdr_input_bind_group_layout),
                Some(&ldr_output_bind_group_layout),
                Some(&params_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tone map pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            ldr_texture,
            ldr_view,
            hdr_input_bind_group_layout,
            hdr_input_bind_group,
            ldr_output_bind_group_layout,
            ldr_output_bind_group,
            params_buffer,
            params_bind_group_layout,
            params_bind_group,
            width,
            height,
        }
    }

    /// Dispatch the tone mapping compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tone map"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.hdr_input_bind_group, &[]);
        pass.set_bind_group(1, &self.ldr_output_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// The LDR output texture (for screenshot staging copy).
    pub fn ldr_texture(&self) -> &wgpu::Texture {
        &self.ldr_texture
    }

    /// Set the tone mapping mode (ACES or AgX).
    pub fn set_mode(&self, queue: &wgpu::Queue, mode: ToneMapMode) {
        let mode_val = mode as u32;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&mode_val));
    }

    /// Set the exposure multiplier.
    pub fn set_exposure(&self, queue: &wgpu::Queue, exposure: f32) {
        queue.write_buffer(&self.params_buffer, 4, bytemuck::bytes_of(&exposure));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_map_params_size_is_16() {
        assert_eq!(std::mem::size_of::<ToneMapParams>(), 16);
    }

    #[test]
    fn tone_map_params_pod_roundtrip() {
        let p = ToneMapParams {
            mode: 1,
            exposure: 2.5,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &ToneMapParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.mode, p2.mode);
        assert_eq!(p.exposure, p2.exposure);
    }

    #[test]
    fn tone_map_mode_values() {
        assert_eq!(ToneMapMode::Aces as u32, 0);
        assert_eq!(ToneMapMode::AgX as u32, 1);
    }

    #[test]
    fn default_exposure() {
        assert_eq!(DEFAULT_EXPOSURE, 1.0);
    }
}
