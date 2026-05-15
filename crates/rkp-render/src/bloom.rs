//! Bloom compute pass (pre-upscale).
//!
//! Extracts bright pixels from the HDR shading output, downsamples through
//! a 4-level mip chain using a 13-tap tent filter, and applies a separable
//! 9-tap Gaussian blur at each level. The blurred bloom textures are stored
//! for later compositing in the post-upscale bloom composite pass (10.5).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Number of bloom mip levels (4 levels: full → 1/2 → 1/4 → 1/8).
pub const BLOOM_MIP_LEVELS: u32 = 4;

/// Default bloom threshold (HDR luminance above this gets extracted).
pub const DEFAULT_BLOOM_THRESHOLD: f32 = 1.0;

/// Default soft knee width.
pub const DEFAULT_BLOOM_KNEE: f32 = 0.5;

/// Bloom texture format — half-precision HDR, supports STORAGE_BINDING.
const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// GPU-uploadable bloom parameters (16 bytes).
///
/// The `threshold` field is dual-purpose:
/// - In the **extract** shader: luminance threshold for bright-pixel extraction.
/// - In the **blur** shader: direction flag — `0.0` = horizontal, `> 0.5` = vertical.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BloomParams {
    /// Extract: luminance threshold. Blur: 0.0 = horizontal, 1.0 = vertical.
    pub threshold: f32,
    /// Soft knee width (extract only).
    pub knee: f32,
    /// Dispatch width (destination pixel width).
    pub width: u32,
    /// Dispatch height (destination pixel height).
    pub height: u32,
}

/// Bloom compute pass.
///
/// Runs 12 compute dispatches per frame:
/// 1. Extract (HDR → mip0)
/// 2. Downsample ×3 (mip0→mip1, mip1→mip2, mip2→mip3)
/// 3. Blur H+V ×4 (ping-pong through scratch textures)
///
/// After `dispatch`, the blurred mip views are accessible via [`BloomPass::mip_views`]
/// for the composite pass (task 10.5).
#[allow(dead_code)]
pub struct BloomPass {
    // Three pipelines, all sharing one bind group layout
    extract_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,

    // Shared bind group layout (texture_2d, storage_2d, uniform)
    bind_group_layout: wgpu::BindGroupLayout,

    // Mip chain: [mip0 (full), mip1 (1/2), mip2 (1/4), mip3 (1/8)]
    mip_textures: [wgpu::Texture; BLOOM_MIP_LEVELS as usize],
    mip_views: [wgpu::TextureView; BLOOM_MIP_LEVELS as usize],

    // Scratch textures for blur ping-pong (same sizes as mips)
    scratch_textures: [wgpu::Texture; BLOOM_MIP_LEVELS as usize],
    scratch_views: [wgpu::TextureView; BLOOM_MIP_LEVELS as usize],

    // Extract: HDR input → mip0
    extract_bind_group: wgpu::BindGroup,
    extract_params_buffer: wgpu::Buffer,

    // Downsample: mip[i] → mip[i+1], for i in 0..3
    downsample_bind_groups: [wgpu::BindGroup; 3],
    downsample_params_buffers: [wgpu::Buffer; 3],

    // Blur horizontal: mip[i] → scratch[i]
    blur_h_bind_groups: [wgpu::BindGroup; BLOOM_MIP_LEVELS as usize],
    blur_h_params_buffers: [wgpu::Buffer; BLOOM_MIP_LEVELS as usize],

    // Blur vertical: scratch[i] → mip[i]
    blur_v_bind_groups: [wgpu::BindGroup; BLOOM_MIP_LEVELS as usize],
    blur_v_params_buffers: [wgpu::Buffer; BLOOM_MIP_LEVELS as usize],

    // Mip dimensions for dispatch sizing
    mip_widths: [u32; BLOOM_MIP_LEVELS as usize],
    mip_heights: [u32; BLOOM_MIP_LEVELS as usize],
}

impl BloomPass {
    /// Create the bloom pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `hdr_view`: HDR texture view from the shading pass (full internal resolution).
    /// - `width`: internal resolution width (e.g. 960).
    /// - `height`: internal resolution height (e.g. 540).
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader module (all 3 entry points) ---
        let shader = crate::compile_pass_shader(device, wesl::include_wesl!("bloom"), "bloom");

        // --- Mip dimensions ---
        let mut mip_widths = [0u32; BLOOM_MIP_LEVELS as usize];
        let mut mip_heights = [0u32; BLOOM_MIP_LEVELS as usize];
        mip_widths[0] = width;
        mip_heights[0] = height;
        for i in 1..BLOOM_MIP_LEVELS as usize {
            mip_widths[i] = (mip_widths[i - 1] / 2).max(1);
            mip_heights[i] = (mip_heights[i - 1] / 2).max(1);
        }

        // --- Create mip + scratch textures ---
        let make_bloom_texture = |w: u32, h: u32, label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: BLOOM_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let mip_textures: [wgpu::Texture; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                make_bloom_texture(mip_widths[i], mip_heights[i], &format!("bloom mip{i}"))
            });
        let mip_views: [wgpu::TextureView; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| mip_textures[i].create_view(&Default::default()));

        let scratch_textures: [wgpu::Texture; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                make_bloom_texture(
                    mip_widths[i],
                    mip_heights[i],
                    &format!("bloom scratch{i}"),
                )
            });
        let scratch_views: [wgpu::TextureView; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| scratch_textures[i].create_view(&Default::default()));

        // --- Shared bind group layout ---
        // binding 0: texture_2d<f32> (non-filterable, sampled input)
        // binding 1: texture_storage_2d<rgba16float, write>
        // binding 2: uniform buffer (BloomParams)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom bind group layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: BLOOM_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        // --- Pipeline layout (shared) ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // --- Three pipelines ---
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("bloom {entry} pipeline")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let extract_pipeline = make_pipeline("extract");
        let downsample_pipeline = make_pipeline("downsample");
        let blur_pipeline = make_pipeline("blur");

        // --- Helper: make a bind group (src view, dst view, params buffer) ---
        let make_bind_group = |label: &str,
                               src: &wgpu::TextureView,
                               dst: &wgpu::TextureView,
                               buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf.as_entire_binding(),
                    },
                ],
            })
        };

        // --- Extract bind group + params buffer ---
        let extract_params = BloomParams {
            threshold: DEFAULT_BLOOM_THRESHOLD,
            knee: DEFAULT_BLOOM_KNEE,
            width,
            height,
        };
        let extract_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bloom extract params"),
            contents: bytemuck::bytes_of(&extract_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let extract_bind_group =
            make_bind_group("bloom extract bg", hdr_view, &mip_views[0], &extract_params_buffer);

        // --- Downsample bind groups + params buffers (mip[i] → mip[i+1]) ---
        let downsample_params_buffers: [wgpu::Buffer; 3] = std::array::from_fn(|i| {
            let params = BloomParams {
                threshold: 0.0,
                knee: 0.0,
                width: mip_widths[i + 1],
                height: mip_heights[i + 1],
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("bloom downsample params mip{i}")),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });
        let downsample_bind_groups: [wgpu::BindGroup; 3] = std::array::from_fn(|i| {
            make_bind_group(
                &format!("bloom downsample bg mip{i}→mip{}", i + 1),
                &mip_views[i],
                &mip_views[i + 1],
                &downsample_params_buffers[i],
            )
        });

        // --- Blur bind groups + params buffers ---
        // H: mip[i] → scratch[i], threshold=0.0 (horizontal)
        let blur_h_params_buffers: [wgpu::Buffer; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                let params = BloomParams {
                    threshold: 0.0,
                    knee: 0.0,
                    width: mip_widths[i],
                    height: mip_heights[i],
                };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("bloom blur_h params mip{i}")),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            });
        let blur_h_bind_groups: [wgpu::BindGroup; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                make_bind_group(
                    &format!("bloom blur_h bg mip{i}"),
                    &mip_views[i],
                    &scratch_views[i],
                    &blur_h_params_buffers[i],
                )
            });

        // V: scratch[i] → mip[i], threshold=1.0 (vertical)
        let blur_v_params_buffers: [wgpu::Buffer; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                let params = BloomParams {
                    threshold: 1.0,
                    knee: 0.0,
                    width: mip_widths[i],
                    height: mip_heights[i],
                };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("bloom blur_v params mip{i}")),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            });
        let blur_v_bind_groups: [wgpu::BindGroup; BLOOM_MIP_LEVELS as usize] =
            std::array::from_fn(|i| {
                make_bind_group(
                    &format!("bloom blur_v bg mip{i}"),
                    &scratch_views[i],
                    &mip_views[i],
                    &blur_v_params_buffers[i],
                )
            });

        Self {
            extract_pipeline,
            downsample_pipeline,
            blur_pipeline,
            bind_group_layout,
            mip_textures,
            mip_views,
            scratch_textures,
            scratch_views,
            extract_bind_group,
            extract_params_buffer,
            downsample_bind_groups,
            downsample_params_buffers,
            blur_h_bind_groups,
            blur_h_params_buffers,
            blur_v_bind_groups,
            blur_v_params_buffers,
            mip_widths,
            mip_heights,
        }
    }

    /// Dispatch all bloom passes into the command encoder.
    ///
    /// Order:
    /// 1. Extract (HDR → mip0)
    /// 2. Downsample ×3 (mip0→mip1, mip1→mip2, mip2→mip3)
    /// 3. Blur H+V ×4 (ping-pong through scratch per mip level)
    ///
    /// Each dispatch is its own compute pass to ensure pipeline barriers between
    /// reads and writes on the same texture.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        // --- Pass 1: Extract ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bloom extract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &self.extract_bind_group, &[]);
            pass.dispatch_workgroups(
                self.mip_widths[0].div_ceil(8),
                self.mip_heights[0].div_ceil(8),
                1,
            );
        }

        // --- Passes 2-4: Downsample ---
        for i in 0..3usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("bloom downsample mip{i}→mip{}", i + 1)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &self.downsample_bind_groups[i], &[]);
            pass.dispatch_workgroups(
                self.mip_widths[i + 1].div_ceil(8),
                self.mip_heights[i + 1].div_ceil(8),
                1,
            );
        }

        // --- Pass 5: Blur H (all mips in one compute pass) ---
        //
        // Each mip's H blur reads `mip[i]` and writes `scratch[i]`, so
        // the 4 dispatches touch disjoint resources and can share one
        // compute pass — the GPU pipelines them freely. The previous
        // code opened a separate pass per mip (8 passes total for H+V),
        // each of which forced a pipeline barrier; merging cuts those
        // boundaries from 8 to 2 and saved measurable frame time on
        // the splat5 elephant scene.
        //
        // H and V can't share a pass — V at mip[i] reads scratch[i]
        // which the matching H write produced, so they need a barrier.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bloom blur H (all mips)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            for i in 0..BLOOM_MIP_LEVELS as usize {
                pass.set_bind_group(0, &self.blur_h_bind_groups[i], &[]);
                pass.dispatch_workgroups(
                    self.mip_widths[i].div_ceil(8),
                    self.mip_heights[i].div_ceil(8),
                    1,
                );
            }
        }

        // --- Pass 6: Blur V (all mips in one compute pass) ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bloom blur V (all mips)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            for i in 0..BLOOM_MIP_LEVELS as usize {
                pass.set_bind_group(0, &self.blur_v_bind_groups[i], &[]);
                pass.dispatch_workgroups(
                    self.mip_widths[i].div_ceil(8),
                    self.mip_heights[i].div_ceil(8),
                    1,
                );
            }
        }
    }

    /// Update the extraction threshold and knee at runtime.
    ///
    /// Writes to the extract params buffer (bytes 0-7).
    pub fn set_threshold(&self, queue: &wgpu::Queue, threshold: f32, knee: f32) {
        queue.write_buffer(
            &self.extract_params_buffer,
            0,
            bytemuck::cast_slice(&[threshold, knee]),
        );
    }

    /// Returns the blurred mip texture views for the composite pass (task 10.5).
    ///
    /// After `dispatch` completes, mip_views[0] holds the full-res blurred bloom,
    /// and mip_views[1..3] hold progressively lower-resolution contributions.
    pub fn mip_views(&self) -> &[wgpu::TextureView; BLOOM_MIP_LEVELS as usize] {
        &self.mip_views
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_params_size_is_16() {
        assert_eq!(std::mem::size_of::<BloomParams>(), 16);
    }

    #[test]
    fn bloom_params_pod_roundtrip() {
        let p = BloomParams {
            threshold: 1.0,
            knee: 0.5,
            width: 960,
            height: 540,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &BloomParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.threshold, p2.threshold);
        assert_eq!(p.width, p2.width);
    }

    #[test]
    fn mip_dimensions() {
        let base_w = 960u32;
        let base_h = 540u32;
        let mut w = base_w;
        let mut h = base_h;
        let mut widths = Vec::new();
        let mut heights = Vec::new();
        for _ in 0..BLOOM_MIP_LEVELS {
            widths.push(w);
            heights.push(h);
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        assert_eq!(widths, vec![960, 480, 240, 120]);
        assert_eq!(heights, vec![540, 270, 135, 67]);
    }

    #[test]
    fn bloom_mip_count() {
        assert_eq!(BLOOM_MIP_LEVELS, 4);
    }

    #[test]
    fn default_threshold() {
        assert_eq!(DEFAULT_BLOOM_THRESHOLD, 1.0);
        assert_eq!(DEFAULT_BLOOM_KNEE, 0.5);
    }
}
