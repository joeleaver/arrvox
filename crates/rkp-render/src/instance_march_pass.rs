//! Stage 5b — instance-march compute pipeline.
//!
//! The march that consumes everything Stages 1-5a produced:
//!
//!   * **Group 0** — pool buffers (`octree_nodes`, `brick_pool`,
//!     `leaf_attr_pool`). Same layout as the Stage 5a test pipeline so
//!     the shared helpers in
//!     `shaders/user_shader_instance_march_helpers.wgsl` work against
//!     either one.
//!   * **Group 1** — per-frame instance state: `regions_buffer`,
//!     `instance_pool`, `tile_index_buffer`, `instance_alloc`. The first
//!     three are produced by Stage 4 (`InstanceRegionCache` +
//!     `flatten_tile_index`); `instance_alloc` is the per-region atomic
//!     counter the emit pass writes.
//!   * **Group 2** — `proto_lookup_buffer` from
//!     [`crate::instance_proto_lookup::flatten_prototype_lookup`].
//!   * **Group 3** — march uniforms, ray buffer, output hit buffer.
//!     Output is `array<InstanceMarchHit>`; one slot per ray.
//!
//! V1 is single-ray-per-dispatch — `dispatch_workgroups(num_rays, 1, 1)`
//! with workgroup_size(1). Stage 6 will batch by screen tile.
//!
//! ## Source composition
//!
//! WGSL source is the concatenation of
//! [`crate::user_shader_instance_march::instance_march_helpers_source`]
//! and `shaders/user_shader_instance_march_main.wgsl`. The helpers
//! file declares the @group(0) pool bindings + helper fns; the main
//! file declares the @group(1/2/3) bindings + the `@compute` entry.

use std::num::NonZeroU64;

use crate::instance_proto_lookup::GpuPrototypeEntry;
use crate::instance_tile_index_gpu::GpuTileIndexEntry;
use crate::user_shader_emit_pass::EmitRegionUniform;
use crate::user_shader_instance_march::instance_march_helpers_source;

/// Per-frame uniform — must match `MarchUniforms` in
/// `user_shader_instance_march_main.wgsl`. 32 bytes.
///
/// Tunable step caps (`march_max_steps_outer/brick`) live here rather
/// than in the shader so the host can adjust without recompiling. The
/// defaults match the Stage 5b values so the e2e test produces
/// identical hits.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchUniforms {
    pub tile_index_count: u32,
    pub proto_lookup_count: u32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub march_max_steps_outer: u32,
    pub march_max_steps_brick: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<MarchUniforms>() == 32);

impl Default for MarchUniforms {
    fn default() -> Self {
        Self {
            tile_index_count: 0,
            proto_lookup_count: 0,
            screen_width: 0,
            screen_height: 0,
            march_max_steps_outer: DEFAULT_MAX_STEPS_OUTER,
            march_max_steps_brick: DEFAULT_MAX_STEPS_BRICK,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Default outer-DDA step cap — covers a depth-2 prototype in worst
/// case. Tunable per-frame via `MarchUniforms.march_max_steps_outer`.
pub const DEFAULT_MAX_STEPS_OUTER: u32 = 256;

/// Default per-brick inner-DDA step cap — at most ~12 cells along the
/// longest 4³ diagonal, so 64 is comfortably above the worst case
/// while still bounding pathological GPU loops.
pub const DEFAULT_MAX_STEPS_BRICK: u32 = 64;

/// Camera state for per-pixel ray construction. **Layout is the FIRST
/// 80 BYTES of [`crate::rkp_scene::CameraUniforms`]** — same field
/// order, same offsets — so a Stage 6c renderer integration can bind
/// a slice of the existing camera buffer with no copy or translation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarchCameraUniform {
    pub position: [f32; 4],
    pub forward: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub resolution: [f32; 2],
    pub jitter: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<MarchCameraUniform>() == 80);

/// Per-ray output — the closest instance hit found for this ray. Mirror
/// of `InstanceMarchHit` in the WGSL. 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceMarchHit {
    pub hit: u32,
    pub region_index: u32,
    pub instance_index: u32,
    pub leaf_attr_slot: u32,
    pub t_world: f32,
    pub material_packed: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub normal: [f32; 3],
    pub _pad2: f32,
}

const _: () = assert!(std::mem::size_of::<InstanceMarchHit>() == 48);

/// Source-text composition: helpers + main entry. Exposed so tests can
/// validate the WGSL with naga without going through pipeline creation.
/// Returns the empty-chunk source — the identity-arm dispatch stubs in
/// the template stand in for missing user `inst_to_local` / `inst_aabb`
/// chunks.
pub fn instance_march_main_source() -> String {
    compose_march_main_source("", "")
}

/// Splice the composer's `inst_to_local` and `inst_aabb` chunks into
/// the march template between their respective BEGIN/END markers. An
/// empty chunk leaves the template's identity-arm stub in place.
pub fn compose_march_main_source(
    inst_to_local_chunk: &str,
    inst_aabb_chunk: &str,
) -> String {
    let template = include_str!("shaders/user_shader_instance_march_main.wgsl");
    // Marker strings constructed via concat so the literal occurrences
    // in this docstring don't fool the splicer.
    let template = splice_marker(
        template,
        concat!("USER_INST_TO_LOCAL_DISPATCH", "_BEGIN"),
        concat!("USER_INST_TO_LOCAL_DISPATCH", "_END"),
        inst_to_local_chunk,
    );
    let template = splice_marker(
        &template,
        concat!("USER_INST_AABB_DISPATCH", "_BEGIN"),
        concat!("USER_INST_AABB_DISPATCH", "_END"),
        inst_aabb_chunk,
    );
    format!("{}\n{}", instance_march_helpers_source(), template)
}

fn splice_marker(template: &str, begin: &str, end: &str, chunk: &str) -> String {
    if chunk.is_empty() {
        return template.to_string();
    }
    let begin_idx = template
        .find(begin)
        .unwrap_or_else(|| panic!("march template missing {begin} marker"));
    let end_idx = template[begin_idx..]
        .find(end)
        .map(|off| begin_idx + off + end.len())
        .unwrap_or_else(|| panic!("march template missing {end} marker"));
    let mut out = String::with_capacity(template.len() + chunk.len());
    out.push_str(&template[..begin_idx]);
    out.push_str(chunk);
    out.push_str(&template[end_idx..]);
    out
}

/// Pipeline owner. Construction validates the composed WGSL with naga
/// (panics on any regression) and creates four bind-group layouts +
/// the compute pipeline.
pub struct InstanceMarchPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub group1_layout: wgpu::BindGroupLayout,
    pub group2_layout: wgpu::BindGroupLayout,
    pub group3_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::ComputePipeline,
    /// Hash of the user-shader source mix the pipeline was last built
    /// against. Comparing to the registry's `source_hash` decides
    /// whether the pipeline needs rebuilding.
    pub source_hash: u64,
}

impl InstanceMarchPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group0 (pools)"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2)],
        });
        let group1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group1 (per-frame instance state)"),
            entries: &[
                // regions_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<EmitRegionUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                // instance_pool
                ro_storage(1),
                // tile_index_buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<GpuTileIndexEntry>() as u64,
                        ),
                    },
                    count: None,
                },
                // instance_alloc (read-only here — written by emit pass)
                ro_storage(3),
            ],
        });
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group2 (proto lookup)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        std::mem::size_of::<GpuPrototypeEntry>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let group3_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_march group3 (uniforms + camera + output)"),
            entries: &[
                // march_uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                // camera (MarchCameraUniform — bind a slice of the renderer's
                // CameraUniforms buffer; first 80 B are layout-compatible)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                // output_hits — `array<InstanceMarchHit>` of length
                // `screen_width * screen_height`
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<InstanceMarchHit>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instance_march pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&group1_layout),
                Some(&group2_layout),
                Some(&group3_layout),
            ],
            immediate_size: 0,
        });

        let source = instance_march_main_source();
        crate::validate_wgsl(&source, "instance_march_main");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance_march_main"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_march_main"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("instance_march_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            group0_layout,
            group1_layout,
            group2_layout,
            group3_layout,
            pipeline_layout,
            pipeline,
            source_hash: 0,
        }
    }

    /// Re-build the compute pipeline against the spliced inst_to_local
    /// + inst_aabb chunks. Returns `true` if rebuilt, `false` if the
    /// hash matched and the existing pipeline was kept. Mirrors
    /// `PrototypeBakePass::reload_user_shaders`.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        inst_to_local_chunk: &str,
        inst_aabb_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        let source = compose_march_main_source(inst_to_local_chunk, inst_aabb_chunk);
        crate::validate_wgsl(&source, "instance_march_main");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance_march_main"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_march_main"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("instance_march_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Workgroup count along one axis for `pixels`: `ceil(pixels / 8)`.
    /// Workgroup_size is (8, 8, 1) so each thread covers one pixel.
    pub fn workgroup_count_for_pixels(pixels: u32) -> u32 {
        pixels.div_ceil(8)
    }

    /// Encode the per-pixel dispatch into an open compute pass. The
    /// caller is responsible for setting all four bind groups (`group0`
    /// pools, `group1` instance state, `group2` proto-lookup, `group3`
    /// uniforms+camera+output) before calling this.
    ///
    /// `screen_width` and `screen_height` MUST match the values written
    /// into the `MarchUniforms` bound at `group3 binding 0`. The shader
    /// bounds-checks each thread's `pixel.xy` against the uniform's
    /// values; mismatched dispatch shape vs uniform sizing would still
    /// be safe but waste threads.
    pub fn dispatch_per_pixel(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        screen_width: u32,
        screen_height: u32,
    ) {
        cpass.set_pipeline(&self.pipeline);
        cpass.dispatch_workgroups(
            Self::workgroup_count_for_pixels(screen_width),
            Self::workgroup_count_for_pixels(screen_height),
            1,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn march_uniforms_layout() {
        let u = MarchUniforms {
            tile_index_count: 1,
            proto_lookup_count: 2,
            screen_width: 1920,
            screen_height: 1080,
            march_max_steps_outer: 256,
            march_max_steps_brick: 64,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&u);
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(words, &[1, 2, 1920, 1080, 256, 64, 0, 0]);
    }

    #[test]
    fn march_uniforms_default_loads_step_caps() {
        let u = MarchUniforms::default();
        assert_eq!(u.march_max_steps_outer, DEFAULT_MAX_STEPS_OUTER);
        assert_eq!(u.march_max_steps_brick, DEFAULT_MAX_STEPS_BRICK);
        assert_eq!(u.tile_index_count, 0);
        assert_eq!(u.screen_width, 0);
    }

    #[test]
    fn camera_uniform_is_80_bytes_prefix_compatible() {
        // The struct must be exactly 80 B so it slices off the front
        // of `rkp_scene::CameraUniforms` cleanly. If this assert fails,
        // Stage 6c's "bind a slice of the renderer camera buffer"
        // optimisation breaks silently.
        assert_eq!(std::mem::size_of::<MarchCameraUniform>(), 80);
        let cam = MarchCameraUniform {
            position: [1.0, 2.0, 3.0, 1.0],
            forward: [4.0, 5.0, 6.0, 0.0],
            right: [7.0, 8.0, 9.0, 0.0],
            up: [10.0, 11.0, 12.0, 0.0],
            resolution: [1920.0, 1080.0],
            jitter: [0.25, -0.25],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&cam);
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(floats[0..4], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(floats[4..8], [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(floats[16..18], [1920.0, 1080.0]);
        assert_eq!(floats[18..20], [0.25, -0.25]);
    }

    #[test]
    fn workgroup_count_rounds_up() {
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(0), 0);
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(1), 1);
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(8), 1);
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(9), 2);
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(1920), 240);
        assert_eq!(InstanceMarchPass::workgroup_count_for_pixels(1921), 241);
    }

    #[test]
    fn instance_march_hit_layout_normal_at_offset_32() {
        // Verify the InstanceMarchHit struct lays out so `normal` lands
        // at byte offset 32 (matching the WGSL struct's vec3<f32>
        // alignment-driven slot). Catches future drift.
        let hit = InstanceMarchHit {
            hit: 1,
            region_index: 2,
            instance_index: 3,
            leaf_attr_slot: 4,
            t_world: 5.0,
            material_packed: 6,
            _pad0: 0,
            _pad1: 0,
            normal: [7.0, 8.0, 9.0],
            _pad2: 0.0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&hit);
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(words[0..6], [1, 2, 3, 4, f32::to_bits(5.0), 6]);
        let normal_floats: [f32; 3] = [
            f32::from_bits(words[8]),
            f32::from_bits(words[9]),
            f32::from_bits(words[10]),
        ];
        assert_eq!(normal_floats, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn march_main_wgsl_validates() {
        let source = instance_march_main_source();
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
            panic!("parse error:\n{}", e.emit_to_string(&source))
        });
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}
