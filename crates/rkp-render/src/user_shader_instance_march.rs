//! Stage 5a — pipeline owner for the WGSL helper library
//! ([`crate::shaders::user_shader_instance_march`]). Stage 5b will
//! consume `inst_ray_aabb_intersect`, `inst_world_to_local`, and
//! `inst_proto_descend` from a real `instance_march_main` compute
//! entry; today we expose the helper module's source plus three
//! standalone test compute pipelines (one per helper) so each can be
//! validated against deterministic inputs on a real GPU.
//!
//! ## Why a separate file from the WGSL
//!
//! The helpers are pure functions, but they read three module-scope
//! storage buffers (`octree_nodes`, `brick_pool`, `leaf_attr_pool`) that
//! the prototype bake's pool layout owns. The Stage 5b march will
//! re-bind the same buffers in its own pipeline; this test pipeline
//! mirrors that wiring so the bind-group shape is what Stage 5b will
//! inherit.
//!
//! ## Test pipelines
//!
//! [`InstanceMarchTestPass`] owns three compute pipelines —
//! `aabb_test_main`, `world_to_local_test_main`, `proto_descend_test_main` —
//! plus their shared bind-group layouts. The integration test in
//! `tests/user_shader_instance_march.rs` constructs one, fills the
//! input uniforms, dispatches a single workgroup per helper, and reads
//! back the result buffer.

/// Source text of the helper library — pure functions + pool bindings.
/// Stage 5b's `instance_march_main` pipeline composes the same chunk
/// ahead of its own `@compute` entry.
pub fn instance_march_helpers_source() -> &'static str {
    include_str!("shaders/user_shader_instance_march_helpers.wgsl")
}

/// Source text of Stage 5a's test entries (three standalone compute
/// kernels — one per helper). Concatenate after
/// [`instance_march_helpers_source`] to build the full pipeline source.
pub fn instance_march_test_source() -> &'static str {
    include_str!("shaders/user_shader_instance_march_test.wgsl")
}

/// Combined helpers + Stage 5a test source — what the test pipeline
/// in this module compiles. Single point of truth for "what the
/// validator should parse" so the inline naga test below stays in
/// sync with what the real device pipeline sees.
pub fn instance_march_wgsl_source() -> String {
    format!(
        "{}\n{}",
        instance_march_helpers_source(),
        instance_march_test_source(),
    )
}

/// Stride between consecutive uniform records when binding multiple
/// test inputs from the same buffer with dynamic offsets. Matches the
/// 256-byte uniform alignment WGPU requires.
pub const TEST_UNIFORM_STRIDE: u64 = 256;

/// Bind-group layouts + compute pipelines for the three helper-test
/// entry points.
pub struct InstanceMarchTestPass {
    pub group0_layout: wgpu::BindGroupLayout,
    pub aabb_group_layout: wgpu::BindGroupLayout,
    pub w2l_group_layout: wgpu::BindGroupLayout,
    pub proto_group_layout: wgpu::BindGroupLayout,
    pub aabb_pipeline: wgpu::ComputePipeline,
    pub w2l_pipeline: wgpu::ComputePipeline,
    pub proto_pipeline: wgpu::ComputePipeline,
}

impl InstanceMarchTestPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let group0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("user_shader_instance_march group0 (pool buffers)"),
            entries: &[
                ro_storage(0), // octree_nodes
                ro_storage(1), // brick_pool
                ro_storage(2), // leaf_attr_pool
            ],
        });
        let aabb_group_layout = test_io_layout(device, "aabb test io");
        let w2l_group_layout = test_io_layout(device, "w2l test io");
        let proto_group_layout = test_io_layout(device, "proto test io");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("user_shader_instance_march pipeline layout"),
            bind_group_layouts: &[
                Some(&group0_layout),
                Some(&aabb_group_layout),
                Some(&w2l_group_layout),
                Some(&proto_group_layout),
            ],
            immediate_size: 0,
        });

        let source = instance_march_wgsl_source();
        crate::validate_wgsl(&source, "user_shader_instance_march");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("user_shader_instance_march"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inst aabb test"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("aabb_test_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let w2l_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inst w2l test"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("world_to_local_test_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let proto_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inst proto descend test"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("proto_descend_test_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            group0_layout,
            aabb_group_layout,
            w2l_group_layout,
            proto_group_layout,
            aabb_pipeline,
            w2l_pipeline,
            proto_pipeline,
        }
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

fn test_io_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

// ── Rust mirrors of the WGSL test-input/output structs ──
//
// Layouts mirror `user_shader_instance_march.wgsl` exactly. The WGSL
// uses `vec3<f32>` (size 12, alignment 16); the Rust mirrors include
// the explicit `_pad*` slots needed for a faithful copy via bytemuck.

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AabbTestInputs {
    pub ro: [f32; 3], pub _pad0: f32,
    pub rd: [f32; 3], pub _pad1: f32,
    pub inv_dir: [f32; 3], pub _pad2: f32,
    pub aabb_min: [f32; 3], pub _pad3: f32,
    pub aabb_max: [f32; 3], pub _pad4: f32,
}
const _: () = assert!(std::mem::size_of::<AabbTestInputs>() == 80);

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldToLocalTestInputs {
    pub world_pos: [f32; 3], pub instance_scale: f32,
    pub instance_pos: [f32; 3], pub _pad0: f32,
}
const _: () = assert!(std::mem::size_of::<WorldToLocalTestInputs>() == 32);

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProtoDescendTestInputs {
    pub local_origin: [f32; 3], pub octree_root: u32,
    pub local_dir: [f32; 3], pub max_depth: u32,
    pub max_steps_outer: u32, pub max_steps_brick: u32,
    pub _pad0: u32, pub _pad1: u32,
}
const _: () = assert!(std::mem::size_of::<ProtoDescendTestInputs>() == 48);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AabbTestResult {
    pub t_near: f32, pub t_far: f32,
    pub _pad0: f32, pub _pad1: f32,
}
const _: () = assert!(std::mem::size_of::<AabbTestResult>() == 16);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldToLocalTestResult {
    pub local: [f32; 3], pub _pad0: f32,
}
const _: () = assert!(std::mem::size_of::<WorldToLocalTestResult>() == 16);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProtoDescendTestResult {
    pub hit: u32, pub leaf_attr_slot: u32,
    pub material_local: u32, pub _pad0: u32,
    pub t: f32, pub _pad1: f32, pub _pad2: f32, pub _pad3: f32,
    pub normal: [f32; 3], pub _pad4: f32,
}
const _: () = assert!(std::mem::size_of::<ProtoDescendTestResult>() == 48);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_helpers_validate() {
        let source = instance_march_wgsl_source();
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

    #[test]
    fn aabb_test_inputs_layout_matches_wgsl() {
        // WGSL fields in order: ro, _pad0, rd, _pad1, inv_dir, _pad2,
        // aabb_min, _pad3, aabb_max, _pad4. Each pair is 16 B → 80 B.
        let inputs = AabbTestInputs {
            ro: [1.0, 2.0, 3.0], _pad0: 0.0,
            rd: [4.0, 5.0, 6.0], _pad1: 0.0,
            inv_dir: [7.0, 8.0, 9.0], _pad2: 0.0,
            aabb_min: [10.0, 11.0, 12.0], _pad3: 0.0,
            aabb_max: [13.0, 14.0, 15.0], _pad4: 0.0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&inputs);
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(floats[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(floats[4..7], [4.0, 5.0, 6.0]);
        assert_eq!(floats[8..11], [7.0, 8.0, 9.0]);
        assert_eq!(floats[12..15], [10.0, 11.0, 12.0]);
        assert_eq!(floats[16..19], [13.0, 14.0, 15.0]);
    }
}
