//! Stage 5a end-to-end tests for the WGSL helper library
//! (`user_shader_instance_march.wgsl`).
//!
//! Each test:
//!   1. Brings up a wgpu device (skips silently when no adapter).
//!   2. Optionally bakes a sphere prototype (for the descent test).
//!   3. Builds the [`InstanceMarchTestPass`] (which validates the WGSL +
//!      compiles three test compute pipelines).
//!   4. Uploads inputs to the per-helper uniform, dispatches one
//!      workgroup, reads back the result, asserts against a CPU-side
//!      expected value.

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_instance_march::{
    AabbTestInputs, AabbTestResult, InstanceMarchTestPass, ProtoDescendTestInputs,
    ProtoDescendTestResult, WorldToLocalTestInputs, WorldToLocalTestResult,
};
use rkp_render::user_shader_proto_pass::{
    build_internal_levels, PrototypeBakePass, PrototypeCache, PrototypeUniform,
};

const PROTO_MAX_DEPTH: u32 = 2;
const POOL_OCTREE_BASE: u32 = 0;
const POOL_BRICK_BASE: u32 = 0;
const POOL_LEAF_ATTR_BASE: u32 = 0;
const SHADER_ID: u32 = 1;
const SOURCE_HASH: u64 = 0xC0FFEE_DEADBEEF;

const SPHERE_SHADER: &str = r#"
// @instance_proto Pt
struct Pt { pos: vec3<f32> }

fn user_sphere_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    let c = vec3<f32>(0.5);
    let r: f32 = 0.4;
    let d = length(uvw - c);
    if (d < r) {
        v.occupancy = 1u;
        v.normal = normalize(uvw - c);
        v.material_primary = 1u;
        v.material_secondary = 0u;
        v.blend_weight = 0u;
    } else {
        v.occupancy = 0u;
    }
    return v;
}

fn user_sphere_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
}
"#;

fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("user_shader_instance_march test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            max_storage_buffers_per_shader_stage: 16,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn write_temp_shader(name: &str, contents: &str) -> std::path::PathBuf {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!(
        "rkp_user_shader_instance_march_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.wgsl"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    dir
}

/// Spin up a wgpu device + the test pass. Returns `None` when no
/// adapter is available so the suite keeps green in headless CI.
fn setup_pass() -> Option<(wgpu::Device, wgpu::Queue, InstanceMarchTestPass)> {
    let (device, queue) = create_device()?;
    let pass = InstanceMarchTestPass::new(&device);
    Some((device, queue, pass))
}

/// Allocate a uniform buffer + storage result buffer for one helper-test
/// dispatch and bind them at group 1/2/3 (per the WGSL).
fn make_io_pair(
    device: &wgpu::Device,
    label: &str,
    uniform_size: u64,
    result_size: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let u = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} uniform")),
        size: uniform_size.max(16),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let r = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} result")),
        size: result_size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (u, r)
}

/// Allocate and zero-init the three pool storage buffers + bind them
/// at group 0. Sized large enough for a depth-2 sphere prototype.
fn make_pool_buffers(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let octree = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inst_march pool octree"),
        size: 8 * 1024,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bricks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inst_march pool bricks"),
        // 1024 bricks × 64 cells × 4 bytes = 256 KB. Plenty for depth-2.
        size: 256 * 1024,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaf_attrs = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("inst_march pool leaf_attrs"),
        // 8192 leaf-attrs × 8 bytes = 64 KB.
        size: 64 * 1024,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (octree, bricks, leaf_attrs)
}

fn read_back<T: bytemuck::Pod + Default>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> T {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_back staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("read_back encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let value: T = *bytemuck::from_bytes(&view[..size as usize]);
    drop(view);
    staging.unmap();
    value
}

#[test]
fn aabb_intersect_ray_through_unit_cube() {
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[aabb_intersect] no wgpu adapter — skipping");
        return;
    };

    // Ray at +X axis through unit cube [0,0,0]-[1,1,1] from x=-1.
    let ro = [-1.0_f32, 0.5, 0.5];
    let rd = [1.0_f32, 0.0, 0.0];
    // CPU-side inv_dir with same 1e-10 nudge convention as octree_march.
    let inv_dir = [1.0 / 1.0_f32, f32::INFINITY, f32::INFINITY];
    // Use a finite stand-in for ±inf that matches the WGSL pattern.
    let inv_dir = [
        inv_dir[0],
        if inv_dir[1].is_infinite() { 1.0 / 1e-10 } else { inv_dir[1] },
        if inv_dir[2].is_infinite() { 1.0 / 1e-10 } else { inv_dir[2] },
    ];
    let inputs = AabbTestInputs {
        ro, _pad0: 0.0,
        rd, _pad1: 0.0,
        inv_dir, _pad2: 0.0,
        aabb_min: [0.0, 0.0, 0.0], _pad3: 0.0,
        aabb_max: [1.0, 1.0, 1.0], _pad4: 0.0,
    };
    let (uniform, result) = make_io_pair(
        &device, "aabb",
        std::mem::size_of::<AabbTestInputs>() as u64,
        std::mem::size_of::<AabbTestResult>() as u64,
    );
    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&inputs));
    queue.write_buffer(
        &result, 0,
        bytemuck::bytes_of(&AabbTestResult::default()),
    );

    let (octree_b, brick_b, leaf_b) = make_pool_buffers(&device);

    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aabb test g0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_b.as_entire_binding() },
        ],
    });
    let group_aabb = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aabb test g1"),
        layout: &pass.aabb_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: result.as_entire_binding() },
        ],
    });
    // Dummy bind groups for the other slots — required because the
    // pipeline layout binds all four. Use any uniform/storage with
    // matching shape; reuse `aabb` IO buffers (read-only on these
    // dispatches anyway).
    let (dummy_u, dummy_r) = make_io_pair(&device, "aabb dummy", 256, 256);
    let group_w2l = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aabb test g2 dummy"),
        layout: &pass.w2l_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let group_proto = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aabb test g3 dummy"),
        layout: &pass.proto_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("aabb test encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aabb_test_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.aabb_pipeline);
        cpass.set_bind_group(0, &group0, &[]);
        cpass.set_bind_group(1, &group_aabb, &[]);
        cpass.set_bind_group(2, &group_w2l, &[]);
        cpass.set_bind_group(3, &group_proto, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let out: AabbTestResult = read_back(
        &device, &queue, &result,
        std::mem::size_of::<AabbTestResult>() as u64,
    );
    // Ray enters at x=0 → t=1, exits at x=1 → t=2.
    assert!((out.t_near - 1.0).abs() < 1e-4, "t_near = {}", out.t_near);
    assert!((out.t_far - 2.0).abs() < 1e-4, "t_far = {}", out.t_far);
}

#[test]
fn aabb_intersect_ray_misses_aabb() {
    // Ray parallel to +X, offset above the cube — should miss (t_near > t_far).
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[aabb miss] no wgpu adapter — skipping");
        return;
    };
    let inv_dir = [1.0_f32, 1.0 / 1e-10, 1.0 / 1e-10];
    let inputs = AabbTestInputs {
        ro: [-1.0, 2.0, 0.5], _pad0: 0.0,
        rd: [1.0, 0.0, 0.0], _pad1: 0.0,
        inv_dir, _pad2: 0.0,
        aabb_min: [0.0, 0.0, 0.0], _pad3: 0.0,
        aabb_max: [1.0, 1.0, 1.0], _pad4: 0.0,
    };
    let out = run_aabb_test(&device, &queue, &pass, &inputs);
    assert!(out.t_near > out.t_far, "expected miss, got t_near={}, t_far={}", out.t_near, out.t_far);
}

fn run_aabb_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &InstanceMarchTestPass,
    inputs: &AabbTestInputs,
) -> AabbTestResult {
    let (uniform, result) = make_io_pair(
        device, "aabb",
        std::mem::size_of::<AabbTestInputs>() as u64,
        std::mem::size_of::<AabbTestResult>() as u64,
    );
    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(inputs));
    queue.write_buffer(
        &result, 0,
        bytemuck::bytes_of(&AabbTestResult::default()),
    );
    let (octree_b, brick_b, leaf_b) = make_pool_buffers(device);
    let (dummy_u, dummy_r) = make_io_pair(device, "dummy", 256, 256);
    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_b.as_entire_binding() },
        ],
    });
    let g_aabb = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.aabb_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: result.as_entire_binding() },
        ],
    });
    let g_w2l = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.w2l_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let g_proto = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.proto_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pass.aabb_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g_aabb, &[]);
        cpass.set_bind_group(2, &g_w2l, &[]);
        cpass.set_bind_group(3, &g_proto, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_back(device, queue, &result, std::mem::size_of::<AabbTestResult>() as u64)
}

#[test]
fn world_to_local_unit_scale_centered() {
    // Instance at world (10, 20, 30) with side 1 — world_pos at center
    // should land at canonical [0.5, 0.5, 0.5].
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[w2l] no wgpu adapter — skipping");
        return;
    };
    let inputs = WorldToLocalTestInputs {
        world_pos: [10.0, 20.0, 30.0],
        instance_scale: 1.0,
        instance_pos: [10.0, 20.0, 30.0],
        _pad0: 0.0,
    };
    let out = run_w2l_test(&device, &queue, &pass, &inputs);
    let expected = [0.5_f32, 0.5, 0.5];
    for (g, e) in out.local.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
    }
}

#[test]
fn world_to_local_scaled_offset() {
    // Instance at world (5, 0, 0) with side 2 — world_pos (4, 0.5, -0.5)
    // should land at canonical
    //   ((4 - 5) / 2 + 0.5, (0.5 - 0) / 2 + 0.5, (-0.5 - 0) / 2 + 0.5)
    //   = (-0.5 + 0.5, 0.25 + 0.5, -0.25 + 0.5)
    //   = (0.0, 0.75, 0.25)
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[w2l offset] no wgpu adapter — skipping");
        return;
    };
    let inputs = WorldToLocalTestInputs {
        world_pos: [4.0, 0.5, -0.5],
        instance_scale: 2.0,
        instance_pos: [5.0, 0.0, 0.0],
        _pad0: 0.0,
    };
    let out = run_w2l_test(&device, &queue, &pass, &inputs);
    let expected = [0.0_f32, 0.75, 0.25];
    for (g, e) in out.local.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
    }
}

fn run_w2l_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &InstanceMarchTestPass,
    inputs: &WorldToLocalTestInputs,
) -> WorldToLocalTestResult {
    let (uniform, result) = make_io_pair(
        device, "w2l",
        std::mem::size_of::<WorldToLocalTestInputs>() as u64,
        std::mem::size_of::<WorldToLocalTestResult>() as u64,
    );
    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(inputs));
    queue.write_buffer(
        &result, 0,
        bytemuck::bytes_of(&WorldToLocalTestResult::default()),
    );
    let (octree_b, brick_b, leaf_b) = make_pool_buffers(device);
    let (dummy_u, dummy_r) = make_io_pair(device, "dummy", 256, 256);
    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_b.as_entire_binding() },
        ],
    });
    let g_aabb = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.aabb_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let g_w2l = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.w2l_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: result.as_entire_binding() },
        ],
    });
    let g_proto = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.proto_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pass.w2l_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g_aabb, &[]);
        cpass.set_bind_group(2, &g_w2l, &[]);
        cpass.set_bind_group(3, &g_proto, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_back(device, queue, &result, std::mem::size_of::<WorldToLocalTestResult>() as u64)
}

/// Bake a sphere prototype into the supplied pool buffers and return
/// the prototype entry / uniform so the caller can configure subsequent
/// dispatches against the live data.
fn bake_sphere_prototype(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    octree_buffer: &wgpu::Buffer,
    brick_buffer: &wgpu::Buffer,
    leaf_attr_buffer: &wgpu::Buffer,
) -> (rkp_render::user_shader_proto_pass::PrototypeEntry, PrototypeUniform) {
    let dir = write_temp_shader("sphere", SPHERE_SHADER);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);

    let mut pass = PrototypeBakePass::new(device);
    pass.reload_user_shaders(device, &chunks.proto, registry.source_hash());

    let mut cache = PrototypeCache::with_capacities(1024, 1024, 8192);
    cache.set_pool_bases(POOL_OCTREE_BASE, POOL_BRICK_BASE, POOL_LEAF_ATTR_BASE);
    let (entry, _dirty) = cache
        .lookup_or_allocate(SHADER_ID, SOURCE_HASH, PROTO_MAX_DEPTH)
        .unwrap();
    let uniform = PrototypeUniform::from_entry(&entry, &cache);

    // Pre-build internal octree levels at the entry's offset.
    let internal = build_internal_levels(POOL_OCTREE_BASE, entry.octree_extent.0, PROTO_MAX_DEPTH);
    let mut octree_init: Vec<u8> = Vec::with_capacity(internal.len() * 8);
    for [v0, v1] in internal {
        octree_init.extend_from_slice(&v0.to_le_bytes());
        octree_init.extend_from_slice(&v1.to_le_bytes());
    }
    queue.write_buffer(octree_buffer, (entry.octree_extent.0 as u64) * 8, &octree_init);

    pass.reset_cursors(queue, 0, 0);
    queue.write_buffer(&pass.overflow_buffer, 0, &[0u8; 12 * 4]);
    queue.write_buffer(&pass.proto_uniform_buffer, 0, bytemuck::bytes_of(&uniform));

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake g0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.cursors_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake g1"),
        layout: &pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.proto_uniform_buffer.as_entire_binding(),
        }],
    });

    let bricks_per_axis = 1u32 << PROTO_MAX_DEPTH;
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pass.bake_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        cpass.dispatch_workgroups(bricks_per_axis, bricks_per_axis, bricks_per_axis);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

    (entry, uniform)
}

#[test]
fn proto_descend_hits_baked_sphere() {
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[proto descend hit] no wgpu adapter — skipping");
        return;
    };
    let (octree_b, brick_b, leaf_b) = make_pool_buffers(&device);
    let (entry, _uniform) =
        bake_sphere_prototype(&device, &queue, &octree_b, &brick_b, &leaf_b);

    // Ray fired straight at canonical (0.5, 0.5, 0.5) along +X. Sphere
    // radius 0.4 → first hit somewhere around uvw.x = 0.1.
    let octree_root = entry.octree_root(POOL_OCTREE_BASE);
    let inputs = ProtoDescendTestInputs {
        local_origin: [-0.5, 0.5, 0.5],
        octree_root,
        local_dir: [1.0, 0.0, 0.0],
        max_depth: PROTO_MAX_DEPTH,
        max_steps_outer: 256,
        max_steps_brick: 64,
        _pad0: 0,
        _pad1: 0,
    };

    let out = run_proto_test(&device, &queue, &pass, &inputs, &octree_b, &brick_b, &leaf_b);
    assert_eq!(out.hit, 1, "expected hit on baked sphere");
    // The ray enters the unit cube at t=0.5 (origin x=-0.5 → enter at x=0).
    // First sphere surface at uvw.x ~ 0.1, so t ~ 0.6.
    assert!(out.t > 0.5 && out.t < 0.95, "t out of plausible range: {}", out.t);
    // Normal should point in roughly -X direction (back toward origin).
    assert!(out.normal[0] < -0.3, "normal.x = {}, expected mostly -X", out.normal[0]);
}

#[test]
fn proto_descend_misses_outside_sphere() {
    let Some((device, queue, pass)) = setup_pass() else {
        eprintln!("[proto descend miss] no wgpu adapter — skipping");
        return;
    };
    let (octree_b, brick_b, leaf_b) = make_pool_buffers(&device);
    let (entry, _uniform) =
        bake_sphere_prototype(&device, &queue, &octree_b, &brick_b, &leaf_b);

    // Ray grazes high above the sphere — y=0.95 puts it outside the
    // r=0.4 sphere centered at 0.5.
    let octree_root = entry.octree_root(POOL_OCTREE_BASE);
    let inputs = ProtoDescendTestInputs {
        local_origin: [-0.5, 0.95, 0.5],
        octree_root,
        local_dir: [1.0, 0.0, 0.0],
        max_depth: PROTO_MAX_DEPTH,
        max_steps_outer: 256,
        max_steps_brick: 64,
        _pad0: 0,
        _pad1: 0,
    };

    let out = run_proto_test(&device, &queue, &pass, &inputs, &octree_b, &brick_b, &leaf_b);
    assert_eq!(out.hit, 0, "expected miss; got hit at t={}", out.t);
}

fn run_proto_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &InstanceMarchTestPass,
    inputs: &ProtoDescendTestInputs,
    octree_b: &wgpu::Buffer,
    brick_b: &wgpu::Buffer,
    leaf_b: &wgpu::Buffer,
) -> ProtoDescendTestResult {
    let (uniform, result) = make_io_pair(
        device, "proto",
        std::mem::size_of::<ProtoDescendTestInputs>() as u64,
        std::mem::size_of::<ProtoDescendTestResult>() as u64,
    );
    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(inputs));
    queue.write_buffer(
        &result, 0,
        bytemuck::bytes_of(&ProtoDescendTestResult::default()),
    );
    let (dummy_u, dummy_r) = make_io_pair(device, "dummy", 256, 256);
    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_b.as_entire_binding() },
        ],
    });
    let g_aabb = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.aabb_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let g_w2l = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.w2l_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: dummy_u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy_r.as_entire_binding() },
        ],
    });
    let g_proto = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pass.proto_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: result.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&pass.proto_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g_aabb, &[]);
        cpass.set_bind_group(2, &g_w2l, &[]);
        cpass.set_bind_group(3, &g_proto, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_back(device, queue, &result, std::mem::size_of::<ProtoDescendTestResult>() as u64)
}
