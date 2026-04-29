//! End-to-end test for the Option B per-region instance emit
//! (leaf-driven dispatch).
//!
//! Builds a synthetic `Vec<PaintedLeaf>` with N records, runs the emit
//! shader on a real wgpu device, and verifies the GPU emitted exactly
//! N instances and that each instance's `pos` matches one of the leaf
//! positions.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_emit_pass::{
    build_emit_region_uniform, workgroups_for_leaf_count, EmitDispatchUniform, EmitPass,
    InstanceRegionCache, InstanceRegionRequest, PaintedLeaf, HOST_NO_HOST_SENTINEL, NO_TILE,
};

const SHADER_NAME: &str = "scatter";
const STRIDE_U32: u32 = 8; // 32-byte instance struct
const MAX_INSTANCES: u32 = 4096;
const POOL_BASE: u32 = 0;
const REGION_INDEX: u32 = 0;
const LEAF_COUNT: u32 = 64;

/// Free-standing instance shader. No host gate — `emit_instance` is
/// always called, once per leaf. Each blade gets its host_pos as `pos`,
/// host_pos.x as yaw, sway from ctx.time, and a fixed height/tint so
/// the readback can verify offsets exactly.
const SCATTER_SHADER: &str = r#"
// @instance_proto Blade
struct Blade {
    pos: vec3<f32>,
    yaw: f32,
    sway_phase: f32,
    height_scale: f32,
    tint: u32,
}
fn user_scatter_proto(uvw: vec3<f32>) -> VoxelEmit {
    var v: VoxelEmit;
    return v;
}
fn user_scatter_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var b: Blade;
    b.pos = host_pos;
    b.yaw = host_pos.x;
    b.sway_phase = ctx.time;
    b.height_scale = 1.5;
    b.tint = 0xC0FFEEu;
    emit_instance(b);
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
        label: Some("user_shader_emit test device"),
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
        "rkp_user_shader_emit_test_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.wgsl"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    dir
}

fn synth_leaves(n: u32) -> Vec<PaintedLeaf> {
    (0..n)
        .map(|i| PaintedLeaf {
            world_pos: [(i as f32) * 0.01, 0.5, 0.25],
            material_packed: 5,
            world_normal: [0.0, 1.0, 0.0],
            _pad: 0.0,
        })
        .collect()
}

#[test]
fn scatter_shader_emits_one_instance_per_leaf() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[scatter_emit] no wgpu adapter — skipping");
        return;
    };

    let dir = write_temp_shader(SHADER_NAME, SCATTER_SHADER);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);
    assert!(chunks.emit.contains("rkp_user_1_emit_instance"));
    let mut pass = EmitPass::new(&device);
    pass.reload_user_shaders(&device, &chunks.emit, registry.source_hash());

    let leaves = synth_leaves(LEAF_COUNT);
    let request = InstanceRegionRequest {
        host_object_id: 1,
        material_id: 5,
        shader_name: SHADER_NAME.to_string(),
        params: vec![],
        aabb_min: [0.0; 3],
        aabb_max: [1.0; 3],
        cell_size: 0.04,
        input_hash: 0,
        animated: false,
        region_thickness: 0.0,
        tile_index: NO_TILE,
        stride_u32: STRIDE_U32,
        max_instances: MAX_INSTANCES,
        host_octree_root: HOST_NO_HOST_SENTINEL,
        host_octree_depth: 0,
        host_octree_extent: 0.0,
        host_grid_origin: [0.0; 3],
        host_inverse_world: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        leaves: leaves.clone(),
    };

    let mut cache = InstanceRegionCache::with_capacity(MAX_INSTANCES * STRIDE_U32 * 2);
    cache.set_pool_base(POOL_BASE);
    let slot = cache.lookup_or_allocate(&request, 0xAA, 0xBB).unwrap();

    let instance_pool_bytes =
        ((slot.instance_block_offset + slot.instance_extent_u32) as u64 + 4) * 4;
    let instance_pool = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test instance_pool"),
        size: instance_pool_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test leaves"),
        size: (leaves.len() as u64) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&leaves_buffer, 0, bytemuck::cast_slice(&leaves));

    queue.write_buffer(&pass.instance_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&pass.overflow_buffer, 0, &[0u8; 16]);

    let region_uniform =
        build_emit_region_uniform(&request, &slot, /* shader_id */ 1, /* time */ 0.0, 0);
    queue.write_buffer(&pass.regions_buffer, 0, bytemuck::bytes_of(&region_uniform));

    let dispatch_u = EmitDispatchUniform {
        region_index: REGION_INDEX,
        leaf_count: LEAF_COUNT,
        _pad0: 0,
        _pad1: 0,
    };
    queue.write_buffer(
        &pass.dispatch_uniforms_buffer,
        0,
        bytemuck::bytes_of(&dispatch_u),
    );

    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test emit group0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.instance_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaves_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test emit group1"),
        layout: &pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.regions_buffer.as_entire_binding(),
        }],
    });
    let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test emit group2"),
        layout: &pass.group2_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &pass.dispatch_uniforms_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<EmitDispatchUniform>() as u64),
            }),
        }],
    });

    let wgs = workgroups_for_leaf_count(LEAF_COUNT);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test emit encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("emit_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.emit_pipeline);
        cpass.set_bind_group(0, &group0, &[]);
        cpass.set_bind_group(1, &group1, &[]);
        cpass.set_bind_group(2, &group2, &[0u32]);
        cpass.dispatch_workgroups(wgs, 1, 1);
    }

    let alloc_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alloc readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &pass.instance_alloc_buffer,
        0,
        &alloc_readback,
        0,
        4,
    );
    let pool_readback_bytes = (slot.instance_extent_u32 as u64) * 4;
    let pool_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool readback"),
        size: pool_readback_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &instance_pool,
        (slot.instance_block_offset as u64) * 4,
        &pool_readback,
        0,
        pool_readback_bytes,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let slice = alloc_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let gpu_count = u32::from_le_bytes(view[0..4].try_into().unwrap());
    drop(view);
    alloc_readback.unmap();

    assert_eq!(
        gpu_count, LEAF_COUNT,
        "GPU emitted {gpu_count} instances, expected {LEAF_COUNT} (one per leaf)",
    );

    let slice = pool_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&view);
    // Each emitted instance's pos must equal one of the synthetic
    // leaves. Collect emitted positions into a set and check coverage.
    let mut emitted_xs: Vec<f32> = Vec::with_capacity(gpu_count as usize);
    for inst in 0..gpu_count as usize {
        let base = inst * STRIDE_U32 as usize;
        let pos_x = f32::from_bits(words[base]);
        let pos_y = f32::from_bits(words[base + 1]);
        let pos_z = f32::from_bits(words[base + 2]);
        let yaw = f32::from_bits(words[base + 3]);
        let sway = f32::from_bits(words[base + 4]);
        let height = f32::from_bits(words[base + 5]);
        let tint = words[base + 6];
        // pos.y / pos.z constants must match the synthetic leaves.
        assert!((pos_y - 0.5).abs() < 1e-5);
        assert!((pos_z - 0.25).abs() < 1e-5);
        // yaw was set to host_pos.x.
        assert!((yaw - pos_x).abs() < 1e-5);
        assert_eq!(sway, 0.0);
        assert!((height - 1.5).abs() < 1e-5);
        assert_eq!(tint, 0xC0FFEE);
        emitted_xs.push(pos_x);
    }
    emitted_xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (i, &x) in emitted_xs.iter().enumerate() {
        let expected = (i as f32) * 0.01;
        assert!(
            (x - expected).abs() < 1e-5,
            "sorted instance {i} pos.x = {x}, expected {expected}",
        );
    }
    drop(view);
    pool_readback.unmap();
}
