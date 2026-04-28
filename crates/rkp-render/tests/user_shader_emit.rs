//! Stage 3 end-to-end test for the Option B per-region instance emit.
//!
//! Runs the emit shader on a real wgpu device with a no-host region.
//! The user's emit body unconditionally calls `emit_instance` once per
//! sample, so the GPU instance count must equal `samples_per_axis³`.
//! Reads back the per-region atomic counter AND a few instance records
//! to verify field writes landed at the right offsets.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_emit_pass::{
    build_emit_region_uniform, samples_per_axis, EmitDispatchUniform, EmitPass,
    InstanceRegionCache, InstanceRegionRequest, HOST_NO_HOST_SENTINEL, NO_TILE,
};

const SHADER_NAME: &str = "scatter";
const STRIDE_U32: u32 = 8; // 32-byte instance struct
const MAX_INSTANCES: u32 = 4096;
const POOL_BASE: u32 = 0;
const REGION_INDEX: u32 = 0;

/// Free-standing instance shader. No host gate — `emit_instance` is
/// always called, once per `emit` invocation. Each blade gets its
/// host_pos as `pos`, an index-derived yaw, sway from ctx.time, and
/// a fixed height/tint so the readback can verify offsets exactly.
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

#[test]
fn scatter_shader_emits_one_instance_per_sample() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[scatter_emit] no wgpu adapter — skipping");
        return;
    };

    // Compose the user-shader chunk for the scatter shader and reload
    // the emit pipeline against it.
    let dir = write_temp_shader(SHADER_NAME, SCATTER_SHADER);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);
    assert!(chunks.emit.contains("rkp_user_1_emit_instance"));
    let mut pass = EmitPass::new(&device);
    pass.reload_user_shaders(&device, &chunks.emit, registry.source_hash());

    // Build a no-host request — `emit_instance` fires unconditionally,
    // so we get exactly one instance per sample.
    let aabb_min = [0.0f32, 0.0, 0.0];
    let aabb_max = [1.0f32, 1.0, 1.0];
    let cell_size = 0.04f32;
    let request = InstanceRegionRequest {
        host_object_id: 1,
        material_id: 5,
        shader_name: SHADER_NAME.to_string(),
        params: vec![],
        aabb_min,
        aabb_max,
        cell_size,
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
    };
    let spa = samples_per_axis(&request);
    let expected_instance_count = spa * spa * spa;
    assert!(
        expected_instance_count <= MAX_INSTANCES,
        "test request would overflow the per-region cap ({expected_instance_count} > {MAX_INSTANCES})",
    );

    let mut cache = InstanceRegionCache::with_capacity(MAX_INSTANCES * STRIDE_U32 * 2);
    cache.set_pool_base(POOL_BASE);
    let slot = cache.lookup_or_allocate(&request, 0xAA, 0xBB).unwrap();

    // Allocate the instance pool buffer locally for this test.
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
    // Stub octree/brick/leaf buffers — host_sample_at returns the
    // no-host sentinel branch (region.host_octree_root == 0xFFFF_FFFF),
    // so these are never read. Allocate something tiny but non-zero.
    let stub = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test stub host buffer"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Reset the per-region atomic counter to 0.
    queue.write_buffer(&pass.instance_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&pass.overflow_buffer, 0, &[0u8; 16]);

    // Upload the region uniform.
    let region_uniform =
        build_emit_region_uniform(&request, &slot, /* shader_id */ 1, /* time */ 0.0);
    queue.write_buffer(&pass.regions_buffer, 0, bytemuck::bytes_of(&region_uniform));

    // Upload the dispatch uniform at offset 0 (region_index 0).
    let dispatch_u = EmitDispatchUniform {
        region_index: REGION_INDEX,
        samples_per_axis: spa,
        _pad0: 0,
        _pad1: 0,
    };
    queue.write_buffer(
        &pass.dispatch_uniforms_buffer,
        0,
        bytemuck::bytes_of(&dispatch_u),
    );

    // Build bind groups.
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test emit group0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.instance_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: stub.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: stub.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: stub.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.overflow_buffer.as_entire_binding() },
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

    // Workgroups per axis = ceil(samples_per_axis / 4).
    let wgs_per_axis = spa.div_ceil(4);
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
        cpass.dispatch_workgroups(wgs_per_axis, wgs_per_axis, wgs_per_axis);
    }

    // Stage the atomic counter + a few instance records for readback.
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
    // Read back enough u32s to cover the full instance buffer extent.
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
        gpu_count, expected_instance_count,
        "GPU emitted {gpu_count} instances, expected {expected_instance_count} (= {spa}³)",
    );

    // Verify a sample of the written records. Each instance is 8 u32s
    // = 32 bytes. The emit order is non-deterministic (atomic-add), so
    // we can't index by sample-coordinate. Instead, scan all instances
    // and check that each one's tint == 0xC0FFEE and height_scale == 1.5
    // (those are constants in the user shader). pos.x must be in [0,1].
    let slice = pool_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&view);
    // The dispatch grid spans `samples_per_axis * bp_cell` per axis,
    // which can exceed the requested extent by up to `bp_cell` (the
    // grid is rounded up to cover the full AABB). Accept positions in
    // [aabb_min, aabb_min + samples_per_axis * bp_cell].
    let bp_cell = cell_size * 4.0;
    let pos_max = aabb_min[0] + (spa as f32) * bp_cell;
    for inst in 0..gpu_count as usize {
        let base = inst * STRIDE_U32 as usize;
        let pos_x = f32::from_bits(words[base]);
        let pos_y = f32::from_bits(words[base + 1]);
        let pos_z = f32::from_bits(words[base + 2]);
        let yaw = f32::from_bits(words[base + 3]);
        let sway = f32::from_bits(words[base + 4]);
        let height = f32::from_bits(words[base + 5]);
        let tint = words[base + 6];
        let in_range =
            |v: f32| v >= aabb_min[0] - 1e-5 && v <= pos_max + 1e-5;
        assert!(in_range(pos_x), "instance {inst} pos.x = {pos_x} out of range");
        assert!(in_range(pos_y), "instance {inst} pos.y = {pos_y} out of range");
        assert!(in_range(pos_z), "instance {inst} pos.z = {pos_z} out of range");
        // yaw was set to host_pos.x — must match.
        assert!(
            (yaw - pos_x).abs() < 1e-5,
            "instance {inst} yaw {yaw} != pos.x {pos_x}",
        );
        assert_eq!(sway, 0.0); // ctx.time was 0.0
        assert!((height - 1.5).abs() < 1e-5);
        assert_eq!(tint, 0xC0FFEE);
    }
    drop(view);
    pool_readback.unmap();
}
