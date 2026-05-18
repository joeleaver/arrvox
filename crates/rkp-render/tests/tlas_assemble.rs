//! Phase 7c Session 1 — wgpu integration tests for the TLAS
//! primitive-assembly compute passes.
//!
//! Each test builds a small input set on the CPU, runs both the GPU
//! pipeline (`TlasBuildPass`) and the CPU reference function
//! (`cpu_reference_assemble_*`) on the same input, and asserts the
//! outputs match as multisets — atomic ordering across workgroups
//! is implementation-defined, so the comparison sorts both sides
//! before checking equality.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance};
use rkp_render::tlas_build_pass::{
    cpu_reference_assemble_host,
    AssembleHostUniform, TlasBuildPass, TlasPrim,
};

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
        label: Some("tlas_assemble test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_asset(min: [f32; 3], max: [f32; 3], shader_id: u32) -> RkpGpuAsset {
    RkpGpuAsset {
        aabb_min: min,
        octree_root: 0,
        aabb_max: max,
        octree_depth: 0,
        octree_extent_bits: 0,
        voxel_size: 0.0,
        geom_type: 0,
        bone_count: 0,
        grid_origin: [0.0; 3],
        rest_octree_root: 0,
        rest_octree_depth: 0,
        rest_octree_extent_bits: 0,
        shader_id,
        _pad: 0,
    }
}

fn make_instance(asset_id: u32, world: [[f32; 4]; 4], material: u32) -> RkpGpuInstance {
    RkpGpuInstance {
        world,
        asset_id,
        material_id: material,
        object_id: 0,
        layer_mask: 0xFFFF_FFFF,
        is_skinned: 0,
        bone_buffer_offset: 0,
        overlay_offset: 0,
        overlay_count: 0,
        sculpt_offset: 0,
        sculpt_count: 0,
        _pad: [0; 2],
    }
}

/// Sort prims by a stable key so two multisets compare equal
/// regardless of GPU atomic ordering.
fn sort_prims(prims: &mut Vec<TlasPrim>) {
    prims.sort_by(|a, b| {
        a.instance_index
            .cmp(&b.instance_index)
            .then(a.asset_id.cmp(&b.asset_id))
            .then(a.material_id.cmp(&b.material_id))
            .then(a.aabb_min[0].partial_cmp(&b.aabb_min[0]).unwrap_or(std::cmp::Ordering::Equal))
    });
}

fn assert_prim_eq(a: &TlasPrim, b: &TlasPrim) {
    assert_eq!(a.aabb_min, b.aabb_min, "aabb_min mismatch");
    assert_eq!(a.aabb_max, b.aabb_max, "aabb_max mismatch");
    assert_eq!(a.asset_id, b.asset_id, "asset_id mismatch");
    assert_eq!(a.material_id, b.material_id, "material_id mismatch");
    assert_eq!(a.instance_state_offset, b.instance_state_offset, "instance_state_offset mismatch");
    assert_eq!(a.instance_index, b.instance_index, "instance_index mismatch");
}

#[test]
fn assemble_host_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[tlas_assemble] no wgpu adapter — skipping");
        return;
    };

    // Mix of host + user-shader assets; only host should produce prims.
    let assets = vec![
        make_asset([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0), // host
        make_asset([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 7), // user-shader
        make_asset([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], 0), // host, larger
    ];
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let translated_10_20_30 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [10.0, 20.0, 30.0, 1.0],
    ];
    let instances = vec![
        make_instance(0, identity, 100),                  // → prim
        make_instance(1, identity, 200),                  // user-shader, skip
        make_instance(2, translated_10_20_30, 300),       // → prim, translated
        make_instance(0, translated_10_20_30, 400),       // → prim, translated unit-cube
    ];
    let prims_capacity = 16u32;

    let mut pass = TlasBuildPass::new(&device);
    pass.ensure_prims_capacity(&device, prims_capacity);

    let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test host_instances"),
        size: (instances.len() as u64) * (std::mem::size_of::<RkpGpuInstance>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let assets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test host_assets"),
        size: (assets.len() as u64) * (std::mem::size_of::<RkpGpuAsset>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&instances_buffer, 0, bytemuck::cast_slice(&instances));
    queue.write_buffer(&assets_buffer, 0, bytemuck::cast_slice(&assets));

    queue.write_buffer(&pass.tlas_prim_count_buffer, 0, &[0u8; 4]);
    queue.write_buffer(
        &pass.host_uniform_buffer,
        0,
        bytemuck::bytes_of(&AssembleHostUniform {
            instance_count: instances.len() as u32,
            asset_count: assets.len() as u32,
            prims_capacity,
            _pad: 0,
        }),
    );

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test host g0"),
        layout: &pass.host_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: instances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: assets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.tlas_prim_count_buffer.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test host g1"),
        layout: &pass.host_g1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.host_uniform_buffer.as_entire_binding(),
        }],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test enc host") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("assemble_host_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.host_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        let wgs = ((instances.len() as u32) + 63) / 64;
        cpass.dispatch_workgroups(wgs.max(1), 1, 1);
    }
    let count_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("count readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let prims_bytes = (prims_capacity as u64) * (std::mem::size_of::<TlasPrim>() as u64);
    let prims_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prims readback"),
        size: prims_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.tlas_prim_count_buffer, 0, &count_readback, 0, 4);
    encoder.copy_buffer_to_buffer(&pass.tlas_prims_buffer, 0, &prims_readback, 0, prims_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let cs = count_readback.slice(..);
    cs.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let cv = cs.get_mapped_range();
    let gpu_count = u32::from_le_bytes(cv[0..4].try_into().unwrap());
    drop(cv);
    count_readback.unmap();

    let ps = prims_readback.slice(..);
    ps.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let pv = ps.get_mapped_range();
    let gpu_prims_all: &[TlasPrim] = bytemuck::cast_slice(&pv);
    let mut gpu_prims = gpu_prims_all[..(gpu_count as usize)].to_vec();
    drop(pv);
    prims_readback.unmap();

    let (mut cpu_prims, cpu_count) = cpu_reference_assemble_host(&instances, &assets, prims_capacity);
    assert_eq!(gpu_count, cpu_count, "count mismatch (cpu={cpu_count}, gpu={gpu_count})");
    assert_eq!(gpu_count, 3, "expected 3 host prims (one user-shader skipped)");

    sort_prims(&mut gpu_prims);
    sort_prims(&mut cpu_prims);
    for (g, c) in gpu_prims.iter().zip(cpu_prims.iter()) {
        assert_prim_eq(g, c);
    }
}
