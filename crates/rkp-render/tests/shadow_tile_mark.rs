//! Phase 7d Session 1 — wgpu integration test for the shadow-tile
//! mark compute pass.
//!
//! Builds a synthetic set of `TlasPrim`s with known world AABBs,
//! runs `mark_main` on a real device, reads back the bitmap, and
//! asserts it matches the CPU reference exactly. Multiple prims
//! exercise the atomic `OR` (multi-prim coverage of overlapping
//! tiles).
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::shadow_tile_cull_pass::{
    cpu_reference_mark, fit_tile_grid, light_space_basis, ShadowTileCullPass,
    ShadowTileUniform, SHADOW_TILE_BITMAP_U32S, SHADOW_TILE_GRID_H, SHADOW_TILE_GRID_W,
};
use rkp_render::tlas_build_pass::TlasPrim;

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
        label: Some("shadow_tile_mark test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_prim(min: [f32; 3], max: [f32; 3]) -> TlasPrim {
    TlasPrim {
        aabb_min: min,
        asset_id: 0,
        aabb_max: max,
        instance_state_offset: 0,
        material_id: 0,
        instance_index: 0,
        _pad0: 0,
        _pad1: 0,
    }
}

fn run_mark(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    prims: &[TlasPrim],
    uniform: ShadowTileUniform,
) -> Vec<u32> {
    let pass = ShadowTileCullPass::new(device);

    let prims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test tlas_prims"),
        size: (prims.len() as u64).max(1) * (std::mem::size_of::<TlasPrim>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !prims.is_empty() {
        queue.write_buffer(&prims_buffer, 0, bytemuck::cast_slice(prims));
    }
    queue.write_buffer(&pass.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    let zeros: Vec<u32> = vec![0u32; SHADOW_TILE_BITMAP_U32S as usize];
    queue.write_buffer(&pass.bitmap_buffer, 0, bytemuck::cast_slice(&zeros));

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mark g0"),
        layout: &pass.g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.bitmap_buffer.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mark g1"),
        layout: &pass.g1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.uniform_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mark enc"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mark_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.mark_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        let wgs = ((prims.len() as u32 + 63) / 64).max(1);
        cpass.dispatch_workgroups(wgs, 1, 1);
    }
    let bitmap_bytes = (SHADOW_TILE_BITMAP_U32S as u64) * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitmap readback"),
        size: bitmap_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.bitmap_buffer, 0, &readback, 0, bitmap_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let s = readback.slice(..);
    s.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let v = s.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&v).to_vec();
    drop(v);
    readback.unmap();
    result
}

fn make_uniform(
    prim_count: u32,
    scene_min: [f32; 3],
    scene_max: [f32; 3],
    light_dir: [f32; 3],
) -> ShadowTileUniform {
    let (right, up) = light_space_basis(light_dir);
    let (tile_size, origin) = fit_tile_grid(scene_min, scene_max, right, up);
    ShadowTileUniform {
        light_origin: origin,
        tile_size,
        light_right: right,
        grid_w: SHADOW_TILE_GRID_W,
        light_up: up,
        grid_h: SHADOW_TILE_GRID_H,
        prim_count,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

#[test]
fn single_prim_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[shadow_tile_mark] no wgpu adapter — skipping");
        return;
    };
    let prims = vec![make_prim([4.9, 0.4, 4.9], [5.1, 0.6, 5.1])];
    let uniform = make_uniform(prims.len() as u32, [0.0; 3], [10.0, 1.0, 10.0], [0.0, -1.0, 0.0]);
    let gpu = run_mark(&device, &queue, &prims, uniform);
    let cpu = cpu_reference_mark(&prims, &uniform);
    assert_eq!(gpu, cpu, "single-prim bitmap mismatch");
    let total: u32 = gpu.iter().map(|w| w.count_ones()).sum();
    assert!(total > 0, "bitmap should have at least one bit set");
}

#[test]
fn multi_prim_with_overlap_matches_cpu() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    // Three prims, two of which overlap on the same tiles.
    let prims = vec![
        make_prim([0.0, 0.0, 0.0], [0.5, 0.1, 0.5]),
        make_prim([0.3, 0.0, 0.3], [0.8, 0.1, 0.8]),    // overlaps with #0
        make_prim([8.0, 0.0, 8.0], [9.0, 0.5, 9.0]),    // disjoint
    ];
    let uniform = make_uniform(prims.len() as u32, [0.0; 3], [10.0, 1.0, 10.0], [0.3, -1.0, 0.5]);
    let gpu = run_mark(&device, &queue, &prims, uniform);
    let cpu = cpu_reference_mark(&prims, &uniform);
    assert_eq!(gpu, cpu, "multi-prim bitmap mismatch");
}

#[test]
fn empty_input_leaves_bitmap_zero() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    let uniform = make_uniform(0, [0.0; 3], [10.0; 3], [0.0, -1.0, 0.0]);
    let gpu = run_mark(&device, &queue, &[], uniform);
    assert!(gpu.iter().all(|w| *w == 0), "bitmap should be all zero with no prims");
}

#[test]
fn many_prims_atomic_or_correctness() {
    // Stress test the atomic-or pattern: 200 prims clustered in
    // the same tile range. Result must match CPU reference (every
    // covered tile bit set).
    let Some((device, queue)) = create_device() else {
        return;
    };
    let mut prims = Vec::with_capacity(200);
    for i in 0..200u32 {
        let x = 5.0 + (i as f32 * 0.001);  // tiny step — all in same tile
        prims.push(make_prim([x, 0.4, 5.0], [x + 0.05, 0.5, 5.05]));
    }
    let uniform = make_uniform(prims.len() as u32, [0.0; 3], [10.0, 1.0, 10.0], [0.0, -1.0, 0.0]);
    let gpu = run_mark(&device, &queue, &prims, uniform);
    let cpu = cpu_reference_mark(&prims, &uniform);
    assert_eq!(gpu, cpu, "200-prim atomic-or bitmap mismatch");
}
