//! Phase 7c Session 2 — wgpu integration test for the
//! Morton-code + 8-bit radix-sort dispatch chain.
//!
//! Bakes a known set of `TlasPrim`s (with deliberately non-sorted
//! centroid positions), runs the full pipeline (Morton compute →
//! 4× radix count/scan/scatter), reads back the sorted (key, val)
//! pairs, and asserts they're sorted ascending. Compares as a
//! multiset against the CPU reference because the GPU's
//! within-bucket atomic order is non-deterministic.
//!
//! Skips silently when no wgpu adapter is available.

use arvx_render::tlas_build_pass::{
    cpu_reference_morton, cpu_reference_radix_sort, scene_aabb_from_prims,
    InstanceTileCullEntry, MortonUniform, RadixUniform, TlasBuildPass, TlasPrim, TlasState,
    RADIX_BUCKETS, RADIX_PASSES, RADIX_WG_SIZE,
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
        label: Some("tlas_morton_sort test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_prim(min: [f32; 3], max: [f32; 3], asset_id: u32) -> TlasPrim {
    TlasPrim {
        aabb_min: min,
        asset_id,
        aabb_max: max,
        instance_state_offset: 0,
        material_id: 0,
        instance_index: asset_id,
        _pad0: 0,
        _pad1: 0,
    }
}

#[test]
fn morton_then_radix_sort_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[tlas_morton_sort] no wgpu adapter — skipping");
        return;
    };

    // Build a deliberately scattered set of primitives — non-sorted
    // centroid positions, varied AABB sizes, repeats to exercise
    // ties.
    let prims: Vec<TlasPrim> = vec![
        make_prim([7.0, 0.0, 0.0], [7.5, 0.5, 0.5], 0),
        make_prim([1.0, 1.0, 1.0], [1.5, 1.5, 1.5], 1),
        make_prim([5.0, 0.0, 0.0], [5.5, 0.5, 0.5], 2),
        make_prim([3.0, 3.0, 3.0], [3.5, 3.5, 3.5], 3),
        make_prim([9.0, 9.0, 9.0], [9.5, 9.5, 9.5], 4),
        make_prim([2.0, 2.0, 2.0], [2.5, 2.5, 2.5], 5),
        make_prim([6.0, 6.0, 6.0], [6.5, 6.5, 6.5], 6),
        make_prim([4.0, 4.0, 4.0], [4.5, 4.5, 4.5], 7),
        make_prim([8.0, 8.0, 8.0], [8.5, 8.5, 8.5], 8),
        make_prim([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], 9),
        // A few clustered centroids that may share Morton buckets:
        make_prim([5.01, 0.01, 0.01], [5.51, 0.51, 0.51], 10),
        make_prim([5.02, 0.02, 0.02], [5.52, 0.52, 0.52], 11),
    ];
    let prim_count = prims.len() as u32;

    let mut pass = TlasBuildPass::new(&device);
    pass.ensure_prims_capacity(&device, prim_count);
    pass.ensure_keys_capacity(&device, prim_count);
    let num_workgroups = ((prim_count + RADIX_WG_SIZE - 1) / RADIX_WG_SIZE).max(1);
    pass.ensure_histogram_capacity(&device, num_workgroups);

    // Upload prims directly (skip the assembly pass — separately
    // tested in tlas_assemble.rs).
    queue.write_buffer(&pass.tlas_prims_buffer, 0, bytemuck::cast_slice(&prims));

    // Tests bypass the dispatch_args pass — write `tlas_state`
    // directly so the shaders see the right per-frame counts.
    queue.write_buffer(
        &pass.tlas_state_buffer,
        0,
        bytemuck::bytes_of(&TlasState {
            prim_count,
            radix_workgroups: num_workgroups,
            internal_wgs: 0,
            total_node_wgs: 0,
        }),
    );

    // ── Morton compute ────────────────────────────────────────────
    let (scene_min, scene_max) = scene_aabb_from_prims(&prims);
    queue.write_buffer(
        &pass.morton_uniform_buffer,
        0,
        bytemuck::bytes_of(&MortonUniform {
            scene_min,
            _pad0: 0,
            scene_max,
            _pad1: 0,
        }),
    );

    let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morton g0"),
        layout: &pass.morton_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.vals_a_buffer.as_entire_binding() },
        ],
    });
    let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("morton g1"),
        layout: &pass.morton_g1_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pass.morton_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pass.tlas_state_buffer.as_entire_binding(),
            },
        ],
    });

    // ── Radix sort uniforms — 4 passes ───────────────────────────
    let radix_uniform_stride: u64 = 256;
    let mut radix_uniform_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_uniform_stride) as usize];
    for p in 0..RADIX_PASSES {
        let u = RadixUniform {
            digit_shift: p * 8,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let off = (p as u64 * radix_uniform_stride) as usize;
        radix_uniform_bytes[off..off + std::mem::size_of::<RadixUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&u));
    }
    queue.write_buffer(&pass.radix_uniform_buffer, 0, &radix_uniform_bytes);

    // Build two ping-pong bind groups for the radix pipeline.
    // pass 0: a → b ; pass 1: b → a ; pass 2: a → b ; pass 3: b → a.
    let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("radix g0 a→b"),
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("radix g0 b→a"),
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("radix g1"),
        layout: &pass.radix_g1_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pass.tlas_state_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("morton+radix encoder"),
    });

    // Morton.
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_morton_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.morton_pipeline);
        cpass.set_bind_group(0, &morton_g0, &[]);
        cpass.set_bind_group(1, &morton_g1, &[]);
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // 4 radix passes.
    let histogram_bytes = (num_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
    for p in 0..RADIX_PASSES {
        let radix_g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
        let dyn_off = (p as u64 * radix_uniform_stride) as u32;
        // Zero the histogram before count (scan_offsets gets fully
        // overwritten by scan_main, no clear needed).
        encoder.clear_buffer(&pass.histogram_buffer, 0, Some(histogram_bytes));
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.radix_count_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.radix_scan_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.radix_scatter_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
    }

    // After 4 (even) passes, sorted output is in keys_a / vals_a.
    let prim_bytes = (prim_count as u64) * 4;
    let keys_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("keys readback"),
        size: prim_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let vals_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vals readback"),
        size: prim_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.keys_a_buffer, 0, &keys_readback, 0, prim_bytes);
    encoder.copy_buffer_to_buffer(&pass.vals_a_buffer, 0, &vals_readback, 0, prim_bytes);

    queue.submit(std::iter::once(encoder.finish()));

    let ks = keys_readback.slice(..);
    ks.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let kv = ks.get_mapped_range();
    let gpu_keys: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&kv).to_vec();
    drop(kv);
    keys_readback.unmap();

    let vs = vals_readback.slice(..);
    vs.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let vv = vs.get_mapped_range();
    let gpu_vals: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&vv).to_vec();
    drop(vv);
    vals_readback.unmap();

    // Sanity: keys ascending.
    for w in gpu_keys.windows(2) {
        assert!(
            w[0] <= w[1],
            "GPU output not sorted: {} -> {}\nfull keys: {:?}",
            w[0],
            w[1],
            gpu_keys,
        );
    }

    // CPU reference: produce (morton, idx) pairs from the same
    // input, sort, compare as multiset.
    let cpu_pairs = cpu_reference_morton(&prims, scene_min, scene_max);
    let cpu_sorted = cpu_reference_radix_sort(&cpu_pairs);
    let mut gpu_sorted: Vec<(u32, u32)> = gpu_keys.iter().copied().zip(gpu_vals.iter().copied()).collect();
    // GPU output is sorted by key but ties may permute. Re-sort
    // by (key, val) so the comparison is order-stable.
    gpu_sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    assert_eq!(gpu_sorted, cpu_sorted, "GPU sort doesn't match CPU multiset");

    // Sanity: every prim_idx in [0, prim_count) appears exactly
    // once in the val output (no duplicates, no drops).
    let mut seen = vec![false; prim_count as usize];
    for v in &gpu_vals {
        let idx = *v as usize;
        assert!(idx < seen.len(), "vals output index {idx} out of range");
        assert!(!seen[idx], "vals output has duplicate prim_idx {idx}");
        seen[idx] = true;
    }
    assert!(seen.iter().all(|s| *s), "vals output missing prim_idx");
}

#[test]
fn radix_handles_single_prim() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    let prims = vec![make_prim([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0)];
    let prim_count = prims.len() as u32;

    let mut pass = TlasBuildPass::new(&device);
    pass.ensure_prims_capacity(&device, prim_count);
    pass.ensure_keys_capacity(&device, prim_count);
    let num_workgroups = 1u32;
    pass.ensure_histogram_capacity(&device, num_workgroups);

    queue.write_buffer(&pass.tlas_prims_buffer, 0, bytemuck::cast_slice(&prims));
    queue.write_buffer(
        &pass.tlas_state_buffer,
        0,
        bytemuck::bytes_of(&TlasState {
            prim_count,
            radix_workgroups: num_workgroups,
            internal_wgs: 0,
            total_node_wgs: 0,
        }),
    );
    let (scene_min, scene_max) = scene_aabb_from_prims(&prims);
    queue.write_buffer(
        &pass.morton_uniform_buffer,
        0,
        bytemuck::bytes_of(&MortonUniform {
            scene_min,
            _pad0: 0,
            scene_max,
            _pad1: 0,
        }),
    );
    let radix_uniform_stride: u64 = 256;
    let mut radix_uniform_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_uniform_stride) as usize];
    for p in 0..RADIX_PASSES {
        let u = RadixUniform {
            digit_shift: p * 8,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let off = (p as u64 * radix_uniform_stride) as usize;
        radix_uniform_bytes[off..off + std::mem::size_of::<RadixUniform>()]
            .copy_from_slice(bytemuck::bytes_of(&u));
    }
    queue.write_buffer(&pass.radix_uniform_buffer, 0, &radix_uniform_bytes);

    let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.morton_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.vals_a_buffer.as_entire_binding() },
        ],
    });
    let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.morton_g1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.morton_uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.tlas_state_buffer.as_entire_binding() },
        ],
    });
    let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_b_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.histogram_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.scan_offsets_buffer.as_entire_binding() },
        ],
    });
    let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pass.radix_g1_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pass.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pass.tlas_state_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cpass.set_pipeline(&pass.morton_pipeline);
        cpass.set_bind_group(0, &morton_g0, &[]);
        cpass.set_bind_group(1, &morton_g1, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    let histogram_bytes = (num_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
    for p in 0..RADIX_PASSES {
        let radix_g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
        let dyn_off = (p as u64 * radix_uniform_stride) as u32;
        encoder.clear_buffer(&pass.histogram_buffer, 0, Some(histogram_bytes));
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_count_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_scan_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&pass.radix_scatter_pipeline);
            cpass.set_bind_group(0, radix_g0, &[]);
            cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }
    }

    let vals_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.vals_a_buffer, 0, &vals_readback, 0, 4);
    queue.submit(std::iter::once(encoder.finish()));
    let s = vals_readback.slice(..);
    s.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let v = s.get_mapped_range();
    let val0 = u32::from_le_bytes(v[0..4].try_into().unwrap());
    drop(v);
    vals_readback.unmap();
    assert_eq!(val0, 0u32, "single-prim sort should pass through prim_idx 0");
    // Quiet unused imports for the no-adapter branch.
    let _ = std::any::type_name::<InstanceTileCullEntry>();
}
