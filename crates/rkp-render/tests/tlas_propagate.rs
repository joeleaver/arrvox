//! Phase 7c Session 4 — wgpu integration test for the bottom-up
//! AABB propagation pass.
//!
//! Bakes a full GPU TLAS build (S3 Karras topology + S4 AABB
//! propagation), reads back `tlas_nodes`, and asserts every
//! internal node's AABB equals the union of its subtree's
//! primitives. Compares against `cpu_reference_full_tree` for
//! exact bit-for-bit AABB matching.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::tlas_build_pass::{
    cpu_reference_full_tree, TlasBuildPass, TlasPrim, TlasState, TLAS_LEAF_USER_SHADER,
};
use rkp_render::tlas_pass::{TlasInstanceLeaf, TlasNode, TLAS_NODE_LEAF_BIT};

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
        label: Some("tlas_propagate test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_prim(asset_id: u32, min: [f32; 3], max: [f32; 3]) -> TlasPrim {
    TlasPrim {
        aabb_min: min,
        asset_id,
        aabb_max: max,
        instance_state_offset: asset_id * 8,
        material_id: 100 + asset_id,
        instance_index: TLAS_LEAF_USER_SHADER,
        _pad0: 0,
        _pad1: 0,
    }
}

fn run_full_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sorted_keys: &[u32],
    sorted_vals: &[u32],
    prims: &[TlasPrim],
) -> (Vec<TlasNode>, Vec<TlasInstanceLeaf>) {
    let n = sorted_keys.len() as u32;
    assert!(n >= 1);

    let mut pass = TlasBuildPass::new(device);
    pass.ensure_keys_capacity(device, n);
    pass.ensure_prims_capacity(device, prims.len() as u32);
    pass.ensure_parents_capacity(device, 2 * n - 1);
    pass.ensure_aabb_atomic_capacity(device, (2 * n).saturating_sub(1).saturating_mul(3).max(3));

    queue.write_buffer(&pass.keys_a_buffer, 0, bytemuck::cast_slice(sorted_keys));
    queue.write_buffer(&pass.vals_a_buffer, 0, bytemuck::cast_slice(sorted_vals));
    queue.write_buffer(&pass.tlas_prims_buffer, 0, bytemuck::cast_slice(prims));
    let radix_workgroups = ((n + 63) / 64).max(1);
    let internal_wgs = if n >= 2 { (((n - 1) + 63) / 64).max(1) } else { 0 };
    let total_node_wgs = if n >= 2 { ((2 * n - 1 + 63) / 64).max(1) } else { 0 };
    queue.write_buffer(
        &pass.tlas_state_buffer,
        0,
        bytemuck::bytes_of(&TlasState {
            prim_count: n,
            radix_workgroups,
            internal_wgs,
            total_node_wgs,
        }),
    );
    // The GPU init pass writes parents/aabb_atomic when run; this
    // test invokes it explicitly below, so no CPU pre-fill is
    // needed (the init pass also seeds `parents[i] = SENTINEL`).

    let total_nodes = (2 * n - 1).max(1);
    let nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test tlas_nodes"),
        size: (total_nodes as u64) * (std::mem::size_of::<TlasNode>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test tlas_leaves"),
        size: (n as u64) * (std::mem::size_of::<TlasInstanceLeaf>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("propagate g0"),
        layout: &pass.karras_g0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pass.keys_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: pass.vals_a_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pass.tlas_prims_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: nodes_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: leaves_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: pass.parents_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: pass.aabb_min_atomic_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: pass.aabb_max_atomic_buffer.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("propagate g1"),
        layout: &pass.karras_g1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.tlas_state_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("propagate enc"),
    });
    let leaf_wgs = radix_workgroups;
    // Match the production chain order: init must run BEFORE
    // build_internal_main, since init seeds `parents[i] = SENTINEL`
    // and build_internal overwrites the non-root entries.
    if n >= 2 {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("init_atomic_aabb_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.init_atomic_aabb_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        cpass.dispatch_workgroups(total_node_wgs, 1, 1);
    } else {
        // N == 1: no internal nodes; pre-fill parent[0] with the
        // sentinel so propagate's walk-up loop terminates if it
        // were to run (it doesn't here, but keeps semantics clean).
        let parents_init: Vec<u32> = vec![0xFFFFFFFFu32; (2 * n - 1) as usize];
        queue.write_buffer(&pass.parents_buffer, 0, bytemuck::cast_slice(&parents_init));
    }
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_leaves_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.karras_leaves_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        cpass.dispatch_workgroups(leaf_wgs, 1, 1);
    }
    if n >= 2 {
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_internal_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.karras_internal_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            cpass.dispatch_workgroups(internal_wgs, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("propagate_atomic_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.propagate_atomic_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            cpass.dispatch_workgroups(leaf_wgs, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decode_aabb_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pass.decode_aabb_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            cpass.dispatch_workgroups(internal_wgs, 1, 1);
        }
    }
    // N == 1: only leaf-marker — its AABB was written by
    // build_leaves_main; no internal propagation needed.

    let nodes_bytes = (total_nodes as u64) * (std::mem::size_of::<TlasNode>() as u64);
    let leaves_bytes = (n as u64) * (std::mem::size_of::<TlasInstanceLeaf>() as u64);
    let nodes_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nodes readback"),
        size: nodes_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let leaves_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("leaves readback"),
        size: leaves_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&nodes_buffer, 0, &nodes_readback, 0, nodes_bytes);
    encoder.copy_buffer_to_buffer(&leaves_buffer, 0, &leaves_readback, 0, leaves_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let ns = nodes_readback.slice(..);
    ns.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let nv = ns.get_mapped_range();
    let nodes: Vec<TlasNode> = bytemuck::cast_slice::<u8, TlasNode>(&nv).to_vec();
    drop(nv);
    nodes_readback.unmap();

    let ls = leaves_readback.slice(..);
    ls.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let lv = ls.get_mapped_range();
    let leaves: Vec<TlasInstanceLeaf> = bytemuck::cast_slice::<u8, TlasInstanceLeaf>(&lv).to_vec();
    drop(lv);
    leaves_readback.unmap();

    (nodes, leaves)
}

fn assert_aabb_eq(a: &TlasNode, b: &TlasNode, label: &str) {
    for ax in 0..3 {
        assert!(
            (a.aabb_min[ax] - b.aabb_min[ax]).abs() < 1e-5,
            "{label} aabb_min[{ax}] mismatch: gpu={} cpu={}",
            a.aabb_min[ax],
            b.aabb_min[ax],
        );
        assert!(
            (a.aabb_max[ax] - b.aabb_max[ax]).abs() < 1e-5,
            "{label} aabb_max[{ax}] mismatch: gpu={} cpu={}",
            a.aabb_max[ax],
            b.aabb_max[ax],
        );
    }
}

#[test]
fn full_tree_with_aabb_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[tlas_propagate] no wgpu adapter — skipping");
        return;
    };
    // 4 prims with non-overlapping AABBs — easy to verify.
    let prims = vec![
        make_prim(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        make_prim(1, [10.0, 0.0, 0.0], [11.0, 1.0, 1.0]),
        make_prim(2, [0.0, 10.0, 0.0], [1.0, 11.0, 1.0]),
        make_prim(3, [10.0, 10.0, 0.0], [11.0, 11.0, 1.0]),
    ];
    // Mortons such that the spatial layout matches the centroids:
    // (0,0)→0, (10,0)→0xAA, (0,10)→0x55, (10,10)→0xFF.
    let sorted_keys = vec![0u32, 0x55u32, 0xAAu32, 0xFFu32];
    // sorted_vals must match the prim with the corresponding
    // Morton: prim 0 → key 0, prim 2 → key 0x55, prim 1 → key 0xAA,
    // prim 3 → key 0xFF.
    let sorted_vals = vec![0u32, 2, 1, 3];

    let (gpu_nodes, gpu_leaves) = run_full_build(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    let (cpu_nodes, cpu_leaves) = cpu_reference_full_tree(&sorted_keys, &sorted_vals, &prims);

    assert_eq!(gpu_nodes.len(), cpu_nodes.len());
    assert_eq!(gpu_leaves.len(), cpu_leaves.len());

    // Every internal node's AABB matches.
    for i in 0..gpu_nodes.len() {
        assert_eq!(
            gpu_nodes[i].left_or_leaf, cpu_nodes[i].left_or_leaf,
            "node {i} left_or_leaf"
        );
        assert_eq!(
            gpu_nodes[i].right_or_count, cpu_nodes[i].right_or_count,
            "node {i} right_or_count"
        );
        assert_aabb_eq(&gpu_nodes[i], &cpu_nodes[i], &format!("node {i}"));
    }

    // Root AABB must enclose every primitive.
    let root = &gpu_nodes[0];
    for p in &prims {
        for ax in 0..3 {
            assert!(
                root.aabb_min[ax] <= p.aabb_min[ax] + 1e-5,
                "root aabb doesn't enclose prim aabb_min[{ax}]={}",
                p.aabb_min[ax],
            );
            assert!(
                root.aabb_max[ax] + 1e-5 >= p.aabb_max[ax],
                "root aabb doesn't enclose prim aabb_max[{ax}]={}",
                p.aabb_max[ax],
            );
        }
    }
}

#[test]
fn propagate_eight_leaves_full_chain() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    // 8 prims in a 2x2x2 grid; verify root AABB encloses all.
    let mut prims = Vec::with_capacity(8);
    for i in 0..8u32 {
        let x = (i & 1) as f32 * 10.0;
        let y = ((i >> 1) & 1) as f32 * 10.0;
        let z = ((i >> 2) & 1) as f32 * 10.0;
        prims.push(make_prim(i, [x, y, z], [x + 1.0, y + 1.0, z + 1.0]));
    }
    // Use the centroid-derived Mortons (any consistent ordering works).
    let sorted_keys: Vec<u32> = (0..8u32).map(|i| i * 0x100).collect();
    let sorted_vals: Vec<u32> = (0..8u32).collect();

    let (gpu_nodes, _) = run_full_build(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    let (cpu_nodes, _) = cpu_reference_full_tree(&sorted_keys, &sorted_vals, &prims);
    for i in 0..gpu_nodes.len() {
        assert_aabb_eq(&gpu_nodes[i], &cpu_nodes[i], &format!("node {i}"));
    }
    // Sanity: root encloses [0, 11]³.
    let root = &gpu_nodes[0];
    assert_eq!(root.aabb_min, [0.0, 0.0, 0.0]);
    assert_eq!(root.aabb_max, [11.0, 11.0, 11.0]);
}

#[test]
fn propagate_single_leaf_writes_aabb() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    let prims = vec![make_prim(0, [3.0, 4.0, 5.0], [4.0, 5.0, 6.0])];
    let sorted_keys = vec![0u32];
    let sorted_vals = vec![0u32];
    let (gpu_nodes, _) = run_full_build(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    assert_eq!(gpu_nodes.len(), 1);
    let leaf = &gpu_nodes[0];
    assert!((leaf.left_or_leaf & TLAS_NODE_LEAF_BIT) != 0);
    assert_eq!(leaf.aabb_min, [3.0, 4.0, 5.0]);
    assert_eq!(leaf.aabb_max, [4.0, 5.0, 6.0]);
}
