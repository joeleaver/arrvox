//! Phase 7c Session 3 — wgpu integration test for the Karras
//! parallel BVH builder.
//!
//! Bakes a known set of `TlasPrim`s + sorted Morton keys, runs
//! both Karras dispatches (`build_leaves_main`, `build_internal_main`)
//! on a real wgpu device, reads back `tlas_nodes` + `tlas_leaves`,
//! and asserts:
//!   1. Each internal node's children match the CPU reference
//!      (`cpu_reference_karras_node`).
//!   2. Each leaf-marker node has `LEAF_BIT | i` in `left_or_leaf`.
//!   3. `tlas_leaves[i]` carries the payload of `tlas_prims[sorted_vals[i]]`.
//!   4. Walking from root visits each leaf exactly once.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::tlas_build_pass::{
    cpu_reference_karras_node, KarrasUniform, TlasBuildPass, TlasPrim,
    TLAS_LEAF_USER_SHADER,
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
        label: Some("tlas_karras test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_prim(asset_id: u32, instance_index: u32) -> TlasPrim {
    TlasPrim {
        aabb_min: [asset_id as f32; 3],
        asset_id,
        aabb_max: [asset_id as f32 + 0.5; 3],
        instance_state_offset: asset_id * 8,
        material_id: 100 + asset_id,
        instance_index,
        _pad0: 0,
        _pad1: 0,
    }
}

fn run_karras(
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
    queue.write_buffer(
        &pass.karras_uniform_buffer,
        0,
        bytemuck::bytes_of(&KarrasUniform {
            prim_count: n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }),
    );
    // Pre-fill parents with the sentinel — the tree builder
    // overwrites for non-root nodes; root's parent slot stays
    // 0xFFFFFFFF.
    let parents_init: Vec<u32> = vec![0xFFFFFFFFu32; (2 * n - 1) as usize];
    queue.write_buffer(&pass.parents_buffer, 0, bytemuck::cast_slice(&parents_init));

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
        label: Some("karras g0"),
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
        label: Some("karras g1"),
        layout: &pass.karras_g1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.karras_uniform_buffer.as_entire_binding(),
        }],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("karras enc") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_leaves_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.karras_leaves_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        let wgs = ((n + 63) / 64).max(1);
        cpass.dispatch_workgroups(wgs, 1, 1);
    }
    if n >= 2 {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_internal_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.karras_internal_pipeline);
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        let wgs = (((n - 1) + 63) / 64).max(1);
        cpass.dispatch_workgroups(wgs, 1, 1);
    }

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

#[test]
fn karras_three_leaves_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[tlas_karras] no wgpu adapter — skipping");
        return;
    };
    let prims = vec![
        make_prim(0, TLAS_LEAF_USER_SHADER),
        make_prim(1, TLAS_LEAF_USER_SHADER),
        make_prim(2, 42),
    ];
    let sorted_keys = vec![1u32, 2, 4];
    let sorted_vals = vec![0u32, 1, 2]; // identity (input already sorted)
    let n = sorted_keys.len() as u32;

    let (nodes, leaves) = run_karras(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    assert_eq!(nodes.len() as u32, 2 * n - 1);
    assert_eq!(leaves.len() as u32, n);

    // Internal node topology matches CPU reference.
    for i in 0..(n - 1) {
        let (cpu_l, cpu_r) = cpu_reference_karras_node(&sorted_keys, i as i32);
        assert_eq!(
            nodes[i as usize].left_or_leaf, cpu_l,
            "internal node {i} left mismatch"
        );
        assert_eq!(
            nodes[i as usize].right_or_count, cpu_r,
            "internal node {i} right mismatch"
        );
    }

    // Leaf-marker nodes: indices N-1 .. 2N-2.
    for i in 0..n {
        let node = &nodes[(n - 1 + i) as usize];
        assert_eq!(node.left_or_leaf, TLAS_NODE_LEAF_BIT | i, "leaf-marker {i} wrong tag");
        assert_eq!(node.right_or_count, 1u32);
    }

    // Leaves carry the payload of the corresponding sorted prim.
    for i in 0..(n as usize) {
        let prim_idx = sorted_vals[i] as usize;
        assert_eq!(leaves[i].asset_id, prims[prim_idx].asset_id);
        assert_eq!(leaves[i].instance_state_offset, prims[prim_idx].instance_state_offset);
        assert_eq!(leaves[i].material_id, prims[prim_idx].material_id);
        assert_eq!(leaves[i].instance_index, prims[prim_idx].instance_index);
    }

    // Tree walk reaches every leaf exactly once.
    let mut leaf_visits = vec![0u32; n as usize];
    let mut visited = vec![false; nodes.len()];
    let mut stack = vec![0u32];
    while let Some(idx) = stack.pop() {
        assert!(!visited[idx as usize], "cycle at node {idx}");
        visited[idx as usize] = true;
        let node = &nodes[idx as usize];
        if (node.left_or_leaf & TLAS_NODE_LEAF_BIT) != 0 {
            let leaf_idx = node.left_or_leaf & 0x7FFFFFFF;
            leaf_visits[leaf_idx as usize] += 1;
        } else {
            stack.push(node.left_or_leaf);
            stack.push(node.right_or_count);
        }
    }
    for (i, &c) in leaf_visits.iter().enumerate() {
        assert_eq!(c, 1, "leaf {i} visited {c} times");
    }
}

#[test]
fn karras_eight_leaves_full_walk() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    let prims: Vec<TlasPrim> = (0..8).map(|i| make_prim(i, i * 10)).collect();
    let sorted_keys: Vec<u32> = (0..8u32).map(|i| i * 0x100 + 0x42).collect();
    let sorted_vals: Vec<u32> = (0..8u32).collect();
    let n = sorted_keys.len() as u32;

    let (nodes, _leaves) = run_karras(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    assert_eq!(nodes.len() as u32, 2 * n - 1);

    for i in 0..(n - 1) {
        let (cpu_l, cpu_r) = cpu_reference_karras_node(&sorted_keys, i as i32);
        assert_eq!(nodes[i as usize].left_or_leaf, cpu_l, "node {i} left");
        assert_eq!(nodes[i as usize].right_or_count, cpu_r, "node {i} right");
    }

    // Every leaf reachable.
    let mut leaf_visits = vec![0u32; n as usize];
    let mut stack = vec![0u32];
    while let Some(idx) = stack.pop() {
        let node = &nodes[idx as usize];
        if (node.left_or_leaf & TLAS_NODE_LEAF_BIT) != 0 {
            leaf_visits[(node.left_or_leaf & 0x7FFFFFFF) as usize] += 1;
        } else {
            stack.push(node.left_or_leaf);
            stack.push(node.right_or_count);
        }
    }
    assert!(leaf_visits.iter().all(|&c| c == 1));
}

#[test]
fn karras_single_leaf_no_internal_nodes() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    let prims = vec![make_prim(7, 99)];
    let sorted_keys = vec![0xFFu32];
    let sorted_vals = vec![0u32];
    let (nodes, leaves) = run_karras(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    assert_eq!(nodes.len(), 1);
    assert_eq!(leaves.len(), 1);
    // Single node = leaf-marker, written by build_leaves_main.
    assert_eq!(nodes[0].left_or_leaf, TLAS_NODE_LEAF_BIT | 0);
    assert_eq!(nodes[0].right_or_count, 1u32);
    assert_eq!(leaves[0].asset_id, 7);
    assert_eq!(leaves[0].instance_index, 99);
}

#[test]
fn karras_handles_duplicate_mortons_on_gpu() {
    let Some((device, queue)) = create_device() else {
        return;
    };
    // Four prims sharing the same Morton — algorithm relies on the
    // index tiebreak in delta() to disambiguate.
    let prims: Vec<TlasPrim> = (0..4).map(|i| make_prim(i, i)).collect();
    let sorted_keys = vec![5u32; 4];
    let sorted_vals = vec![0u32, 1, 2, 3];
    let n = sorted_keys.len() as u32;
    let (nodes, _leaves) = run_karras(&device, &queue, &sorted_keys, &sorted_vals, &prims);
    assert_eq!(nodes.len() as u32, 2 * n - 1);

    // GPU result must match CPU reference even with duplicates.
    for i in 0..(n - 1) {
        let (cpu_l, cpu_r) = cpu_reference_karras_node(&sorted_keys, i as i32);
        assert_eq!(nodes[i as usize].left_or_leaf, cpu_l, "node {i} left (dup mortons)");
        assert_eq!(nodes[i as usize].right_or_count, cpu_r, "node {i} right (dup mortons)");
    }
    // All leaves reachable.
    let mut leaf_visits = vec![0u32; n as usize];
    let mut stack = vec![0u32];
    while let Some(idx) = stack.pop() {
        let node = &nodes[idx as usize];
        if (node.left_or_leaf & TLAS_NODE_LEAF_BIT) != 0 {
            leaf_visits[(node.left_or_leaf & 0x7FFFFFFF) as usize] += 1;
        } else {
            stack.push(node.left_or_leaf);
            stack.push(node.right_or_count);
        }
    }
    assert!(leaf_visits.iter().all(|&c| c == 1));
}
