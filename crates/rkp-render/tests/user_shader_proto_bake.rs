//! Stage 2 end-to-end test for the Option B prototype bake.
//!
//! Bakes a "sphere shader" prototype on a real wgpu device, reads back
//! the per-prototype atomic counter buffers, and verifies the emitted
//! brick + leaf-attr counts match a CPU enumeration of the same
//! sphere over the same canonical lattice.
//!
//! Skips silently when no wgpu adapter is available (CI sandbox /
//! headless without a GPU).

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_proto_pass::{
    build_internal_levels, level_starts_inclusive, max_bricks_for_depth,
    PrototypeBakePass, PrototypeCache, PrototypeUniform,
};
use rkp_render::rkp_scene::{OCTREE_NODE_BYTES, OCTREE_NODE_U32S};

const PROTO_BRICK_CAPACITY: u32 = 1024;
const PROTO_LEAF_ATTR_CAPACITY: u32 = 8192;

const PROTO_MAX_DEPTH: u32 = 2;
const SHADER_ID: u32 = 1;
const SOURCE_HASH: u64 = 0xC0FFEE_DEADBEEF;
const POOL_OCTREE_BASE: u32 = 0;
const POOL_BRICK_BASE: u32 = 0;
const POOL_LEAF_ATTR_BASE: u32 = 0;

/// Sphere shader source — emits a cell iff its uvw is inside a sphere
/// of radius 0.4 centered at (0.5, 0.5, 0.5) in canonical prototype
/// space. The CPU reference below enumerates the same condition.
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
        label: Some("user_shader_proto_bake test device"),
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
        "rkp_user_shader_proto_bake_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.wgsl"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    dir
}

/// CPU enumeration of the sphere condition. Returns `(brick_count,
/// leaf_attr_count)` — bricks where ≥1 cell is inside, and total cells
/// inside across the whole lattice.
fn cpu_sphere_reference(max_depth: u32, center: [f32; 3], radius: f32) -> (u32, u32) {
    let cells_per_axis = 4u32 << max_depth;
    let bricks_per_axis = 1u32 << max_depth;
    let mut leaf_count: u32 = 0;
    let mut brick_count: u32 = 0;
    for bz in 0..bricks_per_axis {
        for by in 0..bricks_per_axis {
            for bx in 0..bricks_per_axis {
                let mut any = false;
                for lz in 0..4 {
                    for ly in 0..4 {
                        for lx in 0..4 {
                            let cx = bx * 4 + lx;
                            let cy = by * 4 + ly;
                            let cz = bz * 4 + lz;
                            let u = (cx as f32 + 0.5) / cells_per_axis as f32;
                            let v = (cy as f32 + 0.5) / cells_per_axis as f32;
                            let w = (cz as f32 + 0.5) / cells_per_axis as f32;
                            let dx = u - center[0];
                            let dy = v - center[1];
                            let dz = w - center[2];
                            if (dx * dx + dy * dy + dz * dz).sqrt() < radius {
                                leaf_count += 1;
                                any = true;
                            }
                        }
                    }
                }
                if any {
                    brick_count += 1;
                }
            }
        }
    }
    (brick_count, leaf_count)
}

#[test]
fn sphere_prototype_bake_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[sphere_proto_bake] no wgpu adapter — skipping");
        return;
    };

    // Compose the user-shader chunk for the sphere shader.
    let dir = write_temp_shader("sphere", SPHERE_SHADER);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);
    assert!(chunks.proto.contains("rkp_user_1_proto"));

    // Build the bake pass and reload the pipeline against the
    // composed chunk.
    let mut pass = PrototypeBakePass::new(&device);
    pass.reload_user_shaders(&device, &chunks.proto, registry.source_hash());

    // Allocate a cache slot for the sphere shader. Pool capacities are
    // small (just enough for one prototype at depth 2). Brick / leaf-
    // attr capacities double as the GPU overflow gates the bake checks.
    let mut cache = PrototypeCache::with_capacities(
        1024, PROTO_BRICK_CAPACITY, PROTO_LEAF_ATTR_CAPACITY,
    );
    cache.set_pool_bases(POOL_OCTREE_BASE, POOL_BRICK_BASE, POOL_LEAF_ATTR_BASE);
    let (entry, dirty) = cache
        .lookup_or_allocate(SHADER_ID, SOURCE_HASH, PROTO_MAX_DEPTH)
        .unwrap();
    assert!(dirty, "first lookup must be dirty");

    // Build the prototype uniform from the entry.
    let uniform = PrototypeUniform::from_entry(&entry, &cache);

    // Allocate small GPU buffers for the pool storage. Octree extent
    // is per-prototype; bricks + leaf-attrs are global, sized to the
    // pool capacity.
    let octree_buffer_bytes =
        ((entry.octree_extent.0 + entry.octree_extent.1) as u64) * OCTREE_NODE_BYTES + 16;
    let octree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test octree_nodes"),
        size: octree_buffer_bytes.max(64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let brick_buffer_bytes = (PROTO_BRICK_CAPACITY as u64) * 64 * 4;
    let brick_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test brick_pool"),
        size: brick_buffer_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaf_attr_buffer_bytes = (PROTO_LEAF_ATTR_CAPACITY as u64) * 8;
    let leaf_attr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test leaf_attr_pool"),
        size: leaf_attr_buffer_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Pre-build the internal octree levels into the octree buffer at
    // the entry's extent offset. Each node is `vec4<u32>` = 16 bytes
    // (post Step 1 of the per-node tight-bounds rollout).
    let internal = build_internal_levels(POOL_OCTREE_BASE, entry.octree_extent.0, PROTO_MAX_DEPTH);
    let mut octree_init: Vec<u8> = Vec::with_capacity(internal.len() * OCTREE_NODE_BYTES as usize);
    for [v0, v1, v2, v3] in internal {
        octree_init.extend_from_slice(&v0.to_le_bytes());
        octree_init.extend_from_slice(&v1.to_le_bytes());
        octree_init.extend_from_slice(&v2.to_le_bytes());
        octree_init.extend_from_slice(&v3.to_le_bytes());
    }
    queue.write_buffer(
        &octree_buffer,
        (entry.octree_extent.0 as u64) * OCTREE_NODE_BYTES,
        &octree_init,
    );

    // Reset the global cursor pair (brick + leaf-attr) to 0 and clear
    // overflow for this isolated test bake. Bases of (0, 0) keep the
    // bake writing to absolute offset 0 in the dedicated test buffers.
    pass.reset_cursors(&queue, 0, 0);
    queue.write_buffer(&pass.overflow_buffer, 0, &[0u8; 12 * 4]);

    // Upload the prototype uniform.
    queue.write_buffer(
        &pass.proto_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniform),
    );

    // Build bind groups.
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test proto group0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pass.cursors_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test proto group1"),
        layout: &pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pass.proto_uniform_buffer.as_entire_binding(),
        }],
    });

    // Dispatch (2^max_depth)³ workgroups.
    let bricks_per_axis = 1u32 << PROTO_MAX_DEPTH;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test proto bake encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proto_bake_main"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pass.bake_pipeline);
        cpass.set_bind_group(0, &group0, &[]);
        cpass.set_bind_group(1, &group1, &[]);
        cpass.dispatch_workgroups(bricks_per_axis, bricks_per_axis, bricks_per_axis);
    }

    // Stage the global cursor pair for readback. Layout matches the
    // WGSL `GlobalCursors` struct: brick cursor at byte 0, leaf-attr
    // cursor at byte 4.
    let alloc_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("proto cursor readback"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&pass.cursors_buffer, 0, &alloc_readback, 0, 8);

    queue.submit(std::iter::once(encoder.finish()));

    let slice = alloc_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let gpu_brick_count = u32::from_le_bytes(view[0..4].try_into().unwrap());
    let gpu_leaf_count = u32::from_le_bytes(view[4..8].try_into().unwrap());
    drop(view);
    alloc_readback.unmap();

    let (cpu_brick_count, cpu_leaf_count) =
        cpu_sphere_reference(PROTO_MAX_DEPTH, [0.5, 0.5, 0.5], 0.4);
    assert_eq!(
        gpu_brick_count, cpu_brick_count,
        "brick count: GPU emitted {gpu_brick_count}, CPU reference is {cpu_brick_count}",
    );
    assert_eq!(
        gpu_leaf_count, cpu_leaf_count,
        "leaf-attr count: GPU emitted {gpu_leaf_count}, CPU reference is {cpu_leaf_count}",
    );

    // Sanity: capacity wasn't exceeded.
    let max_bricks = max_bricks_for_depth(PROTO_MAX_DEPTH);
    assert!(gpu_brick_count <= max_bricks);

    // Confirm the leaf-level octree slots got written. Spot-check: the
    // first few leaf-level nodes should now hold either OCTREE_EMPTY or
    // a valid LEAF+BRICK reference. Read back the whole octree extent
    // and check that any non-EMPTY leaf-level node has the BRICK bit
    // set and its brick_id is within the brick block.
    let level_starts = level_starts_inclusive(PROTO_MAX_DEPTH);
    let leaf_level_offset = level_starts[PROTO_MAX_DEPTH as usize];
    let leaf_level_size = 1u32 << (3 * PROTO_MAX_DEPTH);

    let octree_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("octree readback"),
        size: octree_buffer_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("octree readback encoder"),
    });
    encoder.copy_buffer_to_buffer(&octree_buffer, 0, &octree_readback, 0, octree_buffer_bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = octree_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let octree_words: &[u32] = bytemuck::cast_slice(&view);
    let mut valid_leaf_count = 0;
    for i in 0..leaf_level_size {
        let abs = (entry.octree_extent.0 + leaf_level_offset + i) as usize;
        let stride = OCTREE_NODE_U32S;
        let value = octree_words[abs * stride];
        if value == 0xFFFFFFFF {
            continue; // OCTREE_EMPTY
        }
        let is_leaf = (value & 0x80000000) != 0;
        let is_brick = (value & 0x40000000) != 0;
        assert!(is_leaf && is_brick, "leaf-level node {i} has invalid value {value:08x}");
        let brick_id = value & 0x3FFFFFFF;
        assert!(
            brick_id < PROTO_BRICK_CAPACITY,
            "brick_id {brick_id} outside global proto brick pool [0, {})",
            PROTO_BRICK_CAPACITY,
        );

        // Tight-bounds sanity. The bake writes `(.z, .w)` for every
        // non-empty leaf; the sentinel `(0, 0)` would mean "tight
        // bounds not set, treat as full cell" (Step 1 buffer-init
        // default), which after bake should never occur on a node
        // that allocated a brick.
        let aabb_lo = octree_words[abs * stride + 2];
        let aabb_hi = octree_words[abs * stride + 3];
        assert!(
            !(aabb_lo == 0 && aabb_hi == 0),
            "non-empty leaf {i} has Step-1 sentinel `.zw == 0` — bake \
             didn't write tight bounds (brick_id {brick_id})",
        );
        let lo_x = aabb_lo & 0xFF;
        let lo_y = (aabb_lo >> 8) & 0xFF;
        let lo_z = (aabb_lo >> 16) & 0xFF;
        let hi_x = aabb_hi & 0xFF;
        let hi_y = (aabb_hi >> 8) & 0xFF;
        let hi_z = (aabb_hi >> 16) & 0xFF;
        assert!(
            lo_x <= hi_x && lo_y <= hi_y && lo_z <= hi_z,
            "non-empty leaf {i} has inverted tight bounds: \
             lo=({lo_x}, {lo_y}, {lo_z}) hi=({hi_x}, {hi_y}, {hi_z})",
        );

        valid_leaf_count += 1;
    }
    drop(view);
    octree_readback.unmap();

    assert_eq!(
        valid_leaf_count, gpu_brick_count,
        "leaf-level non-empty count must equal brick allocation count",
    );
}
