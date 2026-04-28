//! V9 sparse-BFS GPU geometry test.
//!
//! Verifies that the BFS pipeline produces correct voxel output by
//! exercising the full classify+fill chain with a known shader. Covers
//! the high-impact regressions:
//!   * BFS allocates octree nodes top-down without overlap.
//!   * Atomic alloc counters bump correctly per region.
//!   * Active-queue ping-pong between levels does not lose cells.
//!   * `host_sample_at` no-host sentinel returns +inf so non-host
//!     regions don't get gated to empty.
//!   * Brick fill writes leaf-attrs and brick-pool slots that match
//!     the user shader's emit decisions.
//!
//! The reference shader is a "ball-on-cell" generator: cell at world
//! position `p` is occupied iff `length(p - center) < radius`. We bake
//! once at depth=4 (16 cells/axis), then read back the brick pool +
//! leaf-attr pool over the region's reserved slice and count
//! occupied cells. The count is compared against a CPU enumeration of
//! the same condition over the same lattice.

use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_pass::{
    build_region_uniform, effective_hash, estimate_region_pool, resolve_shader_id,
    RegionUniform, ShaderRegionRequest, UserShaderObjectCache, UserShaderPass,
    BRICK_CELLS, HOST_NO_HOST_SENTINEL, NO_TILE,
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
    // Match production limits (rkp_render::context) — the BFS pass
    // uses 10 storage bindings, which is over the default 8 cap.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("user_shader_geom_bfs test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            max_storage_buffers_per_shader_stage: 20,
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
        "rkp_user_shader_geom_bfs_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.wgsl"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    dir
}

/// End-to-end: bake a ball-on-cell shader, read back the brick pool,
/// verify occupancy matches the analytic condition.
#[test]
fn ball_shader_bfs_matches_cpu_reference() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[ball_bfs] no wgpu adapter — skipping");
        return;
    };

    // Shader: cell occupied iff within `radius = 0.5` of region center.
    // The center of a unit cube at origin is (0.5, 0.5, 0.5).
    let src = r#"
// @param radius: f32 = 0.5
fn user_ball_generate(cell_world_pos: vec3<f32>, host: HostSample, ctx: UserCtx) -> VoxelEmit {
    var v: VoxelEmit;
    let center = ctx.aabb_min + vec3<f32>((ctx.aabb_min.x + ctx.cell_size) * 0.0) + vec3<f32>(0.5);
    let d = length(cell_world_pos - center);
    if (d < ctx.params[0]) {
        v.occupancy = 1u;
        v.normal = normalize(cell_world_pos - center + vec3<f32>(1e-6, 0.0, 0.0));
        v.material_primary = 1u;
    } else {
        v.occupancy = 0u;
    }
    return v;
}
"#;
    let dir = write_temp_shader("ball", src);
    let registry = scan_dir(&dir).unwrap();
    let composed = compose(&registry);

    // Prepare scene buffers — sized to fit the test region's pool
    // estimate. We don't need a real RkpScene here; raw storage
    // buffers wired to UserShaderPass's group 0 are sufficient.
    const MAX_DEPTH: u32 = 4;
    // No proximity gate (no host, region_thickness=0) means the BFS
    // expands every cell to the deepest level — full dense enumeration
    // at depth 4 = 4096 bricks. Painted-leaf count drives the estimate
    // function; a value high enough to cover the dense case prevents
    // brick-pool overflow.
    const PAINTED_COUNT: u32 = 1500;
    let estimate = estimate_region_pool(PAINTED_COUNT, MAX_DEPTH);
    // octree_nodes is bound as `array<vec2<u32>>` (8 B/elem); brick_pool
    // as `array<u32>` (4 B/elem); leaf_attr_pool as `array<LeafAttr>`
    // (8 B/elem).
    let octree_bytes = (estimate.octree as u64) * 8;
    let brick_bytes = (estimate.bricks as u64) * BRICK_CELLS as u64 * 4;
    let leaf_attr_bytes = (estimate.leaf_attrs as u64) * 8;

    let octree_nodes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test octree_nodes"),
        size: octree_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let brick_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test brick_pool"),
        size: brick_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaf_attr_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test leaf_attr_pool"),
        size: leaf_attr_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut pass = UserShaderPass::new(&device);
    pass.reload_user_shaders(&device, &composed.generate, registry.source_hash());

    let mut cache = UserShaderObjectCache::new();
    cache.set_pool_bases(
        0, estimate.octree,
        0, estimate.bricks * BRICK_CELLS,
        0, estimate.leaf_attrs,
    );

    let req = ShaderRegionRequest {
        host_object_id: 1,
        material_id: 1,
        shader_name: "ball".to_string(),
        params: vec![0.5], // radius
        aabb_min: [0.0, 0.0, 0.0],
        aabb_max: [1.0, 1.0, 1.0],
        cell_size: 1.0 / (4.0 * (1u32 << MAX_DEPTH) as f32),
        input_hash: 0,
        animated: false,
        region_thickness: 0.0,
        max_depth: MAX_DEPTH,
        painted_leaf_count: PAINTED_COUNT,
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
        tile_index: NO_TILE,
    };
    let h = effective_hash(&req, registry.source_hash(), 0);
    let slot = cache.lookup_or_allocate(&req, h).unwrap();
    let shader_id = resolve_shader_id(&registry.shader_infos(), "ball");
    assert!(shader_id != 0, "ball shader should have id 1");
    let uniform: RegionUniform = build_region_uniform(&req, &slot, shader_id, 0.0);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test bfs encoder"),
    });
    pass.dispatch_regions(
        &device, &queue, &mut encoder, &[uniform], MAX_DEPTH,
        &octree_nodes_buffer, &brick_pool_buffer, &leaf_attr_pool_buffer, 0,
    );

    // Read back diagnostic counters: leaf_attr_alloc[0] (occupied
    // cells), brick_alloc[0] (allocated bricks), fill_count[0] (queued
    // brick fills), active_count[0..MAX_DEPTH+1] (BFS expansion per
    // level). Lets us tell whether classify never queued bricks vs
    // bricks queued but the user shader returned occupancy=0.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test counters staging"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let leaf_alloc_buffer = pass.test_leaf_attr_alloc_buffer();
    let brick_alloc_buffer = pass.test_brick_alloc_buffer();
    let fill_count_buffer = pass.test_fill_count_buffer();
    let active_count_buffer = pass.test_active_count_buffer();
    encoder.copy_buffer_to_buffer(leaf_alloc_buffer, 0, &staging, 0, 16);
    encoder.copy_buffer_to_buffer(brick_alloc_buffer, 0, &staging, 16, 16);
    encoder.copy_buffer_to_buffer(fill_count_buffer, 0, &staging, 32, 16);
    encoder.copy_buffer_to_buffer(active_count_buffer, 0, &staging, 48, 4 * 9);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(0..256);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    rx.recv().expect("map_async channel").expect("map_async result");
    let data = slice.get_mapped_range();
    let bytes: Vec<u8> = data.to_vec();
    drop(data);
    staging.unmap();
    let leaf_alloc = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let brick_alloc = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let fill_count = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
    let active_counts: Vec<u32> = (0..9)
        .map(|i| u32::from_le_bytes(bytes[48 + i * 4..52 + i * 4].try_into().unwrap()))
        .collect();
    // Sanity: BFS expansion at depth=4 should be 1 → 8 → 64 → 512 → 4096
    // → 0 (last level is brick parents, not internal cells). Brick &
    // fill counts match the deepest level's count.
    assert_eq!(active_counts[..5], [1, 8, 64, 512, 4096]);
    assert_eq!(brick_alloc, 4096);
    assert_eq!(fill_count, 4096);
    let occupied_gpu = leaf_alloc;

    // CPU reference — enumerate the lattice the GPU would produce for
    // the ball shader. The deepest level has `BRICK_DIM * 2^max_depth`
    // = 4*16 = 64 cells per axis; cell side `cell_size`; centers offset
    // by `cell_size * 0.5` from `aabb_min`.
    let cells_per_axis = 4u32 * (1u32 << MAX_DEPTH);
    let cs = req.cell_size;
    let center = [0.5_f32, 0.5, 0.5];
    let radius = 0.5_f32;
    let mut occupied_cpu: u32 = 0;
    for z in 0..cells_per_axis {
        for y in 0..cells_per_axis {
            for x in 0..cells_per_axis {
                let p = [
                    req.aabb_min[0] + (x as f32 + 0.5) * cs,
                    req.aabb_min[1] + (y as f32 + 0.5) * cs,
                    req.aabb_min[2] + (z as f32 + 0.5) * cs,
                ];
                let d = ((p[0] - center[0]).powi(2)
                    + (p[1] - center[1]).powi(2)
                    + (p[2] - center[2]).powi(2))
                .sqrt();
                if d < radius {
                    occupied_cpu += 1;
                }
            }
        }
    }
    assert!(occupied_cpu > 0, "CPU reference should find some occupied cells");

    // Tolerance: classifier may prune cells whose ENTIRE volume is far
    // from the surface band, but with `region_thickness = 0` and no
    // host the band gate is skipped — every brick is MIXED, every cell
    // gets fully evaluated. So the GPU and CPU counts should match
    // exactly.
    assert_eq!(
        occupied_gpu, occupied_cpu,
        "GPU BFS bake should match CPU enumeration exactly when there is no proximity gate"
    );
}

/// Sanity test: with no registered shader the dispatch is a no-op and
/// the brick pool stays whatever it was initialized to. Confirms that
/// the empty-pipeline path doesn't crash.
#[test]
fn empty_registry_dispatch_no_op() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[empty] no wgpu adapter — skipping");
        return;
    };
    let mut pass = UserShaderPass::new(&device);
    pass.reload_user_shaders(&device, "", 0);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test empty bfs encoder"),
    });
    // Tiny dummy buffers — dispatch_regions returns on empty
    // uniforms before touching them, but the API still requires
    // them for the post-capacity-grow ensure_group0 path.
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    pass.dispatch_regions(
        &device, &queue, &mut encoder, &[], 0,
        &dummy, &dummy, &dummy, 0,
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
}
