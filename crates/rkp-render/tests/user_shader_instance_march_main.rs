//! Stage 5b end-to-end test for the instance march.
//!
//! Stitches together every Option B stage to date:
//!   1. Bake a sphere prototype (Stage 2).
//!   2. Run the emit pass to scatter exactly one instance into the
//!      instance pool (Stage 3) — the scatter shader emits at a
//!      hard-coded position that matches the region AABB center.
//!   3. Build a single-entry GPU TileIndex (Stage 5a) +
//!      proto-lookup (Stage 5b new).
//!   4. Dispatch `instance_march_main` with one ray known to hit the
//!      placed instance.
//!   5. Read back the `InstanceMarchHit` and assert hit + plausible
//!      world `t` + roughly-correct normal.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::instance_march_pass::{
    instance_march_main_source, InstanceMarchHit, InstanceMarchPass, MarchCameraUniform,
    MarchUniforms,
};
use rkp_render::instance_proto_lookup::{
    flatten_prototype_lookup, GpuPrototypeEntry,
};
use rkp_render::instance_tile_index::TileIndexBuilder;
use rkp_render::instance_tile_index_gpu::{flatten_tile_index, GpuTileIndexEntry};
use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_emit_pass::{
    build_emit_region_uniform, workgroups_for_leaf_count, EmitDispatchUniform, EmitPass,
    PaintedLeaf,
    EmitRegionUniform, InstanceRegionCache, InstanceRegionRequest,
    HOST_NO_HOST_SENTINEL, NO_TILE,
};
use rkp_render::user_shader_proto_pass::{
    build_internal_levels, PrototypeBakePass, PrototypeCache, PrototypeUniform,
};

const PROTO_MAX_DEPTH: u32 = 2;
const SHADER_ID: u32 = 1;
const SOURCE_HASH: u64 = 0xC0FFEE_DEADBEEF;
const POOL_OCTREE_BASE: u32 = 0;
const POOL_BRICK_BASE: u32 = 0;
const POOL_LEAF_ATTR_BASE: u32 = 0;
const POOL_INSTANCE_BASE: u32 = 0;
const STRIDE_U32: u32 = 4; // pos (3) + scale (1) = 4 u32 = 16 B

/// Stage 5b test shader. The instance struct is 16 B (pos + scale).
/// The emit hook hard-codes pos = host_pos, scale = 1.0 — with the
/// region's 1-sample grid, this places exactly one instance at the
/// region's center, giving the test a known target to march against.
///
/// File stem must be "sphereinst" so the parser derives hook names
/// `user_sphereinst_proto` / `user_sphereinst_emit`.
const SHADER_SRC: &str = r#"
// @instance_proto Pt
struct Pt {
    pos: vec3<f32>,
    scale: f32,
}

fn user_sphereinst_proto(uvw: vec3<f32>) -> VoxelEmit {
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

fn user_sphereinst_emit(host_pos: vec3<f32>, host: HostSample, ctx: UserCtx) {
    var p: Pt;
    p.pos = host_pos;
    p.scale = 1.0;
    emit_instance(p);
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
        label: Some("instance_march_main test device"),
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
        "rkp_instance_march_main_{name}_{}",
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
fn instance_march_hits_single_scattered_sphere_instance() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[instance_march e2e] no wgpu adapter — skipping");
        return;
    };

    // ── Compose user shaders ──────────────────────────────────────
    let dir = write_temp_shader("sphereinst", SHADER_SRC);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);
    assert!(
        chunks.proto.contains("rkp_user_1_proto"),
        "compose did not produce a rkp_user_1_proto symbol: chunks.proto = {}",
        chunks.proto,
    );
    assert!(
        chunks.emit.contains("rkp_user_1_emit_instance"),
        "compose did not produce a rkp_user_1_emit_instance symbol",
    );
    let registry_entries = registry.entries();
    assert_eq!(registry_entries.len(), 1);
    assert!(registry_entries[0].is_instance_pipeline());

    // ── Pool buffers (octree, brick, leaf-attr) ──────────────────
    let octree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e pool octree"),
        size: 8 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let brick_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e pool bricks"),
        size: 256 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaf_attr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e pool leaf_attrs"),
        size: 64 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Bake sphere prototype ────────────────────────────────────
    let mut proto_pass = PrototypeBakePass::new(&device);
    proto_pass.reload_user_shaders(&device, &chunks.proto, registry.source_hash());

    let mut proto_cache = PrototypeCache::with_capacities(1024, 1024, 8192);
    proto_cache.set_pool_bases(POOL_OCTREE_BASE, POOL_BRICK_BASE, POOL_LEAF_ATTR_BASE);
    let (proto_entry, _proto_dirty) = proto_cache
        .lookup_or_allocate(SHADER_ID, SOURCE_HASH, PROTO_MAX_DEPTH)
        .unwrap();
    let proto_uniform = PrototypeUniform::from_entry(&proto_entry, &proto_cache);

    // Pre-build internal octree levels.
    let internal = build_internal_levels(POOL_OCTREE_BASE, proto_entry.octree_extent.0, PROTO_MAX_DEPTH);
    let mut octree_init: Vec<u8> = Vec::with_capacity(internal.len() * 8);
    for [v0, v1] in internal {
        octree_init.extend_from_slice(&v0.to_le_bytes());
        octree_init.extend_from_slice(&v1.to_le_bytes());
    }
    queue.write_buffer(&octree_buffer, (proto_entry.octree_extent.0 as u64) * 8, &octree_init);

    proto_pass.reset_cursors(&queue);
    queue.write_buffer(&proto_pass.overflow_buffer, 0, &[0u8; 12 * 4]);
    queue.write_buffer(&proto_pass.proto_uniform_buffer, 0, bytemuck::bytes_of(&proto_uniform));

    let proto_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("proto bake g0"),
        layout: &proto_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: proto_pass.cursors_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: proto_pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let proto_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("proto bake g1"),
        layout: &proto_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: proto_pass.proto_uniform_buffer.as_entire_binding(),
        }],
    });

    let bricks_per_axis = 1u32 << PROTO_MAX_DEPTH;
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&proto_pass.bake_pipeline);
        cpass.set_bind_group(0, &proto_g0, &[]);
        cpass.set_bind_group(1, &proto_g1, &[]);
        cpass.dispatch_workgroups(bricks_per_axis, bricks_per_axis, bricks_per_axis);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");

    // ── Build emit-pass region request: one painted leaf at (2, 0.5, 0.5) ──
    //
    // The single leaf sits at the region center; the emit hook will see
    // `host_pos = (2.0, 0.5, 0.5)` and set `pos` to that, scattering one
    // instance into the per-region pool. Region AABB is unchanged for
    // the march's ray-AABB pre-cull, but the emit dispatch is driven
    // entirely by the leaf list now.
    let region_aabb_min = [1.5_f32, 0.0, 0.0];
    let region_aabb_max = [2.5_f32, 1.0, 1.0];
    let cell_size = 0.25_f32;
    let leaves = vec![PaintedLeaf {
        world_pos: [2.0, 0.5, 0.5],
        material_packed: 5,
        world_normal: [0.0, 1.0, 0.0],
        _pad: 0.0,
    }];
    let request = InstanceRegionRequest {
        host_object_id: 1,
        material_id: 5,
        shader_name: "sphereinst".to_string(),
        params: vec![],
        aabb_min: region_aabb_min,
        aabb_max: region_aabb_max,
        cell_size,
        input_hash: 0,
        animated: false,
        region_thickness: 0.0,
        tile_index: NO_TILE,
        stride_u32: STRIDE_U32,
        max_instances: 64,
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
    let leaf_count = leaves.len() as u32;

    let mut instance_cache = InstanceRegionCache::with_capacity(64 * STRIDE_U32 * 2);
    instance_cache.set_pool_base(POOL_INSTANCE_BASE);
    let cached_slot = instance_cache.lookup_or_allocate(&request, 0xAA, 0xBB).unwrap();

    // ── Run the emit pass ──
    let mut emit_pass = EmitPass::new(&device);
    emit_pass.reload_user_shaders(&device, &chunks.emit, registry.source_hash());

    let instance_pool_bytes =
        ((cached_slot.instance_block_offset + cached_slot.instance_extent_u32) as u64 + 4) * 4;
    let instance_pool = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e instance pool"),
        size: instance_pool_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    queue.write_buffer(&emit_pass.instance_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&emit_pass.overflow_buffer, 0, &[0u8; 16]);

    let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e leaves"),
        size: (leaves.len() as u64) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&leaves_buffer, 0, bytemuck::cast_slice(&leaves));

    let region_uniform =
        build_emit_region_uniform(&request, &cached_slot, /* shader_id */ SHADER_ID, /* time */ 0.0, 0);
    queue.write_buffer(&emit_pass.regions_buffer, 0, bytemuck::bytes_of(&region_uniform));
    let dispatch_u = EmitDispatchUniform {
        region_index: 0,
        leaf_count,
        _pad0: 0,
        _pad1: 0,
    };
    queue.write_buffer(&emit_pass.dispatch_uniforms_buffer, 0, bytemuck::bytes_of(&dispatch_u));

    let emit_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("emit g0"),
        layout: &emit_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: emit_pass.instance_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaves_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: emit_pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let emit_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("emit g1"),
        layout: &emit_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: emit_pass.regions_buffer.as_entire_binding(),
        }],
    });
    let emit_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("emit g2"),
        layout: &emit_pass.group2_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &emit_pass.dispatch_uniforms_buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<EmitDispatchUniform>() as u64),
            }),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&emit_pass.emit_pipeline);
        cpass.set_bind_group(0, &emit_g0, &[]);
        cpass.set_bind_group(1, &emit_g1, &[]);
        cpass.set_bind_group(2, &emit_g2, &[0u32]);
        cpass.dispatch_workgroups(workgroups_for_leaf_count(leaf_count), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");

    // ── Build flat GPU TileIndex (single-entry) ──
    let mut tib = TileIndexBuilder::new();
    tib.add_request(&request, /* region_index */ 0).unwrap();
    let tile_index = tib.build();
    let flat_tile_entries = flatten_tile_index(&tile_index);
    assert_eq!(flat_tile_entries.len(), 1);

    let tile_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e tile_index"),
        size: (flat_tile_entries.len() * std::mem::size_of::<GpuTileIndexEntry>()).max(32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &tile_index_buffer,
        0,
        bytemuck::cast_slice(&flat_tile_entries),
    );

    // ── Build proto-lookup buffer ──
    let proto_lookup = flatten_prototype_lookup(registry_entries, &proto_cache).unwrap();
    assert_eq!(proto_lookup.entries.len(), 1);
    assert_eq!(proto_lookup.entries[0].shader_id, SHADER_ID);
    let proto_lookup_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e proto_lookup"),
        size: (proto_lookup.entries.len() * std::mem::size_of::<GpuPrototypeEntry>()).max(32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &proto_lookup_buffer,
        0,
        bytemuck::cast_slice(&proto_lookup.entries),
    );

    // ── March uniforms + camera + output ──
    let march_pass = InstanceMarchPass::new(&device);

    // 1×1 screen — center pixel gets `ndc = (0, 0)` → ray direction =
    // `normalize(forward) = +X`. Ray origin = camera position.
    let screen_width = 1u32;
    let screen_height = 1u32;
    let mut uniforms = MarchUniforms::default();
    uniforms.tile_index_count = flat_tile_entries.len() as u32;
    uniforms.proto_lookup_count = proto_lookup.entries.len() as u32;
    uniforms.screen_width = screen_width;
    uniforms.screen_height = screen_height;

    let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e march uniforms"),
        size: std::mem::size_of::<MarchUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

    // Camera fires its single pixel-center ray from (0, 0.5, 0.5)
    // along +X. Hits the instance AABB at world_t ≈ 1.5, descends
    // the canonical sphere — first cell hit ~ canonical x = 0.1 →
    // world_t ~ 1.6. (Same target as the Stage 5b version.)
    let camera = MarchCameraUniform {
        position: [0.0, 0.5, 0.5, 1.0],
        forward: [1.0, 0.0, 0.0, 0.0],
        // right + up only matter for off-center pixels; pick any
        // orthonormal pair so jitter=0 + 1×1 res still resolves to
        // pure forward.
        right: [0.0, 0.0, 1.0, 0.0],
        up: [0.0, 1.0, 0.0, 0.0],
        resolution: [screen_width as f32, screen_height as f32],
        jitter: [0.0, 0.0],
    };
    let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e march camera"),
        size: std::mem::size_of::<MarchCameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

    let pixel_count = (screen_width * screen_height) as u64;
    let output_size = (pixel_count * std::mem::size_of::<InstanceMarchHit>() as u64).max(48);
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e march output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &output_buf,
        0,
        bytemuck::bytes_of(&InstanceMarchHit::default()),
    );

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g0"),
        layout: &march_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g1"),
        layout: &march_pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: emit_pass.regions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_index_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: emit_pass.instance_alloc_buffer.as_entire_binding() },
        ],
    });
    let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g2"),
        layout: &march_pass.group2_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: proto_lookup_buffer.as_entire_binding() }],
    });
    let g3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g3"),
        layout: &march_pass.group3_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        cpass.set_bind_group(2, &g2, &[]);
        cpass.set_bind_group(3, &g3, &[]);
        march_pass.dispatch_per_pixel(&mut cpass, screen_width, screen_height);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e output staging"),
        size: std::mem::size_of::<InstanceMarchHit>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &output_buf, 0, &staging, 0,
        std::mem::size_of::<InstanceMarchHit>() as u64,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let hit: InstanceMarchHit = *bytemuck::from_bytes(&view[..]);
    drop(view);
    staging.unmap();

    assert_eq!(hit.hit, 1, "march did not register a hit on the placed instance: {hit:?}");
    assert_eq!(hit.region_index, 0);
    assert_eq!(hit.instance_index, 0);
    assert!(
        hit.t_world > 1.5 && hit.t_world < 1.95,
        "world hit t out of plausible range: t_world = {} (instance AABB enters at 1.5, sphere surface near 1.6)",
        hit.t_world,
    );
    assert!(
        hit.normal[0] < -0.3,
        "hit normal should mostly face -X (back toward the ray); normal = {:?}",
        hit.normal,
    );
}

/// Quick check that the composed source string is produced consistently.
/// Catches drift in the helpers/main concat ordering that would silently
/// break the GPU pipeline.
#[test]
fn march_main_source_includes_helpers_and_entry() {
    let src = instance_march_main_source();
    assert!(src.contains("fn inst_ray_aabb_intersect"));
    assert!(src.contains("fn inst_proto_descend"));
    assert!(src.contains("fn instance_march_main"));
}

// Reference-imported items the body uses. Without these, build fails
// to inform "you removed this from the public API by accident."
#[allow(dead_code)]
fn _api_surface_used(
    _r: EmitRegionUniform,
    _g: GpuTileIndexEntry,
    _p: GpuPrototypeEntry,
    _u: MarchUniforms,
    _cam: MarchCameraUniform,
    _hit: InstanceMarchHit,
) {
}
