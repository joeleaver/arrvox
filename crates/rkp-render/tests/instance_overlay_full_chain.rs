//! Stage 6e prep — full-chain GPU integration test.
//!
//! Until this test, the Option B pipeline pieces were only validated
//! individually:
//!   - Stage 2: bake produces correct counts (CPU reference comparison).
//!   - Stage 3: scatter emits instances at expected positions.
//!   - Stage 5b/6a: bake + scatter + march hits the placed instance.
//!   - Stage 6b: composite (synthetic InstanceMarchHit input).
//!   - Stage 6c-3.5a: empty-case dispatch sequence.
//!
//! What this test adds: chain ALL the real passes together — bake →
//! scatter → march → composite — and verify the per-pixel
//! `instance_merged_gbuffer.position` carries the instance's hit data,
//! NOT the synthetic host G-buffer.
//!
//! This is the closest CPU-driven analogue to Stage 6e's "load grass
//! into a project, paint it, observe blades" interactive flow, and it
//! catches the most likely failure mode (wiring drift between bake
//! and composite halves) without needing the editor.
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::gbuffer::{
    GBUFFER_LEAF_SLOT_FORMAT, GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT,
    GBUFFER_POSITION_FORMAT,
};
use rkp_render::instance_composite_pass::InstanceCompositePass;
use rkp_render::instance_march_pass::{
    InstanceMarchHit, InstanceMarchPass, MarchCameraUniform, MarchUniforms,
};
use rkp_render::instance_proto_lookup::flatten_prototype_lookup;
use rkp_render::instance_tile_index::TileIndexBuilder;
use rkp_render::instance_tile_index_gpu::flatten_tile_index;
use rkp_render::shader_composer::{compose, scan_dir};
use rkp_render::user_shader_emit_pass::{
    build_emit_region_uniform, workgroups_for_leaf_count, EmitDispatchUniform, EmitPass,
    PaintedLeaf,
    InstanceRegionCache, InstanceRegionRequest, HOST_NO_HOST_SENTINEL, NO_TILE,
};
use rkp_render::user_shader_proto_pass::{
    build_internal_levels, PrototypeBakePass, PrototypeCache, PrototypeUniform,
};

const PROTO_MAX_DEPTH: u32 = 2;
const SHADER_ID: u32 = 1;
const SOURCE_HASH: u64 = 0xC0FFEE_CAFE_BEEF;
const POOL_OCTREE_BASE: u32 = 0;
const POOL_BRICK_BASE: u32 = 0;
const POOL_LEAF_ATTR_BASE: u32 = 0;
const POOL_INSTANCE_BASE: u32 = 0;
const STRIDE_U32: u32 = 4; // pos (3) + scale (1) = 4 u32 = 16 B

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
        v.material_primary = 0x0123u;
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
        label: Some("instance_overlay_full_chain test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            // March binds 9 storage buffers per stage (3 pool + 4
            // instance state + 1 proto lookup + 1 output_hits).
            // Default cap is 8.
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
        "rkp_instance_overlay_full_chain_{name}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.wgsl"));
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    dir
}

fn make_gbuffer_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[test]
fn full_chain_bake_scatter_march_composite_lands_instance_in_merged_gbuffer() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[full_chain] no wgpu adapter — skipping");
        return;
    };

    // ── Compose user shaders ──────────────────────────────────────
    let dir = write_temp_shader("sphereinst", SHADER_SRC);
    let registry = scan_dir(&dir).unwrap();
    let chunks = compose(&registry);
    let registry_entries = registry.entries();
    assert_eq!(registry_entries.len(), 1);
    assert!(registry_entries[0].is_instance_pipeline());

    // ── Pool buffers ──────────────────────────────────────────────
    let octree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc octree"),
        size: 8 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let brick_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc bricks"),
        size: 256 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaf_attr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc leaf_attrs"),
        size: 64 * 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Bake sphere prototype ────────────────────────────────────
    let mut proto_pass = PrototypeBakePass::new(&device);
    proto_pass.reload_user_shaders(&device, &chunks.proto, registry.source_hash());

    let mut proto_cache = PrototypeCache::with_capacities(1024, 1024, 8192);
    proto_cache.set_pool_bases(POOL_OCTREE_BASE, POOL_BRICK_BASE, POOL_LEAF_ATTR_BASE);
    let (proto_entry, _) = proto_cache
        .lookup_or_allocate(SHADER_ID, SOURCE_HASH, PROTO_MAX_DEPTH)
        .unwrap();
    let proto_uniform = PrototypeUniform::from_entry(&proto_entry, &proto_cache);
    let internal = build_internal_levels(POOL_OCTREE_BASE, proto_entry.octree_extent.0, PROTO_MAX_DEPTH);
    let mut octree_init: Vec<u8> = Vec::with_capacity(internal.len() * 8);
    for [v0, v1] in internal {
        octree_init.extend_from_slice(&v0.to_le_bytes());
        octree_init.extend_from_slice(&v1.to_le_bytes());
    }
    queue.write_buffer(&octree_buffer, (proto_entry.octree_extent.0 as u64) * 8, &octree_init);
    queue.write_buffer(&proto_pass.proto_brick_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&proto_pass.proto_leaf_attr_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&proto_pass.overflow_buffer, 0, &[0u8; 12 * 4]);
    queue.write_buffer(&proto_pass.proto_uniform_buffer, 0, bytemuck::bytes_of(&proto_uniform));

    let proto_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake g0"),
        layout: &proto_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: proto_pass.proto_brick_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: proto_pass.proto_leaf_attr_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: proto_pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let proto_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake g1"),
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

    // ── Scatter one instance at the region center via a single leaf ──
    let region_aabb_min = [1.5_f32, 0.0, 0.0];
    let region_aabb_max = [2.5_f32, 1.0, 1.0];
    let cell_size = 0.25_f32;
    let leaves = vec![PaintedLeaf {
        world_pos: [2.0, 0.5, 0.5],
        material_packed: 5,
        world_normal: [0.0, 1.0, 0.0],
        _pad: 0.0,
    }];
    let leaf_count = leaves.len() as u32;
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

    let mut instance_cache = InstanceRegionCache::with_capacity(64 * STRIDE_U32 * 2);
    instance_cache.set_pool_base(POOL_INSTANCE_BASE);
    let cached_slot = instance_cache.lookup_or_allocate(&request, 0xAA, 0xBB).unwrap();

    let mut emit_pass = EmitPass::new(&device);
    emit_pass.reload_user_shaders(&device, &chunks.emit, registry.source_hash());

    let instance_pool_bytes =
        ((cached_slot.instance_block_offset + cached_slot.instance_extent_u32) as u64 + 4) * 4;
    let instance_pool = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc instance pool"),
        size: instance_pool_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&emit_pass.instance_alloc_buffer, 0, &[0u8; 4]);
    queue.write_buffer(&emit_pass.overflow_buffer, 0, &[0u8; 16]);

    let leaves_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc leaves"),
        size: (leaves.len() as u64) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&leaves_buffer, 0, bytemuck::cast_slice(&leaves));

    let region_uniform =
        build_emit_region_uniform(&request, &cached_slot, SHADER_ID, /* time */ 0.0, 0);
    queue.write_buffer(&emit_pass.regions_buffer, 0, bytemuck::bytes_of(&region_uniform));
    let dispatch_u = EmitDispatchUniform {
        region_index: 0,
        leaf_count,
        _pad0: 0,
        _pad1: 0,
    };
    queue.write_buffer(&emit_pass.dispatch_uniforms_buffer, 0, bytemuck::bytes_of(&dispatch_u));

    let emit_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter g0"),
        layout: &emit_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: emit_pass.instance_alloc_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaves_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: emit_pass.overflow_buffer.as_entire_binding() },
        ],
    });
    let emit_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter g1"),
        layout: &emit_pass.group1_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: emit_pass.regions_buffer.as_entire_binding(),
        }],
    });
    let emit_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter g2"),
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

    // ── TileIndex + ProtoLookup ──
    let mut tib = TileIndexBuilder::new();
    tib.add_request(&request, /* region_index */ 0).unwrap();
    let flat_tile_entries = flatten_tile_index(&tib.build());
    let tile_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc tile_index"),
        size: (flat_tile_entries.len() * 32).max(32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&tile_index_buffer, 0, bytemuck::cast_slice(&flat_tile_entries));

    let proto_lookup = flatten_prototype_lookup(registry_entries, &proto_cache).unwrap();
    let proto_lookup_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc proto_lookup"),
        size: (proto_lookup.entries.len() * 32).max(32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&proto_lookup_buffer, 0, bytemuck::cast_slice(&proto_lookup.entries));

    // ── March + composite at 1×1 ──
    let screen_width = 1u32;
    let screen_height = 1u32;
    let march_pass = InstanceMarchPass::new(&device);
    let composite_pass = InstanceCompositePass::new(&device);

    let uniforms = MarchUniforms {
        tile_index_count: flat_tile_entries.len() as u32,
        proto_lookup_count: proto_lookup.entries.len() as u32,
        screen_width,
        screen_height,
        ..MarchUniforms::default()
    };
    let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc march uniforms"),
        size: std::mem::size_of::<MarchUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

    let camera = MarchCameraUniform {
        position: [0.0, 0.5, 0.5, 1.0],
        forward: [1.0, 0.0, 0.0, 0.0],
        right: [0.0, 0.0, 1.0, 0.0],
        up: [0.0, 1.0, 0.0, 0.0],
        resolution: [screen_width as f32, screen_height as f32],
        jitter: [0.0, 0.0],
    };
    let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc camera"),
        size: std::mem::size_of::<MarchCameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

    let pixel_count = (screen_width * screen_height) as u64;
    let output_size = (pixel_count * std::mem::size_of::<InstanceMarchHit>() as u64).max(48);
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc output_hits"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // Pre-fill with zeros so untouched pixels read as `hit=0`.
    queue.write_buffer(&output_buf, 0, bytemuck::bytes_of(&InstanceMarchHit::default()));

    let m_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g0"),
        layout: &march_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buffer.as_entire_binding() },
        ],
    });
    let m_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g1"),
        layout: &march_pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: emit_pass.regions_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: instance_pool.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_index_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: emit_pass.instance_alloc_buffer.as_entire_binding() },
        ],
    });
    let m_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g2"),
        layout: &march_pass.group2_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: proto_lookup_buffer.as_entire_binding() }],
    });
    let m_g3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("march g3"),
        layout: &march_pass.group3_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
        ],
    });

    // ── Synthetic host G-buffer + merged G-buffer textures ──
    //
    // Host position.w = 100.0 (far away) so the instance always wins
    // the depth test inside the composite. Host normal/material/leaf
    // values are sentinels we'll later look for to confirm they got
    // OVERWRITTEN by the instance hit.
    let host_pos_tex = make_gbuffer_texture(&device, "fc host pos", GBUFFER_POSITION_FORMAT, screen_width, screen_height);
    let host_nor_tex = make_gbuffer_texture(&device, "fc host nor", GBUFFER_NORMAL_FORMAT, screen_width, screen_height);
    let host_mat_tex = make_gbuffer_texture(&device, "fc host mat", GBUFFER_MATERIAL_FORMAT, screen_width, screen_height);
    let host_leaf_tex = make_gbuffer_texture(&device, "fc host leaf", GBUFFER_LEAF_SLOT_FORMAT, screen_width, screen_height);
    let merged_pos_tex = make_gbuffer_texture(&device, "fc merged pos", GBUFFER_POSITION_FORMAT, screen_width, screen_height);
    let merged_nor_tex = make_gbuffer_texture(&device, "fc merged nor", GBUFFER_NORMAL_FORMAT, screen_width, screen_height);
    let merged_mat_tex = make_gbuffer_texture(&device, "fc merged mat", GBUFFER_MATERIAL_FORMAT, screen_width, screen_height);
    let merged_leaf_tex = make_gbuffer_texture(&device, "fc merged leaf", GBUFFER_LEAF_SLOT_FORMAT, screen_width, screen_height);

    let host_pos: [f32; 4] = [42.0, 43.0, 44.0, 100.0]; // depth = 100
    let host_nor: [u32; 2] = [0xDEAD_BEEF, 0];
    let host_mat: [u32; 2] = [0xCAFE_BABE, 0xF00D_FACE];
    let host_leaf: [u32; 1] = [0xFFFF_FFFE];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &host_pos_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&host_pos),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &host_nor_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&host_nor),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &host_mat_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&host_mat),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &host_leaf_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&host_leaf),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );

    let host_pos_view = host_pos_tex.create_view(&Default::default());
    let host_nor_view = host_nor_tex.create_view(&Default::default());
    let host_mat_view = host_mat_tex.create_view(&Default::default());
    let host_leaf_view = host_leaf_tex.create_view(&Default::default());
    let merged_pos_view = merged_pos_tex.create_view(&Default::default());
    let merged_nor_view = merged_nor_tex.create_view(&Default::default());
    let merged_mat_view = merged_mat_tex.create_view(&Default::default());
    let merged_leaf_view = merged_leaf_tex.create_view(&Default::default());

    let c_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g0"),
        layout: &composite_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
        ],
    });
    let c_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g1 (host reads)"),
        layout: &composite_pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&host_pos_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&host_nor_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&host_mat_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&host_leaf_view) },
        ],
    });
    let c_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g2 (merged writes)"),
        layout: &composite_pass.group2_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&merged_pos_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&merged_nor_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&merged_mat_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&merged_leaf_view) },
        ],
    });

    // ── Encode march + composite in one encoder + submit ──
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &m_g0, &[]);
        cpass.set_bind_group(1, &m_g1, &[]);
        cpass.set_bind_group(2, &m_g2, &[]);
        cpass.set_bind_group(3, &m_g3, &[]);
        march_pass.dispatch_per_pixel(&mut cpass, screen_width, screen_height);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &c_g0, &[]);
        cpass.set_bind_group(1, &c_g1, &[]);
        cpass.set_bind_group(2, &c_g2, &[]);
        composite_pass.dispatch_per_pixel(&mut cpass, screen_width, screen_height);
    }

    // Stage merged.position for readback.
    // bytes_per_row min is 256 in copy_texture_to_buffer.
    let row_bytes = 16u32; // 1 px × Rgba32Float = 16 B
    let padded_row = row_bytes.max(256);
    let pos_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc merged pos readback"),
        size: padded_row as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &merged_pos_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &pos_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );

    // Also stage the raw output_hits buffer so a failure can be
    // diagnosed: did the march not find the instance, or did the
    // composite not propagate it?
    let hits_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fc hits readback"),
        size: std::mem::size_of::<InstanceMarchHit>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &output_buf, 0, &hits_readback, 0,
        std::mem::size_of::<InstanceMarchHit>() as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    // ── Read back hits + merged.position ──
    let slice = hits_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let hit: InstanceMarchHit = *bytemuck::from_bytes(&view[..]);
    drop(view);
    hits_readback.unmap();
    assert_eq!(
        hit.hit, 1,
        "march did not register an instance hit at the only pixel — bake or scatter likely broken: {hit:?}"
    );
    assert!(
        hit.t_world > 1.5 && hit.t_world < 1.95,
        "instance t_world out of plausible range: {} (instance AABB enters at 1.5)",
        hit.t_world,
    );

    let slice = pos_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
    let view = slice.get_mapped_range();
    let merged_arr: [f32; 4] = [
        f32::from_le_bytes([view[0], view[1], view[2], view[3]]),
        f32::from_le_bytes([view[4], view[5], view[6], view[7]]),
        f32::from_le_bytes([view[8], view[9], view[10], view[11]]),
        f32::from_le_bytes([view[12], view[13], view[14], view[15]]),
    ];
    drop(view);
    pos_readback.unmap();

    // Composite re-derives world position from camera + ray + t_world.
    // For a +X-facing camera at (0, 0.5, 0.5), the recovered position
    // should be (t_world, 0.5, 0.5) +/- tiny FP noise.
    assert!(
        (merged_arr[0] - hit.t_world).abs() < 0.05,
        "merged.position.x ({}) should match the instance ray's t_world ({}) — composite probably didn't re-derive world pos correctly. Full merged_arr = {:?}",
        merged_arr[0], hit.t_world, merged_arr,
    );
    assert!(
        (merged_arr[1] - 0.5).abs() < 0.05,
        "merged.position.y ({}) should match the camera ray's y (0.5) — full merged_arr = {:?}",
        merged_arr[1], merged_arr,
    );
    assert!(
        (merged_arr[2] - 0.5).abs() < 0.05,
        "merged.position.z ({}) should match the camera ray's z (0.5) — full merged_arr = {:?}",
        merged_arr[2], merged_arr,
    );
    // depth field (.w) — composite should write hit.t_world into the
    // alpha slot for downstream depth-equivalent reads.
    assert!(
        (merged_arr[3] - hit.t_world).abs() < 0.05,
        "merged.position.w ({}) should be the instance's t_world ({}); host depth was 100 — full merged_arr = {:?}",
        merged_arr[3], hit.t_world, merged_arr,
    );

    // Negative invariant: merged.position MUST NOT be the host
    // sentinel `[42, 43, 44, 100]`. If it is, the instance lost the
    // depth test (host.position.w < hit.t_world) or the composite
    // didn't actually write.
    let host_sentinel = [42.0, 43.0, 44.0, 100.0];
    let same_as_host = (0..4).all(|i| (merged_arr[i] - host_sentinel[i]).abs() < 1e-3);
    assert!(
        !same_as_host,
        "merged.position is byte-identical to the host sentinel — composite didn't overlay the instance hit. merged_arr = {merged_arr:?}",
    );
}
