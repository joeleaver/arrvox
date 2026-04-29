//! Stage 6c-3.5a — empty-case smoke test for the per-frame
//! march+composite dispatch sequence that
//! `ViewportRenderer::dispatch_instance_overlay` runs.
//!
//! Doesn't construct a `ViewportRenderer` (which needs the full
//! renderer harness). Instead, exercises the same shape of bind-group
//! construction + dispatch sequence with the empty input case
//! (`tile_index_count=0`, `proto_lookup_count=0`):
//!
//!   1. March runs for all pixels, sees zero tiles → writes hit=0
//!      to every `output_hits` slot.
//!   2. Composite runs, sees hit=0 everywhere → writes host
//!      passthrough into the merged G-buffer.
//!
//! Validates: the bind-group LAYOUTS the dispatch wiring binds
//! against work end-to-end on a real GPU with the empty inputs an
//! editor will see before any instance shaders are registered.
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
use rkp_render::instance_merged_gbuffer::InstanceMergedGBuffer;

const W: u32 = 1;
const H: u32 = 1;

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
        label: Some("instance_overlay_empty test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            // March pipeline binds 9 storage buffers per stage (3 pool
            // + 4 instance state + 1 proto lookup + 1 output_hits);
            // default cap is 8.
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

fn make_buf(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn make_tex(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
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
fn empty_case_dispatch_sequence_runs_without_panic() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[overlay empty] no wgpu adapter — skipping");
        return;
    };

    let march_pass = InstanceMarchPass::new(&device);
    let composite_pass = InstanceCompositePass::new(&device);
    let merged = InstanceMergedGBuffer::new(&device, W, H);

    // ── Per-VR resources mirroring `ViewportRenderer` ────────────────
    let camera_buf = make_buf(
        &device, "camera", 80,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let camera = MarchCameraUniform {
        position: [0.0, 0.0, 0.0, 1.0],
        forward: [0.0, 0.0, -1.0, 0.0],
        right: [1.0, 0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0, 0.0],
        resolution: [W as f32, H as f32],
        jitter: [0.0, 0.0],
    };
    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

    let uniforms_buf = make_buf(
        &device, "uniforms", std::mem::size_of::<MarchUniforms>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let mut uniforms = MarchUniforms::default();
    uniforms.tile_index_count = 0;
    uniforms.proto_lookup_count = 0;
    uniforms.screen_width = W;
    uniforms.screen_height = H;
    queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

    let output_hits_buf = make_buf(
        &device, "output_hits",
        (W * H) as u64 * std::mem::size_of::<InstanceMarchHit>() as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    // Pre-fill with non-zero garbage so we can prove the march wrote hit=0.
    let garbage = InstanceMarchHit {
        hit: 0xDEAD_BEEF, region_index: 1, instance_index: 1, leaf_attr_slot: 1,
        t_world: 999.0, material_packed: 0xFFFF_FFFF, _pad0: 0, _pad1: 0,
        normal: [1.0, 2.0, 3.0], _pad2: 0.0,
    };
    queue.write_buffer(&output_hits_buf, 0, bytemuck::bytes_of(&garbage));

    // ── Empty engine-side buffers ────────────────────────────────────
    //
    // Each at minimum binding size for its layout. Pool buffers carry
    // arbitrary data — empty tile_index means the march never reads
    // them.
    let octree_buf = make_buf(&device, "octree", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let brick_buf = make_buf(&device, "brick", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let leaf_attr_buf = make_buf(&device, "leaf_attr", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

    // Regions buffer: at least one EmitRegionUniform-sized slot so the
    // binding's min_binding_size = 192 B is satisfied.
    let regions_buf = make_buf(&device, "regions", 256,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let instance_pool_buf = make_buf(&device, "instance_pool", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let instance_alloc_buf = make_buf(&device, "instance_alloc", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let tile_index_buf = make_buf(&device, "tile_index", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let proto_lookup_buf = make_buf(&device, "proto_lookup", 64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);

    // ── Host G-buffer (sentinel pos.w = 10.0 → composite passthrough wins) ──
    let host_pos = make_tex(&device, "host_pos", GBUFFER_POSITION_FORMAT);
    let host_normal = make_tex(&device, "host_normal", GBUFFER_NORMAL_FORMAT);
    let host_material = make_tex(&device, "host_material", GBUFFER_MATERIAL_FORMAT);
    let host_leaf_slot = make_tex(&device, "host_leaf_slot", GBUFFER_LEAF_SLOT_FORMAT);

    let host_pos_data: [f32; 4] = [42.0, 43.0, 44.0, 10.0];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_pos, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        bytemuck::bytes_of(&host_pos_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(16), rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let host_material_data: [u32; 2] = [0xAAAA, 0xBBBB];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_material, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_material_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let host_leaf_slot_data: [u32; 1] = [0xCCCC];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_leaf_slot, mip_level: 0,
            origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_leaf_slot_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Views for bind groups.
    let host_pos_v = host_pos.create_view(&Default::default());
    let host_normal_v = host_normal.create_view(&Default::default());
    let host_material_v = host_material.create_view(&Default::default());
    let host_leaf_slot_v = host_leaf_slot.create_view(&Default::default());
    let merged_pos_v = merged.position_view;
    let merged_normal_v = merged.normal_view;
    let merged_material_v = merged.material_view;
    let merged_leaf_slot_v = merged.leaf_slot_view;

    // ── Build march + composite bind groups (mirror dispatch_instance_overlay) ──
    let m_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst march g0"),
        layout: &march_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: octree_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: brick_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: leaf_attr_buf.as_entire_binding() },
        ],
    });
    let m_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst march g1"),
        layout: &march_pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: regions_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: instance_pool_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tile_index_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: instance_alloc_buf.as_entire_binding() },
        ],
    });
    let m_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst march g2"),
        layout: &march_pass.group2_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: proto_lookup_buf.as_entire_binding(),
        }],
    });
    let m_g3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst march g3"),
        layout: &march_pass.group3_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_hits_buf.as_entire_binding() },
        ],
    });

    let c_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst composite g0"),
        layout: &composite_pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: output_hits_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
        ],
    });
    let c_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst composite g1"),
        layout: &composite_pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&host_pos_v) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&host_normal_v) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&host_material_v) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&host_leaf_slot_v) },
        ],
    });
    let c_g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test inst composite g2"),
        layout: &composite_pass.group2_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&merged_pos_v) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&merged_normal_v) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&merged_material_v) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&merged_leaf_slot_v) },
        ],
    });

    // ── Encode + dispatch ────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &m_g0, &[]);
        cpass.set_bind_group(1, &m_g1, &[]);
        cpass.set_bind_group(2, &m_g2, &[]);
        cpass.set_bind_group(3, &m_g3, &[]);
        march_pass.dispatch_per_pixel(&mut cpass, W, H);
    }
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &c_g0, &[]);
        cpass.set_bind_group(1, &c_g1, &[]);
        cpass.set_bind_group(2, &c_g2, &[]);
        composite_pass.dispatch_per_pixel(&mut cpass, W, H);
    }

    // Stage merged_pos for readback (16 B is the only thing we need to
    // verify — it should equal the host's 16 B, proving passthrough).
    let pos_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos staging"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &merged.position_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &pos_staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0, bytes_per_row: Some(256), rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Stage output_hits[0] for readback (verify hit=0).
    let hits_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits staging"),
        size: std::mem::size_of::<InstanceMarchHit>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(
        &output_hits_buf, 0, &hits_staging, 0,
        std::mem::size_of::<InstanceMarchHit>() as u64,
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Read back hits[0].
    let slice = hits_staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let view = slice.get_mapped_range();
    let hit: InstanceMarchHit = *bytemuck::from_bytes(&view[..]);
    drop(view);
    hits_staging.unmap();

    assert_eq!(
        hit.hit, 0,
        "march should write hit=0 with empty inputs; got {hit:?}",
    );

    // Read back merged_pos[0] — should equal host's [42, 43, 44, 10].
    let slice = pos_staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let pos_floats: [f32; 4] = {
        let view = slice.get_mapped_range();
        let s: &[f32] = bytemuck::cast_slice(&view[..16]);
        [s[0], s[1], s[2], s[3]]
    };
    pos_staging.unmap();

    assert_eq!(
        pos_floats, [42.0, 43.0, 44.0, 10.0],
        "composite should pass host position through when there's no instance hit",
    );
}
