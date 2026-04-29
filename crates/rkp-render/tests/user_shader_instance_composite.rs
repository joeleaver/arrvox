//! Stage 6b synthetic end-to-end test for the instance composite pass.
//!
//! Doesn't drive the bake/scatter/march pipelines — just synthesises
//! `InstanceMarchHit` records, populates stub host G-buffer textures,
//! runs `InstanceCompositePass`, and reads back the merged G-buffer to
//! verify per-pixel selection + format packing.
//!
//! 4×1 row covering four scenarios in one dispatch:
//!   - Pixel 0: instance hit, `t_world` beats host depth → instance wins.
//!   - Pixel 1: instance hit, `t_world` LOSES to host depth → host wins.
//!   - Pixel 2: no instance hit → host wins (passthrough).
//!   - Pixel 3: instance hit, `t_world` < host depth → instance wins
//!     (but with a different `material_packed`, so the test verifies the
//!     pack formula against a second value).
//!
//! Skips silently when no wgpu adapter is available.

use rkp_render::gbuffer::{
    GBUFFER_LEAF_SLOT_FORMAT, GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT,
    GBUFFER_POSITION_FORMAT,
};
use rkp_render::instance_composite_pass::InstanceCompositePass;
use rkp_render::instance_march_pass::{
    InstanceMarchHit, MarchCameraUniform, MarchUniforms,
};

const W: u32 = 4;
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
        label: Some("instance_composite test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

fn make_texture(
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

/// Reciprocal-length normalize on three floats (avoids pulling glam in
/// just for a 3-element op).
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn camera_dir_for_pixel(cam: &MarchCameraUniform, pixel_x: u32) -> [f32; 3] {
    let uv_x = (pixel_x as f32 + 0.5 + cam.jitter[0]) / cam.resolution[0];
    let uv_y = (0.0 + 0.5 + cam.jitter[1]) / cam.resolution[1];
    let ndc_x = uv_x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv_y * 2.0;
    let dir = [
        cam.forward[0] + ndc_x * cam.right[0] + ndc_y * cam.up[0],
        cam.forward[1] + ndc_x * cam.right[1] + ndc_y * cam.up[1],
        cam.forward[2] + ndc_x * cam.right[2] + ndc_y * cam.up[2],
    ];
    normalize3(dir)
}

#[test]
fn composite_overlays_instance_hits_and_passes_host_through() {
    let Some((device, queue)) = create_device() else {
        eprintln!("[composite e2e] no wgpu adapter — skipping");
        return;
    };

    let pass = InstanceCompositePass::new(&device);

    // ── Synthetic InstanceMarchHit array ────────────────────────────
    //
    // Pixel 0: hit at t=5, host depth=10 → instance wins.
    // Pixel 1: hit at t=20, host depth=10 → host wins.
    // Pixel 2: no hit → host wins (passthrough).
    // Pixel 3: hit at t=2, host depth=10 → instance wins (second
    //          material to exercise the pack formula at a different
    //          value).
    let hit0_normal = [0.0_f32, 0.0, 1.0];
    let hit3_normal = [0.0_f32, 1.0, 0.0];
    // material_packed bits: pri | (sec << 16) | (bw << 28)
    let mat0_packed: u32 = 0x0000_0123 | (0x0123 << 16) | (0x5 << 28);
    let mat3_packed: u32 = 0x0000_0042 | (0x0042 << 16) | (0xA << 28);

    let hits: [InstanceMarchHit; 4] = [
        InstanceMarchHit {
            hit: 1,
            region_index: 0,
            instance_index: 7,
            leaf_attr_slot: 11,
            t_world: 5.0,
            material_packed: mat0_packed,
            _pad0: 0, _pad1: 0,
            normal: hit0_normal,
            _pad2: 0.0,
        },
        InstanceMarchHit {
            hit: 1,
            region_index: 0,
            instance_index: 0,
            leaf_attr_slot: 99,
            t_world: 20.0,
            material_packed: 0xFFFF_FFFF, // shouldn't be written
            _pad0: 0, _pad1: 0,
            normal: [1.0, 0.0, 0.0],
            _pad2: 0.0,
        },
        InstanceMarchHit::default(), // hit=0
        InstanceMarchHit {
            hit: 1,
            region_index: 1,
            instance_index: 3,
            leaf_attr_slot: 22,
            t_world: 2.0,
            material_packed: mat3_packed,
            _pad0: 0, _pad1: 0,
            normal: hit3_normal,
            _pad2: 0.0,
        },
    ];
    let hits_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e hits"),
        size: (hits.len() * std::mem::size_of::<InstanceMarchHit>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&hits_buf, 0, bytemuck::cast_slice(&hits));

    // ── March uniforms ───────────────────────────────────────────────
    let mut uniforms = MarchUniforms::default();
    uniforms.screen_width = W;
    uniforms.screen_height = H;
    let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e uniforms"),
        size: std::mem::size_of::<MarchUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

    // ── Camera: looking down -Z, 4×1 resolution ─────────────────────
    let camera = MarchCameraUniform {
        position: [0.0, 0.0, 0.0, 1.0],
        forward: [0.0, 0.0, -1.0, 0.0],
        right: [1.0, 0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0, 0.0],
        resolution: [W as f32, H as f32],
        jitter: [0.0, 0.0],
    };
    let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("e2e camera"),
        size: std::mem::size_of::<MarchCameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

    // ── Stub host G-buffer ──────────────────────────────────────────
    //
    // depth=10 everywhere, normal=(0,1,0), material payload distinct
    // per pixel so the passthrough path can be verified positionally.
    let host_position_tex = make_texture(&device, "host position", GBUFFER_POSITION_FORMAT);
    let host_normal_tex = make_texture(&device, "host normal", GBUFFER_NORMAL_FORMAT);
    let host_material_tex = make_texture(&device, "host material", GBUFFER_MATERIAL_FORMAT);
    let host_leaf_slot_tex = make_texture(&device, "host leaf_slot", GBUFFER_LEAF_SLOT_FORMAT);

    // Position rgba32f: (px, 0, 0, 10) per pixel.
    let mut host_pos_data: Vec<f32> = Vec::with_capacity((W * H * 4) as usize);
    for px in 0..W {
        host_pos_data.extend_from_slice(&[px as f32 + 100.0, 0.0, 0.0, 10.0]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_position_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_pos_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 16),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Normal rgba16f: (0, 1, 0, 1) — packed half-float.
    fn f32_to_half(f: f32) -> u16 {
        // Tiny half-float encoder for the constants we use (-1..1).
        let bits = f.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let mantissa = bits & 0x007F_FFFF;
        let exp = ((bits >> 23) & 0xFF) as i32;
        if exp == 0 { return sign as u16; }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 { return sign as u16; }
        if new_exp >= 0x1F { return (sign | 0x7C00) as u16; }
        let new_mant = (mantissa >> 13) as u32;
        (sign | ((new_exp as u32) << 10) | new_mant) as u16
    }
    let mut host_normal_data: Vec<u16> = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..W {
        host_normal_data.extend_from_slice(&[
            f32_to_half(0.0), f32_to_half(1.0), f32_to_half(0.0), f32_to_half(1.0),
        ]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_normal_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_normal_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Material rg32u: per-pixel sentinel so passthrough can be checked
    // by-pixel.
    let host_material_data: Vec<u32> = (0..W).flat_map(|px| [0xAA00 + px, 0xBB00 + px]).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_material_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_material_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Leaf_slot r32u: per-pixel sentinel.
    let host_leaf_slot_data: Vec<u32> = (0..W).map(|px| 0xCC00 + px).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &host_leaf_slot_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&host_leaf_slot_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 4),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // ── Merged (output) G-buffer ────────────────────────────────────
    let merged_position_tex = make_texture(&device, "merged position", GBUFFER_POSITION_FORMAT);
    let merged_normal_tex = make_texture(&device, "merged normal", GBUFFER_NORMAL_FORMAT);
    let merged_material_tex = make_texture(&device, "merged material", GBUFFER_MATERIAL_FORMAT);
    let merged_leaf_slot_tex = make_texture(&device, "merged leaf_slot", GBUFFER_LEAF_SLOT_FORMAT);

    let host_pos_view = host_position_tex.create_view(&Default::default());
    let host_normal_view = host_normal_tex.create_view(&Default::default());
    let host_material_view = host_material_tex.create_view(&Default::default());
    let host_leaf_slot_view = host_leaf_slot_tex.create_view(&Default::default());

    let merged_pos_view = merged_position_tex.create_view(&Default::default());
    let merged_normal_view = merged_normal_tex.create_view(&Default::default());
    let merged_material_view = merged_material_tex.create_view(&Default::default());
    let merged_leaf_slot_view = merged_leaf_slot_tex.create_view(&Default::default());

    let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g0"),
        layout: &pass.group0_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: hits_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: uniforms_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
        ],
    });
    let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g1 (host gbuf)"),
        layout: &pass.group1_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&host_pos_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&host_normal_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&host_material_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&host_leaf_slot_view) },
        ],
    });
    let g2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite g2 (merged gbuf)"),
        layout: &pass.group2_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&merged_pos_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&merged_normal_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&merged_material_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&merged_leaf_slot_view) },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_bind_group(0, &g0, &[]);
        cpass.set_bind_group(1, &g1, &[]);
        cpass.set_bind_group(2, &g2, &[]);
        pass.dispatch_per_pixel(&mut cpass, W, H);
    }

    // Stage merged textures for readback.
    let pos_bytes = (W * H * 16) as u64;
    let normal_bytes = (W * H * 8) as u64;
    let material_bytes = (W * H * 8) as u64;
    let leaf_slot_bytes = (W * H * 4) as u64;

    let pos_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: pos_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let normal_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: normal_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let material_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: material_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let leaf_slot_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: leaf_slot_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    fn copy_tex_to_buf(
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Buffer,
        bytes_per_row: u32,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: dst,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row.max(256)),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
    }
    copy_tex_to_buf(&mut encoder, &merged_position_tex, &pos_staging, W * 16);
    copy_tex_to_buf(&mut encoder, &merged_normal_tex, &normal_staging, W * 8);
    copy_tex_to_buf(&mut encoder, &merged_material_tex, &material_staging, W * 8);
    copy_tex_to_buf(&mut encoder, &merged_leaf_slot_tex, &leaf_slot_staging, W * 4);

    queue.submit(std::iter::once(encoder.finish()));

    fn map_read<'a>(buf: &'a wgpu::Buffer, device: &wgpu::Device) -> wgpu::BufferView {
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        slice.get_mapped_range()
    }

    let pos_view = map_read(&pos_staging, &device);
    let normal_view = map_read(&normal_staging, &device);
    let material_view = map_read(&material_staging, &device);
    let leaf_slot_view = map_read(&leaf_slot_staging, &device);

    let pos_floats: &[f32] = bytemuck::cast_slice(&pos_view);
    let normal_halves: &[u16] = bytemuck::cast_slice(&normal_view);
    let material_words: &[u32] = bytemuck::cast_slice(&material_view);
    let leaf_slot_words: &[u32] = bytemuck::cast_slice(&leaf_slot_view);

    // ── Expected values ─────────────────────────────────────────────
    fn pack_instance_material(material_packed: u32) -> (u32, u32) {
        let r = material_packed & 0x0FFF_FFFFu32;
        let blend4 = (material_packed >> 28) & 0x0F;
        let blend8 = (blend4 << 4) | blend4;
        (r, blend8)
    }

    // Pixel 0 — instance wins.
    {
        let dir = camera_dir_for_pixel(&camera, 0);
        let expected_pos = [dir[0] * 5.0, dir[1] * 5.0, dir[2] * 5.0, 5.0];
        let got_pos = &pos_floats[0..4];
        for i in 0..4 {
            assert!(
                (got_pos[i] - expected_pos[i]).abs() < 1e-3,
                "pixel 0 position[{i}] = {} expected {}", got_pos[i], expected_pos[i],
            );
        }
        let (rr, gg) = pack_instance_material(mat0_packed);
        assert_eq!(material_words[0], rr, "pixel 0 material r");
        assert_eq!(material_words[1], gg, "pixel 0 material g");
        assert_eq!(leaf_slot_words[0], 11, "pixel 0 leaf_slot");
        // normal (0,0,1,1) — assert against half-float decode.
        // Just spot-check via direct equality on the encoded bits.
        assert_eq!(normal_halves[0], f32_to_half(0.0));
        assert_eq!(normal_halves[1], f32_to_half(0.0));
        assert_eq!(normal_halves[2], f32_to_half(1.0));
    }

    // Pixel 1 — host wins (instance t=20 > host depth=10).
    {
        let got_pos = &pos_floats[4..8];
        // Host position was (101, 0, 0, 10).
        assert_eq!(got_pos[0], 101.0);
        assert_eq!(got_pos[1], 0.0);
        assert_eq!(got_pos[2], 0.0);
        assert_eq!(got_pos[3], 10.0);
        // Host material = (0xAA01, 0xBB01).
        assert_eq!(material_words[2], 0xAA01);
        assert_eq!(material_words[3], 0xBB01);
        assert_eq!(leaf_slot_words[1], 0xCC01);
    }

    // Pixel 2 — no instance hit, passthrough.
    {
        let got_pos = &pos_floats[8..12];
        assert_eq!(got_pos[0], 102.0);
        assert_eq!(got_pos[3], 10.0);
        assert_eq!(material_words[4], 0xAA02);
        assert_eq!(material_words[5], 0xBB02);
        assert_eq!(leaf_slot_words[2], 0xCC02);
    }

    // Pixel 3 — instance wins (t=2 < host=10), second material.
    {
        let dir = camera_dir_for_pixel(&camera, 3);
        let expected_pos = [dir[0] * 2.0, dir[1] * 2.0, dir[2] * 2.0, 2.0];
        let got_pos = &pos_floats[12..16];
        for i in 0..4 {
            assert!(
                (got_pos[i] - expected_pos[i]).abs() < 1e-3,
                "pixel 3 position[{i}] = {} expected {}", got_pos[i], expected_pos[i],
            );
        }
        let (rr, gg) = pack_instance_material(mat3_packed);
        assert_eq!(material_words[6], rr, "pixel 3 material r");
        assert_eq!(material_words[7], gg, "pixel 3 material g");
        assert_eq!(leaf_slot_words[3], 22);
    }

    drop(pos_view);
    drop(normal_view);
    drop(material_view);
    drop(leaf_slot_view);
    pos_staging.unmap();
    normal_staging.unmap();
    material_staging.unmap();
    leaf_slot_staging.unmap();
}
