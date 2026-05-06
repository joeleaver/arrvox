//! End-to-end splat-rasterizer integration test (Phase B-2).
//!
//! Loads a real `.rkp`, extracts SplatVertex data, drives the full
//! `SplatPass` (the same pipeline the editor uses) over a 1920×1080
//! G-buffer, and reports GPU time. Optionally dumps the position +
//! normal targets to PPM for visual sanity (set
//! `RKP_SPLAT_DUMP_DIR=/some/dir`).
//!
//! Skipped by default — set `RKP_SPLAT_TEST_ASSET=/path/to/scene.rkp`.
//! Adapter must support `TIMESTAMP_QUERY` and the three Uint render
//! targets the splat fragment writes (R32Uint / Rg32Uint).
//!
//! Runs in isolation: no editor, no scene manager, no march. The
//! numbers reflect raw splat-rasterize GPU time for a single asset
//! with identity world matrix.

use std::io::BufReader;

use glam::{Mat4, Vec3};
use rkp_render::rkp_scene::CameraUniforms;
use rkp_render::splat_pass::{
    extract_splats_with_radius, SplatInstanceUniform, SplatPass, DISC_RADIUS_FACTOR,
};

const W: u32 = 1920;
const H: u32 = 1080;

// G-buffer formats — must match `crate::gbuffer` constants. Replicated
// here so the test can spell them out at the texture-creation site
// without depending on the (private?) re-exports.
const POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const GLASS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
const LEAF_SLOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("splat_render test device"),
        required_features: wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .ok()?;
    Some((device, queue))
}

#[test]
fn splat_renders_elephant_and_reports_gpu_time() {
    let Ok(asset_path) = std::env::var("RKP_SPLAT_TEST_ASSET") else {
        eprintln!("[splat_render] skipping — set RKP_SPLAT_TEST_ASSET");
        return;
    };
    let Some((device, queue)) = create_device() else {
        eprintln!("[splat_render] no wgpu adapter — skipping");
        return;
    };

    // ── 1. Load the asset ───────────────────────────────────────────
    let path = std::path::PathBuf::from(&asset_path);
    assert!(path.exists());
    let mut file = std::fs::File::open(&path).expect("open .rkp");
    let mut reader = BufReader::new(&mut file);

    let header = rkp_core::asset_file::read_rkp_header(&mut reader).expect("header");
    let octree_nodes =
        rkp_core::asset_file::read_rkp_octree(&mut reader, &header).expect("octree");
    let _voxels =
        rkp_core::asset_file::read_rkp_voxels(&mut reader, &header).expect("voxels");

    let normals_bytes = if header.flags & rkp_core::asset_file::FLAG_HAS_NORMALS != 0 {
        rkp_core::asset_file::read_rkp_normals(&mut reader, &header).expect("normals")
    } else {
        Vec::new()
    };
    let bricks_bytes = if header.flags & rkp_core::asset_file::FLAG_HAS_BRICKS != 0 {
        rkp_core::asset_file::read_rkp_bricks(&mut reader, &header).expect("bricks")
    } else {
        Vec::new()
    };
    let bricks: &[u32] = if !bricks_bytes.is_empty() {
        bytemuck::cast_slice(&bricks_bytes)
    } else {
        &[]
    };

    let aabb_min = Vec3::from(header.aabb_min);
    let aabb_max = Vec3::from(header.aabb_max);
    let extent = (1u32 << header.octree_depth as u8) as f32 * header.base_voxel_size;
    let grid_origin = (aabb_min + aabb_max) * 0.5 - Vec3::splat(extent * 0.5);

    // ── 2. Extract splats ───────────────────────────────────────────
    let radius_factor: f32 = std::env::var("RKP_SPLAT_RADIUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DISC_RADIUS_FACTOR);
    let extract_started = std::time::Instant::now();
    let splats = extract_splats_with_radius(
        &octree_nodes,
        header.octree_depth as u8,
        header.base_voxel_size,
        grid_origin,
        bricks,
        radius_factor,
    );
    let extract_ms = extract_started.elapsed().as_secs_f32() * 1000.0;
    eprintln!(
        "[splat_render] extracted {} splats in {extract_ms:.1} ms (radius_factor={radius_factor})",
        splats.len(),
    );

    // ── 3. Upload buffers ───────────────────────────────────────────
    use wgpu::util::DeviceExt;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat vertex buffer"),
        contents: bytemuck::cast_slice(&splats),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // LeafAttr pool — one entry per voxel slot, 8 B each (normal_oct +
    // material_packed). Synthesize from on-disk normals; default the
    // material packing to material_id 1.
    let normals_u32: &[u32] = if normals_bytes.len() >= 4 {
        bytemuck::cast_slice(&normals_bytes)
    } else {
        &[]
    };
    let leaf_attr_count = normals_u32.len();
    let mut leaf_attrs = Vec::with_capacity(leaf_attr_count);
    for &n in normals_u32 {
        leaf_attrs.push([n, 1u32]);
    }
    eprintln!(
        "[splat_render] leaf_attr_pool: {} entries ({} KB)",
        leaf_attr_count,
        leaf_attrs.len() * 8 / 1024,
    );
    let leaf_attr_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_attr_pool"),
        contents: bytemuck::cast_slice(&leaf_attrs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Materials — `GpuMaterial` is 24 × 4 = 96 B. One default plus a
    // few extra slots so material_id=1 maps to a defined entry.
    let mut materials = vec![[0u32; 24]; 8];
    for m in &mut materials {
        m[0] = 0.7f32.to_bits();
        m[1] = 0.7f32.to_bits();
        m[2] = 0.7f32.to_bits();
        m[3] = 0.5f32.to_bits();
        m[12] = 1.0f32.to_bits();
    }
    let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("materials"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Color pool — zero for every slot (= "use material base_color").
    // Same length as leaf_attrs so per-leaf indexing stays bounds-safe.
    let color_pool = vec![0u32; leaf_attr_count.max(1)];
    let color_pool_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("color_pool"),
        contents: bytemuck::cast_slice(&color_pool),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ── 4. Camera (CameraUniforms — 224 B) ─────────────────────────
    let aabb_center = (aabb_min + aabb_max) * 0.5;
    let asset_extent = (aabb_max - aabb_min).length();
    let dist_factor: f32 = std::env::var("RKP_SPLAT_CAM_DIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.55);
    let cam_pos =
        aabb_center + Vec3::new(1.0, 0.6, 1.0).normalize() * (asset_extent * dist_factor);
    let view = Mat4::look_at_rh(cam_pos, aabb_center, Vec3::Y);
    let aspect = W as f32 / H as f32;
    let proj = Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.05, asset_extent * 4.0);
    eprintln!(
        "[splat_render] camera at {:.2} m from center, asset_extent {:.2} m",
        (cam_pos - aabb_center).length(),
        asset_extent,
    );
    let view_proj = proj * view;
    let view_dir = (aabb_center - cam_pos).normalize();
    let world_up = Vec3::Y;
    let cam_right = view_dir.cross(world_up).normalize();
    let cam_up = cam_right.cross(view_dir);

    // CameraUniforms layout: position vec4 / forward vec4 / right vec4
    // / up vec4 / resolution vec2 / jitter vec2 / layer_mask u32 /
    // focus_object_id u32 / pad u32 u32 / prev_vp mat4 / view_proj mat4
    let camera = CameraUniforms {
        position: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
        forward: [view_dir.x, view_dir.y, view_dir.z, 0.0],
        right: [cam_right.x, cam_right.y, cam_right.z, 0.0],
        up: [cam_up.x, cam_up.y, cam_up.z, 0.0],
        resolution: [W as f32, H as f32],
        jitter: [0.0, 0.0],
        layer_mask: u32::MAX,
        focus_object_id: u32::MAX,
        _pad: [0; 2],
        prev_vp: view_proj.to_cols_array_2d(),
        view_proj: view_proj.to_cols_array_2d(),
    };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat camera"),
        contents: bytemuck::bytes_of(&camera),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Per-instance uniform — identity world matrix, object_id 1.
    let instance = SplatInstanceUniform {
        world: Mat4::IDENTITY.to_cols_array_2d(),
        object_id: 1,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat instance"),
        contents: bytemuck::bytes_of(&instance),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ── 5. G-buffer render targets ──────────────────────────────────
    let make_color_tex = |label: &'static str, format: wgpu::TextureFormat| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let position_tex = make_color_tex("gbuf_position", POSITION_FORMAT);
    let normal_tex = make_color_tex("gbuf_normal", NORMAL_FORMAT);
    let material_tex = make_color_tex("gbuf_material", MATERIAL_FORMAT);
    let pick_tex = make_color_tex("gbuf_pick", PICK_FORMAT);
    let glass_tex = make_color_tex("gbuf_glass", GLASS_FORMAT);
    let leaf_slot_tex = make_color_tex("gbuf_leaf_slot", LEAF_SLOT_FORMAT);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gbuf_depth"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let position_view = position_tex.create_view(&Default::default());
    let normal_view = normal_tex.create_view(&Default::default());
    let material_view = material_tex.create_view(&Default::default());
    let pick_view = pick_tex.create_view(&Default::default());
    let glass_view = glass_tex.create_view(&Default::default());
    let leaf_slot_view = leaf_slot_tex.create_view(&Default::default());
    let depth_view = depth_tex.create_view(&Default::default());

    // ── 6. Pipeline + bind groups ──────────────────────────────────
    let pass = SplatPass::new(&device);
    let g0_bg = pass.create_g0_bind_group(
        &device,
        &camera_buffer,
        &leaf_attr_buffer,
        &materials_buffer,
        &color_pool_buffer,
    );
    let g1_bg = pass.create_g1_bind_group(&device, &instance_buffer);

    // ── 7. Timestamps + frames ─────────────────────────────────────
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("splat timestamps"),
        count: 2,
        ty: wgpu::QueryType::Timestamp,
    });
    let timestamp_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp resolve"),
        size: 2 * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let timestamp_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp read"),
        size: 2 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let warmup_frames = 3;
    let measure_frames = 8;
    let mut measure_us: Vec<f32> = Vec::new();

    for frame in 0..(warmup_frames + measure_frames) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("splat frame"),
        });
        {
            let mut rp = pass.begin_pass(
                &mut encoder,
                &position_view,
                &normal_view,
                &material_view,
                &pick_view,
                &glass_view,
                &leaf_slot_view,
                &depth_view,
                Some(wgpu::RenderPassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            );
            rp.set_pipeline(&pass.pipeline);
            rp.set_bind_group(0, &g0_bg, &[]);
            rp.set_bind_group(1, &g1_bg, &[]);
            rp.set_vertex_buffer(0, vertex_buffer.slice(..));
            rp.draw(0..4, 0..(splats.len() as u32));
        }
        encoder.resolve_query_set(&query_set, 0..2, &timestamp_resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&timestamp_resolve_buf, 0, &timestamp_read_buf, 0, 16);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = timestamp_read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let view = slice.get_mapped_range();
        let ticks: &[u64] = bytemuck::cast_slice(&view);
        let dt_ticks = ticks[1].saturating_sub(ticks[0]);
        drop(view);
        timestamp_read_buf.unmap();
        let period_ns = queue.get_timestamp_period();
        let dt_us = (dt_ticks as f32) * period_ns / 1000.0;
        if frame >= warmup_frames {
            measure_us.push(dt_us);
        }
        eprintln!(
            "[splat_render] frame {frame} ({}): {dt_us:.1} µs",
            if frame < warmup_frames { "warmup" } else { "measure" },
        );
    }

    measure_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = measure_us[measure_us.len() / 2];
    let mean_us: f32 = measure_us.iter().sum::<f32>() / measure_us.len() as f32;
    eprintln!(
        "[splat_render] {} splats @ {}×{} = median {:.2} ms (mean {:.2} ms) GPU",
        splats.len(),
        W,
        H,
        median_us / 1000.0,
        mean_us / 1000.0,
    );

    // ── 8. (optional) Image dumps for visual sanity ────────────────
    if let Ok(dir) = std::env::var("RKP_SPLAT_DUMP_DIR") {
        std::fs::create_dir_all(&dir).expect("create dump dir");
        // Position is Rgba32Float (xyz, hit_distance) — visualize as
        // colour-coded per-axis. Normal is Rgba16Float — visualize the
        // unsigned half-vec mapping classic to the eye.
        dump_rgba32f_xyz_to_ppm(
            &device,
            &queue,
            &position_tex,
            W,
            H,
            &format!("{dir}/position.ppm"),
            aabb_min,
            aabb_max,
        );
        dump_rgba16f_normal_to_ppm(
            &device,
            &queue,
            &normal_tex,
            W,
            H,
            &format!("{dir}/normal.ppm"),
        );
        eprintln!("[splat_render] wrote {dir}/{{position,normal}}.ppm");
    }
}

/// Read back a `Rgba32Float` position target and write it as an RGB PPM
/// where each channel is the per-axis world-space position remapped
/// against the asset's AABB. Pixels with the miss sentinel
/// (hit_distance == 1e10) are written black.
fn dump_rgba32f_xyz_to_ppm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    path: &str,
    aabb_min: Vec3,
    aabb_max: Vec3,
) {
    let bytes_per_pixel = 16u32; // Rgba32Float
    let row_bytes = width * bytes_per_pixel;
    assert_eq!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let buffer_size = (row_bytes as u64) * height as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ppm_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ppm copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let view = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&view);

    let mut out = Vec::with_capacity((6 * (width * height) as usize) + 64);
    out.extend_from_slice(format!("P6\n{} {}\n65535\n", width, height).as_bytes());
    let extent = aabb_max - aabb_min;
    for pix in 0..(width * height) as usize {
        let x = floats[pix * 4];
        let y = floats[pix * 4 + 1];
        let z = floats[pix * 4 + 2];
        let hit = floats[pix * 4 + 3];
        let (rr, gg, bb) = if hit > 1e9 {
            (0.0, 0.0, 0.0)
        } else {
            (
                ((x - aabb_min.x) / extent.x).clamp(0.0, 1.0),
                ((y - aabb_min.y) / extent.y).clamp(0.0, 1.0),
                ((z - aabb_min.z) / extent.z).clamp(0.0, 1.0),
            )
        };
        let w = |v: f32| -> [u8; 2] {
            let q = (v * 65535.0).round().clamp(0.0, 65535.0) as u16;
            q.to_be_bytes()
        };
        out.extend_from_slice(&w(rr));
        out.extend_from_slice(&w(gg));
        out.extend_from_slice(&w(bb));
    }
    drop(view);
    staging.unmap();
    std::fs::write(path, &out).expect("write ppm");
}

/// Read back a `Rgba16Float` normal target and write it as an RGB PPM.
/// Maps `(n.xyz * 0.5 + 0.5)` to colour so positive components light up.
fn dump_rgba16f_normal_to_ppm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    path: &str,
) {
    let bytes_per_pixel = 8u32;
    let row_bytes = width * bytes_per_pixel;
    assert_eq!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let buffer_size = (row_bytes as u64) * height as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ppm_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ppm copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let view = slice.get_mapped_range();
    let halfs: &[half::f16] = bytemuck::cast_slice(&view);

    let mut out = Vec::with_capacity((6 * (width * height) as usize) + 64);
    out.extend_from_slice(format!("P6\n{} {}\n65535\n", width, height).as_bytes());
    for pix in 0..(width * height) as usize {
        let r = halfs[pix * 4].to_f32();
        let g = halfs[pix * 4 + 1].to_f32();
        let b = halfs[pix * 4 + 2].to_f32();
        let rr = (r * 0.5 + 0.5).clamp(0.0, 1.0);
        let gg = (g * 0.5 + 0.5).clamp(0.0, 1.0);
        let bb = (b * 0.5 + 0.5).clamp(0.0, 1.0);
        let w = |v: f32| -> [u8; 2] {
            let q = (v * 65535.0).round().clamp(0.0, 65535.0) as u16;
            q.to_be_bytes()
        };
        out.extend_from_slice(&w(rr));
        out.extend_from_slice(&w(gg));
        out.extend_from_slice(&w(bb));
    }
    drop(view);
    staging.unmap();
    std::fs::write(path, &out).expect("write ppm");
}
