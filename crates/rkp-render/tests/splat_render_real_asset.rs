//! Phase B-1 — full splat-rasterizer integration test.
//!
//! Loads a real `.rkp`, builds the SplatVertex buffer, renders a
//! 1920×1080 frame via the splat pipeline with timestamp queries, and
//! reports GPU time. Optionally dumps the albedo + normal targets to
//! PNGs for visual sanity (set `RKP_SPLAT_DUMP_DIR=/some/dir`).
//!
//! Skipped by default — set `RKP_SPLAT_TEST_ASSET=/path/to/scene.rkp`
//! (and ensure the wgpu adapter is available + supports
//! TIMESTAMP_QUERY).
//!
//! This runs the splat path *in isolation*: no editor, no scene
//! manager, no march. The numbers it reports are pure splat-rasterize
//! GPU time for a single asset, fixed camera, identity world matrix.

use std::io::BufReader;

use glam::{Mat4, Vec3};
use rkp_render::splat_pass::{
    extract_splats_with_radius, SplatCamera, SplatPass, SplatPassConfig, DISC_RADIUS_FACTOR,
};

const W: u32 = 1920;
const H: u32 = 1080;
const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
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

    // Per-leaf normals — packed alongside the voxel records on disk;
    // we need them on the GPU as a `LeafAttr` array indexed by
    // leaf_attr_id. The runtime asset_load merges these with the
    // material-id field; for the prototype we pull just the normals
    // (materials all default to material_id 1, the "default" slot).
    let normals_bytes = if header.flags & rkp_core::asset_file::FLAG_HAS_NORMALS != 0 {
        rkp_core::asset_file::read_rkp_normals(&mut reader, &header)
            .expect("normals")
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
    // Disc radius factor — controls overlap between adjacent splats.
    // Default is 0.6 (just enough to cover diagonals on flat surfaces);
    // bump higher to mask glancing-angle silhouette stepping.
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
        Mat4::IDENTITY,
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

    // LeafAttr pool — one entry per voxel slot. The cell value `c`
    // we stuffed into SplatVertex.leaf_attr_id is the absolute index
    // into this pool. For the prototype we synthesize entries from
    // the on-disk normals blob: 4 bytes per leaf = 1× u32 oct-normal.
    // material_packed defaults to 1 (the "default" material slot).
    let normals_u32: &[u32] = if normals_bytes.len() >= 4 {
        bytemuck::cast_slice(&normals_bytes)
    } else {
        &[]
    };
    let leaf_attr_count = normals_u32.len();
    let mut leaf_attrs = Vec::with_capacity(leaf_attr_count);
    for &n in normals_u32 {
        // 8-byte LeafAttr: (normal_oct, material_packed)
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

    // Materials — `GpuMaterial` is 24 × 4 = 96 B (see
    // `shaders/lib/types.wesl`). One default plus a few extra slots
    // so the leaf_attr_pool's stub material_id=1 maps to a defined
    // entry. Index 1 = mid-grey diffuse.
    let mut materials = vec![[0u32; 24]; 8]; // stride MUST match the WGSL struct
    for m in &mut materials {
        m[0] = 0.7f32.to_bits(); // albedo_r
        m[1] = 0.7f32.to_bits(); // albedo_g
        m[2] = 0.7f32.to_bits(); // albedo_b
        m[3] = 0.5f32.to_bits(); // roughness
        m[12] = 1.0f32.to_bits(); // opacity
    }
    let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("materials"),
        contents: bytemuck::cast_slice(&materials),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ── 4. Camera ──────────────────────────────────────────────────
    let aabb_center = (aabb_min + aabb_max) * 0.5;
    let asset_extent = (aabb_max - aabb_min).length();
    // Camera distance scaled by env var so we can A/B sub-pixel
    // sensitivity without recompiling. Default 0.55 puts us close
    // enough that voxels project to ~few pixels each.
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
    let camera = SplatCamera {
        view_proj: view_proj.to_cols_array_2d(),
        position: cam_pos.to_array(),
        _pad0: 0.0,
        resolution: [W as f32, H as f32],
        _pad1: [0.0; 2],
    };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat camera"),
        contents: bytemuck::bytes_of(&camera),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ── 5. Targets ──────────────────────────────────────────────────
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
    let albedo_tex = make_color_tex("splat_albedo", ALBEDO_FORMAT);
    let normal_tex = make_color_tex("splat_normal", NORMAL_FORMAT);
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("splat_depth"),
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
    let albedo_view = albedo_tex.create_view(&Default::default());
    let normal_view = normal_tex.create_view(&Default::default());
    let depth_view = depth_tex.create_view(&Default::default());

    // ── 6. Pipeline + bind group ───────────────────────────────────
    let pass = SplatPass::new(
        &device,
        &SplatPassConfig {
            albedo_format: ALBEDO_FORMAT,
            normal_format: NORMAL_FORMAT,
            depth_format: DEPTH_FORMAT,
            sample_count: 1,
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("splat g0"),
        layout: &pass.g0_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: leaf_attr_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: materials_buffer.as_entire_binding(),
            },
        ],
    });

    // ── 7. Timestamp queries ───────────────────────────────────────
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

    // Warm-up + measurement: render a few frames to let driver/cache
    // settle, then time the average over the next frames.
    let warmup_frames = 3;
    let measure_frames = 8;
    let mut measure_us: Vec<f32> = Vec::new();

    for frame in 0..(warmup_frames + measure_frames) {
        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("splat frame") });
        pass.render(
            &mut encoder,
            &vertex_buffer,
            splats.len() as u32,
            &bind_group,
            &albedo_view,
            &normal_view,
            &depth_view,
            wgpu::Color { r: 0.0, g: 0.0, b: 0.05, a: 1.0 },
            Some(wgpu::RenderPassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        );
        encoder.resolve_query_set(&query_set, 0..2, &timestamp_resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&timestamp_resolve_buf, 0, &timestamp_read_buf, 0, 16);
        queue.submit(std::iter::once(encoder.finish()));

        // Read back the timestamp.
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

    // ── 8. (optional) Image dump for visual sanity ─────────────────
    if let Ok(dir) = std::env::var("RKP_SPLAT_DUMP_DIR") {
        std::fs::create_dir_all(&dir).expect("create dump dir");
        dump_rgba16_to_ppm(&device, &queue, &albedo_tex, W, H, &format!("{dir}/albedo.ppm"), true);
        dump_rgba16_to_ppm(&device, &queue, &normal_tex, W, H, &format!("{dir}/normal.ppm"), false);
        eprintln!("[splat_render] wrote {dir}/{{albedo,normal}}.ppm");
    }
}

/// Read back an `Rgba16Float` render target, tonemap (Reinhard + sRGB
/// gamma) when `tonemap` is set, and write a P6 PPM. Trivial format —
/// any image viewer or `convert in.ppm out.png` opens it.
///
/// `bytes_per_row` is already a multiple of 256 for our 1920-pixel
/// width (1920 × 8 = 15360 = 60 × 256), so the copy lays out cleanly.
fn dump_rgba16_to_ppm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    path: &str,
    tonemap: bool,
) {
    let bytes_per_pixel = 8u32; // Rgba16Float
    let row_bytes = width * bytes_per_pixel;
    assert_eq!(
        row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        0,
        "bytes_per_row must be 256-aligned; pad if you change W"
    );
    let buffer_size = (row_bytes as u64) * height as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ppm_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ppm copy") });
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

    // Header: P6 magic, width height, max value, single newline before
    // raw bytes.
    let mut out = Vec::with_capacity((3 * (width * height) as usize) + 64);
    out.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());

    for pix in 0..(width * height) as usize {
        let r = halfs[pix * 4].to_f32();
        let g = halfs[pix * 4 + 1].to_f32();
        let b = halfs[pix * 4 + 2].to_f32();
        let (rr, gg, bb) = if tonemap {
            // Reinhard: c / (1 + c). Then sRGB-ish gamma 2.2 via sqrt.
            let r = (r / (1.0 + r)).sqrt();
            let g = (g / (1.0 + g)).sqrt();
            let b = (b / (1.0 + b)).sqrt();
            (r, g, b)
        } else {
            // Normal target — values are already in [0, 1] (encoded
            // n*0.5 + 0.5). Just clamp.
            (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
        };
        out.push((rr * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((gg * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((bb * 255.0).round().clamp(0.0, 255.0) as u8);
    }

    drop(view);
    staging.unmap();
    std::fs::write(path, &out).expect("write ppm");
}
