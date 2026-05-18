//! GPU device and queue wrapper.
//!
//! [`RenderContext`] encapsulates the wgpu adapter, device, and queue used
//! by all rendering passes. Created once at startup from a compatible surface.

/// Core GPU context holding the wgpu device and queue.
///
/// Created via [`RenderContext::new`] which requests an adapter compatible
/// with the given surface, then opens a device with default limits.
pub struct RenderContext {
    /// The wgpu device used for resource creation and command encoding.
    pub device: wgpu::Device,
    /// The command queue for submitting GPU work.
    pub queue: wgpu::Queue,
    /// The adapter that was selected (`None` when using a shared device).
    pub adapter: Option<wgpu::Adapter>,
    /// Adapter information (name, vendor ID, etc.) for capability queries.
    pub adapter_info: wgpu::AdapterInfo,
}

impl RenderContext {
    /// Create a new render context compatible with the given surface.
    ///
    /// Blocks on async wgpu initialization using `pollster`.
    ///
    /// # Panics
    ///
    /// Panics if no compatible adapter is found or device creation fails.
    pub fn new(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Self {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(surface),
        }))
        .expect("failed to find a compatible GPU adapter");

        let adapter_info = adapter.get_info();
        log::info!("GPU adapter: {:?}", adapter_info.name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rkp-render device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE
                    | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
                    // Match `new_headless` — see the longer comment
                    // there for the rationale. Phase 6 mesh path
                    // depends on native multi-draw to avoid CPU
                    // encode-loop bottleneck.
                required_limits: wgpu::Limits {
                    max_bind_groups: 8,
                    // Take whatever the adapter offers up to 2 GB. The
                    // shared `brick_pool_buffer` / `leaf_attr_pool_buffer`
                    // stack persistent CPU geometry + Phase C user-shader
                    // tail (768 MB at MAX_GLOBAL_BRICKS=3M) + Option B
                    // proto sub-pool, which routinely exceeds the 1 GB
                    // wgpu downlevel default. Adapters that can't supply
                    // 2 GB still get whatever they expose; ensure_capacity
                    // clamps each buffer to the granted limit.
                    max_storage_buffer_binding_size:
                        adapter.limits().max_storage_buffer_binding_size.min(1 << 31),
                    max_buffer_size: adapter.limits().max_buffer_size.min(1 << 31),
                    // 32 rather than the wgpu default 8 — group 0's
                    // scene bindings are 13 storage buffers (brick_pool,
                    // octree_nodes, objects, color_pool, bone_matrices,
                    // bone_weights, brick_face_links, leaf_attr_pool,
                    // bone_field, bone_field_occ, bone_dual_quats,
                    // assets, instance_overlay).
                    // March group 2 adds 13 (materials, stats, lights,
                    // tile_offsets, tile_object_ids, user_shader_instances,
                    // user_shader_instance_count, tlas_nodes, tlas_leaves,
                    // shader_params, user_shader_instance_aabbs,
                    // user_shader_tile_counts, user_shader_tile_lists)
                    // for 26 total live in the march. Desktop GPUs
                    // (RTX 30+, AMD RDNA2+) support 64+; GLES 3.x / some
                    // mobile top out at 16 but aren't supported targets.
                    max_storage_buffers_per_shader_stage: 32,
                    // wgpu default is 4; the march write group needs
                    // 5 (position + normal + material + pick + glass).
                    // Desktop GPUs support 8+ universally.
                    max_storage_textures_per_shader_stage: 8,
                    // wgpu default is 32 B; the mesh raster path writes
                    // position (Rgba32Float, 16 B) + pick (R32Uint, 4 B) +
                    // leaf_slot (R32Uint, 4 B) + rest_pos (Rgba32Float,
                    // 16 B) = 40 B/sample. rest_pos was added in Phase
                    // 6.7 for per-pixel cell sampling in `mesh_resolve`.
                    // Desktop GPUs (RTX 20+, RDNA2+) report 64-128 B; take
                    // whatever the adapter exposes so the default cap is
                    // lifted.
                    max_color_attachment_bytes_per_sample:
                        adapter.limits().max_color_attachment_bytes_per_sample,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .expect("failed to create GPU device");

        let granted = device.limits();
        eprintln!(
            "[rkp_context] adapter={:?} granted limits: max_storage_buffer_binding_size={} max_buffer_size={}",
            adapter_info.name,
            granted.max_storage_buffer_binding_size,
            granted.max_buffer_size,
        );
        // Log GPU device errors and device lost events to stderr for diagnostics.
        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            eprintln!("[GPU ERROR] {error}");
        }));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("[GPU DEVICE LOST] reason={reason:?} msg={msg}");
        });

        Self {
            device,
            queue,
            adapter: Some(adapter),
            adapter_info,
        }
    }

    /// Create a render context with its own device, without requiring a surface.
    ///
    /// Used when the engine needs a dedicated GPU device (e.g., to avoid
    /// contention with a compositor sharing the same device). The engine
    /// renders offscreen and reads back pixels to CPU.
    pub fn new_headless() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            // `InstanceFlags::empty()` overrides wgpu's
            // `from_build_config()` default. That default is
            // `VALIDATION_INDIRECT_CALL` even in release builds
            // (wgpu-types/src/instance.rs:253-264), which gates
            // wgpu-core's per-indirect-draw validation pass
            // (wgpu-core/src/device/resource.rs:502-507). With
            // Phase 6 mesh-mode issuing ~1.16M cluster draws per
            // frame on real scenes, that validation pass is the
            // sole reason the encode CPU phase ran at ~62ms vs
            // the ~0.7ms of `RKP_MESH_DEBUG_DIRECT=1`. Bisected
            // 2026-05-06 against the splat5 elephant scene.
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("failed to find a compatible GPU adapter (headless)");

        let adapter_info = adapter.get_info();
        log::info!("GPU adapter (headless): {:?}", adapter_info.name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rkp-render headless device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                    | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                    | wgpu::Features::PIPELINE_STATISTICS_QUERY,
                    // `MULTI_DRAW_INDIRECT_COUNT` is load-bearing
                    // for the Phase 6 mesh path even though we don't
                    // use the count-from-GPU variant (yet). Per the
                    // wgpu 29 feature docs: when the feature is
                    // present, ordinary `multi_draw_indexed_indirect`
                    // calls take the native path; when it's absent,
                    // wgpu emulates them as N separate
                    // `draw_indexed_indirect` CPU encoder calls.
                    // With LOD_LEVELS=4 a real multi-asset scene
                    // issues hundreds of thousands of cluster draws
                    // per frame (primary + shadow combined); the
                    // emulated CPU loop dominates frame time
                    // (~50ms encode wall-clock vs ~17ms actual GPU
                    // work). On Vulkan 1.2+ / DX12 + modern desktop
                    // GPUs the feature is universally supported, so
                    // a hard require is fine. Metal + GLES don't
                    // have it and would need a runtime fallback
                    // (e.g., direct `draw_indexed` over LOD-0); not
                    // a supported target today.
                required_limits: wgpu::Limits {
                    max_bind_groups: 8,
                    // Take whatever the adapter offers up to 2 GB. The
                    // shared `brick_pool_buffer` / `leaf_attr_pool_buffer`
                    // stack persistent CPU geometry + Phase C user-shader
                    // tail (768 MB at MAX_GLOBAL_BRICKS=3M) + Option B
                    // proto sub-pool, which routinely exceeds the 1 GB
                    // wgpu downlevel default. Adapters that can't supply
                    // 2 GB still get whatever they expose; ensure_capacity
                    // clamps each buffer to the granted limit.
                    max_storage_buffer_binding_size:
                        adapter.limits().max_storage_buffer_binding_size.min(1 << 31),
                    max_buffer_size: adapter.limits().max_buffer_size.min(1 << 31),
                    // 32 rather than the wgpu default 8 — group 0's
                    // scene bindings are 13 storage buffers (brick_pool,
                    // octree_nodes, objects, color_pool, bone_matrices,
                    // bone_weights, brick_face_links, leaf_attr_pool,
                    // bone_field, bone_field_occ, bone_dual_quats,
                    // assets, instance_overlay).
                    // March group 2 adds 13 (materials, stats, lights,
                    // tile_offsets, tile_object_ids, user_shader_instances,
                    // user_shader_instance_count, tlas_nodes, tlas_leaves,
                    // shader_params, user_shader_instance_aabbs,
                    // user_shader_tile_counts, user_shader_tile_lists)
                    // for 26 total live in the march. Desktop GPUs
                    // (RTX 30+, AMD RDNA2+) support 64+; GLES 3.x / some
                    // mobile top out at 16 but aren't supported targets.
                    max_storage_buffers_per_shader_stage: 32,
                    // wgpu default is 4; the march write group needs
                    // 5 (position + normal + material + pick + glass).
                    // Desktop GPUs support 8+ universally.
                    max_storage_textures_per_shader_stage: 8,
                    // wgpu default is 32 B; the mesh raster path writes
                    // position (Rgba32Float, 16 B) + pick (R32Uint, 4 B) +
                    // leaf_slot (R32Uint, 4 B) + rest_pos (Rgba32Float,
                    // 16 B) = 40 B/sample. rest_pos was added in Phase
                    // 6.7 for per-pixel cell sampling in `mesh_resolve`.
                    // Desktop GPUs (RTX 20+, RDNA2+) report 64-128 B; take
                    // whatever the adapter exposes so the default cap is
                    // lifted.
                    max_color_attachment_bytes_per_sample:
                        adapter.limits().max_color_attachment_bytes_per_sample,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .expect("failed to create headless GPU device");

        // Log granted features so a missing one is visible at startup.
        // Listing only the perf-load-bearing features rather than the
        // full set keeps the line readable.
        let features = device.features();
        eprintln!(
            "[rkp_context] granted features: multi_draw_indirect_count={} timestamp={} timestamp_inside={} float32_filterable={}",
            features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            features.contains(wgpu::Features::TIMESTAMP_QUERY),
            features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            features.contains(wgpu::Features::FLOAT32_FILTERABLE),
        );

        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            eprintln!("[GPU ERROR] {error}");
        }));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("[GPU DEVICE LOST] reason={reason:?} msg={msg}");
        });

        Self {
            device,
            queue,
            adapter: Some(adapter),
            adapter_info,
        }
    }

    /// Create a render context from a shared device and queue.
    ///
    /// Used when the engine shares a wgpu device with an external renderer
    /// (e.g., rinch's `GpuHandle` for zero-copy compositing). No adapter is
    /// available, so surface configuration will panic — use only for offscreen
    /// rendering.
    pub fn from_shared(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        log::info!("RenderContext: using shared device");

        Self {
            device,
            queue,
            adapter: None,
            adapter_info: wgpu::AdapterInfo {
                name: "shared device".into(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Other,
                device_pci_bus_id: String::new(),
                driver: String::new(),
                driver_info: String::new(),
                backend: wgpu::Backend::Vulkan,
                subgroup_min_size: 0,
                subgroup_max_size: 0,
                transient_saves_memory: false,
            },
        }
    }

    /// Configure a surface for presentation with the given dimensions.
    ///
    /// Returns the chosen surface format.
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let adapter = self.adapter.as_ref()
            .expect("configure_surface requires an adapter (not available on shared devices)");
        let caps = surface.get_capabilities(adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        log::info!("Surface configured: {width}x{height}, format={format:?}");
        format
    }
}
