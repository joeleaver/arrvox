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
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: wgpu::Limits {
                    max_bind_groups: 8,
                    max_storage_buffer_binding_size: 1 << 30, // 1 GB
                    max_buffer_size: 1 << 31, // 2 GB
                    // 20 rather than the wgpu default 8 — group 0's
                    // scene bindings are 11 storage buffers (brick_pool,
                    // octree_nodes, objects, color_pool, bone_matrices,
                    // bone_weights, brick_face_links, leaf_attr_pool,
                    // bone_field, bone_field_occ, bone_dual_quats).
                    // March group 2 adds 5 (materials, stats, lights,
                    // tile_offsets, tile_object_ids) for 16 total.
                    // wgpu counts unused-by-shader layout entries too,
                    // so splitting per-shader layouts is the only way
                    // below the limit — not worth it. Desktop GPUs
                    // support 24+; GLES 3.x / some mobile top out at
                    // 16 but aren't supported targets.
                    max_storage_buffers_per_shader_stage: 20,
                    // wgpu default is 4; the march write group needs
                    // 5 (position + normal + material + pick + glass).
                    // Desktop GPUs support 8+ universally.
                    max_storage_textures_per_shader_stage: 8,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .expect("failed to create GPU device");

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
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits {
                    max_bind_groups: 8,
                    max_storage_buffer_binding_size: 1 << 30, // 1 GB
                    max_buffer_size: 1 << 31, // 2 GB
                    // 20 rather than the wgpu default 8 — group 0's
                    // scene bindings are 11 storage buffers (brick_pool,
                    // octree_nodes, objects, color_pool, bone_matrices,
                    // bone_weights, brick_face_links, leaf_attr_pool,
                    // bone_field, bone_field_occ, bone_dual_quats).
                    // March group 2 adds 5 (materials, stats, lights,
                    // tile_offsets, tile_object_ids) for 16 total.
                    // wgpu counts unused-by-shader layout entries too,
                    // so splitting per-shader layouts is the only way
                    // below the limit — not worth it. Desktop GPUs
                    // support 24+; GLES 3.x / some mobile top out at
                    // 16 but aren't supported targets.
                    max_storage_buffers_per_shader_stage: 20,
                    // wgpu default is 4; the march write group needs
                    // 5 (position + normal + material + pick + glass).
                    // Desktop GPUs support 8+ universally.
                    max_storage_textures_per_shader_stage: 8,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .expect("failed to create headless GPU device");

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
