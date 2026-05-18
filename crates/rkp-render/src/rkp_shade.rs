//! RKIPatch deferred PBR shading compute pass.
//!
//! Reads G-buffer + shadow/AO texture, evaluates Cook-Torrance PBR with direct
//! lighting, hemisphere ambient, AO, and emission. Writes final HDR color.

/// The deferred PBR shading pass.
pub struct RkpShadePass {
    pipeline: wgpu::ComputePipeline,
    /// Cached pipeline layout — kept on the struct so `reload_user_shaders`
    /// can rebuild the compute pipeline with a new user-shader chunk
    /// without re-creating the layout (which is independent of source).
    pipeline_layout: wgpu::PipelineLayout,
    pub gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    pub ssao_bind_group_layout: wgpu::BindGroupLayout,
    pub output_bind_group_layout: wgpu::BindGroupLayout,
    pub shade_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub atmo_bind_group_layout: wgpu::BindGroupLayout,
    /// Group 6 — `BrushState` storage buffer written each frame by
    /// the brush-state probe pass (`brush_state.wesl`). Carries the
    /// world-space surface point + entity id under the cursor pixel,
    /// consumed by the screen-space paint cursor in `rkp_shade.wesl`.
    pub brush_state_bind_group_layout: wgpu::BindGroupLayout,
    /// HDR output texture (full-res, Rgba16Float).
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    output_bind_group: wgpu::BindGroup,
    gbuffer_bind_group: Option<wgpu::BindGroup>,
    ssao_bind_group: Option<wgpu::BindGroup>,
    shade_bind_group: Option<wgpu::BindGroup>,
    camera_bind_group: Option<wgpu::BindGroup>,
    atmo_bind_group: Option<wgpu::BindGroup>,
    /// `BrushState` storage-buffer bind group. Set via
    /// [`Self::set_brush_state`] from the per-VR buffer that the
    /// brush-state probe pass writes. The `Option` lets the pass
    /// no-op cleanly before the first viewport setup.
    brush_state_bind_group: Option<wgpu::BindGroup>,
    /// Resident per-material user-shader params buffer. Parallel to
    /// the materials buffer: 32 bytes per material (8 × f32).
    /// `upload_shader_params` grows it as the materials array grows.
    shader_params_buffer: wgpu::Buffer,
    shader_params_buffer_capacity: u64,
    /// PERF_DEBT.md D4: hash of the last `upload_shader_params`
    /// payload. The full slot array ships from sim every tick per
    /// viewport, but the content rarely changes; the hash gate skips
    /// the `queue.write_buffer` when the bytes match. `0` = "never
    /// uploaded" sentinel; the first upload always writes.
    last_shader_params_hash: u64,
    /// Hash of the user-shader chunk currently compiled into the
    /// pipeline. `0` is the "no user shaders" sentinel; matches
    /// `UserShaderRegistry::default()::source_hash()`.
    /// `reload_user_shaders` skips rebuilds when this matches.
    source_hash: u64,
    width: u32,
    height: u32,
}

/// Shading parameters uniform.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadeParams {
    pub num_lights: u32,
    pub ambient_intensity: f32,
    pub camera_altitude: f32,
    pub sun_intensity: f32,
    pub sky_color_top: [f32; 3],
    pub _pad0: f32,
    pub sky_color_horizon: [f32; 3],
    pub _pad1: f32,
    pub sun_dir: [f32; 3],
    pub _pad2: f32,
    pub ambient_color: [f32; 3],
    /// 0 = full PBR + atmosphere + shadows, 1 = isolation studio (neutral
    /// gray sky, fixed ambient, shadows forced 1.0). Per-VR — written
    /// just before the VR's submit, same channel as the other per-VR
    /// frame params.
    pub isolation: u32,
    /// `brush_active = 0` disables the screen-space paint cursor.
    /// `brush_active = 1` arms it; the cursor is drawn iff the
    /// per-frame `BrushState` buffer's `hit_object_id` matches the
    /// `brush_object_id` below (selection-lock).
    pub brush_active: u32,
    pub brush_radius: f32,
    /// `gpu_idx` of the currently-selected entity, or `u32::MAX` when
    /// nothing is selected. Used by the shade pass to gate the cursor
    /// to fragments belonging to that entity (mirrors the CPU paint
    /// stamp's selection-lock check).
    pub brush_object_id: u32,
    /// Engine clock in seconds. Used by user-shader `shade` hooks for
    /// time-driven effects (hologram scroll, fresnel pulse, etc.).
    pub time: f32,
    pub brush_color: [f32; 4],
    /// Non-zero ⇒ directional shadow comes from the CSM shadow
    /// buffer (g1 binding 2). Zero ⇒ unshadowed (used when no CSM
    /// render has happened this frame).
    pub shadow_map_enabled: u32,
    /// Non-zero ⇒ all shadow sampling is bypassed and `light_shadow`
    /// is forced to 1.0. Used when no shadow pass has run this
    /// frame (e.g. isolation mode, or shadow disabled in viewport
    /// settings).
    pub shadow_disabled: u32,
    /// Per-fragment shadow-map tap count used by `sample_shadow_map`.
    /// 1 = single-tap (hard shadow, current cheapest); 4/9/16 =
    /// rotated PCF over a per-cascade-texel disc (soft shadows).
    /// Driven by the Shadow Quality preset; clamped to {1,4,9,16}
    /// on the shader side. Zero is treated as 1 for safety.
    pub pcf_taps: u32,
    pub _pad3: u32,
}

impl Default for ShadeParams {
    fn default() -> Self {
        Self {
            num_lights: 0,
            ambient_intensity: 0.3,
            camera_altitude: 100.0,
            sun_intensity: 20.0,
            sky_color_top: [0.4, 0.6, 1.0],
            _pad0: 0.0,
            sky_color_horizon: [0.8, 0.85, 0.9],
            _pad1: 0.0,
            sun_dir: [0.5, 0.7, 0.5],
            _pad2: 0.0,
            ambient_color: [0.1, 0.15, 0.25],
            isolation: 0,
            brush_active: 0,
            brush_radius: 0.5,
            brush_object_id: u32::MAX,
            time: 0.0,
            brush_color: [1.0, 0.85, 0.2, 1.0],
            shadow_map_enabled: 0,
            shadow_disabled: 0,
            pcf_taps: 1,
            _pad3: 0,
        }
    }
}

/// 32-byte storage buffer written by `brush_state.wesl` once per
/// frame and consumed by `rkp_shade.wesl` to draw the screen-space
/// paint cursor. Mirror of the WGSL `BrushState` struct in
/// `lib/types.wesl`. Default value (`hit_distance >= 1e9`) is
/// equivalent to "no surface under the cursor — hide the cursor."
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushState {
    pub center_world: [f32; 3],
    pub hit_distance: f32,
    pub hit_object_id: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for BrushState {
    fn default() -> Self {
        Self {
            center_world: [0.0; 3],
            hit_distance: 1.0e10,
            hit_object_id: u32::MAX,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<BrushState>() == 32);

/// Per-light GPU data.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub direction: [f32; 4],
    pub params: [f32; 4],
}

// ── Noise channel constants ─────────────────────────────────────────────
// Bit flags for GpuMaterial::noise_channels.
pub const NOISE_CHANNEL_ALBEDO: u32 = 1 << 0;
pub const NOISE_CHANNEL_ROUGHNESS: u32 = 1 << 1;
pub const NOISE_CHANNEL_NORMAL: u32 = 1 << 2;

/// Per-material GPU data — 96 bytes, matches rkifield's `Material` layout.
///
/// | Offset | Field              | Type     |
/// |-------:|--------------------|----------|
/// |      0 | albedo             | [f32; 3] |
/// |     12 | roughness          | f32      |
/// |     16 | metallic           | f32      |
/// |     20 | emission_color     | [f32; 3] |
/// |     32 | emission_strength  | f32      |
/// |     36 | subsurface         | f32      |
/// |     40 | subsurface_color   | [f32; 3] |
/// |     52 | opacity            | f32      |
/// |     56 | ior                | f32      |
/// |     60 | noise_scale        | f32      |
/// |     64 | noise_strength     | f32      |
/// |     68 | noise_channels     | u32      |
/// |     72 | shader_id          | u32      |
/// |     76 | _padding           | [f32; 5] |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission_color: [f32; 3],
    pub emission_strength: f32,
    pub subsurface: f32,
    pub subsurface_color: [f32; 3],
    pub opacity: f32,
    pub ior: f32,
    pub noise_scale: f32,
    pub noise_strength: f32,
    pub noise_channels: u32,
    /// Shade-dispatch id. Non-zero only for shaders that provide a
    /// `shade` hook; the shade pass routes to the per-shader case
    /// when set, or falls through to the PBR path when zero.
    pub shader_id: u32,
    /// Phase B-redux band-cell dispatch id. Non-zero only for shaders
    /// that provide an `instance_at` hook. The march reads this when
    /// it hits a band cell to look up the prototype asset; lookup is
    /// SEPARATE from `shader_id` so a geom-only shader (e.g. grass)
    /// doesn't accidentally route the shade pass through the user-
    /// dispatch default arm (which would emit raw albedo and tone-map
    /// to black against direct sun).
    pub instance_shader_id: u32,
    pub _padding: [f32; 4],
}

// Locks the byte layout against drift; the WGSL struct depends on it.
const _: () = assert!(std::mem::size_of::<GpuMaterial>() == 96);

impl RkpShadePass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uint_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        // Group 0: G-buffer (position, normal, material, glass, pick)
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade gbuf"),
                entries: &[
                    texture_entry(0),
                    texture_entry(1),
                    uint_texture_entry(2),
                    // Glass info (oct-normal + packed thickness/material_id)
                    uint_texture_entry(3),
                    // Pick — primary hit's `gpu_idx`. `0xFFFFFFFFu` =
                    // sky / no hit. Used by the screen-space paint
                    // cursor's per-pixel selection-lock gate.
                    uint_texture_entry(4),
                ],
            });

        // Group 1: SSAO texture + CSM (light_camera uniform +
        // shadow_buffer atomic-u32 depth) + glass shadow texture
        // arrays (front + back depth, one layer per cascade).
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade shadow+ssao+shadowmap"),
                entries: &[
                    texture_entry(0),
                    // light_camera uniform — wire format mirrors
                    // `shadow_map_pass::LightCameraUniform` (160 B).
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // shadow_buffer — `array<u32>` storage buffer
                    // holding bitcast-encoded f32 depths. Read-only
                    // here; the scatter pass owns the writes.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // glass_shadow_front / back — Depth32Float
                    // texture arrays, one layer per cascade. Sampled
                    // with integer coords (no sampler) by
                    // `glass_shadow_attenuation` in `rkp_shade.wesl`.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Group 2: output HDR texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade output"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        // Group 3: shade params + lights + materials
        let shade_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade params"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // User-shader per-material params (Phase B). Parallel
                    // to `materials`: 32 bytes per material (8 × f32).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 4: camera
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade camera"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Group 5: atmosphere LUTs
        let atmo_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade atmo"),
                entries: &[
                    // Atmosphere LUTs — all filterable.
                    Self::filterable_tex_2d(0),  // transmittance LUT
                    Self::filterable_tex_2d(1),  // multi-scatter LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    Self::filterable_tex_2d(3),  // sky view LUT
                    Self::filterable_tex_3d(4),  // aerial perspective LUT
                ],
            });

        // Group 6: `BrushState` storage buffer. Read-only on this
        // side — the brush-state probe pass owns the writes.
        let brush_state_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rkp_shade brush_state"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Output texture.
        let (output_texture, output_view) = Self::create_output(device, width, height);
        let output_bind_group = Self::create_output_bind_group(device, &output_bind_group_layout, &output_view);

        // Placeholder per-material user-shader params buffer. Sized to
        // hold one slot's worth of zeros at startup so the storage
        // binding stays valid even before any material with a shader
        // exists. `upload_shader_params` grows it on demand.
        let shader_params_buffer_capacity: u64 = 32; // one slot
        let shader_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_shade shader_params"),
            size: shader_params_buffer_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rkp_shade pipeline"),
            bind_group_layouts: &[
                Some(&gbuffer_bind_group_layout),
                Some(&ssao_bind_group_layout),
                Some(&output_bind_group_layout),
                Some(&shade_bind_group_layout),
                Some(&camera_bind_group_layout),
                Some(&atmo_bind_group_layout),
                Some(&brush_state_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = build_shade_pipeline(device, &pipeline_layout, "");

        Self {
            pipeline,
            pipeline_layout,
            gbuffer_bind_group_layout,
            ssao_bind_group_layout,
            output_bind_group_layout,
            shade_bind_group_layout,
            camera_bind_group_layout,
            atmo_bind_group_layout,
            brush_state_bind_group_layout,
            output_texture,
            output_view,
            output_bind_group,
            gbuffer_bind_group: None,
            ssao_bind_group: None,
            shade_bind_group: None,
            camera_bind_group: None,
            atmo_bind_group: None,
            brush_state_bind_group: None,
            shader_params_buffer,
            shader_params_buffer_capacity,
            last_shader_params_hash: 0,
            source_hash: 0,
            width,
            height,
        }
    }

    /// Recompile the shade pipeline with a new user-shader chunk from
    /// `compose_shade_source`. Idempotent: matching `source_hash` skips
    /// the rebuild. Returns true if the pipeline was actually rebuilt.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        user_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.source_hash {
            return false;
        }
        self.pipeline = build_shade_pipeline(device, &self.pipeline_layout, user_chunk);
        self.source_hash = source_hash;
        true
    }

    pub fn source_hash(&self) -> u64 {
        self.source_hash
    }

    /// Upload the per-material user-shader params buffer
    /// (`Vec<[f32; 8]>` from `MaterialLibrary::build_shader_params`).
    /// Grows the GPU buffer (and rebuilds the shade bind group on the
    /// next `set_shade_data`) when capacity is exceeded.
    pub fn upload_shader_params(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slots: &[[f32; 8]],
    ) {
        let needed = ((slots.len().max(1)) * std::mem::size_of::<[f32; 8]>()) as u64;
        let grew = needed > self.shader_params_buffer_capacity;
        if grew {
            let mut new_cap = self.shader_params_buffer_capacity.max(32);
            while new_cap < needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.shader_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_shade shader_params"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.shader_params_buffer_capacity = new_cap;
            // Bind group will be rebuilt by the next `set_shade_data`.
            self.shade_bind_group = None;
            // The freshly-created buffer is zero-initialised — last
            // upload's hash no longer represents what's on the GPU.
            // Force the write below by clearing the gate.
            self.last_shader_params_hash = 0;
        }
        if !slots.is_empty() {
            // PERF_DEBT.md D4: hash-gate the upload. The slot array is
            // 32 B per material; for a typical project (~50 materials)
            // it's ~1.5 KiB. Sim ships it on every tick per viewport;
            // most ticks the params don't change. Hashing on the CPU
            // costs ~few µs and lets us skip the queue.write_buffer
            // entirely when content matches the prior upload.
            let bytes: &[u8] = bytemuck::cast_slice(slots);
            let hash = super::rkp_renderer::d4_hash_bytes(bytes);
            if grew || hash != self.last_shader_params_hash {
                queue.write_buffer(&self.shader_params_buffer, 0, bytes);
                self.last_shader_params_hash = hash;
            }
        }
    }

    pub fn shader_params_buffer(&self) -> &wgpu::Buffer {
        &self.shader_params_buffer
    }

    /// Bind the per-VR `BrushState` storage buffer the brush-state
    /// probe pass writes. Idempotent — called from the viewport's
    /// per-resize binding refresh.
    pub fn set_brush_state(&mut self, device: &wgpu::Device, brush_state_buffer: &wgpu::Buffer) {
        self.brush_state_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade brush_state bg"),
            layout: &self.brush_state_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: brush_state_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Set G-buffer views.
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade gbuf bg"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(material_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(glass_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(pick_view) },
            ],
        }));
    }

    /// Set SSAO texture + CSM (light-camera uniform + shadow_buffer)
    /// + glass-shadow front/back texture arrays. WGSL gates the
    /// directional-shadow read on `ShadeParams.shadow_map_enabled`.
    /// Spot / point lights are currently unshadowed.
    pub fn set_shadow_and_ssao(
        &mut self,
        device: &wgpu::Device,
        ssao_view: &wgpu::TextureView,
        shadow_buffer: &wgpu::Buffer,
        light_camera_buffer: &wgpu::Buffer,
        glass_shadow_front_array_view: &wgpu::TextureView,
        glass_shadow_back_array_view: &wgpu::TextureView,
    ) {
        self.ssao_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade shadow+ssao+shadowmap bg"),
            layout: &self.ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(glass_shadow_front_array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(glass_shadow_back_array_view),
                },
            ],
        }));
    }

    /// Set shading data (params uniform, lights buffer, materials
    /// buffer). The user-shader params buffer is owned by this pass
    /// and bound automatically — callers don't pass it.
    pub fn set_shade_data(
        &mut self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        materials_buffer: &wgpu::Buffer,
    ) {
        self.shade_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade params bg"),
            layout: &self.shade_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: materials_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.shader_params_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Set camera uniform buffer.
    /// Set atmosphere LUT textures (all 4 LUTs + sampler).
    pub fn set_atmosphere_luts(
        &mut self,
        device: &wgpu::Device,
        transmittance_view: &wgpu::TextureView,
        multiscatter_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        sky_view_view: &wgpu::TextureView,
        ap_view: &wgpu::TextureView,
    ) {
        self.atmo_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade atmo bg"),
            layout: &self.atmo_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(transmittance_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(multiscatter_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(sky_view_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ap_view) },
            ],
        }));
    }

    pub fn set_camera(&mut self, device: &wgpu::Device, camera_buffer: &wgpu::Buffer) {
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade camera bg"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Dispatch the shading pass.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_with_timestamps(encoder, None);
    }

    pub fn dispatch_with_timestamps(&self, encoder: &mut wgpu::CommandEncoder, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>) {
        let gbuf = match &self.gbuffer_bind_group { Some(bg) => bg, None => return };
        let sao = match &self.ssao_bind_group { Some(bg) => bg, None => return };
        let shade = match &self.shade_bind_group { Some(bg) => bg, None => return };
        let cam = match &self.camera_bind_group { Some(bg) => bg, None => return };
        let atmo = match &self.atmo_bind_group { Some(bg) => bg, None => return };
        let brush_state = match &self.brush_state_bind_group { Some(bg) => bg, None => return };

        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rkp_shade"),
            timestamp_writes: timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gbuf, &[]);
        pass.set_bind_group(1, sao, &[]);
        pass.set_bind_group(2, &self.output_bind_group, &[]);
        pass.set_bind_group(3, shade, &[]);
        pass.set_bind_group(4, cam, &[]);
        pass.set_bind_group(5, atmo, &[]);
        pass.set_bind_group(6, brush_state, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Point the shade pass at an external output texture (e.g., the engine's
    /// shading HDR texture). Rebuilds the output bind group to write there.
    pub fn set_output_view(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) {
        self.output_bind_group =
            Self::create_output_bind_group(device, &self.output_bind_group_layout, view);
    }

    /// Resize output texture.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (tex, view) = Self::create_output(device, width, height);
        self.output_texture = tex;
        self.output_view = view;
        self.output_bind_group =
            Self::create_output_bind_group(device, &self.output_bind_group_layout, &self.output_view);
    }

    fn filterable_tex_2d(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
            }, count: None,
        }
    }

    fn filterable_tex_3d(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3, multisampled: false,
            }, count: None,
        }
    }

    fn create_output(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp_shade output"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::TEXTURE_BINDING
                 | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_output_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rkp_shade output bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }
}

/// Compose the shade-pass WGSL source. The user chunk replaces the
/// in-tree identity stub between the `USER_SHADE_DISPATCH_BEGIN` /
/// `_END` const-decl anchors in `rkp_shade.wgsl`. Pass `""` for the
/// no-shaders case (default identity stub stays put).
pub fn compose_shade_source(user_chunk: &str) -> String {
    let shade_src = wesl::include_wesl!("rkp_shade");
    crate::shader_composer::splice_const_marker(
        shade_src,
        "USER_SHADE_DISPATCH",
        user_chunk,
    )
}

fn build_shade_pipeline(
    device: &wgpu::Device,
    pipeline_layout: &wgpu::PipelineLayout,
    user_chunk: &str,
) -> wgpu::ComputePipeline {
    let source = compose_shade_source(user_chunk);
    let module = crate::compile_pass_shader(device, &source, "rkp_shade");
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rkp_shade"),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_params_size_is_128() {
        // Dropping the geodesic paint cursor freed brush_falloff +
        // brush_center (20 B) and added brush_object_id (4 B), netting
        // -16 B. Lock the new size — the WGSL `ShadeParams` mirror in
        // `lib/types.wesl` depends on it.
        assert_eq!(std::mem::size_of::<ShadeParams>(), 128);
    }

    #[test]
    fn rkp_shade_shader_is_valid_wgsl() {
        // Compose with no user chunk to exercise the in-tree
        // template (identity stub stays put).
        let src = compose_shade_source("");
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("parse error:\n{}", e.emit_to_string(&src)));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module).unwrap_or_else(|e| panic!("validation error: {e:?}"));
    }
}
