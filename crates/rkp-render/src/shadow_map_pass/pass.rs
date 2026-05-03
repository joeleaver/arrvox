//! `ShadowMapPass` — the GPU runtime: pipelines, buffers, and the
//! per-frame `dispatch_*` chain (clear → setup → emit → finalize →
//! scatter). Plus the small bind-group layout helpers and the
//! `ShadowScatterMarchParams` private uniform.

use crate::validate_wgsl;

use super::types::{
    LightCameraUniform, SetupParams, SCATTER_INSTANCE_STRIDE, SHADOW_MAP_MAX_CASTERS_INITIAL,
    SHADOW_MAP_WORK_LIST_INITIAL,
};

/// Pipeline holder for the work-list scatter shadow render. Owns
/// five compute pipelines (clear / setup / emit / finalize /
/// scatter), the depth-target storage buffer, the per-frame
/// uniforms, and per-frame scatter scratch (instances + work
/// list + counter + indirect args).
pub struct ShadowMapPass {
    pub size: u32,
    pub uniform_buffer: wgpu::Buffer,

    /// `array<atomic<u32>>` of length `size * size` — bit-cast
    /// f32 depths.
    pub shadow_buffer: wgpu::Buffer,

    setup_params_buffer: wgpu::Buffer,

    /// `atomic<u32>` global counter. Setup atomic-adds tile counts
    /// here; finalize reads it; engine zeros it before setup
    /// (`encoder.clear_buffer`).
    total_work_buffer: wgpu::Buffer,

    /// `array<u32>` — packed `(instance_idx:16, tile_x:8, tile_y:8)`
    /// per 8×8 tile. Filled by the emit pass; read by the scatter.
    pub work_list_buffer: wgpu::Buffer,

    /// `array<u32>` of length 4: `(x, y, z, total_work)`. Finalize
    /// writes; scatter reads `[3]` for bounds-check.
    pub dispatch_args_buffer: wgpu::Buffer,

    /// `array<ScatterInstance>` — written by setup, read by emit
    /// + scatter.
    pub scatter_instances_buffer: wgpu::Buffer,
    pub scatter_capacity: u32,

    // ── Pipelines + bind groups ────────────────────────────────
    clear_pipeline: wgpu::ComputePipeline,
    clear_g0_bg: wgpu::BindGroup,

    setup_pipeline: wgpu::ComputePipeline,
    setup_g0_layout: wgpu::BindGroupLayout,
    setup_g0_bg: Option<wgpu::BindGroup>,
    setup_g1_bg: wgpu::BindGroup,

    emit_pipeline: wgpu::ComputePipeline,
    emit_g0_layout: wgpu::BindGroupLayout,
    emit_g0_bg: wgpu::BindGroup,

    finalize_pipeline: wgpu::ComputePipeline,
    finalize_g0_bg: wgpu::BindGroup,

    scatter_pipeline: wgpu::ComputePipeline,
    scatter_pipeline_layout: wgpu::PipelineLayout,
    scatter_pass_layout: wgpu::BindGroupLayout,
    scatter_pass_bg: Option<wgpu::BindGroup>,
    user_shader_source_hash: u64,

    // Phase 4 — band-cell shadow dispatch bindings. The scatter
    // pass reads these to drive `dispatch_user_instance_descend`
    // for grass-style instance shaders. Engine sets them per
    // frame; the scatter bind group is rebuilt on changes.
    march_params_buffer: wgpu::Buffer,
    materials_buffer: Option<wgpu::Buffer>,
    shader_params_buffer: Option<wgpu::Buffer>,
}

impl ShadowMapPass {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        size: u32,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // ── Buffers ────────────────────────────────────────────
        let shadow_buffer_bytes = (size as u64) * (size as u64) * 4;
        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map shadow_buffer"),
            size: shadow_buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map light_camera_uniform"),
            size: std::mem::size_of::<LightCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let setup_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map setup_params"),
            size: std::mem::size_of::<SetupParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let total_work_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map total_work"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dispatch_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map dispatch_args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scatter_capacity = SHADOW_MAP_MAX_CASTERS_INITIAL;
        let scatter_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map scatter_instances"),
            size: (scatter_capacity as u64) * SCATTER_INSTANCE_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let work_list_capacity = SHADOW_MAP_WORK_LIST_INITIAL;
        let work_list_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_map work_list"),
            size: (work_list_capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Layouts ────────────────────────────────────────────
        let clear_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_clear g0"),
            entries: &[rw_storage_layout_entry(0)],
        });
        let setup_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_setup g0"),
            entries: &[
                ro_storage_layout_entry(0), // tlas_prims
                rw_storage_layout_entry(1), // scatter_instances
                rw_storage_layout_entry(2), // total_work
            ],
        });
        let setup_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_setup g1"),
            entries: &[uniform_layout_entry(0), uniform_layout_entry(1)],
        });
        let emit_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_emit g0"),
            entries: &[
                ro_storage_layout_entry(0), // scatter_instances
                rw_storage_layout_entry(1), // work_list
            ],
        });
        let finalize_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_finalize g0"),
            entries: &[
                rw_storage_layout_entry(0), // total_work
                rw_storage_layout_entry(1), // dispatch_args
            ],
        });
        let scatter_pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_scatter g1"),
            entries: &[
                uniform_layout_entry(0),    // light_camera
                rw_storage_layout_entry(1), // shadow_buffer (atomic)
                ro_storage_layout_entry(2), // scatter_instances
                ro_storage_layout_entry(3), // work_list
                ro_storage_layout_entry(4), // dispatch_args (read-only here)
                // Phase 4 — band-cell shadow dispatch bindings.
                ro_storage_layout_entry(5), // materials
                ro_storage_layout_entry(6), // shader_params
                uniform_layout_entry(7),    // march_params (lite mirror)
            ],
        });

        // ── Pipelines ──────────────────────────────────────────
        let clear_pipeline = build_pipeline(
            device, "shadow_clear",
            include_str!("../shaders/shadow_clear.wgsl"),
            "clear_main",
            &[Some(&clear_g0_layout)],
        );
        let setup_pipeline = build_pipeline(
            device, "shadow_scatter_setup",
            include_str!("../shaders/shadow_scatter_setup.wgsl"),
            "setup_main",
            &[Some(&setup_g0_layout), Some(&setup_g1_layout)],
        );
        let emit_pipeline = build_pipeline(
            device, "shadow_scatter_emit",
            include_str!("../shaders/shadow_scatter_emit.wgsl"),
            "emit_main",
            &[Some(&emit_g0_layout)],
        );
        let finalize_pipeline = build_pipeline(
            device, "shadow_scatter_finalize",
            include_str!("../shaders/shadow_scatter_finalize.wgsl"),
            "finalize_main",
            &[Some(&finalize_g0_layout)],
        );
        let scatter_src = include_str!("../shaders/shadow_scatter.wgsl");
        validate_wgsl(scatter_src, "shadow_scatter");
        let scatter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_scatter"),
            source: wgpu::ShaderSource::Wgsl(scatter_src.into()),
        });
        let scatter_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_scatter pipeline layout"),
            bind_group_layouts: &[
                Some(scene_bind_group_layout),
                Some(&scatter_pass_layout),
            ],
            immediate_size: 0,
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_scatter"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Bind groups (resources we own) ─────────────────────
        let clear_g0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_clear g0 bg"),
            layout: &clear_g0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_buffer.as_entire_binding(),
            }],
        });
        let setup_g1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_setup g1 bg"),
            layout: &setup_g1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: setup_params_buffer.as_entire_binding() },
            ],
        });
        let emit_g0_bg = build_emit_g0_bg(
            device, &emit_g0_layout,
            &scatter_instances_buffer, &work_list_buffer,
        );
        let finalize_g0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_finalize g0 bg"),
            layout: &finalize_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: total_work_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dispatch_args_buffer.as_entire_binding() },
            ],
        });
        // Phase 4 — band-cell shadow dispatch march_params buffer.
        // Tight 12-u32 mirror; engine writes it each frame via
        // `update_march_params`.
        let march_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_scatter march_params"),
            size: std::mem::size_of::<ShadowScatterMarchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            size,
            uniform_buffer,
            shadow_buffer,
            setup_params_buffer,
            total_work_buffer,
            work_list_buffer,
            dispatch_args_buffer,
            scatter_instances_buffer,
            scatter_capacity,
            clear_pipeline,
            clear_g0_bg,
            setup_pipeline,
            setup_g0_layout,
            setup_g0_bg: None,
            setup_g1_bg,
            emit_pipeline,
            emit_g0_layout,
            emit_g0_bg,
            finalize_pipeline,
            finalize_g0_bg,
            scatter_pipeline,
            scatter_pipeline_layout,
            scatter_pass_layout,
            scatter_pass_bg: None,
            user_shader_source_hash: 0,
            march_params_buffer,
            materials_buffer: None,
            shader_params_buffer: None,
        }
    }

    /// Phase 4 — set the materials buffer (shared with shade /
    /// march). The scatter bind group is rebuilt the next frame
    /// once both materials + shader_params are present.
    pub fn set_materials(&mut self, device: &wgpu::Device, materials: &wgpu::Buffer) {
        self.materials_buffer = Some(materials.clone());
        self.try_rebuild_scatter_pass_bg(device);
    }

    /// Phase 4 — set the per-material shader_params buffer.
    pub fn set_shader_params(&mut self, device: &wgpu::Device, shader_params: &wgpu::Buffer) {
        self.shader_params_buffer = Some(shader_params.clone());
        self.try_rebuild_scatter_pass_bg(device);
    }

    /// Phase 4 — write the lite march_params uniform. Engine calls
    /// each frame.
    pub fn update_march_params(&self, queue: &wgpu::Queue, time: f32, asset_count: u32) {
        let p = ShadowScatterMarchParams {
            object_count: 0,
            mode: 0,
            shadow_max_steps: 0,
            num_lights: 0,
            lod_enabled: 0,
            surfacenet_enabled: 0,
            tile_count_x: 0,
            tlas_node_count: 0,
            shadow_map_enabled: 0,
            time,
            asset_count,
            _pad0: 0,
        };
        queue.write_buffer(&self.march_params_buffer, 0, bytemuck::bytes_of(&p));
    }

    fn try_rebuild_scatter_pass_bg(&mut self, device: &wgpu::Device) {
        let (Some(materials), Some(shader_params)) = (
            &self.materials_buffer, &self.shader_params_buffer,
        ) else { return };
        self.scatter_pass_bg = Some(build_scatter_pass_bg(
            device, &self.scatter_pass_layout,
            &self.uniform_buffer, &self.shadow_buffer,
            &self.scatter_instances_buffer, &self.work_list_buffer,
            &self.dispatch_args_buffer, materials, shader_params,
            &self.march_params_buffer,
        ));
    }

    /// Rebuild the scatter pipeline against spliced user-shader chunks.
    /// Phase 4 — shadow-map scatter now splices the same
    /// `instance_at` chunk the primary march does, so band cells
    /// dispatch the user-shader prototype descent into the
    /// directional shadow path.
    pub fn reload_user_shaders(
        &mut self,
        device: &wgpu::Device,
        instance_at_chunk: &str,
        source_hash: u64,
    ) -> bool {
        if source_hash == self.user_shader_source_hash {
            return false;
        }
        let template = include_str!("../shaders/shadow_scatter.wgsl");
        let source = crate::shader_composer::splice_inst_chunks(
            template, instance_at_chunk,
        );
        validate_wgsl(&source, "shadow_scatter");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_scatter"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow_scatter"),
            layout: Some(&self.scatter_pipeline_layout),
            module: &module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.user_shader_source_hash = source_hash;
        true
    }

    /// Bind the TLAS prims buffer the setup pass reads.
    pub fn set_tlas_prims_buffer(
        &mut self,
        device: &wgpu::Device,
        tlas_prims: &wgpu::Buffer,
    ) {
        self.setup_g0_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_setup g0 bg"),
            layout: &self.setup_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tlas_prims.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.scatter_instances_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.total_work_buffer.as_entire_binding() },
            ],
        }));
    }

    /// Grow the scatter scratch + work-list buffers as needed.
    /// Engine calls this each frame before `dispatch_setup`.
    pub fn ensure_scatter_capacity(
        &mut self,
        device: &wgpu::Device,
        prim_count: u32,
    ) -> bool {
        let mut grew = false;
        if prim_count > self.scatter_capacity {
            let mut new_cap = self.scatter_capacity.max(1);
            while new_cap < prim_count {
                new_cap = new_cap.saturating_mul(2);
            }
            self.scatter_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_map scatter_instances"),
                size: (new_cap as u64) * SCATTER_INSTANCE_STRIDE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.scatter_capacity = new_cap;
            self.setup_g0_bg = None; // engine rebinds via set_tlas_prims_buffer
            self.emit_g0_bg = build_emit_g0_bg(
                device, &self.emit_g0_layout,
                &self.scatter_instances_buffer, &self.work_list_buffer,
            );
            // scatter_pass_bg references the resized
            // scatter_instances_buffer; rebuild on the next call to
            // `try_rebuild_scatter_pass_bg` once materials +
            // shader_params are present.
            self.scatter_pass_bg = None;
            self.try_rebuild_scatter_pass_bg(device);
            grew = true;
        }
        grew
    }

    /// Record the clear pass — fills `shadow_buffer` with FAR_DEPTH
    /// bits AND zeros `total_work` for the upcoming setup pass.
    pub fn dispatch_clear(&self, encoder: &mut wgpu::CommandEncoder) {
        // total_work counter must start at 0 each frame.
        encoder.clear_buffer(&self.total_work_buffer, 0, Some(4));
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_clear"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.clear_pipeline);
        cpass.set_bind_group(0, &self.clear_g0_bg, &[]);
        let groups = self.size.div_ceil(8);
        cpass.dispatch_workgroups(groups, groups, 1);
    }

    /// Record the setup pass — projects TLAS prims, fills
    /// `scatter_instances`, atomic-adds tile counts to `total_work`.
    /// `camera_view_proj` and `scene_extent` drive the shadow-
    /// frustum cull (skip prims whose swept volume can't reach
    /// the camera view).
    pub fn dispatch_setup(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        prim_count: u32,
        camera_view_proj: [[f32; 4]; 4],
        scene_extent: f32,
    ) {
        let Some(ref g0) = self.setup_g0_bg else { return; };
        queue.write_buffer(
            &self.setup_params_buffer,
            0,
            bytemuck::bytes_of(&SetupParams {
                prim_count,
                scene_extent,
                _pad0: 0, _pad1: 0,
                camera_view_proj,
            }),
        );
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_setup"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.setup_pipeline);
        cpass.set_bind_group(0, g0, &[]);
        cpass.set_bind_group(1, &self.setup_g1_bg, &[]);
        let workgroups = self.scatter_capacity.div_ceil(64);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Record the emit pass — fills `work_list[scatter_instances[i]
    /// .work_offset + 0..tile_count]` with packed work tuples.
    /// One workgroup of 64 threads per instance.
    pub fn dispatch_emit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prim_count: u32,
    ) {
        if prim_count == 0 { return; }
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_emit"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.emit_pipeline);
        cpass.set_bind_group(0, &self.emit_g0_bg, &[]);
        cpass.dispatch_workgroups(prim_count, 1, 1);
    }

    /// Record the finalize pass — packs `total_work` into the
    /// scatter pass's indirect-dispatch args.
    pub fn dispatch_finalize(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter_finalize"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.finalize_pipeline);
        cpass.set_bind_group(0, &self.finalize_g0_bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }

    /// Record the scatter pass — single indirect dispatch over
    /// the work list. Each workgroup descends one instance for
    /// one 8×8 tile, atomic-mins depth into `shadow_buffer`.
    pub fn dispatch_scatter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_bind_group: &wgpu::BindGroup,
        prim_count: u32,
    ) {
        if prim_count == 0 { return; }
        // Phase 4 — scatter_pass_bg is built lazily once
        // materials + shader_params land on the pass; if the engine
        // hasn't wired them yet, the scatter dispatch is skipped
        // (correct behavior: no work, no shadow casters).
        let Some(ref scatter_bg) = self.scatter_pass_bg else { return; };
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow_scatter"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.scatter_pipeline);
        cpass.set_bind_group(0, scene_bind_group, &[]);
        cpass.set_bind_group(1, scatter_bg, &[]);
        cpass.dispatch_workgroups_indirect(&self.dispatch_args_buffer, 0);
    }
}

/// Phase 4 — scatter pass's lite march_params uniform mirror.
/// Layout matches `octree_march::MarchParams` (uniform-storage
/// alignment safe; total 48 B).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowScatterMarchParams {
    object_count: u32,
    mode: u32,
    shadow_max_steps: u32,
    num_lights: u32,
    lod_enabled: u32,
    surfacenet_enabled: u32,
    tile_count_x: u32,
    tlas_node_count: u32,
    shadow_map_enabled: u32,
    time: f32,
    asset_count: u32,
    _pad0: u32,
}

fn ro_storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    entry_point: &str,
    bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::ComputePipeline {
    validate_wgsl(src, label);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} pipeline layout")),
        bind_group_layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn build_emit_g0_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scatter_instances_buffer: &wgpu::Buffer,
    work_list_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_emit g0 bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scatter_instances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: work_list_buffer.as_entire_binding() },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn build_scatter_pass_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    shadow_buffer: &wgpu::Buffer,
    scatter_instances_buffer: &wgpu::Buffer,
    work_list_buffer: &wgpu::Buffer,
    dispatch_args_buffer: &wgpu::Buffer,
    materials_buffer: &wgpu::Buffer,
    shader_params_buffer: &wgpu::Buffer,
    march_params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow_scatter pass bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: shadow_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scatter_instances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: work_list_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dispatch_args_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: materials_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: shader_params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: march_params_buffer.as_entire_binding() },
        ],
    })
}
