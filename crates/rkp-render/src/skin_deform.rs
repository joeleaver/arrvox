//! Phase-3a skin-deform scatter pass — CPU driver.
//!
//! Builds + dispatches the [`skin_deform.wgsl`] compute shader once per
//! skinned entity per frame. Scatters each leaf's `(bone_indices,
//! bone_weights)` into a deformed-space 3D bone field so the march
//! shader's Phase-3b inverse-skin branch has per-sample bone data
//! without doing an octree descent of its own.
//!
//! See the plan file for the architectural rationale
//! (`/home/joe/.claude/plans/concurrent-humming-quail.md`, Phase 3).
//! The scatter pattern follows rkifield's
//! `crates/rkf-render/shaders/skin_deform.wgsl` but adapts to rkipatch's
//! brick-terminated octree + shell-only leaves.

use bytemuck::{Pod, Zeroable};

use crate::rkp_scene::RkpScene;

/// Per-entity constants consumed by the scatter shader. One entry per
/// skinned entity in the scene-wide uniforms array each frame; the
/// brick list points at its owner via `uniform_idx`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkinUniforms {
    pub bone_buffer_offset: u32,
    pub bone_count: u32,
    pub bone_field_offset: u32,
    pub bone_field_dim_x: u32,
    pub bone_field_dim_y: u32,
    pub bone_field_dim_z: u32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub voxel_size: f32,
    /// Offset into the scene-wide bone-field occupancy bitmap in u32
    /// words. Each bit covers one 4³-cell brick of this entity's slice.
    pub bone_field_occ_offset: u32,
    /// `0` = Linear Blend Skinning (4-bone weighted matrix sum);
    /// `1` = Dual-Quaternion Skinning (rigid interpolation through
    /// the dual-quat manifold — preserves joint volume, no "candy
    /// wrapper" on axial twists).
    pub skinning_mode: u32,
    /// Offset into the scene-wide `bone_dual_quats` buffer in DualQuat
    /// (32-byte) units. One DQ per bone, forward-pose-only.
    pub bone_dq_offset: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// One entry in the scene-wide `brick_list` storage buffer. Each
/// scatter workgroup is pinned to one brick and finds its per-entity
/// uniforms via `uniform_idx`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SkinBrickEntry {
    pub brick_id: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub origin_z: u32,
    pub uniform_idx: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// One per-entity plan, consumed by [`SkinDeformPass::run`] as a batch.
pub struct SkinDispatch<'a> {
    /// Per-entity uniforms — becomes one entry in the scene-wide
    /// `SkinUniforms` array, indexed by the entity's position in the
    /// batch.
    pub uniforms: SkinUniforms,
    /// Populated bricks for this entity. Appended to the scene-wide
    /// brick list with `uniform_idx` set to the entity's index.
    pub bricks: &'a [SkinBrickEntry],
}

/// Scatter pass driver. Owns one compute pipeline + transient
/// per-dispatch bind groups.
///
/// The scatter reads a subset of the scene's buffers (brick_pool,
/// bone_matrices, bone_weights) and writes the bone field. We cannot
/// reuse `RkpScene::bind_group` directly — that layout binds the bone
/// field as storage-read-only for the march, and WGSL forbids aliasing
/// a buffer as both read-only and read-write within one dispatch. So
/// the scatter has its own `scene_bind_group_layout` that references
/// only the buffers it needs + the bone field as read-write, and we
/// rebuild the bind group each time the scene re-uploads geometry.
pub struct SkinDeformPass {
    pipeline: wgpu::ComputePipeline,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    dispatch_bind_group_layout: wgpu::BindGroupLayout,
    uniforms_buffer: wgpu::Buffer,
    uniforms_capacity: u64,
    bricks_capacity: u64,
    bricks_buffer: wgpu::Buffer,
}

/// Scratch buffer the CPU reuses every frame when concatenating
/// multiple `SkinDispatch`es into the two scene-wide scatter inputs.
#[derive(Default, Clone)]
pub struct SkinBatchScratch {
    pub uniforms: Vec<SkinUniforms>,
    pub bricks: Vec<SkinBrickEntry>,
}

impl SkinBatchScratch {
    pub fn clear(&mut self) {
        self.uniforms.clear();
        self.bricks.clear();
    }

    pub fn push(&mut self, d: &SkinDispatch<'_>) {
        let uniform_idx = self.uniforms.len() as u32;
        self.uniforms.push(d.uniforms);
        self.bricks.extend(d.bricks.iter().map(|b| SkinBrickEntry {
            brick_id: b.brick_id,
            origin_x: b.origin_x,
            origin_y: b.origin_y,
            origin_z: b.origin_z,
            uniform_idx,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }));
    }

    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }

    pub fn total_bricks(&self) -> u32 {
        self.bricks.len() as u32
    }
}

impl SkinDeformPass {
    pub fn new(device: &wgpu::Device, scene: &RkpScene) -> Self {
        // Scene-side layout — bindings 0/5/6 match the scene's main
        // layout so the shader can share binding numbers, but the bone
        // field sits on binding 9 here as read-write (vs. read-only in
        // the main march layout).
        let scene_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let scene_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin_deform scene layout"),
            entries: &[
                scene_ro(0),  // brick_pool
                scene_ro(5),  // bone_matrices
                scene_ro(6),  // bone_weights
                scene_ro(8),  // leaf_attr_pool — scatter reads rest normals to rotate
                wgpu::BindGroupLayoutEntry {
                    binding: 9, // bone_field (read_write here)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10, // bone_field_occ (atomicOr on scatter)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                scene_ro(11), // bone_dual_quats (DQS precomputed palette)
            ],
        });

        let scene_bind_group = Self::create_scene_bind_group(device, &scene_bind_group_layout, scene);

        let dispatch_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skin_deform batch layout"),
            entries: &[
                // uniforms array — one entry per skinned entity this frame
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // brick_list — concatenated bricks across every entity
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
            ],
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_deform uniforms"),
            size: std::mem::size_of::<SkinUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Start at one-entry capacity; `run` grows with the total
        // brick count across every skinned entity this frame.
        let bricks_capacity = std::mem::size_of::<SkinBrickEntry>() as u64;
        let bricks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_deform brick_list"),
            size: bricks_capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = crate::compile_pass_shader(device, wesl::include_wesl!("skin_deform"), "skin_deform");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skin_deform pipeline"),
            bind_group_layouts: &[
                Some(&scene_bind_group_layout),    // group 0 — scene subset (bone_field RW)
                Some(&dispatch_bind_group_layout), // group 1 — per-dispatch
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin_deform"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            scene_bind_group_layout,
            scene_bind_group,
            dispatch_bind_group_layout,
            uniforms_buffer,
            uniforms_capacity: std::mem::size_of::<SkinUniforms>() as u64,
            bricks_capacity,
            bricks_buffer,
        }
    }

    fn create_scene_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene: &RkpScene,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin_deform scene bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene.brick_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: scene.bone_matrices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: scene.bone_weights_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: scene.leaf_attr_pool_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: scene.bone_field_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: scene.bone_field_occ_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: scene.bone_dual_quats_buffer.as_entire_binding() },
            ],
        })
    }

    /// Rebuild the scatter's scene bind group. Call after any buffer
    /// in [`RkpScene`] the scatter reads has been reallocated — e.g.
    /// `upload_geometry` grew the brick_pool, or the bone_field
    /// buffer was resized. The march's main bind group is rebuilt
    /// separately by `RkpScene::rebuild_bind_group`.
    pub fn refresh_scene_bind_group(&mut self, device: &wgpu::Device, scene: &RkpScene) {
        self.scene_bind_group = Self::create_scene_bind_group(device, &self.scene_bind_group_layout, scene);
    }

    /// Clear the scene bone_field to zero before the batched scatter
    /// dispatch. Caller sizes it first via
    /// `RkpScene::ensure_bone_field_capacity`.
    pub fn clear_bone_field(&self, encoder: &mut wgpu::CommandEncoder, scene: &RkpScene) {
        encoder.clear_buffer(&scene.bone_field_buffer, 0, None);
    }

    /// Run the batched scatter — one compute dispatch covering every
    /// skinned entity this frame. Workgroup count is the total brick
    /// count; each workgroup picks its per-entity uniforms via
    /// `SkinBrickEntry.uniform_idx`. A single GPU command fires after
    /// a single `write_buffer` per input, so the write-ordering foot-
    /// gun of per-entity dispatch + shared buffer is gone.
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _scene: &RkpScene,
        batch: &SkinBatchScratch,
    ) {
        if batch.is_empty() { return; }

        // ── Uniforms array ─────────────────────────────────────────
        let uniforms_bytes: &[u8] = bytemuck::cast_slice(&batch.uniforms);
        // Track our own capacity instead of querying `buffer.size()`.
        // wgpu can report a stale-feeling value when a buffer was
        // recreated via `*self.foo = device.create_buffer(...)` while
        // an earlier bind-group still references the old `Arc`-backed
        // handle, and the validator reports the OLD buffer's size in
        // its error path. Storing the cap explicitly + rebuilding the
        // bind group on resize matches the existing `bricks_buffer`
        // pattern below.
        let needed = uniforms_bytes.len() as u64;
        if needed > self.uniforms_capacity {
            let mut new_cap = self.uniforms_capacity.max(std::mem::size_of::<SkinUniforms>() as u64);
            while new_cap < needed {
                new_cap = new_cap.saturating_mul(2);
            }
            self.uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_deform uniforms"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.uniforms_capacity = new_cap;
        }
        queue.write_buffer(&self.uniforms_buffer, 0, uniforms_bytes);

        // ── Brick list ─────────────────────────────────────────────
        let bricks_bytes: &[u8] = bytemuck::cast_slice(&batch.bricks);
        let required = bricks_bytes.len() as u64;
        if required > self.bricks_capacity {
            let mut new_cap = self.bricks_capacity.max(64);
            while new_cap < required {
                new_cap = new_cap.saturating_mul(2);
            }
            self.bricks_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_deform brick_list"),
                size: new_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bricks_capacity = new_cap;
        }
        queue.write_buffer(&self.bricks_buffer, 0, bricks_bytes);

        let batch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin_deform batch bg"),
            layout: &self.dispatch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bricks_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skin_deform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.scene_bind_group, &[]);
        pass.set_bind_group(1, &batch_bg, &[]);
        pass.dispatch_workgroups(batch.total_bricks(), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn skin_deform_shader_is_valid_wgsl() {
        let src = wesl::include_wesl!("skin_deform");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
            panic!("skin_deform.wgsl parse error: {}", e.emit_to_string(src));
        });
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module)
            .unwrap_or_else(|e| panic!("skin_deform.wgsl validation error: {e:?}"));
    }
}
