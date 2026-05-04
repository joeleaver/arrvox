//! Procedural CSG raymarcher — live preview pass for the build viewport.
//!
//! Runs as a compute dispatch (one thread per pixel), sphere-traces the
//! flattened RPN tree against each camera ray, and writes into the same
//! G-buffer layout (`position`, `normal`, `material`) as
//! [`crate::octree_march::OctreeMarchPass`]. Downstream passes (shadow,
//! SSAO, shade, post) don't distinguish its output from the voxel march.
//!
//! The shader input is an array of `ProcInstruction` (see
//! `rkp_procedural::flatten`) — a post-order RPN encoding of the tree.
//! The CPU flattens once per edit; re-uploads to the GPU are cheap
//! (O(tree size), tens of bytes per node). This replaces the voxel
//! march in the build viewport when the user toggles live preview —
//! the voxel path remains the source of truth elsewhere.
//!
//! This pass owns no shared state with `RkpScene`: it binds only the
//! viewport's camera buffer (group 0), the G-buffer textures (group 1),
//! and its own params + instruction buffer (group 2). Keeping it
//! decoupled means adding or removing the preview in the pipeline is a
//! simple toggle at the frame-driver level.

use crate::compile_pass_shader;
use bytemuck::{Pod, Zeroable};
use rkp_procedural::flatten::ProcInstruction;

/// Params uniform mirroring the shader struct of the same name.
///
/// `instruction_count` is the number of entries in the instructions
/// buffer the shader should execute — *not* the buffer's capacity. We
/// grow the buffer to the high-water mark and reuse it across bakes;
/// trees that get smaller just ignore the tail.
///
/// `object_id` is packed into the G-buffer material channel's pick
/// byte so this procedural's hits show up like any other entity when
/// the pick shader reads back.
///
/// `entity_world` / `entity_inverse_world` carry the owning entity's
/// world transform (and its inverse) so the shader can march the ray
/// in the entity's local frame and then convert the hit back to world
/// for the G-buffer. Without this, moving the entity in world space
/// shifts the procedural preview out from under the camera — the tree
/// primitives' own `inverse_world` only encodes intra-tree hierarchy.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ProcRaymarchParams {
    pub instruction_count: u32,
    pub object_id: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub entity_world: [[f32; 4]; 4],
    pub entity_inverse_world: [[f32; 4]; 4],
    /// Tree bounding box in *entity-local* space (the frame the shader
    /// marches in). The shader does a cheap ray/AABB slab test before
    /// sphere-tracing; any pixel whose ray misses the AABB entirely
    /// is guaranteed to miss the geometry and can write the G-buffer
    /// "miss" state and bail. For a small procedural in a big
    /// viewport that skips the vast majority of pixels.
    ///
    /// Padded to `vec4` so the WGSL struct layout matches (vec3's
    /// natural alignment is 16 B in uniform storage).
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
}

/// Procedural CSG raymarch compute pass.
pub struct ProcRaymarchPass {
    /// Pipeline variants keyed on the `HAS_POS_WARPS` shader override.
    /// `pipeline_simple` compiles with the override set to `false`,
    /// which lets the compiler dead-strip the entire `pos_stack` +
    /// all PUSH/POP branches — a measurable win in the common case
    /// where the user's tree is primitives + combinators + post-op
    /// effects only. `pipeline_warps` keeps the full logic for trees
    /// that use NoiseDisplace or Mirror. `upload_instructions` flips
    /// `has_warps` based on the flattened tree's opcodes; `dispatch`
    /// picks the matching pipeline.
    pipeline_simple: wgpu::ComputePipeline,
    pipeline_warps: wgpu::ComputePipeline,
    has_warps: bool,

    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: Option<wgpu::BindGroup>,

    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: Option<wgpu::BindGroup>,

    params_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group: Option<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,

    /// Storage buffer holding the current flattened instruction stream.
    /// Grows on demand; never shrinks (shader only reads the first
    /// `instruction_count` entries).
    instructions_buffer: wgpu::Buffer,
    instructions_capacity: usize,
}

impl ProcRaymarchPass {
    pub fn new(device: &wgpu::Device) -> Self {
        // Group 0: camera uniform. Mirrors `RkpScene`'s camera binding
        // shape (`CameraUniforms` 224 B). We declare our own layout
        // rather than reuse the scene layout because we don't need the
        // scene buffers — keeps the pass self-contained.
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch camera layout"),
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

        // Group 1: G-buffer (same triple as OctreeMarch for position,
        // normal, material) plus a 4th rkp-side pick texture. The pick
        // slot holds the hit primitive's NodeId so packed_r's high 16
        // bits on the shared material G-buffer stay available for
        // `secondary_material_id` — which is what `rkp_shade` reads
        // for dual-material lerp. Not part of the shared GBuffer
        // because it's rkp-specific; see `ViewportRenderer::pick_view`.
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch gbuffer layout"),
                entries: &[
                    bgl_storage_tex(0, wgpu::TextureFormat::Rgba32Float),
                    bgl_storage_tex(1, wgpu::TextureFormat::Rgba16Float),
                    bgl_storage_tex(2, wgpu::TextureFormat::Rg32Uint),
                    bgl_storage_tex(3, wgpu::TextureFormat::R32Uint),
                    // Glass target — procedural primitives don't do
                    // glass (yet); the shader writes 0 to keep this
                    // buffer coherent with what octree_march wrote
                    // elsewhere in the frame.
                    bgl_storage_tex(4, wgpu::TextureFormat::Rg32Uint),
                    // Leaf-slot target — procedurals have no stable
                    // leaf_attr_slot (they're analytical), so the
                    // shader writes 0 (the sentinel the paint cursor
                    // treats as "no hit").
                    bgl_storage_tex(5, wgpu::TextureFormat::R32Uint),
                ],
            });

        // Group 2: params + instructions.
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("proc_raymarch params layout"),
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
                ],
            });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_raymarch params"),
            size: std::mem::size_of::<ProcRaymarchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Minimal seed size — we'll grow this on first upload. The
        // shader reads up to `instruction_count` entries, never past
        // the buffer, so even a zero-instruction tree renders fine
        // against this stub.
        let initial_cap = 4usize;
        let instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("proc_raymarch instructions"),
            size: (initial_cap * std::mem::size_of::<ProcInstruction>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // WESL resolves the imports from `proc_eval_types` (types +
        // opcodes) and `proc_eval` (function bodies); the function
        // bodies reference the `instructions` binding declared below
        // as a module-scope global, resolved at the consumer's import
        // site. WGSL resolves functions across the whole module
        // regardless of declaration order, so `main` calling
        // `eval_tree` before its body appears in the emit works.
        let module = compile_pass_shader(device, wesl::include_wesl!("proc_raymarch"), "proc_raymarch");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("proc_raymarch pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&gbuffer_bind_group_layout),
                Some(&params_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, has_warps: bool| -> wgpu::ComputePipeline {
            let overrides: &[(&str, f64)] = if has_warps {
                &[("HAS_POS_WARPS", 1.0)]
            } else {
                &[("HAS_POS_WARPS", 0.0)]
            };
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: overrides,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };

        let pipeline_simple = make_pipeline("proc_raymarch (simple)", false);
        let pipeline_warps = make_pipeline("proc_raymarch (warps)", true);

        Self {
            pipeline_simple,
            pipeline_warps,
            has_warps: false,
            camera_bind_group_layout,
            camera_bind_group: None,
            gbuffer_bind_group_layout,
            gbuffer_bind_group: None,
            params_bind_group_layout,
            params_bind_group: None,
            params_buffer,
            instructions_buffer,
            instructions_capacity: initial_cap,
        }
    }

    /// Wire the viewport's camera uniform into this pass.
    pub fn set_camera(&mut self, device: &wgpu::Device, camera_buffer: &wgpu::Buffer) {
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch camera"),
            layout: &self.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        }));
    }

    /// Wire the viewport's G-buffer views into this pass. Re-call after
    /// a G-buffer resize. `pick_view` is the rkp-side `R32Uint` pick
    /// texture that receives the hit primitive's NodeId.
    #[allow(clippy::too_many_arguments)]
    pub fn set_gbuffer(
        &mut self,
        device: &wgpu::Device,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
        leaf_slot_view: &wgpu::TextureView,
    ) {
        self.gbuffer_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch gbuffer"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(pick_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(glass_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(leaf_slot_view),
                },
            ],
        }));
        self.rebuild_params_bind_group(device);
    }

    /// Upload (or replace) the flattened instruction stream. If the
    /// new stream exceeds the current buffer capacity we grow the
    /// buffer (doubling past-capacity; keeps the amortized cost
    /// proportional to a single upload) and rebuild the bind group.
    /// Zero-length streams are valid — the shader writes "miss" to
    /// every pixel.
    pub fn upload_instructions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instructions: &[ProcInstruction],
    ) {
        if instructions.len() > self.instructions_capacity {
            let new_cap = instructions.len().next_power_of_two().max(4);
            self.instructions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("proc_raymarch instructions"),
                size: (new_cap * std::mem::size_of::<ProcInstruction>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instructions_capacity = new_cap;
            self.rebuild_params_bind_group(device);
        }
        // Pipeline variant selection: only trees with at least one
        // PUSH opcode (NoiseDisplace or Mirror) need the pos_stack.
        // Everything else — primitives, combinators, post-op
        // attribute rewrites — uses the simple pipeline so the
        // shader compiler can eliminate the stack entirely.
        self.has_warps = instructions.iter().any(|ins| {
            ins.op == rkp_procedural::OpKind::PushNoiseDisplace as u32
                || ins.op == rkp_procedural::OpKind::PushMirror as u32
                || ins.op == rkp_procedural::OpKind::PushArray as u32
        });
        if !instructions.is_empty() {
            queue.write_buffer(&self.instructions_buffer, 0, bytemuck::cast_slice(instructions));
        }
    }

    /// Update the params uniform.
    ///
    /// `entity_world` is the owning entity's world transform; the
    /// shader uses its inverse to pull the camera ray into the entity's
    /// local frame before marching, and the forward transform to push
    /// the hit position + normal back to world space for the G-buffer.
    /// Pass [`glam::Affine3A::IDENTITY`] for entities that live at the
    /// world origin — the math still works, just with no shift.
    pub fn set_params(
        &self,
        queue: &wgpu::Queue,
        instruction_count: u32,
        object_id: u32,
        entity_world: glam::Affine3A,
        aabb_min: glam::Vec3,
        aabb_max: glam::Vec3,
    ) {
        let world = glam::Mat4::from(entity_world);
        let inverse = world.inverse();
        let p = ProcRaymarchParams {
            instruction_count,
            object_id,
            _pad0: 0,
            _pad1: 0,
            entity_world: world.to_cols_array_2d(),
            entity_inverse_world: inverse.to_cols_array_2d(),
            aabb_min: [aabb_min.x, aabb_min.y, aabb_min.z, 0.0],
            aabb_max: [aabb_max.x, aabb_max.y, aabb_max.z, 0.0],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&p));
    }

    fn rebuild_params_bind_group(&mut self, device: &wgpu::Device) {
        self.params_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("proc_raymarch params"),
            layout: &self.params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instructions_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Record the compute dispatch. Caller is responsible for having
    /// called `set_camera` / `set_gbuffer` / `upload_instructions` /
    /// `set_params` first — we skip silently if any bind group is
    /// unset rather than panic inside an encoder.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let (Some(cam), Some(gbuf), Some(params)) = (
            &self.camera_bind_group,
            &self.gbuffer_bind_group,
            &self.params_bind_group,
        ) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("proc_raymarch"),
            timestamp_writes,
        });
        let pipeline = if self.has_warps {
            &self.pipeline_warps
        } else {
            &self.pipeline_simple
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, cam, &[]);
        pass.set_bind_group(1, gbuf, &[]);
        pass.set_bind_group(2, params, &[]);
        let gx = width.div_ceil(8);
        let gy = height.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

fn bgl_storage_tex(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
