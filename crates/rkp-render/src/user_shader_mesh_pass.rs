//! `UserShaderMeshPass` — vertex-shader-driven user-shader path.
//!
//! See `notes/user-shaders-mesh.md` for the full V1 plan. This module
//! owns the bind-group layouts + pipeline-layout objects shared by
//! every user-shader material (they don't depend on the user's code)
//! plus a set of stub pipelines built from the engine skeleton, used
//! as a fallback and to surface layout errors at startup.
//!
//! Per-shader pipelines are built lazily by the orchestration layer
//! in `rkp-engine`'s `user_shader_tick.rs` — that step composes the
//! user's WGSL into the skeleton via `compile_user_shader_pipelines`
//! and caches the result keyed on `(shader_id, source_hash)`.
//!
//! Scheduling order each frame (per active user-shader material):
//!   1. spawn_count compute  — one thread per anchor → out_counts[i]
//!   2. prefix_sum compute   — exclusive scan + indirect args write
//!   3. fill compute         — fan out per-spawn InstanceRecords
//!   4. raster               — indirect draw against own G-buffer pass
//!   5. shadow VS            — per-cascade depth-only indirect draws
//!
//! Bindings:
//!   Raster:    g0 = camera, g1 = (anchors RO, records RO, frame, params)
//!   Compute:   g0 = (anchors RO, counts RW, offsets RW, records RW,
//!                    indirect RW, frame, params, dispatch)

use bytemuck::{Pod, Zeroable};

use crate::gbuffer::{
    GBUFFER_DEPTH_FORMAT, GBUFFER_GLASS_FORMAT, GBUFFER_LEAF_SLOT_FORMAT,
    GBUFFER_MATERIAL_FORMAT, GBUFFER_NORMAL_FORMAT, GBUFFER_POSITION_FORMAT,
};

const GBUFFER_PICK_FORMAT: wgpu::TextureFormat = GBUFFER_LEAF_SLOT_FORMAT;

// ─── Data types shared with rkp-engine ─────────────────────────────

/// One painted *tile*, partitioned by user-shader material. CPU packs
/// one per (entity × material × tile-coord) tuple whose material has
/// a user shader registered against it. Tile size comes from the
/// shader's `@tile_size` directive; without it, the whole entity's
/// painted region for this material collapses into a single anchor
/// (the tile cube falls back to the painted-leaf AABB).
///
/// **Bounds are the tile cube**, not the painted-leaf AABB. Stable
/// across paint additions inside the tile, so a hashed
/// `mix(tile_min, tile_max, r)` blade position doesn't shift when
/// the user paints more inside the same tile. The painted-leaf
/// bounds are tracked separately for `surface_y` only.
///
/// Layout mirrors WGSL `AnchorContext` in `user_shader_mesh.wesl`:
///   offset  0..12  tile_min           vec3<f32>
///   offset 12..16  material_id        u32       (packs with vec3's tail)
///   offset 16..28  tile_max           vec3<f32>
///   offset 28..32  leaf_count         u32       (density signal)
///   offset 32..36  object_id          u32
///   offset 36..40  surface_y          f32       (blade base y; V1 = painted aabb.max.y)
///   offset 40..44  seed               u32
///   offset 44..64  _pad               5×u32
///
/// 64 B total — aligns to 16 (WGSL std430). Field offsets asserted
/// at compile time.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct AnchorRecord {
    pub tile_min: [f32; 3],
    pub material_id: u32,
    pub tile_max: [f32; 3],
    pub leaf_count: u32,
    pub object_id: u32,
    pub surface_y: f32,
    pub seed: u32,
    pub _pad: [u32; 5],
}

const _: () = assert!(std::mem::size_of::<AnchorRecord>() == 64);
const _: () = {
    use std::mem::offset_of;
    assert!(offset_of!(AnchorRecord, tile_min) == 0);
    assert!(offset_of!(AnchorRecord, material_id) == 12);
    assert!(offset_of!(AnchorRecord, tile_max) == 16);
    assert!(offset_of!(AnchorRecord, leaf_count) == 28);
    assert!(offset_of!(AnchorRecord, object_id) == 32);
    assert!(offset_of!(AnchorRecord, surface_y) == 36);
    assert!(offset_of!(AnchorRecord, seed) == 40);
};

/// Per-frame engine uniforms uploaded once per render. Layout matches
/// `FrameContext` in both `user_shader_mesh.wesl` and
/// `user_shader_mesh_compute.wesl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct FrameUniforms {
    pub time: f32,
    pub delta_time: f32,
    pub _pad0: [f32; 2],
    pub wind_dir: [f32; 3],
    pub wind_strength: f32,
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
}

const _: () = assert!(std::mem::size_of::<FrameUniforms>() == 48);

/// V1 user-shader parameter buffer. Eight f32s packed as two vec4s so
/// std140 / std430 alignment is unambiguous. Composer assigns
/// positional indices to the user's `@param` names; the user's WGSL
/// reads them via `ctx_param(i)`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct UserShaderParams {
    pub p: [f32; 8],
}

const _: () = assert!(std::mem::size_of::<UserShaderParams>() == 32);

/// Dispatch metadata uploaded once per (shader, frame). Tells the
/// compute passes the active anchor count and the procedural geometry
/// shape (vertices per spawn).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct DispatchInfo {
    pub num_anchors: u32,
    pub verts_per_spawn: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<DispatchInfo>() == 16);

/// Output of `entry_prefix_sum` — single indexed-indirect args
/// quadruple. wgpu's `DrawIndirectArgs` matches this layout; we use
/// an explicit struct so the WGSL side reads named fields.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

const _: () = assert!(std::mem::size_of::<DrawIndirectArgs>() == 16);

/// One instance record per allocated spawn. The fill pass writes one
/// of these for every spawn in `[0, total_spawns)`; the VS reads it
/// via `@builtin(instance_index)` and dereferences `anchors[anchor_idx]`.
///
/// Dead spawns (filtered by `spawn_alive`) carry `anchor_idx =
/// 0xFFFFFFFF`; the VS short-circuits with a clipped vertex so the
/// rasterizer culls the triangle.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct InstanceRecord {
    pub anchor_idx: u32,
    pub spawn_idx: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceRecord>() == 8);

/// Stable per-anchor seed. Hashes the bit-pattern of `world_pos` so the
/// same painted leaf produces the same seed across frames — required
/// for deterministic per-spawn variation (yaw, jitter, height) that
/// doesn't shimmer between frames.
///
/// Variant of the splitmix64 finalizer applied to xor-folded f32 bits.
pub fn anchor_seed(world_pos: [f32; 3]) -> u32 {
    let bx = world_pos[0].to_bits();
    let by = world_pos[1].to_bits();
    let bz = world_pos[2].to_bits();
    let mut x = bx ^ by.wrapping_mul(0x9E37_79B9) ^ bz.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// V1 ceiling on anchors per user-shader material — the prefix-sum
/// is a 3-pass Blelloch scan of `PREFIX_SUM_WG_SIZE *
/// PREFIX_SUM_MAX_WG_COUNT` elements (256 × 32 = 8192). Note: the
/// "anchor = painted-leaf" design hits this cap quickly on any
/// reasonable hillside paint (24k+ anchors observed at ~38 m²); a
/// shape-of-design rework lives on the V1.1 follow-up list.
pub const MAX_ANCHORS_PER_SHADER_V1: u32 = 8192;

/// Per-workgroup-sum slots used by the Blelloch scan. Must match
/// `PREFIX_SUM_MAX_WG_COUNT` in `user_shader_mesh_compute.wesl`.
pub const PREFIX_SUM_MAX_WG_COUNT: u32 = 32;

// ─── Pipeline objects ──────────────────────────────────────────────

/// Pipeline + layout owner for the V1 user-shader mesh path. The
/// `stub_*` pipelines are built from the engine skeleton (no user
/// code spliced). They serve two purposes:
///   1. Surface any layout / binding mismatch at startup, before any
///      user shader composes.
///   2. Fallback path when a user shader fails to compose at runtime.
pub struct UserShaderMeshPass {
    pub raster_g0_layout: wgpu::BindGroupLayout,
    pub raster_g1_layout: wgpu::BindGroupLayout,
    pub raster_pipeline_layout: wgpu::PipelineLayout,
    pub compute_g0_layout: wgpu::BindGroupLayout,
    pub compute_pipeline_layout: wgpu::PipelineLayout,

    pub stub_raster: wgpu::RenderPipeline,
    pub stub_spawn_count: wgpu::ComputePipeline,
    pub stub_prefix_local: wgpu::ComputePipeline,
    pub stub_prefix_scan_sums: wgpu::ComputePipeline,
    pub stub_prefix_add_back: wgpu::ComputePipeline,
    pub stub_fill: wgpu::ComputePipeline,
}

/// Per-shader pipeline tuple. `rkp-engine` builds one of these per
/// active user-shader material when it sees a new `(shader_id,
/// source_hash)` and caches the result.
pub struct UserShaderMeshPipelines {
    pub raster: wgpu::RenderPipeline,
    pub spawn_count: wgpu::ComputePipeline,
    pub prefix_local: wgpu::ComputePipeline,
    pub prefix_scan_sums: wgpu::ComputePipeline,
    pub prefix_add_back: wgpu::ComputePipeline,
    pub fill: wgpu::ComputePipeline,
}

/// One per-material draw descriptor enqueued by the engine for the
/// renderer's per-VR encode phase. wgpu types are internally
/// reference-counted, so `clone()` here is cheap. The renderer's
/// `dispatch_user_shader_mesh` consumes a slice of these.
#[derive(Clone)]
pub struct UserShaderMeshDraw {
    pub material_id: u16,
    pub shader_id: u32,
    pub vertex_count_per_spawn: u32,
    pub raster_pipeline: wgpu::RenderPipeline,
    pub raster_g1: wgpu::BindGroup,
    pub indirect_buffer: wgpu::Buffer,
}

impl UserShaderMeshPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let raster_g0_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("user_shader_mesh raster g0 (camera)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        );

        let raster_g1_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("user_shader_mesh raster g1 (anchors, records, frame, params)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );

        let raster_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("user_shader_mesh raster pipeline layout"),
                bind_group_layouts: &[Some(&raster_g0_layout), Some(&raster_g1_layout)],
                immediate_size: 0,
            },
        );

        let compute_g0_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("user_shader_mesh compute g0"),
                entries: &[
                    // 0: anchors (RO)
                    storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                    // 1: out_counts (RW)
                    storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                    // 2: out_offsets (RW)
                    storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                    // 3: out_records (RW)
                    storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                    // 4: out_indirect (RW)
                    storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
                    // 5: frame (uniform)
                    uniform_entry(5, wgpu::ShaderStages::COMPUTE),
                    // 6: params (uniform)
                    uniform_entry(6, wgpu::ShaderStages::COMPUTE),
                    // 7: dispatch (uniform)
                    uniform_entry(7, wgpu::ShaderStages::COMPUTE),
                    // 8: wg_sums (storage, RW) — Blelloch scan
                    //    per-workgroup-sum scratch.
                    storage_entry(8, false, wgpu::ShaderStages::COMPUTE),
                ],
            },
        );

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("user_shader_mesh compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_g0_layout)],
                immediate_size: 0,
            },
        );

        let raster_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("user_shader_mesh"),
            "user_shader_mesh",
        );

        let compute_module = crate::compile_pass_shader(
            device,
            wesl::include_wesl!("user_shader_mesh_compute"),
            "user_shader_mesh_compute",
        );

        let stub_raster = build_raster_pipeline(
            device,
            &raster_pipeline_layout,
            &raster_module,
            "user_shader_mesh stub raster",
        );

        let stub = build_compute_pipelines(
            device,
            &compute_pipeline_layout,
            &compute_module,
            "user_shader_mesh stub",
        );

        Self {
            raster_g0_layout,
            raster_g1_layout,
            raster_pipeline_layout,
            compute_g0_layout,
            compute_pipeline_layout,
            stub_raster,
            stub_spawn_count: stub.spawn_count,
            stub_prefix_local: stub.prefix_local,
            stub_prefix_scan_sums: stub.prefix_scan_sums,
            stub_prefix_add_back: stub.prefix_add_back,
            stub_fill: stub.fill,
        }
    }

    /// The build-emitted skeleton WGSL templates the composer
    /// splices user code into. `wesl::include_wesl!` reads from the
    /// emitter crate's OUT_DIR, so cross-crate callers (rkp-engine)
    /// can't invoke the macro themselves — they call this helper.
    pub fn template_sources() -> (&'static str, &'static str) {
        (
            wesl::include_wesl!("user_shader_mesh"),
            wesl::include_wesl!("user_shader_mesh_compute"),
        )
    }

    /// Build a per-shader pipeline tuple from the composed WGSL
    /// sources. The orchestration layer calls this when a user
    /// shader's source hash changes.
    pub fn build_pipelines(
        &self,
        device: &wgpu::Device,
        raster_wgsl: &str,
        compute_wgsl: &str,
        label: &str,
    ) -> UserShaderMeshPipelines {
        let raster_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label} raster module")),
            source: wgpu::ShaderSource::Wgsl(raster_wgsl.into()),
        });
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label} compute module")),
            source: wgpu::ShaderSource::Wgsl(compute_wgsl.into()),
        });
        let raster = build_raster_pipeline(
            device,
            &self.raster_pipeline_layout,
            &raster_module,
            &format!("{label} raster"),
        );
        let computes = build_compute_pipelines(
            device,
            &self.compute_pipeline_layout,
            &compute_module,
            label,
        );
        UserShaderMeshPipelines {
            raster,
            spawn_count: computes.spawn_count,
            prefix_local: computes.prefix_local,
            prefix_scan_sums: computes.prefix_scan_sums,
            prefix_add_back: computes.prefix_add_back,
            fill: computes.fill,
        }
    }

    /// Begin a render pass with the standard 5-color-attachment +
    /// depth setup. Same shape as `MeshProxyPass::begin_pass` — load
    /// + store onto the shared G-buffer, depth-test against the
    /// shared depth attachment.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_raster_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        position_view: &wgpu::TextureView,
        pick_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        glass_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
    ) -> wgpu::RenderPass<'a> {
        let load_op_color = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("user_shader_mesh raster"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: position_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: pick_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: normal_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: material_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: glass_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load_op_color,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct ComputePipelineSet {
    spawn_count: wgpu::ComputePipeline,
    prefix_local: wgpu::ComputePipeline,
    prefix_scan_sums: wgpu::ComputePipeline,
    prefix_add_back: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
}

fn build_compute_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    label: &str,
) -> ComputePipelineSet {
    let build = |entry: &str, suffix: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label} {suffix}")),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    ComputePipelineSet {
        spawn_count:      build("entry_spawn_count",     "spawn_count"),
        prefix_local:     build("entry_prefix_local",    "prefix_local"),
        prefix_scan_sums: build("entry_prefix_scan_sums","prefix_scan_sums"),
        prefix_add_back:  build("entry_prefix_add_back", "prefix_add_back"),
        fill:             build("entry_fill",            "fill"),
    }
}

fn build_raster_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("entry_vert"),
            compilation_options: Default::default(),
            // No vertex buffer — procedural geometry, VS reads
            // vertex_index + instance_index.
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // No back-face cull — user shaders may emit double-sided
            // geometry (grass blades viewed from below, leaves).
            // Re-enable per-shader via a manifest directive in V2.
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: GBUFFER_DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("entry_frag"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_POSITION_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_PICK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_NORMAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_MATERIAL_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: GBUFFER_GLASS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_record_layout() {
        assert_eq!(std::mem::size_of::<AnchorRecord>(), 64);
        assert_eq!(std::mem::align_of::<AnchorRecord>(), 4);
    }

    #[test]
    fn seed_is_deterministic() {
        let p = [1.5_f32, -2.25, 3.125];
        assert_eq!(anchor_seed(p), anchor_seed(p));
    }

    #[test]
    fn seed_distinguishes_nearby_positions() {
        let a = anchor_seed([0.0, 0.0, 0.0]);
        let b = anchor_seed([0.04, 0.0, 0.0]);
        let c = anchor_seed([0.0, 0.04, 0.0]);
        let d = anchor_seed([0.0, 0.0, 0.04]);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(b, c);
        assert_ne!(b, d);
        assert_ne!(c, d);
    }

    #[test]
    fn pod_sizes() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 48);
        assert_eq!(std::mem::size_of::<UserShaderParams>(), 32);
        assert_eq!(std::mem::size_of::<DispatchInfo>(), 16);
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
        assert_eq!(std::mem::size_of::<InstanceRecord>(), 8);
    }
}
