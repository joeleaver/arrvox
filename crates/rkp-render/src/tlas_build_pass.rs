//! Phase 7c — GPU-built TLAS pipeline.
//!
//! Replaces `tlas_pass.rs::TlasPass::build_tlas`'s CPU median-split
//! BVH builder with a fully GPU-resident pipeline:
//!
//! 1. **Session 1 (this file)** — primitive assembly. Two compute
//!    dispatches (`assemble_user_shader_main`, `assemble_host_main`)
//!    walk the per-frame inputs (tile-cull scratch + host
//!    instances) and pack tight world-space AABBs + leaf payloads
//!    into `tlas_prims`.
//! 2. Session 2 — Morton codes + GPU radix sort. (TODO)
//! 3. Session 3 — Karras radix tree. (TODO)
//! 4. Session 4 — bottom-up AABB propagation. (TODO)
//! 5. Session 5 — wire-up + cutover, replacing the CPU
//!    `tlas_pass::build_tlas` call in `render_worker`.
//!
//! ## Why GPU
//!
//! The CPU TLAS used `pos ± region_thickness` per-leaf AABBs for
//! user-shader instances because it had no way to evaluate the
//! shader's `inst_aabb` hook. With grass-style shaders that's a
//! 3 m cube around each painted leaf — 5000 leaves' AABBs all
//! overlap, BVH traversal degenerates to ~linear, shadow trace
//! catastrophically slow (30-40 ms for one .5 m grass splat).
//! Phase 6's tile-cull AABB pass already evaluates `inst_aabb` on
//! the GPU and writes tight per-instance world AABBs into scratch;
//! the GPU-built TLAS reads that scratch directly. Tight per-leaf
//! AABBs → real BVH culling → shadow trace stays fast.

use crate::rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance};
use crate::user_shader_tile_cull_pass::InstanceTileCullEntry;

/// One primitive in the unified TLAS-build input list. Same wire
/// shape as [`InstanceTileCullEntry`] minus the `live` flag (the
/// assembly pass filters those out) plus an `instance_index` field
/// that distinguishes host (real `RkpGpuInstance` index) from
/// user-shader ([`TLAS_LEAF_USER_SHADER`]) leaves.
///
/// `aabb_min` / `aabb_max` are tight world-space bounds:
/// * **Host** — `world × asset.local_aabb` via Arvo's transform.
/// * **User-shader** — the user shader's `inst_aabb` hook,
///   evaluated by the Phase 6 tile-cull AABB pass.
///
/// 48 bytes; vec3<f32> alignment in WGSL packs the trailing u32
/// fields into the same 16-byte slot as each vec3, so no extra
/// padding is needed beyond the explicit `_pad0` / `_pad1` that
/// keep the struct on a 16-byte boundary for the storage-buffer
/// stride invariant.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TlasPrim {
    pub aabb_min: [f32; 3],
    pub asset_id: u32,
    pub aabb_max: [f32; 3],
    pub instance_state_offset: u32,
    pub material_id: u32,
    pub instance_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<TlasPrim>() == 48);

/// Sentinel `instance_index` value for user-shader primitives.
/// Mirror of [`crate::tlas_pass::TLAS_LEAF_USER_SHADER`]. Kept here
/// duplicated so `tlas_build_pass.rs` doesn't need to depend on
/// `tlas_pass.rs` in V1; once Session 5 retires the CPU path we
/// can hoist the constant to a shared module.
pub const TLAS_LEAF_USER_SHADER: u32 = 0xFFFF_FFFEu32;

/// Per-dispatch uniform for the user-shader assembly pass. 16 B —
/// matches `AssembleUserShaderUniform` in
/// `tlas_assemble_user_shader.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AssembleUserShaderUniform {
    pub scratch_count: u32,
    pub prims_capacity: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<AssembleUserShaderUniform>() == 16);

/// Per-dispatch uniform for the host-instance assembly pass. 16 B —
/// matches `AssembleHostUniform` in `tlas_assemble_host.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AssembleHostUniform {
    pub instance_count: u32,
    pub asset_count: u32,
    pub prims_capacity: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<AssembleHostUniform>() == 16);

/// Per-dispatch uniform for the Morton-code compute pass. 32 B —
/// matches `MortonUniform` in `tlas_morton.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MortonUniform {
    pub scene_min: [f32; 3],
    pub _pad0: u32,
    pub scene_max: [f32; 3],
    pub prim_count: u32,
}

const _: () = assert!(std::mem::size_of::<MortonUniform>() == 32);

/// Per-dispatch uniform for one radix-sort sub-pass. 16 B — matches
/// `RadixUniform` in `tlas_radix_sort.wgsl`. The engine bumps
/// `digit_shift` by 8 between the four passes (0, 8, 16, 24).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixUniform {
    pub prim_count: u32,
    pub digit_shift: u32,
    pub num_workgroups: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<RadixUniform>() == 16);

/// Workgroup size of the radix count + scatter passes. 64 threads
/// per workgroup; `num_workgroups = prim_count.div_ceil(64)`.
pub const RADIX_WG_SIZE: u32 = 64;

/// Number of radix buckets (= 1 << bits-per-digit). 8-bit digits → 256.
pub const RADIX_BUCKETS: u32 = 256;

/// Number of radix passes. 32-bit Morton ÷ 8-bit digit = 4 passes.
pub const RADIX_PASSES: u32 = 4;

/// Per-dispatch uniform for the Karras tree builder. 16 B —
/// matches `KarrasUniform` in `tlas_karras.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KarrasUniform {
    pub prim_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<KarrasUniform>() == 16);

/// Initial `tlas_prims` buffer capacity in entries. Grows on demand
/// via [`TlasBuildPass::ensure_prims_capacity`]. Sized for one
/// entry so the buffer exists for bind-group validation before the
/// first dispatch.
pub const TLAS_PRIMS_INITIAL_ENTRIES: u32 = 1;

/// Pipeline holder for Phase 7c GPU TLAS build. Session 1 owns the
/// primitive-assembly pipelines and the shared output
/// (`tlas_prims_buffer` + `tlas_prim_count_buffer`); Session 2
/// adds the Morton + radix-sort pipelines and ping-pong key/value
/// buffers; later sessions extend with the Karras tree builder
/// and AABB propagation.
pub struct TlasBuildPass {
    // ── Session 1 ──────────────────────────────────────────────────
    pub user_shader_pipeline: wgpu::ComputePipeline,
    pub user_shader_g0_layout: wgpu::BindGroupLayout,
    pub user_shader_g1_layout: wgpu::BindGroupLayout,
    pub host_pipeline: wgpu::ComputePipeline,
    pub host_g0_layout: wgpu::BindGroupLayout,
    pub host_g1_layout: wgpu::BindGroupLayout,
    /// Packed `array<TlasPrim>`. Capacity grows; `tlas_prim_count`
    /// holds the per-frame live count after both assembly
    /// dispatches finish.
    pub tlas_prims_buffer: wgpu::Buffer,
    pub tlas_prims_capacity: u32,
    /// Single-element `array<atomic<u32>>` — the assembly passes
    /// `atomicAdd` into slot 0. Engine zeroes per frame before
    /// dispatch.
    pub tlas_prim_count_buffer: wgpu::Buffer,
    /// Per-dispatch uniforms. One slot each — re-uploaded per frame.
    pub user_shader_uniform_buffer: wgpu::Buffer,
    pub host_uniform_buffer: wgpu::Buffer,

    // ── Session 2 — Morton + radix sort ───────────────────────────
    pub morton_pipeline: wgpu::ComputePipeline,
    pub morton_g0_layout: wgpu::BindGroupLayout,
    pub morton_g1_layout: wgpu::BindGroupLayout,
    pub morton_uniform_buffer: wgpu::Buffer,

    pub radix_count_pipeline: wgpu::ComputePipeline,
    pub radix_scan_pipeline: wgpu::ComputePipeline,
    pub radix_scatter_pipeline: wgpu::ComputePipeline,
    pub radix_g0_layout: wgpu::BindGroupLayout,
    pub radix_g1_layout: wgpu::BindGroupLayout,
    /// One uniform per radix pass (4 of them, one per digit shift).
    /// Engine writes all four contiguously per frame; bind group
    /// uses dynamic offset to select the right one. 16 B each;
    /// 256 B per slot for wgpu's dynamic-offset alignment.
    pub radix_uniform_buffer: wgpu::Buffer,

    /// Ping-pong key buffers. After Morton compute, `keys_a` holds
    /// the input; pass 0 writes to `keys_b`; pass 1 to `keys_a`;
    /// passes 2 and 3 alternate. With 4 passes (even count), the
    /// final sorted output lands in `keys_a`. Same for `vals_a`/
    /// `vals_b`.
    pub keys_a_buffer: wgpu::Buffer,
    pub keys_b_buffer: wgpu::Buffer,
    pub vals_a_buffer: wgpu::Buffer,
    pub vals_b_buffer: wgpu::Buffer,
    /// Capacity of each ping-pong buffer in u32 entries.
    pub keys_capacity: u32,

    /// Per-WG histogram. Size `histogram_capacity_workgroups × 256`
    /// u32s (atomic). Re-zeroed between radix passes.
    pub histogram_buffer: wgpu::Buffer,
    /// Per-WG starting offsets after the scan pass. Same shape as
    /// histogram. Mutated in-place by scatter atomics.
    pub scan_offsets_buffer: wgpu::Buffer,
    /// Capacity of histogram + scan_offsets, in workgroup slots.
    pub histogram_capacity_workgroups: u32,

    // ── Session 3 — Karras radix tree ─────────────────────────────
    pub karras_leaves_pipeline: wgpu::ComputePipeline,
    pub karras_internal_pipeline: wgpu::ComputePipeline,
    pub karras_g0_layout: wgpu::BindGroupLayout,
    pub karras_g1_layout: wgpu::BindGroupLayout,
    pub karras_uniform_buffer: wgpu::Buffer,

    // ── Session 4 — bottom-up AABB propagation ───────────────────
    /// Phase 7c.6 — atomic AABB-accumulator propagation. Three
    /// pipelines on the same module + bind layout: init clears
    /// accumulators to ±∞ sentinels, propagate walks each leaf
    /// up the parent chain applying atomicMin/Max, decode reads
    /// the final accumulator values and writes `tlas_nodes` AABBs.
    pub init_atomic_aabb_pipeline: wgpu::ComputePipeline,
    pub propagate_atomic_pipeline: wgpu::ComputePipeline,
    pub decode_aabb_pipeline: wgpu::ComputePipeline,
    /// `array<u32>` of length `2N-1`. Engine fills with
    /// `0xFFFFFFFF` (`PARENT_SENTINEL`) before each frame's tree
    /// build; `build_internal_main` overwrites with parent indices
    /// for non-root nodes.
    pub parents_buffer: wgpu::Buffer,
    /// `array<atomic<u32>>` of length `3 × (2N-1)`. One u32 slot
    /// per (node, axis) for the min accumulator. Init pass fills
    /// with the encoded +∞ sentinel; each leaf-walk-up applies
    /// `atomicMin`; decode pass reads + writes back into
    /// `tlas_nodes[i].aabb_min`.
    pub aabb_min_atomic_buffer: wgpu::Buffer,
    /// Same shape as `aabb_min_atomic_buffer` but for max.
    pub aabb_max_atomic_buffer: wgpu::Buffer,
    /// Capacity of `parents_buffer` in u32 entries.
    pub parents_capacity: u32,
    /// Capacity of each atomic-AABB buffer in u32 entries
    /// (= 3 × (2N-1) target).
    pub aabb_atomic_capacity: u32,
    /// 4-byte staging buffer used to read `tlas_prim_count[0]`
    /// back to CPU after the assembly stage. The Phase 7c V1
    /// uses a synchronous readback to drive the rest of the
    /// pipeline; this buffer is the readback target.
    pub count_staging_buffer: wgpu::Buffer,
}

/// Inputs to [`TlasBuildPass::build_gpu_tlas`]. All buffers are
/// borrowed from external owners; the build pass binds them as
/// pipeline inputs but doesn't take ownership.
pub struct GpuTlasBuildInputs<'a> {
    /// `instance_tile_cull_scratch_buffer` from the engine.
    /// Holds one `InstanceTileCullEntry` per reserved instance
    /// slot across all user-shader regions.
    pub scratch_buffer: &'a wgpu::Buffer,
    /// Total entries in `scratch_buffer` to walk. = sum of
    /// `instance_block_size` across all regions this frame.
    pub scratch_count: u32,
    /// `state.renderer.scene.objects_buffer` — `array<RkpGpuInstance>`.
    pub instances_buffer: &'a wgpu::Buffer,
    pub instance_count: u32,
    /// `state.renderer.scene.assets_buffer` — `array<RkpGpuAsset>`.
    pub assets_buffer: &'a wgpu::Buffer,
    pub asset_count: u32,
    /// CPU-derived scene AABB for Morton normalization.
    /// Conservative is fine — Morton sort just needs a stable
    /// coordinate system.
    pub scene_min: [f32; 3],
    pub scene_max: [f32; 3],
}

impl TlasBuildPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let user_shader_g0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_user_shader g0"),
                entries: &[
                    ro_storage(0), // tile_cull_scratch
                    rw_storage(1), // tlas_prims
                    rw_storage(2), // tlas_prim_count
                ],
            });
        let user_shader_g1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_user_shader g1"),
                entries: &[uniform_entry(0, std::mem::size_of::<AssembleUserShaderUniform>() as u64)],
            });
        let user_shader_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_assemble_user_shader pipeline layout"),
            bind_group_layouts: &[Some(&user_shader_g0_layout), Some(&user_shader_g1_layout)],
            immediate_size: 0,
        });
        let user_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_assemble_user_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/tlas_assemble_user_shader.wgsl").into(),
            ),
        });
        let user_shader_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_assemble_user_shader"),
            layout: Some(&user_shader_layout),
            module: &user_shader_module,
            entry_point: Some("assemble_user_shader_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let host_g0_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_host g0"),
                entries: &[
                    ro_storage(0), // host_instances
                    ro_storage(1), // host_assets
                    rw_storage(2), // tlas_prims
                    rw_storage(3), // tlas_prim_count
                ],
            });
        let host_g1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tlas_assemble_host g1"),
                entries: &[uniform_entry(0, std::mem::size_of::<AssembleHostUniform>() as u64)],
            });
        let host_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_assemble_host pipeline layout"),
            bind_group_layouts: &[Some(&host_g0_layout), Some(&host_g1_layout)],
            immediate_size: 0,
        });
        let host_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_assemble_host"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/tlas_assemble_host.wgsl").into(),
            ),
        });
        let host_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_assemble_host"),
            layout: Some(&host_layout),
            module: &host_module,
            entry_point: Some("assemble_host_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let prims_bytes = (TLAS_PRIMS_INITIAL_ENTRIES as u64)
            * (std::mem::size_of::<TlasPrim>() as u64);
        let tlas_prims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prims"),
            size: prims_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tlas_prim_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prim_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let user_shader_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_assemble_user_shader uniform"),
            size: std::mem::size_of::<AssembleUserShaderUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let host_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_assemble_host uniform"),
            size: std::mem::size_of::<AssembleHostUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Session 2 — Morton + radix sort ───────────────────────
        let morton_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_morton g0"),
            entries: &[
                ro_storage(0), // tlas_prims
                rw_storage(1), // keys_a
                rw_storage(2), // vals_a
            ],
        });
        let morton_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_morton g1"),
            entries: &[uniform_entry(0, std::mem::size_of::<MortonUniform>() as u64)],
        });
        let morton_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_morton pipeline layout"),
            bind_group_layouts: &[Some(&morton_g0_layout), Some(&morton_g1_layout)],
            immediate_size: 0,
        });
        let morton_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_morton"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tlas_morton.wgsl").into()),
        });
        let morton_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_morton"),
            layout: Some(&morton_layout),
            module: &morton_module,
            entry_point: Some("compute_morton_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let morton_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_morton uniform"),
            size: std::mem::size_of::<MortonUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radix_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_radix g0"),
            entries: &[
                ro_storage(0), // keys_in
                ro_storage(1), // vals_in
                rw_storage(2), // keys_out
                rw_storage(3), // vals_out
                rw_storage(4), // histogram
                rw_storage(5), // scan_offsets
            ],
        });
        let radix_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_radix g1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                },
                count: None,
            }],
        });
        let radix_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_radix pipeline layout"),
            bind_group_layouts: &[Some(&radix_g0_layout), Some(&radix_g1_layout)],
            immediate_size: 0,
        });
        let radix_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_radix_sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tlas_radix_sort.wgsl").into()),
        });
        let radix_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_count"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("count_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let radix_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_scan"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("scan_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let radix_scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_radix_scatter"),
            layout: Some(&radix_pipeline_layout),
            module: &radix_module,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Four uniforms, one per radix pass, laid out 256 B apart
        // for wgpu's dynamic-offset alignment.
        let radix_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_radix uniform"),
            size: 256 * RADIX_PASSES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_keys_capacity: u32 = 1;
        let keys_buffer_descriptor = |label: &'static str, cap: u32| wgpu::BufferDescriptor {
            label: Some(label),
            size: (cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        let keys_a_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_keys_a", initial_keys_capacity));
        let keys_b_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_keys_b", initial_keys_capacity));
        let vals_a_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_vals_a", initial_keys_capacity));
        let vals_b_buffer = device.create_buffer(&keys_buffer_descriptor("tlas_vals_b", initial_keys_capacity));

        let initial_histogram_workgroups: u32 = 1;
        let histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_histogram"),
            size: (initial_histogram_workgroups as u64) * (RADIX_BUCKETS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scan_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_scan_offsets"),
            size: (initial_histogram_workgroups as u64) * (RADIX_BUCKETS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Session 3-4 — Karras tree + AABB propagation ──────────
        let karras_g0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_karras g0"),
            entries: &[
                ro_storage(0), // sorted_keys
                ro_storage(1), // sorted_vals
                ro_storage(2), // tlas_prims
                rw_storage(3), // tlas_nodes
                rw_storage(4), // tlas_leaves
                rw_storage(5), // parents
                rw_storage(6), // aabb_min_atomic
                rw_storage(7), // aabb_max_atomic
            ],
        });
        let karras_g1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tlas_karras g1"),
            entries: &[uniform_entry(0, std::mem::size_of::<KarrasUniform>() as u64)],
        });
        let karras_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tlas_karras pipeline layout"),
            bind_group_layouts: &[Some(&karras_g0_layout), Some(&karras_g1_layout)],
            immediate_size: 0,
        });
        let karras_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tlas_karras"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tlas_karras.wgsl").into()),
        });
        let karras_leaves_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_karras_leaves"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("build_leaves_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let karras_internal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_karras_internal"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("build_internal_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let karras_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_karras uniform"),
            size: std::mem::size_of::<KarrasUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Session 4 — AABB propagation ──────────────────────────
        let init_atomic_aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_init_atomic_aabb"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("init_atomic_aabb_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let propagate_atomic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_propagate_atomic"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("propagate_atomic_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let decode_aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tlas_decode_aabb"),
            layout: Some(&karras_pipeline_layout),
            module: &karras_module,
            entry_point: Some("decode_aabb_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let initial_parents_capacity: u32 = 1;
        let initial_aabb_atomic_capacity: u32 = 3;
        let parents_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_parents"),
            size: (initial_parents_capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let aabb_atomic_descriptor = |label: &'static str, cap: u32| wgpu::BufferDescriptor {
            label: Some(label),
            size: (cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        let aabb_min_atomic_buffer = device.create_buffer(&aabb_atomic_descriptor(
            "tlas_aabb_min_atomic",
            initial_aabb_atomic_capacity,
        ));
        let aabb_max_atomic_buffer = device.create_buffer(&aabb_atomic_descriptor(
            "tlas_aabb_max_atomic",
            initial_aabb_atomic_capacity,
        ));
        let count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prim_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            user_shader_pipeline,
            user_shader_g0_layout,
            user_shader_g1_layout,
            host_pipeline,
            host_g0_layout,
            host_g1_layout,
            tlas_prims_buffer,
            tlas_prims_capacity: TLAS_PRIMS_INITIAL_ENTRIES,
            tlas_prim_count_buffer,
            user_shader_uniform_buffer,
            host_uniform_buffer,
            morton_pipeline,
            morton_g0_layout,
            morton_g1_layout,
            morton_uniform_buffer,
            radix_count_pipeline,
            radix_scan_pipeline,
            radix_scatter_pipeline,
            radix_g0_layout,
            radix_g1_layout,
            radix_uniform_buffer,
            keys_a_buffer,
            keys_b_buffer,
            vals_a_buffer,
            vals_b_buffer,
            keys_capacity: initial_keys_capacity,
            histogram_buffer,
            scan_offsets_buffer,
            histogram_capacity_workgroups: initial_histogram_workgroups,
            karras_leaves_pipeline,
            karras_internal_pipeline,
            karras_g0_layout,
            karras_g1_layout,
            karras_uniform_buffer,
            init_atomic_aabb_pipeline,
            propagate_atomic_pipeline,
            decode_aabb_pipeline,
            parents_buffer,
            aabb_min_atomic_buffer,
            aabb_max_atomic_buffer,
            parents_capacity: initial_parents_capacity,
            aabb_atomic_capacity: initial_aabb_atomic_capacity,
            count_staging_buffer,
        }
    }

    /// Grow `parents_buffer` to fit `entries` u32s. Allocation
    /// target = `2N-1` (one parent slot per node, including leaves).
    pub fn ensure_parents_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.parents_capacity {
            return false;
        }
        let mut new_cap = self.parents_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        self.parents_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_parents"),
            size: (new_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.parents_capacity = new_cap;
        true
    }

    /// Drive the full GPU TLAS build (Sessions 1-4) end to end.
    /// Encodes assembly → readback → Morton → 4× radix → Karras
    /// leaves + internal → AABB propagation, writing the final
    /// `tlas_nodes` + `tlas_leaves` into the supplied
    /// [`crate::tlas_pass::TlasPass`] buffers (which the shadow
    /// trace already binds).
    ///
    /// Returns the actual primitive count after assembly (= number
    /// of leaves in the built TLAS). Caller stamps this into
    /// `tlas_pass.last_node_count = 2N-1` and
    /// `tlas_pass.last_leaf_count = N` so the shadow trace's empty-
    /// scene skip works (the WGSL early-outs when `tlas_node_count
    /// == 0`).
    ///
    /// V1 uses a synchronous readback between assembly and the
    /// downstream chain — `device.poll(wait_indefinitely)` blocks
    /// the calling thread for ~1 ms per frame. Acceptable for V1
    /// (we're trading 1 ms here to save 30+ ms of shadow trace);
    /// future refactor to indirect dispatch would remove the stall.
    pub fn build_gpu_tlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &GpuTlasBuildInputs,
        tlas_pass: &mut crate::tlas_pass::TlasPass,
    ) -> u32 {
        let upper_bound = inputs.scratch_count.saturating_add(inputs.instance_count);
        if upper_bound == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the assembly stage.
        self.ensure_prims_capacity(device, upper_bound);

        // Assemble + count readback. Single submit + map_async +
        // device.poll. The blocking poll is the 1 ms stall.
        let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build assemble"),
        });
        enc1.clear_buffer(&self.tlas_prim_count_buffer, 0, Some(4));

        // Upload assembly uniforms.
        queue.write_buffer(
            &self.user_shader_uniform_buffer,
            0,
            bytemuck::bytes_of(&AssembleUserShaderUniform {
                scratch_count: inputs.scratch_count,
                prims_capacity: self.tlas_prims_capacity,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &self.host_uniform_buffer,
            0,
            bytemuck::bytes_of(&AssembleHostUniform {
                instance_count: inputs.instance_count,
                asset_count: inputs.asset_count,
                prims_capacity: self.tlas_prims_capacity,
                _pad: 0,
            }),
        );

        // User-shader assembly bind groups + dispatch.
        if inputs.scratch_count > 0 {
            let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_user_shader g0"),
                layout: &self.user_shader_g0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: inputs.scratch_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.tlas_prims_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prim_count_buffer.as_entire_binding() },
                ],
            });
            let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_user_shader g1"),
                layout: &self.user_shader_g1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.user_shader_uniform_buffer.as_entire_binding(),
                }],
            });
            let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("assemble_user_shader_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.user_shader_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            let wgs = ((inputs.scratch_count + 63) / 64).max(1);
            cpass.dispatch_workgroups(wgs, 1, 1);
        }

        // Host assembly bind groups + dispatch.
        if inputs.instance_count > 0 {
            let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g0"),
                layout: &self.host_g0_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: inputs.instances_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: inputs.assets_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.tlas_prim_count_buffer.as_entire_binding() },
                ],
            });
            let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tlas_assemble_host g1"),
                layout: &self.host_g1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.host_uniform_buffer.as_entire_binding(),
                }],
            });
            let mut cpass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("assemble_host_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.host_pipeline);
            cpass.set_bind_group(0, &g0, &[]);
            cpass.set_bind_group(1, &g1, &[]);
            let wgs = ((inputs.instance_count + 63) / 64).max(1);
            cpass.dispatch_workgroups(wgs, 1, 1);
        }

        // Copy count to staging for readback.
        enc1.copy_buffer_to_buffer(&self.tlas_prim_count_buffer, 0, &self.count_staging_buffer, 0, 4);
        queue.submit(std::iter::once(enc1.finish()));

        // Synchronous readback. Stalls the engine thread ~1 ms.
        let slice = self.count_staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll for tlas count readback");
        let raw_count = {
            let view = slice.get_mapped_range();
            let c = u32::from_le_bytes(view[0..4].try_into().unwrap());
            drop(view);
            self.count_staging_buffer.unmap();
            c
        };
        // Clamp the readback to actual capacity. If the assembly
        // atomic recorded more attempted writes than the buffer
        // holds (overflow), the writes themselves were gated by
        // `if (slot >= u.prims_capacity) return;`; the counter
        // just kept incrementing for telemetry.
        let actual_count = raw_count.min(self.tlas_prims_capacity);
        let upper_bound = inputs.scratch_count.saturating_add(inputs.instance_count);
        // Diagnostic logging — log every frame that involves
        // grass (scratch_count > 0). User observation report
        // can be correlated against these values.
        if inputs.scratch_count > 0 {
            eprintln!(
                "[tlas_build] raw={raw_count} actual={actual_count} scratch={} host={} upper={} caps prims={} keys={}",
                inputs.scratch_count, inputs.instance_count, upper_bound,
                self.tlas_prims_capacity, self.keys_capacity,
            );
        } else if raw_count > upper_bound || (raw_count == 0 && upper_bound > 0) {
            // Suspicious value even without grass.
            eprintln!(
                "[tlas_build] suspect raw={raw_count} upper={upper_bound} scratch={} host={}",
                inputs.scratch_count, inputs.instance_count,
            );
        }

        if actual_count == 0 {
            tlas_pass.last_node_count = 0;
            tlas_pass.last_leaf_count = 0;
            return 0;
        }

        // Capacities for the downstream chain.
        self.ensure_keys_capacity(device, actual_count);
        let radix_workgroups = ((actual_count + 63) / 64).max(1);
        self.ensure_histogram_capacity(device, radix_workgroups);
        self.ensure_parents_capacity(device, 2 * actual_count - 1);
        // Phase 7c.6 — atomic AABB accumulators sized 3 × (2N-1)
        // u32s each (one per axis per node). Init pass clears
        // them to ±∞ sentinels at frame start; `propagate_atomic_main`
        // walks each leaf up the parent chain applying atomicMin/Max.
        let total_nodes = (2 * actual_count).saturating_sub(1).max(1);
        let aabb_atomic_entries = total_nodes.saturating_mul(3).max(3);
        self.ensure_aabb_atomic_capacity(device, aabb_atomic_entries);
        tlas_pass.ensure_capacity(device, 2 * actual_count - 1, actual_count);

        // Upload uniforms for the chain.
        queue.write_buffer(
            &self.morton_uniform_buffer,
            0,
            bytemuck::bytes_of(&MortonUniform {
                scene_min: inputs.scene_min,
                _pad0: 0,
                scene_max: inputs.scene_max,
                prim_count: actual_count,
            }),
        );
        let radix_stride: u64 = 256;
        let mut radix_bytes: Vec<u8> = vec![0u8; (RADIX_PASSES as u64 * radix_stride) as usize];
        for p in 0..RADIX_PASSES {
            let u = RadixUniform {
                prim_count: actual_count,
                digit_shift: p * 8,
                num_workgroups: radix_workgroups,
                _pad: 0,
            };
            let off = (p as u64 * radix_stride) as usize;
            radix_bytes[off..off + std::mem::size_of::<RadixUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        queue.write_buffer(&self.radix_uniform_buffer, 0, &radix_bytes);
        queue.write_buffer(
            &self.karras_uniform_buffer,
            0,
            bytemuck::bytes_of(&KarrasUniform {
                prim_count: actual_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        // Init parents (sentinel) + visit_counter (zero).
        let parents_init: Vec<u32> = vec![0xFFFFFFFFu32; (2 * actual_count - 1) as usize];
        queue.write_buffer(&self.parents_buffer, 0, bytemuck::cast_slice(&parents_init));
        // Phase 7c.6 — `init_atomic_aabb_main` (in the dispatch
        // chain below) writes ±∞ sentinels to both atomic
        // buffers; no CPU pre-fill needed. Just declare we're
        // about to use them at the new size.
        let _ = total_nodes;

        // Encode the chain.
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tlas_build chain"),
        });

        // Morton compute.
        let morton_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g0"),
            layout: &self.morton_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.vals_a_buffer.as_entire_binding() },
            ],
        });
        let morton_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton g1"),
            layout: &self.morton_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.morton_uniform_buffer.as_entire_binding(),
            }],
        });
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_morton_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.morton_pipeline);
            cpass.set_bind_group(0, &morton_g0, &[]);
            cpass.set_bind_group(1, &morton_g1, &[]);
            cpass.dispatch_workgroups(radix_workgroups, 1, 1);
        }

        // Radix sort — 4 passes ping-ponging a→b→a→b.
        let radix_g0_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 a→b"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g0_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g0 b→a"),
            layout: &self.radix_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.histogram_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.scan_offsets_buffer.as_entire_binding() },
            ],
        });
        let radix_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix g1"),
            layout: &self.radix_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.radix_uniform_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<RadixUniform>() as u64),
                }),
            }],
        });
        let histogram_bytes = (radix_workgroups as u64) * (RADIX_BUCKETS as u64) * 4;
        for p in 0..RADIX_PASSES {
            let g0 = if p % 2 == 0 { &radix_g0_a_to_b } else { &radix_g0_b_to_a };
            let dyn_off = (p as u64 * radix_stride) as u32;
            enc2.clear_buffer(&self.histogram_buffer, 0, Some(histogram_bytes));
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix count_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_count_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scan_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scan_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("radix scatter_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.radix_scatter_pipeline);
                cpass.set_bind_group(0, g0, &[]);
                cpass.set_bind_group(1, &radix_g1, &[dyn_off]);
                cpass.dispatch_workgroups(radix_workgroups, 1, 1);
            }
        }

        // Karras tree + AABB propagation. Output goes into the
        // shadow-trace consumer buffers (`tlas_pass.{nodes,leaves}_buffer`).
        let karras_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g0"),
            layout: &self.karras_g0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.keys_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vals_a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tlas_prims_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: tlas_pass.nodes_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tlas_pass.leaves_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.parents_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.aabb_min_atomic_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.aabb_max_atomic_buffer.as_entire_binding() },
            ],
        });
        let karras_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras g1"),
            layout: &self.karras_g1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.karras_uniform_buffer.as_entire_binding(),
            }],
        });
        let leaf_wgs = ((actual_count + 63) / 64).max(1);
        {
            let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_leaves_main"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.karras_leaves_pipeline);
            cpass.set_bind_group(0, &karras_g0, &[]);
            cpass.set_bind_group(1, &karras_g1, &[]);
            cpass.dispatch_workgroups(leaf_wgs, 1, 1);
        }
        if actual_count >= 2 {
            let internal_wgs = (((actual_count - 1) + 63) / 64).max(1);
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("build_internal_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.karras_internal_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }
        // Phase 7c.6 — atomic AABB propagation. Three passes:
        //   1. init: clear accumulators to ±∞ sentinels.
        //   2. propagate: each leaf walks up to root, atomic-min/max
        //      into ancestors. Commutative — no thread ordering
        //      issues, no cross-buffer memory visibility needed.
        //   3. decode: read accumulators, write tlas_nodes AABBs.
        let total_node_wgs = (((2 * actual_count - 1) + 63) / 64).max(1);
        let internal_wgs = if actual_count >= 2 {
            (((actual_count - 1) + 63) / 64).max(1)
        } else {
            1
        };
        if actual_count >= 2 {
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("init_atomic_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.init_atomic_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(total_node_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("propagate_atomic_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.propagate_atomic_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(leaf_wgs, 1, 1);
            }
            {
                let mut cpass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("decode_aabb_main"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.decode_aabb_pipeline);
                cpass.set_bind_group(0, &karras_g0, &[]);
                cpass.set_bind_group(1, &karras_g1, &[]);
                cpass.dispatch_workgroups(internal_wgs, 1, 1);
            }
        }

        queue.submit(std::iter::once(enc2.finish()));

        tlas_pass.last_node_count = 2 * actual_count - 1;
        tlas_pass.last_leaf_count = actual_count;
        actual_count
    }

    /// Grow each `aabb_*_atomic_buffer` to fit `entries` u32s
    /// (= 3 × node count). Both buffers share the same capacity.
    pub fn ensure_aabb_atomic_capacity(
        &mut self,
        device: &wgpu::Device,
        entries: u32,
    ) -> bool {
        if entries <= self.aabb_atomic_capacity {
            return false;
        }
        let mut new_cap = self.aabb_atomic_capacity.max(3);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let descriptor = |label: &'static str| wgpu::BufferDescriptor {
            label: Some(label),
            size: (new_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };
        self.aabb_min_atomic_buffer = device.create_buffer(&descriptor("tlas_aabb_min_atomic"));
        self.aabb_max_atomic_buffer = device.create_buffer(&descriptor("tlas_aabb_max_atomic"));
        self.aabb_atomic_capacity = new_cap;
        true
    }

    /// Grow `keys_*` and `vals_*` ping-pong buffers to fit
    /// `entries` u32s each. Grows with capacity doubling.
    pub fn ensure_keys_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.keys_capacity {
            return false;
        }
        let mut new_cap = self.keys_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * 4;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let descriptor = |label: &'static str| wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        };
        self.keys_a_buffer = device.create_buffer(&descriptor("tlas_keys_a"));
        self.keys_b_buffer = device.create_buffer(&descriptor("tlas_keys_b"));
        self.vals_a_buffer = device.create_buffer(&descriptor("tlas_vals_a"));
        self.vals_b_buffer = device.create_buffer(&descriptor("tlas_vals_b"));
        self.keys_capacity = new_cap;
        true
    }

    /// Grow `histogram_buffer` and `scan_offsets_buffer` to fit
    /// `workgroups` per-WG slices (each `RADIX_BUCKETS` u32 wide).
    pub fn ensure_histogram_capacity(&mut self, device: &wgpu::Device, workgroups: u32) -> bool {
        if workgroups <= self.histogram_capacity_workgroups {
            return false;
        }
        let mut new_cap = self.histogram_capacity_workgroups.max(1);
        while new_cap < workgroups {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * (RADIX_BUCKETS as u64) * 4;
        self.histogram_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_histogram"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.scan_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_scan_offsets"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.histogram_capacity_workgroups = new_cap;
        true
    }

    /// Grow `tlas_prims_buffer` to fit `entries` primitives. Grows
    /// with capacity doubling (mirrors the pattern in
    /// `OctreeMarchPass::ensure_us_tile_grid_capacity`). Returns
    /// `true` if the buffer reallocated — caller is responsible for
    /// invalidating any cached bind groups that referenced the old
    /// handle.
    pub fn ensure_prims_capacity(&mut self, device: &wgpu::Device, entries: u32) -> bool {
        if entries <= self.tlas_prims_capacity {
            return false;
        }
        let mut new_cap = self.tlas_prims_capacity.max(1);
        while new_cap < entries {
            new_cap = new_cap.saturating_mul(2);
        }
        let bytes = (new_cap as u64) * (std::mem::size_of::<TlasPrim>() as u64);
        self.tlas_prims_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas_prims"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.tlas_prims_capacity = new_cap;
        true
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn uniform_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(min_size),
        },
        count: None,
    }
}

/// CPU reference for the user-shader assembly path. Mirrors
/// `tlas_assemble_user_shader.wgsl::assemble_user_shader_main`
/// faithfully (same filter rules, same atomic-driven slot
/// assignment) so the integration test can assert the GPU output
/// matches a known-good CPU walk over the same inputs.
///
/// Returns `(prims_in_emit_order, count)`. Slot order is the order
/// threads "win" their atomic increment — for a single-workgroup
/// dispatch on most GPUs this is monotonic by `gid`, so the
/// reference walks scratch in index order. The test compares
/// outputs as **multisets** (sorted) since the GPU's atomic order
/// across multiple workgroups is implementation-defined.
pub fn cpu_reference_assemble_user_shader(
    scratch: &[InstanceTileCullEntry],
    prims_capacity: u32,
) -> (Vec<TlasPrim>, u32) {
    let mut out: Vec<TlasPrim> = Vec::new();
    let mut count: u32 = 0;
    for entry in scratch {
        if entry.live != 1 {
            continue;
        }
        let extent = [
            entry.aabb_max[0] - entry.aabb_min[0],
            entry.aabb_max[1] - entry.aabb_min[1],
            entry.aabb_max[2] - entry.aabb_min[2],
        ];
        if extent[0] <= 0.0 || extent[1] <= 0.0 || extent[2] <= 0.0 {
            continue;
        }
        let slot = count;
        count += 1;
        if slot >= prims_capacity {
            continue;
        }
        out.push(TlasPrim {
            aabb_min: entry.aabb_min,
            asset_id: entry.asset_id,
            aabb_max: entry.aabb_max,
            instance_state_offset: entry.instance_state_offset,
            material_id: entry.material_id,
            instance_index: TLAS_LEAF_USER_SHADER,
            _pad0: 0,
            _pad1: 0,
        });
    }
    (out, count)
}

/// CPU reference for the host-instance assembly path. Mirrors
/// `tlas_assemble_host.wgsl::assemble_host_main`. Same multiset
/// semantics as the user-shader reference.
pub fn cpu_reference_assemble_host(
    instances: &[RkpGpuInstance],
    assets: &[RkpGpuAsset],
    prims_capacity: u32,
) -> (Vec<TlasPrim>, u32) {
    let mut out: Vec<TlasPrim> = Vec::new();
    let mut count: u32 = 0;
    for (i, inst) in instances.iter().enumerate() {
        let asset_id = inst.asset_id as usize;
        if asset_id >= assets.len() {
            continue;
        }
        let asset = &assets[asset_id];
        if asset.shader_id != 0 {
            continue;
        }
        let (world_min, world_max) =
            transform_aabb(asset.aabb_min, asset.aabb_max, &inst.world);
        let slot = count;
        count += 1;
        if slot >= prims_capacity {
            continue;
        }
        out.push(TlasPrim {
            aabb_min: world_min,
            asset_id: inst.asset_id,
            aabb_max: world_max,
            instance_state_offset: 0,
            material_id: inst.material_id,
            instance_index: i as u32,
            _pad0: 0,
            _pad1: 0,
        });
    }
    (out, count)
}

/// Same Arvo's transform-AABB the CPU `tlas_pass.rs::transform_aabb`
/// uses; duplicated here so this module doesn't depend on
/// `tlas_pass`. Will be deleted along with the CPU `build_tlas` in
/// Session 5.
fn transform_aabb(
    local_min: [f32; 3],
    local_max: [f32; 3],
    world: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let mut new_min = [world[3][0], world[3][1], world[3][2]];
    let mut new_max = [world[3][0], world[3][1], world[3][2]];
    for i in 0..3 {
        for j in 0..3 {
            let a = world[j][i] * local_min[j];
            let b = world[j][i] * local_max[j];
            new_min[i] += a.min(b);
            new_max[i] += a.max(b);
        }
    }
    (new_min, new_max)
}

/// Compute the union AABB of a list of `TlasPrim`s. Used by the
/// engine to derive the [`MortonUniform`] scene bounds before the
/// Morton-code dispatch — Morton sort just needs a stable
/// coordinate system, so a CPU-side conservative bound is fine
/// (saves a GPU reduction pass; lifts cleanly to GPU later if N
/// grows past the point where CPU iteration matters).
///
/// Empty input returns a 1-unit cube at the origin so the
/// downstream `extent.max(1e-6)` clamp in the WGSL doesn't divide
/// by zero on the (no-op) Morton dispatch.
pub fn scene_aabb_from_prims(prims: &[TlasPrim]) -> ([f32; 3], [f32; 3]) {
    if prims.is_empty() {
        return ([0.0; 3], [1.0; 3]);
    }
    let mut min = prims[0].aabb_min;
    let mut max = prims[0].aabb_max;
    for p in &prims[1..] {
        for ax in 0..3 {
            if p.aabb_min[ax] < min[ax] {
                min[ax] = p.aabb_min[ax];
            }
            if p.aabb_max[ax] > max[ax] {
                max[ax] = p.aabb_max[ax];
            }
        }
    }
    (min, max)
}

fn expand_bits_10(v_in: u32) -> u32 {
    let mut v = v_in & 0x3FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

fn morton_30(x: u32, y: u32, z: u32) -> u32 {
    (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z)
}

/// CPU reference for `tlas_morton.wgsl::compute_morton_main`.
/// Returns the (Morton, prim_idx) pairs that the GPU dispatch
/// would produce given the same input.
pub fn cpu_reference_morton(
    prims: &[TlasPrim],
    scene_min: [f32; 3],
    scene_max: [f32; 3],
) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(prims.len());
    let extent = [
        (scene_max[0] - scene_min[0]).max(1e-6),
        (scene_max[1] - scene_min[1]).max(1e-6),
        (scene_max[2] - scene_min[2]).max(1e-6),
    ];
    for (i, p) in prims.iter().enumerate() {
        let centroid = [
            0.5 * (p.aabb_min[0] + p.aabb_max[0]),
            0.5 * (p.aabb_min[1] + p.aabb_max[1]),
            0.5 * (p.aabb_min[2] + p.aabb_max[2]),
        ];
        let normalized = [
            ((centroid[0] - scene_min[0]) / extent[0]).clamp(0.0, 1.0),
            ((centroid[1] - scene_min[1]) / extent[1]).clamp(0.0, 1.0),
            ((centroid[2] - scene_min[2]) / extent[2]).clamp(0.0, 1.0),
        ];
        let q = [
            (normalized[0] * 1023.0).min(1023.0) as u32,
            (normalized[1] * 1023.0).min(1023.0) as u32,
            (normalized[2] * 1023.0).min(1023.0) as u32,
        ];
        out.push((morton_30(q[0], q[1], q[2]), i as u32));
    }
    out
}

/// CPU reference for the GPU radix sort. Standard `Vec::sort_by`
/// over the (key, val) pairs — the GPU's stability guarantees are
/// per-bucket but not within ties, so the integration test
/// compares as a multiset (sort by key first, then by val) instead
/// of asserting bit-for-bit equality.
pub fn cpu_reference_radix_sort(pairs: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    sorted
}

/// Karras' `delta(i, j)` = length of common prefix of the virtual
/// keys `(morton[i] << 32) | i`. Mirrors the WGSL helper in
/// `tlas_karras.wgsl::delta`. Returns -1 for out-of-range j.
pub fn karras_delta(sorted_keys: &[u32], i: i32, j: i32) -> i32 {
    let n = sorted_keys.len() as i32;
    if j < 0 || j >= n {
        return -1;
    }
    let ki = sorted_keys[i as usize];
    let kj = sorted_keys[j as usize];
    if ki != kj {
        return (ki ^ kj).leading_zeros() as i32;
    }
    32 + ((i as u32) ^ (j as u32)).leading_zeros() as i32
}

/// CPU reference for the full Karras tree + AABB propagation. Given
/// sorted Mortons + sorted_vals + the underlying TlasPrim payloads,
/// returns the `tlas_nodes` array (length 2N-1) with both topology
/// AND AABBs filled in, exactly matching what the GPU pipeline
/// (S3 + S4) is expected to produce.
///
/// Returns `(nodes, leaves)`. `leaves[i]` is the TlasInstanceLeaf
/// payload of `prims[sorted_vals[i]]` (also matches the `tlas_leaves`
/// buffer the GPU produces).
pub fn cpu_reference_full_tree(
    sorted_keys: &[u32],
    sorted_vals: &[u32],
    prims: &[TlasPrim],
) -> (Vec<crate::tlas_pass::TlasNode>, Vec<crate::tlas_pass::TlasInstanceLeaf>) {
    use crate::tlas_pass::{TlasInstanceLeaf, TlasNode, TLAS_NODE_LEAF_BIT};
    let n = sorted_keys.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut nodes: Vec<TlasNode> = vec![
        TlasNode {
            aabb_min: [0.0; 3],
            left_or_leaf: 0,
            aabb_max: [0.0; 3],
            right_or_count: 0,
        };
        2 * n - 1
    ];
    let mut leaves: Vec<TlasInstanceLeaf> = Vec::with_capacity(n);

    // Leaves.
    for i in 0..n {
        let prim = &prims[sorted_vals[i] as usize];
        leaves.push(TlasInstanceLeaf {
            asset_id: prim.asset_id,
            instance_state_offset: prim.instance_state_offset,
            material_id: prim.material_id,
            instance_index: prim.instance_index,
        });
        let leaf_node = &mut nodes[n - 1 + i];
        leaf_node.aabb_min = prim.aabb_min;
        leaf_node.aabb_max = prim.aabb_max;
        leaf_node.left_or_leaf = TLAS_NODE_LEAF_BIT | (i as u32);
        leaf_node.right_or_count = 1;
    }

    // Internal nodes — topology first.
    for i in 0..n.saturating_sub(1) {
        let (l, r) = cpu_reference_karras_node(sorted_keys, i as i32);
        nodes[i].left_or_leaf = l;
        nodes[i].right_or_count = r;
    }

    // Internal AABBs — bottom-up via post-order traversal from root.
    if n >= 2 {
        fn fill_aabb(idx: u32, nodes: &mut [crate::tlas_pass::TlasNode]) {
            if (nodes[idx as usize].left_or_leaf & crate::tlas_pass::TLAS_NODE_LEAF_BIT) != 0 {
                return; // Leaf — AABB already set.
            }
            let l = nodes[idx as usize].left_or_leaf;
            let r = nodes[idx as usize].right_or_count;
            fill_aabb(l, nodes);
            fill_aabb(r, nodes);
            let lmin = nodes[l as usize].aabb_min;
            let lmax = nodes[l as usize].aabb_max;
            let rmin = nodes[r as usize].aabb_min;
            let rmax = nodes[r as usize].aabb_max;
            nodes[idx as usize].aabb_min = [
                lmin[0].min(rmin[0]),
                lmin[1].min(rmin[1]),
                lmin[2].min(rmin[2]),
            ];
            nodes[idx as usize].aabb_max = [
                lmax[0].max(rmax[0]),
                lmax[1].max(rmax[1]),
                lmax[2].max(rmax[2]),
            ];
        }
        fill_aabb(0, &mut nodes);
    }

    (nodes, leaves)
}

/// CPU reference for `tlas_karras.wgsl::build_internal_main`.
/// Returns the pair of children (left, right) for internal node
/// `idx`, in the convention the WGSL writes: `< prim_count - 1` is
/// another internal node, `≥ prim_count - 1` is a leaf-marker
/// node at index `prim_count - 1 + leaf_idx` in `tlas_nodes`.
pub fn cpu_reference_karras_node(sorted_keys: &[u32], idx: i32) -> (u32, u32) {
    let n = sorted_keys.len() as i32;
    debug_assert!(idx < n - 1, "internal node index {idx} out of range (n = {n})");

    // Direction.
    let d = (karras_delta(sorted_keys, idx, idx + 1) - karras_delta(sorted_keys, idx, idx - 1))
        .signum();
    let delta_min = karras_delta(sorted_keys, idx, idx - d);

    // Upper bound on range length.
    let mut l_max: i32 = 2;
    while karras_delta(sorted_keys, idx, idx + l_max * d) > delta_min {
        l_max *= 2;
        if l_max > n {
            break;
        }
    }

    // Binary search for length l.
    let mut l: i32 = 0;
    let mut t = l_max / 2;
    while t >= 1 {
        if karras_delta(sorted_keys, idx, idx + (l + t) * d) > delta_min {
            l += t;
        }
        t /= 2;
    }

    let j = idx + l * d;
    let delta_node = karras_delta(sorted_keys, idx, j);

    // Find split.
    let mut s: i32 = 0;
    let mut divisor: i32 = 2;
    loop {
        let t_split = (l + divisor - 1) / divisor;
        if karras_delta(sorted_keys, idx, idx + (s + t_split) * d) > delta_node {
            s += t_split;
        }
        if t_split <= 1 {
            break;
        }
        divisor *= 2;
    }
    let gamma = idx + s * d + d.min(0);

    let range_lo = idx.min(j);
    let range_hi = idx.max(j);

    let left_child = if range_lo == gamma {
        (n - 1 + gamma) as u32
    } else {
        gamma as u32
    };
    let right_child = if range_hi == gamma + 1 {
        (n - 1 + gamma + 1) as u32
    } else {
        (gamma + 1) as u32
    };
    (left_child, right_child)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(min: [f32; 3], max: [f32; 3], asset: u32, live: u32) -> InstanceTileCullEntry {
        InstanceTileCullEntry {
            aabb_min: min,
            asset_id: asset,
            aabb_max: max,
            instance_state_offset: 0,
            material_id: 0,
            live,
            _pad0: 0,
            _pad1: 0,
        }
    }

    #[test]
    fn tlas_prim_size_is_48() {
        assert_eq!(std::mem::size_of::<TlasPrim>(), 48);
    }

    #[test]
    fn user_shader_filters_dead_entries() {
        let scratch = vec![
            entry([0.0; 3], [1.0; 3], 1, 1),
            entry([2.0; 3], [3.0; 3], 2, 0), // dead — skipped
            entry([4.0; 3], [5.0; 3], 3, 1),
        ];
        let (out, count) = cpu_reference_assemble_user_shader(&scratch, 16);
        assert_eq!(count, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].asset_id, 1);
        assert_eq!(out[1].asset_id, 3);
        for p in &out {
            assert_eq!(p.instance_index, TLAS_LEAF_USER_SHADER);
        }
    }

    #[test]
    fn user_shader_filters_degenerate_aabb() {
        // live=1 but zero-volume → filtered.
        let scratch = vec![
            entry([0.0; 3], [0.0; 3], 1, 1),  // zero extent
            entry([0.0; 3], [1.0; 3], 2, 1),  // valid
        ];
        let (out, count) = cpu_reference_assemble_user_shader(&scratch, 16);
        assert_eq!(count, 1);
        assert_eq!(out[0].asset_id, 2);
    }

    #[test]
    fn user_shader_capacity_overflow_drops_excess() {
        let scratch = vec![
            entry([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1, 1),
            entry([2.0, 2.0, 2.0], [3.0, 3.0, 3.0], 2, 1),
            entry([4.0, 4.0, 4.0], [5.0, 5.0, 5.0], 3, 1),
        ];
        let (out, count) = cpu_reference_assemble_user_shader(&scratch, 2);
        // count reflects ALL writes attempted (including overflow);
        // out only carries those that fit.
        assert_eq!(count, 3);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].asset_id, 1);
        assert_eq!(out[1].asset_id, 2);
    }

    fn make_asset(min: [f32; 3], max: [f32; 3], shader_id: u32) -> RkpGpuAsset {
        RkpGpuAsset {
            aabb_min: min,
            octree_root: 0,
            aabb_max: max,
            octree_depth: 0,
            octree_extent_bits: 0,
            voxel_size: 0.0,
            geom_type: 0,
            bone_count: 0,
            grid_origin: [0.0; 3],
            rest_octree_root: 0,
            rest_octree_depth: 0,
            rest_octree_extent_bits: 0,
            shader_id,
            _pad: 0,
        }
    }

    fn make_instance(asset_id: u32, world: [[f32; 4]; 4], material: u32) -> RkpGpuInstance {
        RkpGpuInstance {
            world,
            asset_id,
            material_id: material,
            object_id: 0,
            layer_mask: 0xFFFF_FFFF,
            is_skinned: 0,
            bone_buffer_offset: 0,
            bone_field_offset: 0,
            bone_field_occ_offset: 0,
            bone_field_dim_x: 0,
            bone_field_dim_y: 0,
            bone_field_dim_z: 0,
            bone_field_origin_x: 0.0,
            bone_field_origin_y: 0.0,
            bone_field_origin_z: 0.0,
            overlay_offset: 0,
            overlay_count: 0,
            instance_state_offset: 0,
            _pad: [0, 0, 0],
        }
    }

    #[test]
    fn host_skips_user_shader_assets() {
        let assets = vec![
            make_asset([0.0; 3], [1.0; 3], 0), // host
            make_asset([0.0; 3], [1.0; 3], 7), // user-shader proto
        ];
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let instances = vec![make_instance(0, identity, 1), make_instance(1, identity, 2)];
        let (out, count) = cpu_reference_assemble_host(&instances, &assets, 16);
        assert_eq!(count, 1);
        assert_eq!(out[0].asset_id, 0);
        assert_eq!(out[0].material_id, 1);
        assert_eq!(out[0].instance_index, 0);
    }

    #[test]
    fn host_transforms_aabb_with_translation() {
        let assets = vec![make_asset([0.0; 3], [1.0; 3], 0)];
        let mut t = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        // (matches column-major layout for `world[col][row]`)
        let _ = t;
        t = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let instances = vec![make_instance(0, t, 0)];
        let (out, _) = cpu_reference_assemble_host(&instances, &assets, 16);
        assert_eq!(out[0].aabb_min, [10.0, 20.0, 30.0]);
        assert_eq!(out[0].aabb_max, [11.0, 21.0, 31.0]);
        assert_eq!(out[0].instance_index, 0);
    }

    #[test]
    fn ensure_prims_capacity_doubles_until_fit() {
        // CPU-only sanity check on the doubling logic. No GPU
        // device — just exercise the field math by mirroring it.
        let mut cap: u32 = 1;
        let target = 17u32;
        while cap < target {
            cap = cap.saturating_mul(2);
        }
        assert_eq!(cap, 32);
    }

    fn make_prim(min: [f32; 3], max: [f32; 3]) -> TlasPrim {
        TlasPrim {
            aabb_min: min,
            asset_id: 0,
            aabb_max: max,
            instance_state_offset: 0,
            material_id: 0,
            instance_index: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    #[test]
    fn scene_aabb_handles_empty_input() {
        let (mn, mx) = scene_aabb_from_prims(&[]);
        assert_eq!(mn, [0.0; 3]);
        assert_eq!(mx, [1.0; 3]);
    }

    #[test]
    fn scene_aabb_unions_multiple_prims() {
        let prims = vec![
            make_prim([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            make_prim([-2.0, 5.0, -3.0], [-1.0, 6.0, -2.0]),
            make_prim([3.0, 0.5, 0.5], [3.5, 1.5, 1.5]),
        ];
        let (mn, mx) = scene_aabb_from_prims(&prims);
        assert_eq!(mn, [-2.0, 0.0, -3.0]);
        assert_eq!(mx, [3.5, 6.0, 1.5]);
    }

    #[test]
    fn morton_30_interleaves_correctly() {
        // x=1 (001), y=2 (010), z=3 (011). Morton interleaves with z
        // at bits 0,3,6; y at 1,4,7; x at 2,5,8 (the WGSL writes
        // `(expand(x) << 2) | (expand(y) << 1) | expand(z)`).
        //   bit 0 (z0)=1, bit 1 (y0)=0, bit 2 (x0)=1,
        //   bit 3 (z1)=1, bit 4 (y1)=1, bit 5 (x1)=0,
        //   bit 6+ all zero.
        // = 1 + 4 + 8 + 16 = 29.
        let m = morton_30(1, 2, 3);
        assert_eq!(m, 29);
    }

    #[test]
    fn morton_preserves_locality_along_x() {
        // Centroids at (0,0,0), (1,0,0), (2,0,0) should produce
        // strictly increasing Mortons (since z and y stay 0; x
        // increments → bit 2, 5, 8 etc. flip).
        let prims: Vec<TlasPrim> = (0..4)
            .map(|i| {
                let f = i as f32;
                make_prim([f, 0.0, 0.0], [f + 0.1, 0.1, 0.1])
            })
            .collect();
        let scene_min = [0.0, 0.0, 0.0];
        let scene_max = [10.0, 1.0, 1.0];
        let pairs = cpu_reference_morton(&prims, scene_min, scene_max);
        for w in pairs.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "Morton not strictly increasing along x: {} -> {}",
                w[0].0,
                w[1].0,
            );
        }
    }

    #[test]
    fn cpu_reference_radix_sort_sorts_pairs() {
        let unsorted: Vec<(u32, u32)> = vec![(7, 0), (3, 1), (5, 2), (3, 3), (9, 4)];
        let sorted = cpu_reference_radix_sort(&unsorted);
        assert_eq!(sorted, vec![(3, 1), (3, 3), (5, 2), (7, 0), (9, 4)]);
    }

    #[test]
    fn karras_two_leaves_root_has_two_leaf_children() {
        // N=2: one internal node at idx 0, two leaf-markers at 1, 2.
        // Both children of root must be leaf-markers.
        let keys = [0b01u32, 0b10u32];
        let (l, r) = cpu_reference_karras_node(&keys, 0);
        assert_eq!(l, 1, "left child = leaf-marker for leaf 0");
        assert_eq!(r, 2, "right child = leaf-marker for leaf 1");
    }

    #[test]
    fn karras_three_leaves_balanced_tree() {
        // Mortons [1, 2, 4]: distinct, ascending. Expected topology
        // (matches the trace I worked out by hand):
        //     [0] internal: left=internal[1], right=leaf-marker[4]
        //     [1] internal: left=leaf-marker[2], right=leaf-marker[3]
        let keys = [1u32, 2, 4];
        let (l0, r0) = cpu_reference_karras_node(&keys, 0);
        let (l1, r1) = cpu_reference_karras_node(&keys, 1);
        // Internal node 0
        assert_eq!(l0, 1, "node 0 left = internal 1");
        assert_eq!(r0, 4, "node 0 right = leaf-marker 2 (= idx 4)");
        // Internal node 1
        assert_eq!(l1, 2, "node 1 left = leaf-marker 0 (= idx 2)");
        assert_eq!(r1, 3, "node 1 right = leaf-marker 1 (= idx 3)");
    }

    #[test]
    fn karras_four_leaves_two_subtrees() {
        // Mortons [0b00, 0b01, 0b10, 0b11]: full binary partition
        // expected by the algorithm — root splits 2-2.
        let keys = [0u32, 1, 2, 3];
        // Internal indices 0..2. Verify each.
        let (l0, r0) = cpu_reference_karras_node(&keys, 0);
        let (l1, r1) = cpu_reference_karras_node(&keys, 1);
        let (l2, r2) = cpu_reference_karras_node(&keys, 2);
        // Topology from a balanced 4-leaf Karras tree:
        //     [0] internal:  left=internal[1], right=internal[2]
        //     [1] internal:  left=leaf[3], right=leaf[4]   (= leaves 0, 1)
        //     [2] internal:  left=leaf[5], right=leaf[6]   (= leaves 2, 3)
        // Leaf-marker offset = N-1 = 3.
        assert_eq!((l0, r0), (1, 2));
        assert_eq!((l1, r1), (3, 4));
        assert_eq!((l2, r2), (5, 6));
    }

    #[test]
    fn karras_handles_duplicate_mortons() {
        // Duplicate-Morton case — algorithm relies on the index
        // tiebreak in `delta()`. Algorithm should still produce a
        // valid (topology-wise) tree.
        let keys = [5u32, 5, 5, 5];
        // For 4 identical Mortons, every internal node's delta_min
        // is decided purely by the index tiebreak. Tree should
        // still reach all 4 leaves exactly once.
        let mut leaf_visits = [0u32; 4];
        let mut visit = vec![false; 7]; // 4 leaves + 3 internal = 7 nodes
        // Walk from node 0 (root).
        let mut stack = vec![0u32];
        let n = 4u32;
        while let Some(idx) = stack.pop() {
            assert!(!visit[idx as usize], "node {idx} visited twice — cycle");
            visit[idx as usize] = true;
            if idx >= n - 1 {
                let leaf_idx = idx - (n - 1);
                leaf_visits[leaf_idx as usize] += 1;
                continue;
            }
            let (l, r) = cpu_reference_karras_node(&keys, idx as i32);
            assert!(l < 2 * n - 1, "child {l} out of range for n={n}");
            assert!(r < 2 * n - 1, "child {r} out of range");
            stack.push(l);
            stack.push(r);
        }
        for (i, &count) in leaf_visits.iter().enumerate() {
            assert_eq!(count, 1, "leaf {i} visited {count} times (expected once)");
        }
    }

    #[test]
    fn karras_random_eight_leaves_visits_all() {
        // 8 distinct ascending Mortons → full balanced tree, all
        // leaves reachable from root.
        let keys: Vec<u32> = (0..8u32).map(|i| i * 0x100 + 0x42).collect();
        let n = keys.len() as u32;
        let mut leaf_visits = vec![0u32; n as usize];
        let mut visit = vec![false; (2 * n - 1) as usize];
        let mut stack = vec![0u32];
        while let Some(idx) = stack.pop() {
            assert!(!visit[idx as usize]);
            visit[idx as usize] = true;
            if idx >= n - 1 {
                leaf_visits[(idx - (n - 1)) as usize] += 1;
                continue;
            }
            let (l, r) = cpu_reference_karras_node(&keys, idx as i32);
            stack.push(l);
            stack.push(r);
        }
        assert!(leaf_visits.iter().all(|&c| c == 1));
    }
}
