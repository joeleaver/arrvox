//! `TlasBuildPass` — owns all GPU pipelines + buffers for the Phase 7c
//! GPU TLAS build (assemble → Morton → radix sort → Karras → propagate).
//!
//! [`TlasBuildPass::new`] builds every pipeline + storage buffer at
//! startup; [`TlasBuildPass::dispatch`] encodes the per-frame chain
//! against [`GpuTlasBuildInputs`].
//!
//! Buffer-capacity growth: `ensure_prims_capacity` /
//! `ensure_prim_keys_capacity` etc. realloc storage buffers to fit
//! `prim_count` for the frame. Initial sizes match the wgpu binding-
//! validation rule that storage buffers must exist before the bind
//! group is built (see [`super::types::TLAS_PRIMS_INITIAL_ENTRIES`]).

use super::types::{TlasPrim, RADIX_BUCKETS};

mod build;
mod constructor;

/// Pipeline holder for Phase 7c GPU TLAS build. Session 1 owns the
/// primitive-assembly pipelines and the shared output
/// (`tlas_prims_buffer` + `tlas_prim_count_buffer`); Session 2
/// adds the Morton + radix-sort pipelines and ping-pong key/value
/// buffers; later sessions extend with the Karras tree builder
/// and AABB propagation.
pub struct TlasBuildPass {
    // ── Session 1 ──────────────────────────────────────────────────
    pub host_pipeline: wgpu::ComputePipeline,
    pub host_g0_layout: wgpu::BindGroupLayout,
    pub host_g1_layout: wgpu::BindGroupLayout,
    /// Packed `array<TlasPrim>`. Capacity grows; `tlas_prim_count`
    /// holds the per-frame live count after the host assembly
    /// dispatch finishes.
    pub tlas_prims_buffer: wgpu::Buffer,
    pub tlas_prims_capacity: u32,
    /// Single-element `array<atomic<u32>>` — the assembly pass
    /// `atomicAdd`s into slot 0. Engine zeroes per frame before
    /// dispatch.
    pub tlas_prim_count_buffer: wgpu::Buffer,
    /// Per-dispatch uniform — re-uploaded per frame.
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

    // ── Indirect-dispatch driver ──────────────────────────────────
    /// Shared per-frame state — `prim_count`, `radix_workgroups`,
    /// `internal_wgs`, `total_node_wgs`. Written by
    /// `tlas_compute_dispatch_args.wesl`; read by every chain
    /// shader as a substitute for per-pass-uniform-driven counts.
    pub tlas_state_buffer: wgpu::Buffer,
    /// Packed indirect-dispatch arguments — 7 slots × 12 B
    /// (u32 x/y/z workgroup counts). Drives every chain dispatch
    /// via `dispatch_workgroups_indirect`. Replaced the V1
    /// CPU-side readback + direct dispatch.
    pub tlas_dispatch_args_buffer: wgpu::Buffer,
    /// Single-thread, single-workgroup pipeline that reads
    /// `tlas_prim_count[0]` and writes both
    /// `tlas_state_buffer` + `tlas_dispatch_args_buffer`. Run
    /// once per frame at the head of the chain.
    pub dispatch_args_pipeline: wgpu::ComputePipeline,
    pub dispatch_args_g0_layout: wgpu::BindGroupLayout,
    pub dispatch_args_g1_layout: wgpu::BindGroupLayout,
}

/// Inputs to [`TlasBuildPass::build_gpu_tlas`]. All buffers are
/// borrowed from external owners; the build pass binds them as
/// pipeline inputs but doesn't take ownership.
pub struct GpuTlasBuildInputs<'a> {
    /// `state.renderer.scene.objects_buffer` — `array<ArvxGpuInstance>`.
    pub instances_buffer: &'a wgpu::Buffer,
    pub instance_count: u32,
    /// `state.renderer.scene.assets_buffer` — `array<ArvxGpuAsset>`.
    pub assets_buffer: &'a wgpu::Buffer,
    pub asset_count: u32,
    /// CPU-derived scene AABB for Morton normalization.
    /// Conservative is fine — Morton sort just needs a stable
    /// coordinate system.
    pub scene_min: [f32; 3],
    pub scene_max: [f32; 3],
}

impl TlasBuildPass {

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

pub(super) fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub(super) fn uniform_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
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
