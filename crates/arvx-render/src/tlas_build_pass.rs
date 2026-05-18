//! Phase 7c — GPU-built TLAS pipeline.
//!
//! Replaces `tlas_pass.rs::TlasPass::build_tlas`'s CPU median-split
//! BVH builder with a fully GPU-resident pipeline:
//!
//! 1. **Session 1** — primitive assembly. Two compute dispatches
//!    (`assemble_user_shader_main`, `assemble_host_main`) walk the
//!    per-frame inputs (tile-cull scratch + host instances) and pack
//!    tight world-space AABBs + leaf payloads into `tlas_prims`.
//! 2. Session 2 — Morton codes + GPU radix sort.
//! 3. Session 3 — Karras radix tree.
//! 4. Session 4 — bottom-up AABB propagation.
//! 5. Session 5 — wire-up + cutover, replacing the CPU
//!    `tlas_pass::build_tlas` call in `render_worker`.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — `TlasPrim`, `InstanceTileCullEntry`, all uniform
//!   structs (Assemble/Morton/Radix/Karras), constants.
//! - [`pass`] — `TlasBuildPass` struct + impl + `GpuTlasBuildInputs` +
//!   bind-group helpers. The runtime side of the build.
//! - [`cpu_reference`] — CPU oracle for every stage, used by the
//!   integration tests in `crates/arvx-render/tests/tlas_*`.
//!
//! ## Why GPU
//!
//! The CPU TLAS used `pos ± region_thickness` per-leaf AABBs for
//! user-shader instances because it had no way to evaluate the
//! shader's `inst_aabb` hook. With grass-style shaders that's a
//! 3 m cube around each painted leaf — 5000 leaves' AABBs all
//! overlap, BVH traversal degenerates to ~linear, shadow trace
//! catastrophically slow (30-40 ms for one .5 m grass patch).
//! Phase 6's tile-cull AABB pass already evaluates `inst_aabb` on
//! the GPU and writes tight per-instance world AABBs into scratch;
//! the GPU-built TLAS reads that scratch directly. Tight per-leaf
//! AABBs → real BVH culling → shadow trace stays fast.

pub mod cpu_reference;
pub mod pass;
pub mod types;

// Public re-exports — keep `arvx_render::tlas_build_pass::Foo` stable.
pub use cpu_reference::{
    cpu_reference_assemble_host, cpu_reference_assemble_user_shader, cpu_reference_full_tree,
    cpu_reference_karras_node, cpu_reference_morton, cpu_reference_radix_sort, karras_delta,
    scene_aabb_from_prims,
};
pub use pass::{GpuTlasBuildInputs, TlasBuildPass};
pub use types::{
    AssembleHostUniform, InstanceTileCullEntry, MortonUniform, RadixUniform, TlasPrim, TlasState,
    RADIX_BUCKETS, RADIX_PASSES, RADIX_WG_SIZE, TLAS_DISPATCH_ARG_SLOTS,
    TLAS_DISPATCH_ARG_STRIDE, TLAS_DISPATCH_SLOT_DECODE, TLAS_DISPATCH_SLOT_INIT_ATOMIC,
    TLAS_DISPATCH_SLOT_KARRAS_INTERNAL, TLAS_DISPATCH_SLOT_KARRAS_LEAVES,
    TLAS_DISPATCH_SLOT_MORTON, TLAS_DISPATCH_SLOT_PROPAGATE, TLAS_DISPATCH_SLOT_RADIX,
    TLAS_LEAF_USER_SHADER, TLAS_PRIMS_INITIAL_ENTRIES,
};

#[cfg(test)]
#[path = "tlas_build_pass/tests.rs"]
mod tests;
