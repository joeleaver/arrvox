//! Sparse BFS GPU runtime geometry from user shaders, global-pool variant.
//!
//! Owns the GPU pipelines that build transient octrees by atomically
//! allocating nodes / bricks / leaf-attrs from a SINGLE global pool
//! shared across all regions in the frame. Memory and compute scale
//! with painted surface area rather than the (4·2^depth)³ cube the
//! original dense-brick model demanded.
//!
//! ## Module layout (post-split)
//!
//! - [`cache`] — `BucketPoolAllocator`, `UserShaderObjectCache`,
//!   `ShaderRegionRequest`, `CachedSlot`, `PoolEstimate`,
//!   `estimate_region_pool`, all pool / bucket constants.
//! - [`region`] — `RegionUniform` (GPU-side per-region storage), the
//!   16-byte `GpuBandCell` band-cell payload, `build_region_uniform`.
//! - [`dispatch`] — `UserShaderPass` (BFS pipelines + buffers),
//!   `compose_geom_source`, `resolve_shader_id`, `LevelUniform`,
//!   `MAX_DEPTH`, `MAX_REGIONS`.
//! - [`overflow`] — internal CPU readback ring for the GPU overflow
//!   counters; not part of the public API.
//!
//! All previously-public symbols are re-exported at this module level
//! so `rkp_render::user_shader_pass::Foo` keeps working unchanged.
//!
//! ## Pool layout
//!
//! All regions in a frame draw from the same three flat tails:
//!   - `octree_nodes` — `MAX_GLOBAL_OCTREE_NODES` slots
//!   - `brick_pool`   — `MAX_GLOBAL_BRICKS` bricks
//!   - `leaf_attr_pool` — `MAX_GLOBAL_LEAF_ATTRS` slots
//!
//! These tails live in the scene's flat pools past the CPU-managed
//! head (same buffers the march reads). `RkpScene::ensure_user_shader_capacity`
//! grows the buffers once at startup and they stay stable.
//!
//! Three GLOBAL atomic counters (`octree_alloc`, `brick_alloc`,
//! `leaf_attr_alloc`) — single u32 each — bump-allocate within those
//! ranges as the BFS expands. Overflow degrades to `OCTREE_EMPTY` at
//! the offending node and increments a per-pool counter in the
//! `overflow` buffer; CPU reads that buffer asynchronously and logs
//! when caps are hit.
//!
//! ## Dispatch chain (per frame)
//!
//! 1. Counters reset: `octree_alloc = region_count` (one root per
//!    region, sequentially placed at the head of the transient
//!    octree slice), `brick_alloc = 0`, `leaf_attr_alloc = 0`,
//!    `fill_count = 0`, `active_count[*] = 0`, `overflow[*] = 0`.
//! 2. Active queue seeded with one root cell per region into
//!    `active_queue[level=0]`, `active_count[0] = region_count`.
//!    Each cell's `octree_offset = pool_octree_base + region_index`.
//! 3. For L in 0..=max(max_depth across regions): one
//!    `classify_main` dispatch with `level_u.current_level = L`. Threads
//!    past `active_count[L]` early-out, so we always issue
//!    `(per_level_cap / 64)` workgroups regardless of true active count
//!    (saves a build-indirect-args dispatch).
//! 4. One `brick_fill_main` dispatch over the surviving fill_queue.
//!
//! All dispatches share group-0 (scene + global counters + overflow
//! buffer) and group-1 (region storage array). Group-2 holds the
//! per-dispatch level uniform — which now also carries the global
//! pool bases and caps — at a dynamic offset.
//!
//! ## Compose contract
//!
//! `compose_geom_source` splices the composer's `generate` chunk
//! between the `USER_GENERATE_DISPATCH_BEGIN/_END` const-decl
//! anchors in `user_shader_geom.wgsl`. User shaders that called
//! `host_sample_at(world_pos)` keep working unchanged.

pub mod cache;
pub mod dispatch;
mod overflow;
pub mod region;

// Public re-exports — keep `rkp_render::user_shader_pass::Foo` stable
// for external callers regardless of which submodule a symbol lives in.
pub use cache::{
    estimate_region_pool, BucketPoolAllocator, CachedSlot, PoolEstimate, ShaderRegionRequest,
    UserShaderObjectCache,
    BRICK_BUCKET_MAX, BRICK_BUCKET_MIN, BRICK_CELLS, FILL_TASK_BUCKET_MAX, FILL_TASK_BUCKET_MIN,
    HOST_NO_HOST_SENTINEL, LEAF_ATTR_BUCKET_MAX, LEAF_ATTR_BUCKET_MIN, MAX_GLOBAL_BAND_CELLS,
    MAX_GLOBAL_BRICKS, MAX_GLOBAL_FILL_TASKS, MAX_GLOBAL_LEAF_ATTRS, MAX_GLOBAL_OCTREE_NODES,
    NO_TILE, OCTREE_BUCKET_MAX, OCTREE_BUCKET_MIN,
};
pub use dispatch::{
    compose_geom_source, resolve_shader_id, LevelUniform, UserShaderPass, MAX_DEPTH, MAX_REGIONS,
};
pub use region::{build_region_uniform, GpuBandCell, RegionUniform};
