//! Phase B-redux — prototype bake for instance shaders.
//!
//! For each registered shader with `@instance_proto`, this pass bakes a
//! prototype octree (the canonical mesh of one blade / pebble / particle)
//! into the host scene's pool tail. The host march, on band-cell hits,
//! descends the prototype via `descend_proto_octree` to produce per-pixel
//! ray hits — no per-instance state stored, all derived at march time
//! through the user's `inst_to_local` hook.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — pool sizing constants + `PrototypeEntry` cache record +
//!   per-depth math helpers.
//! - [`cache`] — `PrototypeCache` persistent per-shader cache + octree
//!   extent allocator with free-list reuse + `build_internal_levels`
//!   pre-builder.
//! - [`pass`] — `PrototypeBakePass` GPU runtime + `PrototypeUniform`
//!   wire format + `OCTREE_EMPTY` / `INTERNAL_ATTR_NONE` sentinels +
//!   `compose_proto_source` template splice.
//!
//! ## Why three pools (octree / brick / leaf-attr)
//!
//! Octree extents are per-prototype contiguous (the dense spine demands
//! it) — allocated from a bump cursor + free-list keyed on extent size
//! for re-bake reuse. Brick + leaf-attr extents are NOT tracked
//! per-prototype; the GPU bake atomic-bumps a single cursor pair (in
//! [`pass::PrototypeBakePass`]'s cursor buffers) at GPU time. Different
//! prototypes' slots interleave; the march follows octree → brick_id
//! → leaf_attr_id pointers, all absolute, so layout doesn't matter.
//! This drops the worst-case per-prototype reservation that capped
//! depth at 4 (a depth-8 sparse blade has ~1.2 M leaf-attrs but the
//! bucket allocator was reserving 1 G slots up front, no matter what
//! the bake actually emitted).

pub mod cache;
pub mod pass;
pub mod types;

// Public re-exports — keep `rkp_render::user_shader_proto_pass::Foo` stable.
pub use cache::{build_internal_levels, PrototypeCache};
pub use pass::{
    compose_proto_source, PrototypeBakePass, PrototypeUniform, RollupUniform,
    ROLLUP_UNIFORM_BUFFER_SIZE, ROLLUP_UNIFORM_STRIDE,
};
pub use types::{
    level_starts_inclusive, max_bricks_for_depth, max_leaf_attrs_for_depth,
    octree_node_count_for_depth, PrototypeEntry, DEFAULT_PROTO_MAX_DEPTH, MAX_PROTOTYPES,
    MAX_PROTO_MAX_DEPTH, PROTO_BRICK_POOL_CAPACITY, PROTO_LEAF_ATTR_POOL_CAPACITY,
    PROTO_OCTREE_POOL_CAPACITY, PROTO_TAIL_BRICK_BYTES, PROTO_TAIL_LEAF_ATTR_BYTES,
    PROTO_TAIL_OCTREE_BYTES, INTERNAL_ATTR_NONE, OCTREE_EMPTY,
};

#[cfg(test)]
#[path = "user_shader_proto_pass/tests.rs"]
mod tests;
