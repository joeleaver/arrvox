//! Paint-write operations against the scene's leaf_attr pool.
//!
//! Given a brush sphere in object-local space, enumerate the leaf_attr slots
//! inside the sphere and mutate their material or color. The existing
//! `LeafAttrPool` already owns both the per-leaf material entries and the
//! parallel `colors` array, and GPU upload is driven by the scene manager's
//! `geometry_epoch` — so paint is a CPU-side edit plus one epoch bump.
//!
//! ## Module layout (post-split)
//!
//! - [`select`] — spatial queries: sphere brush ([`leaves_in_sphere`])
//!   and single-cell cursor pick ([`leaf_at_local_pos`]). Plus the
//!   output types ([`PaintedLeaf`], [`LeafHit`]).
//! - [`write`] — paint write operations: [`PaintStamp`] +
//!   [`paint_leaf_material`] / [`paint_leaf_color`] / [`erase_leaf_color`]
//!   + their `compute_*` helpers + brush-weight + color packing.
//!
//! This module is agnostic to commands / input / UI — it operates on raw
//! octree + brick data and a `LeafAttrPool`. Call sites (the engine's paint
//! command handler) are responsible for looking up the target entity's
//! `AssetInfo` and resolving the brush world position into object-local
//! space.

pub mod select;
pub mod write;

// Public re-exports — keep `rkp_render::paint::Foo` stable.
pub use select::{leaf_at_local_pos, leaves_in_sphere, LeafHit, PaintedLeaf};
pub use write::{
    brush_weight, compute_erased_color, compute_painted_attr, compute_painted_color,
    erase_leaf_color, pack_color, paint_leaf_color, paint_leaf_material, unpack_color, PaintStamp,
};

#[cfg(test)]
#[path = "paint/tests.rs"]
mod tests;
