//! Procedural object system for RKIPatch.
//!
//! A procedural object is an arena-based tree of nodes. Each node implements
//! `sample(pos) -> Sample` — leaves are analytical shapes (sphere, box, etc.),
//! combinators are boolean ops (union, intersect, subtract). The tree evaluates
//! at arbitrary positions, producing opacity + material + color that feeds into
//! the voxelization pipeline.

mod arena;
mod bounds;
mod combine;
mod evaluate;
mod leaves;
pub mod node_kind;
mod sample;
mod version;

pub use arena::{Node, NodeId, ProceduralObject};
pub use bounds::{compute_all_bounds, compute_bounds, AabbCache};
pub use combine::{combine_intersect, combine_subtract, combine_union};
pub use evaluate::{sample_tree, sample_tree_cached};
pub use leaves::eval_leaf;
pub use node_kind::{MaterialCombine, NodeKind};
pub use sample::Sample;
pub use version::{bump_node_version, is_stale, subtree_version};
