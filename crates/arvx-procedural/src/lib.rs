//! Procedural object system for Arrvox.
//!
//! A procedural object is an arena-based tree of nodes. Leaves are
//! analytical primitives (sphere, box, capsule, …); combinators are
//! boolean ops (Union, Intersect, Subtract); effects are single-child
//! warps and attribute rewrites (NoiseDisplace, Mirror, by-height /
//! by-noise material+color).
//!
//! The evaluator that turns `(tree, position) → (distance, material,
//! color)` lives entirely on the GPU in `arvx-render`:
//!   * `proc_eval.wgsl`  — RPN interpreter (shared).
//!   * `proc_raymarch.wgsl` — live raymarch preview.
//!   * `proc_sample.wgsl` — GPU voxel bake.
//!
//! This crate only owns the *data* (the tree + its serialization +
//! structural invariants) and the flattening step that turns a tree
//! into the GPU's opcode stream. For BUILD-viewport click-picking
//! against individual primitives, `leaves::eval_leaf_distance` is
//! the one CPU SDF helper that remains.

mod arena;
mod bounds;
pub mod flatten;
mod leaves;
pub mod node_kind;
mod version;

pub use arena::{Node, NodeId, ProceduralObject};
pub use bounds::{compute_all_bounds, compute_bounds, AabbCache};
pub use flatten::{flatten_tree, OpKind, ProcInstruction};
pub use leaves::eval_leaf_distance;
pub use node_kind::{MaterialCombine, NodeKind};
pub use version::{bump_node_version, is_stale, subtree_version};
