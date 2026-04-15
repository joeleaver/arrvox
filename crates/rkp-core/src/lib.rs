//! RKP-Core — sparse surface voxel primitives for the RKIPatch engine.
//!
//! The geometric primitive is a [`SparseOctree`] of leaves. Each leaf
//! carries a single [`LeafAttr`] (material IDs + octahedrally-packed
//! normal) plus an optional per-leaf color, both stored in [`LeafAttrPool`].
//! Surfaces are defined by leaf existence — a leaf is the voxel center that
//! sits inside a mesh. There is no per-voxel opacity field.
//!
//! Transparency is expressed per-material, not per-voxel.

pub mod asset_file;
pub mod brick_pool;
pub mod leaf_attr;
pub mod leaf_attr_pool;
pub mod octree_allocator;
pub mod prefilter;
pub mod sparse_octree;
pub mod voxelize_octree;

pub use brick_pool::{BrickPool, BRICK_BYTES, BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_LEVELS};
pub use leaf_attr::{pack_oct, unpack_oct, LeafAttr};
pub use leaf_attr_pool::LeafAttrPool;
pub use octree_allocator::{OctreeAllocator, OctreeHandle};
pub use sparse_octree::SparseOctree;
