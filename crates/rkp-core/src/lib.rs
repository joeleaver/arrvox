//! RKP-Core — sparse surface voxel primitives for the RKIPatch engine.
//!
//! The geometric primitive is a [`SparseOctree`] of leaves. Each leaf
//! carries a single [`LeafAttr`] (material IDs + octahedrally-packed
//! normal) plus an optional per-leaf color, both stored in [`LeafAttrPool`].
//! Surfaces are defined by leaf existence — a leaf is the voxel center that
//! sits inside a mesh. There is no per-voxel opacity field.
//!
//! Transparency is expressed per-material, not per-voxel.

pub mod aabb;
pub mod asset_file;
pub mod brick_face_links;
pub mod brick_map;
pub mod brick_pool;
pub mod cluster_mesh_data;
pub mod companion;
pub mod constants;
pub mod dirty_ranges;
pub mod laplacian_smooth;
pub mod leaf_attr;
pub mod leaf_attr_overlay;
pub mod leaf_attr_pool;
pub mod mesh_cluster;
pub mod mesh_extract;
pub mod mesh_lod;
pub mod octree_allocator;
pub mod prefilter;
pub mod scene_node;
pub mod sculpt;
pub mod sculpt_overlay;
pub mod sdf;
pub mod sdf_primitive;
pub mod sparse_octree;
pub mod voxel;
pub mod voxelize_octree;
pub mod world_position;

pub use aabb::{Aabb, WorldAabb};
pub use brick_pool::{BrickPool, BRICK_BYTES, BRICK_CELLS, BRICK_DIM, BRICK_EMPTY, BRICK_INTERIOR, BRICK_LEVELS};
pub use companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel, VolumetricBrick, VolumetricVoxel};
pub use dirty_ranges::DirtyRanges;
pub use leaf_attr::{pack_oct, unpack_oct, LeafAttr};
pub use leaf_attr_overlay::{LeafAttrOverlay, OverlayEntry};
pub use leaf_attr_pool::LeafAttrPool;
pub use sculpt_overlay::SculptOverlay;
pub use octree_allocator::{OctreeAllocator, OctreeHandle};
pub use scene_node::{SdfPrimitive, SpatialHandle, Transform};
pub use sdf_primitive::evaluate_primitive;
pub use sparse_octree::SparseOctree;
pub use voxel::VoxelSample;
pub use voxelize_octree::{voxelize_to_artifact, BakeArtifact};
pub use world_position::WorldPosition;
