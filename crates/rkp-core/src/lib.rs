//! RKP-Core: Splat voxel types and brick format for the RKIPatch gaussian splat
//! engine.
//!
//! The core type is [`SplatVoxel`] — a zero-cost wrapper over rkf-core's
//! [`VoxelSample`](rkf_core::voxel::VoxelSample) that reinterprets the SDF
//! distance field as an opacity value. Same 8-byte format, same brick pools,
//! same material system — just different semantics for word0 bits 0–15.

mod splat_voxel;
pub mod asset_file;
pub mod octree_allocator;
pub mod opacity_shaders_cpu;
pub mod sparse_octree;
pub mod voxelize_octree;
pub mod voxelize_opacity;

pub use octree_allocator::{OctreeAllocator, OctreeHandle};
pub use sparse_octree::SparseOctree;
pub use splat_voxel::SplatVoxel;
