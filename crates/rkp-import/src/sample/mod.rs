//! Material transfer: sample per-voxel colour and bone weights from a
//! source mesh at known triangle + barycentric positions.
//!
//! Split by concern — [`texture`] for albedo-colour sampling (with
//! KHR_texture_transform + tiling), [`bone_weights`] for barycentric
//! bone-influence interpolation + top-4 quantization. Both expect the
//! caller (typically [`crate::voxelize`]) to already have a nearest-
//! triangle query result from [`crate::bvh::TriangleBvh`].

pub mod bone_weights;
pub mod texture;

pub use bone_weights::sample_bone_weights_at_triangle;
pub use texture::{VoxelColor, sample_texture_at_triangle};
