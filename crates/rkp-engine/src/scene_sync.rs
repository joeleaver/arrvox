//! Scene-to-GPU synchronization — builds RkpGpuObject arrays from scene state.

use bytemuck::Zeroable;
use rkp_render::rkp_gpu_object::{self, RkpGpuObject};

/// Build an RkpGpuObject from a scene object's transform and spatial handle.
pub fn build_gpu_object(
    world_matrix: &glam::Mat4,
    aabb: &rkf_core::Aabb,
    spatial: &rkf_core::scene_node::SpatialHandle,
    voxel_size: f32,
    material_id: u16,
    object_id: u32,
) -> RkpGpuObject {
    let mut gpu = RkpGpuObject::zeroed();
    gpu.world = world_matrix.to_cols_array_2d();
    gpu.inverse_world = world_matrix.inverse().to_cols_array_2d();
    gpu.aabb_min = aabb.min.into();
    gpu.aabb_max = aabb.max.into();
    gpu.voxel_size = voxel_size;
    gpu.material_id = material_id as u32;
    gpu.object_id = object_id;
    gpu.geom_type = rkp_gpu_object::geom_type::VOXELIZED;

    if let rkf_core::scene_node::SpatialHandle::Octree {
        root_offset, depth, base_voxel_size, ..
    } = spatial
    {
        gpu.octree_root = *root_offset;
        gpu.octree_depth = *depth as u32;
        let extent = (1u32 << depth) as f32 * base_voxel_size;
        gpu.octree_extent_bits = extent.to_bits();
    }

    gpu
}
