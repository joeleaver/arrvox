//! Scene-to-GPU synchronization — builds RkpGpuObject arrays from scene state.

use bytemuck::Zeroable;
use glam::{Mat4, Vec3, Vec4};
use rkp_render::rkp_gpu_object::{self, RkpGpuObject};

/// Screen-space AABB for tile culling (pixel coordinates).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenAabb {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

/// Compute screen-space AABBs for all GPU objects.
/// Projects each object's local AABB (transformed by world matrix) to pixel coordinates.
pub fn compute_screen_aabbs(
    objects: &[RkpGpuObject],
    view_proj: &Mat4,
    width: f32,
    height: f32,
) -> Vec<ScreenAabb> {
    objects.iter().map(|obj| {
        if obj.geom_type == 0 {
            return ScreenAabb::zeroed();
        }

        // Build the 8 corners of the local AABB.
        let extent = f32::from_bits(obj.octree_extent_bits);
        let half = extent * 0.5;
        let world = Mat4::from_cols_array_2d(&obj.world);

        let mut smin = Vec3::splat(f32::MAX);
        let mut smax = Vec3::splat(f32::MIN);

        for corner in 0..8u32 {
            let local = Vec3::new(
                if corner & 1 != 0 { half } else { -half },
                if corner & 2 != 0 { half } else { -half },
                if corner & 4 != 0 { half } else { -half },
            );
            let world_pos = world.transform_point3(local);
            let clip = *view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

            // Behind camera: conservatively expand to full screen.
            if clip.w <= 0.0 {
                return ScreenAabb { min_x: 0.0, min_y: 0.0, max_x: width, max_y: height };
            }

            let ndc = clip.truncate() / clip.w;
            let px = (ndc.x * 0.5 + 0.5) * width;
            let py = (0.5 - ndc.y * 0.5) * height;
            smin = smin.min(Vec3::new(px, py, 0.0));
            smax = smax.max(Vec3::new(px, py, 0.0));
        }

        ScreenAabb {
            min_x: smin.x,
            min_y: smin.y,
            max_x: smax.x,
            max_y: smax.y,
        }
    }).collect()
}

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
