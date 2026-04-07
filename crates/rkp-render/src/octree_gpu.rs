//! GPU octree buffer management.
//!
//! Wraps [`OctreeAllocator`] to coordinate with the existing `GpuBrickMaps`
//! buffer (group 0, binding 1). The octree node data is uploaded to the same
//! buffer that previously held flat brick map entries — same binding, same
//! buffer type, different content and access pattern.
//!
//! # GpuObject field reinterpretation
//!
//! RKIPatch reinterprets the following GpuObject fields for octree objects:
//!
//! | Original field         | Offset | Octree meaning                              |
//! |------------------------|--------|---------------------------------------------|
//! | `brick_map_offset`     | 96     | `octree_root` — offset into octree buffer   |
//! | `brick_map_dims[0]`    | 100    | `octree_depth`                              |
//! | `brick_map_dims[1]`    | 104    | `bitcast::<u32>(octree_extent)` (world-space)|
//! | `brick_map_dims[2]`    | 108    | reserved (0)                                |
//! | `rest_brick_map_offset`| 208    | `rest_octree_root` (skinned rest-pose)      |
//! | `rest_brick_map_dims[0]` | 212  | `rest_octree_depth`                         |
//! | `rest_brick_map_dims[1]` | 216  | `bitcast::<u32>(rest_octree_extent)`        |
//! | `rest_brick_map_dims[2]` | 220  | reserved (0)                                |

use rkp_core::{OctreeAllocator, OctreeHandle, SparseOctree};

/// Manages octree allocation and provides data for GPU upload.
///
/// The backing `OctreeAllocator` packs all octrees into a single `Vec<u32>`.
/// Call [`data()`](Self::data) to get the slice for uploading to the GPU via
/// `GpuScene::upload_brick_maps()`.
pub struct OctreeGpu {
    allocator: OctreeAllocator,
}

impl OctreeGpu {
    /// Create with default capacity.
    pub fn new() -> Self {
        Self {
            allocator: OctreeAllocator::new(),
        }
    }

    /// Allocate an octree and return its handle.
    pub fn allocate(&mut self, octree: &SparseOctree) -> OctreeHandle {
        self.allocator.allocate(octree)
    }

    /// Deallocate an octree.
    pub fn deallocate(&mut self, handle: OctreeHandle) {
        self.allocator.deallocate(handle);
    }

    /// Allocate raw octree nodes directly (for file loading).
    ///
    /// The nodes must already have correct branch offsets (0-based within the
    /// node array). The allocator rebases them to absolute offsets.
    pub fn allocate_raw(&mut self, nodes: &[u32], depth: u8, base_voxel_size: f32) -> OctreeHandle {
        let octree = rkp_core::SparseOctree::from_raw(nodes, depth, base_voxel_size);
        self.allocator.allocate(&octree)
    }

    /// Raw data slice for GPU upload via `GpuScene::upload_brick_maps()`.
    pub fn data(&self) -> &[u32] {
        self.allocator.as_slice()
    }

    /// Total buffer length in u32 entries.
    pub fn buffer_len(&self) -> usize {
        self.allocator.buffer_len()
    }

    /// Write octree handle fields into a GpuObject's brick_map fields.
    ///
    /// This reinterprets the flat brick map fields to carry octree metadata.
    pub fn write_gpu_object_fields(
        handle: &OctreeHandle,
        gpu_obj: &mut rkf_render::gpu_object::GpuObject,
    ) {
        let extent = (1u32 << handle.depth) as f32 * 8.0 * handle.base_voxel_size;
        gpu_obj.brick_map_offset = handle.root_offset;
        gpu_obj.brick_map_dims[0] = handle.depth as u32;
        gpu_obj.brick_map_dims[1] = extent.to_bits();
        gpu_obj.brick_map_dims[2] = 0;
    }

    /// Write octree handle fields into a GpuObject's rest-pose brick map fields
    /// (for skinned objects).
    pub fn write_gpu_object_rest_fields(
        handle: &OctreeHandle,
        gpu_obj: &mut rkf_render::gpu_object::GpuObject,
    ) {
        gpu_obj.rest_brick_map_offset = handle.root_offset;
        gpu_obj.rest_brick_map_dims[0] = handle.depth as u32;
        gpu_obj.rest_brick_map_dims[1] = handle.base_voxel_size.to_bits();
        gpu_obj.rest_brick_map_dims[2] = 0;
    }
}

impl Default for OctreeGpu {
    fn default() -> Self {
        Self::new()
    }
}
