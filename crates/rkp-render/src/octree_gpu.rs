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
    /// Live handles for all allocated octrees (for building RkpGpuObjects).
    live_handles: Vec<OctreeHandle>,
}

impl OctreeGpu {
    /// Create with default capacity.
    pub fn new() -> Self {
        Self {
            allocator: OctreeAllocator::new(),
            live_handles: Vec::new(),
        }
    }

    /// Allocate an octree and return its handle.
    pub fn allocate(&mut self, octree: &SparseOctree) -> OctreeHandle {
        let handle = self.allocator.allocate(octree);
        self.live_handles.push(handle.clone());
        handle
    }

    /// Deallocate an octree.
    pub fn deallocate(&mut self, handle: OctreeHandle) {
        self.live_handles.retain(|h| h.root_offset != handle.root_offset);
        self.allocator.deallocate(handle);
    }

    /// Allocate raw octree nodes directly (for file loading).
    ///
    /// The nodes must already have correct branch offsets (0-based within the
    /// node array). The allocator rebases them to absolute offsets.
    pub fn allocate_raw(&mut self, nodes: &[u32], depth: u8, base_voxel_size: f32) -> OctreeHandle {
        let octree = rkp_core::SparseOctree::from_raw(nodes, depth, base_voxel_size);
        let handle = self.allocator.allocate(&octree);
        self.live_handles.push(handle.clone());
        handle
    }

    /// All currently live octree handles.
    pub fn handles(&self) -> &[OctreeHandle] {
        &self.live_handles
    }

    /// Raw data slice for GPU upload via `GpuScene::upload_brick_maps()`.
    pub fn data(&self) -> &[u32] {
        self.allocator.as_slice()
    }

    /// Parallel slice of prefiltered-LOD attr ids, one per node slot.
    /// The GPU binding (see [`RkpScene`]) interleaves these with the node
    /// values into a single `array<vec2<u32>>` storage buffer.
    pub fn internal_attrs_data(&self) -> &[u32] {
        self.allocator.internal_attrs_slice()
    }

    /// Total buffer length in u32 entries.
    pub fn buffer_len(&self) -> usize {
        self.allocator.buffer_len()
    }

    /// Apply a [`rkp_core::sparse_octree::OctreeMutationLog`] (typically
    /// produced by [`rkp_core::sculpt::apply_delta`]) to the packed
    /// buffer slot at `handle.root_offset`. Bounds-checks each write
    /// against the slot's **reserved capacity** (not its `len`) so
    /// writes into the slack region populated by [`try_extend_in_slack`]
    /// succeed without forcing a full slot re-allocation.
    ///
    /// Drops any write whose local index falls outside the slot's
    /// reservation — those indicate the tree grew past even the slack
    /// and the caller must have re-allocated before calling this.
    pub fn apply_mutation_log(
        &mut self,
        handle: &OctreeHandle,
        log: &rkp_core::sparse_octree::OctreeMutationLog,
    ) {
        let base = handle.root_offset;
        let cap = self
            .allocator
            .reserved_capacity(base)
            .max(handle.len);
        for &(local_idx, value) in &log.node_writes {
            if local_idx >= cap {
                debug_assert!(
                    false,
                    "OctreeMutationLog node write idx {local_idx} past slot reserved cap {cap}.",
                );
                continue;
            }
            self.allocator.write_node(base + local_idx, value, base);
        }
        for &(local_idx, value) in &log.attr_writes {
            if local_idx >= cap {
                debug_assert!(
                    false,
                    "OctreeMutationLog attr write idx {local_idx} past slot reserved cap {cap}.",
                );
                continue;
            }
            self.allocator.write_internal_attr(base + local_idx, value);
        }
    }

    /// Try to grow `handle.len` in place to `new_len` by consuming
    /// slack. Returns `Some(new_handle)` on success; `None` if slack
    /// is exhausted (caller should `deallocate` + `allocate_with_slack`).
    pub fn try_extend_in_slack(
        &self,
        handle: &OctreeHandle,
        new_len: u32,
    ) -> Option<OctreeHandle> {
        self.allocator.try_extend_in_slack(handle, new_len)
    }

    /// Same as [`allocate`] but reserves `slack_factor` × `node_count`
    /// slots so subsequent growth can land in slack via
    /// [`try_extend_in_slack`].
    pub fn allocate_with_slack(
        &mut self,
        octree: &SparseOctree,
        slack_factor: f32,
    ) -> OctreeHandle {
        let handle = self.allocator.allocate_with_slack(octree, slack_factor);
        self.live_handles.push(handle);
        handle
    }

    /// Read-only view of the allocator's dirty range tracker. Used by
    /// the upload path to drive delta writes.
    pub fn dirty_ranges(&self) -> &rkp_core::DirtyRanges {
        self.allocator.dirty_ranges()
    }

    /// Mutable view of the allocator's dirty range tracker. The upload
    /// pass calls `clear()` after writing the deltas.
    pub fn dirty_ranges_mut(&mut self) -> &mut rkp_core::DirtyRanges {
        self.allocator.dirty_ranges_mut()
    }
}

impl Default for OctreeGpu {
    fn default() -> Self {
        Self::new()
    }
}
