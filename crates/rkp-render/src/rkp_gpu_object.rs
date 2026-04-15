//! Per-object GPU metadata for RKIPatch's forward rasterization pipeline.
//!
//! [`RkpGpuObject`] carries everything the GPU needs per object: the forward
//! world transform, octree reference, material, bounds, and skeletal animation
//! data. No `inverse_world` — the raster vertex shader uses `world` directly,
//! and the shadow/AO pass computes the inverse in-shader at half-res.

use bytemuck::{Pod, Zeroable};

/// Per-object GPU data (256 bytes, bytemuck Pod).
///
/// Uploaded to a storage buffer and read by all RKIPatch shaders.
///
/// # Layout (256 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | world (mat4x4<f32>) — local→world |
/// | 64     | 12   | aabb_min (vec3<f32>) |
/// | 76     | 4    | octree_root (u32) |
/// | 80     | 12   | aabb_max (vec3<f32>) |
/// | 92     | 4    | octree_depth (u32) |
/// | 96     | 4    | octree_extent_bits (u32) — bitcast<f32> |
/// | 100    | 4    | voxel_size (f32) |
/// | 104    | 4    | material_id (u32) |
/// | 108    | 4    | object_id (u32) |
/// | 112    | 4    | geom_type (u32) |
/// | 116    | 4    | is_skinned (u32) |
/// | 120    | 4    | bone_count (u32) |
/// | 124    | 4    | bone_buffer_offset (u32) |
/// | 128    | 4    | rest_octree_root (u32) |
/// | 132    | 4    | rest_octree_depth (u32) |
/// | 136    | 4    | rest_octree_extent_bits (u32) |
/// | 140    | 4    | deformed_pool_offset (u32) |
/// | 144    | 4    | layer_mask (u32) — render-layer mask, gated against camera mask |
/// | 148    | 44   | _padding |
/// | 192    | 64   | inverse_world (mat4x4<f32>) — world→local |
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct RkpGpuObject {
    /// Forward world transform (local→world). Column-major.
    pub world: [[f32; 4]; 4],

    /// Object AABB minimum (local-space).
    pub aabb_min: [f32; 3],
    /// Octree root offset in the octree_nodes buffer.
    pub octree_root: u32,

    /// Object AABB maximum (local-space).
    pub aabb_max: [f32; 3],
    /// Octree depth (number of levels).
    pub octree_depth: u32,

    /// Octree world-space extent (one axis), stored as `f32.to_bits()`.
    pub octree_extent_bits: u32,
    /// Voxel size at finest octree level.
    pub voxel_size: f32,
    /// Primary material ID.
    pub material_id: u32,
    /// Unique object ID (matches scene object).
    pub object_id: u32,

    /// Geometry type: 0=None, 2=Voxelized.
    pub geom_type: u32,
    /// Non-zero if this object uses skeletal animation.
    pub is_skinned: u32,
    /// Number of bones in the skeleton.
    pub bone_count: u32,
    /// Offset into bone_matrices buffer (in mat4 units).
    pub bone_buffer_offset: u32,

    /// Rest-pose octree root for inverse skinning lookups.
    pub rest_octree_root: u32,
    /// Rest-pose octree depth.
    pub rest_octree_depth: u32,
    /// Rest-pose octree extent bits.
    pub rest_octree_extent_bits: u32,
    /// Offset into deformed bone-field pool.
    pub deformed_pool_offset: u32,

    /// 32-bit render-layer mask. Visible to a viewport iff
    /// `(layer_mask & camera.layer_mask) != 0  ||  object_id == camera.focus_object_id`.
    /// Default per-entity is `viewport::layer::DEFAULT` (bit 0).
    pub layer_mask: u32,

    /// Padding.
    pub _padding: [u32; 11],

    /// Inverse world transform (world→local). Precomputed on CPU.
    pub inverse_world: [[f32; 4]; 4],
}

/// Geometry type constants.
pub mod geom_type {
    /// No geometry.
    pub const NONE: u32 = 0;
    /// Voxelized geometry (octree lookup).
    pub const VOXELIZED: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn size_is_256_bytes() {
        // The struct's WGSL twin sits at exactly this size; the layout doc
        // comment above this struct lists every field offset. If this fires,
        // the WGSL `RkpObject` declarations need matching adjustments.
        assert_eq!(mem::size_of::<RkpGpuObject>(), 256);
    }

    #[test]
    fn layer_mask_at_offset_144() {
        // Verified by hand against the layout doc comment so the WGSL gate
        // (`obj.layer_mask & camera.layer_mask`) reads the correct bytes.
        let obj = RkpGpuObject::zeroed();
        let base = &obj as *const _ as usize;
        let field = &obj.layer_mask as *const _ as usize;
        assert_eq!(field - base, 144);
    }

    #[test]
    fn is_pod() {
        let obj = RkpGpuObject::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&obj);
    }
}
