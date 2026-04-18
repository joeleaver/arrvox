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
/// | 140    | 4    | bone_field_offset (u32) — in vec2<u32> cells |
/// | 144    | 4    | layer_mask (u32) — render-layer mask, gated against camera mask |
/// | 148    | 4    | bone_field_dim_x (u32) |
/// | 152    | 4    | bone_field_dim_y (u32) |
/// | 156    | 4    | bone_field_dim_z (u32) |
/// | 160    | 4    | bone_field_origin_x (f32 bits) |
/// | 164    | 4    | bone_field_origin_y (f32 bits) |
/// | 168    | 4    | bone_field_origin_z (f32 bits) |
/// | 172    | 4    | bone_field_occ_offset (u32) — start in u32 words |
/// | 176    | 12   | grid_origin (vec3<f32>) — entity-local start of the voxel grid |
/// | 188    | 4    | _post_grid_pad |
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
    /// Offset into the scene-wide bone_field buffer (in `vec2<u32>`
    /// cells, not bytes). Skin-deform scatters this object's entries
    /// starting here; the march's skinned branch reads from here.
    pub bone_field_offset: u32,

    /// 32-bit render-layer mask. Visible to a viewport iff
    /// `(layer_mask & camera.layer_mask) != 0  ||  object_id == camera.focus_object_id`.
    /// Default per-entity is `viewport::layer::DEFAULT` (bit 0).
    pub layer_mask: u32,

    /// Bone-field grid dimensions (voxel cells).
    pub bone_field_dim_x: u32,
    pub bone_field_dim_y: u32,
    pub bone_field_dim_z: u32,
    /// Bone-field grid origin in object-local space (f32 packed as bits).
    pub bone_field_origin_x: f32,
    pub bone_field_origin_y: f32,
    pub bone_field_origin_z: f32,

    /// Offset into the scene-wide bone-field occupancy bitmap, measured
    /// in u32 words. Each bit covers one 4³-cell brick of this object's
    /// bone_field slice; scatter sets bits with `atomicOr` and the
    /// skinned march reads them with `atomicLoad` to skip empty bricks.
    pub bone_field_occ_offset: u32,

    /// Entity-local start of the voxel grid, i.e. `aabb_center - extent/2`
    /// at voxelization time. Shaders convert world→local via
    /// `inverse_world`, then compute octree coords as
    /// `local_pos - grid_origin` (which lands in `[0, extent]`).
    /// Previously the shader hardcoded `local_pos + extent/2`, which
    /// only matched when the AABB was symmetric around the origin.
    pub grid_origin: [f32; 3],

    /// Stride pad after `grid_origin` (WGSL treats vec3 as 16-byte sized).
    pub _post_grid_pad: u32,

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
    fn grid_origin_at_offset_176() {
        // vec3<f32> fields in WGSL structs require 16-byte alignment.
        // With bone_field_* fields at 148..=172 the next 16-byte boundary
        // is 176 — verify so the shader and CPU agree on byte layout.
        let obj = RkpGpuObject::zeroed();
        let base = &obj as *const _ as usize;
        let field = &obj.grid_origin as *const _ as usize;
        assert_eq!(field - base, 176);
    }

    #[test]
    fn is_pod() {
        let obj = RkpGpuObject::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&obj);
    }
}
