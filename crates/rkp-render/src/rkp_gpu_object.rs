//! Per-object GPU metadata for RKIPatch.
//!
//! Two GPU-side types, deduplicated per upload (see `rkp_scene::upload_frame`):
//!
//! - [`RkpGpuAsset`] — per-asset record, ~96 B. One entry per unique
//!   octree (deduped by `octree_root`). Holds the octree topology, local
//!   AABB, voxel size, grid origin, and skinning *template* (rest octree
//!   + bone-field grid). Multiple instances sharing one .rkp asset point
//!   at the same `RkpGpuAsset` slot via `RkpGpuInstance.asset_id`.
//!
//! - [`RkpGpuInstance`] — per-instance record, ~160 B (Phase 1; Phase 2
//!   drops `inverse_world` → 96 B). Holds the world transform, asset id,
//!   material override, picking id, layer mask, and per-frame skinning
//!   runtime offsets (palette + bone-field allocation). One per scene
//!   entity.
//!
//! [`RkpGpuObject`] is the CPU-side construction view (the legacy
//! 256-byte struct). `RkpScene::upload_frame` walks `&[RkpGpuObject]`,
//! dedupes per-asset fields, and uploads to two separate GPU buffers.
//! Phase 1b will replace `RkpGpuObject` with direct construction of
//! `RkpGpuAsset` + `RkpGpuInstance` to reclaim CPU memory too.

use bytemuck::{Pod, Zeroable};

/// Geometry type constants.
pub mod geom_type {
    /// No geometry.
    pub const NONE: u32 = 0;
    /// Voxelized geometry (octree lookup).
    pub const VOXELIZED: u32 = 2;
}

/// Per-asset GPU record (96 bytes). Holds the data that's identical
/// across every instance of one .rkp asset:
/// - octree topology (`octree_root`, `octree_depth`, `octree_extent_bits`)
/// - voxelization (`voxel_size`, `aabb_min/max`, `grid_origin`)
/// - skinning template (`bone_count`, rest octree refs, bone-field grid
///   dimensions and origin)
///
/// Multiple [`RkpGpuInstance`]s share one slot via [`RkpGpuInstance::asset_id`].
/// Built CPU-side (Phase 1a: by `RkpScene::upload_frame` deduping a
/// `&[RkpGpuObject]`; Phase 1b: directly).
///
/// # Layout (80 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 12   | aabb_min (vec3<f32>) |
/// | 12     | 4    | octree_root (u32) |
/// | 16     | 12   | aabb_max (vec3<f32>) |
/// | 28     | 4    | octree_depth (u32) |
/// | 32     | 4    | octree_extent_bits (u32) — bitcast<f32> |
/// | 36     | 4    | voxel_size (f32) |
/// | 40     | 4    | geom_type (u32) |
/// | 44     | 4    | bone_count (u32) — 0 for non-skinned assets |
/// | 48     | 12   | grid_origin (vec3<f32>) |
/// | 60     | 4    | rest_octree_root (u32) |
/// | 64     | 4    | rest_octree_depth (u32) |
/// | 68     | 4    | rest_octree_extent_bits (u32) |
/// | 72     | 4    | shader_id (u32) — 0 = standard host asset; >0 = user-shader instance proto. The march branches on this to call `dispatch_user_inst_to_local` / `dispatch_user_inst_aabb` instead of the affine `inv_world` path. |
/// | 76     | 4    | _pad |
///
/// `bone_count` and `rest_octree_*` are skeleton-template properties —
/// same across every instance of one skinned asset. Phase 1b sources
/// them directly from the asset cache's `SkinningAssetData`, so they
/// stay populated even if a particular instance's per-frame skin plan
/// bails (deformed AABB out of bounds, dimension cap, etc).
///
/// `bone_field_dim_*` and `bone_field_origin_*` are NOT here — those
/// describe the per-frame deformed AABB grid, which depends on the
/// instance's current pose, so they live on `RkpGpuInstance`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct RkpGpuAsset {
    pub aabb_min: [f32; 3],
    pub octree_root: u32,
    pub aabb_max: [f32; 3],
    pub octree_depth: u32,
    pub octree_extent_bits: u32,
    pub voxel_size: f32,
    pub geom_type: u32,
    pub bone_count: u32,
    pub grid_origin: [f32; 3],
    pub rest_octree_root: u32,
    pub rest_octree_depth: u32,
    pub rest_octree_extent_bits: u32,
    pub shader_id: u32,
    pub _pad: u32,
}

/// Per-instance GPU record (128 bytes). One per scene entity; carries:
/// - world transform (inverse computed on demand by the shader via
///   `mat4_affine_inverse(inst.world)` — ~25 ALU vs 64 B/instance saved)
/// - asset reference (`asset_id` indexes into the assets table)
/// - per-entity overrides (material, picking id, layer mask)
/// - per-frame skinning runtime state: bone palette offset + bone-field
///   allocation + per-pose deformed AABB grid
/// - per-instance paint overlay (sparse `(slot, attr, color)` entries
///   in a global overlay buffer; `overlay_count == 0` ⇒ asset's pool
///   values are used directly)
///
/// # Layout (128 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | world (mat4x4<f32>) |
/// | 64     | 4    | asset_id (u32) |
/// | 68     | 4    | material_id (u32) |
/// | 72     | 4    | object_id (u32) |
/// | 76     | 4    | layer_mask (u32) |
/// | 80     | 4    | is_skinned (u32) |
/// | 84     | 4    | bone_buffer_offset (u32) |
/// | 88     | 4    | bone_field_offset (u32) |
/// | 92     | 4    | bone_field_occ_offset (u32) |
/// | 96     | 4    | bone_field_dim_x (u32) |
/// | 100    | 4    | bone_field_dim_y (u32) |
/// | 104    | 4    | bone_field_dim_z (u32) |
/// | 108    | 4    | bone_field_origin_x (f32) |
/// | 112    | 4    | bone_field_origin_y (f32) |
/// | 116    | 4    | bone_field_origin_z (f32) |
/// | 120    | 4    | overlay_offset (u32) |
/// | 124    | 4    | overlay_count (u32) |
///
/// `overlay_offset` + `overlay_count` describe a slice into the
/// scene-global `instance_overlay` buffer (Phase 3). The WGSL fetch
/// helper falls through to `leaf_attr_pool[slot]` when
/// `overlay_count == 0`.
///
/// 128 B is a multiple of 16; the mat4x4 alignment requirement is
/// satisfied without trailing pad.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct RkpGpuInstance {
    pub world: [[f32; 4]; 4],
    pub asset_id: u32,
    pub material_id: u32,
    pub object_id: u32,
    pub layer_mask: u32,
    pub is_skinned: u32,
    pub bone_buffer_offset: u32,
    pub bone_field_offset: u32,
    pub bone_field_occ_offset: u32,
    pub bone_field_dim_x: u32,
    pub bone_field_dim_y: u32,
    pub bone_field_dim_z: u32,
    pub bone_field_origin_x: f32,
    pub bone_field_origin_y: f32,
    pub bone_field_origin_z: f32,
    pub overlay_offset: u32,
    pub overlay_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn asset_size_is_80_bytes() {
        assert_eq!(mem::size_of::<RkpGpuAsset>(), 80);
    }

    #[test]
    fn asset_grid_origin_at_offset_48() {
        // WGSL alignment: vec3<f32> at offset 48 lands on the third
        // 16-byte boundary, with rest_octree_root packed in the trailing
        // 4 bytes. Verifying the offset prevents alignment drift if
        // fields get reordered.
        let a = RkpGpuAsset::zeroed();
        let base = &a as *const _ as usize;
        let field = &a.grid_origin as *const _ as usize;
        assert_eq!(field - base, 48);
    }

    #[test]
    fn instance_size_is_128_bytes() {
        // mat4x4 alignment requires stride to be a multiple of 16; 128
        // satisfies it without any trailing pad.
        assert_eq!(mem::size_of::<RkpGpuInstance>(), 128);
    }

    #[test]
    fn instance_overlay_offset_at_offset_120() {
        let i = RkpGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.overlay_offset as *const _ as usize;
        assert_eq!(field - base, 120);
        let field2 = &i.overlay_count as *const _ as usize;
        assert_eq!(field2 - base, 124);
    }

    #[test]
    fn instance_bone_field_dim_at_offset_96() {
        let i = RkpGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.bone_field_dim_x as *const _ as usize;
        assert_eq!(field - base, 96);
    }

    #[test]
    fn instance_asset_id_at_offset_64() {
        let i = RkpGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.asset_id as *const _ as usize;
        assert_eq!(field - base, 64);
    }

    #[test]
    fn instance_is_pod() {
        let i = RkpGpuInstance::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&i);
        let a = RkpGpuAsset::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&a);
    }
}
