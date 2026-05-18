//! Per-object GPU metadata for Arrvox.
//!
//! Two GPU-side types, deduplicated per upload (see `arvx_scene::upload_frame`):
//!
//! - [`ArvxGpuAsset`] — per-asset record, ~96 B. One entry per unique
//!   octree (deduped by `octree_root`). Holds the octree topology, local
//!   AABB, voxel size, grid origin, and skinning *template* (rest octree
//!   + bone-field grid). Multiple instances sharing one .arvx asset point
//!   at the same `ArvxGpuAsset` slot via `ArvxGpuInstance.asset_id`.
//!
//! - [`ArvxGpuInstance`] — per-instance record, ~160 B (Phase 1; Phase 2
//!   drops `inverse_world` → 96 B). Holds the world transform, asset id,
//!   material override, picking id, layer mask, and per-frame skinning
//!   runtime offsets (palette + bone-field allocation). One per scene
//!   entity.
//!
//! [`ArvxGpuObject`] is the CPU-side construction view (the legacy
//! 256-byte struct). `ArvxScene::upload_frame` walks `&[ArvxGpuObject]`,
//! dedupes per-asset fields, and uploads to two separate GPU buffers.
//! Phase 1b will replace `ArvxGpuObject` with direct construction of
//! `ArvxGpuAsset` + `ArvxGpuInstance` to reclaim CPU memory too.

use bytemuck::{Pod, Zeroable};

/// Geometry type constants.
pub mod geom_type {
    /// No geometry.
    pub const NONE: u32 = 0;
    /// Voxelized geometry (octree lookup).
    pub const VOXELIZED: u32 = 2;
}

/// Per-asset GPU record (96 bytes). Holds the data that's identical
/// across every instance of one .arvx asset:
/// - octree topology (`octree_root`, `octree_depth`, `octree_extent_bits`)
/// - voxelization (`voxel_size`, `aabb_min/max`, `grid_origin`)
/// - skinning template (`bone_count`, rest octree refs, bone-field grid
///   dimensions and origin)
///
/// Multiple [`ArvxGpuInstance`]s share one slot via [`ArvxGpuInstance::asset_id`].
/// Built CPU-side (Phase 1a: by `ArvxScene::upload_frame` deduping a
/// `&[ArvxGpuObject]`; Phase 1b: directly).
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ArvxGpuAsset {
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

/// Per-instance GPU record (112 bytes). One per scene entity; carries:
/// - world transform (inverse computed on demand by the shader via
///   `mat4_affine_inverse(inst.world)` — ~25 ALU vs 64 B/instance saved)
/// - asset reference (`asset_id` indexes into the assets table)
/// - per-entity overrides (material, picking id, layer mask)
/// - per-instance skinning state: bone palette offset only (the mesh
///   VS skins per-vertex against the per-frame bone-matrix buffer; no
///   per-instance deformed-AABB grid lives here anymore)
/// - per-instance paint overlay (sparse `(slot, attr, color)` entries
///   in a global overlay buffer; `overlay_count == 0` ⇒ asset's pool
///   values are used directly)
/// - per-instance sculpt overlay (sparse `leaf_attr_id` removal set in
///   a global sculpt buffer; `sculpt_count == 0` ⇒ no carved leaves)
///
/// # Layout (112 bytes)
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
/// | 88     | 4    | overlay_offset (u32) |
/// | 92     | 4    | overlay_count (u32) |
/// | 96     | 4    | sculpt_offset (u32) |
/// | 100    | 4    | sculpt_count (u32) |
/// | 104    | 8    | _pad (8 B, keeps stride a 16 B multiple for mat4x4) |
///
/// `overlay_offset` + `overlay_count` describe a slice into the
/// scene-global `instance_overlay` buffer (Phase 3 paint). The WGSL fetch
/// helper falls through to `leaf_attr_pool[slot]` when
/// `overlay_count == 0`.
///
/// `sculpt_offset` + `sculpt_count` describe a slice into the
/// scene-global `instance_sculpt` buffer — a sorted `array<u32>` of
/// removed `leaf_attr_id`s (Phase A sculpt overlay). The WGSL helper
/// `is_leaf_removed` returns `false` when `sculpt_count == 0`.
///
/// 112 B is a multiple of 16; the mat4x4 alignment requirement is
/// satisfied via the trailing 8-byte pad.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ArvxGpuInstance {
    pub world: [[f32; 4]; 4],
    pub asset_id: u32,
    pub material_id: u32,
    pub object_id: u32,
    pub layer_mask: u32,
    pub is_skinned: u32,
    pub bone_buffer_offset: u32,
    pub overlay_offset: u32,
    pub overlay_count: u32,
    pub sculpt_offset: u32,
    pub sculpt_count: u32,
    pub _pad: [u32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn asset_size_is_80_bytes() {
        assert_eq!(mem::size_of::<ArvxGpuAsset>(), 80);
    }

    #[test]
    fn asset_grid_origin_at_offset_48() {
        // WGSL alignment: vec3<f32> at offset 48 lands on the third
        // 16-byte boundary, with rest_octree_root packed in the trailing
        // 4 bytes. Verifying the offset prevents alignment drift if
        // fields get reordered.
        let a = ArvxGpuAsset::zeroed();
        let base = &a as *const _ as usize;
        let field = &a.grid_origin as *const _ as usize;
        assert_eq!(field - base, 48);
    }

    #[test]
    fn instance_size_is_112_bytes() {
        // mat4x4 alignment requires stride to be a multiple of 16; 112
        // satisfies it via the 8-byte trailing _pad.
        assert_eq!(mem::size_of::<ArvxGpuInstance>(), 112);
    }

    #[test]
    fn instance_overlay_offset_at_offset_88() {
        let i = ArvxGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.overlay_offset as *const _ as usize;
        assert_eq!(field - base, 88);
        let field2 = &i.overlay_count as *const _ as usize;
        assert_eq!(field2 - base, 92);
    }

    #[test]
    fn instance_sculpt_fields_at_offset_96() {
        let i = ArvxGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let f_off = &i.sculpt_offset as *const _ as usize;
        let f_cnt = &i.sculpt_count as *const _ as usize;
        assert_eq!(f_off - base, 96);
        assert_eq!(f_cnt - base, 100);
    }

    #[test]
    fn instance_asset_id_at_offset_64() {
        let i = ArvxGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.asset_id as *const _ as usize;
        assert_eq!(field - base, 64);
    }

    #[test]
    fn instance_is_pod() {
        let i = ArvxGpuInstance::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&i);
        let a = ArvxGpuAsset::zeroed();
        let _bytes: &[u8] = bytemuck::bytes_of(&a);
    }
}
