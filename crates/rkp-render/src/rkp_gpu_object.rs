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
/// # Layout (64 bytes)
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
/// | 44     | 4    | _pad0 |
/// | 48     | 12   | grid_origin (vec3<f32>) |
/// | 60     | 4    | _pad1 |
///
/// **Skinning template fields** (`bone_count`, `rest_octree_*`,
/// `bone_field_dim_*`, `bone_field_origin_*`) live on `RkpGpuInstance`
/// in Phase 1, not here. Reasoning:
/// - `bone_field_dim_*` / `bone_field_origin_*` are per-frame deformed-
///   pose data — different per instance.
/// - `bone_count` and `rest_octree_*` ARE per-asset in concept, but the
///   current CPU data flow (`scene_sync.rs::build_gpu_object`) only
///   populates them when the per-frame `skinning` binding is `Some(_)`.
///   If two instances of the same asset have different skinning states
///   this frame (plan succeeded vs failed), the dedupe by `octree_root`
///   takes the FIRST instance's values — which can be zero, breaking
///   the second instance's skinned march. Phase 1b sources them from
///   the asset cache directly and moves them back to the asset.
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
    pub _pad0: u32,
    pub grid_origin: [f32; 3],
    pub _pad1: u32,
}

/// Per-instance GPU record (208 bytes — Phase 2 will drop `inverse_world`
/// to land at ~144 B; Phase 1b moves skinning template fields to the
/// asset, dropping ~16 B more). One per scene entity; carries:
/// - world transform (and precomputed inverse)
/// - asset reference (`asset_id` indexes into the assets table)
/// - per-entity overrides (material, picking id, layer mask)
/// - per-frame skinning state (template + runtime offsets, see asset doc)
///
/// # Layout (208 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | world (mat4x4<f32>) |
/// | 64     | 64   | inverse_world (mat4x4<f32>) — Phase 2: drop |
/// | 128    | 4    | asset_id (u32) |
/// | 132    | 4    | material_id (u32) |
/// | 136    | 4    | object_id (u32) |
/// | 140    | 4    | layer_mask (u32) |
/// | 144    | 4    | is_skinned (u32) |
/// | 148    | 4    | bone_count (u32) — Phase 1b: move to asset |
/// | 152    | 4    | bone_buffer_offset (u32) |
/// | 156    | 4    | rest_octree_root (u32) — Phase 1b: move to asset |
/// | 160    | 4    | rest_octree_depth (u32) — Phase 1b: move to asset |
/// | 164    | 4    | rest_octree_extent_bits (u32) — Phase 1b: move to asset |
/// | 168    | 4    | bone_field_offset (u32) |
/// | 172    | 4    | bone_field_occ_offset (u32) |
/// | 176    | 4    | bone_field_dim_x (u32) |
/// | 180    | 4    | bone_field_dim_y (u32) |
/// | 184    | 4    | bone_field_dim_z (u32) |
/// | 188    | 4    | bone_field_origin_x (f32) |
/// | 192    | 4    | bone_field_origin_y (f32) |
/// | 196    | 4    | bone_field_origin_z (f32) |
/// | 200    | 8    | _pad (mat4x4 alignment tail) |
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct RkpGpuInstance {
    pub world: [[f32; 4]; 4],
    pub inverse_world: [[f32; 4]; 4],
    pub asset_id: u32,
    pub material_id: u32,
    pub object_id: u32,
    pub layer_mask: u32,
    pub is_skinned: u32,
    pub bone_count: u32,
    pub bone_buffer_offset: u32,
    pub rest_octree_root: u32,
    pub rest_octree_depth: u32,
    pub rest_octree_extent_bits: u32,
    pub bone_field_offset: u32,
    pub bone_field_occ_offset: u32,
    pub bone_field_dim_x: u32,
    pub bone_field_dim_y: u32,
    pub bone_field_dim_z: u32,
    pub bone_field_origin_x: f32,
    pub bone_field_origin_y: f32,
    pub bone_field_origin_z: f32,
    /// WGSL stride must be multiple of struct alignment (16 from mat4x4).
    /// Without this pad, Rust struct = 200 B, WGSL stride = 208 B, the
    /// array indexing reads from the wrong offset.
    pub _pad: [u32; 2],
}

/// Walk a slice of legacy [`RkpGpuObject`]s and split into deduplicated
/// per-asset + per-instance records.
///
/// Dedupe key is `octree_root`. Each unique value gets one [`RkpGpuAsset`]
/// slot; every input object becomes one [`RkpGpuInstance`] with its
/// `asset_id` set to the slot index.
///
/// `octree_root == 0` is a legitimate value — the octree allocator hands
/// back offset 0 for the first allocation, so the first cached asset
/// loaded in a session has root 0. We dedupe on it like any other key.
/// The march's per-asset `geom_type == 0` check still gates "no
/// geometry" entries; the upstream filter (`scene_gpu.rs` walks only
/// `r.spatial.is_some()` entities) keeps zeroed objects out of the
/// input in practice.
pub fn split_objects(
    objects: &[RkpGpuObject],
) -> (Vec<RkpGpuAsset>, Vec<RkpGpuInstance>) {
    let mut assets: Vec<RkpGpuAsset> = Vec::new();
    let mut instances: Vec<RkpGpuInstance> = Vec::with_capacity(objects.len());
    let mut by_root: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for obj in objects {
        let asset_id = match by_root.get(&obj.octree_root) {
            Some(&id) => id,
            None => {
                let id = assets.len() as u32;
                assets.push(RkpGpuAsset {
                    aabb_min: obj.aabb_min,
                    octree_root: obj.octree_root,
                    aabb_max: obj.aabb_max,
                    octree_depth: obj.octree_depth,
                    octree_extent_bits: obj.octree_extent_bits,
                    voxel_size: obj.voxel_size,
                    geom_type: obj.geom_type,
                    _pad0: 0,
                    grid_origin: obj.grid_origin,
                    _pad1: 0,
                });
                by_root.insert(obj.octree_root, id);
                id
            }
        };
        instances.push(RkpGpuInstance {
            world: obj.world,
            inverse_world: obj.inverse_world,
            asset_id,
            material_id: obj.material_id,
            object_id: obj.object_id,
            layer_mask: obj.layer_mask,
            is_skinned: obj.is_skinned,
            bone_count: obj.bone_count,
            bone_buffer_offset: obj.bone_buffer_offset,
            rest_octree_root: obj.rest_octree_root,
            rest_octree_depth: obj.rest_octree_depth,
            rest_octree_extent_bits: obj.rest_octree_extent_bits,
            bone_field_offset: obj.bone_field_offset,
            bone_field_occ_offset: obj.bone_field_occ_offset,
            bone_field_dim_x: obj.bone_field_dim_x,
            bone_field_dim_y: obj.bone_field_dim_y,
            bone_field_dim_z: obj.bone_field_dim_z,
            bone_field_origin_x: obj.bone_field_origin_x,
            bone_field_origin_y: obj.bone_field_origin_y,
            bone_field_origin_z: obj.bone_field_origin_z,
            _pad: [0; 2],
        });
    }

    (assets, instances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn size_is_256_bytes() {
        // Phase 1a: legacy CPU-side construction view. WGSL no longer
        // has an `RkpObject` struct; `RkpScene::upload_frame` splits
        // this into `RkpGpuAsset` + `RkpGpuInstance` before upload.
        // Phase 1b retires this struct entirely.
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

    #[test]
    fn asset_size_is_64_bytes() {
        // Phase 1: skinning template fields live on the instance (see
        // doc-comment on RkpGpuAsset). Phase 1b moves them back here
        // sourced from the asset cache directly.
        assert_eq!(mem::size_of::<RkpGpuAsset>(), 64);
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
    fn instance_size_is_208_bytes() {
        // Includes 8-byte tail pad to make stride a multiple of mat4x4
        // alignment (16). Phase 1b moves skinning template fields to the
        // asset (-16 B); Phase 2 drops inverse_world (-64 B).
        assert_eq!(mem::size_of::<RkpGpuInstance>(), 208);
    }

    #[test]
    fn instance_bone_field_dim_at_offset_176() {
        let i = RkpGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.bone_field_dim_x as *const _ as usize;
        assert_eq!(field - base, 176);
    }

    #[test]
    fn instance_asset_id_at_offset_128() {
        let i = RkpGpuInstance::zeroed();
        let base = &i as *const _ as usize;
        let field = &i.asset_id as *const _ as usize;
        assert_eq!(field - base, 128);
    }

    #[test]
    fn split_dedupes_by_octree_root() {
        let mut a = RkpGpuObject::zeroed();
        a.octree_root = 100;
        a.material_id = 1;
        a.object_id = 1;
        let mut b = RkpGpuObject::zeroed();
        b.octree_root = 100;
        b.material_id = 2;
        b.object_id = 2;
        let mut c = RkpGpuObject::zeroed();
        c.octree_root = 200;
        c.material_id = 3;
        c.object_id = 3;

        let (assets, instances) = split_objects(&[a, b, c]);
        // Two unique roots → two assets, in encounter order.
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0].octree_root, 100);
        assert_eq!(assets[1].octree_root, 200);

        assert_eq!(instances.len(), 3);
        assert_eq!(instances[0].asset_id, 0);
        assert_eq!(instances[0].material_id, 1);
        assert_eq!(instances[1].asset_id, 0);
        assert_eq!(instances[1].material_id, 2);
        assert_eq!(instances[2].asset_id, 1);
    }

    #[test]
    fn split_handles_octree_root_zero() {
        // Regression: the octree allocator returns offset 0 for the
        // first allocation, so the first cached asset has root 0. An
        // earlier sentinel-at-slot-0 design ate it as "no geometry."
        let mut a = RkpGpuObject::zeroed();
        a.octree_root = 0;
        a.geom_type = geom_type::VOXELIZED;
        a.material_id = 7;

        let (assets, instances) = split_objects(&[a]);
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].octree_root, 0);
        assert_eq!(assets[0].geom_type, geom_type::VOXELIZED);
        assert_eq!(instances[0].asset_id, 0);
        assert_eq!(instances[0].material_id, 7);
    }
}
