//! Per-region GPU uniform + band-cell wire format + builder fn.
//!
//! `RegionUniform` is the std430 storage layout the BFS reads as
//! `array<RegionUniform>` (group 1, binding 0). It carries every
//! per-region input the BFS classifier needs: tile AABB, cell size,
//! shader / material ids, host-octree handles, pool block extents
//! handed out by the cache, plus the V1.1 anchor projection inputs
//! (`host_surface_y`, `painted_world_min/max`).
//!
//! `GpuBandCell` is the 16 B Phase B-redux band-cell payload: the BFS
//! bake writes one per max-depth cell in the band around painted host
//! leaves; the host march reads it via `read_band_cell`.
//!
//! `build_region_uniform` is the lone builder — takes a
//! [`super::cache::ShaderRegionRequest`] + [`super::cache::CachedSlot`]
//! + resolved shader_id + frame time and produces a populated
//! `RegionUniform`.

use super::cache::{CachedSlot, ShaderRegionRequest};

/// Per-region uniform — laid out to match WGSL's std430 storage layout
/// for `array<RegionUniform>`.
///
/// 240 bytes. Carries per-region pool block offsets/sizes (allocator
/// output) so each region's allocator atomicAdd composes a global
/// pool offset as `block_offset + atomic_slot`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionUniform {
    pub aabb_min: [f32; 3],                // offset  0
    pub cell_size: f32,                     // offset 12
    pub aabb_max: [f32; 3],                 // offset 16
    pub shader_id: u32,                     // offset 28
    pub max_depth: u32,                     // offset 32
    pub time: f32,                          // offset 36
    pub material_id: u32,                   // offset 40
    pub region_thickness: f32,              // offset 44
    pub host_octree_root: u32,              // offset 48
    pub host_octree_depth: u32,             // offset 52
    pub host_octree_extent: f32,            // offset 56
    /// Per-region pool block offsets + sizes. Offsets are absolute
    /// GPU-buffer indices; sizes are the bucket-rounded extents the
    /// allocator handed out. Units: octree_block in vec2<u32>;
    /// brick_block in BRICKS; leaf_attr_block in LeafAttr;
    /// fill_task_block in BrickFillTask.
    pub octree_block_offset: u32,           // offset 60
    pub octree_block_size: u32,             // offset 64
    pub brick_block_offset: u32,            // offset 68
    pub brick_block_size: u32,              // offset 72
    pub leaf_attr_block_offset: u32,        // offset 76
    pub leaf_attr_block_size: u32,          // offset 80
    pub fill_task_block_offset: u32,        // offset 84
    pub fill_task_block_size: u32,          // offset 88
    /// Phase B-redux 3b — `1` when the BFS should bake band cells
    /// (instance_at path) instead of voxel bricks; `0` for the
    /// existing voxel emit path. Repurposed from `_pad_host` (the
    /// offset still aligns `host_grid_origin` to 96).
    pub use_band_path: u32,                 // offset 92
    pub host_grid_origin: [f32; 3],         // offset 96
    /// Phase B-redux V1.1 — world-space y of the painted surface,
    /// used by the band-cell BFS as the anchor projection target.
    /// Replaces V1's `host.distance` projection (unreliable on hosts
    /// with finely-subdivided host octrees that don't fold large
    /// empty subtrees). Flat-surface only; sloped/curved surfaces
    /// need a more expressive scheme (per-cell normal projection or
    /// multi-source BFS). Repurposed from `_pad_grid` so the struct
    /// stays 208 B.
    pub host_surface_y: f32,                // offset 108
    pub params: [[f32; 4]; 2],              // offset 112
    pub host_inverse_world: [[f32; 4]; 4],  // offset 144
    /// Phase B-redux V1.1 — world-space AABB of the PAINTED leaves
    /// (not the band region's outer tile cube). The BFS uses this on
    /// the band path to reject cells whose x/z fall outside
    /// `[painted_min - band, painted_max + band]`. Without this gate
    /// cells fill the entire tile cube horizontally.
    pub painted_world_min: [f32; 3],        // offset 208
    pub _pad_painted_min: f32,              // offset 220
    pub painted_world_max: [f32; 3],        // offset 224
    pub _pad_painted_max: f32,              // offset 236
}

const _: () = assert!(std::mem::size_of::<RegionUniform>() == 240);

// ============================================================
// Phase B-redux 3b — band-cell wire format
// ============================================================
//
// Bake (3b.2) emits one `GpuBandCell` per max-depth cell in the band
// around painted host leaves. The cell's octree node carries
// `OCTREE_LEAF_BIT | OCTREE_BAND_BIT | payload_offset`, where
// `payload_offset` is the cell's index in the global `band_cell_pool`.
//
// March (3b.3) detects `OCTREE_BAND_BIT` during per-object DDA, reads
// the payload, looks up `band_regions[region_index]` for the shader/
// material context, and fires `dispatch_user_instance_descend`
// seeded with `anchor_world_pos`.
//
// V1 single-anchor: each band cell points to its single nearest
// painted host leaf's world center. V2 multi-anchor (up to 4) is a
// future revision that uses `_pad0..2` slots in `GpuBandCell` and
// extends the descent loop in march.

/// Per-band-cell payload, 16 B. The bake (`user_shader_geom.wgsl`)
/// packs this across two consecutive `leaf_attr_pool` slots; the
/// march (`octree_march.wgsl::read_band_cell`) reads it back.
///
/// V1 carries `material_id` directly per cell — the painted host
/// material that drove this region's bake. shader_id flows through
/// `materials[material_id].shader_id`. Self-contained so the march
/// doesn't need a per-region metadata table.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBandCell {
    pub anchor_world_pos: [f32; 3],
    pub material_id: u32,
}

const _: () = assert!(std::mem::size_of::<GpuBandCell>() == 16);

/// Build the per-region uniform from a request + region slot.
pub fn build_region_uniform(
    request: &ShaderRegionRequest,
    slot: &CachedSlot,
    shader_id: u32,
    time_seconds: f32,
) -> RegionUniform {
    let mut params = [[0.0f32; 4]; 2];
    for (i, &v) in request.params.iter().take(8).enumerate() {
        params[i / 4][i % 4] = v;
    }
    RegionUniform {
        aabb_min: request.aabb_min,
        cell_size: request.cell_size,
        aabb_max: request.aabb_max,
        shader_id,
        max_depth: slot.max_depth,
        time: time_seconds,
        material_id: request.material_id,
        region_thickness: request.region_thickness,
        host_octree_root: request.host_octree_root,
        host_octree_depth: request.host_octree_depth,
        host_octree_extent: request.host_octree_extent,
        octree_block_offset: slot.octree_block_offset,
        octree_block_size: slot.octree_block_size,
        brick_block_offset: slot.brick_block_offset,
        brick_block_size: slot.brick_block_size,
        leaf_attr_block_offset: slot.leaf_attr_block_offset,
        leaf_attr_block_size: slot.leaf_attr_block_size,
        fill_task_block_offset: slot.fill_task_block_offset,
        fill_task_block_size: slot.fill_task_block_size,
        use_band_path: u32::from(request.is_band_region),
        host_grid_origin: request.host_grid_origin,
        host_surface_y: request.host_surface_y,
        params,
        host_inverse_world: request.host_inverse_world,
        painted_world_min: request.painted_world_min,
        _pad_painted_min: 0.0,
        painted_world_max: request.painted_world_max,
        _pad_painted_max: 0.0,
    }
}

#[cfg(test)]
#[path = "region_tests.rs"]
mod tests;
