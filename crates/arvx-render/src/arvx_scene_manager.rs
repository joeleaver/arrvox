//! Scene management for Arrvox — owns the leaf_attr pool, octrees, and
//! face instances.
//!
//! This is the CPU-side scene representation. It manages the LeafAttrPool
//! (material + normal + color per leaf), the OctreeGpu allocator, and the
//! face instance list (legacy, unused by the active pipeline).
//!
//! No wgpu types, no GPU buffers here — ArvxRenderer consumes the snapshot.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — public data types + private `AssetEntry` /
//!   `AssetCache` machinery + `emit_faces` helper.
//! - [`manager`] — `ArvxSceneManager` struct + core impl
//!   (construction, faces, geometry epoch, slices, deallocation).
//! - [`asset_load`] — `impl ArvxSceneManager` block: asset lifecycle
//!   (`acquire_asset`, `reload_asset`, `release_asset`,
//!   `load_asset_from_disk`, `skinning_data`).
//! - [`paint`] — `impl ArvxSceneManager` block: paint epoch + brush
//!   overlay + `apply_paint_sphere`.
//! - [`voxelize`] — `impl ArvxSceneManager` block: `voxelize_primitive`
//!   + `voxelize_sdf_fn` + `integrate_artifact` + `deallocate_geometry`.

mod asset_load;
mod cluster_spatial_index;
mod manager;
mod paint;
mod sculpt;
mod terrain_integrate;
mod types;
mod voxelize;

// Public re-exports — keep `arvx_render::arvx_scene_manager::Foo` stable.
pub use manager::{ms_since_process_ns, ArvxSceneManager, WalkSnapshot};
pub use sculpt::SculptApplyResult;
pub use types::{
    AssetHandle, AssetInfo, FaceInstance, ReloadResult, SkinningAssetData,
    VoxelizeResult,
};
