//! Scene management for RKIPatch — owns the leaf_attr pool, octrees, and
//! face instances.
//!
//! This is the CPU-side scene representation. It manages the LeafAttrPool
//! (material + normal + color per leaf), the OctreeGpu allocator, and the
//! face instance list (legacy, unused by the active pipeline).
//!
//! No wgpu types, no GPU buffers here — RkpRenderer consumes the snapshot.
//!
//! ## Module layout (post-split)
//!
//! - [`types`] — public data types + private `AssetEntry` /
//!   `AssetCache` machinery + `emit_faces` helper.
//! - [`manager`] — `RkpSceneManager` struct + core impl
//!   (construction, faces, geometry epoch, slices, deallocation).
//! - [`asset_load`] — `impl RkpSceneManager` block: asset lifecycle
//!   (`acquire_asset`, `reload_asset`, `release_asset`,
//!   `load_asset_from_disk`, `skinning_data`).
//! - [`paint`] — `impl RkpSceneManager` block: paint epoch + brush
//!   overlay + `apply_paint_sphere`.
//! - [`voxelize`] — `impl RkpSceneManager` block: `voxelize_primitive`
//!   + `voxelize_sdf_fn` + `integrate_artifact` + `deallocate_geometry`.

mod asset_load;
mod manager;
mod paint;
mod types;
mod voxelize;

// Public re-exports — keep `rkp_render::rkp_scene_manager::Foo` stable.
pub use manager::RkpSceneManager;
pub use types::{
    AssetHandle, AssetInfo, FaceInstance, ReloadResult, SkinBrick, SkinningAssetData,
    VoxelizeResult,
};
