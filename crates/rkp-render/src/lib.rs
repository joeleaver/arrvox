//! RKP-Render: Gaussian splat rendering pipeline.
//!
//! Forward rasterization of surface-shell voxels into a G-buffer, followed by
//! deferred shadow/AO and PBR shading. Post-processing (tone mapping, bloom,
//! etc.) is handled by the caller (RkpEngine) using rkf-render passes.

/// Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
pub mod voxelize_opacity;
/// GPU octree buffer management and GpuObject field reinterpretation.
pub mod octree_gpu;
/// Emit compute pass — traverses octrees, emits transition face quads.
pub mod splat_emit;
/// Rasterization render pipeline — draws face quads into G-buffer via MRT.
pub mod splat_raster;
/// Per-object GPU struct — forward world transform, octree params, no inverse_world.
pub mod rkp_gpu_object;
/// Scene GPU buffer management — single upload path for all data.
pub mod rkp_scene;
/// Shadow + AO compute pass — half-res octree tracing.
pub mod rkp_shadow_ao;
/// Deferred PBR shading compute pass.
pub mod rkp_shade;
/// Frame renderer — orchestrates the full pipeline.
pub mod rkp_renderer;
/// Scene management — voxel pool, octree, face emission, asset loading.
pub mod rkp_scene_manager;

pub use voxelize_opacity::import_mesh_to_opacity_rkf;
pub use voxelize_opacity::import_mesh_to_opacity_rkp;
pub use octree_gpu::OctreeGpu;
pub use splat_emit::SplatEmitPass;
pub use rkp_scene_manager::RkpSceneManager;
