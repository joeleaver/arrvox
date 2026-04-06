//! RKP-Render: Splat rendering pipeline.
//!
//! Replaces rkf-render's ray march with a fixed-step march through the opacity
//! field. All other passes (shading, shadows, GI, post-process) are reused from
//! rkf-render via direct dependency.
//!
//! The only new pass is [`SplatMarchPass`] — everything else is orchestration.

/// Splat march compute pass — surface-finding through opacity field, G-buffer output.
pub mod splat_march;
/// Opacity volume manager — procedural geometry volumes for opacity shaders.
pub mod opacity_volume;
/// Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
pub mod voxelize_opacity;
/// GPU octree buffer management and GpuObject field reinterpretation.
pub mod octree_gpu;
/// GPU surface shell buffer — per-brick occupancy bitmasks.
pub mod surface_shell_gpu;
/// Emit compute pass — traverses octrees, emits transition face quads.
pub mod splat_emit;
/// Rasterization render pipeline — draws face quads into G-buffer via MRT.
pub mod splat_raster;
/// SplatRasterPass — MarchPass implementation using forward rasterization.
pub mod splat_raster_pass;

pub use splat_march::SplatMarchPass;
pub use splat_raster_pass::SplatRasterPass;
pub use opacity_volume::{OpacityVolume, OpacityVolumeManager};
pub use voxelize_opacity::import_mesh_to_opacity_rkf;
pub use voxelize_opacity::import_mesh_to_opacity_rkp;
pub use octree_gpu::OctreeGpu;
pub use surface_shell_gpu::SurfaceShellGpu;
pub use splat_emit::SplatEmitPass;
