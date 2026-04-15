//! RKP-Render: Gaussian splat rendering pipeline.
//!
//! Forward rasterization of surface-shell voxels into a G-buffer, followed by
//! deferred shadow/AO and PBR shading. Post-processing (tone mapping, bloom,
//! etc.) is handled by the caller (RkpEngine) using rkf-render passes.

/// Direct mesh-to-opacity voxelization — bypasses SDF for smooth splat fields.
pub mod voxelize_opacity;
/// GPU octree buffer management and GpuObject field reinterpretation.
pub mod octree_gpu;
/// Octree-accelerated compute ray marcher — primary visibility pass.
pub mod octree_march;
pub mod rkp_shadow_trace;
/// GPU timestamp profiler for per-pass timing.
pub mod gpu_profiler;
/// Per-object GPU struct — forward world transform, octree params, no inverse_world.
pub mod rkp_gpu_object;
/// Scene GPU buffer management — single upload path for all data.
pub mod rkp_scene;
/// Screen-space ambient occlusion compute pass — half-res.
pub mod rkp_ssao;
/// Deferred PBR shading compute pass.
pub mod rkp_shade;
/// Atmosphere LUT computation — transmittance and multi-scattering.
pub mod rkp_atmosphere;
/// Screen-space god rays — radial blur from sun position.
pub mod rkp_god_rays;
/// Volumetric rendering — fog, dust, procedural clouds.
pub mod rkp_volumetric;
/// Infinite world-space grid overlay (isolation-mode build viewport).
pub mod rkp_grid;
/// Frame renderer — orchestrates the full pipeline.
pub mod rkp_renderer;
/// Scene management — voxel pool, octree, face emission, asset loading.
pub mod rkp_scene_manager;
/// Per-viewport render targets and post-process state.
pub mod viewport_renderer;

pub use voxelize_opacity::import_mesh_to_opacity_rkp;
pub use octree_gpu::OctreeGpu;
pub use rkp_scene_manager::{AssetHandle, AssetInfo, RkpSceneManager};
pub use viewport_renderer::ViewportRenderer;

/// What a viewport's render pipeline should look like.
///
/// `InSitu` is the full deferred PBR stack with atmosphere, clouds,
/// volumetrics, god rays, shadows, and bloom — same look as the main
/// edit viewport.
///
/// `Isolation` strips the scene context: neutral gray sky, no clouds /
/// volumetrics / god rays / atmosphere, no sun shadow (SSAO carries
/// grounding), no bloom. An infinite world-space grid composites over
/// the result. Used by the build viewport for clean preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    InSitu,
    Isolation,
}

/// Validate WGSL source with naga at startup. Panics with a clear error message
/// on shader bugs instead of producing cryptic "pipeline invalid" GPU errors.
pub fn validate_wgsl(source: &str, label: &str) {
    match naga::front::wgsl::parse_str(source) {
        Ok(module) => {
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            if let Err(e) = validator.validate(&module) {
                eprintln!("[{label}] WGSL validation error: {e}");
            }
        }
        Err(e) => {
            let msg = e.emit_to_string(source);
            eprintln!("[{label}] WGSL parse error:\n{msg}");
        }
    }
}
