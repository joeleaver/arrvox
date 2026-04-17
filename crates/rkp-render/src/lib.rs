//! RKP-Render: Gaussian splat rendering pipeline.
//!
//! Forward rasterization of surface-shell voxels into a G-buffer, followed by
//! deferred shadow/AO and PBR shading. Post-processing (tone mapping, bloom,
//! wireframe overlay) is handled by passes in this crate.

/// GPU device/queue wrapper.
pub mod context;
/// G-buffer textures for deferred shading.
pub mod gbuffer;
/// Bloom compute pass (pre-upscale).
pub mod bloom;
/// Bloom composite compute pass (post-upscale).
pub mod bloom_composite;
/// Tone mapping compute pass (HDR → LDR).
pub mod tone_map;
/// Wireframe line rendering pass.
pub mod wireframe;

pub use context::RenderContext;
pub use gbuffer::GBuffer;
pub use bloom::{BloomPass, BloomParams, BLOOM_MIP_LEVELS, DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_KNEE};
pub use bloom_composite::{BloomCompositePass, BloomCompositeParams, DEFAULT_BLOOM_INTENSITY};
pub use tone_map::{ToneMapPass, ToneMapMode, ToneMapParams, DEFAULT_EXPOSURE, LDR_FORMAT};
pub use wireframe::{WireframePass, LineVertex};

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
/// Frame renderer — orchestrates the full pipeline.
pub mod rkp_renderer;
/// Scene management — voxel pool, octree, face emission, asset loading.
pub mod rkp_scene_manager;
/// Skeletal skin-deform scatter pass — per-frame bone-field writer.
pub mod skin_deform;

pub use octree_gpu::OctreeGpu;
pub use rkp_scene_manager::{AssetHandle, AssetInfo, ReloadResult, RkpSceneManager, SkinBrick, SkinningAssetData};
pub use skin_deform::{SkinBatchScratch, SkinBrickEntry, SkinDeformPass, SkinDispatch, SkinUniforms};

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
