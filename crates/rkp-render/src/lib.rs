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
/// Glass composite post-pass — Fresnel/Beer/refraction over the
/// shaded HDR for any pixel whose primary ray passed through
/// transparent voxels.
pub mod rkp_glass;
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
/// Procedural CSG raymarch — live preview pass for the build viewport.
pub mod proc_raymarch;
/// Selected-primitive outline overlay for the raymarch preview.
pub mod proc_outline;
/// Ghost-cutter overlay for Subtract/Intersect preview visualization.
pub mod proc_ghost;
/// GPU evaluator — "sample N positions" compute pipeline shared with the
/// (Phase 3+) voxel bake path.
pub mod proc_sample;
/// Composes user-authored WGSL hooks into the procedural evaluator.
pub mod shader_composer;
/// Option B — voxel sprite instancing. Parses per-instance state structs
/// declared by `@instance_proto` and computes their byte layout.
pub mod instance_proto;
/// Phase C — GPU runtime geometry pass. User-shader `generate` hooks
/// run here, materializing voxels in a transient pool the march/shade
/// passes already know how to read.
pub mod user_shader_pass;
/// Option B — prototype bake pipeline. Voxelizes each instance shader's
/// `proto_sample_at(uvw)` into a small dedicated octree+brick+leaf-attr
/// triple, cached by source hash. Shares pool buffers with
/// `user_shader_pass` at a disjoint byte range.
pub mod user_shader_proto_pass;
/// Option B — per-region instance scatter pipeline. Runs each instance
/// shader's `emit` hook over a 3D sample grid at brick-parent
/// granularity, atomic-appending placed instances into a per-region
/// slice of a global instance pool.
pub mod user_shader_emit_pass;
/// Phase 6 — user-shader tile-cull AABB compute pass. Per filled
/// instance slot in `instance_pool`, dispatches the user shader's
/// `inst_aabb` hook to build a world-space AABB scratch buffer the
/// downstream tile-cull (count + prefix + scatter) consumes.
pub mod user_shader_tile_cull_pass;
/// Phase 6 — count phase of tile-cull. Per AABB scratch entry, projects
/// to screen tiles and atomically increments per-tile counts.
pub mod user_shader_tile_count_pass;
/// Phase 6 — prefix-sum phase of tile-cull. Single-WG blocked scan
/// turning per-tile counts into per-tile entry offsets (V1 cap 65536
/// tiles per dispatch).
pub mod user_shader_tile_prefix_pass;
/// Phase 6 — scatter phase of tile-cull. Per AABB scratch entry, writes
/// 16-byte UserShaderTileEntry records into us_tile_entries[] using
/// per-tile atomic cursors initialized from the prefix-summed offsets.
pub mod user_shader_tile_scatter_pass;
/// Phase 7 — TLAS over instance AABBs for shadow rays (and future
/// reflections / AO / GI). Session 1 ships only the wire format +
/// buffer storage; Sessions 2-4 add the CPU builder, GPU upload,
/// and WGSL traversal.
pub mod tlas_pass;
/// Skeletal skin-deform scatter pass — per-frame bone-field writer.
pub mod skin_deform;
/// CPU-side paint writes against the scene's LeafAttrPool (material +
/// per-voxel color mutations). Used by the editor's paint tool; the
/// shader reads the same `color_pool_data` / `leaf_attr_pool` buffers
/// that all other passes already consume.
pub mod paint;

pub use octree_gpu::OctreeGpu;
pub use rkp_scene_manager::{AssetHandle, AssetInfo, ReloadResult, RkpSceneManager, SkinBrick, SkinningAssetData};
pub use viewport_renderer::ViewportRenderer;
pub use skin_deform::{SkinBatchScratch, SkinBrickEntry, SkinDeformPass, SkinDispatch, SkinUniforms};

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

/// What primary-visibility pass runs for the build viewport. Orthogonal
/// to `RenderMode` (Isolation/InSitu) — lighting look is separate from
/// "what geometry are we showing."
///
/// * `Voxel` — the usual octree ray march, same path every other
///   viewport uses. Shows whatever's baked into the voxel pool; may be
///   stale relative to the current procedural tree.
/// * `Raymarch` — the procedural CSG raymarcher. Evaluates the tree
///   analytically per pixel, so edits are live — no bake required.
///   Cheap (microseconds per frame for small trees) because there's no
///   voxelization and no brick bookkeeping.
///
/// The main viewport and play mode are always `Voxel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPreviewMode {
    Voxel,
    Raymarch,
}

impl Default for BuildPreviewMode {
    fn default() -> Self {
        Self::Voxel
    }
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
