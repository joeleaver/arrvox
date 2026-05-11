//! RKP-Render: Gaussian splat rendering pipeline.
//!
//! Forward rasterization of surface-shell voxels into a G-buffer, followed by
//! deferred shadow/AO and PBR shading. Post-processing (tone mapping, bloom,
//! wireframe overlay) is handled by passes in this crate.

/// `0xFFFFFFFFu`-as-sentinel constants shared with WGSL.
pub mod sentinels;
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
/// Surface-splat rasterization prototype — Phase A walks the octree
/// to produce a per-voxel vertex buffer; Phase B will dispatch oriented
/// disc splats as an alternative primary visibility path.
pub mod splat_pass;
/// Surface-mesh path (Phase 1 of the splat-to-mesh pivot) — naive
/// surface-nets extraction at asset load. CPU-only; the Phase 2 forward
/// triangle pipeline reads the resulting `(vertices, indices)` buffer.
pub mod mesh_pass;
/// Mesh-rendered directional shadow map (Phase 3 of the splat-to-mesh
/// pivot). Renders mesh triangles from the light's POV, writes depth
/// into the existing shadow_buffer storage so shade samples it via
/// the unchanged `sample_shadow_map` path.
pub mod mesh_shadow_map_pass;
/// Phase 6.2 LOD-select compute pass — applies the Karis-Nanite admit
/// rule per cluster and writes a `DrawIndexedIndirectArgs` table that
/// `multi_draw_indexed_indirect` consumes in `dispatch_mesh`.
pub mod mesh_lod_select_pass;
/// Mesh-mode glass pipeline (front + back raster + combine compute).
/// Produces the same `gbuf_glass` packing as the march, so the
/// existing `rkp_glass` composite runs unchanged in mesh-mode.
pub mod mesh_glass_pass;
/// Mesh-mode glass shadow pipelines — per-cascade front/back depth
/// captures so the shade pass can apply Beer attenuation on the
/// existing CSM shadow factor.
pub mod mesh_glass_shadow_pass;
/// Splat-rasterizer compute fixup pass — reads the visibility-buffer
/// triplet (leaf_slot, pick) `splat_pass` writes and fills in the
/// remaining G-buffer entries (normal / material / glass) via the
/// scene's `leaf_attr_pool` / `color_pool` / `instances` indirection.
pub mod splat_resolve_pass;
/// Procedural proxy-mesh raster pipeline. First-class triangle-mesh
/// renderer for procedurals baked via GPU surface-nets-from-SDF.
/// Bypasses `LeafAttr` indirection — each `ProxyVertex` carries its
/// own normal + material + color and the FS writes the full G-buffer
/// directly. See `notes/proxy-mesh-first-class.md`.
pub mod mesh_proxy_pass;
pub mod rkp_shadow_trace;
/// Per-object GPU struct — forward world transform, octree params, no inverse_world.
pub mod rkp_gpu_object;
/// Scene GPU buffer management — single upload path for all data.
pub mod rkp_scene;
/// Screen-space ambient occlusion compute pass — half-res.
pub mod rkp_ssao;
/// Deferred PBR shading compute pass.
pub mod rkp_shade;
/// Brush-state probe — single-thread compute that captures the
/// G-buffer at the cursor pixel for the screen-space paint cursor.
pub mod brush_state_pass;
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
/// GPU surface-nets-from-SDF spike — proxy-mesh path for procedurals
/// without going through the octree + brick + DAG + meshlet bake.
pub mod proc_surface_nets;
/// Composes user-authored WGSL hooks into the procedural evaluator.
pub mod shader_composer;
/// V1 mesh-path user-shader pipeline. Vertex-shader-driven path:
/// spawn_count → prefix_sum → fill → indirect draw against own raster.
/// See `notes/user-shaders-mesh.md`.
pub mod user_shader_mesh_pass;
/// Phase 7 — TLAS over instance AABBs for shadow rays (and future
/// reflections / AO / GI). Session 1 ships only the wire format +
/// buffer storage; Sessions 2-4 add the CPU builder, GPU upload,
/// and WGSL traversal.
pub mod tlas_pass;
/// Phase 7c — GPU-built TLAS pipeline. Session 1 ships the
/// primitive-assembly compute passes (`tlas_assemble_*.wgsl`) and
/// the unified `tlas_prims` output buffer; Sessions 2-4 add Morton
/// sort, Karras tree, and AABB propagation; Session 5 cuts over
/// from the CPU `tlas_pass::build_tlas` builder.
pub mod tlas_build_pass;
/// Phase 8 — directional shadow maps. Replaces the per-pixel
/// ray-traced directional shadow path with a single light-POV
/// depth render + per-pixel sample. Sessions 1-5 ship in order:
/// light camera (S1), shadow march (S2), shade-side query (S3),
/// engine wiring (S4), directional cutover (S5).
pub mod shadow_map_pass;
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

/// Validate WGSL source with naga at startup.
///
/// Fail-fast on parse / validation errors so shader bugs surface
/// here with attribution back to the labelled shader, instead of
/// downstream as opaque "pipeline invalid" wgpu errors. Panics by
/// design — every caller is on the renderer's startup path; a bad
/// shader at this layer is unrecoverable.
///
/// Returns the validated `naga::Module` so callers that need it can
/// reuse the parse without re-parsing.
pub fn validate_wgsl(source: &str, label: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        let msg = e.emit_to_string(source);
        panic!("[{label}] WGSL parse error:\n{msg}");
    });
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("[{label}] WGSL validation error: {e}"));
    module
}

/// Validate `source` and create a labelled wgpu shader module from
/// it in one step. The recommended path for every pipeline-creation
/// site — using it makes the "validate before create" contract
/// impossible to forget.
///
/// `wesl::include_wesl!(...)` is a compile-time macro so each caller
/// still expands the include in its own scope; this helper just
/// folds the validate + module-create boilerplate.
pub fn compile_pass_shader(
    device: &wgpu::Device,
    source: &str,
    label: &str,
) -> wgpu::ShaderModule {
    let _ = validate_wgsl(source, label);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
