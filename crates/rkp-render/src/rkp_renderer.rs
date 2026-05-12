//! Shared GPU state + orchestration for per-viewport rendering.
//!
//! [`RkpRenderer`] holds the scene-wide buffers (scene, atmosphere LUTs,
//! shade params, lights, materials) plus the GPU profiler. The
//! resolution-coupled passes (march, shadow_trace, ssao, shade,
//! volumetric, god_rays) live in [`ViewportRenderer`] so each viewport
//! can render at its own resolution without clashing on shared textures.
//!
//! The caller (rkp-editor) drives each frame:
//! 1. Build RkpGpuObjects from ECS/scene state.
//! 2. Call `upload_frame(objects)` — cheap, every frame.
//! 3. Call `upload_geometry(...)` — only when geometry changes.
//! 4. For each visible viewport:
//!    - `vr.upload_camera(queue, cam_uniforms)` and
//!      `vr.refresh_bindings(device, renderer)` to catch up with any
//!      shared-buffer reallocations.
//!    - `renderer.render_to(encoder, queue, vr, ...)`.

use crate::rkp_scene::{RkpScene, GeometryUpload, FrameUpload};
use crate::rkp_shade::{ShadeParams, GpuLight, GpuMaterial};
use crate::rkp_atmosphere::RkpAtmospherePass;
use crate::mesh_pass::{MeshPass, MeshVertex, MeshletCluster};
use crate::mesh_shadow_map_pass::MeshShadowMapPass;
use crate::splat_pass::{SplatDraw, SplatPass, SplatVertex};
use crate::splat_resolve_pass::SplatResolvePass;
use wgpu_profiler::GpuProfiler;

/// Primary-visibility selector. Set by the `RKP_PRIMARY` env var at
/// `RkpRenderer` construction. `Splat` swaps the compute octree-march
/// for the splat raster path; `Mesh` swaps it for the surface-mesh
/// raster path (Phase 2 of the splat-to-mesh pivot); `March` keeps the
/// existing behaviour. The selector is read once at startup so per-
/// frame env reads don't show up in profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimaryMode {
    March,
    Splat,
    Mesh,
}

impl PrimaryMode {
    /// Read `RKP_PRIMARY`. `splat` (case-insensitive) selects splat,
    /// `mesh` selects the surface-mesh path; anything else (including
    /// unset) keeps the existing march path.
    fn from_env() -> Self {
        match std::env::var("RKP_PRIMARY").as_deref() {
            Ok(s) if s.eq_ignore_ascii_case("splat") => PrimaryMode::Splat,
            Ok(s) if s.eq_ignore_ascii_case("mesh") => PrimaryMode::Mesh,
            _ => PrimaryMode::March,
        }
    }
}

/// Read a positive finite f32 from the named env var; fall back to
/// `default` if the var is unset, unparseable, or non-positive. Used
/// for the LOD `pixel_threshold` knobs so we can tune without
/// recompiling.
fn pixel_threshold_env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// The RKIPatch renderer — shared state only. Per-viewport passes live
/// in [`ViewportRenderer`].
pub struct RkpRenderer {
    /// Scene GPU buffers (shared across viewports).
    pub scene: RkpScene,
    /// Atmosphere LUTs (transmittance + multi-scattering + sky-view + AP).
    /// Computed from sun/camera, consumed by shade.
    pub atmosphere: RkpAtmospherePass,
    /// Scene-wide shade params (env-colors, ambient intensity, etc.).
    pub shade_params_buffer: wgpu::Buffer,
    /// Scene-wide light list (directional + point + spot).
    pub lights_buffer: wgpu::Buffer,
    /// Capacity tracking for `lights_buffer`. Avoids using
    /// `buffer.size()` as the grow check, which can race against
    /// pending validation when the buffer is recreated.
    lights_capacity: u64,
    /// Scene-wide material palette.
    pub materials_buffer: wgpu::Buffer,
    /// Capacity tracking for `materials_buffer`. Same rationale as
    /// `lights_capacity`.
    materials_capacity: u64,
    /// Bumps when `lights_buffer` or `materials_buffer` reallocates so
    /// ViewportRenderers know to rebuild their march + shade bindings.
    lights_materials_epoch: u64,
    /// Skeletal skin-deform scatter pass — writes the per-frame
    /// deformed-space bone field (Phase 3a). See `skin_deform.rs`.
    pub skin_deform: crate::skin_deform::SkinDeformPass,
    /// Splat-rasterizer pipeline (Phase B-2). One pipeline shared
    /// across viewports — the per-VR state lives in `ViewportRenderer`
    /// (g0 bind group, per-instance bind groups + uniform buffers).
    pub splat_pass: SplatPass,
    /// Splat-resolve compute fixup. Reads the visibility-buffer
    /// triplet `splat_pass` writes and fills in the remaining G-buffer
    /// entries (normal / material / glass). One pipeline shared across
    /// viewports.
    pub splat_resolve: SplatResolvePass,
    /// Procedural proxy-mesh raster pipeline (GPU surface-nets-from-
    /// SDF). Writes the full G-buffer for proxy pixels directly,
    /// bypassing `splat_resolve`. One pipeline shared across viewports.
    pub mesh_proxy: crate::mesh_proxy_pass::MeshProxyPass,
    /// V1 mesh-path user-shader pipeline owner. Holds bind-group +
    /// pipeline layouts shared across all mesh-path shaders plus a
    /// stub pipeline set built from the engine skeleton at startup.
    /// Per-material pipelines + buffers live on the engine's
    /// `RenderState::mesh_user_shader_cache`.
    pub user_shader_mesh: crate::user_shader_mesh_pass::UserShaderMeshPass,
    /// Brush-state probe — single-thread compute reading the gbuffer
    /// at the cursor pixel, feeding the screen-space paint cursor.
    /// One pipeline shared across viewports; per-VR bind group lives
    /// on `ViewportRenderer`.
    pub brush_state: crate::brush_state_pass::BrushStatePass,
    /// Per-asset vertex-buffer cache for the splat path. Indexed by
    /// `AssetHandle::raw()` — `splat_buffers[handle.raw() as usize]`
    /// is `Some((vbo, splat_count))` for assets whose splat data has
    /// been uploaded, `None` otherwise. Grows as new assets are
    /// loaded; entries are cleared on `release_splats_for_asset`.
    splat_buffers: Vec<Option<(wgpu::Buffer, u32)>>,
    /// Surface-mesh raster pipeline (Phase 2 of the splat-to-mesh
    /// pivot). Shares `g0_layout` / `g1_layout` with `splat_pass` so
    /// the same per-VR bind groups drive both pipelines.
    pub mesh_pass: MeshPass,
    /// Mesh-rendered directional shadow-map pipeline (Phase 3 of the
    /// pivot). Renders the same triangles from the light's POV; per-VR
    /// state (depth texture, g0 bind group) lives in `ViewportRenderer`.
    pub mesh_shadow_map: MeshShadowMapPass,
    /// Per-cluster LOD-select compute pass (Phase 6.2). Applies the
    /// Karis admit rule and writes a `DrawIndexedIndirectArgs` table
    /// the render path consumes via `multi_draw_indexed_indirect`.
    pub mesh_lod_select_pass: crate::mesh_lod_select_pass::MeshLodSelectPass,
    /// Mesh-mode glass pipeline (front + back raster + combine
    /// compute). Produces the same `gbuf_glass` Rg32Uint packing the
    /// march does, so the existing `rkp_glass` composite runs
    /// unchanged.
    pub mesh_glass: crate::mesh_glass_pass::MeshGlassPass,
    /// Mesh-mode glass shadow pipelines — per-cascade front + back
    /// depth captures so the shade pass can apply Beer attenuation
    /// on top of the existing CSM shadow factor.
    pub mesh_glass_shadow: crate::mesh_glass_shadow_pass::MeshGlassShadowPass,
    /// `RKP_MESH_GLASS_DEBUG_FORCE` snapshot (read once at startup).
    /// `1` ⇒ the combine compute spoofs a 100 mm glass shell on every
    /// opaque mesh hit, bypassing the entry-FS classify/discard path.
    /// `2` ⇒ the front/back FS opacity threshold is raised above 1.0,
    /// so every mesh fragment classifies as glass with its actual
    /// leaf-derived material id (lets us tell whether the leaf
    /// lookup itself is wrong vs the discard threshold being wrong
    /// for the user's material). Bisects "no glass visible" —
    /// see `mesh_glass.wesl::GlassFsParams`.
    mesh_glass_debug_force: u32,
    /// Per-asset vertex/index buffer cache for the mesh path. Same
    /// shape as `splat_buffers`, but each entry carries `(vbo, ibo,
    /// index_count)`. Cleared on `release_mesh_for_asset`.
    mesh_buffers: Vec<Option<(wgpu::Buffer, wgpu::Buffer, u32)>>,
    /// Per-asset meshlet cluster table on the GPU (Phase 5).
    /// `(buffer, cluster_count)`; the buffer holds a flat
    /// `[MeshletCluster]` array uploaded via `cast_slice` and is
    /// bound as STORAGE for the Phase 6 LOD-selection compute pass.
    /// Phase 5 uploads but does not yet consume — validates the
    /// upload path without touching the hot dispatch.
    mesh_cluster_buffers: Vec<Option<(wgpu::Buffer, u32)>>,
    /// Per-asset vertex/index buffer cache for the procedural
    /// proxy-mesh path. Separate from `mesh_buffers` because the
    /// proxy vertex layout is `ProxyVertex` (32 B; material + color
    /// payload) — not `MeshVertex`. Indexed by `AssetHandle::raw()`.
    proxy_mesh_buffers: Vec<Option<(wgpu::Buffer, wgpu::Buffer, u32)>>,
    /// Primary-visibility selector — `March` (compute octree march)
    /// or `Splat` (rasterized surface splats). Read from the
    /// `RKP_PRIMARY` env var at construction. See [`PrimaryMode`].
    pub primary_mode: PrimaryMode,
    /// Device for buffer operations.
    pub device: wgpu::Device,
    /// GPU profiler (wgpu-profiler).
    pub profiler: GpuProfiler,
    timestamp_period: f32,
    /// Per-cascade shadow LOD pixel-threshold falloff. The shadow
    /// LOD-select uses `base * shadow_csm_threshold_falloff^cascade`
    /// as the per-cascade pixel-error budget. Set per frame by the
    /// engine from `EnvironmentSettings::shadow_csm_threshold_falloff`
    /// via `set_shadow_csm_threshold_falloff`. Default 2.0; range
    /// clamped 1.0..6.0.
    shadow_csm_threshold_falloff: f32,
}

impl RkpRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, _width: u32, _height: u32) -> Self {
        let scene = RkpScene::new(device);

        let profiler = GpuProfiler::new(device, wgpu_profiler::GpuProfilerSettings {
            enable_timer_queries: true,
            enable_debug_groups: true,
            max_num_pending_frames: 4,
        }).expect("failed to create GPU profiler");
        let timestamp_period = queue.get_timestamp_period();

        let default_params = ShadeParams { num_lights: 1, ..ShadeParams::default() };
        let default_light = GpuLight {
            position: [0.0, 0.0, 0.0, 0.0],
            color: [1.0, 0.95, 0.9, 2.0],
            direction: [-0.5, -0.8, -0.3, 0.0],
            params: [0.0; 4],
        };
        let default_material = GpuMaterial {
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            emission_color: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0, 0.8, 0.6],
            opacity: 1.0,
            ior: 1.5,
            noise_scale: 0.0,
            noise_strength: 0.0,
            noise_channels: 0,
            shader_id: 0,
            instance_shader_id: 0,
            _padding: [0.0; 4],
        };

        let shade_params_buffer = Self::create_init_buffer(device, "rkp_shade_params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_params));
        let lights_buffer = Self::create_init_buffer(device, "rkp_shade_lights",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_light));
        let materials_buffer = Self::create_init_buffer(device, "rkp_shade_materials",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&default_material));

        let atmosphere = RkpAtmospherePass::new(device);

        let skin_deform = crate::skin_deform::SkinDeformPass::new(device, &scene);
        let splat_pass = SplatPass::new(device);
        // `MeshGlassPass` owns the shared `g2_layout` (glass-classify
        // bindings) used by both the primary mesh raster and the
        // glass front/back rasters. Construct it first so the layout
        // can flow into `MeshPass::new`.
        let mesh_glass = crate::mesh_glass_pass::MeshGlassPass::new(
            device,
            &splat_pass.g0_layout,
            &splat_pass.g1_layout,
        );
        let mesh_pass = MeshPass::new(
            device,
            &splat_pass.g0_layout,
            &splat_pass.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_shadow_map = MeshShadowMapPass::new(
            device,
            &splat_pass.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_glass_shadow = crate::mesh_glass_shadow_pass::MeshGlassShadowPass::new(
            device,
            &mesh_shadow_map.render_g0_layout,
            &splat_pass.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_lod_select_pass =
            crate::mesh_lod_select_pass::MeshLodSelectPass::new(device, &splat_pass.g1_layout);
        let splat_resolve = SplatResolvePass::new(device);
        let mesh_proxy = crate::mesh_proxy_pass::MeshProxyPass::new(device);
        let user_shader_mesh = crate::user_shader_mesh_pass::UserShaderMeshPass::new(
            device,
            &scene.bind_group_layout,
        );
        let brush_state = crate::brush_state_pass::BrushStatePass::new(device);
        let primary_mode = PrimaryMode::from_env();
        eprintln!("[RkpRenderer] primary_mode = {primary_mode:?}");

        let mesh_glass_debug_force = std::env::var("RKP_MESH_GLASS_DEBUG_FORCE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        if mesh_glass_debug_force != 0 {
            let mode = match mesh_glass_debug_force {
                1 => "1: combine spoofs glass on every opaque hit (hardcoded normal/material)",
                2 => "2: FS treats every fragment as glass (leaf-derived material)",
                _ => "unknown — values are 1 or 2",
            };
            eprintln!("[RkpRenderer] RKP_MESH_GLASS_DEBUG_FORCE={mesh_glass_debug_force} — {mode}");
        }

        let lights_capacity = lights_buffer.size();
        let materials_capacity = materials_buffer.size();
        Self {
            scene, atmosphere,
            shade_params_buffer,
            lights_buffer, lights_capacity,
            materials_buffer, materials_capacity,
            lights_materials_epoch: 0,
            skin_deform,
            splat_pass,
            splat_resolve,
            mesh_proxy,
            user_shader_mesh,
            brush_state,
            splat_buffers: Vec::new(),
            mesh_pass,
            mesh_shadow_map,
            mesh_lod_select_pass,
            mesh_glass,
            mesh_glass_shadow,
            mesh_glass_debug_force,
            mesh_buffers: Vec::new(),
            mesh_cluster_buffers: Vec::new(),
            proxy_mesh_buffers: Vec::new(),
            primary_mode,
            device: device.clone(),
            profiler, timestamp_period,
            shadow_csm_threshold_falloff: 2.0,
        }
    }

    /// Set the per-cascade LOD pixel-threshold falloff (1.0..6.0).
    /// Engine writes this once per frame from
    /// `frame.shadow_csm_threshold_falloff` so the next
    /// `dispatch_mesh_shadow` picks it up. Replaces the prior
    /// `RKP_CSM_THRESHOLD_FALLOFF` env-var path (still honored as a
    /// CI override, see `dispatch_mesh_shadow`).
    pub fn set_shadow_csm_threshold_falloff(&mut self, v: f32) {
        self.shadow_csm_threshold_falloff = v.clamp(1.0, 6.0);
    }

    /// Upload (or replace) the splat vertex buffer for a given asset.
    /// Caller passes the asset's `AssetHandle::raw()` and the
    /// `&[SplatVertex]` from `RkpSceneManager::asset_splats`. Re-upload
    /// is safe — the previous buffer (if any) is dropped at the end of
    /// the call. Empty splat lists clear the cached entry.
    pub fn upload_splats_for_asset(&mut self, handle_raw: u32, splats: &[SplatVertex]) {
        use wgpu::util::DeviceExt;
        let idx = handle_raw as usize;
        if idx >= self.splat_buffers.len() {
            self.splat_buffers.resize_with(idx + 1, || None);
        }
        if splats.is_empty() {
            self.splat_buffers[idx] = None;
            return;
        }
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat asset vbo"),
            contents: bytemuck::cast_slice(splats),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.splat_buffers[idx] = Some((buffer, splats.len() as u32));
    }

    /// Drop the cached splat vertex buffer for `handle_raw`. Called
    /// when an asset is released or invalidated.
    pub fn release_splats_for_asset(&mut self, handle_raw: u32) {
        let idx = handle_raw as usize;
        if let Some(slot) = self.splat_buffers.get_mut(idx) {
            *slot = None;
        }
    }

    /// Look up the cached splat vertex buffer. Returns `(buffer,
    /// splat_count)` when the asset has been uploaded, else `None`.
    pub fn splat_buffer(&self, handle_raw: u32) -> Option<(&wgpu::Buffer, u32)> {
        self.splat_buffers
            .get(handle_raw as usize)
            .and_then(|s| s.as_ref())
            .map(|(b, c)| (b, *c))
    }

    /// Upload (or replace) the surface-mesh vertex + index buffers for
    /// a given asset. Caller passes the asset's `AssetHandle::raw()`,
    /// the `(vertices, indices)` slices from
    /// `RkpSceneManager::asset_mesh`, and `dispatch_index_count` —
    /// the index range that `dispatch_mesh` should draw. Phase 6.1
    /// passes `lod0_index_count` here so the IBO holds the full DAG
    /// (for Phase 6.2's indirect path) but dispatch keeps drawing
    /// only the LOD-0 prefix (visuals unchanged). Re-upload is safe;
    /// empty mesh clears the cached entry.
    pub fn upload_mesh_for_asset(
        &mut self,
        handle_raw: u32,
        vertices: &[MeshVertex],
        indices: &[u32],
        dispatch_index_count: u32,
    ) {
        use wgpu::util::DeviceExt;
        let idx = handle_raw as usize;
        if idx >= self.mesh_buffers.len() {
            self.mesh_buffers.resize_with(idx + 1, || None);
        }
        if vertices.is_empty() || indices.is_empty() || dispatch_index_count == 0 {
            self.mesh_buffers[idx] = None;
            return;
        }
        let vbo = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh asset vbo"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibo = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh asset ibo (full DAG)"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        });
        self.mesh_buffers[idx] = Some((vbo, ibo, dispatch_index_count));
    }

    /// Drop the cached mesh buffers for `handle_raw`. Called when an
    /// asset is released or invalidated.
    pub fn release_mesh_for_asset(&mut self, handle_raw: u32) {
        let idx = handle_raw as usize;
        if let Some(slot) = self.mesh_buffers.get_mut(idx) {
            *slot = None;
        }
    }

    /// Look up the cached mesh buffers. Returns `(vbo, ibo,
    /// index_count)` when the asset has been uploaded, else `None`.
    pub fn mesh_buffer(&self, handle_raw: u32) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        self.mesh_buffers
            .get(handle_raw as usize)
            .and_then(|s| s.as_ref())
            .map(|(v, i, c)| (v, i, *c))
    }

    /// Upload (or replace) the meshlet cluster table for an asset
    /// (Phase 5). Caller passes `AssetHandle::raw()` and the
    /// `&[MeshletCluster]` slice from
    /// `RkpSceneManager::iter_loaded_asset_clusters`. Re-upload is
    /// safe — the previous buffer is dropped at the end of the
    /// call. An empty cluster list clears the entry.
    pub fn upload_mesh_clusters_for_asset(
        &mut self,
        handle_raw: u32,
        clusters: &[MeshletCluster],
    ) {
        use wgpu::util::DeviceExt;
        let idx = handle_raw as usize;
        if idx >= self.mesh_cluster_buffers.len() {
            self.mesh_cluster_buffers.resize_with(idx + 1, || None);
        }
        if clusters.is_empty() {
            self.mesh_cluster_buffers[idx] = None;
            return;
        }
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh asset cluster table"),
            contents: bytemuck::cast_slice(clusters),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.mesh_cluster_buffers[idx] = Some((buffer, clusters.len() as u32));
    }

    /// Drop the cached cluster table for `handle_raw`. Called when
    /// an asset is released or invalidated.
    pub fn release_mesh_clusters_for_asset(&mut self, handle_raw: u32) {
        let idx = handle_raw as usize;
        if let Some(slot) = self.mesh_cluster_buffers.get_mut(idx) {
            *slot = None;
        }
    }

    /// Look up the cached cluster table. Returns `(buffer,
    /// cluster_count)` when the asset has been uploaded, else
    /// `None`.
    pub fn mesh_cluster_buffer(&self, handle_raw: u32) -> Option<(&wgpu::Buffer, u32)> {
        self.mesh_cluster_buffers
            .get(handle_raw as usize)
            .and_then(|s| s.as_ref())
            .map(|(b, c)| (b, *c))
    }

    /// Upload (or replace) a proxy mesh's vertex + index buffers. Same
    /// shape as `upload_mesh_for_asset` but writes into the proxy
    /// buffer slab (`ProxyVertex` layout, no cluster table — proxy
    /// meshes draw a single direct indexed draw, no LOD select).
    pub fn upload_proxy_mesh_for_asset(
        &mut self,
        handle_raw: u32,
        vertices: &[rkp_core::mesh_extract::ProxyVertex],
        indices: &[u32],
    ) {
        use wgpu::util::DeviceExt;
        let idx = handle_raw as usize;
        if idx >= self.proxy_mesh_buffers.len() {
            self.proxy_mesh_buffers.resize_with(idx + 1, || None);
        }
        if vertices.is_empty() || indices.is_empty() {
            self.proxy_mesh_buffers[idx] = None;
            return;
        }
        let vbo = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("proxy mesh vbo"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibo = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("proxy mesh ibo"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.proxy_mesh_buffers[idx] = Some((vbo, ibo, indices.len() as u32));
    }

    /// Drop the cached proxy mesh buffers for `handle_raw`.
    pub fn release_proxy_mesh_for_asset(&mut self, handle_raw: u32) {
        let idx = handle_raw as usize;
        if let Some(slot) = self.proxy_mesh_buffers.get_mut(idx) {
            *slot = None;
        }
    }

    /// Look up the proxy mesh buffers. Returns `(vbo, ibo,
    /// index_count)` when uploaded, else `None`.
    pub fn proxy_mesh_buffer(
        &self,
        handle_raw: u32,
    ) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        self.proxy_mesh_buffers
            .get(handle_raw as usize)
            .and_then(|s| s.as_ref())
            .map(|(v, i, c)| (v, i, *c))
    }

    /// Splat-raster equivalent of `OctreeMarchPass::dispatch`. Writes
    /// the same G-buffer the compute march writes, so the downstream
    /// shade / SSAO / etc passes are unchanged.
    ///
    /// Steps:
    ///   1. Refresh the per-VR `g0` bind group (if scene buffers /
    ///      lights+materials moved).
    ///   2. Grow per-instance uniform slots to `draws.len()`.
    ///   3. Write each `SplatDraw`'s world matrix + object_id into its
    ///      slot via `queue.write_buffer`.
    ///   4. Begin the splat render pass (clears all six gbuffer
    ///      targets to march-equivalent miss sentinels).
    ///   5. For each draw with a cached vertex buffer, bind the slot's
    ///      g1 + the asset's vbo and `pass.draw(0..4, 0..count)`.
    ///
    /// Draws with no cached vertex buffer (asset not yet uploaded) are
    /// silently skipped — they'll show through as the "miss" clear,
    /// matching how the march path handles missing assets.
    pub fn dispatch_splat(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[SplatDraw],
    ) {
        viewport.refresh_splat_g0(&self.device, self);
        viewport.refresh_splat_resolve_bindings(&self.device, self);
        viewport.ensure_splat_instance_capacity(&self.device, self, draws.len() as u32);
        for (slot, d) in draws.iter().enumerate() {
            viewport.write_splat_instance(
                queue,
                slot as u32,
                &d.world,
                d.object_id,
                d.grid_origin,
                d.bone_offset_lbs,
                d.bone_offset_dqs,
                d.skinning_mode,
            );
        }

        // RKP_SPLAT_STATS=1 prints per-frame draw stats with a
        // per-asset breakdown. The breakdown distinguishes "lots of
        // unique geometry" from "few unique assets × many instances"
        // — different shapes call for different optimizations
        // (frustum cull / hardware-instanced single draw / LOD cut).
        if std::env::var("RKP_SPLAT_STATS").is_ok() {
            use std::collections::HashMap;
            // handle_raw → (instance_count, splats_per_asset)
            let mut per_asset: HashMap<u32, (u32, u32)> = HashMap::new();
            let mut total_splats: u64 = 0;
            let mut drawn = 0u32;
            let mut missing = 0u32;
            for d in draws {
                match self.splat_buffer(d.asset_handle_raw) {
                    Some((_, count)) => {
                        total_splats += count as u64;
                        drawn += 1;
                        let entry = per_asset
                            .entry(d.asset_handle_raw)
                            .or_insert((0, count));
                        entry.0 += 1;
                    }
                    None => missing += 1,
                }
            }
            let unique_splats: u64 = per_asset.values().map(|(_, s)| *s as u64).sum();
            eprintln!(
                "[splat] {}×{} · {} draws ({} drawn, {} skipped) · {} unique assets · {} unique splats · {} total rasterized",
                viewport.width, viewport.height,
                draws.len(), drawn, missing,
                per_asset.len(), unique_splats, total_splats,
            );
            // Sort by total contribution descending so the heaviest
            // hitter shows up first.
            let mut rows: Vec<_> = per_asset.iter().collect();
            rows.sort_by_key(|(_, (n, s))| std::cmp::Reverse(*n as u64 * *s as u64));
            for (handle, (n_inst, splats)) in rows {
                eprintln!(
                    "  asset {}: {} inst × {} splats = {}",
                    handle, n_inst, splats, *n_inst as u64 * *splats as u64,
                );
            }
        }

        let g0_bg = viewport
            .splat_g0_bg
            .as_ref()
            .expect("splat g0 bg present after refresh_splat_g0");

        // 0. Clear the rest_pos texture so its .w stays 0 across the
        //    splat path. The splat raster doesn't bind rest_pos —
        //    only the mesh raster writes it — but `splat_resolve` reads
        //    it to decide whether to do per-pixel octree descent. If
        //    we don't clear, leftover mesh-frame writes leak through
        //    on splat frames and the resolve mis-descends with stale
        //    rest_pos values, producing wrong cell colors. The cheapest
        //    portable clear is a render pass with a single color
        //    attachment whose only op is LoadOp::Clear.
        {
            let _clear_rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat clear rest_pos"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &viewport.gbuffer.rest_pos_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        // 1. Visibility-buffer raster — writes position + pick +
        //    leaf_slot for hit pixels; clears all three (and depth) to
        //    march-equivalent miss sentinels.
        let q_raster = self.profiler.begin_query("splat_raster", encoder);
        {
            let mut rp = self.splat_pass.begin_pass(
                encoder,
                &viewport.gbuffer.position_view,
                &viewport.pick_view,
                &viewport.gbuffer.leaf_slot_view,
                &viewport.gbuffer.depth_view,
                None,
            );
            rp.set_pipeline(&self.splat_pass.pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
            for (slot, d) in draws.iter().enumerate() {
                let Some((vbo, count)) = self.splat_buffer(d.asset_handle_raw) else {
                    continue;
                };
                let g1_bg = &viewport.splat_instance_bind_groups[slot];
                rp.set_bind_group(1, g1_bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.draw(0..4, 0..count);
            }
        }
        self.profiler.end_query(encoder, q_raster);

        // 2. Resolve compute — reads (leaf_slot, pick) per pixel,
        //    writes normal / material / glass via the storage-texture
        //    G-buffer entries. Branches on leaf_slot==0 to write march-
        //    equivalent miss sentinels for non-hit pixels (so those
        //    targets don't need a separate clear).
        let resolve_g0 = viewport
            .splat_resolve_g0_bg
            .as_ref()
            .expect("splat_resolve g0 bg present after refresh");
        let resolve_g1 = viewport
            .splat_resolve_g1_bg
            .as_ref()
            .expect("splat_resolve g1 bg present after refresh");
        let q_resolve = self.profiler.begin_query("splat_resolve", encoder);
        self.splat_resolve.dispatch(
            encoder,
            resolve_g0,
            resolve_g1,
            viewport.width,
            viewport.height,
        );
        self.profiler.end_query(encoder, q_resolve);
    }

    /// Rasterize procedural proxy meshes into the existing G-buffer.
    /// V1 mesh-path user-shader raster — one indirect draw per
    /// active user-shader material. Composites onto the G-buffer
    /// using `LoadOp::Load` + depth-test, same shape as
    /// `dispatch_proxy_meshes`. Caller has already dispatched the
    /// compute trio (spawn_count → prefix_sum → fill) for each
    /// material, so the indirect-args buffer is ready when we get
    /// here.
    pub fn dispatch_user_shader_mesh(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[crate::user_shader_mesh_pass::UserShaderMeshDraw],
    ) {
        if draws.is_empty() {
            return;
        }
        viewport.refresh_user_shader_mesh_g0(&self.device, &self.user_shader_mesh);
        let g0_bg = viewport
            .user_shader_mesh_g0_bg
            .as_ref()
            .expect("user_shader_mesh g0 bg present after refresh");

        let q = self.profiler.begin_query("user_shader_mesh_raster", encoder);
        {
            let mut rp = self.user_shader_mesh.begin_raster_pass(
                encoder,
                &viewport.gbuffer.position_view,
                &viewport.pick_view,
                &viewport.gbuffer.normal_view,
                &viewport.gbuffer.material_view,
                &viewport.gbuffer.glass_view,
                &viewport.gbuffer.depth_view,
                None,
            );
            rp.set_bind_group(0, g0_bg, &[]);
            for d in draws {
                rp.set_pipeline(&d.raster_pipeline);
                rp.set_bind_group(1, &d.raster_g1, &[]);
                rp.draw_indirect(&d.indirect_buffer, 0);
            }
        }
        self.profiler.end_query(encoder, q);
    }

    /// Runs after the primary mode's main pass + `splat_resolve` so the
    /// G-buffer carries octree-mesh/splat output; proxy raster
    /// depth-composites on top using `LoadOp::Load`. Writes all five
    /// gbuf targets directly — no `splat_resolve` participation for
    /// proxy pixels.
    pub fn dispatch_proxy_meshes(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[crate::mesh_proxy_pass::ProxyDraw],
    ) {
        if draws.is_empty() {
            return;
        }
        viewport.refresh_proxy_g0(&self.device, self);
        viewport.ensure_proxy_instance_capacity(&self.device, self, draws.len() as u32);
        for (slot, d) in draws.iter().enumerate() {
            viewport.write_proxy_instance(queue, slot as u32, &d.world, d.object_id);
        }

        let g0_bg = viewport
            .proxy_g0_bg
            .as_ref()
            .expect("proxy g0 bg present after refresh");

        let q = self.profiler.begin_query("proxy_raster", encoder);
        {
            let mut rp = self.mesh_proxy.begin_pass(
                encoder,
                &viewport.gbuffer.position_view,
                &viewport.pick_view,
                &viewport.gbuffer.normal_view,
                &viewport.gbuffer.material_view,
                &viewport.gbuffer.glass_view,
                &viewport.gbuffer.depth_view,
                None,
            );
            rp.set_pipeline(&self.mesh_proxy.pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
            for (slot, d) in draws.iter().enumerate() {
                let Some((vbo, ibo, index_count)) = self.proxy_mesh_buffer(d.handle_raw) else {
                    continue;
                };
                let g1_bg = &viewport.proxy_instance_bind_groups[slot];
                rp.set_bind_group(1, g1_bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..index_count, 0, 0..1);
            }
        }
        self.profiler.end_query(encoder, q);
    }

    /// Surface-mesh equivalent of `dispatch_splat`. Phase 6.3:
    /// per draw, runs the LOD-select compute pass that fills a
    /// `DrawIndexedIndirectArgs` table for the asset's full DAG of
    /// clusters; then issues `multi_draw_indexed_indirect` over that
    /// table. Non-admitted slots carry `index_count = 0` so the no-op
    /// draws cost nothing.
    ///
    /// Visibility-buffer contract is unchanged from Phase 1-3 — the
    /// splat-resolve compute pass still reads (leaf_slot, pick) +
    /// fills normal / material / glass per pixel.
    pub fn dispatch_mesh(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[SplatDraw],
    ) {
        // Diagnostic: skip primary visibility GPU work. Used to
        // measure shadow_render in isolation when chasing the
        // anti-correlation pattern between the two passes. The
        // per-slot instance + bind-group setup below still runs
        // because the shadow path consumes
        // `splat_instance_bind_groups` for per-instance world
        // matrices — early-returning here would crash shadow with
        // an empty Vec. The actual LOD-select compute + raster
        // render passes are gated separately at their dispatch
        // sites.
        let primary_disabled = std::env::var("RKP_MESH_DISABLE_PRIMARY").is_ok();
        viewport.refresh_splat_g0(&self.device, self);
        viewport.refresh_splat_resolve_bindings(&self.device, self);
        viewport.refresh_mesh_glass_bindings(&self.device, self);
        // `RKP_MESH_GLASS_DEBUG_FORCE=2` raises the FS opacity gate
        // above 1.0 so every fragment classifies as glass with its
        // actual leaf-derived material; a bisect for "the FS classify
        // path itself is broken" vs "the discard threshold is wrong
        // for this material".
        let fs_threshold = if self.mesh_glass_debug_force == 2 {
            10.0
        } else {
            crate::mesh_glass_pass::DEFAULT_OPACITY_THRESHOLD
        };
        viewport.write_mesh_glass_combine_params(
            queue,
            self.mesh_glass_debug_force,
            fs_threshold,
        );
        viewport.ensure_splat_instance_capacity(&self.device, self, draws.len() as u32);
        viewport.ensure_mesh_lod_capacity(&self.device, self, draws.len() as u32);
        for (slot, d) in draws.iter().enumerate() {
            viewport.write_splat_instance(
                queue,
                slot as u32,
                &d.world,
                d.object_id,
                d.grid_origin,
                d.bone_offset_lbs,
                d.bone_offset_dqs,
                d.skinning_mode,
            );
        }

        // Phase 6.3: per-draw LOD-select prep. Resolve each draw's
        // cluster count + ensure per-slot args buffer + g2 bind
        // group are sized for it; write the per-slot params uniform.
        // Skip slots whose asset is unloaded (or has no clusters yet).
        //
        // `RKP_MESH_LOD_THRESHOLD` overrides the compile-time default
        // for LOD-effectiveness tuning experiments. Higher = more
        // aggressive LOD culling (fewer fine clusters admit).
        const PIXEL_THRESHOLD_PRIMARY: f32 = 1.0;
        let pixel_threshold_primary = pixel_threshold_env(
            "RKP_MESH_LOD_THRESHOLD",
            PIXEL_THRESHOLD_PRIMARY,
        );
        let lod_stats_enabled = std::env::var("RKP_MESH_LOD_STATS").is_ok();
        let pipestats_enabled = std::env::var("RKP_MESH_PIPESTATS").is_ok();
        let force_admit_flag: u32 = if std::env::var("RKP_MESH_DEBUG_FORCE_ADMIT").is_ok() {
            1
        } else {
            0
        };
        // RKP_MESH_DEBUG_FORCE_LEVEL=N: bypass Karis admit and admit
        // ONLY clusters at LOD level N. Used to diagnose whether
        // mixed-level admit is causing cross-cluster cracks.
        // Sentinel u32::MAX == "use Karis admit (default)".
        let force_level: u32 = std::env::var("RKP_MESH_DEBUG_FORCE_LEVEL")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        let mut slot_active: Vec<bool> = vec![false; draws.len()];
        for (slot, d) in draws.iter().enumerate() {
            let Some((_, _, _)) = self.mesh_buffer(d.asset_handle_raw) else {
                continue;
            };
            let Some((cluster_buf, cluster_count)) =
                self.mesh_cluster_buffer(d.asset_handle_raw)
            else {
                continue;
            };
            if cluster_count == 0 {
                continue;
            }
            // The cluster buffer is owned by `self` (by raw handle);
            // bind group creation only needs `&Buffer` references.
            // Take a local copy of the asset handle so we can release
            // the shared borrow before mutating viewport state.
            let asset_handle_raw = d.asset_handle_raw;

            viewport.ensure_mesh_lod_args_capacity(
                &self.device,
                slot as u32,
                cluster_count,
            );

            // Rebuild g2 bind group if asset changed at this slot or
            // args buffer was reallocated.
            let (args_buf, args_cap) = &viewport.mesh_lod_args_buffers[slot];
            let need_rebuild = match &viewport.mesh_lod_select_g2_bgs[slot] {
                Some((_, cached_handle, cached_cap)) => {
                    *cached_handle != asset_handle_raw || *cached_cap != *args_cap
                }
                None => true,
            };
            if need_rebuild {
                let count_buf = &viewport.mesh_lod_count_buffers[slot];
                let bg = self.mesh_lod_select_pass.create_g2_bind_group(
                    &self.device,
                    cluster_buf,
                    args_buf,
                    &viewport.mesh_lod_admit_stats_primary,
                    count_buf,
                );
                viewport.mesh_lod_select_g2_bgs[slot] = Some((bg, asset_handle_raw, *args_cap));
            }

            // Per-draw uniform — admit threshold + cluster count.
            let params = crate::mesh_lod_select_pass::MeshLodSelectParams {
                pixel_threshold: pixel_threshold_primary,
                cluster_count,
                force_admit: force_admit_flag,
                record_stats: lod_stats_enabled as u32,
                force_level,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &viewport.mesh_lod_params_buffers[slot],
                0,
                bytemuck::bytes_of(&params),
            );

            slot_active[slot] = true;
        }

        // `RKP_MESH_DEBUG_DIRECT=1` bypasses Phase 6.2/6.3 entirely:
        // skip the LOD-select compute and the
        // `multi_draw_indexed_indirect` dispatch, fall back to a
        // Phase 1-3-style direct `draw_indexed` over the LOD-0
        // prefix. Used to bisect "no geometry" issues — if direct
        // mode renders but indirect doesn't, the bug is in the
        // LOD-select / indirect-dispatch path.
        let direct_mode = std::env::var("RKP_MESH_DEBUG_DIRECT").is_ok();

        // 0. Per-cluster LOD-select compute pass. One dispatch per
        //    active draw slot writes that draw's args table.
        if !direct_mode && !primary_disabled {
            // Stats lifecycle (RKP_MESH_LOD_STATS=1 only):
            // (a) drain previous frame's mapped buffer if ready
            // (b) clear the histogram for this frame's atomics
            // (c) dispatch — atomics fire iff record_stats != 0
            // (d) copy histogram → staging + map_async for next frame
            if lod_stats_enabled {
                viewport.lod_stats_drain_primary("primary");
                viewport.lod_stats_clear_primary(encoder);
            }
            if pipestats_enabled {
                viewport.pipestats_drain();
            }

            // Zero each active slot's atomic count buffer before the
            // LOD-select dispatch — admitted threads atomicAdd into
            // it, and `multi_draw_indexed_indirect_count` reads it
            // as the actual draw count for the render pass.
            for (slot, _) in draws.iter().enumerate() {
                if !slot_active[slot] {
                    continue;
                }
                encoder.clear_buffer(&viewport.mesh_lod_count_buffers[slot], 0, None);
            }

            let q_lod = self.profiler.begin_query("mesh_lod_select", encoder);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mesh_lod_select"),
                    timestamp_writes: None,
                });
                if pipestats_enabled {
                    cpass.begin_pipeline_statistics_query(
                        &viewport.mesh_pipestats_query_set, 0,
                    );
                }
                for (slot, d) in draws.iter().enumerate() {
                    if !slot_active[slot] {
                        continue;
                    }
                    let cluster_count = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    let g0 = &viewport.mesh_lod_select_g0_bgs[slot];
                    let g1 = &viewport.splat_instance_bind_groups[slot];
                    let g2 = &viewport
                        .mesh_lod_select_g2_bgs[slot]
                        .as_ref()
                        .expect("g2 set above for active slot")
                        .0;
                    self.mesh_lod_select_pass
                        .dispatch(&mut cpass, g0, g1, g2, cluster_count);
                }
                if pipestats_enabled {
                    cpass.end_pipeline_statistics_query();
                }
            }
            self.profiler.end_query(encoder, q_lod);

            if lod_stats_enabled {
                viewport.lod_stats_finalize_primary(encoder);
            }
        }

        // RKP_MESH_STATS=1 prints per-frame mesh stats with a per-asset
        // breakdown, mirroring RKP_SPLAT_STATS. The cluster counts are
        // load-bearing for diagnosing emulated-multi-draw CPU cost —
        // wgpu emulates `multi_draw_indexed_indirect` as N
        // `draw_indexed_indirect` calls when the adapter lacks
        // `MULTI_DRAW_INDIRECT_COUNT`, where N is the total cluster
        // count across all active draw slots. Total clusters per
        // primary pass × 2 (primary + shadow) = upper bound on
        // emulated indirect-draw calls per frame.
        if std::env::var("RKP_MESH_STATS").is_ok() {
            use std::collections::HashMap;
            // handle_raw → (instance_count, indices_per_asset, clusters_per_asset)
            let mut per_asset: HashMap<u32, (u32, u32, u32)> = HashMap::new();
            let mut total_indices: u64 = 0;
            let mut total_clusters: u64 = 0;
            let mut drawn = 0u32;
            let mut missing = 0u32;
            for d in draws {
                match self.mesh_buffer(d.asset_handle_raw) {
                    Some((_, _, count)) => {
                        total_indices += count as u64;
                        let clusters = self
                            .mesh_cluster_buffer(d.asset_handle_raw)
                            .map(|(_, c)| c)
                            .unwrap_or(0);
                        total_clusters += clusters as u64;
                        drawn += 1;
                        let entry = per_asset
                            .entry(d.asset_handle_raw)
                            .or_insert((0, count, clusters));
                        entry.0 += 1;
                    }
                    None => missing += 1,
                }
            }
            let unique_indices: u64 = per_asset.values().map(|(_, s, _)| *s as u64).sum();
            let unique_clusters: u64 = per_asset.values().map(|(_, _, c)| *c as u64).sum();
            eprintln!(
                "[mesh] {}×{} · {} draws ({} drawn, {} skipped) · {} unique assets · {} unique tris · {} total tris rasterized · {} unique clusters · {} total clusters (= emulated multi-draw call count for this pass)",
                viewport.width, viewport.height,
                draws.len(), drawn, missing,
                per_asset.len(), unique_indices / 3, total_indices / 3,
                unique_clusters, total_clusters,
            );
        }

        if primary_disabled {
            // Diagnostic isolation mode: skip the raster + resolve
            // GPU work entirely. The per-slot setup above still ran
            // so shadow has populated bind groups to consume.
            return;
        }

        let g0_bg = viewport
            .splat_g0_bg
            .as_ref()
            .expect("splat g0 bg present after refresh_splat_g0");
        // `glass_g2` is needed by the primary mesh raster (FS
        // glass-discard) AND by the glass front/back passes. Borrow
        // it once at the top so all three render passes can use the
        // same bind group.
        let glass_g2 = viewport
            .mesh_glass_g2_bg
            .as_ref()
            .expect("mesh_glass g2 bg present after refresh");
        let glass_combine_bg = viewport
            .mesh_glass_combine_bg
            .as_ref()
            .expect("mesh_glass combine bg present after refresh");

        // 1. Visibility-buffer raster — same RT layout as the splat
        //    pass; clears use the same march-equivalent miss sentinels.
        let q_raster = self.profiler.begin_query("mesh_raster", encoder);
        {
            let mut rp = self.mesh_pass.begin_pass(
                encoder,
                &viewport.gbuffer.position_view,
                &viewport.pick_view,
                &viewport.gbuffer.leaf_slot_view,
                &viewport.gbuffer.rest_pos_view,
                &viewport.gbuffer.depth_view,
                None,
            );
            rp.set_pipeline(&self.mesh_pass.pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
            // `g2` (glass-classify) is required so the FS can
            // `discard` glass fragments. Without it, glass meshes
            // write opaque depth into `gbuf_position` and the glass
            // composite then gates them out (entry == opaque).
            rp.set_bind_group(2, glass_g2, &[]);
            if pipestats_enabled {
                rp.begin_pipeline_statistics_query(
                    &viewport.mesh_pipestats_query_set, 1,
                );
            }
            for (slot, d) in draws.iter().enumerate() {
                let Some((vbo, ibo, lod0_index_count)) = self.mesh_buffer(d.asset_handle_raw)
                else {
                    continue;
                };
                if !slot_active[slot] {
                    continue;
                }
                let g1_bg = &viewport.splat_instance_bind_groups[slot];
                rp.set_bind_group(1, g1_bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                if direct_mode {
                    // Phase-1-3-style direct draw of the LOD-0 prefix.
                    // No compute pass / indirect args buffer involved.
                    rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                } else {
                    let cluster_count = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    // `RKP_MESH_DEBUG_MAX_DRAWS` caps the number of
                    // indirect-args entries `multi_draw_indexed_indirect`
                    // walks. With 50K+ clusters per asset and 20+ assets
                    // per scene the total draw count slams the Vulkan
                    // command processor and kills the UI thread; capping
                    // to e.g. 100 lets us prove the dispatch path itself
                    // works without that pressure.
                    let max_draws = std::env::var("RKP_MESH_DEBUG_MAX_DRAWS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|n| n.min(cluster_count))
                        .unwrap_or(cluster_count);
                    let (args_buf, _) = &viewport.mesh_lod_args_buffers[slot];
                    let count_buf = &viewport.mesh_lod_count_buffers[slot];
                    rp.multi_draw_indexed_indirect_count(
                        args_buf, 0, count_buf, 0, max_draws,
                    );
                }
            }
            if pipestats_enabled {
                rp.end_pipeline_statistics_query();
            }
        }
        self.profiler.end_query(encoder, q_raster);

        // 2. Resolve compute — identical to the splat path.
        let resolve_g0 = viewport
            .splat_resolve_g0_bg
            .as_ref()
            .expect("splat_resolve g0 bg present after refresh");
        let resolve_g1 = viewport
            .splat_resolve_g1_bg
            .as_ref()
            .expect("splat_resolve g1 bg present after refresh");
        let q_resolve = self.profiler.begin_query("mesh_resolve", encoder);
        self.splat_resolve.dispatch(
            encoder,
            resolve_g0,
            resolve_g1,
            viewport.width,
            viewport.height,
        );
        self.profiler.end_query(encoder, q_resolve);

        // 3. Mesh-mode glass — front raster + back raster + combine.
        //    Runs after `splat_resolve` (which writes zeros to
        //    `gbuf_glass`); the combine pass overwrites those zeros
        //    with actual glass data wherever a glass fragment was
        //    captured. `glass_g2` and `glass_combine_bg` were borrowed
        //    at the top of this method.

        // Closure that issues every primary mesh draw against the
        // currently-bound glass pipeline. Reuses the LOD-select
        // indirect args + count from the primary path — glass picks
        // the same LOD level as opaque, which keeps thickness
        // consistent across faces (mixing LODs would let one face
        // come from LOD-0 and the other from LOD-3, producing a
        // mismatched shell).
        let q_glass_front = self.profiler.begin_query("mesh_glass_front", encoder);
        {
            let mut rp = self.mesh_glass.begin_front_pass(
                encoder,
                &viewport.glass_entry_packed_view,
                &viewport.glass_depth_front_view,
            );
            rp.set_pipeline(&self.mesh_glass.front_pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
            rp.set_bind_group(2, glass_g2, &[]);
            for (slot, d) in draws.iter().enumerate() {
                if !d.has_glass { continue; }
                let Some((vbo, ibo, lod0_index_count)) = self.mesh_buffer(d.asset_handle_raw)
                else { continue; };
                if !slot_active[slot] { continue; }
                let g1_bg = &viewport.splat_instance_bind_groups[slot];
                rp.set_bind_group(1, g1_bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                if direct_mode {
                    rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                } else {
                    let cluster_count = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    let max_draws = std::env::var("RKP_MESH_DEBUG_MAX_DRAWS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|n| n.min(cluster_count))
                        .unwrap_or(cluster_count);
                    let (args_buf, _) = &viewport.mesh_lod_args_buffers[slot];
                    let count_buf = &viewport.mesh_lod_count_buffers[slot];
                    rp.multi_draw_indexed_indirect_count(
                        args_buf, 0, count_buf, 0, max_draws,
                    );
                }
            }
        }
        self.profiler.end_query(encoder, q_glass_front);

        let q_glass_back = self.profiler.begin_query("mesh_glass_back", encoder);
        {
            let mut rp = self.mesh_glass.begin_back_pass(
                encoder,
                &viewport.glass_exit_dist_view,
                &viewport.glass_depth_back_view,
            );
            rp.set_pipeline(&self.mesh_glass.back_pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
            rp.set_bind_group(2, glass_g2, &[]);
            for (slot, d) in draws.iter().enumerate() {
                if !d.has_glass { continue; }
                let Some((vbo, ibo, lod0_index_count)) = self.mesh_buffer(d.asset_handle_raw)
                else { continue; };
                if !slot_active[slot] { continue; }
                let g1_bg = &viewport.splat_instance_bind_groups[slot];
                rp.set_bind_group(1, g1_bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                if direct_mode {
                    rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                } else {
                    let cluster_count = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    let max_draws = std::env::var("RKP_MESH_DEBUG_MAX_DRAWS")
                        .ok()
                        .and_then(|s| s.parse::<u32>().ok())
                        .map(|n| n.min(cluster_count))
                        .unwrap_or(cluster_count);
                    let (args_buf, _) = &viewport.mesh_lod_args_buffers[slot];
                    let count_buf = &viewport.mesh_lod_count_buffers[slot];
                    rp.multi_draw_indexed_indirect_count(
                        args_buf, 0, count_buf, 0, max_draws,
                    );
                }
            }
        }
        self.profiler.end_query(encoder, q_glass_back);

        let q_glass_combine = self.profiler.begin_query("mesh_glass_combine", encoder);
        self.mesh_glass.dispatch_combine(
            encoder,
            glass_combine_bg,
            viewport.width,
            viewport.height,
        );
        self.profiler.end_query(encoder, q_glass_combine);
    }

    /// Render the mesh-mode directional shadow map. Mirrors
    /// `dispatch_mesh` but draws into the shadow_buffer (atomicMin via
    /// fragment shader) using the light camera's view-proj. Per-asset
    /// vertex/index buffers and per-instance uniforms are reused from
    /// the primary mesh dispatch — they describe the same triangles.
    ///
    /// Caller must:
    ///   1. Have already populated `shadow_map.uniform_buffer` with a
    ///      live `LightCameraUniform` for this frame (engine does this
    ///      in `prepare_shadow_maps`).
    ///   2. Have already populated `splat_instance_buffers` (any
    ///      previous `dispatch_mesh`/`dispatch_splat` this frame did
    ///      this; if mesh-shadow runs before primary mesh dispatch the
    ///      caller should write the per-instance uniforms first).
    ///
    /// This dispatch clears `shadow_map.shadow_buffer` (via the
    /// shared `ShadowMapPass::dispatch_clear` compute pass) before
    /// the render so each frame starts from FAR_DEPTH.
    pub fn dispatch_mesh_shadow(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[SplatDraw],
    ) {
        // Diagnostic: skip shadow visibility entirely. Used to
        // measure mesh_raster in isolation. Visuals: shadows go
        // pure black for a frame as the shadow buffer stays at
        // its last cleared state; shade still samples it.
        if std::env::var("RKP_MESH_DISABLE_SHADOW").is_ok() {
            return;
        }
        viewport.refresh_mesh_shadow_bindings(&self.device, self);
        viewport.refresh_mesh_glass_bindings(&self.device, self);
        viewport.ensure_mesh_lod_shadow_capacity(&self.device, self, draws.len() as u32);

        // `RKP_MESH_DEBUG_DIRECT=1` also bypasses the shadow LOD
        // compute + indirect dispatch — depth-only `draw_indexed` of
        // the LOD-0 prefix, just like the Phase 1-3 baseline.
        let direct_mode = std::env::var("RKP_MESH_DEBUG_DIRECT").is_ok();

        // Per-cascade base pixel threshold + falloff. Each cascade
        // scales the base by `falloff^cascade_index` so the far
        // cascade culls hardest. Default base 2.0 (one LOD coarser
        // than primary's 1.0) and falloff 4.0 → cascades 0..3 use
        // thresholds {2, 8, 32, 128}, which keeps far-cascade cluster
        // counts in the ~1 % range of the primary path. Override
        // with `RKP_MESH_SHADOW_LOD_THRESHOLD` (base) and
        // `RKP_CSM_THRESHOLD_FALLOFF` (per-cascade scale).
        const PIXEL_THRESHOLD_SHADOW: f32 = 2.0;
        let base_threshold = pixel_threshold_env(
            "RKP_MESH_SHADOW_LOD_THRESHOLD",
            PIXEL_THRESHOLD_SHADOW,
        );
        // Per-cascade falloff: env var (CI / headless) takes precedence
        // over the engine-set value so RKP_CSM_THRESHOLD_FALLOFF still
        // works for one-shot diagnostic runs. UI / scene-stored value
        // flows in via `set_shadow_csm_threshold_falloff` (env panel
        // slider, range clamped 1.0..6.0).
        let threshold_falloff = std::env::var("RKP_CSM_THRESHOLD_FALLOFF")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(self.shadow_csm_threshold_falloff)
            .clamp(1.0, 6.0);
        let lod_stats_enabled = std::env::var("RKP_MESH_LOD_STATS").is_ok();
        let pipestats_enabled = std::env::var("RKP_MESH_PIPESTATS").is_ok();
        let force_admit = std::env::var("RKP_MESH_DEBUG_FORCE_ADMIT").is_ok();
        let force_level: u32 = std::env::var("RKP_MESH_DEBUG_FORCE_LEVEL")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        let max_draws_override = std::env::var("RKP_MESH_DEBUG_MAX_DRAWS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());

        let cascade_count = crate::shadow_map_pass::CSM_CASCADE_COUNT as usize;

        // Determine per-slot eligibility once — the same set of slots
        // applies to every cascade (they all read the same cluster
        // table; only `pixel_threshold` differs).
        let mut slot_active: Vec<bool> = vec![false; draws.len()];
        for (slot, d) in draws.iter().enumerate() {
            if self.mesh_buffer(d.asset_handle_raw).is_none() {
                continue;
            }
            let Some((_, cluster_count)) = self.mesh_cluster_buffer(d.asset_handle_raw) else {
                continue;
            };
            if cluster_count == 0 {
                continue;
            }
            slot_active[slot] = true;
        }

        // ── Per-cascade loop ───────────────────────────────────────
        for cascade in 0..cascade_count {
            // Cascade-scaled pixel threshold: cascade 0 admits more
            // (sharp shadows near camera); cascade N-1 admits very
            // few (cheap far coverage). Keeps total shadow GPU cost
            // comparable to the single-cascade baseline.
            let pixel_threshold =
                base_threshold * threshold_falloff.powi(cascade as i32);

            // LOD admit stats are still pass-shared (one buffer per
            // primary/shadow), so we collect on cascade 0 only —
            // mixing all cascades' atomicAdds would double-count.
            // Pipestats has dedicated per-cascade slots in the query
            // set, so it runs every cascade.
            let collect_stats = lod_stats_enabled && cascade == 0;
            let collect_pipestats = pipestats_enabled;

            // Per-slot prep for this cascade: ensure args buffer +
            // (re)build g2 bg + write params with this cascade's
            // pixel threshold.
            for (slot, d) in draws.iter().enumerate() {
                if !slot_active[slot] {
                    continue;
                }
                let asset_handle_raw = d.asset_handle_raw;
                let Some((cluster_buf, cluster_count)) =
                    self.mesh_cluster_buffer(asset_handle_raw)
                else {
                    continue;
                };

                viewport.ensure_mesh_lod_shadow_args_capacity(
                    &self.device,
                    slot as u32,
                    cascade as u32,
                    cluster_count,
                );

                let (args_buf, args_cap) =
                    &viewport.mesh_lod_shadow_args_buffers[slot][cascade];
                let need_rebuild = match &viewport.mesh_lod_shadow_g2_bgs[slot][cascade] {
                    Some((_, cached_handle, cached_cap)) => {
                        *cached_handle != asset_handle_raw || *cached_cap != *args_cap
                    }
                    None => true,
                };
                if need_rebuild {
                    let count_buf = &viewport.mesh_lod_shadow_count_buffers[slot][cascade];
                    let bg = self.mesh_lod_select_pass.create_g2_bind_group(
                        &self.device,
                        cluster_buf,
                        args_buf,
                        &viewport.mesh_lod_admit_stats_shadow,
                        count_buf,
                    );
                    viewport.mesh_lod_shadow_g2_bgs[slot][cascade] =
                        Some((bg, asset_handle_raw, *args_cap));
                }

                let params = crate::mesh_lod_select_pass::MeshLodSelectParams {
                    pixel_threshold,
                    cluster_count,
                    force_admit: force_admit as u32,
                    record_stats: collect_stats as u32,
                    force_level,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                queue.write_buffer(
                    &viewport.mesh_lod_shadow_params_buffers[slot][cascade],
                    0,
                    bytemuck::bytes_of(&params),
                );
            }

            // 0. Shadow-side LOD-select compute pass for this cascade.
            if !direct_mode {
                if collect_stats {
                    viewport.lod_stats_drain_shadow("shadow");
                    viewport.lod_stats_clear_shadow(encoder);
                }

                // Zero this cascade's per-slot atomic count buffers
                // before the dispatch — same role as the primary
                // path; consumed by `multi_draw_indexed_indirect_count`
                // in the matching mesh_shadow_render below.
                for (slot, _) in draws.iter().enumerate() {
                    if !slot_active[slot] {
                        continue;
                    }
                    encoder.clear_buffer(
                        &viewport.mesh_lod_shadow_count_buffers[slot][cascade],
                        0,
                        None,
                    );
                }

                let q_lod = self.profiler.begin_query(
                    &format!("mesh_shadow_lod_select[{cascade}]"),
                    encoder,
                );
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("mesh_shadow_lod_select"),
                        timestamp_writes: None,
                    });
                    if collect_pipestats {
                        // Slots 2..6 = mesh_shadow_lod_select[0..3].
                        cpass.begin_pipeline_statistics_query(
                            &viewport.mesh_pipestats_query_set,
                            2 + cascade as u32,
                        );
                    }
                    for (slot, d) in draws.iter().enumerate() {
                        if !slot_active[slot] {
                            continue;
                        }
                        let cluster_count = self
                            .mesh_cluster_buffer(d.asset_handle_raw)
                            .map(|(_, c)| c)
                            .unwrap_or(0);
                        let g0 = &viewport.mesh_lod_shadow_g0_bgs[slot][cascade];
                        let g1 = &viewport.splat_instance_bind_groups[slot];
                        let g2 = &viewport
                            .mesh_lod_shadow_g2_bgs[slot][cascade]
                            .as_ref()
                            .expect("g2 set above for active slot")
                            .0;
                        self.mesh_lod_select_pass
                            .dispatch(&mut cpass, g0, g1, g2, cluster_count);
                    }
                    if collect_pipestats {
                        cpass.end_pipeline_statistics_query();
                    }
                }
                self.profiler.end_query(encoder, q_lod);

                if collect_stats {
                    viewport.lod_stats_finalize_shadow(encoder);
                }
            }

            // 1. Depth-only render for this cascade. Vertex transforms
            //    through `cascades[cascade].view_proj`; rasterizer
            //    fills `mesh_shadow_depth_views[cascade]`.
            let render_g0 = viewport
                .mesh_shadow_render_g0_bgs[cascade]
                .as_ref()
                .expect("mesh_shadow render g0 bg present after refresh");
            let q_render = self.profiler.begin_query(
                &format!("mesh_shadow_render[{cascade}]"),
                encoder,
            );
            {
                let mut rp = self.mesh_shadow_map.begin_render_pass(
                    encoder,
                    &viewport.mesh_shadow_depth_views[cascade],
                    None,
                );
                rp.set_pipeline(&self.mesh_shadow_map.render_pipeline);
                rp.set_bind_group(0, render_g0, &[]);
                // Glass-classify bg, borrowed after all per-cascade
                // `&mut viewport` work above is done. Same bg the
                // primary mesh raster + glass front/back use — the
                // shadow FS does the same `discard`-on-glass classify
                // so the opaque shadow map only has opaque casters.
                let glass_g2 = viewport
                    .mesh_glass_g2_bg
                    .as_ref()
                    .expect("mesh_glass g2 bg present after refresh");
                rp.set_bind_group(2, glass_g2, &[]);
                if collect_pipestats {
                    // Slots 6..10 = mesh_shadow_render[0..3].
                    rp.begin_pipeline_statistics_query(
                        &viewport.mesh_pipestats_query_set,
                        2 + crate::shadow_map_pass::CSM_CASCADE_COUNT + cascade as u32,
                    );
                }
                for (slot, d) in draws.iter().enumerate() {
                    let Some((vbo, ibo, lod0_index_count)) =
                        self.mesh_buffer(d.asset_handle_raw)
                    else {
                        continue;
                    };
                    if !slot_active[slot] {
                        continue;
                    }
                    let g1_bg = &viewport.splat_instance_bind_groups[slot];
                    rp.set_bind_group(1, g1_bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                    if direct_mode {
                        rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                    } else {
                        let cluster_count = self
                            .mesh_cluster_buffer(d.asset_handle_raw)
                            .map(|(_, c)| c)
                            .unwrap_or(0);
                        let max_draws = max_draws_override
                            .map(|n| n.min(cluster_count))
                            .unwrap_or(cluster_count);
                        let (args_buf, _) =
                            &viewport.mesh_lod_shadow_args_buffers[slot][cascade];
                        let count_buf =
                            &viewport.mesh_lod_shadow_count_buffers[slot][cascade];
                        rp.multi_draw_indexed_indirect_count(
                            args_buf, 0, count_buf, 0, max_draws,
                        );
                    }
                }
                if collect_pipestats {
                    rp.end_pipeline_statistics_query();
                }
            }
            self.profiler.end_query(encoder, q_render);

            // 1b. Glass shadow front + back. Captures glass entry +
            //     exit depth from this cascade's light POV. The
            //     shade pass reads both and applies Beer attenuation
            //     to the existing opaque CSM shadow factor.
            //     Per-instance `has_glass` filter — pure-opaque
            //     instances skip these passes entirely.
            let q_glass_front = self.profiler.begin_query(
                &format!("mesh_glass_shadow_front[{cascade}]"),
                encoder,
            );
            {
                let mut rp = self.mesh_glass_shadow.begin_front_pass(
                    encoder,
                    &viewport.mesh_glass_shadow_front_views[cascade],
                );
                rp.set_pipeline(&self.mesh_glass_shadow.front_pipeline);
                rp.set_bind_group(0, render_g0, &[]);
                let glass_g2 = viewport
                    .mesh_glass_g2_bg
                    .as_ref()
                    .expect("mesh_glass g2 bg present after refresh");
                rp.set_bind_group(2, glass_g2, &[]);
                for (slot, d) in draws.iter().enumerate() {
                    if !d.has_glass { continue; }
                    let Some((vbo, ibo, lod0_index_count)) =
                        self.mesh_buffer(d.asset_handle_raw)
                    else { continue; };
                    if !slot_active[slot] { continue; }
                    let g1_bg = &viewport.splat_instance_bind_groups[slot];
                    rp.set_bind_group(1, g1_bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                    if direct_mode {
                        rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                    } else {
                        let cluster_count = self
                            .mesh_cluster_buffer(d.asset_handle_raw)
                            .map(|(_, c)| c)
                            .unwrap_or(0);
                        let max_draws = max_draws_override
                            .map(|n| n.min(cluster_count))
                            .unwrap_or(cluster_count);
                        let (args_buf, _) =
                            &viewport.mesh_lod_shadow_args_buffers[slot][cascade];
                        let count_buf =
                            &viewport.mesh_lod_shadow_count_buffers[slot][cascade];
                        rp.multi_draw_indexed_indirect_count(
                            args_buf, 0, count_buf, 0, max_draws,
                        );
                    }
                }
            }
            self.profiler.end_query(encoder, q_glass_front);

            let q_glass_back = self.profiler.begin_query(
                &format!("mesh_glass_shadow_back[{cascade}]"),
                encoder,
            );
            {
                let mut rp = self.mesh_glass_shadow.begin_back_pass(
                    encoder,
                    &viewport.mesh_glass_shadow_back_views[cascade],
                );
                rp.set_pipeline(&self.mesh_glass_shadow.back_pipeline);
                rp.set_bind_group(0, render_g0, &[]);
                let glass_g2 = viewport
                    .mesh_glass_g2_bg
                    .as_ref()
                    .expect("mesh_glass g2 bg present after refresh");
                rp.set_bind_group(2, glass_g2, &[]);
                for (slot, d) in draws.iter().enumerate() {
                    if !d.has_glass { continue; }
                    let Some((vbo, ibo, lod0_index_count)) =
                        self.mesh_buffer(d.asset_handle_raw)
                    else { continue; };
                    if !slot_active[slot] { continue; }
                    let g1_bg = &viewport.splat_instance_bind_groups[slot];
                    rp.set_bind_group(1, g1_bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                    if direct_mode {
                        rp.draw_indexed(0..lod0_index_count, 0, 0..1);
                    } else {
                        let cluster_count = self
                            .mesh_cluster_buffer(d.asset_handle_raw)
                            .map(|(_, c)| c)
                            .unwrap_or(0);
                        let max_draws = max_draws_override
                            .map(|n| n.min(cluster_count))
                            .unwrap_or(cluster_count);
                        let (args_buf, _) =
                            &viewport.mesh_lod_shadow_args_buffers[slot][cascade];
                        let count_buf =
                            &viewport.mesh_lod_shadow_count_buffers[slot][cascade];
                        rp.multi_draw_indexed_indirect_count(
                            args_buf, 0, count_buf, 0, max_draws,
                        );
                    }
                }
            }
            self.profiler.end_query(encoder, q_glass_back);

            // 2. Blit compute — copy bitcast(depth) into the cascade's
            //    slice of `shadow_buffer`. Single thread per texel,
            //    full overwrite; no pre-clear needed (uncovered texels
            //    get the depth attachment's clear value of 1.0 =
            //    SHADOW_MAP_FAR_DEPTH_BITS after bitcast).
            let blit_g0 = viewport
                .mesh_shadow_blit_g0_bgs[cascade]
                .as_ref()
                .expect("mesh_shadow blit g0 bg present after refresh");
            let q_blit = self.profiler.begin_query(
                &format!("mesh_shadow_blit[{cascade}]"),
                encoder,
            );
            self.mesh_shadow_map.dispatch_blit(
                encoder,
                blit_g0,
                viewport.shadow_map.size,
            );
            self.profiler.end_query(encoder, q_blit);
        }
    }

    /// Current lights/materials epoch — ViewportRenderers compare against
    /// this to detect when their march/shade bindings have gone stale
    /// (shared buffer reallocated under them).
    pub fn lights_materials_epoch(&self) -> u64 {
        self.lights_materials_epoch
    }

    /// Grow `scene.bone_field_buffer` + `scene.bone_field_occ_buffer`
    /// to at least the requested sizes and clear both. Call once per
    /// frame before any scatter dispatches.
    pub fn prepare_bone_field(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        field_bytes: u64,
        occ_bytes: u64,
    ) {
        let field_bytes = field_bytes.max(16);
        let occ_bytes = occ_bytes.max(16);
        let _ = queue; // unused but matches other pass signatures
        let grew_field = self.scene.ensure_bone_field_capacity(&self.device, field_bytes);
        let grew_occ = self.scene.ensure_bone_field_occ_capacity(&self.device, occ_bytes);
        if grew_field || grew_occ {
            // New buffer handle(s) — rebuild the scatter's scene bind
            // group (the scene's main bind group was already rebuilt
            // inside `ensure_*_capacity`).
            self.skin_deform.refresh_scene_bind_group(&self.device, &self.scene);
        }
        // Clear — scattering leaves gaps by design.
        encoder.clear_buffer(&self.scene.bone_field_buffer, 0, None);
        encoder.clear_buffer(&self.scene.bone_field_occ_buffer, 0, None);
    }

    /// Run the batched skin-deform scatter. `batch` must have every
    /// skinned entity's dispatch folded in via `SkinBatchScratch::push`.
    /// Call once per frame after [`prepare_bone_field`]; fires a single
    /// compute dispatch, so there's no ordering problem with
    /// `queue.write_buffer` across entities.
    pub fn scatter_skin_batch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        batch: &crate::skin_deform::SkinBatchScratch,
    ) {
        self.skin_deform.run(&self.device, queue, encoder, &self.scene, batch);
    }

    pub fn upload_geometry(&mut self, queue: &wgpu::Queue, data: &GeometryUpload) {
        self.scene.upload_geometry(&self.device, queue, data);
        // upload_geometry may have grown brick_pool / bone_weights /
        // bone_matrices; the scatter pass's scene bind group caches
        // those resource references and needs a rebuild to pick up any
        // new buffer handles. Cheap: one bind group alloc.
        self.skin_deform.refresh_scene_bind_group(&self.device, &self.scene);
    }

    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        let prev_epoch = self.scene.buffers_epoch();
        self.scene.upload_frame(&self.device, queue, data);
        if self.scene.buffers_epoch() != prev_epoch {
            // `bone_matrices_buffer` and `bone_dual_quats_buffer` start
            // life as 64 B placeholders and grow on the first non-empty
            // upload. `skin_deform`'s scene bg references both
            // (bindings 5 + 11), so a realloc here invalidates it.
            // Matches the refresh in `upload_geometry`.
            self.skin_deform.refresh_scene_bind_group(&self.device, &self.scene);
        }
    }

    /// Render one frame into `viewport`. Dispatches into the VR's own
    /// per-resolution passes; in `Isolation` mode the atmosphere /
    /// shadow_trace / volumetric / god_rays / bloom passes are skipped
    /// to give a clean studio look.
    ///
    /// `lod_enabled` gates the prefiltered-LOD early-exit in the march;
    /// turn it off for A/B correctness comparison.
    /// `surfacenet_enabled` gates render-time normal reconstruction from
    /// the 3³ in-brick occupancy neighborhood — an A/B toggle for the
    /// Surface-Nets normal POC.
    /// `preview_mode` selects the primary-visibility pass: `Voxel` runs
    /// the usual octree march; `Raymarch` runs the procedural CSG
    /// raymarcher instead. Only the build viewport uses `Raymarch`;
    /// everywhere else passes `Voxel` (the default).
    #[allow(clippy::too_many_arguments)]
    pub fn render_to(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        object_count: u32,
        shadow_steps: u32,
        num_lights: u32,
        lod_enabled: bool,
        surfacenet_enabled: bool,
        tile_offsets: &[u8],
        tile_object_ids: &[u8],
        tile_count_x: u32,
        tlas_node_count: u32,
        // Phase B-redux Phase 3a — frame time + asset count. Threaded
        // through to march params for `instance_at` derivation.
        time: f32,
        asset_count: u32,
        // Phase 8 — TLAS prim count (one per shadow caster). The
        // shadow-map setup pass walks `tlas_prims[0..prim_count]`;
        // `tlas_node_count` is the BVH node count, which is up to
        // `2*prim_count - 1` and not what the setup needs.
        tlas_prim_count: u32,
        // Phase 8 — scene extent (max world dimension) used by the
        // setup pass to extrude per-prim AABBs along `light_dir`
        // for the shadow-frustum cull.
        scene_extent: f32,
        // Phase 8 — camera world→NDC matrix forwarded to the
        // shadow-map setup pass so it can shadow-frustum-cull
        // each prim against the visible region.
        camera_view_proj: [[f32; 4]; 4],
        // Phase 8 — when true, dispatch the shadow-map chain
        // (clear → setup → scatter) after primary visibility.
        // Engine sets this when there's a live directional shadow
        // caster (non-empty TLAS + shadow-casting directional
        // light); ShadeParams.shadow_map_enabled is in lockstep so
        // shade samples the fresh map.
        shadow_map_enabled: bool,
        atmo_frame_params: &crate::rkp_atmosphere::AtmosphereFrameParams,
        mode: crate::RenderMode,
        preview_mode: crate::BuildPreviewMode,
        // Phase B-2 — splat-raster instance list. Used only when
        // `self.primary_mode == PrimaryMode::Splat`; the march path
        // ignores it. One entry per visible scene-instance whose asset
        // has splat data (i.e. came from `acquire_asset`); procedural
        // objects without splats are skipped client-side.
        splat_draws: &[SplatDraw],
        // Procedural proxy-mesh draws (GPU surface-nets-from-SDF).
        // Rendered by `dispatch_proxy_meshes` after the primary mode
        // completes; composites into the G-buffer via depth-test.
        proxy_draws: &[crate::mesh_proxy_pass::ProxyDraw],
        // V1 mesh-path user-shader draws. One per active user-shader
        // material with painted anchors. Compute trio already ran
        // (in engine's `tick_user_shader_mesh`); this raster
        // consumes the indirect-args buffer the prefix_sum pass
        // wrote.
        user_shader_mesh_draws: &[crate::user_shader_mesh_pass::UserShaderMeshDraw],
        // Cursor pixel for the screen-space paint cursor — `Some` when
        // paint mode is active and the mouse sits inside the
        // framebuffer, `None` otherwise. Drives a single-thread
        // compute that captures gbuf_position + gbuf_pick at this
        // pixel into the per-VR `BrushState` buffer the shade pass
        // reads. `None` writes the miss sentinel.
        brush_pixel: Option<(u32, u32)>,
    ) {
        let in_situ = matches!(mode, crate::RenderMode::InSitu);
        let raymarch = matches!(preview_mode, crate::BuildPreviewMode::Raymarch);
        let splat = self.primary_mode == PrimaryMode::Splat;
        let mesh = self.primary_mode == PrimaryMode::Mesh;

        // Upload per-viewport tile-cull data. The per-object screen
        // AABBs feed the CPU-side tile-list builder; only the built
        // lists cross to the GPU now.
        viewport.march.upload_tile_lists(
            &self.device, queue, tile_offsets, tile_object_ids,
        );

        // 0. Atmosphere LUTs (in-situ only — isolation uses a flat sky).
        if in_situ {
            self.atmosphere.dispatch_if_dirty(encoder);
            let q = self.profiler.begin_query("atmo", encoder);
            self.atmosphere.dispatch_per_frame(encoder, queue, atmo_frame_params);
            self.profiler.end_query(encoder, q);
        }

        // 1. Primary visibility → G-buffer. Three mutually-exclusive
        //    paths: procedural raymarch (build viewport), splat raster
        //    (Phase B-2 A/B path, gated on `RKP_PRIMARY=splat`), or
        //    voxel march (default). Each fully populates the G-buffer
        //    including miss sentinels at non-hit pixels, so downstream
        //    passes are unaware of which path ran.
        //    Shadow_trace is skipped in raymarch mode because the
        //    procedural preview doesn't have the world-space voxel
        //    grid that pass needs; isolation mode already forces
        //    shadow=1.0 inside shade.
        if raymarch {
            let q = self.profiler.begin_query("proc_raymarch", encoder);
            viewport.proc_raymarch.dispatch(
                encoder, viewport.width, viewport.height, None,
            );
            self.profiler.end_query(encoder, q);
        } else if splat {
            self.dispatch_splat(queue, encoder, viewport, splat_draws);
        } else if mesh {
            // Mesh path consumes the same per-instance draw list as
            // splat — both record (asset_handle, world, object_id).
            self.dispatch_mesh(queue, encoder, viewport, splat_draws);
        } else {
            viewport.march.clear_stats(encoder);
            let q = self.profiler.begin_query("march", encoder);
            viewport.march.dispatch(
                encoder, queue, &viewport.scene_bind_group,
                object_count, viewport.width, viewport.height, 0,
                shadow_steps, num_lights, lod_enabled, surfacenet_enabled,
                tile_count_x, tlas_node_count,
                shadow_map_enabled, time, asset_count, None,
            );
            self.profiler.end_query(encoder, q);
        }

        // 1a'. Proxy-mesh raster — composites procedural triangle
        //      meshes onto the G-buffer regardless of primary mode.
        //      Skipped in raymarch preview (single-procedural focus
        //      target, no scene composition).
        if !raymarch {
            self.dispatch_proxy_meshes(queue, encoder, viewport, proxy_draws);
        }

        // 1a''. V1 mesh-path user-shader raster. One indirect draw
        //       per active user-shader material with painted
        //       anchors. Composites onto the G-buffer same way the
        //       proxy raster does.
        if !raymarch {
            self.dispatch_user_shader_mesh(encoder, viewport, user_shader_mesh_draws);
        }

        // 1b. Half-res shadow trace. Skipped in isolation — the shade
        // pass forces shadow=1.0 there. Uses march's params bind group.
        // Splat path skips this for the Phase B-2 prototype: the shadow
        // trace expects the world-space voxel grid + march's params bg
        // that the splat path doesn't provide. Splat A/B comparisons
        // therefore render shadow=1.0; document this when interpreting
        // visual diffs.
        if in_situ && !raymarch && !splat && !mesh {
            if let Some(params_bg) = viewport.march.params_bind_group() {
                let q = self.profiler.begin_query("shadow", encoder);
                viewport.shadow_trace.dispatch(encoder, &viewport.scene_bind_group, params_bg);
                self.profiler.end_query(encoder, q);
            }
        }

        // 1c. Phase 8 — directional shadow map. Same in-situ/non-
        // raymarch gate as shadow_trace; the engine flips
        // `shadow_map_enabled` based on whether a directional
        // caster + non-empty TLAS exists this frame. The shade
        // pass reads the resulting depth texture for directional
        // visibility; non-directional lights still pull from the
        // half-res shadow_trace output.
        if in_situ && !raymarch && !splat && !mesh && shadow_map_enabled {
            let q = self.profiler.begin_query("shadow_map", encoder);
            viewport.shadow_map.dispatch_clear(encoder);
            viewport.shadow_map.dispatch_setup(
                encoder, queue, tlas_prim_count, camera_view_proj, scene_extent,
            );
            viewport.shadow_map.dispatch_emit(encoder, tlas_prim_count);
            viewport.shadow_map.dispatch_finalize(encoder);
            viewport.shadow_map.dispatch_scatter(
                encoder, &viewport.scene_bind_group, tlas_prim_count,
            );
            self.profiler.end_query(encoder, q);
        }
        // Mesh-mode directional shadow map: real triangle rasterization
        // from the light's POV into the same `shadow_buffer` shade
        // already samples. Per-instance uniforms were written by the
        // earlier `dispatch_mesh` call this frame.
        if in_situ && !raymarch && mesh && shadow_map_enabled {
            self.dispatch_mesh_shadow(queue, encoder, viewport, splat_draws);
        }

        // Pipeline-statistics resolve: single per-frame point so it
        // fires regardless of which mesh passes (primary / shadow /
        // both) ran. Each pass writes to its own slot; unwritten
        // slots come back as undefined data (caller can detect
        // by all-zero results or just trust the slot it cares about).
        if mesh && std::env::var("RKP_MESH_PIPESTATS").is_ok() {
            viewport.pipestats_finalize(encoder);
        }

        if !raymarch && !splat && !mesh {
            viewport.march.copy_stats(encoder);
        }

        // 2. SSAO (half-res). Kept in isolation — it's the only grounding cue.
        {
            let q = self.profiler.begin_query("ssao", encoder);
            viewport.ssao.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 2b. Brush-state probe — captures `(world_pos, hit_object_id)`
        // at the cursor pixel into the per-VR `BrushState` buffer
        // for the screen-space paint cursor in shade. Always
        // dispatches: `None` writes the miss sentinel so the cursor
        // hides without any extra gating, and the cost (1 thread)
        // is below noise.
        {
            let params = crate::brush_state_pass::BrushParams {
                cursor_x: brush_pixel.map(|(x, _)| x).unwrap_or(0),
                cursor_y: brush_pixel.map(|(_, y)| y).unwrap_or(0),
                enabled: brush_pixel.is_some() as u32,
                _pad0: 0,
            };
            queue.write_buffer(
                &viewport.brush_state_params_buffer,
                0,
                bytemuck::bytes_of(&params),
            );
            if let Some(bg) = viewport.brush_state_pass_bg.as_ref() {
                self.brush_state.dispatch(encoder, bg);
            }
        }

        // 3. Deferred PBR shading. ShadeParams.isolation drives the
        // isolation-mode behavior inside the shader (flat sky, fixed
        // ambient, shadow=1).
        {
            let q = self.profiler.begin_query("shade", encoder);
            viewport.shade.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4. Volumetric march + composite (in-situ only). Runs
        // before glass so clouds / fog land in the "behind" HDR
        // and are refracted / Beer-tinted through any glass in
        // front of them, rather than stamping over the glass
        // composite.
        if in_situ {
            let q = self.profiler.begin_query("vol", encoder);
            // Fog + cloud are separate passes now. Fog runs over every pixel
            // with only fog bindings; cloud runs over sky tiles with its own
            // bindings. Keeping them split avoids the marker-bleed artefact
            // the old combined shader produced when the hardware bilinear
            // sampler blended the history validity sentinel across sky/voxel
            // boundaries.
            viewport.volumetric.dispatch_fog_march(encoder);
            viewport.volumetric.dispatch_cloud_march(encoder);
            viewport.volumetric.update_history(encoder);
            viewport.volumetric.dispatch_sun_atten(encoder);
            viewport.volumetric.dispatch_composite(encoder);
            self.profiler.end_query(encoder, q);
        } else {
            // Isolation: keep the texture chain valid by copying shade
            // output forward into volumetric.output (the texture
            // god_rays' input view is bound to). Cheaper than rebuilding
            // god_rays' bind group on every mode flip.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &viewport.shade.output_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &viewport.volumetric.output_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: viewport.width, height: viewport.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // 4a. Glass composite — reads the (volumetric-composited)
        // HDR + gbuf_glass, applies Fresnel + Beer + screen-space
        // refraction for any pixel whose primary ray passed through
        // transparent voxels, writes to its own HDR target.
        // Downstream god_rays sources from `glass.output_view`.
        {
            let q = self.profiler.begin_query("glass", encoder);
            viewport.glass.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // 4b. God rays (in-situ only). Isolation copies the glass
        // output forward into god_rays.output so bloom_composite's HDR
        // input is correct.
        if in_situ {
            let q = self.profiler.begin_query("god_rays", encoder);
            viewport.god_rays.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        } else {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &viewport.glass.output_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &viewport.god_rays.output_texture,
                    mip_level: 0, origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: viewport.width, height: viewport.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // 5. Bloom (in-situ only). bloom_composite + tone_map always run
        // because tone_map is the only HDR→LDR step. In isolation the
        // engine writes bloom_intensity=0 per-VR so bloom_composite's
        // mip read is zero-weighted — the pass becomes a copy from its
        // HDR input (which we just populated with shade output above).
        if in_situ {
            let q = self.profiler.begin_query("bloom", encoder);
            viewport.bloom.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }
        {
            let q = self.profiler.begin_query("bloom_composite", encoder);
            viewport.bloom_composite.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }
        {
            let q = self.profiler.begin_query("tone_map", encoder);
            viewport.tone_map.dispatch(encoder);
            self.profiler.end_query(encoder, q);
        }

        // Copy LDR into composite for any overlay passes (wireframe) the caller runs.
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: viewport.tone_map.ldr_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &viewport.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: viewport.width,
                height: viewport.height,
                depth_or_array_layers: 1,
            },
        );

        // Isolation: paint the infinite grid over the composite. Done
        // after the LDR copy so the grid blends in display-space rather
        // than competing with HDR scene radiance.
        if !in_situ {
            let q = self.profiler.begin_query("grid", encoder);
            viewport.grid.draw(encoder, &viewport.composite_view);
            self.profiler.end_query(encoder, q);
        }

        // Raymarch preview: stamp the selected-primitive outline on top
        // of the composite (and on top of the grid, so the outline
        // wins visually when a primitive sits against the floor).
        // Self-discarding when no node is selected, so this is cheap
        // whether or not the user is currently pointing at something.
        if raymarch {
            let q = self.profiler.begin_query("proc_outline", encoder);
            viewport.proc_outline.draw(encoder, &viewport.composite_view);
            self.profiler.end_query(encoder, q);

            // Ghost cutters, drawn after the outline so the outline's
            // thin opaque band still wins at the silhouette; ghosts
            // fill-in the carved-away volume behind it.
            let q = self.profiler.begin_query("proc_ghost", encoder);
            viewport.proc_ghost.draw(encoder, &viewport.composite_view);
            self.profiler.end_query(encoder, q);
        }
    }

    pub fn resolve_profiler_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.profiler.resolve_queries(encoder);
    }

    /// End the GPU profiler frame and drain any finished samples.
    ///
    /// Returns `(label, ms)` for each top-level pass that wgpu-profiler
    /// finished resolving this frame — empty during the first ~3-frame
    /// warmup and on frames where the query pool isn't ready yet.
    /// Callers are expected to feed this into `ProfilingHistory`.
    pub fn end_profiler_frame(&mut self, frame_idx: u64) -> Vec<(String, f32)> {
        if let Err(e) = self.profiler.end_frame() {
            if frame_idx > 10 {
                eprintln!("[profiler] end_frame: {e}");
            }
        }
        let Some(results) = self.profiler.process_finished_frame(self.timestamp_period) else {
            return Vec::new();
        };
        results
            .iter()
            .map(|r| {
                let ms = r
                    .time
                    .as_ref()
                    .map(|t| ((t.end - t.start) * 1000.0) as f32)
                    .unwrap_or(0.0);
                (r.label.clone(), ms)
            })
            .collect()
    }

    pub fn update_shade_params(&self, queue: &wgpu::Queue, params: &ShadeParams) {
        queue.write_buffer(&self.shade_params_buffer, 0, bytemuck::bytes_of(params));
    }

    pub fn update_lights(&mut self, queue: &wgpu::Queue, lights: &[GpuLight]) {
        let data: &[u8] = bytemuck::cast_slice(lights);
        let needed = data.len() as u64;
        // Track our own capacity instead of `buffer.size()`. wgpu's
        // validator can carry a stale-feeling size into its error path
        // when the buffer is recreated while bind groups still hold
        // refs to the old `Arc`. The capacity field sidesteps it.
        if needed > self.lights_capacity {
            self.lights_buffer = Self::create_init_buffer(
                &self.device,
                "rkp_shade_lights",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                data,
            );
            self.lights_capacity = self.lights_buffer.size();
            self.lights_materials_epoch += 1;
        } else {
            queue.write_buffer(&self.lights_buffer, 0, data);
        }
    }

    pub fn update_materials(&mut self, queue: &wgpu::Queue, materials: &[GpuMaterial]) {
        let data: &[u8] = bytemuck::cast_slice(materials);
        let needed = data.len() as u64;
        if needed > self.materials_capacity {
            self.materials_buffer = Self::create_init_buffer(
                &self.device,
                "rkp_shade_materials",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                data,
            );
            self.materials_capacity = self.materials_buffer.size();
            self.lights_materials_epoch += 1;
        } else {
            queue.write_buffer(&self.materials_buffer, 0, data);
        }
    }

    fn create_init_buffer(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages, data: &[u8]) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().copy_from_slice(data);
        buf.unmap();
        buf
    }
}
