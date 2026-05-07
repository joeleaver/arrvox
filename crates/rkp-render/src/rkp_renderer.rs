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
    /// Primary-visibility selector — `March` (compute octree march)
    /// or `Splat` (rasterized surface splats). Read from the
    /// `RKP_PRIMARY` env var at construction. See [`PrimaryMode`].
    pub primary_mode: PrimaryMode,
    /// Device for buffer operations.
    pub device: wgpu::Device,
    /// GPU profiler (wgpu-profiler).
    pub profiler: GpuProfiler,
    timestamp_period: f32,
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
        let mesh_pass = MeshPass::new(device, &splat_pass.g0_layout, &splat_pass.g1_layout);
        let mesh_shadow_map = MeshShadowMapPass::new(device, &splat_pass.g1_layout);
        let mesh_lod_select_pass =
            crate::mesh_lod_select_pass::MeshLodSelectPass::new(device, &splat_pass.g1_layout);
        let splat_resolve = SplatResolvePass::new(device);
        let primary_mode = PrimaryMode::from_env();
        eprintln!("[RkpRenderer] primary_mode = {primary_mode:?}");

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
            splat_buffers: Vec::new(),
            mesh_pass,
            mesh_shadow_map,
            mesh_lod_select_pass,
            mesh_buffers: Vec::new(),
            mesh_cluster_buffers: Vec::new(),
            primary_mode,
            device: device.clone(),
            profiler, timestamp_period,
        }
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
            viewport.write_splat_instance(queue, slot as u32, &d.world, d.object_id);
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
        // Diagnostic: skip primary visibility entirely. Used to
        // measure shadow_render in isolation when chasing the
        // anti-correlation pattern between the two passes.
        // Visuals: composite stays as last-frame output; shade
        // sees a sky-cleared G-buffer.
        if std::env::var("RKP_MESH_DISABLE_PRIMARY").is_ok() {
            return;
        }
        viewport.refresh_splat_g0(&self.device, self);
        viewport.refresh_splat_resolve_bindings(&self.device, self);
        viewport.ensure_splat_instance_capacity(&self.device, self, draws.len() as u32);
        viewport.ensure_mesh_lod_capacity(&self.device, self, draws.len() as u32);
        for (slot, d) in draws.iter().enumerate() {
            viewport.write_splat_instance(queue, slot as u32, &d.world, d.object_id);
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
        let force_admit_flag: u32 = if std::env::var("RKP_MESH_DEBUG_FORCE_ADMIT").is_ok() {
            1
        } else {
            0
        };
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
                let bg = self.mesh_lod_select_pass.create_g2_bind_group(
                    &self.device,
                    cluster_buf,
                    args_buf,
                    &viewport.mesh_lod_admit_stats_primary,
                );
                viewport.mesh_lod_select_g2_bgs[slot] = Some((bg, asset_handle_raw, *args_cap));
            }

            // Per-draw uniform — admit threshold + cluster count.
            let params = crate::mesh_lod_select_pass::MeshLodSelectParams {
                pixel_threshold: pixel_threshold_primary,
                cluster_count,
                force_admit: force_admit_flag,
                record_stats: lod_stats_enabled as u32,
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
        if !direct_mode {
            // Stats lifecycle (RKP_MESH_LOD_STATS=1 only):
            // (a) drain previous frame's mapped buffer if ready
            // (b) clear the histogram for this frame's atomics
            // (c) dispatch — atomics fire iff record_stats != 0
            // (d) copy histogram → staging + map_async for next frame
            if lod_stats_enabled {
                viewport.lod_stats_drain_primary("primary");
                viewport.lod_stats_clear_primary(encoder);
            }

            let q_lod = self.profiler.begin_query("mesh_lod_select", encoder);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mesh_lod_select"),
                    timestamp_writes: None,
                });
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

        let g0_bg = viewport
            .splat_g0_bg
            .as_ref()
            .expect("splat g0 bg present after refresh_splat_g0");

        // 1. Visibility-buffer raster — same RT layout as the splat
        //    pass; clears use the same march-equivalent miss sentinels.
        let q_raster = self.profiler.begin_query("mesh_raster", encoder);
        {
            let mut rp = self.mesh_pass.begin_pass(
                encoder,
                &viewport.gbuffer.position_view,
                &viewport.pick_view,
                &viewport.gbuffer.leaf_slot_view,
                &viewport.gbuffer.depth_view,
                None,
            );
            rp.set_pipeline(&self.mesh_pass.pipeline);
            rp.set_bind_group(0, g0_bg, &[]);
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
                    rp.multi_draw_indexed_indirect(args_buf, 0, max_draws);
                }
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
        viewport.ensure_mesh_lod_shadow_capacity(&self.device, self, draws.len() as u32);

        // `RKP_MESH_DEBUG_DIRECT=1` also bypasses the shadow LOD
        // compute + indirect dispatch — depth-only `draw_indexed` of
        // the LOD-0 prefix, just like the Phase 1-3 baseline.
        let direct_mode = std::env::var("RKP_MESH_DEBUG_DIRECT").is_ok();

        // Phase 6.4: per-draw shadow LOD-select prep. Same shape as
        // the primary path in `dispatch_mesh`, but with a doubled
        // pixel threshold so the Karis admit rule effectively picks
        // `lod + 1` from the same chain that primary uses.
        //
        // `RKP_MESH_SHADOW_LOD_THRESHOLD` overrides the compile-time
        // default. Mirrors `RKP_MESH_LOD_THRESHOLD` for the primary path.
        const PIXEL_THRESHOLD_SHADOW: f32 = 2.0;
        let pixel_threshold_shadow = pixel_threshold_env(
            "RKP_MESH_SHADOW_LOD_THRESHOLD",
            PIXEL_THRESHOLD_SHADOW,
        );
        let lod_stats_enabled = std::env::var("RKP_MESH_LOD_STATS").is_ok();
        let mut slot_active: Vec<bool> = vec![false; draws.len()];
        for (slot, d) in draws.iter().enumerate() {
            if self.mesh_buffer(d.asset_handle_raw).is_none() {
                continue;
            }
            let Some((cluster_buf, cluster_count)) =
                self.mesh_cluster_buffer(d.asset_handle_raw)
            else {
                continue;
            };
            if cluster_count == 0 {
                continue;
            }
            let asset_handle_raw = d.asset_handle_raw;

            viewport.ensure_mesh_lod_shadow_args_capacity(
                &self.device,
                slot as u32,
                cluster_count,
            );

            let (args_buf, args_cap) = &viewport.mesh_lod_shadow_args_buffers[slot];
            let need_rebuild = match &viewport.mesh_lod_shadow_g2_bgs[slot] {
                Some((_, cached_handle, cached_cap)) => {
                    *cached_handle != asset_handle_raw || *cached_cap != *args_cap
                }
                None => true,
            };
            if need_rebuild {
                let bg = self.mesh_lod_select_pass.create_g2_bind_group(
                    &self.device,
                    cluster_buf,
                    args_buf,
                    &viewport.mesh_lod_admit_stats_shadow,
                );
                viewport.mesh_lod_shadow_g2_bgs[slot] = Some((bg, asset_handle_raw, *args_cap));
            }

            let params = crate::mesh_lod_select_pass::MeshLodSelectParams {
                pixel_threshold: pixel_threshold_shadow,
                cluster_count,
                force_admit: if std::env::var("RKP_MESH_DEBUG_FORCE_ADMIT").is_ok() {
                    1
                } else {
                    0
                },
                record_stats: lod_stats_enabled as u32,
            };
            queue.write_buffer(
                &viewport.mesh_lod_shadow_params_buffers[slot],
                0,
                bytemuck::bytes_of(&params),
            );

            slot_active[slot] = true;
        }

        // 0. Shadow-side LOD-select compute pass.
        if !direct_mode {
            if lod_stats_enabled {
                viewport.lod_stats_drain_shadow("shadow");
                viewport.lod_stats_clear_shadow(encoder);
            }

            let q_lod = self.profiler.begin_query("mesh_shadow_lod_select", encoder);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mesh_shadow_lod_select"),
                    timestamp_writes: None,
                });
                for (slot, d) in draws.iter().enumerate() {
                    if !slot_active[slot] {
                        continue;
                    }
                    let cluster_count = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    let g0 = &viewport.mesh_lod_shadow_g0_bgs[slot];
                    let g1 = &viewport.splat_instance_bind_groups[slot];
                    let g2 = &viewport
                        .mesh_lod_shadow_g2_bgs[slot]
                        .as_ref()
                        .expect("g2 set above for active slot")
                        .0;
                    self.mesh_lod_select_pass
                        .dispatch(&mut cpass, g0, g1, g2, cluster_count);
                }
            }
            self.profiler.end_query(encoder, q_lod);

            if lod_stats_enabled {
                viewport.lod_stats_finalize_shadow(encoder);
            }
        }

        // 1. Depth-only render. Vertex transforms through light_camera
        //    view-proj; rasterizer fills the depth attachment. No
        //    fragment shader, so the GPU's early-z runs at full speed.
        let render_g0 = viewport
            .mesh_shadow_render_g0_bg
            .as_ref()
            .expect("mesh_shadow render g0 bg present after refresh");
        let q_render = self.profiler.begin_query("mesh_shadow_render", encoder);
        {
            let mut rp = self.mesh_shadow_map.begin_render_pass(
                encoder,
                &viewport.mesh_shadow_depth_view,
                None,
            );
            rp.set_pipeline(&self.mesh_shadow_map.render_pipeline);
            rp.set_bind_group(0, render_g0, &[]);
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
                    let (args_buf, _) = &viewport.mesh_lod_shadow_args_buffers[slot];
                    rp.multi_draw_indexed_indirect(args_buf, 0, max_draws);
                }
            }
        }
        self.profiler.end_query(encoder, q_render);

        // 2. Blit compute — copy bitcast(depth) into shadow_buffer.
        //    Single thread per texel, full overwrite; no need to
        //    pre-clear shadow_buffer because every texel is written
        //    (uncovered ones get the depth attachment's clear value
        //    of 1.0 = SHADOW_MAP_FAR_DEPTH_BITS after bitcast).
        let blit_g0 = viewport
            .mesh_shadow_blit_g0_bg
            .as_ref()
            .expect("mesh_shadow blit g0 bg present after refresh");
        let q_blit = self.profiler.begin_query("mesh_shadow_blit", encoder);
        self.mesh_shadow_map.dispatch_blit(encoder, blit_g0);
        self.profiler.end_query(encoder, q_blit);
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
        if !raymarch && !splat && !mesh {
            viewport.march.copy_stats(encoder);
        }

        // 2. SSAO (half-res). Kept in isolation — it's the only grounding cue.
        {
            let q = self.profiler.begin_query("ssao", encoder);
            viewport.ssao.dispatch(encoder);
            self.profiler.end_query(encoder, q);
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
