//! Shared GPU state + orchestration for per-viewport rendering.
//!
//! [`RkpRenderer`] holds the scene-wide buffers (scene, atmosphere LUTs,
//! shade params, lights, materials) plus the GPU profiler. The
//! resolution-coupled passes (mesh raster, mesh_resolve, ssao, shade,
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
use crate::mesh_instance::{MeshDraw, MeshInstanceLayouts};
use crate::mesh_resolve_pass::MeshResolvePass;
use wgpu_profiler::GpuProfiler;

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

/// PERF_DEBT.md D4 hash gate. Tiny buffers (material palette ~1 KB,
/// lights ~2 KB, per-VR shader_params ~32 B × material count) are
/// shipped from sim to render every frame even when their content
/// hasn't changed; hashing the bytes and skipping the
/// `queue.write_buffer` when the hash matches the previous upload is
/// a cheap win across the steady state.
///
/// `0` is reserved as a "never uploaded" sentinel by the callers — if
/// content legitimately hashes to zero we just upload one extra time
/// on first sight; the next match short-circuits.
#[inline]
pub(crate) fn d4_hash_bytes(data: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(data);
    h.finish()
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
    /// PERF_DEBT.md D4: hash of the last `update_lights` payload.
    /// `update_lights` short-circuits when the incoming hash matches —
    /// the sim ships the full Vec every tick but most ticks have
    /// identical lights, so the per-tick `queue.write_buffer` was
    /// pure waste. `0` is a sentinel "never uploaded yet" so the
    /// first call always writes.
    last_lights_hash: u64,
    /// PERF_DEBT.md D4: same shape as [`Self::last_lights_hash`] for
    /// the material palette upload.
    last_materials_hash: u64,
    /// Per-instance bind-group layouts (g0 scene-wide + g1 per-instance).
    /// Shared across the mesh raster, mesh shadow render, mesh LOD
    /// select compute, mesh glass, and user-shader mesh paths.
    pub mesh_instance: MeshInstanceLayouts,
    /// Mesh-resolve compute fixup. Reads the visibility-buffer triplet
    /// `mesh_pass` writes and fills in the remaining G-buffer entries
    /// (normal / material / glass). One pipeline shared across
    /// viewports.
    pub mesh_resolve: MeshResolvePass,
    /// Procedural proxy-mesh raster pipeline (GPU surface-nets-from-
    /// SDF). Writes the full G-buffer for proxy pixels directly,
    /// bypassing `mesh_resolve`. One pipeline shared across viewports.
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
    /// Surface-mesh raster pipeline. Shares `g0_layout` / `g1_layout`
    /// with `mesh_instance` (the shared layout container).
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
    /// Per-asset vertex/index buffer cache for the mesh path. Each
    /// entry tracks the dispatch index count plus the byte count
    /// already uploaded to each buffer; the upload path uses these to
    /// emit tail-only `queue.write_buffer` calls when the asset only
    /// appended new mesh data (R4c-V2 sculpt path is append-only).
    /// Cleared on `release_mesh_for_asset`.
    mesh_buffers: Vec<Option<MeshBuffersEntry>>,
    /// Per-asset meshlet cluster table on the GPU (Phase 5).
    /// `(buffer, cluster_count)`; the buffer holds a flat
    /// `[MeshletCluster]` array uploaded via `cast_slice` and is
    /// bound as STORAGE for the Phase 6 LOD-selection compute pass.
    /// Phase 5 uploads but does not yet consume — validates the
    /// upload path without touching the hot dispatch.
    mesh_cluster_buffers: Vec<Option<(wgpu::Buffer, u32)>>,
    /// `RKP_RASTER_DIAG=1` per-asset cluster aggregate counts captured
    /// at the most-recent `upload_mesh_clusters_for_asset` call.
    /// Indexed by raw asset handle. `None` when no upload has happened
    /// or the diag flag isn't set. Used by `dispatch_mesh`'s
    /// `RKP_RASTER_DIAG=1` print to surface the per-LOD admit shape
    /// of what's actually being rasterized this frame without paying
    /// the per-frame scan cost when diag is off.
    mesh_cluster_diag: Vec<Option<MeshClusterDiag>>,
    /// Per-asset vertex/index buffer cache for the procedural
    /// proxy-mesh path. Separate from `mesh_buffers` because the
    /// proxy vertex layout is `ProxyVertex` (32 B; material + color
    /// payload) — not `MeshVertex`. Indexed by `AssetHandle::raw()`.
    proxy_mesh_buffers: Vec<Option<(wgpu::Buffer, wgpu::Buffer, u32)>>,
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

/// `RKP_RASTER_DIAG=1` aggregated cluster-state snapshot captured at
/// upload time. Mirrors the load-time `[raster_diag load]` line but
/// reflects whatever is currently on the GPU (post-sculpt etc.). Per-LOD
/// vectors are sized to `max_lod + 1`.
#[derive(Debug, Clone, Default)]
struct MeshClusterDiag {
    clusters_per_lod: Vec<u32>,
    indices_per_lod: Vec<u32>,
    lod_dirty_per_lod: Vec<u32>,
    /// "Post-bake patch" heuristic: LOD-0 cluster with no DAG membership
    /// on either side AND a leaf+root error pair. v5 files (no DAG
    /// topology) light up every cluster under this rule, so the print
    /// site qualifies the number with `dag_present` from the cluster
    /// buffer's first entry's `group_above_idx != 0` heuristic — good
    /// enough for diagnosing the regression.
    patch_count: u32,
    lod_dirty_total: u32,
    /// Total `index_count` across every cluster (= triangle count × 3).
    /// Compare against the actually-admitted index count surfaced in
    /// `RKP_MESH_LOD_STATS` to see how much LOD culling is happening.
    total_indices: u32,
}

/// Per-asset mesh buffer cache state. Tracks how many vertex bytes
/// have already been uploaded so the next `upload_mesh_for_asset` call
/// can emit a tail-only `queue.write_buffer` for the appended VBO
/// region. The IBO no longer uses a "uploaded bytes" cursor — the slab
/// allocator over `mesh_indices` does interior writes, so partial IBO
/// uploads are driven by per-asset `DirtyRanges` instead (see
/// `AssetEntry::mesh_indices_dirty`).
struct MeshBuffersEntry {
    vbo: wgpu::Buffer,
    ibo: wgpu::Buffer,
    dispatch_index_count: u32,
    /// Bytes of vertex data already written to `vbo`. The next upload
    /// streams `vertices_bytes[vbo_uploaded_bytes..]` at this offset.
    vbo_uploaded_bytes: u64,
}

/// Allocate a fresh GPU buffer sized for `data` (with 2× headroom over
/// an existing size if any) and write `data` into it. Used by D6's
/// tail-only mesh upload path on the realloc fallback — the existing
/// buffer is too small (CPU side overshot the GPU capacity) or the
/// CPU side shrunk (we have to rewrite everything to invalidate the
/// stale tail).
fn grow_with_full_upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    existing_size: Option<u64>,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
) -> wgpu::Buffer {
    let needed = data.len() as u64;
    let new_size = needed
        .max(existing_size.unwrap_or(0).saturating_mul(2))
        .max(64);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: new_size,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, data);
    buffer
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

        let mesh_instance = MeshInstanceLayouts::new(device);
        // `MeshGlassPass` owns the shared `g2_layout` (glass-classify
        // bindings) used by both the primary mesh raster and the
        // glass front/back rasters. Construct it first so the layout
        // can flow into `MeshPass::new`.
        let mesh_glass = crate::mesh_glass_pass::MeshGlassPass::new(
            device,
            &mesh_instance.g0_layout,
            &mesh_instance.g1_layout,
        );
        let mesh_pass = MeshPass::new(
            device,
            &mesh_instance.g0_layout,
            &mesh_instance.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_shadow_map = MeshShadowMapPass::new(
            device,
            &mesh_instance.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_glass_shadow = crate::mesh_glass_shadow_pass::MeshGlassShadowPass::new(
            device,
            &mesh_shadow_map.render_g0_layout,
            &mesh_instance.g1_layout,
            &mesh_glass.g2_layout,
        );
        let mesh_lod_select_pass =
            crate::mesh_lod_select_pass::MeshLodSelectPass::new(device, &mesh_instance.g1_layout);
        let mesh_resolve = MeshResolvePass::new(device);
        let mesh_proxy = crate::mesh_proxy_pass::MeshProxyPass::new(device);
        let user_shader_mesh = crate::user_shader_mesh_pass::UserShaderMeshPass::new(
            device,
            &scene.bind_group_layout,
        );
        let brush_state = crate::brush_state_pass::BrushStatePass::new(device);

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
            last_lights_hash: 0,
            last_materials_hash: 0,
            mesh_instance,
            mesh_resolve,
            mesh_proxy,
            user_shader_mesh,
            brush_state,
            mesh_pass,
            mesh_shadow_map,
            mesh_lod_select_pass,
            mesh_glass,
            mesh_glass_shadow,
            mesh_glass_debug_force,
            mesh_buffers: Vec::new(),
            mesh_cluster_buffers: Vec::new(),
            mesh_cluster_diag: Vec::new(),
            proxy_mesh_buffers: Vec::new(),
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
        queue: &wgpu::Queue,
        handle_raw: u32,
        vertices: &[MeshVertex],
        indices: &[u32],
        indices_dirty: &rkp_core::DirtyRanges,
        dispatch_index_count: u32,
    ) -> u64 {
        let idx = handle_raw as usize;
        if idx >= self.mesh_buffers.len() {
            self.mesh_buffers.resize_with(idx + 1, || None);
        }
        if vertices.is_empty() || indices.is_empty() || dispatch_index_count == 0 {
            self.mesh_buffers[idx] = None;
            return 0;
        }

        let vbo_bytes: &[u8] = bytemuck::cast_slice(vertices);
        let ibo_bytes: &[u8] = bytemuck::cast_slice(indices);
        let vbo_needed = vbo_bytes.len() as u64;
        let ibo_needed = ibo_bytes.len() as u64;
        let mut bytes_written: u64 = 0;

        // Take ownership of the existing entry (if any) so we can decide
        // tail-only vs full upload below.
        let existing = self.mesh_buffers[idx].take();
        let (vbo, vbo_uploaded_bytes) = if let Some(e) = existing.as_ref() {
            if e.vbo.size() >= vbo_needed && vbo_needed >= e.vbo_uploaded_bytes {
                // Buffer fits and the CPU side only grew → tail-only.
                // Also handles the `equal` case (skip the write entirely).
                // VBO stays tail-only: sculpt only appends new patch
                // verts to `mesh_vertices`, never rewrites interior.
                if vbo_needed > e.vbo_uploaded_bytes {
                    let tail = &vbo_bytes[e.vbo_uploaded_bytes as usize..];
                    queue.write_buffer(&e.vbo, e.vbo_uploaded_bytes, tail);
                    bytes_written += tail.len() as u64;
                }
                (e.vbo.clone(), vbo_needed)
            } else {
                // CPU side shrunk OR exceeded buffer capacity → reset.
                let buf = grow_with_full_upload(
                    &self.device, queue,
                    Some(e.vbo.size()),
                    "mesh asset vbo",
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    vbo_bytes,
                );
                bytes_written += vbo_needed;
                (buf, vbo_needed)
            }
        } else {
            let buf = grow_with_full_upload(
                &self.device, queue,
                None,
                "mesh asset vbo",
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                vbo_bytes,
            );
            bytes_written += vbo_needed;
            (buf, vbo_needed)
        };

        // IBO upload is driven by `indices_dirty` instead of tail-only.
        // The slab allocator over `mesh_indices` (see AssetEntry) does
        // interior writes when it reuses freed slots — a pure
        // `[uploaded_bytes..]` tail write would silently drop those.
        // Dirty ranges record exactly which byte ranges changed; iterate
        // them and issue one `queue.write_buffer` per range. The full
        // re-upload path (initial alloc or grow) still writes the whole
        // buffer; the dirty tracker is cleared on the scene-manager side
        // by `mark_loaded_asset_uploads_clean` after this call returns.
        let ibo = if let Some(e) = existing.as_ref() {
            if e.ibo.size() >= ibo_needed {
                for (off, len) in indices_dirty.iter() {
                    let start = off as usize;
                    let end = start + len as usize;
                    if end <= ibo_bytes.len() {
                        queue.write_buffer(&e.ibo, off as u64, &ibo_bytes[start..end]);
                        bytes_written += len as u64;
                    }
                }
                e.ibo.clone()
            } else {
                // Capacity exceeded → grow + full upload. The dirty
                // ranges become irrelevant: every byte just got written
                // by `grow_with_full_upload`.
                let buf = grow_with_full_upload(
                    &self.device, queue,
                    Some(e.ibo.size()),
                    "mesh asset ibo (full DAG)",
                    wgpu::BufferUsages::INDEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    ibo_bytes,
                );
                bytes_written += ibo_needed;
                buf
            }
        } else {
            let buf = grow_with_full_upload(
                &self.device, queue,
                None,
                "mesh asset ibo (full DAG)",
                wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                ibo_bytes,
            );
            bytes_written += ibo_needed;
            buf
        };

        self.mesh_buffers[idx] = Some(MeshBuffersEntry {
            vbo, ibo, dispatch_index_count,
            vbo_uploaded_bytes,
        });
        bytes_written
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
            .map(|e| (&e.vbo, &e.ibo, e.dispatch_index_count))
    }

    /// Upload (or replace) the meshlet cluster table for an asset
    /// (Phase 5). Caller passes `AssetHandle::raw()` and the
    /// `&[MeshletCluster]` slice from
    /// `RkpSceneManager::iter_loaded_asset_clusters`.
    ///
    /// Reuses the existing GPU buffer in place via `queue.write_buffer`
    /// when capacity fits; grows with 2× headroom otherwise. Sculpt's
    /// R4c-V2 path rewrites cluster entries in place (filter-and-patch),
    /// so the full slice is written every call — cheap because total
    /// table size is ~1.6 MiB.
    ///
    /// Returns `true` when the underlying `wgpu::Buffer` object was
    /// replaced (initial alloc, grow, or empty-clear of a present slot).
    /// Callers use this to decide whether downstream bind groups that
    /// reference the cluster buffer (`mesh_lod_select_g2_bgs`,
    /// `mesh_lod_shadow_g2_bgs`) must be invalidated. Returns `false`
    /// when the same buffer was reused — bind groups stay valid.
    pub fn upload_mesh_clusters_for_asset(
        &mut self,
        queue: &wgpu::Queue,
        handle_raw: u32,
        clusters: &[MeshletCluster],
    ) -> bool {
        let idx = handle_raw as usize;
        if idx >= self.mesh_cluster_buffers.len() {
            self.mesh_cluster_buffers.resize_with(idx + 1, || None);
        }
        if idx >= self.mesh_cluster_diag.len() {
            self.mesh_cluster_diag.resize_with(idx + 1, || None);
        }
        if clusters.is_empty() {
            let was_present = self.mesh_cluster_buffers[idx].is_some();
            self.mesh_cluster_buffers[idx] = None;
            self.mesh_cluster_diag[idx] = None;
            return was_present;
        }
        // RKP_RASTER_DIAG=1 — aggregate per-LOD counts at upload time
        // so the per-frame dispatch print is O(active draws), not
        // O(total clusters). The walk is ~64 ns per cluster and only
        // fires when the diag flag is set, so it stays free in the
        // default config.
        if std::env::var("RKP_RASTER_DIAG").is_ok() {
            use rkp_core::mesh_cluster::{
                CLUSTER_FLAG_LOD_DIRTY, DAG_GROUP_NONE, PARENT_GROUP_ERROR_ROOT,
            };
            let max_lod = clusters.iter().map(|c| c.lod_level).max().unwrap_or(0) as usize;
            let mut diag = MeshClusterDiag {
                clusters_per_lod: vec![0; max_lod + 1],
                indices_per_lod: vec![0; max_lod + 1],
                lod_dirty_per_lod: vec![0; max_lod + 1],
                ..Default::default()
            };
            // `dag_present` heuristic: any cluster with a non-NONE DAG
            // pointer means the file is v6 (or has been sculpted into v6
            // shape). Used to qualify the patch-count print: v5 files
            // light up every cluster under "both DAG_GROUP_NONE", so the
            // raw number is only meaningful when dag_present.
            let dag_present = clusters
                .iter()
                .any(|c| c.group_above_idx != DAG_GROUP_NONE || c.group_below_idx != DAG_GROUP_NONE);
            for c in clusters {
                let l = c.lod_level as usize;
                if l < diag.clusters_per_lod.len() {
                    diag.clusters_per_lod[l] += 1;
                    diag.indices_per_lod[l] += c.index_count;
                    if c.flags & CLUSTER_FLAG_LOD_DIRTY != 0 {
                        diag.lod_dirty_per_lod[l] += 1;
                    }
                }
                if dag_present
                    && c.lod_level == 0
                    && c.group_above_idx == DAG_GROUP_NONE
                    && c.group_below_idx == DAG_GROUP_NONE
                    && c.cluster_error == 0.0
                    && c.parent_group_error >= PARENT_GROUP_ERROR_ROOT * 0.5
                {
                    diag.patch_count += 1;
                }
                diag.total_indices += c.index_count;
            }
            diag.lod_dirty_total = diag.lod_dirty_per_lod.iter().sum();
            self.mesh_cluster_diag[idx] = Some(diag);
        } else {
            self.mesh_cluster_diag[idx] = None;
        }
        let bytes: &[u8] = bytemuck::cast_slice(clusters);
        let needed = bytes.len() as u64;

        let existing = self.mesh_cluster_buffers[idx].take();
        let (buffer, replaced) = if let Some((buf, _)) = existing.as_ref() {
            if buf.size() >= needed {
                queue.write_buffer(buf, 0, bytes);
                (buf.clone(), false)
            } else {
                let new_buf = grow_with_full_upload(
                    &self.device, queue,
                    Some(buf.size()),
                    "mesh asset cluster table",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    bytes,
                );
                (new_buf, true)
            }
        } else {
            let new_buf = grow_with_full_upload(
                &self.device, queue,
                None,
                "mesh asset cluster table",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                bytes,
            );
            (new_buf, true)
        };
        self.mesh_cluster_buffers[idx] = Some((buffer, clusters.len() as u32));
        replaced
    }

    /// Drop the cached cluster table for `handle_raw`. Called when
    /// an asset is released or invalidated.
    pub fn release_mesh_clusters_for_asset(&mut self, handle_raw: u32) {
        let idx = handle_raw as usize;
        if let Some(slot) = self.mesh_cluster_buffers.get_mut(idx) {
            *slot = None;
        }
        if let Some(slot) = self.mesh_cluster_diag.get_mut(idx) {
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

    /// Render user-shader-mesh blades into ONE cascade's shadow depth
    /// view. Composes on top of the mesh-path opaque shadows already
    /// in the same view (load + store, no clear). Caller drives the
    /// per-cascade loop and provides the matching shadow_g0 bind
    /// group (built per-VR per-cascade in `refresh_mesh_shadow_bindings`).
    pub fn dispatch_user_shader_mesh_shadow(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        shadow_g0: &wgpu::BindGroup,
        draws: &[crate::user_shader_mesh_pass::UserShaderMeshDraw],
    ) {
        if draws.is_empty() {
            return;
        }
        let mut rp = self.user_shader_mesh.begin_shadow_pass(
            encoder,
            depth_view,
            None,
        );
        rp.set_bind_group(0, shadow_g0, &[]);
        for d in draws {
            rp.set_pipeline(&d.shadow_pipeline);
            rp.set_bind_group(1, &d.raster_g1, &[]);
            rp.draw_indirect(&d.indirect_buffer, 0);
        }
    }

    /// Runs after the primary mode's main pass + `mesh_resolve` so the
    /// G-buffer carries mesh-raster output; proxy raster
    /// depth-composites on top using `LoadOp::Load`. Writes all five
    /// gbuf targets directly — no `mesh_resolve` participation for
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

    /// Surface-mesh raster. Per draw, runs the LOD-select compute
    /// pass that fills a `DrawIndexedIndirectArgs` table for the
    /// asset's full DAG of clusters; then issues
    /// `multi_draw_indexed_indirect` over that table. Non-admitted
    /// slots carry `index_count = 0` so the no-op draws cost nothing.
    ///
    /// Writes the visibility-buffer triplet (position, pick,
    /// leaf_slot) + rest_pos. The `mesh_resolve` compute pass then
    /// fills normal / material / glass per pixel.
    pub fn dispatch_mesh(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        draws: &[MeshDraw],
    ) {
        // Diagnostic: skip primary visibility GPU work. Used to
        // measure shadow_render in isolation when chasing the
        // anti-correlation pattern between the two passes. The
        // per-slot instance + bind-group setup below still runs
        // because the shadow path consumes
        // `mesh_instance_bind_groups` for per-instance world
        // matrices — early-returning here would crash shadow with
        // an empty Vec. The actual LOD-select compute + raster
        // render passes are gated separately at their dispatch
        // sites.
        let primary_disabled = std::env::var("RKP_MESH_DISABLE_PRIMARY").is_ok();
        viewport.refresh_mesh_g0(&self.device, self);
        viewport.refresh_mesh_resolve_bindings(&self.device, self);
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
        viewport.ensure_mesh_instance_capacity(&self.device, self, draws.len() as u32);
        viewport.ensure_mesh_lod_capacity(&self.device, self, draws.len() as u32);
        for (slot, d) in draws.iter().enumerate() {
            viewport.write_mesh_instance(
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
                    let g1 = &viewport.mesh_instance_bind_groups[slot];
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

        // RKP_RASTER_DIAG=1 — per-frame, per-active-asset cluster state.
        // Surfaces the hypothesis behind the mesh_raster regression
        // (LOD_DIRTY clusters force-admit at LOD-0 unconditionally;
        // post-bake patch clusters from sculpt always Karis-admit).
        // Throttled to once per ~60 frames so the log stays readable;
        // duplicate `asset_handle_raw` entries (multiple instances of
        // one asset) collapse to a single line.
        if std::env::var("RKP_RASTER_DIAG").is_ok() {
            // Frame counter lives on the profiler-style frame counter;
            // approximate via a static AtomicU64 so we don't need to
            // plumb frame_idx through here.
            use std::sync::atomic::{AtomicU64, Ordering};
            static DIAG_FRAME: AtomicU64 = AtomicU64::new(0);
            let f = DIAG_FRAME.fetch_add(1, Ordering::Relaxed);
            if f % 60 == 0 {
                let mut printed: std::collections::HashSet<u32> =
                    std::collections::HashSet::new();
                for d in draws {
                    if !printed.insert(d.asset_handle_raw) {
                        continue;
                    }
                    let total_clusters = self
                        .mesh_cluster_buffer(d.asset_handle_raw)
                        .map(|(_, c)| c)
                        .unwrap_or(0);
                    let diag = self
                        .mesh_cluster_diag
                        .get(d.asset_handle_raw as usize)
                        .and_then(|s| s.as_ref());
                    match diag {
                        Some(diag) => {
                            let per_lod: String = diag
                                .clusters_per_lod
                                .iter()
                                .enumerate()
                                .map(|(l, &n)| {
                                    format!(
                                        "lod{l}={}c/{}tri/dirty{}",
                                        n,
                                        diag.indices_per_lod[l] / 3,
                                        diag.lod_dirty_per_lod[l],
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(" ");
                            eprintln!(
                                "[raster_diag frame] asset={} total_clusters={} total_tris={} | {} | LOD_DIRTY={} patch_clusters={}",
                                d.asset_handle_raw,
                                total_clusters,
                                diag.total_indices / 3,
                                per_lod,
                                diag.lod_dirty_total,
                                diag.patch_count,
                            );
                        }
                        None => {
                            eprintln!(
                                "[raster_diag frame] asset={} total_clusters={} (diag not captured — upload before setting RKP_RASTER_DIAG?)",
                                d.asset_handle_raw, total_clusters,
                            );
                        }
                    }
                }
            }
        }

        if primary_disabled {
            // Diagnostic isolation mode: skip the raster + resolve
            // GPU work entirely. The per-slot setup above still ran
            // so shadow has populated bind groups to consume.
            return;
        }

        let g0_bg = viewport
            .mesh_g0_bg
            .as_ref()
            .expect("mesh g0 bg present after refresh_mesh_g0");
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

        // 1. Visibility-buffer raster — writes (position, pick, leaf_slot)
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
                let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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

        // 2. Resolve compute — fills normal / material / glass per pixel.
        let resolve_g0 = viewport
            .mesh_resolve_g0_bg
            .as_ref()
            .expect("mesh_resolve g0 bg present after refresh");
        let resolve_g1 = viewport
            .mesh_resolve_g1_bg
            .as_ref()
            .expect("mesh_resolve g1 bg present after refresh");
        let q_resolve = self.profiler.begin_query("mesh_resolve", encoder);
        self.mesh_resolve.dispatch(
            encoder,
            resolve_g0,
            resolve_g1,
            viewport.width,
            viewport.height,
        );
        self.profiler.end_query(encoder, q_resolve);

        // 3. Mesh-mode glass — front raster + back raster + combine.
        //    Runs after `mesh_resolve` (which writes zeros to
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
                let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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
                let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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
    ///   2. Have already populated `mesh_instance_buffers` (any
    ///      previous `dispatch_mesh` this frame did
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
        draws: &[MeshDraw],
        user_shader_mesh_draws: &[crate::user_shader_mesh_pass::UserShaderMeshDraw],
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
        // cascade culls hardest. Default base 6.0 (was 4.0 — bumped
        // after a threshold sweep on splat5 elephant: base=4 vs base=6
        // cut cascade-0 admit count 47.9k → 26.8k and mesh_shadow_render[0]
        // 0.80 → 0.44 ms, gaining ~12 fps at every primary-threshold
        // setting. Shadow visuals indistinguishable in the test scene —
        // ortho shadow pixels are several times coarser than perspective
        // camera pixels, so shadows can tolerate a much higher pixel-
        // error budget than primary visibility before LOD pop is
        // visible. Override with `RKP_MESH_SHADOW_LOD_THRESHOLD`
        // (base) and `RKP_CSM_THRESHOLD_FALLOFF` (per-cascade scale).
        const PIXEL_THRESHOLD_SHADOW: f32 = 6.0;
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

        // ── Phase 1: per-cascade × per-slot prep (no GPU work) ─────
        // Hoisted out of the original per-cascade render loop so all
        // four cascades' params + count clears are queued BEFORE the
        // merged LOD-select compute pass. Lets phase 2 issue one big
        // compute pass instead of four small ones, which removes ~3
        // compute-pass-boundary stalls per frame.
        let collect_pipestats = pipestats_enabled;
        let pixel_thresholds: Vec<f32> = (0..cascade_count)
            .map(|c| base_threshold * threshold_falloff.powi(c as i32))
            .collect();
        for cascade in 0..cascade_count {
            let pixel_threshold = pixel_thresholds[cascade];
            let collect_stats = lod_stats_enabled && cascade == 0;
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
        }

        // ── Phase 2: merged LOD-select compute pass (all cascades) ─
        // One compute pass holds every cascade's dispatch back-to-back.
        // Cascades write to disjoint args+count buffers and read shared
        // cluster tables — no inter-dispatch data dependency, so the
        // GPU pipelines them freely. The previous structure issued one
        // compute pass per cascade interleaved with the shadow render
        // passes; each compute-pass boundary serialized the GPU.
        if !direct_mode {
            // Stats lifecycle is once per pass, not per cascade (the
            // admit-stats buffer is pass-shared).
            if lod_stats_enabled {
                viewport.lod_stats_drain_shadow("shadow");
                viewport.lod_stats_clear_shadow(encoder);
            }
            for cascade in 0..cascade_count {
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
            }

            let q_lod_all = self
                .profiler
                .begin_query("mesh_shadow_lod_select_all", encoder);
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mesh_shadow_lod_select_all"),
                    timestamp_writes: None,
                });
                for cascade in 0..cascade_count {
                    if collect_pipestats {
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
                        let g1 = &viewport.mesh_instance_bind_groups[slot];
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
            }
            self.profiler.end_query(encoder, q_lod_all);

            if lod_stats_enabled {
                viewport.lod_stats_finalize_shadow(encoder);
            }
        }

        // ── Phase 3: per-cascade render passes (unchanged loop) ────
        for cascade in 0..cascade_count {
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
                    let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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

            // 1a. V1 user-shader-mesh blade shadows (grass / leaves /
            //     etc). Composes onto the same cascade depth view the
            //     mesh shadow pass just wrote into. Per-cascade
            //     shadow_g0 is built per-VR in `refresh_mesh_shadow_bindings`.
            if !user_shader_mesh_draws.is_empty() {
                let q_us = self.profiler.begin_query(
                    &format!("user_shader_mesh_shadow[{cascade}]"),
                    encoder,
                );
                let us_g0 = viewport
                    .user_shader_mesh_shadow_g0_bgs[cascade]
                    .as_ref()
                    .expect("user_shader_mesh shadow g0 bg present after refresh");
                self.dispatch_user_shader_mesh_shadow(
                    encoder,
                    &viewport.mesh_shadow_depth_views[cascade],
                    us_g0,
                    user_shader_mesh_draws,
                );
                self.profiler.end_query(encoder, q_us);
            }

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
                    let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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
                    let g1_bg = &viewport.mesh_instance_bind_groups[slot];
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

    pub fn upload_geometry(&mut self, queue: &wgpu::Queue, data: &GeometryUpload) {
        self.scene.upload_geometry(&self.device, queue, data);
    }

    pub fn upload_frame(&mut self, queue: &wgpu::Queue, data: &FrameUpload) {
        self.scene.upload_frame(&self.device, queue, data);
    }

    /// Render one frame into `viewport`. Dispatches into the VR's own
    /// per-resolution passes; in `Isolation` mode the atmosphere /
    /// volumetric / god_rays / bloom passes are skipped to give a
    /// clean studio look.
    ///
    /// `preview_mode` selects the primary-visibility pass: `Voxel`
    /// runs the surface-mesh raster; `Raymarch` runs the procedural
    /// CSG raymarcher instead. Only the build viewport uses
    /// `Raymarch`; everywhere else passes `Voxel` (the default).
    pub fn render_to(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        viewport: &mut crate::viewport_renderer::ViewportRenderer,
        // When true, dispatch the directional CSM render after
        // primary visibility. Engine sets this when there's a live
        // shadow-casting directional light; ShadeParams.shadow_map_
        // enabled is in lockstep so shade samples the fresh map.
        shadow_map_enabled: bool,
        atmo_frame_params: &crate::rkp_atmosphere::AtmosphereFrameParams,
        mode: crate::RenderMode,
        preview_mode: crate::BuildPreviewMode,
        // Mesh-raster instance list. One entry per visible
        // scene-instance whose asset has mesh data (i.e. came from
        // `acquire_asset`); procedural objects without baked meshes
        // are skipped client-side.
        mesh_draws: &[MeshDraw],
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

        // 0. Atmosphere LUTs (in-situ only — isolation uses a flat sky).
        if in_situ {
            self.atmosphere.dispatch_if_dirty(encoder);
            let q = self.profiler.begin_query("atmo", encoder);
            self.atmosphere.dispatch_per_frame(encoder, queue, atmo_frame_params);
            self.profiler.end_query(encoder, q);
        }

        // 1. Primary visibility → G-buffer. Two mutually-exclusive
        //    paths: procedural raymarch (build viewport live SDF
        //    preview) or surface-mesh raster (every other viewport).
        //    Both fully populate the G-buffer including miss sentinels
        //    so downstream passes are unaware of which path ran.
        if raymarch {
            let q = self.profiler.begin_query("proc_raymarch", encoder);
            viewport.proc_raymarch.dispatch(
                encoder, viewport.width, viewport.height, None,
            );
            self.profiler.end_query(encoder, q);
        } else {
            self.dispatch_mesh(queue, encoder, viewport, mesh_draws);
        }

        // 1a'. Proxy-mesh raster — composites procedural triangle
        //      meshes onto the G-buffer. Skipped in raymarch preview
        //      (single-procedural focus target, no scene composition).
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

        // 1c. Directional shadow map — mesh triangle rasterization
        //     from the light's POV into the same `shadow_buffer` the
        //     shade pass samples. Per-instance uniforms were written
        //     by the earlier `dispatch_mesh` call this frame.
        if in_situ && !raymarch && shadow_map_enabled {
            self.dispatch_mesh_shadow(
                queue, encoder, viewport, mesh_draws, user_shader_mesh_draws,
            );
        }

        // Pipeline-statistics resolve: single per-frame point so it
        // fires regardless of which mesh passes (primary / shadow /
        // both) ran. Each pass writes to its own slot; unwritten
        // slots come back as undefined data (caller can detect
        // by all-zero results or just trust the slot it cares about).
        if !raymarch && std::env::var("RKP_MESH_PIPESTATS").is_ok() {
            viewport.pipestats_finalize(encoder);
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
            // Sub-profile every dispatch under the vol pass — the original
            // single `vol` timer hid which of the five sub-dispatches dominated.
            // The umbrella `vol` query stays so existing dashboards / log
            // parsers keep working; the per-dispatch queries nest inside it.
            let q = self.profiler.begin_query("vol", encoder);
            let q_fog = self.profiler.begin_query("vol_fog_march", encoder);
            viewport.volumetric.dispatch_fog_march(encoder);
            self.profiler.end_query(encoder, q_fog);
            let q_cloud = self.profiler.begin_query("vol_cloud_march", encoder);
            viewport.volumetric.dispatch_cloud_march(encoder);
            self.profiler.end_query(encoder, q_cloud);
            let q_hist = self.profiler.begin_query("vol_history", encoder);
            viewport.volumetric.update_history(encoder);
            self.profiler.end_query(encoder, q_hist);
            let q_sun = self.profiler.begin_query("vol_sun_atten", encoder);
            viewport.volumetric.dispatch_sun_atten(encoder);
            self.profiler.end_query(encoder, q_sun);
            let q_comp = self.profiler.begin_query("vol_composite", encoder);
            viewport.volumetric.dispatch_composite(encoder);
            self.profiler.end_query(encoder, q_comp);
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
        //
        // `RKP_DISABLE_BLOOM=1` skips the bloom dispatch entirely for
        // critical-path probing. bloom_composite still runs (reads stale
        // mip data; visually wrong but per-frame timing is what we care
        // about for the probe).
        if in_situ && std::env::var("RKP_DISABLE_BLOOM").is_err() {
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
            // Reset the hash gate on realloc — the new buffer has the
            // bytes already (create_init_buffer mapped + memcpy'd), so
            // record this content as the "last uploaded" so the next
            // identical-content tick still short-circuits the
            // queue.write_buffer.
            self.last_lights_hash = d4_hash_bytes(data);
        } else {
            // PERF_DEBT.md D4: skip the upload when the content matches
            // the prior tick's. Sim hands us the full Vec every tick;
            // most ticks the lights are unchanged (no entity moved,
            // no env tweak). The hash is ~few µs for a ~2 KB buffer —
            // cheaper than the GPU upload it replaces.
            let hash = d4_hash_bytes(data);
            if hash != self.last_lights_hash {
                queue.write_buffer(&self.lights_buffer, 0, data);
                self.last_lights_hash = hash;
            }
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
            self.last_materials_hash = d4_hash_bytes(data);
        } else {
            // PERF_DEBT.md D4: hash-gated skip — see `update_lights`.
            let hash = d4_hash_bytes(data);
            if hash != self.last_materials_hash {
                queue.write_buffer(&self.materials_buffer, 0, data);
                self.last_materials_hash = hash;
            }
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
