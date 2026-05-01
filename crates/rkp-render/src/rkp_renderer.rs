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
use wgpu_profiler::GpuProfiler;

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
            _padding: [0.0; 5],
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

        let lights_capacity = lights_buffer.size();
        let materials_capacity = materials_buffer.size();
        Self {
            scene, atmosphere,
            shade_params_buffer,
            lights_buffer, lights_capacity,
            materials_buffer, materials_capacity,
            lights_materials_epoch: 0,
            skin_deform,
            device: device.clone(),
            profiler, timestamp_period,
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
        // Phase 8 — TLAS prim count (one per shadow caster). The
        // shadow-map setup pass walks `tlas_prims[0..prim_count]`;
        // `tlas_node_count` is the BVH node count, which is up to
        // `2*prim_count - 1` and not what the setup needs.
        tlas_prim_count: u32,
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
    ) {
        let in_situ = matches!(mode, crate::RenderMode::InSitu);
        let raymarch = matches!(preview_mode, crate::BuildPreviewMode::Raymarch);

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

        // 1. Primary visibility → G-buffer. Voxel march *or* procedural
        //    raymarch, never both — each fully populates the G-buffer
        //    (including writing "miss" at non-hit pixels), so the one
        //    that doesn't run would just be overwriting. Shadow_trace
        //    is skipped in raymarch mode because the procedural preview
        //    doesn't have the world-space voxel grid that pass needs;
        //    isolation mode already forces shadow=1.0 inside shade,
        //    which is what we want for the clean preview look anyway.
        if raymarch {
            let q = self.profiler.begin_query("proc_raymarch", encoder);
            viewport.proc_raymarch.dispatch(
                encoder, viewport.width, viewport.height, None,
            );
            self.profiler.end_query(encoder, q);
        } else {
            viewport.march.clear_stats(encoder);
            let q = self.profiler.begin_query("march", encoder);
            viewport.march.dispatch(
                encoder, queue, &viewport.scene_bind_group,
                object_count, viewport.width, viewport.height, 0,
                shadow_steps, num_lights, lod_enabled, surfacenet_enabled,
                tile_count_x, tlas_node_count,
                shadow_map_enabled, None,
            );
            self.profiler.end_query(encoder, q);
        }

        // 1b. Half-res shadow trace. Skipped in isolation — the shade
        // pass forces shadow=1.0 there. Uses march's params bind group.
        if in_situ && !raymarch {
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
        if in_situ && !raymarch && shadow_map_enabled {
            let q = self.profiler.begin_query("shadow_map", encoder);
            viewport.shadow_map.dispatch_clear(encoder);
            viewport.shadow_map.dispatch_setup(encoder, queue, tlas_prim_count);
            viewport.shadow_map.dispatch_emit(encoder, tlas_prim_count);
            viewport.shadow_map.dispatch_finalize(encoder);
            viewport.shadow_map.dispatch_scatter(
                encoder, &viewport.scene_bind_group, tlas_prim_count,
            );
            self.profiler.end_query(encoder, q);
        }
        if !raymarch {
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
