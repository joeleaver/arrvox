//! Per-viewport render-target and pass state.
//!
//! Each `ViewportRenderer` owns a complete resolution-coupled render
//! chain: march → shadow_trace → ssao → shade → volumetric → god_rays →
//! bloom → bloom_composite → tone_map → composite. These are the passes
//! whose intermediate textures vary with the viewport's size; duplicating
//! them per-VR is what lets two viewports of different resolutions render
//! in one frame without clashing on shared textures.
//!
//! Shared on [`RkpRenderer`]: compute pipelines (via each pass's own
//! internal state — small), `RkpScene` buffers, atmosphere LUTs, the
//! scene-wide shade params / lights / materials buffers. The `RkpRenderer`
//! orchestrates a `render_to` call that dispatches into a VR's passes
//! while reading from its own shared bindings.

use crate::rkp_renderer::RkpRenderer;
use crate::rkp_scene::{CameraUniforms, RkpScene};
use crate::octree_march::OctreeMarchPass;
use crate::proc_raymarch::ProcRaymarchPass;
use crate::proc_outline::ProcOutlinePass;
use crate::proc_ghost::ProcGhostPass;
use crate::rkp_shadow_trace::ShadowTracePass;
use crate::shadow_map_pass::{ShadowMapPass, SHADOW_MAP_DEFAULT_SIZE};
use crate::rkp_ssao::RkpSsaoPass;
use crate::rkp_shade::RkpShadePass;
use crate::rkp_volumetric::RkpVolumetricPass;
use crate::rkp_god_rays::RkpGodRayPass;
use crate::rkp_grid::RkpGridPass;

pub struct ViewportRenderer {
    // ── Per-VR scene binding ────────────────────────────────────────
    pub camera_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    scene_epoch: u64,
    /// Epoch at which march's `set_lights`/`set_materials` bindings were
    /// last built. Compared against `RkpRenderer::lights_materials_epoch`
    /// so VRs rebuild march's lights/materials bind groups after the
    /// shared buffers reallocate.
    lights_materials_epoch: u64,

    // ── Per-VR resolution-coupled passes ───────────────────────────
    pub march: OctreeMarchPass,
    /// Live CSG preview for the build viewport. Writes the same G-buffer
    /// as `march`, so downstream passes don't care which one ran — only
    /// one of the two executes per frame, chosen by the host via
    /// `render_to`'s `preview_mode` parameter.
    pub proc_raymarch: ProcRaymarchPass,
    /// Selected-primitive outline overlay. Reads the NodeId channel
    /// the raymarch writes into the material G-buffer and highlights
    /// the silhouette of the currently-selected node. Only dispatched
    /// when the viewport is in raymarch mode — voxel mode doesn't
    /// carry per-primitive NodeIds in the same G-buffer slot.
    pub proc_outline: ProcOutlinePass,
    /// Ghost-cutter overlay — shows Subtract/Intersect operands even
    /// where they've been carved away. No G-buffer input; runs its
    /// own small raymarch over a filtered subset of primitives the
    /// host selects based on the current tree selection.
    pub proc_ghost: ProcGhostPass,
    pub shadow_trace: ShadowTracePass,
    /// Phase 8 — directional shadow map (light-POV depth march).
    /// One shared per-VR depth texture; resolution is fixed at
    /// `SHADOW_MAP_DEFAULT_SIZE` regardless of viewport size (the
    /// map covers the whole scene). S3 just creates the pass and
    /// binds its texture / uniform into shade; S4 will wire the
    /// per-frame dispatch.
    pub shadow_map: ShadowMapPass,
    pub ssao: RkpSsaoPass,
    pub shade: RkpShadePass,
    pub glass: crate::rkp_glass::RkpGlassPass,
    pub volumetric: RkpVolumetricPass,
    pub god_rays: RkpGodRayPass,

    // ── Per-VR render targets + post-process ───────────────────────
    pub gbuffer: crate::GBuffer,
    /// rkp-side pick G-buffer — `R32Uint` sibling of the shared
    /// `crate::GBuffer`'s material texture. Holds `primitive_node_id`
    /// for procedural raymarch hits so packed_r's high 16 bits in the
    /// shared material G-buffer can carry `secondary_material_id` for
    /// dual-material shading. Not part of `crate::GBuffer`
    /// (sibling-project invariants) and not read by `rkp_shade` —
    /// only `proc_outline` and the engine's pick-readback path
    /// consume it. For voxel hits from `octree_march` this slot is
    /// left unwritten; MAIN viewport resolves picks via `object_id`
    /// in packed_g, same as before.
    pub pick_texture: wgpu::Texture,
    pub pick_view: wgpu::TextureView,
    pub bloom: crate::BloomPass,
    pub bloom_composite: crate::BloomCompositePass,
    pub tone_map: crate::ToneMapPass,
    pub composite_texture: wgpu::Texture,
    pub composite_view: wgpu::TextureView,
    /// Async readback ring — the engine writes the composite into one of
    /// the buffers each frame, kicks off `map_async` immediately after
    /// submit, and reads pixels back **without blocking** on `device.poll`.
    /// See [`ReadbackRing`] for the state-machine details.
    pub readback: ReadbackRing,
    pub wireframe_pass: crate::WireframePass,
    /// Isolation-mode infinite grid overlay. Always constructed; the
    /// host only dispatches it when the viewport's mode is `Isolation`.
    pub grid: RkpGridPass,
    pub width: u32,
    pub height: u32,
}

impl ViewportRenderer {
    /// Build a viewport renderer at the given size.
    ///
    /// Creates its own instances of the resolution-coupled passes
    /// (march/shadow/ssao/shade/vol/god_rays) and wires them to the
    /// shared state on `renderer` (atmosphere LUTs, shade params,
    /// lights, materials buffers).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut RkpRenderer,
        width: u32,
        height: u32,
    ) -> Self {
        // Camera buffer + scene bind group (Phase 6a pattern).
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_vr_camera"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (scene_bind_group, scene_epoch) = {
            let scene: &RkpScene = &renderer.scene;
            (scene.build_bind_group(device, &camera_buffer), scene.buffers_epoch())
        };

        // Gbuffer.
        let gbuffer = crate::GBuffer::new(device, width, height);

        // rkp-side pick texture (R32Uint). Written by the procedural
        // raymarch, read by `proc_outline` and the pick readback.
        let (pick_texture, pick_view) = create_pick_texture(device, width, height);

        // Per-VR passes. Each wired to: (a) its own gbuffer views, (b)
        // the shared scene bind-group layout + shared buffers on renderer.
        let mut march = OctreeMarchPass::new(device, &renderer.scene.bind_group_layout);
        march.set_materials(device, &renderer.materials_buffer);
        march.set_lights(device, &renderer.lights_buffer);
        march.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view, &pick_view, &gbuffer.glass_view, &gbuffer.leaf_slot_view);

        // Procedural CSG raymarch — alternative primary-visibility pass
        // for the build viewport. Wired to the same per-VR camera + gbuffer
        // so the host can flip between voxel and raymarch without touching
        // the rest of the chain.
        let mut proc_raymarch = ProcRaymarchPass::new(device);
        proc_raymarch.set_camera(device, &camera_buffer);
        proc_raymarch.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view, &pick_view, &gbuffer.glass_view, &gbuffer.leaf_slot_view);

        // Outline overlay — rebind the pick gbuffer view on resize.
        let mut proc_outline = ProcOutlinePass::new(device, crate::LDR_FORMAT);
        proc_outline.set_gbuffer(device, &pick_view);

        let mut proc_ghost = ProcGhostPass::new(device, crate::LDR_FORMAT);
        proc_ghost.set_camera(device, &camera_buffer);

        let mut ssao = RkpSsaoPass::new(device, queue, width, height);
        ssao.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view);

        let mut shadow_trace = ShadowTracePass::new(
            device, width, height,
            &renderer.scene.bind_group_layout,
            march.params_bind_group_layout(),
        );
        shadow_trace.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view);

        // Phase 8 S3 — shadow map pass. Dormant until S4 plumbs the
        // per-frame dispatch + shade-side enable flag; constructed
        // here so the shade pass's group 1 has a real texture +
        // uniform to bind to.
        let shadow_map = ShadowMapPass::new(
            device,
            SHADOW_MAP_DEFAULT_SIZE,
            &renderer.scene.bind_group_layout,
        );

        let mut shade = RkpShadePass::new(device, width, height);
        shade.set_shade_data(
            device,
            &renderer.shade_params_buffer,
            &renderer.lights_buffer,
            &renderer.materials_buffer,
        );
        shade.set_camera(device, &camera_buffer);
        shade.set_atmosphere_luts(
            device,
            &renderer.atmosphere.transmittance_view,
            &renderer.atmosphere.multiscatter_view,
            &renderer.atmosphere.lut_sampler,
            &renderer.atmosphere.sky_view_view,
            &renderer.atmosphere.ap_view,
        );
        // Phase 4c — shade reads the host G-buffer directly. Phase 5
        // retired Option B's instance-merged G-buffer; user-shader
        // hits land in the host G-buffer through the unified march
        // pipeline.
        shade.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view, &gbuffer.glass_view, &gbuffer.leaf_slot_view);
        shade.set_shadow_and_ssao(
            device,
            &shadow_trace.output_view,
            &ssao.output_view,
            &shadow_map.texture_view,
            &shadow_map.uniform_buffer,
        );

        // Pass order: shade → volumetric → glass → god_rays. Glass
        // runs AFTER volumetric so clouds / fog are composited into
        // the "behind" HDR first, and then refracted / Beer-tinted
        // through the glass (so clouds visible through a glass pane
        // correctly bend rather than getting stamped on top of the
        // glass composite).
        let mut volumetric = RkpVolumetricPass::new(device, width, height);
        volumetric.set_depth_view(device, &gbuffer.position_view);
        volumetric.set_scene_hdr_view(device, &shade.output_view);

        let mut glass = crate::rkp_glass::RkpGlassPass::new(device, width, height);
        glass.set_inputs(
            device,
            &volumetric.output_view,
            &gbuffer.glass_view,
            &camera_buffer,
            &renderer.materials_buffer,
            &renderer.shade_params_buffer,
            &renderer.lights_buffer,
        );

        let mut god_rays = RkpGodRayPass::new(device, width, height);
        god_rays.set_inputs(device, &glass.output_view, &gbuffer.position_view, &volumetric.cloud_view);

        // VR-owned bloom / tonemap chain reads this VR's god_rays output.
        let bloom = crate::BloomPass::new(device, &god_rays.output_view, width, height);
        let bloom_composite = crate::BloomCompositePass::new(
            device, &god_rays.output_view, bloom.mip_views(), width, height,
        );
        let tone_map = crate::ToneMapPass::new(device, &bloom_composite.output_view, width, height);

        let readback = ReadbackRing::new(device, width, height);

        let wireframe_pass = crate::WireframePass::new(device, crate::LDR_FORMAT);

        let mut grid = RkpGridPass::new(device, crate::LDR_FORMAT);
        grid.set_bindings(device, &camera_buffer, &gbuffer.position_view);
        grid.update_params(queue, &crate::rkp_grid::GridParams::default());

        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let composite_view = composite_texture.create_view(&Default::default());

        let lights_materials_epoch = renderer.lights_materials_epoch();

        Self {
            camera_buffer, scene_bind_group, scene_epoch, lights_materials_epoch,
            march, proc_raymarch, proc_outline, proc_ghost,
            shadow_trace, shadow_map, ssao, shade, glass, volumetric, god_rays,
            gbuffer, pick_texture, pick_view, bloom, bloom_composite, tone_map,
            composite_texture, composite_view,
            readback,
            wireframe_pass, grid, width, height,
        }
    }

    /// Upload this viewport's camera uniform.
    pub fn upload_camera(&self, queue: &wgpu::Queue, camera: &CameraUniforms) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    /// Rebuild scene bind group if shared scene buffers reallocated.
    /// Also rebuild march + shade lights/materials bindings if those
    /// buffers reallocated (separate epoch — different write path).
    pub fn refresh_bindings(&mut self, device: &wgpu::Device, renderer: &RkpRenderer) {
        let scene_now = renderer.scene.buffers_epoch();
        if scene_now != self.scene_epoch {
            self.scene_bind_group = renderer.scene.build_bind_group(device, &self.camera_buffer);
            self.scene_epoch = scene_now;
        }
        let lm_now = renderer.lights_materials_epoch();
        if lm_now != self.lights_materials_epoch {
            self.march.set_materials(device, &renderer.materials_buffer);
            self.march.set_lights(device, &renderer.lights_buffer);
            self.shade.set_shade_data(
                device,
                &renderer.shade_params_buffer,
                &renderer.lights_buffer,
                &renderer.materials_buffer,
            );
            // Glass pass also reads the materials SSBO (for glass
            // albedo / IOR) + shade_params + lights (for GGX direct
            // spec). Refresh all three when the lights/materials
            // epoch bumps; any of them may reallocate. Input HDR
            // comes from volumetric so clouds / fog land in the
            // "behind" before glass Fresnel + refraction.
            self.glass.set_inputs(
                device,
                &self.volumetric.output_view,
                &self.gbuffer.glass_view,
                &self.camera_buffer,
                &renderer.materials_buffer,
                &renderer.shade_params_buffer,
                &renderer.lights_buffer,
            );
            self.lights_materials_epoch = lm_now;
        }
    }

    /// Re-create per-resolution resources at a new size. Rebuilds the
    /// entire pass chain because every intermediate texture is tied to
    /// the target's dimensions.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut RkpRenderer,
        width: u32,
        height: u32,
    ) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        // Gbuffer.
        self.gbuffer = crate::GBuffer::new(device, width, height);
        let (pick_texture, pick_view) = create_pick_texture(device, width, height);
        self.pick_texture = pick_texture;
        self.pick_view = pick_view;

        // Per-VR passes — resize internal textures + re-wire gbuffer bindings.
        self.march.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view, &self.pick_view, &self.gbuffer.glass_view, &self.gbuffer.leaf_slot_view);
        self.proc_raymarch.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view, &self.pick_view, &self.gbuffer.glass_view, &self.gbuffer.leaf_slot_view);
        self.proc_outline.set_gbuffer(device, &self.pick_view);

        self.ssao.resize(device, width, height);
        self.ssao.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view);

        self.shadow_trace.resize(device, width, height);
        self.shadow_trace.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view);

        self.shade.resize(device, width, height);
        self.shade.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view, &self.gbuffer.glass_view, &self.gbuffer.leaf_slot_view);
        // Shadow map size doesn't track viewport resolution; the
        // texture stays at SHADOW_MAP_DEFAULT_SIZE through resize.
        // Re-binding picks up the same view (placeholder pre-S4).
        self.shade.set_shadow_and_ssao(
            device,
            &self.shadow_trace.output_view,
            &self.ssao.output_view,
            &self.shadow_map.texture_view,
            &self.shadow_map.uniform_buffer,
        );

        self.volumetric.resize(device, width, height);
        self.volumetric.set_depth_view(device, &self.gbuffer.position_view);
        self.volumetric.set_scene_hdr_view(device, &self.shade.output_view);

        self.glass.resize(device, width, height);
        self.glass.set_inputs(
            device,
            &self.volumetric.output_view,
            &self.gbuffer.glass_view,
            &self.camera_buffer,
            &renderer.materials_buffer,
            &renderer.shade_params_buffer,
            &renderer.lights_buffer,
        );

        self.god_rays.resize(device, width, height);
        self.god_rays.set_inputs(device, &self.glass.output_view, &self.gbuffer.position_view, &self.volumetric.cloud_view);

        // Bloom / tonemap chain — these hold their own output textures.
        self.bloom = crate::BloomPass::new(device, &self.god_rays.output_view, width, height);
        self.bloom_composite = crate::BloomCompositePass::new(
            device, &self.god_rays.output_view, self.bloom.mip_views(), width, height,
        );
        self.tone_map = crate::ToneMapPass::new(device, &self.bloom_composite.output_view, width, height);

        self.readback.resize(device, width, height);

        self.composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp composite"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.composite_view = self.composite_texture.create_view(&Default::default());

        // Grid: rebind to the new gbuffer position view (camera buffer
        // is stable across resize so it doesn't need re-wiring).
        self.grid.set_bindings(device, &self.camera_buffer, &self.gbuffer.position_view);

        // Env-dirty path applies bloom/tonemap params — the engine re-fires it
        // after a resize so VRs rebuilt their bloom/tonemap from defaults here
        // end up with scene-correct values on the next frame.
        let _ = renderer;
    }

    pub fn readback_padded_row(&self) -> u32 {
        (self.width * 4 + 255) & !255
    }

    // Phase 5 — Option B's `dispatch_instance_overlay` was removed.
    // User-shader instances now flow through the unified host march
    // via `asset.shader_id` branch (see `octree_march.wgsl` and
    // `tick_instance_pipeline`). The per-pixel march + composite
    // pipeline is gone.

    /// Encode a copy of the composite texture into the readback ring's
    /// next idle buffer. Returns the index that was written, or `None` if
    /// every buffer is still in flight (caller should skip readback this
    /// frame and reuse cached pixels).
    pub fn encode_composite_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<usize> {
        let idx = self.readback.acquire_write_idx()?;
        let padded_row = self.readback_padded_row();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback.buffers[idx],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        Some(idx)
    }
}

/// Triple-buffered async readback. The engine writes the composite into
/// one buffer per frame, immediately issues `map_async` on it, and reads
/// pixels back the frame (or two) later via [`drain_completed`]. There is
/// no `device.poll(Wait)` anywhere on the hot path — completion is checked
/// with `try_recv` after a non-blocking `poll(Poll)`.
///
/// Three buffers gives one slot for the in-flight write, one for the
/// in-flight map, and one always-idle slot, so [`acquire_write_idx`]
/// effectively never returns `None` when the GPU is keeping up.
pub struct ReadbackRing {
    pub buffers: [wgpu::Buffer; 3],
    pending: [Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>; 3],
    cached: Vec<u8>,
    cached_w: u32,
    cached_h: u32,
}

impl ReadbackRing {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        Self {
            buffers: [
                create_readback_buffer(device, width, height),
                create_readback_buffer(device, width, height),
                create_readback_buffer(device, width, height),
            ],
            pending: [None, None, None],
            cached: Vec::new(),
            cached_w: 0,
            cached_h: 0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Any in-flight maps reference the OLD buffers. Dropping the
        // receiver is safe — the buffers themselves still exist until
        // `Drop` runs in the next assignment, and wgpu cancels the maps
        // when the buffer goes away.
        self.pending = [None, None, None];
        self.buffers = [
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
        ];
        self.cached.clear();
        self.cached_w = 0;
        self.cached_h = 0;
    }

    pub fn acquire_write_idx(&self) -> Option<usize> {
        self.pending.iter().position(|p| p.is_none())
    }

    /// `true` when at least one readback slot is idle — i.e.
    /// [`acquire_write_idx`] would return `Some`. Callers use this as a
    /// GPU-backpressure signal: if every slot is still waiting for a
    /// previously-submitted `map_async` to complete, the CPU is
    /// outpacing the GPU and should back off rather than submit more
    /// work (which would just deepen the queue and delay every pending
    /// readback even further).
    pub fn has_idle_slot(&self) -> bool {
        self.pending.iter().any(|p| p.is_none())
    }

    /// After submit, kick off `map_async` on the buffer that was just
    /// written. Buffer state goes from idle → in-flight; later
    /// `drain_completed` will read it back and return it to idle.
    pub fn issue_map_async(&mut self, idx: usize) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.buffers[idx].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.pending[idx] = Some(rx);
    }

    /// Non-blocking: copy any completed maps' pixels into the cached
    /// frame. Caller must have already done `device.poll(Poll)` so
    /// callbacks can fire. Returns `true` when `cached_pixels` was
    /// updated this call.
    pub fn drain_completed(&mut self, width: u32, height: u32, padded_row: u32) -> bool {
        let mut updated = false;
        for i in 0..self.pending.len() {
            let done = self.pending[i].as_ref().map(|rx| rx.try_recv().is_ok()).unwrap_or(false);
            if !done {
                continue;
            }
            self.pending[i] = None;
            let slice = self.buffers[i].slice(..);
            let data = slice.get_mapped_range();
            let mut out = vec![0u8; (width * height * 4) as usize];
            for y in 0..height as usize {
                let src = y * padded_row as usize;
                let dst = y * width as usize * 4;
                let row = width as usize * 4;
                if src + row <= data.len() && dst + row <= out.len() {
                    out[dst..dst + row].copy_from_slice(&data[src..src + row]);
                }
            }
            drop(data);
            self.buffers[i].unmap();
            self.cached = out;
            self.cached_w = width;
            self.cached_h = height;
            updated = true;
        }
        updated
    }

    pub fn cached_pixels(&self) -> Option<(&[u8], u32, u32)> {
        if self.cached.is_empty() {
            None
        } else {
            Some((&self.cached, self.cached_w, self.cached_h))
        }
    }
}

fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let padded_row = (width * 4 + 255) & !255;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rkp readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn create_pick_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rkp pick"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
