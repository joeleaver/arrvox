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
    pub ssao: RkpSsaoPass,
    pub shade: RkpShadePass,
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
    pub readback_buffers: [wgpu::Buffer; 2],
    pub readback_index: usize,
    pub readback_ready: bool,
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
        march.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);

        // Procedural CSG raymarch — alternative primary-visibility pass
        // for the build viewport. Wired to the same per-VR camera + gbuffer
        // so the host can flip between voxel and raymarch without touching
        // the rest of the chain.
        let mut proc_raymarch = ProcRaymarchPass::new(device);
        proc_raymarch.set_camera(device, &camera_buffer);
        proc_raymarch.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view, &pick_view);

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
        shade.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view);
        shade.set_shadow_and_ssao(device, &shadow_trace.output_view, &ssao.output_view);

        let mut volumetric = RkpVolumetricPass::new(device, width, height);
        volumetric.set_depth_view(device, &gbuffer.position_view);
        volumetric.set_scene_hdr_view(device, &shade.output_view);

        let mut god_rays = RkpGodRayPass::new(device, width, height);
        god_rays.set_input(device, &volumetric.output_view);

        // VR-owned bloom / tonemap chain reads this VR's god_rays output.
        let bloom = crate::BloomPass::new(device, &god_rays.output_view, width, height);
        let bloom_composite = crate::BloomCompositePass::new(
            device, &god_rays.output_view, bloom.mip_views(), width, height,
        );
        let tone_map = crate::ToneMapPass::new(device, &bloom_composite.output_view, width, height);

        let readback_buffers = [
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
        ];

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
            shadow_trace, ssao, shade, volumetric, god_rays,
            gbuffer, pick_texture, pick_view, bloom, bloom_composite, tone_map,
            composite_texture, composite_view,
            readback_buffers, readback_index: 0, readback_ready: false,
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
        self.march.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view);
        self.proc_raymarch.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view, &self.pick_view);
        self.proc_outline.set_gbuffer(device, &self.pick_view);

        self.ssao.resize(device, width, height);
        self.ssao.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view);

        self.shadow_trace.resize(device, width, height);
        self.shadow_trace.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view);

        self.shade.resize(device, width, height);
        self.shade.set_gbuffer(device, &self.gbuffer.position_view, &self.gbuffer.normal_view, &self.gbuffer.material_view);
        self.shade.set_shadow_and_ssao(device, &self.shadow_trace.output_view, &self.ssao.output_view);

        self.volumetric.resize(device, width, height);
        self.volumetric.set_depth_view(device, &self.gbuffer.position_view);
        self.volumetric.set_scene_hdr_view(device, &self.shade.output_view);

        self.god_rays.resize(device, width, height);
        self.god_rays.set_input(device, &self.volumetric.output_view);

        // Bloom / tonemap chain — these hold their own output textures.
        self.bloom = crate::BloomPass::new(device, &self.god_rays.output_view, width, height);
        self.bloom_composite = crate::BloomCompositePass::new(
            device, &self.god_rays.output_view, self.bloom.mip_views(), width, height,
        );
        self.tone_map = crate::ToneMapPass::new(device, &self.bloom_composite.output_view, width, height);

        self.readback_buffers = [
            create_readback_buffer(device, width, height),
            create_readback_buffer(device, width, height),
        ];
        self.readback_ready = false;

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

    pub fn copy_composite_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        let padded_row = self.readback_padded_row();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.composite_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffers[self.readback_index],
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
    }

    pub fn advance_readback(&mut self) {
        self.readback_ready = true;
        self.readback_index = 1 - self.readback_index;
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
