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
use crate::splat_pass::{SplatInstanceUniform, SPLAT_INSTANCE_BYTES};
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
    /// Per-viewport tile-binning pass for user-shader emitted instances.
    /// Resolution-coupled because tile count = (W/8) × (H/8).
    pub user_shader_tile_bin: crate::user_shader_tile_bin_pass::UserShaderTileBinPass,
    /// Per-tile counters for emitted-instance binning (atomic u32 each,
    /// cleared to 0 each frame).
    pub user_shader_tile_counts_buffer: wgpu::Buffer,
    /// Per-tile flat instance-index lists. Sized
    /// `tile_count × MAX_INSTANCES_PER_TILE × 4 bytes`.
    pub user_shader_tile_lists_buffer: wgpu::Buffer,
    /// Cached tile-bin bind group; rebuilt on resize.
    pub user_shader_tile_bin_bg: Option<wgpu::BindGroup>,
    /// Tile counts on the X / Y axes (`W.div_ceil(8)` / `H.div_ceil(8)`).
    pub user_shader_tile_count_x: u32,
    pub user_shader_tile_count_y: u32,
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

    // ── Splat-rasterizer per-VR state (Phase B-2) ───────────────────
    /// Scene-wide bind group for the splat path: camera + leaf_attr_pool
    /// + materials + color_pool. Built lazily on first `dispatch_splat`
    /// and rebuilt whenever the scene-buffers or lights/materials epoch
    /// bumps (same triggers as the existing march bindings).
    pub splat_g0_bg: Option<wgpu::BindGroup>,
    /// Epoch the splat g0 bg was last built against — combination of
    /// scene_buffers_epoch and lights_materials_epoch. Stored separately
    /// from the march's `scene_epoch`/`lights_materials_epoch` because
    /// the splat g0 has a different shape (no objects buffer, no bone
    /// data, but adds materials + color_pool).
    pub splat_g0_scene_epoch: u64,
    pub splat_g0_lights_materials_epoch: u64,
    /// Per-instance uniform buffers (80 B each) — one slot per scene
    /// instance the splat path will draw this frame. Grown on demand
    /// by `ensure_splat_instance_capacity`. Reused frame to frame; the
    /// engine writes the current frame's matrices via `write_splat_instance`
    /// before calling `dispatch_splat`.
    pub splat_instance_buffers: Vec<wgpu::Buffer>,
    /// Bind groups paired one-to-one with `splat_instance_buffers`.
    pub splat_instance_bind_groups: Vec<wgpu::BindGroup>,

    /// Splat-resolve compute pass — per-VR texture bindings (g0).
    /// Bound to (leaf_slot, pick) reads + (normal, material, glass)
    /// storage writes. Rebuilt on resize (gbuffer texture views move).
    pub splat_resolve_g0_bg: Option<wgpu::BindGroup>,
    /// Splat-resolve compute pass — scene-buffers bindings (g1).
    /// Bound to leaf_attr_pool / color_pool / objects_buffer; rebuilt
    /// when scene_buffers_epoch bumps.
    pub splat_resolve_g1_bg: Option<wgpu::BindGroup>,
    pub splat_resolve_scene_epoch: u64,

    // ── Mesh-rendered shadow-map per-VR state (Phase 3) ────────────
    /// 1024×1024 `Depth32Float` texture used by `MeshShadowMapPass`.
    /// Filled by depth-only rasterization, then read by the blit
    /// compute pass which copies bitcast(depth) into `shadow_map.
    /// shadow_buffer` for shade to sample.
    pub mesh_shadow_depth_texture: wgpu::Texture,
    pub mesh_shadow_depth_view: wgpu::TextureView,
    /// Cached render `g0` for the mesh-shadow pipeline — just the
    /// `light_camera` uniform.
    pub mesh_shadow_render_g0_bg: Option<wgpu::BindGroup>,
    /// Cached blit `g0` — `depth_view` + `shadow_map.shadow_buffer`.
    pub mesh_shadow_blit_g0_bg: Option<wgpu::BindGroup>,

    // ── Mesh per-cluster LOD-select per-VR state (Phase 6.2/6.3) ───
    /// Per-draw `MeshLodSelectParams` uniform (16 B). Slot index
    /// matches `splat_instance_buffers`; one buffer per draw slot.
    pub mesh_lod_params_buffers: Vec<wgpu::Buffer>,
    /// Per-draw `DrawIndexedIndirectArgs` storage buffer + capacity
    /// in cluster slots. Grown to fit the largest cluster count any
    /// asset draw at this slot has needed. Bound as STORAGE for
    /// the LOD-select compute and as INDIRECT for the render path.
    pub mesh_lod_args_buffers: Vec<(wgpu::Buffer, u32)>,
    /// Per-draw `g0` bind group: (camera, params).
    pub mesh_lod_select_g0_bgs: Vec<wgpu::BindGroup>,
    /// Per-draw `g2` bind group: (cluster_table, args). Cached with
    /// `(asset_handle_raw, args_capacity)` so we rebuild on asset
    /// change or args resize. `None` until the first dispatch
    /// populates it; `(handle, cap)` as the freshness key.
    pub mesh_lod_select_g2_bgs: Vec<Option<(wgpu::BindGroup, u32, u32)>>,

    // ── Mesh shadow LOD-select per-VR state (Phase 6.4) ────────────
    /// `CameraUniforms`-shaped buffer populated each frame from the
    /// active light camera. Carries the light's `view_proj` (so the
    /// LOD shader gets the ortho focal-length factor),
    /// `resolution = shadow_map_size`, and the eye position derived
    /// from `view_proj_inv * (0,0,0,1)`. Other fields zeroed; the
    /// LOD shader doesn't read them.
    pub mesh_lod_shadow_camera_buffer: wgpu::Buffer,
    /// Parallel to `mesh_lod_params_buffers` — per-draw shadow
    /// params with doubled `pixel_threshold` for `lod + 1` selection.
    pub mesh_lod_shadow_params_buffers: Vec<wgpu::Buffer>,
    /// Parallel to `mesh_lod_args_buffers` — separate args storage
    /// for the shadow path so primary and shadow can run their
    /// compute passes back-to-back without aliasing.
    pub mesh_lod_shadow_args_buffers: Vec<(wgpu::Buffer, u32)>,
    /// Parallel to `mesh_lod_select_g0_bgs` — bound to the
    /// synthetic shadow camera buffer + per-draw shadow params.
    pub mesh_lod_shadow_g0_bgs: Vec<wgpu::BindGroup>,
    /// Parallel to `mesh_lod_select_g2_bgs` — bound to the asset
    /// cluster table + the shadow args buffer.
    pub mesh_lod_shadow_g2_bgs: Vec<Option<(wgpu::BindGroup, u32, u32)>>,

    // ── Mesh LOD admit stats (RKP_MESH_LOD_STATS=1 diagnostic) ─────
    /// Pass-shared 32 B (8 × u32) histograms. Slots 0..4: admitted
    /// clusters per LOD. Slots 4..8: total clusters per LOD. The
    /// shader atomic-adds into these only when `record_stats != 0`
    /// in the per-draw params; the CPU pre-clears at the start of
    /// each LOD-select pass and copies to the matching staging
    /// buffer after.
    pub mesh_lod_admit_stats_primary: wgpu::Buffer,
    pub mesh_lod_admit_stats_shadow: wgpu::Buffer,
    /// Per-pass staging buffer for async readback. Persistent — the
    /// 32 B size makes one allocation cheap. State machine:
    /// `pending = None` (idle) → `submit_copy + issue_map_async`
    /// → `pending = Some(rx)` (in flight or mapped) → `try_recv`
    /// returns Ok → `read + log + unmap` → `pending = None`. Skip
    /// the copy on frames where `pending.is_some()` so we don't
    /// double-map.
    pub mesh_lod_admit_stats_primary_staging: wgpu::Buffer,
    pub mesh_lod_admit_stats_shadow_staging: wgpu::Buffer,
    pub mesh_lod_admit_stats_primary_pending:
        Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub mesh_lod_admit_stats_shadow_pending:
        Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    /// Set by `lod_stats_finalize_*` when it queues a copy this
    /// frame; the engine calls `lod_stats_post_submit` after
    /// `queue.submit()` to pair `map_async` with the matching submit.
    /// Splitting it across the submit boundary avoids the
    /// `Buffer is still mapped` validation error wgpu fires when a
    /// submit touches a buffer the same encoder also map_async'd.
    pub mesh_lod_admit_stats_primary_needs_map: bool,
    pub mesh_lod_admit_stats_shadow_needs_map: bool,

    // ── Mesh pipeline-statistics queries (RKP_MESH_PIPESTATS=1) ────
    /// One QuerySet with 4 slots covering the mesh passes:
    /// 0 = mesh_lod_select (compute), 1 = mesh_raster (graphics),
    /// 2 = mesh_shadow_lod_select (compute), 3 = mesh_shadow_render
    /// (graphics). All five PipelineStatisticsTypes flags are
    /// enabled so each slot returns 5 × u64 = 40 bytes
    /// (vs/clipper-in/clipper-out/fs/cs invocations).
    pub mesh_pipestats_query_set: wgpu::QuerySet,
    /// 4 slots × 40 bytes = 160 bytes; rounded up to 256 for the
    /// COPY_BUFFER_ALIGNMENT requirement on resolve_query_set →
    /// copy_buffer_to_buffer.
    pub mesh_pipestats_resolve_buffer: wgpu::Buffer,
    /// MAP_READ staging buffer for async readback. Same 256 B size.
    pub mesh_pipestats_staging_buffer: wgpu::Buffer,
    pub mesh_pipestats_pending:
        Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub mesh_pipestats_needs_map: bool,

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

        // Synthetic camera buffer used by the mesh-shadow LOD-select
        // compute pass (Phase 6.4). Filled each frame from the active
        // light camera so the shader's `view_proj[1][1]` /
        // `view_proj[3][3]` switch picks the orthographic projection
        // formula and `resolution.y` carries the shadow map size.
        let mesh_lod_shadow_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rkp_vr_mesh_lod_shadow_camera"),
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
        march.set_user_shader_emit_buffers(
            device,
            &renderer.scene.user_shader_instance_buffer,
            &renderer.scene.user_shader_instance_count_buffer,
            &renderer.scene.user_shader_instance_aabbs_buffer,
            &renderer.scene.user_shader_instance_inv_world_buffer,
        );
        march.set_gbuffer(device, &gbuffer.position_view, &gbuffer.normal_view, &gbuffer.material_view, &pick_view, &gbuffer.glass_view, &gbuffer.leaf_slot_view);
        // shader_params binding deferred — `shade` owns the buffer
        // and isn't constructed yet. Wired below after `shade::new`.

        // Tile-bin pass + per-tile buffers.
        let user_shader_tile_bin =
            crate::user_shader_tile_bin_pass::UserShaderTileBinPass::new(device);
        let (
            user_shader_tile_counts_buffer,
            user_shader_tile_lists_buffer,
            user_shader_tile_count_x,
            user_shader_tile_count_y,
        ) = make_tile_bin_buffers(device, width, height);
        let user_shader_tile_bin_bg = Some(user_shader_tile_bin.build_bind_group(
            device,
            &renderer.scene.user_shader_instance_aabbs_buffer,
            &camera_buffer,
            &user_shader_tile_counts_buffer,
            &user_shader_tile_lists_buffer,
            &renderer.scene.user_shader_instance_count_buffer,
        ));
        march.set_user_shader_tile_bin_buffers(
            device,
            &user_shader_tile_counts_buffer,
            &user_shader_tile_lists_buffer,
        );

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

        // Phase 8 — shadow map pass. Three pipelines (clear /
        // setup / scatter) over a u32 storage buffer. The engine
        // writes the light_camera uniform + dispatches the chain
        // each frame; the shade pass reads the buffer directly.
        let shadow_map = ShadowMapPass::new(
            device,
            queue,
            SHADOW_MAP_DEFAULT_SIZE,
            &renderer.scene.bind_group_layout,
        );

        // Phase 3 (splat-to-mesh pivot) — depth attachment for the
        // mesh-shadow render pass. The render writes here (vertex +
        // rasterizer + depth-only, no fragment shader); the blit
        // compute pass reads it back and copies bitcast(depth) into
        // `shadow_map.shadow_buffer`. Hence both `RENDER_ATTACHMENT`
        // and `TEXTURE_BINDING` usage.
        let mesh_shadow_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rkp_mesh_shadow_depth"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_DEFAULT_SIZE,
                height: SHADOW_MAP_DEFAULT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mesh_shadow_depth_view = mesh_shadow_depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut shade = RkpShadePass::new(device, width, height);
        shade.set_shade_data(
            device,
            &renderer.shade_params_buffer,
            &renderer.lights_buffer,
            &renderer.materials_buffer,
        );
        // Phase B-redux Phase 3a — march reads the same per-material
        // shader_params buffer the shade pass owns. set_shader_params
        // is also called on each `refresh_bindings` and after every
        // `upload_shader_params` realloc in render_worker.
        march.set_shader_params(device, shade.shader_params_buffer());
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
            &shadow_map.shadow_buffer,
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
            user_shader_tile_bin,
            user_shader_tile_counts_buffer,
            user_shader_tile_lists_buffer,
            user_shader_tile_bin_bg,
            user_shader_tile_count_x,
            user_shader_tile_count_y,
            shadow_trace, shadow_map, ssao, shade, glass, volumetric, god_rays,
            gbuffer, pick_texture, pick_view, bloom, bloom_composite, tone_map,
            composite_texture, composite_view,
            readback,
            wireframe_pass, grid,
            splat_g0_bg: None,
            splat_g0_scene_epoch: u64::MAX,
            splat_g0_lights_materials_epoch: u64::MAX,
            splat_instance_buffers: Vec::new(),
            splat_instance_bind_groups: Vec::new(),
            splat_resolve_g0_bg: None,
            splat_resolve_g1_bg: None,
            splat_resolve_scene_epoch: u64::MAX,
            mesh_shadow_depth_texture,
            mesh_shadow_depth_view,
            mesh_shadow_render_g0_bg: None,
            mesh_shadow_blit_g0_bg: None,
            mesh_lod_params_buffers: Vec::new(),
            mesh_lod_args_buffers: Vec::new(),
            mesh_lod_select_g0_bgs: Vec::new(),
            mesh_lod_select_g2_bgs: Vec::new(),
            mesh_lod_shadow_camera_buffer,
            mesh_lod_shadow_params_buffers: Vec::new(),
            mesh_lod_shadow_args_buffers: Vec::new(),
            mesh_lod_shadow_g0_bgs: Vec::new(),
            mesh_lod_shadow_g2_bgs: Vec::new(),
            mesh_lod_admit_stats_primary: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_lod_admit_stats_primary"),
                size: 32,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            mesh_lod_admit_stats_shadow: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_lod_admit_stats_shadow"),
                size: 32,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            mesh_lod_admit_stats_primary_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_lod_admit_stats_primary_staging"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mesh_lod_admit_stats_shadow_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_lod_admit_stats_shadow_staging"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mesh_lod_admit_stats_primary_pending: None,
            mesh_lod_admit_stats_shadow_pending: None,
            mesh_lod_admit_stats_primary_needs_map: false,
            mesh_lod_admit_stats_shadow_needs_map: false,
            mesh_pipestats_query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("rkp_vr_mesh_pipestats"),
                ty: wgpu::QueryType::PipelineStatistics(
                    wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                        | wgpu::PipelineStatisticsTypes::CLIPPER_INVOCATIONS
                        | wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT
                        | wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
                        | wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
                ),
                count: 4,
            }),
            mesh_pipestats_resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_pipestats_resolve"),
                size: 256,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            mesh_pipestats_staging_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rkp_vr_mesh_pipestats_staging"),
                size: 256,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            mesh_pipestats_pending: None,
            mesh_pipestats_needs_map: false,
            width, height,
        }
    }

    /// Rebuild the splat-raster scene-wide (`g0`) bind group when its
    /// referenced buffers may have moved. Cheap when no rebuild is
    /// needed (one epoch comparison). Caller invokes once before
    /// `dispatch_splat`.
    pub fn refresh_splat_g0(&mut self, device: &wgpu::Device, renderer: &RkpRenderer) {
        let scene_now = renderer.scene.buffers_epoch();
        let lm_now = renderer.lights_materials_epoch();
        if self.splat_g0_bg.is_some()
            && self.splat_g0_scene_epoch == scene_now
            && self.splat_g0_lights_materials_epoch == lm_now
        {
            return;
        }
        self.splat_g0_bg = Some(renderer.splat_pass.create_g0_bind_group(
            device,
            &self.camera_buffer,
            &renderer.scene.leaf_attr_pool_buffer,
        ));
        self.splat_g0_scene_epoch = scene_now;
        self.splat_g0_lights_materials_epoch = lm_now;
    }

    /// Rebuild the mesh-shadow render + blit `g0` bind groups if not
    /// present. The underlying buffers (`shadow_map.uniform_buffer`,
    /// `shadow_map.shadow_buffer`) and the depth texture all have
    /// stable lifetimes, so once built the bind groups stay valid
    /// for the lifetime of this VR. Called once before
    /// `dispatch_mesh_shadow`.
    pub fn refresh_mesh_shadow_bindings(
        &mut self,
        device: &wgpu::Device,
        renderer: &RkpRenderer,
    ) {
        if self.mesh_shadow_render_g0_bg.is_none() {
            self.mesh_shadow_render_g0_bg =
                Some(renderer.mesh_shadow_map.create_render_g0_bind_group(
                    device,
                    &self.shadow_map.uniform_buffer,
                ));
        }
        if self.mesh_shadow_blit_g0_bg.is_none() {
            self.mesh_shadow_blit_g0_bg =
                Some(renderer.mesh_shadow_map.create_blit_g0_bind_group(
                    device,
                    &self.mesh_shadow_depth_view,
                    &self.shadow_map.shadow_buffer,
                ));
        }
    }

    /// Rebuild the splat-resolve compute pass's bind groups. `g0`
    /// (per-VR textures) is rebuilt unconditionally if absent —
    /// `resize` clears it. `g1` (scene buffers) follows the
    /// scene-buffers epoch, same trigger as the march's
    /// `scene_bind_group`.
    pub fn refresh_splat_resolve_bindings(
        &mut self,
        device: &wgpu::Device,
        renderer: &RkpRenderer,
    ) {
        if self.splat_resolve_g0_bg.is_none() {
            self.splat_resolve_g0_bg = Some(renderer.splat_resolve.create_g0_bind_group(
                device,
                &self.gbuffer.leaf_slot_view,
                &self.pick_view,
                &self.gbuffer.normal_view,
                &self.gbuffer.material_view,
                &self.gbuffer.glass_view,
            ));
        }
        let scene_now = renderer.scene.buffers_epoch();
        if self.splat_resolve_g1_bg.is_none() || self.splat_resolve_scene_epoch != scene_now {
            self.splat_resolve_g1_bg = Some(renderer.splat_resolve.create_g1_bind_group(
                device,
                &renderer.scene.leaf_attr_pool_buffer,
                &renderer.scene.color_pool_buffer,
                &renderer.scene.objects_buffer,
            ));
            self.splat_resolve_scene_epoch = scene_now;
        }
    }

    /// Grow `splat_instance_buffers` + `splat_instance_bind_groups` to
    /// at least `count` slots. Slots are reused across frames; the
    /// engine writes the current frame's matrices via
    /// [`Self::write_splat_instance`] before [`RkpRenderer::dispatch_splat`].
    ///
    /// Each slot is an 80 B uniform buffer holding one
    /// `SplatInstanceUniform` (mat4 world + object_id + 12 B pad).
    pub fn ensure_splat_instance_capacity(
        &mut self,
        device: &wgpu::Device,
        renderer: &RkpRenderer,
        count: u32,
    ) {
        let needed = count as usize;
        while self.splat_instance_buffers.len() < needed {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("splat instance uniform"),
                size: SPLAT_INSTANCE_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = renderer.splat_pass.create_g1_bind_group(device, &buf);
            self.splat_instance_buffers.push(buf);
            self.splat_instance_bind_groups.push(bg);
        }
    }

    /// Ensure per-VR mesh LOD-select state has at least `count` draw
    /// slots. Each slot owns a 16 B `MeshLodSelectParams` uniform +
    /// `g0` bind group; the args buffer is allocated lazily on first
    /// use, sized at the draw's cluster count. Slot index matches
    /// `splat_instance_buffers`.
    pub fn ensure_mesh_lod_capacity(
        &mut self,
        device: &wgpu::Device,
        renderer: &crate::rkp_renderer::RkpRenderer,
        count: u32,
    ) {
        use crate::mesh_lod_select_pass::MeshLodSelectParams;
        let needed = count as usize;
        while self.mesh_lod_params_buffers.len() < needed {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_lod_select params"),
                size: std::mem::size_of::<MeshLodSelectParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = renderer
                .mesh_lod_select_pass
                .create_g0_bind_group(device, &self.camera_buffer, &buf);
            self.mesh_lod_params_buffers.push(buf);
            self.mesh_lod_select_g0_bgs.push(bg);
            self.mesh_lod_args_buffers.push((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh_lod_select args (placeholder)"),
                    size: std::mem::size_of::<crate::mesh_lod_select_pass::DrawIndexedIndirectArgs>()
                        as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                0,
            ));
            self.mesh_lod_select_g2_bgs.push(None);
        }
    }

    /// Make sure the args buffer at `slot` has capacity for at least
    /// `cluster_count` `DrawIndexedIndirectArgs`. Grows the buffer
    /// to a power-of-two size and invalidates the cached `g2` bind
    /// group when it does.
    pub fn ensure_mesh_lod_args_capacity(
        &mut self,
        device: &wgpu::Device,
        slot: u32,
        cluster_count: u32,
    ) {
        use crate::mesh_lod_select_pass::DrawIndexedIndirectArgs;
        let i = slot as usize;
        let (_, cap) = self.mesh_lod_args_buffers[i];
        if cap >= cluster_count {
            return;
        }
        let new_cap = cluster_count.next_power_of_two().max(64);
        let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_lod_select args"),
            size: (new_cap as u64) * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.mesh_lod_args_buffers[i] = (new_buf, new_cap);
        // Invalidate cached g2 — it referenced the old args buffer.
        self.mesh_lod_select_g2_bgs[i] = None;
    }

    /// Phase 6.4 sibling of `ensure_mesh_lod_capacity` for the
    /// shadow LOD-select pass. One per-draw shadow params uniform +
    /// args buffer + g0 bind group bound to the synthetic shadow
    /// camera. Args buffer is allocated lazily by
    /// `ensure_mesh_lod_shadow_args_capacity` to avoid waste on
    /// frames where shadow is gated off.
    pub fn ensure_mesh_lod_shadow_capacity(
        &mut self,
        device: &wgpu::Device,
        renderer: &crate::rkp_renderer::RkpRenderer,
        count: u32,
    ) {
        use crate::mesh_lod_select_pass::MeshLodSelectParams;
        let needed = count as usize;
        while self.mesh_lod_shadow_params_buffers.len() < needed {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_lod_shadow params"),
                size: std::mem::size_of::<MeshLodSelectParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = renderer.mesh_lod_select_pass.create_g0_bind_group(
                device,
                &self.mesh_lod_shadow_camera_buffer,
                &buf,
            );
            self.mesh_lod_shadow_params_buffers.push(buf);
            self.mesh_lod_shadow_g0_bgs.push(bg);
            self.mesh_lod_shadow_args_buffers.push((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh_lod_shadow args (placeholder)"),
                    size: std::mem::size_of::<crate::mesh_lod_select_pass::DrawIndexedIndirectArgs>()
                        as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                0,
            ));
            self.mesh_lod_shadow_g2_bgs.push(None);
        }
    }

    /// Grow the per-slot shadow args buffer to fit `cluster_count`
    /// entries; invalidates the cached `g2` shadow bind group.
    pub fn ensure_mesh_lod_shadow_args_capacity(
        &mut self,
        device: &wgpu::Device,
        slot: u32,
        cluster_count: u32,
    ) {
        use crate::mesh_lod_select_pass::DrawIndexedIndirectArgs;
        let i = slot as usize;
        let (_, cap) = self.mesh_lod_shadow_args_buffers[i];
        if cap >= cluster_count {
            return;
        }
        let new_cap = cluster_count.next_power_of_two().max(64);
        let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_lod_shadow args"),
            size: (new_cap as u64) * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.mesh_lod_shadow_args_buffers[i] = (new_buf, new_cap);
        self.mesh_lod_shadow_g2_bgs[i] = None;
    }

    /// Populate `mesh_lod_shadow_camera_buffer` from the active light
    /// camera. The LOD-select shader reads `position` (eye for
    /// view_distance — unused under ortho), `view_proj` (the [1][1]
    /// is the ortho-or-perspective focal factor; [3][3] = 1 picks
    /// the ortho path), and `resolution.y` (shadow map height).
    /// Other CameraUniforms fields aren't read; we leave them zero.
    pub fn write_mesh_lod_shadow_camera(
        &self,
        queue: &wgpu::Queue,
        light: &crate::shadow_map_pass::LightCameraUniform,
    ) {
        // Eye in world space = view_proj_inv * (0,0,0,1) / w.
        // For ortho `view_proj_inv[3][3] != 0` so the divide is fine.
        let m = light.view_proj_inv;
        let ex = m[3][0];
        let ey = m[3][1];
        let ez = m[3][2];
        let ew = m[3][3].abs().max(1e-6) * m[3][3].signum();
        let eye = [ex / ew, ey / ew, ez / ew, 0.0];

        let cam = CameraUniforms {
            position: eye,
            forward: [0.0; 4],
            right: [0.0; 4],
            up: [0.0; 4],
            resolution: [light.shadow_map_size[0] as f32, light.shadow_map_size[1] as f32],
            jitter: [0.0; 2],
            layer_mask: 0,
            focus_object_id: 0,
            _pad: [0; 2],
            prev_vp: [[0.0; 4]; 4],
            view_proj: light.view_proj,
        };
        queue.write_buffer(
            &self.mesh_lod_shadow_camera_buffer,
            0,
            bytemuck::bytes_of(&cam),
        );
    }

    /// Mesh LOD-select admit-stats lifecycle. Called by `dispatch_mesh`
    /// + `dispatch_mesh_shadow` when `RKP_MESH_LOD_STATS=1`. Three
    /// phases composed in order per frame:
    ///
    /// 1. `lod_stats_drain_*`: try_recv the previous frame's pending
    ///    map; if it's mapped, read 32 B + log + unmap + reset to
    ///    idle. Cheap when nothing is in flight.
    /// 2. `lod_stats_clear_*`: encoder.clear_buffer the histogram so
    ///    this frame's atomics start at zero. Always called before
    ///    the LOD-select dispatch (no-op if stats disabled — clear
    ///    is harmless either way and avoids an env-var-gated branch
    ///    on the hot path).
    /// 3. `lod_stats_finalize_*`: encoder.copy_buffer_to_buffer the
    ///    histogram → staging buffer. Called immediately after the
    ///    LOD-select dispatch finishes. Skipped when a map is already
    ///    pending so we don't overwrite in-flight data.
    pub fn lod_stats_drain_primary(&mut self, label: &str) {
        Self::lod_stats_drain(
            &self.mesh_lod_admit_stats_primary_staging,
            &mut self.mesh_lod_admit_stats_primary_pending,
            label,
        );
    }
    pub fn lod_stats_drain_shadow(&mut self, label: &str) {
        Self::lod_stats_drain(
            &self.mesh_lod_admit_stats_shadow_staging,
            &mut self.mesh_lod_admit_stats_shadow_pending,
            label,
        );
    }
    fn lod_stats_drain(
        staging: &wgpu::Buffer,
        pending: &mut Option<
            std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        >,
        label: &str,
    ) {
        let Some(rx) = pending.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(())) => {
                let slice = staging.slice(..);
                let data = slice.get_mapped_range();
                if data.len() >= 32 {
                    let mut admitted = [0u32; 4];
                    let mut total = [0u32; 4];
                    for i in 0..4 {
                        admitted[i] = u32::from_le_bytes([
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        ]);
                        total[i] = u32::from_le_bytes([
                            data[16 + i * 4],
                            data[16 + i * 4 + 1],
                            data[16 + i * 4 + 2],
                            data[16 + i * 4 + 3],
                        ]);
                    }
                    let admitted_total: u32 = admitted.iter().sum();
                    let evaluated_total: u32 = total.iter().sum();
                    eprintln!(
                        "[mesh_lod_stats {label}] evaluated={evaluated_total} admitted={admitted_total} | \
                         lod0 {}/{} | lod1 {}/{} | lod2 {}/{} | lod3 {}/{}",
                        admitted[0], total[0],
                        admitted[1], total[1],
                        admitted[2], total[2],
                        admitted[3], total[3],
                    );
                }
                drop(data);
                staging.unmap();
                *pending = None;
            }
            Ok(Err(e)) => {
                eprintln!("[mesh_lod_stats {label}] map_async error: {e:?}");
                *pending = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Still in flight; check again next frame.
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Sender dropped without sending — shouldn't happen
                // since the closure stays alive until the callback
                // fires. Reset defensively.
                *pending = None;
            }
        }
    }

    pub fn lod_stats_clear_primary(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.mesh_lod_admit_stats_primary, 0, None);
    }
    pub fn lod_stats_clear_shadow(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.mesh_lod_admit_stats_shadow, 0, None);
    }

    /// Queue the GPU-side copy of the histogram into the staging
    /// buffer. Returns `true` if the copy was actually queued (i.e.,
    /// no map is in flight); the caller MUST then call
    /// `lod_stats_issue_map_async_primary` after `queue.submit()` to
    /// pair the map_async with this submit. Issuing map_async before
    /// submit triggers `Buffer is still mapped` validation errors
    /// because wgpu reads the map state at submit time.
    pub fn lod_stats_finalize_primary(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.mesh_lod_admit_stats_primary_pending.is_some() {
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.mesh_lod_admit_stats_primary,
            0,
            &self.mesh_lod_admit_stats_primary_staging,
            0,
            32,
        );
        self.mesh_lod_admit_stats_primary_needs_map = true;
    }
    pub fn lod_stats_finalize_shadow(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.mesh_lod_admit_stats_shadow_pending.is_some() {
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.mesh_lod_admit_stats_shadow,
            0,
            &self.mesh_lod_admit_stats_shadow_staging,
            0,
            32,
        );
        self.mesh_lod_admit_stats_shadow_needs_map = true;
    }

    /// Mesh pipeline-statistics drain: read the previous frame's
    /// 4-slot u64 array if the staging buffer is mapped, log p
    /// er-pass VS / clipper-in / clipper-out / FS / CS counts.
    /// Cheap when nothing's in flight.
    pub fn pipestats_drain(&mut self) {
        let Some(rx) = self.mesh_pipestats_pending.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(())) => {
                let slice = self.mesh_pipestats_staging_buffer.slice(..);
                let data = slice.get_mapped_range();
                if data.len() >= 160 {
                    // Layout: 4 slots × 5 × u64. Stat order matches
                    // the `PipelineStatisticsTypes` bit order:
                    // VS, CLIPPER_IN, CLIPPER_OUT, FS, CS.
                    let read_slot = |slot: usize| -> [u64; 5] {
                        let base = slot * 40;
                        let mut out = [0u64; 5];
                        for i in 0..5 {
                            let off = base + i * 8;
                            out[i] = u64::from_le_bytes(
                                data[off..off + 8].try_into().unwrap(),
                            );
                        }
                        out
                    };
                    let labels = [
                        "mesh_lod_select",
                        "mesh_raster",
                        "mesh_shadow_lod_select",
                        "mesh_shadow_render",
                    ];
                    for (slot, label) in labels.iter().enumerate() {
                        let s = read_slot(slot);
                        eprintln!(
                            "[mesh_pipestats {label}] vs={} clipper_in={} clipper_out={} fs={} cs={}",
                            s[0], s[1], s[2], s[3], s[4],
                        );
                    }
                }
                drop(data);
                self.mesh_pipestats_staging_buffer.unmap();
                self.mesh_pipestats_pending = None;
            }
            Ok(Err(e)) => {
                eprintln!("[mesh_pipestats] map_async error: {e:?}");
                self.mesh_pipestats_pending = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.mesh_pipestats_pending = None;
            }
        }
    }

    /// Resolve the query set into the resolve buffer + copy to
    /// staging. Called once per frame after both mesh passes have
    /// emitted their begin/end_pipeline_statistics_query pairs.
    /// Skip when an earlier frame's map is still pending so we
    /// don't overwrite mid-flight data.
    pub fn pipestats_finalize(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.mesh_pipestats_pending.is_some() {
            return;
        }
        encoder.resolve_query_set(
            &self.mesh_pipestats_query_set,
            0..4,
            &self.mesh_pipestats_resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.mesh_pipestats_resolve_buffer,
            0,
            &self.mesh_pipestats_staging_buffer,
            0,
            160,
        );
        self.mesh_pipestats_needs_map = true;
    }

    /// Engine call after `queue.submit()`. Issues map_async on any
    /// staging buffers the per-frame finalize methods flagged.
    /// Bundles all the diagnostic readback stagings (LOD admit
    /// histograms + pipeline-statistics query results) so the
    /// engine has one symmetric pre/post-submit pair.
    pub fn lod_stats_post_submit(&mut self) {
        if self.mesh_lod_admit_stats_primary_needs_map {
            let (tx, rx) = std::sync::mpsc::channel();
            self.mesh_lod_admit_stats_primary_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            self.mesh_lod_admit_stats_primary_pending = Some(rx);
            self.mesh_lod_admit_stats_primary_needs_map = false;
        }
        if self.mesh_lod_admit_stats_shadow_needs_map {
            let (tx, rx) = std::sync::mpsc::channel();
            self.mesh_lod_admit_stats_shadow_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            self.mesh_lod_admit_stats_shadow_pending = Some(rx);
            self.mesh_lod_admit_stats_shadow_needs_map = false;
        }
        if self.mesh_pipestats_needs_map {
            let (tx, rx) = std::sync::mpsc::channel();
            self.mesh_pipestats_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            self.mesh_pipestats_pending = Some(rx);
            self.mesh_pipestats_needs_map = false;
        }
    }

    /// Write one frame's `SplatInstanceUniform` for the given slot.
    /// Caller must have already extended the slot vector via
    /// `ensure_splat_instance_capacity`.
    ///
    /// `skinning_mode` follows the [`SplatInstanceUniform`] convention
    /// (`0` LBS / `1` DQS / `SKINNING_MODE_NONE` rest-pose); the two
    /// `bone_offset_*` values are read by the mesh VS only when the
    /// matching mode is selected.
    pub fn write_splat_instance(
        &self,
        queue: &wgpu::Queue,
        slot: u32,
        world: &[[f32; 4]; 4],
        object_id: u32,
        bone_offset_lbs: u32,
        bone_offset_dqs: u32,
        skinning_mode: u32,
    ) {
        let uniform = SplatInstanceUniform {
            world: *world,
            object_id,
            bone_offset_lbs,
            bone_offset_dqs,
            skinning_mode,
        };
        queue.write_buffer(
            &self.splat_instance_buffers[slot as usize],
            0,
            bytemuck::bytes_of(&uniform),
        );
    }

    /// Reset per-tile counts to 0. Call BEFORE the tile-bin dispatch
    /// each frame; otherwise stale counts from the previous frame leak.
    pub fn clear_user_shader_tile_counts(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.user_shader_tile_counts_buffer, 0, None);
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
            // The user-shader instance buffers live on the scene; if
            // the scene buffers epoch bumped they may have moved too.
            // Re-wire so march's params bg points at the live handles.
            self.march.set_user_shader_emit_buffers(
                device,
                &renderer.scene.user_shader_instance_buffer,
                &renderer.scene.user_shader_instance_count_buffer,
                &renderer.scene.user_shader_instance_aabbs_buffer,
                &renderer.scene.user_shader_instance_inv_world_buffer,
            );
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
            // Phase B-redux Phase 3a — refresh march's shader_params
            // binding; mirrors the per-frame call in render_worker so
            // a buffer realloc here doesn't leave march on stale data.
            self.march.set_shader_params(device, self.shade.shader_params_buffer());
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

        // Splat-resolve g0 references gbuffer texture views which all
        // just moved. Drop it — `refresh_splat_resolve_bindings` will
        // rebuild on next dispatch.
        self.splat_resolve_g0_bg = None;

        // Tile-bin per-tile buffers depend on resolution. Reallocate
        // and rebuild the cached bind group + march's tile-bin
        // bindings.
        let (counts_buf, lists_buf, tx, ty) = make_tile_bin_buffers(device, width, height);
        self.user_shader_tile_counts_buffer = counts_buf;
        self.user_shader_tile_lists_buffer = lists_buf;
        self.user_shader_tile_count_x = tx;
        self.user_shader_tile_count_y = ty;
        self.user_shader_tile_bin_bg = Some(self.user_shader_tile_bin.build_bind_group(
            device,
            &renderer.scene.user_shader_instance_aabbs_buffer,
            &self.camera_buffer,
            &self.user_shader_tile_counts_buffer,
            &self.user_shader_tile_lists_buffer,
            &renderer.scene.user_shader_instance_count_buffer,
        ));
        self.march.set_user_shader_tile_bin_buffers(
            device,
            &self.user_shader_tile_counts_buffer,
            &self.user_shader_tile_lists_buffer,
        );

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
        // depth buffer stays at SHADOW_MAP_DEFAULT_SIZE through
        // resize. Re-binding picks up the same buffer.
        self.shade.set_shadow_and_ssao(
            device,
            &self.shadow_trace.output_view,
            &self.ssao.output_view,
            &self.shadow_map.shadow_buffer,
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

/// Allocate the user-shader tile-bin buffers for a given resolution.
/// Returns `(counts_buffer, lists_buffer, tile_count_x, tile_count_y)`.
/// Tile size is hardcoded to 8×8 pixels — matches the march workgroup
/// layout.
fn make_tile_bin_buffers(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Buffer, wgpu::Buffer, u32, u32) {
    use crate::user_shader_tile_bin_pass::MAX_INSTANCES_PER_TILE;
    let tile_count_x = width.div_ceil(8);
    let tile_count_y = height.div_ceil(8);
    let tile_count = tile_count_x * tile_count_y;
    let counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("user_shader_tile_counts"),
        size: tile_count as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let lists_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("user_shader_tile_lists"),
        size: tile_count as u64 * MAX_INSTANCES_PER_TILE as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (counts_buffer, lists_buffer, tile_count_x, tile_count_y)
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
        // RENDER_ATTACHMENT for the splat raster path (splat.wesl writes
        // pick at @location(3)). Compute march writes via STORAGE_BINDING
        // — both bits live here so the same texture is reachable from
        // either pipeline without reallocating.
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
