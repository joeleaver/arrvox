//! Per-tick payload from the sim thread to the render thread.
//!
//! ## Why
//!
//! The engine runs on its own thread (the "sim thread") and used to
//! drive both simulation and rendering in one tick loop. That meant
//! sim time = max(CPU work, GPU submit + readback drain), so a heavy
//! ECS pass would stall GPU submission, and a slow GPU readback would
//! stall the next sim step. As we head toward heavier sim work
//! (water/ocean, particles), that single thread becomes the
//! bottleneck.
//!
//! After the split, sim and render run on independent threads. Each
//! tick the sim thread builds a [`RenderFrame`] — every byte of CPU
//! state the renderer needs to produce one output frame — and pushes
//! it through a single-slot inbox to the render thread. The render
//! thread owns wgpu (`device`, `queue`, [`RkpRenderer`], the per-
//! viewport renderers, the pick readback buffer); it consumes the
//! latest [`RenderFrame`] each iteration, runs the wgpu work, and
//! returns a [`RenderResult`] back to sim.
//!
//! ## Channel semantics
//!
//! The inbox is **newest-wins**: if sim outpaces render (e.g. render
//! is GPU-bound at 200 Hz while sim is at 600 Hz), the older
//! unconsumed frame is dropped. Sim never blocks on render, and
//! render never sees a stale frame when a fresh one is waiting.
//!
//! ## Field ownership rules
//!
//! - **Sim builds, render reads**: anything derived from `World`,
//!   environment settings, gizmo state, procedural trees, or camera
//!   input. Snapshotted into this struct.
//! - **Render owns**: wgpu resources, the GPU profiler, in-flight
//!   pick state, frame readback rings.
//! - **Shared via `Arc<Mutex>`**: `scene_mgr` only — the bake worker
//!   already accesses it lock-free between sim ticks; render takes the
//!   same lock briefly when uploading geometry.
//!
//! Anything sim mutates *after* sending the snapshot must be a fresh
//! allocation or copied — render reads the snapshot's fields directly,
//! so aliasing into sim's live state would race.

use std::sync::Arc;

use glam::{Affine3A, Mat4, Vec3};

use rkp_core::OverlayEntry;
use rkp_render::{
    rkp_atmosphere::AtmosphereFrameParams,
    rkp_god_rays::GodRayParams,
    rkp_gpu_object::{RkpGpuAsset, RkpGpuInstance},
    rkp_grid::GridParams,
    rkp_scene::CameraUniforms,
    rkp_shade::{GpuLight, GpuMaterial, ShadeParams},
    rkp_volumetric::{CloudParams, VolumetricParams},
    BuildPreviewMode, LineVertex, RenderMode, SkinBatchScratch,
};

use crate::viewport::ViewportId;

/// One render frame's worth of CPU state, shipped sim → render.
pub struct RenderFrame {
    /// Monotonic frame counter. Render echoes this back in
    /// [`RenderResult::frame_index`] so sim can correlate timings.
    pub frame_index: u64,

    /// Per-asset GPU records — deduped by `octree_root`. Built by sim's
    /// `update_scene_gpu` alongside `gpu_instances`. The instance side's
    /// `asset_id` indexes into this vec.
    pub gpu_assets: Vec<RkpGpuAsset>,
    /// Per-instance GPU records — one per renderable entity. Render
    /// uploads (`upload_frame`) regardless of dirtiness — the cost is
    /// one `queue.write_buffer`, cheap; sim sets `gpu_objects_dirty`
    /// purely as a hint for stat tracking, not as a gate.
    pub gpu_instances: Vec<RkpGpuInstance>,
    /// Flat per-instance paint overlay entries. Each `RkpGpuInstance`
    /// uses its `overlay_offset` + `overlay_count` to slice into this
    /// vec. Empty when no entity has been painted.
    pub gpu_instance_overlays: Vec<OverlayEntry>,
    pub gpu_objects_dirty: bool,

    /// Monotonic counter from `scene_mgr.geometry_epoch()`. Render
    /// keeps its own `last_uploaded_geometry_epoch` and calls
    /// `upload_geometry` whenever this exceeds it. Replaces the
    /// previous boolean dirty flag, which could be lost if a
    /// snapshot carrying `geometry_dirty=true` was dropped by the
    /// newest-wins inbox before render saw it. The next snapshot
    /// always carries the latest epoch, so render catches up no
    /// matter how many intermediate snapshots are dropped.
    pub geometry_epoch: u64,

    /// Paint cursor's brush-overlay epoch. Bumped every time
    /// `RkpSceneManager::update_brush_overlay` runs (cursor moves or
    /// radius changes). Render uploads the per-leaf distance buffer
    /// to each VR's shade pass when this moves ahead.
    pub brush_overlay_epoch: u64,

    /// Paint-data epoch. Bumped by `apply_paint_sphere` when a stroke
    /// writes to leaf_attr / color. Render slice-uploads only the
    /// dirty slot range when this moves ahead — avoids the full
    /// scene re-upload that `geometry_epoch` would trigger.
    pub paint_epoch: u64,

    /// Current material palette. Sim builds + ships every tick (cheap
    /// — small Vec). Render uploads every frame; the cost is one
    /// `queue.write_buffer` of ~1 KB. Same robustness rationale as
    /// `geometry_epoch`: the previous "ship only when dirty" pattern
    /// could lose the upload if the carrying snapshot got dropped.
    pub materials: Vec<GpuMaterial>,

    /// Per-material user-shader params buffer — one [f32; 8] slot per
    /// material, parallel to `materials`. Sim builds this from the
    /// active `UserShaderRegistry` + each material's `shader_params`
    /// HashMap. Render uploads to each viewport's shade pass alongside
    /// `materials`.
    pub shader_params_slots: Vec<[f32; 8]>,

    /// Composed user-shader chunk for the deferred shade pass, plus
    /// its source hash. Render thread compares hash against each
    /// viewport shade pass's last-seen value and rebuilds the
    /// pipeline when they differ. Empty string on the no-shaders
    /// path; the in-tree identity stub keeps that case working.
    pub user_shader_shade_chunk: String,
    pub user_shader_source_hash: u64,

    /// Composed user-shader chunk for the prototype bake pass.
    /// Defines `dispatch_user_proto(...)`. Spliced into
    /// `user_shader_proto.wgsl` between its USER_PROTO_DISPATCH markers.
    /// Empty when no shader declares an `@instance_proto`; the in-tree
    /// identity stub returns a `voxel_emit_skip()`. Each shader's proto
    /// is baked once into the shared pool and re-used by every emitted
    /// instance.
    pub user_shader_proto_chunk: String,

    /// Editor snapshots of all currently-registered user shaders.
    /// Mirrors `StateUpdate.user_shaders`.
    pub user_shader_infos: Vec<rkp_render::shader_composer::UserShaderInfo>,

    /// Full registry entries used by the prototype bake + emit pass.
    /// The render thread walks these to deduplicate per-shader proto
    /// assets against the proto cache and to drive the per-shader
    /// emit-shader compose. Heavier than `user_shader_infos` (carries
    /// WGSL bodies + InstanceLayout). Cost is one `Vec` clone per
    /// frame — negligible against the snapshot's other allocations.
    /// Empty when no shaders are registered.
    pub user_shader_entries: Vec<rkp_render::shader_composer::UserShaderEntry>,

    /// Per-leaf records for the user-shader emit pass. One entry per
    /// painted leaf cell whose material has an `instance_at` hook.
    /// The emit pass dispatches one thread per leaf, calls the
    /// shader's `instance_at` for k = 0..MAX_EMITS_PER_LEAF, and
    /// writes one `RkpInstance` per accepted instance into
    /// `RkpScene::user_shader_instance_buffer`.
    /// Wrapped in `Arc` so the per-frame snapshot handoff is a refcount
    /// bump rather than a 100+ MB memcpy when paint covers a million-
    /// plus host leaves. Sim rebuilds the inner `Vec` only on paint or
    /// geometry epoch change; in steady state the same `Arc` is shared
    /// across frames.
    pub painted_leaves: std::sync::Arc<Vec<rkp_render::user_shader_emit_pass::EmitLeaf>>,

    /// Composed `emit` chunk — per-shader `instance_at` +
    /// `inst_world_matrix` bodies + dispatch switch, spliced into
    /// `user_shader_emit.wgsl` between its USER_EMIT_DISPATCH markers.
    /// Empty when no shader has an `instance_at` hook.
    pub user_shader_emit_chunk: String,

    /// Scene-wide light list (sun + entity point/spot lights), in the
    /// order the shade shader expects (entry 0 = sun).
    pub lights: Vec<GpuLight>,

    /// Base shade params for this frame. Render writes a per-VR copy
    /// (with the `isolation` flag set) before each viewport's submit.
    pub shade_params_base: ShadeParams,

    /// Current bloom + tonemap settings, shipped every tick. Render
    /// applies to every VR each frame — same drop-safety rationale
    /// as `materials`. The set_* calls are tiny `queue.write_buffer`
    /// writes.
    pub env_update: EnvUpdate,

    /// One entry per visible viewport, in submission order.
    pub viewports: Vec<RenderViewport>,

    /// Skin scatter dispatch payload. `None` when skinning is
    /// disabled, no skinned entities are present, or when sim
    /// detected the pose set was byte-identical to the previous frame
    /// (`skin_reuse`) — in that case render leaves last frame's
    /// `bone_field` intact and skips the scatter encoder entirely.
    pub skin: Option<RenderSkin>,

    /// Bytes packed by sim's `BoneMatrixAllocator` for the shade pass
    /// (LBS) — concatenated per-entity poses. Empty when no skinned
    /// entities exist this frame.
    pub bone_matrix_lbs: Vec<u8>,
    /// Same as above but in dual-quaternion form for the DQS path.
    pub bone_matrix_dqs: Vec<u8>,

    /// Pending click-pick. Render encodes the gbuf copy this frame
    /// and kicks off the async map; the result lands back via
    /// [`RenderResult::pick_result`].
    pub pending_pick: Option<PendingPick>,

    /// Sim's smoothed cloud-sun attenuation. Render multiplies sun
    /// color by this; the *raw* value comes back in
    /// [`RenderResult::cloud_sun_atten_raw`] for sim to feed the next
    /// frame's smoothing.
    pub cloud_sun_atten: f32,

    /// Renderer toggles. Snapshotted so a flip mid-frame doesn't tear.
    pub lod_enabled: bool,
    pub surfacenet_enabled: bool,

    /// Shadow trace step cap (from environment). Snapshotted so
    /// render doesn't need to read environment on its thread.
    pub shadow_steps: u32,
}

/// Per-viewport render data — enough for the render thread to upload
/// camera, screen-AABBs, vol/cloud/atmo/god-ray params, and dispatch
/// the per-VR pass chain without consulting sim state.
pub struct RenderViewport {
    pub id: ViewportId,
    pub width: u32,
    pub height: u32,
    pub mode: RenderMode,
    pub preview_mode: BuildPreviewMode,
    pub camera: CameraUniforms,

    /// Cached `view_proj * world` cull plane data, packed for the
    /// march tile-cull binding. Sim builds against the same gpu_objects
    /// list it ships above so indices line up.
    pub screen_aabbs_bytes: Vec<u8>,

    /// Per-tile object lists for the march pass. `tile_offsets` is a
    /// prefix-sum (length `num_tiles + 1`) and `tile_object_ids` is a
    /// flat u32 array grouped by tile. Replaces the retired 32-object
    /// bitmask so scenes with arbitrary object counts render correctly.
    /// Built on the sim thread from `screen_aabbs_bytes`.
    pub tile_offsets_bytes: Vec<u8>,
    pub tile_object_ids_bytes: Vec<u8>,
    /// Tile grid width (matches the viewport's render width divided by
    /// the shader's workgroup tile size, rounded up). Shader computes
    /// `tile_idx = ty * tile_count_x + tx`.
    pub tile_count_x: u32,

    /// `view * proj` glam matrix used for wireframe overlays. Same
    /// data as `camera.view_proj` but kept as `Mat4` for convenience.
    pub vp_matrix: Mat4,

    /// Per-VR per-frame param uploads.
    pub vol_params: VolumetricParams,
    pub cloud_params: CloudParams,
    pub atmo_frame: AtmosphereFrameParams,
    pub god_ray_params: GodRayParams,

    /// Per-VR shade-params override (isolation bit + clamped lights).
    /// Render writes the shared shade-params buffer with this just
    /// before this VR's submit.
    pub shade_params: ShadeParams,

    /// Bloom-composite intensity for this VR (zero in isolation mode
    /// since the bloom mips are stale — composite acts as passthrough).
    pub bloom_composite_intensity: f32,

    /// Optional grid params override — set on BUILD so the studio
    /// floor pins to the previewed entity instead of world origin.
    pub grid_override: Option<GridParams>,

    /// Wireframe verts to draw on this VR's composite (gizmo on MAIN,
    /// procedural-node gizmo on BUILD). Empty = pass is skipped.
    pub wireframe_verts: Vec<LineVertex>,

    /// Whether the editor overlay layer (gizmo wireframes) is active
    /// for this VR. Sim derives from the layer mask; render gates the
    /// wireframe draw on this.
    pub show_editor_overlays: bool,

    /// Procedural raymarch preview state, present only when
    /// `preview_mode == Raymarch` and an entity with
    /// `ProceduralGeometry` is selected.
    pub proc_raymarch: Option<RenderProcRaymarch>,
}

/// Procedural-raymarch payload for a single viewport. Sim flattens
/// the procedural tree once per frame, here.
pub struct RenderProcRaymarch {
    pub instructions: Vec<rkp_procedural::ProcInstruction>,
    /// Subset of `instructions` containing only ghost-role primitives
    /// (cutters and intersected children). Pre-filtered sim-side so
    /// render can upload directly.
    pub ghost_instructions: Vec<rkp_procedural::ProcInstruction>,
    /// Entity scene-id for the previewed object (zero = none).
    pub object_id: u32,
    /// World transform of the previewed entity — pins the raymarched
    /// tree to the entity's position so MAIN-gizmo edits update the
    /// BUILD preview.
    pub entity_world: Affine3A,
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    /// NodeId of the currently-selected procedural sub-node (for the
    /// outline overlay). `None` = nothing selected → render writes
    /// `OutlineParams::NONE`.
    pub selected_node: Option<u32>,
}

/// Skin-scatter batched dispatch — folded by sim's
/// `plan_skin_dispatch`, fired by render in one compute pass.
pub struct RenderSkin {
    /// Total bytes the bone-field buffer must hold this frame.
    pub bone_field_bytes: u64,
    /// Total bytes the bone-field occupancy bitmap must hold this frame.
    pub bone_field_occ_bytes: u64,
    /// Pre-built batched dispatch — every skinned entity folded in.
    pub batch: SkinBatchScratch,
}

/// One-shot pick request. The render thread encodes the texture copy
/// during the matching viewport's submit and returns the sampled
/// payload via [`RenderResult::pick_result`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingPick {
    pub viewport: ViewportId,
    pub x: u32,
    pub y: u32,
    pub kind: PickKind,
}

/// What to decode from a pick readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickKind {
    /// MAIN viewport — `gbuf_material` (Rg32Uint) → packed material
    /// lo/hi → scene_id resolved sim-side.
    Material,
    /// BUILD raymarch — `gbuf_pick` (R32Uint) → procedural NodeId.
    ProceduralNode,
}

/// Bloom + tonemap params applied scene-wide whenever sim's
/// environment is dirty.
#[derive(Debug, Clone, Copy)]
pub struct EnvUpdate {
    pub exposure: f32,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_intensity: f32,
}

/// Reverse channel: render → sim, one per produced frame.
pub struct RenderResult {
    /// Echoes [`RenderFrame::frame_index`] so sim can correlate.
    pub frame_index: u64,
    /// Decoded pick payload, present iff render finished a pick this
    /// frame (the pick request may have come from this frame or
    /// earlier — async readback can lag a frame or two).
    pub pick_result: Option<PickResult>,
    /// Latest cloud-sun attenuation read from MAIN's volumetric pass.
    /// Sim feeds this into its EMA so the next frame's snapshot
    /// carries a smoothed value to render. NaN if MAIN wasn't visible
    /// or the readback hasn't completed yet (sim should treat NaN as
    /// "no update; keep last").
    pub cloud_sun_atten_raw: f32,
    /// Per-pass GPU timings drained from `wgpu-profiler`. Empty
    /// during the first ~3-frame warmup.
    pub gpu_passes: Vec<(String, f32)>,
    /// Wall-clock interval between this render iteration's start and
    /// the previous iteration's start, in milliseconds. This is the
    /// render thread's actual production cadence — what the editor
    /// surface sees as a frame rate, accounting for `render_pacing`
    /// caps, sim availability, and GPU backpressure. Sim EMA-smooths
    /// this and reports it as the panel's "FPS" so the headline
    /// number reflects what's on screen rather than sim CPU headroom.
    /// `None` for the very first iteration (no prior interval to
    /// measure against yet).
    pub render_dt_ms: Option<f32>,
    /// Wall-clock interval since the *previous iteration that actually
    /// shipped pixels to the editor surface*, in milliseconds.
    /// `render_dt_ms` counts every render-thread iteration, including
    /// ones that re-submitted the same sim snapshot and did not fire
    /// `frame_callback`; this field only counts iterations where a
    /// fresh frame reached the display. Sim EMA-smooths it into the
    /// panel's **Delivered FPS** — the honest "what did the user
    /// actually see" number, which diverges from Render FPS whenever
    /// sim is slower than render or `MIN_FRAME_CALLBACK_INTERVAL`
    /// drops ships. `None` when this iteration didn't ship.
    pub delivered_dt_ms: Option<f32>,
}

/// Decoded pick result returned to sim.
#[derive(Debug, Clone, Copy)]
pub struct PickResult {
    pub viewport: ViewportId,
    pub kind: PickKind,
    /// Two raw u32s sampled from the relevant texture — sim does the
    /// final scene_id → entity lookup since it owns the mapping.
    /// For `Material`: `[packed_r, packed_g]` of the gbuf_material pixel.
    /// For `ProceduralNode`: `[primitive_node_id, 0]`.
    pub raw_payload: [u32; 2],
    /// World-space surface position sampled from `gbuf_position` at the
    /// pick pixel. `None` when the ray missed geometry (the shader
    /// writes `hit_distance = 1e10` for misses, which we filter out
    /// here). Used by drag-drop to snap spawns onto geometry.
    pub position: Option<Vec3>,
}

/// One-time render-thread spawn args.
pub struct RenderInit {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub initial_width: u32,
    pub initial_height: u32,
    pub scene_mgr: Arc<std::sync::Mutex<rkp_render::RkpSceneManager>>,
    /// How fast the render thread re-renders. When this exceeds the
    /// sim tick rate, render interpolates between the last two
    /// received snapshots. When it lags sim, the newest-wins inbox
    /// drops stale snapshots so render stays on the freshest state.
    pub render_pacing: crate::engine::PacingMode,
}

/// Out-of-band command from sim → render that doesn't fit naturally
/// in [`RenderFrame`] (resize, viewport visibility, surface
/// configuration). Most per-frame state still travels in the snapshot;
/// these are the rare aperiodic events.
pub enum RenderCommand {
    /// Resize a viewport's render target. Render reallocates the
    /// gbuffer + composite + readback chain.
    ResizeViewport { id: ViewportId, width: u32, height: u32 },
    /// Show or hide a viewport. Hidden viewports skip render entirely.
    SetViewportVisible { id: ViewportId, visible: bool },
    /// Replace a viewport's render mode (InSitu vs. Isolation).
    SetViewportMode { id: ViewportId, mode: RenderMode },
    /// Replace BUILD viewport's preview mode (Voxel vs. Raymarch).
    SetBuildPreviewMode(BuildPreviewMode),
    /// Graceful shutdown.
    Shutdown,
}
