//! `EngineState` — the internal runtime state of `ArvxEngine`.
//!
//! Single source of truth for the engine's per-tick state: ECS,
//! scene_mgr handles, render/bake workers, input, camera, project state,
//! gizmo state, per-frame dirty flags. Every `impl EngineState` block in
//! a sibling `engine/*_ops.rs` file mutates fields on this struct.
//!
//! The struct and all fields are `pub(crate)` so the sibling modules
//! (entity_ops, procedural_ops, command_handler, etc.) can access them
//! directly. Nothing outside the `arvx_engine` crate touches this type.

use arvx_render::arvx_gpu_object::{ArvxGpuAsset, ArvxGpuInstance};
use arvx_render::arvx_scene_manager::ArvxSceneManager;

mod constructor;

/// One tile's worth of painted-material info: AABB + count of painted
/// leaves that fall inside the tile. Used by Phase B-redux + Phase C
/// region partitioning.
///
/// `normal_sum` is the unnormalized sum of every painted leaf's
/// LeafAttr.normal that overlapped this tile (object-local). The
/// user-shader anchor build normalizes + world-transforms this into
/// `AnchorRecord.surface_normal` so blade orientations conform to the
/// host surface. Sum (not running mean) lets the tile-spanning leaf
/// path contribute the same normal once per overlapped tile without
/// extra state.
#[derive(Debug, Clone)]
pub(crate) struct PaintedTileEntry {
    pub aabb: arvx_core::Aabb,
    pub leaf_count: u32,
    pub normal_sum: glam::Vec3,
}

impl PaintedTileEntry {
    pub fn empty() -> Self {
        Self {
            aabb: arvx_core::Aabb {
                min: glam::Vec3::splat(f32::INFINITY),
                max: glam::Vec3::splat(f32::NEG_INFINITY),
            },
            leaf_count: 0,
            normal_sum: glam::Vec3::ZERO,
        }
    }
}

/// Per-entity painted-material walk cache. Populated by
/// `scan_painted_aabbs` in `lifecycle::tick`'s incremental walk; the
/// flat `painted_materials` view on `EngineState` is the concatenation
/// of every entry's contents.
///
/// Keeping the per-entity result around lets the walk skip entities
/// that haven't been touched since their last cache build — drag-paint
/// stamps mark only the painted entity dirty, so the walk's lock scope
/// shrinks from O(all entities) to O(dirty entities).
#[derive(Debug, Clone, Default)]
pub(crate) struct EntityPaintedCache {
    pub mat_tiles: std::collections::HashMap<
        u16,
        std::collections::HashMap<[i32; 3], PaintedTileEntry>,
    >,
}

use crate::camera::CameraControlState;

use super::CameraState;

/// A click-pick awaiting a G-buffer readback. The viewport tag
/// determines interpretation: MAIN resolves to a scene entity; BUILD
/// resolves to a procedural NodeId (when the build viewport is in
/// raymarch preview mode — the shader packs the hit primitive's
/// NodeId into the material G-buffer).
///
/// `ghost_pick_node_id` holds the result of a synchronous CPU raycast
/// performed at click time against the tree's ghost-role primitives
/// (cutters, Intersect operands). When `Some`, it overrides the
/// G-buffer decode — the visual rule is "if a ghost silhouette is
/// drawn at the click pixel, clicking picks it" even when a solid
/// surface is closer along the ray, because the ghost pass renders
/// depth-free on top of everything.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingPick {
    pub(crate) viewport: crate::viewport::ViewportId,
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) ghost_pick_node_id: Option<u32>,
}

/// A queued drag-drop action awaiting the position-readback result from
/// the pick pipeline. When the matching `PickResult` returns with a
/// world-space position, `process_pick_result` consumes this instead of
/// running the usual selection update — the drop spawns the asset /
/// generator / preset at the hit point (or ground-plane fallback).
///
/// We reuse the pick pipeline (instead of adding a separate
/// position-readback route) because it already handles the
/// async-one-frame-later readback timing and we only need a single
/// coordinate query per drop.
#[derive(Debug, Clone)]
pub(crate) enum PendingDropAction {
    Asset { path: String },
    Generator { name: String },
    GeneratorPreset { path: String },
}

#[derive(Debug, Clone)]
pub(crate) struct PendingDrop {
    pub(crate) viewport: crate::viewport::ViewportId,
    /// The screen pixel that was dropped on. Used to cast a ground-plane
    /// fallback ray when the pick result's `position` is `None` (sky hit).
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) action: PendingDropAction,
}

/// Brush settings captured when a paint stroke sample fires its pick
/// readback. One instance lives in [`EngineState::paint_pick_settings`]
/// at a time — the editor throttles stamps to one pick in flight, and
/// a fresh `PaintAtPixel` replaces the previous settings if the pick
/// hasn't returned yet (latest sample wins, matching drag-preview).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintPickSettings {
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) strength: f32,
    pub(crate) falloff: f32,
    pub(crate) mode: crate::command::PaintMode,
    pub(crate) material_id: u16,
}

/// Sculpt-brush analogue of [`PaintPickSettings`]. Captured when a
/// `SculptAtPixel` command queues its pick; consumed by
/// `process_pick_result` once the matching readback returns.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SculptPickSettings {
    pub(crate) radius: f32,
    pub(crate) falloff_curve: arvx_core::sculpt::FalloffCurve,
    pub(crate) strength: f32,
    pub(crate) stroke_seq: u64,
    pub(crate) mode: crate::command::SculptMode,
    pub(crate) material_id: u16,
}

/// A live drag-and-drop preview. While active, each `DragPreviewOver`
/// (re)sets the pending pick at the cursor pixel. What happens when
/// the pick result arrives depends on `kind`:
///
/// * **Model** — a real asset entity is already spawned; its transform
///   gets updated to the new surface snap (AABB-bottom-snapped).
/// * **Generator** — nothing is spawned yet. A wireframe AABB gizmo is
///   drawn at the cached position every frame; on commit we spawn the
///   real generator there. Baking a live generator while the user
///   drags would produce a trail of stale children at each emit
///   position.
#[derive(Debug, Clone)]
pub(crate) struct DragPreviewState {
    pub(crate) viewport: crate::viewport::ViewportId,
    pub(crate) kind: DragPreviewKind,
    /// Most recent valid surface hit. Reused when the next pick returns
    /// a sky miss or a self-hit to avoid flickering to the ground plane.
    pub(crate) last_surface_pos: Option<glam::Vec3>,
    /// Last pixel the editor asked us to preview at. Used to cast the
    /// ground-plane fallback ray when we never got a valid surface hit.
    pub(crate) last_cursor: (u32, u32),
}

#[derive(Debug, Clone)]
pub(crate) enum DragPreviewKind {
    /// An .arvx asset — spawned as a real entity on DragEnter that
    /// tracks the cursor. Surface hits are bottom-snapped by
    /// subtracting `aabb_min_y` from the hit Y.
    Model {
        entity: hecs::Entity,
        aabb_min_y: f32,
    },
    /// A generator or preset — NO entity is spawned during drag. Just
    /// a gizmo wireframe (half-size `gizmo_half`) centered at
    /// `last_surface_pos`. On commit we spawn the real source.
    Generator {
        source: crate::command::DragPreviewSource,
        gizmo_half: glam::Vec3,
    },
}

pub(crate) struct EngineState {
    /// Render thread handle. Owns the wgpu device/queue/renderer/
    /// viewport renderers; sim communicates only via `render_worker.inbox`
    /// (per-frame `RenderFrame` snapshots), `render_worker.outbox`
    /// (per-frame `RenderResult` returns: pick + atten + GPU timings),
    /// and `render_worker.commands` (aperiodic events: resize, etc.).
    /// Dropping this triggers the render thread to shut down.
    pub(crate) render_worker: crate::render_worker::RenderWorker,

    // Scene management (CPU). Wrapped in `Arc<Mutex<>>` so the bake
    // worker can run the integrate pass (dealloc-prev + memcpy +
    // remap) against the shared pools directly, without shipping
    // artifacts back to the main thread for a 75+ ms copy. The lock
    // is uncontended on most frames — only the render thread's
    // per-frame geometry upload and the sim thread's asset loads
    // touch it, and both finish in a ms or two. See
    // `bake_worker::run_loop` for the worker-side lock scope.
    pub(crate) scene_mgr: std::sync::Arc<std::sync::Mutex<ArvxSceneManager>>,

    /// Lock-free handle on `scene_mgr.geometry_epoch()` so per-tick
    /// reads (in `submit_render_frame`) don't take the scene_mgr
    /// Mutex. Without this, every sim tick blocks for the duration
    /// of any in-progress bake_worker integrate (~50 ms+), dropping
    /// sim from 60 Hz to ~20 Hz and making animation/camera feel
    /// like 0.5 fps.
    pub(crate) geometry_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

    /// Paint-data epoch handle. Bumped by `apply_paint_sphere`
    /// whenever a stroke writes to leaf_attr / color. Separate from
    /// `geometry_epoch` so paint doesn't re-upload octree + brick
    /// buffers — render only slice-uploads the dirty slot range.
    pub(crate) paint_epoch_handle: std::sync::Arc<std::sync::atomic::AtomicU64>,

    /// Sim-side cache of per-asset skinning data. Today only the
    /// bone count (`rest_bone_aabbs.len()`) is read at runtime; the
    /// cache is refreshed lazily under the scene_mgr lock only when
    /// `skinning_data_cache_epoch` falls behind the current
    /// `geometry_epoch_handle` (bake / asset load).
    pub(crate) skinning_data_cache: std::collections::HashMap<
        arvx_render::AssetHandle,
        arvx_render::SkinningAssetData,
    >,
    /// Last `geometry_epoch` we used to build [`Self::skinning_data_cache`].
    /// When `geometry_epoch_handle > this`, the cache is stale and
    /// gets rebuilt next tick.
    pub(crate) skinning_data_cache_epoch: u64,

    /// Async bake pipeline. The worker owns its own GpuEvaluator and
    /// private pools; the engine sends requests + drains results on
    /// each tick via `drain_bake_results`. The same worker also
    /// handles generator jobs (see `generator_system`).
    pub(crate) bake_worker: crate::bake_worker::BakeWorker,

    /// Generator tick driver. Scans entities with `GeneratorState`,
    /// hashes params to detect edits, submits stale runs to the
    /// bake worker's generator channel, and updates the ECS as
    /// results arrive.
    pub(crate) generator_system: crate::generator::GeneratorSystem,

    /// User-shader registry — populated by scanning
    /// `<project_root>/assets/shaders/` on project load and on
    /// filesystem change. The engine resolves `UserShader.shader_name`
    /// via this registry when flattening trees, and the bake worker's
    /// `GpuEvaluator` is recompiled whenever this registry changes
    /// (engine sends `WorkerControl::ReloadUserShaders`).
    pub(crate) user_shader_registry: arvx_render::shader_composer::UserShaderRegistry,

    /// Per-entity painted-material info, cached on
    /// `(paint_epoch, geometry_epoch)`. Layout:
    ///   `object_id → material_id → tile_coord → (tile-local AABB,
    ///    count of painted leaves in that tile)`.
    /// V10 multi-region tiling: shaders with `@tile_size` emit one
    /// region per non-empty tile, so we bucket painted leaves by
    /// tile coord during the scan. For shaders without `@tile_size`,
    /// the inner map has a single entry under the sentinel coord
    /// `NO_TILE` (V9 single-region fallback). The AABB stored is the
    /// painted-leaf bounds within the tile (or the full bounds for
    /// `NO_TILE`); the count drives per-region pool sizing.
    pub(crate) painted_materials: std::collections::HashMap<
        u32,
        std::collections::HashMap<
            u16,
            std::collections::HashMap<[i32; 3], PaintedTileEntry>,
        >,
    >,
    /// Per-material anchor records for the V1 mesh-path user shader.
    /// Each material with painted leaves gets its own anchor list —
    /// the new pipeline runs once per material (one set of compute +
    /// raster passes, sharing the per-shader pipeline objects).
    /// Rebuilt whenever `painted_materials` is rebuilt.
    pub(crate) painted_anchors: std::sync::Arc<
        std::collections::HashMap<
            u16,
            Vec<arvx_render::user_shader_mesh_pass::AnchorRecord>,
        >,
    >,
    /// Debug-only: last rebuild's (object_id, quantized tile_min, material_id)
    /// → seed map. Used by `ARVX_GRASS_DEBUG` to detect per-tile seed
    /// drift across rebuilds (i.e., the "same tile" reporting different
    /// seeds, which would indicate a stability bug). None when the env
    /// flag is off.
    pub(crate) debug_last_anchor_seeds: Option<
        std::collections::HashMap<(u32, u32, u32, u32, u16), u32>,
    >,
    /// Per-entity walk results. The flat `painted_materials` /
    /// `painted_anchors` above are derived views over this map's values.
    /// Mutated by the lifecycle walk only — `apply_paint_stamp` drives
    /// updates by adding the painted entity to
    /// [`Self::painted_dirty_entities`].
    pub(crate) painted_per_entity:
        std::collections::HashMap<hecs::Entity, EntityPaintedCache>,
    /// Entities whose painted-material cache needs a re-scan on the
    /// next lifecycle tick. Populated by `apply_paint_stamp` (one
    /// entry per stamp) and by the geometry-epoch path (every
    /// renderable entity, when scene geometry mutates). The walk
    /// drains this set, locking `scene_mgr` briefly per entity rather
    /// than holding it across all entities for the duration of the
    /// O(all-octrees) walk.
    pub(crate) painted_dirty_entities: std::collections::HashSet<hecs::Entity>,
    /// Per-stamp brush AABBs (world space) accumulated since the last
    /// walk, keyed by entity. Populated alongside `painted_dirty_entities`
    /// by `apply_paint_stamp` (Material mode) and `apply_sculpt_stamp`.
    ///
    /// The walk uses this to scope its octree scan to the brush
    /// footprint plus the largest shader-material `tile_size`, instead
    /// of walking the entire entity octree. An entity in the dirty set
    /// *without* a region entry (e.g. the geometry-epoch fallback when
    /// nobody told us which entities changed) falls back to the full
    /// walk path.
    ///
    /// Drained per tick by the walk; cleared on entity despawn and
    /// scene clear.
    ///
    /// Phase C1 of `docs/PERF_DEBT.md` — replaces the
    /// `entities_known_empty` stopgap from the 2026-05-13 sculpt
    /// session. With region-bounded scans, "zero shader materials"
    /// entities pay the cost of walking the brush footprint only
    /// (~ms) instead of the full ~2.5M-voxel octree (~150 ms), so the
    /// stopgap is no longer needed.
    pub(crate) painted_dirty_regions:
        std::collections::HashMap<hecs::Entity, Vec<arvx_core::Aabb>>,
    /// Typed mutation event log — every CPU-side mutation describes
    /// its scope here. Phase A1 of `docs/PERF_DEBT.md`: scaffolding
    /// only, no consumers. Drained at the start of each
    /// `submit_render_frame` tick. Phase B/C replace the existing
    /// coarse `*_dirty` flags by reading events from this log and
    /// translating them into per-entity dirty sets.
    pub(crate) mutation_log: super::mutation_log::MutationLog,
    /// Epochs the cache was last reconciled against. When either
    /// moves ahead, we invalidate and re-scan affected entities.
    pub(crate) painted_materials_paint_epoch: u64,
    pub(crate) painted_materials_geometry_epoch: u64,
    /// Geometry epoch of the most recent batch we submitted to the
    /// paint-walk worker. Used to suppress redundant
    /// blanket-invalidations while the worker is still computing the
    /// post-bump cache — without this we'd re-mark every renderable
    /// dirty on each tick between submit and merge, then queue a
    /// duplicate batch the moment the worker freed up. Phase E1.
    pub(crate) painted_walk_submitted_geom_epoch: u64,

    /// Dedicated worker thread that runs the painted-material walk
    /// off the sim tick. Sim drains [`Self::painted_dirty_entities`]
    /// into a batch, submits to the worker, and merges the result on a
    /// later tick into [`Self::painted_per_entity`]. Phase E1 of
    /// `docs/PERF_DEBT.md`.
    pub(crate) paint_walk_worker: super::paint_walk::PaintWalkWorker,

    /// Dedicated worker thread that runs collider-cache rebuilds off
    /// the sim tick. Sim drains [`Self::collider_caches_dirty`] into a
    /// batch, submits to the worker, and merges the resulting
    /// `ColliderCache` entries into the ECS on a later tick. Phase E2
    /// of `docs/PERF_DEBT.md`.
    pub(crate) collider_worker: super::collider_worker::ColliderWorker,

    /// Per-entity sparse paint overlays. Each entry holds the leaves
    /// painted on that *specific* instance — decoupled from the
    /// asset's shared `LeafAttrPool`. Shipping these decouples paint
    /// from asset sharing: load bunny.arvx twice, paint one, only that
    /// one sees the new color.
    ///
    /// Lifetime: created on first stamp into an entity, dropped when
    /// the entity is despawned (`delete_entity` / `clear_scene`).
    /// Concatenated each frame into the GPU-side `instance_overlay`
    /// buffer — a per-instance `(offset, count)` pair on
    /// `ArvxGpuInstance` slices into it.
    pub(crate) paint_overlays:
        std::collections::HashMap<hecs::Entity, arvx_core::LeafAttrOverlay>,

    /// Per-entity sparse sculpt overlay — sorted set of removed
    /// `leaf_attr_id`s. Built incrementally as the user carves with
    /// the sculpt brush (`apply_sculpt_brush` writes here); the entry
    /// is dropped when the entity is despawned (`delete_entity` /
    /// `clear_scene`) or when a save flushes the overlay into the
    /// octree. Concatenated each frame into the GPU-side
    /// `instance_sculpt` buffer — a per-instance `(offset, count)` pair
    /// on `ArvxGpuInstance` slices into it. Phase A: Carve only.
    pub(crate) sculpt_overlays:
        std::collections::HashMap<hecs::Entity, arvx_core::SculptOverlay>,

    /// Per-material flag — `true` if the material's `opacity < 0.99`
    /// (i.e., classified as glass by the march and the mesh-mode
    /// glass passes). Indexed by material id. Rebuilt from the
    /// material library at the start of `update_scene_gpu` whenever
    /// `material_glass_lib_epoch` doesn't match the library's
    /// current state. Used to compute `MeshDraw.has_glass`.
    pub(crate) material_is_glass: Vec<bool>,
    /// Snapshot of `MaterialLibrary::slot_count() + an opacity-checksum`
    /// at the last `material_is_glass` rebuild. Tracked so we don't
    /// rebuild every frame; cheap to compute since material counts
    /// stay in the dozens.
    pub(crate) material_glass_lib_epoch: u64,
    /// Per-asset cached "has any glass cell" flag. Key is the asset's
    /// `spatial.root_offset` (same key the per-frame asset_table
    /// uses). Computed lazily — first time a draw touches an asset,
    /// we walk its leaves once with `material_is_glass` and store
    /// the result. Cleared whenever `material_is_glass` rebuilds.
    pub(crate) asset_has_glass_cache: std::collections::HashMap<u32, bool>,

    // Input + Camera
    pub(crate) input_system: arvx_runtime::input::InputSystem,
    pub(crate) camera_control: CameraControlState,
    pub(crate) camera: CameraState,
    /// Viewport scaffolding (Phase 1). Populated with one entry for
    /// `ViewportId::MAIN` and kept in step with `camera` field on writes,
    /// but rendering still consults `camera` directly. Future phases route
    /// reads through here as the renderer is split per-viewport.
    pub(crate) viewports: crate::viewport::Viewports,

    // ECS — the source of truth for scene state.
    pub(crate) world: hecs::World,
    pub(crate) registry: crate::component_registry::ComponentRegistry,
    /// Stable UUID ↔ hecs Entity mapping.
    pub(crate) entity_uuids: std::collections::HashMap<hecs::Entity, uuid::Uuid>,
    pub(crate) uuid_to_entity: std::collections::HashMap<uuid::Uuid, hecs::Entity>,
    /// UUID counter for generating stable IDs.
    pub(crate) next_entity_uuid: u64,
    /// Per-entity scene-tree order key. `f64` so the drag-reorder path
    /// can insert between two neighbors (mid = (a + b) * 0.5) without
    /// renumbering anything. Persisted in the scene file — user-
    /// arranged ordering survives save / reload.
    ///
    /// Deliberately a side-map rather than a hecs component: the
    /// properties panel reflects over components via `ComponentRegistry`,
    /// and we don't want this editor-ordering concern to show up as a
    /// field there.
    pub(crate) entity_tree_order: std::collections::HashMap<hecs::Entity, f64>,
    /// Next TreeOrder value to hand out on a fresh spawn. Monotonic;
    /// reseeds past `max(loaded)` after a scene load so new spawns
    /// still append at the bottom.
    pub(crate) next_tree_order: f64,
    /// Currently selected entity.
    pub(crate) selected_entity: Option<hecs::Entity>,
    /// Currently selected procedural node (within the selected entity's ProceduralGeometry).
    pub(crate) selected_procedural_node: Option<u32>,

    // Derived GPU data — rebuilt from world each frame.
    //
    // PERF_DEBT A3: each of these is `Arc<Vec<…>>` so the per-tick
    // snapshot handoff is a refcount bump rather than a per-Vec memcpy
    // (~30 KB each, ~230 KB combined). Mutations route through
    // `Arc::make_mut`: in steady state (after render's last frame's
    // snapshot has been consumed) refcount=1 and writes are in-place,
    // preserving the previous capacity allocation. While render still
    // holds last frame's `Arc`, the next mutation reallocates fresh —
    // which is the same cost as today's `.clone()` would have been.
    // Net win: the *clean* tick path (no GPU object changes) ships
    // `Arc::clone` instead of cloning the Vec.
    /// Per-asset records, deduped by `octree_root`. Built alongside
    /// `gpu_instances` in `update_scene_gpu`.
    pub(crate) gpu_assets: std::sync::Arc<Vec<ArvxGpuAsset>>,
    /// Per-instance records — one per renderable entity. Indexes into
    /// `gpu_assets` via `ArvxGpuInstance::asset_id`.
    pub(crate) gpu_instances: std::sync::Arc<Vec<ArvxGpuInstance>>,
    /// Per-frame flattened overlay entries — `ArvxGpuInstance.overlay_offset`
    /// + `overlay_count` slice into this. Built alongside `gpu_instances`
    /// in `update_scene_gpu` from the per-entity `paint_overlays` map;
    /// shipped each tick to the render thread for upload.
    pub(crate) gpu_instance_overlays: std::sync::Arc<Vec<arvx_core::OverlayEntry>>,
    /// Per-frame flattened sculpt-removal entries —
    /// `ArvxGpuInstance.sculpt_offset` + `sculpt_count` slice into this.
    /// Built alongside `gpu_instances` in `update_scene_gpu` from the
    /// per-entity `sculpt_overlays` map; shipped each tick to the
    /// render thread for upload to `instance_sculpt`. Each `u32` is a
    /// removed `leaf_attr_id`.
    pub(crate) gpu_instance_sculpts: std::sync::Arc<Vec<u32>>,
    /// PERF_DEBT.md D2: pending-mutation flag for
    /// `gpu_instance_overlays`. Set by paint stamps / overlay-
    /// bearing entity removes / scene reset. Drained by
    /// `submit_render_frame` into the `RenderFrame` snapshot as a
    /// `DirtyRanges` with the buffer's current length (so the
    /// render side's `write_with_dirty` sees `is_full_pool` and does
    /// a single full upload), then cleared. False on ticks where no
    /// overlay-mutating event fired — the snapshot carries empty
    /// `DirtyRanges` and the render skips the upload entirely.
    pub(crate) gpu_instance_overlays_dirty: bool,
    /// PERF_DEBT.md D3: same shape as
    /// [`Self::gpu_instance_overlays_dirty`] but for
    /// `gpu_instance_sculpts`.
    pub(crate) gpu_instance_sculpts_dirty: bool,
    /// Mesh-raster per-instance draws. One entry per `Renderable`
    /// entity whose asset_handle is `Some(_)`. Built alongside
    /// `gpu_instances` in `update_scene_gpu`.
    pub(crate) mesh_draws: std::sync::Arc<Vec<arvx_render::mesh_instance::MeshDraw>>,
    /// Procedural proxy-mesh draws. Built per-frame from entities
    /// whose `Renderable.spatial` is `RenderGeometry::ProxyMesh`.
    /// Disjoint from `mesh_draws`: proxy meshes ride a dedicated
    /// raster pipeline (`mesh_proxy_pass`) that writes the full
    /// G-buffer directly — no LeafAttr indirection.
    pub(crate) proxy_draws: std::sync::Arc<Vec<arvx_render::mesh_proxy_pass::ProxyDraw>>,
    /// Maps gpu_instance index → hecs Entity (for pick resolution).
    pub(crate) gpu_to_entity: Vec<hecs::Entity>,
    /// Maps hecs Entity → gpu_instance index.
    pub(crate) entity_to_gpu: std::collections::HashMap<hecs::Entity, usize>,

    // Project state
    pub(crate) project_loaded: bool,
    pub(crate) project_name: String,
    pub(crate) project_dir: Option<std::path::PathBuf>,
    pub(crate) project_path: Option<std::path::PathBuf>,
    pub(crate) scene_path: Option<std::path::PathBuf>,
    pub(crate) project_dirty: bool,
    /// Available .arvx model files in the project.
    pub(crate) available_models: Vec<crate::snapshot::ModelInfo>,
    pub(crate) models_dirty: bool,
    /// Set whenever the gameplay dylib load/unload changes the set of
    /// registered generators. Drives the next snapshot's
    /// `available_generators` field.
    pub(crate) generators_dirty: bool,
    /// Discovered `.arvxgen` preset files in the project's
    /// `assets/generators/` directory. Repopulated on project open.
    pub(crate) available_generator_presets: Vec<crate::generator::GeneratorPresetInfo>,
    /// Set when the preset list changed; drives the next snapshot.
    pub(crate) generator_presets_dirty: bool,
    /// Per-generator-entity set of slot keys that the *current* run
    /// has emitted so far. Reset on `WillResubmit`; consulted on
    /// `Completed` to delete persistent children whose key wasn't
    /// re-emitted (orphans from the previous generation).
    pub(crate) pending_generator_slot_keys:
        std::collections::HashMap<hecs::Entity, std::collections::HashSet<String>>,
    /// Source paths currently being re-imported. The UI consults this set
    /// to show a progress indicator in place of the Re-import button.
    /// Populated on `ReimportModel` submission, drained on completion.
    pub(crate) importing_sources: std::collections::HashSet<String>,
    /// Publish `importing_sources` to the UI on the next snapshot.
    pub(crate) importing_dirty: bool,
    /// Live per-import progress state keyed by source path string —
    /// reduced from the `ImportEvent` stream each tick, published to
    /// the UI through `StateUpdate.import_progress`. Entries are
    /// removed when the matching completion lands.
    pub(crate) importing_progress: std::collections::HashMap<String, crate::snapshot::ImportProgressInfo>,
    /// Latest editor layout JSON pushed up from the editor. Opaque to
    /// the engine — it just round-trips this through `.arvxproject`.
    pub(crate) editor_layout_json: Option<String>,
    /// Ship `editor_layout_json` to the editor on the next snapshot.
    /// Set on project load so the editor can hydrate its signals; never
    /// set for echoes from the editor itself (no feedback loop).
    pub(crate) editor_layout_pending: bool,

    /// Shared cache of loaded `.arvxskel` skeleton assets. Multiple
    /// entities loaded from the same `.arvx` share a single `Arc`.
    pub(crate) animation_cache: crate::animation::AnimationAssetCache,
    /// Per-frame allocator that packs every skinned entity's
    /// `Skeleton.current_pose` into one contiguous byte buffer for GPU
    /// upload. Rebuilt whenever `update_scene_gpu` runs.
    pub(crate) bone_matrix_allocator: crate::scene_sync::BoneMatrixAllocator,
    /// Master toggle — when false, every entity gets
    /// `skinning_mode = SKINNING_MODE_NONE` so the mesh VS emits the
    /// rest pose. Driven by the AnimationPanel checkbox.
    pub(crate) skinning_enabled: bool,
    /// `true` → Dual-Quaternion Skinning in the VS; `false` →
    /// Linear Blend Skinning. DQS preserves joint volume and fixes
    /// axial-twist candy-wrapper at a slight per-vertex cost.
    /// Defaults off; flip on for extreme poses or A/B compare.
    pub(crate) dqs_enabled: bool,

    /// Latest cloud-sun attenuation read from MAIN's volumetric pass,
    /// fed back over the render→sim result channel each frame. Sim
    /// uses it as the *target* of an EMA into [`Self::cloud_sun_atten`]
    /// (which is what actually scales the sun light on the next
    /// frame). NaN sentinel = render hasn't published a value yet
    /// (e.g. during the first frame or while MAIN is hidden); sim
    /// holds the previous target in that case.
    pub(crate) last_cloud_sun_atten_raw: f32,

    /// Sim-side stash for the most recently submitted pick's
    /// CPU-resolved ghost hint. Rendering is GPU-only; the ghost
    /// priority logic stays sim-side because it depends on the
    /// procedural tree (sim-owned). When the matching `PickResult`
    /// arrives back from render, sim consults this to decide whether
    /// the ghost win overrides the GPU-decoded NodeId.
    pub(crate) in_flight_pick_ghost: Option<u32>,

    /// Material library — manages .arvxmat files and runtime palette.
    pub(crate) material_lib: crate::material_library::MaterialLibrary,
    /// Currently selected material in the materials panel.
    pub(crate) selected_material: Option<u16>,
    /// Currently selected model path (source mesh) for Asset Properties.
    pub(crate) selected_model: Option<String>,

    /// Environment settings (sky, lighting, shadows, tone mapping).
    pub(crate) environment: crate::environment::EnvironmentSettings,
    /// Whether environment settings changed and need GPU update.
    pub(crate) environment_dirty: bool,
    /// Whether the editor UI needs the latest environment (cleared by build_state_update).
    pub(crate) environment_ui_dirty: bool,

    /// Console log buffer.
    pub(crate) console: crate::console::ConsoleLog,
    /// Gameplay dylib loader (hot-reload).
    pub(crate) gameplay_loader: crate::gameplay_loader::GameplayLoader,
    /// Behavior system executor (created when play starts).
    pub(crate) behavior_executor: Option<crate::behavior::BehaviorExecutor>,
    /// Command queue for deferred ECS mutations from gameplay systems.
    pub(crate) behavior_commands: crate::behavior::CommandQueue,
    /// Key-value game state store + event bus.
    pub(crate) game_store: crate::behavior::GameStore,
    /// System entries from the gameplay dylib.
    pub(crate) gameplay_systems: Vec<&'static crate::behavior::SystemEntry>,
    /// Monotonic total play time.
    pub(crate) play_total_time: f64,
    /// Monotonic play frame counter.
    pub(crate) play_frame_count: u64,
    /// Play mode state (None = edit mode).
    pub(crate) play_state: Option<crate::play_mode::PlayModeState>,
    /// View options.
    pub(crate) show_colliders: bool,
    /// Per-entity tracking of which colliders need rebuild
    /// (PERF_DEBT.md B2+C3). Replaces the prior `bool`. Sourced from
    /// `geometry_dirty` (geometry mutation → collider invalidation)
    /// plus direct setters when a `RigidBody` component is added or
    /// modified on a single entity. Consumer in lifecycle iterates
    /// `dirty_entities()` and calls `rebuild_collider_cache_for(e)`
    /// per entity, falling back to the world-walking
    /// `rebuild_collider_caches` only when `is_all()` is true.
    pub(crate) collider_caches_dirty: super::geometry_dirty::GeometryDirty,
    /// EMA of true tick rate (1 / wall-clock tick interval, including the
    /// 60-Hz pacing sleep). Distinct from `fps` in the state update, which
    /// is `1 / frame_work_time` and ignores sleep — useful for profiling but
    /// not what the user perceives.
    pub(crate) tick_hz_ema: f32,
    /// EMA of physics substeps per second across the engine tick. When
    /// physics is stepping at the target 60 Hz this sits near 60.
    pub(crate) physics_hz_ema: f32,
    /// EMA of the render thread's actual iteration rate, in Hz. Fed
    /// from `RenderResult::render_dt_ms` each time sim drains the
    /// render outbox. This is the "FPS" the editor displays — it
    /// reflects the on-screen production cadence, not sim CPU
    /// headroom (which `1 / cpu_total_ms` would be).
    pub(crate) render_hz_ema: f32,
    /// Rate at which fresh pixel frames actually reach the editor
    /// surface, EMA-smoothed. Updated from
    /// `RenderResult::delivered_dt_ms` whenever the render thread
    /// reports a successful pixel ship. Diverges from
    /// `render_hz_ema` whenever render iterates faster than it ships
    /// (interp re-renders, `MIN_FRAME_CALLBACK_INTERVAL`, sim
    /// upstream bottleneck). This is the honest "what did the user
    /// see" number.
    pub(crate) delivered_hz_ema: f32,
    /// Last inspector snapshot we sent to the editor. Used to skip pushing
    /// an identical snapshot every tick — without this, the panel re-renders
    /// 60Hz when physics writes Transform on a selected RigidBody, which
    /// chunks the UI thread.
    pub(crate) prev_inspector: Option<crate::inspector::InspectorSnapshot>,
    /// Cached per-entity `MaterialUsage` list. Computing it walks every
    /// leaf slot in the entity's subtree (including every brick cell
    /// and every prefilter-LOD slot), which is O(voxels) — trivial
    /// before the brick-descent fix, but a tick-killing 50 ms+ per
    /// call on high-voxel entities once bricks were included. Since
    /// the list only changes when the selection or geometry changes,
    /// cache it keyed on `(entity, geometry_epoch)` and reuse across
    /// ticks.
    pub(crate) cached_material_usage:
        Option<(hecs::Entity, u64, Vec<crate::inspector::MaterialUsage>)>,
    /// Same change-detection cache for the procedural snapshot.
    pub(crate) prev_procedural: Option<crate::procedural_snapshot::ProceduralSnapshot>,
    /// Last environment we shipped to the editor — diff-suppression
    /// avoids env-panel churn from any path that pushes env on a
    /// no-op (and means we no longer need the env_ui_dirty gate's
    /// "don't echo back during slider drag" workaround).
    pub(crate) prev_environment: Option<crate::environment::EnvironmentSettings>,
    /// Tracks the source hash of the user-shader registry the editor
    /// most recently saw, so `build_state_update` only sends a fresh
    /// `user_shaders` list when the registry has actually changed.
    /// `0` = "editor hasn't been told yet" (matches the empty-registry
    /// hash, but the first tick always sends regardless to populate it).
    pub(crate) prev_user_shader_hash: u64,
    pub(crate) user_shader_first_send: bool,

    /// File watcher for hot-reload (watches project assets/ directory).
    pub(crate) file_watcher: Option<crate::file_watcher::ArvxFileWatcher>,
    /// Background import worker for mesh → .arvx conversion.
    pub(crate) import_worker: crate::import_worker::ImportWorker,

    /// Per-entity tracking of which entities' geometry has changed
    /// (PERF_DEBT.md B2). Replaces the prior `bool`. Drains into
    /// `collider_caches_dirty` at the top of each tick — every
    /// geometry change implies a collider rebuild for the affected
    /// entity.
    pub(crate) geometry_dirty: super::geometry_dirty::GeometryDirty,
    /// Scene structure changed — push objects list to UI. Per-entity
    /// dirty set + sticky-all bit; see [`super::scene_dirty::SceneDirty`]
    /// and `docs/PERF_DEBT.md` B3. Today's consumer in
    /// `build_state_update` still does a full sorted rebuild on
    /// `is_dirty()`; the per-entity scope is foundation for a future
    /// incremental UI delta path.
    pub(crate) scene_dirty: super::scene_dirty::SceneDirty,
    /// Per-entity dirty tracking for `update_scene_gpu`. Replaced
    /// the prior `bool` flag (PERF_DEBT.md B1) so each mutation can
    /// describe its scope: stamp/gizmo/paint events flag a single
    /// entity; world-level events (project load, scene clear) flag
    /// "all". Today's consumer in `lifecycle.rs` still does a full
    /// rebuild whenever `is_dirty()` is true; C2 will wire the
    /// per-entity fast path that actually iterates `dirty_entities`.
    pub(crate) gpu_objects_dirty: super::gpu_objects_dirty::GpuObjectsDirty,

    // Frame counter
    pub(crate) frame_index: u64,

    /// Ring buffer of per-frame CPU + GPU timings. Fed from the frame
    /// work at the end of `tick`, read by the editor (via `StateUpdate`)
    /// and by MCP once wired.
    pub(crate) profiling: crate::profiling::ProfilingHistory,

    /// Behavior `FixedUpdate` accumulator. Mirrors physics' Rapier-side
    /// accumulator so behavior code that registers in the FixedUpdate
    /// phase ticks at exactly 60 Hz regardless of render rate. We carry
    /// it here (not inside the executor) because the executor is
    /// optional — it doesn't exist before a project loads — and the
    /// accumulator must persist across executor (re)creation so we
    /// don't lose simulation time on hot-reload.
    pub(crate) behavior_fixed_accumulator: f32,

    // Temporally smoothed cloud-sun attenuation (camera→sun ray through the
    // cloud layer). Lerps toward the target each frame so a single noisy ray
    // through FBM doesn't flicker sun intensity.
    pub(crate) cloud_sun_atten: f32,

    // Render dimensions
    pub(crate) width: u32,
    pub(crate) height: u32,

    // (Per-viewport readback / composite / wireframe live in
    // `viewport_renderers[MAIN]` — see `arvx_render::ViewportRenderer`.)

    // Gizmo state
    pub(crate) gizmo: crate::gizmo::GizmoState,
    /// Gizmo state for the BUILD viewport — targets the selected
    /// procedural node's transform rather than an entity Transform.
    /// Separate from `gizmo` so a drag on BUILD doesn't fight a hover
    /// on MAIN (or vice versa).
    pub(crate) proc_gizmo: crate::gizmo::GizmoState,
    /// BUILD viewport cursor position (in BUILD's local pixel space).
    pub(crate) build_mouse_pos: glam::Vec2,
    /// BUILD viewport left-button pressed state. Tracked directly
    /// (rather than feeding `input_system`) so BUILD input doesn't
    /// fight MAIN's WASD/fly camera input.
    pub(crate) build_mouse_left: bool,
    /// Previous tick's value of `build_mouse_left` — used for edge
    /// detection so picking fires once per click rather than every
    /// frame the button is held.
    /// Parent-world transform of the procedural node at drag start —
    /// used to project world-space gizmo deltas back into the node's
    /// local (parent-relative) transform on each frame. Identity when
    /// no drag is active.
    pub(crate) proc_gizmo_parent_world: glam::Affine3A,
    /// Node's local SRT components at drag start. Held separately from
    /// `proc_gizmo.initial_*` (which track world-space) so we can
    /// rebuild the node's Affine3A correctly without redoing the
    /// decompose per frame.
    pub(crate) proc_gizmo_initial_local: (glam::Vec3, glam::Quat, glam::Vec3),
    /// Mouse position in viewport pixels (for gizmo hover).
    pub(crate) mouse_pos: glam::Vec2,

    /// Pending pixel-pick: a (viewport, x, y) plus optional CPU-resolved
    /// ghost-priority hint. Sim populates this on click; it travels in
    /// the next [`crate::render_frame::RenderFrame`] to the render
    /// thread, which encodes the G-buffer copy. The render thread
    /// returns the raw payload via `RenderResult::pick_result`; sim
    /// resolves the final entity / NodeId in `process_pick_result`.
    pub(crate) pending_pick: Option<PendingPick>,
    /// Queued drag-drop. Populated on `DropAsset` / `DropGenerator` /
    /// `DropGeneratorPreset`; consumed when the paired pick readback
    /// returns with a world-space position.
    pub(crate) pending_drop: Option<PendingDrop>,
    /// Active drag-preview: the preview entity + cached AABB offset +
    /// last-known-good surface pos. Populated on `DragAssetEnter`, kept
    /// up-to-date by pick readbacks during `DragAssetOver`, cleared on
    /// commit or cancel.
    pub(crate) drag_preview: Option<DragPreviewState>,
    /// Paint stroke that issued the current pick. When set, the next
    /// pick result bypasses selection / drag-preview handling and is
    /// routed to `apply_paint_stamp` with these settings. Populated by
    /// `EngineCommand::PaintAtPixel` (sim); taken out and consumed by
    /// `process_pick_result` when the matching readback returns.
    pub(crate) paint_pick_settings: Option<PaintPickSettings>,
    /// Wallclock instant of the most recent successful paint stamp.
    /// Used purely as a profiling gate: `ARVX_PAINT_PROFILE` traces
    /// only fire when this is recent, so idle (and non-drag hover)
    /// stays quiet.
    pub(crate) last_paint_stamp_at: Option<std::time::Instant>,
    /// `true` while the editor is in paint mode. Drives both the
    /// brush-state probe pass (cursor) and the paint-stamp's
    /// selection-lock check. Updated by `SetPaintActive` commands.
    pub(crate) paint_mode_active: bool,
    /// Brush world-space radius while paint mode is active. Shared
    /// between the cursor visualization (`shade_params.brush_radius`)
    /// and the next paint stamp's footprint. Updated by `SetPaintActive`.
    pub(crate) paint_mode_radius: f32,
    /// Sculpt-brush stroke that issued the current pick. Same pattern
    /// as [`Self::paint_pick_settings`]; consumed by
    /// `process_pick_result` when the matching readback returns.
    pub(crate) sculpt_pick_settings: Option<SculptPickSettings>,
    /// Wallclock when the most recent SculptAtPixel command arrived
    /// (set in cmd_edit.rs's SculptAtPixel handler, consumed by
    /// `process_pick_result` to log click-to-mutation latency).
    pub(crate) sculpt_pending_at: Option<std::time::Instant>,
    /// `true` while the editor is in sculpt mode. Mutually exclusive
    /// with `paint_mode_active` (the editor's toggle handlers clear
    /// the other before sending). Updated by `SetSculptActive`.
    pub(crate) sculpt_mode_active: bool,
    /// Brush world-space radius while sculpt mode is active. Mirrors
    /// `paint_mode_radius` for the sculpt cursor visualization.
    pub(crate) sculpt_mode_radius: f32,
    /// Cached light count for march pass (set in light upload block, used in render).
    pub(crate) num_lights_cache: u32,
    /// Base ShadeParams (recomputed once per frame from environment +
    /// light list). The per-viewport loop writes this into the shared
    /// shade_params buffer with the VR's `isolation` flag overlaid,
    /// just before that VR's submit.
    pub(crate) shade_params_base: arvx_render::arvx_shade::ShadeParams,
    /// Prefiltered-LOD early-exit toggle. On by default; flipped off for
    /// A/B correctness comparison against the pre-LOD descent behavior.
    pub(crate) lod_enabled: bool,
    /// Surface-Nets render-time normal reconstruction (POC). Off by
    /// default — flip on via `set_surfacenet_enabled` for A/B.
    pub(crate) surfacenet_enabled: bool,
}

impl EngineState {
    /// Flip the prefiltered-LOD march early-exit on or off. Public API
    /// exists mainly for A/B correctness tests and debugging — no UI
    /// wires it yet, but tests and MCP may poke it.
    #[allow(dead_code)]
    pub fn set_lod_enabled(&mut self, enabled: bool) {
        self.lod_enabled = enabled;
    }

    /// Current LOD toggle state.
    #[allow(dead_code)]
    pub fn lod_enabled(&self) -> bool {
        self.lod_enabled
    }

    /// Flip the Surface-Nets normal reconstruction on or off. When on,
    /// the march computes per-voxel normals from the 3³ in-brick
    /// occupancy neighborhood instead of reading the baked octahedral
    /// `LeafAttr.normal_oct`. Dormant infrastructure for the upcoming
    /// sculpt path — runtime normal reconstruction is what sculpting
    /// will need when voxels mutate between bakes.
    #[allow(dead_code)]
    pub fn set_surfacenet_enabled(&mut self, enabled: bool) {
        self.surfacenet_enabled = enabled;
    }

    #[allow(dead_code)]
    pub fn surfacenet_enabled(&self) -> bool {
        self.surfacenet_enabled
    }

    /// Access the profiling ring buffer. Intended for MCP tools and
    /// any other read-only consumer outside the editor snapshot path.
    #[allow(dead_code)]
    pub fn profiling_history(&self) -> &crate::profiling::ProfilingHistory {
        &self.profiling
    }
}
