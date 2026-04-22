//! Central editor state store — typed struct with per-field Signals.
//!
//! Every UI-visible value lives here. Components read via `store.field.get()`.
//! The engine pushes updates via `send()`. UI interactions mutate via `set()`.
//! The `EditorStore` is `Copy` (all Signals are Copy) — no Rc, no RefCell.

use std::sync::Arc;

use rinch::prelude::*;
use uuid::Uuid;

use rkp_engine::{SceneObjectInfo, ModelInfo};
use rkp_engine::gizmo::GizmoMode;
use rkp_engine::inspector::InspectorSnapshot;
use rkp_engine::console::LogEntry;
use rkp_engine::environment::EnvironmentSettings;
use rkp_engine::material_library::MaterialInfo;
use rkp_engine::procedural_snapshot::ProceduralSnapshot;
use rkp_engine::recent_projects::RecentProject;

/// Editor-side view of profiling data used by [`ProfilingPanel`].
///
/// The panel reads each field via `{|| …}` reactive closures. The DOM
/// is built once with fixed slots; every tick only mutates text and
/// style attributes — no row tear-down, no for-loop diffing on the
/// hot path.
///
/// Values in `latest_cpu` / `latest_gpu` are exponentially smoothed in
/// the state callback so the readouts don't jitter. `history` stays
/// raw — the sparkline should show real frame-time variance.
#[derive(Debug, Clone)]
pub struct ProfilingWindow {
    pub latest_cpu: rkp_engine::profiling::CpuPhaseTimings,
    /// GPU pass timings in engine-submit order, already smoothed. The
    /// label set matches [`gpu_pass_labels`] at the moment the panel
    /// reads both signals.
    ///
    /// [`gpu_pass_labels`]: EditorStore::gpu_pass_labels
    pub latest_gpu: Vec<(String, f32)>,
    /// Raw `(frame_idx, render_dt_ms)` for the last `HISTORY_LEN`
    /// frames, oldest first. The `render_dt_ms` is the render
    /// thread's actual iteration interval — actual frame time, what
    /// the editor surface sees as a frame rate. Used for the
    /// sparkline.
    pub history: Vec<(u64, f32)>,
}

impl ProfilingWindow {
    pub const HISTORY_LEN: usize = 128;
}

use crate::ui::layout::{ContainerKind, LayoutConfig, PanelId, default_layout};

/// Editor interaction mode. Sculpt/Paint variants are reserved for
/// the upcoming sculpt tool work; the `#[allow]` silences the transient
/// dead-code warning until that lands.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    Default,
    Sculpt,
    Paint,
}

/// Central editor state — every UI-visible value is a Signal.
///
/// Created once at startup, stored in rinch context via `create_context()`.
/// All fields are `Signal<T>` which is `Copy`, so `EditorStore` is `Copy`.
#[derive(Clone, Copy)]
pub struct EditorStore {
    // ── Engine state (written by engine thread via send()) ────────

    /// Frames per second (1 / frame work time — render-only, ignores
    /// pacing sleep). Use `tick_hz` for the user-perceived rate.
    pub fps: Signal<f32>,
    /// True engine tick rate including pacing sleep.
    pub tick_hz: Signal<f32>,
    /// Smoothed physics substeps per second. Stays near 60 when physics
    /// is meeting its target; drops below when ticks starve the
    /// fixed-timestep accumulator.
    pub physics_hz: Signal<f32>,
    /// Number of GPU objects being rendered.
    pub gpu_object_count: Signal<u32>,
    /// Latest frame's CPU + GPU timings (smoothed), plus a short ring
    /// of recent frames for the sparkline. `None` until the first GPU
    /// timestamp has resolved (a few frames after startup). Updated at
    /// most 60 Hz — the callback throttles and EMA-smooths.
    pub profiling: Signal<Option<Arc<ProfilingWindow>>>,
    /// Ordered list of GPU pass labels. Updated only when the label
    /// set changes (essentially never post-startup), so the panel's
    /// `for` loop over labels doesn't churn per-tick.
    pub gpu_pass_labels: Signal<Arc<Vec<String>>>,
    /// Scene objects (flat list, hierarchy via parent_id).
    pub objects: Signal<Vec<SceneObjectInfo>>,
    /// Currently selected entity (None = nothing selected).
    pub selected_entity: Signal<Option<Uuid>>,

    // ── Layout state (written by UI) ─────────────────────────────

    /// Layout config wrapped in Arc for cheap cloning + cross-thread
    /// Signal::send (the engine thread hydrates this on project open).
    /// Read: `store.layout.get()` returns `Arc<LayoutConfig>`.
    /// Write: clone out, mutate, set back: `store.update_layout(|cfg| { ... })`.
    /// This avoids re-entrant borrow — we clone before mutating, set after.
    pub layout: Signal<Arc<LayoutConfig>>,
    /// Left container width in pixels (driven by splitter drag).
    pub left_width_px: Signal<f32>,
    /// Right container width in pixels.
    pub right_width_px: Signal<f32>,
    /// Bottom container height in pixels.
    pub bottom_height_px: Signal<f32>,

    // ── Editor mode (written by UI) ──────────────────────────────

    /// Current gizmo mode (Translate, Rotate, Scale).
    pub gizmo_mode: Signal<GizmoMode>,
    /// Current editor interaction mode. Reserved for upcoming sculpt
    /// work — see [`EditorMode`].
    #[allow(dead_code)]
    pub editor_mode: Signal<EditorMode>,

    // ── Tool settings (written by UI) ────────────────────────────
    // Reserved for the upcoming sculpt/paint tool. Already plumbed
    // into the store so the toolbar + brush-param panel can bind
    // without a later store-struct churn.

    #[allow(dead_code)]
    pub sculpt_radius: Signal<f32>,
    #[allow(dead_code)]
    pub sculpt_strength: Signal<f32>,
    #[allow(dead_code)]
    pub paint_color: Signal<[f32; 3]>,

    // ── Project state (written by engine) ───────────────────────

    /// Whether a project is loaded (controls welcome screen visibility).
    pub project_loaded: Signal<bool>,
    /// Recent projects list for the welcome screen.
    pub recent_projects: Signal<Vec<RecentProject>>,
    /// Current project name.
    pub project_name: Signal<String>,
    /// Absolute path of the current project root, used by UI display
    /// helpers to strip the prefix from absolute paths shown in the
    /// UI (so users see `assets/bunny.obj` not the full system path).
    /// Empty when no project is loaded.
    pub project_dir: Signal<String>,
    /// Available .rkp model files.
    pub available_models: Signal<Vec<ModelInfo>>,
    /// Source paths currently being re-imported on the engine thread.
    /// The Asset Properties panel uses this to swap the Re-import button
    /// for a progress indicator while a given model's import is running.
    pub importing_models: Signal<Vec<String>>,
    /// Live per-import progress reduced from the engine's `ImportEvent`
    /// stream. Empty when no imports are in flight. Surfaced alongside
    /// the Re-import spinner as a real stage/progress bar.
    pub import_progress: Signal<Vec<rkp_engine::snapshot::ImportProgressInfo>>,
    /// Model path being dragged onto viewport (None = no drag).
    pub model_drag: Signal<Option<String>>,
    /// Generator name being dragged onto viewport (None = no drag).
    pub generator_drag: Signal<Option<String>>,
    /// `.rkgen` preset path being dragged onto viewport (None = no drag).
    pub generator_preset_drag: Signal<Option<String>>,
    /// Registered generator names (from the loaded gameplay dylib).
    /// Sourced from the engine snapshot — empty when no project or no
    /// generators are registered. Rendered in the models panel.
    pub available_generators: Signal<Vec<String>>,
    /// `.rkgen` presets discovered in `assets/generators/`. Rendered
    /// in the models panel alongside bare generators.
    pub available_generator_presets:
        Signal<Vec<rkp_engine::snapshot::GeneratorPresetEntry>>,
    /// Inspector data for the selected entity.
    pub inspector: Signal<Option<InspectorSnapshot>>,
    /// Procedural object snapshot for the selected entity (if it has ProceduralGeometry).
    pub procedural: Signal<Option<ProceduralSnapshot>>,
    /// Components available to add to the selected entity.
    pub available_components: Signal<Vec<String>>,

    // ── Material state (written by engine) ──────────────────────

    /// Available materials in the project.
    pub materials: Signal<Vec<MaterialInfo>>,
    /// Currently selected material in the materials panel.
    pub selected_material: Signal<Option<u16>>,
    /// Material being dragged onto viewport (None = no drag).
    pub material_drag: Signal<Option<u16>>,
    /// Currently selected model source path (for Asset Properties).
    pub selected_model: Signal<Option<String>>,
    /// Environment settings (sky, lighting, shadows, tone mapping).
    pub environment: Signal<EnvironmentSettings>,
    /// Console log entries.
    pub console_entries: Signal<Vec<LogEntry>>,
    /// Whether the engine is in play mode.
    pub play_mode: Signal<bool>,

    // ── View settings ────────────────────────────────────────────
    /// Which primary-visibility pass the build viewport dispatches —
    /// `Voxel` (default, shows the baked octree) or `Raymarch` (live CSG
    /// preview of the procedural tree, no bake required). Updated by the
    /// build panel's preview toggle and echoed back to the engine via
    /// `EngineCommand::SetBuildPreviewMode`.
    pub build_preview_mode: Signal<rkp_render::BuildPreviewMode>,
    /// Skeletal skinning master switch. `false` → the scatter pass is
    /// skipped and the march shader falls back to rigid-mesh rendering
    /// for every skinned entity. Defaults `true`.
    pub skinning_enabled: Signal<bool>,
    /// `true` → Dual-Quaternion Skinning (preserves joint volume);
    /// `false` → Linear Blend Skinning (classic candy-wrapper pinching
    /// at twist joints, volume loss at sharp bends). Defaults `false`
    /// to match the engine's default — DQS has a ~+13% scatter cost
    /// and the visible payoff only matters on extreme poses.
    pub dqs_enabled: Signal<bool>,

    // ── Drag state (tab dragging) ────────────────────────────────

    /// Currently dragged tab (None = no drag in progress).
    pub tab_drag: Signal<Option<TabDragData>>,
    /// Where the dragged tab will drop if released now.
    pub drop_target: Signal<Option<DropTarget>>,

    /// Entity staged for the "Convert to Voxel Object" confirmation
    /// modal. `Some(id)` opens the modal — it's mounted at
    /// `LayoutRoot` (not inside the scene-tree panel) because
    /// rinch's hit-test skips descendants of `overflow: clip/auto`
    /// containers when the click falls outside the container's
    /// bounds, and a centered modal inside the narrow scene-tree
    /// column would never catch a click aimed at its own buttons.
    pub convert_procedural_target: Signal<Option<uuid::Uuid>>,
}

/// Data about the tab being dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabDragData {
    pub panel: PanelId,
    pub source_container: ContainerKind,
    pub source_zone: usize,
}

impl TabDragData {
    /// Find the tab index of this panel in its source zone.
    pub fn tab_index(&self, layout: &LayoutConfig) -> usize {
        layout.container(self.source_container)
            .zones.get(self.source_zone)
            .and_then(|z| z.tabs.iter().position(|&p| p == self.panel))
            .unwrap_or(0)
    }
}

/// Where a tab can be dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropTarget {
    /// Drop into an existing zone (adds as a new tab).
    Zone { container: ContainerKind, zone_idx: usize },
    /// Drop on a zone edge to split it and create a new zone.
    Split { container: ContainerKind, zone_idx: usize, edge: SplitEdge },
}

/// Edge of a zone for split-drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitEdge {
    #[default]
    Top,
    Bottom,
    Left,
    Right,
}

impl EditorStore {
    /// Create the store with default values.
    pub fn new() -> Self {
        Self {
            // Engine state — zeroed, engine will push real values.
            fps: Signal::new(0.0),
            tick_hz: Signal::new(0.0),
            physics_hz: Signal::new(0.0),
            gpu_object_count: Signal::new(0),
            profiling: Signal::new(None),
            gpu_pass_labels: Signal::new(Arc::new(Vec::new())),
            objects: Signal::new(Vec::new()),
            selected_entity: Signal::new(None),

            // Layout.
            layout: Signal::new(Arc::new(default_layout())),
            left_width_px: Signal::new(250.0),
            right_width_px: Signal::new(300.0),
            bottom_height_px: Signal::new(200.0),

            // Editor mode.
            gizmo_mode: Signal::new(GizmoMode::Translate),
            editor_mode: Signal::new(EditorMode::Default),

            // Tool settings.
            sculpt_radius: Signal::new(1.0),
            sculpt_strength: Signal::new(0.5),
            paint_color: Signal::new([0.8, 0.2, 0.2]),

            // Project state.
            project_loaded: Signal::new(false),
            recent_projects: Signal::new(Vec::new()),
            project_name: Signal::new(String::new()),
            project_dir: Signal::new(String::new()),
            available_models: Signal::new(Vec::new()),
            available_generators: Signal::new(Vec::new()),
            available_generator_presets: Signal::new(Vec::new()),
            importing_models: Signal::new(Vec::new()),
            import_progress: Signal::new(Vec::new()),
            model_drag: Signal::new(None),
            generator_drag: Signal::new(None),
            generator_preset_drag: Signal::new(None),
            inspector: Signal::new(None),
            procedural: Signal::new(None),
            available_components: Signal::new(Vec::new()),

            // Material state.
            materials: Signal::new(Vec::new()),
            selected_material: Signal::new(None),
            material_drag: Signal::new(None),
            selected_model: Signal::new(None),
            environment: Signal::new(EnvironmentSettings::default()),
            console_entries: Signal::new(Vec::new()),
            play_mode: Signal::new(false),
            build_preview_mode: Signal::new(rkp_render::BuildPreviewMode::Raymarch),
            skinning_enabled: Signal::new(true),
            dqs_enabled: Signal::new(false),

            // Drag state.
            tab_drag: Signal::new(None),
            drop_target: Signal::new(None),

            convert_procedural_target: Signal::new(None),
        }
    }

    /// Mutate the layout config. Clones out, mutates, sets back.
    /// This avoids re-entrant borrow on the Signal — the old Arc is
    /// dropped before reactive effects fire.
    pub fn update_layout(&self, f: impl FnOnce(&mut LayoutConfig)) {
        let mut cfg = (*self.layout.get()).clone();
        f(&mut cfg);
        self.layout.set(Arc::new(cfg));
    }
}
