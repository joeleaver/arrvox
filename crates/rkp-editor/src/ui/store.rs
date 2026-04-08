//! Central editor state store — typed struct with per-field Signals.
//!
//! Every UI-visible value lives here. Components read via `store.field.get()`.
//! The engine pushes updates via `send()`. UI interactions mutate via `set()`.
//! The `EditorStore` is `Copy` (all Signals are Copy) — no Rc, no RefCell.

use std::rc::Rc;

use rinch::prelude::*;
use uuid::Uuid;

use rkp_engine::{SceneObjectInfo, ModelInfo};
use rkp_engine::gizmo::GizmoMode;
use rkp_engine::inspector::InspectorSnapshot;

use crate::ui::layout::{ContainerKind, LayoutConfig, PanelId, default_layout};

/// Editor interaction mode.
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

    /// Frames per second.
    pub fps: Signal<f32>,
    /// Number of GPU objects being rendered.
    pub gpu_object_count: Signal<u32>,
    /// Scene objects (flat list, hierarchy via parent_id).
    pub objects: Signal<Vec<SceneObjectInfo>>,
    /// Currently selected entity (None = nothing selected).
    pub selected_entity: Signal<Option<Uuid>>,

    // ── Layout state (written by UI) ─────────────────────────────

    /// Layout config wrapped in Rc for cheap cloning.
    /// Read: `store.layout.get()` returns `Rc<LayoutConfig>`.
    /// Write: clone out, mutate, set back: `store.update_layout(|cfg| { ... })`.
    /// This avoids re-entrant borrow — we clone before mutating, set after.
    pub layout: Signal<Rc<LayoutConfig>>,
    /// Left container width in pixels (driven by splitter drag).
    pub left_width_px: Signal<f32>,
    /// Right container width in pixels.
    pub right_width_px: Signal<f32>,
    /// Bottom container height in pixels.
    pub bottom_height_px: Signal<f32>,

    // ── Editor mode (written by UI) ──────────────────────────────

    /// Current gizmo mode (Translate, Rotate, Scale).
    pub gizmo_mode: Signal<GizmoMode>,
    /// Current editor interaction mode.
    pub editor_mode: Signal<EditorMode>,

    // ── Tool settings (written by UI) ────────────────────────────

    pub sculpt_radius: Signal<f32>,
    pub sculpt_strength: Signal<f32>,
    pub paint_color: Signal<[f32; 3]>,

    // ── Project state (written by engine) ───────────────────────

    /// Whether a project is loaded (controls welcome screen visibility).
    pub project_loaded: Signal<bool>,
    /// Current project name.
    pub project_name: Signal<String>,
    /// Available .rkp model files.
    pub available_models: Signal<Vec<ModelInfo>>,
    /// Model path being dragged onto viewport (None = no drag).
    pub model_drag: Signal<Option<String>>,
    /// Inspector data for the selected entity.
    pub inspector: Signal<Option<InspectorSnapshot>>,

    // ── Drag state (tab dragging) ────────────────────────────────

    /// Currently dragged tab (None = no drag in progress).
    pub tab_drag: Signal<Option<TabDragData>>,
    /// Where the dragged tab will drop if released now.
    pub drop_target: Signal<Option<DropTarget>>,
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
            gpu_object_count: Signal::new(0),
            objects: Signal::new(Vec::new()),
            selected_entity: Signal::new(None),

            // Layout.
            layout: Signal::new(Rc::new(default_layout())),
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
            project_name: Signal::new(String::new()),
            available_models: Signal::new(Vec::new()),
            model_drag: Signal::new(None),
            inspector: Signal::new(None),

            // Drag state.
            tab_drag: Signal::new(None),
            drop_target: Signal::new(None),
        }
    }

    /// Mutate the layout config. Clones out, mutates, sets back.
    /// This avoids re-entrant borrow on the Signal — the old Rc is
    /// dropped before reactive effects fire.
    pub fn update_layout(&self, f: impl FnOnce(&mut LayoutConfig)) {
        let mut cfg = (*self.layout.get()).clone();
        f(&mut cfg);
        self.layout.set(Rc::new(cfg));
    }
}
