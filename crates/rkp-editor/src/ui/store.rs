//! Central editor state store — typed struct with per-field Signals.
//!
//! Every UI-visible value lives here. Components read via `store.field.get()`.
//! The engine pushes updates via `send()`. UI interactions mutate via `set()`.
//! The `EditorStore` is `Copy` (all Signals are Copy) — no Rc, no RefCell.

use rinch::prelude::*;
use uuid::Uuid;

use rkp_engine::SceneObjectInfo;
use rkp_engine::gizmo::GizmoMode;

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

    /// Full layout configuration (containers, zones, tabs).
    pub layout: Signal<LayoutConfig>,
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

            // Layout — default editor layout.
            layout: Signal::new(default_layout()),
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

            // Drag state.
            tab_drag: Signal::new(None),
            drop_target: Signal::new(None),
        }
    }
}
