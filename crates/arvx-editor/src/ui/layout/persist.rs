//! Snapshot of editor layout state for persistence in the project file.
//!
//! The engine treats this as an opaque JSON string so it doesn't need to
//! know about UI types like `PanelId` or `LayoutConfig`. The editor is
//! the sole producer and consumer — it serializes at save time via
//! `SetEditorLayout`, and hydrates on `OpenProject` from the snapshot's
//! `editor_layout` field.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::ui::layout::{LayoutConfig, default_layout};
use crate::ui::store::EditorStore;

/// Everything editor-side we want to survive a project close/reopen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedEditorState {
    pub layout: LayoutConfig,
    pub left_width_px: f32,
    pub right_width_px: f32,
    pub bottom_height_px: f32,
}

impl PersistedEditorState {
    /// Snapshot the current editor layout from the reactive store.
    pub fn capture(store: EditorStore) -> Self {
        Self {
            layout: (*store.layout.get()).clone(),
            left_width_px: store.left_width_px.get(),
            right_width_px: store.right_width_px.get(),
            bottom_height_px: store.bottom_height_px.get(),
        }
    }

    /// Apply a previously-captured layout to the store. Uses Signal::send
    /// (cross-thread) rather than set so this can run directly from the
    /// engine state-callback without a manual main-thread hop. Widths are
    /// clamped to a sane minimum so a corrupt project file can't produce
    /// a sliver-thin container that's impossible to drag back open.
    pub fn apply(&self, store: EditorStore) {
        const MIN: f32 = 80.0;
        store.layout.send(Arc::new(self.layout.clone()));
        store.left_width_px.send(self.left_width_px.max(MIN));
        store.right_width_px.send(self.right_width_px.max(MIN));
        store.bottom_height_px.send(self.bottom_height_px.max(MIN));
    }

    /// Deserialize from JSON, falling back to a default layout on parse
    /// error. Logs the failure so a truly broken sidecar isn't silent.
    /// Also runs `migrate_panels` so newer panel ids that didn't exist
    /// when the layout was saved get appended to a sensible zone — the
    /// user shouldn't have to nuke their layout to discover a new
    /// panel after upgrading.
    pub fn from_json_or_default(json: &str) -> Self {
        let mut state = match serde_json::from_str::<Self>(json) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[arvx-editor] failed to parse editor_layout: {e} — using default");
                Self::default()
            }
        };
        state.layout.migrate_panels();
        state
    }
}

impl Default for PersistedEditorState {
    fn default() -> Self {
        Self {
            layout: default_layout(),
            left_width_px: 250.0,
            right_width_px: 300.0,
            bottom_height_px: 200.0,
        }
    }
}
