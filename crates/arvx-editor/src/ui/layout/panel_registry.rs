//! Panel registry — metadata for panel types.
//!
//! Panel rendering is done directly in zone.rs via `match` in `rsx!`.
//! This module provides metadata only (name, canvas status).

use super::PanelId;

/// Human-readable name for a panel.
pub fn panel_name(id: PanelId) -> &'static str {
    match id {
        PanelId::SceneTree => "Scene",
        PanelId::SceneView => "Viewport",
        PanelId::ObjectProperties => "Properties",
        PanelId::AssetProperties => "Asset",
        PanelId::Environment => "Environment",
        PanelId::Materials => "Materials",
        PanelId::Console => "Console",
        PanelId::Profiling => "Profiling",
        PanelId::Models => "Models",
        PanelId::Shaders => "Shaders",
        PanelId::Build => "Build",
    }
}

