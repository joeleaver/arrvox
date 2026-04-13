//! Zone component — tab bar + active panel content.

use rinch::prelude::*;

use super::{ContainerKind, PanelId};
use super::tab_bar::TabBar;
use crate::ui::store::{EditorStore, DropTarget, SplitEdge};
use crate::ui::panels::*;

#[component]
pub fn ZoneComponent(container: ContainerKind, zone_idx: usize) -> NodeHandle {
    let store = use_context::<EditorStore>();

    let tab_count = Memo::new(move || {
        store.layout.get().container(container)
            .zones.get(zone_idx)
            .map(|z| z.tabs.len())
            .unwrap_or(0)
    });

    let active_panel = Memo::new(move || {
        let layout = store.layout.get();
        let c = layout.container(container);
        c.zones.get(zone_idx).and_then(|z| z.tabs.get(z.active_tab).copied())
    });

    let is_dragging = Memo::new(move || store.tab_drag.get().is_some());

    let fraction = Memo::new(move || {
        store.layout.get().container(container)
            .zones.get(zone_idx)
            .map(|z| z.fraction)
            .unwrap_or(1.0)
    });

    rsx! {
        div {
            style: {move || format!(
                "display:flex;flex-direction:column;flex:{};min-height:0;min-width:0;",
                fraction.get()
            )},
            TabBar { container: container, zone_idx: zone_idx }
            // Content area with edge drop targets overlaid.
            div {
                style: {
                    move || {
                        let dt = store.drop_target.get();
                        let is_center_drop = dt == Some(DropTarget::Zone { container, zone_idx });
                        let is_any_edge = matches!(dt,
                            Some(DropTarget::Split { container: c, zone_idx: z, .. })
                            if c == container && z == zone_idx
                        );
                        if is_center_drop {
                            "flex:1;min-height:0;min-width:0;overflow:hidden;position:relative;\
                             outline:2px solid #007acc;outline-offset:-2px;"
                        } else if is_any_edge {
                            "flex:1;min-height:0;min-width:0;overflow:hidden;position:relative;"
                        } else {
                            "flex:1;min-height:0;min-width:0;overflow:hidden;position:relative;"
                        }
                    }
                },
                // Center drop target (the main content area).
                ondragenter: move || {
                    if store.tab_drag.get().is_some() {
                        store.drop_target.set(Some(DropTarget::Zone { container, zone_idx }));
                    }
                },
                ondrop: {
                    let store = store;
                    move || {
                        handle_drop(store, container, zone_idx);
                    }
                },

                // Panel content.
                if active_panel.get() == Some(PanelId::SceneTree) { SceneTree {} }
                if active_panel.get() == Some(PanelId::SceneView) { Viewport {} }
                if active_panel.get() == Some(PanelId::ObjectProperties) { ObjectProperties {} }
                if active_panel.get() == Some(PanelId::AssetProperties) { AssetProperties {} }
                if active_panel.get() == Some(PanelId::Environment) { EnvironmentPanel {} }
                if active_panel.get() == Some(PanelId::Materials) { MaterialsPanel {} }
                if active_panel.get() == Some(PanelId::Console) { ConsolePanel {} }
                if active_panel.get() == Some(PanelId::Profiling) { ProfilingPanel {} }
                if active_panel.get() == Some(PanelId::Models) { ModelsPanel {} }

                // Edge drop zones (only visible during drag).
                if is_dragging.get() {
                    // Top edge
                    EdgeDropZone { container: container, zone_idx: zone_idx, edge: SplitEdge::Top }
                    // Bottom edge
                    EdgeDropZone { container: container, zone_idx: zone_idx, edge: SplitEdge::Bottom }
                    // Left edge
                    EdgeDropZone { container: container, zone_idx: zone_idx, edge: SplitEdge::Left }
                    // Right edge
                    EdgeDropZone { container: container, zone_idx: zone_idx, edge: SplitEdge::Right }
                }
            }
        }
    }
}

fn handle_drop(store: EditorStore, container: ContainerKind, zone_idx: usize) {
    if let Some(data) = store.tab_drag.get() {
        let dt = store.drop_target.get();
        store.update_layout(|layout| {
            let tab_idx = layout.container(data.source_container)
                .zones.get(data.source_zone)
                .and_then(|z| z.tabs.iter().position(|&p| p == data.panel))
                .unwrap_or(0);
            match dt {
                Some(DropTarget::Split { edge, .. }) => {
                    // Remove from source first.
                    let panel = {
                        let src = layout.container_mut(data.source_container);
                        if let Some(zone) = src.zones.get_mut(data.source_zone) {
                            if tab_idx < zone.tabs.len() {
                                let p = zone.tabs.remove(tab_idx);
                                if zone.active_tab >= zone.tabs.len() && zone.active_tab > 0 {
                                    zone.active_tab -= 1;
                                }
                                Some(p)
                            } else { None }
                        } else { None }
                    };
                    if let Some(panel) = panel {
                        let before = matches!(edge, SplitEdge::Top | SplitEdge::Left);
                        layout.split_zone(panel, container, zone_idx, before);
                        layout.cleanup_empty_zones();
                    }
                }
                _ => {
                    // Center drop — add as tab.
                    layout.move_tab(
                        data.source_container, data.source_zone, tab_idx,
                        container, zone_idx,
                    );
                }
            }
        });
    }
    store.tab_drag.set(None);
    store.drop_target.set(None);
}

/// Invisible edge drop zone overlay (25% strip along one edge).
#[component]
fn EdgeDropZone(container: ContainerKind, zone_idx: usize, edge: SplitEdge) -> NodeHandle {
    let store = use_context::<EditorStore>();

    let base_style = match edge {
        SplitEdge::Top => "position:absolute;top:0;left:0;right:0;height:25%;",
        SplitEdge::Bottom => "position:absolute;bottom:0;left:0;right:0;height:25%;",
        SplitEdge::Left => "position:absolute;top:0;left:0;bottom:0;width:25%;",
        SplitEdge::Right => "position:absolute;top:0;right:0;bottom:0;width:25%;",
    };

    let target = DropTarget::Split { container, zone_idx, edge };

    rsx! {
        div {
            style: {
                move || {
                    let active = store.drop_target.get() == Some(target);
                    if active {
                        match edge {
                            SplitEdge::Top => format!("{base_style}background:rgba(0,122,204,0.25);border-bottom:2px solid #007acc;z-index:50;"),
                            SplitEdge::Bottom => format!("{base_style}background:rgba(0,122,204,0.25);border-top:2px solid #007acc;z-index:50;"),
                            SplitEdge::Left => format!("{base_style}background:rgba(0,122,204,0.25);border-right:2px solid #007acc;z-index:50;"),
                            SplitEdge::Right => format!("{base_style}background:rgba(0,122,204,0.25);border-left:2px solid #007acc;z-index:50;"),
                        }
                    } else {
                        format!("{base_style}z-index:50;")
                    }
                }
            },
            ondragenter: move || {
                if store.tab_drag.get().is_some() {
                    store.drop_target.set(Some(target));
                }
            },
            ondrop: move || {
                handle_drop(store, container, zone_idx);
            },
        }
    }
}
