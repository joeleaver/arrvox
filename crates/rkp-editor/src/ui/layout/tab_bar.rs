//! Tab bar — horizontal row of tab buttons within a zone.
//!
//! Click to switch tabs. Drag to move tabs between zones.
//! Drop a tab to float it (if dropped outside any zone).

use rinch::prelude::*;

use super::{ContainerKind, PanelId, panel_registry};
use crate::ui::store::{EditorStore, TabDragData, DropTarget};

#[derive(Clone, PartialEq)]
struct TabInfo {
    idx: usize,
    panel: PanelId,
    name: String,
}

#[component]
pub fn TabBar(container: ContainerKind, zone_idx: usize) -> NodeHandle {
    let store = use_context::<EditorStore>();

    let tabs = Memo::new(move || {
        let layout = store.layout.get();
        let c = layout.container(container);
        if let Some(zone) = c.zones.get(zone_idx) {
            zone.tabs.iter().enumerate().map(|(i, &panel)| TabInfo {
                idx: i,
                panel,
                name: panel_registry::panel_name(panel).to_string(),
            }).collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    });

    rsx! {
        div {
            style: "display:flex;height:28px;background:#2d2d2d;border-bottom:1px solid #3c3c3c;\
                    flex-shrink:0;overflow-x:auto;",
            // Drop target: accept tabs dragged into this zone's tab bar.
            ondragenter: move || {
                if store.tab_drag.get().is_some() {
                    store.drop_target.set(Some(DropTarget::Zone {
                        container,
                        zone_idx,
                    }));
                }
            },
            ondrop: move || {
                if let Some(data) = store.tab_drag.get() {
                    store.update_layout(|layout| {
                        let tab_idx = data.tab_index(layout);
                        layout.move_tab(
                            data.source_container, data.source_zone, tab_idx,
                            container, zone_idx,
                        );
                    });
                }
                store.tab_drag.set(None);
                store.drop_target.set(None);
            },
            for tab in tabs.get() {
                div {
                    key: tab.idx.to_string(),
                    draggable: "true",
                    ondragstart: {
                        let panel = tab.panel;
                        move || {
                            store.tab_drag.set(Some(TabDragData {
                                panel,
                                source_container: container,
                                source_zone: zone_idx,
                            }));
                        }
                    },
                    ondragend: move || {
                        // If no drop target was set, float the panel.
                        let drag_data = store.tab_drag.get();
                        let drop = store.drop_target.get();
                        if drop.is_none() {
                            if let Some(data) = drag_data {
                                store.update_layout(|layout| {
                                    let tab_idx = layout.container(data.source_container)
                                        .zones.get(data.source_zone)
                                        .and_then(|z| z.tabs.iter().position(|&p| p == data.panel))
                                        .unwrap_or(0);
                                    layout.float_panel(data.source_container, data.source_zone, tab_idx);
                                });
                            }
                        }
                        store.tab_drag.set(None);
                        store.drop_target.set(None);
                    },
                    style: {
                        let tab_idx = tab.idx;
                        move || {
                            let layout = store.layout.get();
                            let is_active = layout.container(container)
                                .zones.get(zone_idx)
                                .map(|z| z.active_tab == tab_idx)
                                .unwrap_or(false);
                            let is_drop_target = store.drop_target.get() == Some(DropTarget::Zone {
                                container, zone_idx,
                            });
                            if is_active {
                                "padding:0 12px;display:flex;align-items:center;cursor:grab;\
                                 font-size:11px;color:#fff;background:#1e1e1e;\
                                 border-bottom:2px solid #007acc;user-select:none;"
                            } else if is_drop_target {
                                "padding:0 12px;display:flex;align-items:center;cursor:grab;\
                                 font-size:11px;color:#fff;background:#264f78;\
                                 border-bottom:2px solid #007acc;user-select:none;"
                            } else {
                                "padding:0 12px;display:flex;align-items:center;cursor:grab;\
                                 font-size:11px;color:#888;background:#2d2d2d;\
                                 border-bottom:2px solid transparent;user-select:none;"
                            }
                        }
                    },
                    onclick: {
                        let tab_idx = tab.idx;
                        move || {
                            store.update_layout(|layout| {
                                layout.set_active_tab(container, zone_idx, tab_idx);
                            });
                        }
                    },
                    {tab.name.clone()}
                }
            }
        }
    }
}
