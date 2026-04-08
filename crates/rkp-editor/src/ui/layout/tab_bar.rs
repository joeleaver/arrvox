//! Tab bar — horizontal row of tab buttons within a zone.

use rinch::prelude::*;

use super::{ContainerKind, PanelId, panel_registry};
use crate::ui::store::EditorStore;

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
            for tab in tabs.get() {
                div {
                    key: tab.idx.to_string(),
                    style: {
                        let tab_idx = tab.idx;
                        move || {
                            let layout = store.layout.get();
                            let is_active = layout.container(container)
                                .zones.get(zone_idx)
                                .map(|z| z.active_tab == tab_idx)
                                .unwrap_or(false);
                            if is_active {
                                "padding:0 12px;display:flex;align-items:center;cursor:pointer;\
                                 font-size:11px;color:#fff;background:#1e1e1e;\
                                 border-bottom:2px solid #007acc;user-select:none;"
                            } else {
                                "padding:0 12px;display:flex;align-items:center;cursor:pointer;\
                                 font-size:11px;color:#888;background:#2d2d2d;\
                                 border-bottom:2px solid transparent;user-select:none;"
                            }
                        }
                    },
                    onclick: {
                        let tab_idx = tab.idx;
                        move || {
                            store.layout.update(|layout| {
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
