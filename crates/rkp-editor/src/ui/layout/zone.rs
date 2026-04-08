//! Zone component — tab bar + active panel content.

use rinch::prelude::*;

use super::{ContainerKind, PanelId};
use super::tab_bar::TabBar;
use crate::ui::store::{EditorStore, TabDragData};
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

    rsx! {
        div {
            style: "display:flex;flex-direction:column;flex:1;min-height:0;min-width:0;",
            if tab_count.get() > 1 {
                TabBar { container: container, zone_idx: zone_idx }
            }
            // Panel content — one `if` per panel type. Only the matching one renders.
            div {
                style: {
                    move || {
                        let is_drop = store.drop_target.get() == Some(
                            crate::ui::store::DropTarget::Zone { container, zone_idx }
                        );
                        if is_drop {
                            "flex:1;min-height:0;min-width:0;overflow:hidden;\
                             outline:2px solid #007acc;outline-offset:-2px;"
                        } else {
                            "flex:1;min-height:0;min-width:0;overflow:hidden;"
                        }
                    }
                },
                ondragenter: move || {
                    if store.tab_drag.get().is_some() {
                        store.drop_target.set(Some(crate::ui::store::DropTarget::Zone {
                            container,
                            zone_idx,
                        }));
                    }
                },
                ondragleave: move || {
                    if store.drop_target.get() == Some(crate::ui::store::DropTarget::Zone { container, zone_idx }) {
                        store.drop_target.set(None);
                    }
                },
                ondrop: move || {
                    if let Some(data) = store.tab_drag.get() {
                        store.layout.update(|layout| {
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
                if active_panel.get() == Some(PanelId::SceneTree) {
                    SceneTree {}
                }
                if active_panel.get() == Some(PanelId::SceneView) {
                    Viewport {}
                }
                if active_panel.get() == Some(PanelId::ObjectProperties) {
                    ObjectProperties {}
                }
                if active_panel.get() == Some(PanelId::Materials) {
                    MaterialsPanel {}
                }
                if active_panel.get() == Some(PanelId::Console) {
                    ConsolePanel {}
                }
                if active_panel.get() == Some(PanelId::Profiling) {
                    ProfilingPanel {}
                }
                if active_panel.get() == Some(PanelId::Models) {
                    ModelsPanel {}
                }
            }
        }
    }
}
