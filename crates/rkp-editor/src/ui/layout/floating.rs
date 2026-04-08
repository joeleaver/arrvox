//! Floating panel host — renders detached panels as draggable overlays.
//!
//! Title bar uses HTML5 drag so docked zones show drop targets.
//! Close (×) removes the panel. Drop on a zone to re-dock.

use rinch::prelude::*;

use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use super::{ContainerKind, PanelId, panel_registry};
use crate::ui::store::{EditorStore, TabDragData, DropTarget};
use crate::ui::panels::*;

#[component]
pub fn FloatingPanelHost() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let floating_count = Memo::new(move || {
        store.layout.get().floating.len()
    });

    rsx! {
        for i in 0..floating_count.get() {
            FloatingPanelWindow {
                key: i.to_string(),
                index: i,
            }
        }
    }
}

#[component]
fn FloatingPanelWindow(index: usize) -> NodeHandle {
    let store = use_context::<EditorStore>();

    let panel_info = Memo::new(move || {
        let layout = store.layout.get();
        layout.floating.get(index).cloned()
    });

    let x = Signal::new(panel_info.get().map(|f| f.x).unwrap_or(200.0));
    let y = Signal::new(panel_info.get().map(|f| f.y).unwrap_or(200.0));
    let panel_id = panel_info.get().map(|f| f.panel);

    let name = panel_id.map(|p| panel_registry::panel_name(p)).unwrap_or("Panel");

    rsx! {
        div {
            style: {
                let pi = panel_info;
                move || {
                    let fp = pi.get();
                    let (px, py) = (x.get(), y.get());
                    let (w, h) = fp.map(|f| (f.width, f.height)).unwrap_or((400.0, 300.0));
                    format!(
                        "position:absolute;left:{:.0}px;top:{:.0}px;width:{:.0}px;height:{:.0}px;\
                         z-index:200;background:#252526;border:1px solid #3c3c3c;\
                         border-radius:4px;box-shadow:0 4px 16px rgba(0,0,0,0.4);\
                         display:flex;flex-direction:column;overflow:hidden;",
                        px, py, w, h
                    )
                }
            },
            // Title bar — HTML5 draggable so docked zones show drop targets.
            div {
                style: "height:28px;display:flex;align-items:center;padding:0 8px;\
                        background:#2d2d2d;cursor:grab;flex-shrink:0;\
                        border-bottom:1px solid #3c3c3c;justify-content:space-between;",
                draggable: "true",
                ondragstart: {
                    let panel_id = panel_id;
                    move || {
                        suppress_drag_ghost();
                        if let Some(pid) = panel_id {
                            store.tab_drag.set(Some(TabDragData {
                                panel: pid,
                                source_container: ContainerKind::Left, // dummy for floating
                                source_zone: 0,
                            }));
                        }
                    }
                },
                ondragmove: move || {
                    // Track cursor to update floating panel position during drag.
                    let ctx = get_click_context();
                    // Move panel to follow cursor (offset by half title bar height).
                    x.set(ctx.mouse_x - 100.0);
                    y.set(ctx.mouse_y - 14.0);
                },
                ondragend: move || {
                    restore_drag_ghost();
                    let drop = store.drop_target.get();

                    store.tab_drag.set(None);
                    store.drop_target.set(None);

                    if let Some(dt) = drop {
                        store.update_layout(|layout| {
                            match dt {
                                DropTarget::Zone { container, zone_idx } => {
                                    layout.dock_panel(index, container, zone_idx);
                                }
                                DropTarget::Split { container, zone_idx, edge } => {
                                    if index < layout.floating.len() {
                                        let fp = layout.floating.remove(index);
                                        let before = matches!(edge,
                                            crate::ui::store::SplitEdge::Top |
                                            crate::ui::store::SplitEdge::Left
                                        );
                                        layout.split_zone(fp.panel, container, zone_idx, before);
                                    }
                                }
                            }
                        });
                    } else {
                        // Stayed floating — save new position.
                        store.update_layout(|layout| {
                            if index < layout.floating.len() {
                                layout.floating[index].x = x.get();
                                layout.floating[index].y = y.get();
                            }
                        });
                    }
                },
                span { style: "font-size:11px;color:#ccc;user-select:none;", {name} }
                // Close button (stop propagation so click doesn't start a drag)
                div {
                    style: "cursor:pointer;color:#888;padding:2px;\
                            border-radius:2px;width:14px;height:14px;",
                    onclick: move || {
                        store.update_layout(|layout| {
                            if index < layout.floating.len() {
                                layout.floating.remove(index);
                            }
                        });
                    },
                    {render_tabler_icon(__scope, TablerIcon::X, TablerIconStyle::Outline)}
                }
            }
            // Panel content
            div {
                style: "flex:1;min-height:0;overflow:hidden;",
                if panel_id == Some(PanelId::SceneTree) { SceneTree {} }
                if panel_id == Some(PanelId::ObjectProperties) { ObjectProperties {} }
                if panel_id == Some(PanelId::Materials) { MaterialsPanel {} }
                if panel_id == Some(PanelId::Console) { ConsolePanel {} }
                if panel_id == Some(PanelId::Profiling) { ProfilingPanel {} }
                if panel_id == Some(PanelId::Models) { ModelsPanel {} }
            }
        }
    }
}
