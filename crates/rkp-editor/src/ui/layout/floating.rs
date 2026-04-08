//! Floating panel host — renders detached panels as draggable overlays.
//!
//! Drag the title bar to reposition. Drop onto a docked zone to re-dock.
//! Close (×) removes the panel from the layout entirely.

use rinch::prelude::*;

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
            // Title bar — drag to reposition OR dock into a zone.
            div {
                style: "height:28px;display:flex;align-items:center;padding:0 8px;\
                        background:#2d2d2d;cursor:grab;flex-shrink:0;\
                        border-bottom:1px solid #3c3c3c;justify-content:space-between;",
                onmousedown: {
                    let panel_id = panel_id;
                    move || {
                        let ctx = get_click_context();
                        let start_x = x.get();
                        let start_y = y.get();
                        let start_mx = ctx.mouse_x;
                        let start_my = ctx.mouse_y;

                        // Set up tab drag state so docked zones show drop targets.
                        if let Some(pid) = panel_id {
                            store.tab_drag.set(Some(TabDragData {
                                panel: pid,
                                source_container: ContainerKind::Left, // dummy — floating
                                source_zone: 0,
                            }));
                        }

                        Drag::absolute()
                            .on_move(move |mx, my| {
                                x.set(start_x + mx - start_mx);
                                y.set(start_y + my - start_my);
                            })
                            .on_end(move |_mx, _my| {
                                let drop = store.drop_target.get();

                                // Clear drag state.
                                store.tab_drag.set(None);
                                store.drop_target.set(None);

                                // Defer the layout mutation to avoid re-entrant borrow.
                                // Use run_on_main_thread to schedule after the current event.
                                let store = store;
                                rinch::shell::rinch_runtime::run_on_main_thread(move || {
                                    if let Some(dt) = drop {
                                        store.layout.update(|layout| {
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
                                        store.layout.update(|layout| {
                                            if index < layout.floating.len() {
                                                layout.floating[index].x = x.get();
                                                layout.floating[index].y = y.get();
                                            }
                                        });
                                    }
                                });
                            })
                            .start();
                    }
                },
                span { style: "font-size:11px;color:#ccc;user-select:none;", {name} }
                // Close button
                div {
                    style: "cursor:pointer;font-size:14px;color:#888;padding:0 4px;\
                            border-radius:2px;line-height:1;",
                    onclick: move || {
                        rinch::shell::rinch_runtime::run_on_main_thread(move || {
                            store.layout.update(|layout| {
                                if index < layout.floating.len() {
                                    layout.floating.remove(index);
                                }
                            });
                        });
                    },
                    {"\u{00d7}"} // ×
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
