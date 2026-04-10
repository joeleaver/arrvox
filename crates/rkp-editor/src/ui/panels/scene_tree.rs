//! Scene tree panel — hierarchical view of scene objects.
//!
//! Renders root objects (no parent) at the top level, with children
//! nested below their parents. Right-click for context menu.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::SceneObjectInfo;

#[component]
pub fn SceneTree() -> NodeHandle {
    let store = use_context::<EditorStore>();
    // Context menu: which entity UUID to show actions for (None = closed).
    let ctx_entity = Signal::new(None::<uuid::Uuid>);

    let roots = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id.is_none())
            .collect::<Vec<_>>()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;position:relative;",
            // Close context menu on left-click background.
            onclick: move || ctx_entity.set(None),
            div {
                style: "flex:1;padding:4px;",
                for obj in roots.get() {
                    {tree_node(__scope, &obj, 0, store, ctx_entity)}
                }
            }

            // Context menu overlay
            if ctx_entity.get().is_some() {
                {context_menu_overlay(__scope, ctx_entity)}
            }
        }
    }
}

/// Render a single tree node + its children recursively.
fn tree_node(
    __scope: &mut rinch::core::dom::RenderScope,
    object: &SceneObjectInfo,
    depth: u32,
    store: EditorStore,
    ctx_entity: Signal<Option<uuid::Uuid>>,
) -> rinch::core::dom::NodeHandle {
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);
    let id = object.id;
    let name = object.name.clone();
    let collapsed = Signal::new(false);
    let indent = depth as f32 * 16.0;

    let icon = if object.is_camera { TablerIcon::Camera }
        else if object.is_light { TablerIcon::Bulb }
        else { TablerIcon::Cube };

    let children = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id == Some(id))
            .collect::<Vec<_>>()
    });
    let has_children = Memo::new(move || !children.get().is_empty());

    rsx! {
        div {
            // Node row
            div {
                style: {
                    move || {
                        let selected = store.selected_entity.get() == Some(id);
                        let bg = if selected { "background:#37373d;color:#fff;" } else { "color:#ccc;" };
                        format!(
                            "display:flex;align-items:center;padding:2px 8px 2px {:.0}px;\
                             cursor:pointer;border-radius:3px;font-size:12px;gap:4px;{bg}",
                            8.0 + indent
                        )
                    }
                },
                onclick: move || {
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SelectEntity { entity_id: id });
                },
                oncontextmenu: move || {
                    let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SelectEntity { entity_id: id });
                    ctx_entity.set(Some(id));
                },

                // Chevron
                if has_children.get() {
                    span {
                        style: {move || {
                            if collapsed.get() {
                                "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                                 transform:rotate(-90deg);transition:transform 0.15s;"
                            } else {
                                "font-size:8px;color:#666;cursor:pointer;width:12px;text-align:center;\
                                 transition:transform 0.15s;"
                            }
                        }},
                        onclick: move || collapsed.update(|c| *c = !*c),
                        {"\u{25BC}"}
                    }
                }
                if !has_children.get() {
                    span { style: "width:12px;" }
                }

                // Icon
                span {
                    style: "width:14px;height:14px;display:inline-flex;align-items:center;\
                            justify-content:center;flex-shrink:0;color:#999;",
                    {render_tabler_icon(__scope, icon, TablerIconStyle::Outline)}
                }

                // Name
                span { style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", {name} }
            }

            // Children
            if !collapsed.get() {
                for child in children.get() {
                    {tree_node(__scope, &child, depth + 1, store, ctx_entity)}
                }
            }
        }
    }
}

/// Context menu overlay: backdrop + menu items.
fn context_menu_overlay(
    __scope: &mut rinch::core::dom::RenderScope,
    ctx_entity: Signal<Option<uuid::Uuid>>,
) -> rinch::core::dom::NodeHandle {
    let cmd_tx = Signal::new(use_context::<CommandSender>().0);

    rsx! {
        div {
            // Backdrop
            div {
                style: "position:absolute;inset:0;z-index:999;",
                onclick: move || ctx_entity.set(None),
            }
            // Menu
            div {
                style: "position:absolute;right:8px;top:40px;z-index:1000;\
                        background:#2d2d2d;border:1px solid #3c3c3c;border-radius:4px;\
                        box-shadow:0 4px 12px rgba(0,0,0,0.5);min-width:140px;\
                        padding:4px 0;",

                // Duplicate
                div {
                    style: "padding:6px 16px;cursor:pointer;font-size:12px;color:#ccc;",
                    onclick: move || {
                        if let Some(eid) = ctx_entity.get() {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::DuplicateObject {
                                entity_id: eid,
                            });
                        }
                        ctx_entity.set(None);
                    },
                    {"Duplicate"}
                }

                // Separator
                div { style: "height:1px;background:#3c3c3c;margin:4px 0;" }

                // Delete
                div {
                    style: "padding:6px 16px;cursor:pointer;font-size:12px;color:#ef5350;",
                    onclick: move || {
                        if let Some(eid) = ctx_entity.get() {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::DeleteObject {
                                entity_id: eid,
                            });
                        }
                        ctx_entity.set(None);
                    },
                    {"Delete"}
                }
            }
        }
    }
}
