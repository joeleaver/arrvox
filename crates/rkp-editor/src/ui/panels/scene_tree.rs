//! Scene tree panel — hierarchical view of scene objects.
//!
//! Renders root objects (no parent) at the top level, with children
//! nested below their parents. Right-click for context menu (Duplicate, Delete).

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::SceneObjectInfo;

#[component]
pub fn SceneTree() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let roots = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id.is_none())
            .collect::<Vec<_>>()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",
            div {
                style: "flex:1;padding:4px;",
                for obj in roots.get() {
                    {tree_node(__scope, &obj, 0, store)}
                }
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
            // Context menu wraps the entire node
            ContextMenu {
                ContextMenuTarget {
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
                }
                ContextMenuDropdown {
                    DropdownMenuItem {
                        left_section: TablerIcon::Copy,
                        onclick: move || {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::SelectEntity { entity_id: id });
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::DuplicateObject { entity_id: id });
                        },
                        "Duplicate"
                    }
                    DropdownMenuDivider {}
                    DropdownMenuItem {
                        color: "red",
                        left_section: TablerIcon::Trash,
                        onclick: move || {
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::DeleteObject { entity_id: id });
                        },
                        "Delete"
                    }
                }
            }

            // Children
            if !collapsed.get() {
                for child in children.get() {
                    {tree_node(__scope, &child, depth + 1, store)}
                }
            }
        }
    }
}
