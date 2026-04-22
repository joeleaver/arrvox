//! Scene tree panel — hierarchical view of scene objects.
//!
//! Renders root objects (no parent) at the top level, with children
//! nested below their parents. Right-click for context menu:
//! Duplicate, Delete, and — for procedural objects — "Copy to New
//! Voxel Object" and "Convert to Voxel Object" (the latter stages
//! the entity into `store.convert_procedural_target`; the
//! confirmation modal is mounted by `LayoutRoot` so the dialog can
//! catch clicks even when the scene-tree panel is narrow).

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
    let is_procedural = object.is_procedural;
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
                        draggable: "true",
                        ondragstart: move || {
                            store.scene_tree_drag.set(Some(id));
                        },
                        ondragend: move || {
                            store.scene_tree_drag.set(None);
                        },
                        // `ondragover` must exist for the row to accept
                        // drops. We don't compute live drop-position
                        // feedback here — ondrop reads the cursor
                        // position from the click context instead.
                        ondragover: move || {},
                        ondrop: move || {
                            let Some(source_id) = store.scene_tree_drag.get() else { return };
                            // Self-drop no-op; the engine would also
                            // reject it, but short-circuit the command.
                            if source_id == id { return; }
                            // Row-relative Y → {Before, Inside, After}.
                            // Top/bottom 30 % are sibling insertion;
                            // middle 40 % re-parents inside target.
                            let ctx = rinch::core::get_click_context();
                            let h = ctx.element_height.max(1.0);
                            let rel_y = (ctx.mouse_y - ctx.element_y).max(0.0) / h;
                            let position = if rel_y < 0.30 {
                                rkp_engine::DropPosition::Before
                            } else if rel_y > 0.70 {
                                rkp_engine::DropPosition::After
                            } else {
                                rkp_engine::DropPosition::Inside
                            };
                            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::ReorderEntity {
                                entity: source_id,
                                target: id,
                                position,
                            });
                            store.scene_tree_drag.set(None);
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
                    // Procedural-only entries. Grouped together and
                    // separated from the generic actions with a
                    // divider so the destructive "Convert" item is
                    // visually bracketed and less likely to be hit
                    // by accident on the way to Delete.
                    if is_procedural {
                        DropdownMenuDivider {}
                        DropdownMenuItem {
                            left_section: TablerIcon::Copy,
                            onclick: move || {
                                let _ = cmd_tx.get().send(
                                    rkp_engine::EngineCommand::CopyProceduralToNewVoxel {
                                        entity_id: id,
                                    },
                                );
                            },
                            "Copy to New Voxel Object"
                        }
                        DropdownMenuItem {
                            left_section: TablerIcon::Cube,
                            onclick: move || {
                                // Destructive — stage the target
                                // on the store; `LayoutRoot` mounts
                                // the confirmation modal.
                                store.convert_procedural_target.set(Some(id));
                            },
                            "Convert to Voxel Object…"
                        }
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
