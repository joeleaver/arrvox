//! Scene tree panel — hierarchical view of scene objects.
//!
//! Renders root objects (no parent) at the top level, with children
//! nested below their parents. Supports click-to-select.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::SceneObjectInfo;

#[component]
pub fn SceneTree() -> NodeHandle {
    let store = use_context::<EditorStore>();

    // Only show root objects (no parent) at top level.
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
                    SceneTreeNode {
                        key: obj.id.to_string(),
                        object: obj.clone(),
                        depth: "0".to_string(),
                    }
                }
            }
        }
    }
}

#[component]
fn SceneTreeNode(object: SceneObjectInfo, depth: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let id = object.id;
    let name = object.name.clone();
    let collapsed = Signal::new(false);

    let icon = if object.is_camera { TablerIcon::Camera }
        else if object.is_light { TablerIcon::Bulb }
        else { TablerIcon::Cube };

    // Find children of this object.
    let children = Memo::new(move || {
        store.objects.get().into_iter()
            .filter(|o| o.parent_id == Some(id))
            .collect::<Vec<_>>()
    });

    let has_children = Memo::new(move || !children.get().is_empty());
    let depth_val: u32 = depth.parse().unwrap_or(0);
    let indent = depth_val as f32 * 16.0;

    rsx! {
        div {
            // This node
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
                    cmd.0.send(rkp_engine::EngineCommand::SelectEntity { entity_id: id }).ok();
                },

                // Expand/collapse chevron (only if has children)
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

            // Children (hidden when collapsed)
            if !collapsed.get() {
                for child in children.get() {
                    SceneTreeNode {
                        key: child.id.to_string(),
                        object: child.clone(),
                        depth: (depth_val + 1).to_string(),
                    }
                }
            }
        }
    }
}
