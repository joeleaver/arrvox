//! Scene tree panel — hierarchical view of scene objects.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use rkp_engine::SceneObjectInfo;

#[component]
pub fn SceneTree() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",
            div {
                style: "flex:1;padding:4px;",
                for obj in store.objects.get() {
                    SceneTreeItem {
                        key: obj.id.to_string(),
                        object: obj.clone(),
                    }
                }
            }
        }
    }
}

#[component]
fn SceneTreeItem(object: SceneObjectInfo) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let id = object.id;
    let name = object.name.clone();

    let icon = if object.is_camera { TablerIcon::Camera }
        else if object.is_light { TablerIcon::Bulb }
        else { TablerIcon::Cube };

    rsx! {
        div {
            style: {
                move || {
                    let selected = store.selected_entity.get() == Some(id);
                    if selected {
                        "display:flex;align-items:center;padding:2px 8px;cursor:pointer;\
                         border-radius:3px;background:#37373d;color:#fff;font-size:12px;gap:6px;"
                    } else {
                        "display:flex;align-items:center;padding:2px 8px;cursor:pointer;\
                         border-radius:3px;color:#ccc;font-size:12px;gap:6px;"
                    }
                }
            },
            onclick: move || {
                cmd.0.send(rkp_engine::EngineCommand::SelectEntity { entity_id: id }).ok();
            },
            div { style: "flex-shrink:0;opacity:0.6;width:14px;height:14px;",
                {render_tabler_icon(__scope, icon, TablerIconStyle::Outline)}
            }
            span { {name} }
        }
    }
}
