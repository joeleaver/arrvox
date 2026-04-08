//! Scene tree panel — hierarchical view of scene objects.

use rinch::prelude::*;

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

    let icon = if object.is_camera { "\u{f03d}" }
        else if object.is_light { "\u{f4e2}" }
        else { "\u{f1fc}" };

    rsx! {
        div {
            style: {
                move || {
                    let selected = store.selected_entity.get() == Some(id);
                    if selected {
                        "display:flex;align-items:center;padding:2px 8px;cursor:pointer;\
                         border-radius:3px;background:#37373d;color:#fff;font-size:12px;"
                    } else {
                        "display:flex;align-items:center;padding:2px 8px;cursor:pointer;\
                         border-radius:3px;color:#ccc;font-size:12px;"
                    }
                }
            },
            onclick: move || {
                cmd.0.send(rkp_engine::EngineCommand::SelectEntity { entity_id: id }).ok();
            },
            span { style: "margin-right:6px;font-size:10px;opacity:0.6;", {icon} }
            span { {name} }
        }
    }
}
