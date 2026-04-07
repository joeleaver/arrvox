//! Scene tree panel — hierarchical view of scene objects.
//!
//! Reads the objects list from EngineSignals. Supports expand/collapse
//! and click-to-select.

use rinch::prelude::*;

use crate::{CommandSender, EngineSignals};
use rkp_engine::SceneObjectInfo;

#[component]
pub fn SceneTree() -> NodeHandle {
    let signals = use_context::<EngineSignals>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;\
                    background:#252526;",
            // Header
            div {
                style: "padding:8px 12px;font-size:11px;font-weight:600;color:#bbb;\
                        text-transform:uppercase;letter-spacing:0.5px;",
                "Scene"
            }
            // Tree content
            div {
                style: "flex:1;padding:0 4px;",
                for obj in signals.objects.get() {
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
    let signals = use_context::<EngineSignals>();
    let cmd = use_context::<CommandSender>();
    let id = object.id;
    let name = object.name.clone();

    let icon = if object.is_camera {
        "\u{f03d}" // camera icon (tabler)
    } else if object.is_light {
        "\u{f4e2}" // bulb icon
    } else {
        "\u{f1fc}" // cube icon
    };

    rsx! {
        div {
            style: {
                let signals = signals;
                move || {
                    let selected = signals.selected_entity.get() == Some(id);
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
