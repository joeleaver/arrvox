//! Editor UI — root layout and component modules.

pub mod scene_tree;
pub mod status_bar;
pub mod viewport;

use rinch::prelude::*;

use scene_tree::SceneTree;
use status_bar::StatusBar;
use viewport::Viewport;

#[component]
pub fn EditorUi() -> NodeHandle {
    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;background:#1e1e1e;",
            // Titlebar
            div {
                style: "height:36px;display:flex;align-items:center;padding:0 16px;\
                        background:#323233;color:#ccc;font-size:13px;font-weight:500;\
                        flex-shrink:0;border-bottom:1px solid #3c3c3c;",
                "RKIPatch Editor"
            }
            // Main area: scene tree + viewport
            div {
                style: "display:flex;flex:1;min-height:0;",
                // Left: Scene tree
                div {
                    style: "width:250px;flex-shrink:0;border-right:1px solid #3c3c3c;",
                    SceneTree {}
                }
                // Center: Viewport
                div {
                    style: "flex:1;min-height:0;",
                    Viewport {}
                }
            }
            // Status bar
            StatusBar {}
        }
    }
}
