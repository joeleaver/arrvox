//! Status bar — displays engine state summary.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

#[component]
pub fn StatusBar() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "height:28px;display:flex;align-items:center;padding:0 12px;\
                    background:#252526;color:#858585;font-size:11px;flex-shrink:0;\
                    gap:12px;border-top:1px solid #3c3c3c;",

            // Playing indicator
            if store.play_mode.get() {
                div {
                    style: "color:#4caf50;font-weight:600;font-size:11px;",
                    {"PLAYING"}
                }
            }

            // Spacer
            div { style: "flex:1;" }

            // Stats
            span { {|| format!("{} objects", store.gpu_object_count.get())} }
            span { {|| format!("{:.0} fps", store.fps.get())} }
            span { {|| format!("{:?}", store.gizmo_mode.get())} }
        }
    }
}
