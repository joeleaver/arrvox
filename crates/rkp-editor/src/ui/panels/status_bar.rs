//! Status bar — displays engine state: object count, FPS, gizmo mode.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

#[component]
pub fn StatusBar() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "height:25px;display:flex;align-items:center;padding:0 12px;\
                    background:#252526;color:#858585;font-size:11px;flex-shrink:0;\
                    gap:16px;border-top:1px solid #3c3c3c;",
            span { {|| format!("{} objects", store.gpu_object_count.get())} }
            span { {|| format!("{:.0} fps", store.fps.get())} }
            div { style: "flex:1;" }
            span { {|| format!("{:?}", store.gizmo_mode.get())} }
        }
    }
}
