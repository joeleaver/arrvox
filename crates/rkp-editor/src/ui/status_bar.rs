//! Status bar — displays engine state: object count, FPS.

use rinch::prelude::*;

use crate::EngineSignals;

#[component]
pub fn StatusBar() -> NodeHandle {
    let signals = use_context::<EngineSignals>();

    rsx! {
        div {
            style: "height:25px;display:flex;align-items:center;padding:0 12px;\
                    background:#252526;color:#858585;font-size:11px;flex-shrink:0;\
                    gap:16px;border-top:1px solid #3c3c3c;",
            span { {|| format!("{} objects", signals.gpu_object_count.get())} }
            span { {|| format!("{:.0} fps", signals.fps.get())} }
        }
    }
}
