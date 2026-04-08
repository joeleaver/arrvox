//! Profiling panel — frame timing stats.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

#[component]
pub fn ProfilingPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "padding:8px 12px;color:#ccc;font-size:12px;",
            div { {|| format!("FPS: {:.1}", store.fps.get())} }
            div { {|| format!("Objects: {}", store.gpu_object_count.get())} }
        }
    }
}
