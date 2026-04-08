//! Object properties panel — displays selected object's transform and metadata.

use rinch::prelude::*;

use crate::ui::store::EditorStore;

#[component]
pub fn ObjectProperties() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "padding:8px 12px;color:#ccc;font-size:12px;",
            if store.selected_entity.get().is_some() {
                div {
                    div { style: "font-weight:600;margin-bottom:8px;", "Object Properties" }
                    div { style: "color:#888;", {|| {
                        let id = store.selected_entity.get();
                        format!("Entity: {:?}", id)
                    }} }
                }
            }
            if store.selected_entity.get().is_none() {
                div { style: "color:#666;font-style:italic;", "No object selected" }
            }
        }
    }
}
