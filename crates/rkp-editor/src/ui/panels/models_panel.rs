//! Models panel — lists available .rkp model files.
//!
//! Drag a model from this list onto the viewport to place it in the scene.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::ui::store::EditorStore;
use rkp_engine::ModelInfo;

#[component]
pub fn ModelsPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",
            for model in store.available_models.get() {
                ModelItem {
                    key: model.path.clone(),
                    model: model.clone(),
                }
            }
            if store.available_models.get().is_empty() {
                div {
                    style: "padding:12px;color:#666;font-size:12px;font-style:italic;",
                    {"No .rkp models found in project assets/"}
                }
            }
        }
    }
}

#[component]
fn ModelItem(model: ModelInfo) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let path = model.path.clone();
    let name = model.name.clone();
    let size_str = format_size(model.size);

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:4px 8px;\
                    cursor:grab;font-size:12px;color:#ccc;",
            draggable: "true",
            ondragstart: {
                let path = path.clone();
                move || {
                    store.model_drag.set(Some(path.clone()));
                }
            },
            ondragend: move || {
                store.model_drag.set(None);
            },
            span {
                style: "width:16px;height:16px;display:inline-flex;\
                        align-items:center;justify-content:center;flex-shrink:0;color:#999;",
                {render_tabler_icon(__scope, TablerIcon::Cube, TablerIconStyle::Outline)}
            }
            span { style: "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {name}
            }
            span { style: "color:#666;font-size:10px;flex-shrink:0;",
                {size_str}
            }
        }
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
