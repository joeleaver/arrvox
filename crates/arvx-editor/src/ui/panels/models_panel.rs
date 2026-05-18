//! Models panel — lists available .arvx model files and registered
//! generators.
//!
//! Drag a model onto the viewport to place it in the scene. Click a
//! generator to spawn it. (Drag-preview UX for generators is a future
//! polish item.)

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use arvx_engine::ModelInfo;

#[component]
pub fn ModelsPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",

            // ── Models section ──────────────────────────────────
            for model in store.available_models.get() {
                ModelItem {
                    key: model.path.clone(),
                    model: model.clone(),
                }
            }
            if store.available_models.get().is_empty() {
                div {
                    style: "padding:12px;color:#666;font-size:12px;font-style:italic;",
                    {"No .arvx models found in project assets/"}
                }
            }

            // ── Generators section ─────────────────────────────
            // Only shown when at least one is registered (via the
            // gameplay dylib). Separator header makes the list scope
            // obvious in a mixed panel.
            if !store.available_generators.get().is_empty()
                || !store.available_generator_presets.get().is_empty()
            {
                div {
                    style: "padding:6px 8px;margin-top:4px;border-top:1px solid #2d2d30;\
                            color:#888;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;",
                    {"Generators"}
                }
                // Presets first — they're the "ready-to-go" curated
                // entries; bare generators below are the raw types.
                for preset in store.available_generator_presets.get() {
                    GeneratorPresetItem {
                        key: preset.path.clone(),
                        path: preset.path.clone(),
                        display_name: preset.display_name.clone(),
                        generator_name: preset.generator_name.clone(),
                    }
                }
                for name in store.available_generators.get() {
                    GeneratorItem {
                        key: name.clone(),
                        name: name.clone(),
                    }
                }
            }
        }
    }
}

#[component]
fn ModelItem(model: ModelInfo) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let arvx_path = model.path.clone();
    let source_path = Signal::new(model.source_path.clone());
    let name = model.name.clone();
    let size_str = format_size(model.size);
    let has_source = !model.source_path.is_empty();

    rsx! {
        div {
            style: {move || {
                let sp = source_path.get();
                let selected = store.selected_model.get().as_deref() == Some(sp.as_str());
                if selected && has_source {
                    "display:flex;align-items:center;gap:8px;padding:4px 8px;\
                     cursor:grab;font-size:12px;color:#ccc;background:#37373d;"
                } else {
                    "display:flex;align-items:center;gap:8px;padding:4px 8px;\
                     cursor:grab;font-size:12px;color:#ccc;"
                }
            }},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let sp = source_path.get();
                    if !sp.is_empty() {
                        let _ = cmd.0.send(arvx_engine::EngineCommand::SelectModel {
                            path: Some(sp),
                        });
                    }
                }
            },
            draggable: "true",
            ondragstart: {
                let path = arvx_path.clone();
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

/// A single registered generator. Click spawns it 3m in front of the
/// camera; drag-and-drop starts a live preview (see `viewport.rs`'s
/// `DragEnter/Over/Drop` handlers and `EngineCommand::DragPreview*`).
#[component]
fn GeneratorItem(name: String) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let click_name = name.clone();
    let drag_name = name.clone();

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:4px 8px;\
                    cursor:grab;font-size:12px;color:#ccc;",
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let _ = cmd.0.send(arvx_engine::EngineCommand::SpawnGenerator {
                        generator_name: click_name.clone(),
                    });
                }
            },
            draggable: "true",
            ondragstart: {
                let n = drag_name.clone();
                move || { store.generator_drag.set(Some(n.clone())); }
            },
            ondragend: move || { store.generator_drag.set(None); },
            span {
                style: "width:16px;height:16px;display:inline-flex;\
                        align-items:center;justify-content:center;flex-shrink:0;color:#c09060;",
                {render_tabler_icon(__scope, TablerIcon::Sparkles, TablerIconStyle::Outline)}
            }
            span { style: "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {name}
            }
            span { style: "color:#666;font-size:10px;flex-shrink:0;",
                {"generator"}
            }
        }
    }
}

/// A `.arvxgen` preset row. Click spawns the generator (3m in front of
/// camera) with the preset's overrides applied; drag-and-drop spawns
/// it at the drop pixel's surface point.
#[component]
fn GeneratorPresetItem(
    path: String,
    display_name: String,
    generator_name: String,
) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let click_path = path.clone();
    let drag_path = path.clone();
    let label = format!("{display_name}");
    let type_label = format!("{generator_name} preset");

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:4px 8px;\
                    cursor:grab;font-size:12px;color:#ccc;",
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let _ = cmd.0.send(arvx_engine::EngineCommand::SpawnGeneratorPreset {
                        path: click_path.clone(),
                    });
                }
            },
            draggable: "true",
            ondragstart: {
                let p = drag_path.clone();
                move || { store.generator_preset_drag.set(Some(p.clone())); }
            },
            ondragend: move || { store.generator_preset_drag.set(None); },
            span {
                style: "width:16px;height:16px;display:inline-flex;\
                        align-items:center;justify-content:center;flex-shrink:0;color:#80a0c0;",
                {render_tabler_icon(__scope, TablerIcon::Package, TablerIconStyle::Outline)}
            }
            span { style: "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {label}
            }
            span { style: "color:#666;font-size:10px;flex-shrink:0;",
                {type_label}
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
