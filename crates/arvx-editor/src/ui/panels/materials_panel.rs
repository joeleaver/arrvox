//! Materials panel — material swatch grid for browsing and selecting.
//!
//! Shows all materials in the project as colored swatches. Click to select
//! (properties appear in the Asset Properties panel), drag onto viewport to assign.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;

#[component]
pub fn MaterialsPanel() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            // Toolbar
            div {
                style: "display:flex;align-items:center;padding:4px 8px;\
                        border-bottom:1px solid #333;gap:4px;flex-shrink:0;",
                div {
                    style: "font-size:11px;color:#888;flex:1;",
                    {"Materials"}
                }
                div {
                    style: "width:22px;height:22px;display:flex;align-items:center;\
                            justify-content:center;border-radius:3px;cursor:pointer;\
                            color:#999;background:#2d2d2d;border:1px solid #3c3c3c;",
                    title: "New Material",
                    onclick: {
                        let cmd = cmd.clone();
                        move || {
                            let _ = cmd.0.send(arvx_engine::EngineCommand::CreateMaterial {
                                name: "New Material".into(),
                            });
                        }
                    },
                    span {
                        style: "width:14px;height:14px;display:inline-flex;\
                                align-items:center;justify-content:center;",
                        {render_tabler_icon(__scope, TablerIcon::Plus, TablerIconStyle::Outline)}
                    }
                }
            }

            // Swatch grid
            div {
                style: "display:flex;flex-wrap:wrap;gap:4px;padding:8px;\
                        overflow-y:auto;flex:1;align-content:flex-start;",
                for mat in store.materials.get() {
                    MaterialSwatch {
                        key: mat.id.to_string(),
                        mat_id: mat.id.to_string(),
                        name: mat.name.clone(),
                        color_r: mat.albedo[0].to_string(),
                        color_g: mat.albedo[1].to_string(),
                        color_b: mat.albedo[2].to_string(),
                    }
                }
                if store.materials.get().is_empty() {
                    div {
                        style: "color:#666;font-size:12px;font-style:italic;width:100%;",
                        {"No materials loaded"}
                    }
                }
            }
        }
    }
}

#[component]
fn MaterialSwatch(
    mat_id: String,
    name: String,
    color_r: String,
    color_g: String,
    color_b: String,
) -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    let id: u16 = mat_id.parse().unwrap_or(0);
    let r = color_r.parse::<f32>().unwrap_or(0.8);
    let g = color_g.parse::<f32>().unwrap_or(0.8);
    let b = color_b.parse::<f32>().unwrap_or(0.8);
    let display_name = name.clone();
    let title_name = Signal::new(name);

    rsx! {
        div {
            style: {move || {
                let selected = store.selected_material.get() == Some(id);
                let border = if selected { "#4fc3f7" } else { "#3c3c3c" };
                format!(
                    "width:48px;height:48px;border-radius:4px;\
                     background:rgb({},{},{});\
                     border:2px solid {border};cursor:pointer;\
                     position:relative;flex-shrink:0;",
                    (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                )
            }},
            title: {move || title_name.get()},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let _ = cmd.0.send(arvx_engine::EngineCommand::SelectMaterial {
                        material_id: Some(id),
                    });
                }
            },
            draggable: "true",
            ondragstart: move || {
                store.material_drag.set(Some(id));
            },
            ondragend: move || {
                store.material_drag.set(None);
            },
            // Material name label (small, at bottom)
            div {
                style: "position:absolute;bottom:0;left:0;right:0;\
                        font-size:8px;color:#fff;text-align:center;\
                        background:rgba(0,0,0,0.5);border-radius:0 0 2px 2px;\
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;\
                        padding:1px 2px;",
                {display_name}
            }
        }
    }
}
