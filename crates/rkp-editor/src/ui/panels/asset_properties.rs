//! Asset Properties panel — shows editable properties for the selected asset.
//!
//! Dispatches based on what kind of asset is currently selected:
//! - Material selected → material PBR properties
//! - Model selected → import profile settings
//!
//! When nothing is selected, shows a placeholder message.

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use super::prop_controls::*;

type CmdSignal = Signal<crossbeam::channel::Sender<rkp_engine::EngineCommand>>;

#[component]
pub fn AssetProperties() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let has_material = Memo::new(move || store.selected_material.get().is_some());
    let has_model = Memo::new(move || {
        store.selected_model.get().is_some() && store.selected_material.get().is_none()
    });
    let has_nothing = Memo::new(move || {
        store.selected_material.get().is_none() && store.selected_model.get().is_none()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;overflow-y:auto;",

            if has_material.get() {
                MaterialPropertiesSection {}
            }

            if has_model.get() {
                ModelPropertiesSection {}
            }

            if has_nothing.get() {
                div {
                    style: "padding:12px;color:#555;font-size:12px;font-style:italic;",
                    {"Select an asset to edit its properties"}
                }
            }
        }
    }
}

// ── Material properties ──────────────────────────────────────────────────

#[component]
fn MaterialPropertiesSection() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);

    let mat_info = Memo::new(move || {
        let sel_id = store.selected_material.get()?;
        store.materials.get().iter().find(|m| m.id == sel_id).cloned()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            if let Some(info) = mat_info.get() {
                div {
                    style: "display:flex;flex-direction:column;gap:0;height:100%;",

                    // Header
                    {asset_header(
                        __scope,
                        mat_info,
                        info.id != 0,
                        cmd_tx,
                    )}

                    // Properties
                    {material_fields(__scope, info.id, info.base_color, info.roughness,
                        info.metallic, info.emission_strength, info.opacity, cmd_tx)}
                }
            }
        }
    }
}

fn asset_header(
    __scope: &mut rinch::core::dom::RenderScope,
    mat_info: Memo<Option<rkp_engine::material_library::MaterialInfo>>,
    deletable: bool,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let mat_id = mat_info.get().map(|m| m.id).unwrap_or(0);

    rsx! {
        div {
            style: "display:flex;align-items:center;padding:6px 8px;\
                    border-bottom:1px solid #333;gap:6px;flex-shrink:0;",

            // Color swatch
            div {
                style: {move || {
                    let (r, g, b) = mat_info.get()
                        .map(|m| (m.base_color[0], m.base_color[1], m.base_color[2]))
                        .unwrap_or((0.8, 0.8, 0.8));
                    format!(
                        "width:24px;height:24px;border-radius:4px;flex-shrink:0;\
                         border:1px solid #3c3c3c;\
                         background:rgb({},{},{});",
                        (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                    )
                }},
            }

            // Name + path
            div {
                style: "flex:1;min-width:0;",
                div {
                    style: "font-size:12px;font-weight:600;color:#ddd;\
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                    {|| mat_info.get().map(|m| m.name.clone()).unwrap_or_default()}
                }
                div {
                    style: "font-size:10px;color:#666;",
                    {|| mat_info.get().map(|m| m.path.clone()).unwrap_or_default()}
                }
            }

            // Delete button
            if deletable {
                div {
                    style: "width:20px;height:20px;display:flex;\
                            align-items:center;justify-content:center;\
                            cursor:pointer;color:#666;border-radius:3px;flex-shrink:0;",
                    title: "Delete Material",
                    onclick: move || {
                        let _ = cmd_tx.get().send(
                            rkp_engine::EngineCommand::DeleteMaterial { material_id: mat_id },
                        );
                    },
                    span {
                        style: "width:14px;height:14px;display:inline-flex;\
                                align-items:center;justify-content:center;",
                        {render_tabler_icon(__scope, TablerIcon::Trash, TablerIconStyle::Outline)}
                    }
                }
            }
        }
    }
}

fn material_fields(
    __scope: &mut rinch::core::dom::RenderScope,
    mat_id: u16,
    base_color: [f32; 4],
    roughness: f32,
    metallic: f32,
    emission_strength: f32,
    opacity: f32,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let color_val = Signal::new(base_color);
    let rough_val = Signal::new(roughness);
    let metal_val = Signal::new(metallic);
    let emiss_val = Signal::new(emission_strength);
    let opac_val = Signal::new(opacity);

    // Helper to create a material field update callback.
    let mat_cb = move |field: &'static str| -> Rc<dyn Fn(f32)> {
        Rc::new(move |v: f32| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateMaterialField {
                material_id: mat_id,
                field: field.into(),
                value: v.to_string(),
            });
        })
    };

    rsx! {
        div {
            style: "padding:8px;display:flex;flex-direction:column;gap:6px;\
                    overflow-y:auto;flex:1;",
            {prop_color(__scope, "Base Color", color_val, Rc::new(move |v: [f32; 4]| {
                let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateMaterialField {
                    material_id: mat_id,
                    field: "base_color".into(),
                    value: format!("[{},{},{},{}]", v[0], v[1], v[2], v[3]),
                });
            }))}
            {prop_slider(__scope, "Roughness", rough_val, 0.0, 1.0, 0.01, mat_cb("roughness"))}
            {prop_slider(__scope, "Metallic", metal_val, 0.0, 1.0, 0.01, mat_cb("metallic"))}
            {prop_slider(__scope, "Emission", emiss_val, 0.0, 10.0, 0.1, mat_cb("emission_strength"))}
            {prop_slider(__scope, "Opacity", opac_val, 0.0, 1.0, 0.01, mat_cb("opacity"))}
        }
    }
}

// ── Model import properties ───────────────────────────────────────────────

#[component]
fn ModelPropertiesSection() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);

    let model_info = Memo::new(move || {
        let sel_path = store.selected_model.get()?;
        store.available_models.get().iter()
            .find(|m| m.source_path == sel_path)
            .cloned()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            if let Some(info) = model_info.get() {
                div {
                    style: "display:flex;flex-direction:column;gap:0;height:100%;",

                    // Header
                    div {
                        style: "display:flex;align-items:center;padding:6px 8px;\
                                border-bottom:1px solid #333;gap:6px;flex-shrink:0;",
                        span {
                            style: "width:20px;height:20px;display:inline-flex;\
                                    align-items:center;justify-content:center;\
                                    flex-shrink:0;color:#999;",
                            {render_tabler_icon(__scope, TablerIcon::Cube, TablerIconStyle::Outline)}
                        }
                        div {
                            style: "flex:1;min-width:0;",
                            div {
                                style: "font-size:12px;font-weight:600;color:#ddd;\
                                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                                {|| model_info.get().map(|m| m.name.clone()).unwrap_or_default()}
                            }
                            div {
                                style: "font-size:10px;color:#666;overflow:hidden;\
                                        text-overflow:ellipsis;white-space:nowrap;",
                                {|| model_info.get().map(|m| m.source_path.clone()).unwrap_or_default()}
                            }
                        }
                        div {
                            style: "font-size:10px;color:#666;flex-shrink:0;",
                            {format_size(info.size)}
                        }
                    }

                    // Import fields
                    {model_import_fields(__scope, info.source_path.clone(), info.import_profile.clone(), cmd_tx)}
                }
            }
        }
    }
}

fn model_import_fields(
    __scope: &mut rinch::core::dom::RenderScope,
    source_path: String,
    profile: Option<rkp_engine::import_profile::ImportProfile>,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    let profile = profile.unwrap_or_default();
    let src = Signal::new(source_path);

    // Track whether this source is currently being re-imported on the
    // engine thread. Used below to swap the Re-import button for a
    // progress indicator — voxelizing a high-poly mesh at a fine tier
    // can take minutes, and the button gave no feedback otherwise.
    let is_importing = Memo::new(move || {
        let s = src.get();
        store.importing_models.get().iter().any(|p| *p == s)
    });

    let display_name = Signal::new(profile.display_name.unwrap_or_default());
    // Voxel size is chosen from the four standard tiers (matches the
    // procedural build panel + mesh-import auto-detect tiers in
    // voxelize_opacity::auto_voxel_size). "auto" means pick the coarsest
    // tier that still gives ≥8 bricks on the longest mesh axis.
    let voxel_size_str = Signal::new(
        profile.voxel_size
            .map(format_voxel_tier)
            .unwrap_or_else(|| "auto".into()),
    );
    let target_size = Signal::new(profile.target_size);
    let no_normalize = Signal::new(profile.no_normalize);
    let import_colors = Signal::new(profile.import_colors);
    let rot_x = Signal::new(profile.rotation_offset[0]);
    let rot_y = Signal::new(profile.rotation_offset[1]);
    let rot_z = Signal::new(profile.rotation_offset[2]);

    // Helper to create an import field update callback.
    let import_cb = move |field: &'static str| -> Rc<dyn Fn(String)> {
        Rc::new(move |v: String| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v,
            });
        })
    };
    let import_cb_f32 = move |field: &'static str| -> Rc<dyn Fn(f32)> {
        Rc::new(move |v: f32| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v.to_string(),
            });
        })
    };
    let import_cb_bool = move |field: &'static str| -> Rc<dyn Fn(bool)> {
        Rc::new(move |v: bool| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateImportField {
                source_path: src.get(),
                field: field.into(),
                value: v.to_string(),
            });
        })
    };

    rsx! {
        div {
            style: "padding:8px;display:flex;flex-direction:column;gap:6px;\
                    overflow-y:auto;flex:1;",

            {prop_text(__scope, "Name", display_name, import_cb("display_name"))}
            {prop_select(
                __scope,
                "Voxel Size",
                voxel_size_str,
                &[
                    ("auto", "Auto"),
                    ("0.005", "5mm (finest)"),
                    ("0.02", "2cm"),
                    ("0.08", "8cm"),
                    ("0.32", "32cm (coarsest)"),
                ],
                import_cb("voxel_size"),
            )}
            {prop_slider(__scope, "Target Size", target_size, 0.1, 10.0, 0.1, import_cb_f32("target_size"))}
            {prop_slider(__scope, "Rotate X", rot_x, -180.0, 180.0, 1.0, import_cb_f32("rotation_x"))}
            {prop_slider(__scope, "Rotate Y", rot_y, -180.0, 180.0, 1.0, import_cb_f32("rotation_y"))}
            {prop_slider(__scope, "Rotate Z", rot_z, -180.0, 180.0, 1.0, import_cb_f32("rotation_z"))}
            {prop_checkbox(__scope, "Import Colors", import_colors, import_cb_bool("import_colors"))}
            {prop_checkbox(__scope, "Keep Original Scale", no_normalize, import_cb_bool("no_normalize"))}

            // Re-import button — replaced by a progress indicator while
            // the engine is voxelizing this source.
            div {
                style: "margin-top:4px;",
                if is_importing.get() {
                    div {
                        style: "display:flex;align-items:center;justify-content:center;\
                                gap:8px;padding:6px 12px;background:#2d2d2d;\
                                border:1px solid #3c3c3c;border-radius:4px;\
                                color:#bbb;font-size:11px;",
                        Loader { r#type: "oval", size: "xs", color: "blue" }
                        {"Importing…"}
                    }
                } else {
                    {prop_button(__scope, "Re-import", "#2d5a2d", Rc::new(move || {
                        let _ = cmd_tx.get().send(rkp_engine::EngineCommand::ReimportModel {
                            source_path: src.get(),
                        });
                    }))}
                }
            }
        }
    }
}

/// Snap a stored voxel size to the nearest standard tier string so the
/// dropdown always shows a value it offers. Out-of-tier values (legacy
/// profiles, hand-edited sidecars) get the nearest match.
fn format_voxel_tier(v: f32) -> String {
    const TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];
    let mut best = TIERS[0];
    let mut best_err = (v - best).abs();
    for &t in &TIERS[1..] {
        let err = (v - t).abs();
        if err < best_err {
            best = t;
            best_err = err;
        }
    }
    format!("{best}")
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
