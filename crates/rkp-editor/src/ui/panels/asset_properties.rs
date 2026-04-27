//! Asset Properties panel — shows editable properties for the selected asset.
//!
//! Dispatches based on what kind of asset is currently selected:
//! - Material selected → material PBR properties
//! - Model selected → import profile settings
//!
//! When nothing is selected, shows a placeholder message.

// `info` bound inside `if let Some(info) = ...` expands through the rsx
// macro into generated closures that rustc's usage analysis can't see
// past. The binding looks unused at the pattern site even though the
// expanded code uses it. Local `let _ = &info;` silencers are stripped
// by the macro, so we allow at module scope.
#![allow(unused_variables)]

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

                    // Properties — the built-in Default (id 0) is immutable;
                    // show a nudge to create a new material instead.
                    if info.id == 0 {
                        div {
                            style: "padding:12px;display:flex;flex-direction:column;\
                                    gap:10px;color:#888;font-size:11px;",
                            div {
                                style: "font-style:italic;line-height:1.5;",
                                {"The Default material is read-only. \
                                  Create a new material to customize PBR properties."}
                            }
                            div {
                                style: "padding:6px 10px;border-radius:4px;\
                                        background:#2d5a2d;color:#ddd;\
                                        text-align:center;cursor:pointer;\
                                        font-size:11px;user-select:none;",
                                onclick: move || {
                                    let _ = cmd_tx.get().send(
                                        rkp_engine::EngineCommand::CreateMaterial {
                                            name: "New Material".into(),
                                        },
                                    );
                                },
                                {"+ New Material"}
                            }
                        }
                    } else {
                        // Remount the editable form on selection change
                        // by keying a `for` loop on the material id.
                        // Without this, `material_fields`'s inner
                        // `Signal::new(...)` seeds run only at first
                        // mount and the sliders / color pickers stay
                        // pinned to the first-selected material's
                        // values when you pick a different one. Same
                        // pattern used for `ModelPropertiesForm` below.
                        // Re-reading `mat_info` inside the for keeps
                        // rinch's enclosing body-closure `Fn`-callable;
                        // capturing `info` from the outer `if let`
                        // would move it and force `FnOnce`.
                        for info_keyed in mat_info.get().into_iter() {
                            MaterialPropertiesForm {
                                key: info_keyed.id.to_string(),
                                info: info_keyed,
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn MaterialPropertiesForm(
    info: rkp_engine::material_library::MaterialInfo,
) -> NodeHandle {
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    material_fields(__scope, info.id, info, cmd_tx)
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
                        .map(|m| (m.albedo[0], m.albedo[1], m.albedo[2]))
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
    info: rkp_engine::material_library::MaterialInfo,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    // prop_color takes [f32;4]; we pass albedo through as rgb with alpha=1.0
    // and strip the alpha on commit. The alpha is unused (opacity is a
    // separate material field).
    let albedo_val = Signal::new([info.albedo[0], info.albedo[1], info.albedo[2], 1.0]);
    let emission_col_val = Signal::new([
        info.emission_color[0], info.emission_color[1], info.emission_color[2], 1.0,
    ]);
    let sss_col_val = Signal::new([
        info.subsurface_color[0], info.subsurface_color[1], info.subsurface_color[2], 1.0,
    ]);
    let rough_val = Signal::new(info.roughness);
    let metal_val = Signal::new(info.metallic);
    let emiss_val = Signal::new(info.emission_strength);
    let sss_val = Signal::new(info.subsurface);
    let opac_val = Signal::new(info.opacity);
    let ior_val = Signal::new(info.ior);
    let noise_scale_val = Signal::new(info.noise_scale);
    let noise_strength_val = Signal::new(info.noise_strength);
    let noise_channels = Signal::new(info.noise_channels);

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
    let color_cb = move |field: &'static str| -> Rc<dyn Fn([f32; 4])> {
        Rc::new(move |v: [f32; 4]| {
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateMaterialField {
                material_id: mat_id,
                field: field.into(),
                value: format!("[{},{},{}]", v[0], v[1], v[2]),
            });
        })
    };

    // Noise channel bitflag toggle. Sends the whole bitmask each time.
    let noise_channel_cb = move |bit: u32| -> Rc<dyn Fn(bool)> {
        Rc::new(move |on: bool| {
            let cur = noise_channels.get();
            let new = if on { cur | bit } else { cur & !bit };
            noise_channels.set(new);
            let _ = cmd_tx.get().send(rkp_engine::EngineCommand::UpdateMaterialField {
                material_id: mat_id,
                field: "noise_channels".into(),
                value: new.to_string(),
            });
        })
    };

    const NOISE_ALBEDO: u32 = 1 << 0;
    const NOISE_ROUGHNESS: u32 = 1 << 1;
    const NOISE_NORMAL: u32 = 1 << 2;

    let noise_albedo_on = Signal::new(info.noise_channels & NOISE_ALBEDO != 0);
    let noise_rough_on = Signal::new(info.noise_channels & NOISE_ROUGHNESS != 0);
    let noise_normal_on = Signal::new(info.noise_channels & NOISE_NORMAL != 0);

    rsx! {
        div {
            style: "padding:8px;display:flex;flex-direction:column;gap:6px;\
                    overflow-y:auto;flex:1;",

            // PBR baseline
            {prop_color(__scope, "Albedo", Memo::new(move || albedo_val.get()), color_cb("albedo"))}
            {prop_slider(__scope, "Roughness", rough_val, 0.0, 1.0, 0.01, mat_cb("roughness"))}
            {prop_slider(__scope, "Metallic", metal_val, 0.0, 1.0, 0.01, mat_cb("metallic"))}

            // Emission
            {prop_color(__scope, "Emission Color", Memo::new(move || emission_col_val.get()), color_cb("emission_color"))}
            {prop_slider(__scope, "Emission", emiss_val, 0.0, 10.0, 0.1, mat_cb("emission_strength"))}

            // Translucency
            {prop_slider(__scope, "Subsurface", sss_val, 0.0, 1.0, 0.01, mat_cb("subsurface"))}
            {prop_color(__scope, "SSS Color", Memo::new(move || sss_col_val.get()), color_cb("subsurface_color"))}
            {prop_slider(__scope, "Opacity", opac_val, 0.0, 1.0, 0.01, mat_cb("opacity"))}
            {prop_slider(__scope, "IOR", ior_val, 1.0, 3.0, 0.01, mat_cb("ior"))}

            // Procedural noise
            {prop_slider(__scope, "Noise Scale", noise_scale_val, 0.0, 50.0, 0.1, mat_cb("noise_scale"))}
            {prop_slider(__scope, "Noise Strength", noise_strength_val, 0.0, 1.0, 0.01, mat_cb("noise_strength"))}
            {prop_checkbox(__scope, "Noise: Albedo", noise_albedo_on, noise_channel_cb(NOISE_ALBEDO))}
            {prop_checkbox(__scope, "Noise: Roughness", noise_rough_on, noise_channel_cb(NOISE_ROUGHNESS))}
            {prop_checkbox(__scope, "Noise: Normal", noise_normal_on, noise_channel_cb(NOISE_NORMAL))}

            // User shader binding + dynamic param controls.
            {shader_section(__scope, mat_id, info.clone(), cmd_tx)}
        }
    }
}

/// Shader dropdown + one slider per `@param` for whichever shader the
/// material currently has assigned. Pulls the registered shader list
/// from the editor store; the engine keeps it in sync each tick via
/// the `user_shaders` snapshot field.
///
/// "None" selection clears `MaterialDef.shader`, which the engine
/// resolves to `shader_id = 0` (PBR) on the next material upload.
fn shader_section(
    __scope: &mut rinch::core::dom::RenderScope,
    mat_id: u16,
    info: rkp_engine::material_library::MaterialInfo,
    cmd_tx: CmdSignal,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    let current_shader: String = info.shader.clone().unwrap_or_default();

    // Build the dropdown options at render time. Re-runs only on
    // remount (whole material swap) or when the user_shaders signal
    // changes — both cases want a fresh option list.
    let shader_list = store.user_shaders.get();
    let mut options: Vec<(String, String)> =
        Vec::with_capacity(shader_list.len() + 1);
    options.push((String::new(), "(none — PBR)".to_string()));
    for s in &shader_list {
        options.push((s.name.clone(), s.name.clone()));
    }
    let options_refs: Vec<(&str, &str)> = options
        .iter()
        .map(|(v, l)| (v.as_str(), l.as_str()))
        .collect();

    let current = Signal::new(current_shader.clone());
    let dropdown_value = Memo::new(move || current.get());
    let on_shader_change: Rc<dyn Fn(String)> = Rc::new(move |v: String| {
        current.set(v.clone());
        let shader_name = if v.is_empty() { None } else { Some(v) };
        let _ = cmd_tx.get().send(
            rkp_engine::EngineCommand::SetMaterialShader {
                material_id: mat_id,
                shader_name,
            },
        );
    });

    // Resolve the active shader's param schema and pre-resolve each
    // slider's initial value (from MaterialDef.shader_params, falling
    // back to the shader's declared default). Materializing into a
    // Vec of (name, initial, lo, hi) keeps the rsx! per-iteration
    // closures Copy/'static-friendly without juggling MaterialInfo
    // borrows through the macro expansion.
    #[derive(Debug, Clone, PartialEq)]
    struct ParamRow {
        name: String,
        initial: f32,
        lo: f32,
        hi: f32,
    }
    // Wrap in Rc so the rsx! macro's expanded closure (which is `Fn`,
    // not `FnOnce`) can clone the vec each render rather than moving
    // it. Cheap — param counts are small.
    let param_rows: Rc<Vec<ParamRow>> = Rc::new(if current_shader.is_empty() {
        Vec::new()
    } else {
        shader_list
            .iter()
            .find(|s| s.name == current_shader)
            .map(|s| {
                s.params
                    .iter()
                    .map(|p| {
                        let initial = info
                            .shader_params
                            .get(&p.name)
                            .copied()
                            .unwrap_or(p.default);
                        let (lo, hi) = p.range.unwrap_or((0.0, 1.0));
                        ParamRow {
                            name: p.name.clone(),
                            initial,
                            lo,
                            hi,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:6px;\
                    margin-top:8px;padding-top:8px;\
                    border-top:1px solid #333;",
            div {
                style: "font-size:11px;color:#888;text-transform:uppercase;\
                        letter-spacing:0.5px;",
                "Shader"
            }
            {prop_select(__scope, "Shader", dropdown_value, &options_refs, on_shader_change)}
            for row in (*param_rows).clone() {
                {{
                    let name = row.name.clone();
                    let cmd_field_name = name.clone();
                    let val_signal = Signal::new(row.initial);
                    // 100 ticks across the range, never below 0.001 so
                    // a default (0..1) range lands at 0.01.
                    let step = ((row.hi - row.lo) / 100.0).max(0.001);
                    let on_change: Rc<dyn Fn(f32)> = Rc::new(move |v: f32| {
                        let _ = cmd_tx.get().send(
                            rkp_engine::EngineCommand::SetMaterialShaderParam {
                                material_id: mat_id,
                                name: cmd_field_name.clone(),
                                value: v,
                            },
                        );
                    });
                    prop_slider(__scope, &name, val_signal, row.lo, row.hi, step, on_change)
                }}
            }
        }
    }
}

// ── Model import properties ───────────────────────────────────────────────

#[component]
fn ModelPropertiesSection() -> NodeHandle {
    let store = use_context::<EditorStore>();

    let model_info = Memo::new(move || {
        let sel_path = store.selected_model.get()?;
        store.available_models.get().iter()
            .find(|m| m.source_path == sel_path)
            .cloned()
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;height:100%;",

            // Remount the form on selection change by keying a
            // single-element `for` loop on the source path. Without
            // this, `if let Some(info)` keeps the form's inner
            // Signals alive across selection switches and the form
            // stays pinned to the first-selected model's values —
            // same pattern used by `ComponentSection` in
            // `object_properties.rs` to remount component editors
            // when the selected entity changes.
            for info in model_info.get().into_iter() {
                ModelPropertiesForm {
                    key: info.source_path.clone(),
                    info: info,
                }
            }
        }
    }
}

#[component]
fn ModelPropertiesForm(info: rkp_engine::ModelInfo) -> NodeHandle {
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    let store = use_context::<EditorStore>();
    // Project-relative source path for the header subtitle. Kept as
    // a Memo so it updates when the project root signal changes on
    // project load/switch.
    let display_source_path = Memo::new({
        let sp = info.source_path.clone();
        move || crate::ui::path_display::display_rel_path(&sp, &store.project_dir.get())
    });

    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:0;height:100%;",

            // Header — fields are the initial values from `info`;
            // they never stale out because the parent remounts this
            // whole component on selection change.
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
                        {info.name.clone()}
                    }
                    div {
                        style: "font-size:10px;color:#666;overflow:hidden;\
                                text-overflow:ellipsis;white-space:nowrap;",
                        {|| display_source_path.get()}
                    }
                }
                div {
                    style: "font-size:10px;color:#666;flex-shrink:0;\
                            display:flex;flex-direction:column;align-items:flex-end;",
                    div { {format_size(info.size)} }
                    div { {format_voxel_count(info.voxel_count)} }
                }
            }

            // Import fields
            {model_import_fields(__scope, info.source_path.clone(), info.import_profile.clone(), cmd_tx)}
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
    // Look up the matching live-progress record. Returns None when
    // no import is in flight for this source (button mode) or when
    // the import just started and no events have landed yet.
    let progress = Memo::new(move || {
        let s = src.get();
        store
            .import_progress
            .get()
            .iter()
            .find(|p| p.source_path == s)
            .cloned()
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
                Memo::new(move || voxel_size_str.get()),
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

            // Re-import button — replaced by a live stage + progress
            // bar while the engine is voxelizing this source.
            div {
                style: "margin-top:4px;",
                if is_importing.get() {
                    {render_import_progress(__scope, progress)}
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

/// Render the live import progress indicator: spinner + stage
/// message + a filled bar when the stage reports a `done/total`.
/// Falls back to an indeterminate spinner when the stage is running
/// but total isn't known (e.g. during `load_mesh`).
///
/// Every dynamic value is read inside a `{|| ...}` closure so rinch
/// registers a reactive dependency on `progress` — without that wrap
/// the node's text/width freeze on the value present at first render
/// (which for the common "Starting…" case means the user sees the
/// placeholder for the entire import). Rule 2 in the rinch guide.
fn render_import_progress(
    __scope: &mut rinch::core::dom::RenderScope,
    progress: Memo<Option<rkp_engine::snapshot::ImportProgressInfo>>,
) -> rinch::core::dom::NodeHandle {
    let store = use_context::<EditorStore>();
    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:4px;padding:6px 12px;\
                    background:#2d2d2d;border:1px solid #3c3c3c;border-radius:4px;\
                    color:#bbb;font-size:11px;",
            div {
                style: "display:flex;align-items:center;gap:8px;",
                Loader { r#type: "oval", size: "xs", color: "blue" }
                {|| {
                    // Strip the project-root prefix from any absolute
                    // path the `rkp-import` stage message embeds
                    // (e.g. `Loading mesh: /.../assets/bunny.obj`).
                    // Done at render time because project_dir is a
                    // main-thread signal.
                    let root = store.project_dir.get();
                    progress
                        .get()
                        .as_ref()
                        .map(|p| crate::ui::path_display::relativize_paths_in_text(&p.message, &root))
                        .filter(|m| !m.is_empty())
                        .unwrap_or_else(|| "Starting…".into())
                }}
            }
            div {
                style: {|| {
                    let total = progress.get().as_ref().map(|p| p.total).unwrap_or(0);
                    if total > 0 {
                        "width:100%;height:4px;background:#1e1e1e;\
                         border-radius:2px;overflow:hidden;".to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }},
                div {
                    style: {|| {
                        let (done, total) = progress
                            .get()
                            .as_ref()
                            .map(|p| (p.done, p.total))
                            .unwrap_or((0, 0));
                        let pct = if total > 0 {
                            (done as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
                        } else {
                            0.0
                        };
                        format!(
                            "height:100%;width:{pct:.1}%;background:#4a90e2;\
                             transition:width 120ms ease-out;"
                        )
                    }}
                }
            }
            div {
                style: {|| {
                    let warn_count = progress
                        .get()
                        .as_ref()
                        .map(|p| p.warnings.len())
                        .unwrap_or(0);
                    if warn_count > 0 {
                        "color:#e2a04a;font-size:10px;".to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }},
                {|| {
                    let warn_count = progress
                        .get()
                        .as_ref()
                        .map(|p| p.warnings.len())
                        .unwrap_or(0);
                    if warn_count > 0 {
                        format!(
                            "{warn_count} warning{}",
                            if warn_count == 1 { "" } else { "s" }
                        )
                    } else {
                        String::new()
                    }
                }}
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

/// Pretty-print a shell voxel count for the model header. Zero
/// renders as a blank string so legacy `.rkp` files whose header
/// couldn't be read don't display "0 voxels".
fn format_voxel_count(count: u32) -> String {
    if count == 0 {
        return String::new();
    }
    if count < 1_000 {
        format!("{count} voxels")
    } else if count < 1_000_000 {
        format!("{:.1}K voxels", count as f64 / 1_000.0)
    } else {
        format!("{:.2}M voxels", count as f64 / 1_000_000.0)
    }
}
