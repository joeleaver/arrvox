//! Material asset panel: PBR fields, color, emission, opacity, IOR,
//! roughness, metallic, plus the user-shader picker + parameter UI.

#![allow(unused_variables)]

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::CommandSender;
use crate::ui::store::EditorStore;
use crate::ui::panels::prop_controls::*;

use super::CmdSignal;

#[component]
pub(super) fn MaterialPropertiesSection() -> NodeHandle {
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
                                        arvx_engine::EngineCommand::CreateMaterial {
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
    info: arvx_engine::material_library::MaterialInfo,
) -> NodeHandle {
    let cmd_tx: CmdSignal = Signal::new(use_context::<CommandSender>().0);
    material_fields(__scope, info.id, info, cmd_tx)
}

fn asset_header(
    __scope: &mut rinch::core::dom::RenderScope,
    mat_info: Memo<Option<arvx_engine::material_library::MaterialInfo>>,
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
                            arvx_engine::EngineCommand::DeleteMaterial { material_id: mat_id },
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
    info: arvx_engine::material_library::MaterialInfo,
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
            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::UpdateMaterialField {
                material_id: mat_id,
                field: field.into(),
                value: v.to_string(),
            });
        })
    };
    let color_cb = move |field: &'static str| -> Rc<dyn Fn([f32; 4])> {
        Rc::new(move |v: [f32; 4]| {
            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::UpdateMaterialField {
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
            let _ = cmd_tx.get().send(arvx_engine::EngineCommand::UpdateMaterialField {
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
    info: arvx_engine::material_library::MaterialInfo,
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
            arvx_engine::EngineCommand::SetMaterialShader {
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
                            arvx_engine::EngineCommand::SetMaterialShaderParam {
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

