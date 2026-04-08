//! Field editors — type-specific UI widgets for component field editing.
//!
//! Each editor is a plain function (not a #[component]) that takes a RenderScope
//! and returns a NodeHandle. This avoids the component prop system's limitations
//! with closure types and allows direct closure passing.

use std::rc::Rc;

use rinch::prelude::*;

use rkp_engine::inspector::*;

/// Render the appropriate editor for a field based on its type.
pub fn field_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    match field.field_type {
        FieldType::Float => float_editor(__scope, field, on_change),
        FieldType::Int => int_editor(__scope, field, on_change),
        FieldType::Bool => bool_editor(__scope, field, on_change),
        FieldType::String => string_editor(__scope, field, on_change),
        FieldType::Vec3 => vec3_editor(__scope, field, on_change),
        FieldType::Color => color_editor(__scope, field, on_change),
        _ => fallback_editor(__scope, field),
    }
}

fn float_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Float(v) => *v, _ => 0.0 };
    let has_range = field.range.is_some();
    let (min, max) = field.range.unwrap_or((f64::MIN, f64::MAX));
    let current = Signal::new(val);
    let oc = on_change;

    if has_range {
        // Slider for ranged floats.
        let step = (max - min) / 200.0;
        rsx! {
            div { style: "display:flex;align-items:center;gap:4px;",
                input {
                    r#type: "range",
                    style: "flex:1;height:4px;accent-color:#007acc;",
                    min: {|| format!("{min}")},
                    max: {|| format!("{max}")},
                    step: {|| format!("{step}")},
                    value: move || format!("{:.3}", current.get()),
                    oninput: {
                        let oc = oc.clone();
                        move |v: String| {
                            if let Ok(fv) = v.parse::<f64>() {
                                current.set(fv);
                                oc(FieldValue::Float(fv));
                            }
                        }
                    },
                }
                span { style: "width:45px;text-align:right;font-size:10px;color:#888;font-family:monospace;",
                    {move || format!("{:.2}", current.get())}
                }
            }
        }
    } else {
        rsx! {
            input {
                r#type: "number",
                style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                        color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
                value: move || format!("{:.3}", current.get()),
                oninput: {
                    let oc = oc.clone();
                    move |v: String| {
                        if let Ok(fv) = v.parse::<f64>() {
                            current.set(fv);
                            oc(FieldValue::Float(fv));
                        }
                    }
                },
            }
        }
    }
}

fn int_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Int(v) => *v, _ => 0 };
    let current = Signal::new(val);
    let oc = on_change;

    rsx! {
        input {
            r#type: "number",
            style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                    color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
            value: move || format!("{}", current.get()),
            oninput: {
                let oc = oc.clone();
                move |v: String| {
                    if let Ok(iv) = v.parse::<i64>() {
                        current.set(iv);
                        oc(FieldValue::Int(iv));
                    }
                }
            },
        }
    }
}

fn bool_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Bool(v) => *v, _ => false };
    let checked = Signal::new(val);
    let oc = on_change;

    rsx! {
        div {
            style: {move || {
                if checked.get() {
                    "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                     background:#007acc;border:1px solid #007acc;flex-shrink:0;"
                } else {
                    "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                     background:#1e1e1e;border:1px solid #3c3c3c;flex-shrink:0;"
                }
            }},
            onclick: {
                let oc = oc.clone();
                move || {
                    let new_val = !checked.get();
                    checked.set(new_val);
                    oc(FieldValue::Bool(new_val));
                }
            },
        }
    }
}

fn string_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::String(v) => v.clone(), _ => String::new() };
    let oc = on_change;

    rsx! {
        input {
            r#type: "text",
            style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                    color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;",
            value: {|| val.clone()},
            oninput: {
                let oc = oc.clone();
                move |v: String| {
                    oc(FieldValue::String(v));
                }
            },
        }
    }
}

fn vec3_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Vec3(v) => *v, _ => [0.0; 3] };
    let current = Signal::new(val);

    let axis_data: [(&str, &str, usize); 3] = [
        ("#e06060", "X", 0),
        ("#60e060", "Y", 1),
        ("#6060e0", "Z", 2),
    ];

    // Pre-build the three axis inputs.
    let x_oc = on_change.clone();
    let y_oc = on_change.clone();
    let z_oc = on_change;

    let x_input = axis_number_input(__scope, "#e06060", "X", current, 0, x_oc);
    let y_input = axis_number_input(__scope, "#60e060", "Y", current, 1, y_oc);
    let z_input = axis_number_input(__scope, "#6060e0", "Z", current, 2, z_oc);

    rsx! {
        div { style: "display:flex;gap:2px;",
            {x_input}
            {y_input}
            {z_input}
        }
    }
}

fn axis_number_input(
    __scope: &mut rinch::core::dom::RenderScope,
    color: &'static str,
    label: &'static str,
    current: Signal<[f32; 3]>,
    idx: usize,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let oc = on_change;
    rsx! {
        div { style: "flex:1;display:flex;align-items:center;",
            span {
                style: {|| format!("color:{color};font-size:10px;font-weight:600;margin-right:2px;")},
                {label}
            }
            input {
                r#type: "number",
                style: "width:100%;background:#1e1e1e;border:1px solid #3c3c3c;\
                        color:#ccc;padding:2px 4px;font-size:11px;border-radius:2px;outline:none;",
                value: move || format!("{:.2}", current.get()[idx]),
                oninput: {
                    let oc = oc.clone();
                    move |v: String| {
                        if let Ok(fv) = v.parse::<f32>() {
                            let mut vec = current.get();
                            vec[idx] = fv;
                            current.set(vec);
                            oc(FieldValue::Vec3(vec));
                        }
                    }
                },
            }
        }
    }
}

fn color_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
    on_change: Rc<dyn Fn(FieldValue)>,
) -> rinch::core::dom::NodeHandle {
    let val = match &field.value { FieldValue::Color(v) => *v, _ => [1.0, 1.0, 1.0, 1.0] };
    let hex = Signal::new(format!("#{:02x}{:02x}{:02x}",
        (val[0] * 255.0) as u8, (val[1] * 255.0) as u8, (val[2] * 255.0) as u8));
    let oc = on_change;

    rsx! {
        div { style: "display:flex;align-items:center;gap:6px;",
            // Color swatch
            div {
                style: {move || format!(
                    "width:22px;height:22px;border-radius:3px;border:1px solid #3c3c3c;\
                     background:{};flex-shrink:0;",
                    hex.get()
                )},
            }
            // Hex input
            input {
                r#type: "color",
                style: "width:24px;height:24px;border:none;padding:0;cursor:pointer;\
                        background:transparent;",
                value: move || hex.get(),
                oninput: {
                    let oc = oc.clone();
                    move |v: String| {
                        hex.set(v.clone());
                        // Parse hex → [f32; 4]
                        if v.len() == 7 && v.starts_with('#') {
                            if let (Ok(r), Ok(g), Ok(b)) = (
                                u8::from_str_radix(&v[1..3], 16),
                                u8::from_str_radix(&v[3..5], 16),
                                u8::from_str_radix(&v[5..7], 16),
                            ) {
                                oc(FieldValue::Color([
                                    r as f32 / 255.0,
                                    g as f32 / 255.0,
                                    b as f32 / 255.0,
                                    1.0,
                                ]));
                            }
                        }
                    }
                },
            }
        }
    }
}

fn fallback_editor(
    __scope: &mut rinch::core::dom::RenderScope,
    field: &FieldSnapshot,
) -> rinch::core::dom::NodeHandle {
    let display = field.value.to_string();
    rsx! {
        div {
            style: "color:#888;font-size:11px;font-family:monospace;\
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
            {display}
        }
    }
}
