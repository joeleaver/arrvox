//! Reusable property editor controls — the design system for all property panels.
//!
//! Every control is a plain function taking `&mut RenderScope` → `NodeHandle`.
//! All controls follow the same layout: label (left) + control (right).
//! All values are reactive via `Signal`. Changes are reported via `Rc<dyn Fn(T)>`.
//!
//! # Usage
//!
//! ```ignore
//! use super::prop_controls::*;
//!
//! let on_change = Rc::new(|v: f32| { /* send command */ });
//! prop_slider(__scope, "Roughness", Signal::new(0.5), 0.0, 1.0, 0.01, on_change);
//! ```

use std::rc::Rc;

use rinch::prelude::*;

type Scope = rinch::core::dom::RenderScope;
type Node = rinch::core::dom::NodeHandle;

// ── Style constants ──────────────────────────────────────────────────────

const LABEL_STYLE: &str = "width:72px;flex-shrink:0;font-size:11px;color:#999;\
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
const ROW_STYLE: &str = "display:flex;align-items:center;gap:6px;min-height:22px;";
const INPUT_STYLE: &str = "flex:1;min-width:0;background:#1e1e1e;border:1px solid #3c3c3c;\
                           border-radius:3px;color:#ccc;font-size:11px;padding:3px 6px;\
                           outline:none;font-family:inherit;";
const NUMBER_STYLE: &str = "flex:1;min-width:0;background:#1e1e1e;border:1px solid #3c3c3c;\
                            border-radius:3px;color:#ccc;font-size:11px;padding:3px 6px;\
                            outline:none;font-family:monospace;";
const VALUE_STYLE: &str = "width:40px;text-align:right;font-size:10px;color:#777;\
                           font-family:monospace;flex-shrink:0;";
const CHECKBOX_ON: &str = "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                           background:#4fc3f7;border:1px solid #4fc3f7;flex-shrink:0;\
                           display:flex;align-items:center;justify-content:center;";
const CHECKBOX_OFF: &str = "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                            background:#1e1e1e;border:1px solid #3c3c3c;flex-shrink:0;";

// ── Slider ───────────────────────────────────────────────────────────────

/// Horizontal slider with label and value readout.
///
/// Uses the rinch Slider component for proper drag interaction and visual feedback.
/// Good for bounded f32 values (roughness, opacity, rotation, etc.).
pub fn prop_slider(
    __scope: &mut Scope,
    label: &str,
    value: Signal<f32>,
    min: f32,
    max: f32,
    step: f32,
    on_change: Rc<dyn Fn(f32)>,
) -> Node {
    let label = label.to_string();
    // Bridge f32 → f64 for the rinch Slider component.
    let value_f64 = Signal::new(value.get() as f64);

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;min-width:0;",
                Slider {
                    min: min as f64,
                    max: max as f64,
                    step: step as f64,
                    size: "sm",
                    color: "#4fc3f7",
                    value_signal: value_f64,
                    onchange: move |v: f64| {
                        let f = v as f32;
                        value.set(f);
                        value_f64.set(v);
                        on_change(f);
                    },
                }
            }
            div {
                style: VALUE_STYLE,
                {move || format!("{:.2}", value.get())}
            }
        }
    }
}

/// Slider that operates on f64 (for ECS inspector fields).
pub fn prop_slider_f64(
    __scope: &mut Scope,
    label: &str,
    value: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    on_change: Rc<dyn Fn(f64)>,
) -> Node {
    let label = label.to_string();

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;min-width:0;",
                Slider {
                    min: min,
                    max: max,
                    step: step,
                    size: "sm",
                    color: "#4fc3f7",
                    value_signal: value,
                    onchange: move |v: f64| {
                        value.set(v);
                        on_change(v);
                    },
                }
            }
            div {
                style: VALUE_STYLE,
                {move || format!("{:.2}", value.get())}
            }
        }
    }
}

// ── Scrub input ──────────────────────────────────────────────────────────

/// Compact number field with drag-to-scrub and click-to-type.
///
/// Shows the current value as text. Drag left/right on the field to scrub.
/// Click without dragging to enter text edit mode. Optional range clamping.
/// A subtle fill bar behind the text shows position within the range.
pub fn prop_scrub(
    __scope: &mut Scope,
    label: &str,
    value: Signal<f32>,
    min: f32,
    max: f32,
    step: f32,
    on_change: Rc<dyn Fn(f32)>,
) -> Node {
    let label = label.to_string();
    let editing = Signal::new(false);
    let edit_text = Signal::new(String::new());
    let drag_start = Signal::new(0.0f32);
    let drag_started_pos = Signal::new(0.0f32);
    let did_drag = Signal::new(false);
    let oc_drag: Signal<Rc<dyn Fn(f32)>> = Signal::new(on_change.clone());

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }

            if editing.get() {
                // Text edit mode.
                // edit_text holds the user's raw typed string so the input doesn't
                // get overwritten with a reformatted number on every keystroke.
                input {
                    r#type: "number",
                    style: "flex:1;min-width:0;background:#1a1a2e;border:1px solid #4fc3f7;\
                            border-radius:3px;color:#ccc;font-size:11px;padding:3px 6px;\
                            outline:none;font-family:monospace;",
                    value: {move || edit_text.get()},
                    oninput: move |v: String| {
                        edit_text.set(v.clone());
                        if let Ok(f) = v.parse::<f32>() {
                            let clamped = f.clamp(min, max);
                            value.set(clamped);
                            (oc_drag.get())(clamped);
                        }
                    },
                    onblur: move || { editing.set(false); },
                }
            }

            if !editing.get() {
                // Scrub mode: drag to change, click to edit.
                div {
                    style: {move || {
                        let pct = if max > min {
                            ((value.get() - min) / (max - min)).clamp(0.0, 1.0) * 100.0
                        } else {
                            0.0
                        };
                        format!(
                            "flex:1;min-width:0;position:relative;height:20px;border-radius:3px;\
                             background:#1e1e1e;border:1px solid #3c3c3c;cursor:ew-resize;\
                             overflow:hidden;display:flex;align-items:center;\
                             background:linear-gradient(to right, #2a3a4a {pct:.0}%, #1e1e1e {pct:.0}%);"
                        )
                    }},
                    onclick: move || {
                        let ctx = rinch::core::get_click_context();
                        let start_x = ctx.mouse_x;
                        drag_start.set(value.get());
                        drag_started_pos.set(start_x);
                        did_drag.set(false);

                        Drag::absolute()
                            .on_move(move |mx, _| {
                                let delta = mx - drag_started_pos.get();
                                if delta.abs() > 2.0 {
                                    did_drag.set(true);
                                }
                                let new_val = (drag_start.get() + delta * step).clamp(min, max);
                                value.set(new_val);
                                (oc_drag.get())(new_val);
                            })
                            .on_end(move |_, _| {
                                if !did_drag.get() {
                                    edit_text.set(format!("{:.3}", value.get()));
                                    editing.set(true);
                                }
                            })
                            .start();
                    },
                    // Value text centered on top of the fill bar.
                    span {
                        style: "position:relative;z-index:1;padding:0 6px;\
                                font-size:11px;font-family:monospace;color:#ccc;\
                                user-select:none;white-space:nowrap;",
                        {move || format!("{:.2}", value.get())}
                    }
                }
            }
        }
    }
}

// ── Number input ─────────────────────────────────────────────────────────

/// Numeric text input for unbounded f64 values. Label is drag-to-scrub.
pub fn prop_number_f64(
    __scope: &mut Scope,
    label: &str,
    value: Signal<f64>,
    on_change: Rc<dyn Fn(f64)>,
) -> Node {
    let label = label.to_string();
    let drag_start = Signal::new(0.0f64);
    let oc_drag: Signal<Rc<dyn Fn(f64)>> = Signal::new(on_change.clone());

    rsx! {
        div {
            style: ROW_STYLE,
            div {
                style: "width:72px;flex-shrink:0;font-size:11px;color:#999;\
                        cursor:ew-resize;user-select:none;\
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                onclick: move || {
                    let ctx = rinch::core::get_click_context();
                    let start_x = ctx.mouse_x;
                    drag_start.set(value.get());

                    Drag::absolute()
                        .on_move(move |mx, _| {
                            let delta = (mx - start_x) as f64;
                            let new_val = drag_start.get() + delta * 0.02;
                            value.set(new_val);
                            (oc_drag.get())(new_val);
                        })
                        .start();
                },
                {label}
            }
            input {
                r#type: "number",
                style: NUMBER_STYLE,
                value: {move || format!("{:.3}", value.get())},
                oninput: move |v: String| {
                    if let Ok(f) = v.parse::<f64>() {
                        value.set(f);
                        on_change(f);
                    }
                },
            }
        }
    }
}

/// Numeric text input for i64 values.
pub fn prop_number_i64(
    __scope: &mut Scope,
    label: &str,
    value: Signal<i64>,
    on_change: Rc<dyn Fn(i64)>,
) -> Node {
    let label = label.to_string();
    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            input {
                r#type: "number",
                style: NUMBER_STYLE,
                value: {move || value.get().to_string()},
                oninput: move |v: String| {
                    if let Ok(i) = v.parse::<i64>() {
                        value.set(i);
                        on_change(i);
                    }
                },
            }
        }
    }
}

// ── Text input ───────────────────────────────────────────────────────────

/// Text input with label. Controlled via Signal for two-way binding.
pub fn prop_text(
    __scope: &mut Scope,
    label: &str,
    value: Signal<String>,
    on_change: Rc<dyn Fn(String)>,
) -> Node {
    let label = label.to_string();
    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            input {
                r#type: "text",
                style: INPUT_STYLE,
                value: {move || value.get()},
                oninput: move |v: String| {
                    value.set(v.clone());
                    on_change(v);
                },
            }
        }
    }
}

// ── Checkbox ─────────────────────────────────────────────────────────────

/// Toggle checkbox with label.
pub fn prop_checkbox(
    __scope: &mut Scope,
    label: &str,
    value: Signal<bool>,
    on_change: Rc<dyn Fn(bool)>,
) -> Node {
    let label = label.to_string();
    rsx! {
        div {
            style: "display:flex;align-items:center;gap:6px;min-height:22px;cursor:pointer;",
            onclick: move || {
                let new_val = !value.get();
                value.set(new_val);
                on_change(new_val);
            },
            div {
                style: {move || if value.get() { CHECKBOX_ON } else { CHECKBOX_OFF }},
                // Checkmark icon (only when checked)
                if value.get() {
                    span {
                        style: "color:#fff;font-size:11px;line-height:1;",
                        {"\u{2713}"}
                    }
                }
            }
            div {
                style: "font-size:11px;color:#bbb;",
                {label}
            }
        }
    }
}

// ── Vec3 input ───────────────────────────────────────────────────────────

/// Three color-coded axis inputs (X=red, Y=green, Z=blue).
pub fn prop_vec3(
    __scope: &mut Scope,
    label: &str,
    value: Signal<[f32; 3]>,
    on_change: Rc<dyn Fn([f32; 3])>,
) -> Node {
    let label = label.to_string();
    let oc0 = on_change.clone();
    let oc1 = on_change.clone();
    let oc2 = on_change;

    let x_input = vec3_axis(__scope, "#e06060", "X", value, 0, oc0);
    let y_input = vec3_axis(__scope, "#60e060", "Y", value, 1, oc1);
    let z_input = vec3_axis(__scope, "#6060e0", "Z", value, 2, oc2);

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;display:flex;gap:2px;",
                {x_input}
                {y_input}
                {z_input}
            }
        }
    }
}

fn vec3_axis(
    __scope: &mut Scope,
    color: &'static str,
    label: &'static str,
    value: Signal<[f32; 3]>,
    idx: usize,
    on_change: Rc<dyn Fn([f32; 3])>,
) -> Node {
    // Drag state: the value at drag start + Rc clone for the drag closure.
    let drag_start_val = Signal::new(0.0f32);
    // Wrap callback in Signal so it's Copy for use in nested closures.
    let oc_drag: Signal<Rc<dyn Fn([f32; 3])>> = Signal::new(on_change.clone());

    rsx! {
        div {
            style: "flex:1;display:flex;align-items:center;min-width:0;",
            // Draggable axis label — click and drag left/right to scrub value.
            span {
                style: {|| format!(
                    "color:{color};font-size:10px;font-weight:700;width:12px;\
                     flex-shrink:0;text-align:center;cursor:ew-resize;user-select:none;"
                )},
                onclick: move || {
                    let ctx = rinch::core::get_click_context();
                    let start_x = ctx.mouse_x;
                    drag_start_val.set(value.get()[idx]);

                    Drag::absolute()
                        .on_move(move |mouse_x, _mouse_y| {
                            let delta = mouse_x - start_x;
                            let new_val = drag_start_val.get() + delta * 0.02;
                            let mut vec = value.get();
                            vec[idx] = new_val;
                            value.set(vec);
                            (oc_drag.get())(vec);
                        })
                        .start();
                },
                {label}
            }
            input {
                r#type: "number",
                style: "flex:1;min-width:0;background:#1e1e1e;border:1px solid #3c3c3c;\
                        border-radius:3px;color:#ccc;font-size:11px;padding:2px 4px;\
                        outline:none;font-family:monospace;",
                value: {move || format!("{:.2}", value.get()[idx])},
                oninput: move |v: String| {
                    if let Ok(f) = v.parse::<f32>() {
                        let mut vec = value.get();
                        vec[idx] = f;
                        value.set(vec);
                        on_change(vec);
                    }
                },
            }
        }
    }
}

// ── Color picker ─────────────────────────────────────────────────────────

/// Color picker using rinch's ColorInput component.
/// Works with [f32; 4] RGBA (alpha always 1.0 for now).
pub fn prop_color(
    __scope: &mut Scope,
    label: &str,
    value: Signal<[f32; 4]>,
    on_change: Rc<dyn Fn([f32; 4])>,
) -> Node {
    let label = label.to_string();
    // One-shot seed read, untracked so the caller's render effect
    // doesn't subscribe to `value` (a subscription turns `value.set`
    // inside onchange into a synchronous parent re-render, re-entering
    // an in-flight effect at rinch effect.rs:144 → RefCell panic).
    //
    // NOTE: deliberately not using `value_fn` here. ColorInput's
    // internal binding effect (color_input.rs:221 on `main`) races
    // with the user-edit onchange and clobbers the new value before
    // external state can propagate — see rinch issue
    // https://github.com/joeleaver/rinch/issues/22. The caller's
    // `value` Signal is effectively a write-only bridge anyway (no
    // one in the editor mutates it externally after mount; fresh
    // engine snapshots remount the param field with a new Signal),
    // so losing reactive external → picker sync doesn't affect us.
    // Revisit this once the rinch fix lands.
    let initial_hex = untracked(|| rgba_to_hex(value.get()));

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;min-width:0;",
                ColorInput {
                    value: {initial_hex},
                    format: "hex",
                    alpha: false,
                    onchange: move |v: String| {
                        // Whole body runs untracked. ColorPicker invokes
                        // onchange from inside its own "coordinating"
                        // effect (color_picker.rs:365), so any naked
                        // `value.get()` here — including the guard
                        // below — would subscribe *that* effect to our
                        // Signal. `value.set(rgba)` a few lines later
                        // would then notify and re-enter the still-
                        // borrowed coordinating effect at effect.rs:144.
                        untracked(|| {
                            if let Some(rgba) = hex_to_rgba(&v) {
                                if rgba_to_hex(rgba) == rgba_to_hex(value.get()) {
                                    return;
                                }
                                value.set(rgba);
                                on_change(rgba);
                            }
                        });
                    },
                }
            }
        }
    }
}

// ── Select / dropdown ────────────────────────────────────────────────────

/// Dropdown select for choosing from a fixed set of options.
///
/// `options` is a list of `(value, label)` pairs. `value` is the internal
/// string, `label` is what's displayed. If label is empty, value is shown.
pub fn prop_select(
    __scope: &mut Scope,
    label: &str,
    value: Signal<String>,
    options: &[(&str, &str)],
    on_change: Rc<dyn Fn(String)>,
) -> Node {
    let label = label.to_string();
    let data: Vec<SelectOption> = options
        .iter()
        .map(|(v, l)| SelectOption::new(*v, *l))
        .collect();

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;min-width:0;",
                Select {
                    value: {value.get()},
                    size: "xs",
                    data: data,
                    onchange: move |v: String| {
                        value.set(v.clone());
                        on_change(v);
                    },
                }
            }
        }
    }
}

// ── Read-only label ──────────────────────────────────────────────────────

/// Read-only value display with label. For transient/computed fields.
pub fn prop_label(
    __scope: &mut Scope,
    label: &str,
    value: Signal<String>,
) -> Node {
    let label = label.to_string();
    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;font-size:11px;color:#777;font-family:monospace;\
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
                {move || value.get()}
            }
        }
    }
}

// ── Section header ───────────────────────────────────────────────────────

/// Collapsible section header with chevron toggle.
///
/// Returns just the header — caller should conditionally render content
/// based on the `collapsed` signal:
/// ```ignore
/// let collapsed = Signal::new(false);
/// prop_section_header(__scope, "Transform", collapsed, false, None);
/// if !collapsed.get() { /* render fields */ }
/// ```
pub fn prop_section_header(
    __scope: &mut Scope,
    title: &str,
    collapsed: Signal<bool>,
    on_remove: Option<Rc<dyn Fn()>>,
) -> Node {
    let title = title.to_string();
    let removable = on_remove.is_some();
    let on_remove = Signal::new(on_remove);

    rsx! {
        div {
            style: "display:flex;align-items:center;padding:6px 12px;cursor:pointer;\
                    background:#2a2a2a;gap:6px;border-bottom:1px solid #3c3c3c;",
            onclick: move || collapsed.update(|c| *c = !*c),

            // Chevron
            span {
                style: {move || {
                    if collapsed.get() {
                        "font-size:10px;color:#666;transform:rotate(-90deg);\
                         transition:transform 0.15s;display:inline-block;"
                    } else {
                        "font-size:10px;color:#666;transition:transform 0.15s;\
                         display:inline-block;"
                    }
                }},
                {"\u{25BC}"}
            }

            // Title
            span {
                style: "flex:1;font-size:11px;font-weight:600;color:#bbb;\
                        text-transform:uppercase;letter-spacing:0.3px;",
                {title}
            }

            // Remove button
            if removable {
                div {
                    style: "cursor:pointer;color:#666;width:14px;height:14px;\
                            display:flex;align-items:center;justify-content:center;\
                            border-radius:2px;flex-shrink:0;",
                    onclick: move || {
                        if let Some(ref cb) = on_remove.get() {
                            cb();
                        }
                    },
                    {"\u{2715}"}
                }
            }
        }
    }
}

// ── Action button ────────────────────────────────────────────────────────

/// Styled action button with optional icon text.
pub fn prop_button(
    __scope: &mut Scope,
    label: &str,
    accent: &str,
    on_click: Rc<dyn Fn()>,
) -> Node {
    let label = label.to_string();
    let style = Signal::new(format!(
        "display:flex;align-items:center;justify-content:center;gap:4px;\
         padding:6px 12px;background:{accent};border:1px solid {accent};\
         border-radius:4px;cursor:pointer;color:#ddd;font-size:11px;\
         font-weight:500;"
    ));
    rsx! {
        div {
            style: {move || style.get()},
            onclick: move || on_click(),
            {label}
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn rgba_to_hex(c: [f32; 4]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}

fn hex_to_rgba(hex: &str) -> Option<[f32; 4]> {
    if hex.len() != 7 || !hex.starts_with('#') {
        return None;
    }
    let r = u8::from_str_radix(&hex[1..3], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&hex[3..5], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&hex[5..7], 16).ok()? as f32 / 255.0;
    Some([r, g, b, 1.0])
}
