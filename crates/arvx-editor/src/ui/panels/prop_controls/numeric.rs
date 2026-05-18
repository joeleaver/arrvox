//! Numeric inputs: bounded slider, drag-to-scrub, and free-form number entry.

use std::rc::Rc;

use rinch::prelude::*;

use super::{LABEL_STYLE, NUMBER_STYLE, Node, ROW_STYLE, Scope, bind};

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
    let (m, oc) = bind(value, on_change);
    prop_slider_memo(__scope, label, m, min, max, step, oc)
}

/// Memo-based variant of [`prop_slider`]. Use when the source of truth
/// lives outside the panel (a store-backed reactive projection, a
/// per-field Memo over an inspector snapshot, etc.).
pub fn prop_slider_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<f32>,
    min: f32,
    max: f32,
    step: f32,
    on_change: Rc<dyn Fn(f32)>,
) -> Node {
    prop_scrub(__scope, label, value, min, max, step, on_change)
}

pub fn prop_slider_f64_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<f64>,
    min: f64,
    max: f64,
    step: f64,
    on_change: Rc<dyn Fn(f64)>,
) -> Node {
    let display = Memo::new(move || value.get() as f32);
    let on_change_wrap: Rc<dyn Fn(f32)> = Rc::new(move |v: f32| on_change(v as f64));
    prop_scrub(
        __scope,
        label,
        display,
        min as f32,
        max as f32,
        step as f32,
        on_change_wrap,
    )
}

// ── Scrub input ──────────────────────────────────────────────────────────

/// Compact number field with drag-to-scrub and click-to-type.
///
/// Shows the current value as text. Drag left/right on the field to scrub.
/// Click without dragging to enter text edit mode. Optional range clamping.
/// A subtle fill bar behind the text shows position within the range.
/// Draggable scrub bar + click-to-edit numeric input.
///
/// `value` is a `Memo<f32>` — the control reads the current value
/// reactively from whatever source the caller wires up (e.g. a store-
/// backed Memo that re-fires when the underlying data changes). User
/// interactions route through `on_change`; the caller is responsible
/// for updating the source so the Memo fires back. That gives "store
/// is source of truth" semantics: external updates from the engine
/// / gizmo / MCP flow through the Memo and the control reflects them
/// without a signal-sync Effect.
pub fn prop_scrub(
    __scope: &mut Scope,
    label: &str,
    value: Memo<f32>,
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
                                // Drag sensitivity is based on the full range, not the step.
                                // Target ~100 pixels to traverse the range; snap to step.
                                // This keeps tiny-range sliders (e.g. 1..6) usable.
                                let range = (max - min).max(1e-6);
                                let raw = drag_start.get() + delta * range / 100.0;
                                let snapped = if step > 0.0 {
                                    (raw / step).round() * step
                                } else {
                                    raw
                                };
                                let new_val = snapped.clamp(min, max);
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
pub fn prop_number_f64_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<f64>,
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
                        on_change(f);
                    }
                },
            }
        }
    }
}

/// Numeric text input for i64 values.
pub fn prop_number_i64_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<i64>,
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
                        on_change(i);
                    }
                },
            }
        }
    }
}

