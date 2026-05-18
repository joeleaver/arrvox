//! Vector and color inputs: per-axis Vec3, RGBA color picker.

use std::rc::Rc;

use rinch::prelude::*;

use super::{LABEL_STYLE, Node, ROW_STYLE, Scope, hex_to_rgba, rgba_to_hex};

// ── Vec3 input ───────────────────────────────────────────────────────────

/// Three color-coded axis inputs (X=red, Y=green, Z=blue).
/// Three color-coded axis inputs (X=red, Y=green, Z=blue) for a
/// `[f32; 3]` value.
///
/// `value` is a `Memo` — the display tracks whatever source the caller
/// points it at (typically a store-backed reactive projection).
/// Writes happen through `on_change`; the Memo reflects the updated
/// source on the next tick. See `prop_scrub` for the rationale.
pub fn prop_vec3(
    __scope: &mut Scope,
    label: &str,
    value: Memo<[f32; 3]>,
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
    value: Memo<[f32; 3]>,
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
                            // Other components come straight from the Memo —
                            // the source of truth — so if something external
                            // updated them mid-drag they pass through intact.
                            let mut vec = value.get();
                            vec[idx] = new_val;
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
                        on_change(vec);
                    }
                },
            }
        }
    }
}

// ── Color picker ─────────────────────────────────────────────────────────

/// Color picker using rinch's ColorInput component. `value` is a
/// `Memo<[f32; 4]>`; the picker tracks its source reactively via
/// `value_fn`, and writes route through `on_change`. Rinch >=
/// `2f4945c3` added a first-run guard on ColorPicker's coordinating
/// effect and a set-if-changed check in the `value_fn` binding, so
/// this can use the ordinary controlled-input pattern without the
/// earlier write-only-bridge workaround.
pub fn prop_color(
    __scope: &mut Scope,
    label: &str,
    value: Memo<[f32; 4]>,
    on_change: Rc<dyn Fn([f32; 4])>,
) -> Node {
    let label = label.to_string();
    // Seed `value:` with the current hex so ColorInput's internal
    // `current_value` starts in sync with what `value_fn:` will
    // report on its first fire — otherwise the value_fn binding
    // effect does `set_if_changed("#XXXX" ≠ "#000000")` on initial
    // run and notifies, triggering a nested flush panic when
    // mounted mid-flush (same shape as the fixed Select bug, but
    // distinct trigger — this one is about the seed, not the
    // binding's first-run guard).
    let initial_hex = untracked(|| rgba_to_hex(value.get()));

    rsx! {
        div {
            style: ROW_STYLE,
            div { style: LABEL_STYLE, {label} }
            div {
                style: "flex:1;min-width:0;",
                ColorInput {
                    value: {initial_hex},
                    value_fn: move || rgba_to_hex(value.get()),
                    format: "hex",
                    alpha: false,
                    onchange: move |v: String| {
                        // Skip if the incoming color is effectively the
                        // current stored one (within hex quantization).
                        // Avoids redundant engine commands when the
                        // picker echoes an external-value update.
                        // Body untracked so the guard's `value.get()`
                        // doesn't subscribe ColorPicker's coordinating
                        // effect to our upstream source.
                        untracked(|| {
                            if let Some(rgba) = hex_to_rgba(&v) {
                                let cur = value.get();
                                let eps = 1.01 / 255.0;
                                if (rgba[0] - cur[0]).abs() < eps
                                    && (rgba[1] - cur[1]).abs() < eps
                                    && (rgba[2] - cur[2]).abs() < eps
                                {
                                    return;
                                }
                                on_change(rgba);
                            }
                        });
                    },
                }
            }
        }
    }
}
