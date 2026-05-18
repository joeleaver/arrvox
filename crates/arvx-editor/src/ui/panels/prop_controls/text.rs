//! Text input, checkbox, and dropdown controls.

use std::rc::Rc;

use rinch::prelude::*;

use super::{CHECKBOX_OFF, CHECKBOX_ON, INPUT_STYLE, LABEL_STYLE, Node, ROW_STYLE, Scope, bind};

// ── Text input ───────────────────────────────────────────────────────────

/// Text input with label. Controlled via Signal for two-way binding.
pub fn prop_text(
    __scope: &mut Scope,
    label: &str,
    value: Signal<String>,
    on_change: Rc<dyn Fn(String)>,
) -> Node {
    let (m, oc) = bind(value, on_change);
    prop_text_memo(__scope, label, m, oc)
}

/// Memo-based variant of [`prop_text`]. Standard "controlled component" —
/// the rendered value is always whatever the Memo currently reports.
pub fn prop_text_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<String>,
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
                oninput: move |v: String| { on_change(v); },
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
    let (m, oc) = bind(value, on_change);
    prop_checkbox_memo(__scope, label, m, oc)
}

/// Memo-based variant of [`prop_checkbox`].
pub fn prop_checkbox_memo(
    __scope: &mut Scope,
    label: &str,
    value: Memo<bool>,
    on_change: Rc<dyn Fn(bool)>,
) -> Node {
    let label = label.to_string();
    rsx! {
        div {
            style: "display:flex;align-items:center;gap:6px;min-height:22px;cursor:pointer;",
            onclick: move || { on_change(!value.get()); },
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
// ── Select / dropdown ────────────────────────────────────────────────────

/// Dropdown select for choosing from a fixed set of options.
///
/// `options` is a list of `(value, label)` pairs. `value` is the internal
/// string, `label` is what's displayed. If label is empty, value is shown.
/// Dropdown select. `value` is a `Memo<String>`; display tracks its
/// source reactively, writes route through `on_change`.
pub fn prop_select(
    __scope: &mut Scope,
    label: &str,
    value: Memo<String>,
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
                    value_fn: move || value.get(),
                    size: "xs",
                    data: data,
                    onchange: move |v: String| {
                        on_change(v);
                    },
                }
            }
        }
    }
}
