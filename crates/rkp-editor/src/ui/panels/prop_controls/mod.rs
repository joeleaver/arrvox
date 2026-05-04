//! Reusable property editor controls — the design system for all property panels.
//!
//! Every control is a plain function taking `&mut RenderScope` → `NodeHandle`.
//! All controls follow the same layout: label (left) + control (right).
//!
//! Display values are `Memo<T>` — controls re-read reactively from whatever
//! source the caller wires up (a store-backed projection, a per-field
//! Memo over an inspector snapshot, etc.). User edits route through
//! `on_change: Rc<dyn Fn(T)>`; the caller updates the source so the
//! Memo refires. This avoids signal-sync Effects and gives "store is
//! source of truth" semantics, so external updates (engine writes,
//! gameplay, MCP, undo) flow into the UI without remounting forms.
//!
//! # Usage
//!
//! ```ignore
//! use super::prop_controls::*;
//!
//! let value = Memo::new(move || store.foo.get());
//! let on_change = Rc::new(|v: f32| { /* send command */ });
//! prop_slider(__scope, "Roughness", value, 0.0, 1.0, 0.01, on_change);
//! ```

use std::rc::Rc;

use rinch::prelude::*;

mod numeric;
mod text;
mod vec_color;

pub use numeric::*;
pub use text::*;
pub use vec_color::*;

pub(super) type Scope = rinch::core::dom::RenderScope;
pub(super) type Node = rinch::core::dom::NodeHandle;

/// Wrap a `Signal<T>` for use with the Memo-based prop_* controls.
///
/// Returns a `(Memo<T>, on_change_wrapper)` pair. The Memo reads the
/// Signal reactively for display; the wrapper writes the Signal AND
/// calls the user's `on_change`. Use this at call sites where a local
/// panel Signal is still the source of truth (env/asset panels).
///
/// For panels reading directly from `EditorStore`, build the Memo
/// against the store and skip this helper — the on_change just sends a
/// command to the engine and the store-backed Memo refires when the
/// answer comes back.
pub fn bind<T>(
    s: Signal<T>,
    on_change: Rc<dyn Fn(T)>,
) -> (Memo<T>, Rc<dyn Fn(T)>)
where
    T: Clone + PartialEq + 'static,
{
    let memo = Memo::new(move || s.get());
    let oc: Rc<dyn Fn(T)> = Rc::new(move |v: T| {
        s.set(v.clone());
        on_change(v);
    });
    (memo, oc)
}

// ── Style constants ──────────────────────────────────────────────────────

pub(super) const LABEL_STYLE: &str = "width:72px;flex-shrink:0;font-size:11px;color:#999;\
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
pub(super) const ROW_STYLE: &str = "display:flex;align-items:center;gap:6px;min-height:22px;";
pub(super) const INPUT_STYLE: &str = "flex:1;min-width:0;background:#1e1e1e;border:1px solid #3c3c3c;\
                           border-radius:3px;color:#ccc;font-size:11px;padding:3px 6px;\
                           outline:none;font-family:inherit;";
pub(super) const NUMBER_STYLE: &str = "flex:1;min-width:0;background:#1e1e1e;border:1px solid #3c3c3c;\
                            border-radius:3px;color:#ccc;font-size:11px;padding:3px 6px;\
                            outline:none;font-family:monospace;";
pub(super) const CHECKBOX_ON: &str = "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                           background:#4fc3f7;border:1px solid #4fc3f7;flex-shrink:0;\
                           display:flex;align-items:center;justify-content:center;";
pub(super) const CHECKBOX_OFF: &str = "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                            background:#1e1e1e;border:1px solid #3c3c3c;flex-shrink:0;";


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

pub(super) fn rgba_to_hex(c: [f32; 4]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}

pub(super) fn hex_to_rgba(hex: &str) -> Option<[f32; 4]> {
    if hex.len() != 7 || !hex.starts_with('#') {
        return None;
    }
    let r = u8::from_str_radix(&hex[1..3], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&hex[3..5], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&hex[5..7], 16).ok()? as f32 / 255.0;
    Some([r, g, b, 1.0])
}
