//! Viewport toolbars — fixed header bar + floating gizmo overlay.
//!
//! The fixed bar holds viewport-wide controls (play/stop, camera selectors, etc.)
//! that are always visible. The floating overlay holds mode-specific tools
//! (translate/rotate/scale) that only apply during editing.

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use rkp_engine::gizmo::GizmoMode;

use super::prop_controls::prop_slider;
use crate::CommandSender;
use crate::ui::store::EditorStore;

// ── Fixed header bar ────────────────────────────────────────────────────

/// Fixed toolbar row above the render surface — viewport-wide controls.
#[component]
pub fn ViewportHeaderBar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "height:32px;display:flex;align-items:center;padding:0 6px;\
                    background:#252526;border-bottom:1px solid #3c3c3c;flex-shrink:0;",

            // Play/Stop (left-aligned)
            {play_button(__scope, store, cmd)}
        }
    }
}

fn play_button(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if store.play_mode.get() {
                    "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                     background:#5a2d2d;border:1px solid #8b3a3a;border-radius:4px;\
                     cursor:pointer;color:#ef5350;font-size:11px;font-weight:600;"
                } else {
                    "display:flex;align-items:center;gap:4px;padding:2px 10px;\
                     background:#2d5a2d;border:1px solid #3c7c3c;border-radius:4px;\
                     cursor:pointer;color:#4caf50;font-size:11px;font-weight:600;"
                }
            }},
            title: {move || if store.play_mode.get() { "Stop (F5)" } else { "Play (F5)" }},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    if store.play_mode.get() {
                        let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStop);
                    } else {
                        let _ = cmd.0.send(rkp_engine::EngineCommand::PlayStart);
                    }
                }
            },
            span {
                style: "width:14px;height:14px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                if store.play_mode.get() {
                    {render_tabler_icon(__scope, TablerIcon::PlayerStop, TablerIconStyle::Filled)}
                }
                if !store.play_mode.get() {
                    {render_tabler_icon(__scope, TablerIcon::PlayerPlay, TablerIconStyle::Filled)}
                }
            }
            if store.play_mode.get() {
                {"Stop"}
            }
            if !store.play_mode.get() {
                {"Play"}
            }
        }
    }
}

// ── Floating gizmo overlay ──────────────────────────────────────────────

/// Floating pill overlay for mode-specific editing tools (translate/rotate/scale).
#[component]
pub fn EditModeToolbar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "position:absolute;top:8px;left:8px;z-index:20;\
                    display:flex;flex-direction:column;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;backdrop-filter:blur(8px);",

                {gizmo_button(
                    __scope,
                    TablerIcon::ArrowsMove,
                    "Translate (W)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Translate),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Translate);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Translate,
                            });
                        }
                    },
                )}
                {gizmo_button(
                    __scope,
                    TablerIcon::Rotate,
                    "Rotate (E)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Rotate),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Rotate);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Rotate,
                            });
                        }
                    },
                )}
                {gizmo_button(
                    __scope,
                    TablerIcon::Resize,
                    "Scale (R)",
                    Memo::new(move || store.gizmo_mode.get() == GizmoMode::Scale),
                    {
                        let cmd = cmd.clone();
                        move || {
                            store.gizmo_mode.set(GizmoMode::Scale);
                            let _ = cmd.0.send(rkp_engine::EngineCommand::SetGizmoMode {
                                mode: GizmoMode::Scale,
                            });
                        }
                    },
                )}
        }
    }
}

fn gizmo_button(
    __scope: &mut rinch::core::dom::RenderScope,
    icon: TablerIcon,
    tooltip: &str,
    is_active: Memo<bool>,
    on_click: impl Fn() + Clone + 'static,
) -> rinch::core::dom::NodeHandle {
    let tooltip = tooltip.to_string();
    rsx! {
        div {
            style: {move || {
                if is_active.get() {
                    "width:28px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     background:#4fc3f7;color:#1e1e1e;"
                } else {
                    "width:28px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     color:#999;background:transparent;"
                }
            }},
            title: tooltip,
            onclick: move || on_click(),
            span {
                style: "width:18px;height:18px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                {render_tabler_icon(__scope, icon, TablerIconStyle::Outline)}
            }
        }
    }
}

// ── Floating paint toolbar ──────────────────────────────────────────────

/// Floating pill overlay for paint-tool state, upper-right.
///
/// * Row 1 — always visible: Paint on/off toggle button (mirrors `P` key).
/// * Row 2 — when paint is active: mode toggle (Material / Color / Erase)
///   and, in Color mode, the color picker.
/// * Row 3 — when paint is active: radius / strength / falloff sliders.
///
/// Radius changes push `SetPaintActive` so the engine's cached brush
/// size (used for the shade-pass cursor ring + the per-stamp footprint)
/// stays in sync.
#[component]
pub fn PaintToolbar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "position:absolute;top:8px;right:8px;z-index:20;\
                    display:flex;flex-direction:column;gap:4px;\
                    align-items:flex-end;",

            // Row 1 — always visible: Paint on/off.
            {paint_toggle_button(__scope, store, cmd.clone())}

            // Row 2 — mode toggle + color picker.
            if store.paint_active.get() {
                {paint_mode_row(__scope, store)}
            }

            // Row 3 — brush settings (radius, strength, falloff).
            if store.paint_active.get() {
                {paint_settings_panel(__scope, store, cmd.clone())}
            }
        }
    }
}

fn paint_toggle_button(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if store.paint_active.get() {
                    // Active — warm orange so it reads as "mode is on".
                    "display:flex;align-items:center;gap:6px;\
                     padding:4px 10px;border-radius:6px;cursor:pointer;\
                     background:rgba(255,152,0,0.85);color:#1e1e1e;\
                     border:1px solid #ffb74d;font-size:11px;font-weight:600;\
                     backdrop-filter:blur(8px);"
                } else {
                    "display:flex;align-items:center;gap:6px;\
                     padding:4px 10px;border-radius:6px;cursor:pointer;\
                     background:rgba(30,30,30,0.85);color:#bbb;\
                     border:1px solid #3c3c3c;font-size:11px;font-weight:500;\
                     backdrop-filter:blur(8px);"
                }
            }},
            title: {move || {
                if store.paint_active.get() { "Exit paint mode (P)" }
                else { "Enter paint mode (P)" }
            }},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let on = !store.paint_active.get();
                    store.paint_active.set(on);
                    // Tell the engine so it knows whether to draw the
                    // cursor wireframe and at what radius.
                    let _ = cmd.0.send(rkp_engine::EngineCommand::SetPaintActive {
                        active: on,
                        radius: store.paint_radius.get(),
                    });
                }
            },
            span {
                style: "width:16px;height:16px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                {render_tabler_icon(__scope, TablerIcon::Brush, TablerIconStyle::Outline)}
            }
            if store.paint_active.get() { {"Painting"} }
            if !store.paint_active.get() { {"Paint"} }
        }
    }
}

fn paint_mode_row(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    use rkp_engine::PaintMode;
    rsx! {
        div {
            style: "display:flex;flex-direction:row;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;\
                    backdrop-filter:blur(8px);",

            {gizmo_button(
                __scope,
                TablerIcon::Palette,
                "Material — paint the selected material id",
                Memo::new(move || store.paint_mode.get() == PaintMode::Material),
                move || store.paint_mode.set(PaintMode::Material),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::ColorPicker,
                "Color — paint the brush color",
                Memo::new(move || store.paint_mode.get() == PaintMode::Color),
                move || store.paint_mode.set(PaintMode::Color),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::Eraser,
                "Erase — fade the per-voxel color override",
                Memo::new(move || store.paint_mode.get() == PaintMode::Erase),
                move || store.paint_mode.set(PaintMode::Erase),
            )}

            // The color picker lives in the settings panel below
            // (Row 3) — in the mode row it was too narrow to click
            // reliably, and the popup opened past the right edge of
            // the viewport. See `paint_color_row` in the settings
            // panel.
        }
    }
}

fn paint_color_row(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    let initial_hex = untracked(|| rgb_to_hex(store.paint_color.get()));

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:6px;min-height:22px;",
            div {
                style: "width:72px;flex-shrink:0;font-size:11px;color:#999;",
                {"Color"}
            }
            div {
                style: "flex:1;min-width:0;",
                ColorInput {
                    value: {initial_hex},
                    value_fn: move || rgb_to_hex(store.paint_color.get()),
                    format: "hex",
                    alpha: false,
                    onchange: move |v: String| {
                        // Guard against ColorInput echoing its own
                        // committed value — otherwise the onchange
                        // fires every time value_fn updates,
                        // creating a loop. `untracked` keeps the
                        // guard read from subscribing us to the
                        // upstream Signal.
                        untracked(|| {
                            if let Some(rgb) = hex_to_rgb(&v) {
                                let cur = store.paint_color.get();
                                let eps = 1.01 / 255.0;
                                if (rgb[0] - cur[0]).abs() < eps
                                    && (rgb[1] - cur[1]).abs() < eps
                                    && (rgb[2] - cur[2]).abs() < eps
                                {
                                    return;
                                }
                                store.paint_color.set(rgb);
                            }
                        });
                    },
                }
            }
        }
    }
}

fn rgb_to_hex(c: [f32; 3]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn hex_to_rgb(hex: &str) -> Option<[f32; 3]> {
    if hex.len() != 7 || !hex.starts_with('#') {
        return None;
    }
    let r = u8::from_str_radix(&hex[1..3], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&hex[3..5], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&hex[5..7], 16).ok()? as f32 / 255.0;
    Some([r, g, b])
}

fn paint_settings_panel(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    // Radius: 1 cm to 5 m in world units. The lower bound matches
    // typical voxel sizes; much below that and a single stamp
    // doesn't touch any voxel centers.
    let radius_on_change: Rc<dyn Fn(f32)> = {
        let cmd = cmd.clone();
        Rc::new(move |_r: f32| {
            // The Signal has already been written by `prop_slider`'s
            // on_change wrapper — just mirror the new value to the
            // engine so its cached brush size (cursor ring + flood
            // footprint) stays in lock-step with the slider.
            let _ = cmd.0.send(rkp_engine::EngineCommand::SetPaintActive {
                active: true,
                radius: store.paint_radius.get(),
            });
        })
    };
    // Strength + falloff are read per-stamp on the editor side, so no
    // engine round-trip is needed on change — a no-op on_change after
    // prop_slider's built-in Signal write.
    let noop: Rc<dyn Fn(f32)> = Rc::new(|_| {});

    rsx! {
        div {
            // 260 px gives the ColorInput popup enough width to open
            // downward-right without spilling off the viewport when
            // the toolbar is anchored to the right edge.
            style: "width:260px;padding:8px 10px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;\
                    backdrop-filter:blur(8px);\
                    display:flex;flex-direction:column;gap:4px;",
            // Color row only shows in Color mode — material + erase
            // don't consume a color value, so we hide it to reduce
            // noise and free vertical space.
            if store.paint_mode.get() == rkp_engine::PaintMode::Color {
                {paint_color_row(__scope, store)}
            }
            {prop_slider(__scope, "Radius", store.paint_radius, 0.01, 5.0, 0.01, radius_on_change)}
            {prop_slider(__scope, "Strength", store.paint_strength, 0.0, 1.0, 0.01, noop.clone())}
            {prop_slider(__scope, "Falloff", store.paint_falloff, 0.0, 1.0, 0.01, noop)}
        }
    }
}
