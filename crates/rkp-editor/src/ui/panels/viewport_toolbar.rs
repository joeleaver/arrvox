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

// ── Floating brush toolbar ──────────────────────────────────────────────

/// Floating overlay holding the Paint and Sculpt tool toggles + their
/// settings, upper-right of the viewport. Paint and Sculpt are mutually
/// exclusive — turning one on turns the other off. Only the active
/// tool's settings panel is expanded.
#[component]
pub fn BrushToolbar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();
    // rsx!'s `if` branches generate `Fn` closures that capture by move;
    // a single `cmd` can only be moved into one such closure. Pre-clone
    // into one binding per if-branch, then `.clone()` again inside the
    // branch so each re-render gets a fresh `CommandSender`.
    let cmd_paint_settings = cmd.clone();
    let cmd_sculpt_settings = cmd.clone();

    rsx! {
        div {
            style: "position:absolute;top:8px;right:8px;z-index:20;\
                    display:flex;flex-direction:column;gap:4px;\
                    align-items:flex-end;",

            // Row 1 — both tool toggles side by side. Mutually exclusive.
            div {
                style: "display:flex;flex-direction:row;gap:4px;",
                {paint_toggle_button(__scope, store, cmd.clone())}
                {sculpt_toggle_button(__scope, store, cmd.clone())}
            }

            // Paint settings (active tool only).
            if store.paint_active.get() {
                {paint_mode_row(__scope, store)}
            }
            if store.paint_active.get() {
                {paint_settings_panel(__scope, store, cmd_paint_settings.clone())}
            }

            // Sculpt settings (active tool only).
            if store.sculpt_active.get() {
                {sculpt_mode_row(__scope, store)}
            }
            if store.sculpt_active.get() {
                {sculpt_settings_panel(__scope, store, cmd_sculpt_settings.clone())}
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
                    // Mutual exclusion: enabling paint turns sculpt off
                    // and pushes the engine-side off-signal so its
                    // cursor + stamp gates clear too.
                    if on && store.sculpt_active.get() {
                        store.sculpt_active.set(false);
                        let _ = cmd.0.send(rkp_engine::EngineCommand::SetSculptActive {
                            active: false,
                            radius: store.sculpt_radius.get(),
                        });
                    }
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

// ── Sculpt toolbar pieces ───────────────────────────────────────────────

fn sculpt_toggle_button(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: {move || {
                if store.sculpt_active.get() {
                    // Active — teal so it visually distinguishes from
                    // paint's warm orange (mutual-exclusion makes them
                    // never both active, but the color still cues which
                    // tool you're in).
                    "display:flex;align-items:center;gap:6px;\
                     padding:4px 10px;border-radius:6px;cursor:pointer;\
                     background:rgba(38,166,154,0.85);color:#1e1e1e;\
                     border:1px solid #4db6ac;font-size:11px;font-weight:600;\
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
                if store.sculpt_active.get() { "Exit sculpt mode" }
                else { "Enter sculpt mode" }
            }},
            onclick: {
                let cmd = cmd.clone();
                move || {
                    let on = !store.sculpt_active.get();
                    store.sculpt_active.set(on);
                    // Mutual exclusion with paint.
                    if on && store.paint_active.get() {
                        store.paint_active.set(false);
                        let _ = cmd.0.send(rkp_engine::EngineCommand::SetPaintActive {
                            active: false,
                            radius: store.paint_radius.get(),
                        });
                    }
                    let _ = cmd.0.send(rkp_engine::EngineCommand::SetSculptActive {
                        active: on,
                        radius: store.sculpt_radius.get(),
                    });
                }
            },
            span {
                style: "width:16px;height:16px;display:inline-flex;\
                        align-items:center;justify-content:center;",
                {render_tabler_icon(__scope, TablerIcon::Sphere, TablerIconStyle::Outline)}
            }
            if store.sculpt_active.get() { {"Sculpting"} }
            if !store.sculpt_active.get() { {"Sculpt"} }
        }
    }
}

fn sculpt_mode_row(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    use rkp_engine::SculptMode;
    rsx! {
        div {
            style: "display:flex;flex-direction:row;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;\
                    backdrop-filter:blur(8px);",

            // Phase A: Raise disabled — Carve maps to the overlay (drop
            // an existing leaf_attr_id into the per-instance set), but
            // Raise needs new geometry that doesn't exist yet. Phase B
            // will add per-added-leaf proxy splats / cluster re-bake.
            // Until then the button is greyed out + tooltipped to
            // explain.
            div {
                style: "width:28px;height:28px;display:flex;align-items:center;\
                        justify-content:center;border-radius:4px;\
                        color:#555;background:transparent;\
                        cursor:not-allowed;opacity:0.55;",
                title: "Raise — disabled (requires Phase B: new geometry)",
                span {
                    style: "width:18px;height:18px;display:inline-flex;\
                            align-items:center;justify-content:center;",
                    {render_tabler_icon(__scope, TablerIcon::Plus, TablerIconStyle::Outline)}
                }
            }
            {gizmo_button(
                __scope,
                TablerIcon::Minus,
                "Carve — remove geometry (dig) under the brush",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Carve),
                move || store.sculpt_mode.set(SculptMode::Carve),
            )}
        }
    }
}

fn sculpt_settings_panel(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    let radius_on_change: Rc<dyn Fn(f32)> = {
        let cmd = cmd.clone();
        Rc::new(move |_r: f32| {
            // Mirror the new radius to the engine so the cursor ring
            // visualization (when wired) and the next stamp's footprint
            // stay in lock-step with the slider.
            let _ = cmd.0.send(rkp_engine::EngineCommand::SetSculptActive {
                active: true,
                radius: store.sculpt_radius.get(),
            });
        })
    };
    let noop: Rc<dyn Fn(f32)> = Rc::new(|_| {});

    rsx! {
        div {
            style: "width:260px;padding:8px 10px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;\
                    backdrop-filter:blur(8px);\
                    display:flex;flex-direction:column;gap:4px;",
            {prop_slider(__scope, "Radius", store.sculpt_radius, 0.01, 5.0, 0.01, radius_on_change)}
            {prop_slider(__scope, "Falloff", store.sculpt_falloff, 0.0, 1.0, 0.01, noop)}
        }
    }
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
