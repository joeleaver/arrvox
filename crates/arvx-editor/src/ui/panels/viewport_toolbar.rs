//! Viewport toolbars — fixed header bar + floating gizmo overlay.
//!
//! The fixed bar holds viewport-wide controls (play/stop, camera selectors, etc.)
//! that are always visible. The floating overlay holds mode-specific tools
//! (translate/rotate/scale) that only apply during editing.

use std::rc::Rc;

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use arvx_engine::gizmo::GizmoMode;

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
                        let _ = cmd.0.send(arvx_engine::EngineCommand::PlayStop);
                    } else {
                        let _ = cmd.0.send(arvx_engine::EngineCommand::PlayStart);
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
                            let _ = cmd.0.send(arvx_engine::EngineCommand::SetGizmoMode {
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
                            let _ = cmd.0.send(arvx_engine::EngineCommand::SetGizmoMode {
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
                            let _ = cmd.0.send(arvx_engine::EngineCommand::SetGizmoMode {
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
                        let _ = cmd.0.send(arvx_engine::EngineCommand::SetSculptActive {
                            active: false,
                            radius: store.sculpt_radius.get(),
                        });
                    }
                    // Tell the engine so it knows whether to draw the
                    // cursor wireframe and at what radius.
                    let _ = cmd.0.send(arvx_engine::EngineCommand::SetPaintActive {
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
    use arvx_engine::PaintMode;
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
                        let _ = cmd.0.send(arvx_engine::EngineCommand::SetPaintActive {
                            active: false,
                            radius: store.paint_radius.get(),
                        });
                    }
                    let _ = cmd.0.send(arvx_engine::EngineCommand::SetSculptActive {
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
    use arvx_engine::SculptMode;
    rsx! {
        div {
            style: "display:flex;flex-direction:row;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;\
                    backdrop-filter:blur(8px);",

            // Phase B R2/R4-minimal: Raise + Carve both go through
            // real-geometry mutation (apply_delta) + per-stamp full
            // mesh re-extract. Drag stamps still stutter at asset
            // size — R4-proper (per-cluster re-extract) is the perf
            // path.
            {gizmo_button(
                __scope,
                TablerIcon::Plus,
                "Raise — add geometry (hard SDF union) under the brush",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Raise),
                move || store.sculpt_mode.set(SculptMode::Raise),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::Minus,
                "Carve — remove geometry (hard SDF subtract) under the brush",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Carve),
                move || store.sculpt_mode.set(SculptMode::Carve),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::Mountain,
                "Inflate — soft outward dilation along the existing surface (Blender Draw / Inflate)",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Inflate),
                move || store.sculpt_mode.set(SculptMode::Inflate),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::TrendingDown,
                "Deflate — soft inward erosion (the 'soft Carve' most sculpt programs default to)",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Deflate),
                move || store.sculpt_mode.set(SculptMode::Deflate),
            )}
            {gizmo_button(
                __scope,
                TablerIcon::Blur,
                "Smooth — blend surface normals toward the local 6-neighbour average (Blender Smooth equivalent)",
                Memo::new(move || store.sculpt_mode.get() == SculptMode::Smooth),
                move || store.sculpt_mode.set(SculptMode::Smooth),
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
            let _ = cmd.0.send(arvx_engine::EngineCommand::SetSculptActive {
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
            {prop_slider(__scope, "Falloff", store.sculpt_falloff, 0.0, 1.0, 0.01, noop.clone())}
            // Max-thickness amplitude in finest-voxel units. Only
            // Inflate / Deflate consume this; Carve / Raise ignore it.
            // Range 1..32 voxels covers the typical sculpting feel —
            // 1 voxel = barely-perceptible offset, 32 voxels = deep
            // pit / tall ridge in one stamp.
            {prop_slider(__scope, "Strength", store.sculpt_strength, 1.0, 32.0, 1.0, noop)}
        }
    }
}

// ── Phase 9b: Terrain viewport toolbar ─────────────────────────────────

/// Floating overlay at the top-centre of the viewport, shown when
/// the selected entity is a Terrain. Contains:
///
///   Row 1: Sculpt · Paint · Heatmap
///   Row 2: Region · Revert · Bake · + Stamp ▼
///
/// Sculpt / Paint reuse the same signals as [`BrushToolbar`] — the
/// two toolbars cross-reference the same mode state. Heatmap toggles
/// the overlay visibility signal (the overlay pass itself is a
/// separate Phase 9b item). Region toggles a "next drag = region
/// drag-box" arming state in the viewport. Revert / Bake fire the
/// corresponding `EngineCommand`s against the active region — or the
/// camera-radius fallback when none is set.
#[component]
pub fn TerrainToolbar() -> NodeHandle {
    let store = use_context::<EditorStore>();
    let cmd = use_context::<CommandSender>();

    // Visibility — checks the inspector snapshot for a Terrain
    // component. The inspector is None when nothing is selected; an
    // empty snapshot or one without the Terrain component hides the
    // toolbar.
    let visible = Memo::new(move || {
        store
            .inspector
            .get()
            .map(|snap| snap.components.iter().any(|c| c.name == "Terrain"))
            .unwrap_or(false)
    });

    let stamp_menu_open = Signal::new(false);
    // rsx!'s `if` branches each move-capture `cmd` independently —
    // pre-clone one binding per branch so the closures don't double-
    // move the same value. Same pattern as `BrushToolbar` (where the
    // sub-pane closures clone before consuming).
    let cmd_row1 = cmd.clone();
    let cmd_row2 = cmd.clone();
    let cmd_stamp = cmd.clone();
    let cmd_badge = cmd;

    rsx! {
        div {
            // Always-mounted div so the visibility toggle is a cheap
            // style swap rather than a tree mount/unmount on every
            // selection change.
            style: {move || {
                if visible.get() {
                    "position:absolute;top:8px;left:50%;transform:translateX(-50%);\
                     z-index:20;display:flex;flex-direction:column;gap:4px;\
                     align-items:center;"
                } else {
                    "display:none;"
                }
            }},

            // Row 1 — Sculpt · Paint · Heatmap.
            {terrain_toolbar_row1(__scope, store, cmd_row1)}
            // Row 2 — Region · Revert · Bake · + Stamp ▼.
            {terrain_toolbar_row2(__scope, store, cmd_row2, stamp_menu_open)}

            // Stamp submenu (anchored under row 2 when open).
            if stamp_menu_open.get() {
                {terrain_stamp_menu(__scope, cmd_stamp.clone(), stamp_menu_open)}
            }

            // Active-region badge — only when a region is set. Gives
            // the author a clear "Revert / Bake will scope to this
            // region" cue plus a clear-region affordance.
            if store.active_terrain_region.get().is_some() {
                {terrain_active_region_badge(__scope, store, cmd_badge.clone())}
            }

            // Heatmap counter — shown only when the overlay is on.
            // Live count of divergent tiles so the author can tell
            // at a glance how much editing's happened.
            if store.terrain_heatmap_visible.get() {
                {terrain_heatmap_counter(__scope, store)}
            }
        }
    }
}

fn terrain_toolbar_row1(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    let cmd_sculpt = cmd.clone();
    let cmd_paint = cmd.clone();
    let cmd_heatmap = cmd;
    rsx! {
        div {
            style: "display:flex;flex-direction:row;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;\
                    backdrop-filter:blur(8px);",

            // Sculpt — same signal + mutual-exclusion as BrushToolbar's
            // sculpt button. Re-clicking here turns sculpt off.
            {terrain_mode_button(
                __scope,
                TablerIcon::Sphere,
                "Sculpt — sculpt terrain (S)",
                Memo::new(move || store.sculpt_active.get()),
                {
                    let cmd = cmd_sculpt;
                    move || {
                        let on = !store.sculpt_active.get();
                        store.sculpt_active.set(on);
                        if on && store.paint_active.get() {
                            store.paint_active.set(false);
                            let _ = cmd.0.send(arvx_engine::EngineCommand::SetPaintActive {
                                active: false,
                                radius: store.paint_radius.get(),
                            });
                        }
                        let _ = cmd.0.send(arvx_engine::EngineCommand::SetSculptActive {
                            active: on,
                            radius: store.sculpt_radius.get(),
                        });
                    }
                },
            )}

            // Paint — mirrors sculpt above.
            {terrain_mode_button(
                __scope,
                TablerIcon::Brush,
                "Paint — paint terrain materials / color (P)",
                Memo::new(move || store.paint_active.get()),
                {
                    let cmd = cmd_paint;
                    move || {
                        let on = !store.paint_active.get();
                        store.paint_active.set(on);
                        if on && store.sculpt_active.get() {
                            store.sculpt_active.set(false);
                            let _ = cmd.0.send(arvx_engine::EngineCommand::SetSculptActive {
                                active: false,
                                radius: store.sculpt_radius.get(),
                            });
                        }
                        let _ = cmd.0.send(arvx_engine::EngineCommand::SetPaintActive {
                            active: on,
                            radius: store.paint_radius.get(),
                        });
                    }
                },
            )}

            // Heatmap — outlines every tile divergent from the
            // procedural baseline (sculpt edits this session or a
            // saved `.arvxtile` from a prior session). Wireframes
            // render in the engine's gizmo overlay; the toolbar just
            // gates visibility.
            {terrain_mode_button(
                __scope,
                TablerIcon::Flame,
                "Heatmap — outline tiles diverged from the procedural baseline",
                Memo::new(move || store.terrain_heatmap_visible.get()),
                {
                    let cmd = cmd_heatmap;
                    move || {
                        let on = !store.terrain_heatmap_visible.get();
                        store.terrain_heatmap_visible.set(on);
                        let _ = cmd.0.send(
                            arvx_engine::EngineCommand::SetTerrainHeatmapVisible {
                                visible: on,
                            },
                        );
                    }
                },
            )}
        }
    }
}

fn terrain_toolbar_row2(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
    stamp_menu_open: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    let cmd_region = cmd.clone();
    let cmd_revert = cmd.clone();
    let cmd_bake = cmd.clone();
    rsx! {
        div {
            style: "display:flex;flex-direction:row;gap:1px;\
                    background:rgba(30,30,30,0.85);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;\
                    backdrop-filter:blur(8px);",

            // Region — toggles drag-box arm state. The viewport's
            // mouse handler reads `terrain_region_drag_armed` to
            // decide whether LMB-down begins a drag-box rather than
            // a normal click-select pick.
            {terrain_mode_button(
                __scope,
                TablerIcon::Rectangle,
                "Region — drag-box to set the active region for Revert / Bake",
                Memo::new(move || store.terrain_region_drag_armed.get()),
                {
                    let cmd = cmd_region;
                    move || {
                        let on = !store.terrain_region_drag_armed.get();
                        store.terrain_region_drag_armed.set(on);
                        // Toggling armed off also clears any
                        // in-progress drag rect, so a stale half-drag
                        // doesn't reappear if the author re-arms.
                        if !on {
                            store.terrain_region_drag_rect.set(None);
                        }
                        // Disarming with no active region is also a
                        // good moment to drop the engine's region —
                        // matches the "click again to leave region
                        // mode" mental model.
                        if !on && store.active_terrain_region.get().is_some() {
                            let _ = cmd.0.send(arvx_engine::EngineCommand::ClearTerrainRegion);
                        }
                    }
                },
            )}

            // Revert — fires against the active region (if set) or
            // the camera-radius AABB. The engine warns to console if
            // no Terrain is in the scene.
            {terrain_action_button(
                __scope,
                TablerIcon::Restore,
                "Revert — drop sculpt edits in the active region (or camera radius)",
                {
                    let cmd = cmd_revert;
                    move || {
                        let region = store.active_terrain_region.get();
                        let radius = store.terrain_camera_radius_m.get();
                        if let Some(aabb) = region {
                            let _ = cmd.0.send(
                                arvx_engine::EngineCommand::RevertTerrainInAabb { aabb },
                            );
                        } else {
                            let _ = cmd.0.send(
                                arvx_engine::EngineCommand::RevertTerrainAtCameraRadius { radius },
                            );
                        }
                    }
                },
            )}

            // Bake — persists live tiles in the active region (or
            // camera radius) as `.arvxtile` files next to the scene.
            {terrain_action_button(
                __scope,
                TablerIcon::DeviceFloppy,
                "Bake — persist tiles in the active region (or camera radius) to .arvxtile",
                {
                    let cmd = cmd_bake;
                    move || {
                        let region = store.active_terrain_region.get();
                        let radius = store.terrain_camera_radius_m.get();
                        if let Some(aabb) = region {
                            let _ = cmd.0.send(
                                arvx_engine::EngineCommand::BakeTerrainSnapshotInAabb { aabb },
                            );
                        } else {
                            let _ = cmd.0.send(
                                arvx_engine::EngineCommand::BakeTerrainSnapshotAtCameraRadius {
                                    radius,
                                },
                            );
                        }
                    }
                },
            )}

            // + Stamp — toggle dropdown menu of stamp kinds.
            {terrain_action_button(
                __scope,
                TablerIcon::Mountain,
                "Add Stamp — Mountain / Hill / Lake / Plateau / Flatten",
                move || {
                    stamp_menu_open.set(!stamp_menu_open.get());
                },
            )}
        }
    }
}

fn terrain_stamp_menu(
    __scope: &mut rinch::core::dom::RenderScope,
    cmd: CommandSender,
    stamp_menu_open: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    use arvx_engine::StampKindSpec;
    let entries: [(StampKindSpec, &str); 5] = [
        (StampKindSpec::Mountain, "Mountain"),
        (StampKindSpec::Hill, "Hill"),
        (StampKindSpec::Lake, "Lake"),
        (StampKindSpec::Plateau, "Plateau"),
        (StampKindSpec::Flatten, "Flatten"),
    ];
    rsx! {
        div {
            style: "display:flex;flex-direction:column;gap:1px;\
                    background:rgba(30,30,30,0.92);border-radius:6px;\
                    border:1px solid #3c3c3c;padding:2px;min-width:140px;\
                    backdrop-filter:blur(8px);",
            for (spec, label) in entries {
                {terrain_stamp_menu_item(__scope, cmd.clone(), spec, label, stamp_menu_open)}
            }
        }
    }
}

fn terrain_stamp_menu_item(
    __scope: &mut rinch::core::dom::RenderScope,
    cmd: CommandSender,
    spec: arvx_engine::StampKindSpec,
    label: &str,
    stamp_menu_open: Signal<bool>,
) -> rinch::core::dom::NodeHandle {
    let label_str = label.to_string();
    rsx! {
        div {
            style: "padding:4px 10px;border-radius:4px;cursor:pointer;\
                    color:#ddd;font-size:11px;",
            onclick: move || {
                let _ = cmd.0.send(arvx_engine::EngineCommand::SpawnStamp { kind: spec });
                stamp_menu_open.set(false);
            },
            {label_str}
        }
    }
}

fn terrain_active_region_badge(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
    cmd: CommandSender,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: "display:flex;flex-direction:row;align-items:center;gap:6px;\
                    padding:3px 8px;border-radius:4px;\
                    background:rgba(255,193,7,0.18);border:1px solid #ffc107;\
                    color:#ffe082;font-size:10px;\
                    backdrop-filter:blur(8px);",
            span {
                {move || {
                    match store.active_terrain_region.get() {
                        Some(a) => format!(
                            "Region {:.0}×{:.0} m",
                            a.max.x - a.min.x,
                            a.max.z - a.min.z,
                        ),
                        None => String::new(),
                    }
                }}
            }
            div {
                style: "padding:0 4px;cursor:pointer;color:#ffe082;font-weight:600;",
                title: "Clear active region",
                onclick: move || {
                    let _ = cmd.0.send(arvx_engine::EngineCommand::ClearTerrainRegion);
                },
                {"×"}
            }
        }
    }
}

fn terrain_heatmap_counter(
    __scope: &mut rinch::core::dom::RenderScope,
    store: EditorStore,
) -> rinch::core::dom::NodeHandle {
    rsx! {
        div {
            style: "display:flex;align-items:center;\
                    padding:3px 8px;border-radius:4px;\
                    background:rgba(255,115,20,0.18);border:1px solid #ff7314;\
                    color:#ffcd91;font-size:10px;\
                    backdrop-filter:blur(8px);",
            span {
                {move || {
                    let n = store.terrain_divergent_tile_count.get();
                    if n == 1 {
                        "1 tile edited".to_string()
                    } else {
                        format!("{n} tiles edited")
                    }
                }}
            }
        }
    }
}

/// Smaller variant of [`gizmo_button`] tuned for the Terrain
/// toolbar's pill: text label below the icon (or just icon) and a
/// distinct active colour (terrain-orange to echo the bounds
/// wireframe).
fn terrain_mode_button(
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
                    "width:30px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     background:#ff9800;color:#1e1e1e;"
                } else {
                    "width:30px;height:28px;display:flex;align-items:center;\
                     justify-content:center;border-radius:4px;cursor:pointer;\
                     color:#bbb;background:transparent;"
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

/// Push-button variant (no active state) for one-shot actions like
/// Revert / Bake / Add Stamp.
fn terrain_action_button(
    __scope: &mut rinch::core::dom::RenderScope,
    icon: TablerIcon,
    tooltip: &str,
    on_click: impl Fn() + Clone + 'static,
) -> rinch::core::dom::NodeHandle {
    let tooltip = tooltip.to_string();
    rsx! {
        div {
            style: "width:30px;height:28px;display:flex;align-items:center;\
                    justify-content:center;border-radius:4px;cursor:pointer;\
                    color:#bbb;background:transparent;",
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
            let _ = cmd.0.send(arvx_engine::EngineCommand::SetPaintActive {
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
            if store.paint_mode.get() == arvx_engine::PaintMode::Color {
                {paint_color_row(__scope, store)}
            }
            {prop_slider(__scope, "Radius", store.paint_radius, 0.01, 5.0, 0.01, radius_on_change)}
            {prop_slider(__scope, "Strength", store.paint_strength, 0.0, 1.0, 0.01, noop.clone())}
            {prop_slider(__scope, "Falloff", store.paint_falloff, 0.0, 1.0, 0.01, noop)}
        }
    }
}
